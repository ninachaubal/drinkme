"""Chat templating: OpenAI messages -> prompt token ids.

A thin adapter over tokenizer.apply_chat_template — the template itself is
the model family's, shipped with the tokenizer, and we do not re-implement
it. Three accommodations the raw call lacks:

  1. Families with no system role (gemma) RAISE on {"role": "system"}. The
     community convention is to fold the system text into the first user
     turn; we do that on failure and retry, rather than sniffing model names
     — the template raising IS the detection.
  2. Template kwargs pass through untouched (Qwen3's enable_thinking=False,
     reasoning_effort and preserve_thinking live here), so serve can expose
     thinking-mode control without this file knowing which families have one.
     Two layers, merged by effective_kwargs below: the ENGINE's own (a
     server-wide default, generation_config.json) and the REQUEST's
     (GenerationRequest.template_kwargs), which wins.
  3. The rendered prompt is READ BACK for one fact the serve loop cannot
     recover later: did it end inside an open `<think>` block? Qwen3's
     template writes `<think>\\n` at the end of the generation prompt, so the
     model's output starts in reasoning and only the CLOSING tag ever reaches
     the stream — and serving/think.py has to know which channel byte one
     belongs to. render_prompt RETURNS the verdict beside the ids
     (Prompt.opens_think); the engine hands it on as its StreamStart event.
     Nothing here is ambient: no thread-local, no context variable
     — what a caller gets is a function of what it
     passed.

Unit-tested against a fake tokenizer; no download, no network.
"""

from __future__ import annotations

from dataclasses import dataclass

from .think import OPENS as THINK_OPENS

# apply_chat_template arguments this module supplies itself: a request's
# chat_template_kwargs entry by one of these names would arrive as a duplicate
# argument, not a template variable, so the HTTP dialects refuse them by name
# at the boundary (a 400, not a TypeError deep in the engine). `conversation`
# is the messages parameter's real name upstream.
RESERVED_KWARGS = frozenset({"conversation", "messages", "add_generation_prompt",
                             "tokenize", "tools"})

@dataclass(frozen=True)
class Prompt:
    """One rendered prompt: the ids, and the read-back (accommodation 3)."""

    ids: list[int]
    opens_think: bool


def effective_kwargs(engine_kwargs: dict | None, request_kwargs: dict | None, *,
                     constrained: bool = False) -> dict:
    """The kwargs one render actually gets: the engine's own defaults
    (generation_config.json) under the request's (GenerationRequest.
    template_kwargs) — the request wins — and, for a grammar-constrained
    generation, enable_thinking=False unless either layer said otherwise:
    the constraint bans a `<think>` preamble anyway ("<" is no JSON prefix),
    so leaving thinking on just starves a reasoning model into whitespace
    (live smoke). Families without the kwarg ignore it. ONE
    function so generate() and count_tokens() can never render two
    different prompts for one request. None/{} on both sides leaves the
    apply_chat_template call byte-identical to the default."""
    kw = dict(engine_kwargs or {})
    kw.update(request_kwargs or {})
    if constrained:
        kw.setdefault("enable_thinking", False)
    return kw


def _ends_open_think(tokenizer, ids: list[int]) -> bool:
    """The read-back (accommodation 3). Only the tail can carry the tag, so
    only the tail is decoded — microseconds against a prefill. Asked of the
    PROMPT rather than guessed from the model name or the kwargs: the template
    is the only thing that knows whether it opened a block, and it just
    rendered its answer. Never raises: an unreadable tail means "no", which is
    exactly the default behavior. FakeEngine, which renders nothing, reports
    its `opens_think` flag in the same StreamStart slot.

    Any row's opening tag counts (think.OPENS): Qwen's `<think>`, and gemma's
    `<|channel>thought` — which its template writes at the end of the
    generation prompt after a tool response with thinking on. Decoded with
    the specials kept, so a control-token tag is readable here even though
    the serve loop's decode would strip it."""
    try:
        # 32 ids: gemma's `<|channel>thought\n` is 18 characters, and the
        # one-id-per-character fakes the suite uses must see all of it
        tail = tokenizer.decode(ids[-32:], skip_special_tokens=False)
    except Exception:  # noqa: BLE001 — a tokenizer that cannot decode ids of
        return False   # its own making is not a reason to fail the request
    return isinstance(tail, str) and tail.rstrip().endswith(THINK_OPENS)


def fold_system(messages: list[dict]) -> list[dict]:
    """Fold system content into the first user turn: "<system>\\n\\n<user>".
    Returns the SAME list object when there is nothing to fold, so a caller
    can tell "folded" from "unchanged" by identity."""
    sys_text = "\n\n".join(m.get("content", "") for m in messages if m.get("role") == "system")
    if not sys_text:
        return messages
    rest = [dict(m) for m in messages if m.get("role") != "system"]
    for m in rest:
        if m.get("role") == "user":
            m["content"] = f"{sys_text}\n\n{m.get('content', '')}"
            return rest
    # system with no user turn at all: the system text becomes the user turn
    return [{"role": "user", "content": sys_text}] + rest


def build_prompt(tokenizer, messages: list[dict], add_generation_prompt: bool = True,
                 template_kwargs: dict | None = None,
                 tools: list | None = None) -> list[int]:
    """messages -> token ids: render_prompt's ids, for a caller that only
    wants the count or the ids (the tokenizer routes)."""
    return render_prompt(tokenizer, messages, add_generation_prompt,
                         template_kwargs, tools).ids


def render_prompt(tokenizer, messages: list[dict], add_generation_prompt: bool = True,
                  template_kwargs: dict | None = None,
                  tools: list | None = None) -> Prompt:
    """messages -> Prompt(ids, opens_think) via the tokenizer's own chat
    template. `template_kwargs` are passed through as given — the caller
    has already merged its layers (effective_kwargs).

    On a template error with a system message present, folds and retries
    (accommodation 1 above). Any other failure propagates — a broken
    template must not degrade into a silently different prompt.

    `tools` (OpenAI tool definitions) passes through to apply_chat_template
    only when non-empty — the template renders them (Qwen3 does natively);
    absence must leave the call byte-identical to a tool-less prompt."""
    kw = dict(template_kwargs or {})
    if tools:
        kw["tools"] = tools
    try:
        out = tokenizer.apply_chat_template(
            messages, add_generation_prompt=add_generation_prompt, tokenize=True, **kw)
    except Exception:
        folded = fold_system(messages)
        if folded is messages:  # nothing to fold -> the failure is real
            raise
        out = tokenizer.apply_chat_template(
            folded, add_generation_prompt=add_generation_prompt, tokenize=True, **kw)
    # transformers 5 hands back a BatchEncoding from tokenize=True; this
    # function's declared contract is the ids, so unwrap rather than making
    # every caller know the transformers version.
    ids = out if isinstance(out, list) else out["input_ids"]
    # accommodation 3: whether THIS prompt left a think block open, for
    # whoever has to route the tokens that answer it (serving/think.py)
    return Prompt(ids, _ends_open_think(tokenizer, ids))
