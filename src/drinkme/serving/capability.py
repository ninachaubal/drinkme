"""The off-menu capability announcement: a static, PURE probe over a
model's chat template — no torch, no weights, no generation — answering two
questions serving otherwise only learns by finding out live: will this model
open a reasoning channel, and which tool-call dialect (if any) will it speak
back?

Runs ONCE at engine construction (engines.HFEngine.__init__), cached on the
engine as `.capability`, printed as one `[drinkme] capabilities:` line, and
announced per model on /v1/models (`drinkme.capabilities.thinking` /
`.toolFormat`). The tested
menu tier is not exempt — Qwen3.8-27B and Qwen3-8B both go through this same
probe, and their answers are pinned by test against the real cached
tokenizers (tests/test_serving_capability.py), not asserted from memory.

THINKING: rendered at the server's own default template kwargs (no
overrides — what a request with no chat_template_kwargs gets), through the
same template.render_prompt (ids + the open-think read-back) the engine
uses for real requests, so the classification cannot drift from what
generate() actually sees.

  "open"    the DEFAULT generation prompt ends inside an open <think> block
            (Qwen3.8-27B/qwen3_5: the template force-opens it and only the
            close tag ever reaches the wire — think.py's module docstring).
  "closed"  the template recognizes enable_thinking — EITHER explicit value
            changes the render from the default — but the DEFAULT prompt
            does not force it open. False is checked first; True is only
            rendered when False alone was a no-op, so a template that
            answers False conclusively never risks a raise on True
            downgrading "closed" to "none" (a real fake, not hypothetical —
            tests/test_serving_capability.py). gemma-4 defaults thinking
            OFF, so False alone changes nothing and would read "none";
            True writes a `<|think|>` open tag the default render lacks,
            which is the switch this probe exists to report.
            MEASURED FINDING (real Qwen3-8B tokenizer, revision
            b968826d9c46 — the pinned suite revision, same one suggest.py
            pins): unlike the 27B, Qwen3-8B's cached chat_template does NOT
            write "<think>\n" at the end of the generation prompt at the
            default OR an explicit enable_thinking=True; only
            enable_thinking=False changes anything (it pre-fills the closed
            empty pair). The model still thinks by self-initiative — it
            opens the tag itself, mid-stream, which think.py's splitter
            already handles with no flag needed (the SEEK-state literal-tag
            path) — but that is a GENERATION-TIME fact this template-only
            probe cannot see without running the model, so it reports
            "closed" rather than guessing "open" from what the Qwen3 family
            is known to do elsewhere. What the 8B shares with the 27B's
            prompt-forced open is an OUTCOME (the opening tag never reaches the client as visible
            content), reached by two different mechanisms, not an identical
            render.
  "always"  the model reasons in every reply and no request field turns it
            off. Read off the tool-format row below, not the render: a row
            whose turn is a run of ADDRESSED messages, one of them to the
            model's own reasoning channel (tool_formats: message_open and
            think_channel both set; Muse-Glimmer's `atem`, `to=self`), and a
            template with no enable_thinking. Glimmer's template has a level
            (`reasoning_strength`, default high, or a "Reasoning strength:"
            line in the system turn) and no off; the 30B writes a `to=self`
            message at every level, and the server returns it as reasoning.
  "none"    enable_thinking has no effect on the render at all — the
            template has no thinking control surface (e.g. a family with no
            reasoning mode).

TOOL_FORMAT: rendered once without a tools list and once with one probe
tool, through the same template.build_prompt(tools=...) real requests use,
and the tooled render is matched against THE TABLE — serving/tool_formats.py,
one row per output dialect, ordered, first signature match wins. The
announced value is the ROW NAME, so /v1/models says exactly which parser the
server will run:

  a row name  `json` (Llama/Qwen/DeepSeek `<tool_call>{json}</tool_call>`),
              `qwen-xml` (Qwen3.5+ `<function=...>`), `gemma`
              (`<|tool_call>call:...`), `glm` (`<arg_key>`/`<arg_value>`),
              `minimax` (`<minimax:tool_call>`), `mistral` (`[TOOL_CALLS]`),
              `kimi-k2` (`<|tool_calls_section_begin|>`) — read
              tool_formats.ROWS for the live list, never this comment.
  "unknown"   tools change the render (the template DOES support tool
              definitions) but NO row's signature matches — the model may
              emit calls in a shape this server cannot parse back into
              structured tool_calls.
  "none"      tools have no effect on the render — the template has no
              tool-calling support at all.

THE WRITTEN-BACK CALL. Some templates declare tools without teaching
a call format: MiMo-V2.6-Distill-Qwen-9B's renders only "You are provided
with the following tools: <tools>…</tools>", which matches no row. The same
template writes an assistant `tool_calls` history turn back as
`<tool_call><function=…><parameter=…>`, which is qwen-xml: the dialect it
was trained to emit is in how it replays its own calls. So a second render,
a short history of one assistant tool call and its result (_PROBE_HISTORY,
the shape serving/messages.py hands the template), is matched against the
same table, and the two answers combine:

  the first names a row, the history names none or the same   -> the first
  the first says "unknown", the history names a row            -> the history
  the two name DIFFERENT rows                                  -> "unknown"
  the first says "none"                                        -> "none"

Disagreement is "unknown" because each answer is evidence about one parser,
and a server that picks the wrong one leaks dialect text; "unknown" refuses,
which is safe. A history render that raises counts as naming nothing.

THE REFUSAL (http.py, all three dialects): a tools request against a model whose
tool_format is "unknown" or "none" is refused — 400 (OpenAI) /
invalid_request_error (Anthropic) — naming the model, the reason, and the
rows that DO exist, before a token is generated. Refuse loudly beats a
dialect we cannot parse leaking into the chat as visible text.
`tool_choice: "none"` withholds tools before this check ever runs (nothing
was going to be offered to the model anyway), exactly as it does by default.

Adding a family therefore moves the refusal line by exactly one row, with no
edit here: the probe classifies it, http.py stops refusing it, and the
scanner walks it.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from . import tool_formats
from . import vision
from .template import render_prompt

# One user turn: the plainest prompt every family's template accepts (no
# system role, which is the one thing known to make some templates raise —
# template.py's fold_system accommodation exists for exactly that case, and
# this probe has no reason to exercise it).
_PROBE_MESSAGES = [{"role": "user", "content": "hi"}]

# One minimal tool definition — shape only, never sent to a real model:
# enough for a template that renders tool defs to produce its instruction
# block, nothing in the wording that could collide with a dialect signature.
_PROBE_TOOLS = [{"type": "function", "function": {
    "name": "drinkme_probe", "description": "capability probe, never called",
    "parameters": {"type": "object", "properties": {"x": {"type": "string"}}}}}]

# One assistant turn that called the probe tool, and the tool's answer: the
# history the second tool-format probe renders (module docstring). Arguments
# as a mapping, which is how serving/messages.py hands a history call to the
# template (templates iterate it with |items).
_PROBE_HISTORY = [
    {"role": "user", "content": "hi"},
    {"role": "assistant", "content": "", "tool_calls": [
        {"type": "function", "id": "call_0",
         "function": {"name": "drinkme_probe", "arguments": {"x": "y"}}}]},
    {"role": "tool", "tool_call_id": "call_0", "content": "ok"},
]

# tool_format values this server SERVES tools for: the TESTED rows of
# serving/tool_formats.py's table (menu models stay the tested tier),
# never a second list to keep in sync. A parseable-but-untested row is still
# announced by name on /v1/models; a tools request against it is refused.
PARSEABLE_TOOL_FORMATS = tool_formats.TESTED


def served_tool_formats() -> frozenset:
    """The rows a `tools` request may reach. The TESTED tier, by default and
    for every stranger. `DRINKME_TOOLS_UNTESTED=1` widens it to every row that
    has a parser — the door `bench/serve_dialect_smoke.py` walks through to
    TEST a row on real weights (the gate that earns a row its `tested` flag
    cannot run against a server that refuses the row). An operator opt-in
    announced at serve start, never a default: the refusal exists because an
    unexercised parser leaking dialect text into `content` is worse than a
    400."""
    if os.environ.get("DRINKME_TOOLS_UNTESTED", "").lower() in ("1", "true", "yes"):
        return frozenset(r.name for r in tool_formats.ROWS)
    return tool_formats.TESTED


# The request-level field that flips `thinking` "open"/"closed" off, for a
# client deciding whether to bother sending it: both
# real templates classified "open" (Qwen3.8-27B/qwen3_5, which force-opens
# the DEFAULT render) and "closed" (Qwen3-8B/Qwen3-0.6B, which do not) gate
# their `<think>` block on the literal same Jinja test, `enable_thinking is
# false` — read off the cached templates directly, not inferred from the
# probe's own classification. "none" has no such test anywhere in the
# template, and "always" has no off, so there is nothing to name.
THINKING_SWITCH = "chat_template_kwargs.enable_thinking"


@dataclass(frozen=True)
class Capability:
    thinking: str      # "open" | "closed" | "always" | "none"
    tool_format: str   # a tool_formats row name | "unknown" | "none"

    def as_dict(self) -> dict:
        """The `/v1/models` `drinkme.capabilities` object, camelCase like the
        rest of `drinkme` (docs/serve.md#get-v1models)."""
        switch = THINKING_SWITCH if self.thinking in ("open", "closed") else None
        return {"thinking": self.thinking, "toolFormat": self.tool_format,
                "thinkingSwitch": switch}


def _render(tokenizer, tools=None, messages=None, **template_kwargs) -> str:
    return _prompt(tokenizer, tools, messages, **template_kwargs)[0]


def _prompt(tokenizer, tools=None, messages=None, **template_kwargs) -> tuple[str, bool]:
    """(the rendered text, did it end inside an open think block)."""
    p = render_prompt(tokenizer, messages or _PROBE_MESSAGES,
                      template_kwargs=template_kwargs or None, tools=tools)
    return tokenizer.decode(p.ids, skip_special_tokens=False), p.opens_think


def _thinking(tokenizer) -> str:
    default, opens = _prompt(tokenizer)
    if opens:  # the read-back render_prompt just returned
        return "open"
    closed = _render(tokenizer, enable_thinking=False)
    if closed != default:
        return "closed"
    try:
        opened = _render(tokenizer, enable_thinking=True)
    except Exception:  # noqa: BLE001 — False already proved nothing; a
        # template that raises on True has shown no evidence of a toggle
        return "none"
    return "closed" if opened != default else "none"


def _reasons_always(tool_format: str) -> bool:
    """The row declares addressed messages with a reasoning channel (the
    module docstring's "always")."""
    if tool_format not in tool_formats.NAMES:
        return False
    r = tool_formats.row(tool_format)
    return r.message_open is not None and r.think_channel is not None


def _history_format(tokenizer) -> str | None:
    """The row the template's written-back tool call matches, or None. The
    history is rendered with the probe tool declared too, as a real
    follow-up request would be; only the part the history adds is matched,
    so the declaration block cannot answer for the call."""
    try:
        declared = _render(tokenizer, tools=_PROBE_TOOLS)
        history = _render(tokenizer, tools=_PROBE_TOOLS, messages=_PROBE_HISTORY)
    except Exception:  # noqa: BLE001 — a history this template cannot render names nothing
        return None
    head = 0
    while head < min(len(declared), len(history)) and declared[head] == history[head]:
        head += 1
    return tool_formats.identify(history[head:])


def _tool_format(tokenizer) -> str:
    """The tools-rendered prompt, matched against the TABLE — the row names
    are the announced values, and the classification lives in
    serving/tool_formats.py's `signature` column, not here — then checked
    against the written-back call (the module docstring's rule)."""
    plain = _render(tokenizer)
    tooled = _render(tokenizer, tools=_PROBE_TOOLS)
    if tooled == plain:
        return "none"
    first = tool_formats.identify(tooled)
    history = _history_format(tokenizer)
    if first is None:
        return history or "unknown"
    if history is not None and history != first:
        return "unknown"
    return first


def probe(tokenizer) -> Capability:
    """Run once, at engine construction. Never raises: a template that
    cannot even render the probe prompt has nothing to announce, and a
    probe that could take down `drinkme serve` is worse than one that
    under-reports — the refusal gate stays SAFE either way, since both
    "unknown" and "none" refuse a tools request."""
    try:
        thinking = _thinking(tokenizer)
    except Exception:  # noqa: BLE001 — containment is the point
        thinking = "none"
    try:
        tool_format = _tool_format(tokenizer)
    except Exception:  # noqa: BLE001 — containment is the point
        tool_format = "unknown"
    if thinking == "none" and _reasons_always(tool_format):
        thinking = "always"
    return Capability(thinking, tool_format)


def refusal_message(model_id: str, tool_format: str) -> str:
    """The legible refusal body (every dialect renders it verbatim): names
    the model, says why, and how to proceed."""
    if tool_format == "none":
        why = "its chat template has no tool-calling support"
    elif tool_format in tool_formats.NAMES and tool_format not in tool_formats.TESTED:
        why = (f"its tool-call dialect ({tool_format!r}) is parseable but untested "
               "here — no menu model has exercised it end to end on a GPU, and an "
               "untested parser leaking dialect text into 'content' is the failure "
               "this refusal exists to prevent")
    else:
        why = ("its tool-call output dialect is not one this server can "
               "parse back into structured calls (tool_format: 'unknown')")
    tested = ", ".join(r.name for r in tool_formats.ROWS if r.tested)
    untested = ", ".join(r.name for r in tool_formats.ROWS if not r.tested)
    return (f"model {model_id!r} cannot serve this tools request — {why}. "
            f"Dialects served (tested): {tested}. Parseable but untested: {untested}. "
            "Retry without 'tools', or use one of the tested menu models "
            "(Qwen3.8-27B, Qwen3-8B) — the menu is the tested tier.")


# ------------------------------------------------------- vision capability --
#
# Vision is a structural fact about the served tree (does it carry a ViT and
# a preprocessor for this checkpoint?), not something probe() classifies from
# the chat template the way thinking/tool_format are — so it does not live on
# Capability. Instead `engine.vision` is an OPTIONAL engine surface,
# discovered by getattr exactly like sleep_state/prefix_cache_state
# (engine.py's Engine docstring: "OPTIONAL, discovered by getattr and NOT
# part of this Protocol"): a `vision.Vision` when the engine can read images
# for this model, None otherwise. `engine.vision_reason` is the same pattern
# for WHY not (e.g. "text-only model", "runtime mlx", "disabled by
# DRINKME_VISION=0"); an engine that does not publish one still refuses,
# just with a generic reason.


def engine_vision(engine) -> "vision.Vision | None":
    """The engine's Vision, or None for a text-only engine, an engine whose
    architecture has no registered preprocessor, or one with vision turned
    off. FakeEngine has no such attribute by default — getattr's default
    keeps every non-vision test unchanged."""
    return getattr(engine, "vision", None)


def vision_reason(engine) -> str | None:
    """Why THIS engine has no vision, when engine_vision(engine) is None and
    the engine names one; None when it does not (yet) — the refusal below
    still fires, just less specific."""
    return getattr(engine, "vision_reason", None)


def vision_refusal_message(where: str, model_id: str, why: str | None) -> str:
    """The named 400 for an image part this engine cannot read, worded from
    the engine's own reason (`why`, vision_reason(engine); None reads as a
    generic one): `where` is the offending part
    (messages[i].content[j] / messages.i.content.j / input.i.content.j), so
    every dialect's refusal names both the part and the model. The
    per-source refusals (serving/vision.py's `sources_message`: fetching
    off, no --media-path, a Files-API id) are separate: this refusal is
    about the ENGINE, not the source."""
    why = why or "it has no vision capability"
    return (f"{where}: model {model_id!r} cannot read images on this server "
            f"({why}); send text only, or use a vision-capable model.")


def video_reason(engine) -> str | None:
    """Why THIS engine reads no video: its vision_reason when it reads no
    images either, else its Vision's video_reason (no video processor for
    the architecture or in the checkpoint, PyAV not installed); None when
    it does read video."""
    veng = engine_vision(engine)
    if veng is None:
        return vision_reason(engine)
    return None if getattr(veng, "video", None) is not None else veng.video_reason


def video_refusal_message(where: str, model_id: str, why: str | None, *,
                          images: bool = False) -> str:
    """The named 400 for a video part this engine cannot read:
    vision_refusal_message's shape, worded for video (`why`,
    video_reason(engine)). An engine that reads images (`images`) is
    pointed at sending frames as images instead."""
    why = why or "it has no video capability"
    alt = "send frames as image_url parts" if images else "send text only"
    return (f"{where}: model {model_id!r} cannot read video on this server "
            f"({why}); {alt}, or use a video-capable model.")


def video_input_block(veng: "vision.Vision") -> dict:
    """The /v1/models `drinkme.capabilities.videoInput` object for an
    engine that reads video: the processor's sampling rate and frame and
    pixel budgets, the limits serving/video.py adds (the longest clip, the
    byte cap), the containers it reads, and the sources this server accepts
    (image_input_block's, which video shares)."""
    return dict(veng.video.block(), sources=image_input_block(veng)["sources"])


def check_injection(text: str, veng: "vision.Vision | None", where: str) -> None:
    """The injection guard: text that spells one of the architecture's
    image placeholder literals (`<|image_pad|>` and friends,
    Vision.reserved_text) is refused — a request could otherwise inject
    the pad token's own text into an ordinary turn and hijack the M-RoPE
    positions and prefix-cache key meant for a real image. `veng` is the
    engine's Vision (or None): a non-vision engine has nothing reserved to
    guard. template.py has no general special-token escaping regardless —
    a pre-existing gap this guard narrows for the image markers, not
    closes generally."""
    if veng is None:
        return
    for literal in veng.reserved_text:
        if literal and literal in text:
            raise vision.ImageError(
                where, "image_injection",
                f"text must not contain the reserved image marker {literal!r}")


def image_input_block(veng: "vision.Vision") -> dict:
    """The /v1/models `drinkme.capabilities.imageInput` object for a
    vision-capable engine: its pixel cap, the formats serving/vision.py
    decodes, and the sources it actually accepts on THIS server — "data"
    always, "url" when http(s) fetching is on (the default; --no-image-urls
    turns it off), "file" when --media-path is set (off by default)."""
    sources = ["data"]
    if veng.fetch_urls:
        sources.append("url")
    if veng.media_path:
        sources.append("file")
    return {"maxPixels": veng.max_pixels, "formats": list(vision.FORMATS), "sources": sources}
