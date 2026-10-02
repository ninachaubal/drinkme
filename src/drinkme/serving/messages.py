"""The Anthropic Messages API on this server — translation only, pure CPU.

/v1/messages is a FORMAT ADAPTER over machinery that exists: the request
becomes the same
(messages, SampleParams, tools, template kwargs) the OpenAI boundary in
http.py produces, the SAME generation core runs (http._Handler._generate),
and the split result (reasoning | content | tool calls) is rendered back in
Anthropic's shapes. Nothing here touches an engine, a socket or a lock —
http.py owns those — which is what keeps every mapping decision below
unit-testable in isolation and the dialects provably fed the same bytes
(tests/test_serving_messages.py, the cross-surface parity test — three
dialects since serving/responses.py, built in this module's shape).

Request -> internal
  system (string | text blocks)   one {"role": "system"} turn, first; blocks
                                  join with a blank line
  messages[] role "system"        a system turn IN PLACE — Anthropic's
                                  mid-conversation system message, which
                                  Claude Code sends after the first user turn
                                  on every request (seen on the wire; beta
                                  mid-conversation-system-2026-04-07). Qwen's templates render a
                                  non-first system turn where it stands, and
                                  the OpenAI boundary passes one through too.
                                  Text only; an effort-only message
                                  (content []) renders nothing and is dropped
  content string                  the same string
  user text blocks                one user turn (blocks join with a blank line)
  user tool_result blocks         {"role": "tool", "tool_call_id", "content"}
                                  turns, in block order, BEFORE the turn's
                                  remaining text; is_error prefixes "Error: "
                                  and nothing more
  assistant text/thinking/tool_use
                                  content / reasoning_content / tool_calls with
                                  function.arguments as the DICT — exactly the
                                  history shape http.py's OpenAI boundary
                                  parses into (the qwen3_5 template renders
                                  arguments with |items). thinking.signature
                                  is accepted unverified: it is "" on the way
                                  out (local model, nothing to sign) and never
                                  read on the way in. redacted_thinking is
                                  dropped: nothing readable to render.
  image blocks (base64 or url     accepted in a user turn and inside
  source)                         tool_result.content (Claude Code's
                                  screenshot path) when the engine's
                                  capability.engine_vision is set; decoded
                                  through serving/vision.py into
                                  GenerationRequest.images, in template
                                  order. A `url` source is DOWNLOADED by
                                  default (llama.cpp parity) — several
                                  in one turn concurrently — per the
                                  engine's fetch_urls/media_path; `file`
                                  sources (a Files-API id, not a path) and,
                                  on a text-only engine, any `image` block:
                                  a 400 naming the block. `document` blocks
                                  stay refused always
  images in a system turn,        a named 400; so is
  the reserved placeholder text    text that spells an image placeholder
                                  literal (`<|image_pad|>` and friends),
                                  the injection guard
  tools[] {name, description,     OpenAI {"type": "function", ...}; a server
          input_schema}           tool type (bash_*, web_search_*, ...) is a
                                  400 by name — nothing here can run it
  tool_choice auto | none         tools offered | withheld (as the OpenAI path)
  tool_choice any | tool          400 — the one contract: no constrained
                                  decode to a call exists, and no dialect
                                  promises one (the OpenAI paths refuse
                                  "required" and a named function the same
                                  way) — not prompt-and-pray
  thinking enabled | adaptive     enable_thinking=True. budget_tokens is
                                  RECORDED (Parsed.budget_tokens), NOT
                                  enforced: the template has no budget knob
                                  and a cap we cannot honor is not one we
                                  claim. adaptive reads as "on, the model
                                  decides" — which is what Qwen3 does anyway
  thinking disabled               enable_thinking=False
  output_config.effort            reasoning_effort (Qwen3.8's template knob;
                                  the OpenAI path takes the same word top-level)
  output_config.format            {type: "json_schema", schema} -> response_format,
                                  the SAME grammar-constrained path
                                  /v1/chat/completions uses (constrain.py); any
                                  other type, or a missing/invalid schema, is a
                                  400 NAMING the variant — never a silently-
                                  dropped schema. Forces enable_thinking off
                                  (a grammar banning "<" starves a thinking
                                  model) and refuses to combine with tools,
                                  mirroring the OpenAI boundary's posture
  mcp_servers                     400: no MCP connector here
  temperature / top_p / top_k /   SampleParams; max_tokens is REQUIRED
  stop_sequences / max_tokens     (Anthropic's contract; count_tokens excepted)
  metadata, cache_control, strict, anthropic-version / anthropic-beta
                                  accepted and ignored

Internal -> response: content blocks in production order — thinking, text,
tool_use (input is the PARSED object, id toolu_…) — stop_reason end_turn |
max_tokens | stop_sequence (with the string that fired) | tool_use, and
Anthropic's usage accounting: input_tokens EXCLUDES prefix-cache reads, which
ride in cache_read_input_tokens, so a client summing the usage fields for its
context gauge (Claude Code) counts the prompt exactly once.

Streaming: the documented order, event-named SSE — message_start -> per
block [content_block_start, content_block_delta*, content_block_stop] ->
message_delta -> message_stop. Deltas: text_delta, thinking_delta (plus one
empty signature_delta before a thinking block closes, as the real wire
sends), input_json_delta carrying the whole serialized input in ONE partial —
our calls arrive parsed and complete, and the concatenation of the partials
equals the final serialized input byte for byte (tested).

Posture (llama.cpp's): no strong claim of spec compliance;
we name the clients actually driven: curl, the Anthropic SDKs, and the
claude CLI.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass

from . import capability, gen_config, vision
from .engine import GenerationRequest, SampleParamError, validated_sample_params
from .tools import validate_tools


class MessagesError(Exception):
    """A refusal in Anthropic's envelope: HTTP status, error type, message.
    http.py renders it; nothing here writes to a socket."""

    def __init__(self, status: int, etype: str, message: str):
        super().__init__(message)
        self.status, self.etype, self.message = status, etype, message


def invalid(message: str) -> MessagesError:
    return MessagesError(400, "invalid_request_error", message)


class _NoVision(Exception):
    """An `image` block, at the `where` it carries, for an engine without
    vision. parse_request words the 400 from the model and the engine's
    reason (capability.vision_refusal_message), as the chat boundary does."""


def error_body(etype: str, message: str) -> dict:
    """The envelope, for an HTTP body and the SSE `error` event alike."""
    return {"type": "error", "error": {"type": etype, "message": message}}


def new_message_id() -> str:
    return f"msg_{uuid.uuid4().hex[:24]}"


def new_tool_use_id() -> str:
    return f"toolu_{uuid.uuid4().hex[:24]}"


@dataclass
class Parsed:
    """A /v1/messages request, translated: the GenerationRequest http.py's
    generation core takes (engine.py — the same value the OpenAI boundary
    builds), plus the Anthropic-only facts."""

    request: GenerationRequest
    budget_tokens: int | None = None  # advisory: recorded, never enforced


# ------------------------------------------------------------- request side --


def parse_request(req, model_id: str, *, need_max_tokens: bool = True,
                  defaults: dict | None = None, tkw_base: dict | None = None,
                  vision_engine=None, vision_reason: str | None = None) -> Parsed:
    """The whole request, checked at the boundary. Raises MessagesError with
    the client-facing message; the field naming follows Anthropic's own
    (`messages.3.content.1`, `tool_choice`), so a client's error reads the
    same here as against the real thing.

    `defaults` (generation_config.json defaults / aliases and profiles): the engine's effective sampling table, a
    profile's fields already overlaid — Anthropic's wire has no
    repetition_penalty/presence_penalty/frequency_penalty fields at all, so
    those three are ALWAYS `defaults`' value, never request-settable here;
    temperature/top_p/top_k still take the request's own value when given.
    `tkw_base` (aliases and profiles): a profile's chat-template keys, the base layer under
    whatever `thinking`/`output_config.effort` below set. `vision_engine`
    is the engine's Vision (capability.engine_vision), or None for a
    text-only engine — threaded down to convert_messages, which accepts
    `image` blocks only when it is given; `vision_reason`
    (capability.vision_reason) words the refusal when it is not."""
    if not isinstance(req, dict):
        raise invalid("Request body must be a JSON object.")
    model = req.get("model")
    if model is None:
        raise invalid("model: field required")
    if model != model_id:
        raise MessagesError(404, "not_found_error",
                            f"model: {model} (this server serves {model_id!r})")
    mt = req.get("max_tokens")
    if need_max_tokens:
        if mt is None:
            raise invalid("max_tokens: field required")
        if isinstance(mt, bool) or not isinstance(mt, int) or mt < 1:
            raise invalid("max_tokens: must be a positive integer")
    if vision_engine is not None:
        n_images = _count_images(req.get("messages"))
        if n_images:
            try:
                vision.check_image_count(n_images, where="messages")
            except vision.ImageError as e:
                raise invalid(str(e)) from None
    try:
        messages, images = convert_messages(req.get("system"), req.get("messages"),
                                            veng=vision_engine)
    except vision.ImageError as e:
        raise invalid(str(e)) from None
    except _NoVision as e:
        raise invalid(capability.vision_refusal_message(
            str(e), model_id, vision_reason)) from None
    stops = req.get("stop_sequences") or []
    if not isinstance(stops, list) or not all(isinstance(s, str) for s in stops):
        raise invalid("stop_sequences: must be an array of strings")
    sk = gen_config.resolve_sampling(
        {"temperature": req.get("temperature"), "top_p": req.get("top_p"),
         "top_k": req.get("top_k")},
        defaults or {})
    # THE sampling validator (engine.validated_sample_params), after
    # precedence resolution, before anything is rendered: the field named
    # in its own message, Anthropic's 400 envelope
    try:
        params = validated_sample_params(sk, max_tokens=mt, stop=list(stops))
    except SampleParamError as e:
        raise invalid(str(e)) from None
    tools = convert_tools(req.get("tools"))
    tc = req.get("tool_choice")
    if tc is not None:
        if not isinstance(tc, dict) or not isinstance(tc.get("type"), str):
            raise invalid("tool_choice: must be an object with a string 'type'")
        if tc["type"] == "none":
            tools = None
        elif tc["type"] in ("any", "tool"):
            raise invalid(
                f"tool_choice: type {tc['type']!r} (forced tool use) is not "
                "supported by this server — it cannot constrain decoding to a "
                "call and will not pretend to; use {\"type\": \"auto\"}")
        elif tc["type"] != "auto":
            raise invalid(f"tool_choice: unknown type {tc['type']!r}")
    tkw: dict = dict(tkw_base or {})
    budget = None
    th = req.get("thinking")
    if th is not None:
        if not isinstance(th, dict) or not isinstance(th.get("type"), str):
            raise invalid("thinking: must be an object with a string 'type'")
        if th["type"] in ("enabled", "adaptive"):
            tkw["enable_thinking"] = True
            b = th.get("budget_tokens")
            if b is not None:
                if isinstance(b, bool) or not isinstance(b, int) or b < 1:
                    raise invalid("thinking.budget_tokens: must be a positive integer")
                budget = b
        elif th["type"] == "disabled":
            tkw["enable_thinking"] = False
        else:
            raise invalid(f"thinking.type: unknown type {th['type']!r}")
    oc = req.get("output_config")
    if oc is not None:
        if not isinstance(oc, dict):
            raise invalid("output_config: must be an object")
        output_format = oc.get("format")  # the wire object; its schema is the output_schema
        if output_format is not None:
            if not isinstance(output_format, dict) or output_format.get("type") != "json_schema":
                variant = output_format.get("type") if isinstance(output_format, dict) else output_format
                raise invalid(f"output_config.format: type {variant!r} is not "
                              "supported by this server; only 'json_schema' is")
            schema = output_format.get("schema")
            if not isinstance(schema, dict):
                raise invalid("output_config.format.schema: field required "
                              "and must be an object")
            from .constrain import validate_schema

            try:
                validate_schema(schema)
            except ValueError as e:
                raise invalid(str(e)) from None
            if req.get("tools"):
                raise invalid("output_config.format cannot be combined with tools")
            params.output_schema = schema
            params.output_schema_validated = True  # walked once, here
            # constrain.py's posture, made explicit at the boundary: a grammar
            # that bans "<" starves a thinking model (http.py, same reasoning)
            tkw["enable_thinking"] = False
        eff = oc.get("effort")
        if eff is not None:
            if not isinstance(eff, str):
                raise invalid("output_config.effort: must be a string")
            tkw["reasoning_effort"] = eff
    if req.get("mcp_servers"):
        raise invalid("mcp_servers: the MCP connector is not supported by this server")
    stream = req.get("stream")
    if stream is not None and not isinstance(stream, bool):
        raise invalid("stream: must be a boolean")
    return Parsed(GenerationRequest(messages, params, tools=tools, template_kwargs=tkw,
                                    stream=bool(stream), request_id=new_message_id(),
                                    images=images),
                  budget_tokens=budget)


def _count_images(raw) -> int:
    """Total `image` blocks a messages[] array carries, in user turns and
    inside a tool_result — a cheap shape-only walk (no decode), so
    MAX_IMAGES (vision.check_image_count) can refuse before any of them
    decode."""
    if not isinstance(raw, list):
        return 0
    n = 0
    for m in raw:
        content = m.get("content") if isinstance(m, dict) else None
        if not isinstance(content, list):
            continue
        for b in content:
            if not isinstance(b, dict):
                continue
            if b.get("type") == "image":
                n += 1
            elif b.get("type") == "tool_result" and isinstance(b.get("content"), list):
                n += sum(1 for bb in b["content"]
                        if isinstance(bb, dict) and bb.get("type") == "image")
    return n


def convert_messages(system, raw, *, veng=None) -> tuple[list[dict], tuple]:
    """system + messages[] -> (the internal (OpenAI-shaped) history the chat
    template renders, the images it carries in template order). A turn's
    content is a string, or, for a turn carrying images, a parts list in
    wire order (engine.GenerationRequest's docstring). The output of this
    function for a conversation and the output of http.py's OpenAI boundary
    for the same conversation are asserted EQUAL by the parity test.
    `veng` is the engine's Vision, or None for a text-only
    engine — `image` blocks are accepted only when it is given."""
    out: list[dict] = []
    images: list = []
    sys_turn = _system_turn(system, veng)
    if sys_turn is not None:
        out.append(sys_turn)
    if not isinstance(raw, list) or not raw:
        raise invalid("messages: at least one message is required")
    for i, m in enumerate(raw):
        if not isinstance(m, dict):
            raise invalid(f"messages.{i}: must be an object")
        role, content = m.get("role"), m.get("content")
        if role == "user":
            turns, imgs = _user_turn(i, content, veng)
            out.extend(turns)
            images.extend(imgs)
        elif role == "assistant":
            out.append(_assistant_turn(i, content, veng))
        elif role == "system":
            text = _text_only(content, f"messages.{i}.content", veng)
            if text:  # an effort-only message (content []) has nothing to render
                out.append({"role": "system", "content": text})
        else:
            raise invalid(f"messages.{i}.role: must be 'user', 'assistant' or "
                          f"'system', got {role!r}")
    return out, tuple(images)


def _text_only(content, where: str, veng=None) -> str:
    """A string, or text blocks only (joined with a blank line); anything
    else is a 400 naming the block. For the system prompt and system turns —
    images are never accepted here: a block that is not `text` already
    names itself and its location, which is what "images in a system turn
    are refused by name" means for this path."""
    if isinstance(content, str):
        capability.check_injection(content, veng, where)
        return content
    if isinstance(content, list):
        parts = []
        for j, b in enumerate(content):
            if not isinstance(b, dict) or b.get("type") != "text" \
                    or not isinstance(b.get("text"), str):
                t = b.get("type") if isinstance(b, dict) else None
                raise invalid(f"{where}.{j}: must be a text block, got {t!r}")
            capability.check_injection(b["text"], veng, f"{where}.{j}")
            parts.append(b["text"])
        return "\n\n".join(parts)
    raise invalid(f"{where}: must be a string or an array of text blocks")


def _system_turn(system, veng=None) -> dict | None:
    if system is None:
        return None
    if isinstance(system, list) and not system:
        # an empty top-level system array would otherwise silently render as
        # "no system prompt" — indistinguishable from omitting the field —
        # rather than naming the empty array as the likely client mistake it
        # is. A mid-conversation system TURN's content:[] stays a deliberate
        # no-op (convert_messages' "effort-only message" rule) — this guard
        # is `system` only.
        raise invalid("system: must not be an empty array")
    text = _text_only(system, "system", veng)
    return {"role": "system", "content": text} if text else None


def _block(i: int, j: int, b) -> str:
    """A content block's type, or a 400 for a non-block."""
    if not isinstance(b, dict) or not isinstance(b.get("type"), str):
        raise invalid(f"messages.{i}.content.{j}: must be a content block with a 'type'")
    return b["type"]


def _text(i: int, j: int, b: dict) -> str:
    if not isinstance(b.get("text"), str):
        raise invalid(f"messages.{i}.content.{j}.text: must be a string")
    return b["text"]


def _unsupported(i: int, j: int, t: str, role: str) -> Exception:
    if t == "image" and role == "user":
        return _NoVision(f"messages.{i}.content.{j}")
    if t == "document":
        return invalid(f"messages.{i}.content.{j}: {t!r} blocks are not supported "
                       "by this server.")
    return invalid(f"messages.{i}.content.{j}: unknown or unsupported block type "
                   f"{t!r} in a {role} turn")


def _image_source(where: str, b: dict, veng):
    """An `image` block's `source` -> a PreparedImage (a `base64` source:
    decoded inline, no network) or a ("pending", url, where) marker (a
    `url` source: its fetch is batched with any other url sources in the
    SAME block list — _resolve_images below — so several download
    concurrently). `file` sources are always refused: an Anthropic
    Files-API id, not a path."""
    src = b.get("source")
    if not isinstance(src, dict) or not isinstance(src.get("type"), str):
        raise invalid(f"{where}.source: must be an object with a string 'type'")
    stype = src["type"]
    if stype == "url":
        url = src.get("url")
        if not isinstance(url, str):
            raise invalid(f"{where}.source.url: must be a string")
        return "pending", url, where
    if stype == "file":
        raise vision.ImageError(
            f"{where}.source", "image_files_api",
            "file sources are not supported by this server (no Files API); send "
            + vision.sources_message(fetch_urls=veng.fetch_urls, media_path=veng.media_path))
    if stype != "base64":
        raise invalid(f"{where}.source.type: {stype!r} is not supported; send "
                      "{\"type\": \"base64\", \"media_type\": ..., \"data\": ...} or "
                      "{\"type\": \"url\", \"url\": ...}")
    enc = vision.parse_base64(src.get("data"), src.get("media_type"), where=where)
    return veng.prepare(enc, where=where)


def _resolve_images(slots: list, veng) -> tuple:
    """slots (wire order): a PreparedImage (a base64 source, decoded
    inline) or a ("pending", url, where) marker (`_image_source` above)
    for a url source not yet fetched. Every pending url resolves through
    ONE concurrent batch (vision.parse_image_urls), so several url sources
    in the same block list download at once. Returns PreparedImages in the
    same order as `slots`."""
    pending = [(idx, s[1], s[2]) for idx, s in enumerate(slots) if isinstance(s, tuple)]
    if not pending:
        return tuple(slots)
    encs = vision.parse_image_urls([(u, w) for _, u, w in pending],
                                   fetch_urls=veng.fetch_urls, media_path=veng.media_path)
    out = list(slots)
    for (idx, _, where), enc in zip(pending, encs):
        out[idx] = veng.prepare(enc, where=where)
    return tuple(out)


def _result_content(i: int, j: int, content, veng) -> tuple:
    """tool_result.content: absent, a string, text blocks, or (when `veng`
    is given) text and `image` blocks — Claude Code's screenshot path.
    Returns (content, images): content is a string
    when there is no image, else a parts list in wire order."""
    if content is None:
        return "", ()
    if isinstance(content, str):
        capability.check_injection(content, veng, f"messages.{i}.content.{j}.content")
        return content, ()
    if isinstance(content, list):
        parts: list[dict] = []
        slots: list = []
        for k, b in enumerate(content):
            t = b.get("type") if isinstance(b, dict) else None
            where = f"messages.{i}.content.{j}.content.{k}"
            if t == "text" and isinstance(b.get("text"), str):
                capability.check_injection(b["text"], veng, where)
                parts.append({"type": "text", "text": b["text"]})
            elif t == "image":
                if veng is None:
                    raise _NoVision(where)
                slots.append(_image_source(where, b, veng))
                parts.append({"type": "image"})
            else:
                raise invalid(f"{where}: {t!r} blocks are not supported inside a "
                              "tool_result on this server (text only)")
        images = _resolve_images(slots, veng)
        if not images:
            return "\n\n".join(p["text"] for p in parts), ()
        return parts, images
    raise invalid(f"messages.{i}.content.{j}.content: must be a string or an "
                  "array of text blocks")


def _user_turn(i: int, content, veng=None) -> tuple[list[dict], tuple]:
    if isinstance(content, str):
        capability.check_injection(content, veng, f"messages.{i}.content")
        return [{"role": "user", "content": content}], ()
    if not isinstance(content, list):
        raise invalid(f"messages.{i}.content: must be a string or an array of content blocks")
    turns: list[dict] = []
    parts: list[dict] = []  # text/image parts, wire order, for the trailing user turn
    slots: list = []  # this turn's own image blocks (_image_source), plus a
    # tool_result's already-resolved images, both in wire order
    for j, b in enumerate(content):
        t = _block(i, j, b)
        where = f"messages.{i}.content.{j}"
        if t == "text":
            text = _text(i, j, b)
            capability.check_injection(text, veng, where)
            parts.append({"type": "text", "text": text})
        elif t == "image":
            if veng is None:
                raise _unsupported(i, j, t, "user")
            slots.append(_image_source(where, b, veng))
            parts.append({"type": "image"})
        elif t == "tool_result":
            tid = b.get("tool_use_id")
            if not isinstance(tid, str) or not tid:
                raise invalid(f"messages.{i}.content.{j}.tool_use_id: field required")
            rcontent, rimages = _result_content(i, j, b.get("content"), veng)
            if b.get("is_error"):
                rcontent = ("Error: " + rcontent if isinstance(rcontent, str)
                           else [{"type": "text", "text": "Error: "}] + rcontent)
            turns.append({"role": "tool", "tool_call_id": tid, "content": rcontent})
            slots.extend(rimages)  # already resolved by _result_content
        else:
            raise _unsupported(i, j, t, "user")
    images = _resolve_images(slots, veng)
    if parts:
        if any(p["type"] == "image" for p in parts):
            turns.append({"role": "user", "content": parts})
        else:
            turns.append({"role": "user", "content":
                          "\n\n".join(p["text"] for p in parts)})
    if not turns:
        raise invalid(f"messages.{i}.content: must not be empty")
    return turns, images


def _assistant_turn(i: int, content, veng=None) -> dict:
    if isinstance(content, str):
        capability.check_injection(content, veng, f"messages.{i}.content")
        return {"role": "assistant", "content": content}
    if not isinstance(content, list):
        raise invalid(f"messages.{i}.content: must be a string or an array of content blocks")
    texts: list[str] = []
    thoughts: list[str] = []
    calls: list[dict] = []
    for j, b in enumerate(content):
        t = _block(i, j, b)
        where = f"messages.{i}.content.{j}"
        if t == "text":
            text = _text(i, j, b)
            capability.check_injection(text, veng, where)
            texts.append(text)
        elif t == "thinking":
            if not isinstance(b.get("thinking"), str):
                raise invalid(f"messages.{i}.content.{j}.thinking: must be a string")
            capability.check_injection(b["thinking"], veng, where)
            thoughts.append(b["thinking"])  # signature: accepted, never verified
        elif t == "redacted_thinking":
            continue  # nothing readable; dropped, never echoed as text
        elif t == "tool_use":
            cid, name, inp = b.get("id"), b.get("name"), b.get("input", {})
            if not isinstance(cid, str) or not cid:
                raise invalid(f"messages.{i}.content.{j}.id: field required")
            if not isinstance(name, str) or not name:
                raise invalid(f"messages.{i}.content.{j}.name: field required")
            if not isinstance(inp, dict):
                raise invalid(f"messages.{i}.content.{j}.input: must be an object")
            calls.append({"id": cid, "type": "function",
                          "function": {"name": name, "arguments": inp}})
        else:
            raise _unsupported(i, j, t, "assistant")
    turn: dict = {"role": "assistant", "content": "\n\n".join(texts)}
    if thoughts:
        turn["reasoning_content"] = "\n\n".join(thoughts)
    if calls:
        turn["tool_calls"] = calls
    return turn


def convert_tools(raw) -> list | None:
    """tools[] -> the OpenAI list validate_tools accepts, or None for none.
    Only custom tools: a server-side tool type names something this server
    cannot run, and is refused by name."""
    if raw is None:
        return None
    if not isinstance(raw, list):
        raise invalid("tools: must be an array")
    out = []
    for i, t in enumerate(raw):
        if not isinstance(t, dict):
            raise invalid(f"tools.{i}: must be an object")
        ttype = t.get("type")
        if ttype not in (None, "custom"):
            raise invalid(f"tools.{i}: type {ttype!r} is a server-side tool, which "
                          "this server cannot run; only custom tools (name + "
                          "input_schema) are supported")
        name = t.get("name")
        if not isinstance(name, str) or not name:
            raise invalid(f"tools.{i}.name: must be a non-empty string")
        fn: dict = {"name": name}
        if t.get("description") is not None:
            if not isinstance(t["description"], str):
                raise invalid(f"tools.{i}.description: must be a string")
            fn["description"] = t["description"]
        if t.get("input_schema") is not None:
            if not isinstance(t["input_schema"], dict):
                raise invalid(f"tools.{i}.input_schema: must be an object (a JSON Schema)")
            fn["parameters"] = t["input_schema"]
        out.append({"type": "function", "function": fn})
    return validate_tools(out) or None


# ------------------------------------------------------------ response side --
# `turn` below is http.Turn — duck-typed (reasoning, content, tool_calls,
# finish, stop_sequence, prompt_tokens, completion_tokens, cached_tokens) so
# this module never imports the HTTP layer that imports it.


def stop_reason(finish: str, stop_sequence: str | None) -> tuple[str, str | None]:
    """Engine finish vocabulary -> (stop_reason, stop_sequence)."""
    if finish == "tool_calls":
        return "tool_use", None
    if finish == "length":
        return "max_tokens", None
    if stop_sequence is not None:
        return "stop_sequence", stop_sequence
    return "end_turn", None


def usage(turn, scale: float = 1.0) -> dict:
    """Anthropic's accounting: input_tokens is what was NOT served from the
    prefix cache; the reused part is cache_read_input_tokens. Summing the
    fields gives the prompt length once. Nothing is ever "created" in a cache
    a client can address, so cache_creation_input_tokens is 0.

    `scale` is the advertised context's lie (http.py's _ctx_scale), applied
    to input_tokens ONLY — cache_read/output stay real, because Claude
    Code's auto-compact watches the input side and nothing here should
    change what the engine or the bench actually measured."""
    return {"input_tokens": round((turn.prompt_tokens - turn.cached_tokens) * scale),
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": turn.cached_tokens,
            "output_tokens": turn.completion_tokens}


def content_blocks(reasoning: str, content: str, tool_calls: list | None) -> list[dict]:
    """The blocks in production order: thinking, text, tool_use. Empty
    channels produce no block (Anthropic returns [] for an empty answer)."""
    blocks: list[dict] = []
    if reasoning:
        blocks.append({"type": "thinking", "thinking": reasoning, "signature": ""})
    if content:
        blocks.append({"type": "text", "text": content})
    for c in tool_calls or []:
        blocks.append({"type": "tool_use", "id": new_tool_use_id(),
                       "name": c["name"], "input": c.get("arguments", {})})
    return blocks


def render_message(msg_id: str, model: str, turn, scale: float = 1.0) -> dict:
    reason, seq = stop_reason(turn.finish, turn.stop_sequence)
    return {"id": msg_id, "type": "message", "role": "assistant", "model": model,
            "content": content_blocks(turn.reasoning, turn.content, turn.tool_calls),
            "stop_reason": reason, "stop_sequence": seq, "usage": usage(turn, scale)}


# ---------------------------------------------------------------- streaming --
# Each ev_* returns (event name, payload) — the two halves of one SSE frame.


def ev_message_start(msg_id: str, model: str, input_tokens: int) -> tuple[str, dict]:
    """Announced BEFORE generation, so input_tokens is the pre-render count
    (Engine.count_tokens: the same template, the same tokenizer) and the cache
    split is not known yet; message_delta carries the authoritative usage."""
    return "message_start", {"type": "message_start", "message": {
        "id": msg_id, "type": "message", "role": "assistant", "model": model,
        "content": [], "stop_reason": None, "stop_sequence": None,
        "usage": {"input_tokens": input_tokens, "cache_creation_input_tokens": 0,
                  "cache_read_input_tokens": 0, "output_tokens": 0}}}


def ev_block_start(index: int, block: dict) -> tuple[str, dict]:
    return "content_block_start", {"type": "content_block_start", "index": index,
                                   "content_block": block}


def ev_block_delta(index: int, delta: dict) -> tuple[str, dict]:
    return "content_block_delta", {"type": "content_block_delta", "index": index,
                                   "delta": delta}


def ev_block_stop(index: int) -> tuple[str, dict]:
    return "content_block_stop", {"type": "content_block_stop", "index": index}


def ev_message_delta(turn, scale: float = 1.0) -> tuple[str, dict]:
    reason, seq = stop_reason(turn.finish, turn.stop_sequence)
    return "message_delta", {"type": "message_delta",
                             "delta": {"stop_reason": reason, "stop_sequence": seq},
                             "usage": usage(turn, scale)}


def ev_message_stop() -> tuple[str, dict]:
    return "message_stop", {"type": "message_stop"}


def ev_error(etype: str, message: str) -> tuple[str, dict]:
    return "error", error_body(etype, message)


class StreamBlocks:
    """Content-block bookkeeping for one streamed message: which block is
    open, at what index, and the start/stop events around it. send(name,
    payload) -> bool is the wire; False means the client is gone and is
    returned straight up — the on_delta contract, one layer out."""

    _START = {"thinking": {"type": "thinking", "thinking": "", "signature": ""},
              "text": {"type": "text", "text": ""}}

    def __init__(self, send):
        self._send = send
        self.index = -1
        self.open: str | None = None  # "thinking" | "text" | None

    def delta(self, channel: str, text: str) -> bool:
        """One piece from the core's split: 'reasoning' -> a thinking block,
        'content' -> a text block; a channel change closes the open block
        and opens the next index."""
        kind = "thinking" if channel == "reasoning" else "text"
        if self.open != kind:
            if not self.close():
                return False
            self.index += 1
            self.open = kind
            if not self._send(*ev_block_start(self.index, dict(self._START[kind]))):
                return False
        return self._send(*ev_block_delta(self.index, {"type": f"{kind}_delta", kind: text}))

    def close(self) -> bool:
        if self.open is None:
            return True
        ok = True
        if self.open == "thinking":  # the real wire signs before it closes; ours is empty
            ok = self._send(*ev_block_delta(self.index, {"type": "signature_delta",
                                                         "signature": ""}))
        self.open = None
        return ok and self._send(*ev_block_stop(self.index))

    def tool_use(self, call: dict) -> bool:
        """One complete parsed call as a tool_use block: start (input {}),
        ONE input_json_delta with the whole serialized input, stop."""
        if not self.close():
            return False
        self.index += 1
        block = {"type": "tool_use", "id": new_tool_use_id(), "name": call["name"],
                 "input": {}}
        partial = json.dumps(call.get("arguments", {}))
        return (self._send(*ev_block_start(self.index, block))
                and self._send(*ev_block_delta(self.index, {"type": "input_json_delta",
                                                            "partial_json": partial}))
                and self._send(*ev_block_stop(self.index)))
