"""The OpenAI Responses API on this server — translation only, pure CPU.

/v1/responses is the third dialect and, like /v1/messages, a FORMAT ADAPTER
over machinery that exists: the request
becomes the same (messages, SampleParams, tools, template kwargs) the
chat-completions boundary in http.py produces, the SAME generation core runs
(http._Handler._generate), and the split result (reasoning | content | tool
calls) is rendered back in the Responses shapes — output items, or the
event-named SSE sequence. Nothing here touches an engine, a socket or a
lock; http.py owns those. The cross-surface parity test
(tests/test_serving_messages.py) holds all three dialects to byte-identical
engine inputs for one conversation.

Why a third dialect: the newer OpenAI-SDK agents (the Agents SDK, Codex CLI)
default to Responses and cannot be pointed at /v1/chat/completions. The
surface followed is the official SDK's types (openai@7.15.0
resources/responses/responses.d.ts) and the openai-openapi spec; vLLM's
entrypoints/openai/responses/ was read for its input-item merge rule.

Request -> internal
  instructions                    one {"role": "system"} turn, first
  input string                    one user turn
  input[] message                 roles user / assistant / system / developer
                                  (developer = system, as the chat boundary
                                  folds it; a system turn stands IN PLACE).
                                  content: a string, or parts — input_text /
                                  output_text / refusal carry text, and parts
                                  CONCATENATE with no separator, exactly as
                                  the chat boundary's _content_text joins a
                                  parts array (same client, same bytes on
                                  either OpenAI route). input_image (an
                                  image_url: data:, http(s) — fetched by
                                  default — or file:// under
                                  --media-path; `detail` maps per
                                  serving/vision.py) is accepted in a USER
                                  message when the engine's
                                  capability.engine_vision is set, decoded
                                  into GenerationRequest.images in template
                                  order; its `file_id` variant (no Files
                                  API), input_file, input_audio, and any
                                  image outside a user message, are a 400
                                  naming the item and part index. So is
                                  text that spells an image placeholder
                                  literal (the injection guard)
  input[] function_call           an assistant turn's tool_call: id =
                                  call_id, name, arguments parsed from the
                                  JSON string into the DICT the chat history
                                  shape carries (http._normalize_messages)
  input[] function_call_output    {"role": "tool", "tool_call_id": call_id,
                                  "content": output}; output is a string or
                                  input_text parts
  input[] reasoning               the assistant turn's reasoning_content.
                                  Text is read from content[] (reasoning_text)
                                  or, failing that, summary[] (summary_text —
                                  where this server puts it on the way out);
                                  parts join with a blank line. encrypted_
                                  content is dropped: nothing readable
  ASSISTANT-SIDE MERGE            consecutive reasoning / assistant message /
                                  function_call items are ONE assistant turn
                                  (vLLM's rule): a field already filled starts
                                  the next turn. This is what makes a response
                                  echoed back as input render as the history
                                  shape the chat boundary would have produced —
                                  and the 27B's prefix cache only extends when
                                  the reasoning is passed back (serving/
                                  messages.py's thinking note)
  input[] anything else           400 naming the type: item_reference (no
                                  state here), the hosted tool calls and their
                                  outputs (web_search_call, computer_call,
                                  mcp_call, local_shell_call, custom_tool_call,
                                  ...) — nothing here can run them
  tools[] {type: function, name,  OpenAI {"type": "function", "function":
          description,            {...}} for validate_tools; `strict` is
          parameters, strict}     accepted and ignored (constrained decoding
                                  to a schema exists for text.format, not for
                                  call arguments). Any other type (web_search,
                                  file_search, code_interpreter, mcp, custom,
                                  ...) is a 400 by name
  tool_choice auto | none         tools offered | withheld (as the chat path)
  tool_choice required |          400 — the one contract: no constrained
              {type: function}    decode to a call exists, and no dialect
                                  promises one (the chat path refuses the
                                  same way; the
                                  /v1/messages posture for any/tool)
  max_output_tokens               max_tokens (default SampleParams'; not
                                  required, unlike Anthropic's)
  temperature / top_p             SampleParams via gen_config.resolve_sampling;
                                  top_k / repetition_penalty / presence_
                                  penalty / frequency_penalty / seed are
                                  accepted by the chat path's names as the
                                  same drinkme extensions (local clients send
                                  top_k). There is no `stop` on this wire
  text.format                     {type: text} nothing; {type: json_object}
                                  and {type: json_schema, schema} ->
                                  response_format, the SAME grammar path
                                  /v1/chat/completions uses (constrain.py);
                                  validated here by name, forces
                                  enable_thinking off, refuses to combine with
                                  tools — the chat boundary's posture.
                                  text.verbosity accepted, ignored
  reasoning.effort                reasoning_effort (the chat path's top-level
                                  word); reasoning.summary etc. ignored
  chat_template_kwargs            the chat path's extension, same rule: the
                                  template owns its vocabulary, the server's
                                  own apply_chat_template arguments are
                                  refused by name (template.RESERVED_KWARGS)
  previous_response_id,           400, legibly: this server keeps no response
  conversation                    state; send the full input every turn (the
                                  store:false way, which Codex does)
  prompt (template ref),          400 by name: server-side objects that do
  background                      not exist here
  store                           accepted and IGNORED — nothing is stored,
                                  and the response says store: false whatever
                                  was asked
  parallel_tool_calls, metadata,  accepted and ignored (echoed where the
  user, truncation, include,      response object carries them)
  stream_options, service_tier,
  safety_identifier, top_logprobs,
  prompt_cache_key, ...

Internal -> response: {id: resp_…, object: response, created_at, status,
model, output, usage, error, incomplete_details, and the echoed knobs}.
output items in production order — a `reasoning` item (rs_…, our reasoning
text as ONE summary_text part; there is no encrypted_content) when
reasoning is non-empty, a `message` item (msg_…, one output_text part,
annotations []) when content is non-empty, one `function_call` item per
call (fc_… / call_…, arguments as the JSON STRING). status is `completed`,
or `incomplete` with incomplete_details {reason: max_output_tokens} on a
length stop. usage: input_tokens is the REAL prompt count (the advertised-
ctx lie is scoped to /v1/messages, as for the chat dialect),
input_tokens_details.cached_tokens the prefix-cache reuse,
output_tokens_details.reasoning_tokens is 0 — the engine reports one output
count and the reasoning/content split is by text, not by token; a
re-tokenized guess would be a number that looks measured and is not.

Streaming (`stream: true`; each frame `event: <type>` + `data: <json>`, a
monotonically increasing sequence_number from 0, no [DONE] sentinel):
response.created -> response.in_progress -> per item, output_item.added
-> the item's part/delta/done events -> output_item.done -> finally
response.completed (or response.incomplete) carrying the FULL response
object. Function-call arguments arrive parsed and complete from the core,
so they stream as ONE function_call_arguments.delta carrying the whole
serialized string, and the concatenation of deltas equals the final
`arguments` byte for byte (tested; /v1/messages' input_json_delta rule).
Error mid-stream: an `error` event, then response.failed with the response
object's error filled in. output_index / content_index / item_id are the
same numbers and ids the final object carries — the SDK's accumulator
rejects a stream where they do not line up.

Posture: the clients named are the official openai SDKs (bench/
responses_sdk_smoke.mjs) and Codex CLI (bench/responses_codex_smoke.sh);
no stronger compliance claim than that.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field

from . import capability, gen_config, vision
from .engine import GenerationRequest, SampleParamError, validated_sample_params
from .template import RESERVED_KWARGS
from .tools import validate_tools


class ResponsesError(Exception):
    """A refusal in the OpenAI envelope: HTTP status, message, error type,
    code. http.py renders it (its _error); nothing here writes to a socket."""

    def __init__(self, status: int, message: str, etype: str = "invalid_request_error",
                 code: str | None = None):
        super().__init__(message)
        self.status, self.message, self.etype, self.code = status, message, etype, code


def invalid(message: str, code: str | None = None) -> ResponsesError:
    return ResponsesError(400, message, "invalid_request_error", code)


class _NoVision(Exception):
    """An `input_image` part in a user message, at the `where` it carries,
    for an engine without vision. parse_request words the 400 from the
    model and the engine's reason (capability.vision_refusal_message), as
    the chat boundary does."""


def new_response_id() -> str:
    return f"resp_{uuid.uuid4().hex[:24]}"


def new_reasoning_id() -> str:
    return f"rs_{uuid.uuid4().hex[:24]}"


def new_message_id() -> str:
    return f"msg_{uuid.uuid4().hex[:24]}"


def new_function_call_id() -> str:
    return f"fc_{uuid.uuid4().hex[:24]}"


def new_call_id() -> str:
    return f"call_{uuid.uuid4().hex[:12]}"  # tools.to_openai_calls' shape


# Hosted/built-in tool types the official surface knows and this server can
# never run — refused by name, never silently dropped from the offer.
_HOSTED_TOOL_TYPES = frozenset({
    "web_search", "web_search_preview", "web_search_preview_2025_03_11",
    "file_search", "code_interpreter", "computer", "computer_use_preview",
    "computer_use", "image_generation", "mcp", "local_shell", "shell",
    "apply_patch", "custom", "tool_search", "programmatic_tool_calling"})
# `namespace` is NOT hosted: it is a container of function tools (Codex sends
# one). It is flattened — its function tools are offered under their own
# names and the namespace is stamped back on the function_call item.

# Sampling fields read off the request by the chat boundary's names — the
# official Responses surface has temperature and top_p; the rest are the same
# drinkme extensions /v1/chat/completions honors, so one client sends one
# set of knobs to either OpenAI route.
_SAMPLING = gen_config.SAMPLING_FIELDS


@dataclass
class Parsed:
    """A /v1/responses request, translated: exactly what http.py's chat
    boundary hands the generation core, plus what the response object echoes."""

    request: GenerationRequest
    echo: dict = field(default_factory=dict)  # the request knobs a Response carries back
    namespaces: dict = field(default_factory=dict)  # function name -> namespace it was offered under


# ------------------------------------------------------------- request side --


def parse_request(req, model_id: str, *, defaults: dict | None = None,
                  tkw_base: dict | None = None, vision_engine=None,
                  vision_reason: str | None = None) -> Parsed:
    """The whole request, checked at the boundary. Raises ResponsesError
    with the client-facing message; field naming follows the request's own
    (`input.3.content.1`, `tools.0`, `text.format`).

    `defaults` / `tkw_base`: as for messages.parse_request — the engine's
    effective sampling table with a profile overlaid, and a profile's
    chat-template keys as the base layer under this request's own.
    `vision_engine` is the engine's Vision
    (capability.engine_vision), or None for a text-only engine — threaded
    to convert_input, which accepts `input_image` parts (user turns only)
    only when it is given; `vision_reason` (capability.vision_reason)
    words the refusal when it is not."""
    if not isinstance(req, dict):
        raise invalid("Request body must be a JSON object.")
    model = req.get("model")
    if model is not None and not isinstance(model, str):
        raise invalid("model: must be a string")
    if isinstance(model, str) and model != model_id:
        raise ResponsesError(404, f"model: {model} (this server serves {model_id!r})",
                             code="model_not_found")
    # server-side state and objects this server does not have — refused by
    # name, with the way forward, before anything else is looked at
    if req.get("previous_response_id") is not None:
        raise invalid("previous_response_id: this server keeps no response state; "
                      "send the full input (every prior item) on each request")
    if req.get("conversation") is not None:
        raise invalid("conversation: this server keeps no conversation state; "
                      "send the full input on each request")
    if req.get("prompt") is not None:
        raise invalid("prompt: stored prompt templates are not supported by this "
                      "server; send instructions and input directly")
    if req.get("background"):
        raise invalid("background: this server answers inline only (no background "
                      "mode, nothing to poll)")
    instructions = req.get("instructions")
    if instructions is not None and not isinstance(instructions, str):
        raise invalid("instructions: must be a string")
    if vision_engine is not None:
        n_images = _count_images(req.get("input"))
        if n_images:
            try:
                vision.check_image_count(n_images, where="input")
            except vision.ImageError as e:
                raise invalid(str(e), code=e.code) from None
    try:
        messages, images = convert_input(instructions, req.get("input"), veng=vision_engine)
    except vision.ImageError as e:
        raise invalid(str(e), code=e.code) from None
    except _NoVision as e:
        raise invalid(capability.vision_refusal_message(
            str(e), model_id, vision_reason)) from None
    mt = req.get("max_output_tokens")
    sk = gen_config.resolve_sampling({f: req.get(f) for f in _SAMPLING}, defaults or {})
    # THE sampling validator (engine.validated_sample_params), after
    # precedence resolution, before anything is rendered: the field named
    # in its own message, this API's 400 envelope
    try:
        params = validated_sample_params(sk, seed=req.get("seed"), max_tokens=mt,
                                         max_tokens_field="max_output_tokens")
    except SampleParamError as e:
        raise invalid(str(e)) from None
    tools, namespaces = convert_tools(req.get("tools"))
    tc = req.get("tool_choice")
    if tc is not None:
        if tc == "none":
            tools = None
        elif tc == "required" or (isinstance(tc, dict) and tc.get("type") == "function"):
            what = "required" if tc == "required" else f"function {tc.get('name')!r}"
            raise invalid(
                f"tool_choice: {what} (forced tool use) is not supported by this "
                "server — it cannot constrain decoding to a call and will not "
                "pretend to; use \"auto\" or \"none\"")
        elif tc != "auto":
            variant = tc.get("type") if isinstance(tc, dict) else tc
            raise invalid(f"tool_choice: {variant!r} is not supported by this server; "
                          "use \"auto\" or \"none\"")
    tkw: dict = dict(tkw_base or {})
    rs = req.get("reasoning")
    if rs is not None:
        if not isinstance(rs, dict):
            raise invalid("reasoning: must be an object")
        eff = rs.get("effort")
        if eff is not None:
            if not isinstance(eff, str):
                raise invalid("reasoning.effort: must be a string")
            tkw["reasoning_effort"] = eff
    ctk = req.get("chat_template_kwargs")
    if ctk is not None:
        if not isinstance(ctk, dict):
            raise invalid("chat_template_kwargs: must be an object")
        bad = RESERVED_KWARGS.intersection(ctk)
        if bad:
            raise invalid("chat_template_kwargs may not set "
                          f"{', '.join(sorted(bad))} — the server owns those.")
        tkw.update(ctk)
    text = req.get("text")
    if text is not None:
        if not isinstance(text, dict):
            raise invalid("text: must be an object")
        schema = _output_schema(text.get("format"))
        if schema is not None:
            from .constrain import validate_schema

            try:
                validate_schema(schema)
            except ValueError as e:
                raise invalid(str(e)) from None
            if req.get("tools"):
                raise invalid("text.format cannot be combined with tools")
            params.output_schema = schema
            params.output_schema_validated = True  # walked once, here
            # constrain.py's posture, made explicit at the boundary: a grammar
            # that bans "<" starves a thinking model (http.py, same reasoning)
            tkw["enable_thinking"] = False
    stream = req.get("stream")
    if stream is not None and not isinstance(stream, bool):
        raise invalid("stream: must be a boolean")
    echo = {
        "instructions": instructions,
        "max_output_tokens": mt,
        "temperature": params.temperature,
        "top_p": params.top_p,
        "tools": req.get("tools") or [],
        "tool_choice": tc if tc is not None else "auto",
        "text": text if text is not None else {"format": {"type": "text"}},
        "reasoning": rs,
        "metadata": req.get("metadata") if isinstance(req.get("metadata"), dict) else {},
        "parallel_tool_calls": bool(req.get("parallel_tool_calls", True)),
        "truncation": req.get("truncation") or "disabled",
    }
    return Parsed(GenerationRequest(messages, params, tools=tools, template_kwargs=tkw,
                                    stream=bool(stream), request_id=new_response_id(),
                                    images=images),
                  echo=echo,
                  namespaces=namespaces)


def _output_schema(text_format) -> dict | None:
    """text.format (the wire object) -> the JSON Schema to constrain to
    (SampleParams.output_schema), or None for plain text. Unknown variants
    and a missing/invalid schema are a 400 NAMING the variant — never a
    silently-dropped schema."""
    if text_format is None:
        return None
    if not isinstance(text_format, dict) or not isinstance(text_format.get("type"), str):
        raise invalid("text.format: must be an object with a string 'type'")
    t = text_format["type"]
    if t == "text":
        return None
    if t == "json_object":
        return {"type": "object"}
    if t == "json_schema":
        schema = text_format.get("schema")
        if not isinstance(schema, dict):
            raise invalid("text.format.schema: field required and must be an object")
        return schema
    raise invalid(f"text.format: type {t!r} is not supported by this server; use "
                  "'text', 'json_object' or 'json_schema'")


class _Assistant:
    """The assistant-side merge (module docstring): reasoning, message text
    and function_call items that arrive consecutively are one turn. A field
    is `None` until an item fills it, so a second message (or reasoning)
    item is what starts the next turn — `content` "" alone would not tell
    "filled with nothing" from "not yet"."""

    def __init__(self):
        self.reasoning: str | None = None
        self.content: str | None = None
        self.calls: list[dict] = []

    def render(self) -> dict:
        turn: dict = {"role": "assistant", "content": self.content or ""}
        if self.reasoning:
            turn["reasoning_content"] = self.reasoning
        if self.calls:
            turn["tool_calls"] = self.calls
        return turn


def _count_images(raw) -> int:
    """Total `input_image` parts a Responses input[] array carries — a
    cheap shape-only walk (no decode), so MAX_IMAGES
    (vision.check_image_count) can refuse before any of them decode."""
    if not isinstance(raw, list):
        return 0
    n = 0
    for item in raw:
        content = item.get("content") if isinstance(item, dict) else None
        if isinstance(content, list):
            n += sum(1 for p in content if isinstance(p, dict) and p.get("type") == "input_image")
    return n


def convert_input(instructions, raw, *, veng=None) -> tuple[list[dict], tuple]:
    """instructions + input -> (the internal (chat-shaped) history the chat
    template renders, the images it carries in template order). Asserted
    EQUAL to what the chat boundary and the Anthropic adapter produce for
    the same conversation by the parity test (text-only requests; images
    are this dialect's own extension). `veng` is the
    engine's Vision, or None for a text-only engine — `input_image` parts
    are accepted, in USER messages only, only when it is given."""
    out: list[dict] = []
    images: list = []
    if instructions:
        capability.check_injection(instructions, veng, "instructions")
        out.append({"role": "system", "content": instructions})
    if isinstance(raw, str):
        capability.check_injection(raw, veng, "input")
        out.append({"role": "user", "content": raw})
        return out, ()
    if not isinstance(raw, list) or not raw:
        raise invalid("input: field required; a string or a non-empty array of items")
    pending: _Assistant | None = None

    def flush():
        nonlocal pending
        if pending is not None:
            out.append(pending.render())
            pending = None

    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            raise invalid(f"input.{i}: must be an object")
        t = item.get("type")
        if t is None and "role" in item:
            t = "message"  # EasyInputMessage: {role, content} with no type
        if t == "message":
            role = item.get("role")
            if role == "assistant":
                text = _message_text(i, item.get("content"), veng)
                if pending is None or pending.content is not None:
                    flush()
                    pending = _Assistant()
                pending.content = text
                continue
            flush()
            if role == "user":
                content, imgs = _user_content(i, item.get("content"), veng)
                images.extend(imgs)
                out.append({"role": "user", "content": content})
            elif role in ("system", "developer"):
                out.append({"role": "system" if role == "developer" else role,
                            "content": _message_text(i, item.get("content"), veng)})
            else:
                raise invalid(f"input.{i}.role: must be 'user', 'assistant', 'system' "
                              f"or 'developer', got {role!r}")
        elif t == "function_call":
            if pending is None:
                pending = _Assistant()
            pending.calls.append(_function_call(i, item))
        elif t == "function_call_output":
            flush()
            out.append(_function_call_output(i, item))
        elif t == "reasoning":
            text = _reasoning_text(i, item)
            if pending is None or pending.reasoning is not None:
                flush()
                pending = _Assistant()
            pending.reasoning = text
        elif t == "item_reference":
            raise invalid(f"input.{i}: item_reference is not supported — this server "
                          "keeps no response state; send the item itself")
        else:
            raise invalid(f"input.{i}: item type {t!r} is not supported by this server "
                          "(message, function_call, function_call_output and "
                          "reasoning items only)")
    flush()
    return out, tuple(images)


def _message_text(i: int, content, veng=None) -> str:
    """message.content -> plain text: a string, or parts CONCATENATED (the
    chat boundary's join). Non-text parts are a 400 naming item and part —
    for assistant/system/developer turns, which never carry images:
    `input_image` is a user-turn part on this wire (_user_content)."""
    if isinstance(content, str):
        capability.check_injection(content, veng, f"input.{i}.content")
        return content
    if not isinstance(content, list):
        raise invalid(f"input.{i}.content: must be a string or an array of content parts")
    texts: list[str] = []
    for j, part in enumerate(content):
        if not isinstance(part, dict) or not isinstance(part.get("type"), str):
            raise invalid(f"input.{i}.content.{j}: must be a content part with a 'type'")
        pt = part["type"]
        where = f"input.{i}.content.{j}"
        if pt in ("input_text", "output_text"):
            if not isinstance(part.get("text"), str):
                raise invalid(f"{where}.text: must be a string")
            capability.check_injection(part["text"], veng, where)
            texts.append(part["text"])
        elif pt == "refusal":
            if not isinstance(part.get("refusal"), str):
                raise invalid(f"{where}.refusal: must be a string")
            texts.append(part["refusal"])
        elif pt == "input_image":
            raise invalid(f"{where}: images are only supported in a user message.")
        elif pt in ("input_file", "input_audio"):
            raise invalid(f"{where}: {pt!r} parts are not supported by this server.")
        else:
            raise invalid(f"{where}: unknown or unsupported part type {pt!r}")
    return "".join(texts)


def _user_content(i: int, content, veng=None) -> tuple:
    """A user message's content: string or parts, `input_image` accepted
    (`image_url`: data:, http(s) — fetched per veng.fetch_urls, several in
    this message concurrently — or file:// per veng.media_path; `file_id`
    always gets a 400, there is no Files API) when `veng` is given — the
    same refusal wording as _message_text when it is not, so a text-only
    engine's behaviour is unchanged. Returns (content, images): content is
    a string when there is no image, else a parts list in wire order."""
    if isinstance(content, str):
        capability.check_injection(content, veng, f"input.{i}.content")
        return content, ()
    if not isinstance(content, list):
        raise invalid(f"input.{i}.content: must be a string or an array of content parts")
    parts: list[dict] = []
    pending: list[tuple[int, str, str, object]] = []  # (parts-index, url, where, detail)
    for j, part in enumerate(content):
        if not isinstance(part, dict) or not isinstance(part.get("type"), str):
            raise invalid(f"input.{i}.content.{j}: must be a content part with a 'type'")
        pt = part["type"]
        where = f"input.{i}.content.{j}"
        if pt in ("input_text", "output_text"):
            if not isinstance(part.get("text"), str):
                raise invalid(f"{where}.text: must be a string")
            capability.check_injection(part["text"], veng, where)
            parts.append({"type": "text", "text": part["text"]})
        elif pt == "refusal":
            if not isinstance(part.get("refusal"), str):
                raise invalid(f"{where}.refusal: must be a string")
            parts.append({"type": "text", "text": part["refusal"]})
        elif pt == "input_image":
            if veng is None:
                raise _NoVision(where)
            if part.get("file_id") is not None:
                raise invalid(f"{where}: file_id images are not supported by this "
                              "server (no Files API); send "
                              f"{vision.sources_message(fetch_urls=veng.fetch_urls, media_path=veng.media_path)} "
                              "as image_url instead.")
            url = part.get("image_url")
            if not isinstance(url, str):
                raise invalid(f"{where}.image_url: must be a string")
            detail = part.get("detail")
            if detail is not None and not isinstance(detail, str):
                raise invalid(f"{where}.detail: must be a string")
            parts.append({"type": "image"})
            pending.append((len(parts) - 1, url, where, detail))
        elif pt in ("input_file", "input_audio"):
            raise invalid(f"{where}: {pt!r} parts are not supported by this server.")
        else:
            raise invalid(f"{where}: unknown or unsupported part type {pt!r}")
    if not pending:
        return "".join(p["text"] for p in parts), ()
    encs = vision.parse_image_urls([(u, w) for _, u, w, _ in pending],
                                   fetch_urls=veng.fetch_urls, media_path=veng.media_path)
    # `pending` was appended in wire order, so this list is too.
    images = [veng.prepare(enc, detail=detail, where=where)
             for (_, _, where, detail), enc in zip(pending, encs)]
    return parts, tuple(images)


def _json_or_keep(v):
    """A JSON-encoded arguments STRING -> the object it encodes; anything
    else (already an object, junk that won't parse) passes through untouched
    — http._normalize_messages' rule for the chat history, applied here."""
    if isinstance(v, str):
        try:
            return json.loads(v)
        except ValueError:
            return v
    return v


def _function_call(i: int, item: dict) -> dict:
    cid, name = item.get("call_id"), item.get("name")
    if not isinstance(cid, str) or not cid:
        raise invalid(f"input.{i}.call_id: field required")
    if not isinstance(name, str) or not name:
        raise invalid(f"input.{i}.name: field required")
    args = item.get("arguments", "{}")
    if not isinstance(args, (str, dict)):
        raise invalid(f"input.{i}.arguments: must be a JSON string")
    return {"id": cid, "type": "function",
            "function": {"name": name, "arguments": _json_or_keep(args)}}


def _function_call_output(i: int, item: dict) -> dict:
    cid = item.get("call_id")
    if not isinstance(cid, str) or not cid:
        raise invalid(f"input.{i}.call_id: field required")
    output = item.get("output")
    if output is None:
        text = ""
    elif isinstance(output, str):
        text = output
    elif isinstance(output, list):
        parts = []
        for j, part in enumerate(output):
            pt = part.get("type") if isinstance(part, dict) else None
            if pt != "input_text" or not isinstance(part.get("text"), str):
                raise invalid(f"input.{i}.output.{j}: {pt!r} parts are not supported "
                              "inside a function_call_output on this server (text only)")
            parts.append(part["text"])
        text = "".join(parts)
    else:
        raise invalid(f"input.{i}.output: must be a string or an array of input_text parts")
    return {"role": "tool", "tool_call_id": cid, "content": text}


def _reasoning_text(i: int, item: dict) -> str:
    """content[] (reasoning_text) first, else summary[] (summary_text) — the
    field this server writes. Parts join with a blank line (paragraphs)."""
    for key, ptype in (("content", "reasoning_text"), ("summary", "summary_text")):
        parts = item.get(key)
        if parts is None:
            continue
        if not isinstance(parts, list):
            raise invalid(f"input.{i}.{key}: must be an array")
        texts = []
        for j, part in enumerate(parts):
            if not isinstance(part, dict) or not isinstance(part.get("text"), str):
                raise invalid(f"input.{i}.{key}.{j}: must be a {ptype} part with a 'text'")
            texts.append(part["text"])
        if texts:
            return "\n\n".join(texts)
    return ""  # encrypted_content only, or empty: nothing readable to render


def _function_tool(where: str, t: dict) -> dict:
    """One FLAT function tool -> the OpenAI nested shape validate_tools accepts."""
    name = t.get("name")
    if not isinstance(name, str) or not name:
        raise invalid(f"{where}.name: must be a non-empty string")
    fn: dict = {"name": name}
    if t.get("description") is not None:
        if not isinstance(t["description"], str):
            raise invalid(f"{where}.description: must be a string")
        fn["description"] = t["description"]
    if t.get("parameters") is not None:
        if not isinstance(t["parameters"], dict):
            raise invalid(f"{where}.parameters: must be an object (a JSON Schema)")
        fn["parameters"] = t["parameters"]
    return {"type": "function", "function": fn}  # `strict`: accepted, ignored


def convert_tools(raw) -> tuple[list | None, dict]:
    """tools[] (Responses' FLAT function shape) -> (the OpenAI list
    validate_tools accepts or None, {function name: namespace name}).
    A hosted tool type names something this server cannot run and is refused
    by name. A `namespace` tool is a CONTAINER (Codex CLI sends one): its
    function tools are offered flat under their own names, remembered so the
    call item can carry `namespace` back; a name offered twice (in two
    namespaces, or a namespace and the top level) is refused — the model sees
    one flat name and the call could not be routed."""
    if raw is None:
        return None, {}
    if not isinstance(raw, list):
        raise invalid("tools: must be an array")
    out: list = []
    namespaces: dict = {}
    seen: dict = {}
    for i, t in enumerate(raw):
        if not isinstance(t, dict):
            raise invalid(f"tools.{i}: must be an object")
        ttype = t.get("type")
        if ttype == "namespace":
            ns = t.get("name")
            if not isinstance(ns, str) or not ns:
                raise invalid(f"tools.{i}.name: a namespace needs a non-empty name")
            inner = t.get("tools")
            if not isinstance(inner, list):
                raise invalid(f"tools.{i}.tools: a namespace's tools must be an array")
            for j, u in enumerate(inner):
                where = f"tools.{i}.tools.{j}"
                if not isinstance(u, dict):
                    raise invalid(f"{where}: must be an object")
                if u.get("type") != "function":
                    raise invalid(f"{where}: type {u.get('type')!r} inside namespace {ns!r} "
                                  "is not a function tool; only function tools are supported")
                conv = _function_tool(where, u)
                fname = conv["function"]["name"]
                if fname in seen:
                    raise invalid(f"{where}: function {fname!r} is offered twice "
                                  f"({seen[fname]} and namespace {ns!r}); one flat name "
                                  "per function, or the call cannot be routed")
                seen[fname] = f"namespace {ns!r}"
                namespaces[fname] = ns
                out.append(conv)
            continue
        if ttype != "function":
            if ttype in _HOSTED_TOOL_TYPES:
                raise invalid(f"tools.{i}: type {ttype!r} is a hosted/built-in tool, "
                              "which this server cannot run; only function tools "
                              "(name + parameters) are supported")
            raise invalid(f"tools.{i}: type {ttype!r} is not supported; only "
                          "function tools are")
        conv = _function_tool(f"tools.{i}", t)
        fname = conv["function"]["name"]
        if fname in seen:
            raise invalid(f"tools.{i}: function {fname!r} is offered twice "
                          f"({seen[fname]} and the top level); one flat name per function")
        seen[fname] = "the top level"
        out.append(conv)
    return (validate_tools(out) or None), namespaces


# ------------------------------------------------------------ response side --
# `turn` below is http.Turn — duck-typed (reasoning, content, tool_calls,
# finish, prompt_tokens, completion_tokens, cached_tokens) so this module
# never imports the HTTP layer that imports it.


def reasoning_item(rid: str, text: str, status: str = "completed") -> dict:
    return {"id": rid, "type": "reasoning",
            "summary": [{"type": "summary_text", "text": text}], "status": status}


def message_item(mid: str, text: str, status: str = "completed") -> dict:
    return {"id": mid, "type": "message", "role": "assistant", "status": status,
            "content": [{"type": "output_text", "text": text, "annotations": []}]}


def function_call_item(fid: str, call_id: str, name: str, arguments: str,
                       status: str = "completed", namespace: str | None = None) -> dict:
    item = {"id": fid, "type": "function_call", "call_id": call_id, "name": name,
            "arguments": arguments, "status": status}
    if namespace is not None:
        item["namespace"] = namespace  # the container the function was offered under
    return item


def call_arguments(call: dict) -> str:
    """The parsed call's arguments as the wire's JSON STRING."""
    return json.dumps(call.get("arguments", {}))


def output_items(reasoning: str, content: str, tool_calls: list | None,
                 namespaces: dict | None = None) -> list[dict]:
    """The items in production order: reasoning, message, function_call*.
    Empty channels produce no item."""
    items: list[dict] = []
    if reasoning:
        items.append(reasoning_item(new_reasoning_id(), reasoning))
    if content:
        items.append(message_item(new_message_id(), content))
    for c in tool_calls or []:
        items.append(function_call_item(new_function_call_id(), new_call_id(),
                                        c["name"], call_arguments(c),
                                        namespace=(namespaces or {}).get(c["name"])))
    return items


def usage(turn) -> dict:
    """ResponseUsage. input_tokens is the real prompt count (cached_tokens
    the part the prefix cache served, INCLUDED in it, as the chat dialect
    counts); reasoning_tokens is 0 — see the module docstring."""
    return {"input_tokens": turn.prompt_tokens,
            "input_tokens_details": {"cached_tokens": turn.cached_tokens,
                                     "cache_write_tokens": 0},
            "output_tokens": turn.completion_tokens,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": turn.prompt_tokens + turn.completion_tokens}


def status_of(finish: str) -> tuple[str, dict | None]:
    """Engine finish vocabulary -> (status, incomplete_details)."""
    if finish == "length":
        return "incomplete", {"reason": "max_output_tokens"}
    return "completed", None


def render_response(resp_id: str, created: int, model: str, echo: dict, *,
                    status: str, output: list[dict], usage: dict | None = None,
                    incomplete_details: dict | None = None,
                    error: dict | None = None) -> dict:
    """The Response object — one shape for the non-streamed body and every
    snapshot a lifecycle event carries (created / in_progress / completed /
    incomplete / failed)."""
    return {
        "id": resp_id, "object": "response", "created_at": created,
        "status": status, "error": error, "incomplete_details": incomplete_details,
        "model": model, "output": output, "usage": usage,
        "instructions": echo.get("instructions"),
        "max_output_tokens": echo.get("max_output_tokens"),
        "temperature": echo.get("temperature"),
        "top_p": echo.get("top_p"),
        "tools": echo.get("tools", []),
        "tool_choice": echo.get("tool_choice", "auto"),
        "text": echo.get("text", {"format": {"type": "text"}}),
        "reasoning": echo.get("reasoning"),
        "parallel_tool_calls": echo.get("parallel_tool_calls", True),
        "truncation": echo.get("truncation", "disabled"),
        "metadata": echo.get("metadata", {}),
        "store": False,  # nothing is stored, whatever was asked (module docstring)
        "previous_response_id": None,
    }


def render_turn(resp_id: str, created: int, model: str, echo: dict, turn,
                namespaces: dict | None = None) -> dict:
    """One finished generation as the non-streamed Response."""
    status, details = status_of(turn.finish)
    return render_response(resp_id, created, model, echo, status=status,
                           output=output_items(turn.reasoning, turn.content,
                                               turn.tool_calls, namespaces),
                           usage=usage(turn), incomplete_details=details)


# ---------------------------------------------------------------- streaming --


class Stream:
    """One streamed Response: sequence numbering, the lifecycle snapshots,
    and the open output item — which item is open, at what output_index,
    and the events around it. send(name, payload) -> bool is the wire (http's
    _event); False means the client is gone and is returned straight up —
    the on_delta contract, one layer out. Every item this emits is recorded
    so the final object carries the SAME ids and indices the events did."""

    def __init__(self, send, resp_id: str, created: int, model: str, echo: dict,
                 namespaces: dict | None = None):
        self._send = send
        self.resp_id, self.created, self.model, self.echo = resp_id, created, model, echo
        self.namespaces = namespaces or {}
        self.seq = 0
        self.output: list[dict] = []  # finished items, in output_index order
        self.open: str | None = None  # "reasoning" | "message" | None
        self._id = ""  # the open item's id
        self._text: list[str] = []  # the open item's text so far

    def _emit(self, payload: dict) -> bool:
        payload["sequence_number"] = self.seq
        self.seq += 1
        return self._send(payload["type"], payload)

    def _snapshot(self, status: str, **kw) -> dict:
        return render_response(self.resp_id, self.created, self.model, self.echo,
                               status=status, output=list(self.output), **kw)

    @property
    def index(self) -> int:
        """output_index of the open item (the next slot while none is open)."""
        return len(self.output)

    def start(self) -> bool:
        """response.created, then response.in_progress — the real wire's
        opening pair, announced before generation so a queued client sees
        the stream open at once."""
        snap = self._snapshot("in_progress")
        return (self._emit({"type": "response.created", "response": snap})
                and self._emit({"type": "response.in_progress", "response": dict(snap)}))

    def delta(self, channel: str, text: str) -> bool:
        """One piece from the core's split: 'reasoning' -> a reasoning item's
        summary text, 'content' -> a message item's output_text; a channel
        change closes the open item and opens the next output_index."""
        kind = "reasoning" if channel == "reasoning" else "message"
        if self.open != kind:
            if not self.close():
                return False
            if not self._open(kind):
                return False
        self._text.append(text)
        i, iid = self.index, self._id
        if kind == "reasoning":
            return self._emit({"type": "response.reasoning_summary_text.delta",
                               "item_id": iid, "output_index": i, "summary_index": 0,
                               "delta": text})
        return self._emit({"type": "response.output_text.delta", "item_id": iid,
                           "output_index": i, "content_index": 0, "delta": text,
                           "logprobs": []})

    def _open(self, kind: str) -> bool:
        self.open, self._text = kind, []
        i = self.index
        if kind == "reasoning":
            self._id = new_reasoning_id()
            item = {"id": self._id, "type": "reasoning", "summary": [],
                    "status": "in_progress"}
            return (self._emit({"type": "response.output_item.added", "output_index": i,
                                "item": item})
                    and self._emit({"type": "response.reasoning_summary_part.added",
                                    "item_id": self._id, "output_index": i,
                                    "summary_index": 0,
                                    "part": {"type": "summary_text", "text": ""}}))
        self._id = new_message_id()
        item = {"id": self._id, "type": "message", "role": "assistant",
                "status": "in_progress", "content": []}
        return (self._emit({"type": "response.output_item.added", "output_index": i,
                            "item": item})
                and self._emit({"type": "response.content_part.added", "item_id": self._id,
                                "output_index": i, "content_index": 0,
                                "part": {"type": "output_text", "text": "",
                                         "annotations": []}}))

    def close(self) -> bool:
        """Finish the open item: its text.done / part.done, then
        output_item.done carrying the completed item, which is then
        recorded at its output_index."""
        if self.open is None:
            return True
        kind, i, iid, text = self.open, self.index, self._id, "".join(self._text)
        self.open, self._text = None, []
        if kind == "reasoning":
            item = reasoning_item(iid, text)
            part = {"type": "summary_text", "text": text}
            ok = (self._emit({"type": "response.reasoning_summary_text.done",
                              "item_id": iid, "output_index": i, "summary_index": 0,
                              "text": text})
                  and self._emit({"type": "response.reasoning_summary_part.done",
                                  "item_id": iid, "output_index": i, "summary_index": 0,
                                  "part": part}))
        else:
            item = message_item(iid, text)
            ok = (self._emit({"type": "response.output_text.done", "item_id": iid,
                              "output_index": i, "content_index": 0, "text": text,
                              "logprobs": []})
                  and self._emit({"type": "response.content_part.done", "item_id": iid,
                                  "output_index": i, "content_index": 0,
                                  "part": item["content"][0]}))
        self.output.append(item)
        return ok and self._emit({"type": "response.output_item.done", "output_index": i,
                                  "item": item})

    def function_call(self, call: dict) -> bool:
        """One complete parsed call as a function_call item: added (arguments
        ""), ONE arguments.delta with the whole serialized string,
        arguments.done, output_item.done."""
        if not self.close():
            return False
        i = self.index
        fid, cid, args = new_function_call_id(), new_call_id(), call_arguments(call)
        ns = self.namespaces.get(call["name"])
        item = function_call_item(fid, cid, call["name"], args, namespace=ns)
        self.output.append(item)
        return (self._emit({"type": "response.output_item.added", "output_index": i,
                            "item": function_call_item(fid, cid, call["name"], "",
                                                       "in_progress", namespace=ns)})
                and self._emit({"type": "response.function_call_arguments.delta",
                                "item_id": fid, "output_index": i, "delta": args})
                and self._emit({"type": "response.function_call_arguments.done",
                                "item_id": fid, "output_index": i, "arguments": args})
                and self._emit({"type": "response.output_item.done", "output_index": i,
                                "item": item}))

    def finish(self, turn) -> bool:
        """Generation over: close what is open, the calls, then
        response.completed (or response.incomplete) with the full object."""
        if not self.close():
            return False
        for call in turn.tool_calls or []:
            if not self.function_call(call):
                return False
        status, details = status_of(turn.finish)
        snap = self._snapshot(status, usage=usage(turn), incomplete_details=details)
        return self._emit({"type": f"response.{status}", "response": snap})

    def fail(self, code: str | None, message: str, param: str | None = None) -> bool:
        """Error mid-stream: the `error` event (what the SDKs raise on), then
        response.failed carrying the object with its error filled in. The
        open item, if any, is left as it was — the failure is the news."""
        ok = self._emit({"type": "error", "code": code, "message": message, "param": param})
        snap = self._snapshot("failed", error={"code": code or "server_error",
                                               "message": message})
        return ok and self._emit({"type": "response.failed", "response": snap})
