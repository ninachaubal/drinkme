"""OpenAI-compatible HTTP layer over an Engine — stdlib only, one file.

http.server, not fastapi/uvicorn: serve is THE artifact and the posture is
only-our-authored code — zero new deps, and the whole protocol surface stays
a short read: this file is transport, lock and the generation core; the three
wire dialects are translation (OpenAI chat completions inline below, the
OpenAI Responses API in serving/responses.py, Anthropic in
serving/messages.py). Everything model-shaped goes through the Engine seam
(engine.py); this file never touches weights, tokenizers, or torch.

Routes:
  GET  /health                    {"ok":true,...}; NEVER takes the lock; never gated
  GET  /v1/models                 model list + pack provenance; NEVER takes the lock
  GET  /metrics                   Prometheus text exposition (serving/metrics.py);
                                  NEVER takes the lock; never gated, same posture as
                                  /health — a scraper should not need the bearer
  POST /v1/chat/completions       chat.completion, or SSE chunks with "stream": true
  POST /v1/responses              OpenAI Responses API: a response object, or the
                                  named SSE event sequence with "stream": true
  POST /v1/messages               Anthropic Messages API: a message, or the named
                                  SSE event sequence with "stream": true
  POST /v1/messages/count_tokens  {"input_tokens": N}; NEVER takes the lock
  GET  /tokenizer_info            eos/bos ids, chat_template presence, thinking/
                                  tool_format; NEVER takes the lock (the tokenizer routes)
  POST /tokenize                  {"prompt"|"messages"} -> {count, tokens,
                                  max_model_len}; NEVER takes the lock (the tokenizer routes)
  POST /detokenize                {"tokens": [...]} -> {"prompt": str};
                                  NEVER takes the lock (the tokenizer routes)
  POST /sleep[?level=1|2]         park the model, give the device back (sleep/wake);
                                  takes the lock NON-blockingly -> 409 if a
                                  generation is in flight
  POST /wake_up                   bring it back; same non-blocking lock
Errors are OpenAI-shaped: {"error": {"message", "type", "code"}} — except
under /v1/messages, where they wear Anthropic's envelope
{"type": "error", "error": {"type", "message"}} with Anthropic's codes.
/tokenize, /detokenize and /tokenizer_info always answer OpenAI-shaped —
they are not under /v1/messages and have no dialect of their own.

ONE generation core (_generate): every dialect builds the same value
(engine.GenerationRequest: messages, SampleParams, tools, template kwargs,
stream, the wire id) and hands it over; the core runs the engine under the
lock and splits the stream into reasoning | content | tool calls, and each
dialect only RENDERS the result. The parity test in
tests/test_serving_messages.py holds the three surfaces to the same engine
transcript for the same conversation.

CORS is wide open (`Access-Control-Allow-Origin: *`) — a local inference
server that browser tools can't reach is a worse posture than one they can;
the bearer, not the origin, is the gate. Auth is OPTIONAL: pass
auth_token to start_server (CLI --auth / DRINKME_AUTH_TOKEN) and every /v1/*
request must carry `Authorization: Bearer <token>` or `x-api-key: <token>`
(the Anthropic SDKs' header; honored on every route, not just theirs);
/health stays open so probes don't need the secret.

Concurrency v1: ONE global generation lock. The kernel is batch-1 on a
unified-memory machine — two live generations would interleave weight reads and
both would lose bandwidth, so queueing in arrival order is strictly better
while decode is not batched.
/v1/models, count_tokens and every error path answer without touching the
lock. There is no "busy" refusal (no 429/529): a second request WAITS.

Client disconnect aborts the generation: SSE writes go through _sse() /
_event(), which turn a dead socket into `return False`, and that False IS
the on_delta signal the Engine contract requires generation to stop on
(serve.py: backpressure is a signal, not something to buffer). Because the
stream's 200 status line already went out before the abort happened, the
automatic access log reads clean either way — _generate() logs a distinct
"stream aborted" line (dialect, tokens delivered, elapsed) so a disconnect
never looks like 20 ordinary completions in the journal.

Logging: every 4xx gets ONE line — method, path, status, the error message,
truncated, never the request body or the bearer token — from _json's
chokepoint (_error and _aerror both funnel there) or, mid-stream where no
status line is left to send, from the two exception handlers directly.

The think channel (serving/think.py) lives HERE, not in the engines: a
reasoning model's stream is `reasoning </think> answer`, and splitting it is
wire-shaping, the same as tool_calls rendering. Which channel byte one
belongs to is decided by ONE thing — did the prompt end inside an open
`<think>`, which the engine reads off the prompt it just rendered and
reports as its StreamStart event (engine.py) — never by the model's name
and never by what the request asked for. The TAGS are the
model's tool_formats row's (`<think>`/`</think>` by default, gemma's
`<|channel>thought`/`<channel|>`, Muse-Glimmer's messages addressed
`to=self`), looked up by the announced tool_format.
Reasoning leaves as `reasoning_content` (OpenAI) or a `thinking` block
(Anthropic), the answer as `content` / a `text` block, in both modes.

Off-menu capability honesty (serving/capability.py): every engine
carries a `thinking`/`tool_format` announcement, probed once from its chat
template at construction and surfaced on `/v1/models` (`drinkme.capabilities`).
`tool_format` is a
ROW NAME from serving/tool_formats.py's table (docs/serve-tool-formats.md
— json, qwen-xml, gemma, atem, glm, minimax, mistral, kimi-k2). A `tools` request
that would actually be offered to a model whose template matched no row
(`"unknown"` or `"none"`), or matched a row outside the TESTED tier
(menu models stay the tested tier: tool_formats.TESTED), is refused here,
all three dialects, before a token generates — refuse loudly beats an unparsed or
untested dialect leaking into `content` as visible text. Adding a family
moves that line by exactly one row, with no edit in this file.

Vision (serving/vision.py) is a DIFFERENT kind of
capability: a structural fact about the served tree (does it carry a ViT
and a preprocessor for this checkpoint?), not something probed from the
chat template, so it lives on the engine's OPTIONAL `.vision` surface
(capability.engine_vision — discovered by getattr, exactly like
sleep_state/prefix_cache_state, not part of the Engine Protocol) rather
than on capability.Capability. `drinkme.capabilities.vision` (true/false)
and `.imageInput` still ride on `/v1/models` beside thinking/toolFormat,
assembled here. Every dialect's image part is refused by name — capability
.vision_refusal_message — when the engine has none; accepted parts decode
outside the generation lock (serving/vision.py) into
GenerationRequest.images, in template order. An `image_url` that names an
http(s) URL is DOWNLOADED by default (llama.cpp parity) — several in
one turn concurrently, via vision.parse_image_urls — with no address
filtering (this server runs at the operator's own network position, as
llama.cpp's does); `--no-image-urls` / DRINKME_IMAGE_URLS=0 turns fetching off for
an operator who exposes the server to other machines.

Video (serving/video.py) rides on the same surface: `video_url` parts
(vLLM's shape, Chat Completions only — the Responses and Messages APIs
define no video part) follow the image sources' rules and switches, decode
outside the lock into GenerationRequest.videos, and are refused by name —
capability.video_refusal_message — on an engine whose Vision has no
`.video` (no video processor, PyAV missing) or that has no vision at all.
`drinkme.capabilities.video` and `.videoInput` ride beside `vision`.

Advertised context (borrowed from oMLX): `advertised_ctx`, off by default
(None = the default, real numbers everywhere). When set below the
engine's real `contextWindow`, `/v1/messages`' `usage.input_tokens` and
`/v1/messages/count_tokens` are SCALED UP by real/advertised (_ctx_scale
below) so Claude Code's own auto-compact — which paces off the token counts
this server reports, not off `/v1/models`' `drinkme.contextWindow`, the one
other clients (pi) DO honor and which therefore stays real — fires before
the real window actually runs out. The lie is scoped tight: the OpenAI
dialect, the engine's own accounting (GenResult, `res.prompt_tokens`), the
`/metrics` numbers above, and the bench all see real counts always;
docs/serve.md names exactly which field is fictional.

SSE keep-alive (borrowed from oMLX): a streamed request can sit silent for
a long time before its first delta — queued behind another generation
(gen_lock) or inside the model's own prefill on a big cold prompt — and a
client's read timeout does not know the difference between "stuck" and
"about to answer". `_keepalive_start` writes a `: keep-alive` SSE comment
frame every DRINKME_SSE_KEEPALIVE_S seconds (default 10; 0 disables) until
the first real delta arrives; a comment line is legal SSE (WHATWG "Server-
Sent Events", Interpreting an event stream: "If the line starts with a
U+003A COLON character (:) ... Ignore the line.") and vLLM ships the exact
same 15-byte frame for the same reason (entrypoints/openai/sse_keep_alive.py:
"A non-positive or non-finite `interval` returns `generator` unchanged").
Every write on a stream — a real frame or a keep-alive tick — goes through
one lock (`_sse_lock`) so the two can race but never interleave into one
corrupted frame.

SLEEP / WAKE (vLLM borrow 2 — vLLM's docs/features/sleep_mode.md: "temporarily
release most GPU memory used by a model, including model weights and KV cache,
without stopping the server"): `POST /sleep?level=1` parks the weights in host
RAM and drops the KV; `POST /wake_up` puts them back. While asleep, /health
says so (`state`, `level`), the LOCKLESS routes keep answering — /v1/models,
/tokenize, /detokenize, /tokenizer_info and count_tokens read the tokenizer,
not the weights — and both generate routes refuse with 503 `model_asleep`
naming the wake URL. Both dialects: OpenAI gets code "model_asleep", Anthropic
gets a `model_asleep` error type rather than `overloaded_error`, which would
drive an SDK's retry loop against a server that will not wake on its own.
The two routes take gen_lock NON-BLOCKINGLY and answer 409 when they cannot:
a sleep that queued behind a generation would be a sleep whose caller has no
idea when it happened, and serving/sleep.py's pass must never run beside a
live forward. DRINKME_SLEEP_ON_IDLE_S (serve.py, off by default) drives the
same route from _IdleSleeper below.

Request-time context-length check: `_generate` computes the prompt's
token count the same way `/tokenize` does (Engine.tokenize, hence
Engine.count_tokens — same call generate() itself makes to build the
prompt) BEFORE taking gen_lock, and refuses with a legible 400 naming the
model and all three numbers when prompt_tokens + max_tokens would exceed
the engine's ctx — turning the default silent empty answer (engines.py's
`max_new <= 0` early return, finish_reason "length") into an answer that
says why. `serve.py`'s `_announce_attention` docstring has the related
failure (a 25,057-token prompt, "Tried to allocate 56.13 GiB") — that one
is the O(T^2) math-SDPA fallback, a different bug
`sdpa.py` diagnoses separately; this check catches the general case, a
prompt that will not fit the window at all.
"""

from __future__ import annotations

import dataclasses
import json
import os
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from . import capability, gen_config
from . import messages as anthropic
from . import metrics
from . import generation_profiles
from . import responses
from . import video
from . import vision
from .engine import (Delta, Engine, Finished, GenerationRequest, SampleParamError, SleepError,
                     StreamStart, validated_sample_params)
from .template import RESERVED_KWARGS
from . import think
from .tools import to_openai_calls, validate_tools


# OpenAI-dialect (status) -> Anthropic error type, for the shared paths (auth,
# unknown route, bad JSON) that answer under /v1/messages
_ANTHROPIC_TYPE = {400: "invalid_request_error", 401: "authentication_error",
                   404: "not_found_error", 500: "api_error"}

# requests_total's route label: the known surface, bounded — an unrecognized
# path (a 404 probe, a typo) buckets to "other" rather than growing a new
# label value per garbage request (metrics.py's whole zero-cost posture).
_KNOWN_ROUTES = frozenset({"/health", "/metrics", "/v1/models",
                           "/v1/chat/completions", "/v1/responses", "/v1/messages",
                           "/v1/messages/count_tokens",
                           "/tokenize", "/detokenize", "/tokenizer_info",
                           "/sleep", "/wake_up"})


def _route_label(path: str) -> str:
    return path if path in _KNOWN_ROUTES else "other"


# The SSE keep-alive's prefill window: seconds between comment frames. 0 disables.
_KEEPALIVE_ENV = "DRINKME_SSE_KEEPALIVE_S"
_KEEPALIVE_DEFAULT_SEC = 10.0


def _keepalive_interval() -> float:
    """Read fresh per stream (not cached at server start) so a test — or an
    operator — can change it between requests. Garbage or unset falls back
    to the default rather than silently going quiet, same posture as
    engines.py's env-var readers."""
    raw = os.environ.get(_KEEPALIVE_ENV)
    if not raw:
        return _KEEPALIVE_DEFAULT_SEC
    try:
        return float(raw)
    except ValueError:
        return _KEEPALIVE_DEFAULT_SEC


_CLAMP_ENV = "DRINKME_MAX_TOKENS_CLAMP"


def _clamp_max_tokens() -> bool:
    """The max_tokens clamp: when prompt_tokens + max_tokens would exceed the window but the
    prompt itself fits, clamp max_tokens to the room left (default) instead
    of refusing the request. Found testing against a real Claude Code session:
    Claude Code sends max_tokens 32000 on every /v1/messages call, so a 26k prompt
    on a 40,960-token model 400'd on turn 2 and the session died — while the
    same request against llama.cpp or Ollama simply generates until the
    window is full and reports the length stop. That is the behaviour the
    local-model clients are written against; the OpenAI/Anthropic-style
    refusal is one env var away (DRINKME_MAX_TOKENS_CLAMP=0). Read fresh per
    request, like the keep-alive interval, so a test can flip it."""
    return os.environ.get(_CLAMP_ENV, "1").strip() != "0"


# the request-body cap. Justified against the largest legitimate request
# this server is meant to take, not the largest one a client could send:
# 262,144 tokens (this project's largest advertised context) at a generous
# ~4 bytes/token for JSON-escaped text is ~1 MiB; a handful of embedded
# base64 images (each ~4/3 its raw size) can easily add tens of MiB more.
# 128 MiB comfortably covers a 262k-token prompt PLUS several multi-MiB
# images with headroom to spare, while still bounding what an unauthenticated
# or misbehaving client can make a thread allocate for.
_BODY_CAP_ENV = "DRINKME_MAX_BODY_BYTES"
_BODY_CAP_DEFAULT = 128 * 1024 * 1024  # 128 MiB


def _max_body_bytes() -> int:
    raw = os.environ.get(_BODY_CAP_ENV)
    if not raw:
        return _BODY_CAP_DEFAULT
    try:
        n = int(raw)
    except ValueError:
        return _BODY_CAP_DEFAULT
    return n if n > 0 else _BODY_CAP_DEFAULT


# the read deadline covering header + body — a client that goes silent
# mid-request (no bytes for this many seconds) gets its connection closed
# and the thread returned, rather than holding it forever. Armed fresh at
# the start of EVERY request (handle_one_request below, not just once at
# connection setup — a keep-alive connection serves many requests) and
# CLEARED the instant _body() finishes (success or BodyTooLarge): a
# slow-READING client during the response that follows — SSE keep-alive, a
# long stream — must never trip a SEND timeout meant for the read phase.
_READ_TIMEOUT_ENV = "DRINKME_READ_TIMEOUT_S"
_READ_TIMEOUT_DEFAULT_SEC = 30.0


def _read_timeout_s() -> float:
    raw = os.environ.get(_READ_TIMEOUT_ENV)
    if not raw:
        return _READ_TIMEOUT_DEFAULT_SEC
    try:
        v = float(raw)
    except ValueError:
        return _READ_TIMEOUT_DEFAULT_SEC
    return v if v > 0 else _READ_TIMEOUT_DEFAULT_SEC


class BodyTooLarge(Exception):
    """Raised by _body when Content-Length exceeds DRINKME_MAX_BODY_BYTES
    — before a single byte of the body is read. Every POST route catches
    this through _body_or_413, which renders it 413 in the request's own
    dialect envelope and closes the connection: nothing here ever drains the
    declared body, and closing is what makes that safe — whatever the peer
    still has queued is simply discarded with the socket."""

    def __init__(self, content_length: int, cap: int):
        self.content_length, self.cap = content_length, cap
        super().__init__(f"request body of {content_length} bytes exceeds "
                         f"the {cap}-byte limit ({_BODY_CAP_ENV})")


class ModelAsleep(Exception):
    """Raised by _generate (sleep/wake) when the engine is parked — before gen_lock
    on the fast path, and again INSIDE it, because a request can pass the
    lockless check while a sleep is waiting for the lock and would otherwise
    reach a model whose weights are on the host. ONE exception for both
    dialects, rendered by do_POST and _post_messages in their own envelopes,
    exactly as ContextLengthExceeded is."""

    def __init__(self, state: dict, wake_url: str):
        self.state, self.wake_url = state, wake_url
        level = state.get("level")
        super().__init__(
            f"the model is asleep (level {level}) and is not serving "
            f"generations. Wake it with: POST {wake_url}")


class ContextLengthExceeded(Exception):
    """Raised by _generate (the context-length check), before gen_lock, before any forward pass:
    prompt_tokens + max_tokens would exceed the engine's allocated ctx. ONE
    exception for every dialect — do_POST, _post_responses and _post_messages
    each render it in their own envelope, exactly as they already do for
    MessagesError / the generic Exception fallback.

    A client that classifies overflow by matching its own dialect's
    canonical wording (pi-ai's OVERFLOW_PATTERNS, Claude Code's Anthropic
    match) never recognizes drinkme's own prose and never re-compacts. Every
    dialect's message opens with THAT provider's canonical sentence — the
    thing the client's regex actually looks for — then drinkme's own
    sentence with our exact N/M/C numbers. openai_message serves
    /v1/chat/completions and /v1/responses (both OpenAI-family); Anthropic's
    /v1/messages gets anthropic_message."""

    def __init__(self, prompt_tokens: int, max_tokens: int, ctx: int, model_id: str,
                 *, prompt_only: bool = False):
        self.prompt_tokens, self.max_tokens, self.ctx, self.model_id = (
            prompt_tokens, max_tokens, ctx, model_id)
        self.prompt_only = prompt_only
        completion = 0 if prompt_only else max_tokens
        total = prompt_tokens + completion
        if prompt_only:
            # The max_tokens clamp has nothing to work with: the prompt alone
            # fills the window, so there is no room to generate even one token.
            ours = (f"drinkme: {prompt_tokens} prompt tokens fill this server's "
                    f"context window of {ctx} tokens — no room to generate. "
                    "Shorten the prompt.")
        else:
            ours = (f"drinkme: {prompt_tokens} prompt tokens + {max_tokens} "
                    f"max_tokens = {prompt_tokens + max_tokens}, which exceeds "
                    f"this server's context window of {ctx} tokens. Lower "
                    "max_tokens or shorten the prompt.")
        # OpenAI's own chat.completions wording, matched by pi's
        # `/maximum context length is \d+ tokens/i` and
        # `/reduce the length of the messages/i`, and by OpenAI SDKs on
        # `code == "context_length_exceeded"` (set by the caller, not here).
        self.openai_message = (
            f"This model's maximum context length is {ctx} tokens. However, "
            f"you requested {total} tokens ({prompt_tokens} in the messages, "
            f"{completion} in the completion). Please reduce the length of "
            f"the messages or completion. {ours}")
        # Anthropic's own /v1/messages wording, matched by pi's
        # `/prompt is too long/i` and by Claude Code.
        self.anthropic_message = f"prompt is too long: {total} tokens > {ctx} maximum. {ours}"
        super().__init__(self.openai_message)


@dataclass
class Turn:
    """One finished generation, split into its wire-neutral channels. Both
    dialects render from this and nothing else: OpenAI makes
    reasoning_content / content / tool_calls of it, Anthropic makes
    thinking / text / tool_use blocks. `finish` is the ENGINE's vocabulary
    (stop | length | abort | tool_calls); each dialect maps it."""

    reasoning: str
    content: str
    tool_calls: list | None
    finish: str
    stop_sequence: str | None
    prompt_tokens: int
    completion_tokens: int
    cached_tokens: int


def _template_refusal(e: BaseException) -> bool:
    """Did the chat template itself refuse the request? Qwen's templates
    raise_exception() on an unexpected role or reasoning_effort (Qwen3.8's
    accepts exactly xhigh/medium/low; Claude Code can send high or max) —
    a request-shape problem with a legible message, i.e. a 400, not a 500.
    Decided by the exception's package, so this file still imports no jinja."""
    return type(e).__module__.split(".")[0] == "jinja2"


def _usage(res) -> dict:
    """OpenAI usage object. prompt_tokens_details.cached_tokens is the spec's
    field for prefix-cache reuse (engines.py) — clients that meter real work
    (pi's gauge) can subtract it; 0 when nothing was reused."""
    return {"prompt_tokens": res.prompt_tokens,
            "completion_tokens": res.completion_tokens,
            "total_tokens": res.prompt_tokens + res.completion_tokens,
            "prompt_tokens_details": {"cached_tokens": res.cached_tokens}}


def _json_or_keep(v):
    """A JSON-encoded arguments STRING -> the object it encodes; anything
    else (already a dict, junk that won't parse) passes through untouched."""
    if isinstance(v, str):
        try:
            return json.loads(v)
        except ValueError:
            return v
    return v


def _truncate(s: str, limit: int = 300) -> str:
    """Never let one error message blow out a log line."""
    return s if len(s) <= limit else s[:limit] + "…[truncated]"


# OpenAI content-part types this server does not serve, named so the 400
# says what was sent: the request is refused, never answered with the
# attachment quietly dropped.
# image_url and video_url are handled separately below: whether they are
# refused depends on the engine's vision (and video) capability, not a
# fixed list.
_UNSUPPORTED_PART_TYPES = ("input_audio", "file")


def _unsupported_part_message(where: str, t: str) -> str:
    return f"{where}: {t!r} content parts are not supported by this server."


def _validate_content_parts(i: int, content: list, *, role, eng: Engine) -> str | None:
    """The 400 message for the first bad part of messages[i].content, or
    None: every part is an object with a string `type`; a `text` part
    carries a string `text`; an `image_url` part is refused by the
    model's vision capability (capability.engine_vision) when there is
    none, by role when the turn is a system message, or by shape otherwise;
    a `video_url` part the same way by the video capability
    (capability.video_reason); a part
    of another modality is refused by name; an unknown type is refused as
    unknown. Runs BEFORE _content_text flattens, so nothing is discarded
    on the way to a 200."""
    veng = capability.engine_vision(eng)
    for j, part in enumerate(content):
        where = f"messages[{i}].content[{j}]"
        if not isinstance(part, dict) or not isinstance(part.get("type"), str):
            return f"{where} must be a content part object with a string 'type'."
        t = part["type"]
        if t == "text":
            if not isinstance(part.get("text"), str):
                return f"{where}.text must be a string."
        elif t == "image_url":
            if veng is None:
                return capability.vision_refusal_message(
                    where, eng.model_id, capability.vision_reason(eng))
            if role == "system":
                return f"{where}: images are not supported in a system message."
            iu = part.get("image_url")
            if not isinstance(iu, dict) or not isinstance(iu.get("url"), str):
                return f"{where}.image_url.url must be a string."
            detail = iu.get("detail")
            if detail is not None and not isinstance(detail, str):
                return f"{where}.image_url.detail must be a string."
        elif t == "video_url":
            if veng is None or getattr(veng, "video", None) is None:
                return capability.video_refusal_message(
                    where, eng.model_id, capability.video_reason(eng), images=veng is not None)
            if role == "system":
                return f"{where}: videos are not supported in a system message."
            vu = part.get("video_url")
            if not isinstance(vu, dict) or not isinstance(vu.get("url"), str):
                return f"{where}.video_url.url must be a string."
        elif t in _UNSUPPORTED_PART_TYPES:
            return _unsupported_part_message(where, t)
        else:
            return f"{where}: unknown content part type {t!r}."
    return None


def _content_text(content) -> str:
    """OpenAI message content -> plain text. Strings pass through; parts
    arrays (validated by _validate_content_parts first: text parts only,
    each with a string `text`) are joined; null (assistant tool-call
    turns) -> "". A part the validator would have refused raises rather
    than being dropped — a route that skipped validation gets a JSON 400
    from its handler, never a 200 over a silently emptied turn."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out = []
        for j, p in enumerate(content):
            if not isinstance(p, dict) or p.get("type") != "text" or not isinstance(p.get("text"), str):
                raise ValueError(f"content[{j}] is not a text part with a string 'text'")
            out.append(p["text"])
        return "".join(out)
    if content is None:
        return ""
    raise ValueError("content must be a string, an array of content parts, or null")


def _has_image_part(content) -> bool:
    """Does the parts list carry an image or a video?"""
    return isinstance(content, list) and any(
        isinstance(p, dict) and p.get("type") in ("image_url", "video_url") for p in content)


def _extract_image_parts(i: int, content: list,
                         veng: "vision.Vision") -> tuple[list, list, list]:
    """A content-parts array already shape-validated by
    _validate_content_parts (so `veng` is known non-None, its `.video` too
    when a `video_url` is present, and every media part's shape is sound)
    -> (parts in wire order, the PreparedImages and the PreparedVideos
    decoded from it). Text parts run through the injection guard here;
    `_content_text` never sees an image- or video-bearing turn. Every
    `image_url`'s DOWNLOAD (http(s) URLs, per veng.fetch_urls/media_path)
    is batched through vision.parse_image_urls, so several images in one
    turn fetch concurrently, and every `video_url`'s through
    video.parse_video_urls; decoding (CPU-bound) then runs per image and
    per video, in wire order, same as always."""
    parts: list[dict] = []
    pending: list[tuple[int, str, str, object]] = []  # (parts-index, url, where, detail)
    pending_v: list[tuple[str, str]] = []  # (url, where)
    for j, p in enumerate(content):
        where = f"messages[{i}].content[{j}]"
        if p["type"] == "text":
            text = p["text"]
            capability.check_injection(text, veng, where)
            parts.append({"type": "text", "text": text})
        elif p["type"] == "video_url":
            parts.append({"type": "video"})
            pending_v.append((p["video_url"]["url"], where))
        else:  # image_url
            iu = p["image_url"]
            parts.append({"type": "image"})
            pending.append((len(parts) - 1, iu["url"], where, iu.get("detail")))
    encs = vision.parse_image_urls([(u, w) for _, u, w, _ in pending],
                                   fetch_urls=veng.fetch_urls, media_path=veng.media_path)
    images = [veng.prepare(enc, detail=detail, where=where)
             for (_, _, where, detail), enc in zip(pending, encs)]
    videos = []
    if pending_v:
        vencs = video.parse_video_urls(pending_v, fetch_urls=veng.fetch_urls,
                                       media_path=veng.media_path)
        videos = [veng.video.prepare(enc, where=where)
                  for (_, where), enc in zip(pending_v, vencs)]
    return parts, images, videos


def _normalize_messages(messages: list, *,
                        eng: Engine | None = None) -> tuple[list, tuple, tuple]:
    """The wire-shape fixes every /v1/chat/completions request gets before
    it reaches a template (do_POST's boundary comments carry the full
    reasoning for each): content flattened to plain text (parts arrays,
    null) or, for a turn carrying images or videos, kept as a parts list
    in wire order with each image_url part decoded and replaced by
    `{"type": "image"}` and each video_url part by `{"type": "video"}`
    (engine.GenerationRequest's docstring); history
    tool_calls.arguments parsed JSON-string -> dict; and OpenAI's
    "developer" role folded to "system". /tokenize (one of the tokenizer
    routes) applies the SAME fixes for the SAME reason count_tokens must
    never drift from what generate() would actually see for one
    conversation. Returns (messages, images, videos) — each in template
    order, empty when the request carries none. Raises vision.ImageError
    (video.VideoError for a video) for media that fails to decode
    (fetch-off, bad format, oversize, ...) and for the injection guard;
    callers give it the error's own `code`."""
    veng = capability.engine_vision(eng) if eng is not None else None
    out = []
    images: list = []
    videos: list = []
    for i, m in enumerate(messages):
        if not isinstance(m, dict):
            out.append(m)
            continue
        content = m.get("content")
        if _has_image_part(content):
            parts, imgs, vids = _extract_image_parts(i, content, veng)
            images.extend(imgs)
            videos.extend(vids)
            m = dict(m, content=parts)
        else:
            text = _content_text(content)
            capability.check_injection(text, veng, f"messages[{i}].content")
            m = dict(m, content=text)
        if isinstance(m.get("tool_calls"), list):
            m["tool_calls"] = [
                dict(tc, function=dict(fn, arguments=_json_or_keep(fn.get("arguments"))))
                if isinstance(tc, dict) and isinstance((fn := tc.get("function")), dict)
                else tc
                for tc in m["tool_calls"]]
        out.append(m)
    out = [dict(m, role="system") if isinstance(m, dict) and m.get("role") == "developer"
           else m for m in out]
    return out, tuple(images), tuple(videos)


def _validate_chat_messages(messages: list, *, eng: Engine) -> str | None:
    """None if every message is shape-safe for _normalize_messages/generate;
    else the 400 message naming the first bad one — the message, its
    content, and (a parts array) every part: _validate_content_parts. Also
    refuses a request over MAX_IMAGES images (vision.check_image_count) or
    MAX_VIDEOS videos (video.check_video_count), before any of them decode. messages.py's Anthropic parser and
    responses.py's input parser already check this on the way in; this
    dialect once trusted the shape, and `messages: [5]` and `content: {}`
    probes went straight through to a bare AttributeError deep in
    count_tokens (FakeEngine and every real engine index messages by dict
    access) or a silently-emptied turn, and
    `text: 7` dropped the connection while an
    image-only turn was answered 200 with the image discarded."""
    n_images = n_videos = 0
    for i, m in enumerate(messages):
        if not isinstance(m, dict):
            return f"messages[{i}] must be an object."
        content = m.get("content")
        if content is not None and not isinstance(content, (str, list)):
            return (f"messages[{i}].content must be a string, an array of "
                    "content parts, or null.")
        if isinstance(content, list):
            bad = _validate_content_parts(i, content, role=m.get("role"), eng=eng)
            if bad:
                return bad
            n_images += sum(1 for p in content
                            if isinstance(p, dict) and p.get("type") == "image_url")
            n_videos += sum(1 for p in content
                            if isinstance(p, dict) and p.get("type") == "video_url")
    try:
        if n_images:
            vision.check_image_count(n_images, where="messages")
        if n_videos:
            video.check_video_count(n_videos, where="messages")
    except vision.ImageError as e:
        return str(e)
    return None


class DrinkmeHTTPServer(ThreadingHTTPServer):
    daemon_threads = True  # a stuck client must not block process exit

    def __init__(self, addr: tuple[str, int], engine: Engine,
                 auth_token: str | None = None, advertised_ctx: int | None = None,
                 served_names: list[str] | None = None,
                 generation_profiles: dict | None = None):
        super().__init__(addr, _Handler)
        self.engine = engine
        self.auth_token = auth_token or None  # "" from an unset env = no auth
        self.advertised_ctx = advertised_ctx or None  # advertised context; None = real ctx, unscaled
        self.gen_lock = threading.Lock()  # the batch-1 policy (module docstring)
        self.started = int(time.time())  # /v1/models "created"; stable per process
        # Aliases and generation profiles: every id this server answers to
        # (engine.model_id ALWAYS among them — an alias only ADDS names, the
        # default id never stops working) and the named sampling/template-kwarg
        # overlays exposed beside them as `<id>:<profile>`.
        self.known_ids = {engine.model_id, *(served_names or ())}
        self.generation_profiles = dict(generation_profiles or {})
        # Sleep/wake: monotonic stamp of the last GENERATION activity, which is what
        # "idle" means for sleep-on-idle. /health, /v1/models and a scraper
        # hitting /metrics every 15s deliberately do NOT count — a machine that
        # never sleeps because Prometheus is polite would be a timer that
        # only ever fires when nobody is watching.
        self.last_generation = time.monotonic()
        self.idle_sleeper = None

    def mark_generation(self) -> None:
        self.last_generation = time.monotonic()

    def sleep_state(self) -> dict:
        """The engine's sleep bookkeeping, or the awake default for an engine
        that does not implement sleep/wake at all. getattr, not a method call — the
        Engine Protocol's optional surface (engine.py), same posture serve.py
        takes for persist_slots."""
        fn = getattr(self.engine, "sleep_state", None)
        if fn is None:
            return {"state": "awake", "level": 0}
        return fn()

    def prefix_cache_state(self) -> dict:
        """The engine's prefix-cache bookkeeping for /health, or `slots: None`
        for an engine with no concept of one (MLX, FakeEngine). Same getattr
        posture as sleep_state() — the Engine Protocol's optional surface."""
        fn = getattr(self.engine, "prefix_cache_state", None)
        if fn is None:
            return {"slots": None}
        return fn()

    def shutdown(self):
        if self.idle_sleeper is not None:
            self.idle_sleeper.stop()
        return super().shutdown()


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"  # keep-alive for JSON; SSE sends Connection: close

    server: DrinkmeHTTPServer  # narrowed from socketserver's BaseServer

    def log_message(self, fmt, *args):  # one compact line, stderr, no DNS lookups
        sys.stderr.write(f"[drinkme.http] {self.client_address[0]} {fmt % args}\n")

    def send_response(self, code, message=None):
        """THE chokepoint for drinkme_requests_total: every response, both
        dialects, JSON and SSE (send_response is the one call _json,
        _stream_headers, do_HEAD and do_OPTIONS all make), status known,
        route known (self.path/self.command are set before dispatch), one
        place. Runs on send_response ITSELF rather than after, so a client
        that goes away mid-stream still counted its 200 — the status line
        went out; module docstring's "clean 200" note applies here too."""
        super().send_response(code, message)
        metrics.inc_request(_route_label(self.path.split("?", 1)[0]), code)

    def handle_one_request(self):
        """Arm the read deadline before parsing anything for THIS
        request — the request line and headers (BaseHTTPRequestHandler's own
        machinery) and the declared body (_body, below) all execute inside
        this call, so a stalled client anywhere in that window falls into
        the base class's own `except socket.timeout`: log, close_connection
        = True, return — the connection closes and the thread returns.
        Re-armed HERE every time (once per keep-alive request) because
        setup() — the only other place a timeout could be set — runs once
        per CONNECTION, not per request, and a connection serves many."""
        self.connection.settimeout(_read_timeout_s())
        super().handle_one_request()

    # ------------------------------------------------------------ plumbing --

    def _json(self, status: int, obj: dict, *, close: bool = False) -> None:
        if 400 <= status < 500:
            # THE chokepoint (both dialects: _error and _aerror both funnel
            # here): a 400 that logs only its status (Claude Code's
            # generate_session_title, for one) leaves the reason in a body
            # nobody sees. Both envelopes nest it at error.message.
            err = obj.get("error") or {}
            # drinkme_4xx_total's "reason" label (4xx-reason logging, the metrics
            # endpoint): the envelope's own
            # `code` (OpenAI) or `type` (both dialects) — low cardinality by
            # construction, unlike `message`, which carries interpolated
            # request content (a model name, a bad field list) and would
            # blow the label space open.
            reason = err.get("code") or err.get("type") or "unknown"
            self._log_reason(status, err.get("message", ""), reason)
        body = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        if close:
            # 401 (auth failure, before any body byte is read) and 413
            # (body over cap) both close rather than keep-alive.
            # send_header sets close_connection=True for us on this exact
            # value, which is what makes "never drain the declared body"
            # safe — whatever the peer still has queued is discarded with
            # the socket instead of being read as the next request line.
            self.send_header("Connection", "close")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _log_reason(self, status: int, message: str, reason: str) -> None:
        """One line: method, path, status, the reason — never the request
        body, never the bearer token. Called from _json for every 4xx that
        still has a status line to send, and directly by the two exception
        handlers below for the mid-stream case, which does not. `reason` is
        the low-cardinality bucket for drinkme_4xx_total (the metrics
        endpoint); 5xx (the exception handlers' other branch) logs the same way but is not
        counted there — 4xx-reason logging was about 4xx, and a 500 is already loud."""
        self.log_message("%s %s %d %s", self.command, self.path, status,
                         _truncate(str(message)))
        if 400 <= status < 500:
            metrics.inc_4xx(status, reason)

    def _authed(self) -> bool:
        """The gate for /v1/*: `Authorization: Bearer <token>` or
        `x-api-key: <token>`. No token configured = open (local default).

        Called BEFORE any body byte is read (every POST route checks
        this ahead of _body_or_413), so a failure here has nothing to
        drain — it answers 401 and closes the connection instead, which is
        simpler and safer than draining first and gating second: an
        unauthenticated client that declares a huge Content-Length and
        never sends it cannot hold the thread in a body read at all,
        because none starts."""
        token = self.server.auth_token
        if token is None:
            return True
        if self.headers.get("Authorization") == f"Bearer {token}":
            return True
        if self.headers.get("x-api-key") == token:
            return True
        if self._anthropic:
            self._aerror(401, "authentication_error", "invalid x-api-key", close=True)
        else:
            self._error(401, "Incorrect API key provided.",
                        "invalid_request_error", "invalid_api_key", close=True)
        return False

    def _error(self, status: int, message: str, etype: str, code: str | None,
               param: str | None = None, *, close: bool = False) -> None:
        """OpenAI envelope — or, under /v1/messages, the same fact in
        Anthropic's, typed by status (the shared paths call this). `param`
        names the offending field, OpenAI's own shape for e.g.
        context_length_exceeded (`"param": "messages"`); every other caller
        leaves it None. `close`: see _json — 401/413 only."""
        if self._anthropic:
            return self._aerror(status, _ANTHROPIC_TYPE.get(status, "api_error"), message,
                                close=close)
        self._json(status, {"error": {"message": message, "type": etype,
                                      "param": param, "code": code}}, close=close)

    def _aerror(self, status: int, etype: str, message: str, *, close: bool = False) -> None:
        self._json(status, anthropic.error_body(etype, message), close=close)

    def _body(self) -> bytes:
        """The request body: Content-Length bounded (absent/invalid =
        empty), capped at DRINKME_MAX_BODY_BYTES — over cap raises
        BodyTooLarge before a single byte is read — and read under the
        connection's armed deadline (handle_one_request). Clears that
        deadline the instant the read finishes or the cap rejects it,
        success or failure: nothing downstream of this call (JSON parsing,
        generation, a streamed response) may ever race a timeout that was
        only ever meant to bound THIS read."""
        try:
            n = int(self.headers.get("Content-Length", 0))
        except ValueError:
            n = 0
        if n <= 0:
            self.connection.settimeout(None)
            return b""
        cap = _max_body_bytes()
        if n > cap:
            self.connection.settimeout(None)
            raise BodyTooLarge(n, cap)
        try:
            return self.rfile.read(n)
        finally:
            self.connection.settimeout(None)

    def _body_or_413(self) -> bytes | None:
        """_body(), rendering BodyTooLarge as 413 in the request's own
        dialect envelope and closing the connection. Callers use the
        same shape as `if not self._authed(): return`:
        `raw = self._body_or_413(); if raw is None: return`."""
        try:
            return self._body()
        except BodyTooLarge as e:
            msg = (f"request body of {e.content_length} bytes exceeds this "
                  f"server's {e.cap}-byte limit ({_BODY_CAP_ENV}).")
            if self._anthropic:
                self._aerror(413, "invalid_request_error", msg, close=True)
            elif getattr(self, "_responses_route", False):
                self._responses_error(413, msg, "invalid_request_error",
                                      "request_too_large", close=True)
            else:
                self._error(413, msg, "invalid_request_error",
                           "request_too_large", close=True)
            return None

    def _sse(self, obj) -> bool:
        """One SSE event, unbuffered (wfile is the raw socket, wbufsize=0).
        False means the client is gone — the caller feeds that to on_delta.
        Locked (_sse_lock, set up by _stream_headers): the SSE keep-alive
        thread writes to this same socket, and the lock is what keeps its
        ticks from ever landing INSIDE this frame's bytes."""
        data = b"data: " + (obj if isinstance(obj, bytes) else json.dumps(obj).encode())
        try:
            with self._sse_lock:
                self.wfile.write(data + b"\n\n")
                self.wfile.flush()
            return True
        except OSError:
            return False

    def _event(self, name: str, obj: dict) -> bool:
        """One NAMED SSE event (`event:` line + `data:` line) — Anthropic's
        framing. Same dead-socket contract and write lock as _sse."""
        try:
            with self._sse_lock:
                self.wfile.write(f"event: {name}\ndata: {json.dumps(obj)}\n\n".encode())
                self.wfile.flush()
            return True
        except OSError:
            return False

    def _sse_comment(self, line: bytes) -> bool:
        """A raw SSE comment frame — no `data:`/`event:` line, so it is a
        no-op to every conformant parser (WHATWG "Interpreting an event
        stream": a line starting with U+003A COLON is ignored). Used ONLY by
        the SSE keep-alive; same dead-socket contract and write lock as
        _sse/_event."""
        try:
            with self._sse_lock:
                self.wfile.write(line + b"\n\n")
                self.wfile.flush()
            return True
        except OSError:
            return False

    def _keepalive_start(self) -> threading.Event:
        """Start the SSE keep-alive's prefill window: a `: keep-alive` comment every
        DRINKME_SSE_KEEPALIVE_S seconds (default 10; 0 disables) until the
        returned Event is set. Covers BOTH halves of a client's silent wait
        — queued behind another generation (server.gen_lock) and the
        model's own prefill — since either can outlast a client's read
        timeout on a big cold prompt (a sibling to logging aborted streams
        distinctly: the client aborts, and
        the abort poisons the extends-only prefix reuse in engines.py). The
        caller sets the Event on the first real delta; this thread neither
        knows nor cares when that is, only that it should then stop."""
        stop = threading.Event()
        interval = _keepalive_interval()
        if interval > 0:
            def loop():
                while not stop.wait(interval):
                    if not self._sse_comment(b": keep-alive"):
                        return  # client gone; nothing left to keep alive

            threading.Thread(target=loop, daemon=True).start()
        return stop

    # ------------------------------------------------------ aliases/profiles --

    def _resolve_model(self, requested: str) -> tuple[str, str | None] | None:
        """`requested` -> (base id, generation profile name or None); None = 404."""
        return generation_profiles.resolve_model(requested, self.server.known_ids,
                                        self.server.generation_profiles)

    def _unknown_model(self, requested: str) -> str:
        return generation_profiles.unknown_model_message(requested, self.server.known_ids,
                                                 self.server.generation_profiles)

    def _generation_overlay(self, generation_profile: str | None) -> dict:
        """The named generation profile's overlay dict; {} for None."""
        gp = self.server.generation_profiles
        return gp.get(generation_profile, {}) if generation_profile else {}

    def _effective_defaults(self, eng: Engine, generation_profile: str | None) -> dict:
        """generation_config.json's table (model_meta()["sampling"]["defaults"]) with
        a generation profile's sampling fields overlaid — a request field still wins
        over the result (the caller runs gen_config.resolve_sampling on it)."""
        base = eng.model_meta()["sampling"]["defaults"]
        return generation_profiles.effective_defaults(
            base, self._generation_overlay(generation_profile))

    def _generation_template_kwargs(self, generation_profile: str | None) -> dict:
        return generation_profiles.template_kwargs(self._generation_overlay(generation_profile))

    def _stream_headers(self) -> None:
        self.send_response(200)
        self._streaming = True  # the 500 path must not write a second status line
        # guards _sse/_event/_sse_comment against the SSE keep-alive thread
        self._sse_lock = threading.Lock()
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-cache")
        # close-delimited body: no Content-Length, no chunked framing to hand-roll,
        # and send_header("Connection", "close") flips close_connection for us.
        self.send_header("Connection", "close")
        self.end_headers()

    # -------------------------------------------------------------- routes --

    def do_OPTIONS(self):  # CORS preflight; module docstring's posture
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers",
                         "Content-Type, Authorization, x-api-key, anthropic-version, "
                         "anthropic-beta")
        self.send_header("Access-Control-Max-Age", "86400")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_HEAD(self):
        # Claude Code sends HEAD /api/hello before its first request (seen on
        # the wire); a 501 "Unsupported method" is noise, and a
        # HEAD deserves an honest status with no body: 200 for /health, 404
        # for anything else. Never gated, never locked.
        path = self.path.split("?", 1)[0]
        self.send_response(200 if path == "/health" else 404)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self):
        metrics.inflight_inc()
        try:
            self._do_GET()
        finally:
            metrics.inflight_dec()

    def _do_GET(self):
        path = self.path.split("?", 1)[0]
        self._anthropic = path.startswith("/v1/messages")
        if path == "/health":  # lockless, ungated: probes don't need the secret
            # `device` rides the health payload because {"ok":true} alone
            # can hide a machine serving on the CPU at ~30x slow after a venv
            # resync swaps the pinned ROCm torch.
            # "The process is alive" was never the question anyone was asking.
            # `state`/`level` ride along for the same reason `device` does
            # (sleep/wake): "the process is alive" is not the question. A parked
            # server answers 200 here and 503 on every generate route, and a
            # probe that could not tell those apart would report a healthy
            # bottle that serves nothing.
            return self._json(200, {"ok": True, "model": self.server.engine.model_id,
                                    "device": self.server.engine.model_meta().get("device"),
                                    **self.server.sleep_state(),
                                    "prefix_cache": self.server.prefix_cache_state(),
                                    "uptime_s": int(time.time()) - self.server.started})
        if path == "/metrics":  # lockless, ungated, same posture as /health
            body = metrics.render().encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if path == "/tokenizer_info":  # lockless, gated like /v1/models (a tokenizer route)
            if not self._authed():
                return
            eng = self.server.engine
            meta = eng.model_meta()
            # vLLM-shaped route, so snake_case (docs/serve.md#get-v1models)
            caps = meta["capabilities"]
            veng = capability.engine_vision(eng)
            snake = {"thinking": caps.get("thinking"), "tool_format": caps.get("toolFormat"),
                     "vision": veng is not None,
                     "video": veng is not None and getattr(veng, "video", None) is not None}
            if "thinkingSwitch" in caps:
                snake["thinking_switch"] = caps["thinkingSwitch"]
            return self._json(200, {**eng.tokenizer_info(), **snake,
                                    "max_model_len": meta.get("contextWindow")})
        if path != "/v1/models":
            return self._error(404, f"Unknown request URL: GET {path}",
                               "invalid_request_error", "unknown_url")
        if not self._authed():
            return
        eng = self.server.engine
        data = [self._model_entry(eng, mid, gp)
                for mid in sorted(self.server.known_ids)
                for gp in (None, *sorted(self.server.generation_profiles))]
        self._json(200, {"object": "list", "data": data})

    def _model_entry(self, eng: Engine, mid: str, generation_profile: str | None) -> dict:
        """One /v1/models entry: OpenAI's four fields top-level (`id`,
        `object`, `created`, `owned_by`) and every drinkme key under one
        `drinkme` object (eng.model_meta(); docs/serve.md shows one as
        served). Aliases and profiles: every alias lists its own entry (id
        resolution is unchanged — they all still name the ONE engine), plus
        a `<id>:<profile>` entry per profile carrying the same `drinkme`
        object with `sampling.profile` named and `sampling.defaults`
        overlaid, so a client can read off exactly what that profile would
        sample with — and `compressionProfile` intact beside it (a key of
        its own, so the overlay never clobbers the pack's).

        `capabilities.vision` rides beside
        thinking/toolFormat — true|false, from the engine's OPTIONAL
        `.vision` surface (capability.engine_vision), not
        serving/capability.py's template probe — plus `imageInput` (max
        pixels, formats, sources) when it is true; `capabilities.video`
        likewise, with `videoInput` (capability.video_input_block) when
        true."""
        meta = eng.model_meta()
        if generation_profile is not None:
            mid = f"{mid}:{generation_profile}"
            meta = dict(meta, sampling={"profile": generation_profile,
                                        "defaults": self._effective_defaults(eng, generation_profile)})
        veng = capability.engine_vision(eng)
        has_video = veng is not None and getattr(veng, "video", None) is not None
        caps = dict(meta.get("capabilities") or {}, vision=veng is not None, video=has_video)
        if veng is not None:
            caps["imageInput"] = capability.image_input_block(veng)
        if has_video:
            caps["videoInput"] = capability.video_input_block(veng)
        meta = dict(meta, capabilities=caps)
        return {"id": mid, "object": "model", "created": self.server.started,
                "owned_by": "drinkme", "drinkme": meta}

    def do_POST(self):
        metrics.inflight_inc()
        try:
            self._do_POST()
        finally:
            metrics.inflight_dec()

    def _do_POST(self):
        path = self.path.split("?", 1)[0]
        self._anthropic = path.startswith("/v1/messages")
        self._streaming = False
        self._responses_route = path == "/v1/responses"
        if self._anthropic:
            return self._post_messages(path)
        if self._responses_route:  # OpenAI envelope, own render (serving/responses.py)
            return self._post_responses()
        if path == "/tokenize":  # lockless (a tokenizer route)
            return self._post_tokenize()
        if path == "/detokenize":  # lockless (a tokenizer route)
            return self._post_detokenize()
        if path in ("/sleep", "/wake_up"):  # sleep/wake
            return self._post_sleep(path)
        if path != "/v1/chat/completions":
            return self._error(404, f"Unknown request URL: POST {path}",
                               "invalid_request_error", "unknown_url")
        if not self._authed():  # headers only — no body byte read before the gate
            return
        raw = self._body_or_413()
        if raw is None:
            return
        try:
            req = json.loads(raw)
        except (ValueError, TypeError):
            return self._error(400, "Request body is not valid JSON.",
                               "invalid_request_error", None)
        if not isinstance(req, dict):
            return self._error(400, "Request body must be a JSON object.",
                               "invalid_request_error", None)
        messages = req.get("messages")
        if not isinstance(messages, list) or not messages:
            return self._error(400, "'messages' is required and must be a non-empty array.",
                               "invalid_request_error", None)
        eng = self.server.engine
        # shape-check every message BEFORE normalize/generate ever touch
        # it — `messages: [5]` would otherwise reach a bare `.get()` deep in
        # count_tokens (an AttributeError, not a 400), and `content: {}` be
        # silently coerced to "" by _content_text below rather than named
        # and refused. Also refuses a request over MAX_IMAGES images before
        # any of them decode.
        bad_msg = _validate_chat_messages(messages, eng=eng)
        if bad_msg:
            return self._error(400, bad_msg, "invalid_request_error", None)
        # Normalize AT THE BOUNDARY (content parts-array/null -> text, or,
        # for an image-bearing turn, a parts list with each image_url
        # decoded — _normalize_messages's docstring carries the full
        # reasoning for each fix, and /tokenize (a tokenizer route) applies
        # the identical function so its counts can never drift from what
        # this route actually sends to generate(). Image decoding runs here,
        # OUTSIDE the generation lock (serving/vision.py's Vision.prepare —
        # the ViT forward is the only part that needs it). Inside a guard:
        # a shape the validator did not name, or an image that fails to
        # decode (one of vision.ImageError's named codes), is still a JSON
        # 400 naming the part, never a traceback that drops the connection.
        try:
            messages, images, videos = _normalize_messages(messages, eng=eng)
        except vision.ImageError as e:
            return self._error(400, str(e), "invalid_request_error", e.code)
        except Exception as e:  # noqa: BLE001 — see above
            return self._error(400, f"messages could not be normalized: {e}",
                               "invalid_request_error", None)
        model = req.get("model")
        if model is not None and not isinstance(model, str):
            # `model: []` would otherwise reach a set-membership test
            # (generation_profiles.resolve_model's `requested in known_ids`), which
            # raises TypeError on an unhashable type rather than 404ing.
            return self._error(400, "'model' must be a string.",
                               "invalid_request_error", None)
        generation_profile = None
        if model is not None:
            # Aliases and profiles: any known id OR a known '<id>:<profile>' pair now resolves
            # (the default exact-match-only check stays true when there are no
            # aliases/profiles — known_ids is just {eng.model_id} then).
            resolved = self._resolve_model(model)
            if resolved is None:
                return self._error(404, self._unknown_model(model),
                                   "invalid_request_error", "model_not_found")
            _, generation_profile = resolved
        stop = req.get("stop") or []
        if isinstance(stop, str):
            stop = [stop]
        if not isinstance(stop, list) or not all(isinstance(x, str) for x in stop):
            return self._error(400, "'stop' must be a string or an array of strings.",
                               "invalid_request_error", None)
        # generation_config.json defaults / aliases and profiles: request field > profile >
        # generation_config.json default > SampleParams' own OpenAI default (gen_config.py / generation_profiles.py).
        defaults = self._effective_defaults(eng, generation_profile)
        sk = gen_config.resolve_sampling(
            {f: req.get(f) for f in gen_config.SAMPLING_FIELDS}, defaults)
        # THE sampling validator (engine.validated_sample_params), after
        # precedence resolution and before the prompt is rendered: a NaN
        # temperature, a nucleus of -1, a top_k of 1e309 or a boolean where
        # a number goes is this 400 here, never a sampler exception after
        # the prefill (or after a streaming 200 has gone out).
        # The token limit, under either of OpenAI's names (`max_tokens`,
        # and the newer `max_completion_tokens`): absent or null means the
        # default; anything SUPPLIED reaches the validator as supplied, so
        # `0`, `false`, `""`, `[]` are its named 400 — a `req.get(...) or
        # None` here would turn every falsey value into "unspecified"
        # and generate the full default budget.
        mt_field, mt = "max_tokens", None
        for f in ("max_completion_tokens", "max_tokens"):
            if req.get(f) is not None:
                if mt is not None:
                    return self._error(400, "'max_tokens' and 'max_completion_tokens' cannot "
                                       "both be set.", "invalid_request_error", None)
                mt_field, mt = f, req[f]
        try:
            params = validated_sample_params(
                sk, seed=req.get("seed"), max_tokens=mt, stop=stop, max_tokens_field=mt_field)
        except SampleParamError as e:
            return self._error(400, str(e), "invalid_request_error", None)
        # Tools: validate at the boundary, then hand the OpenAI-shaped list
        # through untouched — the model family's chat template renders the
        # definitions (Qwen3 natively), and the engine scans the output for
        # <tool_call> blocks. role:"tool" messages need nothing here: content
        # normalization above already made their content a plain string.
        tools = None
        if req.get("tools") is not None:
            try:
                tools = validate_tools(req["tools"]) or None  # [] = no tools
            except ValueError as e:
                return self._error(400, str(e), "invalid_request_error", None)
        # response_format: json_object / json_schema -> a schema for the
        # engine's grammar constraint (serving/constrain.py). Validated HERE,
        # by name, so unsupported schema keywords are a 400 at request time —
        # never silently-unvalidated output under a validated flag.
        rf = req.get("response_format")
        if rf is not None:
            if not isinstance(rf, dict) or rf.get("type") not in (
                    "text", "json_object", "json_schema"):
                return self._error(400, "response_format.type must be 'text', "
                                   "'json_object' or 'json_schema'.",
                                   "invalid_request_error", None)
            schema = None
            if rf["type"] == "json_object":
                schema = {"type": "object"}
            elif rf["type"] == "json_schema":
                js = rf.get("json_schema")
                # `response_format.json_schema: 5` would otherwise reach a bare
                # `.get()` on an int (5 is truthy, so `5 or {}` stays 5) —
                # an uncaught AttributeError instead of this 400.
                if js is not None and not isinstance(js, dict):
                    return self._error(400, "response_format.json_schema must "
                                       "be an object.", "invalid_request_error", None)
                schema = (js or {}).get("schema")
                if not isinstance(schema, dict):
                    return self._error(400, "response_format.json_schema.schema "
                                       "is required and must be an object.",
                                       "invalid_request_error", None)
            if schema is not None:
                from .constrain import validate_schema

                try:
                    validate_schema(schema)
                except ValueError as e:
                    return self._error(400, str(e), "invalid_request_error", None)
                params.output_schema_validated = True  # walked once, here
                if req.get("tools"):
                    return self._error(400, "response_format cannot be combined "
                                       "with tools.", "invalid_request_error", None)
                params.output_schema = schema
        # tool_choice: ONE contract across the three dialects (docs/serve.md's
        # capability matrix; responses.py and messages.py refuse the same
        # way). "none" withholds the tools this turn; "auto" offers them and
        # the model decides; "required" and a named function are FORCED tool
        # use, which this server cannot do — there is no constrained decode
        # to a call — and will not pretend to: a named 400, not a plain
        # answer with finish_reason "stop" under a flag that promised a call.
        # Anything else is refused as unsupported rather than ignored.
        tc = req.get("tool_choice")
        if tc is not None and not isinstance(tc, (str, dict)):
            return self._error(400, "'tool_choice' must be a string or an object.",
                               "invalid_request_error", None)
        if tc == "none":
            tools = None
        elif tc == "required" or (isinstance(tc, dict) and tc.get("type") == "function"):
            fn = tc.get("function") if isinstance(tc, dict) else None
            name = fn.get("name") if isinstance(fn, dict) else None
            what = "required" if tc == "required" else f"function {name!r}"
            return self._error(400, f"tool_choice: {what} (forced tool use) is not supported "
                               "by this server — it cannot constrain decoding to a call and "
                               "will not pretend to; use \"auto\" or \"none\".",
                               "invalid_request_error", "unsupported_tool_choice")
        elif tc is not None and tc != "auto":
            variant = tc.get("type") if isinstance(tc, dict) else tc
            return self._error(400, f"tool_choice: {variant!r} is not supported by this "
                               "server; use \"auto\" or \"none\".",
                               "invalid_request_error", "unsupported_tool_choice")
        # The off-menu capability refusal (serving/capability.py):
        # tools are actually about to be OFFERED to the model (tool_choice
        # "none" already zeroed `tools` above, so nothing to refuse there —
        # the model never sees a tool definition it might answer in a
        # dialect nobody can parse). A model whose tool_format is unparsed
        # ("unknown") or absent ("none") is refused loudly here, before a
        # token generates, rather than risking raw dialect text leaking into
        # `content` as an unrecognized reply.
        if tools:
            tool_format = eng.model_meta()["capabilities"]["toolFormat"]
            if tool_format not in capability.served_tool_formats():
                return self._error(400, capability.refusal_message(eng.model_id, tool_format),
                                   "invalid_request_error", "unsupported_tool_format")
        # chat_template_kwargs: the family template's own knobs, handed to
        # apply_chat_template as **kwargs (template.py) — Qwen3 understands
        # enable_thinking, reasoning_effort and preserve_thinking. We do not
        # enumerate them: the template owns its vocabulary, this layer only
        # carries it. reasoning_effort is ALSO accepted top-level (OpenAI's
        # spelling) and folded in, with an explicit chat_template_kwargs
        # entry winning on conflict — the specific beats the general.
        ctk = req.get("chat_template_kwargs")
        if ctk is not None and not isinstance(ctk, dict):
            return self._error(400, "'chat_template_kwargs' must be an object.",
                               "invalid_request_error", None)
        effort = req.get("reasoning_effort")
        if effort is not None and not isinstance(effort, str):
            return self._error(400, "'reasoning_effort' must be a string.",
                               "invalid_request_error", None)
        bad = RESERVED_KWARGS.intersection(ctk or ())
        if bad:
            # Refused BY NAME at the boundary (constrain.py's posture): these
            # are apply_chat_template's own positional/named arguments, which
            # template.py supplies — passing one through would be a TypeError
            # deep in the engine, i.e. a 500 for a client-side mistake.
            return self._error(400, "chat_template_kwargs may not set "
                               f"{', '.join(sorted(bad))} — the server owns "
                               "those.", "invalid_request_error", None)
        # Aliases and profiles: a profile's chat-template keys are the base layer — the
        # request's own reasoning_effort/chat_template_kwargs below still win.
        tkw: dict = self._generation_template_kwargs(generation_profile)
        if effort is not None:
            tkw["reasoning_effort"] = effort
        tkw.update(ctk or {})
        if params.output_schema is not None:
            # constrain.py's posture, made explicit at the boundary instead of
            # left to the engine's setdefault: a grammar that bans "<" starves
            # a thinking model into whitespace (live smoke). A
            # request asking for both gets the constraint, not a 400 — the
            # server can serve it correctly, just not while thinking.
            tkw["enable_thinking"] = False
        stream = req.get("stream")
        if stream is not None and not isinstance(stream, bool):
            return self._error(400, "'stream' must be a boolean.",
                               "invalid_request_error", None)
        opts = req.get("stream_options")
        self._include_usage = bool(isinstance(opts, dict) and opts.get("include_usage"))
        # the ONE value that crosses the seam (engine.GenerationRequest):
        # everything resolved above, built once, frozen
        greq = GenerationRequest(messages, params, tools=tools, template_kwargs=tkw,
                                 stream=bool(stream),
                                 request_id=f"chatcmpl-{uuid.uuid4().hex}", images=images,
                                 videos=videos)
        try:
            if greq.stream:
                self._chat_stream(eng, greq)
            else:
                self._chat(eng, greq)
        except ModelAsleep as e:  # sleep/wake — parked; nothing was generated
            if self._streaming:
                self._log_reason(503, str(e), "model_asleep")
                self._sse({"error": {"message": str(e), "type": "service_unavailable",
                                     "param": None, "code": "model_asleep"}})
            else:
                try:
                    self._error(503, str(e), "service_unavailable",
                                "model_asleep")  # logs via _json
                except OSError:
                    pass  # client gone; nothing left to say
        except ContextLengthExceeded as e:  # the context-length check — before the forward pass ever ran
            if self._streaming:
                self._log_reason(400, e.openai_message, "context_length_exceeded")
                self._sse({"error": {"message": e.openai_message, "type": "invalid_request_error",
                                     "param": "messages", "code": "context_length_exceeded"}})
            else:
                try:
                    self._error(400, e.openai_message, "invalid_request_error",
                               "context_length_exceeded", param="messages")  # logs via _json
                except OSError:
                    pass  # client gone; nothing left to say
        except Exception as e:  # noqa: BLE001 — the 500 must be OpenAI-shaped, not a traceback
            status = 400 if _template_refusal(e) else 500
            etype = "invalid_request_error" if status == 400 else "server_error"
            if self._streaming:
                # headers are out; a status line now would be garbage in the
                # body. Say it as an SSE error event and close (never silent).
                # _json's chokepoint never runs on this path (no status line
                # left to send) — log directly here instead.
                self._log_reason(status, f"{type(e).__name__}: {e}", etype)
                self._sse({"error": {"message": f"{type(e).__name__}: {e}",
                                     "type": etype, "param": None, "code": None}})
            else:
                try:
                    self._error(status, f"{type(e).__name__}: {e}", etype, None)  # logs via _json
                except OSError:
                    pass  # client gone; nothing left to say

    def _wake_url(self) -> str:
        """The absolute URL a 503 tells the caller to POST. Built from the
        Host header the client actually reached us on, so an instruction in
        an error body is one a human can paste — a hardcoded 127.0.0.1 would
        be wrong for everyone on the other side of the tunnel."""
        host = self.headers.get("Host") or "%s:%d" % self.server.server_address[:2]
        return f"http://{host}/wake_up"

    def _post_sleep(self, path: str) -> None:
        """POST /sleep[?level=N] and POST /wake_up (sleep/wake).

        GATED like /v1/* even though the path is not under it: parking the
        weights of a live server is the most consequential thing this port
        does, and "the bearer is the gate, not the origin" (module docstring)
        has to hold most exactly where it matters most.

        The lock is taken NON-BLOCKINGLY and a failure is a 409 with the
        engine's own words. Two reasons it is not a wait: serving/sleep.py's
        pass must never run beside a live forward pass, and a caller that
        asked a busy server to sleep wants to be told that — the systemd
        ExecStartPre this exists for has to decide
        whether to fall back to a Conflict stop, and it cannot decide while
        it is blocked.

        Idempotent by construction: /sleep on a sleeping server and /wake_up
        on an awake one both answer 200 with the current state, because a
        drop-in that must run before every unit start cannot also be a thing
        that fails the second time."""
        if not self._authed():  # headers only — no body byte read before the gate
            return
        if self._body_or_413() is None:
            return
        eng = self.server.engine
        fn = getattr(eng, "sleep" if path == "/sleep" else "wake", None)
        if fn is None or getattr(eng, "sleep_state", None) is None:
            return self._error(501, f"{eng.model_id!r} does not support sleep "
                               "— this engine implements no sleep/wake.",
                               "invalid_request_error", "sleep_unsupported")
        level = 1
        if path == "/sleep":
            raw = (self.path.split("?", 1)[1] if "?" in self.path else "")
            for part in raw.split("&"):
                key, _, val = part.partition("=")
                if key == "level":
                    try:
                        level = int(val)
                    except ValueError:
                        return self._error(
                            400, f"level={val!r} is not 1 or 2.",
                            "invalid_request_error", "invalid_sleep_level")
        if not self.server.gen_lock.acquire(blocking=False):
            return self._error(
                409, "a generation is in flight; the model cannot be parked "
                "or woken while one is running. Retry when it finishes.",
                "invalid_request_error", "generation_in_flight")
        try:
            state = fn(level) if path == "/sleep" else fn()
        except SleepError as e:
            return self._error(409, str(e), "invalid_request_error",
                               "sleep_state_conflict")
        except Exception as e:  # noqa: BLE001 — a failed park must not be a traceback
            return self._error(500, f"{type(e).__name__}: {e}",
                               "server_error", None)
        finally:
            self.server.gen_lock.release()
        # a wake is activity: without this the idle timer would find a stamp
        # from before the sleep and park the model again on its next tick.
        self.server.mark_generation()
        self._json(200, {"ok": True, "model": eng.model_id, **state})

    def _post_tokenize(self) -> None:
        """POST /tokenize (a tokenizer route): {"prompt": str} or {"messages": [...]} ->
        {count, tokens, max_model_len} — vLLM's TokenizeResponse field names
        (vllm/entrypoints/serve/tokenize/protocol.py: count, max_model_len,
        tokens). `messages` goes through _normalize_messages, exactly as
        /v1/chat/completions would, so the render Engine.tokenize does is
        the SAME one generate() would have done for this request — the
        count this route reports and the count the context-length check computes
        can never be two different opinions about one conversation.

        Image expansion: Engine.tokenize()'s signature
        carries no images or videos (only prompt/messages/tools), so it renders the
        template with ONE placeholder per image and tokenizes that — the
        header-only, no-ViT count. This route corrects it: each image's
        prompt tokens (Vision.prompt_tokens: `PreparedImage.tokens`, from
        Vision.count/.prepare, header only, no ViT, plus gemma-4's two
        markers) replace the single placeholder it rendered as, so `count` is
        `len(tokens) + Vision.expansion(images)` — the count generate() would
        actually consume, the same arithmetic a vision-aware
        Engine.count_tokens(req) applies internally where it has `req.images`
        to work with (GenerationRequest, not this route's bare prompt/
        messages). A video's three rendered ids are corrected the same way,
        by its video.PreparedVideo.expansion (timestamps, markers, runs)."""
        if not self._authed():  # headers only — no body byte read before the gate
            return
        raw = self._body_or_413()
        if raw is None:
            return
        try:
            req = json.loads(raw)
        except (ValueError, TypeError):
            return self._error(400, "Request body is not valid JSON.",
                               "invalid_request_error", None)
        if not isinstance(req, dict):
            return self._error(400, "Request body must be a JSON object.",
                               "invalid_request_error", None)
        prompt, messages = req.get("prompt"), req.get("messages")
        if (prompt is None) == (messages is None):
            return self._error(400, "exactly one of 'prompt' or 'messages' is required.",
                               "invalid_request_error", None)
        if prompt is not None and not isinstance(prompt, str):
            return self._error(400, "'prompt' must be a string.",
                               "invalid_request_error", None)
        eng = self.server.engine
        images: tuple = ()
        videos: tuple = ()
        if messages is not None:
            if not isinstance(messages, list) or not messages:
                return self._error(400, "'messages' must be a non-empty array.",
                                   "invalid_request_error", None)
            bad_msg = _validate_chat_messages(messages, eng=eng)  # same guard as /v1/chat/completions
            if bad_msg:
                return self._error(400, bad_msg, "invalid_request_error", None)
            try:
                messages, images, videos = _normalize_messages(messages, eng=eng)
            except vision.ImageError as e:
                return self._error(400, str(e), "invalid_request_error", e.code)
            except Exception as e:  # noqa: BLE001 — a JSON 400, never a dropped connection
                return self._error(400, f"messages could not be normalized: {e}",
                                   "invalid_request_error", None)
        tools = None
        if req.get("tools") is not None:
            try:
                tools = validate_tools(req["tools"]) or None
            except ValueError as e:
                return self._error(400, str(e), "invalid_request_error", None)
        try:
            tokens = eng.tokenize(prompt=prompt, messages=messages, tools=tools)
        except Exception as e:  # noqa: BLE001 — a bad template kwarg is a 400, not a 500
            status = 400 if _template_refusal(e) else 500
            etype = "invalid_request_error" if status == 400 else "server_error"
            return self._error(status, f"{type(e).__name__}: {e}", etype, None)
        veng = capability.engine_vision(eng)
        count = len(tokens) + (veng.expansion(images, videos) if veng is not None else
                               sum(img.tokens - 1 for img in images))
        self._json(200, {"count": count, "tokens": tokens,
                         "max_model_len": eng.model_meta().get("contextWindow")})

    def _post_detokenize(self) -> None:
        """POST /detokenize (a tokenizer route): {"tokens": [int]} -> {"prompt": str} —
        vLLM's DetokenizeResponse field name (same protocol.py: `prompt`,
        not `text`)."""
        if not self._authed():  # headers only — no body byte read before the gate
            return
        raw = self._body_or_413()
        if raw is None:
            return
        try:
            req = json.loads(raw)
        except (ValueError, TypeError):
            return self._error(400, "Request body is not valid JSON.",
                               "invalid_request_error", None)
        tokens = req.get("tokens") if isinstance(req, dict) else None
        if not isinstance(tokens, list) or not all(
                isinstance(t, int) and not isinstance(t, bool) for t in tokens):
            return self._error(400, "'tokens' is required and must be an array of integers.",
                               "invalid_request_error", None)
        eng = self.server.engine
        self._json(200, {"prompt": eng.detokenize(tokens)})

    def _post_messages(self, path: str) -> None:
        """POST /v1/messages and /v1/messages/count_tokens: Anthropic's
        envelope on every exit, never a bare 500 with a traceback."""
        if path not in ("/v1/messages", "/v1/messages/count_tokens"):
            return self._aerror(404, "not_found_error", f"Not Found: POST {path}")
        if not self._authed():  # headers only — no body byte read before the gate
            return
        raw = self._body_or_413()
        if raw is None:
            return
        try:
            req = json.loads(raw)
        except (ValueError, TypeError):
            return self._aerror(400, "invalid_request_error",
                                "Request body is not valid JSON.")
        eng = self.server.engine
        try:
            # Aliases and profiles: alias/profile resolution, same rule as the OpenAI boundary.
            # There is one engine, so a resolved alias rewrites `model` to the
            # canonical eng.model_id before parse_request's own check — the
            # profile name is the only thing that survives past this point.
            generation_profile = None
            m = req.get("model") if isinstance(req, dict) else None
            if isinstance(m, str) and m != eng.model_id:
                resolved = self._resolve_model(m)
                if resolved is None:
                    raise anthropic.MessagesError(404, "not_found_error",
                                                  self._unknown_model(m))
                _, generation_profile = resolved
                req = dict(req, model=eng.model_id)
            defaults = self._effective_defaults(eng, generation_profile)
            tkw_base = self._generation_template_kwargs(generation_profile)
            veng = capability.engine_vision(eng)
            if path.endswith("/count_tokens"):
                p = anthropic.parse_request(req, eng.model_id, need_max_tokens=False,
                                            defaults=defaults, tkw_base=tkw_base,
                                            vision_engine=veng,
                                            vision_reason=capability.vision_reason(eng))
                # Advertised context: the SAME lie usage.input_tokens tells below — Claude
                # Code calls this endpoint too, and a truthful count_tokens
                # beside a scaled usage would just make the client's own
                # arithmetic disagree with what it was just told. count_tokens(req)
                # sees req.images, so a vision-aware engine expands them itself
                # (http.py's own correction is only for /tokenize, whose Engine
                # Protocol signature carries no images at all).
                n = round(self._count(eng, p) * self._ctx_scale(eng))
                return self._json(200, {"input_tokens": n})
            p = anthropic.parse_request(req, eng.model_id, defaults=defaults,
                                        tkw_base=tkw_base, vision_engine=veng,
                                        vision_reason=capability.vision_reason(eng))
            # The off-menu capability refusal, Anthropic's envelope:
            # same gate as the OpenAI boundary, same reasoning — tools are
            # actually about to be offered (tool_choice "none"/"any"/"tool"
            # already resolved by parse_request above), so a model whose
            # tool_format is unparsed is refused before a token generates.
            if p.request.tools:
                tool_format = eng.model_meta()["capabilities"]["toolFormat"]
                if tool_format not in capability.served_tool_formats():
                    raise anthropic.invalid(capability.refusal_message(eng.model_id, tool_format))
            if p.request.stream:
                self._messages_stream(eng, p)
            else:
                self._messages(eng, p)
        except anthropic.MessagesError as e:
            if self._streaming:
                self._log_reason(e.status, e.message, e.etype)
                self._event(*anthropic.ev_error(e.etype, e.message))
            else:
                self._aerror(e.status, e.etype, e.message)  # logs via _json
        except ModelAsleep as e:  # sleep/wake — parked; nothing was generated
            # `model_asleep`, NOT Anthropic's `overloaded_error`: overloaded
            # means "try again shortly" and the SDKs act on it, and this
            # server will not wake on its own. An unknown type an SDK passes
            # through to the caller is the honest outcome here.
            if self._streaming:
                self._log_reason(503, str(e), "model_asleep")
                self._event(*anthropic.ev_error("model_asleep", str(e)))
            else:
                self._aerror(503, "model_asleep", str(e))  # logs via _json
        except ContextLengthExceeded as e:  # the context-length check — before the forward pass ever ran
            if self._streaming:
                self._log_reason(400, e.anthropic_message, "context_length_exceeded")
                self._event(*anthropic.ev_error("invalid_request_error", e.anthropic_message))
            else:
                self._aerror(400, "invalid_request_error", e.anthropic_message)  # logs via _json
        except Exception as e:  # noqa: BLE001 — api_error in the envelope, not a traceback
            status = 400 if _template_refusal(e) else 500
            etype = "invalid_request_error" if status == 400 else "api_error"
            if self._streaming:
                self._log_reason(status, f"{type(e).__name__}: {e}", etype)
                self._event(*anthropic.ev_error(etype, f"{type(e).__name__}: {e}"))
            else:
                try:
                    self._aerror(status, etype, f"{type(e).__name__}: {e}")  # logs via _json
                except OSError:
                    pass  # client gone; nothing left to say

    # ---------------------------------------------------------- generation --

    def _generate(self, eng: Engine, req: GenerationRequest, emit=None) -> Turn:
        """THE generation core, shared by all three dialects, each of which
        hands it the one value it built (engine.GenerationRequest) — and the
        ONE consumer of an engine's event stream in this file.

        gen_lock: the FIFO-ish queue (module docstring).

        The think split runs on the way out, before anything else looks at
        the text: reasoning leaves through the "reasoning" channel while the
        model thinks, the answer through "content" after the close tag, and
        no byte is ever announced in the wrong channel — so a reasoning block
        can never be mistaken for the answer that carries the tool calls. The
        splitter is built from the engine's StreamStart event, the first
        thing every stream yields: the prompt decides which channel byte one
        belongs to, and StreamStart is the prompt's own verdict. A delta
        swallowed whole by the tag lookahead (<= 7 bytes) emits nothing this
        step; the next one carries it, so the disconnect signal is at most
        one delta late.

        emit(channel, text) -> bool is the streaming hook, called under the
        lock for every piece as it is split; False aborts exactly as on_delta
        would. None = collect only. Either way the Turn carries both channels
        whole, so a non-streamed answer IS the streamed one concatenated, by
        construction (think.py's chunk-invariance).

        The context-length check runs FIRST, lockless, before gen_lock
        is even asked for — a doomed request must not queue behind a real
        one just to be refused. eng.model_meta().get("contextWindow") is
        None for an engine that publishes no window (FakeEngine by default,
        and any minimal Engine that only implements what a given test
        exercises) — the check, and the count_tokens call that would be its
        only reason to run, are both skipped rather than guessing."""
        t0 = time.time()
        # Sleep/wake, first of two: lockless, so a request to a parked server is
        # refused at once instead of queueing behind a sleep that is itself
        # waiting for the lock.
        state = self.server.sleep_state()
        if state.get("state") == "asleep":
            raise ModelAsleep(state, self._wake_url())
        params = req.sampling
        ctx = eng.model_meta().get("contextWindow")
        if ctx is not None:
            n_prompt = self._prompt_tokens(eng, req)
            room = ctx - n_prompt
            if room < 1:
                # no room for a single token: a refusal is the only honest
                # answer whatever the clamp setting says
                raise ContextLengthExceeded(n_prompt, params.max_tokens, ctx, eng.model_id,
                                            prompt_only=True)
            if params.max_tokens > room:
                if not _clamp_max_tokens():
                    raise ContextLengthExceeded(n_prompt, params.max_tokens, ctx, eng.model_id)
                # The max_tokens clamp: say so once in the log, and let the generation
                # run to the window — the client sees the length stop it
                # would have seen from llama.cpp or Ollama. The request is a
                # frozen value: the clamped one is a new value, not a
                # mutation something else might still be holding.
                self.log_message("max_tokens %d clamped to %d: %d prompt tokens in a "
                                 "%d-token window (%s)", params.max_tokens, room,
                                 n_prompt, ctx, eng.model_id)
                params = dataclasses.replace(params, max_tokens=room)
                req = dataclasses.replace(req, sampling=params)
        split = None
        reasoning: list[str] = []
        content: list[str] = []
        last_emit_t = None  # None until the first delta; also the "TTFT taken" flag

        def push(r: str, c: str) -> bool:
            nonlocal last_emit_t
            if r or c:
                # TTFT is bench/serve_timing_ab.py's OWN definition
                # (request()'s ttft) read server-side — first delta carrying
                # reasoning OR content, timed from t0 above, which is taken
                # before the lock is even requested, so queueing counts here
                # exactly as it counts in the bench's t0-before-c.request().
                now = time.time()
                if last_emit_t is None:
                    metrics.observe_ttft(now - t0)
                else:
                    metrics.observe_inter_token(now - last_emit_t)
                last_emit_t = now
            reasoning.append(r)
            content.append(c)
            ok = True
            if r and emit is not None:
                ok = emit("reasoning", r)
            if ok and c and emit is not None:
                ok = emit("content", c)
            return ok

        with self.server.gen_lock:
            # Sleep/wake, second of two: the check above is lockless and therefore
            # racy against a sleep that was waiting for this lock. Re-asked
            # HERE, where the answer cannot change under us, so a request can
            # never reach a forward pass whose weights are on the host.
            state = self.server.sleep_state()
            if state.get("state") == "asleep":
                raise ModelAsleep(state, self._wake_url())
            self.server.mark_generation()
            stream = eng.generate(req)
            start = next(stream)
            if not isinstance(start, StreamStart):  # the protocol, not a convention
                raise TypeError(f"{type(eng).__name__}.generate yielded "
                                f"{type(start).__name__} before StreamStart")
            # the row's tags (think.py): the model's announced tool_format
            # names the tool_formats row, and the reasoning channel is a
            # column of the same row — gemma's `<|channel>thought`, Qwen's
            # `<think>` for everyone else, Muse-Glimmer's messages to `self`
            # (think.splitter picks the machine). Whether the prompt opened
            # one is the engine's StreamStart verdict.
            split = think.splitter(start.opens_think,
                                   tool_format=eng.model_meta()["capabilities"]["toolFormat"])
            res = None
            ok = None  # what the last Delta was answered with; False = abort
            while True:
                try:
                    ev = stream.send(ok)
                except StopIteration:
                    break
                if isinstance(ev, Delta):
                    ok = push(*split.feed(ev.text))
                elif isinstance(ev, Finished):
                    res, ok = ev.result, None
            if res is None:
                raise RuntimeError(f"{type(eng).__name__}.generate ended without a "
                                   "Finished event")
        self.server.mark_generation()  # idle is measured from the END too
        finish = res.finish_reason
        if finish != "abort":
            # a partial tag at EOS is the text it really was, not a tag
            if not push(*split.flush()):
                finish = "abort"
        metrics.add_generated_tokens(res.completion_tokens)
        if finish == "abort":
            # _stream_headers() already sent status 200 before a single byte
            # of content existed — SSE has to commit headers up front — so
            # the automatic access log shows a clean 200 no matter how the
            # stream actually ended: a client whose SSH tunnel died mid-stream
            # (its laptop slept) logs the same 200 as a finished one, and only
            # this line tells them apart.
            metrics.inc_aborted()
            dialect = ("messages" if self._anthropic else
                       "responses" if getattr(self, "_responses_route", False) else
                       "chat.completions")
            self.log_message("stream aborted (%s): %d tokens in %.1fs",
                             dialect, res.completion_tokens, time.time() - t0)
        return Turn("".join(reasoning), "".join(content), res.tool_calls, finish,
                    res.stop_sequence, res.prompt_tokens, res.completion_tokens,
                    res.cached_tokens)

    def _prompt_tokens(self, eng: Engine, req: GenerationRequest) -> int:
        """Engine.count_tokens for this request. NO lock: rendering +
        tokenizing touch no model state and no device (engine.py), so a
        count never waits behind a generation. Shared by _generate's
        request-time context check, /v1/messages/count_tokens and the
        streamed Messages reply's up-front input count (_count below)."""
        return eng.count_tokens(req)

    def _count(self, eng: Engine, p: anthropic.Parsed) -> int:
        return self._prompt_tokens(eng, p.request)

    def _ctx_scale(self, eng: Engine) -> float:
        """real/advertised, or 1.0 — the default — whenever there
        is no lie to tell: no --advertised-ctx set, the engine does not
        publish a real contextWindow (FakeEngine's dev path, unless a test
        gives it one), or the ask is not actually BELOW real. The one-sided
        guard matters: this number only ever INFLATES reported usage, never
        shrinks it past the truth, so a misconfigured advertised-ctx above
        the real one just falls back to honest instead of hiding tokens."""
        adv = self.server.advertised_ctx
        real = eng.model_meta().get("contextWindow")
        if not adv or not real or adv >= real:
            return 1.0
        return real / adv

    # ------------------------------------------------------------- openai --

    def _chat(self, eng: Engine, req: GenerationRequest) -> None:
        t = self._generate(eng, req)
        message: dict = {"role": "assistant", "content": t.content}
        if t.reasoning:
            # OMITTED when empty, never null: a client that tests
            # `"reasoning_content" in message` must not see a field for a turn
            # that did no thinking.
            message["reasoning_content"] = t.reasoning
        if t.tool_calls:
            # OpenAI shape: any pre-call text stays as content, else null;
            # function.arguments is a JSON-encoded STRING (to_openai_calls)
            message["content"] = t.content or None
            message["tool_calls"] = to_openai_calls(t.tool_calls)
        self._json(200, {
            "id": req.request_id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": eng.model_id,
            "choices": [{"index": 0,
                         "message": message,
                         "finish_reason": t.finish if t.finish in
                         ("stop", "length", "tool_calls") else "stop"}],
            "usage": _usage(t),
        })

    def _chat_stream(self, eng: Engine, req: GenerationRequest) -> None:
        cid, created = req.request_id, int(time.time())

        def chunk(delta: dict, finish: str | None = None) -> dict:
            return {"id": cid, "object": "chat.completion.chunk", "created": created,
                    "model": eng.model_id,
                    "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}

        self._stream_headers()
        # role chunk first (OpenAI shape), BEFORE the lock: a queued client sees
        # the stream open immediately, and a disconnect while queued aborts on
        # the first content write instead of running the whole generation.
        alive = self._sse(chunk({"role": "assistant", "content": ""}))
        # The SSE keep-alive: comments cover the silent stretch from here to the
        # first real delta (queued + prefill); emit() stops it on arrival.
        keepalive = self._keepalive_start()

        def emit(channel: str, text: str) -> bool:
            keepalive.set()
            key = "reasoning_content" if channel == "reasoning" else "content"
            return alive and self._sse(chunk({key: text}))

        try:
            t = self._generate(eng, req, emit)
        finally:
            keepalive.set()  # generation ended (or never started a delta) either way
        if t.finish == "abort":
            return  # client is gone; there is no one to send [DONE] to
        if t.tool_calls:
            # ONE delta chunk carrying every completed call (visible text
            # already streamed above); "index" per entry is how streaming
            # clients accumulate calls across chunks — ours arrive whole.
            calls = [dict(index=i, **c) for i, c in enumerate(to_openai_calls(t.tool_calls))]
            if not self._sse(chunk({"tool_calls": calls})):
                return
        if not self._sse(chunk({}, finish=t.finish)):
            return
        # stream_options.include_usage (OpenAI spec): one extra chunk before
        # [DONE], empty choices, carrying usage. Clients that count tokens —
        # pi does, for its compaction gauge — get nothing without this and
        # misfire ("Compacted from 0 tokens").
        if self._include_usage:
            self._sse({"id": cid, "object": "chat.completion.chunk", "created": created,
                       "model": eng.model_id, "choices": [],
                       "usage": _usage(t)})
        self._sse(b"[DONE]")

    # ---------------------------------------------------------- anthropic --

    def _messages(self, eng: Engine, p: anthropic.Parsed) -> None:
        t = self._generate(eng, p.request)
        self._json(200, anthropic.render_message(p.request.request_id,
                                                 eng.model_id, t,
                                                 scale=self._ctx_scale(eng)))

    def _messages_stream(self, eng: Engine, p: anthropic.Parsed) -> None:
        # message_start carries the input count, and the real wire sends it
        # before a single token — so the count is taken here, lockless, off
        # the same render generate() is about to do (a second template pass
        # + tokenize: milliseconds against a decode), and BEFORE the headers,
        # so a render failure is still a clean error body.
        scale = self._ctx_scale(eng)  # advertised context: one number, read once per request
        n_in = round(self._count(eng, p) * scale)
        mid = p.request.request_id
        self._stream_headers()
        # message_start BEFORE the lock, for the same reason as the OpenAI
        # role chunk: a queued client sees the stream open at once, and a
        # disconnect while queued aborts on the first block write.
        alive = self._event(*anthropic.ev_message_start(mid, eng.model_id, n_in))
        blocks = anthropic.StreamBlocks(self._event)
        # The SSE keep-alive: same window as the OpenAI role chunk above — queued + prefill,
        # ended by the first real delta.
        keepalive = self._keepalive_start()

        def emit(channel: str, text: str) -> bool:
            keepalive.set()
            return alive and blocks.delta(channel, text)

        try:
            t = self._generate(eng, p.request, emit)
        finally:
            keepalive.set()
        if t.finish == "abort":
            return  # client is gone; nothing left to say
        if not blocks.close():
            return
        for call in t.tool_calls or []:
            if not blocks.tool_use(call):
                return
        if not self._event(*anthropic.ev_message_delta(t, scale=scale)):
            return
        self._event(*anthropic.ev_message_stop())

    # ---------------------------------------------------------- responses --

    def _post_responses(self) -> None:
        """POST /v1/responses: the OpenAI envelope on every exit, the same
        exception ladder as the two other dialects. The translation is
        serving/responses.py's; this method is routing, the model/profile
        resolution the chat boundary does, the capability gate, and the
        dispatch to the two renderers below."""
        if not self._authed():  # headers only — no body byte read before the gate
            return
        raw = self._body_or_413()
        if raw is None:
            return
        try:
            req = json.loads(raw)
        except (ValueError, TypeError):
            return self._error(400, "Request body is not valid JSON.",
                               "invalid_request_error", None)
        eng = self.server.engine
        self._rs = None  # the live Stream, once headers are out (the error path renders through it)
        try:
            # Aliases and profiles: same rule as the other two boundaries — a
            # resolved alias rewrites `model` to the canonical id before
            # parse_request's own check; the profile name is what survives.
            generation_profile = None
            m = req.get("model") if isinstance(req, dict) else None
            if isinstance(m, str) and m != eng.model_id:
                resolved = self._resolve_model(m)
                if resolved is None:
                    raise responses.ResponsesError(404, self._unknown_model(m),
                                                   code="model_not_found")
                _, generation_profile = resolved
                req = dict(req, model=eng.model_id)
            defaults = self._effective_defaults(eng, generation_profile)
            tkw_base = self._generation_template_kwargs(generation_profile)
            p = responses.parse_request(req, eng.model_id, defaults=defaults,
                                        tkw_base=tkw_base,
                                        vision_engine=capability.engine_vision(eng),
                                        vision_reason=capability.vision_reason(eng))
            # The off-menu capability refusal: same gate as the chat boundary,
            # same reasoning — tools are actually about to be offered
            # (tool_choice "none" already zeroed them in parse_request).
            if p.request.tools:
                tool_format = eng.model_meta()["capabilities"]["toolFormat"]
                if tool_format not in capability.served_tool_formats():
                    raise responses.ResponsesError(
                        400, capability.refusal_message(eng.model_id, tool_format),
                        code="unsupported_tool_format")
            if p.request.stream:
                self._responses_stream(eng, p)
            else:
                self._responses(eng, p)
        except responses.ResponsesError as e:
            self._responses_error(e.status, e.message, e.etype, e.code)
        except ModelAsleep as e:  # sleep/wake — parked; nothing was generated
            self._responses_error(503, str(e), "service_unavailable", "model_asleep")
        except ContextLengthExceeded as e:  # before the forward pass ever ran
            self._responses_error(400, e.openai_message, "invalid_request_error",
                                  "context_length_exceeded")
        except Exception as e:  # noqa: BLE001 — the 500 must be OpenAI-shaped, not a traceback
            status = 400 if _template_refusal(e) else 500
            etype = "invalid_request_error" if status == 400 else "server_error"
            self._responses_error(status, f"{type(e).__name__}: {e}", etype, None)

    def _responses_error(self, status: int, message: str, etype: str,
                         code: str | None, *, close: bool = False) -> None:
        """One exit for every refusal on /v1/responses: a JSON error body while
        a status line can still go out; once the stream is open, the `error`
        event and response.failed through the live Stream (no status line
        left to send — _json's chokepoint never runs, so log here). `close`:
        see _json — 413 only, and only reachable pre-stream (the body
        cap fires before `stream` is even known)."""
        if self._streaming:
            self._log_reason(status, message, code or etype)
            if self._rs is not None:
                self._rs.fail(code, message)
            return
        try:
            self._error(status, message, etype, code, close=close)  # logs via _json
        except OSError:
            pass  # client gone; nothing left to say

    def _responses(self, eng: Engine, p: responses.Parsed) -> None:
        rid, created = p.request.request_id, int(time.time())
        t = self._generate(eng, p.request)
        self._json(200, responses.render_turn(rid, created, eng.model_id, p.echo, t,
                                              namespaces=p.namespaces))

    def _responses_stream(self, eng: Engine, p: responses.Parsed) -> None:
        rs = responses.Stream(self._event, p.request.request_id, int(time.time()),
                              eng.model_id, p.echo, namespaces=p.namespaces)
        self._stream_headers()
        self._rs = rs
        # response.created + response.in_progress BEFORE the lock, for the same
        # reason as the OpenAI role chunk: a queued client sees the stream open
        # at once, and a disconnect while queued aborts on the first delta.
        alive = rs.start()
        # The SSE keep-alive: same window as the other two dialects — queued +
        # prefill, ended by the first real delta.
        keepalive = self._keepalive_start()

        def emit(channel: str, text: str) -> bool:
            keepalive.set()
            return alive and rs.delta(channel, text)

        try:
            t = self._generate(eng, p.request, emit)
        finally:
            keepalive.set()
        if t.finish == "abort":
            return  # client is gone; nothing left to say
        rs.finish(t)


class _IdleSleeper(threading.Thread):
    """DRINKME_SLEEP_ON_IDLE_S (sleep/wake), off by default: park the model after
    `seconds` with no generation.

    Here rather than in serving/sleep.py because the only input is
    `server.last_generation` — a fact this file owns — and because
    serving/sleep.py imports torch, which this file must not.

    Deliberately not a scheduler: it polls the stamp _generate bumps on the
    way in AND on the way out, so a long generation can never be interrupted
    by a timer that fired while it ran. It takes the SAME non-blocking lock
    the route takes and simply tries again next tick when it loses, which is
    also what makes it safe to run beside a human hitting /sleep by hand.
    Latched, so a quiet server gets one sleep attempt per idle period rather
    than one per tick."""

    def __init__(self, srv: DrinkmeHTTPServer, seconds: float,
                 tick: float | None = None):
        super().__init__(name="drinkme-idle", daemon=True)
        self.srv, self.seconds = srv, seconds
        self._tick = tick if tick is not None else min(5.0, max(0.05, seconds / 4))
        self._stop = threading.Event()
        self.slept = 0  # how many times it actually parked; tests read it

    def stop(self) -> None:
        self._stop.set()

    def run(self) -> None:
        while not self._stop.wait(self._tick):
            if time.monotonic() - self.srv.last_generation < self.seconds:
                continue
            if self.srv.sleep_state().get("state") != "awake":
                continue
            fn = getattr(self.srv.engine, "sleep", None)
            if fn is None:
                return  # nothing to do, ever: stop burning a thread on it
            if not self.srv.gen_lock.acquire(blocking=False):
                continue  # busy; the stamp will move and reset the clock
            try:
                fn(1)
                self.slept += 1
            except Exception as e:  # noqa: BLE001 — a timer must not kill a server
                sys.stderr.write(f"[drinkme.http] idle sleep failed "
                                 f"({type(e).__name__}: {e}) — serving on\n")
                return
            finally:
                self.srv.gen_lock.release()


def start_server(engine: Engine, host: str = "127.0.0.1", port: int = 8080,
                 auth_token: str | None = None,
                 advertised_ctx: int | None = None,
                 served_names: list[str] | None = None,
                 generation_profiles: dict | None = None,
                 sleep_on_idle: float = 0.0) -> DrinkmeHTTPServer:
    """Bind and serve on a daemon thread; returns the live server. Pass port 0
    to let the OS pick — server.server_address[1] is the real port. Stop with
    server.shutdown(); block on server.thread.join() to serve from a CLI.

    `advertised_ctx`: the advertised context's scaled-down reported window.
    `served_names`/`generation_profiles`: aliases and named sampling/template overlays.

    `sleep_on_idle` (sleep/wake) is seconds of generation quiet before an automatic
    level-1 sleep; 0 = never, which is the default, unchanged from before
    this flag existed. serve.py resolves it from --sleep-on-idle /
    DRINKME_SLEEP_ON_IDLE_S, the same way it resolves --advertised-ctx, so
    this file parses no environment."""
    srv = DrinkmeHTTPServer((host, port), engine, auth_token=auth_token,
                            advertised_ctx=advertised_ctx,
                            served_names=served_names, generation_profiles=generation_profiles)
    # poll_interval only bounds how fast shutdown() is noticed; the default
    # 0.5s turns every test teardown into half a second of nothing.
    srv.thread = threading.Thread(target=lambda: srv.serve_forever(poll_interval=0.05),
                                  name="drinkme-http", daemon=True)
    srv.thread.start()
    if sleep_on_idle > 0:
        srv.idle_sleeper = _IdleSleeper(srv, sleep_on_idle)
        srv.idle_sleeper.start()
    return srv
