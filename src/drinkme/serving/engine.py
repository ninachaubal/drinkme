"""The engine seam: everything the HTTP layer may ask of a model.

One value crosses it going in — GenerationRequest, built by whichever
dialect adapter (or direct caller) wants a generation — and a stream of
typed events comes back: StreamStart, then Delta…, then Finished carrying
the GenResult. Nothing else: no tokenizer, no torch, no pack format leaks
past this file, and nothing travels beside the call: template kwargs and
the prompt's think-block verdict are in the request and the first event,
never in a thread-local.
The real engines (compressed pack and --stock bf16; MLX) and FakeEngine
implement the same Protocol, which is what makes bench's A/B the serve path:
one server, only the Engine underneath differs.

The abort contract is essential (serve.py: backpressure is a signal, not
something to buffer): the consumer answers a Delta with `send(False)` when
the client is gone, and generate() must stop promptly, set finish_reason
"abort", and finish with what was actually delivered. `complete()` below is
the driver every non-streaming caller uses; http.py drives the stream
itself because it needs StreamStart.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Iterator, Protocol

if TYPE_CHECKING:
    from .vision import PreparedImage


@dataclass
class SampleParams:
    """One generation's knobs. Defaults are the OpenAI ones (temperature 1,
    nucleus off) except max_tokens, which must be finite on a local machine."""

    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0  # 0 = off; not OpenAI, but every local client sends it
    repetition_penalty: float = 1.0
    # OpenAI's additive penalties. Applied to GENERATED tokens only (vLLM's
    # reading of "the text so far"; repetition_penalty covers prompt+output).
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    seed: int | None = None
    stop: list[str] = field(default_factory=list)
    # 4096, not 512: an agentic turn through a reasoning model spends output
    # on thinking + tool-call JSON before any visible text, and 512 starves
    # that — reasoning exhaustion served empty content and mimicked a garbage
    # bug (same class). Finite on purpose: a runaway loop on a
    # ~1.5 tok/s machine is real time; requests still override per call.
    max_tokens: int = 4096
    # JSON Schema for grammar-constrained output (serving/constrain.py), or
    # None: the requested output schema, parsed and validated at the HTTP
    # boundary from the dialect's wire field (OpenAI `response_format`,
    # Responses `text.format`, Anthropic `output_config.format`); a plain
    # dict so this seam still never leaks torch or tokenizers.
    output_schema: dict | None = None
    # Set by whoever already ran constrain.validate_schema on that schema —
    # the HTTP boundary does, per request, and the engine was walking the same
    # (possibly deep) schema again to be told the same thing (the hot-loop audit). False by default: a caller that has not validated still gets
    # validated, because unvalidated output under a validated flag is the one
    # thing constrain.py exists to prevent.
    output_schema_validated: bool = False


# ------------------------------------------------- the sampling validator --
#
# The dataclass above validates nothing, and the samplers assume a domain:
# torch.multinomial over a row divided by a NaN temperature is a
# probability-tensor RuntimeError, a nucleus of -1 filters every candidate
# and the draw raises, int(1e309) is an OverflowError no parser caught. On
# a real engine the first two fail AFTER the prefill and, streaming, after
# the 200 has gone out. So ONE validator, torch-free, applied by every wire
# dialect after precedence resolution (request > named profile >
# generation_config.json > SampleParams' default) and before any prefill,
# and by the profile loader at boot. Each API renders SampleParamError as
# its own 400 envelope.

class SampleParamError(ValueError):
    """A sampling field set — by a request, a named profile or a
    checkpoint's generation_config.json — to a value the samplers cannot
    take. `field` names it; `message` says what would have been accepted."""

    def __init__(self, field: str, message: str):
        super().__init__(f"{field}: {message}")
        self.field, self.message = field, message


def _is_finite(v: float) -> bool:
    return v == v and v not in (float("inf"), float("-inf"))


def _number(field: str, v) -> float:
    if isinstance(v, bool):
        raise SampleParamError(field, "must be a number, not a boolean")
    if not isinstance(v, (int, float)):
        raise SampleParamError(field, "must be a number")
    if not _is_finite(v):
        raise SampleParamError(field, "must be a finite number")
    return float(v)


def _integer(field: str, v) -> int:
    """An integer, or an integral float (JSON has one number type: 50.0 is
    50); never a bool, never inf/nan, never 10.5. int() alone accepted True,
    raised OverflowError on 1e309 and rounded 10.5 to 10."""
    if isinstance(v, bool):
        raise SampleParamError(field, "must be an integer, not a boolean")
    if isinstance(v, int):
        return v
    if isinstance(v, float) and _is_finite(v) and v.is_integer():
        return int(v)
    raise SampleParamError(field, "must be an integer")


def check_sampling(sampling: dict) -> dict:
    """The validator over SampleParams' six sampling fields, whichever of
    them `sampling` carries (a request's resolved table has all six; a
    profile or a generation_config.json only what it sets). Returns the
    same keys, coerced (floats, top_k an int). Raises SampleParamError
    naming the first field outside its domain:

      temperature         finite, >= 0 (0 is greedy)
      top_p               finite, in (0, 1]
      top_k               integer, >= 0 (0 is off)
      repetition_penalty  finite, > 0 (HF divides by it)
      presence_penalty    finite
      frequency_penalty   finite
    """
    out: dict = {}
    for f, v in sampling.items():
        if f == "temperature":
            out[f] = _number(f, v)
            if out[f] < 0:
                raise SampleParamError(f, "must be >= 0")
        elif f == "top_p":
            out[f] = _number(f, v)
            if not 0 < out[f] <= 1:
                raise SampleParamError(f, "must be in (0, 1]")
        elif f == "top_k":
            out[f] = _integer(f, v)
            if out[f] < 0:
                raise SampleParamError(f, "must be >= 0")
        elif f == "repetition_penalty":
            out[f] = _number(f, v)
            if out[f] <= 0:
                raise SampleParamError(f, "must be > 0")
        elif f in ("presence_penalty", "frequency_penalty"):
            out[f] = _number(f, v)
        else:
            raise SampleParamError(f, "is not a sampling field")
    return out


def validated_sample_params(sampling: dict, *, seed=None, max_tokens=None,
                            stop: list | None = None,
                            max_tokens_field: str = "max_tokens") -> SampleParams:
    """A request's SampleParams, through check_sampling plus the two
    per-request integers: `seed` (None or an integer) and `max_tokens`
    (None = SampleParams' default; else a positive integer, named on the
    wire by `max_tokens_field` — Responses says max_output_tokens). Every
    parser builds its SampleParams here and nowhere else, so a value that
    fails does so as a SampleParamError before a prompt is even rendered."""
    sk = check_sampling(sampling)
    if seed is not None:
        seed = _integer("seed", seed)
    if max_tokens is None:
        mt = SampleParams.max_tokens
    else:
        mt = _integer(max_tokens_field, max_tokens)
        if mt < 1:
            raise SampleParamError(max_tokens_field, "must be a positive integer")
    return SampleParams(
        temperature=sk["temperature"], top_p=sk["top_p"], top_k=sk["top_k"],
        repetition_penalty=sk["repetition_penalty"],
        presence_penalty=sk["presence_penalty"], frequency_penalty=sk["frequency_penalty"],
        seed=seed, stop=list(stop or []), max_tokens=mt)


@dataclass(frozen=True)
class GenerationRequest:
    """One generation, as the dialect adapter resolved it — the ONE value
    that crosses the seam per request. Frozen: an
    adapter builds it once, and what the engine sees is what was built.

    `messages`  OpenAI-shaped history, already normalized by the dialect
                (http._normalize_messages, messages.convert_messages,
                responses.convert_input). A turn's content is a string, or,
                for a turn carrying images, a parts list in wire order:
                `[{"type": "text", "text": ...}, {"type": "image"}, ...]`.
                Each `{"type": "image"}` is one image, rendered by the chat
                template as the architecture's placeholder (Qwen3.5:
                `<|vision_start|><|image_pad|><|vision_end|>`). A system
                turn never carries one.
    `sampling`  the validated SampleParams, the constraint included.
    `tools`     validated OpenAI tool definitions actually being OFFERED,
                or None (tool_choice "none" already zeroed it).
    `template_kwargs`
                this request's chat-template kwargs — the profile's keys as
                the base, then the request's own (reasoning_effort,
                chat_template_kwargs, thinking on/off, enable_thinking=False
                under a JSON schema). The REQUEST layer only: an engine
                merges these over its own generation_config.json defaults
                (template.build_prompt), the request winning.
    `stream`    the wire is SSE; nothing the engine does differs.
    `request_id`
                the dialect's wire id (chatcmpl-…, msg_…, resp_…), minted
                when the request is built so a log line and the reply name
                the same generation.
    `images`    the request's images, decoded and preprocessed by the
                dialect outside the generation lock (vision.Vision.prepare),
                in template order: images[k] is the k-th `{"type": "image"}`
                part across `messages`, first message first. Empty for a
                text request, and then nothing about the generation differs
                from a request built without the field."""

    messages: list[dict]
    sampling: SampleParams
    tools: list | None = None
    template_kwargs: dict | None = None
    stream: bool = False
    request_id: str = ""
    images: tuple[PreparedImage, ...] = ()


@dataclass
class GenResult:
    text: str  # what was DELIVERED through on_delta, not what was sampled
    finish_reason: str  # "stop" | "length" | "abort" | "tool_calls"
    prompt_tokens: int
    completion_tokens: int
    # Parsed tool calls ({"name": str, "arguments": dict}) in emission order,
    # None when the model called nothing. MODEL-shaped, not wire-shaped: the
    # HTTP layer owns the OpenAI rendering (ids, arguments-as-JSON-string),
    # the same way this seam never leaks tokenizers upward.
    tool_calls: list | None = None
    # Prompt tokens whose KV was reused from the engine's prefix cache instead
    # of re-prefilled (engines.py). Surfaces as OpenAI's
    # usage.prompt_tokens_details.cached_tokens; 0 = nothing reused.
    cached_tokens: int = 0
    # The stop string that ended a finish_reason "stop" generation, else None.
    # The OpenAI wire never asks which one; Anthropic's does (`stop_sequence`
    # beside stop_reason "stop_sequence"), and only the scanner knows.
    stop_sequence: str | None = None


@dataclass(frozen=True)
class StreamStart:
    """The FIRST event of every generate() stream, before any forward pass:
    the prompt has been rendered and the engine knows its shape. The
    explicit stream-start fact.

    `opens_think`: did the rendered prompt end inside an open think block
    (Qwen3's `<think>\\n`, gemma's `<|channel>thought`)? The one fact the
    serve loop cannot recover from the token stream: the model's reply then
    starts in reasoning and only the CLOSING tag crosses the wire, and
    serving/think.py needs to know which channel byte one belongs to. A fact
    about THIS prompt — the same engine answers differently the moment a
    request turns thinking off.

    `cached_tokens`: prompt tokens whose KV the engine's prefix cache is
    about to reuse instead of re-prefilling (0 = nothing reused).

    NOT the TTFT mark: TTFT stays "the first delta carrying reasoning or
    content" (serving/metrics.py), measured by http.py's generation core."""

    prompt_tokens: int
    opens_think: bool = False
    cached_tokens: int = 0


@dataclass(frozen=True)
class Delta:
    """A piece of delivered text, in stream order. The consumer may answer
    it with `send(False)` to abort; plain iteration (or `send(True)`/`None`)
    continues."""

    text: str


@dataclass(frozen=True)
class Finished:
    """The LAST event: the generation's GenResult, including on abort."""

    result: GenResult


GenEvent = StreamStart | Delta | Finished


def complete(eng: "Engine", req: GenerationRequest,
             on_delta: Callable[[str], bool] | None = None) -> GenResult:
    """Drive one generate() stream to its end and return the GenResult.
    `on_delta(text) -> bool` sees every Delta in order; a falsy return
    aborts the generation exactly as a vanished client would (the engine
    answers with a finish_reason "abort" result) — the on_delta contract
    the seam had before GenerationRequest, unchanged for its callers. None = collect
    only."""
    stream = eng.generate(req)
    ok = None
    while True:
        try:
            ev = stream.send(ok)
        except StopIteration:
            raise RuntimeError(f"{type(eng).__name__}.generate ended without a "
                               "Finished event") from None
        if isinstance(ev, Finished):
            return ev.result
        ok = True if on_delta is None or not isinstance(ev, Delta) else bool(on_delta(ev.text))


class SleepError(Exception):
    """A sleep or wake that cannot be done, phrased for a 4xx line.

    On the SEAM rather than in serving/sleep.py because serving/http.py has
    to catch it and serving/http.py imports no torch — its module docstring's
    posture, and the thing DRINKME_FAKE_ENGINE=1 ("no model, no pack, no
    GPU") depends on. serving/sleep.py owns the tensors; this file owns the
    vocabulary that crosses."""


@dataclass
class SleepState:
    """SLEEP / WAKE bookkeeping (sleep/wake), one per engine — what /health, /sleep
    and /wake_up all report, so an operator never learns two vocabularies for
    one fact.

    Mutated in place under the generation lock, read locklessly by /health,
    which is why every field is a scalar or a small dict replaced whole: a
    reader can catch a stale field but never a half-written one."""

    level: int = 0  # 0 = awake, 1 = weights in host RAM, 2 = weights freed
    since: float | None = None  # epoch seconds the current sleep began
    tensors: int = 0
    bytes_moved: int = 0
    by_class: dict = field(default_factory=dict)
    slots_persisted: int = 0
    slots_dropped: int = 0
    slots_restored: int = 0
    took_s: float = 0.0  # the last sleep's wall clock
    wake_s: float | None = None  # the last wake's wall clock

    @property
    def asleep(self) -> bool:
        return self.level > 0

    def as_dict(self) -> dict:
        d = {"state": "asleep" if self.asleep else "awake", "level": self.level}
        if self.asleep:
            d["asleep_since"] = int(self.since or 0)
            d["asleep_s"] = int(time.time() - (self.since or time.time()))
            d["parked_bytes"] = self.bytes_moved
            d["tensors"] = self.tensors
            d["by_class"] = dict(self.by_class)
            d["slots_persisted"] = self.slots_persisted
            d["slots_dropped"] = self.slots_dropped
            d["took_s"] = round(self.took_s, 3)
        elif self.wake_s is not None:
            d["woke_in_s"] = round(self.wake_s, 3)
            d["slots_restored"] = self.slots_restored
        return d


class Engine(Protocol):
    """What serving/http.py serves. model_meta() is the `drinkme` object of
    the model's /v1/models entry (http.py wraps OpenAI's four top-level
    fields around it) and must never block behind a generation —
    provenance is read at load time, not computed on demand. Its
    `capabilities` carry `thinking` ("open"|"closed"|"always"|"none") and
    `tool_format` ("qwen-xml"|"json"|"unknown"|"none") — serving/capability.py's
    static per-model announcement, probed once at construction and cached,
    never recomputed per request; `sampling.defaults` is the effective
    sampling table and `sampling.profile` the generation profile of a
    `<model>:<profile>` entry (None on the bare one).

    DESIGN REQUIREMENT for any future engine on shared/unified memory
    (esp. a Metal engine on a laptop): CHECK the fit. Compare container
    size + generation headroom against the device's working-set ceiling
    (e.g. mx.device_info()'s max_recommended_working_set_size) and REFUSE to
    load with the arithmetic printed, exit non-zero, no bypass flag — a
    headroom knob may only make the check stricter. A tool that can wedge
    the host should decline rather than try. (A 4B load attempt on an 8GB
    MacBook Air can exhaust memory and suspend the session — the shape this
    repo's own check.py refuses against.)

    generate(req) is a generator: StreamStart first (the rendered prompt's
    shape, including whether it ends inside an open `<think>` block —
    Qwen3's template does, so the stream begins in reasoning and only the
    CLOSING tag crosses the wire, and the HTTP layer cannot tell from the
    tokens alone which channel byte one belongs to, serving/think.py), then
    a Delta per delivered piece, then Finished with the GenResult. A
    consumer aborts by answering a Delta with send(False). An engine that
    goes through template.render_prompt gets the verdict for free; one that
    builds no prompt (FakeEngine) reports what it stands in for. Every
    template kwarg the prompt is rendered with is in the request
    (GenerationRequest.template_kwargs, merged over the engine's own
    generation_config.json defaults by template.effective_kwargs); nothing
    ambient can change what a caller gets.

    OPTIONAL, discovered by getattr and NOT part of this Protocol:
    `persist_slots()` (on-disk prefix slots), `sleep(level)` / `wake()` / `sleep_state()`
    (sleep/wake), and `prefix_cache_state()` (the `prefix_cache` field on
    /health — 0 slots means the reuse path is off). serve.py's `_persist_slots`
    set the precedent and its docstring carries the reasoning — a KV cache is
    an HFEngine detail, not something
    every Engine must own — and sleep is the same shape: http.py asks whether
    this engine can be parked and answers a legible 501 when it cannot,
    rather than obliging every future engine to implement a level-2 reload.
    FakeEngine implements all three anyway, because the HTTP CONTRACT they
    drive (the state on /health, the 503 on a generate route while asleep,
    the 409 while a generation holds the lock) has to be testable with no
    model, no pack and no GPU."""

    model_id: str

    def model_meta(self) -> dict: ...

    def generate(self, req: GenerationRequest) -> Iterator[GenEvent]: ...

    def count_tokens(self, req: GenerationRequest) -> int:
        """Prompt length, as generate(req) would render it — the same
        messages, tools and template kwargs, the same tokenizer, just
        counted. Touches no model state and no device, so the HTTP layer
        calls it WITHOUT the generation lock (the request-time context
        check, /v1/messages/count_tokens, and the input count a streamed
        Messages reply announces up front)."""
        ...

    def tokenize(self, prompt: str | None = None, messages: list[dict] | None = None,
                 tools: list | None = None,
                 template_kwargs: dict | None = None) -> list[int]:
        """The `/tokenize` route (a tokenizer route): exactly one of `prompt` (raw text,
        no template) or `messages` (rendered through the SAME path generate()
        uses — `template_kwargs` over the engine's own defaults, thinking
        defaults included; the route passes none) -> the token ids.
        count_tokens is `len(tokenize(...))` by construction on every real
        engine, so the two can never drift apart into two different opinions
        about one prompt's cost."""
        ...

    def detokenize(self, tokens: list[int]) -> str:
        """The `/detokenize` route (a tokenizer route): token ids -> text. Touches no
        model state and no device — lockless, like count_tokens."""
        ...

    def tokenizer_info(self) -> dict:
        """The `/tokenizer_info` route (a tokenizer route): bos/eos token ids and whether
        a chat template exists — the tokenizer facts model_meta()'s
        capability announcement (serving/capability.py) does not
        already carry. http.py merges this with model_meta()'s
        thinking/tool_format/contextWindow, so nothing is asked twice."""
        ...


@dataclass
class FakeEngine:
    """Deterministic Engine for tests and the dev-only serve path
    (DRINKME_FAKE_ENGINE=1): echoes the last user message (or a canned reply)
    one word-delta at a time, honoring max_tokens, stop, and abort exactly as
    a real engine must — the HTTP tests that pass against this are the
    contract the real engines inherit. One word == one token, so usage
    numbers are checkable by eye."""

    model_id: str = "drinkme-fake"
    reply: str | None = None  # None -> "echo: <last user message>"
    # Raw stream override, <tool_call> blocks and all, word-split into deltas
    # and run through the SAME ToolCallScanner path a real engine uses — the
    # HTTP tool tests exercise the real conversion, not a canned GenResult.
    tool_call_script: str | None = None
    # No prompt is built here, so nothing opens a `<think>` block and the
    # canned text is content. True stands in for the Qwen3 prompt shape —
    # only `</think>` in the stream — for tests that want the think channel
    # without a tokenizer; a literal `<think>` in the reply needs no flag.
    opens_think: bool = False
    delay: float = 0.0  # per-delta sleep; tests use it to hold a generation open
    # sleep once, before the FIRST delta — stands in for a real prefill's
    # forward-pass latency (`delay` sleeps AFTER a delta, so it cannot model
    # "the client is still waiting for token one"). The SSE keep-alive's tests
    # hold a generation in this state and assert on what reaches the wire
    # before anything else does.
    prefill_delay: float = 0.0
    aborted: bool = False  # latched by the LAST generate(); disconnect tests read it
    calls: int = 0
    last_tools: list | None = None  # what the LAST generate() received; tests read it
    # serving/capability.py's announcement. Defaults stand in for a
    # well-behaved menu model — "json" + "open" — so every EXISTING tool/think
    # test keeps passing unmodified; tests that want the refusal path or the
    # /v1/models fields override these explicitly (fake(tool_format="unknown")).
    thinking: str = "open"
    tool_format: str = "json"
    # Advertised context / the context-length check: the ONE context field (real engines publish contextWindow
    # from HFEngine.model_meta; advertised-ctx scaling and the request-time
    # length check both read model_meta()['contextWindow']). None = no ctx
    # published, the request-time context check no-ops
    # (every existing test keeps its unbounded FakeEngine behavior); tests
    # that want the 400 set this explicitly (fake(context_window=16384)).
    context_window: int | None = None
    # The tokenizer routes' tokenizer_info stand-ins; overridable the same way thinking/
    # tool_format are.
    bos_token_id: int | None = None
    eos_token_id: int | None = 0
    has_chat_template: bool = True
    # tokenize()/detokenize()'s reversible word<->id vocabulary, built as
    # tokenize() sees new words — lets a route test round-trip without a
    # real tokenizer. Not part of the engine's identity, so out of repr/eq.
    _vocab: dict = field(default_factory=dict, repr=False, compare=False)
    # generation_config.json defaults: stands in for an engine's GenDefaults.sampling (engines.py) — the
    # override subset a generation_config.json would have set. {} is every
    # existing test's behavior, unchanged: OpenAI defaults all the way.
    defaults: dict = field(default_factory=dict)
    # Sleep/wake bookkeeping: the SAME serving/sleep.SleepState the real
    # engine keeps, with nothing to move. What this stands in for is the
    # CONTRACT — /health's state field, the 503 on a generate route while
    # asleep, the 409 when a generation holds the lock — which is HTTP-layer
    # behaviour and must be testable without a model. Out of repr/eq for the
    # same reason _vocab is: it is not part of the engine's identity.
    _sleep: object = field(default=None, repr=False, compare=False)
    # set when a test wants the level-2 path: HFEngine's `_reload` is the
    # loader closure, and level-2 wake refuses without one.
    _reload: object = field(default=None, repr=False, compare=False)

    def _state(self) -> SleepState:
        if self._sleep is None:
            self._sleep = SleepState()
        return self._sleep

    def sleep_state(self) -> dict:
        return self._state().as_dict()

    def sleep(self, level: int = 1) -> dict:
        """HFEngine.sleep's shape with no tensors: the state machine, the
        levels, the idempotence and the refusal to un-deepen, so the HTTP
        contract they drive is exercised by every test in
        tests/test_serving_http.py without a GPU or a pack."""
        if level not in (1, 2):
            raise SleepError(f"sleep level {level} is not 1 or 2")
        st = self._state()
        if st.level == level:
            return st.as_dict()
        if st.asleep and level < st.level:
            raise SleepError(f"already asleep at level {st.level}; sleep only "
                             "deepens — POST /wake_up first")
        st.level, st.since = level, time.time()
        return st.as_dict()

    def wake(self) -> dict:
        st = self._state()
        if not st.asleep:
            return st.as_dict()
        if st.level == 2 and self._reload is None:
            raise SleepError("level-2 wake needs the loader that built this "
                             "engine; this engine was constructed directly")
        t0 = time.perf_counter()
        if st.level == 2:
            self._reload()
        st.level, st.since = 0, None
        st.wake_s = time.perf_counter() - t0
        return st.as_dict()

    def model_meta(self) -> dict:
        from .capability import Capability  # local: no import cycle at module load
        from .gen_config import effective_defaults

        return {"arm": "fake", "runtime": "fake", "hfRepo": None,
                "revision": None, "compressionProfile": None, "bitsPerWeight": None,
                "meanTensorBitsPerWeight": None, "sourceDtype": None,
                "contextWindow": self.context_window,
                "device": None,  # no device behind the fake; /health said null before, still does
                # THE real dict shape (capability.Capability.as_dict), not a
                # second one to keep in sync — the fake's thinking/tool_format
                # fields are the only inputs, so real engines and this one
                # announce `thinking_switch` from the same code.
                "capabilities": Capability(self.thinking, self.tool_format).as_dict(),
                "sampling": {"profile": None, "defaults": effective_defaults(self.defaults)}}

    def count_tokens(self, req: GenerationRequest) -> int:
        """One word == one token, over every turn's content — the same sum
        generate() reports as prompt_tokens, so count_tokens and usage agree."""
        return _fake_prompt_tokens(req.messages)

    def tokenize(self, prompt: str | None = None, messages: list[dict] | None = None,
                 tools: list | None = None,
                 template_kwargs: dict | None = None) -> list[int]:
        text = prompt if prompt is not None else " ".join(
            str(m.get("content", "")) for m in (messages or []))
        return [self._vocab.setdefault(w, len(self._vocab)) for w in text.split()]

    def detokenize(self, tokens: list[int]) -> str:
        inv = {i: w for w, i in self._vocab.items()}
        return " ".join(inv.get(t, "<unk>") for t in tokens)

    def tokenizer_info(self) -> dict:
        return {"bos_token_id": self.bos_token_id, "eos_token_id": self.eos_token_id,
                "chat_template": self.has_chat_template}

    def generate(self, req: GenerationRequest) -> Iterator[GenEvent]:
        from .sampling import StopScanner  # local: no import cycle at module load
        from .tools import ToolCallScanner

        messages, params, tools = req.messages, req.sampling, req.tools
        self.calls += 1
        self.aborted = False
        self.last_tools = tools
        prompt_tokens = _fake_prompt_tokens(messages)
        # what render_prompt would have said for a real engine
        yield StreamStart(prompt_tokens, opens_think=self.opens_think)
        if self.prefill_delay:
            time.sleep(self.prefill_delay)
        last = ""
        for m in messages:
            if m.get("role") == "user":
                c = m.get("content", "")
                last = c if isinstance(c, str) else json.dumps(c)
        text = (self.tool_call_script if self.tool_call_script is not None
                else self.reply if self.reply is not None else f"echo: {last}")
        # Scanner order mirrors HFEngine: tool extraction FIRST, so tool-call
        # text never reaches StopScanner or the client as visible content.
        # Active only when tools were offered — without tools, <tool_call>
        # text is just text, byte-identical to the default.
        toolscan = (ToolCallScanner(tools, self.tool_format) if tools else None)
        parsed_calls: list[dict] = []
        scan = StopScanner(params.stop)
        sent: list[str] = []
        finish = "stop"
        n = 0
        for delta in re.findall(r"\S+\s*", text):  # words, spaces kept -> concat == text
            if n == params.max_tokens:
                finish = "length"
                break
            n += 1
            if toolscan is not None:
                delta, done = toolscan.feed(delta)
                parsed_calls += done
            out = scan.feed(delta)
            if out:
                if (yield Delta(out)) is False:
                    self.aborted, finish = True, "abort"
                    break
                sent.append(out)
            if scan.stopped:
                break
            if self.delay:
                time.sleep(self.delay)
        if finish == "stop" and not scan.stopped:
            # held-back tails, in stream order: the tool scanner's (an
            # un-terminated block re-emerging as text) feeds the stop
            # scanner's, then both flush — it is all generated text
            tail = (scan.feed(toolscan.flush()) if toolscan is not None else "")
            if toolscan is not None:  # a one-sided row resolves its call here
                parsed_calls += toolscan.flush_calls()
            tail += scan.flush()
            if tail:
                if (yield Delta(tail)) is False:
                    self.aborted, finish = True, "abort"
                else:
                    sent.append(tail)
        if parsed_calls and finish == "stop":
            finish = "tool_calls"  # Qwen's shape: emit calls, then stop
        yield Finished(GenResult("".join(sent), finish, prompt_tokens, n,
                                 tool_calls=parsed_calls or None, stop_sequence=scan.matched))


def _fake_prompt_tokens(messages: list[dict]) -> int:
    return sum(len(str(m.get("content", "")).split()) for m in messages)
