"""Logits -> next token, and stop-sequence scanning on the decoded text.

Pure and CPU-testable: sample_next takes a 1-D logits tensor and returns an
int, nothing else moves. torch is imported inside the function (pack.py's
pattern) so the rest of serving/ — FakeEngine, the HTTP layer — imports
without torch in the path.

PENALTY STATE. Rebuilding the penalties from the id LISTS every step —
sorted(set(prev_ids)), a counts dict over gen_ids, a fresh CPU tensor, a
host-to-device copy — is per-token work over a window that grows with the
generation, for a quantity that changes by exactly one entry per step.
PenaltyState keeps the same three facts on the DEVICE (a
seen mask for the repetition window, a seen mask for the generated window, a
counts vector) and advances them with one scatter per ACCEPTED token. The
arithmetic is deliberately unchanged — same operations in the same order on
the same values, so the processed logits are BITWISE what the list path
produced (tests/test_hotloop_equivalence.py asserts torch.equal, not
allclose). Callers that pass no state keep the list path, byte for byte.

SCRATCH: a call that allocated two full-vocabulary f32 tensors
(.to(float32) then .clone()) would do so twice per CANDIDATE in the
constrained loop. A SampleScratch lends the same two buffers for the life
of an engine.

THE DISTRIBUTION, NOT JUST THE DRAW (rejection-sampling MTP).
sample_next's pipeline is two halves — penalties on the raw row, then
temperature / top_k / top_p — followed by one multinomial draw. Speculative
decoding needs the DISTRIBUTION that draw comes from, for both the target
rows and the draft rows, so the halves are factored out (`_penalize`,
`_shape`) and `sample_probs` returns the softmax the serial sampler would
have drawn from. Same ops, same order, same values: `sample_next` is
literally `multinomial(sample_probs(...))` for temperature > 0, and the
hot-loop equivalence tests pin it token-for-token against the oracle
(tests/hotloop_oracle.py). Greedy (temperature 0) has no distribution and never
takes this path.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterable

if TYPE_CHECKING:  # typing only — engine.py imports StopScanner, avoid the cycle
    from .engine import SampleParams


class SampleScratch:
    """Preallocated [V] float32 working buffers, one set per engine.

    `work` is sample_next's own copy of the row; `ban` is the constrained
    loop's copy, the one that accumulates -inf bans across candidates. They
    are separate because pick_token's copy has to survive every sample_next
    call inside its loop. Buffers are (re)allocated only when the shape,
    device or dtype of the row changes — normally once, ever.
    """

    __slots__ = ("work", "ban")

    def __init__(self) -> None:
        self.work = None
        self.ban = None

    def _lend(self, name: str, logits):
        import torch

        buf = getattr(self, name)
        if (buf is None or buf.shape != logits.shape
                or buf.device != logits.device):
            buf = torch.empty_like(logits, dtype=torch.float32)
            setattr(self, name, buf)
        buf.copy_(logits)  # same conversion .to(torch.float32) did, no alloc
        return buf

    def take_work(self, logits):
        return self._lend("work", logits)

    def take_ban(self, logits):
        return self._lend("ban", logits)


class PenaltyState:
    """The penalty windows of ONE generation, resident on the device.

    `prev` is the repetition-penalty window (prompt + generated), `gen` and
    `counts` are the OpenAI presence/frequency window (generated only) — the
    same three facts sample_next's list path rebuilds from python lists every step.
    Everything is allocated once and advanced by one accepted token at a time;
    nothing here is derived from params, so a state cannot fall out of step
    with the knobs a request happens to set.
    """

    __slots__ = ("prev", "gen", "counts", "_idx", "_one")

    def __init__(self, vocab: int, device=None):
        import torch

        self.prev = torch.zeros(vocab, dtype=torch.bool, device=device)
        self.gen = torch.zeros(vocab, dtype=torch.bool, device=device)
        self.counts = torch.zeros(vocab, dtype=torch.float32, device=device)
        # one-element scratch so an accepted token costs no allocation
        self._idx = torch.zeros(1, dtype=torch.long, device=device)
        self._one = torch.ones(1, dtype=torch.float32, device=device)

    def observe(self, ids) -> None:
        """The prompt: seen for repetition, not for presence/frequency (which
        are over generated text only — vLLM's reading, engine.py)."""
        import torch

        ids = list(ids)
        if not ids:
            return
        idx = torch.tensor(ids, dtype=torch.long, device=self.prev.device)
        self.prev.index_fill_(0, idx, True)

    def accept(self, tok: int) -> None:
        """One token joined the output: three scatters, no host round trip."""
        self._idx.fill_(tok)
        self.prev.index_fill_(0, self._idx, True)
        self.gen.index_fill_(0, self._idx, True)
        self.counts.index_add_(0, self._idx, self._one)

    def apply(self, logits, params: SampleParams) -> None:
        """The penalties, in place. Every arithmetic op is the one the list
        path ran, on the same values: HF's divide-positive/multiply-negative
        for repetition, then OpenAI's additive presence + frequency*count."""
        import torch

        rp = params.repetition_penalty
        if rp != 1.0:
            # `where` over the whole row instead of a gather/scatter over the
            # seen ids: same per-element result, and the seen set never has to
            # cross the bus.
            torch.where(self.prev,
                        torch.where(logits > 0, logits / rp, logits * rp),
                        logits, out=logits)
        pp, fp = params.presence_penalty, params.frequency_penalty
        if pp or fp:
            torch.where(self.gen, logits - (pp + fp * self.counts), logits,
                        out=logits)


def _penalize(logits, params: SampleParams, prev_ids, gen_ids, state, scratch):
    """The first half of the pipeline: a PRIVATE float32 copy of the row with
    the penalties applied, in place. Repetition penalty on raw logits (over
    prev_ids = prompt + generated), then the additive OpenAI presence /
    frequency penalties (over gen_ids = generated only) — the HF/vLLM order.
    `state` replaces the id lists with device-resident accumulators and
    `scratch` replaces the allocation; neither changes an arithmetic result."""
    import torch  # local: keep serving importable without torch (see module docstring)

    if scratch is not None:
        logits = scratch.take_work(logits)
    else:
        logits = torch.as_tensor(logits).to(torch.float32).clone()
    if state is not None:
        state.apply(logits, params)
    elif prev_ids is not None and params.repetition_penalty != 1.0:
        ids = sorted({int(i) for i in prev_ids})
        if ids:
            idx = torch.tensor(ids, dtype=torch.long)
            picked = logits[idx]
            # HF convention: DIVIDE positive logits, MULTIPLY negative — both
            # directions push a seen token down; a plain divide would push a
            # negative logit UP.
            logits[idx] = torch.where(picked > 0,
                                      picked / params.repetition_penalty,
                                      picked * params.repetition_penalty)
    if state is None and gen_ids is not None and (params.presence_penalty
                                                  or params.frequency_penalty):
        counts: dict[int, int] = {}
        for i in gen_ids:
            counts[int(i)] = counts.get(int(i), 0) + 1
        if counts:
            ids = sorted(counts)
            idx = torch.tensor(ids, dtype=torch.long)
            # ON THE ROW'S DEVICE: `logits[idx] -= <cpu>` on a CUDA row
            # raises a device-mismatch RuntimeError, so a CPU index tensor
            # here would 500 every GPU request that sets presence/frequency,
            # and a CPU-only test would never see it (the devices agree
            # there). Same values, same arithmetic — f32 ops are correctly
            # rounded on both devices — so the CPU lockstep tests vs the
            # verbatim oracle hold. This shared list path also serves MTP's
            # pick_row (which builds no PenaltyState), so the state path
            # alone would not cover greedy + penalties under MTP.
            cnt = torch.tensor([counts[i] for i in ids], dtype=torch.float32,
                               device=logits.device)
            # OpenAI semantics: presence is a flat tax on any seen token,
            # frequency scales with how often it appeared
            logits[idx] -= params.presence_penalty + params.frequency_penalty * cnt
    return logits


def _shape(logits, params: SampleParams) -> None:
    """The second half, in place, for temperature > 0: temperature, then
    top_k, then top_p. What is left is the pre-softmax row the draw uses."""
    import torch

    logits.div_(params.temperature)
    if 0 < params.top_k < logits.numel():
        kth = torch.topk(logits, params.top_k).values[-1]
        logits[logits < kth] = float("-inf")  # ties at the kth value survive (HF-same)
    if params.top_p < 1.0:
        srt, idx = torch.sort(logits, descending=True)
        probs = torch.softmax(srt, dim=-1)
        cum = torch.cumsum(probs, dim=-1)
        # cum - probs is the mass BEFORE each token: drop a token only when the
        # threshold was already reached without it, so the crossing token stays
        # and at least one token always survives.
        srt[cum - probs > params.top_p] = float("-inf")
        logits.fill_(float("-inf")).scatter_(0, idx, srt)


def sample_next(logits, params: SampleParams, generator=None,
                prev_ids: Iterable[int] | None = None,
                gen_ids: Iterable[int] | None = None,
                state: PenaltyState | None = None,
                scratch: SampleScratch | None = None) -> int:
    """One decode step: logits [V] float -> token id.

    Order matters and is the HF/vLLM convention: repetition penalty on raw
    logits (over prev_ids = prompt + generated), then the additive OpenAI
    presence/frequency penalties (over gen_ids = generated only), then
    temperature, then top_k, then top_p, then sample. Determinism comes from
    the caller's seeded torch.Generator; temperature == 0 is greedy and
    ignores the generator entirely.

    `state` (a PenaltyState) replaces the id lists with device-resident
    accumulators and `scratch` (a SampleScratch) replaces this call's two
    full-vocabulary allocations; both are optional and neither changes a
    single arithmetic result. The row this function works on is always a
    private copy, so the pipeline below runs in place.
    """
    import torch  # local: keep serving importable without torch (see module docstring)

    logits = _penalize(logits, params, prev_ids, gen_ids, state, scratch)
    if params.temperature == 0:
        return int(torch.argmax(logits))
    _shape(logits, params)
    return int(torch.multinomial(torch.softmax(logits, dim=-1), 1, generator=generator))


def sample_probs(logits, params: SampleParams,
                 prev_ids: Iterable[int] | None = None,
                 gen_ids: Iterable[int] | None = None,
                 state: PenaltyState | None = None,
                 scratch: SampleScratch | None = None):
    """The distribution sample_next draws from, as a fresh [V] float32 row.

    THE transform for rejection-sampling MTP (serving/speculative.py): the
    target rows p and the draft rows q both come through here, with the
    penalty history the serial loop would have had at that position, so the
    draft samples from exactly the q the accept rule later divides by and p
    is exactly what the serial sampler would have used. Same two halves as
    sample_next, same values — `multinomial(sample_probs(row), 1, g)` and
    `sample_next(row, ..., generator=g)` pick the same token from the same
    generator state (tests/test_serving_speculative.py pins that).

    temperature 0 is refused rather than one-hotted: greedy has no
    distribution to divide by, and MTP's greedy path is the argmax loop.
    """
    import torch

    if params.temperature == 0:
        raise ValueError("sample_probs: temperature 0 is greedy and has no "
                         "distribution — the speculative path is for sampled "
                         "requests; greedy takes the argmax accept loop")
    logits = _penalize(logits, params, prev_ids, gen_ids, state, scratch)
    _shape(logits, params)
    return torch.softmax(logits, dim=-1)


class StopScanner:
    """Stop sequences on DECODED text, robust to a stop string arriving split
    across deltas ("EN" then "D"). feed() returns only text that is provably
    before any stop: a tail of len(longest stop) - 1 chars is held back until
    the next delta rules a match out, and flush() releases it at end of
    generation. After a match, feed() returns "" forever, .stopped is True and
    .matched names the stop string that fired (the earliest match; on a tie,
    the first in request order) — Anthropic's stop_reason vocabulary reports
    it as `stop_sequence`. The stop string itself is never emitted."""

    def __init__(self, stops: list[str] | None):
        self.stops = [s for s in (stops or []) if s]
        self._hold = max((len(s) for s in self.stops), default=1) - 1
        self._buf = ""
        self.stopped = False
        self.matched: str | None = None

    def feed(self, delta: str) -> str:
        if self.stopped:
            return ""
        self._buf += delta
        hits = [(i, s) for s in self.stops if (i := self._buf.find(s)) >= 0]
        if hits:
            at, self.matched = min(hits, key=lambda h: h[0])  # min is stable: first wins a tie
            out, self._buf, self.stopped = self._buf[:at], "", True
            return out
        if len(self._buf) <= self._hold:
            return ""
        cut = len(self._buf) - self._hold
        out, self._buf = self._buf[:cut], self._buf[cut:]
        return out

    def flush(self) -> str:
        """The held-back tail, once generation is over and it cannot start a stop."""
        out, self._buf = ("", "") if self.stopped else (self._buf, "")
        return out
