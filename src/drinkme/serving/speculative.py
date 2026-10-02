"""Rejection-sampling speculative decoding: the accept/resample rule that
makes MTP exact at ANY temperature, and the acceptance window behind the
adaptive bail. Pure functions over probability rows — no model, no cache, no
device assumptions beyond "the rows live somewhere torch can index".

THE CONSTRUCTION (Leviathan et al. 2023, Chen et al. 2023). For each draft
position i the target model gives p_i (the distribution the serial sampler
would draw from at that position, penalties/temperature/top-k/top-p already
applied) and the draft gave q_i (the distribution it ACTUALLY sampled d_i
from). Accept d_i with probability min(1, p_i[d_i] / q_i[d_i]). On the first
rejection emit ONE token from the normalized residual max(0, p_i - q_i) and
stop. If every draft is accepted, emit a bonus token from p_depth. The
emitted token at each position is then distributed EXACTLY as p_i, for any
q_i whatsoever — a draft can only cost acceptance, never change what is
emitted in distribution. FOR ANY q_i is essential, not decoration:
it is what lets serving/ngram.py's lookup, whose q is a point mass
(point_mass_rows below), share this rule instead of getting its own.
That is the theorem; tests/test_serving_speculative.py
checks it empirically (200k seeded steps per case, total variation against
p_0 and against p_1 conditioned on the first draft being accepted) for q == p,
q far from p, q with zero mass where p has mass, and p's support inside q's.

WHAT "LOSSLESS" MEANS FROM HERE. The greedy MTP path is byte-identical to the
serial loop (same argmax per row). The sampled path is DISTRIBUTION-identical:
a seed-matched MTP transcript is not the seed-matched serial transcript (the
two consume the generator differently), but every token is drawn from the
distribution the serial sampler would have used. The verification shape
changes with that claim — property proof here, seed-matched statistical
checks on the real model after merge — and so does the acceptance accounting,
which is what the adaptive bail reads.

Zero-mass residual: p_i - q_i is nonnegative-and-nonzero somewhere whenever
p_i != q_i (both sum to one), so an empty residual can only follow p_i == q_i,
where the accept rule fires with certainty (u < 1). The branch is reachable
only through float rounding or a draft token outside q's support, both of
measure ~0; it falls back to sampling p_i, which is the right answer at that
point and never a crash.

THE AUDIT (docs/serve-speculation.md). The two-arm first-token check
(bench/mtp_sampled_dist.py) has no power once the drafter is good: with q≈p
the MTP arm's first token IS the serial arm's generator draw, so a wrong rule
that accepted everything would score identically. `speculative_sample`'s
optional `audit` sink is the self-referential replacement — it runs under
real traffic with no control arm, checking the theorem against itself: the
accept probability the theorem promises (sum(min(p, q))) against what
actually got accepted, and the two invariants a resampled or bonus token must
satisfy (positive mass under p; for a resample, more mass under p than under
q — that is what put it in the residual). None (default) touches nothing
extra: no tensor op the day's code path didn't already do, and the generator
draws are identical either way — SpeculativeAudit.observe* never calls into
`generator`.
"""

from __future__ import annotations

from collections import deque


class SpeculativeAudit:
    """Running sums an `audit` sink accumulates as speculative_sample decides
    each draft position — the theorem checked live, module docstring. One
    instance covers however many calls it is given: a Speculator keeps one
    for the request and mtp.py keeps one across requests (A2/A3,
    docs/serve-speculation.md); a test can own one directly, as
    tests/test_serving_speculative.py does."""

    def __init__(self):
        self.decided = 0
        self.accepted = 0
        self.expected_sum = 0.0
        self.resample_violations = 0
        self.zero_mass_emits = 0

    @property
    def actual_rate(self) -> float:
        return self.accepted / self.decided if self.decided else 0.0

    @property
    def expected_rate(self) -> float:
        return self.expected_sum / self.decided if self.decided else 0.0

    @property
    def violations(self) -> int:
        return self.resample_violations + self.zero_mass_emits

    def observe(self, expected: float, actual: bool) -> None:
        """One draft position DECIDED (reached, whichever way it went): the
        theorem's own acceptance probability sum(min(p, q)) for that
        position, regardless of which token was drafted, and whether it was
        actually accepted."""
        self.decided += 1
        self.accepted += int(actual)
        self.expected_sum += expected

    def observe_resample(self, pi, qi, t: int) -> None:
        """The token emitted at a rejection came from the residual
        max(0, p - q): it must have p[t] > q[t] (that is what put it in the
        residual) and p[t] > 0 (it must have mass under the target)."""
        if not bool(pi[t] > qi[t]):
            self.resample_violations += 1
        if not bool(pi[t] > 0):
            self.zero_mass_emits += 1

    def observe_bonus(self, p_depth, t: int) -> None:
        """The all-accept bonus token came from p_depth: it must have mass
        there too."""
        if not bool(p_depth[t] > 0):
            self.zero_mass_emits += 1


def _accept(u_i, pi, qi, d: int) -> bool:
    """accept iff u < min(1, p/q)  <=>  u * q < p  (q[d] > 0 for a token
    actually drawn from q; the product form also settles q[d] == 0 without a
    division: accept iff p gives the token any mass). A free function, not
    inlined, so a test can monkeypatch the decision itself — the audit's
    broken-rule test proves the instrument can fail before trusting it
    green (docs/serve-speculation.md)."""
    return bool(u_i * qi[d] < pi[d])


def speculative_sample(p, q, drafts: list[int], generator=None, audit=None) -> tuple[list[int], int]:
    """Decide one draft/verify cycle.

    p: depth+1 rows (a [M, V] tensor or a sequence of [V] rows) — the target
       distributions at positions 0..depth, position i conditioned on drafts
       0..i-1 having been accepted.
    q: depth rows — the distributions the drafts were sampled from.
    drafts: the depth draft tokens.
    generator: the request's torch.Generator (seeded from the request's seed
       or from entropy — engines.py). None = torch's default generator.
    audit: optional SpeculativeAudit (or anything with the same .observe /
       .observe_resample / .observe_bonus methods) — module docstring. None
       (the default) is the existing code path byte-for-byte.

    Returns (emitted, accepted): `emitted` is what the serial loop would emit
    next in distribution — `accepted` drafts followed by one token that is
    either the resample at the first rejection or the bonus from p[depth] —
    so it always holds accepted + 1 tokens, at least one.
    """
    import torch

    depth = len(drafts)
    if len(q) != depth or len(p) != depth + 1:
        raise ValueError(f"speculative_sample: {depth} drafts need {depth} draft "
                         f"rows and {depth + 1} target rows, got {len(q)} and {len(p)}")
    if depth:
        # one uniform per draft position, drawn up front from the request's
        # generator; the unused tail of a rejected cycle is simply discarded
        u = torch.rand(depth, generator=generator, device=p[0].device)
    for i, d in enumerate(drafts):
        pi, qi = p[i], q[i]
        accept = _accept(u[i], pi, qi, d)
        if audit is not None:
            audit.observe(float(torch.minimum(pi, qi).sum()), accept)
        if accept:
            continue
        resid = (pi - qi).clamp_(min=0)
        if not bool(resid.sum() > 0):
            resid = pi  # module docstring: measure-zero, and p is the answer
        tok = int(torch.multinomial(resid, 1, generator=generator))
        if audit is not None:
            audit.observe_resample(pi, qi, tok)
        return [*drafts[:i], tok], i
    tok = int(torch.multinomial(p[depth], 1, generator=generator))
    if audit is not None:
        audit.observe_bonus(p[depth], tok)
    return [*drafts, tok], depth


def point_mass_rows(like, drafts: list[int]) -> list:
    """q rows for a DETERMINISTIC proposer: one-hot at each drafted token.

    serving/ngram.py proposes by lookup, not by sampling — its q_i puts all
    of its mass on the one token it found. Substituting that q into the rule
    above: `_accept` becomes `u < p_i[d_i]` (q_i[d_i] == 1), the residual
    `(p_i - q_i).clamp(min=0)` becomes p_i with the mass at d_i removed, and
    the audit's own expected acceptance `sum(min(p, q))` becomes p_i[d_i].
    All three are the theorem's rule with a legal q, so every emitted token
    is distributed exactly as p_i — the same conclusion as for a sampled
    draft, and the reason a lookup can plug into this verify unchanged
    instead of getting a second one.

    Built DENSE, from `like` (a target row — same dtype, device and V), and
    built HERE rather than at proposal time, which is before any p row
    exists. The rows are the same shape and count as the p rows the cycle
    already allocated; a sparse q would need speculative_sample to know it
    was sparse, and this file's whole job is that the accept rule sees one
    kind of thing.
    """
    import torch

    rows = []
    for d in drafts:
        q = torch.zeros_like(like)
        q[d] = 1
        rows.append(q)
    return rows


class AcceptanceWindow:
    """Rolling acceptance over the last `size` DRAFT POSITIONS, for the
    adaptive bail: fall back to serial decode when rolling acceptance drops
    below the floor (mtp.BAIL_FLOOR, DRINKME_MTP_BAIL_FLOOR; break-even is
    about 5-7%, docs/serve-speculation.md).

    A cycle that drafted k and had m accepted contributes m ones and k - m
    zeros — the same accounting `Speculator.stats()` reports (accepted /
    drafted), so the rate that trips the bail is the rate the log line
    names. `observe` returns True exactly once, the first time a FULL window
    sits below the floor; a partial window never trips (eight bad cycles at
    depth 4 is evidence, one is not) and the trip latches — the caller has
    already switched to serial, there is nothing further to decide for THIS
    window. Speculation resuming is a fresh window (mtp.Speculator.rearm):
    the old one's evidence was about the region that tripped it."""

    def __init__(self, size: int = 32, floor: float = 0.15):
        self.size, self.floor = size, floor
        self._hits: deque[int] = deque(maxlen=size)
        self.tripped = False
        self.rate_at_trip: float | None = None

    @property
    def rate(self) -> float:
        return sum(self._hits) / len(self._hits) if self._hits else 0.0

    def observe(self, accepted: int, drafted: int) -> bool:
        if self.tripped or drafted <= 0 or self.floor <= 0:
            return False
        self._hits.extend([1] * accepted + [0] * (drafted - accepted))
        if len(self._hits) < self.size:
            return False
        rate = self.rate
        if rate < self.floor:
            self.tripped, self.rate_at_trip = True, rate
            return True
        return False
