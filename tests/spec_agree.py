"""Speculation-axis agreement: what two speculative-decoding arms of the
SAME model are held to. Token/transcript equality between them is not
achievable, and nothing in this repo checks for it.

What actually holds is the bar AGENTS.md already holds every other
arm-vs-arm comparison in this repo to: two numerically different paths
over the SAME weights agree modulo accumulation order, and a genuine
near-tie can flip an argmax for reasons that have nothing to do with a bug.
A batched verify forward over k+1 positions and a serial single-token
forward round differently for exactly that reason — the same reason a dense
kernel and a GEMV kernel, or a compressed weight and its bf16 original,
are held to "agrees except at a near-tie", never to bitwise transcript
identity.

`capture` hooks the one seam every decode path shares — `engines.sample_next`
— proven in tests/test_serving_gemma_spec.py's `_run_ids`: the id AND the
raw (pre-penalty) logits row land here for every emitted position, serial or
speculative, because the accept/reject loop's own per-row greedy decision
goes through it too. It is reliable for GREEDY runs (temperature 0) on
either arm; a NEAR-GREEDY (small but nonzero temperature, used to exercise
the SAMPLED rejection-sampling code path near-deterministically) run may
draw accepted speculative tokens through `speculative.speculative_sample`
instead, so its ids are read back off the decoded text — the same
text->ids round trip this test suite already relies on elsewhere (e.g.
tests/test_serving_speculative.py's `oracle = tok(ref.text)["input_ids"]").

KEPT UNTOUCHED, deliberately, elsewhere: the distribution-level rejection-
sampling audit (acceptance counters vs Sum min(p, q), violation counts,
tests/test_serving_ngram.py's histogram section) — that is a real property,
exact IN DISTRIBUTION, and is not a transcript check.
"""

from __future__ import annotations

import torch

# Generous versus the few-ULP accumulation noise actually measured between
# numerically different arms over the same weights (the greedy A/Bs in
# bench/, docs/dev-environment.md "Known behavior"): a real accept/reject bug forks at
# logit-scale margins, not
# fractional ones. Not a knob to widen quietly if a test gets noisy — that
# is a signal to look at the arm, not the epsilon.
DEFAULT_EPS_ULPS = 4096.0


def top2_margin_ulps(logits: torch.Tensor) -> float:
    """Top-2 logit gap, in the top logit's own fp32 ULPs — the measure the
    greedy A/Bs report, so "near-tie" means the same thing on every axis
    this repo checks it on."""
    row = logits.detach().float()
    top2 = torch.topk(row, 2).values
    gap = (top2[0] - top2[1]).abs()
    ulp = 2.0 ** (torch.floor(torch.log2(top2[:1].abs().clamp(min=1.0))) - 23)
    return float(gap / ulp)


def capture(module):
    """Patch `module.sample_next` to also record (id, raw logits row) per
    emitted position, keyed by generated-position exactly as `gen_ids`
    reports it. Returns (picks, rows, restore); call `restore()` when done
    (a plain function, not a context manager, so callers that already run
    under their own try/finally — most of these tests do, for other spies —
    can fold it in without nesting another `with`)."""
    picks: dict[int, int] = {}
    rows: dict[int, torch.Tensor] = {}
    real = module.sample_next

    def rec(logits, params, generator=None, prev_ids=None, gen_ids=None, **kw):
        pos = len(gen_ids) if gen_ids is not None else len(picks)
        rows[pos] = logits.detach().clone()
        t = real(logits, params, generator, prev_ids=prev_ids, gen_ids=gen_ids, **kw)
        picks[pos] = t
        return t

    module.sample_next = rec

    def restore():
        module.sample_next = real

    return picks, rows, restore


def ordered_ids(picks: dict[int, int]) -> list[int]:
    return [picks[i] for i in range(len(picks))]


def capture_sampled(mtp_module, engines_module):
    """`capture`'s twin for a SAMPLED (temperature > 0) speculative request.
    A sampled cycle's tokens come out of `mtp.speculative_sample` directly —
    they never reach `engines.sample_next` at all, unlike the greedy verify
    loop's per-row `pick` — so both seams are patched and merged into one
    call-ordered stream. The two never fire for the same position, and an
    adaptive bail can hand a request from one to the other mid-stream (a
    sampled MTP request that bails still finishes through plain
    `engines.sample_next` calls), which is why this appends in CALL order
    rather than keying by `gen_ids` the way `capture` does — the two hooks
    do not share that argument.

    Returns (ids, rows, restore): `ids`/`rows` are plain lists, index ==
    generated position by construction (one call, of either kind, is always
    exactly one more emitted token)."""
    ids: list[int] = []
    rows: list = []
    real_sample_next = engines_module.sample_next
    real_spec_sample = mtp_module.speculative_sample

    def rec_sample_next(logits, params, generator=None, prev_ids=None, gen_ids=None, **kw):
        rows.append(logits.detach().clone())
        t = real_sample_next(logits, params, generator, prev_ids=prev_ids, gen_ids=gen_ids, **kw)
        ids.append(t)
        return t

    def rec_spec_sample(p, q, drafts, generator=None, audit=None):
        emitted, m = real_spec_sample(p, q, drafts, generator, audit=audit)
        for i, t in enumerate(emitted):
            ids.append(t)
            rows.append(p[i] if i < len(p) else None)
        return emitted, m

    engines_module.sample_next = rec_sample_next
    mtp_module.speculative_sample = rec_spec_sample

    def restore():
        engines_module.sample_next = real_sample_next
        mtp_module.speculative_sample = real_spec_sample

    return ids, rows, restore


def first_fork(ids_a, ids_b) -> int | None:
    return next((i for i, (x, y) in enumerate(zip(ids_a, ids_b)) if x != y), None)


def _row_at(rows, i):
    """`rows[i]` whether `rows` is a dict keyed by position (`capture`) or a
    plain call-ordered list (`capture_sampled`) — None if there is nothing
    at that position in either shape."""
    if rows is None:
        return None
    if isinstance(rows, dict):
        return rows.get(i)
    return rows[i] if 0 <= i < len(rows) else None


def assert_agrees_or_forks_at_a_near_tie(ids_a, ids_b, rows_a=None, rows_b=None,
                                         eps_ulps: float = DEFAULT_EPS_ULPS) -> int | None:
    """The replacement for `assert transcript_a == transcript_b`: full
    agreement (the expected, common case) returns None. A fork must land at
    a position where at least one arm's OWN decision row was a near-tie —
    checked against both when both are available, because either arm could
    be the one whose accumulation order lost the tie, not just a single
    designated "reference" — otherwise this raises: that is a real
    divergence between the two arms, not float noise, and the old hard
    equality was at least right to flag it."""
    fork = first_fork(ids_a, ids_b)
    if fork is None:
        return None
    margins = []
    ra, rb = _row_at(rows_a, fork), _row_at(rows_b, fork)
    if ra is not None:
        margins.append(top2_margin_ulps(ra))
    if rb is not None:
        margins.append(top2_margin_ulps(rb))
    margin = min(margins) if margins else float("inf")
    assert margin < eps_ulps, (
        f"fork at position {fork}: token {ids_a[fork]} vs {ids_b[fork]}, but "
        f"the tightest top-2 margin either arm's own row shows there is "
        f"{margin:.1f} ULPs — not a near-tie, a real divergence between the "
        "two arms")
    return fork
