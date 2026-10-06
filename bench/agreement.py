"""Output agreement between two greedy runs over the same weights: where they
first part, the top-1 minus top-2 logit margin there, and whether that part
is a near-tie.

Two correct runs whose arithmetic differs only in accumulation order (warm
and cold prefill, a batched verify and a serial step, a CUDA graph and eager
decode, a one-row and an every-row lm_head, the compressed and the stock
arm) can pick different tokens where the top two logits are close
(docs/method.md, "Numerical behavior"). The gates that import this module
report such a part with its position and margin and do not fail on it. They
fail on a part whose margin is above NEAR_TIE_MARGIN: a broken cache, rewind
or accept rule moves logits by whole units and parts where the model had a
clear pick. A gate that sees only text (over HTTP, no logits) reports a
difference and never fails on it.

`PickTap` records the picks and margins from inside the serving process:
the gates that run their engine in-process use it, the rest compare text.
"""

from __future__ import annotations

# Logit units: the top-1 minus top-2 gap (or the gap between the two runs'
# picks) where two runs part. A bf16 logit between 16 and 32 moves in steps
# of 0.125, and the near-tie parts these gates have recorded sat at margins
# of 0 to 0.25; an ordinary row's margin is whole logits. 0.5 is four bf16
# steps at that magnitude.
NEAR_TIE_MARGIN = 0.5


def first_difference(a, b) -> int | None:
    """The first index where two sequences differ; the shorter one's length
    when one is a strict prefix of the other; None when they are equal."""
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def is_near_tie(*margins) -> bool | None:
    """Whether a part with these margins (one per run, None where a run has
    none) is a near-tie: every known margin at most NEAR_TIE_MARGIN. None
    when no margin is known."""
    known = [abs(float(m)) for m in margins if m is not None]
    if not known:
        return None
    return max(known) <= NEAR_TIE_MARGIN


def _at(seq, i):
    return seq[i] if seq is not None and i < len(seq) else None


def fork(ids_a, ids_b, margins_a=None, margins_b=None) -> dict:
    """Two greedy runs' picks over the same prompt, with each run's top-1
    minus top-2 margin per pick when it has them: where they first part.

    {"at": None} when the picks agree. Otherwise "at" (the index), "tokens"
    (each run's pick there; None past a run's end), "margins" (each run's
    margin there; None where unknown) and "near_tie" (is_near_tie of those
    margins)."""
    at = first_difference(ids_a, ids_b)
    if at is None:
        return {"at": None}
    margins = [_at(margins_a, at), _at(margins_b, at)]
    return {"at": at, "tokens": [_at(ids_a, at), _at(ids_b, at)], "margins": margins,
            "near_tie": is_near_tie(*margins)}


def _argmax(row) -> int:
    if hasattr(row, "argmax"):
        return int(row.argmax())
    return max(range(len(row)), key=row.__getitem__)


def row_fork(row_a, row_b) -> dict:
    """Two logits rows for the same position over the same prompt: each
    row's pick and, when the picks differ, each row's gap between its own
    pick and the other row's. Those gaps read both candidates in both rows,
    which a top-2 margin does not when the other pick ranks lower.

    {"picks": [a, b], "agree": True} or {"picks": [a, b], "agree": False,
    "gaps": [row_a[a] - row_a[b], row_b[b] - row_b[a]], "near_tie": ...}."""
    a, b = _argmax(row_a), _argmax(row_b)
    if a == b:
        return {"picks": [a, b], "agree": True}
    gaps = [round(float(row_a[a]) - float(row_a[b]), 6), round(float(row_b[b]) - float(row_b[a]), 6)]
    return {"picks": [a, b], "agree": False, "gaps": gaps, "near_tie": is_near_tie(*gaps)}


def above_margin(rec: dict | None) -> bool:
    """Whether a `fork` or `row_fork` record is a part the gates fail on: the
    runs part and a known margin there is above NEAR_TIE_MARGIN."""
    if not rec:
        return False
    parted = rec.get("at") is not None if "at" in rec else rec.get("agree") is False
    return parted and rec.get("near_tie") is False


def label(rec: dict | None) -> str:
    """One word for a `fork` or `row_fork` record: "agree", "near-tie",
    "ABOVE NEAR-TIE MARGIN", or "no margin" (a part with no logits to read)."""
    if not rec or rec.get("at", 0) is None or rec.get("agree") is True:
        return "agree"
    tie = rec.get("near_tie")
    return "no margin" if tie is None else "near-tie" if tie else "ABOVE NEAR-TIE MARGIN"


def describe(rec: dict | None, unit: str = "token") -> str:
    """A `fork` or `row_fork` record as one line: "agree", or where the runs
    part, each run's pick and margin (or gap) there, and its label."""
    if label(rec) == "agree":
        return "agree"
    if "picks" in rec:
        (ta, tb), (ma, mb) = rec["picks"], rec["gaps"]
        return f"part: {ta} (gap {ma}) against {tb} (gap {mb}) -> {label(rec)}"
    (ta, tb), (ma, mb) = rec["tokens"], rec["margins"]
    return (f"part at {unit} {rec['at']}: {ta} (margin {ma}) against {tb} (margin {mb}) "
            f"-> {label(rec)}")


def text_part(a: str, b: str, context: int = 20) -> dict | None:
    """Where two texts first differ, with `context` characters either side
    from each; None when they are equal."""
    at = first_difference(a, b)
    if at is None:
        return None
    lo, hi = max(0, at - context), at + context
    return {"at_char": at, "a": a[lo:hi], "b": b[lo:hi]}


class PickTap:
    """Each emitted position's pick and top-1 minus top-2 logit margin, by
    wrapping drinkme.serving.engines.sample_next. The serial loop and the
    speculative accept loop both pick through it and pass `gen_ids` (the ids
    emitted before this one) as a keyword, so a pick's position is
    len(gen_ids). Install it around requests served in this process, one at
    a time; `take()` after each request returns that request's picks and
    margins in position order. A grammar-constrained request picks through
    another function and records nothing."""

    def __init__(self):
        self.picks: dict[int, int] = {}
        self.margins: dict[int, float] = {}
        self._engines = self._real = None

    def __enter__(self):
        from drinkme.serving import engines

        self._engines, self._real = engines, engines.sample_next
        real = self._real

        def sample_next(logits, *a, **k):
            t = real(logits, *a, **k)
            at = len(k.get("gen_ids") or ())
            top = logits.detach().float().reshape(-1).topk(2).values
            self.picks[at] = int(t)
            self.margins[at] = round(float(top[0] - top[1]), 4)
            return t

        engines.sample_next = sample_next
        return self

    def __exit__(self, *exc):
        self._engines.sample_next = self._real

    def take(self) -> tuple[list[int], list[float]]:
        """The picks and margins recorded since the last take, in position
        order, and clear them."""
        order = sorted(self.picks)
        out = [self.picks[i] for i in order], [self.margins[i] for i in order]
        self.picks.clear()
        self.margins.clear()
        return out
