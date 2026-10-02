"""N-gram speculation (prompt lookup): a draft model made out of the context.

vLLM ships this as `{"method": "ngram", "num_speculative_tokens": 5,
"prompt_lookup_max": 4, "prompt_lookup_min": 1}` (vLLM's docs/features/
speculative_decoding/n_gram.md). The idea is one sentence long: the tokens a model is about to emit
have very often already been said somewhere earlier in its own context — a
re-sent file, a diff applied twice, a tool result quoted back, a stack trace
echoed — so look the last few tokens up in the context and propose whatever
followed them last time. No draft model, no head, no weights, no residency:
the proposal costs a dict lookup.

WHY THIS IS EXACT, IN TWO LINES. The proposal is a POINT MASS: q_i puts all
of its mass on the one token d_i the lookup found. Rejection sampling
(serving/speculative.py) accepts d_i with probability min(1, p_i[d_i]/q_i[d_i])
= p_i[d_i] and, on rejection, emits one token from the normalized residual
max(0, p_i - q_i), which is p_i with the mass at d_i removed. Both are the
theorem's own rule with q = one-hot substituted in, so the emitted token is
distributed EXACTLY as p_i — the same conclusion, for the same reason, as
for the MTP head. The proposal distribution is arbitrary in that theorem; a
proposer can only cost acceptance, never change what is emitted (a real,
checked-empirically property — tests/test_serving_ngram.py's histogram
section). On the greedy path there is no distribution at all: the verify
pass keeps a draft only when the main model's own pick for that row equals
it (mtp.Speculator) — but "that row" comes from a batched k+1-position
forward where the serial loop used a single-token one, and the two can round
differently in accumulation order, so the stream agrees with the serial loop
except at a genuine near-tie, whatever the drafts were — not byte-identical
as a blanket claim (tests/spec_agree.py is the check; token-exact between
speculation on and off is not a property this repo asserts).

THIS MODULE DOES NOT RUN A VERIFY LOOP. It plugs into mtp.Speculator — the
same accept/commit state machine, the same rejection-sampling call, the same
SpeculativeAudit counters, the same cache rewind. A second verify path would
be a second place for the product's central claim to be wrong.

SELECTION (DRINKME_SPEC, the resolution table in `resolve_mode`)

  off        no speculation, the serial loop
  mtp        the checkpoint's MTP head only (the default behaviour)
  ngram      prompt lookup only — no head is loaded at all (saves its ~0.8 GB)
  ngram+mtp  chained: ngram proposes; when it finds nothing the head does
  auto       (unset, the default) = mtp if the checkpoint carries a head,
             else ngram

`auto` is what makes this feature reach the 8B and gemma, which have no head
and would otherwise have no speculation of any kind. It deliberately does NOT change
the 27B, whose head is measured and shipped: a head beats a lookup on prose,
and `ngram+mtp` is the flag for measuring whether chaining beats either.

DRINKME_SPEC IS THE ONLY SWITCH. `off` turns every proposer off;
DRINKME_MTP_DEPTH sets the head's draft depth and nothing else.

NEUTRALITY IS THE COST MODEL. On text with no repetition the lookup finds
nothing and proposes nothing, and a cycle with zero drafts is an M=1 forward
— a plain decode step plus a dict lookup. That is why this proposer needs no
adaptive bail the way the head does (NGRAM_BAIL_FLOOR): the head runs k
forwards whether or not they are any good, so it needs a floor to stop
paying; prompt lookup stops paying by itself. bench/ngram_proposals.py
measures proposals/token on a random-token prompt (expect ~0) against an
agent-shaped transcript (expect many).

THE INDEX IS ROLLING, NOT A RESCAN. `NgramIndex` keeps one dict per n in
[prompt_lookup_min, prompt_lookup_max], mapping an n-gram to the START
position of its most recent occurrence, and inserts each new position exactly
once as the context grows — O(1) amortized per token per n, instead of the
O(len(context)) scan per step a naive implementation does. The dicts hold
only positions that could legally be matched: the suffix being looked up is
never in its own index, so a query can only find an EARLIER occurrence.
"""

from __future__ import annotations

import os
import sys

# vLLM's names, kept so the knobs are recognizable to anyone who has
# configured this feature there. Two of
# the three values are vLLM's; MIN is ours, moved by measurement:
DEFAULT_TOKENS = 5   # num_speculative_tokens: how many to propose per cycle
DEFAULT_MAX = 4      # prompt_lookup_max: longest suffix to match on
DEFAULT_MIN = 3      # prompt_lookup_min: shortest suffix to fall back to.
# vLLM ships 1. On a Strix Halo (Qwen3-8B, window 6),
# min=1 cost ordinary chat 15% (0.854x) for +1.5% on agent transcripts; min=3 is
# neutral on chat (0.995x), +10% on transcripts and code, and still 3.7x on
# repetitive text. A one-token match is a bet that loses on prose.

# The adaptive bail's floor for an ngram proposer. Zero = no bail (module
# docstring: the no-match path IS the bail). Named rather than typed as a
# literal so moving it moves the argument with it.
NGRAM_BAIL_FLOOR = 0.0

MODES = ("off", "mtp", "ngram", "ngram+mtp", "auto")

_warned: set[str] = set()


def _warn_once(key: str, msg: str) -> None:
    if key not in _warned:
        _warned.add(key)
        print(f"[drinkme.ngram] {msg}", file=sys.stderr)


# ------------------------------------------------------------ selection --


def spec_from_env() -> str:
    """DRINKME_SPEC as one of MODES. Unset or garbage = "auto" (garbage warns
    but is never fatal: a typo in an env var must not take the server down,
    and must not silently change decode either — mtp.depth_from_env's rule)."""
    raw = os.environ.get("DRINKME_SPEC", "").strip().lower()
    if not raw:
        return "auto"
    if raw in MODES:
        return raw
    _warn_once("badspec", f"DRINKME_SPEC={raw!r} is not one of "
                          f"{'|'.join(MODES)} — falling back to auto")
    return "auto"


def resolve_mode(has_head: bool) -> str:
    """The resolution table (module docstring), as a function of DRINKME_SPEC
    and whether a head is actually resident (`has_head`: the engine is
    holding a loaded MTP head).

    Returns "off" | "mtp" | "ngram" | "ngram+mtp" — never "auto".
    """
    spec = spec_from_env()
    if spec in ("off", "ngram"):
        return spec
    if spec == "mtp":
        return "mtp" if has_head else "off"
    if spec == "ngram+mtp":
        return "ngram+mtp" if has_head else "ngram"
    return "mtp" if has_head else "ngram"


def wants_head() -> bool:
    """Should the engine LOAD the MTP head at all? False for the modes that
    can never use one, so `DRINKME_SPEC=ngram` on the 27B gets its ~0.8 GB
    back instead of paying residency for a head nothing will call."""
    return spec_from_env() not in ("off", "ngram")


def _int_env(name: str, default: int, low: int, high: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        v = int(raw)
    except ValueError:
        _warn_once(f"bad{name}", f"{name}={raw!r} is not an integer — using {default}")
        return default
    if v < low or v > high:
        _warn_once(f"rng{name}", f"{name}={v} is outside [{low}, {high}] — "
                                 f"clamped")
        v = max(low, min(high, v))
    return v


def params_from_env(depth_ceiling: int) -> tuple[int, int, int]:
    """(num_speculative_tokens, prompt_lookup_max, prompt_lookup_min) from
    DRINKME_NGRAM_TOKENS / _MAX / _MIN, defaults as vLLM's.

    `depth_ceiling` is mtp.FUSED_M_MAX - 1: the verify batch is drafts + 1
    rows and must stay inside the fused bit-identical window, which is the
    same ceiling DRINKME_MTP_DEPTH is held to and for the same reason."""
    k = _int_env("DRINKME_NGRAM_TOKENS", DEFAULT_TOKENS, 1, depth_ceiling)
    hi = _int_env("DRINKME_NGRAM_MAX", DEFAULT_MAX, 1, 64)
    lo = _int_env("DRINKME_NGRAM_MIN", DEFAULT_MIN, 1, 64)
    if lo > hi:
        _warn_once("minmax", f"DRINKME_NGRAM_MIN={lo} > DRINKME_NGRAM_MAX={hi}"
                             f" — using min = max = {hi}")
        lo = hi
    return k, hi, lo


# ------------------------------------------------------------- the index --


class NgramIndex:
    """n-gram -> start position of its most recent occurrence, one dict per n,
    maintained incrementally.

    The invariant that makes a query correct: after `insert_upto(L)` the dict
    for n holds every start position in [0, L - n - 1] and NOTHING at
    L - n. So looking up `ctx[L-n:L]` — the current suffix — can only hit an
    occurrence that ended before position L, which is the whole point: a
    match is evidence about what came NEXT last time, and the suffix has no
    next yet.
    """

    __slots__ = ("ids", "lo", "hi", "_tables", "_next")

    def __init__(self, lo: int, hi: int, ids: list[int] | None = None):
        self.lo, self.hi = lo, hi
        self.ids: list[int] = []
        self._tables: dict[int, dict[tuple, int]] = {n: {} for n in range(lo, hi + 1)}
        # per n: the next start position not yet inserted
        self._next: dict[int, int] = {n: 0 for n in range(lo, hi + 1)}
        if ids:
            self.extend(ids)

    def __len__(self) -> int:
        return len(self.ids)

    def append(self, token: int) -> None:
        self.ids.append(token)

    def extend(self, tokens) -> None:
        self.ids.extend(tokens)

    def _catch_up(self, n: int) -> None:
        """Insert every n-gram whose start is <= len - n - 1. Each start is
        inserted exactly once over the life of the request, so this is O(1)
        amortized per token per n even though it is written as a loop."""
        ids, table = self.ids, self._tables[n]
        end = len(ids) - n  # exclusive: the suffix's own start is not indexed
        i = self._next[n]
        while i < end:
            table[tuple(ids[i:i + n])] = i
            i += 1
        self._next[n] = i

    def lookup(self, n: int) -> int | None:
        """Start position of the most recent EARLIER occurrence of the last n
        tokens, or None."""
        if n < self.lo or n > self.hi or len(self.ids) <= n:
            return None
        self._catch_up(n)
        return self._tables[n].get(tuple(self.ids[-n:]))

    def entries(self) -> int:
        """Total indexed n-grams — what the memory question is about."""
        return sum(len(t) for t in self._tables.values())


# ---------------------------------------------------------- the proposer --


class NgramProposer:
    """The Speculator-facing object: hold the context, propose k tokens.

    Longest match first (`prompt_lookup_max` down to `prompt_lookup_min`): a
    4-token match is far more likely to be a real repeat than a 1-token one,
    so it is tried first and the shorter ns are the fallback. The proposal is
    whatever followed that match, truncated to what the context actually has
    and to `k`.
    """

    def __init__(self, tokens: int = DEFAULT_TOKENS, hi: int = DEFAULT_MAX,
                 lo: int = DEFAULT_MIN):
        self.k = tokens
        self.hi, self.lo = hi, lo
        self.index = NgramIndex(lo, hi)
        # accounting, per request: how many cycles asked, how many found a
        # match, how many tokens were proposed (the neutrality numbers)
        self.asked = 0
        self.matched = 0
        self.proposed = 0
        self.match_n = 0  # summed n of the matches, for a mean match length

    # -- context ------------------------------------------------------------
    def reset(self, ids: list[int]) -> None:
        """Start a generation over `ids` (the whole prompt)."""
        self.index = NgramIndex(self.lo, self.hi, list(ids))

    def append(self, token: int) -> None:
        self.index.append(token)

    def extend(self, tokens) -> None:
        self.index.extend(tokens)

    def __len__(self) -> int:
        return len(self.index)

    # -- the proposal -------------------------------------------------------
    def propose(self, k: int) -> list[int]:
        """Up to min(k, num_speculative_tokens) tokens, or [] for no match.

        After a prompt with images the context holds each image's run as
        its prefix key (serving/image_prompt.py), a negative id no
        vocabulary has. A run matches only itself, and a proposal stops
        where one begins: a drafted image row is never a token."""
        k = min(k, self.k)
        self.asked += 1
        if k <= 0:
            return []
        ids = self.index.ids
        for n in range(min(self.hi, len(ids) - 1), self.lo - 1, -1):
            at = self.index.lookup(n)
            if at is None:
                continue
            out = ids[at + n:at + n + k]
            for i, t in enumerate(out):
                if t < 0:
                    out = out[:i]
                    break
            if not out:
                continue  # a match with nothing after it proposes nothing
            self.matched += 1
            self.match_n += n
            self.proposed += len(out)
            return out
        return []

    def stats(self) -> dict:
        """Per-request proposer accounting; folded into Speculator.stats()."""
        return {"asked": self.asked, "matched": self.matched,
                "proposed": self.proposed,
                "hit_rate": round(self.matched / self.asked, 3) if self.asked else 0.0,
                "mean_match_n": round(self.match_n / self.matched, 2) if self.matched else 0.0,
                "indexed": self.index.entries()}


def brute_force(ids: list[int], k: int, hi: int, lo: int) -> list[int]:
    """The rescan `NgramProposer.propose` replaces, kept as the reference the
    rolling index is tested against (tests/test_serving_ngram.py). Same
    answer, O(len(ids)) per call instead of O(1) amortized."""
    for n in range(min(hi, len(ids) - 1), lo - 1, -1):
        want = ids[-n:]
        at = None
        for i in range(len(ids) - n):  # never the suffix's own start
            if ids[i:i + n] == want:
                at = i
        if at is None:
            continue
        out = ids[at + n:at + n + k]
        if any(t < 0 for t in out):  # an image run's key: the proposal stops there
            out = out[:next(i for i, t in enumerate(out) if t < 0)]
        if out:
            return out
    return []
