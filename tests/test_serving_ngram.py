"""N-gram (prompt-lookup) speculation — CPU only, no downloads, no GPU.

Three things have to be true and only one of them is about speed:

  EXACTNESS IN DISTRIBUTION. The proposal is a point mass, so rejection
  sampling accepts it with probability p[d] and resamples the residual
  otherwise (speculative.point_mass_rows carries the two-line argument) —
  checked here the way a distribution claim has to be: a histogram over many
  seeded draws against the same histogram with the proposer off. Greedy
  agrees with the serial loop except at a genuine near-tie: the verify keeps
  a draft only when the main model's own pick for that row equals it, but
  "that row" comes from a batched k+1-position forward where the serial loop
  used a single-token one, so the two can round differently in accumulation
  order (spec_agree.py) — token-exact between speculation on and off is not
  a property this repo asserts.

  ROLLING INDEX. The O(1)-per-token index has to give the same answer as the
  O(n) rescan it replaces, on random data, at every prefix length. That is
  `ngram.brute_force`, which exists for exactly this comparison.

  Neutrality. On text that does not repeat, the proposer must propose
  nothing — a cycle with no drafts is an M=1 forward, a plain decode step.
  bench/ngram_proposals.py measures the two prompt classes; the test here
  pins the shape (random ~ 0, repetitive ~ every cycle).

The toy is tests/test_serving_mtp.py's: a genuine hybrid qwen3_5 with a real
MTP head, which is also what makes the CHAINED mode testable — the head's KV
invariant has to survive cycles it did not draft.
"""

import os
import random

import pytest
import torch

from drinkme.serving import mtp, ngram, speculative
from drinkme.serving.engine import SampleParams
from drinkme.serving.engines import HFEngine

from test_serving_mtp import (  # noqa: F401 — fixtures come along by name
    PROMPTS,
    WORDS,
    _cfg,
    _msgs,
    _run,
    _torch_chunk_on_cpu,
    toy,
)

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


class spec_env:
    """DRINKME_SPEC (+ any other knob) for a block, restored after, with both
    modules' once-per-process warning latches cleared so warning tests are
    order-independent — tests/test_serving_mtp.py's `mtp_depth` shape."""

    def __init__(self, **kw):
        self.kw = kw

    def __enter__(self):
        self.old = {k: os.environ.get(k) for k in self.kw}
        for k, v in self.kw.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = str(v)
        ngram._warned.clear()
        mtp._warned.clear()
        return self

    def __exit__(self, *a):
        for k, v in self.old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        ngram._warned.clear()
        mtp._warned.clear()


def _engine(toy, with_head: bool, ctx: int = 512):
    model, tok, head, _ = toy
    os.environ["DRINKME_PREFIX_SLOTS"] = "0"
    try:
        return HFEngine(model, tok, model_id="toy", arm="test", meta={}, ctx=ctx,
                        mtp_head=head if with_head else None)
    finally:
        del os.environ["DRINKME_PREFIX_SLOTS"]


# A prompt built to repeat: the toy's tokenizer is whitespace WordLevel, so a
# repeated phrase is a repeated token n-gram. This is the Claude-Code shape in
# miniature — a block of text the context has already seen verbatim.
REPEAT = " ".join(["the quick brown fox jumps over the lazy dog"] * 6)


def _random_prompt(seed: int, n: int = 60) -> str:
    r = random.Random(seed)
    return " ".join(r.choice(WORDS) for _ in range(n))


# ------------------------------------------------------- the index itself --


def test_rolling_index_matches_a_brute_force_rescan_on_random_data():
    """THE index property: the incremental dicts and the O(n) rescan agree at
    every prefix length, over data with enough collisions to matter (a small
    alphabet, so 4-grams really do recur)."""
    r = random.Random(7)
    ids = [r.randrange(6) for _ in range(400)]
    p = ngram.NgramProposer(tokens=5, hi=4, lo=1)
    for i in range(len(ids)):
        p.append(ids[i])
        got = p.propose(5)
        want = ngram.brute_force(ids[:i + 1], 5, 4, 1)
        assert got == want, f"prefix {i + 1}: {got} != {want}"


def test_rolling_index_agrees_across_lookup_windows():
    """Same property at other (max, min) settings, including max == min."""
    r = random.Random(11)
    ids = [r.randrange(9) for _ in range(220)]
    for hi, lo in ((4, 1), (3, 2), (2, 2), (8, 4), (1, 1)):
        p = ngram.NgramProposer(tokens=3, hi=hi, lo=lo)
        for i in range(len(ids)):
            p.append(ids[i])
            assert p.propose(3) == ngram.brute_force(ids[:i + 1], 3, hi, lo)


def test_the_index_never_matches_the_suffix_against_itself():
    """A lookup must find an EARLIER occurrence. On a context that has never
    repeated anything, every n is a miss — if the suffix were in its own
    index it would match at once and propose the empty tail."""
    p = ngram.NgramProposer(tokens=5, hi=4, lo=1)
    p.reset(list(range(50)))  # strictly increasing: nothing recurs
    assert p.propose(5) == []


def test_nothing_is_proposed_below_prompt_lookup_min():
    """A context shorter than `prompt_lookup_min` + 1 has no window to match
    on at all, and a proposer that guessed there would be guessing."""
    p = ngram.NgramProposer(tokens=5, hi=4, lo=3)
    for i, tok in enumerate([1, 2, 3]):
        p.append(tok)
        assert p.propose(5) == [], f"proposed at length {i + 1}"
    # min=3 becomes reachable only once a 3-gram can repeat
    p = ngram.NgramProposer(tokens=5, hi=4, lo=3)
    p.reset([1, 2, 3, 4, 1, 2, 3])
    assert p.propose(5) == [4, 1, 2, 3]  # everything after the earlier [1,2,3]


def test_the_proposal_is_what_followed_the_MOST_RECENT_match():
    """Two occurrences of the same 4-gram with different continuations: the
    later one wins (the index stores the last start position)."""
    p = ngram.NgramProposer(tokens=3, hi=4, lo=4)
    p.reset([1, 2, 3, 4, 9, 9, 1, 2, 3, 4, 7, 7, 7, 5, 1, 2, 3, 4])
    assert p.propose(3) == [7, 7, 7]


def test_a_match_at_the_very_end_proposes_nothing():
    """A match whose continuation is the suffix itself has nothing to offer;
    the shorter n is then tried, and if it too has nothing the cycle drafts
    zero tokens rather than an empty-ish guess."""
    p = ngram.NgramProposer(tokens=5, hi=2, lo=2)
    p.reset([4, 5, 4, 5])          # the 2-gram [4,5] recurs at 0, followed by 4
    assert p.propose(5) == [4, 5]
    p = ngram.NgramProposer(tokens=5, hi=2, lo=2)
    p.reset([7, 8])
    assert p.propose(5) == []


def test_the_index_is_incremental_not_a_rescan():
    """O(1) amortized per token per n: each start position is inserted once,
    so the total insert count over a whole generation is bounded by
    len(context) * (max - min + 1), not by its square."""
    n = 300
    p = ngram.NgramProposer(tokens=5, hi=4, lo=1)
    for i in range(n):
        p.append(i % 7)
        p.propose(5)
    # 4 tables, each holding at most one entry per start position
    assert p.index.entries() <= 4 * n
    # and the tables really are small: a 7-cycle has very few distinct n-grams
    assert p.index.entries() < 60


# ------------------------------------------------------ the env resolution --


@pytest.mark.parametrize("spec,depth_env,has_head,want", [
    # AUTO (unset): head present -> mtp, head absent -> ngram
    (None, None, True, "mtp"),
    (None, None, False, "ngram"),
    (None, "4", True, "mtp"),
    # DRINKME_MTP_DEPTH is a depth, not a switch: it never changes the mode
    (None, "4", False, "ngram"),
    # explicit modes; off is the only off switch
    ("off", None, True, "off"),
    ("off", "4", True, "off"),
    ("off", None, False, "off"),
    ("mtp", None, True, "mtp"),
    ("mtp", None, False, "off"),        # asked for a head there is none of
    ("ngram", None, True, "ngram"),
    ("ngram", None, False, "ngram"),
    ("ngram+mtp", None, True, "ngram+mtp"),
    ("ngram+mtp", None, False, "ngram"),  # degrade, do not refuse
    ("auto", None, True, "mtp"),
    ("auto", None, False, "ngram"),
])
def test_the_spec_resolution_table(spec, depth_env, has_head, want):
    with spec_env(DRINKME_SPEC=spec, DRINKME_MTP_DEPTH=depth_env):
        assert ngram.resolve_mode(has_head) == want
        assert mtp.spec_plan(has_head, mtp.depth_from_env()).mode == want


def test_garbage_spec_falls_back_to_auto_loudly(capsys):
    with spec_env(DRINKME_SPEC="ngrma"):
        assert ngram.spec_from_env() == "auto"
    err = capsys.readouterr().err
    assert "DRINKME_SPEC" in err and "auto" in err


def test_only_the_modes_that_can_use_a_head_ask_for_one():
    """`wants_head` is what keeps DRINKME_SPEC=ngram from paying the head's
    residency on a checkpoint that has one."""
    for spec, want in (("off", False), ("ngram", False), ("mtp", True),
                       ("ngram+mtp", True), ("auto", True), (None, True)):
        with spec_env(DRINKME_SPEC=spec):
            assert ngram.wants_head() is want, spec


def test_ngram_params_read_vllms_knob_names_and_clamp():
    ceiling = mtp.FUSED_M_MAX - 1
    with spec_env(DRINKME_NGRAM_TOKENS=None, DRINKME_NGRAM_MAX=None,
                  DRINKME_NGRAM_MIN=None):
        # 5 and 4 are vLLM's; 3 is ours, moved by the window-6 A/B (chat
        # 0.854x at min=1 vs 0.995x at min=3 on the real 8B — ngram.py).
        assert ngram.params_from_env(ceiling) == (5, 4, 3)
    with spec_env(DRINKME_NGRAM_TOKENS=3, DRINKME_NGRAM_MAX=6,
                  DRINKME_NGRAM_MIN=2):
        assert ngram.params_from_env(ceiling) == (3, 6, 2)
    # the verify batch must stay inside the fused window: tokens clamp to
    # FUSED_M_MAX - 1, the same ceiling DRINKME_MTP_DEPTH is held to
    with spec_env(DRINKME_NGRAM_TOKENS=99):
        assert ngram.params_from_env(ceiling)[0] == ceiling
    # min > max is a typo, not a config: say so and use max
    with spec_env(DRINKME_NGRAM_MAX=2, DRINKME_NGRAM_MIN=5):
        assert ngram.params_from_env(ceiling)[1:] == (2, 2)
    with spec_env(DRINKME_NGRAM_TOKENS="banana"):
        assert ngram.params_from_env(ceiling)[0] == 5


def test_maybe_speculate_builds_the_proposer_the_mode_asks_for(toy):
    model, _, head, _ = toy
    p = SampleParams(temperature=0.0)
    with spec_env(DRINKME_SPEC="ngram"):
        s = mtp.maybe_speculate(head, model, p, depth=4)
        assert s.mode == "ngram" and s.head is None and s.lookup is not None
        assert s.depth == ngram.DEFAULT_TOKENS and not s.needs_hidden
    with spec_env(DRINKME_SPEC="ngram+mtp"):
        s = mtp.maybe_speculate(head, model, p, depth=4)
        assert s.mode == "ngram+mtp" and s.head is head and s.lookup is not None
        assert s.depth == 4 and s.needs_hidden
    with spec_env(DRINKME_SPEC="mtp"):
        s = mtp.maybe_speculate(head, model, p, depth=4)
        assert s.mode == "mtp" and s.lookup is None and s.needs_hidden
    with spec_env(DRINKME_SPEC="auto", DRINKME_MTP_DEPTH=None):
        assert mtp.maybe_speculate(None, model, p).mode == "ngram"
    with spec_env(DRINKME_SPEC="off"):
        assert mtp.maybe_speculate(head, model, p, depth=4) is None


def test_a_constrained_request_still_falls_back_to_the_serial_loop(toy):
    """The grammar constraint walks token by token; no proposer changes that."""
    model, _, head, _ = toy
    p = SampleParams(temperature=0.0, output_schema={"type": "json_object"})
    for spec in ("ngram", "ngram+mtp", "mtp"):
        with spec_env(DRINKME_SPEC=spec):
            assert mtp.maybe_speculate(head, model, p, depth=4) is None


# ------------------------------------------------------------- exactness --


def test_point_mass_rows_are_a_legal_q():
    """The three quantities speculative_sample reads off q, for a one-hot q:
    q[d] == 1, the residual is p with d removed, and the theorem's expected
    acceptance sum(min(p, q)) is p[d]."""
    torch.manual_seed(0)
    p = torch.softmax(torch.randn(32), dim=-1)
    for d in (0, 7, 31):
        q = speculative.point_mass_rows(p, [d])[0]
        assert q.shape == p.shape and q.dtype == p.dtype
        assert float(q.sum()) == 1.0 and float(q[d]) == 1.0
        assert pytest.approx(float(torch.minimum(p, q).sum())) == float(p[d])
        resid = (p - q).clamp(min=0)
        assert float(resid[d]) == 0.0
        assert pytest.approx(float(resid.sum())) == float(1 - p[d])


def test_point_mass_rows_for_no_drafts_is_the_empty_list():
    """A cycle that proposed nothing has depth 0, and speculative_sample
    wants exactly zero q rows for it."""
    assert speculative.point_mass_rows(torch.zeros(4), []) == []


def _histogram(p, drafts, n, seed):
    """Emitted-first-token counts over n rejection-sampling steps against a
    point-mass proposal at `drafts[0]`, target rows all equal to p."""
    from collections import Counter

    g = torch.Generator().manual_seed(seed)
    rows = [p] * (len(drafts) + 1)
    q = speculative.point_mass_rows(p, drafts)
    c = Counter()
    for _ in range(n):
        emitted, _ = speculative.speculative_sample(rows, q, drafts, generator=g)
        c[emitted[0]] += 1
    return c


def test_the_emitted_histogram_matches_plain_sampling(capsys):
    """THE distribution claim, at temperature 1.0 (these rows ARE a softmax
    at temperature 1) and with many samples: the first emitted token of a
    point-mass-proposed cycle is distributed as p, not as some blend of p and
    the proposal. Checked against p directly AND against a plain multinomial
    draw from p, so the tolerance is calibrated on real sampling noise rather
    than guessed."""
    torch.manual_seed(3)
    p = torch.softmax(torch.randn(24) * 1.5, dim=-1)
    n = 60_000
    for d in (int(p.argmax()), int(p.argmin()), 5):
        c = _histogram(p, [d, d], n, seed=1234 + d)
        g = torch.Generator().manual_seed(99 + d)
        plain = torch.multinomial(p.expand(n, -1), 1, generator=g).flatten()
        got = torch.tensor([c[i] / n for i in range(len(p))])
        ref = torch.bincount(plain, minlength=len(p)).float() / n
        tv_p = 0.5 * float((got - p).abs().sum())
        tv_ref = 0.5 * float((got - ref).abs().sum())
        # the control: two independent plain samplers of the same size
        g2 = torch.Generator().manual_seed(4242 + d)
        plain2 = torch.multinomial(p.expand(n, -1), 1, generator=g2).flatten()
        ref2 = torch.bincount(plain2, minlength=len(p)).float() / n
        noise = 0.5 * float((ref - ref2).abs().sum())
        print(f"draft={d}: TV(spec, p)={tv_p:.5f} TV(spec, plain)={tv_ref:.5f} "
              f"TV(plain, plain')={noise:.5f}")
        assert tv_p < 0.01, f"draft {d}: TV {tv_p:.4f} against p itself"
        assert tv_ref < max(4 * noise, 0.01), f"draft {d}: TV {tv_ref:.4f} vs noise {noise:.4f}"


def test_a_wrong_accept_rule_would_fail_that_histogram(monkeypatch):
    """The instrument has power: an accept-everything rule (the bug the
    histogram exists to catch) skews the emitted distribution toward the
    proposal, and the same tolerance rejects it."""
    torch.manual_seed(3)
    p = torch.softmax(torch.randn(24) * 1.5, dim=-1)
    d = int(p.argmin())
    monkeypatch.setattr(speculative, "_accept", lambda *a, **k: True)
    c = _histogram(p, [d, d], 20_000, seed=5)
    got = torch.tensor([c[i] / 20_000 for i in range(len(p))])
    assert 0.5 * float((got - p).abs().sum()) > 0.5


# --------------------------------------------------- end to end, on the toy --


@pytest.fixture(scope="module")
def serial(toy):
    """Greedy reference runs with speculation OFF, for the agreement bars
    below: {name: {"res", "ids", "rows"}} — `ids`/`rows` come from
    spec_agree.capture, the per-position raw logits a fork gets checked
    against (token-exact between speculation on and off is not a property
    this repo asserts — what holds is argmax agreement except at
    a genuine near-tie, same as every other arm-vs-arm comparison here)."""
    from drinkme.serving import engines
    from spec_agree import capture, ordered_ids

    out = {}
    with spec_env(DRINKME_SPEC="off"):
        eng = _engine(toy, with_head=False)
        for name, text in (("repeat", REPEAT), ("random", _random_prompt(5)),
                           ("short", PROMPTS[1])):
            picks, rows, restore = capture(engines)
            try:
                res, _ = _run(eng, _msgs(text), SampleParams(temperature=0.0,
                                                             max_tokens=64))
            finally:
                restore()
            assert res.completion_tokens == 64
            out[name] = {"res": res, "ids": ordered_ids(picks), "rows": rows}
    return out


@pytest.mark.parametrize("name,text", [("repeat", REPEAT),
                                       ("random", _random_prompt(5)),
                                       ("short", PROMPTS[1])])
def test_greedy_ngram_agrees_with_the_serial_loop_except_at_a_near_tie(toy, serial, name, text):
    """THE bar, on a model with NO head — which is the configuration `auto`
    ships for the 8B and gemma. Repetitive prompts exercise the accept path
    (the lookup fires most cycles); random ones exercise the propose-nothing
    path. The verify keeps a draft only when the main model's OWN pick for
    that row equals it — but "that row" comes from a batched k+1-position
    forward where the serial loop used a single-token one, so the two can
    round differently in accumulation order; a fork is only legal at a
    genuine near-tie (spec_agree), never anywhere else."""
    from drinkme.serving import engines
    from spec_agree import assert_agrees_or_forks_at_a_near_tie, capture, ordered_ids

    with spec_env(DRINKME_SPEC="ngram"):
        eng = _engine(toy, with_head=False)
        picks, rows, restore = capture(engines)
        try:
            res, _ = _run(eng, _msgs(text), SampleParams(temperature=0.0, max_tokens=64))
        finally:
            restore()
    ref = serial[name]
    assert_agrees_or_forks_at_a_near_tie(ordered_ids(picks), ref["ids"], rows, ref["rows"])
    assert res.finish_reason == ref["res"].finish_reason


def test_greedy_ngram_with_penalties_agrees_except_at_a_near_tie(toy):
    """Greedy-with-penalties goes through sampling.sample_next with the id
    history the serial loop would have had, for a lookup draft exactly as
    for a head draft — so the accepted stream agrees with the serial one
    except where a near-tie in the (penalized) row lets the two arms'
    accumulation-order differences flip the pick."""
    from drinkme.serving import engines
    from spec_agree import assert_agrees_or_forks_at_a_near_tie, capture, ordered_ids

    kw = dict(temperature=0.0, max_tokens=48, repetition_penalty=1.3,
              presence_penalty=0.3, frequency_penalty=0.2)
    with spec_env(DRINKME_SPEC="off"):
        picks_off, rows_off, restore = capture(engines)
        try:
            off, _ = _run(_engine(toy, False), _msgs(REPEAT), SampleParams(**kw))
        finally:
            restore()
    with spec_env(DRINKME_SPEC="ngram"):
        picks_on, rows_on, restore = capture(engines)
        try:
            on, _ = _run(_engine(toy, False), _msgs(REPEAT), SampleParams(**kw))
        finally:
            restore()
    assert_agrees_or_forks_at_a_near_tie(ordered_ids(picks_on), ordered_ids(picks_off),
                                         rows_on, rows_off)


@pytest.mark.parametrize("name,text", [("repeat", REPEAT), ("short", PROMPTS[1])])
def test_chained_ngram_then_mtp_agrees_with_the_serial_loop_too(toy, serial, name, text):
    """The chained mode's hard part is the head's KV: on a cycle the LOOKUP
    drafted, the head appended no entries, so the rebuild has to start one
    entry earlier (from h_last and the token the cycle continued from). If
    that were wrong the head's history would drift and acceptance would
    suffer badly enough to fork at a real (non-near-tie) margin — which is
    what this still pins; test_the_chained_head_keeps_its_kv_invariant
    checks the state itself directly."""
    from drinkme.serving import engines
    from spec_agree import assert_agrees_or_forks_at_a_near_tie, capture, ordered_ids

    with spec_env(DRINKME_SPEC="ngram+mtp", DRINKME_MTP_DEPTH="4"):
        eng = _engine(toy, with_head=True)
        picks, rows, restore = capture(engines)
        try:
            res, _ = _run(eng, _msgs(text), SampleParams(temperature=0.0, max_tokens=64))
        finally:
            restore()
    ref = serial[name]
    assert_agrees_or_forks_at_a_near_tie(ordered_ids(picks), ref["ids"], rows, ref["rows"])


def test_the_chained_head_keeps_its_kv_invariant(toy):
    """`head.entries` must equal `written - 1` at every cycle boundary — the
    Speculator's own stated invariant ("entries up to W-2") — whichever
    proposer drafted the cycle. A hole at entry W-1 would be invisible in the
    output and would quietly cost acceptance forever after."""
    seen = []
    real = mtp.Speculator.cycle

    def spy(self, last_token, pick, budget, **kw):
        out = real(self, last_token, pick, budget, **kw)
        seen.append((self.head.entries, self.written, self.depth))
        return out

    with spec_env(DRINKME_SPEC="ngram+mtp", DRINKME_MTP_DEPTH="4"):
        eng = _engine(toy, with_head=True)
        mtp.Speculator.cycle = spy
        try:
            _run(eng, _msgs(REPEAT), SampleParams(temperature=0.0, max_tokens=48))
        finally:
            mtp.Speculator.cycle = real
    assert seen, "no cycles ran"
    for entries, written, _ in seen[:-1]:  # the last cycle may be terminal (k=0)
        assert entries == written - 1, f"head KV {entries} vs written {written}"


def _tool_transcript(seed: int, turns: int = 8) -> list[int]:
    """A Claude-Code-shaped id stream over a realistic vocabulary: a long tool
    result (a file listing) re-sent verbatim every turn, with a short unique
    reply between turns. This is the case prompt lookup exists for."""
    r = random.Random(seed)
    tool = [r.randrange(150_000) for _ in range(300)]
    out: list[int] = []
    for _ in range(turns):
        out += tool
        out += [r.randrange(150_000) for _ in range(40)]
    return out


def _proposals_per_token(ids, tokens=5, hi=4, lo=1):
    """Replay `ids` through the proposer one token at a time, as a generation
    would, and return (proposals per token, fraction of steps that matched)."""
    p = ngram.NgramProposer(tokens=tokens, hi=hi, lo=lo)
    for t in ids:
        p.append(t)
        p.propose(tokens)
    s = p.stats()
    return s["proposed"] / s["asked"], s["hit_rate"]


def test_neutrality_random_prompt_versus_an_agent_transcript(capsys):
    """Neutrality: proposals per token on a
    random-token prompt (expect ~0) against a Claude-Code-shaped transcript
    with tool output repeated (expect many).

    Over a REALISTIC vocabulary, which is the whole point — the toy model
    next door has 96 tokens, so even a 1-gram lookup hits constantly there
    and the measurement would say nothing about a 150k-vocabulary server.
    bench/ngram_proposals.py is the same measurement on real tokenized text.
    """
    r = random.Random(31)
    noise = [r.randrange(150_000) for _ in range(2500)]
    n_rate, n_hit = _proposals_per_token(noise)
    a_rate, a_hit = _proposals_per_token(_tool_transcript(41))
    print(f"random-token prompt : {n_rate:.4f} proposals/token, "
          f"{n_hit:.1%} of steps matched")
    print(f"agent transcript    : {a_rate:.4f} proposals/token, "
          f"{a_hit:.1%} of steps matched")
    # Measured: 0.041 proposals/token, 0.8% of steps. Not a hard
    # zero, and the reason is prompt_lookup_min=1 (vLLM's default): in 2500
    # draws from a 150k vocabulary a few single tokens do recur by birthday,
    # and a 1-gram match is enough to propose. Those proposals cost the
    # widened verify batch on 0.8% of steps and are otherwise rejected —
    # which is why the floor is a band, not an equality. DRINKME_NGRAM_MIN=4
    # takes it to a true zero (the next test).
    assert n_rate < 0.15 and n_hit < 0.03, "the proposer must be ~silent on noise"
    assert a_rate > 3.0, "a re-sent tool result should be proposed on almost every step"
    assert a_hit > 20 * n_hit


def test_the_lookup_fires_and_is_accounted_for_end_to_end(toy, capsys):
    """The same shape through the real engine, on the toy. The toy's
    20-word vocabulary makes the ABSOLUTE numbers meaningless (a 1-gram
    lookup hits on almost anything there — see the test above for the
    honest measurement); what this pins is that the counters reach
    Speculator.stats and that acceptance on a repetitive prompt beats
    acceptance on a random one, which is the ordering that matters."""
    out = {}
    for name, text in (("repeat", REPEAT), ("random", _random_prompt(17, 80))):
        stats = {}
        real = mtp.Speculator.stats

        def grab(self, _stats=stats):
            s = real(self)
            _stats.update(s)
            return s

        with spec_env(DRINKME_SPEC="ngram"):
            eng = _engine(toy, with_head=False)
            mtp.Speculator.stats = grab
            try:
                res, _ = _run(eng, _msgs(text),
                              SampleParams(temperature=0.0, max_tokens=64))
            finally:
                mtp.Speculator.stats = real
        lk = stats["lookup"]
        out[name] = stats["acceptance"]
        print(f"{name}: {lk['proposed'] / res.completion_tokens:.3f} "
              f"proposals/token, lookup hit rate {lk['hit_rate']:.0%}, "
              f"{stats['accepted']}/{stats['drafted']} accepted")
        assert lk["asked"] == stats["cycles"] and lk["proposed"] > 0
    assert out["repeat"] > out["random"]


def test_prompt_lookup_min_4_is_silent_on_a_random_prompt(toy):
    """The hard-zero version of neutrality, with the 1-token fallback off:
    on unrepeated text a 4-gram lookup finds nothing at all."""
    p = ngram.NgramProposer(tokens=5, hi=4, lo=4)
    r = random.Random(23)
    ids = [r.randrange(4000) for _ in range(500)]
    p.reset(ids[:1])
    for t in ids[1:]:
        p.append(t)
        assert p.propose(5) == []


# ---------------------------------------------------------- accounting --


def test_the_accounting_line_names_the_proposer(toy, capsys):
    with spec_env(DRINKME_SPEC="ngram"):
        eng = _engine(toy, with_head=False)
        _run(eng, _msgs(REPEAT), SampleParams(temperature=0.0, max_tokens=32))
    err = capsys.readouterr().err
    assert "[drinkme.spec] ngram:" in err and "lookup" in err


def test_sampled_ngram_requests_run_the_audit(toy):
    """The audit counters (docs/serve-speculation.md) must accrue for a lookup
    proposal exactly as for a head one: the theorem is the same theorem, and
    a point-mass q is a legal q. Zero violations is the bar."""
    stats = {}
    real = mtp.Speculator.stats

    def grab(self):
        s = real(self)
        stats.update(s)
        return s

    # min=1 pinned on purpose: this test is about the AUDIT accruing under a
    # lookup proposer, so it needs proposals to exist — and the toy's sampled
    # output at temperature 1.0 matches the prompt on single tokens only
    # (at the shipped default of 3 the lookup finds 0/63 cycles and the audit
    # correctly has nothing to count).
    with spec_env(DRINKME_SPEC="ngram", DRINKME_NGRAM_MIN=1):
        eng = _engine(toy, with_head=False)
        mtp.Speculator.stats = grab
        try:
            _run(eng, _msgs(REPEAT),
                 SampleParams(temperature=1.0, seed=11, max_tokens=64))
        finally:
            mtp.Speculator.stats = real
    a = stats["audit"]
    assert a is not None and a["decided"] > 0
    assert a["violations"] == 0
    # sum(min(p, q)) for a point mass is p[d], so the expected rate is the
    # mean target probability of the proposed tokens — a real number, not 1
    assert 0.0 <= a["expected_rate"] <= 1.0
    assert abs(a["actual_rate"] - a["expected_rate"]) < 0.15


def test_ngram_does_not_change_prefill(toy, monkeypatch):
    """`needs_hidden` is False for ngram-only, so prefill takes the serial
    loop's own narrow call (logits_to_keep=1) and mtp.forward_with_hidden is
    never reached before the first cycle. This is what the greedy identity
    bar above rests on."""
    calls = []
    real = mtp.forward_with_hidden

    def spy(model, input_ids, cache, cache_position, last_row_only=False):
        calls.append((int(input_ids.shape[1]), last_row_only))
        return real(model, input_ids, cache, cache_position, last_row_only)

    monkeypatch.setattr(mtp, "forward_with_hidden", spy)
    with spec_env(DRINKME_SPEC="ngram"):
        eng = _engine(toy, with_head=False)
        _run(eng, _msgs(REPEAT), SampleParams(temperature=0.0, max_tokens=8))
    assert calls, "the verify pass should still go through forward_with_hidden"
    assert not any(last_row for _, last_row in calls), \
        "last_row_only is prefill's flag; the verify cycle must never set it"


def test_no_head_is_loaded_for_a_lookup_only_mode(toy, monkeypatch):
    """DRINKME_SPEC=ngram hands the head's residency back: engines._mtp_head
    returns None without touching the checkpoint at all."""
    from drinkme.serving import engines

    def boom(*a, **kw):
        raise AssertionError("load_head must not be called in a lookup-only mode")

    monkeypatch.setattr(mtp, "load_head", boom)
    for spec in ("ngram", "off"):
        with spec_env(DRINKME_SPEC=spec, DRINKME_MTP_DEPTH="4"):
            assert engines._mtp_head(toy[0], "toy", None, "cpu") is None


def test_the_arm_toggle_seam_really_drops_the_cached_plan(toy):
    """The stale-instrument class, pinned before an instrument depends on it.

    bench/ngram_gpu_ab.py toggles arms per request by writing DRINKME_SPEC and
    clearing `_spec_plan` / `_mtp_depth`. bench/mtp_gpu_acceptance.py's own
    docstring records what happens when that seam stops working: every "ON"
    run for three weeks was the serial loop wearing an ON label. So assert the
    engine serves the mode it was last told, request after request, on one
    engine build."""
    served = []
    real = mtp.Speculator.stats

    def grab(self):
        s = real(self)
        served.append(s["mode"])
        return s

    eng = _engine(toy, with_head=True)
    mtp.Speculator.stats = grab
    try:
        for want in ("mtp", "ngram", "ngram+mtp", "mtp", "off", "ngram"):
            with spec_env(DRINKME_SPEC=want, DRINKME_MTP_DEPTH="4"):
                eng._spec_plan = eng._mtp_depth = None
                before = len(served)
                _run(eng, _msgs(REPEAT), SampleParams(temperature=0.0, max_tokens=8))
            got = served[before:]
            if want == "off":
                assert got == [], "the off arm must not build a speculator"
            else:
                assert got == [want], f"asked {want!r}, served {got}"
    finally:
        mtp.Speculator.stats = real


def test_a_headless_deltanet_model_speculates_without_the_mtp_loader(toy):
    """The verify's rewind restores DeltaNet state from the capture,
    whichever proposer drafted, so the capture cannot be left to the MTP
    head loader: a qwen3_5 checkpoint that ships no mtp.* tensors (AUTO ->
    ngram) would fail its first rejected draft with "no captured DeltaNet
    state". The
    shared toy installs the capture itself, so this builds a fresh model that
    nothing has wrapped and serves it the way AUTO would."""
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

    _, tok, _, _ = toy
    torch.manual_seed(0)
    model = Qwen3_5ForCausalLM(_cfg()).eval().to(torch.float32)
    for p in model.parameters():
        p.requires_grad_(False)
    stats = {}
    real = mtp.Speculator.stats

    def grab(self):
        s = real(self)
        stats.update(s)
        return s

    with spec_env(DRINKME_SPEC=None, DRINKME_MTP_DEPTH=None):
        eng = _engine((model, tok, None, None), with_head=False)
        mtp.Speculator.stats = grab
        try:
            res, _ = _run(eng, _msgs(REPEAT), SampleParams(temperature=0.0, max_tokens=64))
        finally:
            mtp.Speculator.stats = real
    assert stats["mode"] == "ngram" and res.completion_tokens == 64
    # it drafted, and at least one draft was rejected: the rewind really ran
    assert 0 < stats["drafted"] and stats["accepted"] < stats["drafted"]
