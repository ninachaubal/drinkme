"""Rejection-sampling MTP, CPU only: the theorem checked empirically, the
transform parity that makes it apply, the adaptive bail, and the sampled cycle
end to end on the same toy hybrid tests/test_serving_mtp.py uses.

THE bar here is distribution identity, not transcript identity. A sampled MTP
run and a sampled serial run consume the generator differently, so their
transcripts differ even seed-matched; what must hold is that every emitted
token is drawn from the distribution the serial sampler would have used at
that position. speculative_sample is pure, so that claim is checked where it
is cheap — fixed p and q rows, 200k seeded steps per case, total variation
against p_0 and against p_1 given the first draft was accepted — for the four
shapes of q that matter. The end-to-end tests then pin the PLUMBING: at a
temperature so low every distribution is one-hot, the sampled cycle (draft
sampling, q rows, p rows, the accept rule, state rollback, head rebuild)
agrees with the greedy transcript except at a genuine near-tie (spec_agree)
— token-exact between speculation on and off is not a property this repo
asserts — with and without penalties, with a bad drafter and
with an oracle one.

The toy fixtures are imported from test_serving_mtp rather than duplicated;
the greedy tests there are untouched and remain the proof for that path.
"""

import pytest
import torch

from drinkme.serving import mtp
from drinkme.serving import speculative as speculative_mod
from drinkme.serving.engine import SampleParams
from drinkme.serving.sampling import PenaltyState, sample_next, sample_probs
from drinkme.serving.speculative import AcceptanceWindow, SpeculativeAudit, speculative_sample

# the toy hybrid, its session fixtures (the CPU chunk-kernel swap included)
# and the run helpers — pytest registers imported fixtures in this module too
from test_serving_mtp import (  # noqa: F401
    PROMPTS, _engine, _greedy, _msgs, _run, _torch_chunk_on_cpu, mtp_depth, toy,
)

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


# ------------------------------------------------------------ the theorem --


def _rows(*vals):
    t = torch.tensor(vals, dtype=torch.float32)
    return t / t.sum(dim=-1, keepdim=True)


# depth 2: three target rows p_0..p_2, two draft rows q_0, q_1, vocab 8
P = _rows([0.50, 0.20, 0.10, 0.08, 0.05, 0.04, 0.02, 0.01],
          [0.05, 0.05, 0.30, 0.10, 0.25, 0.05, 0.10, 0.10],
          [0.125] * 8)

CASES = {
    # (a) q == p: every draft accepted, the bonus token does the emitting
    "q_equals_p": P[:2].clone(),
    # (b) q far from p: p's mass reversed — mostly rejected, the residual
    # resample carries the distribution
    "q_far_from_p": _rows([0.01, 0.02, 0.04, 0.05, 0.08, 0.10, 0.20, 0.50],
                          [0.10, 0.10, 0.05, 0.25, 0.10, 0.30, 0.05, 0.05]),
    # (c) q has zero mass where p has mass: tokens 0, 1, 7 can only ever
    # arrive through the residual
    "q_zero_where_p_has_mass": _rows([0.0, 0.0, 0.3, 0.3, 0.2, 0.1, 0.1, 0.0],
                                     [0.0, 0.5, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0]),
    # (d) p's support inside q's: q proposes tokens p forbids; every one of
    # them must be rejected (p/q = 0) and never emitted
    "p_support_inside_q": _rows([1.0] * 8, [1.0] * 8),
}
P_NARROW = _rows([0.6, 0.4, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0.5, 0.3, 0.2, 0, 0],
                 [0.125] * 8)

N_STEPS = 200_000
# Expected TV noise for V=8 at N=200k is ~2e-3 (measured 0.0002-0.002 over
# the four cases); the conditional check has only the accepted trials
# (~25% of N in three cases) so its noise is ~4e-3 (measured up to 0.008).
# A wrong rule — resampling p instead of the residual, ignoring q — lands
# at 0.05+ in the adversarial cases. Seeds are fixed per case: these runs
# are reproducible, which is the point of "seeded".
TV_TOL = 0.01
TV_TOL_COND = 0.015
SEEDS = {name: 100 + i for i, name in enumerate(CASES)}


def _tv(counts, target):
    emp = counts / counts.sum()
    return 0.5 * float((emp - target).abs().sum())


def _run_steps(p, q, n, seed):
    """n seeded speculative steps over fixed rows. Drafts are drawn from q
    up front (that is what "the distributions the draft tokens were actually
    sampled from" means), the decision uses the same generator."""
    g = torch.Generator().manual_seed(seed)
    depth = q.shape[0]
    drafts = [torch.multinomial(q[i], n, replacement=True, generator=g).tolist()
              for i in range(depth)]
    first = torch.zeros(p.shape[1])
    second = torch.zeros(p.shape[1])
    accepted0 = 0
    for t in range(n):
        emitted, m = speculative_sample(p, q, [drafts[i][t] for i in range(depth)], g)
        assert len(emitted) == m + 1
        first[emitted[0]] += 1
        if m >= 1:
            accepted0 += 1
            second[emitted[1]] += 1
    return first, second, accepted0


@pytest.mark.parametrize("name", list(CASES))
def test_first_and_second_tokens_are_distributed_as_the_target(name):
    p = P_NARROW if name == "p_support_inside_q" else P
    q = CASES[name]
    first, second, acc0 = _run_steps(p, q, N_STEPS, seed=SEEDS[name])
    # the first emitted token ~ p_0, whatever q was
    assert _tv(first, p[0]) < TV_TOL, f"{name}: TV(first, p_0) = {_tv(first, p[0]):.4f}"
    # the second emitted token, given the first draft was accepted, ~ p_1
    assert acc0 > 1000, f"{name}: too few first-draft acceptances to check p_1"
    assert _tv(second, p[1]) < TV_TOL_COND, f"{name}: TV(second | acc, p_1) = {_tv(second, p[1]):.4f}"
    # and the acceptance rate at position 0 is the theorem's sum(min(p, q))
    want = float(torch.minimum(p[0], q[0]).sum())
    assert abs(acc0 / N_STEPS - want) < 0.005, f"{name}: acceptance {acc0 / N_STEPS:.4f} vs {want:.4f}"


def test_tokens_outside_the_target_support_are_never_emitted():
    """(d) sharpened: p_0 forbids tokens 2..7; q proposes them 3/4 of the
    time; not one may come out."""
    first, _, _ = _run_steps(P_NARROW, CASES["p_support_inside_q"], 20_000, seed=4)
    assert first[2:].sum() == 0


def test_all_accepted_emits_depth_plus_one_with_the_bonus_from_the_last_row():
    """q == p at depth 3: every draft survives, every cycle emits 4 tokens,
    and the 4th is drawn from p_3."""
    p = _rows([0.5, 0.3, 0.2, 0], [0.1, 0.1, 0.4, 0.4], [0.25] * 4, [0.7, 0.1, 0.1, 0.1])
    q = p[:3]
    g = torch.Generator().manual_seed(9)
    n = 50_000
    drafts = [torch.multinomial(q[i], n, replacement=True, generator=g).tolist()
              for i in range(3)]
    bonus = torch.zeros(4)
    for t in range(n):
        emitted, m = speculative_sample(p, q, [drafts[i][t] for i in range(3)], g)
        assert m == 3 and len(emitted) == 4
        assert emitted[:3] == [drafts[i][t] for i in range(3)]
        bonus[emitted[3]] += 1
    assert _tv(bonus, p[3]) < 0.02


def test_zero_mass_residual_falls_back_to_the_target_row():
    """p == q and a draft token q gave no mass (measure-zero in exact
    arithmetic; reachable through float rounding): the rule rejects it, the
    residual is empty, and the token comes from p — never a crash, never
    the forbidden token."""
    p = _rows([0.5, 0.5, 0, 0], [0.25] * 4)
    q = p[:1]
    g = torch.Generator().manual_seed(1)
    for _ in range(200):
        emitted, m = speculative_sample(p, q, [2], g)
        assert m == 0 and emitted[0] in (0, 1)


def test_no_drafts_is_one_token_from_the_first_row():
    p = _rows([0, 1, 0])
    assert speculative_sample(p, [], [], torch.Generator().manual_seed(0)) == ([1], 0)


def test_row_counts_are_checked():
    with pytest.raises(ValueError, match="target rows"):
        speculative_sample(P[:2], P[:2], [0, 1])


def test_seeded_decisions_reproduce():
    a = [speculative_sample(P, CASES["q_far_from_p"], [7, 5],
                            torch.Generator().manual_seed(s)) for s in range(50)]
    b = [speculative_sample(P, CASES["q_far_from_p"], [7, 5],
                            torch.Generator().manual_seed(s)) for s in range(50)]
    assert a == b and len({tuple(e) for e, _ in a}) > 1


# ------------------------------------------------------------- the audit --
# docs/serve-speculation.md: the self-referential check that has power under real
# traffic, where the two-arm first-token check does not (it scores a
# wrong-rule-shaped TV=0.000 because a good drafter's first token IS the
# serial arm's draw). Same CASES the theorem tests above
# use, pooled into one audit across all four adversarial q shapes.

AUDIT_N_PER_CASE = 50_000


def _run_audited(audit, n_per_case=AUDIT_N_PER_CASE, seed_offset=500):
    """n_per_case seeded speculative_sample calls per CASES entry, all
    decisions fed to `audit`. Mirrors _run_steps's draft-drawing discipline
    (drafts sampled from q up front, decisions off the same generator)."""
    for name, q in CASES.items():
        p = P_NARROW if name == "p_support_inside_q" else P
        g = torch.Generator().manual_seed(SEEDS[name] + seed_offset)
        depth = q.shape[0]
        drafts = [torch.multinomial(q[i], n_per_case, replacement=True,
                                    generator=g).tolist() for i in range(depth)]
        for t in range(n_per_case):
            speculative_sample(p, q, [drafts[i][t] for i in range(depth)], g, audit=audit)


def test_audit_matches_the_theorem_on_adversarial_q():
    """The check itself, green: pooled over >=50k decided positions across
    every adversarial q shape the theorem tests use, the audit's actual
    acceptance rate must track its own expected rate (sum(min(p, q)), the
    theorem's promise) to within 0.5%, and neither residual-support
    invariant may ever be violated."""
    audit = SpeculativeAudit()
    _run_audited(audit)
    assert audit.decided >= 50_000
    assert abs(audit.actual_rate - audit.expected_rate) < 0.005, (
        f"actual {audit.actual_rate:.4f} vs expected {audit.expected_rate:.4f}")
    assert audit.violations == 0


def test_audit_catches_a_broken_accept_everything_rule(monkeypatch):
    """Prove the instrument can fail before trusting it green (docs/serve-speculation.md):
    force the accept decision itself to always fire — a rule that would take
    every draft regardless of p or q, including tokens p forbids outright
    (q_zero_where_p_has_mass, p_support_inside_q both propose such tokens) —
    and the same check the green test runs must now fail loudly: actual
    saturates at 100% while expected stays the theorem's honest, lower
    number."""
    monkeypatch.setattr(speculative_mod, "_accept", lambda u_i, pi, qi, d: True)
    audit = SpeculativeAudit()
    _run_audited(audit, n_per_case=5_000, seed_offset=600)
    assert audit.actual_rate == 1.0
    assert audit.actual_rate - audit.expected_rate > 0.05
    with pytest.raises(AssertionError):
        assert abs(audit.actual_rate - audit.expected_rate) < 0.005


def test_audit_off_by_default_matches_audit_on_byte_for_byte():
    """audit=None is unchanged: distribution-identity tests pass untouched
    (the parametrized theorem tests above never pass `audit`), and turning
    the sink on must not perturb a single draw — same generator state
    consumed, same emitted stream, seed for seed."""
    name = "q_zero_where_p_has_mass"
    p, q = P, CASES[name]
    depth = q.shape[0]
    n = 5_000
    gd = torch.Generator().manual_seed(SEEDS[name] + 700)
    drafts = [torch.multinomial(q[i], n, replacement=True, generator=gd).tolist()
              for i in range(depth)]

    def run(audit):
        g = torch.Generator().manual_seed(321)
        return [speculative_sample(p, q, [drafts[i][t] for i in range(depth)], g,
                                   audit=audit) for t in range(n)]

    off = run(None)
    on = run(SpeculativeAudit())
    assert off == on


# ---------------------------------------------------------- transform parity --


PARAMS = [
    SampleParams(temperature=0.7),
    SampleParams(temperature=1.5, top_k=5),
    SampleParams(temperature=0.9, top_p=0.8),
    SampleParams(temperature=1.0, top_k=7, top_p=0.95, repetition_penalty=1.3,
                 presence_penalty=0.5, frequency_penalty=0.2),
]


@pytest.mark.parametrize("params", PARAMS, ids=[str(i) for i in range(len(PARAMS))])
def test_sample_probs_is_the_distribution_sample_next_draws_from(params):
    """B2's parity, at the level that matters: the serial sampler's draw from
    a generator state equals a multinomial over sample_probs from the same
    state — same row, same penalty history, every seed."""
    torch.manual_seed(0)
    row = torch.randn(50) * 2
    prev, gen_ids = [3, 9, 9, 14, 20], [9, 9, 14]
    for seed in range(60):
        want = sample_next(row, params, torch.Generator().manual_seed(seed),
                           prev_ids=prev, gen_ids=gen_ids)
        probs = sample_probs(row, params, prev_ids=prev, gen_ids=gen_ids)
        got = int(torch.multinomial(probs, 1, generator=torch.Generator().manual_seed(seed)))
        assert got == want
    assert abs(float(probs.sum()) - 1.0) < 1e-5
    if params.top_k:
        assert int((probs > 0).sum()) <= params.top_k


def test_sample_probs_state_path_is_bitwise_the_list_path():
    """The PenaltyState the serial loop advances per accepted token and the
    extended id lists MTP's rows use are the same evolution, bit for bit —
    so p at a speculative position IS the serial sampler's p there."""
    params = PARAMS[3]
    torch.manual_seed(1)
    prompt = [1, 4, 4, 8]
    generated = [4, 12, 12, 12, 30, 2]
    state = PenaltyState(50)
    state.observe(prompt)
    for i in range(len(generated) + 1):
        row = torch.randn(50) * 3
        by_state = sample_probs(row, params, state=state)
        by_lists = sample_probs(row, params, prev_ids=prompt + generated[:i],
                                gen_ids=generated[:i])
        assert torch.equal(by_state, by_lists), f"position {i}"
        if i < len(generated):
            state.accept(generated[i])


def test_sample_probs_refuses_greedy():
    with pytest.raises(ValueError, match="temperature 0"):
        sample_probs(torch.tensor([1.0, 2.0]), SampleParams(temperature=0.0))


def test_sampled_rows_see_the_serial_id_history(toy):
    """The plumbing of B2 inside a cycle: draft row i and verify row i are
    both asked for with exactly the tokens decided ahead of them this cycle,
    so the penalty history is the one the serial loop would have shown that
    position. Checked on a real Speculator over the toy, with a `probs` that
    records what it was told."""
    from transformers import StaticCache

    model, tok, head, cfg = toy
    params = SampleParams(temperature=1.0, repetition_penalty=1.2)
    ids = tok("hello world the quick")["input_ids"]
    cache = StaticCache(config=cfg, max_cache_len=64)
    spec = mtp.Speculator(head, model, depth=3, sampled=True)
    seen: list[list[int]] = []

    def probs(row, extra):
        seen.append(list(extra))
        return sample_probs(row, params)

    with torch.inference_mode():
        hidden, logits = mtp.forward_with_hidden(model, torch.tensor([ids]), cache,
                                                 torch.arange(len(ids)))
        spec.begin(cache, ids, hidden, 0)
        first = int(torch.multinomial(sample_probs(logits[-1], params), 1))
        emitted = spec.cycle(first, None, 10, probs=probs,
                             generator=torch.Generator().manual_seed(0))
    k = 3
    assert len(seen) == k + (k + 1)  # k draft rows, k+1 verify rows
    drafts = seen[2 * k]  # the last verify row is conditioned on every draft
    assert len(drafts) == k
    for i in range(k):
        assert seen[i] == drafts[:i], f"draft row {i}"
    for i in range(k + 1):
        assert seen[k + i] == drafts[:i], f"verify row {i}"
    assert 1 <= len(emitted) <= k + 1 and spec.written == len(ids) + len(emitted)


def test_draft_vocab_lever_gives_zero_mass_outside_the_subset(toy):
    """With the reduced-vocab draft on, the sampled draft's q row is the
    subset's distribution scattered onto the full vocab: zero elsewhere, so
    a token the draft cannot propose is one the accept rule sees as q = 0."""
    from drinkme.serving import draft_vocab

    model, tok, head, cfg = toy
    subset = torch.arange(0, cfg.vocab_size, 3)
    proj, _ = draft_vocab.make_projection(head.lm_head, subset)
    params = SampleParams(temperature=1.0)
    head.reset()
    head.draft_proj = proj
    try:
        with torch.inference_mode():
            h = torch.randn(1, 1, cfg.hidden_size)
            g = torch.Generator().manual_seed(2)

            def propose(row, so_far):
                q = sample_probs(row, params)
                return int(torch.multinomial(q, 1, generator=g)), q

            rows, qs = head.draft_sampled(h, 5, 0, 3, propose)
    finally:
        head.draft_proj = None
    allowed = set(subset.tolist())
    toks = rows[0, 1:].tolist()  # the verify rows: [token, 3 drafts]
    assert len(toks) == 3
    for t, q in zip(toks, qs):
        assert t in allowed
        mask = torch.ones(cfg.vocab_size, dtype=torch.bool)
        mask[subset] = False
        assert float(q[mask].sum()) == 0.0 and abs(float(q.sum()) - 1) < 1e-5


# ------------------------------------------------------------- adaptive bail --


def test_acceptance_window_trips_at_the_floor_once_and_not_before():
    w = AcceptanceWindow(32, 0.15)
    # a partial window never trips, however bad
    assert not any(w.observe(0, 4) for _ in range(7)) and not w.tripped
    # 32 positions, 4 accepted = 12.5% < 15%: trips, once
    assert w.observe(4, 4) is True and w.tripped and w.rate_at_trip == 4 / 32
    assert w.observe(0, 4) is False and w.tripped and w.rate_at_trip == 4 / 32
    # 5 of 32 = 15.6% sits above the floor: no trip — and the window ROLLS:
    # the lone early acceptance keeps it above the floor until it rolls off
    w = AcceptanceWindow(32, 0.15)
    for _ in range(6):
        w.observe(0, 4)
    assert w.observe(1, 4) is False  # 28 positions: still partial
    assert w.observe(4, 4) is False and not w.tripped  # 5 of 32
    assert [w.observe(0, 4) for _ in range(7)] == [False] * 6 + [True]  # 4 of 32
    # rolling: a good start does not protect a bad tail forever
    w = AcceptanceWindow(32, 0.15)
    for _ in range(8):
        assert not w.observe(4, 4)
    tripped = [w.observe(0, 4) for _ in range(8)]
    assert tripped.count(True) == 1 and w.rate_at_trip < 0.15
    # floor 0 = off
    w = AcceptanceWindow(32, 0.0)
    assert not any(w.observe(0, 4) for _ in range(20))


def _spy_speculator(monkeypatch):
    """The Speculator an engine builds next, captured for its stats."""
    box = {}
    real = mtp.Speculator.__init__

    def init(self, *a, **kw):
        real(self, *a, **kw)
        box["spec"] = self

    monkeypatch.setattr(mtp.Speculator, "__init__", init)
    return box


NEAR_GREEDY = 1e-4  # every transformed row is one-hot on the toy, so the
                    # sampled cycle's accept rule degenerates to the greedy
                    # compare — it then agrees with the greedy transcript
                    # except at a genuine near-tie (spec_agree), the same bar
                    # as every other arm-vs-arm comparison in this repo


def test_adaptive_bail_trips_once_and_the_request_finishes_serially(toy, monkeypatch, capsys):
    """Random weights draft badly (acceptance ~0): after one full window the
    request must bail — one stderr line naming the rate — and the remaining
    tokens must come from the serial loop, still agreeing with it (modulo a
    near-tie)."""
    from drinkme.serving import engines
    from spec_agree import (assert_agrees_or_forks_at_a_near_tie, capture,
                            capture_sampled, ordered_ids)

    box = _spy_speculator(monkeypatch)
    p = PROMPTS[0]
    picks_ref, rows_ref, restore = capture(engines)
    try:
        ref, _ = _run(_engine(toy, False), _msgs(p), _greedy())
    finally:
        restore()
    eng = _engine(toy, True)
    ids_on, rows_on, restore = capture_sampled(mtp, engines)
    try:
        with mtp_depth(4):
            res, _ = _run(eng, _msgs(p), SampleParams(temperature=NEAR_GREEDY, seed=1,
                                                      max_tokens=64))
    finally:
        restore()
    err = capsys.readouterr().err
    s = box["spec"].stats()
    assert s["bailed"] and s["bail_rate"] < 0.15
    assert s["drafted"] == 4 * s["cycles"] >= 32  # tripped on a full window, then stopped drafting
    assert err.count("adaptive bail") == 1 and f"{s['bail_rate']:.0%}" in err
    assert "bailed to serial" in err  # the per-request accounting line
    assert res.finish_reason == "length" and res.completion_tokens == 64
    assert res.completion_tokens > s["cycles"] + s["accepted"] + 1  # the serial loop finished it
    assert_agrees_or_forks_at_a_near_tie(ids_on, ordered_ids(picks_ref), rows_on, rows_ref)


def test_adaptive_bail_rebuilds_the_serial_penalty_state(toy, monkeypatch):
    """Greedy with penalties bails on the toy too (guard 2 is a property of
    MTP, not of sampling): the serial loop's PenaltyState is rebuilt at the
    switch, and the stream must still agree with the serial one (modulo a
    near-tie)."""
    from drinkme.serving import engines
    from spec_agree import assert_agrees_or_forks_at_a_near_tie, capture, ordered_ids

    box = _spy_speculator(monkeypatch)
    params = dict(max_tokens=64, repetition_penalty=1.35, presence_penalty=0.4,
                  frequency_penalty=0.3)
    picks_off, rows_off, restore = capture(engines)
    try:
        off, _ = _run(_engine(toy, False), _msgs(PROMPTS[6]), SampleParams(temperature=0.0, **params))
    finally:
        restore()
    picks_on, rows_on, restore = capture(engines)
    try:
        with mtp_depth(4):
            on, _ = _run(_engine(toy, True), _msgs(PROMPTS[6]), SampleParams(temperature=0.0, **params))
    finally:
        restore()
    assert box["spec"].stats()["bailed"]
    assert on.completion_tokens == off.completion_tokens == 64
    assert_agrees_or_forks_at_a_near_tie(ordered_ids(picks_on), ordered_ids(picks_off),
                                         rows_on, rows_off)


def test_no_bail_before_a_full_window(toy, monkeypatch, capsys):
    """8 tokens at depth 4 is at most 7 cycles = 28 draft positions: a bad
    partial window is not evidence yet."""
    box = _spy_speculator(monkeypatch)
    with mtp_depth(4):
        _run(_engine(toy, True), _msgs(PROMPTS[1]), _greedy(max_tokens=8))
    s = box["spec"].stats()
    assert not s["bailed"] and s["bail_rate"] is None and 0 < s["drafted"] < 32
    assert "adaptive bail" not in capsys.readouterr().err


def test_bail_disabled_keeps_every_cycle_speculative(toy, monkeypatch):
    """The greedy identity tests next door now bail after one window on the
    toy's bad drafter; with the floor at 0 the deepest-rewind case (every
    cycle rejected at position 0) runs speculatively for the whole
    generation and must still agree with the serial stream (modulo a
    near-tie)."""
    from test_serving_mtp import _force_draft

    from drinkme.serving import engines
    from spec_agree import assert_agrees_or_forks_at_a_near_tie, capture, ordered_ids

    monkeypatch.setattr(mtp, "BAIL_FLOOR", 0.0)
    box = _spy_speculator(monkeypatch)
    model, tok, head, _ = toy
    p = PROMPTS[9]
    picks_ref, rows_ref, restore = capture(engines)
    try:
        ref, _ = _run(_engine(toy, False), _msgs(p), _greedy())
    finally:
        restore()
    eng = _engine(toy, True)
    real = _force_draft(head, lambda token, k: [(token + 1) % 96] * k)
    picks, rows, restore = capture(engines)
    try:
        with mtp_depth(4):
            res, _ = _run(eng, _msgs(p), _greedy())
    finally:
        head.draft = real
        restore()
    s = box["spec"].stats()
    assert not s["bailed"] and s["accepted"] == 0 and s["cycles"] == 63
    assert_agrees_or_forks_at_a_near_tie(ordered_ids(picks), ordered_ids(picks_ref),
                                         rows, rows_ref)


# ------------------------------------------------ the sampled cycle, end to end --


def test_sampled_requests_speculate_and_constrained_ones_still_do_not(toy):
    with mtp_depth(4):
        s = mtp.maybe_speculate(toy[2], toy[0], SampleParams(temperature=0.7))
        assert s is not None and s.sampled
        g = mtp.maybe_speculate(toy[2], toy[0], SampleParams(temperature=0.0))
        assert g is not None and not g.sampled
        assert mtp.maybe_speculate(toy[2], toy[0], SampleParams(
            temperature=0.7, output_schema={"type": "json_object"})) is None
    with mtp_depth(0):
        assert mtp.maybe_speculate(toy[2], toy[0], SampleParams(temperature=0.7)) is None


@pytest.mark.parametrize("i", [0, 3, 7, 12])
def test_near_greedy_sampling_agrees_with_the_greedy_stream(toy, monkeypatch, i):
    """The plumbing, end to end: at a temperature where every p and q is
    one-hot the accept rule degenerates to the greedy compare, so the
    sampled cycle — sampled drafts, q rows, p rows, rollback, head rebuild —
    agrees with the serial greedy transcript except at a genuine near-tie
    (spec_agree; token-exact between speculation on and off is not a
    property this repo asserts), with the bail off so every
    token goes through it."""
    from drinkme.serving import engines
    from spec_agree import (assert_agrees_or_forks_at_a_near_tie, capture,
                            capture_sampled, ordered_ids)

    monkeypatch.setattr(mtp, "BAIL_FLOOR", 0.0)
    box = _spy_speculator(monkeypatch)
    p = PROMPTS[i]
    picks_ref, rows_ref, restore = capture(engines)
    try:
        ref, _ = _run(_engine(toy, False), _msgs(p), _greedy())
    finally:
        restore()
    ids_on, rows_on, restore = capture_sampled(mtp, engines)
    try:
        with mtp_depth(4):
            res, _ = _run(_engine(toy, True), _msgs(p),
                          SampleParams(temperature=NEAR_GREEDY, seed=i, max_tokens=64))
    finally:
        restore()
    s = box["spec"].stats()
    assert s["sampled"] and s["cycles"] > 0 and not s["bailed"]
    assert res.completion_tokens == 64
    assert_agrees_or_forks_at_a_near_tie(ids_on, ordered_ids(picks_ref), rows_on, rows_ref)


def test_near_greedy_sampling_with_penalties_agrees_with_the_greedy_stream(toy, monkeypatch):
    """Penalties over speculative positions: p_i is built with the id history
    the serial loop would have had, so greedy-with-penalties agrees except
    at a genuine near-tie."""
    from drinkme.serving import engines
    from spec_agree import (assert_agrees_or_forks_at_a_near_tie, capture,
                            capture_sampled, ordered_ids)

    monkeypatch.setattr(mtp, "BAIL_FLOOR", 0.0)
    pen = dict(repetition_penalty=1.35, presence_penalty=0.4, frequency_penalty=0.3,
               max_tokens=48)
    p = PROMPTS[6]
    picks_ref, rows_ref, restore = capture(engines)
    try:
        ref, _ = _run(_engine(toy, False), _msgs(p), SampleParams(temperature=0.0, **pen))
    finally:
        restore()
    ids_on, rows_on, restore = capture_sampled(mtp, engines)
    try:
        with mtp_depth(4):
            res, _ = _run(_engine(toy, True), _msgs(p),
                          SampleParams(temperature=NEAR_GREEDY, seed=3, **pen))
    finally:
        restore()
    assert_agrees_or_forks_at_a_near_tie(ids_on, ordered_ids(picks_ref), rows_on, rows_ref)


def _force_draft_sampled(head, fn, vocab):
    """Swap the sampled draft policy while keeping the head's KV real: the
    genuine draft still runs, the proposals and their q rows are replaced by
    fn(token, k)'s tokens with one-hot q rows."""
    real = head.draft_sampled

    def forced(h, token, first_entry, k, propose):
        real(h, token, first_entry, k, propose)
        toks = fn(token, k)
        qs = []
        for t in toks:
            q = torch.zeros(vocab)
            q[t] = 1.0
            qs.append(q)
        return toks, qs

    head.draft_sampled = forced
    return real


def test_forced_full_acceptance_on_the_sampled_path(toy, monkeypatch):
    """An oracle drafter with one-hot q: every draft accepted, the bonus token
    drawn from p_depth, the head re-keyed on true hiddens — the other extreme
    the toy cannot produce on its own — and still agrees with the greedy
    transcript modulo a genuine near-tie."""
    from drinkme.serving import engines
    from spec_agree import (assert_agrees_or_forks_at_a_near_tie, capture,
                            capture_sampled, ordered_ids)

    monkeypatch.setattr(mtp, "BAIL_FLOOR", 0.0)
    box = _spy_speculator(monkeypatch)
    model, tok, head, cfg = toy
    p = PROMPTS[5]
    picks_ref, rows_ref, restore = capture(engines)
    try:
        ref, _ = _run(_engine(toy, False), _msgs(p), _greedy())
    finally:
        restore()
    oracle = tok(ref.text)["input_ids"]

    def truth(token, k):
        try:
            j = oracle.index(token)
        except ValueError:
            return [oracle[0]] * k
        return (oracle[j + 1:j + 1 + k] + [0] * k)[:k]

    eng = _engine(toy, True)
    real = _force_draft_sampled(head, truth, cfg.vocab_size)
    ids_on, rows_on, restore = capture_sampled(mtp, engines)
    try:
        with mtp_depth(4):
            res, _ = _run(eng, _msgs(p), SampleParams(temperature=NEAR_GREEDY, seed=5,
                                                      max_tokens=64))
    finally:
        head.draft_sampled = real
        restore()
    s = box["spec"].stats()
    assert s["acceptance"] > 0.9 and s["cycles"] < 20
    assert res.completion_tokens == 64
    assert_agrees_or_forks_at_a_near_tie(ids_on, ordered_ids(picks_ref), rows_on, rows_ref)


def test_seed_matched_sampled_mtp_runs_reproduce(toy, monkeypatch):
    """B4: one generator per request, seeded from the request's seed — the
    property the GPU statistical checks will lean on."""
    box = _spy_speculator(monkeypatch)
    params = SampleParams(temperature=0.9, seed=11, max_tokens=32)
    with mtp_depth(4):
        a, _ = _run(_engine(toy, True), _msgs(PROMPTS[2]), params)
        sa = box["spec"].stats()
        b, _ = _run(_engine(toy, True), _msgs(PROMPTS[2]), params)
    assert sa["sampled"] and sa["cycles"] > 0
    assert a.text == b.text and a.completion_tokens == b.completion_tokens == 32


def test_sampled_request_without_a_seed_still_runs(toy, monkeypatch):
    box = _spy_speculator(monkeypatch)
    with mtp_depth(4):
        res, _ = _run(_engine(toy, True), _msgs(PROMPTS[4]),
                      SampleParams(temperature=1.0, top_p=0.9, top_k=20, max_tokens=24))
    assert res.finish_reason == "length" and res.completion_tokens == 24
    assert box["spec"].stats()["sampled"]


def test_sampled_stop_and_max_tokens_mid_cycle_agree(toy, monkeypatch):
    """The mid-cycle stop machinery is shared with the greedy path; make sure
    the sampled cycle's pending tokens go through it the same way — agreeing
    with the serial reference modulo a genuine near-tie (spec_agree)."""
    from drinkme.serving import engines
    from spec_agree import (assert_agrees_or_forks_at_a_near_tie, capture,
                            capture_sampled, ordered_ids)

    monkeypatch.setattr(mtp, "BAIL_FLOOR", 0.0)
    p = PROMPTS[7]
    picks_ref, rows_ref, restore = capture(engines)
    try:
        ref, _ = _run(_engine(toy, False), _msgs(p), _greedy())
    finally:
        restore()
    ids_ref = ordered_ids(picks_ref)
    ids_on, rows_on, restore = capture_sampled(mtp, engines)
    try:
        with mtp_depth(4):
            res, _ = _run(_engine(toy, True), _msgs(p),
                          SampleParams(temperature=NEAR_GREEDY, seed=7, max_tokens=6))
    finally:
        restore()
    assert res.completion_tokens == 6 and res.finish_reason == "length"
    assert_agrees_or_forks_at_a_near_tie(ids_on, ids_ref, rows_on, rows_ref)
    stop = ref.text.split()[3]
    ids_on, rows_on, restore = capture_sampled(mtp, engines)
    try:
        with mtp_depth(4):
            res, _ = _run(_engine(toy, True), _msgs(p),
                          SampleParams(temperature=NEAR_GREEDY, seed=7, stop=[stop],
                                       max_tokens=64))
    finally:
        restore()
    assert res.finish_reason == "stop"
    assert_agrees_or_forks_at_a_near_tie(ids_on, ids_ref, rows_on, rows_ref)


# ------------------------------------------------------- the audit, end to end --
# A2/A3 (docs/serve-speculation.md): the toy-level tests above pin speculative_sample's
# own bookkeeping; these pin the plumbing — a real sampled Speculator over the
# toy picks it up, the accounting line reports it, DRINKME_SPEC_AUDIT=0 turns
# it off, and the module-level cumulative counter fires every 50 cycles.


def test_audit_flows_through_a_real_speculator_and_its_accounting_line(toy, monkeypatch, capsys):
    """A2, on the real toy (not just the pure function above): a sampled
    request's Speculator.stats()["audit"] is populated and honest — no
    violations, even against the toy's bad random-weight drafter — and the
    per-request accounting line carries `expect r_e%` and `audit ok`."""
    box = _spy_speculator(monkeypatch)
    with mtp_depth(4):
        res, _ = _run(_engine(toy, True), _msgs(PROMPTS[8]),
                      SampleParams(temperature=1.0, seed=13, max_tokens=64))
    a = box["spec"].stats()["audit"]
    assert res.completion_tokens == 64
    assert a is not None and a["decided"] > 0 and a["violations"] == 0
    assert 0.0 <= a["actual_rate"] <= 1.0 and 0.0 <= a["expected_rate"] <= 1.0
    err = capsys.readouterr().err
    assert "expect " in err and "audit ok" in err


def test_audit_can_be_disabled_via_env(toy, monkeypatch, capsys):
    """DRINKME_SPEC_AUDIT=0: the per-request sums stay None and the
    accounting line drops `expect` / `audit` entirely — a profile that cares
    about the extra torch.minimum(...).sum() per position can turn it off
    without losing anything else the line reports."""
    monkeypatch.setenv("DRINKME_SPEC_AUDIT", "0")
    box = _spy_speculator(monkeypatch)
    with mtp_depth(4):
        _run(_engine(toy, True), _msgs(PROMPTS[8]),
             SampleParams(temperature=1.0, seed=13, max_tokens=64))
    assert box["spec"].stats()["audit"] is None
    err = capsys.readouterr().err
    assert "expect " not in err and "audit" not in err


def test_accounting_line_carries_both_denominators(toy, monkeypatch, capsys):
    """D4: one line, two honest denominators — drafted (the bail window's own
    accounting: undecided positions after a rejection count as rejected) and
    decided (the audit's — what `expect` is actually a mean over). Comparing
    `expect` against accepted/drafted was the bug this replaces."""
    box = _spy_speculator(monkeypatch)
    with mtp_depth(4):
        _run(_engine(toy, True), _msgs(PROMPTS[8]),
             SampleParams(temperature=1.0, seed=13, max_tokens=64))
    s = box["spec"].stats()
    a = s["audit"]
    assert a is not None and a["decided"] > 0
    assert a["accepted"] == s["accepted"]  # same accepts, counted two ways
    assert a["decided"] <= s["drafted"]  # decided is a subset of drafted
    err = capsys.readouterr().err
    assert f"{s['accepted']}/{s['drafted']} drafted" in err
    assert f"{a['accepted']}/{a['decided']} decided" in err
    assert f"({a['actual_rate']:.0%}, expect {a['expected_rate']:.0%})" in err


def test_accounting_line_stays_one_denominator_without_the_audit(toy, monkeypatch, capsys):
    """Greedy (no audit) keeps the old, single-denominator shape — there is
    no second denominator to honestly report without one."""
    box = _spy_speculator(monkeypatch)
    with mtp_depth(3):
        _run(_engine(toy, True), _msgs(PROMPTS[0]), _greedy(max_tokens=24))
    s = box["spec"].stats()
    assert s["audit"] is None
    err = capsys.readouterr().err
    assert f"{s['accepted']}/{s['drafted']} drafts accepted" in err
    assert "decided" not in err and "expect " not in err


def test_audit_cumulative_prints_every_50_sampled_cycles(toy, monkeypatch, capsys):
    """A3: the module-level counter across requests, reset here so the other
    audit tests in this session don't leave it mid-window. Depth 1 with a
    generous max_tokens guarantees at least 50 sampled cycles (worst case
    for cycle count is every draft accepted, 2 tokens/cycle: 150/2 = 75)
    regardless of how lucky the toy's drafter gets."""
    monkeypatch.setattr(mtp, "_AUDIT_CUMULATIVE", mtp.SpeculativeAudit())
    monkeypatch.setattr(mtp, "_audit_cycles", 0)
    with mtp_depth(1):
        _run(_engine(toy, True), _msgs(PROMPTS[8]),
             SampleParams(temperature=1.0, seed=13, max_tokens=150))
    err = capsys.readouterr().err
    assert "[drinkme.mtp] audit:" in err
    assert "decided, actual" in err and "vs expected" in err and "violations" in err
