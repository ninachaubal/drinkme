"""The adaptive bail RE-ARMS — CPU only,
on the toy hybrid tests/test_serving_mtp.py builds.

The finding: on the lit 27B an ordinary sampled 256-token answer tripped the
bail ~30-40 tokens in on 2 of 3 runs and the remaining ~220 tokens decoded
at spec-off speed, because a trip was a one-way latch for the rest of the
request. What these tests pin is the state machine that replaced the latch,
against a SCRIPTED head — a draft policy that is right or wrong by
generated position, so a request can be walked through "bad region, good
region, bad region" deterministically:

  * drafting RESUMES after the serial stretch and the tokens/step recover
    when the bad region is over; `DRINKME_MTP_REARM=0` is a one-way latch;
  * the backoff doubles on a re-trip and resets once a re-armed window
    survives; the stretch lengths are read off `Speculator.trip` itself;
  * the boundary is exact in both directions: the greedy transcript agrees
    with the serial one (spec_agree, modulo a near-tie) across trip and
    re-arm, with and without penalties; the serial step the stretch takes
    is BITWISE the engine's own; the head's KV after a re-arm is the one a
    head fed every position would hold (no hole, no rope shift);
  * the sampled audit accumulates across trips and re-arms — nothing
    double-counted, nothing dropped, zero violations — and /metrics counts
    trips and re-arms.

Every test here failed on the latch before the change.
"""

import os

import pytest
import torch

from drinkme.serving import engines, metrics, mtp
from drinkme.serving.engine import SampleParams
from drinkme.serving.template import build_prompt
from spec_agree import (assert_agrees_or_forks_at_a_near_tie, capture,
                        capture_sampled, ordered_ids)

# the toy hybrid, its session fixtures (the CPU chunk-kernel swap included)
# and the run helpers — pytest registers imported fixtures in this module too
from test_serving_mtp import (  # noqa: F401
    PROMPTS, _engine, _greedy, _msgs, _run, _torch_chunk_on_cpu, mtp_depth, toy,
)
from test_serving_speculative import NEAR_GREEDY, _spy_speculator

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")

V = 96  # the toy's vocab


class rearm_env:
    """DRINKME_MTP_REARM for a block, restored after. The engine reads it
    once per engine (mtp.spec_plan), so every test builds a fresh engine
    inside the block."""

    def __init__(self, n):
        self.n = n

    def __enter__(self):
        self.old = os.environ.get("DRINKME_MTP_REARM")
        if self.n is None:
            os.environ.pop("DRINKME_MTP_REARM", None)
        else:
            os.environ["DRINKME_MTP_REARM"] = str(self.n)
        mtp._warned.clear()
        return self

    def __exit__(self, *a):
        if self.old is None:
            os.environ.pop("DRINKME_MTP_REARM", None)
        else:
            os.environ["DRINKME_MTP_REARM"] = self.old
        mtp._warned.clear()


def _n_prompt(eng, p):
    return len(build_prompt(eng.tok, _msgs(p), template_kwargs=eng.template_kwargs))


def _serial_reference(toy, p, params=None):
    """The serial greedy run for a prompt: (result, ids by generated
    position, decision rows) — the oracle a scripted head drafts from and
    the transcript the MTP run is held to."""
    picks, rows, restore = capture(engines)
    try:
        with mtp_depth(0):  # the serial loop itself, not the n-gram proposer
            res, _ = _run(_engine(toy, False), _msgs(p), params or _greedy(max_tokens=200))
    finally:
        restore()
    return res, ordered_ids(picks), rows


def _script_head(head, n_prompt, oracle, good, sampled_good="oracle"):
    """Swap the head's draft policy for one that is RIGHT (the serial
    transcript's own next tokens) when the generated position of the token
    it continues from satisfies `good(pos)` and WRONG by construction
    (oracle + 1 mod V — never the true token) elsewhere. The genuine draft
    still runs first so the head's KV bookkeeping stays real
    (test_serving_mtp._force_draft's shape), on both the greedy and the
    sampled entry points; a forced sampled draft hands back no q rows, so
    the cycle treats it as a point-mass proposal (mtp.cycle: `qs is None`
    -> speculative.point_mass_rows), which is exact.

    On the SAMPLED entry point `sampled_good="real"` makes a good region the
    real head's own sampled draft instead: a transcript sampled at a real
    temperature leaves the greedy oracle within a few tokens, so "the
    oracle's next token" is no longer right there, while the toy's random
    head at temperature 0.7 proposes from a q broad enough to be accepted
    well above the floor (its p/q is near 1). The bad region stays the
    forced-wrong point mass, accepted with probability p[wrong] — ~1% on
    this toy. A NEAR-greedy run keeps the oracle: its rows are one-hot and
    its transcript is the greedy one.

    Returns (restore, log): `log` collects (generated position, k) per draft
    so a test can see where the head was asked."""
    real_draft, real_sampled = head.draft, head.draft_sampled
    log = []

    def script(token, first_entry, k):
        pos = first_entry + 1 - n_prompt  # generated index of `token`
        log.append((pos, k))
        nxt = oracle[pos + 1:pos + 1 + k]
        nxt = (nxt + [0] * k)[:k]
        if good(pos):
            return list(nxt)
        return [(t + 1) % V for t in nxt]

    def draft(h, token, first_entry, k):
        real_draft(h, token, first_entry, k)
        return script(token, first_entry, k)

    def draft_sampled(h, token, first_entry, k, propose):
        out = real_sampled(h, token, first_entry, k, propose)
        pos = first_entry + 1 - n_prompt
        if sampled_good == "real" and good(pos):
            log.append((pos, k))
            return out
        return script(token, first_entry, k), None

    head.draft, head.draft_sampled = draft, draft_sampled

    def restore():
        head.draft, head.draft_sampled = real_draft, real_sampled

    return restore, log


def _spy_trips(monkeypatch):
    """Every serial stretch `Speculator.trip` hands the engine, in order.
    (Empty on a tree without `trip` — the latch — so the tests below reach
    their behavioural assertion there instead of an AttributeError.)"""
    stretches = []
    real = getattr(mtp.Speculator, "trip", None)
    if real is None:
        return stretches

    def trip(self):
        n = real(self)
        stretches.append(n)
        return n

    monkeypatch.setattr(mtp.Speculator, "trip", trip)
    return stretches


# ------------------------------------------------------- re-arm vs latch --


def test_drafting_resumes_after_the_bad_region(toy, monkeypatch, capsys):
    """THE finding, on the toy: a head that is wrong for generated positions
    0..39 and right after them. The window trips ~9 tokens in (8 cycles x 4
    rejected drafts = the full 32-position window), the request goes serial
    for 64 tokens, re-arms at ~73 — past the bad region — and every cycle
    after that accepts all four drafts. The latch would have decoded the
    other 190 tokens serially: accepted stays 0 and tok/step ~1. This test
    is about the re-arm, not the trigger, so the window is pinned to the
    32 (the default is 64) — the trip and re-arm
    positions below are counted against that pinned window."""
    model, tok, head, _ = toy
    p = PROMPTS[2]
    ref, oracle, rows_ref = _serial_reference(toy, p)
    box = _spy_speculator(monkeypatch)
    monkeypatch.setattr(mtp, "BAIL_WINDOW", 32)
    with rearm_env(None), mtp_depth(4):  # unset = the default, 64
        eng = _engine(toy, True)
        restore_head, log = _script_head(head, _n_prompt(eng, p), oracle,
                                         good=lambda pos: pos >= 40)
        picks, rows, restore = capture(engines)
        try:
            res, _ = _run(eng, _msgs(p), _greedy(max_tokens=200))
        finally:
            restore()
            restore_head()
    err = capsys.readouterr().err
    s = box["spec"].stats()
    # drafting RESUMED: the latch never accepts a draft again on this head
    # (as a latch: accepted == 0, the head is never asked past position 8)
    assert s["accepted"] > 0, "the latch: no cycle ran after the trip"
    after = [(pos, k) for pos, k in log if pos >= 40]
    assert after and len(after) >= 20
    assert s["trips"] == 1 and s["rearms"] == 1 and not s["serial_at_end"]
    assert s["bailed"] and s["bail_rate"] < 0.15
    # ...and the resumed cycles are the good region's: every one accepts k
    # every draft accepted, every cycle after (the last cycle's k is what
    # max_tokens left it)
    assert s["accepted"] == sum(k for _, k in after)
    assert s["drafted"] == sum(k for _, k in log)
    # tok/step over the whole request: 200 tokens over (cycles + serial
    # forwards) steps. The latch is ~1.0; with the re-arm the ~120 tokens
    # after it come 5 per step.
    serial_forwards = 64
    steps = s["cycles"] + serial_forwards
    assert res.completion_tokens == 200 and res.finish_reason == "length"
    assert res.completion_tokens / steps > 1.8, (res.completion_tokens, steps)
    # the log: one trip line naming the stretch, one re-arm line
    assert err.count("adaptive bail") == 1
    assert "serial for the next 64 tokens" in err
    assert err.count("after 64 serial tokens (trip 1)") == 1
    # the positions: 8 rejecting cycles = 8 tokens to the trip, +64 serial
    assert "adaptive bail at +8:" in err and "re-armed at +72 " in err
    assert "bailed to serial 1× (last at" in err and "re-armed 1×" in err
    assert "serial at end" not in err
    # the boundary is exact in both directions: the transcript is the serial
    # one, modulo a near-tie
    assert_agrees_or_forks_at_a_near_tie(ordered_ids(picks), oracle, rows, rows_ref)


def test_rearm_zero_is_the_latch(toy, monkeypatch, capsys):
    """DRINKME_MTP_REARM=0: the same scripted head, and the request stays on
    the serial loop to the end — no re-arm line, no cycle after the trip,
    the transcript still the serial one. Window pinned to
    32 (the default is 64) — this test is about the latch, not
    the trigger, and +8/drafted==32 below are counted against that window."""
    model, tok, head, _ = toy
    p = PROMPTS[2]
    ref, oracle, rows_ref = _serial_reference(toy, p)
    box = _spy_speculator(monkeypatch)
    monkeypatch.setattr(mtp, "BAIL_WINDOW", 32)
    with rearm_env(0), mtp_depth(4):
        eng = _engine(toy, True)
        restore_head, log = _script_head(head, _n_prompt(eng, p), oracle,
                                         good=lambda pos: pos >= 40)
        picks, rows, restore = capture(engines)
        try:
            res, _ = _run(eng, _msgs(p), _greedy(max_tokens=200))
        finally:
            restore()
            restore_head()
    err = capsys.readouterr().err
    s = box["spec"].stats()
    assert s["trips"] == 1 and s["rearms"] == 0 and s["serial_at_end"]
    assert s["accepted"] == 0 and s["cycles"] == 8 and s["drafted"] == 32
    assert max(pos for pos, _ in log) < 40  # the head was never asked again
    assert res.completion_tokens == 200
    assert err.count("adaptive bail") == 1 and "re-armed" not in err.split("bailed")[0]
    assert "the rest of this request decodes serially" in err
    assert "re-armed 0×, serial at end" in err
    assert_agrees_or_forks_at_a_near_tie(ordered_ids(picks), oracle, rows, rows_ref)


def test_rearm_from_env():
    with rearm_env(None):
        assert mtp.rearm_from_env() == mtp.REARM_DEFAULT == 64
        assert mtp.spec_plan(True, 4).rearm == 64
    with rearm_env(0):
        assert mtp.rearm_from_env() == 0 and mtp.spec_plan(True, 4).rearm == 0
    with rearm_env(128):
        assert mtp.rearm_from_env() == 128 and mtp.spec_plan(True, 4).rearm == 128
    with rearm_env(-5):
        assert mtp.rearm_from_env() == 0
    with rearm_env("abc"):
        assert mtp.rearm_from_env() == 64
    with rearm_env(""):
        assert mtp.rearm_from_env() == 64


# ------------------------------------------------------------ the backoff --


def test_backoff_doubles_on_a_retrip_and_caps(toy, monkeypatch, capsys):
    """A head that is wrong everywhere: every re-armed window trips again
    after one full window, and each stretch is twice the last, capped.
    REARM=8 keeps the whole ladder inside 128 tokens; the cap is lowered
    to 32 so it is reached too. Window pinned to 32 (the
    default is 64) — this test is about the backoff, not the
    trigger, and the 8-cycles-per-trip cadence below is that window's."""
    model, tok, head, _ = toy
    p = PROMPTS[4]
    ref, oracle, rows_ref = _serial_reference(toy, p)
    monkeypatch.setattr(mtp, "BAIL_WINDOW", 32)
    monkeypatch.setattr(mtp, "REARM_CAP", 32, raising=False)
    stretches = _spy_trips(monkeypatch)
    box = _spy_speculator(monkeypatch)
    with rearm_env(8), mtp_depth(4):
        eng = _engine(toy, True)
        restore_head, log = _script_head(head, _n_prompt(eng, p), oracle,
                                         good=lambda pos: False)
        picks, rows, restore = capture(engines)
        try:
            res, _ = _run(eng, _msgs(p), _greedy(max_tokens=128))
        finally:
            restore()
            restore_head()
    err = capsys.readouterr().err
    s = box["spec"].stats()
    # 8 rejecting cycles to a trip, then the stretch, then 8 more cycles to
    # the next: trips at 8, 24, 48, 88, re-arms at 16, 40, 80, 120
    # (as a latch: one trip, one line, and the head is never asked again)
    assert err.count("adaptive bail") == 4, err.count("adaptive bail")
    assert stretches[:4] == [8, 16, 32, 32]
    assert s["trips"] == len(stretches) == 4 and s["rearms"] == 4
    assert [pos for pos, _ in log if pos in (16, 40, 80, 120)] == [16, 40, 80, 120]
    assert s["accepted"] == 0 and s["drafted"] == sum(k for _, k in log)
    assert err.count("adaptive bail") == s["trips"]
    assert err.count("re-armed at +") == s["rearms"]
    assert "serial for the next 8 tokens" in err and "serial for the next 16 tokens" in err
    assert err.count("serial for the next 32 tokens") == 2
    assert f"bailed to serial {s['trips']}×" in err
    assert_agrees_or_forks_at_a_near_tie(ordered_ids(picks), oracle, rows, rows_ref)


def test_backoff_resets_when_a_rearmed_window_survives(toy, monkeypatch, capsys):
    """Bad region, good region, bad region. Trip 1 (stretch 8), re-arm into
    the still-bad region, trip 2 (stretch 16), re-arm into the good region,
    which holds for >= 2 x BAIL_WINDOW drafted positions — the backoff
    resets — and the third trip, in the second bad region, is back at 8.
    Window pinned to 32 (the default is 64) —
    this test is about the backoff reset, not the trigger, and the good
    region's length below is sized for a 2*32 = 64 REARM_SURVIVE, not the
    new default's 128."""
    model, tok, head, _ = toy
    p = PROMPTS[6]
    ref, oracle, rows_ref = _serial_reference(toy, p)
    stretches = _spy_trips(monkeypatch)
    box = _spy_speculator(monkeypatch)
    monkeypatch.setattr(mtp, "BAIL_WINDOW", 32)
    with rearm_env(8), mtp_depth(4):
        eng = _engine(toy, True)
        restore_head, log = _script_head(head, _n_prompt(eng, p), oracle,
                                         good=lambda pos: 40 <= pos < 130)
        picks, rows, restore = capture(engines)
        try:
            res, _ = _run(eng, _msgs(p), _greedy(max_tokens=200))
        finally:
            restore()
            restore_head()
    err = capsys.readouterr().err
    s = box["spec"].stats()
    # (as a latch: one trip, then serial to the end — one line, no stretch)
    assert err.count("adaptive bail") >= 3, err.count("adaptive bail")
    assert stretches[:3] == [8, 16, 8], stretches
    assert s["trips"] >= 3 and s["rearms"] >= 3
    good_cycles = [(pos, k) for pos, k in log if 40 <= pos < 130]
    assert 4 * len(good_cycles) >= 2 * 32  # the pinned window's REARM_SURVIVE; held that long
    assert s["accepted"] == sum(k for _, k in good_cycles)
    assert_agrees_or_forks_at_a_near_tie(ordered_ids(picks), oracle, rows, rows_ref)


# ------------------------------------------------ the boundary is exact --


def test_serial_step_is_bitwise_the_engines_own_step(toy):
    """While a re-arm is pending the engine's serial step goes through
    Speculator.serial_step, which reads the same forward through
    forward_with_hidden(last_row_only=True). Same cache, same position, same
    token: the logits row must be BITWISE the `self.model(...).logits[0, -1]`
    the engine's own line produces — or the serial stretch would be a third
    numerical arm."""
    from transformers import StaticCache

    model, _, head, cfg = toy
    ids = torch.tensor([[5, 7, 11, 2, 9, 13]])
    T = ids.shape[1]
    with torch.inference_mode():
        cache_a = StaticCache(config=cfg, max_cache_len=32)
        cache_b = StaticCache(config=cfg, max_cache_len=32)
        for c in (cache_a, cache_b):
            model(ids, past_key_values=c, use_cache=True, cache_position=torch.arange(T))
        step_in = torch.tensor([[17]])
        step_pos = torch.tensor([T])
        own = model(step_in, past_key_values=cache_a, use_cache=True,
                    cache_position=step_pos).logits[0, -1]
        spec = mtp.Speculator(head, model, 4, rearm=8)
        spec.cache, spec.written, spec.serial_left, spec.drafting = cache_b, T, 8, False
        via = spec.serial_step(step_in, step_pos)
    assert torch.equal(own, via)
    assert spec.written == T + 1 and spec.serial_left == 7
    assert spec._serial_toks == [17] and spec._serial_h[0].shape == (1, 1, cfg.hidden_size)


def test_rearm_rebuilds_the_heads_kv_from_true_hiddens(toy, monkeypatch):
    """The head's KV after a re-arm: one entry per position up to W-2,
    entry p keyed on the trunk's TRUE hidden at p and t_{p+1} at rope
    position p — the invariant Speculator's docstring holds at every cycle
    boundary. Checked against a reference head fed the whole sequence in
    one pass off a fresh trunk forward: K/V agree to accumulation noise. A
    hole (the stretch's positions missing), a wrong `start` (a rope shift
    of 64 positions) or a duplicated entry all miss by orders of magnitude.
    Window pinned to 32 (the default is 64) —
    this test is about the re-arm's KV rebuild, not the trigger, and the
    trip-at-+9 timing below is that window's."""
    from transformers import StaticCache

    model, tok, head, cfg = toy
    p = PROMPTS[8]
    ref, oracle, _ = _serial_reference(toy, p)
    box = _spy_speculator(monkeypatch)
    monkeypatch.setattr(mtp, "BAIL_WINDOW", 32)
    # the head's KV as the request left it: the engine takes it off the
    # shared head at the end (Speculator.history)
    real_history = mtp.Speculator.history

    def history(self, written):
        box["kv"] = real_history(self, written)
        return box["kv"]

    monkeypatch.setattr(mtp.Speculator, "history", history)
    # stop right after the re-arm's first cycle so the KV is inspectable:
    # 9 tokens to the trip, 64 serial, then the first re-armed cycle
    with rearm_env(None), mtp_depth(4):
        eng = _engine(toy, True)
        n_prompt = _n_prompt(eng, p)
        restore_head, log = _script_head(head, n_prompt, oracle, good=lambda pos: pos >= 40)
        try:
            res, _ = _run(eng, _msgs(p), _greedy(max_tokens=78))
        finally:
            restore_head()
    spec = box["spec"]
    # 78 tokens: the last one is emitted, not written, so W = n_prompt + 77
    # and a head with no hole holds entries 0..W-2 (as a latch the head
    # would stop at the trip, n_prompt + 8 entries)
    w = n_prompt + res.completion_tokens - 1
    kv = box["kv"]
    assert kv.entries == w - 1 and kv.spans == [[0, w - 1]], (kv.entries, kv.spans, w)
    s = spec.stats()
    assert s["trips"] == 1 and s["rearms"] == 1 and spec.drafting
    assert spec.written == w
    live_k = kv.cache.layers[0].keys.clone()
    live_v = kv.cache.layers[0].values.clone()
    # the reference: the same ids, one trunk pass for the hiddens, one head
    # pass for the entries — what a head fed every position would hold
    prompt_ids = build_prompt(tok, _msgs(p), template_kwargs=eng.template_kwargs)
    gen = tok(res.text)["input_ids"]
    all_ids = list(prompt_ids) + list(gen)
    assert all_ids[:w] == eng._slots[0].ids[:w] if eng._reuse else True
    seq = torch.tensor([all_ids[:w]])
    with torch.inference_mode():
        cache = StaticCache(config=cfg, max_cache_len=256)
        hidden, _ = mtp.forward_with_hidden(model, seq, cache, torch.arange(w))
        head.reset()
        head.run(hidden[:, :w - 1], seq[:, 1:w], 0)
    ref_k, ref_v = head.cache.layers[0].keys, head.cache.layers[0].values
    assert ref_k.shape == live_k.shape == (1, cfg.num_key_value_heads, w - 1, cfg.head_dim)
    assert torch.allclose(live_k, ref_k, atol=1e-4, rtol=1e-4)
    assert torch.allclose(live_v, ref_v, atol=1e-4, rtol=1e-4)
    # and the entries the stretch rebuilt are the ones that would differ
    # under a rope shift: a 64-position offset moves K by O(1)
    with torch.inference_mode():
        head.reset()
        head.run(hidden[:, :w - 1], seq[:, 1:w], 64)
    assert not torch.allclose(head.cache.layers[0].keys, live_k, atol=1e-2)


def test_penalties_hold_across_trip_and_rearm(toy, monkeypatch):
    """Greedy with penalties, two bad regions: trip (the running
    PenaltyState is rebuilt from gen_ids), re-arm (it is dropped; the verify
    rows go back to the list path), trip again (rebuilt again). The stream
    must still agree with the serial one, modulo a near-tie — the
    equivalence test_adaptive_bail_rebuilds_the_serial_penalty_state pins
    for the one-way switch, extended to the round trip."""
    model, tok, head, _ = toy
    p = PROMPTS[6]
    params = dict(max_tokens=200, repetition_penalty=1.35, presence_penalty=0.4,
                  frequency_penalty=0.3)
    ref, oracle, rows_ref = _serial_reference(toy, p, SampleParams(temperature=0.0, **params))
    stretches = _spy_trips(monkeypatch)
    box = _spy_speculator(monkeypatch)
    with rearm_env(16), mtp_depth(4):
        eng = _engine(toy, True)
        restore_head, _ = _script_head(head, _n_prompt(eng, p), oracle,
                                       good=lambda pos: 30 <= pos < 120)
        picks, rows, restore = capture(engines)
        try:
            res, _ = _run(eng, _msgs(p), SampleParams(temperature=0.0, **params))
        finally:
            restore()
            restore_head()
    s = box["spec"].stats()
    assert s["accepted"] > 0  # drafting resumed (as a latch: 0)
    assert s["trips"] >= 2 and s["rearms"] >= 1
    assert stretches[0] == 16
    assert res.completion_tokens == ref.completion_tokens == 200
    assert_agrees_or_forks_at_a_near_tie(ordered_ids(picks), oracle, rows, rows_ref)


def test_chained_lookup_keeps_its_context_across_the_stretch(toy, monkeypatch, capsys):
    """ngram+mtp: the lookup's invariant is len(lookup) == written at every
    cycle boundary. The serial stretch appends nothing to it, so `rearm`
    must hand it the stretch's tokens — checked at the top of every cycle
    after the re-arm."""
    model, tok, head, _ = toy
    p = PROMPTS[3]
    ref, oracle, rows_ref = _serial_reference(toy, p)
    box = _spy_speculator(monkeypatch)
    seen = []
    real_cycle = mtp.Speculator.cycle

    def cycle(self, *a, **kw):
        if self.lookup is not None:
            seen.append((len(self.lookup), self.written))
        return real_cycle(self, *a, **kw)

    monkeypatch.setattr(mtp.Speculator, "cycle", cycle)
    monkeypatch.setenv("DRINKME_SPEC", "ngram+mtp")
    with rearm_env(8), mtp_depth(4):
        eng = _engine(toy, True)
        restore_head, _ = _script_head(head, _n_prompt(eng, p), oracle,
                                       good=lambda pos: pos >= 30)
        picks, rows, restore = capture(engines)
        try:
            res, _ = _run(eng, _msgs(p), _greedy(max_tokens=120))
        finally:
            restore()
            restore_head()
    err = capsys.readouterr().err
    s = box["spec"].stats()
    assert "re-armed at +" in err  # (as a latch: never)
    assert s["mode"] == "ngram+mtp" and s["trips"] >= 1 and s["rearms"] >= 1
    assert seen and all(n == w for n, w in seen), seen
    assert_agrees_or_forks_at_a_near_tie(ordered_ids(picks), oracle, rows, rows_ref)


# ------------------------------------------------------ audit and metrics --


def test_audit_accumulates_across_trips_and_rearms(toy, monkeypatch, capsys):
    """A sampled request (temperature 0.7) through several trips and
    re-arms — the scripted head is wrong, then right, then wrong again, and
    at this temperature a point-mass draft of the greedy token is accepted
    with probability p[token], well above the floor in the good region and
    ~0 in the bad ones; REARM=8 keeps the ladder short. The request's audit
    must count exactly the positions speculative_sample decided — the
    serial stretches add nothing, the re-armed cycles are not dropped —
    with zero violations, and the process-wide counter must move by the
    same amount."""
    model, tok, head, _ = toy
    p = PROMPTS[1]
    ref, oracle, _ = _serial_reference(toy, p)
    box = _spy_speculator(monkeypatch)
    decided = accepted = 0
    real = mtp.speculative_sample

    def spy(p, q, drafts, generator=None, audit=None):
        nonlocal decided, accepted
        emitted, m = real(p, q, drafts, generator, audit=audit)
        k = len(drafts)
        decided += min(m + 1, k)  # positions reached: 0..m, or all k
        accepted += m
        return emitted, m

    monkeypatch.setattr(mtp, "speculative_sample", spy)
    monkeypatch.delenv("DRINKME_SPEC_AUDIT", raising=False)
    cum = mtp._AUDIT_CUMULATIVE
    before = (cum.decided, cum.accepted, cum.violations)
    with rearm_env(8), mtp_depth(4):
        eng = _engine(toy, True)
        restore_head, log = _script_head(head, _n_prompt(eng, p), oracle,
                                         good=lambda pos: 40 <= pos < 90,
                                         sampled_good="real")
        try:
            res, _ = _run(eng, _msgs(p), SampleParams(temperature=0.7, seed=7,
                                                      max_tokens=120))
        finally:
            restore_head()
    err = capsys.readouterr().err
    s = box["spec"].stats()
    assert err.count("adaptive bail") >= 2  # (as a latch: exactly 1)
    assert s["sampled"] and s["trips"] >= 2 and s["rearms"] >= 2
    assert s["accepted"] > 0  # the good region drafted and was accepted
    a = s["audit"]
    assert a["decided"] == decided > 0 and a["accepted"] == accepted
    assert a["violations"] == 0
    assert (cum.decided - before[0], cum.accepted - before[1]) == (decided, accepted)
    assert cum.violations == before[2]
    assert err.count("adaptive bail") == s["trips"] >= 2
    assert "audit ok" in err and f"re-armed {s['rearms']}×" in err
    assert res.completion_tokens == 120


def test_metrics_count_trips_and_rearms(toy, monkeypatch):
    metrics.reset()
    box = _spy_speculator(monkeypatch)
    with rearm_env(8), mtp_depth(4):
        eng = _engine(toy, True)
        _run(eng, _msgs(PROMPTS[5]), _greedy(max_tokens=64))
    s = box["spec"].stats()
    assert s["trips"] >= 2 and s["rearms"] >= 1
    assert metrics.MTP_BAIL_TRIPS.snapshot()[()] == s["trips"]
    assert metrics.MTP_REARMS.snapshot()[()] == s["rearms"]
    text = metrics.render()
    assert "drinkme_mtp_bail_trips_total" in text and "drinkme_mtp_rearms_total" in text


def test_near_greedy_sampling_agrees_across_a_rearm(toy, monkeypatch):
    """The sampled cycle's plumbing across the round trip: at a
    temperature where every row is one-hot the accept rule is the greedy
    compare, so the scripted head's good region is accepted in full after
    the re-arm, and the stream agrees with the serial greedy transcript
    modulo a near-tie (the bar tests/test_serving_speculative.py holds the
    sampled path to)."""
    model, tok, head, _ = toy
    p = PROMPTS[7]
    ref, oracle, rows_ref = _serial_reference(toy, p)
    box = _spy_speculator(monkeypatch)
    with rearm_env(16), mtp_depth(4):
        eng = _engine(toy, True)
        restore_head, log = _script_head(head, _n_prompt(eng, p), oracle,
                                         good=lambda pos: pos >= 30)
        ids_on, rows_on, restore = capture_sampled(mtp, engines)
        try:
            res, _ = _run(eng, _msgs(p), SampleParams(temperature=NEAR_GREEDY, seed=3,
                                                      max_tokens=160))
        finally:
            restore()
            restore_head()
    s = box["spec"].stats()
    assert s["accepted"] > 0  # drafting resumed (as a latch: 0)
    assert s["sampled"] and s["trips"] >= 1 and s["rearms"] >= 1
    assert s["audit"]["violations"] == 0
    assert res.completion_tokens == 160
    assert_agrees_or_forks_at_a_near_tie(ids_on, oracle, rows_on, rows_ref)
