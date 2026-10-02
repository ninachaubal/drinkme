"""The adaptive bail's TRIGGER is a pair of knobs — CPU only, on the toy hybrid tests/test_serving_mtp.py builds,
with tests/test_serving_mtp_rearm.py's scripted head.

The finding: on the lit 27B the shipped trigger (32 drafted
positions, floor 15%) trips 70% of ordinary sampled 256-token answers at
6-12% over the window while decided acceptance is 62-73%, because the
window counts DRAFTED positions and eight cycles of a code region hit 12%
by ordinary variance. A sweep ran 15%/32 against 8%/32, 8%/64 and 8%/128 on
the lit 27B: 32/0.15 tripped 7/10 free-text runs, every W64 and W128
config tripped 0/40, and 64/0.08 was decided — the DEFAULT is now
(64, 0.08); what these tests pin is that the two numbers are
`DRINKME_MTP_BAIL_WINDOW` / `DRINKME_MTP_BAIL_FLOOR`, read once into
the SpecPlan like `DRINKME_MTP_REARM`, and that they do what they say:

  * env parsing: unset = the decided defaults (64, 0.08), unparsable or
    out of range = the default with one stderr line, `0` floor = never
    bail; the plan carries them to the Speculator; the n-gram-only path
    keeps ngram.NGRAM_BAIL_FLOOR = 0 whatever the env says;
  * the FLOOR decides the trip: one scripted head, right on exactly one
    cycle in eight (12.5% over a 32-position window), trips at the shipped
    floor (15%, pinned explicitly — this test is about the floor, not
    which one ships) and does not trip at floor 0.08 — same script, same
    prompt, same tokens;
  * the WINDOW moves the trip: an all-wrong head trips after one full
    window, so the `+N` in the trip line is N = window / depth, and the
    line names the window it was measured over.
"""

import os

import pytest

from drinkme.serving import engines, mtp, ngram
from drinkme.serving.engine import SampleParams
from spec_agree import assert_agrees_or_forks_at_a_near_tie, capture, ordered_ids

# the toy hybrid, its session fixtures and the run helpers — pytest
# registers imported fixtures in this module too
from test_serving_mtp import (  # noqa: F401
    PROMPTS, _engine, _greedy, _msgs, _run, _torch_chunk_on_cpu, mtp_depth, toy,
)
from test_serving_mtp_rearm import _n_prompt, _script_head, _serial_reference, rearm_env
from test_serving_speculative import _spy_speculator

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


class bail_env:
    """DRINKME_MTP_BAIL_WINDOW / DRINKME_MTP_BAIL_FLOOR for a block, restored
    after (None = unset). The engine reads them once per engine
    (mtp.spec_plan), so every test builds a fresh engine inside the block;
    the once-per-process warning latch is cleared on both sides so the
    stderr assertions are order-independent."""

    VARS = ("DRINKME_MTP_BAIL_WINDOW", "DRINKME_MTP_BAIL_FLOOR")

    def __init__(self, window=None, floor=None):
        self.vals = (window, floor)

    def __enter__(self):
        self.old = tuple(os.environ.get(v) for v in self.VARS)
        for var, val in zip(self.VARS, self.vals):
            if val is None:
                os.environ.pop(var, None)
            else:
                os.environ[var] = str(val)
        mtp._warned.clear()
        return self

    def __exit__(self, *a):
        for var, old in zip(self.VARS, self.old):
            if old is None:
                os.environ.pop(var, None)
            else:
                os.environ[var] = old
        mtp._warned.clear()


# ------------------------------------------------------------- the knobs --


def test_bail_from_env(capsys):
    """Unset = the decided defaults (64, 0.08); each knob parses on its
    own; the plan carries both; garbage, a window below 1 and a floor
    outside 0..1 are the default with ONE stderr line each (never fatal —
    depth_from_env's rule); floor 0 is the n-gram proposer's "never bail"
    and is accepted. The shipped 32/0.15 is still reachable explicitly."""
    with bail_env():
        assert mtp.bail_from_env() == (mtp.BAIL_WINDOW, mtp.BAIL_FLOOR) == (64, 0.08)
        plan = mtp.spec_plan(True, 4)
        assert (plan.bail_window, plan.bail_floor) == (64, 0.08)
    with bail_env(32, 0.15):  # 32/0.15, still reachable
        assert mtp.bail_from_env() == (32, 0.15)
        plan = mtp.spec_plan(True, 4)
        assert (plan.bail_window, plan.bail_floor) == (32, 0.15)
    with bail_env(128):  # one knob alone leaves the other at its default
        assert mtp.bail_from_env() == (128, 0.08)
    with bail_env(None, "0.5"):
        assert mtp.bail_from_env() == (64, 0.5)
    with bail_env(None, 0):  # never bail (ngram.NGRAM_BAIL_FLOOR's meaning)
        assert mtp.bail_from_env() == (64, 0.0)
    with bail_env(None, 1):
        assert mtp.bail_from_env() == (64, 1.0)
    with bail_env(1, "1e-2"):  # the edge of the window range; a float spelling
        assert mtp.bail_from_env() == (1, 0.01)
    with bail_env("", " "):  # blank = unset
        assert mtp.bail_from_env() == (64, 0.08)
    capsys.readouterr()
    # unparsable / out of range: the default, one line per knob, once
    for window, floor in (("abc", "xyz"), (0, 1.5), (-3, -0.1), ("2.5", "nan"), ("", "inf")):
        with bail_env(window, floor):
            assert mtp.bail_from_env() == (64, 0.08), (window, floor)
            assert mtp.bail_from_env() == (64, 0.08)  # the second read is quiet
            err = capsys.readouterr().err
            expect = int(window not in ("",)) + int(floor is not None)
            assert err.count("[drinkme.mtp] DRINKME_MTP_BAIL_") == expect, (window, floor, err)
            if window != "":
                assert f"DRINKME_MTP_BAIL_WINDOW={str(window)!r} is not an integer >= 1; using 64" in err
            assert f"DRINKME_MTP_BAIL_FLOOR={str(floor)!r} is not a number in 0..1; using 0.08" in err


def test_plan_carries_the_knobs_to_the_speculator_and_ngram_keeps_its_zero(toy):
    """maybe_speculate hands the plan's window and floor to the Speculator's
    AcceptanceWindow. The n-gram-only proposer never bails (a lookup miss
    is already a plain decode step): its floor stays ngram.NGRAM_BAIL_FLOOR
    = 0 whatever DRINKME_MTP_BAIL_FLOOR says, and the chained mode keeps
    the head's. The backoff-reset threshold scales with the window (two
    windows' worth of drafted positions; REARM_SURVIVE at the default)."""
    model, tok, head, _ = toy
    params = SampleParams(temperature=0.0, max_tokens=8)
    with bail_env(48, 0.4), rearm_env(None), mtp_depth(4):
        spec = mtp.maybe_speculate(head, model, params)
        assert (spec.bail.size, spec.bail.floor) == (48, 0.4)
        assert spec._survive == 2 * 48 and spec.rearm_after == 64
        plan = mtp.spec_plan(True, 4)._replace(mode="ngram+mtp", tokens=3, hi=4, lo=3)
        chained = mtp.maybe_speculate(head, model, params, plan=plan)
        assert chained.lookup is not None and chained.head is head
        assert (chained.bail.size, chained.bail.floor) == (48, 0.4)
        plan = mtp.spec_plan(False, 4)._replace(mode="ngram", tokens=3, hi=4, lo=3)
        only = mtp.maybe_speculate(None, model, params, plan=plan)
        assert only.head is None and only.lookup is not None
        assert only.bail.floor == ngram.NGRAM_BAIL_FLOOR == 0.0
        assert only.bail.size == 48  # the window is still the plan's; the floor is what disarms it
    with bail_env():
        spec = mtp.maybe_speculate(head, model, params)
        assert (spec.bail.size, spec.bail.floor) == (64, 0.08)
        assert spec._survive == mtp.REARM_SURVIVE == 128
    # built directly (a test, a bench): the module defaults, or its own
    assert mtp.Speculator(head, model, 4).bail.size == 64
    assert mtp.Speculator(head, model, 4, bail_window=16, bail_floor=0.2).bail.floor == 0.2


# ------------------------------------------------------ floor and window --


def _cycle_period_head(head, n_prompt, oracle, period=12):
    """A scripted head that is right on exactly one cycle in eight: a good
    cycle accepts all four drafts and advances the generated position by
    five, a bad one by one, so `right at pos % 12 == 0` is one good cycle
    followed by seven bad ones, repeating — 4 accepted of every 32 drafted
    positions, 12.5% over the window, forever. That sits between 15%
    (trips) and 8% (holds) — the shipped and the swept floors, both pinned
    explicitly here since this test is about the floor, not which one
    ships — which is the whole test."""
    return _script_head(head, n_prompt, oracle, good=lambda pos: pos % period == 0)


def test_floor_decides_the_trip_on_the_same_script(toy, monkeypatch, capsys):
    """One script, two floors, one pinned window (32 — the shipped/old
    window; this test is about the FLOOR, not which window ships). At
    floor 0.15 the first full window reads 4/32 = 12.5% and trips at +12
    (one good cycle of 5 tokens, seven bad ones of 1); at floor 0.08 the
    same 12.5% holds, and every window after it holds too — no trip, no
    serial token, the bail never fires. REARM=0 (the latch) on both arms
    so the 0.15 arm's one trip is the whole story: a re-arm would break
    the script's phase and trip again at 0%, which is the script, not the
    floor. (With the env ignored, the 0.08 arm would trip exactly like the
    0.15 one.)"""
    model, tok, head, _ = toy
    p = PROMPTS[2]
    ref, oracle, rows_ref = _serial_reference(toy, p)
    results = {}
    for floor in (0.15, 0.08):
        box = _spy_speculator(monkeypatch)
        with bail_env(32, floor), rearm_env(0), mtp_depth(4):
            eng = _engine(toy, True)
            restore_head, log = _cycle_period_head(head, _n_prompt(eng, p), oracle)
            picks, rows, restore = capture(engines)
            try:
                res, _ = _run(eng, _msgs(p), _greedy(max_tokens=120))
            finally:
                restore()
                restore_head()
        results[floor] = (res, box["spec"].stats(), log, capsys.readouterr().err,
                          ordered_ids(picks), rows)
    res_d, s_d, log_d, err_d, ids_d, rows_d = results[0.15]
    res_f, s_f, log_f, err_f, ids_f, rows_f = results[0.08]
    assert res_d.completion_tokens == res_f.completion_tokens == 120
    # floor 0.15: the first full window is 12.5%
    # and trips; the trip line names the rate, the window and the floor
    assert s_d["bailed"] and s_d["trips"] == 1 and s_d["bail_rate"] == pytest.approx(0.125)
    assert s_d["serial_at_end"] and s_d["cycles"] == 8 and s_d["drafted"] == 32
    assert "adaptive bail at +12: acceptance 12% over the last 32 draft positions (floor 15%)" in err_d
    good_d = [(pos, k) for pos, k in log_d if pos % 12 == 0]
    assert good_d[0] == (0, 4) and s_d["accepted"] >= 4
    # floor 0.08: the same script, the same first window (12.5%), no trip —
    # ever: 120 tokens of one-good-in-eight, all drafted, none serial
    assert not s_f["bailed"] and s_f["trips"] == 0 and s_f["rearms"] == 0
    assert s_f["bail_rate"] is None and not s_f["serial_at_end"]
    assert "adaptive bail" not in err_f and "bailed to serial" not in err_f
    good_f = [(pos, k) for pos, k in log_f if pos % 12 == 0]
    assert s_f["accepted"] == sum(k for _, k in good_f) and s_f["accepted"] >= 36
    assert s_f["drafted"] == sum(k for _, k in log_f)
    assert s_f["acceptance"] == pytest.approx(0.125, abs=0.02)
    # the whole request was cycles: 120 tokens over the cycle count alone
    assert res_f.completion_tokens / s_f["cycles"] == pytest.approx(1.5, abs=0.05)
    # and the 0.15 arm ran fewer cycles than tokens-minus-serial would
    # allow only because it went serial: it drafted strictly less
    assert s_d["drafted"] < s_f["drafted"]
    # the greedy stream on both arms is the serial one, modulo a near-tie
    # (the accept rule is unchanged; only whether to draft moved)
    assert_agrees_or_forks_at_a_near_tie(ids_d, oracle, rows_d, rows_ref)
    assert_agrees_or_forks_at_a_near_tie(ids_f, oracle, rows_f, rows_ref)


@pytest.mark.parametrize("window,expect", [(32, 8), (64, 16), (16, 4), (48, 12)])
def test_window_moves_the_trip(toy, monkeypatch, capsys, window, expect):
    """An all-wrong head at depth 4: the window fills after window/4 cycles
    of one token each, so the trip lands at +window/4 — +8 at 32, +16 at
    64 (the default), +4 at 16, +12 at 48 — and the trip line names the
    window it was measured over (never +8 and "32 draft positions" whatever
    the env says)."""
    model, tok, head, _ = toy
    p = PROMPTS[4]
    ref, oracle, rows_ref = _serial_reference(toy, p)
    box = _spy_speculator(monkeypatch)
    with bail_env(window), rearm_env(0), mtp_depth(4):
        eng = _engine(toy, True)
        assert eng._spec_plan is None  # resolved at the first generate, below
        restore_head, log = _script_head(head, _n_prompt(eng, p), oracle,
                                         good=lambda pos: False)
        picks, rows, restore = capture(engines)
        try:
            res, _ = _run(eng, _msgs(p), _greedy(max_tokens=expect + 24))
        finally:
            restore()
            restore_head()
    err = capsys.readouterr().err
    s = box["spec"].stats()
    assert s["trips"] == 1 and s["bail_rate"] == 0.0 and s["serial_at_end"]
    assert s["cycles"] == expect and s["drafted"] == window and s["accepted"] == 0
    # the engine's plan (read once, at its first generate) is what the
    # Speculator's window was built from, and what the trip line names
    assert eng._spec_plan.bail_window == box["spec"].bail.size == window
    assert (f"adaptive bail at +{expect}: acceptance 0% over the last {window} "
            f"draft positions (floor 8%)") in err
    assert err.count("adaptive bail") == 1
    assert max(pos for pos, _ in log) == expect - 1  # the head's last ask
    assert res.completion_tokens == expect + 24
    assert_agrees_or_forks_at_a_near_tie(ordered_ids(picks), oracle, rows, rows_ref)
