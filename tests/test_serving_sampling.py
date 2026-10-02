"""Sampling math on hand-built logits — every rule checkable by eye — and
StopScanner's split-across-deltas guarantee. CPU torch only."""

import torch

from drinkme.serving.engine import SampleParams
from drinkme.serving.sampling import StopScanner, sample_next


def draws(logits, params, seed=7, n=50):
    g = torch.Generator().manual_seed(seed)
    return [sample_next(logits, params, generator=g) for _ in range(n)]


# ------------------------------------------------------------------ sampling --


def test_temperature_zero_is_greedy():
    logits = torch.tensor([0.1, 5.0, 0.2])
    assert sample_next(logits, SampleParams(temperature=0.0)) == 1


def test_seeded_generator_is_deterministic():
    logits = torch.linspace(0, 1, 50)
    p = SampleParams(temperature=1.5)
    a, b = draws(logits, p, seed=7), draws(logits, p, seed=7)
    assert a == b  # same seed -> same sequence, call for call
    assert len(set(a)) > 1  # and it actually samples, not argmaxes


def test_top_k_1_is_argmax_regardless_of_temperature():
    logits = torch.tensor([1.0, 3.0, 2.0])
    assert set(draws(logits, SampleParams(temperature=5.0, top_k=1))) == {1}


def test_top_k_masks_everything_below_the_kth():
    logits = torch.tensor([10.0, 9.0, 8.0, 7.0])
    assert set(draws(logits, SampleParams(temperature=3.0, top_k=2))) <= {0, 1}


def test_top_p_keeps_the_crossing_token():
    # probs exactly [0.6, 0.3, 0.1]; top_p=0.7 keeps 0 (mass before it: 0) and
    # 1 (mass before it: 0.6 < 0.7 — the crossing token stays), drops 2 (0.9).
    logits = torch.log(torch.tensor([0.6, 0.3, 0.1]))
    assert set(draws(logits, SampleParams(top_p=0.7))) <= {0, 1}


def test_top_p_always_keeps_at_least_the_top_token():
    logits = torch.tensor([10.0, 0.0, 0.0, 0.0])
    assert set(draws(logits, SampleParams(top_p=0.01))) == {0}


def test_repetition_penalty_divides_positive_logits():
    # HF convention: 2.0 / penalty 2.0 -> 1.0, so the runner-up at 1.9 wins.
    p = SampleParams(temperature=0.0, repetition_penalty=2.0)
    logits = torch.tensor([2.0, 1.9])
    assert sample_next(logits, p, prev_ids=[0]) == 1
    assert sample_next(logits, p, prev_ids=[]) == 0  # nothing seen, nothing penalized


def test_repetition_penalty_multiplies_negative_logits():
    # -1.0 must go DOWN to -2.0 (multiply), not up to -0.5 (a naive divide).
    p = SampleParams(temperature=0.0, repetition_penalty=2.0)
    assert sample_next(torch.tensor([-1.0, -1.5]), p, prev_ids=[0]) == 1


# --------------------------------------------------------------- StopScanner --


def test_presence_penalty_taxes_seen_tokens_flat():
    # token 1 leads by 0.4; a presence tax of 1.0 on it flips greedy to 0
    logits = torch.tensor([2.0, 2.4, 0.0])
    p = SampleParams(temperature=0.0, presence_penalty=1.0)
    assert sample_next(logits, p, gen_ids=[1]) == 0
    # appearing MANY times changes nothing: presence is flat, not scaled
    assert sample_next(logits, p, gen_ids=[1, 1, 1, 1]) == 0
    # a token only in the PROMPT (prev_ids) is never presence-taxed
    assert sample_next(logits, p, prev_ids=[1], gen_ids=[]) == 1


def test_frequency_penalty_scales_with_count():
    logits = torch.tensor([2.0, 3.0, 0.0])
    p = SampleParams(temperature=0.0, frequency_penalty=0.4)
    # one appearance: 3.0 - 0.4 = 2.6 still beats 2.0
    assert sample_next(logits, p, gen_ids=[1]) == 1
    # three appearances: 3.0 - 1.2 = 1.8 loses to 2.0
    assert sample_next(logits, p, gen_ids=[1, 1, 1]) == 0


def test_penalties_absent_leave_logits_untouched():
    logits = torch.tensor([1.0, 2.0, 3.0])
    assert sample_next(logits, SampleParams(temperature=0.0), gen_ids=[2, 2]) == 2


def test_stop_split_across_deltas():
    s = StopScanner(["END"])
    out = s.feed("abc E") + s.feed("N") + s.feed("D xyz")
    assert out == "abc " and s.stopped
    assert s.feed("more") == "" and s.flush() == ""  # dead after the stop


def test_stop_never_emitted_when_whole_in_one_delta():
    s = StopScanner(["STOP"])
    assert s.feed("before STOP after") == "before "
    assert s.stopped


def test_earliest_of_multiple_stops_wins():
    s = StopScanner(["YY", "XX"])
    assert s.feed("aXXbYY") == "a" and s.stopped


def test_no_stops_passes_text_through_immediately():
    s = StopScanner([])
    assert s.feed("hello ") == "hello " and s.feed("world") == "world"
    assert not s.stopped and s.flush() == ""


def test_flush_releases_the_held_tail():
    s = StopScanner(["ZZ"])
    assert s.feed("hello") == "hell"  # one char held back: "o" could be a split stop's start
    assert s.flush() == "o" and not s.stopped
