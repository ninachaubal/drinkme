"""spec_agree.py: what speculative-decoding arms are held to in place of
token/transcript equality, which is not achievable.

The demonstration: a genuine near-tie fork — two id sequences that diverge
at one position because the two arms' own decision rows are a coin-flip
apart — is exactly the case an `assert ids_a == ids_b` would fail on, and
exactly the case that is not a bug. These tests pin that the check tells
the two apart.
"""

import torch

from spec_agree import (assert_agrees_or_forks_at_a_near_tie, first_fork,
                        ordered_ids, top2_margin_ulps)


def _row(top, second, vocab=8):
    r = torch.full((vocab,), -10.0)
    r[0], r[1] = top, second
    return r


def test_full_agreement_returns_none_and_needs_no_rows():
    ids = [1, 2, 3, 4]
    assert assert_agrees_or_forks_at_a_near_tie(ids, list(ids)) is None


def test_a_genuine_near_tie_fork_is_not_a_failure():
    """THE case the old hard-equality assertion got wrong: rows so close the
    top-2 gap is a handful of fp32 ULPs — a real near-tie, the kind AGENTS.md
    says is real. The OLD code (`assert ids_a == ids_b`) would have raised
    here; the new one must not."""
    ids_a, ids_b = [5, 5, 0], [5, 5, 1]  # fork at position 2
    tight = _row(1.0, 1.0 + 2 ** -20)  # a handful of ULPs apart at this scale
    rows_a = {2: tight}
    rows_b = {2: _row(1.0 + 2 ** -20, 1.0)}
    fork = assert_agrees_or_forks_at_a_near_tie(ids_a, ids_b, rows_a, rows_b)
    assert fork == 2


def test_a_real_divergence_still_raises():
    """The old assertion's actual job, preserved: a fork at a wide margin is
    a real bug (a wrong accept/reject decision, not accumulation noise), and
    must still fail loudly."""
    ids_a, ids_b = [5, 5, 0], [5, 5, 1]
    wide = _row(10.0, -10.0)  # nowhere near a tie
    try:
        assert_agrees_or_forks_at_a_near_tie(ids_a, ids_b, {2: wide}, {2: wide})
    except AssertionError as e:
        assert "not a near-tie" in str(e)
    else:
        raise AssertionError("expected a real divergence to raise")


def test_no_rows_available_treats_the_fork_as_unexplained_and_raises():
    """If neither arm's row was captured at the fork (e.g. a sampled-path
    position `engines.sample_next` never saw), the margin is unknown and
    this must not silently wave the fork through."""
    ids_a, ids_b = [1, 2], [1, 3]
    try:
        assert_agrees_or_forks_at_a_near_tie(ids_a, ids_b, {}, {})
    except AssertionError:
        pass
    else:
        raise AssertionError("expected an unexplained fork to raise")


def test_first_fork_and_ordered_ids():
    assert first_fork([1, 2, 3], [1, 2, 3]) is None
    assert first_fork([1, 2, 3], [1, 9, 3]) == 1
    assert ordered_ids({0: 7, 1: 8, 2: 9}) == [7, 8, 9]


def test_top2_margin_ulps_is_small_for_a_tie_and_large_for_a_blowout():
    tie = top2_margin_ulps(_row(1.0, 1.0 + 2 ** -20))
    blowout = top2_margin_ulps(_row(10.0, -10.0))
    assert tie < 100 < blowout
