"""arms_mlx.measure_bandwidth on the real mlx: what the probe leaves behind
and what it holds while it runs.

On an M4 (24 GB, MLX 0.32.2) the probe as first shipped read through an
fp32 cast (a 2x copy of the buffer every rep: 14.3 GB/s recorded where a
bf16 read measures 106) and left 8 GiB in MLX's buffer cache for every arm
after it. These pin the fix on whatever mlx is installed — mlx[cpu] on the
Linux dev box, Metal on a Mac — and skip by name without mlx. The number
itself is only a Mac's to judge: its check is `drinkme bench --runtime mlx`
on the Mac (docs/metal.md). tests/test_bench_units_and_arm_outcomes.py
pins the arithmetic and the sizing rule with mlx stubbed.
"""

from __future__ import annotations

import pytest

mx = pytest.importorskip("mlx.core")

from drinkme import arms_mlx  # noqa: E402

MIB = 1 << 20


def test_the_probe_leaves_nothing_in_mlxs_cache_and_holds_two_buffers_at_most():
    """After the probe MLX holds nothing, active or cached. While it runs
    the peak is the buffer and the copy's destination (the STREAM copy
    needs both) plus a margin; the old fp32 cast held three."""
    probe = 64 * MIB
    mx.clear_cache()
    mx.reset_peak_memory()
    base = mx.get_active_memory()
    bw = arms_mlx.measure_bandwidth(probe_bytes=probe)
    assert bw["probe_bytes"] == probe
    assert bw["read_bytes_s"] > 0 and bw["copy_bytes_s"] > 0
    assert mx.get_cache_memory() == 0
    assert mx.get_active_memory() == base
    assert mx.get_peak_memory() - base <= 2 * probe + 4 * MIB


def test_the_rate_is_bytes_over_seconds_through_the_real_mlx(monkeypatch):
    """The clock stubbed, mlx real: PROBE_REPS reads of the buffer in 0.5 s
    and PROBE_REPS copies (2x the bytes) in 2 s."""
    ticks = iter([10.0, 10.5, 20.0, 22.0])
    monkeypatch.setattr(arms_mlx.time, "perf_counter", lambda: next(ticks))
    probe = 32 * MIB
    bw = arms_mlx.measure_bandwidth(probe_bytes=probe)
    assert bw["read_bytes_s"] == arms_mlx.PROBE_REPS * probe * 2
    assert bw["copy_bytes_s"] == arms_mlx.PROBE_REPS * 2 * probe // 2
    assert mx.get_cache_memory() == 0


def test_the_default_size_follows_the_rule_on_this_machine():
    size = arms_mlx.probe_size_for_mlx()
    assert size % MIB == 0 and 256 * MIB <= size <= arms_mlx.PROBE_REQUEST
    if not mx.metal.is_available():
        assert size == arms_mlx.PROBE_REQUEST
