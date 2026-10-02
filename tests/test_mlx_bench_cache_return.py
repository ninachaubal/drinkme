"""The MLX bench hands MLX's buffer cache back the way the engine does:
mx.synchronize() BEFORE mx.clear_cache(). On Metal, freed buffers are still
owned by in-flight command buffers, and the completion handler returns them
to MLX's cache after a bare clear_cache() has run (4 GiB after the 2 GiB
probe, measured on an M4). mlx[cpu] has no completion handlers, so the real
effect only shows on a Mac (tests/test_mlx_bandwidth_probe.py, which fails
there without the fix); here a stub mx records the call order."""

from __future__ import annotations

import sys
import types

from drinkme import arms_mlx


class _Arr:
    def __add__(self, other):
        return _Arr()


def _stub_mx(calls: list):
    mx = types.ModuleType("mlx.core")
    mx.bfloat16 = "bf16"
    mx.ones = lambda shape, dtype=None: _Arr()
    mx.array = lambda v, dtype=None: _Arr()
    mx.eval = lambda *a: calls.append("eval")
    mx.sum = lambda a: _Arr()
    mx.synchronize = lambda: calls.append("synchronize")
    mx.clear_cache = lambda: calls.append("clear_cache")
    return mx


def test_the_probe_synchronizes_before_it_clears_the_cache(monkeypatch):
    calls: list = []
    monkeypatch.setitem(sys.modules, "mlx", types.ModuleType("mlx"))
    monkeypatch.setitem(sys.modules, "mlx.core", _stub_mx(calls))
    monkeypatch.setattr(arms_mlx, "_device", lambda: "stub")
    arms_mlx.measure_bandwidth(probe_bytes=1 << 20)
    assert calls[-2:] == ["synchronize", "clear_cache"]
    assert calls.count("clear_cache") == 1
    # nothing is evaluated after the return: the return is outside the timed loops
    assert "eval" not in calls[-2:]


# --- the settle after a hand-back -------------------------------

class _Clock:
    def __init__(self):
        self.t = 0.0

    def now(self):
        return self.t

    def sleep(self, s):
        self.t += s


def _settle(monkeypatch, reads, system="Darwin", **kw):
    import platform

    from drinkme import fit

    it = iter(reads)
    monkeypatch.setattr(platform, "system", lambda: system)
    monkeypatch.setattr(fit, "host_available_bytes", lambda: next(it))
    c = _Clock()
    arms_mlx._settle_host(clock=c.now, sleep=c.sleep, **kw)
    return c.t


GIB = 2**30


def test_settle_waits_for_macos_to_see_the_released_pages(monkeypatch):
    # 9.27 GiB at once, 13.01 after the lag, then steady: waits past the jump, then returns
    t = _settle(monkeypatch, [int(9.27 * GIB), int(13.01 * GIB), int(13.01 * GIB)])
    assert 0.75 <= t < 1.0


def test_settle_returns_after_the_minimum_when_nothing_moves(monkeypatch):
    assert _settle(monkeypatch, [13 * GIB, 13 * GIB]) == 0.75


def test_settle_never_waits_past_its_cap(monkeypatch):
    reads = [i * GIB for i in range(100)]  # never agrees
    assert _settle(monkeypatch, reads) <= arms_mlx.SETTLE_MAX_S + 0.1


def test_settle_is_a_no_op_off_macos_or_without_vm_stat(monkeypatch):
    assert _settle(monkeypatch, [], system="Linux") == 0.0
    assert _settle(monkeypatch, [None]) == 0.0
