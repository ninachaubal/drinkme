"""packs.hardware_budget on a Mac budgets like engine_mlx's fit check: the
smaller of MLX's working set and what macOS would hand a process now
(fit.host_available_bytes, the vm_stat read both share), and the one-line
description says which bound it. vm_stat and MLX are mocked; no Mac needed."""

import platform
import subprocess
import sys
import types

import pytest

from drinkme import packs
from drinkme.detect import Hardware

GIB = 1024 ** 3
PAGE = 16384


def vm_stat(free_gib: float) -> str:
    """vm_stat's shape with `free_gib` GiB in the free pages and nothing else."""
    pages = int(free_gib * GIB / PAGE)
    return (f"Mach Virtual Memory Statistics: (page size of {PAGE} bytes)\n"
            f"Pages free:                               {pages}.\n"
            "Pages inactive:                                  0.\n"
            "Pages speculative:                               0.\n"
            "Pages purgeable:                                 0.\n")


@pytest.fixture
def mac(monkeypatch):
    """A Darwin host with a 16 GiB MLX working set; returns a setter for vm_stat's output."""
    monkeypatch.setattr(platform, "system", lambda: "Darwin")
    monkeypatch.setattr("drinkme.detect.detect", lambda: Hardware(
        device_class="apple", memory_gb=24.0, memory_kind="unified", budget_gb=18.0,
        cpu_info=None, gpu_info=None, memory_bytes=24 * 10**9, budget_bytes=18 * 10**9))
    fake = types.ModuleType("mlx.core")
    fake.metal = types.SimpleNamespace(is_available=lambda: True)
    fake.device_info = lambda: {"max_recommended_working_set_size": 16 * GIB}
    monkeypatch.setitem(sys.modules, "mlx", types.ModuleType("mlx"))
    monkeypatch.setitem(sys.modules, "mlx.core", fake)
    out = {"text": vm_stat(10)}

    def run(argv, **kw):
        assert argv == ["vm_stat"]
        if out["text"] is None:
            raise FileNotFoundError("vm_stat")
        return types.SimpleNamespace(stdout=out["text"], returncode=0)

    monkeypatch.setattr(subprocess, "run", run)
    return out


def test_a_busy_mac_is_budgeted_by_what_the_host_has_free(mac):
    mac["text"] = vm_stat(10)
    gib, source = packs.hardware_budget()
    assert gib == pytest.approx(10.0)
    assert source == ("host available 10.0 GiB, the smaller of mlx working set 16.0 GiB "
                      "and host available 10.0 GiB")


def test_an_idle_mac_is_budgeted_by_the_working_set(mac):
    mac["text"] = vm_stat(20)
    gib, source = packs.hardware_budget()
    assert gib == pytest.approx(16.0)
    assert source == ("mlx working set 16.0 GiB, the smaller of mlx working set 16.0 GiB "
                      "and host available 20.0 GiB")


def test_an_unreadable_vm_stat_leaves_the_working_set_and_says_so(mac):
    for bad in (None, "nothing like vm_stat"):
        mac["text"] = bad
        gib, source = packs.hardware_budget()
        assert gib == pytest.approx(16.0)
        assert source == "mlx working set 16.0 GiB (host available unread)"


def test_the_picker_and_the_fit_check_share_one_reader():
    pytest.importorskip("mlx.nn")  # the DRINKME_TEST_HOST stand-in has mlx.core only
    from drinkme import fit
    from drinkme.serving import engine_mlx

    assert engine_mlx.parse_vm_stat is fit.parse_vm_stat
    assert engine_mlx.host_available_bytes is fit.host_available_bytes


# ---- bench's menu on a Mac: the same budget, and no fit point on mlx ----


def _bench_hw(**over):
    kw = dict(device_class="apple", memory_gb=25.77, memory_kind="unified", budget_gb=19.33,
              cpu_info=None, gpu_info=None, memory_bytes=25_770_000_000, budget_bytes=19_330_000_000)
    kw.update(over)
    return Hardware(**kw)


def test_bench_menu_on_mlx_budgets_from_the_serve_pickers_number_and_offers_no_fit_point(mac):
    from drinkme import bench

    mac["text"] = vm_stat(10)  # host available 10 GiB binds under the 16 GiB working set
    seen = {}
    real = bench.suggest

    def spy(budget_gb, *a, **k):
        seen["budget_gb"] = budget_gb
        return real(budget_gb, *a, **k)

    import pytest as _p
    with _p.MonkeyPatch.context() as mp:
        mp.setattr(bench, "suggest", spy)
        s = bench._menu_suggestion(_bench_hw(), "mlx", gemma_ok=False)
    assert seen["budget_gb"] == pytest.approx(10 * GIB / 1e9)  # not sysctl x 0.75 = 19.33
    assert s.fit is None and not s.fit_knife_edge and not s.fit_honest_negative
    assert s.ratio is not None


def test_bench_menu_on_torch_keeps_detects_budget_and_its_fit_point(monkeypatch):
    from drinkme import bench

    seen = {}
    real = bench.suggest

    def spy(budget_gb, *a, **k):
        seen["budget_gb"] = budget_gb
        return real(budget_gb, *a, **k)

    monkeypatch.setattr(bench, "suggest", spy)
    hw = _bench_hw(memory_kind="vram", memory_gb=16.0, budget_gb=15.0)
    s = bench._menu_suggestion(hw, "torch", gemma_ok=False)
    assert seen["budget_gb"] == 15.0
    assert s.ratio is not None
