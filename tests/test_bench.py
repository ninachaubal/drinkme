"""`drinkme bench --no-gemma` (F1): the gemma gate probe sends the user's HF
token, so --no-gemma must skip the probe function entirely, not just discard
its result. --detect-only makes the same promise in its own help string
("no torch, no probe"), so it skips too. detect() is faked so these run with
no GPU and no network."""

import pytest

from drinkme import bench
from drinkme.cli import main
from drinkme.detect import Hardware


def _fake_hw():
    return Hardware(
        device_class="test-device", memory_gb=16.0, memory_kind="vram", budget_gb=16.0,
        cpu_info="test", gpu_info="test", evidence=["fixture"],
    )


def _boom():
    raise AssertionError("gemma_gate_open() was called on a no-probe path")


def test_no_gemma_flag_parses_and_skips_the_probe(monkeypatch):
    monkeypatch.setattr(bench, "detect", _fake_hw)
    monkeypatch.setattr(bench, "gemma_gate_open", _boom)
    rc = main(["bench", "--detect-only", "--no-gemma"])
    assert rc == 0


def test_detect_only_never_probes_even_without_the_flag(monkeypatch):
    """--detect-only's help string promises "no torch, no probe"; honoring it
    means the HF token never leaves the machine on this path."""
    monkeypatch.setattr(bench, "detect", _fake_hw)
    monkeypatch.setattr(bench, "gemma_gate_open", _boom)
    rc = main(["bench", "--detect-only"])
    assert rc == 0


class _Stop(Exception):
    """Sentinel: raised from suggest() to halt run() right after the gate."""


def test_full_bench_path_still_probes_without_the_flag(monkeypatch):
    """Contrast case: on the real bench path (no --detect-only, no --no-gemma)
    the probe still runs — proves the skips above are deliberate, not a dead
    wire. suggest() raises a sentinel so the heavy tail never starts."""
    monkeypatch.setattr(bench, "detect", _fake_hw)
    calls = []
    monkeypatch.setattr(bench, "gemma_gate_open", lambda: calls.append(1) or False)
    monkeypatch.setattr(bench, "suggest", lambda *a, **k: (_ for _ in ()).throw(_Stop()))
    with pytest.raises(_Stop):
        bench.run()
    assert calls == [1]


def test_full_bench_path_skips_probe_with_no_gemma(monkeypatch):
    """And with no_gemma=True the same path reaches suggest() WITHOUT ever
    touching the probe — ordering proof, not just absence."""
    monkeypatch.setattr(bench, "detect", _fake_hw)
    monkeypatch.setattr(bench, "gemma_gate_open", _boom)
    monkeypatch.setattr(bench, "suggest", lambda *a, **k: (_ for _ in ()).throw(_Stop()))
    with pytest.raises(_Stop):
        bench.run(no_gemma=True)


def test_run_hands_suggest_the_decimal_gb_budget_unconverted(monkeypatch):
    """hw.budget_gb is decimal GB (detect.py converts bytes -> GB once, at
    its own boundary), the same convention as suggest()'s menu sizes
    (suggest.Sizes). run() must hand it through as-is — no GiB
    conversion at this boundary."""
    monkeypatch.setattr(bench, "detect", _fake_hw)  # budget_gb=16.0, decimal GB
    captured = {}

    def fake_suggest(budget_gb, *a, **k):
        captured["budget_gb"] = budget_gb
        raise _Stop()

    monkeypatch.setattr(bench, "gemma_gate_open", _boom)
    monkeypatch.setattr(bench, "suggest", fake_suggest)
    with pytest.raises(_Stop):
        bench.run(no_gemma=True)
    assert captured["budget_gb"] == 16.0


def test_detect_only_never_calls_ensure_accelerator(monkeypatch):
    """cli.py must not call bootstrap.ensure_accelerator() for a `bench`
    invocation before checking --detect-only, or that flag's own "no torch"
    promise breaks. On
    a box whose accelerator is deliberately invisible to torch (a masked
    CUDA_VISIBLE_DEVICES/HIP_VISIBLE_DEVICES — e.g. a CPU-only test run,
    which is how this was found) ensure_accelerator reads that as "broken"
    and drives a real `uv sync` — the silent environment mutation
    AGENTS.md's `--no-sync` rule exists to prevent, triggered just by
    asking for the hardware read `--detect-only` promises stays free of it."""
    from drinkme import bootstrap

    monkeypatch.setattr(bench, "detect", _fake_hw)
    monkeypatch.setattr(bootstrap, "ensure_accelerator",
                        lambda **k: (_ for _ in ()).throw(
                            AssertionError("ensure_accelerator must not run")))
    assert main(["bench", "--detect-only", "--no-gemma"]) == 0


def test_dry_run_never_calls_ensure_accelerator(monkeypatch):
    from drinkme import bootstrap

    monkeypatch.setattr(bench, "detect", _fake_hw)
    monkeypatch.setattr(bootstrap, "ensure_accelerator",
                        lambda **k: (_ for _ in ()).throw(
                            AssertionError("ensure_accelerator must not run")))
    assert main(["bench", "--dry-run"]) == 0


def test_runtime_flag_parses_on_the_detect_only_path(monkeypatch):
    """--runtime mlx|torch (docs/metal.md) rides through main_json; on the
    detect-only path nothing loads, so the parse is all that is exercised."""
    monkeypatch.setattr(bench, "detect", _fake_hw)
    assert main(["bench", "--detect-only", "--no-gemma", "--runtime", "mlx"]) == 0
    with pytest.raises(SystemExit):
        main(["bench", "--detect-only", "--runtime", "cuda"])
