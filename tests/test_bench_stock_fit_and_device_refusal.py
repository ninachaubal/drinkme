"""Device refusal and the stock-arm fit check (dual-point bare bench,
--dry-run) at bench.py's own boundary — detect()/suggest()/run_arms are all
faked (Hardware fixtures, Suggestion fixtures, a stubbed arms.run_arms), same
discipline as test_bench_record_shape.py: this is bench's OWN orchestration
logic, not the GPU arms it calls."""

from __future__ import annotations

import os
import subprocess
import sys

import pytest
from menu_checkpoints import pin_sizes

from drinkme import arms, bench
from drinkme.detect import Hardware
from drinkme.fit import GB, GIB
from drinkme.suggest import Model, Suggestion


def _hw(**kw):
    base = dict(device_class="test-device", memory_gb=16.0, memory_kind="vram", budget_gb=16.0,
               cpu_info="test-cpu", gpu_info="test-gpu", evidence=["fixture"])
    base.update(kw)
    # the detectors always carry the exact bytes beside the rounded GB
    base.setdefault("memory_bytes", round(base["memory_gb"] * GB))
    base.setdefault("budget_bytes", round(base["budget_gb"] * GB))
    return Hardware(**base)


TOY_SHA = "deadbeef" * 5  # a 40-hex commit: what the arms resolve for a hub snapshot
# kernel_route.route_kernels's record for a dense toy tree (every measured
# torch arm carries one as raw.<arm>_routing)
TOY_ROUTING = {"deltanet_kernel": "none", "deltanet_conv": "none", "narrow_gemv": True, "narrow_count": 0,
               "stock_gemv": "linear", "raw_gemv": "stock"}


# {toy row name: (bf16_gb, comp_gb)}: what suggest.sizes answers for each
# row _model makes (menu_checkpoints.pin_sizes)
TOY_SIZES = {}

# every bench here is a torch box's: arms.run_arms is the stub, and a Mac
# would resolve the mlx runtime and run arms_mlx instead (hosts.py)
pytestmark = pytest.mark.usefixtures("linux_host")


@pytest.fixture(autouse=True)
def _toy_sizes(monkeypatch):
    TOY_SIZES.clear()
    pin_sizes(monkeypatch, TOY_SIZES)


def _model(name="ToyModel", bf16_gb=1.0, comp_gb=0.5):
    TOY_SIZES[name] = (bf16_gb, comp_gb)
    return Model(name=name, hf_repo=f"toy/{name.lower()}", revision=TOY_SHA)


def _suggestion(ratio=None, fit=None):
    return Suggestion(ratio=ratio, ratio_extra=[], fit=fit, fit_knife_edge=False,
                      fit_honest_negative=False, notes=[])


def _fake_raw(stock=True):
    raw = {
        "model": "toy/model", "revision": TOY_SHA, "gpu": "test-gpu",
        "resolved_revision": TOY_SHA,
        "prompt": bench.PROMPT, "prefill_prompt_len": 512,
        "bandwidth": {"probe_bytes": 4 * GIB, "read_bytes_s": 200_000_000_000,
                      "copy_bytes_s": 180_000_000_000, "device": "cuda",
                      "total_param_bytes_bf16": 1_000_000_000, "ceiling_tok_s_bf16": 200.0},
        "vram_compressed_bytes": 500_000_000,
        "twin_decode_tok_s": 12.0, "twin_decode_samples": [11.9, 12.0, 12.1],
        "vram_twin_bytes": 1_000_000_000,
        "compressed_decode_tok_s": 15.0, "compressed_decode_samples": [14.9, 15.0, 15.1],
        "compressed_prefill_tok_s": 460.0, "compressed_prefill_samples": [450.0, 460.0, 470.0],
        "compressed_ttft_s": 0.07, "compressed_ttft_samples": [0.068, 0.07, 0.072],
        "swapped_linears": 7, "verified_tensors": 7, "twin_outcome": "measured",
        "compression_profile": "sip", "mean_bpw": 12.03, "weighted_bpw": 12.364,
        "twin_routing": dict(TOY_ROUTING), "compressed_routing": dict(TOY_ROUTING),
    }
    if stock:
        raw.update({
            "stock_outcome": "measured", "stock_routing": dict(TOY_ROUTING),
            "vram_bf16_bytes": 1_000_000_000,
            "stock_decode_tok_s": 10.0, "stock_decode_samples": [9.9, 10.0, 10.1],
            "stock_prefill_tok_s": 500.0, "stock_prefill_samples": [490.0, 500.0, 510.0],
            "stock_ttft_s": 0.05, "stock_ttft_samples": [0.048, 0.05, 0.052],
        })
    else:
        raw["stock_outcome"] = "skipped_predicted_nonfit"
        raw["stock_error"] = "skipped by bench's stock-arm fit check (predicted OOM)"
    return raw


class _Stop(Exception):
    pass


# ------------------------------------------------------------ refusal --


def test_bench_refuses_to_run_on_fallback_device(monkeypatch):
    hw = _hw(device_class="amdgpu 0xdead", device_source="fallback", heuristic=True)
    monkeypatch.setattr(bench, "detect", lambda: hw)
    monkeypatch.setattr(bench, "gemma_gate_open",
                        lambda: (_ for _ in ()).throw(AssertionError("must not probe")))
    with pytest.raises(SystemExit):
        bench.run(no_gemma=True)


def test_allow_unknown_device_bypasses_the_refusal(monkeypatch):
    hw = _hw(device_class="amdgpu 0xdead", device_source="fallback", heuristic=True)
    monkeypatch.setattr(bench, "detect", lambda: hw)
    monkeypatch.setattr(bench, "gemma_gate_open", lambda: False)
    monkeypatch.setattr(bench, "suggest", lambda *a, **k: (_ for _ in ()).throw(_Stop()))
    with pytest.raises(_Stop):  # reached suggest() -> refusal did not fire
        bench.run(no_gemma=True, allow_unknown_device=True)


def test_detect_only_is_exempt_from_the_device_refusal(monkeypatch):
    hw = _hw(device_class="amdgpu 0xdead", device_source="fallback", heuristic=True)
    monkeypatch.setattr(bench, "detect", lambda: hw)
    r = bench.run(publishable_only_detect=True, no_gemma=True)
    assert r["environment"]["deviceClass"] == "amdgpu 0xdead"
    # the detector's notes left the environment block (raw.detector in a
    # record) but --detect-only still shows them
    assert r["detector"]["heuristic"] is True


def test_named_device_never_refuses(monkeypatch):
    hw = _hw(device_source="table")
    monkeypatch.setattr(bench, "detect", lambda: hw)
    monkeypatch.setattr(bench, "gemma_gate_open", lambda: False)
    monkeypatch.setattr(bench, "suggest", lambda *a, **k: (_ for _ in ()).throw(_Stop()))
    with pytest.raises(_Stop):
        bench.run(no_gemma=True)


# --------------------------------------------- the stock-arm fit check --


def test_stock_fit_check_true_runs_all_three_arms_and_passes_kwargs(monkeypatch):
    m = _model(bf16_gb=1.0)
    hw = _hw(device_source="table")
    monkeypatch.setattr(bench, "detect", lambda: hw)
    monkeypatch.setattr(bench, "gemma_gate_open", lambda: False)
    monkeypatch.setattr(bench, "suggest", lambda *a, **k: _suggestion(ratio=m))
    monkeypatch.setattr(bench, "_stock_fit_ok", lambda hw_, bf16_gb, *a, **k: True)

    captured = {}

    def fake_run_arms(repo, rev, prompt, **kw):
        captured.update(kw)
        return _fake_raw(stock=True)

    monkeypatch.setattr(arms, "run_arms", fake_run_arms)
    r = bench.run(no_gemma=True)
    assert isinstance(r, dict)
    assert r["stock"] == {"outcome": "measured"}
    assert captured["run_stock"] is True
    assert captured["stock_memory_kind"] == "vram"
    assert captured["stock_bf16_bytes"] == pytest.approx(1.0 * GB)
    names = {x["name"] for x in r["metrics"]}
    assert "stock_decode_tok_s" in names and "stock_weights_gb" in names


def test_stock_fit_check_false_routes_to_fit_point_shape(monkeypatch):
    """The original bug's outcome, at bench's orchestration layer: a predicted
    non-fit never crashes — it produces a fit-point record (stock.outcome
    skipped_predicted_nonfit, stock metrics OMITTED, twin+compressed still
    measured)."""
    m = _model(bf16_gb=58.25)
    hw = _hw(device_source="table", memory_kind="unified", memory_gb=124.0, budget_gb=124.0)
    monkeypatch.setattr(bench, "detect", lambda: hw)
    monkeypatch.setattr(bench, "gemma_gate_open", lambda: False)
    monkeypatch.setattr(bench, "suggest", lambda *a, **k: _suggestion(ratio=m))
    monkeypatch.setattr(bench, "_stock_fit_ok", lambda hw_, bf16_gb, *a, **k: False)

    captured = {}

    def fake_run_arms(repo, rev, prompt, **kw):
        captured.update(kw)
        return _fake_raw(stock=False)

    monkeypatch.setattr(arms, "run_arms", fake_run_arms)
    r = bench.run(no_gemma=True)
    assert r["stock"]["outcome"] == "skipped_predicted_nonfit"
    assert captured["run_stock"] is False
    assert captured["stock_memory_kind"] == "unified"
    assert captured["stock_bf16_bytes"] == pytest.approx(58.25 * GB)
    names = {x["name"] for x in r["metrics"]}
    assert not any(n.startswith("stock_") or n == "stock_weights_gb" for n in names)
    # the twin streams one bf16 copy, so it runs where stock was skipped
    assert {"compressed_decode_tok_s", "twin_decode_tok_s", "twin_weights_gb"} <= names
    assert r["raw"]["stock_error"]


def test_stock_fit_check_uses_real_arithmetic_for_rx7600xt_qwen8b(monkeypatch):
    """Not mocked this time: the real drinkme.fit arithmetic, fed the RX 7600 XT box's
    own numbers, must independently reach the same "does not fit" verdict —
    the exact case that would otherwise crash with torch.OutOfMemoryError."""
    from drinkme.suggest import MODELS, sizes

    m = next(x for x in MODELS if x.name == "Qwen3-8B")
    hw = _hw(device_source="table", memory_kind="vram", budget_gb=16.0, memory_gb=16.0)
    assert bench._stock_fit_ok(hw, sizes(m).bf16_gb) is False


def test_stock_fit_check_uses_real_arithmetic_for_strix_halo_27b(monkeypatch):
    from drinkme.suggest import MODELS, sizes

    m = next(x for x in MODELS if x.name == "Qwen3.8-27B")
    hw = _hw(device_source="table", memory_kind="unified", memory_gb=124.0, budget_gb=124.0)
    assert bench._stock_fit_ok(hw, sizes(m).bf16_gb) is True


# ------------------------------------------------- bare bench, both points --


def test_bare_bench_runs_both_points_to_separate_records(monkeypatch):
    ratio_m, fit_m = _model("RatioModel", bf16_gb=1.0), _model("FitModel", bf16_gb=2.0)
    hw = _hw(device_source="table")
    monkeypatch.setattr(bench, "detect", lambda: hw)
    monkeypatch.setattr(bench, "gemma_gate_open", lambda: False)
    monkeypatch.setattr(bench, "suggest", lambda *a, **k: _suggestion(ratio=ratio_m, fit=fit_m))
    monkeypatch.setattr(bench, "_stock_fit_ok", lambda hw_, bf16_gb, *a, **k: True)
    monkeypatch.setattr(arms, "run_arms", lambda *a, **k: _fake_raw(stock=True))

    r = bench.run(no_gemma=True)
    assert isinstance(r, list) and len(r) == 2
    assert {x["model"]["name"] for x in r} == {"RatioModel", "FitModel"}
    # each point's environment is its own dict, not shared/aliased —
    # mutating one must never leak into the other
    r[0]["environment"]["engine"] = "poked"
    assert r[1]["environment"].get("engine") != "poked"


def test_bare_bench_runs_one_point_when_only_ratio_exists(monkeypatch):
    """Backward-compatible shape: the common case (no fit point) still
    returns a single dict, not a one-element list."""
    ratio_m = _model("RatioModel", bf16_gb=1.0)
    hw = _hw(device_source="table")
    monkeypatch.setattr(bench, "detect", lambda: hw)
    monkeypatch.setattr(bench, "gemma_gate_open", lambda: False)
    monkeypatch.setattr(bench, "suggest", lambda *a, **k: _suggestion(ratio=ratio_m, fit=None))
    monkeypatch.setattr(bench, "_stock_fit_ok", lambda hw_, bf16_gb, *a, **k: True)
    monkeypatch.setattr(arms, "run_arms", lambda *a, **k: _fake_raw(stock=True))

    r = bench.run(no_gemma=True)
    assert isinstance(r, dict)
    assert r["model"]["name"] == "RatioModel"


def test_model_override_still_runs_exactly_one_point(monkeypatch):
    ratio_m, fit_m = _model("RatioModel"), _model("FitModel")
    hw = _hw(device_source="table")
    monkeypatch.setattr(bench, "detect", lambda: hw)
    monkeypatch.setattr(bench, "gemma_gate_open", lambda: False)
    monkeypatch.setattr(bench, "suggest", lambda *a, **k: _suggestion(ratio=ratio_m, fit=fit_m))
    monkeypatch.setattr(bench, "_stock_fit_ok", lambda hw_, bf16_gb, *a, **k: True)
    monkeypatch.setattr(arms, "run_arms", lambda *a, **k: _fake_raw(stock=True))
    from drinkme import suggest as suggest_mod

    monkeypatch.setattr(suggest_mod, "MODELS", [ratio_m, fit_m])
    r = bench.run(no_gemma=True, model_name="FitModel")
    assert isinstance(r, dict)
    assert r["model"]["name"] == "FitModel"


# ------------------------------------------------------------------ main_json --


def test_main_json_writes_two_records_for_two_points(tmp_path, monkeypatch):
    ratio_m, fit_m = _model("RatioModel"), _model("FitModel")
    hw = _hw(device_source="table")
    monkeypatch.setattr(bench, "detect", lambda: hw)
    monkeypatch.setattr(bench, "gemma_gate_open", lambda: False)
    monkeypatch.setattr(bench, "suggest", lambda *a, **k: _suggestion(ratio=ratio_m, fit=fit_m))
    monkeypatch.setattr(bench, "_stock_fit_ok", lambda hw_, bf16_gb, *a, **k: True)
    monkeypatch.setattr(arms, "run_arms", lambda *a, **k: _fake_raw(stock=True))
    monkeypatch.chdir(tmp_path)

    bench.main_json(None, None, no_gemma=True)
    written = sorted((tmp_path / "measurements").glob("*.json"))
    assert len(written) == 2


def test_main_json_warns_and_ignores_explicit_path_for_two_points(tmp_path, monkeypatch, capsys):
    ratio_m, fit_m = _model("RatioModel"), _model("FitModel")
    hw = _hw(device_source="table")
    monkeypatch.setattr(bench, "detect", lambda: hw)
    monkeypatch.setattr(bench, "gemma_gate_open", lambda: False)
    monkeypatch.setattr(bench, "suggest", lambda *a, **k: _suggestion(ratio=ratio_m, fit=fit_m))
    monkeypatch.setattr(bench, "_stock_fit_ok", lambda hw_, bf16_gb, *a, **k: True)
    monkeypatch.setattr(arms, "run_arms", lambda *a, **k: _fake_raw(stock=True))
    monkeypatch.chdir(tmp_path)

    bench.main_json(str(tmp_path / "mine.json"), None, no_gemma=True)
    assert not (tmp_path / "mine.json").exists()
    assert "ignored" in capsys.readouterr().out
    assert len(list((tmp_path / "measurements").glob("*.json"))) == 2


# --------------------------------------------------------------- --dry-run --


def test_dry_run_never_imports_torch():
    # the plan sizes the menu off each checkpoint's metadata: offline, the
    # Hub is never asked (a row not cached here is left unsized)
    src = (
        "import sys\n"
        "import drinkme.bench as bench\n"
        "bench.dry_run()\n"
        "leak = 'torch' in sys.modules\n"
        "print(leak)\n"
        "sys.exit(1 if leak else 0)\n"
    )
    r = subprocess.run([sys.executable, "-c", src], capture_output=True, text=True,
                       cwd=str(__import__("pathlib").Path(__file__).resolve().parents[1] / "src"),
                       env={**os.environ, "HF_HUB_OFFLINE": "1"})
    assert r.returncode == 0, f"dry_run imported torch: {r.stdout} {r.stderr}"


def test_dry_run_plan_shape_both_points(monkeypatch):
    ratio_m, fit_m = _model("RatioModel", bf16_gb=1.0), _model("FitModel", bf16_gb=2.0)
    hw = _hw(device_source="table")
    monkeypatch.setattr(bench, "detect", lambda: hw)
    monkeypatch.setattr(bench, "suggest", lambda *a, **k: _suggestion(ratio=ratio_m, fit=fit_m))
    monkeypatch.setattr(bench, "_stock_fit_ok", lambda hw_, bf16_gb, *a, **k: bf16_gb < 1.5)

    plan = bench.dry_run()
    assert plan["environment"]["deviceClass"] == "test-device"
    assert len(plan["points"]) == 2
    by_name = {p["model"]: p for p in plan["points"]}
    assert by_name["RatioModel"]["stockArmPredictedToFit"] is True
    assert by_name["RatioModel"]["arms"] == ["stock", "twin", "compressed"]
    assert by_name["FitModel"]["stockArmPredictedToFit"] is False
    assert by_name["FitModel"]["arms"] == ["twin", "compressed"]
    assert "predictedFilenameStem" in by_name["RatioModel"]


def test_dry_run_surfaces_the_device_refusal_without_raising(monkeypatch):
    m = _model()
    hw = _hw(device_class="amdgpu 0xdead", device_source="fallback", heuristic=True)
    monkeypatch.setattr(bench, "detect", lambda: hw)
    monkeypatch.setattr(bench, "suggest", lambda *a, **k: _suggestion(ratio=m))
    monkeypatch.setattr(bench, "_stock_fit_ok", lambda hw_, bf16_gb, *a, **k: True)
    plan = bench.dry_run()  # must not raise
    assert "deviceRefusal" in plan
    assert "0xdead" in plan["deviceRefusal"]


def test_dry_run_unknown_model_override(monkeypatch):
    hw = _hw(device_source="table")
    monkeypatch.setattr(bench, "detect", lambda: hw)
    from drinkme import suggest as suggest_mod

    monkeypatch.setattr(suggest_mod, "MODELS", [_model("OnlyModel")])
    plan = bench.dry_run(model_name="NoSuchModel")
    assert plan["points"] == []
    assert "NoSuchModel" in plan["error"]


def test_a_fit_point_whose_bf16_does_not_fit_at_all_is_compressed_only():
    """RX 7600 XT: the 8B fit point — the stock fit check steps aside, and
    the TWIN pass (the same 15.26 GB of bf16) would OOM the 16 GB card. The twin fits iff one bf16 copy fits; otherwise the record is
    compressed-only and carries no twin metric."""
    from drinkme.fit import GB, GIB, fits
    from drinkme.suggest import MODELS, RATIO_HEADROOM, sizes

    m = next(x for x in MODELS if x.name == "Qwen3-8B")
    assert not fits(sizes(m).bf16_gb * GB, 0.0, 16.0 * GIB, RATIO_HEADROOM)
    small = next(x for x in MODELS if x.name == "Qwen3-4B")
    assert fits(sizes(small).bf16_gb * GB, 0.0, 16.0 * GIB, RATIO_HEADROOM)


def test_bench_writes_each_point_as_it_lands():
    """A later point's crash must not lose an earlier point's record
    (on an RX 7600 XT an 8B fit point that crashes would take the measured
    4B ratio record with it if both were written at the end)."""
    from drinkme import bench

    seen = []

    def run_one(m):
        if m == "second":
            raise RuntimeError("second point crashes")
        return {"model": {"name": m}}

    try:
        bench._run_points(["first", "second"], run_one, on_point=seen.append)
    except RuntimeError:
        pass
    assert [r["model"]["name"] for r in seen] == ["first"]


# ------------------------------------------------ --pack-dir's front door --


def _pack_at_version(root, version: int, repo: str = "synthetic/toy-fixture") -> str:
    """A minimal real pack (one raw tensor, hashed) whose meta.json is
    hand-edited to declare `version` — torch-free, so the front door can be
    probed in a bare interpreter."""
    import json
    import os

    import numpy as np

    from drinkme.codec.pack import save_pack_dir
    from drinkme.codec.radix_pack import raw_dict

    d = os.path.join(str(root), f"pack-v{version}")
    save_pack_dir(d, {"a": raw_dict(np.zeros((8, 64), np.uint16))},
                  {"hfRepo": repo, "revision": None, "dtype": "bf16"})
    mp = os.path.join(d, "meta.json")
    meta = json.load(open(mp))
    meta["formatVersion"] = version
    json.dump(meta, open(mp, "w"))
    return d


def test_pack_dir_refuses_a_pack_of_another_format_before_any_arm(monkeypatch, tmp_path):
    """`drinkme bench --pack-dir <pack of another format>`
    only checked that meta.json existed; the format gate was
    build_compressed_model's, reached at pass 3 after the bandwidth probe,
    the stock arm (a full load + three timed shapes) and the twin arm — and
    the run then died with the refusal and no record. Now bench.run puts the
    pack through serve.check_pack_dir — the same torch-free front door
    `drinkme serve` uses — before _run_points, so the refusal is the
    loader's one line, prefixed `drinkme bench:`, and no arm runs."""
    from drinkme import suggest as suggest_mod

    other_format = _pack_at_version(tmp_path, 7)
    m = _model("RungA")
    m = m.__class__(**{**m.__dict__, "hf_repo": "synthetic/toy-fixture", "revision": None})
    hw = _hw(device_source="table")
    monkeypatch.setattr(bench, "detect", lambda: hw)
    monkeypatch.setattr(bench, "gemma_gate_open", lambda: False)
    monkeypatch.setattr(bench, "suggest", lambda *a, **k: _suggestion(ratio=m, fit=None))
    monkeypatch.setattr(suggest_mod, "MODELS", [m])
    ran = []
    monkeypatch.setattr(bench, "_run_point", lambda *a, **k: ran.append(1))
    with pytest.raises(SystemExit) as e:
        bench.run(model_name="RungA", no_gemma=True, runtime="torch", pack_dir=other_format)
    assert str(e.value) == (f"drinkme bench: {other_format}: unsupported pack format 7 (this drinkme "
                            "reads format 1); re-pack with `drinkme pack --model synthetic/toy-fixture --replace`")
    assert ran == []  # no arm, no bandwidth probe, no load
    # the identity half of the same door: a sound pack cut from another repo
    import numpy as np

    from drinkme.codec.pack import save_pack_dir
    from drinkme.codec.radix_pack import raw_dict

    sound = str(tmp_path / "sound")
    save_pack_dir(sound, {"a": raw_dict(np.zeros((8, 64), np.uint16))},
                  {"hfRepo": "synthetic/toy-fixture", "revision": None, "dtype": "bf16"})
    other = m.__class__(**{**m.__dict__, "hf_repo": "someone/else"})
    monkeypatch.setattr(bench, "suggest", lambda *a, **k: _suggestion(ratio=other, fit=None))
    monkeypatch.setattr(suggest_mod, "MODELS", [other])
    with pytest.raises(SystemExit, match=r"drinkme bench: refusing to load: pack .* was cut from synthetic/toy-fixture"):
        bench.run(model_name="RungA", no_gemma=True, runtime="torch", pack_dir=sound)
    assert ran == []


def test_pack_dir_front_door_is_torch_free(tmp_path):
    """The refusal fires before torch is imported: the whole point of a front
    door is that a box with no accelerator environment gets the line."""
    other_format = _pack_at_version(tmp_path, 7)
    src = (
        "import sys\n"
        "from drinkme.serve import check_pack_dir\n"
        "try:\n"
        f"    check_pack_dir('synthetic/toy-fixture', None, {other_format!r}, verb='drinkme bench')\n"
        "except SystemExit as e:\n"
        "    assert str(e).startswith('drinkme bench: ') and 'unsupported pack format 7' in str(e), str(e)\n"
        "else:\n"
        "    sys.exit('did not refuse')\n"
        "sys.exit(1 if 'torch' in sys.modules else 0)\n"
    )
    r = subprocess.run([sys.executable, "-c", src], capture_output=True, text=True,
                       cwd=str(__import__("pathlib").Path(__file__).resolve().parents[1] / "src"))
    assert r.returncode == 0, f"{r.stdout} {r.stderr}"
