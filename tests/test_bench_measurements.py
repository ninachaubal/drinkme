"""`drinkme bench` keeps its record under ./measurements/: the file is
named after the point it holds, never overwrites an earlier run, and `-o`
still wins. And bench never imports publish — auth cannot leak into the run
path — asserted in a fresh interpreter so no other test's import can mask it.
"""

import json
import os
import subprocess
import sys

from drinkme import bench


def _record(model="Qwen3-8B", device="AMD RYZEN AI MAX+ 395 w/ Radeon 8060S"):
    r = {
        "$type": "wtf.petrichor.drinkme.measurement",
        "createdAt": "2026-09-12T20:01:02.345678+00:00",
        "version": "1.0.0",
        "environment": {"deviceClass": device, "memoryBytes": 128_000_000_000, "memoryKind": "unified"},
        "metrics": [],
    }
    if model:
        r["model"] = {"name": model, "hfRepo": "Qwen/Qwen3-8B"}
    return r


def test_path_is_model_device_day_under_measurements(tmp_path):
    p = bench.measurement_path(_record(), root=str(tmp_path))
    assert p == str(tmp_path / "measurements"
                    / "qwen3-8b_amd-ryzen-ai-max-395-w-radeon-8060s_2026-09-12.json")
    assert (tmp_path / "measurements").is_dir()  # created, so the caller can open it


def test_never_overwrites_suffixes_2_then_3(tmp_path):
    first = bench.write_record(_record(), bench.measurement_path(_record(), str(tmp_path)))
    second = bench.write_record(_record(), bench.measurement_path(_record(), str(tmp_path)))
    third = bench.measurement_path(_record(), str(tmp_path))
    assert first.endswith("_2026-09-12.json")
    assert second.endswith("_2026-09-12_2.json")
    assert third.endswith("_2026-09-12_3.json")
    assert json.load(open(first))["$type"] == "wtf.petrichor.drinkme.measurement"


def test_gulp_gets_a_profile_suffix_sip_keeps_the_bare_name(tmp_path):
    # A same-day sip and gulp of one model must not read as
    # two runs of one thing — the default profile owns the bare name, any
    # other profile names itself; the collision counter still applies after.
    sip = _record(); sip["compression"] = {"profile": "sip", "bitsPerWeight": "11.3",
                                            "meanTensorBitsPerWeight": "11.3"}
    gulp = _record(); gulp["compression"] = {"profile": "gulp", "bitsPerWeight": "10.9",
                                              "meanTensorBitsPerWeight": "10.9"}
    assert bench.measurement_path(sip, str(tmp_path)).endswith("_2026-09-12.json")
    g1 = bench.write_record(gulp, bench.measurement_path(gulp, str(tmp_path)))
    assert g1.endswith("_2026-09-12_gulp.json")
    assert bench.measurement_path(gulp, str(tmp_path)).endswith("_2026-09-12_gulp_2.json")
    assert bench.measurement_path(sip, str(tmp_path)).endswith("_2026-09-12.json")  # untouched by gulp's run


def test_incomplete_and_unknown_device_have_names_too(tmp_path):
    r = _record(model=None, device=None)
    r["incomplete"] = "no-ratio-model-fits"
    p = bench.measurement_path(r, root=str(tmp_path))
    assert os.path.basename(p) == "incomplete_unknown-device_2026-09-12.json"


def test_explicit_out_path_wins(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    out = tmp_path / "mine.json"
    where = bench.write_record(_record(), str(out))
    assert where == str(out) and out.exists()
    assert not (tmp_path / "measurements").exists()


def test_default_write_lands_in_cwd_measurements(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    where = bench.write_record(_record())
    assert where.startswith(os.path.join(".", "measurements"))
    assert (tmp_path / "measurements").is_dir()
    assert json.loads((tmp_path / where).read_text()) == _record()  # exact bytes, no reshaping


def test_written_record_is_lexicon_safe(tmp_path):
    """The atproto data model has no float: a PDS refuses 1.5 and turns 124.0
    into 124 (measured against a self-hosted PDS). So the file bench writes — the
    bytes publish sends — spells every float as a decimal string and drops
    None-valued keys; ints, bools and nesting survive untouched."""
    r = _record()
    r["model"]["revision"] = None
    r["raw"] = {"tok_s": 6.92, "samples": [6.9, 7], "ok": True, "n": 3, "nested": {"p50": 0.001}}
    p = bench.write_record(r, str(tmp_path / "r.json"))
    got = json.load(open(p))
    assert "revision" not in got["model"]
    assert got["raw"] == {"tok_s": "6.92", "samples": ["6.9", 7], "ok": True, "n": 3,
                          "nested": {"p50": "0.001"}}
    assert got["environment"]["memoryBytes"] == 128_000_000_000  # the integer stays an integer


def test_bench_never_imports_publish():
    """Fresh interpreter: import the bench path (bench + cli) and check that
    no drinkme.publish module got pulled in behind it."""
    src = (
        "import sys, drinkme.bench, drinkme.cli\n"
        "leak = sorted(m for m in sys.modules if m.startswith('drinkme.publish'))\n"
        "print(leak); sys.exit(1 if leak else 0)\n"
    )
    r = subprocess.run([sys.executable, "-c", src], capture_output=True, text=True)
    assert r.returncode == 0, f"bench imported publish: {r.stdout} {r.stderr}"
