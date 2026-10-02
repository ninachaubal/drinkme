"""Metal records are ordinary records (docs/metal.md): `bench` on Darwin/arm64
prints no preview notice, the summary under a Metal record is the ordinary
one, and `publish` sends a record whose environment.platform is metal like any
other. The old three-part gate (bench notice, publish refusal, aggregator F1)
was lifted together on 2026-09-29. The platform is mocked; no Mac, no GPU."""

import json

import pytest

from drinkme import bench, publish
from drinkme.cli import main
from test_publish_validate import good


@pytest.fixture
def mac(monkeypatch):
    monkeypatch.setattr(bench.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(bench.platform, "machine", lambda: "arm64")


def test_apple_silicon_is_darwin_on_arm64(monkeypatch):
    for system, machine, want in (("Darwin", "arm64", True), ("Darwin", "aarch64", True),
                                  ("Darwin", "x86_64", False), ("Linux", "aarch64", False),
                                  ("Linux", "x86_64", False)):
        monkeypatch.setattr(bench.platform, "system", lambda s=system: s)
        monkeypatch.setattr(bench.platform, "machine", lambda m=machine: m)
        assert bench.apple_silicon() is want, (system, machine)


def test_bench_on_a_mac_prints_no_preview_line_and_runs(mac, monkeypatch, capsys):
    from test_bench_record_shape import _build_record_mlx

    r = _build_record_mlx(monkeypatch)
    out = capsys.readouterr().out
    assert "aren't published" not in out and "preview" not in out
    assert r["environment"]["platform"] == "metal" and r["metrics"]  # the run went on


def test_bench_elsewhere_prints_no_preview_line(monkeypatch, capsys):
    from test_bench_record_shape import _build_record

    monkeypatch.setattr(bench.platform, "system", lambda: "Linux")
    _build_record(monkeypatch)
    assert "Metal results" not in capsys.readouterr().out


def test_a_metal_records_summary_is_the_ordinary_one():
    ordinary = ["   that file is your record: `drinkme publish` sends exactly those bytes"]
    assert bench.record_summary({"environment": {"platform": "metal"}, "raw": {}}) == ordinary
    assert bench.record_summary({"environment": {"platform": "rocm"}, "raw": {}}) == ordinary
    assert bench.record_summary({"incomplete": "x"}) == []


def test_publish_check_record_passes_a_metal_record():
    r = good()
    r["environment"]["platform"] = "metal"
    publish.check_record(r, "m.json")  # no RecordRefused: same validation as any platform


def test_publish_gets_a_metal_record_past_the_gate(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("DRINKME_CONFIG_DIR", str(tmp_path / "cfg"))
    reached = []
    monkeypatch.setattr(publish.identity, "resolve",
                        lambda **k: reached.append(1) or (_ for _ in ()).throw(RuntimeError("stop")))
    r = good()
    r["environment"]["platform"] = "metal"
    p = tmp_path / "m.json"
    p.write_text(json.dumps(r))
    try:
        main(["publish", str(p), "--handle", "alice.example"])
    except Exception:
        pass
    assert reached  # got as far as identity resolution: nothing refused it first
    assert "aren't published" not in capsys.readouterr().err
