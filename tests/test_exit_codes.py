"""One pinned exit-code table for every verb (docs/cli.md#exit-codes):
0 success, 1 an uncaught bug (tests/test_cli_entry.py), 2 usage, 3 refused (a
definite verdict about the model/pack/record), 4 can't-run-here (this
machine or its environment). Scattered tests elsewhere pin each refusal's
exact wording; this file is a table-driven pass over each verb's CODE, so
the mapping here and the one in docs/cli.md cannot drift apart silently.

Every test stubs the network, the encoder, the device and the fit check the
same way the tests next to the code under test already do (test_bootstrap.py,
test_radix_encoder_refusal.py, test_bench*.py) — nothing here touches a real
GPU, the Hub, or a production server.
"""

from __future__ import annotations

import json

import pytest

from drinkme import bench, check as check_mod, cli, exitcodes, publish, serve
from drinkme.bootstrap import BootstrapOutcome
from drinkme.check import CheckUnavailable, Verdict
from drinkme.codec.pack import UnsupportedCheckpoint
from hosts import pin_linux_host


def _ok_bootstrap(monkeypatch) -> None:
    monkeypatch.setattr("drinkme.bootstrap.ensure_accelerator", lambda: BootstrapOutcome("ok"))


def _verdict(ok: bool, reason: str | None = None, encoder_problem: str | None = None,
            encoder_compiler: str | None = None) -> Verdict:
    v = Verdict("x/y", None, ok, reason)
    v.encoder_problem = encoder_problem
    v.encoder_compiler = encoder_compiler
    v.encoder_version = "c++ (GCC) 13" if encoder_compiler else None
    return v


# -------------------------------------------------------------- check ----


def test_check_ok(monkeypatch):
    monkeypatch.setattr(check_mod, "check", lambda repo, rev: _verdict(True, encoder_compiler="/usr/bin/c++"))
    assert cli.main(["check", "--model", "x/y"]) == exitcodes.OK


def test_check_refused(monkeypatch):
    monkeypatch.setattr(check_mod, "check", lambda repo, rev: _verdict(False, reason="quantized checkpoint"))
    assert cli.main(["check", "--model", "x/y"]) == exitcodes.REFUSED


def test_check_usage_unknown_menu_name():
    assert cli.main(["check", "--model", "not-a-real-menu-name"]) == exitcodes.USAGE


def test_check_cant_run_here_no_network(monkeypatch):
    monkeypatch.setattr(check_mod, "check",
                        lambda repo, rev: (_ for _ in ()).throw(CheckUnavailable("offline")))
    assert cli.main(["check", "--model", "x/y"]) == exitcodes.CANT_RUN_HERE


def test_check_cant_run_here_no_compiler_but_verdict_still_prints(monkeypatch, capsys):
    """Eligible AND this machine can't currently pack it: the verdict (the
    RESULT) stays on stdout; the can't-pack-it-here half is a separate
    stderr line (the fix for the old "PROBLEM: drinkme: "
    double prefix)."""
    problem = "drinkme check: this machine can't pack it yet: no C++ compiler on PATH"
    monkeypatch.setattr(check_mod, "check",
                        lambda repo, rev: _verdict(True, encoder_problem=problem))
    assert cli.main(["check", "--model", "x/y"]) == exitcodes.CANT_RUN_HERE
    out, err = capsys.readouterr()
    assert out.startswith("check x/y: eligible")
    assert "PROBLEM" not in out
    assert problem in err


def test_check_json_stdout_is_valid_json_in_the_no_compiler_case(monkeypatch, capsys):
    problem = "drinkme check: this machine can't pack it yet: no C++ compiler on PATH"
    monkeypatch.setattr(check_mod, "check",
                        lambda repo, rev: _verdict(True, encoder_problem=problem))
    rc = cli.main(["check", "--model", "x/y", "--json"])
    assert rc == exitcodes.CANT_RUN_HERE
    out = capsys.readouterr().out
    parsed = json.loads(out)  # must not raise: stdout is JSON only, even here
    assert parsed["ok"] is True and parsed["encoder_problem"] == problem


# --------------------------------------------------------------- pack ----


def test_pack_usage_no_model():
    assert cli.main(["pack"]) == exitcodes.USAGE


def test_pack_usage_sip_and_gulp(monkeypatch):
    _ok_bootstrap(monkeypatch)
    assert cli.main(["pack", "--model", "Qwen/Qwen3-8B", "--sip", "--gulp"]) == exitcodes.USAGE


def test_pack_usage_unknown_menu_name(monkeypatch):
    _ok_bootstrap(monkeypatch)
    assert cli.main(["pack", "--model", "not-a-real-menu-name"]) == exitcodes.USAGE


def test_pack_usage_existing_pack_without_replace(monkeypatch, tmp_path):
    _ok_bootstrap(monkeypatch)
    out = tmp_path / "pack"
    out.mkdir()
    monkeypatch.setattr(
        "drinkme.codec.pack.pack_model",
        lambda *a, **k: (_ for _ in ()).throw(FileExistsError(f"pack destination {out} exists")))
    assert cli.main(["pack", "--model", "Qwen/Qwen3-8B", "-o", str(out)]) == exitcodes.USAGE


def test_pack_refused_fp8(monkeypatch):
    _ok_bootstrap(monkeypatch)
    monkeypatch.setattr(
        "drinkme.codec.pack.pack_model",
        lambda *a, **k: (_ for _ in ()).throw(
            UnsupportedCheckpoint("x/y: FP8 checkpoints are not supported in this release (bf16 only)")))
    assert cli.main(["pack", "--model", "Qwen/Qwen3-8B"]) == exitcodes.REFUSED


def test_pack_ok(monkeypatch, tmp_path):
    _ok_bootstrap(monkeypatch)
    out = tmp_path / "pack"
    monkeypatch.setattr("drinkme.codec.pack.pack_model", lambda *a, **k: str(out))
    assert cli.main(["pack", "--model", "Qwen/Qwen3-8B", "-o", str(out)]) == exitcodes.OK


# -------------------------------------------------------------- verify ----


def test_verify_usage_no_args():
    assert cli.main(["verify"]) == exitcodes.USAGE


def test_verify_usage_no_pack_at_path(tmp_path):
    assert cli.main(["verify", "--pack-dir", str(tmp_path / "nope")]) == exitcodes.USAGE


def test_verify_refused_hash_mismatch(monkeypatch, tmp_path):
    d = tmp_path / "pack"
    d.mkdir()
    (d / "meta.json").write_text("{}")
    monkeypatch.setattr("drinkme.codec.pack.verify_pack",
                        lambda pdir: (_ for _ in ()).throw(ValueError("sha256 mismatch on w.npz")))
    assert cli.main(["verify", "--pack-dir", str(d)]) == exitcodes.REFUSED


def test_verify_ok(monkeypatch, tmp_path):
    d = tmp_path / "pack"
    d.mkdir()
    (d / "meta.json").write_text("{}")
    monkeypatch.setattr("drinkme.codec.pack.verify_pack", lambda pdir: None)
    assert cli.main(["verify", "--pack-dir", str(d)]) == exitcodes.OK


# --------------------------------------------------------------- serve ----


def test_serve_usage_bad_runtime_env(monkeypatch):
    monkeypatch.setenv("DRINKME_RUNTIME", "cuda")
    with pytest.raises(exitcodes.Usage):
        serve.resolve_runtime(None)


def test_serve_cant_run_here_broken_accelerator_install():
    class _Torch:
        __version__ = "2.13.0+cu130"
        version = type("v", (), {"cuda": "13.0", "hip": None})()
        cuda = type("c", (), {"is_available": staticmethod(lambda: False)})()

    with pytest.raises(exitcodes.CantRunHere):
        serve.resolve_device(_Torch())


def test_serve_refused_pack_of_a_format_this_build_does_not_read(tmp_path):
    d = tmp_path / "pack"
    d.mkdir()
    (d / "meta.json").write_text(json.dumps({"formatVersion": 999, "hfRepo": "x/y"}))
    with pytest.raises(exitcodes.Refused):
        serve.check_pack_dir("x/y", None, str(d))


def test_serve_ok_clean_shutdown_is_zero(monkeypatch):
    """SIGINT/SIGTERM during serve_forever is a CLEAN shutdown (docs/cli.md
    #exit-codes): _serve returns exitcodes.OK, not whatever `errno` a signal
    might otherwise suggest."""
    class _Thread:
        def join(self):
            raise KeyboardInterrupt

    class _Srv:
        server_address = ("127.0.0.1", 3215)
        known_ids = {"m"}
        thread = _Thread()

        def shutdown(self):
            pass

    monkeypatch.setattr("drinkme.serving.http.start_server", lambda *a, **k: _Srv())
    engine = type("E", (), {"model_id": "m", "model_meta": staticmethod(lambda: {})})()
    assert serve._serve(engine, "127.0.0.1", 3215) == exitcodes.OK


# --------------------------------------------------------------- bench ----


def test_bench_usage_unknown_model(monkeypatch):
    pin_linux_host(monkeypatch)
    monkeypatch.setattr(bench, "gemma_gate_open", lambda: False)
    with pytest.raises(exitcodes.Usage):
        bench.run(no_gemma=True, model_name="not-a-real-menu-name")


def test_bench_cant_run_here_no_memory_budget(monkeypatch):
    from drinkme.detect import Hardware

    pin_linux_host(monkeypatch)
    monkeypatch.setattr(bench, "detect", lambda: Hardware(
        device_class=None, memory_gb=None, memory_kind=None, budget_gb=None,
        cpu_info="t", gpu_info="t"))
    with pytest.raises(exitcodes.CantRunHere):
        bench.run(no_gemma=True)


def test_bench_cant_run_here_cannot_size(monkeypatch):
    from drinkme.suggest import MODELS

    monkeypatch.setattr("drinkme.suggest.sizes", lambda *a, **k: None)
    with pytest.raises(exitcodes.CantRunHere):
        bench._sizes_or_exit(MODELS[0])


def test_bench_usage_no_pack_at_pack_dir(monkeypatch, tmp_path):
    from drinkme.detect import Hardware
    from drinkme.suggest import Model, Suggestion

    pin_linux_host(monkeypatch)
    monkeypatch.setattr(bench, "detect", lambda: Hardware(
        device_class="test", memory_gb=16.0, memory_kind="vram", budget_gb=16.0,
        cpu_info="t", gpu_info="t"))
    monkeypatch.setattr(bench, "gemma_gate_open", lambda: False)
    monkeypatch.setattr(bench, "suggest", lambda *a, **k: Suggestion(
        ratio=Model(name="Toy", hf_repo="toy/model", revision=None), ratio_extra=[],
        fit=None, fit_knife_edge=False, fit_honest_negative=False, notes=[]))
    with pytest.raises(exitcodes.Usage):
        bench.run(no_gemma=True, pack_dir=str(tmp_path / "no-such-pack"))


# ------------------------------------------------------------- publish ----


def test_publish_ok_logout_alone(monkeypatch, tmp_path):
    monkeypatch.setenv("DRINKME_CONFIG_DIR", str(tmp_path / "cfg"))
    assert publish.run(logout=True) == exitcodes.OK


def test_publish_usage_no_record_and_no_measurements_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("DRINKME_CONFIG_DIR", str(tmp_path / "cfg"))
    monkeypatch.chdir(tmp_path)
    assert publish.run(handle="a.b") == exitcodes.USAGE


def test_publish_refused_all_records_bad(monkeypatch, tmp_path):
    monkeypatch.setenv("DRINKME_CONFIG_DIR", str(tmp_path / "cfg"))
    p = tmp_path / "bad.json"
    p.write_text(json.dumps({"incomplete": "no-ratio-model-fits"}))
    assert publish.run([str(p)], handle="a.b") == exitcodes.REFUSED


def test_publish_cant_run_here_identity_unreachable(monkeypatch, tmp_path):
    from test_publish_validate import good

    monkeypatch.setenv("DRINKME_CONFIG_DIR", str(tmp_path / "cfg"))
    p = tmp_path / "ok.json"
    p.write_text(json.dumps(good()))
    monkeypatch.setattr(publish.identity, "resolve",
                        lambda **k: (_ for _ in ()).throw(publish.identity.IdentityError("dns lookup failed")))
    assert publish.run([str(p)], handle="a.b") == exitcodes.CANT_RUN_HERE


# ------------------------------------------------------------ bootstrap ----


def test_bootstrap_usage_gfx_without_dry_run_or_detect_only():
    """The --gfx misuse check runs before any host probe or lane detection,
    so this needs no mocking at all (run_bootstrap's own docstring)."""
    assert cli.main(["bootstrap", "--gfx", "gfx1201"]) == exitcodes.USAGE


def test_bootstrap_cant_run_here_no_lane_for_this_target(monkeypatch):
    """gfx1010: TheRock publishes no torch for it and the official wheel
    does not carry it either — an unsupported target, not a bad command."""
    from test_bootstrap import _cli_box

    _cli_box(monkeypatch)
    assert cli.main(["bootstrap", "--dry-run", "--gfx", "gfx1010"]) == exitcodes.CANT_RUN_HERE
