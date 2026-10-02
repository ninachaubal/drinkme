"""A failed accelerator bootstrap STOPS the command. ensure_accelerator returns a BootstrapOutcome — ok / installed /
failed-by-name — and cli.main refuses on `failed` before any verb starts:
serve.run, pack_model, bench.main_json and verify_pack are never reached,
main returns 4 (exitcodes.CANT_RUN_HERE — this machine's environment, not a
usage error), and the verdict is a line on stderr beside the diagnostic
and the print-the-next-lane line (an exit code alone can be clobbered to 0
by TheRock torch's atexit once HIP initialises — cli.entry's docstring —
so the line is the record; entry() still exits nonzero through os._exit).

Before: ensure_accelerator returned None after the self-test died of
SIGSEGV at its first kernel launch, and the CLI called serve.run over that
runtime and returned its success code.

All mocked: no dependency is installed, no child interpreter is started.
"""

import os
import sys

import pytest

from drinkme import bootstrap, cli
from drinkme.bootstrap import SelfTestResult

sys.path.insert(0, os.path.dirname(__file__))
from test_bootstrap import _fresh_box, _healthy_host  # noqa: E402


def _crashing(lane, gfx=None, **k):
    return SelfTestResult(verdict="lane_cannot_launch", lane=lane, gfx=gfx, python="py",
                          signal="SIGSEGV", exit_code=-11, failed_rung="kernel-launch",
                          rungs=["rung 1/7 import: PASS — t", "rung 2/7 device: PASS — d"])


def _rung_failed(lane, gfx=None, **k):
    return SelfTestResult(verdict="rung_failed", lane=lane, gfx=gfx, python="py",
                          failed_rung="gemv", detail="mismatch")


def _guard_verbs(monkeypatch):
    """Every verb's model work, replaced by a failure: reaching any of them
    after a failed bootstrap IS the bug. Returns the list serve.run appends
    to when it is (legitimately) reached."""
    from drinkme import serve

    reached = []
    monkeypatch.setattr(serve, "run", lambda *a, **k: reached.append("serve.run") or 0)
    monkeypatch.setattr("drinkme.codec.pack.pack_model",
                        lambda *a, **k: pytest.fail("pack_model reached after a failed bootstrap"))
    monkeypatch.setattr("drinkme.codec.pack.verify_pack",
                        lambda *a, **k: pytest.fail("verify_pack reached after a failed bootstrap"))
    import drinkme.bench as bench

    monkeypatch.setattr(bench, "main_json",
                        lambda *a, **k: pytest.fail("bench.main_json reached after a failed bootstrap"))
    monkeypatch.setattr(cli, "resolve_model", lambda name, verb: ("Qwen/Qwen3-8B", None))
    monkeypatch.setattr(cli, "pin_revision", lambda repo, rev: (repo, rev))  # no hub lookup
    return reached


def _installing_box(monkeypatch, tmp_path, selftest, choice=None):
    _fresh_box(monkeypatch, tmp_path,
               choice or bootstrap.choose_lane("Linux", False, True, "gfx1201"), selftest=selftest)
    calls = []
    monkeypatch.setattr(bootstrap.subprocess, "call", lambda cmd, cwd=None: calls.append(cmd) or 0)
    return calls


@pytest.mark.parametrize("argv", [
    ["serve", "--model", "Qwen3-8B"],
    ["pack", "--model", "Qwen3-8B"],
    ["bench", "--model", "Qwen3-8B"],
    ["verify", "--pack-dir", "/nonexistent/pack"],
])
def test_a_failed_self_test_stops_every_weight_reading_verb(monkeypatch, capsys, tmp_path, argv):
    """The reproduction: install mocked successful, the self-test returning
    lane_cannot_launch / SIGSEGV — serve.run (pack, bench, verify) is NOT
    reached, main returns 4, and stderr carries the diagnostic, the next
    lane, and the verdict."""
    reached = _guard_verbs(monkeypatch)
    calls = _installing_box(monkeypatch, tmp_path, _crashing)
    rc = cli.main(argv)
    assert rc == 4
    assert reached == []
    assert len(calls) == 1  # one install, no second
    err = capsys.readouterr().err
    assert "died of SIGSEGV" in err                                   # the diagnostic
    assert "https://rocm.nightlies.amd.com/v2/gfx120X-all/" in err   # the next lane
    assert f"drinkme {argv[0]}: refused — accelerator bootstrap failed: self-test lane_cannot_launch" in err


def test_a_rung_failure_stops_the_command_too(monkeypatch, capsys, tmp_path):
    reached = _guard_verbs(monkeypatch)
    _installing_box(monkeypatch, tmp_path, _rung_failed)
    assert cli.main(["serve", "--model", "Qwen3-8B"]) == 4
    assert reached == []
    assert "self-test rung_failed" in capsys.readouterr().err


def test_a_passing_self_test_lets_the_verb_run(monkeypatch, capsys, tmp_path):
    """The contract's other side: installed-and-proven proceeds in this
    process (torch was never imported, so a fresh import finds the lane)."""
    reached = _guard_verbs(monkeypatch)
    _installing_box(monkeypatch, tmp_path, selftest=None)  # _fresh_box's passing self-test
    assert cli.main(["serve", "--model", "Qwen3-8B"]) == 0
    assert reached == ["serve.run"]
    assert "self-test passed" in capsys.readouterr().err


def test_a_cpu_lane_with_no_self_test_lets_the_verb_run(monkeypatch, tmp_path):
    reached = _guard_verbs(monkeypatch)
    _installing_box(monkeypatch, tmp_path, selftest=None,
                    choice=bootstrap.choose_lane("Linux", False, False, None))
    monkeypatch.setattr(bootstrap, "selftest", lambda *a, **k: pytest.fail("self-tested a CPU box"))
    assert cli.main(["serve", "--model", "Qwen3-8B"]) == 0
    assert reached == ["serve.run"]


def test_a_healthy_environment_and_the_opt_out_are_ok(linux_host, monkeypatch):
    monkeypatch.delenv("DRINKME_NO_AUTO_DEPS", raising=False)
    monkeypatch.delenv("DRINKME_DEVICE", raising=False)
    monkeypatch.setattr(bootstrap, "classify_torch", lambda m: ("ok", "fine"))
    out = bootstrap.ensure_accelerator()
    assert out.verdict == "ok" and out.ok and out.reason is None
    monkeypatch.setenv("DRINKME_NO_AUTO_DEPS", "1")
    assert bootstrap.ensure_accelerator().verdict == "ok"


# ---------------------------- the other failure returns, under one contract --


def test_every_other_failure_return_is_named(monkeypatch, capsys, tmp_path):
    """Every way ensure_accelerator can fail after explaining returns failed
    with a reason that names it, never None; the CLI stops on all of them
    the same way."""
    reached = _guard_verbs(monkeypatch)

    def outcome_of(setup):
        _fresh_box(monkeypatch, tmp_path, bootstrap.choose_lane("Linux", False, True, "gfx1151"))
        monkeypatch.setattr(bootstrap.subprocess, "call", lambda cmd, cwd=None: 0)
        setup()
        out = bootstrap.ensure_accelerator()
        assert out.verdict == "failed" and not out.ok, out
        capsys.readouterr()
        assert cli.main(["serve", "--model", "Qwen3-8B"]) == 4
        assert reached == []
        assert "refused — accelerator bootstrap failed: " + out.reason in capsys.readouterr().err
        return out.reason

    # a sync that failed
    r = outcome_of(lambda: monkeypatch.setattr(bootstrap.subprocess, "call", lambda cmd, cwd=None: 2))
    assert "failed (exit 2)" in r
    # no checkout to drive uv in
    r = outcome_of(lambda: monkeypatch.setattr(bootstrap, "project_root", lambda: None))
    assert "not a uv checkout" in r and "--group rocm" in r
    # uv missing
    r = outcome_of(lambda: monkeypatch.setattr(bootstrap.shutil, "which", lambda n: None))
    assert "uv is not on PATH" in r
    # the host probe refused (user outside the render group)
    host = bootstrap.HostProbe(system="Linux", ok=False, gfx="gfx1151", gfx_source="sysfs kfd",
                               amdgpu_loaded=True, kfd="no access (root:render 0660)", has_amd=True,
                               refusal="/dev/kfd no access: user u is not in the group (render)",
                               fix=["sudo usermod -aG render $USER"])
    r = outcome_of(lambda: monkeypatch.setattr(bootstrap, "probe_host", lambda **k: host))
    assert "host refused" in r and "render" in r
    # a target with no lane to install (refused), and one handed a line
    r = outcome_of(lambda: monkeypatch.setattr(
        bootstrap, "detect_lane", lambda **k: bootstrap.choose_lane("Linux", False, True, "gfx1010")))
    assert "no lane installed (refused)" in r
    r = outcome_of(lambda: monkeypatch.setattr(
        bootstrap, "detect_lane", lambda **k: bootstrap.choose_lane("Linux", False, True, "gfx941")))
    assert "no lane installed (line)" in r
    # the marker stopped a re-exec loop: the broken torch is still the one loaded
    def marker():
        import types

        monkeypatch.setenv(bootstrap._MARKER, "1")
        monkeypatch.setattr(bootstrap.importlib.util, "find_spec", lambda n: object())
        fake = types.ModuleType("torch")
        fake.__version__ = "2.13.0+cu130"
        fake.version = types.SimpleNamespace(cuda="13.0", hip=None)
        fake.cuda = types.SimpleNamespace(is_available=lambda: False, device_count=lambda: 0)
        monkeypatch.setitem(sys.modules, "torch", fake)
    r = outcome_of(marker)
    assert "marker" in r


def test_the_metal_lane_reports_the_same_contract(monkeypatch, capsys, tmp_path):
    monkeypatch.delenv("DRINKME_NO_AUTO_DEPS", raising=False)
    monkeypatch.delenv("DRINKME_DEVICE", raising=False)
    monkeypatch.setattr(bootstrap, "is_metal_box", lambda: True)
    monkeypatch.setattr(bootstrap, "project_root", lambda: tmp_path)
    monkeypatch.setattr(bootstrap.shutil, "which", lambda n: "/usr/bin/uv")
    monkeypatch.setattr(bootstrap.importlib.util, "find_spec", lambda n: None)
    monkeypatch.setattr(bootstrap.subprocess, "call", lambda cmd, cwd=None: 0)
    assert bootstrap.ensure_accelerator().verdict == "installed"
    monkeypatch.setattr(bootstrap.subprocess, "call", lambda cmd, cwd=None: 3)
    out = bootstrap.ensure_accelerator()
    assert out.verdict == "failed" and "metal failed (exit 3)" in out.reason


def test_the_console_entry_exits_nonzero_after_a_failed_self_test(monkeypatch, capsys, tmp_path):
    """entry() reaches os._exit with main's 4 — the process exit is nonzero
    even where an atexit handler would clobber a normal exit."""
    _guard_verbs(monkeypatch)
    _installing_box(monkeypatch, tmp_path, _crashing)
    monkeypatch.setattr(sys, "argv", ["drinkme", "serve", "--model", "Qwen3-8B"])
    exits = []

    class _Exited(BaseException):
        pass

    def fake_exit(code):
        exits.append(code)
        raise _Exited()

    monkeypatch.setattr(os, "_exit", fake_exit)
    with pytest.raises(_Exited):
        cli.entry()
    assert exits == [4]
    assert "refused — accelerator bootstrap failed" in capsys.readouterr().err
