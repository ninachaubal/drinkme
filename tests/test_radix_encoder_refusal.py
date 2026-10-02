"""Packing REFUSES without a C++ compiler rather than silently
dropping to the (hours-for-an-8B) numpy encoder. DRINKME_RADIX_ENCODER=numpy
is the only door to the numpy encoder; decode (serving an existing pack) is
OUT OF SCOPE and keeps its native-or-numpy fallback unconditionally.

Two distinct unavailable-native cases throughout: (a) no compiler on PATH
at all, (b) a compiler found but one that fails to build the encoder — a
fake `c++` script on PATH standing in for a broken toolchain.
"""

import os
import shutil
import stat
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(__file__))
from fixtures import realistic_bf16_bits  # noqa: E402

from drinkme import check as check_mod  # noqa: E402
from drinkme.codec import radix_native  # noqa: E402
from drinkme.codec import radix_pack as rp  # noqa: E402
from drinkme.codec.pack import pack_model  # noqa: E402
from hosts import pin_linux_host


def _hide_compiler(monkeypatch):
    """Case (a): shutil.which finds nothing, and library()'s per-process
    cache is reset so the next call re-probes instead of reusing an earlier
    verdict from this same test session."""
    monkeypatch.setattr(shutil, "which", lambda *_a, **_k: None)
    monkeypatch.setattr(radix_native, "_LIB", None)
    monkeypatch.setattr(radix_native, "_DIAGNOSIS", None)


FAKE_STDERR = "fake_cxx.cpp:12:3: error: this is the injected failure"


def _fake_failing_compiler(tmp_path, monkeypatch):
    """Case (b): a `c++` on PATH that always exits 1 with a known stderr,
    ahead of any real compiler. DRINKME_HOME moves to an empty directory
    too, so library() cannot skip the (fake) compile by finding an already-
    cached .so from an earlier, real compile in this same test session."""
    bindir = tmp_path / "fakebin"
    bindir.mkdir()
    script = bindir / "c++"
    script.write_text(f"#!/bin/sh\necho '{FAKE_STDERR}' 1>&2\nexit 1\n")
    script.chmod(script.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    monkeypatch.setenv("PATH", f"{bindir}{os.pathsep}{os.environ.get('PATH', '')}")
    monkeypatch.setenv("DRINKME_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(radix_native, "_LIB", None)
    monkeypatch.setattr(radix_native, "_DIAGNOSIS", None)
    return str(script)


# ------------------------------------------------------- radix_native.diagnose --


def test_diagnose_reports_no_compiler(monkeypatch):
    _hide_compiler(monkeypatch)
    d = radix_native.diagnose()
    assert d.ok is False and d.reason == "no_compiler" and d.compiler is None


def test_diagnose_reports_a_compile_failure_with_the_stderr_tail(tmp_path, monkeypatch):
    script = _fake_failing_compiler(tmp_path, monkeypatch)
    d = radix_native.diagnose()
    assert d.ok is False and d.reason == "compile_failed"
    assert d.compiler == script
    assert FAKE_STDERR in d.stderr_tail


def test_diagnose_reports_ok_with_compiler_and_version():
    if not radix_native.available():
        pytest.skip("no C++ compiler on this machine")
    d = radix_native.diagnose()
    assert d.ok is True and d.compiler and d.version


def test_refusal_message_names_the_three_candidates_and_install_lines(monkeypatch):
    _hide_compiler(monkeypatch)
    msg = radix_native.refusal_message()
    assert "c++" in msg and "g++" in msg and "clang++" in msg
    assert "apt install g++" in msg and "dnf install gcc-c++" in msg
    assert "pacman -S gcc" in msg and "xcode-select --install" in msg
    assert "DRINKME_RADIX_ENCODER=numpy" in msg and "hours" in msg


def test_refusal_message_names_the_compiler_and_stderr_tail(tmp_path, monkeypatch):
    script = _fake_failing_compiler(tmp_path, monkeypatch)
    msg = radix_native.refusal_message()
    assert script in msg and FAKE_STDERR in msg
    assert "DRINKME_RADIX_ENCODER=numpy" in msg


# --------------------------------------------------- radix_pack._encoder --


def test_encoder_refuses_with_no_compiler(monkeypatch):
    monkeypatch.delenv("DRINKME_RADIX_ENCODER", raising=False)
    _hide_compiler(monkeypatch)
    with pytest.raises(SystemExit, match="no C\\+\\+ compiler"):
        rp._encoder()


def test_encoder_refuses_with_a_compile_failure(tmp_path, monkeypatch):
    monkeypatch.delenv("DRINKME_RADIX_ENCODER", raising=False)
    _fake_failing_compiler(tmp_path, monkeypatch)
    with pytest.raises(SystemExit, match="failed to build"):
        rp._encoder()


def test_encoder_numpy_explicit_bypasses_the_refusal(monkeypatch):
    _hide_compiler(monkeypatch)
    monkeypatch.setenv("DRINKME_RADIX_ENCODER", "numpy")
    assert rp._encoder() == "numpy"


def test_encoder_bad_value_is_still_refused_by_name(monkeypatch):
    monkeypatch.setenv("DRINKME_RADIX_ENCODER", "onnx")
    with pytest.raises(ValueError, match="expected native or numpy"):
        rp._encoder()


def test_decoder_never_refuses_it_falls_back_to_numpy_silently(monkeypatch):
    """The explicit carve-out: decode_back_radix's resolver (_decoder, not
    _encoder) keeps the OLD silent fallback — decode is out of scope."""
    monkeypatch.delenv("DRINKME_RADIX_ENCODER", raising=False)
    _hide_compiler(monkeypatch)
    assert rp._decoder() == "numpy"


def test_numpy_opt_in_prints_one_line_and_says_roughly_what_it_costs(monkeypatch, capsys):
    monkeypatch.setenv("DRINKME_RADIX_ENCODER", "numpy")
    rp.refuse_unless_encoder_available()
    err = capsys.readouterr().err
    assert err.count("\n") == 1  # one line
    assert "DRINKME_RADIX_ENCODER=numpy" in err and "hours" in err


def test_native_path_prints_nothing(monkeypatch, capsys):
    if not radix_native.available():
        pytest.skip("no C++ compiler on this machine")
    monkeypatch.delenv("DRINKME_RADIX_ENCODER", raising=False)
    rp.refuse_unless_encoder_available()
    assert capsys.readouterr().err == ""


# --------------------------------------------------------- pack_model's door --


def test_pack_model_refuses_before_touching_the_hub_snapshot_no_compiler(monkeypatch):
    _hide_compiler(monkeypatch)
    from drinkme.serving import checkpoint as ckpt

    def boom(*_a, **_k):
        raise AssertionError("snapshot_dir must not run: the front door refuses first")

    monkeypatch.setattr(ckpt, "snapshot_dir", boom)
    with pytest.raises(SystemExit, match="no C\\+\\+ compiler"):
        pack_model("toy/whatever", None, "/nonexistent/out")


def test_pack_model_refuses_before_touching_the_hub_snapshot_compile_failed(tmp_path, monkeypatch):
    _fake_failing_compiler(tmp_path, monkeypatch)
    from drinkme.serving import checkpoint as ckpt

    def boom(*_a, **_k):
        raise AssertionError("snapshot_dir must not run: the front door refuses first")

    monkeypatch.setattr(ckpt, "snapshot_dir", boom)
    with pytest.raises(SystemExit, match="failed to build"):
        pack_model("toy/whatever", None, "/nonexistent/out")


# ------------------------------------------------------------- drinkme check --


def test_check_reports_the_compiler_path_and_version_when_available():
    if not radix_native.available():
        pytest.skip("no C++ compiler on this machine")
    v = check_mod._attach_encoder_status(check_mod.Verdict("x/y", None, True, None))
    assert v.encoder_compiler and v.encoder_version and v.encoder_problem is None
    assert "encoder: native" in v.line() and v.encoder_compiler in v.line()


def test_check_reports_no_compiler_as_a_problem(monkeypatch):
    """encoder_problem carries the can't-pack-it-HERE fact; it is deliberately
    NOT in v.line() (the error-copy pass split them: the verdict
    stays on stdout, this half is cli.py's own stderr print, and stdout
    stays valid JSON in --json mode either way)."""
    _hide_compiler(monkeypatch)
    v = check_mod._attach_encoder_status(check_mod.Verdict("x/y", None, True, None))
    assert v.encoder_problem is not None and "no C++ compiler" in v.encoder_problem
    assert "drinkme check: this machine can't pack it yet" in v.encoder_problem
    assert "apt install g++" in v.encoder_problem
    assert "PROBLEM" not in v.line() and "apt install g++" not in v.line()


def test_check_reports_a_compile_failure_as_a_problem(tmp_path, monkeypatch):
    script = _fake_failing_compiler(tmp_path, monkeypatch)
    v = check_mod._attach_encoder_status(check_mod.Verdict("x/y", None, True, None))
    assert v.encoder_problem is not None
    assert script in v.encoder_problem and FAKE_STDERR in v.encoder_problem
    assert "PROBLEM" not in v.line()


def test_check_reports_nothing_when_numpy_is_explicitly_opted_in(monkeypatch):
    _hide_compiler(monkeypatch)
    monkeypatch.setenv("DRINKME_RADIX_ENCODER", "numpy")
    v = check_mod._attach_encoder_status(check_mod.Verdict("x/y", None, True, None))
    assert v.encoder_problem is None and v.encoder_compiler is None
    assert "PROBLEM" not in v.line() and "encoder:" not in v.line()


def test_check_encoder_status_is_independent_of_repo_eligibility(monkeypatch):
    """A refused repo (ok=False) still carries encoder_problem — the two
    verdicts travel together but neither is the other."""
    _hide_compiler(monkeypatch)
    v = check_mod._attach_encoder_status(
        check_mod.Verdict("x/y", None, False, "some unrelated repo condition"))
    assert v.ok is False and v.reason == "some unrelated repo condition"
    assert v.encoder_problem is not None
    assert "refused" in v.line() and "PROBLEM" not in v.line()


def test_pure_assess_verdicts_print_exactly_as_before():
    """assess() (no network, no encoder probe) leaves the encoder fields
    unset, so every pre-existing line() in tests/test_check.py is untouched."""
    v = check_mod.Verdict("x/y", None, True, None, total_gib=1.0, packable_pct=50.0,
                          est_packed_gib=0.5)
    assert v.line() == "check x/y: eligible — 50.0% of 1.07 GB packs, est. 0.54 GB"


# ------------------------------------------------------ bench's in-memory pack --


def test_bench_refuses_the_in_memory_pack_before_any_arm_runs(monkeypatch):
    pin_linux_host(monkeypatch)  # bench on the torch runtime; a Mac would resolve mlx
    _hide_compiler(monkeypatch)
    from drinkme import arms, bench
    from drinkme.detect import Hardware
    from drinkme.suggest import Model, Suggestion

    def fake_hw():
        return Hardware(device_class="test-device", memory_gb=16.0, memory_kind="vram",
                        budget_gb=16.0, cpu_info="t", gpu_info="t", evidence=["fixture"])

    def fake_suggestion(*_a, **_k):
        return Suggestion(ratio=Model(name="ToyModel", hf_repo="toy/model", revision=None),
                          ratio_extra=[], fit=None, fit_knife_edge=False,
                          fit_honest_negative=False, notes=[])

    def boom(*_a, **_k):
        raise AssertionError("run_arms must not run: the front door refuses first")

    monkeypatch.setattr(bench, "detect", fake_hw)
    monkeypatch.setattr(bench, "suggest", fake_suggestion)
    monkeypatch.setattr(arms, "run_arms", boom)
    with pytest.raises(SystemExit, match="no C\\+\\+ compiler"):
        bench.run(no_gemma=True)


def test_bench_with_pack_dir_never_needs_the_encoder(monkeypatch, tmp_path):
    """--pack-dir times an EXISTING pack (the serve loader), so the encoder
    refusal must not fire even with no compiler — check_pack_dir (a
    different, torch-free door) is reached instead."""
    pin_linux_host(monkeypatch)  # bench on the torch runtime; a Mac would resolve mlx
    _hide_compiler(monkeypatch)
    from drinkme import bench
    from drinkme.detect import Hardware
    from drinkme.suggest import Model, Suggestion

    def fake_hw():
        return Hardware(device_class="test-device", memory_gb=16.0, memory_kind="vram",
                        budget_gb=16.0, cpu_info="t", gpu_info="t", evidence=["fixture"])

    def fake_suggestion(*_a, **_k):
        return Suggestion(ratio=Model(name="ToyModel", hf_repo="toy/model", revision=None),
                          ratio_extra=[], fit=None, fit_knife_edge=False,
                          fit_honest_negative=False, notes=[])

    monkeypatch.setattr(bench, "detect", fake_hw)
    monkeypatch.setattr(bench, "suggest", fake_suggestion)
    missing_pack_dir = str(tmp_path / "no-such-pack")
    with pytest.raises(SystemExit, match="no pack at"):
        bench.run(no_gemma=True, pack_dir=missing_pack_dir)


# --------------------------------------------------------- serve's auto-pack --


def test_serve_ensure_pack_refuses_before_check_no_compiler(monkeypatch, tmp_path):
    _hide_compiler(monkeypatch)
    from drinkme import serve

    def boom(*_a, **_k):
        raise AssertionError("check() must not run: the front door refuses first")

    monkeypatch.setattr(check_mod, "check", boom)
    pack_dir = str(tmp_path / "pack")  # no meta.json here: a cache miss
    with pytest.raises(SystemExit, match="no C\\+\\+ compiler"):
        serve.ensure_pack("toy/model", None, pack_dir, auto_pack=True)


def test_serve_ensure_pack_refuses_before_check_compile_failed(tmp_path, monkeypatch):
    _fake_failing_compiler(tmp_path, monkeypatch)
    from drinkme import serve

    def boom(*_a, **_k):
        raise AssertionError("check() must not run: the front door refuses first")

    monkeypatch.setattr(check_mod, "check", boom)
    pack_dir = str(tmp_path / "pack2")
    with pytest.raises(SystemExit, match="failed to build"):
        serve.ensure_pack("toy/model", None, pack_dir, auto_pack=True)


def test_serve_ensure_pack_cache_hit_never_needs_the_encoder(monkeypatch, tmp_path):
    """A pack already on disk is served as is — check_pack_dir (torch-free,
    format + identity) is the only door, and it must not care that this
    machine has no compiler."""
    _hide_compiler(monkeypatch)
    from drinkme import serve

    pack_dir = str(tmp_path / "pack")
    os.makedirs(pack_dir)

    def fake_check_pack_dir(*_a, **_k):
        return {}

    monkeypatch.setattr(serve, "check_pack_dir", fake_check_pack_dir)
    with open(os.path.join(pack_dir, "meta.json"), "w") as f:
        f.write("{}")
    assert serve.ensure_pack("toy/model", None, pack_dir, auto_pack=True) == pack_dir


# --------------------------------------------- serving an existing pack works --


def test_serving_an_existing_pack_needs_no_compiler(monkeypatch):
    """decode_back_radix (swap's CPU dense route) is OUT OF SCOPE for the
    refusal: a tensor packed with the native encoder while a compiler was
    still around still serves correctly once it disappears."""
    if not radix_native.available():
        pytest.skip("no C++ compiler on this machine to build the pack with")
    import torch.nn.functional as F

    from drinkme.codec.swap import make_module

    g = torch.Generator().manual_seed(11)
    w = (torch.randn(1024, 1024, generator=g) * 0.02).to(torch.bfloat16)
    b = (torch.randn(1024, generator=g) * 0.02).to(torch.bfloat16)
    pack = rp.pack_weight_radix(w, "sip", encoder="native")
    assert pack["codec"] == "radix"  # the decode path under test, not the raw fallback's
    _hide_compiler(monkeypatch)
    comp = make_module(pack, b, "cpu")
    x = (torch.randn(4, 1024, generator=g) * 0.02).to(torch.bfloat16)
    assert torch.equal(comp(x), F.linear(x, w, b))


def test_native_and_numpy_decode_agree_with_no_compiler(monkeypatch):
    """decode_back_radix's silent fallback (native when available, numpy
    otherwise) produces the SAME bits either way — pinned already in
    tests/test_radix_pack.py; here specifically with the compiler hidden."""
    U = realistic_bf16_bits(64, 2048, seed=9, spread=True)
    p = rp.pack_array_radix(U, "sip", encoder="numpy")
    assert p is not None
    _hide_compiler(monkeypatch)
    back = rp.decode_back_radix(p)  # _decoder(): native unavailable -> numpy, silently
    assert np.array_equal(back, U)
