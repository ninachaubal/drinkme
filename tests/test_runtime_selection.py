"""Runtime-aware model selection: the runtime
is resolved FIRST and every selection door reads one runtime-capability
predicate, drinkme.runtimes.

Before: the no-model picker built the same menu for every runtime, so with
an empty pack cache a 44 or 48 GiB Mac was proposed Qwen3.8-27B — the
hybrid DeltaNet family engine_mlx refuses — and a 120 GiB one Qwen2.5-72B
(qwen2, not served by MLX either); `serve --model Qwen3.8-27B --runtime mlx`
downloaded and packed before the loader refused. These are picker
reproductions with mocked Mac budgets, not Mac hardware runs.
"""

from __future__ import annotations

import json

import pytest

from drinkme import cli, packs, serve
from drinkme.bootstrap import BootstrapOutcome


@pytest.fixture(autouse=True)
def pinned_env(tmp_path, monkeypatch):
    monkeypatch.setenv("DRINKME_DEVICE", "cpu")
    monkeypatch.setenv("DRINKME_HOME", str(tmp_path))  # an empty pack cache
    monkeypatch.delenv("DRINKME_RUNTIME", raising=False)
    monkeypatch.setattr("drinkme.bootstrap.ensure_accelerator",
                        lambda: BootstrapOutcome("ok"))
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)


def _mac(monkeypatch, gib: float):
    monkeypatch.setattr(packs, "hardware_budget", lambda: (gib, f"mlx working set {gib:.1f} GiB"))


def _capture_run(monkeypatch):
    seen = {}
    monkeypatch.setattr(serve, "run", lambda repo, rev, **kw: seen.update(repo=repo, rev=rev, kw=kw) or 0)
    return seen


def _menu_model_type(repo: str) -> str | None:
    from drinkme.suggest import MODELS

    return next(m.model_type for m in MODELS if m.hf_repo == repo)


# ------------------------------------------------------- the no-model pick --


@pytest.mark.parametrize("gib, expect", [(44.0, "Qwen/Qwen3-14B"), (48.0, "Qwen/Qwen3-14B"),
                                         (120.0, "Qwen/Qwen3-32B")])
def test_a_mac_is_offered_a_model_its_runtime_serves(monkeypatch, capsys, gib, expect):
    """Three Mac budgets on
    the mlx runtime — the pick is a family MLX serves, never the 27B or
    the 72B, and the excluded rows are named in the summary."""
    _mac(monkeypatch, gib)
    seen = _capture_run(monkeypatch)
    assert cli.main(["serve", "--runtime", "mlx"]) == 0
    assert seen["repo"] == expect  # never 'Qwen/Qwen3.8-27B' at 44/48 or 'Qwen/Qwen2.5-72B-Instruct' at 120
    from drinkme import runtimes

    assert runtimes.supported("mlx", _menu_model_type(seen["repo"]))
    assert seen["kw"]["runtime"] == "mlx"
    err = capsys.readouterr().err
    assert "runtime: mlx" in err
    assert "not offered on the mlx runtime: Qwen3.8-27B — model_type 'qwen3_5' is a hybrid" in err
    assert "not offered on the mlx runtime: Qwen2.5-72B — model_type 'qwen2' is not one the mlx runtime serves" in err


def test_the_platform_default_runtime_is_resolved_before_the_pick(monkeypatch):
    """No --runtime: DRINKME_RUNTIME (then the host) decides, and the
    pick follows the resolved runtime."""
    _mac(monkeypatch, 48.0)
    seen = _capture_run(monkeypatch)
    monkeypatch.setenv("DRINKME_RUNTIME", "mlx")
    assert cli.main(["serve"]) == 0
    assert seen["repo"] == "Qwen/Qwen3-14B"


def test_the_torch_runtime_still_picks_the_showcase(monkeypatch):
    _mac(monkeypatch, 48.0)
    seen = _capture_run(monkeypatch)
    assert cli.main(["serve", "--runtime", "torch"]) == 0
    assert seen["repo"] == "Qwen/Qwen3.8-27B"


def test_when_nothing_the_runtime_serves_fits_the_refusal_names_it(monkeypatch, capsys):
    _mac(monkeypatch, 48.0)
    monkeypatch.setattr(serve, "run", lambda *a, **k: pytest.fail("must not serve"))
    monkeypatch.setattr(packs, "build_candidates", lambda local, gemma_ok=False: [
        packs.Candidate("Qwen3.8-27B", "Qwen/Qwen3.8-27B", None, 38.0, False, None,
                        True, None, None, model_type="qwen3_5")])
    assert cli.main(["serve", "--runtime", "mlx"]) == 2
    err = capsys.readouterr().err
    assert "no candidate the mlx runtime serves (1 excluded above)" in err
    assert "pass --model to override" in err


def test_filter_for_runtime_is_the_one_predicate():
    from drinkme import runtimes

    cands = [packs.Candidate("a", "x/a", None, 1.0, False, None, True, None, None, model_type="qwen3"),
             packs.Candidate("b", "x/b", None, 1.0, False, None, True, None, None, model_type="qwen3_5"),
             packs.Candidate("c", "x/c", None, 1.0, False, None, True, None, None, model_type=None),
             packs.Candidate("d", "x/d", None, 1.0, False, None, True, None, None, model_type="llama")]
    kept, excluded = packs.filter_for_runtime(cands, "mlx")
    assert [c.name for c in kept] == ["a"]
    assert [(c.name, r == runtimes.refusal("mlx", c.model_type)) for c, r in excluded] == \
        [("b", True), ("c", True), ("d", True)]
    kept, excluded = packs.filter_for_runtime(cands, "torch")
    assert len(kept) == 4 and excluded == []


def test_a_local_pack_carries_its_config_model_type(tmp_path, monkeypatch):
    """A locally packed model off the menu: its family comes from the
    cached config (packs._local_config), so the same filter applies."""
    monkeypatch.setattr(packs, "_local_config", lambda repo, rev, pdir: {"model_type": "qwen3_5"})
    monkeypatch.setattr(packs, "_mtp_head_gib", lambda pdir: None)
    local = [packs.LocalPack("some/hybrid", None, str(tmp_path), 10 * packs.GIB, True, None,
                             10 * packs.GIB, False)]
    cands = packs.build_candidates(local)
    assert [c.model_type for c in cands if c.hf_repo == "some/hybrid"] == ["qwen3_5"]
    kept, excluded = packs.filter_for_runtime(cands, "mlx")
    assert "some/hybrid" in [c.name for c, _ in excluded]
    assert "some/hybrid" not in [c.name for c in kept]


# ------------------------------------------------- serve --model's door --


def test_an_explicit_unsupported_model_is_refused_before_any_download_or_pack(monkeypatch):
    monkeypatch.setattr(serve, "ensure_pack", lambda *a, **k: pytest.fail("packed before the door"))
    monkeypatch.setattr(serve, "build_engine", lambda *a, **k: pytest.fail("loaded before the door"))
    with pytest.raises(SystemExit) as ei:
        serve.run("Qwen/Qwen3.8-27B", "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0", runtime="mlx")
    assert str(ei.value).startswith("drinkme serve: Qwen/Qwen3.8-27B: model_type 'qwen3_5' is a hybrid DeltaNet family")
    with pytest.raises(SystemExit) as ei:
        serve.run("Qwen/Qwen2.5-72B-Instruct", None, runtime="mlx")
    assert "model_type 'qwen2' is not one the mlx runtime serves" in str(ei.value)


def test_a_supported_explicit_model_passes_the_door(monkeypatch):
    class Reached(Exception):
        pass

    monkeypatch.setattr(serve, "ensure_pack", lambda *a, **k: (_ for _ in ()).throw(Reached()))
    with pytest.raises(Reached):
        serve.run("Qwen/Qwen3-8B", None, runtime="mlx")
    with pytest.raises(Reached):  # torch refuses nothing here; the loader decides
        serve.run("Qwen/Qwen3.8-27B", None, runtime="torch")


def test_refusal_for_repo_reads_a_local_checkpoint_dir_and_falls_toward_the_working_path(tmp_path, monkeypatch):
    from drinkme import runtimes

    d = tmp_path / "ckpt"
    d.mkdir()
    (d / "config.json").write_text(json.dumps({"model_type": "qwen3_5", "architectures": ["X"]}))
    assert runtimes.refusal_for_repo("mlx", str(d), None).startswith(f"{d}: model_type 'qwen3_5'")
    (d / "config.json").write_text(json.dumps({"model_type": "qwen3"}))
    assert runtimes.refusal_for_repo("mlx", str(d), None) is None
    assert runtimes.refusal_for_repo("torch", str(d), None) is None
    # off the menu, not cached, hub unreachable: None — the loader refuses later, by name
    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache", lambda *a, **k: None)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download",
                        lambda *a, **k: (_ for _ in ()).throw(OSError("offline")))
    assert runtimes.refusal_for_repo("mlx", "nobody/unknown-model", None) is None
    # a config handed in directly wins
    assert runtimes.refusal_for_repo("mlx", "nobody/x", None, config={"model_type": "llama"}) \
        == "nobody/x: " + runtimes.refusal("mlx", "llama")


# ---------------------------------------------------- the predicate itself --


def test_the_predicate_matches_the_mlx_loaders_tables():
    from drinkme import runtimes

    assert runtimes.refusal("torch", None) is None and runtimes.refusal("torch", "anything") is None
    assert runtimes.refusal("mlx", "qwen3") is None
    assert "hybrid DeltaNet" in runtimes.refusal("mlx", "qwen3_5")
    assert "hybrid DeltaNet" in runtimes.refusal("mlx", "qwen3_next")
    assert "not one the mlx runtime serves" in runtimes.refusal("mlx", "qwen2")
    assert "family unknown" in runtimes.refusal("mlx", None)
    with pytest.raises(ValueError):
        runtimes.refusal("cuda", "qwen3")
    engine_mlx = pytest.importorskip("drinkme.serving.engine_mlx")
    assert engine_mlx.SUPPORTED_MODEL_TYPES is runtimes.MLX_SUPPORTED_MODEL_TYPES
    assert engine_mlx.HYBRID_MODEL_TYPES is runtimes.MLX_HYBRID_MODEL_TYPES
    with pytest.raises(ValueError, match="hybrid DeltaNet"):
        engine_mlx.refuse_unsupported({"model_type": "qwen3_5"})


def test_every_menu_row_names_its_family():
    from drinkme.suggest import MODELS

    assert all(m.model_type for m in MODELS), [m.name for m in MODELS if not m.model_type]


@pytest.mark.parametrize("system, machine, expect", [
    ("Darwin", "arm64", True), ("Darwin", "x86_64", False), ("Linux", "aarch64", False),
    ("Linux", "x86_64", False)])
def test_apple_silicon_is_a_mac_on_arm(monkeypatch, system, machine, expect):
    monkeypatch.setattr("platform.system", lambda: system)
    monkeypatch.setattr("platform.machine", lambda: machine)
    assert serve.apple_silicon() is expect
    monkeypatch.delenv("DRINKME_RUNTIME", raising=False)
    assert serve.resolve_runtime() == ("mlx" if expect else "torch")


def _serve_until_bootstrap(monkeypatch, argv):
    """cli.main(argv) up to the accelerator bootstrap, which is the first
    thing serve does after the notice; stopped there so nothing installs,
    resolves or loads."""
    class Stop(Exception):
        pass

    def stop(**k):
        raise Stop

    monkeypatch.setattr("drinkme.bootstrap.ensure_accelerator", stop)
    with pytest.raises(Stop):
        cli.main(argv)


@pytest.mark.parametrize("argv", [["serve", "--model", "Qwen3-0.6B"], ["serve"]])
def test_serve_on_apple_silicon_warns_before_anything_else(monkeypatch, capsys, argv):
    # Apple silicon support is a work in progress: serve says so first, ahead
    # of the bootstrap's install lines, the no-model picker and the revision
    # pin, all of which print before serve.run is reached.
    monkeypatch.setattr(serve, "apple_silicon", lambda: True)
    _serve_until_bootstrap(monkeypatch, argv)
    out, err = capsys.readouterr()
    assert out.splitlines() == [serve.APPLE_SILICON_NOTICE]
    assert err == ""


def test_serve_elsewhere_prints_no_apple_notice(monkeypatch, capsys):
    monkeypatch.setattr(serve, "apple_silicon", lambda: False)
    _serve_until_bootstrap(monkeypatch, ["serve", "--model", "Qwen3-0.6B"])
    assert serve.APPLE_SILICON_NOTICE not in capsys.readouterr().out


def test_serve_run_does_not_repeat_the_notice(monkeypatch, capsys):
    # cli.main owns the line; run() printing it too would say it twice
    monkeypatch.setattr(serve, "apple_silicon", lambda: True)

    class Stop(Exception):
        pass

    def stop(*a, **k):
        raise Stop

    monkeypatch.setattr(serve, "_advertised_ctx_from_env", stop)
    with pytest.raises(Stop):
        serve.run("Qwen/Qwen3-0.6B", None)
    assert serve.APPLE_SILICON_NOTICE not in capsys.readouterr().out


def test_no_pack_and_no_torch_names_the_bootstrap(monkeypatch):
    # every lane installs torch now, the metal lane included, so an
    # environment without it predates that or was made by hand: the one line
    # names the command that installs this machine's lane, not a one-off pip
    # install outside the lock
    import importlib.util

    real = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec",
                        lambda name, *a: None if name == "torch" else real(name, *a))
    with pytest.raises(SystemExit) as ei:
        serve.ensure_pack("Qwen/Qwen3-0.6B", "c1899de289a04d12100db370d81485cdf75e47ca",
                          None, auto_pack=True)
    msg = str(ei.value)
    assert msg.startswith("drinkme serve: no pack at ")
    assert "packing needs torch" in msg and "drinkme bootstrap" in msg
    assert "pip install" not in msg
