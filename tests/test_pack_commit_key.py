"""An off-menu repo's pack is keyed by the commit its default branch is on,
not by `main`: cli.pin_revision resolves the commit
(checkpoint.resolve_commit: the local cache first, the hub on a miss), the
pack is keyed `@<sha12>`, and the served revision is that sha, so a later
push to the branch is never served from this pack's directory. CPU only, no
network: the hub is monkeypatched.
"""

from __future__ import annotations

import json

import pytest

from drinkme import cli, serve
from drinkme.codec import pack as pack_mod
from drinkme.serving import checkpoint

SHA = "2367e865d009c13ac81713a2878291d33ab28177"
OLD = "1111111111111111111111111111111111111111"
REPO = "org/model"


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("DRINKME_HOME", str(tmp_path))
    return tmp_path


def _write_pack(home, dirname, revision, source_rev, repo=REPO):
    d = home / "packs" / dirname
    d.mkdir(parents=True)
    meta = {"formatVersion": 1, "hfRepo": repo, "revision": revision,
            "source": {"identityVersion": 1, "kind": "hub", "repo": repo,
                       "revision": source_rev, "digest": "0" * 64}}
    (d / "meta.json").write_text(json.dumps(meta))
    return str(d)


# ------------------------------------------------------------ resolve_commit --


def test_resolve_commit_reads_the_cached_snapshot_first(monkeypatch):
    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache",
                        lambda repo, fn: f"/cache/models--org--model/snapshots/{SHA}/{fn}")
    monkeypatch.setattr(huggingface_hub, "HfApi",
                        lambda: pytest.fail("a cache hit must not reach the hub"))
    assert checkpoint.resolve_commit(REPO) == SHA


def test_resolve_commit_asks_the_hub_on_a_cache_miss(monkeypatch):
    import huggingface_hub

    class Api:
        def model_info(self, repo):
            return type("Info", (), {"sha": SHA})()

    monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache", lambda repo, fn: None)
    monkeypatch.setattr(huggingface_hub, "HfApi", Api)
    assert checkpoint.resolve_commit(REPO) == SHA


def test_resolve_commit_is_none_offline_with_nothing_cached(monkeypatch):
    import huggingface_hub

    class Api:
        def model_info(self, repo):
            raise OSError("offline")

    monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache", lambda repo, fn: None)
    monkeypatch.setattr(huggingface_hub, "HfApi", Api)
    assert checkpoint.resolve_commit(REPO) is None


def test_resolve_commit_is_none_for_a_local_directory(tmp_path):
    assert checkpoint.resolve_commit(str(tmp_path)) is None


# ------------------------------------------------------------- pin_revision --


def test_pin_revision_keys_an_off_menu_repo_by_its_commit(monkeypatch, capsys):
    monkeypatch.setattr(checkpoint, "resolve_commit", lambda repo: SHA)
    assert cli.pin_revision(REPO, None) == (REPO, SHA)
    assert SHA[:12] in capsys.readouterr().err


def test_pin_revision_keeps_a_menu_pin_and_never_asks(monkeypatch):
    monkeypatch.setattr(checkpoint, "resolve_commit",
                        lambda repo: pytest.fail("a pinned revision needs no lookup"))
    assert cli.pin_revision("Qwen/Qwen3-8B", OLD) == ("Qwen/Qwen3-8B", OLD)


def test_pin_revision_falls_back_to_main_and_says_so(monkeypatch, capsys):
    monkeypatch.setattr(checkpoint, "resolve_commit", lambda repo: None)
    assert cli.pin_revision(REPO, None) == (REPO, None)
    assert "keying the pack by 'main'" in capsys.readouterr().err


def test_serve_of_an_off_menu_repo_reaches_run_with_the_commit(monkeypatch):
    from drinkme.bootstrap import BootstrapOutcome

    monkeypatch.setattr("drinkme.bootstrap.ensure_accelerator", lambda: BootstrapOutcome("ok"))
    monkeypatch.setattr(checkpoint, "resolve_commit", lambda repo: SHA)
    seen = {}
    monkeypatch.setattr(serve, "run", lambda repo, rev, **kw: seen.update(repo=repo, rev=rev) or 0)
    assert cli.main(["serve", "--model", REPO]) == 0
    assert seen == {"repo": REPO, "rev": SHA}


def test_the_pack_directory_is_keyed_by_the_commit(home):
    assert pack_mod.default_pack_dir(REPO, SHA).endswith(f"org--model@{SHA[:12]}")


def test_ensure_pack_serves_the_commit_keyed_pack(home, monkeypatch):
    keyed = _write_pack(home, f"org--model@{SHA[:12]}", SHA, SHA)
    monkeypatch.setattr(serve, "check_pack_dir", lambda repo, rev, p, *a, **k: {})
    assert serve.ensure_pack(REPO, SHA, None, auto_pack=False) == keyed


# ------------------------------------------------------ the reported revision --


def test_a_main_keyed_pack_reports_its_bound_commit_not_null():
    # keyed `@main` when pin_revision could not resolve the commit
    assert checkpoint.pack_revision({"revision": None, "source": {"revision": SHA}}) == SHA
    assert checkpoint.pack_revision({"revision": OLD, "source": {"revision": SHA}}) == OLD
    assert checkpoint.pack_revision({"revision": None}) is None
