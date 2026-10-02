"""Published packs (src/drinkme/hub_packs.py): `drinkme serve` fetching a
pack from the pack hub, verifying it against the upstream hashes, and
falling back to packing locally (docs/serve.md#published-packs).

The pack hub is a local directory standing in for the Hub: FakeApi,
fake_download and fake_snapshot serve `<root>/<repo>/<path>` the way
huggingface_hub would, and upstream.hub_listing answers from the toy
checkpoint's own files (tests/test_upstream.py). Nothing touches the
network; the suite's conftest turns the pack hub off, and each test here
names its own.
"""

from __future__ import annotations

import json
import os
import shutil
import types

import pytest

torch = pytest.importorskip("torch")

import huggingface_hub  # noqa: E402

from drinkme import exitcodes, hub_packs, upstream  # noqa: E402
from tests.test_pack_identity import AAA  # noqa: E402
from tests.test_upstream import REPO, hub_listing_of, toy  # noqa: E402,F401

HUB = "tester/packs"          # a whole repo id: every model's packs in one repo
PACK_REPO_REV = "b" * 40
OTHER = "c" * 40


def _not_found(what: str):
    import httpx

    from huggingface_hub.errors import RepositoryNotFoundError

    return RepositoryNotFoundError(what, response=httpx.Response(
        404, request=httpx.Request("GET", "https://huggingface.co/api/models/" + what)))


class FakeHub:
    """A directory per pack repo, and a record of what was downloaded."""

    def __init__(self, root: str):
        self.root = root
        self.downloads: list[str] = []

    def put(self, pack_dir: str, repo: str, path: str) -> str:
        dst = os.path.join(self.root, repo, *path.split("/"))
        shutil.copytree(pack_dir, dst)
        return dst

    def api(self):
        hub = self

        class FakeApi:
            def model_info(self, repo_id, timeout=None, **_k):
                if not os.path.isdir(os.path.join(hub.root, repo_id)):
                    raise _not_found(repo_id)
                return types.SimpleNamespace(sha=PACK_REPO_REV)

            def list_repo_tree(self, repo_id, path_in_repo=None, recursive=False, revision=None,
                               **_k):
                from huggingface_hub.errors import EntryNotFoundError
                from huggingface_hub.hf_api import RepoFile

                base = os.path.join(hub.root, repo_id, *path_in_repo.split("/"))
                if not os.path.isdir(base):
                    raise EntryNotFoundError(path_in_repo)
                for r, _dirs, files in os.walk(base):
                    for f in files:
                        full = os.path.join(r, f)
                        rel = os.path.relpath(full, os.path.join(hub.root, repo_id))
                        yield RepoFile(path=rel.replace(os.sep, "/"), size=os.path.getsize(full),
                                       oid="0" * 40)

        return FakeApi

    def hf_hub_download(self, repo_id, filename, revision=None, local_dir=None, **_k):
        src = os.path.join(self.root, repo_id, *filename.split("/"))
        dst = os.path.join(local_dir, *filename.split("/"))
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copyfile(src, dst)
        self.downloads.append(filename)
        return dst

    def snapshot_download(self, repo_id, revision=None, local_dir=None, allow_patterns=None,
                          ignore_patterns=None, **_k):
        import fnmatch

        base = os.path.join(self.root, repo_id)
        for r, _dirs, files in os.walk(base):
            for f in files:
                rel = os.path.relpath(os.path.join(r, f), base).replace(os.sep, "/")
                if allow_patterns and not any(fnmatch.fnmatch(rel, p) for p in allow_patterns):
                    continue
                if ignore_patterns and any(fnmatch.fnmatch(rel, p) for p in ignore_patterns):
                    continue
                self.hf_hub_download(repo_id, rel, revision, local_dir)
        return local_dir


@pytest.fixture
def hub(tmp_path, monkeypatch, toy):
    """The toy pack published at its path in HUB, and the listing the Hub
    gives for source/A@AAA."""
    snap, pack_dir = toy
    fake = FakeHub(str(tmp_path / "hub"))
    fake.put(pack_dir, HUB, hub_packs.pack_path(AAA, "sip"))
    monkeypatch.setenv("DRINKME_PACK_HUB", HUB)
    monkeypatch.setattr(huggingface_hub, "HfApi", fake.api())
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake.hf_hub_download)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake.snapshot_download)
    fake.listing = hub_listing_of(snap)
    monkeypatch.setattr(upstream, "hub_listing", lambda repo, commit, timeout=None: fake.listing)
    return fake


def _dest(tmp_path) -> str:
    return str(tmp_path / "cache" / "packs" / "source--A@aaaaaaaaaaaa")


def _leftovers(dest: str) -> list[str]:
    parent = os.path.dirname(dest)
    return [n for n in os.listdir(parent) if ".hub-" in n] if os.path.isdir(parent) else []


# ------------------------------------------------------------ naming --

def test_the_pack_repo_and_path_bind_repo_commit_profile_and_format():
    assert hub_packs.pack_repo("ns", "Qwen/Qwen3-8B") == "ns/Qwen3-8B-drinkme"
    assert hub_packs.pack_repo("ns/mirror", "Qwen/Qwen3-8B") == "ns/mirror"
    sha = "b968826d9c46dd6066d109eabc6255188de91218"
    assert hub_packs.pack_path(sha, "sip") == f"v1/sip/{sha}"
    assert hub_packs.pack_path(sha, "gulp", version=2) == f"v2/gulp/{sha}"


def test_the_namespace_is_drinkme_packs(monkeypatch):
    assert hub_packs.PACK_HUB_NAMESPACE == "drinkme-packs"
    monkeypatch.delenv("DRINKME_PACK_HUB")
    assert hub_packs.pack_hub() == "drinkme-packs"


def test_no_namespace_means_nothing_is_looked_up(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(hub_packs, "PACK_HUB_NAMESPACE", None)
    monkeypatch.delenv("DRINKME_PACK_HUB")
    assert hub_packs.pack_hub() is None

    def no_hub(*_a, **_k):
        raise AssertionError("looked up a pack with no pack hub set")

    monkeypatch.setattr(huggingface_hub, "HfApi", no_hub)
    assert hub_packs.fetch(REPO, AAA, _dest(tmp_path), "sip") is None
    assert capsys.readouterr().out == ""


def test_the_environment_names_a_hub_or_turns_it_off(monkeypatch):
    monkeypatch.setenv("DRINKME_PACK_HUB", "someone")
    assert hub_packs.pack_hub() == "someone"
    monkeypatch.setenv("DRINKME_PACK_HUB", "off")
    assert hub_packs.pack_hub() is None


# ------------------------------------------------------------- fetch --

def test_a_published_pack_is_downloaded_verified_and_installed(hub, tmp_path, capsys):
    dest = _dest(tmp_path)
    assert hub_packs.fetch(REPO, AAA, dest, "sip") == dest
    out = capsys.readouterr().out
    assert f"downloading the published sip pack from {HUB}" in out
    assert "% of the " in out and "BF16 checkpoint" in out
    assert "upstream MATCH: 1 of 1 weight files" in out
    assert os.path.isfile(os.path.join(dest, "meta.json"))
    assert upstream.receipt_current(dest) is not None
    with open(os.path.join(dest, hub_packs.ORIGIN_FILE)) as f:
        origin = json.load(f)
    assert origin["repo"] == HUB and origin["path"] == hub_packs.pack_path(AAA, "sip")
    assert origin["revision"] == PACK_REPO_REV
    assert hub_packs.is_hub_pack(dest) and _leftovers(dest) == []
    assert not os.path.exists(os.path.join(dest, ".cache"))


def test_a_mismatch_is_refused_loudly_and_nothing_is_kept(hub, tmp_path, capsys):
    hub.listing = {**hub.listing, "model.safetensors": {**hub.listing["model.safetensors"],
                                                        "sha256": "f" * 64}}
    dest = _dest(tmp_path)
    assert hub_packs.fetch(REPO, AAA, dest, "sip") is None
    cap = capsys.readouterr()
    assert "REFUSED the published pack" in cap.err and "model.safetensors" in cap.err
    assert "packing locally" in cap.out
    assert not os.path.exists(dest) and _leftovers(dest) == []


def test_a_pack_bound_to_another_commit_is_refused_before_it_downloads(hub, toy, tmp_path,
                                                                       capsys):
    """The pack at OTHER's path is the pack of AAA: its meta.json is read
    first and refused; not one tensor file is fetched."""
    _snap, pack_dir = toy
    hub.put(pack_dir, HUB, hub_packs.pack_path(OTHER, "sip"))
    dest = _dest(tmp_path)
    assert hub_packs.fetch(REPO, OTHER, dest, "sip") is None
    out = capsys.readouterr().out
    assert f"bound to {REPO}@{AAA}" in out and "packing locally" in out
    assert hub.downloads == [f"{hub_packs.pack_path(OTHER, 'sip')}/meta.json"]
    assert not os.path.exists(dest) and _leftovers(dest) == []


def test_no_pack_for_this_commit_is_one_line(hub, tmp_path, capsys):
    assert hub_packs.fetch(REPO, OTHER, _dest(tmp_path), "sip") is None
    out = capsys.readouterr().out
    assert f"no published pack at {HUB}/v1/sip/{OTHER} — packing locally" in out
    assert hub.downloads == []


def test_a_profile_other_than_the_published_one_is_not_found(hub, tmp_path, capsys):
    assert hub_packs.fetch(REPO, AAA, _dest(tmp_path), "gulp") is None
    assert "no published pack at" in capsys.readouterr().out


def test_a_published_pack_failing_its_own_hashes_is_refused(hub, tmp_path, capsys):
    published = os.path.join(hub.root, HUB, *hub_packs.pack_path(AAA, "sip").split("/"))
    with open(os.path.join(published, "meta.json")) as f:
        fn = sorted(json.load(f)["sha256"])[0]
    with open(os.path.join(published, fn), "r+b") as f:
        f.seek(100)
        f.write(b"\xff\xff")
    dest = _dest(tmp_path)
    assert hub_packs.fetch(REPO, AAA, dest, "sip") is None
    assert "failed its own checks" in capsys.readouterr().out
    assert not os.path.exists(dest) and _leftovers(dest) == []


def test_offline_upstream_serves_on_the_file_hashes_and_says_so(hub, tmp_path, monkeypatch,
                                                                capsys):
    def unavailable(repo, commit, timeout=None):
        raise upstream.UpstreamUnavailable("OfflineModeIsEnabled: HF_HUB_OFFLINE is set")

    monkeypatch.setattr(upstream, "hub_listing", unavailable)
    dest = _dest(tmp_path)
    assert hub_packs.fetch(REPO, AAA, dest, "sip") == dest
    err = capsys.readouterr().err
    assert "upstream verification unavailable (OfflineModeIsEnabled" in err
    assert "file-hash check" in err
    assert upstream.read_receipt(dest) is None and hub_packs.is_hub_pack(dest)


# ------------------------------------------------------------- serve --

@pytest.fixture
def local_pack(monkeypatch):
    """serve.ensure_pack's full flow without packing: pack_model records
    the call and returns the path; check and the encoder front door pass."""
    calls = []
    from drinkme import check
    from drinkme.codec import pack, radix_pack

    def fake_pack_model(repo, revision=None, out=None, **_k):
        calls.append((repo, revision, out))
        return out

    def unavailable(*_a, **_k):
        raise check.CheckUnavailable("offline")

    monkeypatch.setattr(pack, "pack_model", fake_pack_model)
    monkeypatch.setattr(radix_pack, "refuse_unless_encoder_available", lambda **_k: None)
    monkeypatch.setattr(check, "check", unavailable)
    return calls


def test_serve_takes_the_published_pack_and_does_not_pack(hub, local_pack, tmp_path):
    from drinkme.serve import ensure_pack

    dest = _dest(tmp_path)
    assert ensure_pack(REPO, AAA, dest, auto_pack=True) == dest
    assert local_pack == [] and upstream.receipt_current(dest) is not None


@pytest.mark.parametrize("how", ["flag", "env"])
def test_no_hub_pack_packs_locally_without_asking(hub, local_pack, tmp_path, monkeypatch, how):
    from drinkme.serve import ensure_pack

    def never(*_a, **_k):
        raise AssertionError("asked the pack hub under --no-hub-pack")

    monkeypatch.setattr(hub_packs, "fetch", never)
    if how == "env":
        monkeypatch.setenv("DRINKME_NO_HUB_PACK", "1")
    dest = _dest(tmp_path)
    ensure_pack(REPO, AAA, dest, auto_pack=True, hub_pack=how != "flag")
    assert local_pack == [(REPO, AAA, dest)]


def test_serve_packs_locally_after_a_mismatch(hub, local_pack, tmp_path):
    from drinkme.serve import ensure_pack

    hub.listing = {**hub.listing, "model.safetensors": {**hub.listing["model.safetensors"],
                                                        "sha256": "f" * 64}}
    dest = _dest(tmp_path)
    ensure_pack(REPO, AAA, dest, auto_pack=True)
    assert local_pack == [(REPO, AAA, dest)]


def test_no_auto_pack_fetches_nothing(hub, tmp_path):
    from drinkme.serve import ensure_pack

    with pytest.raises(exitcodes.Usage, match="--no-auto-pack"):
        ensure_pack(REPO, AAA, _dest(tmp_path), auto_pack=False)
    assert hub.downloads == []


def test_a_cached_hub_pack_without_a_receipt_is_checked_at_the_next_serve(
        hub, local_pack, tmp_path, monkeypatch, capsys):
    from drinkme.serve import ensure_pack

    listing = hub.listing
    monkeypatch.setattr(upstream, "hub_listing", lambda *a, **k: (_ for _ in ()).throw(
        upstream.UpstreamUnavailable("offline")))
    dest = _dest(tmp_path)
    ensure_pack(REPO, AAA, dest, auto_pack=True)
    assert upstream.read_receipt(dest) is None
    # still offline: served on its file hashes, said so, still no receipt
    capsys.readouterr()
    assert ensure_pack(REPO, AAA, dest, auto_pack=True) == dest
    assert "upstream verification unavailable" in capsys.readouterr().err
    # back online: checked now, once, and the receipt written
    monkeypatch.setattr(upstream, "hub_listing", lambda *a, **k: listing)
    assert ensure_pack(REPO, AAA, dest, auto_pack=True) == dest
    assert upstream.receipt_current(dest) is not None
    assert "upstream MATCH" in capsys.readouterr().out
    assert local_pack == []


def test_a_cached_hub_pack_that_mismatches_later_is_deleted_and_packed_locally(
        hub, local_pack, tmp_path, monkeypatch):
    from drinkme.serve import ensure_pack

    monkeypatch.setattr(upstream, "hub_listing", lambda *a, **k: (_ for _ in ()).throw(
        upstream.UpstreamUnavailable("offline")))
    dest = _dest(tmp_path)
    ensure_pack(REPO, AAA, dest, auto_pack=True)
    bad = {**hub.listing, "model.safetensors": {**hub.listing["model.safetensors"],
                                               "sha256": "f" * 64}}
    monkeypatch.setattr(upstream, "hub_listing", lambda *a, **k: bad)
    downloads = len(hub.downloads)
    ensure_pack(REPO, AAA, dest, auto_pack=True)
    assert not os.path.exists(os.path.join(dest, "meta.json"))
    assert local_pack == [(REPO, AAA, dest)]
    assert len(hub.downloads) == downloads  # not fetched again


def test_a_local_pack_is_never_checked_upstream(toy, monkeypatch):
    from drinkme.serve import ensure_pack

    def never(*_a, **_k):
        raise AssertionError("a locally packed pack was checked upstream at serve")

    monkeypatch.setattr(upstream, "hub_listing", never)
    _snap, pack_dir = toy
    assert ensure_pack(REPO, AAA, pack_dir, auto_pack=True) == pack_dir


# ------------------------------------------------------ verify --upstream --

@pytest.mark.parametrize("case,code", [("match", 0), ("mismatch", 3), ("offline", 4)])
def test_verify_upstream_exit_codes(toy, tmp_path, monkeypatch, capsys, case, code):
    from drinkme.cli import main

    snap, src = toy
    pack_dir = str(tmp_path / "p")
    shutil.copytree(src, pack_dir)
    listing = hub_listing_of(snap)
    if case == "mismatch":
        listing["model.safetensors"]["sha256"] = "f" * 64
    if case == "offline":
        def fake(*_a, **_k):
            raise upstream.UpstreamUnavailable("offline")
    else:
        def fake(*_a, **_k):
            return listing
    monkeypatch.setattr(upstream, "hub_listing", fake)
    assert main(["verify", "--pack-dir", pack_dir, "--upstream"]) == code
    cap = capsys.readouterr()
    if case == "match":
        assert "  MATCH       model.safetensors" in cap.out and "upstream MATCH" in cap.out
        assert "receipt:" in cap.out
    elif case == "mismatch":
        assert "MISMATCH    model.safetensors" in cap.out
    else:
        assert "upstream verification unavailable (offline)" in cap.err
