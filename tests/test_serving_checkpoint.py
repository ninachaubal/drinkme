"""serving/checkpoint.py — the local-first snapshot resolver.

A local snapshot counted as a hit if ANY *.safetensors file
existed, so an interrupted multi-shard download (index + shard one of two)
suppressed the network fetch that would have repaired it, forever — the
packer refused on the missing shard, and every retry took the same
local-only path. These tests drive snapshot_dir with a mocked
huggingface_hub.snapshot_download and a fabricated cache dir: no network,
no downloads, no real cache touched.
"""

import json
import os

import pytest

from drinkme.serving import checkpoint


def _snapshot(d, shards=("model-00001-of-00002.safetensors",
                         "model-00002-of-00002.safetensors"),
              present=None, index=True, extra=("config.json", "tokenizer_config.json",
                                               "tokenizer.json")):
    """A fabricated HF snapshot dir: `present` names the shards actually on
    disk (default: all of them); `index` writes the two-shard index."""
    os.makedirs(d, exist_ok=True)
    present = list(shards) if present is None else list(present)
    if index:
        with open(os.path.join(d, "model.safetensors.index.json"), "w") as f:
            json.dump({"weight_map": {f"w{i}": s for i, s in enumerate(shards)}}, f)
    for s in present:
        with open(os.path.join(d, s), "wb") as f:
            f.write(b"\0" * 16)
    for name in extra:
        with open(os.path.join(d, name), "w") as f:
            f.write("{}")
    return str(d)


class _Hub:
    """Records every snapshot_download call; the local-only call returns the
    fabricated partial dir, the real fetch returns `fetched` (or raises)."""

    def __init__(self, local, fetched=None, offline=None):
        self.calls = []
        self.local, self.fetched, self.offline = local, fetched, offline

    def __call__(self, repo, revision=None, allow_patterns=None, local_files_only=False):
        self.calls.append({"repo": repo, "revision": revision,
                           "local_files_only": local_files_only})
        if local_files_only:
            return self.local
        if self.offline is not None:
            raise self.offline
        return self.fetched


@pytest.fixture(autouse=True)
def _fresh_cache():
    checkpoint.snapshot_dir.cache_clear()
    yield
    checkpoint.snapshot_dir.cache_clear()


def test_missing_from_snapshot_names_every_absent_shard_and_serving_file(tmp_path):
    complete = _snapshot(tmp_path / "ok")
    assert checkpoint.missing_from_snapshot(complete) == []

    one_of_two = _snapshot(tmp_path / "half", present=["model-00001-of-00002.safetensors"])
    assert checkpoint.missing_from_snapshot(one_of_two) == ["model-00002-of-00002.safetensors"]

    weights_only = _snapshot(tmp_path / "w", extra=())
    assert set(checkpoint.missing_from_snapshot(weights_only)) == {
        "config.json", "tokenizer_config.json",
        " or ".join(checkpoint._VOCAB_FILES)}

    # no index at all: a bare model.safetensors is the whole checkpoint
    bare = _snapshot(tmp_path / "bare", shards=("model.safetensors",), index=False)
    assert checkpoint.missing_from_snapshot(bare) == []
    empty = _snapshot(tmp_path / "empty", shards=(), index=False)
    assert "*.safetensors" in checkpoint.missing_from_snapshot(empty)


def test_one_shard_of_two_is_not_a_hit_the_real_fetch_runs(tmp_path, monkeypatch):
    """The ruler: a two-shard index with one
    shard present must NOT be returned as a hit — the mocked hub must see a
    second call, the real fetch, with local_files_only False, and its
    result is what snapshot_dir returns. The failure it catches: exactly one
    call, local_files_only=True, and the partial dir back as the checkpoint."""
    partial = _snapshot(tmp_path / "partial", present=["model-00001-of-00002.safetensors"])
    fetched = _snapshot(tmp_path / "fetched")
    hub = _Hub(partial, fetched=fetched)
    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "snapshot_download", hub)

    got = checkpoint.snapshot_dir("review/two-shards", "main")
    assert got == fetched
    assert [c["local_files_only"] for c in hub.calls] == [True, False]
    assert all(c["repo"] == "review/two-shards" and c["revision"] == "main"
               for c in hub.calls)


def test_a_complete_local_snapshot_is_a_hit_and_never_touches_the_network(tmp_path, monkeypatch):
    complete = _snapshot(tmp_path / "complete")
    hub = _Hub(complete, fetched="/never")
    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "snapshot_download", hub)
    assert checkpoint.snapshot_dir("review/complete", None) == complete
    assert [c["local_files_only"] for c in hub.calls] == [True]


def test_offline_with_a_partial_cache_names_the_missing_files(tmp_path, monkeypatch):
    """Genuinely offline: the error is actionable — it names the snapshot
    dir and the exact files it lacks, and chains the hub's own error."""
    partial = _snapshot(tmp_path / "partial", present=["model-00001-of-00002.safetensors"])
    hub = _Hub(partial, offline=ConnectionError("no route to hub"))
    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "snapshot_download", hub)
    with pytest.raises(RuntimeError, match="model-00002-of-00002.safetensors") as ei:
        checkpoint.snapshot_dir("review/offline", "abc")
    assert partial in str(ei.value) and "review/offline@abc" in str(ei.value)
    assert isinstance(ei.value.__cause__, ConnectionError)
    assert [c["local_files_only"] for c in hub.calls] == [True, False]


def test_a_local_directory_is_returned_as_is(tmp_path, monkeypatch):
    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "snapshot_download",
                        lambda *a, **k: pytest.fail("a local dir must not hit the hub"))
    assert checkpoint.snapshot_dir(str(tmp_path), None) == str(tmp_path)


def test_a_genuine_miss_announces_what_it_will_download_and_how_big(tmp_path, monkeypatch, capsys):
    """Before any download: which checkpoint, why (never cached here), and
    how much — sized from the Hub's own listing, filtered to the patterns
    snapshot_dir fetches (a README is never one of them)."""
    import types

    import huggingface_hub

    def miss(repo, revision=None, allow_patterns=None, local_files_only=False):
        if local_files_only:
            raise FileNotFoundError("not cached")
        return _snapshot(tmp_path / "downloaded")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", miss)
    sib = types.SimpleNamespace
    info = sib(siblings=[sib(rfilename="model.safetensors", size=3_000_000_000),
                         sib(rfilename="config.json", size=1000),
                         sib(rfilename="README.md", size=5000)])
    monkeypatch.setattr(huggingface_hub, "HfApi", lambda: sib(model_info=lambda *a, **k: info))
    assert checkpoint.snapshot_dir("org/model", "rev123") == str(tmp_path / "downloaded")
    out = capsys.readouterr().out
    assert "downloading checkpoint org/model@rev123" in out
    assert "not in the local Hugging Face cache" in out
    assert "2 files, 3.00 GB" in out
