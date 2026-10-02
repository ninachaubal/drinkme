"""Upstream verification (src/drinkme/upstream.py): a pack rebuilt into its
source checkpoint's files, byte for byte, against the hashes the Hub
publishes for them (docs/pack-format.md#upstream-verification).

The checkpoint is the toy Llama of tests/test_pack_identity.py in a
fabricated hub-cache layout, packed through `drinkme pack`'s own
pack_model; the Hub's listing is computed from the toy's files the way the
Hub computes it (LFS sha256 for the weights and tokenizer.json, the git
blob id for the rest) and handed in, so nothing here touches the network.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import struct

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from drinkme import upstream  # noqa: E402
from drinkme.codec import identity  # noqa: E402
from drinkme.codec.pack import embedded_dir, verify_pack  # noqa: E402
from tests.test_pack_identity import AAA, _snapshot, _toy_checkpoint  # noqa: E402

REPO = "source/A"


def _sha256(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def hub_listing_of(snap: str) -> dict:
    """What the Hub lists for the files of `snap`: LFS files by sha256,
    the rest by git blob id."""
    out = {}
    for name in sorted(os.listdir(snap)):
        p = os.path.join(snap, name)
        lfs = name.endswith(".safetensors") or name == "tokenizer.json"
        out[name] = {"size": os.path.getsize(p), "sha256": _sha256(p) if lfs else None,
                     "blobId": None if lfs else upstream.git_blob_id(p)}
    return out


@pytest.fixture(scope="module")
def toy(tmp_path_factory):
    """(snapshot dir, pack dir): source/A@AAA packed self-contained."""
    from drinkme import arms
    from drinkme.codec.pack import pack_model
    from drinkme.serving import checkpoint

    cache = str(tmp_path_factory.mktemp("hub"))
    snap = _snapshot(cache, REPO, AAA)
    _toy_checkpoint(snap, seed=3)
    resolve = lambda repo, revision=None: repo if os.path.isdir(repo) else snap  # noqa: E731
    pack_dir = str(tmp_path_factory.mktemp("packs") / "source--A@aaaaaaaaaaaa")
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(checkpoint, "snapshot_dir", resolve)
        mp.setattr(arms, "snapshot_dir", resolve)
        pack_model(REPO, AAA, pack_dir, progress=lambda *_: None)
    return snap, pack_dir


@pytest.fixture
def pack(toy, tmp_path):
    """A copy of the toy pack this test may change."""
    snap, src = toy
    dst = str(tmp_path / os.path.basename(src))
    shutil.copytree(src, dst)
    return snap, dst


def _meta(pack_dir: str) -> dict:
    with open(os.path.join(pack_dir, "meta.json")) as f:
        return json.load(f)


def test_every_file_rebuilds_to_the_publishers_hash(pack):
    snap, pack_dir = pack
    v = upstream.verify_upstream(pack_dir, listing=hub_listing_of(snap))
    assert v.verdict == upstream.MATCH and v.ok, v.summary()
    weights = v.weights()
    assert [f.name for f in weights] == ["model.safetensors"]
    assert weights[0].rebuilt == weights[0].upstream == _sha256(os.path.join(snap, "model.safetensors"))
    small = {f.name: f.verdict for f in v.files if f.kind == "file"}
    assert small and set(small.values()) == {upstream.MATCH}
    assert "config.json" in small and "tokenizer.json" in small
    assert v.summary().startswith("upstream MATCH: 1 of 1 weight files")


def test_the_receipt_is_written_and_goes_stale_with_the_pack(pack):
    snap, pack_dir = pack
    upstream.verify_upstream(pack_dir, listing=hub_listing_of(snap))
    rec = upstream.receipt_current(pack_dir)
    assert rec is not None and rec["verdict"] == upstream.MATCH
    assert rec["repo"] == REPO and rec["commit"] == AAA
    assert rec["pack"]["manifestSha256"] == _meta(pack_dir)["manifestSha256"]
    assert rec["files"][0]["upstream"] == rec["files"][0]["rebuilt"]
    # any change to the file hashes the manifest binds retires the receipt
    meta = _meta(pack_dir)
    fn = next(iter(meta["sha256"]))
    meta["sha256"][fn] = "0" * 64
    with open(os.path.join(pack_dir, "meta.json"), "w") as f:
        json.dump(meta, f)
    assert upstream.receipt_current(pack_dir) is None


def test_a_hash_the_publisher_does_not_list_is_a_mismatch_naming_the_file(pack):
    snap, pack_dir = pack
    listing = hub_listing_of(snap)
    listing["model.safetensors"]["sha256"] = "f" * 64
    v = upstream.verify_upstream(pack_dir, listing=listing)
    assert v.verdict == upstream.MISMATCH and not v.ok
    bad = v.mismatched()
    assert [f.name for f in bad] == ["model.safetensors"]
    assert "the publisher's ffffffffffffffff" in bad[0].reason
    assert upstream.receipt_current(pack_dir) is None  # a MISMATCH receipt is not current
    assert upstream.read_receipt(pack_dir)["verdict"] == upstream.MISMATCH


def test_a_changed_weight_rebuilds_to_another_hash(pack):
    """A tamperer who re-encodes one packed tensor and re-records its hash
    and the manifest passes the pack's own gate; the rebuilt shard does
    not hash to the publisher's file."""
    from drinkme.codec.pack import _scalars_for, read_pack_tensor
    from drinkme.codec.radix_pack import decode_back, pack_array_radix

    snap, pack_dir = pack
    meta = _meta(pack_dir)
    name = sorted(meta["tensors"])[0]
    p = read_pack_tensor(pack_dir, meta, name)
    bits = decode_back(p).copy()
    bits[0, 0] ^= 1  # one mantissa bit
    q = pack_array_radix(bits, p["profile"])
    fn = meta["tensors"][name]
    path = os.path.join(pack_dir, fn)
    np.savez(path, **{k: q[k] for k in ("rx_palette", "rx_offsets", "rx_data")},
             scalars=np.array([json.dumps({k: q[k] for k in _scalars_for(q)})]))
    meta["sha256"][fn] = _sha256(path)
    meta["manifestSha256"] = identity.manifest_digest(meta)
    with open(os.path.join(pack_dir, "meta.json"), "w") as f:
        json.dump(meta, f)
    verify_pack(pack_dir, progress=lambda *_: None)  # the pack's own gate passes
    v = upstream.verify_upstream(pack_dir, listing=hub_listing_of(snap))
    assert v.verdict == upstream.MISMATCH
    assert v.mismatched()[0].name == "model.safetensors"


def test_a_tensor_the_pack_does_not_carry_is_not_covered_by_name(pack):
    """A shard header naming a tensor the pack holds nowhere (as a tower
    drinkme does not serve would be): NOT COVERED, the tensor named, and
    never a pass."""
    snap, pack_dir = pack
    rec_path = os.path.join(embedded_dir(pack_dir), identity.EMBEDDED_IDENTITY_FILE)
    with open(rec_path) as f:
        rec = json.load(f)
    shard = rec["shards"][0]
    header = json.loads(shard["header"])
    end = max(v["data_offsets"][1] for k, v in header.items() if k != "__metadata__")
    header["model.visual.patch_embed.weight"] = {"dtype": "BF16", "shape": [2, 2],
                                                 "data_offsets": [end, end + 8]}
    shard["header"] = json.dumps(header, separators=(",", ":"))
    shard["size"] = 8 + len(shard["header"].encode()) + end + 8
    with open(rec_path, "w") as f:
        json.dump(rec, f)
    listing = hub_listing_of(snap)
    listing["model.safetensors"]["size"] = shard["size"]
    v = upstream.verify_upstream(pack_dir, listing=listing, write_receipt=False)
    assert v.verdict == upstream.NOT_COVERED and not v.ok
    f = v.weights()[0]
    assert f.verdict == upstream.NOT_COVERED
    assert f.tensors == ["model.visual.patch_embed.weight"]
    assert "model.visual.patch_embed.weight" in f.reason


def test_a_weight_file_the_pack_has_no_header_for_is_not_covered(pack):
    snap, pack_dir = pack
    listing = hub_listing_of(snap)
    listing["model-00002-of-00002.safetensors"] = {"size": 10, "sha256": "e" * 64, "blobId": None}
    v = upstream.verify_upstream(pack_dir, listing=listing, write_receipt=False)
    assert v.verdict == upstream.NOT_COVERED
    assert any(f.name == "model-00002-of-00002.safetensors" and f.verdict == upstream.NOT_COVERED
               for f in v.files)


def test_an_edited_embedded_file_is_a_mismatch(pack):
    snap, pack_dir = pack
    listing = hub_listing_of(snap)
    listing["config.json"]["blobId"] = "0" * 40
    v = upstream.verify_upstream(pack_dir, listing=listing, write_receipt=False)
    assert v.verdict == upstream.MISMATCH
    assert [f.name for f in v.mismatched()] == ["config.json"]


def test_the_numpy_decoder_gives_the_same_verdict(pack, monkeypatch):
    snap, pack_dir = pack
    monkeypatch.setenv("DRINKME_RADIX_ENCODER", "numpy")
    v = upstream.verify_upstream(pack_dir, listing=hub_listing_of(snap), write_receipt=False)
    assert v.ok and v.decoder == "numpy"


def test_offline_is_unavailable_not_a_verdict(pack, monkeypatch):
    """No listing: UpstreamUnavailable, which callers report and continue
    past (serve on the file-hash check; verify exits 4). The suite's
    conftest already makes HfApi.model_info fail; offline mode too."""
    _snap, pack_dir = pack
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    with pytest.raises(upstream.UpstreamUnavailable):
        upstream.verify_upstream(pack_dir)
    assert upstream.read_receipt(pack_dir) is None


def test_a_pack_without_a_hub_commit_has_nothing_to_compare_with(pack):
    _snap, pack_dir = pack
    meta = _meta(pack_dir)
    meta["source"] = {**meta["source"], "kind": "local", "revision": None}
    with open(os.path.join(pack_dir, "meta.json"), "w") as f:
        json.dump(meta, f)
    with pytest.raises(ValueError, match="not a Hub commit"):
        upstream.verify_upstream(pack_dir, listing={})


def test_the_rebuilt_bytes_are_the_file_itself(pack):
    """The rebuild's own stream, collected rather than hashed, is the
    source shard byte for byte (the 8-byte length, the header, the data)."""
    snap, pack_dir = pack
    with open(os.path.join(embedded_dir(pack_dir), identity.EMBEDDED_IDENTITY_FILE)) as f:
        shard = json.load(f)["shards"][0]
    p = upstream._Pack(pack_dir, _meta(pack_dir))
    header_bytes = shard["header"].encode()
    parts, missing = p.plan(json.loads(header_bytes))
    assert not missing
    out = bytearray(struct.pack("<Q", len(header_bytes)) + header_bytes)
    with open(p.remainder_path, "rb") as rem:
        for part in parts:
            got = p.tensor_bytes(part, 2)
            if got[0] == "buffer":
                out += got[1].tobytes()
            else:
                rem.seek(got[1])
                out += rem.read(got[2])
    with open(os.path.join(snap, "model.safetensors"), "rb") as f:
        assert bytes(out) == f.read()


def test_the_text_wrap_table_is_arms_own():
    """upstream._TEXT_WRAP is a torch-free copy of arms._TEXT_WRAP."""
    from drinkme import arms

    assert upstream._TEXT_WRAP == arms._TEXT_WRAP


def test_native_decode_on_threads_is_the_same_array():
    from drinkme.codec import radix_native
    from drinkme.codec.radix_pack import _palette_u8, pack_array_radix

    if not radix_native.available():
        pytest.skip("no native codec on this machine")
    rng = np.random.default_rng(0)
    bits = (rng.normal(0, 0.02, (37, 2048)).astype(np.float32).view(np.uint32) >> 16).astype(np.uint16)
    p = pack_array_radix(bits, "sip")
    args = (_palette_u8(p), p["rx_offsets"], p["rx_data"], 37, 2048, tuple(p["widths"]))
    assert np.array_equal(radix_native.decode(*args, workers=4), bits)
    assert np.array_equal(radix_native.decode(*args, workers=1), bits)
