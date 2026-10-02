"""the manifest digest binds tensor NAMES to files.

The failure it guards: hashing each .npz by filename while the map from
tensor name to filename sits unhashed in meta.json. The probe: a two-tensor
pack, a = 1 and b = 2; swap only meta['tensors']['a'] and ['b']. The
iterator then decodes a = 2, and a filename-only verify_hashes() still
returns True with an identical persisted-cache id — no hash recomputation,
no forged digest.

So finish() records the digest of a canonical manifest (name -> file -> hash -> shape/
dtype/layout + the source identity, codec/identity.py) and verify_hashes
recomputes its digest from the live maps; the cache id derives from it.
Every pack records it at pack time; a meta.json without the
manifest (or the hashes) is refused, never served on a weaker check.
"""

import json
import os

import numpy as np
import pytest

from drinkme.codec import identity, radix_pack as rp
from drinkme.codec.pack import iter_pack_dir, save_pack_dir, verify_hashes, verify_pack
from drinkme.serving import slotstore

META = {"hfRepo": "t/t", "revision": None, "dtype": "bf16"}


def _const(v: float, R=8, C=64):
    """A bf16 tensor's bit pattern, every element == v."""
    return np.full((R, C), np.float32(v)).view(np.uint32) >> 16


def _two_tensor_pack(path, profile="sip", meta=META):
    """The probe's pack: a = 1, b = 2, same shape. A constant tensor is one
    exponent, which every profile codes in its first tier — never the raw
    fallback; `profile="raw"` writes them as raw fallbacks instead."""
    def mk(v):
        U = _const(v).astype(np.uint16)
        if profile == "raw":
            return rp.raw_dict(U)
        p = rp.pack_array_radix(U, profile, encoder="numpy")
        assert p is not None
        return p
    save_pack_dir(str(path), {"a": mk(1.0), "b": mk(2.0)}, dict(meta))
    return str(path)


def _meta(path):
    with open(os.path.join(path, "meta.json")) as f:
        return json.load(f)


def _write_meta(path, meta):
    with open(os.path.join(path, "meta.json"), "w") as f:
        json.dump(meta, f)


def _decoded_value(path, name):
    for n, p in iter_pack_dir(path):
        if n == name:
            U = rp.decode_back(p)
            return float((U.astype(np.uint32) << 16).view(np.float32)[0, 0])
    raise KeyError(name)


# ------------------------------------------------------------- the ruler --


@pytest.mark.parametrize("profile", ["sip", "gulp", "raw"])
def test_swapping_two_same_shaped_tensor_entries_fails_verification_and_moves_the_cache_id(tmp_path, profile):
    """The ruler: swap meta['tensors']['a'] and ['b']. Every hashed file is
    unchanged; the iterator now decodes a = 2. Verification must FAIL and
    the cache id must CHANGE; a manifest that hashed the files alone would
    pass verify_hashes with pack_id_from_meta identical."""
    p = _two_tensor_pack(tmp_path / "p", profile=profile)
    assert verify_hashes(p) is True
    before = _meta(p)
    id_before = slotstore.pack_id_from_meta(before)
    assert _decoded_value(p, "a") == 1.0 and _decoded_value(p, "b") == 2.0

    swapped = json.loads(json.dumps(before))
    swapped["tensors"]["a"], swapped["tensors"]["b"] = before["tensors"]["b"], before["tensors"]["a"]
    _write_meta(p, swapped)
    assert _decoded_value(p, "a") == 2.0  # the iterator is fooled; verify_hashes must not be

    with pytest.raises(ValueError, match="manifest broken"):
        verify_hashes(p)
    assert slotstore.pack_id_from_meta(swapped) != id_before


def test_the_manifest_binds_descriptors_and_source_too(tmp_path):
    """Swapping the descriptors along with the names, or touching the source
    identity, is also a manifest mismatch — the manifest is one canonical thing."""
    p = _two_tensor_pack(tmp_path / "p", meta={**META, "source": {
        "identityVersion": 1, "kind": "hub", "repo": "t/t",
        "revision": "a" * 40, "digest": "0" * 64}})
    base = _meta(p)

    both = json.loads(json.dumps(base))
    both["tensors"]["a"], both["tensors"]["b"] = base["tensors"]["b"], base["tensors"]["a"]
    both["tensorInfo"]["a"], both["tensorInfo"]["b"] = base["tensorInfo"]["b"], base["tensorInfo"]["a"]
    _write_meta(p, both)
    with pytest.raises(ValueError, match="manifest broken"):
        verify_hashes(p)

    src = json.loads(json.dumps(base))
    src["source"]["revision"] = "b" * 40
    _write_meta(p, src)
    with pytest.raises(ValueError, match="manifest broken"):
        verify_hashes(p)

    shape = json.loads(json.dumps(base))
    shape["tensorInfo"]["a"]["shape"] = [64, 8]
    _write_meta(p, shape)
    with pytest.raises(ValueError, match="manifest broken"):
        verify_hashes(p)

    _write_meta(p, base)
    assert verify_hashes(p) is True


# ------------------------------------------------------------ new packs --


def test_new_packs_record_the_manifest_at_pack_time(tmp_path):
    p = _two_tensor_pack(tmp_path / "p")
    meta = _meta(p)
    assert meta["tensorInfo"] == {"a": {"shape": [8, 64], "dtype": "bf16", "codec": "radix", "profile": "sip", "widths": [3, 8], "blockSize": 1024},
                                  "b": {"shape": [8, 64], "dtype": "bf16", "codec": "radix", "profile": "sip", "widths": [3, 8], "blockSize": 1024}}
    assert meta["manifestSha256"] == identity.manifest_digest(meta)
    m = identity.pack_manifest(meta)
    assert m["manifestVersion"] == identity.MANIFEST_VERSION
    assert m["formatVersion"] == 1 and m["source"] is None
    assert m["tensors"]["a"] == {"file": "t0000.npz", "sha256": meta["sha256"]["t0000.npz"],
                                 "shape": [8, 64], "dtype": "bf16", "codec": "radix", "profile": "sip",
                                 "widths": [3, 8], "blockSize": 1024}
    assert slotstore.pack_id_from_meta(meta) == "manifest" + meta["manifestSha256"][:16]
    # a raw fallback's descriptor names it raw and nothing else
    p2 = _two_tensor_pack(tmp_path / "p2", profile="raw")
    assert _meta(p2)["tensorInfo"]["a"] == {"shape": [8, 64], "dtype": "bf16", "codec": "raw"}
    # another profile's descriptors differ, so the manifests do
    p3 = _two_tensor_pack(tmp_path / "p3", profile="gulp")
    assert _meta(p3)["tensorInfo"]["a"]["widths"] == [2, 2, 4, 8]
    assert _meta(p3)["manifestSha256"] != meta["manifestSha256"]


def test_the_manifest_digest_is_canonical(tmp_path):
    """Key order in meta.json is not identity: the same maps in any order
    digest the same; and the digest is over the LIVE maps, so a descriptor
    edit is visible even if the recorded digest is left alone."""
    p = _two_tensor_pack(tmp_path / "p")
    meta = _meta(p)
    shuffled = {k: meta[k] for k in reversed(list(meta))}
    shuffled["tensors"] = {k: meta["tensors"][k] for k in reversed(list(meta["tensors"]))}
    assert identity.manifest_digest(shuffled) == meta["manifestSha256"]


# ----------------------------------------- packs without their hashes --


def test_a_pack_without_a_manifest_is_refused_and_has_no_cache_id(tmp_path):
    """The manifest is what binds tensor names to files; a meta.json
    without one is not a pack this drinkme wrote (every pack records it at
    pack time) — verify_hashes refuses it by name, and pack_id_from_meta has
    no tier to fall back to (a swapped tensor map would otherwise land in
    the same slot-store directory as the original)."""
    p = _two_tensor_pack(tmp_path / "p")
    meta = _meta(p)
    del meta["manifestSha256"]
    del meta["tensorInfo"]
    _write_meta(p, meta)
    with pytest.raises(ValueError, match=r"no manifest: .*re-pack: `drinkme pack --model t/t --replace`"):
        verify_hashes(p)
    with pytest.raises(ValueError, match="no manifestSha256"):
        slotstore.pack_id_from_meta(meta)
    del meta["sha256"]
    _write_meta(p, meta)
    with pytest.raises(ValueError, match="not verifiable"):
        verify_hashes(p)


def test_a_pack_without_a_manifest_is_refused_by_the_real_loader(tmp_path):
    """The torch loader on a hashed pack with no manifest: refused before a
    weight is allocated, with verify_hashes's own line."""
    pytest.importorskip("torch")
    from drinkme.codec.pack import pack_model
    from drinkme.serving.engines import load_compressed
    from tests.test_pack_identity import _toy_checkpoint

    model_dir, pack_dir = str(tmp_path / "m"), str(tmp_path / "p")
    _toy_checkpoint(model_dir, seed=3)
    pack_model(model_dir, None, pack_dir, progress=lambda *_: None)
    meta = _meta(pack_dir)
    del meta["manifestSha256"], meta["tensorInfo"]
    _write_meta(pack_dir, meta)
    os.environ["DRINKME_PREFIX_SLOTS"] = "0"
    try:
        with pytest.raises(ValueError, match="no manifest"):
            load_compressed(model_dir, None, pack_dir, device="cpu")
    finally:
        del os.environ["DRINKME_PREFIX_SLOTS"]


def test_drinkme_verify_checks_in_place_and_never_rewrites_meta(tmp_path, monkeypatch):
    """The verify verb on a pack as written: the format gate, then verify_hashes
    (every file, the manifest), one line naming the count and the digest;
    meta.json untouched (no temp file, no os.replace), the npz files
    untouched; the same on the second run. A `mtp/` sub-pack is verified
    the same way (test_serving_mtp covers the head pack's writer)."""
    from drinkme.codec.pack import pack_model
    from tests.test_pack_identity import _toy_checkpoint

    model_dir, pack_dir = str(tmp_path / "m"), str(tmp_path / "p")
    _toy_checkpoint(model_dir, seed=4)
    pack_model(model_dir, None, pack_dir, progress=lambda *_: None)
    full = _meta(pack_dir)
    npz_before = {fn: open(os.path.join(pack_dir, fn), "rb").read() for fn in full["tensors"].values()}
    mtime = os.path.getmtime(os.path.join(pack_dir, "meta.json"))

    replaced = []
    real_replace = os.replace
    monkeypatch.setattr(os, "replace", lambda a, b: replaced.append((a, b)) or real_replace(a, b))
    lines = []
    verify_pack(pack_dir, progress=lines.append)
    from drinkme.codec.identity import describe

    assert lines == [f"verified, manifest and all: {pack_dir} — {len(full['tensors'])} files "
                     f"verified, manifest {full['manifestSha256'][:16]}…; embedded checkpoint: "
                     f"{len(full['embedded']['files'])} files verified, identity matches "
                     f"{describe(full['source'])}"]
    assert replaced == [] and not os.path.exists(os.path.join(pack_dir, "meta.json.tmp"))
    assert _meta(pack_dir) == full and os.path.getmtime(os.path.join(pack_dir, "meta.json")) == mtime
    assert {fn: open(os.path.join(pack_dir, fn), "rb").read()
            for fn in full["tensors"].values()} == npz_before
    assert slotstore.pack_id_from_meta(full) == "manifest" + full["manifestSha256"][:16]
    lines.clear()
    verify_pack(pack_dir, progress=lines.append)
    assert len(lines) == 1 and lines[0].startswith("verified, manifest and all")
    # a hash mismatch is the gate's refusal, not a rehash
    with open(os.path.join(pack_dir, next(iter(full["tensors"].values()))), "ab") as f:
        f.write(b"\0")
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_pack(pack_dir, progress=lines.append)
    assert _meta(pack_dir) == full
