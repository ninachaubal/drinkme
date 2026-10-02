"""PackWriter is transactional.

The failure it guards: a writer that reuses the destination directory,
overwrites tensor files in place and leaves the existing meta.json standing
until finish(). A failed re-pack then leaves old metadata over a mixture of
old and new files — changed files break the previous pack's hashes (a
usable artifact destroyed), unchanged ones leave it apparently valid despite
the failed attempt — and finish() truncates meta.json in place. The probe:
after a valid pack, a second writer declares two tensors, writes one, and
finish() raises 'meta.json withheld' with the previous meta.json still there.

So every write goes to `<dest>.staging-<pid>`, finish() publishes by one
rename, a non-empty destination is refused unless replace=True (stage
fully, then swap; the old pack kept until the swap succeeded), a
`<dest>.lock` serialises writers, and meta.json is temp + os.replace.
"""

import json
import os

import numpy as np
import pytest

from drinkme.codec import radix_pack as rp
from drinkme.codec.pack import PackWriter, load_pack_dir, save_pack_dir, verify_hashes

META = {"hfRepo": "t/t", "revision": None, "dtype": "bf16"}


def _const(v: float, R=8, C=64):
    return (np.full((R, C), np.float32(v)).view(np.uint32) >> 16).astype(np.uint16)


def _radix(U):
    p = rp.pack_array_radix(U, "sip", encoder="numpy")
    assert p is not None
    return p


def _tree(path):
    """Every file under a pack dir, bytes and all — the byte-for-byte oracle."""
    out = {}
    for root, _, files in os.walk(path):
        for fn in files:
            fp = os.path.join(root, fn)
            with open(fp, "rb") as f:
                out[os.path.relpath(fp, path)] = f.read()
    return out


def _siblings(path):
    parent, leaf = os.path.split(os.path.abspath(path))
    return sorted(n for n in os.listdir(parent) if n.startswith(leaf) and n != leaf)


@pytest.fixture
def packed(tmp_path):
    p = str(tmp_path / "pack")
    save_pack_dir(p, {"a": _radix(_const(1.0)), "b": _radix(_const(2.0))},
                  dict(META))
    assert verify_hashes(p) is True
    return p


# ------------------------------------------------------------- the ruler --


def test_a_failed_repack_leaves_the_original_pack_intact_and_verified(packed):
    """The ruler: a second writer over the verified pack
    declares two tensors, writes ONE — with different contents — and
    finish() raises. The original must be byte-for-byte intact and still
    pass verify_hashes, with no staging debris and the lock released. A
    writer that reused the directory would overwrite t0000.npz in place
    under the old meta.json, whose hashes would then not match."""
    before = _tree(packed)
    try:
        w = PackWriter(packed, ["a", "b"], replace=True)
    except TypeError:  # a writer without replace, which reuses the directory
        w = PackWriter(packed, ["a", "b"])
    w.add("a", _radix(_const(3.0)))  # a genuinely changed tensor
    with pytest.raises(ValueError, match="incomplete"):
        w.finish(dict(META))
    after = _tree(packed)
    changed = sorted(k for k in set(before) | set(after) if before.get(k) != after.get(k))
    assert not changed, f"the failed re-pack changed the original pack's files: {changed}"
    assert verify_hashes(packed) is True
    assert _siblings(packed) == []  # no staging dir, no lock file left behind
    # the lock is released: a third writer can take the destination
    PackWriter(packed, ["a"], replace=True).abort()


def test_nothing_reaches_the_destination_before_finish(tmp_path):
    """A writer that dies mid-pack (no finish, no abort) leaves NO
    destination at all — only its staging sibling, which a later writer of
    the same pid reclaims."""
    dest = str(tmp_path / "pack")
    w = PackWriter(dest, ["a", "b"])
    w.add("a", _radix(_const(1.0)))
    assert not os.path.exists(dest)
    assert os.path.isdir(w.staging) and os.path.exists(w.file_path("a"))
    assert not os.path.exists(os.path.join(w.staging, "meta.json"))
    w.abort()  # release the lock so the next writer can take the path
    assert not os.path.exists(w.staging) and _siblings(dest) == []


def test_a_non_empty_destination_is_refused_unless_replace(packed, tmp_path):
    before = _tree(packed)
    with pytest.raises(FileExistsError, match="not empty"):
        PackWriter(packed, ["a"])
    assert _tree(packed) == before and _siblings(packed) == []
    # an existing EMPTY directory is fine
    empty = str(tmp_path / "empty")
    os.makedirs(empty)
    save_pack_dir(empty, {"a": _radix(_const(1.0))}, dict(META))
    assert verify_hashes(empty) is True


def test_replace_stages_fully_then_swaps_and_removes_the_old_pack(packed):
    old = _tree(packed)
    w = PackWriter(packed, ["a", "b", "c"], replace=True)
    for name, v in (("a", 4.0), ("b", 5.0), ("c", 6.0)):
        w.add(name, _radix(_const(v)))
        assert _tree(packed) == old  # the old pack is untouched while staging
    w.finish(dict(META))
    assert verify_hashes(packed) is True
    tensors, meta = load_pack_dir(packed)
    assert set(tensors) == {"a", "b", "c"} and meta["tensorCount"] if "tensorCount" in meta else True
    assert _siblings(packed) == []  # no .staging-*, .replaced-*, or .lock left


def test_competing_writers_are_serialised_by_the_lock(tmp_path):
    dest = str(tmp_path / "pack")
    first = PackWriter(dest, ["a"])
    with pytest.raises(RuntimeError, match="another pack writer holds"):
        PackWriter(dest, ["a"])
    first.add("a", _radix(_const(1.0)))
    first.finish(dict(META))
    assert verify_hashes(dest) is True
    # released with finish: the next writer (replace) gets in
    PackWriter(dest, ["a"], replace=True).abort()


def test_meta_json_is_written_by_temp_file_and_replace(tmp_path, monkeypatch):
    dest = str(tmp_path / "pack")
    replaced = []
    real = os.replace
    monkeypatch.setattr(os, "replace", lambda a, b: replaced.append((a, b)) or real(a, b))
    w = PackWriter(dest, ["a"])
    w.add("a", _radix(_const(1.0)))
    w.finish(dict(META))
    meta_in_staging = os.path.join(w.staging, "meta.json")
    assert (meta_in_staging + ".tmp", meta_in_staging) in replaced
    assert not os.path.exists(os.path.join(dest, "meta.json.tmp"))
    assert json.load(open(os.path.join(dest, "meta.json")))["tensors"] == {"a": "t0000.npz"}


# ------------------------------------------------ through pack_model --


def test_pack_model_refuses_an_existing_pack_and_replaces_on_request(tmp_path):
    pytest.importorskip("torch")
    from drinkme.codec.pack import pack_model
    from tests.test_pack_identity import _toy_checkpoint

    model_dir, pack_dir = str(tmp_path / "m"), str(tmp_path / "p")
    _toy_checkpoint(model_dir, seed=5)
    pack_model(model_dir, None, pack_dir, progress=lambda *_: None)
    first = _tree(pack_dir)
    assert verify_hashes(pack_dir) is True and _siblings(pack_dir) == []
    with pytest.raises(FileExistsError, match="drinkme pack --replace"):
        pack_model(model_dir, None, pack_dir, progress=lambda *_: None)
    assert _tree(pack_dir) == first
    pack_model(model_dir, None, pack_dir, progress=lambda *_: None, replace=True)
    assert verify_hashes(pack_dir) is True and _siblings(pack_dir) == []
    second = _tree(pack_dir)
    assert {k: v for k, v in second.items() if k != "meta.json"} == \
        {k: v for k, v in first.items() if k != "meta.json"}  # deterministic codec: same bytes
    strip = lambda m: {k: v for k, v in json.loads(m).items() if not k.endswith("Seconds")}  # noqa: E731
    assert strip(second["meta.json"]) == strip(first["meta.json"])


def test_serve_front_door_refuses_debris_rather_than_writing_over_it(tmp_path, monkeypatch):
    """A directory with files but no meta.json (a pack that died before its
    commit point on a pre-staging build): ensure_pack's auto-pack must not
    write into it; it names the way out. The real pack_model on a real toy
    checkpoint — only check (a hub call) is stubbed as unavailable."""
    pytest.importorskip("torch")
    from drinkme import check, serve
    from tests.test_pack_identity import _toy_checkpoint

    model_dir = str(tmp_path / "m")
    _toy_checkpoint(model_dir, seed=6)
    debris = tmp_path / "packs" / "debris"
    debris.mkdir(parents=True)
    (debris / "t0000.npz").write_bytes(b"\0" * 8)

    def unavailable(repo, rev):
        raise check.CheckUnavailable("offline")

    monkeypatch.setattr(check, "check", unavailable)
    with pytest.raises(SystemExit, match="not empty"):
        serve.ensure_pack(model_dir, None, str(debris), auto_pack=True)
    assert sorted(os.listdir(debris)) == ["t0000.npz"]
    assert _siblings(str(debris)) == []


def test_cli_pack_takes_a_replace_flag(capsys):
    from drinkme.cli import main

    with pytest.raises(SystemExit) as ei:
        main(["pack", "--help"])
    assert ei.value.code == 0
    assert "--replace" in capsys.readouterr().out
