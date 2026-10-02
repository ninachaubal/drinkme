"""iter_pack_dir: parity with load_pack_dir, and per-tensor laziness — the
whole reason it exists is that load_pack_dir's peak is a model in host RAM."""

import os

import numpy as np
import pytest

from drinkme.codec.pack import iter_pack_dir, load_pack_dir, save_pack_dir
from drinkme.codec.radix_pack import _ARRAYS_RADIX as _ARRAYS, _SCALARS_RADIX as _SCALARS

from fixtures import radix_dict


@pytest.fixture
def pack3(tmp_path):
    tensors = {f"layer{i}.w": radix_dict(16, 64, seed=i) for i in range(3)}
    p = str(tmp_path / "pack")
    save_pack_dir(p, tensors, {"hfRepo": "t/t", "revision": None, "dtype": "bf16"})
    return p


def test_iter_matches_load(pack3):
    eager, _ = load_pack_dir(pack3)
    lazy = dict(iter_pack_dir(pack3))
    assert set(lazy) == set(eager)
    for name in eager:
        for k in _ARRAYS:
            assert np.array_equal(lazy[name][k], eager[name][k]), (name, k)
        for k in _SCALARS:
            assert lazy[name][k] == eager[name][k], (name, k)


def test_iter_is_lazy_per_tensor(pack3, monkeypatch):
    opened = []
    real = np.load

    def counting(path, *a, **k):
        opened.append(os.path.basename(path))
        return real(path, *a, **k)

    monkeypatch.setattr(np, "load", counting)
    it = iter_pack_dir(pack3)
    assert opened == []  # a generator: creating it opens NOTHING
    next(it)
    assert len(opened) == 1  # first tensor pulled -> exactly one npz touched
    next(it)
    assert len(opened) == 2  # one more per pull, never the whole pack
    assert len(list(it)) == 1 and len(opened) == 3


def test_iter_version_enforced(pack3):
    import json

    mp = os.path.join(pack3, "meta.json")
    m = json.load(open(mp))
    m["formatVersion"] = 999
    json.dump(m, open(mp, "w"))
    with pytest.raises(ValueError, match="pack format"):
        next(iter_pack_dir(pack3))


def test_iter_arrays_survive_the_closed_npz(pack3):
    # np.load(mmap-free) hands out lazily-decompressed members; iter must
    # materialize them BEFORE the file closes or reads explode later.
    name, p = next(iter_pack_dir(pack3))
    for k in _ARRAYS:
        assert np.asarray(p[k]).sum() is not None  # readable after close
