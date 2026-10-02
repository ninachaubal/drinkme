"""bitsPerWeight is weight-weighted, not a mean over TENSORS.

A mean of per-tensor bpw counts a 1024x1024 projection and a 151936x4096
head equally: 1,000 weights at 12 bpw and 100 at 16 bpw are 14 by tensor
and 12.364 by weight. The weight-weighted figure (total encoded payload
bits over total packed weights) is bitsPerWeight, the tensor mean is
meanTensorBitsPerWeight (the record and /v1/models) / meanBpw (pack meta),
and a pack's meta carries weightedBpw beside meanBpw. Population, everywhere:
the packed Linears only. CPU-only."""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

from drinkme import arms, bench
from drinkme.codec import pack as P
from drinkme.publish import validate as V


def test_the_reviews_example_two_tensors_of_different_sizes():
    bpws, numels = [12.0, 16.0], [1000, 100]
    assert float(np.mean(bpws)) == 14.0
    assert round(P.weighted_bpw(bpws, numels), 3) == 12.364
    assert P.weighted_bpw(bpws, numels) == pytest.approx((12000 + 1600) / 1100)


def test_weighted_bpw_refuses_a_mismatch_or_nothing():
    with pytest.raises(ValueError):
        P.weighted_bpw([12.0], [1000, 100])
    with pytest.raises(ValueError):
        P.weighted_bpw([], [])


def test_arms_stats_give_both_and_bench_publishes_the_weighted_one(monkeypatch):
    """arms.bpw_stats off make_compressed-shaped stats (bits exact, numel);
    bench's compression block carries the weighted one as bitsPerWeight and
    the tensor mean as meanTensorBitsPerWeight."""
    stats = [{"bpw": 12.0, "bits": 12 * 1000, "numel": 1000},
             {"bpw": 16.0, "bits": 16 * 100, "numel": 100}]
    assert arms.bpw_stats(stats) == (14.0, 12.364)

    from tests.test_bench_record_shape import _build_record, _fake_raw

    raw = _fake_raw()
    raw["mean_bpw"], raw["weighted_bpw"] = arms.bpw_stats(stats)
    monkeypatch.setattr("tests.test_bench_record_shape._fake_raw", lambda: raw)
    r = _build_record(monkeypatch)
    assert r["compression"]["bitsPerWeight"] == "12.364"
    assert r["compression"]["meanTensorBitsPerWeight"] == "14.0"
    assert V.validate(bench.lexicon_safe(r)) == []


def test_pack_meta_carries_mean_and_weighted_over_the_packed_population(tmp_path):
    """pack_model on a toy: meta.json's weightedBpw is total payload bits
    over total packed weights, recomputed here from each written tensor's
    own scalars; meanBpw is untouched; bpw_of_pack_dir reads both off the
    artifact. The raw remainder (embed_tokens, norms, the 512-wide k/v
    projections) is in neither."""
    from tests.test_pack_identity import _toy_checkpoint

    model_dir, pack_dir = str(tmp_path / "m"), str(tmp_path / "p")
    _toy_checkpoint(model_dir, seed=21)
    P.pack_model(model_dir, None, pack_dir, progress=lambda *_: None)
    meta = json.load(open(os.path.join(pack_dir, "meta.json")))
    tensors, _ = P.load_pack_dir(pack_dir)
    bpws = [t["bpw"] for t in tensors.values()]
    numels = [int(t["R"]) * int(t["C"]) for t in tensors.values()]
    assert len(bpws) == meta["tensorCount"] == meta["radixTensorCount"] + meta["rawFallbackTensorCount"]
    assert meta["meanBpw"] == round(float(np.mean(bpws)), 3)
    assert meta["weightedBpw"] == round(sum(b * n for b, n in zip(bpws, numels)) / sum(numels), 3)
    mean, weighted = P.bpw_of_pack_dir(pack_dir)
    assert round(mean, 3) == meta["meanBpw"] and round(weighted, 3) == meta["weightedBpw"]
    # the population is the packed set: rawTensorCount tensors are outside it
    assert meta["rawTensorCount"] > 0


def test_arm_and_pack_agree_on_both_figures(tmp_path):
    """The bench arm's bpw_stats over the same toy == the pack's meta —
    the same population, the same arithmetic."""
    import functools

    from drinkme.codec.swap import make_compressed, swap_linears
    from tests.test_pack_identity import _toy_checkpoint

    model_dir, pack_dir = str(tmp_path / "m"), str(tmp_path / "p")
    _toy_checkpoint(model_dir, seed=22)
    P.pack_model(model_dir, None, pack_dir, progress=lambda *_: None)
    meta = json.load(open(os.path.join(pack_dir, "meta.json")))
    model = arms.load_cpu(model_dir)
    stats = swap_linears(model, functools.partial(make_compressed, device="cpu"), device="cpu")
    mean, weighted = arms.bpw_stats(stats)
    assert (mean, weighted) == (meta["meanBpw"], meta["weightedBpw"])


def test_lexicon_states_the_population():
    props = V.load_lexicon()["defs"]["compression"]["properties"]
    assert "Weight-weighted bits per weight" in props["bitsPerWeight"]["description"]
    assert "raw remainder" in props["bitsPerWeight"]["description"]
    assert "MTP head" in props["bitsPerWeight"]["description"]
    assert "meanTensorBitsPerWeight" in props
