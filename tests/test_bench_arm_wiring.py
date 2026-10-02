""""The compressed arm" must identify the same artifact across bench and
serve.

Were arms.py's compressed arm to build each Linear through one encoder while
`drinkme pack` (pack_model) wrote another's container, or bench.py to label
every record with one literal method string regardless, GPU timing alone
could not reveal the mislabelled artifact — so the wiring is pinned here,
on the CPU:

  * the ARM'S tensors == the PACK'S tensors, per tensor, for one toy
    checkpoint — codec, profile, every array the loader reads, every scalar;
  * the record's compression.profile is read off the loaded arm and equals
    the profile the pack writer put in meta.json;
  * pack_model and the arm resolve one profile default (and one env door).
"""

from __future__ import annotations

import ast
import functools
import json
import os
import pathlib

import numpy as np
import pytest
import torch

from drinkme import arms
from drinkme.codec import pack as P
from drinkme.codec.swap import CompressedLinear, make_compressed, swap_linears

ROOT = pathlib.Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def toy(tmp_path_factory):
    """One toy checkpoint packed by the real pack_model at its defaults —
    what `drinkme pack` writes — beside the same checkpoint's CPU model tree
    for the arm to swap."""
    from tests.test_pack_identity import _toy_checkpoint

    d = tmp_path_factory.mktemp("wiring")
    model_dir, pack_dir = str(d / "m"), str(d / "p")
    _toy_checkpoint(model_dir, seed=11)
    P.pack_model(model_dir, None, pack_dir, progress=lambda *_: None)
    return model_dir, pack_dir


@pytest.fixture(scope="module")
def arm(toy):
    """The bench's compressed arm over the toy: the real swap_linears +
    make_compressed on the CPU, with every raw pack dict it built collected
    beside the stat swap_linears names it with (same order, by construction)."""
    model = arms.load_cpu(toy[0])
    collected = []
    stats = swap_linears(model, functools.partial(make_compressed, device="cpu",
                                                  collect=collected.append), device="cpu")
    packs = {s["name"]: p for s, p in zip(stats, collected)}
    modules = {n: m for n, m in model.named_modules() if isinstance(m, CompressedLinear)}
    return model, stats, packs, modules


def test_arm_tensors_are_the_packs_tensors(toy, arm):
    """The wiring assertion: for every eligible Linear, the tensor the bench
    arm builds is byte-for-byte the tensor `drinkme pack` wrote — the same
    codec and profile, the same arrays the loader reads for that codec, the
    same scalars, the same descriptor the manifest records."""
    from drinkme.codec import radix_pack as rp

    _model, _stats, packs, modules = arm
    written, meta = P.load_pack_dir(toy[1])
    assert set(packs) == set(written), (sorted(packs), sorted(written))
    assert meta["formatVersion"] == P.FORMAT_VERSION == 1
    assert meta["profile"] == rp.DEFAULT_PROFILE == "sip"
    for name in sorted(written):
        a, w = packs[name], written[name]
        assert a["codec"] == w["codec"] == meta["tensorInfo"][name]["codec"] == "radix", name
        assert a["profile"] == w["profile"] == meta["profile"], name
        for key in P._arrays_for(w["format_version"], w.get("dtype"), name=name, codec=w["codec"]):
            assert np.array_equal(np.asarray(a[key]), np.asarray(w[key])), (name, key)
        for key in rp._SCALARS_RADIX:
            assert a[key] == w[key], (name, key)
        # and the MODULE the arm times is the loader's class for the codec
        assert modules[name].tensor_codec == "radix" and modules[name].p["rx_launch"]["family"] == "sip"


def test_arm_stats_match_the_writers_counts(toy, arm):
    _model, stats, _packs, _modules = arm
    _written, meta = P.load_pack_dir(toy[1])
    assert sum(s["codec"] == "radix" for s in stats) == meta["radixTensorCount"]
    assert sum(s["codec"] == "raw" for s in stats) == meta["rawFallbackTensorCount"] == 0
    assert {s["profile"] for s in stats} == {meta["profile"]}
    assert all(s["dtype"] is None for s in stats)  # the bf16 codec: no dtype scalar


def test_arm_forward_is_bitwise_stock_on_the_cpu_reference_route(toy, arm):
    """The CPU route decodes the served-format tensor back and runs stock's
    own F.linear — so on the CPU the swapped tree is bitwise the stock tree.
    Proves the tensors the arm builds decode, not just compare."""
    model, _stats, _packs, _modules = arm
    stock = arms.load_cpu(toy[0])
    ids = torch.tensor([[3, 4, 5, 6]])
    with torch.no_grad():
        a = model(ids).logits
        b = stock(ids).logits
    assert torch.equal(a, b)


def test_record_profile_is_read_off_the_arm_and_equals_the_writers(toy, arm):
    _model, stats, _packs, _modules = arm
    with open(os.path.join(toy[1], "meta.json")) as f:
        meta = json.load(f)
    assert arms.arm_compression_profile(stats) == meta["profile"] == "sip"
    assert "method" not in meta  # the profile is the pack's one name


def test_arm_profile_refuses_a_mixed_arm():
    with pytest.raises(ValueError, match="mixes encodings"):
        arms.arm_compression_profile([{"codec": "radix", "profile": "sip", "dtype": None}, {"codec": "window", "dtype": None}])
    with pytest.raises(ValueError, match="2 profiles"):
        arms.arm_compression_profile([{"codec": "radix", "profile": "sip", "dtype": None},
                          {"codec": "radix", "profile": "gulp", "dtype": None}])
    with pytest.raises(ValueError, match="nothing was timed"):
        arms.arm_compression_profile([])
    # a raw fallback beside the profile's tensors is one artifact, named by the profile
    assert arms.arm_compression_profile([{"codec": "radix", "profile": "gulp", "dtype": None},
                             {"codec": "raw", "profile": None, "dtype": None}]) == "gulp"
    with pytest.raises(ValueError, match="dtype this build does not name"):
        arms.arm_compression_profile([{"format_version": 1, "dtype": "fp8_e4m3"}])  # an FP8 pack's tensor kind


def test_bench_takes_the_profile_from_raw_and_never_hardcodes_one(monkeypatch):
    """bench.py carries no profile literal: the record's string is whatever
    the arm reported (here a sentinel no writer would ever produce, so a
    literal anywhere in bench.py would show up as a mismatch)."""
    from tests.test_bench_record_shape import _build_record, _fake_raw

    raw = _fake_raw()
    raw["compression_profile"] = "sentinel-profile-from-the-arm"
    monkeypatch.setattr("tests.test_bench_record_shape._fake_raw", lambda: raw)
    r = _build_record(monkeypatch)
    assert r["compression"]["profile"] == "sentinel-profile-from-the-arm"
    src = (ROOT / "src" / "drinkme" / "bench.py").read_text()
    literals = {n.value for n in ast.walk(ast.parse(src))
                if isinstance(n, ast.Constant) and isinstance(n.value, str)}
    assert not any(v in ("sip", "balanced", "gulp") for v in literals)


def test_pack_model_and_the_arm_share_one_profile_default(monkeypatch):
    """pack_model's compression_profile default (None -> radix_pack.resolve_compression_profile) IS
    what the arm packs under (pack_weight_served -> the same resolver), so a
    future edit to one without the other fails here rather than in a GPU
    run; and the env door moves both together."""
    import inspect

    from drinkme.codec import radix_pack as rp

    sig = inspect.signature(P.pack_model)
    assert sig.parameters["compression_profile"].default is None
    assert "codec" not in sig.parameters and "format_version" not in sig.parameters
    monkeypatch.delenv("DRINKME_COMPRESSION_PROFILE", raising=False)
    monkeypatch.setenv("DRINKME_RADIX_ENCODER", "numpy")  # this test's only pack_weight_served
                                                          # caller with no explicit encoder=
    w = torch.randn(1024, 1024).to(torch.bfloat16) / 64
    assert P.pack_weight_served(w)["profile"] == rp.DEFAULT_PROFILE == "sip"
    monkeypatch.setenv("DRINKME_COMPRESSION_PROFILE", "gulp")
    assert P.pack_weight_served(w)["profile"] == "gulp"


def test_the_profile_is_the_packs_one_name(toy, arm):
    """No `method` anywhere: the pack's meta.json, the mtp/ sub-pack's, the
    arm and the record all say `profile`, and the lexicon closes it to the
    two public profiles."""
    from drinkme.codec import pack as P_
    from drinkme.publish import validate as V

    assert not hasattr(P_, "method_name")
    _model, stats, _packs, _modules = arm
    assert arms.arm_compression_profile(stats) == "sip"
    lex = V.load_lexicon()["defs"]["compression"]
    assert lex["required"] == ["profile", "bitsPerWeight"] and "method" not in lex["properties"]
    assert lex["properties"]["profile"]["enum"] == ["sip", "gulp"]
