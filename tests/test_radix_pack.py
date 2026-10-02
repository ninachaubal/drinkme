"""The pack format: the radix codec as THE pack tensor, on the CPU.

The gate for registry row (radix, 0, DENSE, CPU): a toy model packs at each
profile (pack_model's `profile` — the CLI flags --sip / --gulp are a one-line
pass-through, pinned separately), loads back through the pack container's own loaders,
every tensor decodes bit-exact, the serve loader builds the module tree on
the CPU and its logits equal stock's BITWISE, and a tensor radix would
expand is stored RAW (the fallback). Also here: the format-version pin,
the refusal of a pack of any other format version at every door, the
profile default and the DRINKME_COMPRESSION_PROFILE door.
"""

import json
import os
import shutil
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(__file__))
from fixtures import realistic_bf16_bits  # noqa: E402

from drinkme.codec import radix, radix_native, radix_pack as rp  # noqa: E402
from drinkme.codec import registry  # noqa: E402
from drinkme.codec.pack import (FORMAT_VERSION, _arrays_for, iter_pack_dir,  # noqa: E402
                                load_pack_dir, pack_model, refusal_for_version, verify_hashes)
from drinkme.codec.swap import RadixCompressedLinear, RawLinear, make_module  # noqa: E402


def _bits(w: torch.Tensor) -> np.ndarray:
    return w.contiguous().view(torch.int16).numpy().view(np.uint16)


# ------------------------------------------------------------------ toys --


def _toy(tmp_path_factory, name: str, hostile: str | None = None):
    """A 2-layer Llama toy whose seven Linears are all 1024-wide (eligible)
    and whose lm_head is tiny (raw). `hostile` names one Linear whose bits
    get a flat, wide exponent distribution — incompressible for radix,
    finite for the forward — the mixed-pack fallback case."""
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace

    d = tmp_path_factory.mktemp(name)
    model_dir = str(d / "model")
    vocab = {"<unk>": 0, "<pad>": 1, "</s>": 2}
    while len(vocab) < 64:
        vocab[f"tok{len(vocab)}"] = len(vocab)
    backend = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
    backend.pre_tokenizer = Whitespace()
    tok = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="<unk>",
                                  pad_token="<pad>", eos_token="</s>")
    tok.chat_template = "{% for m in messages %}{{ m['content'] }}{% endfor %}"
    tok.save_pretrained(model_dir)
    cfg = LlamaConfig(vocab_size=64, hidden_size=1024, intermediate_size=1024,
                      num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                      max_position_embeddings=256, tie_word_embeddings=False,
                      attention_bias=True, mlp_bias=False, eos_token_id=2, pad_token_id=1)
    torch.manual_seed(0)
    model = LlamaForCausalLM(cfg).to(torch.bfloat16).eval()
    with torch.no_grad():
        for n, m in model.named_modules():
            if isinstance(m, torch.nn.Linear) and min(m.weight.shape) >= 1024:
                m.weight.data.mul_(1 / 64)
                if m.bias is not None:
                    m.bias.data.normal_(0, 0.5)
        if hostile:
            m = model.get_submodule(hostile)
            rng = np.random.default_rng(7)
            R, C = m.weight.shape
            exps = rng.integers(60, 200, (R, C), dtype=np.uint16)
            mant = rng.integers(0, 128, (R, C), dtype=np.uint16)
            sign = rng.integers(0, 2, (R, C), dtype=np.uint16)
            bits = ((sign << 15) | (exps << 7) | mant).astype(np.uint16)
            m.weight.data = torch.from_numpy(bits.view(np.int16).copy()).view(torch.bfloat16)
    model.save_pretrained(model_dir, safe_serialization=True)
    return model_dir, model


@pytest.fixture(scope="module")
def toy(tmp_path_factory):
    model_dir, model = _toy(tmp_path_factory, "radix_toy")
    packs = {}
    for profile in rp.PROFILES:
        out = os.path.join(os.path.dirname(model_dir), f"pack-{profile}")
        pack_model(model_dir, None, out, progress=lambda *_: None, compression_profile=profile)
        packs[profile] = out
    return {"model_dir": model_dir, "model": model, "packs": packs}


@pytest.fixture(scope="module")
def hostile_toy(tmp_path_factory):
    model_dir, model = _toy(tmp_path_factory, "radix_hostile",
                            hostile="model.layers.1.mlp.down_proj")
    out = os.path.join(os.path.dirname(model_dir), "pack-sip")
    pack_model(model_dir, None, out, progress=lambda *_: None, compression_profile="sip")
    return {"model_dir": model_dir, "model": model, "pack": out}


def _source_bits(model) -> dict[str, np.ndarray]:
    return {n: _bits(m.weight.data) for n, m in model.named_modules()
            if isinstance(m, torch.nn.Linear) and min(m.weight.shape) >= 1024}


# --------------------------------------------------------- the pack gate --


@pytest.mark.parametrize("profile", list(rp.PROFILES))
def test_radix_pack_writes_loads_and_decodes_bit_exact(toy, profile):
    pack_dir = toy["packs"][profile]
    meta = json.load(open(os.path.join(pack_dir, "meta.json")))
    assert meta["formatVersion"] == FORMAT_VERSION == 1
    assert meta["profile"] == profile
    assert meta["profileWidths"] == list(rp.widths_of(profile))
    assert meta["blockSize"] == 1024
    assert "method" not in meta  # the profile IS the pack's name; no second word for it
    assert meta["dtype"] == meta["sourceDtype"] == "bf16"
    assert meta["tensorCount"] == 10  # q/o/gate/up/down x 2 layers (k/v are 512-wide: raw)
    assert meta["radixTensorCount"] == 10 and meta["rawFallbackTensorCount"] == 0
    assert meta["rawFallbackBytes"] == 0 and "fp8TensorCount" not in meta
    assert meta["radixEncoder"] in ("native", "numpy") and meta["radixEncodeSeconds"] >= 0
    assert meta["residentBytes"] > meta["packedBytes"] > 0  # resident = packed + raw remainder
    assert "codec" not in meta and "radixProfile" not in meta  # no such keys
    assert verify_hashes(pack_dir) is True  # verified, manifest and all
    src = _source_bits(toy["model"])
    tensors, _ = load_pack_dir(pack_dir)
    assert set(tensors) == set(src)
    n_it = 0
    for name, p in iter_pack_dir(pack_dir):
        assert rp.is_radix_pack(p) and p["format_version"] == 1
        assert set(rp._ARRAYS_RADIX) <= set(p)
        assert registry.of_pack(p) == (registry.RADIX, 0)
        back = rp.decode_back_radix(p)
        assert back.dtype == np.uint16 and np.array_equal(back, src[name])
        rp.verify_pack_radix(p, src[name])
        # both decoders agree, and the research validator accepts the descriptors
        assert np.array_equal(radix.decode(rp.as_radixpack(p)), src[name])
        assert p["bpw"] < 16.0
        n_it += 1
    assert n_it == 10


def test_profiles_order_by_bytes(toy):
    def total(pd):
        return sum(os.path.getsize(os.path.join(pd, f)) for f in os.listdir(pd) if f.endswith(".npz"))
    sip, balanced, gulp = (total(toy["packs"][c]) for c in ("sip", "balanced", "gulp"))
    assert gulp < balanced < sip
    src = _source_bits(toy["model"])
    assert sip < sum(u.nbytes for u in src.values())  # and every profile beats raw bf16


def test_default_profile_is_sip_byte_for_byte(toy, tmp_path, monkeypatch):
    """No --sip/--gulp (and no env) is the sip pack: same npz bytes, same hashes."""
    monkeypatch.delenv("DRINKME_COMPRESSION_PROFILE", raising=False)
    out = str(tmp_path / "pack-default")
    pack_model(toy["model_dir"], None, out, progress=lambda *_: None)
    ma, mb = (json.load(open(os.path.join(d, "meta.json"))) for d in (toy["packs"]["sip"], out))
    assert ma["sha256"] == mb["sha256"]
    assert ma["profile"] == mb["profile"] == "sip"
    assert "method" not in mb


def test_default_pack_dir_suffixes_a_non_default_profile(monkeypatch):
    """The default slug is sip's; a gulp pack of the same model defaults to
    `<org>--<model>-gulp@<rev>` beside it (the hidden profile likewise, by
    its own name). serve/verify/bench resolve without a profile: the sip slug."""
    from drinkme.codec.pack import default_pack_dir
    monkeypatch.setenv("DRINKME_HOME", "/h")
    assert default_pack_dir("Qwen/Qwen3-8B", "b968826d9c46dd60") == "/h/packs/Qwen--Qwen3-8B@b968826d9c46"
    assert default_pack_dir("Qwen/Qwen3-8B", "b968826d9c46dd60", "sip") == "/h/packs/Qwen--Qwen3-8B@b968826d9c46"
    assert default_pack_dir("Qwen/Qwen3-8B", "b968826d9c46dd60", "gulp") == "/h/packs/Qwen--Qwen3-8B-gulp@b968826d9c46"
    assert default_pack_dir("Qwen/Qwen3-8B", None, "balanced") == "/h/packs/Qwen--Qwen3-8B-balanced@main"


def test_env_profile_is_the_hidden_door(toy, tmp_path, monkeypatch):
    """DRINKME_COMPRESSION_PROFILE=balanced selects the profile the CLI does not
    offer; an explicit profile wins over it; an unknown name is refused."""
    monkeypatch.setenv("DRINKME_COMPRESSION_PROFILE", "balanced")
    assert rp.resolve_compression_profile() == "balanced"
    assert rp.resolve_compression_profile("gulp") == "gulp"
    out = str(tmp_path / "pack-env")
    pack_model(toy["model_dir"], None, out, progress=lambda *_: None)
    mb = json.load(open(os.path.join(out, "meta.json")))
    assert mb["profile"] == "balanced" and mb["profileWidths"] == [2, 3, 8]
    assert mb["sha256"] == json.load(open(os.path.join(toy["packs"]["balanced"], "meta.json")))["sha256"]
    monkeypatch.setenv("DRINKME_COMPRESSION_PROFILE", "huge")
    with pytest.raises(ValueError, match="unknown profile 'huge'"):
        rp.resolve_compression_profile()
    with pytest.raises(ValueError, match="unknown profile"):
        pack_model(toy["model_dir"], None, str(tmp_path / "pack-huge"), progress=lambda *_: None)


def test_mixed_pack_stores_the_incompressible_tensor_raw(hostile_toy):
    """The raw fallback: the one tensor radix would expand is stored as its
    bf16 bits, counted, hashed, served by RawLinear, and the pack is still
    smaller than the source overall."""
    pack_dir = hostile_toy["pack"]
    meta = json.load(open(os.path.join(pack_dir, "meta.json")))
    assert meta["radixTensorCount"] == 9
    assert meta["rawFallbackTensorCount"] == 1
    assert meta["tensorCount"] == 10
    src = _source_bits(hostile_toy["model"])
    seen = {}
    for name, p in iter_pack_dir(pack_dir):
        seen[name] = rp.is_radix_pack(p)
        assert np.array_equal(rp.decode_back(p), src[name])
        if not rp.is_radix_pack(p):
            assert rp.is_raw_pack(p) and p["format_version"] == 1 and p["bpw"] == 16.0
            assert set(rp._ARRAYS_RAW) <= set(p) and p["raw_bits"].dtype == np.uint16
            assert registry.of_pack(p) == (registry.RAW, 0)
            rp.verify_pack_raw(p, src[name])
            mod = make_module(p, None, "cpu")
            assert isinstance(mod, RawLinear) and mod.tensor_codec == "raw"
            x = torch.randn(2, 3, 1024, dtype=torch.bfloat16)
            w = torch.from_numpy(src[name].view(np.int16).copy()).view(torch.bfloat16)
            assert torch.equal(mod(x), torch.nn.functional.linear(x, w))
            assert meta["tensorInfo"][name] == {"shape": list(src[name].shape), "dtype": "bf16",
                                                "codec": "raw"}
            fn = meta["tensors"][name]
            assert meta["rawFallbackBytes"] == os.path.getsize(os.path.join(pack_dir, fn))
            assert meta["rawFallbackBytes"] >= src[name].nbytes
    assert seen["model.layers.1.mlp.down_proj"] is False
    assert sum(seen.values()) == 9
    assert verify_hashes(pack_dir) is True
    assert meta["weightedBpw"] < 16.0


def test_raw_fallback_serves_through_the_loader_bitwise(hostile_toy):
    from drinkme.serving.engines import build_compressed_model
    from drinkme.arms import load_stock_streaming

    model, _cfg, _snap, _meta = build_compressed_model(
        hostile_toy["model_dir"], None, hostile_toy["pack"], "cpu")
    kinds = {type(m).__name__ for m in model.modules() if isinstance(m, (RadixCompressedLinear, RawLinear))}
    assert kinds == {"RadixCompressedLinear", "RawLinear"}
    stock = load_stock_streaming(hostile_toy["model_dir"], None, device="cpu")
    torch.manual_seed(2)
    ids = torch.randint(3, 64, (1, 9))
    with torch.inference_mode():
        a = model(ids, use_cache=False).logits
        b = stock(ids, use_cache=False).logits
    assert torch.equal(a.view(torch.int16), b.view(torch.int16))


# -------------------------------------------------- the format refusal --


def test_format_version_is_pinned():
    from drinkme.codec.pack import SUPPORTED_FORMAT_VERSIONS
    assert FORMAT_VERSION == 1 and SUPPORTED_FORMAT_VERSIONS == (1,)


def _pack_at_version(toy, tmp_path, version):
    """The sip toy pack — sound, verified, manifest and all — copied with
    meta.json's formatVersion hand-edited to `version` (removed when None).
    Returns (dir, the edited meta)."""
    d = str(tmp_path / f"pack-v{version}")
    shutil.copytree(toy["packs"]["sip"], d)
    mp = os.path.join(d, "meta.json")
    meta = json.load(open(mp))
    if version is None:
        del meta["formatVersion"]
    else:
        meta["formatVersion"] = version
    json.dump(meta, open(mp, "w"))
    return d, meta


@pytest.mark.parametrize("version", [7, 0, "1", None])
def test_a_pack_of_another_format_version_is_refused_at_every_door(toy, tmp_path, version):
    """A pack whose meta.json declares any format version but the one this
    build reads — or none at all — is refused with ONE generic line at
    every entry: the path, the version seen, the version read, and the
    exact re-pack command (the pack's own repo; --replace since the
    directory is not empty). No other reader, no migration. The pack is
    otherwise sound and verified, manifest and all, so `drinkme verify`'s
    "already verified" early return is what the gate has to come before."""
    from drinkme.codec.pack import _check_format_version, verify_pack
    d, meta = _pack_at_version(toy, tmp_path, version)
    seen = "(none declared)" if version is None else (str(version) if isinstance(version, int) else repr(version))
    want = (f"{d}: unsupported pack format {seen} (this drinkme reads format 1); "
            f"re-pack with `drinkme pack --model {meta['hfRepo']} --replace`")
    assert refusal_for_version(version, d, meta["hfRepo"]) == want
    assert "pack format" in want
    lines = []
    for entry in (lambda: list(iter_pack_dir(d)), lambda: load_pack_dir(d),
                  lambda: _check_format_version(meta, d),
                  lambda: verify_pack(d, progress=lines.append)):
        with pytest.raises(ValueError) as e:
            entry()
        assert str(e.value) == want
    assert lines == []  # not "already verified, manifest and all"
    assert json.load(open(os.path.join(d, "meta.json"))) == meta  # verify left it untouched
    # the serve loader refuses before it hashes the pack or resolves a snapshot
    from drinkme.serving.engines import build_compressed_model
    with pytest.raises(ValueError) as e:
        build_compressed_model(meta["hfRepo"], None, d, "cpu")
    assert str(e.value) == want
    # serve's torch-free front door (ensure_pack on a cache hit): the same
    # line, prefixed with the verb
    from drinkme.serve import ensure_pack
    with pytest.raises(SystemExit) as e:
        ensure_pack(meta["hfRepo"], None, d, auto_pack=True)
    assert str(e.value) == "drinkme serve: " + want


def test_a_pack_under_other_profile_names_is_refused_at_every_door(tmp_path, monkeypatch):
    """A verified format-1 pack whose tensors and meta.json name a profile
    this build does not know (`fast`, say) is not a pack this build reads:
    refused by the name seen — the generic unknown-kind refusal —
    at the meta door (serve, bench, verify, the menu) and per tensor at
    iter_pack_dir; re-packing is the fix. The hidden profile, by its own
    name, still loads."""
    from drinkme.codec.pack import _check_format_version, refusal_for_meta, refusal_for_compression_profile, save_pack_dir
    from drinkme import packs
    U = realistic_bf16_bits(8, 2048, seed=21)
    p = rp.pack_array_radix(U, "sip", encoder="numpy")
    p["profile"] = "fast"  # relabelled before hashing: the hashes are valid, the name is not
    d = str(tmp_path / "packs" / "t--t@main")
    meta = {"hfRepo": "t/t", "revision": None, "dtype": "bf16", "sourceDtype": "bf16",
            "profile": "fast", "profileWidths": [3, 8], "blockSize": 1024}
    save_pack_dir(d, {"a": p}, meta)
    want = (f"{d}: pack profile 'fast' is not one this drinkme packs or serves (sip | gulp); "
            "re-pack with `drinkme pack --model t/t --replace`")
    assert refusal_for_compression_profile("fast", d, "t/t") == want
    full = json.load(open(os.path.join(d, "meta.json")))
    assert refusal_for_meta(full, d) == want
    for entry in (lambda: load_pack_dir(d), lambda: _check_format_version(full, d)):
        with pytest.raises(ValueError) as e:
            entry()
        assert str(e.value) == want
    with pytest.raises(ValueError) as e:
        list(iter_pack_dir(d))
    assert str(e.value) == want  # the meta door, first
    # a tool-written meta that says nothing wrong (`profile` sip) over a
    # tensor that carries the unknown name: the per-tensor check
    d2 = str(tmp_path / "packs" / "t--t-tool@main")
    save_pack_dir(d2, {"a": p}, {"hfRepo": "t/t", "revision": None, "dtype": "bf16", "profile": "sip"})
    with pytest.raises(ValueError, match=r"pack tensor a \(radix\): profile 'fast' is not one this drinkme "
                                         r"packs or serves \(sip \| gulp\); re-pack"):
        list(iter_pack_dir(d2))
    # the menu skips the first with the same line, and never offers it (the
    # tool-written one's meta says nothing wrong: listed, refused per tensor at load)
    monkeypatch.setenv("DRINKME_HOME", str(tmp_path))
    assert [lp.pack_dir for lp in packs.local_packs(str(tmp_path))] == [d2]
    # a key this build does not read is ignored, not refused: a format-1
    # pack written with `method` beside `profile` loads unchanged, whatever
    # the key says; an FP8 pack (a pack cut from an FP8
    # checkpoint) is refused by its dtype with the one line
    assert refusal_for_meta({**full, "profile": "sip", "method": "radix-exponent-sip"}, d) is None
    assert refusal_for_meta({**full, "profile": "sip", "method": "fp8-e4m3-block128"}, d) is None
    assert refusal_for_meta({**full, "profile": "sip", "sourceDtype": "fp8_e4m3"}, d) == \
        f"{full['hfRepo']}: FP8 checkpoints are not supported in this release (bf16 only)"
    assert refusal_for_meta({**full, "profile": "gulp", "method": "gulp"}, d) is None
    assert refusal_for_meta({**full, "profile": "balanced"}, d) is None  # hidden, still read
    assert refusal_for_meta({"hfRepo": "t/t", "formatVersion": 1}, d) is None  # a tool-written meta: its tensors decide
    assert refusal_for_meta({"hfRepo": "t/t", "formatVersion": 2, "profile": "fast"}, d).startswith(
        f"{d}: unsupported pack format 2")  # the version comes first


def test_the_refusal_line_without_a_repo_names_a_placeholder():
    assert refusal_for_version(9, "/x") == (
        "/x: unsupported pack format 9 (this drinkme reads format 1); "
        "re-pack with `drinkme pack --model <repo> --replace`")
    assert refusal_for_version(9).startswith("<pack>: unsupported pack format 9")


def test_the_writer_writes_format_1_only():
    from drinkme.codec.pack import PackWriter
    with pytest.raises(ValueError, match="writes pack format 1 only"):
        PackWriter("/nonexistent/never-created", ["a"], format_version=2)


@pytest.mark.parametrize("profile", list(rp.PROFILES))
def test_serve_loader_builds_radix_modules_and_cpu_logits_equal_stock(toy, profile):
    """(radix, 0, DENSE, CPU): the serve loader's module tree over the pack,
    on the CPU, is bitwise stock — every M through F.linear over the
    decoded weight, which is exactly the source bits."""
    from drinkme.serving.engines import build_compressed_model

    pack_dir = toy["packs"][profile]
    model, _cfg, _snap, _meta = build_compressed_model(toy["model_dir"], None, pack_dir, "cpu")
    mods = [m for m in model.modules() if isinstance(m, RadixCompressedLinear)]
    assert len(mods) == 10
    for m in mods:
        assert m.tensor_codec == registry.RADIX and m.layout == 0
        assert m.p["rx_data"].dtype == torch.uint32
        if len(m.p["widths"]) > 2:
            assert m.p["rx_schedule"].numel() == m.p["NBK"]
        else:
            assert m.p["rx_schedule"].numel() == 0
    # stock through the repo's own streaming loader (the same skeleton, the
    # same rotary buffers) — the in-memory toy was built fp32-then-cast and
    # carries a bf16 inv_freq, which is a construction artifact, not a codec one
    from drinkme.arms import load_stock_streaming

    stock = load_stock_streaming(toy["model_dir"], None, device="cpu")
    torch.manual_seed(1)
    ids = torch.randint(3, 64, (2, 17))
    with torch.inference_mode():
        a = model(ids, use_cache=False).logits
        b = stock(ids, use_cache=False).logits
    assert a.dtype == b.dtype == torch.bfloat16
    assert torch.equal(a.view(torch.int16), b.view(torch.int16))
    # the module-level CPU route, directly: every M is F.linear over the source bits
    m = mods[0]
    x = torch.randn(3, 5, 1024, dtype=torch.bfloat16)
    assert m._route(x) == "cpu"
    src = _source_bits(toy["model"])
    name = [n for n, mm in model.named_modules() if mm is m][0]
    w = torch.from_numpy(src[name].view(np.int16).copy()).view(torch.bfloat16)
    assert torch.equal(m(x), torch.nn.functional.linear(x, w, m.bias))


def test_registry_rows_exist_and_cpu_dense_is_earned_here():
    for op in registry.OPS:
        for b in registry.KERNELS:
            assert (registry.RADIX, 0, op, b) in registry.TABLE
            assert (registry.RAW, 0, op, b) in registry.TABLE
    assert registry.supported(registry.RAW, 0, registry.DENSE, registry.CPU) is True
    assert registry.supported(registry.RAW, 0, registry.GEMV, registry.TRITON) is False
    # the row this file's gate earns; the TRITON rows are bench/radix_*_bitpin.py's
    assert registry.supported(registry.RADIX, 0, registry.DENSE, registry.CPU) is True
    assert registry.supported(registry.RADIX, 0, registry.ROW_SUBSET, registry.TRITON) is False
    # the Metal rows: earned by bench/radix_bitpin_mlx.py on an M4;
    # no multi-column kernel there, M > 1 is the dense arm
    assert registry.supported(registry.RADIX, 0, registry.GEMV, registry.METAL) is True
    assert registry.supported(registry.RADIX, 0, registry.DENSE, registry.METAL) is True
    assert registry.supported(registry.RADIX, 0, registry.MC, registry.METAL) is False
    assert "radix" in registry.describe(registry.RADIX, 0)


# ------------------------------------------------------- the encoder pins --


def test_native_and_numpy_encoders_write_the_same_bytes():
    if not radix_native.available():
        pytest.skip("no C++ compiler")
    U = realistic_bf16_bits(96, 2050, seed=11)  # ragged: a partial last block per row
    for profile in ("sip", "balanced", "gulp"):
        a = rp.pack_array_radix(U, profile, encoder="native")
        b = rp.pack_array_radix(U, profile, encoder="numpy")
        for k in rp._ARRAYS_RADIX:
            assert np.array_equal(a[k], b[k]), (profile, k)
        assert a["bpw"] == b["bpw"]
        assert np.array_equal(rp.decode_back_radix(a, "native"), U)
        assert np.array_equal(rp.decode_back_radix(a, "numpy"), U)


def test_raw_fallback_returns_none_so_the_caller_stores_raw():
    rng = np.random.default_rng(3)
    # a flat exponent spread over the whole FINITE range: incompressible
    U = ((rng.integers(0, 2, (32, 1024), dtype=np.uint16) << 15)
         | (rng.integers(1, 255, (32, 1024), dtype=np.uint16) << 7)
         | rng.integers(0, 128, (32, 1024), dtype=np.uint16))
    assert rp.pack_array_radix(U, "sip", encoder="numpy") is None
    if radix_native.available():
        assert rp.pack_array_radix(U, "sip", encoder="native") is None
    p = rp.pack_weight_radix(torch.from_numpy(U.view(np.int16).copy()).view(torch.bfloat16), "sip",
                             encoder="numpy")
    assert rp.is_raw_pack(p) and not rp.is_radix_pack(p)
    assert np.array_equal(p["raw_bits"], U) and p["bpw"] == 16.0 and (p["R"], p["C"]) == (32, 1024)
    assert np.array_equal(rp.decode_back(p), U)


def _schedule_reference(p: dict) -> np.ndarray:
    """radix_kernel_gpu._prepare_schedule, one block at a time in Python
    over radix._get — the slow twin schedule_np is pinned against."""
    R, C, B = p["R"], p["C"], p["block_size"]
    widths = list(p["widths"])
    data, offsets = p["rx_data"], p["rx_offsets"]
    nb = (C + B - 1) // B
    out = np.zeros(R * nb, np.uint32)
    for b in range(R * nb):
        n = min(B, C - (b % nb) * B)
        pos = int(offsets[b]) + (n * 8 + 31) // 32
        count, header = n, 0
        for level in range(len(widths) - 1):
            w = widths[level]
            words = (count * w + 31) // 32
            if level > 0:
                header |= words << ((level - 1) * 8)
            if level < len(widths) - 2:
                codes, _ = radix._get(data, pos, count, w)
                count = int((codes == (1 << w) - 1).sum())
            pos += words
        out[b] = header
    return out.astype(np.uint16)


def test_schedule_np_matches_the_per_block_reference():
    U = realistic_bf16_bits(40, 2050, seed=5)
    p = rp.pack_array_radix(U, "gulp", encoder="numpy")
    got = rp.schedule_np(p)
    assert got.dtype == np.uint16 and got.shape == (40 * 3,)
    assert np.array_equal(got, _schedule_reference(p))
    # balanced (2,3,8) has one nonterminal tier past the first, so it carries
    # the schedule too — one 8-bit word length per block
    pb = rp.pack_array_radix(U, "balanced", encoder="numpy")
    gotb = rp.schedule_np(pb)
    assert gotb.shape == (40 * 3,) and np.array_equal(gotb, _schedule_reference(pb))
    assert (gotb >> 8).max() == 0  # level 2 does not exist: the high byte stays 0
    assert rp.schedule_np(rp.pack_array_radix(U, "sip", encoder="numpy")) is None


def test_to_device_radix_on_cpu_and_resident_bytes(toy):
    from drinkme.codec.pack import resident_bytes
    from drinkme.codec.swap import to_device_radix

    for profile in rp.PROFILES:
        for name, p in iter_pack_dir(toy["packs"][profile]):
            rt = to_device_radix(p, "cpu")
            stored = sum(p[k].nbytes for k in rp._ARRAYS_RADIX)
            # device bytes = stored + palette padding (+ gulp's 2 B/block
            # schedule) + the reach pad after the streams (the widest block's
            # words: radix.block_word_bounds; tests/test_radix_bounds.py)
            extra = rt["rx_palette"].numel() * 4 - p["rx_palette"].nbytes
            extra += rt["rx_schedule"].numel() * 2
            extra += 4 * radix.block_word_bounds(1024, 7, 8, tuple(p["widths"]))[1]
            assert resident_bytes(rt) == stored + extra
            assert rt["codec"] == "radix" and rt["format_version"] == 1 and rt["layout"] == 0
            mod = make_module(p, None, "cpu")
            assert isinstance(mod, RadixCompressedLinear)
            break


def test_manifest_binds_the_profile(toy):
    """Same tensors, different profile -> different tensorInfo and manifest:
    the widths are in the manifest descriptor, not just the file hash."""
    sip = json.load(open(os.path.join(toy["packs"]["sip"], "meta.json")))
    gulp = json.load(open(os.path.join(toy["packs"]["gulp"], "meta.json")))
    name = next(iter(sip["tensorInfo"]))
    assert sip["tensorInfo"][name]["codec"] == "radix"
    assert sip["tensorInfo"][name]["widths"] == [3, 8]
    assert gulp["tensorInfo"][name]["widths"] == [2, 2, 4, 8]
    assert sip["manifestSha256"] != gulp["manifestSha256"]


def test_unknown_codec_scalar_is_refused_by_name():
    with pytest.raises(ValueError, match="codec 'fp4-magic'"):
        _arrays_for(1, None, name="x", codec="fp4-magic")
    with pytest.raises(ValueError, match="pack format 7 tensors"):
        _arrays_for(7, None, name="x", codec="radix")
    with pytest.raises(ValueError, match="names no codec"):
        _arrays_for(1, None, name="x")
    with pytest.raises(ValueError, match="dtype 'fp4_e2m1'"):
        _arrays_for(1, "fp4_e2m1", name="x")
    assert _arrays_for(1, None, codec="radix") == rp._ARRAYS_RADIX
    assert _arrays_for(1, None, codec="raw") == rp._ARRAYS_RAW
    # an FP8 pack's tensor kind: the one line, at the loader and the constructor
    with pytest.raises(ValueError, match="FP8 checkpoints are not supported in this release"):
        _arrays_for(1, "fp8_e4m3", name="x")
    with pytest.raises(ValueError, match="FP8 checkpoints are not supported in this release"):
        make_module({"format_version": 1, "R": 1, "C": 1, "dtype": "fp8_e4m3"}, None, "cpu")
    with pytest.raises(ValueError, match="not a radix or raw"):
        make_module({"format_version": 1, "R": 1, "C": 1, "codec": "window"}, None, "cpu")


def test_cli_pack_threads_profile_through(monkeypatch, capsys):
    """--gulp reaches pack_model as compression_profile="gulp", --sip as "sip", neither
    as None (= resolve the default, and the env door); --sip --gulp together
    is refused on one line; `--profile` and `--codec` are unknown arguments;
    balanced has no flag. The accelerator bootstrap and model resolution are
    stubbed so no torch/hub is touched."""
    import drinkme.cli as cli
    import drinkme.codec.pack as pk

    seen = {}
    monkeypatch.setattr(pk, "pack_model", lambda repo, rev, out, **kw: seen.update(kw) or out)
    monkeypatch.setattr(cli, "resolve_model", lambda m, verb: ("Qwen/Qwen3-8B", None))
    monkeypatch.setattr(cli, "pin_revision", lambda repo, rev: (repo, rev))  # no hub lookup
    monkeypatch.setenv("DRINKME_NO_AUTO_DEPS", "1")
    assert cli.main(["pack", "--model", "Qwen/Qwen3-8B", "--gulp"]) == 0
    assert seen["compression_profile"] == "gulp"
    assert cli.main(["pack", "--model", "Qwen/Qwen3-8B", "--sip"]) == 0
    assert seen["compression_profile"] == "sip"
    assert cli.main(["pack", "--model", "Qwen/Qwen3-8B"]) == 0
    assert seen["compression_profile"] is None
    capsys.readouterr()
    assert cli.main(["pack", "--model", "Qwen/Qwen3-8B", "--sip", "--gulp"]) == 2
    err = capsys.readouterr().err
    assert err.count("\n") == 1 and err.startswith("drinkme pack: --sip and --gulp")  # ONE line
    assert seen["compression_profile"] is None  # pack_model was not reached
    for bad in (["--profile", "gulp"], ["--profile", "sip"], ["--profile", "balanced"], ["--balanced"],
                ["--codec", "sip"], ["--codec", "window"]):
        with pytest.raises(SystemExit) as ei:
            cli.main(["pack", "--model", "Qwen/Qwen3-8B", *bad])
        assert ei.value.code == 2
        assert "unrecognized arguments" in capsys.readouterr().err


# ------------------------------------- descriptor validation at load (F2) --


def _radix_probe(seed=2):
    U = realistic_bf16_bits(64, 2048, seed=seed)
    p = rp.pack_array_radix(U, "sip", encoder="numpy")
    assert p is not None
    return U, p


def _written(tmp_path, name, d):
    """A hashed pack dir holding one tensor dict, as save_pack_dir writes it
    (the hashes are self-attested, so a corrupt dict hashes clean)."""
    from drinkme.codec.pack import save_pack_dir
    pack = str(tmp_path / name)
    save_pack_dir(pack, {"a": d}, {"hfRepo": "x/y", "revision": None, "dtype": "bf16"})
    assert verify_hashes(pack)
    return pack


def test_a_corrupt_hashed_radix_tensor_is_refused_at_load_not_at_the_first_forward(tmp_path):
    """A
    verified pack — every downloaded pack carries hashes, and they are self-
    attested — whose rx_offsets are x4 and whose rx_data is truncated to
    1/8 passed verify_hashes, was yielded by iter_pack_dir, and make_module
    built a RadixCompressedLinear over it; only the first forward raised
    (on the CPU route) or the kernels read out of bounds (on the device).
    Now iter_pack_dir — the one entry the torch, MTP and MLX loaders share
    — runs radix.validate on every radix tensor, hashed or not, and refuses
    by name before any module is built."""
    from drinkme.codec.pack import validate_tensor
    _, p = _radix_probe()
    bad = dict(p)
    bad["rx_offsets"] = (p["rx_offsets"].astype(np.uint64) * 4).astype(np.uint32)
    bad["rx_data"] = p["rx_data"][: len(p["rx_data"]) // 8]
    pack = _written(tmp_path, "corrupt", bad)
    with pytest.raises(ValueError, match=r"pack tensor a \(radix\).*invalid radix block offsets"):
        list(iter_pack_dir(pack))
    with pytest.raises(ValueError, match=r"pack tensor a \(radix\).*invalid radix block offsets"):
        load_pack_dir(pack)
    # the same refusal on the dict, by the function the loaders call
    with pytest.raises(ValueError, match="invalid radix block offsets"):
        validate_tensor("a", bad)
    validate_tensor("a", p)  # the intact tensor passes
    # a good pack still loads and serves
    good = _written(tmp_path, "good", p)
    (name, t), = list(iter_pack_dir(good))
    assert isinstance(make_module(t, None, "cpu"), RadixCompressedLinear)


@pytest.mark.parametrize("field,value,why", [
    ("widths", [2, 2, 2, 2, 8], "tiers"),      # five tiers: the schedule header holds three
    ("widths", [5, 8], "nonterminal"),          # a 5-bit nonterminal tier: no byte fits it
    ("block_size", 512, "block_size"),          # not the 1024-weight block the kernels walk
    ("R", 63, "invalid radix block offsets"),   # the directory no longer covers R x ceil(C/B)
    ("C", 2049, "invalid radix block offsets"),
])
def test_radix_descriptors_the_served_kernels_cannot_take_are_refused_by_name(tmp_path, field, value, why):
    """The served scheduled decoder's per-block header is a uint16 holding one
    byte per nonterminal tier past the first, so a byte only fits widths <= 4
    and the header only three nonterminal tiers; the kernels walk 1024-weight
    blocks. radix.validate allows 2..8 widths of 1..8 bits and blocks of
    32..4096 (the research class), so the loader adds the served line and
    names it — a crafted `widths: [2,2,2,2,8]` decoded silently wrong before."""
    from drinkme.codec.pack import validate_tensor
    _, p = _radix_probe(seed=3)
    bad = dict(p, **{field: value})
    with pytest.raises(ValueError, match=why):
        validate_tensor("model.layers.0.mlp.up_proj", bad)
    pack = _written(tmp_path, "crafted", bad)
    with pytest.raises(ValueError, match=f"pack tensor a \\(radix\\).*{why}"):
        list(iter_pack_dir(pack))


def test_raw_tensor_shape_is_checked_at_load(tmp_path):
    from drinkme.codec.pack import validate_tensor
    U = realistic_bf16_bits(16, 1024, seed=4)
    good = rp.raw_dict(U)
    validate_tensor("a", good)
    bad = dict(good, raw_bits=U[:8])
    with pytest.raises(ValueError, match=r"raw_bits is \(8, 1024\), not \[R, C\] = \(16, 1024\)"):
        validate_tensor("a", bad)
    pack = _written(tmp_path, "rawbad", bad)
    with pytest.raises(ValueError, match=r"pack tensor a \(raw\).*raw_bits is \(8, 1024\)"):
        list(iter_pack_dir(pack))
    bad = dict(good, raw_bits=U.astype(np.uint32))
    with pytest.raises(ValueError, match="raw_bits dtype uint32, not uint16"):
        validate_tensor("a", bad)


def test_np_load_allocation_is_capped_against_r_times_c(tmp_path):
    """A radix tensor's streams are smaller than its bf16 plane by
    construction (pack_array_radix returns None — the raw fallback — when
    they would not be), so an npz whose stream members are larger than
    2 R C bytes is refused from the zip directory BEFORE np.load
    materialises them: a 100 GB rx_data no longer OOMs the loader. The raw
    fallback's member is exactly 2 R C."""
    import zipfile
    from drinkme.codec.pack import _check_member_sizes
    _, p = _radix_probe(seed=5)
    pack = _written(tmp_path, "cap", p)
    fn = json.load(open(os.path.join(pack, "meta.json")))["tensors"]["a"]
    with zipfile.ZipFile(os.path.join(pack, fn)) as z:
        _check_member_sizes(z, "a", {"R": 64, "C": 2048, "codec": "radix"})  # the real sizes pass
        with pytest.raises(ValueError, match=r"pack tensor a \(radix\): rx_data.npy is .* bytes, more than "
                                             r"the 2 x R x C = 2,048 the tensor's bf16 plane takes"):
            _check_member_sizes(z, "a", {"R": 1, "C": 1024, "codec": "radix"})
        with pytest.raises(ValueError, match=r"declares R x C = 64 x 2048 but the npz holds no raw_bits.npy"):
            _check_member_sizes(z, "a", {"R": 64, "C": 2048, "codec": "raw"})
    raw = _written(tmp_path, "capraw", rp.raw_dict(realistic_bf16_bits(16, 1024, seed=6)))
    fn = json.load(open(os.path.join(raw, "meta.json")))["tensors"]["a"]
    with zipfile.ZipFile(os.path.join(raw, fn)) as z:
        _check_member_sizes(z, "a", {"R": 16, "C": 1024, "codec": "raw"})
        with pytest.raises(ValueError, match=r"pack tensor a \(raw\): raw_bits.npy is .* bytes, more than "
                                             r"the 2 x R x C = 16,384 bytes of bf16 bits it declares"):
            _check_member_sizes(z, "a", {"R": 8, "C": 1024, "codec": "raw"})
    # a lying (huge) shape does not unlock a huge allocation either
    from drinkme.codec.pack import MAX_TENSOR_WEIGHTS, validate_tensor
    with pytest.raises(ValueError, match=r"pack tensor a \(radix\): R x C = 4,294,967,296 x 4 is .* more than"):
        validate_tensor("a", dict(p, R=2 ** 32, C=4))
    with pytest.raises(ValueError, match=r"R, C must be positive ints, got 0, 2048"):
        validate_tensor("a", dict(p, R=0))
    assert MAX_TENSOR_WEIGHTS == 2 ** 33


# --------------------------------- the tensors map stays in the pack (F3) --


def test_tensor_filenames_outside_the_pack_are_refused_by_name(tmp_path):
    """Nothing
    constrained meta.json's `tensors` map, so `"a": "../outside.npz"` and an
    absolute path both passed verify_hashes (which hashed the outside file) and
    loaded through iter_pack_dir — a downloaded pack could make `drinkme
    serve` open, hash and parse any file the user can read. A tensor's file
    is the bare basename the writer emits (t0000.npz) and must resolve
    inside the pack directory; every entry that opens a file refuses
    otherwise, by name."""
    from drinkme.codec.pack import bpw_of_pack_dir, save_pack_dir, verify_pack
    _, p = _radix_probe(seed=7)
    good = str(tmp_path / "good")
    save_pack_dir(good, {"a": p}, {"hfRepo": "x/y", "revision": None, "dtype": "bf16"})
    meta = json.load(open(os.path.join(good, "meta.json")))
    fn = meta["tensors"]["a"]
    assert fn == "t0000.npz"
    outside = str(tmp_path / "outside-t0000.npz")
    os.rename(os.path.join(good, fn), outside)

    def pack_with(name, tensors, sha):
        from drinkme.codec import identity
        d = str(tmp_path / name)
        os.makedirs(d)
        m = dict(meta, tensors=tensors, sha256=sha)
        m["manifestSha256"] = identity.manifest_digest(m)  # a recorded manifest over the bad map
        json.dump(m, open(os.path.join(d, "meta.json"), "w"))
        return d

    entries = (lambda d: list(iter_pack_dir(d)), load_pack_dir, bpw_of_pack_dir,
               lambda d: verify_hashes(d),
               lambda d: verify_pack(d, progress=lambda *_: None))
    digest = meta["sha256"][fn]
    # `../` and an absolute path: refused on the name alone, before any open
    for bad in ("../outside-t0000.npz", outside, "sub/t0000.npz", "t0000.npz/../t0000.npz", "meta.json", 7):
        d = pack_with(f"bad{abs(hash(str(bad)))}", {"a": bad}, {str(bad): digest})
        for entry in entries:
            with pytest.raises(ValueError, match=r"names file .*; a pack file is a bare basename of the form t0000.npz"):
                entry(d)
    # a well-formed name that is a symlink out of the directory: refused on the resolved path
    d = pack_with("link", {"a": "t0000.npz"}, {"t0000.npz": digest})
    os.symlink(outside, os.path.join(d, "t0000.npz"))
    for entry in entries:
        with pytest.raises(ValueError, match=r"resolves to .*outside-t0000.npz, outside the pack directory"):
            entry(d)
    # the honest pack still opens through every entry
    d = pack_with("ok", {"a": "t0000.npz"}, {"t0000.npz": digest})
    os.rename(outside, os.path.join(d, "t0000.npz"))
    assert [n for n, _ in iter_pack_dir(d)] == ["a"]
    assert verify_hashes(d)
    verify_pack(d, progress=lambda *_: None)
    assert bpw_of_pack_dir(d)[0] == pytest.approx(p["bpw"])
