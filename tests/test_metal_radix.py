"""The radix codec's Metal kernels (metal/gemv_radix, metal/dense_radix) and
the MLX engine's RadixLinear over them.

Every test here dispatches a Metal kernel, so the whole module skips on a
box without Metal. numpy-only (no torch on the Mac lane): the toy tensors
come from codec/radix.pack_array, the research encoder, at every profile
and at 32- and 1024-weight blocks. The full gate — all 65,536 bf16
patterns, the ragged and production shapes, the real Qwen3-8B tensors —
is bench/radix_bitpin_mlx.py; this file pins the claims on toys so a
regression shows up in a pytest run.

The end-to-end test loads a toy Qwen3 radix pack through
load_compressed_mlx on both paths when DRINKME_RADIX_MLX_TOY names a
directory holding `model/` and `pack/` (built on a torch box by
tests/test_serving_engine_mlx.py's recipe at profile sip); it
skips otherwise.
"""

import os

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
if not mx.metal.is_available():
    pytest.skip("every test here dispatches a Metal kernel; no Metal on this machine",
                allow_module_level=True)

from drinkme.codec import radix, registry  # noqa: E402
from drinkme.metal import dense_radix, gemv_radix  # noqa: E402
from drinkme.serving.engine_mlx import RadixLinear, make_module_mlx  # noqa: E402


def _bf16_of_f32(f):
    u = np.ascontiguousarray(f, dtype=np.float32).view(np.uint32)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)


def _f32_of_bf16(u):
    return (np.asarray(u, dtype=np.uint32) << 16).view(np.float32)


def _weights(r, c, seed, outliers=0.01):
    """bf16 bits at a trained model's scale plus a sprinkle of far-out
    magnitudes — real escapes into the terminal stream."""
    rng = np.random.default_rng(seed)
    w = rng.standard_normal((r, c)) * 0.02
    mask = rng.random((r, c)) < outliers
    w[mask] *= rng.choice([1e-6, 1e-3, 30.0, 4000.0], size=int(mask.sum()))
    return _bf16_of_f32(w)


def _pack(bits, profile="sip", block=1024):
    rpk = radix.pack_array(bits, compression_profile=profile, block_size=block)
    assert not rpk.raw, "the toy must ride the compressed streams"
    stored = rpk.palette.nbytes + rpk.offsets.nbytes + rpk.data.nbytes
    return {"rx_palette": rpk.palette, "rx_offsets": rpk.offsets.astype(np.uint32),
            "rx_data": rpk.data.astype(np.uint32), "R": bits.shape[0], "C": bits.shape[1],
            "bpw": 8.0 * stored / bits.size, "codec": "radix", "profile": profile,
            "widths": list(rpk.widths), "block_size": block, "layout": 0, "format_version": 1}


def _dense(res):
    out = dense_radix.decode_dense_resident(res)
    mx.eval(out)
    return np.array(out)


def _gemv(res, x, bias=None, out_dtype=mx.float32):
    y = gemv_radix.gemv_radix_resident(res, x, bias, out_dtype)
    mx.eval(y)
    return np.array(y.view(mx.uint16) if out_dtype == mx.bfloat16 else y)


@pytest.mark.parametrize("profile", ["sip", "balanced", "gulp"])
@pytest.mark.parametrize("block", [32, 1024])
def test_dense_decode_is_bitwise_the_oracle_and_deterministic(profile, block):
    bits = _weights(37, 2100, seed=1)
    p = _pack(bits, profile, block)
    res = gemv_radix.resident(p)
    got = _dense(res)
    assert np.array_equal(got, bits)
    assert np.array_equal(_dense(res), got)
    assert np.array_equal(gemv_radix.decode_reference(res), bits)


@pytest.mark.parametrize("profile", ["sip", "gulp"])
def test_gemv_within_the_float64_bound_bf16_x_equals_f32_x_and_deterministic(profile):
    bits = _weights(96, 3079, seed=2)
    res = gemv_radix.resident(_pack(bits, profile))
    W64 = _f32_of_bf16(bits).astype(np.float64)
    x16 = _bf16_of_f32(np.random.default_rng(3).standard_normal(3079).astype(np.float32))
    x64 = _f32_of_bf16(x16).astype(np.float64)
    oracle = W64 @ x64
    bound = 2e-6 * (np.abs(W64) @ np.abs(x64)) + 1e-7
    xb = mx.array(x16).view(mx.bfloat16)
    y = _gemv(res, xb)
    assert np.all(np.abs(y.astype(np.float64) - oracle) <= bound)
    assert np.array_equal(y, _gemv(res, xb))
    assert np.array_equal(y, _gemv(res, mx.array(_f32_of_bf16(x16))))


@pytest.mark.parametrize("profile", ["sip", "gulp"])
@pytest.mark.parametrize("shape", [(33, 2560), (8, 9728), (5, 3079), (7, 300)])
def test_paired_rows_are_bitwise_the_one_row_kernel(profile, shape, monkeypatch):
    """_GEMV's PAIR (two rows per SIMD group where the last block is at most
    half full: Qwen3-4B's 2560 and 9728) against the one-row kernel: the
    fp32 sums, the fused bias epilogue and the f32-x path bit for bit, an
    odd row count and a single-block row included."""
    r, c = shape
    res = gemv_radix.resident(_pack(_weights(r, c, seed=11), profile))
    rng = np.random.default_rng(12)
    x16 = _bf16_of_f32(rng.standard_normal(c).astype(np.float32))
    xb, xf = mx.array(x16).view(mx.bfloat16), mx.array(_f32_of_bf16(x16))
    bias = mx.array(np.linspace(-0.2, 0.2, r, dtype=np.float32))

    def run():
        return [_gemv(res, xb).view(np.uint32), _gemv(res, xb, bias, mx.bfloat16),
                _gemv(res, xf).view(np.uint32)]

    assert gemv_radix.pairs_tails(res)
    paired = run()
    monkeypatch.setattr(gemv_radix, "PAIR_TAILS", False)
    assert not gemv_radix.pairs_tails(res)
    for got, want in zip(paired, run()):
        assert np.array_equal(got, want)


def test_the_decode_step_call_is_the_general_call_bit_for_bit():
    """RadixLinear's M = 1 bf16 call (gemv_radix.gemv_radix_step: x handed to
    the kernel as is, one graph node) gives gemv_radix_resident's bits in the
    input's leading shape."""
    bits = _weights(40, 2560, seed=13)
    lin = RadixLinear(_pack(bits), path="fused", name="toy")
    x = mx.array(_bf16_of_f32(np.random.default_rng(14).standard_normal((1, 1, 2560))
                              .astype(np.float32))).view(mx.bfloat16)
    y = lin(x)
    want = gemv_radix.gemv_radix_resident(lin._res, x.reshape(-1), None, mx.bfloat16)
    mx.eval(y, want)
    assert y.shape == (1, 1, 40) and y.dtype == mx.bfloat16
    assert np.array_equal(np.array(y.view(mx.uint16)).reshape(-1), np.array(want.view(mx.uint16)))


def test_fused_epilogue_is_one_rounding_of_the_fp32_sum_plus_bias():
    bits = _weights(64, 1024, seed=4)
    res = gemv_radix.resident(_pack(bits))
    x16 = _bf16_of_f32(np.random.default_rng(5).standard_normal(1024).astype(np.float32))
    xb = mx.array(x16).view(mx.bfloat16)
    bias = np.linspace(-0.3, 0.3, 64, dtype=np.float32)
    y32 = _gemv(res, xb)
    fused = _gemv(res, xb, mx.array(bias), mx.bfloat16)
    assert np.array_equal(fused, _bf16_of_f32(y32 + bias))
    plain = _gemv(res, xb, None, mx.bfloat16)
    assert np.array_equal(plain, _bf16_of_f32(y32))


def test_radix_linear_routes_fused_and_reference_to_the_same_bytes():
    bits = _weights(48, 2048, seed=6)
    p = _pack(bits)
    bias = mx.array(_bf16_of_f32(np.linspace(-0.1, 0.1, 48, dtype=np.float32))).view(mx.bfloat16)
    fused = make_module_mlx(p, bias, path="fused", name="toy")
    assert isinstance(fused, RadixLinear) and fused.tensor_codec == registry.RADIX
    ref = RadixLinear(p, bias, path="reference", name="toy")
    assert np.array_equal(ref.decode(), bits)
    rng = np.random.default_rng(7)
    x1 = mx.array(_bf16_of_f32(rng.standard_normal((1, 2048)).astype(np.float32))).view(mx.bfloat16)
    x9 = mx.array(_bf16_of_f32(rng.standard_normal((2, 5, 2048)).astype(np.float32))).view(mx.bfloat16)
    # M=1: the fused kernel's bf16 output; the reference's matmul is close, never claimed bitwise
    y1 = fused(x1)
    r1 = ref(x1)
    mx.eval(y1, r1)
    got = _f32_of_bf16(np.array(y1.view(mx.uint16))).astype(np.float64)
    want = _f32_of_bf16(np.array(r1.view(mx.uint16))).astype(np.float64)
    assert np.all(np.abs(got - want) <= 2 * np.abs(want) * 2 ** -8 + 1e-6)
    # M>1: the dense arm's transient through the same matmul as the reference -> bitwise
    y9 = fused(x9)
    r9 = ref(x9)
    mx.eval(y9, r9)
    assert y9.shape == (2, 5, 48)
    assert np.array_equal(np.array(y9.view(mx.uint16)), np.array(r9.view(mx.uint16)))
    assert fused.resident_bytes() == gemv_radix.resident_bytes(fused._res)


def test_refusals_by_name():
    bits = _weights(8, 64, seed=8)
    with pytest.raises(ValueError, match="radix codec only"):
        gemv_radix.resident({"codec": "window", "R": 8, "C": 64})
    p = _pack(bits, "sip", 32)
    bad = dict(p, widths=[3, 8], block_size=2048)
    with pytest.raises(ValueError, match="block_size"):
        gemv_radix.resident(bad)
    bad = dict(p, widths=[1, 1, 1, 1, 8])
    with pytest.raises(ValueError, match="tiers"):
        gemv_radix.resident(bad)
    with pytest.raises(ValueError, match="RadixLinear serves the radix codec only"):
        RadixLinear({"codec": "raw", "format_version": 1, "layout": 0}, None, path="fused", name="x")
    res = gemv_radix.resident(p)
    with pytest.raises(ValueError, match="shape"):
        gemv_radix.gemv_radix_resident(res, mx.zeros((65,), dtype=mx.bfloat16))
    with pytest.raises(ValueError, match="bf16 or float32"):
        gemv_radix.gemv_radix_resident(res, mx.zeros((64,), dtype=mx.float16))


def test_registry_rows_are_earned():
    assert registry.supported(registry.RADIX, 0, registry.DENSE, registry.METAL)
    assert registry.supported(registry.RADIX, 0, registry.GEMV, registry.METAL)
    assert not registry.supported(registry.RADIX, 0, registry.MC, registry.METAL)
    registry.refuse_unless_metal_dense({"codec": "radix", "format_version": 1, "layout": 0}, "x")


def test_end_to_end_toy_qwen3_radix_pack_fused_vs_reference():
    """load_compressed_mlx over a toy radix pack on both paths: every packed
    Linear a RadixLinear; prefill (M>1, the dense arm) logits BITWISE the
    reference; the decode step (M=1, the fused GEMV) within bf16 tolerance
    with the same argmax; 12 greedy tokens equal."""
    root = os.environ.get("DRINKME_RADIX_MLX_TOY")
    if not root or not os.path.isdir(os.path.join(root, "pack")):
        pytest.skip("DRINKME_RADIX_MLX_TOY unset (a toy Qwen3 radix pack: model/ + pack/)")
    from mlx_lm.models.cache import make_prompt_cache

    from drinkme.arms_mlx import greedy
    from drinkme.serving.engine_mlx import load_compressed_mlx

    model_dir, pack_dir = os.path.join(root, "model"), os.path.join(root, "pack")
    fused = load_compressed_mlx(model_dir, None, pack_dir, path="fused")
    ref = load_compressed_mlx(model_dir, None, pack_dir, path="reference")
    n = sum(isinstance(m, RadixLinear) for _, m in fused.model.named_modules())
    assert n >= 8, n
    ids = fused.tok.encode("hello world the quick brown fox")
    lf = fused.model(mx.array([ids], dtype=mx.int32))[0]
    lr = ref.model(mx.array([ids], dtype=mx.int32))[0]
    mx.eval(lf, lr)
    assert np.array_equal(np.array(lf.view(mx.uint16)), np.array(lr.view(mx.uint16)))
    cf, cr = make_prompt_cache(fused.model), make_prompt_cache(ref.model)
    fused.model(mx.array([ids], dtype=mx.int32), cache=cf)
    ref.model(mx.array([ids], dtype=mx.int32), cache=cr)
    sf = fused.model(mx.array([[ids[-1]]], dtype=mx.int32), cache=cf)[0, -1].astype(mx.float32)
    sr = ref.model(mx.array([[ids[-1]]], dtype=mx.int32), cache=cr)[0, -1].astype(mx.float32)
    mx.eval(sf, sr)
    a, b = np.array(sf), np.array(sr)
    assert np.allclose(a, b, rtol=2e-2, atol=2e-2) and int(a.argmax()) == int(b.argmax())
    assert greedy(fused, ids, 12) == greedy(ref, ids, 12)
