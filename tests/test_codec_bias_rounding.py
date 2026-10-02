"""A biased packed Linear rounds ONCE, on every arm.

Stock's `F.linear(x, W, b)` is one fused epilogue — fp32 accumulate, add the
bias, cast once. Casting first and adding the bias to the bf16 result
would be a second rounding. The construction that shows it is a
cancellation: weight columns 0 and 1 are 1.0, the rest 0; the input row is
x[0] = 1, x[1] = 1/256; bias -1. The accumulator is exactly 1.00390625, and
bias then cancels it to 0.00390625 (2^-8, a bf16 value). Rounding 1.00390625
to bf16 FIRST gives 1.0 — the residual is gone before the bias arrives, and
the arm answers 0.0 where stock answers 0.00390625. Every output element,
every packing, every arm: 9,216 of 9,216 measured on the dense
route; the GEMV and multi-column routes and the bench's bf16 twin did the same.

The oracle is stock's biased F.linear ON THE DEVICE with the same bf16
inputs — not the CPU fallback (which calls stock's F.linear itself and was
never wrong) and not another compressed arm (which had the same bug). The
CPU ruler for the dense arm lives in test_swap_prefill_dense.py; these are
the device rulers and skip cleanly without an accelerator.
"""

import numpy as np
import pytest
import torch

from drinkme.codec import radix_pack as rp
from drinkme.codec.ops import MC_MAX
from drinkme.codec.swap import CompressedLinear, make_module, make_twin
from drinkme.serving.draft_vocab import make_projection

gpu_gate = pytest.mark.skipif(not torch.cuda.is_available(),
                              reason="the kernels are triton; needs a device")

R = C = 1024
# the same weight packed every way a pack can hold a bf16 tensor:
# the three profiles and the raw fallback
FOUR_WAYS = ("sip", "balanced", "gulp", "raw")


def _bits(w):
    return w.cpu().view(torch.int16).numpy().view(np.uint16)


def _module(way, w, b):
    """The served module for `w` packed as `way` (make_module, the loader's
    own constructor): RadixCompressedLinear at the profile, or RawLinear."""
    u = _bits(w)
    if way == "raw":
        return make_module(rp.raw_dict(u), b, "cuda")
    pk = rp.pack_array_radix(u, way)
    assert pk is not None
    return make_module(pk, b, "cuda")


def _cancellation():
    w = torch.zeros(R, C, dtype=torch.bfloat16)
    w[:, :2] = 1
    b = torch.full((R,), -1.0, device="cuda", dtype=torch.bfloat16)
    return w, b


def _rows(n):
    x = torch.zeros(n, C, device="cuda", dtype=torch.bfloat16)
    x[:, 0] = 1
    x[:, 1] = 1 / 256
    return x


def _stock(x, w, b):
    return torch.nn.functional.linear(x, w.cuda(), b)


@gpu_gate
def test_the_oracle_is_a_single_rounding():
    """What this file measures against: stock's fused biased call keeps the
    2^-8 residual; the post-add composition loses it. If a torch build ever
    made these equal the rest of this file would be measuring nothing."""
    w, b = _cancellation()
    x = _rows(9)
    fused = _stock(x, w, b)
    assert fused[0, 0].item() == 0.00390625
    twice = torch.nn.functional.linear(x, w.cuda()) + b
    assert twice[0, 0].item() == 0.0
    assert not torch.equal(fused, twice)


@gpu_gate
@pytest.mark.parametrize("way", FOUR_WAYS)
@pytest.mark.parametrize("rows,route", [(1, "gemv"), (4, "mc"), (9, "dense")])
def test_every_arm_matches_stocks_biased_linear(way, rows, route):
    """Nine rows is the dense arm, one row the GEMV, four the multi-column —
    asserted, so a threshold change cannot quietly retarget the test. With a
    second rounding every one of these returns 0.0 on every element."""
    w, b = _cancellation()
    m = _module(way, w, b)
    x = _rows(rows)
    assert m._route(x) == route
    got, want = m(x), _stock(x, w, b)
    assert want[0, 0].item() == 0.00390625
    assert torch.equal(got, want), (
        f"{way} {route}: {int((got != want).sum())} of "
        f"{want.numel()} differ; got {got[0, 0].item()} want {want[0, 0].item()}")


@gpu_gate
@pytest.mark.parametrize("way", FOUR_WAYS)
def test_the_row_loop_arm_matches_too(way, monkeypatch):
    """M > MC_MAX with the dense arm off (DRINKME_PREFILL_DENSE_MIN=0) is the
    one-GEMV-per-row loop; it shares the epilogue and must share the fix."""
    from drinkme.codec import swap

    monkeypatch.setenv("DRINKME_PREFILL_DENSE_MIN", "0")
    monkeypatch.setattr(swap, "_DENSE_MIN", None)
    w, b = _cancellation()
    m = _module(way, w, b)
    x = _rows(MC_MAX + 1)
    assert m._route(x) == "loop"
    assert torch.equal(m(x), _stock(x, w, b))


@gpu_gate
@pytest.mark.parametrize("way", ("sip", "balanced", "gulp"))
@pytest.mark.parametrize("rows", (1, MC_MAX))
def test_the_order_matched_twin_matches_stock(way, rows):
    """bench divides the compressed arm by the twin (swap.RadixTwinLinear,
    the same kernels over the raw weight); a twin that rounded twice would
    charge the codec for the twin's bug."""
    w, b = _cancellation()
    tw, _ = make_twin(w, b, "cuda", codec="radix", widths=rp.widths_of(way))
    x = _rows(rows)
    assert torch.equal(tw(x), _stock(x, w, b))


@gpu_gate
@pytest.mark.parametrize("way", ("sip", "raw"))
def test_a_biased_draft_subset_is_the_full_head_since_no_codec_has_the_kernel(way):
    """The reduced-vocab draft projection over a packed head: the registry
    says no tensor kind has a row-subset kernel, so make_projection hands
    back (None, why) and the draft keeps the full head — whose biased rows
    are stock's."""
    w, b = _cancellation()
    m = _module(way, w, b)
    proj, note = make_projection(m, torch.tensor([0, 2], dtype=torch.long))
    assert proj is None and "no triton kernel" in note, note
    x = _rows(1)
    assert torch.equal(m(x)[:, [0, 2]], _stock(x, w, b)[:, [0, 2]])


@gpu_gate
@pytest.mark.parametrize("way", ("sip", "balanced", "gulp"))
def test_multi_column_matches_the_m1_kernel_with_a_bias_within_the_bound(way):
    """The mc contract — each column is the M=1 answer within the float64
    oracle's bound (bench/radix_mc_bitpin.py gates it; the two arms reduce
    in different orders, so bitwise is recorded there, not required) — has
    to hold THROUGH the bias epilogue, on a realistic weight, for every M
    the arm serves. The epilogue itself (bias in fp32, one rounding) is the
    same arithmetic on both arms."""
    from fixtures import realistic_bf16_bits

    u = realistic_bf16_bits(R, C, 11, spread=True)
    w = torch.from_numpy(u.view(np.int16).copy()).view(torch.bfloat16)
    torch.manual_seed(11)
    b = torch.randn(R, dtype=torch.bfloat16).cuda()
    m = _module(way, w, b)
    x = torch.randn(MC_MAX, C, device="cuda", dtype=torch.bfloat16)
    ones = torch.cat([m(x[i:i + 1]) for i in range(MC_MAX)])
    ref = _stock(x, w, b)
    for n in range(2, MC_MAX + 1):
        assert m._route(x[:n]) == "mc"
        got = m(x[:n])
        assert torch.allclose(got.float(), ones[:n].float(), rtol=2e-2, atol=2e-2), (way, n)
        assert torch.allclose(got.float(), ref[:n].float(), rtol=2e-2, atol=2e-2), (way, n)


# ------------------------------------------------ through the real loader --
#
# test_serving_engines' toy declares attention_bias=True, but HF's
# _init_weights zeroes every Linear bias, so that pack never exercised the
# composition at all — its stock/compressed logits are bitwise on the
# device whichever order the epilogue rounds in. This toy is the same recipe with the biases
# drawn nonzero, so the checkpoint bias that arrives AFTER the swap
# (engines.load_compressed's "bias, arrived after swap" branch) goes through
# the fixed epilogue on a real forward.


@pytest.fixture(scope="module")
def biased_toy(tmp_path_factory):
    """(model_dir, pack_dir): the two-layer Llama toy of test_radix_pack.py
    (NONZERO q/o biases on the codec-eligible projections), packed."""
    import json
    import os

    from drinkme.codec.pack import pack_model
    from drinkme.codec.swap import eligible_linears
    from tests.test_radix_pack import _toy

    model_dir, model = _toy(tmp_path_factory, "biased_toy")
    n_biased = sum(1 for _, _, _, child in eligible_linears(model) if child.bias is not None)
    assert n_biased == 4  # q_proj and o_proj, two layers
    pack_dir = os.path.join(os.path.dirname(model_dir), "pack")
    pack_model(model_dir, None, pack_dir, progress=lambda *_: None)
    assert json.load(open(os.path.join(pack_dir, "meta.json")))["tensorCount"] == 10
    return model_dir, pack_dir


@gpu_gate
def test_the_loaders_biased_prefill_is_bitwise_stock_on_the_device(biased_toy):
    """Stock arm vs compressed arm through load_stock/load_compressed on the
    accelerator, a 12-token prompt (the dense arm on every packed Linear):
    the logits are BITWISE — the packed weight is stock's bytes, the matmul
    is stock's F.linear, and the bias goes into that call the way stock
    passes it. A second rounding on q/o breaks this on the first layer and
    the difference compounds through the second."""
    from drinkme.serving.engines import load_compressed, load_stock

    model_dir, pack_dir = biased_toy
    stock = load_stock(model_dir, device="cuda", ctx=256)
    comp = load_compressed(model_dir, None, pack_dir, device="cuda", ctx=256)
    packed_bias = [m for m in comp.model.modules()
                   if isinstance(m, CompressedLinear) and m.bias is not None]
    assert len(packed_bias) == 4 and all(m.bias.abs().sum() > 0 for m in packed_bias)
    ids = torch.arange(3, 15, device="cuda")[None]
    with torch.inference_mode():
        want = stock.model(ids, use_cache=False).logits
        got = comp.model(ids, use_cache=False).logits
    assert torch.equal(got, want), (got.float() - want.float()).abs().max().item()
