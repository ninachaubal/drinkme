"""codec/swap.NarrowLinear — the sub-threshold raw Linears' Triton GEMV
(codec/radix_ops.gemv_narrow).

CPU: what `wants` admits (bf16, on the accelerator, under the codec's row
threshold, R <= MAX_ROWS, C >= MIN_COLS), `adopt` sharing the parameters,
the CPU/M > MC_MAX fall-through being F.linear bit for bit, install_narrow
a no-op off the accelerator, gemv_narrow refusing M outside 1..MC_MAX.
GPU (gated): the kernel against F.linear and an fp64 reference on a
48 x 5120 weight, row m of an M-row call == the M=1 call bit for bit at
every M = 1..MC_MAX on the three narrow shapes of the menu, bf16 x ==
fp32 x bit for bit (the in-kernel promotion is exact), the bias added in
fp32 before the one rounding (swap.py's epilogue rule), M > MC_MAX ==
F.linear, and install_narrow on a device model adopting exactly the
sub-threshold Linears with the same Parameters.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from drinkme.codec.ops import MC_MAX
from drinkme.codec.swap import NarrowLinear, install_narrow

gpu_gate = pytest.mark.skipif(not torch.cuda.is_available(),
                              reason="the narrow GEMV is triton; needs a device")


def _lin(r, c, bias=False, dtype=torch.bfloat16, device="cpu", seed=0):
    torch.manual_seed(seed)
    lin = torch.nn.Linear(c, r, bias=bias).to(dtype)
    with torch.no_grad():
        lin.weight.mul_(0.05)
        if bias:
            lin.bias.mul_(0.5)
    return lin.to(device)


def test_wants_on_cpu_is_false_and_adopt_shares_parameters():
    lin = _lin(48, 5120, bias=True)
    assert not NarrowLinear.wants(lin)  # CPU: the kernel is triton
    n = NarrowLinear.adopt(lin)
    assert isinstance(n, NarrowLinear) and isinstance(n, torch.nn.Linear)
    assert n.weight is lin.weight and n.bias is lin.bias
    assert n.in_features == 5120 and n.out_features == 48
    x = torch.randn(3, 5120).to(torch.bfloat16)
    assert torch.equal(n(x), lin(x))  # the fall-through IS F.linear
    x = torch.randn(2, 4, 5120).to(torch.bfloat16)
    assert torch.equal(n(x), lin(x)) and n(x).shape == (2, 4, 48)


def test_install_narrow_is_a_no_op_off_the_accelerator():
    m = torch.nn.Sequential(_lin(48, 5120))
    assert install_narrow(m, "cpu") == []
    assert type(m[0]) is torch.nn.Linear


@pytest.mark.parametrize("m", [0, MC_MAX + 1])
def test_gemv_narrow_refuses_m_outside_one_to_mc_max(m):
    pytest.importorskip("triton")
    from drinkme.codec.radix_ops import gemv_narrow

    w = torch.zeros(48, 5120, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="outside 1"):
        gemv_narrow(w, torch.zeros(m, 5120, dtype=torch.bfloat16), None, NarrowLinear.BLOCK,
                    NarrowLinear.NUM_WARPS)


@gpu_gate
def test_wants_admits_only_the_sub_threshold_bf16_shape():
    assert NarrowLinear.wants(_lin(48, 5120, device="cuda"))
    assert NarrowLinear.wants(_lin(48, 5120, bias=True, device="cuda"))
    assert not NarrowLinear.wants(_lin(1024, 5120, device="cuda"))   # the codec's (eligible)
    assert not NarrowLinear.wants(_lin(48, 5120, dtype=torch.float32, device="cuda"))
    assert not NarrowLinear.wants(_lin(48, 512, device="cuda"))     # C < MIN_COLS
    assert not NarrowLinear.wants(_lin(48, 5122, device="cuda"))    # C % 4
    assert not NarrowLinear.wants(NarrowLinear.adopt(_lin(48, 5120, device="cuda")))  # once


@gpu_gate
@pytest.mark.parametrize("bias", [False, True])
def test_gpu_kernel_vs_flinear_and_fp64(bias):
    lin = _lin(48, 5120, bias=bias, device="cuda")
    n = NarrowLinear.adopt(lin)
    torch.manual_seed(1)
    for m in (1, 5, MC_MAX):
        x = (torch.randn(m, 5120, device="cuda") * 0.5).to(torch.bfloat16)
        y = n(x)
        ref = F.linear(x, lin.weight, lin.bias)
        ref64 = x.double() @ lin.weight.double().T + (lin.bias.double() if bias else 0)
        assert y.shape == ref.shape and y.dtype == torch.bfloat16
        # a different fp32 reduction order: bf16-ulp-sized disagreement with
        # the BLAS path, and no farther from the truth than it is
        d = (y.float() - ref.float()).abs().max().item()
        assert d < 2e-2, (m, d)
        assert (y.double() - ref64).abs().max().item() <= (ref.double() - ref64).abs().max().item() * 2 + 1e-3


@gpu_gate
@pytest.mark.parametrize("shape", [(48, 5120), (32, 4096), (256, 6656)])  # the menu's narrow shapes
def test_gpu_rows_of_an_m_call_equal_the_m1_call_bitwise(shape):
    r, c = shape
    n = NarrowLinear.adopt(_lin(r, c, bias=True, device="cuda"))
    torch.manual_seed(2)
    x = (torch.randn(MC_MAX, c, device="cuda") * 0.5).to(torch.bfloat16)
    ones = [n(x[j:j + 1]) for j in range(MC_MAX)]
    for m in range(1, MC_MAX + 1):
        ym = n(x[:m])
        for j in range(m):
            assert torch.equal(ym[j:j + 1], ones[j]), (shape, m, j)


@gpu_gate
def test_gpu_bf16_x_equals_fp32_x_bitwise():
    """The kernel promotes x in-register; bf16 -> fp32 is exact, so a bf16 x
    and its fp32 widening give the same bits (and the same output dtype
    as the input's, as F.linear would)."""
    n = NarrowLinear.adopt(_lin(48, 5120, device="cuda"))
    x = (torch.randn(3, 5120, device="cuda") * 0.5).to(torch.bfloat16)
    y16, y32 = n(x), n(x.float())
    assert y16.dtype == torch.bfloat16 and y32.dtype == torch.float32
    assert torch.equal(y16.float(), y32.to(torch.bfloat16).float())


@gpu_gate
def test_gpu_bias_is_added_in_fp32_before_the_one_rounding():
    """swap.py's epilogue rule: acc 1 + 2^-9 (not a bf16 value) + bias -1
    must give 2^-9, not 0 — the residual survives only if the bias meets
    the fp32 accumulator before the round."""
    lin = torch.nn.Linear(1024, 2, bias=True).to(torch.bfloat16).to("cuda")
    with torch.no_grad():
        lin.weight.zero_()
        lin.weight[0, 0] = 1.0
        lin.weight[0, 1] = 1.0
        lin.bias.zero_()
        lin.bias[0] = -1.0
    n = NarrowLinear.adopt(lin)
    x = torch.zeros(1, 1024, dtype=torch.bfloat16, device="cuda")
    x[0, 0] = 1.0
    x[0, 1] = 2.0 ** -9
    assert n(x)[0, 0].item() == 2.0 ** -9
    assert F.linear(x, lin.weight, lin.bias)[0, 0].item() == 2.0 ** -9  # stock's fused epilogue agrees


@gpu_gate
def test_gpu_above_mc_max_is_flinear_bitwise():
    lin = _lin(48, 5120, bias=True, device="cuda")
    n = NarrowLinear.adopt(lin)
    x = (torch.randn(MC_MAX + 1, 5120, device="cuda") * 0.5).to(torch.bfloat16)
    assert torch.equal(n(x), F.linear(x, lin.weight, lin.bias))
    x = (torch.randn(2, 40, 5120, device="cuda") * 0.5).to(torch.bfloat16)
    assert torch.equal(n(x), F.linear(x, lin.weight, lin.bias))


@gpu_gate
def test_gpu_install_narrow_adopts_exactly_the_sub_threshold_linears():
    class Block(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.big = _lin(1024, 5120)          # the codec's
            self.a = _lin(48, 5120)              # narrow
            self.b = _lin(48, 5120, bias=True)   # narrow, biased
            self.short = _lin(48, 512)           # C too short
            self.f32 = _lin(48, 5120, dtype=torch.float32)

    m = torch.nn.Sequential(Block(), Block()).to("cuda")
    w_a = m[0].a.weight
    done = install_narrow(m, "cuda")
    assert sorted(n for n, _ in done) == ["0.a", "0.b", "1.a", "1.b"]
    assert all(shape == (48, 5120) for _, shape in done)
    assert isinstance(m[0].a, NarrowLinear) and m[0].a.weight is w_a
    assert isinstance(m[1].b, NarrowLinear) and m[1].b.bias is not None
    assert type(m[0].big) is torch.nn.Linear and type(m[0].short) is torch.nn.Linear
    assert type(m[0].f32) is torch.nn.Linear
    assert install_narrow(m, "cuda") == []  # idempotent
