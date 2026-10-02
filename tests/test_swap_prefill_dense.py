"""The M>1 branches: routing + dense numerics, all CPU, no GPU.

The GEMV arms are triton and need a GPU (bench/radix_mc_bitpin.py owns the
multi-column arm's numerics, per column against the oracle, on real pack
tensors); what IS testable here is everything the router decides and the
dense arm itself. CompressedLinear._route is split from forward for exactly
this: the threshold arithmetic (which arm a given M picks, and the
DRINKME_PREFILL_DENSE_MIN instrument) asserts directly, and forcing the
router to "dense" runs the REAL forward() dense branch on CPU — the CPU
reference decoder is numpy — where its output pins BITWISE against
F.linear on the stock weight. That bitwise pin is the correctness claim in
full: the codec is lossless, so the dense arm's bytes are the stock
checkpoint's bytes and its matmul is stock's own matmul. Versus the GEMV
arm's numerics (emulated here: the fp32 accumulate over the decoded weight,
one row at a time) only accumulation-order-level
agreement is expected — allclose, never bitwise.
"""

import math
import types

import numpy as np
import pytest
import torch

from drinkme.codec import radix_pack as rp, swap
from drinkme.codec.ops import MC_MAX
from drinkme.codec.swap import CompressedLinear, RadixCompressedLinear, to_device_radix

from fixtures import realistic_bf16_bits


def make_linear(seed, spread=True, r=32, c=64, bias=False, profile="sip"):
    """RadixCompressedLinear on a CPU pack + the stock bf16 weight it encodes."""
    U = realistic_bf16_bits(r, c, seed, spread=spread)
    pack = rp.pack_array_radix(U, profile, encoder="numpy")
    assert pack is not None
    if spread:
        assert len(pack["rx_palette"]) >= 4  # several exponents: more than one tier in use
    w_stock = torch.from_numpy(U.view(np.int16).copy()).view(torch.bfloat16)
    b = None
    if bias:
        torch.manual_seed(seed)
        b = torch.randn(r, dtype=torch.bfloat16)
    return RadixCompressedLinear(to_device_radix(pack, "cpu"), b), w_stock


@pytest.fixture(autouse=True)
def fresh_threshold(monkeypatch):
    """Each test starts with the env unset and the read-once cache empty."""
    monkeypatch.delenv("DRINKME_PREFILL_DENSE_MIN", raising=False)
    monkeypatch.setattr(swap, "_DENSE_MIN", None)


def force_dense(monkeypatch):
    """Route every input down the dense arm regardless of device — the only
    way the triton-free dense branch runs under CPU-only tests, and proof
    that forward dispatches on _route's answer."""
    monkeypatch.setattr(CompressedLinear, "_route", lambda self, x: "dense")


def fake_gpu(shape):
    """Duck-typed stand-in for a device tensor: _route reads exactly
    .device.type, .numel(), .shape — so the device-facing router (including
    M = product of ALL leading dims) is testable without a GPU."""
    return types.SimpleNamespace(
        device=types.SimpleNamespace(type="cuda"),
        numel=lambda: math.prod(shape),
        shape=shape,
    )


# ---------------------------------------------------------------- routing --


def test_route_cpu_always_wins():
    m, _ = make_linear(1)
    for rows in (1, 2, 4096):
        assert m._route(torch.zeros(rows, m.C, dtype=torch.bfloat16)) == "cpu"


def test_route_rows_below_at_above_default_threshold():
    m, _ = make_linear(1)
    t = CompressedLinear.GEMM_MIN_ROWS
    assert m._route_rows(1) == "gemv"  # M==1 is the GEMV op, always
    assert m._route_rows(2) == "mc"
    assert m._route_rows(t - 1) == "mc"
    assert m._route_rows(t) == "dense"
    assert m._route_rows(t + 1) == "dense"
    assert m._route_rows(4096) == "dense"


def test_mc_covers_2_through_mc_max_and_no_further():
    """The multi-column kernel is written out to MC_MAX accumulators — one
    more column would be silently dropped — so the router must never hand it
    a wider batch, whatever the dense threshold is doing."""
    m, _ = make_linear(1)
    assert m._route_rows(MC_MAX) in ("mc", "dense")
    assert m._route_rows(MC_MAX + 1) != "mc"
    assert m._route_rows(10_000) != "mc"


def test_route_m_is_product_of_leading_dims():
    """[B, T, C] input: the decision rides on B*T, not on any single dim —
    so a batch whose dims are each well under the threshold still reaches the
    dense arm once they multiply past it."""
    m, _ = make_linear(1)
    t = CompressedLinear.GEMM_MIN_ROWS
    b, tt = 2, -(-t // 2)  # 2 * ceil(t/2) >= t with both dims < t
    assert b < t and tt < t
    assert m._route(fake_gpu((b, tt, m.C))) == "dense"
    assert m._route(fake_gpu((2, 2, m.C))) == "mc"
    assert m._route(fake_gpu((1, 1, m.C))) == "gemv"


def test_env_overrides_threshold(monkeypatch):
    m, _ = make_linear(1)
    monkeypatch.setenv("DRINKME_PREFILL_DENSE_MIN", "8")
    assert m._route_rows(7) == "mc"
    assert m._route_rows(8) == "dense"
    # even a threshold of 1 never sends a single row to the dense arm
    monkeypatch.setattr(swap, "_DENSE_MIN", None)
    monkeypatch.setenv("DRINKME_PREFILL_DENSE_MIN", "1")
    assert m._route_rows(1) == "gemv"
    assert m._route_rows(2) == "dense"


def test_env_zero_or_negative_disables_dense(monkeypatch):
    """Dense off leaves mc serving 2..MC_MAX and the row loop serving the rest
    — the arm that reads the weights once per row is still the baseline
    instrument, it just does not serve anything a speculative cycle asks
    for."""
    m, _ = make_linear(1)
    for off in ("0", "-3"):
        monkeypatch.setattr(swap, "_DENSE_MIN", None)
        monkeypatch.setenv("DRINKME_PREFILL_DENSE_MIN", off)
        assert m._route_rows(1) == "gemv"
        for rows in (2, MC_MAX):
            assert m._route_rows(rows) == "mc"  # never "dense"
        for rows in (MC_MAX + 1, 100_000):
            assert m._route_rows(rows) == "loop"  # never "dense"


def test_env_is_read_once(monkeypatch):
    """The threshold is cached at first use: flipping the env mid-process
    does nothing (the instrument is set per RUN, not per request)."""
    m, _ = make_linear(1)
    assert m._route_rows(CompressedLinear.GEMM_MIN_ROWS) == "dense"
    monkeypatch.setenv("DRINKME_PREFILL_DENSE_MIN", "0")
    assert m._route_rows(CompressedLinear.GEMM_MIN_ROWS) == "dense"


def test_env_malformed_warns_and_uses_default(monkeypatch):
    m, _ = make_linear(1)
    monkeypatch.setenv("DRINKME_PREFILL_DENSE_MIN", "sixteen")
    with pytest.warns(UserWarning, match="DRINKME_PREFILL_DENSE_MIN"):
        got = m._route_rows(CompressedLinear.GEMM_MIN_ROWS)
    assert got == "dense"
    assert m._route_rows(CompressedLinear.GEMM_MIN_ROWS - 1) == "mc"


# --------------------------------------------------------- dense numerics --


def test_dense_branch_bitwise_equals_stock_flinear(monkeypatch):
    """The whole claim: lossless codec => the dense arm IS stock's matmul.
    Bitwise, not allclose — and against STOCK'S OWN composition, bias
    inside the F.linear call: one fused epilogue, one rounding. An oracle
    of `F.linear(x, W) + bias` would be a second rounding, the very bug
    this pins; the two differ by an ulp on
    ordinary random inputs, so this pin is the CPU-side ruler for it."""
    force_dense(monkeypatch)
    for seed, spread, bias in ((3, False, False), (4, True, False), (5, True, True)):
        m, w_stock = make_linear(seed, spread=spread, bias=bias)
        torch.manual_seed(seed + 100)
        x = torch.randn(40, m.C, dtype=torch.bfloat16)
        got = m(x)
        want = torch.nn.functional.linear(x, w_stock, m.bias)
        assert torch.equal(got.view(torch.int16), want.view(torch.int16))
        if m.bias is not None:  # the old composition is NOT the oracle
            twice = torch.nn.functional.linear(x, w_stock) + m.bias
            assert not torch.equal(twice.view(torch.int16), want.view(torch.int16))


def test_dense_branch_bitwise_on_3d_input(monkeypatch):
    """[B, T, C] flattens to B*T rows and reshapes back; bytes still exact."""
    force_dense(monkeypatch)
    m, w_stock = make_linear(6, spread=True, bias=True)
    torch.manual_seed(6)
    x = torch.randn(2, 20, m.C, dtype=torch.bfloat16)
    got = m(x)
    want = torch.nn.functional.linear(x.reshape(-1, m.C), w_stock, m.bias)
    want = want.reshape(2, 20, m.R)
    assert got.shape == (2, 20, m.R)
    assert torch.equal(got.view(torch.int16), want.view(torch.int16))


def test_dense_branch_bitwise_equals_cpu_reference_path(monkeypatch):
    """Cross-pin the two routes through the real forward: the CPU path
    (_cpu_weight: radix_pack.decode_back_radix) and the dense arm
    (_dense_weight, which on the CPU is the same decoder) must produce
    identical bytes end-to-end. Bias-free so both compositions are the bare
    matmul."""
    m, _ = make_linear(7, spread=True)
    torch.manual_seed(7)
    x = torch.randn(40, m.C, dtype=torch.bfloat16)
    ref = m(x)  # CPU route: F.linear on decode_cpu_weight
    force_dense(monkeypatch)
    got = m(x)
    assert torch.equal(got.view(torch.int16), ref.view(torch.int16))


def test_dense_vs_gemv_numerics_allclose(monkeypatch):
    """The dense arm vs the GEMV arm's ALGORITHM (the fp32 accumulate over
    the decoded weight, one row at a time — emulated in f32 since the kernel
    itself needs a GPU): accumulation-order-level agreement only, which is
    allclose, never bitwise (docs/method.md, numerical behavior)."""
    force_dense(monkeypatch)
    m, w_stock = make_linear(8, spread=True, bias=True, profile="gulp")
    torch.manual_seed(8)
    x = torch.randn(32, m.C, dtype=torch.bfloat16)
    outs = torch.empty(32, m.R, dtype=torch.float32)
    for i in range(32):
        outs[i] = w_stock.float() @ x[i].float()
    want = (outs + m.bias).to(torch.bfloat16)
    got = m(x)
    assert torch.allclose(got.float(), want.float(), rtol=1e-2, atol=1e-2)


def test_dense_decodes_per_call_no_caching(monkeypatch):
    """The bet is decode amortized over M rows WITHIN one call — never a
    resident dense copy (that would decompress the model). Two forwards must
    decode twice."""
    force_dense(monkeypatch)
    m, _ = make_linear(9, spread=True)
    calls = []
    real = RadixCompressedLinear._dense_weight
    monkeypatch.setattr(RadixCompressedLinear, "_dense_weight",
                        lambda self: (calls.append(1), real(self))[1])
    x = torch.randn(40, m.C, dtype=torch.bfloat16)
    m(x)
    m(x)
    assert len(calls) == 2


def test_forward_consults_the_router(monkeypatch):
    """forward() owns no second copy of the branch decision: it asks _route."""
    m, _ = make_linear(10)
    seen = []

    def spy(self, x):
        seen.append(x.shape)
        return "cpu"

    monkeypatch.setattr(CompressedLinear, "_route", spy)
    m(torch.randn(2, m.C, dtype=torch.bfloat16))
    assert len(seen) == 1
