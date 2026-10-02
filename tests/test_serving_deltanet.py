"""serving/deltanet.py — the DeltaNet decode recurrence's kernel route.

CPU: the route's decisions (torch on CPU; the knob; fla refused on CPU by
name; a failed or disagreeing probe falls back to the torch reference and
says so; the adapter's call-shape mapping against a stub). GPU (gated):
fla's fused kernel against the torch reference on the toy's shape — fp32
agreement to reduction-order noise, the bf16 |Δ| reported, ONE T=M call
bitwise equal to M chained T=1 calls (mtp.py's per-position replay ≡ one
call), and the route end to end on the toy model.

The toy is tests/test_serving_mtp.py's hybrid qwen3_5 config; the modeling
module's global is restored after every test that binds it, because it is
process-wide by design.
"""

from __future__ import annotations

import pytest
import torch

from drinkme.serving import deltanet
from test_serving_mtp import _cfg

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")

gpu_gate = pytest.mark.skipif(not torch.cuda.is_available(),
                              reason="fla's fused recurrence is triton; needs a device")


@pytest.fixture(scope="module")
def toy():
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

    torch.manual_seed(0)
    model = Qwen3_5ForCausalLM(_cfg()).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


@pytest.fixture
def modeling(toy):
    """The modeling module, with its recurrence global put back afterwards."""
    mods = deltanet.deltanet_modules(toy)
    assert len(mods) == 2  # the toy's two linear_attention layers
    m = deltanet._modeling_module(mods[0])
    before = getattr(m, deltanet.NAME)
    yield m
    setattr(m, deltanet.NAME, before)


def _bound(modeling):
    return getattr(modeling, deltanet.NAME)


def test_cpu_is_torch(toy, modeling, monkeypatch, capsys):
    monkeypatch.delenv(deltanet.ENV, raising=False)
    assert deltanet.route(toy, "cpu") == "torch"
    assert _bound(modeling) is deltanet.torch_reference(modeling)
    assert getattr(_bound(modeling), "route", None) is None
    out = capsys.readouterr().out
    assert "[drinkme] deltanet recurrence: torch reference (device cpu)" in out


def test_knob_torch_wins_before_any_import(toy, modeling, monkeypatch, capsys):
    monkeypatch.setenv(deltanet.ENV, "torch")
    # "cuda" here never reaches a device: the knob decides first
    assert deltanet.route(toy, "cuda") == "torch"
    assert _bound(modeling) is deltanet.torch_reference(modeling)
    assert f"torch reference ({deltanet.ENV}=torch)" in capsys.readouterr().out


def test_knob_fla_on_cpu_is_refused_by_name(toy, modeling, monkeypatch):
    monkeypatch.setenv(deltanet.ENV, "fla")
    with pytest.raises(SystemExit, match="cannot run on CPU"):
        deltanet.route(toy, "cpu")


def test_knob_garbage_is_refused(toy, modeling, monkeypatch):
    monkeypatch.setenv(deltanet.ENV, "triton")
    with pytest.raises(SystemExit, match="is not one of fla/torch"):
        deltanet.route(toy, "cpu")


def test_no_deltanet_layers_is_none(capsys):
    assert deltanet.route(torch.nn.Linear(4, 4), "cuda") == "none"
    assert capsys.readouterr().out == ""


def test_probe_failure_falls_back_loudly(toy, modeling, monkeypatch, capsys):
    """A Triton compile/launch failure on this device keeps the reference
    and says so on stderr; the same failure under a forced fla refuses."""
    pytest.importorskip("fla.ops.gated_delta_rule")

    def boom(*a, **k):
        raise RuntimeError("no kernel for gfxNNNN")

    monkeypatch.setattr(deltanet, "probe", boom)
    monkeypatch.delenv(deltanet.ENV, raising=False)
    assert deltanet.route(toy, "cuda") == "torch"
    assert _bound(modeling) is deltanet.torch_reference(modeling)
    err = capsys.readouterr()
    assert "drinkme: warning: fla fused kernel failed its probe on cuda (RuntimeError: no kernel" in err.err
    assert "torch reference (fla probe failed" in err.out
    monkeypatch.setenv(deltanet.ENV, "fla")
    with pytest.raises(SystemExit, match="failed its probe"):
        deltanet.route(toy, "cuda")


def test_probe_disagreement_falls_back_loudly(toy, modeling, monkeypatch, capsys):
    pytest.importorskip("fla.ops.gated_delta_rule")
    monkeypatch.setattr(deltanet, "probe", lambda *a, **k: (0.25, 0.25, "float32"))
    monkeypatch.delenv(deltanet.ENV, raising=False)
    assert deltanet.route(toy, "cuda") == "torch"
    assert "disagrees with the torch reference on cuda: max |Δ| 2.500e-01" in capsys.readouterr().err
    monkeypatch.setattr(deltanet, "probe", lambda *a, **k: (float("nan"), 0.0, "float32"))
    assert deltanet.route(toy, "cuda") == "torch"  # NaN is a disagreement, not a pass
    monkeypatch.setenv(deltanet.ENV, "fla")
    with pytest.raises(SystemExit, match="disagrees"):
        deltanet.route(toy, "cuda")


def test_probe_pass_binds_the_adapter(toy, modeling, monkeypatch, capsys):
    """With the probe stubbed to agree, the global becomes the adapter and
    the boot line names fla and both |Δ|s; a second route finds the torch
    reference through the adapter's __wrapped__ (idempotent)."""
    pytest.importorskip("fla.ops.gated_delta_rule")
    monkeypatch.setattr(deltanet, "probe", lambda *a, **k: (1.5e-6, 3.2e-3, "bfloat16"))
    monkeypatch.delenv(deltanet.ENV, raising=False)
    ref = deltanet.torch_reference(modeling)
    assert deltanet.route(toy, "cuda") == "fla"
    bound = _bound(modeling)
    assert bound.route == "fla" and bound.__wrapped__ is ref
    out = capsys.readouterr().out
    assert "fused_recurrent_gated_delta_rule (Triton, one launch per step; probe max |Δ|" in out
    assert "1.5e-06 fp32, 3.2e-03 bfloat16; default" in out
    assert deltanet.route(toy, "cuda") == "fla"
    assert _bound(modeling).__wrapped__ is ref  # not adapter-over-adapter
    assert deltanet.torch_reference(modeling) is ref
    # and back to the reference by the knob
    monkeypatch.setenv(deltanet.ENV, "torch")
    assert deltanet.route(toy, "cuda") == "torch"
    assert _bound(modeling) is ref


def test_adapter_maps_the_forward_call_onto_fla(modeling):
    """The forward's call shape (positional q/k/v, keyword g/beta/initial_state/
    output_final_state/use_qk_l2norm_in_kernel/cu_seqlens plus the
    TransformersKwargs it forwards) reaches a fla-shaped callee with fla's
    names and nothing it did not ask for."""
    seen = {}

    def fused(q, k, v, g=None, gk=None, gv=None, beta=None, scale=None, initial_state=None,
              output_final_state=False, use_qk_l2norm_in_kernel=False, cu_seqlens=None, **kwargs):
        seen.update(q=q, k=k, v=v, g=g, beta=beta, initial_state=initial_state,
                    output_final_state=output_final_state,
                    use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel, cu_seqlens=cu_seqlens,
                    extra=kwargs)
        return "out", "state"

    ref = deltanet.torch_reference(modeling)
    ad = deltanet.make_fla_adapter(fused, ref)
    q, k, v, g, b, h = (object() for _ in range(6))
    out = ad(q, k, v, g=g, beta=b, initial_state=h, output_final_state=True,
             use_qk_l2norm_in_kernel=True, cu_seqlens=None,
             position_ids="pos", cache_position="cp", output_attentions=False)
    assert out == ("out", "state")
    assert seen["q"] is q and seen["k"] is k and seen["v"] is v
    assert seen["g"] is g and seen["beta"] is b and seen["initial_state"] is h
    assert seen["output_final_state"] is True and seen["use_qk_l2norm_in_kernel"] is True
    assert seen["cu_seqlens"] is None
    assert seen["extra"] == {}  # position_ids & co. dropped, as the hub decorator does
    assert ad.__wrapped__ is ref and ad.route == "fla"
    assert ad.__name__ == "fla_recurrent_gated_delta_rule"


def test_reference_is_transformers_own(modeling):
    """The reference the route restores is transformers' pure-torch loop —
    if a transformers bump renames or moves it, this fails here."""
    ref = deltanet.torch_reference(modeling)
    assert ref.__name__ == "torch_recurrent_gated_delta_rule"
    assert ref.__module__.endswith("modeling_qwen3_5")
    assert getattr(ref, "__wrapped__", None) is None


# ----------------------------------------------------------------- GPU --

def _inputs(device, dtype, hv=4, k=16, v=16, t=1, seed=0):
    gen = torch.Generator(device="cpu").manual_seed(seed)
    r = lambda *s: torch.randn(*s, generator=gen).to(device)  # noqa: E731
    q, kk, vv = r(1, t, hv, k).to(dtype), r(1, t, hv, k).to(dtype), r(1, t, hv, v).to(dtype)
    g = -torch.rand(1, t, hv, generator=gen).to(device)
    beta = torch.rand(1, t, hv, generator=gen).to(device).to(dtype)
    h0 = r(1, hv, k, v)
    return q, kk, vv, g, beta, h0


@gpu_gate
def test_gpu_fla_agrees_with_reference_fp32(modeling):
    from fla.ops.gated_delta_rule import fused_recurrent_gated_delta_rule as fused

    ref = deltanet.torch_reference(modeling)
    ad = deltanet.make_fla_adapter(fused, ref)
    q, k, v, g, beta, h0 = _inputs("cuda", torch.float32)
    kw = dict(g=g, beta=beta, initial_state=h0, output_final_state=True, use_qk_l2norm_in_kernel=True)
    o_ref, s_ref = ref(q, k, v, **kw)
    o_fla, s_fla = ad(q, k, v, **kw)
    assert o_fla.shape == o_ref.shape and o_fla.dtype == o_ref.dtype == torch.float32
    assert s_fla.shape == s_ref.shape and s_fla.dtype == s_ref.dtype == torch.float32
    assert s_fla.data_ptr() != h0.data_ptr()  # a new state, initial_state untouched
    assert (o_fla - o_ref).abs().max().item() < 1e-5
    assert (s_fla - s_ref).abs().max().item() < 1e-5


@gpu_gate
def test_gpu_fla_bf16_within_bf16_noise(modeling):
    """bf16 q/k/v (the served dtype): the reference normalizes q/k in bf16,
    the kernel in fp32 — expect bf16-rounding-sized disagreement, not a
    wrong result. Output bf16, state fp32, as the reference."""
    from fla.ops.gated_delta_rule import fused_recurrent_gated_delta_rule as fused

    ref = deltanet.torch_reference(modeling)
    ad = deltanet.make_fla_adapter(fused, ref)
    q, k, v, g, beta, h0 = _inputs("cuda", torch.bfloat16)
    kw = dict(g=g, beta=beta, initial_state=h0, output_final_state=True, use_qk_l2norm_in_kernel=True)
    o_ref, s_ref = ref(q, k, v, **kw)
    o_fla, s_fla = ad(q, k, v, **kw)
    assert o_fla.dtype == o_ref.dtype == torch.bfloat16
    assert s_fla.dtype == s_ref.dtype == torch.float32
    assert (o_fla.float() - o_ref.float()).abs().max().item() < 5e-2
    assert (s_fla - s_ref).abs().max().item() < 5e-2


@gpu_gate
def test_gpu_one_call_of_t_equals_chained_single_steps(modeling):
    """mtp.py replays the recurrence one position at a time with the state
    threaded through; fla's kernel walks T inside one program with the
    state in registers. fp32 stores are exact, the per-step arithmetic is
    the same code, so the two are BITWISE equal — the fact a future
    one-call-per-verify would rest on."""
    from fla.ops.gated_delta_rule import fused_recurrent_gated_delta_rule as fused

    ref = deltanet.torch_reference(modeling)
    ad = deltanet.make_fla_adapter(fused, ref)
    for dtype in (torch.float32, torch.bfloat16):
        q, k, v, g, beta, h0 = _inputs("cuda", dtype, t=5, seed=1)
        o_all, s_all = ad(q, k, v, g=g, beta=beta, initial_state=h0, output_final_state=True,
                          use_qk_l2norm_in_kernel=True)
        state, outs = h0, []
        for j in range(5):
            o, state = ad(q[:, j:j + 1], k[:, j:j + 1], v[:, j:j + 1], g=g[:, j:j + 1],
                          beta=beta[:, j:j + 1], initial_state=state, output_final_state=True,
                          use_qk_l2norm_in_kernel=True)
            outs.append(o)
        assert torch.equal(torch.cat(outs, dim=1), o_all), dtype
        assert torch.equal(state, s_all), dtype


@gpu_gate
def test_gpu_route_binds_fla_on_the_toy(toy, modeling, monkeypatch, capsys):
    """End to end on the device: the real probe passes on the toy's shape,
    the global becomes the adapter, and a cached decode step through the
    model's own forward runs it (the adapter is what the forward calls)."""
    monkeypatch.delenv(deltanet.ENV, raising=False)
    model = toy.to("cuda")
    try:
        assert deltanet.route(model, "cuda") == "fla"
        line = capsys.readouterr().out
        assert "fused_recurrent_gated_delta_rule (Triton" in line and "float32; default" in line
        calls = []
        bound = _bound(modeling)

        def spy(*a, **k):
            calls.append(1)
            return bound(*a, **k)

        setattr(modeling, deltanet.NAME, spy)
        ids = torch.tensor([[3, 4, 5, 6]], device="cuda")
        with torch.no_grad():
            out = model(ids, use_cache=True)
            calls.clear()
            model(ids[:, -1:] * 0 + 7, past_key_values=out.past_key_values, use_cache=True,
                  cache_position=torch.tensor([4], device="cuda"))
        assert calls == [1, 1]  # once per linear_attention layer, M=1 decode
    finally:
        toy.to("cpu")
