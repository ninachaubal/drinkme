"""Prefill convolution routing, numerical agreement, and cache continuity."""
import sys

import pytest
import torch
import torch.nn.functional as F

from drinkme.serving import deltanet_conv as conv

gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a Triton accelerator")


def upstream(x, w, bias=None, activation=None, **kwargs):
    y = F.conv1d(x, w.unsqueeze(1), bias, padding=w.shape[-1]-1,
                 groups=w.shape[0])[..., :x.shape[-1]]
    return F.silu(y) if activation in ("silu", "swish") else y


@pytest.fixture
def model(monkeypatch):
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM
    from test_serving_mtp import _cfg

    torch.manual_seed(3)
    model = Qwen3_5ForCausalLM(_cfg()).eval().requires_grad_(False)
    modeling = sys.modules[type(model).__module__]
    monkeypatch.setattr(modeling, conv.NAME, conv.reference(modeling))
    return model


def test_cpu_adapter_preserves_upstream_dispatch():
    calls = []
    def existing(*a, **kw):
        calls.append(kw)
        return upstream(*a, **kw)
    def forbidden(*a, **kw):
        pytest.fail("Triton must not run on CPU")
    adapter = conv.make_adapter(forbidden, existing)
    x, w = torch.randn(1, 8, 140), torch.randn(8, 4)
    assert torch.equal(adapter(x, w, activation="silu", marker=3), upstream(x, w, activation="silu"))
    assert calls == [{"activation": "silu", "marker": 3}]


def test_route_validation_and_cpu(model, monkeypatch):
    monkeypatch.delenv(conv.ENV, raising=False)
    assert conv.route(model, "cpu") == "torch"
    assert conv.route(torch.nn.Linear(2, 2), "cpu") == "none"
    monkeypatch.setenv(conv.ENV, "fla")
    with pytest.raises(SystemExit, match="requires a CUDA/ROCm device"):
        conv.route(model, "cpu")
    monkeypatch.setenv(conv.ENV, "invalid")
    with pytest.raises(SystemExit, match="not one of fla/torch"):
        conv.route(model, "cpu")


def test_probe_failure_and_rebinding(model, monkeypatch, capsys):
    modeling = sys.modules[type(model).__module__]
    original = conv.reference(modeling)
    monkeypatch.setattr(torch.version, "hip", "test")
    monkeypatch.setattr(conv, "_load_kernel", lambda: object())
    monkeypatch.delenv(conv.ENV, raising=False)
    def broken(*a):
        raise RuntimeError("bad kernel")
    monkeypatch.setattr(conv, "probe", broken)
    assert conv.route(model, "cuda") == "torch"
    assert getattr(modeling, conv.NAME) is original
    assert "bad kernel" in capsys.readouterr().err
    monkeypatch.setenv(conv.ENV, "fla")
    with pytest.raises(SystemExit, match="bad kernel"):
        conv.route(model, "cuda")
    monkeypatch.setattr(conv, "probe", lambda *a: 0.0)
    assert conv.route(model, "cuda") == "fla"
    assert conv.reference(modeling) is original
    assert conv.route(model, "cuda") == "fla"
    assert conv.reference(modeling) is original
    monkeypatch.setenv(conv.ENV, "torch")
    assert conv.route(model, "cuda") == "torch"
    assert getattr(modeling, conv.NAME) is original


@gpu
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("rows", [127, 128, 257])
@pytest.mark.parametrize("layout", ["time_major", "cropped_channel_major"])
def test_gpu_convolution_layouts_threshold_and_rounding(dtype, rows, layout):
    from fla.modules.conv import causal_conv1d

    adapter = conv.make_adapter(causal_conv1d, upstream)
    x = torch.randn(2, rows, 96, device="cuda", dtype=dtype).transpose(1, 2)
    if layout == "cropped_channel_major":
        x = F.pad(x, (1, 2))[..., 1:-2]
    w = torch.randn(96, 4, device="cuda", dtype=dtype) / 4
    bias = torch.randn(96, device="cuda", dtype=dtype) / 4
    before = w.clone()
    with torch.inference_mode():
        for activation, b in ((None, None), ("silu", bias), ("swish", None)):
            a, b = adapter(x, w, b, activation), upstream(x, w, b, activation)
            tol = 2e-5 if dtype == torch.float32 else .016
            torch.testing.assert_close(a, b, atol=tol, rtol=tol)
            if rows >= conv.MIN_ROWS:
                assert a.stride(1) == 1  # the next projection sees time-major backing
    assert torch.equal(w, before)


@gpu
def test_gpu_prefill_suffix_and_mtp_rollback(model, monkeypatch):
    from drinkme.serving import deltanet, mtp
    from drinkme.serving.kvcache import LiveStaticCache
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule

    modeling = sys.modules[type(model).__module__]
    monkeypatch.setattr(modeling, "torch_chunk_gated_delta_rule", chunk_gated_delta_rule)
    monkeypatch.setattr(modeling, deltanet.NAME,
                        deltanet.make_fla_adapter(fused_recurrent_gated_delta_rule,
                                                  deltanet.torch_reference(modeling)))
    model = model.to("cuda")
    monkeypatch.setenv(conv.ENV, "fla")
    assert conv.route(model, "cuda") == "fla"
    mtp.install_deltanet_capture(model)
    prefix = torch.randint(3, 90, (1, 128), device="cuda")
    suffix = torch.randint(3, 90, (1, 128), device="cuda")
    rows = torch.tensor([[3, 41, 8, 62, 15]], device="cuda")

    def prefill():
        cache = LiveStaticCache(config=model.config, max_cache_len=300)
        model(prefix, past_key_values=cache, use_cache=True, logits_to_keep=1)
        logits = model(suffix, past_key_values=cache, use_cache=True, logits_to_keep=1).logits
        return cache, logits

    with torch.inference_mode():
        got, logits = prefill()
        selected = getattr(modeling, conv.NAME)
        monkeypatch.setattr(modeling, conv.NAME, conv.reference(modeling))
        baseline, expected = prefill()
        torch.testing.assert_close(logits, expected, atol=2e-5, rtol=2e-5)
        monkeypatch.setattr(modeling, conv.NAME, selected)
        cap = mtp._Capture()
        with monkeypatch.context() as mp:
            mp.setattr(mtp, "_ACTIVE", cap)
            model(rows, past_key_values=got, use_cache=True, logits_to_keep=1)
        mtp._restore_rows(got, cap, keep=2)
        # The reference sees only the accepted rows, one at a time.
        for j in range(2):
            model(rows[:, j:j+1], past_key_values=baseline, use_cache=True, logits_to_keep=1)
        for a, b in zip(got.layers, baseline.layers):
            if hasattr(a, "live"):
                assert a.live == b.live == 258
                torch.testing.assert_close(a.keys[..., :258, :], b.keys[..., :258, :], atol=2e-5, rtol=2e-5)
            else:
                torch.testing.assert_close(a.conv_states[0], b.conv_states[0], atol=2e-5, rtol=2e-5)
                torch.testing.assert_close(a.recurrent_states[0], b.recurrent_states[0], atol=2e-5, rtol=2e-5)
        # Check the next decode token after rejection, too.
        a = model(rows[:, 2:3], past_key_values=got, use_cache=True).logits
        b = model(rows[:, 2:3], past_key_values=baseline, use_cache=True).logits
        torch.testing.assert_close(a, b, atol=2e-5, rtol=2e-5)
