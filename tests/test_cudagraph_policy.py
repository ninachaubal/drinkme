"""serving/cudagraph.py: when the decode step runs as a replayed CUDA graph,
and what a run says and records when it does not.

The policy (default on for CUDA on the verified list, DRINKME_CUDA_GRAPHS=0,
ROCm/Metal/CPU never, one line on an unverified family or a failed capture),
the engine string's decode segment and the publish door for arms that
disagree, all with fakes; then the static step itself on CPU (its attention
through the SDPA reference, since capture needs CUDA): the same tokens as
today's eager step, the device counter and the Python `live` agreeing, and a
failed capture that decodes eager and says so once. The capture and the
bitwise graph == eager check are bench/cuda_graph_gate.py's, on a GPU.
"""

from __future__ import annotations

import types

import pytest

torch = pytest.importorskip("torch")

from drinkme import arms, bench  # noqa: E402
from drinkme.publish import validate  # noqa: E402
from drinkme.serving import cudagraph  # noqa: E402
from drinkme.serving.kvcache import LiveStaticCache  # noqa: E402


def fake_model(model_type: str):
    return types.SimpleNamespace(config=types.SimpleNamespace(model_type=model_type))


@pytest.fixture
def on_cuda(monkeypatch):
    """A CUDA build with a device, whatever this box is."""
    monkeypatch.setattr(torch.version, "hip", None, raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.delenv(cudagraph.ENV, raising=False)


@pytest.fixture
def said(monkeypatch):
    monkeypatch.setattr(cudagraph, "_SAID", set())


# ---------------------------------------------------------------- policy --


def test_default_on_for_cuda_on_the_verified_list(on_cuda):
    for fam in ("qwen3", "qwen3_5_text"):
        d = cudagraph.decide(fake_model(fam), "cuda")
        assert d.on
        assert "CUDA graphs" in d.line and "experimental" in d.line
        assert cudagraph.ENV + "=0" in d.line  # the way out, named once
    assert cudagraph.decide(fake_model("qwen3"), None).on  # device unknown: the build decides


def test_the_env_override_forces_eager(on_cuda, monkeypatch):
    monkeypatch.setenv(cudagraph.ENV, "0")
    d = cudagraph.decide(fake_model("qwen3"), "cuda")
    assert not d.on and d.line == f"decode step: eager ({cudagraph.ENV}=0)"
    monkeypatch.setenv(cudagraph.ENV, "1")  # anything but 0 is the default
    assert cudagraph.decide(fake_model("qwen3"), "cuda").on


def test_an_unverified_family_decodes_eager_with_one_line(on_cuda):
    d = cudagraph.decide(fake_model("gemma4"), "cuda")
    assert not d.on
    assert "gemma4" in d.line and "not verified" in d.line
    # the verify widths are their own entry
    assert cudagraph.verified(fake_model("qwen3"), "verify")
    assert not cudagraph.verified(fake_model("gemma4"), "verify")
    assert not cudagraph.decide(fake_model("qwen3"), "cuda", mode="prefill").on


def test_every_profile_drinkme_packs_was_gated():
    """VERIFIED is keyed by (family, mode), so a sip pack and a gulp pack of
    one family get the same answer. That holds because every profile
    `drinkme pack` writes passed the gate; a new one fails here until it
    has, or until VERIFIED is keyed by profile."""
    from drinkme.codec import radix_pack
    from drinkme.serving import cudagraph_fit

    assert set(radix_pack.PUBLIC_PROFILES) <= set(cudagraph_fit.GATED_PROFILES) == {"sip", "gulp"}
    assert all(len(key) == 2 for key in cudagraph.VERIFIED)


def test_rocm_metal_and_cpu_never_graph_and_say_nothing(monkeypatch):
    monkeypatch.delenv(cudagraph.ENV, raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.version, "hip", "7.1.0", raising=False)
    assert cudagraph.decide(fake_model("qwen3"), "cuda") == cudagraph.Decision(False, None)
    assert cudagraph.platform("cuda") == "rocm"
    monkeypatch.setattr(torch.version, "hip", None, raising=False)
    assert cudagraph.decide(fake_model("qwen3"), "mps") == cudagraph.Decision(False, None)
    assert cudagraph.decide(fake_model("qwen3"), "cpu") == cudagraph.Decision(False, None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert cudagraph.decide(fake_model("qwen3"), None) == cudagraph.Decision(False, None)


# -------------------------------------------------- bench and its record --


def test_bench_asks_the_policy_per_arm(on_cuda, said, monkeypatch, capsys):
    model = torch.nn.Linear(2, 2)
    model.config = types.SimpleNamespace(model_type="qwen3")
    monkeypatch.setattr(cudagraph, "platform", lambda device=None: "cuda")
    assert arms.decode_step(model) == "cudagraph"
    model.config.model_type = "gemma4"
    assert arms.decode_step(model) == "eager"
    assert arms.decode_step(object()) == "eager"  # the tests' stubs
    err = capsys.readouterr().err
    assert err.count("not verified for gemma4") == 1


def test_a_failed_capture_times_eager_and_says_so(said, monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a: None)
    monkeypatch.setattr(arms, "greedy", lambda model, ids, n: calls.append(n) or "eager-out")

    class Boom:
        def __init__(self, *a):
            raise RuntimeError("operation not permitted when stream is capturing")

    monkeypatch.setattr(arms, "GraphDecode", Boom)
    ran = {}
    out, samples = arms.timed_decode(object(), None, n_new=4, reps=2, step="cudagraph", ran=ran)
    assert ran == {"step": "eager"} and out == "eager-out" and len(samples) == 2
    assert calls == [4] * (2 + int(arms.WARMUP_REP))
    err = capsys.readouterr().err
    assert err.count("CUDA graph capture failed (RuntimeError: operation not permitted") == 1
    assert "timing eager" in err


ROUTING = {"deltanet_kernel": "none", "deltanet_conv": "none", "narrow_gemv": True,
           "narrow_count": 0, "stock_gemv": "linear", "raw_gemv": "stock"}


def test_the_engine_string_carries_the_decode_segment():
    s = bench.engine_string("torch", "2.13.0+cu128", ROUTING, decode="cudagraph")
    assert s.endswith(" / raw-gemv stock / decode cudagraph")
    assert bench.parse_engine(s)["decode"] == "cudagraph"
    e = bench.engine_string("torch", "2.13.0+cu128", ROUTING, decode="eager")
    assert bench.parse_engine(e)["decode"] == "eager"
    # a record from before graph mode (seven segments) still parses, without the field
    old = bench.engine_string("torch", "2.13.0+cu128", ROUTING)
    assert "decode" not in bench.parse_engine(old)
    for bad in (old + " / decode graph", old + " / step cudagraph", old + " / decode cudagraph / x y"):
        with pytest.raises(ValueError):
            bench.parse_engine(bad)
    with pytest.raises(ValueError):
        bench.engine_string("torch", "2.13.0", ROUTING, decode="graph")
    assert bench.engine_string("mlx", "0.32.2") == f"drinkme {bench.__version__} / mlx 0.32.2"


def test_the_record_names_the_step_the_arms_ran_and_refuses_a_mix():
    raw = {"stock_decode_step": "cudagraph", "twin_decode_step": "cudagraph",
           "compressed_decode_step": "cudagraph"}
    assert bench.record_decode_step(raw) == "cudagraph" and "unpublishable" not in raw
    raw = {"stock_decode_step": "eager", "compressed_decode_step": "eager"}
    assert bench.record_decode_step(raw) == "eager" and "unpublishable" not in raw
    assert bench.record_decode_step({}) == "eager"
    raw = {"stock_decode_step": "cudagraph", "compressed_decode_step": "eager"}
    assert bench.record_decode_step(raw) == "eager"
    assert "different step paths (stock cudagraph, compressed eager)" in raw["unpublishable"]
    # the publish door says the same, whatever raw.unpublishable says
    assert validate.decode_step_verdict({"raw": raw}).startswith("the arms decoded on different step paths")
    assert validate.decode_step_verdict({"raw": {"stock_decode_step": "cudagraph",
                                                 "compressed_decode_step": "cudagraph"}}) is None
    assert validate.decode_step_verdict({}) is None


# ------------------------------------------------------- the static step --


def toy_qwen3():
    from transformers import Qwen3Config, Qwen3ForCausalLM

    cfg = Qwen3Config(vocab_size=128, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=16,
                      max_position_embeddings=256)
    cfg._attn_implementation = "sdpa"
    torch.manual_seed(0)
    return Qwen3ForCausalLM(cfg).eval()


def prefilled(model, ids, alloc=64):
    cache = LiveStaticCache(config=model.config, max_cache_len=alloc)
    return cache, model(ids, past_key_values=cache, use_cache=True).logits[0, -1]


def test_the_static_step_decodes_what_the_eager_step_decodes():
    """The whole allocation with the live length in the marker (the SDPA
    reference on CPU) against LiveStaticLayer's live window: same tokens,
    last logits to accumulation order, counter and int agreeing after."""
    model = toy_qwen3()
    ids = torch.randint(0, 128, (1, 9), generator=torch.Generator().manual_seed(1))
    with torch.inference_mode():
        cache, last = prefilled(model, ids)
        t, eager = int(last.argmax()), []
        for _ in range(10):
            el = model(torch.tensor([[t]]), past_key_values=cache, use_cache=True).logits[0, -1]
            t = int(el.argmax())
            eager.append(t)
        cache2, last2 = prefilled(model, ids)
        step = cudagraph.StaticStep(model, cache2, 1, lambda i, p, m: model(
            input_ids=i, position_ids=p, attention_mask=m, past_key_values=cache2, use_cache=True).logits[0, -1])
        snap = step.snapshot()
        t, static = int(last2.argmax()), []
        for _ in range(10):
            step.ids.fill_(t)
            sl = step.eager()
            t = int(sl.argmax())
            static.append(t)
        assert static == eager
        assert torch.allclose(sl, el, atol=1e-5)
        layers = step.layers
        assert [layer.live for layer in layers] == [int(layer.cumulative_length) for layer in layers] == [19, 19]
        assert not any(layer.whole for layer in layers)  # only inside a step
        step.restore(snap)
        assert [layer.live for layer in layers] == [int(layer.cumulative_length) for layer in layers] == [9, 9]


def test_verify_rows_through_the_static_step_match_the_eager_verify():
    from drinkme.serving.mtp import forward_with_hidden

    model = toy_qwen3()
    ids = torch.randint(0, 128, (1, 9), generator=torch.Generator().manual_seed(2))
    rows = torch.randint(0, 128, (1, 4), generator=torch.Generator().manual_seed(3))
    with torch.inference_mode():
        cache, _ = prefilled(model, ids)
        h1, l1 = forward_with_hidden(model, rows, cache, torch.arange(9, 13))
        cache2, _ = prefilled(model, ids)
        step = cudagraph.StaticStep(model, cache2, 4, lambda i, p, m: forward_with_hidden(
            model, i, cache2, None, attention_mask=m, position_ids=p))
        step.ids.copy_(rows)
        h2, l2 = step.eager()
    assert torch.equal(l1.argmax(-1), l2.argmax(-1))
    assert torch.allclose(l1, l2, atol=1e-5) and torch.allclose(h1, h2, atol=1e-5)
    assert [layer.live for layer in step.layers] == [13, 13]


def test_the_marker_attention_ignores_rows_past_the_live_length():
    g = torch.Generator().manual_seed(4)
    q = torch.randn(1, 4, 3, 16, generator=g)
    k = torch.randn(1, 2, 32, 16, generator=g)
    v = torch.randn(1, 2, 32, 16, generator=g)
    m = cudagraph.LiveLength(3, 32, torch.tensor([20], dtype=torch.int32),
                             torch.tensor([0, 3], dtype=torch.int32), torch.tensor([0, 32], dtype=torch.int32))
    a, _ = cudagraph.attention_forward(None, q, k, v, m, scaling=0.25)
    k2, v2 = k.clone(), v.clone()
    k2[:, :, 20:], v2[:, :, 20:] = 1e4, 1e4
    b, _ = cudagraph.attention_forward(None, q, k2, v2, m, scaling=0.25)
    assert torch.equal(a, b)
    # bottom-right causal: the last row sees keys 0..19, the first 0..17
    ref = torch.nn.functional.scaled_dot_product_attention(
        q[:, :, :1], k[:, :, :18], v[:, :, :18], scale=0.25, enable_gqa=True)
    assert torch.allclose(a[:, :1].transpose(1, 2), ref, atol=1e-6)
    with pytest.raises(ValueError):
        cudagraph.attention_forward(None, q, k[:, :, :31], v[:, :, :31], m, scaling=0.25)


def test_a_cache_the_step_cannot_take_is_refused_by_name():
    model = toy_qwen3()
    with torch.inference_mode():
        cache = LiveStaticCache(config=model.config, max_cache_len=16)
        with pytest.raises(cudagraph.Unsupported, match="unwritten"):
            cudagraph.StaticStep(model, cache, 1, None)
    assert cudagraph.mask_mapping(model, "m") == {"full_attention": "m"}
    hybrid = types.SimpleNamespace(config=types.SimpleNamespace(
        layer_types=["linear_attention", "full_attention"]))
    assert cudagraph.mask_mapping(hybrid, "m") == {"full_attention": "m", "linear_attention": None}
    sliding = types.SimpleNamespace(config=types.SimpleNamespace(
        layer_types=["sliding_attention", "full_attention"]))
    with pytest.raises(cudagraph.Unsupported, match="sliding_attention"):
        cudagraph.mask_mapping(sliding, "m")


def test_a_failed_capture_decodes_eager_once_said_and_leaves_the_cache(said, capsys):
    """On this CPU the capture itself fails (no CUDA stream): decode_step
    says so once, remembers it on the cache, and the cache is as it was."""
    model = toy_qwen3()
    ids = torch.randint(0, 128, (1, 9), generator=torch.Generator().manual_seed(5))
    with torch.inference_mode():
        cache, _ = prefilled(model, ids)
        assert cudagraph.decode_step(model, cache) is None
        assert cudagraph.decode_step(model, cache) is None
        cache2, _ = prefilled(model, ids)
        assert cudagraph.decode_step(model, cache2) is None
    err = capsys.readouterr().err
    assert err.count("decode step: CUDA graph capture failed") == 1 and "decoding eager" in err
    assert [layer.live for layer in cache.layers] == [int(layer.cumulative_length) for layer in cache.layers] == [9, 9]


def test_the_engine_serves_eager_when_capture_fails(said, monkeypatch, capsys):
    """An engine the policy put in graph mode, on a machine where capture
    fails: the request is answered, token for token what the eager engine
    answers, graph mode is off for the engine afterwards, and the line is
    printed once."""
    from drinkme.serving.engine import GenerationRequest, SampleParams, complete
    from tests.hotloop_toys import toy_engine

    def text(eng):
        req = GenerationRequest([{"role": "user", "content": "hello there"}],
                                SampleParams(temperature=0.0, max_tokens=12))
        return complete(eng, req, lambda _t: True).text

    monkeypatch.setenv("DRINKME_SPEC", "off")
    eager = text(toy_engine())
    monkeypatch.setattr(cudagraph, "decide", lambda model, device=None, mode="decode": cudagraph.Decision(
        True, "decode step: CUDA graphs (experimental; DRINKME_CUDA_GRAPHS=0 for eager)"))
    eng = toy_engine()
    assert eng._graphs
    assert text(eng) == eager
    assert not eng._graphs
    assert text(eng) == eager
    out = capsys.readouterr()
    assert "decode step: CUDA graphs (experimental" in out.out
    assert out.err.count("decode step:") == 1


def test_live_layer_hands_attention_the_whole_allocation_only_inside_a_step():
    model = toy_qwen3()
    ids = torch.randint(0, 128, (1, 5), generator=torch.Generator().manual_seed(6))
    with torch.inference_mode():
        cache, _ = prefilled(model, ids, alloc=32)
        layer = cache.layers[0]
        kv = torch.zeros(1, 2, 1, 16)
        k, _v = layer.update(kv, kv)
        assert k.shape[-2] == 6 and layer.live == 6
        layer.whole = True
        k, _v = layer.update(kv, kv)
        assert k.shape[-2] == 32 and layer.live == 7 and int(layer.cumulative_length) == 7
