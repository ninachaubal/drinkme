"""The MLX engine on a toy Qwen3 — CPU only, no Metal, no downloads.

The toy mirrors tests/test_serving_engines.py's Llama toy on the Qwen3
architecture (the family this engine serves): a genuine 2-layer
Qwen3ForCausalLM (hidden 1024 so q/o/gate/up/down clear codec eligibility;
kv_heads=2 keeps k/v at 512 wide so they stay unpacked and both attach
paths run) plus a genuine WordLevel fast tokenizer, saved to disk and
packed through pack_model — a radix pack, the same pack the
torch engine loads. mlx here is the CPU build, so every forward runs the
REFERENCE path (the CPU radix decoder, then the bf16 matmul); the fused
kernels are bench/radix_bitpin_mlx.py's business on the Mac, and the
fused-vs-reference pin at the bottom runs there only.

Claims, each its own test:
  (1) the decoded weights are bit-exact to codec.radix_pack.decode_back_radix,
      every packed Linear is a RadixLinear and nothing else was swapped;
  (2) the next-token logits after the prompt are within bf16 tolerance of
      the torch CPU engine's on the same ids, argmax identical;
  (3) a short greedy generation equals HFEngine's — with the near-tie
      caveat: if the two fork, the prefix up to the fork is asserted and the
      fork is reported, since the invariant is bit-identical WEIGHTS, not
      identical transcripts (AGENTS.md);
  (4) the HTTP layer serves it end to end over a real socket;
  (5) against the SHIPPED torch reference (bench/radix_mlx_toy_build.py
      --reference: the torch CPU engine's transcript of four prompts,
      carried with the toy to the box that has no torch), both compute
      paths, on the plain toy AND its biased twin (--bias: attention_bias
      =True, q/o biases on RadixLinear, k/v biases on the Linear mlx-lm
      built without a slot) — prompt logits within the same bf16
      tolerance as (2), every greedy step's row too, ids identical, a
      fork only at a near-tie. Where torch IS present the live engine is
      held to the file as well, so the reference builder is itself
      checked; a reference for other bytes is refused, never used.
The raw fallback's MLX arm (RawLinear) and make_module_mlx's per-kind
dispatch — a pack with raw fallbacks' mechanism — are pinned on dicts below,
and so is the biased epilogue: one rounding on every module, the
cancellation tests/test_codec_bias_rounding.py measures the torch arms with.
"""

import http.client
import json
import os
import sys
import types

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm")
import mlx.nn as nn  # noqa: E402

try:  # the Mac lane has no torch: the toy comes prebuilt (DRINKME_RADIX_MLX_TOY)
    import torch
except ImportError:  # pragma: no cover — the Mac
    torch = None

from drinkme.serving.engine import (Delta, Finished, GenerationRequest,  # noqa: E402
                                    SampleParams, StreamStart, complete)
from drinkme.serving.engine_mlx import (  # noqa: E402
    BiasedLinear, MLXEngine, RadixLinear, RawLinear, _module_at, biased_matmul,
    load_compressed_mlx, load_stock_mlx, make_module_mlx, normalize_config, fit_check,
    refuse_unsupported, sample_next,
)

BENCH_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bench")
sys.path.insert(0, BENCH_DIR)
from radix_mlx_toy_build import REFERENCE, Recorder, load_reference, ulp  # noqa: E402

TOY_ENV = "DRINKME_RADIX_MLX_TOY"
MSGS = [{"role": "user", "content": "hello world the quick brown fox"}]


def greedy(max_tokens=24, **kw):
    return SampleParams(temperature=0.0, max_tokens=max_tokens, **kw)


def run(eng, params, msgs=MSGS):
    deltas = []
    res = complete(eng, GenerationRequest(msgs, params), lambda d: deltas.append(d) or True)
    return res, deltas


def _prebuilt(kind: str) -> str | None:
    """The prebuilt toy root for `kind` ("plain" | "bias") under
    DRINKME_RADIX_MLX_TOY — the biased twin lives in its bias/ subdir
    (bench/radix_mlx_toy_build.py's three commands) — or None."""
    root = os.environ.get(TOY_ENV)
    if not root:
        return None
    sub = root if kind == "plain" else os.path.join(root, "bias")
    return sub if os.path.isdir(os.path.join(sub, "pack")) else None


def _build_toy(tmp_path_factory, kind: str):
    if torch is None:
        pytest.skip(f"no torch to build the {kind} toy and {TOY_ENV} names none "
                    f"(bench/radix_mlx_toy_build.py{' --bias' if kind == 'bias' else ''})")
    from radix_mlx_toy_build import build_toy_qwen3

    d = tmp_path_factory.mktemp(f"toy_qwen3_{kind}")
    return build_toy_qwen3(str(d / "model"), str(d / "pack"), bias=kind == "bias")


@pytest.fixture(scope="session")
def toy(tmp_path_factory):
    """Toy Qwen3 model dir + pack dir, built once — by
    bench/radix_mlx_toy_build.py's recipe where torch exists, else the
    prebuilt one DRINKME_RADIX_MLX_TOY names (model/ + pack/, shipped from a
    torch box; tests/test_metal_radix.py reads the same variable)."""
    root = _prebuilt("plain")
    if root:
        return os.path.join(root, "model"), os.path.join(root, "pack")
    return _build_toy(tmp_path_factory, "plain")


@pytest.fixture(scope="session")
def biased_toy(tmp_path_factory):
    """The biased twin (--bias): the same recipe with attention_bias=True
    and the biases drawn — prebuilt under <toy>/bias/, else built here."""
    root = _prebuilt("bias")
    if root:
        return os.path.join(root, "model"), os.path.join(root, "pack")
    return _build_toy(tmp_path_factory, "bias")


@pytest.fixture(scope="session", params=["plain", "bias"])
def toy_reference(request, tmp_path_factory):
    """(kind, model_dir, pack_dir, reference) — the toy of each kind with
    its torch reference: the shipped reference.npz beside a prebuilt toy
    (REFUSED by load_reference if it was built for other bytes — the test
    errors, it never compares against a stale file), else written here
    from the torch engine, which is what checks the builder wherever torch
    and mlx coexist."""
    kind = request.param
    model_dir, pack_dir = request.getfixturevalue("toy" if kind == "plain" else "biased_toy")
    path = os.path.join(os.path.dirname(model_dir), REFERENCE)
    if not os.path.exists(path):
        if torch is None:
            pytest.skip(f"the {kind} toy carries no {REFERENCE} "
                        "(bench/radix_mlx_toy_build.py --reference on a machine with torch)")
        from radix_mlx_toy_build import write_reference

        path = str(tmp_path_factory.mktemp(f"reference_{kind}") / REFERENCE)
        write_reference(path, model_dir, pack_dir)
    return kind, model_dir, pack_dir, load_reference(path, model_dir, pack_dir)


needs_metal = pytest.mark.skipif(not mx.metal.is_available(),
                                 reason="dispatches the fused Metal kernel; no Metal on this machine")


@pytest.fixture(scope="session")
def mlx_comp(toy):
    """The engine on the REFERENCE path, explicitly — on the Linux box it is
    the only path there is; on a Mac it is the control the fused engine
    below is pinned against."""
    return load_compressed_mlx(toy[0], None, toy[1], path="reference")


@pytest.fixture(scope="session")
def hf_comp(toy):
    if torch is None:
        pytest.skip("the torch engine is the control here; no torch on this machine")
    from drinkme.serving.engines import load_compressed

    os.environ["DRINKME_PREFIX_SLOTS"] = "0"
    try:
        return load_compressed(toy[0], None, toy[1], device="cpu")
    finally:
        del os.environ["DRINKME_PREFIX_SLOTS"]


# ------------------------------------------------------------ (1) bytes --


def test_decoded_weights_are_bit_exact_to_the_codec_oracle(toy, mlx_comp):
    from drinkme.codec.pack import iter_pack_dir
    from drinkme.codec.radix_pack import decode_back_radix

    n = 0
    for name, pack in iter_pack_dir(toy[1]):
        lin = _module_at(mlx_comp.model, name)
        assert isinstance(lin, RadixLinear), name
        assert np.array_equal(lin.decode(), decode_back_radix(pack)), name
        n += 1
    assert n == 10


def test_every_packed_linear_was_swapped_and_nothing_else(toy, mlx_comp):
    from drinkme.codec.pack import load_pack_dir

    packed = set(load_pack_dir(toy[1])[0])
    swapped = set()
    for i, layer in enumerate(mlx_comp.model.model.layers):
        for sub, names in (("self_attn", ("q_proj", "k_proj", "v_proj", "o_proj")),
                           ("mlp", ("gate_proj", "up_proj", "down_proj"))):
            for nm in names:
                mod = getattr(getattr(layer, sub), nm)
                path = f"model.layers.{i}.{sub}.{nm}"
                if isinstance(mod, RadixLinear):
                    swapped.add(path)
                else:  # k/v (512 wide) stay bf16 straight from the checkpoint
                    assert mod.weight.dtype == mx.bfloat16, path
    if isinstance(mlx_comp.model.lm_head, RadixLinear):
        swapped.add("lm_head")
    assert swapped == packed
    assert mlx_comp.model.model.embed_tokens.weight.dtype == mx.bfloat16


# ----------------------------------------------------------- (2) logits --


def test_prompt_logits_within_bf16_tolerance_and_argmax_identical(mlx_comp, hf_comp):
    ids = mlx_comp.tokenize(messages=MSGS)
    assert ids == hf_comp.tokenize(messages=MSGS)
    lm = mlx_comp.prefill_logits(ids)
    with torch.inference_mode():
        lt = hf_comp.model(torch.tensor([ids])).logits[0, -1].float().numpy()
    assert lm.shape == lt.shape
    # both arms round to bf16 at every Linear; two accumulation orders agree
    # to a few bf16 ulps at the logit scale — measured 1 ulp (0.0078 at
    # |logit| 1.79) on this toy, asserted at 4
    ulp = 2.0 ** (np.floor(np.log2(max(np.abs(lt).max(), 1.0))) - 7)
    assert np.abs(lm - lt).max() <= 4 * ulp
    assert int(lm.argmax()) == int(lt.argmax())


# ------------------------------------------------------- (3) transcript --


def test_greedy_generation_matches_the_torch_engine(mlx_comp, hf_comp):
    rm, dm = run(mlx_comp, greedy())
    rt, _ = run(hf_comp, greedy())
    assert rt.finish_reason == "length" and rm.finish_reason == "length"
    assert rm.completion_tokens == rt.completion_tokens == 24
    assert rm.prompt_tokens == rt.prompt_tokens
    assert "".join(dm) == rm.text
    if rm.text != rt.text:
        # the near-tie caveat (AGENTS.md): the invariant is bit-identical
        # weights, not identical transcripts. Assert the common prefix is
        # real and name the fork, rather than fail on a coin-flip.
        m, t = rm.text.split(), rt.text.split()
        k = next(i for i, (a, b) in enumerate(zip(m, t)) if a != b)
        assert k >= 2, f"forked at token {k}: mlx {m[:k + 1]} vs torch {t[:k + 1]}"
        pytest.xfail(f"greedy forked at token {k} (near-tie); prefix of {k} tokens agrees")
    assert rm.text == rt.text


def test_stop_and_max_tokens_and_abort_through_the_seam(mlx_comp):
    ref, _ = run(mlx_comp, greedy())
    words = ref.text.split()
    res, deltas = run(mlx_comp, greedy(stop=[words[2]]))
    assert res.finish_reason == "stop" and res.stop_sequence == words[2]
    assert words[2] not in res.text and ref.text.startswith(res.text)
    res, _ = run(mlx_comp, greedy(max_tokens=3))
    assert res.finish_reason == "length" and res.completion_tokens == 3
    got = []
    res = complete(mlx_comp, GenerationRequest(MSGS, greedy()),
                   lambda d: got.append(d) or len(got) < 2)
    # the abort contract (engine.py): the delta the callback REFUSED was not
    # delivered, so it is not in the text; only what got through is
    assert res.finish_reason == "abort" and res.text == "".join(got[:-1])
    assert res.cached_tokens == 0  # v0: no prefix cache, and it says so


def test_the_event_stream_is_the_torch_engines_protocol(mlx_comp, hf_comp):
    """On this engine: StreamStart first, carrying the
    prompt's shape and its open-think verdict (the toy template opens
    nothing), Delta per delivered piece, Finished last; count_tokens is the
    same render; send(False) aborts. The torch engine over the same toy
    yields the same shape, so http.py's one consumer serves both."""
    req = GenerationRequest(MSGS, greedy(max_tokens=4),
                            template_kwargs={"enable_thinking": False})
    for eng in (mlx_comp, hf_comp):
        evs = list(eng.generate(req))
        assert isinstance(evs[0], StreamStart) and isinstance(evs[-1], Finished)
        assert all(isinstance(e, Delta) for e in evs[1:-1])
        assert evs[0] == StreamStart(prompt_tokens=eng.count_tokens(req),
                                     opens_think=False, cached_tokens=0)
        assert evs[-1].result.prompt_tokens == evs[0].prompt_tokens
        assert "".join(e.text for e in evs[1:-1]) == evs[-1].result.text
        stream = eng.generate(req)
        assert isinstance(next(stream), StreamStart)
        ev = stream.send(True)
        while isinstance(ev, Delta):
            ev = stream.send(False)
        assert isinstance(ev, Finished) and ev.result.finish_reason == "abort"
    with pytest.raises(TypeError):  # the old shape fails at the call
        list(mlx_comp.generate(MSGS, greedy(), lambda _d: True))


def test_control_close_marker_ends_the_turn_on_the_mlx_loop(mlx_comp, monkeypatch):
    """serving/control.py on THIS loop, mirrored from test_serving_engines:
    the toy under a vocabulary whose gemma markers are special ids, the
    sampler scripted to a real gemma stream — the call is lifted,
    the turn ends at the close, the prose after it never exists."""
    from test_serving_control import CALL_THEN_PROSE, HILO, WEATHER_TOOLS, fake_gemma_tokenizer, ids_of

    from drinkme.serving import engine_mlx
    from drinkme.serving.engine_mlx import MLXEngine

    tok = fake_gemma_tokenizer()
    eng = MLXEngine(mlx_comp.model, tok, model_id=mlx_comp.model_id, arm=mlx_comp.arm,
                    meta=mlx_comp.meta, ctx=mlx_comp.ctx, path="reference")
    assert eng.capability.tool_format == "gemma"
    assert eng.control.stop_after == {tok.convert_tokens_to_ids("<tool_call|>")}
    it = iter(ids_of(tok, CALL_THEN_PROSE))
    monkeypatch.setattr(engine_mlx, "sample_next", lambda *a, **k: next(it))
    res = complete(eng, GenerationRequest(MSGS, greedy(max_tokens=64), tools=WEATHER_TOOLS))
    assert res.tool_calls == HILO and res.finish_reason == "tool_calls"
    assert res.text == ""
    assert res.completion_tokens == CALL_THEN_PROSE.index("<tool_call|>") + 2


def test_seeded_sampling_reproduces_and_penalties_move_the_draw(mlx_comp):
    p = dict(temperature=0.8, top_p=0.9, top_k=10, seed=7)
    a, _ = run(mlx_comp, SampleParams(max_tokens=16, **p))
    b, _ = run(mlx_comp, SampleParams(max_tokens=16, **p))
    assert a.text == b.text
    c, _ = run(mlx_comp, SampleParams(max_tokens=16, repetition_penalty=1.5,
                                      presence_penalty=0.5, **p))
    assert c.completion_tokens == 16  # the penalised path runs to length too


def test_the_host_sampler_matches_serving_sampling_on_a_fixed_row():
    """sample_next here is a numpy re-statement of sampling.sample_next; on
    one row with every knob set, the greedy pick and the SHAPED distribution
    must agree with the torch original. The draw itself uses a different
    generator and is not compared."""
    if torch is None:
        pytest.skip("serving.sampling is the torch pipeline; no torch on this machine")
    from drinkme.serving import sampling

    rng = np.random.default_rng(3)
    row = (rng.standard_normal(64) * 3).astype(np.float32)
    prev, gen = [1, 5, 9, 9, 40], [9, 9, 40]
    params = SampleParams(temperature=0.7, top_k=12, top_p=0.85,
                          repetition_penalty=1.3, presence_penalty=0.4,
                          frequency_penalty=0.2)
    greedy_p = SampleParams(temperature=0.0, repetition_penalty=1.3,
                            presence_penalty=0.4, frequency_penalty=0.2)
    assert sample_next(row, greedy_p, None, prev, gen) == sampling.sample_next(
        torch.tensor(row), greedy_p, None, prev_ids=prev, gen_ids=gen)
    ours = sampling.sample_probs(torch.tensor(row), params, prev_ids=prev, gen_ids=gen).numpy()
    from drinkme.serving.engine_mlx import _penalize, _shape

    r = _penalize(row, params, prev, gen)
    _shape(r, params)
    pr = np.exp(r - r.max())
    pr /= pr.sum()
    assert np.allclose(pr, ours, atol=1e-6)


# ------------------------------------------------------------ (4) HTTP --


def test_http_through_the_real_mlx_engine(mlx_comp):
    from drinkme.serving.http import start_server

    ref, _ = run(mlx_comp, greedy())
    srv = start_server(mlx_comp, "127.0.0.1", 0)
    try:
        port = srv.server_address[1]

        def call(method, route, body=None):
            c = http.client.HTTPConnection("127.0.0.1", port, timeout=60)
            c.request(method, route, None if body is None else json.dumps(body),
                      {"Content-Type": "application/json"})
            r = c.getresponse()
            data = r.read()
            c.close()
            return r, data

        req = {"messages": MSGS, "temperature": 0, "max_tokens": 6}
        r, body = call("POST", "/v1/chat/completions", req)
        obj = json.loads(body)
        assert r.status == 200, body
        content = obj["choices"][0]["message"]["content"]
        assert obj["choices"][0]["finish_reason"] == "length"
        assert content and ref.text.startswith(content)
        assert obj["usage"]["completion_tokens"] == 6

        r, body = call("POST", "/v1/chat/completions", {**req, "stream": True})
        assert r.status == 200
        events = [b[len("data: "):] for b in body.decode().split("\n\n")
                  if b.startswith("data: ")]
        assert events[-1] == "[DONE]"
        chunks = [json.loads(e) for e in events[:-1]]
        text = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks)
        assert text == content

        r, body = call("GET", "/v1/models")
        m = json.loads(body)["data"][0]["drinkme"]
        assert m["runtime"] == "mlx" and m["arm"] == "compressed"
        # the device line is honest either way: a CPU-only mlx never dressed
        # as a Mac, a Mac named as itself (the reference path runs on both)
        if mx.metal.is_available():
            assert not m["device"].startswith("cpu"), m["device"]
        else:
            assert m["device"].startswith("cpu (mlx"), m["device"]
        assert m["computePath"] == "reference"

        r, body = call("GET", "/health")
        assert r.status == 200

        # the seam's optional methods: not implemented on v0, and the HTTP
        # layer says so legibly rather than tracebacking
        r, body = call("POST", "/sleep?level=1")
        assert r.status == 501, body
    finally:
        srv.shutdown()


# ------------------------------------- (5) the shipped torch reference --


def _mlx_transcript(eng, prompts, max_tokens):
    """torch_transcript's twin on this engine: prompt ids, prefill_logits,
    and the greedy continuation observed off the engine's own loop
    (Recorder wraps engine_mlx.sample_next: the id picked at every step
    and the row it was picked from)."""
    from drinkme.serving import engine_mlx

    out = []
    for msgs in prompts:
        ids = eng.tokenize(messages=msgs)
        prompt_logits = eng.prefill_logits(ids)
        with Recorder(engine_mlx, lambda r: np.array(r, dtype=np.float32, copy=True)) as rec:
            res = complete(eng, GenerationRequest(msgs, greedy(max_tokens)))
        out.append({"ids": np.asarray(ids), "prompt_logits": prompt_logits,
                    "gen_ids": np.asarray(rec.ids), "step_logits": np.stack(rec.rows),
                    "text": res.text, "finish": res.finish_reason})
    return out


def _assert_matches_reference(rows, ref, label):
    """An engine's transcript against the reference's, prompt by prompt:
    the same ids (tokenization), prompt logits within 4 bf16 ulps at the
    logit scale — the torch-vs-mlx tolerance of (2) — with the argmax
    identical; then the greedy continuation, every step's row within the
    same tolerance up to the first fork, and the ids identical — a fork is
    tolerated only where the reference row is a near-tie (its top-2 gap
    inside the tolerance: AGENTS.md's caveat, two accumulation orders),
    reported as an xfail naming the step; a fork at a clear margin fails."""
    assert len(rows) == len(ref["rows"])
    for i, (got, want) in enumerate(zip(rows, ref["rows"])):
        tag = f"{label} prompt {i}"
        assert got["ids"].tolist() == want["ids"].tolist(), f"{tag}: tokenization differs"
        u = ulp(want["prompt_logits"])
        d = float(np.abs(got["prompt_logits"] - want["prompt_logits"]).max())
        assert d <= 4 * u, f"{tag}: prompt logits differ by {d:.5f} = {d / u:.1f} ulp"
        assert int(got["prompt_logits"].argmax()) == int(want["prompt_logits"].argmax()), tag
        g, w = got["gen_ids"].tolist(), want["gen_ids"].tolist()
        k = next((j for j, (a, b) in enumerate(zip(g, w)) if a != b), min(len(g), len(w)))
        for j in range(k):
            u = ulp(want["step_logits"][j])
            d = float(np.abs(got["step_logits"][j] - want["step_logits"][j]).max())
            assert d <= 4 * u, f"{tag} step {j}: logits differ by {d:.5f} = {d / u:.1f} ulp"
        if g != w:
            assert k < len(w) and k < len(g), f"{tag}: lengths differ, {len(g)} vs torch {len(w)}"
            row = want["step_logits"][k]
            top = np.sort(row)[-2:]
            gap = float(top[1] - top[0])
            msg = (f"{tag}: greedy forked at step {k}: {g[:k + 1]} vs torch {w[:k + 1]}; "
                   f"the torch row's top-2 gap is {gap:.5f} = {gap / ulp(row):.1f} ulp")
            if gap <= 4 * ulp(row):
                pytest.xfail("near-tie: " + msg)
            pytest.fail(msg)
        assert got["text"] == want["text"], tag
        assert got["finish"] == want["finish"], tag


def _biased_modules(model):
    """(path, module) for every Linear of the tree holding a bias."""
    out = []
    for i, layer in enumerate(model.model.layers):
        for sub, names in (("self_attn", ("q_proj", "k_proj", "v_proj", "o_proj")),
                           ("mlp", ("gate_proj", "up_proj", "down_proj"))):
            for nm in names:
                mod = getattr(getattr(layer, sub), nm)
                if "bias" in mod:
                    out.append((f"model.layers.{i}.{sub}.{nm}", mod))
    if "bias" in model.lm_head:
        out.append(("lm_head", model.lm_head))
    return out


def _assert_biased_tree(model, packed_kind):
    """The biased toy's tree: q/o (packed, hidden 1024) are `packed_kind`
    with the checkpoint's bias attached after the swap; k/v (512 wide,
    unpacked) are BiasedLinear — mlx-lm built them bias=False — with the
    weight and bias both bf16; eight biases, none of them zero, nothing
    else biased."""
    mods = _biased_modules(model)
    assert sorted(p.rsplit(".", 1)[1] for p, _ in mods) == sorted(
        ["q_proj", "k_proj", "v_proj", "o_proj"] * 2), [p for p, _ in mods]
    for path, mod in mods:
        rows = mod.R if hasattr(mod, "R") else mod["weight"].shape[0]
        assert mod["bias"].dtype == mx.bfloat16 and mod["bias"].shape == (rows,), path
        assert float(mx.abs(mod["bias"].astype(mx.float32)).sum().item()) > 0, path
        if path.endswith(("q_proj", "o_proj")):
            assert isinstance(mod, packed_kind), path
        else:
            assert isinstance(mod, BiasedLinear) and mod["weight"].dtype == mx.bfloat16, path


@pytest.mark.parametrize("path", ["reference", pytest.param("fused", marks=needs_metal)])
def test_engine_matches_the_shipped_torch_reference(toy_reference, path):
    """The receipt for the bias path (and the plain one): the MLX engine on
    each compute path against the torch CPU engine's transcript, on the toy
    of each kind. The engine reads the same generation defaults and eos
    set the reference was built with (asserted), and on the biased twin
    the tree is the biased one (_assert_biased_tree) — so the comparison
    holds RadixLinear-with-bias on the fused GEMV (decode) and the dense
    arm (prefill), the reference path's matmul, and BiasedLinear."""
    from drinkme.serving import gen_config

    kind, model_dir, pack_dir, ref = toy_reference
    eng = load_compressed_mlx(model_dir, None, pack_dir, path=path)
    assert eng.path == path
    assert set(eng.eos_ids) == set(ref["eos_ids"])
    assert gen_config.effective_defaults(eng.sampling_defaults) == ref["sampling_defaults"]
    if kind == "bias":
        _assert_biased_tree(eng.model, RadixLinear)
    else:
        assert _biased_modules(eng.model) == []
    rows = _mlx_transcript(eng, ref["prompts"], ref["max_tokens"])
    _assert_matches_reference(rows, ref, f"{kind}/{path}")


def test_the_live_torch_engine_matches_the_shipped_reference(toy_reference, monkeypatch):
    """Where torch is present: the control itself against the file the Mac
    gets — the same engine the reference was recorded from, so bitwise on
    the same box and within the tolerance across torch builds. What makes
    reference.npz evidence rather than a claim. The serial loop and no
    prefix cache, as write_reference runs it (monkeypatch: the session
    holds DRINKME_SPEC=mtp, conftest.py, and must get it back)."""
    if torch is None:
        pytest.skip("the torch engine is the control here; no torch on this machine")
    from radix_mlx_toy_build import torch_transcript

    from drinkme.serving.engines import load_compressed

    kind, model_dir, pack_dir, ref = toy_reference
    monkeypatch.setenv("DRINKME_PREFIX_SLOTS", "0")
    monkeypatch.setenv("DRINKME_SPEC", "off")
    hf = load_compressed(model_dir, None, pack_dir, device="cpu")
    rows = torch_transcript(hf, ref["prompts"], ref["max_tokens"])
    _assert_matches_reference(rows, ref, f"{kind}/torch")


def test_a_reference_for_other_bytes_is_refused(toy_reference, tmp_path):
    """load_reference binds the file to the checkpoint's bytes AND the
    pack's manifest digest: a checkpoint dir whose config differs by a
    byte, or a pack whose meta.json's tensor map moved, is named and
    refused — never compared against."""
    import shutil

    kind, model_dir, pack_dir, _ = toy_reference
    path = os.path.join(os.path.dirname(model_dir), REFERENCE)
    if not os.path.exists(path):
        pytest.skip("the reference was written to a temp dir here; the refusal needs the file")
    other_model = tmp_path / "model"
    shutil.copytree(model_dir, other_model)
    with open(other_model / "config.json", "a") as f:
        f.write("\n")
    with pytest.raises(ValueError, match="checkpoint_sha256.*rebuild it with --reference"):
        load_reference(path, str(other_model), pack_dir)
    other_pack = tmp_path / "pack"
    other_pack.mkdir()
    meta = json.load(open(os.path.join(pack_dir, "meta.json")))
    meta["tensors"] = {k: v for k, v in list(meta["tensors"].items())[1:]}
    json.dump(meta, open(other_pack / "meta.json", "w"))
    with pytest.raises(ValueError, match="pack_manifest_sha256"):
        load_reference(path, model_dir, str(other_pack))
    load_reference(path, model_dir, pack_dir)  # the real pair: accepted


def test_stock_arm_carries_the_biases_and_is_bitwise_the_compressed_reference_arm(biased_toy):
    """load_stock_mlx on the biased checkpoint: every biased projection is
    a BiasedLinear (mlx-lm's own tree has no slot; the loader's swap-in is
    both arms'), and against the compressed arm on the reference path the
    whole transcript — prompt logits and every greedy decode row, M>1 and
    M=1 — is BITWISE on Metal: the same bytes (the codec's decode is the
    checkpoint's bf16), the same matmul, the same biased epilogue —
    tests/test_codec_bias_rounding.py's loader-level pin, on this lane
    (measured on an M4; the CPU build of mlx was not, and is
    held to the bf16 floor)."""
    model_dir, pack_dir = biased_toy
    stock = load_stock_mlx(model_dir, None)
    _assert_biased_tree(stock.model, BiasedLinear)
    comp = load_compressed_mlx(model_dir, None, pack_dir, path="reference")
    prompts = [MSGS, [{"role": "user", "content": "alpha beta gamma delta"}]]
    a, b = _mlx_transcript(stock, prompts, 12), _mlx_transcript(comp, prompts, 12)
    for x, y in zip(a, b):
        assert x["gen_ids"].tolist() == y["gen_ids"].tolist() and len(x["gen_ids"]) == 12
        if mx.metal.is_available():
            assert np.array_equal(x["prompt_logits"], y["prompt_logits"])
            assert np.array_equal(x["step_logits"], y["step_logits"])
        else:
            assert np.abs(x["prompt_logits"] - y["prompt_logits"]).max() <= 4 * ulp(x["prompt_logits"])
            for j in range(12):
                assert np.abs(x["step_logits"][j] - y["step_logits"][j]).max() <= 4 * ulp(x["step_logits"][j])


# ------------------------------------------------------------- refusals --


def test_hybrid_and_unknown_families_are_refused_by_name():
    with pytest.raises(ValueError, match=r"hybrid DeltaNet.*27B"):
        refuse_unsupported({"model_type": "qwen3_5"})
    with pytest.raises(ValueError, match=r"llama.*not one the mlx runtime serves"):
        refuse_unsupported({"model_type": "llama"})
    refuse_unsupported({"model_type": "qwen3"})


def test_make_module_mlx_dispatches_by_kind_and_refuses_the_rest():
    """The loader's one constructor: a radix dict is RadixLinear's, a raw
    dict RawLinear's; a dict naming no kind (an FP8 pack's `dtype`
    scalar included) is refused by name with the registry's note, and each
    class refuses the other kinds' dicts by name rather than on a missing
    key."""
    from drinkme.codec import radix

    rng = np.random.default_rng(0)
    bits = (rng.standard_normal((16, 2048)) * 0.02).astype(np.float32).view(np.uint32) >> 16
    bits = bits.astype(np.uint16)
    rpk = radix.pack_array(bits, compression_profile="sip")
    rdict = {"rx_palette": rpk.palette, "rx_offsets": rpk.offsets.astype(np.uint32),
             "rx_data": rpk.data.astype(np.uint32), "R": 16, "C": 2048, "bpw": 11.0,
             "codec": "radix", "profile": "sip", "widths": list(rpk.widths),
             "block_size": 1024, "layout": 0, "format_version": 1}
    assert isinstance(make_module_mlx(rdict, path="reference", name="a"), RadixLinear)
    assert isinstance(make_module_mlx({"raw_bits": bits, "R": 16, "C": 2048, "bpw": 16.0,
                                       "codec": "raw"}, path="reference", name="b"), RawLinear)
    with pytest.raises(ValueError, match=r"no MLX path for model.layers.0.mlp.down_proj \(unknown tensor kind"):
        make_module_mlx({"format_version": 1, "layout": 0}, name="model.layers.0.mlp.down_proj")
    with pytest.raises(ValueError, match=r"no MLX path for .*unknown tensor kind"):
        make_module_mlx({"dtype": "fp8_e4m3", "format_version": 1, "layout": 0}, name="model.layers.0.mlp.down_proj")
    with pytest.raises(ValueError, match="RadixLinear serves the radix codec only"):
        RadixLinear({"raw_bits": bits, "R": 16, "C": 2048, "bpw": 16.0, "codec": "raw"},
                    path="reference", name="x")
    with pytest.raises(ValueError, match="RawLinear serves the raw fallback only"):
        RawLinear(rdict, path="reference", name="x")


def test_raw_fallback_arm_is_the_bf16_bits_through_mlx_matmul():
    """RawLinear (codec: "raw" — the tensor radix would have expanded): the
    resident weight IS the stored bits, resident_bytes is 2 B/weight, and
    the forward is mx's own matmul over them, on either path (there is no
    fused arm to switch to): with a bias, mx.addmm at M>1 and the fp32
    route at M=1 — biased_matmul's two arms, bitwise (the rounding claim
    itself is test_a_biased_linear_rounds_once's)."""
    rng = np.random.default_rng(1)
    w = (rng.standard_normal((24, 64)) * 0.1).astype(np.float32)
    bits = (w.view(np.uint32) >> 16).astype(np.uint16)
    bias = mx.array((rng.standard_normal(24) * 0.1).astype(np.float32)).astype(mx.bfloat16)
    for path in ("reference", "fused"):
        lin = make_module_mlx({"raw_bits": bits, "R": 24, "C": 64, "bpw": 16.0, "codec": "raw"},
                              bias, path=path, name="raw")
        assert isinstance(lin, RawLinear) and lin.tensor_codec == "raw" and lin.path == path
        assert np.array_equal(lin.decode(), bits)
        assert lin.resident_bytes() == 2 * 24 * 64
        W = mx.array(bits).view(mx.bfloat16)
        for shape in ((1, 64), (5, 64), (2, 3, 64)):
            x = mx.array(rng.standard_normal(shape).astype(np.float32)).astype(mx.bfloat16)
            got = lin(x)
            xf = x.reshape(-1, 64)
            if xf.shape[0] == 1:
                want = (xf.astype(mx.float32) @ W.astype(mx.float32).T
                        + bias.astype(mx.float32)).astype(mx.bfloat16)
            else:
                want = mx.addmm(bias, xf, W.T)
            want = want.reshape(*shape[:-1], 24)
            mx.eval(got, want)
            assert got.shape == (*shape[:-1], 24)
            assert np.array_equal(np.array(got.view(mx.uint16)), np.array(want.view(mx.uint16)))
    with pytest.raises(ValueError, match=r"raw_bits is \(24, 64\), not \[R, C\]"):
        RawLinear({"raw_bits": bits, "R": 64, "C": 24, "bpw": 16.0, "codec": "raw"}, name="raw")


def _radix_dict(bits: np.ndarray) -> dict:
    """A radix pack dict (iter_pack_dir's shape) for bf16 bits [R, C], C a
    multiple of the 1024-weight block, at the sip profile."""
    from drinkme.codec import radix

    R, C = bits.shape
    pk = radix.pack_array(bits, compression_profile="sip")
    return {"rx_palette": pk.palette, "rx_offsets": pk.offsets.astype(np.uint32),
            "rx_data": pk.data.astype(np.uint32), "R": R, "C": C, "bpw": 11.0,
            "codec": "radix", "profile": "sip", "widths": list(pk.widths),
            "block_size": 1024, "layout": 0, "format_version": 1}


def test_a_biased_linear_rounds_once():
    """The bias epilogue, at unit level, on every module this engine builds:
    tests/test_codec_bias_rounding.py's cancellation. Weight columns 0 and
    1 are 1.0, the rest 0; x[0] = 1, x[1] = 1/256; bias -1. The fp32
    accumulator is exactly 1.00390625 and the bias cancels it to 2^-8 =
    0.00390625, a bf16 value — which a rounding BEFORE the bias loses
    (1.00390625 -> 1.0, minus 1 -> 0.0). Every element must be 0.00390625
    on RadixLinear (the reference path's matmul; on a Mac the fused path
    too: gemv_radix at M=1, the dense arm at M>1), RawLinear and
    BiasedLinear, at M = 1 (mlx's gemv route) and 2, 3, 9 (its gemm).

    And the second rounding is shown REAL where this box's mlx has it:
    mx.addmm at M == 1 on Metal (0.0 — measured with mlx 0.32.2 on an M4;
    the CPU build rounds once), so biased_matmul's fp32 route for the
    single row is a measured fix, not a precaution: without it this test
    fails on an M4 at M == 1 for RadixLinear/reference, RawLinear and
    BiasedLinear."""
    R, C = 32, 1024
    w = np.zeros((R, C), dtype=np.float32)
    w[:, :2] = 1.0
    bits = (w.view(np.uint32) >> 16).astype(np.uint16)
    W = mx.array(bits).view(mx.bfloat16)
    bias = mx.full((R,), -1.0, dtype=mx.bfloat16)
    rdict = _radix_dict(bits)
    mods = [("radix/reference", RadixLinear(rdict, bias, path="reference", name="r")),
            ("raw", RawLinear({"raw_bits": bits, "R": R, "C": C, "bpw": 16.0, "codec": "raw"},
                              bias, path="reference", name="w")),
            ("biased", BiasedLinear(W, bias))]
    if mx.metal.is_available():
        mods.append(("radix/fused", RadixLinear(rdict, bias, path="fused", name="f")))
    assert np.array_equal(mods[0][1].decode(), bits)  # the codec round-trips the construction

    def rows(m):
        x = mx.zeros((m, C), dtype=mx.bfloat16)
        x[:, 0] = 1.0
        x[:, 1] = 1 / 256
        return x

    for m in (1, 2, 3, 9):
        x = rows(m)
        for name, mod in mods:
            y = mod(x)
            mx.eval(y)
            assert y.dtype == mx.bfloat16 and y.shape == (m, R), (name, m)
            got = np.array(y.astype(mx.float32))
            assert np.all(got == 0.00390625), (
                f"{name} at M={m}: {int((got != 0.00390625).sum())} of {got.size} elements "
                f"are not 2^-8 (got {got[0, 0]}) — a second rounding")
    # the oracle is a single rounding; the post-add composition loses the residual
    x = rows(1)
    once = (x.astype(mx.float32) @ W.astype(mx.float32).T + bias.astype(mx.float32)).astype(mx.bfloat16)
    twice = (x @ W.T) + bias
    mx.eval(once, twice)
    assert once[0, 0].item() == 0.00390625 and twice[0, 0].item() == 0.0
    # and biased_matmul at M=1 is the fp32 route, at M>1 mx.addmm — bitwise
    y1 = biased_matmul(x, W, bias)
    y3 = biased_matmul(rows(3), W, bias)
    a3 = mx.addmm(bias, rows(3), W.T)
    mx.eval(y1, y3, a3)
    assert np.array_equal(np.array(y1.view(mx.uint16)), np.array(once.view(mx.uint16)))
    assert np.array_equal(np.array(y3.view(mx.uint16)), np.array(a3.view(mx.uint16)))
    if mx.metal.is_available():
        # where the fix is measured: mlx's own addmm at M == 1 rounds twice
        a1 = mx.addmm(bias, x, W.T)
        mx.eval(a1)
        assert a1[0, 0].item() == 0.0, "mlx's M=1 addmm rounds once now — biased_matmul's fp32 route can retire"


def test_transformers5_rope_parameters_are_lifted_for_mlx_lm():
    new = {"model_type": "qwen3", "rope_parameters": {"rope_theta": 5.0, "rope_type": "default"}}
    out = normalize_config(new)
    assert out["rope_theta"] == 5.0 and out["rope_scaling"] is None
    yarn = normalize_config({"rope_parameters": {"rope_theta": 5.0, "rope_type": "yarn", "factor": 4}})
    assert yarn["rope_scaling"] == {"rope_type": "yarn", "factor": 4}
    old = {"rope_theta": 7.0, "rope_scaling": None}
    assert normalize_config(old) == old


def test_fit_fit_check_refuses_with_the_arithmetic_and_skips_without_metal(capsys):
    gib = 1024 ** 3
    fit_check(resident=10 * gib, kv_per_token=1024, ctx=8192, working_set=None)
    assert "skipped" in capsys.readouterr().err
    fit_check(resident=10 * gib, kv_per_token=1024, ctx=8192, working_set=20 * gib)
    assert "fit check (measured): ok" in capsys.readouterr().err
    with pytest.raises(SystemExit, match=r"refusing to load \(measured\).*resident 10\.00 GiB.*working set 11\.00 GiB"):
        fit_check(resident=10 * gib, kv_per_token=1024, ctx=8192, working_set=11 * gib)
    # the estimate stage (tests/test_serving_engine_mlx_fit_check.py has the
    # ordering): the one staged tensor is charged beside resident
    with pytest.raises(SystemExit, match=r"refusing to load \(estimate\).*transient 2\.00 GiB"):
        fit_check(resident=8 * gib, kv_per_token=1024, ctx=8192, working_set=11 * gib,
                      transient=2 * gib, stage="estimate")


def test_rope_scaling_is_refused_not_ignored(toy):
    with pytest.raises(ValueError, match="rope-scaling"):
        load_compressed_mlx(toy[0], None, toy[1], rope_scaling={"rope_type": "yarn", "factor": 2})


def test_stock_arm_loads_the_same_checkpoint_uncompressed(toy, mlx_comp):
    stock = load_stock_mlx(toy[0], None)
    assert isinstance(stock, MLXEngine) and stock.arm == "stock"
    assert not any(isinstance(m, (RadixLinear, RawLinear)) for m in
                   [stock.model.lm_head] + [l.mlp.down_proj for l in stock.model.model.layers])
    ids = stock.tokenize(messages=MSGS)
    a, b = stock.prefill_logits(ids), mlx_comp.prefill_logits(ids)
    # same bytes, decoded vs read: the two must agree to the bf16 floor
    ulp = 2.0 ** (np.floor(np.log2(max(np.abs(a).max(), 1.0))) - 7)
    assert np.abs(a - b).max() <= 4 * ulp
    assert int(a.argmax()) == int(b.argmax())


# ------------------------------------------------- the runtime switch --


def test_resolve_runtime_precedence(monkeypatch):
    from drinkme import serve

    monkeypatch.delenv("DRINKME_RUNTIME", raising=False)
    assert serve.resolve_runtime("mlx") == "mlx"
    monkeypatch.setenv("DRINKME_RUNTIME", "mlx")
    assert serve.resolve_runtime(None) == "mlx"
    assert serve.resolve_runtime("torch") == "torch"  # the flag beats the env
    monkeypatch.delenv("DRINKME_RUNTIME")
    import platform

    monkeypatch.setattr(platform, "system", lambda: "Darwin")
    monkeypatch.setattr(platform, "machine", lambda: "arm64")
    assert serve.resolve_runtime(None) == "mlx"
    monkeypatch.setattr(platform, "system", lambda: "Linux")
    assert serve.resolve_runtime(None) == "torch"
    with pytest.raises(SystemExit, match="not 'torch' or 'mlx'"):
        serve.resolve_runtime("cuda")


def test_serve_build_engine_forced_to_mlx_on_the_toy(toy, monkeypatch):
    """serve.build_engine's runtime switch: --runtime mlx on a Linux box
    loads the MLX engine (reference path, CPU) through the SAME entry the
    server boots through, both arms; the seam's unsupported knobs refuse."""
    from drinkme import serve

    monkeypatch.setenv("DRINKME_NO_WARMUP", "1")
    eng = serve.build_engine(toy[0], None, toy[1], stock=False, runtime="mlx")
    assert isinstance(eng, MLXEngine) and eng.arm == "compressed"
    eng = serve.build_engine(toy[0], None, None, stock=True, runtime="mlx")
    assert isinstance(eng, MLXEngine) and eng.arm == "stock"
    with pytest.raises(SystemExit, match="prefix-slots"):
        serve.build_engine(toy[0], None, toy[1], stock=False, runtime="mlx", prefix_slots=2)


def test_cli_serve_and_bench_take_a_runtime_flag():
    import argparse

    from drinkme import cli

    # parse only: the verbs themselves need a box; a bad value must be argparse's refusal
    with pytest.raises(SystemExit):
        cli.main(["serve", "--model", "x", "--runtime", "cuda"])
    ap = argparse.ArgumentParser()
    ap.add_argument("--runtime", choices=["torch", "mlx"], default=None)
    assert ap.parse_args(["--runtime", "mlx"]).runtime == "mlx"


# ----------------------------------------------------- the bench arms --


def test_bench_arms_dry_run_on_the_toy(toy):
    """`drinkme bench --runtime mlx`'s three passes, on the toy, every arm on
    the reference path (no Metal here): the record must carry arms.py's
    field names, the min/drift table, the prefill/ttft names on the
    stock and compressed arms (never the twin), and name the
    compressed path so a CPU number can never wear the fused label.

    roundtripBitExact and the ulp/token-match fidelity block are gone —
    the former is a write-time gate (a run that reaches this point already
    passed it), the latter was retired outright."""
    from drinkme.arms_mlx import run_arms

    # prefill_len well under the toy's max_position_embeddings=256 (test_serving_engine_mlx's
    # fixture): the fixed-512 default is a production invariant, not a test one.
    r = run_arms(toy[0], None, "hello world the quick", pack_dir=toy[1],
                 fused=False, n_new=8, reps=2, prefill_len=64)
    for k in ("stock_decode_tok_s", "twin_decode_tok_s",
              "compressed_decode_tok_s", "bandwidth", "mean_bpw",
              "vram_bf16_bytes", "vram_compressed_bytes", "verified_tensors",
              "ratio_min_of_n", "drift_band_pct", "prefill_prompt_len",
              "stock_prefill_tok_s", "stock_prefill_samples", "stock_ttft_s",
              "stock_ttft_samples", "compressed_prefill_tok_s",
              "compressed_prefill_samples", "compressed_ttft_s",
              "compressed_ttft_samples"):
        assert k in r, k
    for dropped in ("all_tensors_bitwise_roundtrip", "logit_delta_bf16_ulps",
                    "greedy_token_match_pct", "first_divergence", "text_compressed",
                    # the twin has no dense route at M>1 — no
                    # prefill/TTFT timing is left on it, torch or mlx.
                    "twin_prefill_tok_s", "twin_prefill_samples",
                    "twin_ttft_s", "twin_ttft_samples"):
        assert dropped not in r, dropped
    assert r["compressed_path"] == "reference" and r["runtime"] == "mlx"
    assert r["swapped_linears"] == 10 and r["verified_tensors"] == 10
    assert len(r["compressed_decode_samples"]) == 2
    assert r["prefill_prompt_len"] == 64
    assert len(r["stock_prefill_samples"]) == 2 and len(r["stock_ttft_samples"]) == 2
    assert r["bandwidth"]["read_bytes_s"] > 0 and r["bandwidth"]["probe_bytes"] > 0
    assert r["compression_profile"] == "sip"  # off the loaded pack's meta


def test_bench_arms_return_mlxs_cache_between_arms(toy, monkeypatch):
    """Each arm is freed and MLX's cache handed back (synchronize, then
    clear_cache) before the next arm's load (whose fit check counts what is
    free), after the probe and after the last arm; outside every timed
    pass. Recorded as one event list across the real loaders."""
    import mlx.core as mx

    from drinkme import arms_mlx
    from drinkme.serving import engine_mlx

    events: list = []
    real_sync, real_clear = mx.synchronize, mx.clear_cache
    monkeypatch.setattr(mx, "synchronize", lambda *a, **k: (events.append("sync"), real_sync(*a, **k))[1])
    monkeypatch.setattr(mx, "clear_cache", lambda: (events.append("clear"), real_clear())[1])
    for name in ("load_stock_mlx", "load_compressed_mlx"):
        real = getattr(engine_mlx, name)

        def wrapped(*a, _real=real, _name=name, **k):
            events.append(_name)
            return _real(*a, **k)
        monkeypatch.setattr(engine_mlx, name, wrapped)
    arms_mlx.run_arms(toy[0], None, "hello world the quick", pack_dir=toy[1],
                      fused=False, n_new=4, reps=2, prefill_len=64)
    # a "sync","clear" pair right before each load after the first, and at the end
    loads = [i for i, e in enumerate(events) if e.startswith("load_")]
    assert len(loads) == 3
    for i in loads[1:]:
        assert events[i - 2:i] == ["sync", "clear"], events
    assert events[-2:] == ["sync", "clear"]
    for i, e in enumerate(events):
        if e == "clear":
            assert events[i - 1] == "sync"


def test_the_twin_engine_is_the_reference_engine_bit_for_bit(toy, mlx_comp):
    """The bench's twin arm (load_compressed_mlx(path=TWIN)): every radix
    tensor decoded once at load into a resident bf16 plane — the prompt
    logits are the reference engine's bitwise and the greedy transcript is
    the same, so moving the decode out of the forward changed no number the
    twin computes. Resident is the planes (2 B/weight), not the streams."""
    from drinkme.serving.engine_mlx import TWIN, RadixTwinLinear

    twin = load_compressed_mlx(toy[0], None, toy[1], path=TWIN)
    assert twin.path == TWIN and twin.model_meta()["computePath"] == TWIN
    lin = _module_at(twin.model, "model.layers.0.mlp.down_proj")
    ref = _module_at(mlx_comp.model, "model.layers.0.mlp.down_proj")
    assert isinstance(lin, RadixTwinLinear) and np.array_equal(lin.decode(), ref.decode())
    ids = twin.tokenize(messages=MSGS)
    a, b = twin.prefill_logits(ids), mlx_comp.prefill_logits(ids)
    assert np.array_equal(a.view(np.uint32), b.view(np.uint32))
    rt, _ = run(twin, greedy())
    rr, _ = run(mlx_comp, greedy())
    assert rt.text == rr.text
    packed = [m for _, m in twin.model.named_modules() if isinstance(m, RadixTwinLinear)]
    assert len(packed) == 10
    planes = sum(2 * m.R * m.C for m in packed)
    streams = sum(m.resident_bytes() for _, m in mlx_comp.model.named_modules()
                  if isinstance(m, RadixLinear))
    assert twin.resident_bytes - planes == mlx_comp.resident_bytes - streams  # the raw remainder


def test_constrained_pick_token_bans_and_resamples_on_the_host_row():
    """engine_mlx.pick_token is constrain.pick_token's loop over a numpy row.
    A made-up vocabulary of JSON pieces: the greedy favourite is an illegal
    token, so it must be banned and the next legal one picked; EOS is
    allowed only once the object is COMPLETE."""
    from drinkme.serving.constrain import JsonConstraint
    from drinkme.serving.engine_mlx import pick_token

    vocab = ['{', '"a"', ':', '1', '}', 'x', '</s>']
    decode = lambda ids: "".join(vocab[i] for i in ids)  # noqa: E731
    schema = {"type": "object", "properties": {"a": {"type": "integer"}},
              "required": ["a"], "additionalProperties": False}
    c = JsonConstraint(schema)
    eos = {6}
    params = SampleParams(temperature=0.0)
    gen: list[int] = []
    row = np.zeros(7, np.float32)
    row[5] = 10.0  # 'x' — illegal everywhere
    row[6] = 9.0   # EOS — illegal until complete
    row[0] = 1.0
    row[1] = 0.9
    row[2] = 0.8
    row[3] = 0.7
    row[4] = 0.75  # '}' beats '1', so it is tried (and banned) before the value exists
    out = []
    for _ in range(6):
        t = pick_token(row, params, None, [], gen, eos, decode, c)
        out.append(t)
        if t in eos:
            break
        gen.append(t)
    assert decode(gen) == '{"a":1}'
    assert out[-1] == 6  # EOS, only once the object closed
    assert 5 not in out


# ------------------------------------ the prefill's head, the request's memory --


def _tiny_qwen3(tie: bool):
    """A 2-layer mlx-lm Qwen3 at random bf16 weights, tied or not — the
    head wiring only; the toy above is untied."""
    from drinkme.serving.engine_mlx import build_skeleton

    model, _ = build_skeleton({"model_type": "qwen3", "hidden_size": 64, "num_hidden_layers": 2,
                               "intermediate_size": 128, "num_attention_heads": 4,
                               "num_key_value_heads": 2, "head_dim": 16, "rms_norm_eps": 1e-6,
                               "vocab_size": 97, "max_position_embeddings": 512,
                               "rope_theta": 10000.0, "tie_word_embeddings": tie})
    model.set_dtype(mx.bfloat16)
    mx.eval(model.parameters())
    return model


class _HeadSpy(nn.Module):
    """The untied head, wrapped: records the shape of what it is applied to."""

    def __init__(self, inner, seen):
        super().__init__()
        self.inner = inner
        self._seen = seen

    def __call__(self, h):
        self._seen.append(tuple(h.shape))
        return self.inner(h)


def _spy_head(monkeypatch, model) -> list:
    seen: list = []
    if model.args.tie_word_embeddings:
        emb = model.model.embed_tokens
        real = emb.as_linear
        monkeypatch.setattr(emb, "as_linear", lambda h: (seen.append(tuple(h.shape)), real(h))[1])
    else:
        monkeypatch.setattr(model, "lm_head", _HeadSpy(model.lm_head, seen))
    return seen


@pytest.mark.parametrize("tie", [True, False])
def test_the_prefill_head_reads_the_last_hidden_row_only(tie, monkeypatch):
    """_logits runs the backbone over the prompt and the head over ONE row:
    mlx-lm's Model.__call__ heads every position ([1, T, vocab] bf16 — 2.2
    GiB at an 8k prompt on Qwen3's vocabulary) to keep the last. The row is
    that full forward's last row within the bf16 floor with the same
    argmax (bitwise on MLX CPU, measured; an M = 1 head may round
    differently from row T-1 of an M = T GEMM on Metal), and at T = 1 — every
    decode step — bitwise: the same graph."""
    from mlx_lm.models.cache import make_prompt_cache

    model = _tiny_qwen3(tie)
    eng = types.SimpleNamespace(model=model)
    ids = [int(i) for i in np.random.default_rng(5).integers(1, 97, size=40)]
    old = np.array(model(mx.array([ids]), cache=make_prompt_cache(model))[0, -1].astype(mx.float32))
    seen = _spy_head(monkeypatch, model)
    new = MLXEngine._logits(eng, ids, make_prompt_cache(model))
    assert seen == [(1, 1, 64)]
    assert np.abs(new - old).max() <= 4 * ulp(old) and int(new.argmax()) == int(old.argmax())
    seen.clear()
    cache = make_prompt_cache(model)
    one = MLXEngine._logits(eng, [ids[0]], cache)
    assert seen == [(1, 1, 64)]
    monkeypatch.undo()
    want = np.array(model(mx.array([[ids[0]]]), cache=make_prompt_cache(model))[0, -1]
                    .astype(mx.float32))
    assert np.array_equal(one, want)


def test_the_toy_serves_the_last_row_through_its_head(mlx_comp, monkeypatch):
    """The same on the served toy (untied, reference path): one row reaches
    lm_head at prefill and at every decode step, and the row is the full
    forward's."""
    from mlx_lm.models.cache import make_prompt_cache

    ids = mlx_comp.tokenize(messages=MSGS)
    old = np.array(mlx_comp.model(mx.array([ids]), cache=make_prompt_cache(mlx_comp.model))[0, -1]
                   .astype(mx.float32))
    seen = _spy_head(monkeypatch, mlx_comp.model)
    new = mlx_comp.prefill_logits(ids)
    assert np.abs(new - old).max() <= 4 * ulp(old) and int(new.argmax()) == int(old.argmax())
    run(mlx_comp, greedy(max_tokens=3))
    assert seen[0] == (1, 1, 1024) and set(seen) == {(1, 1, 1024)} and len(seen) == 1 + 3



def _full_head_greedy(model, ids, n_new):
    """The bench's greedy loop as it was before the last-row head: mlx-lm's
    Model.__call__, every position headed, the last row's argmax."""
    from mlx_lm.models.cache import make_prompt_cache

    cache, out, step, rows = make_prompt_cache(model), list(ids), ids, []
    for _ in range(n_new):
        row = model(mx.array([step], dtype=mx.int32), cache=cache)[0, -1].astype(mx.float32)
        mx.eval(row)
        rows.append(np.array(row))
        tok = int(mx.argmax(row))
        out.append(tok)
        step = [tok]
    return out, rows


@pytest.mark.parametrize("arm", ["stock", "compressed"])
def test_the_bench_arms_head_the_last_row_and_change_no_token(arm, toy, mlx_comp):
    """arms_mlx.greedy / timed_prefill / timed_ttft run engine_mlx.last_row_logits,
    serve's own path. On the toy, for the stock arm and the compressed arm:
    greedy output equals the old full-head loop's, and the last logits row is
    bitwise the full head's (measured bitwise on MLX CPU; an M = 1 head may
    round differently from row T-1 of an M = T GEMM on Metal, which is the
    bf16 floor test_the_prefill_head_reads_the_last_hidden_row_only allows)."""
    from mlx_lm.models.cache import make_prompt_cache

    from drinkme import arms_mlx
    from drinkme.serving.engine_mlx import last_row_logits

    eng = mlx_comp if arm == "compressed" else load_stock_mlx(toy[0], None)
    ids = eng.tokenize(messages=MSGS)
    want, rows = _full_head_greedy(eng.model, ids, 8)
    assert arms_mlx.greedy(eng, ids, 8) == want
    got = np.array(last_row_logits(eng.model, ids, make_prompt_cache(eng.model)))
    assert got.dtype == np.float32 and np.array_equal(got, rows[0])
    assert np.array_equal(np.array(last_row_logits(eng.model, ids)), rows[0])  # no cache: timed_prefill's call
    assert len(arms_mlx.timed_prefill(eng, ids, reps=2)) == 2
    assert len(arms_mlx.timed_ttft(eng, ids, reps=2)) == 2


@pytest.mark.parametrize("arm", ["stock", "reference", "fused"])
def test_a_decode_step_handed_over_layer_by_layer_is_the_one_graph_step(arm, toy, mlx_comp):
    """last_row_logits at T = 1 evaluates every DECODE_EVAL_EVERY layers
    (mx.async_eval) while it builds the rest; each decode step's row is
    bitwise the row of the backbone built and evaluated as one graph."""
    from mlx_lm.models.cache import make_prompt_cache

    from drinkme.serving import engine_mlx

    if arm == "fused" and not mx.metal.is_available():
        pytest.skip("dispatches the fused Metal kernel; no Metal on this machine")
    eng = {"stock": lambda: load_stock_mlx(toy[0], None), "reference": lambda: mlx_comp,
           "fused": lambda: load_compressed_mlx(toy[0], None, toy[1], path="fused")}[arm]()
    ids = eng.tokenize(messages=MSGS)
    ca, cb = make_prompt_cache(eng.model), make_prompt_cache(eng.model)
    step = ids
    for _ in range(6):
        got = engine_mlx.last_row_logits(eng.model, step, ca)
        h = eng.model.model(mx.array([step], dtype=mx.int32), cache=cb)
        want = engine_mlx._head(eng.model, h[:, -1:, :])[0, -1].astype(mx.float32)
        mx.eval(got, want)
        assert np.array_equal(np.array(got), np.array(want))
        step = [int(mx.argmax(got))]
    assert len(eng.model.model.layers) >= engine_mlx.DECODE_EVAL_EVERY  # the toy has a hand-over


def test_the_bench_arms_reach_the_head_with_one_row(mlx_comp, monkeypatch):
    """The three timers head ONE row, as serve does (no [1, T, vocab] logits)."""
    from drinkme import arms_mlx

    ids = mlx_comp.tokenize(messages=MSGS)
    seen = _spy_head(monkeypatch, mlx_comp.model)
    arms_mlx.greedy(mlx_comp, ids, 2)
    arms_mlx.timed_prefill(mlx_comp, ids, reps=1)
    arms_mlx.timed_ttft(mlx_comp, ids, reps=1)
    assert seen and set(seen) == {(1, 1, 1024)}

@pytest.fixture
def roomy_cache():
    """mlx's cache limit raised for the test (the CPU build's default is 32
    MiB; Metal's is ~1.5x the working set, which is what kept a 12 GB
    footprint on an M4 after an 8k request), so a request's freed buffers
    stay cached the way they do on a Mac; restored after."""
    old = mx.set_cache_limit(1 << 34)
    yield
    mx.clear_cache()
    mx.set_cache_limit(old)


def _cache_returned() -> bool:
    """mlx's cache is empty. On Metal a command buffer's buffers are freed
    by its completion handler, which can run after mx.synchronize() has
    returned, so the last buffer's few outputs may land in the cache after
    the hand-back: up to 16 MiB is allowed there, none on the CPU build."""
    return mx.get_cache_memory() <= (16 << 20 if mx.metal.is_available() else 0)


def _engine_like(eng):
    from drinkme.serving.engine_mlx import MLXEngine as E

    return E(eng.model, eng.tok, model_id=eng.model_id, arm=eng.arm, meta=eng.meta,
             ctx=eng.ctx, path=eng.path)


def test_a_request_hands_its_buffers_back(mlx_comp, roomy_cache, capsys):
    """mlx caches every buffer a request frees. The prefill's go back
    before the first decode step (the fused path's decode never reuses them), the rest at
    the request's end — on the finished path and on a stream its consumer
    closes mid-decode; the line is mlx's own account of both and of the
    request's peak."""
    eng = _engine_like(mlx_comp)
    capsys.readouterr()
    res, _ = run(eng, greedy(max_tokens=4))
    assert _cache_returned()
    line = [ln for ln in capsys.readouterr().out.splitlines() if "mlx memory" in ln]
    assert len(line) == 1 and f"{res.prompt_tokens}-token prompt" in line[0]
    after_prefill = float(line[0].split("cache returned: ")[1].split()[0])
    peak = float(line[0].split("peak ")[1].split()[0])
    assert peak > 0 and after_prefill > 0
    # closed by its consumer mid-decode: the finally still hands back
    stream = eng.generate(GenerationRequest(MSGS, greedy()))
    assert isinstance(next(stream), StreamStart)
    assert isinstance(stream.send(None), Delta)  # the prefill's first token
    prefill_cached = mx.get_cache_memory()
    assert prefill_cached > 0  # what the first decode step hands back
    assert isinstance(stream.send(True), Delta)  # one decode step later
    stream.close()
    assert _cache_returned()
    line = [ln for ln in capsys.readouterr().out.splitlines() if "mlx memory" in ln][-1]
    assert "GiB after prefill" in line
    reported = float(line.split("cache returned: ")[1].split()[0])
    # the engine reads the cache after mx.synchronize(), at the next send; this test read it right
    # after the Delta. On Metal a completion handler can add up to 16 MiB in between
    # (_cache_returned), and the line rounds to 0.01 GiB: an exact match flaked 3 in ~44 runs on
    # the M4. None of it is arithmetic.
    slack = ((16 << 20) if mx.metal.is_available() else 0) / 1024**3 + 0.005
    assert abs(reported - prefill_cached / 1024**3) <= slack


def test_keep_cache_leaves_mlxs_cache_alone(mlx_comp, roomy_cache, monkeypatch, capsys):
    """DRINKME_MLX_KEEP_CACHE=1: mlx's own behaviour, the cache kept for
    the next request — the A/B switch for what the hand-back costs."""
    from drinkme.serving.engine_mlx import KEEP_CACHE_ENV

    monkeypatch.setenv(KEEP_CACHE_ENV, "1")
    eng = _engine_like(mlx_comp)
    capsys.readouterr()
    run(eng, greedy(max_tokens=4))
    assert mx.get_cache_memory() > 0
    assert "cache kept: " in capsys.readouterr().out


# --------------------------------------- the fused path, Mac only --


@needs_metal
def test_fused_path_matches_the_reference_path_on_the_toy(toy, mlx_comp):
    """The reference-vs-fused pin, on the toy (docs/metal.md, M4 day): the
    same pack loaded on the fused path must give logits within the bf16
    floor of the reference path's (the GEMV accumulates in f32 in its own
    order; the reference rounds every Linear through a bf16 matmul) with
    the same argmax, and the same 24-token greedy transcript — or the
    prefix up to a fork, reported."""
    fused = load_compressed_mlx(toy[0], None, toy[1], path="fused")
    assert fused.path == "fused" and fused.model_meta()["computePath"] == "fused"
    assert not fused.device.startswith("cpu")
    ids = fused.tokenize(messages=MSGS)
    a, b = fused.prefill_logits(ids), mlx_comp.prefill_logits(ids)
    ulp = 2.0 ** (np.floor(np.log2(max(np.abs(b).max(), 1.0))) - 7)
    assert np.abs(a - b).max() <= 4 * ulp
    assert int(a.argmax()) == int(b.argmax())
    rf, _ = run(fused, greedy())
    rr, _ = run(mlx_comp, greedy())
    if rf.text != rr.text:
        f, r = rf.text.split(), rr.text.split()
        k = next(i for i, (x, y) in enumerate(zip(f, r)) if x != y)
        assert k >= 2, f"forked at token {k}: fused {f[:k + 1]} vs reference {r[:k + 1]}"
        pytest.xfail(f"greedy forked at token {k} (near-tie); prefix of {k} tokens agrees")
    assert rf.text == rr.text
    # M>1 through the fused path is the dense arm: dense_radix's transient
    # through the same matmul the reference runs over the oracle's bytes ->
    # bitwise the reference module's forward (never claimed against the M=1
    # GEMV, whose accumulation order differs)
    lin = _module_at(fused.model, "model.layers.0.mlp.down_proj")
    ref = _module_at(mlx_comp.model, "model.layers.0.mlp.down_proj")
    assert isinstance(lin, RadixLinear) and lin.path == "fused" and ref.path == "reference"
    x = mx.random.normal((3, lin.C)).astype(mx.bfloat16)
    y_f, y_r = lin(x), ref(x)
    mx.eval(y_f, y_r)
    assert np.array_equal(np.array(y_f.view(mx.uint16)), np.array(y_r.view(mx.uint16)))


def test_the_layerwise_decode_is_only_taken_for_the_family_it_restates(monkeypatch):
    """last_row_logits' layer-by-layer decode (DECODE_EVAL_EVERY) restates mlx-lm's
    Qwen3Model.__call__; an inner model from any other module must take its own __call__."""
    import mlx.core as mx

    from drinkme.serving import engine_mlx

    calls = []

    class Inner:
        def __call__(self, x, cache=None):
            calls.append("own __call__")
            return mx.zeros((1, x.shape[1], 4))

    Inner.__module__ = "mlx_lm.models.gemma3"

    class Model:
        pass

    m = Model()
    m.model = Inner()
    monkeypatch.setattr(engine_mlx, "_head", lambda model, h: h)
    engine_mlx.last_row_logits(m, [1], cache=[object()])
    assert calls == ["own __call__"]
    assert "mlx_lm.models.qwen3" in engine_mlx._LAYERWISE_MODULES
