"""MTP speculative decoding on a tiny synthetic qwen3_5 — CPU only, no
downloads, no GPU.

The toy is a genuine hybrid `Qwen3_5ForCausalLM`: 4 layers, alternating
linear_attention (gated DeltaNet) and full_attention, plus a genuine MTP head
built from the real `Qwen3_5DecoderLayer` — the same classes the 27B serves
through, at 1/100000th the size. Everything runs in float32: a bf16 toy's
logits sit inside kernel-shape noise of each other and greedy near-ties flip
for reasons that have nothing to do with this feature (the prefix-cache tests
next door say the same thing about warm-vs-cold).

THE bar is test_mtp_matches_serial_greedy_over_varied_prompts: MTP on agrees
with MTP off, argmax for argmax, except at a genuine near-tie — token-exact
between speculation on and off is not a property this repo asserts
(tests/spec_agree.py is the shared check every MTP/n-gram test
uses for it). With random weights the head is a terrible drafter
(acceptance ~0), which is exactly the adversarial case wanted —
every cycle rejects at position 0 and every rejection path runs. The
opposite extreme (a draft that is always right, so cycles accept all k and
emit the bonus token) can't be produced by choosing a prompt on random
weights, so it is forced with an oracle draft instead: same machinery, both
extremes, deterministically.

CPU CAVEAT, load-bearing: gated-DeltaNet PREFILL goes through
`torch_chunk_gated_delta_rule`, which resolves to fla's triton kernel and dies
on a CPU tensor. The pure-torch reference is still there as `__wrapped__`, and
these tests swap it in. DECODE does not need the swap — transformers looks up
`fla.ops.gated_delta_rule.recurrent_gated_delta_rule`, fla only exports
`fused_recurrent_gated_delta_rule`, so the name as transformers ships it is
the torch loop; the engine loaders rebind it to fla's fused kernel on an
accelerator (serving/deltanet.py), and these tests
build their engines directly, so here it stays torch. That is why the
per-position replay in serving/mtp.py is testable here at all.
"""

import json
import os

import pytest
import torch

from drinkme.serving import metrics, mtp
from drinkme.serving.engine import GenerationRequest, SampleParams, complete
from drinkme.serving.engines import HFEngine

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")

WORDS = ["hello", "world", "the", "quick", "brown", "fox", "jumps", "over",
         "lazy", "dog", "alpha", "beta", "gamma", "delta", "epsilon", "zeta",
         "user", "assistant", "system", ":"]


def _cfg():
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

    # head_dim 32 x partial_rotary 0.25 -> 8 rotary dims -> 4 inv_freq entries,
    # which is what mrope_section must sum to (the 27B: 256 * 0.25 / 2 = 32 =
    # 11 + 11 + 10). Keep the toy's rope shape honest.
    return Qwen3_5TextConfig(
        vocab_size=96, hidden_size=64, intermediate_size=128,
        num_hidden_layers=4, num_attention_heads=2, num_key_value_heads=1,
        head_dim=32, linear_key_head_dim=16, linear_value_head_dim=16,
        linear_num_key_heads=2, linear_num_value_heads=4,
        linear_conv_kernel_dim=4,
        layer_types=["linear_attention", "full_attention",
                     "linear_attention", "full_attention"],
        max_position_embeddings=512,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0,
                         "mrope_section": [2, 1, 1], "mrope_interleaved": True,
                         "partial_rotary_factor": 0.25},
        # no EOS: the toy's random argmax would hit one every ~96 tokens and
        # truncate the 64-token comparison runs. The EOS paths get their own
        # test, which installs one deliberately.
        tie_word_embeddings=False, eos_token_id=None, pad_token_id=None)


@pytest.fixture(scope="session", autouse=True)
def _torch_chunk_on_cpu():
    """Module docstring: fla's chunk kernel is triton-only. Assert we really
    got the pure-torch reference — if a transformers bump changes the
    decorator shape, this fails here instead of somewhere baffling."""
    from transformers.models.qwen3_5 import modeling_qwen3_5 as mod

    # Two venv-dependent shapes exist for the same substance: the kernels-hub
    # decorator wraps the torch reference (worktree venv, __wrapped__ present)
    # or the name IS already the bare reference (main venv, no wrapper).
    # Assert the INTENT — we run the torch reference — not the wrapping.
    fn = mod.torch_chunk_gated_delta_rule
    ref = getattr(fn, "__wrapped__", fn)
    assert ref.__name__ == "torch_chunk_gated_delta_rule"
    assert ref.__module__.endswith("modeling_qwen3_5")
    kernelized = mod.torch_chunk_gated_delta_rule
    mod.torch_chunk_gated_delta_rule = ref
    yield
    mod.torch_chunk_gated_delta_rule = kernelized


def _fill_random(module, seed: int, scale: float = 0.08):
    """Materialize a meta-device module with random weights (the head is built
    on meta by design — it normally streams from the checkpoint)."""
    torch.manual_seed(seed)
    for name, p in list(module.named_parameters()):
        path, _, attr = name.rpartition(".")
        owner = module.get_submodule(path) if path else module
        owner._parameters[attr] = torch.nn.Parameter(
            torch.randn(p.shape, dtype=torch.float32) * scale, requires_grad=False)
    return module


@pytest.fixture(scope="session")
def toy():
    """(model, tokenizer, head, cfg) — one build, shared by everything."""
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

    cfg = _cfg()
    torch.manual_seed(0)
    model = Qwen3_5ForCausalLM(cfg).eval().to(torch.float32)
    for p in model.parameters():
        p.requires_grad_(False)

    vocab = {"<unk>": 0, "<pad>": 1, "</s>": 2}
    for w in WORDS:
        vocab[w] = len(vocab)
    while len(vocab) < cfg.vocab_size:
        vocab[f"tok{len(vocab)}"] = len(vocab)
    backend = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
    backend.pre_tokenizer = Whitespace()
    tok = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="<unk>",
                                  pad_token="<pad>")
    tok.chat_template = ("{% for m in messages %}{{ m['role'] }} : {{ m['content'] }} "
                         "{% endfor %}{% if add_generation_prompt %}assistant :{% endif %}")

    head = _fill_random(mtp.MTPHead(cfg, model), seed=1)
    head.eval()
    mtp.install_deltanet_capture(model)
    return model, tok, head, cfg


def _engine(toy, with_mtp: bool, ctx: int = 512):
    model, tok, head, _ = toy
    os.environ["DRINKME_PREFIX_SLOTS"] = "0"
    try:
        return HFEngine(model, tok, model_id="toy", arm="test", meta={}, ctx=ctx,
                        mtp_head=head if with_mtp else None)
    finally:
        del os.environ["DRINKME_PREFIX_SLOTS"]


def _greedy(max_tokens=64, **kw):
    return SampleParams(temperature=0.0, max_tokens=max_tokens, **kw)


def _run(eng, msgs, params=None, until=None):
    deltas = []

    def cb(d):
        deltas.append(d)
        return True if until is None else until(deltas)

    return complete(eng, GenerationRequest(msgs, params or _greedy()), cb), deltas


def _msgs(text):
    return [{"role": "user", "content": text}]


# ------------------------------------------------ the transformers seams --


def test_static_layer_writes_at_its_own_counter_not_cache_position(toy):
    """THE footgun, pinned behaviourally rather than by reading one version's
    source: hand StaticLayer a cache_position that lies and watch where the
    rows actually land. If a transformers bump ever makes StaticCache honour
    cache_position, this fails — and serving/mtp.py's rewind (which subtracts
    from the counter) has to be re-derived before MTP can run again."""
    from transformers import StaticCache

    _, _, _, cfg = toy
    cache = StaticCache(config=cfg, max_cache_len=32)
    layer = next(l for l in cache.layers if hasattr(l, "cumulative_length"))
    k = torch.full((1, cfg.num_key_value_heads, 3, cfg.head_dim), 7.0)
    with torch.inference_mode():
        keys, _ = layer.update(k, k, cache_position=torch.tensor([10, 11, 12]))
    assert torch.equal(keys[0, 0, 0], k[0, 0, 0])           # landed at row 0...
    assert torch.all(keys[0, 0, 10] == 0)                    # ...not at row 10
    assert isinstance(layer.cumulative_length, torch.Tensor)  # rewind needs a tensor
    assert int(layer.cumulative_length) == 3


def test_rewind_puts_the_next_write_back_on_the_accepted_row(toy):
    """The rewind itself: drop 2 of 3 written rows and the next write must
    land on row 1, not row 3. This is what makes a rejected draft's KV
    unreachable instead of merely stale."""
    from transformers import StaticCache

    _, _, _, cfg = toy
    cache = StaticCache(config=cfg, max_cache_len=32)
    shape = (1, cfg.num_key_value_heads, 3, cfg.head_dim)
    cap = mtp._Capture()
    cap.rows = 3
    with torch.inference_mode():
        cache.layers[1].update(torch.full(shape, 7.0), torch.full(shape, 7.0))
        mtp._restore_rows(cache, cap, keep=1)
        assert int(cache.layers[1].cumulative_length) == 1
        one = (1, cfg.num_key_value_heads, 1, cfg.head_dim)
        keys, _ = cache.layers[1].update(torch.full(one, 5.0), torch.full(one, 5.0))
    assert torch.all(keys[0, 0, 1] == 5.0)   # overwrote the rejected row
    assert torch.all(keys[0, 0, 0] == 7.0)   # kept the accepted one


def test_rewind_refuses_a_cache_layer_it_does_not_understand():
    """A silent no-op here is a corrupted transcript. Anything without a
    cumulative_length tensor and without crop() must raise."""

    class Alien:
        pass

    class Cache:
        layers = [Alien()]

    cap = mtp._Capture()
    cap.rows = 3
    with pytest.raises(RuntimeError, match="cannot rewind cache layer"):
        mtp._restore_rows(Cache(), cap, keep=1)


def test_default_depth_keeps_the_verify_batch_on_the_bit_pinned_arm(monkeypatch):
    """Why DEFAULT_DEPTH is 4 and the warning fires past FUSED_M_MAX: on the
    compressed arm codec/swap.py routes M in 2..MC_MAX through the mc kernel
    (radix_ops.gemv_mc), whose per-column output is within the float64
    oracle's bound of the M=1 decode step's (bench/radix_mc_bitpin.py:
    bitwise is recorded, not gated), and only switches to decode-once +
    native GEMM at GEMM_MIN_ROWS, whose
    reduction order is a different one. If either constant moves, the depth
    ceiling has to move with it or MTP starts forking greedy near-ties off the
    non-MTP transcript — so this asserts the derivation, not the number.

    A python ROW LOOP over the M=1 kernel would have the same numerics at
    M reads of the weights; mc keeps the numerics and drops the reads, and
    the window is MC_MAX with it. Keep the verify batch inside the pinned
    window."""
    import drinkme.codec.swap as swap
    from drinkme.codec.ops import MC_MAX
    from drinkme.codec.swap import CompressedLinear

    monkeypatch.delenv("DRINKME_PREFILL_DENSE_MIN", raising=False)
    monkeypatch.setattr(swap, "_DENSE_MIN", None)
    route = CompressedLinear._route_rows  # reads no instance state
    assert mtp.FUSED_M_MAX == min(MC_MAX, CompressedLinear.GEMM_MIN_ROWS - 1)
    assert mtp.DEFAULT_DEPTH + 1 <= mtp.FUSED_M_MAX  # the default batch fits
    assert route(None, 1) == "gemv"
    for m in range(2, mtp.FUSED_M_MAX + 1):  # every verify batch inside it
        assert route(None, m) == "mc", m
    assert route(None, mtp.FUSED_M_MAX + 1) != "mc"


# ------------------------------------------------------ the DeltaNet wrap --


def _prefilled(toy, n=8):
    from transformers import StaticCache

    model, _, _, cfg = toy
    cache = StaticCache(config=cfg, max_cache_len=64)
    ids = torch.tensor([[5, 7, 11, 13, 17, 19, 23, 29][:n]])
    with torch.inference_mode():
        model(ids, past_key_values=cache, use_cache=True,
              cache_position=torch.arange(ids.shape[1]))
    return cache


def _snapshot(cache):
    return {i: (l.recurrent_states[0].clone(), l.conv_states[0].clone())
            for i, l in enumerate(cache.layers)
            if getattr(l, "recurrent_states", {}).get(0) is not None}


def _restore_snapshot(cache, snap):
    with torch.inference_mode():
        for i, (rec, conv) in snap.items():
            cache.layers[i].recurrent_states[0].copy_(rec)
            cache.layers[i].conv_states[0].copy_(conv)


def test_deltanet_wrap_is_inert_without_an_active_capture(toy):
    """MTP off (or a 1-row step) must be the upstream forward, bit for bit —
    "inactive means byte-identical to the non-MTP path" is the whole gating promise."""
    model, _, _, _ = toy
    dn = model.model.layers[0].linear_attn
    assert dn._drinkme_mtp_wrapped
    cache = _prefilled(toy)
    snap = _snapshot(cache)
    x = torch.randn(1, 3, 64, generator=torch.Generator().manual_seed(4))
    assert mtp._ACTIVE is None
    with torch.inference_mode():
        wrapped = dn(hidden_states=x, cache_params=cache, attention_mask=None)
    after_wrapped = _snapshot(cache)
    _restore_snapshot(cache, snap)
    with torch.inference_mode():
        original = type(dn).forward(dn, hidden_states=x, cache_params=cache,
                                    attention_mask=None)
    assert torch.equal(wrapped, original)
    for i, (rec, conv) in _snapshot(cache).items():
        assert torch.equal(after_wrapped[i][0], rec)
        assert torch.equal(after_wrapped[i][1], conv)


def test_deltanet_capture_matches_a_position_by_position_reference(toy):
    """The state-restore bar: the state captured after row j of an
    M-row verify batch must be the state an M=1 step-by-step run would have
    had after its j-th step — that is what a rejection restores to.

    Conv states are BITWISE equal (they are raw input columns; the rewind is a
    slice of the window). Recurrent states agree to ~5e-10 in float32, not
    bitwise: the residual enters through `in_proj_b` / `in_proj_a` (64->4 on
    the toy), whose batched and single-row F.linear pick different kernels and
    differ by one float32 ulp — measured here, not assumed. That is the same
    "batched vs stepped" class the engine already documents for warm-vs-cold
    prefill; an actual indexing bug in the capture moves these numbers by
    orders of magnitude, not ulps.
    """
    model, _, _, _ = toy
    dn = model.model.layers[2].linear_attn
    cache = _prefilled(toy)
    base = _snapshot(cache)
    rows = 5
    x = torch.randn(1, rows, 64, generator=torch.Generator().manual_seed(7))

    cap = mtp._Capture()
    mtp._ACTIVE = cap
    try:
        with torch.inference_mode():
            batched = dn(hidden_states=x, cache_params=cache, attention_mask=None)
    finally:
        mtp._ACTIVE = None
    kernel = cache.layers[2].conv_kernel_size[0]
    got_rec = [s.clone() for s in cap.rec[2]]
    got_conv = [cap.win[2][..., j + 1:j + 1 + kernel].clone() for j in range(rows)]

    _restore_snapshot(cache, base)
    ref_rec, ref_conv, outs = [], [], []
    with torch.inference_mode():
        for j in range(rows):
            outs.append(dn(hidden_states=x[:, j:j + 1], cache_params=cache,
                           attention_mask=None))
            ref_rec.append(cache.layers[2].recurrent_states[0].clone())
            ref_conv.append(cache.layers[2].conv_states[0].clone())

    for j in range(rows):
        assert torch.equal(got_conv[j], ref_conv[j]), f"conv state row {j}"
        delta = (got_rec[j] - ref_rec[j]).abs().max().item()
        assert delta < 1e-7, f"recurrent state row {j} drifted {delta:.3e}"
    assert torch.allclose(batched, torch.cat(outs, dim=1), rtol=1e-5, atol=1e-7)


def test_verify_batch_rewind_matches_a_stepwise_run(toy):
    """The whole rejection story on the real model, end to end: write a 5-row
    verify batch through every layer, reject 3 of them, and the cache — KV
    rows, attention counter, DeltaNet recurrent state, conv state, all 4
    layers — must match a cache that only ever saw the 2 accepted tokens.

    This is the test that would catch a wrong index in the capture, a rewind
    that forgets a surface, or a StaticCache that started honouring
    cache_position."""
    model, _, _, cfg = toy
    prompt = torch.tensor([[5, 7, 11, 13, 17, 19, 23, 29]])
    rows = torch.tensor([[3, 41, 8, 62, 15]])

    def prefill(cache):
        with torch.inference_mode():
            model(prompt, past_key_values=cache, use_cache=True,
                  cache_position=torch.arange(8))

    from transformers import StaticCache

    got = StaticCache(config=cfg, max_cache_len=64)
    prefill(got)
    cap = mtp._Capture()
    mtp._ACTIVE = cap
    try:
        with torch.inference_mode():
            model(rows, past_key_values=got, use_cache=True,
                  cache_position=torch.arange(8, 13))
    finally:
        mtp._ACTIVE = None
    cap.rows = 5
    with torch.inference_mode():
        mtp._restore_rows(got, cap, keep=2)

    ref = StaticCache(config=cfg, max_cache_len=64)
    prefill(ref)
    with torch.inference_mode():
        for j in range(2):
            model(rows[:, j:j + 1], past_key_values=ref, use_cache=True,
                  cache_position=torch.tensor([8 + j]))

    for i, (a, b) in enumerate(zip(got.layers, ref.layers)):
        if hasattr(a, "cumulative_length"):
            assert int(a.cumulative_length) == int(b.cumulative_length) == 10
            # only the rows the cache CLAIMS need to agree; anything past the
            # counter is unreachable (the causal mask never offers it)
            assert torch.allclose(a.keys[:, :, :10], b.keys[:, :, :10],
                                  rtol=1e-5, atol=1e-7), f"keys layer {i}"
            assert torch.allclose(a.values[:, :, :10], b.values[:, :, :10],
                                  rtol=1e-5, atol=1e-7), f"values layer {i}"
        if getattr(a, "recurrent_states", {}).get(0) is not None:
            assert torch.equal(a.conv_states[0], b.conv_states[0]), f"conv layer {i}"
            drift = (a.recurrent_states[0] - b.recurrent_states[0]).abs().max().item()
            assert drift < 1e-7, f"recurrent layer {i} drifted {drift:.3e}"


# --------------------------------------------------------- THE greedy bar --


PROMPTS = [" ".join(WORDS[i % len(WORDS):i % len(WORDS) + 1 + (i % 5)])
           for i in range(20)]


@pytest.fixture(scope="session")
def serial_runs(toy):
    """One reference greedy run per prompt, MTP off: {prompt: {"res", "ids",
    "rows"}}. finish must be 'length': a short run would silently weaken the
    comparison. `ids`/`rows` (spec_agree.capture) are what an MTP-on
    comparison checks a fork's margin against — token-exact between
    speculation on and off is not a property this repo asserts;
    argmax agreement except at a genuine near-tie is."""
    from drinkme.serving import engines
    from spec_agree import capture, ordered_ids

    eng = _engine(toy, with_mtp=False)
    out = {}
    for p in PROMPTS:
        picks, rows, restore = capture(engines)
        try:
            res, _ = _run(eng, _msgs(p))
        finally:
            restore()
        assert res.finish_reason == "length" and res.completion_tokens == 64
        out[p] = {"res": res, "ids": ordered_ids(picks), "rows": rows}
    return out


def test_mtp_matches_serial_greedy_over_varied_prompts(toy, serial_runs):
    """THE correctness property: MTP-on agrees with MTP-off, argmax for
    argmax, over 20 varied prompts x 64 tokens — except at a genuine
    near-tie, where the verify's batched k+1-position forward and the serial
    loop's single-token one are free to round differently in accumulation
    order (spec_agree; token-exact between speculation on and off is not a
    property this repo asserts). On random weights the head
    drafts badly, so nearly every cycle rejects at position 0 — the
    adversarial case runs on every prompt here, thousands of times."""
    from drinkme.serving import engines
    from spec_agree import assert_agrees_or_forks_at_a_near_tie, capture, ordered_ids

    eng = _engine(toy, with_mtp=True)
    with mtp_depth(4):
        for p in PROMPTS:
            picks, rows, restore = capture(engines)
            try:
                res, _ = _run(eng, _msgs(p))
            finally:
                restore()
            ref = serial_runs[p]
            assert_agrees_or_forks_at_a_near_tie(ordered_ids(picks), ref["ids"],
                                                 rows, ref["rows"])
            assert res.completion_tokens == ref["res"].completion_tokens
            assert res.finish_reason == ref["res"].finish_reason


@pytest.mark.parametrize("depth", [1, 2, 3, 5])
def test_mtp_matches_at_every_depth(toy, serial_runs, depth):
    """Depth is a speed knob, never a correctness one — including k=5, whose
    M=6 verify batch leaves the fused window (the warning path)."""
    from drinkme.serving import engines
    from spec_agree import assert_agrees_or_forks_at_a_near_tie, capture, ordered_ids

    eng = _engine(toy, with_mtp=True)
    p = PROMPTS[3]
    picks, rows, restore = capture(engines)
    try:
        with mtp_depth(depth):
            res, _ = _run(eng, _msgs(p))
    finally:
        restore()
    ref = serial_runs[p]
    assert_agrees_or_forks_at_a_near_tie(ordered_ids(picks), ref["ids"], rows, ref["rows"])


class mtp_depth:
    """DRINKME_MTP_DEPTH=k for a block, restored after (and the once-per-process
    warning latch cleared, so warning tests are order-independent). k == 0
    is speculation off, which is DRINKME_SPEC=off: the depth variable is not
    an off switch."""

    _NAMES = ("DRINKME_MTP_DEPTH", "DRINKME_SPEC")

    def __init__(self, k):
        self.k = k

    def __enter__(self):
        self.old = {n: os.environ.get(n) for n in self._NAMES}
        if str(self.k) == "0":
            os.environ["DRINKME_SPEC"] = "off"
        else:
            os.environ["DRINKME_MTP_DEPTH"] = str(self.k)
        mtp._warned.clear()
        return self

    def __exit__(self, *a):
        for n, v in self.old.items():
            if v is None:
                os.environ.pop(n, None)
            else:
                os.environ[n] = v
        mtp._warned.clear()


def test_mtp_metrics_record_proposed_and_accepted(toy):
    """Speculator.cycle's one-line call into serving/metrics.py.
    `proposed`/`accepted` are server-lifetime totals, deliberately a
    different accounting from this request's own s['drafted']/s['accepted']
    (the per-request audit line) — one short generation is enough to prove
    the wiring moves them, correctness of the numbers is
    test_mtp_matches_serial_greedy_over_varied_prompts's job."""
    metrics.reset()
    eng = _engine(toy, with_mtp=True)
    with mtp_depth(4):
        res, _ = _run(eng, _msgs(PROMPTS[0]), _greedy(max_tokens=16))
    assert res.completion_tokens > 0
    assert metrics.SPEC_PROPOSED.snapshot()[()] > 0  # at least one cycle drafted


def _force_draft(head, fn):
    """Swap the head's draft policy while keeping its KV bookkeeping real: the
    genuine draft still runs (so the head's cache holds genuine entries), only
    the proposed tokens are replaced."""
    real = head.draft

    def forced(h, token, first_entry, k):
        real(h, token, first_entry, k)
        return fn(token, k)

    head.draft = forced
    return real


def test_forced_full_acceptance_still_matches(toy, serial_runs):
    """The other extreme, which random weights cannot produce: a draft that is
    always right. Cycles accept all k and emit the bonus token, so the
    accept-everything path and the head's re-key-on-true-hiddens commit both
    run — and the output must still agree with the serial one, modulo a
    genuine near-tie (spec_agree)."""
    from drinkme.serving import engines
    from spec_agree import assert_agrees_or_forks_at_a_near_tie, capture, ordered_ids

    model, tok, head, _ = toy
    p = PROMPTS[5]
    ref = serial_runs[p]
    oracle = tok(ref["res"].text)["input_ids"]
    eng = _engine(toy, with_mtp=True)

    def truth(token, k):
        try:
            i = oracle.index(token)
        except ValueError:
            return [oracle[0]] * k
        return (oracle[i + 1:i + 1 + k] + [0] * k)[:k]

    real = _force_draft(head, truth)
    picks, rows, restore = capture(engines)
    try:
        with mtp_depth(4):
            res, _ = _run(eng, _msgs(p))
    finally:
        head.draft = real
        restore()
    assert_agrees_or_forks_at_a_near_tie(ordered_ids(picks), ref["ids"], rows, ref["rows"])
    assert res.completion_tokens == 64


def test_forced_rejection_at_position_zero_still_matches(toy, serial_runs):
    """Every cycle wrong on the first draft: m == 0, k rows discarded per
    cycle, the deepest rewind the scheme can ask for."""
    from drinkme.serving import engines
    from spec_agree import assert_agrees_or_forks_at_a_near_tie, capture, ordered_ids

    model, tok, head, _ = toy
    p = PROMPTS[9]
    eng = _engine(toy, with_mtp=True)
    real = _force_draft(head, lambda token, k: [(token + 1) % 96] * k)
    picks, rows, restore = capture(engines)
    try:
        with mtp_depth(4):
            res, _ = _run(eng, _msgs(p))
    finally:
        head.draft = real
        restore()
    ref = serial_runs[p]
    assert_agrees_or_forks_at_a_near_tie(ordered_ids(picks), ref["ids"], rows, ref["rows"])


# ------------------------------------------- stopping in the middle of a cycle --


def test_stop_string_mid_cycle_leaves_a_truthful_cache(toy, serial_runs):
    """A stop string can land on the FIRST of five tokens a cycle already
    decided. The other four have KV in the cache that nothing will emit — and
    the slot's ids must describe the cache after the rewind, not before, or
    the next turn's extends-only reuse writes its suffix on top of them."""
    from drinkme.serving import engines
    from spec_agree import assert_agrees_or_forks_at_a_near_tie, capture, ordered_ids

    model, tok, head, _ = toy
    p = PROMPTS[2]
    ref = serial_runs[p]
    stop = ref["res"].text.split()[3]
    eng = HFEngine(model, tok, model_id="toy", arm="test", meta={}, ctx=512,
                   mtp_head=head)  # prefix cache ON: the slot's ids are the point
    picks, rows, restore = capture(engines)
    try:
        with mtp_depth(4):
            res, _ = _run(eng, _msgs(p), _greedy(stop=[stop]))
    finally:
        restore()
    assert res.finish_reason == "stop"
    # agrees with the serial reference up to the stop, modulo a near-tie —
    # not `text[:text.find(stop)]` bitwise, which is the removed transcript-
    # identity claim
    assert_agrees_or_forks_at_a_near_tie(ordered_ids(picks), ref["ids"], rows, ref["rows"])
    # the cache claims exactly what it holds: replaying the claimed ids as a
    # prompt must reproduce the same continuation the serial engine gives
    kv_ids = eng._slots[0].ids
    assert len(kv_ids) == len(eng.tok(
        tok.decode(kv_ids, skip_special_tokens=False))["input_ids"])
    fresh_off = _engine(toy, with_mtp=False)
    cont = [{"role": "user", "content": p},
            {"role": "assistant", "content": res.text},
            {"role": "user", "content": "over the lazy dog"}]
    picks_w, rows_w, restore = capture(engines)
    try:
        warm, _ = _run(eng, cont, _greedy(max_tokens=16))
    finally:
        restore()
    picks_c, rows_c, restore = capture(engines)
    try:
        cold, _ = _run(fresh_off, cont, _greedy(max_tokens=16))
    finally:
        restore()
    # the cache-truthfulness claim, not a spec-on-vs-off one — but `eng`
    # still carries its head, so the warm continuation may itself speculate
    # (AUTO) while `fresh_off` cannot; agreement modulo a near-tie is what a
    # correct rewind actually buys.
    assert_agrees_or_forks_at_a_near_tie(ordered_ids(picks_w), ordered_ids(picks_c),
                                         rows_w, rows_c)
    assert warm.cached_tokens > 0  # the rewound cache was reused, not reset


def test_max_tokens_mid_cycle_is_exact(toy, serial_runs):
    """max_tokens 6 with depth 4: a cycle can decide 5 tokens when 2 are
    allowed. The caller must get exactly 6, and they must be the first 6 —
    modulo a genuine near-tie against the serial reference (spec_agree)."""
    from drinkme.serving import engines
    from spec_agree import assert_agrees_or_forks_at_a_near_tie, capture, ordered_ids

    eng = _engine(toy, with_mtp=True)
    p = PROMPTS[7]
    picks, rows, restore = capture(engines)
    try:
        with mtp_depth(4):
            res, _ = _run(eng, _msgs(p), _greedy(max_tokens=6))
    finally:
        restore()
    assert res.completion_tokens == 6 and res.finish_reason == "length"
    ref = serial_runs[p]
    assert_agrees_or_forks_at_a_near_tie(ordered_ids(picks), ref["ids"], rows, ref["rows"])


def test_abort_mid_cycle_halts_and_matches(toy, serial_runs):
    from drinkme.serving import engines
    from spec_agree import assert_agrees_or_forks_at_a_near_tie, capture, ordered_ids

    eng = _engine(toy, with_mtp=True)
    p = PROMPTS[11]
    picks, rows, restore = capture(engines)
    try:
        with mtp_depth(4):
            res, deltas = _run(eng, _msgs(p), _greedy(), until=lambda ds: len(ds) < 2)
    finally:
        restore()
    assert res.finish_reason == "abort"
    assert res.text == deltas[0]
    ref = serial_runs[p]
    assert_agrees_or_forks_at_a_near_tie(ordered_ids(picks), ref["ids"], rows, ref["rows"])


def test_eos_inside_a_cycle_stops_cleanly(toy):
    """EOS can arrive as an accepted draft in the middle of a decided batch:
    the generation must end there, agreeing with what the serial loop gives
    modulo a genuine near-tie (spec_agree)."""
    from drinkme.serving import engines
    from spec_agree import assert_agrees_or_forks_at_a_near_tie, capture, ordered_ids

    model, tok, head, _ = toy
    p = PROMPTS[4]
    ref_eng = _engine(toy, with_mtp=False)
    ref, _ = _run(ref_eng, _msgs(p), _greedy(max_tokens=32))
    eos_id = tok(ref.text)["input_ids"][12]
    old = model.generation_config.eos_token_id
    try:
        model.generation_config.eos_token_id = eos_id
        a = _engine(toy, with_mtp=False)
        b = _engine(toy, with_mtp=True)
        picks_off, rows_off, restore = capture(engines)
        try:
            res_off, _ = _run(a, _msgs(p), _greedy(max_tokens=32))
        finally:
            restore()
        picks_on, rows_on, restore = capture(engines)
        try:
            with mtp_depth(4):
                res_on, _ = _run(b, _msgs(p), _greedy(max_tokens=32))
        finally:
            restore()
    finally:
        model.generation_config.eos_token_id = old
    assert res_off.finish_reason == "stop" and res_off.completion_tokens < 32
    assert_agrees_or_forks_at_a_near_tie(ordered_ids(picks_on), ordered_ids(picks_off),
                                         rows_on, rows_off)
    assert res_on.finish_reason == res_off.finish_reason


# ------------------------------------------------------------- the gating --


def test_sampled_requests_speculate_and_constrained_ones_fall_back(toy):
    """A sampled request gets a SAMPLED speculator (rejection sampling), not
    the serial loop; a constrained one falls back to serial."""
    with mtp_depth(4):
        assert mtp.maybe_speculate(toy[2], toy[0],
                                   SampleParams(temperature=0.7)).sampled
        assert mtp.maybe_speculate(toy[2], toy[0], SampleParams(
            temperature=0.0, output_schema={"type": "json_object"})) is None
        assert mtp.maybe_speculate(toy[2], toy[0],
                                   SampleParams(temperature=0.0)) is not None
    # speculation off stays off (DRINKME_SPEC=off)
    with mtp_depth(0):
        assert mtp.maybe_speculate(toy[2], toy[0], SampleParams(temperature=0.0)) is None


def test_penalties_agree_under_mtp_except_at_a_near_tie(toy):
    """Greedy-with-penalties is still greedy, and MTP must not quietly become
    plain argmax: the per-row pick runs through sampling.sample_next with the
    id history the serial loop would have had — so it agrees with the serial
    stream except where a near-tie in the (penalized) row lets the two arms'
    accumulation-order differences flip the pick (spec_agree)."""
    from drinkme.serving import engines
    from spec_agree import assert_agrees_or_forks_at_a_near_tie, capture, ordered_ids

    params = dict(temperature=0.0, max_tokens=48, repetition_penalty=1.35,
                  presence_penalty=0.4, frequency_penalty=0.3)
    picks_off, rows_off, restore = capture(engines)
    try:
        off, _ = _run(_engine(toy, False), _msgs(PROMPTS[6]), SampleParams(**params))
    finally:
        restore()
    picks_on, rows_on, restore = capture(engines)
    try:
        with mtp_depth(4):
            on, _ = _run(_engine(toy, True), _msgs(PROMPTS[6]), SampleParams(**params))
    finally:
        restore()
    assert_agrees_or_forks_at_a_near_tie(ordered_ids(picks_on), ordered_ids(picks_off),
                                         rows_on, rows_off)


def test_depth_from_env(capsys):
    # Unset = AUTO (None): default-on when the model allows.
    # DRINKME_MTP_DEPTH is a depth only: 0, negative and garbage are not
    # depths, so they warn and mean AUTO; DRINKME_SPEC=off is the off switch.
    for raw, want in (("", None), ("4", 4), ("7", 7)):
        with _depth_env(raw):
            assert mtp.depth_from_env() == want
    for raw in ("0", "-3", "banana"):
        with _depth_env(raw):
            assert mtp.depth_from_env() is None
        assert "not a draft depth" in capsys.readouterr().err
    assert mtp.DEFAULT_DEPTH == 4


class _unset_mtp:
    def __enter__(self):
        self.old = os.environ.pop("DRINKME_MTP_DEPTH", None)

    def __exit__(self, *a):
        if self.old is not None:
            os.environ["DRINKME_MTP_DEPTH"] = self.old


class _depth_env:
    """DRINKME_MTP_DEPTH set to `raw` ("" = unset) for a block."""

    def __init__(self, raw):
        self.raw = raw

    def __enter__(self):
        self.old = os.environ.pop("DRINKME_MTP_DEPTH", None)
        if self.raw:
            os.environ["DRINKME_MTP_DEPTH"] = self.raw
        mtp._warned.clear()

    def __exit__(self, *a):
        os.environ.pop("DRINKME_MTP_DEPTH", None)
        if self.old is not None:
            os.environ["DRINKME_MTP_DEPTH"] = self.old
        mtp._warned.clear()


def test_depth_past_the_fused_window_warns(toy, capsys):
    with mtp_depth(8):
        assert mtp.depth_from_env() == 8
    err = capsys.readouterr().err
    assert "M=9" in err and "fused window" in err


def test_engine_without_a_head_is_untouched(toy, serial_runs):
    """DRINKME_MTP_DEPTH set but no head (a non-qwen3_5 model, or a checkpoint with
    no mtp.* tensors) must serve the default path, not fail. Both sides are
    the plain serial loop (no head means DRINKME_MTP_DEPTH has nothing to turn
    on), so this is a same-arm determinism check, not a speculation-on-vs-off
    one — bitwise equality is the right bar here."""
    eng = _engine(toy, with_mtp=False)
    with mtp_depth(4):
        res, _ = _run(eng, _msgs(PROMPTS[1]))
    assert res.text == serial_runs[PROMPTS[1]]["res"].text


def test_is_supported_is_qwen3_5_only(toy):
    from transformers import LlamaConfig

    assert mtp.is_supported(toy[3])
    assert not mtp.is_supported(LlamaConfig())


def test_load_head_refuses_a_foreign_family(tmp_path, capsys):
    from transformers import LlamaConfig, LlamaForCausalLM

    cfg = LlamaConfig(vocab_size=32, hidden_size=16, intermediate_size=32,
                      num_hidden_layers=1, num_attention_heads=2)
    model = LlamaForCausalLM(cfg).eval()
    assert mtp.load_head(model, str(tmp_path), None, "cpu") is None
    assert "has no MTP head" in capsys.readouterr().err


# ------------------------------------------------------- the load path --


def test_load_head_streams_the_checkpoint_tensors(toy, tmp_path):
    """The 15 `mtp.*` tensors, out of a real safetensors shard, into the real
    module tree — name for name, no translation table."""
    from safetensors.torch import save_file

    model, _, head, cfg = toy
    ckpt = {f"mtp.{k}": v.clone() for k, v in head.state_dict().items()}
    # a decoy from the trunk: load_head must take only the mtp.* keys
    ckpt["model.embed_tokens.weight"] = torch.zeros(4, 4)
    save_file(ckpt, str(tmp_path / "model-00001-of-00001.safetensors"))

    loaded = mtp.load_head(model, str(tmp_path), None, "cpu")
    assert loaded is not None
    assert not any(p.is_meta for p in loaded.parameters())
    got = loaded.state_dict()
    assert len(got) == 15, sorted(got)
    for k, v in head.state_dict().items():
        assert torch.equal(got[k], v), k
    # and it drafts: same tokens as the head the toy already uses
    cache_ids = torch.tensor([[5, 7, 11]])
    from transformers import StaticCache

    for h in (head, loaded):
        h.reset()
    with torch.inference_mode():
        cache = StaticCache(config=cfg, max_cache_len=32)
        hidden, _ = mtp.forward_with_hidden(model, cache_ids, cache,
                                            torch.arange(3))
        a = head.draft(hidden[:, -1:], 5, 2, 3)
        b = loaded.draft(hidden[:, -1:], 5, 2, 3)
    assert a.tolist() == b.tolist()  # the verify rows: [token, 3 drafts]


def test_load_head_reports_an_incomplete_head(toy, tmp_path):
    """Half a head would draft garbage and tank acceptance silently. Refuse."""
    from safetensors.torch import save_file

    model, _, head, _ = toy
    partial = {f"mtp.{k}": v.clone() for k, v in head.state_dict().items()
               if "mlp" not in k}
    save_file(partial, str(tmp_path / "model.safetensors"))
    with pytest.raises(ValueError, match="MTP head incomplete"):
        mtp.load_head(model, str(tmp_path), None, "cpu")


# ---------------------------------- D1: default engages at the real seam --
# depth_from_env/DEFAULT_DEPTH are pinned above; these pin the decision
# _mtp_head actually makes (engines.py), which nothing exercised before.


def _write_head_ckpt(tmp_path, head):
    from safetensors.torch import save_file

    ckpt = {f"mtp.{k}": v.clone() for k, v in head.state_dict().items()}
    save_file(ckpt, str(tmp_path / "model.safetensors"))
    return tmp_path


def test_mtp_head_default_engages_when_the_checkpoint_carries_one(toy, tmp_path):
    from drinkme.serving import engines

    model, _, head, _ = toy
    _write_head_ckpt(tmp_path, head)
    with _unset_mtp():
        loaded = engines._mtp_head(model, str(tmp_path), None, "cpu")
    assert loaded is not None
    assert not any(p.is_meta for p in loaded.parameters())


def test_mtp_head_default_stays_off_without_one(toy, tmp_path):
    """AUTO on a checkpoint with no mtp.* tensors decodes without one,
    quietly — the everyday case; most checkpoints have no head."""
    from safetensors.torch import save_file

    from drinkme.serving import engines

    model, _, _, _ = toy
    save_file({"model.embed_tokens.weight": torch.zeros(4, 4)},
              str(tmp_path / "model.safetensors"))
    with _unset_mtp():
        loaded = engines._mtp_head(model, str(tmp_path), None, "cpu")
    assert loaded is None


def test_mtp_head_is_not_loaded_under_spec_off_even_with_a_head(toy, tmp_path):
    from drinkme.serving import engines

    model, _, head, _ = toy
    _write_head_ckpt(tmp_path, head)
    with mtp_depth(0):
        loaded = engines._mtp_head(model, str(tmp_path), None, "cpu")
    assert loaded is None


def test_mtp_head_forced_depth_still_loud_fails_on_no_head(tmp_path, capsys):
    """=k unchanged, including the loud no-head failure load_head has always
    given a foreign family — through the real _mtp_head seam this time."""
    from transformers import LlamaConfig, LlamaForCausalLM

    from drinkme.serving import engines

    cfg = LlamaConfig(vocab_size=32, hidden_size=16, intermediate_size=32,
                      num_hidden_layers=1, num_attention_heads=2)
    model = LlamaForCausalLM(cfg).eval()
    with mtp_depth(4):
        loaded = engines._mtp_head(model, str(tmp_path), None, "cpu")
    assert loaded is None
    assert "has no MTP head" in capsys.readouterr().err


# --------------------------------------------- D2: guard 1, residency -----


def test_head_gib_reads_the_safetensors_header(toy, tmp_path):
    model, _, head, _ = toy
    _write_head_ckpt(tmp_path, head)
    hg = mtp.head_gib(str(tmp_path), None)
    expected = sum(v.numel() * v.element_size()
                   for v in head.state_dict().values()) / (1024 ** 3)
    assert hg == pytest.approx(expected)


def test_head_gib_is_none_without_mtp_tensors(tmp_path):
    from safetensors.torch import save_file

    save_file({"model.embed_tokens.weight": torch.zeros(4, 4)},
              str(tmp_path / "model.safetensors"))
    assert mtp.head_gib(str(tmp_path), None) is None


def test_residency_check_reads_suggests_own_margin_and_yields_below_it():
    """Do not invent a new number: the margin is suggest.FIT_HEADROOM."""
    from drinkme.suggest import FIT_HEADROOM

    used_gib = 40.0
    yields, left, margin = mtp.residency_check(free_gib=10.0, used_gib=used_gib,
                                               head_gib=6.5)
    assert margin == pytest.approx(used_gib * (FIT_HEADROOM - 1))
    assert left == pytest.approx(3.5)
    assert yields  # 3.5 GB left is under a 4.0 GB margin


def test_residency_check_comfortable_fit_does_not_yield():
    yields, left, margin = mtp.residency_check(free_gib=10.0, used_gib=40.0,
                                               head_gib=2.0)
    assert not yields
    assert left == pytest.approx(8.0)


def test_mtp_head_yields_on_a_knife_edge_fit(toy, tmp_path, monkeypatch, capsys):
    """Guard 1, end to end at the real decision point: a fixture fit that
    leaves less than the margin yields — not loaded, one stderr line naming
    the margin."""
    from drinkme.suggest import FIT_HEADROOM

    from drinkme.serving import engines

    model, _, head, _ = toy
    _write_head_ckpt(tmp_path, head)
    hg = mtp.head_gib(str(tmp_path), None)
    used_gib = 40.0
    margin = used_gib * (FIT_HEADROOM - 1)
    free_gib = hg + margin - 0.01  # just under the margin once the head loads
    monkeypatch.setattr(mtp, "device_headroom_gib",
                        lambda device: (free_gib, used_gib))
    with _unset_mtp():
        loaded = engines._mtp_head(model, str(tmp_path), None, "cpu")
    assert loaded is None
    err = capsys.readouterr().err
    assert "head would leave" in err and "headroom" in err
    assert "MTP yields — DRINKME_MTP_DEPTH=4 to force" in err


def test_mtp_head_forced_depth_overrides_the_yield(toy, tmp_path, monkeypatch):
    """A forced DRINKME_MTP_DEPTH=k overrides the yield — that person asked."""
    from drinkme.suggest import FIT_HEADROOM

    from drinkme.serving import engines

    model, _, head, _ = toy
    _write_head_ckpt(tmp_path, head)
    hg = mtp.head_gib(str(tmp_path), None)
    used_gib = 40.0
    margin = used_gib * (FIT_HEADROOM - 1)
    free_gib = hg + margin - 0.01  # the same hostile fixture as the yield test
    monkeypatch.setattr(mtp, "device_headroom_gib",
                        lambda device: (free_gib, used_gib))
    with mtp_depth(4):
        loaded = engines._mtp_head(model, str(tmp_path), None, "cpu")
    assert loaded is not None


def test_mtp_head_comfortable_fit_loads(toy, tmp_path, monkeypatch):
    from drinkme.serving import engines

    model, _, head, _ = toy
    _write_head_ckpt(tmp_path, head)
    monkeypatch.setattr(mtp, "device_headroom_gib", lambda device: (100.0, 40.0))
    with _unset_mtp():
        loaded = engines._mtp_head(model, str(tmp_path), None, "cpu")
    assert loaded is not None


# ---------------------------------------------------------- the reporting --


def test_acceptance_stats_are_reported(toy):
    eng = _engine(toy, with_mtp=True)
    spec = None
    real = mtp.Speculator.finish

    def spy(self, appended):
        nonlocal spec
        spec = self
        return real(self, appended)

    mtp.Speculator.finish = spy
    try:
        with mtp_depth(3):
            _run(eng, _msgs(PROMPTS[0]), _greedy(max_tokens=24))
    finally:
        mtp.Speculator.finish = real
    s = spec.stats()
    # the last cycles draft fewer than k: the depth is clamped to what
    # max_tokens still allows, so nothing is drafted that can never be emitted
    assert s["cycles"] > 0 and 0 < s["drafted"] <= 3 * s["cycles"]
    assert 0.0 <= s["acceptance"] <= 1.0
    assert s["accepted"] <= s["drafted"]


# ------------------------------------------------- the draft diet, lever 1 --


class draft_vocab_env:
    """DRINKME_MTP_DRAFT_VOCAB for a block."""

    def __init__(self, v):
        self.v = v

    def __enter__(self):
        self.old = os.environ.get("DRINKME_MTP_DRAFT_VOCAB")
        if self.v is None:
            os.environ.pop("DRINKME_MTP_DRAFT_VOCAB", None)
        else:
            os.environ["DRINKME_MTP_DRAFT_VOCAB"] = str(self.v)
        mtp._warned.clear()
        return self

    def __exit__(self, *a):
        if self.old is None:
            os.environ.pop("DRINKME_MTP_DRAFT_VOCAB", None)
        else:
            os.environ["DRINKME_MTP_DRAFT_VOCAB"] = self.old
        mtp._warned.clear()


def test_draft_vocab_does_not_move_the_stream(toy, serial_runs):
    """THE lever-1 property, and the only one that matters: restricting what a
    draft may PROPOSE cannot change what is EMITTED. Verify runs the full
    lm_head over every row and picks with the engine's own sampler, so a
    subset the draft cannot reach costs acceptance and nothing else — the
    stream still agrees with the serial one except at a genuine near-tie
    (spec_agree).

    Same 20 prompts x 64 tokens as the MTP identity test, with the draft
    argmax confined to a third of the toy's vocab."""
    from drinkme.serving import engines
    from spec_agree import assert_agrees_or_forks_at_a_near_tie, capture, ordered_ids

    model, _, head, cfg = toy
    mtp.install_draft_vocab(head, model)  # no env yet -> stays full-vocab
    assert head.draft_proj is None
    eng = _engine(toy, with_mtp=True)
    try:
        with draft_vocab_env(cfg.vocab_size // 3):
            mtp.install_draft_vocab(head, model)
            assert head.draft_proj is not None
            assert head.draft_proj.ids.numel() == cfg.vocab_size // 3
            with mtp_depth(4):
                for p in PROMPTS:
                    picks, rows, restore = capture(engines)
                    try:
                        res, _ = _run(eng, _msgs(p))
                    finally:
                        restore()
                    ref = serial_runs[p]
                    assert_agrees_or_forks_at_a_near_tie(ordered_ids(picks), ref["ids"],
                                                         rows, ref["rows"])
    finally:
        head.draft_proj = None


def test_draft_vocab_proposes_only_from_the_subset(toy):
    """And it really is restricted: the proposal is the full-vocab argmax
    RESTRICTED to the subset, not the full-vocab argmax."""
    from drinkme.serving import draft_vocab

    _, _, head, cfg = toy
    ids = torch.arange(10, cfg.vocab_size, 3)
    proj, note = draft_vocab.make_projection(head.lm_head, ids)
    assert "GATHERED" in note  # the toy's head is a raw nn.Linear
    torch.manual_seed(3)
    for _ in range(8):
        hn = torch.randn(1, 1, cfg.hidden_size)
        full = head.lm_head(hn[:, -1])
        head.draft_proj = proj
        try:
            got = int(head._propose(hn))  # a [1] device tensor (no sync in the chain)
        finally:
            head.draft_proj = None
        assert got in set(ids.tolist())
        assert got == int(ids[int(full[0, ids].argmax())])


def test_draft_projection_refuses_a_multi_row_call(toy):
    """The verify pass must never reach this object — a silent slow path for
    M>1 would hide a routing bug instead of surfacing it."""
    from drinkme.serving import draft_vocab

    _, _, head, cfg = toy
    proj, _ = draft_vocab.make_projection(head.lm_head, torch.arange(0, 32))
    with pytest.raises(ValueError, match="one row"):
        proj(torch.randn(3, cfg.hidden_size))


@pytest.mark.parametrize("raw,expect", [
    (None, None), ("", None), ("0", None), ("-4", None),
    ("999999", None),                 # >= vocab: nothing to cut
    ("/nope/missing.json", None),     # unreadable file
    ("32", 32),
])
def test_draft_vocab_env_degrades_to_full_vocab(toy, raw, expect):
    """Every failure mode serves the full vocab and says so — a lever that
    buys speed must cost speed when it breaks, never correctness."""
    from drinkme.serving import draft_vocab

    model, _, _, cfg = toy
    with draft_vocab_env(raw):
        got = draft_vocab.ids_from_env(model.config, cfg.vocab_size,
                                       lambda *a: None)
    if expect is None:
        assert got is None
    else:
        ids, source = got
        # the config's specials are unioned in; the toy declares none
        assert ids.numel() == expect and ids[0] == 0 and ids[-1] == expect - 1
        assert "CRUDE" in source


def test_draft_vocab_from_a_file(toy, tmp_path):
    from drinkme.serving import draft_vocab

    model, _, _, cfg = toy
    f = tmp_path / "sub.json"
    f.write_text('{"ids": [5, 5, 2, 90], "method": "test"}')
    with draft_vocab_env(str(f)):
        ids, source = draft_vocab.ids_from_env(model.config, cfg.vocab_size,
                                               lambda *a: None)
    assert ids.tolist() == [2, 5, 90]  # sorted, deduped
    assert "test" in source


def test_forward_with_hidden_last_row_only_narrows_logits_not_meaning(toy):
    """Prefill keeps ONE row of lm_head; the cycle must still get every row.

    The all-rows behaviour was deliberate (see forward_with_hidden's
    docstring) and priced against a 4096-token prompt at ~2GB. At vocab
    248320 it costs 0.474 MiB per prompt token, so at the 262144 context the
    server advertises the discarded logits alone are 121.2 GiB on a 124 GiB
    box — found by a live 500 on a 21k-token prompt.

    Pins three things, the third being the one that matters:
      1. narrowed  -> exactly 1 logits row
      2. default   -> every row, so the draft cycle (which reads logits[m])
                      can never be silently narrowed by a later edit
      3. the narrowed row EQUALS the last row of the full computation, i.e.
         this changes HOW MANY rows are produced, not WHICH logits they are
    """
    import torch
    from transformers import StaticCache

    model, _, _, cfg = toy
    ids = torch.tensor([[5, 7, 11, 2, 9]])
    T = ids.shape[1]

    with torch.inference_mode():
        cache_full = StaticCache(config=cfg, max_cache_len=32)
        hidden_full, logits_full = mtp.forward_with_hidden(
            model, ids, cache_full, torch.arange(T))

        cache_last = StaticCache(config=cfg, max_cache_len=32)
        hidden_last, logits_last = mtp.forward_with_hidden(
            model, ids, cache_last, torch.arange(T), last_row_only=True)

    # 1 + 2: the shapes are the contract
    assert logits_full.shape[0] == T, "the draft cycle needs every verify row"
    assert logits_last.shape[0] == 1, "prefill must keep exactly one row"
    assert logits_full.shape[1] == logits_last.shape[1], "vocab must not move"

    # hidden is untouched either way — the draft head reads hidden[:, -1:]
    assert hidden_full.shape == hidden_last.shape == (1, T, cfg.hidden_size)

    # 3: same logits, fewer of them. On CPU with a toy model this is exact;
    # on the compressed GPU arm a 1-row lm_head takes the GEMV kernel and a
    # many-row one the dense GEMM, which agree only to accumulation order —
    # that difference is what the greedy A/B gates measure, and is why both
    # prefill arms had to narrow in the same commit.
    assert torch.equal(logits_last[0], logits_full[-1])


def test_device_headroom_accepts_every_spelling_of_the_accelerator(monkeypatch):
    """"cuda", "cuda:0" and torch.device("cuda") must all measure; a string
    compare (`device != "cuda"`) left the residency guard inert on the real
    box for two of the three. cpu still returns None."""
    import torch
    from drinkme.serving import mtp
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda: (80 * 2**30, 124 * 2**30))
    for dev in ("cuda", "cuda:0", torch.device("cuda")):
        got = mtp.device_headroom_gib(dev)
        assert got is not None, dev
        assert abs(got[0] - 80.0) < 1e-6 and abs(got[1] - 44.0) < 1e-6, (dev, got)
    assert mtp.device_headroom_gib("cpu") is None
    assert mtp.device_headroom_gib(torch.device("cpu")) is None


# ======================================================================
# THE HEAD DIET: the MTP head's eligible Linears served out of
# the pack, compressed, exactly as the trunk's are.
#
# A second toy, because the one above cannot carry this feature: its hidden
# size is 64, so NOTHING on its head clears codec eligibility (min dim >=
# 1024) and a "packed head" there would be an empty pack. This one is bf16 at
# hidden 1024 with 8 KV heads, which puts six of the head's eight Linears over
# the bar and leaves k/v (256 rows) under it — so the compressed-attach and
# the streamed-raw paths are both exercised on one head, as
# test_serving_engines' Llama toy does for the trunk.
# ======================================================================


def _diet_cfg():
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

    return Qwen3_5TextConfig(
        vocab_size=96, hidden_size=1024, intermediate_size=1024,
        num_hidden_layers=1, num_attention_heads=32, num_key_value_heads=8,
        head_dim=32, linear_key_head_dim=16, linear_value_head_dim=16,
        linear_num_key_heads=2, linear_num_value_heads=4,
        linear_conv_kernel_dim=4,
        layer_types=["full_attention"], max_position_embeddings=256,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0,
                         "mrope_section": [2, 1, 1], "mrope_interleaved": True,
                         "partial_rotary_factor": 0.25},
        tie_word_embeddings=False, eos_token_id=None, pad_token_id=None)


@pytest.fixture(scope="session")
def diet(tmp_path_factory):
    """(model_dir, pack_dir, plain_pack_dir, model, tokenizer, cfg).

    A real qwen3_5 checkpoint on disk carrying a real `mtp.*` head, packed
    twice: once with the head (`pack_dir`) and once with `mtp=False`
    (`plain_pack_dir`, a pack without a head sub-pack).
    """
    from safetensors.torch import load_file, save_file
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

    from drinkme.codec.pack import pack_model

    d = tmp_path_factory.mktemp("diet")
    model_dir = str(d / "model")
    cfg = _diet_cfg()
    torch.manual_seed(0)
    model = Qwen3_5ForCausalLM(cfg).to(torch.bfloat16).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    vocab = {"<unk>": 0, "<pad>": 1, "</s>": 2}
    for w in WORDS:
        vocab[w] = len(vocab)
    while len(vocab) < cfg.vocab_size:
        vocab[f"tok{len(vocab)}"] = len(vocab)
    backend = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
    backend.pre_tokenizer = Whitespace()
    tok = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="<unk>",
                                  pad_token="<pad>")
    tok.chat_template = ("{% for m in messages %}{{ m['role'] }} : {{ m['content'] }} "
                         "{% endfor %}{% if add_generation_prompt %}assistant :{% endif %}")
    tok.save_pretrained(model_dir)
    model.save_pretrained(model_dir, safe_serialization=True)

    # the head, written into the checkpoint under `mtp.` exactly as the 27B
    # ships it — /64 for the same reason test_serving_engines scales its toy:
    # it pulls exponents off the escape trapdoor so the packs carry real
    # escapes instead of testing a degenerate all-in-window case.
    head = _fill_random(mtp.MTPHead(cfg, model), seed=3, scale=0.08 / 64)
    shard = os.path.join(model_dir, "model.safetensors")
    tensors = load_file(shard)
    tensors.update({f"mtp.{k}": v.to(torch.bfloat16).contiguous()
                    for k, v in head.state_dict().items()})
    save_file(tensors, shard, metadata={"format": "pt"})
    del tensors

    pack_dir, plain = str(d / "pack"), str(d / "plain")
    pack_model(model_dir, None, pack_dir, progress=lambda *_: None)
    pack_model(model_dir, None, plain, progress=lambda *_: None, mtp=False)
    return model_dir, pack_dir, plain, model, tok, cfg


def _head_meta(pack_dir):
    from drinkme.codec.pack import read_mtp_meta

    return read_mtp_meta(pack_dir)


def _compressed_names(head):
    from drinkme.codec.swap import CompressedLinear

    return sorted(n for n, m in head.named_modules()
                  if isinstance(m, CompressedLinear))


# -- R1: the head packs, and the trunk does not notice -------------------


def test_head_skeleton_is_built_at_the_trunks_dtype(diet):
    """Eligibility is a dtype test, so a float32 skeleton for a bf16
    checkpoint answers "nothing is eligible" and the diet silently packs an
    empty head. Pin the dtype at the source."""
    from drinkme.codec.swap import eligible_linears

    _, _, _, model, _, cfg = diet
    head = mtp.MTPHead(cfg, model)
    assert head.fc.weight.dtype == torch.bfloat16
    assert [n for n, _, _, _ in eligible_linears(head)] == [
        "fc", "layers.0.self_attn.q_proj", "layers.0.self_attn.o_proj",
        "layers.0.mlp.gate_proj", "layers.0.mlp.up_proj",
        "layers.0.mlp.down_proj"]
    # ...and a float32 trunk still gets a float32 head: this follows the
    # trunk, it does not impose bf16 on anybody.
    f32 = mtp.MTPHead(cfg, model.to(torch.float32))
    assert f32.fc.weight.dtype == torch.float32
    model.to(torch.bfloat16)


def test_pack_model_writes_a_head_subpack(diet):
    from drinkme.codec.pack import has_mtp_pack

    _, pack_dir, plain, _, _, _ = diet
    assert has_mtp_pack(pack_dir)
    assert not has_mtp_pack(plain)
    hm = _head_meta(pack_dir)
    assert hm["module"] == "mtp"
    assert sorted(hm["tensors"]) == [
        "fc", "layers.0.mlp.down_proj", "layers.0.mlp.gate_proj",
        "layers.0.mlp.up_proj", "layers.0.self_attn.o_proj",
        "layers.0.self_attn.q_proj"]
    # the codec's own numbers, on the head's tensors
    assert 9.0 < hm["meanBpw"] < 13.5, hm["meanBpw"]
    assert hm["radixTensorCount"] == hm["tensorCount"]  # every head Linear coded, none fell back
    assert hm["formatVersion"] == 1 and hm["sha256"]  # hashed like any pack
    assert hm["profile"] == "sip" and hm["profileWidths"] == [3, 8] and hm["rawFallbackTensorCount"] == 0
    assert "method" not in hm
    # the three residency numbers, all measured
    assert hm["rawBytes"] == 2 * (1024 * 2048 + 2048 * 1024 + 3 * 1024 * 1024
                                  + 1024 * 1024)
    assert hm["packedBytes"] < hm["rawBytes"]
    # resident = the streams + directory + padded palette; the npz adds only
    # its container overhead (a few hundred bytes per array), so the two
    # agree within a percent
    assert hm["residentBytes"] < hm["rawBytes"]
    assert abs(hm["residentBytes"] - hm["packedBytes"]) < 0.01 * hm["packedBytes"]
    # k/v (256 rows) and the seven norms stayed raw and are still charged
    assert hm["unpackedTensorCount"] == 9
    assert hm["unpackedBytes"] == 2 * (2 * 256 * 1024 + 5 * 1024 + 2 * 32)


def test_the_head_subpack_leaves_the_trunk_pack_byte_identical(diet):
    """The receipt that matters for shipping: adding the head changes the
    trunk's files not at all. PackWriter numbers files by sorted position
    WITHIN its own name set, and the head has its own writer — so t0000.npz
    is the same tensor and the same bytes with the head on or off."""
    import hashlib

    _, pack_dir, plain, _, _, _ = diet
    a = json.load(open(os.path.join(pack_dir, "meta.json")))
    b = json.load(open(os.path.join(plain, "meta.json")))
    assert a["tensors"] == b["tensors"]
    assert a["sha256"] == b["sha256"]

    def digest(root, fn):
        return hashlib.sha256(open(os.path.join(root, fn), "rb").read()).hexdigest()

    for fn in a["tensors"].values():
        assert digest(pack_dir, fn) == digest(plain, fn), fn
    # and the trunk meta gained only additive keys
    assert set(a) - set(b) == {"mtpTensorCount", "mtpMeanBpw",
                               "mtpResidentBytes", "mtpRawBytes"}


def test_pack_without_a_head_in_the_checkpoint_writes_no_subpack(toy, tmp_path):
    """qwen3_5 says a head is POSSIBLE; the checkpoint says whether this one
    has it. A head-less checkpoint must leave no empty sub-pack behind."""
    from drinkme.codec.pack import has_mtp_pack, pack_model

    model, _, _, cfg = toy
    model_dir, pack_dir = str(tmp_path / "m"), str(tmp_path / "p")
    model.config.save_pretrained(model_dir)
    model.to(torch.bfloat16).save_pretrained(model_dir, safe_serialization=True)
    model.to(torch.float32)
    pack_model(model_dir, None, pack_dir, progress=lambda *_: None)
    assert not has_mtp_pack(pack_dir)
    assert not os.path.exists(os.path.join(pack_dir, "mtp"))


# -- R2/R3: it loads, and it is the same head ----------------------------


def _load(diet, packed: bool, device="cpu"):
    model_dir, pack_dir, _, model, _, _ = diet
    return mtp.load_head(model, model_dir, None, device,
                         pack_dir=pack_dir if packed else None)


def test_head_pack_installs_compressed_linears(diet, capsys):
    head = _load(diet, packed=True)
    assert _compressed_names(head) == [
        "fc", "layers.0.mlp.down_proj", "layers.0.mlp.gate_proj",
        "layers.0.mlp.up_proj", "layers.0.self_attn.o_proj",
        "layers.0.self_attn.q_proj"]
    # the ineligible ones streamed raw, as they do on the trunk
    assert isinstance(head.layers[0].self_attn.k_proj, torch.nn.Linear)
    assert not any(p.is_meta for p in head.parameters())
    out = capsys.readouterr().out
    assert "DIET on: 6 tensors compressed" in out


def test_the_dieted_head_decodes_the_checkpoints_exact_bytes(diet):
    """The codec claim, on the head: every compressed tensor decodes to the
    bf16 bytes the checkpoint shipped, bit for bit."""
    from safetensors.torch import load_file

    model_dir, _, _, _, _, _ = diet
    ckpt = load_file(os.path.join(model_dir, "model.safetensors"))
    head = _load(diet, packed=True)
    for name in _compressed_names(head):
        mod = head.get_submodule(name)
        got = mod._cpu_weight()
        want = ckpt[f"mtp.{name}.weight"]
        assert got.dtype == want.dtype == torch.bfloat16
        assert torch.equal(got, want), name


def test_the_dieted_head_drafts_bit_identically(diet):
    """Not close — EQUAL. The reference decode is byte-exact by construction
    and CompressedLinear's CPU arm is stock's own F.linear over those bytes,
    so any plumbing error (wrong tensor, wrong module, a transpose) shows up
    as a changed bit here."""
    from transformers import StaticCache

    _, _, _, model, _, cfg = diet
    raw, packed = _load(diet, packed=False), _load(diet, packed=True)
    ids = torch.tensor([[5, 7, 11, 13]])
    for h in (raw, packed):
        h.reset()
    with torch.inference_mode():
        cache = StaticCache(config=cfg, max_cache_len=64)
        hidden, _ = mtp.forward_with_hidden(model, ids, cache, torch.arange(4))
        a = raw.run(hidden, ids, 0)
        b = packed.run(hidden, ids, 0)
        assert torch.equal(a, b)
        assert torch.equal(raw.lm_head(a[:, -1]), packed.lm_head(b[:, -1]))
        assert torch.equal(raw.draft(hidden[:, -1:], 5, 3, 4),
                           packed.draft(hidden[:, -1:], 5, 3, 4))


def test_diet_opt_out_serves_the_raw_head(diet, monkeypatch):
    """DRINKME_MTP_DIET=0: the default head, the default residency charge, out of a
    pack that has a head sub-pack sitting right there."""
    monkeypatch.setenv("DRINKME_MTP_DIET", "0")
    head = _load(diet, packed=True)
    assert _compressed_names(head) == []
    assert isinstance(head.fc, torch.nn.Linear)
    assert head.fc.weight.dtype == torch.bfloat16


def test_a_pack_without_a_head_subpack_loads_the_raw_head(diet):
    """A pack without a head sub-pack: nothing changes for it."""
    model_dir, _, plain, model, _, _ = diet
    head = mtp.load_head(model, model_dir, None, "cpu", pack_dir=plain)
    assert _compressed_names(head) == []
    assert not any(p.is_meta for p in head.parameters())


# -- R2: the residency guard charges the measured compressed size --------


def test_residency_charge_is_the_measured_compressed_size(diet):
    model_dir, pack_dir, plain, _, _, _ = diet
    hm = _head_meta(pack_dir)
    raw = mtp.head_gib(model_dir, None)
    packed = mtp.head_gib(model_dir, None, pack_dir)
    assert packed == pytest.approx(
        (hm["residentBytes"] + hm["unpackedBytes"]) / 1024 ** 3)
    assert packed < raw
    # a pack with no head sub-pack, and the opt-out, both charge raw
    assert mtp.head_gib(model_dir, None, plain) == raw
    with _diet_off():
        assert mtp.head_gib(model_dir, None, pack_dir) == raw


class _diet_off:
    def __enter__(self):
        os.environ["DRINKME_MTP_DIET"] = "0"

    def __exit__(self, *a):
        del os.environ["DRINKME_MTP_DIET"]
        return False


def test_the_charge_matches_what_the_load_actually_puts_on_the_device(diet):
    """meta.json's residentBytes is a pack-time PREDICTION of a load-time
    fact. Measure the fact and compare — a guard that charges a number
    nothing serves is not a guard."""
    from drinkme.codec.pack import resident_bytes

    hm = _head_meta(diet[1])
    head = _load(diet, packed=True)
    measured = sum(resident_bytes(head.get_submodule(n).p)
                   for n in _compressed_names(head))
    assert measured == hm["residentBytes"]


def test_engine_seam_passes_the_pack_dir_to_the_head(diet):
    """_mtp_head is where the decision is actually made; pin it there."""
    from drinkme.serving import engines

    model_dir, pack_dir, _, model, _, _ = diet
    with _unset_mtp():
        head = engines._mtp_head(model, model_dir, None, "cpu", pack_dir)
    assert head is not None and len(_compressed_names(head)) == 6
    with _unset_mtp():
        stock = engines._mtp_head(model, model_dir, None, "cpu")
    assert stock is not None and _compressed_names(stock) == []


# -- R3: the acceptance counters are the audit's own instrument ----------


def _diet_engine(diet, head):
    _, _, _, model, tok, _ = diet
    os.environ["DRINKME_PREFIX_SLOTS"] = "0"
    try:
        return HFEngine(model, tok, model_id="diet", arm="test", meta={},
                        ctx=256, mtp_head=head)
    finally:
        del os.environ["DRINKME_PREFIX_SLOTS"]


@pytest.mark.parametrize("prompt", ["hello world the quick brown fox",
                                    "alpha beta gamma delta"])
def test_the_dieted_head_moves_neither_the_stream_nor_the_counters(diet, prompt):
    """The MTP acceptance counters ARE the rejection-sampling audit's
    instrument (docs/serve-speculation.md). With bit-identical head weights the
    drafts are the same drafts, so on CPU "within noise" is exactly equal —
    which is a far stronger pin than a tolerance, and it fails loudly if the
    diet ever stops being lossless."""
    seen = {}
    for packed in (False, True):
        head = _load(diet, packed=packed)
        eng = _diet_engine(diet, head)
        stats = {}
        real_finish = mtp.Speculator.finish

        def finish(self, appended, _s=stats, _r=real_finish):
            _s.update(self.stats())
            return _r(self, appended)

        mtp.Speculator.finish = finish
        try:
            with mtp_depth(4):
                r, _ = _run(eng, _msgs(prompt), _greedy(max_tokens=24))
        finally:
            mtp.Speculator.finish = real_finish
        seen[packed] = (r.text, stats)
    assert seen[False][0] == seen[True][0]
    assert seen[False][1]["cycles"] and seen[False][1]["drafted"]
    for k in ("cycles", "drafted", "accepted", "acceptance", "bailed"):
        assert seen[False][1][k] == seen[True][1][k], k


def test_retrofitting_a_head_onto_an_existing_pack(diet, tmp_path):
    """The shipping path: the 8B and 27B packs cost hours and are hashed
    tensor by tensor, so giving them a head must not mean repacking the
    trunk. pack_mtp_head writes the sub-pack alone — and writes the SAME
    bytes the integrated pass would have, which is what makes the retrofit
    equivalent rather than merely similar."""
    import shutil

    from drinkme.codec.pack import has_mtp_pack, pack_mtp_head

    model_dir, pack_dir, plain, _, _, _ = diet
    retro = str(tmp_path / "retro")
    shutil.copytree(plain, retro)
    before = json.load(open(os.path.join(retro, "meta.json")))
    assert not has_mtp_pack(retro)

    sub = pack_mtp_head(model_dir, None, retro, progress=lambda *_: None)
    assert sub == os.path.join(retro, "mtp") and has_mtp_pack(retro)
    want, got = _head_meta(pack_dir), _head_meta(retro)
    assert got["sha256"] == want["sha256"]  # same tensors, byte for byte
    for k in ("tensors", "meanBpw", "rawBytes", "packedBytes", "residentBytes",
              "unpackedBytes", "unpackedTensorCount", "bpwByTensor"):
        assert got[k] == want[k], k
    # the trunk's own files are untouched; meta.json gained only the head keys
    after = json.load(open(os.path.join(retro, "meta.json")))
    assert after["sha256"] == before["sha256"]
    assert {k: v for k, v in after.items() if not k.startswith("mtp")} == before
    assert after["mtpResidentBytes"] == got["residentBytes"] + got["unpackedBytes"]
    # and it serves
    head = mtp.load_head(diet[3], model_dir, None, "cpu", pack_dir=retro)
    assert len(_compressed_names(head)) == 6


def test_retrofit_on_a_headless_checkpoint_is_a_no_op(toy, tmp_path):
    from drinkme.codec.pack import has_mtp_pack, pack_mtp_head

    model, _, _, _ = toy
    model_dir, pack_dir = str(tmp_path / "m"), str(tmp_path / "p")
    model.config.save_pretrained(model_dir)
    model.to(torch.bfloat16).save_pretrained(model_dir, safe_serialization=True)
    model.to(torch.float32)
    os.makedirs(pack_dir)
    assert pack_mtp_head(model_dir, None, pack_dir, progress=lambda *_: None) is None
    assert not has_mtp_pack(pack_dir)


def test_load_compressed_serves_a_dieted_head(diet):
    """End to end through the REAL loader: a qwen3_5 checkpoint whose pack
    carries a head comes up with that head compressed, and says the same
    words the raw-head arm says.

    This is the test the diet's own dry-run instrument wrote for it: pointing
    bench/mtp_gpu_acceptance.py at a toy found that load_compressed reached
    `model.get_submodule("mtp.fc")` and died, because arms.ckpt_to_skel keyed
    the mtp-skip off the WRAPPER config's model_type alone.
    """
    from drinkme.serving.engines import load_compressed

    model_dir, pack_dir, plain, _, _, _ = diet
    with _unset_mtp():
        eng = load_compressed(model_dir, None, pack_dir, device="cpu", ctx=256)
        raw = load_compressed(model_dir, None, plain, device="cpu", ctx=256)
    assert len(_compressed_names(eng.mtp_head)) == 6
    assert _compressed_names(raw.mtp_head) == []
    with mtp_depth(4):
        a, _ = _run(eng, _msgs("hello world the quick brown fox"), _greedy(max_tokens=16))
        b, _ = _run(raw, _msgs("hello world the quick brown fox"), _greedy(max_tokens=16))
    assert a.text == b.text


def test_a_head_hash_mismatch_refuses(diet, tmp_path):
    """The hash rule is the trunk's, applied to the sub-pack: a pack that
    fails verification is not served, ever. Under an EXPLICIT DRINKME_MTP_DEPTH=k that
    surfaces as the loud refusal (that person asked for exactly this head);
    under AUTO it is contained, and the request decodes without MTP — the
    same containment _mtp_head already gives every other head-load failure.
    """
    import shutil

    from drinkme.codec.pack import mtp_pack_dir
    from drinkme.serving import engines

    model_dir, pack_dir, _, model, _, _ = diet
    broken = str(tmp_path / "broken")
    shutil.copytree(pack_dir, broken)
    victim = os.path.join(mtp_pack_dir(broken), "t0000.npz")
    with open(victim, "r+b") as f:
        f.seek(-4, os.SEEK_END)
        f.write(b"\x00\x00\x00\x00")
    with pytest.raises(ValueError, match="hash mismatch"):
        mtp.load_head(model, model_dir, None, "cpu", pack_dir=broken)
    with _unset_mtp():
        assert engines._mtp_head(model, model_dir, None, "cpu", broken) is None
