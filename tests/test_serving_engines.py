"""The real engine on a toy causal LM — CPU only, no downloads, no GPU.

The toy is a genuine 2-layer LlamaForCausalLM (hidden 1024 so its Linears
clear codec eligibility) plus a genuine WordLevel fast tokenizer, saved to
disk and loaded back through the SAME paths a real model takes: pack_model ->
load_compressed's meta-skeleton streaming, arms.load_cpu -> load_stock. On
CPU CompressedLinear decodes the exact bf16 bytes and uses stock's F.linear,
so "greedy token-identical to stock" is the codec + load path under test, not
kernel numerics (the GPU probe and bench own those).

Toy outputs are deterministic but not meaningful, so tests derive expectations
from a reference greedy run instead of hardcoding token ids.
"""

import http.client
import json
import os

import pytest
import torch

from drinkme.serving.engine import GenerationRequest, SampleParams, complete
from drinkme.serving.engines import HFEngine, load_compressed, load_stock

MSGS = [{"role": "user", "content": "hello world the quick brown fox"}]


def greedy(max_tokens=24, stop=None, seed=None, temperature=0.0):
    return SampleParams(temperature=temperature, seed=seed,
                        stop=stop or [], max_tokens=max_tokens)


def run(eng, params, until=None):
    """generate() + captured deltas; `until` returns False to abort."""
    deltas = []

    def cb(d):
        deltas.append(d)
        return True if until is None else until(deltas)

    return complete(eng, GenerationRequest(MSGS, params), cb), deltas


@pytest.fixture(scope="session")
def toy(tmp_path_factory):
    """Toy model dir + pack dir, built once; everything downstream reuses it."""
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

    from drinkme.codec.pack import pack_model
    from drinkme.codec.swap import eligible_linears

    d = tmp_path_factory.mktemp("toy")
    model_dir, pack_dir = str(d / "model"), str(d / "pack")

    words = ["hello", "world", "the", "quick", "brown", "fox", "jumps", "over",
             "lazy", "dog", "alpha", "beta", "gamma", "delta", "epsilon",
             "user", "assistant", "system", ":"]
    vocab = {"<unk>": 0, "<pad>": 1, "</s>": 2}
    for w in words:
        vocab[w] = len(vocab)
    while len(vocab) < 64:  # fill to vocab_size: every sampleable id decodes
        vocab[f"tok{len(vocab)}"] = len(vocab)
    backend = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
    backend.pre_tokenizer = Whitespace()
    tok = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="<unk>",
                                  pad_token="<pad>", eos_token="</s>")
    tok.chat_template = ("{% for m in messages %}{{ m['role'] }} : {{ m['content'] }} "
                         "{% endfor %}{% if add_generation_prompt %}assistant :{% endif %}")
    tok.save_pretrained(model_dir)

    # hidden 1024 so q/o/gate/up/down clear codec eligibility (min dim >= 1024);
    # kv_heads=2 keeps k/v at 512 wide -> they stay RAW, so the streamed and
    # compressed attach paths are BOTH exercised. attention_bias=True gives the
    # swapped Linears a checkpoint bias (the bias-after-swap case).
    cfg = LlamaConfig(vocab_size=64, hidden_size=1024, intermediate_size=1024,
                      num_hidden_layers=2, num_attention_heads=4,
                      num_key_value_heads=2, max_position_embeddings=256,
                      tie_word_embeddings=False, attention_bias=True,
                      mlp_bias=False, eos_token_id=2, pad_token_id=1)
    torch.manual_seed(0)
    model = LlamaForCausalLM(cfg).to(torch.bfloat16).eval()
    with torch.no_grad():
        for _, _, _, child in eligible_linears(model):
            # pull exponents down while keeping normal tails, so the packs
            # carry a real exponent spread (every tier of the radix codec used)
            child.weight.data.mul_(1 / 64)
    model.save_pretrained(model_dir, safe_serialization=True)
    pack_model(model_dir, None, pack_dir, progress=lambda *_: None)
    meta = json.load(open(os.path.join(pack_dir, "meta.json")))
    assert meta["tensorCount"] == 10 and meta["radixTensorCount"] == 10  # every eligible tensor coded
    return model_dir, pack_dir


@pytest.fixture(scope="session")
def stock(toy):
    return _no_reuse(load_stock, toy[0], device="cpu")


@pytest.fixture(scope="session")
def comp(toy):
    return _no_reuse(load_compressed, toy[0], None, toy[1], device="cpu")


def _no_reuse(loader, *a, **kw):
    """Session engines are built with the prefix cache OFF so every legacy
    test keeps its pre-cache semantics: on the toy's near-uniform logits,
    warm-vs-cold kernel-shape differences (batched prefill vs 1-token gemv)
    flip greedy near-ties that a real model's peaked logits don't — the
    prefix-cache tests below assert bookkeeping on their own engines, and
    text equality on the real model belongs to the GPU verify."""
    os.environ["DRINKME_PREFIX_SLOTS"] = "0"
    try:
        return loader(*a, **kw)
    finally:
        del os.environ["DRINKME_PREFIX_SLOTS"]


@pytest.fixture(scope="session")
def ref(comp):
    """Reference greedy run all derived expectations come from. finish must be
    'length': an early EOS would mean the seed needs changing, loudly."""
    res, deltas = run(comp, greedy(max_tokens=24))
    assert res.finish_reason == "length" and len([d for d in deltas if d]) >= 4
    return res, deltas


# ----------------------------------------------------------------- the load --


def test_no_meta_tensors_remain(comp):
    assert not any(p.is_meta for p in comp.model.parameters())
    assert not any(b.is_meta for _, b in comp.model.named_buffers())


def test_inv_freq_reinstantiated_and_matches_stock(comp, stock):
    inv_c = comp.model.model.rotary_emb.inv_freq
    inv_s = stock.model.model.rotary_emb.inv_freq
    assert not inv_c.is_meta and torch.all(inv_c > 0)
    assert torch.equal(inv_c, inv_s)  # from the same config => same values

def test_compressed_linears_attached_with_streamed_bias(comp, stock):
    from drinkme.codec.swap import CompressedLinear

    q_c = comp.model.model.layers[0].self_attn.q_proj
    q_s = stock.model.model.layers[0].self_attn.q_proj
    assert isinstance(q_c, CompressedLinear)
    # the bias arrived from the checkpoint AFTER the weight was swapped
    assert q_c.bias is not None and torch.equal(q_c.bias, q_s.bias)
    # k_proj (512 wide) stayed a raw streamed Linear: both paths exercised
    assert isinstance(comp.model.model.layers[0].self_attn.k_proj, torch.nn.Linear)


def test_model_meta_provenance(comp, stock, toy):
    m = comp.model_meta()
    assert m["arm"] == "compressed" and m["hfRepo"] == toy[0]
    # the lexicon's two figures by the lexicon's names, off meta.json's
    # weightedBpw and meanBpw (docs/pack-format.md); never a `meanBpw` key here
    assert isinstance(m["bitsPerWeight"], float) and isinstance(m["meanTensorBitsPerWeight"], float)
    assert "meanBpw" not in m
    s = stock.model_meta()
    assert s["arm"] == "stock"
    assert s["hfRepo"] is None and s["revision"] is None
    assert s["bitsPerWeight"] is None and s["meanTensorBitsPerWeight"] is None


def test_compressed_eos_set_equals_stock_from_the_real_generation_config_file(toy, tmp_path):
    """the compressed arm's model.generation_config is manufactured from
    config.json alone (module docstring) — config.json's own eos_token_id=2
    already matches this toy's tokenizer, so leaving the file untouched
    would pass by accident whichever source the arm reads. Add an id (99) that lives ONLY in
    generation_config.json, not in config.json or the tokenizer, to force
    the two sources apart: an arm reading config.json alone cannot see it, so its
    set would stay {2} while stock's real from_pretrained-loaded
    model.generation_config sees {2, 99}."""
    from drinkme.codec.pack import pack_model

    model_dir, _ = toy
    pack_dir = str(tmp_path / "pack-eos99")
    gen_path = os.path.join(model_dir, "generation_config.json")
    original = open(gen_path).read()
    try:
        obj = json.loads(original)
        obj["eos_token_id"] = [2, 99]
        with open(gen_path, "w") as f:
            json.dump(obj, f)
        # a pack carries the generation_config.json it was cut with
        # (its embedded checkpoint), so the edit is packed, not read live
        pack_model(model_dir, None, pack_dir, progress=lambda *_: None)
        stock_eng = _no_reuse(load_stock, model_dir, device="cpu")
        comp_eng = _no_reuse(load_compressed, model_dir, None, pack_dir, device="cpu")
        assert stock_eng.eos_ids == frozenset({2, 99})
        assert comp_eng.eos_ids == stock_eng.eos_ids
    finally:
        with open(gen_path, "w") as f:
            f.write(original)


def test_compressed_eos_set_ignores_config_jsons_own_manufactured_value(toy, tmp_path):
    """engines._eos_ids must not union model.generation_config.eos_token_id
    as a THIRD source on top of gd.eos_ids: for the compressed arm that
    object is manufactured from config.json alone. Give config.json an
    eos_token_id (7) that lives in NEITHER the file nor the tokenizer and
    set the file to [2, 99] (2 is also the tokenizer's own </s>): a third
    source would give the compressed arm a 7 the stock arm never sees; the
    two must land on exactly {2, 99}, not {2, 7, 99}.

    Editing config.json changes the checkpoint's identity (
    a pack is bound to the config it was cut from), so the edited toy gets
    its own pack — the session pack would be refused, correctly, as a
    different model."""
    from drinkme.codec.pack import pack_model

    model_dir, _ = toy
    pack_dir = str(tmp_path / "pack-eos7")
    cfg_path = os.path.join(model_dir, "config.json")
    gen_path = os.path.join(model_dir, "generation_config.json")
    orig_cfg, orig_gen = open(cfg_path).read(), open(gen_path).read()
    try:
        cfg_obj = json.loads(orig_cfg)
        cfg_obj["eos_token_id"] = 7
        with open(cfg_path, "w") as f:
            json.dump(cfg_obj, f)
        gen_obj = json.loads(orig_gen)
        gen_obj["eos_token_id"] = [2, 99]
        with open(gen_path, "w") as f:
            json.dump(gen_obj, f)
        pack_model(model_dir, None, pack_dir, progress=lambda *_: None)
        stock_eng = _no_reuse(load_stock, model_dir, device="cpu")
        comp_eng = _no_reuse(load_compressed, model_dir, None, pack_dir, device="cpu")
        assert stock_eng.eos_ids == frozenset({2, 99})
        assert comp_eng.eos_ids == frozenset({2, 99})
    finally:
        with open(cfg_path, "w") as f:
            f.write(orig_cfg)
        with open(gen_path, "w") as f:
            f.write(orig_gen)


def test_compressed_eos_set_equals_stock_when_the_file_lacks_the_key(toy, tmp_path):
    """generation_config.json PRESENT but without an
    eos_token_id (e.g. {"temperature": 0.7}). transformers replaces the stock
    arm's generation config with the file's contents (eos -> None) while the
    compressed skeleton keeps config.json's, so with config.json eos 7 the
    arms landed on {2} vs {2, 7}. gen_config.load now falls back to
    config.json's own eos_token_id for that shape — the same set the no-file
    path yields — so both arms stop on {2, 7}.

    Editing config.json changes the checkpoint's identity,
    so the edited toy gets its own pack, as the sibling test above does."""
    from drinkme.codec.pack import pack_model

    model_dir, _ = toy
    pack_dir = str(tmp_path / "pack-eos7-nokey")
    cfg_path = os.path.join(model_dir, "config.json")
    gen_path = os.path.join(model_dir, "generation_config.json")
    orig_cfg, orig_gen = open(cfg_path).read(), open(gen_path).read()
    try:
        cfg_obj = json.loads(orig_cfg)
        cfg_obj["eos_token_id"] = 7
        with open(cfg_path, "w") as f:
            json.dump(cfg_obj, f)
        with open(gen_path, "w") as f:
            json.dump({"temperature": 0.7}, f)
        pack_model(model_dir, None, pack_dir, progress=lambda *_: None)
        stock_eng = _no_reuse(load_stock, model_dir, device="cpu")
        comp_eng = _no_reuse(load_compressed, model_dir, None, pack_dir, device="cpu")
        assert stock_eng.eos_ids == frozenset({2, 7})
        assert comp_eng.eos_ids == frozenset({2, 7})
    finally:
        with open(cfg_path, "w") as f:
            f.write(orig_cfg)
        with open(gen_path, "w") as f:
            f.write(orig_gen)


def test_compressed_eos_set_equals_stock_with_ignore_env_set(toy, monkeypatch, tmp_path):
    """DRINKME_IGNORE_GENERATION_CONFIG=1 is a SAMPLING knob
    (module docstring) — it must not blind gen_config.load to eos_token_id,
    or the compressed arm's stop set silently diverges from stock's again
    the moment an operator sets it for its documented purpose."""
    from drinkme.codec.pack import pack_model

    model_dir, _ = toy
    pack_dir = str(tmp_path / "pack-eos99-ignore")
    gen_path = os.path.join(model_dir, "generation_config.json")
    original = open(gen_path).read()
    try:
        obj = json.loads(original)
        obj["eos_token_id"] = [2, 99]
        with open(gen_path, "w") as f:
            json.dump(obj, f)
        pack_model(model_dir, None, pack_dir, progress=lambda *_: None)
        monkeypatch.setenv("DRINKME_IGNORE_GENERATION_CONFIG", "1")
        stock_eng = _no_reuse(load_stock, model_dir, device="cpu")
        comp_eng = _no_reuse(load_compressed, model_dir, None, pack_dir, device="cpu")
        assert stock_eng.eos_ids == frozenset({2, 99})
        assert comp_eng.eos_ids == stock_eng.eos_ids
    finally:
        with open(gen_path, "w") as f:
            f.write(original)


def test_compressed_eos_set_equals_stock_for_every_pack_model_row_on_cpu(stock, comp):
    """The general test: for every pack/model row
    the CPU suite already builds BOTH a real stock and a real compressed
    engine for, from real on-disk files, compressed_eos_set == stock_eos_set
    must hold on the row's OWN (unmodified) generation_config.json/
    config.json, not just on the fault-injected shapes above. This toy is
    the only such row today — see test_serving_gen_config.py's
    MENU_ROWS-parametrized twin for the real cached menu models (no weights
    needed there either), and that test's module-level comment for the rows
    that cannot be exercised this way on CPU at all."""
    assert comp.eos_ids == stock.eos_ids


# ------------------------------------------------- rope scaling (YaRN) --


def test_rope_scaling_off_path_leaves_checkpoint_config_unchanged(comp, stock, toy):
    """Off-path parity: both toy fixtures above are built with rope_scaling
    absent (the default), so _apply_rope_scaling never ran and the loaded
    config's rope_scaling must be exactly what the checkpoint itself
    carries — byte-identical to a fresh AutoConfig read of the same dir."""
    from transformers import AutoConfig

    ref = AutoConfig.from_pretrained(toy[0]).get_text_config().rope_scaling
    assert comp.model.config.get_text_config().rope_scaling == ref
    assert stock.model.config.get_text_config().rope_scaling == ref


@pytest.fixture(scope="module")
def yarn_toy_dir(tmp_path_factory):
    """A second, separate toy with a small native window (64) — the shared
    `toy` fixture above is 256, too wide to exercise YaRN's window-widening
    without a slow, long generation. Exercised through load_stock (lighter
    than load_compressed: no pack_model/codec eligibility to satisfy) since
    load_stock's `config=cfg` threading into from_pretrained is the code
    path rope scaling actually added; load_compressed's skeleton(cfg) already
    consumed cfg directly before this feature existed."""
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

    d = tmp_path_factory.mktemp("yarn_toy")
    model_dir = str(d / "model")

    words = ["hello", "world", "the", "quick", "brown", "fox", "jumps", "over",
             "lazy", "dog", "user", "assistant", "system", ":"]
    vocab = {"<unk>": 0, "<pad>": 1, "</s>": 2}
    for w in words:
        vocab[w] = len(vocab)
    while len(vocab) < 32:
        vocab[f"tok{len(vocab)}"] = len(vocab)
    backend = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
    backend.pre_tokenizer = Whitespace()
    tok = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="<unk>",
                                  pad_token="<pad>", eos_token="</s>")
    tok.chat_template = ("{% for m in messages %}{{ m['role'] }} : {{ m['content'] }} "
                         "{% endfor %}{% if add_generation_prompt %}assistant :{% endif %}")
    tok.save_pretrained(model_dir)

    cfg = LlamaConfig(vocab_size=32, hidden_size=32, intermediate_size=32,
                      num_hidden_layers=1, num_attention_heads=2,
                      num_key_value_heads=2, max_position_embeddings=64,
                      tie_word_embeddings=False, eos_token_id=2, pad_token_id=1)
    torch.manual_seed(0)
    model = LlamaForCausalLM(cfg).to(torch.bfloat16).eval()
    model.save_pretrained(model_dir, safe_serialization=True)
    return model_dir


def test_rope_scaling_off_leaves_toy_native_window_untouched(yarn_toy_dir):
    """Same toy, flag absent — the control this acceptance bar is measured
    against: the window stays the checkpoint's native 64."""
    eng = _no_reuse(load_stock, yarn_toy_dir, device="cpu")
    tcfg = eng.model.config.get_text_config()
    assert tcfg.max_position_embeddings == 64
    assert tcfg.rope_scaling["rope_type"] == "default"
    assert eng.ctx == 64


def test_rope_scaling_on_path_extends_past_the_native_window(yarn_toy_dir):
    """The acceptance bar (rope scaling): a toy whose native window is 64 tokens, fed
    --rope-scaling yarn:4, completes a request whose prompt+output exceeds
    64 tokens without error or early truncation — the widened window, not
    the checkpoint's original one, is what StaticCache actually allocates,
    and the rotary module built from the mutated cfg is genuinely yarn."""
    eng = _no_reuse(load_stock, yarn_toy_dir, device="cpu",
                    rope_scaling={"rope_type": "yarn", "factor": 4})
    tcfg = eng.model.config.get_text_config()
    assert tcfg.rope_scaling["rope_type"] == "yarn"
    assert tcfg.max_position_embeddings == 256  # 4 x native 64
    assert eng.ctx == 256  # _apply_rope_scaling ran before _ctx: unclamped
    assert eng.model.model.rotary_emb.rope_type == "yarn"

    long_prompt = [{"role": "user", "content": " ".join(["hello world"] * 20)}]
    res = complete(eng, GenerationRequest(long_prompt, greedy(max_tokens=40)))
    assert res.prompt_tokens + res.completion_tokens > 64  # past the ORIGINAL window
    # ran to completion without an error and without the ctx-exhaustion
    # truncation test_prompt_filling_ctx_is_length_zero pins below
    # (finish_reason "stop" is a legitimate natural EOS on this untrained
    # random toy, not a truncation — ctx=256 has 172 tokens of headroom left)
    assert res.finish_reason in ("length", "stop")
    assert res.completion_tokens > 0


# ------------------------------------------------ tokenize/detokenize --

# The real invariant behind the tokenizer routes: count_tokens, tokenize(messages=...) and
# generate()'s own prompt are the SAME apply_chat_template call (build_prompt,
# same template_kwargs) — not three independent renders that happen to agree
# on the toy today and could silently drift apart tomorrow.


def test_tokenize_messages_matches_generate_prompt_tokens(comp, ref):
    assert len(comp.tokenize(messages=MSGS)) == ref[0].prompt_tokens


def test_count_tokens_is_tokenize_length(comp):
    assert comp.count_tokens(GenerationRequest(MSGS, greedy())) == len(comp.tokenize(messages=MSGS))


def test_tokenize_prompt_bypasses_the_template(comp):
    # a raw string skips apply_chat_template entirely — no role/colon framing
    # the toy's chat_template adds, so it tokenizes shorter than the same
    # words wrapped in a message
    ids = comp.tokenize(prompt="hello world")
    assert comp.detokenize(ids).split() == ["hello", "world"]
    assert len(ids) < len(comp.tokenize(messages=MSGS))


def test_detokenize_round_trips_tokenize(comp):
    ids = comp.tokenize(prompt="hello world the quick brown fox")
    assert comp.tokenize(prompt=comp.detokenize(ids)) == ids


def test_tokenizer_info_shape(comp):
    info = comp.tokenizer_info()
    assert info["eos_token_id"] == comp.tok.eos_token_id
    assert info["bos_token_id"] == comp.tok.bos_token_id
    assert info["chat_template"] is True


# ---------------------------------------------------- the acceptance bar --


def test_greedy_token_identical_to_stock(comp, stock, ref):
    """ds4's bar, toy scale: same text AND same
    token count under greedy. WordLevel decode is injective on these vocab
    words, so equal text + equal count == equal tokens."""
    res_s, _ = run(stock, greedy(max_tokens=24))
    res_c = ref[0]
    assert res_c.text == res_s.text
    assert res_c.completion_tokens == res_s.completion_tokens
    assert res_c.prompt_tokens == res_s.prompt_tokens


# ------------------------------------------------------- generate contract --


def test_max_tokens_finishes_length(comp, ref):
    res, _ = run(comp, greedy(max_tokens=6))
    assert res.finish_reason == "length" and res.completion_tokens == 6
    assert ref[0].text.startswith(res.text)


def test_generate_accepts_tools_and_toy_output_is_unchanged(comp, ref):
    # locks the Engine signature on the real engine: tools flow to
    # apply_chat_template (the toy template ignores them -> same prompt,
    # same greedy tokens) and a call-free generation reports no tool_calls
    tools = [{"type": "function",
              "function": {"name": "get_weather",
                           "parameters": {"type": "object", "properties": {}}}}]
    res = complete(comp, GenerationRequest(MSGS, greedy(max_tokens=6), tools=tools))
    assert ref[0].text.startswith(res.text) and res.text
    assert res.tool_calls is None and res.finish_reason == "length"


def test_stop_string_split_across_deltas(comp, ref):
    T = ref[0].text
    ws = T.split()
    assert len(ws) >= 4
    # last char of word 2 + space + start of word 3: the stop string can only
    # ever arrive split across two deltas
    stop = ws[1][-1] + " " + ws[2][: max(1, len(ws[2]) // 2)]
    idx = T.find(stop)
    assert idx > 0
    res, _ = run(comp, greedy(max_tokens=24, stop=[stop]))
    assert res.finish_reason == "stop"
    assert res.text == T[:idx] and stop not in res.text


def test_abort_halts_promptly(comp, ref):
    res, deltas = run(comp, greedy(max_tokens=24), until=lambda ds: len(ds) < 2)
    assert res.finish_reason == "abort"
    assert res.text == deltas[0]  # the refused delta was never delivered
    assert res.completion_tokens < ref[0].completion_tokens  # stopped early


def test_eos_finishes_stop(comp, ref):
    deltas = [d for d in ref[1] if d]
    novel = next(i for i, d in enumerate(deltas)
                 if i > 0 and d.strip() not in "".join(deltas[:i]))
    eos_id = comp.tok.convert_tokens_to_ids(deltas[novel].strip())
    old = comp.model.generation_config.eos_token_id
    try:
        comp.model.generation_config.eos_token_id = eos_id
        eng = HFEngine(comp.model, comp.tok, model_id=comp.model_id,
                       arm=comp.arm, meta=comp.meta, ctx=comp.ctx)
        res, _ = run(eng, greedy(max_tokens=24))
    finally:
        comp.model.generation_config.eos_token_id = old
    assert res.finish_reason == "stop"
    assert res.text == "".join(deltas[:novel])


def test_seeded_sampling_reproducible(comp):
    a, _ = run(comp, greedy(max_tokens=10, temperature=0.8, seed=11))
    b, _ = run(comp, greedy(max_tokens=10, temperature=0.8, seed=11))
    assert a.text == b.text and a.completion_tokens == b.completion_tokens
    c, _ = run(comp, greedy(max_tokens=10, temperature=0.8, seed=12))
    assert c.finish_reason in ("stop", "length")  # different seed MAY differ


def test_prompt_filling_ctx_is_length_zero(comp):
    small = HFEngine(comp.model, comp.tok, model_id=comp.model_id,
                     arm=comp.arm, meta=comp.meta, ctx=4)
    res = complete(small, GenerationRequest(MSGS, greedy(max_tokens=8)))
    assert res.finish_reason == "length" and res.completion_tokens == 0


# ---------------------------------------------------------- control tokens --

# serving/control.py on the REAL loop: the toy causal LM under a vocabulary
# whose gemma markers are special ids (test_serving_control.py's fake), with
# the sampler scripted so the stream is the receipt's. What is under test is
# the loop itself — the decode that keeps the markers, the scanner seeing
# them, the close-marker stop and its one-token lookahead, and the prefix
# cache's bookkeeping across the stop.


def gemma_engine(comp):
    from test_serving_control import fake_gemma_tokenizer

    tok = fake_gemma_tokenizer()
    eng = HFEngine(comp.model, tok, model_id=comp.model_id, arm=comp.arm,
                   meta=comp.meta, ctx=comp.ctx)
    assert eng.capability.tool_format == "gemma"  # the probe read the row off the template
    assert eng.control.stop_after == {tok.convert_tokens_to_ids("<tool_call|>")}
    assert eng.control.reopen == {tok.convert_tokens_to_ids("<|tool_call>")}
    assert tok.convert_tokens_to_ids("<eos>") in eng.eos_ids
    return eng, tok


def scripted(monkeypatch, ids):
    """The serial loop's sampler, replaced by a script; the logits are
    ignored. Running past the script is a test bug, not a stop."""
    from drinkme.serving import engines

    it = iter(ids)
    monkeypatch.setattr(engines, "sample_next", lambda *a, **k: next(it))


def test_control_close_marker_ends_the_turn_on_the_real_loop(comp, monkeypatch):
    from test_serving_control import CALL_THEN_PROSE, HILO, WEATHER_TOOLS, ids_of

    eng, tok = gemma_engine(comp)
    ids = ids_of(tok, CALL_THEN_PROSE)
    scripted(monkeypatch, ids)
    deltas = []
    res = complete(eng, GenerationRequest(MSGS, greedy(max_tokens=64), tools=WEATHER_TOOLS),
                   lambda d: deltas.append(d) or True)
    assert res.tool_calls == HILO
    assert res.finish_reason == "tool_calls"
    assert res.text == "" and "".join(deltas) == ""  # no marker, no prose after the call
    close = CALL_THEN_PROSE.index("<tool_call|>") + 1
    assert res.completion_tokens == close + 1  # the close, plus the refused lookahead
    # the close marker's KV was written (its forward ran before the lookahead
    # was sampled); the refused token's was not — the slot says exactly that
    assert len(eng._slots[0].ids) == res.prompt_tokens + close


def test_control_reopen_continues_the_turn_for_a_second_call(comp, monkeypatch):
    from test_serving_control import WEATHER_TOOLS, ids_of

    eng, tok = gemma_engine(comp)
    one = ["<|tool_call>", "call:", "get_weather", "{", "city", ":", '<|"|>', "Hilo", '<|"|>',
           "}", "<tool_call|>"]
    two = ["<|tool_call>", "call", ":x", "{}", "<tool_call|>"]
    scripted(monkeypatch, ids_of(tok, one + two + ["The", " weather", "."]))
    res = complete(eng, GenerationRequest(MSGS, greedy(max_tokens=64), tools=WEATHER_TOOLS))
    assert [c["name"] for c in (res.tool_calls or [])] == ["get_weather", "x"]
    assert res.finish_reason == "tool_calls" and res.text == ""
    assert res.completion_tokens == len(one) + len(two) + 1


def test_control_eos_right_after_the_close_is_a_plain_stop(comp, monkeypatch):
    from test_serving_control import HILO, WEATHER_TOOLS, ids_of

    eng, tok = gemma_engine(comp)
    one = ["<|tool_call>", "call:", "get_weather", "{", "city", ":", '<|"|>', "Hilo", '<|"|>',
           "}", "<tool_call|>", "<eos>"]
    scripted(monkeypatch, ids_of(tok, one))
    res = complete(eng, GenerationRequest(MSGS, greedy(max_tokens=64), tools=WEATHER_TOOLS))
    assert res.tool_calls == HILO and res.finish_reason == "tool_calls"
    assert res.completion_tokens == len(one)


def test_control_max_tokens_on_the_close_is_still_length(comp, monkeypatch):
    # the close is the last token allowed: no lookahead is sampled, the
    # call is parsed, and finish reports what actually ended the turn
    from test_serving_control import HILO, WEATHER_TOOLS, ids_of

    eng, tok = gemma_engine(comp)
    one = ["<|tool_call>", "call:", "get_weather", "{", "city", ":", '<|"|>', "Hilo", '<|"|>',
           "}", "<tool_call|>"]
    scripted(monkeypatch, ids_of(tok, one))
    res = complete(eng, GenerationRequest(MSGS, greedy(max_tokens=len(one)), tools=WEATHER_TOOLS))
    assert res.tool_calls == HILO and res.finish_reason == "length"


def test_control_thought_channel_reaches_the_wire_with_its_markers(comp, monkeypatch):
    """The engine keeps `<|channel>`/`<channel|>` as text: http.py's splitter
    (not the engine) files them as reasoning, so what leaves generate() is
    the marked-up stream — which a decode that skips special tokens would
    hand over as the bare word `thought`."""
    from test_serving_control import THOUGHT_THEN_CALL, HILO, WEATHER_TOOLS, ids_of

    eng, tok = gemma_engine(comp)
    scripted(monkeypatch, ids_of(tok, THOUGHT_THEN_CALL + ["<eos>"]))
    res = complete(eng, GenerationRequest(MSGS, greedy(max_tokens=64), tools=WEATHER_TOOLS))
    assert res.tool_calls == HILO
    assert res.text == "<|channel>thought\nGiving the weather.\n<channel|>"


def test_control_plain_text_is_the_plain_decode_byte_for_byte(comp, monkeypatch):
    """A stream with no control token in it (a `<turn|>` and an `<|turn>`,
    which are NOT the row's, included): what generate() emits is exactly
    skip_special_tokens=True of the ids."""
    from test_serving_control import ids_of

    eng, tok = gemma_engine(comp)
    pieces = ["The", " current", " weather", "<|turn>", " is", " 78", "°F", "<turn|>", ".", "<eos>"]
    ids = ids_of(tok, pieces)
    scripted(monkeypatch, ids)
    res = complete(eng, GenerationRequest(MSGS, greedy(max_tokens=64)))
    assert res.text == tok.decode(ids, skip_special_tokens=True) == "The current weather is 78°F."
    assert res.finish_reason == "stop" and res.tool_calls is None


# ------------------------------------------------------- the prefix cache --


def fresh(comp, **kw):
    """A cold HFEngine over the session model: its OWN persistent cache, so
    each test controls warm/cold state without touching the shared fixtures."""
    return HFEngine(comp.model, comp.tok, model_id=comp.model_id,
                    arm=comp.arm, meta=comp.meta, ctx=kw.pop("ctx", comp.ctx))


def turn2(r1_text):
    return MSGS + [{"role": "assistant", "content": r1_text},
                   {"role": "user", "content": "the lazy dog jumps"}]


def test_prefix_cache_multiturn_reuses_generated_tokens(comp):
    """The prefix cache's core claim (docs/serve-prefix-slots.md), bookkeeping form: a warm turn-2 (history
    re-sent, agent-style) reuses a prefix that reaches PAST turn 1's prompt
    into the tokens it GENERATED — the conversation is never re-prefilled.
    (Text equality warm-vs-cold on a real model belongs to the GPU verify;
    on the toy, near-uniform logits flip under kernel-shape numerics.)"""
    warm = fresh(comp)
    r1 = complete(warm, GenerationRequest(MSGS, greedy(max_tokens=8)))
    assert r1.cached_tokens == 0  # first request: nothing to reuse
    warm_res = complete(warm, GenerationRequest(turn2(r1.text), greedy(max_tokens=8)))
    # turn 2 re-renders turn 1 verbatim, so the LCP covers at least its whole
    # prompt — the win agents feel is that it covers r1's OUTPUT too
    assert warm_res.cached_tokens >= r1.prompt_tokens
    assert warm_res.finish_reason == "length" and warm_res.completion_tokens == 8


def test_prefix_cache_exact_repeat_resets(comp, monkeypatch):
    """With context checkpoints off (--ctx-checkpoints 0, the extends-only
    rule) an identical request does NOT extend the written cache (the cache
    holds prompt + generation; the repeat is a strict prefix of it), so the
    slot resets and re-prefills."""
    monkeypatch.setenv("DRINKME_CTX_CHECKPOINTS", "0")
    eng = fresh(comp)
    a = complete(eng, GenerationRequest(MSGS, greedy(max_tokens=8)))
    b = complete(eng, GenerationRequest(MSGS, greedy(max_tokens=8)))
    assert b.cached_tokens == 0
    assert a.prompt_tokens == b.prompt_tokens
    assert b.text == a.text  # fresh-cache runs are deterministic


def test_prefix_cache_exact_repeat_rewinds_to_its_last_token(comp):
    """With context checkpoints on (the default), full attention rewinds by
    length (kvcache.LiveStaticLayer.rewind), so an identical request reuses
    all of its prompt but the last token, which re-runs for fresh logits
    (serving/ctx_checkpoints.py)."""
    eng = fresh(comp)
    a = complete(eng, GenerationRequest(MSGS, greedy(max_tokens=8)))
    b = complete(eng, GenerationRequest(MSGS, greedy(max_tokens=8)))
    assert b.cached_tokens == a.prompt_tokens - 1
    assert b.text == a.text


def test_prefix_cache_divergent_history_resets_then_serves_new(comp, monkeypatch):
    """With context checkpoints off, a different conversation resets;
    EXTENDING that new conversation then reuses its cache."""
    monkeypatch.setenv("DRINKME_CTX_CHECKPOINTS", "0")
    other = [{"role": "user", "content": "alpha beta gamma delta epsilon"}]
    eng = fresh(comp)
    complete(eng, GenerationRequest(MSGS, greedy(max_tokens=8)))
    first = complete(eng, GenerationRequest(other, greedy(max_tokens=8)))
    assert first.cached_tokens == 0  # divergence -> reset, never a partial rewind
    cont = other + [{"role": "assistant", "content": first.text},
                    {"role": "user", "content": "over the lazy dog"}]
    res = complete(eng, GenerationRequest(cont, greedy(max_tokens=8)))
    assert res.cached_tokens >= first.prompt_tokens  # the new conversation's cache


def test_prefix_cache_divergent_history_reuses_the_shared_prefix(comp):
    """With context checkpoints on, a different conversation reuses what the
    two renders share (the template's opening, when it is more than a tenth
    of the prompt: llama.cpp's slot similarity) and prefills the rest;
    extending it then reuses its cache."""
    other = [{"role": "user", "content": "alpha beta gamma delta epsilon"}]
    eng = fresh(comp)
    complete(eng, GenerationRequest(MSGS, greedy(max_tokens=8)))
    held = list(eng._slots[0].ids)
    ids = eng.tokenize(messages=other)
    shared = 0
    while ids[shared] == held[shared]:
        shared += 1
    first = complete(eng, GenerationRequest(other, greedy(max_tokens=8)))
    assert first.cached_tokens == (shared if shared > 0.1 * len(ids) else 0)
    cont = other + [{"role": "assistant", "content": first.text},
                    {"role": "user", "content": "over the lazy dog"}]
    res = complete(eng, GenerationRequest(cont, greedy(max_tokens=8)))
    assert res.cached_tokens >= first.prompt_tokens


def test_prefix_cache_abort_leaves_truthful_state(comp, monkeypatch):
    """An aborted generation leaves the slot's ids claiming exactly what was
    written (the sampled-then-aborted id was never forwarded). With context
    checkpoints off the retry is a strict prefix of nothing-extends -> reset,
    full prefill; with them on (the default) it reuses the written prompt
    but its last token. Either way a correct completion."""
    for knob, cached in (("0", lambda n: 0), (None, lambda n: n - 1)):
        if knob is None:
            monkeypatch.delenv("DRINKME_CTX_CHECKPOINTS", raising=False)
        else:
            monkeypatch.setenv("DRINKME_CTX_CHECKPOINTS", knob)
        eng = fresh(comp)
        aborted = complete(eng, GenerationRequest(MSGS, greedy(max_tokens=8)),
                           lambda _d: False)
        assert aborted.finish_reason == "abort"
        assert len(eng._slots[0].ids) == aborted.prompt_tokens
        res = complete(eng, GenerationRequest(MSGS, greedy(max_tokens=8)))
        assert res.cached_tokens == cached(aborted.prompt_tokens)
        assert res.finish_reason == "length" and res.completion_tokens == 8


def test_prefix_cache_geometric_regrow(comp, monkeypatch):
    """Outgrowing the allocation drops the cache (one cold re-prefill, no
    StaticCache-internals copy) and EXTENDING reuse resumes at the new size."""
    monkeypatch.setattr(HFEngine, "KV_FLOOR", 8)
    eng = fresh(comp)
    r1 = complete(eng, GenerationRequest(MSGS, greedy(max_tokens=4)))
    first_alloc = eng._slots[0].alloc
    assert first_alloc >= r1.prompt_tokens + 4 and first_alloc < comp.ctx
    # a longer conversation forces need past the allocation -> regrow, cold
    msgs2 = turn2(r1.text)
    r2 = complete(eng, GenerationRequest(msgs2, greedy(max_tokens=16)))
    assert eng._slots[0].alloc > first_alloc
    assert r2.cached_tokens == 0  # fresh tensor: nothing was reusable
    # and the NEW allocation serves EXTENDING reuse from here on (a short
    # extension, sized to stay inside the regrown allocation)
    msgs3 = msgs2 + [{"role": "assistant", "content": r2.text},
                     {"role": "user", "content": "over"}]
    r3 = complete(eng, GenerationRequest(msgs3, greedy(max_tokens=8)))
    assert r3.cached_tokens >= r2.prompt_tokens


def test_prefix_cache_off_at_zero_slots(comp, monkeypatch):
    monkeypatch.setenv("DRINKME_PREFIX_SLOTS", "0")
    eng = fresh(comp)
    a = complete(eng, GenerationRequest(MSGS, greedy(max_tokens=8)))
    b = complete(eng, GenerationRequest(MSGS, greedy(max_tokens=8)))
    assert a.cached_tokens == 0 and b.cached_tokens == 0
    assert eng._slots[0].cache is None  # never allocated a persistent cache


# ------------------------------------- the Protocol seam, over real sockets --


def test_http_through_real_engine(comp, ref):
    from drinkme.serving.http import start_server

    srv = start_server(comp, "127.0.0.1", 0)
    try:
        port = srv.server_address[1]

        def post(body):
            c = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
            c.request("POST", "/v1/chat/completions", json.dumps(body),
                      {"Content-Type": "application/json"})
            r = c.getresponse()
            data = r.read()
            c.close()
            return r, data

        req = {"messages": MSGS, "temperature": 0, "max_tokens": 6}
        r, body = post(req)
        obj = json.loads(body)
        assert r.status == 200
        content = obj["choices"][0]["message"]["content"]
        assert obj["choices"][0]["finish_reason"] == "length"
        assert content  # non-vacuous: startswith("") would pass on anything
        # the toy's template opens no <think>, so nothing is filed as
        # reasoning (serving/think.py decides off the PROMPT, not the family)
        assert "reasoning_content" not in obj["choices"][0]["message"]
        assert ref[0].text.startswith(content)
        assert obj["usage"]["completion_tokens"] == 6
        assert obj["usage"]["prompt_tokens_details"]["cached_tokens"] >= 0

        r, body = post({**req, "stream": True})
        assert r.status == 200
        events = [b[len("data: "):] for b in body.decode().split("\n\n")
                  if b.startswith("data: ")]
        assert events[-1] == "[DONE]"
        chunks = [json.loads(e) for e in events[:-1]]
        text = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks)
        assert text == content  # stream and non-stream agree, greedy
        assert chunks[-1]["choices"][0]["finish_reason"] == "length"
    finally:
        srv.shutdown()


def test_warmup_runs_and_reports(comp, capsys):
    """serve._warmup: one tiny generation before the port binds. Must not
    raise, must print the warm line, and must leave the engine usable."""
    from drinkme.serve import _warmup

    _warmup(comp)
    out = capsys.readouterr().out
    assert "warm in" in out
    res, _ = run(comp, greedy(max_tokens=4))
    assert res.finish_reason in ("stop", "length")


def test_warmup_skippable_by_env(comp, capsys, monkeypatch):
    monkeypatch.setenv("DRINKME_NO_WARMUP", "1")
    from drinkme.serve import _warmup

    _warmup(comp)
    assert "warm in" not in capsys.readouterr().out


def test_owning_config_is_the_nearest_ancestors_not_the_text_config():
    """gemma-4-31B-it (multimodal) has TWO rotaries built from TWO configs: the
    vision tower's from vision_config, the text model's from text_config. The
    rebuild after a meta load must hand each the config its owner was built
    with; the first draft gave every rotary the text config and the vision
    one raised KeyError('rope_type') on the real checkpoint."""
    from types import SimpleNamespace
    from drinkme.serving.engines import _owning_config

    class Rot(torch.nn.Module):
        pass

    class Owner(torch.nn.Module):
        def __init__(self, config):
            super().__init__()
            self.config = config
            self.rotary_emb = Rot()

    class Root(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(name="top", text=SimpleNamespace(name="text"),
                                          get_text_config=lambda: SimpleNamespace(name="text"))
            self.vision_tower = Owner(SimpleNamespace(name="vision"))
            self.language_model = Owner(SimpleNamespace(name="text"))
            self.bare = Rot()  # a rotary with no config-carrying owner but the root

    m = Root()
    assert _owning_config(m, "vision_tower.rotary_emb", m.config).name == "vision"
    assert _owning_config(m, "language_model.rotary_emb", m.config).name == "text"
    # the root itself carries a config, so a bare rotary gets the root's
    assert _owning_config(m, "bare", m.config).name == "top"
    # no config anywhere on the path → the text config, as before
    class Plain(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.rotary_emb = Rot()
    assert _owning_config(Plain(), "rotary_emb", m.config).name == "text"


def test_rebuild_computed_buffers_handles_per_layer_type_rotary_and_embed_scale():
    """gemma-4-31B-it's two computed buffers the first real load zero-filled:
    the text rotary's PER-LAYER-TYPE inv_freq pairs (no bare `inv_freq`) and
    the scaled embedding's `embed_scale`. Both must be materialized from
    their exact recipes, not zeros."""
    from types import SimpleNamespace
    from drinkme.serving.engines import _rebuild_computed_buffers

    class Rot(torch.nn.Module):
        def __init__(self, config, device=None):
            super().__init__()
            self.register_buffer("sliding_attention_inv_freq",
                                 torch.full((4,), float(config.theta), device=device), persistent=False)
            self.register_buffer("full_attention_inv_freq",
                                 torch.full((4,), float(config.theta) * 2, device=device), persistent=False)

    class ScaledEmb(torch.nn.Embedding):
        def __init__(self, n, d, scale):
            super().__init__(n, d)
            self.scalar_embed_scale = scale
            self.register_buffer("embed_scale", torch.tensor(scale), persistent=False)

    class Text(torch.nn.Module):
        def __init__(self, config):
            super().__init__()
            self.config = config
            self.embed_tokens = ScaledEmb(8, 4, config.hidden ** 0.5)
            self.rotary_emb = Rot(config)

    with torch.device("meta"):
        text = Text(SimpleNamespace(theta=3.0, hidden=16))
    assert text.rotary_emb.sliding_attention_inv_freq.is_meta
    assert text.embed_tokens.embed_scale.is_meta

    _rebuild_computed_buffers(text, SimpleNamespace(get_text_config=lambda: text.config), "cpu")
    assert not text.rotary_emb.sliding_attention_inv_freq.is_meta
    assert torch.equal(text.rotary_emb.sliding_attention_inv_freq, torch.full((4,), 3.0))
    assert torch.equal(text.rotary_emb.full_attention_inv_freq, torch.full((4,), 6.0))
    assert float(text.embed_tokens.embed_scale) == 4.0
    assert text.embed_tokens.weight.is_meta  # the parameter is the stream-attach's job, untouched here
