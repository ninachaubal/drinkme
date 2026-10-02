"""The bench's compressed and twin arms STREAM, like the stock arm.

The fit point is "bf16 does not fit, compressed does" — a compressed arm
built as load_cpu(model_id) (the WHOLE bf16 checkpoint on the host), then
swap_linears(make_compressed) on that tree, then `.cuda()`, cannot exist on
a 122 GB box for Qwen2.5-72B-Instruct (145 GB bf16; the Strix Halo box's fit point),
so `drinkme bench` could not take it. Both arms are built the way the stock
arm is: the tree empty on the meta device, each eligible
Linear swapped from its own checkpoint tensor as that tensor is read
(arms._stream_swapped over swap.make_compressed / make_twin), the remainder
and the biases through the ONE walker (serving.engines.stream_checkpoint).

Pinned here, CPU only:

1. THE LOAD-BEARING TEST: the streamed tree is IDENTICAL to what load_cpu +
   swap_linears produce — the same class at every slot, the same packed
   planes / twin weight / bias per swapped tensor, the same raw remainder,
   and logits BITWISE equal (torch.equal) through the forward and through
   the bench's own greedy loop — for the compressed arm, the twin, and (the
   ratio point's three arms) the stock arm too, on toys that exercise a
   multi-shard checkpoint with attention biases (bias-after-swap), a tied
   head whose shape would be eligible (the tied rule), and the Glimmer
   wrapper with its vision tower (an untied eligible head, and the tower's
   own eligible Linears swapped too).
   The twin's forward is Triton-only, so its logits are taken through a
   CPU reference that reads the planes exactly as the kernel does.
2. The gate: the encoder's round trip runs per tensor INSIDE the load (before the
   tensor is installed) and refuse_unless_verified sits right after the
   load, before any timing — the same place in the flow as before.
3. The arithmetic (fit.streamed_transient_bytes) with the Strix Halo box's
   numbers: the 72B's compressed arm at 110.28 GB + one tensor fits
   133.14 GB, its twin (145.4 GB + one tensor) does not; `--dry-run` plans
   the 72B as a compressed-only fit point; the compressed arm's live guard.
4. The record: a fit point where neither stock nor twin fits carries
   stock.outcome skipped_predicted_nonfit with its budget, raw.twin_outcome
   skipped_predicted_nonfit, and the compressed metrics.
"""

from __future__ import annotations

import ast
import functools
import json
import os
import pathlib
import sys
import tempfile
import types

import numpy as np
import pytest
import torch

from drinkme import arms, bench
from drinkme.codec import swap
from drinkme.codec.radix_pack import raw_dict
from drinkme.codec.swap import (CompressedLinear, RadixCompressedLinear, RadixTwinLinear, RawLinear,
                                eligible_linears, make_compressed, make_twin, swap_linears)
from drinkme.detect import Hardware
from drinkme.fit import GB, GIB, STAGING_SHARD_BYTES, streamed_arm_fits, streamed_transient_bytes
from drinkme.publish import validate as V
from drinkme.suggest import FIT_HEADROOM, MODELS, RATIO_HEADROOM, Model, Suggestion, sizes

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
from fixtures import word_tokenizer as _tokenizer  # noqa: E402
from menu_checkpoints import pin_sizes  # noqa: E402

# Qwen2.5-72B-Instruct's largest tensor: embed_tokens 152064 x 8192 bf16
QWEN72B_EMBED_BYTES = 152064 * 8192 * 2
STRIX_HALO_CEILING = 124.0 * GIB


def _m(name: str):
    return next(x for x in MODELS if x.name == name)


# ------------------------------------------------------------------ toys --


def _save(tmp_path_factory, tag: str, model, vocab: int, max_shard_size=None) -> str:
    d = str(tmp_path_factory.mktemp(tag) / "model")
    os.makedirs(d)
    _tokenizer(vocab).save_pretrained(d)
    with torch.no_grad():
        for _, _, _, child in eligible_linears(model):
            child.weight.data.mul_(1 / 64)
    kw = {"max_shard_size": max_shard_size} if max_shard_size else {}
    model.save_pretrained(d, safe_serialization=True, **kw)
    del model
    return d


@pytest.fixture(scope="module")
def llama_biased_sharded(tmp_path_factory):
    """A 1-layer Llama at hidden 1024 with ATTENTION BIASES, saved as several
    shards: q/o_proj are eligible AND biased (the bias-after-swap path, on
    both arms), k/v_proj (512 rows) are sub-bar raw Linears with biases,
    gate/up/down (2048 x 1024 — twice q/o's size, for the largest-first
    order) eligible without bias, lm_head (64 rows) raw and untied."""
    from transformers import LlamaConfig, LlamaForCausalLM

    cfg = LlamaConfig(vocab_size=64, hidden_size=1024, intermediate_size=2048,
                      num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
                      max_position_embeddings=64, tie_word_embeddings=False,
                      attention_bias=True, eos_token_id=2, pad_token_id=1)
    torch.manual_seed(11)
    return _save(tmp_path_factory, "llama_biased", LlamaForCausalLM(cfg).to(torch.bfloat16).eval(),
                 64, max_shard_size="3MB")


@pytest.fixture(scope="module")
def llama_tied_eligible_head(tmp_path_factory):
    """vocab 1024 x hidden 1024 with tie_word_embeddings: the head's SHAPE
    clears the codec's floor, but it is tied to the embedding — the rule in
    swap.py's docstring says it stays raw — and it is ABSENT from the
    checkpoint (tie_weights must supply it)."""
    from transformers import LlamaConfig, LlamaForCausalLM

    cfg = LlamaConfig(vocab_size=1024, hidden_size=1024, intermediate_size=1024,
                      num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
                      max_position_embeddings=64, tie_word_embeddings=True,
                      eos_token_id=2, pad_token_id=1)
    torch.manual_seed(12)
    return _save(tmp_path_factory, "llama_tied", LlamaForCausalLM(cfg).to(torch.bfloat16).eval(),
                 1024)


@pytest.fixture(scope="module")
def glimmer_with_vision(tmp_path_factory):
    """The wrapper whose vision tower (arms._VISION_TOWERS) the bench's
    default tree carries too: the
    checkpoint SHIPS a vision tower whose Linears are 1024-wide — eligible
    by shape — and the streamed arms swap them alongside the text Linears
    (plus the untied 1024 x 1024 lm_head)."""
    pytest.importorskip("transformers.models.muse_glimmer",
                        reason="muse_glimmer arrived in transformers 5.15.0")
    from transformers.models.muse_glimmer import (MuseGlimmerConfig,
                                                  MuseGlimmerForConditionalGeneration)

    cfg = MuseGlimmerConfig(
        text_config=dict(vocab_size=1024, hidden_size=1024, intermediate_size=1024,
                         num_hidden_layers=1, num_attention_heads=8, num_key_value_heads=2,
                         head_dim=128, max_position_embeddings=128, sliding_window=64,
                         bos_token_id=None, eos_token_id=2, pad_token_id=1),
        vision_config=dict(hidden_size=1024, intermediate_size=1024, num_hidden_layers=1,
                           num_attention_heads=8, pos_emb_height=4, pos_emb_width=4,
                           max_position_embeddings=16),
        out_hidden_size=1024, projector_hidden_size=1024)
    torch.manual_seed(13)
    return _save(tmp_path_factory, "glimmer",
                 MuseGlimmerForConditionalGeneration(cfg).to(torch.bfloat16).eval(), 1024)


TOYS = ("llama_biased_sharded", "llama_tied_eligible_head", "glimmer_with_vision")
IDS = torch.arange(3, 15)[None]


def _old_compressed(model_dir: str):
    """The path being replaced: load_cpu -> swap_linears(make_compressed)."""
    model = arms.load_cpu(model_dir)
    stats = swap_linears(model, functools.partial(make_compressed, device="cpu"), device="cpu")
    return model, stats


def _old_twin(model_dir: str, plan: dict):
    """The twin by the old path's shape: load_cpu, then every eligible
    Linear swapped in place (swap_linears's walk) for make_twin at the
    compressed arm's plan for that slot."""
    model = arms.load_cpu(model_dir)
    for name, parent, child_name, child in eligible_linears(model):
        bias = child.bias.data if child.bias is not None else None
        mod, _ = make_twin(child.weight.data, bias, "cpu", **plan[name])
        setattr(parent, child_name, mod)
        child.weight.data = torch.empty(0)
    return model


def _streamed_twin(model_dir: str):
    """The bench's order: the compressed arm first, its stats the twin's plan."""
    _, stats = arms.load_compressed_streaming(model_dir, None, device="cpu")
    plan = arms.twin_plan(stats)
    model, twin_stats = arms.load_twin_streaming(model_dir, None, device="cpu", plan=plan)
    return model, twin_stats, plan


def _assert_same_raw_tree(streamed, old):
    """Everything that is NOT a swapped Linear: same names, same bytes."""
    assert not any(p.is_meta for p in streamed.parameters())
    assert not any(b.is_meta for b in streamed.buffers())
    assert type(streamed) is type(old)
    assert [(n, type(m).__name__) for n, m in streamed.named_modules()] == \
        [(n, type(m).__name__) for n, m in old.named_modules()]
    sp, kp = dict(streamed.named_parameters()), dict(old.named_parameters())
    assert set(sp) == set(kp), set(sp) ^ set(kp)
    assert [n for n in kp if not torch.equal(sp[n], kp[n])] == []
    sb, kb = dict(streamed.named_buffers()), dict(old.named_buffers())
    assert set(sb) == set(kb), set(sb) ^ set(kb)
    assert [n for n in kb if not torch.equal(sb[n], kb[n])] == []


def _assert_same_bias(a, b, name):
    assert (a.bias is None) == (b.bias is None), name
    if a.bias is not None:
        assert a.bias.dtype == torch.bfloat16 == b.bias.dtype, name
        assert torch.equal(a.bias, b.bias), name


def _swapped(model, cls):
    return [(n, m) for n, m in model.named_modules() if isinstance(m, cls)]


# --------------------------------------------- 1. THE LOAD-BEARING TESTS --


@pytest.mark.parametrize("toy", TOYS)
def test_streamed_compressed_arm_is_bitwise_load_cpu_plus_swap(toy, request):
    """Structure, every packed plane, every bias, the raw remainder, the
    stats (content AND order), and the logits — bitwise."""
    model_dir = request.getfixturevalue(toy)
    old, old_stats = _old_compressed(model_dir)
    new, new_stats = arms.load_compressed_streaming(model_dir, None, device="cpu")
    _assert_same_raw_tree(new, old)
    olds, news = _swapped(old, CompressedLinear), _swapped(new, CompressedLinear)
    assert [n for n, _ in olds] == [n for n, _ in news] and olds
    for (name, mo), (_, mn) in zip(olds, news):
        assert set(mo.p) == set(mn.p), name
        for k in mo.p:
            a, b = mo.p[k], mn.p[k]
            if torch.is_tensor(a):
                assert a.dtype == b.dtype and torch.equal(a, b), (name, k)
            elif isinstance(a, np.ndarray):
                assert np.array_equal(a, b), (name, k)
            else:
                assert a == b, (name, k)
        assert (mo.tensor_codec, mo.layout, mo.R, mo.C) == (mn.tensor_codec, mn.layout, mn.R, mn.C)
        _assert_same_bias(mo, mn, name)
    assert new_stats == old_stats  # make_compressed's stat, in swap_linears's order
    assert all(s["verified"] and s["codec"] == "radix" for s in new_stats)
    with torch.inference_mode():
        want = old(IDS, use_cache=False).logits
        got = new(IDS, use_cache=False).logits
    assert want.dtype == torch.bfloat16 == got.dtype
    assert torch.equal(got, want), (got.float() - want.float()).abs().max().item()
    assert torch.equal(arms.greedy(new, IDS, 6), arms.greedy(old, IDS, 6))


@pytest.mark.parametrize("toy", TOYS)
def test_streamed_twin_arm_is_bitwise_load_cpu_plus_swap(toy, request):
    model_dir = request.getfixturevalue(toy)
    new, stats, plan = _streamed_twin(model_dir)
    old = _old_twin(model_dir, plan)
    assert stats == []  # make_twin has no stat
    _assert_same_raw_tree(new, old)
    olds, news = _swapped(old, RadixTwinLinear), _swapped(new, RadixTwinLinear)
    assert [n for n, _ in olds] == [n for n, _ in news] and olds
    for (name, mo), (_, mn) in zip(olds, news):
        assert (mo.R, mo.C) == (mn.R, mn.C), name
        assert set(mo.p) == set(mn.p), name
        assert torch.equal(mo.p["weight"], mn.p["weight"]), name
        assert {k: v for k, v in mo.p.items() if k != "weight"} == \
            {k: v for k, v in mn.p.items() if k != "weight"}, name
        _assert_same_bias(mo, mn, name)
    with torch.inference_mode():
        want = old(IDS, use_cache=False).logits
        got = new(IDS, use_cache=False).logits
    assert torch.equal(got, want)
    assert torch.equal(arms.greedy(new, IDS, 6), arms.greedy(old, IDS, 6))


def _force_raw_on_square(monkeypatch):
    """Make the compressed arm fall back to raw on the square (attention)
    tensors, as radix does on a tensor it would expand: pack_weight_served
    returns raw_dict for them."""
    real = swap.pack_weight_served

    def pack(w, *a, **k):
        if w.shape[0] == w.shape[1]:
            return raw_dict(w.view(torch.int16).numpy().view(np.uint16))
        return real(w, *a, **k)

    monkeypatch.setattr(swap, "pack_weight_served", pack)


@pytest.mark.parametrize("schedule", ("table", "table=gfx1102", "table=cuda", "spike"))
def test_the_twin_serves_each_slot_the_way_the_compressed_arm_does(llama_biased_sharded,
                                                                  monkeypatch, schedule):
    """Slot by slot: a radix tensor's twin is RadixTwinLinear with the
    compressed module's widths at its own launch row (select_twin), a raw
    fallback's is RawLinear over the same bits (F.linear, as compressed);
    and on the CPU, where both routes are F.linear over the exact weight,
    the two trees' logits are bitwise equal."""
    monkeypatch.setenv("DRINKME_RADIX_SCHEDULE", schedule)
    _force_raw_on_square(monkeypatch)
    d = llama_biased_sharded
    comp, stats = arms.load_compressed_streaming(d, None, device="cpu")
    twin, _ = arms.load_twin_streaming(d, None, device="cpu", plan=arms.twin_plan(stats))
    comps = dict(_swapped(comp, CompressedLinear))
    twins = dict(_swapped(twin, CompressedLinear))
    assert set(comps) == set(twins) == {st["name"] for st in stats}
    kinds = set()
    for name, mc in comps.items():
        mt = twins[name]
        if type(mc) is RadixCompressedLinear:
            assert type(mt) is RadixTwinLinear, name
            # the twin's own row (select_twin: the box's twin gemv row over
            # the compressed row); the mc fields are the compressed tensor's
            from drinkme.codec.radix_schedule import select_twin
            assert mt.p["rx_launch"] == select_twin(mc.R, mc.C, mc.p["block_size"], mc.p["widths"]).as_dict()
            assert [mt.p["rx_launch"][k] for k in ("mc_tiles", "mc_warps")] == \
                [mc.p["rx_launch"][k] for k in ("mc_tiles", "mc_warps")], name
            assert mt.p["widths"] == mc.p["widths"] and mt.p["block_size"] == mc.p["block_size"]
            assert torch.equal(mt.p["weight"], mc._cpu_weight()), name
        else:
            assert type(mc) is RawLinear and type(mt) is RawLinear, name
            assert torch.equal(mt.p["weight"], mc.p["weight"]), name
        kinds.add(type(mc))
        assert (mt.bias is None) == (mc.bias is None) and (mt.bias is None or torch.equal(mt.bias, mc.bias))
    assert kinds == {RadixCompressedLinear, RawLinear}  # both kinds exercised
    with torch.inference_mode():
        assert torch.equal(twin(IDS, use_cache=False).logits, comp(IDS, use_cache=False).logits)


def test_the_twin_refuses_a_slot_the_plan_does_not_name(llama_biased_sharded):
    _, stats = arms.load_compressed_streaming(llama_biased_sharded, None, device="cpu")
    plan = arms.twin_plan(stats)
    plan.pop(next(iter(plan)))
    with pytest.raises(ValueError, match="plan names no tensor"):
        arms.load_twin_streaming(llama_biased_sharded, None, device="cpu", plan=plan)
    with pytest.raises(ValueError, match="needs the compressed arm's plan"):
        arms.load_twin_streaming(llama_biased_sharded, None, device="cpu")


def test_all_three_arms_stream_bitwise_on_the_ratio_point_toy(llama_biased_sharded):
    """The ratio point's three passes, each streamed, each bitwise its
    old-path twin — and the three streamed arms agree with each other on
    the CPU, where the compressed route decodes the exact weight and the
    twin holds it: stock's own F.linear, three times over."""
    d = llama_biased_sharded
    stock_old = arms.load_cpu(d)
    stock_new = arms.load_stock_streaming(d, device="cpu")
    comp_old, _ = _old_compressed(d)
    comp_new, _ = arms.load_compressed_streaming(d, None, device="cpu")
    twin_new, _, plan = _streamed_twin(d)
    twin_old = _old_twin(d, plan)
    with torch.inference_mode():
        logits = {k: m(IDS, use_cache=False).logits for k, m in {
            "stock_old": stock_old, "stock_new": stock_new, "twin_old": twin_old,
            "twin_new": twin_new, "comp_old": comp_old, "comp_new": comp_new}.items()}
    for arm in ("stock", "twin", "comp"):
        assert torch.equal(logits[f"{arm}_new"], logits[f"{arm}_old"]), arm
    assert torch.equal(logits["comp_new"], logits["stock_new"])
    assert torch.equal(logits["twin_new"], logits["stock_new"])


def test_biases_arrive_after_the_swap_on_both_arms(llama_biased_sharded):
    """q/o_proj carry biases in this toy; the swapped module holds them
    bf16, from the checkpoint, and k/v_proj (sub-bar) stay raw Linears with
    theirs."""
    comp, _ = arms.load_compressed_streaming(llama_biased_sharded, None, device="cpu")
    twin, _, _ = _streamed_twin(llama_biased_sharded)
    for model in (comp, twin):
        attn = model.model.layers[0].self_attn
        assert attn.q_proj.bias is not None and attn.o_proj.bias is not None
        assert attn.q_proj.bias.dtype == torch.bfloat16
        assert isinstance(attn.k_proj, torch.nn.Linear) and attn.k_proj.bias is not None
        assert isinstance(model.model.layers[0].mlp.gate_proj, type(attn.q_proj))
        assert model.model.layers[0].mlp.gate_proj.bias is None
        assert isinstance(model.lm_head, torch.nn.Linear)  # 64 rows: sub-bar, raw


def test_tied_head_stays_raw_and_retied_on_both_arms(llama_tied_eligible_head):
    comp, stats = arms.load_compressed_streaming(llama_tied_eligible_head, None, device="cpu")
    twin, _ = arms.load_twin_streaming(llama_tied_eligible_head, None, device="cpu",
                                       plan=arms.twin_plan(stats))
    for model in (comp, twin):
        assert isinstance(model.lm_head, torch.nn.Linear)
        assert model.lm_head.weight is model.model.embed_tokens.weight  # retied
    assert "lm_head" not in {s["name"] for s in stats}


def test_glimmer_swaps_the_tower_too_by_default(glimmer_with_vision):
    """The shipped vision tower's 1024-wide Linears are swapped alongside
    the text Linears (the bench's skeleton keeps the tower by default;
    ckpt_to_skel passes its keys through) and the untied head is too. The
    arm's population is exactly the pack's — `drinkme pack` has always
    carried the tower — with nothing skipped by path."""
    from drinkme.arms import vision_tower_paths
    from drinkme.codec.pack import iter_pack_dir, pack_model

    comp, stats = arms.load_compressed_streaming(glimmer_with_vision, None, device="cpu")
    names = {s["name"] for s in stats}
    assert any(n.startswith("model.vision") for n in names)
    assert "lm_head" in names and type(comp.lm_head) is RadixCompressedLinear
    assert comp.model.vision_tower is not None
    assert comp.model.vision_adapter is not None
    assert comp.model.vision_projection is not None
    # exactly the population `drinkme pack` writes (the writer's own walk)
    pack_dir = str(pathlib.Path(glimmer_with_vision).parent / "pack")
    pack_model(glimmer_with_vision, None, pack_dir, progress=lambda *_: None)
    vision_tower_paths(comp.config)  # confirms the arch is _VISION_TOWERS'
    assert names == {n for n, _ in iter_pack_dir(pack_dir)}


def test_streamed_arms_never_stage_the_model_on_the_host(llama_biased_sharded, monkeypatch):
    """The point: no load_cpu, no from_pretrained — the host never holds the
    bf16 model; and on the finished tree no eligible-shaped bf16 nn.Linear
    weight is left anywhere."""
    import transformers

    def boom(*a, **k):
        raise AssertionError("the streamed arms must not materialize the model on the host")

    monkeypatch.setattr(arms, "load_cpu", boom)
    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", boom)
    monkeypatch.setattr(transformers.AutoModelForImageTextToText, "from_pretrained", boom)
    comp, stats = arms.load_compressed_streaming(llama_biased_sharded, None, device="cpu")
    twin, _ = arms.load_twin_streaming(llama_biased_sharded, None, device="cpu",
                                       plan=arms.twin_plan(stats))
    for model, cls in ((comp, CompressedLinear), (twin, RadixTwinLinear)):
        assert _swapped(model, cls)
        assert [n for n, m in model.named_modules()
                if isinstance(m, torch.nn.Linear) and m.weight.dtype == torch.bfloat16
                and min(m.weight.shape) >= 1024 and m.weight.shape[1] % 4 == 0] == []


def test_eligible_linears_are_swapped_largest_first(llama_biased_sharded, monkeypatch):
    """The packer's host scratch is ~10x the tensor in hand (measured, see
    arms._stream_swapped), so the biggest eligible Linears are packed while
    the least is resident: install order is non-increasing in element
    count (the three 2048 x 1024 MLP tensors before the two 1024 x 1024
    attention ones), whatever the shard order — and the stats still come
    back in swap_linears's (eligible_linears) order."""
    seen = []
    real = arms._install_swapped

    def spy(model, name, w, device, make, stats):
        seen.append((name, w.numel()))
        return real(model, name, w, device, make, stats)

    monkeypatch.setattr(arms, "_install_swapped", spy)
    _, stats = arms.load_compressed_streaming(llama_biased_sharded, None, device="cpu")
    sizes = [n for _, n in seen]
    assert sizes == sorted(sizes, reverse=True) and sizes[0] == 2048 * 1024 and sizes[-1] == 1024 * 1024
    assert {n for n, _ in seen[:3]} == {"model.layers.0.mlp.gate_proj", "model.layers.0.mlp.up_proj",
                                         "model.layers.0.mlp.down_proj"}
    skel = arms.skeleton(__import__("transformers").AutoConfig.from_pretrained(llama_biased_sharded))
    assert [s["name"] for s in stats] == [n for n, _, _, _ in eligible_linears(skel)]
    assert [s["name"] for s in stats] != [n for n, _ in seen]  # the order was really changed


def test_streamed_loaders_refuse_an_fp8_checkpoint_with_the_one_line(tmp_path):
    """The compressed and twin streaming loaders print the same line the
    stock loader and `drinkme pack` do (codec/pack.FP8_REFUSAL), before a
    tensor is read; a quantization_config that is not fp8 is refused in
    the same shape, naming its quant_method."""
    from fixtures import fake_checkpoint

    from drinkme.codec.pack import FP8_REFUSAL, UnsupportedCheckpoint

    d = fake_checkpoint(str(tmp_path / "fp8"), quantization_config={"quant_method": "fp8"})
    for loader in (arms.load_compressed_streaming,
                   functools.partial(arms.load_twin_streaming, plan={})):
        with pytest.raises(UnsupportedCheckpoint) as ei:
            loader(d, None, device="cpu")
        assert str(ei.value) == f"{d}: {FP8_REFUSAL}"
    g = fake_checkpoint(str(tmp_path / "gptq"), quantization_config={"quant_method": "gptq", "bits": 4})
    with pytest.raises(UnsupportedCheckpoint) as ei:
        arms.load_compressed_streaming(g, None, device="cpu")
    assert str(ei.value) == (f"{g}: quantized checkpoints (quant_method='gptq') are not "
                             "supported in this release (bf16 only)")


def test_streamed_loader_refuses_a_checkpoint_missing_an_eligible_weight(llama_biased_sharded, tmp_path):
    """A shard set that lacks a weight the skeleton needs is refused by
    name (never a zero weight served with a straight face)."""
    import shutil

    from safetensors import safe_open
    from safetensors.torch import save_file

    d = str(tmp_path / "missing")
    shutil.copytree(llama_biased_sharded, d)
    index = json.load(open(os.path.join(d, "model.safetensors.index.json")))
    victim = "model.layers.0.mlp.down_proj.weight"
    shard = os.path.join(d, index["weight_map"][victim])
    with safe_open(shard, framework="pt") as sf:
        kept = {k: sf.get_tensor(k) for k in sf.keys() if k != victim}
    save_file(kept, shard, metadata={"format": "pt"})
    with pytest.raises(ValueError, match="missing 1 weights the compressed arm needs"):
        arms.load_compressed_streaming(d, None, device="cpu")


# -------------------------------------------------------- 2. the gate --


def test_verify_pack_gate_fires_inside_the_load_per_tensor(llama_biased_sharded, monkeypatch):
    """The round-trip gate runs inside pack_weight_served (radix_pack
    .pack_array_radix: the encoder decodes back and compares before it
    returns), per tensor, BEFORE the tensor is installed: a failing tensor
    raises out of the load itself (so run_arms never reaches
    refuse_unless_verified, let alone a timing), and the tensors before it
    were already verified."""
    from drinkme.codec import radix_pack as rp

    seen = []
    real = rp.pack_array_radix

    def failing(U, profile, *a, **k):
        seen.append(len(seen))
        if len(seen) == 2:
            raise ValueError("radix pack does not reconstruct the source bits: injected roundtrip mismatch")
        return real(U, profile, *a, **k)

    monkeypatch.setattr(rp, "pack_array_radix", failing)
    with pytest.raises(ValueError, match="injected roundtrip mismatch"):
        arms.load_compressed_streaming(llama_biased_sharded, None, device="cpu")
    assert len(seen) == 2  # the second tensor's gate raised; nothing after it ran


def _run_arms_src() -> str:
    """run_arms's BODY (the docstring dropped — it names the loaders too)."""
    tree = ast.parse((ROOT / "src" / "drinkme" / "arms.py").read_text())
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "run_arms")
    body = fn.body[1:] if ast.get_docstring(fn) else fn.body
    return "\n".join(ast.unparse(n) for n in body)


def test_run_arms_gate_sits_between_the_compressed_load_and_its_timing():
    """Statically: pass 3 is load_compressed_streaming -> refuse_unless_verified
    -> the first timing, in that order, with no timing in between; and
    passes 2/3 never call load_cpu or swap_linears any more (pass 1's
    from_pretrained option is the only load_cpu left)."""
    src = _run_arms_src()
    tail = src[src.index("load_compressed_streaming("):]
    gate = tail.index("refuse_unless_verified(model_id, stats)")
    first_timing = min(tail.index(f) for f in ("timed_decode(", "timed_prefill(", "timed_ttft("))
    assert gate < first_timing
    assert "swap_linears" not in src
    assert src.count("load_cpu(") == 1 and "device_map='cuda'" in src
    assert "load_twin_streaming(" in src and "load_compressed_streaming(" in src


def test_refuse_unless_verified_still_refuses_an_empty_arm():
    with pytest.raises(SystemExit, match="refusing to write the record"):
        arms.refuse_unless_verified("toy/model", [])


# ------------------------------------------------- 3. the arithmetic --


def test_streamed_charge_is_resident_plus_one_tensor_on_unified_only():
    assert streamed_transient_bytes(100, 0, "unified", tensor_bytes=7) == 107
    assert streamed_transient_bytes(100, 5, "unified", tensor_bytes=7) == 112
    assert streamed_transient_bytes(100, 0, "unified") == 100 + STAGING_SHARD_BYTES
    assert streamed_transient_bytes(100, 0, "vram", tensor_bytes=7) == 100
    assert streamed_transient_bytes(100, 0, "vram") == 100


def test_the_72b_compressed_arm_fits_strix_halo_and_its_twin_does_not():
    """105.71 GB (the compressed estimate) + the 2.49 GB embedding = 108.21
    GB, x1.1 = 119.03 GB against 133.14 GB physical: fits (even on the 5 GiB
    bound: 111.08 x 1.1 = 122.19). The twin's 145.41 GB single copy is over
    the pool before any transient."""
    sz = sizes(_m("Qwen2.5-72B"))
    assert streamed_arm_fits(sz.comp_gb * GB, 0.0, STRIX_HALO_CEILING, "unified", FIT_HEADROOM,
                             QWEN72B_EMBED_BYTES)
    assert streamed_arm_fits(sz.comp_gb * GB, 0.0, STRIX_HALO_CEILING, "unified", FIT_HEADROOM)
    assert not streamed_arm_fits(sz.bf16_gb * GB, 0.0, STRIX_HALO_CEILING, "unified",
                                 RATIO_HEADROOM, QWEN72B_EMBED_BYTES)
    assert not streamed_arm_fits(sz.bf16_gb * GB, 0.0, STRIX_HALO_CEILING, "unified", 1.0)


def test_rx7600xt_8b_fit_point_is_compressed_only_under_the_streamed_twin():
    """The 8B/granite fit points on the RX 7600 XT (16 GB VRAM): one
    bf16 copy (16.38 / 17.58 GB) does not fit a 16 GiB budget with headroom
    — the twin skip stands under the streamed charge (discrete: resident
    only), and the compressed arm (11.91 / 12.78 GB estimated, x 1.1) fits."""
    hw = Hardware(device_class="Radeon RX 7600 XT", memory_gb=16.0, memory_kind="vram", budget_gb=16.0,
                  cpu_info="x", gpu_info="y", evidence=[], device_source="table")
    for name in ("Qwen3-8B", "granite-4.2-8b"):
        sz = sizes(_m(name))
        assert not bench._twin_fit_ok(hw, sz.bf16_gb, None), name
        assert bench._compressed_fit(hw, sz.comp_gb, None) == (True, True), name


def _strix_halo(**kw):
    base = dict(device_class="strix-halo", memory_gb=133.144, memory_kind="unified", budget_gb=133.144,
                memory_bytes=133_144_000_000, budget_bytes=133_144_000_000,
                cpu_info="Strix Halo", gpu_info="amdgpu", evidence=["fixture"],
                device_source="cpu-brand")
    base.update(kw)
    return Hardware(**base)


def test_bench_fit_checks_for_the_72b_on_strix_halo():
    hw = _strix_halo()
    sz = sizes(_m("Qwen2.5-72B"))
    assert not bench._stock_fit_ok(hw, sz.bf16_gb, "stream", QWEN72B_EMBED_BYTES)
    assert not bench._twin_fit_ok(hw, sz.bf16_gb, QWEN72B_EMBED_BYTES)
    assert bench._compressed_fit(hw, sz.comp_gb, QWEN72B_EMBED_BYTES) == (True, True)
    # a pack that fits only without the headroom is a knife-edge attempt;
    # one that does not fit at all is neither
    assert bench._compressed_fit(hw, 125.0, QWEN72B_EMBED_BYTES) == (False, True)
    assert bench._compressed_fit(hw, 135.0, QWEN72B_EMBED_BYTES) == (False, False)
    line = bench._streamed_charge_line(hw, sz.comp_gb, bench._comp_what(sz), FIT_HEADROOM,
                                       QWEN72B_EMBED_BYTES)
    assert line.startswith("105.71 GB compressed (estimate) + one tensor (2.491 GB, streaming loader) "
                           "= 108.206 GB x1.1")
    # the ceiling is detect's memory_gb, decimal GB (124 GiB is 133.144 GB)
    assert "133.144 GB unified" in line


def test_dry_run_plans_the_72b_as_a_compressed_only_fit_point(monkeypatch):
    """The plan for the 72B target: no stock, no twin, the compressed
    arm charged ~106 GB (the estimate) + one tensor — the 5 GiB bound when
    the shards are not on the box, the real 2.49 GB embedding when they are."""
    monkeypatch.setattr(bench, "detect", _strix_halo)
    monkeypatch.setattr(bench, "cached_snapshot_dir", lambda repo, rev: None)
    (p,) = bench.dry_run(model_name="Qwen2.5-72B")["points"]
    assert p["arms"] == ["compressed"]
    assert p["stockArmPredictedToFit"] is False and p["twinArmPredictedToFit"] is False
    assert p["compressedArmPredictedToFit"] is True and p["compressedArmFitsWithoutHeadroom"] is False
    assert p["compressedArmChargeGB"] == pytest.approx(105.715 + STAGING_SHARD_BYTES / GB, abs=1e-3)
    assert p["twinArmChargeGB"] == pytest.approx(145.412 + STAGING_SHARD_BYTES / GB, abs=1e-3)
    assert p["compressedGB"] == 105.715 and p["compressedGBEstimated"] is True
    assert p["compressedGBFrom"].startswith("105.71 GB (estimate: 145.41 GB checkpoint x 0.727")
    assert p["compressedArmHeadroom"] == FIT_HEADROOM
    assert p["armLoaders"] == {"stock": "stream", "twin": "stream", "compressed": "stream"}
    assert p["recordShape"].startswith("fit-point (1 arm: compressed only")
    assert "raw.twin_outcome skipped_predicted_nonfit" in p["recordShape"]
    # the shards cached: the real largest tensor
    monkeypatch.setattr(bench, "cached_snapshot_dir", lambda repo, rev: "/snap")
    monkeypatch.setattr(bench, "largest_tensor_bytes", lambda snap: QWEN72B_EMBED_BYTES)
    (p,) = bench.dry_run(model_name="Qwen2.5-72B")["points"]
    assert p["compressedArmChargeGB"] == pytest.approx(108.206, abs=1e-3)
    assert p["largestTensorGB"] == pytest.approx(2.491, abs=1e-3)


def test_dry_run_plan_names_the_knife_edge_and_the_refusal(monkeypatch):
    from drinkme import suggest as suggest_mod

    monkeypatch.setattr(bench, "detect", _strix_halo)
    monkeypatch.setattr(bench, "cached_snapshot_dir", lambda repo, rev: None)
    # 124 GiB = 133.14 GB; the shards are not cached, so + the 5.37 GB bound:
    # 118 + 5.37 = 123.4 fits, x1.1 = 135.7 does not (knife-edge);
    # 130 + 5.37 = 135.4 does not fit at all
    knife = Model("Knife", "toy/knife", None)
    never = Model("Never", "toy/never", None)
    pin_sizes(monkeypatch, {"Knife": (200.0, 118.0), "Never": (200.0, 130.0)})
    monkeypatch.setattr(suggest_mod, "MODELS", [knife, never])
    (p,) = bench.dry_run(model_name="Knife")["points"]
    assert p["arms"] == ["compressed"]
    assert p["compressedArmPredictedToFit"] is False and p["compressedArmFitsWithoutHeadroom"] is True
    assert p["recordShape"].startswith("fit-point (1 arm: compressed only")
    (p,) = bench.dry_run(model_name="Never")["points"]
    assert p["compressedArmPredictedToFit"] is False and p["compressedArmFitsWithoutHeadroom"] is False
    assert p["recordShape"].startswith("none")


def test_dry_run_still_plans_a_two_arm_fit_point_when_the_twin_fits(monkeypatch):
    """gemma under the from_pretrained loader on the Strix Halo box: stock skipped
    (2x), the twin's single copy + one tensor fits — twin + compressed."""
    monkeypatch.setattr(bench, "detect", _strix_halo)
    monkeypatch.setattr(bench, "cached_snapshot_dir", lambda repo, rev: None)
    (p,) = bench.dry_run(model_name="gemma-4-31B-it", stock_loader="from_pretrained")["points"]
    assert p["arms"] == ["twin", "compressed"] and p["twinArmPredictedToFit"] is True
    assert p["recordShape"].startswith("fit-point (2 arms: twin+compressed")


# --------------------------------------------- the compressed live guard --


def test_compressed_live_guard_refuses_only_at_the_hard_line(capsys):
    m = _m("Qwen2.5-72B")
    comp = 110.28 * GB  # a 72B pack's resident bytes, for the knife-edge below
    # the Strix Halo box with a desktop resident: ~120 GB live -> 112.77 fits, not with x1.1: knife-edge, attempts
    assert arms.refuse_if_compressed_wont_fit(m.hf_repo, comp, "unified", 120 * GB,
                                              tensor_bytes=QWEN72B_EMBED_BYTES) is None
    out = capsys.readouterr().out
    assert "knife-edge" in out and "112.77 GB" in out and "120.00 GB" in out
    # plenty live: silent
    assert arms.refuse_if_compressed_wont_fit(m.hf_repo, comp, "unified", 128 * GB,
                                              tensor_bytes=QWEN72B_EMBED_BYTES) is None
    assert capsys.readouterr().out == ""
    # below the charge: refused, named, no record
    with pytest.raises(SystemExit, match="refusing to load") as ei:
        arms.refuse_if_compressed_wont_fit(m.hf_repo, comp, "unified", 100 * GB,
                                           tensor_bytes=QWEN72B_EMBED_BYTES)
    msg = str(ei.value)
    assert "compressed arm" in msg and "one tensor (2.49 GB" in msg and "no record" in msg
    # discrete: the host tensor is not VRAM's problem
    assert arms.refuse_if_compressed_wont_fit(m.hf_repo, 10 * GB, "vram", 11.5 * GB,
                                              tensor_bytes=5 * GB) is None
    # an unreadable box is not a known-safe one
    with pytest.raises(SystemExit, match="could not read live"):
        arms.refuse_if_compressed_wont_fit(m.hf_repo, comp, "unified", None)


# ---------------------------------------------- 4. run_arms + the record --


def _stub_run_arms_device(monkeypatch):
    class _Ids:
        shape = (1, 4)

        def cuda(self):
            return self

    class _Tok:
        def __call__(self, text, **kw):
            if kw.get("return_tensors"):
                return types.SimpleNamespace(input_ids=_Ids())
            return types.SimpleNamespace(input_ids=[1, 2, 3])

    monkeypatch.setitem(sys.modules, "transformers", types.SimpleNamespace(
        AutoTokenizer=types.SimpleNamespace(from_pretrained=lambda *a, **k: _Tok())))
    # the one source every arm loads from (tests/test_bench_one_source.py
    # has the resolution itself): a directory, resolved without the hub
    monkeypatch.setattr(arms, "resolve_source",
                        lambda repo, rev, pack_dir: (tempfile.gettempdir(), "deadbeef"))
    monkeypatch.setattr(arms.torch, "tensor", lambda *a, **k: _Ids())
    monkeypatch.setattr(arms.torch.cuda, "get_device_name", lambda i: "test-gpu")
    monkeypatch.setattr(arms, "measure_bandwidth", lambda: {
        "probe_bytes": GIB, "read_bytes_s": 200 * GB, "copy_bytes_s": 180 * GB, "device": "cuda"})
    monkeypatch.setattr(arms, "free_all", lambda: None)
    monkeypatch.setattr(arms, "vram_bytes", lambda: 500_000_000)
    monkeypatch.setattr(arms, "timed_decode", lambda model, ids, **k: (None, [10.0, 10.0, 10.0]))
    monkeypatch.setattr(arms, "timed_prefill", lambda model, ids, **k: [500.0, 500.0, 500.0])
    monkeypatch.setattr(arms, "timed_ttft", lambda model, ids, **k: [0.05, 0.05, 0.05])


STAT = {"name": "l", "shape": [1024, 1024], "numel": 1024 * 1024, "bits": 12 * 1024 * 1024,
        "bpw": 12.0, "format_version": 1, "codec": "radix", "profile": "sip", "widths": [3, 8],
        "dtype": None, "verified": True}

# raw's keys on the granite fit-point record (RX 7600 XT) — the shape
# a compressed-only fit point has
# the granite record's raw keys, plus `compressed_routing` (every
# measured arm's kernel routing rides in raw — None
# here, where the stubbed loader's tree is not an nn.Module and nothing
# was routed; a real arm's is serving/kernel_route.route_kernels's record),
# and `spec`, the speculation pass's outcome (spec_pass.assemble)
GRANITE_RAW_KEYS = {
    "bandwidth", "compressed_decode_samples", "compressed_decode_tok_s", "compressed_routing",
    "compressed_prefill_samples", "compressed_prefill_tok_s", "compressed_ttft_s",
    "compressed_ttft_samples", "compression_profile", "gpu", "mean_bpw", "model",
    "prefill_prompt_len", "resolved_revision", "revision", "stock_outcome", "stock_error", "stock_loader",
    "swapped_linears", "twin_outcome", "twin_decode_samples", "twin_decode_tok_s",
    "twin_error", "verification", "verified_tensors", "vram_compressed_bytes", "warmup_rep",
    "weighted_bpw", "host_load", "spec", "compressed_decode_step"}


def test_run_arms_compressed_only_streams_guards_and_keeps_the_raw_shape(monkeypatch):
    """The fit point where neither stock nor twin fits: pass 3 alone —
    through load_compressed_streaming, behind the compressed arm's live
    guard fed the same one-tensor term — and raw's keys are exactly the
    granite record's plus the compressed arm's routing, the step path its
    decode ran on (eager or a CUDA graph) and the host's load around its
    timing."""
    _stub_run_arms_device(monkeypatch)
    calls, guard = [], {}
    monkeypatch.setattr(arms, "live_available_bytes", lambda kind: 110 * GB)
    monkeypatch.setattr(arms, "refuse_if_compressed_wont_fit",
                        lambda model_id, comp, kind, live, **kw: guard.update(
                            model_id=model_id, comp=comp, kind=kind, live=live, **kw))
    monkeypatch.setattr(arms, "load_cpu", lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("no load_cpu on the fit point")))
    monkeypatch.setattr(arms, "load_stock_streaming", lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("no stock pass on the fit point")))
    monkeypatch.setattr(arms, "load_twin_streaming", lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("no twin pass when run_twin is False")))

    def load_compressed_streaming(model_id, revision=None, device="cuda", snap=None):
        calls.append((model_id, revision, device, snap))
        return object(), [dict(STAT)]

    monkeypatch.setattr(arms, "load_compressed_streaming", load_compressed_streaming)
    r = arms.run_arms("Qwen/Qwen2.5-72B-Instruct", "495f3936", "hi", run_stock=False,
                      stock_memory_kind="unified", stock_bf16_bytes=145.4 * GB, run_twin=False,
                      stock_loader="stream", stock_tensor_bytes=QWEN72B_EMBED_BYTES,
                      compressed_bytes=110.28 * GB)
    assert calls == [("Qwen/Qwen2.5-72B-Instruct", "495f3936", "cuda", tempfile.gettempdir())]
    assert guard == {"model_id": "Qwen/Qwen2.5-72B-Instruct", "comp": 110.28 * GB,
                     "kind": "unified", "live": 110 * GB, "tensor_bytes": QWEN72B_EMBED_BYTES}
    assert r["stock_outcome"] == "skipped_predicted_nonfit" and r["twin_outcome"] == "skipped_predicted_nonfit"
    assert r["twin_decode_tok_s"] is None and r["twin_decode_samples"] == []
    assert r["compressed_decode_tok_s"] == 10.0 and r["swapped_linears"] == 1
    assert r["compression_profile"] == "sip"
    assert set(r) == GRANITE_RAW_KEYS


def test_run_arms_ratio_point_streams_all_three_passes(monkeypatch):
    _stub_run_arms_device(monkeypatch)
    monkeypatch.setattr(arms, "live_available_bytes", lambda kind: 120 * GB)
    monkeypatch.setattr(arms, "refuse_if_stock_wont_fit", lambda *a, **k: None)
    order = []

    class _Model:
        def parameters(self):
            return iter([types.SimpleNamespace(numel=lambda: 1000)])

    monkeypatch.setattr(arms, "load_stock_streaming",
                        lambda *a, **k: order.append("stock") or _Model())
    plans = []
    monkeypatch.setattr(arms, "load_twin_streaming",
                        lambda *a, **k: order.append("twin") or plans.append(k["plan"]) or (_Model(), []))
    monkeypatch.setattr(arms, "load_compressed_streaming",
                        lambda *a, **k: order.append("compressed") or (_Model(), [dict(STAT)]))
    monkeypatch.setattr(arms, "load_cpu", lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("no load_cpu under the streaming loaders")))
    r = arms.run_arms("toy/model", None, "hi", run_stock=True, stock_memory_kind="unified",
                      stock_bf16_bytes=58.25 * GB, run_twin=True, stock_loader="stream",
                      compressed_bytes=44.82 * GB)
    assert order == ["stock", "compressed", "twin"]  # the twin is built off the compressed arm's stats
    assert r["stock_outcome"] == "measured" and r["twin_outcome"] == "measured"
    assert "twin_kind" not in r and "twin_launch_sources" not in r  # one twin: nothing to tell apart
    # the host's 1-minute load around each arm's timed passes (probe.record_host_load)
    assert list(r["host_load"]["load1"]) == ["stock", "compressed", "twin"]
    assert r["host_load"]["cpu_count"] == os.cpu_count()
    assert plans == [{"l": {"codec": "radix", "widths": [3, 8]}}]  # the compressed arm's stats
    assert r["stock_decode_tok_s"] == 10.0 and r["twin_decode_tok_s"] == 10.0


def _fake_raw_compressed_only() -> dict:
    return {
        "model": "Qwen/Qwen2.5-72B-Instruct", "revision": "495f39366efef23836d0cfae4fbe635880d2be31",
        "resolved_revision": "495f39366efef23836d0cfae4fbe635880d2be31",
        "gpu": "AMD Radeon Graphics", "prefill_prompt_len": 512, "warmup_rep": True,
        "bandwidth": {"probe_bytes": 4 * GIB, "read_bytes_s": 200_000_000_000,
                      "copy_bytes_s": 180_000_000_000, "device": "cuda",
                      "total_param_bytes_bf16": 145_400_000_000, "ceiling_tok_s_bf16": 1.38},
        "stock_loader": "stream",
        "stock_outcome": "skipped_predicted_nonfit",
        "stock_error": "skipped by bench's stock-arm fit check (predicted OOM under the stream loader's transient)",
        "twin_outcome": "skipped_predicted_nonfit",
        "twin_error": "twin skipped: a single bf16 copy does not fit the budget (fit point is compressed-only)",
        "twin_decode_tok_s": None, "twin_decode_samples": [],
        "vram_compressed_bytes": 110_280_000_000,
        "compressed_decode_tok_s": 2.1, "compressed_decode_samples": [2.0, 2.1, 2.2],
        "compressed_prefill_tok_s": 120.0, "compressed_prefill_samples": [119.0, 120.0, 121.0],
        "compressed_ttft_s": 4.3, "compressed_ttft_samples": [4.2, 4.3, 4.4],
        "swapped_linears": 560, "verified_tensors": 560,
        "compression_profile": "sip", "mean_bpw": 12.03, "weighted_bpw": 12.36,
        # the one arm that ran, routed: a dense model
        "compressed_routing": {"deltanet_kernel": "none", "deltanet_conv": "none",
                               "narrow_gemv": True, "narrow_count": 0, "stock_gemv": "mv",
                               "raw_gemv": "twin"},
    }


def test_the_72b_fit_point_record_has_the_granite_shape(linux_host, monkeypatch, capsys):
    """bench.run through the real _run_point (only detect/suggest/run_arms
    stubbed) on the Strix Halo box's numbers: stock.outcome + raw.twin_outcome
    skipped_predicted_nonfit, the budget, compressed metrics only."""
    m = _m("Qwen2.5-72B")
    captured = {}

    def fake_run_arms(repo, rev, prompt, **kw):
        captured.update(kw)
        return _fake_raw_compressed_only()

    monkeypatch.setattr(bench, "detect", _strix_halo)
    monkeypatch.setattr(bench, "gemma_gate_open", lambda: False)
    monkeypatch.setattr(bench, "cached_snapshot_dir", lambda repo, rev: "/snap")
    monkeypatch.setattr(bench, "largest_tensor_bytes", lambda snap: QWEN72B_EMBED_BYTES)
    monkeypatch.setattr(arms, "run_arms", fake_run_arms)
    # the real suggest(): the Strix Halo box's menu is the 32B ratio point + the 72B
    # fit point; --model names the fit point alone
    r = bench.run(no_gemma=True, model_name=m.name)
    out = capsys.readouterr().out
    assert "fit point    Qwen2.5-72B" in out
    assert captured["run_stock"] is False and captured["run_twin"] is False
    assert captured["compressed_bytes"] == pytest.approx(sizes(m).comp_gb * GB)
    assert captured["stock_tensor_bytes"] == QWEN72B_EMBED_BYTES
    assert "compressed arm only" in out
    assert "compressed-arm fit check: 105.71 GB compressed (estimate) + one tensor (2.491 GB" in out
    assert r["stock"] == {"outcome": "skipped_predicted_nonfit", "budgetBytes": 133_144_000_000}
    assert "twinArm" not in r and r["raw"]["twin_outcome"] == "skipped_predicted_nonfit"
    names = [x["name"] for x in r["metrics"]]
    assert names == ["compressed_decode_tok_s", "compressed_prefill_tok_s", "compressed_ttft_s",
                     "read_gb_s", "copy_gb_s", "compressed_weights_gb"]
    assert V.validate(bench.lexicon_safe(r)) == []


def test_the_compressed_only_record_passes_check_points(linux_host, monkeypatch, tmp_path):
    """The same record through the site's own wrap + check_points.py (the
    real scripts, copied under tmp_path — test_bench_record_shape's
    discipline)."""
    from tests.test_bench_record_shape import _check_points_on

    monkeypatch.setattr(bench, "detect", _strix_halo)
    monkeypatch.setattr(bench, "gemma_gate_open", lambda: False)
    monkeypatch.setattr(bench, "cached_snapshot_dir", lambda repo, rev: None)
    monkeypatch.setattr(arms, "run_arms", lambda *a, **k: _fake_raw_compressed_only())
    r = bench.run(no_gemma=True, model_name="Qwen2.5-72B")
    assert r["raw"]["twin_outcome"] == "skipped_predicted_nonfit"
    _check_points_on(tmp_path, r)


def test_run_point_refuses_a_compressed_arm_that_cannot_fit_at_all(linux_host, monkeypatch):
    """No fallback arm: bench says so before any load and writes nothing."""
    from drinkme import suggest as suggest_mod

    never = Model("Never", "toy/never", None)
    pin_sizes(monkeypatch, {"Never": (200.0, 130.0)})
    monkeypatch.setattr(suggest_mod, "MODELS", [never])
    monkeypatch.setattr(bench, "detect", _strix_halo)
    monkeypatch.setattr(bench, "gemma_gate_open", lambda: False)
    monkeypatch.setattr(bench, "cached_snapshot_dir", lambda repo, rev: None)
    monkeypatch.setattr(bench, "suggest", lambda *a, **k: Suggestion(
        ratio=never, ratio_extra=[], fit=None, fit_knife_edge=False, fit_honest_negative=False,
        notes=[]))
    monkeypatch.setattr(arms, "run_arms", lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("run_arms must not run")))
    with pytest.raises(SystemExit, match="refusing to run") as ei:
        bench.run(no_gemma=True, model_name="Never")
    assert "does not fit even without headroom" in str(ei.value)


def test_the_lexicon_keeps_the_twin_in_raw():
    props = V.load_lexicon()["defs"]["main"]["record"]["properties"]
    assert "twinArm" not in props
    assert "the twin arm" in props["raw"]["description"]
