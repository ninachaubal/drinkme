"""Muse-Glimmer end to end at toy scale, CPU only.

muse_glimmer is the composite VLM whose text skeleton IS the wrapper, its
vision tower pruned when images are off (arms._VISION_TOWERS; this toy
ships no processor config, so its server never serves images): transformers
ships no text-only causal class for it, and the wrapper's forward carries the logit
path (output_multiplier, then the tanh softcap) the text tower alone does
not. tests/test_pack.py pins the tables at the key-set level; this file
runs the two verbs over a real (tiny) checkpoint that SHIPS a vision tower:
`pack` carries those tensors and a text server skips them by
name, and `serve` (load_compressed on the
CPU route) must produce logits BITWISE the stock arm's over the same bytes
— the tie_weights step, the rotary rebuild through the owning text config,
and the softcap all on the served tree.
"""

import json
import os

import pytest
import torch

pytest.importorskip("transformers.models.muse_glimmer",
                    reason="muse_glimmer arrived in transformers 5.15.0")

from drinkme.codec.pack import iter_pack_dir, pack_model  # noqa: E402
from drinkme.codec.swap import CompressedLinear  # noqa: E402
from drinkme.serving.engines import load_compressed, load_stock  # noqa: E402

LAYERS = 2
VOCAB = 1024
# per layer: self_attn.{q,o,gate}_proj + mlp.{gate,up,down}_proj clear the
# codec's min-dim floor at hidden 1024; k/v_proj (2 kv heads x 128 = 256
# rows) do not — exactly the 30B's split (313 = 52 x 6 + lm_head)
PACKED_PER_LAYER = 6
RAW_PER_LAYER = 2 + 4  # k_proj, v_proj + four norms
N_PACKED = LAYERS * PACKED_PER_LAYER + 1  # + the untied lm_head
N_RAW = LAYERS * RAW_PER_LAYER + 2  # + embed_tokens, final norm
# the tower's Linears the codec takes: the 1024 x 1176 patch embedding, two
# layers of q/k/v/proj and fc1/fc2, the adapter's fc1/fc2, the projection
N_TOWER_PACKED = 1 + 2 * 6 + 2 + 1


def _config():
    from transformers.models.muse_glimmer import MuseGlimmerConfig

    return MuseGlimmerConfig(
        text_config=dict(vocab_size=VOCAB, hidden_size=1024, intermediate_size=1024,
                         num_hidden_layers=LAYERS, num_attention_heads=8,
                         num_key_value_heads=2, head_dim=128,
                         max_position_embeddings=128, sliding_window=64,
                         bos_token_id=None, eos_token_id=2, pad_token_id=1),
        vision_config=dict(hidden_size=1024, intermediate_size=1024,
                           num_hidden_layers=2, num_attention_heads=8,
                           pos_emb_height=4, pos_emb_width=4, max_position_embeddings=16),
        out_hidden_size=1024, projector_hidden_size=1024,
    )


@pytest.fixture(scope="module")
def glimmer_toy(tmp_path_factory):
    """(model_dir, pack_dir): a saved 2-layer muse_glimmer WITH its vision
    tower, and its pack through the real pack_model."""
    from fixtures import word_tokenizer as _tokenizer
    from transformers.models.muse_glimmer import MuseGlimmerForConditionalGeneration

    from drinkme.codec.swap import eligible_linears

    d = tmp_path_factory.mktemp("glimmer_toy")
    model_dir, pack_dir = str(d / "model"), str(d / "pack")
    os.makedirs(model_dir)
    _tokenizer(VOCAB).save_pretrained(model_dir)
    torch.manual_seed(0)
    model = MuseGlimmerForConditionalGeneration(_config()).to(torch.bfloat16).eval()
    with torch.no_grad():
        for _, _, _, child in eligible_linears(model):
            child.weight.data.mul_(1 / 64)
    model.save_pretrained(model_dir, safe_serialization=True)
    del model
    pack_model(model_dir, None, pack_dir, progress=lambda *_: None)
    return model_dir, pack_dir


def _no_reuse(loader, *a, **kw):
    os.environ["DRINKME_PREFIX_SLOTS"] = "0"
    try:
        return loader(*a, **kw)
    finally:
        del os.environ["DRINKME_PREFIX_SLOTS"]


@pytest.fixture(scope="module")
def comp(glimmer_toy):
    return _no_reuse(load_compressed, glimmer_toy[0], None, glimmer_toy[1], device="cpu")


@pytest.fixture(scope="module")
def stock(glimmer_toy):
    return _no_reuse(load_stock, glimmer_toy[0], None, device="cpu")


# ------------------------------------------------------------------ pack --

def test_the_checkpoint_ships_a_vision_tower(glimmer_toy):
    """The premise: what the packer walks past really is on disk."""
    from safetensors import safe_open

    model_dir, _ = glimmer_toy
    with safe_open(os.path.join(model_dir, "model.safetensors"), framework="pt") as sf:
        keys = list(sf.keys())
    assert any(k.startswith("model.vision_tower.") for k in keys)
    assert "model.vision_projection.weight" in keys
    assert "lm_head.weight" in keys  # untied, shipped


def test_pack_carries_the_vision_tower_beside_the_text(glimmer_toy):
    """The packer's tree holds the tower: its eligible
    Linears are packed and the rest counted raw, all of it in the `vision`
    block; the text half is what it was, and a text server skips the
    tower by path."""
    from drinkme.codec.pack import under

    _, pack_dir = glimmer_toy
    meta = json.load(open(os.path.join(pack_dir, "meta.json")))
    names = [n for n, _ in iter_pack_dir(pack_dir)]
    paths = meta["vision"]["paths"]
    text = [n for n in names if not under(n, paths)]
    assert set(names) == set(meta["tensors"])
    assert len(text) == N_PACKED and len(names) == N_PACKED + N_TOWER_PACKED
    assert meta["tensorCount"] == len(names) == meta["radixTensorCount"]
    assert "fp8TensorCount" not in meta and meta["dtype"] == "bf16"
    assert "lm_head" in names
    assert all(n == "lm_head" or n.startswith("model.language_model.layers.") for n in text)
    assert meta["vision"]["packedTensorCount"] == N_TOWER_PACKED
    assert [n for n, _ in iter_pack_dir(pack_dir, skip=paths)] == text
    # the raw count is the text remainder and the tower's raw rest
    assert meta["rawTensorCount"] == N_RAW + meta["vision"]["tensorCount"] - N_TOWER_PACKED


# ----------------------------------------------------------------- serve --

def test_served_tree_is_the_pruned_wrapper(comp, stock):
    for eng in (comp, stock):
        m = eng.model
        assert type(m).__name__ == "MuseGlimmerForConditionalGeneration"
        assert (m.model.vision_tower, m.model.vision_adapter, m.model.vision_projection) \
            == (None, None, None)
        assert not any(p.is_meta for p in m.parameters())
        assert not any(b.is_meta for b in m.buffers())
        # untied: the pack's lm_head is served from the pack, not retied to
        # the embedding by load_compressed's tie_weights() step
        assert m.lm_head.weight is not m.model.language_model.embed_tokens.weight \
            if isinstance(m.lm_head, torch.nn.Linear) else True
    swapped = [n for n, mod in comp.model.named_modules() if isinstance(mod, CompressedLinear)]
    assert len(swapped) == N_PACKED and "lm_head" in swapped
    assert isinstance(stock.model.lm_head, torch.nn.Linear)
    # the one rotary was rebuilt from the TEXT config (its owner's), not the
    # top-level wrapper config, which has no rope_parameters
    rot = comp.model.model.language_model.rotary_emb
    assert torch.equal(rot.inv_freq, stock.model.model.language_model.rotary_emb.inv_freq)


def test_cpu_logits_are_bitwise_the_stock_arm(comp, stock):
    """The claim at toy scale: the compressed tree over the pack's bytes
    equals from_pretrained over the shards, through the WRAPPER's forward
    (output_multiplier, then the tanh softcap — which a headless text model
    with a bolted-on lm_head would silently drop)."""
    cap = comp.model.config.text_config.final_logit_softcapping
    ids = torch.arange(3, 15)[None]
    with torch.inference_mode():
        want = stock.model(ids, use_cache=False).logits
        got = comp.model(ids, use_cache=False).logits
    assert torch.equal(got, want), (got.float() - want.float()).abs().max().item()
    assert got.abs().max().item() <= cap  # the softcap ran on the served tree


def test_greedy_completion_agrees_with_stock(comp, stock):
    """One real turn through the engine, both arms, same tokens out."""
    from drinkme.serving.engine import GenerationRequest, SampleParams, complete

    msgs = [{"role": "user", "content": "hello world the quick brown fox"}]
    outs = []
    for eng in (comp, stock):
        res = complete(eng, GenerationRequest(
            msgs, SampleParams(temperature=0.0, max_tokens=8, stop=[])))
        assert res.completion_tokens > 0
        outs.append((res.text, res.finish_reason, res.completion_tokens))
    assert outs[0] == outs[1], outs
