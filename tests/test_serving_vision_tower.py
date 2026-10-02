"""The vision tower's own pins, CPU.

BOUNDED ATTENTION: every ViT attention dispatch at most WORK_V query-key
pairs. The ViT's attention over one image is one bidirectional call over all
of its patches. vision.bounded_attention splits the query rows into blocks
(exact: each row sees every key, nothing is combined) and, past MIN_ROWS
rows a block, splits the keys too and combines by log-sum-exp. Checked in
float32 against one SDPA call with the bounds forced tiny, and through a
real (toy) Qwen3.5 tower against the stock forward.

THE TOWER IN THE SERVED TREE: a toy Qwen3.5 checkpoint with a vision config
is packed and loaded both arms: the pack carries the tower (its eligible
Linear compressed, the rest raw in the embedded checkpoint, meta.json's
`vision` block), the served tower is transformers' own bit for bit,
DRINKME_VISION=0 builds none of it, and a pack cut before the tower was
served refuses images with the re-pack line.
"""

from __future__ import annotations

import json
import os
import shutil

import numpy as np
import pytest
import torch

from drinkme.serving import vision

from test_serving_image_prompt import _tokenizer, config, png, user
from test_serving_mtp import _torch_chunk_on_cpu  # noqa: F401 — fixture by name

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


class _Attn(torch.nn.Module):
    """The attributes transformers' sdpa_attention_forward reads."""

    is_causal = False
    num_key_value_groups = 1


def _stock(q, k, v, scale):
    from transformers.integrations.sdpa_attention import sdpa_attention_forward

    return sdpa_attention_forward(_Attn(), q, k, v, None, scaling=scale, is_causal=False)[0]


def _qkv(n, heads=4, dim=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(1, heads, n, dim, generator=g) for _ in range(3)]


def test_under_the_bound_it_is_the_stock_call():
    q, k, v = _qkv(40)
    out, _ = vision.bounded_attention(_Attn(), q, k, v, scaling=0.3, work=40 * 40)
    assert torch.equal(out, _stock(q, k, v, 0.3))


@pytest.mark.parametrize("n,work", [(40, 40 * 7), (97, 97 * 5), (64, 64 * 64 - 1)])
def test_split_queries_are_the_stock_call(n, work, monkeypatch):
    """Query blocks of work // N rows, each the stock call on its rows:
    bitwise one SDPA call on the CPU (a row's softmax never sees another
    row), and no block over `work` pairs."""
    monkeypatch.setattr(vision, "MIN_ROWS", 1)
    q, k, v = _qkv(n, seed=n)
    seen = []
    import transformers.integrations.sdpa_attention as sd

    real = sd.sdpa_attention_forward

    def rec(module, query, key, *a, **kw):
        seen.append(query.shape[-2] * key.shape[-2])
        return real(module, query, key, *a, **kw)

    monkeypatch.setattr(sd, "sdpa_attention_forward", rec)
    out, _ = vision.bounded_attention(_Attn(), q, k, v, scaling=0.25, work=work)
    monkeypatch.setattr(sd, "sdpa_attention_forward", real)
    assert len(seen) > 1 and max(seen) <= work
    assert torch.equal(out, _stock(q, k, v, 0.25))


@pytest.mark.parametrize("n,rows,work", [(50, 4, 4 * 16), (33, 8, 8 * 7), (64, 2, 2 * 5)])
def test_split_keys_combine_by_log_sum_exp(n, rows, work, monkeypatch):
    """Past MIN_ROWS rows a block the keys are split as well: fp32, within
    reduction-order noise of the one call, no piece over `work` pairs."""
    monkeypatch.setattr(vision, "MIN_ROWS", rows)
    q, k, v = _qkv(n, seed=n + 1)
    from drinkme.serving import segmented_attention as sa

    seen = []
    real = sa._partial

    def rec(q_, k_, v_, causal, scale):
        assert not causal
        seen.append(q_.shape[-2] * k_.shape[-2])
        return real(q_, k_, v_, causal, scale)

    monkeypatch.setattr(sa, "_partial", rec)
    out, _ = vision.bounded_attention(_Attn(), q, k, v, scaling=0.2, work=work)
    assert len(seen) > n // rows and max(seen) <= work
    torch.testing.assert_close(out, _stock(q, k, v, 0.2), atol=2e-6, rtol=1e-5)


def _tower():
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5VisionConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionModel

    cfg = Qwen3_5VisionConfig(depth=2, hidden_size=32, intermediate_size=64, num_heads=2,
                              in_channels=3, patch_size=4, spatial_merge_size=2,
                              temporal_patch_size=2, out_hidden_size=48,
                              num_position_embeddings=16)
    cfg._attn_implementation = "sdpa"
    torch.manual_seed(3)
    vit = Qwen3_5VisionModel(cfg).eval().float()
    for p in vit.parameters():
        p.requires_grad_(False)
    root = torch.nn.Module()
    root.model = torch.nn.Module()
    root.model.visual = vit
    return root


@pytest.mark.parametrize("min_rows", [4, 8])
def test_the_tower_under_a_tiny_bound_is_the_stock_tower(monkeypatch, min_rows):
    """bound() routes every ViT attention through bounded_attention: a
    16 x 12 patch grid (192 patches, 36,864 pairs per layer) under a
    1,000-pair bound (blocks of 5 query rows; with MIN_ROWS 8 the keys are
    split too) is the stock forward, and the config the tower was built
    with keeps "sdpa" (its layers get their own copies)."""
    import copy

    root = _tower()
    stock = copy.deepcopy(root)
    tower = vision.Tower("qwen3_5", "model.visual", 0, 2)
    monkeypatch.setattr(vision, "WORK_V", 1000)
    monkeypatch.setattr(vision, "MIN_ROWS", min_rows)
    shared = root.model.visual.config
    assert vision.bound(tower, root) == 2
    assert vision.bound(tower, root) == 2  # idempotent
    assert shared._attn_implementation == "sdpa"
    g = torch.Generator().manual_seed(9)
    pv = torch.randn(16 * 12, 3 * 2 * 4 * 4, generator=g)
    grid = torch.tensor([[1, 16, 12]])
    with torch.inference_mode():
        got = root.model.visual(pv, grid_thw=grid).pooler_output
        want = stock.model.visual(pv, grid_thw=grid).pooler_output
    torch.testing.assert_close(got, want, atol=2e-5, rtol=1e-5)


def test_the_bound_can_be_switched_off_for_the_bit_pin(monkeypatch):
    """DRINKME_VISION_BOUNDED=0: one attention call per image, through
    one_call_attention (so the ROCm head padding still applies), the stock
    tower bit for bit on the CPU; bound() again without the switch
    re-routes to the bound."""
    import copy

    root = _tower()
    stock = copy.deepcopy(root)
    tower = vision.Tower("qwen3_5", "model.visual", 0, 2)
    monkeypatch.setattr(vision, "WORK_V", 1000)
    monkeypatch.setenv(vision.BOUNDED_ENV, "0")
    assert vision.bound(tower, root) == 0
    attn = root.model.visual.blocks[0].attn
    assert attn.config._attn_implementation == vision.UNBOUNDED_NAME
    g = torch.Generator().manual_seed(9)
    pv = torch.randn(16 * 12, 3 * 2 * 4 * 4, generator=g)
    grid = torch.tensor([[1, 16, 12]])
    with torch.inference_mode():
        assert torch.equal(root.model.visual(pv, grid_thw=grid).pooler_output,
                           stock.model.visual(pv, grid_thw=grid).pooler_output)
    monkeypatch.delenv(vision.BOUNDED_ENV)
    assert vision.bound(tower, root) == 2
    assert attn.config._attn_implementation == vision.BOUNDED_NAME


@pytest.mark.parametrize("work,min_rows", [(10**9, 256), (400, 4), (400, 16)])
@pytest.mark.parametrize("scaling", [None, 0.3])
def test_on_rocm_the_head_is_padded_to_a_multiple_of_sixteen(monkeypatch, work, min_rows,
                                                             scaling):
    """AOTriton on gfx1151 is wrong at the ViT's head width (72; the toy's
    is 8): on a ROCm device every attention call below bounded_attention
    (one call, query blocks, key segments) sees the head zero-padded to a
    multiple of HEAD_ALIGN, the scale stays the unpadded head's, and the
    output is the unpadded call's with the padding dropped. Off ROCm the
    head goes through as it is."""
    from drinkme.serving import segmented_attention as sa
    from transformers.integrations import sdpa_attention

    q, k, v = _qkv(40)
    want = _stock(q, k, v, scaling if scaling is not None else 8 ** -0.5)
    seen = []
    orig_sdpa, orig_partial = sdpa_attention.sdpa_attention_forward, sa._partial

    def rec_sdpa(module, query, *a, **kw):
        seen.append(query.shape[-1])
        return orig_sdpa(module, query, *a, **kw)

    def rec_partial(query, *a, **kw):
        seen.append(query.shape[-1])
        return orig_partial(query, *a, **kw)

    monkeypatch.setattr(sdpa_attention, "sdpa_attention_forward", rec_sdpa)
    monkeypatch.setattr(sa, "_partial", rec_partial)
    monkeypatch.setattr(vision, "MIN_ROWS", min_rows)
    monkeypatch.setattr(vision, "_rocm", lambda x: True)
    out, _ = vision.bounded_attention(_Attn(), q, k, v, scaling=scaling, work=work)
    assert seen and set(seen) == {vision.HEAD_ALIGN}
    assert out.shape == (1, 40, 4, 8) and out.is_contiguous()
    torch.testing.assert_close(out, want, atol=2e-6, rtol=1e-5)
    monkeypatch.setattr(vision, "_rocm", lambda x: False)
    seen.clear()
    vision.bounded_attention(_Attn(), q, k, v, scaling=scaling, work=work)
    assert set(seen) == {8}


def _no_conv(*_a, **_k):
    raise AssertionError("the patch embedding called its Conv3d")


@pytest.mark.parametrize("bounded", ["1", "0"])
def test_the_patch_embed_is_one_gemm_on_rocm_and_the_stock_conv_elsewhere(monkeypatch, bounded):
    """route_patch_embed: on a ROCm device the patch embedding never calls
    its Conv3d (MIOpen's first call per input shape runs a multi-second
    naive kernel, vision.linear_patch_embed), and gives the conv's values
    within fp32 rounding, for the embedding and for the whole tower. Off
    ROCm (here, the CPU) it is the stock forward, bit for bit.
    DRINKME_VISION_BOUNDED=0 does not switch it off."""
    import copy

    monkeypatch.setenv(vision.BOUNDED_ENV, bounded)
    root = _tower()
    stock = copy.deepcopy(root)
    tower = vision.Tower("qwen3_5", "model.visual", 0, 2)
    assert vision.route_patch_embed(tower, root) == 1
    g = torch.Generator().manual_seed(5)
    pv = torch.randn(16 * 12, 3 * 2 * 4 * 4, generator=g)
    grid = torch.tensor([[1, 16, 12]])
    with torch.inference_mode():
        want = stock.model.visual.patch_embed(pv)
        want_out = stock.model.visual(pv, grid_thw=grid).pooler_output
        assert torch.equal(root.model.visual.patch_embed(pv), want)
        monkeypatch.setattr(vision, "_rocm", lambda x: True)
        monkeypatch.setattr(torch.nn.Conv3d, "forward", _no_conv)
        got = root.model.visual.patch_embed(pv)
        got_out = root.model.visual(pv, grid_thw=grid).pooler_output
    assert got.shape == want.shape and got.dtype == want.dtype
    torch.testing.assert_close(got, want, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(got_out, want_out, atol=2e-5, rtol=1e-5)


def test_the_rocm_check_names_rocm_devices_only(monkeypatch):
    class _X:  # what _rocm reads off a tensor
        def __init__(self, kind):
            self.device = torch.device(kind)

    monkeypatch.setattr(torch.version, "hip", "7.13", raising=False)
    assert vision._rocm(_X("cuda")) and not vision._rocm(_X("cpu"))
    monkeypatch.setattr(torch.version, "hip", None, raising=False)
    assert not vision._rocm(_X("cuda"))  # a CUDA build: cuDNN and its own SDPA


# ------------------------------------------------ THE TOWER IN THE PACK --
#
# A toy Qwen3.5 checkpoint with a vision config, saved in bf16 the way
# transformers saves Qwen3_5ForConditionalGeneration (model.language_model.*,
# model.visual.*, lm_head), with the chat template and a processor config.
# The ViT is 256 wide, so exactly one of its Linears clears the codec's
# eligibility bar (the merger's fc1, 1024 x 1024): the pack carries it
# compressed and the other 32 tower tensors raw in the embedded checkpoint.

PROCESSOR = {"image_processor_type": "Qwen2VLImageProcessorFast", "patch_size": 4,
             "temporal_patch_size": 2, "merge_size": 2,
             "size": {"shortest_edge": 64, "longest_edge": 4096},
             "image_mean": [0.5, 0.5, 0.5], "image_std": [0.5, 0.5, 0.5]}
MSGS = [user("what is in this picture", None)]


def _quiet(*_a, **_k):
    pass


def _vision_config():
    cfg = config()
    cfg.vision_config.hidden_size = 256
    cfg.vision_config.intermediate_size = 256
    cfg.vision_config.num_heads = 4
    return cfg


def _checkpoint(path: str, processor: dict | None = PROCESSOR) -> None:
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForConditionalGeneration

    torch.manual_seed(0)
    Qwen3_5ForConditionalGeneration(_vision_config()).to(torch.bfloat16).eval().save_pretrained(
        path, safe_serialization=True)
    _tokenizer().save_pretrained(path)
    if processor is not None:
        with open(os.path.join(path, "preprocessor_config.json"), "w") as f:
            json.dump(processor, f)


def _meta(pack):
    with open(os.path.join(pack, "meta.json")) as f:
        return json.load(f)


def _headers(path):
    from drinkme.codec.pack import _shard_headers

    return _shard_headers(path)


@pytest.fixture(scope="module")
def packed(tmp_path_factory):
    """(checkpoint dir, pack dir, the reference's view): packed once. The
    reference (transformers' own composite over the checkpoint) is read
    before any test deletes the checkpoint."""
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForConditionalGeneration

    from drinkme.codec.pack import pack_model

    root = tmp_path_factory.mktemp("vision-pack")
    model, pack = str(root / "model"), str(root / "pack")
    _checkpoint(model)
    pack_model(model, None, pack, progress=_quiet)
    ref = Qwen3_5ForConditionalGeneration.from_pretrained(model, dtype=torch.bfloat16).eval()
    img = _image()
    with torch.inference_mode():
        feats = ref.model.get_image_features(torch.tensor(img.pixel_values.copy()),
                                             torch.tensor([img.grid_thw])).pooler_output[0]
    return model, pack, feats


def _image():
    vis = vision.Vision(vision.QwenVLPreprocessor.from_config("qwen3_5", PROCESSOR))
    return vis.prepare(vision.parse_base64(png(16, 24, 5), "image/png", where="test"))


def _env(**kw):
    from test_serving_prefill_bound import env

    return env(DRINKME_PREFIX_SLOTS=0, DRINKME_SLOT_DIR="off", **kw)


def _load(model, pack, **kw):
    from drinkme.serving.engines import load_compressed

    with _env(**kw):
        return load_compressed(model, None, pack, device="cpu")


def _load_stock(model, **kw):
    from drinkme.serving.engines import load_stock

    with _env(**kw):
        return load_stock(model, None, device="cpu")


def _greedy(eng, images=(), n=8):
    from drinkme.serving.engine import GenerationRequest, SampleParams, complete

    msgs = MSGS if images else [{"role": "user", "content": "hello world the quick brown fox"}]
    with _env(DRINKME_SPEC="off"):
        return complete(eng, GenerationRequest(msgs, SampleParams(temperature=0.0, max_tokens=n),
                                               images=tuple(images)))


def test_the_pack_carries_the_tower_and_its_vision_block(packed):
    """The tower's eligible Linear is packed like any other; every other
    tower tensor (patch_embed, pos_embed, the norms, the biases, the Linears
    below the bar) rides raw in the embedded checkpoint; meta.json's
    `vision` block names the tower and counts it, and its resident bytes
    are the raw tensors' plus the packed one's; `drinkme verify` passes."""
    from safetensors import safe_open

    from drinkme.codec.pack import REMAINDER_FILE, embedded_dir, verify_pack

    model, pack, _ = packed
    meta = _meta(pack)
    tower = sorted(k for k in _headers(model) if k.startswith("model.visual."))
    packed_names = [n for n in meta["tensors"] if n.startswith("model.visual.")]
    assert packed_names == ["model.visual.merger.linear_fc1"]
    with safe_open(os.path.join(embedded_dir(pack), REMAINDER_FILE), "pt") as f:
        raw = sorted(k for k in f.keys() if k.startswith("model.visual."))
    assert raw == [k for k in tower if k != "model.visual.merger.linear_fc1.weight"]
    assert {"model.visual.patch_embed.proj.weight", "model.visual.pos_embed.weight",
            "model.visual.blocks.0.norm1.weight", "model.visual.merger.linear_fc1.bias"} <= set(raw)
    block = meta["vision"]
    assert block["tower"] == "model.visual" and block["imageTokenId"] == config().image_token_id
    assert block["tensorCount"] == len(tower) and block["packedTensorCount"] == 1
    raw_bytes = sum(2 * int(np.prod(shape)) for k, (_dt, shape, _f) in _headers(model).items()
                    if k in raw)
    assert raw_bytes < block["residentBytes"] < meta["residentBytes"]
    verify_pack(pack, progress=_quiet)


def test_the_served_tower_is_the_references_bit_for_bit_offline(packed, tmp_path, monkeypatch):
    """Loaded offline from the pack alone (the checkpoint gone, no snapshot
    resolution): the tower sits at model.visual with its packed Linear, the
    vision rotary rebuilt, and one image's features are transformers'
    get_image_features' bit for bit (the CPU compressed arm decodes the
    exact bf16 weight, bounded attention at this size is the stock call).
    The stock arm (from_pretrained's text tree plus the tower streamed in)
    gives the same features and the same greedy answer about the image."""
    from drinkme import arms
    from drinkme.codec.swap import CompressedLinear
    from drinkme.serving import checkpoint
    from drinkme.serving.engine import GenerationRequest, SampleParams

    model, pack, want = packed
    stock = _load_stock(model)
    assert type(stock.model.model.visual.merger.linear_fc1) is torch.nn.Linear
    stock_text = _greedy(stock, [_image()]).text
    with torch.inference_mode():
        assert torch.equal(stock._tower.features(stock.model, _image()), want)

    copy_dir = str(tmp_path / "model")
    shutil.copytree(model, copy_dir)
    pack_copy = str(tmp_path / "pack")
    shutil.copytree(pack, pack_copy)
    shutil.rmtree(copy_dir)

    def no_snapshot(*a, **k):
        raise AssertionError("a self-contained load resolved a snapshot")

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setattr(checkpoint, "snapshot_dir", no_snapshot)
    monkeypatch.setattr(arms, "snapshot_dir", no_snapshot)
    eng = _load(copy_dir, pack_copy)
    assert eng.vision is not None and eng.vision_reason is None
    vit = eng.model.get_submodule("model.visual")
    for e in (eng, stock):  # the engine routes the patch embedding on both arms
        assert e.model.model.visual.patch_embed.forward.__func__ is vision.linear_patch_embed
    assert isinstance(vit.merger.linear_fc1, CompressedLinear)
    assert torch.equal(vit.rotary_pos_emb.inv_freq,
                       stock.model.model.visual.rotary_pos_emb.inv_freq)
    with torch.inference_mode():
        assert torch.equal(eng._tower.features(eng.model, _image()), want)
    res = _greedy(eng, [_image()])
    assert res.text == stock_text
    assert res.prompt_tokens == eng.count_tokens(
        GenerationRequest(MSGS, SampleParams(), images=(_image(),)))


def test_images_off_builds_no_tower_and_reads_none_of_it(packed, monkeypatch):
    """DRINKME_VISION=0: no tower in the tree, the pack's tower tensors
    never opened, images refused naming the switch, text served as before,
    and the fit charges the pack less the tower's resident bytes."""
    from drinkme import arms, fit
    from drinkme.codec import pack as pack_mod

    model, pack, _ = packed
    meta = _meta(pack)
    by_file = {os.path.basename(fn): name for name, fn in meta["tensors"].items()}
    opened = []
    real = pack_mod.np.load

    def spy(file, *a, **k):
        opened.append(by_file.get(os.path.basename(str(file))))
        return real(file, *a, **k)

    monkeypatch.setattr(pack_mod.np, "load", spy)
    eng = _load(model, pack, DRINKME_VISION=0)
    monkeypatch.setattr(pack_mod.np, "load", real)
    assert eng.vision is None and eng.vision_reason == "disabled by DRINKME_VISION=0"
    assert not arms.has_vision_tower(eng.model, _vision_config())
    # the toy's text Linears are all below the codec's bar, so the tower's
    # merger fc1 is the pack's one tensor: nothing was opened at all, where
    # the same load with images on opens exactly it
    assert list(meta["tensors"]) == ["model.visual.merger.linear_fc1"] and opened == []
    monkeypatch.setattr(pack_mod.np, "load", spy)
    _load(model, pack)
    monkeypatch.setattr(pack_mod.np, "load", real)
    assert opened == ["model.visual.merger.linear_fc1"]
    assert not any(n.startswith("model.visual.") for n, _ in eng.model.named_parameters())
    assert _greedy(eng).completion_tokens == 8
    with pytest.raises(ValueError, match=r"cannot read images on this server "
                                         r"\(disabled by DRINKME_VISION=0\)"):
        _greedy(eng, [_image()])
    assert fit.served_resident_bytes(meta, True) == meta["residentBytes"]
    assert (fit.served_resident_bytes(meta, False)
            == meta["residentBytes"] - meta["vision"]["residentBytes"])
    stock = _load_stock(model, DRINKME_VISION=0)
    assert stock.vision is None and not arms.has_vision_tower(stock.model, _vision_config())


def test_a_pack_without_a_vision_block_is_refused_at_load(packed, tmp_path, monkeypatch):
    """A pack whose meta.json has no `vision` block (built here by packing
    with the tower tables emptied: model.visual skipped) carries no tower,
    so the loader refuses it by name, with the re-pack and the text-only
    way out, rather than building a tower it cannot fill."""
    from drinkme import arms
    from drinkme.codec.pack import pack_model

    model, _, _ = packed
    bare = str(tmp_path / "bare-pack")
    with monkeypatch.context() as m:
        m.setattr(arms, "_VISION_TOWERS", {})
        m.setitem(arms._NON_TEXT_TOWERS, "qwen3_5", ("model.visual", "mtp"))
        pack_model(model, None, bare, progress=_quiet)
    meta = _meta(bare)
    assert "vision" not in meta
    assert not any(n.startswith("model.visual.") for n in meta["tensors"])
    with pytest.raises(ValueError, match=r"carries no vision tower .*re-pack it: `drinkme pack.*"
                                         r"DRINKME_VISION=0"):
        _load(model, bare)


def test_a_checkpoint_without_a_processor_config_serves_no_images(tmp_path):
    from transformers import AutoConfig

    from drinkme.serving.engines import _vision_for

    d = str(tmp_path / "noproc")
    _checkpoint(d, processor=None)
    vis, tower, why = _vision_for(AutoConfig.from_pretrained(d), d, d)
    assert vis is None and tower is None and "no image processor config" in why
    with open(os.path.join(d, "preprocessor_config.json"), "w") as f:
        json.dump({**PROCESSOR, "patch_size": 8}, f)
    with pytest.raises(ValueError, match="patch_size=8"):
        _vision_for(AutoConfig.from_pretrained(d), d, d)


def test_the_tables_agree_tower_on_and_off():
    """skeleton(cfg, vision) and ckpt_to_skel(cfg, vision) are one decision:
    with the tower, model.visual.* passes through unrenamed to a module
    that exists; without it, the tree has no tower and its keys are
    skipped. mtp.* is skipped either way, and the text keys are renamed.
    The bench's default (vision=None) agrees with `on`, not `off`: a
    served Qwen3.5 carries its tower, so its measurement does too."""
    from drinkme import arms
    from drinkme.codec.swap import eligible_linears

    cfg = _vision_config()
    on, off = arms.skeleton(cfg, vision=True), arms.skeleton(cfg, vision=False)
    bench = arms.skeleton(cfg)
    assert arms.has_vision_tower(on, cfg) and not arms.has_vision_tower(off, cfg)
    assert arms.has_vision_tower(bench, cfg)
    assert type(on.model.visual).__name__ == "Qwen3_5VisionModel"
    assert on.model.visual.merger.linear_fc1.weight.dtype == torch.bfloat16
    t_on, t_off = arms.ckpt_to_skel(cfg, vision=True), arms.ckpt_to_skel(cfg, vision=False)
    key = "model.visual.blocks.0.attn.qkv.weight"
    assert t_on(key) == key and on.get_submodule("model.visual.blocks.0.attn.qkv") is not None
    assert t_off(key) is None
    assert arms.ckpt_to_skel(cfg)(key) == key  # None (the bench's) agrees with `on`
    for t in (t_on, t_off):
        assert t("mtp.fc.weight") is None
        assert t("model.language_model.norm.weight") == "model.norm.weight"
    assert [n for n, *_ in eligible_linears(on)
            if n.startswith("model.visual.")] == ["model.visual.merger.linear_fc1"]
    assert [n for n, *_ in eligible_linears(bench)
            if n.startswith("model.visual.")] == ["model.visual.merger.linear_fc1"]
    # a text-only config of the family never gets a tower
    text = cfg.text_config
    assert arms.vision_tower_path(text) is None
    assert not arms.has_vision_tower(arms.skeleton(text, vision=True), text)


def test_a_pack_record_can_say_the_tower_was_included(packed):
    """Records should say the tower was included: the lexicon has no
    field for it yet, but raw.pack — declared `unknown` — can
    carry the pack's own `vision` block today. codec.pack.PACK_RECORD_KEYS
    names it, so pack_record_fields(meta) (what run_arms puts under
    raw.pack under --pack-dir) includes it whenever the pack does, and is
    None for a pack with no tower."""
    from drinkme.codec.pack import pack_record_fields

    _model, pack, _feats = packed
    meta = _meta(pack)
    assert meta["vision"]["residentBytes"] > 0
    assert pack_record_fields(meta)["vision"] == meta["vision"]
    assert pack_record_fields({"formatVersion": 1})["vision"] is None


def test_a_bench_record_says_the_tower_was_included(packed, monkeypatch):
    """raw.vision_tower_included, through run_arms, on the tree the bench
    builds for Qwen3.5 (MiMo-9B, Qwen3.8-27B): the text class, whose
    model.config is the text config, with the tower attached at
    model.visual. Judged against that model.config the record said False
    while the arms held the tower (MiMo-9B on gfx1151: 924 MB
    more stock and 731 MB more compressed than the text-only record);
    judged against the snapshot's config.json, which the tree was built
    from, it says True, and False for a tree built without the tower."""
    from transformers import AutoConfig

    from drinkme import arms
    from drinkme.serving import deltanet

    from test_bench_route_alike import _run, _stub_device, _toy_loaders

    model_dir, _pack, _feats = packed
    cfg = AutoConfig.from_pretrained(model_dir)
    tree = arms.skeleton(cfg)  # the bench's default: the tower
    text_only = arms.skeleton(cfg, vision=False)
    assert tree.config.model_type == "qwen3_5_text"
    assert arms.has_vision_tower(tree, cfg) and not arms.has_vision_tower(tree, tree.config)
    monkeypatch.setenv(deltanet.ENV, "torch")
    _stub_device(monkeypatch)  # stubs torch.tensor: every tree is built above
    monkeypatch.setattr(arms, "resolve_source", lambda m, r, p: (model_dir, "a" * 40))
    _toy_loaders(monkeypatch, tree)
    assert _run()["vision_tower_included"] is True
    _toy_loaders(monkeypatch, text_only)
    assert _run()["vision_tower_included"] is False
