"""Muse-Glimmer image input, CPU, float32, no downloads.

PREPROCESSING. transformers' MuseGlimmerImageProcessor has one backend,
torchvision's, and no drinkme environment has torchvision. The resize is
the only step torchvision computes itself (torch's antialiased LANCZOS,
F.interpolate), so drinkme calls that same torch kernel and does the rest
in numpy. Pinned three ways:
- against a torchvision run: bench/glimmer_preprocess_pin.py, in a
  throwaway venv with torchvision 0.27.1 and transformers 5.15.1, printed
  the reference's digest for each of its synthetic cases (GOLDEN below; an
  RGB array per case from a numpy seed, so no decoder version moves them).
  That run also found Pillow's LANCZOS off the reference by a level on 94
  of its 95 cases, which is why the resize is torch's (the negative
  control below);
- live against the installed transformers where torchvision is not needed:
  its smart_resize and patchify, compiled from the module's own source
  (the module imports torchvision at import time);
- the fused normalize, against the torch arithmetic the backend runs.

THE TEXT SIDE on a toy Muse-Glimmer: the reference is transformers' own
MuseGlimmerForConditionalGeneration (forward(input_ids, pixel_values,
image_grid_thw)), built tiny and random with the 30B's layer pattern (three
sliding-window layers per full NoPE one) and a ViT whose window and full
layers both see several windows. The served tree is a copy of the same
class. Two sliding windows: 16, shorter than an image's run, and 512,
never full. Pinned: prefill logits with images, whole and chunked, and the
greedy continuation through generate() with speculation off and n-gram
are the reference's; the text side is causal with 1-D positions (a
bidirectional image mask is NOT the reference); a text request makes the
calls it always made.

THE TOWER IN THE SERVED TREE: a toy checkpoint (bf16, with a
processor_config.json like the 30B's) is packed and loaded both arms: the
pack's `vision` block names the three subtrees, the served tower (its patch
embedding packed too) is transformers' get_image_features bit for bit
offline, DRINKME_VISION=0 prunes it, a pack cut before this step serves
text, and the bench's glimmer tree stays the pruned one it has always
measured.

THE REAL SNAPSHOT and THE REAL TOKENIZERS, read-only from the local HF
cache and skipped when it lacks them: every tower tensor of
Muse-Glimmer-30B has a same-shaped home and the packable ones are counted;
every modality marker each vision architecture's tokenizer defines is in
its reserved set (the injection guard's literals).
"""

from __future__ import annotations

import ast
import copy
import dataclasses
import importlib.util
import itertools
import json
import math
import os
import sys

import numpy as np
import pytest
import torch

pytest.importorskip("transformers.models.muse_glimmer",
                    reason="muse_glimmer arrived in transformers 5.15.0")

from drinkme.serving import mtp, prefill, vision  # noqa: E402
from drinkme.serving.engine import GenerationRequest, SampleParams  # noqa: E402
from drinkme.serving.engines import HFEngine  # noqa: E402
from drinkme.serving.image_prompt import ImagePrompt  # noqa: E402

from test_serving_gemma_vision import (OuterCalls, TowerCalls, ask,  # noqa: E402
                                       gate_negative_controls, png, rendered, user)
from test_serving_prefill_bound import env  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "bench"))
import glimmer_preprocess_pin as pin  # noqa: E402

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


# ------------------------------------------------------ PREPROCESSING --

# The reference's (grid, tokens, sha256 of pixel_values then the grid) for
# each of bench/glimmer_preprocess_pin.py's synthetic cases, printed by its
# `reference` step: torch 2.12.1+cpu, torchvision 0.27.1+cpu, transformers
# 5.15.1, the 30B's processor config.
GOLDEN = {
    "1080p-1920x1080": ((1, 78, 138), 2691,
                        "36cce2e467ad7568cd0abc60dd0c3df970216d8eee5c37c8308e20b5b6098d28"),
    "budget-40-777x333": ((1, 8, 18), 36,
                          "0fd62961af964fb11a7ef31c6e0e46fc030d6fb336bcbb92ef3a5bef1c9e8916"),
    "cap-2560x1440": ((1, 96, 170), 4080,
                      "6de94c3186e1351ef771742dd26321abe7db0fb843dd63dcadf45da19d376e70"),
    "grid-560x392": ((1, 28, 40), 280,
                     "920d3b0e92dd1d18210da996408cc56626d6b43cdcf54aae22c59cf9d063be83"),
    "low-1280x720": ((1, 26, 48), 312,
                     "a5177f4be8e45fe0176776b02367954299937ee54b17498cea4593ef6f40562c"),
    "odd-641x479": ((1, 34, 46), 391,
                    "d64ab39ebe6800d6185e889b7625a1821832f97f68e4e4a20e377fdac61b3122"),
    "one-token-90x70": ((1, 2, 2), 1,
                        "8f76afa661c71bbd8ffc0e7dbae444280ad01d03e6b8e480d14bce8a8f08f511"),
    "small-37x53": ((1, 2, 2), 1,
                    "376000aceee332611b771adcf9d6add6eb54deb55363fe7084cbd20f46dfd8a2"),
    "square-1000": ((1, 72, 72), 1296,
                    "6eb27e4e19ac31e2880f5d0360eb1a40d3491a05c57226ffac30ad56fb7e8e4b"),
    "tall-3x1500": ((1, 108, 2), 54,
                    "161b1f05711c3f697cc900cb63e135a826ad1e309c3fea59645600d17b2de1aa"),
    "tie-100x100": ((1, 8, 8), 16,
                    "94c71b4e8b5486beac2bc671a521782dcc44e26d7182511933172a50c87f891a"),
    "tiny-5x7": ((1, 2, 2), 1,
                 "0c3fc61d5090e0b5a84f10ada1802dac006026bd39937170ed3156cea3ff305c"),
    "wide-3000x3": ((1, 2, 216), 108,
                    "95337115e338a5e046cc91f88993c28ad4621e7ae5128a11e4a647c7446f6395"),
}
CASES = {name: (w, h, seed, budget) for name, w, h, seed, budget in pin.SYNTHETIC}
SNAPSHOT = vision.preprocessor_for("muse_glimmer", dict(pin.SNAPSHOT_CONFIG))


def _prepared(name: str, pillow: bool = False):
    w, h, seed, budget = CASES[name]
    return pin.ours(SNAPSHOT, pin.synthetic(w, h, seed), budget, pillow=pillow)


def test_every_synthetic_case_has_its_golden():
    assert sorted(CASES) == sorted(GOLDEN)


@pytest.mark.parametrize("name", sorted(GOLDEN))
def test_preprocessing_is_the_torchvision_processors_bit_for_bit(name):
    """Every synthetic case: the grid, the token count and the
    pixel_values bytes are what transformers' processor computed through
    torchvision, and a request's header-only count (a PNG of the same
    pixels) equals the prepared plan."""
    grid, tokens, sha = GOLDEN[name]
    pv, got_grid, got_tokens = _prepared(name)
    assert (got_grid, got_tokens) == (grid, tokens)
    assert pv.dtype == np.float32 and pv.shape == (grid[1] * grid[2], 2 * 3 * 14 * 14)
    assert pin.digest(pv, got_grid) == sha
    w, h, seed, budget = CASES[name]
    enc = vision.parse_base64(png_of(pin.synthetic(w, h, seed)), "image/png", where="t")
    v = vision.Vision(SNAPSHOT, budget)
    img = v.prepare(enc)
    assert v.count(enc) == img.plan and img.digest == sha


def png_of(im) -> str:
    import base64
    import io

    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def test_a_pillow_lanczos_resize_is_not_the_reference():
    """The negative control, and why the resize is torch's: Pillow's
    LANCZOS misses the reference on every case that resizes; the one case
    already on the grid (no resize) is the reference either way."""
    for name, (grid, _tokens, sha) in GOLDEN.items():
        pv, got_grid, _ = _prepared(name, pillow=True)
        assert (pin.digest(pv, got_grid) == sha) == (name == "grid-560x392"), name


def _reference_function(name: str, owner: str | None = None):
    """A function from transformers' image_processing_muse_glimmer, compiled
    from the module's own source: importing the module imports torchvision,
    which no drinkme environment has, and smart_resize and patchify need
    only math, itertools and torch."""
    spec = importlib.util.find_spec("transformers.models.muse_glimmer")
    path = os.path.join(os.path.dirname(spec.origin), "image_processing_muse_glimmer.py")
    with open(path) as f:
        tree = ast.parse(f.read())
    body = tree.body
    if owner is not None:
        body = next(n for n in body if isinstance(n, ast.ClassDef) and n.name == owner).body
    node = next(n for n in body if isinstance(n, ast.FunctionDef) and n.name == name)
    ns = {"math": math, "itertools": itertools, "torch": torch}
    exec(compile(ast.Module(body=[node], type_ignores=[]), path, "exec"), ns)
    return ns[name]


def test_smart_resize_is_the_references_line_for_line():
    """drinkme's port against transformers' own function over a sweep of
    sizes (tiny, square, both extremes, the rounding in between) and
    budgets (the checkpoint's, the server cap's, detail low, one token):
    the same grid every time, ties included."""
    ref = _reference_function("smart_resize")
    sides = [1, 2, 3, 5, 13, 27, 28, 29, 55, 56, 57, 99, 100, 101, 333, 480, 641, 720, 1000,
             1080, 1440, 1500, 1800, 2160, 3000, 3840, 9999]
    for budget in (4096, 334, 36, 1):
        for h in sides:
            for w in sides:
                assert vision.glimmer_smart_resize(h, w, 28, budget) == \
                    ref(height=h, width=w, patch_size=28, max_tokens=budget), (h, w, budget)


# (width, height) -> resized (width, height), tokens, at the server's default
# cap: the checkpoint's own 4,096-token budget is the one that bites, small
# images are upscaled, and detail "low" (512 x 512) is 334 tokens at most
TOKENS = [((640, 480), (644, 476), 391), ((1280, 720), (1288, 728), 1196),
          ((1920, 1080), (1932, 1092), 2691), ((2560, 1440), (2380, 1344), 4080),
          ((3840, 2160), (2380, 1344), 4080), ((2880, 1800), (2240, 1400), 4000),
          ((512, 512), (532, 532), 361), ((1024, 768), (1008, 756), 972)]


@pytest.mark.parametrize("src,size,tokens", TOKENS)
def test_the_token_table(src, size, tokens):
    w, h = src
    got = SNAPSHOT.plan(h, w, vision.DEFAULT_MAX_PIXELS, where="t")
    assert got == ((size[1], size[0]), (1, size[1] // 14, size[0] // 14), tokens)
    low = SNAPSHOT.plan(h, w, vision.LOW_DETAIL_PIXELS, where="t")
    assert low[2] <= 334 and low[0][0] % 28 == 0 and low[0][1] % 28 == 0


def test_the_budget_lowers_max_image_tokens_and_never_raises_it():
    # the server cap covers 4,702 cells: the checkpoint's own budget wins
    assert SNAPSHOT.max_tokens(vision.DEFAULT_MAX_PIXELS) == 4096
    assert SNAPSHOT.max_tokens(vision.LOW_DETAIL_PIXELS) == 334
    assert SNAPSHOT.max_tokens(1 << 40) == 4096
    assert SNAPSHOT.max_tokens(1) == 1
    v = vision.Vision(SNAPSHOT)
    assert v.prompt_tokens(v.count(vision.parse_base64(
        png_of(pin.synthetic(56, 56, 1)), "image/png", where="t"))) == 4 + 2  # the two markers


def test_an_aspect_ratio_whose_grid_overruns_the_budget_is_refused_by_name():
    """The reference falls back to its rounded ideal grid when no candidate
    fits, which can exceed max_image_tokens; drinkme refuses that image
    rather than serve it past the cap."""
    ref = _reference_function("smart_resize")
    h, w = ref(height=1, width=200_000, patch_size=28, max_tokens=4096)
    assert (h // 28) * (w // 28) > 4096
    with pytest.raises(vision.ImageError) as e:
        SNAPSHOT.plan(1, 200_000, vision.DEFAULT_MAX_PIXELS, where="messages[0].content[1]")
    assert e.value.code == "image_aspect_ratio" and "messages[0].content[1]" in str(e.value)
    with pytest.raises(vision.ImageError):
        SNAPSHOT.plan(10, 1000, 16 * 784, where="t")


def test_the_normalize_table_is_the_backends_fused_arithmetic():
    """torchvision's backend fuses rescale into normalize: mean and std
    times 1 / rescale_factor as float32 tensors, then (x - mean) / std on
    float32 pixels. The 256-entry table gives those bits."""
    cfg = pin.SNAPSHOT_CONFIG
    mean = torch.tensor(cfg["image_mean"]) * (1.0 / cfg["rescale_factor"])
    std = torch.tensor(cfg["image_std"]) * (1.0 / cfg["rescale_factor"])
    x = torch.arange(256, dtype=torch.uint8).to(torch.float32)[None, :].expand(3, 256)
    want = x.sub(mean.view(-1, 1)).div_(std.view(-1, 1))
    assert torch.equal(torch.from_numpy(SNAPSHOT._table()), want)


def test_patchify_is_the_references_layout():
    """Patches in grid row-major order, each [T, C, P, P] with the frame
    duplicated along T: the reference's own patchify over the normalized
    image equals drinkme's pixel_values for an image already on the grid."""
    patchify = _reference_function("patchify", "MuseGlimmerImageProcessor")
    rgb = pin.synthetic(84, 56, 7)
    a = torch.from_numpy(np.array(rgb)).permute(2, 0, 1)[None].to(torch.float32)
    x = torch.from_numpy(SNAPSHOT._table())[torch.arange(3)[:, None, None],
                                             a[0].long()][None]
    want, gh, gw = patchify(None, x, patch_size=14, temporal_patch_size=2)
    got = SNAPSHOT.pixel_values(rgb, (56, 84), vision.DEFAULT_MAX_PIXELS)
    assert (gh, gw) == (4, 6) and torch.equal(torch.from_numpy(got), want[0])


def test_the_registry_reads_the_processor_config_and_refuses_what_it_does_not_implement(tmp_path):
    """No processor config means no images (the 30B ships one); the
    snapshot's config gives the class defaults; what is not implemented is
    refused by name, not approximated."""
    assert "muse_glimmer" in vision.architectures()
    assert vision.load("muse_glimmer", str(tmp_path)) is None
    (tmp_path / "processor_config.json").write_text(json.dumps(
        {"image_processor": pin.SNAPSHOT_CONFIG, "processor_class": "MuseGlimmerProcessor"}))
    pre = vision.load("muse_glimmer", str(tmp_path)).preprocessor
    assert pre == vision.MuseGlimmerPreprocessor("muse_glimmer")  # the snapshot IS the defaults
    assert (pre.patch_size, pre.merge_size, pre.max_image_tokens, pre.wrap) == (14, 2, 4096, 2)
    assert pre.unavailable() is None  # this torch has the LANCZOS kernel
    for bad, match in (({"resample": 3}, "resample=3"), ({"do_normalize": False}, "do_normalize"),
                       ({"image_processor_type": "Qwen2VLImageProcessor"}, "Qwen2VL")):
        with pytest.raises(ValueError, match=match):
            vision.preprocessor_for("muse_glimmer", dict(pin.SNAPSHOT_CONFIG, **bad))


def test_a_torch_without_lanczos_gets_no_images_by_name(monkeypatch, tmp_path):
    """On a torch older than the LANCZOS kernel the engine serves text and
    names why images are refused, instead of failing a request."""
    from drinkme.serving.engines import _vision_for

    cfg = config()
    (tmp_path / "processor_config.json").write_text(json.dumps(
        {"image_processor": dict(pin.SNAPSHOT_CONFIG, patch_size=4, max_image_tokens=64)}))

    def old_torch(*a, **k):
        raise NotImplementedError("mode lanczos")

    monkeypatch.setattr(vision, "lanczos_resize", old_torch)
    vis, tower, why = _vision_for(cfg, str(tmp_path), "toy", tokenizer=lambda: _tokenizer())
    assert (vis, tower) == (None, None) and "no LANCZOS resize" in why and "2.12" in why


# --------------------------------------------------------- THE TOY MODEL --

WORDS = ["hello", "world", "the", "quick", "brown", "fox", "what", "is", "in", "this",
         "picture", "and", "user", "assistant", ":"]
SPECIAL = ["<|patch|>", "<|image_start|>", "<|image_end|>", "<|image|>", "<|video|>",
           "<|vid_start|>", "<|vid_end|>", "<|vid_frame_separator|>"]


def _vocab() -> dict:
    vocab = {"<unk>": 0, "<pad>": 1, "</s>": 2}
    for w in WORDS + SPECIAL:
        vocab[w] = len(vocab)
    while len(vocab) < 96:
        vocab[f"tok{len(vocab)}"] = len(vocab)
    return vocab


VOCAB = _vocab()
SOFT = VOCAB["<|patch|>"]
TEMPLATE = (
    "{% for m in messages %}{{ m['role'] }} : {% if m['content'] is string %}"
    "{{ m['content'] }}{% else %}{% for p in m['content'] %}{% if p['type'] == 'image' %}"
    "<|patch|>{% else %}{{ p['text'] }} {% endif %}"
    "{% endfor %}{% endif %} {% endfor %}{% if add_generation_prompt %}assistant :{% endif %}")


def _tokenizer():
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast

    backend = Tokenizer(WordLevel(VOCAB, unk_token="<unk>"))
    backend.pre_tokenizer = Whitespace()
    tok = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="<unk>",
                                  pad_token="<pad>", additional_special_tokens=SPECIAL)
    tok.chat_template = TEMPLATE
    return tok


def config(window: int = 16, *, text_hidden: int = 64, text_mlp: int = 128, text_layers: int = 8,
           text_heads: int = 2, head_dim: int = 32, patch: int = 4, vision_width: int = 32,
           vision_mlp: int = 64, vision_layers: int = 4, vision_heads: int = 2,
           projector: int = 48):
    """A tiny Muse-Glimmer with the 30B's text geometry (three sliding
    layers per full NoPE one, the output multiplier and softcap, the
    scaleless qk-norm) and a small ViT: windows of 4 x 4 patches, every
    fourth layer and the last full."""
    from transformers.models.muse_glimmer import MuseGlimmerConfig

    return MuseGlimmerConfig(
        text_config=dict(
            vocab_size=len(VOCAB), hidden_size=text_hidden, intermediate_size=text_mlp,
            num_hidden_layers=text_layers, num_attention_heads=text_heads,
            num_key_value_heads=max(1, text_heads // 4), head_dim=head_dim,
            max_position_embeddings=1024, sliding_window=window,
            # no EOS: a random argmax would end the comparison runs early
            bos_token_id=None, eos_token_id=None, pad_token_id=VOCAB["<pad>"]),
        vision_config=dict(
            hidden_size=vision_width, intermediate_size=vision_mlp,
            num_hidden_layers=vision_layers, num_attention_heads=vision_heads, patch_size=patch,
            pos_emb_height=4, pos_emb_width=4, max_position_embeddings=16,
            rope_parameters={"rope_theta": 10000.0, "rope_type": "default"}),
        out_hidden_size=vision_width * 4, projector_hidden_size=projector,
        image_token_id=SOFT, video_token_id=VOCAB["<|video|>"])


# the toy's preprocessing: Glimmer's rule at a 4-pixel patch and 64 tokens
VIS = vision.Vision(vision.MuseGlimmerPreprocessor("muse_glimmer", patch_size=4,
                                                   max_image_tokens=64), max_pixels=1 << 30)
WINDOWS = (16, 512)


def _tower(cfg) -> vision.Tower:
    from drinkme import arms

    return vision.tower_for("muse_glimmer", cfg, arms.vision_tower_paths(cfg), _tokenizer())


def image(w: int, h: int, seed: int, vis=VIS):
    """A fresh PreparedImage (the engine releases its pixels once read)."""
    return vis.prepare(vision.parse_base64(png(w, h, seed), "image/png", where="test"))


def two_images():
    # 32 x 48 px: 24 tokens on a 8 x 12 patch grid (six windows); 40 x 24:
    # 15 tokens on 10 x 6 (partial windows)
    return [image(48, 32, 1), image(24, 40, 2)]


@pytest.fixture(scope="module", params=WINDOWS, ids=lambda w: f"window{w}")
def toy(request):
    """(reference, served tree, tokenizer, tower) at one sliding window."""
    from transformers.models.muse_glimmer import MuseGlimmerForConditionalGeneration

    cfg = config(request.param)
    torch.manual_seed(0)
    ref = MuseGlimmerForConditionalGeneration(cfg).eval().float()
    for p in ref.parameters():
        p.requires_grad_(False)
    return ref, copy.deepcopy(ref), _tokenizer(), _tower(cfg)


def engine(toy, *, slots=1, chunk=0, **kw):
    _ref, model, tok, tower = toy
    with env(DRINKME_PREFILL_CHUNK=chunk, DRINKME_PREFIX_SLOTS=slots, DRINKME_SLOT_DIR="off"):
        return HFEngine(model, tok, model_id="toy", arm="test", meta={}, ctx=320,
                        vision=VIS, tower=tower, **kw)


def _batch(images):
    """The processor's batch: every image's pixel_values rows, and the grids."""
    return (torch.cat([torch.tensor(np.array(i.pixel_values)) for i in images]),
            torch.tensor([list(i.grid_thw) for i in images]))


def ref_logits(ref, ids, images):
    """MuseGlimmerForConditionalGeneration over the whole prompt, no cache."""
    pv, grid = _batch(images)
    with torch.inference_mode():
        return ref(input_ids=torch.tensor([ids]), pixel_values=pv, image_grid_thw=grid,
                   use_cache=False).logits[0]


def ref_greedy(ref, ids, images, n):
    """generate()'s greedy: the prompt with its images once, then one token
    per step over the cache (a generated placeholder id is then embedded as
    the wrapper embeds text, not counted as an image slot)."""
    pv, grid = _batch(images)
    out = []
    with torch.inference_mode():
        res = ref(input_ids=torch.tensor([ids]), pixel_values=pv, image_grid_thw=grid,
                  use_cache=True, logits_to_keep=1)
        for _ in range(n):
            out.append(int(res.logits[0, -1].argmax()))
            res = ref(input_ids=torch.tensor([[out[-1]]]), past_key_values=res.past_key_values,
                      use_cache=True)
    return out


MSGS = [user("what is in", None, "the quick brown", None, "hello world")]


# ----------------------------------------------------------- THE SPLICE --

def test_the_placeholder_expands_between_the_markers(toy):
    """<|patch|> -> <|image_start|> + tokens x <|patch|> + <|image_end|>,
    the string MuseGlimmerProcessor.replace_image_token builds; the run is
    the <|patch|> rows alone, keyed by the image; no position_ids."""
    from types import SimpleNamespace

    from transformers.models.muse_glimmer.processing_muse_glimmer import MuseGlimmerProcessor

    _ref, model, tok, tower = toy
    imgs = two_images()
    r = rendered(tok, MSGS)
    ip = ImagePrompt(r, imgs, tower, model)
    assert len(ip.ids) == len(r) + VIS.expansion(imgs)
    proc = SimpleNamespace(image_processor=SimpleNamespace(merge_size=2),
                           image_start_token="<|image_start|>", image_end_token="<|image_end|>",
                           image_token="<|patch|>")
    for (s, e, img), key in zip(ip.runs, (imgs[0].prefix_key, imgs[1].prefix_key)):
        want = MuseGlimmerProcessor.replace_image_token(
            proc, {"image_grid_thw": torch.tensor([list(img.grid_thw)])}, 0)
        assert tok.decode(ip.ids[s - 1:e + 1]).replace(" ", "") == want
        assert e - s == img.tokens and set(ip.ids[s:e]) == {SOFT}
        assert set(ip.key_ids[s:e]) == {key}
        assert (ip.key_ids[s - 1], ip.key_ids[e]) == (VOCAB["<|image_start|>"],
                                                       VOCAB["<|image_end|>"])
    assert ip.pos4 is None and not ip.rope
    assert set(ip.inputs(model.get_decoder(), 0, ip.n, "cpu")) == {"inputs_embeds"}


@pytest.mark.parametrize("chunk", [0, 7, 70])
def test_prefill_logits_are_the_references(toy, chunk):
    """Every prompt row's logits (lm_head, the multiplier and the softcap
    over each span's hidden states) and the last row's (the serial
    prefill's wrapper call), whole and chunked, against
    MuseGlimmerForConditionalGeneration over the same weights. The text
    side is causal (Tower.bidirectional False), so at 7 both image runs (24
    and 15 tokens) are longer than the chunk and cut into chunk-sized spans
    like text (ImagePrompt.whole), and no forward is longer than the chunk;
    at 70 one span holds the whole prompt but for its tail, and no span
    cuts a run."""
    ref, model, tok, tower = toy
    ip = ImagePrompt(rendered(tok, MSGS), two_images(), tower, model)
    want = ref_logits(ref, ip.ids, two_images())
    hs = []
    with torch.inference_mode():
        cache = engine(toy)._cache(256)
        prefill.run(model, ip.ids, 0, cache, "cpu", chunk, hidden=True,
                    on_hidden=lambda h, a, b: hs.append((h, a, b)), image=ip)
        every = mtp.head_logits(model, torch.cat([h for h, _, _ in hs], 1))[0]
        serial = ImagePrompt(rendered(tok, MSGS), two_images(), tower, model)
        last = prefill.run(model, serial.ids, 0, engine(toy)._cache(256), "cpu", chunk,
                           image=serial)
    torch.testing.assert_close(every, want, atol=2e-5, rtol=1e-5)
    torch.testing.assert_close(last, want[-1], atol=2e-5, rtol=1e-5)
    runs = [(s, e) for s, e, _ in ip.runs]
    spans = [(a, b) for _, a, b in hs]
    assert spans == prefill.spans(0, ip.n, chunk, ip.whole(0, chunk))
    for s, e in runs:
        cut = any(s < a < e or s < b < e for a, b in spans)
        assert cut == (0 < chunk < e - s)
    assert not chunk or all(b - a <= chunk for a, b in spans)


def test_the_text_side_is_causal_with_1d_positions(toy):
    """Images are embedding substitution and nothing else: the tower is
    causal and 1-D (no M-RoPE rows, no mask of its own), the full layers
    are NoPE, and the negative control, the same prompt with gemma-4's
    bidirectional image mask on the sliding layers, is NOT the reference."""
    ref, model, tok, tower = toy
    text = model.config.get_text_config()
    assert not tower.bidirectional and not tower.mrope
    assert [t for t, th in zip(text.layer_types, text.layer_rope_theta) if th == 0] == \
        ["full_attention"] * 2
    ids = ImagePrompt(rendered(tok, MSGS), two_images(), tower, model).ids
    want = ref_logits(ref, ids, two_images())[-1]
    bidi = ImagePrompt(rendered(tok, MSGS), two_images(),
                       dataclasses.replace(tower, bidirectional=True), model)
    with torch.inference_mode():
        got = prefill.run(model, bidi.ids, 0, engine(toy)._cache(256), "cpu", 0, image=bidi)
    assert (got - want).abs().max() > 1e-3


@pytest.mark.parametrize("spec,chunk", [("off", 0), ("off", 7), ("ngram", 0), ("ngram", 7)])
def test_the_engines_greedy_is_the_references(toy, spec, chunk):
    """generate() end to end, serial and with the n-gram lookup (Glimmer has
    no draft head, so n-gram is its default), whole and chunked: the
    reference's greedy token for token (the slot holds every written id),
    and the count routes' prompt count is generate()'s. At window 16 the
    toy generates <|patch|> itself; the verify forwards embed it as the
    wrapper does (mtp.trunk_ids), or n-gram leaves the reference there."""
    ref, model, tok, tower = toy
    n = 10
    ids = ImagePrompt(rendered(tok, MSGS), two_images(), tower, model).ids
    want = ref_greedy(ref, ids, two_images(), n)
    if model.config.get_text_config().sliding_window == 16:
        assert SOFT in want  # the case trunk_ids is for
    eng = engine(toy, chunk=chunk)
    got = ask(eng, MSGS, two_images(), spec, n)
    assert got.prompt_tokens == len(ids) == eng.count_tokens(
        GenerationRequest(MSGS, SampleParams(), images=tuple(i.plan for i in two_images())))
    slot = eng._slots[0].ids
    assert slot[len(ids):] == want[:n - 1]
    assert got.text.split() == tok.decode(want, skip_special_tokens=True).split()


def test_the_dialects_count_glimmers_markers_and_refuse_them_in_text(toy):
    """Over HTTP: /tokenize's count of a turn with an image is the prompt
    generate() consumed (the two markers around each image included), and
    text spelling any modality marker the Glimmer tokenizer defines is
    refused by name."""
    from drinkme.serving.http import start_server

    from test_serving_http import post

    srv = start_server(engine(toy), port=0)
    try:
        port = srv.server_address[1]
        url = "data:image/png;base64," + png(48, 32, 7)
        msgs = [{"role": "user", "content": [{"type": "text", "text": "what is in"},
                                             {"type": "image_url", "image_url": {"url": url}}]}]
        r, body = post(port, {"messages": msgs}, "/tokenize")
        assert r.status == 200
        count = json.loads(body)["count"]
        r, body = post(port, {"model": "toy", "messages": msgs, "max_tokens": 2,
                              "temperature": 0})
        assert r.status == 200 and json.loads(body)["usage"]["prompt_tokens"] == count
        for marker in SPECIAL:
            r, body = post(port, {"model": "toy", "max_tokens": 2, "messages": [
                {"role": "user", "content": f"hello {marker} world"}]})
            assert r.status == 400 and marker.encode() in body and b"image_injection" in body
    finally:
        srv.shutdown()


def test_the_vits_attention_is_bounded(toy, monkeypatch):
    """Every ViT attention module, window and full, goes through
    vision.bounded_attention, and no dispatch carries more than WORK_V
    query-key pairs: query blocks alone are the stock call's bits, and with
    the keys split too the log-sum-exp combine is within fp32 rounding."""
    from transformers.integrations import sdpa_attention as sd

    ref, model, _tok, tower = toy
    served = copy.deepcopy(model)
    img = image(48, 32, 3)
    pv, grid = _batch([img])
    with torch.inference_mode():
        want = ref.model.get_image_features(pv, grid).pooler_output[0]
    seen = []
    real = sd.sdpa_attention_forward

    def rec(module, query, key, value, m, **kw):
        seen.append(query.shape[-2] * key.shape[-2])
        return real(module, query, key, value, m, **kw)

    monkeypatch.setattr(sd, "sdpa_attention_forward", rec)
    assert vision.bound(tower, served) == 4
    full = 96 * 96  # the one full layer: 8 x 12 patches, one call
    for work, rows, exact in ((96 * 32, 8, True), (96 * 4, 8, False)):
        monkeypatch.setattr(vision, "WORK_V", work)
        monkeypatch.setattr(vision, "MIN_ROWS", rows)
        seen.clear()
        with torch.inference_mode():
            got = tower.features(served, image(48, 32, 3))
        assert seen and max(seen) <= work < full
        if exact:
            assert torch.equal(got, want)
        else:
            torch.testing.assert_close(got, want, atol=2e-6, rtol=1e-6)


# ----------------------------------------------------- PREFIX CACHE --

def test_the_same_image_resent_reuses_its_kv_and_skips_the_tower(toy):
    """A conversation that carries its image forward extends its slot past
    the image without running the tower again, and answers what a cold
    engine answers. A different image of the same size never shares the
    slot. (Image 25: at both windows the toy's first reply is words its
    word-level tokenizer reads back, so the resent history is the slot's
    ids; a reply with a special in it would not be, a cold prefill.)"""
    _ref, model, tok, _tower = toy
    eng = engine(toy)
    hist = [user("what is in this picture", None)]
    first = ask(eng, hist, [image(48, 32, 25)], n=4)
    hist = hist + [{"role": "assistant", "content": first.text}, user("and the fox")]
    with TowerCalls(model) as calls:
        warm = ask(eng, hist, [image(48, 32, 25)])
    assert calls.n == 0 and warm.cached_tokens > 0
    with TowerCalls(model) as calls:
        cold = ask(engine(toy), hist, [image(48, 32, 25)])
    assert calls.n == 1 and warm.text == cold.text
    other = ask(eng, hist, [image(48, 32, 22)])
    # at most up to the image run's first row: the text before it and the
    # image's opening marker, which both images share (a context checkpoint
    # moved back to the run's start, serving/ctx_checkpoints.py)
    assert other.cached_tokens <= rendered(tok, hist).index(SOFT) + 1
    assert other.text == ask(engine(toy), hist, [image(48, 32, 22)]).text


# ----------------------------------------------- TEXT IS UNCHANGED --

@pytest.mark.parametrize("spec,chunk", [("off", 0), ("off", 7), ("ngram", 7)])
def test_a_text_request_makes_the_calls_it_always_made(toy, spec, chunk, monkeypatch):
    """No position_ids, no inputs_embeds, no attention_mask: the ids as the
    one positional input on every forward of a text generation on the
    image-capable Glimmer engine, and no ImagePrompt is ever built."""
    import drinkme.serving.image_prompt as ip_mod

    from drinkme.serving.engine import complete

    _ref, model, _tok, _tower = toy

    def boom(*a, **k):
        raise AssertionError("a text request built an ImagePrompt")

    monkeypatch.setattr(ip_mod, "ImagePrompt", boom)
    eng = engine(toy, chunk=chunk)
    msgs = [{"role": "user", "content": "the quick brown fox what is this picture " * 3}]
    with OuterCalls(model) as rec:
        with env(DRINKME_SPEC=spec):
            out = complete(eng, GenerationRequest(msgs, SampleParams(temperature=0.0,
                                                                     max_tokens=8)))
    assert out.completion_tokens == 8 and rec.calls
    allowed = {"past_key_values", "use_cache", "cache_position", "logits_to_keep"}
    for _cls, kind, kw in rec.calls:
        assert kind == "Tensor" and set(kw) <= allowed, (kind, kw)


# -------------------------------------------- THE TOWER IN THE PACK --
#
# A toy checkpoint saved in bf16 the way transformers saves
# MuseGlimmerForConditionalGeneration, with a processor_config.json like
# the 30B's (patch 14, LANCZOS, a 16-token budget). The ViT is 1024 wide
# with two layers (one window, one full), so every Linear of the tower
# clears the codec's bar (1024 x 1024 and up): the patch embedding, each
# layer's q/k/v/proj and fc1/fc2, the adapter's fc1/fc2 and the
# projection, 16 packed; the norms, the biases and the position table ride
# raw.

PACK_MSGS = [user("what is in this picture", None)]
TOWER = ("model.vision_tower", "model.vision_adapter", "model.vision_projection")


def in_tower(name: str) -> bool:
    """A tensor or module of the tower: model.vision_projection is one
    Linear, so its pack tensor is named the path itself (codec/pack.under)."""
    from drinkme.codec.pack import under

    return under(name, TOWER)


PACKED_TOWER = 1 + 2 * 6 + 3


def _quiet(*_a, **_k):
    pass


def _pack_config():
    return config(64, text_hidden=1024, text_mlp=1024, text_layers=2, text_heads=8,
                  head_dim=128, patch=14, vision_width=1024, vision_mlp=1024, vision_layers=2,
                  vision_heads=8, projector=1024)


def _pack_vision():
    return vision.Vision(vision.preprocessor_for(
        "muse_glimmer", dict(pin.SNAPSHOT_CONFIG, max_image_tokens=16)))


def _pack_image():
    return image(112, 84, 5, vis=_pack_vision())


def _checkpoint(path: str) -> None:
    from transformers.models.muse_glimmer import MuseGlimmerForConditionalGeneration

    torch.manual_seed(0)
    model = MuseGlimmerForConditionalGeneration(_pack_config())
    model.to(torch.bfloat16).eval().save_pretrained(path, safe_serialization=True)
    _tokenizer().save_pretrained(path)
    with open(os.path.join(path, "processor_config.json"), "w") as f:
        json.dump({"image_processor": dict(pin.SNAPSHOT_CONFIG, max_image_tokens=16),
                   "processor_class": "MuseGlimmerProcessor"}, f)


def _meta(pack):
    with open(os.path.join(pack, "meta.json")) as f:
        return json.load(f)


def _headers(path):
    from drinkme.codec.pack import _shard_headers

    return _shard_headers(path)


@pytest.fixture(scope="module")
def packed(tmp_path_factory):
    """(checkpoint dir, pack dir, the reference's features for _pack_image),
    packed once; the reference is read before any test deletes anything."""
    from transformers.models.muse_glimmer import MuseGlimmerForConditionalGeneration

    from drinkme.codec.pack import pack_model

    root = tmp_path_factory.mktemp("glimmer-vision-pack")
    model, pack = str(root / "model"), str(root / "pack")
    _checkpoint(model)
    pack_model(model, None, pack, progress=_quiet)
    ref = MuseGlimmerForConditionalGeneration.from_pretrained(model, dtype=torch.bfloat16).eval()
    pv, grid = _batch([_pack_image()])
    with torch.inference_mode():
        feats = ref.model.get_image_features(pv, grid).pooler_output[0]
    return model, pack, feats


def _env(**kw):
    return env(DRINKME_PREFIX_SLOTS=0, DRINKME_SLOT_DIR="off", **kw)


def _load(model, pack, **kw):
    from drinkme.serving.engines import load_compressed

    with _env(**kw):
        return load_compressed(model, None, pack, device="cpu", ctx=256)


def _load_stock(model, **kw):
    from drinkme.serving.engines import load_stock

    with _env(**kw):
        return load_stock(model, None, device="cpu", ctx=256)


def _greedy(eng, images=(), n=6):
    from drinkme.serving.engine import complete

    msgs = PACK_MSGS if images else [{"role": "user", "content": "hello world the quick fox"}]
    with _env(DRINKME_SPEC="off"):
        return complete(eng, GenerationRequest(msgs, SampleParams(temperature=0.0, max_tokens=n),
                                               images=tuple(images)))


def test_the_pack_carries_the_whole_tower(packed):
    """The tower's eligible Linears are packed like any other (the patch
    embedding among them), the rest rides raw in the embedded checkpoint,
    and meta.json's `vision` block names the three subtrees and counts
    them; `drinkme verify` passes."""
    from safetensors import safe_open

    from drinkme.codec.pack import REMAINDER_FILE, embedded_dir, verify_pack

    model, pack, _ = packed
    meta = _meta(pack)
    tower = sorted(k for k in _headers(model) if in_tower(k))
    packed_names = sorted(n for n in meta["tensors"] if in_tower(n))
    assert len(packed_names) == PACKED_TOWER
    assert "model.vision_tower.patch_embedder.patch_embedding" in packed_names
    assert {"model.vision_adapter.fc1", "model.vision_adapter.fc2",
            "model.vision_projection"} <= set(packed_names)
    with safe_open(os.path.join(embedded_dir(pack), REMAINDER_FILE), "pt") as f:
        raw = sorted(k for k in f.keys() if in_tower(k))
    assert raw == [k for k in tower if k[:-len(".weight")] not in packed_names]
    assert "model.vision_tower.patch_embedder.position_embedding_table.weight" in raw
    block = meta["vision"]
    assert block["tower"] == "model.vision_tower"
    assert block["paths"] == ["model.vision_tower", "model.vision_adapter",
                              "model.vision_projection"]
    assert block["imageTokenId"] == SOFT
    assert block["tensorCount"] == len(tower) and block["packedTensorCount"] == PACKED_TOWER
    assert block["residentBytes"] < meta["residentBytes"]
    verify_pack(pack, progress=_quiet)


def test_the_served_tower_is_the_references_bit_for_bit_offline(packed, tmp_path, monkeypatch):
    """Loaded offline from the pack alone: the tower sits where the class
    builds it, its Linears packed (the patch embedding routed through
    vision.glimmer_patch_embed), and one image's features are transformers'
    get_image_features' bit for bit (the CPU compressed arm decodes the
    exact bf16 weights; the pixels go in as float32). The stock arm gives
    the same features and the same greedy answer, and the engine's prompt
    count is generate()'s."""
    import shutil

    from drinkme import arms
    from drinkme.codec.swap import CompressedLinear
    from drinkme.serving import checkpoint

    model, pack, want = packed
    stock = _load_stock(model)
    assert stock.vision is not None and isinstance(stock._tower, vision.GlimmerTower)
    assert (stock._tower.boi, stock._tower.eoi) == (VOCAB["<|image_start|>"],
                                                    VOCAB["<|image_end|>"])
    assert type(stock.model.model.vision_tower.patch_embedder.patch_embedding) \
        is torch.nn.Linear
    with torch.inference_mode():
        assert torch.equal(stock._tower.features(stock.model, _pack_image()), want)
    stock_text = _greedy(stock, [_pack_image()]).text

    copy_dir, pack_copy = str(tmp_path / "model"), str(tmp_path / "pack")
    shutil.copytree(model, copy_dir)
    shutil.copytree(pack, pack_copy)
    shutil.rmtree(copy_dir)

    def no_snapshot(*a, **k):
        raise AssertionError("a self-contained load resolved a snapshot")

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setattr(checkpoint, "snapshot_dir", no_snapshot)
    monkeypatch.setattr(arms, "snapshot_dir", no_snapshot)
    eng = _load(copy_dir, pack_copy)
    assert eng.vision is not None and eng.vision_reason is None
    embedder = eng.model.model.vision_tower.patch_embedder
    assert isinstance(embedder.patch_embedding, CompressedLinear)
    assert isinstance(eng.model.model.vision_adapter.fc1, CompressedLinear)
    with torch.inference_mode():
        assert torch.equal(eng._tower.features(eng.model, _pack_image()), want)
    assert "forward" in vars(embedder)  # the packed patch embedding took the shim
    res = _greedy(eng, [_pack_image()])
    assert res.text == stock_text
    assert res.prompt_tokens == eng.count_tokens(
        GenerationRequest(PACK_MSGS, SampleParams(), images=(_pack_image().plan,)))


def test_the_patch_embedding_shim_is_the_stock_forward(packed):
    """vision.glimmer_patch_embed over a plain Linear is
    MuseGlimmerVisionPatchEmbedder.forward bit for bit."""
    from transformers.models.muse_glimmer import MuseGlimmerForConditionalGeneration

    model, _pack, _ = packed
    ref = MuseGlimmerForConditionalGeneration.from_pretrained(model, dtype=torch.bfloat16).eval()
    embedder = ref.model.vision_tower.patch_embedder
    pv, grid = _batch([_pack_image()])
    with torch.inference_mode():
        assert torch.equal(vision.glimmer_patch_embed(embedder, pv, grid), embedder(pv, grid))


def test_images_off_prunes_the_tower_and_reads_none_of_it(packed, monkeypatch):
    """DRINKME_VISION=0: the class builds the tower, and the loaders prune
    it (the three subtrees None, no parameter of it left), never open its
    pack tensors, refuse images naming the switch and serve text; the fit
    leaves the tower's resident bytes out. The stock arm prunes it too."""
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
    cfg = _pack_config()
    assert eng.vision is None and eng.vision_reason == "disabled by DRINKME_VISION=0"
    m = eng.model.model
    assert (m.vision_tower, m.vision_adapter, m.vision_projection) == (None, None, None)
    assert not arms.has_vision_tower(eng.model, cfg)
    assert not any(in_tower(n) for n, _ in eng.model.named_parameters())
    assert not any(in_tower(n) for n in opened if n)
    assert _greedy(eng).completion_tokens == 6
    with pytest.raises(ValueError, match=r"cannot read images on this server "
                                         r"\(disabled by DRINKME_VISION=0\)"):
        _greedy(eng, [_pack_image()])
    assert (fit.served_resident_bytes(meta, False)
            == meta["residentBytes"] - meta["vision"]["residentBytes"])
    stock = _load_stock(model, DRINKME_VISION=0)
    assert stock.vision is None and not arms.has_vision_tower(stock.model, cfg)
    assert not any(in_tower(n) for n, _ in stock.model.named_parameters())
    assert _greedy(stock).text == _greedy(eng).text


def test_the_tables_keep_the_tree_the_bench_measures(packed):
    """skeleton(cfg), ckpt_to_skel(cfg) and load_cpu(dir) with no say (the
    bench's arms) are the tree gemma's and Muse-Glimmer's classes both
    build, tower included (a served model carries its tower, so its
    measurement does); True keeps it explicitly, False prunes it. The
    text keys are never renamed, and the class-built tower is never
    attached (there is nothing to attach — dropping and re-attaching would
    lose nothing, but the class already built it)."""
    from drinkme import arms

    model, _, _ = packed
    cfg = _pack_config()
    bench, on, off = arms.skeleton(cfg), arms.skeleton(cfg, vision=True), \
        arms.skeleton(cfg, vision=False)
    assert arms.vision_tower_paths(cfg) == ("model.vision_tower", "model.vision_adapter",
                                            "model.vision_projection")
    assert arms.has_vision_tower(bench, cfg) and arms.has_vision_tower(on, cfg)
    assert not arms.has_vision_tower(off, cfg)
    assert off.model.vision_projection is None and off.model.perception_emb_norm is not None
    keys = ("model.vision_tower.layers.0.attn.q_proj.weight", "model.vision_adapter.fc1.weight",
            "model.vision_projection.weight")
    for vis_, held in ((None, True), (True, True), (False, False)):
        t = arms.ckpt_to_skel(cfg, vision=vis_)
        assert [t(k) for k in keys] == (list(keys) if held else [None] * 3)
        text = "model.language_model.layers.0.mlp.up_proj.weight"
        assert t(text) == text and t("lm_head.weight") == "lm_head.weight"
    with pytest.raises(ValueError, match="built by its own class"):
        arms.attach_vision_tower(off, cfg)
    assert arms.has_vision_tower(arms.load_cpu(model), cfg)
    assert not arms.has_vision_tower(arms.load_cpu(model, vision=False), cfg)


def test_the_acceptance_gate_passes_on_the_toy(packed, tmp_path, capsys, monkeypatch):
    """bench/glimmer_vision_gate.py, the GPU acceptance's tool, on the CPU
    over the toy pack: the tower alone is transformers' get_image_features
    byte for byte on both arms, and every attention call of every arm is
    within the boot self-test's tolerance of fp32 (--fp32-check, with and
    without the CPU tower); each BROKEN kernel leaves the byte pin passing
    and fails the fp32 check. The first-token row of an image prompt is
    transformers' forward's on the stock arm and the stock arm's on the
    compressed one, with the same greedy answer."""
    import types

    import glimmer_vision_gate as gate

    model, pack, _ = packed

    def verdict():
        line = [ln for ln in capsys.readouterr().out.splitlines() if ln.startswith("VERDICT ")]
        return json.loads(line[-1][8:])

    common = dict(snap=model, pack=pack, size="112x84", device="cpu", need_gb=0)
    gate.tower(types.SimpleNamespace(**common, max_pixels=vision.DEFAULT_MAX_PIXELS, reps=2,
                                     fp32_check=True))
    v = verdict()
    assert v["verdict"] == "PASS" and len(set(v["sha256"].values())) == 1
    assert v["tokens"] == _pack_image().tokens and v["bounded_modules"] == 2
    assert v["fp32_verdict"] == "PASS"
    assert all(s["calls"] and not s["masked"] and not s["failed"]  # Glimmer's ViT never masks
               for s in v["attention_vs_fp32"].values())
    assert all(r["finite"] for r in v["vs_fp32_cpu"].values())
    gate_negative_controls(gate, lambda: gate.tower(types.SimpleNamespace(
        **common, max_pixels=vision.DEFAULT_MAX_PIXELS, reps=1, fp32_check=True)),
        verdict, monkeypatch)
    # --no-fp32-tower (the cap's form): no CPU tower, and the per-call check
    # alone still passes the sound kernel and fails each broken one
    calls_only = dict(common, max_pixels=vision.DEFAULT_MAX_PIXELS, reps=1, fp32_check=True,
                      no_fp32_tower=True)
    gate.tower(types.SimpleNamespace(**calls_only))
    v = verdict()
    assert v["verdict"] == "PASS" and v["fp32_verdict"] == "PASS"
    assert v["vs_fp32_cpu"] == {arm: {"finite": True}
                                for arm in ("reference", "stock", "compressed")}
    gate_negative_controls(gate, lambda: gate.tower(types.SimpleNamespace(**calls_only)),
                           verdict, monkeypatch)
    with _env():
        for arm in ("stock", "compressed"):
            gate.row(types.SimpleNamespace(**common, arm=arm, model=model, ctx=256, n=6,
                                           out=str(tmp_path)))
            v = verdict()
            if arm == "stock":
                assert v["reference_vs_drinkme"]["bytes_equal"]
    gate.compare(types.SimpleNamespace(out=str(tmp_path)))
    v = verdict()
    assert v["verdict"] == "PASS" and v["stock_vs_compressed"]["bytes_equal"]
    assert v["greedy_divergence"]["first_difference"] is None


# -------------------------------------- THE REAL SNAPSHOT AND TOKENIZERS --

GLIMMER = ("meta-models/Muse-Glimmer-30B", "a4e59da52a7bc87ae7251dd5545c0dd437c44b68")
VISION_ROWS = {  # architecture -> (repo, revision) of a menu row that serves its images
    "qwen3_5": ("Qwen/Qwen3.8-27B", "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"),
    "qwen3_5 (MiMo)": ("XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B",
                       "2367e865d009c13ac81713a2878291d33ab28177"),
    "gemma4": ("google/gemma-4-31B-it", "842da3794eaa0b77d5f08bae87a17459d91ff475"),
    "muse_glimmer": GLIMMER,
}
# how a modality marker is spelled in these tokenizers' added vocabularies
MODALITY = ("image", "vision", "video", "vid_", "audio", "patch")


def _cached(repo: str, revision: str, filename: str = "tokenizer_config.json") -> str:
    """The snapshot directory in the local HF cache, or a clean skip."""
    from huggingface_hub import try_to_load_from_cache

    hit = try_to_load_from_cache(repo, filename, revision=revision)
    if not isinstance(hit, str):
        pytest.skip(f"{repo}@{revision[:12]} is not in the local HF cache ({filename})")
    return os.path.dirname(hit)


def test_the_real_snapshot_maps_every_tower_tensor():
    """Muse-Glimmer-30B, read-only (config.json, the shard headers, the
    tokenizer; meta device, no weight bytes): every tower tensor has a
    same-shaped home in the served tree with images on, none without; the
    packable ones are counted; the text side is causal (no bidirectional
    setting) with NoPE full layers; the tower resolves its markers."""
    from transformers import AutoConfig, AutoTokenizer

    from drinkme import arms
    from drinkme.codec.swap import eligible_linears
    from drinkme.serving.engines import _vision_for

    snap = _cached(*GLIMMER, filename="config.json")
    headers = _headers(snap)
    cfg = AutoConfig.from_pretrained(snap)
    tower = {k: v for k, v in headers.items() if in_tower(k)}
    assert len(tower) == 806 + 2 + 1
    tree = arms.skeleton(cfg, vision=True)
    params = dict(tree.named_parameters())
    to_skel = arms.ckpt_to_skel(cfg, vision=True)
    for key, (dtype, shape, _f) in tower.items():
        assert dtype == "BF16" and list(params[to_skel(key)].shape) == shape, key
    off = arms.ckpt_to_skel(cfg, vision=False)
    assert all(off(k) is None for k in tower)
    eligible = [(n, m) for n, _p, _c, m in eligible_linears(tree) if in_tower(n)]
    assert len(eligible) == 50 * 6 + 1 + 2 + 1
    assert sum(m.weight.numel() for _n, m in eligible) == \
        50 * (4 * 1536 * 1536 + 2 * 1536 * 8960) + 1536 * 1176 + 6144 * 4096 + 4096 * 4096 \
        + 4096 * 6656
    text = cfg.get_text_config()
    assert getattr(text, "use_bidirectional_attention", None) is None
    assert all((t == "full_attention") == (th == 0)
               for t, th in zip(text.layer_types, text.layer_rope_theta))
    tok = AutoTokenizer.from_pretrained(snap, local_files_only=True)
    vis, gt, why = _vision_for(cfg, snap, GLIMMER[0], tokenizer=lambda: tok)
    assert why is None and vis.preprocessor == vision.MuseGlimmerPreprocessor("muse_glimmer")
    assert (gt.image_token_id, gt.boi, gt.eoi, gt.merge_size) == (200092, 200080, 200081, 2)


@pytest.mark.parametrize("row", sorted(VISION_ROWS))
def test_every_modality_marker_the_tokenizer_defines_is_reserved(row):
    """The injection guard's literals (Preprocessor.reserved_text) are
    exactly the tokenizer's image, video and audio markers: each one a
    single token of its added vocabulary, and none of them missing
    (gemma-4's audio markers were, before this step)."""
    from transformers import AutoTokenizer

    repo, revision = VISION_ROWS[row]
    snap = _cached(repo, revision)
    tok = AutoTokenizer.from_pretrained(snap, local_files_only=True)
    arch = row.split()[0]
    reserved = {"qwen3_5": vision.QwenVLPreprocessor("qwen3_5"),
                "gemma4": vision.Gemma4Preprocessor("gemma4"),
                "muse_glimmer": vision.MuseGlimmerPreprocessor("muse_glimmer")}[arch].reserved_text
    added = tok.get_added_vocab()
    markers = {t for t in added if any(w in t.lower() for w in MODALITY)}
    assert markers == set(reserved)
    for literal in reserved:
        assert tok.encode(literal, add_special_tokens=False) == [added[literal]]


def test_the_guard_refuses_every_reserved_marker_of_each_architecture():
    """Over HTTP, text spelling any of an architecture's modality markers is
    refused by name whatever the dialect's engine serves: gemma-4's audio
    markers included."""
    from drinkme.serving.http import start_server

    from test_serving_http import VisionEngine, post

    for pre in (vision.QwenVLPreprocessor("qwen3_5"), vision.Gemma4Preprocessor("gemma4"),
                vision.MuseGlimmerPreprocessor("muse_glimmer")):
        eng = VisionEngine(veng=vision.Vision(pre))
        srv = start_server(eng, port=0)
        try:
            for literal in pre.reserved_text:
                r, body = post(srv.server_address[1], {"model": "fake", "max_tokens": 2,
                                                       "messages": [{"role": "user",
                                                                     "content": f"a {literal} b"}]})
                assert r.status == 400 and b"image_injection" in body, (pre.architecture, literal)
        finally:
            srv.shutdown()
        assert eng.calls == 0
    gemma = vision.Gemma4Preprocessor("gemma4").reserved_text
    assert {"<|audio|>", "<|audio>", "<audio|>"} <= set(gemma)
