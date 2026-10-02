"""gemma-4 image input, CPU, float32, no downloads.

PREPROCESSING is pinned BIT FOR BIT against transformers'
Gemma4ImageProcessorPil at its class defaults (the gemma-4-31B-it snapshot
ships no processor config), fed through transformers' own load_image: the
pixel_values bytes (padded to the budget's patch count), the padded
image_position_ids, the grid and the soft-token count, over
test_serving_vision.py's fixture set plus gemma's own resize edges. The
server's pixel cap lowers max_soft_tokens, pinned against the processor at
that budget.

THE TEXT SIDE on a toy gemma-4: the reference is transformers' own
Gemma4ForConditionalGeneration (forward(input_ids, pixel_values,
image_position_ids, mm_token_type_ids)), built tiny and random with the
31B's layer pattern (five sliding-window layers per full one) and
`use_bidirectional_attention: "vision"`. The served tree is a copy of the
same class (gemma-4's causal auto-mapping builds it whole, tower
included). Two sliding windows: 16, full from the first image on and
shorter than an image's run, and 512, never full. Pinned: prefill logits
with images, whole and chunked, and the greedy continuation through
generate() are the reference's; the bidirectional mask is what makes them
so (the causal negative control differs); a reuse point inside an image is
refused; a text request makes the calls it always made.

THE TOWER IN THE SERVED TREE: a toy gemma-4 checkpoint (bf16, no processor
config, like the snapshot) is packed and loaded both arms: the pack's
`vision` block names both of the tower's subtrees, the served tower is
transformers' get_image_features bit for bit offline, DRINKME_VISION=0
prunes it (nothing of it read or resident), and a pack cut before gemma's
tower was served serves text with the tower pruned.
"""

from __future__ import annotations

import base64
import copy
import dataclasses
import io
import json
import os

import numpy as np
import pytest
import torch

from drinkme.serving import prefill, vision
from drinkme.serving.engine import GenerationRequest, SampleParams, complete
from drinkme.serving.engines import HFEngine
from drinkme.serving.image_prompt import ImagePrompt

from test_serving_image_prompt import _Truncated
from test_serving_prefill_bound import env
from test_serving_vision import FIXTURES, _encoded, _picture, _save

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


# ------------------------------------------------------ PREPROCESSING --

def _gemma_fixtures() -> dict[str, bytes]:
    """gemma's own resize edges, beside the shared set: a size that is
    already a budget size (the reference skips the resize), a side that
    floors to 0 (the edge branch, both ways), a source far under the budget
    (upscaled), and aspect ratios either side of square."""
    return {
        "exact-768x816": _save(_picture(768, 816, 40), "PNG"),
        "wide-3000x3": _save(_picture(3000, 3, 41), "PNG"),
        "tall-3x3000": _save(_picture(3, 3000, 42), "PNG"),
        "tiny-5x7": _save(_picture(7, 5, 43), "PNG"),
        "wide-1000x40": _save(_picture(1000, 40, 44), "JPEG", quality=90),
        "square-1024": _save(_picture(1024, 1024, 45), "PNG"),
    }


ALL = {**FIXTURES, **_gemma_fixtures()}
DEFAULTS = vision.Vision(vision.preprocessor_for("gemma4", None), 1 << 40)


def _reference(data: bytes, max_soft_tokens: int | None = None):
    """transformers' answer: load_image on the base64 string, then
    Gemma4ImageProcessorPil at its class defaults (optionally another
    max_soft_tokens): (pixel_values [patches, 768], positions [patches, 2],
    soft tokens)."""
    pil = pytest.importorskip("transformers.models.gemma4.image_processing_pil_gemma4")
    from transformers.image_utils import load_image

    kw = {} if max_soft_tokens is None else {"max_soft_tokens": max_soft_tokens}
    out = pil.Gemma4ImageProcessorPil()(load_image(base64.b64encode(data).decode()),
                                         return_tensors="np", **kw)
    return (out["pixel_values"][0], out["image_position_ids"][0],
            int(out["num_soft_tokens_per_image"][0]))


def _assert_is_reference(img: vision.PreparedImage, ref) -> None:
    pv, pos, tokens = ref
    assert img.tokens == tokens
    assert img.pixel_values.dtype == np.float32 and pv.dtype == np.float32
    assert img.pixel_values.shape == pv.shape
    assert img.pixel_values.tobytes() == np.ascontiguousarray(pv).tobytes()
    assert np.array_equal(vision.patch_positions(img.grid_thw, pv.shape[0]), pos)
    _t, gh, gw = img.grid_thw
    assert img.size == (gh * 16, gw * 16) and int((pos[:, 0] >= 0).sum()) == gh * gw


@pytest.mark.parametrize("name", sorted(ALL))
def test_preprocessing_is_the_pil_processors_bit_for_bit(name):
    """Every fixture: the padded pixel_values bytes, the positions, the
    grid and the token count equal Gemma4ImageProcessorPil's, and the
    header-only count equals the prepared plan."""
    img = DEFAULTS.prepare(_encoded(ALL[name]))
    _assert_is_reference(img, _reference(ALL[name]))
    assert DEFAULTS.count(_encoded(ALL[name])) == img.plan
    assert img.pixel_values.shape == (280 * 9, 16 * 16 * 3)


# (width, height) -> resized (width, height), soft tokens, at the class
# defaults; transformers' get_aspect_ratio_preserving_size gives the same
TOKENS = [((512, 512), (768, 768), 256), ((1280, 800), (1008, 624), 273),
          ((1920, 1080), (1056, 576), 264), ((2560, 1440), (1056, 576), 264),
          ((2880, 1800), (1008, 624), 273), ((3840, 2160), (1056, 576), 264),
          ((768, 816), (768, 816), 272), ((10000, 1), (13440, 48), 280)]


@pytest.mark.parametrize("src,size,tokens", TOKENS)
def test_the_token_table(src, size, tokens):
    """Variable tokens, at most 280, sides multiples of 48: the default
    server cap (2560x1440) never bites, gemma's own budget does."""
    from transformers.models.gemma4.image_processing_pil_gemma4 import (
        get_aspect_ratio_preserving_size)

    w, h = src
    pre = DEFAULTS.preprocessor
    got = pre.plan(h, w, vision.DEFAULT_MAX_PIXELS, where="t")
    assert got == ((size[1], size[0]), (1, size[1] // 16, size[0] // 16), tokens)
    assert get_aspect_ratio_preserving_size(height=h, width=w, patch_size=16, max_patches=2520,
                                            pooling_kernel_size=3) == (size[1], size[0])
    assert tokens <= 280 and size[0] % 48 == 0 and size[1] % 48 == 0


@pytest.mark.parametrize("cap,soft", [(512 * 512, 113), (140 * 2304, 140), (200_000, 86),
                                      (70 * 2304, 70), (1, 1)])
@pytest.mark.parametrize("name", ["1080p", "retina-2880x1800", "wide-1000x40", "tiny-5x7",
                                  "jpeg-exif-6"])
def test_the_pixel_cap_lowers_max_soft_tokens(name, cap, soft, monkeypatch):
    """A budget under gemma's own (detail "low", a server cap) lowers
    max_soft_tokens to what it covers, never below 1: the result is the
    processor's at that max_soft_tokens, bit for bit (transformers accepts
    only five budgets, so the test lets it take the others), and the
    image's pixels stay under the cap wherever one 48 x 48 token fits."""
    import transformers.models.gemma4.image_processing_pil_gemma4 as pil

    monkeypatch.setattr(pil, "_SUPPORTED_SOFT_TOKENS", pil._SUPPORTED_SOFT_TOKENS + (soft,))
    v = vision.Vision(vision.preprocessor_for("gemma4", None), cap)
    img = v.prepare(_encoded(ALL[name]))
    _assert_is_reference(img, _reference(ALL[name], soft))
    assert img.pixel_values.shape[0] == soft * 9 and img.tokens <= soft
    assert v.count(_encoded(ALL[name])) == img.plan
    if cap >= 2304:
        assert img.size[0] * img.size[1] <= cap
    low = vision.Vision(vision.preprocessor_for("gemma4", None)).prepare(
        _encoded(ALL[name]), detail="low")
    assert low.pixel_values.shape[0] == 113 * 9


def test_the_registry_builds_gemma_from_the_class_defaults(tmp_path):
    """No processor config (the 31B's snapshot has none): the class
    defaults. A processor config's keys override them; what is not
    implemented is refused by name."""
    assert "gemma4" in vision.architectures()
    pre = vision.load("gemma4", str(tmp_path)).preprocessor
    assert (pre.patch_size, pre.max_soft_tokens, pre.pooling_kernel_size, pre.wrap) == (16, 280, 3, 2)
    assert set(pre.reserved_text) == {"<|image|>", "<|image>", "<image|>", "<|video|>",
                                      "<|audio|>", "<|audio>", "<audio|>"}
    (tmp_path / "preprocessor_config.json").write_text(json.dumps(
        {"image_processor_type": "Gemma4ImageProcessor", "max_soft_tokens": 560}))
    assert vision.load("gemma4", str(tmp_path)).preprocessor.max_soft_tokens == 560
    for bad, match in (({"max_soft_tokens": 100}, "max_soft_tokens 100"),
                       ({"do_normalize": True}, "do_normalize"),
                       ({"image_processor_type": "SiglipImageProcessor"}, "Siglip")):
        with pytest.raises(ValueError, match=match):
            vision.preprocessor_for("gemma4", bad)


def test_a_prompt_counts_the_markers_around_each_image():
    """gemma's placeholder expands to <|image> + tokens + <image|>: two
    more prompt tokens per image than the soft tokens (Qwen's none)."""
    plans = [DEFAULTS.count(_encoded(ALL[n])) for n in ("1080p", "tiny-5x7")]
    assert [DEFAULTS.prompt_tokens(p) for p in plans] == [264 + 2, plans[1].tokens + 2]
    assert DEFAULTS.expansion(plans) == 264 + 1 + plans[1].tokens + 1


def test_bounded_attention_splits_a_masked_call_by_query_rows(monkeypatch):
    """gemma's ViT attention carries a padding mask: over the bound, the
    queries are split with their rows of the mask (never the keys), and the
    result is one masked SDPA call's bit for bit on the CPU."""
    from transformers.integrations import sdpa_attention as sd

    class _Attn(torch.nn.Module):
        is_causal = False
        num_key_value_groups = 1

    g = torch.Generator().manual_seed(4)
    q, k, v = (torch.randn(1, 2, 45, 8, generator=g) for _ in range(3))
    valid = torch.arange(45) < 36  # the last 9 patches are padding
    mask = (valid[None, :] & valid[:, None])[None, None]
    want = sd.sdpa_attention_forward(_Attn(), q, k, v, mask, scaling=1.0)[0]
    seen = []
    real = sd.sdpa_attention_forward

    def rec(module, query, key, value, m, **kw):
        seen.append((query.shape[-2] * key.shape[-2], None if m is None else tuple(m.shape)))
        return real(module, query, key, value, m, **kw)

    monkeypatch.setattr(sd, "sdpa_attention_forward", rec)
    got, _ = vision.bounded_attention(_Attn(), q, k, v, mask, scaling=1.0, work=45 * 10)
    assert len(seen) == 5 and all(p <= 45 * 10 and m[-2] <= 10 for p, m in seen)
    assert torch.equal(got, want)


# --------------------------------------------------------- THE TOY MODEL --

WORDS = ["hello", "world", "the", "quick", "brown", "fox", "what", "is", "in", "this",
         "picture", "and", "user", "assistant", ":"]
SPECIAL = ["<|image|>", "<|image>", "<image|>", "<|video|>", "<|audio|>"]


def _vocab() -> dict:
    vocab = {"<unk>": 0, "<pad>": 1, "</s>": 2}
    for w in WORDS + SPECIAL:
        vocab[w] = len(vocab)
    while len(vocab) < 96:
        vocab[f"tok{len(vocab)}"] = len(vocab)
    return vocab


VOCAB = _vocab()
SOFT = VOCAB["<|image|>"]
TEMPLATE = (
    "{% for m in messages %}{{ m['role'] }} : {% if m['content'] is string %}"
    "{{ m['content'] }}{% else %}{% for p in m['content'] %}{% if p['type'] == 'image' %}"
    "<|image|>{% else %}{{ p['text'] }} {% endif %}"
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


def config(window: int = 16, *, patch: int = 4, vision_width: int = 32, vision_mlp: int = 64,
           vision_layers: int = 2, vision_head: int = 16):
    """A tiny gemma-4 with the 31B's text geometry (5:1 sliding:full,
    k_eq_v, the logit softcap, no per-layer inputs, bidirectional vision
    attention) and a small ViT (standardized, 3 x 3 pooling; the 31B's
    head width is 72, test_serving_vision_attention.py's)."""
    from transformers import Gemma4Config

    return Gemma4Config(
        text_config=dict(
            vocab_size=len(VOCAB), hidden_size=64, intermediate_size=128, num_hidden_layers=12,
            num_attention_heads=2, num_key_value_heads=1, head_dim=32, global_head_dim=32,
            num_global_key_value_heads=1,
            layer_types=(["sliding_attention"] * 5 + ["full_attention"]) * 2,
            sliding_window=window, max_position_embeddings=1024, final_logit_softcapping=30.0,
            attention_k_eq_v=True, hidden_size_per_layer_input=0,
            vocab_size_per_layer_input=len(VOCAB), tie_word_embeddings=True,
            use_bidirectional_attention="vision",
            # no EOS: a random argmax would end the comparison runs early
            eos_token_id=None, pad_token_id=VOCAB["<pad>"]),
        vision_config=dict(
            hidden_size=vision_width, intermediate_size=vision_mlp,
            num_hidden_layers=vision_layers, num_attention_heads=2, num_key_value_heads=2,
            head_dim=vision_head, patch_size=patch, pooling_kernel_size=3,
            position_embedding_size=256,
            rope_parameters={"rope_theta": 100.0, "rope_type": "default"}, standardize=True),
        audio_config=None, image_token_id=SOFT, boi_token_id=VOCAB["<|image>"],
        eoi_token_id=VOCAB["<image|>"], video_token_id=VOCAB["<|video|>"],
        audio_token_id=VOCAB["<|audio|>"], tie_word_embeddings=True)


# the toy's preprocessing: gemma's rule at a 4-pixel patch and 70 tokens
VIS = vision.Vision(vision.Gemma4Preprocessor("gemma4", patch_size=4, max_soft_tokens=70),
                    max_pixels=1 << 30)
WINDOWS = (16, 512)


def _tower(cfg) -> vision.Tower:
    from drinkme import arms

    return vision.tower_for("gemma4", cfg, arms.vision_tower_paths(cfg))


def png(w: int, h: int, seed: int) -> str:
    from PIL import Image

    a = np.random.default_rng(seed).integers(0, 256, size=(h, w, 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(a).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def image(w: int, h: int, seed: int, vis=VIS):
    """A fresh PreparedImage (the engine releases its pixels once read)."""
    return vis.prepare(vision.parse_base64(png(w, h, seed), "image/png", where="test"))


def two_images():
    return [image(40, 24, 1), image(24, 36, 2)]


@pytest.fixture(scope="module", params=WINDOWS, ids=lambda w: f"window{w}")
def toy(request):
    """(reference, served tree, tokenizer, tower) at one sliding window."""
    from transformers import Gemma4ForConditionalGeneration

    cfg = config(request.param)
    torch.manual_seed(0)
    ref = Gemma4ForConditionalGeneration(cfg).eval().float()
    with torch.no_grad():  # the standardization buffers start empty
        ref.model.vision_tower.std_bias.normal_()
        ref.model.vision_tower.std_scale.uniform_(0.5, 1.5)
    for p in ref.parameters():
        p.requires_grad_(False)
    return ref, copy.deepcopy(ref), _tokenizer(), _tower(cfg)


def engine(toy, *, slots=1, chunk=0, **kw):
    _ref, model, tok, tower = toy
    with env(DRINKME_PREFILL_CHUNK=chunk, DRINKME_PREFIX_SLOTS=slots, DRINKME_SLOT_DIR="off"):
        return HFEngine(model, tok, model_id="toy", arm="test", meta={}, ctx=320,
                        vision=VIS, tower=tower, **kw)


def user(*parts):
    """A user turn from text strings and None (an image placeholder)."""
    return {"role": "user", "content": [{"type": "image"} if p is None else
                                        {"type": "text", "text": p} for p in parts]}


def ask(eng, msgs, images, spec="off", n=8):
    with env(DRINKME_SPEC=spec):
        return complete(eng, GenerationRequest(msgs, SampleParams(temperature=0.0, max_tokens=n),
                                               images=tuple(images)))


def rendered(tok, msgs):
    out = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True)
    return out if isinstance(out, list) else out["input_ids"]


def ref_logits(ref, ids, images):
    """Gemma4ForConditionalGeneration over the whole prompt, no cache: the
    processor's batch (padded pixel_values and positions) and its
    mm_token_type_ids (1 on the soft tokens)."""
    pv = torch.stack([torch.tensor(np.array(i.pixel_values)) for i in images])
    pos = torch.stack([torch.tensor(vision.patch_positions(i.grid_thw, i.pixel_values.shape[0]))
                       for i in images])
    t = torch.tensor([ids])
    with torch.inference_mode():
        return ref(input_ids=t, pixel_values=pv, image_position_ids=pos,
                   mm_token_type_ids=(t == SOFT).int(), use_cache=False).logits[0]


def ref_greedy(ref, ids, images, n):
    ids = list(ids)
    out = []
    for _ in range(n):
        t = int(ref_logits(ref, ids, images)[-1].argmax())
        out.append(t)
        ids.append(t)
    return out


def softcap(model, x):
    cap = model.config.get_text_config().final_logit_softcapping
    return torch.tanh(x / cap) * cap


MSGS = [user("what is in", None, "the quick brown", None, "hello world")]


# ----------------------------------------------------------- THE SPLICE --

def test_the_placeholder_expands_between_the_markers(toy):
    """<|image|> -> <|image> + tokens x <|image|> + <image|>; the run is
    the soft tokens alone, keyed by the image; the count routes agree."""
    _ref, model, tok, tower = toy
    imgs = two_images()
    r = rendered(tok, MSGS)
    ip = ImagePrompt(r, imgs, tower, model)
    assert len(ip.ids) == len(r) + VIS.expansion(imgs)
    for (s, e, img), key in zip(ip.runs, (imgs[0].prefix_key, imgs[1].prefix_key)):
        assert e - s == img.tokens
        assert ip.ids[s - 1] == VOCAB["<|image>"] and ip.ids[e] == VOCAB["<image|>"]
        assert set(ip.ids[s:e]) == {SOFT} and set(ip.key_ids[s:e]) == {key}
        assert ip.key_ids[s - 1] == VOCAB["<|image>"] and ip.key_ids[e] == VOCAB["<image|>"]
    assert ip.pos4 is None and not ip.rope


@pytest.mark.parametrize("chunk", [0, 7, 70])
def test_prefill_logits_are_the_references(toy, chunk):
    """Every prompt row's logits (lm_head and the softcap over each span's
    hidden states) and the last row's (the serial prefill generate() runs
    on gemma, the wrapper's logits_to_keep=1 call), whole and chunked,
    against Gemma4ForConditionalGeneration over the same weights. At 7
    each image run is a span of its own; at 70 a span carries text and a
    whole image (and the window-16 ring is full before the second
    image)."""
    ref, model, tok, tower = toy
    ip = ImagePrompt(rendered(tok, MSGS), two_images(), tower, model)
    want = ref_logits(ref, ip.ids, two_images())
    hs = []
    with torch.inference_mode():
        cache = engine(toy)._cache(256)
        prefill.run(model, ip.ids, 0, cache, "cpu", chunk, hidden=True,
                    on_hidden=lambda h, a, b: hs.append((h, a, b)), image=ip)
        every = softcap(model, model.lm_head(torch.cat([h for h, _, _ in hs], 1)))[0]
        serial = ImagePrompt(rendered(tok, MSGS), two_images(), tower, model)
        last = prefill.run(model, serial.ids, 0, engine(toy)._cache(256), "cpu", chunk,
                           image=serial)
    torch.testing.assert_close(every, want, atol=2e-5, rtol=1e-5)
    torch.testing.assert_close(last, want[-1], atol=2e-5, rtol=1e-5)
    runs = [(s, e) for s, e, _ in ip.runs]
    spans = [(a, b) for _, a, b in hs]
    assert spans == prefill.spans(0, ip.n, chunk, runs)
    assert not any(s < a < e or s < b < e for a, b in spans for s, e in runs)
    if chunk == 70:
        assert len(spans) == 2 and all(any(a <= s and e <= b for s, e in runs) for a, b in spans)


def test_the_bidirectional_mask_is_what_matches(toy):
    """The negative control: the same tree with the image runs causal (a
    Tower without `bidirectional`, what Qwen's text side does) is NOT the
    reference, and the mask a span is handed lets an image row see the
    rows after it in its own run on the sliding layers only."""
    ref, model, tok, tower = toy
    ids = ImagePrompt(rendered(tok, MSGS), two_images(), tower, model).ids
    want = ref_logits(ref, ids, two_images())
    causal = ImagePrompt(rendered(tok, MSGS), two_images(),
                         dataclasses.replace(tower, bidirectional=False), model)
    with torch.inference_mode():
        got = prefill.run(model, causal.ids, 0, engine(toy)._cache(256), "cpu", 0, image=causal)
    assert (got - want[-1]).abs().max() > 1e-3
    ip = ImagePrompt(rendered(tok, MSGS), two_images(), tower, model)
    (s, e, _), (s2, _e2, _) = ip.runs
    with torch.inference_mode():
        cache = engine(toy)._cache(256)
        x = torch.zeros(1, ip.n, 64)
        m = ip.masks(model.get_decoder(), 0, ip.n, cache, x)
    full, sliding = m["full_attention"], m["sliding_attention"]
    assert full is None or not bool(full[0, 0, s, s + 1])  # causal
    assert bool(sliding[0, 0, s, e - 1]) and not bool(sliding[0, 0, s, e])  # its own run only
    assert not bool(sliding[0, 0, s - 1, s]) and not bool(sliding[0, 0, e - 1, s2])
    assert ImagePrompt(rendered(tok, [user("hello")]), [], tower, model).masks(
        model.get_decoder(), 0, 4, cache, x) is None  # no run in the span: its own masks


def test_segmented_attention_never_takes_a_gemma_span(toy):
    """The layout check refuses gemma's sliding layers, so a span holding an
    image run only ever carries the run's own mask: even over a written
    cache with the split threshold at zero, prefill_mask gives None."""
    from drinkme.serving import segmented_attention

    _ref, model, tok, tower = toy
    ip = ImagePrompt(rendered(tok, MSGS), two_images(), tower, model)
    s, _e, _ = ip.runs[0]
    with torch.inference_mode():
        cache = engine(toy)._cache(256)
        prefill.run(model, ip.ids[:s], 0, cache, "cpu", 0, image=_Truncated(ip, s))
        assert segmented_attention.needed(ip.n - s, s, above=0)
        assert segmented_attention.prefill_mask(model.get_decoder(), cache, ip.n - s,
                                                above=0) is None


@pytest.mark.parametrize("spec,chunk", [("off", 0), ("off", 7), ("ngram", 0), ("ngram", 7)])
def test_the_engines_greedy_is_the_references(toy, spec, chunk):
    """generate() end to end, serial and with the n-gram lookup (gemma has
    no draft head), whole and chunked: the reference's greedy token for
    token, and the count routes' prompt count is generate()'s."""
    ref, model, tok, tower = toy
    n = 10
    ids = ImagePrompt(rendered(tok, MSGS), two_images(), tower, model).ids
    want = tok.decode(ref_greedy(ref, ids, two_images(), n))
    eng = engine(toy, chunk=chunk)
    got = ask(eng, MSGS, two_images(), spec, n)
    assert got.prompt_tokens == len(ids) == eng.count_tokens(
        GenerationRequest(MSGS, SampleParams(), images=tuple(i.plan for i in two_images())))
    assert got.text.split() == want.split()


def test_the_dialects_count_gemmas_markers_and_refuse_them_in_text(toy):
    """Over HTTP on the gemma engine: /tokenize's count of a turn with an
    image is the prompt generate() consumed (the two markers around each
    image included), and text spelling one of gemma's image markers is
    refused by name."""
    from drinkme.serving.http import start_server

    from test_serving_http import post

    srv = start_server(engine(toy), port=0)
    try:
        port = srv.server_address[1]
        url = "data:image/png;base64," + png(40, 24, 7)
        msgs = [{"role": "user", "content": [{"type": "text", "text": "what is in"},
                                             {"type": "image_url", "image_url": {"url": url}}]}]
        r, body = post(port, {"messages": msgs}, "/tokenize")
        assert r.status == 200
        count = json.loads(body)["count"]
        r, body = post(port, {"model": "toy", "messages": msgs, "max_tokens": 2,
                              "temperature": 0})
        assert r.status == 200 and json.loads(body)["usage"]["prompt_tokens"] == count
        for marker in ("<|image|>", "<|image>", "<image|>"):
            r, body = post(port, {"model": "toy", "max_tokens": 2, "messages": [
                {"role": "user", "content": f"hello {marker} world"}]})
            assert r.status == 400 and marker.encode() in body and b"image_injection" in body
    finally:
        srv.shutdown()


# ----------------------------------------------------- PREFIX CACHE --

class TowerCalls:
    """How many times the served tree's ViT ran."""

    def __init__(self, model):
        self.n = 0
        self.h = model.get_submodule("model.vision_tower").register_forward_hook(self._hit)

    def _hit(self, *a):
        self.n += 1

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.h.remove()


def test_a_reuse_point_inside_an_image_is_refused(toy):
    """A cache that holds the prompt up to the middle of an image cannot be
    extended on gemma: the run's first rows were written without seeing
    its last (resuming from there is NOT the reference, the negative
    control). The engine sees a slot like that match and prefills cold,
    and answers what the reference answers."""
    ref, model, tok, tower = toy
    ip = ImagePrompt(rendered(tok, MSGS), two_images(), tower, model)
    s, e, _ = ip.runs[1]
    mid = (s + e) // 2
    assert ip.cuts(mid) and not ip.cuts(s) and not ip.cuts(e)
    assert not ImagePrompt(rendered(tok, MSGS), two_images(),
                           dataclasses.replace(tower, bidirectional=False), model).cuts(mid)

    def half_written(eng):
        cache = eng._cache(256)
        head = ImagePrompt(rendered(tok, MSGS), two_images(), tower, model)
        with torch.inference_mode():
            prefill.run(model, head.ids[:mid], 0, cache, "cpu", 0, image=_Truncated(head, mid))
        return cache

    want = ref_logits(ref, ip.ids, two_images())[-1]
    eng = engine(toy)
    cache = half_written(eng)
    rest = ImagePrompt(rendered(tok, MSGS), two_images(), tower, model)
    rest.begin(mid)
    with torch.inference_mode():
        resumed = prefill.run(model, rest.ids, mid, cache, "cpu", 0, image=rest)
    assert (resumed - want).abs().max() > 1e-4
    eng = engine(toy)
    slot = eng._slots[0]
    slot.cache, slot.alloc, slot.ids = half_written(eng), 256, ip.key_ids[:mid]
    got = ask(eng, MSGS, two_images(), n=6)
    assert got.cached_tokens == 0
    assert got.text.split() == tok.decode(ref_greedy(ref, ip.ids, two_images(), 6)).split()


def test_the_same_image_resent_reuses_its_kv_and_skips_the_tower(toy):
    """A conversation that carries its image forward extends its slot past
    the image, whole runs only, without running the tower again, and
    answers what a cold engine answers. A different image of the same size
    never shares the slot."""
    _ref, model, _tok, tower = toy
    eng = engine(toy)
    hist = [user("what is in this picture", None)]
    first = ask(eng, hist, [image(40, 24, 21)], n=4)
    hist = hist + [{"role": "assistant", "content": first.text}, user("and the fox")]
    with TowerCalls(model) as calls:
        warm = ask(eng, hist, [image(40, 24, 21)])
    assert calls.n == 0 and warm.cached_tokens > 0
    with TowerCalls(model) as calls:
        cold = ask(engine(toy), hist, [image(40, 24, 21)])
    assert calls.n == 1 and warm.text == cold.text
    other = ask(eng, hist, [image(40, 24, 22)])
    assert other.cached_tokens <= rendered(_tok, hist).index(SOFT)
    assert other.text == ask(engine(toy), hist, [image(40, 24, 22)]).text


# ----------------------------------------------- TEXT IS UNCHANGED --

class OuterCalls:
    """Every forward drinkme hands the trunk, as (class, positional input
    kind, sorted non-None kwarg names): the wrapper's, and the decoder's
    when drinkme calls it directly (prefill's spans, the hidden path) —
    not the wrapper's own call into its decoder, which always passes
    embeddings and a mask mapping (Gemma4Model.forward)."""

    def __init__(self, model):
        self.calls, self.depth = [], 0
        self.hooks = [model.register_forward_pre_hook(self._enter, with_kwargs=True),
                      model.register_forward_hook(self._leave),
                      model.get_decoder().register_forward_pre_hook(self._decoder,
                                                                    with_kwargs=True)]

    def _rec(self, mod, args, kwargs):
        given = {k for k, v in kwargs.items() if v is not None}
        ids = args[0] if args else kwargs.get("input_ids")
        self.calls.append((type(mod).__name__, type(ids).__name__,
                           tuple(sorted(given - {"input_ids"}))))

    def _enter(self, mod, args, kwargs):
        self._rec(mod, args, kwargs)
        self.depth += 1

    def _leave(self, *_a):
        self.depth -= 1

    def _decoder(self, mod, args, kwargs):
        if not self.depth:
            self._rec(mod, args, kwargs)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        for h in self.hooks:
            h.remove()


@pytest.mark.parametrize("spec,chunk", [("off", 0), ("off", 7), ("ngram", 7)])
def test_a_text_request_makes_the_calls_it_always_made(toy, spec, chunk, monkeypatch):
    """No position_ids, no inputs_embeds, no attention_mask: the ids as the
    one positional input on every forward of a text generation on the
    image-capable gemma engine, and no ImagePrompt is ever built."""
    import drinkme.serving.image_prompt as ip_mod

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
# A toy gemma-4 checkpoint saved in bf16 the way transformers saves
# Gemma4ForConditionalGeneration (model.language_model.*,
# model.vision_tower.*, model.embed_vision.*), with no processor config,
# like the 31B's snapshot: the class defaults (patch 16, 280 tokens) apply,
# lowered by the server cap below to 70 tokens. The ViT is 1024 wide with
# one layer, so its MLP's three Linears clear the codec's bar (1024 x
# 1024); everything else of the tower rides raw.

PACK_CAP = 70 * 16 * 16 * 9  # the server cap: 70 soft tokens
PACK_MSGS = [user("what is in this picture", None)]


def _quiet(*_a, **_k):
    pass


def _pack_config():
    return config(16, patch=16, vision_width=1024, vision_mlp=1024, vision_layers=1)


def _pack_vision():
    return vision.Vision(vision.preprocessor_for("gemma4", None), PACK_CAP)


def _pack_image():
    return image(96, 64, 5, vis=_pack_vision())


def _checkpoint(path: str) -> None:
    from transformers import Gemma4ForConditionalGeneration

    torch.manual_seed(0)
    model = Gemma4ForConditionalGeneration(_pack_config())
    with torch.no_grad():
        model.model.vision_tower.std_bias.normal_()
        model.model.vision_tower.std_scale.uniform_(0.5, 1.5)
    model.to(torch.bfloat16).eval().save_pretrained(path, safe_serialization=True)
    _tokenizer().save_pretrained(path)


def _meta(pack):
    with open(os.path.join(pack, "meta.json")) as f:
        return json.load(f)


def _headers(path):
    from drinkme.codec.pack import _shard_headers

    return _shard_headers(path)


TOWER_PREFIXES = ("model.vision_tower.", "model.embed_vision.")


@pytest.fixture(scope="module")
def packed(tmp_path_factory):
    """(checkpoint dir, pack dir, the reference's features for _pack_image),
    packed once; the reference is read before any test deletes anything."""
    from transformers import Gemma4ForConditionalGeneration

    from drinkme.codec.pack import pack_model

    root = tmp_path_factory.mktemp("gemma-vision-pack")
    model, pack = str(root / "model"), str(root / "pack")
    _checkpoint(model)
    pack_model(model, None, pack, progress=_quiet)
    ref = Gemma4ForConditionalGeneration.from_pretrained(model, dtype=torch.bfloat16).eval()
    img = _pack_image()
    pv = torch.tensor(np.array(img.pixel_values))[None]
    pos = torch.tensor(vision.patch_positions(img.grid_thw, pv.shape[1]))[None]
    with torch.inference_mode():
        feats = ref.model.get_image_features(pv, pos).pooler_output[0]
    return model, pack, feats


def _env(**kw):
    return env(DRINKME_PREFIX_SLOTS=0, DRINKME_SLOT_DIR="off",
               DRINKME_IMAGE_MAX_PIXELS=PACK_CAP, **kw)


def _load(model, pack, **kw):
    from drinkme.serving.engines import load_compressed

    with _env(**kw):
        return load_compressed(model, None, pack, device="cpu", ctx=256)


def _load_stock(model, **kw):
    from drinkme.serving.engines import load_stock

    with _env(**kw):
        return load_stock(model, None, device="cpu", ctx=256)


def _greedy(eng, images=(), n=6):
    msgs = PACK_MSGS if images else [{"role": "user", "content": "hello world the quick fox"}]
    with _env(DRINKME_SPEC="off"):
        return complete(eng, GenerationRequest(msgs, SampleParams(temperature=0.0, max_tokens=n),
                                               images=tuple(images)))


def test_the_pack_carries_both_halves_of_the_tower(packed):
    """The ViT's eligible Linears are packed like any other, the rest of
    the ViT and all of embed_vision ride raw in the embedded checkpoint,
    and meta.json's `vision` block names both subtrees and counts them;
    `drinkme verify` passes."""
    from safetensors import safe_open

    from drinkme.codec.pack import REMAINDER_FILE, embedded_dir, verify_pack

    model, pack, _ = packed
    meta = _meta(pack)
    tower = sorted(k for k in _headers(model) if k.startswith(TOWER_PREFIXES))
    assert any(k.startswith("model.embed_vision.") for k in tower)
    packed_names = sorted(n for n in meta["tensors"] if n.startswith(TOWER_PREFIXES))
    assert packed_names == [f"model.vision_tower.encoder.layers.0.mlp.{p}_proj.linear"
                            for p in ("down", "gate", "up")]
    with safe_open(os.path.join(embedded_dir(pack), REMAINDER_FILE), "pt") as f:
        raw = sorted(k for k in f.keys() if k.startswith(TOWER_PREFIXES))
    assert raw == [k for k in tower if k[:-len(".weight")] not in packed_names]
    assert {"model.vision_tower.std_bias", "model.vision_tower.std_scale",
            "model.vision_tower.patch_embedder.position_embedding_table",
            "model.embed_vision.embedding_projection.weight"} <= set(raw)
    block = meta["vision"]
    assert block["tower"] == "model.vision_tower"
    assert block["paths"] == ["model.vision_tower", "model.embed_vision"]
    assert block["imageTokenId"] == SOFT
    assert block["tensorCount"] == len(tower) and block["packedTensorCount"] == 3
    assert block["residentBytes"] < meta["residentBytes"]
    verify_pack(pack, progress=_quiet)


def test_the_served_tower_is_the_references_bit_for_bit_offline(packed, tmp_path, monkeypatch):
    """Loaded offline from the pack alone: the tower sits where the class
    builds it with its packed Linears, and one image's features are
    transformers' get_image_features' bit for bit (the CPU compressed arm
    decodes the exact bf16 weights; the pixels go in as float32). The
    stock arm gives the same features and the same greedy answer, and the
    engine's prompt count is generate()'s."""
    import shutil

    from drinkme import arms
    from drinkme.codec.swap import CompressedLinear
    from drinkme.serving import checkpoint

    model, pack, want = packed
    stock = _load_stock(model)
    assert stock.vision is not None and isinstance(stock._tower, vision.Gemma4Tower)
    assert type(stock.model.model.vision_tower.encoder.layers[0].mlp.up_proj.linear) \
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
    assert eng.vision.max_pixels == PACK_CAP and eng._tower.bidirectional
    assert isinstance(eng.model.model.vision_tower.encoder.layers[0].mlp.up_proj.linear,
                      CompressedLinear)
    with torch.inference_mode():
        assert torch.equal(eng._tower.features(eng.model, _pack_image()), want)
    res = _greedy(eng, [_pack_image()])
    assert res.text == stock_text
    assert res.prompt_tokens == eng.count_tokens(
        GenerationRequest(PACK_MSGS, SampleParams(), images=(_pack_image().plan,)))


def test_images_off_prunes_the_tower_and_reads_none_of_it(packed, monkeypatch):
    """DRINKME_VISION=0: gemma's class builds the tower, and the loaders
    prune it (both subtrees None, no parameter of it left), never open its
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
    assert eng.model.model.vision_tower is None and eng.model.model.embed_vision is None
    assert not arms.has_vision_tower(eng.model, cfg)
    assert not any(n.startswith(TOWER_PREFIXES) for n, _ in eng.model.named_parameters())
    assert not any(n.startswith(TOWER_PREFIXES) for n in opened if n)
    assert _greedy(eng).completion_tokens == 6
    with pytest.raises(ValueError, match=r"cannot read images on this server "
                                         r"\(disabled by DRINKME_VISION=0\)"):
        _greedy(eng, [_pack_image()])
    assert (fit.served_resident_bytes(meta, False)
            == meta["residentBytes"] - meta["vision"]["residentBytes"])
    stock = _load_stock(model, DRINKME_VISION=0)
    assert stock.vision is None and not arms.has_vision_tower(stock.model, cfg)
    assert not any(n.startswith(TOWER_PREFIXES) for n, _ in stock.model.named_parameters())
    assert _greedy(stock).text == _greedy(eng).text


def _one_nan(out):
    out = out.clone()
    out[..., 0, 0] = float("nan")
    return out


# Wrong attention kernels, each as what it does to a sound kernel's output
# ([B, H, N, D]): the negative controls of the vision gates' --fp32-check
# and of vision.attention_error. AOTriton at head 72 on gfx1151 was off by
# up to 1,240 on outputs under 13, or NaN (serving/vision.py HEAD_ALIGN).
BROKEN = {
    "x3": lambda out: out * 3.0,
    "permuted rows": lambda out: out.roll(1, dims=-2),
    "noise": lambda out: out + torch.randn(out.shape, generator=torch.Generator().manual_seed(0),
                                           dtype=out.dtype),
    "one NaN": _one_nan,
}


def broken_sdpa(kind: str):
    """F.scaled_dot_product_attention made wrong by BROKEN[kind]."""
    real = torch.nn.functional.scaled_dot_product_attention
    return lambda *a, **k: BROKEN[kind](real(*a, **k))


def gate_negative_controls(gate, run, verdict, monkeypatch) -> dict:
    """The vision gate's tower step (`run`, with --fp32-check) under each
    BROKEN kernel: the wrong kernel is the same on every arm, the reference
    included, so the byte pin still PASSes; fp32_verdict FAILs, and on every
    arm because its attention calls do (not only because the features went
    NaN). Returns each control's VERDICT."""
    got = {}
    for kind in BROKEN:
        with monkeypatch.context() as m:
            m.setattr(torch.nn.functional, "scaled_dot_product_attention", broken_sdpa(kind))
            run()
        v = got[kind] = verdict()
        assert v["verdict"] == "PASS" and v["fp32_verdict"] == "FAIL", kind
        assert all(s["calls"] and s["failed"] == s["calls"]
                   for s in v["attention_vs_fp32"].values()), kind
    return got


def test_the_acceptance_gate_passes_on_the_toy(packed, tmp_path, capsys, monkeypatch):
    """bench/gemma_vision_gate.py, the GPU acceptance's tool, on the CPU
    over the toy pack: the tower alone is transformers' get_image_features
    (with drinkme's attention route, as on the GPU) byte for byte on both
    arms, and every attention call of every arm is within the boot
    self-test's tolerance of fp32 (--fp32-check); each BROKEN kernel leaves
    the byte pin passing and fails the fp32 check. The first-token row of
    an image prompt is transformers' forward's on the stock arm and the
    stock arm's on the compressed one, with the same greedy answer."""
    import sys
    import types

    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "bench"))
    import gemma_vision_gate as gate

    model, pack, _ = packed

    def verdict():
        line = [ln for ln in capsys.readouterr().out.splitlines() if ln.startswith("VERDICT ")]
        return json.loads(line[-1][8:])

    common = dict(snap=model, pack=pack, size="96x64", device="cpu", need_gb=0)
    gate.tower(types.SimpleNamespace(**common, max_pixels=PACK_CAP, reps=2, fp32_check=True))
    v = verdict()
    assert v["verdict"] == "PASS" and len(set(v["sha256"].values())) == 1
    assert v["tokens"] == _pack_image().tokens and v["padded_patches"] == 70 * 9
    assert v["bounded_modules"] == 1 and v["fp32_verdict"] == "PASS"  # the one ViT layer
    # one call per arm, masked: the toy image leaves padding patches
    assert {arm: (s["calls"], s["masked"]) for arm, s in v["attention_vs_fp32"].items()} == {
        "reference": (1, 1), "stock": (1, 1), "compressed": (1, 1)}
    assert all(r["finite"] for r in v["vs_fp32_cpu"].values())
    gate_negative_controls(gate, lambda: gate.tower(types.SimpleNamespace(
        **common, max_pixels=PACK_CAP, reps=1, fp32_check=True)), verdict, monkeypatch)
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
    assert v["greedy_divergence"] == {"first_difference": None, "near_tie": None}
    for arm in ("stock", "compressed"):  # every emitted position's pick, at its row's top
        r = torch.load(os.path.join(tmp_path, f"{arm}.pt"))  # (ties included: the toy has one)
        assert len(r["greedy"]) == 6
        assert all(dict(zip(ids, vals))[pick] == vals[0] for pick, ids, vals in r["greedy"])


def _receipt(picks, rows, text):
    """A row step's receipt with these greedy picks and, per position, a
    top list {token: logit} (the rest of the receipt the same on both)."""
    greedy = [(p, list(r), list(r.values())) for p, r in zip(picks, rows)]
    return {"ids": [1, 2, 3], "prompt_tokens": 3, "text": text, "greedy": greedy,
            "reference_row": torch.arange(10.0), "drinkme_row": torch.arange(10.0)}


@pytest.mark.parametrize("stock_gap,compressed_gap,passes", [
    (0.125, 0.0, True),      # gemma-4-31B's own: 23.25 vs 23.125, and an exact tie
    (0.125, 0.125, True),    # adjacent bf16 values on both arms
    (0.25, 0.0, False),      # two steps on one arm: the row could tell them apart
    (None, 0.0, False)])     # the other arm's pick is not in this arm's top
def test_the_gates_compare_passes_a_near_tie_and_fails_a_margin(stock_gap, compressed_gap,
                                                                passes, tmp_path, capsys):
    """gemma_vision_gate.compare (Glimmer's too): greedy answers that part
    at a position where, on each arm, the two picks' logits are at most one
    bf16 step apart at the top (0.125 at 23) PASS; a wider margin on either
    arm, or a pick outside the other's top, FAILs. Equal answers PASS
    whatever the rows."""
    import sys
    import types

    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "bench"))
    import gemma_vision_gate as gate

    same = {5: 30.0, 6: 20.0}
    s_row = {7: 23.25, 8: 23.25 - stock_gap} if stock_gap is not None else {7: 23.25, 9: 1.0}
    c_row = {8: 23.125, 7: 23.125 - compressed_gap}
    s = _receipt([5, 7, 6], [same, s_row, same], "a b c")
    c = _receipt([5, 8, 6], [same, c_row, same], "a d c")
    for arm, r in (("stock", s), ("compressed", c)):
        torch.save(r, os.path.join(tmp_path, f"{arm}.pt"))
    gate.compare(types.SimpleNamespace(out=str(tmp_path)))
    v = json.loads(capsys.readouterr().out.split("VERDICT ", 1)[1])
    assert v["greedy_divergence"]["first_difference"] == 1
    assert v["greedy_divergence"]["tokens"] == [7, 8]
    assert (v["verdict"] == "PASS") == passes and v["greedy_divergence"]["near_tie"] == passes
    c["text"], c["greedy"] = s["text"], s["greedy"]
    torch.save(c, os.path.join(tmp_path, "compressed.pt"))
    gate.compare(types.SimpleNamespace(out=str(tmp_path)))
    assert json.loads(capsys.readouterr().out.split("VERDICT ", 1)[1])["verdict"] == "PASS"


def test_the_tables_keep_the_tree_the_bench_measures():
    """skeleton(cfg) and ckpt_to_skel(cfg) with no say (the bench's arms)
    are the tree gemma's class builds, tower included, as before; True
    keeps it, False prunes it and skips its keys. The text keys are never
    renamed."""
    from drinkme import arms

    cfg = _pack_config()
    bench, on, off = arms.skeleton(cfg), arms.skeleton(cfg, vision=True), \
        arms.skeleton(cfg, vision=False)
    assert arms.vision_tower_paths(cfg) == ("model.vision_tower", "model.embed_vision")
    assert arms.has_vision_tower(bench, cfg) and arms.has_vision_tower(on, cfg)
    assert not arms.has_vision_tower(off, cfg) and off.model.embed_vision is None
    key, proj = ("model.vision_tower.encoder.layers.0.mlp.up_proj.linear.weight",
                 "model.embed_vision.embedding_projection.weight")
    for vis_ in (None, True):
        t = arms.ckpt_to_skel(cfg, vision=vis_)
        assert t(key) == key and t(proj) == proj
    t_off = arms.ckpt_to_skel(cfg, vision=False)
    assert t_off(key) is None and t_off(proj) is None
    text = "model.language_model.layers.0.mlp.up_proj.weight"
    assert all(arms.ckpt_to_skel(cfg, vision=v)(text) == text for v in (None, True, False))
    assert arms.attach_vision_tower(bench, cfg) == "model.vision_tower"  # a no-op
    with pytest.raises(ValueError, match="built by its own class"):
        arms.attach_vision_tower(arms.skeleton(cfg, vision=False), cfg)


def test_a_text_model_it_cannot_splice_into_is_refused_by_name(tmp_path):
    """Per-layer embeddings (the E-models) and a bidirectional setting other
    than "vision" get no images, by name; the image processor and the
    tower must agree on the patch size and the pooling."""
    from drinkme.serving.engines import _vision_for

    cfg = config(patch=16)  # the class defaults' patch: no processor config here
    vis, tower, why = _vision_for(cfg, str(tmp_path), "toy")
    assert why is None and tower.bidirectional and vis.preprocessor.wrap == 2
    assert (tower.boi, tower.eoi, tower.paths) == (VOCAB["<|image>"], VOCAB["<image|>"],
                                                  ("model.vision_tower", "model.embed_vision"))
    ple = config(patch=16)
    ple.text_config.hidden_size_per_layer_input = 8
    assert _vision_for(ple, str(tmp_path), "toy")[2] == \
        "this server has no image input for gemma4 with per-layer embeddings"
    alls = config(patch=16)
    alls.text_config.use_bidirectional_attention = "all"
    assert "use_bidirectional_attention='all'" in _vision_for(alls, str(tmp_path), "toy")[2]
    causal = config(patch=16)
    causal.text_config.use_bidirectional_attention = None
    assert not _vision_for(causal, str(tmp_path), "toy")[1].bidirectional
    (tmp_path / "preprocessor_config.json").write_text(json.dumps({"pooling_kernel_size": 2}))
    with pytest.raises(ValueError, match="pooling_kernel_size=2"):
        _vision_for(config(patch=16), str(tmp_path), "toy")
