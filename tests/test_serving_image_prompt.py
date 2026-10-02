"""Images through the engine: the prompt expanded, keyed and
positioned by serving/image_prompt.py, the tower's output spliced in per
prefill span, and M-RoPE's `pos + delta` on every forward past the prompt.
CPU, float32, no downloads.

THE REFERENCE is transformers' own Qwen3_5ForConditionalGeneration
(`forward(input_ids, pixel_values, image_grid_thw, mm_token_type_ids)`),
built tiny and random: 2 text layers (one gated DeltaNet, one full
attention) and a 2-block ViT. The served tree is what drinkme builds: the
text-only Qwen3_5ForCausalLM with the reference's weights, and the tower as
a sibling at `model.visual`. Every check compares the two paths over the
same weights, never a fixed transcript. The images are real PNGs through
serving/vision.py's Qwen3.5 preprocessor at a toy patch size (4 px).

The pins:
  * end to end: prefill logits (whole and chunked at 7 tokens) and the
    first decode steps equal the reference's; the engine's greedy
    transcript is the reference's greedy, on every speculation branch;
  * the prefix cache: two same-size images never share a slot, the same
    image resent does, the tower is skipped inside the reused prefix, and
    the cold tier's block chain tells the two apart;
  * a text request is unchanged: no position_ids, no inputs_embeds, the
    same forwards;
  * chunked prefill never cuts an image run;
  * pixel_values are released once the tower has read them.
"""

from __future__ import annotations

import base64
import copy
import io

import numpy as np
import pytest
import torch

from drinkme.serving import mtp, prefill, slotstore, vision
from drinkme.serving.engine import GenerationRequest, SampleParams, complete
from drinkme.serving.engines import HFEngine
from drinkme.serving.image_prompt import ImagePrompt, step_rows

from test_serving_mtp import _fill_random, _torch_chunk_on_cpu  # noqa: F401 — fixtures by name
from test_serving_prefill_bound import env

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")

WORDS = ["hello", "world", "the", "quick", "brown", "fox", "what", "is", "in", "this",
         "picture", "and", "user", "assistant", ":"]
SPECIAL = ["<|vision_start|>", "<|image_pad|>", "<|vision_end|>"]
TEMPLATE = (
    "{% for m in messages %}{{ m['role'] }} : {% if m['content'] is string %}"
    "{{ m['content'] }}{% else %}{% for p in m['content'] %}{% if p['type'] == 'image' %}"
    "<|vision_start|><|image_pad|><|vision_end|>{% else %}{{ p['text'] }} {% endif %}"
    "{% endfor %}{% endif %} {% endfor %}{% if add_generation_prompt %}assistant :{% endif %}")


def _vocab() -> dict:
    vocab = {"<unk>": 0, "<pad>": 1, "</s>": 2}
    for w in WORDS + SPECIAL:
        vocab[w] = len(vocab)
    while len(vocab) < 96:
        vocab[f"tok{len(vocab)}"] = len(vocab)
    return vocab


VOCAB = _vocab()
PAD = VOCAB["<|image_pad|>"]


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


def config():
    """A tiny Qwen3.5 with a vision config, shaped like the 27B's (M-RoPE
    sections interleaved, one full-attention layer per DeltaNet one)."""
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config

    return Qwen3_5Config(
        text_config=dict(
            vocab_size=len(VOCAB), hidden_size=64, intermediate_size=128,
            num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
            head_dim=32, linear_key_head_dim=16, linear_value_head_dim=16,
            linear_num_key_heads=2, linear_num_value_heads=4, linear_conv_kernel_dim=4,
            layer_types=["linear_attention", "full_attention"], max_position_embeddings=512,
            rope_parameters={"rope_type": "default", "rope_theta": 10000.0,
                             "mrope_section": [2, 1, 1], "mrope_interleaved": True,
                             "partial_rotary_factor": 0.25},
            tie_word_embeddings=False, eos_token_id=None, pad_token_id=None),
        vision_config=dict(depth=2, hidden_size=32, intermediate_size=64, num_heads=2,
                           in_channels=3, patch_size=4, spatial_merge_size=2,
                           temporal_patch_size=2, out_hidden_size=64,
                           num_position_embeddings=16),
        image_token_id=PAD, video_token_id=VOCAB["tok40"],
        vision_start_token_id=VOCAB["<|vision_start|>"],
        vision_end_token_id=VOCAB["<|vision_end|>"], tie_word_embeddings=False)


def preprocessor():
    return vision.QwenVLPreprocessor("qwen3_5", patch_size=4, temporal_patch_size=2,
                                     merge_size=2, min_pixels=64, max_pixels=64 * 64)


VIS = vision.Vision(preprocessor(), max_pixels=64 * 64)
TOWER = vision.Tower("qwen3_5", "model.visual", PAD, 2)


def png(w: int, h: int, seed: int) -> str:
    a = np.random.default_rng(seed).integers(0, 256, size=(h, w, 3), dtype=np.uint8)
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(a).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def image(w: int, h: int, seed: int):
    """A fresh PreparedImage (the engine releases its pixels once read)."""
    return VIS.prepare(vision.parse_base64(png(w, h, seed), "image/png", where="test"))


@pytest.fixture(scope="module")
def toy():
    """(reference, served model, tokenizer, head): one build per module."""
    from transformers.models.qwen3_5.modeling_qwen3_5 import (
        Qwen3_5ForCausalLM, Qwen3_5ForConditionalGeneration)

    cfg = config()
    torch.manual_seed(0)
    ref = Qwen3_5ForConditionalGeneration(cfg).eval().float()
    model = Qwen3_5ForCausalLM(cfg.text_config).eval().float()
    text = {k.replace("model.language_model.", "model."): v
            for k, v in ref.state_dict().items() if not k.startswith("model.visual.")}
    model.load_state_dict(text, strict=True)
    model.model.visual = copy.deepcopy(ref.model.visual)
    for p in list(ref.parameters()) + list(model.parameters()):
        p.requires_grad_(False)
    head = _fill_random(mtp.MTPHead(cfg.text_config, model), seed=1).eval()
    mtp.install_deltanet_capture(model)
    return ref, model, _tokenizer(), head


def engine(toy, *, head=False, slots=1, chunk=0, vision_on=True, **kw):
    _ref, model, tok, h = toy
    with env(DRINKME_PREFILL_CHUNK=chunk, DRINKME_PREFIX_SLOTS=slots, DRINKME_SLOT_DIR="off"):
        return HFEngine(model, tok, model_id="toy", arm="test", meta={}, ctx=256,
                        mtp_head=h if head else None,
                        vision=VIS if vision_on else None,
                        tower=TOWER if vision_on else None, **kw)


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
    pv = torch.cat([torch.tensor(np.array(i.pixel_values)) for i in images])
    grid = torch.tensor([i.grid_thw for i in images])
    t = torch.tensor([ids])
    with torch.inference_mode():
        return ref(input_ids=t, pixel_values=pv, image_grid_thw=grid,
                   mm_token_type_ids=(t == PAD).int()).logits[0]


def ref_greedy(ref, ids, images, n):
    """The reference's greedy continuation, one full forward per token."""
    ids = list(ids)
    out = []
    for _ in range(n):
        t = int(ref_logits(ref, ids, images)[-1].argmax())
        out.append(t)
        ids.append(t)
    return out


MSGS = [user("what is in", None, "the quick brown", None, "hello world")]


def two_images():
    return [image(16, 24, 1), image(32, 16, 2)]


# ------------------------------------------------------ END TO END --


@pytest.mark.parametrize("chunk", [0, 7])
def test_prefill_logits_are_the_references(toy, chunk):
    """Every prompt row's logits (the hidden path, lm_head over each span)
    and the last row's, whole and chunked at 7 tokens, against
    Qwen3_5ForConditionalGeneration over the same weights."""
    ref, model, tok, _ = toy
    imgs = two_images()
    ip = ImagePrompt(rendered(tok, MSGS), imgs, TOWER, model)
    want = ref_logits(ref, ip.ids, two_images())
    hs = []
    with torch.inference_mode():
        cache = engine(toy)._cache(64)
        last = prefill.run(model, ip.ids, 0, cache, "cpu", chunk, hidden=True,
                           on_hidden=lambda h, a, b: hs.append((h, a, b)), image=ip)
        every = model.lm_head(torch.cat([h for h, _, _ in hs], 1))[0]
    torch.testing.assert_close(every, want, atol=2e-5, rtol=1e-5)
    torch.testing.assert_close(last, want[-1], atol=2e-5, rtol=1e-5)
    # chunked at 7: a run that fits the chunk is whole (its span moves back
    # to the image's start), and the one longer than the chunk is cut like
    # text, so no forward is longer than the chunk
    if chunk:
        runs = [(s, e) for s, e, _ in ip.runs]
        assert sorted(e - s for s, e in runs) == [6, 8]
        assert [(a, b) for _, a, b in hs] == prefill.spans(0, ip.n, 7, ip.whole(0, 7))
        assert len(hs) > 2 and all(b - a <= 7 for _, a, b in hs)
        for s, e in runs:
            assert any(s < a < e for _, a, _b in hs) == (e - s > 7)


def test_a_bidirectional_tower_keeps_every_run_whole(toy):
    """whole(): on a tower whose image tokens are causal (Qwen3.5) a run
    longer than the chunk is left out, so spans cut it; one that attends
    bidirectionally within an image keeps every run, however long, and a
    run that starts before `start` is clipped to it. The same one field,
    Tower.bidirectional, decides cuts(): a reuse point inside a run is
    refused on the bidirectional tower only."""
    import dataclasses

    _ref, model, tok, _ = toy
    causal = ImagePrompt(rendered(tok, MSGS), two_images(), TOWER, model)
    runs = [(s, e) for s, e, _ in causal.runs]
    assert not TOWER.bidirectional
    assert causal.whole(0, 7) == [r for r in runs if r[1] - r[0] <= 7]
    assert causal.whole(0) == causal.whole(0, 8) == runs
    bidi = ImagePrompt(rendered(tok, MSGS), two_images(),
                       dataclasses.replace(TOWER, bidirectional=True), model)
    assert bidi.whole(0, 7) == runs
    s, e = max(runs, key=lambda r: r[1] - r[0])
    assert causal.whole(s + 2, 7) == [(s + 2, e)]  # 6 left: fits the chunk
    assert bidi.whole(s + 1, 2) == [(s + 1, e)]
    assert bidi.cuts(s + 1) and not causal.cuts(s + 1)
    assert not bidi.cuts(s) and not bidi.cuts(e)


def test_a_decode_step_takes_pos_plus_delta(toy):
    """The step after the prompt sits at [p, p+delta x3]: equal to the
    reference over prompt + that token, and NOT what plain positions give
    (the negative control: the delta matters)."""
    ref, model, tok, _ = toy
    ip = ImagePrompt(rendered(tok, MSGS), two_images(), TOWER, model)
    assert ip.delta < 0
    with torch.inference_mode():
        cache = engine(toy)._cache(64)
        first = prefill.run(model, ip.ids, 0, cache, "cpu", 0, image=ip)
        nxt = int(first.argmax())
        p = torch.tensor([ip.n])
        got = model(torch.tensor([[nxt]]), past_key_values=cache, use_cache=True,
                    cache_position=p, position_ids=step_rows(p, ip.delta)).logits[0, -1]
        cache2 = engine(toy)._cache(64)
        prefill.run(model, ip.ids, 0, cache2, "cpu", 0, image=ImagePrompt(
            rendered(tok, MSGS), two_images(), TOWER, model))
        plain = model(torch.tensor([[nxt]]), past_key_values=cache2, use_cache=True,
                      cache_position=p).logits[0, -1]
    want = ref_logits(ref, ip.ids + [nxt], two_images())[-1]
    torch.testing.assert_close(got, want, atol=2e-5, rtol=1e-5)
    assert (plain - want).abs().max() > 1e-4


@pytest.mark.parametrize("spec,chunk", [("off", 0), ("off", 7), ("ngram", 0), ("ngram", 7),
                                        ("auto", 0), ("auto", 7)])
def test_the_engines_greedy_is_the_references(toy, spec, chunk):
    """generate() end to end, serial and speculating (the n-gram lookup;
    the MTP head, whose rope rows follow the trunk's), whole and chunked:
    the reference's greedy token for token. The toy is float32 and far
    from ties (spec_agree's margin would say so if it were not)."""
    ref, model, tok, _ = toy
    n = 10
    ids = ImagePrompt(rendered(tok, MSGS), two_images(), TOWER, model).ids
    want = tok.decode(ref_greedy(ref, ids, two_images(), n))
    eng = engine(toy, head=spec == "auto", chunk=chunk)
    got = ask(eng, MSGS, two_images(), spec, n)
    assert got.prompt_tokens == len(ids)
    assert got.text.split() == want.split()


def test_the_mtp_head_takes_the_prompts_positions_and_delta_past_it(toy, monkeypatch):
    """The head's entries inside the prompt get the trunk's M-RoPE rows,
    and every entry past it sits at index + delta."""
    _ref, model, tok, head = toy
    seen = []
    real = type(head).run

    def spy(self, hidden, token_ids, first_entry, positions=None):
        seen.append((first_entry, None if positions is None else positions.clone(),
                     token_ids.shape[1]))
        return real(self, hidden, token_ids, first_entry, positions=positions)

    monkeypatch.setattr(type(head), "run", spy)
    eng = engine(toy, head=True)
    ip = ImagePrompt(rendered(tok, MSGS), two_images(), TOWER, model)
    ask(eng, MSGS, two_images(), "auto", 6)
    seeded = [s for s in seen if s[1] is not None]
    assert seeded, "the prompt's entries were seeded without positions"
    a, pos, t = seeded[0]
    assert torch.equal(pos[:, 0].cpu(), torch.from_numpy(ip.pos4[1:, a:a + t]))
    past = [s for s in seen if s[1] is None]
    assert past and min(first for first, _, _ in past) == ip.n - 1 + ip.delta


# ----------------------------------------------------- PREFIX CACHE --


class TowerCalls:
    """How many times the served tree's tower ran."""

    def __init__(self, model):
        self.n = 0
        self.h = model.get_submodule(TOWER.path).register_forward_hook(self._hit)

    def _hit(self, *a):
        self.n += 1

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.h.remove()


def test_two_same_size_images_never_share_a_slot(toy):
    """THE correctness trap: two different images of one size
    render to the same ids. The second request must not reuse the first's
    KV: it prefills cold, and answers exactly what a fresh engine answers
    about its own image."""
    _ref, model, tok, _ = toy
    msgs = [user("what is in this picture", None)]
    eng = engine(toy)
    a = ask(eng, msgs, [image(16, 24, 11)])
    b = ask(eng, msgs, [image(16, 24, 12)])
    assert a.cached_tokens == 0
    start = rendered(tok, msgs).index(PAD)
    assert b.cached_tokens <= start  # nothing past the image's first row
    fresh = ask(engine(toy), msgs, [image(16, 24, 12)])
    assert b.text == fresh.text
    ids = ImagePrompt(rendered(tok, msgs), [image(16, 24, 11)], TOWER, model).ids
    assert ids == ImagePrompt(rendered(tok, msgs), [image(16, 24, 12)], TOWER, model).ids


def test_the_same_image_resent_reuses_its_kv_and_skips_the_tower(toy):
    """A conversation that carries its image forward: the second turn
    extends the first's slot past the image, the tower does not run for it
    again, and the answer is the one a cold engine gives."""
    _ref, model, _tok, _ = toy
    eng = engine(toy)
    hist = [user("what is in this picture", None)]
    # short turns: a random toy can emit a special id (<unk>), which the
    # delivered text drops, and a resent history then no longer extends
    first = ask(eng, hist, [image(16, 24, 21)], n=4)
    hist = hist + [{"role": "assistant", "content": first.text}, user("and the fox")]
    with TowerCalls(model) as calls:
        warm = ask(eng, hist, [image(16, 24, 21)])
    assert calls.n == 0
    assert warm.cached_tokens > 0
    with TowerCalls(model) as calls:
        cold = ask(engine(toy), hist, [image(16, 24, 21)])
    assert calls.n == 1
    assert warm.text == cold.text


def test_the_cold_tiers_keys_tell_two_images_apart(toy):
    """slotstore keys on block_chain(key_ids): same-size images differ, the
    same image agrees, and a key is never a vocabulary id."""
    _ref, model, tok, _ = toy
    body = user("what is", None, " ".join(["hello world the quick brown fox"] * 100))
    r = rendered(tok, [body])
    one, two = image(32, 32, 31), image(32, 32, 32)
    k1 = ImagePrompt(r, [one], TOWER, model).key_ids
    k2 = ImagePrompt(r, [two], TOWER, model).key_ids
    k1b = ImagePrompt(r, [image(32, 32, 31)], TOWER, model).key_ids
    assert len(k1) > 2 * slotstore.BLOCK_TOKENS
    chain1, chain2 = slotstore.block_chain(k1), slotstore.block_chain(k2)
    assert len(chain1) == len(chain2) >= 2
    assert all(a != b for a, b in zip(chain1, chain2))  # the image is in block 0
    assert chain1 == slotstore.block_chain(k1b)
    assert slotstore.prefix_key(k1) != slotstore.prefix_key(k2)
    assert {k for k in k1 if k < 0} == {one.prefix_key} and PAD not in k1
    assert len(k1) == len(ImagePrompt(r, [image(32, 32, 31)], TOWER, model).ids)


def test_a_restore_that_lands_inside_an_image_prefills_the_rest_of_it(toy):
    """The cold tier restores at 256-token blocks, which can end inside an
    image run: the tower runs for the whole image and the rows from lcp on
    are prefilled (exact on Qwen3.5, causal). Driven directly: a cache
    holding the prompt up to the middle of the second image, then prefill
    from there, against the whole prompt cold."""
    ref, model, tok, _ = toy
    ip = ImagePrompt(rendered(tok, MSGS), two_images(), TOWER, model)
    s, e, _ = ip.runs[1]
    mid = (s + e) // 2
    with torch.inference_mode():
        cache = engine(toy)._cache(64)
        head = ImagePrompt(rendered(tok, MSGS), two_images(), TOWER, model)
        prefill.run(model, head.ids[:mid], 0, cache, "cpu", 0,
                    image=_Truncated(head, mid))
        rest = ImagePrompt(rendered(tok, MSGS), two_images(), TOWER, model)
        rest.begin(mid)
        assert rest.skipped == 1  # the first image is wholly inside the prefix
        got = prefill.run(model, rest.ids, mid, cache, "cpu", 5, image=rest)
    assert rest.tower_runs == 1
    torch.testing.assert_close(got, ref_logits(ref, ip.ids, two_images())[-1],
                               atol=2e-5, rtol=1e-5)


class _Truncated:
    """An ImagePrompt's first `n` rows only, as a prefill of ids[:n] sees
    them: the stand-in for a slot that holds a prompt cut mid-image."""

    def __init__(self, ip, n):
        self.ip, self.n = ip, n

    def whole(self, start, chunk=0):
        return [(max(s, start), min(e, self.n)) for s, e in self.ip.whole(start, chunk)
                if s < self.n]

    def inputs(self, base, a, b, device, cache=None):
        return self.ip.inputs(base, a, b, device, cache)


# ----------------------------------------------- TEXT IS UNCHANGED --


class Calls:
    """Every forward the trunk (the decoder, and the CausalLM wrapper) is
    handed, as (positional input kind, sorted kwarg names)."""

    def __init__(self, model):
        self.calls = []
        self.hooks = [m.register_forward_pre_hook(self._rec, with_kwargs=True)
                      for m in (model, model.get_decoder())]

    def _rec(self, mod, args, kwargs):
        given = {k for k, v in kwargs.items() if v is not None}
        ids = args[0] if args else kwargs.get("input_ids")
        self.calls.append((type(mod).__name__, type(ids).__name__,
                           tuple(sorted(given - {"input_ids"}))))

    def __enter__(self):
        return self

    def __exit__(self, *a):
        for h in self.hooks:
            h.remove()


@pytest.mark.parametrize("spec,chunk", [("off", 0), ("off", 7), ("ngram", 7), ("auto", 7)])
def test_a_text_request_makes_the_calls_it_always_made(toy, spec, chunk, monkeypatch):
    """No position_ids, no inputs_embeds, the ids as the one positional
    input, on every forward of a text generation (prefill, decode, draft
    verify, re-arm) — and no ImagePrompt is ever built for it."""
    _ref, model, _tok, _ = toy
    import drinkme.serving.image_prompt as ip_mod

    def boom(*a, **k):
        raise AssertionError("a text request built an ImagePrompt")

    monkeypatch.setattr(ip_mod, "ImagePrompt", boom)
    eng = engine(toy, head=spec == "auto", chunk=chunk)
    msgs = [{"role": "user", "content": "the quick brown fox what is this picture " * 3}]
    with Calls(model) as rec:
        with env(DRINKME_SPEC=spec):
            complete(eng, GenerationRequest(msgs, SampleParams(temperature=0.0, max_tokens=12)))
    assert rec.calls
    for _mod, first, kwargs in rec.calls:
        assert first == "Tensor", rec.calls
        assert "position_ids" not in kwargs and "inputs_embeds" not in kwargs, rec.calls
        assert set(kwargs) <= {"past_key_values", "use_cache", "cache_position",
                               "attention_mask", "logits_to_keep"}, kwargs


def test_count_tokens_counts_each_image_as_its_tokens(toy):
    """count_tokens == the prompt generate() runs, with images: a plan
    (vision.Vision.count) counts like the prepared image."""
    _ref, model, tok, _ = toy
    eng = engine(toy)
    imgs = two_images()
    req = GenerationRequest(MSGS, SampleParams(max_tokens=2), images=tuple(imgs))
    n = ImagePrompt(rendered(tok, MSGS), imgs, TOWER, model).n
    assert eng.count_tokens(req) == n
    plans = tuple(VIS.count(vision.parse_base64(png(w, h, s), "image/png", where="t"))
                  for w, h, s in ((16, 24, 1), (32, 16, 2)))
    assert eng.count_tokens(GenerationRequest(MSGS, SampleParams(), images=plans)) == n
    assert complete(eng, req).prompt_tokens == n
    text = [{"role": "user", "content": "hello world"}]
    assert eng.count_tokens(GenerationRequest(text, SampleParams())) == len(rendered(tok, text))


# ------------------------------------------------ SPANS, REFUSALS --


def test_spans_never_cut_a_whole_run():
    assert prefill.spans(0, 30, 7, [(6, 12), (17, 25)]) == [
        (0, 6), (6, 13), (13, 17), (17, 25), (25, 30)]
    # a run longer than a chunk is its own span
    assert prefill.spans(0, 20, 4, [(2, 12)]) == [(0, 2), (2, 12), (12, 16), (16, 20)]
    # a run that starts the span (a restore landed in it) is whole from there
    assert prefill.spans(5, 20, 4, [(5, 11)]) == [(5, 11), (11, 15), (15, 19), (19, 20)]
    # no runs, or a prompt inside one chunk: the text path's spans exactly
    assert prefill.spans(0, 30, 7, []) == prefill.spans(0, 30, 7)
    assert prefill.spans(3, 9, 7, [(4, 6)]) == [(3, 9)]


def test_the_rendered_placeholders_must_match_the_images(toy):
    _ref, model, tok, _ = toy
    with pytest.raises(ValueError, match="2 image placeholders for 1 images"):
        ImagePrompt(rendered(tok, MSGS), [image(16, 24, 1)], TOWER, model)
    wrong = vision.Tower("gemma4", "model.visual", PAD, 2)
    with pytest.raises(ValueError, match="prepared for qwen3_5"):
        ImagePrompt(rendered(tok, MSGS), two_images(), wrong, model)


def test_an_engine_without_vision_refuses_images_by_name(toy):
    eng = engine(toy, vision_on=False, vision_reason="disabled by DRINKME_VISION=0")
    assert eng.vision is None
    with pytest.raises(ValueError, match="cannot read images on this server "
                                         r"\(disabled by DRINKME_VISION=0\)"):
        ask(eng, MSGS, two_images())
    assert engine(toy, vision_on=False).vision_reason == "text-only model"
    _ref, model, tok, _ = toy
    with pytest.raises(ValueError, match="both a Vision and a Tower"):
        HFEngine(model, tok, model_id="toy", arm="test", meta={}, ctx=256, vision=VIS)


# ----------------------------------------------------------- LIFETIME --


def test_pixels_are_released_once_the_tower_has_read_them(toy):
    _ref, model, _tok, _ = toy
    eng = engine(toy)
    imgs = two_images()
    ask(eng, MSGS, imgs)
    assert all(i.released for i in imgs)
    assert all(i.digest and i.prefix_key < 0 for i in imgs)  # the keys survive
    with pytest.raises(RuntimeError, match="prepare the image again"):
        TOWER.features(model, imgs[0])


def test_an_image_inside_the_reused_prefix_is_released_without_the_tower(toy):
    _ref, model, _tok, _ = toy
    eng = engine(toy)
    hist = [user("what is in this picture", None)]
    first = ask(eng, hist, [image(16, 24, 41)], n=4)
    hist = hist + [{"role": "assistant", "content": first.text}, user("and the fox")]
    again = image(16, 24, 41)
    with TowerCalls(model) as calls:
        ask(eng, hist, [again])
    assert calls.n == 0 and again.released


def test_one_image_twice_runs_the_tower_once(toy):
    """Two placeholders over the same pixels (one digest): one tower run,
    and the logits are the reference's."""
    ref, model, tok, _ = toy
    a, b = image(16, 24, 51), image(16, 24, 51)
    assert a == b
    ip = ImagePrompt(rendered(tok, MSGS), [a, b], TOWER, model)
    with torch.inference_mode(), TowerCalls(model) as calls:
        got = prefill.run(model, ip.ids, 0, engine(toy)._cache(64), "cpu", 7, image=ip)
    assert calls.n == 1 and a.released and b.released
    want = ref_logits(ref, ip.ids, [image(16, 24, 51), image(16, 24, 51)])[-1]
    torch.testing.assert_close(got, want, atol=2e-5, rtol=1e-5)


def test_the_ngram_proposer_never_proposes_an_image_key():
    from drinkme.serving import ngram

    p = ngram.NgramProposer(tokens=4, hi=2, lo=1)
    ctx = [5, 6, -9, -9, -9, 7, 5, 6]
    p.reset(ctx)
    assert p.propose(4) == []  # the match at 0 is followed by a key: nothing to propose
    p.reset([5, 6, 8, -9, -9, 5, 6])
    assert p.propose(4) == [8]
    assert ngram.brute_force([5, 6, 8, -9, -9, 5, 6], 4, 2, 1) == [8]



def test_the_cold_tier_stores_and_finds_a_slot_by_its_image_keys(toy, tmp_path):
    """A slot whose ids carry an image's prefix key goes to disk and comes
    back under that key (negative ids survive the file and the header's
    tail), and a prompt with a same-size different image finds nothing."""
    from test_serving_slotstore import filled, make_cache, noprime, store

    _ref, model, tok, _ = toy
    body = user("what is", None, " ".join(["hello world the quick brown fox"] * 60))
    r = rendered(tok, [body])
    keys = ImagePrompt(r, [image(32, 32, 61)], TOWER, model).key_ids
    other = ImagePrompt(r, [image(32, 32, 62)], TOWER, model).key_ids
    held = keys[:-3]
    st = store(tmp_path)
    assert st.put(held, 512, filled(512, len(held))) is not None and st.flush(10)
    fresh = store(tmp_path)
    fresh.index()
    got = fresh.lookup(keys, need=100, ctx=8192)
    assert got is not None
    _live, back = fresh.restore(got, make_cache(512), noprime)
    assert back == held and min(back) < 0
    assert fresh.lookup(other, need=100, ctx=8192) is None
