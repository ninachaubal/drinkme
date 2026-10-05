"""Videos through the engine: the template's placeholder expanded into
timestamped frame groups by serving/image_prompt.py, ONE tower pass per
video spliced in group by group, each group positioned by M-RoPE as a
one-frame image, and the fit check charging all of it. CPU, float32, no
downloads.

THE REFERENCE is transformers' own Qwen3_5ForConditionalGeneration
(`forward(input_ids, pixel_values_videos, video_grid_thw,
mm_token_type_ids)`), built tiny and random as
tests/test_serving_image_prompt.py builds it, with a vocabulary that can
spell the timestamps. The prompt it is given is the template's render with
the placeholder replaced by the reference processor's own string
(Qwen3VLProcessor.replace_video_token, which tests/test_serving_video.py
pins video.PreparedVideo.prompt_text to), so the expansion, the features'
order across groups and the positions are all checked against
transformers. The clips are real PyAV-encoded MP4s through serving/
video.py at a toy patch size (4 px).

The pins:
  * the expansion: `<t seconds><|vision_start|>` + run + `<|vision_end|>`
    per group, the template's three ids replaced whole;
  * prefill logits (whole and chunked), with a video alone and beside an
    image, and the greedy transcript, serial and speculating;
  * the tower runs once per video, its pixels are released, and a
    resent video reuses its KV while a different one of the same size
    never does;
  * count_tokens is what generate() runs, and over HTTP the context
    check refuses a video that does not fit, by its real token count.
"""

from __future__ import annotations

import base64
import copy
import json

import numpy as np
import pytest
import torch

from drinkme.serving import mtp, prefill, video, vision
from drinkme.serving.engine import GenerationRequest, SampleParams, complete
from drinkme.serving.engines import HFEngine
from drinkme.serving.image_prompt import Frames, ImagePrompt

from test_serving_mtp import _fill_random, _torch_chunk_on_cpu  # noqa: F401 — fixtures by name
from test_serving_prefill_bound import env

pytestmark = [pytest.mark.filterwarnings("ignore::DeprecationWarning"),
              pytest.mark.skipif(not video.available(),
                                 reason="PyAV (the drinkme[video] extra) is not installed")]

WORDS = ["hello", "world", "the", "quick", "brown", "fox", "what", "happens", "in", "this",
         "video", "and", "picture", "user", "assistant", ":", "<", ">", ".", "seconds",
         *[str(d) for d in range(10)]]
SPECIAL = ["<|vision_start|>", "<|image_pad|>", "<|vision_end|>", "<|video_pad|>"]
TRIPLE = "<|vision_start|><|video_pad|><|vision_end|>"
TEMPLATE = (
    "{% for m in messages %}{{ m['role'] }} : {% if m['content'] is string %}"
    "{{ m['content'] }}{% else %}{% for p in m['content'] %}{% if p['type'] == 'image' %}"
    "<|vision_start|><|image_pad|><|vision_end|>{% elif p['type'] == 'video' %}"
    "<|vision_start|><|video_pad|><|vision_end|>{% else %}{{ p['text'] }} {% endif %}"
    "{% endfor %}{% endif %} {% endfor %}{% if add_generation_prompt %}assistant :{% endif %}")


def _vocab() -> dict:
    vocab = {"<unk>": 0, "<pad>": 1, "</s>": 2}
    for w in WORDS + SPECIAL:
        vocab[w] = len(vocab)
    while len(vocab) < 96:
        vocab[f"tok{len(vocab)}"] = len(vocab)
    return vocab


VOCAB = _vocab()
PAD, VPAD = VOCAB["<|image_pad|>"], VOCAB["<|video_pad|>"]
VS, VE = VOCAB["<|vision_start|>"], VOCAB["<|vision_end|>"]


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


TOK = _tokenizer()


def config():
    """tests/test_serving_image_prompt.py's tiny Qwen3.5, with this file's
    vocabulary and its video token."""
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
        image_token_id=PAD, video_token_id=VPAD, vision_start_token_id=VS,
        vision_end_token_id=VE, tie_word_embeddings=False)


def _encode(text):
    return TOK.encode(text, add_special_tokens=False)


VIDEO_PRE = video.QwenVideoPreprocessor("qwen3_5", patch_size=4, temporal_patch_size=2,
                                        merge_size=2, min_pixels=64, max_pixels=4 * 32 * 32,
                                        max_frames=16)
VIS = vision.Vision(vision.QwenVLPreprocessor("qwen3_5", patch_size=4, temporal_patch_size=2,
                                              merge_size=2, min_pixels=64, max_pixels=64 * 64),
                    max_pixels=64 * 64, video=video.VideoInput(VIDEO_PRE, _encode))


def tower():
    return vision.tower_for("qwen3_5", config(), ("model.visual",))


TOWER = tower()


def clip_video(frames=20, fps=10, w=24, h=16, seed=0) -> video.PreparedVideo:
    """A fresh PreparedVideo of a PyAV-made clip (the engine releases its
    pixels once read)."""
    from test_serving_video import clip

    enc = video.parse_video_url(
        "data:video/mp4;base64," + base64.b64encode(clip(frames, fps, w, h, seed=seed)).decode(),
        where="test")
    return VIS.video.prepare(enc, where="test")


def image(w, h, seed):
    from test_serving_image_prompt import png

    return VIS.prepare(vision.parse_base64(png(w, h, seed), "image/png", where="test"))


@pytest.fixture(scope="module")
def toy():
    """(reference, served model, head): one build per module."""
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
    return ref, model, head


def engine(toy, *, head=False, slots=1, chunk=0, ctx=256):
    _ref, model, h = toy
    with env(DRINKME_PREFILL_CHUNK=chunk, DRINKME_PREFIX_SLOTS=slots, DRINKME_SLOT_DIR="off"):
        return HFEngine(model, TOK, model_id="toy", arm="test", meta={}, ctx=ctx,
                        mtp_head=h if head else None, vision=VIS, tower=TOWER)


def user(*parts):
    """A user turn from text strings, "V" (a video) and "I" (an image)."""
    kind = {"V": {"type": "video"}, "I": {"type": "image"}}
    return {"role": "user", "content": [kind.get(p) or {"type": "text", "text": p}
                                        for p in parts]}


MSGS = [user("what happens in", "V", "and in this picture", "I", "hello world")]
VMSGS = [user("what happens in this video", "V")]


def rendered(msgs):
    out = TOK.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True)
    return out if isinstance(out, list) else out["input_ids"]


def ask(eng, msgs, images=(), videos=(), spec="off", n=8):
    with env(DRINKME_SPEC=spec):
        return complete(eng, GenerationRequest(msgs, SampleParams(temperature=0.0, max_tokens=n),
                                               images=tuple(images), videos=tuple(videos)))


def reference_ids(msgs, videos, images=()) -> list[int]:
    """The prompt as transformers' processor writes it: the render with
    each video's three-id placeholder replaced by its processor string, and
    each image's pad repeated its token count
    (Qwen3VLProcessor.replace_image_token)."""
    text = TOK.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    for v in videos:
        assert TRIPLE in text
        text = text.replace(TRIPLE, " " + v.prompt_text() + " ", 1)
    for img in images:
        text = text.replace("<|image_pad|>", "<|placeholder|>" * img.tokens, 1)
    return _encode(text.replace("<|placeholder|>", "<|image_pad|>"))


def ref_logits(ref, ids, images=(), videos=()):
    t = torch.tensor([ids])
    kw = {"mm_token_type_ids": (t == PAD).int() + 2 * (t == VPAD).int()}
    if images:
        kw["pixel_values"] = torch.cat([torch.tensor(np.array(i.pixel_values)) for i in images])
        kw["image_grid_thw"] = torch.tensor([i.grid_thw for i in images])
    if videos:
        kw["pixel_values_videos"] = torch.cat([torch.tensor(np.array(v.pixel_values))
                                               for v in videos])
        kw["video_grid_thw"] = torch.tensor([v.grid_thw for v in videos])
    with torch.inference_mode():
        return ref(input_ids=t, **kw).logits[0]


def ref_greedy(ref, ids, images, videos, n):
    ids = list(ids)
    out = []
    for _ in range(n):
        t = int(ref_logits(ref, ids, images, videos)[-1].argmax())
        out.append(t)
        ids.append(t)
    return out


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


# ---------------------------------------------------------- THE PROMPT --

def test_the_tower_reads_the_configs_video_ids():
    assert (TOWER.video_token_id, TOWER.vision_start_id, TOWER.vision_end_id) == (VPAD, VS, VE)


def test_the_placeholder_becomes_timestamped_groups(toy):
    """`<t seconds><|vision_start|>` + run + `<|vision_end|>` per temporal
    group, in place of the template's three ids: the reference processor's
    string, tokenized. The key ids carry the video's prefix key on its
    runs and the timestamps' own ids around them."""
    _ref, model, _ = toy
    v = clip_video()
    assert v.groups == 2 and v.group_tokens == 6 and v.timestamps == (0.3, 1.6)
    ip = ImagePrompt(rendered(VMSGS), (), TOWER, model, videos=(v,))
    assert ip.ids == reference_ids(VMSGS, [v])
    assert ip.n == len(rendered(VMSGS)) + v.expansion
    stamp = _encode("<0.3 seconds>")
    first = ip.ids.index(VPAD)
    assert ip.ids[first - len(stamp) - 1:first] == stamp + [VS]
    assert TOK.decode(ip.ids[first - len(stamp) - 1:first]).replace(" ", "") == \
        "<0.3seconds><|vision_start|>"
    assert [e - s for s, e, _f in ip.runs] == [6, 6]
    assert all(isinstance(f, Frames) and f.video is v for _s, _e, f in ip.runs)
    assert [f.row0 for _s, _e, f in ip.runs] == [0, 6]
    assert {k for k in ip.key_ids if k < 0} == {v.prefix_key} and VPAD not in ip.key_ids
    assert [k for k in ip.key_ids if k >= 0] == [i for i in ip.ids if i != VPAD]
    # each group's run is positioned as a one-frame image (the reference
    # splits a video's grid per group)
    assert [f.grid_thw for _s, _e, f in ip.runs] == [(1, 4, 6)] * 2


def test_the_expanded_position_of_a_later_turn_skips_the_whole_video(toy):
    _ref, model, _ = toy
    v = clip_video()
    msgs = VMSGS + [{"role": "assistant", "content": "hello"}, user("the fox")]
    r = rendered(msgs)
    head = TOK.apply_chat_template(msgs[:2], add_generation_prompt=False, tokenize=True)
    head = head if isinstance(head, list) else head["input_ids"]
    assert r[:len(head)] == head
    ip = ImagePrompt(r, (), TOWER, model, videos=(v,))
    assert ip.expanded(len(head)) == len(head) + v.expansion
    assert ip.ids[ip.expanded(len(head)):] == r[len(head):]


def test_a_template_that_does_not_render_the_triple_is_refused(toy):
    _ref, model, _ = toy
    bare = [VOCAB["hello"], VPAD, VOCAB["world"]]
    with pytest.raises(ValueError, match="does not render a video"):
        ImagePrompt(bare, (), TOWER, model, videos=(clip_video(),))
    with pytest.raises(ValueError, match="1 video placeholders for 2 videos"):
        ImagePrompt(rendered(VMSGS), (), TOWER, model, videos=(clip_video(), clip_video()))
    import dataclasses

    no_video = dataclasses.replace(TOWER, video_token_id=None)
    with pytest.raises(ValueError, match="reads no video"):
        ImagePrompt(rendered(VMSGS), (), no_video, model, videos=(clip_video(),))


# ------------------------------------------------------ END TO END --

@pytest.mark.parametrize("chunk", [0, 5])
@pytest.mark.parametrize("which", ["video", "video and image"])
def test_prefill_logits_are_the_references(toy, chunk, which):
    """Every prompt row's logits, whole and chunked at 5 (a group's run of
    6 is then cut like text), against Qwen3_5ForConditionalGeneration over
    the same weights: the tower over the whole video, its rows split
    across the groups in order, each group at its own M-RoPE positions."""
    ref, model, _ = toy
    msgs = VMSGS if which == "video" else MSGS
    imgs = () if which == "video" else (image(16, 24, 1),)
    v = clip_video(30, 10, seed=3)  # 3 groups
    assert v.groups == 3
    ip = ImagePrompt(rendered(msgs), imgs, TOWER, model, videos=(v,))
    assert ip.ids == reference_ids(msgs, [v], imgs)
    want = ref_logits(ref, ip.ids, [image(16, 24, 1)] if imgs else (),
                      [clip_video(30, 10, seed=3)])
    hs = []
    with torch.inference_mode():
        cache = engine(toy)._cache(128)
        last = prefill.run(model, ip.ids, 0, cache, "cpu", chunk, hidden=True,
                           on_hidden=lambda h, a, b: hs.append((h, a, b)), image=ip)
        every = model.lm_head(torch.cat([h for h, _, _ in hs], 1))[0]
    torch.testing.assert_close(every, want, atol=2e-5, rtol=1e-5)
    torch.testing.assert_close(last, want[-1], atol=2e-5, rtol=1e-5)
    assert ip.tower_runs == len(imgs) + 1  # one pass for the whole video
    assert v.released


@pytest.mark.parametrize("spec,chunk", [("off", 0), ("off", 5), ("ngram", 0), ("auto", 5)])
def test_the_engines_greedy_is_the_references(toy, spec, chunk):
    """generate() end to end, serial and speculating: the reference's
    greedy token for token, and prompt_tokens is the expanded prompt."""
    ref, _model, _ = toy
    n = 8
    ids = reference_ids(MSGS, [clip_video(seed=5)], [image(16, 24, 2)])
    want = TOK.decode(ref_greedy(ref, ids, [image(16, 24, 2)], [clip_video(seed=5)], n))
    got = ask(engine(toy, head=spec == "auto", chunk=chunk), MSGS, [image(16, 24, 2)],
              [clip_video(seed=5)], spec, n)
    assert got.prompt_tokens == len(ids)
    assert got.text.split() == want.split()


def test_the_tower_runs_once_per_video_and_its_pixels_go(toy):
    _ref, model, _ = toy
    v = clip_video(30, 10, seed=7)
    with TowerCalls(model) as calls:
        ask(engine(toy), VMSGS, videos=[v], n=2)
    assert calls.n == 1 and v.released


# ----------------------------------------------------- PREFIX CACHE --

def test_two_same_size_videos_never_share_a_slot(toy):
    """Same size, same rate, so the same ids and the same timestamps: the
    second request must not reuse the first's KV past its first video
    token, and answers what a fresh engine answers about its own video."""
    _ref, model, _ = toy
    eng = engine(toy)
    a = ask(eng, VMSGS, videos=[clip_video(seed=11)])
    b = ask(eng, VMSGS, videos=[clip_video(seed=12)])
    ip = ImagePrompt(rendered(VMSGS), (), TOWER, model, videos=(clip_video(seed=12),))
    assert a.cached_tokens == 0
    assert b.cached_tokens <= ip.ids.index(VPAD)
    assert b.text == ask(engine(toy), VMSGS, videos=[clip_video(seed=12)]).text
    assert ip.ids == ImagePrompt(rendered(VMSGS), (), TOWER, model,
                                 videos=(clip_video(seed=11),)).ids


def test_the_same_video_resent_reuses_its_kv_and_skips_the_tower(toy):
    _ref, model, _ = toy
    eng = engine(toy)
    hist = list(VMSGS)
    first = ask(eng, hist, videos=[clip_video(seed=21)], n=4)
    hist = hist + [{"role": "assistant", "content": first.text}, user("and the fox")]
    with TowerCalls(model) as calls:
        warm = ask(eng, hist, videos=[clip_video(seed=21)])
    assert calls.n == 0 and warm.cached_tokens > 0
    with TowerCalls(model) as calls:
        cold = ask(engine(toy), hist, videos=[clip_video(seed=21)])
    assert calls.n == 1 and warm.text == cold.text


# ------------------------------------------------------- THE FIT CHECK --

def test_count_tokens_is_what_generate_runs(toy):
    _ref, model, _ = toy
    eng = engine(toy)
    v, img = clip_video(), image(16, 24, 1)
    req = GenerationRequest(MSGS, SampleParams(max_tokens=2), images=(img,), videos=(v,))
    n = ImagePrompt(rendered(MSGS), (img,), TOWER, model, videos=(v,)).n
    assert eng.count_tokens(req) == n == len(rendered(MSGS)) + VIS.expansion((img,), (v,))
    assert complete(eng, req).prompt_tokens == n


def test_an_engine_without_video_refuses_it_by_name(toy):
    import dataclasses

    _ref, model, _ = toy
    eng = engine(toy)
    eng.vision = dataclasses.replace(VIS, video=None, video_reason="PyAV is not installed")
    with pytest.raises(ValueError, match=r"cannot read video on this server \(PyAV"):
        ask(eng, VMSGS, videos=[clip_video()])


def _post(port, body, path="/v1/chat/completions"):
    import http.client

    c = http.client.HTTPConnection("127.0.0.1", port, timeout=60)
    c.request("POST", path, json.dumps(body), {"Content-Type": "application/json"})
    r = c.getresponse()
    data = r.read()
    c.close()
    return r, json.loads(data)


def test_over_http_the_context_check_charges_every_video_token(toy):
    """The chat route end to end on the toy: a video_url data URL decodes,
    /tokenize counts it as generate() runs it, a request that fits answers
    with that prompt_tokens, and with a window too small for the video the
    context check refuses before a forward, naming the real count."""
    from drinkme.serving.http import start_server
    from test_serving_video import clip

    url = "data:video/mp4;base64," + base64.b64encode(clip(20, 10, 24, 16, seed=9)).decode()
    body = {"messages": [{"role": "user", "content": [
        {"type": "text", "text": "what happens in this video"},
        {"type": "video_url", "video_url": {"url": url}}]}], "max_tokens": 3,
        "temperature": 0}
    v = clip_video(seed=9)
    n = len(rendered(VMSGS)) + v.expansion
    srv = start_server(engine(toy), "127.0.0.1", 0)
    try:
        port = srv.server_address[1]
        r, tok = _post(port, {"messages": body["messages"]}, path="/tokenize")
        assert r.status == 200 and tok["count"] == n and len(tok["tokens"]) == n - v.expansion
        r, out = _post(port, body)
        assert r.status == 200, out
        assert out["usage"]["prompt_tokens"] == n
    finally:
        srv.shutdown()
    small = engine(toy, ctx=n - 1)
    srv = start_server(small, "127.0.0.1", 0)
    try:
        with TowerCalls(toy[1]) as calls:
            r, out = _post(srv.server_address[1], body)
        assert r.status == 400 and calls.n == 0
        assert str(n) in out["error"]["message"], out
        assert out["error"]["code"] == "context_length_exceeded"
    finally:
        srv.shutdown()
