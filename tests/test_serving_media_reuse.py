"""A follow-up about the same image or video: the context checkpoint at the
media item's end (serving/ctx_checkpoints.py, MEDIA ENDS) and the vision
tower's output cache (serving/tower_cache.py). CPU, float32, no downloads:
tests/test_serving_video_prompt.py's tiny random-weight
Qwen3_5ForConditionalGeneration and its PyAV-made clips.

The pins:
  * two different questions after the same media: the second reuses the
    slot through the media's end and the tower runs once in total; with
    media checkpoints off it does not, which is the case main measured on
    the 27B (a restore mid-video and a second tower run);
  * the same media in another conversation (another system prompt): no
    prefix to reuse, and the tower is skipped by its cache;
  * EXACTNESS: a warm request (restored at the media end, the tower
    skipped) gives the cold request's prefill logits bit for bit and its
    transcript; a tower-cache hit gives a miss's; caches on, media
    checkpoints and the tower cache off, and checkpoints off altogether
    agree to reduction order (the prefill is split at other places) and
    give one transcript, the reference's;
  * the cache's cap, its LRU order and its 0 = off knob through the engine;
  * the media checkpoints a slot keeps, and the fit charge that counts them.
"""

from __future__ import annotations

import pytest
import torch

from drinkme.serving import ctx_checkpoints as cc
from drinkme.serving import prefill, tower_cache, video
from drinkme.serving.engines import HFEngine
from drinkme.serving.image_prompt import ImagePrompt

from test_serving_mtp import _torch_chunk_on_cpu  # noqa: F401 — fixture by name
from test_serving_prefill_bound import env
from test_serving_video_prompt import (TOWER, TowerCalls, ask, clip_video, engine, image,
                                       ref_greedy, ref_logits, reference_ids, rendered, toy,
                                       user)  # noqa: F401 — toy is a fixture

pytestmark = [pytest.mark.filterwarnings("ignore::DeprecationWarning")]

@pytest.fixture(autouse=True)
def _knobs(monkeypatch):
    """The knobs these tests set per engine start unset (their defaults),
    whatever an earlier test left in the environment."""
    for k in (cc.ENV, cc.MEDIA_ENV, tower_cache.ENV):
        monkeypatch.delenv(k, raising=False)


needs_av = pytest.mark.skipif(not video.available(),
                              reason="PyAV (the drinkme[video] extra) is not installed")
KINDS = [pytest.param("video", marks=needs_av), "image"]

Q1 = "what happens in this video"
Q2 = "the quick brown fox"
SYSTEM = {"role": "system", "content": "hello world the quick brown fox"}


HISTORY = [user("hello world"), {"role": "assistant", "content": "the fox"}]


def turn(kind, text, system=False, history=False):
    """One user turn: the media item, then the question after it; after
    another system prompt, or after an earlier exchange (whose end is the
    last user message's start, where the engine takes a checkpoint below
    the media's)."""
    msgs = [user("V" if kind == "video" else "I", text)]
    return [SYSTEM] * system + HISTORY * history + msgs


def media(kind, seed):
    """(images, videos): a fresh item of `kind` (the engine releases its
    pixels once the tower reads them)."""
    if kind == "video":
        return (), (clip_video(seed=seed),)
    return (image(16, 24, seed),), ()


def run(eng, msgs, kind, seed, n=6):
    imgs, vids = media(kind, seed)
    return ask(eng, msgs, imgs, vids, n=n)


def prompt(model, msgs, kind, seed) -> ImagePrompt:
    imgs, vids = media(kind, seed)
    return ImagePrompt(rendered(msgs), imgs, TOWER, model, videos=vids)


def announced(capsys) -> str:
    """The last request line image_prompt.ImagePrompt.announce printed."""
    lines = [x for x in capsys.readouterr().err.splitlines() if "the tower ran" in x]
    return lines[-1]


class Logits:
    """Every prefill's last-row logits, as prefill.run returned them."""

    def __init__(self, monkeypatch):
        self.rows = []
        real = prefill.run

        def wrapped(*a, **kw):
            out = real(*a, **kw)
            self.rows.append(out.detach().clone())
            return out

        monkeypatch.setattr(prefill, "run", wrapped)


# ------------------------------------------------ THE MEDIA-END CHECKPOINT --

@pytest.mark.parametrize("history", [False, True])
@pytest.mark.parametrize("tower_cache_gib", ["0", None])
@pytest.mark.parametrize("kind", KINDS)
def test_a_second_question_reuses_the_slot_through_the_media_end(toy, kind, tower_cache_gib,
                                                                  history, capsys):
    """Two different questions after the same media, one conversation each:
    the second parts from the slot right after the media, restores the
    checkpoint at its end and prefills only its own question. The tower
    runs once in total, with the tower cache off too: the media lies in the
    reused prefix. A third question does the same: the second request's own
    checkpoints, taken within MIN_STEP of it, do not thin it out, though
    after an earlier exchange a checkpoint of the first request's (at the
    last user message's start) lies below it, the shape of the 27B's
    restore point mid-video."""
    _ref, model, _ = toy
    with env(DRINKME_TOWER_CACHE_GIB=tower_cache_gib):
        eng = engine(toy)
    end, = prompt(model, turn(kind, Q2, history=history), kind, 61).media_ends()
    with TowerCalls(model) as calls:
        first = run(eng, turn(kind, Q1, history=history), kind, 61)
        if history:
            below = [c.n for c in eng._slots[0].ckpts.items if c.n < end and not c.media]
            assert below, "the first request took no checkpoint below the media end"
        second = run(eng, turn(kind, Q2, history=history), kind, 61)
        third = run(eng, turn(kind, "hello world", history=history), kind, 61)
    assert first.cached_tokens == 0
    assert second.cached_tokens >= end and third.cached_tokens >= end
    assert calls.n == 1
    line = announced(capsys)
    assert "the tower ran 0x, 1 inside the reused prefix" in line
    if tower_cache_gib == "0":
        assert "tower cache" not in line
    else:
        assert "tower cache: 0 hits, 0 misses" in line
    # the checkpoint sits past the media's closing marker
    ip = prompt(model, turn(kind, Q2, history=history), kind, 61)
    assert end == ip.runs[-1][1] + 1 and ip.ids[end - 1] == TOWER.vision_end_id


@pytest.mark.parametrize("kind", KINDS)
def test_without_media_checkpoints_the_second_question_restores_short_of_the_media(toy, kind):
    """The case main measured on the 27B: no checkpoint at the media's end,
    so the second question cannot reuse through it, and with the tower cache
    off the tower runs again."""
    _ref, model, _ = toy
    with env(DRINKME_MEDIA_CHECKPOINTS=0, DRINKME_TOWER_CACHE_GIB=0):
        eng = engine(toy)
    end, = prompt(model, turn(kind, Q2), kind, 62).media_ends()
    with TowerCalls(model) as calls:
        run(eng, turn(kind, Q1), kind, 62)
        second = run(eng, turn(kind, Q2), kind, 62)
    assert second.cached_tokens < end and calls.n == 2
    assert not any(c.media for c in eng._slots[0].ckpts.items)


# ------------------------------------------------------- THE TOWER CACHE --

@pytest.mark.parametrize("kind", KINDS)
def test_the_same_media_in_another_conversation_skips_the_tower_by_its_cache(toy, kind,
                                                                             capsys):
    """Another system prompt: the prompt shares no prefix with the slot, so
    nothing is reused, and the tower's output comes from the cache. With
    the cache off the tower runs again."""
    _ref, model, _ = toy
    eng = engine(toy)
    run(eng, turn(kind, Q1), kind, 63)
    assert "tower cache: 0 hits, 1 miss " in announced(capsys)
    with TowerCalls(model) as calls:
        other = run(eng, turn(kind, Q1, system=True), kind, 63)
    assert other.cached_tokens == 0 and calls.n == 0
    assert "the tower ran 0x, 0 inside the reused prefix; tower cache: 1 hit, 0 misses" \
        in announced(capsys)
    with env(DRINKME_TOWER_CACHE_GIB=0):
        off = engine(toy)
    run(off, turn(kind, Q1), kind, 63)
    with TowerCalls(model) as calls:
        run(off, turn(kind, Q1, system=True), kind, 63)
    assert calls.n == 1


@pytest.mark.parametrize("kind", KINDS)
def test_exact_with_the_caches_on_and_off_and_against_the_reference(toy, kind, monkeypatch):
    """Q1, then Q2 after the same media (restored at its end, KV reused),
    then Q1 under another system prompt (the tower skipped by its cache),
    each on one engine per configuration:
      * caches on (the defaults): the tower runs once over the three; each
        warm request's prefill logits equal, bit for bit, those of the same
        request on a fresh engine of the same configuration (the same
        spans, the tower run), and so does its transcript;
      * media checkpoints and the tower cache off, and context checkpoints
        off altogether: the prefill is split at other places, so the logits
        agree to reduction order, the class tests/test_serving_video_prompt
        pins against the reference; the transcripts are one;
      * the reference (Qwen3_5ForConditionalGeneration over the processor's
        prompt): logits to that tolerance, its greedy tokens exactly."""
    ref, model, _ = toy
    seed, n = 64, 6
    reqs = [turn(kind, Q1), turn(kind, Q2), turn(kind, Q1, system=True)]
    configs = {"on": {},
               "off": {"DRINKME_MEDIA_CHECKPOINTS": 0, "DRINKME_TOWER_CACHE_GIB": 0},
               "none": {"DRINKME_CTX_CHECKPOINTS": 0, "DRINKME_TOWER_CACHE_GIB": 0}}
    got = {}
    for name, knobs in configs.items():
        logits = Logits(monkeypatch)
        with env(**knobs):
            eng = engine(toy)
        with TowerCalls(model) as calls:
            outs = [run(eng, msgs, kind, seed, n) for msgs in reqs]
        got[name] = (logits.rows, outs, calls.n)
        monkeypatch.undo()
    rows, outs, calls = got["on"]
    end, = prompt(model, reqs[1], kind, seed).media_ends()
    assert calls == 1
    assert outs[1].cached_tokens >= end and outs[2].cached_tokens == 0
    assert got["off"][2] == 3 and got["none"][2] == 3
    # warm against cold, caches on: the same computation, bit for bit
    for i in (1, 2):
        logits = Logits(monkeypatch)
        cold = run(engine(toy), reqs[i], kind, seed, n)
        assert torch.equal(rows[i], logits.rows[0]), i
        assert outs[i].text == cold.text and cold.cached_tokens == 0
        monkeypatch.undo()
    for name in ("off", "none"):
        for i in range(3):
            torch.testing.assert_close(got[name][0][i], rows[i], atol=2e-5, rtol=1e-5)
            assert got[name][1][i].text == outs[i].text, (name, i)
    for i, msgs in enumerate(reqs):
        imgs, vids = media(kind, seed)
        ids = reference_ids(msgs, vids, imgs)
        assert outs[i].prompt_tokens == len(ids)
        imgs, vids = media(kind, seed)
        torch.testing.assert_close(rows[i], ref_logits(ref, ids, imgs, vids)[-1],
                                   atol=2e-5, rtol=1e-5)
        imgs, vids = media(kind, seed)
        want = ref_greedy(ref, ids, imgs, vids, n)
        assert outs[i].text.split() == model_text(want).split(), i


def model_text(ids):
    from test_serving_video_prompt import TOK

    return TOK.decode(ids)


@needs_av
def test_the_cap_evicts_the_least_recently_used_and_0_turns_it_off(toy):
    """With the prefix cache off every request prefills cold, so only the
    tower cache can skip the tower. A cap that holds one video's output:
    A, B, B, A runs the tower for A, B, then A again (B evicted A, and A
    evicts B); the default cap runs it twice; 0 runs it four times."""
    _ref, model, _ = toy
    one = clip_video(seed=71)
    entry = one.tokens * TOWER_HIDDEN * 4  # float32 rows of the toy's text width
    for cap, want_calls, want_held in ((str(1.5 * entry / 1024 ** 3), 3, 1),
                                       (None, 2, 2), ("0", 4, 0)):
        with env(DRINKME_TOWER_CACHE_GIB=cap):
            eng = engine(toy, slots=0)
        with TowerCalls(model) as calls:
            for seed in (71, 72, 72, 71):
                run(eng, turn("video", Q1), "video", seed, n=2)
        assert calls.n == want_calls, cap
        assert len(eng._tower_cache) == want_held
        if cap is not None and cap != "0":
            assert eng._tower_cache.evictions == 2 and eng._tower_cache.nbytes == entry


TOWER_HIDDEN = 64  # tests/test_serving_video_prompt.config's out_hidden_size


def test_the_engine_announces_the_cache_and_keys_it_on_its_tower(toy, capsys):
    _ref, model, _ = toy
    eng = engine(toy)
    out = capsys.readouterr().out
    assert f"tower cache: up to {tower_cache.human(tower_cache.cap_from_env())} of the " \
           "tower's output in host RAM" in out
    assert eng._tower_cache.identity == ("toy", "test", "qwen3_5", TOWER.path, "torch.float32")
    with env(DRINKME_TOWER_CACHE_GIB=0):
        engine(toy)
    assert "tower cache: off (DRINKME_TOWER_CACHE_GIB=0)" in capsys.readouterr().out
    # the same model served text-only: no cache, no media checkpoints
    from test_serving_video_prompt import TOK

    with env(DRINKME_PREFIX_SLOTS=1, DRINKME_SLOT_DIR="off"):
        plain = HFEngine(model, TOK, model_id="toy", arm="test", meta={}, ctx=256)
    assert plain._tower_cache is None and plain._ckpt_media == 0


# --------------------------------------------- THE MEDIA CHECKPOINTS' BOUND --

@needs_av
@pytest.mark.parametrize("media_max", [1, 2, 3])
def test_a_request_takes_the_last_media_ends_and_a_slot_keeps_at_most_media_max(
        toy, media_max):
    """Two videos and an image in one prompt: three media ends. A request
    takes the last `media_max`, and over follow-ups that part from the slot
    at each of them in turn the slot never holds more than media_max media
    checkpoints, nor more than max_held in all."""
    _ref, model, _ = toy
    with env(DRINKME_MEDIA_CHECKPOINTS=media_max, DRINKME_TOWER_CACHE_GIB=0):
        eng = engine(toy)
    held = cc.max_held(eng._ckpt_max, eng.ctx, media=media_max)
    assert eng._ckpt_media == media_max

    def items(seed_a, seed_b, seed_i, text):
        msgs = [user("V", "hello", "V", "world", "I", text)]
        imgs, vids = (image(16, 24, seed_i),), (clip_video(seed=seed_a), clip_video(seed=seed_b))
        return msgs, imgs, vids

    msgs, imgs, vids = items(81, 82, 83, Q1)
    ends = ImagePrompt(rendered(msgs), imgs, TOWER, model, videos=vids).media_ends()
    assert len(ends) == 3
    ask(eng, msgs, imgs, vids, n=2)
    ck = eng._slots[0].ckpts
    assert [c.n for c in ck.items if c.media] == ends[-media_max:]
    # follow-ups that part after the image, after the second video, after
    # the first, and a new conversation over the same first video
    for seeds, text in (((81, 82, 83), Q2), ((81, 82, 84), Q1), ((81, 85, 84), Q2),
                        ((81, 82, 83), "hello world"), ((81, 85, 86), Q1)):
        msgs, imgs, vids = items(*seeds, text)
        ask(eng, msgs, imgs, vids, n=2)
        ck = eng._slots[0].ckpts
        assert sum(c.media for c in ck.items) <= media_max
        assert len(ck) <= held


def test_the_fit_charge_counts_the_media_checkpoints(toy, monkeypatch, capsys):
    """The residency check is handed the slot, its checkpoints at max_held
    with the media bound in it, and the boot line says so."""
    _ref, model, _ = toy
    seen = []
    monkeypatch.setattr(HFEngine, "_check_slot_residency",
                        lambda self, total, n: seen.append((total, n)))
    eng = engine(toy, slots=2)
    out = capsys.readouterr().out
    one = eng.checkpoint_bytes()
    held = cc.max_held(32, eng.ctx, media=cc.MEDIA_DEFAULT)
    assert held == cc.max_held(32, eng.ctx) + cc.MEDIA_DEFAULT == 7
    assert (f"context checkpoints: up to {held} per slot (--ctx-checkpoints 32, "
            f"{cc.MIN_STEP} apart at ctx {eng.ctx}, {cc.MEDIA_DEFAULT} at media ends "
            f"({cc.MEDIA_ENV})) x {one:,} B") in out
    fixed, ring, per_token = eng._slot_fixed_bytes, eng._slot_ring_bytes, eng._slot_token_bytes
    assert seen[-1] == (2 * (fixed + ring + per_token * eng.ctx + held * one), 2)
    with env(DRINKME_MEDIA_CHECKPOINTS=0):
        engine(toy, slots=2)
    assert seen[-1] == (2 * (fixed + ring + per_token * eng.ctx
                             + cc.max_held(32, eng.ctx) * one), 2)
    assert "at media ends" not in capsys.readouterr().out
