"""Context checkpoints (serving/ctx_checkpoints.py) on toy models — CPU only,
no downloads.

Reusing a slot only when the next prompt extends everything it holds is
not enough: chat templates re-render history differently from what the
model wrote, so a real conversation parts from its slot a few tokens
before or after the previous prompt's end and would re-prefill in full.
Here the three
state kinds a rewind by length cannot reach are checkpointed and restored,
on the three families that have them: gemma-4 (sliding-window rings),
Qwen3.5 (gated-DeltaNet recurrent and conv states beside full attention)
and Muse-Glimmer (sliding-window rings). The toys are float32, so a warm
turn and a cold one agree to reduction order and their greedy transcripts
are equal, the bar the prefix cache's own identity tests set.

Two re-renders are emulated on the word-level toys:
  PAIR: the generation prompt ends in two tokens the history drops (gemma's
    empty thought pair, thinking off): the next prompt parts from the slot
    two tokens before the previous prompt's end, and the checkpoint at
    n_prompt - 4 serves it;
  TRIM: the history drops the last token the model wrote (Qwen's trailing
    newline): the next prompt parts from the slot inside the generation, and
    the checkpoint at n_prompt serves it.
"""

from __future__ import annotations

import copy
import json
import os
import random

import pytest
import torch

from drinkme.serving import ctx_checkpoints as cc
from drinkme.serving import engines, metrics, prefill
from drinkme.serving.engine import GenerationRequest, SampleParams, complete
from drinkme.serving.engines import HFEngine, pick_slot

from test_serving_gemma_spec import gemma_toys  # noqa: F401 — fixture by name
from test_serving_mtp import _torch_chunk_on_cpu, toy as qwen_toy  # noqa: F401
from test_serving_prefill_bound import env

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")

PAIR = ("{% for m in messages %}{{ m['role'] }} : {{ m['content'] }} {% endfor %}"
        "{% if add_generation_prompt %}assistant : alpha beta{% endif %}")
PLAIN = ("{% for m in messages %}{{ m['role'] }} : {{ m['content'] }} {% endfor %}"
         "{% if add_generation_prompt %}assistant :{% endif %}")
LONG = ("the quick brown fox jumps over the lazy dog hello world alpha gamma "
        "delta epsilon zeta the quick brown fox jumps over the lazy dog")


def retemplate(tok, template):
    tok = copy.deepcopy(tok)
    tok.chat_template = template
    return tok


def engine(model, tok, ckpts=None, slots=1, ctx=512, head=None, chunk=None, meta=None,
           **kw):
    with env(DRINKME_CTX_CHECKPOINTS=ckpts, DRINKME_PREFIX_SLOTS=slots,
             DRINKME_PREFILL_CHUNK=chunk):
        return HFEngine(model, tok, model_id="toy", arm="test",
                        meta={} if meta is None else meta, ctx=ctx, mtp_head=head, **kw)


def ask(eng, msgs, spec="off", n=6, images=()):
    with env(DRINKME_SPEC=spec):
        return complete(eng, GenerationRequest(
            msgs, SampleParams(temperature=0.0, max_tokens=n), images=tuple(images)))


def written(eng, res):
    """The generated ids the slot holds after `res` (the last emitted token is
    never forwarded), special tokens and all, as the history must carry them
    to re-render them (tests/test_serving_slotstore.py's agent_turn)."""
    slot = max(eng._slots, key=lambda s: s.stamp)
    return slot.ids[res.prompt_tokens:]


def as_is(tok, gen):
    return tok.decode(gen, skip_special_tokens=False)


def trim(tok, gen):
    return tok.decode(gen[:-1], skip_special_tokens=False)


# ------------------------------------------------------------ the knob --


@pytest.mark.parametrize("raw,want", [(None, 32), ("", 32), ("0", 0), ("4", 4), (" 7 ", 7)])
def test_the_knob(raw, want):
    with env(DRINKME_CTX_CHECKPOINTS=raw):
        assert cc.max_from_env() == want


@pytest.mark.parametrize("raw", ["-1", "four", "1.5"])
def test_a_bad_knob_warns_and_keeps_the_default(raw, capsys):
    with env(DRINKME_CTX_CHECKPOINTS=raw):
        assert cc.max_from_env() == cc.DEFAULT_MAX
    assert cc.ENV in capsys.readouterr().err


def test_the_flag_wins_over_the_env():
    with env(DRINKME_CTX_CHECKPOINTS="5"):
        assert cc.max_from_env(9) == 9 and cc.max_from_env(0) == 0
    assert cc.max_from_env(-3) == cc.DEFAULT_MAX


def test_llama_cpps_numbers():
    """common/common.h and server-context.cpp at 1e6f9e4a5."""
    assert (cc.DEFAULT_MAX, cc.MIN_STEP, cc.TAIL, cc.UBATCH) == (32, 8192, 4, 512)
    assert (cc.SIMILARITY, cc.KEEP) == (0.1, 0.5)


def test_the_cli_flag_reaches_serve_run_and_serve_writes_the_env(monkeypatch):
    """`drinkme serve --ctx-checkpoints N` through argparse
    (test_serving_wiring's --prefix-slots shape), and serve.run applies it by
    writing DRINKME_CTX_CHECKPOINTS, which the engine reads at construction."""
    from drinkme import cli, serve
    from drinkme.bootstrap import BootstrapOutcome

    seen = {}
    monkeypatch.setattr(serve, "run", lambda *a, **kw: seen.update(kw) or 0)
    monkeypatch.setattr("drinkme.bootstrap.ensure_accelerator", lambda: BootstrapOutcome("ok"))
    assert cli.main(["serve", "--model", "Qwen3-8B", "--ctx-checkpoints", "3"]) == 0
    assert seen["ctx_checkpoints"] == 3
    seen.clear()
    cli.main(["serve", "--model", "Qwen3-8B"])
    assert seen["ctx_checkpoints"] is None


def test_serve_run_writes_the_env_before_the_engine_is_built(monkeypatch):
    from drinkme import serve

    # setenv, so teardown puts back what was there before serve.run wrote
    # it (a delenv after the write would restore the "5")
    monkeypatch.setenv(cc.ENV, "")
    monkeypatch.setenv("DRINKME_FAKE_ENGINE", "1")
    monkeypatch.setattr(serve, "_serve", lambda *a, **kw: 0)
    serve.run("org/toy", None, ctx_checkpoints=5)
    assert os.environ[cc.ENV] == "5"


# ------------------------------------------------------------ positions --


def test_positions_are_llama_cpps_two_near_the_end_and_the_end():
    assert cc.positions(1000, 0) == [1000 - 4 - 512, 996, 1000]
    assert cc.positions(100, 0) == [96, 100]           # 100 - 516 < 1
    assert cc.positions(100, 96) == [100]               # the state at start is held
    assert cc.positions(100, 97) == [100]
    assert cc.positions(3, 0) == [3]


def test_and_the_last_user_messages_start():
    assert cc.positions(1000, 0, user=300) == [300, 484, 996, 1000]
    assert cc.positions(1000, 300, user=300) == [484, 996, 1000]
    assert cc.positions(100, 0, user=96) == [96, 100]


def test_a_position_inside_a_whole_run_moves_to_its_start():
    # a run [80, 98) that a span may not cut: n_prompt - 4 = 96 is inside it
    assert cc.positions(100, 0, whole=[(80, 98)]) == [80, 100]
    # and one that would move to or before start is dropped
    assert cc.positions(100, 85, whole=[(80, 98)]) == [100]
    # the run's end is not inside it
    assert cc.positions(100, 0, whole=[(80, 96)]) == [96, 100]


def test_spans_end_at_the_cuts():
    assert prefill.spans(0, 20, 0, cuts=[16, 20]) == [(0, 16), (16, 20)]
    assert prefill.spans(0, 20, 8, cuts=[10, 20]) == [(0, 8), (8, 10), (10, 16), (16, 20)]
    assert prefill.spans(4, 20, 8, cuts=[4, 8]) == [(4, 8), (8, 12), (12, 20)]
    # a cut never lands inside a run a span may not cut (positions() moved it)
    assert prefill.spans(0, 30, 8, whole=[(6, 12)], cuts=[6]) == [(0, 6), (6, 14),
                                                                   (14, 22), (22, 30)]


# ------------------------------------------------------- the list rules --


def ck(n, task=0):
    return cc.Checkpoint(n, task, {}, 0)


def listing(n_max, items, min_step=cc.MIN_STEP):
    c = cc.Checkpoints(n_max, min_step)
    c.items = [ck(n, t) for n, t in items]
    return c


def test_add_erases_other_requests_checkpoints_within_min_step():
    """llama.cpp #25472: a checkpoint of another request within min_step of the
    previous kept one goes; the current request's own are kept."""
    c = listing(32, [(10, 1), (50, 1), (9000, 1), (9030, 2)], min_step=100)
    gone = c.add(ck(9050, 2))
    assert [g.n for g in gone] == [50]
    assert c.positions() == [10, 9000, 9030, 9050]


def test_add_erases_the_oldest_when_full():
    c = listing(3, [(10, 1), (20, 2), (30, 3)], min_step=0)
    assert [g.n for g in c.add(ck(40, 4))] == [10]
    assert c.positions() == [20, 30, 40]
    one = listing(1, [(10, 1)], min_step=0)
    one.add(ck(20, 2))
    assert one.positions() == [20]


def test_reach_extends_rewinds_or_restores():
    c = listing(32, [(10, 1), (20, 1)])
    # the prompt extends the slot's 30 ids: all of them, checkpoints or not
    assert c.reach(30, 30, 40) == 30
    # parts from it at 25: the newest checkpoint at or below
    assert c.reach(25, 30, 40) == 20
    assert c.reach(19, 30, 40) == 10
    assert c.reach(9, 30, 40) == 0
    # the prompt is a prefix of the slot: one token is left to compute
    assert c.reach(20, 30, 20) == 10
    # a cache that rewinds freely reaches the prefix itself
    c.free = True
    assert c.reach(25, 30, 40) == 25 and c.reach(20, 30, 20) == 19
    assert c.reach(25, 30, 40, fit=lambda n: 22 if 22 < n < 27 else n) == 22
    # a checkpoint `fit` would move is not usable
    c.free = False
    assert c.reach(25, 30, 40, fit=lambda n: 15 if 15 < n < 21 else n) == 10
    # off: the extends-only rule
    off = listing(0, [(10, 1), (20, 1)])
    assert off.reach(25, 30, 40) == 0 and off.reach(30, 30, 40) == 30


def test_drop_beyond():
    c = listing(32, [(10, 1), (20, 1), (30, 2)])
    c.drop_beyond(20)
    assert c.positions() == [10, 20]


@pytest.mark.parametrize("media", [0, cc.MEDIA_DEFAULT, 4])
@pytest.mark.parametrize("n_max,ctx,min_step", [(32, 8192, cc.MIN_STEP), (32, 32768, cc.MIN_STEP),
                                                (32, 262144, cc.MIN_STEP), (5, 4000, 100),
                                                (2, 4000, 100), (1, 500, 10)])
def test_no_request_sequence_holds_more_than_max_held(n_max, ctx, min_step, media):
    """THE BOUND the fit charge rests on, against random conversations: every
    request restores somewhere at or below what the slot holds, drops what
    lies past it, and takes the checkpoints positions() names, and with
    `media` the media-end ones media_positions() names among up to six
    images and videos (MEDIA ENDS)."""
    rng = random.Random(n_max * 1000 + ctx + media)
    bound = cc.max_held(n_max, ctx, min_step, media=media)
    worst = 0
    for trial in range(40):
        c = cc.Checkpoints(n_max, min_step, media_max=media)
        held = 0
        for task in range(1, 60):
            n_prompt = rng.randint(held + 1, ctx - 1) if held < ctx - 2 and rng.random() < 0.8 \
                else rng.randint(2, ctx - 1)
            start = min(rng.randint(0, held), n_prompt - 1)
            reach = c.reach(start, held, n_prompt)
            c.drop_beyond(reach)
            user = rng.randint(1, n_prompt) if rng.random() < 0.7 else None
            ends = rng.sample(range(1, n_prompt), min(rng.randint(0, 6), n_prompt - 1))
            at_media = set(cc.media_positions(ends, reach, n_prompt, media))
            for p in sorted(set(cc.positions(n_prompt, reach, user=user)) | at_media):
                c.add(cc.Checkpoint(p, task, {}, 0, media=p in at_media))
                worst = max(worst, len(c))
                assert len(c) <= bound
                assert sum(x.media for x in c.items) <= media
            held = min(ctx - 1, n_prompt + rng.randint(0, 50))
    assert worst <= bound


def test_max_held_at_the_defaults():
    assert cc.max_held(32, 8192) == 5
    assert cc.max_held(32, 32768) == 8
    assert cc.max_held(32, 262144) == 32
    assert cc.max_held(0, 8192) == 0 and cc.max_held(2, 8192) == 2
    # a model that reads images: the media ends on top, under N
    assert cc.max_held(32, 8192, media=cc.MEDIA_DEFAULT) == 7
    assert cc.max_held(32, 262144, media=cc.MEDIA_DEFAULT) == 32
    assert cc.max_held(3, 8192, media=2) == 3 and cc.max_held(0, 8192, media=2) == 0


# ---------------------------------------------------------- media ends --


def mck(n, task=0):
    return cc.Checkpoint(n, task, {}, 0, media=True)


def test_media_positions_are_the_last_ends_past_start_and_before_the_end():
    assert cc.media_positions([10, 30, 50], 0, 100, 2) == [30, 50]
    assert cc.media_positions([50, 10, 30], 30, 100, 2) == [50]
    assert cc.media_positions([10, 30, 100], 0, 100, 5) == [10, 30]  # n_prompt has its own
    assert cc.media_positions([30, 30], 0, 100, 2) == [30]
    assert cc.media_positions([10, 30, 50], 0, 100, 0) == []


def test_media_checkpoints_are_kept_apart_from_the_thinning():
    """llama.cpp #25472's rule would erase an earlier request's checkpoint at
    a media end lying within min_step of a kept one below it (on the 27B, a
    video's end at 2,750 above a checkpoint at 2,062). It is skipped, and
    it does not count as the previous kept one, so the rest are thinned
    exactly as without it."""
    c = listing(32, [(10, 1), (50, 1), (9000, 1)], min_step=100)
    c.items.insert(1, mck(40, 1))
    gone = c.add(ck(9050, 2))
    assert [g.n for g in gone] == [50]  # what the list without 40 loses
    assert c.positions() == [10, 40, 9000, 9050]
    assert [x.n for x in c.items if x.media] == [40]


def test_a_media_checkpoint_past_media_max_erases_the_lowest_media_one():
    c = cc.Checkpoints(32, 0, media_max=2)
    c.items = [ck(10, 1), mck(20, 1), mck(30, 1), ck(40, 1)]
    gone = c.add(mck(50, 2))
    assert [g.n for g in gone] == [20]
    assert c.positions() == [10, 30, 40, 50]
    assert [x.n for x in c.items if x.media] == [30, 50]
    # a checkpoint that is not at a media end erases none by that rule
    c.add(ck(60, 2))
    assert [x.n for x in c.items if x.media] == [30, 50]


def test_n_counts_media_checkpoints_too():
    c = cc.Checkpoints(3, 0, media_max=2)
    c.items = [mck(10, 1), ck(20, 1), mck(30, 1)]
    assert [g.n for g in c.add(ck(40, 2))] == [10]  # the oldest goes, media or not
    assert c.positions() == [20, 30, 40]


@pytest.mark.parametrize("raw,want", [("", cc.MEDIA_DEFAULT), ("0", 0), ("3", 3), (" 1 ", 1)])
def test_the_media_knob(raw, want, monkeypatch):
    monkeypatch.setenv(cc.MEDIA_ENV, raw)
    assert cc.media_from_env() == want


@pytest.mark.parametrize("raw", ["x", "-1"])
def test_a_bad_media_knob_warns_unless_told_not_to(raw, monkeypatch, capsys):
    monkeypatch.setenv(cc.MEDIA_ENV, raw)
    assert cc.media_from_env(warn=False) == cc.MEDIA_DEFAULT
    assert capsys.readouterr().err == ""
    assert cc.media_from_env() == cc.MEDIA_DEFAULT
    assert cc.MEDIA_ENV in capsys.readouterr().err


def test_the_cold_tiers_form_keeps_the_media_flag():
    k, v = torch.randn(1, 1, 3, 4), torch.randn(1, 1, 3, 4)
    c = torch.tensor(3)
    for media in (False, True):
        tensors, meta = cc.flatten(cc.Checkpoint(3, 7, {0: (k, v, c, 3)}, 0, media=media), "ck0.")
        back = cc.unflatten(tensors, json.loads(json.dumps(meta)), "cpu")
        assert back.media is media and back.n == 3 and torch.equal(back.layers[0][0], k)
    # a file written before media checkpoints has no flag: not one
    meta.pop("media")
    assert cc.unflatten(tensors, meta, "cpu").media is False


# ----------------------------------------------------------- pick_slot --


def slot(n, ids, stamp=0, reach=None, free=False, n_max=32):
    s = engines._Slot(n, n_max)
    s.ids, s.alloc, s.stamp = list(ids), 4096, stamp
    s.cache = "a cache"
    s.ckpts.items = [ck(p) for p in (reach or [])]
    s.ckpts.free = free
    return s


def test_a_slot_the_prompt_parts_from_serves_what_its_checkpoint_reaches():
    a = slot(0, range(100), stamp=1, reach=[60, 90])
    prompt = list(range(95)) + [-1] * 10
    assert pick_slot([a], prompt, len(prompt), 200) == (a, 90)


def test_under_a_tenth_of_the_prompt_is_not_a_match():
    """llama.cpp's --slot-prompt-similarity 0.1."""
    a = slot(0, range(100), stamp=1, free=True)
    prompt = list(range(10)) + [-1] * 91
    got, n = pick_slot([a], prompt, len(prompt), 200)
    assert n == 0
    prompt = list(range(11)) + [-1] * 90
    assert pick_slot([a], prompt, len(prompt), 200) == (a, 11)


def test_another_conversation_waits_for_an_untouched_slot():
    """A reuse that keeps less than half of a live slot does not take it
    while a slot is free; with none free it does."""
    a = slot(0, range(100), stamp=1, free=True)
    empty = engines._Slot(1, 32)
    prompt = list(range(40)) + [-1] * 20
    assert pick_slot([a, empty], prompt, len(prompt), 200) == (empty, 0)
    assert pick_slot([a], prompt, len(prompt), 200) == (a, 40)
    # keeping half or more is the same conversation: taken even with a free slot
    prompt = list(range(60)) + [-1] * 20
    assert pick_slot([a, empty], prompt, len(prompt), 200) == (a, 60)


def test_ties_go_to_an_extension_then_to_the_least_recently_used():
    ext = slot(0, range(50), stamp=5)
    part = slot(1, list(range(50)) + [7, 7, 7], stamp=1, free=True)
    prompt = list(range(50)) + [1, 2, 3]
    assert pick_slot([part, ext], prompt, len(prompt), 200) == (ext, 50)
    old = slot(0, list(range(50)) + [9], stamp=1, free=True)
    new = slot(1, list(range(50)) + [8], stamp=5, free=True)
    assert pick_slot([new, old], prompt, len(prompt), 200) == (old, 50)


def test_off_is_the_extends_only_picker():
    a = slot(0, range(100), stamp=1, reach=[60, 90], n_max=0)
    prompt = list(range(95)) + [-1] * 10
    assert pick_slot([a], prompt, len(prompt), 200)[1] == 0


# ----------------------------------------------------- snapshot / restore --


def _cache_after(model, eng, ids, alloc=256):
    cache = eng._cache(alloc)
    with torch.inference_mode():
        prefill.run(model, ids, 0, cache, "cpu", 0)
    return cache


def _same_state(a, b):
    """Every non-rewindable surface equal, and the full layers the same length."""
    for i, (la, lb) in enumerate(zip(a.layers, b.layers)):
        kind = cc.layer_kind(la)
        if kind == cc.FULL:
            assert la.live == lb.live, i
            assert torch.equal(la.keys[:, :, :la.live], lb.keys[:, :, :lb.live]), i
        elif kind == cc.SLIDING:
            assert la.cumulative_length_int == lb.cumulative_length_int, i
            assert torch.equal(la.cumulative_length, lb.cumulative_length), i
            rows = min(la.cumulative_length_int, la.max_cache_len)
            assert torch.equal(la.keys[:, :, :rows], lb.keys[:, :, :rows]), i
            assert torch.equal(la.values[:, :, :rows], lb.values[:, :, :rows]), i
        elif kind == cc.LINEAR:
            for s in la.conv_states:
                assert torch.equal(la.conv_states[s], lb.conv_states[s]), i
                assert torch.equal(la.recurrent_states[s], lb.recurrent_states[s]), i


@pytest.mark.parametrize("window", [16, 64, 512])
def test_a_restored_gemma_cache_is_the_cache_that_stopped_there(gemma_toys, window):
    """The snapshot at n, restored after the cache went on past it, IS the
    state at n: byte for byte against the same prefill that stopped at n.
    Below the window, across it, and far past it (the ring shifted)."""
    model, tok, _ = gemma_toys[window]
    eng = engine(model, tok, ctx=1024)
    ids = [(3 + 7 * i) % 96 for i in range(120)]
    for n in (10, 70, 100):
        cache = _cache_after(model, eng, ids[:n])
        with torch.inference_mode():  # as generate() runs it
            snap = cc.snapshot(cache, task=1)
            prefill.run(model, ids, n, cache, "cpu", 0)
            cc.restore(cache, snap)
        _same_state(cache, _cache_after(model, eng, ids[:n]))


def test_a_restored_deltanet_cache_is_the_cache_that_stopped_there(qwen_toy):
    model, tok, _head, _cfg = qwen_toy
    eng = engine(model, tok)
    ids = [(3 + 7 * i) % 96 for i in range(80)]
    cache = _cache_after(model, eng, ids[:50])
    with torch.inference_mode():
        snap = cc.snapshot(cache, task=1)
        prefill.run(model, ids, 50, cache, "cpu", 0)
        cc.restore(cache, snap)
    _same_state(cache, _cache_after(model, eng, ids[:50]))


def test_the_kinds_the_families_carry(gemma_toys, qwen_toy):
    model, tok, _ = gemma_toys[16]
    kinds = set(cc.kinds(engine(model, tok, ctx=1024)._cache(64)))
    assert kinds == {cc.FULL, cc.SLIDING}
    model, tok, _h, _c = qwen_toy
    assert set(cc.kinds(engine(model, tok)._cache(64))) == {cc.FULL, cc.LINEAR}


def test_a_ring_that_has_not_wrapped_rewinds_by_length(gemma_toys):
    """A sliding cache whose rings never shifted a row out reaches any
    earlier position by its counters alone: the rewound cache holds the
    same counts as one that stopped there, and continuing from it gives the
    logits a cold prefill gives (to reduction order: the rows below 70 came
    from a 120-token forward). Once a ring wraps it cannot."""
    model, tok, _ = gemma_toys[512]
    eng = engine(model, tok, ctx=1024)
    ids = [(3 + 7 * i) % 96 for i in range(120)]
    alt = ids[:70] + [(5 + 3 * i) % 96 for i in range(20)]
    cache = _cache_after(model, eng, ids)
    assert cc.rewinds_freely(cache)
    stopped = _cache_after(model, eng, ids[:70])
    with torch.inference_mode():
        cc.rewind(cache, 70)
        for la, lb in zip(cache.layers, stopped.layers):
            assert int(la.get_seq_length()) == int(lb.get_seq_length()) == 70
            if cc.layer_kind(la) == cc.SLIDING:
                assert torch.equal(la.cumulative_length, lb.cumulative_length)
        warm = prefill.run(model, alt, 70, cache, "cpu", 0)
        cold = prefill.run(model, alt, 0, eng._cache(256), "cpu", 0)
    torch.testing.assert_close(warm, cold, atol=2e-5, rtol=1e-5)
    model, tok, _ = gemma_toys[16]
    assert not cc.rewinds_freely(_cache_after(model, engine(model, tok, ctx=1024), ids))


# ------------------------------------------------ warm == cold, by family --

import test_serving_glimmer_vision as gv  # noqa: E402
from test_serving_prefill_bound import Rows, dense  # noqa: E402,F401


@pytest.fixture(scope="module")
def glimmer_toys():
    """{window: (model, tok, tower)}: test_serving_glimmer_vision's Muse-Glimmer
    toy (three sliding layers per full one) at two windows."""
    from transformers.models.muse_glimmer import MuseGlimmerForConditionalGeneration

    out = {}
    for w in (16, 512):
        cfg = gv.config(w)
        torch.manual_seed(0)
        model = MuseGlimmerForConditionalGeneration(cfg).eval().float()
        for p in model.parameters():
            p.requires_grad_(False)
        out[w] = (model, gv._tokenizer(), gv._tower(cfg))
    return out


def family(request, name):
    """(model, tokenizer, MTP head or None, ctx) for one family's toy."""
    if name.startswith("gemma"):
        model, tok, _ = request.getfixturevalue("gemma_toys")[int(name[5:])]
        return model, tok, None, 1024
    if name == "qwen":
        model, tok, head, _ = request.getfixturevalue("qwen_toy")
        return model, tok, head, 512
    model, tok, _tower = request.getfixturevalue("glimmer_toys")[int(name[7:])]
    return model, tok, None, 320


FAMILIES = ["gemma16", "gemma64", "gemma512", "qwen", "glimmer16", "glimmer512"]
FIRST = " ".join([LONG] * 3)


def common(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def conversation(eng, reply, spec="off", turns=2, first=FIRST, n=6):
    """`turns` agent-shaped turns, each re-sending the history with the last
    reply re-rendered by `reply`. Yields (result, messages, what the slot
    held before the turn, its checkpoint positions, whether it rewound
    freely) per turn."""
    hist = [{"role": "user", "content": first}]
    for t in range(turns):
        slot = eng._slots[0]
        before = (list(slot.ids), slot.ckpts.positions(), slot.ckpts.free)
        res = ask(eng, hist, spec, n)
        yield (res, list(hist), *before)
        hist = hist + [{"role": "assistant", "content": reply(eng.tok, written(eng, res))},
                       {"role": "user", "content": f"and the fox {t}"}]


def expect(eng, msgs, held, ckpts, free):
    """What a turn should reuse, from the rules (ctx_checkpoints.Checkpoints.
    reach), given what the slot held."""
    ids = eng.tokenize(messages=msgs)
    lcp = common(ids, held)
    if 0 < lcp == len(held) and lcp < len(ids):
        return lcp
    target = min(lcp, len(ids) - 1)
    if free:
        n = target
    else:
        n = max([c for c in ckpts if c <= target], default=0)
    return n if n > cc.SIMILARITY * len(ids) else 0


@pytest.mark.parametrize("name", FAMILIES)
@pytest.mark.parametrize("template,reply", [(PAIR, as_is), (PLAIN, trim)], ids=["pair", "trim"])
def test_a_re_rendered_turn_reuses_the_slot_and_answers_the_cold_answer(request, name,
                                                                        template, reply):
    """THE claim, per family: turn 2 re-renders turn 1 the way a template does
    (PAIR drops the generation prompt's last two tokens, TRIM the reply's last
    token), is served from turn 1's slot up to what a checkpoint (or a rewind
    by length) reaches, and answers exactly what a cold engine answers."""
    model, tok, _head, ctx = family(request, name)
    tok = retemplate(tok, template)
    eng = engine(model, tok, ctx=ctx)
    turns = list(conversation(eng, reply, turns=3))
    for i, (res, msgs, held, ckpts, free) in enumerate(turns):
        if i == 0:
            assert res.cached_tokens == 0
            continue
        want = expect(eng, msgs, held, ckpts, free)
        assert res.cached_tokens == want and want > 0, (i, ckpts, free)
        # what it would have reused before: nothing, the prompt parts from the slot
        assert want < len(held)
        cold = ask(engine(model, tok, ctx=ctx), msgs)
        assert res.text == cold.text, i
    # the ring wraps on the 16-window toys and the DeltaNet state never rewinds:
    # there it was a checkpoint, not a rewind, that served the turn
    if name in ("gemma16", "qwen", "glimmer16"):
        assert not turns[1][4]


@pytest.mark.parametrize("name,spec", [("gemma16", "ngram"), ("gemma512", "ngram"),
                                       ("qwen", "ngram"), ("qwen", "auto"),
                                       ("glimmer16", "ngram")])
def test_speculation_after_a_restore_answers_the_cold_answer(request, name, spec):
    """A restored slot under the n-gram proposer and under MTP (Qwen's toy
    head): the proposer's table and the head's KV are rebuilt from the
    prompt on every request (mtp.Speculator.open), and the verify's rewind
    works on restored rings and states as on prefilled ones."""
    model, tok, head, ctx = family(request, name)
    tok = retemplate(tok, PAIR)
    h = head if spec == "auto" else None
    eng = engine(model, tok, ctx=ctx, head=h)
    turns = list(conversation(eng, as_is, spec=spec, turns=3, n=12))
    for res, msgs, held, ckpts, free in turns[1:]:
        assert res.cached_tokens > 0
        cold = ask(engine(model, tok, ctx=ctx, head=h), msgs, spec, n=12)
        assert res.text == cold.text


def test_the_checkpoints_a_turn_takes(gemma_toys):
    """n_prompt - 4 and n_prompt on a short prompt (n_prompt - 516 < 1), each
    a device snapshot of the ten sliding rings; the prefill split there."""
    model, tok, _ = gemma_toys[16]
    eng = engine(model, retemplate(tok, PAIR), ctx=1024)
    with Rows(model) as rows:
        res = ask(eng, [{"role": "user", "content": FIRST}])
    n = res.prompt_tokens
    slot = eng._slots[0]
    assert slot.ckpts.positions() == [n - 4, n]
    assert rows.trunk[:2] == [n - 4, 4]  # the prefill, then the decode steps
    assert all(len(c.layers) == 10 for c in slot.ckpts.items)
    assert slot.ckpts.items[0].nbytes == cc.config_bytes(model.config.to_dict(), 4, 1024)


def test_the_second_near_end_checkpoint_serves_an_edit_further_back(gemma_toys, monkeypatch):
    """llama.cpp's n_prompt - 4 - n_ubatch: a client that edits the last few
    words of its message parts from the slot more than four tokens before
    the prompt's end. n_ubatch shrunk to 8 to fit the toy."""
    monkeypatch.setattr(cc, "UBATCH", 8)
    model, tok, _ = gemma_toys[16]
    tok = retemplate(tok, PLAIN)
    eng = engine(model, tok, ctx=1024)
    words = FIRST.split()
    first = ask(eng, [{"role": "user", "content": FIRST}])
    n = first.prompt_tokens
    assert eng._slots[0].ckpts.positions() == [n - 12, n - 4, n]
    edited = [{"role": "user", "content": " ".join(words[:-6] + ["hello"] * 6)}]
    res = ask(eng, edited)
    assert res.cached_tokens == n - 12
    assert res.text == ask(engine(model, tok, ctx=1024), edited).text


# ---------------------------------- inside the window: copied from the rings --


def _fake(kind, window=None):
    """A cache layer that ctx_checkpoints.layer_kind reads as `kind`."""
    from types import SimpleNamespace

    if kind == cc.SLIDING:
        return SimpleNamespace(cumulative_length_int=0, cumulative_length=0,
                               max_cache_len=window)
    return SimpleNamespace(conv_states={}, recurrent_states={})


def test_the_rings_hold_a_prompt_inside_every_window_and_no_recurrent_state(
        gemma_toys, qwen_toy):
    """ring_holds, the rule that decides whether a prefill is split: a
    sliding model inside its window is not, outside it is; a recurrent
    model always is; so is a mixed one (sliding and linear layers), and the
    smallest window decides."""
    from types import SimpleNamespace

    model, tok, _ = gemma_toys[64]
    cache = engine(model, tok, ctx=1024)._cache(256)
    assert cc.ring_holds(cache, 1) and cc.ring_holds(cache, 64)
    assert not cc.ring_holds(cache, 65)
    model, tok, _h, _c = qwen_toy
    qwen = engine(model, tok)._cache(64)
    assert not any(cc.ring_holds(qwen, n) for n in (1, 5, 64))
    full = next(layer for layer in qwen.layers if cc.layer_kind(layer) == cc.FULL)
    mixed = SimpleNamespace(layers=[_fake(cc.SLIDING, 1024), full, _fake(cc.LINEAR)])
    assert cc.kinds(mixed) == [cc.SLIDING, cc.FULL, cc.LINEAR]
    assert not cc.ring_holds(mixed, 5)
    two = SimpleNamespace(layers=[_fake(cc.SLIDING, 1024), _fake(cc.SLIDING, 128), full])
    assert cc.ring_holds(two, 128) and not cc.ring_holds(two, 129)
    assert cc.ring_holds(SimpleNamespace(layers=[full]), 10 ** 6)


def test_a_checkpoint_copied_from_the_rings_is_the_cache_rewound_there(gemma_toys):
    """snapshot(at=p) on a cache prefilled past p: byte for byte what the
    same cache rewound by length to p holds, and restoring it after the
    ring wrapped gives that state back. Refused once the ring has wrapped."""
    model, tok, _ = gemma_toys[64]
    eng = engine(model, tok, ctx=1024)
    ids = [(3 + 7 * i) % 96 for i in range(120)]
    cache = _cache_after(model, eng, ids[:60])
    with torch.inference_mode():
        lazy = cc.snapshot(cache, task=1, at=40)
        assert lazy.n == 40 and len(lazy.layers) == 10
        prefill.run(model, ids, 60, cache, "cpu", 0)  # the ring wraps at 64
        assert not cc.rewinds_freely(cache)
        with pytest.raises(ValueError):
            cc.snapshot(cache, task=1, at=40)
        cc.restore(cache, lazy)
    rewound = _cache_after(model, eng, ids[:60])
    cc.rewind(rewound, 40)
    _same_state(cache, rewound)
    with torch.inference_mode():
        eager = cc.snapshot(rewound, task=1)
    for i, (k, v, c, n_int) in eager.layers.items():
        lk, lv, lc, ln = lazy.layers[i]
        assert torch.equal(k, lk) and torch.equal(v, lv) and torch.equal(c, lc)
        assert c.dtype == lc.dtype and n_int == ln == 40
    assert eager.nbytes == lazy.nbytes


def _split_always(monkeypatch):
    """The prefill as it was before ring_holds: split at every checkpoint."""
    monkeypatch.setattr(cc, "ring_holds", lambda cache, n: False)


def test_inside_the_window_the_prefill_is_the_unsplit_one(gemma_toys, monkeypatch):
    """A prompt inside every window: one prefill forward, exactly
    --ctx-checkpoints 0's, and the same checkpoints the split prefill took
    (positions, layers, bytes), copied out of the rings after it."""
    model, tok, _ = gemma_toys[64]
    tok = retemplate(tok, PAIR)
    msgs = [{"role": "user", "content": LONG}]
    with Rows(model) as rows:
        res = ask(engine(model, tok, ctx=1024), msgs)
    n = res.prompt_tokens
    assert n <= 64
    eng = engine(model, tok, ctx=1024)
    with Rows(model) as rows:
        res = ask(eng, msgs)
    assert rows.trunk == [n] + [1] * 5
    with Rows(model) as off_rows:
        off = ask(engine(model, tok, ctx=1024, ckpts=0), msgs)
    assert off_rows.trunk == rows.trunk and off.text == res.text
    lazy = eng._slots[0].ckpts
    _split_always(monkeypatch)
    split = engine(model, tok, ctx=1024)
    with Rows(model) as split_rows:
        ask(split, msgs)
    assert split_rows.trunk[:2] == [n - 4, 4]
    eager = split._slots[0].ckpts
    assert lazy.positions() == eager.positions() == [n - 4, n]
    assert [c.nbytes for c in lazy.items] == [c.nbytes for c in eager.items]
    assert [sorted(c.layers) for c in lazy.items] == [sorted(c.layers) for c in eager.items]


def test_outside_the_window_the_prefill_splits_as_before(gemma_toys):
    """A prompt past the window shifts the rings during its own prefill:
    split at each checkpoint, as before ring_holds."""
    model, tok, _ = gemma_toys[64]
    eng = engine(model, retemplate(tok, PAIR), ctx=1024)
    with Rows(model) as rows:
        res = ask(eng, [{"role": "user", "content": FIRST}])
    n = res.prompt_tokens
    assert n > 64
    assert rows.trunk[:2] == [n - 4, 4]
    assert eng._slots[0].ckpts.positions() == [n - 4, n]


def test_a_recurrent_model_splits_as_before(qwen_toy):
    """Qwen3.5's DeltaNet state is overwritten by every token, so its
    checkpoints split the prefill at any length."""
    model, tok, _h, _c = qwen_toy
    eng = engine(model, retemplate(tok, PAIR))
    with Rows(model) as rows:
        res = ask(eng, [{"role": "user", "content": LONG}])
    n = res.prompt_tokens
    assert rows.trunk[:2] == [n - 4, 4]
    assert eng._slots[0].ckpts.positions() == [n - 4, n]


def test_a_ring_that_wraps_in_the_reply_still_has_its_checkpoints(gemma_toys, monkeypatch):
    """The prompt fits the window, the reply takes the slot past it: the
    ring can no longer rewind, and the re-rendered next turn is served by
    the checkpoint copied out of the rings after turn 1's prefill. It
    reuses what the split prefill's checkpoint reuses, and answers the cold
    answer."""
    model, tok, _ = gemma_toys[64]
    tok = retemplate(tok, PAIR)

    def run(eng):
        return list(conversation(eng, as_is, turns=2, first=LONG, n=40))

    turns = run(engine(model, tok, ctx=1024))
    (r1, *_), (r2, msgs, held, ckpts, free) = turns
    assert r1.prompt_tokens <= 64 < len(held) and not free
    assert ckpts == [r1.prompt_tokens - 4, r1.prompt_tokens]
    assert r2.cached_tokens == r1.prompt_tokens - 4
    assert r2.text == ask(engine(model, tok, ctx=1024), msgs, n=40).text
    _split_always(monkeypatch)
    split = run(engine(model, tok, ctx=1024))
    assert [t[0].cached_tokens for t in split] == [t[0].cached_tokens for t in turns]
    assert [t[3] for t in split] == [t[3] for t in turns]


def test_off_is_the_extends_only_engine_byte_for_byte(gemma_toys):
    """--ctx-checkpoints 0: no checkpoint taken, the prefill in the spans it
    always had, and a re-rendered turn prefilled cold."""
    model, tok, _ = gemma_toys[16]
    tok = retemplate(tok, PAIR)
    eng = engine(model, tok, ctx=1024, ckpts=0)
    with Rows(model) as rows:
        turns = list(conversation(eng, as_is, turns=2))
    assert turns[1][0].cached_tokens == 0
    assert eng._slots[0].ckpts.positions() == []
    n1, n2 = turns[0][0].prompt_tokens, turns[1][0].prompt_tokens
    assert rows.trunk == [n1] + [1] * 5 + [n2] + [1] * 5


def test_full_attention_takes_no_checkpoints_and_rewinds(dense):
    """A cache of full-attention layers only rewinds by length to any shared
    prefix: no snapshot, no split prefill, and the turn is served up to the
    common prefix itself."""
    model, tok = dense
    tok = retemplate(tok, PAIR)
    eng = engine(model, tok)
    with Rows(model) as rows:
        turns = list(conversation(eng, as_is, turns=2))
    (r1, *_), (r2, msgs, held, ckpts, free) = turns
    assert ckpts == [] and free
    assert r2.cached_tokens == r1.prompt_tokens - 2
    assert rows.trunk[0] == r1.prompt_tokens  # one forward: nothing to snapshot
    assert r2.text == ask(engine(model, tok), msgs).text


def test_a_restore_is_its_own_slot_event(gemma_toys):
    metrics.reset()
    model, tok, _ = gemma_toys[16]
    eng = engine(model, retemplate(tok, PAIR), ctx=1024)
    list(conversation(eng, as_is, turns=3))
    events = metrics.SLOT_EVENTS.snapshot()
    counts = {ev: events.get((("event", ev),), 0) for ev in ("hit", "restore", "miss", "evict")}
    assert counts == {"hit": 0, "restore": 2, "miss": 1, "evict": 0}


def test_a_failed_turn_leaves_neither_ids_nor_checkpoints(gemma_toys, monkeypatch):
    model, tok, _ = gemma_toys[16]
    eng = engine(model, retemplate(tok, PAIR), ctx=1024)
    list(conversation(eng, as_is, turns=1))
    assert len(eng._slots[0].ckpts) == 2

    def boom(*a, **k):
        raise RuntimeError("mid-prefill")

    monkeypatch.setattr(prefill, "run", boom)
    with pytest.raises(RuntimeError):
        ask(eng, [{"role": "user", "content": FIRST + " more"}])
    assert eng._slots[0].ids == [] and len(eng._slots[0].ckpts) == 0


# ----------------------------------------------------------------- images --

import test_serving_gemma_vision as gmv  # noqa: E402
import test_serving_image_prompt as qiv  # noqa: E402
from test_serving_image_prompt import toy as toy_qiv  # noqa: E402,F401 — fixture by name
from drinkme.serving.image_prompt import ImagePrompt  # noqa: E402


@pytest.fixture(scope="module")
def gemma_vision_toys():
    """{window: toy} of test_serving_gemma_vision's image toy (bidirectional
    image runs on the sliding layers)."""
    from transformers import Gemma4ForConditionalGeneration

    out = {}
    for w in (16, 512):
        cfg = gmv.config(w)
        torch.manual_seed(0)
        model = Gemma4ForConditionalGeneration(cfg).eval().float()
        with torch.no_grad():
            model.model.vision_tower.std_bias.normal_()
            model.model.vision_tower.std_scale.uniform_(0.5, 1.5)
        for p in model.parameters():
            p.requires_grad_(False)
        out[w] = (None, model, gmv._tokenizer(), gmv._tower(cfg))
    return out


def _runs(eng, msgs, images, toy):
    """The [s, e) image runs of this prompt, expanded."""
    _ref, model, tok, tower = toy
    ids = eng.tokenize(messages=msgs)
    return [(s, e) for s, e, _img in ImagePrompt(ids, images, tower, model).runs]


@pytest.mark.parametrize("media", [0, cc.MEDIA_DEFAULT])
@pytest.mark.parametrize("window", [16, 512])
def test_no_gemma_checkpoint_lands_inside_an_image(gemma_vision_toys, window, media,
                                                   monkeypatch):
    """A prompt that ends right after its image (gemma's tool loop:
    `...<image|>` and the model writes next): n_prompt - 4 falls inside the
    bidirectional run and moves back to its start; n_prompt is after it, and
    so is the image's end, past `<image|>` (MEDIA ENDS), when media
    checkpoints are on. The next turn, parting from the slot after the
    prompt's end, restores the one at n_prompt: the image lies inside the
    reused prefix and the tower does not run, and the answer is the cold
    one."""
    toy = gemma_vision_toys[window]
    _ref, model, tok, tower = toy
    monkeypatch.setenv(cc.MEDIA_ENV, str(media))
    eng = gmv.engine(toy)
    msgs = [gmv.user("what is in this picture", None)]
    img = lambda: [gmv.image(40, 24, 21)]  # noqa: E731
    first = gmv.ask(eng, msgs, img(), n=6)
    n = first.prompt_tokens
    (s, e), = _runs(eng, msgs, img(), toy)
    assert s < n - 4 < e
    held = eng._slots[0].ckpts.positions()
    if window == 16:
        assert held == ([s, n] if media == 0 else [s, e + 1, n])
    assert not any(a < p < b for p in held for a, b in [(s, e)])
    gen = written(eng, first)
    hist = msgs + [{"role": "assistant", "content": trim(tok, gen)}, gmv.user("and the fox")]
    with gmv.TowerCalls(model) as calls:
        warm = gmv.ask(eng, hist, img())
    assert calls.n == 0 and warm.cached_tokens >= n
    assert warm.text == gmv.ask(gmv.engine(toy), hist, img()).text


@pytest.mark.parametrize("media,tower_cache", [(0, "0"), (0, "1"), (cc.MEDIA_DEFAULT, "0")])
@pytest.mark.parametrize("window", [16, 512])
def test_a_turn_that_parts_inside_the_images_markers_reuses_whole_runs_only(
        gemma_vision_toys, window, media, tower_cache, monkeypatch):
    """The next prompt parts from the slot right after the image's closing
    marker. Without media checkpoints the 16-window cache has only its
    checkpoints at the run's start and the prompt's end, so it restores the
    one at the start and runs the tower again, unless the tower cache holds
    the image (serving/tower_cache.py). With them it holds one at the
    image's end, past the marker (MEDIA ENDS), and restores that: the image
    is in the reused prefix and the tower does not run. The 512-window cache
    never wrapped, rewinds to the part point and skips the tower. Either
    way the answer is the cold one."""
    toy = gemma_vision_toys[window]
    _ref, model, tok, tower = toy
    monkeypatch.setenv(cc.MEDIA_ENV, str(media))
    monkeypatch.setenv("DRINKME_TOWER_CACHE_GIB", tower_cache)
    eng = gmv.engine(toy)
    msgs = [gmv.user("what is in this picture", None)]
    img = lambda: [gmv.image(40, 24, 21)]  # noqa: E731
    gmv.ask(eng, msgs, img(), n=6)
    (s, e), = _runs(eng, msgs, img(), toy)
    nxt = [gmv.user("what is in this picture", None), gmv.user("and the fox quick brown")]
    with gmv.TowerCalls(model) as calls:
        warm = gmv.ask(eng, nxt, img())
    if window == 16 and media == 0:
        assert warm.cached_tokens == s and calls.n == (1 if tower_cache == "0" else 0)
    else:
        assert warm.cached_tokens == e + 1 and calls.n == 0
    assert warm.text == gmv.ask(gmv.engine(toy), nxt, img()).text


def qwen_images():
    return [qiv.image(48, 32, 7)]


@pytest.mark.parametrize("media", [0, cc.MEDIA_DEFAULT])
@pytest.mark.parametrize("chunk,spec", [(0, "off"), (7, "off"), (7, "auto")])
def test_a_causal_run_follows_the_chunking_rule(chunk, spec, media, toy_qiv, monkeypatch):
    """Qwen3.5's image runs are causal. A run that fits the prefill chunk is
    one span (ImagePrompt.whole), so a checkpoint inside it moves to its
    start; a run longer than the chunk is cut like text, so one may land
    inside it, and without media checkpoints a restore there prefills the
    rest of the run from the whole image's output. MTP (the toy's head)
    after that restore too. With media checkpoints the next turn restores
    the one at the image's end (MEDIA ENDS), whatever the chunk. The answer
    is the cold one."""
    toy = toy_qiv
    _ref, model, tok, _head = toy
    monkeypatch.setenv(cc.MEDIA_ENV, str(media))
    eng = qiv.engine(toy, chunk=chunk, head=spec == "auto")
    msgs = [qiv.user("what is in this picture", None)]
    first = qiv.ask(eng, msgs, qwen_images(), spec=spec, n=6)
    n = first.prompt_tokens
    ip = ImagePrompt(eng.tokenize(messages=msgs), qwen_images(), qiv.TOWER, model)
    (s, e), = [(s, e) for s, e, _i in ip.runs]
    # this toy's config names no video ids, so its tower knows no
    # <|vision_end|> id and the image ends with its run
    end, = ip.media_ends()
    assert e - s > 7 and s < n - 4 < e and end == e
    held = eng._slots[0].ckpts.positions()
    assert (n - 4 in held) == (chunk == 7) and (s in held) == (chunk == 0)
    assert (end in held) == (media > 0)
    nxt = msgs + [qiv.user("and the fox")]
    warm = qiv.ask(eng, nxt, qwen_images(), spec=spec)
    if media:
        assert warm.cached_tokens == end
    else:
        assert warm.cached_tokens == (n - 4 if chunk == 7 else s)
    cold = qiv.ask(qiv.engine(toy, chunk=chunk, head=spec == "auto"), nxt, qwen_images(),
                   spec=spec)
    assert warm.text == cold.text


# ------------------------------------------------- the cold tier and sleep --

from drinkme.serving import slotstore  # noqa: E402

TOY_META = {"resolvedRevision": "toyrev0000000000000000000000000000000000"}
SAID = " ".join([LONG] * 12)   # ~290 words: past one 256-token block
OTHER = " ".join(["hello world zeta epsilon delta"] * 60)


def cold(model, tok, tmp_path, monkeypatch, ckpts=None, slots=1):
    monkeypatch.setenv("DRINKME_SLOT_DIR", str(tmp_path))
    return engine(model, tok, ckpts=ckpts, slots=slots, ctx=1024, meta=dict(TOY_META))


def test_an_evicted_slot_keeps_its_checkpoints_on_disk_and_serves_a_re_render(
        gemma_toys, tmp_path, monkeypatch):
    """llama.cpp's host prompt cache keeps a prompt's checkpoints with it, and
    finds a cached prompt by common prefix. The cold tier does both: A is
    evicted by an unrelated B with its two checkpoints beside it
    (checkpoints.safetensors, listed in the header), and A's next turn, which
    parts from A two tokens before its prompt's end, is served from disk by
    the checkpoint at n_prompt - 4, answering the cold answer."""
    model, tok, _ = gemma_toys[16]
    tok = retemplate(tok, PAIR)
    eng = cold(model, tok, tmp_path, monkeypatch)
    a = [{"role": "user", "content": SAID}]
    r1 = ask(eng, a)
    n = r1.prompt_tokens
    a = a + [{"role": "assistant", "content": as_is(tok, written(eng, r1))},
             {"role": "user", "content": "and the fox"}]
    ask(eng, [{"role": "user", "content": OTHER}])  # evicts A to the cold tier
    assert eng._cold.flush(30)
    heads = [json.load(open(os.path.join(d, "header.json")))
             for d in (os.path.join(eng._cold.dir, x) for x in os.listdir(eng._cold.dir))]
    stored = [h for h in heads if h["n_ids"] == n + 5]
    assert len(stored) == 1 and [c["n"] for c in stored[0]["checkpoints"]] == [n - 4, n]
    assert stored[0]["checkpoint_bytes"] > 0
    back = ask(eng, a)
    assert back.cached_tokens == n - 4 and eng._cold.hits == 1
    assert back.text == ask(engine(model, tok, ctx=1024), a).text


def test_with_checkpoints_off_the_cold_tier_is_extends_only(gemma_toys, tmp_path,
                                                            monkeypatch):
    model, tok, _ = gemma_toys[16]
    tok = retemplate(tok, PAIR)
    eng = cold(model, tok, tmp_path, monkeypatch, ckpts=0)
    a = [{"role": "user", "content": SAID}]
    r1 = ask(eng, a)
    a = a + [{"role": "assistant", "content": as_is(tok, written(eng, r1))},
             {"role": "user", "content": "and the fox"}]
    ask(eng, [{"role": "user", "content": OTHER}])
    assert eng._cold.flush(30)
    assert not any(os.path.exists(os.path.join(eng._cold.dir, x, slotstore.CKPT_FILE))
                   for x in os.listdir(eng._cold.dir))
    assert ask(eng, a).cached_tokens == 0


def test_a_slot_about_to_lose_most_of_itself_is_saved_and_the_store_asked_first(
        gemma_toys, tmp_path, monkeypatch):
    """llama.cpp's update_cache: B shares only a system prompt with A's slot,
    and the checkpoint at the start of A's last user message serves it that
    far, which keeps under half of A. So A is saved to the cold tier first,
    and B is served from the shared prefix. A's next turn, which B's slot
    serves only that far too, is served whole from the store."""
    model, tok, _ = gemma_toys[16]
    tok = retemplate(tok, PLAIN)
    eng = cold(model, tok, tmp_path, monkeypatch)
    system = {"role": "system", "content": " ".join([LONG] * 5)}
    a = [system, {"role": "user", "content": SAID}]
    r1 = ask(eng, a)
    head = len(eng._render(GenerationRequest(a, SampleParams()), upto=1).ids)
    assert head in eng._slots[0].ckpts.positions()
    held = len(eng._slots[0].ids)
    a = a + [{"role": "assistant", "content": as_is(tok, written(eng, r1))},
             {"role": "user", "content": "and the fox"}]
    b = [system, {"role": "user", "content": OTHER}]
    rb = ask(eng, b)
    assert rb.cached_tokens == head < held / 2
    assert rb.text == ask(engine(model, tok, ctx=1024), b).text
    assert eng._cold.flush(30)
    back = ask(eng, a)
    assert eng._cold.hits == 1 and back.cached_tokens == held
    assert back.text == ask(engine(model, tok, ctx=1024), a).text


def test_sleep_parks_the_checkpoints_with_their_slot(gemma_toys, tmp_path, monkeypatch):
    """A level-1 sleep parks the slots through the cold tier, checkpoints
    included; the wake restores both, and the next re-rendered turn is
    served by the checkpoint as it would have been awake."""
    model, tok, _ = gemma_toys[16]
    tok = retemplate(tok, PAIR)
    eng = cold(model, tok, tmp_path, monkeypatch)
    a = [{"role": "user", "content": SAID}]
    r1 = ask(eng, a)
    n = r1.prompt_tokens
    a = a + [{"role": "assistant", "content": as_is(tok, written(eng, r1))},
             {"role": "user", "content": "and the fox"}]
    before = eng._slots[0].ckpts.positions()
    st = eng.sleep(1)
    assert st["slots_persisted"] == 1 and len(eng._slots[0].ckpts) == 0
    st = eng.wake()
    assert st["slots_restored"] == 1 and eng._slots[0].ckpts.positions() == before
    back = ask(eng, a)
    assert back.cached_tokens == n - 4
    assert back.text == ask(engine(model, tok, ctx=1024), a).text


# --------------------------------------------------------------- the charge --

from drinkme import packs  # noqa: E402


@pytest.mark.parametrize("name,alloc", [("gemma16", 256), ("gemma64", 256), ("gemma512", 256),
                                        ("gemma512", 1024), ("qwen", 256), ("glimmer16", 256),
                                        ("glimmer512", 1024)])
def test_the_formula_is_a_full_snapshots_bytes(request, name, alloc):
    """ctx_checkpoints.config_bytes, the fit charge's unit, from config.json
    alone equals the bytes of a real snapshot taken past the window (every
    ring full) on each family, with the window capped by the allocation as
    StaticSlidingWindowLayer caps it. The toys are float32 throughout."""
    model, tok, _head, _ctx = family(request, name)
    eng = engine(model, tok, ctx=1024)
    ids = [(3 + 7 * i) % 90 for i in range(min(alloc, 600) - 8)]
    cache = _cache_after(model, eng, ids, alloc=alloc)
    with torch.inference_mode():
        snap = cc.snapshot(cache, task=1)
    cfg = model.config.to_dict()
    tc = cfg.get("text_config", cfg)
    window = tc.get("sliding_window") or 0
    if len(ids) >= min(window, alloc) or not window:
        assert snap.nbytes == cc.config_bytes(cfg, 4, alloc, state_bytes=4)
    else:
        assert snap.nbytes < cc.config_bytes(cfg, 4, alloc, state_bytes=4)


def test_the_menu_models_per_checkpoint_bytes():
    """The formula on the served configs' shapes (the report's table): bf16
    rings and conv states, float32 recurrent states. Literal configs, so the
    suite needs no download; they are the checkpoints' config.json fields."""
    gemma = {"text_config": {"layer_types": ["sliding_attention"] * 50 + ["full_attention"] * 10,
                             "num_attention_heads": 32, "num_key_value_heads": 16,
                             "head_dim": 256, "sliding_window": 1024}}
    qwen27 = {"text_config": {"layer_types": ["linear_attention"] * 48 + ["full_attention"] * 16,
                              "num_attention_heads": 24, "num_key_value_heads": 4, "head_dim": 256,
                              "linear_num_key_heads": 16, "linear_num_value_heads": 48,
                              "linear_key_head_dim": 128, "linear_value_head_dim": 128,
                              "linear_conv_kernel_dim": 4}}
    glimmer = {"text_config": {"layer_types": ["sliding_attention"] * 39 + ["full_attention"] * 13,
                               "num_attention_heads": 16, "num_key_value_heads": 2,
                               "head_dim": 128, "sliding_window": 2048}}
    assert cc.config_bytes(gemma) == 50 * (2 * 16 * 1024 * 256 * 2 + 8) == 838_861_200
    # the 27B's measured slot state, 154,927,232 B, less its 16 full layers'
    # 8-byte counters: the same 48 linear layers
    assert cc.config_bytes(qwen27) == 154_927_232 - 16 * 8
    assert cc.config_bytes(glimmer) == 39 * (2 * 2 * 2048 * 128 * 2 + 8) == 81_789_240
    assert cc.config_bytes({"layer_types": ["full_attention"] * 4, "num_attention_heads": 2,
                            "num_key_value_heads": 2, "head_dim": 8}) == 0


def test_the_picker_charges_the_checkpoints(monkeypatch):
    sliding = {"layer_types": ["sliding_attention", "full_attention"],
               "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 64,
               "sliding_window": 1024, "num_hidden_layers": 2}
    dense = {"num_hidden_layers": 2, "num_attention_heads": 2, "num_key_value_heads": 1,
             "head_dim": 64}
    cand = lambda name, cfg: packs.Candidate(name, f"org/{name}", None, 1.0, True, None,  # noqa
                                             True, cfg, None)
    one = cc.config_bytes(sliding, 2, 8192)
    f, = packs.rank_fits(100.0, [cand("s", sliding)], ctx=8192, slots=2)
    assert f.ckpt_gib == pytest.approx(one * cc.max_held(32, 8192) * 2 / packs.GIB)
    assert f.total_gib == pytest.approx(1.0 + f.kv_gib + f.ckpt_gib)
    assert packs.rank_fits(100.0, [cand("s", sliding)], 8192, 2, ckpts=0)[0].ckpt_gib == 0
    assert packs.rank_fits(100.0, [cand("d", dense)], 8192, 2)[0].ckpt_gib == 0
    line = packs.format_summary(100.0, "test", [f], [], "/r", 8192, 2)
    assert any("checkpoints" in x for x in line)
    monkeypatch.setenv(cc.ENV, "3")
    assert packs.checkpoints_estimate(None) == 3 and packs.checkpoints_estimate(7) == 7
    monkeypatch.setenv(cc.ENV, "junk")
    assert packs.checkpoints_estimate(None) == cc.DEFAULT_MAX


def test_the_picker_charges_a_vision_checkpoints_media_ends(monkeypatch):
    """A checkpoint with a vision tower may serve images, so the no-model
    charge counts the media-end checkpoints too (MEDIA ENDS)."""
    sliding = {"layer_types": ["sliding_attention", "full_attention"],
               "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 64,
               "sliding_window": 1024, "num_hidden_layers": 2}
    seeing = dict(sliding, vision_config={"depth": 2})
    one = cc.config_bytes(sliding, 2, 8192)
    assert packs.checkpoint_gib(sliding, 8192, 2, 32) == pytest.approx(one * 5 * 2 / packs.GIB)
    assert packs.checkpoint_gib(seeing, 8192, 2, 32) == pytest.approx(one * 7 * 2 / packs.GIB)
    monkeypatch.setenv(cc.MEDIA_ENV, "0")
    assert packs.checkpoint_gib(seeing, 8192, 2, 32) == pytest.approx(one * 5 * 2 / packs.GIB)
    monkeypatch.setenv(cc.MEDIA_ENV, "junk")
    assert packs.checkpoint_gib(seeing, 8192, 2, 32) == pytest.approx(one * 7 * 2 / packs.GIB)


def test_the_engine_announces_them_and_charges_them_in_its_residency_check(
        gemma_toys, monkeypatch, capsys):
    model, tok, _ = gemma_toys[64]
    seen = []
    monkeypatch.setattr(HFEngine, "_check_slot_residency",
                        lambda self, total, n: seen.append((total, n)))
    eng = engine(model, tok, ctx=1024, slots=2)
    out = capsys.readouterr().out
    one = eng.checkpoint_bytes()
    held = cc.max_held(32, 1024)
    assert f"context checkpoints: up to {held} per slot" in out and f"{one:,} B" in out
    fixed, ring, per_token = eng._slot_fixed_bytes, eng._slot_ring_bytes, eng._slot_token_bytes
    assert seen == [(2 * (fixed + ring + per_token * 1024 + held * one), 2)]
    engine(model, tok, ctx=1024, ckpts=0)
    assert "context checkpoints: off (--ctx-checkpoints 0)" in capsys.readouterr().out


@pytest.mark.parametrize("window", [16, 64, 80, 512])
def test_the_slot_charge_is_what_a_ctx_wide_slot_holds(gemma_toys, window):
    """A sliding layer holds min(window, ctx) rows at any ctx, and that is
    what the load-time charge counts: it equals the bytes of a real cache
    built ctx wide and written once. The probes (16 and 48 rows) sit below
    windows 64, 80 and 512, where differencing the whole cache would charge
    every ring per token of ctx."""
    model, tok, _ = gemma_toys[window]
    eng = engine(model, tok, ctx=1024)
    cache = eng._cache(1024)
    eng._prime_cache(cache)
    fixed, ring, per_token = eng._slot_fixed_bytes, eng._slot_ring_bytes, eng._slot_token_bytes
    assert fixed + ring + per_token * 1024 == engines.cache_bytes(cache)
    assert (ring > 0) == (window > engines.PROBE_SHORT)


# The real geometry, from config.json (asserted against the cached file when
# it is on the box): gemma-4-31B-it's 50 sliding layers (window 1,024, 16 kv
# heads x 256) and 10 full ones (4 kv heads x 512, attention_k_eq_v), and
# Muse-Glimmer-30B's 39 sliding layers (window 2,048) and 13 full ones, 2 kv
# heads x 128 throughout.
REAL_SHAPES = {
    "google/gemma-4-31B-it": ("Gemma4TextConfig", dict(
        num_hidden_layers=60, hidden_size=5376, num_attention_heads=32,
        num_key_value_heads=16, head_dim=256, num_global_key_value_heads=4,
        global_head_dim=512, attention_k_eq_v=True, sliding_window=1024,
        layer_types=(["sliding_attention"] * 5 + ["full_attention"]) * 10)),
    "meta-models/Muse-Glimmer-30B": ("MuseGlimmerTextConfig", dict(
        num_hidden_layers=52, hidden_size=6656, num_attention_heads=32,
        num_key_value_heads=2, head_dim=128, sliding_window=2048,
        layer_types=(["sliding_attention"] * 3 + ["full_attention"]) * 13)),
}


def _real_text_config(repo):
    import transformers
    from huggingface_hub import try_to_load_from_cache

    name, fields = REAL_SHAPES[repo]
    if not hasattr(transformers, name):
        pytest.skip(f"this transformers has no {name}")
    hit = try_to_load_from_cache(repo, "config.json")
    if isinstance(hit, str):
        with open(hit) as f:
            real = json.load(f)
        real = real.get("text_config", real)
        assert {k: real[k] for k in fields if k in real} == {
            k: (list(v) if isinstance(v, list) else v) for k, v in fields.items() if k in real}
    return getattr(transformers, name)(**fields)


@pytest.mark.parametrize("repo,ctx,fixed,ring,per_token", [
    # 60 and 52 layers' 8-byte counters; the rings at the window, bf16:
    # 50 x 2 x 16 x 256 x 2 B x 1,024 and 39 x 2 x 2 x 128 x 2 B x 2,048;
    # the full layers per token: 10 x 2 x 4 x 512 x 2 B and 13 x 2 x 2 x 128 x 2 B
    ("google/gemma-4-31B-it", 8192, 480, 838_860_800, 81_920),
    ("google/gemma-4-31B-it", 131072, 480, 838_860_800, 81_920),
    ("meta-models/Muse-Glimmer-30B", 8192, 416, 81_788_928, 13_312),
    # a ctx inside the window: every ring spans the ctx, so it is per token
    ("meta-models/Muse-Glimmer-30B", 1024, 416, 0, 13_312 + 39_936),
])
def test_the_slot_charge_on_the_real_shapes(repo, ctx, fixed, ring, per_token):
    """The engine's measurement over a cache of the real config, each layer
    written once in bf16 at its own heads and width (per_layer_config), no
    weights. Differencing the whole cache charged gemma-4-31B 480 B +
    901,120 B/token (6.88 GiB at ctx 8,192) for 1.41 GiB of slot."""
    from drinkme.serving.kvcache import LiveStaticCache

    cfg = _real_text_config(repo)
    per_layer = list(cfg.per_layer_config)

    def prime(cache):
        for layer, lc in zip(cache.layers, per_layer):
            kv = torch.zeros(1, lc.num_key_value_heads, 1, lc.head_dim, dtype=torch.bfloat16)
            layer.lazy_initialization(kv, kv)

    def make(width):
        return LiveStaticCache(config=cfg, max_cache_len=width)

    layers = engines.measure_slot_layers(make, prime, engines.PROBE_SHORT, engines.PROBE_LONG)
    assert engines.slot_cost_at(layers, make(ctx), ctx) == (fixed, ring, per_token)
