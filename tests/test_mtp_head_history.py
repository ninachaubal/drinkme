"""The MTP head's KV kept with its prefix slot between requests — CPU
only, on the toy hybrid tests/test_serving_mtp.py builds.

The finding: `Speculator.open` reset the head on every request, so a
request served from a prefix slot seeded the head from the reuse point on
and drafted over a shorter history than the trunk attends over. Measured
on the served 27B: 75% of drafts accepted warm against 90% cold, 64 against
73 tok/s. The slot now keeps the head's KV (mtp.HeadKV) and the engine
crops it at the reuse point the way it rewinds the trunk's cache. Drafts
only propose, so what these tests pin is the history itself across every
slot operation:

  * HeadKV.keep crops rows and spans across holes and drops a tail at or
    past the reuse point;
  * a repeated request's head history runs from entry 0, its rows are the
    ones a head fed every position in one pass holds, and its tokens agree
    with the old behaviour's (spec_agree, modulo a near-tie);
  * a turn that extends the whole slot fills entry r - 1 from the tail;
  * a prompt that parts from the slot keeps nothing at or past the reuse
    point, and nothing at r - 1 when the slot's token at r differs;
  * a stop inside a cycle leaves entries 0..written-2, no more;
  * a slot restored from the cold tier or rebuilt bigger starts the head
    fresh, a request without the head keeps the slot's history for the
    next, a request that ends serial stores a history the slot can use,
    sleep and a failed request drop it;
  * the residency line and charge carry the head's bytes per token.
"""

from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest
import torch

from drinkme.serving import engines, mtp
from drinkme.serving.engine import GenerationRequest, SampleParams, complete
from drinkme.serving.engines import HFEngine
from spec_agree import assert_agrees_or_forks_at_a_near_tie, capture, ordered_ids

from test_serving_mtp import (  # noqa: F401 — fixtures by name
    WORDS, _torch_chunk_on_cpu, mtp_depth, toy,
)
from test_serving_image_prompt import toy as image_toy  # noqa: F401 — fixture by name
from test_serving_prefill_bound import env

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")

PROMPT = " ".join(WORDS[2:14] * 2)
TOY_META = {"resolvedRevision": "toyrev0000000000000000000000000000000000"}


_TOKS: dict = {}


def _tokenizer(tok):
    """The toy's tokenizer with </s> matched whole, as <unk> and <pad> are:
    the random toy emits it, and a history re-rendered from the slot's ids
    has to tokenize back to them (`reply`)."""
    if id(tok) not in _TOKS:
        t = copy.deepcopy(tok)
        t.add_special_tokens({"additional_special_tokens": ["</s>"]})
        _TOKS[id(tok)] = t
    return _TOKS[id(tok)]


def engine(toy, *, head=True, slots=1, ckpts=None, meta=None):
    model, tok, h, _ = toy
    with env(DRINKME_PREFIX_SLOTS=slots, DRINKME_CTX_CHECKPOINTS=ckpts):
        return HFEngine(model, _tokenizer(tok), model_id="toy", arm="test", meta=meta or {},
                        ctx=512, mtp_head=h if head else None)


def user(text):
    return {"role": "user", "content": text}


def run(eng, msgs, n=12, **kw):
    """(result, generated ids, decision rows): spec_agree's capture."""
    picks, rows, restore = capture(engines)
    try:
        res = complete(eng, GenerationRequest(
            list(msgs), SampleParams(temperature=0.0, max_tokens=n, **kw)))
    finally:
        restore()
    return res, ordered_ids(picks), rows


def reply(eng, res):
    """The assistant turn that re-renders to exactly the ids the slot holds
    (special tokens included; tests/test_serving_slotstore.py's agent_turn)."""
    slot = max(eng._slots, key=lambda s: s.stamp)
    gen = slot.ids[res.prompt_tokens:]
    text = eng.tok.decode(gen, skip_special_tokens=False)
    assert eng.tok(text, add_special_tokens=False)["input_ids"] == gen
    return {"role": "assistant", "content": text}


@pytest.fixture
def opened(monkeypatch):
    """What every Speculator.open was handed and left: the history's spans
    and whether it carried a tail (None for no history), `start`, and the
    head's spans once open returned (before the prefill seeds)."""
    seen = []
    real = mtp.Speculator.open

    def spy(self, cache, ids, image=None, history=None, start=0):
        had = None if history is None else ([list(s) for s in history.spans],
                                            history.tail is not None)
        real(self, cache, ids, image=image, history=history, start=start)
        seen.append(SimpleNamespace(history=had, start=start,
                                    spans=[list(s) for s in self._spans]))

    monkeypatch.setattr(mtp.Speculator, "open", spy)
    return seen


def reference_kv(toy, ids):
    """The head's keys and values over entries 0..len(ids)-2 as a head fed
    every position in one pass holds them (the re-arm test's reference):
    one trunk forward for the hiddens, one head pass for the entries."""
    from transformers import StaticCache

    model, _, head, cfg = toy
    w = len(ids)
    seq = torch.tensor([ids])
    saved = head.cache, head.entries
    try:
        with torch.inference_mode():
            cache = StaticCache(config=cfg, max_cache_len=w + 8)
            hidden, _ = mtp.forward_with_hidden(model, seq, cache, torch.arange(w))
            head.reset()
            head.run(hidden[:, :w - 1], seq[:, 1:w], 0)
            layer = head.cache.layers[0]
            return layer.keys.clone(), layer.values.clone(), hidden
    finally:
        head.cache, head.entries = saved


def assert_rows_are(toy, kv, ids, entries):
    """`kv`'s rows are the reference's rows for `entries`, in order."""
    k, v, _ = reference_kv(toy, ids)
    layer = kv.cache.layers[0]
    assert layer.keys.shape[2] == kv.entries == len(entries)
    torch.testing.assert_close(layer.keys, k[:, :, entries], atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(layer.values, v[:, :, entries], atol=1e-4, rtol=1e-4)


def entries_of(spans):
    return [p for a, b in spans for p in range(a, b)]


# ------------------------------------------------------------ HeadKV.keep --


def _kv(head, spans, tail=None):
    """A HeadKV whose row i holds the value of the entry it stands for, so a
    crop can be read back off the tensor."""
    from transformers import DynamicCache

    cache = DynamicCache(config=head.cfg)
    entries = entries_of(spans)
    if entries:
        cfg = head.cfg
        x = torch.tensor(entries, dtype=torch.float32).view(1, 1, -1, 1)
        x = x.expand(1, cfg.num_key_value_heads, -1, cfg.head_dim).contiguous()
        cache.layers[0].update(x, -x)
    return mtp.HeadKV(cache, len(entries), [list(s) for s in spans], tail)


def _held(kv):
    layer = kv.cache.layers[0]
    if not getattr(layer, "is_initialized", False):
        return []
    return [int(t) for t in layer.keys[0, 0, :, 0]]


def test_keep_crops_rows_and_spans_across_holes(toy):
    head = toy[2]
    kv = _kv(head, [[0, 3], [5, 8], [10, 12]])
    kv.keep(6, 6)
    assert kv.spans == [[0, 3], [5, 6]] and kv.entries == 4
    assert _held(kv) == [0, 1, 2, 5]
    assert torch.equal(kv.cache.layers[0].values, -kv.cache.layers[0].keys)
    kv.keep(4, 4)  # inside a hole: the span before it stays whole
    assert kv.spans == [[0, 3]] and _held(kv) == [0, 1, 2]
    kv.keep(9, 9)  # past everything: nothing to drop
    assert kv.spans == [[0, 3]] and _held(kv) == [0, 1, 2]
    kv.keep(0, 0)
    assert kv.spans == [] and kv.entries == 0 and _held(kv) == []
    assert kv.nbytes == 0


def test_keep_drops_the_tail_at_or_past_the_reuse_point(toy):
    head = toy[2]
    h = torch.zeros(1, 1, head.cfg.hidden_size)
    kv = _kv(head, [[0, 7]], tail=(7, h))
    kv.keep(7, 8)  # the next request extends all 8 positions: h_7 holds
    assert kv.tail is not None and kv.tail[0] == 7 and kv.spans == [[0, 7]]
    kv.keep(6, 7)  # it keeps 7: position 7 is rewritten, h_7 goes
    assert kv.tail is None and kv.spans == [[0, 6]]


def test_keep_on_an_empty_history(toy):
    from transformers import DynamicCache

    head = toy[2]
    for kv in (mtp.HeadKV(DynamicCache(config=head.cfg), 0, []), mtp.HeadKV(None, 0, [])):
        kv.keep(5, 5)
        assert kv.entries == 0 and kv.spans == [] and kv.nbytes == 0


# ------------------------------------------------ the history across requests --


def test_a_repeated_request_keeps_the_heads_history_from_entry_zero(toy, opened, monkeypatch):
    """The same prompt twice on one slot. The DeltaNet state cannot rewind,
    so the context checkpoint at n - 4 serves the second request, and the
    head keeps entries 0..r-1 (the slot's token at r is the prompt's) —
    where it used to start at r. Its history at the end runs 0..written-2
    with no hole, row for row the reference's, and the tokens agree with
    the first request's and with a run whose history is thrown away."""
    msgs = [user(PROMPT)]
    with mtp_depth(4):
        eng = engine(toy)
        first, ids1, rows1 = run(eng, msgs)
        second, ids2, rows2 = run(eng, msgs)
    r, n = second.cached_tokens, second.prompt_tokens
    assert 0 < r < n
    o = opened[-1]
    assert o.start == r and o.history == ([[0, r]], False) and o.spans == [[0, r]]
    slot = eng._slots[0]
    w = len(slot.ids)
    assert slot.head.spans == [[0, w - 1]] and slot.head.tail[0] == w - 1
    assert_rows_are(toy, slot.head, slot.ids, list(range(w - 1)))
    assert_agrees_or_forks_at_a_near_tie(ids2, ids1, rows2, rows1)
    # the old behaviour: no history kept, the head starts at the reuse point
    monkeypatch.setattr(mtp.Speculator, "history", lambda self, written: None)
    with mtp_depth(4):
        old = engine(toy)
        run(old, msgs)
        third, ids3, rows3 = run(old, msgs)
    assert third.cached_tokens == r
    assert opened[-1].history is None and opened[-1].spans == []
    assert_agrees_or_forks_at_a_near_tie(ids2, ids3, rows2, rows3)


def test_a_turn_that_extends_the_whole_slot_fills_the_join_from_the_tail(toy, opened):
    """Turn 2 re-sends turn 1 and adds to it: the slot is reused whole
    (r = written), turn 1 stored entries 0..r-2 and h_{r-1}, and entry
    r-1 = (h_{r-1}, t_r) is built from them at open, so the history has no
    hole where the suffix joins it."""
    hist = [user(PROMPT)]
    with mtp_depth(4):
        eng = engine(toy)
        first, _, _ = run(eng, hist)
        slot = eng._slots[0]
        w1 = len(slot.ids)
        assert slot.head.spans == [[0, w1 - 1]] and slot.head.tail[0] == w1 - 1
        hist += [reply(eng, first), user("and the fox")]
        second, _, _ = run(eng, hist)
    assert second.cached_tokens == w1  # extends all of it
    o = opened[-1]
    assert o.start == w1 and o.history == ([[0, w1 - 1]], True)
    assert o.spans == [[0, w1]]  # the tail filled entry w1 - 1
    w2 = len(slot.ids)
    assert slot.head.spans == [[0, w2 - 1]]
    assert_rows_are(toy, slot.head, slot.ids, list(range(w2 - 1)))


@pytest.mark.parametrize("same", [True, False], ids=["same-token-at-r", "new-token-at-r"])
def test_a_prompt_that_parts_from_the_slot_keeps_nothing_past_the_reuse_point(
        toy, opened, same):
    """Turn 2 parts from the slot inside turn 1's reply, and the checkpoint at
    turn 1's prompt end (r) serves it. Entry r - 1 = (h_{r-1}, t_r) stays
    when the prompt's token at r is the slot's (the reply trimmed by its
    last token) and goes when it is not (a different reply); nothing at or
    past r survives either way, nor turn 1's tail."""
    hist = [user(PROMPT)]
    with mtp_depth(4):
        eng = engine(toy)
        first, _, _ = run(eng, hist)
        slot = eng._slots[0]
        r = first.prompt_tokens
        gen = slot.ids[r:]
        assert len(gen) >= 2
        if same:
            text = eng.tok.decode(gen[:-1], skip_special_tokens=False)
        else:
            text = next(t for t in ("zeta", "delta") if eng.tok.convert_tokens_to_ids(t) != gen[0])
        hist += [{"role": "assistant", "content": text}, user("and the fox")]
        second, _, _ = run(eng, hist)
    assert second.cached_tokens == r
    k = r if same else r - 1
    o = opened[-1]
    assert o.start == r and o.history == ([[0, k]], False) and o.spans == [[0, k]]
    w2 = len(slot.ids)
    want = [[0, w2 - 1]] if same else [[0, r - 1], [r, w2 - 1]]
    assert slot.head.spans == want
    assert_rows_are(toy, slot.head, slot.ids, entries_of(want))


def test_a_stop_inside_a_cycle_leaves_entries_up_to_written_minus_two(toy, opened,
                                                                     monkeypatch):
    """A draft that is always right (the serial transcript's own next
    tokens) makes every cycle accept all k, and the head rebuilds an entry
    per accepted draft. A stop string that lands inside a cycle rewinds the
    cache past the rest of it (finish), and the head's entries go with it:
    the slot keeps 0..written-2, row for row the reference's, and the tail
    is the trunk's hidden state at written - 1."""
    model, tok, head, _ = toy
    msgs = [user(PROMPT)]
    with mtp_depth(0):
        ref, oracle, _ = run(engine(toy, head=False, slots=0), msgs, n=16)
    # the cycle from token 0 decides tokens 1..5, the next 6..10; the stop
    # scanner sees a word once the text after it rules out a longer match,
    # a token or so later, so a stop on 1 or 2 (6 or 7) lands inside its
    # cycle. A word the visible text carries (not a special id) that the
    # text before it does not
    words = [tok.decode([t]) for t in oracle]
    at = next(i for i in (1, 2, 6, 7)
              if (words[i] in WORDS or words[i].startswith("tok"))
              and words[i] not in " ".join(words[:i]))
    stop = words[at]
    n_prompt = ref.prompt_tokens
    finished = []
    real_finish = mtp.Speculator.finish

    def finish(self, appended):
        finished.append((appended, self._rows))
        return real_finish(self, appended)

    def draft(h, token, first_entry, k):
        type(head).draft(head, h, token, first_entry, k)  # the real chain, for the KV
        pos = first_entry + 1 - n_prompt
        return (oracle[pos + 1:pos + 1 + k] + [0] * k)[:k]

    monkeypatch.setattr(mtp.Speculator, "finish", finish)
    head.draft = draft
    try:
        with mtp_depth(4):
            eng = engine(toy)
            res, ids, _ = run(eng, msgs, n=16, stop=[stop])
    finally:
        del head.draft
    assert res.finish_reason == "stop" and ids[:at + 1] == oracle[:at + 1]
    appended, rows = finished[-1]
    assert appended < rows - 1  # the stop landed inside the cycle
    slot = eng._slots[0]
    w = len(slot.ids)
    assert slot.head.spans == [[0, w - 1]] and slot.head.entries == w - 1
    assert_rows_are(toy, slot.head, slot.ids, list(range(w - 1)))
    # the tail is the verify row the stop landed on, not the cycle's last
    _, _, hidden = reference_kv(toy, slot.ids)
    assert slot.head.tail[0] == w - 1
    torch.testing.assert_close(slot.head.tail[1], hidden[:, w - 1:w], atol=1e-4, rtol=1e-4)


# ------------------------------------------------------- slot operations --


LONG = " ".join(WORDS[2:16] * 21)       # ~290 tokens: past one 256-token block
OTHER = " ".join(["hello world zeta epsilon delta"] * 58)


def test_a_slot_restored_from_the_cold_tier_starts_the_head_fresh(
        toy, opened, tmp_path, monkeypatch):
    """The cold tier stores the trunk's cache and not the head's: A's slot
    goes to disk when B takes the one slot, and A's next turn, restored
    from there, opens the head with no history and seeds it from the reuse
    point on."""
    monkeypatch.setenv("DRINKME_SLOT_DIR", str(tmp_path))
    hist_a, hist_b = [user(LONG)], [user(OTHER)]
    with mtp_depth(4):
        eng = engine(toy, meta=dict(TOY_META))
        assert eng._cold is not None
        a1, _, _ = run(eng, hist_a, n=6)
        assert eng._slots[0].head is not None
        hist_a += [reply(eng, a1), user("and the fox")]
        run(eng, hist_b, n=6)  # evicts A to the cold tier
        assert eng._cold.flush(30)
        back, _, _ = run(eng, hist_a, n=6)
    assert eng._cold.hits == 1 and back.cached_tokens > 0
    o = opened[-1]
    assert o.history is None and o.spans == []
    assert eng._slots[0].head.spans[0][0] == back.cached_tokens


def test_a_slot_that_outgrows_its_allocation_starts_the_head_fresh(toy, opened):
    """A request the slot's allocation cannot hold rebuilds the cache bigger
    and prefills cold: the head's history goes with the old cache."""
    hist = [user(PROMPT)]
    with mtp_depth(4):
        eng = engine(toy)
        eng.KV_FLOOR = 64
        first, _, _ = run(eng, hist, n=6)
        assert eng._slots[0].alloc == 64 and eng._slots[0].head is not None
        hist += [reply(eng, first), user(PROMPT)]
        second, _, _ = run(eng, hist, n=6)
    assert second.cached_tokens == 0 and eng._slots[0].alloc == 128
    o = opened[-1]
    assert o.history is None and o.start == 0
    assert eng._slots[0].head.spans == [[0, len(eng._slots[0].ids) - 1]]


def test_a_request_without_the_head_keeps_the_slots_history_for_the_next(toy, opened):
    """Turn 2 runs serially (DRINKME_SPEC=off): the head never runs, and the
    slot keeps turn 1's history, cropped at the reuse point. Turn 3 drafts
    over it again, with a hole where turn 2 ran without the head."""
    hist = [user(PROMPT)]
    with mtp_depth(4):
        eng = engine(toy)
        first, _, _ = run(eng, hist)
    slot = eng._slots[0]
    w1 = len(slot.ids)
    kept = slot.head
    hist += [reply(eng, first), user("and the fox")]
    eng._spec_plan = None  # the plan is read once per engine; read it again
    with mtp_depth(0):
        second, _, _ = run(eng, hist)
    assert second.cached_tokens == w1 and eng.last_spec_stats is None
    assert slot.head is kept and kept.spans == [[0, w1 - 1]] and kept.entries == w1 - 1
    w2 = len(slot.ids)
    hist += [reply(eng, second), user("over the lazy dog")]
    eng._spec_plan = None
    with mtp_depth(4):
        third, _, _ = run(eng, hist)
    assert third.cached_tokens == w2
    o = opened[-1]
    assert o.history == ([[0, w1 - 1]], True) and o.spans == [[0, w1 - 1]]
    w3 = len(slot.ids)
    assert slot.head.spans == [[0, w1 - 1], [w2, w3 - 1]]
    assert_rows_are(toy, slot.head, slot.ids, entries_of(slot.head.spans))


@pytest.mark.parametrize("rearm", [0, None], ids=["latch", "rearm-pending"])
def test_a_request_that_ends_serial_stores_a_history_the_slot_can_use(toy, opened, rearm):
    """The adaptive bail trips on the random head and the request ends on the
    serial loop. The head stops at the trip, so the slot keeps entries up
    to there and none past written - 1. On the latch's loop the steps are
    the engine's own and the speculator never saw the last hidden state: no
    tail. With a re-arm pending the stretch kept every serial row, so the
    tail is h_{written-1}, the trunk's own, and the next turn fills its
    entry from it."""
    hist = [user(PROMPT)]
    with env(DRINKME_MTP_BAIL_WINDOW=8, DRINKME_MTP_BAIL_FLOOR=0.9,
             DRINKME_MTP_REARM=rearm), mtp_depth(4):
        eng = engine(toy)
        first, _, _ = run(eng, hist, n=24)
        s = eng.last_spec_stats
        assert s["trips"] == 1 and s["serial_at_end"]
        slot = eng._slots[0]
        w = len(slot.ids)
        kv = slot.head
        assert kv.spans and kv.spans[0][0] == 0 and kv.spans[-1][1] <= w - 1
        assert kv.entries == len(entries_of(kv.spans))
        assert_rows_are(toy, kv, slot.ids, entries_of(kv.spans))
        if rearm == 0:
            assert kv.tail is None
            return
        assert kv.tail[0] == w - 1
        _, _, hidden = reference_kv(toy, slot.ids)
        torch.testing.assert_close(kv.tail[1], hidden[:, w - 1:w], atol=1e-4, rtol=1e-4)
        stored = [list(x) for x in kv.spans]
        hist += [reply(eng, first), user("and the fox")]
        second, _, _ = run(eng, hist)
    assert second.cached_tokens == w
    assert opened[-1].spans == stored + [[w - 1, w]]


def test_sleep_and_a_failed_request_drop_the_slots_history(toy, monkeypatch):
    """A slot's history goes where its ids go: parked by sleep (the cold
    tier does not store it) and emptied by a request that raised."""
    msgs = [user(PROMPT)]
    with mtp_depth(4):
        eng = engine(toy)
        run(eng, msgs)
        assert eng._slots[0].head is not None and toy[2].cache is None  # detached
        eng._park_slots()
        assert eng._slots[0].head is None
        run(eng, msgs)
        assert eng._slots[0].head is not None

        def boom(self, *a, **kw):
            raise RuntimeError("boom")

        monkeypatch.setattr(mtp.Speculator, "cycle", boom)
        with pytest.raises(RuntimeError, match="boom"):
            run(eng, msgs)
    assert eng._slots[0].head is None and eng._slots[0].ids == []


# ------------------------------------------------------------ residency --


def _charged(toy, monkeypatch, capsys, **kw):
    seen = []
    monkeypatch.setattr(HFEngine, "_check_slot_residency",
                        lambda self, total, n: seen.append((total, n)))
    capsys.readouterr()
    eng = engine(toy, ckpts=0, **kw)
    line = [x for x in capsys.readouterr().out.splitlines() if "prefix cache:" in x]
    assert len(line) == 1
    return eng, line[0], seen


@pytest.mark.parametrize("slots", [1, 0])
def test_the_residency_charge_carries_the_heads_kv_per_token(toy, monkeypatch, capsys, slots):
    """2 (K and V) x 1 kv head x 32 x 4 B (float32) = 256 B per position on
    the toy, on top of the trunk's measured figure, in the line and in the
    guard's total — per slot, or for the one per-request cache with the
    prefix cache off. The formula is the real tensors': a slot's HeadKV
    holds exactly that per entry."""
    _, _, head, cfg = toy
    eng, line, seen = _charged(toy, monkeypatch, capsys, slots=slots)
    head_b = eng._head_token_bytes()
    assert head_b == 2 * cfg.num_key_value_heads * cfg.head_dim * 4 == 256
    fixed, ring, per_token = eng._slot_fixed_bytes, eng._slot_ring_bytes, eng._slot_token_bytes
    per_slot = fixed + ring + (per_token + head_b) * eng.ctx
    off = "" if slots else " (prefix cache OFF — per request)"
    assert line == (f"[drinkme] prefix cache: 1 slot x {per_slot:,.0f} B ({fixed:,.0f} B state "
                    f"+ {per_token + head_b:,.0f} B/token x ctx {eng.ctx}, the MTP head's KV "
                    f"256 B/token of it) = {engines._human(per_slot)} at full width{off}")
    assert seen == [(per_slot, 1)]
    if slots:
        with mtp_depth(4):
            run(eng, [user(PROMPT)])
        kv = eng._slots[0].head
        assert kv.entries > 0 and kv.nbytes == kv.entries * head_b


def test_without_a_head_the_residency_line_is_unchanged(toy, monkeypatch, capsys):
    eng, line, seen = _charged(toy, monkeypatch, capsys, head=False)
    assert eng._head_token_bytes() == 0
    fixed, per_token = eng._slot_fixed_bytes, eng._slot_token_bytes
    per_slot = fixed + eng._slot_ring_bytes + per_token * eng.ctx
    assert line == (f"[drinkme] prefix cache: 1 slot x {per_slot:,.0f} B ({fixed:,.0f} B state "
                    f"+ {per_token:,.0f} B/token x ctx {eng.ctx}) = "
                    f"{engines._human(per_slot)} at full width")
    assert seen == [(per_slot, 1)]


def test_an_image_turn_that_extends_the_slot_fills_the_join_at_the_trunks_position(
        image_toy, monkeypatch):
    """After an image prompt the head's entries inside the prompt take the
    trunk's M-RoPE rows (ImagePrompt.head_positions): so does the entry the
    tail fills at the join, and the extended turn answers the cold answer."""
    import test_serving_image_prompt as qiv
    from drinkme.serving.image_prompt import ImagePrompt

    toy = image_toy
    _ref, model, tok, head = toy
    seen = []
    real = type(head).run

    def spy(self, hidden, token_ids, first_entry, positions=None):
        seen.append((first_entry, None if positions is None else positions.clone(),
                     token_ids.shape[1]))
        return real(self, hidden, token_ids, first_entry, positions=positions)

    monkeypatch.setattr(type(head), "run", spy)
    eng = qiv.engine(toy, head=True)
    hist = [qiv.user("what is in this picture", None)]
    first = qiv.ask(eng, hist, [qiv.image(16, 24, 21)], "auto", n=4)
    hist = hist + [reply(eng, first), qiv.user("and the fox")]
    w = len(eng._slots[0].ids)
    seen.clear()
    warm = qiv.ask(eng, hist, [qiv.image(16, 24, 21)], "auto")
    assert warm.cached_tokens == w
    ip = ImagePrompt(qiv.rendered(tok, hist), [qiv.image(16, 24, 21)], qiv.TOWER, model)
    first_entry, pos, t = seen[0]  # the tail's entry, before the prefill seeds
    assert t == 1 and first_entry == w - 1
    assert torch.equal(pos[:, 0].cpu(), torch.from_numpy(ip.pos4[1:, w - 1:w]))
    cold = qiv.ask(qiv.engine(toy, head=True), hist, [qiv.image(16, 24, 21)], "auto")
    assert warm.text == cold.text
