"""Speculation on a tiny random-init gemma-4 — the sliding-window rewind
. CPU only, no downloads, no GPU.

The bug, measured on the real 31B: with `DRINKME_SPEC=auto`
(n-gram, gemma has no head) a greedy transcript had CHARACTERS MISSING —
`$17 \\times 2 = 340$` where the serial loop wrote `$17 \\times 20 = 340$` —
and fell apart a rejection at a time; `DRINKME_SPEC=off` was clean. Two
hypotheses were on the table: (a) the incremental detokenizer loses text
under multi-token acceptance, (b) the KV rewind after a rejected draft is
wrong on gemma-4's `sliding_attention` cache layers. Separating them by
TOKEN IDS on this toy picked (b): ids diverge one position after the first
rejection, at every accept depth, in every window regime — while the detok,
replayed over the real gemma and Qwen3-8B tokenizers on the clean receipt's
text, reproduces the one-shot decode for every chunk partition (the tests
at the bottom carry that evidence).

WHY (b): transformers' `StaticSlidingWindowLayer` keeps a python
`cumulative_length_int` beside the `cumulative_length` tensor, and it is the
INT that `get_seq_length()` returns — which is where masking_utils takes the
mask's q_offset from (`cache_position` is documented "deprecated and
unused"). serving/mtp.py's rewind subtracted from the tensor alone, so after
a rejection every sliding-layer query was masked `drop` positions late and
attended the rejected drafts' stale rows: the model really had "seen" the
token it never emitted. Once the window is full the write path is also a
RING that shifts the oldest rows out for good, so a rewind there needs the
rows back — serving/mtp.py's docstring, surface 1b, carries the fix.

The toy is a genuine `Gemma4ForCausalLM` with the real 31B's layer pattern
(five `sliding_attention` per `full_attention`, read off the cached
config.json when it is there), tiny dims, float32 (the qwen toy's reasoning:
bf16 near-ties flip for reasons that have nothing to do with the feature).
Three sliding windows put the ring in each regime the 31B's 1024 reaches at
different context lengths: 16 (full from the prompt on), 64 and 80 (fills
DURING the generation), 512 (never full — the receipt's own regime at ~230
tokens). The proposer is either the real lookup at prompt_lookup_min=1 (so
it fires on a 96-token vocabulary) or an ORACLE that drafts the serial run's
own next j tokens and then a wrong one — forced multi-token acceptance and
a rejection at depth j, every cycle, which random weights cannot produce.
"""

from __future__ import annotations

import json
import os
import random

import pytest
import torch

from drinkme.serving import engines, mtp, ngram
from drinkme.serving.detok import IncrementalDetok, SuffixWindow
from drinkme.serving.engine import GenerationRequest, SampleParams, complete
from drinkme.serving.engines import HFEngine

from test_serving_capability import GEMMA, QWEN8B, _cached_tokenizer
from test_serving_mtp import WORDS, _msgs  # noqa: F401
from test_serving_ngram import spec_env

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")

# The 31B's pattern, the first 12 of 60: 5:1 sliding:full, full last. Read
# from the cached config when it is on the box; asserted equal to this
# literal so the toy never silently drifts from the real geometry.
LAYER_TYPES = (["sliding_attention"] * 5 + ["full_attention"]) * 2
WINDOWS = (16, 64, 80, 512)
REPEAT = " ".join(["the quick brown fox jumps over the lazy dog"] * 6)


def _real_layer_types() -> list[str] | None:
    from huggingface_hub import try_to_load_from_cache

    hit = try_to_load_from_cache(GEMMA[0], "config.json", revision=GEMMA[1])
    if not isinstance(hit, str):
        return None
    with open(hit) as f:
        return json.load(f)["text_config"]["layer_types"]


def _cfg(sliding_window: int):
    from transformers import Gemma4TextConfig

    real = _real_layer_types()
    if real is not None:
        assert real[:12] == LAYER_TYPES and real[-1] == "full_attention"
    return Gemma4TextConfig(
        vocab_size=96, hidden_size=64, intermediate_size=128,
        num_hidden_layers=12, num_attention_heads=2, num_key_value_heads=1,
        head_dim=32, global_head_dim=32, num_global_key_value_heads=1,
        layer_types=list(LAYER_TYPES), sliding_window=sliding_window,
        max_position_embeddings=1024, final_logit_softcapping=30.0,
        attention_k_eq_v=True, hidden_size_per_layer_input=0,
        vocab_size_per_layer_input=96, tie_word_embeddings=True,
        # no EOS: a random argmax would hit one every ~96 tokens and truncate
        # the comparison runs (the qwen toy's reasoning)
        eos_token_id=None, pad_token_id=None)


def _word_tokenizer(vocab_size: int):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast

    vocab = {"<unk>": 0, "<pad>": 1, "</s>": 2}
    for w in WORDS:
        vocab[w] = len(vocab)
    while len(vocab) < vocab_size:
        vocab[f"tok{len(vocab)}"] = len(vocab)
    backend = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
    backend.pre_tokenizer = Whitespace()
    tok = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="<unk>",
                                  pad_token="<pad>")
    tok.chat_template = ("{% for m in messages %}{{ m['role'] }} : {{ m['content'] }} "
                         "{% endfor %}{% if add_generation_prompt %}assistant :{% endif %}")
    return tok


@pytest.fixture(scope="session")
def gemma_toys():
    """{sliding_window: (model, tok, cfg)} — one build per regime, shared."""
    from transformers import Gemma4ForCausalLM

    out = {}
    for w in WINDOWS:
        cfg = _cfg(w)
        torch.manual_seed(0)
        model = Gemma4ForCausalLM(cfg).eval().to(torch.float32)
        for p in model.parameters():
            p.requires_grad_(False)
        out[w] = (model, _word_tokenizer(cfg.vocab_size), cfg)
    return out


def _engine(toy, prefix_cache: bool = False, ctx: int = 1024):
    model, tok, _ = toy
    os.environ["DRINKME_PREFIX_SLOTS"] = "" if prefix_cache else "0"
    try:
        return HFEngine(model, tok, model_id="gemma-toy", arm="test", meta={},
                        ctx=ctx, mtp_head=None)
    finally:
        del os.environ["DRINKME_PREFIX_SLOTS"]


def _run_ids(eng, text, max_tokens=64, **kw):
    """(GenResult, generated ids, Speculator.stats or None, rows). Ids are
    taken at the pick seam BOTH loops share — engines.sample_next — because
    the toy's argmax lands on `<unk>` now and then and skip_special_tokens
    drops it from the text, so re-encoding the text would lose ids. `rows`
    (position -> raw logits) is what a spec-on-vs-off comparison checks a
    fork's margin against (tests/spec_agree.py) — token-exact between
    speculation on and off is not a property this repo asserts."""
    picks: dict[int, int] = {}
    rows: dict[int, torch.Tensor] = {}
    real = engines.sample_next
    stats: dict = {}
    real_stats = mtp.Speculator.stats

    def rec(row, params, gen, prev_ids=None, gen_ids=None, **kw2):
        rows[len(gen_ids)] = row.detach().clone()
        t = real(row, params, gen, prev_ids=prev_ids, gen_ids=gen_ids, **kw2)
        picks[len(gen_ids)] = t
        return t

    def grab(self):
        s = real_stats(self)
        stats.update(s)
        return s

    engines.sample_next = rec
    mtp.Speculator.stats = grab
    try:
        res = complete(eng, GenerationRequest(
            _msgs(text) if isinstance(text, str) else text,
            SampleParams(temperature=0.0, max_tokens=max_tokens, **kw)))
    finally:
        engines.sample_next = real
        mtp.Speculator.stats = real_stats
    return res, [picks[i] for i in range(len(picks))], (stats or None), rows


class Oracle:
    """A proposer that drafts the SERIAL run's next j tokens and then a
    wrong one — so every cycle accepts exactly j and rejects at depth j.
    Installed over NgramProposer.propose for a block; the prompt length it
    needs to index the serial ids is learned from Speculator.begin."""

    def __init__(self, serial_ids: list[int], j: int):
        self.serial, self.j, self.n_prompt = serial_ids, j, None

    def __enter__(self):
        me = self
        self._propose, self._begin = ngram.NgramProposer.propose, mtp.Speculator.begin

        def begin(spec, cache, ids, hidden, start):
            me.n_prompt = len(ids)
            return me._begin(spec, cache, ids, hidden, start)

        def propose(prop, k):
            k = min(k, prop.k)
            prop.asked += 1
            if k <= 0:
                return []
            pos = len(prop.index) - me.n_prompt  # the next generated position
            good = list(me.serial[pos:pos + min(me.j, k - 1)])
            nxt = me.serial[pos + len(good)] if pos + len(good) < len(me.serial) else 3
            out = (good + [3 if nxt != 3 else 4])[:k]
            prop.matched += 1
            prop.proposed += len(out)
            return out

        ngram.NgramProposer.propose = propose
        mtp.Speculator.begin = begin
        return self

    def __exit__(self, *a):
        ngram.NgramProposer.propose = self._propose
        mtp.Speculator.begin = self._begin


@pytest.fixture(scope="session")
def serial(gemma_toys):
    """Greedy reference ids (+ rows, for a fork's margin check) per window,
    speculation OFF."""
    out = {}
    with spec_env(DRINKME_SPEC="off"):
        for w, toy in gemma_toys.items():
            res, ids, _, rows = _run_ids(_engine(toy), REPEAT)
            assert res.completion_tokens == 64 and len(ids) == 64
            out[w] = (res, ids, rows)
    return out


# ------------------------------------------------ the transformers seams --


def test_sliding_layer_reads_its_int_counter_not_the_tensor(gemma_toys):
    """The seam the bug lived on, pinned behaviourally: a sliding layer keeps
    two counters, and `get_seq_length()` — the mask's q_offset — is the INT.
    A transformers bump that merges them makes this fail here, and the
    rewind (which now moves both) gets re-derived instead of silently
    doing half a job again."""
    from transformers import StaticCache

    _, _, cfg = gemma_toys[16]
    cache = StaticCache(config=cfg, max_cache_len=64)
    assert [type(l).__name__ for l in cache.layers[:6]] == (
        ["StaticSlidingWindowLayer"] * 5 + ["StaticLayer"])
    layer = cache.layers[0]
    k = torch.full((1, cfg.num_key_value_heads, 3, cfg.head_dim), 7.0)
    with torch.inference_mode():
        layer.update(k, k)
    assert int(layer.cumulative_length) == layer.cumulative_length_int == 3
    layer.cumulative_length_int = 7  # the two really are separate surfaces
    assert int(layer.get_seq_length()) == 7 and int(layer.cumulative_length) == 3


def test_a_full_ring_shifts_the_oldest_rows_out(gemma_toys):
    """Why a rewind past the window needs a snapshot: an M-row update on a
    full sliding layer drops the M oldest rows from the storage for good."""
    from transformers import StaticCache

    _, _, cfg = gemma_toys[16]
    cache = StaticCache(config=cfg, max_cache_len=64)
    layer = cache.layers[0]
    shape = (1, cfg.num_key_value_heads, 16, cfg.head_dim)
    with torch.inference_mode():
        layer.update(torch.arange(16.0).view(1, 1, 16, 1).expand(shape).contiguous(),
                     torch.zeros(shape))
        three = (1, cfg.num_key_value_heads, 3, cfg.head_dim)
        layer.update(torch.full(three, 99.0), torch.zeros(three))
    assert layer.cumulative_length_int == 19 and int(layer.cumulative_length) == 16
    assert layer.keys[0, 0, 0, 0].item() == 3.0    # rows 0..2 are gone
    assert torch.all(layer.keys[0, 0, 13:, 0] == 99.0)


def test_the_rewind_refuses_a_shifted_ring_it_has_no_snapshot_for(gemma_toys):
    """A rewind that cannot be reconstructed must be loud, never a no-op."""
    from transformers import StaticCache

    _, _, cfg = gemma_toys[16]
    cache = StaticCache(config=cfg, max_cache_len=64)
    shape = (1, cfg.num_key_value_heads, 18, cfg.head_dim)
    with torch.inference_mode():
        cache.layers[0].update(torch.zeros(shape), torch.zeros(shape))  # crossed 16
    cap = mtp._Capture()
    cap.rows = 5
    with pytest.raises(RuntimeError, match="no sliding-window snapshot"):
        mtp._restore_rows(cache, cap, keep=2)


def test_a_verify_batch_wider_than_the_window_is_refused(gemma_toys):
    from transformers import StaticCache

    _, _, cfg = gemma_toys[16]
    cache = StaticCache(config=cfg, max_cache_len=64)
    shape = (1, cfg.num_key_value_heads, 4, cfg.head_dim)
    with torch.inference_mode():
        cache.layers[0].update(torch.zeros(shape), torch.zeros(shape))
    with pytest.raises(RuntimeError, match="exceeds sliding window"):
        mtp._snapshot_sliding(cache, mtp._Capture(), rows=17)


# ----------------------------------------- the rewind against a stepwise run --


def _prefilled(model, cfg, n: int, max_cache_len: int = 128):
    from transformers import StaticCache

    cache = StaticCache(config=cfg, max_cache_len=max_cache_len)
    prompt = torch.tensor([[(3 + 7 * i) % 96 for i in range(n)]])
    with torch.inference_mode():
        model(prompt, past_key_values=cache, use_cache=True,
              cache_position=torch.arange(n))
    return cache


def _live(layer):
    """The rows a sliding layer's storage CLAIMS: the whole ring once full,
    the first `length` rows before that."""
    n = layer.cumulative_length_int
    w = layer.max_cache_len
    if n >= w:
        return layer.keys, layer.values
    return layer.keys[:, :, :n], layer.values[:, :, :n]


def _assert_same_cache(got, ref):
    for i, (a, b) in enumerate(zip(got.layers, ref.layers)):
        assert int(a.get_seq_length()) == int(b.get_seq_length()), f"layer {i} length"
        if mtp._is_sliding_layer(a):
            assert a.cumulative_length_int == b.cumulative_length_int, f"layer {i} int"
            if a.cumulative_length_int < a.max_cache_len:
                assert int(a.cumulative_length) == int(b.cumulative_length), f"layer {i} tensor"
            ka, va = _live(a)
            kb, vb = _live(b)
        else:
            n = int(a.cumulative_length)
            ka, va, kb, vb = (a.keys[:, :, :n], a.values[:, :, :n],
                              b.keys[:, :, :n], b.values[:, :, :n])
        # batched-vs-stepped projections agree to rounding, not bitwise
        # (the "warm-vs-cold" class the engine documents): measured 5e-6 at
        # layer 11 of this float32 toy. A wrong or stale row is O(1).
        assert torch.allclose(ka, kb, rtol=1e-4, atol=2e-5), f"keys layer {i}"
        assert torch.allclose(va, vb, rtol=1e-4, atol=2e-5), f"values layer {i}"


# (window, prefill n, verify rows, keep): every regime the ring has. A rewind
# of `rows - keep` rows must leave a cache indistinguishable from one that
# only ever stepped `keep` of them.
REGIMES = [
    pytest.param(64, 8, 5, 2, id="below-the-window"),
    pytest.param(16, 12, 5, 2, id="crossing-then-back-below"),   # 17 > 16, back to 14
    pytest.param(16, 12, 5, 4, id="crossing-then-exactly-full"), # back to 16
    pytest.param(16, 14, 5, 4, id="crossing-and-still-full"),    # 19 > 16, back to 18
    pytest.param(16, 30, 5, 1, id="full-reject-everything"),
    pytest.param(16, 30, 5, 4, id="full-reject-one"),
    pytest.param(16, 16, 5, 2, id="exactly-full-before-the-batch"),
]


@pytest.mark.parametrize("window,n,rows,keep", REGIMES)
def test_rewind_matches_a_stepwise_run_on_sliding_layers(gemma_toys, window, n, rows, keep):
    """The whole rejection story on the sliding layers: write a verify batch
    through every layer, reject `rows - keep` of it, and the cache — both
    counters, the ring's rows, and the NEXT token's logits — must match a
    cache that only ever saw the accepted tokens. Then the same again from
    finish(): a second rewind on the same capture (a stop mid-cycle)."""
    model, _, cfg = gemma_toys[window]
    batch = torch.tensor([[(41 + 13 * i) % 96 for i in range(rows)]])
    got = _prefilled(model, cfg, n)
    cap = mtp._Capture()
    mtp._snapshot_sliding(got, cap, rows)
    mtp._ACTIVE = cap
    try:
        with torch.inference_mode():
            model(batch, past_key_values=got, use_cache=True,
                  cache_position=torch.arange(n, n + rows))
    finally:
        mtp._ACTIVE = None
    cap.rows = rows
    with torch.inference_mode():
        mtp._restore_rows(got, cap, keep=keep)

    ref = _prefilled(model, cfg, n)
    with torch.inference_mode():
        for j in range(keep):
            model(batch[:, j:j + 1], past_key_values=ref, use_cache=True,
                  cache_position=torch.tensor([n + j]))
    _assert_same_cache(got, ref)

    # and the thing that actually matters: the next forward agrees
    nxt = torch.tensor([[77]])
    with torch.inference_mode():
        lg = model(nxt, past_key_values=got, use_cache=True,
                   cache_position=torch.tensor([n + keep])).logits[0, -1]
        lr = model(nxt, past_key_values=ref, use_cache=True,
                   cache_position=torch.tensor([n + keep])).logits[0, -1]
    assert torch.allclose(lg, lr, rtol=1e-4, atol=2e-5)
    assert int(lg.argmax()) == int(lr.argmax())


def test_finish_rewinds_the_same_capture_a_second_time(gemma_toys):
    """cycle() keeps m+1 rows; a stop string or max_tokens inside the cycle
    then makes finish() keep fewer, off the SAME capture. Full ring."""
    model, _, cfg = gemma_toys[16]
    n, rows = 30, 5
    batch = torch.tensor([[(41 + 13 * i) % 96 for i in range(rows)]])
    got = _prefilled(model, cfg, n)
    cap = mtp._Capture()
    mtp._snapshot_sliding(got, cap, rows)
    mtp._ACTIVE = cap
    try:
        with torch.inference_mode():
            model(batch, past_key_values=got, use_cache=True,
                  cache_position=torch.arange(n, n + rows))
    finally:
        mtp._ACTIVE = None
    cap.rows = rows
    with torch.inference_mode():
        mtp._restore_rows(got, cap, keep=4)   # the cycle: 3 accepted + 1
        mtp._restore_rows(got, cap, keep=2)   # finish(): the caller kept 1
    ref = _prefilled(model, cfg, n)
    with torch.inference_mode():
        for j in range(2):
            model(batch[:, j:j + 1], past_key_values=ref, use_cache=True,
                  cache_position=torch.tensor([n + j]))
    _assert_same_cache(got, ref)


def test_after_a_rejection_every_layer_agrees_on_the_sequence_length(gemma_toys):
    """THE hole, as a one-line regression: after a rewind the sliding layers
    must report the same `get_seq_length()` as the full-attention layers.
    A sliding layer whose counter misses the rewind reports `drop` more —
    and the mask believes it."""
    model, _, cfg = gemma_toys[512]
    got = _prefilled(model, cfg, 10)
    cap = mtp._Capture()
    mtp._snapshot_sliding(got, cap, 5)
    with torch.inference_mode():
        model(torch.tensor([[5, 6, 7, 8, 9]]), past_key_values=got, use_cache=True,
              cache_position=torch.arange(10, 15))
    cap.rows = 5
    mtp._restore_rows(got, cap, keep=2)
    lengths = {int(l.get_seq_length()) for l in got.layers}
    assert lengths == {12}, lengths


# ------------------------------------------------ end to end, on the toy --


@pytest.mark.parametrize("window", WINDOWS)
def test_greedy_ngram_agrees_with_serial(gemma_toys, serial, window):
    """THE bar, in ids: the real lookup at prompt_lookup_min=1 (so it fires
    on a 96-token vocabulary) agrees with the serial stream except at a
    genuine near-tie (spec_agree; token-exact between speculation on and off
    is not a property this repo asserts) — and the run must
    contain rejections, or it proves nothing about the rewind."""
    from spec_agree import assert_agrees_or_forks_at_a_near_tie

    with spec_env(DRINKME_SPEC="ngram", DRINKME_NGRAM_MIN="1"):
        res, ids, stats, rows = _run_ids(_engine(gemma_toys[window]), REPEAT)
    ref, ref_ids, ref_rows = serial[window]
    assert stats["drafted"] > stats["accepted"], stats  # rejections happened
    assert_agrees_or_forks_at_a_near_tie(ids, ref_ids, rows, ref_rows)
    assert res.completion_tokens == ref.completion_tokens


@pytest.mark.parametrize("window", WINDOWS)
@pytest.mark.parametrize("j", [0, 1, 2, 3, 4])
def test_forced_acceptance_then_rejection_agrees(gemma_toys, serial, window, j):
    """Accept exactly j drafts and reject at depth j, EVERY cycle — the
    multi-token acceptance random weights cannot produce, at every depth
    the default cycle has, in every ring regime. A rewind that misses the
    sliding layers' counters diverges one position after the first
    rejection, at a real (non-near-tie) margin — this catches that via
    spec_agree, just without also asserting the near-tie-impossible bar."""
    from spec_agree import assert_agrees_or_forks_at_a_near_tie

    ref, ref_ids, ref_rows = serial[window]
    with spec_env(DRINKME_SPEC="ngram"), Oracle(ref_ids, j):
        res, ids, stats, rows = _run_ids(_engine(gemma_toys[window]), REPEAT)
    assert_agrees_or_forks_at_a_near_tie(ids, ref_ids, rows, ref_rows)
    # the oracle did what it says: j accepted per cycle and one rejection
    # each (the last cycle or two run against a budget of fewer tokens)
    assert stats["drafted"] > stats["accepted"] >= j * (stats["cycles"] - 2)


def test_max_tokens_mid_cycle_agrees_and_the_next_turn_extends_it(gemma_toys, serial):
    """max_tokens lands inside a 5-token cycle (51 = 10 x 5 + 1), so
    finish() rewinds the sliding ring a second time — and with the prefix
    cache ON the NEXT turn must extend that rewound cache (extends-only
    reuse) and still agree with the serial two-turn run (modulo a genuine
    near-tie). Window 64 over a 58-token prompt: the ring is full from
    generated token 6 on, so both the mid-cycle rewind and the next turn's
    prefill work a full ring."""
    from spec_agree import assert_agrees_or_forks_at_a_near_tie

    toy = gemma_toys[64]
    _, tok, _ = toy
    ref, ref_ids, ref_rows = serial[64]
    n1 = 51
    # the assistant turn re-renders from TEXT; a `<unk>` in the ids would be
    # dropped by skip_special_tokens and the turn would no longer extend the
    # cache — this window's first 51 ids carry none (measured; pinned)
    assert all(t >= 3 for t in ref_ids[:n1])
    turn2 = _msgs(REPEAT) + [{"role": "assistant",
                              "content": tok.decode(ref_ids[:n1], skip_special_tokens=True)},
                             {"role": "user", "content": "hello world"}]
    with spec_env(DRINKME_SPEC="off"):
        eng = _engine(toy, prefix_cache=True)
        _run_ids(eng, REPEAT, max_tokens=n1)
        res, off2, _, rows_off2 = _run_ids(eng, turn2, max_tokens=32)
        assert res.cached_tokens > 0
    with spec_env(DRINKME_SPEC="ngram"), Oracle(ref_ids, 4):
        eng = _engine(toy, prefix_cache=True)
        res1, on1, _, rows_on1 = _run_ids(eng, REPEAT, max_tokens=n1)
        assert res1.finish_reason == "length"
        assert_agrees_or_forks_at_a_near_tie(on1, ref_ids[:n1], rows_on1, ref_rows)
        res2, on2, _, rows_on2 = _run_ids(eng, turn2, max_tokens=32)
    assert_agrees_or_forks_at_a_near_tie(on2, off2, rows_on2, rows_off2)
    assert res2.cached_tokens == res.cached_tokens > 0  # the turn really rode the cache


# --------------------------------- hypothesis (a): the detok, on real tokenizers --

# The clean receipt's text (gemma-4-31B-it, greedy, DRINKME_SPEC=off):
# the transcript the speculative run should have produced.
# Carried here verbatim, because the streamed detok must reproduce its
# one-shot decode under EVERY partition into accepted chunks.
CLEAN = (
    "To solve 17 * 23, you can use a few different methods. Here are two ways "
    "to think it through:\n\n**Method 1: Breaking it down (Distributive Property)**\n"
    "1. Break 23 into 20 and 3.\n2. Multiply 17 by 20: $17 \\times 2 = 34$, so "
    "$17 \\times 20 = 340$.\n3. Multiply 17 by 3: $10 \\times 3 = 30$ and "
    "$7 \\times 3 = 21$. $30 + 21 = 51$.\n4. Add the two results together: "
    "$340 + 51 = 391$.\n\n**Method 2: Difference of Squares**\n1. Notice that 17 "
    "and 23 are both equidistant from 20 (17 is $20 - 3$ and 23 is $20 + 3$).\n"
    "2. Use the formula $(a - b)(a + b) = a^2 - b^2$.\n3. $20^2 - 3^2 = 400 - 9$.\n"
    "4. $400 - 9 = 391$.\n\n**Answer:**\n17 * 23 = 391"
)
# ... and the text the holes appeared in, for the record: its ids are simply
# not the clean ids (hypothesis b), and it re-encodes to the receipt's own
# completion_tokens - 1 (EOS): nothing was lost between ids and text.
HOLES = (
    "2. Multiply 17 by doing: $(17 \\times 20) + (17 \\times 3$\n"
    "3. $17 \\times 2 = 340$\n4. $17 \\times 3 = 51$\n"
    "5. Add the results together: $340 + 51 = **391**"
)


def _stream(tok, ids, partition, window: bool) -> str:
    """Replay engines.generate's emit path over `ids` fed in chunks: after
    each chunk, the full decode goes through the SuffixWindow (or not) and
    the IncrementalDetok, and the stream's flush closes it."""
    def decode(l):
        return tok.decode(l, skip_special_tokens=True)

    win = SuffixWindow(decode, verify=True) if window else None
    detok, gen, out, i = IncrementalDetok(), [], [], 0
    for k in partition:
        gen.extend(ids[i:i + k])
        i += k
        out.append(detok.push(win.full(gen) if win else decode(gen)))
    out.append(detok.flush(decode(gen)))
    return "".join(out)


def _partitions(n: int):
    """Per token (what the engine actually feeds: it pops `pending` one
    token at a time), the fixed chunk sizes the accept loop produces (m+1
    for m in 0..5), and random mixes of them."""
    yield "per-token", [1] * n
    for k in range(2, 7):
        yield f"k={k}", [k] * n
    for seed in range(12):
        r = random.Random(seed)
        parts, t = [], 0
        while t < n:
            parts.append(r.choice([1, 2, 3, 4, 5, 6]))
            t += parts[-1]
        yield f"rand{seed}", parts


@pytest.mark.parametrize("name,repo", [("gemma", GEMMA), ("qwen3-8b", QWEN8B)])
def test_streamed_detok_is_partition_invariant_on_the_receipt(name, repo):
    """Hypothesis (a), tested and refuted on the real tokenizers: the
    streamed text equals the one-shot decode for every accepted-chunk
    partition, with and without the suffix window. Skips (by name) when the
    tokenizer is not in the local HF cache; never downloads."""
    tok = _cached_tokenizer(*repo)
    ids = tok.encode(CLEAN, add_special_tokens=False)
    one = tok.decode(ids, skip_special_tokens=True)
    assert one == CLEAN
    for pname, parts in _partitions(len(ids)):
        for window in (True, False):
            got = _stream(tok, ids, parts, window)
            assert got == one, f"{name} {pname} window={window}"


def test_the_holes_are_in_the_ids_not_the_text():
    """On gemma's tokenizer the hole text re-encodes to a DIFFERENT id list
    from the clean text at the same place, and decodes back to itself: the
    detok reported the ids it was given faithfully. Digits are single
    tokens there, which is why a lost `0` reads as a dropped character."""
    tok = _cached_tokenizer(*GEMMA)
    for text in (CLEAN, HOLES):
        ids = tok.encode(text, add_special_tokens=False)
        assert tok.decode(ids, skip_special_tokens=True) == text
    two_zero = tok.encode("$17 \\times 20 = 340$", add_special_tokens=False)
    two = tok.encode("$17 \\times 2 = 340$", add_special_tokens=False)
    assert two_zero != two and len(two_zero) == len(two) + 1
    assert tok.decode(two_zero[7:8]) == "0"
