"""The multi-slot prefix cache on toy models — CPU only, no downloads.

Two toys, because the two questions have different shapes. Slot SELECTION,
eviction and the config default are id bookkeeping, so they run on a tiny
Llama through the real HFEngine. Slot COST is about what a cache actually
allocates, so the accounting runs on the same tiny hybrid qwen3_5 the MTP
tests use: 2 full-attention layers whose KV scales with the allocated window
and 2 gated-DeltaNet layers whose recurrent and conv states do not. That
split IS the memory story on the 27B (16 full-attention + 48 linear layers),
at 1/100000th the size.

Numbers here are exact rather than approximate on purpose: a memory guard
that quietly stops counting a cache surface is worse than one that fails a
test when transformers adds one.
"""

import os

import pytest
import torch

from drinkme.serving import engines, metrics, mtp
from drinkme.serving.engine import GenerationRequest, SampleParams, complete
from drinkme.serving.engines import HFEngine

WORDS = ["hello", "world", "the", "quick", "brown", "fox", "jumps", "over",
         "lazy", "dog", "alpha", "beta", "gamma", "delta", "epsilon", "zeta",
         "user", "assistant", "system", ":"]

# Four conversations that diverge at their FIRST content token, so switching
# between them is the non-extends case the single slot pays full price for.
CONVS = {
    "A": "alpha alpha beta gamma delta",
    "B": "beta beta gamma delta epsilon",
    "C": "gamma gamma delta epsilon zeta",
    "D": "delta delta epsilon zeta hello",
}


def _tokenizer():
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast

    vocab = {"<unk>": 0, "<pad>": 1}
    for w in WORDS:
        vocab[w] = len(vocab)
    while len(vocab) < 64:  # every sampleable id decodes to something
        vocab[f"tok{len(vocab)}"] = len(vocab)
    backend = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
    backend.pre_tokenizer = Whitespace()
    # No EOS on purpose (the MTP toy says the same thing): random weights hit
    # one every ~64 tokens and truncate the multi-turn runs these tests build.
    tok = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="<unk>",
                                  pad_token="<pad>")
    tok.chat_template = ("{% for m in messages %}{{ m['role'] }} : {{ m['content'] }} "
                         "{% endfor %}{% if add_generation_prompt %}assistant :{% endif %}")
    return tok


@pytest.fixture(scope="module")
def llama():
    """(model, tokenizer). float32, not bf16: these tests read cached_tokens
    and slot identity, and a bf16 toy's near-uniform logits flip greedy
    near-ties for reasons that have nothing to do with slots."""
    from transformers import LlamaConfig, LlamaForCausalLM

    cfg = LlamaConfig(vocab_size=64, hidden_size=64, intermediate_size=128,
                      num_hidden_layers=2, num_attention_heads=4,
                      num_key_value_heads=2, head_dim=16,
                      max_position_embeddings=512, tie_word_embeddings=False,
                      eos_token_id=None, pad_token_id=None)
    torch.manual_seed(0)
    model = LlamaForCausalLM(cfg).eval().to(torch.float32)
    for p in model.parameters():
        p.requires_grad_(False)
    return model, _tokenizer()


@pytest.fixture(scope="module")
def _torch_chunk_on_cpu():
    """fla's chunk kernel is triton-only, so the hybrid's PREFILL dies on a
    CPU tensor unless the pure-torch reference is swapped in — the same swap
    tests/test_serving_mtp.py makes, and for the same reason."""
    from transformers.models.qwen3_5 import modeling_qwen3_5 as mod

    fn = mod.torch_chunk_gated_delta_rule
    mod.torch_chunk_gated_delta_rule = getattr(fn, "__wrapped__", fn)
    yield
    mod.torch_chunk_gated_delta_rule = fn


@pytest.fixture(scope="module")
def hybrid(_torch_chunk_on_cpu):
    """(model, tokenizer, config) for the 4-layer hybrid: linear, full,
    linear, full — the 27B's alternation, shrunk."""
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

    cfg = Qwen3_5TextConfig(
        vocab_size=64, hidden_size=64, intermediate_size=128,
        num_hidden_layers=4, num_attention_heads=2, num_key_value_heads=1,
        head_dim=32, linear_key_head_dim=16, linear_value_head_dim=16,
        linear_num_key_heads=2, linear_num_value_heads=4,
        linear_conv_kernel_dim=4,
        layer_types=["linear_attention", "full_attention",
                     "linear_attention", "full_attention"],
        max_position_embeddings=512,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0,
                         "mrope_section": [2, 1, 1], "mrope_interleaved": True,
                         "partial_rotary_factor": 0.25},
        tie_word_embeddings=False, eos_token_id=None, pad_token_id=None)
    torch.manual_seed(0)
    model = Qwen3_5ForCausalLM(cfg).eval().to(torch.float32)
    for p in model.parameters():
        p.requires_grad_(False)
    return model, _tokenizer(), cfg


@pytest.fixture(autouse=True)
def _extends_only(monkeypatch):
    """This file pins the EXTENDS-ONLY rule: a slot is reused only when the
    prompt extends all of it, and its CONVS share just their first two
    tokens. That rule is --ctx-checkpoints 0 now; with context checkpoints
    on, a prompt that parts from a slot is served up to the shared prefix,
    which tests/test_serving_ctx_checkpoints.py pins."""
    monkeypatch.setenv("DRINKME_CTX_CHECKPOINTS", "0")


def engine(toy, slots=None, ctx=512, meta=None):
    """An HFEngine over a toy, with DRINKME_PREFIX_SLOTS bound only for the
    construction that reads it (test_serving_engines._no_reuse's shape) —
    slot count is decided once, at load, like every other engine knob.
    `meta` is what a loader would pass; the cold tier (off for this file)
    needs a resolvedRevision in it, and
    test_serving_slotstore.py's cold_engine supplies one."""
    model, tok = toy[0], toy[1]
    if slots is not None:
        os.environ["DRINKME_PREFIX_SLOTS"] = str(slots)
    try:
        return HFEngine(model, tok, model_id="toy", arm="test",
                        meta={} if meta is None else meta, ctx=ctx)
    finally:
        os.environ.pop("DRINKME_PREFIX_SLOTS", None)


def greedy(max_tokens=6):
    return SampleParams(temperature=0.0, max_tokens=max_tokens)


def turn(eng, hist, text, max_tokens=6):
    """One agent-shaped turn: append the user message, re-send the WHOLE
    history, append what came back."""
    hist.append({"role": "user", "content": text})
    res = complete(eng, GenerationRequest(list(hist), greedy(max_tokens)))
    hist.append({"role": "assistant", "content": res.text})
    return res


def slot(n, ids, alloc=4096, stamp=0, cold=False):
    s = engines._Slot(n)
    s.ids, s.alloc, s.stamp = list(ids), alloc, stamp
    s.cache = None if cold else "a cache"  # pick_slot only asks: is it there?
    return s


# --------------------------------------------------- the selection rule --
# pick_slot is pure, so these drive it with fabricated slots and no model at
# all — the nested-prefix case below cannot be reached by running requests
# (a matching turn UPDATES its slot rather than adding a second one), and it
# is exactly the case "longest match wins" exists for.


def test_the_longest_extends_only_match_wins():
    slots = [slot(0, [1, 2]), slot(1, [1, 2, 3, 4]), slot(2, [9])]
    got, lcp = engines.pick_slot(slots, [1, 2, 3, 4, 5], 5, 100)
    assert got is slots[1] and lcp == 4


def test_a_match_takes_its_own_slot_even_when_it_is_the_lru():
    """The eviction rule must never fire on a hit: the matching slot here is
    the most stale one, and two empty slots are sitting there."""
    slots = [slot(0, [], cold=True), slot(1, [], cold=True),
             slot(2, [1, 2], stamp=9)]
    got, lcp = engines.pick_slot(slots, [1, 2, 3], 3, 100)
    assert got is slots[2] and lcp == 2


def test_no_match_takes_the_least_recently_used_slot():
    slots = [slot(0, [7], stamp=5), slot(1, [8], stamp=2), slot(2, [9], stamp=8)]
    got, lcp = engines.pick_slot(slots, [1, 2, 3], 3, 100)
    assert got is slots[1] and lcp == 0  # stamp 2 is the stalest


def test_untouched_slots_are_filled_before_any_live_slot_is_taken():
    slots = [slot(0, [7], stamp=5), slot(1, [], cold=True), slot(2, [], cold=True)]
    got, lcp = engines.pick_slot(slots, [1, 2, 3], 3, 100)
    assert got is slots[1] and lcp == 0


def test_a_slot_holding_more_than_the_prompt_is_never_rewound():
    """The DeltaNet law, in slot form: a slot whose ids run PAST the prompt
    (an exact repeat, a shortened history) is not a partial-rewind
    opportunity, it is a cold take."""
    slots = [slot(0, [1, 2, 3, 4, 5], stamp=3)]
    got, lcp = engines.pick_slot(slots, [1, 2, 3], 3, 100)
    assert got is slots[0] and lcp == 0


def test_a_prompt_that_only_repeats_a_slot_is_not_an_extension():
    """lcp == n_prompt leaves no token to compute fresh logits from, so it
    does not qualify — the sampled row is always computed, never remembered."""
    slots = [slot(0, [1, 2, 3])]
    got, lcp = engines.pick_slot(slots, [1, 2, 3], 3, 100)
    assert lcp == 0


def test_a_slot_too_small_for_the_whole_request_is_not_eligible():
    """An extends-only match the allocation cannot hold is not a match: the
    request goes to a slot that will be rebuilt at the size it needs."""
    slots = [slot(0, [1, 2], alloc=8, stamp=5), slot(1, [], cold=True)]
    got, lcp = engines.pick_slot(slots, [1, 2, 3], 3, need=64)
    assert got is slots[1] and lcp == 0


# ------------------------------------------------------- the config knob --


@pytest.mark.parametrize("raw,expect", [
    (None, 1), ("", 1), ("  ", 1), ("1", 1), ("3", 3), ("16", 16),
    ("0", 0), ("-2", 1), ("banana", 1), ("2.5", 1),
])
def test_slots_from_env(monkeypatch, raw, expect):
    """Unset is one slot; 0 is the prefix cache off. Garbage warns and serves
    the default engine rather than taking the server down or silently
    changing what is served."""
    monkeypatch.delenv("DRINKME_PREFIX_SLOTS", raising=False)
    if raw is not None:
        monkeypatch.setenv("DRINKME_PREFIX_SLOTS", raw)
    assert engines.slots_from_env() == expect


# --------------------------------------- --prefix-slots CLI precedence --


def test_explicit_cli_value_wins_over_the_env_var(monkeypatch):
    monkeypatch.setenv("DRINKME_PREFIX_SLOTS", "5")
    assert engines.slots_from_env(explicit=2) == 2


def test_unset_cli_falls_through_to_the_env_var(monkeypatch):
    monkeypatch.setenv("DRINKME_PREFIX_SLOTS", "5")
    assert engines.slots_from_env(explicit=None) == 5


def test_unset_cli_and_unset_env_is_the_default(monkeypatch):
    monkeypatch.delenv("DRINKME_PREFIX_SLOTS", raising=False)
    assert engines.slots_from_env(explicit=None) == engines.DEFAULT_SLOTS


def test_garbage_explicit_cli_value_warns_and_falls_back(capsys, monkeypatch):
    monkeypatch.delenv("DRINKME_PREFIX_SLOTS", raising=False)
    assert engines.slots_from_env(explicit=-1) == engines.DEFAULT_SLOTS
    assert "--prefix-slots" in capsys.readouterr().err


def test_explicit_zero_is_the_prefix_cache_off(monkeypatch):
    monkeypatch.setenv("DRINKME_PREFIX_SLOTS", "5")
    assert engines.slots_from_env(explicit=0) == 0


def test_the_default_is_one_slot_and_interleaving_still_pays_cold(llama, monkeypatch):
    """The compatibility bar: with nothing set, the engine has exactly one
    slot and two interleaved conversations behave as they did before the
    prefix cache — every switch is a full cold prefill."""
    monkeypatch.delenv("DRINKME_PREFIX_SLOTS", raising=False)
    eng = engine(llama)
    assert len(eng._slots) == 1
    a, b = [], []
    a1 = turn(eng, a, CONVS["A"])
    b1 = turn(eng, b, CONVS["B"])
    assert a1.cached_tokens == 0 and b1.cached_tokens == 0
    a2 = turn(eng, a, "over the lazy dog")
    assert a2.cached_tokens == 0  # B holds the only slot: the default behaviour
    a3 = turn(eng, a, "jumps over")
    assert a3.cached_tokens >= a2.prompt_tokens  # extending the same one: reuse


# ----------------------------------------------- slots through the engine --


def test_three_interleaved_conversations_each_keep_their_own_slot(llama):
    """In one test: A, B and C round-robin and each SECOND turn
    reuses at least its own first prompt instead of prefilling cold."""
    eng = engine(llama, slots=3)
    hist = {k: [] for k in "ABC"}
    first = {k: turn(eng, hist[k], CONVS[k]) for k in "ABC"}
    assert all(r.cached_tokens == 0 for r in first.values())  # three cold takes
    second = {k: turn(eng, hist[k], "over the lazy dog") for k in "ABC"}
    for k in "ABC":
        assert second[k].cached_tokens >= first[k].prompt_tokens, k
    assert len({id(s.cache) for s in eng._slots}) == 3  # three distinct caches


def test_a_matching_turn_evicts_nothing(llama):
    eng = engine(llama, slots=2)
    a, b = [], []
    a1, b1 = turn(eng, a, CONVS["A"]), turn(eng, b, CONVS["B"])
    a2 = turn(eng, a, "over")  # a hit on A's slot must not disturb B's
    assert a2.cached_tokens >= a1.prompt_tokens
    b2 = turn(eng, b, "over")
    assert b2.cached_tokens >= b1.prompt_tokens


def test_a_fourth_conversation_evicts_the_least_recently_used(llama):
    eng = engine(llama, slots=3)
    hist = {k: [] for k in "ABCD"}
    for k in "ABC":
        turn(eng, hist[k], CONVS[k])
    turn(eng, hist["B"], "over")   # B and C are touched again, so A is the
    turn(eng, hist["C"], "over")   # least recently used when D arrives
    d1 = turn(eng, hist["D"], CONVS["D"])
    assert d1.cached_tokens == 0   # a new conversation is always cold
    for k in "BC":
        assert turn(eng, hist[k], "the lazy dog").cached_tokens > 0, k
    assert turn(eng, hist["A"], "the lazy dog").cached_tokens == 0  # A was evicted


def test_prefix_slot_metrics_record_hit_miss_and_evict(llama):
    """engines.py's THREE outcomes at pick_slot's call site: `miss` =
    an empty slot filled cold, `hit` = the code's own word for a match
    (EXTENDS), `evict` = an occupied slot reset cold — each recorded at
    exactly the print() lines above, so this watches serving/metrics.py
    over the SAME turns test_a_fourth_conversation_evicts... makes."""
    metrics.reset()
    eng = engine(llama, slots=2)
    a, b, c = [], [], []
    turn(eng, a, CONVS["A"])  # slot 0 empty -> miss
    turn(eng, b, CONVS["B"])  # slot 1 empty -> miss
    turn(eng, a, "over")      # extends A's slot -> hit
    turn(eng, c, CONVS["C"])  # both slots occupied; C is new -> LRU (B) evicted
    events = metrics.SLOT_EVENTS.snapshot()
    counts = {ev: events.get((("event", ev),), 0) for ev in ("hit", "miss", "evict")}
    assert counts == {"hit": 1, "miss": 2, "evict": 1}
    assert metrics.SLOT_OCCUPANCY.snapshot()[()] == 2  # both slots filled at least once


def test_a_failed_generation_invalidates_only_its_own_slot(llama):
    """One lock, one slot in hand: an exception mid-generate leaves the slot
    it was writing INVALID (never a lying one) and every other slot exactly
    as it was — nothing is half-swapped, because nothing is swapped at all."""
    eng = engine(llama, slots=2)
    a, b = [], []
    turn(eng, a, CONVS["A"])
    b1 = turn(eng, b, CONVS["B"])

    def boom(_delta):
        raise RuntimeError("client exploded")

    with pytest.raises(RuntimeError):
        complete(eng, GenerationRequest(a + [{"role": "user", "content": "over"}], greedy(4)), boom)
    assert eng._slots[0].ids == []  # A's slot: invalid, not lying
    assert turn(eng, b, "over").cached_tokens >= b1.prompt_tokens  # B untouched


def test_zero_slots_is_the_prefix_cache_off(llama, monkeypatch):
    """--prefix-slots 0 / DRINKME_PREFIX_SLOTS=0: every request builds its own
    cache, nothing is kept, and /health says slots 0."""
    eng = engine(llama, slots=0)
    a = []
    assert turn(eng, a, CONVS["A"]).cached_tokens == 0
    assert turn(eng, a, "over").cached_tokens == 0
    assert all(s.cache is None for s in eng._slots)
    assert eng.prefix_cache_state() == {"slots": 0}


def test_reset_prefix_cache_drops_every_slot(llama):
    metrics.reset()
    eng = engine(llama, slots=2)
    a, b = [], []
    turn(eng, a, CONVS["A"])
    turn(eng, b, CONVS["B"])
    assert metrics.SLOT_OCCUPANCY.snapshot()[()] == 2
    eng.reset_prefix_cache()
    assert all(s.cache is None and s.ids == [] and s.alloc == 0
               for s in eng._slots)
    assert metrics.SLOT_OCCUPANCY.snapshot()[()] == 0  # occupancy follows the reset
    assert turn(eng, a, "over").cached_tokens == 0  # cold in its own terms


# --------------------------------------------------- what a slot costs --


def test_cache_bytes_reads_the_real_tensors_of_a_static_cache(llama):
    """Known shapes, exactly: 2 layers x (keys + values) of
    [1, kv_heads, width, head_dim] float32, plus one int64 cumulative_length
    per layer. The KV itself is not allocated until a forward runs
    (transformers-5 layers are lazy), which is precisely why the engine
    measures a slot AFTER driving a token through it and not at construction:
    a fresh cache honestly reports only its counters."""
    from transformers import StaticCache

    model, tok = llama
    cfg = model.config
    cache = StaticCache(config=cfg, max_cache_len=48)
    assert engines.cache_bytes(cache) == cfg.num_hidden_layers * 8  # counters only
    with torch.inference_mode():
        model(torch.tensor([[3]]), past_key_values=cache, use_cache=True,
              cache_position=torch.arange(1), logits_to_keep=1)
    kv = cfg.num_hidden_layers * 2 * cfg.num_key_value_heads * 48 * cfg.head_dim * 4
    counters = cfg.num_hidden_layers * 8
    assert engines.cache_bytes(cache) == kv + counters


def test_slot_cost_separates_the_deltanet_state_from_the_kv(hybrid):
    """The hybrid's accounting against known shapes. The DeltaNet recurrent
    state ([1, v_heads, k_dim, v_dim]) and conv state ([1, 2*key_dim +
    value_dim, kernel]) have NO sequence dimension, so they are FIXED per
    slot; only the 2 full-attention layers' KV scales with the window. That
    is why a hybrid slot is cheap to keep and the 8B's is not."""
    model, tok, cfg = hybrid
    eng = engine((model, tok), slots=1, ctx=512)
    fixed, per_token = eng._slot_fixed_bytes, eng._slot_token_bytes
    key_dim = cfg.linear_num_key_heads * cfg.linear_key_head_dim
    value_dim = cfg.linear_num_value_heads * cfg.linear_value_head_dim
    n_linear = cfg.layer_types.count("linear_attention")
    n_full = cfg.layer_types.count("full_attention")
    recurrent = (cfg.linear_num_value_heads * cfg.linear_key_head_dim
                 * cfg.linear_value_head_dim) * 4
    conv = (2 * key_dim + value_dim) * cfg.linear_conv_kernel_dim * 4
    assert per_token == n_full * 2 * cfg.num_key_value_heads * cfg.head_dim * 4
    assert fixed == n_linear * (recurrent + conv) + n_full * 8
    # and the number the guard actually charges: the slot at its full width
    assert fixed + per_token * eng.ctx == 12288 + 16 + 512 * 512


def test_the_measured_cost_predicts_a_real_allocation(hybrid):
    """The linear model is not a model: rebuild a cache at an arbitrary width
    and the prediction must hit the real byte count on the nose."""
    model, tok, cfg = hybrid
    eng = engine((model, tok), slots=1, ctx=512)
    cache = eng._cache(200)
    with torch.inference_mode():
        model(torch.tensor([[3]]), past_key_values=cache, use_cache=True,
              cache_position=torch.arange(1), logits_to_keep=1)
    assert (eng._slot_fixed_bytes + eng._slot_token_bytes * 200
            == engines.cache_bytes(cache))


def test_startup_announces_the_bytes_per_slot(hybrid, capsys):
    model, tok, cfg = hybrid
    eng = engine((model, tok), slots=3, ctx=512)
    line = [l for l in capsys.readouterr().out.splitlines()
            if "prefix cache:" in l]
    assert len(line) == 1
    assert "3 slots" in line[0]
    assert f"{12304 + 512 * 512:,} B" in line[0]  # per slot, at full width
    assert "12,304 B state" in line[0] and "512 B/token" in line[0]


# ----------------------------------------------------- the fit guard --
# monkeypatched headroom, never a real device: mtp.residency_check is pure
# arithmetic and this is the caller that has to get the multiplication right.


def test_the_guard_refuses_an_explicit_slot_count_that_does_not_fit(hybrid, monkeypatch):
    monkeypatch.setattr(mtp, "device_headroom_gib", lambda device: (0.5, 40.0))
    with pytest.raises(SystemExit) as e:
        engine(hybrid, slots=3, ctx=512)
    msg = str(e.value)
    assert "refusing to serve" in msg and "3 prefix slots" in msg
    assert "DRINKME_PREFIX_SLOTS" in msg  # what to change is in the refusal


def test_the_guard_only_warns_about_the_default_single_slot(hybrid, monkeypatch, capsys):
    """A guard that refuses an allocation that serves would be a regression,
    not a safeguard."""
    monkeypatch.setattr(mtp, "device_headroom_gib", lambda device: (0.5, 40.0))
    eng = engine(hybrid, slots=1, ctx=512)
    assert len(eng._slots) == 1
    err = capsys.readouterr().err
    assert "warning" in err and "1 prefix slot at ctx 512" in err


def test_the_guard_passes_a_comfortable_fit(hybrid, monkeypatch, capsys):
    monkeypatch.setattr(mtp, "device_headroom_gib", lambda device: (100.0, 40.0))
    eng = engine(hybrid, slots=3, ctx=512)
    assert len(eng._slots) == 3
    assert "refusing" not in capsys.readouterr().err


def test_nothing_to_measure_is_nothing_to_guard(hybrid, monkeypatch):
    """cpu and mps have no mem_get_info; that is not a failure and must not
    become one (mtp.py's guard 1 takes the same posture)."""
    monkeypatch.setattr(mtp, "device_headroom_gib", lambda device: None)
    assert len(engine(hybrid, slots=4, ctx=512)._slots) == 4


# ------------------------------------------- the GPU instrument, dry-run --


def _bench_module():
    """bench/ is not a package and nothing else imports from it; load the
    file directly rather than adding a sys.path convention for one test."""
    import importlib.util
    import pathlib

    path = (pathlib.Path(__file__).resolve().parents[1]
            / "bench" / "prefix_slots_verify.py")
    spec = importlib.util.spec_from_file_location("prefix_slots_verify", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# The bench's own conversations are English prose; this tokenizer knows 20
# words and maps everything else to <unk>, which would make all four threads
# the same run of <unk> and quietly destroy the divergence the schedule
# depends on. Same shapes, toy vocabulary.
TOY_CONVS = {
    "A": ("alpha alpha beta gamma delta epsilon",
          ["hello world", "the quick", "brown fox", "jumps over"]),
    "B": ("beta beta gamma delta epsilon zeta",
          ["lazy dog", "hello quick", "world brown", "fox jumps"]),
    "C": ("gamma gamma delta epsilon zeta hello",
          ["over lazy", "dog hello", "quick brown", "world fox"]),
    "D": ("delta delta epsilon zeta hello world",
          ["fox jumps", "over the", "lazy dog", "hello alpha"]),
}


def test_the_gpu_instrument_runs_end_to_end_on_a_toy(hybrid):
    """The dry run. bench/prefix_slots_verify.py cannot touch the GPU
    in this CPU-only suite, so the exact functions that will run on the 27B are driven
    here against the 4-layer hybrid: the model tap, the three schedules, the
    cold replay, the comparison and the verdict.

    What this proves is that the instrument works and reports what it claims
    — including the eviction arm's LRU prediction, which is a property of the
    engine and not of the model size. What it cannot prove is the real
    model's numerics: on a toy with random weights, warm-vs-cold near-ties
    flip greedy argmax for reasons that have nothing to do with slots (the
    prefix-cache tests next door say the same), so the argmax and text claims
    belong to the GPU run. The delta bound here is loose on purpose."""
    verify = _bench_module()
    model, tok, cfg = hybrid
    tap = verify.ModelTap(model)
    eng = engine((tap, tok), slots=3, ctx=512)

    warm, hists = verify.run_schedule(eng, tap, verify.INTERLEAVED, TOY_CONVS,
                                      max_tokens=4, label="toy")
    rows = verify.compare(warm, verify.run_cold(eng, tap, warm, max_tokens=4))
    v = verify.verdict(rows)
    assert all(r["logits_seen"] for r in rows)  # the tap saw every prefill row
    assert v["all_cache_as_expected"]  # three threads, three slots, no resets
    assert v["max_abs_delta"] < 1e-2

    ev, _ = verify.run_schedule(eng, tap, verify.EVICT, TOY_CONVS,
                                max_tokens=4, hists=hists, label="toy-evict")
    ev_rows = verify.compare(ev, verify.run_cold(eng, tap, ev, max_tokens=4))
    # the schedule's expectations ARE the eviction claim: D cold (new), B and
    # C untouched by that eviction, A cold because A's slot is the one D took
    assert verify.verdict(ev_rows)["all_cache_as_expected"]
    assert [r["cached_tokens"] > 0 for r in ev_rows] == [False, True, True, False]


def test_zero_slots_are_charged_for_one_cache(llama, monkeypatch, capsys):
    """--prefix-slots 0 holds no slots at all — every request builds its own
    cache and frees it — so the guard charges one cache, the one each
    request allocates."""
    monkeypatch.setattr(mtp, "device_headroom_gib", lambda device: (0.5, 40.0))
    eng = engine(llama, slots=0)
    line = [l for l in capsys.readouterr().out.splitlines() if "prefix cache:" in l]
    assert "1 slot" in line[0] and "prefix cache OFF" in line[0]
