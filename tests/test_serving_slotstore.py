"""The SSD-persisted prefix slots (on-disk prefix slots) on toy models — CPU only, no downloads.

Three shapes of question, three sections.

  The keying is pure arithmetic over id lists, so it runs with no store, no
  model and no files: what a chain hash promises (agreement on h_d means
  agreement on every token up to the boundary) and what the walk down it must
  pick (the deepest shared boundary, then an exact tail).

  The store is files, so it runs against a real directory with a stand-in
  cache object — a plain layer holding tensors and flags. That stand-in is
  not a shortcut: cache_state/apply_state walk vars() precisely so they do
  not know what a transformers layer is, and a test that handed them a
  transformers layer would be testing transformers.

  The gate is warm == cold across a restart, and it runs on the tiny hybrid
  (2 full-attention layers whose KV scales with the window, 2 gated-DeltaNet
  layers whose recurrent and conv states do not — the 27B's story at
  1/100000th the size), through the real HFEngine, with a real StaticCache
  saved and restored. The claim is IDENTITY, not similarity: a slot restored
  from disk must produce the same tokens and the same text as the live slot
  it was copied from.

The toys are test_serving_prefix_slots.py's — the same models, the same
tokenizer, the same greedy params — because on-disk prefix slots is the prefix cache plus a disk, and a
second set of toys would let the two drift.
"""

import json
import os

import pytest
import torch

from drinkme.serving import slotstore
from drinkme.serving.engine import GenerationRequest, complete
from drinkme.serving.slotstore import SlotStore, _Entry, block_chain
from test_serving_engines import toy as toy_pack  # noqa: F401 — the real toy pack
from test_serving_gemma_spec import _prefilled, gemma_toys  # noqa: F401
from test_serving_prefix_slots import (_torch_chunk_on_cpu,  # noqa: F401
                                       engine, greedy, hybrid, llama)

BLOCK = slotstore.BLOCK_TOKENS
HEADER = {"model_id": "toy", "arm": "test", "pack_id": "packdeadbeef",
          "dtype": "float32", "transformers": "5.15.1"}


def ids(n, start=0):
    return list(range(start, start + n))


# ------------------------------------------------------------- the keying --


def test_only_full_blocks_are_hashed():
    """The tail past the last boundary is not a link in the chain — it is not
    a boundary anything can be shared at, and it rides in the header as ids."""
    assert block_chain(ids(BLOCK - 1)) == []
    assert len(block_chain(ids(BLOCK))) == 1
    assert len(block_chain(ids(2 * BLOCK - 1))) == 1
    assert len(block_chain(ids(3 * BLOCK))) == 3


def test_a_shared_prefix_shares_the_chain_prefix():
    a = block_chain(ids(3 * BLOCK))
    b = block_chain(ids(3 * BLOCK) + ids(BLOCK, start=9999))
    assert b[:3] == a  # the same three blocks hash to the same three links


def test_a_changed_token_changes_every_later_link():
    """The parent fold is the whole point: agreeing on h_d means agreeing on
    every token before it, so one differing token early cannot be hidden by
    identical blocks after it."""
    base = ids(3 * BLOCK)
    poked = list(base)
    poked[1] = 999999
    a, b = block_chain(base), block_chain(poked)
    assert a[0] != b[0] and a[1] != b[1] and a[2] != b[2]


def test_the_chain_is_position_sensitive():
    """Same blocks, swapped order, different chain — a hash that ignored the
    parent would call these the same prefix."""
    one, two = ids(BLOCK), ids(BLOCK, start=5000)
    assert block_chain(one + two) != block_chain(two + one)


def test_the_prefix_key_is_deterministic_and_length_tagged():
    assert slotstore.prefix_key(ids(300)) == slotstore.prefix_key(ids(300))
    assert slotstore.prefix_key(ids(300)) != slotstore.prefix_key(ids(301))
    assert slotstore.prefix_key(ids(300)).startswith("300-")


def test_the_pack_id_is_the_manifest_digest():
    """Keyed on the manifest digest recomputed from the live maps (tensors,
    sha256, tensorInfo, source), so a repack — or a swapped tensor map —
    changes it and nothing else does; a meta.json without a manifest has no
    id at all (it is not a pack this drinkme wrote)."""
    from drinkme.codec import identity

    meta = {"hfRepo": "Q/M", "revision": "abc", "formatVersion": 1,
            "tensors": {"a": "t0000.npz", "b": "t0001.npz"},
            "sha256": {"t0000.npz": "11", "t0001.npz": "22"},
            "tensorInfo": {"a": {"shape": [8, 64]}, "b": {"shape": [8, 64]}}, "source": None}
    meta["manifestSha256"] = identity.manifest_digest(meta)
    same = {k: meta[k] for k in reversed(list(meta))}
    same["hfRepo"] = "OTHER"  # not part of the manifest: the id is the bytes and the binding
    moved = json.loads(json.dumps(meta))
    moved["sha256"]["t0001.npz"] = "33"
    swapped = json.loads(json.dumps(meta))
    swapped["tensors"] = {"a": "t0001.npz", "b": "t0000.npz"}
    assert slotstore.pack_id_from_meta(meta) == slotstore.pack_id_from_meta(same)
    assert slotstore.pack_id_from_meta(meta) != slotstore.pack_id_from_meta(moved)
    assert slotstore.pack_id_from_meta(meta) != slotstore.pack_id_from_meta(swapped)
    assert slotstore.pack_id_from_meta(meta) == "manifest" + meta["manifestSha256"][:16]
    with pytest.raises(ValueError, match="no manifestSha256"):
        slotstore.pack_id_from_meta({"hfRepo": "Q/M", "sha256": {"t0000.npz": "11"}})


# ------------------------------------------------- the walk down the chain --
# lookup is the disk index's whole decision, so these drive it with fabricated
# entries and no files at all — test_serving_prefix_slots.py fabricates _Slots
# for pick_slot for the same reason.


def store(tmp_path, **kw):
    return SlotStore(str(tmp_path), "ident", HEADER, **kw)


def entry(st, n_ids, alloc=4096, used=0.0, src=None):
    """An index entry describing a stored prefix of `src[:n_ids]`."""
    src = ids(4 * BLOCK) if src is None else src
    chain = block_chain(src[:n_ids])
    e = _Entry(slotstore.prefix_key(src[:n_ids]), n_ids, alloc, chain,
               list(src[len(chain) * BLOCK:n_ids]), 1024, used, path=None)
    st._entries[e.key] = e
    return e


def test_the_deepest_shared_boundary_wins(tmp_path):
    st = store(tmp_path)
    shallow = entry(st, BLOCK + 5)
    deep = entry(st, 3 * BLOCK + 7)
    assert st.lookup(ids(4 * BLOCK), need=100, ctx=8192) is deep
    assert shallow.n_ids < deep.n_ids


def test_a_tail_that_diverges_is_not_a_match(tmp_path):
    """The chain says the full blocks agree; the tail is the rest of the
    claim, and it is checked exactly."""
    st = store(tmp_path)
    entry(st, BLOCK + 5)
    prompt = ids(BLOCK) + [7, 7, 7, 7, 7] + ids(BLOCK, start=5000)
    assert st.lookup(prompt, need=100, ctx=8192) is None


def test_an_entry_below_the_first_boundary_still_matches(tmp_path):
    """A stored prefix shorter than one block has no chain and sits in the
    depth-0 bucket. It is checked LAST, which is longest-match-first."""
    st = store(tmp_path)
    short = entry(st, 10)
    assert st.lookup(ids(4 * BLOCK), need=100, ctx=8192) is short


def test_a_stored_prefix_is_never_truncated_to_a_boundary(tmp_path):
    """The DeltaNet law on disk: a stored slot that runs PAST the prompt is
    not a rewind opportunity even though it shares whole blocks with it."""
    st = store(tmp_path)
    entry(st, 3 * BLOCK)
    assert st.lookup(ids(2 * BLOCK), need=100, ctx=8192) is None


def test_an_exact_repeat_leaves_no_token_to_compute(tmp_path):
    st = store(tmp_path)
    entry(st, 2 * BLOCK)
    assert st.lookup(ids(2 * BLOCK), need=100, ctx=8192) is None


def test_an_entry_too_narrow_for_the_request_is_not_a_match(tmp_path):
    st = store(tmp_path)
    entry(st, BLOCK + 5, alloc=512)
    assert st.lookup(ids(4 * BLOCK), need=4096, ctx=8192) is None


def test_an_entry_wider_than_this_servers_ctx_is_not_a_match(tmp_path):
    """A slot saved at ctx 32768 is 32768-wide tensors; a server that came
    back up with a smaller window has nowhere to put them."""
    st = store(tmp_path)
    entry(st, BLOCK + 5, alloc=8192)
    assert st.lookup(ids(4 * BLOCK), need=100, ctx=4096) is None


def test_a_disabled_store_answers_nothing(tmp_path):
    st = store(tmp_path)
    entry(st, BLOCK + 5)
    st.disable()
    assert st.lookup(ids(4 * BLOCK), need=100, ctx=8192) is None


# ------------------------------------------------------------- the files --
# A stand-in cache: cache_state/apply_state walk vars() so that a transformers
# bump that adds a surface gets carried rather than dropped, which means they
# must work on anything shaped like a cache. The real StaticCache is exercised
# by the gate below.


class FakeLayer:
    def __init__(self, width, seed):
        g = torch.Generator().manual_seed(seed)
        self.keys = torch.randn(1, 2, width, 4, generator=g)
        self.values = torch.randn(1, 2, width, 4, generator=g)
        self.cumulative_length = torch.tensor(0, dtype=torch.int64)
        self.is_initialized = True
        self.recurrent_states = {0: torch.randn(1, 2, 4, 4, generator=g)}
        self.has_previous_state = {0: True}
        self.max_cache_len = width
        self.device = torch.device("cpu")


class FakeCache:
    def __init__(self, width, seed=0, layers=2):
        self.layers = [FakeLayer(width, seed + i) for i in range(layers)]


def make_cache(width, seed=0):
    return lambda alloc: FakeCache(alloc, seed=seed)


def noprime(_cache):
    """The real prime drives a token through the model so transformers' lazy
    layers materialize; the stand-in allocates in its constructor."""


def filled(width, n_ids, seed=0):
    cache = FakeCache(width, seed=seed)
    for layer in cache.layers:
        layer.cumulative_length = torch.tensor(n_ids, dtype=torch.int64)
    return cache


def test_a_slot_round_trips_through_a_file(tmp_path):
    st = store(tmp_path)
    tok_ids = ids(BLOCK + 40)
    cache = filled(512, len(tok_ids))
    assert st.put(tok_ids, 512, cache) is not None
    assert st.flush(10)
    fresh = store(tmp_path)
    assert fresh.index() == (1, os.path.getsize(
        os.path.join(fresh.dir, slotstore.prefix_key(tok_ids), "cache.safetensors")))
    got = fresh.lookup(tok_ids + [4242], need=100, ctx=8192)
    assert got is not None
    live, back = fresh.restore(got, make_cache(512), noprime)
    assert back == tok_ids
    for a, b in zip(cache.layers, live.layers):
        assert torch.equal(a.keys, b.keys) and torch.equal(a.values, b.values)
        assert torch.equal(a.recurrent_states[0], b.recurrent_states[0])
        assert int(b.cumulative_length) == len(tok_ids)


def test_a_prefix_below_one_block_is_not_stored(tmp_path):
    """It has no chain to be found by and would pay a copy of the whole
    allocation for a prefill nobody notices."""
    st = store(tmp_path)
    assert st.put(ids(BLOCK - 1), 512, filled(512, BLOCK - 1)) is None
    assert st.flush(10)
    assert store(tmp_path).index() == (0, 0)


def test_the_disk_cap_evicts_least_recently_used_first(tmp_path):
    st = store(tmp_path)
    sizes = []
    for k in range(3):
        tok_ids = ids(BLOCK + 40, start=k * 10000)
        st.put(tok_ids, 512, filled(512, len(tok_ids)))
        assert st.flush(10)
        sizes.append(st._entries[slotstore.prefix_key(tok_ids)].nbytes)
    st.cap = int(2.5 * sizes[0])  # room for two of the three
    st.put(ids(BLOCK + 40, start=99000), 512, filled(512, BLOCK + 40))
    assert st.flush(10)
    kept = store(tmp_path)
    assert kept.index()[0] == 2
    assert slotstore.prefix_key(ids(BLOCK + 40, start=0)) not in kept._entries
    assert slotstore.prefix_key(ids(BLOCK + 40, start=99000)) in kept._entries


def test_a_slot_bigger_than_the_whole_cap_is_not_kept(tmp_path):
    st = store(tmp_path, cap_bytes=1024)
    st.put(ids(BLOCK + 40), 512, filled(512, BLOCK + 40))
    assert st.flush(10)
    assert store(tmp_path).index() == (0, 0)


def test_a_pack_id_mismatch_is_skipped_and_logged(tmp_path, capsys):
    st = store(tmp_path)
    tok_ids = ids(BLOCK + 40)
    st.put(tok_ids, 512, filled(512, len(tok_ids)))
    assert st.flush(10)
    other = SlotStore(str(tmp_path), "ident", dict(HEADER, pack_id="packbeefdead"))
    assert other.index() == (0, 0)
    assert "pack_id" in capsys.readouterr().err
    assert other.lookup(tok_ids + [1], need=10, ctx=8192) is None


def test_a_transformers_version_mismatch_is_skipped(tmp_path):
    """The tensors ARE transformers' internal cache layout. A file written
    against another version is skipped, never guessed at."""
    st = store(tmp_path)
    st.put(ids(BLOCK + 40), 512, filled(512, BLOCK + 40))
    assert st.flush(10)
    other = SlotStore(str(tmp_path), "ident", dict(HEADER, transformers="9.9.9"))
    assert other.index() == (0, 0)


def test_a_format_version_bump_is_skipped(tmp_path, monkeypatch):
    st = store(tmp_path)
    st.put(ids(BLOCK + 40), 512, filled(512, BLOCK + 40))
    assert st.flush(10)
    monkeypatch.setattr(slotstore, "FORMAT_VERSION", slotstore.FORMAT_VERSION + 1)
    assert store(tmp_path).index() == (0, 0)


def test_a_corrupt_header_is_skipped_and_logged(tmp_path, capsys):
    st = store(tmp_path)
    tok_ids = ids(BLOCK + 40)
    st.put(tok_ids, 512, filled(512, len(tok_ids)))
    assert st.flush(10)
    hpath = os.path.join(st.dir, slotstore.prefix_key(tok_ids), "header.json")
    with open(hpath, "w") as f:
        f.write("{not json at all")
    fresh = store(tmp_path)
    assert fresh.index() == (0, 0)
    err = capsys.readouterr().err
    assert "unreadable header" in err and "1 skipped" not in err


def test_a_truncated_tensor_file_is_skipped(tmp_path):
    """The header records the byte count it wrote; a file that is not that
    size is a partial write, and a partial cache is not a cache."""
    st = store(tmp_path)
    tok_ids = ids(BLOCK + 40)
    st.put(tok_ids, 512, filled(512, len(tok_ids)))
    assert st.flush(10)
    tpath = os.path.join(st.dir, slotstore.prefix_key(tok_ids), "cache.safetensors")
    with open(tpath, "r+b") as f:
        f.truncate(os.path.getsize(tpath) - 64)
    assert store(tmp_path).index() == (0, 0)


def test_tensors_without_a_header_are_not_indexed(tmp_path):
    """The write order is tensors, then header. A crash between them leaves a
    directory the index passes over in silence — the only failure mode a
    derived artifact is allowed."""
    st = store(tmp_path)
    tok_ids = ids(BLOCK + 40)
    st.put(tok_ids, 512, filled(512, len(tok_ids)))
    assert st.flush(10)
    os.remove(os.path.join(st.dir, slotstore.prefix_key(tok_ids), "header.json"))
    assert store(tmp_path).index() == (0, 0)


def test_a_state_from_a_different_shape_is_refused_not_applied(tmp_path, capsys):
    """The restore is checked against a LIVE cache of this process, so a file
    whose tensors do not fit costs a prefill and never a wrong answer."""
    st = store(tmp_path)
    tok_ids = ids(BLOCK + 40)
    st.put(tok_ids, 512, filled(512, len(tok_ids)))
    assert st.flush(10)
    fresh = store(tmp_path)
    fresh.index()
    got = fresh.lookup(tok_ids + [1], need=10, ctx=8192)
    assert fresh.restore(got, lambda alloc: FakeCache(alloc, layers=3), noprime) is None
    assert "cache layout differs" in capsys.readouterr().err
    assert not fresh._entries  # and the unusable entry leaves the index


def test_a_state_whose_counter_disagrees_with_the_header_is_refused(tmp_path, capsys):
    """cumulative_length is what transformers-5 actually writes at. If the
    file's counter is not the token count the index promised, the two
    descriptions of the same cache disagree and neither is trusted."""
    st = store(tmp_path)
    tok_ids = ids(BLOCK + 40)
    st.put(tok_ids, 512, filled(512, len(tok_ids) - 3))  # a lying cache
    assert st.flush(10)
    fresh = store(tmp_path)
    fresh.index()
    got = fresh.lookup(tok_ids + [1], need=10, ctx=8192)
    assert fresh.restore(got, make_cache(512), noprime) is None
    assert "cache counter says" in capsys.readouterr().err


def test_the_warm_tier_answers_before_the_file_does(tmp_path):
    """DRINKME_SLOT_RAM_GIB keeps the eviction's host copy, so the
    evicted-but-recent case never deserializes. The copy is the same bytes
    either way — this asserts it is the same bytes."""
    st = store(tmp_path, ram_bytes=1 << 30)
    tok_ids = ids(BLOCK + 40)
    cache = filled(512, len(tok_ids))
    e = st.put(tok_ids, 512, cache)
    assert st.flush(10)
    assert e.ram is not None  # kept, not dropped after the write
    live, back = st.restore(st.lookup(tok_ids + [9], need=10, ctx=8192),
                            make_cache(512), noprime)
    assert back == tok_ids
    assert torch.equal(live.layers[0].keys, cache.layers[0].keys)


def test_the_warm_tier_is_off_by_default(tmp_path):
    st = store(tmp_path)
    e = st.put(ids(BLOCK + 40), 512, filled(512, BLOCK + 40))
    assert st.flush(10)
    assert e.ram is None  # written, then let go: unified memory is the model's


def test_slot_root_reads_the_env(monkeypatch):
    monkeypatch.setenv("DRINKME_SLOT_DIR", "off")
    assert slotstore.slot_root() is None
    monkeypatch.setenv("DRINKME_SLOT_DIR", "")
    assert slotstore.slot_root() is None
    monkeypatch.setenv("DRINKME_SLOT_DIR", "/tmp/somewhere")
    assert slotstore.slot_root() == "/tmp/somewhere"
    monkeypatch.delenv("DRINKME_SLOT_DIR")
    assert slotstore.slot_root().endswith("/.cache/drinkme/slots")


def test_a_garbage_cap_warns_and_falls_back(monkeypatch, capsys):
    monkeypatch.setenv("DRINKME_SLOT_DISK_GIB", "banana")
    assert slotstore.disk_cap_bytes() == int(slotstore.DEFAULT_DISK_GIB * 1024 ** 3)
    assert "DRINKME_SLOT_DISK_GIB" in capsys.readouterr().err


# ---------------------------------------------------------------- the gate --
# warm == cold across a restart, on the real engine and a real StaticCache.


LONG = " ".join(["the quick brown fox jumps over lazy dog alpha beta"] * 40)
OTHER = " ".join(["zeta epsilon delta gamma hello world"] * 60)


# what a loader passes: the resolved, immutable revision of the weights
# — the store refuses to open without one
TOY_META = {"resolvedRevision": "toyrev0000000000000000000000000000000000"}


def cold_engine(toy, tmp_path, monkeypatch, slots=1, ctx=1024):
    monkeypatch.setenv("DRINKME_SLOT_DIR", str(tmp_path))
    return engine(toy, slots=slots, ctx=ctx, meta=dict(TOY_META))


def agent_turn(eng, hist, text, max_tokens=6):
    """One agent-shaped turn whose next render is guaranteed to EXTEND the
    slot this one wrote.

    The assistant content is the decode of exactly the ids the slot holds,
    special tokens INCLUDED, rather than res.text. These toys have <unk> and
    <pad> in their vocabulary, a randomly-initialized model emits them, and
    skip_special_tokens drops them from the visible text — so a history
    re-rendered from res.text is a token short of the cache and every reuse
    assertion below would be measuring the toy's vocabulary instead of the
    cold tier. Same shape as the thinking-history finding in
    bench/prefix_slots_verify.py's docstring: reuse needs the SAME token
    prefix the picker will match against, and getting that wrong is an
    instrument that never exercises the claim."""
    hist.append({"role": "user", "content": text})
    res = complete(eng, GenerationRequest(list(hist), greedy(max_tokens)))
    slot = max(eng._slots, key=lambda s: s.stamp)
    hist.append({"role": "assistant",
                 "content": eng.tok.decode(slot.ids[res.prompt_tokens:],
                                           skip_special_tokens=False)})
    return res


def test_a_restored_slot_answers_identically_to_the_live_one(hybrid, tmp_path,
                                                             monkeypatch):
    """The gate (on-disk prefix slots), on the hybrid whose DeltaNet state is the part that
    cannot be rewound: a slot written to disk, indexed by a SECOND engine that
    never saw the first one's memory, and restored, must produce the same
    tokens and the same text as the live slot it was copied from.

    Two engines over one loaded model is the closest a CPU test gets to a
    process restart: engine B's slots start empty, its store is indexed from
    the directory alone, and the only thing it knows about A's conversation is
    what A wrote to a file."""
    a = cold_engine(hybrid, tmp_path, monkeypatch)
    hist_a = []
    agent_turn(a, hist_a, LONG)
    assert a.persist_slots(30) == 1

    b = cold_engine(hybrid, tmp_path, monkeypatch)
    assert b._cold is not None and len(b._cold._entries) == 1
    assert all(s.cache is None for s in b._slots)  # nothing preloaded at boot

    hist_b = list(hist_a)
    live = agent_turn(a, hist_a, "over the lazy dog", max_tokens=8)
    restored = agent_turn(b, hist_b, "over the lazy dog", max_tokens=8)
    assert restored.cached_tokens == live.cached_tokens > 0
    assert restored.text == live.text
    assert hist_b[-1]["content"] == hist_a[-1]["content"]


def test_a_restored_slot_matches_the_live_one_on_the_dense_toy(llama, tmp_path,
                                                               monkeypatch):
    """The same gate on the all-attention toy: a slot is keys and values and
    a counter there, and the restore has to land all three."""
    a = cold_engine(llama, tmp_path, monkeypatch)
    hist_a = []
    agent_turn(a, hist_a, LONG)
    assert a.persist_slots(30) == 1
    b = cold_engine(llama, tmp_path, monkeypatch)
    hist_b = list(hist_a)
    live = agent_turn(a, hist_a, "over the lazy dog", max_tokens=8)
    restored = agent_turn(b, hist_b, "over the lazy dog", max_tokens=8)
    assert restored.cached_tokens == live.cached_tokens > 0
    assert restored.text == live.text


def test_a_restored_gemma_slot_continues_with_identical_ids_past_the_window(
        gemma_toys, tmp_path, monkeypatch):
    """The gate on gemma's sliding layers, by ids —
    stronger than the two gates above's `.text` compare: these toys emit
    `<unk>` now and then and skip_special_tokens drops it from `.text`
    (test_serving_gemma_spec.py's `_run_ids` docstring), so a one-token
    divergence there could hide inside an unchanged string. `slot.ids` is the
    cache's own record of every token it was fed and every token it emitted.

    `cached_tokens > 0` is the load-bearing assertion, not decoration: when
    the sliding counters are not restored, written_tokens' own disagreement check (a sliding layer's tensor frozen at
    0 or the window, next to a full-attention layer's tensor that is not)
    already refuses this exact restore and falls back to a full cold
    reprefill — which, being the same deterministic greedy weights, reproduces
    identical ids for the wrong reason (recomputed from scratch, not
    restored). Measured with the counters left out: 'cache counter says -1
    tokens, header says 409 — prefilling cold', restored.cached_tokens == 0.
    Without this assertion the ids-equal check alone passes on BOTH sides of
    the fix and proves nothing."""
    toy = gemma_toys[16]
    a = cold_engine(toy, tmp_path, monkeypatch, ctx=1024)
    hist_a = []
    agent_turn(a, hist_a, LONG)  # crosses window 16 many times over, live
    assert a.persist_slots(30) == 1

    b = cold_engine(toy, tmp_path, monkeypatch, ctx=1024)
    hist_b = list(hist_a)
    live = agent_turn(a, hist_a, "over the lazy dog", max_tokens=8)   # the live cache continues
    restored = agent_turn(b, hist_b, "over the lazy dog", max_tokens=8)  # the restored cache continues
    assert restored.cached_tokens == live.cached_tokens > 0  # a genuine restore, not a refusal

    slot_a = max(a._slots, key=lambda s: s.stamp)
    slot_b = max(b._slots, key=lambda s: s.stamp)
    assert len(slot_a.ids) > 16 and slot_a.ids == slot_b.ids


def test_sleep_wake_carries_a_sliding_layers_python_counter(gemma_toys, tmp_path,
                                                            monkeypatch):
    """The same counter carried through sleep/wake's _park_slots/_restore_slots
    (serving/engines.py), which round-trips cache_state on the exact same
    SlotStore SIGTERM's persist_slots does — one bug, one fix, both callers."""
    toy = gemma_toys[16]
    eng = cold_engine(toy, tmp_path, monkeypatch, ctx=1024)
    hist = []
    agent_turn(eng, hist, LONG)  # crosses window 16 many times over

    live_slot = max(eng._slots, key=lambda s: s.stamp)
    sliding = [i for i, l in enumerate(live_slot.cache.layers)
              if slotstore._is_sliding_layer(l)]
    assert sliding
    live_ints = {i: int(live_slot.cache.layers[i].cumulative_length_int)
                for i in sliding}
    assert min(live_ints.values()) > 16  # actually past the window
    live_written = slotstore.written_tokens(live_slot.cache)

    st = eng.sleep(1)
    assert st["slots_persisted"] == 1 and st["slots_dropped"] == 0
    assert live_slot.cache is None  # genuinely dropped, vLLM's level 1

    st = eng.wake()
    assert st["slots_restored"] == 1
    woke_slot = max(eng._slots, key=lambda s: s.stamp)
    woke_ints = {i: int(woke_slot.cache.layers[i].cumulative_length_int)
                for i in sliding}
    assert woke_ints == live_ints
    assert slotstore.written_tokens(woke_slot.cache) == live_written


def test_an_eviction_lands_on_disk_and_comes_back(llama, tmp_path, monkeypatch):
    """The cold tier's own loop, one slot deep: A is evicted by B, and A's
    next turn is restored from disk instead of prefilled cold — which is the
    thing a single-slot server could never do before."""
    eng = cold_engine(llama, tmp_path, monkeypatch)
    hist_a, hist_b = [], []
    agent_turn(eng, hist_a, LONG)
    # evicts A: a conversation that shares only the template's opening with
    # it (with context checkpoints on, one that shared most of A's prompt
    # would be served from A's own slot up to that prefix)
    agent_turn(eng, hist_b, OTHER)
    assert eng._cold.flush(30)
    back = agent_turn(eng, hist_a, "over the lazy dog")
    assert back.cached_tokens > 0
    assert eng._cold.hits == 1


def test_a_slot_computed_against_another_pack_is_never_loaded(llama, tmp_path,
                                                              monkeypatch):
    """The pack's hashes are the key: the same model served from a different pack is
    different bytes, and KV computed against one has nothing to say about the
    other."""
    monkeypatch.setenv("DRINKME_SLOT_DIR", str(tmp_path))
    from drinkme.serving.engines import HFEngine

    model, tok = llama[0], llama[1]
    a = HFEngine(model, tok, model_id="toy", arm="compressed",
                 meta=dict(TOY_META, packId="packaaaa"), ctx=1024)
    hist = []
    agent_turn(a, hist, LONG)
    assert a.persist_slots(30) == 1
    b = HFEngine(model, tok, model_id="toy", arm="compressed",
                 meta=dict(TOY_META, packId="packbbbb"), ctx=1024)
    assert b._cold is not None and b._cold._entries == {}
    assert agent_turn(b, list(hist), "over the lazy dog").cached_tokens == 0


def test_reset_prefix_cache_stops_the_cold_tier_answering(llama, tmp_path,
                                                          monkeypatch):
    """"Cold in its own terms" cannot mean "and then handed yesterday's prefix
    off an SSD" — bench/prefix_cache_verify.py's whole instrument rests on
    this."""
    eng = cold_engine(llama, tmp_path, monkeypatch)
    hist = []
    agent_turn(eng, hist, LONG)
    assert eng.persist_slots(30) == 1
    eng.reset_prefix_cache()
    assert agent_turn(eng, list(hist), "over the lazy dog").cached_tokens == 0
    assert eng._cold.hits == 0


def test_zero_slots_leave_no_cold_tier_at_all(llama, tmp_path, monkeypatch):
    eng = cold_engine(llama, tmp_path, monkeypatch, slots=0)
    assert eng._cold is None
    assert eng.persist_slots(1) == 0


def test_slot_dir_off_leaves_no_cold_tier_at_all(llama, monkeypatch):
    monkeypatch.setenv("DRINKME_SLOT_DIR", "off")
    eng = engine(llama, slots=1, ctx=1024)
    assert eng._cold is None


def test_a_slot_mid_generation_is_not_persisted(llama, tmp_path, monkeypatch):
    """persist_slots runs on the SIGTERM path while a generation may still be
    in flight. A slot being written to presents as having no valid ids —
    engines empties them on entry — so it is skipped rather than serialized
    half-written."""
    eng = cold_engine(llama, tmp_path, monkeypatch)
    hist = []
    agent_turn(eng, hist, LONG)
    eng._slots[0].ids = []  # exactly what generate() does on entry
    assert eng.persist_slots(30) == 0


def test_the_boot_index_announces_what_it_found(llama, tmp_path, monkeypatch,
                                                capsys):
    a = cold_engine(llama, tmp_path, monkeypatch)
    hist = []
    agent_turn(a, hist, LONG)
    a.persist_slots(30)
    capsys.readouterr()
    cold_engine(llama, tmp_path, monkeypatch)
    out = capsys.readouterr().out
    assert "cold tier" in out and "1 slot(s)" in out


def test_a_real_static_cache_round_trips_every_surface(hybrid, tmp_path):
    """The generic walk against the real thing: whatever transformers-5 hangs
    on a hybrid's layers — the StaticLayers' keys/values/cumulative_length and
    the LinearAttentionLayers' conv and recurrent dicts — comes back, and
    state_mismatch says so before anything is applied."""
    from transformers import StaticCache

    model, tok, cfg = hybrid
    src = StaticCache(config=cfg, max_cache_len=64)
    with torch.inference_mode():
        model(torch.tensor([[3, 4, 5]]), past_key_values=src, use_cache=True,
              cache_position=torch.arange(3), logits_to_keep=1)
    tensors, flags, ints = slotstore.cache_state(src)
    assert {"0.conv_states.0", "0.recurrent_states.0", "1.keys", "1.values",
            "1.cumulative_length"} <= set(tensors)
    assert flags["0.has_previous_state.0"] and flags["1.is_initialized"]
    assert ints == {}  # this hybrid has no sliding layer

    dst = StaticCache(config=cfg, max_cache_len=64)
    with torch.inference_mode():
        model(torch.tensor([[9]]), past_key_values=dst, use_cache=True,
              cache_position=torch.arange(1), logits_to_keep=1)
    assert slotstore.state_mismatch(dst, tensors, flags, ints) is None
    slotstore.apply_state(dst, tensors, flags, ints)
    assert slotstore.written_tokens(dst) == 3
    for key, want in tensors.items():
        got = slotstore.cache_state(dst)[0][key]
        assert torch.equal(got, want), key


def test_a_width_mismatch_is_named_not_applied(hybrid):
    from transformers import StaticCache

    model, tok, cfg = hybrid
    src = StaticCache(config=cfg, max_cache_len=64)
    narrow = StaticCache(config=cfg, max_cache_len=32)
    with torch.inference_mode():
        for cache in (src, narrow):
            model(torch.tensor([[3]]), past_key_values=cache, use_cache=True,
                  cache_position=torch.arange(1), logits_to_keep=1)
    why = slotstore.state_mismatch(narrow, *slotstore.cache_state(src))
    assert why and "keys" in why


# ---------------------------------- the sliding-window counter --
# transformers' StaticSlidingWindowLayer keeps a python cumulative_length_int
# beside the tensor cumulative_length (mtp.py's module docstring, surface 1b);
# cache_state carried no ints at all, so a restored slot's sliding layers came
# back holding a FRESH layer's own counter (0, then the prime token's 1) no
# matter how far past the window the original had gone.


def test_a_restored_sliding_layer_carries_its_python_counter_past_the_window(
        gemma_toys, tmp_path):
    """The gate on cache_state/apply_state/written_tokens directly (on-disk
    prefix slots' own SIGTERM path), on a REAL gemma-4 StaticCache: window 16,
    56 tokens in one prefill (3.5 ring wraps). The tensor alone would report
    far short of 56 (it never even reaches the window on a one-shot prefill
    this size — see the module docstring on why get_seq_length(), not the raw
    tensor, is the only correct read)."""
    model, tok, cfg = gemma_toys[16]
    n = 56
    src = _prefilled(model, cfg, n, max_cache_len=128)
    sliding = [i for i, l in enumerate(src.layers) if slotstore._is_sliding_layer(l)]
    assert sliding  # the toy really does carry sliding layers
    assert all(src.layers[i].cumulative_length_int == n for i in sliding)

    tok_ids = list(range(n))
    st = SlotStore(str(tmp_path), "ident", HEADER, block=8)
    entry = st.put(tok_ids, 128, src)
    assert entry is not None
    assert st.flush(10)

    def make(alloc):
        from transformers import StaticCache
        return StaticCache(config=cfg, max_cache_len=alloc)

    def prime(cache):
        with torch.inference_mode():
            model(torch.tensor([[0]]), past_key_values=cache, use_cache=True,
                  cache_position=torch.arange(1), logits_to_keep=1)

    got = st.restore(entry, make, prime)
    assert got is not None
    dst, ids_back = got
    assert ids_back == tok_ids
    assert slotstore.written_tokens(dst) == n
    for i in sliding:
        assert dst.layers[i].cumulative_length_int == n


def test_a_restored_sliding_layer_below_the_window_continues_identically_to_never_persisted(
        gemma_toys, tmp_path):
    """The regime every test above skips: 10 tokens
    into a window-16 cache, never having crossed it once. Past the window,
    written_tokens' own disagreement check (a sliding layer's tensor frozen
    next to a full-attention layer's that keeps counting) already refuses a
    restore that skips the int's resync and falls back to a safe cold reprefill — the case every
    other test here exercises. BELOW the window that disagreement never
    fires: both tensors still agree at 10 (the sliding layer's ring is not
    even full yet), so a restore that skips the resync is ACCEPTED, not refused, onto a
    cache whose sliding layer silently keeps the FRESH layer's own
    cumulative_length_int (1, the prime token) instead of the true 10 — the
    mask then reads the wrong offset from the very first generated token.
    Measured with the resync removed: restore is accepted, the
    int comes back 1, and a stepwise greedy
    continuation diverges at the first token, then IndexErrors once absolute
    position 16 (the window) is crossed under the wrong book-keeping."""
    model, tok, cfg = gemma_toys[16]
    n = 10
    src = _prefilled(model, cfg, n, max_cache_len=128)
    sliding = [i for i, l in enumerate(src.layers) if slotstore._is_sliding_layer(l)]
    assert sliding
    assert all(src.layers[i].cumulative_length_int == n for i in sliding)

    tok_ids = list(range(n))
    st = SlotStore(str(tmp_path), "ident", HEADER, block=8)
    entry = st.put(tok_ids, 128, src)
    assert entry is not None
    assert st.flush(10)

    def make(alloc):
        from transformers import StaticCache
        return StaticCache(config=cfg, max_cache_len=alloc)

    def prime(cache):
        with torch.inference_mode():
            model(torch.tensor([[0]]), past_key_values=cache, use_cache=True,
                  cache_position=torch.arange(1), logits_to_keep=1)

    got = st.restore(entry, make, prime)
    assert got is not None  # accepted below the window, not refused
    dst, ids_back = got
    assert ids_back == tok_ids
    assert slotstore.written_tokens(dst) == n
    for i in sliding:
        assert dst.layers[i].cumulative_length_int == n

    def continue_greedy(cache, steps=20):
        ids, tok_id = [], 0
        with torch.inference_mode():
            for j in range(steps):
                out = model(torch.tensor([[tok_id]]), past_key_values=cache,
                           use_cache=True, cache_position=torch.tensor([n + j]),
                           logits_to_keep=1)
                tok_id = int(out.logits[0, -1].argmax())
                ids.append(tok_id)
        return ids

    live = continue_greedy(src)      # never persisted/restored
    restored = continue_greedy(dst)  # went through put/restore below the window
    assert restored == live


def test_a_stale_slot_file_with_no_counter_is_refused_not_silently_restored(
        gemma_toys, tmp_path):
    """A slot file without the python counter (any writer that never learned about
    cumulative_length_int) must be REFUSED for a sliding-layer cache, never
    restored with the fresh layer's own counter standing in for the true one
    — the same posture the store takes for any other layout mismatch."""
    model, tok, cfg = gemma_toys[16]
    n = 56
    src = _prefilled(model, cfg, n, max_cache_len=128)
    tensors, flags, ints = slotstore.cache_state(src)
    assert ints  # this cache does have sliding layers to lose

    from transformers import StaticCache

    dst = StaticCache(config=cfg, max_cache_len=128)
    with torch.inference_mode():
        model(torch.tensor([[0]]), past_key_values=dst, use_cache=True,
              cache_position=torch.arange(1), logits_to_keep=1)
    why = slotstore.state_mismatch(dst, tensors, flags, {})  # no ints carried
    assert why and "cumulative_length_int" in why


def test_the_header_carries_the_whole_index(tmp_path):
    """The on-disk layout in one assertion: identity, the chain, the tail and
    the sizes, all in a file small enough to scan at boot."""
    st = store(tmp_path)
    tok_ids = ids(BLOCK + 40)
    st.put(tok_ids, 512, filled(512, len(tok_ids)))
    assert st.flush(10)
    with open(os.path.join(st.dir, slotstore.prefix_key(tok_ids),
                           "header.json")) as f:
        head = json.load(f)
    assert head["format_version"] == slotstore.FORMAT_VERSION
    assert head["pack_id"] == HEADER["pack_id"] and head["arm"] == "test"
    assert head["n_ids"] == BLOCK + 40 and head["alloc"] == 512
    assert head["block"] == BLOCK
    assert head["chain"] == block_chain(tok_ids) and head["tail"] == tok_ids[BLOCK:]
    assert head["bytes"] > 0 and head["used"] > 0
    assert head["flags"]["0.is_initialized"] is True


@pytest.mark.parametrize("field", ["n_ids", "alloc", "chain", "tail"])
def test_a_header_missing_a_field_is_skipped(tmp_path, field):
    st = store(tmp_path)
    tok_ids = ids(BLOCK + 40)
    st.put(tok_ids, 512, filled(512, len(tok_ids)))
    assert st.flush(10)
    hpath = os.path.join(st.dir, slotstore.prefix_key(tok_ids), "header.json")
    with open(hpath) as f:
        head = json.load(f)
    del head[field]
    with open(hpath, "w") as f:
        json.dump(head, f)
    assert store(tmp_path).index() == (0, 0)


# ------------------------------------ the writer's queue --
# put() hands the writer an entry whose ONLY copy is entry.ram. The RAM-cap
# sweep after each landed write must not clear `ram` for an entry that has
# not reached disk: with the default zero-byte warm tier, every entry
# queued behind the one being written would lose its payload before its
# turn, _write would see `ram is None` and return in silence, and flush()
# would report the drained queue as success. Seeded from a
# GPU race reproduction.


class HeldWriter:
    """Hold the FIRST write on the writer thread until released — the
    deterministic stand-in for "a second slot was evicted while the first
    was still on its way to disk"."""

    def __init__(self, st):
        import threading

        self.st, self.real = st, st._write
        self.entered, self.go = threading.Event(), threading.Event()
        st._write = self

    def __call__(self, entry):
        if not self.entered.is_set():
            self.entered.set()
            assert self.go.wait(10)
        self.real(entry)


def test_a_queued_snapshot_survives_the_sweep_of_the_write_before_it(tmp_path):
    """The ruler for the writer's queue: two writes queued, ram_bytes=0, the first held
    before its disk write. Releasing it must leave BOTH restorable — the
    second entry's payload is pending, not a warm-tier copy the cap may drop.
    The failure it catches: flush() True, saved 1, the second entry indexed
    with no file and no payload."""
    st = store(tmp_path, ram_bytes=0)
    held = HeldWriter(st)
    a_ids, b_ids = ids(BLOCK + 40), ids(BLOCK + 40, start=5000)
    a = st.put(a_ids, 512, filled(512, len(a_ids), seed=1))
    assert held.entered.wait(10)
    b = st.put(b_ids, 512, filled(512, len(b_ids), seed=2))
    assert b.ram is not None  # queued behind the held write, payload in hand
    held.go.set()
    assert st.flush(10)
    assert st.saved == 2  # 1 would mean the second write found no payload
    for e in (a, b):
        assert os.path.isfile(os.path.join(e.path, "header.json"))
        assert os.path.isfile(os.path.join(e.path, "cache.safetensors"))
        assert e.ram is None  # the warm tier is off: let go AFTER the write
    fresh = store(tmp_path)
    assert fresh.index()[0] == 2
    for tok_ids, seed in ((a_ids, 1), (b_ids, 2)):
        got = fresh.lookup(tok_ids + [7], need=10, ctx=8192)
        assert got is not None
        live, back = fresh.restore(got, make_cache(512), noprime)
        assert back == tok_ids
        assert torch.equal(live.layers[0].keys, filled(512, len(tok_ids), seed=seed).layers[0].keys)
    assert a.committed and b.committed and st.failed == 0


def test_a_pending_entry_with_no_payload_is_a_surfaced_error(tmp_path, capsys):
    """_write meeting a pending entry whose payload is gone is a bug in the
    store, not a condition to return quietly from: it is logged as a failed
    write, the entry leaves the index, flush() says False."""
    st = store(tmp_path, ram_bytes=0)
    held = HeldWriter(st)
    a = st.put(ids(BLOCK + 40), 512, filled(512, BLOCK + 40))
    assert held.entered.wait(10)
    b = st.put(ids(BLOCK + 40, start=5000), 512, filled(512, BLOCK + 40))
    b.ram = None  # what the old sweep did to it
    held.go.set()
    assert st.flush(10) is False
    assert st.saved == 1 and st.failed == 1
    assert a.committed and not b.committed
    assert b.key not in st._entries
    assert not os.path.exists(os.path.join(b.path, "header.json"))
    err = capsys.readouterr().err
    assert f"write {b.key} failed" in err and "no payload" in err


def test_an_interrupted_write_is_not_indexed_and_never_restored(tmp_path, capsys):
    """The writer dies mid-file (the tensor write raises after the directory
    exists). The entry must not be indexed as persisted — not in this
    process, not by a fresh boot index — flush() must say so, and the
    failure must read as a write failure, not a cap eviction."""
    st = store(tmp_path, ram_bytes=0)
    tok_ids = ids(BLOCK + 40)

    real_write = st._write

    def dies(entry):
        os.makedirs(entry.path, exist_ok=True)
        with open(os.path.join(entry.path, "cache.safetensors.tmp"), "wb") as f:
            f.write(b"half a tensor")
        raise OSError(28, "No space left on device")

    st._write = dies
    e = st.put(tok_ids, 512, filled(512, len(tok_ids)))
    assert st.flush(10) is False
    assert st.failed == 1 and st.saved == 0
    assert e.pending is False and e.committed is False
    assert e.key not in st._entries and e.ram is None
    assert st.lookup(tok_ids + [1], need=10, ctx=8192) is None
    assert store(tmp_path).index() == (0, 0)  # nothing partial is indexed
    err = capsys.readouterr().err
    assert f"write {e.key} failed" in err and "No space left" in err
    assert "evicted" not in err
    st._write = real_write  # a later write still lands
    e2 = st.put(tok_ids, 512, filled(512, len(tok_ids)))
    assert st.flush(10) is True and e2.committed
    assert store(tmp_path).index()[0] == 1


def test_flush_is_true_only_when_every_queued_write_committed(tmp_path):
    """One failure among three queued writes: the other two land, but the
    queue did not all commit and flush() must not say it did. The next
    flush, with nothing new failed, is clean again."""
    st = store(tmp_path, ram_bytes=0)
    real_write = st._write
    doomed = slotstore.prefix_key(ids(BLOCK + 40, start=5000))

    def flaky(entry):
        if entry.key == doomed:
            raise RuntimeError("disk went away")
        real_write(entry)

    st._write = flaky
    entries = [st.put(ids(BLOCK + 40, start=s), 512, filled(512, BLOCK + 40))
               for s in (0, 5000, 9000)]
    assert st.flush(10) is False
    assert [e.committed for e in entries] == [True, False, True]
    assert st.saved == 2 and st.failed == 1
    assert st.flush(10) is True  # the failure was reported once, to its flush


def test_sleep_counts_committed_writes_not_queued_ones(llama, tmp_path, monkeypatch):
    """sleep's `slots_persisted` is what /health tells an operator survived
    the sleep. A queued write that failed is not a persisted slot, and the
    wake must not try to restore it from a file that is not there."""
    eng = cold_engine(llama, tmp_path, monkeypatch)
    hist = []
    agent_turn(eng, hist, LONG)

    def dies(entry):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(eng._cold, "_write", dies)
    st = eng.sleep(1)
    assert st["slots_persisted"] == 0 and st["slots_dropped"] == 1
    assert eng._cold.failed == 1
    assert eng._slept_entries == []
    st = eng.wake()
    assert st["slots_restored"] == 0


# ------------------------------------ the files' permissions --
# A slot header carries `tail`: real token ids past the last block boundary,
# up to 255 of them, decodable with the public tokenizer. Under umask 022 the
# header would land 0644 in a 0755 tree — readable by any local user who
# can traverse the ancestors — so the store sets its own modes.


def _mode(path):
    import stat

    return stat.S_IMODE(os.stat(path).st_mode)


@pytest.fixture
def umask_022():
    old = os.umask(0o022)
    try:
        yield
    finally:
        os.umask(old)


def test_slot_files_are_private_under_a_permissive_umask(tmp_path, umask_022):
    """The ruler for the file modes: block 4, a 7-token prompt — three ids ride in the
    header's tail. Written under umask 022, the header must be 0600, the
    tensor file 0600, and every directory of the store's own tree 0700.
    The failure it catches: header 0644, directories 0755."""
    st = SlotStore(str(tmp_path), "ident", HEADER, block=4)
    tok_ids = [60, 61, 62, 63, 64, 65, 66]
    e = st.put(tok_ids, 16, filled(16, len(tok_ids)))
    assert e is not None and st.flush(10)
    hpath = os.path.join(e.path, "header.json")
    with open(hpath) as f:
        head = json.load(f)
    assert head["tail"] == [64, 65, 66]  # the ids that must not be world-readable
    assert _mode(hpath) == 0o600
    assert _mode(os.path.join(e.path, "cache.safetensors")) == 0o600
    assert _mode(e.path) == 0o700
    assert _mode(st.dir) == 0o700
    # touch() rewrites the header through a temp name: still private
    st.touch(e)
    assert _mode(hpath) == 0o600
    assert not [n for n in os.listdir(e.path) if n.endswith(".tmp")]


def test_a_permissive_existing_store_is_tightened_on_open(tmp_path, umask_022, capsys):
    """A store written by an older drinkme (or loosened by hand) is not
    assumed fixed by the new default: opening it chmods every directory to
    0700 and every file to 0600, and says so in one line."""
    st = store(tmp_path)
    tok_ids = ids(BLOCK + 40)
    e = st.put(tok_ids, 512, filled(512, len(tok_ids)))
    assert st.flush(10)
    hpath = os.path.join(e.path, "header.json")
    tpath = os.path.join(e.path, "cache.safetensors")
    os.chmod(hpath, 0o644)
    os.chmod(tpath, 0o644)
    os.chmod(e.path, 0o755)
    os.chmod(st.dir, 0o755)
    capsys.readouterr()
    fresh = store(tmp_path)
    assert fresh.index()[0] == 1
    assert _mode(hpath) == 0o600 and _mode(tpath) == 0o600
    assert _mode(e.path) == 0o700 and _mode(fresh.dir) == 0o700
    out = capsys.readouterr().out
    assert out.count("tightened") == 1 and "4 path(s)" in out
    capsys.readouterr()
    assert store(tmp_path).index()[0] == 1
    assert "tightened" not in capsys.readouterr().out  # nothing left to tighten


def test_open_store_creates_the_tree_private(tmp_path, umask_022, monkeypatch):
    """The root the default config creates under ~/.cache and the identity
    directory beneath it: 0700 each, from creation, umask notwithstanding."""
    root = tmp_path / "slots"
    monkeypatch.setenv("DRINKME_SLOT_DIR", str(root))
    st = _open_store(model_id="toy", arm="test", pack_id=None, dtype="float32")
    assert st is not None
    assert _mode(str(root)) == 0o700
    assert _mode(st.dir) == 0o700


# the effective attention configuration a loader reads off model.config
#: a Llama-shaped default, and a YaRN-scaled twin
ATTN = {"rope_theta": 10000.0, "rope_parameters": {"rope_theta": 10000.0, "rope_type": "default"},
        "rope_scaling": {"rope_theta": 10000.0, "rope_type": "default"},
        "max_position_embeddings": 512, "sliding_window": None, "layer_types": None}
ATTN_YARN = dict(ATTN, rope_parameters={"rope_theta": 10000.0, "rope_type": "yarn", "factor": 4.0,
                                        "original_max_position_embeddings": 512},
                 rope_scaling={"rope_theta": 10000.0, "rope_type": "yarn", "factor": 4.0,
                               "original_max_position_embeddings": 512},
                 max_position_embeddings=2048)


def _open_store(model_id="toy", arm="test", pack_id=None, dtype="float32",
                block=BLOCK, attention=ATTN, **meta):
    """open_store with the arguments this file cares about; `meta` is the
    loader's dict (resolvedRevision, packId). `attention` is passed only
    where open_store takes it, so that against a (model, arm, pack id)-only store the identity
    tests fail on the IDENTITY (equal paths, equal headers) and not on a
    keyword such a store's signature never had."""
    import inspect

    if pack_id is not None:
        meta["packId"] = pack_id
    meta.setdefault("resolvedRevision", TOY_META["resolvedRevision"])
    kw = {"block": block}
    if "attention" in inspect.signature(slotstore.open_store).parameters:
        kw["attention"] = attention
    return slotstore.open_store(model_id, arm, meta, 1024, dtype, **kw)


# ---------------------------------- the store's identity --
# A KV cache is a transformers cache layout serving one exact set of weights
# under one positional transformation. An identity of only (model_id, arm,
# pack_id, dtype, transformers) is not enough: stock engines pass meta={},
# so two revisions of one repo would share a directory and identical
# headers, and without the RoPE settings `--rope-scaling` could reuse keys
# computed under a different rotation.


def test_two_revisions_of_the_same_model_have_different_stores(tmp_path, monkeypatch):
    """The ruler for the store identity: same stock model, resolved
    revisions aaa vs bbb -> different store paths AND unequal headers.
    The failure it catches: equal paths, equal headers."""
    monkeypatch.setenv("DRINKME_SLOT_DIR", str(tmp_path))
    a = _open_store(resolvedRevision="aaa")
    b = _open_store(resolvedRevision="bbb")
    assert a is not None and b is not None
    assert a.dir != b.dir
    assert a.header != b.header
    assert a.header["revision"] == "aaa" and b.header["revision"] == "bbb"


def test_two_rope_configurations_have_different_identities(tmp_path, monkeypatch):
    """Same weights, same revision, one served with YaRN x4: the keys in a
    slot were rotated under one of the two and are wrong under the other."""
    monkeypatch.setenv("DRINKME_SLOT_DIR", str(tmp_path))
    plain = _open_store(attention=ATTN)
    yarn = _open_store(attention=ATTN_YARN)
    assert plain.dir != yarn.dir
    assert plain.header != yarn.header
    assert plain.header["attention"]["rope_scaling"]["rope_type"] == "default"
    assert yarn.header["attention"]["rope_scaling"]["factor"] == 4.0
    # and a header written by one is refused by the other, field named
    tok_ids = ids(BLOCK + 40)
    assert plain.put(tok_ids, 512, filled(512, len(tok_ids))) is not None
    assert plain.flush(10)
    other = SlotStore(str(tmp_path), os.path.basename(plain.dir), yarn.header)
    assert other.index() == (0, 0)


@pytest.mark.parametrize("field", ["revision", "attention"])
def test_a_header_from_before_the_identity_fields_is_refused(tmp_path, monkeypatch,
                                                            capsys, field):
    """An existing store whose header lacks the new fields is a slot file
    that does not name the weights or the rotation it was computed under:
    skipped with one line, never restored as if compatible."""
    monkeypatch.setenv("DRINKME_SLOT_DIR", str(tmp_path))
    st = _open_store()
    tok_ids = ids(BLOCK + 40)
    e = st.put(tok_ids, 512, filled(512, len(tok_ids)))
    assert st.flush(10)
    hpath = os.path.join(e.path, "header.json")
    with open(hpath) as f:
        head = json.load(f)
    del head[field]
    with open(hpath, "w") as f:
        json.dump(head, f)
    capsys.readouterr()
    fresh = _open_store()
    assert fresh.index() == (0, 0)
    err = capsys.readouterr().err
    assert f"lacks {field}" in err
    assert fresh.lookup(tok_ids + [1], need=10, ctx=8192) is None


def test_a_store_needs_a_resolved_revision_and_an_attention_config(tmp_path, monkeypatch,
                                                                   capsys):
    """No revision, no cold tier: a store that cannot name its weights has
    nothing to key on. Same for an engine that cannot say how it rotates."""
    monkeypatch.setenv("DRINKME_SLOT_DIR", str(tmp_path))
    assert slotstore.open_store("toy", "test", {}, 1024, "float32", attention=ATTN) is None
    assert "revision" in capsys.readouterr().err
    assert slotstore.open_store("toy", "test", dict(TOY_META), 1024, "float32",
                                attention=None) is None
    assert "attention" in capsys.readouterr().err
    assert not os.listdir(tmp_path)  # and no directory was made for either


def test_an_engine_without_a_resolved_revision_serves_without_a_cold_tier(
        llama, tmp_path, monkeypatch):
    monkeypatch.setenv("DRINKME_SLOT_DIR", str(tmp_path))
    eng = engine(llama, slots=1, ctx=1024)  # meta={} — the old load_stock
    assert eng._cold is None
    assert eng.persist_slots(1) == 0


def _yarn_twin(llama):
    """The llama toy's weights under a YaRN x2 rotation — what
    `--rope-scaling` does to the same checkpoint."""
    import copy

    from transformers import LlamaForCausalLM

    from drinkme.serving.engines import _apply_rope_scaling

    model, tok = llama[0], llama[1]
    cfg = copy.deepcopy(model.config)
    _apply_rope_scaling(cfg, {"rope_type": "yarn", "factor": 2})
    twin = LlamaForCausalLM(cfg).eval().to(torch.float32)
    twin.load_state_dict(model.state_dict())
    return twin, tok


def test_a_slot_computed_under_another_rope_scaling_is_never_loaded(llama, tmp_path,
                                                                    monkeypatch):
    """Through the engine: A persists a slot; B, the same weights served
    with YaRN x2, must prefill cold rather than restore A's keys. The
    control — A's own twin restoring A's slot — proves the store is live.
    The failure it catches: B restores A's slot (cached_tokens > 0)."""
    monkeypatch.setenv("DRINKME_SLOT_DIR", str(tmp_path))
    from drinkme.serving.engines import HFEngine

    a = engine(llama, slots=1, ctx=1024, meta=dict(TOY_META))
    hist = []
    agent_turn(a, hist, LONG)
    assert a.persist_slots(30) == 1

    twin, tok = _yarn_twin(llama)
    b = HFEngine(twin, tok, model_id="toy", arm="test", meta=dict(TOY_META), ctx=1024)
    assert b._cold is not None
    assert agent_turn(b, list(hist), "over the lazy dog").cached_tokens == 0
    assert b._cold.dir != a._cold.dir
    assert b._cold.header["attention"]["rope_scaling"]["rope_type"] == "yarn"

    control = engine(llama, slots=1, ctx=1024, meta=dict(TOY_META))
    assert control._cold.dir == a._cold.dir
    assert agent_turn(control, list(hist), "over the lazy dog").cached_tokens > 0


def test_attention_identity_reads_the_effective_config(llama, gemma_toys):
    """What the engine keys on: RoPE (theta, the full parameters dict, the
    scaling alias), the window, the sliding window, the layer types — read
    off the text config, JSON-shaped so the header on disk equals the one
    in memory."""
    ident = slotstore.attention_identity(llama[0].config)
    assert ident["rope_parameters"]["rope_theta"] == 10000.0
    assert ident["rope_scaling"]["rope_type"] == "default"
    assert ident["max_position_embeddings"] == 512
    assert ident["sliding_window"] is None and ident["layer_types"] is None
    assert json.loads(json.dumps(ident)) == ident

    g = slotstore.attention_identity(gemma_toys[16][2])
    assert g["sliding_window"] == 16
    assert g["layer_types"][:2] == ["sliding_attention", "sliding_attention"]
    assert g["rope_parameters"]["full_attention"]["rope_theta"] == 1000000.0

    twin, _ = _yarn_twin(llama)
    y = slotstore.attention_identity(twin.config)
    assert y["rope_scaling"]["rope_type"] == "yarn" and y["rope_scaling"]["factor"] == 2
    assert y["max_position_embeddings"] == 1024
    assert y != ident


# --- the loaders: what load_stock / load_compressed pass ---------------


@pytest.fixture
def toy_checkpoint(tmp_path):
    """A tiny real Llama + tokenizer on disk, test_serving_gen_config.py's
    toy_dir shape, function-scoped because these tests rewrite it."""
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

    d = tmp_path / "ckpt"
    d.mkdir()
    vocab = {"<unk>": 0, "<pad>": 1, "</s>": 2, "hi": 3, "user": 4, "assistant": 5, ":": 6}
    backend = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
    backend.pre_tokenizer = Whitespace()
    tok = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="<unk>",
                                  pad_token="<pad>", eos_token="</s>")
    tok.chat_template = ("{% for m in messages %}{{ m['role'] }} : {{ m['content'] }} "
                         "{% endfor %}{% if add_generation_prompt %}assistant :{% endif %}")
    tok.save_pretrained(str(d))

    def write(seed):
        cfg = LlamaConfig(vocab_size=32, hidden_size=32, intermediate_size=32,
                          num_hidden_layers=1, num_attention_heads=2,
                          num_key_value_heads=1, max_position_embeddings=64,
                          tie_word_embeddings=True, eos_token_id=2, pad_token_id=1)
        torch.manual_seed(seed)
        LlamaForCausalLM(cfg).to(torch.bfloat16).eval().save_pretrained(
            str(d), safe_serialization=True)

    write(0)
    return str(d), write


def test_resolved_revision_is_the_snapshot_sha_or_a_local_digest(toy_checkpoint):
    from drinkme.serving.checkpoint import resolved_revision

    sha = "0123456789abcdef0123456789abcdef01234567"
    assert resolved_revision(f"/hf/hub/models--Q--M/snapshots/{sha}") == sha
    assert resolved_revision(f"/hf/hub/models--Q--M/snapshots/{sha}/") == sha

    d, write = toy_checkpoint
    first = resolved_revision(d)
    assert first.startswith("local-") and first == resolved_revision(d)
    write(1)  # different weights, same shapes, same file names
    assert resolved_revision(d) != first


def test_load_stock_binds_the_store_to_the_checkpoint_and_its_rotation(
        toy_checkpoint, tmp_path, monkeypatch):
    """The loader, end to end on a local checkpoint: the store it opens is
    keyed on the weights' resolved revision and the effective attention
    config; a rewrite of the weights and a --rope-scaling each move it."""
    from drinkme.serving.engines import load_stock

    monkeypatch.setenv("DRINKME_SLOT_DIR", str(tmp_path / "slots"))
    d, write = toy_checkpoint
    eng = load_stock(d, None, device="cpu")
    assert eng._cold is not None
    rev = eng.meta["resolvedRevision"]
    assert rev.startswith("local-")
    assert eng._cold.header["revision"] == rev
    assert eng._cold.header["attention"]["max_position_embeddings"] == 64
    assert rev[:12] in os.path.basename(eng._cold.dir)

    yarn = load_stock(d, None, device="cpu", rope_scaling={"rope_type": "yarn", "factor": 2})
    assert yarn.meta["resolvedRevision"] == rev  # same bytes
    assert yarn._cold.dir != eng._cold.dir  # different rotation
    assert yarn._cold.header["attention"]["rope_scaling"]["rope_type"] == "yarn"
    assert yarn._cold.header["attention"]["max_position_embeddings"] == 128

    write(1)
    again = load_stock(d, None, device="cpu")
    assert again.meta["resolvedRevision"] != rev
    assert again._cold.dir != eng._cold.dir


def test_load_compressed_keys_the_store_on_the_packs_source_revision(
        toy_pack, tmp_path, monkeypatch):
    """The compressed arm, end to end on the real toy pack: the store's
    revision is meta.json's `source.revision` (the packer's record of the
    checkpoint commit it read — the field rv-identity writes) when the pack
    carries one, else — a local-directory pack, which never records a hub
    revision — the source digest. Either way it is in the header AND the
    path, next to the pack's hashes, and rebinding a pack to another
    checkpoint (source edited, the manifest rehashed) moves the cache id,
    even though the self-contained pack itself is untouched."""
    import shutil

    from drinkme.serving.engines import load_compressed

    monkeypatch.setenv("DRINKME_SLOT_DIR", str(tmp_path / "slots"))
    model_dir, self_contained = toy_pack[0], toy_pack[1]
    # a self-contained pack cut from a local directory streams from its own
    # checkpoint/ and keys on the source identity it was checked against
    sc = load_compressed(model_dir, None, self_contained, device="cpu", ctx=256)
    digest = json.load(open(os.path.join(self_contained, "meta.json")))["source"]["digest"]
    assert sc.meta["resolvedRevision"] == sc._cold.header["revision"] == "local-" + digest[:16]
    assert sc._cold.header["pack_id"] == sc.meta["packId"]
    assert sc._cold.header["attention"]["max_position_embeddings"] == 256

    recorded = str(tmp_path / "pack-with-source")
    shutil.copytree(self_contained, recorded)
    from drinkme.codec import identity

    with open(os.path.join(recorded, "meta.json")) as f:
        meta = json.load(f)
    # keep the digest so the embedded checkpoint still matches; the source
    # block is hashed into the manifest, so rehash after editing
    meta["source"]["revision"] = "c" * 40
    meta["manifestSha256"] = identity.manifest_digest(meta)
    with open(os.path.join(recorded, "meta.json"), "w") as f:
        json.dump(meta, f)
    again = load_compressed(model_dir, None, recorded, device="cpu", ctx=256)
    assert again.meta["resolvedRevision"] == "c" * 40
    # source is part of the manifest — a pack rebound to another checkpoint
    # is a different served model, so its cache id moves too
    assert again.meta["packId"] != sc.meta["packId"]
    assert again._cold.dir != sc._cold.dir  # a different checkpoint beside them
