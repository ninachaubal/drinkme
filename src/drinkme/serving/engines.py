"""The real engines: ONE HFEngine class, two arms — the product's core claim.

`load_stock` and `load_compressed` both return HFEngine over the same HTTP
layer, tokenizer, chat template, sampling loop, and StaticCache KV; the ONLY
difference is how the Linear weights are read ("same HTTP/tokenizer/sampling/KV,
only the weight-read differs"). That is what makes
serve's A/B a measurement instead of a comparison of two stacks.

`load_compressed` is THE FIT PATH: the skeleton is built on the meta device
and every tensor streams in one at a time — checkpoint tensors from the HF
safetensors, compressed Linears from the pack via iter_pack_dir — so peak host
memory is one tensor, never one model. The BF16 weight of a compressed Linear
is never materialized on the way to the device.

Decode loop: cache_position bookkeeping and preallocated step/position
buffers, so the hot loop allocates no tensors of its own.

PREFIX/KV CACHE: ONE StaticCache per SLOT (serving/kvcache.py's
LiveStaticCache), reused across requests, allocated GEOMETRICALLY (KV_FLOOR,
doubling, capped at ctx) rather than ctx-wide. The allocation bounds MEMORY
only: the cache's full-attention layers hand attention the live window, so a
decode step costs the conversation's length, not the allocation's. (Plain
StaticCache attends across the full ALLOCATED width — a 32k allocation for a
4k conversation taxed decode ~2.6x in a verify run, and
the 4096 floor cost a 204-token request 28 ms of a 60 ms token;
kvcache.py's docstring has the measurement.) When a request outgrows the
allocation the cache is DROPPED and rebuilt bigger — one cold re-prefill per
size class per session, logged, in exchange for never touching StaticCache
internals (`layers[i].keys` is mid-deprecation-cycle; a copy would ride on
it). Each
generate() computes the longest common prefix between the incoming prompt ids
and the ids whose KV already sits in the cache — which includes the tokens
GENERATED last turn, so an agent harness resending full history re-prefills
only the delta (its last assistant turn as re-rendered + the new user turn),
not the conversation. Correctness posture:
  - a prompt that EXTENDS the entire written cache (lcp == len(slot.ids),
    strictly less than n_prompt) prefills only the suffix. transformers-5
    static layers IGNORE the cache_position you pass and write at an
    internal cumulative counter (cache_utils.py, StaticLayer.update), so a
    mid-cache rewind is the counter's move, never a cache_position: in the
    extending case lcp == cumulative and the suffix lands true;
  - a prompt that PARTS from the cache (a re-rendered history, an exact
    repeat, a post-abort retry) is served up to the shared prefix by
    CONTEXT CHECKPOINTS (serving/ctx_checkpoints.py, llama.cpp's
    --ctx-checkpoints): full-attention layers rewind by length
    (kvcache.LiveStaticLayer.rewind moves the counter), and the state that
    cannot rewind — sliding-window rings, DeltaNet recurrent and conv
    states — comes back from a snapshot taken at that position during an
    earlier prefill. With --ctx-checkpoints 0 all of that RESETS to a fresh
    cache and a full prefill instead, the extends-only rule byte for byte;
  - either way at least one prompt token always re-runs, so the sampled
    logits are computed, never remembered;
  - slot.ids (and its checkpoints) are emptied on entry and restored on
    exit with exactly the ids whose KV was written, so an exception
    mid-generate leaves an INVALID cache, never a lying one; the next
    request simply resets.
Numerics: a reused prefix is the earlier forward's KV verbatim; re-prefilling
the suffix batches differently than a cold full prefill, so warm-vs-cold is
"same computation modulo kernel reduction order" (chunked-prefill class), while
the stock-vs-compressed A/B stays exact — both arms run the identical policy.
--prefix-slots 0 (DRINKME_PREFIX_SLOTS=0) restores per-request caches.

MULTI-SLOT (the prefix cache): DRINKME_PREFIX_SLOTS=N keeps N whole cache
states instead of one. A slot IS a StaticCache object plus the ids it holds:
on the hybrid 27B that one object carries the attention layers' KV, the
DeltaNet recurrent state AND the conv state together (transformers builds
both layer kinds inside the same cache), so nothing here has to know which
is which. A request runs on the slot that serves the most of it — by
extension, or up to a shared prefix a rewind or checkpoint reaches
(pick_slot below, with llama.cpp's slot rules); with no such slot the LEAST
RECENTLY USED slot is taken cold and rebuilt — which is why a match never
evicts anything. Default N=1: one slot, and with --ctx-checkpoints 0
pick_slot reduces to the extending test above verbatim.

SSD-PERSISTED SLOTS (serving/slotstore.py, docs/serve-prefix-slots.md): under the N live
slots sits a COLD TIER on disk. An evicted slot is copied to host RAM and
written to `DRINKME_SLOT_DIR` as safetensors; a prompt that matches nothing
live consults that store BEFORE prefilling cold, and a restored slot is a
live slot again — same tensors, same extends-only rule. Three things about it
live in THIS file: the store is built at engine construction (so the boot
index and its byte count are announced in the load block, and NOTHING is
loaded onto the accelerator until a request matches), `_take_from_cold` is
the one place generate() consults it (only on the lcp == 0 path — the in-RAM
picker is untouched, byte for byte), and `persist_slots` writes the live
slots out for serve.py's SIGTERM handler. The keying, the block chain, the
LRU and every refusal are slotstore.py's. A restore that cannot be verified
against a live cache of this process costs the prefill it would have saved
and nothing else.

SLEEP / WAKE (serving/sleep.py, docs/serve-sleep.md): `sleep(level)` parks every
device tensor this engine holds — the trunk's raw parameters and buffers, every
CompressedLinear's packed planes and escape/mode/CSR sidecars, the MTP head —
in host RAM and gives the accelerator back, without the process stopping;
`wake()` reverses it. Four things about it live in THIS file: `_park_slots`
sends the live slots down the SAME cold-tier path SIGTERM takes (persist,
then drop — vLLM's "discard the KV cache", with on-disk prefix slots meaning discarding is not
losing), `_restore_slots` brings back exactly the slots this sleep wrote,
`_wake_prime` drives one token through the woken model so a half-landed move
fails in the wake's own log line, and `_reload` — the loader closure — is how
a LEVEL 2 wake comes back: it replays load_stock/load_compressed, so level 2
is the current stop and its wake is a cold boot, with no second code path to keep
in sync. The tensor pass itself, and the reason pinning is opt-in, are
serving/sleep.py's. The CALLER owns the exclusion: serving/http.py holds the
generation lock across both calls.

IMAGES (serving/image_prompt.py, serving/vision.py): a request carrying
images is expanded, keyed and positioned by an ImagePrompt before the slot
is picked. Every slot and cold-tier comparison uses its `key_ids`, in which
each image's run is that image's content key, because two same-size images
render to the same ids and must never share KV. The model sees the expanded
ids through `inputs_embeds` (the tower's output over the image rows, one
prefill span at a time), and every forward, the prompt's and every decode,
verify and re-arm step after it, takes Qwen3.5's M-RoPE `position_ids`. A
text request passes neither and keys on its own ids: its calls are the
ones it made before images existed.

The extends-only rule was not a simplification that more slots could relax:
the DeltaNet recurrent state has NO sequence dimension, so it cannot be
partially rewound to an earlier position at all (extends-only reuse on the
real 27B: max abs delta < 1.5e-7). A context checkpoint does not rewind it
either: it puts back a copy of the state as it stood at that position.

A slot costs memory, so the engine measures what one costs — `fixed_bytes +
ring_bytes + per_token_bytes * alloc`, read off the REAL cache tensors by
allocating two short probe caches and differencing each layer against itself
(layer_bytes/measure_slot_layers); a sliding layer's ring is charged at the
rows it holds, min(window, ctx), not at ctx —
announces it at load, and puts N slots at the FULL ctx through the fit
story's own margin (suggest.FIT_HEADROOM, reached via mtp.residency_check so
the number is read and not invented, against a live mem_get_info). An
explicit N>1 that does not fit is REFUSED at load with the arithmetic
printed; the default single slot only warns, because refusing a config that
served yesterday would be a regression wearing a safeguard's clothes.

MTP SPECULATIVE DECODING (serving/mtp.py): with DRINKME_MTP_DEPTH=k the
one-token step becomes a draft/verify cycle — the checkpoint's own MTP head
drafts k tokens, the trunk verifies all k+1 positions in ONE forward, and the
longest agreeing prefix is emitted. Same tokens, fewer weight-reads. A
SAMPLED request speculates too, by rejection sampling
(serving/speculative.py): same distribution per token, fewer weight-reads.
Four things about it live in THIS file: `pending` (tokens a cycle decided but
the emit machinery has not delivered yet), `from_cycle` (how many of them
reached all_ids, which is how far mtp.Speculator.finish rewinds when a stop
string or max_tokens lands mid-cycle), the prefill call, which needs the
trunk's HIDDEN state and not just its logits, and the ADAPTIVE BAIL switch —
when the speculator reports rolling acceptance below the floor, the loop
decodes serially, exactly as if `spec` had been None from that token on, for
the speculator's serial stretch (`spec.drafting` is the state it branches
on); then the speculator RE-ARMS (mtp.py) and the
loop hands it the token it just sampled as the next cycle's start. While a
re-arm is ahead the serial step goes through `spec.serial_step` — the same
M=1 forward, its hidden row kept for the head — and `DRINKME_MTP_REARM=0`
keeps the request serial to the end. A prefix slot keeps the head's own KV
beside the trunk's (mtp.HeadKV): cropped at the reuse point as the cache
is, handed to the speculator with the slot and taken back at the end, and
charged per slot at full width in the residency line. The cold tier does
not store it, so a slot restored from disk starts the head at the reuse
point. Everything else — the accept/commit
state machine, the three cache surfaces a rejection has to rewind, the
accept/resample rule — is in mtp.py and speculative.py. Unset = AUTO, which
speculates at depth 4 when the checkpoint carries an MTP head;
`DRINKME_SPEC=off` opts out to the serial loop above, byte for byte.

Concurrency: the HTTP layer serializes generations (one global lock), so one
generate() runs at a time; the persistent cache is single-flight by that lock,
and an abort mid-loop leaves slot.ids reflecting only what was truly written.
The slot switch is inside that same lock and is a pointer move — a slot's
tensors are never copied, swapped or shared — so no request can be answered
out of a half-swapped state: whichever slot generate() picked, it holds it
for the whole call.
"""

from __future__ import annotations

import dataclasses
import gc
import json
import os
import sys
import time
from typing import Iterator

import torch

from .engine import (Delta, Finished, GenerationRequest, GenEvent, GenResult, SleepError,
                     SleepState, StreamStart)
from .sampling import (PenaltyState, SampleScratch, StopScanner, sample_next,
                       sample_probs)
from . import (capability, control, ctx_checkpoints, cudagraph, cudagraph_fit, gen_config, metrics, mtp,
               prefill, sleep, slotstore)
from .. import exitcodes
from ..codec import identity
from .detok import IncrementalDetok, SuffixWindow
from .template import Prompt, effective_kwargs, render_prompt
from .tools import ToolCallScanner

# The context and revision helpers and the tokenizer loader live in
# serving/checkpoint.py (torch-free, shared with engine_mlx.py).
from .checkpoint import pack_revision, resolve_ctx, resolved_revision  # noqa: E402
from .checkpoint import tokenizer as _tokenizer  # noqa: E402,F401
from .kernel_route import route_kernels  # noqa: E402


def _ctx(cfg, ctx: int | None) -> int:
    """The allocated context window for a transformers config: the text
    config's native max_position_embeddings through checkpoint.resolve_ctx
    (which carries the clamp and its reasoning)."""
    tcfg = cfg.get_text_config() if hasattr(cfg, "get_text_config") else cfg
    return resolve_ctx(int(getattr(tcfg, "max_position_embeddings", 0) or 0), ctx)


def _eos_ids(tokenizer, model_generation_config, gd_eos_ids) -> frozenset:
    """EOS can live in three places depending on the family; union them.

    `model_generation_config.eos_token_id` is `model.generation_config` — real
    for the stock arm (from_pretrained reads the repo's generation_config.json
    into it), but MANUFACTURED for the compressed arm: load_compressed builds
    on the meta device from config.json alone (gen_config.py's module
    docstring), so gemma-4's copy there carries only config.json's [1, 106],
    missing the file's 50 (`<|tool_response>`, its end-of-calls token), and
    Qwen3-8B's copy is missing `<|endoftext|>` entirely. `gd_eos_ids`
    (GenDefaults.eos_ids, gen_config.load's own read of THAT file) takes over
    from `model_generation_config` the moment the file itself carried an
    eos_token_id: consulting BOTH would make the compressed arm's set
    `stock ∪ config.json.eos_token_id`, equal to stock only when
    config.json's ids happen to be a subset of the file's — not a structural
    guarantee, and config.json can carry an id (or lack one) the real file
    disagrees with. `model_generation_config` is used only as the fallback
    when gd_eos_ids is empty — which after gen_config.load's own config.json
    fallback means neither generation_config.json nor config.json declared an
    eos at all — uniformly on both arms rather than branched on `arm`, so the
    two arms end up with the same stop set by construction."""
    eos: set[int] = set()
    sources = [tokenizer.eos_token_id]
    sources.append(gd_eos_ids if gd_eos_ids else
                   getattr(model_generation_config, "eos_token_id", None))
    for src in sources:
        if isinstance(src, int):
            eos.add(src)
        elif isinstance(src, (list, tuple)):
            eos.update(int(e) for e in src)
    return frozenset(eos)


def _apply_rope_scaling(cfg, spec: dict | None) -> None:
    """spec (--rope-scaling / DRINKME_ROPE_SCALING, rope scaling; cli.parse_rope_scaling's
    shape, e.g. {"rope_type": "yarn", "factor": 4}; None = off, the default
    behaviour exactly) mutates cfg's text config IN PLACE: YaRN serves
    factor x the model's NATIVE max_position_embeddings, read here before
    anything else (this call or a caller) touches it.

    Must run BEFORE _ctx: _ctx clamps an explicit --ctx to
    max_position_embeddings, and this call is what raises that ceiling, so an
    over-native --ctx has to see the WIDENED window or it clamps against a
    number this feature was asked to move past.

    RoPE is entirely transformers' rotary module (module docstring — codec/
    never touches it): every field set here is read back by the rotary
    class's own __init__ (ROPE_INIT_FUNCTIONS[rope_type]) through
    config.rope_parameters (rope_scaling is a transformers-5 property alias
    for it), not interpreted by drinkme itself.

    MERGES into whatever rope_parameters already holds rather than replacing
    it outright — measured against Qwen2Config/Qwen3Config/LlamaConfig on
    the transformers pinned here (5.15.1): none of the three keep a
    top-level `rope_theta` field any more, only `rope_parameters`, and
    `standardize_rope_params` (called by ROPE_INIT_FUNCTIONS itself, right
    before it reads `rope_parameters["rope_theta"]`) falls back to
    `getattr(config, "rope_theta", None)` when that key is missing — a bare
    `tcfg.rope_scaling = {rope_type, factor, original_max_position_embeddings}`
    (this function's first draft, and the original rope-scaling spec) wipes the
    dict's own rope_theta entry, so YaRN construction reads back None and
    `_compute_yarn_parameters` dies on `None ** tensor`. Merging keeps
    rope_theta (and anything else already there) and overrides only the
    three YaRN fields.

    Never silent: a checkpoint that already carries a non-default,
    non-yarn rope_type gets its old value printed alongside the
    replacement (the everyday {'rope_type': 'default', ...} case is not a
    'replacement' worth a line every boot), and the new window is always
    announced — this changes what every position past the old native
    length actually attends to."""
    if spec is None:
        return
    tcfg = cfg.get_text_config() if hasattr(cfg, "get_text_config") else cfg
    native = int(getattr(tcfg, "max_position_embeddings", 0) or 0)
    factor = spec["factor"]
    # The vendor's recipe names the pre-extension window explicitly (Qwen3's
    # card: original 32768, factor 4 -> 131,072); a checkpoint's
    # max_position_embeddings can differ from it (Qwen3-8B says 40,960 =
    # 32k context + 8k generation). Scale from the recipe's number when the
    # spec carries one, from the native window otherwise.
    original = int(spec.get("original_max_position_embeddings") or native)
    existing = dict(getattr(tcfg, "rope_scaling", None) or {})
    existing_type = existing.get("rope_type")
    if existing_type not in (None, "default", "yarn"):
        print(f"[drinkme] rope scaling: replacing existing config {existing!r} "
              f"with yarn x{factor:g}", file=sys.stderr, flush=True)
    scaled = int(factor * original)
    existing.update({"rope_type": "yarn", "factor": factor,
                     "original_max_position_embeddings": original})
    tcfg.rope_scaling = existing
    tcfg.max_position_embeddings = scaled
    print(f"[drinkme] rope scaling: yarn x{factor:g} on original {original} — "
          f"window {native} -> {scaled} (opt-in; short-text quality may change)",
          file=sys.stderr, flush=True)


# ------------------------------------------------------------ prefix slots --

# One slot = the default engine. Anything else is an explicit ask
# (DRINKME_PREFIX_SLOTS), because more slots is more memory and the default
# must stay the allocation every release has made.
DEFAULT_SLOTS = 1
_GIB = 1024 ** 3
# Two short probe allocations, differenced layer by layer, give each layer's
# (fixed, per-row) cost exactly: the attention layers' keys/values scale with
# the rows they allocate, the DeltaNet recurrent and conv states do not.
# Short so the probe itself is never the allocation that hurts.
PROBE_SHORT, PROBE_LONG = 16, 48


def _human(nbytes: float) -> str:
    """GiB once it is GiB, MiB below that. The exact byte count is always
    printed beside this, so the only job here is to be readable."""
    return (f"{nbytes / _GIB:.2f} GiB" if nbytes >= _GIB
            else f"{nbytes / (1024 ** 2):.1f} MiB")


class _Slot:
    """One whole cache state at a prefix boundary: the StaticCache object
    (attention KV + DeltaNet recurrent state + conv state, all of it — the
    hybrid keeps both layer kinds inside the one cache), the width it was
    allocated at, the ids whose KV is actually written, when it was last
    used, its context checkpoints (serving/ctx_checkpoints.py; off unless
    the engine turns them on), and the MTP head's KV for those ids
    (mtp.HeadKV; None without a head, or when the slot holds none).
    Deliberately NOT a dataclass: slots are
    compared by IDENTITY (which slot object is this?), and a dataclass __eq__
    would make two empty slots equal and quietly answer "which one" wrong."""

    __slots__ = ("n", "cache", "alloc", "ids", "stamp", "ckpts", "head")

    def __init__(self, n: int = 0, ckpt_max: int = 0, ckpt_media: int = 0):
        self.n = n  # its index, for the log lines
        self.cache = None
        self.alloc = 0
        self.ids: list[int] = []
        self.stamp = 0  # LRU clock reading of the last request that used it
        self.ckpts = ctx_checkpoints.Checkpoints(ckpt_max, media_max=ckpt_media)
        self.head = None

    def clear(self) -> None:
        self.cache, self.alloc, self.ids, self.stamp, self.head = None, 0, [], 0, None
        self.ckpts.clear()


def slots_from_env(explicit: int | None = None) -> int:
    """`explicit` (CLI --prefix-slots, prefix slots) -> how many whole cache states to
    keep; else DRINKME_PREFIX_SLOTS; else DEFAULT_SLOTS = 1, the default engine
    exactly. 0 is the prefix cache OFF: a per-request cache, nothing kept
    between calls (what /health reports as `slots: 0`).

    PRECEDENCE (--ctx/DRINKME_CTX is the standing precedent in this file's
    own `_ctx`): an explicit CLI value WINS over the
    env var when both are set — the flag is the more specific ask, given at
    the exact invocation that will use it, where the env var may be an old
    export sitting in a shell profile. Unset CLI (None) falls through to the
    env var untouched, so every existing DRINKME_PREFIX_SLOTS-only caller
    (bench/prefix_slots_verify.py, the tests) is unaffected.

    Garbage or a negative count warns and falls back rather than dying: a typo
    must not take the server down, and must not silently change what is
    served either (mtp.depth_from_env, same posture). There is no upper
    clamp on purpose — the residency guard is the real ceiling, and it
    refuses with arithmetic instead of a made-up constant."""
    if explicit is not None:
        if explicit < 0:
            print(f"[drinkme] --prefix-slots {explicit} is not a slot count of "
                  f"0 or more — using {DEFAULT_SLOTS}", file=sys.stderr, flush=True)
            return DEFAULT_SLOTS
        return explicit
    raw = os.environ.get("DRINKME_PREFIX_SLOTS", "").strip()
    if not raw:
        return DEFAULT_SLOTS
    try:
        n = int(raw)
    except ValueError:
        n = -1
    if n < 0:
        print(f"[drinkme] DRINKME_PREFIX_SLOTS={raw!r} is not a slot count of "
              f"0 or more — using {DEFAULT_SLOTS}", file=sys.stderr, flush=True)
        return DEFAULT_SLOTS
    return n


def pick_slot(slots: list, ids: list[int], n_prompt: int,
              need: int, fit=None) -> tuple[_Slot, int]:
    """Which slot this prompt runs on, and how many of its tokens that slot
    can serve. Pure: every mutation belongs to the caller.

    A slot is ELIGIBLE when its allocation covers the whole request and it
    can serve a prefix of the prompt while leaving at least one token to
    compute fresh logits from: `Checkpoints.reach`. With context checkpoints
    off (--ctx-checkpoints 0, or a cache they cannot handle) that is the
    single-slot rule, unchanged: the prompt must extend the ENTIRE written
    cache (lcp == len(slot.ids)). With them on, a prompt that parts from the
    slot is served up to the furthest point a rewind by length or a
    checkpoint reaches at or below the common prefix
    (serving/ctx_checkpoints.py; `fit` keeps that point out of a
    bidirectional image run), under llama.cpp's slot rule: only when that
    is more than ctx_checkpoints.SIMILARITY of the prompt. And one drinkme
    rule: a slot that would keep less than ctx_checkpoints.KEEP of its
    tokens holds another conversation, and is not taken while an untouched
    slot is free — N slots exist to keep N conversations warm. Among
    eligible slots the one serving MOST of the prompt wins; on a tie, one
    the prompt extends (nothing is discarded), then the least recently used.

    With no eligible slot the LEAST RECENTLY USED slot comes back with 0
    for the caller to rebuild cold. Two consequences worth naming: a match
    never evicts anything (it returns before this line), and untouched slots
    (stamp 0) fill in index order before any live slot is taken, because
    min() keeps the first of a tie."""
    free = any(s.cache is None for s in slots)
    best, best_key = None, (0,)
    for slot in slots:
        if slot.cache is None or need > slot.alloc:
            continue
        lcp = 0
        for a, b in zip(ids, slot.ids):
            if a != b:
                break
            lcp += 1
        n = slot.ckpts.reach(lcp, len(slot.ids), n_prompt, fit)
        extends = n == len(slot.ids)
        if not extends and (n <= ctx_checkpoints.SIMILARITY * n_prompt
                            or free and n < ctx_checkpoints.KEEP * len(slot.ids)):
            continue
        key = (n, extends, -slot.stamp)
        if n > 0 and key > best_key:
            best, best_key = slot, key
    if best is not None:
        return best, best_key[0]
    return min(slots, key=lambda s: s.stamp), 0


def _common(a: list[int], b: list[int]) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def cache_bytes(cache) -> int:
    """Bytes a cache object is actually holding, summed over the REAL tensors
    on its layers.

    Walks whatever each layer carries — attributes, and the dicts the linear-
    attention layers keep their conv/recurrent states in — instead of naming
    fields: a transformers bump that renames or adds a surface then gets
    COUNTED rather than missed, which is the direction a memory guard has to
    fail in. Tensors are deduped by identity so an alias is not paid twice.

    On a FRESHLY CONSTRUCTED cache this counts only the layers' scalar
    counters — transformers-5 allocates the KV itself lazily, at the first
    update — which is why the engine measures a slot after driving a token
    through it and not at construction."""
    return sum(layer_bytes(cache))


def layer_bytes(cache) -> list[int]:
    """cache_bytes per layer, in layer order; a tensor two layers share is
    paid by the first."""
    seen: set[int] = set()
    out = []
    for layer in cache.layers:
        n = 0
        for value in vars(layer).values():
            for t in (value.values() if isinstance(value, dict) else (value,)):
                if isinstance(t, torch.Tensor) and id(t) not in seen:
                    seen.add(id(t))
                    n += t.numel() * t.element_size()
        out.append(n)
    return out


def _layer_rows(layer, width: int) -> int | None:
    """The rows a cache layer allocates in a cache `width` wide: its
    max_cache_len, which transformers' StaticSlidingWindowLayer sets to
    min(window, width). None for a layer that does not say (a
    linear-attention layer, whose state has no rows)."""
    rows = getattr(layer, "max_cache_len", None)
    return rows if isinstance(rows, int) else None


def measure_slot_layers(make_cache, prime, lo: int, hi: int) -> list[tuple[float, float, bool]]:
    """Per cache layer, (fixed bytes, bytes per row, by_rows): measured by
    building a cache `lo` and one `hi` wide (`make_cache`), driving a token
    through each (`prime`) so the lazy layers allocate, and differencing
    each layer against itself.

    A row is one the layer itself allocated (_layer_rows), so a sliding
    ring the probes catch below its window is charged per row of the ring
    and never per token of ctx: gemma-4-31B's 50 rings hold 1,024 rows at
    any ctx past the window. `by_rows` is False for a layer that grows
    without saying how many rows it holds; it is charged per token of
    width, the side a memory guard errs on. The probes are freed one at a
    time, so the peak is one short cache."""
    seen = []
    for width in (lo, hi):
        cache = make_cache(width)
        prime(cache)
        seen.append((layer_bytes(cache), [_layer_rows(l, width) for l in cache.layers]))
        del cache  # before the next probe: peak stays one probe, not two
    (b_lo, r_lo), (b_hi, r_hi) = seen
    out = []
    for blo, bhi, rlo, rhi in zip(b_lo, b_hi, r_lo, r_hi):
        if bhi == blo:
            out.append((float(blo), 0.0, True))
            continue
        by_rows = rlo is not None and rhi is not None and rhi != rlo
        if not by_rows:
            rlo, rhi = lo, hi
        delta, span = bhi - blo, rhi - rlo
        # exact when it divides (it does for every shape seen so far); a
        # float rather than a lie when some future layer rounds its width
        rate = delta // span if delta % span == 0 else delta / span
        out.append((blo - rate * rlo, rate, by_rows))
    return out


def slot_cost_at(layers: list, cache_at_ctx, ctx: int) -> tuple[float, float, float]:
    """measure_slot_layers' per-layer costs -> (fixed_bytes, ring_bytes,
    bytes_per_token) for one slot `ctx` wide. `cache_at_ctx` is a cache
    built `ctx` wide and never written (transformers 5 allocates at the
    first write), read only for the rows each layer takes at that width: a
    layer whose rows stop short of ctx is a sliding ring, charged at those
    rows; every other growing layer is charged per token of ctx."""
    rows = [_layer_rows(l, ctx) for l in cache_at_ctx.layers]
    if len(rows) != len(layers):
        rows = [None] * len(layers)
    fixed = ring = per_token = 0
    for (f, rate, by_rows), r in zip(layers, rows):
        fixed += f
        if rate and by_rows and r is not None and r < ctx:
            ring += rate * r
        elif rate:
            per_token += rate
    return fixed, ring, per_token


class HFEngine:
    """serving.engine.Engine over a transformers causal LM (either arm)."""

    # smallest persistent-KV allocation (module docstring: the allocation
    # bounds memory; decode attends over the live window). Class attr so
    # tests can shrink it to exercise regrowth on the toy model.
    KV_FLOOR = 4096

    def __init__(self, model, tokenizer, model_id: str, arm: str, meta: dict,
                 ctx: int, template_kwargs: dict | None = None,
                 mtp_head=None, prefix_slots: int | None = None,
                 gen_defaults: gen_config.GenDefaults | None = None,
                 vision=None, tower=None, vision_reason: str | None = None,
                 n_ctx_checkpoints: int | None = None):
        self.model = model
        self.tok = tokenizer
        self.model_id = model_id
        self.arm = arm
        self.meta = meta  # pack provenance (hfRepo/revision/meanBpw/weightedBpw) or {}
        self.ctx = ctx
        # generation_config.json's sampling overrides (gen_config.py),
        # and its recommended chat-template kwargs merged UNDER whatever this
        # constructor call was given explicitly — nobody passes template_kwargs
        # explicitly today, so in practice this is simply the checkpoint's own
        # recommendation, unless profiles.py overrides it per request later.
        gd = gen_defaults or gen_config.GenDefaults()
        # kept on the instance so anything that rebuilds a second HFEngine
        # over an already-loaded model (bench/prefix_slots_verify.py,
        # bench/slot_restore_verify.py) can thread the SAME GenDefaults
        # through rather than defaulting to an empty one and losing eos_ids
        # — one read of generation_config.json per real load,
        # never a second, silently different one per rebuild.
        self.gen_defaults = gd
        self.sampling_defaults = gd.sampling
        self.template_kwargs = {**gd.template_kwargs, **(template_kwargs or {})} or None
        print(f"[drinkme] sampling defaults from generation_config.json "
              f"({gd.source}): {self.sampling_defaults or '{}'}", flush=True)
        # serving/capability.py (the capability probe): the static
        # thinking/tool_format announcement, probed once here and cached —
        # never recomputed per request, never guessed from the model name.
        self.capability = capability.probe(tokenizer)
        print(f"[drinkme] capabilities: thinking={self.capability.thinking} "
              f"tool_format={self.capability.tool_format}", flush=True)
        served = capability.served_tool_formats()
        if served != capability.PARSEABLE_TOOL_FORMATS:
            print("[drinkme] DRINKME_TOOLS_UNTESTED is set: tools requests reach untested "
                  f"dialects too ({', '.join(sorted(served - capability.PARSEABLE_TOOL_FORMATS))}) "
                  "— the dialect smoke's door, not a default", flush=True)
        # serving/control.py: the row's markers that are SPECIAL ids in
        # this vocab, resolved once — the decode below keeps exactly these
        # as text so the scanners see them, and the row's call-close id ends
        # the turn. Empty for every row whose markers are plain text (Qwen),
        # and then nothing in generate() differs from before.
        self.control = control.resolve(tokenizer, self.capability.tool_format)
        if self.control.ids:
            print(f"[drinkme] control tokens: {self.control.describe()}", flush=True)
        # Image input (serving/vision.py, serving/image_prompt.py): `vision`
        # is the preprocessor and pixel cap the dialects prepare and count
        # images with, `_tower` how generate() runs the tower this tree
        # holds. Both or neither; `vision_reason` is the reason images are
        # refused when neither ("text-only model" unless the loader knows
        # better), which the dialects read for their named 400.
        if (vision is None) != (tower is None):
            raise ValueError("an engine takes image input with both a Vision and a Tower, "
                             "or neither")
        self.vision, self._tower = vision, tower
        self.vision_reason = None if vision is not None else (vision_reason or "text-only model")
        if tower is not None:
            from . import vision as _vision

            routed = _vision.bound(tower, model)
            _vision.route_patch_embed(tower, model)
            sources = "urls " + ("on" if vision.fetch_urls else "off") + \
                ", file:// " + (vision.media_path or "off")
            print(f"[drinkme] image input: {tower.architecture} tower at {tower.path}, pixel cap "
                  f"{vision.max_pixels:,}, sources: data, {sources}; attention "
                  + (f"bounded at {_vision.WORK_V:,} query-key pairs per dispatch "
                     f"({routed} layers)" if routed else
                     f"unbounded ({_vision.BOUNDED_ENV}=0)"), flush=True)
            # the tower's attention against fp32 on this device, before a
            # request can reach it; a kernel that fails refuses images here,
            # and a level-2 wake (which builds a fresh engine) checks again
            check = _vision.attention_self_test(tower, model)
            if check is not None:
                for line in check.lines():
                    print(f"[drinkme] {line}", flush=True)
                if not check.ok:
                    self.vision = self._tower = None
                    self.vision_reason = check.reason
                    print(f"[drinkme] image input: off — {self.vision_reason}", flush=True)
            if self.vision is not None and self.vision.video is not None:
                pre = self.vision.video.preprocessor
                print(f"[drinkme] video input: {pre.architecture}, {pre.fps:g} fps from the "
                      f"clip's own frame rate, at most {pre.max_frames} frames "
                      f"({pre.max_seconds:g} s), {pre.max_pixels:,} pixels per clip",
                      flush=True)
            elif self.vision is not None:
                print(f"[drinkme] video input: off — {self.vision.video_reason}", flush=True)
        elif self.vision_reason != "text-only model":
            print(f"[drinkme] image input: off — {self.vision_reason}", flush=True)
        # the tower's outputs in host RAM, by what the tower read
        # (serving/tower_cache.py): an image or video seen before skips the
        # tower in any conversation. None without a tower.
        self._tower_cache = None
        if self._tower is not None:
            from . import tower_cache

            dtype = next(self._tower.module(model).parameters()).dtype
            self._tower_cache = tower_cache.TowerCache(
                tower_cache.cap_from_env(),
                identity=(model_id, arm, self._tower.architecture, self._tower.path, str(dtype)))
            print(f"[drinkme] tower cache: {self._tower_cache.describe()}", flush=True)
        # The trained MTP draft head (serving/mtp.py), or None = the
        # one-token-at-a-time decode. Loaded once at engine build; per-request
        # gating (greedy and sampled both speculate) happens in
        # mtp.maybe_speculate.
        self.mtp_head = mtp_head
        # THE HOT LOOP's per-token savings.
        # One [V] scratch pair for the life of the engine (not two
        # full-vocabulary f32 tensors per candidate), and
        # one env read per ENGINE rather than per request. DRINKME_HOTLOOP_OFF=1
        # selects the unoptimized per-token path — full re-detok, list-rebuilt
        # penalties, fresh clones, a constraint that re-parses from byte 0 —
        # so the two can be measured against each other on one machine
        # (bench/hotloop_micro.py) and the equivalence tests can assert the
        # two streams are identical rather than merely both plausible.
        self._scratch = SampleScratch()
        self._hot_off = os.environ.get("DRINKME_HOTLOOP_OFF") == "1"
        # DRINKME_DETOK_VERIFY=1 re-decodes the whole output every token and
        # asserts the windowed decode matches it exactly. The equivalence
        # tests run whole generations under it; it is not for production.
        self._detok_verify = os.environ.get("DRINKME_DETOK_VERIFY") == "1"
        # DRINKME_MTP_DEPTH, parsed at most once per engine instead of once per
        # request (the hot-loop audit). Read at the first generation rather than
        # here: a server sets its environment before it builds an engine, but
        # mtp.py's own tests set it around a GENERATION, and this is a cache,
        # not a new contract. None = not read yet.
        self._mtp_depth = None
        # n-gram speculation: the rest of the same read — DRINKME_SPEC and the DRINKME_NGRAM_*
        # knobs, resolved into an mtp.SpecPlan on the same schedule and
        # dropped by the same seam (`_mtp_depth = None`, which is what
        # bench/mtp_gpu_acceptance.set_arm sets and what that file's
        # stale-instrument note is about).
        self._spec_plan = None
        # the last request's Speculator.stats(), None when it decoded serially:
        # what `drinkme bench`'s speculation pass reads its accepted drafts
        # per verify step from (arms.spec_pass)
        self.last_spec_stats = None
        # DRINKME_PREFILL_CHUNK (serving/prefill.py): tokens per prefill
        # forward, read once at load like the slot count; 0 = one forward
        self._prefill_chunk = prefill.chunk_from_env()
        prefill.announce(self._prefill_chunk)
        p = next(model.parameters())
        self.device, self.dtype = p.device, p.dtype
        # CUDA graphs for the serial decode step (serving/cudagraph.py):
        # decided once per engine and said once, as the slot cost is (the
        # line waits for the memory check below). A capture that fails later
        # turns it off for the engine's life.
        graphs = cudagraph.decide(model, self.device)
        self._graphs = graphs.on
        self._graph_bytes = 0
        self._scratch_kv, self._scratch_alloc = None, 0
        self.eos_ids = _eos_ids(tokenizer, getattr(model, "generation_config", None),
                                gd.eos_ids)
        # prefix cache (module docstring): N geometrically-grown StaticCaches,
        # one per slot; slot.ids = the ids whose KV is valid, in position order
        n_slots = slots_from_env(prefix_slots)
        self._reuse = n_slots > 0
        # context checkpoints (serving/ctx_checkpoints.py): how many each slot
        # keeps (--ctx-checkpoints / DRINKME_CTX_CHECKPOINTS, llama.cpp's 32),
        # and whether this model needs them. A fresh cache names its layer
        # kinds without allocating anything. `_ckpt_max` 0 is the extends-only
        # rule; `_ckpt_take` is False for a cache of full-attention layers
        # only, which rewinds by length to any shared prefix without one.
        probe = self._cache(PROBE_SHORT)
        self._ckpt_max, self._ckpt_off = 0, None
        if self._reuse:
            if not ctx_checkpoints.supported(probe):
                self._ckpt_off = "a cache layer they cannot handle: " + ", ".join(sorted(
                    {type(layer).__name__ for layer in probe.layers
                     if ctx_checkpoints.layer_kind(layer) is None}))
            else:
                self._ckpt_max = ctx_checkpoints.max_from_env(n_ctx_checkpoints)
                if self._ckpt_max == 0:
                    self._ckpt_off = "--ctx-checkpoints 0"
        self._ckpt_take = self._ckpt_max > 0 and ctx_checkpoints.needs_checkpoints(probe)
        del probe
        # and how many of them may sit at the end of an image or a video
        # (ctx_checkpoints.py, MEDIA ENDS): none for an engine that takes no
        # checkpoints or reads no images
        self._ckpt_media = (ctx_checkpoints.media_from_env()
                            if self._ckpt_take and self.vision is not None else 0)
        # off (0 slots) keeps one slot object
        self._slots = [_Slot(i, self._ckpt_max, self._ckpt_media)
                       for i in range(max(n_slots, 1))]
        self._slot_clock = 0  # monotonic; the LRU order is these stamps
        # what one slot costs, measured off the real cache tensors, and the
        # fit check that follows from it. Announced even at one slot: a
        # reader deserves to know what the KV allocation is costing them.
        self._slot_fixed_bytes: float | None = None
        self._slot_ring_bytes: float | None = None
        self._slot_token_bytes: float | None = None
        line = self._charge_graphs(graphs.line) if graphs.on else graphs.line
        if line:
            print(f"[drinkme] {line}", flush=True)
        self._announce_slots()
        # the cold tier (on-disk prefix slots), indexed and announced here so its count and
        # bytes land in the load block beside the slot cost — and NOT loaded:
        # preloading N slots' worth of accelerator memory at boot is the one
        # thing the live tier already refuses to do. Off with the prefix cache
        # itself (there is nothing to evict), and never fatal: a store that
        # cannot be opened leaves the engine serving exactly as it does now.
        # SLEEP / WAKE (sleep/wake, serving/sleep.py). `_sleep` is the state /health
        # reports and the two routes mutate; `_sleep_refs` is the ref list a
        # level-1 sleep parked and the wake reverses, held so wake moves back
        # EXACTLY what sleep moved rather than re-deriving it from a model
        # that is, by then, half on the host. `_slept_entries` are the cold-
        # tier entries this sleep wrote, so the wake restores the slots this
        # process had rather than whatever the store's LRU happens to hold.
        # `_reload` is set by the loaders below — the level-2 wake calls the
        # SAME load_stock/load_compressed the server booted through, which is
        # what makes "level 2 is the current stop" true instead of aspirational.
        self._sleep = SleepState()
        self._sleep_refs: list = []
        self._slept_entries: list = []
        self._reload = None
        self._cold = None
        if self._reuse:
            try:
                # keyed on the weights' resolved revision (meta, from the
                # loader) and the EFFECTIVE attention configuration — read
                # off model.config here, after --rope-scaling mutated it,
                # so both arms bind their slots the same way
                self._cold = slotstore.open_store(
                    model_id, arm, meta, ctx,
                    str(self.dtype).replace("torch.", ""),
                    attention=slotstore.attention_identity(model.config))
            except Exception as e:  # noqa: BLE001 — containment is the point
                print(f"[drinkme.slots] cold tier unavailable "
                      f"({type(e).__name__}: {e}) — serving without it",
                      file=sys.stderr, flush=True)

    def reset_prefix_cache(self) -> None:
        """Drop every slot: caches, the ids they claimed, the LRU order. The
        next request of any shape prefills cold. This is what an A/B harness
        wants between arms (bench/prefix_cache_verify.py, serve_timing_ab.py)
        — "start cold in your own terms" — and it is the only supported way
        to say it, since reaching in to null one attribute would leave the
        others describing a cache that is gone.

        This also DISABLES the cold tier for the rest of the
        process. "Cold in your own terms" cannot mean "and then handed
        yesterday's prefix off an SSD" — an A/B whose cold arm restores from
        disk is measuring nothing — and a bench arm's evictions have no
        business landing in a real server's store either."""
        for slot in self._slots:
            slot.clear()
        self._slot_clock = 0
        metrics.reset_slot_occupancy()
        if self._cold is not None:
            self._cold.disable()

    def prefix_cache_state(self) -> dict:
        """/health's `prefix_cache` field (docs/bench.md's served-ruler rule):
        0 slots means the prefix cache is off (--prefix-slots 0) — a per-request cache with
        nothing kept between calls; N>=1 is the resident slot count the reuse
        path keeps, so a served-throughput number can tell a shipped config
        from one that switched the cache off."""
        return {"slots": len(self._slots) if self._reuse else 0}

    def _measure_slot_cost(self) -> tuple[float, float, float] | None:
        """(fixed_bytes, ring_bytes, bytes_per_token) for ONE slot at this
        engine's ctx, measured — not modelled — per cache layer
        (measure_slot_layers).

        The per-layer difference isolates what scales with the rows a layer
        allocates (the attention layers' keys and values) from what does
        not (the DeltaNet recurrent state and the conv state, which have no
        sequence dimension at all — the same fact that makes a checkpoint
        the only way back into them shows up here as a constant). A layer
        whose rows at this ctx stop short of it is a sliding ring, charged
        at those rows (`ring_bytes`); a layer that spans the ctx is charged
        per token of it. Family-agnostic: it asks the cache what it
        allocated, and a cache built `ctx` wide how many rows each layer
        takes (slot_cost_at).

        None when the ctx is too small to difference (a toy ctx), which is
        a measurement failure, not a serving one.

        The per-layer costs are MEMOIZED ON THE MODEL, because they are a
        property of the model and not of the engine wrapped around it:
        serving builds one engine, but the benches and the tests build
        several over one loaded model (bench/prefix_slots_verify.py needs a
        1-slot and a 3-slot engine side by side), and two forwards apiece is
        a real cost on a hybrid. The rows at ctx are read per engine."""
        layers = getattr(self.model, "_drinkme_slot_layers", None)
        if layers is None:
            lo, hi = PROBE_SHORT, PROBE_LONG
            if self.ctx < hi:
                lo, hi = max(1, self.ctx // 4), self.ctx
            if hi <= lo:
                return None
            layers = measure_slot_layers(self._cache, self._prime_cache, lo, hi)
            try:
                self.model._drinkme_slot_layers = layers
            except AttributeError:  # a model that refuses attributes: measure again
                pass
        return slot_cost_at(layers, self._cache(self.ctx), self.ctx)

    def _charge_graphs(self, line: str) -> str:
        """What this engine's graphs will hold (serving/cudagraph_fit.py:
        per cache, one per prefix slot or the one scratch cache, at the
        widest verify step the speculation plan replays), judged the way the
        MTP head is (mtp.residency_check, against a live mem_get_info): graph
        mode that would leave less than the fit check's margin decodes eager
        instead, and says so. Otherwise the bytes join the slots' residency
        charge (_announce_slots). Returns the line to print."""
        cfg = self.model.config
        rows = 1
        if cudagraph.verified(self.model, "verify"):
            plan = mtp.spec_plan(self.mtp_head is not None, mtp.depth_from_env())
            rows = 1 if plan.mode == "off" else plan.depth + 1
        tcfg = cfg.get_text_config() if hasattr(cfg, "get_text_config") else cfg
        caches = len(self._slots) if self._reuse else 1
        per = cudagraph_fit.cache_bytes(tcfg.to_dict(), rows)
        total = caches * per
        headroom = mtp.device_headroom_gib(self.device)
        if headroom is not None:
            free_gib, used_gib = headroom
            yields, left, margin = mtp.residency_check(free_gib, used_gib, total / _GIB)
            if yields:
                self._graphs = False
                return (f"decode step: eager; CUDA graphs would hold {total / _GIB:.2f} GiB "
                        f"({caches} cache{'' if caches == 1 else 's'} x {_human(per)}), leaving "
                        f"{left:.2f} GiB free, under the {margin:.2f} GiB of headroom drinkme "
                        f"requires (10% of the {used_gib:.2f} GiB already resident)")
        self._graph_bytes = total
        return (f"{line}; they hold {_human(per)} per cache (verify up to {rows} row"
                f"{'' if rows == 1 else 's'})")

    def _announce_slots(self) -> None:
        """Say what the prefix cache costs, then let the fit check judge it.

        Never fatal on the MEASUREMENT: a probe that cannot run leaves the
        server serving unchanged, minus a number (the SDPA
        announce next door takes the same posture, for the same reason). The
        JUDGEMENT can be fatal, and only for an explicit multi-slot ask.

        With the prefix cache off (--prefix-slots 0) there are no
        slots to hold: every request allocates its own cache and frees it, so
        the cost is ONE cache at a time, and charging the guard for none
        would admit a config that cannot allocate it."""
        n = 1 if not self._reuse else len(self._slots)
        off = "" if self._reuse else " (prefix cache OFF — per request)"
        try:
            cost = self._measure_slot_cost()
        except Exception as e:  # noqa: BLE001 — containment is the point
            print(f"[drinkme] prefix cache: slot size not measurable "
                  f"({type(e).__name__}: {e}) — serving {n} slot(s) unguarded",
                  file=sys.stderr, flush=True)
            self._announce_checkpoints()
            return
        if cost is None:
            self._announce_checkpoints()
            return
        self._slot_fixed_bytes, self._slot_ring_bytes, self._slot_token_bytes = cost
        fixed, ring, per_token = cost
        # a slot keeps the MTP head's KV too (mtp.HeadKV), up to one entry
        # per position, so it is charged per token at full width beside the
        # trunk's measured figure
        head = self._head_token_bytes()
        per_slot = fixed + ring + (per_token + head) * self.ctx
        rings = f" + {ring:,.0f} B sliding-window rings" if ring else ""
        heads = f", the MTP head's KV {head:,} B/token of it" if head else ""
        print(f"[drinkme] prefix cache: {n} slot{'' if n == 1 else 's'} x "
              f"{per_slot:,.0f} B ({fixed:,.0f} B state{rings} + {per_token + head:,.0f} B/token "
              f"x ctx {self.ctx}{heads}) = {_human(n * per_slot)} at full width{off}",
              flush=True)
        self._check_slot_residency(n * (per_slot + self._announce_checkpoints())
                                   + self._graph_bytes, n)

    def _head_token_bytes(self) -> int:
        """Bytes per position of the MTP head's KV: keys and values for its
        one attention layer, num_key_value_heads x head_dim each, at the
        model's dtype, read off the head's config (mtp._head_config).
        Qwen3.8-27B: 2 x 4 x 256 x 2 B = 4,096 B/token. 0 without a head."""
        if self.mtp_head is None:
            return 0
        cfg = self.mtp_head.cfg
        dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
        return 2 * cfg.num_key_value_heads * dim * self.dtype.itemsize

    def checkpoint_bytes(self) -> int:
        """One full context checkpoint of this model, in bytes: its sliding
        rings at the whole window and its linear-attention states
        (ctx_checkpoints.config_bytes, which the tests hold equal to a real
        snapshot). 0 when it needs none."""
        cfg = self.model.config
        tcfg = cfg.get_text_config() if hasattr(cfg, "get_text_config") else cfg
        return ctx_checkpoints.config_bytes(tcfg.to_dict(), self.dtype.itemsize, self.ctx)

    def _new_checkpoints(self):
        """An empty checkpoint list under this engine's rules: its
        --ctx-checkpoints count and its media-end bound."""
        return ctx_checkpoints.Checkpoints(self._ckpt_max, media_max=self._ckpt_media)

    def _announce_checkpoints(self) -> int:
        """Say what the context checkpoints do and cost; return the most one
        slot's checkpoints can hold, in bytes, for the residency check. They
        live on the device beside their slot, so they are charged like it,
        the media-end ones (ctx_checkpoints.py, MEDIA ENDS) included."""
        if not self._reuse:
            return 0
        if self._ckpt_off is not None:
            print(f"[drinkme] context checkpoints: off ({self._ckpt_off}) — a slot is "
                  "reused only when a prompt extends all of it", flush=True)
            return 0
        if not self._ckpt_take:
            print("[drinkme] context checkpoints: none needed (full attention "
                  "rewinds by length) — a slot is reused up to any shared prefix",
                  flush=True)
            return 0
        held = ctx_checkpoints.max_held(self._ckpt_max, self.ctx, media=self._ckpt_media)
        one = self.checkpoint_bytes()
        media = (f", {self._ckpt_media} at media ends ({ctx_checkpoints.MEDIA_ENV})"
                 if self._ckpt_media else "")
        print(f"[drinkme] context checkpoints: up to {held} per slot (--ctx-checkpoints "
              f"{self._ckpt_max}, {ctx_checkpoints.MIN_STEP} apart at ctx {self.ctx}{media}) x "
              f"{one:,} B = {_human(held * one)} at most per slot", flush=True)
        return held * one

    def _check_slot_residency(self, total_bytes: float, n: int) -> None:
        """FIT OUTRANKS SPEED, the prefix cache's turn — the same decided
        guard serving/mtp.py applies to the draft head, applied to the slots.
        N slots grow geometrically to at most ctx, so ctx-width is what they
        must be charged at; anything less is a guard that passes at boot and
        OOMs at hour three.

        The margin is suggest.FIT_HEADROOM, reached through
        mtp.residency_check so this file does not invent a second number for
        the same question, against a live mem_get_info. Off a real
        accelerator (cpu, mps, a monkeypatched fixture) there is nothing to
        measure and nothing to guard.

        An explicit DRINKME_PREFIX_SLOTS above the default that does not fit
        is REFUSED, with the arithmetic printed: that is the ask the person
        made, and serving it anyway trades a line at boot for an OOM in the
        middle of someone's conversation. The DEFAULT allocation only WARNS —
        it is what every release has made, and a new guard
        that refuses a config which served yesterday is a regression, not a
        safeguard."""
        headroom = mtp.device_headroom_gib(self.device)
        if headroom is None:
            return
        free_gib, used_gib = headroom
        yields, left, margin = mtp.residency_check(free_gib, used_gib,
                                                   total_bytes / _GIB)
        if not yields:
            return
        line = (f"{n} prefix slot{'' if n == 1 else 's'} at ctx {self.ctx}"
                + (" and their context checkpoints" if self._ckpt_take else "")
                + (" and CUDA graphs" if self._graph_bytes else "") + " want "
                f"{total_bytes / _GIB:.2f} GiB, which would leave {left:.2f} GiB "
                f"free — under the {margin:.2f} GiB of headroom drinkme "
                f"requires (10% of the {used_gib:.2f} GiB already resident)")
        ckpt = " or --ctx-checkpoints" if self._ckpt_take else ""
        if n <= DEFAULT_SLOTS:
            print(f"drinkme: warning: {line}. This is the single slot every "
                  f"release allocates, so serving anyway — lower --ctx{ckpt} if a "
                  "long request OOMs.", file=sys.stderr, flush=True)
            return
        raise exitcodes.CantRunHere(
            f"drinkme: refusing to serve: {line}. Lower DRINKME_PREFIX_SLOTS "
            f"(default {DEFAULT_SLOTS})" + (f", --ctx{ckpt}." if ckpt else " or --ctx."))

    # ---------------------------------------------------- the cold tier --

    def _prime_cache(self, cache) -> None:
        """Drive one token through a fresh cache so transformers-5's lazy
        layers materialize.

        A freshly constructed cache honestly holds only its counters
        (cache_bytes' docstring), and a saved state has to be checked against
        a cache that has ACTUALLY allocated — otherwise the check is against
        an empty description and passes anything. Every tensor this token
        writes is overwritten by the restore, `cumulative_length` included.
        logits_to_keep=1 for the launch-blocker's reason: one row, never
        [n, vocab]."""
        inp = torch.tensor([[0]], dtype=torch.long, device=self.device)
        with torch.inference_mode():
            self.model(inp, past_key_values=cache, use_cache=True,
                       cache_position=torch.arange(1, device=self.device),
                       logits_to_keep=1)

    def _take_from_cold(self, slot: _Slot, ids: list[int], need: int,
                        where: str, n_prompt: int, fit=None, floor: int = 0) -> int:
        """Evict the slot the picker chose to the cold tier, then try to
        refill it from there for THIS prompt. Returns how many prompt tokens
        the restored slot can serve — 0 means nothing usable was found and
        the caller prefills cold, which is the current behaviour exactly.
        `floor` is what the slot as it stands serves (a reuse that keeps
        less than half of it): only a stored slot serving more replaces it,
        and otherwise the slot is left as it was, saved.

        The eviction comes first because this slot's tensors are about to be
        overwritten either way: a prefix that cost 22k tokens to build should
        not be dropped on the floor merely because the next request happens
        to have one waiting on disk. Its context checkpoints go with it, and
        come back with a restored slot (serving/ctx_checkpoints.py): with
        them on, a stored slot the prompt parts from is served up to what
        its checkpoints reach, as a live one is.

        The ids the file carries are re-checked against the prompt HERE, not
        in the store: the store's chain walk is an index, and an index is
        allowed to be a filter, but the decision to reuse KV computed for one
        token sequence as if it were another belongs where the reuse rule
        lives. That check is also the backstop no hash has: a collision
        costs a log line, not a wrong answer.

        Never fatal. Every failure inside costs the prefill it would have
        saved."""
        cold = self._cold
        partial = self._ckpt_max > 0
        try:
            if slot.cache is not None and slot.ids:
                cold.put(slot.ids, slot.alloc, slot.cache, slot.ckpts)
            entry = cold.lookup(ids, need, self.ctx, partial=partial, fit=fit, floor=floor)
            if entry is None:
                if not floor:
                    cold.misses += 1
                return floor
            got = cold.restore(entry, self._cache, self._prime_cache)
            if got is not None and partial:
                cks = cold.restore_checkpoints(entry, got[0], self.device)
        except Exception as e:  # noqa: BLE001 — containment is the point
            print(f"[drinkme.engine] cold tier{where} failed "
                  f"({type(e).__name__}: {e}); "
                  + ("the live slot serves this turn" if floor else "full prefill this turn"),
                  file=sys.stderr)
            return floor
        if got is None:
            return floor
        cache, held = got
        ckpts = self._new_checkpoints()
        if partial:
            ckpts.items = cks
            ckpts.free = ctx_checkpoints.rewinds_freely(cache)
        n = ckpts.reach(_common(ids, held), len(held), n_prompt, fit)
        if n <= floor:
            print(f"[drinkme.engine] cold tier{where}: restored ids serve {n} tokens "
                  f"of this prompt, not more than {floor} — ignoring", file=sys.stderr)
            if n == 0:
                cold.forget_entry(entry)
            return floor
        slot.cache, slot.alloc, slot.ids, slot.ckpts = cache, entry.alloc, list(held), ckpts
        # the cold tier stores no MTP head KV: the head starts at the reuse point
        slot.head = None
        print(f"[drinkme.engine] prefix cache{where}: restored {len(held)} "
              f"tokens from the cold tier (alloc {entry.alloc}"
              + (f", {len(ckpts)} context checkpoint(s)" if len(ckpts) else "") + ")",
              file=sys.stderr)
        return n

    def _reach(self, slot: _Slot, ckpts, key_ids: list[int], n: int, where: str) -> None:
        """Bring `slot`'s cache back to position n, below everything it holds,
        which the picker (or the cold tier) found this prompt can reuse: by
        length when the cache reaches n that way (`ckpts.free`), else from
        the context checkpoint at n (serving/ctx_checkpoints.py). The
        checkpoints past n go, since the tokens after n are about to be
        overwritten."""
        common = _common(key_ids, slot.ids)
        if ckpts.free:
            ctx_checkpoints.rewind(slot.cache, n)
            how = "rewound by length"
        else:
            ck = ckpts.at(n)
            if ck is None:
                raise RuntimeError(f"no context checkpoint at {n} (held: {ckpts.positions()})")
            ctx_checkpoints.restore(slot.cache, ck)
            how = f"restored the context checkpoint ({_human(ck.nbytes)})"
        ckpts.drop_beyond(n)
        metrics.record_slot_event("restore")
        print(f"[drinkme.engine] prefix cache{where}: the prompt parts from the "
              f"slot's {len(slot.ids)} tokens at {common}; {how} to {n}",
              file=sys.stderr)

    def persist_slots(self, timeout: float = 300.0) -> int:
        """Write every live slot to the cold tier and WAIT for the writes.
        Returns how many COMMITTED — reached disk, header and all — not how
        many were queued. serve.py calls this on SIGTERM
        and on the Ctrl-C path; `timeout` is what the unit's TimeoutStopSec
        has to cover.

        A slot mid-generation is SKIPPED, and correctly: engines empties
        slot.ids on entry and restores it on exit, so a slot another thread
        is writing KV into presents as having no valid ids — the same
        invariant that makes an exception mid-generate leave an invalid cache
        rather than a lying one."""
        return len(self._write_live_slots(timeout)[0])

    def _write_live_slots(self, timeout: float) -> tuple[list, int]:
        """([(slot index, ids, entry)] for every slot whose write COMMITTED,
        how many live slots did NOT reach disk).

        ONE path for SIGTERM (persist_slots above) and sleep (_park_slots
        below), because "sleep must persist live slots the same way SIGTERM
        does" is a correctness claim, not a convenience: two writers of the
        same cold tier would be two chances to disagree about what a stored
        slot is. Sleep is the caller that needs the ENTRIES back — it restores
        exactly the slots this process had, not whatever the store's LRU is
        holding by the time it wakes — so this is the shape both use and
        persist_slots just counts.

        Counted AFTER the flush, by entry state, not by what was queued:
        a queued write that failed — or is still in the
        queue when the timeout runs out — is not a persisted slot, sleep's
        `slots_persisted` must not say it is, and the wake must not go
        looking for its file."""
        live = [s for s in self._slots if s.cache is not None and s.ids]
        if self._cold is None:
            return [], len(live)
        queued = []
        for slot in live:
            entry = self._cold.put(slot.ids, slot.alloc, slot.cache, slot.ckpts)
            if entry is not None:
                queued.append((slot.n, list(slot.ids), entry))
        self._cold.flush(timeout)
        stored = [(n, ids, e) for n, ids, e in queued if e.committed]
        still = sum(1 for _, _, e in queued if e.pending)
        lost = len(queued) - len(stored) - still
        if still or lost:
            print(f"[drinkme.slots] {len(stored)} of {len(queued)} queued slot "
                  f"write(s) committed ({still} still queued after "
                  f"{timeout:.0f}s, {lost} failed or evicted) — the rest will "
                  "prefill cold", file=sys.stderr, flush=True)
        return stored, len(live) - len(stored)

    # ------------------------------------------------------ sleep / wake --

    def sleep_state(self) -> dict:
        """What /health reports. Lockless by construction (engine.SleepState's
        docstring): scalars written under the generation lock, read here."""
        return self._sleep.as_dict()

    def _park_slots(self) -> tuple[int, int]:
        """(persisted, dropped) — the prefix slots on the way down.

        vLLM level 1 "discards the KV cache"; drinkme discards it to the cold
        tier first when there is one (on-disk prefix slots), so a wake gets the 22k-token
        prefix back instead of re-prefilling it. With no cold tier the slots
        are DROPPED and the count is said out loud rather than left for
        someone to infer from a slow first request after the wake.

        The MTP head's own KV goes too: each slot's (mtp.HeadKV, which the
        cold tier does not store, so a restored slot's head starts at the
        reuse point) and whatever a request that failed mid-generation left
        on the head, real device memory that nothing across a sleep needs
        (mtp.Speculator.open installs a slot's history or a fresh cache on
        every request)."""
        stored, dropped = self._write_live_slots(300.0)
        self._slept_entries = stored
        for slot in self._slots:
            slot.clear()
        # graph mode's scratch cache (and the graphs bound to it) goes too:
        # a graph replays the device addresses it was captured against
        self._scratch_kv, self._scratch_alloc = None, 0
        self._slot_clock = 0
        metrics.reset_slot_occupancy()
        if self.mtp_head is not None:
            self.mtp_head.cache = None
            self.mtp_head.entries = 0
        return len(stored), dropped

    def sleep(self, level: int = 1) -> dict:
        """Park this model and give the device back. Returns sleep_state().

        Level 1 moves every device tensor to host RAM (serving/sleep.py's
        ordered pass) and drops the KV; level 2 additionally frees the host
        copies, which is the current `systemctl stop` in every respect except
        that the process, its port, its tokenizer and its cold-tier index
        stay up. Both are idempotent, and level 2 may be asked for while
        already at level 1 — sleep only ever goes DEEPER, because "wake" is
        the one word for coming back and a 2 -> 1 transition would have to
        reload the pack to mean anything.

        The CALLER owns the exclusion: serving/http.py takes the generation
        lock (or answers 409), the same lock generate() runs under, so this
        never races a live forward pass. Nothing here reaches for that lock
        itself — an engine that could put itself to sleep from under a
        generation would be a worse bug than the one this guards."""
        if level not in (1, 2):
            raise SleepError(f"sleep level {level} is not 1 or 2")
        st = self._sleep
        if st.level == level:
            # idempotent, and at level 2 not quite a no-op: the repeat is the
            # one call a "why is the device still full?" operator makes, so
            # it hands the allocator's cached blocks back again (otherwise a
            # first level-2 sleep that left a model's worth reserved would
            # see the repeat take this return without releasing).
            if level == 2:
                sleep.release(self.device)
            return st.as_dict()  # already exactly there
        if st.asleep and level < st.level:
            raise SleepError(
                f"already asleep at level {st.level}; sleep only deepens — "
                "POST /wake_up first")
        t0 = time.perf_counter()
        if not st.asleep:
            st.slots_persisted, st.slots_dropped = self._park_slots()
        refs = self._sleep_refs or sleep.walk(self.model, self.mtp_head,
                                              self.device)
        # measured BEFORE the move, so a level-2 sleep — which frees rather
        # than copies — can still report what it gave back
        st.tensors, st.bytes_moved, st.by_class = sleep.inventory(refs)
        if level == 1:
            sleep.move(refs, "cpu", pin=sleep.pin_from_env())
            self._sleep_refs = refs
        else:
            # the whole model goes, host copies included. gc first: the
            # skeleton holds reference cycles (nn.Module parent/child), and
            # empty_cache() before they are collected reports memory the
            # allocator has not actually been handed back yet.
            #
            # `refs` goes FIRST: each _Ref owns the
            # Parameter / dict / object it would write back into, so the
            # local inventory was keeping every device tensor alive across
            # gc.collect() + empty_cache() below — allocated read 0 after
            # this returned, reserved read the whole model (128 MiB of a
            # 128 MiB toy; 5.92 GiB of the 4B), and the memory was only
            # handed back by the NEXT empty_cache somebody happened to call.
            # A level-2 sleep is the "give the device to the other model"
            # move, so the blocks have to be released here, not eventually.
            self._sleep_refs = []
            self.model = None
            self.mtp_head = None
            # and the tower's outputs it kept in host RAM
            # (serving/tower_cache.py); the wake's fresh engine brings its own
            if self._tower_cache is not None:
                self._tower_cache.clear()
            del refs
            gc.collect()
        sleep.release(self.device)
        st.level, st.since = level, time.time()
        st.took_s = time.perf_counter() - t0
        where = "parked in host RAM" if level == 1 else "freed"
        print(f"[drinkme.sleep] level {level}: {st.tensors} tensors, "
              f"{sleep.human(st.bytes_moved)} ({st.bytes_moved:,} B) {where} "
              f"in {st.took_s:.1f}s; slots {st.slots_persisted} persisted, "
              f"{st.slots_dropped} dropped"
              + (" (no cold tier)" if self._cold is None else ""), flush=True)
        return st.as_dict()

    def wake(self) -> dict:
        """Reverse the sleep and say how long it took. Returns sleep_state().

        Level 1 walks the SAME ref list back to the device — same tensors,
        same order, and no second opinion about what the model consists of.
        Level 2 calls `self._reload`, which is the loaders' own
        load_stock/load_compressed closure: the wake IS the normal load path,
        so a measurement of it is a measurement of boot, and there is no
        second code path to keep in sync with the first.

        Then the model is PRIMED — one token through a throwaway cache,
        exactly what on-disk prefix slots' _prime_cache does before a restore — and the slots
        this sleep persisted are restored. Priming is not decoration: it is
        the first forward after the move, and having it happen here means a
        wake that half-landed fails at the wake, in the wake's own log line,
        instead of inside some stranger's request."""
        st = self._sleep
        if not st.asleep:
            return st.as_dict()
        t0 = time.perf_counter()
        if st.level == 1:
            sleep.move(self._sleep_refs, str(self.device))
            self._sleep_refs = []
        else:
            if self._reload is None:
                raise SleepError(
                    "level-2 wake needs the loader that built this engine; "
                    "this engine was constructed directly, not through "
                    "load_stock/load_compressed")
            fresh = self._reload()
            self.model, self.mtp_head, self.tok = (fresh.model, fresh.mtp_head,
                                                   fresh.tok)
            self.device, self.dtype = fresh.device, fresh.dtype
            self.vision, self._tower, self.vision_reason = (
                fresh.vision, fresh._tower, fresh.vision_reason)
            self._tower_cache = fresh._tower_cache
        self._wake_prime()
        st.slots_restored = self._restore_slots()
        st.level, st.since = 0, None
        st.wake_s = time.perf_counter() - t0
        print(f"[drinkme.sleep] wake in {st.wake_s:.1f}s "
              f"({sleep.human(st.bytes_moved)} back on {self.device}); "
              f"{st.slots_restored} prefix slot(s) restored", flush=True)
        return st.as_dict()

    def _wake_prime(self) -> None:
        """One token through a short throwaway cache: transformers-5's lazy
        layers materialize, the triton signatures JIT, and every tensor the
        move touched is read by a real forward before a user's request is.
        PROBE_SHORT wide, not KV_FLOOR — this is a liveness check, not an
        allocation, and _restore_slots below builds the real widths."""
        cache = self._cache(PROBE_SHORT)
        try:
            self._prime_cache(cache)
        finally:
            del cache

    def _restore_slots(self) -> int:
        """Put back the slots this sleep persisted, by ENTRY rather than by
        lookup: the store's lookup() answers "what do you hold for this
        prompt", and a wake has no prompt — it has the slots it just wrote.
        Going through restore() directly keeps the verification (layout
        check against a live primed cache, the written-token counter, the
        ids) exactly as on-disk prefix slots wrote it.

        Never fatal, and never a lie: a slot whose ids come back different
        from what went in is dropped rather than installed. The cost of every
        failure here is the prefill it would have saved."""
        entries, self._slept_entries = self._slept_entries, []
        if self._cold is None or not entries:
            return 0
        n = 0
        for idx, ids, entry in entries:
            try:
                got = self._cold.restore(entry, self._cache, self._prime_cache)
            except Exception as e:  # noqa: BLE001 — containment is the point
                print(f"[drinkme.sleep] slot {idx} restore failed "
                      f"({type(e).__name__}: {e}); it will prefill cold",
                      file=sys.stderr)
                continue
            if got is None:
                continue
            cache, held = got
            if held != ids:
                print(f"[drinkme.sleep] slot {idx}: restored {len(held)} ids "
                      f"are not the {len(ids)} that were parked — dropping",
                      file=sys.stderr)
                continue
            slot = self._slots[idx]
            slot.cache, slot.alloc, slot.ids = cache, entry.alloc, list(held)
            # the slot's context checkpoints were parked with it
            slot.ckpts = self._new_checkpoints()
            if self._ckpt_max > 0:
                slot.ckpts.items = self._cold.restore_checkpoints(entry, cache, self.device)
                slot.ckpts.free = ctx_checkpoints.rewinds_freely(cache)
            self._slot_clock += 1
            slot.stamp = self._slot_clock
            metrics.slot_occupied()
            n += 1
        return n

    def model_meta(self) -> dict:
        """The `drinkme` object of this model's /v1/models entry (http.py adds
        OpenAI's four top-level fields around it). Every key here is
        drinkme's own; the compression profile and a sampling profile never
        share a name (`compressionProfile` vs `sampling.profile`)."""
        return {"arm": self.arm,
                "runtime": "torch",  # the engine's runtime: torch here, mlx on engine_mlx
                "hfRepo": self.meta.get("hfRepo"),
                "revision": self.meta.get("revision"),
                # the pack's compression profile (sip / gulp), the pack's own
                # name; None for the stock arm and an engine built with no meta
                "compressionProfile": self.meta.get("profile"),
                # the lexicon's two bits-per-weight figures, by the lexicon's
                # names (docs/pack-format.md): bitsPerWeight is meta.json's
                # weightedBpw (total encoded payload bits over the packed
                # population's weights), meanTensorBitsPerWeight is meanBpw
                # (the unweighted mean over tensors); None when meta.json
                # has no such key, and on the stock arm
                "bitsPerWeight": self.meta.get("weightedBpw"),
                "meanTensorBitsPerWeight": self.meta.get("meanBpw"),
                # the released precision served exactly (bf16; the codec
                # serves a release as released); None only for an engine
                # built with no meta
                "sourceDtype": self.meta.get("sourceDtype"),
                # the window this server ACTUALLY allocates (StaticCache bound)
                # — clients size their compaction gauges off this; advertising
                # anything but the truth misfires them (pi's, for one).
                "contextWindow": self.ctx,
                # the device the weights actually sit on — a CPU fallback can
                # hide behind GPU-labeled numbers, so the wire says it
                "device": str(self.device),
                # serving/capability.py (the capability probe): the static
                # off-menu-honesty announcement, probed once at construction
                "capabilities": self.capability.as_dict(),
                # generation_config defaults: the FULL effective sampling table
                # (generation_config.json overrides beating SampleParams' own
                # OpenAI defaults) — what a bare request (no sampling fields,
                # no profiles overlay) samples with. `profile` is the
                # generation profile of a `<model>:<profile>` entry, filled by
                # http.py; None on the bare entry.
                "sampling": {"profile": None,
                             "defaults": gen_config.effective_defaults(self.sampling_defaults)}}

    def _render(self, req: GenerationRequest, upto: int | None = None) -> Prompt:
        """THE render for one request — generate() and count_tokens() both
        come here, so they can never disagree about one prompt: the
        request's template kwargs over this engine's own
        (template.effective_kwargs, thinking off under a constraint), the
        tools, the family's own chat template. `upto` renders only the
        messages before that index, with no generation prompt: the history
        a message starts after (_last_user_start)."""
        return render_prompt(
            self.tok, req.messages if upto is None else req.messages[:upto],
            add_generation_prompt=upto is None,
            template_kwargs=effective_kwargs(self.template_kwargs, req.template_kwargs,
                                             constrained=req.sampling.output_schema is not None),
            tools=req.tools)

    def _last_user_start(self, req: GenerationRequest, rendered: list[int],
                         image) -> int | None:
        """Where the request's last user message starts in its prompt, for
        the context checkpoint llama.cpp takes there (serving/
        ctx_checkpoints.py): the length of the history before it, rendered
        by the same template — when that is a prefix of the whole render.
        A template that renders earlier turns differently once a later one
        follows is not, and then there is no such checkpoint. None too for
        a first message, which starts where the prompt does."""
        k = max((i for i, m in enumerate(req.messages) if m.get("role") == "user"),
                default=0)
        if k == 0:
            return None
        try:
            head = self._render(req, upto=k).ids
        except Exception:  # noqa: BLE001 — a template that cannot end there
            return None
        if not head or rendered[:len(head)] != head:
            return None
        return len(head) if image is None else image.expanded(len(head))

    def count_tokens(self, req: GenerationRequest) -> int:
        """The Engine contract's count: the request rendered exactly as
        generate() renders it (_render), and the ids counted. Rendering
        and tokenizing read only the tokenizer; no weights, no cache, no
        device — lockless by design. Each image's one rendered placeholder
        counts as what it expands to (image_prompt.ImagePrompt: its `tokens`,
        and gemma-4's two markers around them; vision.Vision.prompt_tokens),
        so an ImagePlan (vision.Vision.count) counts the same as the image.
        A video's three rendered ids count as its timestamps, markers and
        runs (video.PreparedVideo.prompt_tokens)."""
        n = len(self._render(req).ids)
        if not req.images and not req.videos:
            return n
        if self.vision is None:  # generate() refuses these; the count is the placeholders'
            return (n + sum(img.tokens - 1 for img in req.images)
                    + sum(v.expansion for v in req.videos))
        return n + self.vision.expansion(req.images, req.videos)

    def _image_prompt(self, req: GenerationRequest, ids: list[int]):
        """The request's ImagePrompt, or None for a text request. Images or
        videos on an engine that cannot read them are refused here too, by
        the same reason the dialects give, in case a caller skipped their
        check."""
        if not req.images and not req.videos:
            return None
        if self.vision is None:
            raise ValueError(f"{self.model_id} cannot read images on this server "
                             f"({self.vision_reason})")
        if req.videos and self.vision.video is None:
            raise ValueError(f"{self.model_id} cannot read video on this server "
                             f"({self.vision.video_reason})")
        from .image_prompt import ImagePrompt

        return ImagePrompt(ids, req.images, self._tower, self.model, videos=req.videos,
                           cache=self._tower_cache)

    def tokenize(self, prompt: str | None = None, messages: list[dict] | None = None,
                 tools: list | None = None,
                 template_kwargs: dict | None = None) -> list[int]:
        """/tokenize and /detokenize: `messages` renders through the same
        template call generate() makes — `template_kwargs` over this
        engine's own defaults (the route passes none) — so the count this
        route reports and the prompt a request of the same shape would
        render can never disagree. `prompt` bypasses the template entirely:
        raw text in, raw ids out, vLLM's TokenizeCompletionRequest shape (a
        template-free prompt has no thinking defaults to apply)."""
        if messages is not None:
            return render_prompt(self.tok, messages,
                                 template_kwargs=effective_kwargs(self.template_kwargs,
                                                                  template_kwargs),
                                 tools=tools).ids
        return self.tok.encode(prompt)

    def detokenize(self, tokens: list[int]) -> str:
        # skip_special_tokens=False: a client handing back raw ids (possibly
        # including specials it saw from /tokenize) gets exactly what they
        # decode to, not a silently-edited string.
        return self.tok.decode(tokens, skip_special_tokens=False)

    def tokenizer_info(self) -> dict:
        return {"bos_token_id": self.tok.bos_token_id, "eos_token_id": self.tok.eos_token_id,
                "chat_template": self.tok.chat_template is not None}

    def _precapture(self, cache, spec) -> bool:
        """Graph mode, right after prefill: capture every step this request
        can replay against `cache` (the decode step, and the speculator's
        widths) before the first token, so no width is captured mid-stream;
        a cache's later requests find them captured. The server's warm-up
        request does this for the first slot allocation. False when a
        capture failed."""
        if cudagraph.decode_step(self.model, cache) is None:
            return False
        return spec is None or spec.precapture()

    def _graph_scratch(self, need: int):
        """--prefix-slots 0 in graph mode: the per-request cache is one
        cache, reset per request (in place, so the graphs captured against
        its tensors stay valid), sized as a slot is: KV_FLOOR doubling to
        `need`, capped at ctx, rebuilt bigger when a request outgrows it."""
        size = self.KV_FLOOR
        while size < need:
            size *= 2
        size = min(size, self.ctx)
        if self._scratch_kv is None or self._scratch_alloc < size:
            self._scratch_kv = None
            self._scratch_kv, self._scratch_alloc = self._cache(size), size
        else:
            self._scratch_kv.reset()
        return self._scratch_kv

    def _cache(self, n_total: int):
        # serving/kvcache.py: a StaticCache whose full-attention layers hand
        # attention the live window, so decode cost tracks the conversation
        # and the allocation only bounds memory
        from .kvcache import LiveStaticCache

        try:
            return LiveStaticCache(config=self.model.config, max_cache_len=n_total)
        except TypeError:  # pre-5.x signature
            return LiveStaticCache(config=self.model.config, max_batch_size=1,
                                   max_cache_len=n_total, device=self.device,
                                   dtype=self.dtype)

    def generate(self, req: GenerationRequest) -> Iterator[GenEvent]:
        params, tools = req.sampling, req.tools
        # tools render into the prompt via the family's own chat template
        # (Qwen3 does natively); absent tools = byte-identical prompt to
        # before. Constrained output disables thinking at the TEMPLATE
        # (template.effective_kwargs, the reasoning there).
        prompt = self._render(req)
        ids = prompt.ids
        # image input (serving/image_prompt.py): the placeholders expanded,
        # the prefix cache's keys, the M-RoPE positions. None for a text
        # request, and then every call below is the one it was before images:
        # the slots key on `ids` itself, no forward takes position_ids.
        image = self._image_prompt(req, ids)
        if image is not None:
            ids = image.ids
        key_ids = ids if image is None else image.key_ids
        # the image prompt rides along only when there is one, so a text
        # request calls prefill.run and the Speculator exactly as it did
        with_image = {} if image is None else {"image": image}
        n_prompt = len(ids)
        max_new = min(params.max_tokens, self.ctx - n_prompt)
        if max_new <= 0:  # prompt alone fills the window; nothing to sample
            yield StreamStart(n_prompt, opens_think=prompt.opens_think)
            yield Finished(GenResult("", "length", n_prompt, 0))
            return
        # Tool extraction runs BEFORE stop scanning and the Delta, so tool-call
        # text never reaches the client as visible content. Active only when
        # tools were offered; no forced stop on TEXT markers — Qwen emits its
        # calls and then stops on its own EOS, and mid-text calls must not
        # truncate the turn. A row whose close marker is a CONTROL TOKEN
        # (gemma's `<tool_call|>`, serving/control.py) does end the turn on
        # that id, tools offered or not — see `closed` in the loop below.
        # The DIALECT is the row the capability probe picked off this model's own
        # chat template (serving/tool_formats.py's table) — never guessed
        # from the model name, never a second classification.
        toolscan = (ToolCallScanner(tools, self.capability.tool_format)
                    if tools else None)
        parsed_calls: list[dict] = []
        constraint = cursor = None
        if params.output_schema is not None:
            from .constrain import JsonConstraint, pick_token

            constraint = JsonConstraint(
                params.output_schema,
                validate=not params.output_schema_validated)
        scratch = None if self._hot_off else self._scratch
        pstate = None
        scan = StopScanner(params.stop)
        detok = IncrementalDetok()
        all_ids = list(ids)  # prompt + generated: repetition-penalty window
        gen_ids: list[int] = []
        # the decoded text of gen_ids, carried across steps so the
        # constrained path does not decode it again for every candidate
        # token, and the detok below wants the same string anyway
        full = ""

        # decode(skip_special_tokens=True), with the row's control tokens
        # kept as their marker strings (serving/control.py) — for a row that
        # declares none this IS that call, unchanged. Everything downstream
        # (the suffix window, the detok, the constrained path's candidate
        # decodes, the end-of-stream flush) reads through this one function,
        # so the scanners and the client see one consistent string.
        decode_ids = self.control.decoder(self.tok)

        # the hot-loop audit: the full decode of the output, off a suffix window
        # instead of the whole id list, once per token instead of once per
        # token AND once per constrained candidate
        win = None if self._hot_off else SuffixWindow(
            decode_ids, verify=self._detok_verify)

        def decode_cand(t):
            """The candidate text for one token id — the constrained loop's
            per-candidate decode, hoisted out of the loop."""
            return win.peek(gen_ids, t)

        if constraint is not None:
            full = decode_ids([])
            if not self._hot_off:
                cursor = constraint.cursor(full)
        sent: list[str] = []
        finish = "stop"
        n = 0
        # MTP speculative decoding (serving/mtp.py), or None for the serial
        # loop. `pending` holds tokens a verify pass has already DECIDED but
        # the emit machinery below has not yet delivered; `from_cycle` counts
        # how many of the current cycle's tokens reached all_ids, which is
        # what tells the speculator how far to rewind if we stop mid-cycle.
        if self._spec_plan is None or (self.mtp_head is not None
                                       and self._mtp_depth is None):
            d = mtp.depth_from_env()
            if self.mtp_head is not None:
                # None = AUTO: the head exists (we are holding it), so auto
                # resolves to the shipped default depth.
                self._mtp_depth = mtp.DEFAULT_DEPTH if d is None else d
            self._spec_plan = mtp.spec_plan(self.mtp_head is not None, d)
            if self._spec_plan.mode != "off":
                # The verify's rewind restores DeltaNet state from this
                # capture whichever proposer drafted, so a head-less qwen3_5
                # (AUTO -> ngram) needs it as much as MTP does.
                # Idempotent, and a no-op on a model without DeltaNet layers.
                mtp.install_deltanet_capture(self.model)
        spec = mtp.maybe_speculate(self.mtp_head, self.model, params,
                                   plan=self._spec_plan)
        # serving/cudagraph.py: this request's trunk forwards replay CUDA
        # graphs when the engine graphs, the request is text and, when it
        # speculates, the family's verify widths are verified too, so a
        # graphed M = 1 row never meets an eager verify row
        graph = (self._graphs and image is None
                 and (spec is None or cudagraph.verified(self.model, "verify")))
        if spec is not None:
            spec.graphs = graph
        pending: list[int] = []
        from_cycle = 0
        mtp_stats = None
        self.last_spec_stats = None
        # ONE torch.Generator per request: seeded from the request's `seed`
        # when it sends one (the OpenAI field; http.py already parses it) —
        # seed-matched runs reproduce, which is what makes the seed-matched
        # statistical checks possible — else, for a sampled MTP request, from
        # entropy. A sampled SERIAL request without a seed keeps torch's
        # default generator, as it always has; greedy never draws.
        gen = None
        if params.temperature != 0 and (params.seed is not None or spec is not None):
            gen = torch.Generator(device=self.device)
            if params.seed is not None:
                gen.manual_seed(params.seed)
            else:
                gen.seed()

        def with_history(extra, fn):
            """Run fn() with the id windows EXTENDED by `extra` — the tokens
            this cycle has already decided ahead of the row being asked
            about — then truncated back, so a verify row (or a draft row)
            sees exactly the id history the serial loop would have shown it.

            Extended and truncated rather than concatenated: this runs once
            per verify row, which is once per drafted token per cycle, and
            copying both id lists each time would cost a copy per row."""
            k = len(extra)
            all_ids.extend(extra)
            gen_ids.extend(extra)
            try:
                return fn()
            finally:
                if k:
                    del all_ids[-k:]
                    del gen_ids[-k:]

        def pick_row(row, extra):
            """The main model's own next token for one verify row, chosen by
            the SAME function the serial loop uses, with the id history it
            would have had — so penalties (greedy but not plain argmax) stay
            exact instead of being a documented divergence."""
            return with_history(extra, lambda: sample_next(
                row, params, gen, prev_ids=all_ids, gen_ids=gen_ids))

        def probs_row(row, extra):
            """The sampled path's counterpart: the DISTRIBUTION the serial
            sampler would draw from for this row, same history. Used for
            both the draft rows (q) and the verify rows (p), which is what
            makes the accept rule's p/q a ratio of like things."""
            return with_history(extra, lambda: sample_probs(
                row, params, prev_ids=all_ids, gen_ids=gen_ids))
        with torch.inference_mode():
            slot, where, ckpts, head_kv = None, "", None, None
            if self._reuse:
                need = n_prompt + max_new
                # the reuse rule (module docstring), over N slots: the slot
                # serving most of the prompt (extending it, or reaching a
                # shared prefix by a rewind or a context checkpoint), or the
                # LRU slot cold. With one slot and --ctx-checkpoints 0 this is
                # the single-slot extends-only test verbatim.
                fit = None if image is None else image.fit
                slot, lcp = pick_slot(self._slots, key_ids, n_prompt, need, fit)
                where = "" if len(self._slots) == 1 else f" [slot {slot.n}]"
                if self._cold is not None and (
                        lcp == 0 or lcp < ctx_checkpoints.KEEP * len(slot.ids)):
                    # on-disk prefix slots: nothing live matched (or the slot
                    # picked would keep less than half of what it holds:
                    # llama.cpp's update_cache), so the slot goes to the
                    # cold tier and the cold tier is asked for this prompt —
                    # both BEFORE a prefill is committed to. A better hit
                    # leaves `slot` holding a restored cache, which the code
                    # below then treats exactly as an in-RAM match.
                    lcp = self._take_from_cold(slot, key_ids, need, where, n_prompt, fit,
                                               floor=lcp)
                # the slot's checkpoints are detached while this request
                # mutates the cache, as slot.ids is emptied below: an
                # exception leaves the slot with neither
                ckpts, slot.ckpts = slot.ckpts, self._new_checkpoints()
                # and so is the MTP head's KV it kept (mtp.HeadKV), cropped
                # below once the reuse point is settled
                head_kv, slot.head = slot.head, None
                old_ids = slot.ids
                if image is not None and image.cuts(lcp):
                    # a reuse point inside an image run whose tokens attend
                    # forward within it (gemma-4): the rows below lcp were
                    # written without the rest of the run, so none of it is
                    # reused (image_prompt.py, "HOW A RUN MAY BE PREFILLED")
                    print(f"[drinkme.engine] prefix cache{where}: the reusable prefix "
                          f"({lcp} tokens) ends inside an image; full prefill this turn",
                          file=sys.stderr)
                    lcp = 0
                if lcp == 0:
                    size = self.KV_FLOOR
                    while size < need:
                        size *= 2
                    size = min(size, self.ctx)
                    if slot.cache is not None:
                        metrics.record_slot_event("evict")
                        print(f"[drinkme.engine] prefix cache reset{where} "
                              f"(alloc {slot.alloc} -> {size}); full "
                              f"prefill this turn", file=sys.stderr)
                    else:
                        metrics.record_slot_event("miss")
                        metrics.slot_occupied()
                    slot.cache = self._cache(size)
                    slot.alloc = size
                    ckpts.clear()
                elif lcp == len(slot.ids):
                    metrics.record_slot_event("hit")  # the code's own word: EXTENDS
                else:
                    self._reach(slot, ckpts, key_ids, lcp, where)
                if head_kv is not None:
                    if lcp == 0:
                        head_kv = None
                    else:
                        # entry p = (h_p, t_{p+1}) holds while tokens 0..p+1
                        # do: below lcp - 1, and entry lcp - 1 too when the
                        # slot's token at lcp is the prompt's (mtp.HeadKV)
                        same = lcp < len(old_ids) and old_ids[lcp] == key_ids[lcp]
                        head_kv.keep(lcp if same else lcp - 1, lcp)
                self._slot_clock += 1
                slot.stamp = self._slot_clock  # taken or matched, it is MRU now
                cache = slot.cache
                slot.ids = []  # invalid while we mutate; restored on exit
            else:
                # per-request (0 slots); in graph mode one scratch cache,
                # reset per request, so its graphs are captured once
                cache = (self._graph_scratch(n_prompt + max_new) if graph
                         else self._cache(n_prompt + max_new))
                lcp = 0
            if image is not None:
                image.begin(lcp)  # images inside the reused prefix never run the tower
            # the explicit stream-start fact (engine.py): the prompt's shape
            # is settled, the prefix cache has answered, nothing has run yet
            yield StreamStart(n_prompt, opens_think=prompt.opens_think, cached_tokens=lcp)
            # serving/prefill.py: ids[lcp:] in forwards of at most
            # self._prefill_chunk tokens (0 = one forward, the unchunked path),
            # so no single GPU dispatch grows with the prompt — a desktop APU's
            # compositor shares this GPU and amdgpu resets its ring after 2 s
            chunk = self._prefill_chunk
            # context checkpoints (serving/ctx_checkpoints.py): a prefill span
            # ends at each position this request snapshots — near the prompt's
            # end, where the next request's re-rendered history parts from
            # this one — and the snapshot is taken as soon as the cache holds
            # everything before it. When the rings will still hold every
            # position after the prefill (ctx_checkpoints.ring_holds), the
            # prefill is not split: all of them are copied out of the rings
            # once it ends at n_prompt. A prompt with images or videos takes
            # one more at each media item's end, where a new question about
            # the same media parts from it (MEDIA ENDS); a text prompt's
            # stops are the ones it always took
            at = {}
            if ckpts is not None and self._ckpt_take:
                stops = ctx_checkpoints.positions(
                    n_prompt, lcp, () if image is None else image.whole(lcp, chunk),
                    user=self._last_user_start(req, prompt.ids, image))
                media = set() if image is None else set(ctx_checkpoints.media_positions(
                    image.media_ends(), lcp, n_prompt, self._ckpt_media))
                stops = sorted(set(stops) | media)
                if ctx_checkpoints.ring_holds(cache, n_prompt):
                    def take(_end, task=slot.stamp, stops=stops):
                        for p in stops:
                            ckpts.add(ctx_checkpoints.snapshot(cache, task, at=p,
                                                               media=p in media))

                    at = {"stops": (n_prompt,), "at_stop": take}
                else:
                    def take(p, task=slot.stamp):
                        ckpts.add(ctx_checkpoints.snapshot(cache, task, media=p in media))

                    at = {"stops": stops, "at_stop": take}
            if spec is None:
                # logits_to_keep=1 (inside prefill.run): prefill needs the
                # LAST row and nothing else. transformers defaults it to 0,
                # meaning "every position", which would build an
                # [n_prompt, 248320] tensor and discard all but one row —
                # 0.474 MiB per prompt token, 121.2 GiB at the 262144 context
                # we advertise, on a 124 GiB machine. A live 500 at 21k tokens is
                # what found it. PAIRED with the last_row_only below: both
                # arms narrow or neither does, or MTP-on and MTP-off fork on
                # a near-tie (see mtp.forward_with_hidden's docstring for the
                # full reasoning). These are ONE change.
                logits = prefill.run(self.model, ids, lcp, cache, self.device, chunk,
                                     **with_image, **at)
                if not self._hot_off and (params.repetition_penalty != 1.0
                                          or params.presence_penalty
                                          or params.frequency_penalty):
                    # the hot-loop audit: the penalty windows go on the device once
                    # and advance by one accepted token, instead of being
                    # rebuilt from the id lists every step. MTP keeps the list
                    # path — its verify rows ask about hypothetical futures
                    # (pick_row below), which no single running state can hold.
                    pstate = PenaltyState(logits.numel(), logits.device)
                    pstate.observe(ids)
            elif spec.needs_hidden:
                # the draft head needs the TRUNK HIDDEN at the last prompt
                # position, not just its logits — prefill itself is unchanged.
                # last_row_only (inside prefill.run) is the MTP half of the
                # pair above: prefill keeps one row of lm_head, the draft
                # CYCLE still computes every row of its verify window (it
                # reads logits[m]). The head is seeded span by span as the
                # trunk's hidden states come out of each chunk, on top of
                # the KV the slot kept for the reused prefix.
                spec.open(cache, ids, history=head_kv, start=lcp, **with_image)
                logits = prefill.run(self.model, ids, lcp, cache, self.device, chunk,
                                     hidden=True, on_hidden=spec.seed, **with_image, **at)
                pending.append(pick_row(logits, []))
            else:
                # n-gram speculation, ngram-only: the proposer reads token ids, never hidden
                # states, so prefill is the SERIAL loop's prefill, character
                # for character. That is what lets this arm's greedy identity
                # rest on "nothing before the decode loop changed" instead of
                # on a near-tie argument about two spellings of lm_head.
                logits = prefill.run(self.model, ids, lcp, cache, self.device, chunk,
                                     **with_image, **at)
                spec.begin(cache, ids, None, lcp, **with_image)
                pending.append(pick_row(logits, []))
            if graph and not self._precapture(cache, spec):
                # a capture that failed, said once: eager from here on
                self._graphs = graph = False
                if spec is not None:
                    spec.graphs = False
            if image is not None:
                image.announce(where)
            written = n_prompt  # tokens whose KV now sits in the cache
            # hot loop writes these in place; it allocates no tensors itself
            step_in = torch.empty((1, 1), dtype=torch.long, device=self.device)
            step_pos = torch.empty((1,), dtype=torch.long, device=self.device)
            # after an image prompt the step's M-RoPE rows ([p, p+delta x3],
            # image_prompt.ImagePrompt.step), preallocated like step_pos; a
            # text request passes no position_ids at all
            step_rope = (torch.empty((4, 1, 1), dtype=torch.long, device=self.device)
                         if image is not None and image.rope else None)
            step_kw = {} if step_rope is None else {"position_ids": step_rope}
            # serving/control.py: True while the last accepted token was the
            # row's call-close marker — the one-token lookahead that ends the
            # turn unless the model re-opens a call
            closed = False
            while True:
                if spec is not None and spec.drafting:
                    tok_id = pending.pop(0)
                elif constraint is None:
                    tok_id = sample_next(logits, params, gen, prev_ids=all_ids,
                                         gen_ids=gen_ids, state=pstate,
                                         scratch=scratch)
                else:
                    tok_id = pick_token(
                        logits, params, gen, all_ids, gen_ids, self.eos_ids,
                        decode_ids, constraint,
                        cur=None if cursor is None else full,
                        decode_cand=None if cursor is None else decode_cand,
                        cursor=cursor, state=pstate, scratch=scratch)
                n += 1
                if tok_id in self.eos_ids:
                    break  # natural end: finish stays "stop", tail flushes below
                if closed and tok_id not in self.control.reopen:
                    # the previous token closed a tool call (the row's control
                    # close marker, already fed through the scanner below).
                    # Only the row's own re-open continues the turn — a second
                    # call, rendered back to back. Anything else is the model
                    # talking past its call (a hallucinated answer), so the
                    # turn ends here with this token
                    # unemitted, exactly as at EOS. finish stays "stop" and
                    # becomes "tool_calls" below once the parsed call is seen.
                    break
                closed = tok_id in self.control.stop_after
                all_ids.append(tok_id)
                gen_ids.append(tok_id)
                if pstate is not None:
                    pstate.accept(tok_id)
                from_cycle += 1
                # incremental detok (serving/detok.py): prefix diff + grapheme
                # hold-back — never split a ZWJ/VS cluster across deltas (a
                # split emoji cluster stacks stale lines in a real client's
                # TUI), and the end-of-stream
                # flush below guarantees held text is never lost at EOS
                full = win.full(gen_ids) if win is not None else decode_ids(gen_ids)
                if cursor is not None and cursor.text != full:
                    # the window slid and corrected its own tail (or gave up):
                    # the constraint follows the TEXT, never the other way
                    cursor.probe(full)
                    cursor.accept()
                delta = detok.push(full)
                if toolscan is not None:
                    delta, done = toolscan.feed(delta)
                    parsed_calls += done
                out = scan.feed(delta)
                if out:
                    if (yield Delta(out)) is False:
                        finish = "abort"
                        break
                    sent.append(out)
                if scan.stopped:
                    break
                if n == max_new:
                    finish = "length"
                    break
                if spec is not None and spec.drafting and not pending and spec.bailed:
                    # ADAPTIVE BAIL (decided guard 2): rolling acceptance fell
                    # below the floor, so this request decodes on the serial
                    # loop below — for the speculator's serial stretch, or to
                    # the end (DRINKME_MTP_REARM=0). Every token the last
                    # cycle decided has been appended, so finish() drops
                    # nothing and `written` is exactly the cache. The serial
                    # loop's own penalty state is rebuilt to where it would
                    # have been — bitwise the list path MTP was using
                    # (equivalence tests).
                    written = spec.finish(from_cycle)
                    spec.trip()
                    if not self._hot_off and (params.repetition_penalty != 1.0
                                              or params.presence_penalty
                                              or params.frequency_penalty):
                        pstate = PenaltyState(logits.numel(), logits.device)
                        pstate.observe(ids)
                        for t in gen_ids:
                            pstate.accept(t)
                elif spec is not None and spec.rearm_due:
                    # RE-ARM: the stretch is spent. `tok_id` was just sampled
                    # serially and sits at position `written`, the cache
                    # holds everything before it — the same boundary the
                    # trip left, in the other direction — and the speculator
                    # rebuilds the head's side of it from the rows
                    # serial_step kept. The cycle below takes over from the
                    # serial step; penalties go back to the list path the
                    # verify rows need (pick_row), so the running state is
                    # dropped, to be rebuilt at the next trip if there is one.
                    spec.rearm()
                    pstate = None
                if spec is None or not spec.drafting:
                    step_in[0, 0] = tok_id
                    step_pos[0] = n_prompt + n - 1
                    if step_rope is not None:
                        image.step(step_rope, n_prompt + n - 1)
                    if spec is not None and spec.rearm_pending:
                        # the same M=1 forward, its hidden row kept for the
                        # head (mtp.Speculator.serial_step: bitwise the
                        # logits of the line below)
                        logits = spec.serial_step(step_in, step_pos, **step_kw)
                    else:
                        st = cudagraph.decode_step(self.model, cache) if graph else None
                        if graph and st is None:
                            # a capture that failed, said once: eager from here on
                            self._graphs = graph = False
                            if spec is not None:
                                spec.graphs = False
                        if st is not None:
                            st.ids.fill_(tok_id)
                            logits = st.replay().clone()
                        else:
                            logits = self.model(step_in, past_key_values=cache,
                                                use_cache=True,
                                                cache_position=step_pos, **step_kw).logits[0, -1]
                    written += 1
                elif not pending:
                    # one draft/verify cycle replaces the one-token step: it
                    # returns every token the serial loop would have emitted
                    # next (m accepted drafts + the main model's own token —
                    # or, sampled, + the residual resample / bonus token)
                    pending = spec.cycle(tok_id, pick_row, max_new - n,
                                         probs=probs_row, generator=gen)
                    written, from_cycle = spec.written, 0
            if spec is not None:
                # tokens still queued (or one popped and refused) have KV in
                # the cache that nothing will ever emit: rewind to what the
                # caller actually kept, or `written` below would lie. On the
                # serial loop there is no cycle to rewind and `written` is
                # this loop's own count.
                if spec.drafting:
                    written = spec.finish(from_cycle)
                mtp_stats = spec.stats()
                self.last_spec_stats = mtp_stats
                if spec.head is not None:
                    # the head's KV for what the cache now holds, detached
                    # from the shared head: the slot keeps it below; with the
                    # prefix cache off it is dropped here, and freed
                    head_kv = spec.history(written)
            if mtp_stats is not None and mtp_stats["cycles"]:
                s = mtp_stats
                bail = ""
                if s["bailed"]:
                    # the adaptive bail's accounting: how many times it
                    # tripped (the last trip's rolling rate) and re-armed,
                    # and whether the request ended serial
                    bail = (f", bailed to serial {s['trips']}× (last at "
                            f"{s['bail_rate']:.0%} rolling), re-armed "
                            f"{s['rearms']}×")
                    if s["serial_at_end"]:
                        bail += ", serial at end"
                mode = " sampled" if s["sampled"] else ""
                # D4 (one denominator): `drafted` is the bail window's own
                # accounting (undecided positions after a rejection count as
                # rejected — "is MTP paying" on that basis, unchanged) and
                # is NOT the audit's — the audit's `expected_rate` is a mean
                # over DECIDED positions only, so comparing it against
                # accepted/drafted (A2, docs/serve-speculation.md) was two different
                # denominators wearing one percentage. Print both, honestly.
                # n-gram speculation: which proposer served this request, and — when it was
                # a lookup — how often it found anything at all. A proposer
                # that finds nothing is the neutral case, not a failure, so
                # the line says so rather than reporting 0% acceptance.
                look = ""
                if s.get("lookup"):
                    lk = s["lookup"]
                    look = (f" · lookup {lk['matched']}/{lk['asked']} cycles "
                            f"matched ({lk['hit_rate']:.0%}, mean n="
                            f"{lk['mean_match_n']}), {lk['proposed']} proposed")
                audit = s.get("audit")
                if audit is not None and audit["decided"]:
                    v = audit["violations"]
                    body = (f"{s['accepted']}/{s['drafted']} drafted · "
                            f"{audit['accepted']}/{audit['decided']} decided "
                            f"({audit['actual_rate']:.0%}, "
                            f"expect {audit['expected_rate']:.0%}) · audit "
                            f"{'ok' if v == 0 else f'VIOLATION {v}'}")
                else:
                    body = (f"{s['accepted']}/{s['drafted']} drafts accepted "
                            f"({s['acceptance']:.0%})")
                print(f"[drinkme.spec] {s['mode']}: {n} tokens in "
                      f"{s['cycles']}{mode} cycles, {body}{bail}{look}",
                      file=sys.stderr)
        if self._reuse:
            # exactly the ids whose KV was written — on the break-before-forward
            # paths (stop/length/abort) the last appended id is NOT in the cache.
            # An image prompt's slot holds its KEYS (image_prompt.py): the
            # prompt as key_ids, then the generated ids.
            slot.ids = (all_ids[:written] if image is None
                        else key_ids + all_ids[n_prompt:written])
            # the checkpoints come back with the ids; whether this cache now
            # reaches any earlier position by length alone is read off it here
            ckpts.free = ckpts.on and ctx_checkpoints.rewinds_freely(cache)
            slot.ckpts = ckpts
            # and the head's KV: what this request left in it, or, when it
            # ran without the head (serially, or grammar-constrained), what
            # the slot kept for the prefix it reused
            slot.head = head_kv
            if lcp:
                print(f"[drinkme.engine] prefix cache{where}: reused "
                      f"{lcp}/{n_prompt} prompt tokens", file=sys.stderr)
        if finish in ("stop", "length") and not scan.stopped:
            # held-back tails, stream order: the DETOK's own hold first (text
            # the grapheme/FFFD guards deferred — at EOS it must flush or it
            # is silently lost), then
            # the tool scanner's tail (an un-terminated block re-emerging as
            # text) feeds the stop scanner, then the stop lookahead flushes —
            # all of it is generated text
            rem = detok.flush(decode_ids(gen_ids))
            if toolscan is not None:
                r, done = toolscan.feed(rem)
                parsed_calls += done
                tail = scan.feed(r) + scan.feed(toolscan.flush())
                # a ONE-SIDED row (mistral) finishes its call on EOS, so the
                # flush is where its calls surface — empty for every other row
                parsed_calls += toolscan.flush_calls()
            else:
                tail = scan.feed(rem)
            tail += scan.flush()
            if tail:
                if (yield Delta(tail)) is False:
                    finish = "abort"
                else:
                    sent.append(tail)
        if parsed_calls and finish == "stop":
            finish = "tool_calls"  # Qwen's shape: emit calls, then stop
        yield Finished(GenResult("".join(sent), finish, n_prompt, n,
                                 tool_calls=parsed_calls or None, cached_tokens=lcp,
                                 stop_sequence=scan.matched))


# ------------------------------------------------------------------ loading --


def _mtp_head(model, repo: str, revision: str | None, device: str,
              pack_dir: str | None = None):
    """The MTP draft head, or None. Both arms load it identically — MTP is a
    decode-loop change, not a weight-read change, so it must not become a
    third arm: whatever it does for compressed it must do for --stock, or the
    A/B stops measuring the codec.

    `pack_dir` (the head diet) is the one asymmetry, and it is the
    RIGHT one: it names the pack the compressed arm is already serving from,
    so if that pack carries a packed head the head is served compressed too —
    which is the codec applied to one more group of tensors, not a second
    decode loop. --stock has no pack by definition and keeps its raw head, so
    the arms still differ in exactly one thing: whether weights are packed."""
    from . import mtp, ngram

    if not ngram.wants_head():
        # n-gram speculation: DRINKME_SPEC=off|ngram can never call a head, so don't pay its
        # residency. On the 27B that is ~0.8 GB handed back to the KV cache.
        return None
    d = mtp.depth_from_env()
    if d is not None:
        return mtp.load_head(model, repo, revision, device, pack_dir=pack_dir)
    # AUTO (unset): default-on when the model allows.
    # load_head already degrades politely for a head-less family; anything
    # else (I/O, OOM on the ~0.8GB head) must degrade too rather than keep a
    # stranger's first `drinkme serve` from coming up — an EXPLICIT
    # DRINKME_MTP_DEPTH=k keeps its loud failure, because that person asked for
    # exactly this.
    #
    # Guard 1 (D2, decided guard 1 — FIT OUTRANKS SPEED): AUTO only, never
    # a forced depth (that person asked for exactly this, same as above).
    # device_headroom_gib is None off a real accelerator (cpu/mps, or a
    # monkeypatched fixture in tests) — nothing to guard there, load as
    # before.
    headroom = mtp.device_headroom_gib(device)
    if headroom is not None:
        free_gib, used_gib = headroom
        hg = mtp.head_gib(repo, revision, pack_dir)
        if hg is not None:
            yields, left, margin = mtp.residency_check(free_gib, used_gib, hg)
            if yields:
                mtp._warn_once(
                    "residency",
                    f"head would leave {left:.2f} GB headroom (< {margin:.2f} "
                    "GB); MTP yields — DRINKME_MTP_DEPTH=4 to force")
                return None
    try:
        return mtp.load_head(model, repo, revision, device, pack_dir=pack_dir)
    except Exception as err:  # noqa: BLE001 — containment is the point
        mtp._warn_once("autoload", f"MTP auto-load failed ({err}) — decoding without it")
        return None


def _vision_for(cfg, snap: str, repo: str, tokenizer=None):
    """Image input for this checkpoint on this server: (vision.Vision,
    vision.Tower, None), and the loader builds the tower into the served
    tree; or (None, None, why not), the reason the refusal of an image
    names. `cfg` is the checkpoint's own config (the wrapper, for a
    composite), `snap` the directory its processor config is read from,
    and `tokenizer` a zero-arg
    callable returning the checkpoint's tokenizer, called only for a tower
    whose image markers the config does not carry (Muse-Glimmer's; see
    vision.tower_for). DRINKME_VISION=0 turns it off (serving/vision.py).

    A text model this server cannot splice images into is refused by
    name: gemma-4's per-layer embeddings (the E-models; they read the
    image rows' token ids, which drinkme does not thread), or a
    `use_bidirectional_attention` other than none or "vision" (the one
    image mask serving/image_prompt.py builds)."""
    from ..arms import vision_tower_paths
    from . import vision

    mt = getattr(cfg, "model_type", None)
    vc = getattr(cfg, "vision_config", None)
    if vc is None:
        return None, None, "text-only model"
    paths = vision_tower_paths(cfg)
    if paths is None:
        return None, None, f"this server has no image input for {mt}"
    text = cfg.get_text_config() if hasattr(cfg, "get_text_config") else cfg
    if getattr(text, "hidden_size_per_layer_input", None):
        return None, None, f"this server has no image input for {mt} with per-layer embeddings"
    if getattr(text, "use_bidirectional_attention", None) not in (None, "vision"):
        return None, None, (f"this server has no image input for {mt} with "
                            f"use_bidirectional_attention={text.use_bidirectional_attention!r}")
    if not vision.enabled_from_env():
        return None, None, f"disabled by {vision.VISION_ENV}=0"

    def encode(text: str) -> list[int]:  # a video's timestamps (serving/video.py)
        return tokenizer().encode(text, add_special_tokens=False)

    vis = vision.load(mt, snap, encode=encode if tokenizer else None)
    if vis is None:
        return None, None, "the checkpoint carries no image processor config"
    pre = vis.preprocessor
    unavailable = getattr(pre, "unavailable", None)
    why = unavailable() if unavailable is not None else None
    if why:
        return None, None, why
    # the tower configs' spellings: Qwen3.5's spatial_merge_size and
    # temporal_patch_size, Muse-Glimmer's merge_size and patch_temporal
    for mine, theirs in (("patch_size", "patch_size"), ("merge_size", "spatial_merge_size"),
                         ("merge_size", "merge_size"),
                         ("temporal_patch_size", "temporal_patch_size"),
                         ("temporal_patch_size", "patch_temporal"),
                         ("pooling_kernel_size", "pooling_kernel_size")):
        a, b = getattr(pre, mine, None), getattr(vc, theirs, None)
        if a is not None and b is not None and a != b:
            raise ValueError(f"{repo}: the image processor config says {mine}={a}, the "
                             f"vision tower's config {theirs}={b}; refusing to serve images "
                             "cut for one into the other")
    tower = vision.tower_for(mt, cfg, paths,
                             tokenizer() if mt == "muse_glimmer" and tokenizer else None)
    # the markers the prompt carries around each image's run: the count
    # routes add pre.wrap per image, ImagePrompt inserts boi/eoi
    if pre.wrap != (tower.boi is not None) + (tower.eoi is not None):
        raise ValueError(f"{repo}: the {mt} preprocessor counts {pre.wrap} marker tokens per "
                         "image, the tower inserts a different number")
    if vis.video is not None:
        vpre = vis.video.preprocessor
        if tower.video_token_id is None:
            vis = dataclasses.replace(vis, video=None, video_reason=(
                "the checkpoint's config names no video_token_id, vision_start_token_id "
                "and vision_end_token_id"))
        else:
            for mine, theirs in (("patch_size", "patch_size"),
                                 ("merge_size", "spatial_merge_size"),
                                 ("temporal_patch_size", "temporal_patch_size")):
                a, b = getattr(vpre, mine), getattr(vc, theirs, None)
                if b is not None and a != b:
                    raise ValueError(f"{repo}: the video processor config says {mine}={a}, "
                                     f"the vision tower's config {theirs}={b}; refusing to "
                                     "serve video cut for one into the other")
    return vis, tower, None


def load_stock(repo: str, revision: str | None = None, device: str = "cuda",
               ctx: int | None = None, prefix_slots: int | None = None,
               rope_scaling: dict | None = None) -> HFEngine:
    """The A/B control arm: plain bf16 through the identical engine class.

    rope_scaling (rope scaling) is None on the default path — no extra config
    resolve, no behaviour change, load_cpu(repo, revision) exactly as
    before. Only when a scaling spec is given do we resolve AutoConfig
    ourselves (mirroring load_compressed's local_files_only-then-network
    fallback), mutate it, and hand it to load_cpu as an explicit config= so
    from_pretrained builds the model from the WIDENED window instead of
    re-resolving its own config.json over it."""
    import itertools

    from transformers import AutoConfig

    from ..arms import attach_vision_tower, has_vision_tower, load_cpu, snapshot_dir

    cfg = None
    if rope_scaling is not None:
        kw = {"revision": revision} if revision else {}
        try:  # local-first, same reasoning as _tokenizer/load_compressed
            cfg = AutoConfig.from_pretrained(repo, local_files_only=True, **kw)
        except (OSError, ValueError):
            cfg = AutoConfig.from_pretrained(repo, **kw)
        _apply_rope_scaling(cfg, rope_scaling)

    snap = snapshot_dir(repo, revision)
    # image input: from_pretrained builds Qwen3.5's text-only tree, so its
    # served tower is attached below and streamed from the same snapshot;
    # gemma-4's and Muse-Glimmer's classes build and load their own, which
    # load_cpu prunes, before it reaches the device, when images are off
    full = cfg if cfg is not None else AutoConfig.from_pretrained(snap)
    tok = _tokenizer(repo, revision)
    vis, tower, vision_reason = _vision_for(full, snap, repo, tokenizer=lambda: tok)
    model = load_cpu(repo, revision, config=cfg, vision=tower is not None)
    # transformers-5 hands weights back as views onto the mmap'd safetensors
    # shards (MAP_PRIVATE, file-backed — verified against /proc/self/maps).
    # ROCm's pageable H2D copy from such pages crawls at ~33 MB/s: a 16 GiB
    # model took 492s inside .to(cuda), ~4M minor faults, GPU idle throughout
    # (the compressed fit path never sees this because it moves
    # freshly-DECODED anonymous memory). One clone
    # per tensor materializes into anonymous memory first: page-cache -> anon
    # is a memcpy, anon -> device is a normal staged DMA — seconds, not
    # minutes. Peak extra host memory = one tensor. Parameter identity is
    # preserved (parameters() dedups, so tied weights move once, still tied).
    for p in itertools.chain(model.parameters(), model.buffers()):
        p.data = p.data.clone().to(device)
    model = model.to(device)  # stragglers only; weights are already there
    if tower is not None and not has_vision_tower(model, full):
        attach_vision_tower(model, full)
        stream_checkpoint(model, snap, full, device, name=repo,
                          only=tuple(p + "." for p in tower.paths))
    # the runtime's kernel routes (serving/kernel_route.py): the narrow
    # GEMV and the DeltaNet recurrence, the ONE call the bench's arms and
    # load_compressed make too, so the arms cannot drift from serve
    route_kernels(model, device)
    gd = gen_config.load(model, snap)
    # resolvedRevision: the immutable identity of the weights just loaded —
    # the hub snapshot's commit sha, or a digest of a local directory —
    # which is what the cold tier keys its slots on (a
    # bare meta={} let two revisions of one repo share a KV cache)
    meta = {"resolvedRevision": resolved_revision(snap), "sourceDtype": "bf16"}
    engine = HFEngine(model, tok, model_id=repo,
                      arm="stock", meta=meta, ctx=_ctx(model.config, ctx),
                      mtp_head=_mtp_head(model, repo, revision, device),
                      prefix_slots=prefix_slots, gen_defaults=gd,
                      vision=vis, tower=tower, vision_reason=vision_reason)
    # sleep/wake: how a level-2 wake comes back — THIS function, THESE arguments.
    # A closure rather than a saved dict of arguments so there is exactly one
    # place that knows how this engine was built, and a level-2 wake is
    # measurable against a cold boot because it IS a cold boot.
    engine._reload = lambda: load_stock(repo, revision, device, ctx=ctx,
                                        prefix_slots=prefix_slots,
                                        rope_scaling=rope_scaling)
    return engine


def _rebuild_computed_buffers(model, cfg, device) -> None:
    """Computed, non-persistent buffers are NOT in the checkpoint and arrive
    from the meta skeleton as meta tensors. Two kinds exist in the models we
    serve, and each has an exact recipe — zero-filling either one is a
    silent 200 with garbage in it (gemma-4-31B-it: 400 tokens of nothing at
    100% n-gram acceptance, with `embed_scale` zeroed and the text rotary
    not rebuilt).

    1. ROTARY `inv_freq`. Re-instantiate the parameter-free rotary module from
       the config its OWNER was built with. Matched by buffer-name SUFFIX:
       gemma-4's text rotary keeps one pair per layer type
       (`sliding_attention_inv_freq`, `full_attention_inv_freq`, plus their
       `_original_` twins), not a bare `inv_freq`. Qwen3.5's vision rotary
       takes no config at all — `(dim, theta)`, which it keeps on itself —
       so it is rebuilt from those.
    2. `embed_scale` on a scaled word embedding. HF's own `_init_weights` does
       `init.constant_(module.embed_scale, module.scalar_embed_scale)` — the
       scalar lives on the module as a plain attribute, so the buffer is
       recomputed from it, never guessed from a formula."""
    for mname, mod in list(model.named_modules()):
        if any(bn.endswith("inv_freq") and b is not None and b.is_meta
               for bn, b in mod._buffers.items()):
            parent_path, _, leaf = mname.rpartition(".")
            rcfg = _owning_config(model, mname, cfg)
            try:
                fresh = type(mod)(config=rcfg, device=device)
            except (TypeError, AttributeError, KeyError):
                if hasattr(mod, "dim") and hasattr(mod, "theta"):
                    fresh = type(mod)(mod.dim, mod.theta).to(device)
                else:
                    fresh = type(mod)(rcfg).to(device)
            setattr(model.get_submodule(parent_path), leaf, fresh)
    for mod in model.modules():
        b = mod._buffers.get("embed_scale")
        if b is not None and b.is_meta and hasattr(mod, "scalar_embed_scale"):
            mod._buffers["embed_scale"] = torch.tensor(mod.scalar_embed_scale,
                                                       dtype=b.dtype, device=device)


def _owning_config(model, mname: str, cfg):
    """The config a rotary module was BUILT with: the nearest ancestor module
    carrying a `.config`. HF instantiates each rotary inside its owner —
    `Gemma4VisionModel(vision_config)` builds the vision rotary, the text
    model builds the text one — so the owner's config is the right one by
    construction. Handing every rotary the text config (the first draft) broke
    on gemma-4-31B-it: the vision rotary reads `rope_parameters["rope_type"]`,
    a key the TEXT config does not have (its rope_parameters are keyed by
    layer type). Falls back to the text config for a bare rotary at the root."""
    path = mname
    while path:
        path = path.rpartition(".")[0]
        owner = model.get_submodule(path) if path else model
        oc = getattr(owner, "config", None)
        if oc is not None:
            return oc
    return cfg.get_text_config() if hasattr(cfg, "get_text_config") else cfg


def stream_checkpoint(model, snap: str, cfg, device, name: str | None = None,
                      only: str | tuple[str, ...] | None = None) -> None:
    """Materialize a meta-device skeleton (arms.skeleton) from a checkpoint
    snapshot ONE TENSOR AT A TIME, straight onto `device`: every safetensors
    shard under `snap` is opened in turn, each tensor is read, cloned into
    anonymous host memory, moved to `device` (float tensors cast to bf16,
    exactly as `from_pretrained(dtype=bf16)` does) and attached in place of
    its meta parameter/buffer — then the host copy is released before the
    next tensor is read. Peak host memory beyond the mmap'd page cache is
    one tensor, never one model. Then the tail every meta-built tree needs:
    tie_weights (a tied lm_head is ABSENT from the checkpoint), the
    computed-buffer rebuild (_rebuild_computed_buffers: rotary inv_freq,
    embed_scale — constructed, never loaded), the leftover-meta refusal,
    eval() and requires_grad off.

    THE ONE WALKER. It was the tail of
    load_compressed; it is shared now with the bench's streaming stock
    loader (arms.load_stock_streaming), which needs exactly this over a
    tree holding NO CompressedLinear at all — on that tree every branch
    below that names a swapped Linear is simply never taken — and with the
    bench's streaming compressed and twin arms (arms.load_compressed_streaming
    / load_twin_streaming), whose trees hold CompressedLinears at the
    eligible slots (the twin's RadixTwinLinear is one). Two walkers would be two
    opinions on the key mapping (arms.ckpt_to_skel) and on which tensors
    are computed rather than loaded, and the bit-identical claim rests on
    there being one. Nothing here reads the pack: a swapped Linear's weight
    is skipped by module type (swap.SWAPPED_LINEAR_TYPES), so `model` may
    hold packed Linears (load_compressed), twins (the bench) or none (the
    stock arm).

    Wrapped checkpoints (qwen3_5's model.language_model.* -> model.*,
    the vision/MTP towers the text skeleton never builds) go through
    ckpt_to_skel, whose skip list is the same table skeleton() pruned by
    (arms._NON_TEXT_TOWERS) — a tensor is never offered a home the tree does
    not have, and no parameter the tree has is left meta. A served vision
    tower (arms._VISION_TOWERS) is streamed when the tree holds it and
    skipped when it does not: the tree decides, so a caller never says it
    twice. `only` (a skeleton-name prefix, or a tuple of them) streams
    that subtree alone: load_stock's tower, attached to a tree
    from_pretrained already filled.

    THE RAW-TRUNK DOOR: an FP8 checkpoint (any float8 plane in the shard
    headers, or an fp8 quantization_config) is refused by name before a
    tensor is read — the one line (codec/pack.refuse_checkpoint) — since
    the bf16 cast below would otherwise store unscaled codes as the
    weight. Every loader that walks a checkpoint comes through here, so
    a --pack-dir cut from a bf16 pack cannot be served over fp8 shards.
    `name` is what the line calls the checkpoint (the repo id); the
    snapshot path when the caller has nothing better."""
    import glob

    from safetensors import safe_open

    from ..arms import ckpt_to_skel
    from ..codec.pack import refuse_checkpoint
    from ..codec.swap import SWAPPED_LINEAR_TYPES

    files = sorted(glob.glob(os.path.join(snap, "*.safetensors")))
    if not files:
        raise FileNotFoundError(f"no *.safetensors under {snap}")
    refuse_checkpoint(name or snap, snap)
    from ..arms import has_vision_tower

    # wrapped checkpoints name text weights differently; the tree says
    # whether its vision tower is there to be streamed
    to_skel = ckpt_to_skel(cfg, vision=has_vision_tower(model, cfg))
    for fpath in files:
        with safe_open(fpath, framework="pt") as sf:
            for tname in sf.keys():
                sname = to_skel(tname)
                if sname is None or (only is not None and not sname.startswith(only)):
                    continue  # a tower this tree never builds (vision/MTP)
                mod_path, _, attr = sname.rpartition(".")
                try:
                    mod = model.get_submodule(mod_path)
                except AttributeError as e:
                    raise ValueError(f"checkpoint tensor {tname} has no module") from e
                if isinstance(mod, SWAPPED_LINEAR_TYPES) and attr != "bias":
                    # the pack (or the twin's planes) IS this tensor;
                    # attaching it would materialize the BF16 read the codec
                    # exists to avoid. get_tensor is not called at all — its
                    # laziness is an upstream detail we refuse to depend on
                    # (the hot-loop audit).
                    continue
                t = sf.get_tensor(tname)
                if isinstance(mod, SWAPPED_LINEAR_TYPES):  # bias, arrived after swap
                    mod.bias = t.clone().to(device=device, dtype=torch.bfloat16)
                    del t
                    continue
                # float tensors go to bf16 exactly as from_pretrained(dtype=bf16)
                # does for the stock arm — both arms must serve identical bytes.
                # .clone() first: get_tensor returns a view onto a MAP_PRIVATE
                # shard, and ROCm pageable H2D from file-backed pages measured
                # 33MB/s on a Strix Halo / Ryzen AI Max+ 395 with 128 GB unified
                # memory (load_stock's fix, applied here too — the hot-loop
                # audit; peak extra host = one tensor).
                dt = t.clone().to(device=device,
                                  dtype=torch.bfloat16 if t.is_floating_point() else None)
                if attr in mod._parameters:
                    mod._parameters[attr] = torch.nn.Parameter(dt, requires_grad=False)
                elif attr in mod._buffers:
                    mod._buffers[attr] = dt
                else:
                    raise ValueError(f"checkpoint tensor {tname} has no home on {mod_path}")
                del t
        gc.collect()  # one file's tensors released before the next opens

    # tied weights (e.g. a tied lm_head) are ABSENT from the checkpoint; retie
    # so they point at the attached embedding instead of a dead meta param
    model.tie_weights()

    _rebuild_computed_buffers(model, cfg, device)

    # leftover meta = a tensor nothing above accounted for. A parameter means
    # the checkpoint is missing a weight: refuse (zeros would serve garbage
    # with a 200 status). Buffers are computed values: materialize zeros as
    # the last-resort valve, but say so — never silently.
    left_p = [n for n, p in model.named_parameters() if p.is_meta]
    if left_p:
        raise ValueError(f"meta parameters after streaming (checkpoint incomplete?): {left_p}")
    left_b = []
    for mod in model.modules():
        for bn, b in list(mod._buffers.items()):
            if b is not None and b.is_meta:
                mod._buffers[bn] = torch.zeros(b.shape, dtype=b.dtype, device=device)
                left_b.append(bn)
    if left_b:
        print(f"[drinkme] zero-materialized leftover meta buffers: {left_b}")
    bad = [n for n, b in model.named_buffers() if b.is_meta]
    if bad:  # a silent meta tensor = garbage at first forward
        raise ValueError(f"meta tensors survived materialization: {bad}")

    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)


def build_compressed_model(repo: str, revision: str | None, pack_dir: str,
                           device: str = "cuda", rope_scaling: dict | None = None,
                           on_tensor=None, snap: str | None = None,
                           vision: bool | None = None):
    """The compressed MODULE TREE, and nothing else: load_compressed's model
    build (hash gate, one bound snapshot, meta skeleton, every packed
    Linear installed through swap.make_module straight off the pack, the
    raw remainder streamed by the one walker) factored out so the bench's
    pack-dir compressed arm (arms.load_pack_compressed)
    times exactly the modules `drinkme serve` installs — the artifact on
    disk, never a re-pack in memory — without an HFEngine's own
    allocations (KV cache, prefix slots, an MTP head) inside its resident
    figure. `on_tensor(name, pack)` is called with each RAW pack dict as it
    is installed (the arm's per-tensor stats: bpw, dtype, codec, format_version).
    `snap` is the bound snapshot when the caller already resolved it
    (load_compressed, whose tokenizer thread needs it first); None resolves
    it here. `vision` True builds the served vision tower
    (arms._VISION_TOWERS) into the tree, its Linears from the pack and the
    rest from the embedded checkpoint, which only a pack with a `vision`
    block carries (one without it is refused, naming the re-pack); False
    (DRINKME_VISION=0) builds none, its pack tensors skipped by name and
    nothing of it resident; None (the bench's arms) builds the tree with
    the tower for every architecture that serves one
    (arms._TOWER_BY_DEFAULT), so a --pack-dir measurement describes the
    full model.
    Returns (model, cfg, snap, pack_meta)."""
    from transformers import AutoConfig

    from ..codec.pack import _check_format_version, iter_pack_dir
    from ..codec.swap import make_module

    with open(os.path.join(pack_dir, "meta.json")) as f:
        pack_meta = json.load(f)
    # THE FORMAT GATE, first: a pack of a format version this build does not
    # read is refused by name before the pack is hashed or a snapshot
    # resolved — no other reader, no migration, `drinkme pack` again
    _check_format_version(pack_meta, pack_dir)

    # THE HASH GATE: proof against the source
    # lives at PACK time; load verifies the sha256 file hashes and the manifest —
    # the exact bytes that passed are the exact bytes served, at hash speed
    # instead of a double CPU decode of the model (measured: 49s of a 54s
    # Qwen3-8B load, and the decode-vs-own-output half could not even catch
    # a bit flip in the quantized stream; the hash catches every file-level
    # corruption). A pack without its hashes is refused, not served on less.
    from ..codec.pack import verify_hashes

    verify_hashes(pack_dir)  # raises loudly on a mismatched or missing hash

    # ONE snapshot: the pack names the checkpoint it was
    # cut from (meta.json `source`); config, tokenizer, every raw tensor and
    # the MTP head below all come from THAT resolved directory, and a
    # --model/--pack-dir pair whose identities disagree is refused here,
    # before a single weight is allocated. `repo` stays the model id. A
    # self-contained pack resolves to its own checkpoint/ directory
    # (checkpoint.resolve_pack_source), which the code below reads the same way.
    from .checkpoint import resolve_pack_source

    if snap is None:
        snap = resolve_pack_source(repo, revision, pack_meta, pack_dir)

    cfg = AutoConfig.from_pretrained(snap)  # the snapshot's own config.json
    _apply_rope_scaling(cfg, rope_scaling)  # rope scaling — before skeleton(cfg), so
    # the meta-device model's rotary module is built from the widened window
    from ..arms import has_vision_tower, skeleton, vision_tower_paths

    paths = vision_tower_paths(cfg)
    if vision and not (paths and pack_meta.get("vision")):
        from ..codec.pack import repack_command

        raise ValueError(f"{pack_dir}: this pack carries no vision tower (meta.json has no "
                         f"`vision` block); re-pack it: "
                         f"`{repack_command(pack_meta.get('hfRepo') or repo)}`, or serve "
                         "text only with DRINKME_VISION=0")
    model = skeleton(cfg, vision=vision)
    # the tree decides: a tower it does not hold has its pack tensors skipped
    skip = None if paths is None or has_vision_tower(model, cfg) else tuple(paths)

    # -- compressed Linears, straight from the pack, one tensor at a time --
    kinds: dict = {}
    schedules: dict = {}
    for name, pack in iter_pack_dir(pack_dir, skip=skip):  # skip: the tower, not built
        # a ROOT-level Linear (an untied lm_head) is named "lm_head", no
        # dot (eligible_linears emits dot-free root names), so rpartition
        # hands get_submodule "" for the parent and the root is taken below
        parent_path, _, child = name.rpartition(".")
        old = model.get_submodule(name)
        if not isinstance(old, torch.nn.Linear):
            raise ValueError(f"pack tensor {name} is not a Linear in this config")
        # bias=None even when the Linear has one: the pack carries weights
        # only; the bias streams from the checkpoint below (bias-after-swap).
        # Empty parent_path = the root module itself (get_submodule("") does
        # NOT mean root in this torch — it raises).
        parent = model.get_submodule(parent_path) if parent_path else model
        # the class follows the tensor's kind (swap.make_module):
        # RadixCompressedLinear / RawLinear — both CompressedLinear to
        # everything below
        mod = make_module(pack, None, device)
        setattr(parent, child, mod)
        if hasattr(mod, "p") and "rx_launch" in mod.p:
            fam = mod.p["rx_launch"]["family"]
            schedules[fam] = mod.p["rx_launch"]
        kinds[type(mod).__name__] = kinds.get(type(mod).__name__, 0) + 1
        if on_tensor is not None:
            on_tensor(name, pack)
    # the boot-log line for the pack: what it is and how it is launched
    # (the profile and its launch table row, docs/pack-format.md), so a
    # served number is attributable to (pack, code) from the log alone
    from ..codec.radix_schedule import Launch, box_arch, measured_note

    sched = "; ".join(f"{fam}: {Launch(**cfg).describe()} ({cfg['box_class']} row"
                      f"{'' if cfg['source'] == 'table' else ', ' + cfg['source']})"
                      for fam, cfg in schedules.items())
    # launch-table honesty: a row from the table is a measurement only on
    # the parts it was measured on (radix_schedule.MEASURED); on any other
    # device the line says which class's rows stood in. Silent on the CPU
    # route, which ignores the schedule.
    if device == "cuda" and sched and any(cfg["source"].startswith("table")
                                          for cfg in schedules.values()):
        boxes = {cfg["box_class"] for cfg in schedules.values()}
        note = measured_note(next(iter(boxes)), box_arch())
        if note:
            sched += f" ({note})"
    # the profile (sip / gulp) is the pack's name
    from ..codec.pack import embedded_dir

    raw_from = ("the pack's embedded checkpoint"
                if os.path.normpath(snap) == os.path.normpath(embedded_dir(pack_dir)) else snap)
    print(f"[drinkme] pack: format {pack_meta.get('formatVersion')}, "
          f"profile {pack_meta.get('profile')} (widths "
          f"{pack_meta.get('profileWidths')}), {', '.join(f'{k} x{v}' for k, v in kinds.items())}"
          f"{'; schedule ' + sched if sched else ''}; raw tensors from {raw_from}", flush=True)

    # -- everything else, streamed per-tensor from the checkpoint (the same
    # `snap` the pack is bound to, resolved above): the ONE walker, shared
    # with the bench's streaming stock loader (arms.load_stock_streaming) --
    stream_checkpoint(model, snap, cfg, device, name=repo)
    # NOT routed here: the kernel routes (serving/kernel_route.route_kernels
    # — the narrow GEMV, the DeltaNet recurrence) are the caller's, so the
    # bench's pack arm is routed where its stock and twin arms are
    # (arms.run_arms) and load_compressed routes where load_stock does
    return model, cfg, snap, pack_meta


def load_compressed(repo: str, revision: str | None, pack_dir: str,
                    device: str = "cuda", ctx: int | None = None,
                    prefix_slots: int | None = None,
                    rope_scaling: dict | None = None) -> HFEngine:
    """The fit path: meta skeleton + per-tensor streaming, BF16 weights of
    compressed Linears never materialized. Peak host memory = one tensor.
    The module tree is build_compressed_model's (above); this wraps it in
    the engine — tokenizer, generation defaults, the MTP head, KV."""
    # Tokenizer on a thread: ~11MB rust parse with zero dependency on the
    # model — it was serialized AFTER all 253 packs streamed (the hot-loop audit).
    from concurrent.futures import ThreadPoolExecutor

    from .checkpoint import resolve_pack_source

    with open(os.path.join(pack_dir, "meta.json")) as f:
        pack_meta = json.load(f)
    # the same resolution build_compressed_model makes (it is deterministic
    # and network-free once the snapshot is cached), so the tokenizer parse
    # can start on the bound snapshot before the model build begins
    snap = resolve_pack_source(repo, revision, pack_meta, pack_dir)
    _tok_ex = ThreadPoolExecutor(max_workers=1)
    tok_fut = _tok_ex.submit(_tokenizer, snap, None)

    # image input: decided before the tree is built, since it decides
    # whether the tree holds the vision tower (DRINKME_VISION, the
    # processor config; build_compressed_model refuses a pack whose
    # `vision` block is missing)
    from transformers import AutoConfig

    vis, tower, vision_reason = _vision_for(AutoConfig.from_pretrained(snap), snap, repo,
                                         tokenizer=tok_fut.result)
    model, cfg, snap, pack_meta = build_compressed_model(
        repo, revision, pack_dir, device, rope_scaling=rope_scaling, snap=snap,
        vision=tower is not None)
    # the runtime's kernel routes (serving/kernel_route.py): the narrow
    # GEMV and the DeltaNet recurrence — bound here, after the modeling
    # module is imported, before the head (whose verify replays the same
    # recurrence function per position); the ONE call load_stock and the
    # bench's arms make too
    route_kernels(model, device)

    meta = {k: pack_meta.get(k) for k in ("hfRepo", "revision", "meanBpw", "weightedBpw", "profile")}
    meta["revision"] = pack_revision(pack_meta)  # /v1/models' revision
    meta["source"] = pack_meta.get("source")  # the bound snapshot
    # sourceDtype: the released precision this engine serves exactly —
    # "bf16" (the lossless codec serves the release as released), as the
    # packer recorded it
    meta["sourceDtype"] = pack_meta.get("sourceDtype")
    # packId (on-disk prefix slots) is the manifest digest, the file hashes folded to one — the pack's own statement
    # of "these exact bytes". It keys the cold tier's directory so KV computed
    # against one pack can never be served against another, and it is NOT on
    # the wire: model_meta() names its fields one by one.
    meta["packId"] = identity.pack_id_from_meta(pack_meta)
    # resolvedRevision: the pack's own record of the
    # checkpoint revision it was made from — meta.json `source.revision`,
    # the resolved commit sha the packer writes — else the snapshot the
    # non-packed tensors were just streamed from. The file hashes name the
    # compressed tensors; this names the checkpoint beside them.
    source = pack_meta.get("source")
    recorded = source.get("revision") if isinstance(source, dict) else None
    from ..codec.pack import embedded_dir

    # an embedded checkpoint has no hub name of its own; it passed the
    # source identity check, so it IS the recorded commit — or, cut from a
    # local directory, the identity digest it was checked against (stable
    # across re-packs, unlike a digest of the pack's own files, which
    # carries their mtimes)
    if os.path.normpath(snap) == os.path.normpath(embedded_dir(pack_dir)):
        streamed = recorded or "local-" + str(source["digest"])[:16]
    else:
        streamed = resolved_revision(snap)
    if isinstance(recorded, str) and recorded:
        meta["resolvedRevision"] = recorded
        if recorded != streamed:
            print(f"[drinkme] pack records source revision {recorded[:12]} but "
                  f"the non-packed tensors streamed from {streamed[:12]} — "
                  "slots are keyed on the pack's record", file=sys.stderr,
                  flush=True)
    else:
        meta["resolvedRevision"] = streamed
    tok = tok_fut.result()  # propagate any tokenizer-load error here
    _tok_ex.shutdown(wait=False)
    # `snap` (above): the same resolved checkpoint dir the streamed tensors
    # came from — the compressed arm's model.generation_config never saw the
    # repo's file (module docstring: skeleton() builds from AutoConfig, not
    # from_pretrained), so this is the arm that most needs the file read here.
    gd = gen_config.load(model, snap)
    engine = HFEngine(model, tok, model_id=repo,
                      arm="compressed", meta=meta, ctx=_ctx(cfg, ctx),
                      # the head's raw norms from the SAME snapshot (a dir is
                      # its own snapshot_dir), never a second resolve of repo
                      mtp_head=_mtp_head(model, snap, None, device, pack_dir),
                      prefix_slots=prefix_slots, gen_defaults=gd,
                      vision=vis, tower=tower, vision_reason=vision_reason)
    # sleep/wake (load_stock's twin, same reasoning): the level-2 wake replays THIS
    # call, which is why "does a level-2 wake equal a cold load" has an
    # answer instead of an argument.
    engine._reload = lambda: load_compressed(repo, revision, pack_dir, device,
                                             ctx=ctx, prefix_slots=prefix_slots,
                                             rope_scaling=rope_scaling)
    return engine
