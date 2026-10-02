"""Prefill attention over a long cache, split into bounded dispatches
(docs/serve-kernels.md#segmented-attention-over-a-long-cache).

One SDPA call for a prefill chunk of C queries over L cached keys is one GPU
dispatch whose run time grows with C * (L + C): measured at Qwen3.8-27B
shapes on Strix Halo (gfx1151, AOTriton `attn_fwd`), 4.3e-6 ms per
query-key pair — 295 ms at C=2048 over 32K keys, ~2.3 s extrapolated to
262,144. Chunking the queries does not bound it: one program walks every key
of its query block, and a single query block over 262,144 keys already takes
104 ms (the M=1 decode step). A dispatch that holds the CUs past amdgpu's
2 s `lockup_timeout` resets the ring and kills the desktop's GPU contexts.

So, above SPLIT_ABOVE query-key pairs, the key range is split. The chunk's own keys
(the last C) form a C x C causal square; the cached prefix is cut into
segments of `segment` keys (WORK query-key pairs each), each attended
WITHOUT a mask. Every piece returns
its output and its log-sum-exp, and the pieces combine exactly the way
flash-decoding combines its splits:

    lse = logaddexp(lse_a, lse_b)
    out = out_a * exp(lse_a - lse) + out_b * exp(lse_b - lse)

in fp32, rounded once to the activation dtype at the end. Grouped-query heads
are folded into the query rows for the prefix segments (the n_rep query
heads that share a KV head become n_rep * C rows over that one head), so the
cached K/V are read in place: no repeat_kv copy of the cache and no
materialized [C, L + C] mask, both of which the stock path builds per layer.

Numerics: the same softmax, a different reduction order — each piece's
output is rounded to bf16 before the fp32 combine. The "chunked-prefill
class" engines.py's docstring already assigns to warm-vs-cold.

HOW IT IS SELECTED. The engine's prefill (serving/prefill.py) passes
`prefill_mask(...)` as the model's attention-mask mapping. It returns None —
today's kernels, today's masks — unless the forward is a multi-row extend
over more than SPLIT_ABOVE query-key pairs on a layout this module understands: only
full_attention / linear_attention layers, every full-attention layer a
LiveStaticLayer with one shared live length, every attention module on
SDPA. Then its full_attention entry is a `SegmentedCausal` marker, and the
attention function the modules look up (transformers' "sdpa" entry, wrapped
by `register`, or serving/suffix_attention's adapter) sees the marker and
routes here. Nothing else ever carries the marker, so every other forward —
decode, MTP verify, short prompts — runs the kernels it ran before.

gemma-4 is outside that layout (its sliding_attention layers), so its
prefill is never segmented, with or without images. That refusal is also
what its image prompts rely on: a span holding an image run brings its own
mask mapping, bidirectional within the run on the sliding layers
(serving/image_prompt.py), and prefill.run refuses a span that would carry
both.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

# Query-key pairs per attention dispatch. At the 27B's shape (24 query heads
# x 256, GQA 6:1 folded) one prefix segment of 2**25 pairs measured 56-57 ms
# for chunks of 1024-4096 queries on Strix Halo (95 ms at 8192, where
# the per-row cost of a 49,152-row call dominates) — under a quarter of the
# 250 ms budget, leaving room for the ~2x a shape's FIRST dispatch measured
# over its steady run (552 vs 295 ms) and for a wider head. Smaller costs
# throughput: every segment adds two fp32 passes over the chunk's output.
WORK = 1 << 25
# Split only above this many query-key pairs. Below it today's call runs —
# the masked one, which costs ~3.9e-6 ms per pair at the 27B's shape (2.3x a
# folded segment: it materializes the [C, L + C] mask and copies K/V to every
# query head), so 2**23 pairs is ~33 ms there.
SPLIT_ABOVE = 1 << 23
# the smallest prefix segment: below this the per-segment combine costs more
# than the attention it bounds
MIN_SEGMENT = 1024


@dataclass(frozen=True)
class SegmentedCausal:
    """The mask mapping's full_attention entry for a segmented forward:
    `rows` new queries, `past` keys cached before them (the queries sit at
    positions past..past+rows-1, lower-right causal), and `segment` prefix
    keys per dispatch."""
    rows: int
    past: int
    segment: int


def segment_for(rows: int, work: int | None = None) -> int:
    """Prefix keys per dispatch for `rows` queries: WORK pairs' worth."""
    work = WORK if work is None else work
    return max(MIN_SEGMENT, work // max(1, rows))


def needed(rows: int, past: int, above: int | None = None) -> bool:
    """True when one attention dispatch over this extend would exceed
    `above` (SPLIT_ABOVE) query-key pairs — the only case that leaves
    today's kernels."""
    above = SPLIT_ABOVE if above is None else above
    return rows > 1 and past > 0 and rows * (past + rows) > above


def _partial(q, k, v, causal: bool, scale: float):
    """(out [B, H, M, D] in q's dtype, lse [B, H, M] fp32) for one piece."""
    if q.device.type == "cpu":
        out, lse = torch.ops.aten._scaled_dot_product_flash_attention_for_cpu(
            q, k, v, 0.0, causal, scale=scale)[:2]
    else:
        out, lse = torch.ops.aten._scaled_dot_product_efficient_attention(
            q, k, v, None, True, 0.0, causal, scale=scale)[:2]
    return out, lse[..., :q.shape[-2]].float()


def attend(query, key, value, past: int, segment: int, scale: float | None = None):
    """Lower-right causal attention of `query` [B, H, M, D] over `key`/`value`
    [B, KV, past + M, D], in pieces of at most M x M and M x `segment` pairs.
    Returns [B, H, M, D] in query's dtype."""
    B, H, M, D = query.shape
    KV = key.shape[1]
    if H % KV:
        raise ValueError(f"{H} query heads over {KV} KV heads")
    if key.shape[-2] != past + M:
        raise ValueError(f"segmented attention: {key.shape[-2]} keys, expected "
                         f"{past} cached + {M} new")
    rep = H // KV
    scale = 1.0 / math.sqrt(D) if scale is None else scale
    # the chunk's own keys: a square, so top-left causal IS lower-right
    ko, vo = key[:, :, past:], value[:, :, past:]
    if rep > 1:
        ko, vo = ko.repeat_interleave(rep, 1), vo.repeat_interleave(rep, 1)
    out, lse = _partial(query, ko, vo, True, scale)
    # everything below is in the FOLDED layout [B, KV, rep * M, D]: head h
    # reads KV head h // rep, the grouping repeat_interleave (transformers'
    # repeat_kv) uses, so the n_rep query heads of one KV head are rows of
    # one attention over that head's keys, read in place
    acc = out.float().reshape(B, KV, rep * M, D)
    lse = lse.reshape(B, KV, rep * M, 1)
    qf = query.reshape(B, KV, rep * M, D)
    for a in range(0, past, segment):
        b = min(past, a + segment)
        o, l = _partial(qf, key[:, :, a:b], value[:, :, a:b], False, scale)
        l = l.unsqueeze(-1)
        new = torch.logaddexp(lse, l)
        # acc <- acc * exp(lse - new) + o * exp(l - new), in place: the two
        # passes over [rep * M, D] fp32 are what a segment costs beyond its
        # own attention
        acc.mul_(torch.exp(lse - new)).addcmul_(o, torch.exp(l - new))
        lse = new
    return acc.reshape(B, H, M, D).to(query.dtype)


def attention_forward(module, query, key, value, attention_mask, dropout=0.0,
                      scaling=None, **kwargs):
    """The marker's path, in transformers' attention-function shape."""
    m = attention_mask
    if query.shape[-2] != m.rows or key.shape[-2] != m.past + m.rows:
        raise ValueError("segmented prefill mask does not match the live KV window")
    out = attend(query, key, value, m.past, m.segment, scaling)
    return out.transpose(1, 2).contiguous(), None


def _dispatch(inner):
    def sdpa_or_segmented(module, query, key, value, attention_mask, *args, **kwargs):
        if isinstance(attention_mask, SegmentedCausal):
            return attention_forward(module, query, key, value, attention_mask, *args, **kwargs)
        return inner(module, query, key, value, attention_mask, *args, **kwargs)

    sdpa_or_segmented.__wrapped__ = inner
    return sdpa_or_segmented


def register() -> None:
    """Wrap transformers' "sdpa" attention function so the marker reaches
    `attention_forward`; everything else passes through untouched.
    Idempotent."""
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    fn = ALL_ATTENTION_FUNCTIONS["sdpa"]
    if getattr(fn, "__name__", "") != "sdpa_or_segmented":
        ALL_ATTENTION_FUNCTIONS.register("sdpa", _dispatch(fn))


def prefill_mask(base, cache, rows: int, work: int | None = None,
                 above: int | None = None):
    """The attention-mask mapping for one prefill forward of `rows` tokens
    on `base` (the decoder, pre-lm_head) over `cache`, or None for today's
    masks. Module docstring: the marker only where it is understood."""
    from . import suffix_attention
    from .kvcache import LiveStaticLayer

    cfg = getattr(base, "config", None)
    kinds = getattr(cfg, "layer_types", None)
    if cache is None or not kinds or any(
            k not in ("full_attention", "linear_attention") for k in kinds):
        return None
    full = [i for i, k in enumerate(kinds) if k == "full_attention"]
    layers = getattr(base, "layers", None)
    if not full or layers is None or len(cache.layers) != len(kinds):
        return None
    if any(type(cache.layers[i]) is not LiveStaticLayer for i in full):
        return None
    past = cache.layers[full[0]].live
    if any(cache.layers[i].live != past for i in full) or not needed(rows, past, above):
        return None
    impls = [getattr(getattr(getattr(layers[i], "self_attn", None), "config", None),
                     "_attn_implementation", None) for i in full]
    if any(impl not in ("sdpa", suffix_attention.NAME) for impl in impls):
        return None
    register()
    return {"full_attention": SegmentedCausal(rows, past, segment_for(rows, work)),
            "linear_attention": None}
