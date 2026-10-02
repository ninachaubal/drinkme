"""Chunked prefill: every prompt forward is at most CHUNK tokens
(docs/serve.md#chunked-prefill).

WHY. On a desktop APU the compositor and drinkme share one GPU, and amdgpu
resets the graphics ring when a job waits past `lockup_timeout` (2000 ms
by default) — which kills the desktop session and every open app with it.
Over ONE forward of the whole uncached suffix, the longest dispatch grows
with the suffix. Measured per dispatch with
`rocprofv3 --kernel-trace` at Qwen3.8-27B shapes on Strix Halo (gfx1151;
bench/prefill_dispatch.py): the causal attention kernel
takes 1.42e-6 ms x T^2 — 382 ms at T=16,384, ~24 s extrapolated to a
131,072-token prompt — and the largest GEMM, the DeltaNet chunk kernels
and the rest grow linearly (the GEMM: 102 ms at T=16,384). A forward over
C tokens bounds all of those by C; attention over a long CACHED prefix is
bounded separately, by serving/segmented_attention.py.

HOW. `run` walks ids[start:] in spans of `chunk` tokens, each an extend of
the same cache at its own cache_position — the extends-only reuse
engines.py already does across requests, so the DeltaNet recurrent and conv
state carry across chunks exactly as they carry across turns. Only the LAST
span computes the logits row; the spans before it call the decoder
without lm_head. For MTP every span's hidden states go to `on_hidden`, which
seeds the draft head span by span (mtp.Speculator.seed). `chunk` = 0 is the
whole-prompt path, kept for A/B: one call over the suffix, and no
segmented attention — split only at the context
checkpoints' positions (serving/ctx_checkpoints.py) when a slot takes
them. A prompt with images
(serving/image_prompt.py) feeds each span's embeddings and M-RoPE positions
instead of its ids, and no span boundary cuts an image run that
ImagePrompt.whole lists (`spans`). On gemma-4 a span holding an image run also takes that run's bidirectional
sliding-window mask (ImagePrompt.masks); segmented attention never
applies there (its layout check refuses sliding-window layers), so the two
masks never meet.

Numerics: a chunked prefill batches differently from a whole one, so its
logits agree to reduction order — the "chunked-prefill class" engines.py
assigns to warm-vs-cold already.
"""

from __future__ import annotations

import os
import sys

import torch

from .. import exitcodes

ENV = "DRINKME_PREFILL_CHUNK"
# Tokens per prefill forward. At 4096 the longest dispatch measured at the
# 27B's shapes is 27 ms (the MLP GEMM and the cold causal attention, both
# ~26 ms), with 8192 at 98 ms and 16384 over the 250 ms budget. The
# throughput cost of smaller chunks is in docs/serve.md#chunked-prefill.
DEFAULT_CHUNK = 4096


def chunk_from_env() -> int:
    """DRINKME_PREFILL_CHUNK: tokens per prefill forward; 0 = the whole
    prompt in one forward (the unchunked path). Unset = DEFAULT_CHUNK."""
    raw = os.environ.get(ENV, "").strip()
    if not raw:
        return DEFAULT_CHUNK
    try:
        n = int(raw)
    except ValueError:
        n = -1
    if n < 0:
        raise exitcodes.CantRunHere(f"[drinkme] {ENV}={raw!r}: expected a token count >= 0 "
                                    "(0 = whole-prompt prefill)")
    return n


def describe(chunk: int) -> str:
    if chunk == 0:
        return f"whole prompt in one forward ({ENV}=0)"
    from .segmented_attention import SPLIT_ABOVE, WORK

    return (f"chunks of {chunk} tokens; attention over a long cache split above "
            f"{SPLIT_ABOVE:,} query-key pairs into {WORK:,}-pair dispatches "
            f"({ENV}=0 for one forward)")


def spans(start: int, end: int, chunk: int, whole=(), cuts=()) -> list[tuple[int, int]]:
    """ids[start:end] in forwards of at most `chunk` tokens. `whole` holds
    [s, e) runs no boundary may cut, sorted and disjoint: an image's
    placeholder run (serving/image_prompt.py). A boundary that would fall
    inside one moves back to the run's start; a run that starts where its
    span does, and so is at least a chunk long, is a span of its own. On a
    causal tower (Qwen3.5, Muse-Glimmer) only runs that fit the chunk are
    listed (ImagePrompt.whole), so the longest forward is still the chunk:
    at the default pixel cap an image is at most 3,600 tokens on Qwen3.5
    and 4,096 on Muse-Glimmer, inside the default 4,096-token chunk, and a
    longer one is cut like text. A bidirectional tower's runs (gemma-4, at
    most 1,120 tokens) are all listed.

    `cuts` are positions a span must END at: the context checkpoints
    (serving/ctx_checkpoints.py), which are never inside a `whole` run. They
    apply with chunk 0 too, so the whole-prompt path is one forward per
    checkpoint span."""
    if chunk <= 0 or end - start <= chunk:
        out = [(start, end)]
    elif not whole:
        out = [(a, min(end, a + chunk)) for a in range(start, end, chunk)]
    else:
        out = []
        a = start
        while a < end:
            b = min(end, a + chunk)
            for s, e in whole:
                if s < b < e:
                    b = s if s > a else e
                    break
            out.append((a, b))
            a = b
    for c in sorted(cuts):
        for i, (a, b) in enumerate(out):
            if a < c < b:
                out[i:i + 1] = [(a, c), (c, b)]
                break
    return out


def decoder(model):
    return model.get_decoder() if hasattr(model, "get_decoder") else model.model


def run(model, ids: list[int], start: int, cache, device, chunk: int, *,
        hidden: bool = False, on_hidden=None, image=None, stops=(), at_stop=None):
    """Prefill ids[start:] into `cache`; return the last position's logits
    row [V]. With hidden=True (MTP) the trunk's hidden states of every span
    go to `on_hidden(h [1, t, H], a, b)` — the last span's too — and the
    last row's logits come from mtp.forward_with_hidden's narrowed
    lm_head, the one the whole-prompt path used.

    `image` (serving/image_prompt.ImagePrompt) is None for a text prompt,
    and then every call below is the one it always was. With one, `ids` are
    its expanded ids, no span cuts a run ImagePrompt.whole lists, and each
    forward takes `inputs_embeds` (the tower's output over the image rows),
    the M-RoPE `position_ids` (Qwen3.5), and a span holding a bidirectional
    run its attention_mask mapping (gemma-4), instead of the ids.

    `stops` are positions a span ends at (spans' `cuts`), and `at_stop(p)`
    is called once the cache holds ids[:p]: engines.py takes a context
    checkpoint there (serving/ctx_checkpoints.py)."""
    from . import mtp, segmented_attention

    n = len(ids)
    todo = (spans(start, n, chunk, cuts=stops) if image is None
            else spans(start, n, chunk, image.whole(start, chunk), cuts=stops))
    base = decoder(model)
    logits = None
    for a, b in todo:
        if image is None:
            inp, x = torch.tensor([ids[a:b]], dtype=torch.long, device=device), {}
        else:
            inp, x = None, image.inputs(base, a, b, device, cache)
        cpos = torch.arange(a, b, device=device)
        mask = (segmented_attention.prefill_mask(base, cache, b - a)
                if chunk > 0 else None)
        if mask is not None and "attention_mask" in x:
            # never reached: segmented attention refuses every layout with
            # sliding layers, the only ones an image mask is built for
            raise ValueError("a segmented prefill span and an image run's mask met; "
                             "neither is dropped silently")
        kw = {} if mask is None else {"attention_mask": mask}
        kw.update(x)
        last = b == n
        if hidden:
            if last:
                h, row = mtp.forward_with_hidden(model, inp, cache, cpos,
                                                 last_row_only=True, **kw)
                logits = row[-1]
            else:
                h = base(mtp.trunk_ids(model, inp), past_key_values=cache, use_cache=True,
                         cache_position=cpos, **kw).last_hidden_state
            on_hidden(h, a, b)
        elif last:
            # logits_to_keep=1: engines.py's prefill comment (one row, never
            # [n, vocab])
            logits = model(inp, past_key_values=cache, use_cache=True,
                           cache_position=cpos, logits_to_keep=1, **kw).logits[0, -1]
        else:
            # the text model itself, so the ids as the wrapper embeds them
            # (mtp.trunk_ids: gemma-4's and Muse-Glimmer's placeholder ids)
            base(mtp.trunk_ids(model, inp), past_key_values=cache, use_cache=True,
                 cache_position=cpos, **kw)
        if at_stop is not None and b in stops:
            at_stop(b)
    return logits


def announce(chunk: int) -> None:
    print(f"[drinkme] prefill: {describe(chunk)}", file=sys.stdout, flush=True)
