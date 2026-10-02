"""Qwen3.5's M-RoPE positions for a prompt that carries images.
Torch-free: numpy int64, computed once per request on the CPU over the
whole expanded prompt. The engine slices it per forward
(`torch.from_numpy(pos4[:, a:b])`).

A port of transformers' `Qwen3_5Model.get_rope_index`
(models/qwen3_5/modeling_qwen3_5.py), pinned against it by
tests/test_serving_mrope.py:

- A text run of length L takes `cur + arange(L)` in all three rows, then
  `cur += L`.
- An image run over a (t, h, w) patch grid covers the MERGED grid
  (t, h/m, w/m), in t, h, w order. T = cur + t_index, H = cur + row and
  W = cur + col. Then `cur += max(h, w) // m`.
- `delta = max over every position + 1 - len(ids)`. With no image it is
  0, and each image adds `max(h', w') - h'*w'` (h', w' merged). A
  2x3-merged image adds -3.
- `<|vision_start|>` and `<|vision_end|>` are ordinary text tokens.

The text model takes four rows (`Qwen3_5TextModel.forward`): row 0 the
plain sequential position (mask building, and what the decoder layers get
as `position_ids`), rows 1-3 the t/h/w positions the rotary embedding
interleaves. `rope_positions` returns all four, with row 0 = arange(T),
exactly what the model builds itself when it is passed no position_ids at
all. transformers' own wrapper passes only rows 1-3, which leaves row 0
None. The two agree under a cache. The mask builder reads row 0 only to
detect packed sequences, and it skips that when a cache is present.
Otherwise row 0 only reaches the attention function's kwargs: SDPA and the
functions drinkme registers ignore it, and flash-attention would find one
sequence in it.

With no image in the prompt all four rows are `arange(T)` and delta is
0: exactly today's positions. The engine's text-only rule is stricter
still: pass no position_ids at all, so a text request's call is unchanged
byte for byte.

Past the prompt (decode, draft verify, MTP re-arm), position p is
`[p, p + delta, p + delta, p + delta]` (`step_positions`). That is what
the reference's `compute_3d_position_ids` does with its stored
`rope_deltas`, and it is also what the prompt's own trailing text holds.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


def image_token_count(grid_thw: Sequence[int], merge_size: int = 2) -> int:
    """How many placeholder tokens a (t, h, w) patch grid expands to:
    t * (h / m) * (w / m)."""
    t, h, w = (int(x) for x in grid_thw)
    return t * (h // merge_size) * (w // merge_size)


def rope_positions(ids: Sequence[int], grids: Sequence[Sequence[int]], image_token_id: int,
                   merge_size: int = 2) -> tuple[np.ndarray, int]:
    """(pos4, delta) for an expanded prompt.

    `ids` is the prompt as the model sees it, each image's placeholder
    already expanded to `image_token_count(grid)` copies of
    `image_token_id`. `grids` holds one (t, h, w) per image in prompt
    order: PreparedImage.grid_thw. Returns pos4 as int64 [4, len(ids)]
    and delta as a Python int.

    Image runs are cut by count, not by adjacency: image k owns the next
    `image_token_count(grids[k])` placeholder ids from wherever its run
    starts. The template always puts `<|vision_end|><|vision_start|>`
    between two images, so this agrees with the reference, which groups
    maximal runs. A mismatch (a run shorter than its grid says, a
    placeholder with no image left, an image with no placeholder left)
    raises ValueError: the prompt and its images disagree."""
    ids = np.asarray(ids, dtype=np.int64).reshape(-1)
    n = ids.shape[0]
    pos = np.empty((4, n), dtype=np.int64)
    pos[0] = np.arange(n)
    is_img = ids == image_token_id
    img_at = np.flatnonzero(is_img)
    i = cur = used = 0  # next id to place, next rope position, placeholders consumed
    for k, grid in enumerate(grids):
        t, h, w = (int(x) for x in grid)
        count = image_token_count((t, h, w), merge_size)
        if count < 1:
            raise ValueError(f"image {k}: grid {tuple(grid)} expands to no tokens")
        if used >= img_at.shape[0]:
            raise ValueError(f"image {k}: no placeholder run left in the prompt for it "
                             f"({len(grids)} images, {img_at.shape[0]} placeholder ids)")
        s = int(img_at[used])
        e = s + count
        if e > n or not is_img[s:e].all():
            raise ValueError(f"image {k}: its placeholder run at {s} is shorter than the "
                             f"{count} tokens its grid {tuple(grid)} expands to")
        pos[1:, i:s] = cur + np.arange(s - i)
        cur += s - i
        lt, lh, lw = t, h // merge_size, w // merge_size
        tt, hh, ww = np.meshgrid(np.arange(lt), np.arange(lh), np.arange(lw), indexing="ij")
        pos[1, s:e] = cur + tt.reshape(-1)
        pos[2, s:e] = cur + hh.reshape(-1)
        pos[3, s:e] = cur + ww.reshape(-1)
        cur += max(h, w) // merge_size
        used += count
        i = e
    if used != img_at.shape[0]:
        raise ValueError(f"{img_at.shape[0] - used} placeholder ids in the prompt have no image "
                         f"({len(grids)} images given)")
    pos[1:, i:] = cur + np.arange(n - i)
    delta = int(pos[1:].max()) + 1 - n if n else 0
    return pos, delta


def step_positions(start: int, count: int, delta: int) -> np.ndarray:
    """int64 [4, count] for the positions start .. start+count-1 past the
    prompt: row 0 the position itself, rows 1-3 the position plus delta."""
    p = np.arange(start, start + count, dtype=np.int64)
    out = np.empty((4, count), dtype=np.int64)
    out[0] = p
    out[1:] = p + delta
    return out
