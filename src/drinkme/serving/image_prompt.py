"""One request's images, as the engine's forwards need them.

The chat template renders each image as ONE placeholder token (Qwen3.5:
`<|vision_start|><|image_pad|><|vision_end|>`; gemma-4: `<|image|>`). An
ImagePrompt turns that rendered prompt into the four things generate()
needs:

- `ids`: the prompt the model sees, each placeholder EXPANDED to its image's
  `tokens` copies (transformers does this in the processor,
  Qwen3VLProcessor.replace_image_token), between the tower's `boi`/`eoi`
  markers where it has them (Gemma4Processor.replace_image_token:
  `<|image>` + N x `<|image|>` + `<image|>`). An image's RUN is its
  `tokens` rows, markers excluded. The rendered prompt must carry exactly
  one placeholder per image, or the request is refused.
- `key_ids`: `ids` with each image's run replaced by its
  PreparedImage.prefix_key, which is what the prefix cache and the cold tier
  key on. The placeholder id is the same for every image, so two same-size
  screenshots render to identical ids; keyed on `ids`, the second request
  would reuse KV computed from the first image and answer about it, with a
  200. The keys are negative, so they never equal a vocabulary id, and the
  n-gram proposer never proposes one (serving/ngram.py).
- positions: Qwen3.5's M-RoPE rows (serving/mrope.py), sliced per forward;
  past the prompt, position p is [p, p + delta, p + delta, p + delta].
- embeddings: per forward, embed_tokens over the ids with each image's rows
  replaced by the tower's output (the reference's masked_scatter, one span at
  a time, so no [T, hidden] tensor is ever built for the whole prompt). The
  tower's rows go in as they come: gemma-4 scales its token embeddings by
  sqrt(hidden) inside embed_tokens and scatters the image features without
  that scale (modeling_gemma4.Gemma4Model.forward), which is what
  overwriting embed_tokens' output gives.
- the mask, for a tower whose runs attend BIDIRECTIONALLY (gemma-4,
  `use_bidirectional_attention: "vision"`): on the sliding-window layers an
  image's tokens attend forward within their own run during prefill,
  AND(sliding window, OR(causal, same run)); the full-attention layers
  stay causal (modeling_gemma4.create_masks_for_vision_model). A span that
  holds a run is handed that mapping as its attention_mask
  (`masks`); every other forward builds its own masks as a text one does.

A text-only request never builds one: the engine passes no position_ids and
no inputs_embeds at all, so its calls are the ones it made before images
existed.

LIFETIME. The tower runs lazily, once per distinct image (by digest), when
the first span that needs the image's rows is prefilled. Its output stays on
the device until the span that holds the run's last row, and the image's
pixel_values are released (PreparedImage.release) as soon as the tower has
read them. An image whose run lies entirely inside the reused prefix never
runs the tower, since its KV is already in the slot, and is released at
`begin`.

HOW A RUN MAY BE PREFILLED depends on one fact, how the text model attends
among the run's rows (vision.Tower.bidirectional):

- CAUSAL (Qwen3.5, Muse-Glimmer): a row sees only the rows before it, as
  text does, so a run may be prefilled in pieces. Chunked prefill keeps a
  run that fits the chunk inside one span (`whole`, prefill.spans), so such
  an image's output is made and dropped inside one span; a run longer than
  the chunk (a cap raised past ~4,096 tokens) is cut into chunk-sized spans
  like text. A reused prefix (a cold-tier restore) may end inside a run.
  Both prefill part of a run from the whole image's output, which is exact.
- BIDIRECTIONAL (gemma-4's sliding layers): a run's first rows see its
  last, so neither holds. `whole` keeps every run inside one span whatever
  its length, and `cuts` names a reuse point inside a run (its rows below
  the point were written without seeing the rest), for which the engine
  prefills cold. The engine never makes such a reuse point: a slot ends at
  a generated token, a cold-tier restore at a stored slot's end, and a
  rewind or context checkpoint (serving/ctx_checkpoints.py) at a point
  `fit` moved out of every run; the check keeps it so.
"""

from __future__ import annotations

import sys

import numpy as np
import torch

from . import mrope


def step_rows(pos, delta: int):
    """[4, 1, T] position_ids for the positions `pos` (a 1-D long tensor)
    past an image prompt: row 0 the position, rows 1-3 the position plus
    delta. Built on pos's device, so the draft chain stays sync-free."""
    d = pos + delta
    return torch.stack((pos, d, d, d)).unsqueeze(1)


class ImagePrompt:
    """The rendered prompt `rendered` (one placeholder per image) and the
    request's `images` (GenerationRequest.images, in template order), for
    the served `model`, whose tower is `tower` (vision.Tower). Raises
    ValueError when the two disagree."""

    def __init__(self, rendered: list[int], images, tower, model):
        tid = tower.image_token_id
        at = [i for i, t in enumerate(rendered) if t == tid]
        if len(at) != len(images):
            raise ValueError(f"the rendered prompt carries {len(at)} image placeholders for "
                             f"{len(images)} images")
        before = [] if tower.boi is None else [tower.boi]
        after = [] if tower.eoi is None else [tower.eoi]
        ids: list[int] = []
        keys: list[int] = []
        runs = []
        prev = 0
        for i, img in zip(at, images):
            if img.architecture != tower.architecture:
                raise ValueError(f"{img!r} was prepared for {img.architecture}, and this "
                                 f"model's tower is {tower.architecture}")
            ids += rendered[prev:i] + before
            keys += rendered[prev:i] + before
            s = len(ids)
            ids += [tid] * img.tokens
            keys += [img.prefix_key] * img.tokens
            runs.append((s, len(ids), img))
            ids += after
            keys += after
            prev = i + 1
        ids += rendered[prev:]
        keys += rendered[prev:]
        self.ids, self.key_ids, self.runs = ids, keys, runs
        # (rendered index, how many ids its expansion adds) per placeholder
        self._grow = [(i, len(before) + img.tokens + len(after) - 1)
                      for i, img in zip(at, images)]
        self.tower, self.model = tower, model
        self.n = len(ids)
        if tower.mrope:
            self.pos4, self.delta = mrope.rope_positions(
                ids, [img.grid_thw for img in images], tid, tower.merge_size)
        else:
            self.pos4, self.delta = None, 0
        # the last row each distinct image is needed for, and its output while
        # a span still needs it
        self._last: dict[str, int] = {}
        for _s, e, img in runs:
            self._last[img.digest] = max(e, self._last.get(img.digest, 0))
        self._feats: dict = {}
        self.tower_runs = 0
        self.skipped = 0

    @property
    def rope(self) -> bool:
        """Does the text model take this prompt's positions explicitly?"""
        return self.pos4 is not None

    def begin(self, lcp: int) -> None:
        """The prefix cache has answered: rows below `lcp` are in the slot.
        Every image whose runs all end there never needs the tower, so its
        pixel_values go now."""
        for d, end in self._last.items():
            if end <= lcp:
                self.skipped += 1
                for _s, _e, img in self.runs:
                    if img.digest == d:
                        img.release()

    def cuts(self, lcp: int) -> bool:
        """Would reusing the cache's first `lcp` rows leave an image run half
        written? Only on a bidirectional tower (module docstring), where the
        run's rows below lcp were computed without seeing the rest."""
        return self.tower.bidirectional and any(s < lcp < e for s, e, _img in self.runs)

    def expanded(self, rendered: int) -> int:
        """A position in the rendered prompt as a position in `ids`: past each
        placeholder before it, its expansion's extra ids."""
        return rendered + sum(g for i, g in self._grow if i < rendered)

    def fit(self, n: int) -> int:
        """The largest reuse point at or below n that `cuts` nothing: n, or
        the start of the bidirectional run n falls inside. A rewind or a
        context checkpoint (serving/ctx_checkpoints.py) reuses a prefix only
        at such a point; a causal run may be cut anywhere."""
        if self.tower.bidirectional:
            for s, e, _img in self.runs:
                if s < n < e:
                    return s
        return n

    def whole(self, start: int, chunk: int = 0) -> list[tuple[int, int]]:
        """The image runs from `start` on, clipped to it: the [s, e) ranges
        a prefill span boundary never cuts (serving/prefill.spans). On a
        causal tower (module docstring), a run longer than `chunk` (> 0) is
        not one of them: kept whole, it would be one forward past the chunk,
        and at 8,160 tokens (a 4K image with the cap raised) the 27B's
        longest dispatch in it measured 107 ms on gfx1151
        (bench/vision_dispatch.py --engine), against 26 ms at the default
        cap. It is prefilled in chunk-sized spans like text. A bidirectional
        tower's runs are all listed, whatever their length."""
        out = []
        for s, e, _img in self.runs:
            s = max(s, start)
            if e > s and (self.tower.bidirectional or not 0 < chunk < e - s):
                out.append((s, e))
        return out

    def positions(self, a: int, b: int, device):
        """position_ids for prompt rows a..b-1: [4, 1, b - a] long, or None
        when the text model builds its own (no M-RoPE)."""
        if self.pos4 is None:
            return None
        return torch.from_numpy(np.ascontiguousarray(self.pos4[:, a:b])).view(4, 1, b - a).to(device)

    def inputs(self, base, a: int, b: int, device, cache=None) -> dict:
        """The decoder's inputs for prompt rows a..b-1 over `cache`:
        `inputs_embeds`, `position_ids` when the model takes them, and the
        `attention_mask` mapping when a bidirectional run lies in the span
        (`masks`)."""
        out = {"inputs_embeds": self.embeds(base, a, b, device)}
        pos = self.positions(a, b, device)
        if pos is not None:
            out["position_ids"] = pos
        mask = self.masks(base, a, b, cache, out["inputs_embeds"])
        if mask is not None:
            out["attention_mask"] = mask
        return out

    def masks(self, base, a: int, b: int, cache, x):
        """The attention-mask mapping for prompt rows a..b-1 (their
        embeddings `x`) over `cache`, or None when the span needs none of
        its own: a causal tower, or no image run in the span. Built the way
        transformers builds gemma-4's (create_masks_for_vision_model): the
        full-attention entry is the causal mask a text forward gets, the
        sliding entry create_causal_mask with the run overlay OR'd onto the
        causal mask and the window AND'd over both. It is sized against the
        first sliding layer, as a text forward's sliding mask is: the
        reference sizes both entries against a full layer, which agrees
        only when there is no cache. The run overlay is indexed by absolute
        position, each run its own block, text -1."""
        if not self.tower.bidirectional:
            return None
        runs = [(s, e) for s, e, _img in self.runs if s < b and e > a]
        if not runs:
            return None
        from transformers.masking_utils import (blockwise_overlay, create_causal_mask,
                                                sliding_window_overlay)

        cfg = base.config
        sliding = cfg.layer_types.index("sliding_attention")
        kv_length, kv_offset = cache.get_mask_sizes(b - a, sliding)
        blocks = torch.full((1, max(b, kv_offset + kv_length)), -1, dtype=torch.long,
                            device=x.device)
        for k, (s, e) in enumerate(runs):
            blocks[0, s:e] = k
        common = dict(config=cfg, inputs_embeds=x, attention_mask=None, past_key_values=cache)
        return {"full_attention": create_causal_mask(**common),
                "sliding_attention": create_causal_mask(
                    **common, or_mask_function=blockwise_overlay(blocks),
                    and_mask_function=sliding_window_overlay(cfg.sliding_window),
                    layer_idx=sliding)}

    def embeds(self, base, a: int, b: int, device):
        """[1, b - a, hidden]: the token embeddings of rows a..b-1, each
        image row replaced by the tower's output for it. Outputs no later
        span needs are dropped on the way out."""
        x = base.get_input_embeddings()(torch.tensor([self.ids[a:b]], dtype=torch.long,
                                                     device=device))
        for s, e, img in self.runs:
            lo, hi = max(s, a), min(e, b)
            if lo < hi:
                x[0, lo - a:hi - a] = self._features(img)[lo - s:hi - s].to(x.dtype)
        for d in [d for d in self._feats if self._last[d] <= b]:
            del self._feats[d]
        return x

    def _features(self, img):
        f = self._feats.get(img.digest)
        if f is None:
            f = self._feats[img.digest] = self.tower.features(self.model, img)
            self.tower_runs += 1
            for _s, _e, other in self.runs:
                if other.digest == img.digest:
                    other.release()
        return f

    def step(self, buf, p: int) -> None:
        """Write the decode step's position_ids for position p into `buf`
        ([4, 1, 1] long, preallocated): the hot loop allocates nothing."""
        buf[0].fill_(p)
        buf[1:].fill_(p + self.delta)

    def head_positions(self, a: int, b: int, device):
        """The MTP head's rope rows for entries a..b-1 inside the prompt
        ([3, 1, b - a]): entry p sits at the trunk's position p."""
        pos = self.positions(a, b, device)
        return None if pos is None else pos[1:]

    def announce(self, where: str = "") -> None:
        n_tok = sum(e - s for s, e, _img in self.runs)
        print(f"[drinkme.engine] images{where}: {len(self.runs)} in the prompt "
              f"({n_tok} image tokens); the tower ran {self.tower_runs}x, "
              f"{self.skipped} inside the reused prefix", file=sys.stderr)
