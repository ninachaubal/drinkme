"""The prefix slot's KV cache: a StaticCache whose full-attention layers hand
attention the LIVE window of their allocation, never the whole of it.

transformers-5's `StaticLayer.update` returns keys/values at their full
`max_cache_len` and `get_mask_sizes` reports that width, so every decode
step attends across the ALLOCATION, not the conversation. With the slot at
`HFEngine.KV_FLOOR` = 4096 and a 204-token request that is, per layer, per
token: a `[1, 1, 1, 4096]` mask (masking_utils never skips the q_len-1 mask
for a compileable cache), `repeat_kv` copying K and V across all 4096
columns for GQA (`use_gqa_in_sdpa` requires no mask), and the efficient
kernel over 4096 columns. Measured on Strix Halo (Qwen3-8B sip,
torch.profiler):
88.6 ms/token at alloc 4096 against 60.2 at 256 — attn_fwd 16.4 ms vs 1.0,
the repeat_kv clone 13.1 vs 0.9 — served 11.29 tok/s against the bench's
16.55 (0.68x); with the window live-sized, 16.6 (1.00x), same transcript.
`--prefix-slots 0` never shows this: its per-request cache is sized
n_prompt + max_new.

`LiveStaticLayer` keeps a python int `live` beside the tensor counter — the
rows written so far, advanced in `update()` by the rows just written (a
shape, never a device read) — returns `keys[:, :, :live]` /
`values[:, :, :live]` and reports `(live, 0)` to the mask. The write itself
is StaticLayer's own (index_copy_ at `cumulative_length`, the 0-dim tensor
mtp.py rewinds in place); the int only says how much of the buffer is
conversation. Two paths move the tensor counter behind the int's back and
call the int back into line: `rewind(drop)` after a rejected verify batch
(mtp.py's full-attention branch) and `resync()` after a slot restore wrote
the counter tensor (slotstore.apply_state). Sliding-window layers (gemma-4)
track their window with `cumulative_length_int` and retain their static mask
policy; linear-attention layers have no width. Both keep their upstream class.

For full-attention/DeltaNet layouts the live layer reports
`is_compileable=False`: its returned shape grows, and this lets SDPA omit
the redundant single-query causal mask and use grouped KV heads directly.
Mixed layouts with untouched StaticLayer subclasses keep StaticLayer's mask
policy, since those layers can still return unwritten buffer capacity.
See docs/serve-kernels.md for the attention-backend implications.

Numerics: the columns the window drops were masked to -inf — exp(-inf) is
exactly 0 and adds nothing — so the same kernel over the live width returns
the same values it did over the allocation (measured: the served transcript
sha 17e9a68af34a54dd at alloc 4096, 256, 204 and with this window).
"""

from __future__ import annotations

from transformers import StaticCache
from transformers.cache_utils import StaticLayer


class LiveStaticLayer(StaticLayer):
    """StaticLayer that returns its live window (module docstring)."""

    # The returned width grows on each forward, so this is not a fixed-shape
    # compileable cache. Leaving StaticLayer's True here also forces SDPA to
    # build a mask for a single query, even though every returned KV row is
    # valid. That mask prevents native grouped-query attention and makes
    # Transformers copy K/V once per query-head group on every decode step.
    # Padding and multi-token suffixes still get their masks from
    # Transformers' normal checks; mixed static layouts opt back in below.
    is_compileable = False

    def __init__(self, max_cache_len: int, **kwargs):
        super().__init__(max_cache_len=max_cache_len, **kwargs)
        self.live = 0
        # True only inside a static step (serving/cudagraph.py): the whole
        # allocation goes to attention, whose marker carries the live
        # length on the device; a slice by the int would be baked into a
        # captured graph
        self.whole = False

    def update(self, key_states, value_states, *args, **kwargs):
        keys, values = super().update(key_states, value_states, *args, **kwargs)
        self.live += key_states.shape[-2]
        if self.whole:
            return keys, values
        return keys[:, :, :self.live], values[:, :, :self.live]

    def get_mask_sizes(self, query_length: int) -> tuple[int, int]:
        # called before this forward's update: the width attention will see
        return self.live + query_length, 0

    def get_seq_length(self) -> int:
        """The int, not the tensor. StaticLayer's own returns the 0-dim
        `cumulative_length`, and masking_utils tests the mask's q_offset
        with `q_offset == 0` on every multi-row forward (`_ignore_causal_
        mask_sdpa`): on a tensor that `if` is a `bool(tensor)` — a
        device->host sync at the top of every MTP verify batch, before one
        of its kernels is enqueued (the M=1 step
        returns early on q_length == 1 and never reaches it). The int is the
        same number (update/rewind/resync/reset keep it so) and is what
        transformers' own StaticSlidingWindowLayer returns here."""
        return self.live

    def rewind(self, drop: int) -> None:
        """Un-write the last `drop` rows: the tensor counter in place (its
        static address survives, as mtp.py's rewind requires) and the int
        with it. The rows stay in the buffer; the window no longer reaches
        them."""
        if drop < 0 or drop > self.live:
            raise ValueError(f"rewind {drop} rows of a {self.live}-row window")
        if drop:
            self.cumulative_length.sub_(drop)
            self.live -= drop

    def resync(self) -> None:
        """The int from the tensor — after a restore overwrote the counter
        (one device read, at restore time only)."""
        self.live = int(self.cumulative_length) if self.is_initialized else 0

    def reset(self) -> None:
        super().reset()
        self.live = 0


class LiveStaticCache(StaticCache):
    """StaticCache whose plain full-attention layers are LiveStaticLayer.
    Same constructor; every other layer kind is transformers' own."""

    def __init__(self, config, max_cache_len: int, **kwargs):
        super().__init__(config=config, max_cache_len=max_cache_len, **kwargs)
        # The flag is consumed cache-wide. Untouched StaticLayer subclasses
        # (sliding/indexed caches) can still return unwritten rows. In
        # particular, a sliding allocation smaller than the model's window
        # passes SDPA's local-window skip check before the buffer is full.
        # Preserve the original mask policy for every such mixed layout.
        needs_static_mask = any(isinstance(layer, StaticLayer) and type(layer) is not StaticLayer
                                for layer in self.layers)
        self.layers = [LiveStaticLayer(max_cache_len=layer.max_cache_len)
                       if type(layer) is StaticLayer else layer
                       for layer in self.layers]
        if needs_static_mask:
            for layer in self.layers:
                if isinstance(layer, LiveStaticLayer):
                    layer.is_compileable = True
