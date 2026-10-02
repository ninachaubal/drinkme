"""Native grouped-query SDPA for short, unpadded Qwen MTP suffixes.

Transformers' SDPA adapter repeats KV heads whenever it receives a mask.
MTP verifies several queries against an existing cache, so it needs a
lower-right causal mask even though serial decode does not need a mask.
PyTorch's CausalBias expresses that alignment without materializing a mask
or repeating the KV heads. See docs/serve-kernels.md.

Only our explicit CausalBias takes this route. Ordinary masks, including
padding and sliding windows, still go through the upstream adapter. The
trunk constructs the bias only for Qwen3_5TextModel with live full-attention
cache layers; the head owns its unpadded dynamic cache and its length.
"""

from __future__ import annotations

import copy
import os

import torch
from torch.nn.attention.bias import CausalBias, causal_lower_right
from transformers.integrations.sdpa_attention import sdpa_attention_forward
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from .kvcache import LiveStaticLayer

NAME = "drinkme_suffix_sdpa"
ENV = "DRINKME_MTP_SUFFIX_SDPA"
MAX_ROWS = 8


def attention_forward(module, query, key, value, attention_mask,
                      dropout=0.0, scaling=None, **kwargs):
    from . import cudagraph, segmented_attention

    if isinstance(attention_mask, cudagraph.LiveLength):
        # a static step's whole-allocation attention (serving/cudagraph.py)
        return cudagraph.attention_forward(module, query, key, value, attention_mask,
                                           dropout=dropout, scaling=scaling, **kwargs)
    if isinstance(attention_mask, segmented_attention.SegmentedCausal):
        # a long prefill extend (serving/segmented_attention.py)
        return segmented_attention.attention_forward(
            module, query, key, value, attention_mask, dropout=dropout,
            scaling=scaling, **kwargs)
    if not isinstance(attention_mask, CausalBias):
        return sdpa_attention_forward(module, query, key, value, attention_mask,
                                      dropout=dropout, scaling=scaling, **kwargs)
    # A mismatch is a bookkeeping error, not permission to drop a mask.
    if (query.shape[-2] != attention_mask.seq_len_q
            or key.shape[-2] != attention_mask.seq_len_kv):
        raise ValueError("MTP suffix mask does not match the live KV window")
    out = torch.nn.functional.scaled_dot_product_attention(
        query, key, value, attn_mask=attention_mask, dropout_p=dropout,
        scale=scaling, enable_gqa=query.shape[1] != key.shape[1])
    return out.transpose(1, 2).contiguous(), None


def install(model) -> None:
    """Give Qwen attention instances an adapter without changing model config.

    The model still builds ordinary SDPA masks. Copying just the attention
    config also leaves other models sharing that config untouched.
    """
    ALL_ATTENTION_FUNCTIONS.register(NAME, attention_forward)
    for mod in model.modules():
        if (type(mod).__name__ == "Qwen3_5Attention"
                and mod.config._attn_implementation == "sdpa"):
            mod.config = copy.copy(mod.config)
            mod.config._attn_implementation = NAME


def enabled(rows: int, device) -> bool:
    return (1 < rows <= MAX_ROWS and torch.device(device).type == "cuda"
            and os.environ.get(ENV, "1") != "0")


def trunk_mask(base, input_ids, cache):
    """Return a mask mapping only for the unpadded MTP trunk call we own.

    No external masks or position IDs enter forward_with_hidden. A live
    full cache returns exactly past+rows entries, unlike a plain static or
    sliding cache. Never infer this property from the model name alone.
    """
    rows = input_ids.shape[1]
    if (not enabled(rows, input_ids.device) or input_ids.shape[0] != 1
            or type(base).__name__ != "Qwen3_5TextModel" or cache is None):
        return None
    kinds = base.config.layer_types
    if any(kind not in ("full_attention", "linear_attention") for kind in kinds):
        return None
    full = [i for i, kind in enumerate(kinds) if kind == "full_attention"]
    if not full or any(
        type(cache.layers[i]) is not LiveStaticLayer
        or base.layers[i].self_attn.config._attn_implementation != NAME
        for i in full
    ):
        return None
    past = cache.layers[full[0]].live
    if any(cache.layers[i].live != past for i in full):
        return None
    return {"full_attention": causal_lower_right(rows, past + rows),
            "linear_attention": None}
