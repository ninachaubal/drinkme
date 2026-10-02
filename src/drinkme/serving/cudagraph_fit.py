"""What CUDA graphs cost in device memory, and whether a model will use them:
the torch-free half of serving/cudagraph.py, so the fit checks that run
before torch loads (packs.rank_fits, suggest) charge the same bytes the
engine charges at load.

WHAT A CACHE'S GRAPHS HOLD (serving/cudagraph.py, "MEMORY"): every graph
captured against one cache (the decode step, the MTP re-arm step and each
speculative verify width) allocates from ONE private pool, and on a
DeltaNet hybrid the verify graphs write each row's recurrent state into one
buffer per cache that every width shares. So a cache's graphs cost

    POOL_BYTES + (rows - 1) x one row's DeltaNet recurrent states

where `rows` is the widest verify step (MTP depth + 1, or the n-gram
proposer's draft count + 1) and the last row is never stored (a verify
that keeps every row needs no rewind; mtp._restore_rows). Measured with
bench/cuda_graph_gate.py `--memory` after capturing every step serve
replays into one cache (one private pool
per graph, the old layout, in brackets):

    Qwen3-8B, L4, decode + n-gram verify 1..6 (7 graphs)
        pool: stock 24 MiB (94), compressed 38 MiB (114); no row buffer
    Qwen3.8-27B, H100, decode + re-arm + MTP verify 1..5 (7 graphs)
        pool: stock 54 MiB, compressed 98 MiB; row buffer 4 x 144 MiB
        = 576 MiB; together 630 / 674 MiB (2,454 / 2,470 MiB)

POOL_BYTES is the largest pool measured; the row buffer is exact from
the config. Capturing the same widths on a second cache reserved the same
bytes.

The charge is per cache: a prefix slot holds one (and `--prefix-slots 0`
one reusable scratch cache). Nothing is charged where graphs do not run:
ROCm, Metal, the CPU, `DRINKME_CUDA_GRAPHS=0`, a family not on VERIFIED.
"""

from __future__ import annotations

import os

ENV = "DRINKME_CUDA_GRAPHS"

# (model_type, mode) pairs that passed bench/cuda_graph_gate.py's bar on
# CUDA. mode "decode" is the serial M = 1 step; "verify" the speculative
# verify widths (MTP and n-gram), which are graphed only together with
# "decode" (engines.py never mixes a graphed step with an eager verify).
# Keyed by family, not by compression profile: each entry passed the bar
# on a sip pack and on a gulp pack (gulp's scheduled decoder, one more scan
# level per tier), which are GATED_PROFILES.
VERIFIED = frozenset({
    ("qwen3", "decode"),         # Qwen3-8B sip and gulp, L4; sip, H100
    ("qwen3", "verify"),         # n-gram, M = 2..6, 8: sip and gulp, L4; sip M = 2..5, 8, H100
    ("qwen3_5_text", "decode"),  # Qwen3.8-27B sip and gulp (DeltaNet hybrid), H100
    ("qwen3_5_text", "verify"),  # MTP at depth 4, M = 2..5; sip and gulp, H100
})

# The compression profiles every VERIFIED entry was gated on: every profile
# `drinkme pack` writes (codec/radix_pack.PUBLIC_PROFILES). A new one keys
# VERIFIED by profile, or passes the gate first. "balanced", which a pack
# may name but `drinkme pack` does not write, was not gated.
GATED_PROFILES = ("sip", "gulp")

# The largest graph pool one cache's graphs reserved, over the models and
# cards measured (module docstring), rounded up to the allocator's 2 MiB
# segment: temporaries of an M <= 8 row forward, the graphs' static outputs
# and the conv windows of the verify rows.
POOL_BYTES = 98 * 1024**2

# The verify widths a speculating request replays, as the default plans
# draw them: MTP at mtp.DEFAULT_DEPTH (4) drafts, n-gram at
# ngram.DEFAULT_TOKENS (5). Copied, not imported: both modules import torch.
MTP_ROWS = 4 + 1
NGRAM_ROWS = 5 + 1


def env_off() -> bool:
    return os.environ.get(ENV, "").strip() == "0"


def family_of(config: dict | None) -> str | None:
    """The VERIFIED key a config.json loads as: a vision checkpoint whose
    text tree is what the engine serves (Qwen3.8-27B: `qwen3_5` with a
    `qwen3_5_text` text_config) keys on the text config's model_type, as
    the loaded model's config does."""
    if not config:
        return None
    text = config.get("text_config")
    if isinstance(text, dict) and text.get("model_type"):
        return text["model_type"]
    return config.get("model_type")


def _text(config: dict) -> dict:
    text = config.get("text_config")
    return text if isinstance(text, dict) else config


def row_state_bytes(config: dict) -> int:
    """One verify row's DeltaNet recurrent states across every linear
    layer, float32 (the recurrence's own output dtype): heads x key dim x
    value dim x 4 bytes per layer. 0 for a model without linear attention."""
    t = _text(config)
    kinds = t.get("layer_types") or []
    n = sum(1 for k in kinds if k == "linear_attention")
    if not n:
        return 0
    heads = int(t.get("linear_num_value_heads") or 0)
    dk = int(t.get("linear_key_head_dim") or 0)
    dv = int(t.get("linear_value_head_dim") or 0)
    return n * heads * dk * dv * 4


def cache_bytes(config: dict, rows: int) -> int:
    """What one cache's graphs hold when the widest verify step is `rows`
    rows (1 = no speculation): the module docstring's formula."""
    return POOL_BYTES + max(rows - 1, 0) * row_state_bytes(config)


def spec_rows(config: dict) -> int:
    """The widest verify step the default plan replays: 1 under
    DRINKME_SPEC=off, MTP's (DRINKME_MTP_DEPTH, default 4, plus one) when
    the checkpoint carries a head, else n-gram's (serving/ngram.py's AUTO)."""
    if os.environ.get("DRINKME_SPEC", "").strip() == "off":
        return 1
    if int(_text(config).get("mtp_num_hidden_layers") or 0) > 0:
        raw = os.environ.get("DRINKME_MTP_DEPTH", "").strip()
        return int(raw) + 1 if raw.isdigit() and int(raw) > 0 else MTP_ROWS
    return NGRAM_ROWS


def will_graph(config: dict | None, platform: str | None) -> bool:
    """serving/cudagraph.decide()'s answer ahead of loading: `platform`
    is "cuda" | "rocm" | "metal" | "cpu" (the device the model will run on)
    or None when unknown (nothing charged)."""
    return (platform == "cuda" and not env_off()
            and (family_of(config), "decode") in VERIFIED)


def charge_bytes(config: dict | None, platform: str | None, caches: int) -> int:
    """The graph memory a fit check charges: `caches` x cache_bytes at the
    default plan's widest verify step, or 0 where graphs will not run (and
    when there is no config to read). A family whose verify widths are not
    verified replays the decode step only (engines.py decodes a speculating
    request eager then), so it is charged the pool alone."""
    if config is None or not will_graph(config, platform):
        return 0
    rows = spec_rows(config) if (family_of(config), "verify") in VERIFIED else 1
    return max(caches, 1) * cache_bytes(config, rows)
