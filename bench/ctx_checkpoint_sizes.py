"""What one context checkpoint (serving/ctx_checkpoints.py) costs for each menu
model, measured on the CPU from the checkpoint's real shapes, and what a slot's
checkpoints can hold at most.

For each menu row whose config.json is in the HF cache, the text model is
built with the real attention and linear-attention dimensions, in bfloat16,
with one layer of each kind that holds state a rewind cannot reach and one
full-attention layer (a small
MLP and vocabulary: neither is in a checkpoint). Enough tokens are driven
through it to fill every sliding ring, a real snapshot is taken, and its bytes
per layer kind, times the config's layer counts, must equal
ctx_checkpoints.config_bytes, the formula the fit charge uses. A model of
full attention only takes no checkpoints and is listed at 0.

    HF_HUB_OFFLINE=1 DRINKME_NO_AUTO_DEPS=1 CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= \\
      PYTHONPATH=src .venv/bin/python bench/ctx_checkpoint_sizes.py --out DIR

Writes DIR/sizes.json and prints the table. Needs no weights.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

import torch

CTXS = (8192, 32768, 131072, 262144)


def _config(repo: str, revision: str | None):
    from transformers import AutoConfig

    return AutoConfig.from_pretrained(repo, revision=revision, local_files_only=True)


def _measure(tc) -> dict[str, int]:
    """Bytes of a real snapshot per layer kind, from a model with one layer
    of each non-rewindable kind at the real dimensions."""
    from transformers import AutoModel

    from drinkme.serving import ctx_checkpoints as cc
    from drinkme.serving.kvcache import LiveStaticCache

    kinds = [t for t in dict.fromkeys(tc.layer_types)
             if t in ("sliding_attention", "linear_attention")]
    if not kinds:
        return {}
    # and one full-attention layer, which every real pattern has and which
    # the cache needs to know its length; its snapshot holds nothing
    # (gemma-4's per_layer_config is keyed by its real 60 layers' indices;
    # it sets the full layers' own head shapes, which hold nothing here)
    raw = {k: v for k, v in tc.to_dict().items() if k != "per_layer_config"}
    cfg = tc.__class__.from_dict({**raw, "num_hidden_layers": len(kinds) + 1,
                                  "layer_types": kinds + ["full_attention"],
                                  "intermediate_size": 64, "vocab_size": 512})
    if hasattr(cfg, "moe_intermediate_size"):
        cfg.moe_intermediate_size = 64
    if hasattr(cfg, "vocab_size_per_layer_input"):
        cfg.vocab_size_per_layer_input = 512
    torch.manual_seed(0)
    model = AutoModel.from_config(cfg).to(torch.bfloat16).eval()
    if model.__class__.__name__.startswith("Qwen3_5"):
        from transformers.models.qwen3_5 import modeling_qwen3_5 as mod

        fn = mod.torch_chunk_gated_delta_rule
        mod.torch_chunk_gated_delta_rule = getattr(fn, "__wrapped__", fn)
    n = (cfg.sliding_window or 0) + 8 if "sliding_attention" in kinds else 16
    cache = LiveStaticCache(config=cfg, max_cache_len=n + 8)
    ids = torch.arange(n).remainder(500)[None] + 3
    with torch.inference_mode():
        model(ids, past_key_values=cache, use_cache=True, cache_position=torch.arange(n))
        snap = cc.snapshot(cache, task=0)
    out: dict[str, int] = {}
    for i, t in enumerate(kinds):
        entry = snap.layers[i]
        tensors = entry[:3] if isinstance(entry, tuple) else [
            x for s in entry.values() for x in s[:2] if x is not None]
        out[t] = sum(x.numel() * x.element_size() for x in tensors)
    return out


def main(argv=None) -> int:
    from drinkme.serving import ctx_checkpoints as cc
    from drinkme.suggest import MODELS

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    rows, ok = [], True
    for m in MODELS:
        try:
            real = _config(m.hf_repo, m.revision)
        except (OSError, ValueError) as e:
            print(f"{m.name}: no cached config.json ({type(e).__name__}) — skipped", flush=True)
            continue
        tc = real.get_text_config()
        types = list(getattr(tc, "layer_types", None) or [])
        counts = Counter(types)
        formula = cc.config_bytes(tc.to_dict())
        per_kind = _measure(tc) if types else {}
        measured = sum(per_kind.get(t, 0) * c for t, c in counts.items())
        same = measured == formula
        ok &= same
        row = {"model": m.name, "repo": m.hf_repo, "layers": dict(counts),
               "window": getattr(tc, "sliding_window", None), "per_kind_bytes": per_kind,
               "per_checkpoint_bytes": measured, "formula_bytes": formula,
               "agrees": same, "native_ctx": getattr(tc, "max_position_embeddings", None)}
        for ctx in CTXS:
            held = cc.max_held(cc.DEFAULT_MAX, ctx)
            one = cc.config_bytes(tc.to_dict(), 2, ctx)
            row[f"held@{ctx}"] = held if formula else 0
            row[f"slot_bytes@{ctx}"] = held * one if formula else 0
        rows.append(row)
        gib = 1024 ** 3
        print(f"{m.name}: {dict(counts) or 'no layer_types'}; one checkpoint "
              f"{measured:,} B ({measured / 2**20:.1f} MiB) measured, formula "
              f"{'agrees' if same else f'SAYS {formula:,}'}; per slot at most "
              + ", ".join(f"{row[f'held@{c}']} x = {row[f'slot_bytes@{c}'] / gib:.2f} GiB @ {c}"
                          for c in CTXS), flush=True)
    with open(os.path.join(a.out, "sizes.json"), "w") as f:
        json.dump(rows, f, indent=1)
    print("VERDICT", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
