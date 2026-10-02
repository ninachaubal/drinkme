#!/usr/bin/env python3
"""Build the Qwen3-8B radix fixture for the Metal gate: the seven layer-0
projections + lm_head's first HEAD_ROWS rows,
read straight off the safetensors snapshot, packed at one profile
(codec/radix_pack.pack_array_radix — the pack writer's own encoder), into
ONE npz with, per tensor:

  <name>.rx_palette / .rx_offsets / .rx_data   the streams, as the pack carries them
  <name>.x      uint16 [3, C]   three bf16 activation vectors (seeded N(0, 0.5))
  <name>.y64    float64 [3, R]  x @ W.T in float64 — the GEMV oracle
  <name>.bound  float64 [3, R]  2e-6 * |x| @ |W|.T + 1e-7 — the research gate's bound
  meta          json: model, snapshot, profile, per tensor R/C/bpw/widths/block_size
                and the sha256 of the SOURCE bits (uint16 [R, C] little-endian) —
                the Mac compares the dense kernel's bytes against it, so the source
                tensor itself never travels (the fixture stays under the disk budget)

    PYTHONPATH=src .venv/bin/python bench/radix_fixture_build.py --snapshot DIR --out fixture.npz [--profile sip]

torch is used for the bf16 read only (safetensors has no numpy bf16).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, "src")

from drinkme.codec import radix_pack as rp  # noqa: E402

TENSORS = ("model.layers.0.self_attn.q_proj", "model.layers.0.self_attn.k_proj",
           "model.layers.0.self_attn.v_proj", "model.layers.0.self_attn.o_proj",
           "model.layers.0.mlp.gate_proj", "model.layers.0.mlp.up_proj",
           "model.layers.0.mlp.down_proj", "lm_head")
HEAD_ROWS = 4096
SEED = 20260917


def _read_bits(snapshot: str, name: str, rows: int | None) -> np.ndarray:
    import torch
    from safetensors import safe_open

    with open(os.path.join(snapshot, "model.safetensors.index.json")) as f:
        shard = json.load(f)["weight_map"][name + ".weight"]
    with safe_open(os.path.join(snapshot, shard), framework="pt") as f:
        t = f.get_tensor(name + ".weight")
    if t.dtype != torch.bfloat16:
        raise ValueError(f"{name}: {t.dtype}, expected bf16")
    if rows is not None:
        t = t[:rows]
    return t.contiguous().view(torch.int16).numpy().view(np.uint16).copy()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--profile", default="sip", choices=list(rp.PROFILES))
    ap.add_argument("--head-rows", type=int, default=HEAD_ROWS)
    a = ap.parse_args()
    rng = np.random.default_rng(SEED)
    arrays: dict[str, np.ndarray] = {}
    meta = {"model": "Qwen/Qwen3-8B", "snapshot": os.path.basename(a.snapshot.rstrip("/")),
            "profile": a.profile, "seed": SEED, "head_rows": a.head_rows,
            "built": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "tensors": []}
    total_stored = 0
    for name in TENSORS:
        t0 = time.perf_counter()
        bits = _read_bits(a.snapshot, name, a.head_rows if name == "lm_head" else None)
        R, C = bits.shape
        p = rp.pack_array_radix(bits, a.profile)
        if p is None:
            raise SystemExit(f"{name}: radix would expand this tensor — no fixture for it")
        digest = hashlib.sha256(np.ascontiguousarray(bits).view(np.uint8).tobytes()).hexdigest()
        W64 = (bits.astype(np.uint32) << 16).view(np.float32).astype(np.float64)
        x32 = (rng.standard_normal((3, C)) * 0.5).astype(np.float32)
        u = x32.view(np.uint32)
        x16 = ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)
        x64 = (x16.astype(np.uint32) << 16).view(np.float32).astype(np.float64)
        y64 = x64 @ W64.T
        bound = 2e-6 * (np.abs(x64) @ np.abs(W64).T) + 1e-7
        del W64
        for k in ("rx_palette", "rx_offsets", "rx_data"):
            arrays[f"{name}.{k}"] = p[k]
        arrays[f"{name}.x"] = x16
        arrays[f"{name}.y64"] = y64
        arrays[f"{name}.bound"] = bound
        stored = sum(p[k].nbytes for k in ("rx_palette", "rx_offsets", "rx_data"))
        total_stored += stored
        meta["tensors"].append({"name": name, "R": int(R), "C": int(C), "bpw": float(p["bpw"]),
                                "widths": [int(w) for w in p["widths"]], "block_size": int(p["block_size"]),
                                "stored_bytes": int(stored), "sha256": digest})
        print(f"{name:40s} {R:6d}x{C:<5d} bpw {p['bpw']:.3f} stored {stored / 1e6:8.1f} MB "
              f"sha256 {digest[:16]}  {time.perf_counter() - t0:.1f}s", flush=True)
    meta["stored_bytes"] = int(total_stored)
    np.savez(a.out, meta=np.array(json.dumps(meta)), **arrays)
    size = os.path.getsize(a.out)
    print(f"wrote {a.out}: {size / 1e6:.1f} MB ({len(TENSORS)} tensors, {total_stored / 1e6:.1f} MB of streams)",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
