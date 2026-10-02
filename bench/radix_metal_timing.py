#!/usr/bin/env python3
"""The sip fused GEMV on Metal, timed against the wall: the seven
Qwen3-8B layer-0 projections (+ lm_head's first rows) at
M=1 off the gate's fixture, two arms per tensor —

  radix   metal/gemv_radix's fused decode+GEMV over the fixture's streams
          (bf16 x in, bf16 out)
  raw     mlx's own bf16 matmul, `x[None] @ W.T`, over the bf16 tensor —
          what stock mlx-lm runs

(The original Metal measurements also carried the prior codec's Metal GEMV as a
third arm; that port is not on this tree.)

each as PASSES back-to-back dispatches inside one mx.eval (the GPU-side
time per call, no host round trip between them), REPEATS times, the
minimum and the median kept; bytes each arm streams (the resident bytes
+ x + y), GB/s, and the efficiency against the wall — a plain mx read
reduction over a bf16 buffer, measured first on the empty device. The
dense transient decode is timed the same way, for the record.

The projection:
the GEMV time per token is 36 x the layer-0 seven's sum (every Qwen3-8B
layer has these shapes) + lm_head at its full 151,936 rows (the measured
4096-row time scaled by rows); tok/s = 1000 / that is the GEMV-ONLY
ceiling — the non-GEMV time per token (attention, norms, the sampler, the
host) on this chip is NOT measured here, so the table also prints the
ceiling under stated allowances.

    PYTHONPATH=src python bench/radix_metal_timing.py --fixture path.npz [--json out.json]
                                                     [--passes 20] [--repeats 3] [--sweep]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time

import numpy as np

sys.path.insert(0, "src")
import mlx.core as mx  # noqa: E402

from drinkme.codec.pack import FORMAT_VERSION  # noqa: E402
from drinkme.metal import dense_radix, gemv_radix  # noqa: E402

LAYERS = 36
VOCAB = 151936
LAYER_TENSORS = ("model.layers.0.self_attn.q_proj", "model.layers.0.self_attn.k_proj",
                 "model.layers.0.self_attn.v_proj", "model.layers.0.self_attn.o_proj",
                 "model.layers.0.mlp.gate_proj", "model.layers.0.mlp.up_proj",
                 "model.layers.0.mlp.down_proj")
NON_GEMV_ALLOWANCES_MS = (0.0, 10.0, 20.0)


def _timed(fn, passes: int, repeats: int) -> dict:
    """fn() -> an unevaluated mx array. One warm-up eval, then `repeats`
    timings of `passes` back-to-back dispatches under one mx.eval."""
    ys = [fn() for _ in range(passes)]  # one untimed pass: the kernel compiled, the clocks up
    mx.eval(*ys)
    del ys
    samples = []
    for _ in range(repeats):
        ys = [fn() for _ in range(passes)]
        t0 = time.perf_counter()
        mx.eval(*ys)
        samples.append((time.perf_counter() - t0) / passes)
        del ys
    return {"min_ms": round(1e3 * min(samples), 4), "median_ms": round(1e3 * float(np.median(samples)), 4),
            "samples_ms": [round(1e3 * s, 4) for s in samples]}


def measure_wall(gib: float, repeats: int) -> dict:
    """A plain read reduction over a bf16 buffer on the empty device: the
    bandwidth wall the GEMVs are measured against. mx.sum and mx.max over
    the raw bf16 buffer (no cast — a cast would write a copy); the better
    of the two is the wall."""
    n = int(gib * 1024 ** 3) // 2
    buf = mx.ones((n,), dtype=mx.bfloat16)
    mx.eval(buf)
    out = {"probe_bytes": n * 2}
    best = 0.0
    for name, fn in (("sum", lambda: mx.sum(buf)), ("max", lambda: mx.max(buf))):
        mx.eval(fn())
        samples = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            mx.eval(fn())
            samples.append(time.perf_counter() - t0)
        gbs = n * 2 / min(samples) / 1e9
        out[name] = {"min_s": round(min(samples), 5), "gb_s": round(gbs, 2),
                     "samples_s": [round(s, 5) for s in samples]}
        best = max(best, gbs)
    out["wall_gb_s"] = round(best, 2)
    del buf
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fixture", required=True)
    ap.add_argument("--json", default=None)
    ap.add_argument("--passes", type=int, default=20)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--probe-gib", type=float, default=1.0)
    ap.add_argument("--sweep", action="store_true", help="rows-per-threadgroup sweep on the radix GEMV")
    a = ap.parse_args()
    if not mx.metal.is_available():
        print("RADIX_METAL_TIMING FAIL: no Metal")
        return 2
    dev = mx.device_info()
    print(f"device:  {dev.get('device_name')}  mlx {mx.__version__}  passes {a.passes} x repeats {a.repeats}", flush=True)
    rec = {"device": dev.get("device_name"), "mlx": mx.__version__, "passes": a.passes, "repeats": a.repeats,
           "fixture": a.fixture, "tensors": []}
    rec["wall"] = measure_wall(a.probe_gib, max(5, a.repeats))
    wall = rec["wall"]["wall_gb_s"]
    print(f"WALL  {a.probe_gib:.1f} GiB bf16 read reduction: sum {rec['wall']['sum']['gb_s']} GB/s, "
          f"max {rec['wall']['max']['gb_s']} GB/s -> wall {wall} GB/s", flush=True)

    with np.load(a.fixture) as z:
        meta = json.loads(str(z["meta"]))
        arrays = {k: z[k] for k in z.files if k != "meta"}
    print(f"fixture {a.fixture}: {meta['model']} profile {meta['profile']}, {len(meta['tensors'])} tensors", flush=True)

    hdr = f"{'tensor':40s} {'R':>6s}x{'C':<5s} {'arm':6s} {'min ms':>9s} {'med ms':>9s} {'MB':>8s} {'GB/s':>8s} {'eff':>6s}"
    print(hdr, flush=True)
    for t in meta["tensors"]:
        name, R, C = t["name"], t["R"], t["C"]
        p = {"rx_palette": arrays[f"{name}.rx_palette"], "rx_offsets": arrays[f"{name}.rx_offsets"],
             "rx_data": arrays[f"{name}.rx_data"], "R": R, "C": C, "bpw": t["bpw"], "codec": "radix",
             "profile": meta["profile"], "widths": t["widths"], "block_size": t["block_size"],
             "layout": 0, "format_version": FORMAT_VERSION}
        res = gemv_radix.resident(p)
        x16 = arrays[f"{name}.x"][0]
        xb = mx.array(x16).view(mx.bfloat16)
        mx.eval(xb)
        entry = {"name": name, "R": R, "C": C, "bpw": t["bpw"], "arms": {}}

        def arm(label, fn, nbytes):
            tm = _timed(fn, a.passes, a.repeats)
            gbs = nbytes / (tm["min_ms"] / 1e3) / 1e9
            entry["arms"][label] = {**tm, "bytes": int(nbytes), "gb_s": round(gbs, 2),
                                    "efficiency": round(gbs / wall, 3) if wall else None}
            print(f"{name:40s} {R:6d}x{C:<5d} {label:6s} {tm['min_ms']:9.4f} {tm['median_ms']:9.4f} "
                  f"{nbytes / 1e6:8.1f} {gbs:8.2f} {gbs / wall:6.3f}", flush=True)

        # sip fused (bf16 x in, bf16 out)
        rb = gemv_radix.resident_bytes(res) + 2 * C + 2 * R
        arm("radix", lambda: gemv_radix.gemv_radix_resident(res, xb), rb)
        if a.sweep:
            entry["sweep"] = {}
            for spr, rpt in ((1, 2), (1, 4), (1, 8), (1, 16), (2, 2), (2, 4), (2, 8), (4, 1), (4, 2), (4, 4), (8, 1), (8, 2)):
                tm = _timed(lambda: gemv_radix.gemv_radix_resident(res, xb, rows_per_tg=rpt, simd_per_row=spr),
                            a.passes, a.repeats)
                entry["sweep"][f"spr{spr}/rpt{rpt}"] = tm["min_ms"]
            best = min(entry["sweep"], key=entry["sweep"].get)
            entry["sweep_best"] = best
            print(f"{'':40s} {'':12s} sweep  " + "  ".join(f"{k}={v:.4f}" for k, v in entry["sweep"].items())
                  + f"  best {best}", flush=True)
        # the dense transient (M > 1's decode), for the record
        arm("dense", lambda: dense_radix.decode_dense_resident(res), gemv_radix.resident_bytes(res) + 2 * R * C)

        # the bytes, off the dense kernel, checked against the safetensors digest
        bits_mx = dense_radix.decode_dense_resident(res)
        mx.eval(bits_mx)
        bits = np.array(bits_mx)
        digest = hashlib.sha256(bits.view(np.uint8).tobytes()).hexdigest()
        if digest != t["sha256"]:
            raise SystemExit(f"{name}: dense kernel bytes {digest[:16]} != source {t['sha256'][:16]} — not timing a wrong tensor")
        entry["sha256_checked"] = True

        # raw bf16 matmul (mlx's own M=1 path)
        W = bits_mx.view(mx.bfloat16)
        xrow = xb[None, :]
        arm("raw", lambda: (xrow @ W.T)[0], 2 * R * C + 2 * C + 2 * R)

        rec["tensors"].append(entry)
        del res, bits_mx, bits, W

    # the projection
    by = {e["name"]: e for e in rec["tensors"]}
    proj = {}
    for label in ("radix", "raw"):
        if not all(label in by[n]["arms"] for n in LAYER_TENSORS) or label not in by.get("lm_head", {}).get("arms", {}):
            continue
        S = sum(by[n]["arms"][label]["min_ms"] for n in LAYER_TENSORS)
        h = by["lm_head"]
        H = h["arms"][label]["min_ms"] * (VOCAB / h["R"])
        T = LAYERS * S + H
        proj[label] = {"layer_ms": round(S, 4), "lm_head_ms_full": round(H, 4), "gemv_ms_per_token": round(T, 3),
                       "tok_s_gemv_only": round(1000 / T, 2),
                       "tok_s_with_allowance": {str(ms): round(1000 / (T + ms), 2) for ms in NON_GEMV_ALLOWANCES_MS}}
    rec["projection"] = proj
    print("\nPROJECTION (36 layers x the seven + lm_head at 151,936 rows; GEMV time only, M=1)", flush=True)
    print(f"{'arm':6s} {'layer ms':>9s} {'lm_head ms':>11s} {'GEMV ms/tok':>12s} {'tok/s ceiling':>14s}  "
          + "  ".join(f"+{int(ms)}ms" for ms in NON_GEMV_ALLOWANCES_MS), flush=True)
    for label, pr in proj.items():
        print(f"{label:6s} {pr['layer_ms']:9.4f} {pr['lm_head_ms_full']:11.3f} {pr['gemv_ms_per_token']:12.3f} "
              f"{pr['tok_s_gemv_only']:14.2f}  " + "  ".join(f"{v:5.2f}" for v in pr["tok_s_with_allowance"].values()), flush=True)
    if a.json:
        with open(a.json, "w") as f:
            json.dump(rec, f, indent=1)
        print(f"wrote {a.json}", flush=True)
    print("RADIX_METAL_TIMING DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
