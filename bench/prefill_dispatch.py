"""Longest single GPU dispatch of one prefill (or decode) forward, per kernel,
at Qwen3.8-27B shapes.

The desktop's amdgpu ring resets a graphics job that waits longer than
`lockup_timeout` (2000 ms on this box); a compute dispatch that holds the CUs
that long takes the compositor with it. This instrument measures how long the
longest dispatch of one forward runs as a function of the new-token count T
and the cached context L, so the large-L numbers can be EXTRAPOLATED rather
than run.

WORKER (`--T --L`): one synthetic two-layer model — one linear_attention
(DeltaNet) layer and one full_attention layer at the 27B's real dimensions
(read from its config.json), vocab and lm_head at the real width — routed
through serving/kernel_route.route_kernels exactly as the loaders route it,
with the SDPA backend the served units enable
(TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1). `--compressed` swaps every Linear
the pack holds for the pack's own module (layer 0 <- pack layer 0, layer 1 <-
pack layer 3, lm_head <- pack lm_head), so the dense arm's decode-once + GEMM
at M = T is what runs. The cache is a LiveStaticCache: one primed token, then
L-1 random K/V rows written straight into the full-attention buffers (the
DeltaNet state has no sequence dimension — its cost cannot depend on L).
Then the forward: `model(ids[T], cache_position=arange(L, L+T),
logits_to_keep=1)` — engines.py's prefill call — twice (the first compiles and
autotunes; both are traced). `--chunk C` runs the T tokens as ceil(T/C)
extends through serving/prefill.run — the engine's chunked prefill, with
serving/segmented_attention.py's split above its threshold.

DRIVER (`--sweep`): runs each worker under `rocprofv3 --kernel-trace`, reads
the per-dispatch CSV, keeps the longest dispatch per kernel family, and walks
a doubling ladder. Before each step it PREDICTS the next step's longest
dispatch from the last two measured points (the observed growth ratio, at
least the step's own size ratio) and refuses to launch a step predicted
over --launch-ms (500), counting a first-dispatch factor of 2; it stops a ladder the moment any measured dispatch
passes --stop-ms (250). Output: one JSON per ladder under --out.

Hold `flock -n /tmp/drinkme-gpu.lock` around the driver. No model is loaded
unless MemAvailable >= 25 GB.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

PY = os.environ.get("PY", ".venv/bin/python")
ROCPROF = os.path.join(os.path.dirname(PY), "rocprofv3")
SNAP = os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen3.8-27B/snapshots")
PACK = os.path.expanduser("~/.cache/drinkme/packs/Qwen--Qwen3.8-27B@1d4bf0f2ff60")
LAYER_SRC = {0: 0, 1: 3}  # synthetic layer -> pack layer of the same kind
# rep 0 compiles and autotunes; reps 1.. are identical dispatch sequences, and
# a dispatch's INTRINSIC duration is its minimum over them. The GPU may be
# shared (a desktop session, another server): the same GEMM measured 2.7 ms and 19.6 ms
# in consecutive reps, so a single rep's max is interference, not the
# kernel. Both are recorded.
REPS = 4
VERIFY = False  # the driver's --verify, passed to every worker


def mem_available_gb() -> float:
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1e6
    return 0.0


# --------------------------------------------------------------- worker --

def _subset_pack(tmp: str) -> str:
    """A pack dir holding only the tensors the synthetic model uses, renamed
    to its layer numbering (hard links — the pack reader refuses a symlink
    out of its directory; meta.json rewritten). Read-only: nothing is written
    into the pack itself."""
    meta = json.load(open(os.path.join(PACK, "meta.json")))
    keep = {}
    for dst, src in LAYER_SRC.items():
        for name, fn in meta["tensors"].items():
            pre = f"model.layers.{src}."
            if name.startswith(pre):
                keep[f"model.layers.{dst}." + name[len(pre):]] = fn
    keep["lm_head"] = meta["tensors"]["lm_head"]
    for fn in set(keep.values()):
        os.link(os.path.join(PACK, fn), os.path.join(tmp, fn))
    meta = dict(meta, tensors=dict(sorted(keep.items())))
    json.dump(meta, open(os.path.join(tmp, "meta.json"), "w"))
    return tmp


def build(compressed: bool):
    import torch
    from transformers import AutoConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

    snap = sorted(glob.glob(os.path.join(SNAP, "*")))[-1]
    cfg = AutoConfig.from_pretrained(snap).get_text_config()
    cfg.num_hidden_layers = 2
    cfg.layer_types = ["linear_attention", "full_attention"]
    cfg._attn_implementation = "sdpa"
    torch.manual_seed(0)
    with torch.device("cuda"):
        model = Qwen3_5ForCausalLM._from_config(cfg, dtype=torch.bfloat16).eval()
    if compressed:
        from drinkme.codec.pack import iter_pack_dir
        from drinkme.codec.swap import make_module

        scratch = os.path.expanduser("~/.cache/drinkme")  # same filesystem as the pack
        with tempfile.TemporaryDirectory(dir=scratch, prefix="prefill-dispatch-") as tmp:
            for name, pack in iter_pack_dir(_subset_pack(tmp)):
                parent, _, child = name.rpartition(".")
                old = model.get_submodule(name)
                assert isinstance(old, torch.nn.Linear), name
                setattr(model.get_submodule(parent) if parent else model, child,
                        make_module(pack, None, "cuda"))
        torch.cuda.empty_cache()
    from drinkme.serving.kernel_route import route_kernels

    route = route_kernels(model, "cuda")
    return model, cfg, route


def fill_cache(model, cfg, L: int, room: int):
    """A LiveStaticCache holding L positions: one real token, then L-1 random
    K/V rows written into the full-attention buffers."""
    import torch
    from drinkme.serving.kvcache import LiveStaticCache, LiveStaticLayer

    cache = LiveStaticCache(config=model.config, max_cache_len=L + room)
    if L == 0:
        return cache
    with torch.inference_mode():
        model(torch.zeros(1, 1, dtype=torch.long, device="cuda"), past_key_values=cache,
              use_cache=True, cache_position=torch.arange(1, device="cuda"), logits_to_keep=1)
        for layer in cache.layers:
            if isinstance(layer, LiveStaticLayer) and L > 1:
                layer.keys[:, :, 1:L].normal_()
                layer.values[:, :, 1:L].normal_()
                layer.cumulative_length.fill_(L)
                layer.live = L
    return cache


def worker(args) -> None:
    import torch

    from drinkme.serving import mtp, prefill, suffix_attention

    assert torch.cuda.is_available(), "REFUSING: no accelerator"
    model, cfg, route = build(args.compressed)
    if args.verify:
        suffix_attention.install(model)  # what mtp.py does for a served head
    T, L = args.T, args.L
    g = torch.Generator(device="cuda").manual_seed(1)
    ids = torch.randint(0, cfg.vocab_size, (1, T), device="cuda", generator=g)
    walls, windows = [], []
    for rep in range(args.reps):
        cache = fill_cache(model, cfg, L, T + 1)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        m0 = time.monotonic_ns()  # rocprofv3's timestamps are CLOCK_MONOTONIC
        with torch.inference_mode():
            if args.verify:
                # the MTP verify batch as served: the trunk through
                # mtp.forward_with_hidden with the suffix CausalBias mask
                mtp.forward_with_hidden(model, ids, cache,
                                        torch.arange(L, L + T, device="cuda"))
            elif args.chunk:
                # the engine's own prefill: spans of C, segmented attention
                # above segmented_attention.WORK (serving/prefill.py)
                prefill.run(model, [0] * L + ids[0].tolist(), L, cache, "cuda", args.chunk)
            else:  # the unchunked call: one forward over all T tokens
                model(ids, past_key_values=cache, use_cache=True,
                      cache_position=torch.arange(L, L + T, device="cuda"),
                      logits_to_keep=1)
        torch.cuda.synchronize()
        walls.append(time.perf_counter() - t0)
        windows.append((m0, time.monotonic_ns()))
        del cache
    print("VERDICT " + json.dumps({"T": T, "L": L, "chunk": args.chunk,
                                   "compressed": args.compressed, "route": route,
                                   "wall_s": walls, "windows": windows,
                                   "peak_gib": torch.cuda.max_memory_allocated() / 2**30}),
          flush=True)


# --------------------------------------------------------------- driver --

FAMILIES = [
    ("attn_fwd", r"attn_fwd|fmha|flash|aotriton|attention"),
    ("gemm", r"Cijk_|gemm|Gemm|GEMM|wvSplitK|matmul"),
    ("radix_decode", r"radix|k_decode|decode_weight|k_rx"),
    ("gdn_chunk", r"chunk|fwd_h|fwd_o|wy_|solve_tril|kkt|cumsum"),
    ("conv", r"causal_conv|conv1d|conv"),
    ("softmax", r"softmax"),
    ("elementwise", r"elementwise|vectorized|unrolled|reduce|index|copy|fill|cat|where"),
]


def family(name: str) -> str:
    for fam, pat in FAMILIES:
        if re.search(pat, name):
            return fam
    return "other"


def read_trace(d: str, windows=None) -> list[dict]:
    """Every dispatch, or only those that START inside one of `windows`
    (the forwards; model build and cache fill excluded)."""
    rows = []
    for fn in glob.glob(os.path.join(d, "**", "*kernel_trace.csv"), recursive=True):
        with open(fn) as f:
            for r in csv.DictReader(f):
                t0 = int(r["Start_Timestamp"])
                rep = next((i for i, (a, b) in enumerate(windows or []) if a <= t0 <= b), None)
                if windows and rep is None:
                    continue
                ms = (int(r["End_Timestamp"]) - t0) / 1e6
                rows.append({"name": r["Kernel_Name"], "ms": ms, "rep": rep, "t0": t0})
    return rows


def summarize(rows: list[dict]) -> dict:
    fam: dict = {}
    for r in rows:
        k = family(r["name"])
        cur = fam.get(k)
        if cur is None or r["ms"] > cur["max_ms"]:
            fam[k] = {"max_ms": r["ms"], "kernel": r["name"][:160],
                      "n": (cur or {}).get("n", 0)}
        fam[k]["n"] = fam[k].get("n", 0) + 1
    return fam


def run_one(T: int, L: int, compressed: bool, chunk: int, outdir: Path,
            reps: int = REPS) -> dict:
    if mem_available_gb() < 25:
        raise SystemExit(f"REFUSING: MemAvailable {mem_available_gb():.1f} GB < 25")
    tag = f"T{T}_L{L}_C{chunk}_{'comp' if compressed else 'stock'}{'_verify' if VERIFY else ''}"
    d = outdir / tag
    d.mkdir(parents=True, exist_ok=True)
    cmd = [ROCPROF, "--kernel-trace", "-f", "csv", "-d", str(d), "-o", "trace", "--",
           PY, __file__, "--T", str(T), "--L", str(L), "--chunk", str(chunk),
           "--reps", str(reps)]
    if compressed:
        cmd.append("--compressed")
    if VERIFY:
        cmd.append("--verify")
    env = dict(os.environ, TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL="1",
               OMP_NUM_THREADS="16", MKL_NUM_THREADS="16")
    p = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=900)
    (d / "run.log").write_text(p.stdout + "\n--- stderr ---\n" + p.stderr)
    verdict = [ln for ln in p.stdout.splitlines() if ln.startswith("VERDICT ")]
    if not verdict:
        raise SystemExit(f"worker {tag} gave no verdict; see {d/'run.log'}")
    res = json.loads(verdict[0][8:])
    rows = read_trace(str(d), res["windows"])
    if not rows:
        raise SystemExit(f"worker {tag}: no dispatch inside the forward windows")
    by_rep = [sorted((r for r in rows if r["rep"] == i), key=lambda r: r["t0"])
              for i in range(len(res["windows"]))]
    steady = by_rep[1:]
    if len({len(x) for x in steady}) != 1:
        raise SystemExit(f"worker {tag}: reps 1.. dispatched {[len(x) for x in steady]} "
                         "kernels — not the same sequence")
    intrinsic = [{"name": col[0]["name"], "ms": min(r["ms"] for r in col)}
                 for col in zip(*steady)]
    res["n_dispatch"] = len(intrinsic)
    res["rep_max_ms"] = [max((r["ms"] for r in x), default=0.0) for x in by_rep]
    res["families"] = summarize(intrinsic)
    res["families_observed"] = summarize(rows)
    res["max_ms"] = max(r["ms"] for r in intrinsic)
    res["max_kernel"] = max(intrinsic, key=lambda r: r["ms"])["name"][:160]
    res["observed_max_ms"] = max(r["ms"] for r in rows)
    for fn in glob.glob(str(d / "**" / "*kernel_trace.csv"), recursive=True):
        subprocess.run(["gzip", "-f", fn])  # raw per-dispatch receipt, compressed
    (d / "summary.json").write_text(json.dumps(res, indent=1))
    return res


def ladder(points: list[tuple[int, int]], compressed: bool, chunk: int, outdir: Path,
           stop_ms: float, launch_ms: float, label: str, reps: int = REPS) -> list[dict]:
    """points: (T, L) in increasing size. Predict, launch, measure, stop."""
    done: list[dict] = []
    for T, L in points:
        if done:
            last = done[-1]
            size_ratio = ((T * (L + T)) / max(1, last["T"] * (last["L"] + last["T"])))
            if len(done) >= 2:
                prev = done[-2]
                growth = last["max_ms"] / max(1e-6, prev["max_ms"])
            else:
                growth = size_ratio
            # x2: a shape's FIRST dispatch ran up to 1.9x its intrinsic time
            # (T=2048 L=32768 attn_fwd: 552 ms then 295, 295, 295)
            pred = 2 * last["max_ms"] * max(growth, min(size_ratio, 4.0), 1.0)
            if last["max_ms"] > stop_ms:
                print(f"[{label}] STOP: measured {last['max_ms']:.1f} ms > {stop_ms}", flush=True)
                break
            if pred > launch_ms:
                print(f"[{label}] NOT LAUNCHING T={T} L={L}: predicted {pred:.0f} ms "
                      f"> {launch_ms}", flush=True)
                break
        r = run_one(T, L, compressed, chunk, outdir, reps)
        if r["observed_max_ms"] > launch_ms:
            print(f"[{label}] ABORT: an observed dispatch ran {r['observed_max_ms']:.0f} ms "
                  f"({r['max_kernel'][:60]}) — over the launch bound", flush=True)
            done.append(r)
            break
        top = sorted(r["families"].items(), key=lambda kv: -kv[1]["max_ms"])[:4]
        print(f"[{label}] T={T:>6} L={L:>7} C={chunk}: max {r['max_ms']:7.2f} ms  "
              + "  ".join(f"{k} {v['max_ms']:.2f}" for k, v in top)
              + f"  wall {r['wall_s'][-1]*1e3:.0f} ms  observed rep max "
              + f"{[round(x, 1) for x in r['rep_max_ms']]}",
              flush=True)
        done.append(r)
    (outdir / f"{label}.json").write_text(json.dumps(done, indent=1))
    return done


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--T", type=int)
    ap.add_argument("--L", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=0)
    ap.add_argument("--compressed", action="store_true")
    ap.add_argument("--verify", action="store_true",
                    help="worker: T rows through mtp.forward_with_hidden (the MTP verify)")
    ap.add_argument("--reps", type=int, default=REPS,
                    help="forwards per worker (rep 0 compiles; >= 2)")
    ap.add_argument("--sweep", choices=["T", "L", "decode", "cold"],
                    help="cold: T doubling from --start at L=0, chunked by --chunk "
                         "(a whole long prompt, the engine's way)")
    ap.add_argument("--fixed", type=int, default=1024,
                    help="the other axis: L for --sweep T, T for --sweep L")
    ap.add_argument("--start", type=int, default=256)
    ap.add_argument("--max", type=int, default=262144)
    ap.add_argument("--out", type=Path, default=Path("/tmp/prefill_dispatch"))
    ap.add_argument("--stop-ms", type=float, default=250.0)
    ap.add_argument("--launch-ms", type=float, default=500.0)
    args = ap.parse_args()
    if args.sweep is None:
        return worker(args)
    global VERIFY
    VERIFY = args.verify
    args.out.mkdir(parents=True, exist_ok=True)
    arm = "comp" if args.compressed else "stock"
    xs = []
    x = args.start
    while x <= args.max:
        xs.append(x)
        x *= 2
    if args.sweep == "T":
        pts = [(t, args.fixed) for t in xs]
        label = f"sweepT_L{args.fixed}_{arm}_C{args.chunk}"
    elif args.sweep == "L":
        pts = [(args.fixed, l) for l in xs]
        label = f"sweepL_T{args.fixed}_{arm}_C{args.chunk}"
    elif args.sweep == "cold":
        pts = [(t, 0) for t in xs]
        label = f"cold_{arm}_C{args.chunk}"
    else:
        pts = [(args.fixed, l) for l in xs]
        label = f"decodeM{args.fixed}_{arm}{'_verify' if args.verify else ''}"
    ladder(pts, args.compressed, args.chunk, args.out, args.stop_ms, args.launch_ms, label,
           args.reps)


if __name__ == "__main__":
    main()
