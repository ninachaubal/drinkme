"""What each SDPA backend costs and whether it agrees — swept, on this box.

A 25,057-token prompt to Qwen3.8-27B on gfx1151 fails with `Tried to allocate
56.13 GiB` (measured), and `[1, 24, 25057, 25057]` fp32 IS 56.13 GiB. On this TheRock ROCm
gfx1151 build no efficient SDPA backend is available, so attention resolves to
MATH, which materializes that whole score matrix in fp32 — quadratic in MEMORY.
TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 is the documented opt-in for an arch
AOTriton classifies experimental (src/drinkme/sdpa.py has the mechanism).

`drinkme serve` runs the CHEAP version of this at every load — availability at
T=128, plus one self-test at T<=1024 when the flag is set. This file is the
expensive version, and it exists because the load-time announce quotes numbers
that have to come from somewhere:

  - peak device memory per backend across a T ladder, which is where "MATH is
    O(T^2), flash is O(T)" stops being a claim;
  - the LARGEST SINGLE ALLOCATION under MATH, which is the number an OOM
    actually prints and the reason sdpa.py quotes the fp32 score matrix rather
    than peak;
  - max|diff| and mean|diff| against MATH, which is what the self-test's
    tolerances were sized against;
  - the same sweep with enable_gqa on and off. NOT cosmetic: with the flag on,
    grouped-kv (what transformers passes on the prefill path) leaves
    mem_efficient unavailable here while flash still runs.

Run it BOTH WAYS — the flag is read by torch, so it has to be in the
environment of the process, and a receipt of one setting says nothing about
the other:

    uv run --no-sync python bench/sdpa_backend_sweep.py --json off.json
    TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 \\
        uv run --no-sync python bench/sdpa_backend_sweep.py --json on.json

Defaults are Qwen3.8-27B's full-attention shape (head_dim 256, 24q/4kv) — the
showcase model, and the one whose long prompt OOM'd on MATH (sdpa.py).
`--head-dim/--heads/--kv-heads`
for anything else; `--max-t` to cap the ladder on a smaller card.
"""

import argparse
import datetime
import json
import os
import socket
import sys
import warnings

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

sys.path.insert(0, "src")
from drinkme import sdpa as dsdpa  # noqa: E402

BACKENDS = {"math": SDPBackend.MATH, "flash": SDPBackend.FLASH_ATTENTION,
            "mem_efficient": SDPBackend.EFFICIENT_ATTENTION,
            "cudnn": SDPBackend.CUDNN_ATTENTION}


def inputs(t, head_dim, heads, kv, device, dtype):
    # One seed for every arm: this compares KERNELS, so the inputs must be
    # bit-identical and not merely identically distributed.
    gen = torch.Generator(device=device).manual_seed(20260824)
    mk = lambda h: torch.randn(1, h, t, head_dim, generator=gen,  # noqa: E731
                               device=device, dtype=dtype)
    return mk(heads), mk(kv), mk(kv)


def run(name, t, head_dim, heads, kv, gqa, device, dtype):
    """One backend at one T. Returns (output_or_None, stats)."""
    kvh = kv if gqa else heads
    kw = {"enable_gqa": True} if gqa and kvh != heads else {}
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    q, k, v = inputs(t, head_dim, heads, kvh, device, dtype)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # torch warns the very advice we print
            with sdpa_kernel([BACKENDS[name]]):
                out = F.scaled_dot_product_attention(q, k, v, is_causal=True, **kw)
        if device == "cuda":
            torch.cuda.synchronize()
    except Exception as e:  # noqa: BLE001 — "unavailable" is a result
        return None, {"available": False,
                      "error": f"{type(e).__name__}: {str(e).splitlines()[0][:160]}"}
    stats = {"available": True}
    if device == "cuda":
        stats["peak_gib"] = round(torch.cuda.max_memory_allocated() / 1024**3, 4)
        # The tensor an OOM names is the biggest single block, not the peak.
        blocks = [s["total_size"] for s in torch.cuda.memory_snapshot()]
        stats["largest_block_mib"] = round(max(blocks) / 1024**2, 2) if blocks else None
    del q, k, v
    return out.float(), stats


def sweep(args, device, dtype):
    rows = []
    for gqa in (True, False):
        if gqa and args.kv_heads == args.heads:
            continue  # nothing to group
        for t in args.ladder:
            ref, ref_stats = run("math", t, args.head_dim, args.heads,
                                 args.kv_heads, gqa, device, dtype)
            row = {"seq_len": t, "enable_gqa": gqa,
                   "score_matrix_gib": round(
                       dsdpa.score_matrix_bytes(args.heads, t) / 1024**3, 4),
                   "backends": {"math": ref_stats}}
            for name in ("flash", "mem_efficient", "cudnn"):
                out, stats = run(name, t, args.head_dim, args.heads,
                                 args.kv_heads, gqa, device, dtype)
                if out is not None and ref is not None:
                    d = (out - ref).abs()
                    stats["max_abs_diff"] = float(d.max())
                    stats["mean_abs_diff"] = float(d.mean())
                    if ref_stats.get("peak_gib"):
                        stats["peak_ratio_vs_math"] = round(
                            stats["peak_gib"] / ref_stats["peak_gib"], 3)
                    del d
                del out
                row["backends"][name] = stats
            del ref
            if device == "cuda":
                torch.cuda.empty_cache()
            rows.append(row)
            print(json.dumps(row), flush=True)
    return rows


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--head-dim", type=int, default=256)
    p.add_argument("--heads", type=int, default=24)
    p.add_argument("--kv-heads", type=int, default=4)
    p.add_argument("--max-t", type=int, default=4096)
    p.add_argument("--json", default=None)
    args = p.parse_args()
    args.ladder = [t for t in (128, 512, 1024, 2048, 4096, 8192, 16384)
                   if t <= args.max_t]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    shape = dsdpa.AttnShape(args.head_dim, args.heads, args.kv_heads)
    report = dsdpa.probe_backends(shape, device=device)
    box = {}
    if device == "cuda":
        props = torch.cuda.get_device_properties(0)
        box = {"gpu": props.name, "arch": getattr(props, "gcnArchName", None),
               "total_memory_gib": round(props.total_memory / 1024**3, 2)}
    print(json.dumps({"device": device, "dtype": str(dtype), **box,
                      dsdpa.EXPERIMENTAL_ENV: os.environ.get(
                          dsdpa.EXPERIMENTAL_ENV)}), flush=True)

    rows = sweep(args, device, dtype)
    st = None
    if report.has_efficient:
        st = dsdpa.self_test(shape, report.efficient[0], device=device)
    out = {
        "what": "bench/sdpa_backend_sweep.py — SDPA backend availability, peak "
                "memory and agreement vs the math kernel",
        "why": "long-prompt OOMs are the MATH fallback "
               "materializing [1, heads, T, T] in fp32",
        "box": socket.gethostname().upper(),
        "date": datetime.date.today().isoformat(),
        "device": device,
        "dtype": str(dtype),
        "torch": torch.__version__,
        **box,
        dsdpa.EXPERIMENTAL_ENV: os.environ.get(dsdpa.EXPERIMENTAL_ENV),
        "shape": {"head_dim": args.head_dim, "n_heads": args.heads,
                  "n_kv_heads": args.kv_heads},
        "load_time_probe": report.to_dict(),
        "load_time_self_test": st.to_dict() if st else None,
        "rows": rows,
    }
    if args.json:
        with open(args.json, "w") as f:
            json.dump(out, f, indent=2)
        print(f"wrote {args.json}", flush=True)
    return 0


if __name__ == "__main__":
    # Never trust $? from a GPU run on TheRock ROCm torch (AGENTS.md): an atexit
    # handler _exit(0)s over the real status once HIP initializes. The verdict
    # rides in the output.
    rc = main()
    print(f"[sdpa_backend_sweep] rc={rc}", flush=True)
    sys.exit(rc)
