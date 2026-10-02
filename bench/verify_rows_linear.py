#!/usr/bin/env python3
"""What an M-row verify costs a model's Linears, per shape, against M=1.

An MTP verify forward runs every trunk Linear at M = k+1 rows (k drafts plus
the last token); the serial decode step runs them at M=1. On the pack, M=1
is radix_ops.gemv_fused and 2 <= M <= MC_MAX is radix_ops.gemv_mc, which
decodes each block once for all M rows (codec/swap.py's header); on the BF16
checkpoint both are F.linear. This times each route on the model's REAL
resident layers, through the module's own forward (what the verify pays,
the epilogue included), and for the pack also the bare kernels.

ROTATION, not do_bench. Every layer of a shape is called in turn, one event
pair per call, back to back, so each call reads weights the previous call
did not (the 27B's 64 layers of one MLP shape are far past any cache on the
box) — the served access pattern. do_bench's cache eviction distorts
bandwidth-bound kernels on this box. The first round of each (shape, M) is
a warm-up and is dropped; the median of the remaining calls is reported,
and the per-token sum over every layer of every shape.

    # the 27B pack (GPU, ~1 min after the load)
    PYTHONPATH=src python bench/verify_rows_linear.py --model Qwen/Qwen3.8-27B \\
        --revision 1d4bf0f2ff60 --pack ~/.cache/drinkme/packs/Qwen--Qwen3.8-27B@1d4bf0f2ff60 \\
        --rows 1,2,3,4,5,6,8 --json /tmp/vrl_pack.json
    # the BF16 checkpoint
    PYTHONPATH=src python bench/verify_rows_linear.py --model Qwen/Qwen3.8-27B \\
        --revision 1d4bf0f2ff60 --stock --json /tmp/vrl_bf16.json

Verdict is in the output, never in $?: TheRock torch _exit(0)s on atexit.
"""

import argparse
import json
import os
import statistics as st
import sys
import time


def parse():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", default="Qwen/Qwen3.8-27B")
    ap.add_argument("--revision", default=None)
    ap.add_argument("--pack", default=os.path.expanduser(
        "~/.cache/drinkme/packs/Qwen--Qwen3.8-27B@1d4bf0f2ff60"))
    ap.add_argument("--stock", action="store_true", help="the BF16 checkpoint, not the pack")
    ap.add_argument("--rows", default="1,5", help="the M values to time")
    ap.add_argument("--rounds", type=int, default=6,
                    help="rotations over each shape's layers (the first is dropped)")
    ap.add_argument("--ctx", type=int, default=4096)
    ap.add_argument("--json", default=None)
    return ap.parse_args()


def groups(model):
    """{(class, R, C): [modules]} over the model's Linears, compressed or not."""
    import torch

    from drinkme.codec.swap import CompressedLinear

    out = {}
    for name, m in model.named_modules():
        if isinstance(m, CompressedLinear):
            R, C = m.R, m.C
        elif isinstance(m, torch.nn.Linear):
            R, C = m.out_features, m.in_features
        else:
            continue
        out.setdefault((type(m).__name__, R, C), []).append((name, m))
    return out


def time_rotation(mods, call, rounds):
    """Event pair per call, back to back over `mods`, `rounds` times; the
    first round dropped. -> per-call ms list."""
    import torch

    evs = []
    for r in range(rounds):
        for m in mods:
            e0 = torch.cuda.Event(enable_timing=True)
            e1 = torch.cuda.Event(enable_timing=True)
            e0.record()
            call(m)
            e1.record()
            evs.append((r, e0, e1))
    torch.cuda.synchronize()
    return [e0.elapsed_time(e1) for r, e0, e1 in evs if r > 0]


def resident_bytes(m) -> int | None:
    """The bytes one call reads from its weights: the runtime dict's tensors
    for a compressed module, the weight for a plain Linear."""
    import torch

    p = getattr(m, "p", None)
    if isinstance(p, dict):
        return sum(v.numel() * v.element_size() for v in p.values() if torch.is_tensor(v))
    w = getattr(m, "weight", None)
    return w.numel() * w.element_size() if w is not None else None


def main() -> int:
    args = parse()
    os.environ.setdefault("DRINKME_PREFIX_SLOTS", "0")
    os.environ["DRINKME_SPEC"] = "off"  # no head: this times the trunk's Linears only
    import torch

    from drinkme.serve import build_engine

    t0 = time.time()
    engine = build_engine(args.model, args.revision, None if args.stock else args.pack,
                          stock=args.stock, ctx=args.ctx)
    model = engine.model
    dev = next(model.parameters()).device if any(True for _ in model.parameters()) else "cuda"
    rows = [int(x) for x in args.rows.split(",")]
    radix_ops = None
    if not args.stock:
        from drinkme.codec import radix_ops
    torch.manual_seed(0)
    receipt = {"model": args.model, "revision": args.revision,
               "pack": None if args.stock else args.pack, "stock": args.stock,
               "gpu": torch.cuda.get_device_name(0), "rows": rows, "rounds": args.rounds,
               "date": time.strftime("%Y-%m-%d %H:%M %Z"), "shapes": []}
    totals = {M: 0.0 for M in rows}
    kernel_totals = {M: 0.0 for M in rows}
    with torch.inference_mode():
        for (cls, R, C), named in sorted(groups(model).items(), key=lambda kv: -kv[0][1] * kv[0][2]):
            mods = [m for _, m in named]
            nbytes = resident_bytes(mods[0])
            rec = {"class": cls, "R": R, "C": C, "count": len(mods),
                   "example": named[0][0], "bytes_per_call": nbytes, "forward": {}, "kernel": {}}
            for M in rows:
                x = torch.randn(1, M, C, device=dev, dtype=torch.bfloat16)
                ms = time_rotation(mods, lambda m: m(x), args.rounds)
                med = st.median(ms)
                rec["forward"][M] = {"median_ms": round(med, 4), "min_ms": round(min(ms), 4),
                                     "gb_s": round(nbytes / med / 1e6, 1) if nbytes else None}
                totals[M] += med * len(mods)
                if radix_ops is not None and cls == "RadixCompressedLinear":
                    xf = x.reshape(M, C)
                    if M == 1:
                        call = lambda m: radix_ops.gemv_fused(m.p, xf[0], m.bias, torch.bfloat16)  # noqa: E731
                    else:
                        call = lambda m: radix_ops.gemv_mc(m.p, xf)  # noqa: E731
                    km = st.median(time_rotation(mods, call, args.rounds))
                    rec["kernel"][M] = {"median_ms": round(km, 4),
                                        "gb_s": round(nbytes / km / 1e6, 1) if nbytes else None}
                    kernel_totals[M] += km * len(mods)
            base = rec["forward"].get(1, {}).get("median_ms")
            rec["forward_over_m1"] = {M: round(v["median_ms"] / base, 3) for M, v in rec["forward"].items()} if base else None
            receipt["shapes"].append(rec)
            print(f"[vrl] {cls:24s} {R:6d}x{C:<6d} x{len(mods):3d}  " + "  ".join(
                f"M={M}: {v['median_ms']:.3f} ms ({v['gb_s']} GB/s)" for M, v in rec["forward"].items())
                + (f"  | kernel " + "  ".join(f"M={M}: {v['median_ms']:.3f}" for M, v in rec["kernel"].items())
                   if rec["kernel"] else ""), flush=True)
    receipt["per_token_linear_ms"] = {M: round(v, 2) for M, v in totals.items()}
    receipt["per_token_kernel_ms"] = {M: round(v, 2) for M, v in kernel_totals.items()} if radix_ops else None
    receipt["elapsed_s"] = round(time.time() - t0, 1)
    print("=== PER TOKEN (sum over every Linear of its median call, ms) ===")
    print("  forward: " + "  ".join(f"M={M}: {v:.2f}" for M, v in receipt["per_token_linear_ms"].items()))
    if radix_ops:
        print("  radix kernels only: " + "  ".join(f"M={M}: {v:.2f}" for M, v in receipt["per_token_kernel_ms"].items()))
    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "w") as f:
            json.dump(receipt, f, indent=1)
        print(f"wrote {args.json}")
    print("VERDICT: measured" if receipt["shapes"] else "VERDICT: nothing measured")
    return 0


if __name__ == "__main__":
    code = 1
    try:
        code = main()
    except Exception:
        import traceback

        traceback.print_exc()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)
