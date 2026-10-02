"""THE RADIX MULTI-COLUMN GATE: for M in 2..8, every profile, random +
adversarial + production-shaped tensors, the mc kernel (codec/radix_ops._gemv_mc)
against the M=1 GEMV it claims to amortize.

    PYTHONPATH=src .venv/bin/python bench/radix_mc_bitpin.py [--json out.json]
                                    [--pack-dir DIR --limit N] [--selftest]

Earns (radix, 0, MC, TRITON). Claims, per (tensor, M):

  oracle    every mc column m and the M=1 GEMV of x[m] both lie within the
            float64 dot's cancellation-aware bound (2e-6 * |x|.|W| + 1e-7,
            the research gate's) — THE GATE. Whether mc column m is also
            BITWISE the M=1 result is RECORDED per case (`bitwise`), never
            gated: Triton may lay the two kernels' reductions out
            differently, and on this box it does on partial
            (ragged) blocks while every production shape (C a multiple of
            1024) comes out bitwise.
  bits      the block decode inside the mc program is the M=1 program's
            (same _decode, same schedule): pinned by the dense decode of
            the same runtime dict being the source bits exactly, and by
            the oracle bound above holding for every column.
  forward   the module: one M-row bf16 call (route "mc") and M one-row
            calls (route "gemv", the fused kernel) must each be EXACTLY
            their own arm's fp32 result + bias rounded once to bf16 (the
            epilogue contract, CompressedLinear._epilogue / _finish) — THE
            GATE; whether the two arms' bf16 rows are also bitwise equal is
            recorded (`forward_bitwise`), a partial-block reduction-order
            flip being the one way they can differ.

--pack-dir runs the same claims on the first --limit tensors of a REAL
radix pack (a Qwen3-8B pack) with random activations.

--selftest perturbs one mc column by 1e-3 and must print FAIL and exit 1.
ROCm torch cannot exit nonzero once HIP is up (AGENTS.md): read the verdict
line; the exit goes through os._exit.
"""

import argparse
import json
import os
import sys
import time
import traceback
import zlib

# every M in 2..8 must route to mc regardless of GEMM_MIN_ROWS — the gate is
# the kernel, not the threshold (mc_bitpin.py's own rule); read once by swap
os.environ["DRINKME_PREFILL_DENSE_MIN"] = "0"

import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, "src")
from drinkme.codec import radix_ops, radix_pack as rp  # noqa: E402
from drinkme.codec.ops import MC_MAX  # noqa: E402
from drinkme.codec.pack import iter_pack_dir  # noqa: E402
from drinkme.codec.swap import RadixCompressedLinear, to_device_radix  # noqa: E402

PROFILES = ("sip", "balanced", "gulp")
COLS = tuple(range(2, MC_MAX + 1))  # 2..8, every M the mc arm serves
SYNTHETIC = (  # (R, C, kind)
    (1024, 4096, "real"), (4096, 4096, "real"), (256, 12288, "real"), (4096, 1024, "real"),
    (1024, 4096, "spread"), (256, 12288, "spread"),
    (129, 260, "real"), (17, 3079, "real"), (3, 1025, "spread"), (7, 33, "real"),
)


def _bits(rng, R, C, kind):
    if kind == "spread":
        k = np.arange(40)
        pk = 0.75 ** k
        exps = rng.choice(100 + k, size=(R, C), p=pk / pk.sum())
    else:
        exps = rng.choice(np.arange(113, 122), size=(R, C),
                          p=np.array([.002, .003, .005, .01, .025, .055, .1, .3, .5]))
    bits = (exps.astype(np.uint16) << 7) | rng.integers(0, 128, (R, C), dtype=np.uint16)
    bits |= rng.integers(0, 2, (R, C), dtype=np.uint16) << 15
    return bits


def check_tensor(name, p, W, receipt, corrupt=None, corrupt_forward=False):
    """Every M x every column, three claims. Returns a list of failures."""
    bad, records = [], []
    rt = to_device_radix(p, "cuda")
    R, C = int(p["R"]), int(p["C"])
    dense = radix_ops.decode_bits(rt).view(torch.int16)
    if not torch.equal(dense, W.view(torch.int16)):
        bad.append({"tensor": name, "claim": "bits", "M": 0, "note": "dense decode != source bits"})
        return bad, records
    g = torch.Generator(device="cuda").manual_seed(zlib.crc32(name.encode()))
    bias = torch.linspace(-.1, .1, R * 2, device="cuda", dtype=torch.bfloat16)[::2].contiguous()
    mod = RadixCompressedLinear(rt, bias)
    Wd = W.double()
    for M in COLS:
        # bf16 activations, AS SERVED, into every kernel call: Triton
        # specializes on the activation pointer's dtype, and with fp fusion
        # on the fp32-x compilation contracts multiply-adds differently
        # from the bf16-x one (measured here: a few elements per tensor
        # differ between the two builds), so a gate that fed one arm fp32
        # and the other bf16 would be comparing compilations, not arms
        xb = torch.randn(M, C, device="cuda", dtype=torch.bfloat16, generator=g)
        x = xb.float()  # the oracle's view of the same values
        oracle = x.double() @ Wd.T
        bound = 2e-6 * (x.double().abs() @ Wd.abs().T) + 1e-7
        mc = radix_ops.gemv_mc(rt, xb)
        if corrupt is not None and M == corrupt[0]:
            mc = mc.clone()
            mc[corrupt[1], corrupt[2]] += corrupt[3]
        g1 = torch.stack([radix_ops.gemv(rt, xb[m]) for m in range(M)])
        e_mc = ((mc.double() - oracle).abs() / bound).max().item()
        e_g1 = ((g1.double() - oracle).abs() / bound).max().item()
        bitwise = torch.equal(mc, g1)
        rows_diff = int((mc != g1).any(dim=1).sum()) if not bitwise else 0
        if e_mc > 1 or e_g1 > 1:
            bad.append({"tensor": name, "claim": "oracle", "M": M, "mc_err_over_bound": e_mc,
                        "gemv_err_over_bound": e_g1})
        # the module: bf16 rows in, one mc call vs M gemv calls
        if mod._route(xb) != "mc":
            bad.append({"tensor": name, "claim": "routing", "M": M, "note": f"routed {mod._route(xb)}"})
        y_mc = mod(xb)
        if corrupt_forward and M == COLS[0]:
            y_mc = y_mc.clone()
            y_mc[0, 0] = y_mc[0, 0] + 1
        y_g1 = torch.cat([mod(xb[m:m + 1]) for m in range(M)], dim=0)
        fwd_bitwise = torch.equal(y_mc.view(torch.int16), y_g1.view(torch.int16))
        # the epilogue contract, exactly: each arm's own fp32 rows + bias, one rounding
        want_mc = (radix_ops.gemv_mc(rt, xb) + bias.float()).to(torch.bfloat16)
        want_g1 = (g1 + bias.float()).to(torch.bfloat16)
        mc_ok = torch.equal(y_mc.view(torch.int16), want_mc.view(torch.int16))
        g1_ok = torch.equal(y_g1.view(torch.int16), want_g1.view(torch.int16))
        if not (mc_ok and g1_ok):
            bad.append({"tensor": name, "claim": "forward", "M": M, "mc_epilogue_exact": mc_ok,
                        "gemv_epilogue_exact": g1_ok,
                        "elements_differing": int((y_mc != want_mc).sum() + (y_g1 != want_g1).sum())})
        records.append({"tensor": name, "M": M, "bitwise": bitwise, "rows_differing": rows_diff,
                        "mc_err_over_bound": round(e_mc, 4), "gemv_err_over_bound": round(e_g1, 4),
                        "forward_bitwise": fwd_bitwise, "forward_epilogue_exact": mc_ok and g1_ok})
    receipt["cases"].extend(records)
    return bad, records


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--json", default=None)
    ap.add_argument("--pack-dir", default=None, help="run on the first --limit tensors of a real radix pack")
    ap.add_argument("--limit", type=int, default=8)
    a = ap.parse_args()
    if not torch.cuda.is_available():
        print("RADIX_MC_BITPIN FAIL: no GPU — this gate cannot be satisfied on CPU")
        return 2
    print(f"device:   {torch.cuda.get_device_name(0)}  torch {torch.__version__}", flush=True)
    print(f"claims:   oracle (gate) + bits + forward   M in {COLS}   dense arm off (all 2..8 is mc)", flush=True)
    if a.selftest:
        print("SELFTEST: one mc column is perturbed on purpose; this run MUST print FAIL", flush=True)
    receipt = {"gate": "radix multi-column bit-pin", "device": torch.cuda.get_device_name(0),
               "torch": torch.__version__, "selftest": a.selftest, "pack_dir": a.pack_dir,
               "columns": list(COLS), "cases": [], "tensors": []}
    failures, n, t0, status = [], 0, time.perf_counter(), 1
    try:
        rng = np.random.default_rng(193)
        if a.pack_dir:
            for name, p in iter_pack_dir(a.pack_dir):
                if n >= a.limit:
                    break
                if not rp.is_radix_pack(p):
                    print(f"  skip {name}: not a radix tensor (the raw fallback)", flush=True)
                    continue
                U = rp.decode_back_radix(p)
                W = torch.from_numpy(U.view(np.int16).copy()).view(torch.bfloat16).cuda()
                bad, recs = check_tensor(name, p, W, receipt,
                                         corrupt=(5, 2, 17, 1e-3) if (a.selftest and n == 0) else None,
                                         corrupt_forward=(a.selftest and n == 0))
                failures += bad
                n += 1
                bw = sum(r["bitwise"] for r in recs)
                receipt["tensors"].append({"tensor": name, "profile": p["profile"], "shape": [p["R"], p["C"]],
                                           "bitwise_Ms": bw, "of_Ms": len(recs)})
                print(f"  {n:3d} {name:42s} {p['profile']:8s} {p['R']}x{p['C']}  bitwise {bw}/{len(recs)} Ms  "
                      f"{len(failures)} failures  {time.perf_counter() - t0:6.1f}s", flush=True)
                if failures:
                    break
        else:
            for (R, C, kind) in SYNTHETIC:
                bits = _bits(rng, R, C, kind)
                W = torch.from_numpy(bits.view(np.int16).copy()).view(torch.bfloat16).cuda()
                for profile in PROFILES:
                    p = rp.pack_array_radix(bits, profile)
                    name = f"{kind}-{R}x{C}-{profile}"
                    if p is None:
                        print(f"  skip {name}: radix would expand (the raw fallback)", flush=True)
                        continue
                    bad, recs = check_tensor(name, p, W, receipt,
                                             corrupt=(5, 2, min(17, R - 1), 1e-3) if (a.selftest and n == 0) else None,
                                             corrupt_forward=(a.selftest and n == 0))
                    failures += bad
                    n += 1
                    bw = sum(r["bitwise"] for r in recs)
                    receipt["tensors"].append({"tensor": name, "profile": profile, "shape": [R, C],
                                               "bpw": round(p["bpw"], 4), "bitwise_Ms": bw, "of_Ms": len(recs)})
                    print(f"  {n:3d} {name:28s} bpw {p['bpw']:6.3f}  mc==gemv bitwise {bw}/{len(recs)} Ms  "
                          f"{len(failures)} failures  {time.perf_counter() - t0:6.1f}s", flush=True)
                    if failures:
                        break
                if failures:
                    break
        dt = time.perf_counter() - t0
        for f in failures[:10]:
            print(f"  MISMATCH {f}", flush=True)
        n_cases = len(receipt["cases"])
        n_bit = sum(c["bitwise"] for c in receipt["cases"])
        n_fwd = sum(c["forward_bitwise"] for c in receipt["cases"])
        worst = max((c["mc_err_over_bound"] for c in receipt["cases"]), default=0.0)
        receipt.update({"tensors_checked": n, "cases_checked": n_cases, "bitwise_cases": n_bit,
                        "forward_bitwise_cases": n_fwd, "worst_mc_err_over_bound": worst,
                        "failures": failures, "seconds": round(dt, 1)})
        verdict = "FAIL" if failures or n == 0 else "PASS"
        receipt["verdict"] = verdict
        print(f"RADIX_MC_BITPIN {verdict}: {n} tensors, {n_cases} (tensor, M) cases, {len(failures)} failures; "
              f"mc column bitwise the M=1 GEMV in {n_bit}/{n_cases} cases (recorded, not gated), "
              f"module forward bitwise in {n_fwd}/{n_cases}; worst mc err/bound {worst:.3f}; {dt:.1f}s",
              flush=True)
        status = 0 if verdict == "PASS" else 1
    except BaseException as e:  # noqa: BLE001 — the verdict line is the contract
        receipt["verdict"] = "FAIL"
        receipt["error"] = f"{type(e).__name__}: {e}"
        print(f"RADIX_MC_BITPIN FAIL: {type(e).__name__}: {e}", flush=True)
        traceback.print_exc()
        status = 1
    if a.json:
        with open(a.json, "w") as f:
            json.dump(receipt, f, indent=1)
        print(f"wrote {a.json}", flush=True)
    if a.selftest:
        if status == 1:
            print("SELFTEST PASSED: the corruption was caught (this run's FAIL is correct)", flush=True)
        else:
            print("SELFTEST FAILED: the corruption was NOT caught — this gate is broken", flush=True)
            status = 1
    return status


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(rc)
