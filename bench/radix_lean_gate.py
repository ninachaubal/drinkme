"""THE LEAN DECODERS' PACK GATE AND TIMING: radix_kernel_gpu's
_decode_sip_lean (DECODER 3, sip's whole-block rows) or _decode_gulp_lean
(DECODER 4, gulp's whole-block rows on ROCm) or _decode_gulp_lean_cuda
(DECODER 5, the same on CUDA) against the scheduled decoder (DECODER 2) on every
radix tensor of a real pack, then both timed per shape class under
sustained load. Each tensor's lean decoder is the one
radix_ops._decoder selects under DRINKME_RADIX_DECODER=lean; a tensor no
lean decoder reads is skipped.

    python bench/radix_lean_gate.py --model Qwen3-8B --pack-dir DIR --json out.json

Claims, each over every radix tensor of the pack (--limit narrows):

  bits    the dense decode (radix_kernel_gpu._dense) under the lean decoder
          equals the scheduled decoder's, bit for bit
  source  on --source-tensors tensors: the lean decode equals the
          checkpoint's own safetensors bf16 bits (the pack's bound snapshot)
  gemv    the fused M=1 GEMV (radix_ops.gemv_fused, the tensor's launch
          schedule, bf16 x) under the lean decoder equals the scheduled
          decoder's output bit for bit: same weights, same loop, same order
  mc      radix_ops.gemv_mc at M = 2 and 8 likewise, on every tensor

Then --passes passes over each shape class's tensors per decoder, the
decoders alternating for --repeats rounds, CUDA events around each
measurement and nvidia-smi sampled (clocks, power; on a ROCm build the
sysfs sensors of parity_common.Sensors): ms per pass, GB/s of the
tensors' read bytes (radix_ops.read_bytes) and the clocks each decoder ran
at. A pass runs the class's tensors back to back, every one of them read
once from memory (the set is larger than the cache): the rotation
protocol, no flush between calls.

--selftest flips one decoded bit of one tensor's lean result and must print
FAIL. Prints RADIX_LEAN_GATE PASS/FAIL; exits through os._exit.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback

import numpy as np
import torch
import triton

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from drinkme.codec import radix_ops as ro  # noqa: E402
from drinkme.codec import radix_pack as rp  # noqa: E402
from drinkme.codec.pack import iter_pack_dir  # noqa: E402
from drinkme.codec.radix_kernel_gpu import _dense  # noqa: E402
from drinkme.codec.swap import to_device_radix  # noqa: E402

from decode_step_profile import GpuSampler, menu_model  # noqa: E402
from parity_common import Sensors  # noqa: E402

SCHED = ro.DECODER_SCHEDULED
LEAN = (ro.DECODER_SIP_LEAN, ro.DECODER_GULP_LEAN, ro.DECODER_GULP_LEAN_CUDA)


def lean_for(p: dict) -> int | None:
    """The lean decoder radix_ops._decoder runs this tensor at under
    DRINKME_RADIX_DECODER=lean, or None when none reads it."""
    set_decoder("lean")
    d = ro._decoder(tuple(int(w) for w in p["widths"]), int(p["C"]) % int(p["block_size"]) == 0)
    set_decoder("")
    return d if d in LEAN else None


def sampler():
    """nvidia-smi on a CUDA build, the sysfs sensors on a ROCm one."""
    return Sensors() if getattr(torch.version, "hip", None) else GpuSampler()


def clock(summary: dict):
    """The median core clock of a sampler's summary, either kind."""
    return (summary.get("clocks.sm") or summary.get("sclk_mhz") or {}).get("median")


def power(summary: dict):
    return (summary.get("power.draw") or summary.get("power_w") or {}).get("median")


def bit_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Bitwise equality (a NaN equals the same NaN)."""
    view = {2: torch.int16, 4: torch.int32}[a.element_size()]
    return a.shape == b.shape and bool(torch.equal(a.contiguous().view(view), b.contiguous().view(view)))


def set_decoder(mode: str) -> None:
    os.environ[ro.DECODER_ENV] = mode
    ro._decoder.cache_clear()


def dense_bits(p: dict, decoder: int) -> torch.Tensor:
    R, C, B = int(p["R"]), int(p["C"]), int(p["block_size"])
    out = torch.empty((R, C), dtype=torch.uint16, device=p["rx_data"].device)
    args = list(ro._args(p))
    args[-1] = decoder
    _dense[(R * triton.cdiv(C, B),)](out, *args, False, num_warps=ro.NUM_WARPS)
    return out


def source_bits(snap: str, name: str) -> torch.Tensor | None:
    """The checkpoint's own tensor `name + '.weight'` as uint16 bits, or None."""
    from safetensors import safe_open

    idx = os.path.join(snap, "model.safetensors.index.json")
    key = name + ".weight"
    if os.path.exists(idx):
        with open(idx) as f:
            shard = json.load(f)["weight_map"].get(key)
        if shard is None:
            return None
        path = os.path.join(snap, shard)
    else:
        path = os.path.join(snap, "model.safetensors")
    with safe_open(path, framework="pt") as f:
        if key not in f.keys():
            return None
        return f.get_tensor(key).view(torch.uint16)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen3-8B")
    ap.add_argument("--pack-dir", required=True)
    ap.add_argument("--json", required=True)
    ap.add_argument("--limit", type=int, default=0, help="first N radix tensors (0 = all)")
    ap.add_argument("--source-tensors", type=int, default=8)
    ap.add_argument("--seconds", type=float, default=2.5, help="per timed measurement: sustained, so the clocks settle")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--no-timing", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    report: dict = {"gate": "lean decoder vs scheduled decoder, per pack tensor", "lean_decoders": list(LEAN),
                    "device": torch.cuda.get_device_name(0), "torch": torch.__version__,
                    "triton": triton.__version__, "pack_dir": a.pack_dir, "selftest": a.selftest,
                    "tensors": [], "failures": []}
    status = "FAIL"
    try:
        from drinkme.serving.checkpoint import resolve_source

        m = menu_model(a.model)
        snap, rev = resolve_source(m.hf_repo, m.revision, a.pack_dir)
        report["revision"] = rev
        gen = torch.Generator(device="cuda").manual_seed(1234)
        resident: dict = {}
        n = 0
        t0 = time.perf_counter()
        for name, pack in iter_pack_dir(a.pack_dir):
            if a.limit and n >= a.limit:
                break
            if not rp.is_radix_pack(pack):
                continue
            p = to_device_radix(pack, "cuda")
            rec = {"tensor": name, "shape": [p["R"], p["C"]], "widths": list(p["widths"]),
                   "class": p["rx_launch"]["shape_class"], "launch": p["rx_launch"]}
            lean = lean_for(p)
            if lean is None:
                rec["skipped"] = "no lean decoder reads these widths and rows"
                report["tensors"].append(rec)
                continue
            rec["lean_decoder"] = lean
            b_s = dense_bits(p, SCHED)
            b_l = dense_bits(p, lean)
            if a.selftest and n == 0:
                b_l.view(torch.int16).view(-1)[12345 % b_l.numel()] ^= 1  # no uint16 xor on CUDA
            rec["bits_equal"] = bool(torch.equal(b_s, b_l))
            if n < a.source_tensors:
                src = source_bits(snap, name)
                if src is not None:
                    rec["source_equal"] = bool(torch.equal(src.to("cuda").reshape(b_l.shape), b_l))
            x = (torch.randn(p["C"], generator=gen, device="cuda") * 0.05).to(torch.bfloat16)
            set_decoder("scheduled")
            g_s = ro.gemv_fused(p, x, None, torch.bfloat16)
            xm = (torch.randn(8, p["C"], generator=gen, device="cuda") * 0.05).to(torch.bfloat16)
            mc_s = {M: ro.gemv_mc(p, xm[:M]) for M in (2, 8)}
            set_decoder("lean")
            g_l = ro.gemv_fused(p, x, None, torch.bfloat16)
            mc_l = {M: ro.gemv_mc(p, xm[:M]) for M in (2, 8)}
            set_decoder("")
            rec["gemv_equal"] = bit_equal(g_s, g_l)
            rec["mc_equal"] = {M: bit_equal(mc_s[M], mc_l[M]) for M in mc_s}
            ok = rec["bits_equal"] and rec["gemv_equal"] and all(rec["mc_equal"].values()) \
                and rec.get("source_equal", True)
            if not ok:
                report["failures"].append(name)
            report["tensors"].append(rec)
            n += 1
            if n <= 3 or n % 25 == 0 or not ok:
                print(f"  {n:4d} {name:42s} {p['R']}x{p['C']} bits {rec['bits_equal']} source "
                      f"{rec.get('source_equal', '-')} gemv {rec['gemv_equal']} mc {rec['mc_equal']} "
                      f"{time.perf_counter() - t0:6.1f}s", flush=True)
            del b_s, b_l
            if not a.no_timing:
                resident.setdefault(rec["class"], []).append(p)
            else:
                del p
        report["checked"] = n
        report["source_checked"] = sum(1 for r in report["tensors"] if "source_equal" in r)
        print(f"checked {n} radix tensors ({report['source_checked']} against the checkpoint's bits): "
              f"{len(report['failures'])} failures", flush=True)

        if not a.no_timing and not report["failures"]:
            timing: dict = {}
            for cls, ps in resident.items():
                nbytes = sum(ro.read_bytes(p) for p in ps)
                xs = [(torch.randn(p["C"], generator=gen, device="cuda") * 0.05).to(torch.bfloat16) for p in ps]
                row = {"tensors": len(ps), "bytes_per_pass": nbytes, "scheduled": [], "lean": []}
                for mode in ("scheduled", "lean"):  # compile + warm
                    set_decoder(mode)
                    for p, x in zip(ps, xs):
                        ro.gemv_fused(p, x, None, torch.bfloat16)
                torch.cuda.synchronize()
                t1 = time.perf_counter()
                for p, x in zip(ps, xs):
                    ro.gemv_fused(p, x, None, torch.bfloat16)
                torch.cuda.synchronize()
                passes = max(2, int(a.seconds / max(time.perf_counter() - t1, 1e-4)))
                row["passes"] = passes
                for _rep in range(a.repeats):
                    for mode in ("scheduled", "lean"):
                        set_decoder(mode)
                        with sampler() as smp:
                            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                            e0.record()
                            for _ in range(passes):
                                for p, x in zip(ps, xs):
                                    ro.gemv_fused(p, x, None, torch.bfloat16)
                            e1.record()
                            torch.cuda.synchronize()
                        ms = e0.elapsed_time(e1) / passes
                        smp_s = smp.summary()
                        row[mode].append({"ms_per_pass": round(ms, 3), "gb_s": round(nbytes / (ms * 1e-3) / 1e9, 1),
                                          "sm_mhz": clock(smp_s), "power_w": power(smp_s)})
                set_decoder("")
                for mode in ("scheduled", "lean"):
                    xs_ = sorted(r["ms_per_pass"] for r in row[mode])
                    row[mode + "_median_ms"] = xs_[len(xs_) // 2]
                row["lean_over_scheduled"] = round(row["lean_median_ms"] / row["scheduled_median_ms"], 4)
                timing[cls] = row
                print(f"TIMING {cls:7s} {len(ps):3d} tensors {nbytes / 1e9:6.2f} GB/pass  scheduled "
                      f"{row['scheduled_median_ms']:8.2f} ms  lean {row['lean_median_ms']:8.2f} ms  "
                      f"({row['lean_over_scheduled']:.3f}x)  clocks s/l "
                      f"{[r['sm_mhz'] for r in row['scheduled']]} / {[r['sm_mhz'] for r in row['lean']]}", flush=True)
            report["timing"] = timing
        status = "FAIL" if report["failures"] or not report["tensors"] else "PASS"
    except Exception as e:  # noqa: BLE001
        report["error"] = f"{type(e).__name__}: {e}"
        report["traceback"] = traceback.format_exc()
        print(report["traceback"], flush=True)
        status = "FAIL"
    report["verdict"] = status
    with open(a.json, "w") as f:
        json.dump(report, f, indent=1)
    if a.selftest:
        print(f"SELFTEST: one lean bit flipped on purpose; this run printed {status} (must be FAIL)", flush=True)
    print(f"RADIX_LEAN_GATE {status}: {report.get('checked', 0)} tensors, {len(report['failures'])} failures",
          flush=True)
    sys.stdout.flush()
    os._exit(0 if status == "PASS" else 1)


if __name__ == "__main__":
    main()
