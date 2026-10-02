"""The narrow raw Linears (codec/swap.NarrowLinear: bf16, under the codec's
row threshold, R <= 1024, C >= 1024) under every GEMV route there is for
them, per M = 1..8: which one NarrowLinear should launch at each M. The
instrument behind NarrowLinear's choice of radix_ops.gemv_narrow for M <= 8.

    PYTHONPATH=src flock /tmp/drinkme-gpu.lock .venv/bin/python bench/narrow_linear_knobs.py \
        --json out.json [--shapes 48x5120:96 32x4096:48 256x6656:104] [--ms 1 2 3 4 5 6 7 8]

The shapes are every narrow Linear on the menu (suggest.MODELS, read off
the meta skeletons): Qwen3.8-27B's in_proj_a / in_proj_b (48 x 5120, 96
per token), MiMo-V2.6-Distill-Qwen-9B's (32 x 4096, 48 per token) and
Muse-Glimmer-30B's k_proj / v_proj (256 x 6656, 104 per token), none with
a bias. For each shape the token's count of distinct random bf16 weights
sits on the device (a GEMV's time does not depend on the values) and x
is M bf16 rows. The routes:
  triton     NarrowLinear's forward (radix_ops.gemv_narrow at its BLOCK /
             NUM_WARPS), one launch for all M rows
  twin-mc    the twin's multi-column kernel (radix_ops._gemv_mc, RAW=True)
             with the whole row in one program at the same BLOCK /
             NUM_WARPS, stored straight to x's dtype (no bias): the same
             program with C a constexpr, where gemv_narrow's is a runtime
             argument
  mv         swap.stock_linear's one-row call: torch.mv with hipBLASLt
             preferred for that call (M = 1 only)
  mv-rows    that call once per row, M launches (every row the M=1 call's
             bits)
  rocblas    F.linear with rocBLAS preferred (PyTorch's default on gfx1151)
  hipblaslt  F.linear with hipBLASLt preferred
Each route is timed three ways:
  device  a pass over the token's weights with an event pair around each
          call and a filler read (--filler-mb, default 64 MB) enqueued
          before each pair, outside it: the weight arrives cold, as in the
          model, where hundreds of MB of other weights stream between two
          narrow calls, and the GPU queue is never empty when a pair opens
          (the host's launch cost stays outside it, as bench/
          parity_common.primer arranges for the rotation protocol).
          Per-call medians summed per pass; the median pass over --repeats.
  rotation  the rotation protocol as bench/parity_common.rotation runs it
          (back-to-back calls, an event pair around each, the primer at
          the head of each pass): the weights warm from the pass before
          when the token's narrow weights fit the last-level cache, and a
          call whose host launch outlasts the kernel before it times the
          host's launch inside its pair
  wall    back-to-back calls, no filler, no events, one synchronize per
          pass: the host-inclusive cost, what a launch-bound loop pays.
Every route's output is checked against a float64 reference (2e-3 of
|W|.|x|, bf16 rounding of the result), and whether row m of an M-row call
equals the same route's M=1 call bit for bit is recorded (`rows_bitwise`):
the speculative verify runs M = 2..8 against the serial M=1 step.

Prints NARROW_LINEAR_KNOBS PASS/FAIL; exits through os._exit (ROCm torch
can exit 0 after a failure: read the verdict line).
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import traceback

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "src")
sys.path.insert(0, "bench")
import parity_common as pc  # noqa: E402

# the menu's narrow Linears: RxC:count per token (module docstring)
SHAPES = ("48x5120:96", "32x4096:48", "256x6656:104")
ROUTES = ("triton", "twin-mc", "mv", "mv-rows", "rocblas", "hipblaslt")
LIB = {"rocblas": torch._C._BlasBackend.Cublas, "hipblaslt": torch._C._BlasBackend.Cublaslt}


def _parse(spec: str) -> tuple[int, int, int]:
    shape, _, n = spec.partition(":")
    r, c = shape.split("x")
    return int(r), int(c), int(n or 1)


def make_call(route: str, W: torch.Tensor, x: torch.Tensor):
    """The route's call for one weight, as a closure over its operands."""
    from drinkme.codec import radix_ops
    from drinkme.codec.swap import NarrowLinear, stock_linear

    M, C = x.shape
    R = W.shape[0]
    if route == "triton":
        lin = torch.nn.Linear(C, R, bias=False, device="meta")
        lin.weight = torch.nn.Parameter(W, requires_grad=False)
        mod = NarrowLinear.adopt(lin)
        return lambda: mod(x)
    if route == "twin-mc":
        B = NarrowLinear.BLOCK
        nb = -(-C // B)

        def twin_mc():
            y = torch.empty((M, R), dtype=x.dtype, device=x.device)
            radix_ops._gemv_mc[(R,)](y, x, W, W, W, W, W, C, radix_ops.MANT, radix_ops.EXP, (3, 8), B,
                                     radix_ops.DECODER_SCHEDULED, nb, 1, M, RAW=True,
                                     num_warps=NarrowLinear.NUM_WARPS)
            return y
        return twin_mc
    if route == "mv":
        return lambda: stock_linear(x, W, None, "mv")
    if route == "mv-rows":
        rows = [x[m:m + 1] for m in range(M)]
        return lambda: torch.cat([stock_linear(r, W, None, "mv") for r in rows])
    lib = LIB[route]

    def call():
        prev = torch._C._get_blas_preferred_backend()
        torch._C._set_blas_preferred_backend(lib)
        try:
            return F.linear(x, W)
        finally:
            torch._C._set_blas_preferred_backend(prev)
    return call


def device_pass(fns, filler: torch.Tensor, passes: int) -> list[float]:
    """ms per pass: the per-call event times summed, a filler read before
    each call outside its pair (module docstring)."""
    out = []
    for _ in range(passes + 1):  # the first pass warms and is dropped
        ev = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in fns]
        for (s, e), fn in zip(ev, fns):
            filler.sum()
            s.record()
            fn()
            e.record()
        torch.cuda.synchronize()
        out.append(sum(s.elapsed_time(e) for s, e in ev))
    return out[1:]


def wall_pass(fns, passes: int) -> list[float]:
    for fn in fns:
        fn()
    torch.cuda.synchronize()
    out = []
    for _ in range(passes):
        t0 = time.perf_counter()
        for fn in fns:
            fn()
        torch.cuda.synchronize()
        out.append((time.perf_counter() - t0) * 1e3)
    return out


@torch.inference_mode()
def run_shape(R: int, C: int, count: int, a, report: dict) -> None:
    g = torch.Generator(device="cuda").manual_seed(2026)
    Ws = [(torch.randn((R, C), device="cuda", generator=g) * 0.02).to(torch.bfloat16) for _ in range(count)]
    xall = torch.randn((8, C), device="cuda", generator=g).to(torch.bfloat16)
    filler = torch.ones(a.filler_mb * 2**18, dtype=torch.float32, device="cuda")
    rec = dict(shape=f"{R}x{C}", R=R, C=C, calls_per_token=count, weight_bytes=R * C * 2, ms={})
    ref64 = Ws[0].double() @ xall.double().T  # [R, 8]
    bound = 2e-3 * (Ws[0].double().abs() @ xall.double().abs().T) + 1e-6
    for M in a.ms:
        x = xall[:M].contiguous()
        cell = {}
        for route in a.routes:
            if route == "mv" and M != 1:
                continue
            if route == "mv-rows" and M == 1:
                continue
            y = make_call(route, Ws[0], x)().float()
            err = float(((y.double().T - ref64[:, :M]).abs() / bound[:, :M]).max())
            if not err <= 1:
                raise AssertionError(f"{R}x{C} M={M} {route}: output off the fp64 reference ({err:.3g} of the bound)")
            base = "mv" if route == "mv-rows" else route
            one = make_call(base, Ws[0], x[:1])().float()
            rows_bitwise = bool(torch.equal(y, one.expand(M, R))) if M == 1 else all(
                torch.equal(y[m], make_call(base, Ws[0], x[m:m + 1])().float().reshape(-1)) for m in range(M))
            fns = [make_call(route, W, x) for W in Ws]
            dev, rot, wall = [], [], []
            for _ in range(a.repeats):
                dev.append(float(np.median(device_pass(fns, filler, a.passes))))
                per, _totals = pc.rotation(fns, a.passes, prime=True)
                rot.append(sum(float(np.median(p)) for p in per))
                wall.append(float(np.median(wall_pass(fns, a.passes))))
            cell[route] = dict(device_ms_per_token=float(np.median(dev)), wall_ms_per_token=float(np.median(wall)),
                               rotation_ms_per_token=float(np.median(rot)),
                               device_us_per_call=float(np.median(dev)) * 1e3 / count,
                               rotation_us_per_call=float(np.median(rot)) * 1e3 / count,
                               wall_us_per_call=float(np.median(wall)) * 1e3 / count,
                               device_ms_per_repeat=dev, rotation_ms_per_repeat=rot, wall_ms_per_repeat=wall,
                               err_over_bound=err, rows_bitwise=rows_bitwise)
        best_dev = min(cell, key=lambda k: cell[k]["device_ms_per_token"])
        best_rot = min(cell, key=lambda k: cell[k]["rotation_ms_per_token"])
        best_wall = min(cell, key=lambda k: cell[k]["wall_ms_per_token"])
        rec["ms"][str(M)] = dict(routes=cell, best_device=best_dev, best_rotation=best_rot, best_wall=best_wall)
        print(f"  {R}x{C} x{count} M={M}: " + "  ".join(
            f"{k} {v['device_us_per_call']:.1f}/{v['rotation_us_per_call']:.1f}/{v['wall_us_per_call']:.1f}"
            for k, v in cell.items())
            + f"  -> {best_dev} / {best_rot} / {best_wall}", flush=True)
    report["shapes"].append(rec)
    del Ws, filler
    torch.cuda.empty_cache()


def load1() -> float:
    return os.getloadavg()[0]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shapes", nargs="*", default=list(SHAPES), help="RxC:count per token")
    ap.add_argument("--ms", nargs="*", type=int, default=list(range(1, 9)))
    ap.add_argument("--routes", nargs="*", default=list(ROUTES), choices=list(ROUTES))
    ap.add_argument("--passes", type=int, default=20)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--filler-mb", type=int, default=64)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    if not torch.cuda.is_available():
        print("NARROW_LINEAR_KNOBS FAIL: no GPU")
        return 2
    facts = pc.machine_facts()
    report = dict(instrument="narrow_linear_knobs", verdict="INCOMPLETE", machine=facts,
                  default_blas=str(torch.backends.cuda.preferred_blas_library()),
                  args=vars(a), shapes=[], started=time.strftime("%Y-%m-%d %H:%M:%S %Z"),
                  load1_before=load1(), cpu_count=os.cpu_count())
    status = 1
    try:
        from drinkme.codec.swap import NarrowLinear

        report["triton_config"] = dict(BLOCK=NarrowLinear.BLOCK, NUM_WARPS=NarrowLinear.NUM_WARPS)
        print(f"device {facts['device']} {facts.get('arch')} torch {facts['torch']} default BLAS "
              f"{report['default_blas']} load1 {report['load1_before']:.2f}; us per call, device/rotation/wall", flush=True)
        for spec in a.shapes:
            run_shape(*_parse(spec), a, report)
        report["load1_after"] = load1()
        print("BEST PER M (device / rotation / wall):", flush=True)
        for s in report["shapes"]:
            print(f"  {s['shape']:9s} " + "  ".join(
                f"M{m} {c['best_device']}/{c['best_rotation']}/{c['best_wall']}" for m, c in s["ms"].items()),
                flush=True)
        report["verdict"] = "PASS"
        print("NARROW_LINEAR_KNOBS PASS", flush=True)
        status = 0
    except BaseException as e:  # noqa: BLE001 — the verdict line is the contract
        report["verdict"] = "FAIL"
        report["error"] = f"{type(e).__name__}: {e}"
        print(f"NARROW_LINEAR_KNOBS FAIL: {type(e).__name__}: {e}", flush=True)
        traceback.print_exc()
    report["finished"] = time.strftime("%Y-%m-%d %H:%M:%S %Z")
    if a.json:
        pc.write_json(a.json, report)
        print(f"wrote {a.json}", flush=True)
    return status


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(rc)
