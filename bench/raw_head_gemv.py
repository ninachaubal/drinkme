"""A tied lm_head's one-row call (a raw bf16 Linear the codec leaves alone:
codec/swap.py's StockLinear in the compressed and twin arms) under the
stock GEMV and under each GEMV of ours that reads raw bf16: which one the
compressed arm's raw Linears should launch at M=1.

    PYTHONPATH=src flock /tmp/drinkme-gpu.lock .venv/bin/python bench/raw_head_gemv.py \
        --json out.json [--models Qwen/Qwen3-0.6B google/gemma-4-31B-it] [--shapes 151936x1024]

The shape is each model's vocabulary projection off its config (the HF
cache's config.json: vocab_size x hidden_size), as random bf16 weights on
the device (a GEMV's time does not depend on the values) and one bf16 row
of x. The routes:
  mv         swap.stock_linear's one-row call: torch.mv with hipBLASLt
             preferred for that call (what StockLinear runs today on gfx1151)
  rocblas    F.linear with rocBLAS preferred (PyTorch's default on gfx1151)
  hipblaslt  F.linear with hipBLASLt preferred
  twin:tTwW  radix_ops.gemv_fused over a swap.to_device_twin dict: the
             radix M=1 kernel with RAW=True, what the twin arm's
             RadixTwinLinear launches, at T blocks per program (0 = the
             whole row) and W warps. twin:t0w8 is the twin's own row for
             the head class on gfx1151 (radix_schedule.TWIN_TABLES)
  narrow:bBwW  radix_ops.gemv_narrow at M=1, BLOCK B and W warps:
             NarrowLinear's kernel, one program per output row.
             narrow:b512w4 is NarrowLinear's own config
Every route runs in one rotation (bench/parity_common.rotation, with the
primer): back-to-back calls, an event pair around each, the per-call
median over --passes, the median over --repeats. Consecutive routes read
two different copies of the weight, so no call finds the previous call's
tail in the last-level cache. `wall` is the same route called
back to back --passes times, one synchronize: the host-inclusive cost.

Every route's output is checked against a float64 reference (2e-3 of
|W|.|x|), and against the mv route's bits: bitwise, and the largest
difference in bf16 units in the last place.

The gate (`parity`, every shape, with and without a bias): a toy codec tree
holding the weight as a Linear beside the twin arm's module over the same
bits (swap.RadixTwinLinear), routed by swap.install_stock_gemv at this box's
stock and raw GEMV modes, as serving/kernel_route.route_kernels routes the
compressed and twin arms. Under raw GEMV "twin" the Linear's one-row
output is bitwise the twin module's (the same kernel at the same launch
row); every other row count is bitwise F.linear, today's route; under
raw GEMV "stock" the one-row output is bitwise the stock GEMV's
(swap.stock_linear), today's route. Then the same Linear in a tree with
no codec module (the stock arm, `serve --stock`), routed the same way:
its one-row output is bitwise the stock GEMV's, and under stock GEMV
"triton" also bitwise the twin module's; every other row count is
bitwise F.linear. Any failure fails the run.

Prints RAW_HEAD_GEMV PASS/FAIL; exits through os._exit (ROCm torch can
exit 0 after a failure: read the verdict line).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "src")
sys.path.insert(0, "bench")
import parity_common as pc  # noqa: E402

MODELS = ("Qwen/Qwen3-0.6B", "Qwen/Qwen3-1.7B", "Qwen/Qwen3-4B", "google/gemma-4-31B-it")
TWIN = tuple(f"twin:t{t}w{w}" for t in (0, 1, 2, 4) for w in (1, 2, 4, 8))
NARROW = tuple(f"narrow:b{b}w{w}" for b in (256, 512, 1024, 2048) for w in (1, 2, 4, 8))
ROUTES = ("mv", "rocblas", "hipblaslt") + TWIN + NARROW
LIB = {"rocblas": torch._C._BlasBackend.Cublas, "hipblaslt": torch._C._BlasBackend.Cublaslt}


def head_shape(model_id: str) -> tuple[int, int]:
    from huggingface_hub import hf_hub_download

    cfg = json.loads(Path(hf_hub_download(model_id, "config.json", local_files_only=True)).read_text())
    cfg = cfg.get("text_config", cfg)
    return int(cfg["vocab_size"]), int(cfg["hidden_size"])


def _cfg(route: str) -> tuple[int, int]:
    body = route.split(":", 1)[1]
    a, w = body[1:].split("w")
    return int(a), int(w)


def make_call(route: str, W: torch.Tensor, x: torch.Tensor):
    """The route's call for one weight, as a closure over its operands; x
    is [1, 1, C] bf16, every call returns [1, R] (or [1, 1, R])."""
    from drinkme.codec import radix_ops
    from drinkme.codec.swap import stock_linear, to_device_twin

    R, C = W.shape
    if route == "mv":
        return lambda: stock_linear(x, W, None, "mv")
    if route in LIB:
        lib = LIB[route]

        def call():
            prev = torch._C._get_blas_preferred_backend()
            torch._C._set_blas_preferred_backend(lib)
            try:
                return F.linear(x, W)
            finally:
                torch._C._set_blas_preferred_backend(prev)
        return call
    if route.startswith("twin:"):
        tiles, warps = _cfg(route)
        p = to_device_twin(W, (3, 8), "cuda")
        p["rx_launch"] = dict(p["rx_launch"], gemv_tiles=tiles, gemv_warps=warps)
        xr = x.reshape(-1)
        return lambda: radix_ops.gemv_fused(p, xr, None, x.dtype)
    if route.startswith("narrow:"):
        block, warps = _cfg(route)
        xm = x.reshape(1, C).contiguous()
        return lambda: radix_ops.gemv_narrow(W, xm, None, block, warps)
    raise ValueError(route)


def wall_ms(fn, passes: int) -> float:
    fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(passes):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1e3 / passes


def ulps(a: torch.Tensor, b: torch.Tensor) -> int:
    """The largest difference between two bf16 vectors in units in the last
    place (the bit patterns as sign-magnitude integers)."""
    def key(t):
        i = t.reshape(-1).contiguous().view(torch.int16).to(torch.int32)
        return torch.where(i < 0, -(i & 0x7FFF), i)
    return int((key(a) - key(b)).abs().max())


@torch.inference_mode()
def run_shape(label: str, R: int, C: int, a, wall_bytes_s: float, report: dict) -> None:
    g = torch.Generator(device="cuda").manual_seed(2026)
    W0 = (torch.randn((R, C), device="cuda", generator=g) * 0.02).to(torch.bfloat16)
    W1 = W0.clone()
    x = torch.randn((1, 1, C), device="cuda", generator=g).to(torch.bfloat16)
    byts = R * C * 2
    y64 = torch.empty(R, dtype=torch.float64, device="cuda")
    bound = torch.empty_like(y64)
    xd = x.reshape(-1).double()
    for r0 in range(0, R, 16384):
        w = W0[r0:r0 + 16384].double()
        y64[r0:r0 + 16384] = w @ xd
        bound[r0:r0 + 16384] = 2e-3 * (w.abs() @ xd.abs()) + 1e-6
        del w
    want = make_call("mv", W0, x)().reshape(-1)
    rec = dict(model=label, shape=f"{R}x{C}", R=R, C=C, weight_bytes=byts, routes={})
    fns, names = [], []
    for route in a.routes:
        try:
            y = make_call(route, W0, x)().reshape(-1)
        except Exception as e:  # noqa: BLE001 — a config the compiler refuses is a result, not a crash
            rec["routes"][route] = dict(error=f"{type(e).__name__}: {str(e)[:200]}")
            print(f"  {label} {route}: {rec['routes'][route]['error']}", flush=True)
            continue
        err = float(((y.double() - y64).abs() / bound).max())
        if not err <= 1:
            raise AssertionError(f"{label} {route}: output off the fp64 reference ({err:.3g} of the bound)")
        rec["routes"][route] = dict(err_over_bound=err, bitwise_mv=bool(torch.equal(y, want)),
                                    max_ulps_vs_mv=ulps(y, want))
        fns.append(make_call(route, W0 if len(fns) % 2 == 0 else W1, x))
        names.append(route)
    dev = {n: [] for n in names}
    for _ in range(a.repeats):
        per, _totals = pc.rotation(fns, a.passes, prime=True)
        for n, p in zip(names, per):
            dev[n].append(float(np.median(p)))
    for n, fn in zip(names, fns):
        ms = float(np.median(dev[n]))
        rec["routes"][n].update(device_ms=ms, device_ms_per_repeat=dev[n], wall_ms=wall_ms(fn, a.passes),
                                gb_s=byts / ms / 1e6, of_wall=byts / wall_bytes_s * 1e3 / ms)
    timed = [n for n in names]
    mv = rec["routes"]["mv"]["device_ms"] if "mv" in rec["routes"] else None
    for fam in ("twin", "narrow"):
        cands = [n for n in timed if n.startswith(fam + ":")]
        if cands:
            rec[f"best_{fam}"] = min(cands, key=lambda n: rec["routes"][n]["device_ms"])
    rec["best"] = min(timed, key=lambda n: rec["routes"][n]["device_ms"])
    if mv:
        for n in timed:
            rec["routes"][n]["vs_mv"] = rec["routes"][n]["device_ms"] / mv
    rec["parity"] = parity(W0, x, a)
    report["shapes"].append(rec)
    show = [n for n in ("mv", "rocblas", "hipblaslt", "twin:t0w8", "narrow:b512w4",
                        rec.get("best_twin"), rec.get("best_narrow")) if n in rec["routes"]]
    print(f"  {label:22s} {R}x{C} ({byts / 1e9:.3f} GB): " + "  ".join(
        f"{n} {rec['routes'][n]['device_ms']:.3f} ms ({rec['routes'][n]['of_wall']:.2f})"
        for n in dict.fromkeys(show)) + f"  -> {rec['best']}", flush=True)
    del W0, W1, y64, bound, fns
    torch.cuda.empty_cache()


def parity(W: torch.Tensor, x: torch.Tensor, a) -> dict:
    """The module docstring's gate over weight W: raises on the first
    output that is not the bits it must be; returns what was compared."""
    from drinkme.codec import swap

    R, C = W.shape
    g = torch.Generator(device="cuda").manual_seed(7)
    stock_mode, raw_mode = swap.stock_gemv_mode("cuda"), swap.raw_gemv_mode("cuda")
    out = dict(stock_gemv=stock_mode, raw_gemv=raw_mode, checks=[])
    for has_bias in (False, True):
        bias = (torch.randn(R, device="cuda", generator=g) * 0.02).to(torch.bfloat16) if has_bias else None
        tree = torch.nn.Module()
        tree.head = torch.nn.Linear(C, R, bias=has_bias, device="meta")
        tree.head.weight = torch.nn.Parameter(W, requires_grad=False)
        if has_bias:
            tree.head.bias = torch.nn.Parameter(bias, requires_grad=False)
        tree.twin = swap.RadixTwinLinear(swap.to_device_twin(W, (3, 8), "cuda"), bias)
        for raw in ("twin", "stock"):
            n = swap.install_stock_gemv(tree, "cuda", stock_mode, raw)
            if n != 1 or type(tree.head) is not swap.StockLinear:
                raise AssertionError(f"{R}x{C}: install_stock_gemv routed {n} module(s), head "
                                     f"{type(tree.head).__name__}")
            y = tree.head(x)
            if raw == "twin":
                want, what = tree.twin(x), "the twin arm's module"
            else:
                want, what = swap.stock_linear(x, W, bias, stock_mode), f"the stock GEMV ({stock_mode})"
            same = torch.equal(y, want)
            out["checks"].append(dict(bias=has_bias, raw_gemv=raw, rows=1, against=what, bitwise=same))
            if not same:
                raise AssertionError(f"{R}x{C} raw gemv {raw} bias {has_bias}: the one-row output is not "
                                     f"{what} bit for bit")
            for rows in (2, 5, 8, 64):
                xm = torch.randn((1, rows, C), device="cuda", generator=g).to(torch.bfloat16)
                same = torch.equal(tree.head(xm), torch.nn.functional.linear(xm, W, bias))
                out["checks"].append(dict(bias=has_bias, raw_gemv=raw, rows=rows, against="F.linear",
                                          bitwise=same))
                if not same:
                    raise AssertionError(f"{R}x{C} raw gemv {raw} bias {has_bias} rows {rows}: not F.linear "
                                         "bit for bit")
        # the stock arm's tree: no codec module beside the head
        stock = torch.nn.Module()
        stock.head = torch.nn.Linear(C, R, bias=has_bias, device="meta")
        stock.head.weight = tree.head.weight
        stock.head.bias = tree.head.bias
        n = swap.install_stock_gemv(stock, "cuda", stock_mode, raw_mode)
        routed = n == 1 and type(stock.head) is swap.StockLinear
        if stock_mode != "linear" and not routed:
            raise AssertionError(f"{R}x{C}: install_stock_gemv routed {n} module(s) in the stock tree")
        y = stock.head(x)
        wants = [(swap.stock_linear(x, W, bias, stock_mode), f"the stock GEMV ({stock_mode})")]
        if stock_mode == "triton":
            wants.append((tree.twin(x), "the twin arm's module"))
        for want, what in wants:
            same = torch.equal(y, want)
            out["checks"].append(dict(bias=has_bias, tree="stock", rows=1, against=what, bitwise=same))
            if not same:
                raise AssertionError(f"{R}x{C} stock tree bias {has_bias}: the one-row output is not "
                                     f"{what} bit for bit")
        for rows in (2, 5, 8, 64):
            xm = torch.randn((1, rows, C), device="cuda", generator=g).to(torch.bfloat16)
            same = torch.equal(stock.head(xm), torch.nn.functional.linear(xm, W, bias))
            out["checks"].append(dict(bias=has_bias, tree="stock", rows=rows, against="F.linear", bitwise=same))
            if not same:
                raise AssertionError(f"{R}x{C} stock tree bias {has_bias} rows {rows}: not F.linear bit for bit")
        # the head back to a plain Linear for the next bias
        del tree, stock
    print(f"    parity {R}x{C}: {len(out['checks'])} checks bitwise (stock gemv {stock_mode}, "
          f"raw gemv {raw_mode} on this box)", flush=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="*", default=list(MODELS))
    ap.add_argument("--shapes", nargs="*", default=[], help="extra RxC shapes")
    ap.add_argument("--routes", nargs="*", default=list(ROUTES))
    ap.add_argument("--passes", type=int, default=20)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    if not torch.cuda.is_available():
        print("RAW_HEAD_GEMV FAIL: no GPU")
        return 2
    from drinkme.probe import measure_bandwidth

    report = dict(instrument="raw_head_gemv", verdict="INCOMPLETE", machine=pc.machine_facts(),
                  args=vars(a), shapes=[], started=time.strftime("%Y-%m-%d %H:%M:%S %Z"),
                  load1_before=os.getloadavg()[0])
    status = 1
    try:
        bw = measure_bandwidth()
        report["bandwidth"] = bw
        wall = float(bw["read_bytes_s"])
        print(f"device {report['machine']['device']} {report['machine'].get('arch')} torch "
              f"{report['machine']['torch']} read wall {wall / 1e9:.1f} GB/s, load {os.getloadavg()[0]:.2f}",
              flush=True)
        for model in a.models:
            R, C = head_shape(model)
            run_shape(model.split("/")[-1], R, C, a, wall, report)
        for spec in a.shapes:
            R, C = (int(v) for v in spec.split("x"))
            run_shape(spec, R, C, a, wall, report)
        report["verdict"] = "PASS"
        print("RAW_HEAD_GEMV PASS", flush=True)
        status = 0
    except BaseException as e:  # noqa: BLE001 — the verdict line is the contract
        report["verdict"] = "FAIL"
        report["error"] = f"{type(e).__name__}: {e}"
        print(f"RAW_HEAD_GEMV FAIL: {type(e).__name__}: {e}", flush=True)
        traceback.print_exc()
    report["load1_after"] = os.getloadavg()[0]
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
