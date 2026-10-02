"""The narrow GEMV's numerics (codec/swap.NarrowLinear) on the REAL narrow
weights: every raw bf16 Linear NarrowLinear adopts in Qwen3.8-27B
(in_proj_a / in_proj_b, 48 x 5120, 96 of them), MiMo-V2.6-Distill-Qwen-9B
(32 x 4096, 48) and Muse-Glimmer-30B (k_proj / v_proj, 256 x 6656, 104),
read straight from the HF cache's safetensors: no model load, no download.
--synthetic N checks N seeded N(0, 0.02) bf16 weights at each model's
narrow shape instead (a machine without the checkpoints: the Modal L4 of
bench/modal_narrow_ab.py).

    PYTHONPATH=src flock /tmp/drinkme-gpu.lock .venv/bin/python bench/narrow_linear_numerics.py \
        --json out.json [--models Qwen3.8-27B MiMo-V2.6-Distill-Qwen-9B Muse-Glimmer-30B] [--synthetic 8]

Per tensor, x is eight seeded N(0, 1) bf16 rows, and each route is checked
three ways:
  rows   for M = 1..8, row m of the route's M-row call is its M=1 call on
         row m, bit for bit (the MTP verify's batched projections rest on
         it)
  bf16   max |y - y64| over the tensor's outputs at M = 8: y the route's
         bf16 output, y64 the float64 product of the same bf16 operands
  fp32   the same with x widened to fp32, so y is the fp32 accumulator
         itself: the reduction's error without the output rounding
The routes:
  triton  NarrowLinear's forward (radix_ops.gemv_narrow)
  linear  F.linear, the BLAS path NarrowLinear replaces (its rows are
          recorded, not claimed)
Every output must sit within 2e-3 of |W|.|x| of the float64 product (the
bound bench/narrow_linear_knobs.py applies), and the triton route's rows
must be bitwise. `bits_equal` records, per route, whether its outputs are
the triton route's bits.

Prints NARROW_LINEAR_NUMERICS PASS/FAIL; exits through os._exit (ROCm torch
can exit 0 after a failure: read the verdict line).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, "src")
sys.path.insert(0, "bench")
import parity_common as pc  # noqa: E402

# the menu models with narrow Linears, and their shape (module docstring)
MODELS = {"Qwen3.8-27B": (48, 5120), "MiMo-V2.6-Distill-Qwen-9B": (32, 4096), "Muse-Glimmer-30B": (256, 6656)}
ROUTES = ("triton", "linear")
CLAIMED = ("triton",)  # the routes whose rows must be bitwise


def synthetic_weights(name: str, count: int):
    """`count` seeded N(0, 0.02) bf16 weights at the model's narrow shape."""
    g = torch.Generator(device="cuda").manual_seed(7)
    R, C = MODELS[name]
    for i in range(count):
        yield f"synthetic.{i}", (torch.randn((R, C), device="cuda", generator=g) * 0.02).to(torch.bfloat16)


def narrow_weights(name: str):
    """(key, bf16 weight on the device) for every tensor of the menu model
    `name` that NarrowLinear would adopt: a 2-D bf16 `.weight` with R <=
    MAX_ROWS, C >= MIN_COLS, C % 4 == 0 and a dimension under the codec's
    1024 (NarrowLinear.wants over the loaded Linear)."""
    from safetensors import safe_open

    from drinkme.codec.swap import NarrowLinear
    from drinkme.serving.checkpoint import cached_snapshot_dir
    from drinkme.suggest import MODELS as MENU

    model = next(m for m in MENU if m.name == name)
    snap = cached_snapshot_dir(model.hf_repo, model.revision)
    if snap is None:
        raise FileNotFoundError(f"{model.hf_repo}@{model.revision} is not in the HF cache")
    index = json.loads((Path(snap) / "model.safetensors.index.json").read_text())["weight_map"]
    shards: dict[str, list[str]] = {}
    for key, shard in index.items():
        if key.endswith(".weight"):
            shards.setdefault(shard, []).append(key)
    for shard, keys in sorted(shards.items()):
        with safe_open(str(Path(snap) / shard), "pt") as f:
            for key in sorted(keys):
                sl = f.get_slice(key)
                shape = sl.get_shape()
                if (len(shape) == 2 and sl.get_dtype() == "BF16" and shape[0] <= NarrowLinear.MAX_ROWS
                        and shape[1] >= NarrowLinear.MIN_COLS and shape[1] % 4 == 0 and min(shape) < 1024):
                    yield key, f.get_tensor(key).to("cuda")


def make_call(route: str, W: torch.Tensor):
    from drinkme.codec.swap import NarrowLinear

    if route == "triton":
        lin = torch.nn.Linear(W.shape[1], W.shape[0], bias=False, device="meta")
        lin.weight = torch.nn.Parameter(W, requires_grad=False)
        return NarrowLinear.adopt(lin)
    if route == "linear":
        return lambda x: F.linear(x, W.to(x.dtype))  # an fp32 x: the exact fp32 widening of W
    raise ValueError(route)


@torch.inference_mode()
def run_model(name: str, a, report: dict) -> bool:
    g = torch.Generator(device="cuda").manual_seed(2026)
    rec = dict(model=name, tensors=0, shapes=[], routes={r: dict(max_abs_bf16=0.0, max_abs_fp32=0.0,
                                                                 max_err_over_bound=0.0, rows_bitwise=True,
                                                                 bits_equal=True) for r in a.routes})
    ok = True
    weights = synthetic_weights(name, a.synthetic) if a.synthetic else narrow_weights(name)
    for key, W in weights:
        R, C = W.shape
        if f"{R}x{C}" not in rec["shapes"]:
            rec["shapes"].append(f"{R}x{C}")
        rec["tensors"] += 1
        x = torch.randn((8, C), device="cuda", generator=g).to(torch.bfloat16)
        y64 = x.double() @ W.double().T  # [8, R]
        bound = 2e-3 * (x.double().abs() @ W.double().abs().T) + 1e-6
        want = None
        for route in a.routes:
            fn = make_call(route, W)
            r = rec["routes"][route]
            y = fn(x)
            y32 = fn(x.float())
            r["max_abs_bf16"] = max(r["max_abs_bf16"], float((y.double() - y64).abs().max()))
            r["max_abs_fp32"] = max(r["max_abs_fp32"], float((y32.double() - y64).abs().max()))
            eob = float(((y.double() - y64).abs() / bound).max())
            r["max_err_over_bound"] = max(r["max_err_over_bound"], eob)
            if not eob <= 1:
                ok = False
                print(f"  {name} {key} {route}: off the float64 product ({eob:.3g} of the bound)", flush=True)
            if want is None:
                want = (y, y32)
            elif not (torch.equal(y, want[0]) and torch.equal(y32, want[1])):
                r["bits_equal"] = False
            ones = [fn(x[m:m + 1]) for m in range(8)]
            for M in range(1, 9):
                yM = fn(x[:M].contiguous())
                same = all(torch.equal(yM[m:m + 1], ones[m]) for m in range(M))
                if not same:
                    r["rows_bitwise"] = False
                    if route in CLAIMED:
                        ok = False
                        print(f"  {name} {key} {route}: M={M} rows are not the M=1 call's bits", flush=True)
        del W
    torch.cuda.empty_cache()
    if rec["tensors"] == 0:
        raise AssertionError(f"{name}: no narrow tensor found")
    report["models"].append(rec)
    print(f"  {name}: {rec['tensors']} tensors {', '.join(rec['shapes'])}: " + "  ".join(
        f"{k} bf16 {v['max_abs_bf16']:.3g} fp32 {v['max_abs_fp32']:.3g} rows {'bitwise' if v['rows_bitwise'] else 'DIFFER'}"
        f"{'' if v['bits_equal'] else ' (bits differ from ' + a.routes[0] + ')'}"
        for k, v in rec["routes"].items()), flush=True)
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="*", default=list(MODELS), choices=list(MODELS))
    ap.add_argument("--routes", nargs="*", default=list(ROUTES), choices=list(ROUTES))
    ap.add_argument("--synthetic", type=int, default=0, help="N seeded weights per model, not the checkpoints")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    if not torch.cuda.is_available():
        print("NARROW_LINEAR_NUMERICS FAIL: no GPU")
        return 2
    report = dict(instrument="narrow_linear_numerics", verdict="INCOMPLETE", machine=pc.machine_facts(),
                  args=vars(a), models=[], started=time.strftime("%Y-%m-%d %H:%M:%S %Z"))
    status = 1
    try:
        ok = True
        for name in a.models:
            ok = run_model(name, a, report) and ok
        if not ok:
            raise AssertionError("a claimed property failed (lines above)")
        report["verdict"] = "PASS"
        print("NARROW_LINEAR_NUMERICS PASS", flush=True)
        status = 0
    except BaseException as e:  # noqa: BLE001 — the verdict line is the contract
        report["verdict"] = "FAIL"
        report["error"] = f"{type(e).__name__}: {e}"
        print(f"NARROW_LINEAR_NUMERICS FAIL: {type(e).__name__}: {e}", flush=True)
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
