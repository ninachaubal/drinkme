"""The runtime's kernel routes over a loaded module tree, in ONE place.

Once a model's weights are on the device, drinkme selects these operations in
transformers' forward — none changes a weight:
  * the DeltaNet decode recurrence is bound to fla's fused Triton kernel
    (serving/deltanet.py: one launch per layer-step instead of ~60);
  * the raw Linears under the codec's row threshold (Qwen3.8's 48-row
    in_proj_a / in_proj_b) are adopted by swap.NarrowLinear, the Triton
    GEMV, instead of the BLAS one-workgroup GEMM (codec/swap.py).
  * large ROCm DeltaNet convolutions use FLA's time-major Triton path
    after a startup check (serving/deltanet_conv.py);
  * the raw Linears at or above the threshold (the stock arm's every
    Linear, a tied lm_head in the others) make their one-row call through
    the box's stock GEMV (codec/swap.py's STOCK_GEMV: on gfx1151 the twin
    arm's Triton GEMV over the raw weight, where PyTorch's default F.linear
    is rocBLAS at a quarter of the wall and torch.mv under hipBLASLt is
    slower than the twin's kernel), swap.StockLinear; in a codec tree (the
    compressed arm, the twin, `drinkme serve` over a pack) through the
    box's raw GEMV instead (codec/swap.py's RAW_GEMV: the twin's kernel on
    gfx1151, 0.67-0.94 of torch.mv's time on the tied lm_heads measured);
  * on CUDA, attention runs without torch's cuDNN SDPA backend
    (sdpa.route, process-wide: it builds a plan per new sequence length on
    every thread). Not in the record below.
Arms routed differently would compare two runtimes and call it a
weight-read ratio (~7% in the compressed arm's favour on the 27B, when
the stock arm kept the torch recurrence and F.linear).

`route_kernels(model, device)` is the one call. It does exactly what the
loaders need — recurrence, prefill convolution, narrow, stock and raw GEMV
routes and their knobs — and returns a plain record of
what it did:

    {"deltanet_kernel": "fla" | "torch" | "none",   # deltanet.route's answer
     "deltanet_conv": "fla" | "torch" | "none",    # prefill convolution
     "narrow_gemv": bool,                          # False iff DRINKME_NARROW_GEMV=0
     "narrow_count": int,                          # NarrowLinears the tree now holds
     "stock_gemv": "linear" | "mv" | "triton",     # the raw Linears' one-row call
     "raw_gemv": "stock" | "twin"}                 # ... in a codec tree

so the bench can print it per arm (`describe`), keep it in the record's
raw (`raw.<arm>_routing`) and refuse a record whose arms routed
differently (arms.refuse_unless_routed_alike). engines.load_stock,
engines.load_compressed and every arm arms.run_arms times call it, and
nothing else calls deltanet.route, deltanet_conv.route or install_narrow, so the paths
cannot drift (tests/test_bench_route_alike.py pins the call sites).
Idempotent: a second call finds the adapter and the adopted Linears,
preserves those selections and returns the SAME
record — the count is what the tree holds, not what this call adopted.

Whatever the route, the weights are the same bits: what changes is the
recurrence's rounding order (deltanet.py's docstring), the convolution's
(deltanet_conv.py), and the GEMV's (NarrowLinear's), the same on every arm
once every arm goes through here.
"""

from __future__ import annotations

import os

from .. import sdpa
from . import deltanet, deltanet_conv

NARROW_ENV = "DRINKME_NARROW_GEMV"


def route_kernels(model, device: str) -> dict:
    """Bind the recurrence and prefill convolution and adopt narrow Linears
    for `model` on `device`, report the routes, and return the
    record above. The narrow adoption comes first, as the loaders always
    did it: NarrowLinear.adopt keeps the same Parameter objects, and the
    route's probe never looks at a Linear."""
    # the attention backend (sdpa.route: cuDNN attention excluded on CUDA),
    # process-wide and so alike on every arm; not in the record, whose keys
    # compose the published engine string
    sdpa.route(device)
    narrow_gemv, narrow_count = _install_narrow(model, device)
    stock_gemv, raw_gemv = _install_stock_gemv(model, device)
    kernel = deltanet.route(model, device)
    conv = deltanet_conv.route(model, device)
    # the modes, not counts: the stock arm holds every Linear raw and no
    # codec tensor, the others only what the codec left, and every arm
    # must record alike
    return {"deltanet_kernel": kernel, "deltanet_conv": conv,
            "narrow_gemv": narrow_gemv, "narrow_count": narrow_count, "stock_gemv": stock_gemv,
            "raw_gemv": raw_gemv}


def describe(route: dict) -> str:
    """The one-line form a human reads in a sweep log beside each arm:
    'deltanet fla, narrow 96, conv fla'; 'narrow off (DRINKME_NARROW_GEMV=0)' when
    the knob kept F.linear."""
    narrow = (f"narrow {route['narrow_count']}" if route["narrow_gemv"]
              else f"narrow off ({NARROW_ENV}=0)")
    stock = "" if route.get("stock_gemv", "linear") == "linear" else f", stock gemv {route['stock_gemv']}"
    raw = "" if route.get("raw_gemv", "stock") == "stock" else f", raw gemv {route['raw_gemv']}"
    return (f"deltanet {route['deltanet_kernel']}, {narrow}, conv {route.get('deltanet_conv', 'torch')}"
            f"{stock}{raw}")


def _install_narrow(model, device: str) -> tuple[bool, int]:
    """The raw Linears under the codec's row threshold onto
    swap.NarrowLinear — the Triton GEMV instead of the BLAS one-workgroup
    GEMM (the class docstring has the numbers).
    After the weights are on the device, so the same Parameter objects are
    adopted; DRINKME_NARROW_GEMV=0 keeps F.linear (the A/B instrument).
    One line says what was done. Returns (the knob allowed it, how many
    NarrowLinears the tree holds now — adopted here or earlier)."""
    from ..codec.ops import MC_MAX
    from ..codec.swap import NarrowLinear, install_narrow

    on = os.environ.get(NARROW_ENV, "1") != "0"
    if not on:
        print(f"[drinkme] narrow gemv: {NARROW_ENV}=0 — F.linear (BLAS) on every raw Linear",
              flush=True)
    else:
        done = install_narrow(model, device)
        if done:
            shapes = sorted({f"{r}x{c}" for _, (r, c) in done})
            print(f"[drinkme] narrow gemv: {len(done)} x NarrowLinear ({', '.join(shapes)} bf16; "
                  f"Triton GEMV B{NarrowLinear.BLOCK}/w{NarrowLinear.NUM_WARPS} for M<={MC_MAX}, "
                  f"F.linear above; {NARROW_ENV}=0 for F.linear)", flush=True)
    return on, sum(1 for m in model.modules() if isinstance(m, NarrowLinear))


def _install_stock_gemv(model, device: str) -> tuple[str, str]:
    """The box's stock GEMV (swap.stock_gemv_mode) and, in a codec tree,
    its raw GEMV (swap.raw_gemv_mode) onto the raw Linears: swap.StockLinear
    adopts them, RawLinears are told the modes. After the narrow adoption,
    so a narrow Linear stays NarrowLinear. One line per route says what was
    done; returns the two modes."""
    from ..codec.swap import (RAW_GEMV_ENV, STOCK_GEMV_ENV, RawLinear, StockLinear, install_stock_gemv,
                              raw_gemv_mode, stock_gemv_mode)

    mode, raw = stock_gemv_mode(device), raw_gemv_mode(device)
    if install_stock_gemv(model, device, mode, raw):
        routed = [m for m in model.modules() if isinstance(m, (StockLinear, RawLinear))]
        twin = sum(1 for m in routed if m.raw_gemv is not None)
        if mode == "triton":
            print(f"[drinkme] stock gemv: {twin} raw Linear(s) make their one-row call through the twin's "
                  f"Triton GEMV over the raw weight; {STOCK_GEMV_ENV}=mv for torch.mv under hipBLASLt",
                  flush=True)
        elif twin:
            print(f"[drinkme] raw gemv: {twin} raw Linear(s) of the codec tree make their one-row call "
                  f"through the twin's Triton GEMV; {RAW_GEMV_ENV}=stock for the stock gemv", flush=True)
        if mode == "mv" and len(routed) > twin:
            print(f"[drinkme] stock gemv: {len(routed) - twin} raw Linear(s) make their one-row call through "
                  f"torch.mv under hipBLASLt; {STOCK_GEMV_ENV}=linear for F.linear", flush=True)
    return mode, raw
