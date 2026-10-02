"""THE ORDER-MATCHED TWIN GATE: the bench's twin (swap.RadixTwinLinear over
swap.to_device_twin — the served radix kernels with RAW=True, the decode
replaced by a load of the raw bf16 weight) against the compressed module
(swap.RadixCompressedLinear over swap.to_device_radix) on the same tensor,
the same activations and the same launch schedule.

    # on a GPU (hold the GPU lock; ROCm torch can exit 0 after a failure,
    # so read the verdict line):
    PYTHONPATH=src flock /tmp/drinkme-gpu.lock .venv/bin/python bench/radix_twin_bitpin.py [--json out.json]
    PYTHONPATH=src ... bench/radix_twin_bitpin.py --no-real      # no Qwen3-8B packs on this box
    PYTHONPATH=src ... bench/radix_twin_bitpin.py --selftest     # must print FAIL
    # on the CPU, no device touched (triton.compile for explicit targets):
    PYTHONPATH=src .venv/bin/python bench/radix_twin_bitpin.py --compile

The decode is lossless and everything after it is the same source, so the
claim is BITWISE equality of the two modules' outputs:

  kernels   radix_ops.gemv_fused (M=1, with and without a bias; bf16 out),
            radix_ops.gemv (M=1, fp32 out) and radix_ops.gemv_mc (M=2..8)
            at every launch-table row (codec/radix_schedule.TABLES: each
            box class's row for the tensor, via override="table=<box>"),
            the `spike` schedule, and a tiles x warps grid.
  module    the two modules' forward at this box's own table row and at
            the twin's own row (radix_schedule.select_twin: where the
            bench's twin arm runs), both modules at the same row each time,
            M = 1..8 (the fused M=1 route and the mc route), 9 and 33 (the
            dense route: F.linear over the decoded weight vs over the raw
            one).
  tensors   fixtures (the 65,536 bf16 bit patterns, NaN positions compared
            as positions; ragged shapes), synthetic weights at real layer
            widths (Qwen3-0.6B, Qwen3-8B, Qwen2.5-7B and a Qwen2.5-72B
            down_proj-width row: C = 3584, 18944 and 29568 are not multiples
            of the 1024-weight block), and — with the packs present — the
            real Qwen3-8B layer-0 tensors and lm_head off the sip pack and
            layer 0 off the gulp pack (bench/parity_common.py), every
            profile (sip, balanced, gulp) for the fixtures and synthetics.

The verdict GATES the schedules a box serves (every table row and spike)
and the twin's own rows (radix_schedule.TWIN_TABLES: the bench's twin arm
runs there, and the compressed kernels at the same row must match it),
and RECORDS the grid, which runs on the fixtures and the first tensor of
each model's widths (--grid all: every tensor; each schedule is a Triton
compile per tensor width, so the full grid is hours of compiling);
--strict gates the grid too. On a mismatch the case
is re-run with fp fusion off in both kernels (radix_ops.FUSION) and the
receipt says whether that made them equal.

TWO WAYS the same source can compile to a different sum:

  the layout  tl.sum(acc) adds each thread's share of the accumulator in
              the loop's register layout, which Triton takes from the
              widest vectorized load feeding it. The decoder's loads are
              gathers, so in the served kernel the activation's load sets
              it (16 bytes: 8 elements of a bf16 x, 4 of an fp32 x); the
              twin caps its weight load at the same vector
              (radix_kernel_gpu._raw_weight). Uncapped, the twin's 8-wide
              bf16 load outvoted an fp32 x's 4 and every fp32-x GEMV at a
              C both loads could vectorize differed in the last bits (the
              first gfx1151 run of this gate: every table row at C = 1024
              to 12288). --compile compares the two kernels' layout of
              every fp32 product and gates on it.
  fp fusion   (radix_ops.FUSION: enable_fp_fusion). The twin's loop body is
              much smaller than the decoder's, so the compiler may unroll
              the twin's loop where it keeps the served one rolled; on a
              row whose C is not a multiple of B the unrolled copy can
              prove whole blocks in bounds, drop the tl.where and contract
              those products into FMAs the served kernel computes as a
              multiply, a select and an add. A bf16 x times a bf16 weight
              is exact in fp32, so only an fp32 x can round differently.
              --compile records how each product reaches its accumulator
              in the optimized LLIR; the device gate decides.

--compile runs on the CPU: it compiles both specializations for explicit
targets (gfx1151, gfx1102, sm_80, sm_89, sm_90) at the pointer
specialization a device launch gets (16-byte alignment, and on ROCm the
2 GB pointer range that selects buffer loads — without it no load
vectorizes, which is how the layout difference went unseen), with a bf16
and an fp32 activation, and checks that the twin's accumulator layout is
the served kernel's, in every specialization.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import traceback

os.environ.setdefault("DRINKME_PREFILL_DENSE_MIN", "9")

# --compile's workers (--_dump) compile whichever tree PYTHONPATH names —
# the base ref's included, which has no twin — so they import only triton
# and that tree's kernels, inside _compile_one.
if "--_dump" not in sys.argv:
    import numpy as np  # noqa: E402
    import torch  # noqa: E402
    import triton  # noqa: E402

    sys.path.insert(0, "src")
    sys.path.insert(0, "bench")
    from drinkme.codec import radix_ops, radix_pack as rp, radix_schedule as rs  # noqa: E402
    from drinkme.codec.ops import MC_MAX  # noqa: E402
    from drinkme.codec.swap import (RadixCompressedLinear, RadixTwinLinear, to_device_radix,  # noqa: E402
                                    to_device_twin)
else:
    MC_MAX = 8

PROFILES = ("sip", "balanced", "gulp")
RAGGED = ((7, 33), (3, 1025), (129, 260), (17, 3079))
# synthetic weights at real layer widths (the eligible Linears: min dim >= 1024)
REAL_WIDTHS = {
    "qwen3-0.6b": ((2048, 1024), (1024, 1024), (1024, 2048), (3072, 1024), (1024, 3072)),
    "qwen3-8b": ((4096, 4096), (1024, 4096), (12288, 4096), (4096, 12288)),
    "qwen2.5-7b": ((3584, 3584), (18944, 3584), (3584, 18944)),
    "qwen2.5-72b-row": ((1024, 29568),),
    "head-class": ((32768, 1024),),
}
GRID_WARPS = (1, 2, 4, 8)  # every warp count a table row names (radix_schedule._validate's set)
GRID_TILES = (1, 2, 4, 0)  # 0 = the whole row
MC_GRID = ((1, 1), (1, 2), (1, 4), (2, 1), (0, 1))
MODULE_ROWS = tuple(range(1, MC_MAX + 1)) + (MC_MAX + 1, 33)
DEVICE = "cuda"  # a module global so a harness smoke can run the logic elsewhere


def _launch(tiles, warps, mc_tiles, mc_warps, label) -> dict:
    return {"gemv_tiles": int(tiles), "gemv_warps": int(warps), "mc_tiles": int(mc_tiles),
            "mc_warps": int(mc_warps), "shape_class": "", "family": "", "box_class": "", "source": label}


def _cfg(launch: dict) -> tuple:
    return tuple(int(launch[k]) for k in ("gemv_tiles", "gemv_warps", "mc_tiles", "mc_warps"))


def schedules(R: int, C: int, B: int, widths, grid: bool) -> list[tuple[str, dict, bool]]:
    """(label, rx_launch, gated): every box class's table row for this
    tensor and `spike` (gated; rows two boxes share run once, under both
    labels), then — when `grid` — the tiles x warps grid paired with the
    mc grid (recorded). Every schedule is a compile per tensor width (C is
    a constexpr), so the grid runs on a subset of tensors (run_device)."""
    out, gated = [], {}
    for box in rs.BOX_CLASSES:
        launch = rs.select(R, C, B, widths, override=f"table={box}").as_dict()
        gated.setdefault(_cfg(launch), (launch, []))[1].append(f"table={box}")
    launch = rs.select(R, C, B, widths, override="spike").as_dict()
    gated.setdefault(_cfg(launch), (launch, []))[1].append("spike")
    for box in rs.TWIN_TABLES:  # the bench's twin arm runs here: the compressed kernels must match it
        launch = rs.select_twin(R, C, B, widths, override=f"table={box}").as_dict()
        gated.setdefault(_cfg(launch), (launch, []))[1].append(f"twin={box}")
    out += [("+".join(labels), launch, True) for launch, labels in gated.values()]
    if not grid:
        return out
    seen = {(c[0], c[1]) for c in gated}
    nb = triton.cdiv(C, B)
    for t in GRID_TILES:
        eff = 0 if (t == 0 or t >= nb) else t
        for w in GRID_WARPS:
            if (eff, w) in seen:
                continue
            seen.add((eff, w))
            mt, mw = MC_GRID[len(seen) % len(MC_GRID)]
            out.append((f"t{'row' if eff == 0 else eff}/w{w} mc t{'row' if mt == 0 else mt}/w{mw}",
                        _launch(eff, w, mt, mw, "grid"), False))
    return out


def _bits_equal(a: torch.Tensor, b: torch.Tensor) -> tuple[bool, int, float]:
    """Bitwise, NaN positions compared as positions (payloads may differ
    through arithmetic): (equal, differing elements, max |a - b| finite)."""
    ia = a.contiguous().view(torch.int32 if a.dtype == torch.float32 else torch.int16)
    ib = b.contiguous().view(torch.int32 if b.dtype == torch.float32 else torch.int16)
    na, nb_ = torch.isnan(a), torch.isnan(b)
    diff = (ia != ib) & ~(na & nb_)
    n = int(diff.sum().item())
    if n == 0:
        return True, 0, 0.0
    fa, fb = a.float(), b.float()
    finite = torch.isfinite(fa) & torch.isfinite(fb)
    mad = float((fa - fb)[finite].abs().max().item()) if bool(finite.any()) else float("nan")
    return False, n, mad


class Mismatch(AssertionError):
    pass


def _compare_arm(rec: list, label: str, arm: str, fn, rt: dict, tw: dict, gated: bool) -> int:
    a, b = fn(rt), fn(tw)
    ok, n, mad = _bits_equal(a, b)
    if ok:
        return 1
    del a, b
    fusion = radix_ops.FUSION
    try:
        radix_ops.FUSION = False
        ok_off = _bits_equal(fn(rt), fn(tw))[0]
    finally:
        radix_ops.FUSION = fusion
    rec.append({"schedule": label, "arm": arm, "differing": n, "max_abs_diff": mad,
                "gated": gated, "equal_with_fp_fusion_off": ok_off})
    return 1


def check_tensor(receipt: list, name: str, W: torch.Tensor, pack: dict, x: torch.Tensor,
                 strict: bool, grid: bool, corrupt: bool = False) -> tuple[int, int]:
    """Every schedule x arm, then the modules at this box's row. Returns
    (cases, gated mismatches)."""
    R, C = int(pack["R"]), int(pack["C"])
    B = int(pack["block_size"])
    rt = to_device_radix(pack, DEVICE)
    tw = to_device_twin(W, pack["widths"], DEVICE)
    want = rs.select_twin(R, C, B, pack["widths"]).as_dict()
    if tw["rx_launch"] != want:
        raise Mismatch(f"{name}: the twin's rx_launch {tw['rx_launch']} != radix_schedule.select_twin's {want}")
    if corrupt:
        tw["weight"].view(torch.int16)[0, 0] ^= 0x4000  # an exponent bit: no flush-to-zero hides it
    bias = torch.linspace(-0.1, 0.1, R, device=DEVICE, dtype=torch.bfloat16)
    x1 = x[0]
    bad: list = []
    cases = 0
    for label, launch, gated in schedules(R, C, B, pack["widths"], grid):
        g = gated or strict
        rt2, tw2 = dict(rt, rx_launch=launch), dict(tw, rx_launch=launch)
        cases += _compare_arm(bad, label, "gemv_fused+bias",
                              lambda p: radix_ops.gemv_fused(p, x1, bias, torch.bfloat16), rt2, tw2, g)
        cases += _compare_arm(bad, label, "gemv_fused",
                              lambda p: radix_ops.gemv_fused(p, x1, None, torch.bfloat16), rt2, tw2, g)
        cases += _compare_arm(bad, label, "gemv_fp32",
                              lambda p: radix_ops.gemv(p, x1.float()), rt2, tw2, g)
        cases += _compare_arm(bad, label, "gemv_fp32 x bf16",
                              lambda p: radix_ops.gemv(p, x1), rt2, tw2, g)
        for M in range(2, MC_MAX + 1):
            cases += _compare_arm(bad, label, f"mc M={M}",
                                  lambda p, M=M: radix_ops.gemv_mc(p, x[:M]), rt2, tw2, g)
        for M in (2, MC_MAX):  # an fp32 activation: its 4-wide load sets the loop's layout
            cases += _compare_arm(bad, label, f"mc M={M} x fp32",
                                  lambda p, M=M: radix_ops.gemv_mc(p, x[:M].float()), rt2, tw2, g)
    # the modules at the compressed tensor's row and at the twin's own (the
    # bench's twin arm runs there), both modules at the same one each time
    rows = {"module": rt["rx_launch"]}
    if _cfg(tw["rx_launch"]) != _cfg(rt["rx_launch"]):
        rows["module@twin row"] = tw["rx_launch"]
    for label, launch in rows.items():
        comp = RadixCompressedLinear(dict(rt, rx_launch=launch), bias)
        twin = RadixTwinLinear(dict(tw, rx_launch=launch), bias)
        for M in MODULE_ROWS:
            xm = x[:M].reshape(1, M, C)
            route = comp._route(xm)
            if twin._route(xm) != route:
                raise Mismatch(f"{name}: M={M} routes {route} (compressed) vs {twin._route(xm)} (twin)")
            a, b = comp(xm), twin(xm)
            ok, n, mad = _bits_equal(a, b)
            cases += 1
            if not ok:
                bad.append({"schedule": label, "arm": f"forward M={M} ({route})", "differing": n,
                            "max_abs_diff": mad, "gated": True, "equal_with_fp_fusion_off": None})
    gated_bad = [m for m in bad if m["gated"]]
    receipt.append({"name": name, "shape": [R, C], "profile": pack.get("profile"),
                    "widths": list(pack["widths"]), "rx_launch": rt["rx_launch"],
                    "shape_class": rs.shape_class(R, C, B), "family": rs.family(pack["widths"]),
                    "cases": cases, "mismatches": bad})
    print(f"{'PASS' if not gated_bad else 'FAIL'}  {name:44s} {R}x{C} {pack.get('profile') or '':8s} "
          f"{cases} cases, {len(bad)} mismatched ({len(gated_bad)} gated)"
          + "".join(f"\n      {m['schedule']:28s} {m['arm']:16s} {m['differing']} differ, max |d| "
                    f"{m['max_abs_diff']:.3g}, fusion off equal: {m['equal_with_fp_fusion_off']}"
                    for m in bad[:8]), flush=True)
    del rt, tw, comp, twin
    torch.cuda.empty_cache()
    return cases, len(gated_bad)


def _finite_bits(rng, shape) -> np.ndarray:
    R, C = shape
    exps = rng.choice(np.arange(113, 122), size=(R, C),
                      p=np.array([.002, .003, .005, .01, .025, .055, .1, .3, .5]))
    bits = (exps.astype(np.uint16) << 7) | rng.integers(0, 128, (R, C), dtype=np.uint16)
    bits |= rng.integers(0, 2, (R, C), dtype=np.uint16) << 15
    return bits


def _bf16(bits: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(bits).view(np.int16).copy()).view(torch.bfloat16)


def _x(C: int, g) -> torch.Tensor:
    return torch.randn((MODULE_ROWS[-1], C), device=DEVICE, dtype=torch.bfloat16, generator=g)


def run_device(a, receipt) -> tuple[int, int]:
    rng = np.random.default_rng(924)
    g = torch.Generator(device=DEVICE).manual_seed(924)
    cases = bad = 0
    corrupt = a.selftest
    patterns = np.zeros((512, 513), np.uint16)
    patterns.flat[:65536] = np.arange(65536, dtype=np.uint16)
    # (name, bits, grid): the grid on the fixtures and the first tensor of
    # each model's widths (--grid all: everywhere)
    tensors = [] if a.real_only else [("fixture/patterns", patterns, True)]
    tensors += [] if a.real_only else [(f"fixture/ragged {R}x{C}", _finite_bits(rng, (R, C)), True)
                                       for R, C in RAGGED]
    if not a.fixtures_only and not a.real_only:
        tensors += [(f"{model} {R}x{C}", _finite_bits(rng, (R, C)), i == 0 or a.grid == "all")
                    for model, shapes in REAL_WIDTHS.items() for i, (R, C) in enumerate(shapes)]
    for name, bits, grid in tensors:
        W = _bf16(bits)
        for profile in a.profiles:
            pack = rp.pack_weight_radix(W, profile)
            if pack["codec"] != "radix":
                receipt["tensors"].append({"name": name, "profile": profile,
                                           "note": "radix would expand: raw fallback, F.linear on both arms"})
                continue
            c, b = check_tensor(receipt["tensors"], f"{name}", W, pack, _x(int(pack["C"]), g),
                                a.strict, grid, corrupt)
            corrupt = False
            cases, bad = cases + c, bad + b
    if not a.no_real:
        import parity_common as pc

        for i, arm in enumerate(a.real_arms):
            names = pc.layer_names(a.layer) + ((pc.LM_HEAD,) if i == 0 else ())
            packs = pc.load_pack_dicts(pc.PACKS[arm], names)
            for name in names:
                p = packs[name]
                W = _bf16(pc.source_bits(name))
                if p.get("codec") != "radix":
                    receipt["tensors"].append({"name": f"{arm} {name}", "note": "raw fallback in the pack"})
                    continue
                c, b = check_tensor(receipt["tensors"], f"{arm} {pc.short(name)}", W, p,
                                    _x(int(p["C"]), g), a.strict, a.grid == "all")
                cases, bad = cases + c, bad + b
            del packs
    return cases, bad


# ---------------------------------------------------------------- --compile --
# CPU only: triton.compile for explicit GPUTargets — nothing is launched and
# no device is opened (the environment hides every accelerator first).

TARGETS = {"gfx1151": ("hip", "gfx1151", 32), "gfx1102": ("hip", "gfx1102", 32),
           "sm80": ("cuda", 80, 32), "sm89": ("cuda", 89, 32), "sm90": ("cuda", 90, 32)}
COMPILE_WIDTHS = ((3, 8), (2, 3, 8), (2, 2, 4, 8))
STAGES = ("ttir", "ttgir", "llir", "amdgcn", "ptx")
X_DTYPES = ("bf16", "fp32")  # the activation: bf16 on the served paths, fp32 through radix_ops.gemv's callers


def _compile_one(job):
    """One kernel specialization -> {stage: text} (location info stripped),
    at the pointer specialization a launch gets on the device: every
    pointer 16-byte aligned (tt.divisibility 16) and, on a ROCm target,
    inside 2 GB (tt.pointer_range 32: buffer loads) — except the twin's
    weight under raw="big", a bf16 weight over 2 GB (the 27B's lm_head,
    2.5 GB), which the launcher leaves on global loads. Without the
    alignment the loads never vectorize and the layout the device runs
    is not the one compiled here."""
    kind, target, warps, C, widths, tiles, splits, bias, mc, raw, xdt = job
    import triton
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource

    from drinkme.codec import radix_kernel_gpu as K, radix_ops as O

    if kind == "gemv":
        fn = K._gemv
        sig = {"out": "*fp32", "x": f"*{xdt}", "bias": "*bf16", "data": "*u32", "offsets": "*u32",
               "palette": "*u32", "schedule": "*u16", "scale": "*u32"}
        const = dict(C=C, M=7, E=8, WIDTHS=widths, B=1024, DECODER=2, TILES=tiles, SPLITS=splits, BIAS=bias)
    else:
        fn = O._gemv_mc
        sig = {"out": "*fp32", "x": f"*{xdt}", "data": "*u32", "offsets": "*u32",
               "palette": "*u32", "schedule": "*u16", "scale": "*u32"}
        const = dict(C=C, M=7, E=8, WIDTHS=widths, B=1024, DECODER=2, TILES=tiles, SPLITS=splits, MC=mc)
    if raw is not None:
        const["RAW"] = bool(raw)
        if raw:
            sig["data"] = "*bf16"
    hip = TARGETS[target][0] == "hip"
    attrs = {(i,): [["tt.divisibility", 16]] + ([["tt.pointer_range", 32]]
                                                if hip and not (raw == "big" and name == "data") else [])
             for i, name in enumerate(sig)}
    for k in const:
        sig[k] = "constexpr"
    src = ASTSource(fn, sig, {(list(sig).index(k),): v for k, v in const.items()}, attrs)
    ck = triton.compile(src, target=GPUTarget(*TARGETS[target]),
                        options=dict(num_warps=warps, enable_fp_fusion=True))
    text = {s: _strip(ck.asm[s], s) for s in STAGES if s in ck.asm}
    # hashes, not text: two trees' worth of every stage runs to gigabytes
    return {"stages": {s: hashlib.sha256(t.encode()).hexdigest() for s, t in text.items()},
            "layouts": _mul_layouts(text["ttgir"]), "products": list(_products(text["llir"]))}


def _strip(text: str, stage: str) -> str:
    """Location metadata out (paths and line numbers move with any edit):
    MLIR loc(), LLVM !dbg, and the DWARF .debug_* sections of the ISA."""
    if stage in ("amdgcn", "ptx"):
        out, skip = [], False
        for line in text.splitlines():
            m = re.match(r"\s*\.section\s+(\S+)", line)
            if m:
                skip = m.group(1).startswith(".debug")
            elif stage == "amdgcn" and re.match(r"\s*\.(text|data|rodata|amdgpu_metadata)\b", line):
                skip = False
            if not skip:
                out.append(line)
        text = "\n".join(out)
    text = re.sub(r"\s*loc\(#loc\d*\)", "", text)
    text = re.sub(r"^#loc.*$", "", text, flags=re.M)
    text = re.sub(r'loc\("[^"]*"[^)]*\)', "", text)
    text = re.sub(r"^\s*\.(file|loc)\b.*$", "", text, flags=re.M)
    text = re.sub(r"^\s*;.*$", "", text, flags=re.M)
    text = re.sub(r",?\s*!dbg ![0-9]+", "", text)
    text = re.sub(r"^!.*$", "", text, flags=re.M)
    return "\n".join(line for line in text.splitlines() if line.strip())


def _jobs(targets, Cs, raw):
    for target in targets:
        if raw == "big" and TARGETS[target][0] != "hip":
            continue  # no buffer loads off ROCm: "big" is "true" there
        for warps in (1, 2, 4, 8):  # 8: the twin rows (radix_schedule.TWIN_TABLES)
            for C in Cs:
                nb = -(-C // 1024)
                for widths in COMPILE_WIDTHS:
                    for tiles, splits in ((nb, 1), (1, nb), (2, -(-nb // 2))):
                        for xdt in X_DTYPES:
                            for bias in (True, False):
                                yield ("gemv", target, warps, C, widths, tiles, splits, bias, None, raw, xdt)
                            for mc in (1, 2, 5, 8):
                                yield ("mc", target, warps, C, widths, tiles, splits, False, mc, raw, xdt)


def _key(job) -> str:
    kind, target, warps, C, widths, tiles, splits, bias, mc, _, xdt = job
    return (f"{target} w{warps} C{C} {kind} {'_'.join(map(str, widths))} t{tiles}s{splits} x={xdt}"
            + (f" bias={int(bias)}" if kind == "gemv" else f" MC={mc}"))


def _mul_layouts(ttgir: str) -> list[str]:
    """The register layout of every fp32 product (arith.mulf) in the TTGIR,
    in program order: the loop's accumulator layout, and so which elements
    each thread's partial sum of tl.sum(acc) holds (radix_kernel_gpu._raw_weight)."""
    defs = dict(re.findall(r"^(#\w+) = (#ttg\.\w+<\{[^\n]*\}>)", ttgir, re.M))
    return [defs.get(m, m) for m in re.findall(r"arith\.mulf [^\n]*: tensor<[^,>]+, (#\w+)>", ttgir)]


def _dump_main(a) -> int:
    """The worker half of --compile: compile every job of this source tree
    (PYTHONPATH) and write {key: {stage: text}} to --dump-out."""
    from concurrent.futures import ProcessPoolExecutor

    raw = {"none": None, "false": False, "true": True, "big": "big"}[a.dump_raw]
    jobs = list(_jobs(a.targets, a.widths_c, raw))
    with ProcessPoolExecutor(a.workers) as ex:
        results = list(ex.map(_compile_one, jobs, chunksize=4))
    with open(a.dump_out, "w") as f:
        json.dump({_key(j): r for j, r in zip(jobs, results)}, f)
    return 0


def _run_dump(src_root: str, raw: str, a, out: str) -> dict:
    # A fresh Triton cache per tree, removed as soon as its dump is written:
    # three of them left in /tmp (RAM-backed on a Strix Halo) fill 20 GB
    # of it.
    with tempfile.TemporaryDirectory(prefix="twin-compile-cache-") as cache:
        env = dict(os.environ, PYTHONPATH=src_root, HIP_VISIBLE_DEVICES="", CUDA_VISIBLE_DEVICES="",
                   ROCR_VISIBLE_DEVICES="", TRITON_CACHE_DIR=cache)
        cmd = [sys.executable, os.path.abspath(__file__), "--_dump", "--dump-raw", raw, "--dump-out", out,
               "--workers", str(a.workers), "--targets", *a.targets, "--widths-c", *map(str, a.widths_c)]
        subprocess.run(cmd, env=env, check=True)
    with open(out) as f:
        return json.load(f)


def _products(llir: str) -> tuple[int, int, int]:
    """(fmuls feeding an fadd directly, fmuls feeding a select, other) —
    contractible into an FMA under fp fusion or not."""
    defs = {}
    for line in llir.splitlines():
        m = re.match(r"\s*(%[\w.]+) = (\w+)", line)
        if m:
            defs[m.group(1)] = (m.group(2), line)
    users: dict = {}
    for _name, (op, line) in defs.items():
        for u in re.findall(r"%[\w.]+", line.split("=", 1)[1]):
            users.setdefault(u, []).append(op)
    direct = select = other = 0
    for name, (op, _) in defs.items():
        if op == "fmul":
            us = users.get(name, [])
            direct += us == ["fadd"]
            select += us == ["select"]
            other += us not in (["fadd"], ["select"])
    return direct, select, other


def _on_table(key: str) -> bool:
    """Is this compile case a (tiles, warps) some box's table serves (or
    spike), on that box's target family?"""
    target, w, C, kind, widths, ts = key.split()[:6]
    warps = int(w[1:])
    nb = -(-int(C[1:]) // 1024)
    tiles = int(ts[1:ts.index("s")])
    tiles = 0 if tiles >= nb else tiles
    fam = rs.family(tuple(int(x) for x in widths.split("_")))
    box = target if target.startswith("gfx") else "cuda"
    field = "gemv" if kind == "gemv" else "mc"
    rows = {(r[f"{field}_tiles"], r[f"{field}_warps"]) for r in rs.TABLES[box][fam].values()}
    rows.add((rs.SPIKE[f"{field}_tiles"], rs.SPIKE[f"{field}_warps"]))
    if field == "gemv" and box in rs.TWIN_TABLES:  # the bench's twin arm's own rows (gemv only)
        rows |= {(r["gemv_tiles"], r["gemv_warps"]) for r in rs.TWIN_TABLES[box].values()}
    return (tiles, warps) in rows


def run_compile(a) -> int:
    # the three dumps: removed on every exit, a failed compile included
    # (see _run_dump for why /tmp matters here)
    with tempfile.TemporaryDirectory(prefix="twin-compile-") as tmp:
        return _compile_in(a, tmp)


def _compile_in(a, tmp: str) -> int:
    repo = subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True,
                          check=True).stdout.strip()
    here = os.path.join(repo, "src")
    t0 = time.perf_counter()
    print(f"compiling on the CPU for {', '.join(a.targets)} at C in {a.widths_c}, x in {X_DTYPES}, at the "
          f"device's pointer specialization: this tree at RAW=False, this tree at RAW=True (and with a "
          f">2 GB twin weight on the ROCm targets)", flush=True)
    served = _run_dump(here, "false", a, os.path.join(tmp, "served.json"))
    twin = _run_dump(here, "true", a, os.path.join(tmp, "twin.json"))
    big = _run_dump(here, "big", a, os.path.join(tmp, "big.json"))
    # the accumulator layout: what tl.sum's per-thread partial sums hold. It
    # must be the served kernel's for the twin's sum to be the served sum.
    layout_differs = [(k, which) for which, other in (("twin", twin), ("twin >2GB", big)) for k in served
                      if k in other and other[k]["layouts"] != served[k]["layouts"]]
    n_big = sum(1 for k in served if k in big)
    print(f"RAW=True vs RAW=False accumulator layout (every fp32 product's): {len(layout_differs)} of "
          f"{len(served) + n_big} specializations differ ({len(served)} twin, {n_big} with a >2 GB twin weight)",
          flush=True)
    for k, which in layout_differs[:10]:
        other = twin if which == "twin" else big
        print(f"  LAYOUT {which} {k}: served {served[k]['layouts'][:1]} twin {other[k]['layouts'][:1]}", flush=True)
    on, off = [], []
    for k in served:
        ps, pt = tuple(served[k]["products"]), tuple(twin[k]["products"])
        kinds_s, kinds_t = tuple(v > 0 for v in ps), tuple(v > 0 for v in pt)
        ratio = (lambda p: None if p[1] == 0 else p[0] / p[1])
        if kinds_s != kinds_t or ratio(ps) != ratio(pt):
            (on if _on_table(k) else off).append((k, ps, pt))
    n_on = sum(1 for k in served if _on_table(k))
    print(f"RAW=True vs RAW=False product contraction (fmul -> fadd direct : via select : other): "
          f"{n_on - len(on)} of {n_on} table-row specializations the same, "
          f"{len(served) - n_on - len(off)} of {len(served) - n_on} off-table", flush=True)
    for k, ps, pt in (on + off)[:24]:
        print(f"  {'TABLE' if (k, ps, pt) in on else 'grid '} {k}: served {ps} twin {pt}", flush=True)
    receipt = {"mode": "compile", "targets": a.targets, "C": a.widths_c,
               "x_dtypes": list(X_DTYPES), "specializations": len(served), "big_twin_specializations": n_big,
               "layout_differs": [f"{which}: {k}" for k, which in layout_differs],
               "contraction_differs_on_table": [k for k, _, _ in on],
               "contraction_differs_off_table": [k for k, _, _ in off]}
    if a.json:
        with open(a.json, "w") as f:
            json.dump(receipt, f, indent=1)
    ok = not layout_differs
    print(f"RADIX_TWIN_COMPILE {'PASS' if ok else 'FAIL'}: the twin's accumulator layout is the served one "
          f"at {len(served) + n_big - len(layout_differs)}/{len(served) + n_big}; contraction differs at "
          f"{len(on)} table-row and {len(off)} off-table specializations (recorded; the device gate decides); "
          f"{time.perf_counter() - t0:.0f}s", flush=True)
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true", help="flip one twin weight bit: must FAIL")
    ap.add_argument("--json", default=None)
    ap.add_argument("--strict", action="store_true", help="gate the tiles x warps grid too")
    ap.add_argument("--no-real", action="store_true", help="skip the real Qwen3-8B tensors (no packs here)")
    ap.add_argument("--fixtures-only", action="store_true", help="fixtures only (a quick run)")
    ap.add_argument("--real-only", action="store_true", help="the real pack tensors only (--real-arms)")
    ap.add_argument("--grid", choices=("some", "all"), default="some",
                    help="the tiles x warps grid on the fixtures and one tensor per model (some) or every tensor")
    ap.add_argument("--profiles", nargs="*", default=list(PROFILES))
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--real-arms", nargs="*", default=["sip", "gulp"])
    ap.add_argument("--compile", action="store_true", help="CPU only: the compiled-kernel comparison")
    ap.add_argument("--targets", nargs="*", default=list(TARGETS))
    ap.add_argument("--widths-c", nargs="*", type=int, default=[4096, 3584, 18944])
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument("--_dump", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--dump-raw", default="none", help=argparse.SUPPRESS)
    ap.add_argument("--dump-out", default=None, help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a._dump:
        return _dump_main(a)
    if a.compile:
        return run_compile(a)
    if not torch.cuda.is_available():
        print("RADIX_TWIN_BITPIN FAIL: no GPU — run --compile for the CPU half")
        return 2
    print(f"device:  {torch.cuda.get_device_name(0)}  torch {torch.__version__}  triton {triton.__version__}  "
          f"box class {rs.box_class()}  fp fusion {radix_ops.FUSION}", flush=True)
    if a.selftest:
        print("SELFTEST: one twin weight bit is flipped on purpose; this run MUST print FAIL", flush=True)
    receipt = {"gate": "radix twin bit-pin", "device": torch.cuda.get_device_name(0),
               "torch": torch.__version__, "triton": triton.__version__, "box_class": rs.box_class(),
               "fp_fusion": radix_ops.FUSION, "strict": a.strict, "selftest": a.selftest,
               "profiles": a.profiles, "launch_table": rs.TABLES, "tensors": []}
    t0 = time.perf_counter()
    status = 1
    try:
        cases, bad = run_device(a, receipt)
        recorded = sum(len(t.get("mismatches", [])) for t in receipt["tensors"])
        receipt.update(cases=cases, gated_mismatches=bad, recorded_mismatches=recorded)
        if bad:
            raise Mismatch(f"{bad} gated (tensor, schedule, arm) cases are not bitwise equal")
        receipt["verdict"] = "PASS"
        print(f"RADIX_TWIN_BITPIN PASS: {cases} cases — the twin is bitwise the compressed module at every "
              f"launch-table row, spike and twin row, M = 1..{MC_MAX}, {MC_MAX + 1} and 33; {recorded} off-gate grid "
              f"mismatches recorded; {time.perf_counter() - t0:.0f}s", flush=True)
        status = 0
    except BaseException as e:  # noqa: BLE001 — the verdict line is the contract
        receipt["verdict"] = "FAIL"
        receipt["error"] = f"{type(e).__name__}: {e}"
        print(f"RADIX_TWIN_BITPIN FAIL: {type(e).__name__}: {e}", flush=True)
        traceback.print_exc()
    if a.json:
        os.makedirs(os.path.dirname(os.path.abspath(a.json)), exist_ok=True)
        with open(a.json, "w") as f:
            json.dump(receipt, f, indent=1, default=str)
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
