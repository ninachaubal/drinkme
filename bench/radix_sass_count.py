"""STATIC INSTRUCTION COUNTS PER WEIGHT: the radix GEMV's SASS for NVIDIA
targets, compiled on the CPU, its tile loop found and counted per decoder.

    # CPU only, no device opened (triton.compile for explicit GPUTargets):
    PYTHONPATH=src .venv/bin/python bench/radix_sass_count.py                # decoders 2,3; sm_89, sm_90; C=4096
    ... bench/radix_sass_count.py --decoders 2,3 --C 4096 12288 --diff 2,3 --json out.json
    ... bench/radix_sass_count.py --kernel mc --mc 2,8                       # radix_ops._gemv_mc
    ... bench/radix_sass_count.py --widths 2,2,4,8 --decoders 2              # gulp's tiers
    ... bench/radix_sass_count.py --widths 2,2,4,8 --decoders 2,4 --warps 1 --tiles 1   # gulp's CUDA table row
    ... bench/radix_sass_count.py --dump /tmp/sass                           # the SASS and PTX, per specialization

On CUDA the sip GEMV is believed to be instruction-issue-bound rather than
read-bound (radix_kernel_gpu._decode_sip_lean's docstring: an H100 reached
0.20 of its read bandwidth on it under the previous lean decoder), so what
predicts a decoder's speed there
is how many instructions each thread issues per weight. This counts that
statically, per (target, C, decoder):

  compile   radix_kernel_gpu._gemv (--kernel mc: radix_ops._gemv_mc at each
            --mc) at the pointer specialization a device launch gets, every
            pointer 16-byte aligned (bench/radix_twin_bitpin.py's
            _compile_one), with C, M=7, E=8, WIDTHS, B=1024, DECODER, TILES
            = the whole row, SPLITS=1 (--tiles t: TILES=t, SPLITS = the
            row's blocks / t, the split-K program the scheduled row runs at
            t=1), BIAS=False, RAW=False, fp fusion on, --warps warps (4)
            and a bf16 (or --x fp32) activation
  the loop  the cubin's SASS (nvdisasm) split at its backward branches; the
            tile loop is the outermost loop holding the activation's global
            load (the line-table entry of the kernel's `tl.load(x` lines).
            Its instructions / (B / (32 * warps)) = instructions per weight
            per thread: at 4 warps each thread holds 8 of a block's 1024
            lanes, and the loop body runs once per tile
  counts    the loop's opcode histogram (full opcode and base opcode),
            barriers, shuffles, global loads by width and predication,
            shared loads/stores, special-register reads, POPC, local
            (spill) traffic, 64-bit integer work (IADD3.X, IMAD.WIDE*,
            IMAD.X, LEA.HI.X, ISETP .EX, SHF .U64/.S64, uniform-datapath
            forms included; SHF .HI is counted apart, since it is also the
            plain 32-bit right shift), predicate and select
            ops, and each forward branch inside the loop with what its
            region holds (the terminal tier's escape read is one)
  registers cuobjdump -res-usage, and ptxas's own log of the compile (spill
            bytes, stack), with the register-limited resident warps per SM
  source    the loop's instructions per source line: SASS from the cubin's
            line table (the innermost line only), PTX from its .loc
            directives, whose inlined-at chain names the call site, so
            radix_gpu._read's lines say which field they read

A static count is not a timing: a forward branch's region issues only when
the branch is not taken, and latency hiding depends on occupancy. It is the
number to compare between decoders; --diff A,B prints the per-opcode delta
of the tile loop. --json PATH writes everything, per-line tables included.

Prints RADIX_SASS_COUNT OK when every requested specialization compiled and
its activation load was found in the line table (so the tile loop is the
one counted), else FAIL naming the ones that were not.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import linecache
import os
import re
import subprocess
import sys
import tempfile
import time
from collections import Counter

# no device is needed or opened: hide every accelerator before triton loads
for _v in ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"):
    os.environ[_v] = ""
sys.path.insert(0, os.path.realpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")))

BLOCK = 1024  # radix_pack.RADIX_BLOCK
MANT, EXP = 7, 8  # bf16 (radix_ops.MANT, EXP)
# CUDA C Programming Guide, "Technical Specifications per Compute Capability":
# (max resident warps, max resident blocks) per SM; every listed arch has 64K
# 32-bit registers per SM, allocated per warp in units of 256 (a thread's
# count rounded up to a multiple of 8)
SM_LIMITS = {80: (64, 32), 86: (48, 16), 87: (48, 16), 89: (48, 24), 90: (64, 32)}
REGS_PER_SM = 65536
PRED_SELECT = ("SEL", "FSEL", "ISETP", "FSETP", "PLOP3", "LOP3", "P2R", "R2P")

SASS_INSTR = re.compile(r"^\s*/\*([0-9a-f]+)\*/\s+(.+?)\s*;?\s*$")
SASS_LABEL = re.compile(r"^(\.L_x_\d+):")
SASS_LINE = re.compile(r'//## File "([^"]+)", line (\d+)')
SASS_PRED = re.compile(r"^(@!?U?P(?:T|\d+))\s+(.*)$")
PTX_FILE = re.compile(r'^\s*\.file\s+(\d+)\s+"([^"]+)"', re.M)
PTX_LOC = re.compile(r"^\.loc\s+(\d+)\s+(\d+)\s+\d+(.*)$")
PTX_LABEL = re.compile(r"^(\$L__\w+):")
PTX_BRA = re.compile(r"^(?:@!?%p\d+\s+)?bra(?:\.uni)?\s+(\$L__\w+)$")


def _list(values, cast=str) -> list:
    """argparse lists given either way: `--decoders 2,3` or `--C 4096 12288`."""
    return [cast(v) for token in values for v in str(token).split(",") if v.strip()]


# ------------------------------------------------------------------ compile --

def _compile_job(job: tuple) -> dict:
    """One specialization -> its PTX, SASS (with the line table), resource
    usage and ptxas log. Runs in a worker; imports triton here, after the
    parent set the cache directory and the ptxas-log knob."""
    kernel, target, C, widths, decoder, xdt, mc, warps, tiles = job
    import triton
    from triton import knobs
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource

    from drinkme.codec import radix_kernel_gpu as K, radix_ops as O

    nb = -(-C // BLOCK)
    tiles = min(tiles, nb) if tiles else nb
    splits = -(-nb // tiles)
    if kernel == "gemv":
        fn = K._gemv
        sig = {"out": "*fp32", "x": f"*{xdt}", "bias": "*bf16", "data": "*u32", "offsets": "*u32",
               "palette": "*u32", "schedule": "*u16", "scale": "*u32"}
        const = dict(C=C, M=MANT, E=EXP, WIDTHS=widths, B=BLOCK, DECODER=decoder, TILES=tiles, SPLITS=splits,
                     BIAS=False, RAW=False)
    else:
        fn = O._gemv_mc
        sig = {"out": "*fp32", "x": f"*{xdt}", "data": "*u32", "offsets": "*u32",
               "palette": "*u32", "schedule": "*u16", "scale": "*u32"}
        const = dict(C=C, M=MANT, E=EXP, WIDTHS=widths, B=BLOCK, DECODER=decoder, TILES=tiles, SPLITS=splits,
                     MC=mc, RAW=False)
    attrs = {(i,): [["tt.divisibility", 16]] for i in range(len(sig))}
    for k in const:
        sig[k] = "constexpr"
    src = ASTSource(fn, sig, {(list(sig).index(k),): v for k, v in const.items()}, attrs)
    res = {"job": list(job), "x_lines": _x_load_lines(fn)}
    log = io.StringIO()
    try:
        # TRITON_DUMP_PTXAS_LOG (set by the parent) prints ptxas -v's log
        with contextlib.redirect_stdout(log):
            ck = triton.compile(src, target=GPUTarget("cuda", target, 32),
                                options=dict(num_warps=warps, enable_fp_fusion=True))
    except Exception as e:  # noqa: BLE001 — reported per specialization
        res["error"] = _error_text(e)
        return res
    res["ptxas_log"] = log.getvalue()
    res["ptx"] = ck.asm["ptx"]
    with tempfile.TemporaryDirectory(prefix="sass-count-") as tmp:
        cubin = os.path.join(tmp, "k.cubin")
        with open(cubin, "wb") as f:
            f.write(ck.asm["cubin"])
        res["sass"] = subprocess.run([knobs.nvidia.nvdisasm.path, "-c", "-g", cubin], capture_output=True,
                                     text=True, check=True).stdout
        res["res_usage"] = subprocess.run([knobs.nvidia.cuobjdump.path, "-res-usage", cubin], capture_output=True,
                                          text=True, check=True).stdout
    res["shared"] = int(ck.metadata.shared)
    return res


def _x_load_lines(fn) -> list:
    """(file, line) of every `tl.load(x ...` in the kernel's own source: the
    activation load that marks the tile loop."""
    import inspect

    py = getattr(fn, "fn", fn)
    lines, start = inspect.getsourcelines(py)
    path = os.path.realpath(inspect.getsourcefile(py))
    return [[path, start + i] for i, text in enumerate(lines) if re.search(r"tl\.load\(\s*x\b", text)]


def _error_text(e: BaseException) -> str:
    """The innermost cause's message (a static_assert's text, not the
    call-site excerpt Triton wraps it in)."""
    while e.__cause__ is not None:
        e = e.__cause__
    lines = [s for s in str(e).strip().splitlines() if s.strip()]
    return f"{type(e).__name__}: {lines[-1].strip() if lines else ''}"


# -------------------------------------------------------------------- parse --

def parse_sass(text: str) -> list[dict]:
    """nvdisasm -c -g output -> instructions in address order: addr, pred,
    op (full opcode with modifiers), base, branch target address, and the
    line-table (file, line) in force."""
    instrs, labels, pending, loc = [], {}, [], None
    for line in text.splitlines():
        m = SASS_LINE.search(line)
        if m:
            loc = (os.path.realpath(m.group(1)), int(m.group(2)))
            continue
        m = SASS_LABEL.match(line)
        if m:
            pending.append(m.group(1))
            continue
        m = SASS_INSTR.match(line)
        if not m:
            continue
        addr = int(m.group(1), 16)
        for label in pending:
            labels[label] = addr
        pending = []
        body = m.group(2).strip()
        p = SASS_PRED.match(body)
        pred, body = (p.group(1), p.group(2)) if p else (None, body)
        op, _, operands = body.partition(" ")
        target = re.search(r"`\((\.L_x_\d+)\)", operands)
        instrs.append({"addr": addr, "pred": pred, "op": op, "base": op.split(".")[0],
                       "mods": op.split(".")[1:], "target": target.group(1) if target else None, "loc": loc})
    for i in instrs:
        i["target_addr"] = labels.get(i["target"]) if i["target"] else None
    return instrs


def sass_loops(instrs: list[dict]) -> list[tuple[int, int]]:
    """[target, branch] of every backward branch (the self-branch trap after
    the last EXIT excluded), outermost first."""
    loops = {(i["target_addr"], i["addr"]) for i in instrs
             if i["base"] in ("BRA", "JMP") and i["target_addr"] is not None and i["target_addr"] < i["addr"]}
    return sorted(loops, key=lambda lp: (lp[0] - lp[1], lp[0]))


def parse_ptx(text: str) -> tuple[list[dict], list[tuple[int, int]]]:
    """PTX -> (statements in order, each with its .loc frames innermost first;
    [first, last] statement index of every backward-branch loop, overlapping
    ones merged: LLVM's rotated loops branch back to one header from more than
    one place)."""
    files = {int(i): p for i, p in PTX_FILE.findall(text)}
    by_base = {os.path.basename(p): p for p in files.values()}
    stmts, labels, branches, frames = [], {}, [], None
    for raw in text.splitlines():
        s = raw.strip()
        m = PTX_LOC.match(s)
        if m:
            chain = re.findall(r"([\w.\-]+\.py):(\d+):\d+", m.group(3))
            frames = [(os.path.realpath(files.get(int(m.group(1)), m.group(1))), int(m.group(2)))]
            frames += [(os.path.realpath(by_base.get(b, b)), int(n)) for b, n in chain[1:]]
            continue
        s = s.split("//", 1)[0].strip()
        m = PTX_LABEL.match(s)
        if m:
            labels[m.group(1)] = len(stmts)
            continue
        if not s or s.startswith(".") or ";" not in s:
            continue
        for part in s.split(";"):
            part = part.strip().strip("{}").strip()
            if not part or part.startswith("."):
                continue
            b = PTX_BRA.match(part)
            if b:
                branches.append((len(stmts), b.group(1)))
            stmts.append({"text": part, "frames": frames})
    spans = sorted((labels[t], i) for i, t in branches if t in labels and labels[t] <= i)
    merged: list[list[int]] = []
    for lo, hi in spans:
        if merged and lo <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], hi)
        else:
            merged.append([lo, hi])
    return stmts, [tuple(m) for m in merged]


def _parse_usage(res_usage: str, log: str) -> dict:
    out: dict = {}
    m = re.search(r"REG:(\d+)\s+STACK:(\d+)\s+SHARED:(\d+)\s+LOCAL:(\d+)", res_usage)
    if m:
        out.update(regs=int(m.group(1)), stack=int(m.group(2)), shared_static=int(m.group(3)),
                   local=int(m.group(4)))
    m = re.search(r"(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads", log)
    if m:
        out.update(ptxas_stack=int(m.group(1)), spill_stores=int(m.group(2)), spill_loads=int(m.group(3)))
    m = re.search(r"Used (\d+) registers", log)
    if m:
        out["ptxas_regs"] = int(m.group(1))
    m = re.search(r"used (\d+) barriers", log)
    out["barriers"] = int(m.group(1)) if m else 0
    return out


def resident_warps(regs: int, warps: int, cap: int) -> tuple[int, int] | None:
    """(register-limited resident warps per SM, the arch's maximum)."""
    if cap not in SM_LIMITS or not regs:
        return None
    max_warps, max_blocks = SM_LIMITS[cap]
    per_warp = -(-regs // 8) * 8 * 32
    blocks = min(REGS_PER_SM // (per_warp * warps), max_blocks, max_warps // warps)
    return blocks * warps, max_warps


# ------------------------------------------------------------------ analyse --

def specifics(region: list[dict]) -> dict:
    """The named counts over a run of SASS instructions."""
    base = Counter(i["base"] for i in region)
    ldg = Counter(i["op"] for i in region if i["base"] == "LDG")
    wide = Counter(i["op"] for i in region
                   if (i["base"].lstrip("U") in ("IADD3", "IMAD", "LEA", "IADD") and
                       ("X" in i["mods"] or any(m.startswith("WIDE") for m in i["mods"])))
                   or (i["base"].lstrip("U") == "ISETP" and "EX" in i["mods"])
                   or (i["base"].lstrip("U") == "SHF" and any(m in ("U64", "S64") for m in i["mods"])))
    shf_hi = sum(1 for i in region if i["base"].lstrip("U") == "SHF" and "HI" in i["mods"]
                 and not any(m in ("U64", "S64") for m in i["mods"]))
    return {"BAR": base["BAR"], "SHFL": base["SHFL"], "SHFL_ops": dict(Counter(i["op"] for i in region
                                                                               if i["base"] == "SHFL")),
            "LDG": sum(ldg.values()), "LDG_ops": dict(ldg),
            "LDG_predicated": sum(1 for i in region if i["base"] == "LDG" and i["pred"]),
            "LDS": base["LDS"], "STS": base["STS"], "S2R": base["S2R"], "S2UR": base["S2UR"],
            "POPC": base["POPC"], "LDL": base["LDL"], "STL": base["STL"],
            "wide64": sum(wide.values()), "wide64_ops": dict(wide), "SHF_HI": shf_hi,
            "pred_select": {k: base[k] for k in PRED_SELECT}}


def _repo_frame(path: str) -> bool:
    import triton

    return not path.startswith(os.path.realpath(os.path.dirname(triton.__file__)))


def _via(frames: list) -> tuple | None:
    """The call site worth naming: the immediate caller of a repo line, or
    the first repo frame above a Triton-library line (tl.cumsum's body)."""
    if not frames:
        return None
    if _repo_frame(frames[0][0]):
        return frames[1] if len(frames) > 1 else None
    return next((f for f in frames[1:] if _repo_frame(f[0])), None)


def analyse(res: dict, warps: int, cap: int) -> dict:
    instrs = parse_sass(res["sass"])
    x_lines = {tuple(v) for v in res["x_lines"]}
    x_addrs = [i["addr"] for i in instrs if i["base"] == "LDG" and i["loc"] in x_lines]
    loops = sass_loops(instrs)
    per_thread = BLOCK // (32 * warps)
    out = {"kernel_instructions": len(instrs), **_parse_usage(res["res_usage"], res["ptxas_log"]),
           "shared": res["shared"], "weights_per_thread_per_tile": per_thread}
    out["resident_warps"] = resident_warps(out.get("regs", 0), warps, cap)

    def inside(addr, lp):
        return lp[0] <= addr <= lp[1]

    depth = {lp: sum(1 for o in loops if o != lp and inside(lp[0], o) and inside(lp[1], o)) for lp in loops}
    cands = [lp for lp in loops if any(inside(a, lp) for a in x_addrs)]
    if cands:
        tile, out["tile_loop_by"] = min(cands, key=lambda lp: depth[lp]), "x load"
    elif not x_addrs and loops:  # no line table to find the activation load by
        tile, out["tile_loop_by"] = loops[0], "largest loop (x load not found)"
    else:  # the activation load is in no loop: one tile, or the loop unrolled
        tile, out["tile_loop_by"] = None, "none"
    out["loops"] = [{"start": f"0x{a:04x}", "end": f"0x{b:04x}",
                     "instructions": sum(1 for i in instrs if a <= i["addr"] <= b), "depth": depth[(a, b)],
                     "has_x_load": any(a <= x <= b for x in x_addrs), "tile_loop": (a, b) == tile}
                    for a, b in sorted(loops)]
    if tile is None:
        # no loop around the activation load: the whole kernel stands in,
        # spread over its program's tiles (--tiles, else the whole row)
        region = instrs
        out["tile_loop"] = None
        out["loop_instructions"] = len(instrs)
        nb = -(-res["job"][2] // BLOCK)
        out["per_weight_per_thread"] = len(instrs) / (per_thread * min(res["job"][8] or nb, nb))
    else:
        region = [i for i in instrs if inside(i["addr"], tile)]
        out["tile_loop"] = {"start": f"0x{tile[0]:04x}", "end": f"0x{tile[1]:04x}"}
        out["loop_instructions"] = len(region)
        out["per_weight_per_thread"] = len(region) / per_thread
    hi_addr = region[-1]["addr"]
    out["forward_branches"] = []
    for i in region:
        t = i["target_addr"]
        if i["base"] == "BRA" and t is not None and i["addr"] < t:
            body = [j for j in region if i["addr"] < j["addr"] < t]
            sp = specifics(body)
            out["forward_branches"].append({
                "at": f"0x{i['addr']:04x}", "pred": i["pred"], "target": f"0x{t:04x}",
                "leaves_loop": t > hi_addr, "instructions": len(body),
                "BAR": sp["BAR"], "SHFL": sp["SHFL"], "LDG": sp["LDG"]})
    out["histogram"] = [[op, base, n] for (op, base), n in
                        Counter((i["op"], i["base"]) for i in region).most_common()]
    out["base_histogram"] = Counter(i["base"] for i in region).most_common()
    out["kernel_histogram"] = [[op, n] for op, n in Counter(i["op"] for i in instrs).most_common()]
    out["specific"] = specifics(region)

    # per source line: SASS by its innermost line-table entry, PTX by .loc
    rows: dict = {}

    def row(key):
        return rows.setdefault(key, {"sass": 0, "ptx": 0, "via": Counter(), "ops": Counter()})

    for i in region:
        r = row(i["loc"] or ("?", 0))
        r["sass"] += 1
        r["ops"][i["op"]] += 1
    stmts, ptx_loops = parse_ptx(res["ptx"])
    x_stmts = [k for k, s in enumerate(stmts) if "ld.global" in s["text"] and s["frames"]
               and s["frames"][0] in x_lines]
    ptx_tile = next((lp for lp in ptx_loops if any(lp[0] <= k <= lp[1] for k in x_stmts)), None)
    if ptx_tile is None and not x_stmts and ptx_loops:
        ptx_tile = max(ptx_loops, key=lambda lp: lp[1] - lp[0])
    lo, hi = ptx_tile if ptx_tile else (0, len(stmts) - 1)
    out["ptx_instructions"] = len(stmts)
    out["ptx_loop_instructions"] = hi - lo + 1 if ptx_tile else None
    for s in stmts[lo:hi + 1]:
        r = row(s["frames"][0] if s["frames"] else ("?", 0))
        r["ptx"] += 1
        via = _via(s["frames"] or [])
        if via:
            r["via"][via] += 1
    out["lines"] = [{"file": f, "line": ln, "sass": r["sass"], "ptx": r["ptx"],
                     "text": linecache.getline(f, ln).strip() if ln else "",
                     "via": [[vf, vl, n] for (vf, vl), n in r["via"].most_common()],
                     "sass_ops": r["ops"].most_common()}
                    for (f, ln), r in sorted(rows.items(), key=lambda kv: (-kv[1]["sass"], -kv[1]["ptx"]))]
    out["x_load_addresses"] = [f"0x{a:04x}" for a in x_addrs]
    return out


# ------------------------------------------------------------------- report --

def _label(job) -> str:
    kernel, target, C, widths, decoder, xdt, mc, warps, tiles = job
    k = "_gemv" if kernel == "gemv" else f"_gemv_mc MC={mc}"
    t = f", {tiles} block(s) per program" if tiles else ""
    return f"sm{target} C={C} {k} DECODER={decoder} widths {tuple(widths)} x {xdt} {warps} warps{t}"


def _group(job) -> tuple:
    """Everything but the decoder: the specializations --diff pairs up."""
    kernel, target, C, widths, _decoder, xdt, mc, warps, tiles = job
    return (kernel, target, C, tuple(widths), xdt, mc, warps, tiles)


def _cols(items: list[str], width: int, per_row: int, indent: str) -> str:
    return "\n".join(indent + "".join(s.ljust(width) for s in items[k:k + per_row]).rstrip()
                     for k in range(0, len(items), per_row))


def _short(path: str) -> str:
    return os.path.basename(path)


def print_job(job, a: dict, top: int) -> None:
    kernel, target, C, widths, decoder, xdt, mc, warps, tiles = job
    sp = a["specific"]
    rw = a["resident_warps"]
    print(f"\n== {_label(job)}")
    print(f"   kernel {a['kernel_instructions']} SASS instructions; {a.get('regs', '?')} registers "
          f"(ptxas: {a.get('spill_stores', '?')} B spill stores, {a.get('spill_loads', '?')} B spill loads, "
          f"{a.get('ptxas_stack', '?')} B stack), {a['barriers']} barrier(s), {a['shared']} B shared"
          + (f"; resident warps/SM (register-limited) {rw[0]} of {rw[1]}" if rw else ""))
    if a["tile_loop"]:
        print(f"   tile loop {a['tile_loop']['start']}..{a['tile_loop']['end']}: {a['loop_instructions']} SASS "
              f"= {a['per_weight_per_thread']:.2f} per weight per thread ({a['weights_per_thread_per_tile']} "
              f"weights per thread per tile); PTX loop {a['ptx_loop_instructions']}; found by {a['tile_loop_by']}")
    else:
        print(f"   no loop around the activation load (one tile, or the tile loop unrolled): kernel "
              f"{a['loop_instructions']} SASS over {-(-C // BLOCK)} tile(s) = {a['per_weight_per_thread']:.2f} "
              f"per weight per thread")
    others = [lp for lp in a["loops"] if not lp["tile_loop"]]
    if others:
        print("   other loops: " + ", ".join(f"{lp['start']}..{lp['end']} ({lp['instructions']}, depth "
                                            f"{lp['depth']})" for lp in others))
    for fb in a["forward_branches"]:
        print(f"   {fb['pred'] or ''} BRA at {fb['at']} -> {fb['target']}: "
              + ("leaves the loop" if fb["leaves_loop"] else
                 f"skips {fb['instructions']} SASS (BAR {fb['BAR']}, SHFL {fb['SHFL']}, LDG {fb['LDG']}), "
                 f"issued only when not taken"))
    ldg = ", ".join(f"{k} {v}" for k, v in sorted(sp["LDG_ops"].items(), key=lambda kv: -kv[1]))
    shfl = ", ".join(f"{k} {v}" for k, v in sp["SHFL_ops"].items())
    print(f"   BAR {sp['BAR']}  SHFL {sp['SHFL']}" + (f" ({shfl})" if shfl else "")
          + f"  LDG {sp['LDG']} ({ldg}; {sp['LDG_predicated']} predicated)  LDS {sp['LDS']}  STS {sp['STS']}  "
          f"S2R {sp['S2R']}  S2UR {sp['S2UR']}  POPC {sp['POPC']}  LDL/STL {sp['LDL']}/{sp['STL']}")
    wide = ", ".join(f"{k} {v}" for k, v in sorted(sp["wide64_ops"].items(), key=lambda kv: -kv[1]))
    ps = "  ".join(f"{k} {v}" for k, v in sp["pred_select"].items() if v)
    print(f"   64-bit {sp['wide64']}" + (f" ({wide})" if wide else "") + f"  SHF .HI {sp['SHF_HI']}"
          f"   pred/select: {ps or 'none'}")
    print("   loop opcodes by base: " + ", ".join(f"{b} {n}" for b, n in a["base_histogram"]))
    print("   loop opcodes (full):")
    print(_cols([f"{n:4d} {op}" for op, _b, n in a["histogram"]], 30, 4, "     "))
    print(f"   loop instructions per source line (SASS: innermost line-table entry, its top opcodes; "
          f"PTX: .loc, and the call site where it varies), top {top}:")
    print("      sass   ptx  line                     text")
    for r in a["lines"][:top]:
        where = f"{_short(r['file'])}:{r['line']}"
        vias = r["via"]
        via = ""
        if len(vias) > 1 or (vias and not _repo_frame(r["file"])):
            via = "; PTX via " + ", ".join(f"{_short(f)}:{ln} x{n}" for f, ln, n in vias[:4])
        print(f"     {r['sass']:5d} {r['ptx']:5d}  {where:24s} {r['text'][:80]}")
        ops = ", ".join(f"{op} {n}" for op, n in r["sass_ops"][:5])
        if ops or via:
            print(f"                 {' ' * 24} {ops}{via}")


def print_summary(results: list) -> None:
    print("\nsummary (tile loop, per specialization):")
    hdr = (f"  {'target':6s} {'C':>6s} {'kernel':8s} {'dec':>3s} {'x':4s} {'SASS':>5s} {'loop':>5s} "
           f"{'/w/thr':>7s} {'PTXloop':>7s} {'BAR':>3s} {'SHFL':>4s} {'LDG':>4s} {'LDS/STS':>7s} {'64b':>4s} "
           f"{'regs':>4s} {'spill':>5s} {'warps/SM':>8s}")
    print(hdr)
    unrolled = False
    for job, a in results:
        kernel, target, C, widths, decoder, xdt, mc, warps, tiles = job
        k = "gemv" if kernel == "gemv" else f"mc{mc}"
        if "error" in a:
            print(f"  sm{target:<4d} {C:6d} {k:8s} {decoder:3d} {xdt:4s} did not compile: {a['error']}")
            continue
        sp = a["specific"]
        rw = a["resident_warps"]
        loop = f"{a['loop_instructions']:5d}" if a["tile_loop"] else "    -"
        unrolled = unrolled or not a["tile_loop"]
        print(f"  sm{target:<4d} {C:6d} {k:8s} {decoder:3d} {xdt:4s} {a['kernel_instructions']:5d} "
              f"{loop} {a['per_weight_per_thread']:7.2f} {a['ptx_loop_instructions'] or '-':>7} "
              f"{sp['BAR']:3d} {sp['SHFL']:4d} {sp['LDG']:4d} {sp['LDS']:3d}/{sp['STS']:<3d} {sp['wide64']:4d} "
              f"{a.get('regs', 0):4d} {a.get('spill_stores', 0) + a.get('spill_loads', 0):5d} "
              f"{(f'{rw[0]}/{rw[1]}' if rw else '-'):>8s}")
    if unrolled:
        print("  loop -: no loop around the activation load (one tile, or the loop unrolled); /w/thr and the "
              "counts are the whole kernel's, spread over its tiles")


def diff(results: list, da: int, db: int) -> list[dict]:
    by = {(_group(job), job[4]): a for job, a in results if "error" not in a}
    out = []
    for (grp, dec), a in by.items():
        if dec != da or (grp, db) not in by:
            continue
        b = by[(grp, db)]
        ca = Counter({op: n for op, _base, n in a["histogram"]})
        cb = Counter({op: n for op, _base, n in b["histogram"]})
        rows = sorted(((op, ca[op], cb[op], cb[op] - ca[op]) for op in set(ca) | set(cb) if ca[op] != cb[op]),
                      key=lambda r: (-abs(r[3]), r[0]))
        kernel, target, C, widths, xdt, mc, warps, tiles = grp
        out.append({"group": {"kernel": kernel, "target": f"sm{target}", "C": C, "widths": list(widths), "x": xdt,
                              "mc": mc, "warps": warps, "tiles": tiles or "row"}, "a": da, "b": db,
                    "loop": [a["loop_instructions"], b["loop_instructions"]],
                    "per_weight_per_thread": [a["per_weight_per_thread"], b["per_weight_per_thread"]],
                    "regs": [a.get("regs"), b.get("regs")], "opcodes": [list(r) for r in rows]})
    return out


def print_diff(d: dict) -> None:
    g = d["group"]
    k = "_gemv" if g["kernel"] == "gemv" else f"_gemv_mc MC={g['mc']}"
    la, lb = d["loop"]
    pa, pb = d["per_weight_per_thread"]
    print(f"\n== diff DECODER {d['a']} -> {d['b']}: {g['target']} C={g['C']} {k} widths {tuple(g['widths'])} "
          f"x {g['x']}: loop {la} -> {lb} SASS ({lb - la:+d}); per weight per thread {pa:.2f} -> {pb:.2f} "
          f"({pb - pa:+.2f}); registers {d['regs'][0]} -> {d['regs'][1]}")
    print(f"     {'opcode':28s} {d['a']:>5d} {d['b']:>5d} {'delta':>6s}")
    for op, na, nb, dn in d["opcodes"]:
        print(f"     {op:28s} {na:5d} {nb:5d} {dn:+6d}")


# --------------------------------------------------------------------- main --

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--decoders", nargs="+", default=["2,3"], help="DECODER indices (radix_kernel_gpu._decode_weight)")
    ap.add_argument("--targets", nargs="+", default=["sm89", "sm90"], help="smNN: GPUTarget('cuda', NN, 32)")
    ap.add_argument("--C", nargs="+", default=["4096"], help="row widths (constexpr C)")
    ap.add_argument("--widths", nargs="+", default=["3,8"], help="tier widths, one comma list each: 3,8 (sip), "
                                                                  "2,2,4,8 (gulp)")
    ap.add_argument("--x", nargs="+", default=["bf16"], help="activation dtype(s): bf16, fp32")
    ap.add_argument("--kernel", nargs="+", default=["gemv"], help="gemv (_gemv) and/or mc (_gemv_mc)")
    ap.add_argument("--mc", nargs="+", default=["2,8"], help="MC for --kernel mc")
    ap.add_argument("--warps", type=int, default=4)
    ap.add_argument("--tiles", type=int, default=0, help="blocks per program (TILES, SPLITS = the row's blocks / "
                                                         "TILES): 0 = the whole row; 1 = the scheduled row's")
    ap.add_argument("--top", type=int, default=14, help="source lines printed per specialization")
    ap.add_argument("--diff", default=None, help="A,B: the tile loop's per-opcode delta from decoder A to B")
    ap.add_argument("--json", default=None)
    ap.add_argument("--dump", default=None, help="directory: write each specialization's SASS and PTX there")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument("--quiet", action="store_true", help="the summary table (and --diff) only")
    args = ap.parse_args()

    decoders = _list(args.decoders, int)
    targets = [int(t.lower().removeprefix("sm").removeprefix("_")) for t in _list(args.targets)]
    Cs = _list(args.C, int)
    widths_list = [tuple(int(w) for w in s.split(",")) for s in args.widths]
    xs = _list(args.x)
    kernels = _list(args.kernel)
    mcs = _list(args.mc, int)
    for x in xs:
        if x not in ("bf16", "fp32"):
            ap.error(f"--x {x}: bf16 or fp32")
    for k in kernels:
        if k not in ("gemv", "mc"):
            ap.error(f"--kernel {k}: gemv or mc")
    pair = _list([args.diff], int) if args.diff else None
    if pair is not None and len(pair) != 2:
        ap.error("--diff A,B: two decoder indices")
    jobs = [(k, t, C, w, d, x, mc if k == "mc" else None, args.warps, args.tiles)
            for t in targets for C in Cs for w in widths_list for x in xs
            for k in kernels for mc in (mcs if k == "mc" else [None]) for d in decoders]
    jobs = list(dict.fromkeys(jobs))

    t0 = time.perf_counter()
    # a fresh Triton cache, so every specialization really runs ptxas (whose
    # log is the spill count) and nothing is left behind
    with tempfile.TemporaryDirectory(prefix="sass-count-cache-") as cache:
        os.environ["TRITON_CACHE_DIR"] = cache
        os.environ["TRITON_DUMP_PTXAS_LOG"] = "1"
        if len(jobs) == 1 or args.workers <= 1:
            raw = [_compile_job(j) for j in jobs]
        else:
            from concurrent.futures import ProcessPoolExecutor

            with ProcessPoolExecutor(min(args.workers, len(jobs))) as ex:
                raw = list(ex.map(_compile_job, jobs))
    import triton
    from triton import knobs

    ptxas = subprocess.run([knobs.nvidia.ptxas.path, "--version"], capture_output=True, text=True).stdout
    ptxas = (re.findall(r"release [\d.]+, V[\d.]+", ptxas) or ["?"])[0]
    print(f"radix_sass_count: {len(jobs)} specialization(s) compiled on the CPU in "
          f"{time.perf_counter() - t0:.1f}s (triton {triton.__version__}, ptxas {ptxas}); B={BLOCK}, "
          f"{'TILES = the whole row, SPLITS=1' if not args.tiles else f'TILES={args.tiles}, SPLITS = the row blocks / TILES'}, BIAS=False, RAW=False, fp fusion on, every pointer 16-byte aligned", flush=True)

    results, failed = [], []
    for job, res in zip(jobs, raw):
        if "error" in res:
            results.append((job, {"error": res["error"]}))
            failed.append(f"{_label(job)}: {res['error']}")
            if not args.quiet:
                print(f"\n== {_label(job)}\n   did not compile: {res['error']}")
            continue
        a = analyse(res, args.warps, job[1])
        results.append((job, a))
        if not a["x_load_addresses"]:
            failed.append(f"{_label(job)}: the activation load is not in the line table (tile loop unidentified)")
        if args.dump:
            os.makedirs(args.dump, exist_ok=True)
            stem = os.path.join(args.dump, re.sub(r"[^\w.=-]+", "_", _label(job)))
            for ext in ("sass", "ptx"):
                with open(f"{stem}.{ext}", "w") as f:
                    f.write(res[ext])
        if not args.quiet:
            print_job(job, a, args.top)
    print_summary(results)
    diffs = diff(results, *pair) if pair else []
    for d in diffs:
        print_diff(d)
    if pair and not diffs:
        print(f"\n--diff {pair[0]},{pair[1]}: no specialization compiled under both decoders")
    if args.json:
        receipt = {"tool": "radix_sass_count", "triton": triton.__version__, "ptxas": ptxas, "block": BLOCK,
                   "warps": args.warps, "tiles": args.tiles or "row",
                   "specializations": [{"kernel": j[0], "target": f"sm{j[1]}", "C": j[2], "widths": list(j[3]),
                                        "decoder": j[4], "x": j[5], "mc": j[6], "warps": j[7],
                                        "tiles": j[8] or "row", **a}
                                       for j, a in results],
                   "diffs": diffs, "failed": failed}
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "w") as f:
            json.dump(receipt, f, indent=1, default=str)
        print(f"\nwrote {args.json}")
    status = "OK" if not failed else "FAIL"
    print(f"\nRADIX_SASS_COUNT {status}: {len(jobs) - len(failed)} of {len(jobs)} specializations counted "
          f"(targets {', '.join(f'sm{t}' for t in targets)}; C {', '.join(map(str, Cs))}; decoders "
          f"{', '.join(map(str, decoders))}); {time.perf_counter() - t0:.1f}s", flush=True)
    for f in failed:
        print(f"  not counted: {f}")
    return 0 if not failed else 1


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(rc)
