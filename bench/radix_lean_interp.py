"""THE RADIX DECODERS UNDER THE TRITON INTERPRETER: every requested DECODER
index of radix_kernel_gpu (_dense, _gemv, radix_ops._gemv_mc) run on the
CPU (TRITON_INTERPRET=1),
bit for bit against the source and against the scheduled decoder
(DECODER 2), on tensors built to reach every field of the stream.

    # CPU only, no device (the interpreter runs each program in numpy):
    PYTHONPATH=src .venv/bin/python bench/radix_lean_interp.py                   # decoders 2,3, every case
    ... bench/radix_lean_interp.py --decoders 2,3 --cases gauss,ragged --json out.json
    ... bench/radix_lean_interp.py --selftest                                    # must print VERDICT: FAIL

Each case is encoded by the repo's own encoder (radix_pack.pack_array_radix,
--profile sip), put in CPU runtime buffers by swap.to_device_radix (the
zero pad after the payload that the decoders' reach relies on, the padded
palette words, the gulp schedule), and the kernels are launched directly
with an explicit DECODER, not through radix_ops._decoder (which picks by
backend and row shape):

  dense  radix_kernel_gpu._dense, DEQUANT off, one program per block
         (radix_ops.decode_bits' launch): == the source bits (uint16), and
         == decoder 2's output
  gemv   radix_kernel_gpu._gemv, whole-row tiles, SPLITS=1, no bias, fp32
         out (radix_ops.gemv_fused's launch at gemv_tiles = 0), once with a
         bf16 x and once with an fp32 x: == decoder 2's output bits
  mc     radix_ops._gemv_mc at MC = 2 and 8, bf16 x, whole-row tiles,
         SPLITS=1 (radix_ops.gemv_mc's launch): == decoder 2's

The GEMV comparisons are NaN-aware: the fp32 outputs' int32 views, with any
NaN equal to any NaN (the payload of a NaN sum is the arithmetic's, not the
decoder's), and the mismatches counted.

Cases:
  patterns  each of the 65,536 bf16 bit patterns exactly 4 times, shuffled
            (default_rng(0)), 64 x 4096: every sign, exponent and
            mantissa, so every palette entry and the escape path. Its
            exponents are uniform, so nearly every weight escapes the
            seven-entry tier 0 and radix would expand the tensor
            (pack_array_radix returns None, the raw fallback); the case then
            appends N(0, 0.02) rows, 64 at a time, until it compresses
  gauss     64 x 4096 N(0, 0.02) rounded to bf16 (default_rng(1)): a trained
            weight's exponent spread, most exponents in tier 0
  ragged    3 x 1025, 4 x 2048 and 2 x 3079 (one default_rng(2) stream, in
            that order): the first R // 2 rows distinct random bit
            patterns, the rest N(0, 0.02); two of the three end in a short
            block (n < B), whose few weights often all hit tier 0, so the
            decoders' no-escape branch (no terminal read) runs too; each
            case prints how many of its blocks escape nothing

Under the interpreter each tl operation is a numpy operation: no fp
contraction, no vectorized load, no register layout. So this checks what a
decoder computes, not how a GPU compiler schedules it, and does not replace
the device gates (bench/radix_gemv_bitpin.py, bench/radix_mc_bitpin.py,
bench/radix_lean_gate.py): a GEMV equal here can still round differently on
a device that contracts a different set of products into FMAs
(radix_ops._decoder's docstring: the lean decoder's ragged rows on an L4).

--selftest XORs 1 into one element of the first non-reference decoder's
dense output and into one finite element of its bf16-x GEMV output before
the comparisons; the run must end VERDICT: FAIL. The last line is
`VERDICT: PASS <n> cases` or `VERDICT: FAIL <k> of <n> cases: <names>`.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback

# the interpreter must be on before triton (and the kernels' @triton.jit) load
os.environ["TRITON_INTERPRET"] = "1"
for _v in ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"):
    os.environ[_v] = ""
sys.path.insert(0, os.path.realpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import triton  # noqa: E402

from drinkme.codec import radix_kernel_gpu as K  # noqa: E402
from drinkme.codec import radix_native  # noqa: E402
from drinkme.codec import radix_ops as O  # noqa: E402
from drinkme.codec import radix_pack as rp  # noqa: E402
from drinkme.codec.swap import to_device_radix  # noqa: E402

# the patterns case's Inf and NaN weights overflow numpy's float ops inside
# the interpreter on purpose; the results are compared bitwise, not warned about
np.seterr(all="ignore")

REFERENCE = O.DECODER_SCHEDULED  # decoder 2: every other decoder's outputs are compared to its
GAUSS_STD = 0.02
RAGGED = ((3, 1025), (4, 2048), (2, 3079))
CASES = ("patterns", "gauss") + tuple(f"ragged{R}x{C}" for R, C in RAGGED)
MCS = (2, 8)
WARPS = 4  # num_warps: the launch's shape only; the interpreter runs a program as one numpy computation
X_SEED = 3  # the activations: a CPU torch.Generator, re-seeded for each case


def _list(values, cast=str) -> list:
    """argparse lists given either way: `--decoders 2,3` or `--decoders 2 3`."""
    return [cast(v) for token in values for v in str(token).split(",") if v.strip()]


# -------------------------------------------------------------------- cases --

def bf16_bits(values: np.ndarray) -> np.ndarray:
    """float -> bf16 bit patterns (uint16), rounded to nearest even by torch's cast."""
    t = torch.from_numpy(np.ascontiguousarray(values, dtype=np.float32)).to(torch.bfloat16)
    return t.view(torch.int16).numpy().view(np.uint16)


def gauss(rng: np.random.Generator, shape: tuple) -> np.ndarray:
    return bf16_bits(rng.normal(0.0, GAUSS_STD, shape))


def candidates() -> dict[str, list[tuple[np.ndarray, str]]]:
    """Every case's tensors in order of preference: as specified first, then
    the fallbacks tried when radix would expand it. Built whole (a few MB) so
    that --cases does not move any case's random draws."""
    out: dict = {}
    rng = np.random.default_rng(0)
    bits = np.tile(np.arange(1 << 16, dtype=np.uint16), 4)
    rng.shuffle(bits)
    pat = bits.reshape(64, 4096)
    extra = gauss(rng, (256, 4096))
    out["patterns"] = [(pat, "64x4096: each bf16 bit pattern 4 times")] + [
        (np.concatenate([pat, extra[:k]]), f"the 64x4096 pattern rows + {k} N(0, {GAUSS_STD}) rows")
        for k in (64, 128, 192, 256)]
    out["gauss"] = [(gauss(np.random.default_rng(1), (64, 4096)), f"N(0, {GAUSS_STD}) rounded to bf16")]
    rng = np.random.default_rng(2)
    for R, C in RAGGED:
        g = gauss(rng, (R, C))
        n = (R // 2) * C
        draw = rng.permutation(1 << 16)[:n].astype(np.uint16)
        cands, k = [], n
        while k >= 1:
            U = g.copy()
            U.reshape(-1)[:k] = draw[:k]
            rows = k // C
            what = (f"the first {k} weights" if k % C else "the first row" if rows == 1 else
                    f"the first {rows} rows")
            cands.append((U, f"{what} distinct random bit patterns, the rest N(0, {GAUSS_STD})"))
            k //= 2
        out[f"ragged{R}x{C}"] = cands
    return out


def pick(name: str, cands: list, profile: str, encoder: str) -> tuple[np.ndarray, dict, str, list[str]]:
    """The first candidate radix compresses: (bits, pack, description, the
    rejected ones' descriptions)."""
    rejected = []
    for U, what in cands:
        pack = rp.pack_array_radix(U, profile, encoder)
        if pack is not None:
            return U, pack, what, rejected
        rejected.append(what)
    raise RuntimeError(f"{name}: radix expands every candidate tensor ({rejected})")


def coverage(U: np.ndarray, pack: dict) -> tuple[float, int, int]:
    """(share of weights whose exponent is a tier-0 palette entry, blocks
    whose every exponent is: they escape nothing and read no field past
    tier 0, blocks)."""
    R, C, B = int(pack["R"]), int(pack["C"]), int(pack["block_size"])
    w0 = int(pack["widths"][0])
    tier0 = np.asarray(pack["rx_palette"], dtype=np.uint8)[: (1 << w0) - 1]
    hit = np.isin(((U >> 7) & 0xFF).astype(np.uint8), tier0)
    nb = -(-C // B)
    padded = np.ones((R, nb * B), dtype=bool)
    padded[:, :C] = hit
    no_terminal = int(padded.reshape(R * nb, B).all(axis=1).sum())
    return float(hit.mean()), no_terminal, R * nb


# ------------------------------------------------------------------ launches --

def kernel_args(p: dict, decoder: int) -> tuple:
    """radix_ops._args' tail with DECODER given: (data, offsets, palette,
    schedule, scale, C, M, E, WIDTHS, B, DECODER); `scale` is a dummy (bf16's
    E == 8 never reads it) and rx_data stands in, as there."""
    return (p["rx_data"], p["rx_offsets"], p["rx_palette"], p["rx_schedule"], p["rx_data"], int(p["C"]),
            O.MANT, O.EXP, tuple(int(w) for w in p["widths"]), int(p["block_size"]), decoder)


def _error_text(e: BaseException) -> str:
    """The innermost cause (the interpreter wraps a kernel's own exception,
    a static_assert's text included, in InterpreterErrors)."""
    while e.__cause__ is not None:
        e = e.__cause__
    lines = str(e).strip().splitlines()
    return f"{type(e).__name__}: {lines[0] if lines else ''}"


def run_decoder(p: dict, decoder: int, x1: torch.Tensor, x1f: torch.Tensor, xm: torch.Tensor) -> dict:
    """Every arm under one decoder -> {arm: tensor or the exception's text}."""
    R, C, B = int(p["R"]), int(p["C"]), int(p["block_size"])
    nb = triton.cdiv(C, B)
    args = kernel_args(p, decoder)
    out: dict = {}

    def arm(name, fn):
        try:
            out[name] = fn()
        except Exception as e:  # noqa: BLE001 — recorded as that check's failure
            out[name] = _error_text(e)
            out[name + " traceback"] = traceback.format_exc()

    def dense():
        bits = torch.empty((R, C), dtype=torch.uint16)
        K._dense[(R * nb,)](bits, *args, False, num_warps=O.NUM_WARPS)
        return bits

    def gemv(x):
        y = torch.empty((1, R), dtype=torch.float32)
        K._gemv[(R, 1)](y, x.reshape(1, -1).contiguous(), args[0], *args, nb, 1, False, RAW=False,
                        num_warps=WARPS, enable_fp_fusion=O.FUSION)
        return y.reshape(-1)

    def mc(M):
        y = torch.empty((M, R), dtype=torch.float32)
        O._gemv_mc[(R,)](y, xm[:M].contiguous(), *args, nb, 1, M, RAW=False, num_warps=WARPS,
                         enable_fp_fusion=O.FUSION)
        return y

    arm("dense", dense)
    arm("gemv x bf16", lambda: gemv(x1))
    arm("gemv x fp32", lambda: gemv(x1f))
    for M in MCS:
        arm(f"mc M={M} x bf16", lambda M=M: mc(M))
    return out


# --------------------------------------------------------------- comparison --

def bits_differ(a: torch.Tensor, b: torch.Tensor) -> int:
    return int((a.contiguous().view(torch.int16) != b.contiguous().view(torch.int16)).sum())


def f32_differ(a: torch.Tensor, b: torch.Tensor) -> tuple[int, int]:
    """(elements whose int32 bits differ, not counting NaN against NaN;
    elements NaN in both)."""
    nan = torch.isnan(a) & torch.isnan(b)
    diff = (a.contiguous().view(torch.int32) != b.contiguous().view(torch.int32)) & ~nan
    return int(diff.sum()), int(nan.sum())


def corrupt(outs: dict) -> None:
    """--selftest: one bit of the dense output and of one finite GEMV output."""
    if isinstance(outs.get("dense"), torch.Tensor):
        flat = outs["dense"].view(torch.int16).view(-1)
        flat[12345 % flat.numel()] ^= 1
    y = outs.get("gemv x bf16")
    if isinstance(y, torch.Tensor):
        finite = torch.nonzero(torch.isfinite(y)).reshape(-1)
        k = int(finite[0]) if finite.numel() else 0
        y.view(torch.int32)[k] ^= 1


def check_case(name: str, U: np.ndarray, p: dict, decoders: list[int], selftest: bool) -> list[dict]:
    R, C = int(p["R"]), int(p["C"])
    g = torch.Generator().manual_seed(X_SEED)
    x1 = torch.randn(C, generator=g).to(torch.bfloat16)
    x1f = torch.randn(C, generator=g)
    xm = torch.randn(max(MCS), C, generator=g).to(torch.bfloat16)
    src = torch.from_numpy(np.ascontiguousarray(U).view(np.int16))
    order = [REFERENCE] + [d for d in decoders if d != REFERENCE]
    outs = {}
    for d in order:
        t0 = time.perf_counter()
        outs[d] = run_decoder(p, d, x1, x1f, xm)
        outs[d]["seconds"] = time.perf_counter() - t0
    if selftest:
        victim = next((d for d in order if d != REFERENCE), REFERENCE)
        corrupt(outs[victim])
        print(f"  SELFTEST: decoder {victim}'s dense output and bf16-x GEMV output each carry one flipped bit",
              flush=True)
    checks = []

    def record(d, check, ok, detail, differ=None, total=None, nan=None, error=None):
        checks.append({"decoder": d, "check": check, "ok": ok, "differ": differ, "total": total,
                       "nan_both": nan, "error": error})
        print(f"  {name:15s} dec {d}  {check:26s} {'PASS' if ok else 'FAIL'}  {detail}", flush=True)

    for d in order:
        mine, ref = outs[d], outs[REFERENCE]
        dense = mine["dense"]
        if isinstance(dense, str):
            record(d, "dense == source", False, f"ERROR {dense}", error=dense)
        else:
            n = bits_differ(dense, src)
            record(d, "dense == source", n == 0, f"{n} of {R * C} differ", n, R * C)
        if d == REFERENCE:
            continue
        for arm in ["dense"] + [k for k in mine if k.startswith(("gemv", "mc")) and not k.endswith("traceback")]:
            check = f"{arm} == dec {REFERENCE}"
            a, b = mine[arm], ref[arm]
            if isinstance(a, str) or isinstance(b, str):
                err = a if isinstance(a, str) else f"reference: {b}"
                record(d, check, False, f"ERROR {err}", error=err)
            elif arm == "dense":
                n = bits_differ(a, b)
                record(d, check, n == 0, f"{n} of {a.numel()} differ", n, a.numel())
            else:
                n, nan = f32_differ(a, b)
                record(d, check, n == 0, f"{n} of {a.numel()} differ ({nan} NaN in both)", n, a.numel(), nan)
    for d in order:  # one traceback per decoder: its arms fail alike
        tb = next(((k, v) for k, v in outs[d].items() if k.endswith("traceback")), None)
        if tb:
            print(f"  {name} dec {d} {tb[0]}:\n{tb[1]}", flush=True)
    for c in checks:
        c["seconds"] = outs[c["decoder"]]["seconds"]
    return checks


# --------------------------------------------------------------------- main --

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--decoders", nargs="+", default=["2,3"],
                    help=f"DECODER indices (radix_kernel_gpu._decode_weight); {REFERENCE} is the reference and always runs")
    ap.add_argument("--cases", nargs="+", default=list(CASES),
                    help=f"any of {', '.join(CASES)}; `ragged` = the three ragged shapes")
    ap.add_argument("--profile", default="sip", choices=rp.PROFILES)
    ap.add_argument("--selftest", action="store_true", help="flip one decoded bit and one GEMV bit: must FAIL")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    decoders = list(dict.fromkeys(_list(a.decoders, int)))
    names = []
    for c in _list(a.cases):
        for n in ([k for k in CASES if k.startswith("ragged")] if c == "ragged" else [c]):
            if n not in CASES:
                ap.error(f"--cases {n}: one of {', '.join(CASES)} or ragged")
            if n not in names:
                names.append(n)
    encoder = os.environ.get("DRINKME_RADIX_ENCODER", "").strip() or (
        "native" if radix_native.available() else "numpy")

    t0 = time.perf_counter()
    print(f"radix_lean_interp: Triton interpreter (TRITON_INTERPRET=1, triton {triton.__version__}, torch "
          f"{torch.__version__}), decoders {decoders} against decoder {REFERENCE}, profile {a.profile}, "
          f"{encoder} encoder", flush=True)
    if a.selftest:
        print("SELFTEST: one decoded bit and one GEMV bit are flipped on purpose; this run MUST end VERDICT: FAIL",
              flush=True)
    receipt = {"tool": "radix_lean_interp", "triton": triton.__version__, "torch": torch.__version__,
               "numpy": np.__version__, "profile": a.profile, "decoders": decoders, "reference": REFERENCE,
               "selftest": a.selftest, "encoder": encoder, "cases": []}
    failed = []
    selftest = a.selftest
    all_cands = candidates()
    for name in names:
        tc = time.perf_counter()
        rec = {"name": name}
        receipt["cases"].append(rec)
        try:
            U, pack, what, rejected = pick(name, all_cands[name], a.profile, encoder)
            for r in rejected:
                print(f"case {name}: pack_array_radix returned None for {r} (radix would expand it: the raw "
                      f"fallback, no decoder runs); trying the next candidate", flush=True)
            share, no_term, blocks = coverage(U, pack)
            p = to_device_radix(pack, "cpu")
            print(f"case {name}: {pack['R']}x{pack['C']} {a.profile} widths {tuple(pack['widths'])}, "
                  f"{pack['bpw']:.3f} bpw: {what}; tier 0 holds {100 * share:.1f}% of the exponents, "
                  f"{no_term} of {blocks} blocks escape nothing", flush=True)
            rec.update(shape=[int(pack["R"]), int(pack["C"])], bpw=pack["bpw"], data=what, rejected=rejected,
                       tier0_share=share, blocks=blocks, blocks_without_escape=no_term)
            checks = check_case(name, U, p, decoders, selftest)
            selftest = False  # the first case carries the corruption
            rec["checks"] = checks
            ok = all(c["ok"] for c in checks)
        except Exception as e:  # noqa: BLE001 — the verdict line is the contract
            rec["error"] = f"{type(e).__name__}: {e}"
            rec["traceback"] = traceback.format_exc()
            print(f"case {name}: ERROR {rec['error']}\n{rec['traceback']}", flush=True)
            ok = False
        rec["ok"] = ok
        rec["seconds"] = time.perf_counter() - tc
        print(f"case {name}: {'PASS' if ok else 'FAIL'} ({rec['seconds']:.1f}s)", flush=True)
        if not ok:
            failed.append(name)
    wall = time.perf_counter() - t0
    verdict = (f"VERDICT: PASS {len(names)} cases" if not failed else
               f"VERDICT: FAIL {len(failed)} of {len(names)} cases: {', '.join(failed)}")
    receipt.update(wall_seconds=wall, failed=failed, verdict=verdict)
    if a.json:
        os.makedirs(os.path.dirname(os.path.abspath(a.json)), exist_ok=True)
        with open(a.json, "w") as f:
            json.dump(receipt, f, indent=1, default=str)
        print(f"wrote {a.json}", flush=True)
    if a.selftest:
        print(f"SELFTEST {'PASSED: the corruption was caught' if failed else 'FAILED: the corruption was NOT caught'}",
              flush=True)
    print(f"wall time {wall:.1f}s", flush=True)
    print(verdict, flush=True)
    return 0 if not failed else 1


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(rc)
