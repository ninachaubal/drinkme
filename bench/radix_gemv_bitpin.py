"""THE RADIX GEMV / DENSE GATE: the bf16 codec's device gate against THIS
repo's module arms — swap.RadixCompressedLinear over a to_device_radix
runtime dict, codec/radix_ops.py's kernels.

    PYTHONPATH=src .venv/bin/python bench/radix_gemv_bitpin.py [--json out.json]
    PYTHONPATH=src .venv/bin/python bench/radix_gemv_bitpin.py --selftest

Earns (radix, 0, DENSE, TRITON) and (radix, 0, GEMV, TRITON). Claims:

  bits      all 65,536 bf16 bit patterns (signed zeros, subnormals, inf,
            every NaN payload) laid on a compressible background so every
            one of them rides the compressed streams — the palette tiers,
            the escapes, the terminal exponent stream — for every profile
            and for 32- and 1024-weight blocks: radix_ops.decode_bits (the
            dense arm's decode) returns them EXACTLY; the gulp schedule
            the GPU kernel prepares equals radix_pack.schedule_np's.
  fallback  a flat-exponent tensor radix would expand comes back from
            pack_weight_radix as the RAW dict (the pack's fallback);
            make_module installs a RawLinear whose dense weight is the
            source bits and whose forward is stock's F.linear, bitwise.
  math      on finite ragged and production-shaped tensors: the fp32 M=1
            GEMV vs a float64 dot with a cancellation-aware bound
            (2e-6 * |x|.|W| + 1e-7, the research gate's); the fused bf16
            output with bias == (fp32 result + bias).to(bf16) exactly; the
            module's M=1 forward == the fused kernel; the module's dense
            route (M >= dense_min) == F.linear(x, W, bias) bitwise; the CPU
            route == F.linear over the source bits bitwise.

--selftest flips one decoded bit and must print FAIL and exit 1: an
unfailable gate is worse than none. ROCm torch cannot exit nonzero once
HIP is up (AGENTS.md), so the verdict line is authoritative and the exit
goes through os._exit.
"""

import argparse
import json
import os
import sys
import time
import traceback

os.environ.setdefault("DRINKME_PREFILL_DENSE_MIN", "9")  # the shipped threshold, pinned for the run

import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, "src")
from drinkme.codec import radix, radix_ops, radix_pack as rp  # noqa: E402
from drinkme.codec.pack import FORMAT_VERSION  # noqa: E402
from drinkme.codec.swap import RadixCompressedLinear, RawLinear, make_module, to_device_radix  # noqa: E402

PROFILES = ("sip", "balanced", "gulp")
RAGGED = ((1, 1), (7, 33), (3, 1025), (129, 260), (17, 3079))
PRODUCTION = ((1024, 4096), (256, 12288), (4096, 1024))  # k/v-proj-like, down-proj-like, o-proj-like rows


def _dict_from_radixpack(rpk: radix.RadixPack, profile: str) -> dict:
    """A pack tensor dict off the research dataclass — the way to get a
    32-weight-block tensor through our loader (the pack path writes 1024)."""
    stored = rpk.palette.nbytes + rpk.offsets.nbytes + rpk.data.nbytes
    return {"rx_palette": rpk.palette, "rx_offsets": rpk.offsets.astype(np.uint32),
            "rx_data": rpk.data.astype(np.uint32), "R": rpk.shape[0], "C": rpk.shape[1],
            "bpw": 8.0 * stored / (rpk.shape[0] * rpk.shape[1]), "codec": "radix",
            "profile": profile, "widths": list(rpk.widths), "block_size": rpk.block_size,
            "layout": 0, "format_version": FORMAT_VERSION}


def _finite_bits(rng, shape, spread=False):
    """Finite bf16 bits with a realistic (or wide, `spread`) exponent
    distribution — no inf/NaN, so a dot product has an oracle."""
    R, C = shape
    if spread:
        # a long geometric tail over 40 exponents: many escapes into the
        # later tiers and the terminal stream, still compressible
        k = np.arange(40)
        pk = 0.75 ** k
        exps = rng.choice(100 + k, size=(R, C), p=pk / pk.sum())
    else:
        exps = rng.choice(np.arange(113, 122), size=(R, C),
                          p=np.array([.002, .003, .005, .01, .025, .055, .1, .3, .5]))
    bits = (exps.astype(np.uint16) << 7) | rng.integers(0, 128, (R, C), dtype=np.uint16)
    bits |= rng.integers(0, 2, (R, C), dtype=np.uint16) << 15
    return bits


def check_all_patterns(receipt, corrupt: bool) -> int:
    cases = 0
    exhaustive = np.zeros((512, 513), np.uint16)
    exhaustive.flat[:65536] = np.arange(65536, dtype=np.uint16)
    for profile in PROFILES:
        for block in (32, 1024):
            rpk = radix.pack_array(exhaustive, compression_profile=profile, block_size=block)
            if rpk.raw:
                raise AssertionError("the all-pattern tensor must ride the compressed streams")
            p = _dict_from_radixpack(rpk, profile)
            rt = to_device_radix(p, "cuda")
            if rt["rx_schedule"].numel():
                gpu = rt["rx_schedule"].view(torch.int16).cpu().numpy().view(np.uint16)
                if not np.array_equal(gpu, rp.schedule_np(p)):
                    raise AssertionError(f"schedule: GPU kernel != schedule_np ({profile} B{block})")
            got = radix_ops.decode_bits(rt).view(torch.int16).cpu().numpy().view(np.uint16)
            if corrupt:
                got = got.copy()
                got.flat[0] ^= 1
            if not np.array_equal(got, exhaustive):
                where = np.argwhere(got != exhaustive)[0]
                raise AssertionError(f"bits differ: {profile} B{block} at {tuple(where)}")
            # deterministic
            again = radix_ops.decode_bits(rt).view(torch.int16).cpu().numpy().view(np.uint16)
            if not np.array_equal(again, exhaustive):
                raise AssertionError("decode is not deterministic")
            cases += 1
            receipt["bits"].append({"profile": profile, "block": block, "bpw": round(p["bpw"], 4),
                                    "schedule_checked": bool(rt["rx_schedule"].numel())})
            print(f"BITS PASS  {profile:8s} B{block:<5d} 65,536 patterns, bpw {p['bpw']:.3f}", flush=True)
    return cases


def check_fallback(receipt) -> int:
    rng = np.random.default_rng(5)
    R, C = 64, 1024
    bits = (rng.integers(1, 255, (R, C), dtype=np.uint16) << 7) | rng.integers(0, 128, (R, C), dtype=np.uint16)
    bits |= rng.integers(0, 2, (R, C), dtype=np.uint16) << 15
    w = torch.from_numpy(bits.view(np.int16).copy()).view(torch.bfloat16)
    x = torch.randn(3, C, device="cuda", dtype=torch.bfloat16)
    for profile in PROFILES:
        p = rp.pack_weight_radix(w, profile)
        if rp.is_radix_pack(p) or not rp.is_raw_pack(p):
            raise AssertionError(f"{profile}: the flat-exponent tensor should have fallen back to raw")
        mod = make_module(p, None, "cuda")
        if type(mod) is not RawLinear:
            raise AssertionError("fallback tensor did not install as RawLinear")
        dense = mod._dense_weight().view(torch.int16).cpu().numpy().view(np.uint16)
        if not np.array_equal(dense, bits):
            raise AssertionError(f"{profile}: raw fallback's resident weight differs from the source")
        want = torch.nn.functional.linear(x, w.cuda())
        if not torch.equal(mod(x), want):
            raise AssertionError(f"{profile}: raw fallback forward differs from stock's F.linear")
        receipt["fallback"].append({"profile": profile, "stored_as": "raw", "format_version": FORMAT_VERSION, "bpw": p["bpw"]})
        print(f"FALLBACK PASS  {profile:8s} flat-exponent {R}x{C} -> raw (16 bpw), resident bits exact, "
              "forward bitwise F.linear", flush=True)
    return len(PROFILES)


# CUDA finding (L4): with a STRIDED linspace view as the reference's bias
# while the module holds bias.contiguous(), torch on CUDA takes a different
# GEMM path for the two
# (cuBLASLt's fused-bias epilogue wants a contiguous 1-D bias) and 16 of 27
# dense-route cases then differ at accumulation-order level — the module
# equals F.linear over the contiguous bias in 27/27. ROCm took one path for
# both, so the gate passed there either way. The reference therefore holds
# the SAME contiguous tensor the module holds; --strided-bias is the opt-in
# for the strided reference, kept so the CUDA artifact stays reproducible.
CONTIGUOUS_BIAS = True


def check_math(receipt) -> int:
    rng = np.random.default_rng(831)
    g = torch.Generator(device="cuda").manual_seed(916)
    cases = 0
    for shape in RAGGED + PRODUCTION:
        for spread in (False, True):
            bits = _finite_bits(rng, shape, spread)
            W = torch.from_numpy(bits.view(np.int16).copy()).view(torch.bfloat16).cuda()
            for profile in PROFILES:
                p = rp.pack_array_radix(bits, profile)
                if p is None:
                    receipt["math"].append({"shape": list(shape), "spread": spread, "profile": profile,
                                            "note": "radix would expand: pack stores raw (fallback claim)"})
                    continue
                rt = to_device_radix(p, "cuda")
                got = radix_ops.decode_bits(rt).view(torch.int16).cpu().numpy().view(np.uint16)
                if not np.array_equal(got, bits):
                    raise AssertionError(f"finite bits differ {shape} {profile}")
                R, C = shape
                bias = torch.linspace(-.1, .1, R * 2, device="cuda", dtype=torch.bfloat16)[::2]
                if CONTIGUOUS_BIAS:
                    bias = bias.contiguous()
                mod = RadixCompressedLinear(rt, bias.contiguous())
                x = torch.randn((3, C), device="cuda", dtype=torch.float32, generator=g)
                oracle = x.double() @ W.double().T
                bound = 2e-6 * (x.double().abs() @ W.double().abs().T) + 1e-7
                y32 = torch.stack([radix_ops.gemv(rt, x[i]) for i in range(3)])
                err = (y32.double() - oracle).abs() / bound
                if not torch.all(err <= 1):
                    raise AssertionError(f"GEMV exceeds the float64 bound {shape} {profile}: {err.max().item():.3f}")
                # the fused epilogue: bias in fp32, one rounding
                unfused = (y32 + bias.float()).to(torch.bfloat16)
                fused = torch.cat([radix_ops.gemv_fused(rt, x[i], bias, torch.bfloat16) for i in range(3)])
                if not torch.equal(unfused, fused):
                    raise AssertionError(f"fused bf16 output differs {shape} {profile}")
                # the module, M=1, bf16 in: routes to gemv, equals the fused kernel on the same bf16 x
                xb = x.to(torch.bfloat16)
                if mod._route(xb[:1]) != "gemv":
                    raise AssertionError("M=1 did not route to gemv")
                m1 = torch.cat([mod(xb[i:i + 1]) for i in range(3)])
                fused_b = torch.cat([radix_ops.gemv_fused(rt, xb[i], bias, torch.bfloat16) for i in range(3)])
                if not torch.equal(m1, fused_b):
                    raise AssertionError(f"module M=1 forward != fused kernel {shape} {profile}")
                # the dense route: F.linear over the decoded weight, bitwise stock's arithmetic
                xd = torch.randn((9, C), device="cuda", dtype=torch.bfloat16, generator=g)
                if mod._route(xd) != "dense":
                    raise AssertionError("M=9 did not route to dense")
                if not torch.equal(mod(xd), torch.nn.functional.linear(xd, W, bias)):
                    raise AssertionError(f"dense route != F.linear {shape} {profile}")
                # noncontiguous, batched input through the dense route. The
                # module (CompressedLinear.forward, inherited) reshapes to
                # [M, C] rows first — so the reference is F.linear over the
                # same rows, not over the 3-D strided view (torch picks a
                # different GEMM path for that and rounds differently; a
                # comparison artifact of the reference, not of the kernel)
                xn = torch.randn((3, 4, C), device="cuda", dtype=torch.bfloat16, generator=g).transpose(0, 1)
                want = torch.nn.functional.linear(xn.reshape(-1, C), W, bias).reshape(4, 3, R)
                if not torch.equal(mod(xn), want):
                    raise AssertionError(f"dense route (noncontiguous) != F.linear {shape} {profile}")
                # the CPU reference route
                cpu_mod = make_module(p, bias.cpu(), "cpu")
                xc = xb[:2].cpu()
                if not torch.equal(cpu_mod(xc), torch.nn.functional.linear(xc, W.cpu(), bias.cpu())):
                    raise AssertionError(f"CPU route != F.linear {shape} {profile}")
                cases += 1
                receipt["math"].append({"shape": list(shape), "spread": spread, "profile": profile,
                                        "bpw": round(p["bpw"], 4),
                                        "gemv_max_normalized_err": round(err.max().item(), 4),
                                        "gemv_max_abs_err": float((y32.double() - oracle).abs().max()),
                                        "fused_equal": True, "dense_equal": True, "cpu_equal": True})
                print(f"MATH PASS  {str(shape):14s} spread={spread!s:5s} {profile:8s} bpw {p['bpw']:.3f}  "
                      f"gemv err/bound {err.max().item():.3f}", flush=True)
    return cases


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--json", default=None)
    ap.add_argument("--strided-bias", action="store_true",
                    help="the math reference's bias as the STRIDED linspace view (the pre-parity default; "
                         "reproduces the CUDA cuBLASLt-path artifact)")
    ap.add_argument("--contiguous-bias", action="store_true",
                    help="(now the default) the reference's bias contiguous, as the module's is")
    a = ap.parse_args()
    global CONTIGUOUS_BIAS
    if a.strided_bias and a.contiguous_bias:
        print("RADIX_GEMV_BITPIN FAIL: --strided-bias and --contiguous-bias exclude each other")
        return 2
    CONTIGUOUS_BIAS = not a.strided_bias
    if not torch.cuda.is_available():
        print("RADIX_GEMV_BITPIN FAIL: no GPU — this gate cannot be satisfied on CPU")
        return 2
    print(f"device:  {torch.cuda.get_device_name(0)}  torch {torch.__version__}", flush=True)
    if a.selftest:
        print("SELFTEST: one decoded bit is flipped on purpose; this run MUST print FAIL", flush=True)
    receipt = {"gate": "radix gemv/dense bit-pin", "device": torch.cuda.get_device_name(0),
               "torch": torch.__version__, "selftest": a.selftest, "contiguous_bias": CONTIGUOUS_BIAS,
               "bits": [], "fallback": [], "math": []}
    if CONTIGUOUS_BIAS:
        print("CONTIGUOUS BIAS: the math reference's bias is contiguous, as the module's is (default)", flush=True)
    else:
        print("STRIDED BIAS: the math reference's bias is the strided linspace view (--strided-bias)", flush=True)
    t0 = time.perf_counter()
    status = 1
    try:
        cases = check_all_patterns(receipt, a.selftest)
        cases += check_fallback(receipt)
        cases += check_math(receipt)
        receipt["cases"] = cases
        receipt["verdict"] = "PASS"
        print(f"RADIX_GEMV_BITPIN PASS: {cases} cases — all 65,536 bf16 patterns through every profile "
              f"{PROFILES} at 32/1024-weight blocks, raw fallback, GEMV vs float64, fused epilogue, dense "
              f"and CPU routes bitwise; {time.perf_counter() - t0:.1f}s", flush=True)
        status = 0
    except BaseException as e:  # noqa: BLE001 — the verdict line is the contract
        receipt["verdict"] = "FAIL"
        receipt["error"] = f"{type(e).__name__}: {e}"
        print(f"RADIX_GEMV_BITPIN FAIL: {type(e).__name__}: {e}", flush=True)
        traceback.print_exc()
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
