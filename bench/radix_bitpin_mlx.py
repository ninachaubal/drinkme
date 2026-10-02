#!/usr/bin/env python3
"""THE RADIX GEMV / DENSE GATE ON METAL:
bench/radix_gemv_bitpin.py ported against the MLX arm — metal/gemv_radix
(the fused M=1 decode+GEMV), metal/dense_radix (the decode-to-weight
transient) and serving/engine_mlx.RadixLinear over them. Needs Metal.

    PYTHONPATH=src python bench/radix_bitpin_mlx.py [--fixture path.npz] [--json out.json]
    PYTHONPATH=src python bench/radix_bitpin_mlx.py --selftest
    PYTHONPATH=src python bench/radix_bitpin_mlx.py --kernels-only   # before the registry rows flip

Earns (radix, 0, DENSE, METAL) and (radix, 0, GEMV, METAL). Claims:

  bits      all 65,536 bf16 bit patterns (signed zeros, subnormals, inf,
            every NaN payload) laid on a compressible background so every
            one of them rides the compressed streams — the palette tiers,
            the escapes, the terminal exponent stream — for sip, balanced
            and gulp, at 32- and 1024-weight blocks: the dense kernel
            returns them EXACTLY (vs codec/radix.decode, the CPU oracle),
            and again on a second dispatch (determinism).
  math      on finite ragged and production-shaped tensors, both spreads,
            every profile: the dense kernel bitwise the source; the fp32
            M=1 GEMV (bf16 x) vs a float64 dot with the cancellation-aware
            bound (2e-6 * |x|.|W| + 1e-7, the research gate's); the bf16-x
            read bitwise the fp32-x read of the same values; the fused bf16
            output with bias == RNE(fp32 sum + bias) exactly; determinism
            of the GEMV; and, unless --kernels-only, the module: RadixLinear
            fused at M=1 == the fused kernel, its dense route (M=9) == mlx's
            own matmul over the oracle's bytes bitwise, and the reference
            path the same.
  real      (--fixture) the Qwen3-8B tensors packed on Strix Halo
            (bench/radix_fixture_build.py — seven layer-0 projections +
            lm_head's first 4096 rows at profile sip): the dense kernel's
            bytes hash to the SAFETENSORS bits' sha256 recorded at build;
            the GEMV within the recorded float64 bound; the smallest two
            tensors also decoded by the CPU oracle on this box and compared
            bitwise.

--selftest flips one decoded bit and must print FAIL and exit 1: an
unfailable gate is worse than none. The verdict line is authoritative;
the exit code is printed beside it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import traceback

import numpy as np

sys.path.insert(0, "src")
import mlx.core as mx  # noqa: E402

from drinkme.codec import radix, radix_pack as rp, registry  # noqa: E402
from drinkme.codec.pack import FORMAT_VERSION  # noqa: E402
from drinkme.metal import dense_radix, gemv_radix  # noqa: E402

PROFILES = ("sip", "balanced", "gulp")
RAGGED = ((1, 1), (7, 33), (3, 1025), (129, 260), (17, 3079))
PRODUCTION = ((1024, 4096), (256, 12288), (4096, 1024))
DETERMINISM_REPEATS = 5


def _dict_from_radixpack(rpk: radix.RadixPack, profile: str) -> dict:
    stored = rpk.palette.nbytes + rpk.offsets.nbytes + rpk.data.nbytes
    return {"rx_palette": rpk.palette, "rx_offsets": rpk.offsets.astype(np.uint32),
            "rx_data": rpk.data.astype(np.uint32), "R": rpk.shape[0], "C": rpk.shape[1],
            "bpw": 8.0 * stored / (rpk.shape[0] * rpk.shape[1]), "codec": "radix",
            "profile": profile, "widths": list(rpk.widths), "block_size": rpk.block_size,
            "layout": 0, "format_version": FORMAT_VERSION}


def _finite_bits(rng, shape, spread=False):
    """radix_gemv_bitpin._finite_bits: finite bf16 bits with a realistic (or
    wide, `spread`) exponent distribution — no inf/NaN, so a dot product
    has an oracle."""
    R, C = shape
    if spread:
        k = np.arange(40)
        pk = 0.75 ** k
        exps = rng.choice(100 + k, size=(R, C), p=pk / pk.sum())
    else:
        exps = rng.choice(np.arange(113, 122), size=(R, C),
                          p=np.array([.002, .003, .005, .01, .025, .055, .1, .3, .5]))
    bits = (exps.astype(np.uint16) << 7) | rng.integers(0, 128, (R, C), dtype=np.uint16)
    bits |= rng.integers(0, 2, (R, C), dtype=np.uint16) << 15
    return bits


def _bf16_of_f32(f: np.ndarray) -> np.ndarray:
    u = np.ascontiguousarray(f, dtype=np.float32).view(np.uint32)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)


def _f32_of_bf16(u: np.ndarray) -> np.ndarray:
    return (np.asarray(u, dtype=np.uint32) << 16).view(np.float32)


def _dense_np(res: dict) -> np.ndarray:
    out = dense_radix.decode_dense_resident(res)
    mx.eval(out)
    return np.array(out)


def _gemv_np(res: dict, x: mx.array, bias=None, out_dtype=mx.float32) -> np.ndarray:
    y = gemv_radix.gemv_radix_resident(res, x, bias, out_dtype)
    mx.eval(y)
    return np.array(y.view(mx.uint16) if out_dtype == mx.bfloat16 else y)


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
            res = gemv_radix.resident(p)
            got = _dense_np(res)
            if corrupt:
                got = got.copy()
                got.flat[0] ^= 1
            if not np.array_equal(got, exhaustive):
                where = np.argwhere(got != exhaustive)[0]
                raise AssertionError(f"bits differ: {profile} B{block} at {tuple(where)}")
            if not np.array_equal(_dense_np(res), exhaustive):
                raise AssertionError("dense decode is not deterministic")
            cases += 1
            receipt["bits"].append({"profile": profile, "block": block, "bpw": round(p["bpw"], 4),
                                    "widths": list(rpk.widths)})
            print(f"BITS PASS  {profile:8s} B{block:<5d} 65,536 patterns, widths {list(rpk.widths)}, "
                  f"bpw {p['bpw']:.3f}", flush=True)
    return cases


def _module_checks(p: dict, bits: np.ndarray, bias_np: np.ndarray, xb: mx.array, fused_b: np.ndarray):
    """RadixLinear over the pack dict: the fused M=1 forward == the fused
    kernel's bf16 output on the same bf16 x; the dense route (M=9) and the
    reference path == mlx's own `x @ W.T + bias` over the oracle's bytes."""
    from drinkme.serving.engine_mlx import RadixLinear

    R, C = bits.shape
    bias_mx = mx.array(bias_np).view(mx.bfloat16)
    mod = RadixLinear(p, bias_mx, path="fused", name="gate")
    m1 = np.stack([np.array(mod(xb[i:i + 1]).view(mx.uint16))[0] for i in range(3)])
    if not np.array_equal(m1, fused_b):
        raise AssertionError("module M=1 forward != fused kernel")
    W = mx.array(bits).view(mx.bfloat16)
    rng = np.random.default_rng(77)
    xd = mx.array(_bf16_of_f32(rng.standard_normal((9, C)).astype(np.float32))).view(mx.bfloat16)
    want = mx.addmm(bias_mx, xd, W.T)
    mx.eval(want)
    got = mod(xd)
    mx.eval(got)
    if not np.array_equal(np.array(got.view(mx.uint16)), np.array(want.view(mx.uint16))):
        raise AssertionError("module dense route (M=9) != mx.addmm over the oracle's bytes")
    # a batched, non-contiguous input through the dense route (the engine's prefill shape)
    xn = mx.transpose(mx.array(_bf16_of_f32(rng.standard_normal((3, 4, C)).astype(np.float32))).view(mx.bfloat16), (1, 0, 2))
    want = mx.addmm(bias_mx, xn.reshape(-1, C), W.T).reshape(4, 3, R)
    got = mod(xn)
    mx.eval(want, got)
    if not np.array_equal(np.array(got.view(mx.uint16)), np.array(want.view(mx.uint16))):
        raise AssertionError("module dense route (batched) != mx.addmm over the oracle's bytes")
    ref = RadixLinear(p, bias_mx, path="reference", name="gate-ref")
    if not np.array_equal(ref.decode(), bits):
        raise AssertionError("module reference decode != source bits")
    got = ref(xd[:2])
    want = mx.addmm(bias_mx, xd[:2], W.T)
    mx.eval(got, want)
    if not np.array_equal(np.array(got.view(mx.uint16)), np.array(want.view(mx.uint16))):
        raise AssertionError("module reference route != mx.addmm over the oracle's bytes")


def check_math(receipt, kernels_only: bool) -> int:
    rng = np.random.default_rng(831)
    cases = 0
    for shape in RAGGED + PRODUCTION:
        for spread in (False, True):
            bits = _finite_bits(rng, shape, spread)
            R, C = shape
            W64 = _f32_of_bf16(bits).astype(np.float64)
            for profile in PROFILES:
                p = rp.pack_array_radix(bits, profile)
                if p is None:
                    receipt["math"].append({"shape": list(shape), "spread": spread, "profile": profile,
                                            "note": "radix would expand: pack stores the raw fallback (fallback claim)"})
                    continue
                res = gemv_radix.resident(p)
                got = _dense_np(res)
                if not np.array_equal(got, bits):
                    raise AssertionError(f"finite bits differ {shape} {profile}")
                # x as bf16 bits (the engine's activation dtype); the same values as fp32
                x16 = _bf16_of_f32(rng.standard_normal((3, C)).astype(np.float32))
                x64 = _f32_of_bf16(x16).astype(np.float64)
                oracle = x64 @ W64.T
                bound = 2e-6 * (np.abs(x64) @ np.abs(W64).T) + 1e-7
                xb = mx.array(x16).view(mx.bfloat16)
                xf = mx.array(_f32_of_bf16(x16))
                y32 = np.stack([_gemv_np(res, xb[i]) for i in range(3)])
                err = np.abs(y32.astype(np.float64) - oracle) / bound
                if not np.all(err <= 1):
                    raise AssertionError(f"GEMV exceeds the float64 bound {shape} {profile}: {err.max():.3f}")
                y32f = np.stack([_gemv_np(res, xf[i]) for i in range(3)])
                if not np.array_equal(y32.view(np.uint32), y32f.view(np.uint32)):
                    raise AssertionError(f"bf16-x read != fp32-x read {shape} {profile}")
                again = np.stack([_gemv_np(res, xb[i]) for i in range(3)])
                if not np.array_equal(y32.view(np.uint32), again.view(np.uint32)):
                    raise AssertionError(f"GEMV is not deterministic {shape} {profile}")
                # the fused epilogue: bias in fp32, one rounding
                bias_np = _bf16_of_f32(np.linspace(-.1, .1, R, dtype=np.float32))
                bias_mx = mx.array(_f32_of_bf16(bias_np))
                unfused = _bf16_of_f32(y32 + _f32_of_bf16(bias_np)[None, :])
                fused = np.stack([_gemv_np(res, xb[i], bias_mx, mx.bfloat16) for i in range(3)])
                if not np.array_equal(unfused, fused):
                    raise AssertionError(f"fused bf16 output differs {shape} {profile}")
                if not kernels_only:
                    _module_checks(p, bits, bias_np, xb, fused)
                cases += 1
                receipt["math"].append({"shape": list(shape), "spread": spread, "profile": profile,
                                        "bpw": round(p["bpw"], 4), "widths": list(p["widths"]),
                                        "gemv_max_normalized_err": round(float(err.max()), 4),
                                        "gemv_max_abs_err": float(np.abs(y32 - oracle).max()),
                                        "bf16x_equals_f32x": True, "fused_equal": True,
                                        "module_checked": not kernels_only})
                print(f"MATH PASS  {str(shape):14s} spread={spread!s:5s} {profile:8s} bpw {p['bpw']:.3f}  "
                      f"gemv err/bound {err.max():.3f}{'' if not kernels_only else '  (kernels only)'}", flush=True)
    return cases


def check_real(receipt, fixture: str, corrupt: bool) -> int:
    with np.load(fixture) as z:
        meta = json.loads(str(z["meta"]))
        names = [t["name"] for t in meta["tensors"]]
        arrays = {k: z[k] for k in z.files if k != "meta"}
    print(f"FIXTURE  {fixture}: {meta['model']} @ {meta['snapshot'][:12]}, profile {meta['profile']}, "
          f"{len(names)} tensors, built {meta['built']}", flush=True)
    cases = 0
    cpu_checked = sorted(meta["tensors"], key=lambda t: t["R"] * t["C"])[:2]
    cpu_names = {t["name"] for t in cpu_checked}
    for t in meta["tensors"]:
        name = t["name"]
        p = {"rx_palette": arrays[f"{name}.rx_palette"], "rx_offsets": arrays[f"{name}.rx_offsets"],
             "rx_data": arrays[f"{name}.rx_data"], "R": t["R"], "C": t["C"], "bpw": t["bpw"],
             "codec": "radix", "profile": meta["profile"], "widths": t["widths"],
             "block_size": t["block_size"], "layout": 0, "format_version": FORMAT_VERSION}
        res = gemv_radix.resident(p)
        t0 = time.perf_counter()
        got = _dense_np(res)
        dense_s = time.perf_counter() - t0
        if corrupt:
            got = got.copy()
            got.flat[0] ^= 1
        digest = hashlib.sha256(np.ascontiguousarray(got).view(np.uint8).tobytes()).hexdigest()
        if digest != t["sha256"]:
            raise AssertionError(f"{name}: dense decode sha256 {digest[:16]} != source bits {t['sha256'][:16]}")
        cpu = None
        if name in cpu_names:
            t0 = time.perf_counter()
            oracle_bits = gemv_radix.decode_reference(res)
            cpu = time.perf_counter() - t0
            if not np.array_equal(oracle_bits, got):
                raise AssertionError(f"{name}: CPU oracle decode != dense kernel")
        x16 = arrays[f"{name}.x"]
        y64 = arrays[f"{name}.y64"]
        bound = arrays[f"{name}.bound"]
        xb = mx.array(x16).view(mx.bfloat16)
        y32 = np.stack([_gemv_np(res, xb[i]) for i in range(x16.shape[0])])
        err = np.abs(y32.astype(np.float64) - y64) / bound
        if not np.all(err <= 1):
            raise AssertionError(f"{name}: GEMV exceeds the float64 bound: {err.max():.3f}")
        again = np.stack([_gemv_np(res, xb[i]) for i in range(x16.shape[0])])
        if not np.array_equal(y32.view(np.uint32), again.view(np.uint32)):
            raise AssertionError(f"{name}: GEMV is not deterministic")
        # bf16 out: the fused bf16 output == RNE of the fp32 sum
        yb = np.stack([_gemv_np(res, xb[i], None, mx.bfloat16) for i in range(x16.shape[0])])
        if not np.array_equal(yb, _bf16_of_f32(y32)):
            raise AssertionError(f"{name}: bf16 output != RNE(fp32 sum)")
        cases += 1
        rec = {"name": name, "R": t["R"], "C": t["C"], "bpw": t["bpw"], "sha256": digest,
               "dense_s_first": round(dense_s, 4), "gemv_max_normalized_err": round(float(err.max()), 4),
               "gemv_max_abs_err": float(np.abs(y32 - y64).max()),
               "cpu_oracle_checked": name in cpu_names, "cpu_oracle_s": None if cpu is None else round(cpu, 2),
               "resident_bytes": gemv_radix.resident_bytes(res)}
        receipt["real"].append(rec)
        print(f"REAL PASS  {name:40s} {t['R']:6d}x{t['C']:<5d} bpw {t['bpw']:.3f}  sha256 ok  "
              f"gemv err/bound {err.max():.3f}{'  cpu-oracle ok' if cpu is not None else ''}", flush=True)
        del res, got
    return cases


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--json", default=None)
    ap.add_argument("--fixture", default=None, help="the Qwen3-8B fixture npz (bench/radix_fixture_build.py)")
    ap.add_argument("--kernels-only", action="store_true",
                    help="skip the RadixLinear module checks (the run BEFORE the registry rows flip)")
    ap.add_argument("--skip-synthetic", action="store_true", help="the fixture section only")
    a = ap.parse_args()
    if not mx.metal.is_available():
        print("RADIX_BITPIN_MLX FAIL: no Metal — this gate cannot be satisfied without one")
        return 2
    dev = mx.device_info()
    print(f"device:  {dev.get('device_name')}  mlx {mx.__version__}", flush=True)
    if a.selftest:
        print("SELFTEST: one decoded bit is flipped on purpose; this run MUST print FAIL", flush=True)
    rows = {op: registry.supported(registry.RADIX, 0, op, registry.METAL)
            for op in (registry.DENSE, registry.GEMV)}
    print(f"registry: (radix, 0, dense, metal)={rows['dense']} (radix, 0, gemv, metal)={rows['gemv']}"
          f"{'  — kernels only' if a.kernels_only else ''}", flush=True)
    receipt = {"gate": "radix gemv/dense bit-pin on Metal",
               "device": dev.get("device_name"), "mlx": mx.__version__, "selftest": a.selftest,
               "kernels_only": a.kernels_only, "registry_rows": rows, "fixture": a.fixture,
               "bits": [], "math": [], "real": []}
    t0 = time.perf_counter()
    status = 1
    try:
        cases = 0
        if not a.skip_synthetic:
            cases += check_all_patterns(receipt, a.selftest)
            cases += check_math(receipt, a.kernels_only)
        if a.fixture:
            cases += check_real(receipt, a.fixture, a.selftest and a.skip_synthetic)
        receipt["cases"] = cases
        receipt["verdict"] = "PASS"
        claims = []
        if not a.skip_synthetic:
            claims.append(f"all 65,536 bf16 patterns through {len(PROFILES)} profiles at 32/1024-weight "
                          "blocks, GEMV vs float64, bf16-x == fp32-x, fused epilogue, determinism"
                          + (", module routes" if not a.kernels_only else " (kernels only)"))
        if a.fixture:
            claims.append("real Qwen3-8B tensors vs the safetensors digests, GEMV vs the float64 oracle, "
                          "CPU oracle on the two smallest")
        print(f"RADIX_BITPIN_MLX PASS: {cases} cases — {'; '.join(claims)}; "
              f"{time.perf_counter() - t0:.1f}s", flush=True)
        status = 0
    except BaseException as e:  # noqa: BLE001 — the verdict line is the contract
        receipt["verdict"] = "FAIL"
        receipt["error"] = f"{type(e).__name__}: {e}"
        print(f"RADIX_BITPIN_MLX FAIL: {type(e).__name__}: {e}", flush=True)
        traceback.print_exc()
    receipt["seconds"] = round(time.perf_counter() - t0, 1)
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
    print(f"exit {status}", flush=True)
    return status


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(rc)
