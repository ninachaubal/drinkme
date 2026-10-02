"""THE RADIX SCHEDULE GATE: every
launch schedule the M=1 GEMV can run at — blocks-per-program in {1, 2, 4,
whole row} x warps in {1, 2, 4} (codec/radix_schedule.py; split-K partials
+ _finish when the program holds less than a row, the kernel's own
fused output when it holds the row) — against the reference decoder, on
all 65,536 bf16 bit patterns and on the real Qwen3-8B layer-0 + lm_head
tensors off the verified packs.

    PYTHONPATH=src .venv/bin/python bench/radix_schedule_bitpin.py [--json out.json]
    PYTHONPATH=src .venv/bin/python bench/radix_schedule_bitpin.py --selftest

Claims:

  bits      all 65,536 bf16 patterns (signed zeros, subnormals, inf, every
            NaN payload) on a compressible background, every profile
            (sip, balanced, gulp), 32-
            and 1024-weight blocks: the dense decode (radix_ops.decode_bits)
            returns them EXACTLY at every warp count of the grid; and the
            GEMV PATH's own decode — the same _decode inside _gemv — is
            pinned per schedule variant by unit-vector probes: y = W.e_j for
            every column j equals column j of the source bits, bitwise, on
            every row whose values are all finite (0 x inf is NaN by IEEE,
            so the three inf/NaN rows of the pattern tensor are excluded
            from the probe and covered by the dense decode).
  math      finite ragged + production-shaped tensors, every profile, every
            variant: the fp32 GEMV within the float64 oracle's bound
            (2e-6 * |x|.|W| + 1e-7); the fused bf16 output with bias ==
            (fp32 + bias).to(bf16) exactly; the module's M=1 forward at
            that launch == the fused kernel. Whether a variant's fp32 GEMV
            is BITWISE the `spike` schedule's (t1/w2) is recorded, not gated: the
            summation order differs by design.
  real      Qwen3-8B off the packs (--real-arms; default sip layer 0
            (seven) + lm_head, gulp layer 0): the CPU reference decoder
            (radix_pack.decode_back_radix) == the checkpoint's safetensors
            bits; the dense decode at every warp count == those bits; every
            variant's GEMV on a random bf16 activation within the float64
            bound of the source bits; the fused bf16 output == the fp32
            result rounded once.
  table     every row of codec/radix_schedule.TABLES — box class (gfx1151,
            gfx1102, cuda) x profile family (sip, scheduled) x shape class — is a schedule
            this gate ran: the grid is widened to hold any row the table
            names, and per real tensor the row each box class's table selects
            for it (select(..., override="table=<box class>")) is pinned to be
            among the schedules checked above. The receipt carries the table.

--selftest flips one decoded bit and must print FAIL and exit 1. ROCm torch
cannot exit nonzero once HIP is up, so the verdict line is authoritative
and the exit goes through os._exit.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback

os.environ.setdefault("DRINKME_PREFILL_DENSE_MIN", "9")

import numpy as np  # noqa: E402
import torch  # noqa: E402
import triton  # noqa: E402

sys.path.insert(0, "src")
sys.path.insert(0, "bench")
from drinkme.codec import radix, radix_ops, radix_pack as rp, radix_schedule as rs  # noqa: E402
from drinkme.codec.pack import FORMAT_VERSION  # noqa: E402
from drinkme.codec.swap import RadixCompressedLinear  # noqa: E402
import parity_common as pc  # noqa: E402

PROFILES = ("sip", "balanced", "gulp")
# the launch table's rows (box class x family x class), so the grid this gate
# runs is never narrower than what a box would serve at
TABLE_ROWS = sorted({(row["gemv_tiles"], row["gemv_warps"])
                     for table in rs.TABLES.values() for rows in table.values() for row in rows.values()})
WARPS = tuple(sorted({1, 2, 4} | {w for _, w in TABLE_ROWS}))
TILES = tuple(sorted({1, 2, 4} | {t for t, _ in TABLE_ROWS if t})) + (0,)
RAGGED = ((7, 33), (3, 1025), (129, 260), (17, 3079))
PRODUCTION = ((1024, 4096), (256, 12288), (4096, 1024))
SPIKE = "t1/w2"


def configs_for(nb: int):
    return pc.configs_for(nb, TILES, WARPS)


def table_rows_for(R: int, C: int, block_size: int, widths) -> dict:
    """label per box class (gfx1151, gfx1102, cuda): the schedule each box
    class's table selects for this tensor (its family and shape class)."""
    out = {}
    for box in rs.BOX_CLASSES:
        l = rs.select(R, C, block_size, widths, override=f"table={box}")
        out[box] = f"t{'row' if l.gemv_tiles == 0 else l.gemv_tiles}/w{l.gemv_warps}"
    return out


def _dict_from_radixpack(rpk: radix.RadixPack, profile: str) -> dict:
    stored = rpk.palette.nbytes + rpk.offsets.nbytes + rpk.data.nbytes
    return {"rx_palette": rpk.palette, "rx_offsets": rpk.offsets.astype(np.uint32),
            "rx_data": rpk.data.astype(np.uint32), "R": rpk.shape[0], "C": rpk.shape[1],
            "bpw": 8.0 * stored / (rpk.shape[0] * rpk.shape[1]), "codec": "radix",
            "profile": profile, "widths": list(rpk.widths), "block_size": rpk.block_size,
            "layout": 0, "format_version": FORMAT_VERSION}


def _finite_bits(rng, shape, spread=False):
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


def _u16(t: torch.Tensor) -> np.ndarray:
    return t.view(torch.int16).cpu().numpy().view(np.uint16)


def check_dense(rt: dict, want: np.ndarray, label: str, corrupt: bool = False) -> None:
    for w in WARPS:
        got = _u16(radix_ops.decode_bits(rt, num_warps=w))
        if corrupt:
            got = got.copy()
            got.flat[0] ^= 1
            corrupt = False
        if not np.array_equal(got, want):
            where = np.argwhere(got != want)[0]
            raise AssertionError(f"bits differ: {label} dense decode at {w} warps, at {tuple(where)}")


def check_patterns(receipt, corrupt: bool) -> int:
    cases = 0
    exhaustive = np.zeros((512, 513), np.uint16)
    exhaustive.flat[:65536] = np.arange(65536, dtype=np.uint16)
    W = torch.from_numpy(exhaustive.view(np.int16).copy()).view(torch.bfloat16).cuda()
    finite_rows = torch.isfinite(W.float()).all(dim=1)
    R, C = exhaustive.shape
    for profile in PROFILES:
        for block in (32, 1024):
            rpk = radix.pack_array(exhaustive, compression_profile=profile, block_size=block)
            if rpk.raw:
                raise AssertionError("the all-pattern tensor must ride the compressed streams")
            p = _dict_from_radixpack(rpk, profile)
            rt = pc.to_runtime(p, "cuda")
            label = f"{profile} B{block}"
            check_dense(rt, exhaustive, label, corrupt)
            corrupt = False
            nb = triton.cdiv(C, block)
            configs = configs_for(nb)
            probes = 0
            for cfg_label, tiles, warps in configs:
                rt2 = pc.with_launch(rt, tiles, warps)
                # unit-vector probes: y = W.e_j is column j exactly on finite rows
                for j in range(C):
                    x = torch.zeros(C, dtype=torch.bfloat16, device="cuda")
                    x[j] = 1
                    y = radix_ops.gemv(rt2, x)
                    want = W[:, j].float()
                    if not torch.equal(y[finite_rows], want[finite_rows]):
                        bad = torch.nonzero(y[finite_rows] != want[finite_rows])[0].item()
                        raise AssertionError(f"GEMV-path bits differ: {label} {cfg_label} column {j} (finite row #{bad})")
                    probes += 1
                cases += 1
            receipt["bits"].append({"profile": profile, "block": block, "bpw": round(p["bpw"], 4),
                                    "warps_dense": list(WARPS), "configs": [c[0] for c in configs],
                                    "unit_probes": probes, "finite_rows": int(finite_rows.sum().item()), "rows": R})
            print(f"BITS PASS  {label:14s} 65,536 patterns dense-exact at warps {WARPS}; GEMV-path exact "
                  f"over {probes} unit probes across {len(configs)} schedules ({int(finite_rows.sum())}/{R} finite rows)", flush=True)
    return cases


def check_math(receipt) -> int:
    rng = np.random.default_rng(831)
    g = torch.Generator(device="cuda").manual_seed(916)
    cases = 0
    for shape in RAGGED + PRODUCTION:
        for spread in (False, True):
            bits = _finite_bits(rng, shape, spread)
            R, C = shape
            for profile in PROFILES:
                p = rp.pack_array_radix(bits, profile)
                if p is None:
                    receipt["math"].append({"shape": list(shape), "spread": spread, "profile": profile,
                                            "note": "radix would expand: pack stores raw"})
                    continue
                rt = pc.to_runtime(p, "cuda")
                check_dense(rt, bits, f"{shape} {profile}")
                bias = torch.linspace(-.1, .1, R, device="cuda", dtype=torch.bfloat16).contiguous()
                x = torch.randn((C,), device="cuda", dtype=torch.bfloat16, generator=g)
                y, bound = pc.oracle(x, torch.from_numpy(bits.view(np.int16).copy()).cuda().view(torch.uint16))
                nb = triton.cdiv(C, int(p["block_size"]))
                configs = configs_for(nb)
                spike_y = None
                row = {"shape": list(shape), "spread": spread, "profile": profile, "bpw": round(p["bpw"], 4), "configs": {}}
                for cfg_label, tiles, warps in configs:
                    rt2 = pc.with_launch(rt, tiles, warps)
                    y32 = radix_ops.gemv(rt2, x)
                    err = pc.gemv_error(y32, y, bound)
                    if not err <= 1:
                        raise AssertionError(f"GEMV exceeds the float64 bound {shape} {profile} {cfg_label}: {err:.3f}")
                    fused = radix_ops.gemv_fused(rt2, x, bias, torch.bfloat16)
                    unfused = (y32.reshape(1, -1) + bias.float()).to(torch.bfloat16)
                    if not torch.equal(fused, unfused):
                        raise AssertionError(f"fused bf16 output differs {shape} {profile} {cfg_label}")
                    mod = RadixCompressedLinear(rt2, bias)
                    xb = x.reshape(1, -1)
                    if mod._route(xb) != "gemv":
                        raise AssertionError("M=1 did not route to gemv")
                    if not torch.equal(mod(xb), fused):
                        raise AssertionError(f"module M=1 forward != fused kernel {shape} {profile} {cfg_label}")
                    if cfg_label == SPIKE:
                        spike_y = y32.clone()
                    row["configs"][cfg_label] = {"err_over_bound": round(err, 4),
                                                 "bitwise_spike": None if spike_y is None else bool(torch.equal(y32, spike_y))}
                    cases += 1
                receipt["math"].append(row)
                worst = max(c["err_over_bound"] for c in row["configs"].values())
                bitwise = sum(1 for c in row["configs"].values() if c["bitwise_spike"])
                print(f"MATH PASS  {str(shape):13s} spread={spread!s:5s} {profile:8s} {len(configs)} schedules: worst err/bound "
                      f"{worst:.3f}, fused+module exact; {bitwise}/{len(configs)} bitwise t1/w2", flush=True)
    return cases


def check_real(receipt, names_by_pack) -> int:
    g = torch.Generator(device="cuda").manual_seed(917)
    cases = 0
    for arm, names in names_by_pack.items():
        packs = pc.load_pack_dicts(pc.PACKS[arm], names)
        for name in names:
            p = packs[name]
            t0 = time.perf_counter()
            src = pc.source_bits(name)
            ref = rp.decode_back_radix(p)
            if not np.array_equal(ref, src):
                raise AssertionError(f"{arm} {name}: CPU reference decoder != checkpoint bits")
            rt = pc.to_runtime(p, "cuda")
            check_dense(rt, src, f"{arm} {name}")
            src_gpu = torch.from_numpy(src.view(np.int16).copy()).cuda().view(torch.uint16)
            R, C = src.shape
            x = torch.randn((C,), device="cuda", dtype=torch.bfloat16, generator=g)
            y, bound = pc.oracle(x, src_gpu)
            nb = triton.cdiv(C, int(p["block_size"]))
            configs = configs_for(nb)
            table_rows = table_rows_for(R, C, int(p["block_size"]), p["widths"])
            for box, label in table_rows.items():
                if label not in [c[0] for c in configs]:
                    raise AssertionError(f"{arm} {name}: the {box} table's row {label} is not a schedule this gate runs")
            row = {"arm": arm, "name": name, "shape": [R, C], "cpu_reference_equal": True,
                   "shape_class": rs.shape_class(R, C, int(p["block_size"])), "family": rs.family(p["widths"]),
                   "table_rows": table_rows, "warps_dense": list(WARPS), "configs": {}}
            spike_y = None
            for cfg_label, tiles, warps in configs:
                rt2 = pc.with_launch(rt, tiles, warps)
                y32 = radix_ops.gemv(rt2, x)
                err = pc.gemv_error(y32, y, bound)
                if not err <= 1:
                    raise AssertionError(f"GEMV exceeds the float64 bound {arm} {name} {cfg_label}: {err:.3f}")
                fused = radix_ops.gemv_fused(rt2, x, None, torch.bfloat16)
                if not torch.equal(fused, y32.reshape(1, -1).to(torch.bfloat16)):
                    raise AssertionError(f"fused bf16 output differs {arm} {name} {cfg_label}")
                if cfg_label == SPIKE:
                    spike_y = y32.clone()
                row["configs"][cfg_label] = {"err_over_bound": round(err, 4),
                                             "bitwise_spike": None if spike_y is None else bool(torch.equal(y32, spike_y))}
                cases += 1
            receipt["real"].append(row)
            worst = max(c["err_over_bound"] for c in row["configs"].values())
            bitwise = sum(1 for c in row["configs"].values() if c["bitwise_spike"])
            rows = " ".join(f"{b}={l}" for b, l in table_rows.items())
            print(f"REAL PASS  {arm:13s} {name:36s} {R}x{C}: CPU reference == checkpoint; dense exact at warps {WARPS}; "
                  f"{len(configs)} schedules worst err/bound {worst:.3f}; {bitwise}/{len(configs)} bitwise t1/w2; "
                  f"table rows {rows}; {time.perf_counter() - t0:.1f}s", flush=True)
            del rt, src_gpu, y, bound
            torch.cuda.empty_cache()
        del packs
    return cases


def check_table(receipt) -> int:
    """Every (box class, family, class) row of the table is a schedule inside
    the grid the patterns/math sections ran, at the class's Qwen3-8B shape."""
    shapes = {"head": (151936, 4096), "narrow": (1024, 4096), "long": (4096, 12288),
              "wide": (12288, 4096), "square": (4096, 4096)}
    widths = {"sip": (3, 8), "scheduled": (2, 2, 4, 8)}
    cases = 0
    for box, table in rs.TABLES.items():
        for fam, rows in table.items():
            for cls, cfg in rows.items():
                R, C = shapes[cls]
                if rs.shape_class(R, C) != cls:
                    raise AssertionError(f"{cls}: the class's shape {R}x{C} is not in that class")
                l = rs.select(R, C, 1024, widths[fam], override=f"table={box}")
                want = tuple(cfg[k] for k in ("gemv_tiles", "gemv_warps", "mc_tiles", "mc_warps"))
                if (l.gemv_tiles, l.gemv_warps, l.mc_tiles, l.mc_warps) != want:
                    raise AssertionError(f"{box}/{fam}/{cls}: select() does not return the table's row")
                if (l.family, l.box_class, l.source) != (fam, box, "table:" + box):
                    raise AssertionError(f"{box}/{fam}/{cls}: select() mislabels the row: {l}")
                nb = triton.cdiv(C, 1024)
                label = f"t{'row' if l.gemv_tiles == 0 else l.gemv_tiles}/w{l.gemv_warps}"
                if label not in [c[0] for c in configs_for(nb)]:
                    raise AssertionError(f"{box}/{fam}/{cls}: row {label} is not a schedule this gate runs")
                receipt["table"].append({"box_class": box, "family": fam, "class": cls, "shape": [R, C],
                                         "row": l.describe(), "gemv_label": label})
                cases += 1
    per_box = {b: {f: sorted({r["row"] for r in receipt["table"] if r["box_class"] == b and r["family"] == f})
                       for f in rs.FAMILIES} for b in rs.BOX_CLASSES}
    print(f"TABLE PASS {cases} rows (box class x family x class) inside the grid tiles {TILES} x warps {WARPS}: "
          f"{json.dumps(per_box)}", flush=True)
    return cases


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--json", default=None)
    ap.add_argument("--no-real", action="store_true", help="skip the real-tensor section (packs not present)")
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--real-arms", nargs="*", default=["sip", "gulp"],
                    help="the packs of the real section (parity_common.PACKS keys); lm_head rides with the first")
    a = ap.parse_args()
    if not torch.cuda.is_available():
        print("RADIX_SCHEDULE_BITPIN FAIL: no GPU — this gate cannot be satisfied on CPU")
        return 2
    print(f"device:  {torch.cuda.get_device_name(0)}  torch {torch.__version__}  triton {triton.__version__}", flush=True)
    if a.selftest:
        print("SELFTEST: one decoded bit is flipped on purpose; this run MUST print FAIL", flush=True)
    receipt = {"gate": "radix schedule bit-pin", "device": torch.cuda.get_device_name(0),
               "torch": torch.__version__, "triton": triton.__version__, "selftest": a.selftest,
               "profiles": list(PROFILES), "real_arms": list(a.real_arms),
               "warps": list(WARPS), "tiles": [t or "row" for t in TILES],
               "launch_table": rs.TABLES, "this_backend_family": rs._backend_family(), "this_box_class": rs.box_class(),
               "bits": [], "math": [], "real": [], "table": []}
    t0 = time.perf_counter()
    status = 1
    try:
        cases = check_patterns(receipt, a.selftest)
        cases += check_math(receipt)
        cases += check_table(receipt)
        if not a.no_real:
            names = {arm: pc.layer_names(a.layer) + ((pc.LM_HEAD,) if i == 0 else ())
                     for i, arm in enumerate(a.real_arms)}
            cases += check_real(receipt, names)
        receipt["cases"] = cases
        receipt["verdict"] = "PASS"
        real = "" if a.no_real else (f", {len(receipt['real'])} real Qwen3-8B tensors (layer {a.layer} + lm_head; "
                                     f"{', '.join(a.real_arms)})")
        print(f"RADIX_SCHEDULE_BITPIN PASS: {cases} (tensor, schedule) cases — all 65,536 bf16 patterns dense-exact at "
              f"warps {WARPS} and GEMV-path-exact per schedule, finite shapes within the float64 bound with exact "
              f"fused/module output, every launch-table row ({len(receipt['table'])}: {' x '.join(rs.BOX_CLASSES)} x "
              f"{' x '.join(rs.FAMILIES)} x {len(rs.CLASSES)} classes) inside the grid{real}; "
              f"{time.perf_counter() - t0:.1f}s", flush=True)
        status = 0
    except BaseException as e:  # noqa: BLE001 — the verdict line is the contract
        receipt["verdict"] = "FAIL"
        receipt["error"] = f"{type(e).__name__}: {e}"
        print(f"RADIX_SCHEDULE_BITPIN FAIL: {type(e).__name__}: {e}", flush=True)
        traceback.print_exc()
    if a.json:
        pc.write_json(a.json, receipt)
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
