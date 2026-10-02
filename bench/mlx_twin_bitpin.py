#!/usr/bin/env python3
"""THE MLX TWIN'S DECODER ON A REAL PACK: metal/twin_radix (the decoder the
bench's twin arm runs once per tensor at load, engine_mlx.RadixTwinLinear)
against the CPU oracle (radix_pack.decode_back_radix: the native decoder
where a C++ compiler is at hand, else codec/radix.decode), on EVERY radix
tensor of a pack. Runs wherever mlx does: mlx[cpu] on Linux, Metal on a Mac.

    PYTHONPATH=src python bench/mlx_twin_bitpin.py --pack <pack dir> [--json out.json]
    PYTHONPATH=src python bench/mlx_twin_bitpin.py --pack <pack dir> --selftest
    PYTHONPATH=src python bench/mlx_twin_bitpin.py --pack <pack dir> --time-tokens 8

Claim: for every radix tensor, the twin decoder's uint16 [R, C] is bitwise
the oracle's. tests/test_twin_radix.py pins the same claim on toys (every
bf16 pattern, ragged blocks, three profiles) in the CPU suite.

--time-tokens N additionally loads the model twice from the pack (its
embedded checkpoint; no network), once on the reference path (the CPU
oracle's decode on every forward, the twin before this decoder) and once
on the twin path (decoded once at load), and times N greedy tokens on
each after one warm-up token: seconds per token on THIS machine. On a
machine without Metal both run on CPU mlx, so the ratio is a rough one
and not a Mac number.

--selftest flips one decoded bit and must print FAIL: an unfailable gate is
worse than none. The verdict line is authoritative; the exit code is
printed beside it.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback

import numpy as np

sys.path.insert(0, "src")
import mlx.core as mx  # noqa: E402

from drinkme.codec import radix_pack as rp  # noqa: E402
from drinkme.codec.pack import iter_pack_dir, verify_hashes  # noqa: E402
from drinkme.metal import twin_radix  # noqa: E402


def check_pack(pack_dir: str, corrupt: bool, receipt: dict) -> int:
    verify_hashes(pack_dir)
    n = 0
    weights = 0
    t_twin = t_oracle = 0.0
    for name, p in iter_pack_dir(pack_dir):
        if not rp.is_radix_pack(p):
            receipt["skipped_raw"].append(name)
            continue
        t0 = time.perf_counter()
        got = np.array(twin_radix.decode(p, name))
        t_twin += time.perf_counter() - t0
        t0 = time.perf_counter()
        want = rp.decode_back_radix(p)
        t_oracle += time.perf_counter() - t0
        if corrupt and n == 0:
            got = got.copy()
            got.flat[0] ^= 1
        if got.shape != want.shape or not np.array_equal(got, want):
            bad = int((got != want).sum()) if got.shape == want.shape else -1
            raise AssertionError(f"{name}: {bad} of {want.size} bf16 patterns differ from the oracle")
        n += 1
        weights += int(want.size)
    receipt.update(tensors=n, weights=weights, twin_seconds=round(t_twin, 2),
                   oracle_seconds=round(t_oracle, 2))
    return n


def time_tokens(pack_dir: str, tokens: int, receipt: dict) -> None:
    from drinkme.arms_mlx import greedy
    from drinkme.serving.engine_mlx import TWIN, load_compressed_mlx

    with open(os.path.join(pack_dir, "meta.json")) as f:
        repo = json.load(f)["hfRepo"]
    out = {}
    for path in ("reference", TWIN):
        t0 = time.perf_counter()
        eng = load_compressed_mlx(repo, None, pack_dir, path=path)
        load_s = time.perf_counter() - t0
        ids = eng.tok.encode("The key idea of lossless weight compression is")
        greedy(eng, ids, 1)  # warm-up
        t0 = time.perf_counter()
        toks = greedy(eng, ids, tokens)
        per = (time.perf_counter() - t0) / tokens
        out[path] = {"load_s": round(load_s, 2), "s_per_token": round(per, 4), "tokens": toks[len(ids):]}
        print(f"TIME {path:9s} load {load_s:6.1f}s  {per:8.4f} s/token over {tokens} tokens "
              f"({eng.device})", flush=True)
        del eng
        mx.clear_cache()
    same = out["reference"]["tokens"] == out[TWIN]["tokens"]
    ratio = out["reference"]["s_per_token"] / out[TWIN]["s_per_token"]
    print(f"TIME the twin path is {ratio:.1f}x the reference path per token on this machine; "
          f"greedy tokens identical: {same}", flush=True)
    receipt["timing"] = {k: {kk: vv for kk, vv in v.items() if kk != "tokens"} for k, v in out.items()}
    receipt["timing"].update(ratio=round(ratio, 2), same_tokens=same, device=mx.metal.is_available()
                             and "metal" or "cpu")
    if not same:
        raise AssertionError("the twin path's greedy tokens differ from the reference path's")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--pack", required=True, help="a pack directory (drinkme pack -o ...)")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--time-tokens", type=int, default=0)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    print(f"mlx {mx.__version__}, metal {mx.metal.is_available()}, pack {a.pack}", flush=True)
    if a.selftest:
        print("SELFTEST: one decoded bit is flipped on purpose; this run MUST print FAIL", flush=True)
    receipt = {"gate": "mlx twin decoder vs the CPU oracle", "mlx": mx.__version__,
               "metal": mx.metal.is_available(), "pack": os.path.abspath(a.pack),
               "selftest": a.selftest, "skipped_raw": []}
    t0 = time.perf_counter()
    status = 1
    try:
        n = check_pack(a.pack, a.selftest, receipt)
        print(f"MLX_TWIN_BITPIN PASS: {n}/{n} radix tensors bitwise the oracle "
              f"({receipt['weights']:,} weights; twin decoder {receipt['twin_seconds']}s, "
              f"oracle {receipt['oracle_seconds']}s; {len(receipt['skipped_raw'])} raw fallbacks "
              f"need no decode)", flush=True)
        if a.time_tokens and not a.selftest:
            time_tokens(a.pack, a.time_tokens, receipt)
        receipt["verdict"] = "PASS"
        status = 0
    except BaseException as e:  # noqa: BLE001 — the verdict line is the contract
        receipt["verdict"] = "FAIL"
        receipt["error"] = f"{type(e).__name__}: {e}"
        print(f"MLX_TWIN_BITPIN FAIL: {type(e).__name__}: {e}", flush=True)
        if not a.selftest:
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
    sys.exit(main())
