"""Bit-pin for the MTP head sub-pack (the head diet).

THE CLAIM, rooted at the checkpoint rather than at the pack: every tensor in
`<pack>/mtp` decodes to the EXACT bf16 bytes the checkpoint ships under
`mtp.<name>.weight`. radix_mc_bitpin.py --pack-dir gates a pack dir against
its own oracle (and works on `<pack>/mtp` unchanged — point it there for the
mc/forward claims on the GPU); this one closes the other half, the one the
head diet is actually asserting: the head the server drafts with is the head
the checkpoint shipped, bit for bit.

Runs on CPU — the reference decode is numpy — so it gates a head sub-pack
before anything reaches a GPU. `--gpu` additionally pins the DEVICE decode
(the served module's _dense_weight: radix_ops.decode_weight for a radix
tensor, the resident weight for a raw fallback) against the same checkpoint
bytes, which is the claim that matters at serve time.

EXITS NONZERO ON THE FIRST MISMATCH. `--selftest` flips one bit on purpose and
asserts this file notices — prove the gate can fail before trusting a green
run (an inverted assertion passes everything).

    uv run --no-sync python bench/mtp_head_bitpin.py <pack_dir> [--repo R]
            [--revision REV] [--gpu] [--selftest] [--json out.json]

`--repo`/`--revision` default to the sub-pack's own recorded provenance, so
the usual invocation is just the pack dir.
"""

from __future__ import annotations

import glob
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, "src")
from drinkme.codec.pack import has_mtp_pack, iter_pack_dir, mtp_pack_dir, read_mtp_meta  # noqa: E402
from drinkme.codec.radix_pack import decode_back  # noqa: E402


def checkpoint_head(repo: str, revision: str | None) -> dict:
    """{name -> uint16 view of the checkpoint's bf16 bytes} for every mtp.*
    tensor, read one at a time out of the shards."""
    import torch
    from safetensors import safe_open

    from drinkme.arms import snapshot_dir

    out = {}
    for fpath in sorted(glob.glob(os.path.join(snapshot_dir(repo, revision),
                                               "*.safetensors"))):
        with safe_open(fpath, framework="pt") as sf:
            for key in sf.keys():
                if not key.startswith("mtp.") or not key.endswith(".weight"):
                    continue
                t = sf.get_tensor(key).to(torch.bfloat16)
                out[key[len("mtp."):-len(".weight")]] = (
                    t.view(torch.int16).numpy().view(np.uint16).copy())
                del t
    return out


def main() -> int:
    argv = sys.argv[1:]

    def opt(flag, default=None):
        if flag in argv:
            i = argv.index(flag)
            v = argv[i + 1]
            del argv[i:i + 2]
            return v
        return default

    out_json = opt("--json")
    repo = opt("--repo")
    revision = opt("--revision")
    use_gpu = "--gpu" in argv
    selftest = "--selftest" in argv
    args = [a for a in argv if not a.startswith("--")]
    if not args:
        print(__doc__, file=sys.stderr)
        return 2
    pack_dir = args[0]
    if not has_mtp_pack(pack_dir):
        print(f"mtp_head_bitpin: {pack_dir} carries no MTP head sub-pack "
              f"({mtp_pack_dir(pack_dir)}/meta.json is absent) — nothing to "
              "gate. Pack one with codec.pack.pack_mtp_head.", file=sys.stderr)
        return 2
    hm = read_mtp_meta(pack_dir)
    repo = repo or hm["hfRepo"]
    revision = revision if revision is not None else hm["revision"]

    print(f"pack:     {mtp_pack_dir(pack_dir)}")
    print(f"source:   {repo} @ {revision}")
    print(f"claim:    every packed head tensor == the checkpoint's bf16 bytes"
          f"{' (CPU + GPU decode)' if use_gpu else ' (CPU decode)'}")
    if selftest:
        print("SELFTEST: one bit is flipped on purpose; this run MUST fail")

    t0 = time.perf_counter()
    truth = checkpoint_head(repo, revision)
    failures, n, elements = [], 0, 0
    for name, pack in iter_pack_dir(mtp_pack_dir(pack_dir)):
        want = truth.get(name)
        if want is None:
            failures.append({"tensor": name, "claim": "present",
                             "detail": "no such tensor in the checkpoint"})
            break
        got = decode_back(pack)
        if selftest and n == 0:
            got = got.copy()
            got[0, 0] ^= 1
        if got.shape != want.shape:
            failures.append({"tensor": name, "claim": "shape",
                             "got": list(got.shape), "want": list(want.shape)})
            break
        if not np.array_equal(got, want):
            failures.append({"tensor": name, "claim": "cpu-decode",
                             "mismatched_elements": int((got != want).sum()),
                             "of_elements": int(want.size)})
            break
        if use_gpu:
            import torch

            from drinkme.codec.swap import make_module

            mod = make_module(pack, None, "cuda")
            W = mod._dense_weight().view(torch.int16).cpu().numpy().view(np.uint16)
            p = mod.p
            if selftest and n == 0:
                W = W.copy()
                W[0, 0] ^= 1
            if not np.array_equal(W, want):
                failures.append({"tensor": name, "claim": "gpu-decode",
                                 "mismatched_elements": int((W != want).sum()),
                                 "of_elements": int(want.size)})
                break
            del p, W
        n += 1
        elements += int(want.size)
        print(f"  {name:34s} {want.shape[0]}x{want.shape[1]}  "
              f"{want.size} elements  OK")
        del got
    dt = time.perf_counter() - t0

    missed = sorted(set(truth) - set(hm["tensors"]))
    verdict = "FAIL" if failures else "PASS"
    for f in failures:
        print(f"  MISMATCH {f}")
    print(f"raw (not packed, streamed from the checkpoint at load): "
          f"{len(missed)} tensors — {', '.join(missed) or 'none'}")
    print(f"{verdict}: {n}/{hm['tensorCount']} head tensors, {elements} "
          f"elements, {len(failures)} mismatches, {dt:.1f}s")

    if out_json:
        with open(out_json, "w") as f:
            json.dump({"gate": "MTP head sub-pack bit-pin",
                       "pack_dir": pack_dir, "repo": repo, "revision": revision,
                       "tensors": n, "elements": elements, "gpu": use_gpu,
                       "raw_tensors": missed, "mismatches": failures,
                       "meanBpw": hm["meanBpw"], "rawBytes": hm["rawBytes"],
                       "residentBytes": hm["residentBytes"],
                       "verdict": verdict, "seconds": round(dt, 1)}, f, indent=1)
        print(f"wrote {out_json}")

    if selftest:
        if failures:
            print("SELFTEST PASSED: the flip was caught (this run's FAIL is correct)")
            return 0
        print("SELFTEST FAILED: the flip was NOT caught — this gate is broken",
              file=sys.stderr)
        return 1
    return 1 if failures else 0


if __name__ == "__main__":
    import os as _os

    try:
        _code = main()
    except SystemExit as e:
        _code = e.code if isinstance(e.code, int) else (1 if e.code else 0)
        if e.code and not isinstance(e.code, int):
            print(e.code, file=sys.stderr)
    sys.stdout.flush()
    sys.stderr.flush()
    _os._exit(_code)  # os._exit discipline: TheRock torch _exit(0)s on atexit
