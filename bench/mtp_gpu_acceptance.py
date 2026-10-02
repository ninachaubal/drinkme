#!/usr/bin/env python3
"""MTP GPU acceptance on the real 27B, compressed arm.

One gate and one informational report (the CPU suite covers the rest):
  1. AGREEMENT (informational, never PASS/FAIL): MTP-on greedy text vs
     MTP-off greedy text per prompt. Token-exact between speculation on and
     off is not a property this repo asserts — a batched
     verify forward and a serial single-token one can round differently in
     accumulation order, and a near-tie can flip an argmax for reasons that
     have nothing to do with a bug. `compare()` below never puts identity in
     `fails`; this file just names the first-divergence index so a human can
     judge whether it looks like a near-tie or a real bug.
  2. SPEED (the gate): decode tok/s on vs off, plus the engine's own
     acceptance log.

One engine build (DRINKME_MTP_DEPTH=4 loads the head); per-request arm toggle via
`set_arm`. DRINKME_PREFIX_SLOTS=0 (per-request caches, so every run is
cold; not the shipped default), thinking disabled.

THE ARM TOGGLE: HFEngine caches the depth in `_mtp_depth` at the FIRST
generation, so setting the depth variable alone between requests would pin
the first arm's depth for the life of the engine, and every later "ON" run
would be the serial loop wearing an ON label, reporting ~1.0x. set_arm drops
the cache, which is the same seam test_mtp_depth_is_not_re_read_per_request uses.

THE HEAD DIET is the third gate, and it is a
BETWEEN-PROCESS A/B, not a per-request one: whether the head is served
compressed is decided once, when the head is loaded. So run this file twice,
`--diet 1` and `--diet 0`, and diff the receipts. Each run prints a RESIDENCY
line (free VRAM before and after the whole build, plus the charge the fit
guard was handed) and per-arm acceptance counters, so the pass criteria —
acceptance and counters within noise, decode tok/s within the 3% band or
better, residency line shows the saving — are all read off the two JSONs.

Two runs rather than two heads in one process on purpose: holding a raw and a
compressed head at once would distort the very number the diet exists to move.

    uv run --no-sync python bench/mtp_gpu_acceptance.py \\
        --pack ~/.cache/drinkme/packs/Qwen--Qwen3.8-27B@1d4bf0f2ff60 \\
        --diet 1 --json /tmp/mtp_diet_on.json
    uv run --no-sync python bench/mtp_gpu_acceptance.py \\
        --pack ~/.cache/drinkme/packs/Qwen--Qwen3.8-27B@1d4bf0f2ff60 \\
        --diet 0 --json /tmp/mtp_diet_off.json
    uv run --no-sync python bench/mtp_gpu_acceptance.py \\
        --compare /tmp/mtp_diet_on.json /tmp/mtp_diet_off.json   # the verdict
"""
import json
import os
import sys
import time

ARGV = sys.argv[1:]


def opt(flag, default=None):
    if flag in ARGV:
        i = ARGV.index(flag)
        v = ARGV[i + 1]
        del ARGV[i:i + 2]
        return v
    return default


COMPARE = "--compare" in ARGV
STOCK = "--stock" in ARGV
MODEL = opt("--model", "Qwen/Qwen3.8-27B")
REVISION = opt("--revision")
# the 27B's default pack directory (`drinkme pack` with no -o, at the pinned revision)
PACK = os.path.expanduser(opt(
    "--pack", "~/.cache/drinkme/packs/Qwen--Qwen3.8-27B@1d4bf0f2ff60"))
N_TOKENS = int(opt("--tokens", "256"))
OUT_JSON = opt("--json")
DIET = opt("--diet")
CTX = int(opt("--ctx", "16384"))

os.environ["DRINKME_MTP_DEPTH"] = "4"              # head loads at engine build
os.environ["DRINKME_PREFIX_SLOTS"] = "0"  # cold caches for a like-for-like A/B (the shipped default reuses them)
if DIET is not None:
    # read by mtp.diet_enabled at head-load time (and by the residency guard,
    # which must charge whatever this run is about to load)
    os.environ["DRINKME_MTP_DIET"] = DIET

import torch  # noqa: E402

from drinkme.codec.swap import CompressedLinear  # noqa: E402
from drinkme.serve import build_engine  # noqa: E402
from drinkme.serving import mtp  # noqa: E402
from drinkme.serving.engine import GenerationRequest, SampleParams, complete  # noqa: E402

PROMPTS = {
    "short-essay": "Write a detailed, multi-paragraph explanation of why the sky is blue.",
    "code": "Write a Python function that parses an ISO-8601 timestamp string into a "
            "datetime, handling optional fractional seconds and timezone offsets, with tests.",
    "counting": "Count from 1 to 30, one number per line, no other text.",
}
RUNS_ON = 2  # speed stability check on the MTP arm

# Speculator.stats is the engine's own instrument (it is what the accounting
# line prints); capture it rather than parsing stderr, so the counters in the
# receipt are the counters the server used.
_LAST_STATS = {}
_real_stats = mtp.Speculator.stats


def _stats(self):
    s = _real_stats(self)
    _LAST_STATS.clear()
    _LAST_STATS.update(s)
    return s


mtp.Speculator.stats = _stats


# The diet's pass criteria:
# acceptance and the counters within noise, decode tok/s within a 3% band or
# better, and the residency line showing the saving.
TOKS_BAND = 0.03
ACCEPTANCE_BAND = 0.02  # absolute, on the accepted/drafted rate


def compare(on_path: str, off_path: str) -> int:
    """Diff two receipts (diet ON vs OFF) into a VERDICT. Written here rather
    than left to a reader with two JSONs because a gate whose criteria live in
    somebody's head is not a gate."""
    on = json.load(open(on_path))
    off = json.load(open(off_path))
    fails = []
    print(f"ON : {on_path}  ({on['head_compressed_tensors']} compressed head tensors)")
    print(f"OFF: {off_path}  ({off['head_compressed_tensors']} compressed head tensors)")
    if on["head_compressed_tensors"] == 0:
        fails.append("the ON run served ZERO compressed head tensors — it is "
                     "not the arm it claims to be (pack without a head "
                     "sub-pack? DRINKME_MTP_DIET=0 in the environment?)")
    if off["head_compressed_tensors"]:
        fails.append("the OFF run served compressed head tensors")

    ron, roff = on["residency"], off["residency"]
    saved = None
    if ron.get("head_charge_gib") and roff.get("head_charge_gib"):
        saved = roff["head_charge_gib"] - ron["head_charge_gib"]
        print(f"residency: head charged {ron['head_charge_gib']:.3f} GiB vs "
              f"{roff['head_charge_gib']:.3f} GiB — saves {saved:.3f} GiB "
              f"({100 * saved / roff['head_charge_gib']:.1f}%)")
        if saved <= 0:
            fails.append("the residency charge did not go down")
    if ron.get("build_resident_gib") and roff.get("build_resident_gib"):
        d = roff["build_resident_gib"] - ron["build_resident_gib"]
        print(f"residency: measured build footprint {ron['build_resident_gib']:.3f} "
              f"vs {roff['build_resident_gib']:.3f} GiB — {d:+.3f} GiB")

    for name in on["prompts"]:
        a, b = on["prompts"][name], off["prompts"].get(name)
        if b is None:
            fails.append(f"{name}: absent from the OFF receipt")
            continue
        ma, mb = a["on"][0]["mtp"], b["on"][0]["mtp"]
        d_acc = abs(ma.get("acceptance", 0) - mb.get("acceptance", 0))
        ta, tb = a["on"][0]["decode_toks_per_s"], b["on"][0]["decode_toks_per_s"]
        ratio = ta / tb if tb else 0.0
        print(f"  {name:12s} acceptance {ma.get('acceptance')} vs "
              f"{mb.get('acceptance')} (d={d_acc:.3f})  "
              f"drafted {ma.get('drafted')}/{mb.get('drafted')}  "
              f"decode {ta:.2f} vs {tb:.2f} tok/s ({ratio:.3f}x)  "
              f"identity {a['identity'] if isinstance(a['identity'], str) else 'FORK'}")
        if d_acc > ACCEPTANCE_BAND:
            fails.append(f"{name}: acceptance moved {d_acc:.3f} (> {ACCEPTANCE_BAND})")
        if ratio < 1 - TOKS_BAND:
            fails.append(f"{name}: decode {ratio:.3f}x, below the "
                         f"{1 - TOKS_BAND:.2f}x band")
    for f in fails:
        print(f"  FAIL: {f}")
    print("PASS: the diet is free" if not fails else f"FAIL: {len(fails)} criteria")
    return 1 if fails else 0


def set_arm(engine, depth: int) -> None:
    """Point the engine at MTP depth `depth` for the NEXT request; depth 0 is
    speculation off (DRINKME_SPEC=off). The env vars are what the engine
    reads; `_mtp_depth = None` drops the once-per-engine cache that would
    otherwise serve every later request the first request's answer (the
    hot-loop audit — see the module docstring)."""
    if depth:
        os.environ["DRINKME_MTP_DEPTH"] = str(depth)
        os.environ.pop("DRINKME_SPEC", None)
    else:
        os.environ["DRINKME_SPEC"] = "off"
    engine._mtp_depth = None


def free_gib():
    if not torch.cuda.is_available():
        return None
    return round(torch.cuda.mem_get_info()[0] / 1024 ** 3, 3)


def run_once(engine, prompt):
    deltas = []
    t_first = None
    t0 = time.monotonic()

    def on_delta(s):
        nonlocal t_first
        if t_first is None:
            t_first = time.monotonic()
        deltas.append(s)
        return True

    _LAST_STATS.clear()
    p = SampleParams(temperature=0.0, max_tokens=N_TOKENS)
    r = complete(engine, GenerationRequest([{"role": "user", "content": prompt}], p), on_delta)
    wall = time.monotonic() - t0
    ttft = (t_first - t0) if t_first else wall
    dec = (r.completion_tokens - 1) / (wall - ttft) if wall > ttft and r.completion_tokens > 1 else 0.0
    return {"text": r.text, "finish": r.finish_reason, "tokens": r.completion_tokens,
            "wall_s": round(wall, 2), "ttft_s": round(ttft, 2),
            "decode_toks_per_s": round(dec, 3),
            "mtp": {k: v for k, v in _LAST_STATS.items() if k != "audit"},
            "audit": _LAST_STATS.get("audit")}


def main():
    arm = "stock bf16" if STOCK else "compressed"
    diet_note = "unset (on if the pack carries a head)" if DIET is None else DIET
    print(f"[acceptance] building {arm} engine (head loads now); "
          f"DRINKME_MTP_DIET={diet_note}", flush=True)
    # build_engine does NOT resolve a None pack_path (the CLI resolves it
    # earlier) — first acceptance run died on exactly this, and the failure
    # rode home as exited-0 through the job's own tee pipe. Explicit path.
    head_charge = None if STOCK else mtp.head_gib(MODEL, REVISION, PACK)
    residency = {"free_gib_before": free_gib(),
                 "head_charge_gib": head_charge,
                 "head_charge_raw_gib": mtp.head_gib(MODEL, REVISION),
                 "head_pack": None if STOCK else __import__(
                     "drinkme.codec.pack", fromlist=["x"]).read_mtp_meta(PACK)}
    engine = build_engine(MODEL, REVISION, None if STOCK else PACK,
                          stock=STOCK, ctx=CTX)
    residency["free_gib_after"] = free_gib()
    if residency["free_gib_before"] is not None:
        residency["build_resident_gib"] = round(
            residency["free_gib_before"] - residency["free_gib_after"], 3)
    hp = residency["head_pack"]
    print(f"[acceptance] RESIDENCY: free {residency['free_gib_before']} -> "
          f"{residency['free_gib_after']} GiB; head charged "
          f"{head_charge if head_charge is None else round(head_charge, 3)} GiB "
          f"(raw would be {residency['head_charge_raw_gib'] and round(residency['head_charge_raw_gib'], 3)} GiB)"
          + (f"; head pack {hp['tensorCount']} tensors @ {hp['meanBpw']} bpw (unweighted mean over tensors)"
             if hp else "; no head sub-pack in this pack"), flush=True)
    engine.template_kwargs = {"enable_thinking": False}
    report = {"model": MODEL, "revision": REVISION, "pack": PACK,
              "arm": engine.arm, "meta": engine.meta, "diet": DIET,
              "diet_enabled": mtp.diet_enabled(),
              # isinstance, not the class NAME: a head's projections are
              # CompressedLinear subclasses (RadixCompressedLinear / RawLinear),
              # and a name test counted them as zero once — "the ON run served
              # ZERO compressed head tensors" on a head that was fully packed
              "head_compressed_tensors": sum(
                  1 for m in (engine.mtp_head.modules() if engine.mtp_head else [])
                  if isinstance(m, CompressedLinear)),
              "residency": residency, "n_tokens": N_TOKENS, "prompts": {}}
    print(f"[acceptance] head: {report['head_compressed_tensors']} compressed "
          f"tensors", flush=True)
    for name, prompt in PROMPTS.items():
        entry = {}
        set_arm(engine, 0)
        off = run_once(engine, prompt)
        entry["off"] = {k: v for k, v in off.items() if k != "text"}
        print(f"[{name}] OFF  {entry['off']}", flush=True)
        set_arm(engine, 4)
        ons = []
        for i in range(RUNS_ON):
            on = run_once(engine, prompt)
            ons.append({k: v for k, v in on.items() if k != "text"})
            print(f"[{name}] ON#{i+1} {ons[-1]}", flush=True)
            if i == 0:
                if on["text"] == off["text"]:
                    entry["identity"] = "IDENTICAL"
                else:
                    div = next((j for j, (a, b) in enumerate(zip(off["text"], on["text"]))
                                if a != b), min(len(off["text"]), len(on["text"])))
                    entry["identity"] = {
                        "verdict": "FORK", "first_divergence_char": div,
                        "off_context": off["text"][max(0, div-60):div+40],
                        "on_context": on["text"][max(0, div-60):div+40],
                    }
        entry["on"] = ons
        sp = entry["on"][0]["decode_toks_per_s"]
        entry["speedup"] = round(sp / entry["off"]["decode_toks_per_s"], 2) if entry["off"]["decode_toks_per_s"] else None
        acc = entry["on"][0]["mtp"]
        print(f"[{name}] identity={entry['identity'] if isinstance(entry['identity'], str) else 'FORK'} "
              f"speedup={entry['speedup']}x acceptance={acc.get('acceptance')} "
              f"({acc.get('accepted')}/{acc.get('drafted')} drafted)", flush=True)
        report["prompts"][name] = entry
    print("=== RECEIPT JSON ===")
    print(json.dumps(report, indent=1))
    if OUT_JSON:
        with open(OUT_JSON, "w") as f:
            json.dump(report, f, indent=1)
        print(f"wrote {OUT_JSON}")


if __name__ == "__main__":
    import os as _os

    code = 0
    try:
        if COMPARE:
            i = ARGV.index("--compare")
            code = compare(ARGV[i + 1], ARGV[i + 2])
        else:
            main()
    except Exception:
        import traceback

        traceback.print_exc()
        code = 1
    sys.stdout.flush()
    sys.stderr.flush()
    _os._exit(code)  # os._exit discipline: TheRock torch _exit(0)s on atexit
