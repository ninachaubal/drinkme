#!/usr/bin/env python3
"""Hot-loop GPU acceptance — one ARM of the A/B.

The hot loop's equivalence tests run on the CPU, which leaves a gap: the items
that touch device behaviour — PenaltyState's O(V) masked `where`, the scratch
buffers, the once-per-step isfinite — want the same on/off check on the
device. DRINKME_HOTLOOP_OFF is read once per ENGINE, so each arm is its
own process; this script runs one arm and writes a JSON receipt; the driver
shell compares the two receipts byte-for-byte.

Arms (set by the driver, not here):
  old: DRINKME_HOTLOOP_OFF=1   — the pre-audit path on the new binary
  new: DRINKME_DETOK_VERIFY=1  — the audited path, window checked against a
       true full decode on EVERY token (the real-tokenizer-family check
       needed before trusting the window on Qwen3)

Matrix per arm: greedy x3, greedy+penalties x2, seeded sampling x1,
constrained JSON x2 — every new code path (cursor, window, PenaltyState,
scratch) under the real tokenizer on the real device.

Verdicts ride the OUTPUT and the receipt, and exit goes through os._exit —
TheRock ROCm torch clobbers exit codes once HIP inits (an atexit handler
calls `_exit(0)` over the real status; never trust `$?` alone from a GPU run).

Usage: hotloop_gpu_acceptance.py <hf-repo> <pack-dir> <out.json> [revision]
"""
import json
import os
import sys
import time

MODEL = sys.argv[1]
PACK = os.path.expanduser(sys.argv[2])  # "-" = no pack (stock arm)
OUT = sys.argv[3]
REV = next((a for a in sys.argv[4:] if a != "--stock"), None)
STOCK = "--stock" in sys.argv

os.environ.setdefault("DRINKME_SPEC", "off")            # A/B purity: serial loop
os.environ.setdefault("DRINKME_PREFIX_SLOTS", "0")

from drinkme.serving.engine import GenerationRequest, SampleParams, complete  # noqa: E402

GREEDY = [
    "Write a detailed, multi-paragraph explanation of why the sky is blue.",
    "Write a Python function that parses an ISO-8601 timestamp string into a "
    "datetime, handling optional fractional seconds and timezone offsets.",
    "Count from 1 to 30, one number per line, no other text.",
]
PENALIZED = [  # long enough history that PenaltyState's windows matter
    "Write a rambling story about a lighthouse keeper. Keep going.",
    "List as many animals as you can, comma-separated.",
]
SEEDED = "Describe an imaginary city in vivid detail."
SCHEMAS = [
    {"type": "object",
     "properties": {"name": {"type": "string"},
                    "population": {"type": "integer"},
                    "landmarks": {"type": "array", "items": {"type": "string"}}},
     "required": ["name", "population", "landmarks"]},
    {"type": "object",
     "properties": {"answer": {"type": "number"},
                    "reasoning": {"type": "string"}},
     "required": ["answer", "reasoning"]},
]


def run(engine, prompt, p):
    deltas = []
    t0 = time.monotonic()
    tf = [None]

    def on_delta(s):
        if tf[0] is None:
            tf[0] = time.monotonic()
        deltas.append(s)
        return True

    r = complete(engine, GenerationRequest([{"role": "user", "content": prompt}], p), on_delta)
    wall = time.monotonic() - t0
    ttft = (tf[0] - t0) if tf[0] else wall
    dec = ((r.completion_tokens - 1) / (wall - ttft)
           if wall > ttft and r.completion_tokens > 1 else 0.0)
    return {"text": r.text, "finish": r.finish_reason,
            "tokens": r.completion_tokens, "wall_s": round(wall, 2),
            "ttft_s": round(ttft, 2), "decode_toks_per_s": round(dec, 2)}


def main():
    arm = ("stock" if STOCK else "compressed") + (
        "/hotloop-OFF" if os.environ.get("DRINKME_HOTLOOP_OFF") == "1" else "/hotloop-ON")
    spec = os.environ.get("DRINKME_SPEC", "off")
    if spec != "off":
        arm += f"/{spec}-{os.environ.get('DRINKME_MTP_DEPTH', 'auto')}"
    from drinkme.serve import build_engine
    # build_engine does NOT resolve a None pack_path (the CLI does that) —
    # mtp_gpu_acceptance.py's first run died on exactly this. Explicit path.
    engine = build_engine(MODEL, REV, None if STOCK else PACK,
                          stock=STOCK, ctx=16384)
    engine.template_kwargs = {"enable_thinking": False}
    dev = str(next(engine.model.parameters()).device)
    print(f"[arm {arm}] device={dev}", flush=True)
    if not dev.startswith("cuda"):
        print("VERDICT: REFUSING — engine is not on the GPU (CPU-day class)", flush=True)
        os._exit(2)

    out = {"arm": arm, "model": MODEL, "device": dev, "cells": {}}
    cells = out["cells"]
    for i, q in enumerate(GREEDY):
        cells[f"greedy-{i}"] = run(engine, q, SampleParams(
            temperature=0.0, max_tokens=256))
        print(f"[arm {arm}] greedy-{i} done "
              f"({cells[f'greedy-{i}']['tokens']} toks)", flush=True)
    for i, q in enumerate(PENALIZED):
        cells[f"penal-{i}"] = run(engine, q, SampleParams(
            temperature=0.0, max_tokens=384, repetition_penalty=1.3,
            presence_penalty=0.6, frequency_penalty=0.4))
        print(f"[arm {arm}] penal-{i} done", flush=True)
    cells["seeded"] = run(engine, SEEDED, SampleParams(
        temperature=0.8, top_p=0.95, max_tokens=256, seed=1151,
        repetition_penalty=1.1, presence_penalty=0.2, frequency_penalty=0.1))
    print(f"[arm {arm}] seeded done", flush=True)
    for i, s in enumerate(SCHEMAS):
        cells[f"json-{i}"] = run(engine, "Describe Hilo, Hawaii.", SampleParams(
            temperature=0.0, max_tokens=512, output_schema=s))
        # the receipt should also say the constrained output PARSES
        try:
            json.loads(cells[f"json-{i}"]["text"])
            cells[f"json-{i}"]["valid_json"] = True
        except Exception:
            cells[f"json-{i}"]["valid_json"] = False
        print(f"[arm {arm}] json-{i} done valid={cells[f'json-{i}']['valid_json']}",
              flush=True)

    with open(OUT, "w") as f:
        json.dump(out, f, indent=1)
    print(f"[arm {arm}] receipt -> {OUT}", flush=True)
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
