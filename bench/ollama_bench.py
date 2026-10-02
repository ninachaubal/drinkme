#!/usr/bin/env python3
"""Batch-1 decode benchmark against a local Ollama server.

Measures what the COMMUNITY stack gets on this machine, drinkme
deliberately excluded: a comparison outside drinkme's own stock arm.

Method: /api/generate, temperature 0, fixed prompts. First call per model
is the load (reported separately, not a decode point). Then N timed runs
at a short prompt and a ~2k-token prompt. tok/s = eval_count /
eval_duration (Ollama reports both, monotonic ns). Receipts printed as
JSON at the end — pipe to a file in verification/ to keep them.

Usage:
  python3 bench/ollama_bench.py qwen3.8:27b-q4_K_M [more models...]
  OLLAMA_HOST=http://localhost:11434 python3 bench/ollama_bench.py ...
"""
import json
import os
import sys
import time
import urllib.request

HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
if not HOST.startswith("http"):
    HOST = "http://" + HOST
N_RUNS = 3
NUM_PREDICT = 256

SHORT_PROMPT = "Write a detailed, multi-paragraph explanation of why the sky is blue."
# ~2k tokens of neutral filler + a question, to measure decode at depth + prefill
DEEP_FILLER = ("The history of measurement is a history of agreeing on rulers. " * 220)
DEEP_PROMPT = DEEP_FILLER + "\n\nSummarize the above in one paragraph, then explain what makes a good benchmark."


def generate(model, prompt, num_predict=NUM_PREDICT, keep_alive="10m"):
    body = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
        "keep_alive": keep_alive,
        "options": {"temperature": 0, "num_predict": num_predict},
    }).encode()
    req = urllib.request.Request(f"{HOST}/api/generate", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=3600) as r:
        d = json.load(r)
    d["_wall_s"] = time.monotonic() - t0
    return d


def stats(d):
    out = {}
    if d.get("prompt_eval_count") and d.get("prompt_eval_duration"):
        out["prefill_tok"] = d["prompt_eval_count"]
        out["prefill_toks_per_s"] = round(d["prompt_eval_count"] / d["prompt_eval_duration"] * 1e9, 2)
    if d.get("eval_count") and d.get("eval_duration"):
        out["decode_tok"] = d["eval_count"]
        out["decode_toks_per_s"] = round(d["eval_count"] / d["eval_duration"] * 1e9, 3)
    out["wall_s"] = round(d["_wall_s"], 2)
    return out


def bench_model(model):
    rec = {"model": model, "host": HOST, "runs": {}}
    print(f"\n=== {model} ===")
    # load (not a decode point — includes model load + warmup prefill)
    d = generate(model, "Hello.", num_predict=8)
    rec["load"] = {"total_s": round(d.get("total_duration", 0) / 1e9, 1),
                   "load_s": round(d.get("load_duration", 0) / 1e9, 1)}
    print(f"load: {rec['load']}")
    for name, prompt in (("short", SHORT_PROMPT), ("deep", DEEP_PROMPT)):
        runs = []
        for i in range(N_RUNS):
            d = generate(model, prompt)
            s = stats(d)
            runs.append(s)
            print(f"{name} run {i+1}: {s}")
        rec["runs"][name] = runs
        dec = [r["decode_toks_per_s"] for r in runs if "decode_toks_per_s" in r]
        if dec:
            rec["runs"][name + "_decode_median"] = sorted(dec)[len(dec) // 2]
            print(f"{name} decode median: {rec['runs'][name + '_decode_median']} tok/s")
    return rec


def main():
    models = sys.argv[1:]
    if not models:
        print(__doc__)
        sys.exit(2)
    # context: what the server says it is
    try:
        with urllib.request.urlopen(f"{HOST}/api/version", timeout=10) as r:
            ver = json.load(r)
    except Exception as e:
        print(f"cannot reach ollama at {HOST}: {e}")
        sys.exit(1)
    report = {"ollama_version": ver.get("version"),
              "ts_note": "stamp externally; box: Strix Halo 395 128GB, OLLAMA_VULKAN=1 OLLAMA_IGPU_ENABLE=1",
              "n_runs": N_RUNS, "num_predict": NUM_PREDICT,
              "models": []}
    print(f"ollama {ver.get('version')} at {HOST}")
    for m in models:
        try:
            report["models"].append(bench_model(m))
        except Exception as e:
            report["models"].append({"model": m, "error": str(e)})
            print(f"{m}: ERROR {e}")
    print("\n=== RECEIPT JSON ===")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
