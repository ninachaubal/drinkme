#!/usr/bin/env python3
"""llama.cpp llama-server /completion bench — replicates AMD's published
llama.cpp number.

AMD's published config (blog footnote SHO-79): llama.cpp, Vulkan backend,
MTP=4, "average token generation throughput over three or more runs",
Ryzen AI Max+ 395 128GB (VGM 64GB), Windows. We replicate on Linux/RADV
and label the OS/driver difference in the receipt.

Uses llama-server's native /completion endpoint: its `timings` object
reports prompt_per_second / predicted_per_second directly, and
cache_prompt:false makes repeat runs honest (no exact-prompt slot reuse —
the artifact that inflates Ollama's repeat-run prefill).

Usage: python3 bench/llamacpp_bench.py [http://localhost:8899] [label]
"""
import json
import os
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ollama_bench import SHORT_PROMPT, DEEP_PROMPT  # same workloads, same denominators

HOST = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8899"
LABEL = sys.argv[2] if len(sys.argv) > 2 else "unlabeled"
N_RUNS = 3
N_PREDICT = 256


def run(prompt, n=N_PREDICT):
    body = json.dumps({"prompt": prompt, "n_predict": n, "temperature": 0,
                       "cache_prompt": False}).encode()
    req = urllib.request.Request(f"{HOST}/completion", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=3600) as r:
        d = json.load(r)
    t = d.get("timings", {})
    return {"prefill_tok": t.get("prompt_n"),
            "prefill_toks_per_s": round(t.get("prompt_per_second") or 0, 2),
            "decode_tok": t.get("predicted_n"),
            "decode_toks_per_s": round(t.get("predicted_per_second") or 0, 3),
            "draft_n": t.get("draft_n"), "draft_accepted": t.get("draft_n_accepted")}


def main():
    with urllib.request.urlopen(f"{HOST}/props", timeout=30) as r:
        props = json.load(r)
    rec = {"label": LABEL, "host": HOST,
           "build": props.get("build_info"),
           "model": (props.get("model_path") or "").split("/")[-1],
           "runs": {}}
    print(f"build {rec['build']} model {rec['model']}")
    for name, prompt in (("short", SHORT_PROMPT), ("deep", DEEP_PROMPT)):
        runs = []
        for i in range(N_RUNS):
            s = run(prompt)
            runs.append(s)
            print(f"{name} run {i+1}: {s}")
        rec["runs"][name] = runs
        dec = sorted(r["decode_toks_per_s"] for r in runs)
        rec["runs"][name + "_decode_median"] = dec[len(dec) // 2]
        rec["runs"][name + "_decode_avg"] = round(sum(dec) / len(dec), 2)  # AMD's stat
        print(f"{name}: median {dec[len(dec)//2]} avg {rec['runs'][name+'_decode_avg']} tok/s")
    print("=== RECEIPT JSON ===")
    print(json.dumps(rec, indent=1))


if __name__ == "__main__":
    main()
