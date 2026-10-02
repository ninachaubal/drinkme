#!/usr/bin/env python3
"""Rejection-sampling MTP acceptance on real weights: first-token
distribution over many seeds, one arm per server.

The claim (serving/speculative.py): at temperature > 0 the token MTP emits
is distributed exactly as the serial sampler's. On the toy that is checked
against the known p (tests/test_serving_speculative.py). On the real model
there is no known p, and a seed-matched MTP transcript is NOT the
seed-matched serial transcript (the generator is consumed differently) — so
the check is statistical: the same N seeds, max_tokens=1, against a server
running MTP and against one running serial (DRINKME_SPEC=off), and the two
empirical first-token histograms compared by total variation. Also records
acceptance by workload from a handful of longer sampled requests (the
server's own [drinkme.spec] accounting line is the source for the rate; this
script only records what came back on the wire).

Usage:
  python bench/mtp_sampled_dist.py --base http://127.0.0.1:3298 --arm mtp --n 400 --out /tmp/arm-mtp.json
  python bench/mtp_sampled_dist.py --base http://127.0.0.1:3299 --arm serial --n 400 --out /tmp/arm-serial.json
  python bench/mtp_sampled_dist.py --compare /tmp/arm-mtp.json /tmp/arm-serial.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
import urllib.request
from collections import Counter

PROMPTS = {
    "answer": "In one word, what color is the sky on a clear day?",
    "story": "Begin a short story with a single vivid sentence.",
    "code": "Write a Python function that returns the nth Fibonacci number.",
}
WORKLOADS = {
    "prose": "Write a paragraph about a lighthouse keeper's morning.",
    "code": "Write a Python class implementing an LRU cache with get and put.",
    "list": "List twelve unrelated nouns, one per line, no numbering.",
}


def chat(base: str, model: str, prompt: str, *, seed: int | None, temperature: float, max_tokens: int) -> dict:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if seed is not None:
        body["seed"] = seed
    req = urllib.request.Request(f"{base}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.load(r)


def served_model(base: str) -> str:
    with urllib.request.urlopen(f"{base}/v1/models", timeout=30) as r:
        return json.load(r)["data"][0]["id"]


def run_arm(base: str, arm: str, n: int, temperature: float, out: str) -> None:
    model = served_model(base)
    hist: dict[str, Counter] = {k: Counter() for k in PROMPTS}
    t0 = time.time()
    for name, prompt in PROMPTS.items():
        for seed in range(1, n + 1):
            d = chat(base, model, prompt, seed=seed, temperature=temperature, max_tokens=1)
            hist[name][d["choices"][0]["message"]["content"]] += 1
        print(f"[{arm}] {name}: {n} draws, {len(hist[name])} distinct first tokens, "
              f"top {hist[name].most_common(3)}  ({time.time() - t0:.0f}s)", file=sys.stderr)
    work = {}
    for name, prompt in WORKLOADS.items():
        d = chat(base, model, prompt, seed=1, temperature=temperature, max_tokens=200)
        work[name] = {"completion_tokens": d["usage"]["completion_tokens"],
                      "text_head": d["choices"][0]["message"]["content"][:160]}
        print(f"[{arm}] workload {name}: {work[name]['completion_tokens']} tokens", file=sys.stderr)
    json.dump({"arm": arm, "base": base, "model": model, "n": n, "temperature": temperature,
               "hist": {k: dict(v) for k, v in hist.items()}, "workloads": work,
               "wall_s": round(time.time() - t0, 1)}, open(out, "w"), indent=1)


def compare(a_path: str, b_path: str) -> None:
    a, b = json.load(open(a_path)), json.load(open(b_path))
    print(f"{a['arm']} (n={a['n']}) vs {b['arm']} (n={b['n']}), model {a['model']}, T={a['temperature']}")
    for name in a["hist"]:
        ha, hb = Counter(a["hist"][name]), Counter(b["hist"][name])
        na, nb = sum(ha.values()), sum(hb.values())
        keys = set(ha) | set(hb)
        tv = 0.5 * sum(abs(ha[k] / na - hb[k] / nb) for k in keys)
        # noise floor for TV between two independent samples of the same
        # distribution with k effective categories: roughly sqrt(k / (2 pi n))
        k_eff = len([k for k in keys if ha[k] + hb[k] >= 3])
        floor = math.sqrt(max(k_eff, 1) / (2 * math.pi * min(na, nb)))
        print(f"  {name:7s} TV={tv:.3f}  noise~{floor:.3f}  distinct {len(ha)}/{len(hb)}  "
              f"top {a['arm']}={ha.most_common(2)} {b['arm']}={hb.most_common(2)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base")
    ap.add_argument("--arm")
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--out")
    ap.add_argument("--compare", nargs=2)
    args = ap.parse_args()
    if args.compare:
        compare(*args.compare)
    else:
        run_arm(args.base, args.arm, args.n, args.temperature, args.out)
