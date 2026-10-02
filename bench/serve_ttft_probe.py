#!/usr/bin/env python3
"""A urllib streaming client against a running `drinkme serve`: decode
tok/s and time to first token, greedy, thinking off, one request at a time.

  warm-up (16 tokens, discarded), then RUNS x 128 greedy tokens on a short
  prompt (decode tok/s from the second token on, TTFT to the first content
  delta), then TTFT on longer prompts built to ~N tokens by repetition.

    python bench/serve_ttft_probe.py --port 3999 --model Qwen/Qwen3-4B [--runs 3] [--long 160,512]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request

SHORT = [
    "Write a short paragraph about the history of aqueducts.",
    "Explain what a rose is, in detail.",
    "Describe how a bicycle's gears work.",
]

FILLER = ("The town sits where the river bends, and the market opens before the light "
          "does; stalls of bread, brass and wool line the square while carts roll in from "
          "the orchards. ")


def request(base, model, prompt, max_tokens):
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": prompt}],
                       "max_tokens": max_tokens, "temperature": 0, "stream": True,
                       "chat_template_kwargs": {"enable_thinking": False}}).encode()
    req = urllib.request.Request(f"{base}/v1/chat/completions", body,
                                 {"content-type": "application/json"})
    t0 = time.time()
    first = None
    k = 0
    txt = ""
    with urllib.request.urlopen(req) as r:
        for line in r:
            if not line.startswith(b"data: ") or line.strip() == b"data: [DONE]":
                continue
            d = json.loads(line[6:])
            c = d["choices"][0]["delta"].get("content")
            if c:
                if first is None:
                    first = time.time()
                k += 1
                txt += c
    t1 = time.time()
    ttft = (first - t0) if first else float("nan")
    tps = (k - 1) / (t1 - first) if k > 1 else 0.0
    return k, ttft, tps, txt


def count_tokens(base, model, prompt):
    """/v1/messages/count_tokens (Anthropic's envelope; the lockless route):
    the prompt's input_tokens as the server itself counts them, -1 if the
    route is unavailable — informational, so the long prompts are reported
    in the server's own tokens rather than in words."""
    body = json.dumps({"model": model, "max_tokens": 1,
                       "messages": [{"role": "user", "content": prompt}]}).encode()
    req = urllib.request.Request(f"{base}/v1/messages/count_tokens", body,
                                 {"content-type": "application/json"})
    try:
        with urllib.request.urlopen(req) as r:
            return int(json.loads(r.read())["input_tokens"])
    except Exception:  # noqa: BLE001
        return -1


def long_prompt(n_words_approx):
    reps = max(1, n_words_approx // len(FILLER.split()))
    return FILLER * reps + "Summarise the passage above in one sentence."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=3999)
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--tokens", type=int, default=128)
    ap.add_argument("--long", default="160,512")
    a = ap.parse_args()
    base = f"http://localhost:{a.port}"

    with urllib.request.urlopen(f"{base}/health") as r:
        health = json.loads(r.read())
    with urllib.request.urlopen(f"{base}/v1/models") as r:
        models = json.loads(r.read())
    m0 = (models.get("data") or [{}])[0]
    d0 = m0.get("drinkme") or {}
    print(f"health: {health}")
    print(f"model: {m0.get('id')} runtime={d0.get('runtime')} computePath={d0.get('computePath')} "
          f"device={d0.get('device')} bitsPerWeight={d0.get('bitsPerWeight')} sourceDtype={d0.get('sourceDtype')}")

    k, ttft, tps, _ = request(base, a.model, "warm up", 16)
    print(f"warm-up: {k} tokens, ttft {ttft:.2f}s, {tps:.2f} tok/s (discarded)")
    for i in range(a.runs):
        p = SHORT[i % len(SHORT)]
        k, ttft, tps, txt = request(base, a.model, p, a.tokens)
        print(f"run {i + 1}: tokens={k} ttft={ttft:.2f}s decode={tps:.2f} tok/s | {txt[:60]!r}")
    for n in [int(x) for x in a.long.split(",") if x]:
        p = long_prompt(int(n * 0.75))  # ~1.3 Qwen tokens per English word
        nt = count_tokens(base, a.model, p)
        k, ttft, tps, txt = request(base, a.model, p, 32)
        k2, ttft2, _, _ = request(base, a.model, p, 32)
        print(f"long prompt ({nt} tokens by the server's count, target ~{n}): "
              f"ttft={ttft:.2f}s then {ttft2:.2f}s (repeat), decode={tps:.2f} tok/s | {txt[:50]!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
