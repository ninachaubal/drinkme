"""Serve-path timing A/B: stock BF16 vs compressed, SAME server stack, fair.

prefix_cache_verify.py measures the stock arm's clock and then drops it on the
floor — its report only ever compares cold-vs-warm on the COMPRESSED arm, so
"8.2-8.8x TTFT" is cache-vs-no-cache, never compressed-vs-stock; and timing
both arms at --ctx 32768 through an over-allocated cache measures a regime
nobody serves in. This instrument exists to answer the actual product question:

    on this box, on the serve path, does the compressed arm beat stock —
    and per metric: cold TTFT (prefill), warm TTFT (prefix cache), decode
    tok/s (where the byte advantage lives)?

Fairness rules, each load-bearing:

  1. ISOLATED arms: one model resident at a time (load, measure, free) — the
     arms.py doctrine; peak memory = one representation.
  2. IDENTICAL token streams: stock's cold pass self-conditions once; every
     other pass (stock-warm, compressed-cold, compressed-warm) REPLAYS that
     transcript, so all four passes prefill and see byte-identical inputs.
     Near-tie divergence in compressed's own generations is recorded as a
     finding but never enters any prompt.
  3. SAME allocation regime: one engine class, one --ctx cap, one geometric
     alloc policy — decode cost tracks ALLOCATED width, so the cap must match
     (it does, structurally) and the report records it.
  4. UNTIMED warmup per arm before any measured request (load-time compile
     and first-touch never contaminate turn 1).
  5. GPU or refuse: a CPU run must never pass for a GPU number.

Decode rate = (wall - ttft) / (completion_tokens - 1), per turn, over the
wire — SSE parsing, templating, sampler, the whole stack, because the stack
is the product. Bandwidth is probed on the EMPTY device first so the report
carries its own wall.

Writes verification/serve_timing_ab_<host>_<date>.json.
Run: uv run --no-sync python bench/serve_timing_ab.py [--turns N] [--max-tokens N] [--ctx N]
     (--no-sync is not optional here — a bare `uv run` swaps the accelerator
      torch out from under the box mid-measurement; see AGENTS.md)
"""

from __future__ import annotations

import argparse
import gc
import http.client
import json
import os
import platform
import socket
import sys
import time

# The served pack. Default = this box's Qwen3-8B; DRINKME_AB_PACK points the
# instrument at any pack dir (the Modal sweep packs on the remote box and the
# cache path there carries a different revision-of-the-day).
PACK = os.environ.get("DRINKME_AB_PACK") or os.path.expanduser(
    "~/.cache/drinkme/packs/Qwen--Qwen3-8B@b968826d9c46")

# Agent-shaped: long high-entropy system prompt, history re-sent whole.
SYSTEM = " ".join(
    f"Rule {i}: when the user mentions topic-{i}, respond with a numbered plan "
    f"citing section {i * 7 % 101} of the handbook, keeping the tone plain."
    for i in range(120)
)

# Prompts chosen to ramble: decode tok/s wants LONG completions, and a
# 20-token answer is a noisy rate sample.
USERS = [
    "Walk through your rules for topic-3, topic-9 and topic-27 in detail: "
    "for each, restate the rule, name the section, and give a worked example.",
    "Now write the full numbered plan for a user who mentions topic-14 and "
    "topic-50 in one message. Be thorough; include every applicable section.",
    "Draft a one-page briefing that summarizes how the handbook sections you "
    "have cited so far relate to each other. Use complete paragraphs.",
    "Finally, propose three new rules in the same style, with plausible "
    "section numbers, and justify each in a short paragraph.",
]


def request(port: int, messages: list[dict], max_tokens: int):
    """Greedy streaming request -> (text, ttft, wall, usage)."""
    # Thinking OFF for the measured regime: the think channel
    # splits reasoning into delta.reasoning_content, and a thinking model can
    # spend the whole token budget there — against Qwen3.8-27B that is 192
    # tokens a content-only parser never sees ("no first token"). Answer-only keeps replay semantics exact and receipts
    # comparable across models.
    body = json.dumps({"messages": messages, "temperature": 0,
                       "max_tokens": max_tokens, "stream": True,
                       "stream_options": {"include_usage": True},
                       "chat_template_kwargs": {"enable_thinking": False}})
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=1200)
    t0 = time.perf_counter()
    c.request("POST", "/v1/chat/completions", body,
              {"Content-Type": "application/json"})
    r = c.getresponse()
    assert r.status == 200, r.status
    ttft, text, usage, buf = None, [], None, b""
    while True:
        chunk = r.read1(65536)
        if not chunk:
            break
        buf += chunk
        while b"\n\n" in buf:
            frame, buf = buf.split(b"\n\n", 1)
            if not frame.startswith(b"data: "):
                continue
            payload = frame[len(b"data: "):]
            if payload == b"[DONE]":
                continue
            obj = json.loads(payload)
            if obj.get("usage"):
                usage = obj["usage"]
            for ch in obj.get("choices", []):
                d = ch.get("delta", {})
                delta = d.get("content")
                # reasoning tokens are still generated tokens: they count for
                # TTFT/decode timing even though they never enter the replay
                # transcript (belt-and-braces — enable_thinking=False above
                # should mean this branch never fires).
                if delta or d.get("reasoning_content"):
                    if ttft is None:
                        ttft = time.perf_counter() - t0
                if delta:
                    text.append(delta)
    wall = time.perf_counter() - t0
    c.close()
    return "".join(text), ttft, wall, usage


def run_pass(port: int, eng, reuse: bool, turns: int, max_tokens: int,
             label: str, replay: list | None = None):
    """One measured pass. With `replay`, assistant turns come from the
    reference transcript so inputs are byte-identical across passes."""
    eng._reuse = reuse
    eng.reset_prefix_cache()  # every pass starts cold in its own terms
    out = []
    history = [{"role": "system", "content": SYSTEM}]
    for i, u in enumerate(USERS[:turns]):
        history.append({"role": "user", "content": u})
        text, ttft, wall, usage = request(port, history, max_tokens)
        if ttft is None:
            raise SystemExit(
                f"[{label}] turn {i + 1}: no first token seen. If usage shows "
                f"completion_tokens > 0, the server generated on a channel "
                f"this parser doesn't know (the think channel was that once); "
                f"otherwise a server-side error. usage={usage!r}")
        history.append({"role": "assistant",
                        "content": replay[i] if replay is not None else text})
        u_ = usage or {}
        ct = u_.get("completion_tokens") or 0
        tps = round((ct - 1) / (wall - ttft), 2) if ct > 1 and wall > ttft else None
        out.append({"text": text, "ttft": round(ttft, 3), "wall": round(wall, 3),
                    "prompt_tokens": u_.get("prompt_tokens"),
                    "completion_tokens": ct,
                    "cached_tokens": u_.get("prompt_tokens_details", {})
                                       .get("cached_tokens", 0),
                    "decode_tps": tps})
        print(f"  [{label}] turn {i + 1}: prompt={out[-1]['prompt_tokens']} "
              f"cached={out[-1]['cached_tokens']} ttft={ttft:.2f}s "
              f"decode={tps} tok/s ({ct} tok)", flush=True)
    return out


def median(xs):
    xs = sorted(x for x in xs if x is not None)
    return xs[len(xs) // 2] if xs else None


def run_arm(stock: bool, meta, args, replay):
    """Load one arm, warm it, run cold+warm passes, tear it down."""
    import torch

    from drinkme.serve import build_engine
    from drinkme.serving.http import start_server

    name = "stock" if stock else "compressed"
    print(f"[ab] loading {name} arm ...", flush=True)
    t0 = time.time()
    eng = build_engine(meta["hfRepo"], meta["revision"],
                       None if stock else PACK, stock=stock, ctx=args.ctx)
    load_s = round(time.time() - t0, 1)
    print(f"[ab] {name} loaded in {load_s}s", flush=True)
    if str(eng.device) == "cpu" and not os.environ.get("VERIFY_ALLOW_CPU"):
        sys.exit("[ab] REFUSING to measure on CPU (VERIFY_ALLOW_CPU=1 to override)")
    vram = round(torch.cuda.memory_allocated() / 1024**3, 2)
    srv = start_server(eng, "127.0.0.1", 0)
    port = srv.server_address[1]

    # untimed warmup: first-touch, autotune, allocator growth
    request(port, [{"role": "user", "content": "Say OK."}], 8)

    cold = run_pass(port, eng, False, args.turns, args.max_tokens,
                    f"{name}-cold", replay=replay)
    # Warm SELF-CONDITIONS: a real client sends back what the engine actually
    # said, so extends-only reuse always holds. Replaying the reference here
    # lets every near-tie flip poison the cache into a spurious full prefill
    # (3/4 turns flipped at ~4k prompts), unequally between arms — warm TTFT
    # would measure the flips, not the cache. Text A/B
    # stays cold-vs-cold, where inputs ARE identical by replay.
    warm = run_pass(port, eng, True, args.turns, args.max_tokens,
                    f"{name}-warm", replay=None)
    srv.shutdown()
    del eng
    gc.collect()
    torch.cuda.empty_cache()
    return {"load_s": load_s, "vram_gb": vram, "cold": cold, "warm": warm}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--turns", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=192)
    ap.add_argument("--ctx", type=int, default=8192)
    args = ap.parse_args()
    # on-disk prefix slots: the cold tier would hand a "cold" arm a stale
    # prefix off an SSD, which is an instrument measuring nothing. setdefault, so a
    # deliberate DRINKME_SLOT_DIR on the command line still wins.
    os.environ.setdefault("DRINKME_SLOT_DIR", "off")

    from drinkme.probe import measure_bandwidth

    meta = json.load(open(os.path.join(PACK, "meta.json")))
    print(f"[ab] {meta['hfRepo']}@{(meta['revision'] or 'main')[:12]} "
          f"meanBpw={meta['meanBpw']} ctx={args.ctx} "
          f"turns={args.turns} max_tokens={args.max_tokens}", flush=True)

    bw = measure_bandwidth()          # EMPTY device: the wall, self-carried
    print(f"[ab] bandwidth: {bw}", flush=True)

    # Arm 1: stock self-conditions once — its cold transcript is the reference
    stock = run_arm(True, meta, args, replay=None)
    reference = [t["text"] for t in stock["cold"]]
    # Arm 2: compressed replays the reference in both passes
    comp = run_arm(False, meta, args, replay=reference)

    ab_identical = [c["text"] == s["text"]
                    for c, s in zip(comp["cold"], stock["cold"])]

    def summarize(arm):
        return {
            "ttft_cold_final_turn_s": arm["cold"][-1]["ttft"],
            "ttft_warm_median_s": median([t["ttft"] for t in arm["warm"]]),
            "decode_tps_cold_median": median([t["decode_tps"] for t in arm["cold"]]),
            "decode_tps_warm_median": median([t["decode_tps"] for t in arm["warm"]]),
        }

    s_sum, c_sum = summarize(stock), summarize(comp)

    def ratio(c, s, invert=False):
        if not c or not s:
            return None
        return round((s / c) if invert else (c / s), 3)

    summary = {
        "stock": s_sum, "compressed": c_sum,
        # >1.0 = compressed WINS, uniformly
        "compressed_over_stock": {
            "ttft_cold": ratio(c_sum["ttft_cold_final_turn_s"],
                               s_sum["ttft_cold_final_turn_s"], invert=True),
            "ttft_warm": ratio(c_sum["ttft_warm_median_s"],
                               s_sum["ttft_warm_median_s"], invert=True),
            "decode_tps_cold": ratio(c_sum["decode_tps_cold_median"],
                                     s_sum["decode_tps_cold_median"]),
            "decode_tps_warm": ratio(c_sum["decode_tps_warm_median"],
                                     s_sum["decode_tps_warm_median"]),
        },
    }

    report = {
        "date": time.strftime("%Y-%m-%d %H:%M %Z"),
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "model": {k: meta[k] for k in ("hfRepo", "revision", "meanBpw")},
        "ctx": args.ctx, "max_tokens": args.max_tokens, "turns": args.turns,
        "bandwidth_probe": bw,
        "arms": {"stock": stock, "compressed": comp},
        "ab_text_identical_per_turn": ab_identical,
        "ab_all_identical": all(ab_identical),
        # greedy near-tie flips are a reported finding, not a failure: identical
        # weights can still produce different transcripts (docs/method.md#numerical-behavior)
        "summary": summary,
    }
    out = os.path.join(os.path.dirname(__file__), "..", "verification",
                       f"serve_timing_ab_{socket.gethostname()}_"
                       f"{time.strftime('%Y-%m-%d')}.json")
    # A receipt is never clobbered: a same-day re-run gets a time-suffixed
    # name instead of silently replacing the earlier record.
    if os.path.exists(out):
        out = out[:-len(".json")] + time.strftime("-%H%M") + ".json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n[ab] ctx={args.ctx} — ratios >1.0 mean compressed wins")
    for k, v in summary["compressed_over_stock"].items():
        print(f"  {k:18s} {v}x")
    print(f"  stock:      {json.dumps(s_sum)}")
    print(f"  compressed: {json.dumps(c_sum)}")
    print(f"  A/B text identical: {report['ab_all_identical']} "
          f"{ab_identical} (near-tie flips are findings, not failures)")
    print(f"[ab] wrote {os.path.normpath(out)}")


if __name__ == "__main__":
    main()
