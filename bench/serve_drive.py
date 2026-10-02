"""Served-throughput driver: one fixed prompt against a running `drinkme
serve`, speculation on or off, decode tok/s and TTFT per run.

bench/serve_timing_ab.py is a stock-vs-compressed A/B over a 4-turn
agent-shaped conversation with its OWN in-process server, so it does not
fit "an external server, one fixed prompt, speculation on and off". This
driver keeps its request() shape (SSE parse, TTFT = first content delta,
thinking off) and points it at a server that is already up. The Modal
drivers (bench/modal_radix.py, bench/modal_nvidia2.py) run it on the
container against the served pair of every cell.

Per invocation: one fixed ~70-token prompt (64 raw / 76 templated tokens),
128 decode tokens, temperature 0, streamed over /v1/chat/completions,
1 untimed warm-up + N timed runs. Decode tok/s = (completion_tokens - 1) /
(t_last_delta - t_first_delta) — TTFT excluded, reported separately.
Speculation accounting comes from the server's own /metrics deltas
(drinkme_spec_proposed/accepted_tokens_total, generated_tokens_total) per
run and from the [drinkme.spec] line the server prints per request
(`N tokens in C cycles` -> tokens/step = N/C), parsed off the server log.
It refuses a server whose /health device is the CPU.

    drinkme serve --pack-dir <sip pack> --port 3299 > serve.log 2>&1 &
    PYTHONPATH=src .venv/bin/python bench/serve_drive.py --port 3299 \
        --label sip_spec-on --server-log serve.log \
        --out verification/serve/sip_spec-on.json

Speculation off is the server's `--spec off` (DRINKME_SPEC=off); the label is free text
that names the run in the JSON.

A served number is taken against the server AS SHIPPED — prefix cache on,
default slot count, --ctx stated; a run that switches the cache off, or any
other serving default, says so in its label (docs/bench.md). By default
this driver refuses to write a run's label plain when /health reports the
prefix cache off; --allow-cache-off overrides that and marks the label itself.
"""

from __future__ import annotations

import argparse
import http.client
import json
import os
import re
import socket
import time

PROMPT = ("Repeat the following sentence exactly six times, each on its own numbered line, and then in "
          "one short sentence say why repetition helps memory: 'The quick brown fox jumps over the lazy "
          "dog near the quiet river bank at dawn while the old grey owl watches from the tall oak tree "
          "and the small red boat drifts by.'")

SPEC_RE = re.compile(r"\[drinkme\.spec\] (\w+): (\d+) tokens in (\d+)(?: sampled)? cycles, (\d+)/(\d+) drafts? accepted")


def request(port: int, max_tokens: int, prompt: str = PROMPT):
    """Greedy streaming request -> dict(text, ttft, wall, usage, t_first, t_last, deltas)."""
    body = json.dumps({"messages": [{"role": "user", "content": prompt}], "temperature": 0,
                       "max_tokens": max_tokens, "stream": True,
                       "stream_options": {"include_usage": True},
                       "chat_template_kwargs": {"enable_thinking": False}})
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=600)
    t0 = time.perf_counter()
    c.request("POST", "/v1/chat/completions", body, {"Content-Type": "application/json"})
    r = c.getresponse()
    assert r.status == 200, (r.status, r.read()[:400])
    t_first = t_last = None
    text, usage, buf, deltas = [], None, b"", 0
    while True:
        chunk = r.read1(65536)
        if not chunk:
            break
        now = time.perf_counter()
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
                if delta or d.get("reasoning_content"):
                    if t_first is None:
                        t_first = now
                    t_last = now
                    deltas += 1
                if delta:
                    text.append(delta)
    wall = time.perf_counter() - t0
    c.close()
    u = usage or {}
    ct = int(u.get("completion_tokens") or 0)
    ttft = None if t_first is None else t_first - t0
    tps = ((ct - 1) / (t_last - t_first)) if (ct > 1 and t_last and t_first and t_last > t_first) else None
    return {"text": "".join(text), "ttft_s": ttft, "wall_s": wall, "completion_tokens": ct,
            "prompt_tokens": u.get("prompt_tokens"), "decode_tok_s": tps, "sse_deltas": deltas,
            "decode_span_s": None if tps is None else (t_last - t_first)}


def metrics(port: int) -> dict:
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
    c.request("GET", "/metrics")
    r = c.getresponse()
    body = r.read().decode()
    c.close()
    out = {}
    for line in body.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        name, _, val = line.rpartition(" ")
        key = name.split("{")[0]
        try:
            out[key] = out.get(key, 0.0) + float(val)
        except ValueError:
            pass
    return out


def health(port: int) -> dict | None:
    try:
        c = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        c.request("GET", "/health")
        r = c.getresponse()
        body = json.loads(r.read().decode() or "{}")
        c.close()
        return body if r.status == 200 else None
    except (OSError, ValueError):
        return None


def spec_lines(log_path: str, start_offset: int) -> list[dict]:
    """The [drinkme.spec] audit lines appended to the server log since
    start_offset -> [{mode, tokens, cycles, accepted, drafted}]."""
    if not log_path or not os.path.exists(log_path):
        return []
    with open(log_path, "rb") as f:
        f.seek(start_offset)
        tail = f.read().decode(errors="replace")
    out = []
    for m in SPEC_RE.finditer(tail):
        mode, n, cyc, acc, drafted = m.groups()
        out.append({"mode": mode, "tokens": int(n), "cycles": int(cyc), "accepted": int(acc),
                    "drafted": int(drafted), "tokens_per_step": round(int(n) / int(cyc), 4) if int(cyc) else None,
                    "line": m.group(0)})
    return out


def check_shipped(h: dict, label: str, require_shipped: bool, allow_cache_off: bool) -> str:
    """A served number is taken against the server AS SHIPPED (docs/bench.md);
    refuse to record one against a server whose /health reports the prefix
    cache off unless --allow-cache-off was given, in which case the label
    says so itself. `prefix_cache.slots` absent (an older server, or an
    engine with no concept of one) is not evidence the cache is off, so it
    passes unmarked."""
    if not require_shipped:
        return label
    slots = (h.get("prefix_cache") or {}).get("slots")
    if slots != 0:
        return label
    if not allow_cache_off:
        raise SystemExit(
            "[drive] REFUSING: /health reports prefix_cache.slots=0 — this server "
            "did not ship with the prefix cache on, so this is not a served number "
            "as shipped. Pass --allow-cache-off to record it anyway.")
    return f"{label}-cache-off"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--label", required=True,
                    help="Free text naming the run in the JSON. A served number is "
                         "taken against the server AS SHIPPED (prefix cache on, "
                         "default slot count, --ctx stated); a label for a run that "
                         "switches off the cache or any other serving default must "
                         "say so (docs/bench.md).")
    ap.add_argument("--out", required=True)
    ap.add_argument("--server-log", default=None)
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--wait", type=int, default=900, help="seconds to wait for /health")
    ap.add_argument("--require-shipped", action=argparse.BooleanOptionalAction, default=True,
                    help="Refuse to record a served number when /health reports the "
                         "prefix cache off (default on; --no-require-shipped skips "
                         "the check).")
    ap.add_argument("--allow-cache-off", action="store_true",
                    help="Let --require-shipped pass when the prefix cache is off; "
                         "appends -cache-off to --label instead of refusing.")
    a = ap.parse_args()

    t0 = time.time()
    h = None
    while time.time() - t0 < a.wait:
        h = health(a.port)
        if h and h.get("state", "ready") in ("ready", "awake") and h.get("device"):
            break
        time.sleep(2)
    if not h:
        raise SystemExit(f"[drive] no /health on port {a.port} after {a.wait}s")
    a.label = check_shipped(h, a.label, a.require_shipped, a.allow_cache_off)
    print(f"[drive] {a.label}: /health {json.dumps(h)} (after {time.time() - t0:.0f}s)", flush=True)
    if str(h.get("device", "")).startswith("cpu"):
        raise SystemExit("[drive] REFUSING: the server is on the CPU")

    def _log_size():
        return os.path.getsize(a.server_log) if a.server_log and os.path.exists(a.server_log) else 0

    runs = []
    # warm-up (untimed): compile, first-touch, allocator growth
    off = _log_size()
    m0 = metrics(a.port)
    w = request(a.port, a.max_tokens)
    time.sleep(0.5)
    warm = {**w, "metrics_delta": _delta(m0, metrics(a.port)), "spec": spec_lines(a.server_log, off)}
    print(f"[drive] warm-up: {w['completion_tokens']} tok, ttft {w['ttft_s']:.3f}s, "
          f"decode {w['decode_tok_s']:.2f} tok/s, spec {[s['line'] for s in warm['spec']]}", flush=True)
    for i in range(a.runs):
        off = _log_size()
        m0 = metrics(a.port)
        r = request(a.port, a.max_tokens)
        time.sleep(0.5)  # let the server flush its audit line
        r["metrics_delta"] = _delta(m0, metrics(a.port))
        r["spec"] = spec_lines(a.server_log, off)
        runs.append(r)
        sp = r["spec"][-1] if r["spec"] else None
        print(f"[drive] run {i + 1}: {r['completion_tokens']} tok, ttft {r['ttft_s']:.3f}s, "
              f"decode {r['decode_tok_s']:.2f} tok/s ({r['sse_deltas']} deltas), "
              f"{'no spec line' if sp is None else sp['line']}", flush=True)

    tps = sorted(x["decode_tok_s"] for x in runs if x["decode_tok_s"])
    ttft = sorted(x["ttft_s"] for x in runs if x["ttft_s"])
    tokens = [x["completion_tokens"] for x in runs]
    steps = [x["spec"][-1]["tokens_per_step"] for x in runs if x["spec"]]
    acc = [x["spec"][-1]["accepted"] / x["spec"][-1]["drafted"] for x in runs
           if x["spec"] and x["spec"][-1]["drafted"]]
    md = {"proposed": sum(x["metrics_delta"].get("drinkme_spec_proposed_tokens_total", 0) for x in runs),
          "accepted": sum(x["metrics_delta"].get("drinkme_spec_accepted_tokens_total", 0) for x in runs),
          "generated": sum(x["metrics_delta"].get("drinkme_generated_tokens_total", 0) for x in runs)}
    texts = {x["text"] for x in runs}
    summary = {
        "decode_tok_s_median": tps[len(tps) // 2] if tps else None,
        "decode_tok_s_samples": [round(x, 3) for x in tps],
        "ttft_s_median": ttft[len(ttft) // 2] if ttft else None,
        "ttft_s_samples": [round(x, 4) for x in ttft],
        "completion_tokens": tokens,
        "tokens_per_step_samples": steps,
        "tokens_per_step_median": sorted(steps)[len(steps) // 2] if steps else None,
        "acceptance_samples": [round(x, 4) for x in acc],
        "metrics_totals_over_runs": md,
        "metrics_acceptance": (md["accepted"] / md["proposed"]) if md["proposed"] else None,
        "text_identical_across_runs": len(texts) == 1,
        "text_sha_first_run": __import__("hashlib").sha256(runs[0]["text"].encode()).hexdigest()[:16],
    }
    report = {"label": a.label, "date": time.strftime("%Y-%m-%d %H:%M %Z"), "host": socket.gethostname(),
              "port": a.port, "health": h, "prompt": PROMPT, "max_tokens": a.max_tokens,
              "runs_requested": a.runs, "warmup": warm, "runs": runs, "summary": summary,
              "env": {k: os.environ.get(k) for k in ("DRINKME_MTP_DEPTH", "DRINKME_SPEC", "DRINKME_PREFIX_SLOTS", "DRINKME_RADIX_SCHEDULE",
                                                      "OMP_NUM_THREADS")}}
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(report, f, indent=1)
    print(f"[drive] {a.label}: decode median {summary['decode_tok_s_median']} tok/s {summary['decode_tok_s_samples']}"
          f" · TTFT median {summary['ttft_s_median']} · tokens/step {summary['tokens_per_step_samples']}"
          f" · text identical across runs {summary['text_identical_across_runs']} -> {a.out}", flush=True)


def _delta(m0: dict, m1: dict) -> dict:
    return {k: m1.get(k, 0.0) - m0.get(k, 0.0) for k in
            ("drinkme_spec_proposed_tokens_total", "drinkme_spec_accepted_tokens_total",
             "drinkme_generated_tokens_total", "drinkme_requests_total")}


if __name__ == "__main__":
    main()
