"""One prompt against a running `drinkme serve`, N sampled runs, the MTP
adaptive bail's accounting per run — the instrument the bail-trigger sweep
(bench/bail_sweep.sh) drives.

bench/serve_drive.py is the served ruler of record: one FIXED prompt at
temperature 0, where the head is accepted ~100% and the bail never trips.
This is the other ruler — a FREE-TEXT prompt at the server's own sampling
defaults (no temperature sent, so generation_config.json's), where trip
incidence is a property of the run, not the prompt (at the default bail,
70% of ordinary 256-token answers tripped on Qwen3.8-27B, gfx1151). So it
reports per run, never only a
median: which runs tripped, where, and what it cost.

Per run, streamed over /v1/chat/completions:
  * proposed / accepted / generated off the server's /metrics deltas (the
    same accounting as serve_drive.py); steps = generated - accepted, so
    tok/step = generated / steps (one verify pass or one serial forward
    per step; the prefill's own token counts as a step);
  * trips and re-arms off the same deltas (drinkme_mtp_bail_trips_total,
    drinkme_mtp_rearms_total);
  * TTFT (first content delta), decode tok/s = generated / (wall - TTFT)
    (the receipts' definition, kept so the numbers compare), and the
    inter-token time (TBT) distribution off the SSE stream: p50 / p95 /
    max of the gaps between consecutive content deltas. Tokens a cycle
    decided together arrive together, so TBT is bimodal by construction —
    p50 says how a token usually lands, p95 what a step costs;
  * with --server-log, the [drinkme.spec] summary line and every
    `adaptive bail at +N` / `re-armed at +N` line the server appended
    during the run (generated positions: the like-for-like comparison is
    "tripped with >= 64 tokens left", which needs +N).

    PYTHONPATH=src .venv/bin/python bench/mtp_ask.py --port 3299 \\
        --model Qwen/Qwen3.8-27B --runs 10 --max-tokens 256 \\
        --prompt "How do I kill a Python process that is hogging my GPU?" \\
        --server-log verification/bail-sweep/W32-F0.15-R64/serve.log \\
        --out verification/bail-sweep/W32-F0.15-R64/P1.json

One untimed 32-token warm-up first (--no-warmup skips it), one JSON line
per run, then a SUMMARY line; the verdict is in the output, not in $?
(AGENTS.md: TheRock's exit handler can mask a failure). It refuses a
server whose /health device is the CPU (--allow-cpu overrides): a CPU
serve passes every check and measures nothing.
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import os
import re
import statistics as st
import time

METRIC_KEYS = ("drinkme_spec_proposed_tokens_total", "drinkme_spec_accepted_tokens_total",
               "drinkme_generated_tokens_total", "drinkme_requests_total",
               "drinkme_mtp_bail_trips_total", "drinkme_mtp_rearms_total")

# the server's own lines (serving/mtp.py trip / rearm, engines.py summary)
TRIP_RE = re.compile(r"\[drinkme\.mtp\] adaptive bail at \+(\d+): acceptance (\d+)% over the last (\d+) "
                     r"draft positions \(floor (\d+)%\) — (.*)")
REARM_RE = re.compile(r"\[drinkme\.mtp\] re-armed at \+(\d+) after (\d+) serial tokens \(trip (\d+)\)")
SPEC_RE = re.compile(r"\[drinkme\.spec\] (\w[\w+]*): (\d+) tokens in (\d+)( sampled)? cycles, (\d+)/(\d+) "
                     r"(?:drafted · (\d+)/(\d+) decided \((\d+)%, expect (\d+)%\)|drafts? accepted)[^\n]*")


def metrics(port: int) -> dict:
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
    c.request("GET", "/metrics")
    body = c.getresponse().read().decode()
    c.close()
    out: dict[str, float] = {}
    for line in body.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        name, _, val = line.rpartition(" ")
        key = name.split("{")[0]
        try:
            out[key] = out.get(key, 0.0) + float(val)
        except ValueError:
            pass
    return {k: out.get(k) for k in METRIC_KEYS}


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


def pct(xs: list[float], q: float) -> float | None:
    """Linear-interpolated percentile of xs (q in 0..1)."""
    if not xs:
        return None
    s = sorted(xs)
    if len(s) == 1:
        return s[0]
    pos = q * (len(s) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def ask(port: int, model: str, prompt: str, max_tokens: int) -> dict:
    """One streamed request at the server's sampling defaults. Returns the
    timing and the text, or {"error": ...}."""
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": prompt}],
                       "max_tokens": max_tokens, "stream": True,
                       "stream_options": {"include_usage": True}})
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=900)
    t0 = time.perf_counter()
    c.request("POST", "/v1/chat/completions", body=body,
              headers={"content-type": "application/json"})
    r = c.getresponse()
    if r.status != 200:
        raw = r.read().decode(errors="replace")[:600]
        c.close()
        return {"error": f"HTTP {r.status}: {raw}"}
    stamps: list[float] = []  # one per content/reasoning delta, at chunk arrival
    text: list[str] = []
    finish = None
    usage = None
    buf = b""
    done = False
    while not done:
        chunk = r.read1(65536)
        if not chunk:
            break
        now = time.perf_counter()
        buf += chunk
        while b"\n" in buf:
            line, buf = buf.split(b"\n", 1)
            line = line.strip()
            if not line.startswith(b"data:"):
                continue
            payload = line[5:].strip()
            if payload == b"[DONE]":
                done = True
                break
            try:
                d = json.loads(payload)
            except ValueError:
                continue
            if d.get("usage"):
                usage = d["usage"]
            for ch in d.get("choices") or []:
                finish = ch.get("finish_reason") or finish
                delta = ch.get("delta") or {}
                piece = delta.get("content") or delta.get("reasoning_content") or ""
                if piece:
                    stamps.append(now)
                    text.append(piece)
    t1 = time.perf_counter()
    c.close()
    full = "".join(text)
    first = stamps[0] if stamps else t1
    gaps = [b - a for a, b in zip(stamps, stamps[1:])]
    u = usage or {}
    return {"wall_s": round(t1 - t0, 3), "ttft_s": round(first - t0, 3),
            "sse_deltas": len(stamps), "finish_reason": finish,
            "completion_tokens": u.get("completion_tokens"),
            "prompt_tokens": u.get("prompt_tokens"),
            "tbt_n": len(gaps),
            "tbt_p50_s": None if not gaps else round(pct(gaps, 0.50), 4),
            "tbt_p95_s": None if not gaps else round(pct(gaps, 0.95), 4),
            "tbt_max_s": None if not gaps else round(max(gaps), 4),
            "text_sha": hashlib.sha256(full.encode()).hexdigest()[:16],
            "text_head": full[:200]}


def log_since(path: str | None, offset: int, expect_tokens: int | None = None,
              wait_s: float = 3.0) -> dict | None:
    """The server log's bail/re-arm/summary lines for ONE request: the
    [drinkme.spec] line appended since `offset` whose token count is
    `expect_tokens` (the run's generated count off /metrics; None = the
    last one), and the trip / re-arm lines between the previous request's
    summary and it. The summary is printed as the generation ends, which
    can land a moment after the stream's [DONE] — and the previous
    request's (the warm-up's, say) can land a moment after `offset` was
    taken — so the run's line is found by its count, and polled for."""
    if not path:
        return None
    deadline = time.perf_counter() + wait_s
    while True:
        try:
            with open(path, "rb") as f:
                f.seek(offset)
                tail = f.read().decode(errors="replace")
        except OSError:
            return None
        specs = list(SPEC_RE.finditer(tail))
        mine = [m for m in specs if expect_tokens is None or int(m.group(2)) == int(expect_tokens)]
        if mine or time.perf_counter() > deadline:
            break
        time.sleep(0.2)
    spec = mine[-1] if mine else None
    if spec is not None:
        prev = [m for m in specs if m.end() < spec.start()]
        seg = tail[prev[-1].end() if prev else 0:spec.end()]
    else:
        seg = tail
    out = {"spec_line": spec.group(0) if spec else None,
           "trips": [{"pos": int(m.group(1)), "rate": int(m.group(2)), "window": int(m.group(3)),
                      "floor": int(m.group(4)), "stretch": m.group(5)} for m in TRIP_RE.finditer(seg)],
           "rearms": [{"pos": int(m.group(1)), "after": int(m.group(2)), "trip": int(m.group(3))}
                      for m in REARM_RE.finditer(seg)]}
    if spec:
        mode, n, cyc, sampled, acc, drafted, dec_acc, dec_n, dec_pct, exp_pct = spec.groups()
        out.update(mode=mode, tokens=int(n), cycles=int(cyc), sampled=bool(sampled),
                   accepted=int(acc), drafted=int(drafted),
                   decided_accepted=None if dec_acc is None else int(dec_acc),
                   decided=None if dec_n is None else int(dec_n),
                   decided_rate=None if dec_pct is None else int(dec_pct) / 100,
                   expected_rate=None if exp_pct is None else int(exp_pct) / 100,
                   audit_ok=None if "audit" not in spec.group(0) else ("audit ok" in spec.group(0)),
                   serial_at_end="serial at end" in spec.group(0))
    return out


def log_size(path: str | None) -> int:
    if not path:
        return 0
    try:
        return os.path.getsize(path)
    except OSError:
        return 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--model", required=True, help="the model id the server answers to")
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--max-tokens", type=int, default=256)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--prompt", help="the user message")
    g.add_argument("--prompt-file", help="a file whose contents are the user message")
    ap.add_argument("--out", required=True, help="the JSON record (per-run rows + summary)")
    ap.add_argument("--label", default=None, help="free text naming the run in the JSON")
    ap.add_argument("--server-log", default=None,
                    help="the server's stderr log; per-run trip/re-arm positions and the summary line come from it")
    ap.add_argument("--warmup", action=argparse.BooleanOptionalAction, default=True,
                    help="one untimed 32-token request first (default on)")
    ap.add_argument("--allow-cpu", action="store_true", help="measure a CPU serve anyway")
    args = ap.parse_args()

    prompt = args.prompt if args.prompt is not None else open(args.prompt_file).read().strip()
    h = health(args.port)
    if h is None:
        print(f"[ask] ERROR: no /health on :{args.port}")
        return
    device = str(h.get("device", ""))
    if device.startswith("cpu") and not args.allow_cpu:
        print(f"[ask] REFUSING: /health reports device={device!r} — a CPU serve measures nothing "
              "(--allow-cpu to record it anyway)")
        return
    print(f"[ask] :{args.port} {h.get('model')} device={device} prefix_cache={h.get('prefix_cache')} "
          f"runs={args.runs} max_tokens={args.max_tokens}", flush=True)
    if args.warmup:
        w = ask(args.port, args.model, prompt, 32)
        if "error" in w:
            print("[ask] ERROR warm-up:", w["error"])
            return
        print(f"[ask] warm-up: {w['sse_deltas']} deltas in {w['wall_s']}s (not a sample)", flush=True)

    rows = []
    for i in range(args.runs):
        off = log_size(args.server_log)
        m0 = metrics(args.port)
        r = ask(args.port, args.model, prompt, args.max_tokens)
        m1 = metrics(args.port)
        if "error" in r:
            print(f"[ask] ERROR run {i}:", r["error"], flush=True)
            continue
        d = {k: (m1[k] - m0[k]) if (m1[k] is not None and m0[k] is not None) else None
             for k in METRIC_KEYS}
        gen, acc, prop = (d["drinkme_generated_tokens_total"], d["drinkme_spec_accepted_tokens_total"],
                          d["drinkme_spec_proposed_tokens_total"])
        steps = (gen - acc) if (gen is not None and acc is not None) else None
        span = r["wall_s"] - r["ttft_s"]
        trips = d["drinkme_mtp_bail_trips_total"]
        rearms = d["drinkme_mtp_rearms_total"]
        r.update({
            "metrics_delta": d,
            "generated": gen,
            "acceptance": (acc / prop) if prop else None,
            "steps": steps,
            "tok_per_step": (gen / steps) if steps else None,
            "decode_tok_s": round(gen / span, 2) if (gen and span > 0) else None,
            "trips": None if trips is None else int(trips),
            "rearms": None if rearms is None else int(rearms),
            "log": log_since(args.server_log, off, None if gen is None else int(gen)),
        })
        rows.append(r)
        print(json.dumps({"run": i, **r}), flush=True)

    def med(key):
        xs = [x[key] for x in rows if x.get(key) is not None]
        return round(st.median(xs), 4) if xs else None

    def mean(key):
        xs = [x[key] for x in rows if x.get(key) is not None]
        return round(st.mean(xs), 3) if xs else None

    tripped = [x for x in rows if x.get("trips")]
    summ = {"label": args.label, "model": args.model, "port": args.port, "prompt": prompt,
            "max_tokens": args.max_tokens, "date": time.strftime("%Y-%m-%d %H:%M %Z"),
            "health": h, "server_log": args.server_log, "n": len(rows), "runs": rows,
            "decode_tok_s_median": med("decode_tok_s"), "acceptance_median": med("acceptance"),
            "tok_per_step_median": med("tok_per_step"), "ttft_s_median": med("ttft_s"),
            "tbt_p50_s_median": med("tbt_p50_s"), "tbt_p95_s_median": med("tbt_p95_s"),
            "trip_rate": (len(tripped) / len(rows)) if rows else None,
            "trips_mean": mean("trips"), "rearms_mean": mean("rearms"),
            "acceptance_samples": [x.get("acceptance") for x in rows],
            "tok_per_step_samples": [x.get("tok_per_step") for x in rows],
            "trips_samples": [x.get("trips") for x in rows],
            "rearms_samples": [x.get("rearms") for x in rows]}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(summ, f, indent=1)
    print("SUMMARY", json.dumps({k: summ[k] for k in (
        "label", "n", "decode_tok_s_median", "acceptance_median", "tok_per_step_median", "ttft_s_median",
        "tbt_p50_s_median", "tbt_p95_s_median", "trip_rate", "trips_mean", "rearms_mean",
        "tok_per_step_samples", "trips_samples")}), flush=True)
    f = lambda k, p=2: "—" if summ.get(k) is None else f"{summ[k]:.{p}f}"  # noqa: E731
    print(f"[ask] line: tok/step {f('tok_per_step_median')} · tripped {len(tripped)}/{len(rows)} "
          f"· trips/run {f('trips_mean', 1)} · re-arms/run {f('rearms_mean', 1)} "
          f"· tok/s {f('decode_tok_s_median')} · TBT p50/p95 {f('tbt_p50_s_median', 3)}/"
          f"{f('tbt_p95_s_median', 3)}s")
    if len(rows) < args.runs:
        print(f"[ask] VERDICT: INCOMPLETE — {len(rows)}/{args.runs} runs recorded")
    else:
        print(f"[ask] VERDICT: OK — {len(rows)}/{args.runs} runs recorded to {args.out}")


if __name__ == "__main__":
    main()
