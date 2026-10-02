"""Render bench/bail_sweep.sh's records as Markdown: the adaptive bail's
trigger configurations side by side, per prompt.

Reads <out-dir>/<config-dir>/<prompt>.json (bench/mtp_ask.py's records;
config-dir is W<window>-F<floor>-R<rearm>) and prints three tables:

  1. per config x prompt: n, tok/step median / mean / range, trip rate
     (runs that tripped / n), mean trips and re-arms per run, decode tok/s
     median, TBT p50 / p95 (medians over runs of each run's percentile);
  2. the LIKE-FOR-LIKE rows (trip incidence is random per
     sampled run and identical code on every arm up to the first trip, so
     the all-runs median is not the comparison): per config x prompt, the
     runs that never tripped, the runs that tripped with >= 64 tokens left
     (first trip at or before max_tokens - 65, where a re-arm's cost and a
     latch's cost both fit inside the run), and the late trips;
  3. every run: tok/step, tok/s, cycles, serial forwards, each trip's
     position and rolling rate, each re-arm's position, how it ended.

Trip positions come from the server log via mtp_ask's --server-log (the
`adaptive bail at +N` lines); a record without them still lands in table
1 and shows as `+?` in table 3, and is left out of the tripped subsets of
table 2 rather than guessed at.

    PYTHONPATH=src .venv/bin/python bench/bail_sweep_table.py verification/bail-sweep-<date>
    ... --out verification/bail-sweep-<date>/table.md
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics as st

DEFAULT_ORDER = ["W32/F0.15/R64", "W64/F0.15/R64", "W128/F0.15/R64", "W32/F0.08/R64",
                 "W64/F0.08/R64", "W128/F0.08/R64", "W64/F0.08/R16", "W64/F0.08/R32"]
CFG_RE = re.compile(r"^W(\d+)-F([0-9.]+)-R(\d+)$")


def fmt(x, p=2):
    return "—" if x is None else f"{x:.{p}f}"


def med(xs):
    xs = [x for x in xs if x is not None]
    return st.median(xs) if xs else None


def mean(xs):
    xs = [x for x in xs if x is not None]
    return st.mean(xs) if xs else None


def rng(xs):
    xs = [x for x in xs if x is not None]
    return f"{min(xs):.2f}–{max(xs):.2f}" if xs else "—"


def load(out_dir: str, order: list[str] | None):
    """{config: {prompt: record}} in sweep order, then anything else sorted."""
    found = {}
    for d in sorted(os.listdir(out_dir)):
        m = CFG_RE.match(d)
        if not m or not os.path.isdir(os.path.join(out_dir, d)):
            continue
        cfg = f"W{m.group(1)}/F{m.group(2)}/R{m.group(3)}"
        recs = {}
        for f in sorted(os.listdir(os.path.join(out_dir, d))):
            if f.endswith(".json") and f[:-5].startswith("P"):
                with open(os.path.join(out_dir, d, f)) as fh:
                    recs[f[:-5]] = json.load(fh)
        if recs:
            found[cfg] = recs
    order = order or DEFAULT_ORDER
    keys = [c for c in order if c in found] + sorted(c for c in found if c not in order)
    return {c: found[c] for c in keys}


def first_trip(run: dict):
    """(position, rolling rate %) of the run's first trip, or None when the
    record has no server-log lines for it; False when it never tripped."""
    if not run.get("trips"):
        return False
    log = run.get("log") or {}
    trips = log.get("trips") or []
    return (trips[0]["pos"], trips[0]["rate"]) if trips else None


def summarize(runs: list[dict]) -> dict:
    ts = [r.get("tok_per_step") for r in runs]
    return {"n": len(runs), "ts_med": med(ts), "ts_mean": mean(ts), "ts_rng": rng(ts),
            "tripped": sum(1 for r in runs if r.get("trips")),
            "trips_mean": mean([r.get("trips") for r in runs]),
            "rearms_mean": mean([r.get("rearms") for r in runs]),
            "toks_med": med([r.get("decode_tok_s") for r in runs]),
            "p50_med": med([r.get("tbt_p50_s") for r in runs]),
            "p95_med": med([r.get("tbt_p95_s") for r in runs])}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out_dir")
    ap.add_argument("--configs", default=None, help="comma list: the row order (default: the sweep's)")
    ap.add_argument("--left", type=int, default=64,
                    help="the like-for-like cut: tripped with at least this many tokens left (default 64)")
    ap.add_argument("--out", default=None, help="write the Markdown here as well as to stdout")
    args = ap.parse_args()
    data = load(args.out_dir, args.configs.split(",") if args.configs else None)
    if not data:
        print(f"no <config-dir>/P*.json under {args.out_dir}")
        return
    prompts = sorted({p for recs in data.values() for p in recs})
    lines = []
    w = lines.append

    w(f"# bail sweep — {os.path.basename(os.path.abspath(args.out_dir))}")
    w("")
    for p in prompts:
        rec = next(recs[p] for recs in data.values() if p in recs)
        w(f"- **{p}**: “{rec['prompt']}” — {rec['max_tokens']} tokens, sampled at the server's defaults")
    w("")
    w("Config = W<window in drafted positions>/F<floor>/R<re-arm stretch>. tok/step = generated / "
      "(generated − accepted) off `/metrics` deltas; trip rate = runs with ≥1 trip / n; TBT = inter-token "
      "time off the SSE stream (median over runs of each run's p50 / p95).")
    w("")
    w("## 1. Per config × prompt")
    w("")
    w("| config | prompt | n | tok/step median | mean | range | trip rate | trips/run | re-arms/run | decode tok/s median | TBT p50 / p95 (s) |")
    w("|---|---|---|---|---|---|---|---|---|---|---|")
    for cfg, recs in data.items():
        for p in prompts:
            if p not in recs:
                continue
            s = summarize(recs[p]["runs"])
            w(f"| {cfg} | {p} | {s['n']} | {fmt(s['ts_med'])} | {fmt(s['ts_mean'])} | {s['ts_rng']} | "
              f"{s['tripped']}/{s['n']} | {fmt(s['trips_mean'], 1)} | {fmt(s['rearms_mean'], 1)} | "
              f"{fmt(s['toks_med'])} | {fmt(s['p50_med'], 3)} / {fmt(s['p95_med'], 3)} |")
    w("")
    w(f"## 2. Like for like: no trip · tripped with ≥{args.left} tokens left · tripped late")
    w("")
    w("A run's cost depends on where its first trip landed, and whether it trips at all is random per "
      "sampled run (identical code on every config up to the first trip). The row that compares configs "
      f"is **tripped with ≥{args.left} tokens left**: the trip's cost — a bounded stretch under a re-arm, "
      "the rest of the answer under a latch — fits inside the run.")
    w("")
    w("| config | prompt | subset | n | tok/step median | mean | range | decode tok/s median | first trip (+pos @rolling) |")
    w("|---|---|---|---|---|---|---|---|---|")
    for cfg, recs in data.items():
        for p in prompts:
            if p not in recs:
                continue
            runs = recs[p]["runs"]
            cut = recs[p]["max_tokens"] - args.left - 1
            no_trip = [r for r in runs if first_trip(r) is False]
            known = [(r, first_trip(r)) for r in runs if first_trip(r)]
            early = [r for r, t in known if t[0] <= cut]
            late = [r for r, t in known if t[0] > cut]
            unknown = [r for r in runs if first_trip(r) is None]
            for name, sel in (("no trip", no_trip), (f"tripped, ≥{args.left} left (first trip ≤ +{cut})", early),
                              (f"tripped, <{args.left} left", late), ("tripped, position unknown", unknown)):
                if not sel:
                    continue
                ts = [r.get("tok_per_step") for r in sel]
                pos = ", ".join(f"+{t[0]} @{t[1]}%" for t in (first_trip(r) for r in sel) if t) or "—"
                w(f"| {cfg} | {p} | {name} | {len(sel)} | {fmt(med(ts))} | {fmt(mean(ts))} | {rng(ts)} | "
                  f"{fmt(med([r.get('decode_tok_s') for r in sel]))} | {pos} |")
    w("")
    w("## 3. Every run")
    w("")
    w("| config | prompt | run | tok/step | decode tok/s | cycles | serial fwd | trips (+pos @rolling) | re-arms (at) | ended | TBT p50 / p95 (s) | wall (s) |")
    w("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for cfg, recs in data.items():
        for p in prompts:
            if p not in recs:
                continue
            for i, r in enumerate(recs[p]["runs"], 1):
                log = r.get("log") or {}
                cyc = log.get("cycles")
                # steps = 1 (the prefill's own token) + cycles + serial forwards
                serial = None if (cyc is None or r.get("steps") is None) else int(r["steps"] - cyc - 1)
                trips = ", ".join(f"+{t['pos']} @{t['rate']}%" for t in log.get("trips") or [])
                if not trips and r.get("trips"):
                    trips = f"{r['trips']}× (+?)"
                rearms = ", ".join(f"+{x['pos']}" for x in log.get("rearms") or [])
                if not rearms and r.get("rearms"):
                    rearms = f"{r['rearms']}× (+?)"
                ended = ("serial" if log.get("serial_at_end") else ("drafting" if r.get("trips") else "no trip")
                         if log else ("tripped" if r.get("trips") else "no trip"))
                w(f"| {cfg} | {p} | {i} | {fmt(r.get('tok_per_step'))} | {fmt(r.get('decode_tok_s'))} | "
                  f"{'—' if cyc is None else cyc} | {'—' if serial is None else serial} | {trips or '—'} | "
                  f"{rearms or '—'} | {ended} | {fmt(r.get('tbt_p50_s'), 3)} / {fmt(r.get('tbt_p95_s'), 3)} | "
                  f"{fmt(r.get('wall_s'), 1)} |")
    text = "\n".join(lines) + "\n"
    print(text, end="")
    if args.out:
        with open(args.out, "w") as f:
            f.write(text)


if __name__ == "__main__":
    main()
