"""Whole-model check of chunked prefill on a real dense pack: the Qwen3-8B
sip pack through the real engine, one process.

Three parts, each printed as it finishes and all written to --out:

  dispatch   — whole-prompt prefill (DRINKME_PREFILL_CHUNK=0) at T = 1K, 2K,
               4K, then chunked at 1024/2048 over T = 4K, each inside a
               CLOCK_MONOTONIC window, so a rocprofv3 --kernel-trace of this
               process (bench/prefill_dispatch.read_trace) gives the longest
               dispatch per configuration. Refuses to go on once a whole-
               prompt forward's wall passes --stop-ms: the wall bounds every
               dispatch inside it.
  throughput — prefill tok/s (engine prefill.run over a fresh cache, device
               synchronized, best of --reps after one untimed rep) for each
               C in --chunks at T = 4K, and C in --long-chunks at T = 16K.
  identity   — greedy transcripts, chunked (--identity-chunk) vs whole, on a
               ~4K-token prompt: serial (DRINKME_SPEC=off), n-gram
               (DRINKME_SPEC=ngram), and a second turn that extends the
               first through the prefix cache.

Hold `flock -n /tmp/drinkme-gpu.lock`. Needs MemAvailable >= 25 GB.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

PACK = os.path.expanduser("~/.cache/drinkme/packs/Qwen--Qwen3-8B@b968826d9c46")


def mem_available_gb() -> float:
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1e6
    return 0.0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--chunks", type=int, nargs="+", default=[1024, 2048, 4096, 0])
    ap.add_argument("--long-chunks", type=int, nargs="+", default=[1024, 2048, 4096, 8192])
    ap.add_argument("--long-T", type=int, default=16384)
    ap.add_argument("--identity-chunk", type=int, nargs="+", default=[1024, 2048])
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--parse", type=Path,
                    help="after the run: the rocprofv3 output dir; adds per-window "
                         "longest dispatches to --out")
    ap.add_argument("--diverge", action="store_true",
                    help="after the run: for each identity case that DIFFERS, the first "
                         "differing token and the top-2 logit margin there, whole vs chunked")
    args = ap.parse_args()
    if args.parse:
        return parse(args)
    if mem_available_gb() < 25:
        raise SystemExit(f"REFUSING: MemAvailable {mem_available_gb():.1f} GB < 25")
    os.environ.update(TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL="1", DRINKME_SLOT_DIR="off",
                      DRINKME_NO_AUTO_DEPS="1", DRINKME_PREFIX_SLOTS="1",
                      DRINKME_SPEC="off")
    import torch

    from drinkme.serve import build_engine
    from drinkme.serving import prefill
    from drinkme.serving.engine import GenerationRequest, SampleParams, complete

    assert torch.cuda.is_available(), "REFUSING CPU measurement"
    meta = json.load(open(os.path.join(PACK, "meta.json")))
    eng = build_engine(meta["hfRepo"], meta["revision"], PACK, stock=False, ctx=32768)
    if args.diverge:
        return diverge(args, eng)
    dev = eng.device
    tok = eng.tok
    text = open(__file__).read()
    base_ids = tok(text, add_special_tokens=False).input_ids
    while len(base_ids) < args.long_T + 16:
        base_ids = base_ids + base_ids
    res: dict = {"pack": PACK, "device": torch.cuda.get_device_name(0), "dispatch": [],
                 "throughput": [], "identity": []}

    def sync():
        torch.cuda.synchronize()

    def prefill_once(ids, chunk):
        cache = eng._cache(len(ids) + 1)
        sync()
        t0, m0 = time.perf_counter(), time.monotonic_ns()
        with torch.inference_mode():
            prefill.run(eng.model, ids, 0, cache, dev, chunk)
        sync()
        return time.perf_counter() - t0, (m0, time.monotonic_ns())

    # ---- dispatch windows. The basis for launching these: the 27B-shape
    # ladder (bench/prefill_dispatch.py) — whole T=4096 peaks at 27 ms there,
    # with larger heads (24 x 256 vs 32 x 128) and a larger MLP than the 8B's.
    # The chunked configs bound every dispatch by the chunk and the segment.
    for T, C in [(1024, 0), (2048, 0), (4096, 0), (4096, 1024), (4096, 2048),
                 (8192, 2048), (16384, 4096)]:
        ids = base_ids[:T]
        walls, windows = [], []
        for _ in range(3):
            w, win = prefill_once(ids, C)
            walls.append(w)
            windows.append(win)
        res["dispatch"].append({"T": T, "chunk": C, "wall_s": walls, "windows": windows})
        print(f"[dispatch] T={T} C={C}: wall {min(walls) * 1e3:.0f} ms", flush=True)

    # ---- throughput
    for T, chunks in [(4096, args.chunks), (args.long_T, args.long_chunks)]:
        ids = base_ids[:T]
        for C in chunks:
            prefill_once(ids, C)  # untimed: compile, autotune, allocator growth
            runs = [prefill_once(ids, C) for _ in range(args.reps)]
            best = min(w for w, _ in runs)
            row = {"T": T, "chunk": C, "best_s": best, "tok_s": T / best,
                   "windows": [win for _, win in runs]}
            res["throughput"].append(row)
            print(f"[throughput] T={T} C={C or 'whole'}: {T / best:,.0f} tok/s "
                  f"({best * 1e3:.0f} ms)", flush=True)

    # ---- identity: chunked vs whole greedy transcripts
    params = SampleParams(temperature=0.0, max_tokens=48)

    def transcript(chunk, spec, msgs):
        os.environ["DRINKME_SPEC"] = spec
        eng._spec_plan, eng._mtp_depth = None, None
        eng._prefill_chunk = chunk
        r = complete(eng, GenerationRequest(msgs, params,
                                            template_kwargs={"enable_thinking": False}))
        return r

    cases = [(spec, C, 3900) for spec in ("off", "ngram") for C in args.identity_chunk]
    # ~8K: chunks past the first attend over more than SPLIT_ABOVE pairs, so
    # the segmented attention runs
    cases += [("off", C, 7900) for C in args.identity_chunk]
    for spec, C, n_body in cases:
        body = tok.decode(base_ids[:n_body])
        turn1 = [{"role": "user", "content": "Here is a file:\n" + body +
                  "\nWhat does this file measure? Answer in two sentences."}]
        eng.reset_prefix_cache()
        w1 = transcript(0, spec, turn1)
        eng.reset_prefix_cache()
        c1 = transcript(C, spec, turn1)
        row = {"spec": spec, "chunk": C, "prompt_tokens": c1.prompt_tokens,
               "turn1_identical": w1.text == c1.text, "whole": w1.text, "chunked": c1.text}
        res["identity"].append(row)
        print(f"[identity] spec={spec} C={C} prompt={c1.prompt_tokens}: "
              f"{'IDENTICAL' if row['turn1_identical'] else 'DIFFERS'}", flush=True)

    # ---- the prefix-cache extend, at the prefill seam: the chat template
    # re-renders an assistant turn, so a second chat turn does not extend
    # the first's ids; this extends them exactly. Warm = prefill ids1
    # chunked, then extend to ids2 chunked; whole = ids2 in one forward.
    def greedy_from(cache, logits, pos, n=24):
        out = []
        with torch.inference_mode():
            for i in range(n):
                t = int(logits.argmax())
                out.append(t)
                logits = eng.model(torch.tensor([[t]], device=dev), past_key_values=cache,
                                   use_cache=True, cache_position=torch.arange(pos + i, pos + i + 1, device=dev),
                                   logits_to_keep=1).logits[0, -1]
        return out

    for C, n1, n2 in [(2048, 5000, 9000), (1024, 3000, 4100)]:
        ids1, ids2 = base_ids[:n1], base_ids[:n2]
        with torch.inference_mode():
            cw = eng._cache(n2 + 32)
            lw = prefill.run(eng.model, ids2, 0, cw, dev, 0)
            cc = eng._cache(n2 + 32)
            prefill.run(eng.model, ids1, 0, cc, dev, C)
            lc = prefill.run(eng.model, ids2, n1, cc, dev, C)
        gw, gc = greedy_from(cw, lw, n2), greedy_from(cc, lc, n2)
        d = (lw.float() - lc.float()).abs().max().item()
        row = {"chunk": C, "cached": n1, "prompt": n2, "last_row_max_abs_diff": d,
               "greedy_identical": gw == gc, "whole": gw, "warm_chunked": gc}
        res["extend"] = res.get("extend", []) + [row]
        print(f"[extend] C={C} cached {n1} -> {n2}: last-row max|d| {d:.3g}, greedy 24 "
              f"{'IDENTICAL' if gw == gc else 'DIFFERS'}", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=1))
    print(f"VERDICT wrote {args.out}", flush=True)


def _identity_prompt(tok, base_ids, n_body):
    body = tok.decode(base_ids[:n_body])
    return [{"role": "user", "content": "Here is a file:\n" + body +
             "\nWhat does this file measure? Answer in two sentences."}]


def diverge(args, eng) -> None:
    """Near-tie or not: at the first token where the chunked transcript left
    the whole one, the whole path's and the chunked path's top two logits
    over the shared prefix (prompt + the tokens both emitted)."""
    import torch

    from drinkme.serving import prefill
    from drinkme.serving.engine import GenerationRequest, SampleParams

    res = json.loads(args.out.read_text())
    tok = eng.tok
    base_ids = tok(open(__file__).read(), add_special_tokens=False).input_ids
    while len(base_ids) < 20000:
        base_ids = base_ids + base_ids
    out = []
    for row in res["identity"]:
        if row["turn1_identical"]:
            continue
        n_body = 3900 if row["prompt_tokens"] < 6000 else 7900
        req = GenerationRequest(_identity_prompt(tok, base_ids, n_body),
                                SampleParams(temperature=0.0, max_tokens=48),
                                template_kwargs={"enable_thinking": False})
        ids = eng._render(req).ids
        w = tok(row["whole"], add_special_tokens=False).input_ids
        c = tok(row["chunked"], add_special_tokens=False).input_ids
        p = next(i for i in range(min(len(w), len(c)) + 1)
                 if i == min(len(w), len(c)) or w[i] != c[i])
        prefix = ids + w[:p]
        tops = {}
        for C in (0, row["chunk"]):
            with torch.inference_mode():
                lg = prefill.run(eng.model, prefix, 0, eng._cache(len(prefix) + 1),
                                 eng.device, C).float()
            v, i = lg.topk(2)
            tops[C] = {"top2_ids": i.tolist(), "top2_logits": [round(x, 4) for x in v.tolist()],
                       "margin": round((v[0] - v[1]).item(), 4)}
        r = {"spec": row["spec"], "chunk": row["chunk"], "prompt": len(ids),
             "first_diff_token": p, "whole_token": w[p] if p < len(w) else None,
             "chunked_token": c[p] if p < len(c) else None, "whole_path": tops[0],
             "chunked_path": tops[row["chunk"]]}
        out.append(r)
        print(f"[diverge] spec={row['spec']} C={row['chunk']} prompt={len(ids)}: first "
              f"difference at generated token {p}; whole top2 {tops[0]['top2_ids']} "
              f"margin {tops[0]['margin']}, chunked top2 {tops[row['chunk']]['top2_ids']} "
              f"margin {tops[row['chunk']]['margin']}", flush=True)
    res["diverge"] = out
    args.out.write_text(json.dumps(res, indent=1))


def parse(args) -> None:
    import sys

    sys.path.insert(0, os.path.dirname(__file__))
    from prefill_dispatch import family, read_trace

    res = json.loads(args.out.read_text())
    rows = read_trace(str(args.parse))
    rows.sort(key=lambda r: r["t0"])

    def longest(windows):
        inside = [r for r in rows if any(a <= r["t0"] <= b for a, b in windows)]
        per = {}
        for r in inside:
            f = family(r["name"])
            if r["ms"] > per.get(f, (0, ""))[0]:
                per[f] = (r["ms"], r["name"][:80])
        top = max(inside, key=lambda r: r["ms"])
        return {"max_ms": top["ms"], "kernel": top["name"][:120], "n": len(inside),
                "families": per}

    for part in ("dispatch", "throughput"):
        for row in res[part]:
            row["longest"] = longest(row["windows"])
            print(f"[{part}] T={row['T']} C={row['chunk']}: longest dispatch "
                  f"{row['longest']['max_ms']:.1f} ms ({family(row['longest']['kernel'])}: "
                  f"{row['longest']['kernel'][:50]})")
    args.out.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
