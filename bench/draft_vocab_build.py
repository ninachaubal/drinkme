#!/usr/bin/env python3
"""The draft diet, lever 1: build a draft-vocab subset, and measure both halves
of what it trades.

The lever (serving/draft_vocab.py) lets a DRAFT step argmax over K of V vocab
rows, cutting the biggest read in a speculative cycle by V/K. It is lossless by
construction — verify runs the full projection — so the only cost is
ACCEPTANCE, and the only two numbers that matter are:

  COVERAGE   what fraction of the argmax picks a real generation makes fall
             inside the subset. A miss costs that cycle's tail, so this is the
             acceptance cost's dominant term. `--evaluate` measures it on
             HELD-OUT prompts (never the ones the subset was built from —
             evaluating on your own training set is not a measurement).
  READ CUT   how far the projection's latency actually falls with K/R.
             (no codec has a row-selected kernel that reads K/R of the
             bytes — codec/registry.py's ROW_SUBSET rows; the subset file
             still measures coverage, and such a kernel would use it)

Two modes:

    # build from the model's own output statistics, over a prompt set
    uv run --no-sync python bench/draft_vocab_build.py --out sub.json -k 32768

    # what fraction of held-out argmax picks does it contain?
    uv run --no-sync python bench/draft_vocab_build.py --evaluate sub.json

METHOD, stated because it is a judgement call. The subset is the union of
(a) every token the model's own greedy argmax picked over the build prompts,
(b) the top-N runners-up at each position — the near-misses a draft head is
    most likely to land on,
(c) padding by ASCENDING TOKEN ID up to K, which leans on BPE id order being a
    rough frequency order. That assumption is crude and it is where the method
    is weakest: no corpus this script can run in a few minutes has 32k distinct
    token types in it, so most of a large K is padding. `--evaluate` is what
    keeps that honest — it measures the result, not the intention.
(d) plus the config's special ids, so a draft can always propose end-of-turn.
"""
import argparse
import json
import os
import sys
import time
from collections import Counter

sys.path.insert(0, "src")

import torch  # noqa: E402

BUILD_PROMPTS = [
    "Write a detailed, multi-paragraph explanation of why the sky is blue.",
    "Write a Python function that parses an ISO-8601 timestamp into a datetime.",
    "Explain the difference between a mutex and a semaphore, with examples.",
    "Summarize the causes of the 1929 stock market crash.",
    "Write a bash script that finds and deletes empty directories, safely.",
    "Describe how a B-tree index speeds up a database range query.",
    "List ten common cooking mistakes and how to avoid each one.",
    "Explain memory bandwidth versus compute in one paragraph, then in five.",
]
EVAL_PROMPTS = [
    "Write three paragraphs about why local inference matters for privacy.",
    "Write a Rust function that reverses a linked list in place, with tests.",
    "Explain what a Kalman filter does to someone who knows basic algebra.",
    "Count from 1 to 30, one number per line, no other text.",
    "Draft a polite email declining a meeting and proposing async notes.",
    "What is the difference between TCP and UDP? Give a concrete example.",
]
N_TOKENS = 192
TOPN = 8


def _engine(args):
    os.environ.setdefault("DRINKME_SPEC", "off")
    os.environ.setdefault("DRINKME_PREFIX_SLOTS", "0")
    from drinkme.serve import build_engine

    e = build_engine(args.model, None, args.pack, stock=args.stock, ctx=4096)
    e.template_kwargs = {"enable_thinking": False}
    return e


def _greedy_picks(engine, prompts, n_tokens, topn):
    """Run the model greedily and record, at every generated position, the
    full-vocab argmax and its top-N runners-up. Uses the trunk directly rather
    than the MTP head: the head is not always present, and what a subset has
    to contain is the tokens that actually get EMITTED."""
    from transformers import StaticCache

    from drinkme.serving.template import build_prompt

    model, tok = engine.model, engine.tok
    dev = next(model.parameters()).device
    argmax_counts, top_counts, total = Counter(), Counter(), 0
    for prompt in prompts:
        ids = build_prompt(tok, [{"role": "user", "content": prompt}],
                           template_kwargs={"enable_thinking": False})
        ids = list(ids)
        cache = StaticCache(config=model.config,
                            max_cache_len=len(ids) + n_tokens + 8, device=dev,
                            dtype=next(model.parameters()).dtype)
        cur = torch.tensor([ids], device=dev)
        pos = torch.arange(len(ids), device=dev)
        with torch.inference_mode():
            for _ in range(n_tokens):
                out = model(cur, past_key_values=cache, use_cache=True,
                            cache_position=pos)
                logits = out.logits[0, -1]
                top = torch.topk(logits, topn).indices.tolist()
                argmax_counts[top[0]] += 1
                for t in top:
                    top_counts[t] += 1
                total += 1
                nxt = top[0]
                if nxt in (engine.stop_ids if hasattr(engine, "stop_ids") else ()):
                    break
                cur = torch.tensor([[nxt]], device=dev)
                pos = torch.tensor([int(pos[-1]) + 1], device=dev)
        del cache
    return argmax_counts, top_counts, total


def _ks(args, V):
    if not args.k.strip():
        return [V // 8]
    return [int(x) for x in args.k.split(",") if x.strip()]


def build(args) -> int:
    """One greedy pass over the build prompts produces the statistics; EVERY
    requested K is then cut from that same ranking, so the subsets are nested
    and the curve --evaluate draws is a curve in K alone."""
    engine = _engine(args)
    cfg = engine.model.config
    text = cfg.get_text_config() if hasattr(cfg, "get_text_config") else cfg
    V = int(text.vocab_size)
    ks = _ks(args, V)
    print(f"building draft vocabs {ks} of {V} from {len(BUILD_PROMPTS)} prompts "
          f"x {args.tokens} tokens...", flush=True)
    t0 = time.perf_counter()
    _, top_counts, total = _greedy_picks(engine, BUILD_PROMPTS, args.tokens, args.topn)

    from drinkme.serving.draft_vocab import _special_ids

    observed = [t for t, _ in top_counts.most_common()]
    ranked = list(dict.fromkeys(observed + _special_ids(cfg)))
    n_obs = len(ranked)
    # padding by ASCENDING TOKEN ID: the crude half of the method, and the
    # half --evaluate exists to keep honest
    ranked += [t for t in range(V) if t not in top_counts]
    dt = round(time.perf_counter() - t0, 1)

    stem = args.out[:-5] if args.out.endswith(".json") else args.out
    written = []
    for K in ks:
        ids = sorted(set(ranked[:K]))
        path = args.out if len(ks) == 1 else f"{stem}.k{K}.json"
        with open(path, "w") as f:
            json.dump({
                "ids": ids, "model": args.model, "vocab_size": V, "k": len(ids),
                "method": (f"argmax+top{args.topn} over {len(BUILD_PROMPTS)} "
                           f"prompts x {args.tokens} tokens ({n_obs} distinct "
                           f"observed), padded by ascending id, + config specials"),
                "distinct_observed": n_obs,
                "padded_by_id": max(0, len(ids) - n_obs),
                "positions_sampled": total, "build_seconds": dt,
            }, f)
        written.append(path)
        print(f"  K={len(ids):7d} ({100 * len(ids) / V:5.1f}% of vocab): "
              f"{min(n_obs, len(ids))} observed, "
              f"{max(0, len(ids) - n_obs)} padded by id -> {path}")
    print(f"{dt}s")
    print("now measure them:  --evaluate " + ",".join(written))
    return 0


def evaluate(args) -> int:
    """Score every named subset off ONE held-out pass: the argmax picks do not
    depend on the subset, so the whole curve costs one generation run."""
    paths = [p for p in args.evaluate.split(",") if p.strip()]
    blobs = []
    for path in paths:
        with open(path) as f:
            b = json.load(f)
        b["_set"] = set(int(i) for i in b["ids"])
        b["_path"] = path
        blobs.append(b)
    engine = _engine(args)
    print(f"evaluating {len(blobs)} subset(s) on {len(EVAL_PROMPTS)} HELD-OUT "
          f"prompts x {args.tokens} tokens...", flush=True)
    argmax_counts, _, total = _greedy_picks(engine, EVAL_PROMPTS, args.tokens, 1)
    V = blobs[0].get("vocab_size") or 0
    print(f"\n  positions sampled: {total}   distinct argmax tokens: "
          f"{len(argmax_counts)}")
    print(f"\n  {'K':>8s} {'K/V':>7s} {'read cut':>9s} {'coverage':>9s} "
          f"{'c^4':>7s} {'distinct':>10s}")
    rows = []
    for b in blobs:
        sub = b["_set"]
        hit = sum(c for t, c in argmax_counts.items() if t in sub)
        cov = hit / total if total else 0.0
        dh = sum(1 for t in argmax_counts if t in sub)
        kv = len(sub) / V if V else 0.0
        print(f"  {len(sub):8d} {kv:7.3f} {(1 / kv if kv else 0):8.1f}x "
              f"{100 * cov:8.2f}% {cov ** 4:7.3f} "
              f"{dh:4d}/{len(argmax_counts):<5d}")
        rows.append({"path": b["_path"], "k": len(sub), "k_over_v": kv,
                     "read_cut": (1 / kv if kv else 0),
                     "argmax_in_subset": hit, "coverage": round(cov, 5),
                     "coverage_pow_depth4": round(cov ** 4, 5),
                     "distinct_in_subset": dh,
                     "method": b.get("method")})
    print("\n  A miss costs that cycle's TAIL, not a token: verify runs the full")
    print("  projection and emits the serial greedy stream regardless. At depth k a")
    print("  per-step coverage c caps the drafted prefix at c^k BEFORE the head's")
    print("  own errors are counted, so c^4 is the multiplier on the measured")
    print("  acceptance — read it against the read cut in the same row.")
    out = {
        "gate": "rung3 part B lever 1 — draft-vocab coverage (held out)",
        "model": args.model, "vocab_size": V,
        "eval_prompts": len(EVAL_PROMPTS), "tokens_per_prompt": args.tokens,
        "positions": total, "distinct_argmax_tokens": len(argmax_counts),
        "subsets": rows,
    }
    if args.json:
        with open(args.json, "w") as f:
            json.dump(out, f, indent=1)
        print(f"wrote {args.json}")
    return 0




def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--pack", default=os.path.expanduser(
        "~/.cache/drinkme/packs/Qwen--Qwen3-4B@1cfa9a720891"))
    ap.add_argument("--stock", action="store_true")
    ap.add_argument("--out", default="draft_vocab.json")
    ap.add_argument("-k", default="", help="subset size(s), comma-separated "
                                          "(default V/8)")
    ap.add_argument("--tokens", type=int, default=N_TOKENS)
    ap.add_argument("--topn", type=int, default=TOPN)
    ap.add_argument("--evaluate")
    ap.add_argument("--json")
    args = ap.parse_args()
    if args.evaluate:
        return evaluate(args)
    return build(args)


if __name__ == "__main__":
    sys.exit(main())
