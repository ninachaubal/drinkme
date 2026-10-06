#!/usr/bin/env python3
"""The gate for prefill's one-row lm_head: does narrowing lm_head to one row
change that row?

Prefill does not materialize logits for every prompt position (that would be
0.474 MiB discarded per token at vocab 248320, 121 GiB at the advertised 262k
context): it narrows lm_head to the last row, on BOTH arms.

THE STATED RISK, which this measures rather than assumes: on the compressed
arm a 1-row lm_head takes the GEMV kernel while a many-row one takes
decode-once + a dense GEMM, and those agree only to accumulation order. So a
near-tie at token one could land the other way than the every-row product.
Arm-vs-arm agreement is the prefix-cache A/B's question; this is the
one-row-vs-every-row one.

Compares, at a real prompt on the real model:
    lm_head(hidden)[-1]      every row, then discard
    lm_head(hidden[:, -1:])  one row, as prefill runs it

on bitwise equality (reported), on argmax (does the sampled token move), and
where it moves, each row's gap between the two picks (agreement.row_fork).
Two orders of the same sums can flip a near-tie, which is reported and
passes; a flip with a gap above agreement.NEAR_TIE_MARGIN, or a row that is
not finite, exits 2.

Run:  uv run --no-sync python bench/logits_row_identity.py --pack <dir>
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from agreement import NEAR_TIE_MARGIN, above_margin, describe, row_fork  # noqa: E402

DEFAULT_PACK = os.path.expanduser(
    "~/.cache/drinkme/packs/Qwen--Qwen3.8-27B@1d4bf0f2ff60")

PROMPTS = [
    "Explain in two sentences why a downward raycast cannot see a surface "
    "above its own origin.",
    "List three reasons a long prompt might exhaust GPU memory during prefill.",
    "Write one sentence about a library that keeps what it has been told.",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", default=DEFAULT_PACK)
    ap.add_argument("--ctx", type=int, default=8192)
    args = ap.parse_args()
    pack = os.path.expanduser(args.pack)

    import torch
    from transformers import AutoTokenizer, StaticCache
    from drinkme.serve import build_engine

    meta = json.load(open(os.path.join(pack, "meta.json")))
    print(f"[gate] loading {meta['hfRepo']} (meanBpw {meta['meanBpw']}) ...",
          flush=True)
    t0 = time.time()
    eng = build_engine(meta["hfRepo"], meta["revision"], pack, stock=False,
                       ctx=args.ctx)
    print(f"[gate] loaded in {time.time() - t0:.0f}s  device={eng.device}",
          flush=True)
    if str(eng.device) == "cpu":
        sys.exit("[gate] REFUSING to measure on CPU")

    tok = AutoTokenizer.from_pretrained(meta["hfRepo"])
    model = eng.model
    base = model.get_decoder() if hasattr(model, "get_decoder") else model.model

    rows = []
    for i, p in enumerate(PROMPTS):
        text = tok.apply_chat_template([{"role": "user", "content": p}],
                                       add_generation_prompt=True,
                                       tokenize=False)
        ids = tok(text, add_special_tokens=False)["input_ids"]
        inp = torch.tensor([ids], dtype=torch.long, device=eng.device)
        with torch.inference_mode():
            cache = StaticCache(config=model.config.get_text_config(),
                                max_cache_len=args.ctx, max_batch_size=1,
                                dtype=torch.bfloat16, device=eng.device)
            out = base(inp, past_key_values=cache, use_cache=True,
                       cache_position=torch.arange(len(ids), device=eng.device))
            h = out.last_hidden_state
            old = model.lm_head(h)[0][-1].float()        # every row, keep last
            new = model.lm_head(h[:, -1:])[0][-1].float()  # one row

        bitwise = bool(torch.equal(old, new))
        finite = bool(torch.isfinite(old).all() and torch.isfinite(new).all())
        pick = row_fork(old, new)
        top2 = torch.topk(old, 2).values
        gap = float(top2[0] - top2[1])
        maxdiff = float((old - new).abs().max())
        rows.append({"prompt_tokens": len(ids), "bitwise_identical": bitwise,
                     "finite": finite, "same_argmax": pick["agree"], "pick": pick,
                     "top2_gap": round(gap, 6), "max_abs_diff": maxdiff})
        print(f"  [{i+1}] tokens={len(ids):>5} bitwise={bitwise} finite={finite} "
              f"token {describe(pick)} top2_gap={gap:.4f} "
              f"max|diff|={maxdiff:.3e}", flush=True)

    all_same_token = all(r["same_argmax"] for r in rows)
    all_bitwise = all(r["bitwise_identical"] for r in rows)
    above = [i + 1 for i, r in enumerate(rows) if above_margin(r["pick"])]
    report = {
        "date": time.strftime("%Y-%m-%d %H:%M %Z"),
        "model": meta["hfRepo"],
        "device": str(eng.device),
        "near_tie_margin": NEAR_TIE_MARGIN,
        "rows": rows,
        "all_bitwise_identical": all_bitwise,
        "all_same_sampled_token": all_same_token,
        "all_finite": all(r["finite"] for r in rows),
        "flips_above_near_tie_margin": above,
    }
    report["pass"] = report["all_finite"] and not above
    out_dir = os.path.join(os.path.dirname(__file__), "..", "verification")
    path = os.path.abspath(os.path.join(
        out_dir, f"logits_row_identity_{meta['hfRepo'].split('/')[-1]}_"
                 f"{time.strftime('%Y-%m-%d')}.json"))
    os.makedirs(out_dir, exist_ok=True)
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps({k: report[k] for k in
                      ("all_bitwise_identical", "all_same_sampled_token", "all_finite",
                       "flips_above_near_tie_margin")},
                     indent=2))
    print(f"[gate] {'PASS' if report['pass'] else 'FAIL'} — wrote {path}")
    # bitwise equality and the sampled token are reported, not required: the
    # two paths use different kernels by design and agree only to
    # accumulation order, so a flip passes when it is a near-tie.
    return 0 if report["pass"] else 2


if __name__ == "__main__":
    # os._exit: TheRock torch _exit(0)s from atexit once HIP initialises, so a
    # bare exit would report success for a gate that failed. Same reason the
    # drinkme CLI and prefix_cache_verify exit this way.
    import os as _os
    try:
        _code = main() or 0
    except SystemExit as e:
        _code = e.code if isinstance(e.code, int) else (1 if e.code else 0)
        if e.code and not isinstance(e.code, int):
            print(e.code, file=sys.stderr)
    sys.stdout.flush()
    sys.stderr.flush()
    _os._exit(_code)
