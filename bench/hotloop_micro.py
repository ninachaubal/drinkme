#!/usr/bin/env python3
"""Per-token cost of the hot loop, on a CPU toy.

One table: per-token cost at 256 / 1024 / 4096 generated tokens for
(a) plain greedy, (b) penalties on, (c) constrained JSON, with the hot loop's
per-token savings on and off (DRINKME_HOTLOOP_OFF=1, the unoptimized path,
serving/engines.py). The same file measures both sides:

    PYTHONPATH=src python bench/hotloop_micro.py --label branch
    PYTHONPATH=src DRINKME_HOTLOOP_OFF=1 python bench/hotloop_micro.py --label off

The model is deliberately tiny (2 layers, hidden 64) so the table is dominated
by the HOST work under measurement — detok, penalties, the JSON validator,
the full-vocabulary clones — rather than by a matmul neither side changed. It
is a real HFEngine over a real tokenizer running the real generate() loop, not
a mock of it; the model forward is identical on both sides, so the DIFFERENCE
between two columns is that host work and nothing else.

EOS is disabled for the run (a randomly-initialised toy likes its own end
token) so every cell generates exactly the requested number of tokens.

No GPU, no downloads, no network.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                os.pardir, "tests"))

# The toy lives with the tests that also use it (tests/hotloop_toys.py): one
# definition, so the bench and the equivalence battery measure the same thing.
from hotloop_toys import toy_engine  # noqa: E402

LENGTHS = (256, 1024, 4096)

# An array that can never be complete, so the constraint never lets EOS
# through and the run reaches its token budget: long strings inside a growing
# document, which is exactly the shape whose validator cost was quadratic.
JSON_SCHEMA = {"type": "array", "items": {"type": "string"}, "minItems": 100000}

MSGS = [{"role": "user", "content": "write a plan"}]


def generate(eng, params):
    """One generation through the seam (engine.complete), collected."""
    from drinkme.serving.engine import GenerationRequest, complete

    return complete(eng, GenerationRequest(MSGS, params))


def params_for(mode: str, n: int):
    from drinkme.serving.engine import SampleParams

    if mode == "greedy":
        return SampleParams(temperature=0.0, max_tokens=n)
    if mode == "penalties":
        return SampleParams(temperature=0.0, max_tokens=n,
                            repetition_penalty=1.15, presence_penalty=0.6,
                            frequency_penalty=0.4)
    if mode == "json":
        return SampleParams(temperature=0.0, max_tokens=n,
                            output_schema=JSON_SCHEMA)
    raise SystemExit(f"unknown mode {mode}")


def measure_floor(eng, n: int) -> dict:
    """The FLOOR: the model forward plus an argmax, no serve loop at all — no
    detok, no scanners, no penalties, no constraint. Neither side of this
    comparison touches it, so it is what every other row is standing on, and the
    difference between a row and this row is the host work under measurement.
    Same geometric StaticCache allocation generate() would have made."""
    import torch

    from drinkme.serving.template import build_prompt

    ids = build_prompt(eng.tok, MSGS)
    need, size = len(ids) + n, eng.KV_FLOOR
    while size < need:
        size *= 2
    size = min(size, eng.ctx)
    best = None
    for _ in range(2 if n <= 1024 else 1):  # same warm-up rule as measure()
      with torch.inference_mode():
        cache = eng._cache(size)
        logits = eng.model(torch.tensor([ids], dtype=torch.long),
                           past_key_values=cache, use_cache=True,
                           cache_position=torch.arange(0, len(ids))).logits[0, -1]
        step_in = torch.empty((1, 1), dtype=torch.long)
        step_pos = torch.empty((1,), dtype=torch.long)
        t0 = time.perf_counter()
        for i in range(n):
            step_in[0, 0] = int(torch.argmax(logits))
            step_pos[0] = len(ids) + i
            logits = eng.model(step_in, past_key_values=cache, use_cache=True,
                               cache_position=step_pos).logits[0, -1]
        dt = time.perf_counter() - t0
        best = dt if best is None else min(best, dt)
    return {"mode": "forward", "n": n, "tokens": n, "seconds": round(best, 3),
            "ms_per_token": round(1000 * best / n, 3), "finish": "floor"}


def measure(eng, mode: str, n: int) -> dict:
    """One cell. Short runs go twice and keep the faster — python warms up,
    and a receipt should not be reporting that."""
    p = params_for(mode, n)
    best, got, finish = None, 0, ""
    for _ in range(2 if n <= 1024 else 1):
        t0 = time.perf_counter()
        try:
            res = generate(eng, p)
        except RuntimeError as e:  # a constrained dead end is a result
            return {"mode": mode, "n": n, "ms_per_token": None,
                    "note": str(e)[:60]}
        dt = time.perf_counter() - t0
        best = dt if best is None else min(best, dt)
        got, finish = res.completion_tokens, res.finish_reason
    return {"mode": mode, "n": n, "tokens": got, "seconds": round(best, 3),
            "ms_per_token": round(1000 * best / max(got, 1), 3),
            "finish": finish}


def penalty_sweep(lengths) -> list:
    """Items B and F at a REAL vocabulary size, which the toy cannot show.

    What one decode step pays to apply penalties: the OLD block (verbatim, out
    of tests/hotloop_oracle.py — sorted(set(prev_ids)), a counts dict over
    gen_ids, two fresh [V] f32 allocations, a host-to-device copy) against the
    new one (lend a preallocated buffer, one pass over the device-resident
    masks). The trade is O(history) host work for O(vocabulary) device work,
    so the honest question is where they cross — these are CPU numbers, and on
    a GPU the vocabulary pass is a kernel while the list rebuild stays python,
    which moves the crossing left, not right.

    Branch-only: it needs both implementations in one process.
    """
    import torch

    from drinkme.serving.engine import GenerationRequest, SampleParams, complete
    from drinkme.serving.sampling import PenaltyState, SampleScratch
    from hotloop_oracle import old_penalties

    rows = []
    params = SampleParams(temperature=0.0, repetition_penalty=1.15,
                          presence_penalty=0.6, frequency_penalty=0.4)
    print(f"\n# penalties per step, old block vs new (branch-only)")
    print(f"{'vocab':>8}{'history':>9}{'old ms':>10}{'new ms':>10}{'speedup':>9}")
    for vocab in (4096, 32000, 151936):
        row = torch.randn(vocab)
        for n in lengths:
            g = torch.Generator().manual_seed(n)
            hist = torch.randint(0, vocab, (n,), generator=g).tolist()
            state, scratch = PenaltyState(vocab), SampleScratch()
            state.observe(range(64))
            for t in hist:
                state.accept(t)
            prev = list(range(64)) + hist

            def new_step():
                buf = scratch.take_work(row)
                state.apply(buf, params)

            def old_step():
                old_penalties(row, params, prev_ids=prev, gen_ids=hist)

            def timed(fn, reps=40):
                for _ in range(5):
                    fn()
                best = None
                for _ in range(3):
                    t0 = time.perf_counter()
                    for _ in range(reps):
                        fn()
                    dt = (time.perf_counter() - t0) / reps
                    best = dt if best is None else min(best, dt)
                return 1000 * best

            ms_old, ms_new = timed(old_step), timed(new_step)
            rows.append({"kind": "penalty", "vocab": vocab, "history": n,
                         "old_ms": round(ms_old, 4), "new_ms": round(ms_new, 4)})
            print(f"{vocab:>8}{n:>9}{ms_old:>10.3f}{ms_new:>10.3f}"
                  f"{ms_old / ms_new:>8.1f}x")
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="run", help="what this column is")
    ap.add_argument("--lengths", default=",".join(str(x) for x in LENGTHS))
    ap.add_argument("--modes", default="forward,greedy,penalties,json")
    ap.add_argument("--json-out", default=None, help="write the rows as JSON")
    ap.add_argument("--penalty-sweep", action="store_true",
                    help="also measure the penalty path at real vocabulary sizes")
    args = ap.parse_args()

    os.environ.setdefault("DRINKME_PREFIX_SLOTS", "0")  # every cell cold
    lengths = [int(x) for x in args.lengths.split(",")]
    modes = args.modes.split(",")

    import drinkme
    import torch

    ctx = max(lengths) + 256
    eng = toy_engine(seed=0, ctx=ctx, kv_floor=256)
    eng.eos_ids = frozenset()  # generate exactly what was asked for
    hot_off = os.environ.get("DRINKME_HOTLOOP_OFF") == "1"
    try:
        rev = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True).stdout.strip()
    except OSError:
        rev = "?"
    print(f"# hotloop_micro  label={args.label}  rev={rev}  "
          f"DRINKME_HOTLOOP_OFF={'1' if hot_off else '0'}")
    print(f"# drinkme from {os.path.dirname(drinkme.__file__)}  "
          f"torch {torch.__version__}  vocab {len(eng.tok)}  ctx {ctx}")

    generate(eng, params_for("greedy", 8))  # warm python

    rows = []
    print(f"{'mode':<11}{'tokens':>8}{'seconds':>10}{'ms/token':>11}")
    for mode in modes:
        for n in lengths:
            row = measure_floor(eng, n) if mode == "forward" else measure(eng, mode, n)
            row["label"] = args.label
            rows.append(row)
            if row["ms_per_token"] is None:
                print(f"{mode:<11}{n:>8}{'—':>10}{'dead-end':>11}  {row['note']}")
            else:
                print(f"{mode:<11}{row['tokens']:>8}{row['seconds']:>10.3f}"
                      f"{row['ms_per_token']:>11.3f}")
            sys.stdout.flush()
    if args.penalty_sweep:
        rows += penalty_sweep(lengths)
    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump({"label": args.label, "rev": rev, "hot_off": hot_off,
                       "rows": rows}, f, indent=2)
        print(f"# wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
