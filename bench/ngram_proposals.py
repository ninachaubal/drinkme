#!/usr/bin/env python3
"""What does the n-gram proposer actually propose, per prompt class.

The CPU-measurable half of the ngram gate. The proposer reads token ids and
nothing else, so NO MODEL RUNS HERE: its behaviour is fully determined by the
id stream, which makes this measurement exact on CPU and no better on a GPU.
What a GPU adds is the other half — what those proposals are WORTH, which is
acceptance and tok/s.

Two numbers per prompt class:

  proposals/token   draft tokens offered per emitted token. Zero means the
                    cycle drafts nothing, which is an M=1 verify — a plain
                    decode step. High means the lookup is betting, and the
                    verify batch widens whether or not the bet pays.
  hit rate          fraction of cycles where the lookup found anything.

FOUR CLASSES, and the interesting one is not the one you would guess:

  random-tokens     uniform draws over the vocabulary. The entropy floor:
                    what the proposer does when there is provably nothing to
                    find. Must be ~0.
  prose             real English (this repo's README), which repeats WORDS
                    constantly and phrases almost never.
  agent-transcript  a tool result re-sent verbatim across turns — the case
                    prompt lookup exists for, and the shape an agent sends.
  code-edit         a file re-sent with one hunk changed.

THE SWEEP is the point of the file. `prompt_lookup_min` (vLLM's default: 1)
decides how short a match may be before the proposer will bet on it, and it
moves the prose number by more than an order of magnitude while barely
touching the agent number. Ship the vLLM defaults, print the sweep, and let
the tok/s gate on real hardware pick — that is a bandwidth question this file
cannot answer.

    uv run --no-sync python bench/ngram_proposals.py
    uv run --no-sync python bench/ngram_proposals.py --model Qwen/Qwen3-8B \
        --json /tmp/ngram_proposals.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time

sys.path.insert(0, "src")

from drinkme.serving import ngram  # noqa: E402

# The pass criteria, written down here rather than left to a reader with a
# JSON: a gate whose bar lives in somebody's head is not a gate. Note what is
# NOT gated — the prose number. It is a COST datum, not a correctness one,
# and what it costs is a wider verify batch, which is a bandwidth question
# this file cannot answer (module docstring).
NOISE_MAX = 0.15     # proposals/token on uniform random ids, at min=1
AGENT_MIN = 1.50     # and an agent transcript must clear this


def prompt_classes(args) -> dict[str, str]:
    """Text, not ids — the tokenizer turns these into the streams measured."""
    r = random.Random(args.seed)
    words = ("handbook section plan rule tone user assistant respond cite "
             "example paragraph summarize propose justify draft").split()

    # Real English, not a template: a generated "prose" sample made of one
    # sentence with the numbers changed is an agent transcript wearing a
    # disguise, and measures nothing. This repo's own README is prose a
    # server would plausibly be asked about, and it is on disk.
    try:
        prose = open(args.prose, encoding="utf-8").read()[:args.chars]
    except OSError as err:  # noqa: PERF203 — one read, one fallback
        print(f"[ngram] {args.prose}: {err} — prose class skipped", file=sys.stderr)
        prose = None

    # Claude-Code-shaped: one tool result (a file listing) re-sent verbatim
    # every turn, with a short unique reply between turns.
    tool = "\n".join(
        f"  src/drinkme/serving/{n}.py  {200 + i * 37} lines  modified 2026-09-0{i % 7 + 1}"
        for i, n in enumerate(["http", "engine", "engines", "mtp", "ngram",
                               "sampling", "constrain", "template", "detok",
                               "think", "tools", "metrics", "slotstore"]))
    agent = "\n\n".join(
        f"<tool_result>\n{tool}\n</tool_result>\nassistant: turn {i}, "
        f"checking {r.choice(words)} before the next edit." for i in range(8))

    # A file re-sent with one hunk changed, four times — the edit loop.
    body = [f"def step_{i}(state, budget):\n    return state.advance({i}, budget)"
            for i in range(24)]
    edits = []
    for turn in range(4):
        b = list(body)
        b[turn * 5] = (f"def step_{turn * 5}(state, budget):  # turn {turn}\n"
                       f"    return state.advance({turn * 5}, budget * 2)")
        edits.append("\n\n".join(b))
    code = "\n\n---\n\n".join(edits)

    out = {"random-tokens": None, "agent-transcript": agent, "code-edit": code}
    if prose is not None:
        out["prose"] = prose
    return out


def tokenize(args, texts: dict[str, str | None]):
    """Ids per class. `random-tokens` is generated directly at the id level —
    tokenizing "random text" does not produce random IDS (the tokenizer maps
    it onto the same few thousand common pieces, which is how an earlier
    draft of this file measured 4.95 proposals/token for its own "noise"
    class and believed it)."""
    tok = None
    if not args.synthetic:
        try:
            from transformers import AutoTokenizer

            tok = AutoTokenizer.from_pretrained(args.model)
        except Exception as err:  # noqa: BLE001 — a bench must still produce numbers
            print(f"[ngram] no tokenizer for {args.model} ({type(err).__name__}: "
                  f"{err}) — text classes fall back to synthetic id streams",
                  file=sys.stderr)
    r = random.Random(args.seed)
    vocab = tok.vocab_size if tok is not None else args.vocab
    out = {}
    for name, text in texts.items():
        if name == "random-tokens":
            out[name] = [r.randrange(vocab) for _ in range(args.noise_tokens)]
        elif tok is not None:
            out[name] = tok(text)["input_ids"]
        else:
            out[name] = _synthetic(name, args)
    return (tok.name_or_path if tok is not None else None), out


def _synthetic(name: str, args) -> list[int]:
    """Stand-ins with the right SHAPE for a box with no tokenizer: a repeated
    block plus unique filler, more filler for the less repetitive classes."""
    r = random.Random(args.seed + len(name))
    block = [r.randrange(args.vocab) for _ in range(300)]
    filler = {"agent-transcript": 40, "code-edit": 120}.get(name, 300)
    out: list[int] = []
    for _ in range(8):
        out += block
        out += [r.randrange(args.vocab) for _ in range(filler)]
    return out


def measure(ids: list[int], tokens: int, hi: int, lo: int) -> dict:
    """Replay the id stream one token at a time, exactly as a generation
    feeds the proposer, and time the whole replay so the per-token cost of
    the rolling index is a measured number and not an assurance."""
    p = ngram.NgramProposer(tokens=tokens, hi=hi, lo=lo)
    t0 = time.perf_counter()
    for t in ids:
        p.append(t)
        p.propose(tokens)
    wall = time.perf_counter() - t0
    s = p.stats()
    return {"tokens": len(ids), "asked": s["asked"], "matched": s["matched"],
            "proposed": s["proposed"], "hit_rate": s["hit_rate"],
            "mean_match_n": s["mean_match_n"],
            "proposals_per_token": round(s["proposed"] / s["asked"], 4) if s["asked"] else 0.0,
            "indexed_ngrams": s["indexed"],
            "index_us_per_token": round(1e6 * wall / max(1, len(ids)), 2)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B",
                    help="tokenizer to measure with (weights are NOT loaded)")
    ap.add_argument("--synthetic", action="store_true",
                    help="skip the tokenizer; use synthetic id streams")
    ap.add_argument("--vocab", type=int, default=151936,
                    help="vocabulary size for the synthetic streams")
    ap.add_argument("--prose", default="README.md",
                    help="a file of real prose for the `prose` class")
    ap.add_argument("--chars", type=int, default=12000)
    ap.add_argument("--noise-tokens", type=int, default=4000)
    ap.add_argument("--tokens", type=int, default=ngram.DEFAULT_TOKENS)
    ap.add_argument("--max", type=int, default=ngram.DEFAULT_MAX)
    ap.add_argument("--min", type=int, default=ngram.DEFAULT_MIN)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--json")
    args = ap.parse_args()

    src, streams = tokenize(args, prompt_classes(args))
    report = {"tokenizer": src or f"synthetic(vocab={args.vocab})",
              "num_speculative_tokens": args.tokens,
              "prompt_lookup_max": args.max, "prompt_lookup_min": args.min,
              "classes": {}, "sweep": {}}
    print(f"[ngram] tokenizer: {report['tokenizer']}; "
          f"num_speculative_tokens={args.tokens} prompt_lookup_max={args.max} "
          f"prompt_lookup_min={args.min}")
    for name, ids in streams.items():
        m = measure(ids, args.tokens, args.max, args.min)
        report["classes"][name] = m
        print(f"  {name:17s} {m['tokens']:6d} tok  "
              f"{m['proposals_per_token']:6.3f} proposals/token  "
              f"hit {m['hit_rate']:6.1%}  mean n={m['mean_match_n']:.2f}  "
              f"{m['indexed_ngrams']:6d} indexed  "
              f"{m['index_us_per_token']:.2f} us/token")

    # THE SWEEP (module docstring): prompt_lookup_min against every class.
    print(f"\n[ngram] prompt_lookup_min sweep (max={args.max}, "
          f"num_speculative_tokens={args.tokens}) — proposals/token, hit rate")
    print("  " + "min".ljust(6) + "".join(n[:17].ljust(24) for n in streams))
    for lo in range(1, args.max + 1):
        row = {}
        for name, ids in streams.items():
            m = measure(ids, args.tokens, args.max, lo)
            row[name] = {"proposals_per_token": m["proposals_per_token"],
                         "hit_rate": m["hit_rate"],
                         "indexed_ngrams": m["indexed_ngrams"]}
        report["sweep"][lo] = row
        print("  " + str(lo).ljust(6) + "".join(
            f"{row[n]['proposals_per_token']:8.3f}  ({row[n]['hit_rate']:5.1%})     "
            for n in streams))

    fails = []
    noise = report["sweep"].get(1, {}).get("random-tokens", {}).get("proposals_per_token")
    agent = report["classes"].get("agent-transcript", {}).get("proposals_per_token")
    if noise is not None and noise > NOISE_MAX:
        fails.append(f"random-tokens proposes {noise:.3f}/token at min=1 "
                     f"(> {NOISE_MAX}): the proposer is inventing matches where "
                     "there provably are none")
    for lo in range(2, args.max + 1):
        n2 = report["sweep"].get(lo, {}).get("random-tokens", {}).get("proposals_per_token")
        if n2:
            fails.append(f"random-tokens proposes {n2:.3f}/token at min={lo}: "
                         "a >=2-gram match in uniform ids is an index bug")
    if agent is not None and agent < AGENT_MIN:
        fails.append(f"agent-transcript proposes {agent:.3f}/token (< {AGENT_MIN}): "
                     "the case this feature exists for is not being served")
    print()
    for f in fails:
        print(f"  FAIL: {f}")
    print("PASS: silent where there is nothing to find, loud on a re-sent "
          "tool result" if not fails else f"FAIL: {len(fails)} criteria")
    if args.json:
        report["verdict"] = "PASS" if not fails else "FAIL"
        with open(args.json, "w") as f:
            json.dump(report, f, indent=1)
        print(f"wrote {args.json}")
    return 1 if fails else 0


if __name__ == "__main__":
    import os as _os

    code = 1
    try:
        code = main()
    except Exception:
        import traceback

        traceback.print_exc()
    sys.stdout.flush()
    sys.stderr.flush()
    _os._exit(code)  # os._exit discipline (AGENTS.md): never trust $? alone
