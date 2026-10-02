"""On-disk prefix slots verify: warm-vs-RESTORED logits on the REAL model.

Sibling to bench/prefix_slots_verify.py, which asks whether a slot that came
back after two other conversations gets the same computation it would have got
cold. This one asks the question the cold tier adds — does a slot that came
back after the PROCESS DIED get the same computation as the live slot it was
copied from? — and it asks it in logits, because that is where an answer
arrives before rounding hides it.

The claim is stronger than the prefix cache's, and deliberately so. A warm prefix reused in
RAM is "same computation modulo kernel reduction order" (chunked-prefill
class: the suffix batches differently than a cold full prefill). A RESTORED
prefix is the same tensors, moved to disk and back: the row it feeds must be
BIT-IDENTICAL to the live slot's, not merely argmax-identical. So this bench
reports both, and fails on either.

  arm `live`      one engine, one conversation, N turns, cold tier ON. Turn 1
                  builds the slot; every later turn extends it in RAM. The
                  prefill row of each turn is tapped and kept.
  arm `restored`  a SECOND engine over the SAME loaded model, whose slots
                  start empty and whose store is indexed from the directory
                  alone — the closest one process gets to a restart without
                  paying for a second model load. It replays the live arm's
                  exact messages. Turn 2 onward must RESTORE from disk (or
                  from the live arm's own eviction) and its prefill row must
                  match the live arm's byte for byte.

A real restart is the systemctl gate; this file is the one that
can run in a single GPU window and points at the same claim.

MTP is pinned OFF (DRINKME_SPEC=off), same reason as prefix_slots_verify.py: the
draft path forwards through the inner text model, so the tap would see no
prefill row and the instrument would compare None against None and pass.

THINKING AND THE HISTORY RE-RENDER: pass --no-think for the 27B, leave the
default for the 8B. The reasoning is prefix_slots_verify.py's docstring, in
full; getting it wrong is not a numerics failure, it is `cache False` on every
turn — an instrument that never exercised the claim.

Writes verification/slot_restore_<model>_<host>_<date>.json.
Run: uv run --no-sync python bench/slot_restore_verify.py --pack DIR [--no-think]
     (--no-sync is not optional here — see AGENTS.md)
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import socket
import sys
import time

import torch

DEFAULT_PACK = os.path.expanduser(
    "~/.cache/drinkme/packs/Qwen--Qwen3.8-27B@1d4bf0f2ff60")


def _system(rules: int = 200) -> str:
    """An agent-sized system prompt — long enough to cross several 256-token
    block boundaries, which is what makes the chain walk do any work at all."""
    return " ".join(
        f"Handbook section {i}: when the user mentions topic-{i}, answer with "
        f"a numbered plan citing section {(i * 7) % 101}, tone plain."
        for i in range(rules))


USERS = [
    "Summarize your rules for topic-3, two sentences.",
    "Now topic-14, and one difference from topic-3.",
    "Which section number applies to topic-50? One line.",
    "List the section numbers you have cited so far.",
]


def tap(model):
    """Keep the PREFILL row — the logits the first sampled token of a turn
    comes from, which is the row every reused prefix feeds. Returns (restore,
    box); box[0] is the last prefill row seen."""
    box = [None]
    real = model.forward

    def forward(*a, **kw):
        out = real(*a, **kw)
        logits = getattr(out, "logits", None)
        if logits is not None and logits.shape[1] >= 1:
            ids = kw.get("input_ids", a[0] if a else None)
            if ids is not None and ids.shape[-1] > 1:  # a prefill, not a step
                box[0] = logits[0, -1].detach().float().cpu().clone()
        return out

    model.forward = forward
    return (lambda: setattr(model, "forward", real)), box


def run_arm(eng, box, turns: int, max_tokens: int, replay=None):
    """One arm's turns. With `replay`, the assistant turns come from the other
    arm's transcript, so both engines see IDENTICAL inputs per turn — the A/B
    condition."""
    from drinkme.serving.engine import GenerationRequest, SampleParams, complete

    out, history = [], [{"role": "system", "content": _system()}]
    for i in range(turns):
        history.append({"role": "user", "content": USERS[i % len(USERS)]})
        box[0] = None
        t0 = time.perf_counter()
        res = complete(eng, GenerationRequest(
            list(history), SampleParams(temperature=0.0, max_tokens=max_tokens)))
        row = box[0]
        out.append({"turn": i, "prompt_tokens": res.prompt_tokens,
                    "cached_tokens": res.cached_tokens, "text": res.text,
                    "wall_s": round(time.perf_counter() - t0, 3),
                    "argmax": None if row is None else int(row.argmax()),
                    "row": row})
        reply = (replay[i]["text"] if replay else res.text)
        history.append({"role": "assistant", "content": reply})
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", default=DEFAULT_PACK)
    ap.add_argument("--ctx", type=int, default=32768)
    ap.add_argument("--turns", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=48)
    ap.add_argument("--slots", type=int, default=2)
    ap.add_argument("--no-think", action="store_true",
                    help="enable_thinking=False — REQUIRED for the 27B, see "
                         "bench/prefix_slots_verify.py's docstring")
    ap.add_argument("--slot-dir", default=None,
                    help="cold tier root (default: a scratch dir this run "
                         "creates and removes; NEVER the live server's)")
    args = ap.parse_args()

    pack = os.path.expanduser(args.pack)
    if not os.path.isdir(pack):
        sys.exit(f"[slots] no such pack: {pack}")
    scratch = args.slot_dir or os.path.join(
        os.path.expanduser("~/.cache/drinkme"), f"slots-verify-{os.getpid()}")
    os.environ["DRINKME_SPEC"] = "off"
    os.environ["DRINKME_PREFIX_SLOTS"] = str(args.slots)
    os.environ["DRINKME_SLOT_DIR"] = scratch

    from drinkme.serve import build_engine
    from drinkme.serving.engines import HFEngine

    meta = json.load(open(os.path.join(pack, "meta.json")))
    print(f"[slots] loading {meta['hfRepo']} (meanBpw {meta['meanBpw']}) ...",
          flush=True)
    t0 = time.time()
    live = build_engine(meta["hfRepo"], meta["revision"], pack, stock=False,
                        ctx=args.ctx)
    print(f"[slots] loaded in {time.time() - t0:.0f}s; slot dir {scratch}",
          flush=True)
    if args.no_think:
        live.template_kwargs = dict(live.template_kwargs or {},
                                    enable_thinking=False)

    undo, box = tap(live.model)
    try:
        live_turns = run_arm(live, box, args.turns, args.max_tokens)
        n = live.persist_slots()
        print(f"[slots] persisted {n} live slot(s)", flush=True)

        # A SECOND engine over the SAME model: empty slots, and a store it
        # indexes from the directory alone. Everything the first arm knows is
        # in a file or it is not known.
        cold = HFEngine(live.model, live.tok, model_id=live.model_id,
                        arm=live.arm, meta=live.meta, ctx=live.ctx,
                        template_kwargs=live.template_kwargs,
                        prefix_slots=args.slots, gen_defaults=live.gen_defaults)
        restored_turns = run_arm(cold, box, args.turns, args.max_tokens,
                                 replay=live_turns)
    finally:
        undo()

    rows = []
    ok = True
    for a, b in zip(live_turns, restored_turns):
        same_text = a["text"] == b["text"]
        same_argmax = a["argmax"] == b["argmax"]
        delta = None
        if a["row"] is not None and b["row"] is not None:
            delta = float((a["row"] - b["row"]).abs().max())
        exact = delta == 0.0
        # turn 0 is cold on both arms by construction; from turn 1 the
        # restored arm must have come off the tier
        used_tier = b["cached_tokens"] > 0
        if a["turn"] > 0 and not (same_text and same_argmax and exact and used_tier):
            ok = False
        rows.append({"turn": a["turn"], "prompt_tokens": a["prompt_tokens"],
                     "live_cached": a["cached_tokens"],
                     "restored_cached": b["cached_tokens"],
                     "same_text": same_text, "same_argmax": same_argmax,
                     "max_abs_delta": delta, "bit_identical": exact,
                     "live_wall_s": a["wall_s"], "restored_wall_s": b["wall_s"]})
        print(f"[slots] turn {a['turn']}: prompt {a['prompt_tokens']} · "
              f"cached live {a['cached_tokens']} / restored {b['cached_tokens']} · "
              f"text {'==' if same_text else 'DIFFERS'} · "
              f"argmax {'==' if same_argmax else 'DIFFERS'} · "
              f"max|delta| {delta} · live {a['wall_s']}s "
              f"restored {b['wall_s']}s", flush=True)

    out = {"tool": "slot_restore_verify", "date": time.strftime("%Y-%m-%d"),
           "host": socket.gethostname(), "platform": platform.platform(),
           "torch": torch.__version__, "pack": pack,
           "hfRepo": meta["hfRepo"], "ctx": args.ctx, "slots": args.slots,
           "no_think": args.no_think, "turns": rows, "pass": ok}
    os.makedirs("verification", exist_ok=True)
    name = meta["hfRepo"].replace("/", "--")
    path = os.path.join("verification",
                        f"slot_restore_{name}_{out['host']}_{out['date']}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[slots] wrote {path}", flush=True)
    if args.slot_dir is None:
        shutil.rmtree(scratch, ignore_errors=True)
    # exit code alone is not trustworthy from a GPU process on this box
    # (AGENTS.md); the verdict is this line.
    print(f"[slots] VERDICT: {'PASS' if ok else 'FAIL'}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
