"""Multi-slot prefix cache verify: warm-vs-cold LOGITS on the REAL model.

Sibling to prefix_cache_verify.py, which stays the ruler for the single-slot
claim over the wire. This one asks the question multi-slot adds — does a
conversation that comes back after two OTHER conversations get the same
computation it would have got cold? — and it asks it in logits rather than
text, because that is where an answer arrives before rounding hides it.

One process, one loaded model, three engines over it (1 slot, 3 slots, and
the same 3-slot engine again for the eviction arm), and a tap on the model
that keeps the PREFILL row — the logits the first sampled token comes from,
which is the row every reused prefix feeds. Each arm then replays its own
recorded messages with the prefix cache OFF (per-request StaticCache, full
prefill from position 0 — computation-identical to the pre-cache code) and
the two rows are compared. Same weights, same prompts, same sampler: warm
vs cold here is exactly reuse vs no reuse.

  arm `single`  one slot, one conversation, three turns. The shipped
                behaviour, unchanged by the prefix cache: every turn extends, so every
                turn is a split prefill. argmax must match cold.
  arm `slots3`  three slots, conversations A/B/C round-robin. Turn 1 of each
                is cold (nothing to match); turns 2 and 3 must REUSE — on
                one slot they could not, because the other two conversations
                would have taken it — and must still match cold argmax.
  arm `evict`   a fourth conversation D arrives. It takes the least recently
                used slot (A's), so B and C stay warm and A's next turn is
                cold and merely slower. Correctness is the claim; the wall
                times say "merely".

THINKING AND THE HISTORY RE-RENDER (measured on the GPU: the 27B arm showed `cache False` on every warm turn while the
8B reused fine). This bench drives the engine directly and feeds each reply
back as plain assistant `content`, the way a client without a think channel
would. Whether that re-rendered history EXTENDS the ids the model actually
generated is a property of the family's chat template, and the two families
are mirror images (checked token-exact through template.build_prompt):
  - Qwen3.8 (the 27B): the generation prompt ends INSIDE an open think block
    (`<|im_start|>assistant\n<think>\n`), and a prior assistant turn with plain
    content re-renders with an EMPTY one (`<think>\n\n</think>\n\n`) — so with
    thinking on, history never extends unless the reasoning is passed back as
    `reasoning_content` (preserved thinking — which serving/http.py does at
    line ~350, and which is why prefix_cache_verify.py reuses over the wire).
    With `enable_thinking=False` the empty block is in BOTH renders: extends.
  - Qwen3 (the 8B): the generation prompt has NO forced think block and prior
    turns re-render with the think block STRIPPED — plain replies extend with
    thinking on (the default), and `enable_thinking=False` is the shape that
    does NOT extend (the prompt got an empty block the re-render drops).
So: pass --no-think for the 27B; leave the default for the 8B. Getting this
wrong is not a numerics failure — it is `cache False` with identical logits,
i.e. the instrument never exercised the claim.

MTP is pinned OFF (DRINKME_SPEC=off). The draft path forwards through the inner
text model rather than the CausalLM wrapper (mtp.forward_with_hidden), so the
tap would see no prefill row at all and this instrument would compare None
against None and call it a pass. Run bench/mtp_gpu_acceptance.py for that
axis; this one is about the cache.

Writes verification/prefix_slots_<model>_<host>_<date>.json.
Run: uv run --no-sync python bench/prefix_slots_verify.py [--pack DIR]
     (--no-sync is not optional here — see AGENTS.md)
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import socket
import sys
import time

import torch

# Same default as prefix_cache_verify.py: the 8B reproduces a settled
# receipt from a bare invocation. The HYBRID is where slots earn their keep
# (its DeltaNet state is the part that cannot be rewound), so the 27B is the
# run that matters — pass --pack.
DEFAULT_PACK = os.path.expanduser("~/.cache/drinkme/packs/Qwen--Qwen3-8B@b968826d9c46")


def _system(name: str, rules: int = 120) -> str:
    """An ~agent-sized system prompt that diverges from every other one at
    its FIRST content token: switching conversations must be a genuine
    non-extends event, not a long shared prefix with a different tail."""
    return " ".join(
        f"Handbook {name}. Rule {i}: when the user mentions topic-{i} in "
        f"thread {name}, answer with a numbered plan citing section "
        f"{(i * 7 + ord(name)) % 101}, keeping the tone plain."
        for i in range(rules))


def _users(name: str) -> list[str]:
    return [
        f"Summarize your rules for topic-3 in thread {name}, two sentences.",
        f"Now topic-14 in thread {name}, and one difference from topic-3.",
        f"Which section number applies to topic-50 in thread {name}? One line.",
        f"List the section numbers you have cited in thread {name}.",
    ]


CONVS = {n: (_system(n), _users(n)) for n in "ABCD"}

# (conversation, which user turn, what the cache should do). The expectation
# is part of the schedule because it is the claim: a reader should be able to
# see what each arm is asserting without reconstructing the LRU order.
SINGLE = [("A", 0, "cold"), ("A", 1, "warm"), ("A", 2, "warm")]
INTERLEAVED = [("A", 0, "cold"), ("B", 0, "cold"), ("C", 0, "cold"),
               ("A", 1, "warm"), ("B", 1, "warm"), ("C", 1, "warm"),
               ("A", 2, "warm"), ("B", 2, "warm"), ("C", 2, "warm")]
# D takes the least recently used slot, which after INTERLEAVED is A's (A ran
# first of the last three). B and C are asked BEFORE A comes back, because
# with four live threads over three slots LRU means somebody is always the
# next victim: A returning cold evicts D in turn. That cascade is the design,
# not a defect — the claim is that the eviction is correct and merely slower.
EVICT = [("D", 0, "cold"),   # a new thread: nothing to match, takes A's slot
         ("B", 3, "warm"),   # untouched by that eviction
         ("C", 3, "warm"),   # untouched by that eviction
         ("A", 3, "cold")]   # A really was the one evicted


class ModelTap:
    """Wraps the model so the bench can read the PREFILL logits — the row the
    first sampled token comes from — without reaching inside generate().

    Every forward passes through; only the first of each armed window is
    kept, which is the prefill (warm: prefix reused, suffix computed; cold:
    the whole prompt). Delegates everything else to the model, so an HFEngine
    built over the tap is the engine built over the model."""

    def __init__(self, model):
        self._m = model
        self.n = 0
        self.first = None

    def arm(self) -> None:
        self.n, self.first = 0, None

    def __call__(self, *a, **kw):
        out = self._m(*a, **kw)
        self.n += 1
        if self.n == 1:
            self.first = out.logits[0, -1].detach().float().cpu().clone()
        return out

    def __getattr__(self, k):
        return getattr(self._m, k)


def engine_over(model, tok, proto, slots: int):
    """Another HFEngine over an already-loaded model, with its own slots.
    `proto` is the engine build_engine returned — everything but the slot
    count is copied from it, so the arms differ in ONE thing."""
    from drinkme.serving.engines import HFEngine

    os.environ["DRINKME_PREFIX_SLOTS"] = str(slots)
    try:
        return HFEngine(model, tok, model_id=proto.model_id, arm=proto.arm,
                        meta=proto.meta, ctx=proto.ctx,
                        template_kwargs=proto.template_kwargs, mtp_head=None,
                        gen_defaults=proto.gen_defaults)
    finally:
        os.environ.pop("DRINKME_PREFIX_SLOTS", None)


def _generate(eng, msgs, max_tokens: int):
    """One greedy generation through the seam (engine.complete), collected."""
    from drinkme.serving.engine import GenerationRequest, SampleParams, complete

    return complete(eng, GenerationRequest(
        msgs, SampleParams(temperature=0.0, max_tokens=max_tokens)))


def run_schedule(eng, tap, schedule, convs, max_tokens: int, hists=None,
                 label: str = "warm"):
    """Run (conversation, turn, expectation) triples in order on ONE engine,
    each conversation carrying its own history — the agent shape: the whole
    thread is re-sent every turn. Returns one record per turn, each holding
    the exact messages it ran, so the cold arm replays inputs rather than
    rebuilding them."""
    hists = {} if hists is None else hists
    out = []
    for name, turn, expect in schedule:
        system, users = convs[name]
        hist = hists.setdefault(name, [{"role": "system", "content": system}])
        hist.append({"role": "user", "content": users[turn]})
        msgs = list(hist)
        tap.arm()
        t0 = time.perf_counter()
        res = _generate(eng, msgs, max_tokens)
        wall = time.perf_counter() - t0
        hist.append({"role": "assistant", "content": res.text})
        out.append({"conv": name, "turn": turn + 1, "expect": expect,
                    "messages": msgs, "text": res.text,
                    "prompt_tokens": res.prompt_tokens,
                    "cached_tokens": res.cached_tokens,
                    "wall": round(wall, 3), "logits": tap.first})
        print(f"  [{label}] {name}{turn + 1}: prompt={res.prompt_tokens} "
              f"cached={res.cached_tokens} ({expect}) wall={wall:.2f}s",
              flush=True)
    return out, hists


def run_cold(eng, tap, records, max_tokens: int, label: str = "cold"):
    """The reference: the SAME messages with the prefix cache off, which is a
    per-request StaticCache and a full prefill from position 0 — what the
    engine did before any of this existed."""
    was = eng._reuse
    eng._reuse = False
    try:
        out = []
        for rec in records:
            tap.arm()
            t0 = time.perf_counter()
            res = _generate(eng, rec["messages"], max_tokens)
            wall = time.perf_counter() - t0
            out.append({"text": res.text, "wall": round(wall, 3),
                        "cached_tokens": res.cached_tokens, "logits": tap.first})
            print(f"  [{label}] {rec['conv']}{rec['turn']}: wall={wall:.2f}s",
                  flush=True)
        return out
    finally:
        eng._reuse = was


def compare(warm, cold):
    """Per-turn verdict. The claim is argmax identity and text identity; the
    max abs delta is REPORTED whatever it is, because a reused prefix batches
    its suffix differently than a cold full prefill and pretending that is
    bit-exact would be the one lie this file exists to prevent."""
    rows = []
    for w, c in zip(warm, cold):
        if w["logits"] is None or c["logits"] is None:
            rows.append({"conv": w["conv"], "turn": w["turn"],
                         "logits_seen": False})
            continue
        delta = (w["logits"] - c["logits"]).abs().max().item()
        rows.append({
            "conv": w["conv"], "turn": w["turn"], "logits_seen": True,
            "expect": w["expect"], "cached_tokens": w["cached_tokens"],
            "prompt_tokens": w["prompt_tokens"],
            "cache_as_expected": (w["cached_tokens"] > 0) == (w["expect"] == "warm"),
            "max_abs_delta": delta,
            "argmax_identical": int(w["logits"].argmax()) == int(c["logits"].argmax()),
            "text_identical": w["text"] == c["text"],
            "wall_warm_s": w["wall"], "wall_cold_s": c["wall"],
        })
    return rows


def verdict(rows):
    return {
        "turns": len(rows),
        "all_argmax_identical": all(r.get("argmax_identical") for r in rows),
        "all_text_identical": all(r.get("text_identical") for r in rows),
        "all_cache_as_expected": all(r.get("cache_as_expected") for r in rows),
        "max_abs_delta": max((r.get("max_abs_delta", 0.0) for r in rows),
                             default=0.0),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-tokens", type=int, default=48)
    ap.add_argument("--ctx", type=int, default=32768)
    ap.add_argument("--pack", default=DEFAULT_PACK,
                    help="pack dir to verify. The 8B default re-answers a "
                         "settled question; point it at the 27B hybrid pack "
                         "to exercise the DeltaNet state a slot carries.")
    ap.add_argument("--no-think", action="store_true",
                    help="render with enable_thinking=False. REQUIRED for the "
                         "Qwen3.8 27B (its template puts the model inside an "
                         "open think block and re-renders history with an "
                         "empty one, so thinking-on history never extends "
                         "here); leave OFF for the Qwen3 8B, whose template "
                         "is the mirror image. See the module docstring.")
    args = ap.parse_args()
    pack = os.path.expanduser(args.pack)
    if not os.path.isdir(pack):
        sys.exit(f"[slots] no such pack: {pack}")

    # before build_engine: the engine reads both of these at construction
    os.environ["DRINKME_SPEC"] = "off"          # see the module docstring
    os.environ["DRINKME_PREFIX_SLOTS"] = "1"  # the loaded engine is the ruler
    # on-disk prefix slots: the cold tier would hand a "cold" arm a stale
    # prefix off an SSD, which is an instrument measuring nothing. setdefault, so a
    # deliberate DRINKME_SLOT_DIR on the command line still wins.
    os.environ.setdefault("DRINKME_SLOT_DIR", "off")

    from drinkme.serve import build_engine

    meta = json.load(open(os.path.join(pack, "meta.json")))
    print(f"[slots] loading {meta['hfRepo']}@{str(meta['revision'])[:12]} "
          f"(meanBpw {meta['meanBpw']}) ...", flush=True)
    t0 = time.time()
    proto = build_engine(meta["hfRepo"], meta["revision"], pack, stock=False,
                         ctx=args.ctx)
    print(f"[slots] loaded in {time.time() - t0:.0f}s", flush=True)
    if str(proto.device) == "cpu" and not os.environ.get("VERIFY_ALLOW_CPU"):
        # a CPU verify already produced a day of mislabeled numbers once
        sys.exit("[slots] REFUSING to measure on CPU (VERIFY_ALLOW_CPU=1 to override)")
    if proto.mtp_head is not None:
        sys.exit("[slots] MTP head loaded despite DRINKME_SPEC=off — the tap "
                 "would miss the prefill row; refusing to report a pass")
    if args.no_think:
        # engine_over copies template_kwargs from the proto, so every arm
        # renders the same way; the family asymmetry is in the docstring.
        proto.template_kwargs = {**(proto.template_kwargs or {}),
                                 "enable_thinking": False}
        print("[slots] rendering with enable_thinking=False (--no-think)")

    tap = ModelTap(proto.model)
    results = {}

    print("[slots] arm `single` — one slot, one conversation (the default engine)")
    one = engine_over(tap, proto.tok, proto, 1)
    warm, _ = run_schedule(one, tap, SINGLE, CONVS, args.max_tokens, label="1slot")
    results["single"] = compare(warm, run_cold(one, tap, warm, args.max_tokens))

    print("[slots] arm `slots3` — three slots, A/B/C interleaved")
    three = engine_over(tap, proto.tok, proto, 3)
    warm3, hists = run_schedule(three, tap, INTERLEAVED, CONVS, args.max_tokens,
                                label="3slot")
    results["slots3"] = compare(warm3, run_cold(three, tap, warm3, args.max_tokens))

    print("[slots] arm `evict` — a fourth conversation takes the LRU slot")
    warm4, _ = run_schedule(three, tap, EVICT, CONVS, args.max_tokens,
                            hists=hists, label="evict")
    results["evict"] = compare(warm4, run_cold(three, tap, warm4, args.max_tokens))

    report = {
        "date": time.strftime("%Y-%m-%d %H:%M %Z"),
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "model": {k: meta[k] for k in ("hfRepo", "revision", "meanBpw")},
        "device": str(proto.device),
        "ctx": args.ctx,
        "max_tokens": args.max_tokens,
        "enable_thinking": False if args.no_think else "template default",
        "slot_bytes": {"fixed": one._slot_fixed_bytes,
                       "rings": one._slot_ring_bytes,
                       "per_token": one._slot_token_bytes,
                       "at_ctx": (None if one._slot_fixed_bytes is None else
                                  one._slot_fixed_bytes + one._slot_ring_bytes
                                  + one._slot_token_bytes * one.ctx)},
        "arms": {name: {"verdict": verdict(rows), "turns": rows}
                 for name, rows in results.items()},
    }
    ok = all(v["all_argmax_identical"] and v["all_text_identical"]
             and v["all_cache_as_expected"]
             for v in (report["arms"][a]["verdict"] for a in report["arms"]))
    report["pass"] = ok
    out = os.path.join(os.path.dirname(__file__), "..", "verification",
                       # the model in the filename: two packs verified on one
                       # day must not overwrite each other
                       f"prefix_slots_{meta['hfRepo'].split('/')[-1]}"
                       f"_{socket.gethostname()}_{time.strftime('%Y-%m-%d')}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    for name in results:
        v = report["arms"][name]["verdict"]
        print(f"  {name}: argmax {v['all_argmax_identical']} · text "
              f"{v['all_text_identical']} · cache {v['all_cache_as_expected']} "
              f"· max|delta| {v['max_abs_delta']:.3e} over {v['turns']} turns")
    print(f"[slots] {'PASS' if ok else 'FAIL'} — wrote {os.path.normpath(out)}")
    return 0 if ok else 2


if __name__ == "__main__":
    # os._exit, NOT a bare return: on this box's TheRock ROCm torch an atexit
    # handler calls _exit(0) once HIP has initialised, so a verify that
    # REFUSED to run would otherwise report success (prefix_cache_verify.py
    # carries the same footer).
    import os as _os
    try:
        _code = main() or 0
    except SystemExit as e:
        _code = e.code if isinstance(e.code, int) else (1 if e.code else 0)
        if e.code and not isinstance(e.code, int):
            print(e.code, file=sys.stderr)
    sys.stderr.flush()
    sys.stdout.flush()
    _os._exit(_code)
