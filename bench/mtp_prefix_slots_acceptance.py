#!/usr/bin/env python3
"""MTP depth>0 x N>1 prefix slots — the interaction this bench exists to have
MEASURED, sibling to mtp_gpu_acceptance.py (which pins DRINKME_PREFIX_SLOTS=0
and never turns slots on) and to prefix_slots_verify.py (which pins
DRINKME_SPEC=off because the draft path forwards through the inner text model,
so the logits tap it uses would see nothing).

THE QUESTION neither sibling answers alone: with N=3 prefix slots live, a
draft/verify cycle usually begins from a WARM (prefix-reused) prefill instead
of a cold one. A warm prefill only hands the head the suffix's hidden
states; the slot keeps the head's KV for the reused prefix
(mtp.HeadKV), and a slot just taken over cold, or restored from disk,
starts it at the reuse point, which costs acceptance, never correctness.
Does that acceptance cost show up, and does MTP-on still
agree with MTP-off (modulo a genuine near-tie) when slots are also
interleaving THREE conversations against each other (so some cycles start
from a slot that was just taken over cold, LRU-evicted from a different
thread)?

THE SCHEDULE mirrors prefix_slots_verify.py's INTERLEAVED case (three
conversations, A/B/C, round-robin, each turn resending its own full
history) run TWICE per model build — once with DRINKME_SPEC=off (serial,
the default decode, still with the same 3 slots so the SPEED comparison isolates
MTP's own contribution rather than mixing it with the slots' own TTFT win),
once with DRINKME_MTP_DEPTH=4 (the shipped default depth) — and reports per-turn
text agreement (informational — see below) and decode tok/s, at THIS
interleaved-with-slots setting instead of mtp_gpu_acceptance.py's
single-conversation one.

DRY RUN, on the toy (--toy, default; this is what actually RUNS in this
environment — no GPU, no checkpoint download): a genuine tiny hybrid
Qwen3_5ForCausalLM (test_serving_mtp.py's `toy` fixture shape) with a real
MTP head built from random weights, CPU, float32. Random weights make the
head a terrible drafter — acceptance near zero — which is fine: --toy proves
the HARNESS (interleaving + slot reuse + MTP + the identity/speed
accounting) runs end to end and produces a well-shaped receipt, not that
acceptance is good. It cannot dry-run the acceptance NUMBER, only the
plumbing that will carry it.

THE REAL RUN (--toy is not passed) needs a GPU and the packed Qwen3.8-27B
checkpoint, on a machine whose GPU is free (never against a server someone
relies on; docs/dev-environment.md, Running servers). The exact command:

    uv run --no-sync python bench/mtp_prefix_slots_acceptance.py \\
        --pack ~/.cache/drinkme/packs/Qwen--Qwen3.8-27B@1d4bf0f2ff60 --ctx 16384

TEXT AGREEMENT IS REPORTED, NEVER PASS/FAILED: token-exact
between speculation on and off is not a property this repo asserts — a
batched verify forward and a serial single-token one can round differently
in accumulation order, and a near-tie can flip an argmax for reasons that
have nothing to do with a bug (the same bar mtp_gpu_acceptance.py already
holds this comparison to; AGENTS.md). A clean run looks like `agreed` on
every turn, or a DIVERGED line with its first-divergence context for a human
to judge — either way this script exits 0, and a written receipt lands
under verification/ naming both knobs so it cannot be confused with either
single-axis sibling.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import socket
import sys
import time

N_SLOTS = 3
MAX_TOKENS = 48

CONVS = {
    "A": ("You are handbook-bot for thread A. Rule 1: cite section 12 for "
          "topic-alpha. Rule 2: cite section 40 for topic-beta.",
          ["Summarize your rules for topic-alpha, two sentences.",
           "Now topic-beta, and one difference from topic-alpha.",
           "List the section numbers you have cited so far."]),
    "B": ("You are handbook-bot for thread B. Rule 1: cite section 7 for "
          "topic-gamma. Rule 2: cite section 91 for topic-delta.",
          ["Summarize your rules for topic-gamma, two sentences.",
           "Now topic-delta, and one difference from topic-gamma.",
           "List the section numbers you have cited so far."]),
    "C": ("You are handbook-bot for thread C. Rule 1: cite section 55 for "
          "topic-epsilon. Rule 2: cite section 3 for topic-zeta.",
          ["Summarize your rules for topic-epsilon, two sentences.",
           "Now topic-zeta, and one difference from topic-epsilon.",
           "List the section numbers you have cited so far."]),
}
# round-robin: every conversation's turn 1 is cold (nothing to match), turns
# 2 and 3 must reuse — on ONE slot they could not, since the other two
# conversations would have taken it (prefix_slots_verify.py's same claim).
SCHEDULE = [(name, t) for t in range(3) for name in "ABC"]


def _generate(eng, msgs, max_tokens: int):
    """One greedy generation through the seam (engine.complete), collected."""
    from drinkme.serving.engine import GenerationRequest, SampleParams, complete

    return complete(eng, GenerationRequest(
        msgs, SampleParams(temperature=0.0, max_tokens=max_tokens)))


def run_schedule(eng, label: str):
    """One full pass over SCHEDULE, fresh histories. Returns per-turn records
    (conv, turn, text, wall_s, decode_toks_s, cached_tokens, prompt_tokens)."""
    hists: dict[str, list[dict]] = {}
    out = []
    for name, turn in SCHEDULE:
        system, users = CONVS[name]
        hist = hists.setdefault(name, [{"role": "system", "content": system}])
        hist.append({"role": "user", "content": users[turn]})
        msgs = list(hist)

        t0 = time.monotonic()
        res = _generate(eng, msgs, MAX_TOKENS)
        wall = time.monotonic() - t0
        hist.append({"role": "assistant", "content": res.text})
        dec = ((res.completion_tokens - 1) / wall) if wall > 0 and res.completion_tokens > 1 else 0.0
        rec = {"conv": name, "turn": turn + 1, "text": res.text,
              "wall_s": round(wall, 3), "decode_toks_per_s": round(dec, 2),
              "prompt_tokens": res.prompt_tokens, "cached_tokens": res.cached_tokens}
        out.append(rec)
        print(f"  [{label}] {name}{turn + 1}: prompt={res.prompt_tokens} "
              f"cached={res.cached_tokens} wall={wall:.2f}s "
              f"dec={rec['decode_toks_per_s']}tok/s", flush=True)
    return out


def compare(off, on):
    """Per-turn records, `agreed` reported informationally — never a
    pass/fail input (module docstring): text agreement between
    MTP-on and MTP-off is not a property this repo asserts."""
    rows = []
    for o, n in zip(off, on):
        assert (o["conv"], o["turn"]) == (n["conv"], n["turn"])
        agreed = o["text"] == n["text"]
        row = {"conv": o["conv"], "turn": o["turn"], "agreed": agreed,
              "off_decode_toks_per_s": o["decode_toks_per_s"],
              "on_decode_toks_per_s": n["decode_toks_per_s"],
              "on_cached_tokens": n["cached_tokens"],
              "on_prompt_tokens": n["prompt_tokens"]}
        if not agreed:
            div = next((i for i, (a, b) in enumerate(zip(o["text"], n["text"]))
                       if a != b), min(len(o["text"]), len(n["text"])))
            row["first_divergence_char"] = div
            row["off_context"] = o["text"][max(0, div - 40):div + 40]
            row["on_context"] = n["text"][max(0, div - 40):div + 40]
        rows.append(row)
    return rows


def _build_real(pack: str, ctx: int, stock: bool, mtp_depth: int):
    """The GPU path (not exercised in this environment): one engine build per
    arm, mirroring mtp_gpu_acceptance.py's build_engine call, with
    DRINKME_PREFIX_SLOTS held at N_SLOTS across BOTH arms (only the
    speculation env varies) so the speed comparison isolates MTP's own contribution rather
    than mixing it with the slots' own TTFT win. Both env vars are read once,
    at construction — a fresh engine per arm is required, same as
    mtp_gpu_acceptance.py's own arm toggling."""
    if mtp_depth:
        os.environ["DRINKME_MTP_DEPTH"] = str(mtp_depth)
        os.environ.pop("DRINKME_SPEC", None)
    else:
        os.environ["DRINKME_SPEC"] = "off"
    os.environ["DRINKME_PREFIX_SLOTS"] = str(N_SLOTS)
    from drinkme.serve import build_engine

    engine = build_engine("Qwen/Qwen3.8-27B", None, None if stock else pack,
                          stock=stock, ctx=ctx)
    engine.template_kwargs = {"enable_thinking": False}  # the thinking/cache-
    # extension hazard: keep the 27B history-extending — see
    # prefix_slots_verify.py's module docstring for
    # why thinking-on history never extends here.
    return engine


def _build_toy():
    """The dry-run path: a genuine tiny hybrid Qwen3_5 + a real (random-
    weight) MTP head, CPU, float32 — test_serving_mtp.py's `toy` fixture,
    condensed to a plain function since this is a script, not a test module.
    Proves the harness runs; says nothing about real acceptance (module
    docstring)."""
    import torch
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast
    from transformers.models.qwen3_5 import modeling_qwen3_5 as mod
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

    from drinkme.serving import mtp
    from drinkme.serving.engines import HFEngine

    # DeltaNet prefill's chunk kernel is triton-only (fla); swap in the pure
    # torch reference for CPU, same as tests/test_serving_mtp.py.
    fn = mod.torch_chunk_gated_delta_rule
    mod.torch_chunk_gated_delta_rule = getattr(fn, "__wrapped__", fn)

    cfg = Qwen3_5TextConfig(
        vocab_size=128, hidden_size=64, intermediate_size=128,
        num_hidden_layers=4, num_attention_heads=2, num_key_value_heads=1,
        head_dim=32, linear_key_head_dim=16, linear_value_head_dim=16,
        linear_num_key_heads=2, linear_num_value_heads=4,
        linear_conv_kernel_dim=4,
        layer_types=["linear_attention", "full_attention",
                     "linear_attention", "full_attention"],
        max_position_embeddings=2048,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0,
                         "mrope_section": [2, 1, 1], "mrope_interleaved": True,
                         "partial_rotary_factor": 0.25},
        tie_word_embeddings=False, eos_token_id=None, pad_token_id=None)
    torch.manual_seed(0)
    model = Qwen3_5ForCausalLM(cfg).eval().to(torch.float32)
    for p in model.parameters():
        p.requires_grad_(False)

    words = set()
    for system, users in CONVS.values():
        words.update(system.split())
        for u in users:
            words.update(u.split())
    vocab = {"<unk>": 0, "<pad>": 1, "</s>": 2}
    for w in sorted(words):
        if w not in vocab:
            vocab[w] = len(vocab)
    while len(vocab) < cfg.vocab_size:
        vocab[f"tok{len(vocab)}"] = len(vocab)
    backend = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
    backend.pre_tokenizer = Whitespace()
    tok = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="<unk>",
                                  pad_token="<pad>")
    tok.chat_template = ("{% for m in messages %}{{ m['role'] }} : {{ m['content'] }} "
                         "{% endfor %}{% if add_generation_prompt %}assistant :{% endif %}")

    torch.manual_seed(1)
    head = mtp.MTPHead(cfg, model)
    for name, p in list(head.named_parameters()):
        path, _, attr = name.rpartition(".")
        owner = head.get_submodule(path) if path else head
        owner._parameters[attr] = torch.nn.Parameter(
            torch.randn(p.shape, dtype=torch.float32) * 0.08, requires_grad=False)
    head.eval()
    mtp.install_deltanet_capture(model)

    os.environ["DRINKME_PREFIX_SLOTS"] = str(N_SLOTS)
    try:
        eng_on = HFEngine(model, tok, model_id="toy-mtp-slots", arm="test", meta={"resolvedRevision": "toy-mtp-slots"},
                          ctx=1024, mtp_head=head)
        eng_off = HFEngine(model, tok, model_id="toy-mtp-slots", arm="test", meta={"resolvedRevision": "toy-mtp-slots"},
                           ctx=1024, mtp_head=None)
    finally:
        os.environ.pop("DRINKME_PREFIX_SLOTS", None)
    return eng_off, eng_on


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--real", action="store_true",
                    help="the GPU path: build the real Qwen3.8-27B engine. "
                         "Default is the --toy dry run (CPU, no download); "
                         "the GPU command is in the module docstring.")
    ap.add_argument("--pack", default=os.path.expanduser(
        "~/.cache/drinkme/packs/Qwen--Qwen3.8-27B@1d4bf0f2ff60"))
    ap.add_argument("--ctx", type=int, default=16384)
    ap.add_argument("--stock", action="store_true")
    args = ap.parse_args()

    if args.real:
        print("[mtp-slots] building the REAL Qwen3.8-27B engine, arm OFF "
              f"(DRINKME_SPEC=off, DRINKME_PREFIX_SLOTS={N_SLOTS})...", flush=True)
        eng_off = _build_real(args.pack, args.ctx, args.stock, mtp_depth=0)
        off = run_schedule(eng_off, "off")
        print("[mtp-slots] building the REAL Qwen3.8-27B engine, arm ON "
              f"(DRINKME_MTP_DEPTH=4, DRINKME_PREFIX_SLOTS={N_SLOTS})...", flush=True)
        eng_on = _build_real(args.pack, args.ctx, args.stock, mtp_depth=4)
        on = run_schedule(eng_on, "on")
        meta = eng_on.meta
        device = str(eng_on.device)
    else:
        print(f"[mtp-slots] --toy dry run: tiny hybrid + random MTP head, "
              f"{N_SLOTS} prefix slots, CPU", flush=True)
        eng_off, eng_on = _build_toy()
        off = run_schedule(eng_off, "off")
        on = run_schedule(eng_on, "on")
        meta = {"toy": True}
        device = "cpu"

    rows = compare(off, on)
    n_agreed = sum(1 for r in rows if r["agreed"])
    for r in rows:
        if not r["agreed"]:
            print(f"[mtp-slots] DIVERGED {r['conv']}{r['turn']} at char "
                  f"{r['first_divergence_char']}: off={r['off_context']!r} "
                  f"on={r['on_context']!r} — informational, judge it yourself "
                  "(near-tie or real bug); this does not fail the run")
    reused = sum(1 for r in rows if r["on_cached_tokens"] > 0)
    report = {
        "date": time.strftime("%Y-%m-%d %H:%M %Z"),
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "toy": not args.real,
        "model": meta,
        "device": device,
        "mtp_depth": 4,
        "prefix_slots": N_SLOTS,
        "max_tokens": MAX_TOKENS,
        "schedule": SCHEDULE,
        "turns": rows,
        # informational only (module docstring) — never gates
        # the exit code
        "turns_agreed": f"{n_agreed}/{len(rows)}",
        "turns_that_reused_the_prefix_cache": reused,
    }
    out_dir = os.path.join(os.path.dirname(__file__), "..", "verification")
    os.makedirs(out_dir, exist_ok=True)
    kind = "toy" if not args.real else meta.get("hfRepo", "unknown").split("/")[-1]
    out = os.path.join(out_dir, f"mtp_prefix_slots_{kind}_{socket.gethostname()}_"
                       f"{time.strftime('%Y-%m-%d')}.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[mtp-slots] turns_agreed={n_agreed}/{len(rows)} · {reused}/{len(rows)} turns "
          f"reused the prefix cache — wrote {os.path.normpath(out)}")
    if not args.real:
        print("[mtp-slots] this was --toy: a PLUMBING proof, not an "
              "acceptance measurement. The real command (GPU, "
              "not run here):\n"
              "  uv run --no-sync python bench/mtp_prefix_slots_acceptance.py "
              "--real --pack ~/.cache/drinkme/packs/Qwen--Qwen3.8-27B@1d4bf0f2ff60 "
              "--ctx 16384")
    return 0


if __name__ == "__main__":
    # os._exit, not a bare return — the ROCm atexit scar every GPU-facing
    # bench in this repo carries (mtp_gpu_acceptance.py has none because it
    # never wraps main in a try; prefix_slots_verify.py's footer is the model
    # this follows, since --real will run on the same ROCm box eventually).
    try:
        _code = main() or 0
    except SystemExit as e:
        _code = e.code if isinstance(e.code, int) else (1 if e.code else 0)
    sys.stderr.flush()
    sys.stdout.flush()
    os._exit(_code)
