"""The slot-spy proof for context checkpoints (serving/ctx_checkpoints.py), on
the CPU: where each request's prompt parts from the slot the last one left,
and how much of it the engine reuses, before (--ctx-checkpoints 0) and after.

It spies on engines.pick_slot, as a GPU run over the real 31B can, but
over the REAL tokenizer and chat template of gemma-4-31B-it, Qwen3.8-27B and
Muse-Glimmer-30B, through drinkme's real HTTP dialects, on a tiny random
model of each family's layer kinds with its real vocabulary, sliding window
and generation config. The model's replies are SCRIPTED to what the real
models wrote on gfx1151, so every slot holds the ids the real model would
have left, and every re-render is the template's own. The prompts carry an
agent-sized system prompt, which puts each conversation past its model's
sliding window (gemma 1,024, Glimmer 2,048): the rings wrap, so a reuse is a
checkpoint's, not a rewind by length. Qwen3.5's DeltaNet state never rewinds.

Cases (--case, default all):
  gemma-off    two user turns, thinking off: the template drops the empty
               `<|channel>thought\\n<channel|>` the generation prompt ends in
  gemma-tools  the Messages tool loop, thinking off: after a tool response
               the model writes the empty pair itself, and the template
               drops it
  gemma-think  thinking on, the thinking block passed back: the template
               adds a newline before `<channel|>` the model did not write
  qwen-nl      Qwen3.8-27B, thinking off, a reply ending in "\\n" that the
               template trims
  glimmer      Muse-Glimmer's ` to=self` and ` to=user` channels sent back as
               the reply's text

A turn PASSes when it reuses everything its prompt shares with the slot up
to the previous prompt's end: min(common prefix, previous prompt's length).
The before run (checkpoints off) is the control: every such turn reused 0.

    HF_HUB_OFFLINE=1 DRINKME_NO_AUTO_DEPS=1 CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= \\
      PYTHONPATH=src .venv/bin/python bench/ctx_checkpoint_probe.py --out DIR

Needs the three checkpoints' tokenizer and config files in the HF cache (no
weights). Writes DIR/probe.json and prints one line per turn.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request

import torch

GEMMA = ("google/gemma-4-31B-it", "842da3794eaa0b77d5f08bae87a17459d91ff475")
QWEN = ("Qwen/Qwen3.8-27B", "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0")
GLIMMER = ("meta-models/Muse-Glimmer-30B", "a4e59da52a7bc87ae7251dd5545c0dd437c44b68")

SYSTEM = ("You are a careful operations assistant. " * 3 + "Rules: " +
          " ".join(f"rule {i}: answer from the deployments table and nothing else."
                   for i in range(220)))


def _snapshot(repo, rev):
    from drinkme.serving.checkpoint import snapshot_dir

    return snapshot_dir(repo, rev)


def _toy(repo, rev):
    """A tiny random model with the checkpoint's layer kinds, vocabulary and
    window, in float32, and its real generation config."""
    from transformers import AutoConfig

    real = AutoConfig.from_pretrained(repo, revision=rev, local_files_only=True)
    tc = real.get_text_config()
    torch.manual_seed(0)
    if tc.model_type in ("gemma4_text", "gemma4"):
        from transformers import Gemma4ForCausalLM, Gemma4TextConfig

        cfg = Gemma4TextConfig(
            vocab_size=tc.vocab_size, hidden_size=64, intermediate_size=128,
            num_hidden_layers=6, num_attention_heads=2, num_key_value_heads=1, head_dim=32,
            global_head_dim=32, num_global_key_value_heads=1,
            layer_types=["sliding_attention"] * 5 + ["full_attention"],
            sliding_window=tc.sliding_window, max_position_embeddings=8192,
            final_logit_softcapping=30.0, attention_k_eq_v=True,
            hidden_size_per_layer_input=0, vocab_size_per_layer_input=tc.vocab_size,
            tie_word_embeddings=True)
        model = Gemma4ForCausalLM(cfg)
    elif tc.model_type == "qwen3_5_text":
        from transformers.models.qwen3_5 import modeling_qwen3_5 as mod
        from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

        # fla's chunk kernel is triton-only: the torch reference on the CPU
        # (tests/test_serving_mtp.py's swap)
        fn = mod.torch_chunk_gated_delta_rule
        mod.torch_chunk_gated_delta_rule = getattr(fn, "__wrapped__", fn)
        cfg = Qwen3_5TextConfig(
            vocab_size=tc.vocab_size, hidden_size=64, intermediate_size=128,
            num_hidden_layers=4, num_attention_heads=2, num_key_value_heads=1, head_dim=32,
            linear_key_head_dim=16, linear_value_head_dim=16, linear_num_key_heads=2,
            linear_num_value_heads=4, linear_conv_kernel_dim=4,
            layer_types=["linear_attention", "full_attention"] * 2,
            max_position_embeddings=8192,
            rope_parameters={"rope_type": "default", "rope_theta": 10000.0,
                             "mrope_section": [2, 1, 1], "mrope_interleaved": True,
                             "partial_rotary_factor": 0.25},
            tie_word_embeddings=False)
        model = mod.Qwen3_5ForCausalLM(cfg)
    else:
        from transformers.models.muse_glimmer import (MuseGlimmerConfig,
                                                      MuseGlimmerForConditionalGeneration)

        cfg = MuseGlimmerConfig(
            text_config=dict(vocab_size=tc.vocab_size, hidden_size=64, intermediate_size=128,
                             num_hidden_layers=4, num_attention_heads=2,
                             num_key_value_heads=1, head_dim=32,
                             max_position_embeddings=8192, sliding_window=tc.sliding_window),
            vision_config=dict(hidden_size=32, intermediate_size=64, num_hidden_layers=1,
                               num_attention_heads=2, patch_size=4, pos_emb_height=4,
                               pos_emb_width=4, max_position_embeddings=16),
            out_hidden_size=128, projector_hidden_size=48)
        model = MuseGlimmerForConditionalGeneration(cfg)
    model = model.eval().float()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


class Script:
    """engines.sample_next, replaced: the next id of the current reply, then
    the reply's end-of-turn id forever."""

    def __init__(self):
        self.ids: list[int] = []
        self.end = None

    def __call__(self, *_a, **_k):
        return self.ids.pop(0) if self.ids else self.end


class Spy:
    """engines.pick_slot and HFEngine.generate, watched: per request, the
    prompt's key ids, what the slot held, and what the engine reused."""

    def __init__(self, engines, eng):
        self.turns: list[dict] = []
        real_pick, real_gen = engines.pick_slot, eng.generate

        def pick(slots, ids, n_prompt, need, fit=None):
            held = max(slots, key=lambda s: s.stamp).ids
            lcp = 0
            for a, b in zip(ids, held):
                if a != b:
                    break
                lcp += 1
            self.turns.append({"ids": list(ids), "held": list(held), "lcp": lcp,
                               "n_prompt": n_prompt})
            return real_pick(slots, ids, n_prompt, need, fit)

        def gen(req):
            for ev in real_gen(req):
                if type(ev).__name__ == "StreamStart":
                    self.turns[-1]["reused"] = ev.cached_tokens
                yield ev

        engines.pick_slot = pick
        eng.generate = gen


def _post(port, path, body):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json",
                                          "anthropic-version": "2023-06-01"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.load(r)


def _tokens(tok, text):
    return tok(text, add_special_tokens=False).input_ids


LOOKUP = [{"name": "lookup_owner", "description": "The owner of a service.",
           "input_schema": {"type": "object", "properties": {"service": {"type": "string"}},
                            "required": ["service"]}},
          {"name": "lookup_region", "description": "The region of a service.",
           "input_schema": {"type": "object", "properties": {"service": {"type": "string"}},
                            "required": ["service"]}}]


def case_gemma_off(eng, tok, port, script):
    replies = ["The deploy token is PELICAN-7391.", "bottle", "local"]
    asks = ["What is the deploy token?", "Which service is degraded?", "Its region?"]
    msgs = []
    for ask, reply in zip(asks, replies):
        msgs.append({"role": "user", "content": ask})
        script.ids = _tokens(tok, reply)
        out = _post(port, "/v1/messages", {"model": eng.model_id, "max_tokens": 64,
                                           "system": SYSTEM, "messages": msgs,
                                           "thinking": {"type": "disabled"}})
        msgs.append({"role": "assistant", "content": out["content"]})


def case_gemma_tools(eng, tok, port, script):
    writes = ['<|tool_call>call:lookup_owner{service:<|"|>bottle<|"|>}<tool_call|>',
              '<|channel>thought\n<channel|><|tool_call>call:lookup_region{service:<|"|>'
              'bottle<|"|>}<tool_call|>',
              "<|channel>thought\n<channel|>bottle, local, team-halibut"]
    results = [None, "team-halibut", "local"]
    msgs = [{"role": "user", "content": "Which team owns the degraded service, and where "
                                        "does it run? Use the tools."}]
    for write, result in zip(writes, results):
        if result is not None:
            msgs.append({"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": use_id, "content": result}]})
        script.ids = _tokens(tok, write)
        out = _post(port, "/v1/messages", {"model": eng.model_id, "max_tokens": 64,
                                           "system": SYSTEM, "messages": msgs, "tools": LOOKUP,
                                           "thinking": {"type": "disabled"}})
        msgs.append({"role": "assistant", "content": out["content"]})
        use_id = next((b["id"] for b in out["content"] if b["type"] == "tool_use"), None)


def case_gemma_think(eng, tok, port, script):
    writes = ["<|channel>thought\nThe user wants the owner; I need the tool.<channel|>"
              '<|tool_call>call:lookup_owner{service:<|"|>bottle<|"|>}<tool_call|>',
              "<|channel>thought\nThe tool says team-halibut.<channel|>team-halibut"]
    results = [None, "team-halibut"]
    msgs = [{"role": "user", "content": "Which team owns the bottle service? Use the tools."}]
    for write, result in zip(writes, results):
        if result is not None:
            msgs.append({"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": use_id, "content": result}]})
        script.ids = _tokens(tok, write)
        out = _post(port, "/v1/messages", {"model": eng.model_id, "max_tokens": 64,
                                           "system": SYSTEM, "messages": msgs, "tools": LOOKUP,
                                           "thinking": {"type": "enabled",
                                                        "budget_tokens": 1024}})
        msgs.append({"role": "assistant", "content": out["content"]})
        use_id = next((b["id"] for b in out["content"] if b["type"] == "tool_use"), None)


def case_qwen_nl(eng, tok, port, script):
    replies = ["PELICAN-7391 OSPREY-2846\n", "bottle\n", "local"]
    asks = ["Read both nonces.", "Which service is degraded?", "Its region?"]
    msgs = [{"role": "system", "content": SYSTEM}]
    for ask, reply in zip(asks, replies):
        msgs.append({"role": "user", "content": ask})
        script.ids = _tokens(tok, reply)
        out = _post(port, "/v1/chat/completions", {
            "model": eng.model_id, "messages": msgs, "max_tokens": 64, "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False}})
        msgs.append({"role": "assistant", "content": out["choices"][0]["message"]["content"]})


def case_glimmer(eng, tok, port, script):
    writes = [" to=self<|message|>The user asks for the token.<|eom|><|start|>assistant "
              "to=user<|message|>PELICAN-7391",
              " to=self<|message|>The degraded row is bottle.<|eom|><|start|>assistant "
              "to=user<|message|>bottle"]
    asks = ["What is the deploy token?", "Which service is degraded?"]
    msgs = [{"role": "system", "content": SYSTEM + "\n\nReasoning strength: low."}]
    for ask, write in zip(asks, writes):
        msgs.append({"role": "user", "content": ask})
        script.ids = _tokens(tok, write)
        out = _post(port, "/v1/chat/completions", {
            "model": eng.model_id, "messages": msgs, "max_tokens": 96, "temperature": 0})
        msgs.append({"role": "assistant", "content": out["choices"][0]["message"]["content"]})


CASES = {"gemma-off": (GEMMA, case_gemma_off), "gemma-tools": (GEMMA, case_gemma_tools),
         "gemma-think": (GEMMA, case_gemma_think), "qwen-nl": (QWEN, case_qwen_nl),
         "glimmer": (GLIMMER, case_glimmer)}


def run(name: str, ckpts: str) -> list[dict]:
    from drinkme.serving import engines, gen_config
    from drinkme.serving.checkpoint import tokenizer
    from drinkme.serving.engines import HFEngine
    from drinkme.serving.http import start_server

    (repo, rev), drive = CASES[name]
    tok = tokenizer(repo, rev)
    model = _toy(repo, rev)
    gd = gen_config.load(model, _snapshot(repo, rev))
    os.environ.update(DRINKME_CTX_CHECKPOINTS=ckpts, DRINKME_SPEC="off",
                      DRINKME_SLOT_DIR="off", DRINKME_PREFIX_SLOTS="1")
    eng = HFEngine(model, tok, model_id=repo, arm="test", meta={}, ctx=8192,
                   gen_defaults=gd)
    script = Script()
    script.end = sorted(eng.eos_ids)[0]
    real_sample = engines.sample_next
    engines.sample_next = script
    real_pick = engines.pick_slot
    spy = Spy(engines, eng)
    srv = start_server(eng, "127.0.0.1", 0)  # serves on its own daemon thread
    try:
        drive(eng, tok, srv.server_address[1], script)
    finally:
        srv.shutdown()
        engines.sample_next, engines.pick_slot = real_sample, real_pick
    out, prev = [], None
    for i, t in enumerate(spy.turns):
        lcp, held = t["lcp"], t["held"]

        def show(xs):
            return "".join(tok.decode([x], skip_special_tokens=False) for x in xs)

        row = {"turn": i + 1, "n_prompt": t["n_prompt"], "held": len(held), "lcp": lcp,
               "previous_prompt": prev, "reused": t.get("reused"),
               "prompt_after": show(t["ids"][max(0, lcp - 4):lcp + 8]),
               "slot_after": show(held[max(0, lcp - 4):lcp + 8])}
        if prev is not None:
            row["want"] = min(lcp, prev)
            row["pass"] = t.get("reused", 0) >= row["want"] > 0
        out.append(row)
        prev = t["n_prompt"]
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--case", choices=sorted(CASES), action="append")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    report, ok = {}, True
    for name in a.case or list(CASES):
        report[name] = {}
        for arm, ckpts in (("before", "0"), ("after", "32")):
            rows = run(name, ckpts)
            report[name][arm] = rows
            for r in rows:
                verdict = "" if "pass" not in r else ("PASS" if r["pass"] else "FAIL")
                print(f"{name:12s} {arm:6s} turn {r['turn']}: prompt {r['n_prompt']}, "
                      f"previous prompt {r['previous_prompt']}, slot {r['held']}, "
                      f"parts at {r['lcp']}, reused {r['reused']}"
                      + (f" (want {r['want']}) {verdict}" if verdict and arm == "after" else "")
                      + f"\n    prompt: {r['prompt_after']!r}\n    slot:   {r['slot_after']!r}",
                      flush=True)
                if arm == "after" and "pass" in r:
                    ok &= r["pass"]
    with open(os.path.join(a.out, "probe.json"), "w") as f:
        json.dump(report, f, indent=1)
    print("VERDICT", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
