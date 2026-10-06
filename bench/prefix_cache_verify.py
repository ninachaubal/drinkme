"""Prefix/KV cache verify: warm-vs-cold on the REAL model, over the real wire.

One process, one engine, one server. The COLD arm flips the engine's _reuse
off, which is computation-identical to the pre-cache code (per-request
StaticCache, full prefill from position 0); the WARM arm is the shipped
default. Same weights, same everything — so cold-vs-warm here IS old-vs-new.

An agent-shaped conversation (long system prompt, history re-sent whole each
turn, greedy) runs through both arms. For each turn we record text, TTFT
(request sent -> first content delta on the SSE stream), total wall time,
usage.prompt_tokens_details.cached_tokens, and each emitted token's pick and
top-1 minus top-2 logit margin (agreement.PickTap: the engine runs in this
process). The report:

  1. cold vs warm text per turn: same computation modulo kernel batching,
     so the two agree or part at a near-tie. Where they part, the token
     index and both runs' margins there are printed. Each arm conditions on
     its own replies, so turns after the first that parts are not compared.
  2. cached_tokens grows with history (the bookkeeping claim): 0 on every
     cold turn and on the first warm one, then larger every warm turn and
     below the prompt (at least one prompt token re-runs).
  3. TTFT ratio cold/warm per turn (the speedup agents feel)
  4. a json_schema request's reply parses to its schema, and stock replays
     the warm arm's history for the compressed-vs-stock A/B, compared like 1.

Exit 2 on a bookkeeping failure, a structured-output failure, or a part
whose margin is above agreement.NEAR_TIE_MARGIN; a near-tie part is
reported and passes.

Writes verification/prefix_cache_<model>_<host>_<date>.json next to the strix A/B.
Run: uv run --no-sync python bench/prefix_cache_verify.py [--turns N] [--max-tokens N]
     (--no-sync is not optional here — see AGENTS.md)
"""

from __future__ import annotations

import argparse
import http.client
import json
import os
import platform
import socket
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from agreement import NEAR_TIE_MARGIN, PickTap, above_margin, describe, fork  # noqa: E402

# DEFAULT stays the 8B, the dense reference. Override with --pack to ask about a different model — notably a
# HYBRID, where prefix reuse is a live question rather than a shipped default.
DEFAULT_PACK = os.path.expanduser("~/.cache/drinkme/packs/Qwen--Qwen3-8B@b968826d9c46")

# ~agent-sized system prompt: instructions with enough entropy to be a real
# prefill load, not a run of repeated tokens.
SYSTEM = " ".join(
    f"Rule {i}: when the user mentions topic-{i}, respond with a numbered plan "
    f"citing section {i * 7 % 101} of the handbook, keeping the tone plain."
    for i in range(120)
)

USERS = [
    "Summarize your rules for topic-3 in two sentences.",
    "Now do the same for topic-14, and note one difference from topic-3.",
    "Which section number applies to topic-50? Answer in one line.",
    "List the section numbers you cited so far, comma separated.",
]


def request(port: int, messages: list[dict], max_tokens: int):
    """Greedy streaming request. Returns (text, ttft, wall, usage)."""
    # THINKING OFF. The think channel splits reasoning into
    # delta.reasoning_content, and a thinking model can spend its whole token
    # budget there — a content-only parser then sees no first token and
    # reports a server-side error for a request that worked fine
    # (bench/serve_timing_ab.py turns thinking off for the same reason).
    body = json.dumps({"messages": messages, "temperature": 0,
                       "max_tokens": max_tokens, "stream": True,
                       "stream_options": {"include_usage": True},
                       "chat_template_kwargs": {"enable_thinking": False}})
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=600)
    t0 = time.perf_counter()
    c.request("POST", "/v1/chat/completions", body,
              {"Content-Type": "application/json"})
    r = c.getresponse()
    assert r.status == 200, r.status
    ttft, text, usage, buf = None, [], None, b""
    while True:
        chunk = r.read1(65536)
        if not chunk:
            break
        buf += chunk
        while b"\n\n" in buf:
            frame, buf = buf.split(b"\n\n", 1)
            if not frame.startswith(b"data: "):
                continue
            payload = frame[len(b"data: "):]
            if payload == b"[DONE]":
                continue
            obj = json.loads(payload)
            if obj.get("usage"):
                usage = obj["usage"]
            for ch in obj.get("choices", []):
                delta = ch.get("delta", {}).get("content")
                if delta:
                    if ttft is None:
                        ttft = time.perf_counter() - t0
                    text.append(delta)
    wall = time.perf_counter() - t0
    c.close()
    return "".join(text), ttft, wall, usage


def run_arm(port: int, eng, reuse: bool, turns: int, max_tokens: int,
            label: str | None = None, replay: list | None = None,
            tap: PickTap | None = None):
    """One arm of the conversation. With `replay`, the assistant turns come
    from another arm's transcript (so both engines see the same inputs per
    turn — the A/B condition) instead of self-conditioning. With `tap`, each
    turn keeps its picks and margins."""
    eng._reuse = reuse
    eng.reset_prefix_cache()  # each arm starts cold in its own terms
    label = label or ("warm" if reuse else "cold")
    out = []
    history = [{"role": "system", "content": SYSTEM}]
    for i, u in enumerate(USERS[:turns]):
        history.append({"role": "user", "content": u})
        text, ttft, wall, usage = request(port, history, max_tokens)
        ids, margins = tap.take() if tap is not None else ([], [])
        if ttft is None:
            # the server answered without a single content delta (e.g. an
            # in-stream 500) — fail as a REPORT, not a traceback on round(None)
            raise SystemExit(
                f"[{label}] turn {i + 1}: no first token — server-side error; "
                f"text={text!r} usage={usage!r}")
        history.append({"role": "assistant",
                        "content": replay[i] if replay is not None else text})
        cached = (usage or {}).get("prompt_tokens_details", {}).get("cached_tokens", 0)
        out.append({"text": text, "ttft": round(ttft, 3), "wall": round(wall, 3),
                    "prompt_tokens": (usage or {}).get("prompt_tokens"),
                    "cached_tokens": cached, "ids": ids, "margins": margins})
        print(f"  [{label}] turn {len(out)}: "
              f"prompt={out[-1]['prompt_tokens']} cached={cached} "
              f"ttft={ttft:.2f}s wall={wall:.2f}s", flush=True)
    return out


def compare_turns(a: list, b: list, own_history: bool) -> list[dict]:
    """Per turn: whether two arms' texts are the same and, where they are
    not, where their picks part (agreement.fork). With `own_history` each arm
    re-sent its own replies, so every turn after the first that parts saw a
    different prompt on each arm and is marked not compared."""
    out, parted = [], False
    for x, y in zip(a, b):
        if parted and own_history:
            out.append({"compared": False})
            continue
        same = x["text"] == y["text"]
        out.append({"compared": True, "same_text": same,
                    "fork": None if same else fork(x["ids"], y["ids"], x["margins"], y["margins"])})
        parted = parted or not same
    return out


def cache_bookkeeping(cold: list, warm: list) -> list[str]:
    """What is wrong with the arms' cached_tokens, one line per problem
    (empty when right): 0 on every cold turn and on the first warm one (each
    arm starts from an emptied cache), then on every later warm turn more
    than the turn before and less than its prompt."""
    bad = []
    for i, t in enumerate(cold):
        if t["cached_tokens"]:
            bad.append(f"cold turn {i + 1}: cached {t['cached_tokens']}, expected 0")
    for i, t in enumerate(warm):
        c, p = t["cached_tokens"] or 0, t["prompt_tokens"]
        if i == 0:
            if c:
                bad.append(f"warm turn 1: cached {c}, expected 0 after the reset")
            continue
        prev = warm[i - 1]["cached_tokens"] or 0
        if c <= prev:
            bad.append(f"warm turn {i + 1}: cached {c}, not more than turn {i}'s {prev}")
        if p is None or c >= p:
            bad.append(f"warm turn {i + 1}: cached {c} of {p} prompt tokens; at least one re-runs")
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--turns", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=48)
    ap.add_argument("--ctx", type=int, default=32768)
    ap.add_argument("--pack", default=DEFAULT_PACK,
                    help="pack dir to verify. The DEFAULT is the 8B, which is "
                         "NON-HYBRID and already ships with prefix reuse ON, "
                         "so a bare run re-answers a settled question. Point "
                         "it at a hybrid pack to learn anything new.")
    args = ap.parse_args()
    # on-disk prefix slots: the cold tier would hand a "cold" arm a stale
    # prefix off an SSD, which is an instrument measuring nothing. setdefault, so a
    # deliberate DRINKME_SLOT_DIR on the command line still wins.
    os.environ.setdefault("DRINKME_SLOT_DIR", "off")
    pack = os.path.expanduser(args.pack)
    if not os.path.isdir(pack):
        sys.exit(f"[verify] no such pack: {pack}")

    from drinkme.serve import build_engine
    from drinkme.serving.http import start_server

    meta = json.load(open(os.path.join(pack, "meta.json")))
    print(f"[verify] loading {meta['hfRepo']}@{meta['revision'][:12]} "
          f"(meanBpw {meta['meanBpw']}) ...", flush=True)
    t0 = time.time()
    eng = build_engine(meta["hfRepo"], meta["revision"], pack, stock=False,
                       ctx=args.ctx)
    print(f"[verify] loaded in {time.time() - t0:.0f}s", flush=True)
    if str(eng.device) == "cpu" and not os.environ.get("VERIFY_ALLOW_CPU"):
        # a CPU verify already produced a day of mislabeled numbers once
        sys.exit("[verify] REFUSING to measure on CPU (VERIFY_ALLOW_CPU=1 to override)")
    srv = start_server(eng, "127.0.0.1", 0)
    port = srv.server_address[1]

    tap = PickTap().__enter__()  # every request below is served in this process
    print("[verify] COLD arm (reuse off — computation-identical to pre-cache code)")
    cold = run_arm(port, eng, False, args.turns, args.max_tokens, tap=tap)
    print("[verify] WARM arm (prefix cache on)")
    warm = run_arm(port, eng, True, args.turns, args.max_tokens, tap=tap)

    # -- structured output smoke, live on the real model ----------------------
    print("[verify] structured output (json_schema, grammar-constrained)")
    schema = {"type": "object",
              "properties": {"topic": {"type": "string"},
                             "section": {"type": "integer"},
                             "confident": {"type": "boolean"}},
              "required": ["topic", "section", "confident"],
              "additionalProperties": False}
    body = json.dumps({"messages": [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": "Which handbook section applies to "
         "topic-9? Answer as JSON."}],
        "temperature": 0, "max_tokens": 64,
        "response_format": {"type": "json_schema",
                            "json_schema": {"name": "ans", "schema": schema}}})
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=600)
    c.request("POST", "/v1/chat/completions", body,
              {"Content-Type": "application/json"})
    r = c.getresponse()
    structured_raw = json.loads(r.read())
    c.close()
    s_content = structured_raw["choices"][0]["message"]["content"]
    try:
        s_obj = json.loads(s_content)
        structured_ok = (isinstance(s_obj.get("topic"), str)
                         and isinstance(s_obj.get("section"), int)
                         and isinstance(s_obj.get("confident"), bool)
                         and set(s_obj) == {"topic", "section", "confident"})
    except (ValueError, AttributeError):
        s_obj, structured_ok = None, False
    print(f"  structured: ok={structured_ok} content={s_content!r}")
    tap.take()  # nothing the structured request recorded belongs to a turn

    # -- the A/B under the new policy: stock replays the SAME inputs ----------
    print("[verify] loading STOCK bf16 arm for the A/B replay ...", flush=True)
    t0 = time.time()
    stock_eng = build_engine(meta["hfRepo"], meta["revision"], None, stock=True,
                             ctx=args.ctx)
    print(f"[verify] stock loaded in {time.time() - t0:.0f}s", flush=True)
    srv2 = start_server(stock_eng, "127.0.0.1", 0)
    port2 = srv2.server_address[1]
    stock = run_arm(port2, stock_eng, True, args.turns, args.max_tokens,
                    label="stock", replay=[w["text"] for w in warm], tap=tap)
    srv2.shutdown()
    tap.__exit__()

    cold_warm = compare_turns(cold, warm, own_history=True)
    stock_comp = compare_turns(stock, warm, own_history=False)
    bookkeeping = cache_bookkeeping(cold, warm)
    above = [f"{name} turn {i + 1}" for name, rows in (("cold vs warm", cold_warm),
                                                      ("stock vs compressed", stock_comp))
             for i, r in enumerate(rows) if above_margin(r.get("fork"))]
    report = {
        "date": time.strftime("%Y-%m-%d %H:%M %Z"),
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "model": {k: meta[k] for k in ("hfRepo", "revision", "meanBpw")},
        "device": str(eng.device),
        "ctx": args.ctx,
        "max_tokens": args.max_tokens,
        "near_tie_margin": NEAR_TIE_MARGIN,
        "turns": [
            {"turn": i + 1,
             "prompt_tokens": w["prompt_tokens"],
             "cached_tokens": w["cached_tokens"],
             "cold_vs_warm": cold_warm[i],
             "stock_vs_compressed": stock_comp[i],
             "ttft_cold_s": cold[i]["ttft"], "ttft_warm_s": w["ttft"],
             "ttft_ratio": round(cold[i]["ttft"] / w["ttft"], 2) if w["ttft"] else None,
             "wall_cold_s": cold[i]["wall"], "wall_warm_s": w["wall"],
             "text": w["text"]}
            for i, w in enumerate(warm)],
        "cold_warm_same_text": all(r.get("same_text") for r in cold_warm),
        "stock_compressed_same_text": all(r.get("same_text") for r in stock_comp),
        "parts_above_near_tie_margin": above,
        "cache_bookkeeping_problems": bookkeeping,
        "structured_output_ok": structured_ok,
        "structured_output": s_obj,
    }
    report["pass"] = not above and not bookkeeping and structured_ok
    out = os.path.join(os.path.dirname(__file__), "..", "verification",
                       # MODEL IN THE FILENAME. Without it, two packs verified
                       # on the same day silently overwrite each other and the
                       # survivor's name lies about what it measured — exactly
                       # how an A100-40 receipt can come to hold an 80GB
                       # run. The name must carry what varied.
                       f"prefix_cache_{meta['hfRepo'].split('/')[-1]}"
                       f"_{socket.gethostname()}_{time.strftime('%Y-%m-%d')}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps({k: report[k] for k in
                      ("cold_warm_same_text", "stock_compressed_same_text",
                       "parts_above_near_tie_margin", "cache_bookkeeping_problems",
                       "structured_output_ok")}, indent=2))

    def said(r):
        if not r["compared"]:
            return "not compared (the arms' histories differ)"
        return "same text" if r["same_text"] else describe(r["fork"])

    for t in report["turns"]:
        print(f"  turn {t['turn']}: cold vs warm {said(t['cold_vs_warm'])}; "
              f"stock vs compressed {said(t['stock_vs_compressed'])}; "
              f"cached={t['cached_tokens']}/{t['prompt_tokens']} "
              f"ttft {t['ttft_cold_s']}s -> {t['ttft_warm_s']}s "
              f"({t['ttft_ratio']}x)")
    print(f"[verify] {'PASS' if report['pass'] else 'FAIL'} — wrote {os.path.normpath(out)}")
    srv.shutdown()
    sys.exit(0 if report["pass"] else 2)


if __name__ == "__main__":
    # os._exit, NOT a bare return. On this box's TheRock ROCm torch an atexit
    # handler calls _exit(0) once HIP has initialised, so a SystemExit or an
    # uncaught traceback leaves the process reporting SUCCESS — a verify that
    # refused to run looks identical to one that passed (the drinkme CLI
    # exits the same way).
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
