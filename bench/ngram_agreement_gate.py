#!/usr/bin/env python3
"""Speculation agreement gate against a LIVE server: greedy, the same
prompts, `DRINKME_SPEC=off` vs `DRINKME_SPEC=auto`.

The GPU half of the sliding-window rewind fix (n-gram speculation forked badly on
gemma-4-31B-it: the rewind after a rejected draft left every sliding_attention
layer's mask `drop` positions late, so the model attended the rejected
drafts' stale KV rows — serving/mtp.py's docstring, surface 1b). The CPU half
is tests/test_serving_gemma_spec.py, which pins TOKEN IDS on a tiny gemma-4;
this file asks the real weights on the real box. Run it on gemma (the fix)
AND on Qwen3-8B (no regression on a model with no sliding layers).

TWO SERVER PROCESSES, because DRINKME_SPEC is read once per engine and there
is no runtime toggle. Either both live at once on two ports:

    DRINKME_SPEC=off  drinkme serve --model google/gemma-4-31B-it --port 3301 &
    DRINKME_SPEC=auto drinkme serve --model google/gemma-4-31B-it --port 3302 &
    uv run --no-sync python bench/ngram_agreement_gate.py \\
        --off http://127.0.0.1:3301 --auto http://127.0.0.1:3302 \\
        --json measurements/ngram_agreement_gemma.json

or — the 31B does not fit twice beside anything — one at a time, on the same
port, saving the first arm's transcripts and comparing the second against them:

    DRINKME_SPEC=off  drinkme serve --model google/gemma-4-31B-it --port 3301
    uv run --no-sync python bench/ngram_agreement_gate.py \\
        --url http://127.0.0.1:3301 --arm off --save /tmp/gate_gemma_off.json
    # stop it; start the other arm on the same port
    DRINKME_SPEC=auto drinkme serve --model google/gemma-4-31B-it --port 3301
    uv run --no-sync python bench/ngram_agreement_gate.py \\
        --url http://127.0.0.1:3301 --arm auto --against /tmp/gate_gemma_off.json \\
        --json measurements/ngram_agreement_gemma.json

NEVER against drinkme serve's default port 3215, where a working server may
be listening (AGENTS.md) — stand your own up on another port.

TOKEN-EXACT BETWEEN SPECULATION ON AND OFF IS NOT A PROPERTY THIS GATE
ASSERTS (token-exact is impossible, not worth chasing,
and no code should check for it). A batched verify
forward over k+1 positions and the serial loop's single-token one can round
differently in accumulation order, and a genuine near-tie can flip an argmax
for reasons that have nothing to do with a bug — the same bar AGENTS.md holds
every other arm-vs-arm comparison in this repo to. This gate cannot compute a
logit margin from outside the server process (no generated ids, no logprobs
on the wire), so a divergence is reported INFORMATIONALLY — the first
divergent token position and the surrounding text of both arms — with no
PASS/FAIL riding on it. What IS pass/fail is the ARM LABELS: `off` must
have proposed nothing and `auto` must have proposed something, or the run
proves nothing about either arm.

WHAT IS COMPARED, and how strong it is. The server exposes no generated ids
and no logprobs, so the comparison is the TEXT of each greedy completion plus
its `completion_tokens` count, and the text re-encoded through /tokenize so a
divergence is reported at a token position rather than a character offset.
That is WEAKER than id identity: `skip_special_tokens` could hide a
special-token difference, and two id sequences can decode to one string. It
is the strongest check available from outside the process; the id-level,
margin-aware check is the CPU toy's job (tests/spec_agree.py).

THE ARM IS VERIFIED, NOT TRUSTED. Each server's /metrics
`drinkme_spec_proposed_tokens_total` is read before and after its requests:
the `off` arm must propose nothing, the `auto` arm must propose something on
the repetitive prompt. A run whose deltas disagree with the labels is
INVALID (a green check on the wrong process is worse than none). `/health`'s `device` is recorded for the same reason.
INVALID is the only hard verdict this gate gives; agreement/divergence is
reported, never PASS/FAILed.

VERDICT LINES GO IN THE OUTPUT and the receipt, never only in `$?` (AGENTS.md:
TheRock ROCm torch _exit(0)s over the real status once HIP initializes). The
exit code reflects only arm validity, as a courtesy, and is not a transcript
verdict.

A clean run looks like:

    [gate] arm off  : 4 prompts, proposed +0 (serial, as labelled), device cuda:0
    [gate] arm auto : 4 prompts, proposed +812 (speculating, as labelled), device cuda:0
    [gate] math      : agreed  296 tokens
    [gate] repeat    : agreed  300 tokens
    [gate] code      : agreed  300 tokens
    [gate] prose     : agreed  300 tokens
    [gate] VERDICT: arms verified (off serial, auto speculating) — 4/4 greedy
                    transcripts agreed (text + token count; text-level, see
                    the docstring); read any DIVERGED line yourself, this
                    gate does not judge it
    [gate] receipt: measurements/ngram_agreement_gemma.json

and a divergence names the prompt, the first divergent token (index, both
ids, both surrounding texts) — informational, not a failure by itself.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request

# ------------------------------------------------------------------ prompts --

# The receipt's own prompt first (bench/serve_dialect_smoke.py's plain probe —
# the one that showed the holes on gemma), then the shapes that make the
# lookup fire and reject: a block to re-quote (high acceptance), a code edit
# (agent-shaped), and prose with a repeated phrase (mostly rejections).
QUOTE = ("The prefix cache keeps one StaticCache per slot, reused across "
         "requests and allocated geometrically. A request runs on the slot "
         "whose written ids it extends by the longest prefix; with no such "
         "slot the least recently used slot is taken cold and rebuilt.")
CODE = '''def parse(line: str) -> dict:
    key, _, value = line.partition("=")
    if not key.strip():
        raise ValueError(f"bad line: {line!r}")
    return {"key": key.strip(), "value": value.strip()}
'''
PROMPTS = {
    "math": "What is 17 * 23? Reply with just the number.",
    "repeat": f"Repeat the following paragraph exactly, word for word:\n\n{QUOTE}",
    "code": (f"Here is a function:\n\n```python\n{CODE}```\n\nRewrite it so that a "
             "missing '=' raises too, keeping everything else identical. Show the "
             "whole function."),
    "prose": (f"{QUOTE}\n\nExplain the paragraph above in two sentences, then quote "
              "its second sentence back verbatim."),
}

# ---------------------------------------------------------------- transport --


def _req(url: str, path: str, body: dict | None, timeout: float, token: str | None):
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url + path, data=data, headers=headers,
                                 method="POST" if body is not None else "GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read()
            ctype = r.headers.get("Content-Type", "")
            return r.status, (json.loads(raw) if "json" in ctype else raw.decode(errors="replace"))
    except urllib.error.HTTPError as e:
        raw = e.read().decode(errors="replace")
        try:
            return e.code, json.loads(raw)
        except json.JSONDecodeError:
            return e.code, raw


def proposed_total(url: str, timeout: float) -> int | None:
    """drinkme_spec_proposed_tokens_total off /metrics (both proposers count
    there — docs/serve-speculation.md), or None if the route is missing."""
    status, body = _req(url, "/metrics", None, timeout, None)
    if status != 200 or not isinstance(body, str):
        return None
    m = re.search(r"^drinkme_spec_proposed_tokens_total\s+([0-9.eE+]+)", body, re.M)
    return int(float(m.group(1))) if m else 0


def health(url: str, timeout: float) -> dict:
    status, body = _req(url, "/health", None, timeout, None)
    return body if status == 200 and isinstance(body, dict) else {"ok": False, "status": status}


def run_arm(url: str, arm: str, model: str | None, max_tokens: int,
            timeout: float, token: str | None, no_think: bool) -> dict:
    """Every prompt, greedy, against one server. Returns the arm's record:
    per-prompt text / token count / /tokenize ids, plus the /metrics delta
    and /health that say what this process actually was."""
    h = health(url, timeout)
    status, models = _req(url, "/v1/models", None, timeout, token)
    if status != 200:
        raise SystemExit(f"[gate] {url}/v1/models -> {status}: {models}")
    model = model or models["data"][0]["id"]
    before = proposed_total(url, timeout)
    out = {"url": url, "arm": arm, "model": model, "health": h,
           "models": models, "proposed_before": before, "prompts": {}}
    print(f"[gate] arm {arm:<4}: {url} model={model} device={h.get('device')} "
          f"proposed_total={before}")
    for name, prompt in PROMPTS.items():
        body = {"model": model, "temperature": 0, "max_tokens": max_tokens,
                "messages": [{"role": "user", "content": prompt}]}
        if no_think:
            body["chat_template_kwargs"] = {"enable_thinking": False}
        t0 = time.time()
        status, resp = _req(url, "/v1/chat/completions", body, timeout, token)
        dt = time.time() - t0
        if status != 200:
            raise SystemExit(f"[gate] {arm}/{name}: {status}: {resp}")
        msg = resp["choices"][0]["message"]
        text = msg.get("content") or ""
        usage = resp.get("usage", {})
        st, tok = _req(url, "/tokenize", {"prompt": text}, timeout, token)
        ids = tok["tokens"] if st == 200 and isinstance(tok, dict) else None
        rec = {"text": text, "reasoning": msg.get("reasoning_content"),
               "finish_reason": resp["choices"][0].get("finish_reason"),
               "completion_tokens": usage.get("completion_tokens"),
               "prompt_tokens": usage.get("prompt_tokens"),
               "tokenize_ids": ids, "seconds": round(dt, 2)}
        out["prompts"][name] = rec
        print(f"[gate] arm {arm:<4}: {name:<7} {rec['completion_tokens']} tokens, "
              f"finish={rec['finish_reason']}, {dt:.1f}s")
    after = proposed_total(url, timeout)
    out["proposed_after"] = after
    out["proposed_delta"] = None if (before is None or after is None) else after - before
    return out


# ------------------------------------------------------------------- verdict --


def first_divergence(a: list[int] | None, b: list[int] | None) -> int | None:
    if a is None or b is None:
        return None
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return None if len(a) == len(b) else n


def compare(off: dict, auto: dict) -> tuple[bool, list[str], dict]:
    """(arms_valid, lines, detail). `arms_valid` is the ONLY pass/fail this
    gate gives — whether `off`/`auto` were what their labels said. Text
    agreement between the two arms is reported per prompt (agreed/DIVERGED)
    but never turns `arms_valid` false: token-exact between speculation on
    and off is not a property this gate asserts (module docstring)
    — informational, always, however many prompts diverge."""
    lines, detail = [], {}
    # the arm labels, checked against what the processes did — the one real
    # pass/fail question a gate with no logits can answer
    d_off, d_auto = off.get("proposed_delta"), auto.get("proposed_delta")
    off_ok = d_off == 0
    auto_ok = d_auto is not None and d_auto > 0
    lines.append(f"[gate] arm off  : {len(off['prompts'])} prompts, proposed "
                 f"{'+' + str(d_off) if d_off is not None else 'n/a'} "
                 f"({'serial, as labelled' if off_ok else 'NOT SERIAL — wrong process?'}), "
                 f"device {off['health'].get('device')}")
    lines.append(f"[gate] arm auto : {len(auto['prompts'])} prompts, proposed "
                 f"{'+' + str(d_auto) if d_auto is not None else 'n/a'} "
                 f"({'speculating, as labelled' if auto_ok else 'NOT SPECULATING — wrong process?'}), "
                 f"device {auto['health'].get('device')}")
    valid = off_ok and auto_ok
    if off.get("model") != auto.get("model"):
        lines.append(f"[gate] MODEL MISMATCH: off={off.get('model')} auto={auto.get('model')}")
        valid = False
    n_same = 0
    for name in PROMPTS:
        a, b = off["prompts"].get(name), auto["prompts"].get(name)
        if a is None or b is None:
            lines.append(f"[gate] {name:<9}: MISSING in one arm")
            valid = False
            continue
        same_text = a["text"] == b["text"]
        same_n = a["completion_tokens"] == b["completion_tokens"]
        div = first_divergence(a["tokenize_ids"], b["tokenize_ids"])
        detail[name] = {"agreed": same_text and same_n, "same_text": same_text,
                        "same_completion_tokens": same_n, "first_divergent_token": div,
                        "off_tokens": a["completion_tokens"], "auto_tokens": b["completion_tokens"]}
        if same_text and same_n:
            n_same += 1
            lines.append(f"[gate] {name:<9}: agreed  {a['completion_tokens']} tokens")
            continue
        # INFORMATIONAL from here — a fork is not this gate's failure to
        # judge without a margin (it has no logits); it names where and lets
        # a human decide whether the surrounding text looks like a near-tie
        # or a real bug.
        if div is None and not same_text:
            # /tokenize was unavailable, or the texts differ only where the
            # tokenizer could not see it: fall back to a character offset
            i = next((k for k, (x, y) in enumerate(zip(a["text"], b["text"])) if x != y),
                     min(len(a["text"]), len(b["text"])))
            lines.append(f"[gate] {name:<9}: DIVERGED  at char {i}: "
                         f"off={a['text'][max(0, i - 30):i + 30]!r} "
                         f"auto={b['text'][max(0, i - 30):i + 30]!r}")
        elif div is not None:
            ia, ib = a["tokenize_ids"], b["tokenize_ids"]
            lines.append(f"[gate] {name:<9}: DIVERGED  first divergent token #{div}: "
                         f"off id {ia[div] if div < len(ia) else 'END'} vs "
                         f"auto id {ib[div] if div < len(ib) else 'END'}; "
                         f"tokens off={a['completion_tokens']} auto={b['completion_tokens']}")
            # a little context around it, from the text each arm produced
            lines.append(f"[gate]            off : ...{a['text'][-120:]!r}")
            lines.append(f"[gate]            auto: ...{b['text'][-120:]!r}")
        else:
            lines.append(f"[gate] {name:<9}: DIVERGED  same text, completion_tokens "
                         f"{a['completion_tokens']} vs {b['completion_tokens']}")
    if not valid:
        lines.append("[gate] VERDICT: INVALID — the arms are not what their labels say "
                     "(see the proposed deltas above); nothing was proven either way")
        return False, lines, detail
    lines.append(f"[gate] VERDICT: arms verified (off serial, auto speculating) — "
                 f"{n_same}/{len(PROMPTS)} greedy transcripts agreed (text + token "
                 "count; text-level, see the docstring); read any DIVERGED line "
                 "yourself, this gate does not judge it")
    return True, lines, detail


# ---------------------------------------------------------------------- main --


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--off", help="base URL of a server started with DRINKME_SPEC=off")
    ap.add_argument("--auto", help="base URL of a server started with DRINKME_SPEC=auto")
    ap.add_argument("--url", help="one server at a time: its base URL (with --arm)")
    ap.add_argument("--arm", choices=["off", "auto"], help="which arm --url is serving")
    ap.add_argument("--save", help="with --url: write this arm's transcripts here")
    ap.add_argument("--against", help="with --url: compare against a --save file of the other arm")
    ap.add_argument("--model", help="model id to request (default: the server's first)")
    ap.add_argument("--max-tokens", type=int, default=300)
    ap.add_argument("--timeout", type=float, default=900.0)
    ap.add_argument("--token", default=os.environ.get("DRINKME_AUTH_TOKEN"),
                    help="bearer, if the server was started with --auth")
    ap.add_argument("--no-think", action="store_true",
                    help="send chat_template_kwargs.enable_thinking=false (Qwen3-8B: "
                         "its card says greedy thinking loops; the gate wants answers)")
    ap.add_argument("--json", help="receipt path (a lexicon-shaped JSON, one per run)")
    args = ap.parse_args()

    receipt = {"started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "argv": sys.argv[1:],
               "max_tokens": args.max_tokens, "prompts": PROMPTS}
    if args.off and args.auto:
        off = run_arm(args.off, "off", args.model, args.max_tokens, args.timeout,
                      args.token, args.no_think)
        auto = run_arm(args.auto, "auto", args.model, args.max_tokens, args.timeout,
                       args.token, args.no_think)
    elif args.url and args.arm:
        mine = run_arm(args.url, args.arm, args.model, args.max_tokens, args.timeout,
                       args.token, args.no_think)
        if args.save:
            with open(args.save, "w") as f:
                json.dump(mine, f, indent=1)
            print(f"[gate] saved arm {args.arm} transcripts: {args.save}")
        if not args.against:
            print(f"[gate] VERDICT: PENDING — one arm recorded ({args.arm}); run the "
                  "other arm with --against to compare")
            if args.json:
                with open(args.json, "w") as f:
                    json.dump({**receipt, "arms": {args.arm: mine}, "verdict": "PENDING"},
                              f, indent=1)
            return 2
        with open(args.against) as f:
            other = json.load(f)
        if other.get("arm") == args.arm:
            raise SystemExit(f"[gate] --against holds arm {other.get('arm')!r} too; "
                             "it must be the OTHER arm")
        off, auto = (mine, other) if args.arm == "off" else (other, mine)
    else:
        ap.error("give --off and --auto, or --url with --arm (and --save / --against)")

    arms_valid, lines, detail = compare(off, auto)
    print("\n".join(lines))
    n_agreed = sum(1 for d in detail.values() if d.get("agreed"))
    receipt.update({"arms": {"off": off, "auto": auto}, "detail": detail,
                    # "verdict" names arm validity only — the one pass/fail
                    # question this gate answers; transcript agreement is in
                    # `detail` and the printed lines, informational always
                    "verdict": "ARMS_VALID" if arms_valid else "INVALID",
                    "prompts_agreed": f"{n_agreed}/{len(detail)}",
                    "finished": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "w") as f:
            json.dump(receipt, f, indent=1)
        print(f"[gate] receipt: {args.json}")
    return 0 if arms_valid else 1


if __name__ == "__main__":
    sys.exit(main())
