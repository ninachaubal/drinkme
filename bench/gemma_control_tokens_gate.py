"""GPU gate for gemma's control tokens on real weights, against a
RUNNING server: does the tool call come back structured, does the thought
channel land in `reasoning_content`, does generation STOP at the call's
close marker, and is plain chat unchanged?

Sits beside bench/serve_dialect_smoke.py (whose helpers it borrows) and is
what flips gemma's `tested` flag in serving/tool_formats.py: a smoke
measured the leak this gate exists to close — `call:get_weather{city:Hilo}`
as prose, `thought` as prose, a hallucinated answer after the call — because
every one of gemma's markers is a special token the decode stripped before
the scanner looked.
serving/control.py keeps
those ids as text and ends the turn at `<tool_call|>`; this asks the weights
whether that holds.

HOW TO RUN (on a GPU, with a server of your own on a port other than
drinkme serve's default 3215, where a working server may be listening):

    # 1. the server, compressed arm, speculation off (the recorded runs
    #    used --spec off; sampled speculation changes the draws),
    #    the untested-dialect door open (gemma is refused for tools otherwise)
    cd /path/to/drinkme
    DRINKME_TOOLS_UNTESTED=1 uv run --no-sync drinkme serve \\
        --model google/gemma-4-31B-it --port 3216 --spec off

    #    wait for the boot log's `[drinkme] control tokens: gemma: kept as
    #    text [...]; turn ends after id 49` line and `[drinkme] device:`
    #    naming the accelerator; /health must say device cuda:0

    # 2. the gate, FIRST thing after boot (see FRESH SERVER below)
    uv run --no-sync python bench/gemma_control_tokens_gate.py \\
        --base-url http://127.0.0.1:3216 \\
        --baseline verification/gemma_plain_compressed_specoff_<host>_<date>.json \\
        -o measurements/gemma_control_tokens_gate_$(hostname)_$(date +%F).json

VERDICTS ARE IN THE OUTPUT, never only in `$?`: one `[gate] <probe>: ... →
OK|FAIL` line per probe, then ONE summary line —

    [gate] PASS: 9/9 probes hold · receipt → <path>
    [gate] FAIL: tool_call_stops_at_close, reasoning_populated · receipt → <path>

— and the JSON receipt carries every request body and raw response. ROCm
torch's atexit handler clobbers exit codes (AGENTS.md), so read the line.

WHAT PASS MEANS, probe by probe:

    models_announce_gemma      /v1/models says tool_format=gemma (the probe
                               picked the row) and names a real device
    plain_defaults             the receipt's exact plain-chat request (no
                               temperature: the server's generation_config
                               defaults, as a real client gets). Compared
                               against a SAMPLED --baseline as a printed
                               note only — never a verdict (see FRESH
                               SERVER below for why that compare is not a
                               ruler).
    plain_greedy_clean         temperature 0 plain chat: no marker in
                               content, the answer (391) present. Recorded
                               so the NEXT gate has a greedy baseline.
    plain_unchanged_vs_baseline the greedy probe's content IDENTICAL to the
                               --baseline's greedy plain probe — control
                               tokens changed nothing outside the markers.
                               Skipped, not failed, when the baseline has
                               no greedy plain probe (a serve_dialect_smoke receipt).
    tool_call_structured       the receipt's tool prompt, greedy: tool_calls
                               == [get_weather(city~Hilo)], finish_reason
                               tool_calls, no marker anywhere
    tool_call_stops_at_close   content is empty on that turn: nothing after
                               the close marker was generated (the
                               hallucinated answer it closes was 20+ tokens
                               of prose here). A non-empty content is
                               either a pre-call preamble — read the receipt
                               — or the stop not firing; either way, look.
    tool_call_stream           the same, streamed and reassembled
    tool_result_in             the call's result fed back: an answer that
                               uses it, no re-call, no marker
    reasoning_populated        a reasoning-heavy prompt with
                               chat_template_kwargs.enable_thinking=true:
                               reasoning_content non-empty, content carries
                               the answer and no marker (gemma's
                               `<|channel>thought` lands in reasoning)
    reasoning_stream           the same, streamed
    anthropic_tool_use         /v1/messages with the tool: a tool_use block,
                               stop_reason tool_use, no marker in text

FRESH SERVER. The baselines were taken at the server's defaults
(temperature 1.0, top_k 64, top_p 0.95) as the FIRST request of a freshly
booted process, on the assumption that torch's deterministically seeded
default generator makes an unseeded sampled run reproduce exactly once per
boot. The gate still sends plain_defaults first. But a run
ran it exactly that way — fresh process, first request — and got `391`
(4 tokens) against the baseline's 296-token explanation, common prefix 0:
the assumption does not hold here (ROCm kernel nondeterminism in the
logits is enough to move a temperature-1.0 draw), so a sampled compare is
a note and only the greedy compare is a verdict.

ON PASS: flip `tested=False` to `tested=True` on the gemma row in
src/drinkme/serving/tool_formats.py (the ONE line, in ROWS), quote this
receipt's path in the row's comment, and re-run the suite (the tested-tier
test pins the set and moves with the flag).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from serve_dialect_smoke import (LEAK_MARKERS, WEATHER_TOOL, assemble,  # noqa: E402
                                 leaks, post, stream)

# the baseline receipt's prompts, verbatim
PLAIN_PROMPT = "What is 17 * 23? Reply with just the number."
TOOL_PROMPT = "What's the weather in Hilo right now? Use the tool."
# a prompt that earns a thought channel: a known trap with a known answer
REASONING_PROMPT = ("A bat and a ball cost $1.10 in total. The bat costs $1.00 more "
                    "than the ball. How much does the ball cost? Think it through "
                    "carefully, then give the answer in dollars.")
REASONING_ANSWER = re.compile(r"0\.05|5 cents|five cents|\$\.05", re.I)


def ok_line(name: str, ok: bool, detail: str) -> None:
    print(f"[gate] {name}: {detail} → {'OK' if ok else 'FAIL'}", flush=True)


def marker_free(*texts) -> list[str]:
    found: list[str] = []
    for t in texts:
        found += [m for m in leaks(t) if m not in found]
    return found


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--base-url", default="http://127.0.0.1:3216")
    ap.add_argument("--model", default=None, help="model id to send (default: first on /v1/models)")
    ap.add_argument("--baseline", default=None,
                    help="a receipt (serve_dialect_smoke / this gate) whose "
                         "probes.plain.response content the defaults-sampled plain probe "
                         "must reproduce exactly")
    ap.add_argument("--timeout", type=float, default=900.0)
    ap.add_argument("--max-tokens", type=int, default=1500)
    ap.add_argument("-o", "--out", help="receipt JSON path")
    args = ap.parse_args()
    base = args.base_url.rstrip("/")

    receipt: dict = {"base_url": base, "gate": "gemma_control_tokens_gate",
                     "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                     "baseline": args.baseline, "probes": {}}
    verdicts: dict[str, bool] = {}

    def record(name: str, body: dict, status: int, resp, **extra) -> None:
        receipt["probes"][name] = {"request": body, "status": status, "response": resp, **extra}

    # ---- /v1/models: the row, the device -----------------------------------
    with urllib.request.urlopen(base + "/v1/models", timeout=30) as r:
        models = json.loads(r.read())
    receipt["models"] = models
    model = args.model or models["data"][0]["id"]
    ann = next((m for m in models["data"] if m["id"] == model), models["data"][0])["drinkme"]
    caps = ann["capabilities"]
    ok = caps.get("toolFormat") == "gemma" and bool(ann.get("device")) and "cpu" not in str(ann.get("device"))
    verdicts["models_announce_gemma"] = ok
    ok_line("models_announce_gemma", ok,
            f"{model} · tool_format={caps.get('tool_format')} thinking={caps.get('thinking')} "
            f"device={ann.get('device')} arm={ann.get('arm')}")

    # ---- 1. plain chat at the server's defaults — the baseline's request, FIRST --
    body = {"model": model, "max_tokens": args.max_tokens,
            "messages": [{"role": "user", "content": PLAIN_PROMPT}]}
    status, resp = post(base, "/v1/chat/completions", body, args.timeout)
    msg = resp["choices"][0]["message"] if status == 200 else {}
    content = msg.get("content") or ""
    lk = marker_free(content, msg.get("reasoning_content"))
    # The baseline receipt's `plain` probe is SAMPLED in serve_dialect_smoke's receipts
    # (no temperature in its request) and GREEDY in every receipt this gate
    # writes (temperature 0). Only the greedy one is a ruler: the sampled
    # compare is a note, never a verdict — a fresh-process run showed the
    # "reproduces exactly once per boot" assumption does not hold on ROCm
    # (391 vs a 296-token explanation, common prefix 0, nothing to do with
    # control tokens).
    sampled_baseline = greedy_baseline = None
    if args.baseline:
        with open(args.baseline) as f:
            b = json.load(f)
        bp = b["probes"].get("plain") or {}
        btext = bp.get("response", {}).get("choices", [{}])[0].get("message", {}).get("content")
        if btext is not None:
            if bp.get("request", {}).get("temperature") == 0:
                greedy_baseline = btext
            else:
                sampled_baseline = btext
    if sampled_baseline is not None:
        common = 0
        for x, y in zip(content, sampled_baseline):
            if x != y:
                break
            common += 1
        record("plain_defaults", body, status, resp, leaks=lk,
               baseline_content=sampled_baseline, common_prefix_chars=common)
        print(f"[gate] plain_defaults_vs_sampled_baseline (note, not a verdict): {status} · "
              f"{len(content)} chars vs baseline {len(sampled_baseline)} · common prefix {common} chars · "
              f"leaks={lk or 'none'}", flush=True)
    else:
        record("plain_defaults", body, status, resp, leaks=lk)
        print(f"[gate] plain_defaults: {status} · {len(content)} chars · leaks={lk or 'none'}", flush=True)

    # ---- 2. plain chat, greedy — the ruler ---------------------------------
    gbody = dict(body, temperature=0)
    status, resp = post(base, "/v1/chat/completions", gbody, args.timeout)
    msg = resp["choices"][0]["message"] if status == 200 else {}
    content = msg.get("content") or ""
    lk = marker_free(content, msg.get("reasoning_content"))
    ok = status == 200 and not lk and "391" in content
    verdicts["plain_greedy_clean"] = ok
    record("plain", gbody, status, resp, leaks=lk)
    ok_line("plain_greedy_clean", ok,
            f"{status} · {len(content)} chars · answer_present={'391' in content} · "
            f"reasoning_content={'yes' if msg.get('reasoning_content') else 'no'} · leaks={lk or 'none'}")
    if greedy_baseline is not None:
        same = status == 200 and content == greedy_baseline
        verdicts["plain_unchanged_vs_baseline"] = same
        ok_line("plain_unchanged_vs_baseline", same,
                f"greedy · {len(content)} chars vs baseline {len(greedy_baseline)} chars")
    else:
        print("[gate] plain_unchanged_vs_baseline: skipped (baseline has no greedy plain probe; "
              "this receipt's greedy probe is the next gate's baseline)", flush=True)

    # ---- 3. the tool call, greedy ------------------------------------------
    tbody = {"model": model, "max_tokens": args.max_tokens, "temperature": 0,
             "tools": [WEATHER_TOOL],
             "messages": [{"role": "user", "content": TOOL_PROMPT}]}
    status, resp = post(base, "/v1/chat/completions", tbody, args.timeout)
    msg = resp["choices"][0]["message"] if status == 200 else {}
    calls = msg.get("tool_calls") or []
    finish = resp.get("choices", [{}])[0].get("finish_reason") if status == 200 else None
    content = msg.get("content") or ""
    lk = marker_free(content, msg.get("reasoning_content"))
    parsed = bool(calls) and calls[0]["function"]["name"] == "get_weather"
    argok = False
    if parsed:
        try:
            argok = "hilo" in json.loads(calls[0]["function"]["arguments"]).get("city", "").lower()
        except (json.JSONDecodeError, AttributeError):
            argok = False
    ok = status == 200 and parsed and argok and finish == "tool_calls" and not lk
    verdicts["tool_call_structured"] = ok
    ok_line("tool_call_structured", ok,
            f"{status} · parsed={parsed} args_ok={argok} finish={finish} leaks={lk or 'none'} · "
            f"completion_tokens={resp.get('usage', {}).get('completion_tokens') if status == 200 else '-'}")
    stopped = status == 200 and parsed and content == ""
    verdicts["tool_call_stops_at_close"] = stopped
    record("tool_call", tbody, status, resp, leaks=lk)
    ok_line("tool_call_stops_at_close", stopped,
            f"content={content[:80]!r}{'…' if len(content) > 80 else ''} "
            f"({'empty: nothing after the close' if content == '' else f'{len(content)} chars — read the receipt'})")

    # ---- 4. the tool call, streamed ----------------------------------------
    status, chunks = stream(base, "/v1/chat/completions", tbody, args.timeout)
    asm = assemble(chunks) if status == 200 else {}
    scalls = asm.get("tool_calls") or []
    scontent = asm.get("content") or ""
    lk = marker_free(scontent, asm.get("reasoning_content"))
    sparsed = bool(scalls) and scalls[0]["name"] == "get_weather"
    ok = (status == 200 and sparsed and asm.get("finish_reason") == "tool_calls"
          and scontent == "" and not lk)
    verdicts["tool_call_stream"] = ok
    record("tool_call_stream", tbody, status, chunks, assembled=asm, n_chunks=len(chunks), leaks=lk)
    ok_line("tool_call_stream", ok,
            f"{status} · {len(chunks)} chunks · parsed={sparsed} finish={asm.get('finish_reason')} "
            f"content_empty={scontent == ''} leaks={lk or 'none'}")

    # ---- 5. the result fed back --------------------------------------------
    if parsed:
        call = calls[0]
        rbody = {"model": model, "max_tokens": args.max_tokens, "temperature": 0,
                 "tools": [WEATHER_TOOL],
                 "messages": tbody["messages"] + [
                     {"role": "assistant", "content": msg.get("content") or None, "tool_calls": calls},
                     {"role": "tool", "tool_call_id": call["id"],
                      "content": json.dumps({"city": "Hilo", "temp_c": 27, "sky": "light rain"})}]}
        status, resp = post(base, "/v1/chat/completions", rbody, args.timeout)
        msg2 = resp["choices"][0]["message"] if status == 200 else {}
        text = msg2.get("content") or ""
        lk = marker_free(text, msg2.get("reasoning_content"))
        used = bool(re.search(r"27|rain", text, re.I))
        ok = status == 200 and not lk and used and not msg2.get("tool_calls")
        verdicts["tool_result_in"] = ok
        record("tool_result_in", rbody, status, resp, leaks=lk)
        ok_line("tool_result_in", ok,
                f"{status} · used_result={used} re_called={bool(msg2.get('tool_calls'))} leaks={lk or 'none'}")
    else:
        verdicts["tool_result_in"] = False
        print("[gate] tool_result_in: skipped (no parsed call to feed back) → FAIL", flush=True)

    # ---- 6. reasoning: thinking on, a prompt that earns a thought ----------
    rbody = {"model": model, "max_tokens": args.max_tokens, "temperature": 0,
             "chat_template_kwargs": {"enable_thinking": True},
             "messages": [{"role": "user", "content": REASONING_PROMPT}]}
    status, resp = post(base, "/v1/chat/completions", rbody, args.timeout)
    msg = resp["choices"][0]["message"] if status == 200 else {}
    reasoning = msg.get("reasoning_content") or ""
    content = msg.get("content") or ""
    lk = marker_free(content, reasoning)
    answered = bool(REASONING_ANSWER.search(content))
    ok = status == 200 and bool(reasoning) and answered and not lk
    verdicts["reasoning_populated"] = ok
    record("reasoning", rbody, status, resp, leaks=lk)
    ok_line("reasoning_populated", ok,
            f"{status} · reasoning_content={len(reasoning)} chars · content={len(content)} chars · "
            f"answer_present={answered} · leaks={lk or 'none'} · "
            f"finish={resp.get('choices', [{}])[0].get('finish_reason') if status == 200 else '-'}")

    # ---- 7. reasoning, streamed --------------------------------------------
    status, chunks = stream(base, "/v1/chat/completions", rbody, args.timeout)
    asm = assemble(chunks) if status == 200 else {}
    sreason = asm.get("reasoning_content") or ""
    scontent = asm.get("content") or ""
    lk = marker_free(scontent, sreason)
    ok = status == 200 and bool(sreason) and bool(REASONING_ANSWER.search(scontent)) and not lk
    verdicts["reasoning_stream"] = ok
    record("reasoning_stream", rbody, status, chunks, assembled=asm, n_chunks=len(chunks), leaks=lk)
    ok_line("reasoning_stream", ok,
            f"{status} · {len(chunks)} chunks · reasoning={len(sreason)} chars · "
            f"content={len(scontent)} chars · leaks={lk or 'none'}")

    # ---- 8. the Anthropic dialect with the tool ----------------------------
    abody = {"model": model, "max_tokens": args.max_tokens, "temperature": 0,
             "tools": [{"name": "get_weather", "description": "Current weather for a city.",
                        "input_schema": WEATHER_TOOL["function"]["parameters"]}],
             "messages": [{"role": "user", "content": TOOL_PROMPT}]}
    status, resp = post(base, "/v1/messages", abody, args.timeout)
    blocks = resp.get("content", []) if status == 200 else []
    tool_use = [b for b in blocks if b.get("type") == "tool_use"]
    texts = "".join(b.get("text", "") for b in blocks if b.get("type") == "text")
    thinking = "".join(b.get("thinking", "") for b in blocks if b.get("type") == "thinking")
    lk = marker_free(texts, thinking)
    ok = (status == 200 and bool(tool_use) and tool_use[0].get("name") == "get_weather"
          and resp.get("stop_reason") == "tool_use" and not lk)
    verdicts["anthropic_tool_use"] = ok
    record("anthropic_tool", abody, status, resp, leaks=lk)
    ok_line("anthropic_tool_use", ok,
            f"{status} · tool_use={bool(tool_use)} stop={resp.get('stop_reason') if status == 200 else '-'} "
            f"text={len(texts)} chars leaks={lk or 'none'}")

    # ---- the summary line and the receipt ----------------------------------
    receipt["verdicts"] = verdicts
    receipt["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    receipt["leak_markers_checked"] = LEAK_MARKERS
    path = args.out
    if path:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w") as f:
            json.dump(receipt, f, indent=1, ensure_ascii=False)
    failed = [k for k, v in verdicts.items() if not v]
    where = f" · receipt → {path}" if path else " · (no -o: receipt not written)"
    if failed:
        print(f"[gate] FAIL: {', '.join(failed)}{where}", flush=True)
        return 1
    print(f"[gate] PASS: {len(verdicts)}/{len(verdicts)} probes hold{where}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
