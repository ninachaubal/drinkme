"""Dialect smoke against a RUNNING drinkme server: does this model's reasoning
stay out of `content`, and does its tool-call dialect round-trip through the
server's table row — both directions, streamed and not?

This is the gate that flips a tool_formats row's `tested` flag. The static
capability probe (serving/capability.py) reads the chat template and names
the row; this script asks the real weights whether the row's parser and the
think splitter actually hold on what the model emits.

    uv run --no-sync python bench/serve_dialect_smoke.py \
        --base-url http://127.0.0.1:3216 -o measurements/dialect_smoke_<model>.json

Verdicts printed, one per probe, and every raw response kept in the receipt.
A verdict is a claim about THIS server on THIS model; the exit code is 0 when
every probe holds and 1 otherwise, and the receipt says which one did not.
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

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name"},
                "unit": {"type": "string", "enum": ["c", "f"]},
            },
            "required": ["city"],
        },
    },
}

# No temperature is sent: the server applies the model's own generation_config
# defaults, which is what a real client gets. Forcing 0 on a thinking model
# (Qwen3's card: "DO NOT use greedy decoding" in thinking mode) loops the
# reasoning — measured on Qwen3-8B, 3,650 chars of "Wait, hold on".

# Any of these in `content` means a reasoning or tool channel leaked into the
# user-visible text. Deliberately broad: the point is to catch what the table
# row did NOT anticipate.
LEAK_MARKERS = [
    "<think>", "</think>", "<|channel>", "<channel|>", "<|tool_call>",
    "<tool_call|>", "<tool_call>", "<function=", "<start_function_call>",
    "<|tool>", "[TOOL_CALLS]", "<minimax:tool_call>", "<arg_key>",
    # Muse-Glimmer's ATEM message headers, its message end, and its call block
    "<|start|>", "<|message|>", "<|eom|>", "to=self", "to=user", "<atem:",
]


def post(base: str, path: str, body: dict, timeout: float) -> tuple[int, dict | str]:
    req = urllib.request.Request(base + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        raw = e.read().decode(errors="replace")
        try:
            return e.code, json.loads(raw)
        except json.JSONDecodeError:
            return e.code, raw


def stream(base: str, path: str, body: dict, timeout: float) -> tuple[int, list[dict]]:
    body = dict(body, stream=True)
    req = urllib.request.Request(base + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    chunks: list[dict] = []
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            for line in r:
                line = line.decode(errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                chunks.append(json.loads(payload))
            return r.status, chunks
    except urllib.error.HTTPError as e:
        return e.code, [{"error": e.read().decode(errors="replace")}]


def leaks(text: str | None) -> list[str]:
    return [m for m in LEAK_MARKERS if text and m in text]


def assemble(chunks: list[dict]) -> dict:
    """Fold OpenAI stream deltas into one message shape."""
    content, reasoning = [], []
    calls: dict[int, dict] = {}
    finish = None
    for c in chunks:
        for ch in c.get("choices", []):
            d = ch.get("delta", {})
            if d.get("content"):
                content.append(d["content"])
            if d.get("reasoning_content"):
                reasoning.append(d["reasoning_content"])
            for tc in d.get("tool_calls", []) or []:
                slot = calls.setdefault(tc.get("index", 0), {"name": "", "arguments": ""})
                fn = tc.get("function", {})
                slot["name"] += fn.get("name", "") or ""
                slot["arguments"] += fn.get("arguments", "") or ""
            if ch.get("finish_reason"):
                finish = ch["finish_reason"]
    return {"content": "".join(content) or None,
            "reasoning_content": "".join(reasoning) or None,
            "tool_calls": [calls[k] for k in sorted(calls)],
            "finish_reason": finish}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--base-url", default="http://127.0.0.1:3215")
    ap.add_argument("--model", default=None, help="model id to send (default: first on /v1/models)")
    ap.add_argument("--timeout", type=float, default=900.0)
    ap.add_argument("--max-tokens", type=int, default=1500,
                    help="per probe; a thinking model spends most of it reasoning before the answer")
    ap.add_argument("-o", "--out", help="receipt JSON path")
    ap.add_argument("--system", default=None,
                    help="a system turn on every request (Muse-Glimmer reads its reasoning "
                         "strength there: 'Reasoning strength: low.')")
    ap.add_argument("--only", default=None,
                    help="comma-separated probe names to run (plain, plain_stream, tool_out, "
                         "tool_out_stream, tool_in, anthropic_tool); default all")
    args = ap.parse_args()
    base = args.base_url.rstrip("/")
    only = set(args.only.split(",")) if args.only else None
    want = lambda name: only is None or name in only  # noqa: E731

    receipt: dict = {"base_url": base, "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "probes": {},
                     "system": args.system}
    sys_turn = [{"role": "system", "content": args.system}] if args.system else []
    verdicts: dict[str, bool] = {}

    with urllib.request.urlopen(base + "/v1/models", timeout=30) as r:
        models = json.loads(r.read())
    receipt["models"] = models
    model = args.model or models["data"][0]["id"]
    announced = next((m for m in models["data"] if m["id"] == model), models["data"][0])
    caps = announced["drinkme"]["capabilities"]
    print(f"[smoke] model {model} · announced: thinking={caps.get('thinking')} "
          f"toolFormat={caps.get('toolFormat')}")

    # ---- probe 1: plain chat, does reasoning stay out of content? ----------
    body = {"model": model, "max_tokens": args.max_tokens,
            "messages": sys_turn + [{"role": "user", "content":
                                     "What is 17 * 23? Reply with just the number."}]}
    status, resp = post(base, "/v1/chat/completions", body, args.timeout)
    msg = resp["choices"][0]["message"] if status == 200 else {}
    lk = leaks(msg.get("content"))
    ok = status == 200 and not lk and "391" in (msg.get("content") or "")
    verdicts["plain_no_leak"] = ok
    receipt["probes"]["plain"] = {"status": status, "response": resp, "leaks": lk}
    print(f"[smoke] plain: {status} · content leaks={lk or 'none'} · "
          f"reasoning_content={'yes' if msg.get('reasoning_content') else 'no'} · "
          f"answer_present={'391' in (msg.get('content') or '')} → {'OK' if ok else 'FAIL'}")

    # ---- probe 2: same, streamed ----------------------------------------------
    if want("plain_stream"):
        status, chunks = stream(base, "/v1/chat/completions", body, args.timeout)
        asm = assemble(chunks) if status == 200 else {}
        lk = leaks(asm.get("content"))
        ok = status == 200 and not lk and "391" in (asm.get("content") or "")
        verdicts["plain_stream_no_leak"] = ok
        receipt["probes"]["plain_stream"] = {"status": status, "assembled": asm, "n_chunks": len(chunks), "leaks": lk}
        print(f"[smoke] plain/stream: {status} · {len(chunks)} chunks · leaks={lk or 'none'} → {'OK' if ok else 'FAIL'}")
    if only is not None and not (only - {"plain", "plain_stream"}):
        receipt["verdicts"] = verdicts
        receipt["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        if args.out:
            os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
            with open(args.out, "w") as f:
                json.dump(receipt, f, indent=1)
            print(f"[smoke] receipt → {args.out}")
        allok = all(verdicts.values())
        print(f"[smoke] {'ALL HOLD' if allok else 'FAILED: ' + ', '.join(k for k, v in verdicts.items() if not v)}")
        return 0 if allok else 1

    # ---- probe 3: tool call OUT — does the row's parser lift the call? -------
    tbody = {"model": model, "max_tokens": args.max_tokens,
             "tools": [WEATHER_TOOL],
             "messages": sys_turn + [{"role": "user", "content":
                                      "What's the weather in Hilo right now? Use the tool."}]}
    status, resp = post(base, "/v1/chat/completions", tbody, args.timeout)
    msg = resp["choices"][0]["message"] if status == 200 else {}
    calls = msg.get("tool_calls") or []
    lk = leaks(msg.get("content"))
    parsed = bool(calls) and calls[0]["function"]["name"] == "get_weather"
    argok = False
    if parsed:
        try:
            argok = "hilo" in json.loads(calls[0]["function"]["arguments"]).get("city", "").lower()
        except (json.JSONDecodeError, AttributeError):
            argok = False
    ok = status == 200 and parsed and argok and not lk
    verdicts["tool_call_out"] = ok
    receipt["probes"]["tool_out"] = {"status": status, "response": resp, "leaks": lk}
    print(f"[smoke] tool out: {status} · parsed={parsed} args_ok={argok} leaks={lk or 'none'} "
          f"finish={resp.get('choices', [{}])[0].get('finish_reason') if status == 200 else '-'} → {'OK' if ok else 'FAIL'}")

    # ---- probe 4: tool call OUT, streamed -------------------------------------
    status, chunks = stream(base, "/v1/chat/completions", tbody, args.timeout)
    asm = assemble(chunks) if status == 200 else {}
    scalls = asm.get("tool_calls") or []
    lk = leaks(asm.get("content"))
    sparsed = bool(scalls) and scalls[0]["name"] == "get_weather"
    ok = status == 200 and sparsed and not lk
    verdicts["tool_call_out_stream"] = ok
    receipt["probes"]["tool_out_stream"] = {"status": status, "assembled": asm, "n_chunks": len(chunks), "leaks": lk}
    print(f"[smoke] tool out/stream: {status} · parsed={sparsed} leaks={lk or 'none'} → {'OK' if ok else 'FAIL'}")

    # ---- probe 5: tool result IN — does the row's emitter render the history? --
    if verdicts["tool_call_out"]:
        call = calls[0]
        rbody = {"model": model, "max_tokens": args.max_tokens,
                 "tools": [WEATHER_TOOL],
                 "messages": tbody["messages"] + [
                     {"role": "assistant", "content": msg.get("content") or None, "tool_calls": calls},
                     {"role": "tool", "tool_call_id": call["id"],
                      "content": json.dumps({"city": "Hilo", "temp_c": 27, "sky": "light rain"})},
                 ]}
        status, resp = post(base, "/v1/chat/completions", rbody, args.timeout)
        msg2 = resp["choices"][0]["message"] if status == 200 else {}
        text = (msg2.get("content") or "")
        lk = leaks(text)
        ok = status == 200 and not lk and bool(re.search(r"27|rain", text, re.I)) and not msg2.get("tool_calls")
        verdicts["tool_result_in"] = ok
        receipt["probes"]["tool_in"] = {"status": status, "response": resp, "leaks": lk}
        print(f"[smoke] tool in: {status} · used_result={bool(re.search(r'27|rain', text, re.I))} "
              f"re-called={bool(msg2.get('tool_calls'))} leaks={lk or 'none'} → {'OK' if ok else 'FAIL'}")
    else:
        verdicts["tool_result_in"] = False
        print("[smoke] tool in: skipped (no parsed call to feed back)")

    # ---- probe 6: the Anthropic dialect with a tool (the Claude Code shape) ----
    abody = {"model": model, "max_tokens": args.max_tokens,
             "tools": [{"name": "get_weather", "description": "Current weather for a city.",
                        "input_schema": WEATHER_TOOL["function"]["parameters"]}],
             "messages": [{"role": "user", "content": "What's the weather in Hilo right now? Use the tool."}]}
    if args.system:
        abody["system"] = args.system
    status, resp = post(base, "/v1/messages", abody, args.timeout)
    blocks = resp.get("content", []) if status == 200 else []
    tool_use = [b for b in blocks if b.get("type") == "tool_use"]
    texts = "".join(b.get("text", "") for b in blocks if b.get("type") == "text")
    lk = leaks(texts)
    ok = status == 200 and bool(tool_use) and tool_use[0].get("name") == "get_weather" and not lk
    verdicts["anthropic_tool_use"] = ok
    receipt["probes"]["anthropic_tool"] = {"status": status, "response": resp, "leaks": lk}
    print(f"[smoke] /v1/messages tool: {status} · tool_use={bool(tool_use)} stop={resp.get('stop_reason') if status == 200 else '-'} "
          f"leaks={lk or 'none'} → {'OK' if ok else 'FAIL'}")

    receipt["verdicts"] = verdicts
    receipt["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(receipt, f, indent=1)
        print(f"[smoke] receipt → {args.out}")
    allok = all(verdicts.values())
    print(f"[smoke] {'ALL HOLD' if allok else 'FAILED: ' + ', '.join(k for k, v in verdicts.items() if not v)}")
    return 0 if allok else 1


if __name__ == "__main__":
    sys.exit(main())
