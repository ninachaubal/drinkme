"""The exactness gate's driver: a FIXED request matrix against a running
drinkme server, greedy, one transcript sha per case, plus where the server's
TTFT came from — so a server built before the seam change and one built
after can be held sha-identical case by case.

    python bench/genreq_gate.py --base-url http://127.0.0.1:3299 -o before.json

The matrix: three dialects (chat completions, Responses, Messages) x
thinking on/off x stream on/off, plus one json_schema request and one
tool-call request through the chat dialect — 14 cases. Every case is
temperature 0 with a fixed max_tokens, so a differing sha is a differing
generation, not sampling.

The transcript of a case is the assembled reply in a wire-neutral shape
({reasoning, content, tool_calls, finish, usage}) — the streamed and the
non-streamed form of one request assemble to the same shape, and each
dialect's own ids/timestamps are left out. The sha is over its canonical
JSON.

TTFT: the server's own `drinkme_ttft_seconds` histogram on /metrics is read
before and after each case; the count moving by exactly 1 says the server
observed one TTFT for the case (serving/metrics.py's definition: the first
delta carrying reasoning or content), and the sum's delta is that value.
The client-side first-delta time is recorded beside it for streamed cases.

Verdict-free on its own: this script records; the compare step is
`python bench/genreq_gate.py --compare before.json after.json`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from serve_dialect_smoke import WEATHER_TOOL, post, stream  # noqa: E402

QUESTION = "What is 17 * 23? Reply with just the number."
TITLE_SCHEMA = {"type": "object", "properties": {"title": {"type": "string"}},
                "required": ["title"], "additionalProperties": False}
ANTHROPIC_TOOL = {"name": "get_weather", "description": "Current weather for a city.",
                  "input_schema": WEATHER_TOOL["function"]["parameters"]}


def metrics_ttft(base: str) -> tuple[int, float]:
    with urllib.request.urlopen(base + "/metrics", timeout=30) as r:
        text = r.read().decode()
    count = total = None
    for line in text.splitlines():
        if line.startswith("drinkme_ttft_seconds_count "):
            count = int(float(line.split()[1]))
        elif line.startswith("drinkme_ttft_seconds_sum "):
            total = float(line.split()[1])
    if count is None or total is None:
        raise SystemExit("no drinkme_ttft_seconds on /metrics")
    return count, total


def sse(base: str, path: str, body: dict, timeout: float):
    """Every SSE frame as (event-name-or-None, parsed data), with the wall
    time of the first frame that carries text — the client-side TTFT."""
    body = dict(body, stream=True)
    req = urllib.request.Request(base + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    frames, name, t_first = [], None, None
    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        status = r.status
        for raw in r:
            line = raw.decode(errors="replace").rstrip("\r\n")
            if line.startswith("event:"):
                name = line[6:].strip()
                continue
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            data = json.loads(payload)
            if t_first is None and _carries_text(name, data):
                t_first = time.monotonic() - t0
            frames.append((name, data))
            name = None
    return status, frames, t_first


def _carries_text(name, data) -> bool:
    if name is None:  # chat completions chunk
        for ch in data.get("choices", []):
            d = ch.get("delta", {})
            if d.get("content") or d.get("reasoning_content"):
                return True
        return False
    if name in ("response.output_text.delta", "response.reasoning_summary_text.delta"):
        return bool(data.get("delta"))
    if name == "content_block_delta":
        d = data.get("delta", {})
        return bool(d.get("text") or d.get("thinking"))
    return False


# ------------------------------------------------------------ assemblers --


def chat_obj(resp: dict) -> dict:
    m = resp["choices"][0]["message"]
    calls = [{"name": c["function"]["name"], "arguments": json.loads(c["function"]["arguments"])}
             for c in m.get("tool_calls") or []]
    return {"reasoning": m.get("reasoning_content") or "", "content": m.get("content") or "",
            "tool_calls": calls, "finish": resp["choices"][0]["finish_reason"],
            "usage": {"prompt": resp["usage"]["prompt_tokens"],
                      "completion": resp["usage"]["completion_tokens"]}}


def chat_stream(frames) -> dict:
    content, reasoning, calls, finish, usage = [], [], {}, None, None
    for _, c in frames:
        if c.get("usage") and not c.get("choices"):
            usage = {"prompt": c["usage"]["prompt_tokens"],
                     "completion": c["usage"]["completion_tokens"]}
        for ch in c.get("choices", []):
            d = ch.get("delta", {})
            if d.get("content"):
                content.append(d["content"])
            if d.get("reasoning_content"):
                reasoning.append(d["reasoning_content"])
            for tc in d.get("tool_calls") or []:
                slot = calls.setdefault(tc.get("index", 0), {"name": "", "arguments": ""})
                slot["name"] += tc["function"].get("name", "") or ""
                slot["arguments"] += tc["function"].get("arguments", "") or ""
            if ch.get("finish_reason"):
                finish = ch["finish_reason"]
    return {"reasoning": "".join(reasoning), "content": "".join(content),
            "tool_calls": [{"name": calls[k]["name"], "arguments": json.loads(calls[k]["arguments"])}
                           for k in sorted(calls)],
            "finish": finish, "usage": usage}


def responses_obj(resp: dict) -> dict:
    reasoning, content, calls = [], [], []
    for item in resp["output"]:
        if item["type"] == "reasoning":
            reasoning += [p["text"] for p in item.get("summary") or [] if p.get("type") == "summary_text"]
            reasoning += [p["text"] for p in item.get("content") or [] if p.get("type") == "reasoning_text"]
        elif item["type"] == "message":
            content += [p["text"] for p in item["content"] if p["type"] == "output_text"]
        elif item["type"] == "function_call":
            calls.append({"name": item["name"], "arguments": json.loads(item["arguments"])})
    inc = resp.get("incomplete_details") or {}
    finish = ("tool_calls" if calls else
              "length" if inc.get("reason") == "max_output_tokens" else
              "stop" if resp["status"] == "completed" else resp["status"])
    return {"reasoning": "".join(reasoning), "content": "".join(content), "tool_calls": calls,
            "finish": finish, "usage": {"prompt": resp["usage"]["input_tokens"],
                                        "completion": resp["usage"]["output_tokens"]}}


def responses_stream(frames) -> dict:
    final = next((d["response"] for n, d in frames
                  if n in ("response.completed", "response.incomplete")), None)
    if final is None:
        raise SystemExit("responses stream ended without response.completed/incomplete")
    out = responses_obj(final)
    # and the deltas must concatenate to the same text the final object carries
    text = "".join(d["delta"] for n, d in frames if n == "response.output_text.delta")
    if text != out["content"]:
        raise SystemExit("responses stream deltas != final object text")
    return out


def messages_obj(resp: dict) -> dict:
    reasoning, content, calls = [], [], []
    for b in resp["content"]:
        if b["type"] == "thinking":
            reasoning.append(b["thinking"])
        elif b["type"] == "text":
            content.append(b["text"])
        elif b["type"] == "tool_use":
            calls.append({"name": b["name"], "arguments": b["input"]})
    finish = {"end_turn": "stop", "max_tokens": "length", "tool_use": "tool_calls",
              "stop_sequence": "stop"}.get(resp["stop_reason"], resp["stop_reason"])
    u = resp["usage"]
    return {"reasoning": "".join(reasoning), "content": "".join(content), "tool_calls": calls,
            "finish": finish,
            "usage": {"prompt": u["input_tokens"] + u.get("cache_read_input_tokens", 0),
                      "completion": u["output_tokens"]}}


def messages_stream(frames) -> dict:
    blocks: dict[int, dict] = {}
    stop_reason, out_tokens, in_tokens = None, None, None
    for n, d in frames:
        if n == "message_start":
            in_tokens = (d["message"]["usage"]["input_tokens"]
                         + d["message"]["usage"].get("cache_read_input_tokens", 0))
        elif n == "content_block_start":
            b = dict(d["content_block"])
            b.setdefault("_json", "")
            blocks[d["index"]] = b
        elif n == "content_block_delta":
            b, delta = blocks[d["index"]], d["delta"]
            if delta["type"] == "text_delta":
                b["text"] = b.get("text", "") + delta["text"]
            elif delta["type"] == "thinking_delta":
                b["thinking"] = b.get("thinking", "") + delta["thinking"]
            elif delta["type"] == "input_json_delta":
                b["_json"] += delta["partial_json"]
        elif n == "message_delta":
            stop_reason = d["delta"]["stop_reason"]
            out_tokens = d["usage"]["output_tokens"]
    content = []
    for i in sorted(blocks):
        b = blocks[i]
        if b["type"] == "tool_use":
            b["input"] = json.loads(b["_json"]) if b["_json"] else b.get("input", {})
        content.append(b)
    return messages_obj({"content": content, "stop_reason": stop_reason,
                         "usage": {"input_tokens": in_tokens, "output_tokens": out_tokens}})


# ------------------------------------------------------------- the matrix --


def cases(model: str, max_tokens: int):
    q = [{"role": "user", "content": QUESTION}]
    for think in (True, False):
        for st in (False, True):
            tag = f"think={'on' if think else 'off'}/{'stream' if st else 'json'}"
            yield (f"chat/{tag}", "/v1/chat/completions", st,
                   {"model": model, "messages": q, "temperature": 0, "max_tokens": max_tokens,
                    "chat_template_kwargs": {"enable_thinking": think},
                    **({"stream_options": {"include_usage": True}} if st else {})})
            yield (f"responses/{tag}", "/v1/responses", st,
                   {"model": model, "input": QUESTION, "temperature": 0,
                    "max_output_tokens": max_tokens,
                    "chat_template_kwargs": {"enable_thinking": think}})
            yield (f"messages/{tag}", "/v1/messages", st,
                   {"model": model, "messages": q, "temperature": 0, "max_tokens": max_tokens,
                    "thinking": {"type": "enabled" if think else "disabled",
                                 **({"budget_tokens": 1024} if think else {})}})
    yield ("chat/json_schema", "/v1/chat/completions", False,
           {"model": model, "temperature": 0, "max_tokens": max_tokens,
            "messages": [{"role": "user", "content":
                          "Give a short title for a story about a lighthouse keeper."}],
            "response_format": {"type": "json_schema",
                                "json_schema": {"name": "title", "schema": TITLE_SCHEMA}}})
    yield ("chat/tool_call", "/v1/chat/completions", False,
           {"model": model, "temperature": 0, "max_tokens": max_tokens,
            "tools": [WEATHER_TOOL], "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "user", "content":
                          "What's the weather in Hilo right now? Use the tool."}]})


ASSEMBLE = {"/v1/chat/completions": (chat_obj, chat_stream),
            "/v1/responses": (responses_obj, responses_stream),
            "/v1/messages": (messages_obj, messages_stream)}


def run(base: str, out_path: str | None, max_tokens: int, timeout: float) -> dict:
    with urllib.request.urlopen(base + "/v1/models", timeout=30) as r:
        models = json.loads(r.read())
    with urllib.request.urlopen(base + "/health", timeout=30) as r:
        health = json.loads(r.read())
    model = models["data"][0]["id"]
    receipt = {"base_url": base, "model": model, "health": health,
               "drinkme": models["data"][0]["drinkme"], "max_tokens": max_tokens,
               "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "cases": {}}
    print(f"[gate] {model} on {health.get('device')} · arm {models['data'][0]['drinkme'].get('arm')}")
    for name, path, st, body in cases(model, max_tokens):
        c0, s0 = metrics_ttft(base)
        t0 = time.monotonic()
        if st:
            status, frames, t_first = sse(base, path, body, timeout)
            obj = ASSEMBLE[path][1](frames) if status == 200 else {"error": frames}
        else:
            status, resp = post(base, path, body, timeout)
            obj = ASSEMBLE[path][0](resp) if status == 200 else {"error": resp}
            t_first = None
        wall = time.monotonic() - t0
        c1, s1 = metrics_ttft(base)
        canon = json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        sha = hashlib.sha256(canon.encode()).hexdigest()
        rec = {"status": status, "sha256": sha, "transcript": obj, "wall_s": round(wall, 3),
               "server_ttft_observations": c1 - c0,
               "server_ttft_s": round(s1 - s0, 4) if c1 - c0 == 1 else None,
               "client_ttft_s": None if t_first is None else round(t_first, 4)}
        receipt["cases"][name] = rec
        print(f"[gate] {name:28s} {status} sha {sha[:12]} · finish {obj.get('finish')!s:10s} "
              f"· {obj.get('usage', {}).get('completion') if isinstance(obj.get('usage'), dict) else '?'} tok "
              f"· server ttft {rec['server_ttft_s']} ({rec['server_ttft_observations']} obs)"
              f" · client ttft {rec['client_ttft_s']}")
    receipt["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    if out_path:
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(receipt, f, indent=1, ensure_ascii=False)
        print(f"[gate] receipt → {out_path}")
    return receipt


def compare(before_path: str, after_path: str) -> int:
    before, after = json.load(open(before_path)), json.load(open(after_path))
    bad = []
    print(f"{'case':28s} {'before sha':14s} {'after sha':14s} {'ttft obs':9s} before/after ttft (s)")
    for name in before["cases"]:
        b, a = before["cases"][name], after["cases"].get(name)
        if a is None:
            bad.append(f"{name}: missing after"); continue
        same = b["sha256"] == a["sha256"]
        obs = f"{b['server_ttft_observations']}/{a['server_ttft_observations']}"
        print(f"{name:28s} {b['sha256'][:12]}   {a['sha256'][:12]}   {obs:9s} "
              f"{b['server_ttft_s']} / {a['server_ttft_s']}{'' if same else '   *** DIFFERS ***'}")
        if not same:
            bad.append(f"{name}: sha differs")
        if b["server_ttft_observations"] != a["server_ttft_observations"]:
            bad.append(f"{name}: server TTFT observed {obs} times — a different event")
    zero = [n for n, c in after["cases"].items() if c["server_ttft_observations"] == 0]
    if zero:
        # a reply that is ALL tool call never carries reasoning or content, so
        # the metric's definition never fires — the same on both arms; noted,
        # not a difference
        print(f"note: no server TTFT observed on either arm for {zero} "
              "(the reply carried no reasoning/content delta)")
    print("VERDICT: " + ("ALL CASES SHA-IDENTICAL, TTFT OBSERVED THE SAME NUMBER OF TIMES PER CASE"
                         if not bad else "; ".join(bad)))
    return 0 if not bad else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--base-url", default="http://127.0.0.1:3299")
    ap.add_argument("--max-tokens", type=int, default=96)
    ap.add_argument("--timeout", type=float, default=900.0)
    ap.add_argument("-o", "--out")
    ap.add_argument("--compare", nargs=2, metavar=("BEFORE", "AFTER"))
    args = ap.parse_args()
    if args.compare:
        return compare(*args.compare)
    run(args.base_url.rstrip("/"), args.out, args.max_tokens, args.timeout)
    return 0


if __name__ == "__main__":
    sys.exit(main())
