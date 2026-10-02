"""Image input end to end on a real model, over the wire: the screenshot smoke,
the pixel-cap readability ladder, the prefix cache with images, speculation
with images, and sleep/wake with the tower.

One process: engines via serve.build_engine (the compressed pack), served by
http.start_server on an OS-chosen port (never 3215), a fresh cold-tier
directory under --out (DRINKME_SLOT_DIR), two live prefix slots. Screenshots
are drawn at run time (bench/vision_screens.py). Every request is greedy with
thinking off. The engine's own log lines ([drinkme.engine] images: ...,
prefix cache ..., [drinkme.spec] ...) are captured from stderr and kept per
request.

  smoke  A 2560x1440 terminal screenshot carrying two nonces: the deploy
         token on the terminal's last line (15 px monospace) and a table
         cell (13 px UI text). Asked through /v1/chat/completions (data
         URL), /v1/messages (base64 block, and inside a tool_result after a
         tool_use), and /v1/responses (input_image). PASS when every reply
         carries both nonces. Then the same screenshot drawn at Retina scale
         (2880x1800, 2x UI), at the default cap.
  ladder A readability ladder: one line per font size, each with its own
         code, on a Retina 2880x1800 capture at the default cap (resized to
         ~2432x1504), the same capture with the cap raised to its own size,
         and the 1x 1440x900 capture. Reports which sizes were read exactly.
  cache  A three-turn image conversation, warm (the prefix cache on) and
         cold (off): the same greedy transcript, and the tower skipped for
         the image inside the reused prefix. Two different same-size
         screenshots, asked the same question in turn: each answer carries
         its own image's nonce. A cold-tier restore: two text conversations
         evict the image conversation's live slot, and its next turn is
         restored from the tier with the tower skipped.
  spec   With the image in context, the same greedy request under
         DRINKME_SPEC=off, ngram and auto (MTP): identical transcripts (a
         difference is reported with the first differing token), and the
         acceptance with vs without the image (the same task over the
         terminal's text).
  sleep  POST /sleep?level=1 and /wake_up around the image conversation:
         the tower's parameters leave the device and come back bit for bit,
         the parked slot is restored at wake, and the next turn skips the
         tower.
  agent  The prefix cache in a Messages-API tool loop inside one user turn
         (screenshot, then a lookup, then the answer): the request after the
         screenshot's tool_result reuses the prefix that holds the image and
         skips the tower; warm equals cold; a cold-tier restore of the loop
         skips it too. --agent-thinking sends thinking on and passes every
         thinking block back.

On gemma-4-31B-it no history a client sends back extends what the model
generated, so `cache` and `agent` find nothing to reuse there: its template
drops the empty `<|channel>thought\n<channel|>` that its thinking-off
generation prompt ends in and that the model itself writes after a tool
response, and with thinking passed back it writes a newline before
`<channel|>` that the model did not (measured).
  sleep2 POST /sleep?level=2 and /wake_up: the engine is rebuilt by its
         loader, the tower comes back the same bytes, its boot attention
         self-test runs again, and an image request after it reads the nonce.

--model gemma serves gemma-4-31B-it (its speculation's `auto` is n-gram: no
MTP head); give it --ui-scale 2.4 (UI_SCALE).

--model glimmer serves Muse-Glimmer-30B (no MTP head either). It reasons in
a ` to=self` message before its `to=user` answer, which the server returns
as `reasoning_content` and `content` (the `atem` row; docs/serve-tool-formats.md,
"Addressed messages"). So every request carries a "Reasoning strength: low."
system turn (SYSTEM) and REASONING_TOKENS more, and history goes back with
the reasoning (assistant_turn()): the template renders a `to=self` message
only from `reasoning_content`, and without it no later turn extends the slot.
Its tool calls are the `atem` row, in the tested tier, so the tool requests
(smoke's tool_result, `agent`) run. A model whose row is outside the tested
tier has them refused by the server, and skipped here unless
DRINKME_TOOLS_UNTESTED=1.

Hold `flock -w 3600 /tmp/drinkme-gpu.lock`. Set
TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1, as the served units are.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import http.client
import json
import os
import random
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from vision_screens import png_bytes, screenshot, small_text_screenshot  # noqa: E402

MODELS = {"27b": "Qwen/Qwen3.8-27B", "mimo": "XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B",
          "gemma": "google/gemma-4-31B-it", "glimmer": "meta-models/Muse-Glimmer-30B"}
# The 2560x1440 screenshots' UI scale (--ui-scale). gemma-4 resizes every
# image to its own 280-token budget (2560x1440 -> 1056x576), so at 1x the
# nonce's 15 px monospace would be 6 px; at 2.4 it is 15 px again. The
# ladder and the Retina capture keep their own scales: they measure
# resolution.
UI_SCALE = 1.0
# Muse-Glimmer (the module docstring): a system turn on every request, and
# the tokens its reasoning channel takes on top of each request's own budget
SYSTEM: str | None = None
REASONING_TOKENS = 0
NONCE, CELL = "PELICAN-7391", "OSPREY-2846"
NONCE_B, CELL_B = "KESTREL-5082", "PLOVER-6613"
Q_BOTH = ("Two tokens are shown in this screenshot: the deploy token on the terminal's last "
          "output line, and the Token in the bottle row of the Deployments table. Reply with "
          "just those two tokens, separated by a space.")
LADDER_PT = [6, 7, 8, 9, 10, 11, 12, 14]
Q_LADDER = ("The image lists lines of the form 'label N: code XXX-999'. Transcribe every such "
            "line you can read, exactly, one per line, and nothing else.")
Q_TRANSCRIBE = ("Transcribe the git log lines shown in the terminal (the lines after "
                "'$ git log --oneline -6'), exactly, one per line, and nothing else.")
Q_DESCRIBE = "Describe what this screenshot shows in about 80 words."


# ------------------------------------------------------------- plumbing --

class Tee:
    """stderr and stdout, kept: the engine's log lines, per request (its
    boot lines, which a level-2 wake prints again, go to stdout). A second
    Tee made with `share` writes into the first one's buffer."""

    def __init__(self, inner, share=None):
        self.inner = inner
        self.buf, self.lock = (share.buf, share.lock) if share else ([], threading.Lock())

    def write(self, s):
        with self.lock:
            self.buf.append(s)
        return self.inner.write(s)

    def flush(self):
        self.inner.flush()

    def __getattr__(self, name):  # isatty, fileno, encoding: the stream's own
        return getattr(self.inner, name)

    def mark(self) -> int:
        with self.lock:
            return len(self.buf)

    def since(self, i: int) -> list[str]:
        with self.lock:
            text = "".join(self.buf[i:])
        return [ln for ln in text.splitlines() if ln.startswith("[drinkme")]


class Client:
    def __init__(self, port: int, tee: Tee, model: str):
        self.port, self.tee, self.model = port, tee, model

    def call(self, method: str, path: str, body=None, timeout=900):
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=timeout)
        mark = self.tee.mark()
        t0 = time.perf_counter()
        c.request(method, path, None if body is None else json.dumps(body),
                  {"Content-Type": "application/json"})
        r = c.getresponse()
        raw = r.read()
        wall = time.perf_counter() - t0
        c.close()
        try:
            data = json.loads(raw)
        except ValueError:
            data = raw.decode(errors="replace")
        return {"status": r.status, "data": data, "wall_s": round(wall, 2),
                "log": self.tee.since(mark)}

    def chat(self, messages, max_tokens=48):
        if SYSTEM:
            messages = [{"role": "system", "content": SYSTEM}] + messages
        res = self.call("POST", "/v1/chat/completions",
                        {"messages": messages, "temperature": 0,
                         "max_tokens": max_tokens + REASONING_TOKENS,
                         "chat_template_kwargs": {"enable_thinking": False}})
        d = res["data"]
        msg = d["choices"][0]["message"] if res["status"] == 200 else {}
        res["text"] = msg.get("content") or ""
        res["reasoning"] = msg.get("reasoning_content") or ""
        res["usage"] = d.get("usage") if isinstance(d, dict) else None
        return res

    def messages(self, messages, max_tokens=48, tools=None, thinking=False):
        body = {"model": self.model, "messages": messages,
                "max_tokens": max_tokens + REASONING_TOKENS, "temperature": 0,
                "thinking": ({"type": "enabled", "budget_tokens": max_tokens} if thinking
                             else {"type": "disabled"})}
        if SYSTEM:
            body["system"] = SYSTEM
        if tools:
            body["tools"] = tools
        res = self.call("POST", "/v1/messages", body)
        d = res["data"]
        res["text"] = ("".join(b.get("text", "") for b in d.get("content", [])
                               if b.get("type") == "text") if res["status"] == 200 else "")
        res["usage"] = d.get("usage") if isinstance(d, dict) else None
        return res

    def responses(self, content, max_tokens=48):
        res = self.call("POST", "/v1/responses",
                        {"input": [{"role": "user", "content": content}], "temperature": 0,
                         "max_output_tokens": max_tokens + REASONING_TOKENS,
                         "chat_template_kwargs": {"enable_thinking": False},
                         **({"instructions": SYSTEM} if SYSTEM else {})})
        d = res["data"]
        text = ""
        if res["status"] == 200:
            for item in d.get("output", []):
                for part in item.get("content", []) or []:
                    if part.get("type") == "output_text":
                        text += part.get("text", "")
        res["text"], res["usage"] = text, (d.get("usage") if isinstance(d, dict) else None)
        return res


def data_url(png: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(png).decode()


def oa_user(png: bytes | None, text: str):
    parts = [] if png is None else [{"type": "image_url", "image_url": {"url": data_url(png)}}]
    return {"role": "user", "content": parts + [{"type": "text", "text": text}]}


def assistant_turn(res) -> dict:
    """A chat reply as the next request's history, its reasoning included
    when it had any, as a client that keeps the prefix cache sends it
    (Muse-Glimmer's template renders a `to=self` message only from
    `reasoning_content`; the module docstring)."""
    turn = {"role": "assistant", "content": res["text"]}
    if res.get("reasoning"):
        turn["reasoning_content"] = res["reasoning"]
    return turn


def both(text: str, a=NONCE, b=CELL) -> bool:
    return a in text and b in text


def images_line(res) -> str | None:
    return next((ln for ln in res["log"] if ln.startswith("[drinkme.engine] images")), None)


def tower_skipped(res) -> bool:
    ln = images_line(res) or ""
    return "the tower ran 0x" in ln and "1 inside the reused prefix" in ln


def cached(res) -> int:
    u = res.get("usage") or {}
    return int((u.get("prompt_tokens_details") or {}).get("cached_tokens")
               or u.get("cache_read_input_tokens") or 0)


def short(res) -> dict:
    return {k: res.get(k) for k in ("status", "text", "wall_s", "usage", "log")}


# --------------------------------------------------------------- phases --

def phase_smoke(cl: Client, eng) -> dict:
    png = png_bytes(screenshot(2560, 1440, nonce=NONCE, cell=CELL, scale=UI_SCALE))
    b64 = base64.b64encode(png).decode()
    out = {}
    out["chat"] = cl.chat([oa_user(png, Q_BOTH)])
    out["messages"] = cl.messages([{"role": "user", "content": [
        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b64}},
        {"type": "text", "text": Q_BOTH}]}])
    tools = [{"name": "screenshot", "description": "Capture the user's screen.",
              "input_schema": {"type": "object", "properties": {}}}]
    from drinkme.serving import capability

    if eng.capability.tool_format not in capability.served_tool_formats():
        tools = None  # a tools request is refused
    out["tool_result"] = None if tools is None else cl.messages([
        {"role": "user", "content": "Take a screenshot of my screen, then answer: " + Q_BOTH},
        {"role": "assistant", "content": [{"type": "tool_use", "id": "toolu_01", "name": "screenshot",
                                           "input": {}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_01", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b64}}]}]},
    ], tools=tools)
    out["responses"] = cl.responses([{"type": "input_image", "image_url": data_url(png)},
                                     {"type": "input_text", "text": Q_BOTH}])
    retina = png_bytes(screenshot(2880, 1800, nonce=NONCE, cell=CELL, scale=2.0))
    out["retina_at_cap"] = cl.chat([oa_user(retina, Q_BOTH)])
    # the same capture with the cap raised to its own size: whether a miss
    # at the cap is the cap's (informational, not part of PASS)
    import dataclasses

    base = eng.vision
    try:
        eng.vision = dataclasses.replace(base, max_pixels=2880 * 1800)
        raised = cl.chat([oa_user(retina, Q_BOTH)])
    finally:
        eng.vision = base
    res = {k: dict(short(v), pass_=both(v["text"])) for k, v in out.items() if v is not None}
    res["pass"] = all(v["pass_"] for v in res.values() if isinstance(v, dict))
    res["retina_cap_raised"] = dict(short(raised), pass_=both(raised["text"]))
    return res


def _codes(seed: int) -> list[str]:
    rng = random.Random(seed)
    letters, digits = "ACDEFHJKMNPRTUVWXY", "2345679"
    return ["".join(rng.choice(letters) for _ in range(3)) + "-" + "".join(
        rng.choice(digits) for _ in range(3)) for _ in LADDER_PT]


def phase_ladder(cl: Client, eng) -> dict:
    import dataclasses

    base = eng.vision
    out = {}
    cases = [("retina_2880x1800_at_cap", 2880, 1800, 2.0, None, 11),
             ("retina_2880x1800_cap_raised", 2880, 1800, 2.0, 2880 * 1800, 11),
             ("1x_1440x900", 1440, 900, 1.0, None, 12),
             ("1x_2560x1440", 2560, 1440, 1.0, None, 13)]
    try:
        for name, w, h, scale, cap, seed in cases:
            codes = _codes(seed)
            png = png_bytes(small_text_screenshot(w, h, LADDER_PT, codes, scale=scale))
            eng.vision = base if cap is None else dataclasses.replace(base, max_pixels=cap)
            r = cl.chat([oa_user(png, Q_LADDER)], max_tokens=160)
            read = {pt: code in r["text"] for pt, code in zip(LADDER_PT, codes)}
            out[name] = dict(short(r), codes=dict(zip(LADDER_PT, codes)), read=read,
                             cap=eng.vision.max_pixels,
                             smallest_all_read_from=next(
                                 (pt for i, pt in enumerate(LADDER_PT)
                                  if all(read[p] for p in LADDER_PT[i:])), None))
    finally:
        eng.vision = base
    return out


def _conversation(cl: Client, png: bytes, turns: list[str], history=None) -> list[dict]:
    msgs = list(history or [])
    out = []
    for i, q in enumerate(turns):
        msgs.append(oa_user(png if i == 0 and not history else None, q))
        r = cl.chat(msgs, max_tokens=40)
        out.append(r)
        msgs.append(assistant_turn(r))
    return out


TURNS = [Q_BOTH, "Which service in the Deployments table has the status degraded?",
         "What is that service's Region? Answer in one word."]


def phase_cache(cl: Client, eng) -> dict:
    png = png_bytes(screenshot(2560, 1440, nonce=NONCE, cell=CELL, scale=UI_SCALE))
    pngb = png_bytes(screenshot(2560, 1440, nonce=NONCE_B, cell=CELL_B, seed=1,
                                scale=UI_SCALE))
    out = {}
    warm = _conversation(cl, png, TURNS)
    eng._reuse = False
    try:
        cold = _conversation(cl, png, TURNS)
    finally:
        eng._reuse = True
    out["warm"] = [short(r) | {"cached": cached(r), "images": images_line(r)} for r in warm]
    out["cold"] = [short(r) | {"cached": cached(r), "images": images_line(r)} for r in cold]
    out["warm_equals_cold"] = [a["text"] == b["text"] for a, b in zip(warm, cold)]
    # a turn whose prompt extends the last one's slot reuses the image's KV:
    # there the tower must not run (a turn can miss the slot for a text
    # reason, e.g. the template trimming the last reply's trailing newline)
    out["warm_reused"] = [cached(r) > 0 for r in warm]
    out["warm_tower_skipped_where_reused"] = [tower_skipped(r) for r in warm if cached(r) > 0]
    a = cl.chat([oa_user(png, Q_BOTH)])
    b = cl.chat([oa_user(pngb, Q_BOTH)])
    out["same_size_a"] = short(a) | {"images": images_line(a), "own": both(a["text"])}
    out["same_size_b"] = short(b) | {"images": images_line(b),
                                     "own": both(b["text"], NONCE_B, CELL_B),
                                     "not_a": NONCE not in b["text"]
                                     and CELL not in b["text"]}
    # the cold tier: the image conversation's first two turns, then two text
    # conversations to push its live slot out (two live slots), then turn 3
    msgs = []
    for i, q in enumerate(TURNS[:2]):
        msgs.append(oa_user(png if i == 0 else None, q))
        r = cl.chat(msgs, max_tokens=40)
        msgs.append(assistant_turn(r))
    others = [cl.chat([{"role": "user", "content": f"Name {k} prime numbers above 100."}],
                      max_tokens=24) for k in (3, 5)]
    msgs.append(oa_user(None, TURNS[2]))
    r3 = cl.chat(msgs, max_tokens=40)
    out["cold_tier"] = {"evictors": [short(o) for o in others], "turn3": short(r3),
                        "images": images_line(r3), "cached": cached(r3),
                        "restored": any("from the cold tier" in ln for ln in r3["log"]),
                        "tower_skipped": tower_skipped(r3),
                        "equals_warm": r3["text"] == warm[2]["text"]}
    out["pass"] = (all(out["warm_equals_cold"]) and any(out["warm_reused"])
                   and all(out["warm_tower_skipped_where_reused"])
                   and out["same_size_a"]["own"] and out["same_size_b"]["own"]
                   and out["same_size_b"]["not_a"] and out["cold_tier"]["restored"]
                   and out["cold_tier"]["tower_skipped"])
    return out


AGENT_TOOLS = [
    {"name": "screenshot", "description": "Capture the user's screen.",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "lookup_owner", "description": "The team that owns a deployed service.",
     "input_schema": {"type": "object", "properties": {"service": {"type": "string"}},
                      "required": ["service"]}}]
AGENT_ASK = ("Take a screenshot of my screen. Then call lookup_owner with the name of the "
             "service whose status is degraded in the Deployments table. Then reply with that "
             "service's name, its Region from the table, and its owner, and nothing else.")
AGENT_OWNER = "team-halibut"
AGENT_THINKING = False  # --agent-thinking
SPEC_TASKS = ("transcribe", "describe")  # --spec-tasks


def _agent_loop(cl: Client, png: bytes, before_last=None, thinking=False) -> list[dict]:
    """A Messages-API tool loop inside ONE user turn: the model asks for a
    screenshot, reads it from a tool_result, calls lookup_owner with the
    degraded service, then answers. Each request resends the whole history
    with the model's own tool calls. before_last() runs before the third
    request. With thinking, every turn's thinking blocks go back with it
    (preserved thinking). Returns the responses (three when the model
    follows the ask)."""
    b64 = base64.b64encode(png).decode()
    msgs = [{"role": "user", "content": AGENT_ASK}]
    out = []
    for i in range(3):
        if i == 2 and before_last is not None:
            before_last()
        r = cl.messages(msgs, max_tokens=512 if thinking else 64, tools=AGENT_TOOLS,
                        thinking=thinking)
        out.append(r)
        d = r["data"] if r["status"] == 200 else {}
        uses = [b for b in d.get("content", []) if b.get("type") == "tool_use"]
        if not uses:
            break
        msgs.append({"role": "assistant", "content": d["content"]})
        results = []
        for u in uses:
            body = ([{"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                                   "data": b64}}]
                    if u["name"] == "screenshot" else f"owner: {AGENT_OWNER}")
            results.append({"type": "tool_result", "tool_use_id": u["id"], "content": body})
        msgs.append({"role": "user", "content": results})
    return out


def _blocks(r) -> list | None:
    """A Messages response's content blocks without their tool_use ids
    (fresh random ids on every response)."""
    if r["status"] != 200:
        return None
    return [{k: v for k, v in b.items() if k != "id"} for b in r["data"].get("content", [])]


def phase_agent(cl: Client, eng) -> dict:
    """The prefix cache with an image in an agent's tool loop inside one
    user turn, the shape a harness sends: the request after the
    screenshot's tool_result must reuse the prefix holding the image, and
    the tower must not run for it; warm equals cold; and after two text
    conversations evict the loop's slot, the same last request is restored
    from the cold tier with the tower skipped. Whether the re-rendered
    history extends what the model generated is the family template's
    (bench/prefix_slots_verify.py, "THINKING AND THE HISTORY RE-RENDER";
    gemma-4: the module docstring)."""
    png = png_bytes(screenshot(2560, 1440, nonce=NONCE, cell=CELL, scale=UI_SCALE))
    out = {"thinking": AGENT_THINKING}
    warm = _agent_loop(cl, png, thinking=AGENT_THINKING)
    eng._reuse = False
    try:
        cold = _agent_loop(cl, png, thinking=AGENT_THINKING)
    finally:
        eng._reuse = True
    view = lambda r: short(r) | {"cached": cached(r), "images": images_line(r),  # noqa: E731
                                 "content": r["data"].get("content") if r["status"] == 200
                                 else r["data"]}
    out["warm"] = [view(r) for r in warm]
    out["cold"] = [view(r) for r in cold]
    out["warm_equals_cold"] = [_blocks(r) for r in warm] == [_blocks(r) for r in cold]
    last = warm[-1]
    out["answer_ok"] = ("bottle" in last["text"] and "local" in last["text"]
                        and AGENT_OWNER in last["text"])
    out["after_image_reused"] = [cached(r) > 0 for r in warm[2:]]
    out["after_image_tower_skipped"] = [tower_skipped(r) for r in warm[2:]]
    # the cold tier: the loop again to its second request, then two text
    # conversations push its slot out (two live slots), then the last one
    evict = []
    again = _agent_loop(cl, png, lambda: evict.extend(
        cl.chat([{"role": "user", "content": f"Name {k} prime numbers above 100."}],
                max_tokens=24) for k in (3, 5)), thinking=AGENT_THINKING)
    out["cold_tier"] = {"evictors": [short(o) for o in evict],
                        "requests": [view(r) for r in again],
                        "restored": any("from the cold tier" in ln for r in again[2:]
                                        for ln in r["log"]),
                        "tower_skipped_after_image": [tower_skipped(r) for r in again[2:]],
                        "equals_warm": [_blocks(r) for r in again] == [_blocks(r) for r in warm]}
    out["ends_inside_an_image"] = any("ends inside an image" in ln
                                      for r in warm + again for ln in r["log"])
    out["pass"] = (len(warm) == 3 and out["answer_ok"] and out["warm_equals_cold"]
                   and all(out["after_image_reused"]) and all(out["after_image_tower_skipped"])
                   and out["cold_tier"]["restored"]
                   and all(out["cold_tier"]["tower_skipped_after_image"])
                   and not out["ends_inside_an_image"])
    return out


def _spec_arm(eng, mode: str | None):
    """The next request's proposer (bench/mtp_gpu_acceptance.set_arm's seam)."""
    if mode is None:
        os.environ.pop("DRINKME_SPEC", None)
    else:
        os.environ["DRINKME_SPEC"] = mode
    eng._mtp_depth = None
    eng._spec_plan = None


def _counters():
    from drinkme.serving import metrics

    def val(c):
        snap = getattr(c, "snapshot", None)
        v = snap() if snap else getattr(c, "_values", {})
        return sum(v.values()) if isinstance(v, dict) else float(v)

    return val(metrics.SPEC_PROPOSED), val(metrics.SPEC_ACCEPTED)


class _Rows:
    """An arm's picks and logits rows by position (len(gen_ids) at each
    engines.sample_next call, tests/spec_agree.py's seam; a speculative arm
    samples each verified row there too), the rows kept on the host: what
    the sampler reads, after the wrapper's own transform (gemma-4's
    softcap; lm_head's raw output is not the logits there)."""

    def __init__(self, eng):
        self.rows, self.picks, self.eng = {}, {}, eng

    def __enter__(self):
        from drinkme.serving import engines

        self.real = real = engines.sample_next

        def rec(logits, *a, **k):
            at = len(k.get("gen_ids") or ())
            self.rows[at] = logits.detach().reshape(-1).to("cpu")
            t = self.picks[at] = int(real(logits, *a, **k))
            return t

        engines.sample_next = rec
        return self

    def __exit__(self, *exc):
        from drinkme.serving import engines

        engines.sample_next = self.real


def _margin(off: _Rows, arm: _Rows, eng) -> dict:
    """Where an arm left the serial transcript, by the ids each picked (not
    by re-encoding the reply's text, which drops special tokens: Muse-
    Glimmer's channel markers): the serial decode's own logit margin there
    between its token and the arm's (a near-tie is a margin at bf16
    noise)."""
    off_ids = [off.picks[i] for i in sorted(off.picks)]
    arm_ids = [arm.picks[i] for i in sorted(arm.picks)]
    k = next((i for i, (x, y) in enumerate(zip(off_ids, arm_ids)) if x != y), None)
    if k is None:
        return {"token_index": None}
    row = off.rows[k].float()
    top = row.topk(2)
    return {"token_index": k, "off_token": eng.tok.decode([off_ids[k]]),
            "arm_token": eng.tok.decode([arm_ids[k]]),
            "margin": float(row[off_ids[k]] - row[arm_ids[k]]),
            "top2_gap": float(top.values[0] - top.values[1])}


def phase_spec(cl: Client, eng) -> dict:
    from vision_screens import _SHELL

    png = png_bytes(screenshot(2560, 1440, nonce=NONCE, cell=CELL, scale=UI_SCALE))
    as_text = "Here is my terminal:\n```\n" + "\n".join(_SHELL) + "\n```\n"
    tasks = {k: v for k, v in {"transcribe": (Q_TRANSCRIBE, 120),
                               "describe": (Q_DESCRIBE, 120)}.items() if k in SPEC_TASKS}
    out = {}
    eng._reuse = False  # every arm prefills the same prompt cold
    try:
        for task, (q, n) in tasks.items():
            for kind in ("image", "text"):
                msgs = ([oa_user(png, q)] if kind == "image"
                        else [{"role": "user", "content": as_text + q}])
                arms, recs = {}, {}
                # without an MTP head `auto` is n-gram (gemma-4): not run twice
                for mode in ("off", "ngram") + ((None,) if eng.mtp_head is not None else ()):
                    _spec_arm(eng, mode)
                    p0, a0 = _counters()
                    with _Rows(eng) as rec:
                        r = cl.chat(msgs, max_tokens=n)
                    recs[mode or "auto"] = rec
                    p1, a1 = _counters()
                    arms[mode or "auto"] = short(r) | {
                        "proposed": p1 - p0, "accepted": a1 - a0,
                        "acceptance": (a1 - a0) / (p1 - p0) if p1 > p0 else None,
                        "spec_line": next((ln for ln in r["log"]
                                           if ln.startswith("[drinkme.spec]")), None)}
                texts = {k: v["text"] for k, v in arms.items()}
                ref = texts["off"]
                diff = {}
                for k, t in texts.items():
                    if t != ref:
                        i = next((j for j, (x, y) in enumerate(zip(t, ref)) if x != y),
                                 min(len(t), len(ref)))
                        diff[k] = {"at_char": i, "off": ref[max(0, i - 20):i + 20],
                                   "arm": t[max(0, i - 20):i + 20]}
                for k in diff:  # the serial decode's margin where an arm left it
                    diff[k]["logits"] = _margin(recs["off"], recs[k], eng)
                out[f"{task}_{kind}"] = {"arms": arms,
                                         "identical": all(t == ref for t in texts.values()),
                                         "diff": diff}
    finally:
        eng._reuse = True
        _spec_arm(eng, None)
    # with an image in context every arm is the serial transcript, or leaves
    # it at a near-tie: a serial-decode margin within one bf16 step at the
    # logits' magnitude (0.25 under 64). The text-only rows are the
    # acceptance baseline; their differences are reported the same way.
    out["pass"] = all(v["identical"] or all(
        abs((d.get("logits") or {}).get("margin", 1e9)) <= 0.25 for d in v["diff"].values())
        for k, v in out.items() if isinstance(v, dict) and k.endswith("_image"))
    return out


def phase_retina(cl: Client, eng) -> dict:
    """The Retina capture (2880x1800, 2x UI) with four nonce pairs, at the
    default cap and with the cap raised to the capture's own size: how often
    each reads both nonces exactly (the pixel-cap decision)."""
    import dataclasses

    rng = random.Random(7)
    birds = ["HERON", "EGRET", "IBIS", "CRANE", "STORK", "TERN", "GANNET", "PETREL"]
    base = eng.vision
    out = {"cases": []}
    try:
        for i in range(4):
            a = f"{birds[2 * i]}-{rng.randint(1000, 9999)}"
            b = f"{birds[2 * i + 1]}-{rng.randint(1000, 9999)}"
            png = png_bytes(screenshot(2880, 1800, nonce=a, cell=b, scale=2.0, seed=i))
            row = {"nonce": a, "cell": b}
            for name, cap in (("at_cap", None), ("cap_raised", 2880 * 1800)):
                eng.vision = base if cap is None else dataclasses.replace(base, max_pixels=cap)
                r = cl.chat([oa_user(png, Q_BOTH)])
                row[name] = {"text": r["text"], "both": both(r["text"], a, b),
                             "wall_s": r["wall_s"], "images": images_line(r)}
            out["cases"].append(row)
    finally:
        eng.vision = base
    out["at_cap"] = sum(c["at_cap"]["both"] for c in out["cases"])
    out["cap_raised"] = sum(c["cap_raised"]["both"] for c in out["cases"])
    out["pass"] = out["at_cap"] == len(out["cases"])
    return out


def _tower_digest(eng) -> tuple[str, str]:
    import torch

    h = hashlib.sha256()
    devs = set()
    for path in eng._tower.paths:  # every subtree (gemma-4: the ViT and embed_vision)
        vit = eng.model.get_submodule(path)
        for name, t in sorted(list(vit.named_parameters()) + list(vit.named_buffers())):
            devs.add(str(t.device))
            h.update(f"{path}.{name}".encode())
            h.update(t.detach().to("cpu").contiguous().view(torch.uint8).numpy().tobytes())
        _digest_packed(h, devs, vit, path)
    return h.hexdigest(), ",".join(sorted(devs))


def _digest_packed(h, devs, vit, path) -> None:
    import torch

    for name, m in sorted(vit.named_modules()):  # packed Linears: their `p` dicts
        p = getattr(m, "p", None)
        if isinstance(p, dict):
            for k in sorted(p):
                v = p[k]
                if isinstance(v, torch.Tensor):  # the device copies; numpy stays on the host
                    devs.add(str(v.device))
                    h.update(f"{path}.{name}.{k}".encode())
                    h.update(v.detach().to("cpu").contiguous().view(torch.uint8).numpy().tobytes())


def phase_sleep(cl: Client, eng) -> dict:
    import torch

    png = png_bytes(screenshot(2560, 1440, nonce=NONCE, cell=CELL, scale=UI_SCALE))
    out = {}
    msgs = []
    for i, q in enumerate(TURNS[:2]):
        msgs.append(oa_user(png if i == 0 else None, q))
        r = cl.chat(msgs, max_tokens=40)
        msgs.append(assistant_turn(r))
    d0, dev0 = _tower_digest(eng)
    torch.cuda.synchronize()
    m0 = torch.cuda.memory_allocated()
    s = cl.call("POST", "/sleep?level=1")
    h = cl.call("GET", "/health")
    d1, dev1 = _tower_digest(eng)
    m1 = torch.cuda.memory_allocated()
    w = cl.call("POST", "/wake_up")
    d2, dev2 = _tower_digest(eng)
    m2 = torch.cuda.memory_allocated()
    msgs.append(oa_user(None, TURNS[2]))
    r3 = cl.chat(msgs, max_tokens=40)
    out.update(sleep=short(s), health_asleep=h["data"], wake=short(w),
               tower_devices=[dev0, dev1, dev2], tower_digest_equal=[d0 == d1, d0 == d2],
               allocated_gib=[round(x / 2**30, 3) for x in (m0, m1, m2)],
               turn3=short(r3), images=images_line(r3), tower_skipped=tower_skipped(r3),
               health_awake=cl.call("GET", "/health")["data"])
    out["pass"] = (s["status"] == 200 and w["status"] == 200 and "cuda" not in dev1
                   and d0 == d1 == d2 and dev2 == dev0 and tower_skipped(r3))
    return out


def phase_sleep2(cl: Client, eng) -> dict:
    """POST /sleep?level=2 (the host copies freed too) and /wake_up, which
    rebuilds the engine through its loader (engines.HFEngine.wake): the
    rebuilt tower is the same bytes, its boot attention self-test runs again
    (its line is in the wake's log), and a new image request after the wake
    reads the nonce."""
    import torch

    png = png_bytes(screenshot(2560, 1440, nonce=NONCE, cell=CELL, scale=UI_SCALE))
    before = cl.chat([oa_user(png, Q_BOTH)])
    d0, dev0 = _tower_digest(eng)
    s = cl.call("POST", "/sleep?level=2")
    torch.cuda.synchronize()
    m1 = torch.cuda.memory_allocated()
    w = cl.call("POST", "/wake_up")
    d2, dev2 = _tower_digest(eng)
    after = cl.chat([oa_user(png, Q_BOTH)])
    selftest = [ln for ln in w["log"] if "attention self-test" in ln]
    out = {"before": short(before), "sleep": short(s),
           "allocated_gib_asleep": round(m1 / 2**30, 3), "wake": short(w),
           "selftest_lines": selftest, "tower_digest_equal": d0 == d2,
           "tower_devices": [dev0, dev2], "after": short(after) | {"images": images_line(after)}}
    out["pass"] = (s["status"] == 200 and w["status"] == 200 and d0 == d2
                   and any(ln.endswith("AGREES") for ln in selftest)
                   and both(before["text"]) and both(after["text"]))
    return out


def main() -> None:
    global UI_SCALE, AGENT_THINKING, SYSTEM, REASONING_TOKENS, SPEC_TASKS

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=sorted(MODELS), default="27b")
    ap.add_argument("--pack", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ctx", type=int, default=16384)
    ap.add_argument("--phases", default="smoke,ladder,cache,spec,sleep")
    ap.add_argument("--ui-scale", type=float, default=1.0,
                    help="the 2560x1440 screenshots' UI scale (2.4 for gemma-4: UI_SCALE)")
    ap.add_argument("--agent-thinking", action="store_true",
                    help="the agent phase with thinking on, its blocks passed back")
    ap.add_argument("--spec-tasks", default=",".join(SPEC_TASKS),
                    help="the spec phase's tasks (one process each on a slow decode)")
    args = ap.parse_args()
    UI_SCALE, AGENT_THINKING = args.ui_scale, args.agent_thinking
    SPEC_TASKS = tuple(args.spec_tasks.split(","))
    if args.model == "glimmer":
        SYSTEM, REASONING_TOKENS = "Reasoning strength: low.", 256
    os.makedirs(args.out, exist_ok=True)
    slots = os.path.join(args.out, "slots")
    os.environ.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")
    os.environ.update(HF_HUB_OFFLINE="1", DRINKME_NO_AUTO_DEPS="1", DRINKME_SLOT_DIR=slots,
                      DRINKME_PREFIX_SLOTS="2")
    tee = Tee(sys.stderr)
    sys.stderr, sys.stdout = tee, Tee(sys.stdout, share=tee)

    from drinkme.serve import build_engine
    from drinkme.serving.http import start_server

    t0 = time.time()
    eng = build_engine(MODELS[args.model], None, args.pack, stock=False, ctx=args.ctx)
    assert str(eng.device) != "cpu", "REFUSING to measure on the CPU"
    assert eng.vision is not None, f"no image input: {eng.vision_reason}"
    srv = start_server(eng, "127.0.0.1", 0)
    cl = Client(srv.server_address[1], tee, eng.model_id)
    report = {"model": args.model, "pack": args.pack, "load_s": round(time.time() - t0, 1),
              "port": srv.server_address[1], "ui_scale": UI_SCALE}
    fns = {"smoke": lambda: phase_smoke(cl, eng), "ladder": lambda: phase_ladder(cl, eng),
           "cache": lambda: phase_cache(cl, eng), "spec": lambda: phase_spec(cl, eng),
           "sleep": lambda: phase_sleep(cl, eng), "retina": lambda: phase_retina(cl, eng),
           "sleep2": lambda: phase_sleep2(cl, eng), "agent": lambda: phase_agent(cl, eng)}
    for ph in args.phases.split(","):
        t = time.time()
        try:
            report[ph] = fns[ph]()
        except Exception as e:  # noqa: BLE001 — one phase's failure is reported, the rest run
            import traceback

            report[ph] = {"pass": False, "error": f"{type(e).__name__}: {e}",
                          "traceback": traceback.format_exc()}
        report[ph]["phase_s"] = round(time.time() - t, 1)
        with open(os.path.join(args.out, "report.json"), "w") as f:
            json.dump(report, f, indent=1, default=str)
        print(f"VERDICT {ph} {'PASS' if report[ph].get('pass') else 'FAIL'} "
              f"({report[ph]['phase_s']} s)", flush=True)
    srv.shutdown()


if __name__ == "__main__":
    main()
