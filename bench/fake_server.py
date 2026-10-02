"""A standalone FakeEngine server for driving real clients on a CPU box — no
model, no pack, no torch: the whole HTTP layer (all three dialects) over
canned generation that answers the smoke scripts' prompts the way a
well-behaved model would.

    PYTHONPATH=src python bench/fake_server.py --port 3216

DRINKME_FAKE_ENGINE=1 already serves a FakeEngine, but that one only echoes:
it can never emit a tool call, so a client smoke against it cannot exercise
the function-call path. This engine is FakeEngine with a script chooser —
it reads the conversation it is handed and picks the reply the smoke expects:

  tools offered + a user turn mentioning weather   a <tool_call> for
                                                   get_weather(city=<the city
                                                   named in the prompt>), after
                                                   a short visible sentence
  a tool turn last                                 a sentence that repeats the
                                                   tool result's numbers
  "17 * 23"                                        a think block, then 391
  anything else                                    "echo: <last user text>"

So bench/responses_sdk_smoke.mjs and bench/serve_dialect_smoke.py both
report ALL HOLD against this server — which says the WIRE holds (shapes,
ids, stream accumulation, round trips), and says nothing about a model. The
GPU run of the same scripts against a real bottle is the acceptance.

Serves the model id given by --model (default drinkme-fake); the tool format
announced is `json` (the default tested row) so tools are offered, not
refused. Ctrl-C / SIGTERM stops it.
"""

from __future__ import annotations

import argparse
import json
import re
import signal
import sys
import threading

from drinkme.serving.engine import FakeEngine
from drinkme.serving.http import start_server


class ScriptedFakeEngine(FakeEngine):
    """FakeEngine whose reply depends on the conversation (module docstring)."""

    def generate(self, req):
        messages, tools = req.messages, req.tools
        last = messages[-1] if messages else {}
        user = ""
        for m in messages:
            if m.get("role") == "user":
                c = m.get("content", "")
                user = c if isinstance(c, str) else json.dumps(c)
        self.reply, self.tool_call_script, self.opens_think = None, None, False
        if last.get("role") == "tool":
            nums = re.findall(r"\d+", str(last.get("content", "")))
            words = re.findall(r"[a-z]+", str(last.get("content", "")).lower())
            detail = ", ".join(nums[:2]) or "no numbers"
            sky = next((w for w in words if w in ("rain", "sun", "sunny", "cloudy", "clear")), "")
            self.reply = (f"The tool says: {detail}{(' and ' + sky) if sky else ''}. "
                          "That is the current weather.")
        elif tools and re.search(r"weather", user, re.I):
            m = re.search(r"in ([A-Z][a-zA-Z]+)", user)
            city = m.group(1) if m else "Paris"
            self.tool_call_script = (
                "Let me check that for you.\n<tool_call>\n"
                + json.dumps({"name": "get_weather", "arguments": {"city": city}})
                + "\n</tool_call>")
        elif re.search(r"17\s*\*\s*23", user):
            self.reply = ("17 times 23: 17 * 20 = 340, 17 * 3 = 51, 340 + 51 = 391."
                          "</think>\n\n17 * 23 = 391.")
            self.opens_think = True
        return super().generate(req)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=3216)
    ap.add_argument("--model", default="drinkme-fake", help="the model id to serve as")
    ap.add_argument("--auth", default=None, help="require this bearer on /v1/*")
    args = ap.parse_args()
    eng = ScriptedFakeEngine(model_id=args.model)
    try:
        srv = start_server(eng, args.host, args.port, auth_token=args.auth)
    except OSError as e:
        print(f"[fake_server] cannot bind {args.host}:{args.port}: {e}", file=sys.stderr)
        return 1
    port = srv.server_address[1]
    print(f"[fake_server] FakeEngine {args.model!r} on http://{args.host}:{port} "
          "— /v1/chat/completions, /v1/responses, /v1/messages; canned generation, "
          "no model loaded", flush=True)
    done = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: done.set())
    done.wait()
    srv.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
