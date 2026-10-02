"""chat_template_kwargs.enable_thinking is the vendor-neutral thinking
switch, on all three dialects it actually reaches: each dialect's switch
reaches the template, /v1/models names it, and the body pi sends for
`--thinking off` turns thinking off.
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from test_serving_http import FakeTok, Templating, fake, msgs, post, templating  # noqa: E402,F401
from test_serving_messages import post as post_messages  # noqa: E402
from test_serving_messages import req as req_messages  # noqa: E402
from test_serving_responses import post as post_responses  # noqa: E402
from test_serving_responses import req as req_responses  # noqa: E402

# -------------------------------------------------- the switch per dialect --


def test_each_dialects_thinking_switch_reaches_the_template(fake):
    """Chat Completions and Responses take chat_template_kwargs.enable_thinking;
    Messages takes Anthropic's own `thinking`."""
    eng, port = fake(engine=templating(reply="hi"))
    post(port, {"messages": msgs(), "chat_template_kwargs": {"enable_thinking": False}})
    assert eng.tok.calls[-1]["enable_thinking"] is False

    eng2, port2 = fake(engine=templating(reply="hi"))
    post_responses(port2, req_responses(chat_template_kwargs={"enable_thinking": False}))
    assert eng2.tok.calls[-1]["enable_thinking"] is False

    eng3, port3 = fake(engine=templating(reply="hi"))
    post_messages(port3, req_messages(thinking={"type": "disabled"}))
    assert eng3.tok.calls[-1]["enable_thinking"] is False


def test_models_capability_names_the_switch(fake):
    _, port = fake(thinking="closed")
    from test_serving_http import get  # local: avoid a wildcard re-export

    (m,) = json.loads(get(port, "/v1/models")[1])["data"]
    switch = m["drinkme"]["capabilities"]["thinkingSwitch"]
    assert switch == "chat_template_kwargs.enable_thinking"


# ------------------------------------------------------ the pi models.json --


def test_the_pi_wire_body_actually_turns_thinking_off(fake):
    """The literal body pi's "qwen-chat-template" formatter sends for
    `--thinking off` (per pi 0.84.4's openai-completions-ERMU2SS7.js:
    `chat_template_kwargs={enable_thinking:false,preserve_thinking:true}`)
    — fed to drinkme for real, not just asserted
    ABOUT. preserve_thinking must ALSO reach the template (it is not a
    reserved kwarg http.py refuses)."""
    eng, port = fake(engine=templating(reply="hi"))
    r, _ = post(port, {"messages": msgs(),
                       "chat_template_kwargs": {"enable_thinking": False,
                                                "preserve_thinking": True}})
    assert r.status == 200
    assert eng.tok.calls[-1]["enable_thinking"] is False
    assert eng.tok.calls[-1]["preserve_thinking"] is True
