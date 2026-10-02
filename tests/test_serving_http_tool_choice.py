"""Forced tool choice is one contract across the dialects.
/v1/chat/completions accepted `tool_choice:
"required"` and a named function and IGNORED them — reproduced: a valid
`ping` tool, tool_choice "required", 200, plain answer text,
finish_reason "stop", no tool call — while /v1/responses and /v1/messages
refused the same request. Now the chat route refuses forced tool use with
a named 400 (code `unsupported_tool_choice`), as the other two do, until
it can be enforced; unknown tool_choice values are refused rather than
ignored; "auto" and "none" are unchanged.
"""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(__file__))
from test_serving_http import assert_error_shape, fake, msgs, post  # noqa: E402,F401
from test_serving_messages import post as post_messages  # noqa: E402
from test_serving_responses import post as post_responses  # noqa: E402

MODEL = "drinkme-fake"
PING = {"type": "function", "function": {"name": "ping", "description": "ping",
                                         "parameters": {"type": "object", "properties": {}}}}


@pytest.mark.parametrize("tc, what", [
    ("required", "required"),
    ({"type": "function", "function": {"name": "ping"}}, "function 'ping'"),
])
@pytest.mark.parametrize("stream", [False, True])
def test_chat_completions_refuses_forced_tool_use_by_name(fake, tc, what, stream):
    eng, port = fake()
    calls = []
    orig = eng.generate
    eng.generate = lambda *a, **k: calls.append(1) or orig(*a, **k)
    r, data = post(port, {"messages": msgs(), "tools": [PING], "tool_choice": tc, "stream": stream})
    assert r.status == 400, (tc, r.status, data)
    err = assert_error_shape(data)
    assert err["code"] == "unsupported_tool_choice"
    assert f"tool_choice: {what} (forced tool use) is not supported" in err["message"]
    assert calls == []


@pytest.mark.parametrize("tc", ["bogus", {"type": "bogus"}, {"type": "auto", "extra": 1}])
def test_chat_completions_refuses_an_unknown_tool_choice_instead_of_ignoring_it(fake, tc):
    _, port = fake()
    r, data = post(port, {"messages": msgs(), "tools": [PING], "tool_choice": tc})
    assert r.status == 400, (tc, r.status, data)
    assert "tool_choice: 'bogus' is not supported" in assert_error_shape(data)["message"] \
        or "tool_choice: 'auto' is not supported" in assert_error_shape(data)["message"]


def test_auto_and_none_are_unchanged(fake):
    eng, port = fake()
    seen = []
    orig = eng.generate
    eng.generate = lambda req: seen.append(req.tools) or orig(req)
    r, _ = post(port, {"messages": msgs(), "tools": [PING], "tool_choice": "auto"})
    assert r.status == 200
    r, _ = post(port, {"messages": msgs(), "tools": [PING], "tool_choice": "none"})
    assert r.status == 200
    r, _ = post(port, {"messages": msgs(), "tools": [PING]})
    assert r.status == 200
    assert [bool(t) for t in seen] == [True, False, True]


def test_the_three_dialects_agree(fake):
    """The same forced request on each wire: a 400 naming tool_choice."""
    _, port = fake()
    r, data = post(port, {"messages": msgs(), "tools": [PING], "tool_choice": "required"})
    assert r.status == 400 and "forced tool use" in json.loads(data)["error"]["message"]
    r, data = post_responses(port, {"model": MODEL, "input": "hi", "tool_choice": "required",
                                    "tools": [{"type": "function", "name": "ping", "description": "ping",
                                               "parameters": {"type": "object", "properties": {}}}]})
    assert r.status == 400 and "forced tool use" in json.loads(data)["error"]["message"]
    r, data = post_messages(port, {"model": MODEL, "max_tokens": 8, "tool_choice": {"type": "any"},
                                   "messages": [{"role": "user", "content": "hi"}],
                                   "tools": [{"name": "ping", "description": "ping",
                                              "input_schema": {"type": "object", "properties": {}}}]})
    assert r.status == 400 and "forced tool use" in json.loads(data)["error"]["message"]
