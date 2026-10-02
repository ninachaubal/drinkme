"""A supplied token limit reaches the validator as supplied.
/v1/chat/completions built its SampleParams from
`req.get('max_tokens') or None`, so `0`, `false`, `""`, `[]`, `{}` became
"unspecified" and the request generated the full default budget with a
200 — the shared validator's positive-integer / no-boolean rule was
present and never reached. Now absent and null mean the default, and any
value supplied is validated under its own field name, streaming and not,
on all three dialects' equivalents: `max_tokens` and (newly read)
`max_completion_tokens` on Chat Completions, `max_output_tokens` on
Responses, `max_tokens` on Messages (required there) and its count_tokens
route. Responses and Messages already passed the value through; they are
pinned here so the three cannot drift.
"""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(__file__))
from test_serving_http import assert_error_shape, fake, msgs, post, sse_events  # noqa: E402,F401
from test_serving_messages import post as post_messages  # noqa: E402
from test_serving_responses import post as post_responses  # noqa: E402

FALSEY = [0, False, "", [], {}]
MODEL = "drinkme-fake"


def _needle(v, field):
    if v is False:
        return f"{field}: must be an integer, not a boolean"
    if v == 0:
        return f"{field}: must be a positive integer"
    return f"{field}: must be an integer"


@pytest.mark.parametrize("value", FALSEY)
@pytest.mark.parametrize("field", ["max_tokens", "max_completion_tokens"])
@pytest.mark.parametrize("stream", [False, True])
def test_chat_completions_falsey_limits_are_the_validators_named_400(fake, value, field, stream):
    eng, port = fake()
    calls = []
    orig = eng.generate
    eng.generate = lambda *a, **k: calls.append(1) or orig(*a, **k)
    r, data = post(port, {"messages": msgs(), field: value, "stream": stream})
    assert r.status == 400, (field, value, r.status, data)
    err = assert_error_shape(data)
    assert err["type"] == "invalid_request_error"
    assert _needle(value, field) in err["message"], (field, value, err["message"])
    assert calls == []


@pytest.mark.parametrize("stream", [False, True])
def test_chat_completions_absent_and_null_mean_the_default(fake, stream):
    _, port = fake()
    for body in ({"messages": msgs()}, {"messages": msgs(), "max_tokens": None},
                 {"messages": msgs(), "max_completion_tokens": None}):
        r, data = post(port, dict(body, stream=stream))
        assert r.status == 200, (body, data)


def test_chat_completions_reads_max_completion_tokens_and_refuses_both(fake):
    _, port = fake()
    r, data = post(port, {"messages": msgs(), "max_completion_tokens": 1})
    assert r.status == 200
    assert json.loads(data)["choices"][0]["finish_reason"] == "length"
    assert json.loads(data)["usage"]["completion_tokens"] == 1
    r, data = post(port, {"messages": msgs(), "max_tokens": 2, "max_completion_tokens": 2})
    assert r.status == 400 and "cannot both be set" in assert_error_shape(data)["message"]
    r, data = post(port, {"messages": msgs(), "max_tokens": None, "max_completion_tokens": 1})
    assert r.status == 200


@pytest.mark.parametrize("value", FALSEY)
@pytest.mark.parametrize("stream", [False, True])
def test_responses_max_output_tokens(fake, value, stream):
    _, port = fake()
    r, data = post_responses(port, {"model": MODEL, "input": "hi", "max_output_tokens": value,
                                    "stream": stream})
    assert r.status == 400, (value, r.status, data)
    assert _needle(value, "max_output_tokens") in json.loads(data)["error"]["message"]


@pytest.mark.parametrize("value", FALSEY)
@pytest.mark.parametrize("stream", [False, True])
def test_messages_max_tokens(fake, value, stream):
    _, port = fake()
    r, data = post_messages(port, {"model": MODEL, "max_tokens": value, "stream": stream,
                                   "messages": [{"role": "user", "content": "hi"}]})
    assert r.status == 400, (value, r.status, data)
    assert "max_tokens: must be a positive integer" in json.loads(data)["error"]["message"]
    # count_tokens: max_tokens optional, but a supplied falsey one is still refused
    r, data = post_messages(port, {"model": MODEL, "max_tokens": value,
                                   "messages": [{"role": "user", "content": "hi"}]},
                            path="/v1/messages/count_tokens")
    assert r.status == 400, (value, r.status, data)
    assert _needle(value, "max_tokens") in json.loads(data)["error"]["message"]
