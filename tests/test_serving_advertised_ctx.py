"""Advertised context: /v1/messages' usage.input_tokens and
count_tokens SCALED by real/advertised, off by default. FakeEngine + real
sockets, same posture as test_serving_http.py/test_serving_messages.py.

The known prompt throughout: system "be brief" (2 words) + user "hello
world" (2 words) = 4 real tokens under FakeEngine's one-word-one-token
count (test_serving_messages.py's test_count_tokens uses the same body).
"""

import json

from test_serving_http import fake  # noqa: F401 — shared fixture
from test_serving_messages import events, post  # Anthropic-dialect helpers

BODY = {"model": "drinkme-fake", "system": "be brief",
        "messages": [{"role": "user", "content": "hello world"}]}


def test_default_is_identity_even_with_a_real_ctx_published(fake):
    # no --advertised-ctx: real ctx published, but nothing scales
    _, port = fake(context_window=32000)
    r, data = post(port, BODY, path="/v1/messages/count_tokens")
    assert r.status == 200 and json.loads(data) == {"input_tokens": 4}
    obj = json.loads(post(port, dict(BODY, max_tokens=8))[1])
    assert obj["usage"]["input_tokens"] == 4


def test_advertised_below_real_scales_usage_and_count_tokens(fake):
    # real 32000, advertised 8000 -> scale 4x: 4 real tokens report as 16
    _, port = fake(context_window=32000, advertised_ctx=8000)
    r, data = post(port, BODY, path="/v1/messages/count_tokens")
    assert r.status == 200 and json.loads(data) == {"input_tokens": 16}
    obj = json.loads(post(port, dict(BODY, max_tokens=8))[1])
    u = obj["usage"]
    assert u["input_tokens"] == 16  # the lie
    assert u["output_tokens"] == 3  # "echo:", "hello", "world" — real, untouched
    assert u["cache_read_input_tokens"] == 0  # real, untouched (nothing cached here)


def test_streaming_message_start_and_message_delta_both_carry_the_lie(fake):
    _, port = fake(context_window=32000, advertised_ctx=8000)
    r, body = post(port, dict(BODY, max_tokens=8, stream=True))
    assert r.status == 200
    evs = dict(events(body))
    assert evs["message_start"]["message"]["usage"]["input_tokens"] == 16
    assert evs["message_delta"]["usage"]["input_tokens"] == 16


def test_advertised_at_or_above_real_ctx_is_identity(fake):
    # "BELOW the real ctx" (http.py's own words) — an ask that is not
    # actually below real must not shrink reported usage past the truth
    for adv in (32000, 64000):
        _, port = fake(context_window=32000, advertised_ctx=adv)
        r, data = post(port, BODY, path="/v1/messages/count_tokens")
        assert json.loads(data) == {"input_tokens": 4}, adv


def test_engine_with_no_published_ctx_falls_back_to_identity(fake):
    # FakeEngine's default ctx=None (no contextWindow) — nothing to scale by
    _, port = fake(advertised_ctx=8000)
    r, data = post(port, BODY, path="/v1/messages/count_tokens")
    assert json.loads(data) == {"input_tokens": 4}


def test_openai_dialect_stays_unscaled(fake):
    import http.client

    _, port = fake(context_window=32000, advertised_ctx=8000)
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    c.request("POST", "/v1/chat/completions",
              json.dumps({"messages": [{"role": "user", "content": "hello world"}]}),
              {"Content-Type": "application/json"})
    r = c.getresponse()
    obj = json.loads(r.read())
    c.close()
    assert obj["usage"]["prompt_tokens"] == 2  # real, never the advertised-ctx lie
