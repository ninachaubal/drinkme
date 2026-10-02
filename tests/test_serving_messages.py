"""/v1/messages — the Anthropic Messages API over the same core, fake engine, CPU.

Every request goes over a real socket to a port-0 server, like
test_serving_http.py (whose fixture, fake tokenizer and scripts this file
reuses): what passes here is what curl, the Anthropic SDKs and the claude
CLI see. The two load-bearing tests are the cross-surface parity test (R1:
the three dialects — chat completions, Messages, Responses — feed the
engine identical inputs and get the identical delta transcript back) and
the input_json_delta concatenation identity (R4).
"""

import http.client
import http.server
import json
import socket
import threading
import time

from drinkme.serving import vision
from drinkme.serving.engine import Delta, FakeEngine, Finished
from test_serving_http import (  # noqa: F401 — `fake` is the shared fixture
    SCRIPT, THINK, THINK_TOOL, WEATHER_TOOL, VisionEngine, FakeTok, Templating, data_url, fake,
    get, relay, templating, tiny_png, vision_engine)

MODEL = "drinkme-fake"

# the Anthropic spelling of test_serving_http's WEATHER_TOOL
TOOL = {"name": "get_weather", "description": "Current weather for a city.",
        "input_schema": {"type": "object",
                         "properties": {"city": {"type": "string"}},
                         "required": ["city"]}}


def post(port, body, path="/v1/messages", headers=None):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    payload = body if isinstance(body, (bytes, str)) else json.dumps(body)
    h = {"Content-Type": "application/json", "anthropic-version": "2023-06-01"}
    h.update(headers or {})
    c.request("POST", path, payload, h)
    r = c.getresponse()
    data = r.read()
    c.close()
    return r, data


def req(text="hello world", **kw):
    body = {"model": MODEL, "max_tokens": 64,
            "messages": [{"role": "user", "content": text}]}
    body.update(kw)
    return body


def events(body: bytes) -> list[tuple[str, dict]]:
    """Named SSE frames -> [(event, payload)] in wire order."""
    out = []
    for frame in body.decode().split("\n\n"):
        if not frame.strip():
            continue
        name = data = None
        for line in frame.split("\n"):
            if line.startswith("event: "):
                name = line[len("event: "):]
            elif line.startswith("data: "):
                data = json.loads(line[len("data: "):])
        out.append((name, data))
    return out


def assemble(evs) -> dict:
    """Reassemble a streamed message the way the SDKs do: blocks by index,
    deltas appended, tool input parsed from the partial_json buffer at
    content_block_stop, message_delta folded into the message."""
    msg = None
    blocks: dict[int, dict] = {}
    buf: dict[int, str] = {}
    for name, ev in evs:
        assert name == ev["type"]  # the event line names the payload type
        if name == "message_start":
            msg = ev["message"]
        elif name == "content_block_start":
            blocks[ev["index"]] = dict(ev["content_block"])
            buf[ev["index"]] = ""
        elif name == "content_block_delta":
            b, d = blocks[ev["index"]], ev["delta"]
            if d["type"] == "text_delta":
                b["text"] += d["text"]
            elif d["type"] == "thinking_delta":
                b["thinking"] += d["thinking"]
            elif d["type"] == "signature_delta":
                b["signature"] = d["signature"]
            elif d["type"] == "input_json_delta":
                buf[ev["index"]] += d["partial_json"]
            else:
                raise AssertionError(d)
        elif name == "content_block_stop":
            b = blocks[ev["index"]]
            if b["type"] == "tool_use":
                b["input"] = json.loads(buf[ev["index"]] or "{}")
        elif name == "message_delta":
            msg["stop_reason"] = ev["delta"]["stop_reason"]
            msg["stop_sequence"] = ev["delta"]["stop_sequence"]
            msg["usage"].update(ev["usage"])
    msg["content"] = [blocks[i] for i in sorted(blocks)]
    return msg


def envelope(data) -> dict:
    """Anthropic's error envelope, exactly those keys; returns the inner error."""
    obj = json.loads(data)
    assert set(obj) == {"type", "error"} and obj["type"] == "error"
    assert set(obj["error"]) == {"type", "message"}
    return obj["error"]


class Recording(FakeEngine):
    """What the LAST generate() received (the internal history)."""

    def generate(self, req):
        self.seen = req.messages
        return super().generate(req)


# ------------------------------------------------------------- non-stream --


def test_nonstream_text(fake):
    _, port = fake()
    r, body = post(port, req())
    obj = json.loads(body)
    assert r.status == 200
    assert obj["id"].startswith("msg_") and obj["type"] == "message"
    assert obj["role"] == "assistant" and obj["model"] == MODEL
    assert obj["content"] == [{"type": "text", "text": "echo: hello world"}]
    assert obj["stop_reason"] == "end_turn" and obj["stop_sequence"] is None
    assert obj["usage"]["input_tokens"] == 2  # "hello", "world"
    assert obj["usage"]["output_tokens"] == 3  # "echo:", "hello", "world"


def test_max_tokens_stops_with_max_tokens(fake):
    _, port = fake(reply="one two three four five")
    obj = json.loads(post(port, req(max_tokens=3))[1])
    assert obj["content"] == [{"type": "text", "text": "one two three "}]
    assert obj["stop_reason"] == "max_tokens" and obj["stop_sequence"] is None
    assert obj["usage"]["output_tokens"] == 3


def test_stop_sequence_names_the_string_that_fired(fake):
    _, port = fake(reply="alpha beta STOP gamma")
    obj = json.loads(post(port, req(stop_sequences=["END", "STOP"]))[1])
    assert obj["content"] == [{"type": "text", "text": "alpha beta "}]
    assert obj["stop_reason"] == "stop_sequence" and obj["stop_sequence"] == "STOP"
    # streamed: message_delta says the same
    m = assemble(events(post(port, req(stop_sequences=["END", "STOP"], stream=True))[1]))
    assert m["stop_reason"] == "stop_sequence" and m["stop_sequence"] == "STOP"
    assert m["content"] == [{"type": "text", "text": "alpha beta "}]


def test_system_string_and_text_blocks_become_the_system_turn(fake):
    eng, port = fake(engine=Recording())
    assert post(port, req(system="be brief"))[0].status == 200
    assert eng.seen[0] == {"role": "system", "content": "be brief"}
    assert eng.seen[1] == {"role": "user", "content": "hello world"}
    post(port, req(system=[{"type": "text", "text": "a"},
                           {"type": "text", "text": "b",
                            "cache_control": {"type": "ephemeral"}}]))
    assert eng.seen[0] == {"role": "system", "content": "a\n\nb"}


def test_user_text_blocks_join_and_history_strings_pass(fake):
    eng, port = fake(engine=Recording())
    post(port, req(messages=[
        {"role": "user", "content": [{"type": "text", "text": "hello"},
                                     {"type": "text", "text": "world"}]},
        {"role": "assistant", "content": "hi"},
        {"role": "user", "content": "again"}]))
    assert eng.seen == [{"role": "user", "content": "hello\n\nworld"},
                        {"role": "assistant", "content": "hi"},
                        {"role": "user", "content": "again"}]


def test_mid_conversation_system_message_is_a_system_turn_in_place(fake):
    # Claude Code sends {"role": "system"} after the first user turn on EVERY
    # request (seen on the wire)
    eng, port = fake(engine=Recording())
    r, body = post(port, req(messages=[
        {"role": "user", "content": [{"type": "text", "text": "say hi"}]},
        {"role": "system", "content": [{"type": "text", "text": "Available agent types"}]},
        {"role": "system", "content": []},  # effort-only: nothing to render
        {"role": "system", "content": "terse mode"}]))
    assert r.status == 200
    assert eng.seen == [{"role": "user", "content": "say hi"},
                        {"role": "system", "content": "Available agent types"},
                        {"role": "system", "content": "terse mode"}]
    assert json.loads(body)["content"] == [{"type": "text", "text": "echo: say hi"}]
    r, data = post(port, req(messages=[
        {"role": "user", "content": "x"},
        {"role": "system", "content": [{"type": "image", "source": {}}]}]))
    assert r.status == 400 and "messages.1.content.0" in envelope(data)["message"]


def test_head_answers_without_a_body(fake):
    # Claude Code probes HEAD /api/hello before its first request
    _, port = fake(auth="sekrit")
    for path, status in (("/health", 200), ("/api/hello", 404)):
        c = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
        c.request("HEAD", path)
        r = c.getresponse()
        assert r.status == status and r.read() == b""
        c.close()


def test_usage_splits_cached_prompt_tokens_the_anthropic_way(fake):
    # input_tokens EXCLUDES prefix-cache reads; the sum of the fields is the
    # prompt length once (Claude Code's context gauge sums them)
    class Cached(FakeEngine):
        def generate(self, req):
            def mark(ev):
                if isinstance(ev, Finished):
                    ev.result.cached_tokens = 1
            return relay(super().generate(req), mark)

    _, port = fake(engine=Cached())
    u = json.loads(post(port, req("hello world"))[1])["usage"]
    assert u == {"input_tokens": 1, "cache_read_input_tokens": 1,
                 "cache_creation_input_tokens": 0, "output_tokens": 3}


def test_metadata_headers_and_unknown_knobs_are_accepted_and_ignored(fake):
    _, port = fake()
    r, _ = post(port, req(metadata={"user_id": "u1"}, temperature=0.2, top_p=0.9,
                          top_k=40),
                headers={"anthropic-version": "2023-06-01",
                         "anthropic-beta": "prompt-caching-2024-07-31"})
    assert r.status == 200


# -------------------------------------------------------------- streaming --


def test_stream_event_order_and_assembled_text_equals_nonstream(fake):
    _, port = fake(reply="alpha beta gamma")
    r, body = post(port, req(stream=True))
    assert r.status == 200 and r.getheader("Content-Type") == "text/event-stream"
    evs = events(body)
    names = [n for n, _ in evs]
    assert names[:2] == ["message_start", "content_block_start"]
    assert names[2:-3] == ["content_block_delta"] * 3  # one per word-delta
    assert names[-3:] == ["content_block_stop", "message_delta", "message_stop"]
    start = evs[0][1]["message"]
    assert start["id"].startswith("msg_") and start["content"] == []
    assert start["stop_reason"] is None and start["usage"]["input_tokens"] == 2
    assert evs[1][1] == {"type": "content_block_start", "index": 0,
                         "content_block": {"type": "text", "text": ""}}
    assert evs[2][1]["delta"] == {"type": "text_delta", "text": "alpha "}
    m = assemble(evs)
    plain = json.loads(post(port, req())[1])
    assert m["content"] == plain["content"] == [{"type": "text", "text": "alpha beta gamma"}]
    assert m["stop_reason"] == plain["stop_reason"] == "end_turn"
    assert m["usage"] == plain["usage"]


def test_stream_thinking_block_then_text_block(fake):
    _, port = fake(reply=THINK, opens_think=True)
    evs = events(post(port, req(stream=True))[1])
    starts = [e["content_block"] for n, e in evs if n == "content_block_start"]
    assert starts == [{"type": "thinking", "thinking": "", "signature": ""},
                      {"type": "text", "text": ""}]
    kinds = [e["delta"]["type"] for n, e in evs if n == "content_block_delta"]
    i = kinds.index("signature_delta")  # one empty signature closes the block
    assert kinds[:i] and set(kinds[:i]) == {"thinking_delta"}
    assert kinds[i + 1:] and set(kinds[i + 1:]) == {"text_delta"}  # never interleaved
    m = assemble(evs)
    assert m["content"] == [{"type": "thinking", "thinking": "weighing it", "signature": ""},
                            {"type": "text", "text": "The answer."}]
    assert m["stop_reason"] == "end_turn"


def test_stream_block_indices_thinking_text_tool_use(fake):
    _, port = fake(tool_call_script=THINK_TOOL, opens_think=True)
    evs = events(post(port, req(tools=[TOOL], stream=True))[1])
    starts = [(e["index"], e["content_block"]["type"])
              for n, e in evs if n == "content_block_start"]
    assert starts == [(0, "thinking"), (1, "text"), (2, "tool_use")]
    assert [e["index"] for n, e in evs if n == "content_block_stop"] == [0, 1, 2]
    m = assemble(evs)
    assert m["stop_reason"] == "tool_use"
    assert m["content"][0]["thinking"] == "the user wants weather"
    assert m["content"][1]["text"] == "I will check.\n"
    assert m["content"][2]["name"] == "get_weather"
    assert m["content"][2]["input"] == {"city": "Paris"}
    assert m["content"][2]["id"].startswith("toolu_")


def test_truncated_mid_think_is_a_thinking_block_and_max_tokens(fake):
    _, port = fake(reply="still weighing the third", opens_think=True)
    obj = json.loads(post(port, req(max_tokens=3))[1])
    assert obj["content"] == [{"type": "thinking", "thinking": "still weighing the ",
                               "signature": ""}]
    assert obj["stop_reason"] == "max_tokens"


def test_disconnect_mid_stream_aborts_and_releases_the_lock(fake):
    eng, port = fake(reply="w " * 500, delay=0.01)  # ~5s if allowed to run out
    s = socket.create_connection(("127.0.0.1", port), timeout=5)
    body = json.dumps(req(stream=True)).encode()
    s.sendall(b"POST /v1/messages HTTP/1.1\r\nHost: t\r\n"
              b"Content-Type: application/json\r\n"
              b"Content-Length: " + str(len(body)).encode() + b"\r\n\r\n" + body)
    s.recv(1024)  # stream is live (headers + message_start)
    s.close()  # walk away mid-generation
    t0 = time.time()
    while time.time() - t0 < 5 and not eng.aborted:
        time.sleep(0.02)
    assert eng.aborted  # on_delta returned False and the engine stopped
    assert time.time() - t0 < 3  # promptly
    # and the lock is free: the next request answers at once
    eng.reply, eng.delay = "ok", 0.0
    t0 = time.time()
    r, data = post(port, req())
    assert r.status == 200 and time.time() - t0 < 2
    assert json.loads(data)["content"] == [{"type": "text", "text": "ok"}]


def test_disconnect_logs_a_distinct_line(fake, capsys):
    # A client whose SSH tunnel dies mid-stream (its laptop slept) gets a
    # clean 200 in the access log — the 200 status line went out at
    # _stream_headers() time, before the disconnect existed to see. This is
    # the line that says so.
    eng, port = fake(reply="w " * 500, delay=0.01)
    s = socket.create_connection(("127.0.0.1", port), timeout=5)
    body = json.dumps(req(stream=True)).encode()
    s.sendall(b"POST /v1/messages HTTP/1.1\r\nHost: t\r\n"
              b"Content-Type: application/json\r\n"
              b"Content-Length: " + str(len(body)).encode() + b"\r\n\r\n" + body)
    s.recv(1024)
    s.close()
    t0 = time.time()
    log = ""
    while time.time() - t0 < 5 and "stream aborted" not in log:
        log += capsys.readouterr().err
        time.sleep(0.02)
    assert "stream aborted (messages):" in log
    assert "tokens in" in log and "s" in log  # elapsed, formatted "%.1fs"


# ------------------------------------------------------------------ tools --


def test_tool_use_then_tool_result_round_trip(fake):
    eng, port = fake(engine=Recording(tool_call_script=SCRIPT))
    r, body = post(port, req("weather?", tools=[TOOL]))
    obj = json.loads(body)
    assert r.status == 200 and obj["stop_reason"] == "tool_use"
    text, call = obj["content"]
    assert text == {"type": "text", "text": "I will check.\n"}  # pre-call text survives
    assert call["type"] == "tool_use" and call["id"].startswith("toolu_")
    assert call["name"] == "get_weather"
    assert call["input"] == {"city": "Paris"}  # the PARSED object, never a string
    assert eng.last_tools == [WEATHER_TOOL]  # the OpenAI shape the template renders
    # turn 2: the assistant blocks echoed back + a tool_result -> the history
    # reaches the engine in exactly the shape the OpenAI boundary produces
    eng.tool_call_script = None
    r, body = post(port, req(tools=[TOOL], messages=[
        {"role": "user", "content": "weather?"},
        {"role": "assistant", "content": obj["content"]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": call["id"],
                                      "content": "72F"}]}]))
    assert r.status == 200
    assert eng.seen == [
        {"role": "user", "content": "weather?"},
        {"role": "assistant", "content": "I will check.\n",
         "tool_calls": [{"id": call["id"], "type": "function",
                         "function": {"name": "get_weather",
                                      "arguments": {"city": "Paris"}}}]},
        {"role": "tool", "tool_call_id": call["id"], "content": "72F"}]
    assert json.loads(body)["content"] == [{"type": "text", "text": "echo: weather?"}]


def test_tool_results_in_order_then_text_and_is_error_prefix(fake):
    eng, port = fake(engine=Recording())
    post(port, req(messages=[
        {"role": "user", "content": "go"},
        {"role": "assistant", "content": [
            {"type": "thinking", "thinking": "two calls", "signature": "abc"},
            {"type": "tool_use", "id": "toolu_a", "name": "a", "input": {}},
            {"type": "tool_use", "id": "toolu_b", "name": "b", "input": {"x": 1}}]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "toolu_a",
             "content": [{"type": "text", "text": "ok"}]},
            {"type": "tool_result", "tool_use_id": "toolu_b", "content": "boom",
             "is_error": True},
            {"type": "text", "text": "and now?"}]}]))
    assert eng.seen[1]["reasoning_content"] == "two calls"  # signature: accepted, dropped
    assert eng.seen[1]["content"] == ""
    assert [c["function"]["name"] for c in eng.seen[1]["tool_calls"]] == ["a", "b"]
    assert eng.seen[2:] == [
        {"role": "tool", "tool_call_id": "toolu_a", "content": "ok"},
        {"role": "tool", "tool_call_id": "toolu_b", "content": "Error: boom"},
        {"role": "user", "content": "and now?"}]


def test_input_json_delta_concatenation_equals_serialized_input(fake):
    two = ('<tool_call>\n{"name": "a", "arguments": {"city": "Par\\"is", "n": 1, '
           '"nested": {"k": [1, 2], "t": "café"}}}\n</tool_call>\n'
           '<tool_call>\n{"name": "b", "arguments": {}}\n</tool_call>')
    _, port = fake(tool_call_script=two)
    plain = json.loads(post(port, req(tools=[TOOL]))[1])
    plain_calls = [b for b in plain["content"] if b["type"] == "tool_use"]
    evs = events(post(port, req(tools=[TOOL], stream=True))[1])
    starts = {e["index"]: e["content_block"] for n, e in evs if n == "content_block_start"}
    partials: dict[int, list[str]] = {}
    for n, e in evs:
        if n == "content_block_delta" and e["delta"]["type"] == "input_json_delta":
            partials.setdefault(e["index"], []).append(e["delta"]["partial_json"])
    assert len(plain_calls) == len(partials) == 2
    for (idx, parts), b in zip(sorted(partials.items()), plain_calls):
        assert starts[idx]["type"] == "tool_use" and starts[idx]["name"] == b["name"]
        assert starts[idx]["input"] == {}  # the start block carries no input
        assert "".join(parts) == json.dumps(b["input"])  # byte for byte
        assert json.loads("".join(parts)) == b["input"]
    m = assemble(evs)
    assert [(b["name"], b["input"]) for b in m["content"] if b["type"] == "tool_use"] == \
        [(b["name"], b["input"]) for b in plain_calls]
    assert m["stop_reason"] == plain["stop_reason"] == "tool_use"


def test_a_tool_without_description_or_input_schema_reaches_the_engine_with_the_defaults(fake):
    # llama.cpp's defaults (common/chat.cpp): the template sees "" and {}
    eng, port = fake()
    r, _ = post(port, req(tools=[{"name": "ping"}]))
    assert r.status == 200
    assert eng.last_tools == [{"type": "function", "function": {
        "name": "ping", "description": "", "parameters": {}}}]
    r, _ = post(port, req(tools=[{"name": "ping", "description": None, "input_schema": None}]))
    assert r.status == 200
    assert eng.last_tools == [{"type": "function", "function": {
        "name": "ping", "description": "", "parameters": {}}}]


def test_tool_choice_auto_and_none(fake):
    eng, port = fake()
    post(port, req(tools=[TOOL], tool_choice={"type": "auto"}))
    assert eng.last_tools == [WEATHER_TOOL]
    post(port, req(tools=[TOOL], tool_choice={"type": "none"}))
    assert eng.last_tools is None
    post(port, req(tools=[]))
    assert eng.last_tools is None


def test_no_tools_means_tag_text_passes_through(fake):
    _, port = fake(tool_call_script=SCRIPT)
    obj = json.loads(post(port, req())[1])
    assert obj["content"] == [{"type": "text", "text": SCRIPT}]
    assert obj["stop_reason"] == "end_turn"


# ------------------------------------------ off-menu capability refusal --


def test_tools_refused_in_anthropics_envelope_when_unparseable(fake):
    _, port = fake(tool_format="unknown")
    r, body = post(port, req(tools=[TOOL]))
    assert r.status == 400
    err = envelope(body)
    assert err["type"] == "invalid_request_error"
    assert "drinkme-fake" in err["message"]


def test_tools_refused_when_format_is_none(fake):
    _, port = fake(tool_format="none")
    r, body = post(port, req(tools=[TOOL]))
    assert r.status == 400
    assert "no tool-calling support" in envelope(body)["message"]


def test_tool_choice_none_bypasses_the_anthropic_refusal(fake):
    _, port = fake(tool_format="unknown")
    r, _ = post(port, req(tools=[TOOL], tool_choice={"type": "none"}))
    assert r.status == 200


# --------------------------------------------------------------- thinking --


def test_thinking_enabled_reaches_the_template_and_renders_a_block(fake):
    eng, port = fake(engine=templating(reply=THINK))
    r, body = post(port, req(thinking={"type": "enabled", "budget_tokens": 1024}))
    assert r.status == 200
    assert eng.tok.calls[0]["enable_thinking"] is True
    assert json.loads(body)["content"] == [
        {"type": "thinking", "thinking": "weighing it", "signature": ""},
        {"type": "text", "text": "The answer."}]


def test_thinking_adaptive_is_enabled(fake):
    eng, port = fake(engine=templating(reply=THINK))
    post(port, req(thinking={"type": "adaptive"}))
    assert eng.tok.calls[0]["enable_thinking"] is True


def test_thinking_disabled_reaches_the_template_and_no_block(fake):
    # the template pre-fills the closed pair, the prompt opens nothing, so what
    # comes back is content byte for byte — the OpenAI path's contract
    eng, port = fake(engine=templating(reply=THINK))
    r, body = post(port, req(thinking={"type": "disabled"}))
    assert r.status == 200 and eng.tok.calls[0]["enable_thinking"] is False
    assert eng.tok.prompt.endswith("</think>\n\n")
    assert json.loads(body)["content"] == [{"type": "text", "text": THINK}]


def test_thinking_absent_sends_no_kwarg_and_effort_maps(fake):
    eng, port = fake(engine=templating())
    post(port, req())
    assert "enable_thinking" not in eng.tok.calls[0]
    post(port, req(output_config={"effort": "low"}))
    assert eng.tok.calls[1]["reasoning_effort"] == "low"


# --------------------------------------------------- output_config.format --

# The schema shape Claude Code sends for session-title generation: a small
# object, one required string field.
TITLE_SCHEMA = {"type": "object", "properties": {"title": {"type": "string"}},
                "required": ["title"], "additionalProperties": False}


def test_output_config_format_parses_and_reaches_engine(fake):
    class Capture(FakeEngine):
        def generate(self, req):
            self.rf = req.sampling.output_schema
            self.validated = req.sampling.output_schema_validated
            return super().generate(req)

    eng, port = fake(engine=Capture(reply='{"title": "ok"}'))
    r, _ = post(port, req(output_config={"format": {"type": "json_schema",
                                                     "schema": TITLE_SCHEMA}}))
    assert r.status == 200 and eng.rf == TITLE_SCHEMA
    # this layer walked the schema by name, and SAYS so, so the engine does
    # not walk it again (the OpenAI boundary's same posture)
    assert eng.validated is True


def test_claude_code_title_request_completes_with_schema_valid_content(fake):
    eng, port = fake(reply='{"title": "Fix the flaky retry test"}')
    r, data = post(port, req("Generate a five-word-or-fewer title for this session.",
                             output_config={"format": {"type": "json_schema",
                                                       "schema": TITLE_SCHEMA}}))
    assert r.status == 200
    obj = json.loads(data)
    assert obj["stop_reason"] == "end_turn"
    (block,) = obj["content"]
    assert block["type"] == "text"
    title = json.loads(block["text"])  # schema-valid: parses, has exactly "title"
    assert set(title) == {"title"} and isinstance(title["title"], str)


def test_claude_code_title_request_streams_schema_valid_content(fake):
    eng, port = fake(reply='{"title": "Fix the flaky retry test"}')
    r, body = post(port, req("Generate a five-word-or-fewer title for this session.",
                             output_config={"format": {"type": "json_schema",
                                                       "schema": TITLE_SCHEMA}},
                             stream=True))
    assert r.status == 200
    msg = assemble(events(body))
    assert msg["stop_reason"] == "end_turn"
    (block,) = msg["content"]
    title = json.loads(block["text"])
    assert set(title) == {"title"} and isinstance(title["title"], str)


def test_output_config_format_forces_thinking_off_and_the_splitter_with_it(fake):
    eng, port = fake(engine=templating(reply=THINK, opens_think=True))
    _, data = post(port, req(thinking={"type": "enabled"},
                             output_config={"format": {"type": "json_schema",
                                                       "schema": {"type": "object"}}}))
    assert eng.tok.calls[0]["enable_thinking"] is False
    assert json.loads(data)["content"] == [{"type": "text", "text": THINK}]


def test_output_config_format_unsupported_keyword_400_names_it(fake):
    _, port = fake()
    r, data = post(port, req(output_config={"format": {
        "type": "json_schema",
        "schema": {"type": "string", "pattern": "^a"}}}))
    assert r.status == 400
    assert "pattern" in envelope(data)["message"]


def test_output_config_format_with_tools_400(fake):
    _, port = fake()
    r, data = post(port, req(tools=[TOOL], output_config={
        "format": {"type": "json_schema", "schema": {"type": "object"}}}))
    assert r.status == 400
    assert "tools" in envelope(data)["message"]


def test_output_config_format_unsupported_variant_400_names_it(fake):
    _, port = fake()
    r, data = post(port, req(output_config={"format": {"type": "text"}}))
    assert r.status == 400
    assert "'text'" in envelope(data)["message"]


# ----------------------------------------------------------- count_tokens --


def test_count_tokens(fake):
    _, port = fake()
    body = {"model": MODEL, "system": "be brief",  # no max_tokens: not required here
            "messages": [{"role": "user", "content": "hello world"}]}
    r, data = post(port, body, path="/v1/messages/count_tokens")
    assert r.status == 200 and json.loads(data) == {"input_tokens": 4}
    # agrees with what a generation of the same request reports
    obj = json.loads(post(port, dict(body, max_tokens=8))[1])
    assert obj["usage"]["input_tokens"] == 4
    r, data = post(port, dict(body, model="claude-opus-5"), path="/v1/messages/count_tokens")
    assert r.status == 404 and envelope(data)["type"] == "not_found_error"


def test_count_tokens_renders_with_tools_and_thinking(fake):
    class TemplatingCount(Templating):  # HFEngine's count: the same render, counted
        def count_tokens(self, req):
            return len(self.render(req).ids)

    eng = TemplatingCount()
    eng.tok, eng.engine_kwargs = FakeTok(), None
    _, port = fake(engine=eng)
    r, data = post(port, {"model": MODEL, "tools": [TOOL], "thinking": {"type": "disabled"},
                          "messages": [{"role": "user", "content": "hi"}]},
                   path="/v1/messages/count_tokens")
    assert r.status == 200
    assert eng.tok.calls[0]["tools"] == [WEATHER_TOOL]
    assert eng.tok.calls[0]["enable_thinking"] is False
    assert json.loads(data)["input_tokens"] == len(eng.tok.prompt)
    assert eng.calls == 0  # counted, never generated


def test_count_tokens_malformed_fields_are_400_not_a_crash(fake):
    # count_tokens shares parse_request with the full
    # /v1/messages route, so the same guards apply — proven directly here
    # rather than assumed from the sibling table above.
    _, port = fake()
    for body, needle in ((dict(model=MODEL, system=[],
                               messages=[{"role": "user", "content": "hi"}]), "system"),
                         (dict(model=MODEL, tools=5,
                               messages=[{"role": "user", "content": "hi"}]), "tools"),
                         (dict(model=MODEL, stream="yes",
                               messages=[{"role": "user", "content": "hi"}]), "stream")):
        r, data = post(port, body, path="/v1/messages/count_tokens")
        assert r.status == 400, (body, data)
        assert needle in envelope(data)["message"]


def test_count_tokens_never_waits_for_the_lock(fake):
    eng, port = fake(reply="w " * 60, delay=0.02)  # ~1.2s generation
    t = threading.Thread(target=post, args=(port, req()))
    t.start()
    time.sleep(0.2)  # let the generation take the lock
    t0 = time.time()
    r, data = post(port, {"model": MODEL, "messages": [{"role": "user", "content": "a b c"}]},
                   path="/v1/messages/count_tokens")
    assert r.status == 200 and json.loads(data) == {"input_tokens": 3}
    assert time.time() - t0 < 1.0  # answered mid-generation
    t.join()


# ------------------------------------------------------------------- auth --


def test_x_api_key_and_bearer_both_open_the_gate(fake):
    _, port = fake(auth="sekrit")
    r, data = post(port, req())
    assert r.status == 401
    assert envelope(data) == {"type": "authentication_error", "message": "invalid x-api-key"}
    assert post(port, req(), headers={"x-api-key": "wrong"})[0].status == 401
    assert post(port, req(), headers={"x-api-key": "sekrit"})[0].status == 200
    assert post(port, req(), headers={"Authorization": "Bearer sekrit"})[0].status == 200
    r, data = post(port, {"model": MODEL, "messages": req()["messages"]},
                   path="/v1/messages/count_tokens")
    assert r.status == 401 and envelope(data)["type"] == "authentication_error"
    # one gate, two spellings: the OpenAI route honors x-api-key too
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    c.request("POST", "/v1/chat/completions", json.dumps({"messages": req()["messages"]}),
              {"Content-Type": "application/json", "x-api-key": "sekrit"})
    assert c.getresponse().status == 200
    c.close()
    assert get(port, "/health")[0].status == 200  # still open


# ----------------------------------------------------------------- errors --


def test_invalid_request_errors_name_the_field(fake):
    _, port = fake()
    no_max = {k: v for k, v in req().items() if k != "max_tokens"}
    no_model = {k: v for k, v in req().items() if k != "model"}
    cases = [
        ("{not json", "JSON"),
        (no_max, "max_tokens"),
        (req(max_tokens=0), "max_tokens"),
        (req(max_tokens="lots"), "max_tokens"),
        (no_model, "model"),
        (req(messages=[]), "messages"),
        (req(messages="hi"), "messages"),
        (req(messages=[5]), "messages.0"),
        (req(messages=[{"role": "tool", "content": "x"}]), "role"),
        (req(messages=[{"role": "user", "content": []}]), "empty"),
        (req(messages=[{"role": "user", "content": [{"type": "tool_result"}]}]),
         "tool_use_id"),
        (req(messages=[{"role": "assistant", "content": [
            {"type": "tool_use", "name": "f", "input": {}}]}]), "id"),
        (req(system=[{"type": "image"}]), "system"),
        (req(system=[]), "system"),  # an empty
                                     # top-level system array is named as
                                     # the likely mistake, never rendered as
                                     # silently "no system prompt"
        (req(tools=[{"type": "bash_20250124", "name": "bash"}]), "bash_20250124"),
        (req(tools=[{"name": "f", "input_schema": "x"}]), "input_schema"),
        (req(tools=5), "tools"),
        (req(tools=[TOOL], tool_choice={"type": "any"}), "tool_choice"),
        (req(tools=[TOOL], tool_choice={"type": "tool", "name": "get_weather"}), "tool_choice"),
        (req(tool_choice=[]), "tool_choice"),
        (req(thinking={"type": "bogus"}), "thinking"),
        (req(thinking={"type": "enabled", "budget_tokens": "x"}), "budget_tokens"),
        (req(output_config={"format": {"type": "text"}}), "output_config.format"),
        (req(output_config={"format": {"type": "json_schema"}}), "schema"),
        (req(output_config={"format": {"type": "json_schema",
                                       "schema": {"type": "string", "pattern": "^a"}}}),
         "pattern"),
        (req(output_config={"format": {"type": "json_schema", "schema": {
            "type": "object", "properties": []}}}), "properties"),
        (req(mcp_servers=[{"type": "url", "url": "http://x", "name": "x"}]), "mcp_servers"),
        (req(temperature="hot"), "temperature"),
        (req(stop_sequences="STOP"), "stop_sequences"),
        (req(stream="yes"), "stream"),
    ]
    for body, needle in cases:
        r, data = post(port, body)
        assert r.status == 400, (body, data)
        err = envelope(data)
        assert err["type"] == "invalid_request_error", body
        assert needle in err["message"], (needle, err["message"])


def test_malformed_model_type_is_a_safe_404_not_a_crash(fake):
    # `model: []`: the OpenAI boundary would crash
    # on it (a set-membership test on an unhashable list); this dialect
    # compares `model != model_id` directly, which never crashes on any
    # type, so a mismatched-type model is the SAME safe "unknown model" 404
    # every other wrong model value gets — recorded here so the table above
    # (400-only) doesn't have to special-case it.
    _, port = fake()
    r, data = post(port, req(model=[]))
    assert r.status == 404
    err = envelope(data)
    assert err["type"] == "not_found_error"


def test_malformed_fields_keep_the_connection_alive(fake):
    # a malformed field must never leave the connection in
    # a state a following valid request can't use — the ORIGINAL
    # reproduction was http.client.RemoteDisconnected. A fresh connection
    # per case (like the table above) would not catch that; this reuses ONE
    # connection across a bad request and a following good one.
    _, port = fake()
    for body in (req(messages=[5]),
                 req(tools=5),
                 req(tool_choice=[]),
                 req(output_config={"format": {"type": "json_schema", "schema": {
                     "type": "object", "properties": []}}}),
                 req(stream="yes"),
                 req(system=[])):
        c = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
        c.request("POST", "/v1/messages", json.dumps(body),
                  {"Content-Type": "application/json", "anthropic-version": "2023-06-01"})
        r = c.getresponse()
        r.read()
        assert r.status == 400, body
        c.request("POST", "/v1/messages", json.dumps(req()),
                  {"Content-Type": "application/json", "anthropic-version": "2023-06-01"})
        r2 = c.getresponse()
        assert r2.status == 200, (body, r2.read())
        c.close()


def test_image_and_document_blocks_are_refused_by_name(fake):
    _, port = fake()
    for kind in ("image", "document"):
        block = {"type": kind, "source": {"type": "base64", "media_type": "image/png",
                                          "data": "AAAA"}}
        r, data = post(port, req(messages=[{"role": "user", "content": [
            {"type": "text", "text": "what is this?"}, block]}]))
        assert r.status == 400
        err = envelope(data)
        assert err["type"] == "invalid_request_error"
        assert kind in err["message"] and "messages.0.content.1" in err["message"]
        # inside a tool_result too
        r, data = post(port, req(messages=[
            {"role": "user", "content": "x"},
            {"role": "assistant", "content": [{"type": "tool_use", "id": "t", "name": "f",
                                               "input": {}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t",
                                          "content": [block]}]}]))
        assert r.status == 400 and kind in envelope(data)["message"]


# ------------------------------------------------------------------ vision --
# A vision-capable engine accepts a base64 `image` block in a user turn AND
# inside `tool_result.content` (Claude Code's screenshot path), refuses
# `url`/`file` sources by name, and `document` blocks stay refused (test
# above, unaffected).


def _b64(data: bytes) -> str:
    return data_url(data).split(",", 1)[1]


class _Recording(VisionEngine):
    def generate(self, req):
        self.seen_images = req.images
        return super().generate(req)


def test_a_base64_image_block_is_accepted_in_a_user_turn(fake):
    eng, port = fake(engine=_Recording())
    block = {"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                         "data": _b64(tiny_png(1))}}
    r, data = post(port, req(messages=[{"role": "user", "content": [
        {"type": "text", "text": "what is this?"}, block]}]))
    assert r.status == 200, data
    assert len(eng.seen_images) == 1
    n = eng.seen_images[0].tokens
    # "what is this?" -- 3 words, FakeEngine's one-word-one-token count
    assert json.loads(data)["usage"]["input_tokens"] == 3 + n


def test_a_base64_image_is_accepted_inside_a_tool_result(fake):
    # Claude Code's screenshot path: a tool_result.content array carrying
    # both text and an image block.
    eng, port = fake(engine=_Recording())
    block = {"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                         "data": _b64(tiny_png(2))}}
    r, data = post(port, req(messages=[
        {"role": "user", "content": "x"},
        {"role": "assistant", "content": [{"type": "tool_use", "id": "t", "name": "f",
                                           "input": {}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t",
                                      "content": [{"type": "text", "text": "see:"}, block]}]}]))
    assert r.status == 200, data
    assert len(eng.seen_images) == 1


class _ImgHandler(http.server.BaseHTTPRequestHandler):
    """One PNG at any path — the "url sources are fetched by default" test's server."""

    def log_message(self, fmt, *a):
        pass

    def do_GET(self):
        body = tiny_png(98)
        self.send_response(200)
        self.send_header("Content-Type", "image/png")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def _img_server():
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _ImgHandler)
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def test_a_url_source_is_fetched_by_default(fake):
    # llama.cpp parity: Anthropic's source.type: "url" is DOWNLOADED
    # unless fetching is off — the download itself is
    # test_serving_vision_urls.py's job; this is the dialect wiring.
    srv = _img_server()
    try:
        eng, port = fake(engine=_Recording())
        url = f"http://127.0.0.1:{srv.server_address[1]}/pic.png"
        block = {"type": "image", "source": {"type": "url", "url": url}}
        r, data = post(port, req(messages=[{"role": "user", "content": [
            {"type": "text", "text": "what is this?"}, block]}]))
        assert r.status == 200, data
        assert len(eng.seen_images) == 1
    finally:
        srv.shutdown()


def test_url_sources_are_refused_when_fetching_is_off(fake):
    _, port = fake(engine=VisionEngine(veng=vision_engine(fetch_urls=False)))
    r, data = post(port, req(messages=[{"role": "user", "content": [
        {"type": "image", "source": {"type": "url", "url": "https://example.com/x.png"}}]}]))
    assert r.status == 400
    err = envelope(data)
    assert "base64 data URL" in err["message"]  # says what IS accepted, not a toggle
    assert "messages.0.content.0" in err["message"]


def test_file_sources_are_always_refused_no_files_api(fake):
    _, port = fake(engine=VisionEngine())
    r, data = post(port, req(messages=[{"role": "user", "content": [
        {"type": "image", "source": {"type": "file", "file_id": "f_1"}}]}]))
    assert r.status == 400
    err = envelope(data)
    assert "Files API" in err["message"] and "messages.0.content.0" in err["message"]


def test_document_blocks_stay_refused_even_with_vision(fake):
    _, port = fake(engine=VisionEngine())
    block = {"type": "document", "source": {"type": "base64", "media_type": "image/png",
                                            "data": _b64(tiny_png(1))}}
    r, data = post(port, req(messages=[{"role": "user", "content": [block]}]))
    assert r.status == 400 and "document" in envelope(data)["message"]


def test_count_tokens_expands_images_the_same_as_generate(fake):
    eng, port = fake(engine=_Recording())
    block = {"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                         "data": _b64(tiny_png(3))}}
    body = req(messages=[{"role": "user", "content": [
        {"type": "text", "text": "two words"}, block]}])
    r, data = post(port, body, path="/v1/messages/count_tokens")
    assert r.status == 200
    counted = json.loads(data)["input_tokens"]
    r2, data2 = post(port, body)
    assert r2.status == 200
    u = json.loads(data2)["usage"]
    assert counted == u["input_tokens"] + u["cache_read_input_tokens"]


def test_the_injection_guard_refuses_the_placeholder_literal(fake):
    eng, port = fake(engine=VisionEngine())
    r, data = post(port, req(messages=[
        {"role": "user", "content": f"say {vision.QwenVLPreprocessor.reserved_text[0]} back"}]))
    assert r.status == 400
    assert vision.QwenVLPreprocessor.reserved_text[0] in envelope(data)["message"]
    assert eng.calls == 0


def test_a_non_fetch_image_error_code_still_names_the_part(fake):
    # any of vision.ImageError's codes propagate through this dialect too,
    # not just the fetch-off one.
    _, port = fake(engine=VisionEngine())
    block = {"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                         "data": "not-base64!!"}}
    r, data = post(port, req(messages=[{"role": "user", "content": [block]}]))
    assert r.status == 400
    assert "messages.0.content.0" in envelope(data)["message"]


def test_an_is_error_tool_result_with_an_image_prefixes_a_text_part(fake):
    class _RecordingMessages(_Recording):
        def generate(self, req):
            self.seen_messages = req.messages
            return super().generate(req)

    eng, port = fake(engine=_RecordingMessages())
    block = {"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                         "data": _b64(tiny_png(4))}}
    r, data = post(port, req(messages=[
        {"role": "user", "content": "x"},
        {"role": "assistant", "content": [{"type": "tool_use", "id": "t", "name": "f",
                                           "input": {}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t",
                                      "is_error": True, "content": [block]}]}]))
    assert r.status == 200, data
    assert len(eng.seen_images) == 1
    tool_turn = next(m for m in eng.seen_messages if m.get("role") == "tool")
    assert tool_turn["content"] == [{"type": "text", "text": "Error: "}, {"type": "image"}]


def test_not_found_errors(fake):
    _, port = fake()
    r, data = post(port, req(model="claude-opus-5"))
    assert r.status == 404
    err = envelope(data)
    assert err["type"] == "not_found_error" and "claude-opus-5" in err["message"]
    r, data = post(port, req(), path="/v1/messages/batches")
    assert r.status == 404 and envelope(data)["type"] == "not_found_error"
    r, data = get(port, "/v1/messages")
    assert r.status == 404 and envelope(data)["type"] == "not_found_error"


def test_4xx_logs_one_line_with_the_reason(fake, capsys):
    # The exact shape of the 4xx-reason logging incident: before the Anthropic
    # Messages API mapped output_config.format to response_format, EVERY
    # output_config.format 400'd via MessagesError -> _aerror, which never
    # logged anything — Claude Code's title-generation request 400'd with a
    # log showing only the status.
    # An unsupported format variant still takes that same path by default.
    _, port = fake()
    capsys.readouterr()  # drain setup noise
    r, data = post(port, req(output_config={"format": {"type": "text"}}))
    assert r.status == 400
    msg = envelope(data)["message"]
    err = capsys.readouterr().err
    assert "POST" in err and "/v1/messages" in err
    assert "400" in err and msg in err


def test_engine_failure_is_api_error_in_the_envelope(fake):
    class Boom(FakeEngine):
        def generate(self, req):
            raise RuntimeError("boom")

    _, port = fake(engine=Boom())
    r, data = post(port, req())
    assert r.status == 500
    err = envelope(data)
    assert err["type"] == "api_error" and "boom" in err["message"]
    # mid-stream: headers are out, so the error is the last SSE event
    r, data = post(port, req(stream=True))
    assert r.status == 200
    evs = events(data)
    assert evs[0][0] == "message_start"
    assert evs[-1] == ("error", {"type": "error", "error": {
        "type": "api_error", "message": "RuntimeError: boom"}})


def test_template_refusal_is_a_400_naming_the_template_message(fake):
    # Qwen3.8's template raise_exception()s on a reasoning_effort it does not
    # know (it takes exactly xhigh/medium/low; Claude Code can send high/max):
    # a request-shape problem, answered as 400 with the template's own words —
    # on both dialects, and BEFORE the stream opens on this one (count_tokens
    # renders first)
    import jinja2

    class Picky(FakeTok):
        def apply_chat_template(self, messages, add_generation_prompt=False,
                                tokenize=True, **kw):
            if kw.get("reasoning_effort") not in (None, "xhigh", "medium", "low"):
                raise jinja2.exceptions.TemplateError(
                    f"Unexpected reasoning effort {kw['reasoning_effort']}. "
                    "Supported types are xhigh (default), medium, and low.")
            return super().apply_chat_template(messages, add_generation_prompt, tokenize, **kw)

    class Counting(Templating):  # HFEngine's count_tokens renders the template
        def count_tokens(self, req):
            return len(self.render(req).ids)

    eng = Templating()
    eng.tok, eng.engine_kwargs = Picky(), None
    _, port = fake(engine=eng)
    assert post(port, req(output_config={"effort": "xhigh"}))[0].status == 200
    r, data = post(port, req(output_config={"effort": "max"}))
    assert r.status == 400
    err = envelope(data)
    assert err["type"] == "invalid_request_error"
    assert "Unexpected reasoning effort max" in err["message"]
    # streamed against an engine whose count does NOT render: the headers are
    # out before the template speaks, so the refusal is the SSE error event
    r, data = post(port, req(output_config={"effort": "max"}, stream=True))
    assert r.status == 200
    name, ev = events(data)[-1]
    assert name == "error" and ev["error"]["type"] == "invalid_request_error"
    assert "Unexpected reasoning effort max" in ev["error"]["message"]
    # streamed against the real shape (count_tokens renders first): a clean
    # 400 before any stream opens
    eng2 = Counting()
    eng2.tok, eng2.engine_kwargs = Picky(), None
    _, port2 = fake(engine=eng2)
    r, data = post(port2, req(output_config={"effort": "max"}, stream=True))
    assert r.status == 400
    assert "Unexpected reasoning effort max" in envelope(data)["message"]
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    c.request("POST", "/v1/chat/completions",
              json.dumps({"messages": req()["messages"], "reasoning_effort": "max"}),
              {"Content-Type": "application/json"})
    r = c.getresponse()
    obj = json.loads(r.read())
    c.close()
    assert r.status == 400 and obj["error"]["type"] == "invalid_request_error"
    assert "Unexpected reasoning effort max" in obj["error"]["message"]


def test_openai_route_errors_keep_the_openai_shape(fake):
    # the dialect is decided by the path, never leaks across
    _, port = fake(auth="sekrit")
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    c.request("POST", "/v1/chat/completions", json.dumps({"messages": req()["messages"]}),
              {"Content-Type": "application/json"})
    r = c.getresponse()
    obj = json.loads(r.read())
    c.close()
    assert r.status == 401 and set(obj) == {"error"}
    assert obj["error"]["code"] == "invalid_api_key"


# ----------------------------------------------------- cross-surface parity --


class Recorder(Templating):
    """Everything the engine saw and said, per generate(): the internal
    history, the params, the tools, what reached apply_chat_template, and
    the exact delta transcript it delivered through on_delta."""

    def generate(self, req):
        deltas, result = [], []

        def tap(ev):
            if isinstance(ev, Delta):
                deltas.append(ev.text)
            elif isinstance(ev, Finished):
                result.append(ev.result)

        yield from relay(super().generate(req), tap)
        self.transcripts.append({"messages": req.messages, "params": req.sampling,
                                 "tools": req.tools, "template": self.tok.calls[-1],
                                 "deltas": deltas, "result": result[0]})


# turn 2 of an agentic exchange, thinking on, a system prompt and sampling
# knobs — once in each of the THREE dialects. No stop: the Responses wire
# has no such field, and the parity is over what all three can say (the
# plain-text test below holds stop across the two that have it)
OPENAI_REQ = {
    "model": MODEL, "max_tokens": 32, "temperature": 0.5, "top_p": 0.9, "top_k": 40,
    "chat_template_kwargs": {"enable_thinking": True},
    "tools": [WEATHER_TOOL],
    "messages": [
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": "weather in Paris?"},
        {"role": "assistant", "content": "I will check.\n", "reasoning_content": "hmm",
         "tool_calls": [{"id": "toolu_1", "type": "function",
                         "function": {"name": "get_weather",
                                      "arguments": "{\"city\": \"Paris\"}"}}]},
        {"role": "tool", "tool_call_id": "toolu_1", "content": "72F"},
        {"role": "user", "content": "and Rome?"}]}

ANTHROPIC_REQ = {
    "model": MODEL, "max_tokens": 32, "temperature": 0.5, "top_p": 0.9, "top_k": 40,
    "thinking": {"type": "enabled", "budget_tokens": 1024},
    "system": [{"type": "text", "text": "be brief"}],
    "tools": [TOOL],
    "messages": [
        {"role": "user", "content": [{"type": "text", "text": "weather in Paris?"}]},
        {"role": "assistant", "content": [
            {"type": "thinking", "thinking": "hmm", "signature": ""},
            {"type": "text", "text": "I will check.\n"},
            {"type": "tool_use", "id": "toolu_1", "name": "get_weather",
             "input": {"city": "Paris"}}]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "toolu_1", "content": "72F"},
            {"type": "text", "text": "and Rome?"}]}]}


# the Responses spelling: instructions + the output items of turn 1 echoed
# back as input, exactly as an SDK agent loop sends them (the flat tool
# shape; the chat path's top_k as the same extension; thinking on through
# the same chat_template_kwargs extension the chat path takes)
RESPONSES_REQ = {
    "model": MODEL, "max_output_tokens": 32, "temperature": 0.5, "top_p": 0.9, "top_k": 40,
    "chat_template_kwargs": {"enable_thinking": True},
    "instructions": "be brief",
    "tools": [{"type": "function", "name": "get_weather",
               "description": "Current weather for a city.",
               "parameters": WEATHER_TOOL["function"]["parameters"], "strict": False}],
    "input": [
        {"role": "user", "content": [{"type": "input_text", "text": "weather in Paris?"}]},
        {"type": "reasoning", "id": "rs_1",
         "summary": [{"type": "summary_text", "text": "hmm"}]},
        {"type": "message", "role": "assistant", "id": "msg_1", "status": "completed",
         "content": [{"type": "output_text", "text": "I will check.\n", "annotations": []}]},
        {"type": "function_call", "id": "fc_1", "call_id": "toolu_1", "name": "get_weather",
         "arguments": "{\"city\": \"Paris\"}", "status": "completed"},
        {"type": "function_call_output", "call_id": "toolu_1", "output": "72F"},
        {"role": "user", "content": "and Rome?"}]}


def _openai(port, body, path="/v1/chat/completions"):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    c.request("POST", path, json.dumps(body), {"Content-Type": "application/json"})
    r = c.getresponse()
    data = r.read()
    c.close()
    return r, data


def _responses(port, body):
    return _openai(port, body, path="/v1/responses")


def test_cross_surface_parity_same_engine_transcript(fake):
    """R1: one core. The same conversation, spoken in each of the three
    dialects, reaches the engine as the SAME history, params, tools and
    template kwargs, and the engine delivers the SAME delta transcript to
    all — streamed or not. Only the rendering differs, and the renderings
    correspond field for field."""
    eng = Recorder(tool_call_script=THINK_TOOL)
    eng.tok, eng.engine_kwargs, eng.transcripts = FakeTok(), None, []
    _, port = fake(engine=eng)
    r_oa, oa = _openai(port, OPENAI_REQ)
    r_an, an = post(port, ANTHROPIC_REQ)
    r_rs, rs = _responses(port, RESPONSES_REQ)
    assert r_oa.status == 200 and r_an.status == 200 and r_rs.status == 200
    _openai(port, dict(OPENAI_REQ, stream=True))
    post(port, dict(ANTHROPIC_REQ, stream=True))
    _responses(port, dict(RESPONSES_REQ, stream=True))
    t_oa, t_an, t_rs, s_oa, s_an, s_rs = eng.transcripts
    for a, b in ((t_oa, t_an), (t_oa, t_rs), (s_oa, s_an), (s_oa, s_rs), (t_oa, s_oa)):
        assert a["messages"] == b["messages"]
        assert a["params"] == b["params"]
        assert a["tools"] == b["tools"]
        assert a["template"] == b["template"]
        assert a["deltas"] == b["deltas"]  # transcript-identical
    assert t_oa["messages"] == [
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": "weather in Paris?"},
        {"role": "assistant", "content": "I will check.\n", "reasoning_content": "hmm",
         "tool_calls": [{"id": "toolu_1", "type": "function",
                         "function": {"name": "get_weather",
                                      "arguments": {"city": "Paris"}}}]},
        {"role": "tool", "tool_call_id": "toolu_1", "content": "72F"},
        {"role": "user", "content": "and Rome?"}]
    assert t_oa["template"]["enable_thinking"] is True
    assert t_oa["params"].stop == [] and t_oa["params"].temperature == 0.5
    assert t_oa["params"].top_k == 40 and t_oa["params"].max_tokens == 32
    # the three renderings of the one result correspond field for field
    m_oa = json.loads(oa)["choices"][0]["message"]
    m_an = json.loads(an)
    m_rs = json.loads(rs)
    think, text, call = m_an["content"]
    assert think == {"type": "thinking", "thinking": m_oa["reasoning_content"], "signature": ""}
    assert text == {"type": "text", "text": m_oa["content"]}
    (oa_call,) = m_oa["tool_calls"]
    assert call["name"] == oa_call["function"]["name"]
    assert call["input"] == json.loads(oa_call["function"]["arguments"])
    assert m_an["stop_reason"] == "tool_use"
    assert json.loads(oa)["choices"][0]["finish_reason"] == "tool_calls"
    reasoning, message, fcall = m_rs["output"]
    assert reasoning["type"] == "reasoning"
    assert reasoning["summary"] == [{"type": "summary_text", "text": m_oa["reasoning_content"]}]
    assert message["type"] == "message"
    assert message["content"] == [{"type": "output_text", "text": m_oa["content"],
                                   "annotations": []}]
    assert fcall["type"] == "function_call" and fcall["name"] == oa_call["function"]["name"]
    assert fcall["arguments"] == oa_call["function"]["arguments"]  # the same JSON string
    assert m_rs["status"] == "completed"
    u_oa, u_an, u_rs = json.loads(oa)["usage"], m_an["usage"], m_rs["usage"]
    assert u_an["input_tokens"] == u_oa["prompt_tokens"] == u_rs["input_tokens"]
    assert u_an["output_tokens"] == u_oa["completion_tokens"] == u_rs["output_tokens"]


def test_cross_surface_parity_plain_text_and_thinking_off(fake):
    eng = Recorder(reply="alpha beta gamma END omega")
    eng.tok, eng.engine_kwargs, eng.transcripts = FakeTok(), None, []
    _, port = fake(engine=eng)
    _, oa = _openai(port, {"model": MODEL, "max_tokens": 8, "stop": "END",
                           "chat_template_kwargs": {"enable_thinking": False},
                           "messages": [{"role": "user", "content": "hi"}]})
    _, an = post(port, {"model": MODEL, "max_tokens": 8, "stop_sequences": ["END"],
                        "thinking": {"type": "disabled"},
                        "messages": [{"role": "user", "content": "hi"}]})
    a, b = eng.transcripts
    assert (a["messages"], a["params"], a["tools"], a["template"], a["deltas"]) == \
        (b["messages"], b["params"], b["tools"], b["template"], b["deltas"])
    assert json.loads(oa)["choices"][0]["message"]["content"] == "alpha beta gamma "
    assert json.loads(an)["content"] == [{"type": "text", "text": "alpha beta gamma "}]
    assert json.loads(an)["stop_reason"] == "stop_sequence"
    assert json.loads(oa)["choices"][0]["finish_reason"] == "stop"


# --------------------------------------------------------------- untouched --


def test_models_route_unchanged(fake):
    _, port = fake()
    r, body = get(port, "/v1/models")
    obj = json.loads(body)
    assert r.status == 200 and obj["object"] == "list"
    assert obj["data"][0]["id"] == MODEL and "error" not in obj


# ------------------------------------------------------- SSE keep-alive --

# prefill_delay sleeps ONCE before the first delta, unlike test_serving_http's
# `delay` (per-DELTA) — the shape a real forward pass's prefill has.


def test_keepalive_ticks_during_prefill_then_stops(fake, monkeypatch):
    monkeypatch.setenv("DRINKME_SSE_KEEPALIVE_S", "0.03")
    _, port = fake(reply="alpha beta", prefill_delay=0.25)
    r, body = post(port, req(stream=True))
    assert r.status == 200
    blocks = [b for b in body.decode().split("\n\n") if b]
    # every frame is EITHER a keep-alive comment OR a whole named event —
    # the write lock (_sse_lock, http.py) is what keeps a tick from ever
    # landing inside one
    assert all(b == ": keep-alive" or b.startswith("event: ") for b in blocks)
    ka = [i for i, b in enumerate(blocks) if b == ": keep-alive"]
    assert len(ka) >= 1
    first_block = next(i for i, b in enumerate(blocks)
                       if b.startswith("event: content_block_start"))
    assert ka[0] < first_block  # the tick(s) precede the first real content
    # events() yields (None, None) for a frame with neither an 'event:' nor
    # a 'data:' line (a keep-alive comment is neither) and the caller drops
    # those — a real SDK-shaped parser reads the identical sequence and
    # reconstructs the message whole
    evs = [(n, d) for n, d in events(body) if n is not None]
    msg = assemble(evs)
    assert msg["content"] == [{"type": "text", "text": "alpha beta"}]


def test_keepalive_disabled_by_zero(fake, monkeypatch):
    monkeypatch.setenv("DRINKME_SSE_KEEPALIVE_S", "0")
    _, port = fake(reply="alpha beta", prefill_delay=0.15)
    r, body = post(port, req(stream=True))
    assert r.status == 200
    assert ": keep-alive" not in [b for b in body.decode().split("\n\n") if b]


# ---------------------------------------------- request-time context check --

# FakeEngine's default context_window (None) leaves every other test in this
# file unbounded; these set it explicitly. "hello world" is 2 fake tokens.


TWELVE = "w1 w2 w3 w4 w5 w6 w7 w8 w9 w10 w11 w12"  # a reply longer than any room below


def test_max_tokens_clamped_to_the_room_left(fake):
    """The max_tokens clamp: the Claude Code shape — max_tokens far above the room left (it
    sends 32000 on every call). Prompt 2 + 20 > ctx 10 but the prompt fits:
    clamp to 8, generate, stop_reason max_tokens. Same rule as vLLM's
    get_max_tokens (min(max_model_len - prompt, requested))."""
    eng, port = fake(context_window=10, reply=TWELVE)
    r, body = post(port, req(max_tokens=20))
    assert r.status == 200
    out = json.loads(body)
    assert out["stop_reason"] == "max_tokens"
    assert out["usage"]["output_tokens"] == 8
    assert eng.calls == 1


def test_context_length_refused_when_clamping_is_off(fake, monkeypatch):
    monkeypatch.setenv("DRINKME_MAX_TOKENS_CLAMP", "0")
    _, port = fake(context_window=10)
    r, body = post(port, req(max_tokens=20))
    assert r.status == 400
    err = json.loads(body)["error"]
    assert err["type"] == "invalid_request_error"
    # Anthropic's own canonical wording, not drinkme's.
    assert err["message"].startswith("prompt is too long:")
    assert "2" in err["message"] and "20" in err["message"] and "10" in err["message"]


def test_prompt_filling_the_window_is_refused(fake):
    eng, port = fake(context_window=2)  # "hello world" is exactly 2 tokens
    r, body = post(port, req(max_tokens=20))
    assert r.status == 400
    err = json.loads(body)["error"]
    assert err["type"] == "invalid_request_error" and "no room" in err["message"]
    assert eng.calls == 0


def test_prompt_filling_the_window_400_streaming(fake):
    _, port = fake(context_window=2)
    # message_start already went out before the check runs inside _generate,
    # so the stream answers 200 and the refusal rides a named `error` event
    r, body = post(port, req(max_tokens=20, stream=True))
    assert r.status == 200
    evs = [(n, d) for n, d in events(body) if n is not None]
    (err_ev,) = [d for n, d in evs if n == "error"]
    assert err_ev["error"]["type"] == "invalid_request_error"
    # Anthropic's own canonical wording, not drinkme's — pi and Claude
    # Code classify overflow by matching a provider's own dialect.
    assert err_ev["error"]["message"].startswith("prompt is too long:")


def test_context_length_within_window_is_fine(fake):
    _, port = fake(context_window=100)
    r, _ = post(port, req(max_tokens=20))
    assert r.status == 200


def test_context_length_check_shared_with_openai_dialect(fake):
    # ONE exception (http.ContextLengthExceeded), raised once from the
    # shared _generate core — both dialects refuse the SAME request
    eng, port = fake(context_window=2)  # the prompt fills the window
    r_an, _ = post(port, req(max_tokens=20))
    r_oa, _ = _openai(port, {"model": MODEL, "max_tokens": 20,
                             "messages": [{"role": "user", "content": "hello world"}]})
    assert r_an.status == r_oa.status == 400
    assert eng.calls == 0


def test_max_tokens_clamp_shared_with_openai_dialect(fake):
    # The max_tokens clamp is the same shared decision — both dialects run the
    # request with max_tokens cut to the room left, neither refuses
    eng, port = fake(context_window=10, reply=TWELVE)
    r_an, b_an = post(port, req(max_tokens=20))
    r_oa, b_oa = _openai(port, {"model": MODEL, "max_tokens": 20,
                                "messages": [{"role": "user", "content": "hello world"}]})
    assert r_an.status == r_oa.status == 200
    assert json.loads(b_an)["usage"]["output_tokens"] == 8
    assert json.loads(b_oa)["usage"]["completion_tokens"] == 8
    assert eng.calls == 2
