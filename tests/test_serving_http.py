"""The endpoint contract, exercised over real sockets against FakeEngine.

Every request here goes through http.client or a raw socket to a port-0
server — no mocked transport, so what passes here is what curl and the
OpenAI SDK will see. A real engine that satisfies serving/engine.Engine
inherits this whole contract untouched.
"""

import base64
import http.client
import io
import json
import socket
import threading
import time

import pytest
from PIL import Image

from drinkme.serving import vision as vision_mod
from drinkme.serving.engine import (Delta, FakeEngine, Finished, GenerationRequest, GenResult,
                                    SampleParams, StreamStart)
from drinkme.serving.http import start_server
from drinkme.serving.template import effective_kwargs, render_prompt


def relay(stream, on_event=None):
    """Re-yield an engine's event stream unchanged — every send() value
    passes through, so the abort contract survives the wrapper — while
    `on_event` sees each event. What a test engine that wants to LOOK at a
    generation wraps super().generate(req) in."""
    ok = None
    while True:
        try:
            ev = stream.send(ok)
        except StopIteration:
            return
        if on_event is not None:
            on_event(ev)
        ok = yield ev


@pytest.fixture
def fake():
    """fake(**FakeEngine kwargs) -> (engine, port); servers shut down after."""
    servers = []

    def make(engine=None, auth=None, advertised_ctx=None, **kw):
        eng = engine or FakeEngine(**kw)
        srv = start_server(eng, "127.0.0.1", 0, auth_token=auth,
                           advertised_ctx=advertised_ctx)
        servers.append(srv)
        return eng, srv.server_address[1]

    yield make
    for srv in servers:
        srv.shutdown()


def get(port, path):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    c.request("GET", path)
    r = c.getresponse()
    body = r.read()
    c.close()
    return r, body


def post(port, body, path="/v1/chat/completions"):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    payload = body if isinstance(body, (bytes, str)) else json.dumps(body)
    c.request("POST", path, payload,
              {"Content-Type": "application/json"})
    r = c.getresponse()
    data = r.read()
    c.close()
    return r, data


def sse_events(body: bytes) -> list[str]:
    return [b[len("data: "):] for b in body.decode().split("\n\n") if b.startswith("data: ")]


# ------------------------------------------------------------------- vision --
# Shared by the three dialect test files (test_serving_http_content_parts.py,
# test_serving_messages.py, test_serving_responses.py): a real vision.Vision
# (Qwen3.8-27B's processor config, verbatim from test_serving_vision.py's
# QWEN38), tiny synthetic PNGs, and a FakeEngine subclass that stands in for
# a vision engine — count_tokens/tokenize expand each
# `{"type": "image"}` part to its PreparedImage's real token count, the
# same arithmetic split between the engine (count_tokens, which carries
# GenerationRequest.images) and http.py's own /tokenize correction (whose
# Engine Protocol signature carries no images at all).

QWEN38_PROCESSOR = {
    "size": {"longest_edge": 16777216, "shortest_edge": 65536}, "patch_size": 16,
    "temporal_patch_size": 2, "merge_size": 2, "image_mean": [0.5, 0.5, 0.5],
    "image_std": [0.5, 0.5, 0.5], "processor_class": "Qwen3VLProcessor",
    "image_processor_type": "Qwen2VLImageProcessorFast",
}


def vision_engine(max_pixels=vision_mod.DEFAULT_MAX_PIXELS, fetch_urls=True,
                  media_path=None) -> vision_mod.Vision:
    return vision_mod.Vision(vision_mod.preprocessor_for("qwen3_5", QWEN38_PROCESSOR), max_pixels,
                             fetch_urls, media_path)


def tiny_png(seed: int = 0) -> bytes:
    """A small deterministic PNG, distinct per seed (different digest) —
    smaller than the processor's min_pixels, so smart_resize scales it up;
    still real pixels through the full decode/resize/patchify path."""
    im = Image.new("RGB", (32, 32), (seed % 256, (seed * 7) % 256, (seed * 13) % 256))
    buf = io.BytesIO()
    im.save(buf, "PNG")
    return buf.getvalue()


def data_url(data: bytes, mt: str = "image/png") -> str:
    return f"data:{mt};base64," + base64.b64encode(data).decode()


class VisionEngine(FakeEngine):
    """A FakeEngine with a real Vision attached (engine.vision, the OPTIONAL
    surface capability.engine_vision reads) whose count_tokens/tokenize/
    generate are image-aware, so the dialect tests can check the accept
    path and the token-count arithmetic without a real model."""

    def __init__(self, *a, veng=None, **kw):
        super().__init__(*a, **kw)
        self.vision = veng if veng is not None else vision_engine()

    def count_tokens(self, req):
        n, it = 0, iter(req.images)
        for m in req.messages:
            c = m.get("content", "")
            if isinstance(c, list):
                for p in c:
                    n += next(it).tokens if p.get("type") == "image" else len(
                        p.get("text", "").split())
            else:
                n += len(str(c).split())
        return n

    def tokenize(self, prompt=None, messages=None, tools=None, template_kwargs=None):
        if prompt is not None:
            words = prompt.split()
        else:
            words = []
            for m in (messages or []):
                c = m.get("content", "")
                if isinstance(c, list):
                    for p in c:
                        if p.get("type") == "image":
                            words.append("<image>")
                        else:
                            words.extend(p.get("text", "").split())
                else:
                    words.extend(str(c).split())
        return [self._vocab.setdefault(w, len(self._vocab)) for w in words]

    def generate(self, req):
        correct = self.count_tokens(req)

        def fix(ev):
            if isinstance(ev, Finished):
                ev.result.prompt_tokens = correct

        return relay(super().generate(req), fix)


def msgs(text="hello world"):
    return [{"role": "user", "content": text}]


# ---------------------------------------------------------------- /v1/models --


def test_models_shape(fake):
    _, port = fake()
    r, body = get(port, "/v1/models")
    obj = json.loads(body)
    assert r.status == 200 and obj["object"] == "list"
    (m,) = obj["data"]
    assert m["id"] == "drinkme-fake" and m["object"] == "model"
    assert m["owned_by"] == "drinkme" and isinstance(m["created"], int)
    # OpenAI's four top-level, every drinkme key under one `drinkme` object
    assert set(m) == {"id", "object", "created", "owned_by", "drinkme"}
    d = m["drinkme"]
    # pack provenance rides along when the engine carries it
    assert "hfRepo" in d and "revision" in d and "bitsPerWeight" in d and "meanTensorBitsPerWeight" in d
    assert "meanBpw" not in d  # the lexicon's names on the wire
    assert "compressionProfile" in d and "sourceDtype" in d and "contextWindow" in d
    assert set(d["sampling"]) == {"profile", "defaults"} and d["sampling"]["profile"] is None
    # vision is always present; imageInput only when true —
    # FakeEngine has no .vision by default, so it is false with no imageInput here
    assert set(d["capabilities"]) == {"thinking", "toolFormat", "thinkingSwitch", "vision"}
    assert d["capabilities"]["vision"] is False


def test_models_carries_the_capability_announcement(fake):
    # serving/capability.py: thinking/tool_format ride on /v1/models,
    # not recomputed per request — FakeEngine's defaults stand in for a
    # well-behaved menu model
    _, port = fake()
    (m,) = json.loads(get(port, "/v1/models")[1])["data"]
    assert m["drinkme"]["capabilities"] == {
        "thinking": "open", "toolFormat": "json", "vision": False,
        "thinkingSwitch": "chat_template_kwargs.enable_thinking"}
    _, port2 = fake(thinking="closed", tool_format="unknown")
    (m2,) = json.loads(get(port2, "/v1/models")[1])["data"]
    assert m2["drinkme"]["capabilities"] == {
        "thinking": "closed", "toolFormat": "unknown", "vision": False,
        "thinkingSwitch": "chat_template_kwargs.enable_thinking"}


def test_models_thinking_switch_is_null_when_there_is_no_thinking_knob(fake):
    # a template with no enable_thinking test anywhere has no key to name
    _, port = fake(thinking="none")
    (m,) = json.loads(get(port, "/v1/models")[1])["data"]
    assert m["drinkme"]["capabilities"]["thinkingSwitch"] is None


def test_models_bypasses_generation_lock(fake):
    eng, port = fake(reply="w " * 60, delay=0.02)  # ~1.2s generation
    t = threading.Thread(target=post, args=(port, {"messages": msgs()}))
    t.start()
    time.sleep(0.2)  # let the generation take the lock
    t0 = time.time()
    r, _ = get(port, "/v1/models")
    assert r.status == 200 and time.time() - t0 < 1.0  # answered mid-generation
    t.join()


# ---------------------------------------------------- chat.completion, plain --


def test_nonstream_contract(fake):
    _, port = fake()
    r, body = post(port, {"model": "drinkme-fake", "messages": msgs()})
    obj = json.loads(body)
    assert r.status == 200
    assert obj["id"].startswith("chatcmpl-") and obj["object"] == "chat.completion"
    (ch,) = obj["choices"]
    assert ch["index"] == 0 and ch["message"]["role"] == "assistant"
    assert ch["message"]["content"] == "echo: hello world"
    assert ch["finish_reason"] == "stop"
    u = obj["usage"]
    assert u["total_tokens"] == u["prompt_tokens"] + u["completion_tokens"]
    assert u["completion_tokens"] == 3  # "echo:", "hello", "world" — one word, one token


def test_max_tokens_finishes_length(fake):
    _, port = fake(reply="one two three four five")
    _, body = post(port, {"messages": msgs(), "max_tokens": 3})
    obj = json.loads(body)
    assert obj["choices"][0]["message"]["content"].split() == ["one", "two", "three"]
    assert obj["choices"][0]["finish_reason"] == "length"
    assert obj["usage"]["completion_tokens"] == 3


def test_stop_string_finishes_stop_and_is_not_emitted(fake):
    _, port = fake(reply="alpha beta STOP gamma")
    _, body = post(port, {"messages": msgs(), "stop": "STOP"})  # bare-string form
    obj = json.loads(body)
    content = obj["choices"][0]["message"]["content"]
    assert content == "alpha beta " and "STOP" not in content
    assert obj["choices"][0]["finish_reason"] == "stop"


def test_generations_queue_and_both_complete(fake):
    eng, port = fake(reply="w " * 10, delay=0.01)
    results = []
    ts = [threading.Thread(target=lambda: results.append(post(port, {"messages": msgs()})))
          for _ in range(2)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert [r.status for r, _ in results] == [200, 200] and eng.calls == 2


# ------------------------------------------------------------------ streaming --


def test_stream_sse_contract(fake):
    _, port = fake(reply="alpha beta gamma")
    r, body = post(port, {"messages": msgs(), "stream": True})
    assert r.status == 200 and r.getheader("Content-Type") == "text/event-stream"
    events = sse_events(body)
    assert events[-1] == "[DONE]"
    chunks = [json.loads(e) for e in events[:-1]]
    assert all(c["object"] == "chat.completion.chunk" for c in chunks)
    assert len({c["id"] for c in chunks}) == 1 and chunks[0]["id"].startswith("chatcmpl-")
    first = chunks[0]["choices"][0]
    assert first["delta"]["role"] == "assistant" and first["finish_reason"] is None
    final = chunks[-1]["choices"][0]
    assert final["delta"] == {} and final["finish_reason"] == "stop"
    text = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks)
    assert text == "alpha beta gamma"


def test_stream_matches_nonstream(fake):
    _, port = fake()
    req = {"messages": msgs("same input"), "max_tokens": 8}
    _, plain = post(port, req)
    _, streamed = post(port, {**req, "stream": True})
    chunks = [json.loads(e) for e in sse_events(streamed)[:-1]]
    text = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks)
    assert text == json.loads(plain)["choices"][0]["message"]["content"]


def test_disconnect_aborts_generation(fake):
    eng, port = fake(reply="w " * 500, delay=0.01)  # ~5s if allowed to run out
    s = socket.create_connection(("127.0.0.1", port), timeout=5)
    body = json.dumps({"messages": msgs(), "stream": True}).encode()
    s.sendall(b"POST /v1/chat/completions HTTP/1.1\r\nHost: t\r\n"
              b"Content-Type: application/json\r\n"
              b"Content-Length: " + str(len(body)).encode() + b"\r\n\r\n" + body)
    s.recv(1024)  # stream is live (headers + first chunks)
    s.close()  # walk away mid-generation
    t0 = time.time()
    while time.time() - t0 < 5 and not eng.aborted:
        time.sleep(0.02)
    assert eng.aborted  # on_delta returned False and the engine stopped
    assert time.time() - t0 < 3  # promptly — not after the full 5s of deltas


def test_disconnect_logs_a_distinct_line(fake, capsys):
    # A client whose SSH tunnel dies mid-stream (its laptop slept) gets a
    # clean 200 in the access log — the 200 status line went out at
    # _stream_headers() time, before the disconnect existed to see. This is
    # the line that says so.
    eng, port = fake(reply="w " * 500, delay=0.01)
    s = socket.create_connection(("127.0.0.1", port), timeout=5)
    body = json.dumps({"messages": msgs(), "stream": True}).encode()
    s.sendall(b"POST /v1/chat/completions HTTP/1.1\r\nHost: t\r\n"
              b"Content-Type: application/json\r\n"
              b"Content-Length: " + str(len(body)).encode() + b"\r\n\r\n" + body)
    s.recv(1024)
    s.close()
    t0 = time.time()
    log = ""
    while time.time() - t0 < 5 and "stream aborted" not in log:
        log += capsys.readouterr().err
        time.sleep(0.02)
    assert "stream aborted (chat.completions):" in log
    assert "tokens in" in log


# --------------------------------------------------------------- error shapes --


def assert_error_shape(body):
    err = json.loads(body)["error"]
    assert set(err) == {"message", "type", "param", "code"}
    return err


def test_bad_json_400(fake):
    _, port = fake()
    r, body = post(port, "{not json")
    assert r.status == 400
    assert assert_error_shape(body)["type"] == "invalid_request_error"


def test_4xx_logs_one_line_with_the_reason(fake, capsys):
    # A 400 whose log line shows only the status hides the reason in the
    # body (Claude Code's generate_session_title: "output_config.format...").
    _, port = fake()
    capsys.readouterr()  # drain setup noise
    r, body = post(port, {"messages": []})
    assert r.status == 400
    msg = assert_error_shape(body)["message"]
    err = capsys.readouterr().err
    assert "POST" in err and "/v1/chat/completions" in err
    assert "400" in err and msg in err


def test_4xx_log_line_never_carries_the_bearer_token(fake, capsys):
    _, port = fake(auth="sekrit-token-nobody-should-see")
    capsys.readouterr()
    r, _ = post(port, {"messages": msgs()})  # no Authorization header -> 401
    assert r.status == 401
    assert "sekrit-token-nobody-should-see" not in capsys.readouterr().err


def test_4xx_log_line_never_carries_the_request_body(fake, capsys):
    _, port = fake()
    capsys.readouterr()
    r, _ = post(port, {"messages": msgs("a secret prompt nobody should see in the log"),
                       "response_format": {"type": "json_object"},
                       "tools": [{"type": "function", "function": {"name": "f"}}]})
    assert r.status == 400
    assert "secret prompt" not in capsys.readouterr().err


def test_4xx_log_line_truncates_long_messages(fake, capsys):
    _, port = fake()
    capsys.readouterr()
    long_model = "x" * 1000
    r, _ = post(port, {"model": long_model, "messages": msgs()})
    assert r.status == 404
    assert len(capsys.readouterr().err) < len(long_model)


def test_missing_messages_400(fake):
    _, port = fake()
    for bad in ({}, {"messages": []}, {"messages": "hi"}):
        r, body = post(port, bad)
        assert r.status == 400
        assert "messages" in assert_error_shape(body)["message"]


def test_unknown_route_404(fake):
    _, port = fake()
    r, body = get(port, "/v1/nope")
    assert r.status == 404 and assert_error_shape(body)["code"] == "unknown_url"


def test_unknown_model_404(fake):
    _, port = fake()
    r, body = post(port, {"model": "gpt-4", "messages": msgs()})
    assert r.status == 404 and assert_error_shape(body)["code"] == "model_not_found"


def test_engine_failure_is_a_500_error_object(fake):
    class Boom(FakeEngine):
        def generate(self, req):
            raise RuntimeError("boom")

    _, port = fake(engine=Boom())
    r, body = post(port, {"messages": msgs()})
    assert r.status == 500
    err = assert_error_shape(body)
    assert err["type"] == "server_error" and "boom" in err["message"]


def test_parts_array_content_is_normalized_to_text(fake):
    # OpenAI clients (pi) legally send content as a parts array; a list handed
    # to a chat template renders as NOTHING and the model sees a blank turn —
    # found live; the model's own <think> was the bug report.
    _, port = fake()
    r, body = post(port, {"messages": [
        {"role": "user", "content": [{"type": "text", "text": "hello "},
                                     {"type": "text", "text": "world"}]}]})
    assert r.status == 200
    assert json.loads(body)["choices"][0]["message"]["content"] == "echo: hello world"


def test_null_content_normalizes_to_empty_not_500(fake):
    _, port = fake()
    r, body = post(port, {"messages": [
        {"role": "assistant", "content": None},
        {"role": "user", "content": "hello world"}]})
    assert r.status == 200


def test_stream_include_usage_sends_final_usage_chunk(fake):
    _, port = fake()
    _, body = post(port, {"messages": msgs(), "stream": True,
                          "stream_options": {"include_usage": True}})
    events = sse_events(body)
    assert events[-1] == "[DONE]"
    frames = [json.loads(e) for e in events if e != "[DONE]"]
    usage_frames = [f for f in frames if f.get("usage")]
    assert len(usage_frames) == 1
    uf = usage_frames[0]
    assert uf["choices"] == []  # spec: the usage chunk carries no delta
    assert uf["usage"]["total_tokens"] == (uf["usage"]["prompt_tokens"]
                                           + uf["usage"]["completion_tokens"])
    # and it arrives after the finish chunk (last frame before [DONE])
    assert frames[-1] is uf
    assert frames[-2]["choices"][0]["finish_reason"] == "stop"


def test_stream_without_stream_options_has_no_usage_chunk(fake):
    _, port = fake()
    _, body = post(port, {"messages": msgs(), "stream": True})
    frames = [json.loads(e) for e in sse_events(body) if e != "[DONE]"]
    assert not [f for f in frames if f.get("usage")]


# ---------------------------------------------------------------- tool calling --

# FakeEngine's tool_call_script replays this raw stream through the real
# ToolCallScanner path, word-split — the block arrives across many deltas,
# exactly as incremental detok delivers it.

WEATHER_TOOL = {"type": "function",
                "function": {"name": "get_weather",
                             "description": "Current weather for a city.",
                             "parameters": {"type": "object",
                                            "properties": {"city": {"type": "string"}},
                                            "required": ["city"]}}}

SCRIPT = ('I will check.\n<tool_call>\n{"name": "get_weather", '
          '"arguments": {"city": "Paris"}}\n</tool_call>')


def test_nonstream_tool_calls_shape(fake):
    _, port = fake(tool_call_script=SCRIPT)
    r, body = post(port, {"messages": msgs(), "tools": [WEATHER_TOOL]})
    obj = json.loads(body)
    assert r.status == 200
    (ch,) = obj["choices"]
    assert ch["finish_reason"] == "tool_calls"
    msg = ch["message"]
    assert msg["content"] == "I will check.\n"  # pre-call text survives as content
    (call,) = msg["tool_calls"]
    assert call["id"].startswith("call_") and call["type"] == "function"
    assert call["function"]["name"] == "get_weather"
    args = call["function"]["arguments"]
    assert isinstance(args, str)  # OpenAI shape: a JSON-encoded STRING
    assert json.loads(args) == {"city": "Paris"}
    assert obj["usage"]["completion_tokens"] == 10  # raw words incl. the block


def test_stream_tool_calls_contract(fake):
    _, port = fake(tool_call_script=SCRIPT)
    r, body = post(port, {"messages": msgs(), "stream": True, "tools": [WEATHER_TOOL]})
    assert r.status == 200
    events = sse_events(body)
    assert events[-1] == "[DONE]"
    chunks = [json.loads(e) for e in events[:-1]]
    text = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks)
    assert text == "I will check.\n"  # tool-call bytes never streamed as content
    tc = [c for c in chunks if "tool_calls" in c["choices"][0]["delta"]]
    assert len(tc) == 1  # ONE delta chunk carrying the complete calls
    (call,) = tc[0]["choices"][0]["delta"]["tool_calls"]
    assert call["index"] == 0 and call["type"] == "function"
    assert call["function"]["name"] == "get_weather"
    assert json.loads(call["function"]["arguments"]) == {"city": "Paris"}
    final = chunks[-1]["choices"][0]
    assert final["delta"] == {} and final["finish_reason"] == "tool_calls"
    assert chunks[-2] is tc[0]  # calls chunk arrives just before the finish chunk


def test_stream_two_calls_indexed_in_one_chunk(fake):
    two = ('<tool_call>\n{"name": "a", "arguments": {}}\n</tool_call>\n'
           '<tool_call>\n{"name": "b", "arguments": {"x": 1}}\n</tool_call>')
    _, port = fake(tool_call_script=two)
    _, body = post(port, {"messages": msgs(), "stream": True, "tools": [WEATHER_TOOL]})
    chunks = [json.loads(e) for e in sse_events(body)[:-1]]
    (tc,) = [c for c in chunks if "tool_calls" in c["choices"][0]["delta"]]
    calls = tc["choices"][0]["delta"]["tool_calls"]
    assert [c["index"] for c in calls] == [0, 1]
    assert [c["function"]["name"] for c in calls] == ["a", "b"]
    assert chunks[-1]["choices"][0]["finish_reason"] == "tool_calls"


def test_tools_reach_the_engine(fake):
    eng, port = fake()
    r, _ = post(port, {"messages": msgs(), "tools": [WEATHER_TOOL]})
    assert r.status == 200 and eng.last_tools == [WEATHER_TOOL]


def test_a_tool_without_description_or_parameters_reaches_the_engine_with_the_defaults(fake):
    # llama.cpp's defaults (common/chat.cpp): the template sees "" and {}
    eng, port = fake()
    r, _ = post(port, {"messages": msgs(), "tools": [{"type": "function",
                                                      "function": {"name": "ping"}}]})
    assert r.status == 200
    assert eng.last_tools == [{"type": "function", "function": {
        "name": "ping", "description": "", "parameters": {}}}]
    # an explicit null is a missing field, as in the Messages and Responses dialects
    r, body = post(port, {"messages": msgs(), "tools": [{"type": "function", "function": {
        "name": "ping", "description": None, "parameters": None}}]})
    assert r.status == 200, body
    assert eng.last_tools == [{"type": "function", "function": {
        "name": "ping", "description": "", "parameters": {}}}]


def test_tool_choice_none_withholds_tools_from_engine(fake):
    eng, port = fake()
    r, body = post(port, {"messages": msgs(), "tools": [WEATHER_TOOL],
                          "tool_choice": "none"})
    assert r.status == 200 and eng.last_tools is None
    assert "tool_calls" not in json.loads(body)["choices"][0]["message"]


def test_no_tools_means_tag_text_passes_through(fake):
    # a request WITHOUT tools must see the default bytes exactly — the scanner
    # only runs when tools were offered
    _, port = fake(tool_call_script=SCRIPT)
    _, body = post(port, {"messages": msgs()})
    obj = json.loads(body)
    msg = obj["choices"][0]["message"]
    assert "tool_calls" not in msg and msg["content"] == SCRIPT
    assert obj["choices"][0]["finish_reason"] == "stop"


def test_malformed_block_reemerges_as_text_not_a_call(fake):
    bad = '<tool_call>\n{"name": get_weather}\n</tool_call>'  # unquoted -> bad JSON
    _, port = fake(tool_call_script=bad)
    _, body = post(port, {"messages": msgs(), "tools": [WEATHER_TOOL]})
    obj = json.loads(body)
    msg = obj["choices"][0]["message"]
    assert "tool_calls" not in msg and msg["content"] == bad
    assert obj["choices"][0]["finish_reason"] == "stop"


def test_bad_tools_400(fake):
    _, port = fake()
    for bad in ("get_weather", [{"type": "retrieval"}], [{"type": "function"}],
                [{"type": "function", "function": {"name": ""}}],
                [{"type": "function", "function": {"name": "f", "parameters": []}}]):
        r, body = post(port, {"messages": msgs(), "tools": bad})
        assert r.status == 400
        assert assert_error_shape(body)["type"] == "invalid_request_error"


# ------------------------------------------ off-menu capability refusal --


def test_tools_refused_400_when_tool_format_is_unknown(fake):
    _, port = fake(tool_format="unknown")
    r, body = post(port, {"messages": msgs(), "tools": [WEATHER_TOOL]})
    assert r.status == 400
    err = assert_error_shape(body)
    assert err["type"] == "invalid_request_error" and err["code"] == "unsupported_tool_format"
    assert "drinkme-fake" in err["message"] and "tools" in err["message"]


def test_tools_refused_400_when_tool_format_is_none(fake):
    _, port = fake(tool_format="none")
    r, body = post(port, {"messages": msgs(), "tools": [WEATHER_TOOL]})
    assert r.status == 400
    assert "no tool-calling support" in assert_error_shape(body)["message"]


def test_refusal_never_reaches_generation(fake):
    # the refusal fires at the boundary — the engine must never see the call
    eng, port = fake(tool_format="unknown")
    post(port, {"messages": msgs(), "tools": [WEATHER_TOOL]})
    assert eng.calls == 0


def test_tool_choice_none_bypasses_the_refusal(fake):
    # tools withheld before the format check ever runs — nothing was going
    # to be offered to the model, so there is nothing to refuse
    _, port = fake(tool_format="unknown")
    r, _ = post(port, {"messages": msgs(), "tools": [WEATHER_TOOL],
                       "tool_choice": "none"})
    assert r.status == 200


def test_qwen_xml_tool_format_is_accepted_not_refused(fake):
    _, port = fake(tool_call_script=SCRIPT, tool_format="qwen-xml")
    r, _ = post(port, {"messages": msgs(), "tools": [WEATHER_TOOL]})
    assert r.status == 200


def test_tool_role_message_accepted(fake):
    # the second half of the agentic loop: assistant tool_calls turn (content
    # null) + a role:"tool" result must not be rejected anywhere
    _, port = fake()
    r, body = post(port, {"messages": [
        {"role": "user", "content": "weather?"},
        {"role": "assistant", "content": None,
         "tool_calls": [{"id": "call_1", "type": "function",
                         "function": {"name": "get_weather", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "call_1", "content": "72F"}],
        "tools": [WEATHER_TOOL]})
    assert r.status == 200
    assert json.loads(body)["choices"][0]["message"]["content"] == "echo: weather?"


# --------------------------------------------------------------- think channel --

# The measured Qwen3 shape: the template opens the block in the PROMPT, so the
# stream carries only the closing tag (opens_think=True says the prompt did
# it). A stream that opens the tag itself needs no flag.

THINK = "weighing it</think>\n\nThe answer."


class FakeTok:
    """Records what reaches apply_chat_template — the only place a template
    kwarg can be said to have ARRIVED — and renders Qwen3's two prompt tails:
    `<think>\\n` when thinking is on (the model opens in reasoning and only the
    CLOSING tag reaches the stream), the pre-filled empty pair when it is off.
    One id per character, so decode() of the tail is exact."""

    def __init__(self):
        self.calls = []
        self.prompt = ""

    def apply_chat_template(self, messages, add_generation_prompt=False,
                            tokenize=True, **kw):
        self.calls.append({"messages": messages, **kw})
        self.prompt = "assistant\n" + ("<think>\n\n</think>\n\n"
                                       if kw.get("enable_thinking") is False
                                       else "<think>\n")
        return list(range(len(self.prompt)))

    def decode(self, ids, skip_special_tokens=False):
        return self.prompt[-len(ids):]


class Templating(FakeEngine):
    """A FakeEngine that really renders a prompt, so the request's kwargs are
    checked where they LAND rather than where they were parsed, and the think
    channel is decided the way the real server decides it — off the prompt."""

    def render(self, req):
        """HFEngine._render, on the fake tokenizer: the request's kwargs over
        the engine's own (template.effective_kwargs), the tools, the
        family's template."""
        return render_prompt(self.tok, req.messages,
                             template_kwargs=effective_kwargs(
                                 self.engine_kwargs, req.template_kwargs,
                                 constrained=req.sampling.output_schema is not None),
                             tools=req.tools)

    # count_tokens stays FakeEngine's word count (no render): the tests that
    # want HFEngine's counting shape subclass this and count self.render(req)

    def generate(self, req):
        self.opens_think = self.render(req).opens_think  # the prompt has spoken
        return super().generate(req)


def templating(engine_kwargs=None, **kw):
    eng = Templating(**kw)
    eng.tok, eng.engine_kwargs = FakeTok(), engine_kwargs
    return eng


def channels(body: bytes) -> tuple[str, str]:
    """Streamed (reasoning, content), concatenated in arrival order."""
    deltas = [json.loads(e)["choices"][0]["delta"] for e in sse_events(body)
              if e != "[DONE]" and json.loads(e)["choices"]]
    return ("".join(d.get("reasoning_content", "") for d in deltas),
            "".join(d.get("content", "") for d in deltas))


def test_nonstream_splits_reasoning_off_the_answer(fake):
    _, port = fake(reply=THINK, opens_think=True)
    _, body = post(port, {"messages": msgs()})
    msg = json.loads(body)["choices"][0]["message"]
    assert msg["content"] == "The answer."  # the tag and its blank line, gone
    assert msg["reasoning_content"] == "weighing it"


def test_the_split_reads_the_models_rows_tags(fake):
    """gemma's channel through the same wire: the announced tool_format
    names the tool_formats row, and its think_open/think_close are what the
    splitter looks for — non-streamed and streamed alike."""
    reply = "<|channel>thought\nGiving the weather.\n<channel|>The weather is fine."
    _, port = fake(reply=reply, tool_format="gemma")
    _, body = post(port, {"messages": msgs()})
    msg = json.loads(body)["choices"][0]["message"]
    assert msg["reasoning_content"] == "Giving the weather.\n"
    assert msg["content"] == "The weather is fine."
    _, body = post(port, {"messages": msgs(), "stream": True})
    assert channels(body) == ("Giving the weather.\n", "The weather is fine.")
    # and on the default row the same bytes are content, untouched
    _, port = fake(reply=reply)
    _, body = post(port, {"messages": msgs()})
    msg = json.loads(body)["choices"][0]["message"]
    assert msg["content"] == reply and "reasoning_content" not in msg


def test_reasoning_content_is_absent_not_null_when_nothing_was_thought(fake):
    _, port = fake()  # no tags anywhere: a plain stream, a plain answer
    _, body = post(port, {"messages": msgs()})
    msg = json.loads(body)["choices"][0]["message"]
    assert msg["content"] == "echo: hello world"
    assert "reasoning_content" not in msg


def test_empty_think_block_omits_the_field_too(fake):
    _, port = fake(reply="</think>\n\nstraight to it", opens_think=True)
    _, body = post(port, {"messages": msgs()})
    msg = json.loads(body)["choices"][0]["message"]
    assert msg["content"] == "straight to it" and "reasoning_content" not in msg


def test_a_literal_opening_tag_splits_without_the_flag(fake):
    _, port = fake(reply="<think>\nweighing it</think>\n\nThe answer.")
    _, body = post(port, {"messages": msgs()})
    msg = json.loads(body)["choices"][0]["message"]
    assert msg["content"] == "The answer." and msg["reasoning_content"] == "weighing it"


def test_truncated_mid_think_is_reasoning_with_an_empty_answer(fake):
    # max_tokens ran out before the model stopped thinking — the honest shape
    # is an empty answer, not the thinking wearing the answer's field
    _, port = fake(reply="still weighing the third", opens_think=True)
    _, body = post(port, {"messages": msgs(), "max_tokens": 3})
    ch = json.loads(body)["choices"][0]
    assert ch["message"]["content"] == ""
    assert ch["message"]["reasoning_content"] == "still weighing the "
    assert ch["finish_reason"] == "length"


def test_stream_sends_reasoning_deltas_then_content_deltas(fake):
    _, port = fake(reply=THINK, opens_think=True)
    r, body = post(port, {"messages": msgs(), "stream": True})
    assert r.status == 200
    events = sse_events(body)
    assert events[-1] == "[DONE]"
    chunks = [json.loads(e) for e in events[:-1]]
    deltas = [c["choices"][0]["delta"] for c in chunks]
    assert deltas[0] == {"role": "assistant", "content": ""}  # role chunk unchanged
    assert channels(body) == ("weighing it", "The answer.")
    # no chunk carries both channels, and reasoning is DONE before content opens
    assert not any("reasoning_content" in d and "content" in d for d in deltas)
    kinds = [k for d in deltas[1:] for k in ("reasoning_content", "content") if d.get(k)]
    assert kinds == ["reasoning_content"] * 2 + ["content"] * 2  # never interleaved
    assert chunks[-1]["choices"][0] == {"index": 0, "delta": {}, "finish_reason": "stop"}


def test_stream_never_opens_the_content_channel_mid_think(fake):
    _, port = fake(reply="still weighing the third option", opens_think=True)
    _, body = post(port, {"messages": msgs(), "stream": True})
    deltas = [json.loads(e)["choices"][0]["delta"] for e in sse_events(body)[:-1]]
    assert channels(body) == ("still weighing the third option", "")
    assert not any("content" in d for d in deltas[1:])  # role chunk aside


def test_stream_think_matches_nonstream_think(fake):
    _, port = fake(reply=THINK, opens_think=True)
    _, plain = post(port, {"messages": msgs()})
    _, streamed = post(port, {"messages": msgs(), "stream": True})
    msg = json.loads(plain)["choices"][0]["message"]
    assert channels(streamed) == (msg["reasoning_content"], msg["content"])


def test_enable_thinking_false_no_ops_the_splitter(fake):
    # end to end: the request's kwarg reaches the template, the template
    # pre-fills the closed pair, the prompt therefore opens nothing, and what
    # comes back is the answer byte for byte — tags and all
    eng, port = fake(engine=templating(reply=THINK))
    _, body = post(port, {"messages": msgs(),
                          "chat_template_kwargs": {"enable_thinking": False}})
    assert eng.tok.prompt.endswith("</think>\n\n")
    msg = json.loads(body)["choices"][0]["message"]
    assert msg["content"] == THINK and "reasoning_content" not in msg


def test_an_engine_that_opens_nothing_keeps_its_whole_answer_as_content(fake):
    # The channel is never GUESSED. An engine whose prompt opened no block —
    # a family with no thinking mode, or none built at all — must keep every
    # byte in content; filing an answer as reasoning is an empty answer.
    class Minimal:  # the Engine seam, nothing else
        model_id = "drinkme-fake"

        def model_meta(self):  # the contract's two required sub-objects, empty
            return {"capabilities": {"thinking": "none", "toolFormat": "none"},
                    "sampling": {"profile": None, "defaults": {}}}

        def generate(self, req):
            yield StreamStart(1)
            yield Delta(THINK)
            yield Finished(GenResult(THINK, "stop", 1, 4))

    _, port = fake(engine=Minimal())
    _, body = post(port, {"messages": msgs()})
    msg = json.loads(body)["choices"][0]["message"]
    assert msg["content"] == THINK and "reasoning_content" not in msg


def test_a_prompt_that_ends_inside_think_turns_the_channel_on(fake):
    # the real decision path: the template wrote "<think>\n" at the end of the
    # prompt, so byte one of the answer is reasoning (nobody had to be told)
    eng, port = fake(engine=templating(reply=THINK))
    _, body = post(port, {"messages": msgs()})
    assert eng.tok.prompt.endswith("<think>\n")
    msg = json.loads(body)["choices"][0]["message"]
    assert msg["reasoning_content"] == "weighing it" and msg["content"] == "The answer."


def test_a_prompt_with_the_pair_pre_filled_leaves_the_channel_off(fake):
    # enable_thinking=False at the ENGINE (a server-wide default, no request
    # field): the template pre-fills "<think>\n\n</think>\n\n", so the prompt
    # closes what it opened and the answer is content, tags and all
    eng, port = fake(engine=templating(engine_kwargs={"enable_thinking": False},
                                       reply=THINK))
    _, body = post(port, {"messages": msgs()})
    assert eng.tok.prompt.endswith("</think>\n\n")
    msg = json.loads(body)["choices"][0]["message"]
    assert msg["content"] == THINK and "reasoning_content" not in msg


# ------------------------------------------------- chat_template_kwargs plumbing --


def test_chat_template_kwargs_reach_apply_chat_template(fake):
    eng, port = fake(engine=templating())
    r, _ = post(port, {"messages": msgs(),
                       "chat_template_kwargs": {"enable_thinking": False}})
    assert r.status == 200 and eng.tok.calls[0]["enable_thinking"] is False


def test_reasoning_effort_merges_into_the_template_kwargs(fake):
    eng, port = fake(engine=templating())
    r, _ = post(port, {"messages": msgs(), "reasoning_effort": "low"})
    assert r.status == 200 and eng.tok.calls[0]["reasoning_effort"] == "low"


def test_explicit_chat_template_kwargs_win_over_reasoning_effort(fake):
    eng, port = fake(engine=templating())
    post(port, {"messages": msgs(), "reasoning_effort": "low",
                "chat_template_kwargs": {"reasoning_effort": "xhigh",
                                         "preserve_thinking": False}})
    assert eng.tok.calls[0]["reasoning_effort"] == "xhigh"
    assert eng.tok.calls[0]["preserve_thinking"] is False


def test_request_kwargs_beat_the_engines_own_but_leave_the_rest(fake):
    eng, port = fake(engine=templating(engine_kwargs={"enable_thinking": False,
                                                      "preserve_thinking": True}))
    post(port, {"messages": msgs(),
                "chat_template_kwargs": {"enable_thinking": True}})
    assert eng.tok.calls[0]["enable_thinking"] is True
    assert eng.tok.calls[0]["preserve_thinking"] is True


def test_no_kwargs_leaves_the_template_call_exactly_as_today(fake):
    eng, port = fake(engine=templating())
    post(port, {"messages": msgs()})
    post(port, {"messages": msgs(), "chat_template_kwargs": {}})
    assert eng.tok.calls[0] == eng.tok.calls[1] == {"messages": eng.tok.calls[0]["messages"]}


def test_kwargs_do_not_leak_from_one_request_to_the_next(fake):
    eng, port = fake(engine=templating())
    post(port, {"messages": msgs(), "chat_template_kwargs": {"enable_thinking": False}})
    post(port, {"messages": msgs()})
    assert "enable_thinking" not in eng.tok.calls[1]


def test_bad_template_kwargs_400(fake):
    _, port = fake()
    for bad in ({"chat_template_kwargs": "enable_thinking"},
                {"chat_template_kwargs": ["enable_thinking"]},
                {"reasoning_effort": 3}):
        r, body = post(port, {"messages": msgs(), **bad})
        assert r.status == 400, bad
        assert assert_error_shape(body)["type"] == "invalid_request_error"


def test_kwargs_that_collide_with_the_template_call_are_400_not_500(fake):
    # a duplicate argument would blow up inside the engine; say so at the door
    _, port = fake(engine=templating())
    for name in ("tokenize", "add_generation_prompt", "conversation", "tools"):
        r, body = post(port, {"messages": msgs(), "chat_template_kwargs": {name: 1}})
        assert r.status == 400, name
        assert name in assert_error_shape(body)["message"]


def test_history_reasoning_content_survives_normalization(fake):
    # the second turn of a reasoning conversation: Qwen3's template renders the
    # assistant turn's reasoning_content, and content still normalizes to text
    eng, port = fake(engine=templating())
    r, _ = post(port, {"messages": [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": [{"type": "text", "text": "42"}],
         "reasoning_content": "counted twice"},
        {"role": "user", "content": "again?"}]})
    assert r.status == 200
    hist = eng.tok.calls[0]["messages"]
    assert hist[1]["reasoning_content"] == "counted twice"
    assert hist[1]["content"] == "42"  # parts array still flattened


def test_response_format_forces_thinking_off_and_the_splitter_with_it(fake):
    # constrain.py's posture: a grammar that bans "<" starves a thinking model.
    # The boundary says so out loud, and the splitter no-ops on what comes back.
    eng, port = fake(engine=templating(reply=THINK, opens_think=True))
    _, body = post(port, {"messages": msgs(),
                          "chat_template_kwargs": {"enable_thinking": True},
                          "response_format": {"type": "json_object"}})
    assert eng.tok.calls[0]["enable_thinking"] is False
    msg = json.loads(body)["choices"][0]["message"]
    assert msg["content"] == THINK and "reasoning_content" not in msg


# ------------------------------------------------------- think + tool calling --

THINK_TOOL = ('the user wants weather</think>\n\nI will check.\n<tool_call>\n'
              '{"name": "get_weather", "arguments": {"city": "Paris"}}\n</tool_call>')


def test_nonstream_tool_call_after_think(fake):
    _, port = fake(tool_call_script=THINK_TOOL, opens_think=True)
    _, body = post(port, {"messages": msgs(), "tools": [WEATHER_TOOL]})
    ch = json.loads(body)["choices"][0]
    assert ch["finish_reason"] == "tool_calls"
    msg = ch["message"]
    assert msg["reasoning_content"] == "the user wants weather"
    assert msg["content"] == "I will check.\n"  # the ANSWER side, never the think
    (call,) = msg["tool_calls"]
    assert call["function"]["name"] == "get_weather"


def test_stream_tool_call_after_think(fake):
    _, port = fake(tool_call_script=THINK_TOOL, opens_think=True)
    _, body = post(port, {"messages": msgs(), "stream": True,
                          "tools": [WEATHER_TOOL]})
    chunks = [json.loads(e) for e in sse_events(body)[:-1]]
    assert channels(body) == ("the user wants weather", "I will check.\n")
    (tc,) = [c for c in chunks if "tool_calls" in c["choices"][0]["delta"]]
    assert tc["choices"][0]["delta"]["tool_calls"][0]["function"]["name"] == "get_weather"
    assert chunks[-2] is tc  # calls chunk still lands just before the finish
    assert chunks[-1]["choices"][0]["finish_reason"] == "tool_calls"


# ------------------------------------------------ the cheap sweep --


def test_health_open_and_lockless(fake):
    _, port = fake(auth="sekrit")  # gated server: /health must STILL answer
    r, body = get(port, "/health")
    obj = json.loads(body)
    assert r.status == 200 and obj["ok"] is True
    assert obj["model"] == "drinkme-fake" and isinstance(obj["uptime_s"], int)
    # `device` must be PRESENT even when the engine reports none (null), so a
    # probe can assert on it unconditionally. {"ok":true} alone can hide
    # weights sitting on CPU after a venv resync — liveness is not the
    # question. A real engine puts "cuda:0" here.
    assert "device" in obj


def test_bearer_gates_v1_routes(fake):
    _, port = fake(auth="sekrit")
    for path in ("/v1/models",):
        r, body = get(port, path)
        assert r.status == 401
        assert json.loads(body)["error"]["code"] == "invalid_api_key"
    r, body = post(port, {"messages": msgs()})
    assert r.status == 401
    # wrong scheme/token also refused
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    c.request("GET", "/v1/models", headers={"Authorization": "Bearer wrong"})
    assert c.getresponse().status == 401
    c.close()
    # the right token opens both
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    c.request("GET", "/v1/models", headers={"Authorization": "Bearer sekrit"})
    assert c.getresponse().status == 200
    c.close()
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    c.request("POST", "/v1/chat/completions", json.dumps({"messages": msgs()}),
              {"Content-Type": "application/json",
               "Authorization": "Bearer sekrit"})
    assert c.getresponse().status == 200
    c.close()


def test_no_auth_configured_stays_open(fake):
    _, port = fake()  # no token: local default, everything answers
    assert get(port, "/v1/models")[0].status == 200
    assert post(port, {"messages": msgs()})[0].status == 200


def test_cors_preflight_and_headers(fake):
    _, port = fake()
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    c.request("OPTIONS", "/v1/chat/completions")
    r = c.getresponse()
    r.read()
    assert r.status == 204
    assert r.getheader("Access-Control-Allow-Origin") == "*"
    assert "POST" in r.getheader("Access-Control-Allow-Methods")
    assert "Authorization" in r.getheader("Access-Control-Allow-Headers")
    c.close()
    r, _ = post(port, {"messages": msgs()})
    assert r.getheader("Access-Control-Allow-Origin") == "*"
    r, _ = post(port, {"messages": msgs(), "stream": True})
    assert r.getheader("Access-Control-Allow-Origin") == "*"


def test_penalty_params_parse_and_reject(fake):
    eng, port = fake()
    r, _ = post(port, {"messages": msgs(), "presence_penalty": 0.5,
                       "frequency_penalty": 0.25})
    assert r.status == 200  # FakeEngine ignores them; parsing is what's under test
    r, body = post(port, {"messages": msgs(), "presence_penalty": "high"})
    assert r.status == 400
    assert assert_error_shape(body)["type"] == "invalid_request_error"


# ------------------------------------------------------- response_format --


def test_response_format_parses_and_reaches_engine(fake):
    class Capture(FakeEngine):
        def generate(self, req):
            self.rf = req.sampling.output_schema
            self.validated = req.sampling.output_schema_validated
            return super().generate(req)

    eng, port = fake(engine=Capture(reply="ok"))
    schema = {"type": "object", "properties": {"a": {"type": "string"}},
              "required": ["a"], "additionalProperties": False}
    r, _ = post(port, {"messages": msgs(), "response_format":
                       {"type": "json_schema",
                        "json_schema": {"name": "t", "schema": schema}}})
    assert r.status == 200 and eng.rf == schema
    # this layer walked the schema by name (that is what makes an unsupported
    # keyword a 400), and SAYS so, so the engine does not walk it again
    # (see SampleParams.output_schema_validated)
    assert eng.validated is True
    r, _ = post(port, {"messages": msgs(),
                       "response_format": {"type": "json_object"}})
    assert r.status == 200 and eng.rf == {"type": "object"}
    r, _ = post(port, {"messages": msgs(), "response_format": {"type": "text"}})
    assert r.status == 200 and eng.rf is None


def test_response_format_unsupported_keyword_400_names_it(fake):
    _, port = fake()
    r, body = post(port, {"messages": msgs(), "response_format":
                          {"type": "json_schema",
                           "json_schema": {"schema": {"type": "string",
                                                      "pattern": "^a"}}}})
    assert r.status == 400
    assert "pattern" in assert_error_shape(body)["message"]


def test_response_format_unsatisfiable_const_enum_400(fake):
    """Enum/const alongside sibling constraints (here `const`
    beside `required`) must be refused through the SAME door an unsupported
    keyword uses — a structured 400 naming the schema unsatisfiable, before
    a token generates — not manufactured into a fake valid instance."""
    _, port = fake()
    r, body = post(port, {"messages": msgs(), "response_format":
                          {"type": "json_schema",
                           "json_schema": {"schema": {"type": "object",
                                                      "required": ["x"],
                                                      "const": {}}}}})
    assert r.status == 400
    assert "unsatisfiable" in assert_error_shape(body)["message"]


def test_response_format_bad_shapes_400(fake):
    _, port = fake()
    for rf in ("json", {"type": "yaml"}, {"type": "json_schema"},
               {"type": "json_schema", "json_schema": {"schema": "x"}}):
        r, body = post(port, {"messages": msgs(), "response_format": rf})
        assert r.status == 400, rf
        assert_error_shape(body)


def test_response_format_with_tools_400(fake):
    _, port = fake()
    r, body = post(port, {"messages": msgs(),
                          "response_format": {"type": "json_object"},
                          "tools": [{"type": "function",
                                     "function": {"name": "f",
                                                  "parameters": {"type": "object"}}}]})
    assert r.status == 400
    assert "tools" in assert_error_shape(body)["message"]


def test_developer_role_maps_to_system(fake):
    """OpenAI's system->developer rename (pi sends it); Qwen templates accept
    only system/user/assistant/tool and raise on anything else. Found live,
    on an agentic pi request against Qwen3.8-27B."""
    seen = {}

    class Eng(FakeEngine):
        def generate(self, req):
            seen["roles"] = [m.get("role") for m in req.messages]
            return super().generate(req)

    _, port = fake(engine=Eng())
    r, data = post(port, {
        "model": "drinkme-fake", "max_tokens": 8,
        "messages": [{"role": "developer", "content": "be brief"},
                     {"role": "user", "content": "hi"}]})
    assert r.status == 200
    assert seen["roles"][0] == "system"
    assert json.loads(data)["choices"][0]["message"]["role"] == "assistant"


def test_history_tool_call_arguments_parsed_to_dict(fake):
    """OpenAI wire shape carries function.arguments as a JSON string; the
    qwen3_5 template renders arguments with |items and TypeErrors on a
    string — turn 2 of every agentic exchange. Boundary parses it back."""
    seen = {}

    class Eng(FakeEngine):
        def generate(self, req):
            seen["msgs"] = req.messages
            return super().generate(req)

    _, port = fake(engine=Eng())
    r, data = post(port, {
        "model": "drinkme-fake", "max_tokens": 8,
        "messages": [
            {"role": "user", "content": "list files"},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "call_1", "type": "function",
                 "function": {"name": "bash", "arguments": "{\"command\": \"ls\"}"}}]},
            {"role": "tool", "content": "file1"}]})
    assert r.status == 200
    tc = seen["msgs"][1]["tool_calls"][0]
    assert tc["function"]["arguments"] == {"command": "ls"}
    # junk stays junk (fails legibly downstream), never invented structure
    _, port2 = fake(engine=Eng())
    post(port2, {
        "model": "drinkme-fake", "max_tokens": 8,
        "messages": [
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "c", "type": "function",
                 "function": {"name": "x", "arguments": "not json"}}]},
            {"role": "user", "content": "hi"}]})
    assert seen["msgs"][0]["tool_calls"][0]["function"]["arguments"] == "not json"


# ------------------------------------------------------- SSE keep-alive --

# FakeEngine's prefill_delay sleeps ONCE, before the first delta — the shape
# of a real forward pass's prefill, which `delay` (a per-DELTA sleep) cannot
# stand in for. Interval is set far below prefill_delay so >=1 tick is
# reliable without a slow test.

def blocks_of(body: bytes) -> list[str]:
    """Every non-empty SSE frame, keep-alive comments included — sse_events()
    (data-only) deliberately filters those out; this is the complement: what
    actually crossed the wire, one frame per element."""
    return [b for b in body.decode().split("\n\n") if b]


def test_keepalive_ticks_during_prefill_then_stops_openai(fake, monkeypatch):
    monkeypatch.setenv("DRINKME_SSE_KEEPALIVE_S", "0.03")
    _, port = fake(reply="alpha beta", prefill_delay=0.25)
    r, body = post(port, {"messages": msgs(), "stream": True})
    assert r.status == 200
    blocks = blocks_of(body)
    # every frame is EITHER a keep-alive comment OR a whole data event —
    # the write lock's whole point is that a tick can never land inside one
    assert all(b == ": keep-alive" or b.startswith("data: ") for b in blocks)
    ka = [i for i, b in enumerate(blocks) if b == ": keep-alive"]
    assert len(ka) >= 1
    first_content = next(i for i, b in enumerate(blocks) if b.startswith("data: ")
                         and json.loads(b[len("data: "):]).get("choices", [{}])[0]
                         .get("delta", {}).get("content"))
    assert ka[0] < first_content  # the tick(s) precede the first real delta
    # and the real stream is untouched: sse_events() reconstructs it whole
    chunks = [json.loads(e) for e in sse_events(body)[:-1]]
    text = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks)
    assert text == "alpha beta"


def test_keepalive_disabled_by_zero(fake, monkeypatch):
    monkeypatch.setenv("DRINKME_SSE_KEEPALIVE_S", "0")
    _, port = fake(reply="alpha beta", prefill_delay=0.15)
    r, body = post(port, {"messages": msgs(), "stream": True})
    assert r.status == 200 and ": keep-alive" not in blocks_of(body)


def test_no_keepalive_when_prefill_is_fast(fake):
    # default interval (10s) far exceeds a FakeEngine's instant prefill
    _, port = fake(reply="alpha beta")
    r, body = post(port, {"messages": msgs(), "stream": True})
    assert ": keep-alive" not in blocks_of(body)


def test_keepalive_never_fires_for_non_streaming(fake, monkeypatch):
    monkeypatch.setenv("DRINKME_SSE_KEEPALIVE_S", "0.02")
    _, port = fake(reply="alpha beta", prefill_delay=0.15)
    r, body = post(port, {"messages": msgs()})  # no "stream": true
    assert r.status == 200
    assert b": keep-alive" not in body
    assert json.loads(body)["choices"][0]["message"]["content"] == "alpha beta"


# ---------------------------------------------- /tokenize, /detokenize --


def test_tokenize_prompt_shape(fake):
    _, port = fake()
    r, body = post(port, {"prompt": "hello world"}, path="/tokenize")
    obj = json.loads(body)
    assert r.status == 200
    assert obj["count"] == 2 and obj["tokens"] == [0, 1]
    assert obj["max_model_len"] is None  # FakeEngine publishes no ctx by default


def test_tokenize_messages_matches_count_tokens(fake):
    eng, port = fake()
    m = msgs("hello world foo")
    r, body = post(port, {"messages": m}, path="/tokenize")
    obj = json.loads(body)
    assert r.status == 200
    assert obj["count"] == len(obj["tokens"]) == eng.count_tokens(
        GenerationRequest(m, SampleParams()))


def test_tokenize_requires_exactly_one_of_prompt_or_messages(fake):
    _, port = fake()
    for bad in ({}, {"prompt": "hi", "messages": msgs()}):
        r, body = post(port, bad, path="/tokenize")
        assert r.status == 400
        assert "exactly one" in assert_error_shape(body)["message"]


def test_tokenize_normalizes_messages_like_chat_completions(fake):
    # the SAME boundary fixes /v1/chat/completions applies — parts-array
    # content and the developer->system role fold — so a client cannot see
    # two different opinions about what one conversation costs
    seen = {}

    class Eng(FakeEngine):
        def tokenize(self, prompt=None, messages=None, tools=None, template_kwargs=None):
            seen["messages"] = messages
            return super().tokenize(prompt=prompt, messages=messages, tools=tools,
                                    template_kwargs=template_kwargs)

    _, port = fake(engine=Eng())
    r, _ = post(port, {"messages": [
        {"role": "developer", "content": "be brief"},
        {"role": "user", "content": [{"type": "text", "text": "hi"}]}]}, path="/tokenize")
    assert r.status == 200
    assert seen["messages"][0]["role"] == "system"
    assert seen["messages"][1]["content"] == "hi"


def test_detokenize_round_trips_tokenize(fake):
    _, port = fake()
    _, tok_body = post(port, {"prompt": "hello world"}, path="/tokenize")
    tokens = json.loads(tok_body)["tokens"]
    r, body = post(port, {"tokens": tokens}, path="/detokenize")
    assert r.status == 200 and json.loads(body)["prompt"] == "hello world"


def test_detokenize_bad_tokens_400(fake):
    _, port = fake()
    for bad in ({}, {"tokens": "nope"}, {"tokens": [1, "x"]}, {"tokens": [1, True]}):
        r, body = post(port, bad, path="/detokenize")
        assert r.status == 400
        assert assert_error_shape(body)["type"] == "invalid_request_error"


def test_tokenizer_info_shape(fake):
    _, port = fake()
    r, body = get(port, "/tokenizer_info")
    obj = json.loads(body)
    assert r.status == 200
    assert obj["bos_token_id"] is None and obj["eos_token_id"] == 0
    assert obj["chat_template"] is True
    assert obj["thinking"] == "open" and obj["tool_format"] == "json"
    assert obj["max_model_len"] is None


def test_tokenize_and_tokenizer_info_are_lockless(fake):
    eng, port = fake(reply="w " * 60, delay=0.02)  # ~1.2s generation
    t = threading.Thread(target=post, args=(port, {"messages": msgs()}))
    t.start()
    time.sleep(0.2)  # let the generation take the lock
    t0 = time.time()
    r1, _ = post(port, {"prompt": "hi"}, path="/tokenize")
    r2, _ = get(port, "/tokenizer_info")
    assert r1.status == 200 and r2.status == 200
    assert time.time() - t0 < 1.0  # answered mid-generation
    t.join()


def test_tokenize_family_gated_when_auth_configured(fake):
    _, port = fake(auth="sekrit")
    r, _ = post(port, {"prompt": "hi"}, path="/tokenize")
    assert r.status == 401
    r, _ = post(port, {"tokens": [0]}, path="/detokenize")
    assert r.status == 401
    r, _ = get(port, "/tokenizer_info")
    assert r.status == 401


def test_unknown_route_still_404s_alongside_the_new_ones(fake):
    _, port = fake()
    r, _ = get(port, "/tokenizeXYZ")
    assert r.status == 404


# ---------------------------------------------- request-time context check --

# FakeEngine's default context_window (None) means the check no-ops — every
# test above ran with an effectively unbounded window. These set it
# explicitly. "hello world" is 2 fake tokens (one word == one token).


TWELVE = "w1 w2 w3 w4 w5 w6 w7 w8 w9 w10 w11 w12"  # a reply longer than any room below


def test_max_tokens_clamped_to_the_room_left(fake):
    """The max_tokens clamp: prompt 2 + max_tokens 20 > ctx 10, but the prompt fits — clamp
    max_tokens to the 8 tokens of room and generate, exactly what llama.cpp,
    Ollama and vLLM (get_max_tokens: min(max_model_len - prompt, requested))
    do. The client sees the length stop, not a 400."""
    eng, port = fake(context_window=10, reply=TWELVE)
    r, body = post(port, {"messages": msgs(), "max_tokens": 20})
    assert r.status == 200
    out = json.loads(body)
    assert out["choices"][0]["finish_reason"] == "length"
    assert out["usage"]["completion_tokens"] == 8
    assert eng.calls == 1


def test_max_tokens_within_room_is_untouched(fake):
    eng, port = fake(context_window=10, reply=TWELVE)
    r, body = post(port, {"messages": msgs(), "max_tokens": 5})
    assert r.status == 200
    assert json.loads(body)["usage"]["completion_tokens"] == 5


def test_context_length_refused_when_clamping_is_off(fake, monkeypatch):
    """DRINKME_MAX_TOKENS_CLAMP=0 restores the OpenAI-style refusal: the
    message opens with OpenAI's own canonical wording (a client that
    classifies overflow by matching a provider's own dialect, e.g. pi-ai's
    OVERFLOW_PATTERNS) and still carries all three numbers and `param`."""
    monkeypatch.setenv("DRINKME_MAX_TOKENS_CLAMP", "0")
    eng, port = fake(context_window=10)
    r, body = post(port, {"messages": msgs(), "max_tokens": 20})
    assert r.status == 400
    err = assert_error_shape(body)
    assert err["type"] == "invalid_request_error" and err["code"] == "context_length_exceeded"
    assert err["param"] == "messages"
    assert err["message"].startswith("This model's maximum context length is 10 tokens.")
    assert "2" in err["message"] and "20" in err["message"] and "10" in err["message"]
    assert eng.calls == 0


def test_context_length_within_window_is_fine(fake):
    _, port = fake(context_window=100)
    r, _ = post(port, {"messages": msgs(), "max_tokens": 20})
    assert r.status == 200


def test_prompt_filling_the_window_is_refused_before_generation(fake):
    """The prompt alone leaves no room: nothing to clamp, 400 whatever the
    clamp setting, and eng.generate() is never reached."""
    eng, port = fake(context_window=2)  # "hello world" is exactly 2 tokens
    r, body = post(port, {"messages": msgs(), "max_tokens": 20})
    assert r.status == 400
    err = assert_error_shape(body)
    assert err["code"] == "context_length_exceeded"
    assert "no room" in err["message"]
    assert err["message"].startswith("This model's maximum context length is 2 tokens.")
    assert eng.calls == 0


def test_prompt_filling_the_window_400_streaming(fake):
    _, port = fake(context_window=2)
    # headers (and the role chunk) are already out by the time the check
    # runs inside _generate, so streaming answers 200 and the refusal rides
    # an SSE error event instead of a status line
    r, body = post(port, {"messages": msgs(), "max_tokens": 20, "stream": True})
    assert r.status == 200
    frames = [json.loads(e) for e in sse_events(body)]
    (err_frame,) = [f for f in frames if "error" in f]
    err = err_frame["error"]
    assert err["code"] == "context_length_exceeded" and "no room" in err["message"]


def test_max_tokens_clamped_streaming(fake):
    eng, port = fake(context_window=10, reply=TWELVE)
    r, body = post(port, {"messages": msgs(), "max_tokens": 20, "stream": True})
    assert r.status == 200
    frames = [json.loads(e) for e in sse_events(body) if e.strip() != "[DONE]"]
    assert not any("error" in f for f in frames)
    finish = [f["choices"][0]["finish_reason"] for f in frames
              if f.get("choices") and f["choices"][0].get("finish_reason")]
    assert finish == ["length"]


def test_context_length_check_no_ops_when_ctx_unpublished(fake):
    _, port = fake()  # context_window=None (default)
    r, _ = post(port, {"messages": msgs(), "max_tokens": 10 ** 6})
    assert r.status == 200


def test_context_length_4xx_logs_one_line(fake, capsys):
    _, port = fake(context_window=2)  # the prompt fills the window: the 400 path
    capsys.readouterr()
    r, body = post(port, {"messages": msgs(), "max_tokens": 20})
    assert r.status == 400
    msg = assert_error_shape(body)["message"]
    err = capsys.readouterr().err
    assert "POST" in err and "400" in err and msg in err


# ------------------------------------------------------- body/auth/deadline --
# An unauthenticated client must not hold a server thread in an
# unbounded, undeadlined body read, which it could if _body() drained the
# declared body BEFORE _authed() ran. Every case below goes over a raw
# socket — real Content-Length headers, real timing.


def _recv_response(s: socket.socket, timeout: float = 5) -> bytes:
    """Headers + declared body, over however many TCP segments they land
    in — a single recv() call is not guaranteed to carry a whole small HTTP
    response."""
    s.settimeout(timeout)
    data = b""
    while b"\r\n\r\n" not in data:
        chunk = s.recv(65536)
        if not chunk:
            return data
        data += chunk
    head, _, rest = data.partition(b"\r\n\r\n")
    n = 0
    for line in head.split(b"\r\n"):
        if line.lower().startswith(b"content-length:"):
            n = int(line.split(b":", 1)[1].strip())
    while len(rest) < n:
        chunk = s.recv(65536)
        if not chunk:
            break
        rest += chunk
    return head + b"\r\n\r\n" + rest


def test_unauthenticated_huge_body_gets_401_promptly_without_reading_it(fake):
    # Content-Length: 1000000000, no token, no bytes. Auth after the body
    # read would never answer within the probe's timeout — _body() would
    # still be trying to read a billion bytes that never arrive. _authed()
    # runs off headers alone, before _body() is ever called.
    _, port = fake(auth="sekrit")
    s = socket.create_connection(("127.0.0.1", port), timeout=5)
    s.sendall(b"POST /v1/chat/completions HTTP/1.1\r\nHost: t\r\n"
              b"Content-Type: application/json\r\n"
              b"Content-Length: 1000000000\r\n\r\n")  # no Authorization; no body bytes
    t0 = time.time()
    data = _recv_response(s)
    elapsed = time.time() - t0
    assert elapsed < 2  # promptly — not "whenever the client eventually closes"
    status_line = data.split(b"\r\n", 1)[0]
    assert b"401" in status_line
    assert b"connection: close" in data.lower()
    # the connection is actually torn down: a further read gets EOF, not a
    # hang and not "the next request line" (the old drain-then-gate order
    # would have left 1_000_000_000 undeclared bytes sitting in the stream)
    s.settimeout(2)
    more = b""
    try:
        more = s.recv(4096)
    except socket.timeout:
        pass
    assert more == b""
    s.close()


def test_stalled_authenticated_body_closes_at_the_read_deadline(fake, monkeypatch):
    monkeypatch.setenv("DRINKME_READ_TIMEOUT_S", "0.2")
    _, port = fake(auth="sekrit")
    s = socket.create_connection(("127.0.0.1", port), timeout=5)
    s.sendall(b"POST /v1/chat/completions HTTP/1.1\r\nHost: t\r\n"
              b"Authorization: Bearer sekrit\r\n"
              b"Content-Type: application/json\r\n"
              b"Content-Length: 1000000\r\n\r\n")  # declares a body, sends none of it
    t0 = time.time()
    s.settimeout(5)
    data = b""
    try:
        data = s.recv(4096)
    except socket.timeout:
        pass
    elapsed = time.time() - t0
    assert data == b""  # closed with no response (BaseHTTPRequestHandler's own
                        # `except socket.timeout`: log, close_connection, return)
    assert 0.15 < elapsed < 3  # closed AT roughly the deadline — not instantly,
                               # and not "never"
    s.close()


def test_body_over_cap_413_rejects_before_reading_any_bytes(fake, monkeypatch):
    monkeypatch.setenv("DRINKME_MAX_BODY_BYTES", "1000")
    _, port = fake()
    s = socket.create_connection(("127.0.0.1", port), timeout=5)
    s.sendall(b"POST /v1/chat/completions HTTP/1.1\r\nHost: t\r\n"
              b"Content-Type: application/json\r\n"
              b"Content-Length: 999999999\r\n\r\n")  # declared only; no body sent
    t0 = time.time()
    data = _recv_response(s)
    elapsed = time.time() - t0
    assert elapsed < 2  # answered off the Content-Length header alone — no
                        # read was attempted, so a body that never arrives
                        # cannot make this hang
    status_line = data.split(b"\r\n", 1)[0]
    assert b"413" in status_line
    assert b"connection: close" in data.lower()
    obj = json.loads(data.split(b"\r\n\r\n", 1)[1])
    assert obj["error"]["code"] == "request_too_large"
    s.close()


def test_stream_completes_for_a_slow_reading_client_past_the_read_deadline(fake, monkeypatch):
    # The read deadline must be cleared BEFORE generation begins: a client
    # that reads slowly enough to stall a write on TCP backpressure must not
    # have that write mistaken for a stalled READ and killed by a stale
    # deadline armed for a phase that already finished.
    monkeypatch.setenv("DRINKME_READ_TIMEOUT_S", "0.05")
    eng, port = fake(reply="w " * 8000)  # ~1.5MB of SSE frames: far more than
                                         # a client that reads nothing for a
                                         # while lets any default kernel send
                                         # buffer absorb without blocking
    body = json.dumps({"messages": msgs(), "max_tokens": 8000, "stream": True}).encode()
    s = socket.create_connection(("127.0.0.1", port), timeout=15)
    s.sendall(b"POST /v1/chat/completions HTTP/1.1\r\nHost: t\r\n"
              b"Content-Type: application/json\r\n"
              b"Content-Length: " + str(len(body)).encode() + b"\r\n\r\n" + body)
    time.sleep(1.0)  # read NOTHING for well past the (0.05s) read deadline —
                     # forces the server to block on a write mid-stream
    data = b""
    t0 = time.time()
    while b"[DONE]" not in data and time.time() - t0 < 15:
        chunk = s.recv(65536)
        if not chunk:
            break
        data += chunk
    s.close()
    assert b"[DONE]" in data  # the stream completed — a stale deadline would
                              # have killed the write and aborted it early
    assert not eng.aborted


# --------------------------------------------------------- malformed fields --
# a handful of client-controlled types reached raw
# dict/set/list operations before the route's broad generation exception
# handler, so a malformed field produced a traceback and
# http.client.RemoteDisconnected instead of a structured 400. Each case here
# also proves the connection SURVIVES: a following valid request on the SAME
# http.client.HTTPConnection must still get 200 — a fresh connection per
# request would not catch a connection the fix failed to keep alive.


def test_malformed_fields_are_400_and_the_connection_survives(fake):
    _, port = fake()
    cases = [
        ({"model": [], "messages": msgs()}, "model"),
        ({"messages": [5]}, "messages"),
        ({"messages": [{"role": "user", "content": {}}]}, "content"),
        ({"messages": msgs(), "response_format":
          {"type": "json_schema", "json_schema": 5}}, "json_schema"),
        ({"messages": msgs(), "response_format": {
            "type": "json_schema",
            "json_schema": {"schema": {"type": "object", "properties": []}}}},
         "properties"),
        ({"messages": msgs(), "tools": 5}, "tools"),
        ({"messages": msgs(), "tool_choice": []}, "tool_choice"),
        ({"messages": msgs(), "max_tokens": "x"}, "max_tokens: must be an integer"),
        ({"messages": msgs(), "temperature": "hot"}, "temperature: must be a number"),
        ({"messages": msgs(), "stream": "yes"}, "stream"),
        ({"messages": msgs(), "stop": 5}, "stop"),
    ]
    for body, needle in cases:
        c = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
        c.request("POST", "/v1/chat/completions", json.dumps(body),
                  {"Content-Type": "application/json"})
        r = c.getresponse()
        data = r.read()
        assert r.status == 400, (body, r.status, data)
        err = assert_error_shape(data)
        assert needle in err["message"], (body, needle, err["message"])
        # the connection must still be good: a valid request on the SAME
        # connection (no new TCP handshake) must still succeed
        c.request("POST", "/v1/chat/completions", json.dumps({"messages": msgs()}),
                  {"Content-Type": "application/json"})
        r2 = c.getresponse()
        assert r2.status == 200, (body, r2.status, r2.read())
        c.close()
