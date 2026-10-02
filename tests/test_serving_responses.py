"""/v1/responses — the OpenAI Responses API over the same core, fake engine, CPU.

Every request goes over a real socket to a port-0 server, like the two
sibling files (test_serving_http.py's fixture, fake tokenizer and scripts are
reused): what passes here is what curl, the openai SDKs and Codex see. The
load-bearing tests are the three-way cross-surface parity test (in
test_serving_messages.py — the same conversation via chat completions,
Messages and Responses feeds the engine byte-identical inputs), the
stream-reassembles-to-the-non-stream-object test (`assemble` below applies
the events with the SDK accumulator's own checks: ids and indices must line
up or it raises), and the function_call_arguments concatenation identity.
"""

import http.client
import http.server
import json
import socket
import threading
import time

from drinkme.serving import responses, vision
from drinkme.serving.engine import FakeEngine, Finished
from test_serving_http import (  # noqa: F401 — `fake` is the shared fixture
    SCRIPT, THINK, THINK_TOOL, WEATHER_TOOL, VisionEngine, FakeTok, Templating, data_url, fake,
    get, relay, templating, tiny_png, vision_engine)

MODEL = "drinkme-fake"

# the Responses (flat) spelling of test_serving_http's WEATHER_TOOL
TOOL = {"type": "function", "name": "get_weather",
        "description": "Current weather for a city.",
        "parameters": {"type": "object",
                       "properties": {"city": {"type": "string"}},
                       "required": ["city"]},
        "strict": False}

TITLE_SCHEMA = {"type": "object", "properties": {"title": {"type": "string"}},
                "required": ["title"], "additionalProperties": False}


def post(port, body, path="/v1/responses", headers=None):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    payload = body if isinstance(body, (bytes, str)) else json.dumps(body)
    h = {"Content-Type": "application/json"}
    h.update(headers or {})
    c.request("POST", path, payload, h)
    r = c.getresponse()
    data = r.read()
    c.close()
    return r, data


def req(q="hello world", **kw):
    body = {"model": MODEL, "input": q}
    body.update(kw)
    return body


def events(body: bytes) -> list[tuple[str, dict]]:
    """Named SSE frames -> [(event, payload)] in wire order; keep-alive
    comments are dropped, as a conformant parser drops them."""
    out = []
    for frame in body.decode().split("\n\n"):
        if not frame.strip() or frame.startswith(":"):
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
    """Reassemble a streamed response the way the openai SDK's accumulator
    does (internal/responses/response-accumulator.js), with its checks: an
    item-scoped event's item_id must equal the item's id at output_index,
    added indices must append, done events replace, and the lifecycle
    events carry whole snapshots. Raises on anything the SDK would reject."""
    snap = None
    seqs = []
    for name, ev in evs:
        assert name == ev["type"]  # the event line names the payload type
        seqs.append(ev["sequence_number"])
        t = ev["type"]
        if t in ("response.created", "response.in_progress", "response.completed",
                 "response.incomplete", "response.failed"):
            snap = json.loads(json.dumps(ev["response"]))
            continue
        if t == "error":
            continue
        assert snap is not None
        out = snap["output"]
        if t == "response.output_item.added":
            assert ev["output_index"] == len(out), "added must append"
            out.append(json.loads(json.dumps(ev["item"])))
            continue
        item = out[ev["output_index"]]
        if t == "response.output_item.done":
            assert ev["item"]["id"] == item["id"] and ev["item"]["type"] == item["type"]
            if item["type"] == "function_call":
                assert ev["item"]["call_id"] == item["call_id"]
            out[ev["output_index"]] = ev["item"]
            continue
        assert ev["item_id"] == item["id"], (t, ev["item_id"], item["id"])
        if t == "response.content_part.added":
            assert item["type"] == "message"
            assert ev["content_index"] == len(item["content"])
            item["content"].append(dict(ev["part"]))
        elif t == "response.content_part.done":
            item["content"][ev["content_index"]] = ev["part"]
        elif t == "response.output_text.delta":
            part = item["content"][ev["content_index"]]
            assert part["type"] == "output_text"
            part["text"] += ev["delta"]
        elif t == "response.output_text.done":
            item["content"][ev["content_index"]]["text"] = ev["text"]
        elif t == "response.reasoning_summary_part.added":
            assert item["type"] == "reasoning"
            assert ev["summary_index"] == len(item["summary"])
            item["summary"].append(dict(ev["part"]))
        elif t == "response.reasoning_summary_part.done":
            item["summary"][ev["summary_index"]] = ev["part"]
        elif t == "response.reasoning_summary_text.delta":
            item["summary"][ev["summary_index"]]["text"] += ev["delta"]
        elif t == "response.reasoning_summary_text.done":
            item["summary"][ev["summary_index"]]["text"] = ev["text"]
        elif t == "response.function_call_arguments.delta":
            assert item["type"] == "function_call"
            item["arguments"] += ev["delta"]
        elif t == "response.function_call_arguments.done":
            item["arguments"] = ev["arguments"]
        else:
            raise AssertionError(f"unexpected event {t}")
    assert seqs == list(range(len(seqs)))  # monotonic from 0, no gaps
    return snap


def without_ids(obj: dict) -> dict:
    """A response with the per-request ids and clock removed, so two
    renderings of one generation compare equal."""
    o = json.loads(json.dumps(obj))
    o.pop("id"), o.pop("created_at")
    for item in o["output"]:
        item.pop("id")
        item.pop("call_id", None)
    return o


def error_of(data) -> dict:
    """The OpenAI envelope, exactly those keys; returns the inner error."""
    obj = json.loads(data)
    assert set(obj) == {"error"}
    assert set(obj["error"]) == {"message", "type", "param", "code"}
    return obj["error"]


class Recording(FakeEngine):
    """What the LAST generate() received (the internal history)."""

    def generate(self, req):
        self.seen = req.messages
        self.seen_params = req.sampling
        return super().generate(req)


# ------------------------------------------------------------- non-stream --


def test_nonstream_text(fake):
    _, port = fake()
    r, body = post(port, req())
    obj = json.loads(body)
    assert r.status == 200
    assert obj["id"].startswith("resp_") and obj["object"] == "response"
    assert obj["status"] == "completed" and obj["model"] == MODEL
    assert obj["error"] is None and obj["incomplete_details"] is None
    (msg,) = obj["output"]
    assert msg["id"].startswith("msg_") and msg["type"] == "message"
    assert msg["role"] == "assistant" and msg["status"] == "completed"
    assert msg["content"] == [{"type": "output_text", "text": "echo: hello world",
                               "annotations": []}]
    u = obj["usage"]
    assert u["input_tokens"] == 2 and u["output_tokens"] == 3  # "hello world" / "echo: hello world"
    assert u["total_tokens"] == 5
    assert u["input_tokens_details"] == {"cached_tokens": 0, "cache_write_tokens": 0}
    assert u["output_tokens_details"] == {"reasoning_tokens": 0}
    assert isinstance(obj["created_at"], int)


def test_response_echoes_the_knobs_and_says_store_false(fake):
    _, port = fake()
    obj = json.loads(post(port, req(
        instructions="be brief", max_output_tokens=50, temperature=0.3, top_p=0.8,
        tools=[TOOL], tool_choice="auto", metadata={"k": "v"}, store=True,
        parallel_tool_calls=False, truncation="auto",
        reasoning={"effort": "low"}, text={"format": {"type": "text"}}))[1])
    assert obj["instructions"] == "be brief" and obj["max_output_tokens"] == 50
    assert obj["temperature"] == 0.3 and obj["top_p"] == 0.8
    assert obj["tools"] == [TOOL] and obj["tool_choice"] == "auto"
    assert obj["metadata"] == {"k": "v"} and obj["parallel_tool_calls"] is False
    assert obj["truncation"] == "auto" and obj["reasoning"] == {"effort": "low"}
    assert obj["text"] == {"format": {"type": "text"}}
    assert obj["store"] is False  # nothing is stored, whatever was asked
    assert obj["previous_response_id"] is None
    # the defaults, when nothing was sent
    obj = json.loads(post(port, req())[1])
    assert obj["instructions"] is None and obj["max_output_tokens"] is None
    assert obj["temperature"] == 1.0 and obj["top_p"] == 1.0
    assert obj["tools"] == [] and obj["tool_choice"] == "auto"
    assert obj["metadata"] == {} and obj["parallel_tool_calls"] is True
    assert obj["truncation"] == "disabled" and obj["reasoning"] is None
    assert obj["text"] == {"format": {"type": "text"}}


def test_max_output_tokens_stops_incomplete(fake):
    _, port = fake(reply="one two three four five")
    obj = json.loads(post(port, req(max_output_tokens=3))[1])
    assert obj["status"] == "incomplete"
    assert obj["incomplete_details"] == {"reason": "max_output_tokens"}
    assert obj["output"][0]["content"][0]["text"] == "one two three "
    assert obj["usage"]["output_tokens"] == 3
    # streamed: the closing event is response.incomplete with the same object
    r, body = post(port, req(max_output_tokens=3, stream=True))
    evs = events(body)
    assert evs[-1][0] == "response.incomplete"
    assert evs[-1][1]["response"]["incomplete_details"] == {"reason": "max_output_tokens"}
    assert without_ids(assemble(evs)) == without_ids(obj)


def test_max_output_tokens_absent_is_the_default_not_required(fake):
    eng, port = fake(engine=Recording())
    r, _ = post(port, req())
    assert r.status == 200 and eng.seen_params.max_tokens == 4096


def test_instructions_become_the_system_turn_first(fake):
    eng, port = fake(engine=Recording())
    post(port, req("hi", instructions="be brief"))
    assert eng.seen == [{"role": "system", "content": "be brief"},
                        {"role": "user", "content": "hi"}]


def test_input_items_map_to_turns(fake):
    eng, port = fake(engine=Recording())
    r, _ = post(port, req(input=[
        {"type": "message", "role": "developer", "content": "be brief"},
        {"role": "user", "content": [{"type": "input_text", "text": "hello "},
                                     {"type": "input_text", "text": "world"}]},
        {"type": "message", "role": "assistant",
         "content": [{"type": "output_text", "text": "hi!", "annotations": []}],
         "id": "msg_1", "status": "completed"},
        {"type": "message", "role": "system", "content": "mid-conversation system"},
        {"role": "user", "content": "and?"}]))
    assert r.status == 200
    assert eng.seen == [
        {"role": "system", "content": "be brief"},  # developer -> system
        {"role": "user", "content": "hello world"},  # parts concatenate (chat's join)
        {"role": "assistant", "content": "hi!"},
        {"role": "system", "content": "mid-conversation system"},  # in place
        {"role": "user", "content": "and?"}]


def test_function_call_items_merge_into_one_assistant_turn(fake):
    # reasoning + message + two function_calls, as a response echoes back,
    # are ONE assistant turn — the history shape the chat boundary produces
    eng, port = fake(engine=Recording())
    post(port, req(input=[
        {"role": "user", "content": "weather in Paris and Rome?"},
        {"type": "reasoning", "id": "rs_1",
         "summary": [{"type": "summary_text", "text": "two cities"}]},
        {"type": "message", "role": "assistant", "id": "msg_1", "status": "completed",
         "content": [{"type": "output_text", "text": "Checking both.", "annotations": []}]},
        {"type": "function_call", "id": "fc_1", "call_id": "call_1", "name": "get_weather",
         "arguments": "{\"city\": \"Paris\"}", "status": "completed"},
        {"type": "function_call", "id": "fc_2", "call_id": "call_2", "name": "get_weather",
         "arguments": "{\"city\": \"Rome\"}"},
        {"type": "function_call_output", "call_id": "call_1", "output": "72F"},
        {"type": "function_call_output", "call_id": "call_2",
         "output": [{"type": "input_text", "text": "75F"}]},
        {"role": "user", "content": "thanks"}]))
    assert eng.seen == [
        {"role": "user", "content": "weather in Paris and Rome?"},
        {"role": "assistant", "content": "Checking both.", "reasoning_content": "two cities",
         "tool_calls": [
             {"id": "call_1", "type": "function",
              "function": {"name": "get_weather", "arguments": {"city": "Paris"}}},
             {"id": "call_2", "type": "function",
              "function": {"name": "get_weather", "arguments": {"city": "Rome"}}}]},
        {"role": "tool", "tool_call_id": "call_1", "content": "72F"},
        {"role": "tool", "tool_call_id": "call_2", "content": "75F"},
        {"role": "user", "content": "thanks"}]


def test_reasoning_item_merges_before_or_after_the_message(fake):
    eng, port = fake(engine=Recording())
    # after: the preceding assistant turn takes it
    post(port, req(input=[
        {"role": "user", "content": "q"},
        {"type": "message", "role": "assistant", "content": "a"},
        {"type": "reasoning", "content": [{"type": "reasoning_text", "text": "why"}],
         "summary": []},
        {"role": "user", "content": "q2"}]))
    assert eng.seen[1] == {"role": "assistant", "content": "a", "reasoning_content": "why"}
    # a second message item starts a NEW assistant turn (content already filled)
    post(port, req(input=[
        {"role": "user", "content": "q"},
        {"type": "message", "role": "assistant", "content": "a"},
        {"type": "message", "role": "assistant", "content": "b"}]))
    assert eng.seen[1:] == [{"role": "assistant", "content": "a"},
                            {"role": "assistant", "content": "b"}]
    # a function_call with no message: content "" (the chat boundary's null -> "")
    post(port, req(input=[
        {"role": "user", "content": "q"},
        {"type": "function_call", "call_id": "c1", "name": "f", "arguments": "{}"},
        {"type": "function_call_output", "call_id": "c1", "output": "ok"}]))
    assert eng.seen[1] == {"role": "assistant", "content": "", "tool_calls": [
        {"id": "c1", "type": "function", "function": {"name": "f", "arguments": {}}}]}
    # encrypted_content only: nothing readable, no reasoning_content
    post(port, req(input=[
        {"role": "user", "content": "q"},
        {"type": "reasoning", "summary": [], "encrypted_content": "gAAAA"},
        {"type": "message", "role": "assistant", "content": "a"}]))
    assert eng.seen[1] == {"role": "assistant", "content": "a"}


def test_usage_counts_cached_prompt_tokens_the_openai_way(fake):
    # input_tokens is the whole prompt; cached_tokens rides in the details
    # (the chat dialect's prompt_tokens_details rule, not Anthropic's split)
    class Cached(FakeEngine):
        def generate(self, req):
            def mark(ev):
                if isinstance(ev, Finished):
                    ev.result.cached_tokens = 1
            return relay(super().generate(req), mark)

    _, port = fake(engine=Cached())
    u = json.loads(post(port, req("hello world"))[1])["usage"]
    assert u["input_tokens"] == 2 and u["input_tokens_details"]["cached_tokens"] == 1


def test_advertised_ctx_does_not_scale_responses_usage(fake):
    # the advertised-context lie is scoped to /v1/messages (docs/serve.md)
    _, port = fake(context_window=100, advertised_ctx=50)
    u = json.loads(post(port, req("hello world"))[1])["usage"]
    assert u["input_tokens"] == 2


def test_unknown_knobs_are_accepted_and_ignored(fake):
    _, port = fake()
    r, _ = post(port, req(store=True, metadata={"a": "b"}, user="u", truncation="auto",
                          include=["reasoning.encrypted_content"],
                          stream_options={"include_obfuscation": False},
                          service_tier="default", safety_identifier="s",
                          prompt_cache_key="k", top_logprobs=2,
                          text={"verbosity": "low"}, reasoning={"summary": "auto"},
                          parallel_tool_calls=False))
    assert r.status == 200


def test_sampling_extensions_reach_the_engine_and_there_is_no_stop(fake):
    eng, port = fake(engine=Recording(reply="alpha END beta"))
    r, body = post(port, req(temperature=0.5, top_p=0.9, top_k=40, seed=7,
                             repetition_penalty=1.1, stop=["END"]))
    assert r.status == 200
    p = eng.seen_params
    assert (p.temperature, p.top_p, p.top_k, p.seed, p.repetition_penalty) == \
        (0.5, 0.9, 40, 7, 1.1)
    assert p.stop == []  # Responses has no stop; the word is not read
    assert json.loads(body)["output"][0]["content"][0]["text"] == "alpha END beta"


def test_head_answers_without_a_body(fake):
    _, port = fake()
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    c.request("HEAD", "/v1/responses")
    r = c.getresponse()
    assert r.status == 404 and r.read() == b""
    c.close()


# -------------------------------------------------------------- streaming --


def test_stream_event_order_and_assembled_equals_nonstream(fake):
    _, port = fake(reply="alpha beta gamma")
    r, body = post(port, req(stream=True))
    assert r.status == 200 and r.getheader("Content-Type") == "text/event-stream"
    evs = events(body)
    names = [n for n, _ in evs]
    assert names == ["response.created", "response.in_progress",
                     "response.output_item.added", "response.content_part.added",
                     "response.output_text.delta", "response.output_text.delta",
                     "response.output_text.delta",  # one per word-delta
                     "response.output_text.done", "response.content_part.done",
                     "response.output_item.done", "response.completed"]
    created = evs[0][1]["response"]
    assert created["id"].startswith("resp_") and created["status"] == "in_progress"
    assert created["output"] == [] and created["usage"] is None
    assert evs[1][1]["response"]["id"] == created["id"]
    added = evs[2][1]
    assert added["output_index"] == 0
    assert added["item"] == {"id": added["item"]["id"], "type": "message",
                             "role": "assistant", "status": "in_progress", "content": []}
    assert added["item"]["id"].startswith("msg_")
    part = evs[3][1]
    assert part["item_id"] == added["item"]["id"] and part["content_index"] == 0
    assert part["part"] == {"type": "output_text", "text": "", "annotations": []}
    assert evs[4][1]["delta"] == "alpha " and evs[4][1]["logprobs"] == []
    assert evs[7][1]["text"] == "alpha beta gamma"
    assert evs[9][1]["item"]["status"] == "completed"
    assert "[DONE]" not in body.decode()
    plain = json.loads(post(port, req())[1])
    streamed = assemble(evs)
    assert without_ids(streamed) == without_ids(plain)
    assert streamed["status"] == "completed" and streamed["usage"] == plain["usage"]
    # the final object's ids are the ones the item events carried
    assert streamed["output"][0]["id"] == added["item"]["id"]


def test_stream_reasoning_item_then_message_item(fake):
    _, port = fake(reply=THINK, opens_think=True)
    r, body = post(port, req(stream=True))
    evs = events(body)
    names = [n for n, _ in evs]
    assert names[2:10] == ["response.output_item.added",
                           "response.reasoning_summary_part.added",
                           "response.reasoning_summary_text.delta",
                           "response.reasoning_summary_text.delta",
                           "response.reasoning_summary_text.done",
                           "response.reasoning_summary_part.done",
                           "response.output_item.done",
                           "response.output_item.added"]
    rs = evs[2][1]["item"]
    assert rs["id"].startswith("rs_") and rs["type"] == "reasoning"
    assert rs["summary"] == [] and rs["status"] == "in_progress"
    assert evs[3][1]["summary_index"] == 0 and evs[3][1]["item_id"] == rs["id"]
    assert evs[6][1]["text"] == "weighing it"
    assert evs[8][1]["item"]["summary"] == [{"type": "summary_text", "text": "weighing it"}]
    assert evs[9][1]["output_index"] == 1 and evs[9][1]["item"]["type"] == "message"
    obj = assemble(evs)
    assert [i["type"] for i in obj["output"]] == ["reasoning", "message"]
    assert obj["output"][1]["content"][0]["text"] == "The answer."
    plain = json.loads(post(port, req())[1])
    assert without_ids(obj) == without_ids(plain)


def test_stream_function_call_items_and_arguments_identity(fake):
    _, port = fake(tool_call_script=THINK_TOOL, opens_think=True)
    r, body = post(port, req(stream=True, tools=[TOOL]))
    evs = events(body)
    names = [n for n, _ in evs]
    # reasoning, message, then the call — three items, indices 0 1 2
    added = [d for n, d in evs if n == "response.output_item.added"]
    assert [a["item"]["type"] for a in added] == ["reasoning", "message", "function_call"]
    assert [a["output_index"] for a in added] == [0, 1, 2]
    fc = added[2]["item"]
    assert fc["id"].startswith("fc_") and fc["call_id"].startswith("call_")
    assert fc["name"] == "get_weather" and fc["arguments"] == ""
    assert fc["status"] == "in_progress"
    i = names.index("response.function_call_arguments.delta")
    assert names[i:i + 4] == ["response.function_call_arguments.delta",
                              "response.function_call_arguments.done",
                              "response.output_item.done", "response.completed"]
    deltas = [d["delta"] for n, d in evs if n == "response.function_call_arguments.delta"]
    done = evs[i + 1][1]
    assert done["item_id"] == fc["id"] and done["output_index"] == 2
    assert "".join(deltas) == done["arguments"]  # concatenation identity
    assert json.loads(done["arguments"]) == {"city": "Paris"}
    final_item = evs[i + 2][1]["item"]
    assert final_item == dict(fc, arguments=done["arguments"], status="completed")
    obj = assemble(evs)
    assert obj["output"][2] == final_item
    # no tool-call bytes ever streamed as text
    text = "".join(d["delta"] for n, d in evs if n == "response.output_text.delta")
    assert text == "I will check.\n"
    plain = json.loads(post(port, req(tools=[TOOL]))[1])
    assert without_ids(obj) == without_ids(plain)


def test_stream_two_calls_indexed_in_order(fake):
    two = ('<tool_call>\n{"name": "a", "arguments": {}}\n</tool_call>\n'
           '<tool_call>\n{"name": "b", "arguments": {"x": 1}}\n</tool_call>')
    _, port = fake(tool_call_script=two)
    obj = assemble(events(post(port, req(stream=True, tools=[TOOL]))[1]))
    calls = [i for i in obj["output"] if i["type"] == "function_call"]
    assert [i["name"] for i in calls] == ["a", "b"]
    assert [json.loads(i["arguments"]) for i in calls] == [{}, {"x": 1}]
    assert len({i["call_id"] for i in calls}) == 2
    assert obj["output"][-2:] == calls  # calls come last, in emission order


def test_truncated_mid_reasoning_is_a_reasoning_item_and_incomplete(fake):
    _, port = fake(reply="a b c d e f g h", opens_think=True)
    obj = assemble(events(post(port, req(max_output_tokens=3, stream=True))[1]))
    assert obj["status"] == "incomplete"
    (rs,) = obj["output"]
    assert rs["type"] == "reasoning" and rs["summary"][0]["text"] == "a b c "


def test_disconnect_mid_stream_aborts_and_releases_the_lock(fake):
    eng, port = fake(reply="w " * 500, delay=0.01)  # ~5s if allowed to run out
    s = socket.create_connection(("127.0.0.1", port), timeout=5)
    body = json.dumps(req(stream=True)).encode()
    s.sendall(b"POST /v1/responses HTTP/1.1\r\nHost: t\r\n"
              b"Content-Type: application/json\r\n"
              b"Content-Length: " + str(len(body)).encode() + b"\r\n\r\n" + body)
    s.recv(1024)  # stream is live (headers + response.created)
    s.close()  # walk away mid-generation
    t0 = time.time()
    while time.time() - t0 < 5 and not eng.aborted:
        time.sleep(0.02)
    assert eng.aborted  # on_delta returned False and the engine stopped
    assert time.time() - t0 < 3  # promptly
    eng.reply, eng.delay = "ok", 0.0
    t0 = time.time()
    r, data = post(port, req())
    assert r.status == 200 and time.time() - t0 < 2
    assert json.loads(data)["output"][0]["content"][0]["text"] == "ok"


def test_disconnect_logs_a_distinct_line(fake, capsys):
    eng, port = fake(reply="w " * 500, delay=0.01)
    s = socket.create_connection(("127.0.0.1", port), timeout=5)
    body = json.dumps(req(stream=True)).encode()
    s.sendall(b"POST /v1/responses HTTP/1.1\r\nHost: t\r\n"
              b"Content-Type: application/json\r\n"
              b"Content-Length: " + str(len(body)).encode() + b"\r\n\r\n" + body)
    s.recv(1024)
    s.close()
    t0 = time.time()
    log = ""
    while time.time() - t0 < 5 and "stream aborted" not in log:
        log += capsys.readouterr().err
        time.sleep(0.02)
    assert "stream aborted (responses):" in log
    assert "tokens in" in log


def test_keepalive_ticks_during_prefill_then_stops(fake, monkeypatch):
    monkeypatch.setenv("DRINKME_SSE_KEEPALIVE_S", "0.03")
    _, port = fake(reply="alpha beta", prefill_delay=0.25)
    r, body = post(port, req(stream=True))
    assert r.status == 200
    blocks = [b for b in body.decode().split("\n\n") if b]
    assert all(b == ": keep-alive" or b.startswith("event: ") for b in blocks)
    ticks = [i for i, b in enumerate(blocks) if b == ": keep-alive"]
    assert len(ticks) >= 3  # 0.25s of prefill at 0.03s
    first_delta = next(i for i, b in enumerate(blocks)
                       if b.startswith("event: response.output_text.delta"))
    assert all(i < first_delta for i in ticks)  # none after the first real delta
    assert assemble(events(body))["output"][0]["content"][0]["text"] == "alpha beta"


# ------------------------------------------------------------------ tools --


def test_function_tool_reaches_the_engine_in_the_openai_shape(fake):
    eng, port = fake()
    r, _ = post(port, req(tools=[TOOL]))
    assert r.status == 200
    assert eng.last_tools == [WEATHER_TOOL]  # flat -> {"type","function":{...}}; strict dropped


def test_a_tool_without_description_or_parameters_reaches_the_engine_with_the_defaults(fake):
    # llama.cpp's defaults (common/chat.cpp): the template sees "" and {}
    eng, port = fake()
    r, body = post(port, req(tools=[{"type": "function", "name": "ping"}]))
    assert r.status == 200
    assert eng.last_tools == [{"type": "function", "function": {
        "name": "ping", "description": "", "parameters": {}}}]
    assert json.loads(body)["tools"] == [{"type": "function", "name": "ping"}]  # echoed as sent
    r, _ = post(port, req(tools=[{"type": "function", "name": "ping", "description": None,
                                  "parameters": None}]))
    assert r.status == 200
    assert eng.last_tools == [{"type": "function", "function": {
        "name": "ping", "description": "", "parameters": {}}}]


def test_nonstream_function_call_shape(fake):
    _, port = fake(tool_call_script=SCRIPT)
    obj = json.loads(post(port, req(tools=[TOOL]))[1])
    assert obj["status"] == "completed"
    msg, fc = obj["output"]
    assert msg["type"] == "message" and msg["content"][0]["text"] == "I will check.\n"
    assert fc["type"] == "function_call" and fc["status"] == "completed"
    assert fc["id"].startswith("fc_") and fc["call_id"].startswith("call_")
    assert fc["name"] == "get_weather"
    assert isinstance(fc["arguments"], str) and json.loads(fc["arguments"]) == {"city": "Paris"}


def test_function_call_output_round_trip(fake):
    # turn 1: the model calls; turn 2: the client sends the call + its output
    # back exactly as it received them, and the engine sees the tool turn
    eng, port = fake(engine=Recording(tool_call_script=SCRIPT))
    obj = json.loads(post(port, req("weather in Paris?", tools=[TOOL]))[1])
    eng.tool_call_script = None
    eng.reply = "It is 72F in Paris."
    r, body = post(port, req(input=[{"role": "user", "content": "weather in Paris?"},
                                    *obj["output"],
                                    {"type": "function_call_output",
                                     "call_id": obj["output"][1]["call_id"],
                                     "output": "72F"}],
                             tools=[TOOL]))
    assert r.status == 200
    assert eng.seen == [
        {"role": "user", "content": "weather in Paris?"},
        {"role": "assistant", "content": "I will check.\n", "tool_calls": [
            {"id": obj["output"][1]["call_id"], "type": "function",
             "function": {"name": "get_weather", "arguments": {"city": "Paris"}}}]},
        {"role": "tool", "tool_call_id": obj["output"][1]["call_id"], "content": "72F"}]
    assert json.loads(body)["output"][0]["content"][0]["text"] == "It is 72F in Paris."


def test_tool_choice_auto_and_none(fake):
    eng, port = fake()
    r, _ = post(port, req(tools=[TOOL], tool_choice="auto"))
    assert r.status == 200 and eng.last_tools == [WEATHER_TOOL]
    r, body = post(port, req(tools=[TOOL], tool_choice="none"))
    assert r.status == 200 and eng.last_tools is None
    assert json.loads(body)["tool_choice"] == "none"  # echoed as sent


def test_tool_choice_forced_is_refused_legibly(fake):
    _, port = fake()
    r, body = post(port, req(tools=[TOOL], tool_choice="required"))
    assert r.status == 400
    assert "required" in error_of(body)["message"]
    assert "will not pretend" in error_of(body)["message"]
    r, body = post(port, req(tools=[TOOL], tool_choice={"type": "function",
                                                        "name": "get_weather"}))
    assert r.status == 400 and "get_weather" in error_of(body)["message"]
    r, body = post(port, req(tools=[TOOL], tool_choice={"type": "file_search"}))
    assert r.status == 400 and "file_search" in error_of(body)["message"]


def test_no_tools_means_tag_text_passes_through(fake):
    _, port = fake(tool_call_script=SCRIPT)
    obj = json.loads(post(port, req())[1])
    (msg,) = obj["output"]
    assert msg["type"] == "message" and msg["content"][0]["text"] == SCRIPT


def test_hosted_tools_are_refused_by_name(fake):
    _, port = fake()
    for t in ("web_search", "file_search", "code_interpreter", "mcp", "computer_use_preview",
              "image_generation", "local_shell", "custom"):
        r, body = post(port, req(tools=[TOOL, {"type": t, "name": "x"}]))
        assert r.status == 400, t
        assert f"tools.1: type '{t}'" in error_of(body)["message"]
    r, body = post(port, req(tools=[{"type": "retrieval"}]))
    assert r.status == 400 and "retrieval" in error_of(body)["message"]
    r, body = post(port, req(tools=[{"type": "function", "name": ""}]))
    assert r.status == 400 and "tools.0.name" in error_of(body)["message"]
    r, body = post(port, req(tools=[{"type": "function", "name": "f", "parameters": []}]))
    assert r.status == 400 and "tools.0.parameters" in error_of(body)["message"]


def test_tools_refused_when_the_dialect_is_untested_exactly_as_chat(fake):
    # the same capability gate (serving/capability.py) as /v1/chat/completions:
    # same status, same code, same message, before a token generates
    for tool_format in ("unknown", "none", "glm"):
        eng, port = fake(tool_format=tool_format)
        r, body = post(port, req(tools=[TOOL]))
        c = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
        c.request("POST", "/v1/chat/completions",
                  json.dumps({"messages": [{"role": "user", "content": "hello world"}],
                              "tools": [WEATHER_TOOL]}),
                  {"Content-Type": "application/json"})
        rc = c.getresponse()
        chat = json.loads(rc.read())
        c.close()
        assert r.status == rc.status == 400
        assert error_of(body) == chat["error"]
        assert error_of(body)["code"] == "unsupported_tool_format"
        assert "drinkme-fake" in error_of(body)["message"]
        assert eng.calls == 0
        # tool_choice "none" withholds the tools, so nothing to refuse
        assert post(port, req(tools=[TOOL], tool_choice="none"))[0].status == 200


# ---------------------------------------------------- reasoning / text.format --


def test_reasoning_effort_reaches_the_template(fake):
    eng, port = fake(engine=templating(reply="ok"))
    r, _ = post(port, req(reasoning={"effort": "low", "summary": "auto"}))
    assert r.status == 200 and eng.tok.calls[-1]["reasoning_effort"] == "low"
    post(port, req())
    assert "reasoning_effort" not in eng.tok.calls[-1]
    r, body = post(port, req(reasoning={"effort": 3}))
    assert r.status == 400 and "reasoning.effort" in error_of(body)["message"]


def test_chat_template_kwargs_extension_same_rules_as_chat(fake):
    eng, port = fake(engine=templating(reply=THINK))
    r, body = post(port, req(chat_template_kwargs={"enable_thinking": False}))
    assert r.status == 200 and eng.tok.calls[-1]["enable_thinking"] is False
    assert [i["type"] for i in json.loads(body)["output"]] == ["message"]  # nothing thought
    r, body = post(port, req(chat_template_kwargs={"tools": []}))
    assert r.status == 400 and "the server owns those" in error_of(body)["message"]


def test_text_format_json_schema_reaches_the_engine_validated(fake):
    class Capture(FakeEngine):
        def generate(self, req):
            self.rf = req.sampling.output_schema
            self.validated = req.sampling.output_schema_validated
            return super().generate(req)

    eng, port = fake(engine=Capture(reply='{"title": "ok"}'))
    r, body = post(port, req(text={"format": {"type": "json_schema", "name": "title",
                                              "schema": TITLE_SCHEMA, "strict": True}}))
    assert r.status == 200 and eng.rf == TITLE_SCHEMA and eng.validated is True
    assert json.loads(json.loads(body)["output"][0]["content"][0]["text"]) == {"title": "ok"}
    r, _ = post(port, req(text={"format": {"type": "json_object"}}))
    assert r.status == 200 and eng.rf == {"type": "object"}
    r, _ = post(port, req(text={"format": {"type": "text"}}))
    assert r.status == 200 and eng.rf is None


def test_text_format_forces_thinking_off_and_refuses_tools(fake):
    eng, port = fake(engine=templating(reply='{"title": "x"}'))
    _, _ = post(port, req(text={"format": {"type": "json_schema", "name": "t",
                                           "schema": TITLE_SCHEMA}}))
    assert eng.tok.calls[-1]["enable_thinking"] is False
    r, body = post(port, req(tools=[TOOL], text={"format": {"type": "json_schema", "name": "t",
                                                             "schema": TITLE_SCHEMA}}))
    assert r.status == 400 and "cannot be combined with tools" in error_of(body)["message"]


def test_text_format_errors_name_the_variant(fake):
    _, port = fake()
    r, body = post(port, req(text={"format": {"type": "grammar"}}))
    assert r.status == 400 and "'grammar'" in error_of(body)["message"]
    r, body = post(port, req(text={"format": {"type": "json_schema", "name": "t"}}))
    assert r.status == 400 and "text.format.schema" in error_of(body)["message"]
    r, body = post(port, req(text={"format": {"type": "json_schema", "name": "t",
                                              "schema": {"type": "object",
                                                         "patternProperties": {}}}}))
    assert r.status == 400 and "patternProperties" in error_of(body)["message"]


# ----------------------------------------------------------------- errors --


def test_state_and_server_side_objects_are_refused_legibly(fake):
    eng, port = fake()
    r, body = post(port, req(previous_response_id="resp_123"))
    assert r.status == 400
    assert "keeps no response state" in error_of(body)["message"]
    assert "send the full input" in error_of(body)["message"]
    r, body = post(port, req(conversation="conv_1"))
    assert r.status == 400 and "conversation" in error_of(body)["message"]
    r, body = post(port, req(prompt={"id": "pmpt_1"}))
    assert r.status == 400 and "prompt" in error_of(body)["message"]
    r, body = post(port, req(background=True))
    assert r.status == 400 and "background" in error_of(body)["message"]
    r, body = post(port, req(input=[{"type": "item_reference", "id": "msg_1"}]))
    assert r.status == 400 and "item_reference" in error_of(body)["message"]
    assert eng.calls == 0


def test_image_and_file_parts_are_refused_naming_the_index(fake):
    _, port = fake()
    for part in ({"type": "input_image", "image_url": "data:..."},
                 {"type": "input_file", "file_id": "f"},
                 {"type": "input_audio", "input_audio": {}}):
        r, body = post(port, req(input=[
            {"role": "user", "content": "hi"},
            {"role": "user", "content": [{"type": "input_text", "text": "x"}, part]}]))
        assert r.status == 400
        msg = error_of(body)["message"]
        if part["type"] == "input_image":
            assert msg.startswith("input.1.content.1: model 'drinkme-fake' cannot read images")
        else:
            assert msg == f"input.1.content.1: '{part['type']}' parts are not supported by this server."
    r, body = post(port, req(input=[
        {"type": "function_call_output", "call_id": "c",
         "output": [{"type": "input_image", "image_url": "x"}]}]))
    assert r.status == 400 and "input.0.output.0" in error_of(body)["message"]


# ------------------------------------------------------------------ vision --
# A vision-capable engine accepts `input_image` data URLs in a user message,
# refuses `file_id` (there is no Files API) and images outside a user turn,
# by name — the no-vision wording above
# (`test_image_and_file_parts_are_refused_naming_the_index`) is unaffected.


class _Recording(VisionEngine):
    def generate(self, req):
        self.seen_images = req.images
        return super().generate(req)


def test_input_image_data_url_is_accepted_in_a_user_message(fake):
    eng, port = fake(engine=_Recording())
    r, body = post(port, req(input=[
        {"role": "user", "content": [
            {"type": "input_text", "text": "what is this"},
            {"type": "input_image", "image_url": data_url(tiny_png(1))}]}]))
    assert r.status == 200, body
    assert len(eng.seen_images) == 1
    obj = json.loads(body)
    n = eng.seen_images[0].tokens
    assert obj["usage"]["input_tokens"] == 3 + n  # "what is this" -- 3 words


def test_file_id_images_are_refused_no_files_api(fake):
    _, port = fake(engine=VisionEngine())
    r, body = post(port, req(input=[
        {"role": "user", "content": [{"type": "input_image", "file_id": "f_1"}]}]))
    assert r.status == 400
    msg = error_of(body)["message"]
    assert "no Files API" in msg and "input.0.content.0" in msg


class _ImgHandler(http.server.BaseHTTPRequestHandler):
    """One PNG at any path — the "image_url is fetched by default" test's server."""

    def log_message(self, fmt, *a):
        pass

    def do_GET(self):
        body = tiny_png(97)
        self.send_response(200)
        self.send_header("Content-Type", "image/png")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def test_an_http_image_url_is_fetched_by_default(fake):
    # llama.cpp parity: the download itself (timeout, oversize-abort,
    # redirects, concurrency) is test_serving_vision_urls.py's job; this is
    # the Responses dialect's wiring.
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _ImgHandler)
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        eng, port = fake(engine=_Recording())
        url = f"http://127.0.0.1:{srv.server_address[1]}/pic.png"
        r, body = post(port, req(input=[
            {"role": "user", "content": [
                {"type": "input_text", "text": "what is this"},
                {"type": "input_image", "image_url": url}]}]))
        assert r.status == 200, body
        assert len(eng.seen_images) == 1
    finally:
        srv.shutdown()


def test_http_image_urls_are_refused_when_fetching_is_off(fake):
    _, port = fake(engine=VisionEngine(veng=vision_engine(fetch_urls=False)))
    r, body = post(port, req(input=[
        {"role": "user", "content": [
            {"type": "input_image", "image_url": "https://example.com/x.png"}]}]))
    assert r.status == 400
    msg = error_of(body)["message"]
    assert "base64 data URL" in msg  # says what IS accepted, not a toggle


def test_images_are_refused_outside_a_user_message_even_with_vision(fake):
    _, port = fake(engine=VisionEngine())
    r, body = post(port, req(input=[
        {"role": "system", "content": [
            {"type": "input_image", "image_url": data_url(tiny_png(1))}]}]))
    assert r.status == 400
    assert error_of(body)["message"] == ("input.0.content.0: images are only supported "
                                         "in a user message.")


def test_the_injection_guard_refuses_the_placeholder_literal(fake):
    eng, port = fake(engine=VisionEngine())
    r, body = post(port, req(input=[
        {"role": "user", "content": f"say {vision.QwenVLPreprocessor.reserved_text[0]} back"}]))
    assert r.status == 400
    assert vision.QwenVLPreprocessor.reserved_text[0] in error_of(body)["message"]
    assert eng.calls == 0


def test_a_non_fetch_image_error_code_still_names_the_part(fake):
    _, port = fake(engine=VisionEngine())
    r, body = post(port, req(input=[
        {"role": "user", "content": [
            {"type": "input_image", "image_url": "data:image/png;base64,not-base64!!"}]}]))
    assert r.status == 400
    assert "input.0.content.0" in error_of(body)["message"]


def test_unsupported_item_types_are_refused_by_name(fake):
    _, port = fake()
    for t in ("web_search_call", "computer_call", "mcp_call", "local_shell_call",
              "custom_tool_call", "shell_call", "apply_patch_call", "compaction"):
        r, body = post(port, req(input=[{"role": "user", "content": "hi"},
                                        {"type": t, "id": "x"}]))
        assert r.status == 400, t
        assert f"input.1: item type '{t}'" in error_of(body)["message"]


def test_invalid_request_errors_name_the_field(fake):
    _, port = fake()
    cases = [
        (b"not json", "not valid JSON"),
        ([1, 2], "must be a JSON object"),
        ({"model": MODEL}, "input: field required"),
        ({"model": MODEL, "input": []}, "input: field required"),
        ({"model": MODEL, "input": 5}, "input: field required"),
        ({"model": 5, "input": "x"}, "model: must be a string"),
        (req(instructions=["a"]), "instructions: must be a string"),
        (req(max_output_tokens=0), "max_output_tokens: must be a positive integer"),
        (req(max_output_tokens=True), "max_output_tokens: must be an integer, not a boolean"),
        (req(temperature="hot"), "temperature: must be a number"),
        (req(input=[5]), "input.0: must be an object"),
        (req(input=[{"role": "tool", "content": "x"}]), "input.0.role"),
        (req(input=[{"type": "message", "role": "user", "content": 5}]),
         "input.0.content: must be a string"),
        (req(input=[{"type": "message", "role": "user", "content": [5]}]),
         "input.0.content.0: must be a content part"),
        (req(input=[{"type": "message", "role": "user",
                     "content": [{"type": "input_text", "text": 5}]}]),
         "input.0.content.0.text"),
        (req(input=[{"type": "message", "role": "user",
                     "content": [{"type": "bogus"}]}]), "input.0.content.0: unknown"),
        (req(input=[{"type": "function_call", "name": "f", "arguments": "{}"}]),
         "input.0.call_id: field required"),
        (req(input=[{"type": "function_call", "call_id": "c", "arguments": "{}"}]),
         "input.0.name: field required"),
        (req(input=[{"type": "function_call", "call_id": "c", "name": "f",
                     "arguments": 5}]), "input.0.arguments"),
        (req(input=[{"type": "function_call_output", "output": "x"}]),
         "input.0.call_id: field required"),
        (req(input=[{"type": "function_call_output", "call_id": "c", "output": 5}]),
         "input.0.output: must be a string"),
        (req(input=[{"type": "reasoning", "summary": "x"}]), "input.0.summary: must be an array"),
        (req(input=[{"type": "reasoning", "summary": [{"type": "summary_text"}]}]),
         "input.0.summary.0"),
        (req(tools="get_weather"), "tools: must be an array"),
        (req(tools=[5]), "tools.0: must be an object"),
        (req(tools=[{"type": "function", "name": "f", "description": 5}]),
         "tools.0.description"),
        (req(tool_choice=5), "tool_choice"),
        (req(reasoning="low"), "reasoning: must be an object"),
        (req(text="json"), "text: must be an object"),
        (req(text={"format": "json"}), "text.format: must be an object"),
        (req(text={"format": {"type": "json_schema", "schema": {
            "type": "object", "properties": []}}}), "properties"),
        (req(chat_template_kwargs=[]), "chat_template_kwargs: must be an object"),
        (req(stream="yes"), "stream"),
    ]
    for body, needle in cases:
        r, data = post(port, body)
        assert r.status == 400, (body, r.status, data)
        err = error_of(data)
        assert err["type"] == "invalid_request_error"
        assert needle in err["message"], (needle, err["message"])


def test_malformed_fields_keep_the_connection_alive(fake):
    # a malformed field must never leave the connection in
    # a state a following valid request can't use — the ORIGINAL
    # reproduction (the OpenAI chat-completions boundary) was
    # http.client.RemoteDisconnected. A fresh connection per case (like the
    # table above) would not catch that; this reuses ONE connection across a
    # bad request and a following good one.
    _, port = fake()
    for body in (req(model=5),
                 req(input=[5]),
                 req(tools=[5]),
                 req(tool_choice=5),
                 req(text={"format": {"type": "json_schema", "schema": {
                     "type": "object", "properties": []}}}),
                 req(stream="yes")):
        c = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
        c.request("POST", "/v1/responses", json.dumps(body), {"Content-Type": "application/json"})
        r = c.getresponse()
        r.read()
        assert r.status == 400, body
        c.request("POST", "/v1/responses", json.dumps(req()), {"Content-Type": "application/json"})
        r2 = c.getresponse()
        assert r2.status == 200, (body, r2.read())
        c.close()


def test_not_found_errors(fake):
    _, port = fake()
    r, data = post(port, req(model="nope"))
    assert r.status == 404
    err = error_of(data)
    assert err["code"] == "model_not_found" and "nope" in err["message"]
    r, data = post(port, req(), path="/v1/responses/resp_1")
    assert r.status == 404 and "Unknown request URL" in error_of(data)["message"]
    r, data = get(port, "/v1/responses")
    assert r.status == 404


def test_model_absent_is_the_served_model(fake):
    _, port = fake()
    r, body = post(port, {"input": "hi"})
    assert r.status == 200 and json.loads(body)["model"] == MODEL


def test_aliases_and_profiles_resolve(fake):
    from drinkme.serving.http import start_server

    eng = Recording()
    srv = start_server(eng, "127.0.0.1", 0, served_names=["gpt-4o"],
                       generation_profiles={"fast": {"temperature": 0.1, "enable_thinking": False}})
    try:
        port = srv.server_address[1]
        r, body = post(port, req(model="gpt-4o"))
        assert r.status == 200 and json.loads(body)["model"] == MODEL
        r, body = post(port, req(model="gpt-4o:fast"))
        assert r.status == 200 and eng.seen_params.temperature == 0.1
        r, body = post(port, req(model="gpt-4o:slow"))
        assert r.status == 404 and error_of(body)["code"] == "model_not_found"
    finally:
        srv.shutdown()


def test_auth_gate_openai_shape(fake):
    _, port = fake(auth="secret")
    r, data = post(port, req())
    assert r.status == 401 and error_of(data)["code"] == "invalid_api_key"
    r, _ = post(port, req(), headers={"Authorization": "Bearer secret"})
    assert r.status == 200


def test_4xx_logs_one_line_with_the_reason(fake, capsys):
    _, port = fake()
    capsys.readouterr()
    r, data = post(port, req(previous_response_id="resp_1"))
    assert r.status == 400
    msg = error_of(data)["message"]
    err = capsys.readouterr().err
    assert "POST" in err and "/v1/responses" in err and "400" in err and msg in err


def test_engine_failure_is_server_error_and_stream_failed(fake):
    class Boom(FakeEngine):
        def generate(self, req):
            raise RuntimeError("boom")

    _, port = fake(engine=Boom())
    r, data = post(port, req())
    assert r.status == 500
    err = error_of(data)
    assert err["type"] == "server_error" and "boom" in err["message"]
    # mid-stream: headers are out, so the news is the `error` event, then
    # response.failed carrying the object with its error filled in
    r, data = post(port, req(stream=True))
    assert r.status == 200
    evs = events(data)
    names = [n for n, _ in evs]
    assert names == ["response.created", "response.in_progress", "error", "response.failed"]
    err_ev = evs[2][1]
    assert err_ev == {"type": "error", "code": None, "message": "RuntimeError: boom",
                      "param": None, "sequence_number": 2}
    failed = evs[3][1]["response"]
    assert failed["status"] == "failed" and failed["id"] == evs[0][1]["response"]["id"]
    assert failed["error"] == {"code": "server_error", "message": "RuntimeError: boom"}
    assert assemble(evs)["status"] == "failed"


def test_template_refusal_is_a_400_naming_the_template_message(fake):
    import jinja2

    class Picky(FakeTok):
        def apply_chat_template(self, messages, add_generation_prompt=False,
                                tokenize=True, **kw):
            if kw.get("reasoning_effort") not in (None, "xhigh", "medium", "low"):
                raise jinja2.exceptions.TemplateError(
                    f"Unexpected reasoning effort {kw['reasoning_effort']}. "
                    "Supported types are xhigh (default), medium, and low.")
            return super().apply_chat_template(messages, add_generation_prompt, tokenize, **kw)

    eng = Templating()
    eng.tok, eng.engine_kwargs = Picky(), None
    _, port = fake(engine=eng)
    assert post(port, req(reasoning={"effort": "xhigh"}))[0].status == 200
    r, data = post(port, req(reasoning={"effort": "max"}))
    assert r.status == 400
    err = error_of(data)
    assert err["type"] == "invalid_request_error"
    assert "Unexpected reasoning effort max" in err["message"]
    r, data = post(port, req(reasoning={"effort": "max"}, stream=True))
    assert r.status == 200
    evs = events(data)
    assert [n for n, _ in evs][-2:] == ["error", "response.failed"]
    assert "Unexpected reasoning effort max" in evs[-2][1]["message"]


def test_context_length_check_and_clamp_shared_with_the_other_dialects(fake):
    eng, port = fake(context_window=2)  # "hello world" fills the window
    r, body = post(port, req(max_output_tokens=20))
    assert r.status == 400
    err = error_of(body)
    assert err["code"] == "context_length_exceeded" and "no room" in err["message"]
    assert eng.calls == 0
    r, body = post(port, req(max_output_tokens=20, stream=True))
    assert r.status == 200  # response.created already went out
    evs = events(body)
    assert [n for n, _ in evs][-2:] == ["error", "response.failed"]
    assert evs[-2][1]["code"] == "context_length_exceeded"
    twelve = "w1 w2 w3 w4 w5 w6 w7 w8 w9 w10 w11 w12"
    eng, port = fake(context_window=10, reply=twelve)
    obj = json.loads(post(port, req(max_output_tokens=20))[1])
    assert obj["status"] == "incomplete" and obj["usage"]["output_tokens"] == 8


def test_model_asleep_is_503_in_the_openai_envelope(fake):
    _, port = fake()
    r, _ = post(port, {}, path="/sleep")
    assert r.status == 200
    r, body = post(port, req())
    assert r.status == 503 and error_of(body)["code"] == "model_asleep"
    r, body = post(port, req(stream=True))
    evs = events(body)
    assert [n for n, _ in evs][-2:] == ["error", "response.failed"]
    assert evs[-2][1]["code"] == "model_asleep"


def test_metrics_route_label_is_the_known_route(fake):
    _, port = fake()
    post(port, req())
    _, body = get(port, "/metrics")
    assert 'route="/v1/responses"' in body.decode()


# ------------------------------------------------------------- pure units --


def test_convert_input_and_tools_are_pure():
    msgs, images = responses.convert_input("sys", [{"role": "user", "content": "hi"}])
    assert msgs == [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}]
    assert images == ()
    assert responses.convert_input(None, "hi") == ([{"role": "user", "content": "hi"}], ())
    assert responses.convert_input("", "hi") == ([{"role": "user", "content": "hi"}], ())
    assert responses.convert_tools(None) == (None, {})
    assert responses.convert_tools([]) == (None, {})
    assert responses.convert_tools([TOOL]) == ([WEATHER_TOOL], {})
    assert responses.status_of("length") == ("incomplete", {"reason": "max_output_tokens"})
    assert responses.status_of("stop") == ("completed", None)
    assert responses.status_of("tool_calls") == ("completed", None)
    assert responses.output_items("", "", None) == []
    items = responses.output_items("r", "c", [{"name": "f", "arguments": {"a": 1}}])
    assert [i["type"] for i in items] == ["reasoning", "message", "function_call"]
    assert items[2]["arguments"] == '{"a": 1}'


def test_namespace_tools_flatten_and_the_call_carries_the_namespace(fake):
    """Codex CLI sends a `namespace` tool (a container of function tools). The
    functions are offered flat under their own names and the function_call
    item that comes back carries `namespace`; a name offered twice is refused
    because the model sees one flat name."""
    import pytest
    from drinkme.serving import responses as R
    ns = {"type": "namespace", "name": "crm", "description": "customer tools",
          "tools": [{"type": "function", "name": "lookup",
                     "parameters": {"type": "object", "properties": {"id": {"type": "string"}}}}]}
    tools, names = R.convert_tools([TOOL, ns])
    assert [t["function"]["name"] for t in tools] == [TOOL["name"], "lookup"]
    assert names == {"lookup": "crm"}
    items = R.output_items("", "", [{"name": "lookup", "arguments": {"id": "7"}}], names)
    assert items[0]["type"] == "function_call" and items[0]["namespace"] == "crm"
    items = R.output_items("", "", [{"name": TOOL["name"], "arguments": {}}], names)
    assert "namespace" not in items[0]
    # duplicates: same function in two namespaces, or a namespace and the top level
    dup = dict(ns, name="other")
    with pytest.raises(R.ResponsesError) as e:
        R.convert_tools([ns, dup])
    assert "offered twice" in str(e.value)
    with pytest.raises(R.ResponsesError):
        R.convert_tools([{"type": "function", "name": "lookup"}, ns])
    # a non-function inside a namespace is refused by position
    with pytest.raises(R.ResponsesError) as e:
        R.convert_tools([{"type": "namespace", "name": "n", "tools": [{"type": "custom", "name": "x"}]}])
    assert "tools.0.tools.0" in str(e.value)
    # and over the wire the request is accepted
    _, port = fake()
    r, body = post(port, req(tools=[TOOL, ns]))
    assert r.status == 200, body
