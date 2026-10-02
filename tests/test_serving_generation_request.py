"""One GenerationRequest at the Engine seam, nothing
ambient. CPU only, no downloads.

The thinking-kwargs gotcha as a regression test, on both template families. Qwen3.8's
template opens a `<think>` block at the end of every generation prompt and
re-renders assistant history with an empty pair unless reasoning_content
comes back; Qwen3-8B is the mirror image — its default prompt opens
nothing, and enable_thinking=False pre-fills the closed pair. The template
kwargs are IN the request and the prompt's verdict is IN the first event
(nothing rides a thread-local), so a direct caller and the wire produce the
same ids by construction — which is what these tests hold them to, on a
real HFEngine over a toy model whose tokenizer carries each family's
template shape.

The rest pins the protocol itself: StreamStart first (every engine),
Finished last, send(False) aborts, count_tokens agrees with generate's own
render.
"""

import http.client
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hotloop_toys import json_tokenizer, toy_engine  # noqa: E402

from drinkme.serving import template  # noqa: E402
from drinkme.serving.engine import (Delta, FakeEngine, Finished, GenerationRequest,  # noqa: E402
                                    SampleParams, StreamStart, complete)
from drinkme.serving.http import start_server  # noqa: E402

# The generation-prompt tail of each family, in the template's own words.
# Everything else is shared: roles, an assistant turn's reasoning_content
# rendered inside its own pair (or an empty pair when the history lost it).
_HISTORY = (
    "{%- for m in messages %}"
    "{{- m['role'] + ': ' }}"
    "{%- if m['role'] == 'assistant' %}"
    "{{- '<think>\\n' + (m['reasoning_content'] if m['reasoning_content'] is defined else '')"
    " + '\\n</think>\\n\\n' }}"
    "{%- endif %}"
    "{{- m['content'] + '\\n' }}"
    "{%- endfor %}"
    "{%- if add_generation_prompt %}{{- 'assistant: ' }}"
    "{%- if enable_thinking is defined and enable_thinking is false %}"
    "{{- '<think>\\n\\n</think>\\n\\n' }}"
    "{%- else %}{{- OPEN }}{%- endif %}{%- endif %}")

FAMILIES = {
    # Qwen3.8's shape: the default prompt ENDS INSIDE an open block
    "open": _HISTORY.replace("OPEN", "'<think>\\n'"),
    # Qwen3-8B's shape: the default prompt opens nothing; the model itself
    # decides to emit <think> in the stream
    "closed": _HISTORY.replace("OPEN", "''"),
}

TURN2 = [{"role": "system", "content": "be brief"},
         {"role": "user", "content": "hello"},
         {"role": "assistant", "content": "hi", "reasoning_content": "greet back"},
         {"role": "user", "content": "world"}]


def _family_engine(family: str):
    tok = json_tokenizer()
    tok.chat_template = FAMILIES[family]
    eng = toy_engine(tokenizer=tok)
    # every render this tokenizer does, in order — the ids the ENGINE built
    # its prompt from, whoever asked for the generation
    rendered = []
    real = tok.apply_chat_template

    def recording(*a, **kw):
        out = real(*a, **kw)
        rendered.append(list(out if isinstance(out, list) else out["input_ids"]))
        return out

    tok.apply_chat_template = recording
    return eng, rendered


@pytest.fixture(scope="module", params=sorted(FAMILIES))
def family(request):
    eng, rendered = _family_engine(request.param)
    srv = start_server(eng, "127.0.0.1", 0)
    try:
        yield request.param, eng, rendered, srv.server_address[1]
    finally:
        srv.shutdown()


def _post(port, body, path="/v1/chat/completions"):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
    c.request("POST", path, json.dumps(body), {"Content-Type": "application/json"})
    r = c.getresponse()
    data = r.read()
    c.close()
    return r.status, data


def _greedy(n=3):
    return SampleParams(temperature=0.0, max_tokens=n)


# ------------------------------------------ the thinking-kwargs gotcha, closed --


def test_a_direct_caller_with_thinking_off_gets_the_ids_the_wire_produces(family):
    """The regression: thinking off, turn 2 of an exchange whose first
    assistant turn carries reasoning_content. Through /v1/chat/completions
    and through a GenerationRequest built by hand, the engine renders the
    SAME ids — and in both families the prompt opens nothing."""
    name, eng, rendered, port = family
    del rendered[:]
    status, _ = _post(port, {"messages": TURN2, "max_tokens": 3, "temperature": 0,
                             "chat_template_kwargs": {"enable_thinking": False}})
    assert status == 200
    wire = rendered[-1]  # the render generate() ran (the count ran first)
    del rendered[:]

    req = GenerationRequest(TURN2, _greedy(), template_kwargs={"enable_thinking": False})
    start = next(eng.generate(req))
    assert isinstance(start, StreamStart)
    assert rendered == [wire], name
    assert start.opens_think is False and start.prompt_tokens == len(wire)
    assert eng.count_tokens(req) == len(wire)  # the count is the same render


def test_thinking_left_on_says_so_in_the_first_event_per_family(family):
    """The mirror image: NO kwargs. The open family's prompt ends inside a
    block and StreamStart says so; the closed family's does not. Before GenerationRequest
    a direct caller could not learn this without reading a thread-local
    that only http.py ever bound."""
    name, eng, rendered, port = family
    del rendered[:]
    status, _ = _post(port, {"messages": TURN2, "max_tokens": 3, "temperature": 0})
    assert status == 200
    wire = rendered[-1]
    del rendered[:]

    start = next(eng.generate(GenerationRequest(TURN2, _greedy())))
    assert rendered == [wire], name
    assert start.opens_think is (name == "open")
    # and the two settings really render two different prompts, so the
    # equality above is not vacuous
    off = GenerationRequest(TURN2, _greedy(), template_kwargs={"enable_thinking": False})
    assert eng.count_tokens(off) != len(wire)


def test_the_messages_dialect_and_a_direct_caller_agree_too(family):
    """`thinking: {type: disabled}` on /v1/messages is the same request
    value as the chat dialect's chat_template_kwargs — and the same ids."""
    name, eng, rendered, port = family
    del rendered[:]
    status, _ = _post(port, {"model": eng.model_id, "max_tokens": 3, "temperature": 0,
                             "system": "be brief",
                             "thinking": {"type": "disabled"},
                             "messages": [{"role": "user", "content": "hello"},
                                          {"role": "assistant", "content": "hi"},
                                          {"role": "user", "content": "world"}]},
                     path="/v1/messages")
    assert status == 200, name
    wire = rendered[-1]
    del rendered[:]
    hist = [{"role": "system", "content": "be brief"},
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
            {"role": "user", "content": "world"}]
    next(eng.generate(GenerationRequest(hist, _greedy(),
                                        template_kwargs={"enable_thinking": False})))
    assert rendered == [wire], name


# -------------------------------------------------------- the protocol --


def test_every_stream_is_stream_start_then_deltas_then_finished(family):
    _, eng, _, _ = family
    evs = list(eng.generate(GenerationRequest(TURN2, _greedy(4))))
    assert isinstance(evs[0], StreamStart) and isinstance(evs[-1], Finished)
    assert all(isinstance(e, Delta) for e in evs[1:-1])
    assert evs[-1].result.prompt_tokens == evs[0].prompt_tokens
    assert "".join(e.text for e in evs[1:-1]) == evs[-1].result.text


def test_send_false_aborts_and_finished_still_arrives(family):
    _, eng, _, _ = family
    stream = eng.generate(GenerationRequest(TURN2, _greedy(50)))
    assert isinstance(next(stream), StreamStart)
    ev = stream.send(True)
    assert isinstance(ev, Delta)
    ev = stream.send(False)  # the client is gone
    while isinstance(ev, Delta):  # a held-back tail may still be flushed
        ev = stream.send(False)
    assert isinstance(ev, Finished) and ev.result.finish_reason == "abort"
    with pytest.raises(StopIteration):
        stream.send(None)


def test_complete_is_the_stream_collected(family):
    _, eng, _, _ = family
    req = GenerationRequest(TURN2, _greedy(4))
    evs = list(eng.generate(req))
    seen = []
    res = complete(eng, req, lambda d: seen.append(d) or True)
    assert res == evs[-1].result and seen == [e.text for e in evs[1:-1]]
    assert complete(eng, req, lambda d: False).finish_reason == "abort"


def test_a_prompt_that_fills_the_window_still_starts_the_stream():
    eng = toy_engine(ctx=8)
    req = GenerationRequest([{"role": "user", "content": "x" * 64}], _greedy())
    evs = list(eng.generate(req))
    assert isinstance(evs[0], StreamStart) and isinstance(evs[1], Finished) and len(evs) == 2
    assert evs[1].result.finish_reason == "length" and evs[1].result.completion_tokens == 0


def test_the_fake_engine_speaks_the_same_protocol():
    req = GenerationRequest([{"role": "user", "content": "a b"}], SampleParams(max_tokens=2))
    evs = list(FakeEngine(reply="x y z", opens_think=True).generate(req))
    assert evs[0] == StreamStart(prompt_tokens=2, opens_think=True, cached_tokens=0)
    assert [e.text for e in evs[1:-1]] == ["x ", "y "]
    assert evs[-1].result.finish_reason == "length"


def test_the_request_is_frozen():
    req = GenerationRequest([{"role": "user", "content": "hi"}], SampleParams())
    with pytest.raises(AttributeError):
        req.stream = True


def test_a_constrained_request_renders_with_thinking_off_unless_told(family):
    """The engine's own safety net for a direct caller with a schema: the
    constraint bans '<', so the render turns thinking off — the same rule
    the wire applies explicitly — unless the request said otherwise."""
    name, eng, rendered, _ = family
    schema = {"type": "object"}
    off = GenerationRequest(TURN2, _greedy(), template_kwargs={"enable_thinking": False})
    con = GenerationRequest(TURN2, SampleParams(temperature=0.0, max_tokens=3,
                                                output_schema=schema))
    assert eng.count_tokens(con) == eng.count_tokens(off)
    on = GenerationRequest(TURN2, SampleParams(temperature=0.0, max_tokens=3,
                                               output_schema=schema),
                           template_kwargs={"enable_thinking": True})
    assert eng.count_tokens(on) == eng.count_tokens(GenerationRequest(TURN2, _greedy()))


# ----------------------------------------------------- the channel is gone --


def test_the_implicit_channel_no_longer_exists():
    for name in ("request_kwargs", "bind_request", "bound_request", "prompt_opens_think",
                 "note_open_think", "_request"):
        assert not hasattr(template, name), name
    assert "threading" not in sys.modules or not hasattr(template, "threading")
    with pytest.raises(ImportError):
        from drinkme.serving.template import request_kwargs  # noqa: F401
    with pytest.raises(ImportError):
        from drinkme.serving.template import prompt_opens_think  # noqa: F401


def test_the_old_signature_is_refused_not_misread():
    """A caller still passing (messages, params, on_delta) fails at the
    call, loudly, on every engine — never silently served with defaults."""
    msgs = [{"role": "user", "content": "hi"}]
    for eng in (FakeEngine(), toy_engine()):
        with pytest.raises(TypeError):
            list(eng.generate(msgs, SampleParams(), lambda _d: True))
        with pytest.raises((TypeError, AttributeError)):
            eng.count_tokens(msgs)
