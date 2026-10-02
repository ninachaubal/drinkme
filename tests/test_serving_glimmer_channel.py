"""Muse-Glimmer's reasoning channel and the cross-turn prefix cache — CPU only,
no downloads.

THE MISS THIS FIXES, measured on the real 30B (greedy, a
two-turn text chat). Muse-Glimmer answers in ATEM messages: its generation
prompt ends `<|start|>assistant` and it writes

    ` to=self<|message|>REASONING<|eom|><|start|>assistant to=user<|message|>ANSWER<|eot|>`

With no row for it, drinkme stripped the special tokens and returned all of
that as `content`. A client sending it back got `<|start|>assistant
to=user<|message|> to=self…` from the template, where the slot held
`<|start|>assistant to=self<|message|>…`: the prompts parted at token 36 of
125, and no later turn reused a token (every image in the conversation
re-ran the tower, 4.3 s each at the pixel cap). Sent back split, the same
turn reused 125 of 136.

The fix is the `atem` row (serving/tool_formats.py) and think.AddressedSplitter:
the `to=self` messages leave as reasoning in all three dialects, `content`
carries the rest, and the bytes are exactly the ones the template renders
back. What is pinned here:

  the splitter    every stream shape (reasoning then answer, a call left
                  by the scanner, prose, interleaving, truncation) at every
                  chunking
  the row         the official chat template (tests/templates/, Apache-2.0)
                  identifies it, and the calls it renders parse back to
                  themselves and render again byte for byte
  the declaration the checkpoint's own response_template, parsed by
                  transformers, agrees with the split — up to whitespace,
                  which it strips and which the cache needs kept
  the real vocab  from the local HF cache when present: the markers are
                  special ids, the probe names the row, and the model's
                  emission from the 30B run re-renders as a prefix of the
                  next prompt
  the cache       a toy HFEngine behind the real HTTP server, with the
                  vendored template on a character tokenizer and a scripted
                  sampler, and a spy on engines.pick_slot: turn 2 and a tool
                  loop EXTEND the slot through chat, messages and responses,
                  streamed and not. The two misses the template cannot avoid
                  are pinned where they part.
"""

from __future__ import annotations

import json
import os

import pytest
import torch

from drinkme.serving import capability, control, engines, think
from drinkme.serving import tool_formats as tf
from drinkme.serving.detok import IncrementalDetok
from drinkme.serving.template import render_prompt
from drinkme.serving.tools import ToolCallScanner, split_calls, to_openai_calls

HERE = os.path.dirname(os.path.abspath(__file__))
# The checkpoint's chat_template.jinja and its tokenizer_config.json's
# response_template, at the pinned revision (the .LICENSE file beside them)
TEMPLATE = os.path.join(HERE, "templates", "muse-glimmer-30b.jinja")
RESPONSE_TEMPLATE = os.path.join(HERE, "templates", "muse-glimmer-30b.response_template.json")
GLIMMER = ("meta-models/Muse-Glimmer-30B", "a4e59da52a7bc87ae7251dd5545c0dd437c44b68")

OWNER_TOOLS = [{"type": "function", "function": {
    "name": "lookup_owner", "description": "Who owns a service.",
    "parameters": {"type": "object", "properties": {
        "service": {"type": "string"}, "limit": {"type": "integer"},
        "strict": {"type": "boolean"}, "tags": {"type": "array"},
        "where": {"type": "object"}, "note": {"type": "null"}},
        "required": ["service"]}}}]

# What the scanner hands the splitter (engines.py: tool extraction first),
# decoded with the row's control tokens kept.
REASON_THEN_ANSWER = (" to=self<|message|>Name one prime.\n\n101 works.\n\n<|eom|>"
                      "<|start|>assistant to=user<|message|>101 is prime.\n\nAlso 103.")
CALL = ('<atem:function_calls>\n<atem:invoke name="lookup_owner">\n'
        '<atem:parameter name="service">bottle</atem:parameter>\n'
        '</atem:invoke>\n</atem:function_calls>')
CALL_STREAM = (" to=self<|message|>I need the owner.<|eom|>"
               "<|start|>assistant to=lookup_owner<|message|>" + CALL)


def split(text: str, in_think: bool = False) -> tuple[str, str]:
    return think.split_text(text, in_think, "atem")


def pieces(text: str, chunks: list[str], in_think: bool = False) -> tuple[str, str]:
    s = think.splitter(in_think, "atem")
    r, c = [], []
    for p in chunks:
        a, b = s.feed(p)
        r.append(a)
        c.append(b)
    a, b = s.flush()
    return "".join(r) + a, "".join(c) + b


# The streams the splitter must hold, with their two channels. Every one is
# also run at every split offset and a character at a time below.
CASES = {
    "reasoning, then the answer": (
        REASON_THEN_ANSWER, False, ("Name one prime.\n\n101 works.\n\n", "101 is prime.\n\nAlso 103.")),
    "the answer alone": (" to=user<|message|>Hi.", False, ("", "Hi.")),
    "a call the scanner consumed": (
        " to=self<|message|>I need the owner.<|eom|><|start|>assistant to=lookup_owner<|message|>",
        False, ("I need the owner.", "")),
    "two calls the scanner consumed": (
        " to=self<|message|>Two.<|eom|><|start|>assistant to=a<|message|><|eom|>"
        "<|start|>assistant to=b<|message|>", False, ("Two.", "")),
    "a call no scanner consumed is content": (CALL_STREAM, False, ("I need the owner.", CALL)),
    "interleaved messages keep their order per channel": (
        " to=self<|message|>R1<|eom|><|start|>assistant to=user<|message|>A1<|eom|>"
        "<|start|>assistant to=self<|message|>R2<|eom|><|start|>assistant to=user<|message|>A2",
        False, ("R1R2", "A1A2")),
    "prose is content byte for byte": ("echo: hello world", False, ("", "echo: hello world")),
    "leading whitespace before prose is kept": ("  plain words", False, ("", "  plain words")),
    "a prompt that opened the reasoning message": (
        "weighing<|eom|><|start|>assistant to=user<|message|>A", True, ("weighing", "A")),
    "truncated inside the reasoning": (" to=self<|message|>still weighing", False,
                                       ("still weighing", "")),
    "truncated on a partial close": (" to=self<|message|>R<|eo", False, ("R<|eo", "")),
    "truncated inside the next header: the text it was": (
        " to=self<|message|>R<|eom|><|start|>assistant to=us", False,
        ("R", "<|start|>assistant to=us")),
    "a header without a recipient is the answer's": ("<|start|>assistant<|message|>A", False,
                                                     ("", "A")),
    "text after a close that is not a header": (" to=self<|message|>R<|eom|>then prose", False,
                                                ("R", "then prose")),
    "a recipient past the bound is not a header": (
        " to=" + "x" * 200 + "<|message|>A", False, ("", " to=" + "x" * 200 + "<|message|>A")),
    "an empty recipient is not a header": (" to=<|message|>A", False, ("", " to=<|message|>A")),
    "a header with a stray attribute is not one": (
        " to=user json<|message|>A", False, ("", " to=user json<|message|>A")),
}


@pytest.mark.parametrize("name", sorted(CASES))
def test_the_split(name):
    text, in_think, want = CASES[name]
    assert split(text, in_think) == want


@pytest.mark.parametrize("name", sorted(CASES))
def test_the_split_is_the_same_at_every_offset_and_a_character_at_a_time(name):
    text, in_think, want = CASES[name]
    for k in range(len(text) + 1):
        assert pieces(text, [text[:k], text[k:]], in_think) == want, k
    assert pieces(text, list(text), in_think) == want


def test_the_header_is_held_until_it_completes():
    s = think.splitter(False, "atem")
    assert s.feed(" to=se") == ("", "")
    assert s.feed("lf<|mess") == ("", "")
    assert s.feed("age|>weigh") == ("weigh", "")
    assert s.feed("ing<|e") == ("ing", "")  # the partial close is held
    assert s.feed("om|><|start|>assis") == ("", "")
    assert s.feed("tant to=user<|message|>ans") == ("", "ans")
    assert s.flush() == ("", "")


def test_prose_is_released_as_soon_as_it_cannot_be_a_header():
    s = think.splitter(False, "atem")
    assert s.feed("t") == ("", "")  # could still be `to=`
    assert s.feed("he answer") == ("", "the answer")


def test_the_splitter_is_the_rows():
    assert isinstance(think.splitter(False, "atem"), think.AddressedSplitter)
    for name in [r.name for r in tf.ROWS if r.name != "atem"] + [None, "unknown"]:
        assert isinstance(think.splitter(False, name), think.ThinkSplitter), name
    with pytest.raises(ValueError):
        think.AddressedSplitter(False, "json")


def test_the_default_row_leaves_an_atem_stream_alone():
    """Nothing about ATEM is hardcoded outside its row: on `json` the same
    bytes are content, untouched."""
    assert think.split_text(REASON_THEN_ANSWER, False, "json") == ("", REASON_THEN_ANSWER)


# -------------------------------------------------------------- the row --


def test_the_row_declares_its_messages_and_its_markers():
    row = tf.row("atem")
    assert (row.open, row.close) == ("<atem:function_calls>", "</atem:function_calls>")
    assert row.control_tokens == ("<|start|>", "<|message|>", "<|eom|>")
    assert (row.message_open, row.recipient_open, row.message_body) == (
        "<|start|>assistant", "to=", "<|message|>")
    assert row.think_open == row.recipient_open + row.think_channel + row.message_body
    assert (row.think_channel, row.think_close) == ("self", "<|eom|>")
    assert row.tested  # the GPU dialect smoke flipped it (the row quotes it)
    assert row.think_open in think.OPENS


class CharTok:
    """The vendored template through transformers' own jinja renderer, one
    id per character (tests/test_serving_capability.py's JinjaTok shape) —
    enough for capability.probe and render_prompt."""

    def __init__(self, template: str):
        self.template = template

    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=True,
                            tools=None, **kw):
        from transformers.utils.chat_template_utils import render_jinja_template

        rendered, _ = render_jinja_template(
            conversations=[messages], tools=tools, chat_template=self.template,
            add_generation_prompt=add_generation_prompt, **kw)
        return [ord(c) for c in rendered[0]]

    def decode(self, ids, skip_special_tokens=False):
        return "".join(chr(i) for i in ids)


@pytest.fixture(scope="module")
def official():
    return official_text()


def render(template: str, messages, tools=None, add_generation_prompt=False) -> str:
    from transformers.utils.chat_template_utils import render_jinja_template

    rendered, _ = render_jinja_template(
        conversations=[messages], tools=tools, chat_template=template,
        add_generation_prompt=add_generation_prompt, bos_token="<|begin_of_text|>")
    return rendered[0]


def test_the_official_template_is_probed_as_the_atem_row(official):
    """It was "unknown" before the row existed, which refused every tools
    request by name. Both probes agree: the declaration block and the
    written-back call."""
    cap = capability.probe(CharTok(official))
    assert cap.tool_format == "atem"
    assert capability._history_format(CharTok(official)) == "atem"


def test_the_official_template_reasons_always(official):
    """No enable_thinking (its control is a reasoning strength, default
    high, with no off), and every reply opens with a `to=self` message the
    server returns as reasoning: `always`, with no switch to name."""
    tok = CharTok(official)
    assert capability._thinking(tok) == "none"  # the render alone has no switch
    cap = capability.probe(tok)
    assert cap.thinking == "always"
    assert cap.as_dict()["thinkingSwitch"] is None


def test_the_generation_prompt_opens_no_message():
    tok = CharTok(official_text())
    p = render_prompt(tok, [{"role": "user", "content": "hi"}])
    assert tok.decode(p.ids).endswith("<|start|>assistant") and not p.opens_think


def official_text() -> str:
    with open(TEMPLATE) as f:
        return f.read()


ARGS = {"service": "bottle", "limit": 3, "strict": True, "tags": ["a", "b"],
        "where": {"region": "us-west", "tier": 2}, "note": None}


def test_a_call_the_template_renders_parses_back_and_renders_again(official):
    """The row's parser against the dialect's own producer: render a call,
    take back the block between the row's markers, parse it with the
    request's schema, render the parse — byte for byte the first render.
    That round trip is what a tool loop's history needs to extend the cache."""
    call = {"id": "call_0", "type": "function",
            "function": {"name": "lookup_owner", "arguments": ARGS}}
    first = render(official, [{"role": "user", "content": "q"},
                              {"role": "assistant", "content": "", "tool_calls": [call]}],
                   tools=OWNER_TOOLS)
    i = first.rindex("<atem:function_calls>")
    j = first.index("</atem:function_calls>", i) + len("</atem:function_calls>")
    block = first[i:j]
    assert first[:i].endswith("<|start|>assistant to=lookup_owner<|message|>")
    visible, calls = split_calls(block, OWNER_TOOLS, "atem")
    assert visible == "" and calls == [{"name": "lookup_owner", "arguments": ARGS}]
    back = {"id": "call_0", "type": "function",
            "function": {"name": "lookup_owner",
                         "arguments": json.loads(to_openai_calls(calls)[0]["function"]["arguments"])}}
    again = render(official, [{"role": "user", "content": "q"},
                              {"role": "assistant", "content": "", "tool_calls": [back]}],
                   tools=OWNER_TOOLS)
    assert again == first


def test_the_official_template_needs_every_tool_described(official):
    """`fn.description | tojson` is unguarded in render_tool_defs, so a tool
    without a description (optional in OpenAI's shape) fails the render
    itself: a TypeError out of tojson, not a jinja2 exception, which
    http.py's _template_refusal does not classify as the client's. The
    server never hands it such a tool: validate_tools fills the default
    (test_a_tool_without_description_or_parameters_renders_in_every_dialect)."""
    bare = [{"type": "function", "function": {
        "name": "lookup_owner", "parameters": OWNER_TOOLS[0]["function"]["parameters"]}}]
    with pytest.raises(TypeError, match="Undefined"):
        render(official, [{"role": "user", "content": "q"}], tools=bare,
               add_generation_prompt=True)
    render(official, [{"role": "user", "content": "q"}], tools=OWNER_TOOLS,
           add_generation_prompt=True)


def test_values_are_typed_by_the_schema_only():
    block = ('<atem:function_calls>\n<atem:invoke name="lookup_owner">\n'
             '<atem:parameter name="service">42</atem:parameter>\n'
             '<atem:parameter name="limit">3</atem:parameter>\n'
             '<atem:parameter name="undeclared">[1, 2]</atem:parameter>\n'
             '</atem:invoke>\n</atem:function_calls>')
    _, calls = split_calls(block, OWNER_TOOLS, "atem")
    assert calls[0]["arguments"] == {"service": "42", "limit": 3, "undeclared": "[1, 2]"}
    _, bare = split_calls(block, None, "atem")  # no schemas: all text
    assert bare[0]["arguments"] == {"service": "42", "limit": "3", "undeclared": "[1, 2]"}


def test_string_values_keep_their_spaces_and_newlines():
    """The template's instruction: "spaces for string values are not
    stripped" — and a value may span lines."""
    block = ('<atem:function_calls>\n<atem:invoke name="lookup_owner">\n'
             '<atem:parameter name="service"> two\nlines </atem:parameter>\n'
             '</atem:invoke>\n</atem:function_calls>')
    _, calls = split_calls(block, OWNER_TOOLS, "atem")
    assert calls[0]["arguments"] == {"service": " two\nlines "}


def test_one_block_may_carry_several_invokes():
    block = ('<atem:function_calls>\n<atem:invoke name="a">\n'
             '<atem:parameter name="x">1</atem:parameter>\n</atem:invoke>\n'
             '<atem:invoke name="b">\n</atem:invoke>\n</atem:function_calls>')
    _, calls = split_calls(block, None, "atem")
    assert calls == [{"name": "a", "arguments": {"x": "1"}}, {"name": "b", "arguments": {}}]


@pytest.mark.parametrize("body", [
    "\nprose, not a call\n",
    '\n<atem:invoke name="not a name">\n</atem:invoke>\n',
    '\n<atem:invoke name="a">\nstray<atem:parameter name="x">1</atem:parameter>\n</atem:invoke>\n',
    '\n<atem:invoke name="a">\n</atem:invoke>\ntrailing\n',
])
def test_a_block_that_is_not_a_call_reemerges_verbatim(body):
    block = "<atem:function_calls>" + body + "</atem:function_calls>"
    visible, calls = split_calls("a " + block + " b", OWNER_TOOLS, "atem")
    assert calls == [] and visible == "a " + block + " b"


def test_two_calls_render_as_two_messages_and_scan_back(official):
    """The template writes each call as its own message, `<|eom|>` between
    them and the turn's end token after the last; the scanner lifts both
    and the splitter is left two empty tool messages."""
    calls = [{"id": f"c{i}", "type": "function",
              "function": {"name": n, "arguments": {"service": s}}}
             for i, (n, s) in enumerate([("lookup_owner", "bottle"), ("lookup_owner", "cork")])]
    text = render(official, [{"role": "user", "content": "q"},
                             {"role": "assistant", "content": "", "reasoning_content": "Both.",
                              "tool_calls": calls},
                             {"role": "tool", "tool_call_id": "c0", "content": "x"}],
                  tools=OWNER_TOOLS)
    turn = text[text.index("<|start|>assistant") + len("<|start|>assistant"):
                text.index("<|eot|><|start|>tool")]
    visible, got = split_calls(turn, OWNER_TOOLS, "atem")
    assert [c["arguments"]["service"] for c in got] == ["bottle", "cork"]
    assert split(visible) == ("Both.", "")


# ------------------------------------- what the template can render back --
# One id per character (CharTok): a history re-renders as a prefix of the
# next prompt exactly when its text does. The real tokenizer's own
# boundaries are the real-vocab tests' below.

ANSWER_THEN_CALL = (" to=self<|message|>R<|eom|><|start|>assistant to=user<|message|>Checking."
                    "<|eom|><|start|>assistant to=lookup_owner<|message|>" + CALL)
INTERLEAVED = CASES["interleaved messages keep their order per channel"][0]


def served_history(generated: str, tools=None) -> dict:
    """The assistant turn a client sends back: every channel the server
    returned, unchanged (the calls as the chat dialect's wire carries them,
    arguments parsed back the way http.py's boundary parses them)."""
    visible, calls = split_calls(generated, tools, "atem") if tools else (generated, [])
    reasoning, content = split(visible)
    turn = {"role": "assistant", "content": content}
    if reasoning:
        turn["reasoning_content"] = reasoning
    if calls:
        turn["tool_calls"] = [dict(c, function=dict(c["function"], arguments=json.loads(
            c["function"]["arguments"]))) for c in to_openai_calls(calls)]
    return turn


def rerendered(generated: str, turn: dict, tools=None) -> tuple[str, str]:
    """(what the slot holds after turn 1, turn 2's prompt), as text."""
    tok = CharTok(official_text())
    first = [SYS, {"role": "user", "content": Q1}]
    held = tok.decode(render_prompt(tok, first, tools=tools).ids) + generated
    nxt = ({"role": "tool", "tool_call_id": turn["tool_calls"][0]["id"], "content": "ok"}
           if turn.get("tool_calls") else {"role": "user", "content": Q2})
    return held, tok.decode(render_prompt(tok, first + [turn, nxt], tools=tools).ids)


@pytest.mark.parametrize("generated,tools", [
    (REASON_THEN_ANSWER, None), (" to=user<|message|>Hi.", None), (CALL_STREAM, OWNER_TOOLS)],
    ids=["reasoning then answer", "answer alone", "reasoning then a call"])
def test_every_turn_the_template_can_write_renders_back_as_written(generated, tools):
    held, prompt = rerendered(generated, served_history(generated, tools), tools)
    assert prompt.startswith(held) and len(prompt) > len(held)


def parted(held: str, prompt: str, slot_goes_on: str, prompt_goes_on: str) -> bool:
    """The prompt leaves the slot, and from that point the two read as
    given."""
    n = lcp(list(held), list(prompt))
    return (n < len(held) and held[n:].startswith(slot_goes_on)
            and prompt[n:].startswith(prompt_goes_on))


def test_the_template_cannot_render_what_the_client_dropped():
    """reasoning_content is the only thing the template writes a `to=self`
    message from: dropped, the turn starts `to=user` where the model wrote
    `to=self`; trimmed, it ends early."""
    turn = served_history(REASON_THEN_ANSWER)
    held, prompt = rerendered(REASON_THEN_ANSWER, {"role": "assistant", "content": turn["content"]})
    assert parted(held, prompt, "self<|message|>", "user<|message|>")
    trimmed = dict(turn, reasoning_content=turn["reasoning_content"].strip())
    held, prompt = rerendered(REASON_THEN_ANSWER, trimmed)
    assert parted(held, prompt, "\n\n<|eom|><|start|>", "<|eom|><|start|>")


def test_the_template_cannot_render_an_answer_written_before_a_call():
    """An assistant turn with tool_calls renders its reasoning and its calls
    and no content at all, so a `to=user` message the model wrote before its
    call has nowhere to go: the history re-renders the call where the model
    wrote the answer."""
    turn = served_history(ANSWER_THEN_CALL, OWNER_TOOLS)
    assert turn["content"] == "Checking." and turn["tool_calls"]
    held, prompt = rerendered(ANSWER_THEN_CALL, turn, OWNER_TOOLS)
    assert parted(held, prompt, "user<|message|>Checking.", "lookup_owner<|message|>")


def test_the_template_cannot_render_alternating_messages():
    """One assistant turn renders one reasoning message and one answer; a
    turn that alternated has both channels concatenated and re-renders them
    as two messages."""
    turn = served_history(INTERLEAVED)
    assert (turn["reasoning_content"], turn["content"]) == ("R1R2", "A1A2")
    held, prompt = rerendered(INTERLEAVED, turn)
    assert parted(held, prompt, "<|eom|><|start|>assistant to=user<|message|>A1",
                  "R2<|eom|>")


# ------------------------------------ the checkpoint's own response_template --


def _parse_response():
    try:
        from transformers.utils.chat_parsing import parse_response
    except ImportError:
        pytest.skip("this transformers has no chat_parsing (response_template support)")
    with open(RESPONSE_TEMPLATE) as f:
        rt = json.load(f)
    return lambda text: parse_response(text, rt, prefix="<|start|>assistant")


# the model's whole turn, its <|eot|> included, as the response_template sees it
DECLARED = {
    "reasoning, then the answer": REASON_THEN_ANSWER + "<|eot|>",
    "the answer alone": " to=user<|message|>Hi.<|eot|>",
    "a call": CALL_STREAM + "<|eot|>",
    "two calls": (" to=self<|message|>Both.<|eom|><|start|>assistant to=lookup_owner<|message|>"
                  + CALL + "<|eom|><|start|>assistant to=lookup_owner<|message|>"
                  + CALL.replace("bottle", "cork") + "<|eot|>"),
}


def ours(text: str) -> dict:
    """The serve path's composition: the scanner, then the splitter (the
    turn's <|eot|> is an EOS, so the decode never carries it)."""
    visible, calls = split_calls(text.removesuffix("<|eot|>"), OWNER_TOOLS, "atem")
    reasoning, content = split(visible)
    return {"reasoning": reasoning, "content": content, "calls": calls}


@pytest.mark.parametrize("name", sorted(DECLARED))
def test_the_split_agrees_with_transformers_parse_of_the_response_template(name):
    """The checkpoint declares its output side in tokenizer_config.json, and
    transformers' chat_parsing reads it. Up to whitespace — its `text`
    fields strip, ours keep every byte — the two agree on reasoning,
    content and calls."""
    theirs = _parse_response()(DECLARED[name])
    got = ours(DECLARED[name])
    assert got["reasoning"].strip() == theirs.get("reasoning_content", "")
    assert got["content"].strip() == theirs.get("content", "")
    assert [{"name": c["name"], "arguments": c["arguments"]} for c in got["calls"]] == [
        {"name": t["function"]["name"], "arguments": t["function"]["arguments"]}
        for t in theirs.get("tool_calls", [])]


def test_where_the_declaration_and_the_split_differ_and_why():
    """Three differences, each on purpose:
      whitespace   transformers strips every field; the template renders
                   reasoning_content back byte for byte, so a stripped field
                   re-renders `…works.<|eom|>` where the model wrote
                   `…works.\\n\\n<|eom|>` and the cache misses (the real
                   tokenizer's test below measures it)
      prose        text outside every message is dropped there; here it is
                   content (never drop)
      interleaving a field that repeats keeps its last message there; here
                   each channel keeps all of them, in order"""
    parse = _parse_response()
    theirs = parse(DECLARED["reasoning, then the answer"])
    assert theirs["reasoning_content"] == "Name one prime.\n\n101 works."
    assert ours(DECLARED["reasoning, then the answer"])["reasoning"] == "Name one prime.\n\n101 works.\n\n"
    assert "content" not in parse("echo: hello world") and ours("echo: hello world")["content"]
    inter = CASES["interleaved messages keep their order per channel"][0]
    assert parse(inter)["reasoning_content"] == "R2" and ours(inter)["reasoning"] == "R1R2"


# ---------------------------------------------------------- the real vocab --


def _cached_tokenizer():
    """The real tokenizer from the local HF cache, offline — or a clean skip
    (tests/test_serving_control.py's helper, same contract)."""
    from huggingface_hub import try_to_load_from_cache

    hit = try_to_load_from_cache(GLIMMER[0], "tokenizer_config.json", revision=GLIMMER[1])
    if not isinstance(hit, str):
        pytest.skip(f"{GLIMMER[0]}@{GLIMMER[1][:12]} is not in the local HF cache")
    os.environ["HF_HUB_OFFLINE"] = "1"
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(GLIMMER[0], revision=GLIMMER[1], local_files_only=True)


@pytest.fixture(scope="module")
def real():
    return _cached_tokenizer()


# The real 30B's first reply in the two-turn chat of the module docstring
# (greedy, low reasoning strength; the ids its slot held, decoded with the
# special tokens). Its reasoning ends with a blank line before <|eom|>.
REAL_EMISSION = (
    " to=self<|message|>Name one prime number above 100.\n\nWe need name one prime number "
    "above 100. Simple. 101, 103, 107, 109, 113...\n\nProbably 101.\n\nAnswer: 101.\n\n"
    "Could give any. Provide one.\n\nProbably comply.\n\n<|eom|><|start|>assistant "
    "to=user<|message|>101 is prime.\n\nOther examples above 100 are 103, 107, 109, 113, "
    "127, etc.")
SYS = {"role": "system", "content": "Reasoning strength: low."}
Q1, Q2 = "Name one prime number above 100.", "Name another one."


def walk(tok, ids, tools=None, stride=1):
    """The serve loop's text path over a token stream (engines.py, then
    http.py): the row's control decode, the incremental detok, the tool
    scanner, the splitter. -> (reasoning, content, calls)."""
    decode = control.resolve(tok, "atem").decoder(tok)
    detok = IncrementalDetok()
    scan = ToolCallScanner(tools, "atem") if tools else None
    s = think.splitter(False, "atem")
    r, c, calls, gen = [], [], [], []

    def deliver(delta):
        if scan is not None:
            delta, done = scan.feed(delta)
            calls.extend(done)
        a, b = s.feed(delta)
        r.append(a)
        c.append(b)

    for t in ids:
        gen.append(t)
        if len(gen) % stride == 0:
            deliver(detok.push(decode(gen)))
    rem = detok.flush(decode(gen))
    if scan is not None:
        rem, done = scan.feed(rem)
        calls.extend(done)
        rem += scan.flush()
    deliver(rem)
    a, b = s.flush()
    return "".join(r) + a, "".join(c) + b, calls


def test_real_markers_are_special_ids_and_the_turn_end_is_an_eos(real):
    from huggingface_hub import try_to_load_from_cache

    for m in tf.row("atem").control_tokens:
        i = real.convert_tokens_to_ids(m)
        assert real.convert_ids_to_tokens(i) == m and i in real.all_special_ids, m
    cs = control.resolve(real, "atem")
    assert cs.ids == {200022: "<|start|>", 200023: "<|message|>", 200007: "<|eom|>"}
    assert cs.stop_after == frozenset() and cs.reopen == frozenset()
    with open(try_to_load_from_cache(*GLIMMER[:1], "generation_config.json",
                                     revision=GLIMMER[1])) as f:
        eos = json.load(f)["eos_token_id"]
    assert real.convert_tokens_to_ids("<|eot|>") in eos
    assert real.convert_tokens_to_ids("<|eom|>") not in eos  # the model goes on after it


def test_real_probe_names_the_row(real):
    cap = capability.probe(real)
    assert cap.tool_format == "atem" and cap.thinking == "always"


def test_real_emission_splits_at_every_chunking(real):
    ids = real.encode(REAL_EMISSION, add_special_tokens=False)
    whole = walk(real, ids, stride=len(ids))
    assert walk(real, ids, stride=1) == walk(real, ids, stride=3) == whole
    reasoning, content, calls = whole
    assert reasoning.startswith("Name one prime") and reasoning.endswith("comply.\n\n")
    assert content == "101 is prime.\n\nOther examples above 100 are 103, 107, 109, 113, 127, etc."
    assert calls == []
    for m in ("<|start|>", "<|message|>", "<|eom|>", "to=self", "to=user"):
        assert m not in reasoning and m not in content


def lcp(a: list, b: list) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def test_real_history_sent_back_split_extends_the_slot(real):
    """Turn 1's prompt plus what the model wrote is what the slot holds (its
    <|eot|> is an EOS, never written). Turn 2, rendered from the split reply,
    must start with exactly those ids."""
    p1 = render_prompt(real, [SYS, {"role": "user", "content": Q1}]).ids
    gen = real.encode(REAL_EMISSION, add_special_tokens=False)
    reasoning, content, _ = walk(real, gen)
    held = p1 + gen
    p2 = render_prompt(real, [SYS, {"role": "user", "content": Q1},
                              {"role": "assistant", "reasoning_content": reasoning,
                               "content": content},
                              {"role": "user", "content": Q2}]).ids
    assert p2[:len(held)] == held and len(p2) > len(held)


def test_real_misses_part_where_the_template_cannot_follow(real):
    """The two histories the template cannot render as written, measured:
    the reply as drinkme returned it before this row (all of it `content`,
    specials stripped) and the reply with its reasoning dropped both part
    at token 36, `to=user` where the model wrote `to=self`. So does the
    reasoning as transformers' parse returns it, stripped, at the end of the
    reasoning instead."""
    p1 = render_prompt(real, [SYS, {"role": "user", "content": Q1}]).ids
    gen = real.encode(REAL_EMISSION, add_special_tokens=False)
    held = p1 + gen
    reasoning, content, _ = walk(real, gen)

    def turn2(assistant):
        return render_prompt(real, [SYS, {"role": "user", "content": Q1}, assistant,
                                    {"role": "user", "content": Q2}]).ids

    raw = real.decode(gen, skip_special_tokens=True)
    for assistant in ({"role": "assistant", "content": raw},
                      {"role": "assistant", "content": content}):
        p2 = turn2(assistant)
        n = lcp(p2, held)
        assert n == 36 < len(held)
        assert real.decode(p2[n - 3:n + 1], skip_special_tokens=False) == "<|start|>assistant to=user"
        assert real.decode(held[n - 3:n + 1], skip_special_tokens=False) == "<|start|>assistant to=self"
    p2 = turn2({"role": "assistant", "reasoning_content": reasoning.strip(), "content": content})
    n = lcp(p2, held)
    assert len(p1) < n < len(held)
    # the model's last reasoning token is `.\n\n` (one BPE token); stripped,
    # the template writes `.` and then `<|eom|>`
    assert real.decode(held[n:n + 2], skip_special_tokens=False) == ".\n\n<|eom|>"
    assert real.decode(p2[n:n + 2], skip_special_tokens=False) == ".<|eom|>"


def test_real_tool_loop_extends_the_slot(real):
    q = "Who owns the bottle service?"
    p1 = render_prompt(real, [SYS, {"role": "user", "content": q}], tools=OWNER_TOOLS).ids
    gen = real.encode(CALL_STREAM, add_special_tokens=False)
    reasoning, content, calls = walk(real, gen, tools=OWNER_TOOLS)
    assert calls == [{"name": "lookup_owner", "arguments": {"service": "bottle"}}]
    assert (reasoning, content) == ("I need the owner.", "")
    wire = to_openai_calls(calls)
    history = [dict(c, function=dict(c["function"], arguments=json.loads(c["function"]["arguments"])))
               for c in wire]
    p2 = render_prompt(real, [SYS, {"role": "user", "content": q},
                              {"role": "assistant", "reasoning_content": reasoning,
                               "content": content, "tool_calls": history},
                              {"role": "tool", "tool_call_id": wire[0]["id"],
                               "content": "team-halibut"}], tools=OWNER_TOOLS).ids
    held = p1 + gen
    assert p2[:len(held)] == held


# ---------------------------------------------- the cache, end to end --
# A toy HFEngine behind the real HTTP server. The tokenizer is one id per
# character plus Glimmer's special tokens, carrying the vendored template;
# the model is a random 2-layer Llama whose sampler is scripted to write
# ATEM. Everything between the sampler and the wire is the serve path:
# prefill and decode into the slot, the control decode, the scanner, the
# splitter, the dialect, and on the next request the template and the slot
# picker, spied on.

SPECIALS = ["<|begin_of_text|>", "<|end_of_text|>", "<|start|>", "<|message|>",
            "<|eom|>", "<|eot|>", "<|finetune_right_pad|>"]


def char_tokenizer():
    from tokenizers import Regex, Tokenizer, decoders, pre_tokenizers
    from tokenizers.models import WordLevel
    from transformers import PreTrainedTokenizerFast

    vocab = {"<unk>": 0}
    for w in SPECIALS + [chr(i) for i in range(32, 127)] + ["\n", "\t"]:
        vocab[w] = len(vocab)
    backend = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.Split(Regex(r"[\s\S]"), behavior="isolated")
    backend.decoder = decoders.Fuse()
    tok = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="<unk>",
                                  bos_token="<|begin_of_text|>", eos_token="<|end_of_text|>",
                                  pad_token="<|finetune_right_pad|>")
    tok.add_special_tokens({"additional_special_tokens": SPECIALS[2:6]})
    tok.chat_template = official_text()
    return tok


class Script:
    """engines.sample_next, scripted: the next queued id, logits ignored."""

    def __init__(self):
        self.ids: list[int] = []

    def __call__(self, logits, params, generator=None, **kw):
        assert self.ids, "the engine sampled past the script"
        return self.ids.pop(0)


class SlotSpy:
    """engines.pick_slot, recorded: every prompt, the ids each slot held,
    and the prefix the picker reused."""

    def __init__(self, real):
        self.real, self.calls = real, []

    def __call__(self, slots, ids, n_prompt, need, *args, **kwargs):
        held = [list(s.ids) for s in slots]
        slot, n = self.real(slots, ids, n_prompt, need, *args, **kwargs)
        self.calls.append({"prompt": list(ids), "held": held, "reused": n})
        return slot, n


@pytest.fixture(scope="module")
def toy():
    from transformers import LlamaConfig, LlamaForCausalLM

    tok = char_tokenizer()
    cfg = LlamaConfig(vocab_size=len(tok), hidden_size=32, intermediate_size=64,
                      num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
                      head_dim=16, max_position_embeddings=8192, tie_word_embeddings=False,
                      eos_token_id=[tok.convert_tokens_to_ids("<|end_of_text|>"),
                                    tok.convert_tokens_to_ids("<|eot|>")],
                      pad_token_id=None)
    torch.manual_seed(0)
    model = LlamaForCausalLM(cfg).eval().to(torch.float32)
    for p in model.parameters():
        p.requires_grad_(False)
    return model, tok


@pytest.fixture
def served(toy, monkeypatch):
    """(post, script, spy, tok) over a fresh engine and server."""
    from drinkme.serving.http import start_server

    model, tok = toy
    eng = engines.HFEngine(model, tok, model_id="toy-glimmer", arm="test", meta={}, ctx=8192)
    assert eng.capability.tool_format == "atem"
    assert tok.convert_tokens_to_ids("<|eot|>") in eng.eos_ids
    script, spy = Script(), SlotSpy(engines.pick_slot)
    monkeypatch.setattr(engines, "sample_next", script)
    monkeypatch.setattr(engines, "pick_slot", spy)
    srv = start_server(eng, "127.0.0.1", 0)
    port = srv.server_address[1]

    def post(path, body, expect=200):
        import http.client

        c = http.client.HTTPConnection("127.0.0.1", port, timeout=60)
        c.request("POST", path, json.dumps(body), {"Content-Type": "application/json"})
        r = c.getresponse()
        data = r.read()
        c.close()
        assert r.status == expect, data
        if body.get("stream") and expect == 200:
            return [json.loads(line[len("data: "):]) for line in data.decode().split("\n")
                    if line.startswith("data: ") and line != "data: [DONE]"]
        return json.loads(data)

    def say(text):
        script.ids = tok.encode(text, add_special_tokens=False) + [tok.convert_tokens_to_ids("<|eot|>")]

    yield post, say, spy, tok
    srv.shutdown()


TURN1 = (" to=self<|message|>A prime above 100.\n\n101 works.\n\n<|eom|>"
         "<|start|>assistant to=user<|message|>101 is prime.")
TURN2 = " to=self<|message|>Another: 103.<|eom|><|start|>assistant to=user<|message|>103."
TOOL1 = CALL_STREAM
TOOL2 = (" to=self<|message|>The tool said team-halibut.<|eom|>"
         "<|start|>assistant to=user<|message|>team-halibut owns bottle.")
R1, A1 = "A prime above 100.\n\n101 works.\n\n", "101 is prime."
MAX = 400


def extends(spy, tok) -> int:
    """The last pick: the prompt extends everything the slot held. Returns
    how much was reused; the failure names where the two part."""
    call = spy.calls[-1]
    held = call["held"][0]
    n = lcp(call["prompt"], held)
    dec = lambda ids: tok.decode(ids, skip_special_tokens=False)  # noqa: E731
    assert held and n == len(held) == call["reused"], (
        f"parts at {n} of {len(held)}: prompt {dec(call['prompt'][max(0, n - 20):n + 12])!r} "
        f"| slot {dec(held[max(0, n - 20):n + 12])!r}")
    return n


def parts(spy, tok) -> tuple[str, str]:
    """The last prompt left the slot inside a message header, so the slot
    served at most the tokens before that point (a context checkpoint at or
    below it). -> (the prompt's header, the slot's), from the header's
    `<|start|>` to a few characters past where they part."""
    call = spy.calls[-1]
    held = call["held"][0]
    n = lcp(call["prompt"], held)
    assert 0 < n < len(held) and call["reused"] <= n
    dec = lambda ids: tok.decode(ids, skip_special_tokens=False)  # noqa: E731
    return dec(call["prompt"][n - 14:n + 11]), dec(held[n - 14:n + 11])


def chat_body(messages, stream=False, tools=None):
    body = {"model": "toy-glimmer", "messages": [SYS] + messages, "max_tokens": MAX,
            "temperature": 0, "stream": stream}
    if stream:
        body["stream_options"] = {"include_usage": True}
    if tools:
        body["tools"] = tools
    return body


def chat_turn(post, messages, stream, tools=None):
    """-> (the assistant message a client sends back, cached_tokens)."""
    out = post("/v1/chat/completions", chat_body(messages, stream, tools))
    if not stream:
        msg = out["choices"][0]["message"]
        return {k: v for k, v in msg.items() if k != "refusal"}, out["usage"]
    reasoning, content, calls, usage = "", "", [], None
    for ch in out:
        usage = ch.get("usage") or usage
        for c in ch["choices"]:
            d = c["delta"]
            reasoning += d.get("reasoning_content") or ""
            content += d.get("content") or ""
            calls += d.get("tool_calls") or []
    msg = {"role": "assistant", "content": content or None}
    if reasoning:
        msg["reasoning_content"] = reasoning
    if calls:
        msg["tool_calls"] = [{k: v for k, v in c.items() if k != "index"} for c in calls]
    return msg, usage


@pytest.mark.parametrize("stream", [False, True])
def test_chat_turn_two_extends_the_slot(served, stream):
    post, say, spy, tok = served
    say(TURN1)
    msg, _ = chat_turn(post, [{"role": "user", "content": Q1}], stream)
    assert msg["reasoning_content"] == R1 and msg["content"] == A1
    say(TURN2)
    history = [{"role": "user", "content": Q1}, msg, {"role": "user", "content": Q2}]
    msg2, usage = chat_turn(post, history, stream)
    n = extends(spy, tok)
    assert usage["prompt_tokens_details"]["cached_tokens"] == n
    assert msg2["content"] == "103." and msg2["reasoning_content"] == "Another: 103."
    # and a third turn extends the second, reasoning and all
    say(TURN2)
    chat_turn(post, history + [msg2, {"role": "user", "content": "And one more."}], stream)
    assert extends(spy, tok) > n


def messages_turn(post, messages, stream, tools=None):
    body = {"model": "toy-glimmer", "system": SYS["content"], "messages": messages,
            "max_tokens": MAX, "temperature": 0, "stream": stream}
    if tools:
        body["tools"] = [{"name": t["function"]["name"], "description": t["function"]["description"],
                          "input_schema": t["function"]["parameters"]} for t in tools]
    out = post("/v1/messages", body)
    if not stream:
        return out["content"], out["usage"]
    blocks: dict[int, dict] = {}
    partial: dict[int, str] = {}
    usage = None
    for ev in out:
        if ev["type"] == "content_block_start":
            blocks[ev["index"]] = dict(ev["content_block"])
        elif ev["type"] == "content_block_delta":
            b, d = blocks[ev["index"]], ev["delta"]
            if d["type"] == "thinking_delta":
                b["thinking"] += d["thinking"]
            elif d["type"] == "text_delta":
                b["text"] += d["text"]
            elif d["type"] == "input_json_delta":
                partial[ev["index"]] = partial.get(ev["index"], "") + d["partial_json"]
        elif ev["type"] == "message_delta":
            usage = ev["usage"]
    for i, js in partial.items():
        blocks[i]["input"] = json.loads(js) if js else {}
    return [blocks[i] for i in sorted(blocks)], usage


@pytest.mark.parametrize("stream", [False, True])
def test_messages_turn_two_extends_the_slot(served, stream):
    post, say, spy, tok = served
    say(TURN1)
    blocks, _ = messages_turn(post, [{"role": "user", "content": Q1}], stream)
    assert [b["type"] for b in blocks] == ["thinking", "text"]
    assert blocks[0]["thinking"] == R1 and blocks[1]["text"] == A1
    say(TURN2)
    _, usage = messages_turn(post, [{"role": "user", "content": Q1},
                                    {"role": "assistant", "content": blocks},
                                    {"role": "user", "content": Q2}], stream)
    n = extends(spy, tok)
    assert usage["cache_read_input_tokens"] == n


def responses_turn(post, items, stream, tools=None):
    body = {"model": "toy-glimmer", "instructions": SYS["content"], "input": items,
            "max_output_tokens": MAX, "temperature": 0, "stream": stream}
    if tools:
        body["tools"] = [{"type": "function", **t["function"]} for t in tools]
    out = post("/v1/responses", body)
    if stream:
        out = next(ev["response"] for ev in out if ev["type"] == "response.completed")
    return out["output"], out["usage"]


@pytest.mark.parametrize("stream", [False, True])
def test_responses_turn_two_extends_the_slot(served, stream):
    post, say, spy, tok = served
    say(TURN1)
    output, _ = responses_turn(post, [{"role": "user", "content": Q1}], stream)
    assert [o["type"] for o in output] == ["reasoning", "message"]
    assert output[0]["summary"][0]["text"] == R1
    assert output[1]["content"][0]["text"] == A1
    say(TURN2)
    _, usage = responses_turn(post, [{"role": "user", "content": Q1}] + output
                              + [{"role": "user", "content": Q2}], stream)
    n = extends(spy, tok)
    assert usage["input_tokens_details"]["cached_tokens"] == n


@pytest.fixture
def door_closed(monkeypatch):
    """`atem` is in the tested tier (the row quotes its GPU dialect smoke), so
    a tools request reaches it with DRINKME_TOOLS_UNTESTED unset, as every
    client's does (capability.served_tool_formats)."""
    monkeypatch.delenv("DRINKME_TOOLS_UNTESTED", raising=False)


@pytest.mark.parametrize("stream", [False, True])
def test_chat_tool_loop_extends_the_slot(served, door_closed, stream):
    post, say, spy, tok = served
    q = [{"role": "user", "content": "Who owns the bottle service?"}]
    say(TOOL1)
    msg, _ = chat_turn(post, q, stream, OWNER_TOOLS)
    assert msg["reasoning_content"] == "I need the owner." and not msg.get("content")
    (call,) = msg["tool_calls"]
    assert call["function"]["name"] == "lookup_owner"
    assert json.loads(call["function"]["arguments"]) == {"service": "bottle"}
    say(TOOL2)
    msg2, usage = chat_turn(post, q + [msg, {"role": "tool", "tool_call_id": call["id"],
                                             "content": "team-halibut"}], stream, OWNER_TOOLS)
    n = extends(spy, tok)
    assert usage["prompt_tokens_details"]["cached_tokens"] == n
    assert msg2["content"] == "team-halibut owns bottle."


@pytest.mark.parametrize("stream", [False, True])
def test_messages_tool_loop_extends_the_slot(served, door_closed, stream):
    post, say, spy, tok = served
    q = [{"role": "user", "content": "Who owns the bottle service?"}]
    say(TOOL1)
    blocks, _ = messages_turn(post, q, stream, OWNER_TOOLS)
    assert [b["type"] for b in blocks] == ["thinking", "tool_use"]
    use = blocks[1]
    assert use["name"] == "lookup_owner" and use["input"] == {"service": "bottle"}
    say(TOOL2)
    messages_turn(post, q + [{"role": "assistant", "content": blocks},
                             {"role": "user", "content": [{"type": "tool_result",
                                                           "tool_use_id": use["id"],
                                                           "content": "team-halibut"}]}],
                  stream, OWNER_TOOLS)
    extends(spy, tok)


@pytest.mark.parametrize("stream", [False, True])
def test_responses_tool_loop_extends_the_slot(served, door_closed, stream):
    post, say, spy, tok = served
    q = [{"role": "user", "content": "Who owns the bottle service?"}]
    say(TOOL1)
    output, _ = responses_turn(post, q, stream, OWNER_TOOLS)
    assert [o["type"] for o in output] == ["reasoning", "function_call"]
    fc = output[1]
    say(TOOL2)
    responses_turn(post, q + output + [{"type": "function_call_output", "call_id": fc["call_id"],
                                        "output": "team-halibut"}], stream, OWNER_TOOLS)
    extends(spy, tok)


def test_models_and_tokenizer_info_announce_reasoning_always(toy):
    """/v1/models said `thinking: none` while every reply carried reasoning;
    it announces what the server returns."""
    import http.client

    from drinkme.serving.http import start_server

    model, tok = toy
    eng = engines.HFEngine(model, tok, model_id="toy-glimmer", arm="test", meta={}, ctx=8192)
    srv = start_server(eng, "127.0.0.1", 0)
    try:
        got = {}
        for path in ("/v1/models", "/tokenizer_info"):
            c = http.client.HTTPConnection("127.0.0.1", srv.server_address[1], timeout=30)
            c.request("GET", path)
            got[path] = json.loads(c.getresponse().read())
            c.close()
    finally:
        srv.shutdown()
    caps = got["/v1/models"]["data"][0]["drinkme"]["capabilities"]
    assert (caps["thinking"], caps["toolFormat"], caps["thinkingSwitch"]) == ("always", "atem", None)
    info = got["/tokenizer_info"]
    assert info["thinking"] == "always" and info["thinking_switch"] is None


BARE = {"name": "ping"}  # no description, no parameters: both optional on the wire
BARE_RENDERED = '{"name": "ping", "description": "", "parameters": {}}'


@pytest.mark.parametrize("dialect", ["chat", "messages", "responses"])
def test_a_tool_without_description_or_parameters_renders_in_every_dialect(
        served, door_closed, dialect):
    """The official template writes `fn.description | tojson` and
    `fn.parameters | tojson` unguarded (the render fails without them, pinned
    above). validate_tools gives both llama.cpp's defaults before the
    template runs, so each dialect answers instead of a 500, and the prompt
    carries "" and {}."""
    post, say, spy, tok = served
    q = [{"role": "user", "content": "Ping."}]
    say(TURN1)
    if dialect == "chat":
        post("/v1/chat/completions",
             chat_body(q, tools=[{"type": "function", "function": BARE}]))
    elif dialect == "messages":
        post("/v1/messages", {"model": "toy-glimmer", "messages": q, "max_tokens": MAX,
                              "temperature": 0, "tools": [BARE]})
    else:
        post("/v1/responses", {"model": "toy-glimmer", "input": q, "max_output_tokens": MAX,
                               "temperature": 0, "tools": [{"type": "function", **BARE}]})
    prompt = tok.decode(spy.calls[-1]["prompt"], skip_special_tokens=False)
    assert BARE_RENDERED in prompt


def test_the_misses_the_template_cannot_avoid(served):
    """Sent back without its reasoning (a client that drops it), or as the
    single `content` string drinkme returned before this row, the history
    re-renders `to=user` where the model wrote `to=self`: the prompt leaves
    the slot at the turn's first header. The official template has nothing
    else to render: it writes a reasoning message only from
    reasoning_content. A context checkpoint at the previous prompt's end
    still serves everything before that point (with --ctx-checkpoints 0 the
    whole prompt would be prefilled)."""
    post, say, spy, tok = served
    say(TURN1)
    msg, _ = chat_turn(post, [{"role": "user", "content": Q1}], False)
    raw = TURN1.replace("<|message|>", "").replace("<|eom|>", "").replace("<|start|>", "")
    for assistant in ({"role": "assistant", "content": msg["content"]},
                      {"role": "assistant", "content": raw}):
        say(TURN2)
        _, usage = chat_turn(post, [{"role": "user", "content": Q1}, assistant,
                                    {"role": "user", "content": Q2}], False)
        prompt, slot = parts(spy, tok)
        assert slot == "<|start|>assistant to=self<|message|>A prim"
        assert prompt.startswith("<|start|>assistant to=user<|message|>")
        reused = spy.calls[-1]["reused"]
        assert 0 < reused == usage["prompt_tokens_details"]["cached_tokens"]
        # the next request starts over from the full prompt it just wrote
        say(TURN1)
        chat_turn(post, [{"role": "user", "content": Q1}], False)
