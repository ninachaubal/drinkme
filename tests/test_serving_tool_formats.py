"""serving/tool_formats.py — the table, one fixture per row, pure CPU.

Every fixture below is QUOTED, never invented: its source URL sits beside it
and the bytes are the ones that artifact carries. Three of the seven rows go
further and are checked against the family's OWN chat template out of the
local HF cache (the round-trip tests at the bottom): render an assistant
`tool_calls` message, take back exactly what the template wrote, and prove
the row parses it into the call that went in. That is the strongest
provenance available without a GPU — the template is the normative producer
of the dialect.

The invariants each row must hold, all four of them the same ones the
hardcoded scanner held:

  chunk invariance   the same output split at EVERY byte offset produces the
                     same (visible text, calls) as feeding it whole — else
                     streamed and non-streamed answers disagree
  never invent       a block whose body the row's chain does not recognize
                     re-emerges verbatim as visible text
  never drop         an un-terminated tail flushes as the text it really was
  no leakage         no marker byte reaches the client as content
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import pytest

from drinkme.serving import tool_formats as tf
from drinkme.serving.tools import ToolCallScanner, split_calls

WEATHER_TOOLS = [{"type": "function", "function": {
    "name": "get_weather",
    "parameters": {"type": "object", "properties": {
        "city": {"type": "string"},
        "location": {"type": "string"},
        "unit": {"type": "string"},
        "days": {"type": "integer"}}}}}]


@dataclass(frozen=True)
class Fixture:
    """One real emission for one row, with where it came from."""

    block: str          # the tool-call markup exactly as the source carries it
    calls: list         # what the row must parse it into
    source: str         # URL — checked by a test, so it cannot rot silently
    tools: list = field(default_factory=lambda: WEATHER_TOOLS)


FIXTURES: dict[str, Fixture] = {

    # Qwen3-8B's own chat template renders an assistant tool_call back as
    # these exact bytes (the round-trip test below asserts it, offline,
    # against the cached tokenizer at the suite's pinned revision), and the
    # template's instruction block declares the form in as many words:
    # "For each function call, return a json object with function name and
    # arguments within <tool_call></tool_call> XML tags".
    "json": Fixture(
        block='<tool_call>\n{"name": "get_weather", "arguments": {"city": "Paris"}}\n</tool_call>',
        calls=[{"name": "get_weather", "arguments": {"city": "Paris"}}],
        source="https://huggingface.co/Qwen/Qwen3-8B/blob/main/tokenizer_config.json"),

    # Qwen3.8-27B's template renders the same call in the qwen3_5 XML form —
    # again byte-for-byte what the round-trip test takes back off the cached
    # tokenizer. Cross-checked against vllm/parser/qwen3.py's documented
    # shape (`<tool_call>\n<function=func_name>\n<parameter=key>value...`).
    "qwen-xml": Fixture(
        block=("<tool_call>\n<function=get_weather>\n"
               "<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>"),
        calls=[{"name": "get_weather", "arguments": {"city": "Paris"}}],
        source="https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/tokenizer_config.json"),

    # gemma-4-31B-it's canonical template (Google, published 2026-07-09)
    # renders an assistant tool_call as `<|tool_call>call:NAME{k:v}<tool_call|>`
    # with strings wrapped in its own `<|"|>` delimiter — the round-trip test
    # takes these exact bytes back out of the cached tokenizer.
    "gemma": Fixture(
        block='<|tool_call>call:get_weather{city:<|"|>Paris<|"|>}<tool_call|>',
        calls=[{"name": "get_weather", "arguments": {"city": "Paris"}}],
        source="https://huggingface.co/google/gemma-4-31B-it/blob/main/chat_template.jinja"),

    # vllm/parser/glm47_moe.py's module docstring: "GLM-4.7 uses XML-like
    # tool calls:: <tool_call>func_name<arg_key>key</arg_key>
    # <arg_value>value</arg_value></tool_call>". zai-org/GLM-4.7's own
    # chat_template.jinja declares the same shape and writes string values
    # raw, everything else through tojson — which is why `days` comes back
    # an int (declared integer) and `city` stays the text it was.
    "glm": Fixture(
        block=("<tool_call>get_weather<arg_key>city</arg_key><arg_value>Paris</arg_value>"
               "<arg_key>days</arg_key><arg_value>3</arg_value></tool_call>"),
        calls=[{"name": "get_weather", "arguments": {"city": "Paris", "days": 3}}],
        source="https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/parser/glm47_moe.py"),

    # vllm/parser/minimax_m2.py's module docstring, verbatim including the
    # newlines. MiniMaxAI/MiniMax-M2's chat_template.jinja teaches the same
    # block in its system prompt ("When making tool calls, use XML format to
    # invoke tools and pass parameters:").
    "minimax": Fixture(
        block=('<minimax:tool_call><invoke name="get_weather">\n'
               '<parameter name="city">Seattle</parameter>\n'
               '</invoke></minimax:tool_call>'),
        calls=[{"name": "get_weather", "arguments": {"city": "Seattle"}}],
        source="https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/parser/minimax_m2.py"),

    # vllm/parser/mistral.py, MistralParser._extract_tool_calls_pre_v11:
    # 'Handles ``[TOOL_CALLS][{"name": "add", "arguments":{"a": 3.5}}]``'.
    # mistralai/Mistral-7B-Instruct-v0.3's chat_template writes the same
    # `[TOOL_CALLS] [` + tojson'd calls, and declares tools with
    # `[AVAILABLE_TOOLS] [` — which is this row's signature.
    "mistral": Fixture(
        block='[TOOL_CALLS][{"name": "add", "arguments":{"a": 3.5}}]',
        calls=[{"name": "add", "arguments": {"a": 3.5}}],
        source="https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/parser/mistral.py"),

    # vllm/parser/kimi_k2.py's module docstring, verbatim: "The header
    # before ``<|tool_call_argument_begin|>`` is Kimi's native tool call id.
    # The function name is the final component before ``:N``."
    # moonshotai/Kimi-K2-Instruct's chat_template.jinja emits the same
    # section.
    "kimi-k2": Fixture(
        block=("<|tool_calls_section_begin|>\n"
               "<|tool_call_begin|>functions.get_weather:0\n"
               '<|tool_call_argument_begin|>{"city": "Tokyo"}<|tool_call_end|>\n'
               "<|tool_calls_section_end|>"),
        calls=[{"name": "get_weather", "arguments": {"city": "Tokyo"}}],
        source="https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/parser/kimi_k2.py"),

    # meta-models/Muse-Glimmer-30B's own chat template renders an assistant
    # tool_call as these exact bytes (its render_atem macro; the round-trip
    # test below takes them back off the cached tokenizer), the body of a
    # message addressed `to=get_weather`. The checkpoint's response_template
    # declares the same invoke and parameter patterns
    # (tests/test_serving_glimmer_channel.py holds the row to it).
    "atem": Fixture(
        block=('<atem:function_calls>\n<atem:invoke name="get_weather">\n'
               '<atem:parameter name="city">Paris</atem:parameter>\n'
               '</atem:invoke>\n</atem:function_calls>'),
        calls=[{"name": "get_weather", "arguments": {"city": "Paris"}}],
        source="https://huggingface.co/meta-models/Muse-Glimmer-30B/blob/main/chat_template.jinja",
        # the template writes `fn.description | tojson` unguarded, so a tool
        # without one fails the render (tests/test_serving_glimmer_channel.py)
        tools=[{"type": "function", "function": dict(WEATHER_TOOLS[0]["function"],
                                                     description="Current weather.")}]),
}

ROW_NAMES = [r.name for r in tf.ROWS]


def run(text: str, tool_format: str, tools=None, splits=None):
    """Feed `text` through a scanner on row `tool_format`, optionally pre-split, and
    end the stream. -> (visible text, calls). One helper so every test below
    exercises the SAME terminal sequence the engines do (feed*, flush,
    flush_calls)."""
    s = ToolCallScanner(tools, tool_format)
    visible, calls = [], []
    for part in (splits if splits is not None else [text]):
        v, c = s.feed(part)
        visible.append(v)
        calls += c
    visible.append(s.flush())
    calls += s.flush_calls()
    return "".join(visible), calls


# ------------------------------------------------------------------- table --


def test_every_row_has_a_fixture():
    """The coverage gate: a new row lands with a quoted emission or not at
    all. This is the test that fails when someone adds a row from memory."""
    assert set(FIXTURES) == tf.NAMES == set(ROW_NAMES)


def test_every_fixture_names_its_source():
    for name, fx in FIXTURES.items():
        assert fx.source.startswith("https://"), name


def test_row_names_are_unique_and_ordered_specific_first():
    assert len(ROW_NAMES) == len(set(ROW_NAMES))
    # GLM's instruction block contains BOTH `<tool_call>` and the word
    # "arguments", so it must be tried before `json` or it classifies as it.
    assert ROW_NAMES.index("glm") < ROW_NAMES.index("json")
    # the precedence the hardcoded probe had, kept
    assert ROW_NAMES.index("qwen-xml") < ROW_NAMES.index("json")


def test_row_lookup_falls_back_to_the_default_row():
    assert tf.row(None).name == tf.DEFAULT
    assert tf.row("no-such-dialect").name == tf.DEFAULT
    assert tf.row("mistral").name == "mistral"


def test_no_row_has_a_prompt_hint():
    # the `hint` column exists so a family whose template declares tools but
    # never teaches the emission form can be added as DATA. Every row here
    # writes its own instruction block, so nothing consumes it yet — this
    # test is the tripwire for the day that changes.
    assert all(r.hint is None for r in tf.ROWS)
    assert tf.hint_for("json") == "" and tf.hint_for(None) == ""


# ----------------------------------------------------------- the fixtures --


@pytest.mark.parametrize("name", ROW_NAMES)
def test_fixture_parses_to_its_calls(name):
    fx = FIXTURES[name]
    visible, calls = run(fx.block, name, fx.tools)
    assert calls == fx.calls
    assert visible == ""  # no marker byte reaches the client as content


@pytest.mark.parametrize("name", ROW_NAMES)
def test_text_around_the_block_survives_verbatim(name):
    fx = FIXTURES[name]
    visible, calls = run("Checking now.\n" + fx.block, name, fx.tools)
    assert calls == fx.calls
    # a one-sided row suppresses to the END of the turn by construction, so
    # only the leading text can be visible; every other row resumes after
    # the close marker.
    assert visible == "Checking now.\n"


@pytest.mark.parametrize("name", ROW_NAMES)
def test_chunk_boundary_invariance_at_every_offset(name):
    """THE gate: the same output split at every byte offset must give
    the same two channels. think.py's discipline, applied to tool markup."""
    fx = FIXTURES[name]
    text = "before " + fx.block
    whole = run(text, name, fx.tools)
    for k in range(len(text) + 1):
        assert run(text, name, fx.tools, splits=[text[:k], text[k:]]) == whole, k


@pytest.mark.parametrize("name", ROW_NAMES)
def test_one_byte_at_a_time_matches_whole(name):
    fx = FIXTURES[name]
    text = "before " + fx.block
    assert run(text, name, fx.tools, splits=list(text)) == run(text, name, fx.tools)


@pytest.mark.parametrize("name", ROW_NAMES)
def test_split_calls_equals_the_streamed_answer(name):
    fx = FIXTURES[name]
    text = "before " + fx.block
    assert split_calls(text, fx.tools, name) == run(text, name, fx.tools)


# ------------------------------------------------------- never invent/drop --


@pytest.mark.parametrize("name", ROW_NAMES)
def test_unrecognized_body_reemerges_verbatim(name):
    """A block the row's chain cannot read is TEXT, not a guessed call."""
    row = tf.row(name)
    block = row.open + "nothing here is a call" + (row.close or "")
    visible, calls = run("a " + block + " b", name, FIXTURES[name].tools)
    assert calls == []
    # every generated byte comes back; a one-sided row's suppression means
    # the trailing " b" is inside its block, so compare on containment
    assert visible.startswith("a " + row.open) and "nothing here is a call" in visible
    assert len(visible) == len("a " + block + " b")


@pytest.mark.parametrize("name", ROW_NAMES)
def test_unterminated_block_flushes_as_text(name):
    row = tf.row(name)
    truncated = FIXTURES[name].block
    truncated = truncated[:len(truncated) - len(row.close or "") - 3]
    visible, calls = run(truncated, name, FIXTURES[name].tools)
    assert calls == []
    assert visible == truncated  # not one byte lost


@pytest.mark.parametrize("name", ROW_NAMES)
def test_partial_start_marker_is_held_then_flushed(name):
    row = tf.row(name)
    s = ToolCallScanner(FIXTURES[name].tools, name)
    visible, _ = s.feed("done " + row.open[:-1])
    assert visible == "done "  # the maybe-a-marker suffix is held, not emitted
    assert s.flush() == row.open[:-1]


@pytest.mark.parametrize("name", ROW_NAMES)
def test_lookahead_released_once_ruled_out(name):
    row = tf.row(name)
    s = ToolCallScanner(FIXTURES[name].tools, name)
    a, _ = s.feed("x " + row.open[:2])
    b, _ = s.feed("¡nope! y")
    assert a + b + s.flush() == "x " + row.open[:2] + "¡nope! y"


@pytest.mark.parametrize("name", ROW_NAMES)
def test_prose_inside_a_stray_marker_is_never_a_call(name):
    """The identifier guard. GLM's no-arg render is `<tool_call>name</tool_call>`
    and gemma/mistral read bare names too — a sentence must not become one."""
    row = tf.row(name)
    visible, calls = run(row.open + "call the weather tool please" + (row.close or ""),
                         name, FIXTURES[name].tools)
    assert calls == []
    assert "call the weather tool please" in visible


# ------------------------------------------------------- per-row specifics --


def test_glm_no_arg_call_is_the_templates_own_render():
    # zai-org/GLM-4.7's template writes `<tool_call>` + name + no pairs when
    # arguments is empty; the docstring says "tool calls may have no arguments"
    _, calls = run("<tool_call>ping</tool_call>", "glm")
    assert calls == [{"name": "ping", "arguments": {}}]


def test_glm_values_are_typed_off_the_request_schema_only():
    block = ("<tool_call>get_weather<arg_key>city</arg_key><arg_value>123</arg_value>"
             "<arg_key>days</arg_key><arg_value>3</arg_value></tool_call>")
    _, calls = run(block, "glm", WEATHER_TOOLS)
    assert calls[0]["arguments"] == {"city": "123", "days": 3}  # city declared string
    _, bare = run(block, "glm")  # no schemas at all: everything stays text
    assert bare[0]["arguments"] == {"city": "123", "days": "3"}


def test_gemma_carries_its_own_types_and_needs_no_schema():
    block = ('<|tool_call>call:f{s:<|"|>7<|"|>,n:42,b:true,z:null,'
             'o:{k:<|"|>v<|"|>},a:[1,<|"|>two<|"|>]}<tool_call|>')
    _, calls = run(block, "gemma")  # no tools passed
    assert calls == [{"name": "f", "arguments": {
        "s": "7", "n": 42, "b": True, "z": None,
        "o": {"k": "v"}, "a": [1, "two"]}}]


def test_gemma_string_delimiter_protects_structural_characters():
    block = '<|tool_call>call:sh{cmd:<|"|>a,b}c{d<|"|>}<tool_call|>'
    _, calls = run(block, "gemma")
    assert calls == [{"name": "sh", "arguments": {"cmd": "a,b}c{d"}}]


def test_minimax_block_can_carry_several_invokes():
    block = ('<minimax:tool_call><invoke name="a">\n'
             '<parameter name="x">1</parameter>\n</invoke>\n'
             '<invoke name="b">\n</invoke></minimax:tool_call>')
    _, calls = run(block, "minimax")
    assert calls == [{"name": "a", "arguments": {"x": "1"}},
                     {"name": "b", "arguments": {}}]


def test_kimi_section_can_carry_several_calls():
    block = ("<|tool_calls_section_begin|>"
             '<|tool_call_begin|>functions.a:0<|tool_call_argument_begin|>{"x": 1}<|tool_call_end|>'
             '<|tool_call_begin|>functions.b:1<|tool_call_argument_begin|>{}<|tool_call_end|>'
             "<|tool_calls_section_end|>")
    _, calls = run(block, "kimi-k2")
    assert [c["name"] for c in calls] == ["a", "b"]
    assert calls[0]["arguments"] == {"x": 1}


def test_mistral_v11_name_brace_form_also_parses():
    # vllm/parser/mistral.py: "Tool calls use the ``[TOOL_CALLS]func_name{...}``
    # format" on the v11+ engine path; the pre-v11 array form is the fixture.
    _, calls = run('[TOOL_CALLS]add{"a": 3.5}', "mistral")
    assert calls == [{"name": "add", "arguments": {"a": 3.5}}]


def test_mistral_ignores_the_id_the_template_adds():
    block = '[TOOL_CALLS] [{"name": "f", "arguments": {}, "id": "abcdefghi"}]'
    _, calls = run(block, "mistral")
    assert calls == [{"name": "f", "arguments": {}}]


def test_mistral_suppresses_to_the_end_of_the_turn():
    """The one-sided rule: nothing after the start marker is content, and
    the calls surface from flush_calls(), not from feed()."""
    s = ToolCallScanner(None, "mistral")
    visible, calls = s.feed('say hi [TOOL_CALLS] [{"name": "f", "arguments": {}}]')
    assert visible == "say hi " and calls == []
    assert s.flush() == ""
    assert s.flush_calls() == [{"name": "f", "arguments": {}}]
    assert s.flush_calls() == []  # drained once


def test_flush_calls_is_empty_for_every_close_delimited_row():
    for name in ROW_NAMES:
        if tf.row(name).close is None:
            continue
        s = ToolCallScanner(FIXTURES[name].tools, name)
        s.feed(FIXTURES[name].block)
        s.flush()
        assert s.flush_calls() == [], name


# --------------------------------------------------------------- identify --


@pytest.mark.parametrize("name", ROW_NAMES)
def test_a_rows_signature_identifies_that_row(name):
    """Concatenating a row's signature substrings must resolve to it — or to
    an EARLIER row, which is what ordering means. The real-template tests in
    tests/test_serving_capability.py are the ones that pin actual models."""
    rendered = " ".join(tf.row(name).signature)
    got = tf.identify(rendered)
    assert got is not None
    assert ROW_NAMES.index(got) <= ROW_NAMES.index(name)


def test_identify_returns_none_when_nothing_matches():
    assert tf.identify("please emit calls in the zzz-format") is None


def test_glm_render_would_be_json_without_the_ordering():
    # the actual instruction line from zai-org/GLM-4.7's chat_template.jinja
    glm_block = ("For each function call, output the function name and arguments "
                 "within the following XML format:\n<tool_call>{function-name}"
                 "<arg_key>{arg-key-1}</arg_key><arg_value>{arg-value-1}</arg_value>"
                 "</tool_call>")
    assert tf.identify(glm_block) == "glm"


# --------------------------------------------- control tokens / reasoning --


def test_gemma_declares_its_markers_as_control_tokens():
    """The row's own open and close are among them (they are what the
    close-marker stop resolves from), the channel pair, and the string
    delimiter — without which `{city:<|"|>a,b<|"|>}` reaches the parser as
    `{city:a,b}`. tests/test_serving_control.py proves each is a special id
    in the real cached vocab."""
    row = tf.row("gemma")
    assert row.open in row.control_tokens and row.close in row.control_tokens
    assert "<|channel>" in row.control_tokens and "<channel|>" in row.control_tokens
    assert tf.GEMMA_STRING_DELIM in row.control_tokens
    assert row.think_open == "<|channel>thought" and row.think_close == "<channel|>"
    assert row.think_channel == "thought"


def test_atem_declares_its_message_markers_and_addressed_messages():
    """Muse-Glimmer's header and message-end tokens are special ids
    (tests/test_serving_glimmer_channel.py reads them off the real vocab);
    its reasoning is the run of messages addressed to `self`."""
    row = tf.row("atem")
    assert row.control_tokens == ("<|start|>", "<|message|>", "<|eom|>")
    assert (row.think_open, row.think_close, row.think_channel) == (
        "to=self<|message|>", "<|eom|>", "self")
    assert row.message_open == "<|start|>assistant"


@pytest.mark.parametrize("name", [n for n in ROW_NAMES if n not in ("gemma", "atem")])
def test_every_other_row_declares_nothing_today(name):
    """The empty columns ARE today's behaviour: no kept ids, the Qwen think
    tags, the exact old decode. A row that gains control tokens moves out of
    this list deliberately, with a real-tokenizer test beside it."""
    row = tf.row(name)
    assert row.control_tokens == ()
    assert (row.think_open, row.think_close, row.think_channel) == ("<think>", "</think>", None)
    assert (row.message_open, row.recipient_open, row.message_body) == (None, None, None)


def test_a_named_channel_is_part_of_its_open_tag():
    """gemma's channel name closes its open tag; an addressed row's opens
    with its header's recipient and message markers around the name."""
    for r in tf.ROWS:
        if r.message_open is not None:
            assert r.think_open == r.recipient_open + r.think_channel + r.message_body, r.name
        elif r.think_channel is not None:
            assert r.think_open.endswith(r.think_channel), r.name


def test_an_addressed_row_declares_all_three_header_literals():
    for r in tf.ROWS:
        cols = (r.message_open, r.recipient_open, r.message_body)
        assert all(c is None for c in cols) or all(cols), r.name
        if r.message_open is not None:
            assert r.think_channel is not None, r.name


# --------------------------------- round-trip against the real templates --

CACHED = {  # row -> (repo, revision) whose OWN template emits this dialect
    "json": ("Qwen/Qwen3-8B", "b968826d9c46dd6066d109eabc6255188de91218"),
    "qwen-xml": ("Qwen/Qwen3.8-27B", "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"),
    "gemma": ("google/gemma-4-31B-it", "842da3794eaa0b77d5f08bae87a17459d91ff475"),
    "atem": ("meta-models/Muse-Glimmer-30B", "a4e59da52a7bc87ae7251dd5545c0dd437c44b68"),
}


def _cached_tokenizer(repo: str, revision: str):
    """The real tokenizer from the local HF cache, offline — or a clean
    pytest.skip naming exactly what's missing (test_serving_capability.py's
    helper, same contract)."""
    from huggingface_hub import try_to_load_from_cache

    hit = try_to_load_from_cache(repo, "tokenizer_config.json", revision=revision)
    if not isinstance(hit, str):
        pytest.skip(f"{repo}@{revision[:12]} is not in the local HF cache "
                    "(tokenizer_config.json not found) — nothing to render")
    os.environ["HF_HUB_OFFLINE"] = "1"
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(repo, revision=revision, local_files_only=True)


@pytest.mark.parametrize("name", sorted(CACHED))
def test_the_families_own_template_emits_the_fixture(name):
    """Render an assistant tool_call through the model's REAL chat template
    and take back exactly what it wrote: the markup between this row's
    markers must be the fixture, byte for byte, and must parse to the call
    that went in. No GPU, no weights — the template is the producer of the
    dialect, so this is the dialect's own word on itself."""
    tok = _cached_tokenizer(*CACHED[name])
    row, fx = tf.row(name), FIXTURES[name]
    rendered = tok.apply_chat_template(
        [{"role": "user", "content": "weather?"},
         {"role": "assistant", "content": "",
          "tool_calls": [{"id": "call_abc123", "type": "function",
                          "function": {"name": "get_weather",
                                       "arguments": {"city": "Paris"}}}]}],
        add_generation_prompt=False, tokenize=False, tools=fx.tools)
    i = rendered.rindex(row.open)
    j = rendered.index(row.close, i) + len(row.close)
    assert rendered[i:j] == fx.block
    assert run(rendered[i:j], name, fx.tools) == ("", fx.calls)


# ------------------------------------------------- the tested tier rule --

def test_tested_tier_is_exactly_the_menu_dialects():
    # "menu models stay the tested tier". json = Qwen3-8B, qwen-xml =
    # Qwen3.8-27B, gemma = gemma-4-31B-it (GPU-gated), atem =
    # Muse-Glimmer-30B (GPU-gated); each receipt is quoted on its
    # row — the dialects a real model on the menu has
    # exercised end to end. Every other row is parseable (announced by name
    # on /v1/models) and REFUSED for tools requests until a GPU smoke flips it.
    assert tf.TESTED == {"json", "qwen-xml", "gemma", "atem"}
    assert tf.TESTED < tf.NAMES


def test_capability_serves_only_the_tested_tier():
    from drinkme.serving import capability
    assert capability.PARSEABLE_TOOL_FORMATS is tf.TESTED


def test_refusal_for_an_untested_row_says_so_and_names_the_served_rows():
    from drinkme.serving import capability
    msg = capability.refusal_message("moonshotai/Kimi-K2-Instruct", "kimi-k2")
    assert "untested" in msg.lower() and "'kimi-k2'" in msg
    assert "json" in msg and "qwen-xml" in msg and "gemma" in msg
    # the old wording for a dialect we cannot parse at all is unchanged
    assert "'unknown'" in capability.refusal_message("x/y", "unknown")


def test_untested_door_is_closed_by_default_and_opens_only_by_env(monkeypatch):
    """The tested tier is what strangers get; DRINKME_TOOLS_UNTESTED=1 is the
    operator door bench/serve_dialect_smoke.py needs to exercise a row on real
    weights — a gate that cannot run against a server refusing the row would
    never flip anything."""
    from drinkme.serving import capability, tool_formats as tf
    monkeypatch.delenv("DRINKME_TOOLS_UNTESTED", raising=False)
    assert capability.served_tool_formats() == tf.TESTED
    monkeypatch.setenv("DRINKME_TOOLS_UNTESTED", "1")
    assert capability.served_tool_formats() == frozenset(r.name for r in tf.ROWS)
    assert "gemma" in capability.served_tool_formats()
    monkeypatch.setenv("DRINKME_TOOLS_UNTESTED", "0")
    assert capability.served_tool_formats() == tf.TESTED
