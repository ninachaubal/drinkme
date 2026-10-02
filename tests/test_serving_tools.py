"""serving/tools.py — the tool-call scanner and the OpenAI shapes, pure CPU.

The scanner's two absolutes get the coverage: never invent a call (every
malformed body re-emerges verbatim as text) and never drop text (unterminated
tails flush). Split-across-deltas cases mirror StopScanner's tests: the tag
arriving one character at a time is the normal case under incremental detok,
not the edge case.

Every test here is the REGRESSION GATE for the two dialects drinkme serves
by default (`json` and `qwen-xml`). The scanner became table-driven
(serving/tool_formats.py, one row per family); these tests were not rewritten
for it, deliberately — they pass unchanged against a scanner constructed the
old way, which is what "byte-for-byte behaviour-identical" has to mean. The
per-row invariants for the whole table live in
tests/test_serving_tool_formats.py.
"""

import json

import pytest

from drinkme.serving.tools import ToolCallScanner, to_openai_calls, validate_tools

CALL = '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Paris"}}\n</tool_call>'
PARSED = {"name": "get_weather", "arguments": {"city": "Paris"}}


def feed_all(scanner, deltas):
    text, calls = [], []
    for d in deltas:
        t, c = scanner.feed(d)
        text.append(t)
        calls.extend(c)
    return "".join(text), calls


# ------------------------------------------------------------------- scanner --


def test_whole_block_in_one_delta():
    s = ToolCallScanner()
    text, calls = s.feed(CALL)
    assert text == "" and calls == [PARSED]
    assert s.flush() == ""


def test_block_split_across_many_deltas():
    s = ToolCallScanner()
    text, calls = feed_all(s, list(CALL))  # one character per delta
    assert text == "" and calls == [PARSED]
    assert s.flush() == ""


def test_text_then_call():
    s = ToolCallScanner()
    text, calls = feed_all(s, ["Checking now.\n", CALL])
    assert text == "Checking now.\n" and calls == [PARSED]


def test_two_sequential_calls_in_one_feed():
    s = ToolCallScanner()
    text, calls = s.feed(CALL + "\n" + '<tool_call>{"name": "b", "arguments": {}}</tool_call>')
    assert text == "\n"  # only the between-blocks byte is visible
    assert [c["name"] for c in calls] == ["get_weather", "b"]


def test_malformed_json_reemerges_verbatim_never_a_call():
    block = '<tool_call>{"name": broken}</tool_call>'
    s = ToolCallScanner()
    text, calls = s.feed(block)
    assert text == block and calls == []


def test_non_call_bodies_reemerge():
    # valid JSON that is not a {name, arguments-object} call is still no call
    for body in ("[1, 2]", '"hi"', '{"arguments": {}}', '{"name": ""}',
                 '{"name": 3}', '{"name": "f", "arguments": [1]}'):
        s = ToolCallScanner()
        text, calls = s.feed(f"<tool_call>{body}</tool_call>")
        assert calls == [] and text == f"<tool_call>{body}</tool_call>"


def test_missing_arguments_is_a_no_arg_call():
    s = ToolCallScanner()
    _, calls = s.feed('<tool_call>{"name": "ping"}</tool_call>')
    assert calls == [{"name": "ping", "arguments": {}}]


def test_unterminated_block_flushes_as_text():
    s = ToolCallScanner()
    text, calls = s.feed('<tool_call>{"name": "x"')
    assert text == "" and calls == []
    assert s.flush() == '<tool_call>{"name": "x"'  # open tag included: no bytes lost


def test_partial_open_tag_flushes_as_text():
    s = ToolCallScanner()
    text, _ = s.feed("done <tool_ca")
    assert text == "done "  # the maybe-a-tag suffix is held, not emitted
    assert s.flush() == "<tool_ca"


def test_lookahead_released_once_ruled_out():
    s = ToolCallScanner()
    a, _ = s.feed("a <t")
    b, _ = s.feed("oy> b")
    assert a + b == "a <toy> b"
    assert s.flush() == ""


def test_text_resumes_after_a_call():
    s = ToolCallScanner()
    text, calls = feed_all(s, [CALL, " and then some"])
    assert text == " and then some" and len(calls) == 1


# ------------------------------------------------------------- wire rendering --


def test_to_openai_calls_arguments_is_a_json_string():
    (c,) = to_openai_calls([PARSED])
    assert c["id"].startswith("call_") and len(c["id"]) == len("call_") + 12
    assert c["type"] == "function"
    assert c["function"]["name"] == "get_weather"
    assert isinstance(c["function"]["arguments"], str)  # STRING, not object
    assert json.loads(c["function"]["arguments"]) == {"city": "Paris"}


def test_to_openai_calls_ids_unique():
    a, b = to_openai_calls([PARSED, PARSED])
    assert a["id"] != b["id"]


# ------------------------------------------------------------- validate_tools --


WEATHER = {"type": "function",
           "function": {"name": "get_weather", "description": "Weather for a city.",
                        "parameters": {"type": "object",
                                       "properties": {"city": {"type": "string"}},
                                       "required": ["city"]}}}


def test_validate_tools_accepts_openai_shapes():
    assert validate_tools([WEATHER]) == [WEATHER]
    minimal = [{"type": "function", "function": {"name": "f"}}]  # description/parameters optional
    assert validate_tools(minimal) == [{"type": "function", "function": {
        "name": "f", "description": "", "parameters": {}}}]
    assert validate_tools([]) == []


def test_a_missing_description_or_parameters_gets_llama_cpps_default():
    """Every dialect's tools pass through validate_tools before a template
    renders them, and llama.cpp's defaults apply there (common/chat.cpp):
    description "" and parameters {}. A tool that has both comes out
    unchanged, key order included, since `tool | tojson` renders that order;
    the client's own dicts are never modified."""
    no_desc = {"type": "function", "function": {"name": "f", "parameters": {"type": "object"}}}
    no_params = {"type": "function", "function": {"name": "g", "description": "G."}}
    sent = json.dumps([no_desc, no_params, WEATHER])
    out = validate_tools([no_desc, no_params, WEATHER])
    assert out[0]["function"] == {"name": "f", "parameters": {"type": "object"}, "description": ""}
    assert out[1]["function"] == {"name": "g", "description": "G.", "parameters": {}}
    assert json.dumps(out[2]) == json.dumps(WEATHER)
    assert json.dumps([no_desc, no_params, WEATHER]) == sent


def test_a_null_description_or_parameters_reads_as_missing():
    """An explicit null gets the same default as a missing field, in place:
    the Messages and Responses adapters already drop a null before this
    runs, so all three dialects agree."""
    nulls = {"type": "function", "function": {"name": "f", "description": None,
                                              "parameters": None, "strict": True}}
    sent = json.dumps(nulls)
    (out,) = validate_tools([nulls])
    assert json.dumps(out["function"]) == json.dumps(
        {"name": "f", "description": "", "parameters": {}, "strict": True})
    assert json.dumps(nulls) == sent


def test_validate_tools_rejects_garbage_with_a_message():
    for bad in ("get_weather", {"type": "function"}, [42], [{"type": "retrieval"}],
                [{"type": "function"}],
                [{"type": "function", "function": []}],
                [{"type": "function", "function": {}}],
                [{"type": "function", "function": {"name": ""}}],
                [{"type": "function", "function": {"name": "f", "description": 3}}],
                [{"type": "function", "function": {"name": "f", "parameters": "x"}}]):
        with pytest.raises(ValueError, match="tools"):
            validate_tools(bad)


# --- qwen3_5 XML-parameter form (Qwen3.8 / Coder lineage) ------

BASH_TOOL = [{"type": "function", "function": {
    "name": "bash",
    "parameters": {"type": "object", "properties": {
        "command": {"type": "string"},
        "timeout": {"type": "integer"},
        "detach": {"type": "boolean"}}}}}]


def _xml_block(inner):
    return f"<tool_call>\n{inner}\n</tool_call>"


def test_xml_call_single_multiline_param():
    from drinkme.serving.tools import ToolCallScanner
    sc = ToolCallScanner(BASH_TOOL)
    block = _xml_block(
        "<function=bash>\n<parameter=command>\n"
        "find ~/x -type f | head -100; echo '---'; du -sh ~/x\n"
        "</parameter>\n</function>")
    vis, calls = sc.feed("before " + block)
    vis += sc.flush()
    assert vis == "before "
    assert calls == [{"name": "bash", "arguments":
                      {"command": "find ~/x -type f | head -100; echo '---'; du -sh ~/x"}}]


def test_xml_call_schema_coercion_and_string_verbatim():
    from drinkme.serving.tools import ToolCallScanner
    sc = ToolCallScanner(BASH_TOOL)
    block = _xml_block(
        "<function=bash>\n"
        "<parameter=command>\n123\n</parameter>\n"      # string param: stays "123"
        "<parameter=timeout>\n45\n</parameter>\n"        # integer param: 45
        "<parameter=detach>\ntrue\n</parameter>\n"       # boolean param: True
        "</function>")
    _, calls = sc.feed(block)
    assert calls[0]["arguments"] == {"command": "123", "timeout": 45, "detach": True}


def test_xml_call_unknown_param_and_no_schema_stay_strings():
    from drinkme.serving.tools import ToolCallScanner
    sc = ToolCallScanner()  # no schemas at all
    block = _xml_block("<function=go>\n<parameter=n>\n7\n</parameter>\n</function>")
    _, calls = sc.feed(block)
    assert calls == [{"name": "go", "arguments": {"n": "7"}}]


def test_xml_call_chunked_one_byte_at_a_time():
    from drinkme.serving.tools import ToolCallScanner
    sc = ToolCallScanner(BASH_TOOL)
    block = _xml_block("<function=bash>\n<parameter=command>\nls -la\n</parameter>\n</function>")
    vis, calls = "", []
    for ch in "pre " + block + " post":
        v, c = sc.feed(ch)
        vis += v
        calls += c
    vis += sc.flush()
    assert vis == "pre  post"
    assert calls == [{"name": "bash", "arguments": {"command": "ls -la"}}]


def test_xml_malformed_reemerges_verbatim():
    from drinkme.serving.tools import ToolCallScanner
    sc = ToolCallScanner(BASH_TOOL)
    block = _xml_block("<function=bash>\nstray text between\n<parameter=command>\nx\n</parameter>\n</function>")
    vis, calls = sc.feed(block)
    vis += sc.flush()
    assert calls == []
    assert "stray text between" in vis and vis.startswith("<tool_call>")


def test_json_form_still_parses():
    from drinkme.serving.tools import ToolCallScanner
    sc = ToolCallScanner(BASH_TOOL)
    _, calls = sc.feed('<tool_call>\n{"name": "bash", "arguments": {"command": "ls"}}\n</tool_call>')
    assert calls == [{"name": "bash", "arguments": {"command": "ls"}}]


def test_xml_no_arg_call():
    from drinkme.serving.tools import ToolCallScanner
    sc = ToolCallScanner()
    _, calls = sc.feed(_xml_block("<function=ping>\n</function>"))
    assert calls == [{"name": "ping", "arguments": {}}]


# --- the table: the two rows we serve by default ---


def test_default_row_is_what_the_hardcoded_scanner_was():
    from drinkme.serving import tool_formats

    assert tool_formats.DEFAULT == "json"
    # every test above constructs the scanner WITHOUT a format and passes;
    # this pins what that default resolves to
    assert ToolCallScanner().row.name == "json"
    assert ToolCallScanner(BASH_TOOL, "unknown").row.name == "json"
    assert ToolCallScanner(BASH_TOOL, None).row.name == "json"


@pytest.mark.parametrize("tool_format", [None, "json", "qwen-xml"])
def test_both_served_rows_parse_both_body_forms(tool_format):
    """`json` and `qwen-xml` share markers AND parser chain on purpose — a
    Qwen model that emits the other family's body still parses, exactly as
    before the table. Narrowing a row to its own body parser would be a
    behaviour change, not a cleanup."""
    xml = _xml_block("<function=bash>\n<parameter=command>\nls\n</parameter>\n</function>")
    for block, expected in ((CALL, PARSED),
                            (xml, {"name": "bash", "arguments": {"command": "ls"}})):
        sc = ToolCallScanner(BASH_TOOL, tool_format)
        text, calls = sc.feed(block)
        assert text == "" and calls == [expected]
        assert sc.flush() == "" and sc.flush_calls() == []


def test_flush_calls_is_empty_for_the_rows_we_serve():
    """Only a ONE-SIDED row (mistral) resolves a call at end of turn. For
    these two the engines' extra drain is always a no-op — which is why
    adding it could not change the default behaviour."""
    for tool_format in (None, "json", "qwen-xml"):
        sc = ToolCallScanner(BASH_TOOL, tool_format)
        sc.feed('<tool_call>{"name": "x"')  # unterminated
        assert sc.flush() == '<tool_call>{"name": "x"'
        assert sc.flush_calls() == []


def test_split_calls_is_feed_plus_flush():
    from drinkme.serving.tools import split_calls

    assert split_calls("hi " + CALL + " bye", BASH_TOOL) == ("hi  bye", [PARSED])
