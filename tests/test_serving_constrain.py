"""The JSON constraint engine: exactness of the prefix validator (the
dead-end-freedom guarantee), and pick_token's rejection loop over a
hand-built vocab — no model, no GPU, every case checkable by eye."""

import json

import pytest
import torch

from drinkme.serving.constrain import (COMPLETE, FAIL, PARTIAL, JsonConstraint,
                                       pick_token, validate_schema,
                                       validate_supported)
from drinkme.serving.engine import SampleParams

PLAN = {"type": "object",
        "properties": {"name": {"type": "string"},
                       "count": {"type": "integer"},
                       "tags": {"type": "array", "items": {"type": "string"},
                                "maxItems": 2}},
        "required": ["name", "count"],
        "additionalProperties": False}


def st(schema, text):
    return JsonConstraint(schema).status(text)


# ------------------------------------------------------------ the validator --


def test_complete_document():
    assert st(PLAN, '{"name": "x", "count": 3}') == COMPLETE
    assert st(PLAN, ' {"count": 0, "name": ""} ') == COMPLETE  # order-free, ws ok
    assert st(PLAN, '{"name":"a","count":1,"tags":["t"]}') == COMPLETE


def test_partial_prefixes_every_cut():
    """Every prefix of a valid document must be PARTIAL or COMPLETE — one FAIL
    would mean the generator can never produce this document."""
    doc = '{"name": "hi \\n\\u0041", "count": -12, "tags": ["a", "b"]}'
    c = JsonConstraint(PLAN)
    for i in range(len(doc)):
        assert c.status(doc[:i]) == PARTIAL, f"prefix {doc[:i]!r}"
    assert c.status(doc) == COMPLETE


def test_fail_wrong_key_at_first_impossible_char():
    assert st(PLAN, '{"nx') == FAIL         # no property starts "nx"
    assert st(PLAN, '{"na') == PARTIAL      # "name" still possible
    assert st(PLAN, '{"name": "a", "name') == FAIL  # duplicate key


def test_fail_wrong_types_and_missing_required():
    assert st(PLAN, '{"name": 3') == FAIL           # string property, number given
    assert st(PLAN, '{"count": 1.5') == FAIL        # integer forbids fraction
    assert st(PLAN, '{"name": "a"}') == FAIL        # required "count" absent
    assert st(PLAN, '{"tags": "a"') == FAIL         # array property, string given
    assert st(PLAN, '{"tags": ["a", "b", "c"') == FAIL  # maxItems 2


def test_trailing_junk_fails():
    assert st(PLAN, '{"name": "a", "count": 1} x') == FAIL
    assert st(PLAN, '{"name": "a", "count": 1}  ') == COMPLETE  # ws is fine


def test_numbers_grammar():
    num = {"type": "number"}
    assert st(num, "-") == PARTIAL
    assert st(num, "-1.5e+3") == COMPLETE
    assert st(num, "1e") == PARTIAL
    assert st(num, "01") == FAIL          # leading zero
    assert st(num, "1.") == PARTIAL
    assert st(num, "1.e3") == FAIL        # digit required after the point
    assert st({"type": "integer"}, "-0") == COMPLETE


def test_enum_and_const():
    e = {"enum": ["alpha", "beta", 12, True]}
    assert st(e, '"alpha"') == COMPLETE
    assert st(e, '"alp') == PARTIAL
    assert st(e, '"alx') == FAIL
    # FORM rule: escapes in constrained strings only where JSON
    # REQUIRES them. 'p' is printable, so escape-spelling it is banned at the
    # backslash — the only legal path through "alpha" is its literal chars
    # (a greedy model starved its budget hex-spelling a banned word's letters)
    assert st(e, '"al\\') == FAIL
    assert st(e, '"al\\u0070') == FAIL
    e2 = {"enum": ['a"b', "a\nb"]}
    assert st(e2, '"a\\') == PARTIAL        # quote/control DO need escaping
    assert st(e2, '"a\\"b"') == COMPLETE
    assert st(e2, '"a\\n') == PARTIAL
    assert st(e2, '"a\\u000ab"') == COMPLETE  # \u form of a control: allowed
    assert st(e2, '"a\\u0062') == FAIL      # but not of printable 'b'
    assert st(e, "1") == PARTIAL            # extendable to 12
    assert st(e, "12") == COMPLETE
    assert st(e, "13") == FAIL
    assert st(e, "true") == COMPLETE
    assert st(e, "fals") == FAIL            # False not a member
    assert st({"const": "yes"}, '"yes"') == COMPLETE
    assert st({"const": "yes"}, '"no"') == FAIL


UNICODE_DIGITS = ["١٢", "１２", "²"]  # Arabic-Indic, fullwidth, superscript


def test_unicode_digits_rejected_top_level():
    """Str.isdigit() accepts far more than ASCII 0-9, so the
    grammar could complete on text json.loads() then refuses. Every accepted
    completion must round-trip through json.loads; every one of these must
    be REJECTED by both the incremental and reference parsers."""
    num = {"type": "number"}
    for digits in UNICODE_DIGITS:
        text = "1" + digits
        with pytest.raises(json.JSONDecodeError):
            json.loads(text)
        c = JsonConstraint(num)
        assert c.status(text) == FAIL, f"status accepted {text!r}"
        assert c.status_reference(text) == FAIL, f"status_reference accepted {text!r}"


def test_unicode_digits_rejected_inside_object_value():
    """Same defect, reached through structured output: a number nested as an
    object property value."""
    schema = {"type": "object", "properties": {"n": {"type": "number"}},
              "required": ["n"]}
    for digits in UNICODE_DIGITS:
        text = '{"n": 1' + digits + '}'
        with pytest.raises(json.JSONDecodeError):
            json.loads(text)
        c = JsonConstraint(schema)
        assert c.status(text) == FAIL, f"status accepted {text!r}"
        assert c.status_reference(text) == FAIL, f"status_reference accepted {text!r}"


def test_enum_bypass_sibling_type_rejected():
    """Enum/const dispatch matched by literal spelling alone
    and never re-checked sibling `type`, so a non-conforming member could
    complete. validate_schema must filter it out of the schema BEFORE
    generation, for both parsers."""
    schema = {"type": "string", "enum": ["ok", 7]}
    c = JsonConstraint(schema)
    assert c.status('"ok"') == COMPLETE
    assert c.status_reference('"ok"') == COMPLETE
    assert c.status("7") == FAIL, "7 is not a string — sibling type bypassed"
    assert c.status_reference("7") == FAIL
    assert schema["enum"] == ["ok"]  # filtered in place


def test_const_unsatisfiable_with_required_refused():
    """The other const case: a const value that cannot itself satisfy a
    sibling `required` must refuse the whole schema as unsatisfiable, rather
    than manufacture a fake valid instance by ignoring `required`."""
    schema = {"type": "object", "required": ["x"], "const": {}}
    with pytest.raises(ValueError, match="unsatisfiable"):
        validate_schema(schema)
    with pytest.raises(ValueError, match="unsatisfiable"):
        JsonConstraint(schema)


def test_enum_all_members_filtered_is_unsatisfiable():
    schema = {"type": "boolean", "enum": [1, "x"]}
    with pytest.raises(ValueError, match="unsatisfiable"):
        JsonConstraint(schema)


def test_enum_bypass_nested_under_properties():
    schema = {"type": "object",
              "properties": {"color": {"type": "string", "enum": ["red", 5]}}}
    c = JsonConstraint(schema)
    assert c.status('{"color": "red"}') == COMPLETE
    assert c.status_reference('{"color": "red"}') == COMPLETE
    assert c.status('{"color": 5}') == FAIL
    assert c.status_reference('{"color": 5}') == FAIL


def test_enum_bypass_nested_under_items():
    schema = {"type": "array", "items": {"type": "string", "enum": ["a", 1]}}
    c = JsonConstraint(schema)
    assert c.status('["a"]') == COMPLETE
    assert c.status_reference('["a"]') == COMPLETE
    assert c.status("[1]") == FAIL
    assert c.status_reference("[1]") == FAIL


def test_enum_const_independent_oracle():
    """Parser-to-parser agreement (status vs status_reference) is not
    independent evidence — the reference parser carries the exact same bug
    when it exists. Compare acceptance against the `jsonschema` package
    instead, over a table where enum/const sits beside a sibling
    constraint. Skipped if jsonschema is not importable (test-only oracle,
    never a runtime dependency)."""
    jsonschema = pytest.importorskip("jsonschema")
    import copy

    TABLE = [
        ({"type": "string", "enum": ["ok", 7]}, '"ok"'),
        ({"type": "string", "enum": ["ok", 7]}, "7"),
        ({"type": "object", "properties": {"tag": {"const": "yes"}},
          "required": ["tag"]}, '{"tag": "yes"}'),
        ({"type": "object", "properties": {"tag": {"const": "yes"}},
          "required": ["tag"]}, '{"tag": "no"}'),
        ({"type": "array", "items": {"type": "integer", "enum": [1, 2, "x"]}},
         "[1, 2]"),
        ({"type": "array", "items": {"type": "integer", "enum": [1, 2, "x"]}},
         '["x"]'),
        ({"type": "integer", "enum": [True, 1, 2]}, "1"),
        ({"type": "integer", "enum": [True, 1, 2]}, "true"),
    ]
    for schema, text in TABLE:
        instance = json.loads(text)
        try:
            jsonschema.validate(instance, copy.deepcopy(schema))
            oracle = COMPLETE
        except jsonschema.ValidationError:
            oracle = FAIL
        c = JsonConstraint(copy.deepcopy(schema))
        assert c.status(text) == oracle, (schema, text)
        assert c.status_reference(text) == oracle, (schema, text)


def test_type_lists_and_null():
    s = {"type": ["string", "null"]}
    assert st(s, "null") == COMPLETE
    assert st(s, '"x"') == COMPLETE
    assert st(s, "nul") == PARTIAL
    assert st(s, "3") == FAIL


def test_any_value_when_no_type():
    anyv = {}
    for doc in ('"s"', "3.5", "true", "null", '{"a": [1, {"b": null}]}'):
        assert st(anyv, doc) == COMPLETE


def test_additional_properties_schema():
    s = {"type": "object", "additionalProperties": {"type": "integer"}}
    assert st(s, '{"whatever": 3}') == COMPLETE
    assert st(s, '{"whatever": "s"') == FAIL
    assert st(s, "{}") == COMPLETE


def test_min_items():
    s = {"type": "array", "items": {"type": "integer"}, "minItems": 2}
    assert st(s, "[1]") == FAIL
    assert st(s, "[1, 2]") == COMPLETE
    assert st(s, "[") == PARTIAL


def test_string_escapes_and_controls():
    s = {"type": "string"}
    assert st(s, '"a\\qb"') == FAIL       # unknown escape
    assert st(s, '"a\nb"') == FAIL        # raw control char
    assert st(s, '"a\\u00zz"') == FAIL    # bad hex
    assert st(s, '"a\\u00"') == FAIL      # hex cut short by the closing quote
    assert st(s, '"a�') == FAIL      # partial UTF-8 marker is never valid


def test_validate_supported_refuses_by_name():
    for bad, kw in (({"pattern": "x"}, "pattern"),
                    ({"anyOf": []}, "anyOf"),
                    ({"minimum": 0}, "minimum"),
                    ({"properties": {"a": {"$ref": "#/x"}}}, "$ref"),
                    ({"items": {"format": "date"}}, "format")):
        with pytest.raises(ValueError) as e:
            validate_supported(bad)
        assert kw in str(e.value)
    validate_supported(PLAN)  # the supported subset passes
    validate_supported({"title": "t", "description": "d", "type": "object"})


def test_validate_supported_rejects_non_object_properties():
    # The exact reproduction: `{"type": "object",
    # "required": ["x"], "const": {}}` accepted `{}` by ignoring `required`
    # is a separate case; THIS one is `properties` itself being a
    # non-object, which unchecked reaches a bare `.values()` on a list —
    # AttributeError, not ValueError, so http.py's `except ValueError`
    # boundary would miss it and it would escape as a 500/traceback instead
    # of a 400 naming the field.
    with pytest.raises(ValueError, match="properties"):
        validate_supported({"type": "object", "properties": []})


# ------------------------------------------------------------- pick_token --


VOCAB = {0: "{", 1: '"name"', 2: ": ", 3: '"x"', 4: ", ", 5: '"count"',
         6: "7", 7: "}", 8: "hello", 9: "<think>", 10: ""}
EOS = frozenset([11])


def decode(ids):
    return "".join(VOCAB.get(i, "") for i in ids)


def greedy_seq(constraint, prefer):
    """Greedy-decode a whole document: `prefer` ranks the vocab (the "model's"
    preference); pick_token must ban its way to the valid one every step."""
    params = SampleParams(temperature=0.0)
    gen: list[int] = []
    for _ in range(32):
        logits = torch.full((12,), -10.0)
        for rank, tok in enumerate(prefer):
            logits[tok] = 5.0 - rank * 0.1
        tok = pick_token(logits, params, None, gen, gen, EOS, decode, constraint)
        if tok in EOS:
            return decode(gen), True
        gen.append(tok)
    return decode(gen), False


def test_pick_token_forces_valid_document():
    """The model 'wants' to say hello/<think>/EOS at every step; the constraint
    walks it to the only valid document in the vocab and only then allows EOS."""
    c = JsonConstraint(PLAN)
    text, ended = greedy_seq(c, prefer=[8, 9, 11, 10, 7, 4, 2, 3, 6, 5, 1, 0])
    assert ended and c.status(text) == COMPLETE
    import json

    assert json.loads(text) == {"name": "x", "count": 7}  # order is schema-free


def test_pick_token_bans_invisible_tokens():
    """Token 10 decodes to nothing (a special token under skip_special);
    accepting it would loop forever — it must be banned, not accepted."""
    c = JsonConstraint({"type": "object"})
    logits = torch.full((12,), -10.0)
    logits[10] = 5.0  # invisible token is the favorite
    logits[0] = 4.0   # "{" is second
    tok = pick_token(logits, SampleParams(temperature=0.0), None, [], [], EOS,
                     decode, c)
    assert tok == 0


def test_pick_token_eos_only_when_complete():
    c = JsonConstraint({"type": "object"})
    params = SampleParams(temperature=0.0)
    logits = torch.full((12,), -10.0)
    logits[11] = 5.0  # EOS is the favorite
    logits[0] = 4.0
    assert pick_token(logits, params, None, [], [], EOS, decode, c) == 0  # not yet
    assert pick_token(logits, params, None, [0, 7], [0, 7], EOS, decode, c) == 11


def test_pick_token_dead_end_raises():
    c = JsonConstraint({"type": "object"})
    logits = torch.full((3,), 0.0)  # tiny vocab: "{", '"name"', ": " — no close
    with pytest.raises(RuntimeError, match="dead-end"):
        # gen so far is "hello" (invalid); every candidate fails, EOS not in vocab
        pick_token(logits, SampleParams(temperature=0.0), None, [8], [8],
                   frozenset([99]), decode, c)


def test_pick_token_sampling_stays_valid():
    """Under real (seeded) sampling the result is still always schema-valid."""
    c = JsonConstraint({"type": "object", "additionalProperties": False})
    params = SampleParams(temperature=1.5, seed=3)
    g = torch.Generator().manual_seed(3)
    gen: list[int] = []
    for _ in range(8):
        logits = torch.zeros(12)
        tok = pick_token(logits, params, g, gen, gen, EOS, decode, c)
        if tok in EOS:
            break
        gen.append(tok)
    text = decode(gen)
    assert c.status(text) in (PARTIAL, COMPLETE)
    assert text.startswith("{")


def test_whitespace_runs_are_capped():
    """Whitespace is always legal JSON, so an unconstrained-preamble model
    pours banned probability into it; runs beyond MAX_WS_RUN are FAIL so
    generation cannot stall on formatting (live smoke)."""
    c = JsonConstraint({"type": "object"})
    assert c.status("   {") == PARTIAL          # 3 = at the cap, fine
    assert c.status("    ") == FAIL             # 4 in a row = banned
    assert c.status('{"a": 1}   ') == COMPLETE  # trailing at cap ok
    assert c.status('{ \n\t "a"') == FAIL       # mixed run past cap
    # a doc that never exceeds the cap parses exactly as before
    assert c.status('{ "a" : [1, 2] }') == COMPLETE
