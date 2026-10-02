"""Grammar-constrained JSON output: response_format done honestly.

The guarantee: when a request carries response_format (json_object or
json_schema), the SAMPLED TOKEN STREAM is constrained so the finished text
always parses as JSON and validates against the schema — never
prompt-and-pray wearing the spec's field name.

Mechanism: exact rejection sampling. Each step samples from the model's
distribution (through the normal temperature/top-k/top-p pipeline); a
candidate whose decoded text can no longer extend to a valid document is
BANNED (logit -> -inf) and the step resamples. Renormalizing over the
surviving support is mathematically identical to masking the whole vocab up
front, at a fraction of the per-step cost (JSON output rarely rejects — the
model wants to write JSON once it's mid-object). Greedy picks the best VALID
token by construction. EOS is a candidate like any other: allowed exactly
when the text is a COMPLETE document.

The validator is EXACT, not optimistic: prefix_ok(text) is true iff some
continuation completes a valid document. Optimism would dead-end generation
(an accepted prefix nothing can extend); pessimism would bias output. The
one guard for the unreachable: if every token including EOS is banned,
raise — a 500 says what happened, silence never does.

Supported JSON Schema subset (validate_schema refuses the rest BY NAME at
the HTTP boundary, so a client learns at request time, not from quietly
unvalidated output): type (single or list), properties, required,
additionalProperties (bool or schema), items, minItems, maxItems, enum,
const. Annotations (title, description, default, examples) pass through.
Known v1 edges, chosen not hidden: enum/const numbers and booleans match
their canonical rendering (json.dumps), not exotic-but-equal spellings;
multi-byte characters must arrive as whole chars or \\u escapes (a partial
UTF-8 token decodes to U+FFFD, which is never valid JSON text, so it's
banned — the model routes through escapes); thinking preambles are banned by
construction (a "<" is not a valid JSON prefix), so response_format implies
no <think> block.

COST: the validator is INCREMENTAL. Re-decoding the whole output and
re-parsing it from byte 0 for every candidate token of every step would be
quadratic in the output length, per candidate — measured as
the constrained path's dominant per-token cost. A JsonCursor carries the
parse state of the accepted text forward and forks it in O(nesting depth)
to judge a candidate. The recursive-descent
parser stays in the file as status_reference(), the oracle the incremental
machine is fuzzed against — the verdicts did not move, only their price.

Pure text + stdlib at module level (http.py imports this for boundary
validation); torch is imported inside pick_token only, sampling.py's pattern.
"""

from __future__ import annotations

import json
from typing import Callable

FAIL, PARTIAL, COMPLETE = "fail", "partial", "complete"

_SUPPORTED = {"type", "properties", "required", "additionalProperties", "items",
              "minItems", "maxItems", "enum", "const",
              # annotations: legal everywhere, constrain nothing
              "title", "description", "default", "examples"}

_ESCAPES = {"n": "\n", "t": "\t", "r": "\r", "b": "\b", "f": "\f",
            '"': '"', "\\": "\\", "/": "/"}

_WS = " \t\n\r"


def _json_type(value) -> str:
    """The JSON type name of a Python value already decoded from JSON (so the
    only types possible are the ones json.loads() produces). bool is checked
    before int — Python's bool is an int subclass. A float with no fractional
    part (3.0) is reported "integer": JSON draws no wire distinction, and
    json.dumps(3.0) round-trips through a number literal a "type": "integer"
    schema must accept."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "integer" if value.is_integer() else "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    return "object"


def _type_matches(vt: str, t: str) -> bool:
    return vt == t or (vt == "integer" and t == "number")


def _conforms(value, schema: dict) -> bool:
    """Does `value` satisfy `schema`, restricted to the keywords this module
    enforces elsewhere (type, properties/required/additionalProperties,
    items/minItems/maxItems)? `schema` here is always a sibling schema with
    enum/const already peeled off — this is the check the enum/const literal
    dispatch in _value/_Val never performs: it matches a
    candidate purely by spelling, so any sibling constraint has to be
    enforced by filtering the enum/const values themselves, ahead of time."""
    types = _types_of(schema)
    if types is not None:
        vt = _json_type(value)
        if not any(_type_matches(vt, t) for t in types):
            return False
    if isinstance(value, dict):
        props = schema.get("properties", {})
        required = schema.get("required", [])
        addl = schema.get("additionalProperties", True)
        if any(k not in value for k in required):
            return False
        for k, v in value.items():
            if k in props:
                if not _conforms(v, props[k]):
                    return False
            elif addl is False:
                return False
            elif isinstance(addl, dict):
                if not _conforms(v, addl):
                    return False
    elif isinstance(value, list):
        items = schema.get("items", {})
        lo = schema.get("minItems", 0)
        hi = schema.get("maxItems")
        if len(value) < lo or (hi is not None and len(value) > hi):
            return False
        if not all(_conforms(v, items) for v in value):
            return False
    return True


def validate_supported(schema: dict) -> None:
    """Refuse, by name, any keyword the constraint engine does not enforce —
    accepting it silently would serve unvalidated output under a validated
    flag. Raises ValueError naming the keyword.

    NOTE: this function's recursive self-calls (by bare name, not `self`)
    are relied on by tests/test_hotloop_equivalence.py::
    test_schema_is_validated_once_per_request_not_twice, which monkeypatches
    the module attribute with a ONE-ARGUMENT spy and counts how many times
    recursion re-enters it. Do not add a parameter here — that is why the
    sibling-constraint gate below is a separate function with its own
    recursion, and why callers wanting the FULL check call validate_schema(),
    not this one directly."""
    if not isinstance(schema, dict):
        raise ValueError("response_format schema must be a JSON object")
    for k, v in schema.items():
        if k not in _SUPPORTED:
            raise ValueError(
                f"response_format: unsupported JSON Schema keyword {k!r} — "
                f"supported: {', '.join(sorted(_SUPPORTED - {'title', 'description', 'default', 'examples'}))}")
        if k == "properties":
            # `schema.properties: []` would otherwise reach a bare `.values()`
            # on a list (AttributeError) instead of this ValueError, which
            # http.py's boundary already catches and 400s by name.
            if not isinstance(v, dict):
                raise ValueError("response_format: 'properties' must be an object")
            for sub in v.values():
                validate_supported(sub)
        elif k == "items":
            validate_supported(v)
        elif k == "additionalProperties" and isinstance(v, dict):
            validate_supported(v)


def _filter_enum_const(schema: dict, path: str) -> None:
    """The SIBLING-CONSTRAINT gate: _value/_Val dispatch an
    enum/const schema straight to a literal matcher and never re-check a
    sibling type/required/properties/items on the value that matches — so
    {"type": "string", "enum": ["ok", 7]} accepted 7, and a const that can't
    itself satisfy `required` manufactured a fake valid instance instead of
    refusing. Walks the same tree validate_supported does (properties/items/
    additionalProperties), but as its OWN recursion, so it can carry a JSON
    path without touching validate_supported's pinned single-argument call
    signature (see its docstring). Only reachable once validate_supported
    has already confirmed every node is a dict of supported keywords.

    Mutates `schema` in place, keeping only conforming enum members — const
    has none to keep, so it either conforms or the schema is unsatisfiable.
    Raises ValueError naming the JSON path when nothing conforms."""
    for k, v in schema.items():
        if k == "properties":
            for pk, sub in v.items():
                _filter_enum_const(sub, f"{path}.properties.{pk}")
        elif k == "items":
            _filter_enum_const(v, f"{path}.items")
        elif k == "additionalProperties" and isinstance(v, dict):
            _filter_enum_const(v, f"{path}.additionalProperties")
    if "const" in schema or "enum" in schema:
        siblings = {k: v for k, v in schema.items() if k not in ("enum", "const")}
        if "const" in schema:
            if not _conforms(schema["const"], siblings):
                raise ValueError(
                    f"response_format: schema at {path} is unsatisfiable — its "
                    f"const value does not satisfy the schema's own sibling constraints")
        else:
            kept = [m for m in schema["enum"] if _conforms(m, siblings)]
            if not kept:
                raise ValueError(
                    f"response_format: schema at {path} is unsatisfiable — no "
                    f"enum member satisfies the schema's own sibling constraints")
            schema["enum"] = kept


def validate_schema(schema: dict) -> None:
    """The full response_format validation gate: validate_supported's
    by-name keyword refusal, PLUS the sibling-constraint filter/refusal for
    enum/const (_filter_enum_const above). Call THIS at a
    request boundary, not validate_supported alone — the split exists only
    to keep validate_supported's recursive call signature pinned for the
    hot-loop equivalence spy test."""
    validate_supported(schema)
    _filter_enum_const(schema, "$")


class _Partial(Exception):
    """Ran out of input mid-construct — the text is a valid prefix."""


class _Fail(Exception):
    """Mismatch before end of input — no continuation can fix it."""


class _P:
    __slots__ = ("t", "i", "n")

    def __init__(self, text: str):
        self.t, self.i, self.n = text, 0, len(text)

    def eoi(self) -> bool:
        return self.i >= self.n

    def peek(self) -> str:
        if self.eoi():
            raise _Partial()
        return self.t[self.i]

    def ws(self) -> None:
        while not self.eoi() and self.t[self.i] in _WS:
            self.i += 1


def _canon(value) -> str:
    """Canonical rendering for non-string enum/const members."""
    return json.dumps(value)


def _is_digit(c: str) -> bool:
    """ASCII '0'-'9' ONLY. str.isdigit() also returns True
    for Arabic-Indic ١٢, fullwidth １２, superscript ², etc. — none of which
    are valid JSON grammar — so using it here let the constraint accept text
    that json.loads() then rejects. Every digit test in this module (the
    reference number parser, the incremental number DFA, and both value
    dispatchers) goes through this predicate instead."""
    return "0" <= c <= "9"


def _literal(p: _P, lit: str) -> None:
    """Match an exact literal ('true', 'false', 'null', or a canonical
    number/bool rendering). Prefix at EOI -> PARTIAL."""
    end = min(p.n, p.i + len(lit))
    got = p.t[p.i:end]
    if not lit.startswith(got):
        raise _Fail()
    if len(got) < len(lit):
        raise _Partial()
    p.i += len(lit)


def _string_free(p: _P) -> str:
    """A generic JSON string (opening quote at p.i). Returns decoded value."""
    if p.peek() != '"':
        raise _Fail()
    p.i += 1
    out: list[str] = []
    while True:
        c = p.peek()
        if c == '"':
            p.i += 1
            return "".join(out)
        if c == "\\":
            p.i += 1
            e = p.peek()
            if e == "u":
                p.i += 1
                hexs = p.t[p.i:p.i + 4]
                if any(h not in "0123456789abcdefABCDEF" for h in hexs):
                    raise _Fail()
                if len(hexs) < 4:
                    raise _Partial()
                out.append(chr(int(hexs, 16)))
                p.i += 4
            elif e in _ESCAPES:
                out.append(_ESCAPES[e])
                p.i += 1
            else:
                raise _Fail()
        elif ord(c) < 0x20 or c == "�":  # raw control / partial UTF-8
            raise _Fail()
        else:
            out.append(c)
            p.i += 1


def _string_constrained(p: _P, values: list[str]) -> str:
    """A JSON string whose DECODED value must be one of `values`. Incremental:
    bans a divergent key/enum at the first impossible character, including
    inside a half-written escape (so an accepted prefix can always still
    complete — the exactness the module docstring promises)."""
    if p.peek() != '"':
        raise _Fail()
    p.i += 1
    out: list[str] = []

    def prefix_alive() -> bool:
        d = "".join(out)
        return any(v.startswith(d) for v in values)

    def next_chars() -> set[str]:
        d = "".join(out)
        return {v[len(d)] for v in values if v.startswith(d) and len(v) > len(d)}

    while True:
        if p.eoi():
            if prefix_alive():
                raise _Partial()
            raise _Fail()
        c = p.t[p.i]
        if c == '"':
            p.i += 1
            if "".join(out) in values:
                return "".join(out)
            raise _Fail()
        if c == "\\":
            # FORM constraint, same move as MAX_WS_RUN: escapes are allowed
            # only where JSON REQUIRES them (quote, backslash, controls).
            # Escape-spelling a printable char (c for 'c') is legal JSON
            # but 6 tokens per letter — a greedy model whose preferred word-
            # token was banned falls into it and starves its budget spelling
            # "confident" one hex quad at a time (live verify).
            # Every VALUE stays expressible: printables go in literally.
            need_escape = {ch for ch in next_chars()
                           if ch in '"\\' or ord(ch) < 0x20}
            if not need_escape:
                raise _Fail()
            if p.i + 1 >= p.n:
                raise _Partial()
            e = p.t[p.i + 1]
            if e == "u":
                hexs = p.t[p.i + 2:p.i + 6]
                if any(h not in "0123456789abcdefABCDEF" for h in hexs):
                    raise _Fail()
                if len(hexs) < 4:
                    if any(("%04x" % ord(ch)).startswith(hexs.lower())
                           for ch in need_escape):
                        raise _Partial()
                    raise _Fail()
                ch = chr(int(hexs, 16))
                if ch not in need_escape:
                    raise _Fail()
                out.append(ch)
                p.i += 6
            elif e in _ESCAPES:
                if _ESCAPES[e] not in need_escape:
                    raise _Fail()
                out.append(_ESCAPES[e])
                p.i += 2
            else:
                raise _Fail()
        elif ord(c) < 0x20 or c == "�":
            raise _Fail()
        else:
            out.append(c)
            p.i += 1
        if not prefix_alive():
            raise _Fail()


def _number(p: _P, integer: bool) -> None:
    """JSON number grammar. integer=True forbids fraction/exponent outright
    (they could never extend into an integer)."""
    t, n = p.t, p.n
    i = p.i
    if i < n and t[i] == "-":
        i += 1
    if i >= n:
        raise _Partial()
    if not _is_digit(t[i]):
        raise _Fail()
    if t[i] == "0" and i + 1 < n and _is_digit(t[i + 1]):
        raise _Fail()  # leading zero
    while i < n and _is_digit(t[i]):
        i += 1
    if i < n and t[i] == ".":
        if integer:
            raise _Fail()
        i += 1
        if i >= n:
            raise _Partial()
        if not _is_digit(t[i]):
            raise _Fail()
        while i < n and _is_digit(t[i]):
            i += 1
    if i < n and t[i] in "eE":
        if integer:
            raise _Fail()
        i += 1
        if i < n and t[i] in "+-":
            i += 1
        if i >= n:
            raise _Partial()
        if not _is_digit(t[i]):
            raise _Fail()
        while i < n and _is_digit(t[i]):
            i += 1
    p.i = i


def _types_of(sch: dict) -> list[str] | None:
    t = sch.get("type")
    if t is None:
        return None  # any
    return t if isinstance(t, list) else [t]


def _value(p: _P, sch: dict) -> None:
    p.ws()
    c = p.peek()
    if "const" in sch or "enum" in sch:
        members = [sch["const"]] if "const" in sch else list(sch["enum"])
        if c == '"':
            strs = [m for m in members if isinstance(m, str)]
            if not strs:
                raise _Fail()
            _string_constrained(p, strs)
            return
        # non-string members: canonical renderings, first-char dispatch
        lits = [_canon(m) for m in members if not isinstance(m, str)]
        alive = [l for l in lits if l.startswith(c)]
        if not alive:
            raise _Fail()
        # longest-match: try each; PARTIAL if any could continue
        errs = []
        for l in alive:
            q = _P(p.t)
            q.i = p.i
            try:
                _literal(q, l)
                p.i = q.i
                return
            except (_Partial, _Fail) as e:
                errs.append(e)
        if any(isinstance(e, _Partial) for e in errs):
            raise _Partial()
        raise _Fail()
    types = _types_of(sch)

    def has(t: str) -> bool:
        return types is None or t in types

    if c == "{":
        if not has("object"):
            raise _Fail()
        _object(p, sch)
    elif c == "[":
        if not has("array"):
            raise _Fail()
        _array(p, sch)
    elif c == '"':
        if not has("string"):
            raise _Fail()
        _string_free(p)
    elif c == "t":
        if not has("boolean"):
            raise _Fail()
        _literal(p, "true")
    elif c == "f":
        if not has("boolean"):
            raise _Fail()
        _literal(p, "false")
    elif c == "n":
        if not has("null"):
            raise _Fail()
        _literal(p, "null")
    elif c == "-" or _is_digit(c):
        if has("number"):
            _number(p, integer=False)
        elif has("integer"):
            _number(p, integer=True)
        else:
            raise _Fail()
    else:
        raise _Fail()


def _object(p: _P, sch: dict) -> None:
    props: dict = sch.get("properties", {})
    addl = sch.get("additionalProperties", True)
    required = set(sch.get("required", []))
    if p.peek() != "{":
        raise _Fail()
    p.i += 1
    seen: set[str] = set()
    p.ws()
    if p.peek() == "}":
        if required - seen:
            raise _Fail()
        p.i += 1
        return
    while True:
        p.ws()
        if addl is False:
            remaining = [k for k in props if k not in seen]
            key = _string_constrained(p, remaining)
        else:
            key = _string_free(p)
            if key in seen:
                raise _Fail()  # duplicate key
        seen.add(key)
        p.ws()
        if p.peek() != ":":
            raise _Fail()
        p.i += 1
        vsch = props.get(key)
        if vsch is None:
            vsch = addl if isinstance(addl, dict) else {}
        _value(p, vsch)
        p.ws()
        c = p.peek()
        if c == "}":
            if required - seen:
                raise _Fail()
            p.i += 1
            return
        if c != ",":
            raise _Fail()
        p.i += 1


def _array(p: _P, sch: dict) -> None:
    items = sch.get("items", {})
    lo = sch.get("minItems", 0)
    hi = sch.get("maxItems")
    if p.peek() != "[":
        raise _Fail()
    p.i += 1
    count = 0
    p.ws()
    if p.peek() == "]":
        if count < lo:
            raise _Fail()
        p.i += 1
        return
    while True:
        if hi is not None and count >= hi:
            raise _Fail()  # one item too many
        _value(p, items)
        count += 1
        p.ws()
        c = p.peek()
        if c == "]":
            if count < lo:
                raise _Fail()
            p.i += 1
            return
        if c != ",":
            raise _Fail()
        p.i += 1

# ------------------------------------------------- the incremental parser --
#
# THE SAME GRAMMAR, FED ONE CHARACTER AT A TIME. Re-parsing the entire
# output from byte 0 for EVERY candidate token of EVERY step is O(n^2)
# across a generation — measured as the constrained path's
# dominant per-token cost. The frames below
# are an explicit-stack transcription of the recursive descent above; a
# machine CLONES in O(nesting depth), so probing a candidate costs that
# candidate's own delta instead of the whole document.
#
# The recursive-descent parser stays in this file as the REFERENCE ORACLE. It
# is the definition of the verdicts — tests/test_serving_constrain.py fuzzes
# the machine against it over generated documents, every prefix of them, and
# mutations, and JsonConstraint.status_reference() is the same entry point the
# fuzz uses. Each frame names the function it transcribes; when the two
# disagree the reference is right and the frame is broken.

_POP = "pop"  # finish(): this frame would COMPLETE if the input ended here
_HEX = "0123456789abcdefABCDEF"


class _Frame:
    """One open construct. step() consumes a character (True) or hands it back
    for the frame that replaced/uncovered it (False); finish() answers what
    end-of-input would mean here."""

    __slots__ = ()

    def copy(self) -> "_Frame":
        raise NotImplementedError

    def step(self, c: str, m: "_Machine") -> bool:
        raise NotImplementedError

    def at_eoi(self) -> str:
        """FAIL, PARTIAL, or _POP (this frame would complete). No mutation."""
        return PARTIAL

    def finish(self, m: "_Machine") -> None:
        """Perform the _POP at_eoi promised. Only ever called on a clone."""
        raise AssertionError("frame cannot finish at end of input")

    def child(self, value, m: "_Machine") -> None:  # only containers get one
        raise AssertionError("frame took a child it cannot own")


class _Val(_Frame):
    """`_value`: skip whitespace, then let the first character choose the
    concrete frame. Replaces itself with that frame (the reference returns as
    soon as its sub-parser does, so nothing survives underneath)."""

    __slots__ = ("sch",)

    def __init__(self, sch: dict):
        self.sch = sch

    def copy(self):
        return _Val(self.sch)

    def step(self, c, m):
        if c in _WS:
            return True  # p.ws()
        sch = self.sch
        if "const" in sch or "enum" in sch:
            members = [sch["const"]] if "const" in sch else list(sch["enum"])
            if c == '"':
                strs = [x for x in members if isinstance(x, str)]
                if not strs:
                    raise _Fail()
                m.replace(_StrC(strs, False))
                return False
            lits = [_canon(x) for x in members if not isinstance(x, str)]
            alive = [l for l in lits if l.startswith(c)]
            if not alive:
                raise _Fail()
            m.replace(_Lit(alive))
            return False
        types = _types_of(sch)

        def has(t: str) -> bool:
            return types is None or t in types

        if c == "{":
            if not has("object"):
                raise _Fail()
            m.replace(_Obj(sch))
        elif c == "[":
            if not has("array"):
                raise _Fail()
            m.replace(_Arr(sch))
        elif c == '"':
            if not has("string"):
                raise _Fail()
            m.replace(_StrF(False))
        elif c == "t":
            if not has("boolean"):
                raise _Fail()
            m.replace(_Lit(["true"]))
        elif c == "f":
            if not has("boolean"):
                raise _Fail()
            m.replace(_Lit(["false"]))
        elif c == "n":
            if not has("null"):
                raise _Fail()
            m.replace(_Lit(["null"]))
        elif c == "-" or _is_digit(c):
            if has("number"):
                m.replace(_Num(False))
            elif has("integer"):
                m.replace(_Num(True))
            else:
                raise _Fail()
        else:
            raise _Fail()
        return False


class _Obj(_Frame):
    """`_object`. States: 0 the opening brace, 1 after it (ws, '}' or a key),
    2 a key in flight, 3 after the key (ws, ':'), 4 a value in flight, 5 after
    the value (ws, '}' or ','), 6 after a comma (ws, then a key — never '}':
    the reference's loop demands a key there, so a trailing comma fails)."""

    __slots__ = ("props", "addl", "required", "seen", "st", "key")

    def __init__(self, sch: dict):
        self.props = sch.get("properties", {})
        self.addl = sch.get("additionalProperties", True)
        self.required = set(sch.get("required", []))
        self.seen: set = set()
        self.st = 0
        self.key = None

    def copy(self):
        o = _Obj.__new__(_Obj)
        o.props, o.addl, o.required = self.props, self.addl, self.required
        o.seen, o.st, o.key = set(self.seen), self.st, self.key
        return o

    def _push_key(self, m):
        if self.addl is False:
            remaining = [k for k in self.props if k not in self.seen]
            m.push(_StrC(remaining, True))
        else:
            m.push(_StrF(True))
        self.st = 2

    def _close(self, m):
        if self.required - self.seen:
            raise _Fail()
        m.pop()

    def step(self, c, m):
        st = self.st
        if st == 0:
            if c != "{":
                raise _Fail()
            self.st = 1
            return True
        if st == 1:
            if c in _WS:
                return True
            if c == "}":
                self._close(m)
                return True
            self._push_key(m)
            return False
        if st == 3:
            if c in _WS:
                return True
            if c != ":":
                raise _Fail()
            vsch = self.props.get(self.key)
            if vsch is None:
                vsch = self.addl if isinstance(self.addl, dict) else {}
            m.push(_Val(vsch))
            self.st = 4
            return True
        if st == 5:
            if c in _WS:
                return True
            if c == "}":
                self._close(m)
                return True
            if c != ",":
                raise _Fail()
            self.st = 6
            return True
        if st == 6:
            if c in _WS:
                return True
            self._push_key(m)
            return False
        raise AssertionError("object frame stepped while a child was open")

    def child(self, value, m):
        if self.st == 2:
            if self.addl is not False and value in self.seen:
                raise _Fail()  # duplicate key (the free-key branch's check)
            self.seen.add(value)
            self.key = value
            self.st = 3
        else:
            self.st = 5


class _Arr(_Frame):
    """`_array`. States: 0 the opening bracket, 1 after it (ws, ']' or the
    first item), 2 an item in flight, 3 after an item (ws, ']' or ','). The
    maxItems refusal fires the moment a comma is consumed, exactly where the
    reference's loop-top check sits — so `[1,2,` with maxItems 2 is FAIL, not
    PARTIAL."""

    __slots__ = ("items", "lo", "hi", "count", "st")

    def __init__(self, sch: dict):
        self.items = sch.get("items", {})
        self.lo = sch.get("minItems", 0)
        self.hi = sch.get("maxItems")
        self.count = 0
        self.st = 0

    def copy(self):
        a = _Arr.__new__(_Arr)
        a.items, a.lo, a.hi = self.items, self.lo, self.hi
        a.count, a.st = self.count, self.st
        return a

    def _close(self, m):
        if self.count < self.lo:
            raise _Fail()
        m.pop()

    def step(self, c, m):
        st = self.st
        if st == 0:
            if c != "[":
                raise _Fail()
            self.st = 1
            return True
        if st == 1:
            if c in _WS:
                return True
            if c == "]":
                self._close(m)
                return True
            if self.hi is not None and self.count >= self.hi:
                raise _Fail()
            m.push(_Val(self.items))
            self.st = 2
            return False
        if st == 3:
            if c in _WS:
                return True
            if c == "]":
                self._close(m)
                return True
            if c != ",":
                raise _Fail()
            if self.hi is not None and self.count >= self.hi:
                raise _Fail()  # one item too many
            m.push(_Val(self.items))
            self.st = 2
            return True
        raise AssertionError("array frame stepped while a child was open")

    def child(self, value, m):
        self.count += 1
        self.st = 3


class _StrF(_Frame):
    """`_string_free`. States: 0 the opening quote, 1 body, 2 after a
    backslash, 3 inside a \\uXXXX quad. `capture` is on only for object keys —
    a value's decoded text is discarded by the reference too."""

    __slots__ = ("capture", "out", "st", "hexs")

    def __init__(self, capture: bool):
        self.capture = capture
        self.out: list = []
        self.st = 0
        self.hexs = ""

    def copy(self):
        s = _StrF.__new__(_StrF)
        s.capture, s.st, s.hexs = self.capture, self.st, self.hexs
        s.out = list(self.out)
        return s

    def _emit(self, ch):
        if self.capture:
            self.out.append(ch)

    def step(self, c, m):
        st = self.st
        if st == 0:
            if c != '"':
                raise _Fail()
            self.st = 1
            return True
        if st == 1:
            if c == '"':
                m.pop("".join(self.out))
                return True
            if c == "\\":
                self.st = 2
                return True
            if ord(c) < 0x20 or c == "�":  # raw control / partial UTF-8
                raise _Fail()
            self._emit(c)
            return True
        if st == 2:
            if c == "u":
                self.st, self.hexs = 3, ""
                return True
            if c in _ESCAPES:
                self._emit(_ESCAPES[c])
                self.st = 1
                return True
            raise _Fail()
        if c not in _HEX:
            raise _Fail()
        self.hexs += c
        if len(self.hexs) == 4:
            self._emit(chr(int(self.hexs, 16)))
            self.st = 1
        return True


class _StrC(_Frame):
    """`_string_constrained`: a string whose decoded value must be one of
    `values`, banned at the first impossible character — including inside a
    half-written escape. Same states as _StrF plus `need`, the escape set the
    reference recomputes at each backslash."""

    __slots__ = ("values", "capture", "out", "st", "hexs", "need")

    def __init__(self, values: list, capture: bool):
        self.values = values
        self.capture = capture
        self.out: list = []
        self.st = 0
        self.hexs = ""
        self.need: set = set()

    def copy(self):
        s = _StrC.__new__(_StrC)
        s.values, s.capture, s.st = self.values, self.capture, self.st
        s.hexs, s.need = self.hexs, self.need
        s.out = list(self.out)
        return s

    def _alive(self) -> bool:
        d = "".join(self.out)
        return any(v.startswith(d) for v in self.values)

    def _next_chars(self) -> set:
        d = "".join(self.out)
        return {v[len(d)] for v in self.values if v.startswith(d) and len(v) > len(d)}

    def _append(self, ch):
        self.out.append(ch)
        if not self._alive():
            raise _Fail()

    def step(self, c, m):
        st = self.st
        if st == 0:
            if c != '"':
                raise _Fail()
            self.st = 1
            return True
        if st == 1:
            if c == '"':
                d = "".join(self.out)
                if d in self.values:
                    m.pop(d)
                    return True
                raise _Fail()
            if c == "\\":
                # FORM constraint (see _string_constrained): escapes only where
                # JSON requires them.
                need = {ch for ch in self._next_chars()
                        if ch in '"\\' or ord(ch) < 0x20}
                if not need:
                    raise _Fail()
                self.need, self.st = need, 2
                return True
            if ord(c) < 0x20 or c == "�":
                raise _Fail()
            self._append(c)
            return True
        if st == 2:
            if c == "u":
                self.st, self.hexs = 3, ""
                return True
            if c in _ESCAPES:
                if _ESCAPES[c] not in self.need:
                    raise _Fail()
                self._append(_ESCAPES[c])
                self.st = 1
                return True
            raise _Fail()
        if c not in _HEX:
            raise _Fail()
        self.hexs += c
        if len(self.hexs) == 4:
            ch = chr(int(self.hexs, 16))
            if ch not in self.need:
                raise _Fail()
            self._append(ch)
            self.st = 1
        return True

    def at_eoi(self):
        # mid-quad: PARTIAL only while some needed character still spells out
        # this prefix; body: PARTIAL only while some value still does
        if self.st == 3 and not any(("%04x" % ord(ch)).startswith(self.hexs.lower())
                                    for ch in self.need):
            return FAIL
        if self.st == 1 and not self._alive():
            return FAIL
        return PARTIAL


class _Num(_Frame):
    """`_number`, as a DFA. The states that can END a number (an integer part,
    a fraction, an exponent) POP at end-of-input instead of committing, which
    is what keeps "12" one number and not the number 1 followed by junk."""

    __slots__ = ("integer", "st")

    # sign, first digit, after a leading 0, in the integer part, after '.',
    # in the fraction, after e/E, after the exponent sign, in the exponent
    SIGN, FIRST, INT0, INT, DOT, FRAC, EXP, EXPSIGN, EXPD = range(9)
    _ENDABLE = frozenset((INT0, INT, FRAC, EXPD))

    def __init__(self, integer: bool):
        self.integer = integer
        self.st = _Num.SIGN

    def copy(self):
        n = _Num.__new__(_Num)
        n.integer, n.st = self.integer, self.st
        return n

    def step(self, c, m):
        st = self.st
        if st == _Num.SIGN:
            if c == "-":
                self.st = _Num.FIRST
                return True
            st = self.st = _Num.FIRST
        if st == _Num.FIRST:
            if not _is_digit(c):
                raise _Fail()
            self.st = _Num.INT0 if c == "0" else _Num.INT
            return True
        if st in (_Num.INT0, _Num.INT):
            if _is_digit(c):
                if st == _Num.INT0:
                    raise _Fail()  # leading zero
                return True
            if c == ".":
                if self.integer:
                    raise _Fail()
                self.st = _Num.DOT
                return True
            if c in "eE":
                if self.integer:
                    raise _Fail()
                self.st = _Num.EXP
                return True
            m.pop()  # the number ended; this character belongs to whoever asked
            return False
        if st == _Num.DOT:
            if not _is_digit(c):
                raise _Fail()
            self.st = _Num.FRAC
            return True
        if st == _Num.FRAC:
            if _is_digit(c):
                return True
            if c in "eE":
                self.st = _Num.EXP
                return True
            m.pop()
            return False
        if st == _Num.EXP:
            if c in "+-":
                self.st = _Num.EXPSIGN
                return True
            if not _is_digit(c):
                raise _Fail()
            self.st = _Num.EXPD
            return True
        if st == _Num.EXPSIGN:
            if not _is_digit(c):
                raise _Fail()
            self.st = _Num.EXPD
            return True
        if _is_digit(c):
            return True
        m.pop()
        return False

    def at_eoi(self):
        return _POP if self.st in _Num._ENDABLE else PARTIAL

    def finish(self, m):
        m.pop()


class _Lit(_Frame):
    """`_literal` over the candidates `_value` kept alive by the first
    character (one for true/false/null, several for an enum of non-strings).

    The reference has the whole text at once and commits to the FIRST
    candidate the text spells out in full; a candidate that could still grow
    is only a PARTIAL. So this frame keeps buffering while any candidate can
    still extend, and commits the moment none can — handing back whatever the
    committed candidate did not use, exactly as the reference leaves those
    characters for the caller."""

    __slots__ = ("cands", "buf")

    def __init__(self, cands: list):
        self.cands = cands
        self.buf = ""

    def copy(self):
        l = _Lit.__new__(_Lit)
        l.cands, l.buf = self.cands, self.buf
        return l

    def _full(self, buf: str):
        return next((l for l in self.cands if buf.startswith(l)), None)

    def step(self, c, m):
        nbuf = self.buf + c
        if any(l.startswith(nbuf) and len(l) > len(nbuf) for l in self.cands):
            self.buf = nbuf
            return True
        full = self._full(nbuf)
        if full is None:
            raise _Fail()
        m.pop()
        if len(full) == len(nbuf):
            return True
        m.pushback(nbuf[len(full):-1])  # the characters between it and c
        return False

    def at_eoi(self):
        if self._full(self.buf) is not None:
            return _POP
        if any(l.startswith(self.buf) for l in self.cands):
            return PARTIAL
        return FAIL

    def finish(self, m):
        full = self._full(self.buf)
        m.pop()
        m.pushback(self.buf[len(full):])


class _Machine:
    """The parse of one text, resumable. feed() advances it, clone() forks it
    in O(depth), status() answers the same three words status() always did."""

    __slots__ = ("schema", "max_ws", "stack", "dead", "run", "_pb")

    def __init__(self, schema: dict, max_ws: int):
        self.schema, self.max_ws = schema, max_ws
        self.stack: list = [_Val(schema)]
        self.dead = False
        self.run = 0  # current whitespace run (the MAX_WS_RUN rule)
        self._pb = ""

    def clone(self) -> "_Machine":
        m = _Machine.__new__(_Machine)
        m.schema, m.max_ws = self.schema, self.max_ws
        m.stack = [f.copy() for f in self.stack]
        m.dead, m.run, m._pb = self.dead, self.run, self._pb
        return m

    # -- what frames call --

    def push(self, frame: _Frame) -> None:
        self.stack.append(frame)

    def replace(self, frame: _Frame) -> None:
        self.stack[-1] = frame

    def pop(self, value=None) -> None:
        self.stack.pop()
        if self.stack:
            self.stack[-1].child(value, self)

    def pushback(self, text: str) -> None:
        self._pb += text

    # -- driving --

    def feed(self, text: str) -> None:
        if self.dead or not text:
            return
        # The whitespace-run rule is over the RAW text and independent of the
        # parse — status() checked it first and returned FAIL for the whole
        # document, so a run that trips inside a string banned that string too.
        run = self.run
        for ch in text:
            run = run + 1 if ch in _WS else 0
            if run > self.max_ws:
                self.dead = True
                return
        self.run = run
        try:
            self._run(text)
        except _Fail:
            self.dead = True

    def _run(self, text: str) -> None:
        pb, i, n = self._pb, 0, len(text)
        self._pb = ""
        while True:
            if pb:
                c = pb[0]
            elif i < n:
                c = text[i]
            else:
                break
            if self.stack:
                consumed = self.stack[-1].step(c, self)
            else:  # the document is complete: only whitespace may follow it
                if c not in _WS:
                    raise _Fail()
                consumed = True
            extra, self._pb = self._pb, ""
            if pb:
                pb = extra + (pb[1:] if consumed else pb)
            else:
                if consumed:
                    i += 1
                pb = extra
        self._pb = pb

    def status(self) -> str:
        """FAIL | PARTIAL | COMPLETE for everything fed so far. Non-mutating:
        a frame that would END at end-of-input (a number, a literal) is
        resolved on a clone, because the input has not actually ended."""
        if self.dead:
            return FAIL
        if not self.stack:
            return COMPLETE
        r = self.stack[-1].at_eoi()
        if r is not _POP:
            return r
        return self.clone()._resolve()

    def _resolve(self) -> str:
        """Run the end-of-input cascade for real (clones only): a number that
        ends may uncover an array that does not, and a literal that ends may
        hand back characters the frame beneath it has to eat."""
        while self.stack:
            r = self.stack[-1].at_eoi()
            if r is not _POP:
                return r
            try:
                self.stack[-1].finish(self)
                if self._pb:
                    self._run("")
            except _Fail:
                return FAIL
        return COMPLETE


class JsonConstraint:
    """status(text) -> FAIL | PARTIAL | COMPLETE against one schema.

    COMPLETE means the text IS a valid document now (EOS is legal); it may
    still be extendable (e.g. a bare number). PARTIAL means some continuation
    completes it. FAIL means none does.

    The verdicts come from the incremental machine above; status_reference()
    is the recursive descent that defines them, kept for the fuzz that pins
    the two together. `validate=False` is for a caller that already ran
    validate_schema — the HTTP boundary does, per request, and walking a
    deep schema twice to hear the same answer is the hot-loop audit's problem."""

    def __init__(self, schema: dict, validate: bool = True):
        if validate:
            validate_schema(schema)
        self.schema = schema

    # Longest run of whitespace generation may emit. Whitespace is always
    # legal JSON, so a model whose preferred tokens are banned (e.g. a
    # thinking preamble) pours probability into it — a live smoke spent a
    # third of its budget on '\t\r\n' spam before the object opened. Capping runs loses no expressible VALUE, only formatting.
    MAX_WS_RUN = 3

    def status(self, text: str) -> str:
        m = _Machine(self.schema, self.MAX_WS_RUN)
        m.feed(text)
        return m.status()

    def status_reference(self, text: str) -> str:
        """The recursive-descent verdict: the DEFINITION of status(), and the
        oracle tests/test_hotloop_equivalence.py fuzzes the machine against.
        Never on the serve path — it re-reads the whole text every call, which
        is the O(n^2) the machine exists to end."""
        run = 0
        for c in text:
            run = run + 1 if c in _WS else 0
            if run > self.MAX_WS_RUN:
                return FAIL
        p = _P(text)
        try:
            _value(p, self.schema)
            p.ws()
            if p.i < p.n:
                return FAIL  # trailing junk after the document
            return COMPLETE
        except _Partial:
            return PARTIAL
        except _Fail:
            return FAIL

    def cursor(self, text: str = "") -> "JsonCursor":
        """A parse state for one generation, optionally already holding the
        text decoded so far."""
        c = JsonCursor(self)
        if text:
            c.probe(text)
            c.accept()
        return c


class JsonCursor:
    """The parse of the text a generation has ACCEPTED, carried forward.

    Without it every candidate token re-parses the whole output from byte 0
    (the hot-loop audit): probe() forks the machine in O(nesting depth) and
    feeds only the candidate's own characters, and accept() promotes the last
    probe in O(1). The verdicts are the ones status() would give — the same
    machine, reached by a shorter road."""

    __slots__ = ("_c", "_m", "text", "_pm", "_pt")

    def __init__(self, constraint: JsonConstraint):
        self._c = constraint
        self._m = _Machine(constraint.schema, constraint.MAX_WS_RUN)
        self.text = ""
        self._pm = self._pt = None

    def status(self) -> str:
        """The verdict for the accepted text (what EOS is judged against)."""
        return self._m.status()

    def probe(self, text: str) -> str:
        """status(text) for a text that usually EXTENDS the accepted one. When
        it does not — a tokenizer re-forming its tail across a multi-byte
        boundary — the parse restarts from byte 0."""
        if text.startswith(self.text):
            m = self._m.clone()
            m.feed(text[len(self.text):])
        else:
            m = _Machine(self._c.schema, self._c.MAX_WS_RUN)
            m.feed(text)
        self._pm, self._pt = m, text
        return m.status()

    def accept(self) -> None:
        """Commit the last probe: its text is the accepted text from here."""
        if self._pm is None:
            raise AssertionError("accept() without a probe")
        self._m, self.text = self._pm, self._pt
        self._pm = self._pt = None


_DEAD_END = ("json constraint dead-end: every token banned (validator bug — "
             "please report the schema and partial output)")

# torch.isfinite over the whole vocabulary is a kernel and a host sync, so it
# is asked once per STEP, not once per CANDIDATE (the hot-loop audit), and as
# a COUNT: the dead end is "every finite logit banned", so after one reduction
# the verdict is a python integer compare per ban, exact and free. (A
# ban-count threshold cannot answer for a row whose support is smaller than
# the threshold; this can.)


def pick_token(logits, params, generator, prev_ids, gen_ids: list[int],
               eos_ids, decode: Callable[[list[int]], str],
               constraint: JsonConstraint, cur: str | None = None,
               decode_cand: Callable[[int], str] | None = None,
               cursor: JsonCursor | None = None, state=None, scratch=None) -> int:
    """One constrained decode step: sample through the normal pipeline, ban
    candidates whose decoded text goes FAIL, resample. EOS allowed iff the
    current text is COMPLETE. A candidate that adds no visible text (a
    special token decoding to nothing) is banned — invisible tokens would
    loop forever. Raises RuntimeError on the unreachable all-banned state.

    Five optional arguments let a caller stop paying for what it already
    knows (the hot-loop audit); all five default to computing it
    here. `cur` is the decoded text of gen_ids, which the engine decoded
    this token anyway; `decode_cand(tok)` returns that text with one candidate
    appended (a suffix-window decode instead of a fresh full one); `cursor`
    carries the JSON parse state so no candidate re-parses from byte 0;
    `state` and `scratch` are sampling.py's PenaltyState and SampleScratch,
    threaded through to sample_next — the constrained path is where the two
    full-vocabulary clones per candidate hurt most."""
    import torch

    from .sampling import sample_next

    if scratch is not None:
        logits = scratch.take_ban(logits)  # the row bans accumulate on
    else:
        logits = torch.as_tensor(logits).to(torch.float32).clone()
    if cur is None:
        cur = decode(gen_ids)
    live = int(torch.isfinite(logits).sum())  # the one full-vocab pass
    banned: set[int] = set()
    while True:
        if len(banned) >= live:
            raise RuntimeError(_DEAD_END)
        tok = sample_next(logits, params, generator, prev_ids=prev_ids,
                          gen_ids=gen_ids, state=state, scratch=scratch)
        if tok in banned:
            # unreachable while the row is sane — the count above raises first.
            # It is not sane if the row carries NaNs: isfinite() does not count
            # them, argmax happily returns one, and re-banning the same id
            # forever is the one failure mode worse than a 500.
            raise RuntimeError(_DEAD_END)
        if tok in eos_ids:
            done = cursor.status() if cursor is not None else constraint.status(cur)
            if done == COMPLETE:
                return tok
        else:
            cand = (decode_cand(tok) if decode_cand is not None
                    else decode(gen_ids + [tok]))
            if len(cand) > len(cur):
                if cursor is not None:
                    if cursor.probe(cand) != FAIL:
                        cursor.accept()
                        return tok
                elif constraint.status(cand) != FAIL:
                    return tok
        logits[tok] = float("-inf")
        banned.add(tok)
