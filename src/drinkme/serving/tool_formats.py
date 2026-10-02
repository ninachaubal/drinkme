"""The tool-format table — one row per output dialect, data not branches.

`chat_template` declares a checkpoint's INPUT side completely. Nothing
declares the OUTPUT side: how the model emits tool calls back. The de-facto
registry is code other people maintain (vLLM's `vllm/parser/`, one family per
file; LiteLLM's per-provider transforms on the client side). This module is
drinkme's half of that — the collection is called the
COMPENDIUM, and this is it: ONE
table, each row

    name           what /v1/models announces as `tool_format`
    signature      substrings that must ALL appear in the tools-rendered
                   prompt for this row to be the answer (capability.probe)
    open / close   the wire markers; close=None means ONE-SIDED — the block
                   runs to the end of the turn (Mistral)
    parse          ordered body-parser chain, most specific first; the first
                   one to return calls wins, None from all of them means
                   "this was not a call" and the block re-emerges as text
    hint           prompt text a template that declares tools but never
                   teaches the emission form would need. No row in this table has
                   one — every template here writes its own instruction
                   block — but the column is the table's, not the code's.
    control_tokens the markers above that are SPECIAL TOKENS in the family's
                   vocab (gemma's `<|tool_call>`, read off its tokenizer):
                   the serve loop keeps exactly these ids as text so the
                   scanner can see them (serving/control.py); empty means
                   every marker is plain text and nothing changes
    think_open /   the reasoning channel on the same row, walked by
    think_close    serving/think.py: what the model writes to open and
                   close it. Qwen's `<think>`/`</think>` is the default;
                   a family that NAMES its channel (gemma's
                   `<|channel>thought`) declares the name in think_channel
    message_open / a family whose turn is a run of ADDRESSED messages
    recipient_open (Muse-Glimmer's ATEM: `<|start|>assistant to=self
    message_body   <|message|>…<|eom|><|start|>assistant to=user<|message|>…`)
                   declares its header's three literals, and think.py walks
                   messages by recipient instead of one tag pair. None (every
                   other row) = one stream with one think tag pair

serving/tools.py's ToolCallScanner is the ONE engine that walks any row;
capability.probe picks the row at serve start; http.py refuses a tools
request when no row matched. Adding a family is a row, not a branch.

THE TWO WE SERVE BY DEFAULT. `json` and `qwen-xml` share BOTH markers
(`<tool_call>`/`</tool_call>`) and, deliberately, the SAME parser chain
(JSON body first, then the qwen3_5 XML-parameter body). That is exactly what
the single hardcoded scanner did before this table existed, and a Qwen model
that emits the other family's body still parses — narrowing each row to its
own body parser would be a behaviour change, not a cleanup, so it is not
done here. The rows differ in what the TEMPLATE says (the signature) and so
in what /v1/models announces, which is the honest distinction.

NEVER INVENT, NEVER DROP — the two absolutes, inherited from tools.py and
applied to every row: a block whose body no parser in the chain recognizes
re-emerges verbatim as visible text, and nothing generated is silently
dropped. Every row that reads a bare identifier out of the wire (glm's
no-arg `<tool_call>name</tool_call>`, gemma's `call:name{...}`, mistral's
v11 `[TOOL_CALLS]name{...}`) requires it to LOOK like an identifier, so
prose inside a stray marker degrades to text instead of becoming a call the
model never made.

TYPING. Dialects that carry their own types on the wire (json, gemma,
mistral, kimi-k2 — all JSON or JSON-shaped) are read as written. Dialects
that write parameter values as raw text (qwen-xml, glm, minimax, atem) are typed
by the REQUEST'S OWN TOOL SCHEMAS, vLLM's qwen3-coder posture: a declared
integer/number/boolean/object/array/null parses as JSON when it can,
everything else — string params, undeclared params, schema-less calls —
stays the verbatim text. Never guess a type the client didn't declare.

SOURCES. Every row's markers were read off a real artifact on 2026-09-08, not
recalled; the fixture beside each row in tests/test_serving_tool_formats.py
carries its URL. Prior art consulted per family (read, not copied; vLLM is
Apache-2.0 and nothing here is a transcription of its code — the grammars
below were written against each family's own chat template, with vLLM's
parser used only to cross-check marker names):

    json       Qwen3 / Hermes lineage — the family's own template block
    qwen-xml   Qwen3.8 / Qwen3-Coder template; vllm/parser/qwen3.py
    gemma      google/gemma-4-*-it chat_template.jinja (Google, published
               2026-07-09); vllm/parser/gemma4.py
    glm        zai-org/GLM-4.7 chat_template.jinja; vllm/parser/glm47_moe.py
    minimax    MiniMaxAI/MiniMax-M2 chat_template.jinja;
               vllm/parser/minimax_m2.py
    mistral    mistralai/Mistral-7B-Instruct-v0.3 tokenizer_config.json;
               vllm/parser/mistral.py
    kimi-k2    moonshotai/Kimi-K2-Instruct chat_template.jinja;
               vllm/parser/kimi_k2.py
    atem       meta-models/Muse-Glimmer-30B chat_template.jinja (read
               2026-09-25), and the `response_template` its
               tokenizer_config.json ships — the checkpoint's own
               declaration of its output side, parsed by transformers'
               chat_parsing; tests/test_serving_glimmer_channel.py checks
               this row against it

oMLX's README table is the borrow that named
the row set. ONE correction, measured: it lists Gemma as
`<start_function_call>`, and no Gemma artifact reachable on 2026-09-08 emits
that — the canonical Gemma 4 template and vLLM's gemma4 parser both use
`<|tool_call>` ... `<tool_call|>`. The row implements what the template
actually writes. Longcat (`<longcat_tool_call>`) is on their table and not
here: no template to read it off, so no row rather than a guessed one.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass

# A parsed call, the shape every consumer downstream already speaks
# (tools.to_openai_calls, messages.tool_use): {"name": str, "arguments": dict}.
ToolCall = dict

# A body parser: (block body, {fn name -> {param -> declared type}}) -> the
# calls in the block, or None for "not this dialect / not a call at all".
BodyParser = Callable[[str, dict], "list[ToolCall] | None"]

# What a name may look like anywhere a dialect writes one unquoted. Prose
# inside a stray marker must not become a call.
_NAME_RE = re.compile(r"\A[A-Za-z_][\w.\-]*\Z")


def _coerce(text: str, ptype: str | None):
    """Raw parameter text -> argument value, schema-driven. Only a declared
    non-string type earns a JSON parse; failure keeps the verbatim text (a
    malformed "5x" for an integer is the model's problem to see echoed, not
    ours to guess at)."""
    if ptype in ("integer", "number", "boolean", "object", "array", "null"):
        try:
            return json.loads(text)
        except ValueError:
            return text
    return text


def _types_for(types: dict, name: str) -> dict:
    return (types or {}).get(name) or {}


# ------------------------------------------------------------------- json --
# <tool_call>\n{"name": ..., "arguments": {...}}\n</tool_call>
# Qwen3 / Llama / DeepSeek / Hermes lineage. Unchanged from tools.py's
# original _parse_call except for returning a LIST (one block, one call).


def json_body(body: str, types: dict) -> list[ToolCall] | None:
    """Block body -> [{"name", "arguments"}], or None when the body is not a
    call (malformed JSON, non-object, missing/empty name, non-object
    arguments). None means "re-emerge as text", never "guess"."""
    try:
        obj = json.loads(body)
    except ValueError:
        return None
    if not isinstance(obj, dict):
        return None
    name = obj.get("name")
    if not isinstance(name, str) or not name:
        return None
    args = obj.get("arguments", {})  # absent arguments = a no-arg call
    if not isinstance(args, dict):
        return None
    return [{"name": name, "arguments": args}]


# --------------------------------------------------------------- qwen-xml --
# <tool_call>\n<function=NAME>\n<parameter=KEY>\nVALUE\n</parameter>...\n</function>\n</tool_call>
# qwen3_5 (Qwen3.8, Coder lineage): outer tags kept, body became XML with
# raw multi-line VALUEs. The template's own system-prompt instruction is the
# normative format.

_FUNC_RE = re.compile(r"\A\s*<function=([^>\s]+)>\n?(.*)</function>\s*\Z", re.S)
_PARAM_RE = re.compile(r"<parameter=([^>\s]+)>\n?(.*?)\n?</parameter>\s*", re.S)


def qwen_xml_body(body: str, types: dict) -> list[ToolCall] | None:
    """qwen3_5 body form -> [{"name", "arguments"}], or None (re-emerge as
    text). One <function=...> block; zero or more <parameter=...> children;
    one structural newline trimmed at each value edge (the template writes
    values on their own lines); anything non-whitespace between parameters
    means the block is not a well-formed call."""
    m = _FUNC_RE.match(body)
    if m is None:
        return None
    name, inner = m.group(1), m.group(2)
    ptypes = _types_for(types, name)
    args: dict = {}
    pos = 0
    for pm in _PARAM_RE.finditer(inner):
        if inner[pos:pm.start()].strip():
            return None
        args[pm.group(1)] = _coerce(pm.group(2), ptypes.get(pm.group(1)))
        pos = pm.end()
    if inner[pos:].strip():
        return None
    return [{"name": name, "arguments": args}]


# -------------------------------------------------------------------- glm --
# <tool_call>NAME<arg_key>k</arg_key><arg_value>v</arg_value>...</tool_call>
# GLM 4.7 / 5. Same outer tags as json/qwen-xml, a third body form. The
# template writes string values RAW and everything else through tojson, so
# the schema-driven _coerce reads back exactly what the encoder wrote.

_GLM_PAIR_RE = re.compile(
    r"<arg_key>(.*?)</arg_key>\s*<arg_value>(.*?)</arg_value>\s*", re.S)


def glm_body(body: str, types: dict) -> list[ToolCall] | None:
    """GLM body form -> [{"name", "arguments"}], or None. The name is the
    text before the first <arg_key> and must look like one; a bare body with
    no pairs at all is the template's own no-arg render, which is why the
    identifier check is essential rather than decorative (prose in a
    stray <tool_call> would otherwise become a call)."""
    head, sep, rest = body.partition("<arg_key>")
    name = head.strip()
    if not _NAME_RE.match(name):
        return None
    if not sep:
        return [{"name": name, "arguments": {}}]
    ptypes = _types_for(types, name)
    args: dict = {}
    pos = 0
    inner = "<arg_key>" + rest
    for pm in _GLM_PAIR_RE.finditer(inner):
        if inner[pos:pm.start()].strip():
            return None
        args[pm.group(1).strip()] = _coerce(pm.group(2), ptypes.get(pm.group(1).strip()))
        pos = pm.end()
    if inner[pos:].strip() or not args:
        return None
    return [{"name": name, "arguments": args}]


# ------------------------------------------------------------------ gemma --
# <|tool_call>call:NAME{key:<|"|>value<|"|>,num:42}<tool_call|>
# Gemma 4. Self-describing on the wire: the template's format_argument macro
# writes strings wrapped in the delimiter <|"|>, mappings as {k:v,...},
# sequences as [a,b], and null/true/false/numbers bare — so this row needs
# NO schema to type its arguments, unlike qwen-xml/glm/minimax.

GEMMA_STRING_DELIM = '<|"|>'


def _gemma_value(s: str, i: int):
    """Scan one value at s[i:]. Returns (value, next index) or None."""
    if s.startswith(GEMMA_STRING_DELIM, i):
        i += len(GEMMA_STRING_DELIM)
        end = s.find(GEMMA_STRING_DELIM, i)
        if end < 0:
            return None
        return s[i:end], end + len(GEMMA_STRING_DELIM)
    if s.startswith("{", i):
        return _gemma_mapping(s, i + 1)
    if s.startswith("[", i):
        return _gemma_sequence(s, i + 1)
    # a bare token: null / true / false / a number, ending at the first
    # delimiter of the enclosing structure
    j = i
    while j < len(s) and s[j] not in ",}]":
        j += 1
    raw = s[i:j].strip()
    if not raw:
        return None
    try:
        return json.loads(raw), j
    except ValueError:
        return raw, j  # not a JSON scalar: the text it really was


def _gemma_key(s: str, i: int):
    """Scan one key at s[i:] — bare (tool-call args, escape_keys=False) or
    delimiter-wrapped. Returns (key, index of the ':') or None."""
    if s.startswith(GEMMA_STRING_DELIM, i):
        i += len(GEMMA_STRING_DELIM)
        end = s.find(GEMMA_STRING_DELIM, i)
        if end < 0 or not s.startswith(":", end + len(GEMMA_STRING_DELIM)):
            return None
        return s[i:end], end + len(GEMMA_STRING_DELIM)
    end = s.find(":", i)
    if end < 0:
        return None
    key = s[i:end].strip()
    if not key or any(c in key for c in ",{}[]"):
        return None  # the colon belonged to a later pair: this is not a key
    return key, end


def _gemma_mapping(s: str, i: int):
    """Scan `k:v,k:v}` starting just past the '{'. -> (dict, next index)."""
    out: dict = {}
    if s.startswith("}", i):
        return out, i + 1
    while True:
        k = _gemma_key(s, i)
        if k is None:
            return None
        key, colon = k
        v = _gemma_value(s, colon + 1)
        if v is None:
            return None
        out[key], i = v
        if s.startswith(",", i):
            i += 1
            continue
        if s.startswith("}", i):
            return out, i + 1
        return None


def _gemma_sequence(s: str, i: int):
    """Scan `a,b]` starting just past the '['. -> (list, next index)."""
    out: list = []
    if s.startswith("]", i):
        return out, i + 1
    while True:
        v = _gemma_value(s, i)
        if v is None:
            return None
        val, i = v
        out.append(val)
        if s.startswith(",", i):
            i += 1
            continue
        if s.startswith("]", i):
            return out, i + 1
        return None


def gemma_body(body: str, types: dict) -> list[ToolCall] | None:
    """Gemma 4 body form -> [{"name", "arguments"}], or None."""
    s = body.strip()
    if not s.startswith("call:"):
        return None
    brace = s.find("{")
    if brace < 0 or not s.endswith("}"):
        return None
    name = s[len("call:"):brace].strip()
    if not _NAME_RE.match(name):
        return None
    scanned = _gemma_mapping(s, brace + 1)
    if scanned is None:
        return None
    args, end = scanned
    if s[end:].strip():
        return None
    return [{"name": name, "arguments": args}]


# ---------------------------------------------------------------- minimax --
# <minimax:tool_call><invoke name="NAME">
# <parameter name="k">v</parameter>
# </invoke></minimax:tool_call>
# MiniMax M2. ONE block may carry SEVERAL <invoke>s — the first row here
# whose parse returns more than one call. Values are raw text (strings
# verbatim, everything else tojson'd by the template), so _coerce types them
# off the request's schemas; unlike qwen-xml the template writes no
# structural newline inside the value, so nothing is trimmed.

_MM_INVOKE_RE = re.compile(
    r"<invoke\s+name\s*=\s*(?:\"([^\"]*)\"|'([^']*)'|([^\s>]+))\s*>(.*?)</invoke>\s*",
    re.S)
_MM_PARAM_RE = re.compile(
    r"<parameter\s+name\s*=\s*(?:\"([^\"]*)\"|'([^']*)'|([^\s>]+))\s*>(.*?)</parameter>\s*",
    re.S)


def _mm_name(m: re.Match, first: int) -> str:
    return next(g for g in m.groups()[first:first + 3] if g is not None)


def minimax_body(body: str, types: dict) -> list[ToolCall] | None:
    """MiniMax body form -> the block's calls in order, or None."""
    calls: list[ToolCall] = []
    pos = 0
    for im in _MM_INVOKE_RE.finditer(body):
        if body[pos:im.start()].strip():
            return None
        name = _mm_name(im, 0)
        if not _NAME_RE.match(name):
            return None
        inner = im.group(4)
        ptypes = _types_for(types, name)
        args: dict = {}
        ipos = 0
        for pm in _MM_PARAM_RE.finditer(inner):
            if inner[ipos:pm.start()].strip():
                return None
            key = _mm_name(pm, 0)
            args[key] = _coerce(pm.group(4), ptypes.get(key))
            ipos = pm.end()
        if inner[ipos:].strip():
            return None
        calls.append({"name": name, "arguments": args})
        pos = im.end()
    if body[pos:].strip() or not calls:
        return None
    return calls


# ---------------------------------------------------------------- mistral --
# [TOOL_CALLS] [{"name": ..., "arguments": {...}, "id": "..."}]
# Mistral / Devstral. The ONE-SIDED row: there is no closing marker, so
# everything after [TOOL_CALLS] is suppressed to the end of the turn and
# parsed at flush — vLLM's and oMLX's rule for one-sided markers both. The
# v11 engine path writes `[TOOL_CALLS]name{args}` instead; both forms parse
# here. Extra keys the template adds (`id`) are ignored, not required.


def _mistral_entry(obj) -> ToolCall | None:
    if not isinstance(obj, dict):
        return None
    name = obj.get("name")
    if not isinstance(name, str) or not name:
        return None
    args = obj.get("arguments", {})
    if isinstance(args, str):  # some emissions pre-encode the arguments
        try:
            args = json.loads(args)
        except ValueError:
            return None
    if not isinstance(args, dict):
        return None
    return {"name": name, "arguments": args}


def mistral_body(body: str, types: dict) -> list[ToolCall] | None:
    """Everything after [TOOL_CALLS] -> the turn's calls, or None."""
    s = body.strip()
    if not s:
        return None
    if s[0] not in "[{":  # v11 shape: name{...}
        brace = s.find("{")
        if brace < 0:
            return None
        name = s[:brace].strip()
        if not _NAME_RE.match(name):
            return None
        try:
            args, end = json.JSONDecoder().raw_decode(s, brace)
        except ValueError:
            return None
        if s[end:].strip() or not isinstance(args, dict):
            return None
        return [{"name": name, "arguments": args}]
    try:
        obj, end = json.JSONDecoder().raw_decode(s)
    except ValueError:
        return None
    if s[end:].strip():
        return None
    entries = obj if isinstance(obj, list) else [obj]
    calls = [_mistral_entry(e) for e in entries]
    if not calls or any(c is None for c in calls):
        return None
    return calls


# ---------------------------------------------------------------- kimi-k2 --
# <|tool_calls_section_begin|>
# <|tool_call_begin|>functions.NAME:0<|tool_call_argument_begin|>{...}<|tool_call_end|>
# <|tool_calls_section_end|>
# Kimi K2. The header before the argument marker is Kimi's native call id;
# the function name is the component before `:N`, minus the `functions.`
# namespace.

_KIMI_CALL_RE = re.compile(
    r"<\|tool_call_begin\|>(.*?)<\|tool_call_argument_begin\|>(.*?)<\|tool_call_end\|>\s*",
    re.S)


def kimi_body(body: str, types: dict) -> list[ToolCall] | None:
    """Kimi K2 section body -> the section's calls in order, or None."""
    calls: list[ToolCall] = []
    pos = 0
    for m in _KIMI_CALL_RE.finditer(body):
        if body[pos:m.start()].strip():
            return None
        name = m.group(1).strip().split(":")[0].removeprefix("functions.")
        if not _NAME_RE.match(name):
            return None
        try:
            args = json.loads(m.group(2))
        except ValueError:
            return None
        if not isinstance(args, dict):
            return None
        calls.append({"name": name, "arguments": args})
        pos = m.end()
    if body[pos:].strip() or not calls:
        return None
    return calls


# ------------------------------------------------------------------- atem --
# <atem:function_calls>
# <atem:invoke name="NAME">
# <atem:parameter name="k">v</atem:parameter>
# </atem:invoke>
# </atem:function_calls>
# Muse-Glimmer's ATEM. The block is the body of a message addressed to the
# tool (`<|start|>assistant to=NAME<|message|>`, the header think.py's
# AddressedSplitter drops); its render_atem macro writes strings raw,
# booleans and null as JSON words, mappings and sequences through tojson,
# and "spaces for string values are not stripped" (the template's own
# instruction), so a value is the exact text between its tags, typed by the
# request's schemas. The invoke and parameter patterns are the ones the
# checkpoint's response_template declares. One block may carry several
# invokes (the response_template's `repeats`), as minimax's may.

_ATEM_INVOKE_RE = re.compile(r'<atem:invoke\b[^>]*?\bname="([^"]+)">(.*?)</atem:invoke>\s*', re.S)
_ATEM_PARAM_RE = re.compile(
    r'<atem:parameter\b[^>]*?\bname="([^"]+)"[^>]*?>(.*?)</atem:parameter>\s*', re.S)


def atem_body(body: str, types: dict) -> list[ToolCall] | None:
    """ATEM function_calls body -> the block's calls in order, or None."""
    calls: list[ToolCall] = []
    pos = 0
    for im in _ATEM_INVOKE_RE.finditer(body):
        if body[pos:im.start()].strip():
            return None
        name = im.group(1)
        if not _NAME_RE.match(name):
            return None
        inner = im.group(2)
        ptypes = _types_for(types, name)
        args: dict = {}
        ipos = 0
        for pm in _ATEM_PARAM_RE.finditer(inner):
            if inner[ipos:pm.start()].strip():
                return None
            args[pm.group(1)] = _coerce(pm.group(2), ptypes.get(pm.group(1)))
            ipos = pm.end()
        if inner[ipos:].strip():
            return None
        calls.append({"name": name, "arguments": args})
        pos = im.end()
    if body[pos:].strip() or not calls:
        return None
    return calls


# ------------------------------------------------------------------ table --


@dataclass(frozen=True)
class ToolFormat:
    """One dialect. Read the module docstring for what each column means."""

    name: str
    signature: tuple[str, ...]
    open: str
    close: str | None
    parse: tuple[BodyParser, ...]
    hint: str | None
    # The TESTED tier ("menu models stay the tested tier"): True only for
    # rows a real model on the menu has exercised end to end (json = Qwen3-8B,
    # qwen-xml = Qwen3.8-27B, gemma = gemma-4-31B-it by
    # bench/gemma_control_tokens_gate.py, atem = Muse-Glimmer-30B
    # by bench/serve_dialect_smoke.py). A parseable-but-untested row is announced on
    # /v1/models by name and REFUSED for tools requests until a GPU smoke flips
    # it — refusing loudly beats a dialect leaking into content as text.
    tested: bool = True
    # CONTROL TOKENS. The markers of this row that are SPECIAL ids in the
    # family's vocab. `decode(skip_special_tokens=True)` erases those before
    # any scanner looks — measured on real gemma-4-31B-it weights:
    # the model emitted its dialect exactly, and the client got
    # `call:get_weather{city:Hilo}` as prose followed by a hallucinated
    # answer, because `<|tool_call>` / `<tool_call|>` / `<|channel>` / the
    # `<|"|>` string delimiter had all been stripped. serving/control.py
    # resolves these strings to ids at engine load, keeps exactly those ids
    # as text in the serve loop's decode, and makes the row's close marker
    # end the turn. Empty (every row but gemma today) = the decode is the
    # exact call it always was, byte for byte. Qwen's markers are plain-text
    # added tokens in its vocab, which is the only reason they ever worked.
    control_tokens: tuple[str, ...] = ()
    # THE REASONING CHANNEL on the same row (serving/think.py walks these).
    # think_open is the whole literal the model writes to open it — for a
    # family that names channels that is the control token PLUS the name
    # (gemma: `<|channel>` + `thought`) — and think_close what ends it. The
    # default is the Qwen convention every row had before these columns
    # existed. think_channel is the name alone, for families that have one.
    think_open: str = "<think>"
    think_close: str = "</think>"
    think_channel: str | None = None
    # ADDRESSED MESSAGES (the module docstring's last three columns). A
    # message is `message_open` WS* `recipient_open` NAME `message_body`
    # BODY `think_close`; the generation prompt writes the first message's
    # message_open, and the turn's last message ends at EOS. A body goes to
    # the reasoning channel when NAME is think_channel, to content otherwise,
    # and think_open is then recipient_open + think_channel + message_body:
    # what a prompt that opened the reasoning message ends with.
    message_open: str | None = None
    recipient_open: str | None = None
    message_body: str | None = None

    def matches(self, rendered: str) -> bool:
        """Is this the row the tools-rendered prompt is teaching?"""
        return all(s in rendered for s in self.signature)

    def parse_body(self, body: str, types: dict) -> list[ToolCall] | None:
        """The chain, most specific first. None = not a call: the caller
        re-emerges the whole block verbatim as visible text."""
        for parser in self.parse:
            calls = parser(body, types)
            if calls:
                return calls
        return None


# The shared chain for the `<tool_call>` families we already serve — JSON
# body first, then the qwen3_5 XML body, exactly as the single hardcoded
# scanner did. See the module docstring on why both rows carry it.
_TOOL_CALL_CHAIN = (json_body, qwen_xml_body)

# ORDERED — first signature match wins. The specific rows come before the
# generic `<tool_call>` ones because GLM's instruction block contains both
# `<tool_call>` and the word "arguments" and would otherwise be read as
# `json`. The relative order of `qwen-xml` before `json` is the one the
# hardcoded probe had, kept byte-for-byte.
ROWS: tuple[ToolFormat, ...] = (
    ToolFormat(
        name="gemma",
        signature=("<|tool>declaration:",),
        open="<|tool_call>", close="<tool_call|>",
        parse=(gemma_body,), hint=None,
        # GPU-gated on real weights (Strix Halo, compressed arm,
        # --spec off): bench/gemma_control_tokens_gate.py 9/9 verdicts —
        # structured call, stop-at-close, stream, tool result, reasoning
        # split (594/441 chars) + stream, Anthropic tool_use, all leak-free.
        # n-gram
        # speculation agreed with serial in the same window
        # (ngram_agreement_gate 4/4).
        tested=True,
        # Read off google/gemma-4-31B-it's tokenizer (all_special_ids,
        # 2026-09-13): stc/etc 48/49, soc/eoc 100/101, escape 52 — the
        # tokenizer_config names them by those keys. The string delimiter
        # is a control token too: without it `{city:<|"|>a,b<|"|>}` reaches
        # the parser as `{city:a,b}`. `<|tool_response>` (50) is the
        # family's own end-of-calls token and already an EOS in its
        # generation_config, so it is a stop, not a marker.
        control_tokens=("<|tool_call>", "<tool_call|>", "<|channel>", "<channel|>",
                        GEMMA_STRING_DELIM),
        think_open="<|channel>thought", think_close="<channel|>",
        think_channel="thought"),
    ToolFormat(
        name="atem",
        signature=("<atem:function_calls>", "<atem:invoke"),
        open="<atem:function_calls>", close="</atem:function_calls>",
        parse=(atem_body,), hint=None,
        # Parsed against the checkpoint's own template and response_template
        # on the CPU (tests/test_serving_glimmer_channel.py). GPU-gated on
        # real weights (Strix Halo, compressed arm, `drinkme serve
        # --ctx 16384`, "Reasoning strength: low." in the system turn):
        # bench/serve_dialect_smoke.py held all six probes — plain and
        # streamed with the reasoning split and no leak, a structured call
        # out (finish tool_calls, content null) and streamed, the tool
        # result used, and Anthropic tool_use, all leak-free.
        tested=True,
        # Read off meta-models/Muse-Glimmer-30B's tokenizer (all_special_ids,
        # 2026-09-25): <|start|> 200022, <|message|> 200023, <|eom|> 200007.
        # Its other turn end, <|eot|> (200008), is an EOS in the checkpoint's
        # generation_config, so it ends the turn and never reaches a scanner.
        # The call markers are plain text: nothing here is a stop.
        control_tokens=("<|start|>", "<|message|>", "<|eom|>"),
        # the response_template's reasoning_content field: opens at
        # `to=self<|message|>`, closes at `<|eom|>`
        think_open="to=self<|message|>", think_close="<|eom|>", think_channel="self",
        message_open="<|start|>assistant", recipient_open="to=", message_body="<|message|>"),
    ToolFormat(
        name="kimi-k2",
        signature=("<|im_system|>tool_declare",),
        open="<|tool_calls_section_begin|>", close="<|tool_calls_section_end|>",
        parse=(kimi_body,), hint=None, tested=False),
    ToolFormat(
        name="minimax",
        signature=("<minimax:tool_call>",),
        open="<minimax:tool_call>", close="</minimax:tool_call>",
        parse=(minimax_body,), hint=None, tested=False),
    ToolFormat(
        name="mistral",
        signature=("[AVAILABLE_TOOLS]",),
        open="[TOOL_CALLS]", close=None,
        parse=(mistral_body,), hint=None, tested=False),
    ToolFormat(
        name="glm",
        signature=("<arg_key>", "<arg_value>"),
        open="<tool_call>", close="</tool_call>",
        parse=(glm_body, json_body), hint=None, tested=False),
    ToolFormat(
        name="qwen-xml",
        signature=("<function=", "<parameter="),
        open="<tool_call>", close="</tool_call>",
        parse=_TOOL_CALL_CHAIN, hint=None),
    ToolFormat(
        name="json",
        signature=("<tool_call>", "arguments"),
        open="<tool_call>", close="</tool_call>",
        parse=_TOOL_CALL_CHAIN, hint=None),
)

BY_NAME: dict[str, ToolFormat] = {r.name: r for r in ROWS}

# The dialects this server can parse back into structured tool_calls —
# capability.PARSEABLE_TOOL_FORMATS is this set, and http.py's refusal is
# "your model's template matched no row here".
NAMES = frozenset(BY_NAME)
# The served tier — what capability.PARSEABLE_TOOL_FORMATS actually is.
TESTED = frozenset(r.name for r in ROWS if r.tested)

# The row a scanner walks when nobody said which: the plain JSON shape.
# Unreachable in production (http.py refuses a
# tools request whose model matched no row), so this is a default, not a
# fallback with opinions.
DEFAULT = "json"


def row(name: str | None) -> ToolFormat:
    """Row by name, DEFAULT for anything unrecognized."""
    return BY_NAME.get(name or DEFAULT, BY_NAME[DEFAULT])


def identify(rendered_with_tools: str) -> str | None:
    """The tools-rendered prompt -> row name, or None for "no row matched".
    Ordered, first match wins; see ROWS on why the order is what it is."""
    for r in ROWS:
        if r.matches(rendered_with_tools):
            return r.name
    return None


def hint_for(name: str | None) -> str:
    """The row's prompt hint, or "" — see the `hint` column. No row in this
    table has one, so no serving path consumes this yet; it exists so a family
    whose template declares tools without teaching the emission form can be
    added as DATA rather than as a new branch in the prompt builder."""
    return row(name).hint or ""


def _cell(s: str) -> str:  # GFM: a pipe splits the cell even inside backticks
    return "`" + s.replace("|", r"\|") + "`"


def control_markdown_table() -> str:
    """The control-token and reasoning columns, as the docs render them —
    a second table rather than three more columns on the first, which is
    already as wide as a page. The doc's copy is pasted from this output;
    regenerate it when a row changes."""
    lines = ["| row | control tokens (special ids kept as text) | reasoning opens | reasoning closes |",
             "|---|---|---|---|"]
    for r in ROWS:
        ctl = ", ".join(_cell(t) for t in r.control_tokens) or "_(none declared)_"
        lines.append(f"| `{r.name}` | {ctl} | {_cell(r.think_open)} | {_cell(r.think_close)} |")
    return "\n".join(lines)
