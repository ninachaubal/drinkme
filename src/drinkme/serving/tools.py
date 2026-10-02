"""OpenAI tool calling over any row of the format table — pure, CPU-only.

Three names, no torch, no tokenizer: validate_tools checks the OpenAI tool
shape at the HTTP boundary, ToolCallScanner turns a raw decoded delta stream
into (visible text, completed calls), and to_openai_calls renders parsed
calls in the chat.completion wire shape.

The DIALECT is data, not code: serving/tool_formats.py holds the table (one
row per family — markers, parser chain, streaming rule), capability.probe
picks the row off the model's own chat template at serve start, and the
scanner below walks whichever row it was handed. Adding a family is a row
there, not a branch here. The default row is `json`, the shape drinkme
served before the table existed, so a scanner constructed without a format
behaves exactly as it always did.

Every family's chat template renders the tool definitions into the prompt
itself, so the ONLY serving-side work is scanning the decoded text for that
row's markers.

The scanner is StopScanner's buffering model (sampling.py): feed() returns
only text that is provably not part of a tool call — a suffix that could
still open the row's start marker is held back until the next delta rules it
out, and text inside an open block is buffered until the close marker
arrives. A ONE-SIDED row (tool_formats: close is None, Mistral's
`[TOOL_CALLS]`) has no close marker, so its block is suppressed to the end
of the turn and resolved in flush() — the one case where calls surface from
flush_calls() instead of feed().

Two safety rules, both absolute: NEVER invent a call (a complete block whose
body no parser in the row's chain recognizes re-emerges verbatim as visible
text), and NEVER drop text (flush() releases an un-terminated tail — partial
start marker, or a block that never closed and never parsed — as visible
text at end of generation). Markers are paired first-open to first-close, no
nesting: no family here nests, and a stray inner marker degrades to visible
text rather than a guessed call.
"""

from __future__ import annotations

import json
import uuid

from . import tool_formats


def validate_tools(raw) -> list[dict]:
    """OpenAI `tools` request field -> the list the chat template renders.
    Raises ValueError with a client-facing message on anything malformed;
    the HTTP layer returns it as a 400 verbatim.

    Every dialect's tools come through here, so this is where the two
    optional fields get their defaults, before any template sees them: a
    missing or null `description` becomes "" and missing or null
    `parameters` become {}. llama.cpp's defaults are the same
    (common/chat.cpp, common_chat_tools_parse_oaicompat:
    `function.value("description", "")`, `function.value("parameters",
    json::object())`), and null reads as missing here as it does in the
    Messages and Responses adapters. Some official templates read both
    unguarded (Muse-Glimmer's `fn.description | tojson`), and a missing one
    would otherwise fail the render. The client's dicts are not modified."""
    if not isinstance(raw, list):
        raise ValueError("'tools' must be an array of tool definitions.")
    out = []
    for i, t in enumerate(raw):
        if not isinstance(t, dict):
            raise ValueError(f"tools[{i}] must be an object.")
        if t.get("type") != "function":
            raise ValueError(f"tools[{i}].type must be 'function'.")
        fn = t.get("function")
        if not isinstance(fn, dict):
            raise ValueError(f"tools[{i}].function must be an object.")
        name = fn.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError(f"tools[{i}].function.name must be a non-empty string.")
        if fn.get("description") is not None and not isinstance(fn["description"], str):
            raise ValueError(f"tools[{i}].function.description must be a string.")
        if fn.get("parameters") is not None and not isinstance(fn["parameters"], dict):
            raise ValueError(f"tools[{i}].function.parameters must be an object "
                             "(a JSON Schema).")
        # a null keeps its place and a missing field is appended, so a tool
        # that has both renders byte for byte as sent (`tool | tojson` writes
        # keys in dict order)
        fn = dict(fn)
        if fn.get("description") is None:
            fn["description"] = ""
        if fn.get("parameters") is None:
            fn["parameters"] = {}
        out.append({**t, "function": fn})
    return out


def _open_prefix_len(s: str, marker: str) -> int:
    """Length of the longest suffix of s that is a proper prefix of the row's
    start marker — the lookahead that must be held back (StopScanner's hold,
    marker-shaped; think.py's _tag_prefix_len is the same function)."""
    for k in range(min(len(s), len(marker) - 1), 0, -1):
        if marker.startswith(s[-k:]):
            return k
    return 0


class ToolCallScanner:
    """Incremental tool-call extraction over a decoded delta stream.

    feed(delta) -> (visible_text, completed_calls). Visible text is safe to
    send to the client immediately; completed calls are parsed
    {"name", "arguments"} dicts in emission order. flush() ends the stream:
    whatever is still buffered (a partial start marker, an unclosed block)
    comes back as visible text — generated bytes are never silently dropped
    — and flush_calls(), read straight after, carries the calls a ONE-SIDED
    row resolved at the end of the turn (always empty for every other row).

    `tool_format` is a tool_formats row name, normally the model's announced
    capability.tool_format. Anything unrecognized (including None) walks the
    DEFAULT row."""

    def __init__(self, tools: list | None = None, tool_format: str | None = None):
        self.row = tool_formats.row(tool_format)
        self._buf = ""
        self._in_call = False
        self._end_calls: list[dict] = []
        # function name -> {param: declared type} from the request's own tool
        # schemas — the only honest source for typing the rows whose values
        # arrive as raw text (qwen-xml, glm, minimax)
        self._types: dict = {}
        for t in tools or []:
            fn = t.get("function") if isinstance(t, dict) else None
            if not isinstance(fn, dict):
                continue
            props = (fn.get("parameters") or {}).get("properties") or {}
            self._types[fn.get("name")] = {
                k: (v.get("type") if isinstance(v, dict) else None)
                for k, v in props.items()}

    def feed(self, delta: str) -> tuple[str, list[dict]]:
        self._buf += delta
        row = self.row
        visible: list[str] = []
        done: list[dict] = []
        while True:
            if self._in_call:
                if row.close is None:
                    break  # one-sided row: suppressed to the end of the turn
                j = self._buf.find(row.close)
                if j < 0:
                    break  # block still open: buffer, emit nothing from inside
                body, self._buf = self._buf[:j], self._buf[j + len(row.close):]
                self._in_call = False
                calls = row.parse_body(body, self._types)
                if calls is None:  # not a call -> the block re-emerges verbatim
                    visible.append(row.open + body + row.close)
                else:
                    done += calls
            else:
                i = self._buf.find(row.open)
                if i >= 0:
                    visible.append(self._buf[:i])
                    self._buf = self._buf[i + len(row.open):]
                    self._in_call = True
                    continue  # scan the rest: the close may already be here
                keep = _open_prefix_len(self._buf, row.open)  # still a marker?
                cut = len(self._buf) - keep
                visible.append(self._buf[:cut])
                self._buf = self._buf[cut:]
                break
        return "".join(visible), done

    def flush(self) -> str:
        """End of stream: the un-terminated tail, as the text it really was.

        For a ONE-SIDED row this is also where the block is PARSED — the end
        of the turn is its only terminator. Recognised calls go to
        flush_calls(), which the caller reads immediately after this; a body
        no parser recognized re-emerges as text, as everywhere else."""
        buf, self._buf = self._buf, ""
        self._end_calls = []
        if not self._in_call:
            return buf
        self._in_call = False
        if self.row.close is None:
            calls = self.row.parse_body(buf, self._types)
            if calls is not None:
                self._end_calls = calls
                return ""
        return self.row.open + buf

    def flush_calls(self) -> list[dict]:
        """The calls flush() resolved at the end of the turn, drained once.
        Always empty for a close-delimited row — only a one-sided row
        (Mistral's `[TOOL_CALLS]`) can finish a call on EOS."""
        calls, self._end_calls = self._end_calls, []
        return calls


def split_calls(text: str, tools: list | None = None,
                tool_format: str | None = None) -> tuple[str, list[dict]]:
    """Whole-string form: (visible text, calls). Feed + flush, so the
    non-streaming answer is the streamed one concatenated, by construction
    (think.split_text's shape, for the same reason)."""
    s = ToolCallScanner(tools, tool_format)
    visible, calls = s.feed(text)
    visible += s.flush()
    return visible, calls + s.flush_calls()


def to_openai_calls(parsed: list[dict]) -> list[dict]:
    """Parsed calls -> chat.completion `tool_calls` entries. Per the OpenAI
    shape, function.arguments is a JSON-ENCODED STRING, not an object."""
    return [{"id": f"call_{uuid.uuid4().hex[:12]}",
             "type": "function",
             "function": {"name": c["name"],
                          "arguments": json.dumps(c.get("arguments", {}))}}
            for c in parsed]
