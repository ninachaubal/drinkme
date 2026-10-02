"""The think channel: reasoning split off the content stream, incrementally.

The two showcase models do NOT open the block the same way (the capability
probe):

  Qwen3.8-27B  the chat template writes `<think>\\n` at the end of every
               generation prompt, so what the model actually GENERATES is
               reasoning + "</think>\\n\\n" + answer, and the OPENING tag
               never crosses the wire (measured on the real template).
               Re-rendering a prior plain-content reply from
               history also re-opens an EMPTY `<think>\\n\\n</think>\\n\\n`
               pair for that turn — the prefix-cache consequence of that is
               docs/serve-prefix-slots.md and
               bench/prefix_slots_verify.py's module docstring (the
               "THINKING AND THE HISTORY RE-RENDER" section).
  Qwen3-8B     thinking is self-initiated, not template-forced: at the
               pinned revision's default (enable_thinking left on), the
               generation prompt opens no block at all, and the model emits
               its own literal `<think>` mid-generation if it decides to
               reason — the SEEK state's literal-tag branch below is what
               catches that case, so the SPLITTING behaviour is already
               right regardless. Only `enable_thinking: false` makes the
               8B's template pre-fill the whole empty pair the way the 27B's
               default does, and then the output carries no tags at all.

See bench/prefix_slots_verify.py's asymmetry note for the two families'
full, measured, mirror-image behaviour under history re-render.

So the closing tag is the only landmark the STREAM carries, and whether the
text before it is reasoning depends on something the stream cannot tell you:
did the prompt open a block? That is `in_think`, supplied by the caller, who
reads it off the prompt itself (the engine's StreamStart.opens_think, from
template.render_prompt) rather than guessing from the model's name — a family that does not think would have its
whole answer filed as reasoning, which is an empty answer. Everything else
follows from the bytes:

  in_think=False  the default and the no-think path: text is content, byte
                  for byte, held for nothing — UNLESS the stream itself opens
                  with a literal `<think>` (models that do emit it), which
                  switches to reasoning exactly as if the prompt had.
  in_think=True   the stream begins inside the block: reasoning until the
                  first `</think>`, content after it. Never closed = the turn
                  was truncated mid-think: all reasoning, content empty.

Chunk-invariance is the essential property: feeding "a</think>\\n\\nb" whole,
in two pieces, or one byte at a time must produce the same two channels, or
streaming and non-streaming answers would disagree. Nothing here may look
ahead past what it has; a suffix that could still be growing into the tag
being watched for is HELD (tools.py's model, tag-shaped) and released the
moment the match fails. No byte ever crosses channels, and the only bytes
dropped are structural: the tags themselves, whitespace before an opening
tag, and ONE `\\n\\n` (or `\\n`) after the close.

THE TAGS ARE THE ROW'S. `<think>`/`</think>` is Qwen's convention and
the default; gemma names its channel — `<|channel>thought\\n` ... `\\n<channel|>`
— and the same machine walks it, reading think_open/think_close off the
tool_formats row the capability probe picked for the model (the splitter
takes the row NAME, http.py hands it the announced tool_format). Gemma's
markers are special tokens a plain decode would strip before this
splitter ever saw them; serving/control.py keeps them as text, and from here on
they are just a different pair of tags. The machine is otherwise unchanged:
the same states, the same holds, the same one structural newline eaten
after the open and one separator after the close.

ADDRESSED MESSAGES are the other shape a row can declare (its message_open /
recipient_open / message_body columns), and AddressedSplitter walks them.
Muse-Glimmer's ATEM turn is not one stream with a tag pair but a run of
messages, each to a recipient:

  <|start|>assistant to=self<|message|>REASONING<|eom|><|start|>assistant to=user<|message|>ANSWER<|eot|>

Its generation prompt ends `<|start|>assistant`, so what the model generates
starts at ` to=self`. A message to `self` (the row's think_channel) is
reasoning, and every other message is content. The response_template the
checkpoint ships in its tokenizer_config.json declares the same split
(reasoning_content opens at `to=self<|message|>`, content at
`to=user<|message|>`), and tests/test_serving_glimmer_channel.py holds this
splitter to transformers' parse of it. Nothing is eaten around a header or after `<|eom|>`, because the
chat template renders `reasoning_content` and `content` back with no
separator of its own. The channels therefore carry exactly the bytes the model
wrote, and a client that sends them back gets a prompt that EXTENDS the prefix
cache (engines.py's extending rule). Measured on the real 30B before this
splitter existed: the whole reply came back as `content`, the template
re-rendered it under `to=user` where the model had written `to=self`, and no
later turn reused a token (tests/test_serving_glimmer_channel.py reproduces
that miss and the fix). transformers' own parse of the response_template
strips each field's whitespace, which would miss the same way wherever the
model ended its reasoning with a newline, as it does.

`splitter()` picks the machine off the row, which is what http.py calls.
"""

from __future__ import annotations

from . import tool_formats

# Every distinct opening tag any row declares: what a prompt's tail can end
# with when the template force-opened the channel (template._ends_open_think
# checks all of them — the row is the model's, the prompt is what it wrote).
OPENS: tuple[str, ...] = tuple(dict.fromkeys(r.think_open for r in tool_formats.ROWS))

_WS = " \t\r\n"


def _tag_prefix_len(s: str, tag: str) -> int:
    """Length of the longest suffix of s that is a proper prefix of tag — the
    lookahead a chunked feed must hold back (tools.py's _open_prefix_len,
    generalized over the tag)."""
    for k in range(min(len(s), len(tag) - 1), 0, -1):
        if tag.startswith(s[-k:]):
            return k
    return 0


class ThinkSplitter:
    """Incremental reasoning/content split over a decoded delta stream.

    feed(piece) -> (reasoning, content): either may be "", both may be
    non-empty when a piece straddles the close tag. flush() ends the stream
    and releases everything still held. Held text is bounded by the tag being
    watched for (one character short of the row's longest tag), except a run
    of leading whitespace before a possible opening tag, which cannot be
    classified until it ends."""

    SEEK, OPENED, THINK, GAP, TEXT = "seek", "opened", "think", "gap", "text"

    def __init__(self, in_think: bool = False, tool_format: str | None = None) -> None:
        # the row's tags: `tool_format` is a tool_formats row name (the model's
        # announced tool_format); anything else — None, "unknown", "none" —
        # walks the DEFAULT row's `<think>`/`</think>`, which is what every
        # splitter did before rows carried these columns
        row = tool_formats.row(tool_format)
        self.open, self.close = row.think_open, row.think_close
        self._buf = ""
        self._state = self.THINK if in_think else self.SEEK

    def feed(self, piece: str) -> tuple[str, str]:
        self._buf += piece
        reasoning: list[str] = []
        content: list[str] = []
        while True:
            if self._state == self.SEEK:
                # optional leading open tag, possibly preceded by whitespace
                stripped = self._buf.lstrip(_WS)
                if stripped.startswith(self.open):
                    self._buf = stripped[len(self.open):]  # tag + its padding: gone
                    self._state = self.OPENED
                    continue
                if not stripped or self.open.startswith(stripped):
                    break  # all whitespace so far, or a partial tag — hold
                content.append(self._buf)  # not a tag: the bytes were content
                self._buf = ""
                self._state = self.TEXT
                continue
            if self._state == self.OPENED:
                if not self._buf:
                    break  # one byte of lookahead: the tag's optional "\n"
                if self._buf[0] == "\n":
                    self._buf = self._buf[1:]
                self._state = self.THINK
                continue
            if self._state == self.THINK:
                i = self._buf.find(self.close)
                if i >= 0:
                    reasoning.append(self._buf[:i])
                    self._buf = self._buf[i + len(self.close):]
                    self._state = self.GAP
                    continue
                keep = _tag_prefix_len(self._buf, self.close)  # could still be a tag
                cut = len(self._buf) - keep
                reasoning.append(self._buf[:cut])
                self._buf = self._buf[cut:]
                break
            if self._state == self.GAP:
                # exactly ONE separator after the close tag, never more: the
                # template writes "</think>\n\n", and a blank line the model
                # meant to write would be the third newline.
                if not self._buf:
                    break
                if self._buf.startswith("\n\n"):
                    self._buf = self._buf[2:]
                elif self._buf == "\n":
                    break  # could still grow into "\n\n" — hold the one byte
                elif self._buf.startswith("\n"):
                    self._buf = self._buf[1:]
                self._state = self.TEXT
                continue
            if self._buf:  # TEXT: passthrough, nothing is ever held here
                content.append(self._buf)
                self._buf = ""
            break
        return "".join(reasoning), "".join(content)

    def flush(self) -> tuple[str, str]:
        """End of stream: everything still held, in its true channel.

        THINK means the block was opened (by the prompt or by a literal tag)
        and never closed — a turn truncated mid-think, which is reasoning, not
        an empty answer. A partial tag that never completed is the text it
        really was, never dropped; the one held separator newline is."""
        buf, self._buf = self._buf, ""
        if self._state == self.THINK:
            return buf, ""
        if self._state == self.GAP:
            return "", ""  # the held "\n" WAS the separator
        return "", buf  # SEEK (partial/absent open tag), OPENED (empty), TEXT


class AddressedSplitter:
    """Incremental reasoning/content split for a row that declares addressed
    messages (the module docstring's ATEM shape). ThinkSplitter's contract:
    feed(piece) -> (reasoning, content), flush() releases what is held, and
    any chunking of one stream gives the same two channels.

    HEAD holds the start of a message header until it completes or cannot:

        [message_open] WS* [recipient_open NAME] message_body

    A complete header is dropped and opens BODY in its recipient's channel.
    It is bounded by MAX_RECIPIENT, and text that cannot be a header is
    content, byte for byte: a stream that never speaks the grammar comes out
    exactly as it went in. BODY routes text until the row's think_close
    (`<|eom|>`), which is dropped and returns to HEAD. Only the partial
    close tag is ever held there.

    A message to a tool whose call block the engine's ToolCallScanner
    consumed has an empty body. One whose block it did not consume (no
    tools offered, or a block no parser read) is content. Nothing is
    dropped but the headers and the closes. `in_think` (the prompt ended
    inside the reasoning message, template.py's read-back of think_open)
    starts in BODY on the reasoning channel."""

    HEAD, BODY = "head", "body"
    # a recipient is a tool's name; OpenAI caps those at 64 characters
    MAX_RECIPIENT = 128

    def __init__(self, in_think: bool = False, tool_format: str | None = None) -> None:
        row = tool_formats.row(tool_format)
        if row.message_open is None:
            raise ValueError(f"tool_format {row.name!r} declares no addressed messages")
        self.start, self.to, self.body = row.message_open, row.recipient_open, row.message_body
        self.close, self.reasoner = row.think_close, row.think_channel
        self._buf = ""
        self._state = self.BODY if in_think else self.HEAD
        self._reasoning = in_think  # the open message's channel

    def _header(self, buf: str) -> tuple[str, int, str | None]:
        """("full", end, recipient) when buf starts with a whole header,
        ("partial", 0, None) while it could still grow into one, ("none", 0,
        None) once it cannot. Each verdict is a prefix relation, so more
        text can only move "partial" on, never a "none" back."""
        i = 0
        if buf.startswith(self.start):
            i = len(self.start)
        elif self.start.startswith(buf):
            return "partial", 0, None
        n = len(buf)
        while i < n and buf[i] in _WS:
            i += 1
        if i == n:
            return "partial", 0, None
        recipient = None
        if buf.startswith(self.to, i):
            j = k = i + len(self.to)
            while k < n and buf[k] not in _WS and buf[k] != "<":
                k += 1
            if k - j > self.MAX_RECIPIENT:
                return "none", 0, None
            if k == n:
                return "partial", 0, None
            if k == j:
                return "none", 0, None
            recipient, i = buf[j:k], k
        elif self.to.startswith(buf[i:]):
            return "partial", 0, None
        if buf.startswith(self.body, i):
            return "full", i + len(self.body), recipient
        if self.body.startswith(buf[i:]):
            return "partial", 0, None
        return "none", 0, None

    def feed(self, piece: str) -> tuple[str, str]:
        self._buf += piece
        reasoning: list[str] = []
        content: list[str] = []
        while self._buf:
            if self._state == self.HEAD:
                verdict, end, recipient = self._header(self._buf)
                if verdict == "partial":
                    break
                # a header opens its recipient's channel; text that is not
                # one is content, and BODY scans it for a close like any other
                self._buf = self._buf[end:]
                self._reasoning = verdict == "full" and recipient == self.reasoner
                self._state = self.BODY
                continue
            out = reasoning if self._reasoning else content
            i = self._buf.find(self.close)
            if i >= 0:
                out.append(self._buf[:i])
                self._buf = self._buf[i + len(self.close):]
                self._state = self.HEAD
                continue
            cut = len(self._buf) - _tag_prefix_len(self._buf, self.close)
            out.append(self._buf[:cut])
            self._buf = self._buf[cut:]
            break
        return "".join(reasoning), "".join(content)

    def flush(self) -> tuple[str, str]:
        """End of stream. A partial close is its message's text; a header
        that never completed (a turn cut off between messages) is the text
        it really was, as ThinkSplitter's partial tag is."""
        buf, self._buf = self._buf, ""
        if self._state == self.BODY and self._reasoning:
            return buf, ""
        return "", buf


def splitter(in_think: bool = False, tool_format: str | None = None):
    """The splitter the row calls for: AddressedSplitter for a row that
    declares addressed messages (message_open), ThinkSplitter for every
    other row and for anything that is not a row name."""
    if tool_formats.row(tool_format).message_open is not None:
        return AddressedSplitter(in_think, tool_format)
    return ThinkSplitter(in_think, tool_format)


def split_text(text: str, in_think: bool = False,
               tool_format: str | None = None) -> tuple[str, str]:
    """Whole-string form: (reasoning, content). Feed + flush, so the
    non-streaming answer is the streamed one concatenated, by construction."""
    s = splitter(in_think, tool_format)
    r1, c1 = s.feed(text)
    r2, c2 = s.flush()
    return r1 + r2, c1 + c2
