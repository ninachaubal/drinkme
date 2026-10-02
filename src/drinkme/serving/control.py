"""Control tokens: a row's markers that are SPECIAL ids in the family's
vocab, resolved once per engine and kept visible to the scanners.

THE BUG THIS EXISTS FOR (measured on real gemma-4-31B-it weights). The model spoke its row's dialect to the letter —
`<|tool_call>call:get_weather{city:<|"|>Hilo<|"|>}<tool_call|>` — but every
one of those markers is a special token in gemma's vocab, and the serve
loop decoded with `skip_special_tokens=True` before the tool scanner
looked. The client got `call:get_weather{city:Hilo}` as prose, a
`thought` channel as more prose, and a hallucinated answer after it,
because nothing ended the turn at the call's close. Qwen's `<tool_call>`
and `<think>` survive only because Qwen's vocab adds them as PLAIN-TEXT
tokens (special=False); the scanners were never told the difference.

Two things, both read off the row (tool_formats.ToolFormat.control_tokens)
and resolved against THIS tokenizer at engine construction:

  1. THE DECODE KEEPS THE ROW'S CONTROL IDS AS TEXT. `decoder()` hands the
     serve loop a decode function that filters every special id EXCEPT the
     row's own out of the id list and decodes the rest with
     `skip_special_tokens=False` — so the markers come out as their own
     strings and everything else comes out exactly as before. Why this and
     not scanning on ids: the tool scanner and the think splitter are text
     machines whose ONE essential property is chunk-invariance (the same
     stream split anywhere gives the same verdict), proven per row at every
     byte offset. Handing them the same string they always scanned, with
     the markers present, keeps that proof; a second id-level scanner would
     need its own. A row with no control tokens gets the identical
     `decode(ids, skip_special_tokens=True)` call it always had — not an
     equivalent one, the same one — so every existing row is byte-for-byte
     unchanged. (For rows WITH control tokens the two are still the same
     bytes outside the markers: transformers' decode, fast and slow, is
     filter-then-decode, and the equivalence is asserted on the real gemma
     tokenizer at every prefix in tests/test_serving_control.py.)

     The client-facing text still carries no marker: the tool scanner
     consumes the call block into `tool_calls`, the think splitter consumes
     the channel into `reasoning_content`. A marker the model emits where
     no scanner is looking (a call with no tools offered) is visible
     text with its markers — the same rule as Qwen's plain-text
     `<tool_call>`, and more honest than marker-less prose.

  2. THE CLOSE MARKER ENDS THE TURN. `stop_after` is the row's call-close
     id: the serve loop feeds it through the decode and the scanner (so the
     block completes and the call is parsed) and then ends the turn — with
     ONE token of lookahead, so the row's own re-open (`reopen`, a second
     call: gemma's template renders parallel calls back to back) continues
     the turn and anything else — the family's own end-of-calls EOS, or
     hallucinated prose — ends it, that token unemitted, exactly as at EOS.
     A row whose close marker is plain text (Qwen's `</tool_call>`) resolves
     to no stop at all; Qwen ends its calls on its own EOS and that path is
     untouched.

Pure over the tokenizer's vocabulary: no torch, no model. Every engine
resolves at construction, beside the capability probe.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from . import tool_formats


@dataclass(frozen=True)
class ControlSet:
    """One engine's resolved control tokens. Empty (`ids` == {}) for every
    row whose markers are plain text — the common case, and the one where
    nothing downstream changes."""

    row: str
    ids: dict[int, str] = field(default_factory=dict)   # special id -> marker
    stop_after: frozenset[int] = frozenset()             # the call-close id
    reopen: frozenset[int] = frozenset()                 # the call-open id

    def decoder(self, tokenizer) -> Callable[[list[int]], str]:
        """The serve loop's decode: `skip_special_tokens=True` with this
        set's ids kept as their marker strings. The empty set returns the
        exact call every engine made before control tokens existed."""
        if not self.ids:
            return lambda ids: tokenizer.decode(ids, skip_special_tokens=True)
        drop = frozenset(_special_ids(tokenizer)) - frozenset(self.ids)

        def decode(ids: list[int]) -> str:
            return tokenizer.decode([i for i in ids if i not in drop],
                                    skip_special_tokens=False)
        return decode

    def describe(self) -> str:
        """One boot-log line's worth."""
        kept = ", ".join(f"{s}={i}" for i, s in sorted(self.ids.items(), key=lambda kv: kv[0]))
        stop = ", ".join(str(i) for i in sorted(self.stop_after)) or "none"
        return f"{self.row}: kept as text [{kept}]; turn ends after id {stop}"


def _special_ids(tokenizer) -> list[int]:
    ids = getattr(tokenizer, "all_special_ids", None)
    return list(ids) if ids else []


def resolve(tokenizer, tool_format: str | None) -> ControlSet:
    """The row named `tool_format` (a capability.tool_format value; anything that is
    not a row resolves to the DEFAULT row, which declares nothing) against
    this tokenizer's vocabulary. A declared marker that is NOT a single
    special id here — a fine-tune that re-added it as plain text — is left
    out: the scanner already sees it as text, and a stop on it would need
    a text match this module does not do. Never raises: a tokenizer that
    cannot answer resolves to the empty set."""
    row = tool_formats.row(tool_format)
    ids: dict[int, str] = {}
    if row.control_tokens:
        specials = set(_special_ids(tokenizer))
        for marker in row.control_tokens:
            try:
                i = tokenizer.convert_tokens_to_ids(marker)
                if (isinstance(i, int) and i in specials
                        and tokenizer.convert_ids_to_tokens(i) == marker):
                    ids[i] = marker
            except Exception:  # noqa: BLE001 — containment: no id, no change
                continue
    return ControlSet(
        row=row.name, ids=ids,
        stop_after=frozenset(i for i, s in ids.items() if s == row.close),
        reopen=frozenset(i for i, s in ids.items() if s == row.open))
