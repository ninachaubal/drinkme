"""Incremental detokenization: prefix-diff with grapheme-cluster hold-back.

Streaming decode emits deltas by diffing successive full decodes. Two hazards,
both seen in a real client (the pi TUI, on the 🕵️‍♀️ emoji):

1. **Partial graphemes are legal but hostile.** A ZWJ emoji (🕵️‍♀️ = five
   codepoints) arrives across several tokens; emitting each codepoint as its
   own SSE delta is spec-legal, but TUI clients redraw per delta and their
   width math on a half-assembled cluster leaves stale duplicate lines on
   screen. The polite server never splits a grapheme cluster across deltas:
   we withhold the trailing run of could-still-combine codepoints until a
   non-combining character lands or the stream ends.
2. **Held text must FLUSH at stream end.** Holding a trailing U+FFFD
   (partial UTF-8 inside one codepoint) for the next token is right until
   EOS, where there is no next token; if the scanners' flushes did not
   drain the detok's own hold, a stream ending mid-cluster would silently LOSE the
   tail. flush() returns everything still owed, including a genuinely
   dangling U+FFFD (the model really stopped mid-sequence; honesty over
   tidiness — OpenAI emits it too).

Pure class, no tokenizer dependency: callers pass the CURRENT full decode.

SuffixWindow sits just upstream of that: producing "the CURRENT full
decode" would otherwise mean decoding every generated token again, every
token. It produces the same string off a sliding window — byte-identical
or it does not run at all.
"""

from __future__ import annotations

import sys


def _could_extend(o: int) -> bool:
    """Codepoints that may still combine with what follows: ZWJ, variation
    selectors, skin modifiers, keycap, regional indicators, tags, and the
    emoji/pictographic blocks (approximation of Extended_Pictographic —
    over-holding a symbol for one token is harmless; splitting a cluster is
    the bug)."""
    return (
        o == 0x200D                      # zero-width joiner
        or o in (0xFE0E, 0xFE0F)         # variation selectors
        or 0x1F3FB <= o <= 0x1F3FF       # skin-tone modifiers
        or o == 0x20E3                   # combining enclosing keycap
        or 0x1F1E6 <= o <= 0x1F1FF       # regional indicators
        or 0xE0020 <= o <= 0xE007F       # tag sequence
        or 0x1F000 <= o <= 0x1FAFF       # emoji & pictograph blocks
        or 0x2600 <= o <= 0x27BF         # misc symbols / dingbats
        or 0x2190 <= o <= 0x2BFF         # arrows / misc technical (safe over-hold)
    )


class IncrementalDetok:
    """Feed successive full decodes; get back only safe-to-emit deltas."""

    # Longest real ZWJ sequences (family combos) run ~7-10 codepoints; the cap
    # keeps an emoji-spam tail from deferring the whole stream.
    HOLD_CAP = 12

    def __init__(self) -> None:
        self.emitted = ""  # what has actually been handed to the caller

    def push(self, full: str) -> str:
        """The per-token step. Returns the delta that is safe to emit now
        ("" when everything new is still combinable or the decode is not yet
        a clean extension of what was emitted)."""
        if full.endswith("�") or not full.startswith(self.emitted):
            return ""  # partial UTF-8, or the tail is being re-formed — hold
        hold = 0
        for ch in reversed(full):
            if hold >= self.HOLD_CAP or not _could_extend(ord(ch)):
                break
            hold += 1
        upto = len(full) - hold
        if upto <= len(self.emitted):
            return ""  # everything new is still potentially combining
        delta = full[len(self.emitted):upto]
        self.emitted = full[:upto]
        return delta

    def flush(self, full: str) -> str:
        """Stream over: emit everything still owed, hold nothing."""
        if full.startswith(self.emitted):
            delta = full[len(self.emitted):]
            self.emitted = full
            return delta
        # The final decode is not an extension of what we emitted (tail was
        # re-formed at the last token). We cannot unsend; emit the suffix
        # after the longest common prefix so no text is silently dropped.
        i = 0
        for a, b in zip(self.emitted, full):
            if a != b:
                break
            i += 1
        delta = full[i:]
        self.emitted = full
        return delta


class SuffixWindow:
    """decode(gen_ids), without re-decoding the whole generation every token.

    The serve loop called tok.decode(gen_ids) once per accepted token: O(n)
    work, n times, for a string that grew by one token — and in the
    constrained path the same string was then decoded AGAIN for every
    candidate (the hot-loop audit). This keeps a COMMITTED PREFIX and decodes only
    the tokens after it, so a token costs the window instead of the output.

    THE BAR IS BYTE-IDENTITY, not "close enough": deltas are diffs of
    successive full strings, so one wrong character at the join is a wrong
    delta on the wire and a client that renders it. Two things buy it.

    1. The anchor is never assumed. When the window slides, the true full
       decode is computed ONCE and the committed prefix is DERIVED from it
       (base_text = true minus the window's own decode), so
       base_text + window == decode(ids) exactly at that moment. If the two
       cannot be reconciled — a multi-byte character straddling the anchor
       makes the window's decode start with U+FFFD where the truth has the
       real character — the anchor simply does not move, and the honest full
       decode stands until a later boundary is clean.
    2. Every slide re-checks what the PREVIOUS window produced against that
       truth. Between slides the anchor rests on one assumption — that a
       decoder joins its window to the text before it the same way as more
       tokens arrive — which holds for byte-level BPE (concatenative),
       SentencePiece (a leading-space strip that cancels on both sides of the
       derivation) and the space-joining WordLevel toy, and is CHECKED rather
       than trusted: a decoder that violates it loses the window out loud on
       stderr instead of quietly emitting a wrong tail. DRINKME_DETOK_VERIFY=1
       moves that check to every token.
    """

    # Tokens, not characters: the grapheme hold-back downstream is 12
    # CHARACTERS, and even a one-character-per-token vocabulary keeps the
    # window comfortably wider than the longest cluster it must not split.
    WINDOW = 64

    def __init__(self, decode, window: int = WINDOW, verify: bool = False):
        self._decode = decode
        self.window = window
        self.verify = verify
        self.live = True
        self.base = 0        # tokens whose text is committed in base_text
        self.base_text = ""
        self._retry = 0      # token count before which sliding is not retried

    def full(self, ids: list[int]) -> str:
        """The decode of `ids` — the same string the tokenizer would give."""
        if not self.live:
            return self._decode(ids)
        n = len(ids)
        if n - self.base > 2 * self.window and n >= self._retry:
            return self._slide(ids)
        out = self.base_text + self._decode(ids[self.base:])
        if self.verify:
            self._verify(out, ids)
        return out

    def peek(self, ids: list[int], tok: int) -> str:
        """The decode of `ids + [tok]`, for a candidate that may be rejected."""
        if not self.live:
            return self._decode(list(ids) + [tok])
        out = self.base_text + self._decode(ids[self.base:] + [tok])
        if self.verify:
            self._verify(out, list(ids) + [tok])
        return out

    def _slide(self, ids: list[int]) -> str:
        true = self._decode(ids)
        if self.base_text + self._decode(ids[self.base:]) != true:
            self._drop("the decoder does not join its window to the text "
                       "before it the same way as tokens arrive")
            return true
        k = len(ids) - self.window
        win = self._decode(ids[k:])
        if true.endswith(win):
            self.base, self.base_text = k, true[: len(true) - len(win)]
        else:
            # e.g. a multi-byte character split across the boundary: not a
            # failure, just not an anchor. Try again a window from now, so a
            # decoder that never offers a clean boundary costs one full
            # decode and not two.
            self._retry = len(ids) + self.window
        return true

    def _verify(self, out: str, ids: list[int]) -> None:
        true = self._decode(ids)
        if out != true:
            raise AssertionError(
                f"suffix-window decode diverged from the tokenizer at "
                f"{len(ids)} tokens: {out[-40:]!r} vs {true[-40:]!r}")

    def _drop(self, why: str) -> None:
        self.live = False
        print(f"[drinkme.detok] suffix-window decode disabled: {why}; "
              f"falling back to a full decode per token", file=sys.stderr)
