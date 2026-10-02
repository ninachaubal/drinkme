"""serving/control.py — a row's markers that are SPECIAL ids, kept visible.

The premise, measured on real gemma-4-31B-it weights: the
model emitted `<|tool_call>call:get_weather{city:<|"|>Hilo<|"|>}<tool_call|>`
and the client got `call:get_weather{city:Hilo}` as prose, because every
marker is a special token and the serve loop decoded with
skip_special_tokens=True before any scanner looked. Three things are pinned
here, all CPU:

  the fake      a genuine `tokenizers` WordLevel vocabulary wrapped in
                PreTrainedTokenizerFast, with gemma's markers ADDED AS
                SPECIAL tokens — so the decode under test is transformers'
                real filter-then-decode, not a stand-in
  the pipeline  `walk()` runs the serve loop's exact composition — control
                decode -> IncrementalDetok -> ToolCallScanner -> ThinkSplitter,
                with the close-marker stop and its one-token lookahead —
                over a token stream, pushing the decode every `stride`
                tokens: 1, 3 and all-at-once must agree on calls, reasoning
                and content, or streaming and non-streaming disagree
  the real one  google/gemma-4-31B-it from the local HF cache (offline; a
                clean skip if absent): the markers ARE special ids — the
                bug's premise, proven rather than recalled — and the same
                walk lifts the call, files the thought, leaks no marker

The stop set: gemma's row resolves its close marker `<tool_call|>` to the
stop and `<|tool_call>` to the re-open; every other row resolves to
nothing, and Qwen's `</tool_call>` is confirmed plain text in the real
cached vocab rather than assumed.
"""

from __future__ import annotations

import os

import pytest

from drinkme.serving import control, tool_formats as tf
from drinkme.serving.detok import IncrementalDetok
from drinkme.serving.think import ThinkSplitter
from drinkme.serving.tools import ToolCallScanner

WEATHER_TOOLS = [{"type": "function", "function": {
    "name": "get_weather",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}}]

# the bookkeeping tokens first — `<eos>` at id 2, which is the toy causal
# LM's generation_config eos in test_serving_engines.py, so that engine's
# EOS set and this vocabulary agree — then gemma's control tokens, then two
# specials that are NOT the row's and must keep being stripped
SPECIALS = ["<unk>", "<pad>", "<eos>",
            "<|tool_call>", "<tool_call|>", "<|channel>", "<channel|>", '<|"|>',
            "<|turn>", "<turn|>"]
# the plain pieces the fake vocabulary can spell an emission from — one id
# each, decoded by concatenation (Fuse), so any string built from these
# round-trips exactly
PIECES = ["call:", "get_weather", "{", "city", ":", "Hilo", "}", ",", "unit", "f",
          "thought", "\n", "Giving", " the", " weather", ".", "The", " current",
          " is", " 78", "°F", " and", " cloudy", "hello", " world", "<think>",
          "</think>", "a", "b", "c", "[", "]", " ", "call", ":x", "{}", "y",
          "<|tool>declaration:"]  # the row's signature, so the probe can read it

VOCAB_SIZE = 64  # the toy causal LM in test_serving_engines.py has 64 logits


def fake_gemma_tokenizer():
    """A real PreTrainedTokenizerFast whose vocabulary spells gemma's markers
    as SPECIAL added tokens. WordLevel + a Fuse decoder: decode is the
    concatenation of the token strings, so text == "".join(pieces)."""
    from tokenizers import Tokenizer, decoders
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import WhitespaceSplit
    from transformers import PreTrainedTokenizerFast

    vocab = {}
    for w in SPECIALS + PIECES:
        vocab[w] = len(vocab)
    while len(vocab) < VOCAB_SIZE:
        vocab[f"tok{len(vocab)}"] = len(vocab)
    assert len(vocab) == VOCAB_SIZE
    backend = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
    backend.pre_tokenizer = WhitespaceSplit()
    backend.decoder = decoders.Fuse()
    tok = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="<unk>",
                                  pad_token="<pad>", eos_token="<eos>")
    tok.add_special_tokens({"additional_special_tokens":
                            [s for s in SPECIALS if s not in ("<unk>", "<pad>", "<eos>")]})
    # the gemma row's signature, so capability.probe names the row off the
    # tooled render (the real template writes `<|tool>declaration:NAME{...}`);
    # whitespace-delimited so WordLevel keeps the signature piece whole
    tok.chat_template = (
        "{% for m in messages %}{{ m['role'] }} : {{ m['content'] }} {% endfor %}"
        "{% if tools %}{% for t in tools %}<|tool>declaration: {{ t['function']['name'] }} "
        "<tool|> {% endfor %}{% endif %}"
        "{% if add_generation_prompt %}assistant :{% endif %}")
    return tok


def ids_of(tok, pieces: list[str]) -> list[int]:
    """Token ids for a list of vocabulary pieces — asserted known, so a typo
    in a test is a failure here and not an <unk> that decodes to nothing."""
    out = []
    for p in pieces:
        i = tok.convert_tokens_to_ids(p)
        assert tok.convert_ids_to_tokens(i) == p, f"{p!r} is not one token"
        out.append(i)
    return out


@pytest.fixture(scope="module")
def tok():
    return fake_gemma_tokenizer()


@pytest.fixture(scope="module")
def cs(tok):
    return control.resolve(tok, "gemma")


# gemma's emission for the receipt's prompt, in the order its template
# renders a thinking turn: the thought channel, then the call. The string
# delimiter around the value is the part the old decode silently ate.
THOUGHT_THEN_CALL = ["<|channel>", "thought", "\n", "Giving", " the", " weather", ".", "\n",
                     "<channel|>",
                     "<|tool_call>", "call:", "get_weather", "{", "city", ":",
                     '<|"|>', "Hilo", '<|"|>', "}", "<tool_call|>"]
# the receipt's ACTUAL order at thinking-off: the call, then a thought
# channel, then a hallucinated answer — everything after the close marker
# is what the stop exists to cut
CALL_THEN_PROSE = (["<|tool_call>", "call:", "get_weather", "{", "city", ":",
                    '<|"|>', "Hilo", '<|"|>', "}", "<tool_call|>"]
                   + ["<|channel>", "thought", "\n", "<channel|>",
                      "The", " current", " weather", " is", " 78", "°F", " and", " cloudy", "."])
HILO = [{"name": "get_weather", "arguments": {"city": "Hilo"}}]


def walk(tokenizer, cset: control.ControlSet, ids: list[int], *, tool_format: str,
         tools=None, stride: int = 1, in_think: bool = False):
    """The serve loop's text pipeline over a token stream, exactly as
    engines.py composes it (decode -> detok -> tool scanner -> think
    splitter, then the end-of-stream flushes in that order), with the
    close-marker stop and its lookahead. `stride` is how many tokens land
    before the decode is pushed again — the chunking under test.
    -> (reasoning, content, calls, stopped_at: index of the token the stop
    refused, or None)."""
    decode = cset.decoder(tokenizer)
    detok = IncrementalDetok()
    toolscan = ToolCallScanner(tools, tool_format) if tools else None
    split = ThinkSplitter(in_think, tool_format)
    reasoning, content, calls = [], [], []
    gen: list[int] = []
    stopped_at = None
    closed = False

    def deliver(delta: str) -> None:
        if toolscan is not None:
            delta, done = toolscan.feed(delta)
            calls.extend(done)
        r, c = split.feed(delta)
        reasoning.append(r)
        content.append(c)

    for k, t in enumerate(ids):
        if closed and t not in cset.reopen:
            stopped_at = k
            break
        closed = t in cset.stop_after
        gen.append(t)
        if len(gen) % stride == 0:
            deliver(detok.push(decode(gen)))
    rem = detok.flush(decode(gen))
    if toolscan is not None:
        r, done = toolscan.feed(rem)
        calls.extend(done)
        r += toolscan.flush()
        calls.extend(toolscan.flush_calls())
        rem = r
    a, b = split.feed(rem)
    reasoning.append(a)
    content.append(b)
    a, b = split.flush()
    reasoning.append(a)
    content.append(b)
    return "".join(reasoning), "".join(content), calls, stopped_at


def every_chunking(tokenizer, cset, ids, **kw):
    """The walk at stride 1, 3 and all-at-once — asserted identical, so one
    assertion on the return covers the chunk-invariance claim."""
    whole = walk(tokenizer, cset, ids, stride=len(ids) or 1, **kw)
    assert walk(tokenizer, cset, ids, stride=1, **kw) == whole, "one token at a time diverged"
    assert walk(tokenizer, cset, ids, stride=3, **kw) == whole, "three tokens at a time diverged"
    return whole


# --------------------------------------------------------------- resolve --


def test_the_fake_vocab_makes_the_markers_special(tok):
    """The premise, on the fake: without this the rest proves nothing."""
    for m in tf.row("gemma").control_tokens:
        assert tok.convert_tokens_to_ids(m) in tok.all_special_ids, m


def test_gemma_resolves_its_five_markers_and_the_close_is_the_stop(tok, cs):
    assert cs.row == "gemma"
    assert set(cs.ids.values()) == set(tf.row("gemma").control_tokens)
    assert all(i in tok.all_special_ids for i in cs.ids)
    assert cs.stop_after == {tok.convert_tokens_to_ids("<tool_call|>")}
    assert cs.reopen == {tok.convert_tokens_to_ids("<|tool_call>")}
    # the channel close is NOT a stop: gemma's answer follows it
    assert tok.convert_tokens_to_ids("<channel|>") not in cs.stop_after


@pytest.mark.parametrize("name", [r.name for r in tf.ROWS if r.name != "gemma"])
def test_every_other_row_resolves_to_nothing(tok, name):
    """Nothing new for Qwen (or any row whose markers are plain text): no
    kept ids, no stop, and the decoder is the plain skip_special_tokens
    call — the plain decode."""
    got = control.resolve(tok, name)
    assert got.ids == {} and got.stop_after == frozenset() and got.reopen == frozenset()
    ids = ids_of(tok, ["hello", "<|tool_call>", " world", "<eos>"])
    assert got.decoder(tok)(ids) == tok.decode(ids, skip_special_tokens=True) == "hello world"


@pytest.mark.parametrize("tool_format", [None, "unknown", "none", "no-such-row"])
def test_a_non_row_resolves_like_the_default_row(tok, tool_format):
    assert control.resolve(tok, tool_format).ids == {}


def test_a_marker_that_is_not_special_in_this_vocab_is_left_out():
    """A fine-tune that re-added `<|tool_call>` as plain text: the scanner
    already sees it, and no id-level stop is claimed for it."""
    from tokenizers import Tokenizer, decoders
    from tokenizers.models import WordLevel
    from transformers import PreTrainedTokenizerFast

    backend = Tokenizer(WordLevel({"<unk>": 0, "<|tool_call>": 1, "<tool_call|>": 2,
                                   "<|channel>": 3}, unk_token="<unk>"))
    backend.decoder = decoders.Fuse()
    plain = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="<unk>")
    plain.add_special_tokens({"additional_special_tokens": ["<|channel>"]})
    got = control.resolve(plain, "gemma")
    assert set(got.ids.values()) == {"<|channel>"}  # the one that IS special
    assert got.stop_after == frozenset() and got.reopen == frozenset()


def test_a_tokenizer_that_cannot_answer_resolves_to_nothing():
    class Mute:  # no all_special_ids, no convert_*: resolves to nothing, never a raise
        def decode(self, ids, skip_special_tokens=False):
            return ""
    got = control.resolve(Mute(), "gemma")
    assert got.ids == {} and got.stop_after == frozenset()


def test_describe_names_the_row_the_ids_and_the_stop(cs):
    d = cs.describe()
    assert d.startswith("gemma:") and "<|tool_call>=" in d and "turn ends after id" in d


# --------------------------------------------------------------- decoder --


def test_decoder_keeps_the_rows_markers_and_strips_every_other_special(tok, cs):
    ids = ids_of(tok, ["<|turn>", "<|tool_call>", "call:", '<|"|>', "Hilo", '<|"|>',
                       "<tool_call|>", "<turn|>", "<eos>"])
    assert cs.decoder(tok)(ids) == '<|tool_call>call:<|"|>Hilo<|"|><tool_call|>'
    # and the old call, for contrast: the bug, reproduced on the fake
    assert tok.decode(ids, skip_special_tokens=True) == "call:Hilo"


def test_decoder_matches_the_plain_call_wherever_no_marker_is_present(tok, cs):
    """Outside the markers the bytes are the old bytes: filter-then-decode
    is what skip_special_tokens does, at every prefix."""
    ids = ids_of(tok, ["<|turn>", "hello", " world", "<turn|>", "The", " weather", "<eos>"])
    dec = cs.decoder(tok)
    for n in range(len(ids) + 1):
        assert dec(ids[:n]) == tok.decode(ids[:n], skip_special_tokens=True)


def test_decoder_is_prefix_stable_across_a_marker(tok, cs):
    """IncrementalDetok diffs successive full decodes and requires each to
    extend the last; a marker landing must not re-form the text before it."""
    ids = ids_of(tok, THOUGHT_THEN_CALL)
    dec = cs.decoder(tok)
    prev = ""
    for n in range(1, len(ids) + 1):
        cur = dec(ids[:n])
        assert cur.startswith(prev), (n, prev, cur)
        prev = cur


# ---------------------------------------------------------- the pipeline --


def test_the_scanner_lifts_the_call_and_the_splitter_files_the_thought(tok, cs):
    ids = ids_of(tok, THOUGHT_THEN_CALL)
    reasoning, content, calls, stopped = every_chunking(tok, cs, ids, tool_format="gemma",
                                                        tools=WEATHER_TOOLS)
    assert calls == HILO
    assert reasoning == "Giving the weather.\n"
    assert content == ""
    assert stopped is None  # the close was the last token: nothing to refuse


def test_the_string_delimiter_reaches_the_parser(tok, cs):
    """`<|"|>` is a control token too — without it `{city:a,b}` is two keys."""
    ids = ids_of(tok, ["<|tool_call>", "call:", "get_weather", "{", "city", ":",
                       '<|"|>', "Hilo", ",", "f", '<|"|>', "}", "<tool_call|>"])
    _, content, calls, _ = every_chunking(tok, cs, ids, tool_format="gemma", tools=WEATHER_TOOLS)
    assert calls == [{"name": "get_weather", "arguments": {"city": "Hilo,f"}}]
    assert content == ""


def test_generation_stops_at_the_close_marker_before_the_hallucinated_answer(tok, cs):
    """The receipt's stream: call, then a thought channel and prose the
    model should never have produced. The stop refuses the first token after
    the close, and neither the thought nor the answer exists."""
    ids = ids_of(tok, CALL_THEN_PROSE)
    reasoning, content, calls, stopped = every_chunking(tok, cs, ids, tool_format="gemma",
                                                        tools=WEATHER_TOOLS)
    assert calls == HILO
    assert content == "" and reasoning == ""
    assert stopped == CALL_THEN_PROSE.index("<|channel>")


def test_a_second_call_continues_the_turn(tok, cs):
    """The one-token lookahead: gemma renders parallel calls back to back,
    so the row's own re-open after a close is not the end of the turn."""
    one = ["<|tool_call>", "call:", "get_weather", "{", "city", ":", '<|"|>', "Hilo", '<|"|>',
           "}", "<tool_call|>"]
    two = ["<|tool_call>", "call", ":x", "{}", "<tool_call|>"]
    ids = ids_of(tok, one + two + ["The", " weather"])
    _, content, calls, stopped = every_chunking(tok, cs, ids, tool_format="gemma", tools=WEATHER_TOOLS)
    assert [c["name"] for c in calls] == ["get_weather", "x"]
    assert content == ""
    assert stopped == len(one) + len(two)  # the prose after the SECOND close


def test_without_tools_the_markers_are_visible_text_not_marker_less_prose(tok, cs):
    """No scanner is looking (no tools offered), so the call is content —
    WITH its markers, the rule Qwen's plain-text `<tool_call>` has always
    had — and the stop still ends the turn at the close."""
    ids = ids_of(tok, CALL_THEN_PROSE)
    reasoning, content, calls, stopped = every_chunking(tok, cs, ids, tool_format="gemma")
    assert calls == [] and reasoning == ""
    assert content == '<|tool_call>call:get_weather{city:<|"|>Hilo<|"|>}<tool_call|>'
    assert stopped == CALL_THEN_PROSE.index("<|channel>")


def test_a_prompt_opened_channel_is_reasoning_until_the_close(tok, cs):
    """After a tool response with thinking on, gemma's template ends the
    prompt with `<|channel>thought\\n`: the stream begins inside the channel
    (in_think, from template.prompt_opens_think) and only the close crosses."""
    ids = ids_of(tok, ["Giving", " the", " weather", ".", "\n", "<channel|>",
                       "The", " current", " weather", "."])
    reasoning, content, calls, _ = every_chunking(tok, cs, ids, tool_format="gemma", in_think=True)
    assert reasoning == "Giving the weather.\n" and content == "The current weather."


def test_an_empty_thought_channel_is_no_reasoning_and_the_answer_is_content(tok, cs):
    """The thinking-off shape the template itself pre-fills, if the model
    writes it again: `<|channel>thought\\n<channel|>` then the answer."""
    ids = ids_of(tok, ["<|channel>", "thought", "\n", "<channel|>", "The", " weather", "."])
    reasoning, content, _, _ = every_chunking(tok, cs, ids, tool_format="gemma")
    assert reasoning == "" and content == "The weather."


def test_a_qwen_row_stream_through_the_same_walk_is_unchanged(tok):
    """The same pipeline on the json row: the `<think>` pieces are plain
    tokens in this vocab (as in Qwen's), nothing is kept or stopped, and
    the split is the one think.py always made."""
    got = control.resolve(tok, "json")
    ids = ids_of(tok, ["<think>", "hello", "</think>", "\n", "\n", "The", " weather", "."])
    reasoning, content, calls, stopped = every_chunking(tok, got, ids, tool_format="json")
    assert (reasoning, content, calls, stopped) == ("hello", "The weather.", [], None)


# ------------------------------------------------------ the real tokenizer --

GEMMA = ("google/gemma-4-31B-it", "842da3794eaa0b77d5f08bae87a17459d91ff475")
QWEN = {"json": ("Qwen/Qwen3-8B", "b968826d9c46dd6066d109eabc6255188de91218"),
        "qwen-xml": ("Qwen/Qwen3.8-27B", "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0")}


def _cached_tokenizer(repo: str, revision: str):
    """The real tokenizer from the local HF cache, offline — or a clean
    pytest.skip naming what's missing (test_serving_tool_formats.py's
    helper, same contract). Never downloads."""
    from huggingface_hub import try_to_load_from_cache

    hit = try_to_load_from_cache(repo, "tokenizer_config.json", revision=revision)
    if not isinstance(hit, str):
        pytest.skip(f"{repo}@{revision[:12]} is not in the local HF cache "
                    "(tokenizer_config.json not found) — nothing to decode")
    os.environ["HF_HUB_OFFLINE"] = "1"
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(repo, revision=revision, local_files_only=True)


REAL_EMISSION = ('<|channel>thought\nGiving the user the weather in Hilo.\n<channel|>'
                 '<|tool_call>call:get_weather{city:<|"|>Hilo<|"|>}<tool_call|>')


def test_real_gemma_markers_are_special_ids_and_the_pipeline_lifts_the_call():
    """THE premise and THE fix on the real vocabulary. Encode the receipt's
    dialect; every marker must be ONE special id (the premise — the old
    decode strips exactly these); then the walk at every chunking must
    return the structured call, the thought as reasoning, and content
    carrying neither marker."""
    tok = _cached_tokenizer(*GEMMA)
    row = tf.row("gemma")
    for m in row.control_tokens:
        i = tok.convert_tokens_to_ids(m)
        assert tok.convert_ids_to_tokens(i) == m and i in tok.all_special_ids, m
    ids = tok.encode(REAL_EMISSION, add_special_tokens=False)
    assert tok.decode(ids, skip_special_tokens=False) == REAL_EMISSION
    # the failure mode: skip_special_tokens hands the scanners this string
    assert tok.decode(ids, skip_special_tokens=True) == (
        "thought\nGiving the user the weather in Hilo.\ncall:get_weather{city:Hilo}")
    cs = control.resolve(tok, "gemma")
    assert cs.stop_after == {tok.convert_tokens_to_ids("<tool_call|>")} == {49}
    assert cs.reopen == {tok.convert_tokens_to_ids("<|tool_call>")} == {48}
    assert cs.decoder(tok)(ids) == REAL_EMISSION  # all specials here are the row's
    reasoning, content, calls, stopped = every_chunking(tok, cs, ids, tool_format="gemma",
                                                        tools=WEATHER_TOOLS)
    assert calls == HILO
    assert reasoning == "Giving the user the weather in Hilo.\n"
    assert content == "" and stopped is None
    for m in row.control_tokens:
        assert m not in content and m not in reasoning


def test_real_gemma_decode_outside_the_markers_is_byte_identical_to_the_plain_decode():
    """For text with no control token in it, the control decode must be the
    plain decode at EVERY prefix — the plain-chat transcript is unchanged by
    construction, not by luck. Includes gemma's other specials (`<turn|>`,
    `<eos>`, `<|think|>`) to prove they are still stripped, a grapheme
    cluster and non-Latin text for the byte-level path."""
    tok = _cached_tokenizer(*GEMMA)
    cs = control.resolve(tok, "gemma")
    dec = cs.decoder(tok)
    text = ("To solve 17 * 23, break 23 into 20 and 3.\n\n**Method 2** 🕵️‍♀️ é ü 日本語 "
            "<turn|> <eos> <|think|> <|tool_response>done")
    ids = tok.encode(text, add_special_tokens=False)
    assert tok.convert_tokens_to_ids("<turn|>") in ids  # the other specials are really in it
    for n in range(len(ids) + 1):
        assert dec(ids[:n]) == tok.decode(ids[:n], skip_special_tokens=True), n


def test_real_gemma_stop_after_the_close_cuts_the_receipts_hallucination():
    """The receipt's actual stream shape, on the real ids: the call, then
    `<|channel>thought\\n<channel|>` and an answer. The stop refuses the
    `<|channel>` and nothing after the call is generated."""
    tok = _cached_tokenizer(*GEMMA)
    cs = control.resolve(tok, "gemma")
    ids = tok.encode('<|tool_call>call:get_weather{city:<|"|>Hilo<|"|>}<tool_call|>'
                     '<|channel>thought\n<channel|>The current weather in Hilo is 78°F.',
                     add_special_tokens=False)
    reasoning, content, calls, stopped = every_chunking(tok, cs, ids, tool_format="gemma",
                                                        tools=WEATHER_TOOLS)
    assert calls == HILO and content == "" and reasoning == ""
    assert ids[stopped] == tok.convert_tokens_to_ids("<|channel>")


@pytest.mark.parametrize("name", sorted(QWEN))
def test_real_qwen_markers_are_plain_text_so_nothing_new_is_stopped(name):
    """Confirm, don't assume: Qwen's `</tool_call>` and `</think>` are added
    tokens with special=False in the real cached vocab, which is why the
    scanners always saw them — and why the row declares no control tokens
    and resolves to no stop."""
    tok = _cached_tokenizer(*QWEN[name])
    row = tf.row(name)
    for m in (row.open, row.close, row.think_open, row.think_close):
        i = tok.convert_tokens_to_ids(m)
        assert tok.convert_ids_to_tokens(i) == m, m  # one token...
        assert i not in tok.all_special_ids, m       # ...and not special
    got = control.resolve(tok, name)
    assert got.ids == {} and got.stop_after == frozenset()
