"""ThinkSplitter: the two channels, and the chunk boundaries that hazard them.

Every case here is asserted under THREE feeds — the whole string, one byte at
a time, and every single two-piece split — because the splitter's contract is
that chunking is invisible: a tag arriving across two decode deltas must land
exactly where it lands when it arrives whole, or a streamed answer and a
non-streamed one disagree about what the model said.
"""

from drinkme.serving.think import ThinkSplitter, split_text


def run(pieces, in_think=False, tool_format=None):
    """Feed the pieces in order, flush, return (reasoning, content)."""
    s = ThinkSplitter(in_think, tool_format)
    r, c = [], []
    for p in pieces:
        a, b = s.feed(p)
        r.append(a)
        c.append(b)
    a, b = s.flush()
    return "".join(r) + a, "".join(c) + b


def split(text, in_think=False, tool_format=None):
    """(reasoning, content) — asserted identical across every chunking, so a
    single assertion in a test covers the straddled-tag case for free."""
    want = run([text], in_think, tool_format)
    assert run(list(text), in_think, tool_format) == want, "one byte at a time diverged"
    for k in range(1, len(text)):
        got = run([text[:k], text[k:]], in_think, tool_format)
        assert got == want, f"split at {k} ({text[:k]!r} | {text[k:]!r}) diverged"
    assert split_text(text, in_think, tool_format) == want  # the whole-string convenience
    return want


# ------------------------------------------------- the prompt opened it (Qwen3) --

# The measured shape: with add_generation_prompt the prompt ends "<think>\n",
# so generation starts INSIDE the block and only the close tag is in the
# stream (real Qwen3.8-27B template).


def test_close_tag_splits_the_two_channels():
    assert split("weighing it</think>\n\nThe answer.", True) == (
        "weighing it", "The answer.")


def test_only_the_first_close_tag_splits():
    # a stray tag in the answer is the answer's own text, not a second split
    assert split("hm</think>\n\nas in </think> the tag", True) == (
        "hm", "as in </think> the tag")


def test_truncated_mid_think_is_all_reasoning():
    # max_tokens ran out before the model stopped thinking: content is empty,
    # not "the thinking, mislabelled as an answer"
    assert split("still weighing the third option", True) == (
        "still weighing the third option", "")


def test_partial_close_tag_at_eos_is_reasoning_not_a_tag():
    assert split("done</thi", True) == ("done</thi", "")


def test_empty_stream():
    assert split("", True) == ("", "")
    assert split("", False) == ("", "")


# ----------------------------------------------------- the stream opened it --


def test_literal_opening_tag_switches_without_the_flag():
    assert split("<think>\nweighing</think>\n\nanswer") == ("weighing", "answer")


def test_opening_tag_may_be_preceded_by_whitespace_and_needs_no_newline():
    assert split("  \n<think>weighing</think>\n\nanswer") == ("weighing", "answer")


def test_literal_opening_tag_never_closed_is_all_reasoning():
    assert split("<think>\nran out of budget") == ("ran out of budget", "")


def test_only_one_newline_after_the_opening_tag_is_eaten():
    assert split("<think>\n\nweighing</think>\n\nanswer") == ("\nweighing", "answer")


# ------------------------------------------------------- no think, no touch --


def test_no_tags_is_content_byte_for_byte():
    assert split("Just the answer, thanks.") == ("", "Just the answer, thanks.")


def test_no_think_passthrough_holds_nothing():
    """The default path must not buffer: a delta in is a delta out."""
    s = ThinkSplitter()
    assert s.feed("Just ") == ("", "Just ")
    assert s.feed("the answer.") == ("", "the answer.")
    assert s.flush() == ("", "")


def test_bare_close_tag_without_an_opening_is_left_alone():
    # nothing opened a block, so nothing before the tag was reasoning; the
    # bytes stay exactly what the model wrote (the default behavior)
    assert split("a </think> b") == ("", "a </think> b")


def test_angle_brackets_that_are_not_tags_stream_as_content():
    assert split("<thin> <thinking> x < y") == ("", "<thin> <thinking> x < y")


def test_unfinished_opening_tag_at_eos_is_content_not_dropped():
    assert split("<thi") == ("", "<thi")
    assert split("   ") == ("", "   ")


# ------------------------------------------------ the separator after the tag --


def test_one_blank_line_after_the_close_is_stripped():
    assert split("r</think>\n\nc", True) == ("r", "c")


def test_a_single_newline_after_the_close_is_stripped():
    assert split("r</think>\nc", True) == ("r", "c")


def test_no_newline_after_the_close_strips_nothing():
    assert split("r</think>c", True) == ("r", "c")


def test_exactly_one_separator_is_stripped_not_the_answers_own_blank_line():
    assert split("r</think>\n\n\nc", True) == ("r", "\nc")


def test_stream_that_ends_on_the_separator_emits_no_empty_content():
    assert split("r</think>\n", True) == ("r", "")
    assert split("r</think>\n\n", True) == ("r", "")


# --------------------------------------------------------------- invariants --

CASES = [("weighing it</think>\n\nThe answer.", True),
         ("<think>\nweighing</think>\n\nThe answer.", False),
         ("no tags at all", False),
         ("truncated mid-thought", True),
         ("r</think>\n\n</think>\n\nc", True),
         ("  <think>x</think>y", False)]


def test_channels_never_cross_and_only_structure_is_dropped():
    """Every byte lands in exactly one channel, in order, and what does NOT
    arrive is only ever the tags, the whitespace before an opening tag, and
    one separator newline run."""
    for text, in_think in CASES:
        reasoning, content = split(text, in_think)
        rest = text
        for got in (reasoning, content):  # both are subsequences, in order
            for ch in got:
                i = rest.find(ch)
                assert i >= 0, f"{ch!r} of {text!r} is not in stream order"
                rest = rest[i + 1:]
        dropped = len(text) - len(reasoning) - len(content)
        assert 0 <= dropped <= len("<think>") + len("</think>") + 3


def test_incremental_equals_whole_string_on_pathological_chunkings():
    for text, in_think in CASES:
        split(text, in_think)  # asserts all three feeds internally


def test_flush_is_idempotent_and_safe_to_repeat():
    s = ThinkSplitter(True)
    assert s.feed("a</thi") == ("a", "")
    assert s.flush() == ("</thi", "")
    assert s.flush() == ("", "")


# ------------------------------------------------- the row's own tags (gemma) --

# gemma names its channel: the template writes `<|channel>thought\n` + text +
# `\n<channel|>` and the answer follows the close directly. The markers are
# special tokens serving/control.py keeps as text; from here they are just
# a different pair of tags on the same machine (tool_format="gemma" reads them off
# the tool_formats row). Every case is asserted under every chunking, as
# above.

GEMMA = "<|channel>thought\nGiving the user the weather.\n<channel|>The weather is fine."


def test_gemma_literal_channel_at_stream_start_is_reasoning():
    assert split(GEMMA, tool_format="gemma") == (
        "Giving the user the weather.\n", "The weather is fine.")


def test_gemma_prompt_opened_channel_only_the_close_crosses():
    # after a tool response with thinking on, the template ends the prompt
    # inside `<|channel>thought\n`; in_think comes from the prompt read-back
    assert split("Giving the weather.\n<channel|>The weather.", True, tool_format="gemma") == (
        "Giving the weather.\n", "The weather.")


def test_gemma_empty_channel_is_no_reasoning():
    # the thinking-off pre-fill, if the model writes it again itself
    assert split("<|channel>thought\n<channel|>The weather.", tool_format="gemma") == (
        "", "The weather.")


def test_gemma_truncated_mid_channel_is_all_reasoning():
    assert split("<|channel>thought\nstill weighing", tool_format="gemma") == ("still weighing", "")


def test_gemma_partial_close_at_eos_is_reasoning_not_a_tag():
    assert split("<|channel>thought\nhm <chan", tool_format="gemma") == ("hm <chan", "")


def test_gemma_qwen_tags_are_plain_content_on_the_gemma_row():
    # the row's tags are the only tags: Qwen's are text here, byte for byte
    assert split("<think>not a tag here</think>\n\nx", tool_format="gemma") == (
        "", "<think>not a tag here</think>\n\nx")


def test_gemma_tags_are_plain_content_on_the_default_row():
    # and the mirror image: nothing about gemma's tags is hardcoded anywhere
    assert split(GEMMA) == ("", GEMMA)


def test_gemma_no_tags_is_content_byte_for_byte():
    assert split("The weather is fine.", tool_format="gemma") == ("", "The weather is fine.")


def test_gemma_channel_open_holds_only_a_partial_tag():
    # `<|channel>thought` is 16 characters: a prefix of it is held, anything
    # that rules it out is released at once as content
    s = ThinkSplitter(False, "gemma")
    assert s.feed("<|chan") == ("", "")
    assert s.feed("nel>tho") == ("", "")
    assert s.feed("ught\nreason") == ("reason", "")
    assert s.feed("\n<channel|>ans") == ("\n", "ans")
    assert s.flush() == ("", "")


def test_a_non_row_name_walks_the_default_row():
    for tool_format in (None, "unknown", "none", "no-such-row"):
        assert split(THINK_QWEN, True, tool_format=tool_format) == ("weighing it", "The answer.")


THINK_QWEN = "weighing it</think>\n\nThe answer."


def test_opens_lists_every_rows_opening_tag_once():
    from drinkme.serving import think, tool_formats as tf
    assert set(think.OPENS) == {r.think_open for r in tf.ROWS}
    assert len(think.OPENS) == len(set(think.OPENS))
    assert "<think>" in think.OPENS and "<|channel>thought" in think.OPENS
