"""IncrementalDetok: grapheme-cluster hold-back + guaranteed end flush.

🕵️‍♀️ is five codepoints across several tokens; emitting each as its own
SSE delta makes a TUI client (pi) stack four stale copies of
the line, and a flush that drains the scanners but not the detok LOSES
held text entirely at EOS.
"""

from drinkme.serving.detok import IncrementalDetok

DETECTIVE = "\U0001F575️‍♀️"  # 🕵️‍♀️


def run(fulls):
    d = IncrementalDetok()
    deltas = [d.push(f) for f in fulls]
    return d, deltas


def test_zwj_cluster_never_splits_across_deltas():
    fulls = ["Hi! "]
    for i in range(1, len(DETECTIVE) + 1):
        fulls.append("Hi! " + DETECTIVE[:i])
    fulls.append("Hi! " + DETECTIVE + " ok")
    d, deltas = run(fulls)
    # the assembling cluster produced NO partial emissions...
    assert deltas == ["Hi! ", "", "", "", "", "", DETECTIVE + " ok"]
    # ...and nothing is owed at the end
    assert d.flush(fulls[-1]) == ""
    assert "".join(deltas) == fulls[-1]


def test_trailing_cluster_flushes_at_stream_end():
    """The loss case: the stream ends while the tail is held."""
    d, deltas = run(["x ", "x \U0001F575"])
    assert deltas == ["x ", ""]  # emoji held — could still combine
    assert d.flush("x \U0001F575") == "\U0001F575"  # EOS: owed text delivered


def test_partial_utf8_held_then_delivered():
    d, deltas = run(["A�", "AB"])
    assert deltas == ["", "AB"]


def test_dangling_fffd_is_emitted_at_flush_not_dropped():
    """Model genuinely stopped mid-sequence: honesty over tidiness."""
    d, deltas = run(["ok �"])
    assert deltas == [""]
    assert d.flush("ok �") == "ok �"


def test_hold_cap_bounds_emoji_spam_latency():
    spam = "\U0001F525" * 40  # 🔥×40, all could-extend
    d = IncrementalDetok()
    out = d.push("go " + spam)
    # at most HOLD_CAP codepoints stay held; the rest streams
    assert out.startswith("go ")
    held = len("go " + spam) - len(d.emitted)  # BEFORE flush mutates emitted
    assert 0 < held <= IncrementalDetok.HOLD_CAP
    assert d.flush("go " + spam) == spam[-held:]


def test_reformed_tail_emits_suffix_after_common_prefix():
    """Cannot unsend; must not silently drop either."""
    d = IncrementalDetok()
    assert d.push("abc") == "abc"
    assert d.push("abX") == ""     # not an extension — held
    assert d.flush("abXY") == "XY"  # suffix past the common prefix


def test_plain_text_streams_unheld():
    d, deltas = run(["He", "Hel", "Hello", "Hello world"])
    assert deltas == ["He", "l", "lo", " world"]
    assert d.flush("Hello world") == ""
