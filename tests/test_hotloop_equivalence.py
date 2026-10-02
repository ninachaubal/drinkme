"""The equivalence bar for the hot-loop audit's per-token savings.

Every test here answers one question: does the new path produce EXACTLY what
the old path produced? Not "close", not "also valid" — the same token ids, the
same delta strings, bitwise-equal logits. The old path is not described from
memory: tests/hotloop_oracle.py holds verbatim copies of the unoptimized
code, and constrain.JsonConstraint.status_reference is the recursive-descent
parser beside the incremental machine, kept in the module for exactly this.

CPU only, no model downloads, MTP off everywhere (Tier 2 owns mtp.py).
"""

import json
import random

import pytest
import torch

from drinkme.serving.constrain import (COMPLETE, FAIL, PARTIAL, JsonConstraint,
                                       pick_token)
from drinkme.serving.engine import GenerationRequest, SampleParams, complete
from hotloop_oracle import old_pick_token, old_sample_next

# ------------------------------------------------------------- the schemas --
# The battery: deep nesting, long strings, arrays, enum/const.

DEEP = {"type": "object", "additionalProperties": False,
        "required": ["a"],
        "properties": {"a": {"type": "object", "additionalProperties": False,
                             "required": ["b"],
                             "properties": {"b": {"type": "array", "items": {
                                 "type": "object", "additionalProperties": False,
                                 "required": ["c"],
                                 "properties": {"c": {"type": "array",
                                                      "items": {"type": "number"},
                                                      "minItems": 1}}}}}}}}
LONGSTR = {"type": "object", "properties": {"text": {"type": "string"}},
           "required": ["text"], "additionalProperties": False}
ARRAYS = {"type": "array", "items": {"type": "string"}, "minItems": 3}
ENUMS = {"type": "object", "additionalProperties": False,
         "required": ["pick"],
         "properties": {"pick": {"enum": ["alpha", "beta", 12, 1, 123, True, None]},
                        "tag": {"const": "yes"}}}
FREE = {"type": "object"}
NUMS = {"type": "number"}
BATTERY = {"deep": DEEP, "longstr": LONGSTR, "arrays": ARRAYS, "enums": ENUMS,
           "free": FREE, "nums": NUMS}


# --------------------------------------- the incremental parser machine --


def _doc(rng, sch, depth=0):
    """A random document for a schema (valid more often than not)."""
    if "const" in sch:
        return json.dumps(sch["const"])
    if "enum" in sch:
        return json.dumps(rng.choice(sch["enum"]))
    t = sch.get("type")
    if isinstance(t, list):
        t = rng.choice(t)
    if t is None:
        t = rng.choice(["object", "array", "string", "number", "boolean", "null"])
    if t == "object" and depth < 4:
        props, addl = sch.get("properties", {}), sch.get("additionalProperties", True)
        keys = list(props)
        if addl is not False and rng.random() < 0.5:
            keys.append("extra%d" % rng.randrange(3))
        rng.shuffle(keys)
        req = set(sch.get("required", []))
        keys = [k for k in keys if k in req or rng.random() < 0.7]
        body = []
        for k in keys:
            vs = props.get(k)
            if vs is None:
                vs = addl if isinstance(addl, dict) else {}
            body.append("%s:%s" % (json.dumps(k), _doc(rng, vs, depth + 1)))
        return "{" + (", " if rng.random() < 0.5 else ",").join(body) + "}"
    if t == "array" and depth < 4:
        lo, hi = sch.get("minItems", 0), sch.get("maxItems")
        n = rng.randint(lo, hi if hi is not None else lo + 2)
        return "[" + ", ".join(_doc(rng, sch.get("items", {}), depth + 1)
                               for _ in range(n)) + "]"
    if t == "string":
        return json.dumps(rng.choice(["x", "hello there", "a\nb", "A" * 40,
                                      "café", "中文", "a/b", ""]))
    if t == "integer":
        return str(rng.choice([0, 1, 12, -7, 1000]))
    if t == "number":
        return rng.choice(["0", "-1.5e+3", "1e5", "3.25", "-0.5", "12"])
    if t == "boolean":
        return rng.choice(["true", "false"])
    return "null"


ALPHABET = list('{}[]",:0123456789.eE+-truefalsn \t\n\\/ux�AZé中')


def _mutate(rng, doc):
    d = list(doc)
    for _ in range(rng.randint(1, 3)):
        if not d:
            break
        k = rng.randrange(len(d))
        r = rng.random()
        if r < 0.4:
            d[k] = rng.choice(ALPHABET)
        elif r < 0.7:
            d.insert(k, rng.choice(ALPHABET))
        else:
            del d[k]
    return "".join(d)


def test_machine_agrees_with_the_reference_parser_on_every_prefix():
    """The incremental machine IS the recursive descent, transcribed. Every
    prefix of a generated document, of a mutated one, and of pure garbage,
    across the whole battery — one disagreement is a bug in the machine."""
    rng = random.Random(20260821)
    checked = 0
    for _ in range(40):
        for name, schema in BATTERY.items():
            c = JsonConstraint(schema)
            texts = [_doc(rng, schema)]
            texts += [_mutate(rng, texts[0]) for _ in range(4)]
            texts.append("".join(rng.choice(ALPHABET) for _ in range(rng.randint(0, 14))))
            for text in texts:
                for i in range(len(text) + 1):
                    assert c.status(text[:i]) == c.status_reference(text[:i]), \
                        f"{name}: {text[:i]!r}"
                    checked += 1
    assert checked > 20000  # the battery is not accidentally empty


def test_cursor_is_status_fed_in_pieces():
    """The cursor's verdict after N appends == status() of the whole text.
    Chunk boundaries must not be able to change a verdict — that is what makes
    it safe to carry parse state across tokens."""
    rng = random.Random(7)
    for schema in BATTERY.values():
        c = JsonConstraint(schema)
        for _ in range(6):
            doc = _mutate(rng, _doc(rng, schema))
            cur = c.cursor()
            i = 0
            while i < len(doc):
                j = min(len(doc), i + rng.randint(1, 5))
                assert cur.probe(doc[:j]) == c.status_reference(doc[:j])
                cur.accept()
                i = j
            assert cur.text == doc


def test_cursor_probe_does_not_commit():
    """A rejected candidate must leave the cursor exactly where it was —
    otherwise a banned token would poison the next one."""
    c = JsonConstraint(LONGSTR)
    cur = c.cursor()
    assert cur.probe('{"text": "ab') == PARTIAL
    cur.accept()
    assert cur.probe('{"text": "ab�') == FAIL   # banned: partial UTF-8
    assert cur.probe('{"text": "abc') == PARTIAL     # the ban left no trace
    cur.accept()
    assert cur.text == '{"text": "abc'
    assert cur.status() == c.status_reference('{"text": "abc')


def test_cursor_restarts_when_the_text_does_not_extend():
    """A tokenizer may RE-FORM its tail (a multi-byte char completing across a
    token boundary), so a candidate is not always an extension of the accepted
    text. The cursor must fall back to a full parse, not to a wrong answer."""
    c = JsonConstraint({"type": "string"})
    cur = c.cursor('"ab�')
    assert cur.status() == FAIL
    assert cur.probe('"abé') == PARTIAL          # not an extension: reparsed
    cur.accept()
    assert cur.probe('"abé"') == COMPLETE


def test_whitespace_run_rule_is_unchanged_including_inside_strings():
    """MAX_WS_RUN is a rule over the RAW text — the reference scanned the
    whole string before parsing it, so a 4-space run inside a string value
    failed the document. Incremental tracking must keep that quirk."""
    c = JsonConstraint(FREE)
    for text in ('{"a": "   x"}', '{"a": "    x"}', '{  \n\t "a": 1}', "    ",
                 '{"a": 1}   ', '{"a": 1}    '):
        assert c.status(text) == c.status_reference(text), text
    assert c.status('{"a": "    x"}') == FAIL   # the quirk, pinned


# ------------------------------------------------ pick_token, old vs new --

# A vocabulary that can spell JSON, plus the two hostile cases the rejection
# loop exists for: a token that decodes to nothing, and a thinking preamble.
VOCAB = ['{', '}', '[', ']', '"', ':', ',', ' ', '\n', 'a', 'b', 'c',
         'name', 'count', 'tags', 'text', 'pick', 'tag',
         'alpha', 'beta', 'yes', 'hello there', 'A', 'AAAA',
         '12', '1', '123', '0', '3', '-', '.', 'e', '+',
         'true', 'false', 'null', 'é', '中', '\\n', '\\u0041',
         '', '<think>', 'x"']
EOS = frozenset([len(VOCAB)])


def decode(ids):
    return "".join(VOCAB[i] for i in ids if i < len(VOCAB))


def logits_for(step, seed, n=len(VOCAB) + 1):
    """Deterministic pseudo-logits: both implementations see the same row."""
    g = torch.Generator().manual_seed(seed * 1000003 + step)
    return torch.randn(n, generator=g)


def _or_dead(call):
    """The dead-end RuntimeError is an OUTCOME to compare, not a test error:
    a hostile vocabulary really can walk a schema into a corner, and old and
    new have to arrive there on the same step."""
    try:
        return call()
    except RuntimeError as e:
        assert "dead-end" in str(e)
        return "dead-end"


def run_both(schema, params, seed, steps=64):
    """Drive the OLD and NEW pick_token in lockstep over the same rows and the
    same history; return the transcript (and blow up at the first divergence).
    The new call gets everything the engine gives it: the text it already
    decoded, a candidate decoder, and a live cursor."""
    c_new, c_old = JsonConstraint(schema), JsonConstraint(schema)
    cursor = c_new.cursor()
    g_new = torch.Generator().manual_seed(seed) if params.seed is not None else None
    g_old = torch.Generator().manual_seed(seed) if params.seed is not None else None
    gen: list[int] = []
    full = ""
    for step in range(steps):
        row = logits_for(step, seed)
        new = _or_dead(lambda: pick_token(
            row, params, g_new, gen, gen, EOS, decode, c_new, cur=full,
            decode_cand=lambda t: decode(gen + [t]), cursor=cursor))
        old = _or_dead(lambda: old_pick_token(
            row, params, g_old, gen, gen, EOS, decode, c_old))
        assert new == old, f"step {step}: new picked {new}, old picked {old}"
        if new == "dead-end":  # a real one: the vocabulary cannot continue
            return gen, "dead-end"
        if new in EOS:
            return gen, "eos"
        gen.append(new)
        full = cursor.text
        assert full == decode(gen), "the cursor's text drifted from the decode"
    return gen, "length"


@pytest.mark.parametrize("name", sorted(BATTERY))
def test_pick_token_greedy_is_token_identical(name):
    gen, why = run_both(BATTERY[name], SampleParams(temperature=0.0), seed=11)
    assert decode(gen) or why == "eos"


@pytest.mark.parametrize("name", sorted(BATTERY))
def test_pick_token_seeded_sampling_is_token_identical(name):
    p = SampleParams(temperature=1.1, top_p=0.95, top_k=20, seed=5)
    run_both(BATTERY[name], p, seed=5)


def test_pick_token_with_penalties_is_token_identical():
    """Penalties change which candidate wins, and the constrained path threads
    the same id history through them — so they belong in this comparison."""
    p = SampleParams(temperature=0.8, seed=3, repetition_penalty=1.3,
                     presence_penalty=0.7, frequency_penalty=0.4)
    for name in ("longstr", "arrays", "free"):
        run_both(BATTERY[name], p, seed=3, steps=96)


def test_pick_token_long_generation_is_token_identical():
    """Long enough that the O(n^2) re-parse would have mattered: the whole
    point is that the answers did not move when the cost did."""
    p = SampleParams(temperature=0.9, seed=9)
    gen, _ = run_both({"type": "array", "items": {"type": "string"},
                       "minItems": 200}, p, seed=9, steps=600)
    assert len(gen) > 300


def test_pick_token_bans_the_same_candidates_in_the_same_order():
    """Not just the surviving token: the REJECTION path must match too, or the
    RNG stream diverges the moment a real model rejects anything."""
    schema = {"type": "object", "additionalProperties": False,
              "properties": {"name": {"type": "string"}}, "required": ["name"]}
    c_new, c_old = JsonConstraint(schema), JsonConstraint(schema)
    seen_new, seen_old = [], []
    row = torch.zeros(len(VOCAB) + 1)
    for i, tokid in enumerate([41, 39, 0, 4, 12]):  # <think>, '', '{', '"', name
        row[tokid] = 10.0 - i  # rank the hostile ones first
    p = SampleParams(temperature=0.0)

    def spy(fn, seen):
        def wrapped(text):
            seen.append(text)
            return fn(text)
        return wrapped

    c_new.status = spy(c_new.status, seen_new)
    c_old.status_reference = spy(c_old.status_reference, seen_old)
    new = pick_token(row, p, None, [], [], EOS, decode, c_new)  # no cursor: same road
    old = old_pick_token(row, p, None, [], [], EOS, decode, c_old)
    assert new == old == 0  # '{' — the first candidate that is a JSON prefix
    assert seen_new == seen_old and seen_new  # same texts, same order


def test_dead_end_raises_the_same_error_on_every_path():
    """The full-vocab isfinite() check waits for a ban threshold. Every
    way of reaching the all-banned state must land on the same error."""
    c = JsonConstraint({"type": "object"})
    tiny = torch.zeros(3)  # '{', '}', '[' — 'hello' cannot be fixed by any
    for params in (SampleParams(temperature=0.0), SampleParams(temperature=1.0)):
        g = torch.Generator().manual_seed(1)
        with pytest.raises(RuntimeError, match="dead-end"):
            pick_token(tiny, params, g, [21], [21], frozenset([99]), decode, c)
        with pytest.raises(RuntimeError, match="dead-end"):
            old_pick_token(tiny, params, g, [21], [21], frozenset([99]), decode, c)


def test_dead_end_raises_on_a_row_with_nothing_finite_left():
    """A caller that hands in a dead row gets the dead-end error, not a torch
    traceback and not a hang — old behaviour, kept, and it is why the finite
    support is COUNTED rather than threshold-guessed (a row with fewer live
    logits than the threshold would have slipped past one)."""
    c = JsonConstraint({"type": "object"})
    for row in (torch.full((8,), float("-inf")),
                torch.full((8,), float("nan"))):
        for params in (SampleParams(temperature=0.0), SampleParams(temperature=1.0)):
            with pytest.raises(RuntimeError, match="dead-end"):
                pick_token(row, params, torch.Generator().manual_seed(2), [], [],
                           EOS, decode, c)
            with pytest.raises(RuntimeError, match="dead-end"):
                old_pick_token(row, params, torch.Generator().manual_seed(2),
                               [], [], EOS, decode, c)


def test_full_vocab_finiteness_is_read_once_per_step_not_per_candidate():
    """The receipt for the change itself. The old loop asked isfinite() over
    the whole vocabulary before EVERY candidate; the new one asks once and
    counts, however many candidates the step bans."""
    real_isfinite, calls = torch.isfinite, []

    def counting(t):
        calls.append(t.numel())
        return real_isfinite(t)

    c = JsonConstraint({"type": "object"})
    row = torch.full((len(VOCAB) + 1,), -20.0)
    for rank, tokid in enumerate([41, 39, 21, 34, 0]):  # 4 bans, then '{'
        row[tokid] = 10.0 - rank
    torch.isfinite = counting
    try:
        assert pick_token(row, SampleParams(temperature=0.0), None, [], [], EOS,
                          decode, c) == 0
        new_calls = list(calls)
        calls.clear()
        assert old_pick_token(row, SampleParams(temperature=0.0), None, [], [],
                              EOS, decode, c) == 0
        old_calls = list(calls)
    finally:
        torch.isfinite = real_isfinite
    assert len(new_calls) == 1                     # once per step
    assert len(old_calls) == 5                     # once per candidate
    assert new_calls[0] == len(VOCAB) + 1


# ------------------------------------------- penalty state, scratch --

PENALTY_CASES = [
    SampleParams(temperature=0.0, repetition_penalty=1.15),
    SampleParams(temperature=0.0, presence_penalty=0.8),
    SampleParams(temperature=0.0, frequency_penalty=0.35),
    SampleParams(temperature=0.7, seed=4, repetition_penalty=1.3,
                 presence_penalty=0.6, frequency_penalty=0.45),
    SampleParams(temperature=1.0, seed=8, repetition_penalty=0.85,  # < 1: a
                 presence_penalty=-0.5, frequency_penalty=-0.25),   # reward
    SampleParams(temperature=0.0),  # all off: the path must stay untouched
]


def _history(rng, vocab, n):
    """An id history with the shape a real one has: a few tokens carrying most
    of the mass (so counts get large) over a long tail."""
    hot = [rng.randrange(vocab) for _ in range(8)]
    return [rng.choice(hot) if rng.random() < 0.6 else rng.randrange(vocab)
            for _ in range(n)]


@pytest.mark.parametrize("params", PENALTY_CASES, ids=range(len(PENALTY_CASES)))
@pytest.mark.parametrize("n_gen", [1, 512, 1500])
def test_penalty_state_is_bitwise_the_list_path(params, n_gen):
    """The hot-loop audit's bar: torch.equal on the PROCESSED LOGITS, over
    generations long enough that the O(n) rebuild would have mattered.
    Bitwise, not allclose — the op order is part of the contract."""
    from drinkme.serving.sampling import PenaltyState
    from hotloop_oracle import old_penalties

    rng = random.Random(1234 + n_gen)
    vocab = 331
    torch.manual_seed(99)
    logits = torch.randn(vocab) * 4.0  # both signs, well outside [-1, 1]
    logits[7], logits[9] = 0.0, -0.0   # the sign boundary, both spellings
    prompt = _history(rng, vocab, 97)
    gen = _history(rng, vocab, n_gen)

    state = PenaltyState(vocab)
    state.observe(prompt)
    for t in gen:
        state.accept(t)

    new = logits.to(torch.float32).clone()
    state.apply(new, params)
    old = old_penalties(logits, params, prev_ids=prompt + gen, gen_ids=gen)
    assert torch.equal(new, old)


@pytest.mark.parametrize("params", PENALTY_CASES, ids=range(len(PENALTY_CASES)))
def test_sample_next_with_state_picks_the_same_tokens(params):
    """The whole call, not just the penalty block: same rows, same seeded
    generator, same tokens — including the sampling paths where a one-bit
    difference in a logit would move the draw."""
    from drinkme.serving.sampling import PenaltyState, SampleScratch, sample_next

    vocab = 257
    state, scratch = PenaltyState(vocab), SampleScratch()
    prompt = list(range(11))
    state.observe(prompt)
    all_ids, gen_ids = list(prompt), []
    g_new = torch.Generator().manual_seed(21)
    g_old = torch.Generator().manual_seed(21)
    for step in range(600):
        row = torch.randn(vocab, generator=torch.Generator().manual_seed(step))
        new = sample_next(row, params, g_new, state=state, scratch=scratch)
        old = old_sample_next(row, params, g_old, prev_ids=all_ids,
                              gen_ids=gen_ids)
        assert new == old, f"step {step}: {new} != {old}"
        all_ids.append(new)
        gen_ids.append(new)
        state.accept(new)


def test_scratch_lends_the_same_buffers_and_changes_no_answer():
    """One allocation for the life of an engine, not two per call —
    and the row it hands back is the row .to(float32).clone() handed back."""
    from drinkme.serving.sampling import SampleScratch, sample_next

    scratch = SampleScratch()
    params = SampleParams(temperature=0.9, seed=2, top_k=32, top_p=0.9)
    ptrs = set()
    for step in range(50):
        row = torch.randn(1024, generator=torch.Generator().manual_seed(step))
        a = sample_next(row, params, torch.Generator().manual_seed(3),
                        scratch=scratch)
        ptrs.add(scratch.work.data_ptr())
        b = old_sample_next(row, params, torch.Generator().manual_seed(3))
        assert a == b
    assert len(ptrs) == 1  # one buffer, reused for every step


def test_scratch_survives_a_bfloat16_row():
    """Real logits arrive bf16 off the model; copy_ must do exactly what
    .to(torch.float32) did to them."""
    from drinkme.serving.sampling import SampleScratch, sample_next

    row = (torch.randn(512, generator=torch.Generator().manual_seed(6))
           .to(torch.bfloat16))
    p = SampleParams(temperature=0.0, repetition_penalty=1.2)
    scratch = SampleScratch()
    assert (sample_next(row, p, prev_ids=[3, 9], scratch=scratch)
            == old_sample_next(row, p, prev_ids=[3, 9]))
    # the lend itself (the call above left its penalties in the buffer)
    assert torch.equal(scratch.take_work(row), row.to(torch.float32))


# ------------------------------------------------ the suffix-window decode --


def _stream(rng, tok, n):
    """A token stream with the shapes that break naive windows: whole
    multi-byte pieces, emoji with ZWJ, and single BYTES that only become a
    character when the next token lands."""
    from hotloop_toys import BYTE, piece_id

    fancy = [piece_id(tok, p) for p in ("é", "中", "中文", "🙂", "🕵️‍♀️", "👍🏽", '"', "{", " ")]
    raw = [tok.convert_tokens_to_ids(BYTE[b])
           for ch in "中文é🙂" for b in ch.encode("utf-8")]
    ids = []
    for _ in range(n):
        r = rng.random()
        if r < 0.25:
            ids.append(rng.choice(fancy))
        elif r < 0.45:
            ids.append(rng.choice(raw))       # a lone byte: FFFD until finished
        else:
            ids.append(rng.randrange(len(tok) - 1))
    return ids


@pytest.mark.parametrize("window", [4, 16, 64])
def test_window_decode_is_byte_identical_on_a_byte_level_bpe(window):
    """The bar: the string the loop diffs must be the tokenizer's own
    string at EVERY step, including mid-character and mid-grapheme."""
    from drinkme.serving.detok import SuffixWindow
    from hotloop_toys import json_tokenizer

    tok = json_tokenizer()
    dec = lambda ids: tok.decode(ids, skip_special_tokens=True)  # noqa: E731
    ids = _stream(random.Random(4), tok, 400)
    win = SuffixWindow(dec, window=window)
    for i in range(len(ids) + 1):
        assert win.full(ids[:i]) == dec(ids[:i]), f"at {i} tokens"
        if i < len(ids):
            assert win.peek(ids[:i], ids[i]) == dec(ids[:i + 1])
    assert win.live and win.base > 0  # it really did slide


def test_window_decode_is_byte_identical_on_a_space_joining_tokenizer():
    """A decoder that is NOT concatenative (WordLevel joins with spaces, the
    toy in test_serving_engines.py) must be just as exact — the anchor is
    derived from a real decode, not assumed."""
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast

    from drinkme.serving.detok import SuffixWindow

    vocab = {w: i for i, w in enumerate(
        ["<unk>", "alpha", "beta", "gamma", "delta", "epsilon", "zeta"])}
    backend = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
    backend.pre_tokenizer = Whitespace()
    tok = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="<unk>")
    dec = lambda ids: tok.decode(ids, skip_special_tokens=True)  # noqa: E731
    rng = random.Random(5)
    ids = [rng.randrange(1, len(vocab)) for _ in range(200)]
    win = SuffixWindow(dec, window=8)
    for i in range(len(ids) + 1):
        assert win.full(ids[:i]) == dec(ids[:i]), f"at {i} tokens"
    assert win.live and win.base > 0


def test_window_defers_the_anchor_across_a_split_character():
    """A multi-byte character straddling the boundary is not a failure, it is
    a bad place to stand: the window keeps decoding fully and tries again."""
    from drinkme.serving.detok import SuffixWindow
    from hotloop_toys import BYTE, json_tokenizer

    tok = json_tokenizer()
    dec = lambda ids: tok.decode(ids, skip_special_tokens=True)  # noqa: E731
    three = [tok.convert_tokens_to_ids(BYTE[b]) for b in "中".encode("utf-8")]
    ids = []
    for _ in range(40):
        ids += three  # every third token starts a character
    win = SuffixWindow(dec, window=3)
    for i in range(len(ids) + 1):
        assert win.full(ids[:i]) == dec(ids[:i]), f"at {i} tokens"
    assert win.live  # never wrong, never given up on


def test_window_never_anchors_where_it_cannot_reconcile(capsys):
    """A decoder whose window decode is not a suffix of its full decode never
    gets an anchor at all: full decodes forever, exact forever, and nothing
    printed — being unable to optimise is not an error."""
    from drinkme.serving.detok import SuffixWindow

    def unreconcilable(ids):  # a trailing length stamp: no suffix ever matches
        return "".join("abc"[i % 3] for i in ids) + f"[{len(ids)}]"

    win = SuffixWindow(unreconcilable, window=4)
    ids = list(range(60))
    outs = [win.full(ids[:i]) for i in range(len(ids) + 1)]
    assert outs == [unreconcilable(ids[:i]) for i in range(len(ids) + 1)]
    assert win.base == 0 and win.live
    assert capsys.readouterr().err == ""


def test_window_gives_up_out_loud_when_a_late_rewrite_breaks_the_join(capsys):
    """The residual risk, pinned. Between slides the window trusts one thing:
    that text already committed stays committed. A decoder that rewrites its
    own past (a global cleanup pass that only kicks in at some length) breaks
    it — the next slide CATCHES that, drops the window and says so on stderr.
    What it cannot do is un-emit the tokens in between, which is why
    DRINKME_DETOK_VERIFY=1 exists and why the engine equivalence tests run
    whole generations under it."""
    from drinkme.serving.detok import SuffixWindow

    def rewrites_its_past(ids):
        s = "".join("abc"[i % 3] for i in ids)
        return s.replace("ab", "AB") if len(ids) >= 40 else s

    win = SuffixWindow(rewrites_its_past, window=4)
    ids = list(range(80))
    outs = [win.full(ids[:i]) for i in range(len(ids) + 1)]
    truth = [rewrites_its_past(ids[:i]) for i in range(len(ids) + 1)]
    assert not win.live
    assert "suffix-window decode disabled" in capsys.readouterr().err
    assert outs[-1] == truth[-1] and outs[:40] == truth[:40]
    assert any(o != t for o, t in zip(outs, truth))  # the window was wrong,
    #                                                  and the check is what saw it


def test_window_verify_mode_raises_rather_than_lies():
    from drinkme.serving.detok import SuffixWindow

    calls = {"n": 0}

    def drifting(ids):
        calls["n"] += 1
        return "x" * len(ids) + ("!" if calls["n"] > 30 else "")

    win = SuffixWindow(drifting, window=2, verify=True)
    with pytest.raises(AssertionError, match="suffix-window decode diverged"):
        for i in range(60):
            win.full(list(range(i)))


# ------------------------------------------- the whole loop, old path vs new --
#
# DRINKME_HOTLOOP_OFF=1 restores the pre-audit per-token path inside
# HFEngine.generate: full re-detok every token, penalties rebuilt from the id
# lists, a fresh [V] clone per candidate, a constraint that re-parses from
# byte 0. Every test below runs the SAME generation both ways on the same
# model and compares what came out on the wire — deltas, text, finish reason,
# usage — with the window's own verification turned on for good measure.

MSGS = [{"role": "user", "content": 'write {"name": "x"} 🕵️‍♀️ 中文'}]


@pytest.fixture(scope="module")
def toys():
    """One tiny model, two engines over it: the new path and the old one."""
    import os

    from hotloop_toys import toy_engine

    os.environ["DRINKME_PREFIX_SLOTS"] = "0"  # cold every turn, both sides
    os.environ["DRINKME_DETOK_VERIFY"] = "1"
    try:
        new = toy_engine(seed=3)
        os.environ["DRINKME_HOTLOOP_OFF"] = "1"
        old = toy_engine(seed=3, tokenizer=new.tok)
        old.model = new.model  # the same weights, so only the paths differ
    finally:
        for k in ("DRINKME_PREFIX_SLOTS", "DRINKME_DETOK_VERIFY",
                  "DRINKME_HOTLOOP_OFF"):
            os.environ.pop(k, None)
    assert new._hot_off is False and old._hot_off is True
    assert new._detok_verify is True
    return new, old


@pytest.fixture
def endless(toys):
    """Both engines with EOS disabled, so a generation runs to max_tokens —
    the only way to get the >= 512 tokens the hot-loop audit's bar asks for out of a
    randomly-initialised toy that likes its own end token."""
    saved = [e.eos_ids for e in toys]
    for e in toys:
        e.eos_ids = frozenset()
    yield toys
    for e, s in zip(toys, saved):
        e.eos_ids = s


def both(toys, params, msgs=None, tools=None):
    """Run one generation down each path; return (deltas, result). A dead-end
    RuntimeError is an outcome like any other and must match too."""
    out = []
    for eng in toys:
        deltas = []
        try:
            res = complete(eng, GenerationRequest(msgs or MSGS, params, tools=tools),
                           lambda d: deltas.append(d) or True)
        except RuntimeError as e:
            assert "dead-end" in str(e)
            res = e
        out.append((deltas, res))
    (dn, rn), (do, ro) = out
    if isinstance(rn, RuntimeError) or isinstance(ro, RuntimeError):
        assert type(rn) is type(ro) and str(rn) == str(ro)
        assert dn == do
        return dn, None
    assert dn == do, "delta streams differ"
    assert rn.text == ro.text and rn.finish_reason == ro.finish_reason
    assert rn.completion_tokens == ro.completion_tokens
    assert rn.prompt_tokens == ro.prompt_tokens
    assert rn.tool_calls == ro.tool_calls
    return dn, rn


def test_engine_greedy_stream_is_identical(toys):
    deltas, res = both(toys, SampleParams(temperature=0.0, max_tokens=120))
    assert res.completion_tokens > 0 and "".join(deltas) == res.text


def test_engine_seeded_sampling_stream_is_identical(toys):
    both(toys, SampleParams(temperature=1.1, top_p=0.92, top_k=40, seed=17,
                            max_tokens=120))


@pytest.mark.parametrize("params", [
    SampleParams(temperature=0.0, max_tokens=600, repetition_penalty=1.2),
    SampleParams(temperature=0.0, max_tokens=600, presence_penalty=0.9,
                 frequency_penalty=0.5),
    SampleParams(temperature=1.0, seed=2, max_tokens=600,
                 repetition_penalty=1.15, presence_penalty=0.4,
                 frequency_penalty=0.3),
], ids=["repetition", "presence+frequency", "all-three-sampled"])
def test_engine_penalties_over_a_long_generation_are_identical(endless, params):
    """>= 512 tokens, which is where rebuilding the penalty windows from the
    id lists every step stopped being free (the hot-loop audit's bar)."""
    deltas, res = both(endless, params)
    assert res.completion_tokens >= 512


def test_engine_stop_string_split_across_deltas_is_identical(toys):
    """The stop scanner sits downstream of the detok, so a windowed decode
    that moved a character across a delta boundary would show up here."""
    deltas, res = both(toys, SampleParams(temperature=0.0, max_tokens=300,
                                          stop=["a", "\\u", "中"]))
    assert res.finish_reason in ("stop", "length")


def test_engine_multibyte_and_zwj_output_is_identical(toys):
    """The toy's vocabulary spells emoji, ZWJ sequences and CJK both as whole
    tokens and one byte at a time, so a long greedy run walks the grapheme
    hold-back and the U+FFFD hold with real material."""
    deltas, res = both(toys, SampleParams(temperature=1.3, seed=31,
                                          max_tokens=400))
    text = "".join(deltas)
    assert any(ord(ch) > 0x7F for ch in text), "no multi-byte output to compare"


@pytest.mark.parametrize("name", ["free", "longstr", "arrays", "enums", "deep"])
def test_engine_constrained_json_is_identical(toys, name):
    """Constrained generation end to end: the cursor, the reused `full`, the
    windowed candidate decode and the scratch buffers all at once, against a
    loop that had none of them."""
    params = SampleParams(temperature=0.0, max_tokens=160,
                          output_schema=BATTERY[name])
    deltas, res = both(toys, params)
    assert res is None or res.completion_tokens > 0  # None = both dead-ended


def test_engine_constrained_json_sampled_is_identical(toys):
    both(toys, SampleParams(temperature=1.2, seed=6, max_tokens=160,
                            output_schema=BATTERY["free"]))


def test_engine_abort_midstream_is_identical(toys):
    """The abort path returns what was DELIVERED; both paths must agree on
    where that stops."""
    outs = []
    for eng in toys:
        deltas = []
        res = complete(eng, GenerationRequest(MSGS, SampleParams(temperature=0.0, max_tokens=200)),
                       lambda d: (deltas.append(d), len(deltas) < 5)[1])
        outs.append((deltas, res.text, res.finish_reason, res.completion_tokens))
    assert outs[0] == outs[1] and outs[0][2] == "abort"


# --------------------------------------------------- the confetti items --


def test_to_device_lets_go_of_the_numpy_arrays():
    """CompressedLinear.p outlives the load, so anything numpy left in
    it is host memory held for the life of the server: the radix runtime
    dict keeps exactly one numpy array — the unpadded palette, at most 21
    bytes (radix_pack._palette_u8's reason) — and the CPU reference decode
    must still be bit-exact off what remains."""
    import numpy as np

    from drinkme.codec import radix_pack as rp
    from drinkme.codec.swap import make_module, to_device_radix

    rng = np.random.default_rng(11)
    sign = rng.integers(0, 2, (32, 64), dtype=np.uint16)
    exp = rng.integers(55, 66, (32, 64), dtype=np.uint16)  # a spread: every tier in use
    mant = rng.integers(0, 128, (32, 64), dtype=np.uint16)
    U = (sign << 15) | (exp << 8) | mant
    raw = rp.pack_array_radix(U, "gulp", encoder="numpy")
    assert raw is not None and len(raw["rx_palette"]) >= 8

    dev = to_device_radix(raw, "cpu")
    assert not set(rp._ARRAYS_RADIX) & {k for k, v in dev.items() if isinstance(v, np.ndarray)}
    numpy_left = {k: v for k, v in dev.items() if isinstance(v, np.ndarray)}
    assert set(numpy_left) == {"rx_palette_np"} and numpy_left["rx_palette_np"].nbytes <= 21
    # and the reference decode is still the pack's own bytes, bit for bit
    got = make_module(raw, None, "cpu")._cpu_weight().view(torch.int16).numpy().view(np.uint16)
    assert np.array_equal(got, rp.decode_back(raw))
    assert np.array_equal(got, U)  # lossless, which is the whole product


def test_schema_is_validated_once_per_request_not_twice(toys):
    """The HTTP boundary walks the schema by name (that is what makes
    an unsupported keyword a 400); the engine was walking it again."""
    from drinkme.serving import constrain

    new, _ = toys
    calls = []
    real = constrain.validate_supported

    def spy(schema):
        calls.append(schema)
        return real(schema)

    constrain.validate_supported = spy
    try:
        p = SampleParams(temperature=0.0, max_tokens=4, output_schema=DEEP,
                         output_schema_validated=True)
        complete(new, GenerationRequest(MSGS, p))
        assert calls == []  # the boundary already said so
        p.output_schema_validated = False
        complete(new, GenerationRequest(MSGS, p))
        walked = len(calls)          # nobody said so: validate, as always
        calls.clear()
        real(DEEP)                   # what ONE walk of this schema costs
        one_walk = len(calls) + 1    # (the spy misses the outermost call)
    finally:
        constrain.validate_supported = real
    assert walked == one_walk > 1    # once, over a schema with nested levels


# (the HTTP half of this item — that the boundary MARKS what it validated —
# is asserted where the boundary is tested: test_serving_http.py's
# test_response_format_parses_and_reaches_engine.)


def test_mtp_depth_is_not_re_read_per_request(toys):
    """DRINKME_MTP_DEPTH was parsed on every request. It is parsed ONCE per
    engine now — and since n-gram speculation landed that is true with or without a head, because a
    headless engine still resolves its speculation plan once
    (serving/ngram.resolve_mode). The invariant
    this test is about is "not once per request"; the count below is per
    ENGINE STATE, and it goes up by one when the head appears because that
    drops the cache (`_mtp_depth = None` is the same seam
    bench/mtp_gpu_acceptance.set_arm uses).
    (DRINKME_SPEC=off pinned here because unset means AUTO — this test's
    head is a bare object() that could never actually speculate.)"""
    import os

    from drinkme.serving import mtp

    new, _ = toys
    calls = []
    real = mtp.depth_from_env

    def spy():
        calls.append(1)
        return real()

    mtp.depth_from_env = spy
    # this engine is shared with the tests above, so its cache is warm: drop
    # it the way an instrument does before counting reads
    new._spec_plan = new._mtp_depth = None
    old_env = os.environ.get("DRINKME_SPEC")
    os.environ["DRINKME_SPEC"] = "off"
    try:
        for _ in range(3):
            complete(new, GenerationRequest(MSGS, SampleParams(temperature=0.0, max_tokens=3)))
        assert len(calls) == 1      # once per engine, not once per request
        new.mtp_head = object()     # a head, but speculation is off
        for _ in range(3):
            complete(new, GenerationRequest(MSGS, SampleParams(temperature=0.0, max_tokens=3)))
        assert len(calls) == 2      # the head dropped the cache: one more read
    finally:
        mtp.depth_from_env = real
        if old_env is None:
            os.environ.pop("DRINKME_SPEC", None)
        else:
            os.environ["DRINKME_SPEC"] = old_env
        new.mtp_head = None
        new._mtp_depth = None


@pytest.mark.parametrize("seed", [7, 42])
def test_engine_streams_are_identical_on_other_models(seed):
    """The equivalence is structural, not an artifact of one random init: two
    more models, built and compared from scratch."""
    import os

    from hotloop_toys import toy_engine

    os.environ["DRINKME_PREFIX_SLOTS"] = "0"
    os.environ["DRINKME_DETOK_VERIFY"] = "1"
    try:
        new = toy_engine(seed=seed)
        os.environ["DRINKME_HOTLOOP_OFF"] = "1"
        old = toy_engine(seed=seed, tokenizer=new.tok)
        old.model = new.model
    finally:
        for k in ("DRINKME_PREFIX_SLOTS", "DRINKME_DETOK_VERIFY",
                  "DRINKME_HOTLOOP_OFF"):
            os.environ.pop(k, None)
    pair = (new, old)
    both(pair, SampleParams(temperature=0.0, max_tokens=200,
                            repetition_penalty=1.2, presence_penalty=0.5))
    both(pair, SampleParams(temperature=1.2, seed=5, max_tokens=150,
                            output_schema=BATTERY["longstr"]))
