"""Chunked prefill on toy models — CPU only.

The contract on the GPU is "no prefill dispatch over 250 ms"; what a CPU
test can pin is the mechanism that delivers it and the numerics it may not
break:

  BOUNDED. tests/test_serving_prefill_bound.py: the engine never hands the
  model more than C tokens (it imports nothing new, so an unchunked prefill
  fails it on the row counts rather than at import). Here: the knob itself.

  The same answer. Chunked == whole on a dense toy (Qwen3) and a hybrid
  DeltaNet toy (qwen3_5, the 27B's layer alternation): greedy transcripts
  identical through every branch and across a prefix-cache extend, and the
  last row's logits allclose at the prefill seam. The toys are float32, so
  "allclose" is reduction-order noise, not bf16 rounding.

  SEGMENTED ATTENTION. The split-key attention (serving/segmented_
  attention.py) is checked against SDPA with a lower-right causal mask, and
  then forced on through the engine (thresholds shrunk to toy size) with the
  same transcript and logits as the whole-prompt path.
"""

import pytest
import torch

from drinkme.serving import mtp, prefill
from drinkme.serving import segmented_attention as sa

from test_serving_mtp import _torch_chunk_on_cpu, toy  # noqa: F401 — fixtures by name
from test_serving_prefill_bound import (  # noqa: F401 — fixtures by name
    BRANCHES, C, LONG, _toy, ask, dense, engine, env, hybrid, user)

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")

@pytest.mark.parametrize("raw,want", [(None, prefill.DEFAULT_CHUNK), ("", prefill.DEFAULT_CHUNK),
                                      ("0", 0), ("512", 512), (" 64 ", 64)])
def test_the_knob(raw, want):
    with env(DRINKME_PREFILL_CHUNK=raw):
        assert prefill.chunk_from_env() == want


@pytest.mark.parametrize("raw", ["-1", "4k", "1.5"])
def test_a_bad_knob_is_refused_by_name(raw):
    with env(DRINKME_PREFILL_CHUNK=raw), pytest.raises(SystemExit, match="DRINKME_PREFILL_CHUNK"):
        prefill.chunk_from_env()


def test_the_load_log_says_what_prefill_does(dense, capsys):
    model, tok = dense
    engine(model, tok, 2048)
    assert "[drinkme] prefill: chunks of 2048 tokens" in capsys.readouterr().out
    engine(model, tok, 0)
    assert "whole prompt in one forward (DRINKME_PREFILL_CHUNK=0)" in capsys.readouterr().out


def test_spans():
    assert prefill.spans(0, 10, 0) == [(0, 10)]
    assert prefill.spans(3, 10, 4) == [(3, 7), (7, 10)]
    assert prefill.spans(0, 8, 8) == [(0, 8)]
    assert prefill.spans(5, 6, 4) == [(5, 6)]


# ------------------------------------------------------ THE SAME ANSWER --

@pytest.mark.parametrize("name,spec", BRANCHES)
def test_chunked_greedy_is_the_whole_prompt_greedy(request, name, spec):
    model, tok, head = _toy(request, name)
    h = head if spec == "auto" else None
    whole = ask(engine(model, tok, 0, head=h), user(LONG), spec, n=24)
    for chunk in (1, 3, C):
        got = ask(engine(model, tok, chunk, head=h), user(LONG), spec, n=24)
        assert got.text == whole.text, (chunk, got.text, whole.text)


@pytest.mark.parametrize("name,spec", BRANCHES)
def test_chunked_greedy_survives_a_prefix_cache_extend(request, name, spec):
    model, tok, head = _toy(request, name)
    h = head if spec == "auto" else None
    out = {}
    for chunk in (0, C):
        eng = engine(model, tok, chunk, head=h)
        hist = user(LONG)
        a = ask(eng, hist, spec)
        hist += [{"role": "assistant", "content": a.text}] + user(LONG)
        b = ask(eng, hist, spec)
        assert b.cached_tokens > 0
        out[chunk] = (a.text, b.text)
    assert out[C] == out[0]


def _ids(tok, text):
    return tok(text, add_special_tokens=False).input_ids


@pytest.mark.parametrize("name", ["dense", "hybrid"])
def test_last_row_logits_allclose_at_the_prefill_seam(request, name):
    model, tok, _ = _toy(request, name)
    eng = engine(model, tok, 0)
    ids = _ids(tok, LONG)
    with torch.inference_mode():
        want = prefill.run(model, ids, 0, eng._cache(len(ids) + 1), "cpu", 0)
        for chunk in (1, 5, C, len(ids)):
            got = prefill.run(model, ids, 0, eng._cache(len(ids) + 1), "cpu", chunk)
            torch.testing.assert_close(got, want, atol=2e-5, rtol=1e-5)
        # a warm extend: 20 tokens cached whole, the rest chunked
        cache = eng._cache(len(ids) + 1)
        prefill.run(model, ids[:20], 0, cache, "cpu", 0)
        got = prefill.run(model, ids, 20, cache, "cpu", C)
    torch.testing.assert_close(got, want, atol=2e-5, rtol=1e-5)


def test_mtp_prefill_hidden_states_arrive_span_by_span(hybrid):
    model, tok, _ = hybrid
    eng = engine(model, tok, 0)
    ids = _ids(tok, LONG)
    whole_h = []
    with torch.inference_mode():
        want = prefill.run(model, ids, 0, eng._cache(len(ids) + 1), "cpu", 0, hidden=True,
                           on_hidden=lambda h, a, b: whole_h.append((h, a, b)))
        spans = []
        got = prefill.run(model, ids, 0, eng._cache(len(ids) + 1), "cpu", C, hidden=True,
                          on_hidden=lambda h, a, b: spans.append((h, a, b)))
    assert [(a, b) for _, a, b in spans] == prefill.spans(0, len(ids), C)
    torch.testing.assert_close(torch.cat([h for h, _, _ in spans], 1), whole_h[0][0],
                               atol=2e-5, rtol=1e-5)
    torch.testing.assert_close(got, want, atol=2e-5, rtol=1e-5)


def test_seeding_the_head_span_by_span_is_begin(hybrid):
    """Speculator.seed over spans writes the head KV begin() writes whole."""
    model, tok, head = hybrid
    eng = engine(model, tok, 0)
    ids = _ids(tok, LONG)
    with torch.inference_mode():
        cache = eng._cache(len(ids) + 1)
        hidden, _ = mtp.forward_with_hidden(model, torch.tensor([ids]), cache,
                                            torch.arange(len(ids)), last_row_only=True)
        spec = mtp.Speculator(head, model, depth=2)
        spec.begin(cache, ids, hidden, 0)
        want = (head.cache.layers[0].keys.clone(), head.entries, spec.h_last.clone())
        spec = mtp.Speculator(head, model, depth=2)
        spec.open(cache, ids)
        for a, b in prefill.spans(0, len(ids), C):
            spec.seed(hidden[:, a:b], a, b)
    assert head.entries == want[1] == len(ids) - 1
    torch.testing.assert_close(head.cache.layers[0].keys, want[0], atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(spec.h_last, want[2])


# -------------------------------------------------- SEGMENTED ATTENTION --

@pytest.mark.parametrize("H,KV,D,M,L,S", [
    (4, 2, 16, 7, 50, 8),    # GQA 2:1, ragged last segment
    (6, 6, 8, 5, 33, 1),     # MHA, one key per segment
    (4, 1, 16, 3, 64, 64),   # MQA, one segment
    (2, 1, 32, 9, 5, 100),   # segment wider than the prefix
])
def test_segmented_attention_is_lower_right_causal_sdpa(H, KV, D, M, L, S):
    from torch.nn.attention.bias import causal_lower_right

    g = torch.Generator().manual_seed(M * 100 + L)
    q = torch.randn(1, H, M, D, generator=g)
    k, v = torch.randn(1, KV, L + M, D, generator=g), torch.randn(1, KV, L + M, D, generator=g)
    got = sa.attend(q, k, v, L, S, 0.3)
    want = torch.nn.functional.scaled_dot_product_attention(
        q, k.repeat_interleave(H // KV, 1), v.repeat_interleave(H // KV, 1),
        attn_mask=causal_lower_right(M, L + M), scale=0.3)
    torch.testing.assert_close(got, want, atol=2e-6, rtol=1e-5)


def test_segmentation_is_decided_by_size():
    assert not sa.needed(1, 10**6)            # decode: never
    assert not sa.needed(4096, 0)             # a cold chunk: today's causal call
    assert sa.SPLIT_ABOVE == 2048 * 4096
    assert not sa.needed(2048, 2048)          # 2048 x 4096: at SPLIT_ABOVE, not over
    assert sa.needed(2048, 2049)              # one key more
    assert sa.needed(4096, 4096)
    assert sa.segment_for(4096) == sa.WORK // 4096
    assert sa.segment_for(10**9) == sa.MIN_SEGMENT


@pytest.fixture
def tiny_segments(monkeypatch):
    """Toy-sized thresholds, and a count of the segmented calls."""
    monkeypatch.setattr(sa, "SPLIT_ABOVE", 16)
    monkeypatch.setattr(sa, "WORK", 24)
    monkeypatch.setattr(sa, "MIN_SEGMENT", 3)
    calls = []
    real = sa.attend

    def spy(q, k, v, past, segment, scale=None):
        calls.append((q.shape[-2], past, segment))
        return real(q, k, v, past, segment, scale)

    monkeypatch.setattr(sa, "attend", spy)
    return calls


@pytest.mark.parametrize("name,spec", BRANCHES)
def test_segmented_prefill_gives_the_whole_prompt_answer(request, name, spec, tiny_segments):
    model, tok, head = _toy(request, name)
    h = head if spec == "auto" else None
    # 0 = the whole-prompt path: no segmentation at any size — with no context
    # checkpoint splitting it, which would give the MTP head's seed a
    # second span with a past to segment over
    whole = ask(engine(model, tok, 0, head=h, ckpts=0), user(LONG), spec, n=24)
    assert not tiny_segments
    got = ask(engine(model, tok, C, head=h), user(LONG), spec, n=24)
    assert tiny_segments, "the segmented path never ran"
    assert all(seg < past for _, past, seg in tiny_segments[-3:])  # several segments
    assert got.text == whole.text


@pytest.mark.parametrize("name", ["dense", "hybrid"])
def test_segmented_logits_allclose(request, name, tiny_segments):
    model, tok, _ = _toy(request, name)
    eng = engine(model, tok, 0)
    ids = _ids(tok, LONG)
    with torch.inference_mode():
        want = prefill.run(model, ids, 0, eng._cache(len(ids) + 1), "cpu", 0)
        got = prefill.run(model, ids, 0, eng._cache(len(ids) + 1), "cpu", C)
    assert tiny_segments
    torch.testing.assert_close(got, want, atol=2e-5, rtol=1e-5)


def test_the_mtp_head_segments_its_own_long_seed(hybrid, tiny_segments):
    model, tok, head = hybrid
    ids = _ids(tok, LONG)
    n = len(ids) - 1
    g = torch.Generator().manual_seed(3)
    hidden = torch.randn(1, n, model.config.hidden_size, generator=g)
    toks = torch.tensor([ids[1:]])
    with torch.inference_mode():
        head.reset()
        want = head.run(hidden, toks, 0)  # no past: the cold causal call
        assert not tiny_segments
        head.reset()
        got = torch.cat([head.run(hidden[:, a:b], toks[:, a:b], a)
                         for a, b in prefill.spans(0, n, C)], 1)
    assert tiny_segments
    torch.testing.assert_close(got, want, atol=2e-5, rtol=1e-5)


def test_the_marker_never_reaches_a_layout_it_does_not_understand(dense, monkeypatch):
    """Anything but full/linear attention on LiveStaticLayers over SDPA keeps
    transformers' own masks — here, a sliding-window layer."""
    model, tok = dense
    eng = engine(model, tok, 0)
    cache = eng._cache(64)
    base = prefill.decoder(model)
    with torch.inference_mode():
        prefill.run(model, list(range(3, 40)), 0, cache, "cpu", 0)
    assert sa.prefill_mask(base, cache, 20, work=8, above=8) is not None
    monkeypatch.setattr(base.config, "layer_types", ["sliding_attention", "full_attention"])
    assert sa.prefill_mask(base, cache, 20, work=8, above=8) is None
