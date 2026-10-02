"""serving/kvcache.py: the prefix slot's cache hands attention its LIVE
window, not its allocation (a 4096-wide slot attended in full costs a
204-token request 28 ms of a 60 ms token, every token — the
module docstring has the measurement).

Every test here is CPU, on the tiny Llama the hot-loop tests use. The
first is the one that fails on a plain StaticCache: it records the K width
SDPA is handed on a decode step.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from tests.hotloop_toys import toy_engine  # noqa: E402


def _widths(model, cache, ids: list[int]):
    """Prefill `ids` then decode one step through `cache`; returns the K
    widths SDPA saw (one per layer per forward) and the last-row logits."""
    import torch.nn.functional as F

    seen: list[int] = []
    real = F.scaled_dot_product_attention

    def spy(q, k, v, *a, **kw):
        seen.append(k.shape[-2])
        return real(q, k, v, *a, **kw)

    F.scaled_dot_product_attention = spy
    try:
        n = len(ids)
        with torch.inference_mode():
            model(torch.tensor([ids]), past_key_values=cache, use_cache=True,
                  cache_position=torch.arange(n), logits_to_keep=1)
            step = model(torch.tensor([[ids[-1]]]), past_key_values=cache, use_cache=True,
                         cache_position=torch.tensor([n])).logits[0, -1]
    finally:
        F.scaled_dot_product_attention = real
    return seen, step


def test_attention_sees_the_live_window_not_the_allocation():
    """A 5-token prompt and one decode step over a 64-wide slot: the step's
    attention runs over 6 columns. (StaticCache: 64 — the regression.)"""
    eng = toy_engine()
    layers = eng.model.config.num_hidden_layers
    seen, _ = _widths(eng.model, eng._cache(64), [3, 4, 5, 6, 7])
    assert seen[:layers] == [5] * layers  # prefill: the prompt
    assert seen[layers:] == [6] * layers  # the step: prompt + one token


def test_live_window_matches_the_full_width_cache():
    """Same model, same prompt, same step: the live window and a plain
    StaticCache over the same allocation return the same last-row logits to
    accumulation order — the columns the window drops were masked to
    exactly zero weight, and what remains is the CPU math kernel summing 7
    columns instead of 64 (measured on this toy: max|diff| 6e-8 against
    logits of magnitude 0.6, one fp32 ulp; AGENTS.md's near-tie caveat is
    exactly this class of difference). The argmax agrees."""
    from transformers import StaticCache

    eng = toy_engine()
    ids = [3, 4, 5, 6, 7, 8, 9]
    _, live = _widths(eng.model, eng._cache(64), ids)
    _, full = _widths(eng.model, StaticCache(config=eng.model.config, max_cache_len=64), ids)
    assert torch.allclose(live, full, rtol=0, atol=1e-6)
    assert live.argmax().item() == full.argmax().item()


def test_rewind_moves_the_counter_and_the_window_together():
    """mtp.py's rejected-draft rewind: three rows written, two dropped — the
    tensor counter and the window agree at 1, and the next write lands at
    row 1 and is handed back as a 3-column window."""
    from drinkme.serving.kvcache import LiveStaticLayer

    eng = toy_engine()
    cache = eng._cache(16)
    layer = cache.layers[0]
    assert isinstance(layer, LiveStaticLayer)
    k = torch.randn(1, 2, 3, 16)
    layer.update(k, k)
    assert layer.live == 3 and int(layer.cumulative_length) == 3
    layer.rewind(2)
    assert layer.live == 1 and int(layer.cumulative_length) == 1
    k2 = torch.randn(1, 2, 2, 16)
    keys, _ = layer.update(k2, k2)
    assert keys.shape[-2] == 3
    assert torch.equal(keys[:, :, 1:], k2)  # written at row 1, not row 3
    assert layer.get_mask_sizes(1) == (4, 0)
    with pytest.raises(ValueError):
        layer.rewind(5)


def test_resync_reads_the_window_back_off_a_restored_counter():
    """slotstore.apply_state overwrites the counter tensor from a slot file
    and calls resync(): the window follows the tensor, not the prime token."""
    eng = toy_engine()
    layer = eng._cache(16).layers[0]
    k = torch.randn(1, 2, 1, 16)
    layer.update(k, k)  # the prime token
    layer.cumulative_length.fill_(7)  # what a restore writes
    layer.resync()
    assert layer.live == 7
    assert layer.get_mask_sizes(1) == (8, 0)


def test_a_restored_slot_attends_over_what_it_holds(tmp_path, monkeypatch):
    """End to end through the store: a slot persisted with 6 tokens and
    restored into a fresh cache attends over 6 + 1 on its next step."""
    from drinkme.serving import slotstore

    eng = toy_engine()
    cache = eng._cache(64)
    ids = [3, 4, 5, 6, 7, 8]
    with torch.inference_mode():
        eng.model(torch.tensor([ids]), past_key_values=cache, use_cache=True,
                  cache_position=torch.arange(6), logits_to_keep=1)
    tensors, flags, ints = slotstore.cache_state(cache)
    fresh = eng._cache(64)
    eng._prime_cache(fresh)
    assert fresh.layers[0].live == 1
    slotstore.apply_state(fresh, {k: v.clone() for k, v in tensors.items()}, flags, ints)
    assert [l.live for l in fresh.layers] == [6] * len(fresh.layers)
    import torch.nn.functional as F

    widths: list[int] = []
    real = F.scaled_dot_product_attention

    def spy(q, k, v, *a, **kw):
        widths.append(k.shape[-2])
        return real(q, k, v, *a, **kw)

    F.scaled_dot_product_attention = spy
    try:
        with torch.inference_mode():
            eng.model(torch.tensor([[9]]), past_key_values=fresh, use_cache=True,
                      cache_position=torch.tensor([6]))
    finally:
        F.scaled_dot_product_attention = real
    assert widths == [7] * len(fresh.layers)


@pytest.mark.parametrize("padded", [False, True])
@pytest.mark.parametrize("suffix", [1, 3])
def test_live_attention_masks_and_grouped_heads(monkeypatch, padded, suffix):
    """Only an unpadded, one-token suffix may omit the causal mask.

    Check what reaches SDPA, and compare logits with the old, always-masked
    decode. Multi-token suffixes are used by both warm prefill and MTP.
    """
    import torch.nn.functional as F
    from drinkme.serving.kvcache import LiveStaticLayer

    eng = toy_engine()
    eng.model.set_attn_implementation("sdpa")
    n = 5
    prompt = torch.tensor([[3, 4, 5, 6, 7]])
    continuation = torch.tensor([[8, 9, 10]])[:, :suffix]
    padding = torch.ones(1, n + suffix, dtype=torch.long) if padded else None
    if padded:
        padding[:, 0] = 0
    seen = []
    real = F.scaled_dot_product_attention

    def spy(q, k, v, *args, **kw):
        seen.append((k.shape[1], kw.get("attn_mask"), kw.get("enable_gqa", False),
                     kw.get("is_causal")))
        return real(q, k, v, *args, **kw)

    monkeypatch.setattr(F, "scaled_dot_product_attention", spy)

    def run():
        cache = eng._cache(64)
        with torch.inference_mode():
            eng.model(prompt, past_key_values=cache, use_cache=True,
                      attention_mask=None if padding is None else padding[:, :n])
            seen.clear()
            out = eng.model(continuation, past_key_values=cache, use_cache=True,
                            attention_mask=padding).logits.clone()
        return out

    current = run()
    assert len(seen) == eng.model.config.num_hidden_layers
    for heads, mask, gqa, causal in seen:
        assert not causal  # a suffix must never use SDPA's upper-left causal triangle
        if not padded and suffix == 1:
            assert mask is None
            assert gqa
            assert heads == eng.model.config.num_key_value_heads
        else:
            assert not gqa
            assert heads == eng.model.config.num_attention_heads
            expected = torch.arange(n + suffix)[None, :] <= torch.arange(n, n + suffix)[:, None]
            if padded:
                expected[:, 0] = False
            assert torch.equal(mask[0, 0], expected)
    # Reinstates the inherited StaticLayer flag as the reference.
    monkeypatch.setattr(LiveStaticLayer, "is_compileable", True)
    reference = run()
    torch.testing.assert_close(current, reference, rtol=0, atol=1e-6)


def test_mixed_cache_keeps_sliding_window_mask():
    """Mixed full/sliding layouts retain masks, including across the
    sliding-window boundary in a multi-token verification batch."""
    from transformers import LlamaConfig
    from transformers.cache_utils import StaticSlidingWindowLayer
    from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
    from drinkme.serving.kvcache import LiveStaticCache

    cfg = LlamaConfig(hidden_size=32, num_attention_heads=2, num_key_value_heads=1,
                      num_hidden_layers=2, layer_types=["sliding_attention", "full_attention"],
                      sliding_window=4)
    cfg._attn_implementation = "sdpa"
    cache = LiveStaticCache(cfg, max_cache_len=32)
    assert type(cache.layers[0]) is StaticSlidingWindowLayer
    assert cache.is_compileable  # mixed static layouts retain their mask policy
    for i in range(2):
        k = torch.randn(1, 1, 7, 16)
        cache.update(k, k, i)
    args = dict(config=cfg, inputs_embeds=torch.randn(1, 1, 32),
                attention_mask=None, past_key_values=cache)
    assert create_causal_mask(**args) is not None
    mask = create_sliding_window_causal_mask(**args)
    assert mask is not None
    # The ring returns only the last four rows for a single-token decode.
    assert mask[0, 0, 0].tolist() == [True] * 4
    # A verification batch also sees rows that fall outside each query's
    # window. Both those old rows and future tokens must stay masked.
    args["inputs_embeds"] = torch.randn(1, 3, 32)
    mask = create_sliding_window_causal_mask(**args)
    assert mask[0, 0].tolist() == [
        [True, True, True, True, False, False],
        [False, True, True, True, True, False],
        [False, False, True, True, True, True],
    ]


def test_sliding_allocation_smaller_than_window_masks_unwritten_rows():
    """The local-window skip check alone cannot protect a short allocation:
    kv_length < sliding_window even though KV still includes unused rows."""
    from transformers import LlamaConfig
    from transformers.masking_utils import create_sliding_window_causal_mask
    from drinkme.serving.kvcache import LiveStaticCache

    cfg = LlamaConfig(hidden_size=32, num_attention_heads=2, num_key_value_heads=1,
                      num_hidden_layers=2, layer_types=["sliding_attention", "full_attention"],
                      sliding_window=64)
    cfg._attn_implementation = "sdpa"
    cache = LiveStaticCache(cfg, max_cache_len=16)
    for i in range(2):
        k = torch.randn(1, 1, 5, 16)
        cache.update(k, k, i)
    mask = create_sliding_window_causal_mask(
        config=cfg, inputs_embeds=torch.randn(1, 1, 32),
        attention_mask=None, past_key_values=cache)
    assert mask is not None
    assert mask[0, 0, 0].tolist() == [True] * 6 + [False] * 10
