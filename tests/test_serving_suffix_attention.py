"""MTP suffix causality, conservative routing, and rejected-row rollback."""
import copy
from types import SimpleNamespace

import pytest
import torch
from transformers import StaticCache
from transformers.integrations.sdpa_attention import sdpa_attention_forward

from drinkme.serving import mtp, suffix_attention as suffix
from drinkme.serving.kvcache import LiveStaticCache
from test_serving_mtp import _cfg, _torch_chunk_on_cpu, toy  # noqa: F401


@pytest.mark.parametrize('rows', [2, 5, 8])
@pytest.mark.parametrize('past', [0, 3, 31])
def test_suffix_queries_see_all_past_and_only_their_own_prefix(rows, past):
    length = past + rows
    q = torch.zeros(1, 6, rows, 16)
    k = torch.zeros(1, 2, length, 16)
    v = torch.arange(length, dtype=q.dtype)[None, None, :, None].expand_as(k)
    out, _ = suffix.attention_forward(None, q, k, v, suffix.causal_lower_right(rows, length))
    expected = (past + torch.arange(rows, dtype=q.dtype))/2
    torch.testing.assert_close(out, expected[None, :, None, None].expand_as(out))


@pytest.mark.parametrize('additive', [False, True])
def test_custom_padding_and_sliding_masks_use_upstream(additive):
    torch.manual_seed(11)
    q = torch.randn(1, 6, 3, 16)
    k, v = torch.randn(2, 1, 2, 9, 16)
    allowed = torch.tensor([[0, 0, 0, 1, 1, 1, 1, 0, 0],
                            [0, 0, 0, 0, 1, 1, 1, 1, 0],
                            [0, 0, 0, 0, 0, 1, 1, 1, 1]], dtype=torch.bool)[None, None]
    mask = torch.zeros_like(allowed, dtype=q.dtype).masked_fill(~allowed, -torch.inf) if additive else allowed
    module = SimpleNamespace(num_key_value_groups=3, is_causal=True)
    expected = sdpa_attention_forward(module, q, k, v, mask, scaling=.2)[0]
    actual = suffix.attention_forward(module, q, k, v, mask, scaling=.2)[0]
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_install_is_local_idempotent_and_preserves_parameters():
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM
    model = Qwen3_5ForCausalLM(_cfg()).eval()
    model.set_attn_implementation('sdpa')
    config = model.config
    parameters = tuple(model.parameters())
    suffix.install(model)
    attention = model.model.layers[1].self_attn
    own = attention.config
    assert own is not config and config._attn_implementation == 'sdpa'
    assert own._attn_implementation == suffix.NAME
    suffix.install(model)
    assert attention.config is own
    assert all(a is b for a, b in zip(parameters, model.parameters()))
    attention.config = copy.copy(config)
    attention.config._attn_implementation = 'eager'
    suffix.install(model)
    assert attention.config._attn_implementation == 'eager'


def test_trunk_requires_live_full_cache_and_supported_layout(toy, monkeypatch):
    model, _, _, cfg = toy
    base = model.model
    ids = torch.tensor([[3, 4, 5]])
    live = LiveStaticCache(config=cfg, max_cache_len=32)
    assert suffix.trunk_mask(base, ids, live) is None  # CPU remains upstream.
    monkeypatch.setattr(suffix, 'enabled', lambda rows, device: 1 < rows <= 8)
    assert suffix.trunk_mask(base, ids, StaticCache(config=cfg, max_cache_len=32)) is None
    assert suffix.trunk_mask(base, ids, None) is None
    assert suffix.trunk_mask(base, ids.expand(2, -1), live) is None
    assert suffix.trunk_mask(base, ids[:, :1], live) is None
    assert suffix.trunk_mask(base, ids.repeat(1, 3), live) is None
    mask = suffix.trunk_mask(base, ids, live)
    assert mask is not None and mask['linear_attention'] is None
    live.layers[1].live = 2
    assert suffix.trunk_mask(base, ids, live) is None
    monkeypatch.setattr(base.config, 'layer_types', ['sliding_attention']*4)
    assert suffix.trunk_mask(base, ids, live) is None


@pytest.mark.parametrize('keep', [1, 3, 5])
def test_suffix_verify_rewind_and_next_write_match_serial(toy, monkeypatch, keep):
    model, _, _, cfg = toy
    monkeypatch.setattr(suffix, 'enabled', lambda rows, device: 1 < rows <= 8)
    caches = [LiveStaticCache(config=cfg, max_cache_len=64) for _ in range(2)]
    prompt = torch.tensor([[5, 7, 11, 13, 17, 19, 23, 29]])
    ids = torch.tensor([[31, 37, 41, 43, 47]])
    with torch.inference_mode():
        for cache in caches:
            model(prompt, past_key_values=cache, use_cache=True)
        cap = mtp._Capture()
        monkeypatch.setattr(mtp, '_ACTIVE', cap)
        _, actual = mtp.forward_with_hidden(model, ids, caches[0], torch.arange(8, 13))
        monkeypatch.setattr(mtp, '_ACTIVE', None)
        cap.rows = 5
        mtp._restore_rows(caches[0], cap, keep)
        expected = []
        for i in range(keep):
            _, row = mtp.forward_with_hidden(model, ids[:, i:i+1], caches[1], torch.tensor([8+i]))
            expected.append(row)
        torch.testing.assert_close(actual[:keep], torch.cat(expected), atol=2e-6, rtol=2e-5)
        next_id = torch.tensor([[53]])
        next_rows = [mtp.forward_with_hidden(model, next_id, c, torch.tensor([8+keep]))[1] for c in caches]
        torch.testing.assert_close(*next_rows, atol=2e-6, rtol=2e-5)
        assert [c.layers[1].live for c in caches] == [9+keep]*2


def test_suffix_rejects_a_wrong_cache_length():
    q = torch.zeros(1, 2, 3, 16)
    k = torch.zeros(1, 1, 8, 16)
    with pytest.raises(ValueError, match='live KV window'):
        suffix.attention_forward(None, q, k, k, suffix.causal_lower_right(3, 7))


@pytest.mark.parametrize('rows', [2, 4, 8])
def test_head_rebuild_and_crop_match_explicit_mask(toy, monkeypatch, rows):
    _, _, head, cfg = toy
    torch.manual_seed(21)
    h = torch.randn(1, 3 + rows + 1, cfg.hidden_size)
    ids = torch.randint(0, cfg.vocab_size, (1, h.shape[1]))
    outputs = []
    with torch.inference_mode():
        for native in (False, True):
            monkeypatch.setattr(suffix, 'enabled', lambda n, device: native and 1 < n <= 8)
            head.reset()
            head.run(h[:, :3], ids[:, :3], 0)
            result = head.run(h[:, 3:3+rows], ids[:, 3:3+rows], 3)
            head.crop(rows-1)
            next_row = head.run(h[:, -1:], ids[:, -1:], 4)
            outputs.append((result, next_row))
        for a, b in zip(*outputs):
            torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-5)


def test_device_row_and_opt_out_guards(monkeypatch):
    monkeypatch.delenv(suffix.ENV, raising=False)
    assert suffix.enabled(5, 'cuda')
    assert not suffix.enabled(5, 'cpu')
    assert not suffix.enabled(1, 'cuda')
    assert not suffix.enabled(9, 'cuda')
    monkeypatch.setenv(suffix.ENV, '0')
    assert not suffix.enabled(5, 'cuda')


@pytest.mark.skipif(not torch.cuda.is_available(), reason='native GPU suffix attention gate')
@pytest.mark.parametrize('rows', [2, 4, 5, 8])
@pytest.mark.parametrize('length', [128, 2048, 8192])
def test_gpu_grouped_suffix_matches_explicit_mask(rows, length):
    torch.manual_seed(42)
    q = torch.randn(1, 24, rows, 256, dtype=torch.bfloat16, device='cuda')
    # Both KV views have the strides of the larger resident allocation.
    k, v = [torch.randn(1, 4, 16384, 256, dtype=q.dtype, device=q.device)[:, :, :length]
            for _ in range(2)]
    mask = torch.arange(length, device=q.device)[None, :] <= (
        length-rows+torch.arange(rows, device=q.device))[:, None]
    module = SimpleNamespace(num_key_value_groups=6, is_causal=True)
    expected = sdpa_attention_forward(module, q, k, v, mask)[0]
    actual = suffix.attention_forward(module, q, k, v, suffix.causal_lower_right(rows, length))[0]
    torch.testing.assert_close(actual, expected, atol=.004, rtol=.02)
