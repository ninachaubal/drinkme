"""mtp.head_logits: a speculative verify row's logits are the wrapper's. CPU,
float32, tiny random models, no downloads.

mtp.forward_with_hidden runs the decoder and lm_head itself (the draft
head needs the hidden states the wrapper drops), so any transform a
family's forward applies after lm_head has to be applied there too. gemma-4
softcaps (the serial step and the prefill read the softcapped logits through
the wrapper; a sampled n-gram verify on gemma drew from the raw ones),
Muse-Glimmer multiplies then softcaps, granite divides. Each family's
verify rows, every row and the last row alone, are pinned bit for bit
against its own wrapper's logits over the same cache layout; for the
transforming families the raw lm_head output is not (the negative
control); Qwen3's are lm_head's own.

mtp.trunk_ids: the same forwards embed a modality placeholder id the way
the wrapper does (gemma-4's image, video and audio ids as the pad id,
Muse-Glimmer's image and video ids as 0), pinned bit for bit over ids that
carry them, with the text model over the raw ids as the negative control;
every other family's ids pass through as the same tensor.
"""

from __future__ import annotations

import pytest
import torch

from drinkme.serving import mtp
from drinkme.serving.kvcache import LiveStaticCache

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")

V = 96


def _gemma4():
    from transformers import Gemma4ForConditionalGeneration

    from test_serving_gemma_vision import config

    return Gemma4ForConditionalGeneration(config(16))


def _gemma4_text():
    from transformers import Gemma4ForCausalLM, Gemma4TextConfig

    return Gemma4ForCausalLM(Gemma4TextConfig(
        vocab_size=V, hidden_size=64, intermediate_size=128, num_hidden_layers=6,
        num_attention_heads=2, num_key_value_heads=1, head_dim=32, global_head_dim=32,
        num_global_key_value_heads=1, layer_types=["sliding_attention"] * 5 + ["full_attention"],
        sliding_window=16, max_position_embeddings=256, final_logit_softcapping=30.0,
        attention_k_eq_v=True, hidden_size_per_layer_input=0, vocab_size_per_layer_input=V,
        tie_word_embeddings=True, eos_token_id=None, pad_token_id=None))


def _glimmer():
    pytest.importorskip("transformers.models.muse_glimmer")
    from transformers.models.muse_glimmer import (MuseGlimmerConfig,
                                                  MuseGlimmerForConditionalGeneration)

    return MuseGlimmerForConditionalGeneration(MuseGlimmerConfig(
        text_config=dict(vocab_size=V, hidden_size=64, intermediate_size=128,
                         num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
                         head_dim=32, max_position_embeddings=256, sliding_window=16,
                         output_multiplier=1.7, bos_token_id=None, eos_token_id=None,
                         pad_token_id=1),
        vision_config=dict(hidden_size=32, intermediate_size=64, num_hidden_layers=1,
                           num_attention_heads=2, pos_emb_height=4, pos_emb_width=4,
                           max_position_embeddings=16),
        out_hidden_size=64, projector_hidden_size=64))


def _granite():
    from transformers import GraniteConfig, GraniteForCausalLM

    return GraniteForCausalLM(GraniteConfig(
        vocab_size=V, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=256,
        logits_scaling=8.0, eos_token_id=None, pad_token_id=None, bos_token_id=None))


def _qwen3():
    from transformers import Qwen3Config, Qwen3ForCausalLM

    return Qwen3ForCausalLM(Qwen3Config(
        vocab_size=V, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=1, head_dim=32,
        max_position_embeddings=256, tie_word_embeddings=False, eos_token_id=None,
        pad_token_id=None))


FAMILIES = {"gemma4": (_gemma4, True), "gemma4_text": (_gemma4_text, True),
            "muse_glimmer": (_glimmer, True), "granite": (_granite, True),
            "qwen3": (_qwen3, False)}


@pytest.mark.parametrize("family", sorted(FAMILIES))
def test_verify_rows_are_the_wrappers_logits(family):
    build, transforms = FAMILIES[family]
    torch.manual_seed(0)
    model = build().eval().float()
    assert model.config.model_type == family
    with torch.no_grad():  # logits the size a trained model's are (10-30), where the cap bites
        model.lm_head.weight.mul_(40 / model.lm_head.weight.norm(dim=1).mean())
    # above the toys' specials: the placeholder ids are
    # test_a_placeholder_id_is_embedded_as_the_wrapper_embeds_it's
    ids = torch.randint(24, V, (1, 24), generator=torch.Generator().manual_seed(1))
    pos = torch.arange(24)

    def cache():
        return LiveStaticCache(config=model.config, max_cache_len=64)

    with torch.inference_mode():
        want = model(ids, past_key_values=cache(), use_cache=True, cache_position=pos).logits[0]
        last = model(ids, past_key_values=cache(), use_cache=True, cache_position=pos,
                     logits_to_keep=1).logits[0]
        hidden, got = mtp.forward_with_hidden(model, ids, cache(), pos)
        _, got_last = mtp.forward_with_hidden(model, ids, cache(), pos, last_row_only=True)
        raw = model.lm_head(hidden)[0]
    assert torch.equal(got, want)
    assert torch.equal(got_last, last)
    if transforms:
        assert (raw - want).abs().max() > 0.1
    else:
        assert torch.equal(raw, want)


def _glimmer_with_placeholders():
    """_glimmer with its image and video token ids inside the toy vocabulary."""
    model = _glimmer()
    cfg = model.config
    cfg.image_token_id, cfg.video_token_id = 20, 21
    return model


PLACEHOLDERS = {"gemma4": _gemma4, "muse_glimmer": _glimmer_with_placeholders}


@pytest.mark.parametrize("family", sorted(PLACEHOLDERS))
def test_a_placeholder_id_is_embedded_as_the_wrapper_embeds_it(family):
    """A prompt's text or a generated token can be a modality placeholder
    id; the wrapper embeds it as another id, and so do the verify rows
    (forward_with_hidden) and a chunked prefill's spans before the last
    (prefill.run), which call the text model themselves. The text model
    over the raw ids is not the wrapper (the negative control)."""
    from drinkme.serving import prefill

    torch.manual_seed(0)
    model = PLACEHOLDERS[family]().eval().float()
    cfg = model.config
    placeholders = [t for t in (cfg.image_token_id, cfg.video_token_id,
                                getattr(cfg, "audio_token_id", None)) if t is not None]
    ids = torch.randint(24, V, (1, 24), generator=torch.Generator().manual_seed(1))
    ids[0, 3], ids[0, 9], ids[0, 17] = placeholders[0], placeholders[1], placeholders[-1]
    pos = torch.arange(24)

    def cache():
        return LiveStaticCache(config=model.config, max_cache_len=64)

    with torch.inference_mode():
        want = model(ids, past_key_values=cache(), use_cache=True, cache_position=pos).logits[0]
        _, got = mtp.forward_with_hidden(model, ids, cache(), pos)
        raw = mtp.head_logits(model, model.get_decoder()(
            ids, past_key_values=cache(), use_cache=True, cache_position=pos).last_hidden_state)[0]
        chunked = prefill.run(model, ids[0].tolist(), 0, cache(), "cpu", 7)
    assert torch.equal(got, want)
    assert (raw - want).abs().max() > 1e-4
    torch.testing.assert_close(chunked, want[-1], atol=2e-5, rtol=1e-5)
    mapped = mtp.trunk_ids(model, ids)
    assert set(mapped[0].tolist()) & set(placeholders) == set()


@pytest.mark.parametrize("family", ["gemma4_text", "granite", "qwen3"])
def test_other_families_ids_pass_through_untouched(family):
    torch.manual_seed(0)
    model = FAMILIES[family][0]().eval()
    ids = torch.arange(24)[None]
    assert mtp.trunk_ids(model, ids) is ids
    assert mtp.trunk_ids(model, None) is None
