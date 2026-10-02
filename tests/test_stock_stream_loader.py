"""The bench's STOCK arm loads by streaming, not by `from_pretrained`.

MEASURED on a Strix Halo (124 GiB unified):
transformers' `from_pretrained(device_map="cuda", low_cpu_mem_usage=True)`
on gemma-4-31B-it (58.25 GB bf16) took MemAvailable 107 -> 5 GB in ~30 s of
"Loading weights" — a ~2x transient whether or not the load is asked to go
straight to the device. The 2x charge in fit.stock_transient_bytes is
therefore right FOR THAT LOADER, and it correctly refused the 27B (55.6 GB)
and Muse-Glimmer-30B (55.7 GB) stock arms at 09:2x. The streaming loader
(arms.load_stock_streaming: meta skeleton + serving.engines.stream_checkpoint,
the raw-remainder walker `drinkme serve` already uses) holds one tensor of
host memory beyond the resident model, so its charge is resident + the
largest tensor in the checkpoint's shard headers.

Three things are pinned here, CPU only:

1. THE LOAD-BEARING TEST: the streamed tree's forward is BITWISE the
   `from_pretrained` tree's — `torch.equal` on logits, on toys that exercise
   tied embeddings (lm_head absent from the checkpoint), a multi-shard
   index, gemma-4's computed `embed_scale` and per-layer-type `inv_freq`
   (text tower alone, and the real checkpoint's text+vision wrapper), and
   the vision-pruned muse_glimmer wrapper.
2. The arithmetic, with the Strix Halo box's numbers: gemma fits 124 GiB under the
   streaming charge and does not under the from_pretrained charge (which
   stays selectable as `direct=False`); the 72B (145.4 GB) fits under
   neither.
3. bench's plan: `--dry-run` shows gemma / the 27B / Glimmer as 3-arm ratio
   points and the 72B as a fit point on a 124 GiB unified box; the old
   arithmetic is still the plan when the from_pretrained loader is chosen.
"""

from __future__ import annotations

import ast
import os
import pathlib
import sys
import tempfile

import pytest
import torch

from drinkme import arms, bench
from drinkme.detect import Hardware
from drinkme.fit import GB, GIB, STAGING_SHARD_BYTES, stock_arm_fits, stock_transient_bytes
from drinkme.suggest import MODELS, RATIO_HEADROOM, sizes

ROOT = pathlib.Path(__file__).resolve().parents[1]

# Strix Halo: 124 GiB unified; MemAvailable read 107 GB right
# before the gemma load that the watchdog killed
STRIX_HALO_CEILING = 124.0 * GIB
STRIX_HALO_LIVE_AVAILABLE = 107 * GB
# gemma-4-31B-it's largest tensor is embed_tokens: 262144 x 5376 x 2 bytes
GEMMA_EMBED_BYTES = 262144 * 5376 * 2


def _m(name: str):
    return next(x for x in MODELS if x.name == name)


def _bf16_gb(name: str) -> float:
    """The row's checkpoint bytes, decimal GB (suggest.sizes)."""
    return sizes(_m(name)).bf16_gb


# ------------------------------------------------------------ arithmetic --


def test_streaming_charge_is_resident_plus_one_tensor_on_unified():
    assert stock_transient_bytes(100, 0, "unified", direct=True, tensor_bytes=7) == 107
    assert stock_transient_bytes(100, 5, "unified", direct=True, tensor_bytes=7) == 112


def test_streaming_charge_falls_back_to_the_staging_shard_without_headers():
    """No snapshot on disk yet (bench downloads at load time) -> the largest
    tensor is unknown; charge HF's own max shard size, an upper bound on
    any tensor that lives inside a shard."""
    assert stock_transient_bytes(100, 0, "unified", direct=True) == 100 + STAGING_SHARD_BYTES
    assert stock_transient_bytes(100, 0, "unified", direct=True, tensor_bytes=None) \
        == 100 + STAGING_SHARD_BYTES


def test_from_pretrained_charge_stays_selectable_as_2x():
    """The old arithmetic is the from_pretrained path's cost (measured) and
    stays reachable — `direct=False` — for the loader that still runs it."""
    assert stock_transient_bytes(100, 0, "unified", direct=False) == 200
    assert stock_transient_bytes(100, 10, "unified", direct=False) == 210
    assert stock_transient_bytes(100, 0, "unified", direct=False, tensor_bytes=7) == 200


def test_the_default_is_the_conservative_from_pretrained_charge():
    """An unqualified call must never approve the load that OOM'd the box:
    only a caller that KNOWS the streaming loader will run says so."""
    assert stock_transient_bytes(100, 0, "unified") == 200
    assert not stock_arm_fits(_bf16_gb("gemma-4-31B-it") * GB, 0.0, STRIX_HALO_CEILING,
                              "unified", RATIO_HEADROOM)


def test_discrete_charges_resident_under_either_loader():
    """VRAM is a separate pool: the one host-side tensor (streaming) or the
    whole host staging copy (from_pretrained) never touches it."""
    assert stock_transient_bytes(100, 0, "vram", direct=True, tensor_bytes=7) == 100
    assert stock_transient_bytes(100, 0, "vram", direct=False) == 100


def test_strix_halo_gemma_fits_124gib_under_the_streaming_loader():
    """The target: 62.55 GB + one 2.82 GB tensor, x1.15 = 75.2 GB
    against 133 GB physical — a 3-arm ratio point. Under from_pretrained's
    2x it is 143.9 GB and stays refused (an OOM)."""
    m = _m("gemma-4-31B-it")
    assert stock_arm_fits(_bf16_gb(m.name) * GB, 0.0, STRIX_HALO_CEILING, "unified", RATIO_HEADROOM,
                          direct=True, tensor_bytes=GEMMA_EMBED_BYTES)
    assert stock_arm_fits(_bf16_gb(m.name) * GB, 0.0, STRIX_HALO_CEILING, "unified", RATIO_HEADROOM,
                          direct=True)  # even on the 5 GiB fallback
    assert not stock_arm_fits(_bf16_gb(m.name) * GB, 0.0, STRIX_HALO_CEILING, "unified", RATIO_HEADROOM,
                              direct=False)


def test_strix_halo_27b_and_glimmer_fit_under_the_streaming_loader():
    for name in ("Qwen3.8-27B", "Muse-Glimmer-30B"):
        m = _m(name)
        assert stock_arm_fits(_bf16_gb(m.name) * GB, 0.0, STRIX_HALO_CEILING, "unified",
                              RATIO_HEADROOM, direct=True), name


def test_strix_halo_72b_does_not_fit_under_either_loader():
    """145.41 GB bf16 is over the 133 GB pool before any transient: the 72B
    stays a fit point on a 124 GiB box, streaming loader or not."""
    m = _m("Qwen2.5-72B")
    assert not stock_arm_fits(_bf16_gb(m.name) * GB, 0.0, STRIX_HALO_CEILING, "unified",
                              RATIO_HEADROOM, direct=True, tensor_bytes=152064 * 8192 * 2)
    assert not stock_arm_fits(_bf16_gb(m.name) * GB, 0.0, STRIX_HALO_CEILING, "unified",
                              RATIO_HEADROOM, direct=False)


# ------------------------------------------------------- the live guard --


def test_live_guard_passes_gemma_at_strix_halos_live_reading_when_streaming():
    m = _m("gemma-4-31B-it")
    assert arms.refuse_if_stock_wont_fit(m.hf_repo, _bf16_gb(m.name) * GB, "unified",
                                         STRIX_HALO_LIVE_AVAILABLE, direct=True,
                                         tensor_bytes=GEMMA_EMBED_BYTES) is None


def test_live_guard_still_refuses_gemma_for_the_from_pretrained_loader():
    m = _m("gemma-4-31B-it")
    with pytest.raises(SystemExit, match=r"refusing to load.*x2") as ei:
        arms.refuse_if_stock_wont_fit(m.hf_repo, _bf16_gb(m.name) * GB, "unified",
                                      STRIX_HALO_LIVE_AVAILABLE, direct=False)
    assert "from_pretrained" in str(ei.value)


def test_live_guard_names_the_streaming_term_when_it_refuses():
    """A refusal under the streaming charge says what was charged: resident
    + one tensor, never 'x2'."""
    m = _m("Qwen2.5-72B")
    with pytest.raises(SystemExit, match="refusing to load") as ei:
        arms.refuse_if_stock_wont_fit(m.hf_repo, _bf16_gb(m.name) * GB, "unified",
                                      STRIX_HALO_LIVE_AVAILABLE, direct=True,
                                      tensor_bytes=2 * GB)
    msg = str(ei.value)
    assert "x2" not in msg and "one tensor" in msg and "2.00 GB" in msg


# ------------------------------------------------------ the largest tensor --


def _word_tokenizer(vocab_size: int):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast

    vocab = {"<unk>": 0, "<pad>": 1, "</s>": 2}
    for w in ("hello", "world", "the", "quick", "brown", "fox", "jumps", "over"):
        vocab[w] = len(vocab)
    while len(vocab) < vocab_size:
        vocab[f"tok{len(vocab)}"] = len(vocab)
    backend = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
    backend.pre_tokenizer = Whitespace()
    return PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="<unk>",
                                   pad_token="<pad>", eos_token="</s>")


def _save(tmp_path_factory, tag: str, cfg, cls, max_shard_size=None) -> str:
    torch.manual_seed(0)
    d = str(tmp_path_factory.mktemp(tag) / "model")
    os.makedirs(d)
    _word_tokenizer(cfg.get_text_config().vocab_size).save_pretrained(d)
    model = cls(cfg).to(torch.bfloat16).eval()
    kw = {"max_shard_size": max_shard_size} if max_shard_size else {}
    model.save_pretrained(d, safe_serialization=True, **kw)
    del model
    return d


@pytest.fixture(scope="module")
def llama_tied_sharded(tmp_path_factory):
    """A tied-embedding Llama saved as SEVERAL shards with an index: lm_head
    is absent from the checkpoint (tie_weights must supply it) and the
    walker must cross shard boundaries."""
    from transformers import LlamaConfig, LlamaForCausalLM

    cfg = LlamaConfig(vocab_size=64, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128,
                      tie_word_embeddings=True, attention_bias=True, eos_token_id=2, pad_token_id=1)
    return _save(tmp_path_factory, "llama_tied", cfg, LlamaForCausalLM, max_shard_size="20KB")


@pytest.fixture(scope="module")
def qwen3_untied(tmp_path_factory):
    from transformers import Qwen3Config, Qwen3ForCausalLM

    cfg = Qwen3Config(vocab_size=64, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=16,
                      max_position_embeddings=128, tie_word_embeddings=False,
                      eos_token_id=2, pad_token_id=1)
    return _save(tmp_path_factory, "qwen3", cfg, Qwen3ForCausalLM)


@pytest.fixture(scope="module")
def gemma4_text_tied(tmp_path_factory):
    """gemma-4's text tower at toy scale: tied head, a computed embed_scale
    buffer and one rotary pair per layer type — every buffer the checkpoint
    does NOT carry and the loader must construct (engines'
    _rebuild_computed_buffers), on gemma, the first arch that needed it."""
    from transformers import Gemma4ForCausalLM, Gemma4TextConfig

    layer_types = (["sliding_attention"] * 5 + ["full_attention"]) * 2
    cfg = Gemma4TextConfig(
        vocab_size=96, hidden_size=64, intermediate_size=128, num_hidden_layers=12,
        num_attention_heads=2, num_key_value_heads=1, head_dim=32, global_head_dim=32,
        num_global_key_value_heads=1, layer_types=layer_types, sliding_window=16,
        max_position_embeddings=1024, final_logit_softcapping=30.0, attention_k_eq_v=True,
        hidden_size_per_layer_input=0, vocab_size_per_layer_input=96,
        tie_word_embeddings=True, eos_token_id=None, pad_token_id=None)
    return _save(tmp_path_factory, "gemma4", cfg, Gemma4ForCausalLM)


@pytest.fixture(scope="module")
def gemma4_wrapper_with_vision(tmp_path_factory):
    """gemma-4-31B-it's actual shape: the ForConditionalGeneration WRAPPER
    (text tower + vision tower, tied head). gemma4 is NOT in
    arms._NON_TEXT_TOWERS, so — exactly as load_cpu keeps it — the vision
    tower stays on the tree and its tensors stream too; its rotary is the
    one whose rebuild through the OWNING config matters on the served path
    (engines._owning_config)."""
    from transformers import Gemma4Config, Gemma4ForConditionalGeneration

    layer_types = (["sliding_attention"] * 5 + ["full_attention"]) * 2
    cfg = Gemma4Config(
        text_config=dict(vocab_size=96, hidden_size=64, intermediate_size=128,
                         num_hidden_layers=12, num_attention_heads=2, num_key_value_heads=1,
                         head_dim=32, global_head_dim=32, num_global_key_value_heads=1,
                         layer_types=layer_types, sliding_window=16,
                         max_position_embeddings=1024, final_logit_softcapping=30.0,
                         attention_k_eq_v=True, hidden_size_per_layer_input=0,
                         vocab_size_per_layer_input=96, tie_word_embeddings=True,
                         eos_token_id=None, pad_token_id=1),
        vision_config=dict(hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                           num_attention_heads=4, image_size=32, patch_size=16))
    return _save(tmp_path_factory, "gemma4_wrap", cfg, Gemma4ForConditionalGeneration)


@pytest.fixture(scope="module")
def glimmer_with_vision(tmp_path_factory):
    """The vision-pruned wrapper case (arms._NON_TEXT_TOWERS): the checkpoint
    SHIPS a vision tower the text skeleton never builds; the stock arm must
    stay text-only exactly as load_cpu makes it."""
    pytest.importorskip("transformers.models.muse_glimmer",
                        reason="muse_glimmer arrived in transformers 5.15.0")
    from transformers.models.muse_glimmer import (MuseGlimmerConfig,
                                                  MuseGlimmerForConditionalGeneration)

    cfg = MuseGlimmerConfig(
        text_config=dict(vocab_size=64, hidden_size=64, intermediate_size=128,
                         num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                         head_dim=16, max_position_embeddings=128, sliding_window=64,
                         bos_token_id=None, eos_token_id=2, pad_token_id=1),
        vision_config=dict(hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                           num_attention_heads=4, pos_emb_height=4, pos_emb_width=4,
                           max_position_embeddings=16),
        out_hidden_size=64, projector_hidden_size=64)
    return _save(tmp_path_factory, "glimmer", cfg, MuseGlimmerForConditionalGeneration)


def test_largest_tensor_bytes_reads_the_shard_headers(llama_tied_sharded, qwen3_untied):
    from drinkme.serving.checkpoint import largest_tensor_bytes

    # at this toy geometry the MLP gate/up (intermediate 128 x hidden 64,
    # bf16) is the biggest tensor; in the sharded llama it sits in one of
    # several files and the headers of all of them are read
    assert len([f for f in os.listdir(llama_tied_sharded) if f.endswith(".safetensors")]) > 1
    assert largest_tensor_bytes(llama_tied_sharded) == 128 * 64 * 2
    assert largest_tensor_bytes(qwen3_untied) == 128 * 64 * 2
    assert largest_tensor_bytes(str(pathlib.Path(qwen3_untied).parent)) is None  # no shards there


# ---------------------------------------------- THE LOAD-BEARING TEST --


def _assert_same_tree_and_forward(streamed, stock):
    """Bitwise: parameters, buffers, and the forward's logits."""
    assert not any(p.is_meta for p in streamed.parameters())
    assert not any(b.is_meta for b in streamed.buffers())
    sp, kp = dict(streamed.named_parameters()), dict(stock.named_parameters())
    assert set(sp) == set(kp), set(sp) ^ set(kp)
    assert [n for n in kp if not torch.equal(sp[n], kp[n])] == []
    sb, kb = dict(streamed.named_buffers()), dict(stock.named_buffers())
    assert set(sb) == set(kb), set(sb) ^ set(kb)
    assert [n for n in kb if not torch.equal(sb[n], kb[n])] == []
    assert type(streamed) is type(stock)
    assert streamed.config._attn_implementation == stock.config._attn_implementation
    ids = torch.arange(3, 15)[None]
    with torch.inference_mode():
        want = stock(ids, use_cache=False).logits
        got = streamed(ids, use_cache=False).logits
    assert want.dtype == torch.bfloat16 == got.dtype
    assert torch.equal(got, want), (got.float() - want.float()).abs().max().item()
    # and through the bench's own KV-cached greedy loop, the shape that is timed
    assert torch.equal(arms.greedy(streamed, ids, 6), arms.greedy(stock, ids, 6))


def test_streamed_llama_tied_sharded_is_bitwise_from_pretrained(llama_tied_sharded):
    stock = arms.load_cpu(llama_tied_sharded)
    streamed = arms.load_stock_streaming(llama_tied_sharded, device="cpu")
    assert streamed.lm_head.weight is streamed.model.embed_tokens.weight  # retied
    _assert_same_tree_and_forward(streamed, stock)


def test_streamed_qwen3_untied_is_bitwise_from_pretrained(qwen3_untied):
    stock = arms.load_cpu(qwen3_untied)
    streamed = arms.load_stock_streaming(qwen3_untied, device="cpu")
    assert streamed.lm_head.weight is not streamed.model.embed_tokens.weight
    _assert_same_tree_and_forward(streamed, stock)


def test_streamed_gemma4_text_is_bitwise_from_pretrained(gemma4_text_tied, capsys):
    stock = arms.load_cpu(gemma4_text_tied)
    streamed = arms.load_stock_streaming(gemma4_text_tied, device="cpu")
    _assert_same_tree_and_forward(streamed, stock)
    # every computed buffer was REBUILT, none zero-filled by the last-resort valve
    assert "zero-materialized" not in capsys.readouterr().out


def test_streamed_gemma4_wrapper_is_bitwise_from_pretrained(gemma4_wrapper_with_vision, capsys):
    from safetensors import safe_open

    with safe_open(os.path.join(gemma4_wrapper_with_vision, "model.safetensors"),
                   framework="pt") as sf:
        keys = list(sf.keys())
    assert any(k.startswith("model.vision_tower.") for k in keys) and "lm_head.weight" not in keys
    stock = arms.load_cpu(gemma4_wrapper_with_vision)
    streamed = arms.load_stock_streaming(gemma4_wrapper_with_vision, device="cpu")
    assert type(streamed).__name__ == "Gemma4ForConditionalGeneration"
    assert streamed.model.vision_tower is not None  # kept, exactly as load_cpu keeps it
    _assert_same_tree_and_forward(streamed, stock)
    assert "zero-materialized" not in capsys.readouterr().out


def test_streamed_glimmer_keeps_the_tower_bitwise_from_pretrained(glimmer_with_vision):
    stock = arms.load_cpu(glimmer_with_vision)
    streamed = arms.load_stock_streaming(glimmer_with_vision, device="cpu")
    assert type(streamed).__name__ == "MuseGlimmerForConditionalGeneration"
    # kept, exactly as load_cpu keeps it (every vision-capable architecture
    # is in the bench's default tree, not just gemma-4's)
    assert streamed.model.vision_tower is not None
    assert streamed.model.vision_adapter is not None
    assert streamed.model.vision_projection is not None
    _assert_same_tree_and_forward(streamed, stock)


def test_streamed_loader_accepts_a_revisioned_hub_style_call(qwen3_untied):
    """The bench calls it as it calls load_cpu: (model_id, revision). A local
    dir ignores the revision, same as snapshot_dir does."""
    streamed = arms.load_stock_streaming(qwen3_untied, "deadbeef", device="cpu")
    assert isinstance(streamed, torch.nn.Module)


def test_streamed_loader_refuses_an_fp8_checkpoint_with_the_one_line(tmp_path):
    """An FP8 checkpoint is not this loader's: refused before any tensor is
    read, with the one line every door prints (codec/pack.FP8_REFUSAL) —
    off config.json's quantization_config, and off a float8 plane in the
    shard header when config.json says nothing."""
    sys.path.insert(0, str(ROOT / "tests"))
    from fixtures import fake_checkpoint

    from drinkme.codec.pack import FP8_REFUSAL, UnsupportedCheckpoint

    d = fake_checkpoint(str(tmp_path / "fp8"),
                        quantization_config={"quant_method": "fp8", "fmt": "e4m3",
                                             "weight_block_size": [128, 128]})
    with pytest.raises(UnsupportedCheckpoint) as ei:
        arms.load_stock_streaming(d, device="cpu")
    assert str(ei.value) == f"{d}: {FP8_REFUSAL}"
    d2 = fake_checkpoint(str(tmp_path / "plane"),
                         tensors={"model.layers.0.mlp.up_proj.weight": ("F8_E4M3", [64, 64]),
                                  "model.layers.0.mlp.up_proj.weight_scale_inv": ("BF16", [1, 1])})
    with pytest.raises(UnsupportedCheckpoint) as ei:
        arms.load_stock_streaming(d2, device="cpu")
    assert str(ei.value) == f"{d2}: {FP8_REFUSAL}"


def test_streamed_loader_never_calls_from_pretrained(qwen3_untied, monkeypatch):
    """The point of the loader: no transformers weight-loading path at all —
    from_pretrained is what has the 2x transient."""
    import transformers

    def boom(*a, **k):
        raise AssertionError("from_pretrained must not run")

    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", boom)
    monkeypatch.setattr(transformers.AutoModelForImageTextToText, "from_pretrained", boom)
    arms.load_stock_streaming(qwen3_untied, device="cpu")


# --------------------------------------------------------- bench's plan --


def _hw(**kw):
    base = dict(device_class="strix-halo", memory_gb=133.144, memory_kind="unified", budget_gb=133.144,
                memory_bytes=133_144_000_000, budget_bytes=133_144_000_000,
                cpu_info="Strix Halo", gpu_info="amdgpu", evidence=["fixture"],
                device_source="cpu-brand")
    base.update(kw)
    return Hardware(**base)


def test_bench_fit_check_strix_halo_menu_under_the_streaming_loader():
    hw = _hw()
    for name in ("gemma-4-31B-it", "Qwen3.8-27B", "Muse-Glimmer-30B"):
        assert bench._stock_fit_ok(hw, _bf16_gb(name), "stream", None), name
    assert not bench._stock_fit_ok(hw, _bf16_gb("Qwen2.5-72B"), "stream", None)
    # the from_pretrained loader keeps its own (2x) verdict
    assert not bench._stock_fit_ok(hw, _bf16_gb("gemma-4-31B-it"), "from_pretrained", None)
    assert bench._stock_fit_ok(hw, _bf16_gb("Qwen3.8-27B"), "from_pretrained", None)


def test_bench_fit_check_rejects_an_unknown_loader_name():
    with pytest.raises(ValueError, match="stock loader"):
        bench._stock_fit_ok(_hw(), 1.0, "mmap", None)


def test_dry_run_plans_three_arms_for_gemma_and_a_fit_point_for_the_72b(monkeypatch):
    monkeypatch.setattr(bench, "detect", _hw)
    monkeypatch.setattr(bench, "cached_snapshot_dir", lambda repo, rev: None)  # nothing cached
    for name in ("gemma-4-31B-it", "Qwen3.8-27B", "Muse-Glimmer-30B"):
        plan = bench.dry_run(model_name=name)
        (p,) = plan["points"]
        assert p["arms"] == ["stock", "twin", "compressed"], name
        assert p["stockArmPredictedToFit"] is True
        assert p["stockLoader"] == "stream"
        assert p["stockArmChargeGB"] == pytest.approx(_bf16_gb(name) + STAGING_SHARD_BYTES / GB, abs=0.01)
        assert p["largestTensorGB"] is None and p["largestTensorBytes"] is None  # not cached: the 5 GiB fallback
        # hw.memory_gb is decimal GB — the fixture's
        # 133.144 (124 GiB physical) passes through unconverted, no GiB->GB
        # step left at this edge.
        assert p["stockArmCeilingGB"] == pytest.approx(133.144, abs=0.01)
    plan = bench.dry_run(model_name="Qwen2.5-72B")
    (p,) = plan["points"]
    # the 72B's twin (one 145.41 GB bf16 copy) does not fit 133.14 GB either:
    # compressed only (tests/test_stream_compressed_arms.py has the arithmetic)
    assert p["arms"] == ["compressed"] and p["stockArmPredictedToFit"] is False
    assert p["twinArmPredictedToFit"] is False and p["compressedArmPredictedToFit"] is True
    assert p["stockLoader"] == "stream"


def test_dry_run_with_the_from_pretrained_loader_shows_the_2x_plan(monkeypatch):
    monkeypatch.setattr(bench, "detect", _hw)
    monkeypatch.setattr(bench, "cached_snapshot_dir", lambda repo, rev: None)
    plan = bench.dry_run(model_name="gemma-4-31B-it", stock_loader="from_pretrained")
    (p,) = plan["points"]
    assert p["arms"] == ["twin", "compressed"] and p["stockLoader"] == "from_pretrained"
    # gemma's checkpoint is 62.546 decimal GB (suggest.sizes)
    assert p["stockArmChargeGB"] == pytest.approx(2 * 62.546, abs=1e-3)
    assert p["largestTensorGB"] is None


def test_dry_run_charges_the_real_largest_tensor_when_the_snapshot_is_cached(monkeypatch, qwen3_untied,
                                                                            checkpoint_table):
    from drinkme import suggest as suggest_mod
    from drinkme.suggest import Model

    monkeypatch.setattr(bench, "detect", _hw)
    monkeypatch.setattr(bench, "cached_snapshot_dir", lambda repo, rev: qwen3_untied)
    m = Model("Toy", "toy/toy", None)
    checkpoint_table["toy/toy"] = 58_250_000_000
    monkeypatch.setattr(suggest_mod, "MODELS", [m])
    plan = bench.dry_run(model_name="Toy")
    (p,) = plan["points"]
    assert p["largestTensorBytes"] == 128 * 64 * 2  # the toy's MLP gate/up, exact
    assert p["largestTensorGB"] == 0.0  # 16 KB rounds to nothing at the record's 3 places
    assert p["stockArmChargeGB"] == pytest.approx(58.25 + 128 * 64 * 2 / GB, abs=1e-3)


def test_run_point_hands_the_loader_and_tensor_term_to_run_arms(linux_host, monkeypatch):
    """bench -> run_arms: the loader name and the one-tensor term ride along
    so arms' live guard charges the same thing bench's fit check did."""
    from tests.test_bench_stock_fit_and_device_refusal import _fake_raw

    from drinkme.suggest import Model, Suggestion

    m = Model("gemma-4-31B-it", "google/gemma-4-31B-it", "842da379", gated=True)
    monkeypatch.setattr(bench, "detect", _hw)
    monkeypatch.setattr(bench, "gemma_gate_open", lambda: False)
    monkeypatch.setattr(bench, "cached_snapshot_dir", lambda repo, rev: None)
    monkeypatch.setattr(bench, "suggest", lambda *a, **k: Suggestion(
        ratio=m, ratio_extra=[], fit=None, fit_knife_edge=False, fit_honest_negative=False,
        notes=[]))
    captured = {}

    def fake_run_arms(repo, rev, prompt, **kw):
        captured.update(kw)
        return _fake_raw(stock=True)

    monkeypatch.setattr(arms, "run_arms", fake_run_arms)
    r = bench.run(no_gemma=True)
    assert captured["run_stock"] is True  # gemma is a ratio point on the Strix Halo box now
    assert captured["stock_loader"] == "stream"
    assert captured["stock_tensor_bytes"] is None
    assert r["stock"]["outcome"] == "measured"

    captured.clear()
    r = bench.run(no_gemma=True, stock_loader="from_pretrained")
    assert captured["run_stock"] is False  # the 2x path still routes it to a fit point
    assert captured["stock_loader"] == "from_pretrained"


def test_cli_stock_loader_flag_reaches_dry_run(monkeypatch, capsys):
    from drinkme.cli import main

    seen = {}

    def fake_dry_run(model_name=None, runtime=None, stock_loader="stream"):
        seen["loader"] = stock_loader
        return {"points": []}

    monkeypatch.setattr(bench, "dry_run", fake_dry_run)
    assert main(["bench", "--dry-run", "--stock-loader", "from_pretrained"]) == 0
    assert seen["loader"] == "from_pretrained"
    assert main(["bench", "--dry-run"]) == 0
    assert seen["loader"] == "stream"
    with pytest.raises(SystemExit):
        main(["bench", "--dry-run", "--stock-loader", "mmap"])


# ------------------------------------------- run_arms's pass 1, statically --


def _run_stock_branch() -> ast.If:
    tree = ast.parse((ROOT / "src" / "drinkme" / "arms.py").read_text())
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "run_arms")
    for node in ast.walk(fn):
        if isinstance(node, ast.If) and isinstance(node.test, ast.Name) \
                and node.test.id == "run_stock":
            return node
    raise AssertionError("run_arms has no `if run_stock:` branch")


def test_pass_1_dispatches_on_the_loader_name():
    """The streaming loader is pass 1's default; from_pretrained's
    device_map path is the selectable alternative; neither leaks into the
    fit-point branch (pass 2/3 keep load_cpu's plain CPU path: they swap
    Linears on the CPU tree first)."""
    node = _run_stock_branch()
    body = "\n".join(ast.unparse(n) for n in node.body)
    orelse = "\n".join(ast.unparse(n) for n in node.orelse)
    assert "load_stock_streaming(" in body and "device_map" in body
    assert "load_stock_streaming(" not in orelse and "device_map" not in orelse


def test_run_arms_stock_pass_uses_the_streaming_loader_and_survives_its_oom(monkeypatch):
    """run_arms with the device-touching pieces stubbed (the shape of
    test_bench_units_and_arm_outcomes' failed_load test): pass 1 calls
    load_stock_streaming, and an OOM out of it is still `failed_load`."""
    import types

    class _Ids:
        shape = (1, 4)

        def cuda(self):
            return self

    class _Tok:
        def __call__(self, text, **kw):
            if kw.get("return_tensors"):
                return types.SimpleNamespace(input_ids=_Ids())
            return types.SimpleNamespace(input_ids=[1, 2, 3])

    monkeypatch.setitem(sys.modules, "transformers", types.SimpleNamespace(
        AutoTokenizer=types.SimpleNamespace(from_pretrained=lambda *a, **k: _Tok())))
    # the one source every arm loads from, resolved without the hub
    # (tests/test_bench_one_source.py has the resolution itself)
    monkeypatch.setattr(arms, "resolve_source",
                        lambda repo, rev, pack_dir: (tempfile.gettempdir(), "deadbeef"))
    monkeypatch.setattr(arms.torch, "tensor", lambda *a, **k: _Ids())
    monkeypatch.setattr(arms.torch.cuda, "get_device_name", lambda i: "test-gpu")
    monkeypatch.setattr(arms, "measure_bandwidth", lambda: {
        "probe_bytes": GIB, "read_bytes_s": 200 * GB, "copy_bytes_s": 180 * GB, "device": "cuda"})
    monkeypatch.setattr(arms, "free_all", lambda: None)
    monkeypatch.setattr(arms, "live_available_bytes", lambda kind: 100 * GB)

    class _Model:
        def cuda(self):
            return self

        def parameters(self):  # pass 1 sizes the ceiling off the loaded tree
            return iter([types.SimpleNamespace(numel=lambda: 1000)])

    guard = {}

    def fake_guard(model_id, bf16_bytes, memory_kind, available_bytes, **kw):
        guard.update(kw)

    monkeypatch.setattr(arms, "refuse_if_stock_wont_fit", fake_guard)
    calls = []

    def load_stock_streaming(model_id, revision=None, device="cuda", snap=None):
        calls.append((model_id, revision, device, snap))
        return _Model()

    def load_cpu_must_not_take_device_map(model_id, revision=None, config=None, device_map=None,
                                          snap=None):
        assert device_map is None, "pass 1 must not run from_pretrained when streaming"
        return _Model()

    monkeypatch.setattr(arms, "load_stock_streaming", load_stock_streaming)
    monkeypatch.setattr(arms, "load_cpu", load_cpu_must_not_take_device_map)
    stat = {"name": "l", "shape": [1024, 1024], "numel": 1024 * 1024, "bits": 12 * 1024 * 1024,
            "bpw": 12.0, "format_version": 1, "codec": "radix", "profile": "sip", "dtype": None, "verified": True}
    # passes 2/3 stream too (tests/test_stream_compressed_arms.py): stubbed
    # here the way pass 1 is
    monkeypatch.setattr(arms, "load_twin_streaming",
                        lambda model_id, revision=None, device="cuda", snap=None, plan=None: (_Model(), []))
    monkeypatch.setattr(arms, "load_compressed_streaming",
                        lambda model_id, revision=None, device="cuda", snap=None: (_Model(), [dict(stat)]))
    monkeypatch.setattr(arms, "vram_bytes", lambda: 500_000_000)
    monkeypatch.setattr(arms, "timed_decode", lambda model, ids, **k: (None, [10.0, 10.0, 10.0]))
    monkeypatch.setattr(arms, "timed_prefill", lambda model, ids, **k: [500.0, 500.0, 500.0])
    monkeypatch.setattr(arms, "timed_ttft", lambda model, ids, **k: [0.05, 0.05, 0.05])

    r = arms.run_arms("toy/model", "rev1", "hi", run_stock=True, stock_memory_kind="unified",
                      stock_bf16_bytes=58.25 * GB, run_twin=True, stock_loader="stream",
                      stock_tensor_bytes=GEMMA_EMBED_BYTES)
    assert calls == [("toy/model", "rev1", "cuda", tempfile.gettempdir())]
    assert guard == {"direct": True, "tensor_bytes": GEMMA_EMBED_BYTES}
    assert r["stock_outcome"] == "measured" and r["stock_loader"] == "stream"

    def oom(model_id, revision=None, device="cuda", snap=None):
        raise RuntimeError("HIP out of memory. Tried to allocate 2.82 GiB")

    monkeypatch.setattr(arms, "load_stock_streaming", oom)
    r = arms.run_arms("toy/model", "rev1", "hi", run_stock=True, stock_memory_kind="unified",
                      stock_bf16_bytes=58.25 * GB, run_twin=True, stock_loader="stream")
    assert r["stock_outcome"] == "failed_load" and "out of memory" in r["stock_error"]
    assert r["stock_loader"] == "stream"
    assert r["twin_outcome"] == "measured" and r["compressed_decode_tok_s"] == 10.0

    # the from_pretrained loader is still selectable, and charges its own 2x
    guard.clear()
    monkeypatch.setattr(arms, "load_stock_streaming",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("wrong loader")))

    def load_cpu(model_id, revision=None, config=None, device_map=None, snap=None):
        calls.append(("load_cpu", device_map))
        return _Model()

    monkeypatch.setattr(arms, "load_cpu", load_cpu)
    r = arms.run_arms("toy/model", "rev1", "hi", run_stock=True, stock_memory_kind="unified",
                      stock_bf16_bytes=15 * GB, run_twin=True, stock_loader="from_pretrained")
    assert ("load_cpu", "cuda") in calls
    assert guard == {"direct": False, "tensor_bytes": None}
    assert r["stock_outcome"] == "measured" and r["stock_loader"] == "from_pretrained"
