"""pack_model's `residentBytes` — npz bytes plus every tensor the
loader streams RAW (untied embed_tokens, norms, k/v below the eligibility
bar) — CPU only, no GPU, no real weights loaded.

The toy is a genuine 2-layer LlamaForCausalLM (hidden 1024 so its q/o/mlp
Linears clear codec eligibility; kv_heads=2 keeps k/v at 512 wide, so they
stay RAW alongside embed_tokens/lm_head/norms — the same toy shape
test_serving_engines.py already uses). `tie_word_embeddings` is pulled from
a real cached Qwen3 config (untied: Qwen3-8B;
tied: Qwen3-1.7B) so the toy's two cases are grounded in an architecture
that really ships that way, and the tokenizer saved alongside the toy
checkpoint is the real cached one for that repo — both offline, skipped
cleanly if the cache does not have them.

The raw-set claim is checked against the ACTUAL loaded model
(serving.engines.load_compressed), not against a second copy of pack.py's
own predicate: every real Parameter the loader attached OUTSIDE a
CompressedLinear is exactly what "streamed raw" means, by construction.
"""

from __future__ import annotations

import glob
import json
import os

import pytest
import torch

from drinkme.codec.pack import pack_model, resident_bytes
from drinkme.codec.swap import CompressedLinear
from drinkme.serving.engines import load_compressed

QWEN8B = ("Qwen/Qwen3-8B", "b968826d9c46dd6066d109eabc6255188de91218")     # untied
QWEN17B = ("Qwen/Qwen3-1.7B", "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e")  # tied


def _cached_tie_word_embeddings(repo: str, revision: str) -> bool:
    from huggingface_hub import try_to_load_from_cache

    hit = try_to_load_from_cache(repo, "config.json", revision=revision)
    if not isinstance(hit, str):
        pytest.skip(f"{repo}@{revision[:12]} is not in the local HF cache "
                    "(config.json not found) — nothing to ground the toy in")
    with open(hit) as f:
        return json.load(f)["tie_word_embeddings"]


def _cached_tokenizer(repo: str, revision: str):
    from huggingface_hub import try_to_load_from_cache

    hit = try_to_load_from_cache(repo, "tokenizer_config.json", revision=revision)
    if not isinstance(hit, str):
        pytest.skip(f"{repo}@{revision[:12]} is not in the local HF cache "
                    "(tokenizer_config.json not found)")
    os.environ["HF_HUB_OFFLINE"] = "1"
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(repo, revision=revision, local_files_only=True)


def _build_and_pack(tmp_path, repo_rev: tuple[str, str], tied: bool) -> tuple[str, str]:
    from transformers import LlamaConfig, LlamaForCausalLM

    repo, revision = repo_rev
    real_tied = _cached_tie_word_embeddings(repo, revision)
    assert real_tied is tied, (
        f"{repo}@{revision[:12]}'s tie_word_embeddings drifted from {tied} — "
        "this toy's whole point is to test the case that model represents")
    tok = _cached_tokenizer(repo, revision)

    model_dir, pack_dir = str(tmp_path / "model"), str(tmp_path / "pack")
    cfg = LlamaConfig(vocab_size=64, hidden_size=1024, intermediate_size=1024,
                      num_hidden_layers=2, num_attention_heads=8,
                      num_key_value_heads=2, max_position_embeddings=256,
                      tie_word_embeddings=tied, attention_bias=False,
                      mlp_bias=False, eos_token_id=2, pad_token_id=1)
    torch.manual_seed(0)
    model = LlamaForCausalLM(cfg).to(torch.bfloat16).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    tok.save_pretrained(model_dir)
    model.save_pretrained(model_dir, safe_serialization=True)
    pack_model(model_dir, None, pack_dir, progress=lambda *_: None, mtp=False)
    return model_dir, pack_dir


def _npz_sum(pack_dir: str) -> int:
    return sum(os.path.getsize(f) for f in glob.glob(os.path.join(pack_dir, "*.npz")))


def _packed_resident_bytes(model) -> int:
    """What the loader actually holds for the packed tensors: the runtime
    dict to_device produced, measured by the codec's own resident_bytes —
    NOT the npz size (the runtime dict carries widened/derived sidecars the
    file does not; ~0.75% over disk on the packs measured)."""
    return sum(resident_bytes(mod.p) for mod in model.modules()
               if isinstance(mod, CompressedLinear))


def _materialized_raw_bytes(model) -> int:
    """Every real (non-meta) byte the loader put somewhere OTHER than a
    CompressedLinear.p — i.e. exactly what "streamed raw" means, read off
    the model load_compressed actually produced. Tied parameters (a shared
    embed_tokens/lm_head) are the SAME Python object under two module
    paths, so dedup by identity or a tied pair double-counts."""
    seen: set[int] = set()
    total = 0
    for mod in model.modules():
        if isinstance(mod, CompressedLinear):
            if mod.bias is not None and id(mod.bias) not in seen:
                seen.add(id(mod.bias))
                total += mod.bias.numel() * mod.bias.element_size()
            continue  # .p is the pack itself, not a raw-streamed tensor
        for p in mod.parameters(recurse=False):
            if id(p) in seen:
                continue
            seen.add(id(p))
            total += p.numel() * p.element_size()
    return total


def test_residentbytes_untied_equals_npz_sum_plus_materialized_raw(tmp_path):
    model_dir, pack_dir = _build_and_pack(tmp_path, QWEN8B, tied=False)
    meta = json.load(open(os.path.join(pack_dir, "meta.json")))

    eng = load_compressed(model_dir, None, pack_dir, device="cpu")
    expected = _packed_resident_bytes(eng.model) + _materialized_raw_bytes(eng.model)

    assert meta["residentBytes"] == expected
    # the fix is real: raw-streamed bytes are a nonzero share (embed_tokens,
    # lm_head, k/v, norms all ride raw in this toy)
    assert meta["residentBytes"] > _npz_sum(pack_dir)


def test_residentbytes_tied_does_not_double_count_embed_tokens(tmp_path):
    model_dir, pack_dir = _build_and_pack(tmp_path, QWEN17B, tied=True)
    meta = json.load(open(os.path.join(pack_dir, "meta.json")))

    from safetensors import safe_open

    shard = sorted(glob.glob(os.path.join(model_dir, "*.safetensors")))[0]
    with safe_open(shard, framework="pt") as sf:
        assert "lm_head.weight" not in sf.keys()  # HF's own tied-save dedup

    eng = load_compressed(model_dir, None, pack_dir, device="cpu")
    assert eng.model.lm_head.weight is eng.model.model.embed_tokens.weight
    expected = _packed_resident_bytes(eng.model) + _materialized_raw_bytes(eng.model)

    assert meta["residentBytes"] == expected


def test_residentbytes_tied_vs_untied_differ_by_exactly_one_embed_table(tmp_path):
    """The untied toy carries embed_tokens AND a separate (also raw, at this
    toy's tiny vocab) lm_head; tied drops lm_head from the checkpoint
    entirely (HF's own save-time dedup) and reties it to embed_tokens at
    load — one fewer raw tensor, not zero difference and not two."""
    _, untied_pack = _build_and_pack(tmp_path / "untied", QWEN8B, tied=False)
    _, tied_pack = _build_and_pack(tmp_path / "tied", QWEN17B, tied=True)
    untied_meta = json.load(open(os.path.join(untied_pack, "meta.json")))
    tied_meta = json.load(open(os.path.join(tied_pack, "meta.json")))

    embed_bytes = 64 * 1024 * 2  # bf16, vocab=64 x hidden=1024, this toy's shape
    assert untied_meta["residentBytes"] - tied_meta["residentBytes"] == embed_bytes
