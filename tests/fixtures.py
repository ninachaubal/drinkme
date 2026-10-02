"""Shared bf16 bit-pattern generators for the pack test suite.

A private `rand_bf16_bits` whose exponent draw puts the "weights" at
~1e34-1e38 is fine for a bit-exact round trip and inf/NaN for anything
numeric (a matmul, an allclose): a test that runs a numeric gate on top of
a round-trip overflows silently, and it looks like a kernel bug. One
fixture, calibrated to real-tensor exponents, plus the radix pack dict
over it. No private copies: fold any
new `bf16_bits` def into these instead.
"""

import numpy as np


def realistic_bf16_bits(r, c, seed, spread=False):
    """Bit patterns that are safe to put through a matmul — magnitudes land
    where real trained weights do (~N(0, 0.02) after bf16 rounding): the
    exponent draw into bits 8..14 is calibrated to real-tensor exponents.

    spread=False: exponent in [60, 64) — four exponents, a tight tensor.
    spread=True: exponent in [55, 66) — eleven exponents, so every tier of
    every radix profile is in use (the second-tier and terminal streams
    are exercised, not just the first palette).

    Used by: test_swap_prefill_dense.py, test_codec_bias_rounding.py,
    test_radix_pack.py, test_radix_schedule.py, radix_dict below.
    """
    rng = np.random.default_rng(seed)
    sign = rng.integers(0, 2, (r, c), dtype=np.uint16)
    if spread:
        exp = rng.integers(55, 66, (r, c), dtype=np.uint16)  # 11 octaves
    else:
        exp = rng.integers(60, 64, (r, c), dtype=np.uint16)  # 4 octaves
    mant = rng.integers(0, 128, (r, c), dtype=np.uint16)
    return (sign << 15) | (exp << 8) | mant


def radix_dict(r, c, seed, spread=False, profile="sip"):
    """A radix pack tensor dict (codec/radix_pack.pack_array_radix, the numpy
    encoder, deterministic) over realistic_bf16_bits — the container tests'
    unit of currency for the pack container: something save_pack_dir / PackWriter
    can write and iter_pack_dir hand back. Shapes stay tiny; the 4-11
    distinct exponents of realistic_bf16_bits compress at every profile, so
    the dict is never the raw fallback (asserted).

    Used by: test_pack.py, test_pack_verify_manifest.py,
    test_pack_writer_transactional.py, test_serving_pack_iter.py.
    """
    from drinkme.codec import radix_pack as rp

    p = rp.pack_array_radix(realistic_bf16_bits(r, c, seed, spread=spread), profile, encoder="numpy")
    assert p is not None, "the fixture's bits must compress"
    return p


def word_tokenizer(vocab_size: int):
    """A word-level toy tokenizer with a chat template, saved beside toy
    checkpoints (test_stream_compressed_arms, test_glimmer_toy)."""
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast

    words = ["hello", "world", "the", "quick", "brown", "fox", "jumps", "over",
             "lazy", "dog", "alpha", "beta", "gamma", "delta", "epsilon",
             "user", "assistant", "system", ":"]
    vocab = {"<unk>": 0, "<pad>": 1, "</s>": 2}
    for w in words:
        vocab[w] = len(vocab)
    while len(vocab) < vocab_size:
        vocab[f"tok{len(vocab)}"] = len(vocab)
    backend = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
    backend.pre_tokenizer = Whitespace()
    tok = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="<unk>",
                                  pad_token="<pad>", eos_token="</s>")
    tok.chat_template = ("{% for m in messages %}{{ m['role'] }} : {{ m['content'] }} "
                         "{% endfor %}{% if add_generation_prompt %}assistant :{% endif %}")
    return tok


# safetensors dtype string -> bytes per element, for the fake checkpoints
# below (header-only checkpoints: the doors read config.json and the shard
# headers, never a weight byte)
_SF_NBYTES = {"BF16": 2, "F16": 2, "F32": 4, "F8_E4M3": 1, "F8_E5M2": 1, "I8": 1, "I32": 4}


def write_safetensors(path: str, tensors: dict[str, tuple[str, list[int]]]) -> None:
    """A minimal, valid safetensors file — the 8-byte header length, the JSON
    header, then zero bytes for every tensor — from {name: (dtype, shape)}.
    No torch, no safetensors: json + struct only, so the torch-free door
    tests can build an FP8-looking checkpoint without either."""
    import json
    import math
    import struct

    header, offset = {}, 0
    for name, (dtype, shape) in tensors.items():
        n = _SF_NBYTES[dtype] * math.prod(shape)
        header[name] = {"dtype": dtype, "shape": list(shape), "data_offsets": [offset, offset + n]}
        offset += n
    blob = json.dumps(header).encode()
    blob += b" " * (-len(blob) % 8)  # the header is padded to 8 bytes
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(blob)))
        f.write(blob)
        f.write(bytes(offset))


def fake_checkpoint(model_dir: str, tensors: dict[str, tuple[str, list[int]]] | None = None,
                    quantization_config: dict | None = None, **config) -> str:
    """A checkpoint DIRECTORY the doors will look at and nothing else will:
    config.json (a llama-shaped stub plus `quantization_config` when given)
    and one model.safetensors whose HEADER names `tensors` ({name: (dtype,
    shape)}; a bf16 pair by default). Returns `model_dir`."""
    import json
    import os

    os.makedirs(model_dir, exist_ok=True)
    cfg = {"architectures": ["LlamaForCausalLM"], "model_type": "llama",
           "hidden_size": 64, "intermediate_size": 64, "num_hidden_layers": 1,
           "num_attention_heads": 4, "num_key_value_heads": 2, "vocab_size": 64,
           "tie_word_embeddings": False, **config}
    if quantization_config is not None:
        cfg["quantization_config"] = quantization_config
    with open(os.path.join(model_dir, "config.json"), "w") as f:
        json.dump(cfg, f)
    write_safetensors(os.path.join(model_dir, "model.safetensors"),
                      tensors or {"model.embed_tokens.weight": ("BF16", [64, 64]),
                                  "model.layers.0.mlp.up_proj.weight": ("BF16", [64, 64])})
    return model_dir
