"""Toys the hot-loop equivalence tests and bench/hotloop_micro.py share.

Two things live here, both built offline in a second or two, no downloads:

  * `json_tokenizer()` — a real BYTE-LEVEL BPE tokenizer (the Qwen/GPT family
    shape) whose vocabulary can spell JSON, English, CJK, and an emoji ZWJ
    sequence, and whose 256 single-byte tokens let a multi-byte character be
    split across tokens exactly the way a real model splits one. That last
    property is the whole reason the toy in test_serving_engines.py (a
    WordLevel tokenizer, which joins with spaces) cannot serve here: the
    suffix-window detok has to be proven against a decoder whose output is
    concatenative, and against one whose output is not.
  * `toy_engine()` — a tiny genuine LlamaForCausalLM behind a real HFEngine,
    so a test can drive the actual generate() loop rather than a mock of it.
"""

from __future__ import annotations

import functools


def _byte_map() -> dict[int, str]:
    """GPT-2's bytes-to-unicode: the printable-safe alphabet a ByteLevel BPE
    stores its vocabulary in."""
    bs = (list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1))
          + list(range(ord("®"), ord("ÿ") + 1)))
    cs = list(bs)
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


BYTE = _byte_map()


def bl(s: str) -> str:
    """A python string as its byte-level vocabulary spelling."""
    return "".join(BYTE[b] for b in s.encode("utf-8"))


PIECES = [
    # JSON punctuation and whitespace
    "{", "}", "[", "]", '"', ":", ",", " ", "\n", "  ",
    # keys and values the test schemas use
    "name", "count", "tags", "text", "pick", "tag", "alpha", "beta", "yes",
    "true", "false", "null", "hello", " there", "AAAA", "12", "123", "-",
    "\\n", "\\u0041",
    # text that is not ASCII: whole-character tokens AND the pieces a model
    # would use to spell them one byte at a time (byte-level BPE does both)
    "é", "中", "中文", "文字", "🙂", "🕵", "🕵️‍♀️", "👍", "👍🏽",
]


@functools.lru_cache(maxsize=None)
def json_tokenizer():
    """A PreTrainedTokenizerFast over a hand-built byte-level BPE."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    vocab: dict[str, int] = {}
    for ch in sorted(BYTE.values()):  # the 256 single-byte tokens come first
        vocab[ch] = len(vocab)
    for piece in PIECES:
        spelled = bl(piece)
        if spelled not in vocab:
            vocab[spelled] = len(vocab)
    vocab["<|end|>"] = len(vocab)
    backend = Tokenizer(models.BPE(vocab, []))  # no merges: encoding falls
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()      # back to single bytes, which
    tok = PreTrainedTokenizerFast(              # is all these tests need
        tokenizer_object=backend, eos_token="<|end|>", pad_token="<|end|>")
    tok.chat_template = (
        "{% for m in messages %}{{ m['role'] }}: {{ m['content'] }}\n"
        "{% endfor %}{% if add_generation_prompt %}assistant:{% endif %}")
    return tok


def piece_id(tok, piece: str) -> int:
    """The id of one vocabulary piece, by its plain-text spelling."""
    return tok.convert_tokens_to_ids(bl(piece))


def toy_engine(seed: int = 0, ctx: int = 2048, hidden: int = 64,
               tokenizer=None, kv_floor: int = 256):
    """A real HFEngine over a tiny Llama. Deterministic weights; the outputs
    are meaningless, which is the point — every test here compares two paths
    against each other, never against a fixed transcript."""
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM

    from drinkme.serving.engines import HFEngine

    tok = tokenizer or json_tokenizer()
    cfg = LlamaConfig(vocab_size=len(tok), hidden_size=hidden,
                      intermediate_size=hidden * 2, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2,
                      max_position_embeddings=ctx, tie_word_embeddings=False,
                      eos_token_id=tok.eos_token_id, pad_token_id=tok.eos_token_id)
    torch.manual_seed(seed)
    model = LlamaForCausalLM(cfg).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    eng = HFEngine(model, tok, model_id="toy", arm="toy", meta={}, ctx=ctx)
    # the slot allocates its full width (decode attends over the live window,
    # serving/kvcache.py): keep the toy's allocation near what it uses
    eng.KV_FLOOR = kv_floor
    return eng
