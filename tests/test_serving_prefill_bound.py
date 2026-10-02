"""The prefill bound, at the engine — CPU only.

With DRINKME_PREFILL_CHUNK=C the engine never hands the trunk (or the MTP
head) more than C tokens in one forward: a recording hook on the decoder,
through the serial, n-gram and MTP branches and a prefix-cache extend, on a
dense toy (Qwen3) and a hybrid DeltaNet toy (qwen3_5, the 27B's layer
alternation). The bound is what keeps every prefill GPU dispatch short
(serving/prefill.py's docstring has the measurement).

This file imports nothing chunked prefill added, so on a tree without it it
collects and FAILS on the recorded row counts — the whole prompt goes in one
forward there — rather than at import. `test_zero_is_the_whole_prompt_in_one_forward` is
the A/B knob's contract and passes on both trees by design.
tests/test_serving_prefill_chunk.py has the numerics.
"""

import os

import pytest
import torch

from drinkme.serving.engine import GenerationRequest, SampleParams, complete
from drinkme.serving.engines import HFEngine

from test_serving_mtp import _torch_chunk_on_cpu, toy  # noqa: F401 — fixtures by name
from test_serving_prefix_slots import _tokenizer

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")

C = 8  # tokens per chunk on the toys: every prompt below is several chunks
LONG = ("the quick brown fox jumps over the lazy dog alpha beta gamma delta "
        "epsilon zeta hello world ") * 4


@pytest.fixture(scope="module")
def dense():
    """(model, tokenizer): a 2-layer Qwen3 — all full attention, GQA 2:1."""
    from transformers import Qwen3Config, Qwen3ForCausalLM

    cfg = Qwen3Config(vocab_size=64, hidden_size=64, intermediate_size=128,
                      num_hidden_layers=2, num_attention_heads=4,
                      num_key_value_heads=2, head_dim=16,
                      max_position_embeddings=512, tie_word_embeddings=False,
                      eos_token_id=None, pad_token_id=None)
    torch.manual_seed(0)
    model = Qwen3ForCausalLM(cfg).eval().to(torch.float32)
    for p in model.parameters():
        p.requires_grad_(False)
    return model, _tokenizer()


@pytest.fixture
def hybrid(toy):  # noqa: F811
    """(model, tokenizer, head): test_serving_mtp's qwen3_5 toy with its head."""
    model, tok, head, _ = toy
    return model, tok, head


class env:
    def __init__(self, **kw):
        self.kw, self.old = kw, {}

    def __enter__(self):
        for k, v in self.kw.items():
            self.old[k] = os.environ.get(k)
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = str(v)

    def __exit__(self, *a):
        for k, v in self.old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def engine(model, tok, chunk, head=None, reuse=True, ckpts=None):
    """`ckpts` is DRINKME_CTX_CHECKPOINTS for the construction (None = the
    default): a context checkpoint also ends a prefill span
    (serving/ctx_checkpoints.py)."""
    with env(DRINKME_PREFILL_CHUNK=chunk,
             DRINKME_PREFIX_SLOTS=None if reuse else "0",
             DRINKME_CTX_CHECKPOINTS=ckpts):
        return HFEngine(model, tok, model_id="toy", arm="test", meta={}, ctx=512,
                        mtp_head=head)


def ask(eng, msgs, spec, n=12):
    with env(DRINKME_SPEC=spec):
        return complete(eng, GenerationRequest(msgs, SampleParams(temperature=0.0,
                                                                  max_tokens=n)))


def user(text):
    return [{"role": "user", "content": text}]


class Rows:
    """Every row count the decoder (and the MTP head) is handed."""

    def __init__(self, model, head=None):
        self.trunk, self.head = [], []
        base = model.get_decoder()
        self.hooks = [base.register_forward_pre_hook(self._trunk, with_kwargs=True)]
        if head is not None:
            self.hooks.append(head.fc.register_forward_pre_hook(
                lambda m, a: self.head.append(a[0].shape[1])))

    def _trunk(self, mod, args, kwargs):
        ids = args[0] if args else kwargs.get("input_ids")
        if ids is None:
            ids = kwargs["inputs_embeds"]
        self.trunk.append(ids.shape[1])

    def __enter__(self):
        return self

    def __exit__(self, *a):
        for h in self.hooks:
            h.remove()


BRANCHES = [("dense", "off"), ("dense", "ngram"),
            ("hybrid", "off"), ("hybrid", "ngram"), ("hybrid", "auto")]


def _toy(request, name):
    t = request.getfixturevalue(name)
    model, tok = t[0], t[1]
    head = t[2] if name == "hybrid" else None
    return model, tok, head


# ------------------------------------------------------------ BOUNDED --

@pytest.mark.parametrize("name,spec", BRANCHES)
def test_the_engine_never_hands_the_model_more_than_a_chunk(request, name, spec):
    model, tok, head = _toy(request, name)
    eng = engine(model, tok, C, head=head if spec == "auto" else None)
    with Rows(model, head) as rows:
        res = ask(eng, user(LONG), spec)
    assert res.prompt_tokens > 4 * C  # the prompt really is several chunks
    assert max(rows.trunk) <= C, rows.trunk
    assert max(rows.head, default=0) <= C, rows.head
    if spec == "auto":
        assert rows.head, "the MTP head was never seeded"


def test_a_prefix_cache_extend_is_chunked_too(dense):
    model, tok = dense
    eng = engine(model, tok, C)
    hist = user(LONG)
    first = ask(eng, hist, "off")
    hist += [{"role": "assistant", "content": first.text}] + user(LONG)
    with Rows(model) as rows:
        second = ask(eng, hist, "off")
    assert second.cached_tokens > 0  # the extend, not a cold prefill
    assert second.prompt_tokens - second.cached_tokens > 2 * C
    assert max(rows.trunk) <= C, rows.trunk


def test_zero_is_the_whole_prompt_in_one_forward(dense):
    """DRINKME_PREFILL_CHUNK=0 keeps the unchunked path for A/B."""
    model, tok = dense
    eng = engine(model, tok, 0)
    with Rows(model) as rows:
        res = ask(eng, user(LONG), "off", n=3)
    assert rows.trunk[0] == res.prompt_tokens
    assert rows.trunk[1:] == [1] * (len(rows.trunk) - 1)
