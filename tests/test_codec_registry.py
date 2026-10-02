"""codec/registry.py — the capability table — and its first consumer, the
draft-vocab installation.

The table: complete over the known matrix (every (tensor_codec, layout) x op x
backend has an explicit True or False — "decided against" is written down,
not inferred from a missing key), unknown keys — a dict that names no kind
among them — answer False without raising, and the rows are not wishes:
every CPU dense-decode row that says True is backed by a real decode here.

The consult: make_projection asks the table before it builds anything. No
codec has a row-subset kernel on any backend, so every packed head gets
(None, why) instead of a KeyError from a projection's constructor,
install_draft_vocab keeps the
full-vocabulary head with one line on stderr (the documented
DRINKME_MTP_DRAFT_VOCAB fallback), and an explicit DRINKME_MTP_DEPTH=4 through the
real loader still starts — on the CPU here, on the device at the bottom.
"""

import os

import numpy as np
import pytest
import torch

from drinkme.codec import radix_pack as rp, registry
from drinkme.codec.registry import (KERNELS, CPU, DENSE, FORMATS, GEMV, MC,
                                    METAL, OPS, RADIX, RAW, ROW_SUBSET, TABLE, TRITON)
from drinkme.codec.swap import make_module
from drinkme.serving import draft_vocab, mtp
from drinkme.serving.engines import load_compressed
from test_serving_mtp import (_diet_cfg, _fill_random, _greedy, _msgs, _run,
                              WORDS, draft_vocab_env, mtp_depth)

# ------------------------------------------------------------- the table --


def test_the_table_is_complete_over_the_known_matrix():
    keys = {(f, l, op, b) for f, l in FORMATS for op in OPS for b in KERNELS}
    assert set(TABLE) == keys
    assert all(isinstance(v, bool) for v in TABLE.values())


def test_unknown_keys_are_unsupported_not_errors():
    assert registry.supported("window", 0, GEMV, TRITON) is False
    assert registry.supported(registry.UNKNOWN, 0, DENSE, TRITON) is False
    assert registry.supported(7, 0, DENSE, CPU) is False  # a number is not a kind
    assert registry.supported(RADIX, 7, DENSE, CPU) is False
    assert registry.supported(RADIX, 0, "gemm", TRITON) is False
    assert registry.supported(RADIX, 0, GEMV, "cuda") is False  # backend names are the table's
    assert "unknown combination" in registry.unsupported_note("window", 0, GEMV, TRITON)
    assert "unknown combination" in registry.unsupported_note(registry.UNKNOWN, 0, DENSE, TRITON)
    assert "unknown tensor kind" in registry.describe(registry.UNKNOWN, 0)
    assert "no codec named" in registry.describe(registry.UNKNOWN, 1)
    assert "unsupported" in registry.unsupported_note(RADIX, 0, ROW_SUBSET, TRITON)


def test_the_formats_are_the_two_kinds_of_a_pack():
    assert set(FORMATS) == {(RADIX, 0), (RAW, 0)}
    assert registry.of_pack({"codec": "radix"}) == (RADIX, 0)
    assert registry.of_pack({"codec": "raw"}) == (RAW, 0)
    # a dict naming no kind — including an FP8 pack's `dtype` scalar:
    # UNKNOWN, no rows (the loaders refuse it first, by the one line)
    assert registry.of_pack({"dtype": "fp8_e4m3"}) == (registry.UNKNOWN, 0)
    assert registry.of_pack({"format_version": 1, "layout": 2}) == (registry.UNKNOWN, 2)
    assert registry.of_pack({}) == (registry.UNKNOWN, 0)




def test_every_kind_serves_its_forward_arms_on_triton():
    """radix has the three arms (its gates: bench/radix_*_bitpin.py); the
    raw fallback is stock's F.linear, DENSE only; every kind decodes on the
    CPU."""
    for f, l in FORMATS:
        for op in ((DENSE,) if f == RAW else (DENSE, GEMV, MC)):
            assert registry.supported(f, l, op, TRITON), (f, l, op)
        assert registry.supported(f, l, DENSE, CPU), (f, l)


def test_no_kind_has_a_row_subset_kernel_anywhere():
    """The draft keeps the full head everywhere."""
    for f, l in FORMATS:
        for b in KERNELS:
            assert not registry.supported(f, l, ROW_SUBSET, b), (f, l, b)


def test_the_metal_rows_are_radix_and_raw_dense_only():
    """The radix tensor (RadixLinear: metal/gemv_radix + dense_radix, gated
    by bench/radix_bitpin_mlx.py on an M4) has dense + gemv rows;
    the raw fallback (engine_mlx.RawLinear, mlx's own matmul) has dense
    only, as on every kernel; no Metal row has a multi-column or
    row-subset kernel."""
    for f in (RADIX,):
        assert registry.supported(f, 0, GEMV, METAL) and registry.supported(f, 0, DENSE, METAL)
        assert not registry.supported(f, 0, MC, METAL)
    assert registry.supported(RAW, 0, DENSE, METAL)
    assert not registry.supported(RAW, 0, GEMV, METAL) and not registry.supported(RAW, 0, MC, METAL)
    for key, ok in TABLE.items():
        f, l, op, b = key
        if b == METAL and op in (MC, ROW_SUBSET):
            assert not ok, key


def test_refuse_unless_metal_dense_names_the_tensor_and_kind():
    """serving/engine_mlx.py's callsite, torch-free: the table's (tensor_codec,
    layout, DENSE, METAL) row decides. A radix or raw dict passes
    (RadixLinear / RawLinear serve them); a dict that names no kind — an
    FP8 pack's `dtype` scalar included — and a kind the table has never
    heard of are refused by name, with the registry's own note, never a
    bare KeyError."""
    for d in ({"codec": "radix", "format_version": 1}, {"codec": "raw", "format_version": 1}):
        registry.refuse_unless_metal_dense(d, "model.layers.0.mlp.down_proj")
    with pytest.raises(ValueError, match=r"no MLX path for model.layers.0.mlp.down_proj \(unknown tensor kind 'unknown' layout-0 \(no codec named.*codec/registry.py.*torch runtime"):
        registry.refuse_unless_metal_dense({"format_version": 1, "layout": 0}, "model.layers.0.mlp.down_proj")
    with pytest.raises(ValueError, match=r"no MLX path for .*unknown tensor kind 'unknown' layout-0"):
        registry.refuse_unless_metal_dense({"dtype": "fp8_e4m3", "format_version": 1, "layout": 0}, "model.layers.0.mlp.down_proj")
    with pytest.raises(ValueError, match=r"no MLX path for .*unknown tensor kind 'unknown' layout-1"):
        registry.refuse_unless_metal_dense({"codec": "window", "layout": 1}, "model.layers.0.mlp.down_proj")


def test_kernel_of_reads_the_device_type():
    assert registry.kernel_of("cpu") == CPU
    assert registry.kernel_of(torch.device("cpu")) == CPU
    assert registry.kernel_of("cuda") == TRITON
    assert registry.kernel_of(torch.device("cuda:0")) == TRITON


# ------------------------------------------------- the rows are not wishes --


def _bits(w):
    return w.view(torch.int16).numpy().view(np.uint16)


def _every_way(w):
    """(name, module on CPU) for the same weight packed every way a pack
    holds a bf16 tensor: the three profiles and the raw fallback
    — the cancellation construction (two columns of ones, the rest zero)."""
    u = _bits(w)
    out = []
    for name in ("sip", "balanced", "gulp", "raw"):
        raw = rp.raw_dict(u) if name == "raw" else rp.pack_array_radix(u, name, encoder="numpy")
        assert raw is not None
        mod = make_module(raw, None, "cpu")
        assert registry.of_pack(mod.p) == (RAW if name == "raw" else RADIX, 0), name
        out.append((name, mod))
    return out


@pytest.fixture(scope="module")
def cancellation_weight():
    w = torch.zeros(1024, 1024, dtype=torch.bfloat16)
    w[:, :2] = 1
    return w


def test_cpu_dense_rows_decode_for_real(cancellation_weight):
    for name, mod in _every_way(cancellation_weight):
        tensor_codec, layout = registry.of_pack(mod.p)
        assert registry.supported(tensor_codec, layout, DENSE, CPU)
        assert torch.equal(mod._cpu_weight(), cancellation_weight), name
        assert torch.equal(mod._dense_weight(), cancellation_weight), name




# ----------------------------------------------------------- the consult --




def test_every_packed_head_falls_back_and_the_table_is_the_reason(
        cancellation_weight, monkeypatch):
    """No backend has a row-subset kernel for any kind, so a packed head
    gets (None, why) on the CPU and — pointing kernel_of at triton —
    on the device too; it IS the table's answer, not the device's. A row
    flipped True without a projection class is a bug named as such, not a
    served fallback."""
    ids = torch.tensor([0, 2], dtype=torch.long)
    ways = _every_way(cancellation_weight)
    for name, mod in ways:
        proj, note = draft_vocab.make_projection(mod, ids)
        assert proj is None and "no cpu kernel" in note, (name, note)
    monkeypatch.setattr(registry, "kernel_of", lambda device: TRITON)
    for name, mod in ways:
        proj, note = draft_vocab.make_projection(mod, ids)
        assert proj is None and "no triton kernel" in note, (name, note)
    monkeypatch.setitem(registry.TABLE, (RADIX, 0, ROW_SUBSET, TRITON), True)
    with pytest.raises(NotImplementedError, match="no projection class"):
        draft_vocab.make_projection(ways[0][1], ids)


class _Head:
    """install_draft_vocab's view of an MTPHead: an lm_head and the slot."""

    def __init__(self, lm_head):
        self.lm_head = lm_head
        self.draft_proj = None


class _Model:
    def __init__(self, vocab):
        from transformers import LlamaConfig

        self.config = LlamaConfig(vocab_size=vocab, eos_token_id=2)


def test_install_keeps_the_full_head_with_one_line(cancellation_weight, capsys):
    """The documented fallback, at the installation seam: the head's
    draft_proj stays None (= the full-vocabulary head), one
    `[drinkme.mtp]` line says why, nothing raises."""
    for name, mod in _every_way(cancellation_weight):
        head = _Head(mod)
        with draft_vocab_env(300):
            mtp.install_draft_vocab(head, _Model(1024))
        assert head.draft_proj is None, name
        err = capsys.readouterr().err
        lines = [ln for ln in err.splitlines() if ln.startswith("[drinkme.mtp]")]
        assert len(lines) == 1, err
        assert "DRINKME_MTP_DRAFT_VOCAB is set but" in lines[0]
        assert "drafting over the full vocab" in lines[0]
        assert registry.describe(*registry.of_pack(mod.p)) in lines[0], lines[0]


# ------------------------------------------- explicit DRINKME_MTP_DEPTH=4 starts --


@pytest.fixture(scope="session")
def head_1k(tmp_path_factory):
    """test_serving_mtp's diet toy with a 1024-wide vocabulary, so its
    lm_head (1024 x 1024) is CODEC-ELIGIBLE and packs — the diet toy's
    96-row head stays raw and never reaches SubsetProjection. Packed twice:
    the sip profile (the default) and gulp.
    Returns (model_dir, {profile: pack_dir}, cfg)."""
    from safetensors.torch import load_file, save_file
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

    from drinkme.codec.pack import pack_model

    d = tmp_path_factory.mktemp("head1k")
    model_dir = str(d / "model")
    cfg = _diet_cfg()
    cfg.vocab_size = 1024
    torch.manual_seed(0)
    model = Qwen3_5ForCausalLM(cfg).to(torch.bfloat16).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    vocab = {"<unk>": 0, "<pad>": 1, "</s>": 2}
    for w in WORDS:
        vocab[w] = len(vocab)
    while len(vocab) < cfg.vocab_size:
        vocab[f"tok{len(vocab)}"] = len(vocab)
    kernel = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
    kernel.pre_tokenizer = Whitespace()
    tok = PreTrainedTokenizerFast(tokenizer_object=kernel, unk_token="<unk>",
                                  pad_token="<pad>")
    tok.chat_template = ("{% for m in messages %}{{ m['role'] }} : {{ m['content'] }} "
                         "{% endfor %}{% if add_generation_prompt %}assistant :{% endif %}")
    tok.save_pretrained(model_dir)
    model.save_pretrained(model_dir, safe_serialization=True)
    head = _fill_random(mtp.MTPHead(cfg, model), seed=3, scale=0.08 / 64)
    shard = os.path.join(model_dir, "model.safetensors")
    tensors = load_file(shard)
    tensors.update({f"mtp.{k}": v.to(torch.bfloat16).contiguous()
                    for k, v in head.state_dict().items()})
    save_file(tensors, shard, metadata={"format": "pt"})
    del tensors
    packs = {}
    for profile in ("sip", "gulp"):
        packs[profile] = str(d / f"pack-{profile}")
        pack_model(model_dir, None, packs[profile], progress=lambda *_: None,
                   compression_profile=profile)
    return model_dir, packs, cfg


def _lm_head_layout(pack_dir):
    from drinkme.codec.pack import iter_pack_dir

    for name, raw in iter_pack_dir(pack_dir):
        if name.lstrip(".") == "lm_head":
            return registry.of_pack(raw)
    raise AssertionError("the 1k toy's lm_head did not pack")


def test_the_1k_toy_packs_its_head_at_both_profiles(head_1k):
    _, packs, _ = head_1k
    for profile in ("sip", "gulp"):
        assert _lm_head_layout(packs[profile]) == (registry.RADIX, 0)


@pytest.mark.parametrize("profile", ["sip", "gulp"])
def test_explicit_mtp_with_a_draft_vocab_still_starts_on_cpu(head_1k, profile, capsys):
    """DRINKME_MTP_DEPTH=4 is the loud configuration: the loader propagates head
    failures instead of containing them. The engine comes up with its head,
    the head drafts over the full vocabulary (no row-subset kernel for a
    radix head, on any backend), and one line says so."""
    model_dir, packs, _ = head_1k
    with mtp_depth(4), draft_vocab_env(300):
        eng = load_compressed(model_dir, None, packs[profile], device="cpu", ctx=256)
    assert eng.mtp_head is not None and eng.mtp_head.draft_proj is None
    err = capsys.readouterr().err
    assert "drafting over the full vocab" in err
    assert registry.describe(registry.RADIX, 0) in err
    with mtp_depth(4):
        res, _ = _run(eng, _msgs("hello world the quick brown fox"), _greedy(max_tokens=8))
    assert res.completion_tokens > 0


# -------------------------------------------------------- on the device --

gpu_gate = pytest.mark.skipif(not torch.cuda.is_available(),
                              reason="the radix kernels are triton; needs a device")






@gpu_gate
def test_explicit_mtp_on_the_device_keeps_the_full_head_and_the_stream(head_1k, capsys):
    """End to end through load_compressed on the accelerator with
    DRINKME_MTP_DEPTH=4 and DRINKME_MTP_DRAFT_VOCAB=300, both profiles: the head
    starts, keeps the full projection, says so once, and emits the serial
    greedy stream — the lever cannot move a token, and neither can the
    fallback."""
    model_dir, packs, cfg = head_1k
    with mtp_depth(0):
        serial = load_compressed(model_dir, None, packs["sip"], device="cuda", ctx=256)
        want, _ = _run(serial, _msgs("hello world the quick brown fox"),
                       _greedy(max_tokens=16))
    del serial
    for profile in ("sip", "gulp"):
        with mtp_depth(4), draft_vocab_env(300):
            eng = load_compressed(model_dir, None, packs[profile], device="cuda", ctx=256)
        err = capsys.readouterr().err
        assert eng.mtp_head is not None and eng.mtp_head.draft_proj is None
        assert "drafting over the full vocab" in err
        assert registry.describe(RADIX, 0) in err
        with mtp_depth(4):
            res, _ = _run(eng, _msgs("hello world the quick brown fox"), _greedy(max_tokens=16))
        assert res.completion_tokens > 0
        assert res.text == want.text, profile
        del eng
