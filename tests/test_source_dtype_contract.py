"""The source-dtype contract at the packing and loading boundary
(codec/pack.py SOURCE_DTYPE):

  * a checkpoint that stores a tensor the codec would pack as anything but
    BF16 — an FP16 release, an FP32 one, a BF16 one with a single F32
    eligible Linear — is REFUSED BY NAME (tensor, dtype, checkpoint) by
    `pack_model`, before a shard is opened and before torch is imported;
    nothing is written and nothing is cast. Before this, pack_model cast
    such a tensor to bf16, round-tripped the ROUNDED bits, and stamped
    `sourceDtype: bf16` — a lossy conversion labelled lossless, on a
    checkpoint `drinkme check` had already refused.
  * check.assess, refuse_checkpoint (every loader's door, so the raw
    walker engines.stream_checkpoint too) and the packing boundary itself
    (_pack_one / pack_weight_radix) read ONE decision:
    codec/pack.source_dtype_refusal_reason over codec/pack.header_eligible.
  * tensors outside the packed set (norms, embeddings, biases) are outside
    the contract: a BF16 checkpoint with an F32 norm still packs, and its
    `sourceDtype` is DERIVED from the dtypes the packer read.

These drive pack_model end to end over a saved one-layer Llama; a codec-only
round trip cannot see the cast (it was applied before the round trip).
"""

import json
import os
import pathlib
import subprocess
import sys
import textwrap

import numpy as np
import pytest
import torch

from drinkme.codec import pack as P
from drinkme.codec import radix_pack as rp
from drinkme.codec.pack import UnsupportedCheckpoint
from drinkme.check import assess

ROOT = pathlib.Path(__file__).resolve().parents[1]

UP = "model.layers.0.mlp.up_proj.weight"


def _one_layer_llama(path: str, dtype, overrides: dict | None = None) -> str:
    """A one-layer Llama (1024-wide, so every Linear but the tiny lm_head is
    eligible) saved as safetensors at `dtype`; `overrides` = {tensor name:
    torch dtype} re-saves those tensors at another dtype (the mixed toys).
    Returns the checkpoint dir."""
    from safetensors.torch import load_file, save_file
    from transformers import LlamaConfig, LlamaForCausalLM

    cfg = LlamaConfig(vocab_size=64, hidden_size=1024, intermediate_size=1024,
                      num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
                      max_position_embeddings=256, tie_word_embeddings=False)
    torch.manual_seed(0)
    model = LlamaForCausalLM(cfg).to(dtype).eval()
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, torch.nn.Linear) and min(m.weight.shape) >= 1024:
                m.weight.data.mul_(1 / 64)
    model.save_pretrained(path, safe_serialization=True)
    if overrides:
        shard = os.path.join(path, "model.safetensors")
        tensors = load_file(shard)
        for name, dt in overrides.items():
            tensors[name] = tensors[name].to(dt)
        save_file(tensors, shard, metadata={"format": "pt"})
    return path


def _headers(path: str) -> dict:
    return {k: (d, s) for k, (d, s, _f) in P._shard_headers(path).items()}


def _pack(path: str, out: str) -> str:
    return P.pack_model(path, None, out, progress=lambda *_: None)


@pytest.fixture(scope="module")
def toys(tmp_path_factory):
    d = tmp_path_factory.mktemp("source_dtype")
    return {
        "f16": _one_layer_llama(str(d / "f16"), torch.float16),
        "f32": _one_layer_llama(str(d / "f32"), torch.float32),
        # a BF16 release with ONE eligible Linear stored wider
        "mixed_eligible": _one_layer_llama(str(d / "mixed_eligible"), torch.bfloat16,
                                           overrides={UP: torch.float32}),
        # a BF16 release whose norm (never packed) is stored wider
        "mixed_raw": _one_layer_llama(str(d / "mixed_raw"), torch.bfloat16,
                                      overrides={"model.norm.weight": torch.float32}),
        "bf16": _one_layer_llama(str(d / "bf16"), torch.bfloat16),
        "root": d,
    }


# ------------------------------------------------------ pack_model's door --


@pytest.mark.parametrize("toy, dtype, first", [
    ("f16", "F16", "model.layers.0.mlp.down_proj.weight"),
    ("f32", "F32", "model.layers.0.mlp.down_proj.weight"),
    ("mixed_eligible", "F32", UP),
])
def test_pack_model_refuses_a_non_bf16_eligible_tensor_by_name(toys, toy, dtype, first):
    """The FP16 twin of the bf16 toy, its FP32 twin, and a bf16 checkpoint
    with one F32 eligible Linear: pack_model refuses, the line names the
    checkpoint, the tensor and its dtype, and nothing is written."""
    ckpt, out = toys[toy], str(toys["root"] / f"pack-{toy}")
    with pytest.raises(UnsupportedCheckpoint) as ei:
        _pack(ckpt, out)
    line = str(ei.value)
    assert line.startswith(f"{ckpt}: {first} is stored as {dtype}, not BF16"), line
    assert "never converts" in line and "bf16 (served losslessly)" in line
    assert not os.path.exists(out) and not [p for p in os.listdir(toys["root"]) if "staging" in p]


def test_pack_model_refuses_before_torch_is_imported(toys):
    """The refusal is torch-free: `drinkme pack --model <fp16 dir>` in a
    fresh interpreter prints the line, returns 3 (a definite verdict about
    this checkpoint — exitcodes.REFUSED), and never imports torch (the
    same proof tests/test_unsupported_format.py gives for FP8)."""
    ckpt, out = toys["f16"], str(toys["root"] / "pack-cli")
    code = f"""
        import sys
        from drinkme.cli import main
        rc = main(["pack", "--model", {ckpt!r}, "-o", {out!r}])
        sys.stderr.flush()
        print("TORCH_IMPORTED=" + str("torch" in sys.modules))
        print("RC=" + str(rc))
        """
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(
               [str(ROOT / "src")] + [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p]),
           "DRINKME_NO_AUTO_DEPS": "1", "HIP_VISIBLE_DEVICES": "", "CUDA_VISIBLE_DEVICES": "",
           "OMP_NUM_THREADS": "4"}
    r = subprocess.run([sys.executable, "-c", textwrap.dedent(code)],
                       capture_output=True, text=True, timeout=120, env=env)
    assert r.returncode == 0, r.stderr
    assert r.stderr.strip().startswith(
        f"drinkme pack: {ckpt}: model.layers.0.mlp.down_proj.weight is stored as F16, not BF16"), r.stderr
    assert "RC=3" in r.stdout and "TORCH_IMPORTED=False" in r.stdout, r.stdout
    assert not os.path.exists(out)


def test_a_bf16_checkpoint_with_a_wider_norm_still_packs_and_derives_source_dtype(toys):
    """Outside the packed set the contract is silent: the F32 norm rides
    raw (the loader's bf16 rule, docs/pack-format.md), the pack is written,
    and meta.json's sourceDtype says what the packer READ — bf16 — for
    both the plain bf16 toy and the mixed-raw one."""
    for toy in ("bf16", "mixed_raw"):
        out = _pack(toys[toy], str(toys["root"] / f"pack-{toy}"))
        meta = json.load(open(os.path.join(out, "meta.json")))
        assert meta["sourceDtype"] == "bf16" and meta["dtype"] == "bf16"
        assert meta["tensorCount"] == 5  # q o gate up down; k/v are 512-wide, lm_head 64
        # and every packed tensor decodes to the SOURCE bits, not a cast of them
        from safetensors import safe_open

        with safe_open(os.path.join(toys[toy], "model.safetensors"), framework="pt") as sf:
            for name, p in P.iter_pack_dir(out):
                src = sf.get_tensor(name + ".weight")
                assert src.dtype == torch.bfloat16
                back = rp.decode_back(p)
                assert np.array_equal(back, src.contiguous().view(torch.int16).numpy().view(np.uint16))


# ------------------------------------------- one decision, at every door --


def test_check_and_the_packer_give_the_same_reason(toys):
    """check.assess (the hub-side estimate) and pack_model's door read
    the same function: the reason is the same string, the line differs
    only by the checkpoint's name prefix."""
    for toy in ("f16", "f32", "mixed_eligible"):
        ckpt = toys[toy]
        v = assess(ckpt, None, P.read_config_json(ckpt), _headers(ckpt))
        assert not v.ok
        with pytest.raises(UnsupportedCheckpoint) as ei:
            _pack(ckpt, str(toys["root"] / "never"))
        assert str(ei.value) == f"{ckpt}: {v.reason}"
    for toy in ("bf16", "mixed_raw"):
        v = assess(toys[toy], None, P.read_config_json(toys[toy]), _headers(toys[toy]))
        assert v.ok, v.reason
        assert v.n_packable == 5


def test_the_raw_loader_refuses_the_same_checkpoint_by_name(toys):
    """engines.stream_checkpoint — every loader's walker, whose float->bf16
    cast is the serving-side twin of the packer's — comes through the same
    door and refuses before a tensor is read."""
    from drinkme.serving.engines import stream_checkpoint

    with pytest.raises(UnsupportedCheckpoint) as ei:
        stream_checkpoint(torch.nn.Module(), toys["f16"], None, "cpu", name="toy/f16")
    assert str(ei.value).startswith("toy/f16: model.layers.0.mlp.down_proj.weight is stored as F16, not BF16")


def test_the_streaming_arms_refuse_too(toys):
    """arms._stream_swapped (the bench's compressed and twin arms) cast a
    wider float to bf16 before installing it; now its door refuses."""
    from drinkme.arms import load_compressed_streaming

    with pytest.raises(UnsupportedCheckpoint, match="is stored as F16, not BF16"):
        load_compressed_streaming(toys["f16"], None, "cpu")


def test_the_packing_boundary_itself_refuses_never_casts():
    """_pack_one and pack_weight_radix, handed a non-bf16 tensor directly
    (no door in front of them): refused by name, never cast."""
    w = torch.randn(1024, 1024, dtype=torch.float16)
    with pytest.raises(UnsupportedCheckpoint) as ei:
        P._pack_one(w, "sip", name="model.layers.0.mlp.up_proj.weight")
    assert str(ei.value).startswith("model.layers.0.mlp.up_proj.weight is stored as F16, not BF16")
    with pytest.raises(TypeError, match="not bf16"):
        rp.pack_weight_radix(w, "sip", name="x")
    with pytest.raises(TypeError, match="not bf16"):
        rp.pack_weight_radix(w.float(), "sip", name="x")
    with pytest.raises(UnsupportedCheckpoint):
        P.pack_weight_served(w, name="x")


def test_the_shared_decision_names_the_first_eligible_offender():
    """The pure function both doors call: the FIRST eligible tensor stored
    as anything but BF16, by name and dtype; a non-eligible F32 tensor
    (a norm, an embedding table) is not an offender."""
    from drinkme.codec.pack import SOURCE_DTYPE, header_eligible, source_dtype_refusal_reason

    cfg = {"tie_word_embeddings": True}
    tensors = [("model.embed_tokens.weight", "F32"), ("model.norm.weight", "F32"),
               ("lm_head.weight", "F32"),  # tied: not packed
               ("model.layers.0.mlp.up_proj.weight", "BF16"),
               ("model.layers.0.mlp.down_proj.weight", "F16"),
               ("model.layers.1.mlp.down_proj.weight", "F32")]
    shapes = {n: [1024, 1024] for n, _ in tensors}
    shapes["model.norm.weight"] = [1024]
    elig = lambda n: header_eligible(n, shapes[n], cfg)  # noqa: E731
    reason = source_dtype_refusal_reason(tensors, elig)
    assert reason.startswith("model.layers.0.mlp.down_proj.weight is stored as F16, not BF16")
    assert source_dtype_refusal_reason(tensors[:4], elig) is None
    assert SOURCE_DTYPE == "BF16"
    # header_eligible is the shape-gate estimate, dtype-blind
    assert header_eligible("model.layers.0.mlp.up_proj.weight", [1024, 1024], {})
    assert not header_eligible("model.layers.0.mlp.up_proj.weight", [1024, 1022], {})
    assert not header_eligible("model.layers.0.mlp.up_proj.weight", [512, 1024], {})
    assert not header_eligible("model.embed_tokens.weight", [151936, 4096], {})
    assert header_eligible("lm_head.weight", [151936, 4096], {"tie_word_embeddings": False})
    assert not header_eligible("lm_head.weight", [151936, 4096], {"tie_word_embeddings": True})
