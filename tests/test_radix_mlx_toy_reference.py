"""bench/radix_mlx_toy_build.py's biased twin and torch reference, on the
torch box that builds them (the mlx suite that CONSUMES them runs where
mlx is — tests/test_serving_engine_mlx.py (5)).

  (1) --bias is a Qwen3 with attention_bias=True whose eight q/k/v/o biases
      are nonzero on disk, packed to the same 10 radix tensors as the plain
      toy; the torch engine attaches the four q/o biases after the swap
      (CompressedLinear.bias) and k/v keep theirs as plain Linears;
  (2) write_reference -> load_reference round-trips, and the torch engine
      re-transcribed on the same box is BITWISE the file (a reference that
      does not reproduce on its own box is no reference);
  (3) a reference for other bytes — a checkpoint byte flipped, a pack's
      tensor map moved — is refused by name.
"""

import json
import os
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bench"))
from radix_mlx_toy_build import (  # noqa: E402
    MAX_TOKENS, PROMPTS, REFERENCE, build_toy_qwen3, load_reference, torch_transcript,
    toy_digests, write_reference,
)


@pytest.fixture(scope="module")
def biased_toy(tmp_path_factory):
    d = tmp_path_factory.mktemp("radix_mlx_toy_bias")
    return build_toy_qwen3(str(d / "model"), str(d / "pack"), bias=True)


@pytest.fixture(scope="module")
def reference(biased_toy):
    model_dir, pack_dir = biased_toy
    path = os.path.join(os.path.dirname(model_dir), REFERENCE)
    ref = write_reference(path, model_dir, pack_dir)
    return path, ref


@pytest.fixture(scope="module")
def serial_env():
    """The serial loop, no prefix cache — write_reference's own settings,
    held for the whole module: the torch engine resolves its speculation
    plan at its first generate, so the env must still say so THEN."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("DRINKME_PREFIX_SLOTS", "0")
        mp.setenv("DRINKME_SPEC", "off")
        yield


def _torch_engine(model_dir, pack_dir):
    from drinkme.serving.engines import load_compressed

    return load_compressed(model_dir, None, pack_dir, device="cpu")


def test_the_biased_twin_carries_eight_nonzero_biases_the_engine_attaches(biased_toy, serial_env):
    from safetensors import safe_open

    from drinkme.codec.swap import CompressedLinear

    model_dir, pack_dir = biased_toy
    cfg = json.load(open(os.path.join(model_dir, "config.json")))
    assert cfg["attention_bias"] is True
    with safe_open(os.path.join(model_dir, "model.safetensors"), "pt") as f:
        biases = sorted(k for k in f.keys() if k.endswith(".bias"))
        assert len(biases) == 8 and all(f.get_tensor(k).abs().sum() > 0 for k in biases), biases
    assert all(k.split(".")[-2] in ("q_proj", "k_proj", "v_proj", "o_proj") for k in biases)
    meta = json.load(open(os.path.join(pack_dir, "meta.json")))
    assert meta["tensorCount"] == 10 and meta["radixTensorCount"] == 10
    eng = _torch_engine(model_dir, pack_dir)
    packed_bias = [m for m in eng.model.modules()
                   if isinstance(m, CompressedLinear) and m.bias is not None]
    assert len(packed_bias) == 4 and all(m.bias.abs().sum() > 0 for m in packed_bias)
    kv = [m for n, m in eng.model.named_modules()
          if n.endswith(("k_proj", "v_proj")) and isinstance(m, torch.nn.Linear)]
    assert len(kv) == 4 and all(m.bias is not None for m in kv)


def test_the_reference_round_trips_and_the_engine_reproduces_it_bitwise(biased_toy, reference, serial_env):
    model_dir, pack_dir = biased_toy
    path, ref = reference
    assert ref["prompts"] == PROMPTS and ref["max_tokens"] == MAX_TOKENS
    assert ref["eos_ids"] == [2] and ref["sampling_defaults"]["temperature"] == 1.0
    assert len(ref["rows"]) == len(PROMPTS)
    again = load_reference(path, model_dir, pack_dir)
    eng = _torch_engine(model_dir, pack_dir)
    rows = torch_transcript(eng, PROMPTS, MAX_TOKENS)
    for got, want, back in zip(rows, ref["rows"], again["rows"]):
        assert got["ids"].tolist() == want["ids"].tolist() == back["ids"].tolist()
        assert np.array_equal(got["prompt_logits"], want["prompt_logits"])
        assert np.array_equal(want["prompt_logits"], back["prompt_logits"])
        assert got["gen_ids"].tolist() == want["gen_ids"].tolist() == back["gen_ids"].tolist()
        assert len(got["gen_ids"]) == MAX_TOKENS and got["finish"] == "length"
        assert got["step_logits"].shape == (MAX_TOKENS, 64)
        assert np.array_equal(got["step_logits"], want["step_logits"])
        assert got["text"] == want["text"] == back["text"] and got["finish"] == want["finish"]
        # the first step's row is the prompt-position row the engine sampled from
        assert int(got["step_logits"][0].argmax()) == int(got["gen_ids"][0])
    assert set(eng.eos_ids) == {2}


def test_a_reference_for_other_bytes_is_refused_by_name(biased_toy, reference, tmp_path):
    import shutil

    model_dir, pack_dir = biased_toy
    path, _ = reference
    digests = toy_digests(model_dir, pack_dir)
    assert len(digests["checkpoint_sha256"]) == 64 and len(digests["pack_manifest_sha256"]) == 64
    other_model = tmp_path / "model"
    shutil.copytree(model_dir, other_model)
    with open(other_model / "config.json", "a") as f:
        f.write("\n")
    with pytest.raises(ValueError, match="checkpoint_sha256.*rebuild it with --reference"):
        load_reference(path, str(other_model), pack_dir)
    other_pack = tmp_path / "pack"
    other_pack.mkdir()
    meta = json.load(open(os.path.join(pack_dir, "meta.json")))
    meta["tensors"] = dict(list(meta["tensors"].items())[1:])
    json.dump(meta, open(other_pack / "meta.json", "w"))
    with pytest.raises(ValueError, match="pack_manifest_sha256"):
        load_reference(path, model_dir, str(other_pack))
    load_reference(path, model_dir, pack_dir)


def test_torch_transcript_refuses_a_speculating_engine(biased_toy, monkeypatch):
    """A transcript recorded through the verify batch would carry batched
    rows the serial loop never samples; the recorder refuses the engine
    rather than record them."""
    monkeypatch.setenv("DRINKME_PREFIX_SLOTS", "0")
    monkeypatch.delenv("DRINKME_SPEC", raising=False)  # auto: ngram, no head
    model_dir, pack_dir = biased_toy
    eng = _torch_engine(model_dir, pack_dir)
    with pytest.raises(RuntimeError, match="serial loop.*'ngram'.*DRINKME_SPEC=off"):
        torch_transcript(eng, PROMPTS[:1], 4)
