"""THE one refusal for what this release does not serve (codec/pack.py):

  * an FP8 checkpoint — any float8 plane in its shard headers, or an fp8
    quantization_config in config.json — is refused BY NAME with ONE line,
      `<repo>: FP8 checkpoints are not supported in this release (bf16 only)`
    at every door it can reach: `drinkme pack`, the raw-trunk loader
    (engines.stream_checkpoint), `check`, `serve --model` (its torch-
    free front door for an existing pack, and its pack-on-demand path) and
    `bench` (the pack-dir door and the streaming arms). Nothing fp8 may
    pack, load or serve.
  * every other by-name refusal for a layout or dtype this build does not
    implement (a named quant_method, an unnamed quantized checkpoint, an
    unrecognized per-tensor `dtype` scalar in a pack) ends with the same
    tail naming what IS supported — bf16 alone — so what is supported is
    stated once and cannot drift between call sites.

The pack door is TORCH-FREE: `drinkme pack --model <dir>` on an FP8
checkpoint prints the line, exits 3 (a definite verdict about this
checkpoint — exitcodes.REFUSED), and never imports torch (the
subprocess below asserts `torch` is not in sys.modules at exit).
"""

import json
import os
import pathlib
import subprocess
import sys
import textwrap

import pytest

from drinkme.codec import pack as P
from drinkme.codec.pack import (FP8_REFUSAL, UnsupportedCheckpoint, checkpoint_refusal_reason,
                                refusal_for_checkpoint, unsupported_format)

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
from fixtures import fake_checkpoint  # noqa: E402

THE_LINE = "FP8 checkpoints are not supported in this release (bf16 only)"
QWEN_FP8_QC = {"quant_method": "fp8", "fmt": "e4m3", "weight_block_size": [128, 128],
               "activation_scheme": "dynamic"}


def test_the_line_is_the_line():
    assert FP8_REFUSAL == THE_LINE
    assert refusal_for_checkpoint("Qwen/Qwen3-8B-FP8", {"quantization_config": QWEN_FP8_QC}, ["BF16"]) \
        == f"Qwen/Qwen3-8B-FP8: {THE_LINE}"
    assert "yet" not in THE_LINE


@pytest.mark.parametrize("config, dtypes", [
    ({"quantization_config": QWEN_FP8_QC}, ["BF16"]),                      # config.json says fp8
    ({"quantization_config": {"quant_method": "fbgemm_fp8"}}, ["BF16"]),   # Meta's spelling
    ({"quantization_config": {"quant_method": "compressed-tensors",
                              "config_groups": {"g": {"weights": {"type": "float", "num_bits": 8}}}}},
     ["F8_E4M3"]),                                                         # words do not say it; the plane does
    ({}, ["BF16", "F8_E4M3"]),                                             # a float8 plane, no config
    ({}, ["F8_E5M2"]),                                                     # the other float8
    ({"quantization_config": {"quant_method": "modelopt", "quant_algo": "FP8"}}, ["BF16"]),
])
def test_every_fp8_spelling_gets_the_one_line(config, dtypes):
    assert checkpoint_refusal_reason(config, dtypes) == THE_LINE


def test_a_bf16_checkpoint_is_not_refused():
    assert checkpoint_refusal_reason({"architectures": ["Qwen3ForCausalLM"]}, ["BF16", "F32"]) is None
    assert checkpoint_refusal_reason({"quantization_config": None}, ["BF16"]) is None
    assert checkpoint_refusal_reason({"quantization_config": {}}, ["BF16"]) is None


def test_a_named_non_fp8_quant_method_is_refused_in_the_same_shape():
    """compressed-tensors (int), gptq, awq, bitsandbytes — the int4/fp4
    family: any other quantization_config is refused by name, one line,
    naming its quant_method."""
    for method in ("gptq", "awq", "bitsandbytes", "compressed-tensors"):
        reason = checkpoint_refusal_reason({"quantization_config": {"quant_method": method, "bits": 4}}, ["I32"])
        assert reason == f"quantized checkpoints (quant_method={method!r}) are not supported in this release (bf16 only)"
    assert checkpoint_refusal_reason({"quantization_config": {"bits": 4}}, ["I32"]) \
        == "quantized checkpoints (a quantization_config) are not supported in this release (bf16 only)"


def test_unsupported_format_names_what_was_seen_and_lists_bf16_alone():
    msg = unsupported_format("dtype 'fp4_e2m1'")
    assert "dtype 'fp4_e2m1'" in msg
    assert "bf16 (served losslessly)" in msg
    assert "fp8" not in msg.lower() and "yet" not in msg.lower()
    assert P.SUPPORTED_FORMATS == ("bf16 (served losslessly)",)


# ------------------------------------------------------------- the doors --


def test_check_refuses_an_fp8_release_with_the_line():
    from drinkme.check import assess

    v = assess("Qwen/Qwen3-8B-FP8", None, {"quantization_config": QWEN_FP8_QC},
               {"model.layers.0.mlp.up_proj.weight": ("F8_E4M3", [4096, 4096]),
                "model.layers.0.mlp.up_proj.weight_scale_inv": ("BF16", [32, 32])})
    assert not v.ok and v.reason == THE_LINE
    assert v.line() == f"check Qwen/Qwen3-8B-FP8: refused — {THE_LINE}"
    # a float8 plane with no quantization_config at all: the same line
    v = assess("toy/plane", None, {}, {"w": ("F8_E4M3", [1024, 1024])})
    assert not v.ok and v.reason == THE_LINE


def test_check_refuses_a_named_non_fp8_quant_method_by_name():
    from drinkme.check import assess

    for method in ("bitsandbytes", "gptq"):
        v = assess(f"toy/{method}", None, {"quantization_config": {"quant_method": method}},
                   {"w": ("I8", [1024, 1024])})
        assert not v.ok and f"quant_method={method!r}" in v.reason and "(bf16 only)" in v.reason


def test_check_an_unnamed_quantized_checkpoint_names_bf16_alone():
    from drinkme.check import assess

    v = assess("toy/unnamed-quant", None, {}, {"w": ("I8", [2048, 2048])})
    assert not v.ok
    assert "bf16 (served losslessly)" in v.reason and "fp8" not in v.reason.lower()


def test_pack_model_refuses_an_fp8_checkpoint_with_the_line(tmp_path):
    """pack_model's door, in-process: the line, as UnsupportedCheckpoint
    (a ValueError), off config.json — and off the shard header alone."""
    d = fake_checkpoint(str(tmp_path / "fp8"), quantization_config=QWEN_FP8_QC)
    with pytest.raises(UnsupportedCheckpoint) as ei:
        P.pack_model(d, None, str(tmp_path / "pack"), progress=lambda *_: None)
    assert str(ei.value) == f"{d}: {THE_LINE}"
    assert not os.path.exists(tmp_path / "pack")
    d2 = fake_checkpoint(str(tmp_path / "plane"),
                         tensors={"model.layers.0.mlp.up_proj.weight": ("F8_E4M3", [64, 64])})
    with pytest.raises(UnsupportedCheckpoint) as ei:
        P.pack_model(d2, None, str(tmp_path / "pack2"), progress=lambda *_: None)
    assert str(ei.value) == f"{d2}: {THE_LINE}"
    with pytest.raises(UnsupportedCheckpoint) as ei:
        P.pack_mtp_head(d, None, str(tmp_path / "pack3"), progress=lambda *_: None)
    assert str(ei.value) == f"{d}: {THE_LINE}"


def _drinkme_pack_in_a_fresh_interpreter(model_dir: str, out: str) -> subprocess.CompletedProcess:
    """`drinkme pack --model <dir>` through cli.main in a fresh interpreter,
    reporting whether torch was ever imported.

    DRINKME_NO_AUTO_DEPS=1 is NOT optional here: cli.main runs
    bootstrap.ensure_accelerator before `pack`, which imports torch to
    classify the accelerator and — under a hidden accelerator, which is how
    this suite runs — would decide the environment is broken and drive
    `uv sync` against the shared .venv (it did, once). With the guard the bootstrap returns before importing
    anything, and what is measured is pack's own door."""
    code = f"""
        import sys
        from drinkme.cli import main
        rc = main(["pack", "--model", {model_dir!r}, "-o", {out!r}])
        sys.stderr.flush()
        print("TORCH_IMPORTED=" + str("torch" in sys.modules))
        print("RC=" + str(rc))
        """
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(
               [str(ROOT / "src")] + [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p]),
           "DRINKME_NO_AUTO_DEPS": "1", "HIP_VISIBLE_DEVICES": "",
           "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "4"}
    return subprocess.run([sys.executable, "-c", textwrap.dedent(code)],
                          capture_output=True, text=True, timeout=120, env=env)


@pytest.mark.parametrize("kind", ["quantization_config", "float8-plane"])
def test_drinkme_pack_prints_the_line_exits_3_and_never_imports_torch(tmp_path, kind):
    """The proof: a fake
    checkpoint dir whose config.json carries a quantization_config — and one
    whose safetensors header names a float8 dtype — through the real CLI:
    the one line on stderr, exit 3, `torch` not in sys.modules."""
    if kind == "quantization_config":
        d = fake_checkpoint(str(tmp_path / "fp8"), quantization_config=QWEN_FP8_QC)
    else:
        d = fake_checkpoint(str(tmp_path / "plane"),
                            tensors={"model.layers.0.mlp.up_proj.weight": ("F8_E4M3", [64, 64]),
                                     "model.layers.0.mlp.up_proj.weight_scale_inv": ("BF16", [1, 1])})
    r = _drinkme_pack_in_a_fresh_interpreter(d, str(tmp_path / "pack"))
    assert r.returncode == 0, r.stderr  # main() RETURNED (the code is in RC=)
    assert r.stderr.strip() == f"drinkme pack: {d}: {THE_LINE}", r.stderr
    assert "RC=3" in r.stdout and "TORCH_IMPORTED=False" in r.stdout, r.stdout
    assert not os.path.exists(tmp_path / "pack")


def test_serve_auto_pack_and_the_pack_dir_front_door_print_the_line(tmp_path, monkeypatch):
    """serve's two doors: pack-on-demand for a repo with no pack (check
    unavailable offline, so pack_model's own door speaks), and the torch-
    free front door for an EXISTING pack cut from an FP8 checkpoint (its
    meta.json says so)."""
    from drinkme import serve
    from drinkme.check import CheckUnavailable

    d = fake_checkpoint(str(tmp_path / "fp8"), quantization_config=QWEN_FP8_QC)
    monkeypatch.setattr("drinkme.check.check",
                        lambda repo, rev=None: (_ for _ in ()).throw(CheckUnavailable("offline")))
    with pytest.raises(SystemExit) as ei:
        serve.ensure_pack(d, None, str(tmp_path / "pack"), auto_pack=True)
    assert str(ei.value) == f"drinkme serve: {d}: {THE_LINE}"
    # an FP8 pack on disk: refused at the front door before any load
    pdir = tmp_path / "fp8pack"
    pdir.mkdir()
    (pdir / "meta.json").write_text(json.dumps({
        "formatVersion": 1, "hfRepo": "Qwen/Qwen3-8B-FP8", "revision": None,
        "dtype": "fp8_e4m3", "sourceDtype": "fp8_e4m3", "method": "fp8-e4m3-block128",
        "tensors": {}}))
    with pytest.raises(SystemExit) as ei:
        serve.check_pack_dir("Qwen/Qwen3-8B-FP8", None, str(pdir))
    assert str(ei.value) == f"drinkme serve: Qwen/Qwen3-8B-FP8: {THE_LINE}"
    with pytest.raises(UnsupportedCheckpoint, match=THE_LINE.replace("(", r"\(").replace(")", r"\)")):
        dict(P.iter_pack_dir(str(pdir)))


def test_the_raw_trunk_loader_refuses_an_fp8_checkpoint_with_the_line(tmp_path):
    """engines.stream_checkpoint — the one walker every loader shares —
    refuses before a tensor is read, naming the repo."""
    import torch

    from drinkme.serving.engines import stream_checkpoint

    d = fake_checkpoint(str(tmp_path / "fp8"),
                        tensors={"model.layers.0.mlp.up_proj.weight": ("F8_E4M3", [64, 64])})
    with pytest.raises(UnsupportedCheckpoint) as ei:
        stream_checkpoint(torch.nn.Module(), d, None, "cpu", name="Qwen/Qwen3-8B-FP8")
    assert str(ei.value) == f"Qwen/Qwen3-8B-FP8: {THE_LINE}"


def test_an_fp8_pack_tensor_is_refused_by_name_at_load_and_in_identity(tmp_path):
    """A pack tensor whose scalars carry an FP8 pack's `dtype: fp8_e4m3`
    (or any dtype scalar) is refused by the loader, the module constructor
    and the manifest describer — never labelled bf16."""
    import numpy as np

    from drinkme.codec.identity import tensor_info
    from drinkme.codec.swap import make_module

    with pytest.raises(UnsupportedCheckpoint, match="FP8 checkpoints are not supported"):
        P._arrays_for(1, "fp8_e4m3", name="x")
    with pytest.raises(ValueError) as ei:
        P._arrays_for(1, "fp4_e2m1", name="x")
    assert "bf16 (served losslessly)" in str(ei.value) and "fp4_e2m1" in str(ei.value)
    with pytest.raises(UnsupportedCheckpoint, match="FP8 checkpoints are not supported"):
        make_module({"format_version": 1, "R": 4, "C": 4, "dtype": "fp8_e4m3", "w8": np.zeros((4, 4), np.uint8)},
                    None, "cpu")
    with pytest.raises(ValueError, match="FP8 checkpoints are not supported"):
        tensor_info({"R": 4, "C": 4, "dtype": "fp8_e4m3"}, "x")
    with pytest.raises(ValueError) as ei:
        tensor_info({"R": 4, "C": 4, "dtype": "fp4_e2m1"}, "x")
    assert "bf16 (served losslessly)" in str(ei.value)
