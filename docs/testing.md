# Testing

The CPU suite's command is in [AGENTS.md](../AGENTS.md#cpu-tests):

```sh
DRINKME_NO_AUTO_DEPS=1 HIP_VISIBLE_DEVICES='' CUDA_VISIBLE_DEVICES='' \
  OMP_NUM_THREADS=4 DRINKME_HOME=$(mktemp -d) \
  .venv/bin/python -m pytest -q
```

This page explains its settings, what a clean run looks like, and which tests
skip. The hardware checks each kind of change needs are in
[checks by change type](checks.md).

## Settings

| Setting | Why it is there |
|---|---|
| `DRINKME_NO_AUTO_DEPS=1` | Disables automatic dependency installation when accelerators are hidden. `tests/conftest.py` also sets this; setting it here covers subprocesses and runs that bypass conftest. |
| `HIP_VISIBLE_DEVICES=''` `CUDA_VISIBLE_DEVICES=''` | Hides every accelerator from torch, so the GPU-gated tests skip instead of dispatching Triton kernels on a device that may be serving. Both are set because ROCm and CUDA read different variables; the real-self-test test (`test_bootstrap.py`) skips when either hides the device. |
| `OMP_NUM_THREADS=4` | Bounds the CPU threads torch and numpy spawn. The suite's own subprocess helpers pin the same value. |
| `DRINKME_HOME=<tmp>` | The pack-cache root. A temporary directory isolates selection and packing tests from `~/.cache/drinkme/packs`; tests requiring a real Qwen3-8B pack and GPU skip. `DRINKME_SLOT_DIR` is already forced off by conftest. The menu's checkpoint sizes come from `tests/menu_checkpoints.py` through conftest, never from this machine's Hugging Face cache or the Hub. |
| `.venv/bin/python -m pytest` | Uses the project interpreter directly. With the same environment settings, `uv run --no-sync pytest -q` is equivalent when uv's cache lock is free. Keep the full output visible, including setup messages and the test verdict. |

## A clean run

The suite takes about four to five minutes. Read pytest's printed verdict, not
the exit status:
TheRock's ROCm torch can clobber the exit code to 0 ([known behavior](dev-environment.md#known-behavior)).
The warnings are Transformers' `device=` deprecation on rotary-embedding
construction, a TorchScript deprecation, and FLA's "Triton is not supported on
current platform" — the expected CPU fallback with accelerators hidden.

Expected skips on a Linux CPU run:

| Where | Reason | Turned on by |
|---|---|---|
| `test_codec_bias_rounding.py`, `test_serving_sleep.py`, `test_codec_mc.py`, `test_radix_bounds.py`, `test_codec_registry.py`, `test_bootstrap.py` | Triton kernels, a real device's memory, or the real bootstrap self-test | A visible accelerator. These are not the acceptance gates; the `bench/` scripts are ([developer instruments](../bench/README.md)). `test_codec_mc.py` also wants the real Qwen3-8B pack under `DRINKME_HOME`. |
| `test_metal_radix.py`, `test_serving_engine_mlx.py`, `test_identity_torch_free.py`, `test_pack_identity.py` | `mlx` is not installed (a Linux machine) | The `metal` lane on Apple silicon; `test_metal_radix.py` also wants `DRINKME_RADIX_MLX_TOY`, and so do the mlx tests in `test_identity_torch_free.py` and `test_pack_identity.py` (the plain toy and the `--identity` toy, `bench/radix_mlx_toy_build.py`). |
| `test_serving_constrain.py` | `jsonschema` is not installed | `uv pip install jsonschema`. |

An mlx-gated test never needs torch at collection or in-process to reach its
own `pytest.importorskip("mlx.core")`: it loads a prebuilt toy from
`DRINKME_RADIX_MLX_TOY` and skips by name, naming the missing variable and the
builder flag, when torch is absent and the variable is unset.

## The same suite on a Mac

The suite gives one verdict on Linux and on a Mac. drinkme reads the host
from `platform.system()` and `platform.machine()` (Apple silicon selects the
mlx runtime, the metal lane and MLX's working set) and from
`DRINKME_RUNTIME`. A test written for a Linux torch box pins that host with
`pin_linux_host` in [`tests/hosts.py`](../tests/hosts.py), or with the
`linux_host` fixture, so it tests the same thing on every host.

To check this on Linux, run the suite as a Mac with
`DRINKME_TEST_HOST=darwin-arm64` added to the command above: every test sees
Darwin on arm64, and an `mlx.core` stand-in that reports Metal and an M4's
working set to the host probes. The stand-in only answers those probes, and
the mlx-gated tests skip on it by name. Use the project interpreter, which
has no mlx: a real mlx imported at collection would meet the stand-in at run
time. The verdict should match the plain run's, test for test.

## Tests that need more than the tree

Tests that pin behavior against real cached checkpoints' small files
(`Qwen3-8B`, `Qwen3.8-27B`, `gemma-4-31B-it` in `test_serving_capability.py`,
`test_serving_control.py`, `test_serving_gen_config.py`,
`test_serving_tool_formats.py`; `Qwen3-8B` and `Qwen3-1.7B` in
`test_pack_resident_bytes.py`; the four vision models' configs and tokenizers
in `test_serving_vision_attention.py`, `test_serving_glimmer_vision.py` and
`test_serving_glimmer_channel.py`; Qwen3.8-27B's video config, template and
tokenizer in `test_serving_video.py`) skip on a machine whose Hugging Face
cache lacks them, naming the missing revision. They never download.

The video tests that decode (`test_serving_video.py`,
`test_serving_video_prompt.py`, the video halves of
`test_serving_http_content_parts.py` and `test_serving_media_reuse.py`) need
PyAV, the optional `drinkme[video]` extra, and skip without it, naming the
extra.

The HTTP tests bind a random localhost port over `FakeEngine`; a sandbox that
forbids local sockets fails them rather than skipping.
