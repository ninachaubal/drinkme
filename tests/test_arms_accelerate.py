"""`drinkme serve --stock` without `accelerate` installed: transformers
raises a bare ImportError three frames inside `from_pretrained` from its
own loading paths. arms.load_cpu refuses by name instead, naming the fix
(pyproject.toml's cuda/rocm dependency groups carry `accelerate`).

load_cpu's door (codec/pack.refuse_checkpoint) reads the snapshot's
config.json and shard headers first, so each test points arms.snapshot_dir
at a bf16-shaped fake checkpoint directory — no hub, no network."""

import pathlib
import sys

import pytest

from drinkme import arms

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from fixtures import fake_checkpoint  # noqa: E402


@pytest.fixture
def local_bf16(tmp_path, monkeypatch):
    d = fake_checkpoint(str(tmp_path / "bf16"))
    monkeypatch.setattr(arms, "snapshot_dir", lambda repo, rev=None: d)
    return d


def test_an_accelerate_importerror_is_refused_by_name(monkeypatch, local_bf16):
    class _FakeCls:
        @staticmethod
        def from_pretrained(*a, **kw):
            raise ImportError(
                "Loading this model requires accelerate "
                "(`pip install accelerate`)")

    import transformers

    monkeypatch.setattr(transformers, "AutoModelForCausalLM", _FakeCls)
    with pytest.raises(RuntimeError) as ei:
        arms.load_cpu("toy/some-repo")
    msg = str(ei.value)
    assert "accelerate" in msg
    assert "toy/some-repo" in msg
    assert "uv sync" in msg or "uv pip install accelerate" in msg
    assert "compressed arm does not need it" in msg


def test_an_unrelated_importerror_is_not_swallowed(monkeypatch, local_bf16):
    class _FakeCls:
        @staticmethod
        def from_pretrained(*a, **kw):
            raise ImportError("no module named definitely_not_a_real_module")

    import transformers

    monkeypatch.setattr(transformers, "AutoModelForCausalLM", _FakeCls)
    with pytest.raises(ImportError, match="definitely_not_a_real_module"):
        arms.load_cpu("toy/some-repo")


def test_load_cpu_refuses_an_fp8_checkpoint_before_from_pretrained(monkeypatch, tmp_path):
    """The `serve --stock` door: an FP8 repo prints the one line and
    transformers' loader is never called."""
    from drinkme.codec.pack import FP8_REFUSAL, UnsupportedCheckpoint

    d = fake_checkpoint(str(tmp_path / "fp8"), quantization_config={"quant_method": "fp8"})
    monkeypatch.setattr(arms, "snapshot_dir", lambda repo, rev=None: d)

    class _Never:
        @staticmethod
        def from_pretrained(*a, **kw):
            raise AssertionError("from_pretrained must not run on an FP8 checkpoint")

    import transformers

    monkeypatch.setattr(transformers, "AutoModelForCausalLM", _Never)
    with pytest.raises(UnsupportedCheckpoint) as ei:
        arms.load_cpu("Qwen/Qwen3-8B-FP8")
    assert str(ei.value) == f"Qwen/Qwen3-8B-FP8: {FP8_REFUSAL}"


def test_device_map_without_accelerate_is_named_not_routed_to_the_vlm_retry(monkeypatch, local_bf16):
    """transformers raises a ValueError (not ImportError) when device_map is
    asked for without accelerate; on the RX 7600 XT box that fell into
    load_cpu's multimodal retry and came out as 'Unrecognized configuration
    class ... AutoModelForImageTextToText'. It must be the named refusal."""
    import pytest
    from drinkme import arms

    monkeypatch.setattr(arms, "snapshot_dir", lambda repo, rev=None: local_bf16)

    class _Cls:
        @staticmethod
        def from_pretrained(*a, **k):
            raise ValueError("Using a `device_map`, `tp_plan`, `torch.device` context manager "
                             "or setting `torch.set_default_device(device)` requires `accelerate`.")

    import transformers
    monkeypatch.setattr(transformers, "AutoModelForCausalLM", _Cls)
    with pytest.raises(RuntimeError, match="needs the `accelerate` package"):
        arms.load_cpu("Qwen/Qwen3-4B", None, device_map="cuda")
