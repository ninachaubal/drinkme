"""sdpa.route: cuDNN attention excluded on CUDA, for decode and prefill alike,
and nowhere else (the H100 numbers are in sdpa.route's docstring)."""

import pytest
import torch

from drinkme import sdpa
from drinkme.serving import kernel_route


@pytest.fixture
def cuda_build(monkeypatch):
    """A CUDA torch as far as the route can tell: no HIP version."""
    monkeypatch.setattr(torch.version, "hip", None)
    monkeypatch.delenv(sdpa.CUDNN_ENV, raising=False)


@pytest.fixture
def cudnn_flag():
    """torch's process-wide cuDNN SDPA flag, put back after the test."""
    before = torch.backends.cuda.cudnn_sdp_enabled()
    torch.backends.cuda.enable_cudnn_sdp(True)
    sdpa._routed.clear()
    yield
    torch.backends.cuda.enable_cudnn_sdp(before)
    sdpa._routed.clear()


def test_excluded_on_cuda_only(cuda_build):
    assert sdpa.cudnn_excluded("cuda")
    assert sdpa.cudnn_excluded(torch.device("cuda", 0))
    assert not sdpa.cudnn_excluded("cpu")
    assert not sdpa.cudnn_excluded("mps")


def test_rocm_keeps_torchs_choice(monkeypatch):
    # ROCm's torch names its devices "cuda" too; the HIP version tells it apart
    monkeypatch.setattr(torch.version, "hip", "7.1.0")
    monkeypatch.delenv(sdpa.CUDNN_ENV, raising=False)
    assert not sdpa.cudnn_excluded("cuda")


def test_the_env_keeps_torchs_choice_on_cuda(cuda_build, monkeypatch):
    monkeypatch.setenv(sdpa.CUDNN_ENV, "1")
    assert not sdpa.cudnn_excluded("cuda")
    monkeypatch.setenv(sdpa.CUDNN_ENV, "0")
    assert sdpa.cudnn_excluded("cuda")


def test_route_turns_the_process_flag_off_for_every_shape(cuda_build, cudnn_flag, capsys):
    # one flag, read by torch's dispatch for every attention call whatever its
    # query rows: decode-shaped (M <= 8) and prefill-shaped alike, since the
    # measured prefill lost with cuDNN too
    assert sdpa.route("cuda") is True
    assert torch.backends.cuda.cudnn_sdp_enabled() is False
    assert sdpa.route("cuda") is True  # idempotent, and says so once
    out = capsys.readouterr().out
    assert out.count("cuDNN SDPA excluded on CUDA") == 1
    # an explicit sdpa_kernel request still reaches cuDNN (sdpa.probe_backends
    # asks each backend by name), and the flag comes back off after it
    from torch.nn.attention import SDPBackend, sdpa_kernel

    with sdpa_kernel([SDPBackend.CUDNN_ATTENTION]):
        assert torch.backends.cuda.cudnn_sdp_enabled() is True
    assert torch.backends.cuda.cudnn_sdp_enabled() is False


@pytest.mark.parametrize("device", ["cpu", "mps"])
def test_route_leaves_other_devices_untouched(cuda_build, cudnn_flag, capsys, device):
    assert sdpa.route(device) is False
    assert torch.backends.cuda.cudnn_sdp_enabled() is True
    assert capsys.readouterr().out == ""


def test_route_leaves_rocm_untouched(cudnn_flag, monkeypatch, capsys):
    monkeypatch.setattr(torch.version, "hip", "7.1.0")
    assert sdpa.route("cuda") is False
    assert torch.backends.cuda.cudnn_sdp_enabled() is True
    assert capsys.readouterr().out == ""


def test_route_kernels_routes_attention_and_keeps_its_record(monkeypatch):
    seen = []
    monkeypatch.setattr(kernel_route.sdpa, "route", lambda device: seen.append(device) or True)
    monkeypatch.setattr(kernel_route, "_install_narrow", lambda model, device: (True, 0))
    monkeypatch.setattr(kernel_route, "_install_stock_gemv", lambda model, device: ("linear", "stock"))
    monkeypatch.setattr(kernel_route.deltanet, "route", lambda model, device: "none")
    monkeypatch.setattr(kernel_route.deltanet_conv, "route", lambda model, device: "none")
    rec = kernel_route.route_kernels(object(), "cuda")
    assert seen == ["cuda"]
    # the record composes the published engine string: no attention key
    assert set(rec) == {"deltanet_kernel", "deltanet_conv", "narrow_gemv", "narrow_count",
                        "stock_gemv", "raw_gemv"}
