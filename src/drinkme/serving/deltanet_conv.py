"""ROCm DeltaNet prefill convolution without the channel-major copy.

The projection already stores [batch, time, channels]. Transformers transposes
that view for torch conv1d, whose contiguous conversion becomes expensive at
long context. FLA's Triton convolution reads the original layout and leaves
the output time-major too. See docs/serve-kernels.md#deltanet-prefill-convolution.

Cache updates stay in the model: its input includes any preceding convolution
state, and its caller crops the output to the new positions. Small MTP windows
and single-step decode retain their existing operations. Weights are untouched.
"""

from __future__ import annotations

import os
import sys

from .. import exitcodes
from .deltanet import deltanet_modules, _modeling_module

ENV = "DRINKME_DELTANET_CONV"
NAME = "causal_conv1d_fn"
MIN_ROWS = 128


def reference(modeling):
    """Recover the upstream callable, preserving its own hub dispatch."""
    fn = getattr(modeling, NAME)
    return getattr(fn, "_drinkme_conv_reference", fn)


def make_adapter(conv, upstream):
    import torch
    import torch.nn.functional as F

    def causal_conv1d_fn(hidden_states, weight, bias=None, activation=None, **kwargs):
        if (hidden_states.device.type != "cuda" or hidden_states.ndim != 3
                or hidden_states.shape[-1] < MIN_ROWS
                or hidden_states.dtype not in (torch.float32, torch.float16, torch.bfloat16)
                or weight.dtype != hidden_states.dtype or weight.device != hidden_states.device
                or activation not in (None, "silu", "swish")):
            return upstream(hidden_states, weight, bias, activation=activation, **kwargs)
        # The upstream conv rounds to the input dtype BEFORE SiLU. Fusing
        # SiLU into the convolution would remove that rounding boundary.
        y, _ = conv(hidden_states.transpose(1, 2), weight, bias=bias,
                    activation=None, output_final_state=False, backend="triton")
        if activation is not None:
            y = F.silu(y)
        return y.transpose(1, 2)

    causal_conv1d_fn._drinkme_conv_reference = upstream
    return causal_conv1d_fn


def probe(adapter, upstream, mod, device):
    """Check the actual convolution weights, including a cropped history view."""
    import torch

    w, bias = mod.conv1d.weight.squeeze(1), mod.conv1d.bias
    gen = torch.Generator(device=device).manual_seed(1729)
    # Five extra rows exercise a history-bearing, non-square-sized window.
    x = torch.randn(1, MIN_ROWS + 5, w.shape[0], generator=gen,
                    device=device, dtype=w.dtype).transpose(1, 2)
    max_diff = 0.0
    with torch.inference_mode():
        for value in (x, x.contiguous()[..., 1:]):
            expected = upstream(value, w, bias, activation=mod.activation)
            actual = adapter(value, w, bias, activation=mod.activation)
            tol = 2e-5 if w.dtype == torch.float32 else 0.016
            torch.testing.assert_close(actual, expected, atol=tol, rtol=tol)
            max_diff = max(max_diff, (actual.float() - expected.float()).abs().max().item())
    return max_diff


def _load_kernel():
    from fla.modules.conv import causal_conv1d

    return causal_conv1d


def route(model, device: str) -> str:
    """Select and probe the prefill route equally for stock and packed models."""
    import torch

    want = os.environ.get(ENV, "").strip().lower()
    if want not in ("", "fla", "torch"):
        raise exitcodes.CantRunHere(f"[drinkme] {ENV}={want!r} is not one of fla/torch")
    mods = deltanet_modules(model)
    if not mods:
        return "none"
    modeling = {_modeling_module(mod) for mod in mods}
    if any(not hasattr(m, NAME) for m in modeling):
        if want == "fla":
            raise exitcodes.CantRunHere(f"[drinkme] {ENV}=fla requires a {NAME} model interface")
        return "torch"
    originals = {m: reference(m) for m in modeling}

    def fallback(why):
        for m, fn in originals.items():
            setattr(m, NAME, fn)
        print(f"[drinkme] deltanet prefill convolution: upstream ({why})", flush=True)
        return "torch"

    if want == "torch":
        return fallback(f"{ENV}=torch")
    if torch.device(device).type != "cuda":
        if want == "fla":
            raise exitcodes.CantRunHere(f"[drinkme] {ENV}=fla requires a CUDA/ROCm device")
        return fallback(f"device {device}")
    # NVIDIA can already have causal-conv1d's CUDA implementation. The
    # measured opportunity here is ROCm's torch fallback.
    if not torch.version.hip and not want:
        return fallback("default on this platform")
    try:
        causal_conv1d = _load_kernel()
        adapters = {m: make_adapter(causal_conv1d, fn) for m, fn in originals.items()}
        checked, max_diff = set(), 0.0
        for mod in mods:
            m = _modeling_module(mod)
            key = (m, tuple(mod.conv1d.weight.shape), mod.conv1d.weight.dtype,
                   mod.conv1d.weight.device, mod.activation, mod.conv1d.bias is not None)
            if key not in checked:
                max_diff = max(max_diff, probe(adapters[m], originals[m], mod, device))
                checked.add(key)
    except Exception as exc:
        why = f"FLA probe failed: {type(exc).__name__}: {exc}"
        if want == "fla":
            raise exitcodes.CantRunHere(f"drinkme: {ENV}=fla but {why}") from exc
        print(f"drinkme: warning: {why}", file=sys.stderr, flush=True)
        return fallback("see probe warning")
    for m, adapter in adapters.items():
        setattr(m, NAME, adapter)
    print(f"[drinkme] deltanet prefill convolution: FLA Triton for >= {MIN_ROWS} rows "
          f"(BF16 rounding before activation retained; probe max |Δ| {max_diff:.2e}; "
          f"{ENV}=torch for upstream)", flush=True)
    return "fla"
