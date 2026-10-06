"""The DeltaNet decode recurrence's kernel route.

WHAT MISSES. transformers' `Qwen3_5GatedDeltaNet.forward` calls
`torch_recurrent_gated_delta_rule` for the cached M=1 decode step. That name
is decorated with `use_kernel_func_from_hub_with_fallback(
"recurrent_gated_delta_rule", "fla")`, which — once, at IMPORT of the
modeling module — looks for `fla.ops.gated_delta_rule.recurrent_gated_delta_rule`
and keeps the pure-torch loop when the attribute is absent. fla 0.5.2
exports `fused_recurrent_gated_delta_rule` (and `naive_…`), not that name,
so on this stack the fallback runs: ~60 torch launches per DeltaNet layer
(`S * g`, `sum(-2)`, `S + k⊗δ`, casts, transposes), 14.1 ms of the 27B's
210 ms serial step, and 202 launches / ~36 ms per M=5 MTP verify, where
mtp.py replays it once per position. The fused Triton kernel does the same
recurrence in ONE launch per call: it reads and writes the fp32 state once
(48 heads × 128 × 128 × 4 B = 3 MB, ~25 µs at this box's wall). A rocprof
trace on gfx1151 measured the miss; this module routes around it.

HOW. `route(model, device)` rebinds the modeling module's GLOBAL
`torch_recurrent_gated_delta_rule` — the name both the installed forward and
mtp.py's per-position replay look up at CALL time — to an adapter over
fla's fused kernel. Rebinding the global rather than registering the fla
alias first is deliberate: the decorator resolves at import, and nothing
in this tree controls whether `modeling_qwen3_5` was imported before the
loader ran (tests do; a second engine in one process does). The adapter
maps the forward's call — positional q/k/v `[B, T, H, K]`/`[B, T, HV, V]`,
keyword `g` `[B, T, HV]` fp32, `beta` `[B, T, HV]`, `initial_state`
`[B, HV, K, V]` fp32, `output_final_state`, `use_qk_l2norm_in_kernel`,
`cu_seqlens` — onto fla's, which takes the same tensors under the same
names in the same layout (fla's `input_guard` makes them contiguous), and
drops the TransformersKwargs the forward passes through. Output: bf16
`[B, T, HV, V]` and a NEW fp32 final state, exactly the torch reference's
shapes and dtypes; neither implementation writes `initial_state` in place.

WHAT CHANGES, NUMERICALLY. The recurrence's arithmetic order: the torch
reference L2-normalizes q and k in the input dtype (bf16 on the served
27B) before casting to fp32, and reduces `S·k` and `S·q` as torch `sum`
kernels; the fused kernel normalizes and reduces in fp32 registers. Same
distribution, a different rounding — the greedy transcript of a near-tie
can flip (AGENTS.md: the invariant is bit-identical WEIGHTS). Serial decode
and the MTP replay still run the same recurrence: both paths call the one
function bound here.

THE KNOB. `DRINKME_DELTANET_KERNEL=fla|torch`; unset = fla when the device
is an accelerator and the kernel passes a probe (below), torch otherwise.
The probe runs the fused kernel ONCE at route time on the model's own head
shape in fp32 and compares it with the torch reference: a Triton compile
failure on this device, or a result that disagrees beyond fp32
reduction-order noise, keeps the torch reference and says why — loudly,
because a silently-different recurrence would serve a different model.
CPU is torch always (Triton kernels do not run on CPU tensors); asking for
fla there is refused by name. One boot-log line says which ran.
"""

from __future__ import annotations

import inspect
import os
import sys

from .. import exitcodes

ENV = "DRINKME_DELTANET_KERNEL"
CHOICES = ("fla", "torch")
# the modeling module's name for the M=1 decode recurrence (transformers 5.x
# qwen3_5; qwen3_next spells it the same)
NAME = "torch_recurrent_gated_delta_rule"
# fp32 inputs, unit scale, one step: the two implementations differ by
# reduction order only, ~1e-6; 1e-3 is far above that and far below a
# wrong-kernel result (a mislaid head or a wrong state layout is O(1))
PROBE_ATOL = 1e-3


def _choice() -> str | None:
    raw = os.environ.get(ENV, "").strip().lower()
    if not raw:
        return None
    if raw not in CHOICES:
        raise exitcodes.CantRunHere(f"[drinkme] {ENV}={raw!r} is not one of {'/'.join(CHOICES)}")
    return raw


def deltanet_modules(model) -> list:
    """Every GatedDeltaNet INSTANCE in the tree (the class name is the
    family's own: `Qwen3_5GatedDeltaNet`, `Qwen3NextGatedDeltaNet`)."""
    return [m for m in model.modules() if type(m).__name__.endswith("GatedDeltaNet")]


def _modeling_module(mod):
    return sys.modules[type(mod).__module__]


def torch_reference(modeling):
    """The pure-torch recurrence as the modeling module shipped it — through
    the hub decorator's wrapper, or an adapter this module installed
    earlier (both carry `__wrapped__`; the reference carries none)."""
    fn = getattr(modeling, NAME)
    while getattr(fn, "__wrapped__", None) is not None:
        fn = fn.__wrapped__
    return fn


def make_fla_adapter(fused, reference):
    """The forward's call shape → fla's `fused_recurrent_gated_delta_rule`.
    `reference` rides along as `__wrapped__` so `torch_reference` can find
    it again and tests can tell the two apart by name."""
    accepted = frozenset(inspect.signature(fused).parameters) - {"kwargs"}

    def fla_recurrent_gated_delta_rule(query, key, value, g, beta, initial_state=None,
                                       output_final_state=False,
                                       use_qk_l2norm_in_kernel=False, **kwargs):
        # the forward forwards TransformersKwargs (position_ids, cache_position,
        # …) and fla's own **kwargs knows only a deprecated layout flag: pass
        # what fla names, drop the rest — the hub decorator's own rule
        extra = {k: v for k, v in kwargs.items() if k in accepted}
        return fused(query, key, value, g=g, beta=beta, initial_state=initial_state,
                     output_final_state=output_final_state,
                     use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel, **extra)

    fla_recurrent_gated_delta_rule.__wrapped__ = reference
    fla_recurrent_gated_delta_rule.route = "fla"
    return fla_recurrent_gated_delta_rule


def probe(adapter, reference, mod, device: str) -> tuple[float, float, str]:
    """One decode step of the model's own DeltaNet shape, random inputs,
    through both implementations, twice: in fp32 (the agreement check —
    the two differ by reduction order only) and in the model's own dtype
    (bf16 on the served 27B: the specialization the decode loop will
    launch, compiled here rather than on the first request; its |Δ| is
    reported, not judged — the reference normalizes q/k in bf16, the
    kernel in fp32, see the module docstring). Returns (max |Δ| fp32,
    max |Δ| model dtype, model dtype name). Raises on a compile/launch
    failure or a shape/dtype contract break (the caller catches)."""
    import torch

    hv, k, v = mod.num_v_heads, mod.head_k_dim, mod.head_v_dim
    model_dtype = mod.conv1d.weight.dtype
    gen = torch.Generator(device="cpu").manual_seed(0)
    base = dict(q=torch.randn(1, 1, hv, k, generator=gen), k=torch.randn(1, 1, hv, k, generator=gen),
                v=torch.randn(1, 1, hv, v, generator=gen),
                g=-torch.rand(1, 1, hv, generator=gen),  # log-space decay in (-1, 0]
                beta=torch.rand(1, 1, hv, generator=gen), h0=torch.randn(1, hv, k, v, generator=gen))

    def step(dtype):
        t = {n: x.to(device) for n, x in base.items()}
        q, kk, vv = t["q"].to(dtype), t["k"].to(dtype), t["v"].to(dtype)
        # g is fp32 and the state fp32 on every path (the forward computes g
        # in .float(); the cache holds the chunk kernel's fp32 final state)
        beta = t["beta"].to(dtype)
        args = dict(g=t["g"], beta=beta, initial_state=t["h0"], output_final_state=True,
                    use_qk_l2norm_in_kernel=True)
        o_ref, s_ref = reference(q, kk, vv, **args)
        o_fla, s_fla = adapter(q, kk, vv, **args)
        if o_fla.shape != o_ref.shape or s_fla.shape != s_ref.shape:
            raise RuntimeError(f"shape mismatch: out {tuple(o_fla.shape)} vs {tuple(o_ref.shape)}, "
                               f"state {tuple(s_fla.shape)} vs {tuple(s_ref.shape)}")
        if o_fla.dtype != o_ref.dtype or s_fla.dtype != s_ref.dtype:
            raise RuntimeError(f"dtype mismatch: out {o_fla.dtype} vs {o_ref.dtype}, "
                               f"state {s_fla.dtype} vs {s_ref.dtype}")
        return max((o_fla.float() - o_ref.float()).abs().max().item(),
                   (s_fla - s_ref).abs().max().item())

    d32 = step(torch.float32)
    dm = d32 if model_dtype == torch.float32 else step(model_dtype)
    if device != "cpu":
        torch.cuda.synchronize()
    return d32, dm, str(model_dtype).replace("torch.", "")


def _say(msg: str) -> None:
    print(f"[drinkme] deltanet recurrence: {msg}", flush=True)


def route(model, device: str) -> str:
    """Bind the decode recurrence for THIS process (the modeling module's
    global) and say which one runs. Returns "fla" / "torch"; "none" when the
    model has no DeltaNet layers (nothing said, nothing bound)."""
    mods = deltanet_modules(model)
    if not mods:
        return "none"
    modeling = _modeling_module(mods[0])
    if not hasattr(modeling, NAME):
        raise RuntimeError(f"{modeling.__name__} has no {NAME}: the transformers "
                           "seam this route binds has moved; this transformers version "
                           "is not supported by the DeltaNet route")
    reference = torch_reference(modeling)
    want = _choice()
    on_cpu = str(device).startswith("cpu")

    def use_torch(why: str) -> str:
        setattr(modeling, NAME, reference)
        _say(f"torch reference ({why})")
        return "torch"

    if want == "torch":
        return use_torch(f"{ENV}=torch")
    if on_cpu:
        if want == "fla":
            raise exitcodes.CantRunHere(f"[drinkme] {ENV}=fla on device {device!r}: fla's fused "
                                        "recurrence is a Triton kernel and cannot run on CPU tensors")
        return use_torch(f"device {device}")
    try:
        import fla
        from fla.ops.gated_delta_rule import fused_recurrent_gated_delta_rule as fused
    except Exception as e:  # noqa: BLE001 — the import is the thing being tested
        if want == "fla":
            raise exitcodes.CantRunHere(f"drinkme: {ENV}=fla but fla's fused recurrence cannot be "
                                        f"imported ({type(e).__name__}: {e})") from e
        return use_torch(f"fla import failed: {type(e).__name__}: {e}")
    adapter = make_fla_adapter(fused, reference)
    try:
        d, dm, mdt = probe(adapter, reference, mods[0], device)
    except Exception as e:  # noqa: BLE001 — a compile or launch failure on this device
        msg = f"fla fused kernel failed its probe on {device} ({type(e).__name__}: {e})"
        if want == "fla":
            raise exitcodes.CantRunHere(f"drinkme: {ENV}=fla but the {msg}") from e
        print(f"drinkme: warning: {msg} — decoding on the torch reference",
              file=sys.stderr, flush=True)
        return use_torch("fla probe failed, see the warning above")
    if not d < PROBE_ATOL:  # NaN fails too
        msg = (f"fla fused kernel disagrees with the torch reference on {device}: "
               f"max |Δ| {d:.3e} (limit {PROBE_ATOL:.0e})")
        if want == "fla":
            raise exitcodes.CantRunHere(f"drinkme: {ENV}=fla but the {msg}")
        print(f"drinkme: warning: {msg} — decoding on the torch reference",
              file=sys.stderr, flush=True)
        return use_torch("fla probe disagreed, see the warning above")
    setattr(modeling, NAME, adapter)
    _say(f"fla {getattr(fla, '__version__', '?')} fused_recurrent_gated_delta_rule "
         f"(Triton, one launch per step; probe max |Δ| vs the torch reference "
         f"{d:.1e} fp32, {dm:.1e} {mdt}; {'forced' if want else 'default'} — "
         f"{ENV}=torch for the reference)")
    return "fla"
