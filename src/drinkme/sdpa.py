"""Which SDPA backend this machine will actually run — and what it costs when the
answer is "math".

Qwen3.8-27B on a Strix Halo (gfx1151) box answers a 25,057-token prompt with
`OutOfMemoryError: ... Tried to allocate 56.13 GiB` (measured),
and `[1, 24, 25057, 25057]` fp32 is 56.13 GiB — the number in the error
IS the attention score matrix. On this TheRock ROCm gfx1151 build torch has no
efficient SDPA backend available, so attention resolves to the MATH kernel,
which materializes that whole matrix in fp32: O(T^2) in MEMORY, not just time.
Nothing in the error points at attention, and nothing suggests that an AMD
environment variable is the fix.

The mechanism, read out of upstream `sdp_utils.cpp` (the hip build hipifies the
cuda file): torch asks `aotriton::isArchExperimentallySupported(stream)` and,
for archs AOTriton ITSELF classifies experimental, demands an opt-in —
TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1. AOTriton ships real gfx1151 kernels
(457 MB of `amd-gfx11xx` images on this disk; the arch is named in compiled
kernel filenames like `bf16@16_128_0_F_F_0___gfx1151.cc`). So this is
"supported, classified experimental" — NOT unsupported hardware, and NOT
kernels built for some other chip.

This module never sets that variable; setting it is left to the operator.
It measures instead. `probe_backends()` asks the machine which backends actually
run at the model's real head shape — T=128, so it costs microseconds and no
memory worth naming — and `self_test()`, which runs only when the flag IS set,
compares the efficient kernel against MATH at the model's real head_dim before
a token is served. drinkme's posture everywhere else is verify-on-the-box-in-
front-of-you rather than assert; this is that, pointed at a kernel someone else
flagged experimental.

Everything here degrades rather than raises: a CPU-only laptop, a torch too old
to have `torch.nn.attention`, no torch at all — each returns a report saying so.
A probe that can crash a server start is worse than no probe.
"""

from __future__ import annotations

import math
import os
import sys
import warnings
from dataclasses import dataclass, field

EXPERIMENTAL_ENV = "TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"

# Our names for torch's SDPBackend members, in the order we would rather have
# them. "math" is last because it is the fallback whose cost this file exists
# to explain.
BACKENDS = ("flash", "mem_efficient", "cudnn", "math")
EFFICIENT = ("flash", "mem_efficient", "cudnn")

# Upstream caps the whole gfx11xx family at head_dim 256 on AOTriton 0.11/0.12
# ("gfx11xx only support hdim <= 256"). Qwen3.8-27B is EXACTLY 256 — inside the
# boundary, sitting on it. A model with a larger head_dim falls back to math on
# this machine no matter what the flag says, which is worth saying out loud rather
# than letting someone conclude the flag is broken.
GFX11_HEAD_DIM_CAP = 256

# The MATH fallback's failing allocation is one fp32 [1, H, T, T] tensor: that
# is the number the OOM prints, and bench/sdpa_backend_sweep.py confirms it by
# reading the largest allocated block back out of the allocator (1,536 MiB at
# T=4096/24 heads — exactly the fp32 score matrix). Peak is higher, 2.6x at
# that T, because softmax needs more than one copy. We quote the score matrix
# rather than the peak because it is the tensor the allocator names when it
# refuses.
FP32 = 4

# bf16 carries 8 mantissa bits, so one ulp at magnitude m is m * 2^-8.
BF16_EPS = 2.0**-8
# Tolerance for "the experimental kernel agrees with the reference". Measured on
# gfx1151/TheRock-7.13 at head_dim 256, 24q/4kv, causal bf16: max|diff| is
# 1.562e-2 at every T from 128 to 4096 — one ulp at magnitude 4, which is where
# these outputs top out — and mean|diff| runs 5.6e-5 (T=4096) to 2.1e-4 (T=128).
# 8 ulps of the reference's own peak magnitude leaves ~8x headroom over that,
# and one bf16 eps on the mean leaves ~20x. Wide enough that ordinary
# algorithmic disagreement passes; narrow enough that a kernel returning garbage
# for this arch cannot.
MAX_ULPS = 8
MEAN_TOL = BF16_EPS

# The loud blocks wrap at 68 so that with the "[drinkme] " prefix they still
# fit an 80-column journal without the rule wrapping onto its own second line.
RULE = "=" * 68


@dataclass(frozen=True)
class AttnShape:
    """The three numbers that decide which attention kernels exist for a model."""

    head_dim: int
    n_heads: int
    n_kv_heads: int

    def __str__(self) -> str:
        return (f"head_dim {self.head_dim}, {self.n_heads} q heads / "
                f"{self.n_kv_heads} kv")


# Layer kinds that run no softmax attention, so no SDPA call: Qwen3.5's
# gated DeltaNet ("linear_attention"), and the mamba and conv mixers of
# other hybrids. Their per-layer head fields describe nothing SDPA runs.
_NO_SDPA_KINDS = ("linear_attention", "mamba", "conv")


def shape_from_config(cfg) -> AttnShape | None:
    """The first attention layer's shape (shapes_from_config has every
    distinct one), or None rather than raising when the config is not one
    we recognize — the caller's job is to say something useful about
    attention, and having nothing to say is a legitimate outcome.
    """
    shapes = shapes_from_config(cfg)
    return shapes[0][0] if shapes else None


def shapes_from_config(cfg) -> list[tuple[AttnShape, str]]:
    """Every distinct attention shape off a transformers config (or a plain
    dict), in layer order, each with the layers it covers: `""` when every
    attention layer shares one shape, else e.g. "10 of 60 layers
    (full_attention)". Empty rather than raising when the config is not one
    we recognize.

    gemma-4-31B's sliding layers are head_dim 256 with 16 kv heads and its
    full layers head_dim 512 with 4. transformers 5 refuses to read such a
    per-layer field off the whole config (`head_dim` raises
    AmbiguousGlobalPerLayerAttributeError), so a config with
    `per_layer_config` is read one layer at a time. A plain config.json
    dict spells the full layers' fields `global_head_dim` and
    `num_global_key_value_heads`, beside `layer_types`.
    """
    if cfg is None:
        return []
    if isinstance(cfg, dict):
        tcfg = cfg.get("text_config") if isinstance(cfg.get("text_config"), dict) else cfg
        getters = _dict_layer_getters(tcfg)
        kinds = tcfg.get("layer_types")
    else:
        try:
            tcfg = cfg.get_text_config() if hasattr(cfg, "get_text_config") else cfg
            getters = _object_layer_getters(tcfg)
            kinds = getattr(tcfg, "layer_types", None)
        except Exception:  # noqa: BLE001 — an unreadable config says nothing
            return []
    if not isinstance(kinds, (list, tuple)) or len(kinds) != len(getters):
        kinds = [None] * len(getters)
    layers = [(g, k) for g, k in zip(getters, kinds) if k not in _NO_SDPA_KINDS]
    layers = layers or list(zip(getters, kinds))
    seen: dict[AttnShape, list] = {}
    for g, k in layers:
        try:
            shape = _shape(g)
        except Exception:  # noqa: BLE001 — as unrecognized, never a crash
            shape = None
        if shape is None:
            return []
        seen.setdefault(shape, []).append(k)
    if len(seen) == 1:
        return [(next(iter(seen)), "")]
    total = sum(len(v) for v in seen.values())
    out = []
    for shape, ks in seen.items():
        names = sorted({k for k in ks if isinstance(k, str)})
        label = f"{len(ks)} of {total} layers" + (f" ({', '.join(names)})" if names else "")
        out.append((shape, label))
    return out


def _object_layer_getters(tcfg) -> list:
    """One field reader per layer: per_layer_config's views when the config
    has them, else the whole config once."""
    plc = getattr(tcfg, "per_layer_config", None)
    views = list(plc) if plc is not None else []
    if not views:
        views = [tcfg]

    def reader(view):
        return lambda key, default=None: getattr(view, key, default)

    return [reader(v) for v in views]


def _dict_layer_getters(d: dict) -> list:
    """config.json's per-layer spelling, as transformers' Gemma4TextConfig
    reads it: a `full_attention` layer takes `global_head_dim`, and
    `num_global_key_value_heads` when `attention_k_eq_v` is set or absent;
    every other layer the plain fields."""
    kinds = d.get("layer_types")
    if not isinstance(kinds, list) or "global_head_dim" not in d:
        return [d.get]
    global_kv = d.get("attention_k_eq_v", True)

    def full(key, default=None):
        if key == "head_dim":
            return d.get("global_head_dim") or d.get("head_dim", default)
        if key == "num_key_value_heads" and global_kv:
            return d.get("num_global_key_value_heads") or d.get(key, default)
        return d.get(key, default)

    return [full if k == "full_attention" else d.get for k in kinds]


def _shape(get) -> AttnShape | None:
    n_heads = get("num_attention_heads")
    if not n_heads:
        return None
    head_dim = get("head_dim") or 0
    if not head_dim:
        hidden = get("hidden_size") or 0
        if not hidden:
            return None
        head_dim = hidden // int(n_heads)
    n_kv = get("num_key_value_heads") or n_heads
    try:
        return AttnShape(int(head_dim), int(n_heads), int(n_kv))
    except (TypeError, ValueError):
        return None


def score_matrix_bytes(n_heads: int, seq_len: int, itemsize: int = FP32) -> int:
    """`[1, n_heads, T, T]` — the tensor the MATH kernel materializes, and the
    one the OOM names. 24 heads at T=25,057 in fp32 is 56.13 GiB, exactly
    the allocation the module docstring's OOM asks for."""
    return n_heads * seq_len * seq_len * itemsize


def max_prompt_before(n_heads: int, budget_bytes: int, itemsize: int = FP32) -> int:
    """The prompt length at which the score matrix alone eats `budget_bytes`.
    Inverse of score_matrix_bytes; the honest ceiling is lower still, because
    weights and KV are already resident and peak attention runs ~2.6x the score
    matrix at the T where it matters."""
    if n_heads <= 0 or budget_bytes <= 0:
        return 0
    return int(math.isqrt(budget_bytes // (n_heads * itemsize)))


def flag_value() -> str | None:
    """Whatever TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL is set to, or None.

    torch reads it with `c10::utils::check_env(...) == true`, which accepts
    1/y/yes/true/on case-insensitively and WARNS on anything else, so we report
    the raw value rather than pretending to parse it."""
    return os.environ.get(EXPERIMENTAL_ENV)


def flag_is_set() -> bool:
    v = flag_value()
    return v is not None and v.strip().lower() in ("1", "y", "yes", "true", "on")


@dataclass
class SdpaReport:
    """Structured, printable-by-someone-else. `announce()` is the only thing
    here that writes to a stream."""

    shape: AttnShape
    device: str = "cpu"
    backends: dict[str, bool] = field(default_factory=dict)
    errors: dict[str, str] = field(default_factory=dict)
    # transformers passes enable_gqa=True on the prefill path when the mask is
    # None and head_dim <= 256 (integrations/sdpa_attention.py:use_gqa_in_sdpa),
    # so that is what we probe. It is NOT a neutral detail: on this machine, with
    # the flag on, enable_gqa=True leaves mem_efficient unavailable while flash
    # still runs. `backends_repeat_kv` is the same probe with the kv heads
    # expanded — transformers' other path, taken when a float mask is present.
    gqa: bool = False
    backends_repeat_kv: dict[str, bool] | None = None
    # The layers this shape covers when a model's layers differ
    # (shapes_from_config's label), "" when they all share it.
    layers: str = ""
    flag: str | None = None
    arch: str | None = None
    torch_version: str | None = None
    total_memory: int | None = None
    probed: bool = True
    note: str = ""

    @property
    def efficient(self) -> list[str]:
        return [b for b in EFFICIENT if self.backends.get(b)]

    @property
    def has_efficient(self) -> bool:
        return bool(self.efficient)

    @property
    def flag_set(self) -> bool:
        v = self.flag
        return v is not None and v.strip().lower() in ("1", "y", "yes", "true", "on")

    @property
    def gfx11_head_dim_capped(self) -> bool:
        """True when this arch's own cap, not the flag, is what excludes us."""
        return bool(self.arch and self.arch.startswith("gfx11")
                    and self.shape.head_dim > GFX11_HEAD_DIM_CAP)

    def to_dict(self) -> dict:
        return {
            "device": self.device,
            "head_dim": self.shape.head_dim,
            "n_heads": self.shape.n_heads,
            "n_kv_heads": self.shape.n_kv_heads,
            "layers": self.layers,
            "backends": dict(self.backends),
            "backends_repeat_kv": self.backends_repeat_kv,
            "errors": dict(self.errors),
            "enable_gqa": self.gqa,
            "efficient": self.efficient,
            "has_efficient": self.has_efficient,
            EXPERIMENTAL_ENV: self.flag,
            "flag_set": self.flag_set,
            "arch": self.arch,
            "torch_version": self.torch_version,
            "total_memory": self.total_memory,
            "probed": self.probed,
            "note": self.note,
        }


def _backend_enums():
    """torch's SDPBackend members, or None on a torch without the API. Import
    is lazy and failure is a return value: `drinkme bench` must keep working on
    a machine where torch is not installed at all (probe.py's rule, same reason)."""
    try:
        from torch.nn.attention import SDPBackend
    except Exception:  # noqa: BLE001 — no torch, old torch, broken install
        return None
    try:
        return {
            "flash": SDPBackend.FLASH_ATTENTION,
            "mem_efficient": SDPBackend.EFFICIENT_ATTENTION,
            "cudnn": SDPBackend.CUDNN_ATTENTION,
            "math": SDPBackend.MATH,
        }
    except AttributeError:  # a torch that has the module but not every member
        return None


def _uses_gqa(shape: AttnShape) -> bool:
    """Mirror transformers' `use_gqa_in_sdpa` for the prefill call: grouped kv,
    no attention mask, head_dim <= 256. Probing the configuration the model will
    not use would answer a question nobody asked."""
    return shape.n_kv_heads != shape.n_heads and shape.head_dim <= 256


def _try_backend(name, enums, shape: AttnShape, device: str, seq_len: int,
                 dtype, gqa: bool) -> tuple[bool, str]:
    import torch
    import torch.nn.functional as F
    from torch.nn.attention import sdpa_kernel

    kv = shape.n_kv_heads if gqa else shape.n_heads
    kw = {"enable_gqa": True} if gqa and kv != shape.n_heads else {}
    try:
        q = torch.zeros(1, shape.n_heads, seq_len, shape.head_dim,
                        device=device, dtype=dtype)
        k = torch.zeros(1, kv, seq_len, shape.head_dim, device=device, dtype=dtype)
        v = torch.zeros_like(k)
        with warnings.catch_warnings():
            # torch TORCH_WARN_ONCEs the very advice this module is about to
            # print properly; two copies of it help nobody.
            warnings.simplefilter("ignore")
            with sdpa_kernel([enums[name]]):
                out = F.scaled_dot_product_attention(q, k, v, is_causal=True, **kw)
        if device == "cuda":
            torch.cuda.synchronize()
        del q, k, v, out
        return True, ""
    except Exception as e:  # noqa: BLE001 — "it raised" IS the answer
        return False, f"{type(e).__name__}: {str(e).splitlines()[0][:160]}"


def probe_backends(shape: AttnShape, device: str | None = None,
                   seq_len: int = 128, dtype=None) -> SdpaReport:
    """Ask the machine, one backend at a time, which SDPA kernels actually run at
    this model's head shape.

    T=128 is plenty: availability is decided by arch, dtype and head_dim, never
    by sequence length. This runs at every server start, so it must cost neither
    real memory nor real time — measured on Qwen3.8-27B's shape, 133 ms and a
    56 MiB peak, and that peak is the caching allocator's segment granularity
    rather than the tensors, which come to about 3 MB.
    """
    enums = _backend_enums()
    if enums is None:
        return SdpaReport(shape=shape, probed=False,
                          note="torch.nn.attention.SDPBackend unavailable "
                               "(no torch, or a torch older than 2.3)")
    import torch

    rep = SdpaReport(shape=shape, torch_version=torch.__version__,
                     flag=flag_value())
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    rep.device = device
    if device == "cuda":
        try:
            props = torch.cuda.get_device_properties(0)
            rep.arch = getattr(props, "gcnArchName", None) or props.name
            rep.total_memory = int(props.total_memory)
        except Exception:  # noqa: BLE001 — a name is a nicety, not a gate
            pass
    if dtype is None:
        # bf16 is what both arms serve; fp32 on CPU, where bf16 kernels are a
        # different (and irrelevant) availability question.
        dtype = torch.bfloat16 if device == "cuda" else torch.float32

    rep.gqa = _uses_gqa(shape)
    for name in BACKENDS:
        ok, err = _try_backend(name, enums, shape, device, seq_len, dtype, rep.gqa)
        rep.backends[name] = ok
        if not ok:
            rep.errors[name] = err
    if rep.gqa:
        rep.backends_repeat_kv = {}
        for name in BACKENDS:
            ok, _ = _try_backend(name, enums, shape, device, seq_len, dtype, False)
            rep.backends_repeat_kv[name] = ok
    return rep


@dataclass
class SelfTest:
    """What the efficient kernel and the math kernel actually agree to, on this
    machine, at this model's shape."""

    backend: str
    seq_len: int
    max_abs: float
    mean_abs: float
    ref_max: float
    tol_max: float
    tol_mean: float
    ok: bool

    def line(self) -> str:
        return (f"max|diff| {self.max_abs:.2e}  mean {self.mean_abs:.2e}  "
                f"(tolerances {self.tol_max:.2e} / {self.tol_mean:.2e})")

    def to_dict(self) -> dict:
        return {
            "backend": self.backend, "seq_len": self.seq_len,
            "max_abs": self.max_abs, "mean_abs": self.mean_abs,
            "ref_max": self.ref_max, "tol_max": self.tol_max,
            "tol_mean": self.tol_mean, "ok": self.ok,
        }


# The MATH reference is the expensive half of the self-test — it is the
# quadratic kernel, run on purpose. Cap the fp32 score matrix it materializes,
# and cap T on top of that, so a self-test can never be the thing that OOMs a
# load.
SELFTEST_BUDGET = 512 * 1024**2
SELFTEST_MAX_T = 1024


def selftest_seq_len(n_heads: int, free_bytes: int | None = None) -> int:
    """Largest T whose fp32 score matrix fits the self-test budget (and a
    quarter of free device memory, when we know it), rounded down to a multiple
    of 64, floor 128."""
    budget = SELFTEST_BUDGET
    if free_bytes:
        budget = min(budget, max(free_bytes // 4, 0))
    t = max_prompt_before(n_heads, budget)
    t = min(t, SELFTEST_MAX_T)
    return max(128, (t // 64) * 64)


def self_test(shape: AttnShape, backend: str, device: str = "cuda",
              seq_len: int | None = None, dtype=None) -> SelfTest | None:
    """Run the efficient kernel and MATH over identical inputs at the model's
    REAL head_dim and compare. None when the comparison cannot be made.

    Tolerances are derived, not eyeballed: MAX_ULPS ulps of the reference's own
    peak magnitude, and one bf16 eps on the mean. See the constants above for
    the measured numbers they were sized against.
    """
    enums = _backend_enums()
    if enums is None or backend not in enums or backend == "math":
        return None
    import torch
    import torch.nn.functional as F
    from torch.nn.attention import sdpa_kernel

    if dtype is None:
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
    if seq_len is None:
        free = None
        if device == "cuda":
            try:
                free = int(torch.cuda.mem_get_info()[0])
            except Exception:  # noqa: BLE001
                free = None
        seq_len = selftest_seq_len(shape.n_heads, free)

    gqa = _uses_gqa(shape)
    kv = shape.n_kv_heads if gqa else shape.n_heads
    kw = {"enable_gqa": True} if gqa and kv != shape.n_heads else {}

    def run(name):
        # Same seed for both arms: this compares KERNELS, so the inputs have to
        # be bit-identical, not merely identically distributed.
        gen = torch.Generator(device=device).manual_seed(20260824)
        q = torch.randn(1, shape.n_heads, seq_len, shape.head_dim, generator=gen,
                        device=device, dtype=dtype)
        k = torch.randn(1, kv, seq_len, shape.head_dim, generator=gen,
                        device=device, dtype=dtype)
        v = torch.randn(1, kv, seq_len, shape.head_dim, generator=gen,
                        device=device, dtype=dtype)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with sdpa_kernel([enums[name]]):
                out = F.scaled_dot_product_attention(q, k, v, is_causal=True, **kw)
        del q, k, v
        return out.float()

    try:
        ref = run("math")
        got = run(backend)
    except Exception:  # noqa: BLE001 — an unrunnable comparison is not a verdict
        return None
    diff = (got - ref).abs()
    max_abs = float(diff.max())
    mean_abs = float(diff.mean())
    ref_max = float(ref.abs().max())
    del ref, got, diff
    if device == "cuda":
        torch.cuda.empty_cache()
    tol_max = MAX_ULPS * BF16_EPS * max(1.0, ref_max)
    return SelfTest(backend=backend, seq_len=seq_len, max_abs=max_abs,
                    mean_abs=mean_abs, ref_max=ref_max, tol_max=tol_max,
                    tol_mean=MEAN_TOL, ok=max_abs <= tol_max and mean_abs <= MEAN_TOL)


def _gib(n: float) -> str:
    return f"{n / 1024**3:,.1f} GiB"


def warning_lines(rep: SdpaReport, ctx: int = 0) -> list[str]:
    """The warning block: what falls back, what it costs at this model's head
    count, and the one line that fixes it. Returned rather than printed so the
    tests can read it and so a caller can route it wherever it needs to go."""
    s = rep.shape
    out = [
        RULE,
        "No efficient attention backend — long prompts will OOM.",
        f"  {rep.arch or rep.device}, torch {rep.torch_version}",
        "  Every SDPA kernel but MATH is unavailable at this model's shape",
        f"  ({s}). MATH materializes the whole",
        f"  [1, {s.n_heads}, T, T] score matrix in fp32, so prompt memory grows",
        "  quadratically, not linearly:",
    ]
    if rep.layers:
        out[4:5] = [f"  ({s}),", f"  on {rep.layers}. MATH materializes the whole"]
    ladder = [t for t in (8192, 25000, 65536) if not ctx or t <= ctx] or [ctx]
    for t in ladder:
        out.append(f"      {t:>7,} tokens ->"
                   f"{_gib(score_matrix_bytes(s.n_heads, t)):>12}")
    wall = 0
    if rep.total_memory:
        wall = max_prompt_before(s.n_heads, rep.total_memory)
        out += [
            f"  Past ~{wall:,} tokens that one tensor outgrows this device's",
            f"  {_gib(rep.total_memory)} on its own — before weights, before KV,"
            " and peak",
            "  attention runs ~2.6x the score matrix. The ceiling is lower.",
        ]
    if ctx:
        out.append(f"  Context window this server advertises: {ctx:,} tokens"
                   + (" — most of it" if wall and wall < ctx else "."))
        if wall and wall < ctx:
            out.append("  unreachable until an efficient backend is available.")
    if rep.flag_set:
        # The flag is on and it did not help. Say which of the two reasons.
        if rep.gfx11_head_dim_capped:
            these, uses = ("these layers are", "these layers use") if rep.layers else (
                "this model is", "this model uses")
            out += [
                f"  {EXPERIMENTAL_ENV}",
                f"  is already set (={rep.flag}) and changed nothing: AOTriton "
                "caps the",
                f"  gfx11xx family at head_dim {GFX11_HEAD_DIM_CAP} and {these} "
                f"{s.head_dim}. No other setting",
                f"  changes that, so {uses} MATH on this device.",
            ]
        else:
            out += [
                f"  {EXPERIMENTAL_ENV}",
                f"  is already set (={rep.flag}) and made no efficient backend "
                "available.",
                "  The gate is AOTriton's own arch classification, so this "
                "device may",
                "  simply have no kernel for this shape.",
            ]
    else:
        out += [
            "  On AMD/ROCm, set this in the environment before the process",
            "  starts (torch reads it inside the attention dispatch; exporting",
            "  it afterwards does nothing):",
            "",
            f"      {EXPERIMENTAL_ENV}=1 drinkme serve",
            "",
            "  This is AOTriton's documented opt-in for architectures it",
            "  classifies as experimental; the gfx1151 kernels already ship in",
            "  the AOTriton images torch bundles. drinkme does not set it for",
            "  you; docs/rocm.md has the systemd drop-in.",
        ]
    out.append(RULE)
    return out


def selftest_lines(rep: SdpaReport, st: SelfTest) -> list[str]:
    s = rep.shape
    head = (f"self-test: {st.backend} vs math @ {s}, T={st.seq_len}, causal "
            f"{'bf16' if rep.device == 'cuda' else 'fp32'}")
    if st.ok:
        return [f"SDPA {head}", f"  {st.line()} — AGREES"]
    return [
        RULE,
        f"SDPA self-test failed — {st.backend} disagrees with math.",
        f"  {s}, T={st.seq_len}, causal, on {rep.arch or rep.device}",
        f"  {st.line()}",
        f"  {EXPERIMENTAL_ENV}={rep.flag}",
        "  enables a kernel AOTriton classifies as experimental for this arch,",
        "  and at this model's shape its output does not match the MATH",
        "  reference within tolerance.",
        "  Serving anyway, because you set the flag. Unset it to use the MATH",
        "  reference: slower, and its memory grows quadratically with prompt",
        "  length.",
        RULE,
    ]


def report_lines(rep: SdpaReport, ctx: int = 0,
                 st: SelfTest | None = None) -> list[str]:
    """Everything announce() would print, as text. Three shapes: no efficient
    backend (loud), flag set and an efficient backend present (self-test), and
    the quiet one-liner for a machine where this was never a problem."""
    if not rep.probed:
        return [f"SDPA: not probed — {rep.note}"]
    if not rep.has_efficient:
        return warning_lines(rep, ctx)
    avail = ", ".join(rep.efficient)
    why = f" ({EXPERIMENTAL_ENV}={rep.flag})" if rep.flag_set else ""
    where = f" at {rep.shape}, on {rep.layers}" if rep.layers else ""
    lines = [f"SDPA: {avail} available{where}{why} — attention is O(T) in memory."]
    if st is not None:
        lines += selftest_lines(rep, st)
    return lines


def _say(line: str) -> None:
    # flush, always: print() block-buffers into a pipe, so under systemd an
    # unflushed line can stay out of the journal for a whole boot, written,
    # correct and invisible.
    stream = sys.stdout
    print(f"[drinkme] {line}" if line else "[drinkme]", file=stream, flush=True)


def announce(rep: SdpaReport, ctx: int = 0, emit=_say) -> SelfTest | None:
    """Print the verdict, running the self-test first when the flag is set.

    The self-test is gated on the flag deliberately: it is the answer to "you
    opted in to an experimental kernel — does it work here?", and a machine that
    never had to opt in has nothing to verify.
    """
    st = None
    if rep.probed and rep.has_efficient and rep.flag_set:
        st = self_test(rep.shape, rep.efficient[0], device=rep.device)
    for line in report_lines(rep, ctx, st):
        emit(line)
    return st


def announce_for_engine(engine, device: str, emit=_say) -> list[SdpaReport] | None:
    """The serve-path entry point: one probe and one verdict per distinct
    attention shape (shapes_from_config), so a model whose full layers are
    wider than its sliding ones (gemma-4: 512 against 256, past AOTriton's
    gfx11xx cap) gets the MATH warning for exactly those layers. Silent —
    and harmless — when the engine has no config to read a head shape from
    (FakeEngine, the wiring tests); there is nothing honest to say about
    attention without one."""
    cfg = getattr(getattr(engine, "model", None), "config", None)
    shapes = shapes_from_config(cfg)
    if not shapes:
        return None
    reps = []
    for shape, layers in shapes:
        rep = probe_backends(shape, device=device)
        rep.layers = layers
        announce(rep, ctx=int(getattr(engine, "ctx", 0) or 0), emit=emit)
        reps.append(rep)
        if not rep.probed:  # no torch API: one "not probed" line says it all
            break
    return reps


# ------------------------------------------------ the backend drinkme runs --

# DRINKME_CUDNN_SDPA=1 keeps torch's own choice on CUDA, cuDNN included: the
# A/B instrument for the route below (docs/serve-kernels.md).
CUDNN_ENV = "DRINKME_CUDNN_SDPA"
_routed: list = []


def cudnn_excluded(device) -> bool:
    """Whether drinkme's attention runs without torch's cuDNN SDPA backend on
    `device`: on CUDA (NVIDIA), for every attention shape, decode and
    prefill alike; never on ROCm, Metal or the CPU, which keep torch's own
    choice. DRINKME_CUDNN_SDPA=1 keeps torch's choice on CUDA too.

    Why, measured on an H100 (sm90, torch 2.13): torch picks cuDNN attention there, and cuDNN builds an
    execution plan for every new sequence length, cached per thread.
    `drinkme serve` runs each request on a fresh HTTP handler thread, so an
    eager decode paid a build every step (every step is a new length):
    Qwen3-8B served eager (DRINKME_CUDA_GRAPHS=0) at 21-41 tok/s, against
    71-80 with cuDNN excluded, same text; the MTP head's attention paid
    21 ms a call on a fresh thread, and Qwen3.8-27B served
    in graph mode at 18.8 tok/s with the head back on cuDNN against 68.4
    without. A persistent thread would not fix decode: the lengths
    grow past anything cached. Prefill on a fresh thread, torch's choice
    against cuDNN excluded (serve's engine, max_tokens 1, median of three):
    Qwen3-8B 104 / 57 ms at 931 prompt tokens, 195 / 161 ms at 3,848,
    1,146 / 1,067 ms at 16,022; Qwen3.8-27B 392 / 323, 943 / 689 and
    2,966 / 2,862 ms at 980, 4,044 and 16,803 (bench/serve_fixes.py
    --what ttft). cuDNN lost at every length, so it is excluded there too.
    Flash, then mem-efficient, then math take any length without a build.
    On an L4 torch offers cuDNN attention but picks flash at Qwen3-8B's
    decode and prefill shapes, so nothing changes there (served rates and
    prefill within 1% either way)."""
    import torch

    if torch.device(device).type != "cuda" or getattr(torch.version, "hip", None):
        return False
    return os.environ.get(CUDNN_ENV) != "1"


def route(device) -> bool:
    """Apply `cudnn_excluded` for this process: torch's cuDNN SDPA flag off
    (torch.backends.cuda.enable_cudnn_sdp), which every attention call reads
    at dispatch, on every thread. serving/kernel_route.route_kernels calls
    this for every loaded model, so serve and every bench arm run the same
    attention; one line says so, once per process. Returns whether cuDNN
    is excluded. Elsewhere it touches nothing."""
    import torch

    if not cudnn_excluded(device):
        return False
    torch.backends.cuda.enable_cudnn_sdp(False)
    if not _routed:
        _routed.append(True)
        _say("attention: cuDNN SDPA excluded on CUDA (flash, then mem-efficient, then math): "
             f"it builds a plan per new sequence length on every thread; {CUDNN_ENV}=1 for "
             "torch's choice")
    return True


if __name__ == "__main__":
    # `python -m drinkme.sdpa [head_dim [n_heads [n_kv_heads]]]` — defaults are
    # Qwen3.8-27B's full-attention layers. This is the cheap test from the
    # README: run it WITHOUT the env var, and if an efficient backend answers,
    # AOTriton has promoted this arch and the flag can go.
    import json

    argv = sys.argv[1:]
    dims = [int(a) for a in argv[:3]]
    while len(dims) < 3:
        dims.append([256, 24, 4][len(dims)])
    report = probe_backends(AttnShape(dims[0], dims[1], dims[2]))
    out = report.to_dict()
    if report.probed and report.has_efficient:
        test = self_test(report.shape, report.efficient[0], device=report.device)
        out["self_test"] = test.to_dict() if test else None
    print(json.dumps(out, indent=2))
