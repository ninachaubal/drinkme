"""The MLX engine: serving/engine.py's Engine over mlx-lm's Qwen3, with every
packed Linear resident and read by a Metal kernel.

The torch engine (engines.py) is one class with two arms; this is the same
shape on the mlx runtime (docs/metal.md): `load_compressed_mlx` builds
mlx-lm's Qwen3 module tree, replaces each Linear the pack carries with a
module holding the PACKED bytes as mx arrays (never the bf16 weight), and
streams everything else — embeddings, norms, the small projections the
codec leaves alone — as bf16 straight from the checkpoint's safetensors,
the same bytes and the same tying rule as engines.load_compressed.
`load_stock_mlx` is the control: the identical class over mlx-lm's own bf16
model, so `--stock` and the bench's stock arm are the Mac community's baseline
through drinkme's server and nothing else changes.

SCOPE: DENSE Qwen3 text models (0.6B / 1.7B / 4B / 8B), packed either way
the format allows and chosen PER TENSOR by the scalars every pack tensor
carries — a radix tensor (the bf16 codec, `codec: "radix"`) is RadixLinear
over metal/gemv_radix.py and
metal/dense_radix.py, gated bit-exact on the M4 (bench/radix_bitpin_mlx.py,
docs/metal.md#the-bf16-codec-on-metal); the raw fallback (`codec: "raw"`, a tensor
radix would have expanded) is RawLinear, the bf16 bits through mlx's own
matmul. A pack of another format version is refused by name before any
tensor is read (codec/pack.py), and so is a pack cut from an FP8 checkpoint
(the one line). The hybrid
DeltaNet 27B (Qwen3.5 / qwen3_next lineage) is refused at load, by name —
its linear attention layers are the torch engine's fla path, and nothing
here speaks them.

TWO COMPUTE PATHS, ONE SWITCH (RadixLinear.path):

  "fused"      — M=1: the Metal GEMV, one dispatch, the decode inside the
                 dot, W never built. M>1 (prefill): the DENSE arm — a
                 transient bf16 weight (one Linear at a time, 2 B/weight,
                 dropped by the graph as soon as the matmul has read it)
                 through mx.matmul, the torch side's prefill arm rather than
                 a row loop. There is no multi-column kernel (registry:
                 (radix, 0, mc, metal) is False); 2 <= M is the dense arm.
  "reference"  — the CPU oracle's decode and the same matmul, on CPU mlx
                 too. This is the ONLY path a machine without Metal can
                 run — the Linux dev machine tests the whole engine through it —
                 and the control the fused path is pinned against.

  The default is "fused" when mx.metal.is_available() and "reference"
  otherwise; DRINKME_MLX_PATH forces either. A forced "fused" on a machine with
  no Metal fails at the first dispatch with mlx's own "No Metal back-end" —
  loud, not slow.

  A third path, "twin" (TWIN), is the bench's twin arm and is only ever
  asked for by name (arms_mlx): RadixTwinLinear, the reference path's
  forward over the pack decoded ONCE at load by metal/twin_radix (plain
  mlx ops, bitwise the oracle) and held as bf16 — the same outputs as
  "reference", bit for bit, without the per-forward CPU decode.

  RadixLinear (the radix codec — metal/gemv_radix.py and metal/dense_radix.py,
  gated bit-exact on the M4 by bench/radix_bitpin_mlx.py) takes the same
  switch: "fused" is gemv_radix at M=1 (x read as bf16 bits, fp32
  accumulate, bias in fp32, one rounding in-kernel) and dense_radix's
  transient bf16 weight through mx.matmul at M>1; "reference" is the CPU
  oracle's decode (radix_pack.decode_back_radix) and the same matmul — a
  correctness twin, hours per token at model level. The bias, where a
  config has one, is added in fp32 before the single cast on the gemv arm
  and through biased_matmul on the dense arm — the torch CompressedLinear's
  epilogue rule (codec/swap.py _epilogue: fp32 accumulate, the bias, ONE
  rounding). NOT mx.addmm at M == 1: mlx's gemv kernel rounds the
  accumulator before it adds the bias (biased_matmul's docstring — the
  measurement behind it, and why a single row takes the fp32 matmul).

  RawLinear (the raw fallback: a tensor radix would have expanded,
  stored as its bf16 bits) has no switch: the resident bf16 weight through
  mlx's own matmul on either path, every M — stock's arithmetic, the same
  as swap.RawLinear's F.linear on the torch engine; with a bias, the same
  biased_matmul epilogue.

  BiasedLinear: an UNPACKED projection whose checkpoint carries a bias
  the skeleton has no slot for (mlx-lm's Qwen3 builds every Linear
  bias=False, whatever config.json's attention_bias says) — the loader's
  swap-in when the bias arrives, both arms, forward = biased_matmul.

WHAT IS NOT HERE (v0, each named so nobody has to discover it):
  - no prefix cache: every request re-prefills. The KV cache is mlx-lm's
    own (mlx_lm.models.cache.KVCache, one per layer, grown in 256-token
    steps), built per generate() and dropped after. engines.py's slot
    machinery — extends-only reuse, LRU slots, the on-disk tier — is not
    on this lane; `cached_tokens` is always 0.
  - no sleep/wake, no persisted slots: the HTTP layer answers 501 for
    /sleep and /wake_up by the seam's optional-method rule, and SIGTERM has
    nothing to persist.
  - no MTP / n-gram speculation: one token per step, always.
  - no chunked prefill: the prompt is one forward (the torch engine splits
    it at 4096 tokens, serving/prefill.py), so its activations grow with
    the prompt and the fit check charges them at the full ctx
    (prefill_bytes_per_token).
  - no rope scaling: --rope-scaling is refused at load rather than
    silently ignored.
  - the sampler is a numpy re-statement of serving/sampling.py's pipeline
    (repetition penalty on prompt+generated, presence/frequency on
    generated, temperature, top_k, top_p, one multinomial draw) — same
    order, same arithmetic in float32, a seeded np.random.Generator per
    request in place of the torch.Generator. Greedy is argmax of the
    penalised row, ties to the lowest id, torch.argmax's rule.

MEMORY (a request's, on unified memory):
  - the prefill heads ONE row: _logits runs the backbone over the prompt
    and the output head over the last hidden row only. mlx-lm's
    Model.__call__ heads every position and mlx does not push a later
    slice back through the matmul — T x vocab x 2 B to keep one row, 2.2
    GiB for an 8k prompt to Qwen3-4B (_head, _logits).
  - mlx caches every buffer it frees for reuse — on Metal up to its memory
    limit, 1.5x the working set by default, for the life of the process
    (mx.set_cache_limit's documentation): an M4 server held its 12 GB
    8k-request peak until it exited. So a request hands the cache back
    (mx.clear_cache) after its first token (the prefill's transients,
    which the fused path's decode never reuses) and at its end, on every
    way out; mlx-lm's own generate clears after each prefill chunk. The
    next request pays for re-allocating what it uses (0.11 s per GiB on
    MLX CPU on the dev box).
    DRINKME_MLX_KEEP_CACHE=1 keeps mlx's default instead (KEEP_CACHE_ENV).
  - each request ends with one line of mlx's own account: its peak active
    bytes (mx.get_peak_memory, reset when it starts) and what went back.

FIT CHECK (the seam docstring's design requirement for a unified-memory
engine): on a Mac the load is judged TWICE, in fit.py's vocabulary (one
memory pool; resident; one transient tensor; KV and the prefill's
activations at the allocated ctx; headroom — fit_check), against the
smaller of two budgets: mx.device_info()'s max_recommended_working_set_size
and what macOS would hand this process now (_host_budget: vm_stat's free,
inactive, speculative and purgeable pages, plus what mlx already holds —
the working set is 16 GiB on a 24 GB M4 whoever else is using it). It
REFUSES with the arithmetic printed when the need x suggest.FIT_HEADROOM
does not fit, naming both budgets, the one that bound and the largest
--ctx that fits. No bypass flag. Where there is no Metal there is no
working-set ceiling to read, and the check is skipped and says so.

  1. the ESTIMATE, before any weight is materialised (estimate_load_bytes):
     resident = the pack's per-tensor member sizes (its zip directories)
     + the raw remainder off the checkpoint's safetensors headers at the
     loader's own bf16 rule; transient = the largest single tensor the
     engine ever holds beyond that — the largest raw tensor staged at load
     or the largest packed Linear's dense bf16 plane at prefill (M>1),
     whichever is bigger (the reference path: every plane at once). The
     refusal that has to come first: a Mac whose memory is smaller than
     the model must decline before the allocation, not after the swap
     storm (a 24 GB M4 swapped this way).
  2. the MEASURED check, after the load, on what the modules actually hold
     (RadixLinear/RawLinear.resident_bytes + the streamed bytes) with the
     same transient — the truer resident number, kept as a second gate.

  The twin path charges its bf16 planes as resident and its decode as the
  transient (estimate_load_bytes' `twin`).
"""

from __future__ import annotations

import glob
import json
import os
import sys
import time
from typing import Iterator, NamedTuple

import numpy as np

import mlx.core as mx
import mlx.nn as nn

from .. import exitcodes
from . import capability, checkpoint, control, gen_config
from .detok import IncrementalDetok, SuffixWindow
from .engine import (Delta, Finished, GenerationRequest, GenEvent, GenResult, SampleParams,
                     StreamStart)
from .sampling import StopScanner
from .template import Prompt, effective_kwargs, render_prompt
from .tools import ToolCallScanner
from ..codec import registry
from ..metal import dense_radix, gemv_radix

# The families this engine serves, and the ones it refuses by name, live
# in drinkme.runtimes (torch-free, mlx-free) so the no-model picker and
# `serve --model` can apply the SAME predicate before a download or a
# pack; these names stay importable from here.
from ..runtimes import MLX_HYBRID_MODEL_TYPES as HYBRID_MODEL_TYPES  # noqa: E402
from ..runtimes import MLX_SUPPORTED_MODEL_TYPES as SUPPORTED_MODEL_TYPES  # noqa: E402

PATH_ENV = "DRINKME_MLX_PATH"
# the bench's twin arm: a path load_compressed_mlx takes by name (arms_mlx),
# never a default or DRINKME_MLX_PATH's; also RadixTwinLinear.tensor_codec,
# as swap.TWIN is on the torch side
TWIN = "twin"
# "1" keeps MLX's buffer cache from one request to the next (mlx's own
# default); unset, every request ends by handing it back (MEMORY, above)
KEEP_CACHE_ENV = "DRINKME_MLX_KEEP_CACHE"


def default_path() -> str:
    """The compute path this machine gets: DRINKME_MLX_PATH if set (fused |
    reference), else fused where there is Metal and reference where there
    is not."""
    forced = os.environ.get(PATH_ENV, "").strip().lower()
    if forced:
        if forced not in ("fused", "reference"):
            raise ValueError(f"{PATH_ENV}={forced!r}: expected 'fused' or 'reference'")
        return forced
    return "fused" if mx.metal.is_available() else "reference"


def device_name() -> str:
    """What model_meta()['device'] says: the SoC by name where there is a
    Metal device ("Apple M4 Pro"), else an honest CPU line. On the wire
    because a CPU fallback can hide behind GPU-labelled numbers; here the reference path on CPU is a
    legitimate test configuration, and it must never look like a Mac."""
    if mx.metal.is_available():
        try:
            return str(mx.device_info().get("device_name") or "Metal device")
        except Exception:  # noqa: BLE001 — a diagnostic must not take the load down
            return "Metal device"
    return f"cpu (mlx {mx.__version__}, no Metal)"


# ---------------------------------------------------------- the Linear ----


def biased_matmul(xf: mx.array, W: mx.array, bias: mx.array) -> mx.array:
    """xf [M, C] @ W.T + bias with the bias in the fp32 ACCUMULATOR and one
    rounding to xf's dtype — stock F.linear's epilogue, the torch
    CompressedLinear's (codec/swap.py _epilogue: 1 + 1/256 rounded to bf16
    is 1.0, and a bias of -1 then answers 0.0 where the fused epilogue
    keeps 0.00390625), and this engine's promise for every biased Linear
    it builds.

    mx.addmm IS that at M >= 2: the steel gemm's epilogue is alpha * acc +
    beta * c in fp32, cast once. At M == 1 it is NOT: mlx routes a single
    row to its gemv kernel, whose epilogue casts the accumulator to the
    output dtype BEFORE adding the bias — the second rounding (measured
    with mlx 0.32.2 on an M4: the cancellation above answers 0.0
    through mx.addmm at M == 1 and 0.00390625 at M in 2..33; the CPU build
    rounds once at every M — tests/test_serving_engine_mlx.py's
    test_a_biased_linear_rounds_once pins both). So the one row goes
    through the fp32 matmul instead: bf16 products are exact in fp32, the
    sum is an fp32 accumulate either way, then the bias, then the one
    cast. The cost is a 4 B/weight transient of W for that step — paid
    only by a biased Linear at decode on the dense/reference arms and
    RawLinear/BiasedLinear, which no dense Qwen3 has (attention_bias is
    false on every one; the fused RadixLinear never comes here at M == 1,
    gemv_radix adds the bias in-kernel)."""
    if xf.shape[0] == 1:
        y = xf.astype(mx.float32) @ W.astype(mx.float32).T + bias.astype(mx.float32)
        return y.astype(xf.dtype)
    return mx.addmm(bias, xf, W.T)


class RadixLinear(nn.Module):
    """A Linear whose weight is a RADIX pack tensor (codec/radix_pack.py, the
    bf16 codec): the block streams, the word
    directory and the palette resident as mx arrays (metal/gemv_radix
    .resident), decoded on every read. The same two-path switch as
    CompressedLinear:

      fused      M=1: gemv_radix, one dispatch, W never built — x read as
                 bf16 bits, fp32 accumulate, the bias added in fp32, ONE
                 rounding to the activation dtype (the torch arm's
                 radix_ops.gemv_fused epilogue).
                 M>1: the dense arm — dense_radix's transient bf16 weight
                 (one Linear at a time, 2 B/weight, dropped by the graph as
                 soon as the matmul has read it) through mx.matmul, which
                 is bitwise the reference's forward over the oracle's bytes
                 (same matmul, same bytes: the dense decode is pinned
                 bit-exact by bench/radix_bitpin_mlx.py).
      reference  the CPU oracle (radix_pack.decode_back_radix — native when
                 a C++ compiler is at hand, else codec/radix.decode) and the
                 same matmul at every M. A CORRECTNESS twin only: it decodes
                 every tensor on the CPU per forward, seconds per 8B tensor,
                 so at model level it is hours per token — the tests' and
                 gates' path, never a served one.

    `tensor_codec` is the registry's "radix" row. Bias epilogue: fp32 before the
    single cast on the gemv arm, biased_matmul on the dense one.
    """

    def __init__(self, pack: dict, bias=None, path: str | None = None,
                 name: str | None = None):
        super().__init__()
        name = name or "<unnamed>"
        registry.refuse_unless_metal_dense(pack, name)
        if registry.of_pack(pack)[0] != registry.RADIX:
            raise ValueError(f"{name} is not a radix pack tensor "
                             f"({registry.describe(*registry.of_pack(pack))}); "
                             "RadixLinear serves the radix codec only")
        if not registry.supported(registry.RADIX, 0, registry.GEMV, registry.METAL):
            raise ValueError(f"{name}: {registry.unsupported_note(registry.RADIX, 0, registry.GEMV, registry.METAL)}")
        self._res = gemv_radix.resident(pack)
        self.R, self.C = self._res["R"], self._res["C"]
        self.tensor_codec, self.layout = registry.RADIX, 0
        self.path = path or default_path()
        if bias is not None:
            self.bias = bias

    def resident_bytes(self) -> int:
        return gemv_radix.resident_bytes(self._res)

    def decode(self) -> np.ndarray:
        """The bf16 bit patterns [R, C] uint16 the forward multiplies, by the
        CPU oracle — what the bit-exactness test compares the dense kernel
        against."""
        return gemv_radix.decode_reference(self._res)

    def weight_bf16(self) -> mx.array:
        """The transient dense weight for THIS path: the decode-to-weight
        kernel's bytes on the fused path (unevaluated; NEVER stored on
        self), the CPU oracle's elsewhere."""
        if self.path == "fused":
            return dense_radix.weight_bf16_resident(self._res)
        return mx.array(self.decode()).view(mx.bfloat16)

    def __call__(self, x: mx.array) -> mx.array:
        shape = x.shape
        if (self.path == "fused" and x.dtype == mx.bfloat16 and x.size == self.C
                and self.C >= gemv_radix.MIN_BOUND and "bias" not in self):
            # the decode step: one graph node, the same kernel and bits as below
            return gemv_radix.gemv_radix_step(self._res, x, (*shape[:-1], self.R))
        xf = x.reshape(-1, self.C)
        if self.path == "fused" and xf.shape[0] == 1:
            xr = xf[0] if x.dtype in (mx.bfloat16, mx.float32) else xf[0].astype(mx.float32)
            b = self["bias"] if "bias" in self else None
            # fp32 accumulate (+ bias in fp32), ONE rounding — in-kernel to
            # bf16 when that is the activation dtype, else the fp32 sum cast
            y = gemv_radix.gemv_radix_resident(
                self._res, xr, b, mx.bfloat16 if x.dtype == mx.bfloat16 else mx.float32)
            out = y.astype(x.dtype)[None, :]
        else:
            W = self.weight_bf16()
            out = biased_matmul(xf, W, self["bias"]) if "bias" in self else xf @ W.T
        return out.reshape(*shape[:-1], self.R)


class RadixTwinLinear(nn.Module):
    """The bench's MLX TWIN of a radix tensor (codec/swap.RadixTwinLinear's
    counterpart): the UNCOMPRESSED bf16 weight, resident, decoded from the
    pack ONCE at construction by metal/twin_radix (plain mlx ops, bitwise
    the CPU oracle; it shares no code with the fused kernel), and the
    reference path's forward over it: `xf @ W.T`, or biased_matmul with a
    bias. Every output is therefore bitwise RadixLinear(path="reference")'s
    over the same pack. The difference is when the decode runs: once at
    load here, on every forward there (0.85 s per token on an M4's
    Qwen3-0.6B, with the native CPU decoder).
    The streams are not kept: resident is the 2 B/weight plane, the same
    footprint as the stock arm's weight."""

    def __init__(self, pack: dict, bias=None, path: str | None = None,
                 name: str | None = None):
        from ..metal import twin_radix

        super().__init__()
        name = name or "<unnamed>"
        registry.refuse_unless_metal_dense(pack, name)
        if registry.of_pack(pack)[0] != registry.RADIX:
            raise ValueError(f"{name} is not a radix pack tensor "
                             f"({registry.describe(*registry.of_pack(pack))}); "
                             "RadixTwinLinear serves the radix codec only")
        gemv_radix.refuse_unsupported(pack)  # the compressed arm's lines, so both arms serve one set
        self.R, self.C = int(pack["R"]), int(pack["C"])
        self._w = twin_radix.decode(pack, name).view(mx.bfloat16)  # underscore: not a parameter
        self.tensor_codec, self.layout = TWIN, 0
        self.path = path or TWIN
        if bias is not None:
            self.bias = bias

    def resident_bytes(self) -> int:
        return int(self._w.nbytes)

    def decode(self) -> np.ndarray:
        """The bf16 bit patterns [R, C] uint16 this module multiplies by."""
        return np.array(self._w.view(mx.uint16))

    def weight_bf16(self) -> mx.array:
        return self._w

    def __call__(self, x: mx.array) -> mx.array:
        shape = x.shape
        xf = x.reshape(-1, self.C)
        out = biased_matmul(xf, self._w, self["bias"]) if "bias" in self else xf @ self._w.T
        return out.reshape(*shape[:-1], self.R)


class RawLinear(nn.Module):
    """The raw fallback (codec/radix_pack.raw_dict: a tensor radix would have
    expanded, stored as its bf16 bits verbatim) on the mlx runtime — swap.RawLinear's
    twin: the resident bf16 weight through mlx's own matmul on every path
    and at every M. Stock's arithmetic, nothing to gate; `path` is taken
    for the loader's uniform call and recorded, but there is no fused arm
    to switch to. `tensor_codec` is the registry's "raw" row (DENSE on METAL is its
    only op, as on the torch runtime). The bias, where a config has one,
    goes through biased_matmul — the dense-arm rule the other modules use.
    """

    def __init__(self, pack: dict, bias=None, path: str | None = None,
                 name: str | None = None):
        super().__init__()
        name = name or "<unnamed>"
        registry.refuse_unless_metal_dense(pack, name)
        if registry.of_pack(pack)[0] != registry.RAW:
            raise ValueError(f"{name} is not a raw fallback tensor "
                             f"({registry.describe(*registry.of_pack(pack))}); "
                             "RawLinear serves the raw fallback only")
        bits = np.ascontiguousarray(pack["raw_bits"], dtype=np.uint16)
        if bits.ndim != 2 or bits.shape != (int(pack["R"]), int(pack["C"])):
            raise ValueError(f"{name}: raw_bits is {bits.shape}, not [R, C] = "
                             f"({int(pack['R'])}, {int(pack['C'])})")
        self.R, self.C = int(pack["R"]), int(pack["C"])
        self._w = mx.array(bits).view(mx.bfloat16)  # underscore: not a parameter to the tree walk
        self.tensor_codec, self.layout = registry.RAW, 0
        self.path = path or default_path()
        if bias is not None:
            self.bias = bias

    def resident_bytes(self) -> int:
        return int(self._w.nbytes)

    def decode(self) -> np.ndarray:
        """The bf16 bit patterns [R, C] uint16 — the stored bytes themselves."""
        return np.array(self._w.view(mx.uint16))

    def weight_bf16(self) -> mx.array:
        return self._w

    def __call__(self, x: mx.array) -> mx.array:
        shape = x.shape
        xf = x.reshape(-1, self.C)
        out = biased_matmul(xf, self._w, self["bias"]) if "bias" in self else xf @ self._w.T
        return out.reshape(*shape[:-1], self.R)


class BiasedLinear(nn.Linear):
    """An UNPACKED projection the checkpoint carries a bias for and the
    skeleton has no slot for: mlx-lm's Qwen3 builds every Linear
    bias=False whatever config.json's attention_bias says (the field is
    not in its ModelArgs), so a biased checkpoint's k/v biases would have
    "no home" — _stream_checkpoint swaps the module for this one when the
    bias arrives (either arm: stock mlx-lm's own strict load refuses such
    a checkpoint outright, so there is no community arithmetic to keep),
    and the forward is biased_matmul's — the bias in fp32, one rounding —
    where nn.Linear's own mx.addmm rounds twice at M == 1. Built from the
    arrays, not nn.Linear's __init__: no random init to schedule."""

    def __init__(self, weight: mx.array, bias: mx.array):
        nn.Module.__init__(self)
        self.weight = weight
        self.bias = bias

    def __call__(self, x: mx.array) -> mx.array:
        shape = x.shape
        out = biased_matmul(x.reshape(-1, shape[-1]), self["weight"], self["bias"])
        return out.reshape(*shape[:-1], out.shape[-1])


def make_module_mlx(pack: dict, bias=None, path: str | None = None,
                    name: str | None = None) -> nn.Module:
    """The loader's one constructor (swap.make_module's twin): a pack dict
    (iter_pack_dir's) -> the resident module for it, by the tensor's own
    kind — RadixLinear for a radix tensor (the `codec` scalar), RawLinear
    for the raw fallback — after the registry has said the tensor has a
    Metal path at all (refuse_unless_metal_dense, by name).
    load_compressed_mlx builds every packed Linear through this, so a pack
    with raw fallbacks installs the matching module per tensor without the
    tensor loop knowing about the mix. On the bench's twin path (`path`
    TWIN) a radix tensor is RadixTwinLinear's and a raw fallback the
    RawLinear the compressed arm serves it with (swap.make_twin's rule).
    A registry row flipped True for a
    kind with no module class here is named as the bug it would be, not
    served."""
    name = name or "<unnamed>"
    registry.refuse_unless_metal_dense(pack, name)
    tensor_codec, _ = registry.of_pack(pack)
    if tensor_codec == registry.RADIX and path == TWIN:
        return RadixTwinLinear(pack, bias, path=path, name=name)
    if tensor_codec == registry.RADIX:
        return RadixLinear(pack, bias, path=path, name=name)
    if tensor_codec == registry.RAW:
        return RawLinear(pack, bias, path=path, name=name)
    raise NotImplementedError(
        f"codec/registry.py says {registry.describe(tensor_codec, 0)} has a Metal dense row, but "
        "serving/engine_mlx.py has no module class for it")


# ------------------------------------------------------------ sampling ----


def _penalize(logits: np.ndarray, params: SampleParams, prev_ids, gen_ids) -> np.ndarray:
    """sampling._penalize in numpy: a private float32 copy with the
    repetition penalty (HF's divide-positive / multiply-negative over
    prompt+generated) then OpenAI's additive presence + frequency*count
    over generated only. Same ops, same order."""
    logits = np.array(logits, dtype=np.float32, copy=True)
    if params.repetition_penalty != 1.0 and prev_ids:
        idx = np.fromiter(sorted({int(i) for i in prev_ids}), dtype=np.int64)
        picked = logits[idx]
        logits[idx] = np.where(picked > 0, picked / params.repetition_penalty,
                               picked * params.repetition_penalty)
    if (params.presence_penalty or params.frequency_penalty) and gen_ids:
        ids, counts = np.unique(np.asarray(list(gen_ids), dtype=np.int64),
                                return_counts=True)
        logits[ids] -= params.presence_penalty + params.frequency_penalty * counts.astype(np.float32)
    return logits


def _shape(logits: np.ndarray, params: SampleParams) -> None:
    """sampling._shape in numpy, in place: temperature, top_k (ties at the
    kth value survive), top_p (the crossing token stays; at least one token
    always survives)."""
    logits /= params.temperature
    n = logits.shape[0]
    if 0 < params.top_k < n:
        kth = np.partition(logits, n - params.top_k)[n - params.top_k]
        logits[logits < kth] = -np.inf
    if params.top_p < 1.0:
        order = np.argsort(-logits, kind="stable")
        srt = logits[order]
        m = srt.max()
        probs = np.exp(srt - m)
        probs /= probs.sum()
        cum = np.cumsum(probs)
        srt[(cum - probs) > params.top_p] = -np.inf
        logits[:] = -np.inf
        logits[order] = srt


def sample_next(logits: np.ndarray, params: SampleParams, rng, prev_ids=None,
                gen_ids=None) -> int:
    """One decode step, logits [V] -> token id, sampling.sample_next's order
    on the host. `rng` is the request's np.random.Generator (None for a
    greedy request, which never draws)."""
    row = _penalize(logits, params, prev_ids, gen_ids)
    if params.temperature == 0:
        return int(np.argmax(row))
    _shape(row, params)
    m = row.max()
    probs = np.exp(row - m)
    probs /= probs.sum()
    u = rng.random()
    return int(min(np.searchsorted(np.cumsum(probs), u, side="right"), len(probs) - 1))


def pick_token(logits: np.ndarray, params: SampleParams, rng, prev_ids, gen_ids,
               eos_ids, decode, constraint, cur: str | None = None) -> int:
    """constrain.pick_token's loop on the host row: sample through the normal
    pipeline, ban a candidate whose decoded text goes FAIL, resample; EOS
    allowed iff the text so far is COMPLETE. The JsonConstraint itself is
    torch-free; only the row it bans on had to change representation."""
    from .constrain import COMPLETE, FAIL

    row = np.array(logits, dtype=np.float32, copy=True)
    if cur is None:
        cur = decode(gen_ids)
    live = int(np.isfinite(row).sum())
    banned: set[int] = set()
    while True:
        if len(banned) >= live:
            raise RuntimeError("constrained decode: every candidate token was banned")
        tok = sample_next(row, params, rng, prev_ids=prev_ids, gen_ids=gen_ids)
        if tok in banned:
            raise RuntimeError("constrained decode: the row is not sane (NaN?)")
        if tok in eos_ids:
            if constraint.status(cur) == COMPLETE:
                return tok
        else:
            cand = decode(gen_ids + [tok])
            if len(cand) > len(cur) and constraint.status(cand) != FAIL:
                return tok
        row[tok] = -np.inf
        banned.add(tok)


# -------------------------------------------------------------- engine ----


def _head(model, h: mx.array) -> mx.array:
    """The output head of mlx-lm's Qwen3 Model.__call__ (mlx_lm/models/
    qwen3.py), applied to the hidden states `h` the backbone (`model.model`)
    returned: the embedding as a Linear when the config ties them, else
    lm_head — a packed RadixLinear/RawLinear when the pack carries it."""
    if model.args.tie_word_embeddings:
        return model.model.embed_tokens.as_linear(h)
    return model.lm_head(h)


# A decode step hands the GPU its graph every this many layers (last_row_logits)
DECODE_EVAL_EVERY = 2
# The layer-by-layer decode restates ONE family's forward (mlx-lm's Qwen3Model.__call__); any
# other inner model takes its own __call__, so adding a family to runtimes can't silently change
# its arithmetic here.
_LAYERWISE_MODULES = frozenset({"mlx_lm.models.qwen3"})


def last_row_logits(model, ids: list[int], cache=None) -> mx.array:
    """The LAST position's logits after `ids` as a float32 [vocab] row, not
    yet evaluated: the backbone over the prompt, the head over the last hidden
    row only (MEMORY, module docstring). The one path serve (MLXEngine._logits)
    and bench (arms_mlx greedy, timed_prefill, timed_ttft) both run, so the
    bench times what serve does. At T = 1 (decode) the graph is the one
    Model.__call__ builds, handed to the GPU (mx.async_eval) every
    DECODE_EVAL_EVERY layers, so the host builds and encodes the next layers
    while the GPU runs the last ones instead of before the GPU starts: the
    same ops in the same order, the same bits (tests/test_serving_engine_mlx.py);
    on an M4's Qwen3-4B +5.5% decode on the compressed arm, +0.8% on stock."""
    x = mx.array([ids], dtype=mx.int32)
    inner = model.model
    if cache is None or x.shape[1] != 1 or type(inner).__module__ not in _LAYERWISE_MODULES:
        h = inner(x, cache=cache)
    else:
        from mlx_lm.models.base import create_attention_mask

        # mlx-lm's Qwen3Model.__call__, layer by layer
        h = inner.embed_tokens(x)
        mask = create_attention_mask(h, cache[0])
        for i, (layer, c) in enumerate(zip(inner.layers, cache)):
            h = layer(h, mask, c)
            if (i + 1) % DECODE_EVAL_EVERY == 0:
                mx.async_eval(h)
        h = inner.norm(h)
    return _head(model, h[:, -1:, :])[0, -1].astype(mx.float32)


class MLXEngine:
    """serving.engine.Engine over an mlx-lm causal LM (either arm)."""

    def __init__(self, model, tokenizer, model_id: str, arm: str, meta: dict,
                 ctx: int, gen_defaults: gen_config.GenDefaults | None = None,
                 path: str | None = None, resident_bytes: int = 0):
        self.model = model
        self.tok = tokenizer
        self.model_id = model_id
        self.arm = arm
        self.meta = meta  # pack provenance (hfRepo/revision/meanBpw/weightedBpw) or {}
        self.ctx = ctx
        self.path = path or default_path()
        self.device = device_name()
        self.resident_bytes = resident_bytes
        gd = gen_defaults or gen_config.GenDefaults()
        self.sampling_defaults = gd.sampling
        self.template_kwargs = dict(gd.template_kwargs) or None
        print(f"[drinkme] sampling defaults from generation_config.json "
              f"({gd.source}): {self.sampling_defaults or '{}'}", flush=True)
        # serving/capability.py: tokenizer-only, so the SAME probe the torch
        # engine runs — nothing about the accelerator is in the answer
        self.capability = capability.probe(tokenizer)
        print(f"[drinkme] capabilities: thinking={self.capability.thinking} "
              f"tool_format={self.capability.tool_format}", flush=True)
        # serving/control.py, the same resolution HFEngine does: the
        # row's special-id markers kept as text, the call-close id a stop
        self.control = control.resolve(tokenizer, self.capability.tool_format)
        if self.control.ids:
            print(f"[drinkme] control tokens: {self.control.describe()}", flush=True)
        eos: set[int] = set()
        if isinstance(tokenizer.eos_token_id, int):
            eos.add(tokenizer.eos_token_id)
        for e in meta.get("eos_token_ids") or ():
            eos.add(int(e))
        self.eos_ids = frozenset(eos)
        self._detok_verify = os.environ.get("DRINKME_DETOK_VERIFY") == "1"
        self._keep_cache = os.environ.get(KEEP_CACHE_ENV) == "1"

    # ---- the seam's tokenizer routes: identical to HFEngine's ----

    def model_meta(self) -> dict:
        # the torch engine's object (engines.HFEngine.model_meta), plus this
        # runtime's compute path (fused / reference — docs/metal.md)
        return {"arm": self.arm,
                "runtime": "mlx",
                "hfRepo": self.meta.get("hfRepo"),
                "revision": self.meta.get("revision"),
                # the pack's compression profile (sip / gulp), the pack's own name
                "compressionProfile": self.meta.get("profile"),
                # the lexicon's names (engines.HFEngine.model_meta): weightedBpw
                # is bitsPerWeight, meanBpw is meanTensorBitsPerWeight
                "bitsPerWeight": self.meta.get("weightedBpw"),
                "meanTensorBitsPerWeight": self.meta.get("meanBpw"),
                # the released precision served exactly (bf16), the torch
                # engine's field
                "sourceDtype": self.meta.get("sourceDtype"),
                "contextWindow": self.ctx,
                "device": self.device,
                "computePath": self.path,
                "capabilities": self.capability.as_dict(),
                "sampling": {"profile": None,
                             "defaults": gen_config.effective_defaults(self.sampling_defaults)}}

    def _render(self, req: GenerationRequest) -> Prompt:
        # HFEngine._render, verbatim: one render for generate() and count_tokens()
        return render_prompt(
            self.tok, req.messages,
            template_kwargs=effective_kwargs(self.template_kwargs, req.template_kwargs,
                                             constrained=req.sampling.output_schema is not None),
            tools=req.tools)

    def count_tokens(self, req: GenerationRequest) -> int:
        return len(self._render(req).ids)

    def tokenize(self, prompt: str | None = None, messages: list[dict] | None = None,
                 tools: list | None = None,
                 template_kwargs: dict | None = None) -> list[int]:
        if messages is not None:
            return render_prompt(self.tok, messages,
                                 template_kwargs=effective_kwargs(self.template_kwargs,
                                                                  template_kwargs),
                                 tools=tools).ids
        return self.tok.encode(prompt)

    def detokenize(self, tokens: list[int]) -> str:
        return self.tok.decode(tokens, skip_special_tokens=False)

    def tokenizer_info(self) -> dict:
        return {"bos_token_id": self.tok.bos_token_id, "eos_token_id": self.tok.eos_token_id,
                "chat_template": self.tok.chat_template is not None}

    # ---- the forward ----

    def _logits(self, ids: list[int], cache) -> np.ndarray:
        """Run `ids` through the model against `cache`; the LAST position's
        logits as a float32 numpy row. The head reads the last hidden row
        only: mlx-lm's Model.__call__ applies it to all T positions and mlx
        does not push a later slice back through the matmul, so the prefill
        of a T-token prompt built [1, T, vocab] bf16 logits to keep one row
        (7862 x 151936 x 2 B = 2.2 GiB for an 8k prompt to Qwen3-4B, the
        jump an M4's footprint took at the end of that prefill). At T = 1
        (decode) the graph is the one Model.__call__ builds. One mx.eval
        per call: the whole step is one graph."""
        row = last_row_logits(self.model, ids, cache)
        mx.eval(row)
        return np.array(row)

    def _return_cache(self) -> int:
        """Hand mlx's buffer cache back (mx.clear_cache) unless KEEP_CACHE_ENV
        keeps it; the bytes it held (module docstring, MEMORY)."""
        mx.synchronize()  # nothing of this request still in flight
        cached = int(mx.get_cache_memory())
        if not self._keep_cache:
            mx.clear_cache()
        return cached

    def prefill_logits(self, ids: list[int]) -> np.ndarray:
        """The next-token logits after `ids`, fresh cache — what the toy
        tests compare against the torch engine's."""
        from mlx_lm.models.cache import make_prompt_cache

        return self._logits(ids, make_prompt_cache(self.model))

    def generate(self, req: GenerationRequest) -> Iterator[GenEvent]:
        from mlx_lm.models.cache import make_prompt_cache

        params, tools = req.sampling, req.tools
        prompt = self._render(req)
        ids = prompt.ids
        n_prompt = len(ids)
        max_new = min(params.max_tokens, self.ctx - n_prompt)
        if max_new <= 0:
            yield StreamStart(n_prompt, opens_think=prompt.opens_think)
            yield Finished(GenResult("", "length", n_prompt, 0))
            return
        # the explicit stream-start fact (engine.py); no prefix cache on this
        # engine, so nothing is ever reused
        yield StreamStart(n_prompt, opens_think=prompt.opens_think, cached_tokens=0)
        toolscan = (ToolCallScanner(tools, self.capability.tool_format)
                    if tools else None)
        parsed_calls: list[dict] = []
        constraint = None
        if params.output_schema is not None:
            from .constrain import JsonConstraint

            constraint = JsonConstraint(params.output_schema,
                                        validate=not params.output_schema_validated)
        scan = StopScanner(params.stop)
        detok = IncrementalDetok()
        all_ids = list(ids)
        gen_ids: list[int] = []

        # decode(skip_special_tokens=True) with the row's control tokens kept
        # as text (serving/control.py); the unchanged call for a row with none
        decode_ids = self.control.decoder(self.tok)

        win = SuffixWindow(decode_ids, verify=self._detok_verify)
        rng = None
        if params.temperature != 0:
            rng = np.random.default_rng(params.seed if params.seed is not None else None)
        sent: list[str] = []
        finish = "stop"
        n = 0
        cache = make_prompt_cache(self.model)  # per request: no prefix cache (module docstring)
        mx.reset_peak_memory()  # this request's peak, for the line at its end
        after_prefill = None
        try:
            logits = self._logits(ids, cache)
            closed = False  # last token was the row's call-close marker (control.py)
            while True:
                if constraint is None:
                    tok_id = sample_next(logits, params, rng, prev_ids=all_ids, gen_ids=gen_ids)
                else:
                    tok_id = pick_token(logits, params, rng, all_ids, gen_ids, self.eos_ids,
                                        decode_ids, constraint)
                n += 1
                if tok_id in self.eos_ids:
                    break
                if closed and tok_id not in self.control.reopen:
                    break  # a completed call, not re-opened: the turn ends (engines.py)
                closed = tok_id in self.control.stop_after
                all_ids.append(tok_id)
                gen_ids.append(tok_id)
                full = win.full(gen_ids)
                delta = detok.push(full)
                if toolscan is not None:
                    delta, done = toolscan.feed(delta)
                    parsed_calls += done
                out = scan.feed(delta)
                if out:
                    if (yield Delta(out)) is False:
                        finish = "abort"
                        break
                    sent.append(out)
                if scan.stopped:
                    break
                if n == max_new:
                    finish = "length"
                    break
                if after_prefill is None:
                    # the prefill's transients (activations, dense planes)
                    # are free and the fused path's decode (the GEMV) never
                    # reuses them: back before the first decode step, so the
                    # time to first token does not pay for it
                    after_prefill = self._return_cache()
                logits = self._logits([tok_id], cache)
        finally:
            # the request's buffers go back on every way out: the loop's
            # break, and a consumer that closes the stream mid-decode
            del cache
            at_end = self._return_cache()
            print(f"[drinkme.engine] mlx memory: {n_prompt}-token prompt, peak "
                  f"{mx.get_peak_memory() / 1024**3:.2f} GiB active; cache "
                  f"{'kept' if self._keep_cache else 'returned'}: "
                  f"{(after_prefill or 0) / 1024**3:.2f} GiB after prefill, "
                  f"{at_end / 1024**3:.2f} GiB at the end", flush=True)
        if finish in ("stop", "length") and not scan.stopped:
            rem = detok.flush(decode_ids(gen_ids))
            if toolscan is not None:
                r, done = toolscan.feed(rem)
                parsed_calls += done
                tail = scan.feed(r) + scan.feed(toolscan.flush())
                parsed_calls += toolscan.flush_calls()
            else:
                tail = scan.feed(rem)
            tail += scan.flush()
            if tail:
                if (yield Delta(tail)) is False:
                    finish = "abort"
                else:
                    sent.append(tail)
        if parsed_calls and finish == "stop":
            finish = "tool_calls"
        yield Finished(GenResult("".join(sent), finish, n_prompt, n,
                                 tool_calls=parsed_calls or None, cached_tokens=0,
                                 stop_sequence=scan.matched))


# ------------------------------------------------------------- loading ----


def _read_config(snap: str) -> dict:
    with open(os.path.join(snap, "config.json")) as f:
        return normalize_config(json.load(f))


def normalize_config(config: dict) -> dict:
    """HF config.json -> the spelling mlx-lm's ModelArgs reads.

    transformers 5 writes the rotary settings as one `rope_parameters`
    object ({"rope_theta": ..., "rope_type": "default", ...}); mlx-lm 0.31
    reads the pre-5 top-level `rope_theta` + `rope_scaling`, which is what
    the hub's Qwen3 checkpoints (written by transformers 4) still carry. A
    config saved by transformers 5 — every toy this suite builds, and any
    re-saved checkpoint — would otherwise fail ModelArgs with a missing
    rope_theta. Lifts the two fields when only the new spelling is present;
    a config that already has the old one is returned as it came."""
    rp = config.get("rope_parameters")
    if not isinstance(rp, dict):
        return config
    out = dict(config)
    if "rope_theta" not in out and "rope_theta" in rp:
        out["rope_theta"] = rp["rope_theta"]
    if "rope_scaling" not in out:
        rest = {k: v for k, v in rp.items() if k != "rope_theta"}
        # {"rope_type": "default"} is no scaling at all; anything else
        # (yarn's factor / original_max_position_embeddings) IS rope_scaling
        out["rope_scaling"] = None if rest.get("rope_type", "default") == "default" else rest
    return out


def refuse_unsupported(config: dict) -> None:
    """Name what arrived and what this engine serves, in one breath — the
    one runtime-capability predicate (runtimes.refusal), at the loader."""
    from ..runtimes import MLX, refusal

    reason = refusal(MLX, str(config.get("model_type", "?")))
    if reason is not None:
        raise ValueError(reason)


def build_skeleton(config: dict):
    """mlx-lm's model class for this config, constructed but NOT evaluated:
    mlx is lazy, so the random init nn.Linear schedules is never computed
    for a module that gets replaced or overwritten before anything reads
    it. Returns (model, args)."""
    import importlib

    refuse_unsupported(config)
    mod = importlib.import_module(SUPPORTED_MODEL_TYPES[config["model_type"]])
    args = mod.ModelArgs.from_dict(config)
    return mod.Model(args), args


def _module_at(model, path: str):
    node = model
    for part in path.split("."):
        node = node[int(part)] if part.isdigit() else getattr(node, part)
    return node


def _set_at(model, path: str, value) -> None:
    parent_path, _, child = path.rpartition(".")
    parent = _module_at(model, parent_path) if parent_path else model
    if child.isdigit():
        parent[int(child)] = value
    else:
        setattr(parent, child, value)


def kv_bytes_per_token(config: dict) -> int:
    """K and V, bf16, every layer, at this config's head geometry."""
    head_dim = int(config.get("head_dim") or config["hidden_size"] // config["num_attention_heads"])
    return 2 * int(config["num_hidden_layers"]) * int(config["num_key_value_heads"]) * head_dim * 2


class LoadBytes(NamedTuple):
    """estimate_load_bytes' answer, in fit.py's words: what the loaded
    engine holds (`resident`) and the largest single tensor it ever holds
    beyond that (`transient`) — the two terms fit.streamed_transient_bytes
    charges on unified memory, which a Mac always is."""
    resident: int    # packed streams + the raw remainder at the loader's bf16 rule
    transient: int   # max(largest raw tensor staged at load, largest dense bf16 plane at prefill)
    packed: int      # tensors the pack carries (0 for the stock arm)
    raw: int         # tensors streamed from the checkpoint
    dense_total: int = 0  # every radix tensor's dense bf16 plane: the reference path's transient


def estimate_load_bytes(snap: str, pack_dir: str | None, tie: bool,
                        twin: bool = False) -> LoadBytes:
    """The load budget BEFORE a weight is materialised, from descriptors
    alone: the checkpoint's safetensors headers (dtype x shape per tensor,
    codec.pack._shard_headers — zero weight bytes read) and, for the
    compressed arm, the pack's per-tensor scalars and zip directories
    (codec.pack.iter_pack_descriptors — zero array bytes read).

    Resident is what _stream_checkpoint will attach — every header tensor
    the pack does not carry, minus the tied lm_head the skeleton never
    holds, floats at bf16 width (codec.pack.raw_resident_bytes: its astype
    rule) — plus, per packed tensor, the member bytes the module keeps
    (codec.pack.packed_resident_estimate). Transient is the largest
    tensor held beyond that at any moment: at load, one raw tensor staged
    from its shard at its source width (checkpoint.largest_tensor_bytes'
    rule, over the raw remainder only — a packed tensor's bytes are never
    read); at prefill, one packed Linear's dense bf16 plane (2 x R x C,
    RadixLinear.weight_bf16 — a raw fallback IS its plane, no transient).
    The two never coincide, so the charge is their max: fit.py's ONE
    `tensor_bytes` term. That is the FUSED path's transient. The reference
    path decodes every plane on the CPU while the forward's graph is BUILT
    (weight_bf16 -> mx.array of a numpy decode), so a forward holds all of
    them at once before mlx evaluates the first: `dense_total`, their sum
    (Qwen3-0.6B's 196 planes, 0.82 GiB: a 512-token request on MLX CPU
    peaked 0.90 GiB over resident). `tie` is the config's
    tie_word_embeddings, the key mlx-lm's ModelArgs reads under the same
    default.

    `twin` (the bench's twin path, RadixTwinLinear): a radix tensor is
    resident as its bf16 plane (2 x R x C) and prefill needs no dense
    transient; the transient is instead the decode's, one tensor at a
    time: its streams as read from the pack plus the decoder's chunk
    planes (metal/twin_radix.chunk_transient_bytes). Those resident planes
    are the ones the reference path builds per forward, held once instead
    (RadixTwinLinear's forward and weight_bf16 read self._w and decode
    nothing), so `dense_total` stays 0: the twin is never charged them a
    second time as a transient."""
    from ..codec.pack import (_shard_headers, iter_pack_descriptors, packed_resident_estimate,
                              raw_resident_bytes)
    from ..check import _dtype_nbytes

    packed: dict[str, tuple[dict, dict]] = {}
    if pack_dir is not None:
        for name, sc, sizes in iter_pack_descriptors(pack_dir):
            packed[name] = (sc, sizes)
    headers = _shard_headers(snap)
    if not headers:
        raise FileNotFoundError(f"no *.safetensors under {snap}")
    resident = 0
    largest_raw = 0
    n_raw = 0
    for tname, (dtype, shape, _path) in headers.items():
        mod_path, _, attr = tname.rpartition(".")
        if mod_path in packed and attr == "weight":
            continue  # the pack IS this tensor; never read, never resident raw
        if tie and tname == "lm_head.weight":
            continue  # tied: the skeleton has no lm_head to hold it
        resident += raw_resident_bytes(dtype, shape)
        n = 1
        for d in shape:
            n *= int(d)
        largest_raw = max(largest_raw, _dtype_nbytes(dtype) * n)
        n_raw += 1
    largest_dense = 0
    dense_total = 0
    for name, (sc, sizes) in packed.items():
        if twin and sc.get("codec") == registry.RADIX:
            from ..metal.twin_radix import chunk_transient_bytes

            R, C = int(sc["R"]), int(sc["C"])
            resident += 2 * R * C
            largest_dense = max(largest_dense, packed_resident_estimate(sc, sizes)
                                + chunk_transient_bytes(R, C, int(sc["block_size"])))
            continue
        resident += packed_resident_estimate(sc, sizes)
        if sc.get("codec") == registry.RADIX:
            largest_dense = max(largest_dense, 2 * int(sc["R"]) * int(sc["C"]))
            dense_total += 2 * int(sc["R"]) * int(sc["C"])
    return LoadBytes(resident, max(largest_raw, largest_dense), len(packed), n_raw, dense_total)


def prefill_bytes_per_token(config: dict) -> int:
    """The activations a prefill holds per prompt token beyond resident,
    KV and the one dense plane: 8 x (hidden + intermediate) bytes.

    A Qwen3 layer's live set peaks in its MLP — the residual and its norm
    [T, hidden] and gate and up [T, intermediate], bf16: 4 x (hidden +
    intermediate) B/token (MLX CPU, a 0.6B layer's MLP at 512 tokens: 7.00
    MiB over its input, gate + up + out) — and on Metal a command buffer's
    buffers are held until it completes. An M4 serving Qwen3-4B took 742
    MiB in the first layer of a 7862-token prefill (footprint 7983 -> 8725
    MiB, then the KV's steady climb), 96.6 KiB/token = 7.9 x (2560 + 9728)
    B: this term. A 1976-token prefill there took 382 MiB, 197 MiB over
    it: a constant the term does not carry, which for the 4B sits inside
    the transient (its 0.72 GiB is the embedding staged at load; a dense
    plane is 0.05 GiB) and the headroom. The attention's live set is
    smaller at every dense Qwen3 geometry, and Metal's fused attention
    never builds [heads, T, T] scores; MLX CPU's fallback does (2.1 GiB at
    8k tokens on a 0.6B), which this term does not model — the check runs
    only where there is Metal."""
    return 8 * (int(config["hidden_size"]) + int(config["intermediate_size"]))


def fit_check(resident: int, kv_per_token: int, ctx: int,
              working_set: int | None, stream=None, transient: int = 0,
              stage: str = "measured", prefill_per_token: int = 0,
              host: int | None = None) -> None:
    """The unified-memory fit check (module docstring, FIT CHECK), fit.py's
    arithmetic: fits(need, 0, budget, FIT_HEADROOM), need =
    streamed_transient_bytes(resident, KV + prefill, "unified", transient)
    — a Mac is one pool, the one tensor beyond resident is `transient`
    (estimate_load_bytes' term), and KV and the prefill's activations
    (prefill_bytes_per_token) are charged at a ctx-long prompt, the longest
    the server accepts. The budget is the smaller of `working_set`
    (mx.device_info()['max_recommended_working_set_size']) and `host`
    (_host_budget: what macOS would hand this process now; None = unread,
    and the working set alone is the budget). `working_set` None means no
    Metal device to read one from: skipped, and said. `stage` names which
    resident number is being judged in the printed line: "estimate"
    (descriptors, before any weight exists) or "measured" (the loaded
    modules). Refuses with SystemExit naming both budgets, the one that
    bound, and the largest --ctx that fits; no bypass."""
    from ..fit import fits, streamed_transient_bytes
    from ..suggest import FIT_HEADROOM

    stream = sys.stderr if stream is None else stream
    kv = kv_per_token * ctx
    prefill = prefill_per_token * ctx
    # fit.py's charge for a streamed arm on unified memory: resident + the
    # one tensor beyond it + KV (+ the prefill's activations). A transient
    # of 0 charges nothing beyond resident (a bare accounting call) — not
    # fit.py's shard-sized stand-in, which is for a snapshot not yet on
    # disk; the loaders always have theirs.
    need = (streamed_transient_bytes(resident, kv + prefill, "unified", tensor_bytes=transient)
            if transient else resident + kv + prefill)
    gib = 1024 ** 3
    terms = (f"resident {resident / gib:.2f} GiB + transient {transient / gib:.2f} GiB "
             f"+ KV {kv / gib:.2f} GiB + prefill {prefill / gib:.2f} GiB at ctx {ctx}")
    if working_set is None:
        print(f"[drinkme] fit check ({stage}): no Metal device to read a working-set "
              f"ceiling from — skipped ({terms})", file=stream, flush=True)
        return
    if host is not None and host < working_set:
        budget, bound = host, "memory macOS has free for this process"
    else:
        budget, bound = working_set, "GPU working set"
    host_s = f"{host / gib:.2f} GiB" if host is not None else "unread"
    line = (f"{terms} = {need / gib:.2f} GiB; budget min(GPU working set "
            f"{working_set / gib:.2f} GiB, host available {host_s}) = {budget / gib:.2f} GiB "
            f"/ {FIT_HEADROOM} headroom = {budget / FIT_HEADROOM / gib:.2f} GiB")
    if not fits(need, 0.0, budget, FIT_HEADROOM):
        per_token = kv_per_token + prefill_per_token
        room = budget / FIT_HEADROOM - (need - per_token * ctx)
        fit_ctx = int(room // per_token) // 1024 * 1024 if per_token and room > 0 else 0
        todo = (f"--ctx {fit_ctx} fits it" if fit_ctx >= 1024
                else "no --ctx fits it: serve a smaller model")
        if bound != "GPU working set":
            todo += ", or close apps to free memory"
        raise exitcodes.LoadRefused(
            f"drinkme: refusing to load ({stage}): does not fit this Mac with the required "
            f"headroom — {line}. The {bound} is the bound: {todo}. There is no flag "
            f"that skips this check.")
    print(f"[drinkme] fit check ({stage}): ok — {line}", file=stream, flush=True)


# parse_vm_stat and the vm_stat read live in fit.py (torch- and mlx-free), where
# packs.hardware_budget's serve picker reads the same number.
from ..fit import host_available_bytes, parse_vm_stat  # noqa: E402,F401

def _host_budget() -> int | None:
    """What this process's mlx memory could grow to on the host now: what
    macOS would hand over (fit.host_available_bytes: parse_vm_stat of
    `vm_stat`, no sudo) plus what mlx already holds (active + cache) — at
    the measured check the loaded weights have left vm_stat's free pages
    and are counted back here. None off macOS, or when vm_stat cannot be
    read or parsed."""
    if sys.platform != "darwin":
        return None
    avail = host_available_bytes()
    if avail is None:
        return None
    return avail + int(mx.get_active_memory()) + int(mx.get_cache_memory())


def _working_set() -> int | None:
    if not mx.metal.is_available():
        return None
    try:
        return int(mx.device_info()["max_recommended_working_set_size"])
    except Exception:  # noqa: BLE001
        return None


def _stream_checkpoint(model, snap: str, packed: set[str], tie: bool,
                       name: str | None = None) -> tuple[int, set[str]]:
    """Every non-packed tensor of the checkpoint, as bf16, into the module
    tree — one tensor evaluated at a time (mx.load of a safetensors file is
    lazy per tensor, so a packed tensor's bytes are never read). Returns
    (bytes attached, parameter paths attached).

    THE RAW-TRUNK DOOR, on this engine: an FP8 checkpoint (any float8 plane
    in the shard headers — mx.load would hand one over as uint8 and say
    nothing — or an fp8 quantization_config) is refused BY NAME before a
    tensor is read, with the one line (codec/pack.refuse_checkpoint, torch-
    free). `name` is what the line calls it (the repo id), else the path."""
    from ..codec.pack import refuse_checkpoint

    files = sorted(glob.glob(os.path.join(snap, "*.safetensors")))
    if not files:
        raise FileNotFoundError(f"no *.safetensors under {snap}")
    refuse_checkpoint(name or snap, snap)
    attached: set[str] = set()
    nbytes = 0
    for fpath in files:
        for tname, arr in mx.load(fpath).items():
            mod_path, _, attr = tname.rpartition(".")
            if mod_path in packed and attr == "weight":
                continue  # the pack IS this tensor; never read its bytes
            if tie and tname == "lm_head.weight":
                continue  # tied: mlx-lm's Model has no lm_head, as_linear reads the embedding
            try:
                mod = _module_at(model, mod_path)
            except (AttributeError, IndexError, KeyError) as e:
                raise ValueError(f"checkpoint tensor {tname} has no module") from e
            if mod_path in packed:  # a bias arriving after the swap
                if attr != "bias":
                    raise ValueError(f"checkpoint tensor {tname} has no home on a packed Linear")
                t = arr.astype(mx.bfloat16)
                _check_bias_shape(tname, t, (mod.R,))
            elif attr in mod:
                t = arr.astype(mx.bfloat16) if mx.issubdtype(arr.dtype, mx.floating) else arr
            elif attr == "bias" and isinstance(mod, nn.Linear):
                # a bias for a Linear the skeleton built without one
                # (BiasedLinear's docstring): the module becomes the
                # one-rounding kind, keeping whatever weight it holds — the
                # checkpoint's if it already arrived, else the lazy init the
                # weight overwrites when it does (keys arrive sorted: bias first)
                t = arr.astype(mx.bfloat16)
                _check_bias_shape(tname, t, (mod["weight"].shape[0],))
                mod = BiasedLinear(mod["weight"], t)
                _set_at(model, mod_path, mod)
            else:
                raise ValueError(f"checkpoint tensor {tname} has no home on {mod_path}")
            mx.eval(t)
            mod[attr] = t
            attached.add(tname)
            nbytes += int(t.nbytes)
    return nbytes, attached


def _check_bias_shape(tname: str, t: mx.array, want: tuple) -> None:
    if tuple(t.shape) != tuple(want):
        raise ValueError(f"checkpoint tensor {tname} has shape {tuple(t.shape)}, "
                         f"not the {want} its Linear's rows want")


def load_compressed_mlx(repo: str, revision: str | None, pack_dir: str,
                        ctx: int | None = None, path: str | None = None,
                        rope_scaling: dict | None = None, snap: str | None = None) -> MLXEngine:
    """The fit path on the mlx runtime: mlx-lm skeleton + per-tensor
    streaming, bf16 weights of packed Linears never materialised. `snap`
    is the bound snapshot when the caller already resolved it (the bench,
    for every arm — checkpoint.resolve_source); None resolves it here."""
    from ..codec.identity import pack_id_from_meta
    from ..codec.pack import _check_format_version, iter_pack_dir, verify_hashes

    if rope_scaling is not None:
        raise ValueError("--rope-scaling is not implemented on the MLX engine (v0)")
    with open(os.path.join(pack_dir, "meta.json")) as f:
        pack_meta = json.load(f)
    # the format gate first, then the hashes (as the torch loader): a pack of
    # another version or without its hashes is refused before any tensor is read
    _check_format_version(pack_meta, pack_dir)
    verify_hashes(pack_dir)  # raises loudly on a mismatched or missing hash
    # ONE snapshot: the one the pack is bound to, or a
    # refusal naming both identities — before the skeleton is built.
    # config, tokenizer, raw tensors and generation defaults all read from it.
    if snap is None:
        snap = checkpoint.resolve_pack_source(repo, revision, pack_meta, pack_dir)
    config = _read_config(snap)
    refuse_unsupported(config)
    window = checkpoint.resolve_ctx(int(config.get("max_position_embeddings", 0) or 0), ctx)
    # THE ESTIMATE, before the skeleton exists and before any tensor is
    # read: descriptors only (module docstring, FIT CHECK 1). A model
    # that cannot fit is refused here, with nothing allocated.
    path = path or default_path()
    est = estimate_load_bytes(snap, pack_dir, bool(config.get("tie_word_embeddings", False)),
                              twin=path == TWIN)
    # the fused path holds one dense plane at a time, the reference path
    # every one at once, and the twin none: its planes are resident, in
    # est.resident, and its transient is one tensor's decode at load
    # (estimate_load_bytes)
    transient = (est.transient if path in ("fused", TWIN)
                 else max(est.transient, est.dense_total))
    fit_check(est.resident, kv_bytes_per_token(config), window, _working_set(),
              transient=transient, stage="estimate",
              prefill_per_token=prefill_bytes_per_token(config), host=_host_budget())
    model, args = build_skeleton(config)
    expected = parameter_paths(model)

    packed: set[str] = set()
    resident = 0
    n_radix = 0
    n_raw = 0
    t0 = time.perf_counter()
    for name, pack in iter_pack_dir(pack_dir):
        old = _module_at(model, name)
        if not isinstance(old, nn.Linear):
            raise ValueError(f"pack tensor {name} is not a Linear in this config")
        lin = make_module_mlx(pack, None, path=path, name=name)
        _set_at(model, name, lin)
        packed.add(name)
        if isinstance(lin, (RadixLinear, RadixTwinLinear)):
            n_radix += 1
        elif isinstance(lin, RawLinear):
            n_raw += 1
        resident += lin.resident_bytes()
        del pack

    nbytes, attached = _stream_checkpoint(model, snap, packed, bool(args.tie_word_embeddings),
                                          name=repo)
    resident += nbytes
    _check_materialized(expected, attached, packed)
    # THE MEASURED check (FIT CHECK 2): the modules' own accounting,
    # same transient, same ceilings — the truer resident number, second.
    fit_check(resident, kv_bytes_per_token(config), window, _working_set(),
              transient=transient, stage="measured",
              prefill_per_token=prefill_bytes_per_token(config), host=_host_budget())

    tok = checkpoint.tokenizer(snap, None)  # the bound snapshot's, not a re-resolve
    meta = {k: pack_meta.get(k) for k in ("hfRepo", "revision", "meanBpw", "weightedBpw", "profile")}
    meta["revision"] = checkpoint.pack_revision(pack_meta)  # /v1/models' revision
    meta["sourceDtype"] = pack_meta.get("sourceDtype")  # the released precision served
    meta["source"] = pack_meta.get("source")  # the bound snapshot
    meta["packId"] = pack_id_from_meta(pack_meta)  # torch-free
    meta["eos_token_ids"] = _eos_from_generation_config(snap)
    gd = gen_config.load(None, snap)
    mix = f" ({n_radix} radix, {n_raw} raw)"
    # the twin's decode is paid here, once, outside any timed pass
    once = (f", decoded to bf16 once at load in {time.perf_counter() - t0:.1f}s"
            if path == TWIN else "")
    print(f"[drinkme] device: {device_name()}, mlx {mx.__version__}, "
          f"compute path: {path}, {len(packed)} packed Linears{mix}{once}, "
          f"{resident / 1024**3:.2f} GiB resident", flush=True)
    return MLXEngine(model, tok, model_id=repo, arm="compressed", meta=meta,
                     ctx=window, gen_defaults=gd, path=path, resident_bytes=resident)


def load_stock_mlx(repo: str, revision: str | None, ctx: int | None = None,
                   snap: str | None = None) -> MLXEngine:
    """The control arm: the same MLXEngine over mlx-lm's own bf16 model —
    every tensor streamed from the checkpoint, no packed Linear anywhere.
    What `--stock --runtime mlx` serves and what the bench's stock arm
    times. `snap`: the already-resolved snapshot (the bench's one source
    for every arm); None resolves `repo`@`revision` here."""
    snap = snap or checkpoint.snapshot_dir(repo, revision)
    config = _read_config(snap)
    refuse_unsupported(config)
    window = checkpoint.resolve_ctx(int(config.get("max_position_embeddings", 0) or 0), ctx)
    # the estimate first, off the headers alone (FIT CHECK 1) ...
    est = estimate_load_bytes(snap, None, bool(config.get("tie_word_embeddings", False)))
    fit_check(est.resident, kv_bytes_per_token(config), window, _working_set(),
              transient=est.transient, stage="estimate",
              prefill_per_token=prefill_bytes_per_token(config), host=_host_budget())
    model, args = build_skeleton(config)
    expected = parameter_paths(model)
    nbytes, attached = _stream_checkpoint(model, snap, set(), bool(args.tie_word_embeddings),
                                          name=repo)
    _check_materialized(expected, attached, set())
    # ... then the measured one on what was attached (FIT CHECK 2)
    fit_check(nbytes, kv_bytes_per_token(config), window, _working_set(),
              transient=est.transient, stage="measured",
              prefill_per_token=prefill_bytes_per_token(config), host=_host_budget())
    tok = checkpoint.tokenizer(snap, None)  # the one snapshot's, not a re-resolve
    gd = gen_config.load(None, snap)
    print(f"[drinkme] device: {device_name()}, mlx {mx.__version__}, stock bf16, "
          f"{nbytes / 1024**3:.2f} GiB resident", flush=True)
    return MLXEngine(model, tok, model_id=repo, arm="stock",
                     meta={"eos_token_ids": _eos_from_generation_config(snap), "sourceDtype": "bf16"},
                     ctx=window, gen_defaults=gd, path="stock", resident_bytes=nbytes)


def _eos_from_generation_config(snap: str) -> list[int]:
    """The checkpoint's generation_config.json eos ids (Qwen3 lists two:
    <|im_end|> and <|endoftext|>) — HFEngine reads them off
    model.generation_config; there is no such object here, so the file."""
    p = os.path.join(snap, "generation_config.json")
    if not os.path.exists(p):
        return []
    try:
        with open(p) as f:
            e = json.load(f).get("eos_token_id")
    except (OSError, ValueError):
        return []
    if isinstance(e, int):
        return [e]
    if isinstance(e, list):
        return [int(x) for x in e]
    return []


def parameter_paths(model) -> set[str]:
    """Every parameter path in the tree right now — taken BEFORE loading, so
    the check below can name what the load never touched."""
    from mlx.utils import tree_flatten

    return {path for path, arr in tree_flatten(model.parameters())
            if isinstance(arr, mx.array)}


def _check_materialized(expected: set[str], attached: set[str], packed: set[str]) -> None:
    """Every parameter the skeleton was built with must have come from the
    checkpoint or the pack: mlx's lazy init means an unassigned one is still
    the random-init graph, and evaluating it would serve garbage with a 200
    status — engines.load_compressed's leftover-meta refusal, by bookkeeping
    rather than by device, since mlx has no meta device to leave a hole on."""
    owned = set(attached) | {f"{n}.weight" for n in packed}
    left = sorted(expected - owned)
    if left:
        raise ValueError(f"parameters never loaded from the checkpoint "
                         f"(incomplete checkpoint?): {left[:8]}"
                         f"{' ...' if len(left) > 8 else ''}")
