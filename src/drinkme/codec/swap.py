"""Module surgery: nn.Linear -> resident-compressed (or twin), bit-exact.

The one non-obvious rule, learned once and kept: Linears whose weight is tied
to an Embedding (e.g. a tied lm_head) are left raw — the embedding needs the
full table resident regardless, so compressing the tied head saves nothing
and severing it breaks the embedding.

THE MODULES (make_module picks by the tensor's own kind):
  RadixCompressedLinear  a radix tensor (codec/radix_pack.py) over
                         to_device_radix — the served bf16 path
  RawLinear              the raw fallback (bf16 bits verbatim) over
                         to_device_raw — stock's F.linear on every route
                         (the one-row call through the box's raw GEMV)
  RadixTwinLinear        the bench's twin arm: the uncompressed bf16 weight
                         through the served radix kernels (RAW=True) over
                         to_device_twin, at the twin's own launch rows
                         (radix_schedule.select_twin)
CompressedLinear is the shared base: the M dispatch, the epilogue, the CPU
reference route; the arm bodies are each subclass's own. The kernel route
(serving/kernel_route.py) also adopts raw Linears the codec does not hold:
NarrowLinear below the row threshold, StockLinear (the stock GEMV; in a
codec tree the raw GEMV, RAW_GEMV) at or above it.

Prefill routing (the M>1 story): forward() picks its arm by M, the product of
the input's leading dims — total rows through the matmul. M == 1 is the GEMV
kernel; M >= the dense threshold decodes the weight once (transient) and
runs stock's own F.linear; 2 <= M <= ops.MC_MAX in between goes to the
MULTI-COLUMN GEMV, which serves all M columns off ONE read of the compressed
weights, each column within the oracle's bound of the M=1 kernel (bench/
radix_mc_bitpin.py). A python row loop that calls the M=1 kernel once per
row — same numerics, M times the reads — survives only as the M > MC_MAX
fallback when the dense arm is switched off. The crossover is a MEASURED
quantity, never reasoned about: it is the ratio of two independently moving
arms and shifts with every kernel change and every platform. GEMM_MIN_ROWS
is MC_MAX + 1 (mc serves every M it can, see the class); whether dense
would beat mc below that on a given machine is what DRINKME_PREFILL_DENSE_MIN is
for — the number has not been swept with the radix mc kernel (no crossover
instrument is in bench/). On any other platform re-measure, and
DRINKME_PREFILL_DENSE_MIN is
the instrument — it overrides the threshold for a run (read once, at first
forward), and 0 or negative disables the dense arm entirely (for baselines,
or a machine where decode bandwidth makes dense a loss at any M). Measure with
it, then bake the result into GEMM_MIN_ROWS.
"""

from __future__ import annotations

import os
import warnings

import numpy as np
import torch

from .ops import MC_MAX
from .pack import FORMAT_VERSION, pack_weight_served

_radix_ops = None  # lazily-bound codec.radix_ops (triton), the radix arms' kernels


def _bind_radix_ops():
    global _radix_ops
    if _radix_ops is None:
        from . import radix_ops as _mod
        _radix_ops = _mod
    return _radix_ops


def to_device_raw(pack: dict, device: str = "cuda") -> dict:
    """The raw fallback tensor (codec/radix_pack.py: `codec: "raw"`, the bf16
    bits verbatim) -> its runtime dict: the bf16 weight resident on the
    device, and nothing else. pack.resident_bytes over it is 2 bytes/weight."""
    R, C = int(pack["R"]), int(pack["C"])
    bits = np.ascontiguousarray(pack["raw_bits"], dtype=np.uint16).reshape(R, C)
    w = torch.from_numpy(bits.view(np.int16).copy()).view(torch.bfloat16).to(torch.device(device))
    return {"weight": w, "R": R, "C": C, "codec": "raw",
            "format_version": FORMAT_VERSION,  # the version the tensor was written under
            "layout": 0}  # no code-plane layout: the codec scalar is the discriminator


def to_device_radix(pack: dict, device: str = "cuda") -> dict:
    """A radix tensor dict (codec/radix_pack.py: rx_palette / rx_offsets /
    rx_data + scalars) -> device-resident runtime buffers.

      rx_data      uint32 [NW + pad]  the block streams, verbatim, then `pad`
                                      zero words: the reader's reach past any
                                      block's start is at most the widest
                                      block's word count (radix
                                      .block_word_bounds(B)[1]: sip 608,
                                      gulp 768), so with that many words
                                      after the payload no read of a block
                                      short of its data-dependent streams
                                      leaves the allocation — the readers
                                      carry no per-load bound (a mask cost
                                      +2% on gfx1102; radix_gpu._read).
                                      NW is the payload's word count.
      rx_offsets   uint32 [NBK + 1]   the word directory, verbatim
      rx_palette   uint32 [...]       the per-tier lookup tables: each
                                      nonterminal tier's (1 << w) - 1 palette
                                      entries, zero-padded to whole uint32
                                      words, concatenated in tier order
                                      (radix_pack.padded_palette_words —
                                      what radix_gpu._lookup indexes)
      rx_schedule  uint16 [NBK]       gulp only (> 2 tiers): each block's
                                      nonterminal stream word lengths, so the
                                      scheduled decoder skips the per-tier
                                      escape-count reductions; computed
                                      on the device by the research kernel,
                                      on the CPU by radix_pack.schedule_np.
                                      Empty for sip.
      rx_launch    dict               the M=1 / mc launch schedule for this
                                      tensor's profile family and shape
                                      class (codec/radix_schedule.py): blocks per
                                      program and warps per arm. Not
                                      resident, not a pack field — the
                                      checkpoint bytes are the same whatever
                                      it says. DRINKME_RADIX_SCHEDULE
                                      overrides it for A/B runs.

    Every byte the arms read at decode time is in these four tensors, so
    pack.resident_bytes over this dict is the tensor's resident charge —
    the schedule and the pad included (the research's "active bytes"
    counted the schedule too; the pad is 4 x block_word_bounds(B)[1] bytes
    per tensor, 2.4-3 KB)."""
    from .radix import block_word_bounds
    from .radix_pack import _ARRAYS_RADIX, padded_palette_words, schedule_np
    from .radix_schedule import select as select_launch

    R, C = int(pack["R"]), int(pack["C"])
    widths = tuple(int(w) for w in pack["widths"])
    B = int(pack["block_size"])
    nblk = R * ((C + B - 1) // B)
    dev = torch.device(device)
    payload = np.ascontiguousarray(pack["rx_data"], dtype=np.uint32).reshape(-1)
    pad = int(block_word_bounds(B, 7, 8, widths)[1])
    data = torch.zeros(payload.size + pad, dtype=torch.uint32, device=dev)
    data[: payload.size].copy_(torch.from_numpy(payload))
    offsets = torch.from_numpy(np.ascontiguousarray(pack["rx_offsets"], dtype=np.uint32)).to(dev)
    palette = torch.from_numpy(padded_palette_words(pack)).to(dev)
    if len(widths) > 2:
        if dev.type == "cpu":
            schedule = torch.from_numpy(np.ascontiguousarray(schedule_np(pack)))
        else:
            schedule = _bind_radix_ops().prepare_schedule(data, offsets, C, widths, B, nblk)
    else:
        schedule = torch.empty(0, dtype=torch.uint16, device=dev)
    return {
        **{k: v for k, v in pack.items() if k not in _ARRAYS_RADIX},
        "rx_data": data,
        "rx_offsets": offsets,
        "rx_palette": palette,
        "rx_schedule": schedule,
        # the unpadded palette bytes (<= 21 of them) for the CPU reference
        # decoder — radix_pack._palette_u8; numpy on purpose, not resident
        "rx_palette_np": np.ascontiguousarray(pack["rx_palette"], dtype=np.uint8),
        "rx_launch": select_launch(R, C, B, widths).as_dict(),
        "NW": int(payload.size), "rx_pad_words": pad,
        "R": R, "C": C, "widths": widths, "block_size": B, "NBK": nblk,
        "codec": "radix",
        "format_version": FORMAT_VERSION,  # the version the tensor was written under
        "layout": 0,  # no code-plane layout; the codec scalar is the discriminator
    }


# the runtime dict's `codec` scalar for to_device_twin's dicts: not a pack
# codec (codec/registry.py knows radix and raw), the bench's twin only
TWIN = "twin"


def to_device_twin(w_bf16, widths, device: str = "cuda") -> dict:
    """The bench's order-matched twin of a radix tensor -> its runtime dict:
    the UNCOMPRESSED bf16 weight [R, C], row-major and resident, and the
    twin's launch schedule — radix_schedule.select_twin(R, C, B, widths):
    the box class's twin row where it has one (the gemv schedule that
    reads raw bf16 fastest there), else the row to_device_radix gives the
    radix tensor. radix_ops runs the served kernels over it with
    RAW=True: the decode replaced by a load of these bits at the same
    (row, column), the rest of the kernel the same source
    (radix_kernel_gpu._raw_weight), so at any one schedule the twin is
    bitwise the compressed kernel (bench/radix_twin_bitpin.py gates every
    table row, spike and twin row). `widths` is the compressed tensor's
    own (its profile's)."""
    from .radix_pack import RADIX_BLOCK
    from .radix_schedule import select_twin as select_launch

    w = w_bf16.detach()
    if w.dtype != torch.bfloat16:
        raise TypeError(f"to_device_twin: {w.dtype} is not bf16")
    R, C = int(w.shape[0]), int(w.shape[1])
    widths = tuple(int(x) for x in widths)
    B = RADIX_BLOCK
    return {
        "weight": w.contiguous().to(torch.device(device)),
        "rx_launch": select_launch(R, C, B, widths).as_dict(),
        "R": R, "C": C, "widths": widths, "block_size": B,
        "NBK": R * ((C + B - 1) // B),
        "codec": TWIN,
        "layout": 0,
    }


# DRINKME_PREFILL_DENSE_MIN, parsed once (first forward) and cached; None =
# not yet read. Tests reset this to force a re-read; production reads the
# environment exactly once, so the router costs an int compare per forward,
# not a getenv.
_DENSE_MIN: int | None = None


def _dense_min() -> int:
    """The dense-branch row threshold: DRINKME_PREFILL_DENSE_MIN if set, else
    GEMM_MIN_ROWS (the reference box's measured crossover). <= 0 disables the
    dense arm entirely. Read ONCE, then cached — see the module docstring for
    why this is an instrument (per-platform measurement), not a tuning knob to
    leave set. A malformed value warns and falls back to the measured default:
    a typo in an env var must not change serve behaviour silently or fatally."""
    global _DENSE_MIN
    if _DENSE_MIN is None:
        raw = os.environ.get("DRINKME_PREFILL_DENSE_MIN", "").strip()
        try:
            _DENSE_MIN = int(raw) if raw else CompressedLinear.GEMM_MIN_ROWS
        except ValueError:
            warnings.warn(
                f"DRINKME_PREFILL_DENSE_MIN={raw!r} is not an integer; using "
                f"the measured default {CompressedLinear.GEMM_MIN_ROWS}")
            _DENSE_MIN = CompressedLinear.GEMM_MIN_ROWS
    return _DENSE_MIN


class CompressedLinear(torch.nn.Module):
    """The resident-compressed Linear's shared half: the M dispatch (_route),
    the fp32-accumulate-then-round-once epilogue, the CPU reference route
    and the M > MC_MAX loop. The four arm bodies (_gemv_row, _gemv_cols,
    _dense_weight, _cpu_weight) are each subclass's — RadixCompressedLinear,
    RawLinear below — and every routing and epilogue decision
    stays one code path per class tree."""

    # Rows at which prefill switches to decode-once + native GEMM (the
    # transient decode + F.linear). This separates DENSE from the
    # multi-column GEMV, not from the row loop: 2 <= M <= MC_MAX is mc's, and
    # mc is FLAT in weight reads like dense is, so the threshold is the
    # narrow question — does dense beat mc anywhere at or below MC_MAX? It is
    # MC_MAX + 1 by construction, and not on speed alone: mc is bitwise the
    # M=1 kernel per column and dense is not, so uniform mc through MC_MAX
    # keeps every compressed Linear of an MTP verify batch in one numerics
    # window, the serial decode step's. Routing per shape (dense beats mc
    # earlier on the small square attention projections than on the MLP
    # tensors that dominate a layer) might win a few percent; that has not
    # been measured.
    # The whole-layer crossover has not been swept with the radix mc kernel;
    # DRINKME_PREFILL_DENSE_MIN overrides this at runtime (module docstring +
    # _dense_min): the per-platform measurement instrument, 0/negative = off.
    GEMM_MIN_ROWS = 9  # MC_MAX + 1

    def __init__(self, runtime: dict, bias):
        super().__init__()
        self.p = runtime
        self.bias = bias  # kept bf16 (tiny)
        self._w_cpu = None  # lazy CPU reference weight; see forward()
        self.R, self.C = int(runtime["R"]), int(runtime["C"])
        # codec/registry.py's tensor kind ("radix" | "raw"): the runtime dict's
        # `codec` scalar, the same discriminator the subclasses check.
        self.tensor_codec = runtime.get("codec")
        self.layout = int(runtime.get("layout", 0))  # no code-plane layouts: always 0

    # -- the four arm bodies, each subclass's own --

    def _gemv_row(self, xi):
        """The M=1 op on one fp32 row [C] -> fp32 [R]."""
        raise NotImplementedError

    def _gemv_cols(self, xm):
        """The multi-column op on fp32 [M, C] -> fp32 [M, R]."""
        raise NotImplementedError

    def _dense_weight(self):
        """The bf16 weight [R, C] on the device, TRANSIENT — the dense arm's
        one decode per forward."""
        raise NotImplementedError

    def _cpu_weight(self):
        """The bf16 weight on the CPU, cached by the caller — the reference
        path's F.linear operand."""
        raise NotImplementedError

    def _epilogue(self, y, x, shape):
        """The GEMV arms' shared tail: fp32 accumulator, PLUS the bias in
        fp32, then ONE rounding to the activation dtype.

        Adding the bias AFTER `y.to(x.dtype)` — bf16 + bf16, a second
        rounding — would lose the residual a biased projection's accumulator
        carries when the bias then cancels it (1 + 1/256 rounded to bf16 is
        1.0; minus 1 is 0.0 where stock's fused epilogue keeps 0.00390625).
        Stock's F.linear(x, W, b) adds the bias to the fp32 accumulator
        before its single cast, and this is the same arithmetic: the bf16
        bias promotes into the fp32 `y` (one add, exact widening), then one
        cast. Every arm rounds once, and the M=1 / multi-column contract (mc
        is bitwise the M=1 kernel per column) holds through it because both
        arrive here with the same fp32 rows."""
        if self.bias is not None:
            y = y + self.bias
        return y.to(x.dtype).reshape(*shape[:-1], self.R)

    def _route(self, x) -> str:
        """Which forward arm serves this input: "cpu" | "gemv" | "mc" |
        "dense" | "loop". Split from forward so the choice is assertable in
        tests without a GPU — and forceable, which is how the pure-torch dense
        arm gets exercised end-to-end on CPU."""
        if x.device.type == "cpu":
            return "cpu"
        return self._route_rows(x.numel() // x.shape[-1])

    def _route_rows(self, m: int) -> str:
        """The threshold arithmetic, on M = product of the input's leading
        dims (total rows through the matmul). M == 1 always takes the single
        GEMV op, whatever the threshold says — decoding a whole weight to
        multiply one row is never the move.

        2 <= M <= MC_MAX goes to the multi-column kernel unconditionally,
        because it STRICTLY DOMINATES the row loop it replaced: identical
        per-column numerics (bit-pinned) at 1/M the weight reads. There is no
        threshold to measure between those two — the only measured crossover
        left is mc vs dense, which is what GEMM_MIN_ROWS now carries.

        The row loop survives for M > MC_MAX with the dense arm disabled
        (DRINKME_PREFILL_DENSE_MIN=0, the per-platform baseline instrument):
        somebody has to serve those rows, and one M=1 kernel per row is still
        correct, just slow.
        """
        if m == 1:
            return "gemv"
        dense_min = _dense_min()
        if 0 < dense_min <= m:
            return "dense"
        if m <= MC_MAX:
            return "mc"
        return "loop"

    def forward(self, x):
        return self._forward(x, self._route(x))

    def _forward(self, x, route: str):
        """forward() at a decided route — the one place the arms are wired,
        so a subclass with a fused route of its own (RadixCompressedLinear)
        consults _route exactly once."""
        if route == "cpu":
            # CPU = reference/test path (triton needs a GPU): decode the exact
            # bf16 bytes once, cache, and use the same F.linear stock uses —
            # so on CPU the compressed arm is bit-identical to stock by
            # construction. Materializing the weight here is fine ONLY because
            # this path never counts against the fit check.
            if self._w_cpu is None:
                self._w_cpu = self._cpu_weight()
            return torch.nn.functional.linear(x, self._w_cpu, self.bias)

        shape = x.shape
        xf = x.reshape(-1, self.C)
        rowsn = xf.shape[0]
        if route == "gemv":
            # batch-1 decode: the hot path, one op, no python loop to trace
            xi = xf[0].float().contiguous()
            out = self._epilogue(self._gemv_row(xi), x, shape)
        elif route == "dense":
            # THE PREFILL PATH: decode the weight once —
            # transient, one layer's worth — and run the SAME F.linear stock
            # runs. The codec is LOSSLESS: the decoded bytes are
            # bit-identical to the stock checkpoint (pinned against the CPU
            # reference decoder in the gates), so this arm's numerics ARE stock's own
            # matmul numerics — bitwise-equal to F.linear(x, W_stock) at the
            # same dtype/device, with only accumulation-order-level
            # differences vs the GEMV arm (allclose, never bitwise). One
            # decode + a native GEMM, amortized over M rows, in place of M
            # GEMV launches. The bias goes INTO the call: stock's biased
            # F.linear is one fused epilogue — fp32 accumulate, add bias,
            # round once — and adding it to the already-rounded bf16 output
            # afterwards would be a second rounding that differs from stock
            # on every element of a cancellation-shaped input.
            W = self._dense_weight()
            out = torch.nn.functional.linear(xf, W, self.bias).reshape(*shape[:-1], self.R)
            del W  # NO caching — VRAM is the product; one transient layer only
        elif route == "mc":
            # THE MULTI-COLUMN ARM: one read of the weights, M
            # columns of activations, each column bitwise the M=1 kernel's
            # answer (_gemv_cols).
            xm = xf.float().contiguous()
            out = self._epilogue(self._gemv_cols(xm), x, shape)
        else:
            # M > MC_MAX with the dense arm disabled: one GEMV call per row.
            # Correct, and the only arm left that reads the weights M times.
            outs = torch.empty(rowsn, self.R, device=x.device, dtype=torch.float32)
            for i in range(rowsn):
                outs[i] = self._gemv_row(xf[i].float().contiguous())
            out = self._epilogue(outs, x, shape)
        return out


class RadixCompressedLinear(CompressedLinear):
    """A radix-coded Linear (codec/radix_pack.py, codec/radix_ops.py) — THE
    served bf16 module: the M dispatch (_route /
    _dense_min / MC_MAX), the fp32-accumulate-then-round-once epilogue, the
    CPU reference route and the M > MC_MAX loop inherited; the arm bodies
    radix's own:

      gemv     radix_ops.gemv_fused — the scheduled decoder at the tensor's
               launch-table row (codec/radix_schedule.py), fused epilogue
               (bias in fp32, one rounding, bf16 out) — so this route
               bypasses _epilogue and the host-side `.float()` on x
      mc       radix_ops.gemv_mc — the multi-column kernel: each block
               decoded once and applied to all M rows; fp32 [M, R] into the
               inherited _epilogue (the same bias-then-round arithmetic
               _finish does for M=1)
      dense    radix_ops.decode_weight -> F.linear (the prefill)
      cpu      radix_pack.decode_back_radix -> F.linear, the reference

    `tensor_codec` is the registry's "radix" row."""

    def __init__(self, runtime: dict, bias):
        torch.nn.Module.__init__(self)
        if runtime.get("codec") != "radix":
            raise ValueError("RadixCompressedLinear needs a to_device_radix runtime dict")
        self.p = runtime
        self.bias = bias
        self._w_cpu = None
        self.R, self.C = int(runtime["R"]), int(runtime["C"])
        self.tensor_codec = "radix"
        self.layout = 0

    def _gemv_row(self, xi):
        return _bind_radix_ops().gemv(self.p, xi)

    def _gemv_cols(self, xm):
        return _bind_radix_ops().gemv_mc(self.p, xm)

    def _dense_weight(self):
        if self.p["rx_data"].device.type == "cpu":
            return self._cpu_weight()
        return _bind_radix_ops().decode_weight(self.p)

    def _cpu_weight(self):
        from .radix_pack import decode_back_radix

        U = decode_back_radix(self.p)
        return torch.from_numpy(U.view(np.int16).copy()).view(torch.bfloat16)

    def _forward(self, x, route: str):
        if route in ("gemv", "mc") and x.dtype in (torch.bfloat16, torch.float32):
            ops = _bind_radix_ops()
            shape = x.shape
            xf = x.reshape(-1, self.C)
            if route == "gemv":
                # the fused M=1 kernel: bf16 (or fp32) x read in-kernel, bias
                # added to the fp32 accumulator, ONE rounding to x.dtype
                out = ops.gemv_fused(self.p, xf[0], self.bias, x.dtype)
                return out.reshape(*shape[:-1], self.R)
            # the multi-column kernel reads x's own dtype too — no `.float()`
            # launch; fp32 rows out, then the inherited bias-then-round tail
            return self._epilogue(ops.gemv_mc(self.p, xf), x, shape)
        return super()._forward(x, route)


class RawLinear(CompressedLinear):
    """The raw fallback (codec/radix_pack.py: a tensor radix would have
    expanded, stored as its bf16 bits): a plain bf16 Linear over the resident
    weight — stock's own F.linear on every route, every device, every M
    (the one-row call through the box's stock GEMV, or the twin kernel
    under stock GEMV "triton" or raw GEMV "twin", once the kernel route
    has set `stock_gemv` and `raw_gemv`: stock_linear). A
    CompressedLinear to everything that walks the tree (the checkpoint
    walker skips its weight, the head diet counts it, the bench names it),
    with no kernel of its own: `tensor_codec` is the registry's "raw" row."""

    stock_gemv = "linear"
    raw_gemv = None  # raw_gemv_dict's dict when the one-row call is the twin kernel (install_stock_gemv)

    def __init__(self, runtime: dict, bias):
        torch.nn.Module.__init__(self)
        if runtime.get("codec") != "raw":
            raise ValueError("RawLinear needs a to_device_raw runtime dict")
        self.p = runtime
        self.bias = bias
        self._w_cpu = None
        self.R, self.C = int(runtime["R"]), int(runtime["C"])
        self.tensor_codec = "raw"
        self.layout = 0

    def _gemv_row(self, xi):
        return torch.nn.functional.linear(xi, self.p["weight"].float())

    def _gemv_cols(self, xm):
        return torch.nn.functional.linear(xm, self.p["weight"].float())

    def _dense_weight(self):
        return self.p["weight"]

    def _cpu_weight(self):
        return self.p["weight"].cpu()

    def forward(self, x):
        return stock_linear(x, self.p["weight"].to(x.device), self.bias, self.stock_gemv, self.raw_gemv)


class RadixTwinLinear(RadixCompressedLinear):
    """The bench's TWIN arm: the uncompressed bf16 weight (to_device_twin)
    through RadixCompressedLinear's own routing — _route, _forward, the
    fused M=1 epilogue, the mc arm, _epilogue, the dense threshold — and
    the served radix kernels at the compressed tensor's launch schedule,
    with the decode replaced by a load of the raw weight (radix_ops,
    RAW=True). Only the arm bodies that produce a weight differ: dense
    (M >= GEMM_MIN_ROWS) runs F.linear over the resident weight itself, and
    the CPU route uses it as it is. The difference from the compressed arm
    is the weight read: raw bf16 in place of the radix streams and their
    decode."""

    def __init__(self, runtime: dict, bias):
        torch.nn.Module.__init__(self)
        if runtime.get("codec") != TWIN:
            raise ValueError("RadixTwinLinear needs a to_device_twin runtime dict")
        self.p = runtime
        self.bias = bias
        self._w_cpu = None
        self.R, self.C = int(runtime["R"]), int(runtime["C"])
        self.tensor_codec = TWIN
        self.layout = 0

    def _dense_weight(self):
        return self.p["weight"]

    def _cpu_weight(self):
        return self.p["weight"].cpu()


class NarrowLinear(torch.nn.Linear):
    """A raw bf16 Linear BELOW the codec's row threshold (`eligible`: min dim
    >= 1024), served through the Triton bf16 GEMV instead of the BLAS GEMM.
    On Qwen3.8-27B these are `in_proj_a` /
    `in_proj_b`, 48 x 5120 per DeltaNet layer — one scalar per head for the
    decay and the beta gate — 96 per token (MiMo-V2.6-Distill-Qwen-9B's are
    32 x 4096, 48 per token; Muse-Glimmer-30B's k_proj / v_proj are
    256 x 6656, 104), and PyTorch's default F.linear on gfx1151 (rocBLAS)
    answers M=1, N=48, K=5120 with a 128x128 macro-tile: ONE workgroup
    walking K, 108-140 us per launch, 11.8 ms of the 27B's 210 ms serial
    step and 11.0 ms of every M=5 verify (rocprof). The
    two tensors are 0.5 MB each: ~2 us at the wall.

    Not a codec tensor: the pack, its hashes and the walker are untouched —
    the checkpoint's own bf16 weight and bias stay this module's
    parameters (the same Parameter objects; `adopt` copies nothing), so
    the pack of record serves as it is. radix_ops.gemv_narrow runs M <=
    MC_MAX rows in ONE launch of R programs (fp32 accumulate, the bias
    added in fp32, one rounding — CompressedLinear._epilogue's order, the
    served radix kernel's own); above MC_MAX (prefill) and off the
    accelerator it is stock's F.linear, as before. Row m of an M-row call
    is the M=1 call's result bit for bit (the kernel's docstring; pinned),
    which is what the MTP verify's "batched projections, replayed
    recurrence" rests on. Bitwise equality with F.linear is NOT claimed —
    a different reduction order, both fp32-accumulated; the max |Δ| on the
    real weights is measured (bench/narrow_linear_numerics.py), not
    assumed."""

    # the launch config: BLOCK-wide chunks per program at NUM_WARPS. Over
    # BLOCK {256, 512, 1024} x warps {1, 2, 4, 8} (gfx1151, device us per
    # call with the weight cold, M = 1, 2, 4, 6, 8), 512 / 4 is the one
    # config within 1% of the best on 256 x 6656 at every M; 1024 / 8 is
    # 0.78-0.89x its time on 48 x 5120 and 32 x 4096 and 1.05-1.14x on
    # 256 x 6656 at M >= 4. One config serves every shape.
    # Against every BLAS route, per M (bench/narrow_linear_knobs.py,
    # gfx1151, device us per call with the weight cold, on 48 x 5120 /
    # 32 x 4096 / 256 x 6656): this kernel 9.5-11.4 / 8.9-10.6 / 20.3-23.0
    # at M = 1..8; hipBLASLt F.linear 22-24 / 24-25 / 33; torch.mv under
    # hipBLASLt (stock_linear's one-row call, M=1 only) 30 / 26 / 40, and
    # once per row 49-247 at M = 2..8; rocBLAS F.linear, PyTorch's default
    # there, 119-120 / 97-100 / 168-170. It wins at every M on every shape,
    # so it serves M = 1..MC_MAX. Host-inclusive, back to back, hipBLASLt's
    # launch is the cheaper one on 32 x 4096 (11.7-12.0 us per call against
    # 12.3-12.6): that decides only a host-bound loop, and the models with
    # these shapes (9B to 30B) decode device-bound. On an NVIDIA L4
    # (bench/modal_narrow_ab.py) it is 5.3-15.6 us on the two small shapes
    # against cuBLAS F.linear's 6.7-38.7, and 17.2-26.3 on 256 x 6656
    # against 20.9-23.4: slower there at M = 4 and 5.
    BLOCK = 512
    NUM_WARPS = 4
    # what this class is for: rows the BLAS path tiles as one workgroup,
    # under the codec's own threshold; columns long enough that a program
    # per row streams a real row. Anything else keeps F.linear.
    MAX_ROWS = 1024
    MIN_COLS = 1024

    @classmethod
    def wants(cls, mod) -> bool:
        if type(mod) is not torch.nn.Linear or eligible(mod):
            return False
        w = mod.weight
        return (w.dtype == torch.bfloat16 and w.device.type == "cuda"
                and w.shape[0] <= cls.MAX_ROWS and w.shape[1] >= cls.MIN_COLS
                and w.shape[1] % 4 == 0)

    @classmethod
    def adopt(cls, lin: torch.nn.Linear) -> "NarrowLinear":
        """The same parameters under this class — no copy, no re-stream."""
        new = cls(lin.in_features, lin.out_features, bias=lin.bias is not None, device="meta")
        new.weight = lin.weight
        new.bias = lin.bias
        return new

    def forward(self, x):
        shape = x.shape
        xm = x.reshape(-1, self.in_features)
        M = xm.shape[0]
        if M > MC_MAX or x.device.type != "cuda":
            return torch.nn.functional.linear(x, self.weight, self.bias)
        y = _bind_radix_ops().gemv_narrow(self.weight, xm, self.bias, self.BLOCK, self.NUM_WARPS)
        return y.reshape(*shape[:-1], self.out_features)


def install_narrow(model, device: str) -> list[tuple[str, tuple[int, int]]]:
    """Every raw Linear NarrowLinear.wants, adopted in place; returns the
    (name, (R, C)) list so the loader can say what it did. Nothing on CPU
    (the kernel is triton), nothing for a tensor the codec holds."""
    if not str(device).startswith("cuda"):
        return []
    done = []
    for parent_name, parent in list(model.named_modules()):
        for child_name, child in list(parent.named_children()):
            if NarrowLinear.wants(child):
                setattr(parent, child_name, NarrowLinear.adopt(child))
                name = f"{parent_name}.{child_name}" if parent_name else child_name
                done.append((name, tuple(child.weight.shape)))
    return done


# THE STOCK GEMV: what a raw bf16 Linear's single-row forward (decode, M=1)
# calls, per gfx target. The stock arm's every Linear and `serve --stock`'s
# run it, so it is the baseline the published ratio divides by: the fastest
# one-row bf16 call drinkme has on the box. PyTorch's default F.linear on
# gfx1151 is rocBLAS (preferred_blas_library "cublas"; hipBLASLt is not its
# default there), a 128x128-tile Tensile GEMM for a one-row problem.
# bench/stock_blas_knobs.py timed every knob that needs no kernel of ours on
# every decode Linear of Qwen3-0.6B / -1.7B / -8B (a quiet box, rotation
# protocol, device ms per token's Linears; the fraction of the 239 GB/s
# wall): rocBLAS F.linear 18.4 / 55.5 / 130.3 (0.26-0.49); hipBLASLt
# F.linear 10.9 / 33.0 / 96.7; TunableOp (tuned in 7 / 11 / 24 s, a CSV per
# box) 10.3 / 30.3 / 94.2; torch.mv under hipBLASLt 7.3 / 22.0 / 83.0
# (0.66-0.76), and no tuning; on the 27B's shapes (under load) TunableOp led
# torch.mv by 3% after 58 s of tuning. That made "mv" (torch.mv with
# hipBLASLt preferred for that call alone) gfx1151's stock GEMV until the
# same instrument timed "triton", the twin arm's kernel over the raw weight
# (radix_ops.gemv_fused with RAW=True at radix_schedule.select_twin's row,
# the RAW GEMV below): on gfx1151 it took 0.63-0.80x torch.mv's time on
# every shape of Qwen3-1.7B, Qwen3-8B and MiMo-V2.6-Distill-Qwen-9B, 15.1 /
# 63.7 / 66.2 ms per token's Linears against 22.0 / 83.0 / 84.8 (0.95-1.0
# of the wall), so gfx1151's stock GEMV is "triton"
# (docs/serve-kernels.md#stock-gemv). On gfx1102 (RX 7600 XT) the same
# kernel took 0.36-0.97x F.linear's (rocBLAS) time on every shape of
# Qwen3-0.6B, 1.7B, 4B and 8B (0.49-0.84x per token's Linears), and end to end
# the stock arm decoded 1.40x (1.7B) and 1.44x (4B) faster; the 0.6B, whose
# decode is host-bound there, was 4% slower, so its baseline is a little
# softer under "triton" than under F.linear. gfx1102 runs at the compressed
# tensor's sip row, as it has no twin rows of its own. Prefill and M = 2..8
# keep F.linear on the default library under every mode. Other targets were
# not measured and keep F.linear. DRINKME_STOCK_GEMV=linear | mv | triton
# overrides.
STOCK_GEMV = {"gfx1151": "triton", "gfx1102": "triton"}
STOCK_GEMV_ENV = "DRINKME_STOCK_GEMV"
STOCK_GEMV_MODES = ("linear", "mv", "triton")


def stock_gemv_mode(device: str) -> str:
    """The stock GEMV for this process's device: DRINKME_STOCK_GEMV when
    set to a mode, else STOCK_GEMV's entry for the gfx target, else
    "linear"; always "linear" off the accelerator."""
    spec = os.environ.get(STOCK_GEMV_ENV, "auto").strip() or "auto"
    if spec not in ("auto",) + STOCK_GEMV_MODES:
        raise ValueError(f"{STOCK_GEMV_ENV}={spec!r}: one of auto, {', '.join(STOCK_GEMV_MODES)}")
    if not str(device).startswith("cuda"):
        return "linear"
    if spec != "auto":
        return spec
    from .radix_schedule import _arch

    return STOCK_GEMV.get(_arch(), "linear")


def stock_linear(x, weight, bias, mode: str, raw: dict | None = None):
    """A raw Linear's forward under the stock GEMV `mode`: F.linear, or for
    "mv" a single row on the accelerator through torch.mv (torch.addmv
    with a bias) with hipBLASLt preferred for this call only, the
    process's preference restored after (a 0.3 us pair of setter calls).
    `raw`, the module's twin dict (raw_gemv_dict: install_stock_gemv gives
    one under stock GEMV "triton", and to a codec tree's raw Linears under
    raw GEMV "twin"), sends a single bf16 row on the accelerator through
    the twin's kernel instead (fp32 accumulate, the bias added in fp32, one
    rounding); "triton" without one builds it for the weight's shape. Every
    other call is F.linear, or torch.mv under "mv"."""
    if raw is None and mode == "triton":
        raw = raw_gemv_dict(int(weight.shape[0]), int(weight.shape[1]))
    if raw is not None and x.device.type == "cuda" and x.numel() == weight.shape[1] \
            and x.dtype == torch.bfloat16 and weight.dtype == torch.bfloat16 and weight.is_contiguous():
        out = _bind_radix_ops().gemv_fused(dict(raw, weight=weight), x.reshape(-1), bias, x.dtype)
        return out.reshape(*x.shape[:-1], weight.shape[0])
    if mode != "mv" or x.device.type != "cuda" or x.numel() != weight.shape[1]:
        return torch.nn.functional.linear(x, weight, bias)
    prev = torch._C._get_blas_preferred_backend()
    torch._C._set_blas_preferred_backend(torch._C._BlasBackend.Cublaslt)
    try:
        v = x.reshape(-1)
        y = torch.mv(weight, v) if bias is None else torch.addmv(bias, weight, v)
    finally:
        torch._C._set_blas_preferred_backend(prev)
    return y.reshape(*x.shape[:-1], weight.shape[0])


class StockLinear(torch.nn.Linear):
    """A raw bf16 Linear at or above the codec's row threshold (both dims
    >= 1024: the shapes STOCK_GEMV was measured on) whose forward is
    stock_linear's: the one-row call through the box's stock GEMV (in a
    codec tree under raw GEMV "twin", the twin kernel: RAW_GEMV), every
    other M through F.linear.
    The stock arm's every Linear, and in the compressed and twin arms the
    Linears the codec leaves raw (a tied lm_head). Not a codec tensor: the
    checkpoint's own Parameters, adopted as they are (a tied head stays
    tied)."""

    stock_gemv = "mv"
    raw_gemv = None  # raw_gemv_dict's dict when the one-row call is the twin kernel (install_stock_gemv)
    MIN_DIM = 1024

    @classmethod
    def wants(cls, mod) -> bool:
        if type(mod) is not torch.nn.Linear:
            return False
        w = mod.weight
        return w.dtype == torch.bfloat16 and w.device.type == "cuda" and min(w.shape) >= cls.MIN_DIM

    @classmethod
    def adopt(cls, lin: torch.nn.Linear, mode: str) -> "StockLinear":
        """The same parameters under this class — no copy, no re-stream."""
        new = cls(lin.in_features, lin.out_features, bias=lin.bias is not None, device="meta")
        new.weight = lin.weight
        new.bias = lin.bias
        new.stock_gemv = mode
        return new

    def forward(self, x):
        return stock_linear(x, self.weight, self.bias, self.stock_gemv, self.raw_gemv)


# THE RAW GEMV: what a raw bf16 Linear in a CODEC TREE (codec_tree: the
# compressed arm, the twin and `drinkme serve` over a pack; a tied lm_head,
# a raw fallback tensor) calls for its one-row forward, per gfx target:
# "stock", the stock GEMV above, or "twin", the kernel the twin arm reads
# raw bf16 with (radix_ops.gemv_fused with RAW=True at
# radix_schedule.select_twin's row). bench/raw_head_gemv.py timed the tied
# lm_heads of Qwen3-0.6B / -1.7B / -4B and gemma-4-31B-it (151936 x 1024 /
# 2048 / 2560, 262144 x 5376) on gfx1151 (rotation protocol,
# device ms per call): torch.mv under hipBLASLt 1.485 / 3.762 / 3.470 /
# 12.220; the twin's kernel at its head row (the whole row at eight warps)
# 1.249 / 2.527 / 3.224 / 11.479, 0.84 / 0.67 / 0.93 / 0.94 of torch.mv and
# within 2% of the best tiles x warps on each; hipBLASLt F.linear 0.88-0.96
# of torch.mv. The twin arm runs every Linear through that kernel. So on
# gfx1151 a codec tree's raw Linears make their one-row call through it;
# the stock arm and `serve --stock` (no codec in the tree) make theirs
# through the stock GEMV, which on gfx1151 is the same kernel ("triton").
# Other targets were not measured and keep "stock".
# DRINKME_RAW_GEMV=stock | twin overrides.
RAW_GEMV = {"gfx1151": "twin"}
RAW_GEMV_ENV = "DRINKME_RAW_GEMV"
RAW_GEMV_MODES = ("stock", "twin")


def raw_gemv_mode(device: str) -> str:
    """The raw GEMV for this process's device: DRINKME_RAW_GEMV when set
    to a mode, else RAW_GEMV's entry for the gfx target, else "stock";
    always "stock" off the accelerator."""
    spec = os.environ.get(RAW_GEMV_ENV, "auto").strip() or "auto"
    if spec not in ("auto",) + RAW_GEMV_MODES:
        raise ValueError(f"{RAW_GEMV_ENV}={spec!r}: one of auto, {', '.join(RAW_GEMV_MODES)}")
    if not str(device).startswith("cuda"):
        return "stock"
    if spec != "auto":
        return spec
    from .radix_schedule import _arch

    return RAW_GEMV.get(_arch(), "stock")


def codec_tree(model) -> bool:
    """Does the tree hold a codec module (CompressedLinear: the compressed
    arm's radix and raw tensors, the twin's)? The stock arm's and `serve
    --stock`'s do not: their raw Linears are all the stock GEMV's, whatever
    the raw GEMV."""
    return any(isinstance(m, CompressedLinear) for m in model.modules())


def raw_gemv_dict(R: int, C: int) -> dict:
    """The twin dict (to_device_twin's, less the weight) a raw [R, C]
    Linear's one-row call runs radix_ops.gemv_fused over: the twin's
    launch row for the shape (radix_schedule.select_twin), sip's widths
    (RAW=True reads none of the decoder's constants). stock_linear adds the
    module's own weight per call, so the dict holds no tensor."""
    from .radix_pack import RADIX_BLOCK
    from .radix_schedule import select_twin

    widths = (3, 8)
    return {"rx_launch": select_twin(R, C, RADIX_BLOCK, widths).as_dict(), "R": R, "C": C,
            "widths": widths, "block_size": RADIX_BLOCK, "codec": TWIN, "layout": 0}


def install_stock_gemv(model, device: str, mode: str, raw: str = "stock") -> int:
    """The stock GEMV `mode` onto the tree: every raw Linear StockLinear
    wants adopted, and every RawLinear (the codec's raw fallback, stock's
    F.linear) told the mode. Under stock GEMV "triton", and in a codec tree
    under raw GEMV "twin", each of them is also given its raw_gemv_dict,
    and its one-row call is the twin's kernel. Returns how many modules
    run a route. Nothing off the accelerator, or for stock GEMV "linear"
    where the raw GEMV does not apply (F.linear is what they already
    run)."""
    if not str(device).startswith("cuda"):
        return 0
    twin = mode == "triton" or (raw == "twin" and codec_tree(model))
    if mode == "linear" and not twin:
        return 0
    done = 0
    for _parent_name, parent in list(model.named_modules()):
        for child_name, child in list(parent.named_children()):
            if StockLinear.wants(child):
                child = StockLinear.adopt(child, mode)
                setattr(parent, child_name, child)
            elif isinstance(child, (StockLinear, RawLinear)):
                child.stock_gemv = mode
            else:
                continue
            R, C = (child.out_features, child.in_features) if isinstance(child, StockLinear) \
                else (child.R, child.C)
            child.raw_gemv = raw_gemv_dict(R, C) if twin else None
            done += 1
    return done


# What a swapped Linear may BE, for the one checkpoint walker
# (serving/engines.stream_checkpoint): a module of one of these types at a
# Linear's slot already holds its weight (packed, or the twin's raw bf16),
# so the walker skips the checkpoint's `.weight` for it and attaches only
# the bias — the same bias-after-swap the serve loader has always done for
# CompressedLinear. The bench's twin arm (RadixTwinLinear, RawLinear) is a
# CompressedLinear too, so it streams through the same walker
# (arms.load_twin_streaming): a second walker is exactly what the
# bit-identical claim forbids (one key mapping, one opinion on what is
# computed vs loaded).
SWAPPED_LINEAR_TYPES = (CompressedLinear,)


def eligible(child) -> bool:
    if not isinstance(child, torch.nn.Linear):
        return False
    w = child.weight.data
    return w.dtype == torch.bfloat16 and w.shape[1] % 4 == 0 and min(w.shape) >= 1024


def eligible_linears(model):
    """Yield (qualified_name, parent, child_name, module) for every Linear the
    codec should touch — eligible shape/dtype AND not weight-tied to an
    Embedding (the tied-lm_head rule).

    Works on real AND meta-device models, which is what lets pack_model decide
    eligibility from a zero-byte skeleton. The trap: on
    meta EVERY tensor's data_ptr() is 0, so a storage-pointer tied check calls
    every Linear tied and the pack comes out empty. tie_weights() makes the
    tied head the SAME Parameter object, so object identity is the meta-safe
    check; storage identity stays as the belt-and-braces check on real models.

    Root-level Linears (an untied lm_head) are emitted DOT-FREE ("lm_head",
    not ".lm_head"), which is the name the loader's get_submodule takes."""
    embs = [m.weight for m in model.modules() if isinstance(m, torch.nn.Embedding)]
    tied_ids = {id(w) for w in embs}
    tied_ptrs = {w.data_ptr() for w in embs if not w.is_meta}
    for parent_name, parent in model.named_modules():
        for child_name, child in list(parent.named_children()):
            if not eligible(child):
                continue
            w = child.weight
            if id(w) in tied_ids or (not w.is_meta and w.data_ptr() in tied_ptrs):
                continue
            name = f"{parent_name}.{child_name}" if parent_name else child_name
            yield name, parent, child_name, child


def swap_linears(model, make_module, device: str = "cuda"):
    """Replace every eligible Linear via make_module(cpu_weight, bias)->(module, stat)."""
    stats = []
    for name, parent, child_name, child in eligible_linears(model):
        w = child.weight.data
        bias = child.bias.data.to(device) if child.bias is not None else None
        mod, stat = make_module(w, bias)
        setattr(parent, child_name, mod)
        child.weight.data = torch.empty(0)  # sever the old storage explicitly
        if stat:
            stat["name"] = name
            stats.append(stat)
    return stats


def make_module(pack: dict, bias, device: str = "cuda"):
    """The loader's one constructor: a pack dict (iter_pack_dir's) -> the
    resident module for it, by the tensor's own kind — RadixCompressedLinear
    over to_device_radix for a radix tensor, RawLinear over to_device_raw for
    the raw fallback. engines.load_compressed and mtp.install_head_pack both
    build through this, so the two module trees cannot disagree on which
    class serves which tensor. A dict that names a `dtype` instead of a
    codec (an FP8 pack's tensors) is refused by the one line."""
    if pack.get("codec") == "radix":  # the per-tensor codec scalar
        return RadixCompressedLinear(to_device_radix(pack, device), bias)
    if pack.get("codec") == "raw":    # the raw fallback
        return RawLinear(to_device_raw(pack, device), bias)
    if pack.get("dtype") is not None:
        from .pack import UnsupportedCheckpoint, refusal_for_dtype

        raise UnsupportedCheckpoint(f"pack tensor (dtype {pack.get('dtype')!r}): "
                                    f"{refusal_for_dtype(pack.get('dtype'))}")
    raise ValueError(
        f"pack tensor declares codec {pack.get('codec')!r}: not a radix or raw tensor "
        "this build serves — refusing to build a module for it")


def make_compressed(w, bias, device: str = "cuda", collect=None):
    """The bench's compressed arm, one Linear: the tensor packed EXACTLY as
    `drinkme pack` writes it (pack.pack_weight_served — the radix codec at
    the resolved profile, or the raw fallback) and installed through the
    loader's own constructor (make_module), so the module the arm times
    dispatches to the same kernels `drinkme serve` runs.

    pack_weight_served round-trips on the CPU and RAISES on any failure;
    reaching the return means the check passed for this tensor. The stat
    carries the tensor's codec/profile/dtype so arms.arm_compression_profile can name the
    record's compression.profile off the loaded arm, and its numel so the
    weight-weighted bits-per-weight can be formed."""
    pack = pack_weight_served(w.cpu())
    if collect is not None:
        collect(pack)
    return make_module(pack, bias, device), {
        "shape": list(w.shape),
        "numel": int(pack["R"]) * int(pack["C"]),
        # the encoded payload bits, exactly (bpw is bits/numel by construction)
        "bits": int(round(pack["bpw"] * int(pack["R"]) * int(pack["C"]))),
        "bpw": round(pack["bpw"], 3),
        "format_version": pack.get("format_version", FORMAT_VERSION),
        "codec": pack.get("codec"),          # "radix" | "raw"
        "profile": pack.get("profile"),      # the radix profile (None for a raw fallback)
        # the tier widths (None for a raw fallback): the twin arm's launch-table key (make_twin)
        "widths": [int(x) for x in pack["widths"]] if pack.get("widths") is not None else None,
        "dtype": pack.get("dtype"),  # None = the bf16 codec (the only thing this arm packs)
        "verified": True,
    }


def make_twin(w, bias, device: str = "cuda", *, codec: str, widths=None):
    """The bench's twin arm, one Linear, order-matched to the compressed
    arm's module for the same tensor: `codec` / `widths` are that tensor's
    (its make_compressed or pack_stat stat — arms.twin_plan). A radix
    tensor's twin is RadixTwinLinear over to_device_twin (the same kernels
    and launch schedule, the raw weight read in place of the decode); a raw
    fallback tensor's is the RawLinear the compressed arm serves it with
    (F.linear over the same bf16 bits)."""
    if codec == "radix":
        if widths is None:
            raise ValueError("make_twin: a radix tensor's twin needs its widths")
        return RadixTwinLinear(to_device_twin(w.cpu(), widths, device), bias), None
    if codec == "raw":
        from .radix_pack import raw_dict

        U = w.detach().cpu().to(torch.bfloat16).view(torch.int16).numpy().view(np.uint16)
        return make_module(raw_dict(U), bias, device), None
    raise ValueError(f"make_twin: no twin for codec {codec!r} (radix | raw)")
