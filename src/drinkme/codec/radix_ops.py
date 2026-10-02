"""The radix serve arms over a to_device_radix runtime dict: the M=1 GEMV,
the multi-column kernel, and the dense decode. Imports triton, so swap.py
binds it lazily (like codec/ops.py).

THE THREE ARMS

  gemv (M=1)   radix_kernel_gpu._gemv (+ _finish when split-K) — the
               scheduled decoder with the fused epilogue (bias added in
               fp32, ONE rounding to bf16), at the LAUNCH SCHEDULE the
               runtime dict carries:
               p["rx_launch"] = radix_schedule.select(R, C, B, widths).as_dict(),
               chosen per (profile family, tensor shape class) at load by
               swap.to_device_radix.
               gemv_tiles = t > 0: t consecutive 1024-weight blocks of one
               row per program, split-K partials, the per-row reduction is
               _finish's second launch (the `spike` schedule of
               radix_schedule.py is t=1, 2 warps for every tensor);
               gemv_tiles = 0: the whole row in ONE program (the blocks
               accumulate in registers, no partial buffer, no second
               launch, the bias fused in-kernel). Absent rx_launch = the
               `spike` schedule.
  mc (2..8)    _gemv_mc below — what lets speculative decoding verify M
               drafts off ONE read of the weights. The SAME program shape as the M=1 kernel (one program per
               (row, block), split-K partials, the same _finish), the
               block decoded ONCE, then applied to all M activation rows
               through eight named accumulators (a [MC, B] tile would
               reduce in a different order than the M=1 kernel's 1-D
               tl.sum, and eight 1-D accumulators each reduced by their
               own tl.sum is what keeps column m's arithmetic the M=1
               kernel's). Whether that makes each column BITWISE the M=1
               result is measured, not assumed — bench/radix_mc_bitpin.py
               records it and gates on the float64-oracle bound.
  dense        radix_kernel_gpu._dense with the scheduled decoder -> the
               transient bf16 weight -> F.linear (swap.RadixCompressedLinear's
               prefill, M >= GEMM_MIN_ROWS).

The activation read: bf16 (or fp32) rows loaded and widened in the kernel,
no host-side `.float()` conversion launch.

THE TWIN. A swap.to_device_twin dict (the bench's order-matched twin:
`codec` "twin", the raw bf16 weight, the radix tensor's widths, and its own
rx_launch from radix_schedule.select_twin) runs gemv and mc with RAW=True:
_raw_weight's load in place of _decode_weight, everything else the same
source, so at any one launch schedule it is the compressed kernel reading
raw bf16 (bench/radix_twin_bitpin.py checks the outputs bitwise at every
table, spike and twin row). Its dense arm is F.linear over the raw weight
(swap.RadixTwinLinear), not decode_weight.

THE NARROW GEMV. gemv_narrow over _gemv_narrow: swap.NarrowLinear's
kernel for the raw bf16 Linears under the codec's row threshold, M = 1..8
in one launch. It is _gemv_mc's program with the raw load, one program
per output row, C a runtime argument and _gemv's fused epilogue; the
kernel's docstring says where it departs from the twin and why.
"""

from __future__ import annotations

import functools
import os

import torch
import triton
import triton.language as tl

from .ops import MC_MAX
from .radix_kernel_gpu import _decode_weight, _dense, _finish, _gemv, _prepare_schedule, _raw_weight, _widen

# scheduled decoder, two warps, one block per program, fp fusion on.
# NUM_WARPS / TILES are the `spike` schedule (radix_schedule.SPIKE) — what a
# runtime dict without rx_launch runs at, and the dense arm's launch; the
# gemv / mc arms take theirs from rx_launch.
DECODER_SCHEDULED = 2
# radix_kernel_gpu._decode_sip_lean: the same bits as the scheduled decoder
# for sip's (3, 8) streams, fewer instructions per weight; the default on a CUDA
# build (_decoder)
DECODER_SIP_LEAN = 3
# radix_kernel_gpu._decode_gulp_lean: the same bits as the scheduled decoder
# for gulp's (2, 2, 4, 8) streams, fewer instructions per weight; the default
# on a ROCm build (_decoder)
DECODER_GULP_LEAN = 4
# radix_kernel_gpu._decode_gulp_lean_cuda: _decode_gulp_lean's bits in fewer
# SASS instructions per weight; the default for gulp on a CUDA build (_decoder)
DECODER_GULP_LEAN_CUDA = 5
DECODER_ENV = "DRINKME_RADIX_DECODER"
NUM_WARPS = 2
TILES = 1
FUSION = True
MANT, EXP = 7, 8  # bf16: 7 mantissa bits (+ sign = the 8-bit literal), 8 exponent bits


@triton.jit
def _gemv_mc(out, x, data, offsets, palette, schedule, scale, C: tl.constexpr,
             M: tl.constexpr, E: tl.constexpr, WIDTHS: tl.constexpr, B: tl.constexpr,
             DECODER: tl.constexpr, TILES: tl.constexpr, SPLITS: tl.constexpr,
             MC: tl.constexpr, RAW: tl.constexpr = False):
    """radix_kernel_gpu._gemv holding MC (<= 8) activation rows against ONE
    block decode. Program = (row, split) exactly as _gemv's; `x` is [MC, C]
    (bf16 or fp32, row stride C); `out` is the split-K partial buffer laid
    out [(m * R + row) * SPLITS + split] — _gemv's own layout with the
    batch index folded in, so _finish reduces it unchanged with TOTAL =
    MC * R."""
    tl.static_assert(MC <= 8, "_gemv_mc holds at most 8 columns (ops.MC_MAX)")
    program = tl.program_id(0)
    row = program // SPLITS
    start = program % SPLITS * TILES
    lane = tl.arange(0, B)
    a0 = tl.zeros((B,), tl.float32)
    a1 = tl.zeros((B,), tl.float32)
    a2 = tl.zeros((B,), tl.float32)
    a3 = tl.zeros((B,), tl.float32)
    a4 = tl.zeros((B,), tl.float32)
    a5 = tl.zeros((B,), tl.float32)
    a6 = tl.zeros((B,), tl.float32)
    a7 = tl.zeros((B,), tl.float32)
    NB: tl.constexpr = triton.cdiv(C, B)
    for tile in range(start, tl.minimum(start + TILES, NB)):
        col = tile * B + lane
        inb = col < C
        if RAW:
            weight = _raw_weight(data, row, col, x, C)
        else:
            weight = _decode_weight(data, offsets, palette, schedule, scale, row * NB + tile,
                                    tl.minimum(C - tile * B, B), row, col, C, M, E, WIDTHS, B, DECODER)
        # each column: _gemv's own two lines, verbatim, over the one `weight`
        value = _widen(tl.load(x + col, mask=inb, other=0), DECODER)
        a0 += tl.where(inb, value * weight, 0)
        if MC > 1:
            value = _widen(tl.load(x + C + col, mask=inb, other=0), DECODER)
            a1 += tl.where(inb, value * weight, 0)
        if MC > 2:
            value = _widen(tl.load(x + 2 * C + col, mask=inb, other=0), DECODER)
            a2 += tl.where(inb, value * weight, 0)
        if MC > 3:
            value = _widen(tl.load(x + 3 * C + col, mask=inb, other=0), DECODER)
            a3 += tl.where(inb, value * weight, 0)
        if MC > 4:
            value = _widen(tl.load(x + 4 * C + col, mask=inb, other=0), DECODER)
            a4 += tl.where(inb, value * weight, 0)
        if MC > 5:
            value = _widen(tl.load(x + 5 * C + col, mask=inb, other=0), DECODER)
            a5 += tl.where(inb, value * weight, 0)
        if MC > 6:
            value = _widen(tl.load(x + 6 * C + col, mask=inb, other=0), DECODER)
            a6 += tl.where(inb, value * weight, 0)
        if MC > 7:
            value = _widen(tl.load(x + 7 * C + col, mask=inb, other=0), DECODER)
            a7 += tl.where(inb, value * weight, 0)
    stride = tl.num_programs(0)  # R * SPLITS: one batch's worth of partials
    base = out + program
    tl.store(base, tl.sum(a0))
    if MC > 1:
        tl.store(base + stride, tl.sum(a1))
    if MC > 2:
        tl.store(base + 2 * stride, tl.sum(a2))
    if MC > 3:
        tl.store(base + 3 * stride, tl.sum(a3))
    if MC > 4:
        tl.store(base + 4 * stride, tl.sum(a4))
    if MC > 5:
        tl.store(base + 5 * stride, tl.sum(a5))
    if MC > 6:
        tl.store(base + 6 * stride, tl.sum(a6))
    if MC > 7:
        tl.store(base + 7 * stride, tl.sum(a7))


@triton.jit
def _narrow_tail(acc, bias, row, BIAS: tl.constexpr):
    """_gemv's epilogue for one column: the fp32 sum, plus the row's bias in
    fp32 when there is one; the caller's store is the one rounding."""
    value = tl.sum(acc)
    if BIAS:
        value += tl.load(bias + row).to(tl.float32)
    return value


@triton.jit
def _gemv_narrow(out, x, w, bias, C, MC: tl.constexpr, B: tl.constexpr, BIAS: tl.constexpr):
    """swap.NarrowLinear's kernel: _gemv_mc's program over a raw row-major
    bf16 weight `w` [R, C], with no decoder and no split-K. One program per
    output row walks the whole row in B-wide chunks, loads each chunk of
    the weight row once and applies it to all MC (<= 8) rows of `x` [MC, C]
    (bf16 or fp32, row stride C) through _gemv_mc's eight named
    accumulators. The epilogue is _gemv's fused one: each sum plus the bias
    in fp32, one rounding at the store into `out` [MC, R]. Column m of an
    MC-row call is the MC = 1 call's result bit for bit, and a bf16 x gives
    the bits of its fp32 widening (tests/test_narrow_linear.py pins both).

    C is a runtime argument; the twin's _gemv_mc takes it as a constexpr.
    At the same B and warps the twin's program (bench/narrow_linear_knobs.py's
    twin-mc route) was 1.3-1.4x this kernel's device time at MC = 6..8 on
    the 48- and 32-row narrow shapes on gfx1151, though faster at MC <= 2."""
    tl.static_assert(MC >= 1 and MC <= 8, "_gemv_narrow holds 1..8 columns (ops.MC_MAX)")
    row = tl.program_id(0)
    lane = tl.arange(0, B)
    a0 = tl.zeros((B,), tl.float32)
    a1 = tl.zeros((B,), tl.float32)
    a2 = tl.zeros((B,), tl.float32)
    a3 = tl.zeros((B,), tl.float32)
    a4 = tl.zeros((B,), tl.float32)
    a5 = tl.zeros((B,), tl.float32)
    a6 = tl.zeros((B,), tl.float32)
    a7 = tl.zeros((B,), tl.float32)
    data = w + row.to(tl.int64) * C
    for tile in range(tl.cdiv(C, B)):
        col = tile * B + lane
        inb = col < C
        weight = tl.load(data + col, mask=inb, other=0).to(tl.float32)
        # each column: _gemv_mc's two lines
        value = tl.load(x + col, mask=inb, other=0).to(tl.float32)
        a0 += tl.where(inb, value * weight, 0)
        if MC > 1:
            value = tl.load(x + C + col, mask=inb, other=0).to(tl.float32)
            a1 += tl.where(inb, value * weight, 0)
        if MC > 2:
            value = tl.load(x + 2 * C + col, mask=inb, other=0).to(tl.float32)
            a2 += tl.where(inb, value * weight, 0)
        if MC > 3:
            value = tl.load(x + 3 * C + col, mask=inb, other=0).to(tl.float32)
            a3 += tl.where(inb, value * weight, 0)
        if MC > 4:
            value = tl.load(x + 4 * C + col, mask=inb, other=0).to(tl.float32)
            a4 += tl.where(inb, value * weight, 0)
        if MC > 5:
            value = tl.load(x + 5 * C + col, mask=inb, other=0).to(tl.float32)
            a5 += tl.where(inb, value * weight, 0)
        if MC > 6:
            value = tl.load(x + 6 * C + col, mask=inb, other=0).to(tl.float32)
            a6 += tl.where(inb, value * weight, 0)
        if MC > 7:
            value = tl.load(x + 7 * C + col, mask=inb, other=0).to(tl.float32)
            a7 += tl.where(inb, value * weight, 0)
    stride = tl.num_programs(0)  # R: one column's outputs
    base = out + row
    tl.store(base, _narrow_tail(a0, bias, row, BIAS))
    if MC > 1:
        tl.store(base + stride, _narrow_tail(a1, bias, row, BIAS))
    if MC > 2:
        tl.store(base + 2 * stride, _narrow_tail(a2, bias, row, BIAS))
    if MC > 3:
        tl.store(base + 3 * stride, _narrow_tail(a3, bias, row, BIAS))
    if MC > 4:
        tl.store(base + 4 * stride, _narrow_tail(a4, bias, row, BIAS))
    if MC > 5:
        tl.store(base + 5 * stride, _narrow_tail(a5, bias, row, BIAS))
    if MC > 6:
        tl.store(base + 6 * stride, _narrow_tail(a6, bias, row, BIAS))
    if MC > 7:
        tl.store(base + 7 * stride, _narrow_tail(a7, bias, row, BIAS))


def _raw(p: dict) -> bool:
    """A swap.to_device_twin dict: the gemv / mc kernels run RAW=True over
    its bf16 weight (the bench's order-matched twin)."""
    return p.get("codec") == "twin"


@functools.lru_cache(maxsize=16)
def _decoder(widths: tuple, whole_blocks: bool = True) -> int:
    """The decoder a tensor of these tier widths runs: a lean decoder where
    one reads the widths and was measured on the build's hardware family,
    else the scheduled one. The decoded bits are the same either way.

      sip   the lean sip decoder (radix_kernel_gpu._decode_sip_lean) on a
            CUDA build for sip's two-tier widths (3, 8), the only ones it
            reads, with rows of whole blocks (C % block_size == 0). On a
            ragged row its kernel contracts a different set of products
            into FMAs, so its GEMV stopped being bitwise the twin's
            (bench/radix_twin_bitpin.py's ragged fixtures on an L4), and
            those rows keep the scheduled decoder the twin is order-matched
            to. It was measured on NVIDIA only, so a ROCm build keeps the
            scheduled decoder for sip.
      gulp  a lean gulp decoder for gulp's widths (2, 2, 4, 8), the only
            ones it reads, with rows of whole blocks: on a ROCm build
            radix_kernel_gpu._decode_gulp_lean (measured on gfx1151), on a
            CUDA build _decode_gulp_lean_cuda, the same bits in fewer SASS
            instructions per weight (measured on an L4, an L40S and an
            H100 at the CUDA table's scheduled row). Their decoded bits are the
            scheduled decoder's on ragged rows too (bench/radix_lean_interp.py),
            but there the GEMV is not bitwise the twin's: on gfx1151 the
            ragged fixtures of bench/radix_twin_bitpin.py (512x513, 7x33,
            3x1025, 17x3079) differed from the twin in the last bits of the
            fp32 GEMV at every table row, fp fusion off or on, as the lean
            sip decoder's did on an L4.

    DRINKME_RADIX_DECODER=scheduled or =lean forces one for A/B runs (lean
    only where it applies). Read once per (widths, whole_blocks)."""
    from .radix_schedule import _backend_family

    mode = os.environ.get(DECODER_ENV, "").strip()
    if mode not in ("", "scheduled", "lean"):
        raise ValueError(f"{DECODER_ENV}={mode!r}: scheduled or lean")
    widths = tuple(int(w) for w in widths)
    sip = len(widths) == 2 and widths[0] == 3 and widths[1] == EXP and whole_blocks
    gulp = widths == (2, 2, 4, EXP) and whole_blocks
    if mode == "scheduled" or not (sip or gulp):
        return DECODER_SCHEDULED
    if sip:
        return DECODER_SIP_LEAN if mode == "lean" or _backend_family() == "cuda" else DECODER_SCHEDULED
    return DECODER_GULP_LEAN_CUDA if _backend_family() == "cuda" else DECODER_GULP_LEAN


def _args(p: dict):
    """The argument tail every radix_kernel_gpu kernel takes after its own
    outputs/inputs: (data, offsets, palette, schedule, scale, C, M, E,
    WIDTHS, B, DECODER) — the runtime dict's three streams and the gulp
    schedule buffer (empty for sip), a `scale` pointer (a dummy: bf16's
    E == 8 never reads it, so rx_data stands in), the row width, bf16's 7
    mantissa / 8 exponent bits, the tier widths, the block size, and the
    scheduled decoder's index. For a twin dict (_raw) `data` is the bf16
    weight and the other four pointers stand in for streams RAW=True never
    reads; the scalars are the radix tensor's."""
    if _raw(p):
        w = p["weight"]
        return (w, w, w, w, w, int(p["C"]), MANT, EXP, tuple(int(x) for x in p["widths"]),
                int(p["block_size"]), DECODER_SCHEDULED)
    widths = tuple(int(w) for w in p["widths"])
    C, B = int(p["C"]), int(p["block_size"])
    return (p["rx_data"], p["rx_offsets"], p["rx_palette"], p["rx_schedule"], p["rx_data"],
            C, MANT, EXP, widths, B, _decoder(widths, C % B == 0))


def launch_of(p: dict) -> dict:
    """The tensor's launch schedule (radix_schedule.Launch.as_dict()); a
    dict without one runs at the `spike` schedule."""
    return p.get("rx_launch") or dict(gemv_tiles=TILES, gemv_warps=NUM_WARPS,
                                       mc_tiles=TILES, mc_warps=NUM_WARPS)


def _splits(p: dict, arm: str = "gemv") -> tuple[int, int, int, int]:
    """(nb, tiles, splits, num_warps) for the arm: tiles = blocks per
    program (the whole row when the schedule says 0), splits = programs per
    row (1 = no split-K: the kernel writes the output itself)."""
    nb = triton.cdiv(int(p["C"]), int(p["block_size"]))
    cfg = launch_of(p)
    tiles = int(cfg[arm + "_tiles"]) or nb
    tiles = min(tiles, nb)
    return nb, tiles, triton.cdiv(nb, tiles), int(cfg[arm + "_warps"])


def prepare_schedule(data, offsets, C: int, widths, B: int, nblocks: int):
    """The gulp profile's per-block stream lengths (2 bytes/block), on
    the device — radix_kernel_gpu._prepare_schedule; radix_pack.schedule_np
    is its CPU twin (pinned equal in bench/radix_gemv_bitpin.py)."""
    sched = torch.empty(nblocks, dtype=torch.uint16, device=data.device)
    _prepare_schedule[(nblocks,)](data, offsets, sched, C, MANT, tuple(int(w) for w in widths),
                                  B, num_warps=2)
    return sched


def gemv_fused(p: dict, x: torch.Tensor, bias, out_dtype=torch.bfloat16) -> torch.Tensor:
    """The M=1 arm: x [C] (bf16 or fp32) -> [1, R] in `out_dtype`, bias (if
    any) added in fp32 before the single rounding: one _gemv launch of
    R * splits programs (the bias fused in-kernel when splits == 1), else
    the split-K partials [R, splits] reduced and biased by _finish — at
    the tensor's launch schedule (_splits)."""
    R = int(p["R"])
    x = x.reshape(1, -1).contiguous()
    nb, tiles, splits, warps = _splits(p, "gemv")
    out = torch.empty((1, R), dtype=out_dtype, device=x.device)
    partial = out if splits == 1 else torch.empty((R, splits), dtype=torch.float32, device=x.device)
    has_bias = bias is not None
    args = _args(p)
    b = bias.contiguous() if has_bias else args[0]
    _gemv[(R * splits, 1)](partial, x, b, *args, tiles, splits, has_bias, RAW=_raw(p),
                           num_warps=warps, enable_fp_fusion=FUSION)
    if splits != 1:
        _finish[(triton.cdiv(R, 128),)](out, partial, b, R, R, splits,
                                        triton.next_power_of_2(splits), has_bias, num_warps=4)
    return out


def gemv(p: dict, x: torch.Tensor) -> torch.Tensor:
    """The M=1 arm's fp32 [R] result, no bias — the unfused half, for the
    M > MC_MAX row loop and the gates (CompressedLinear._gemv_row's contract)."""
    return gemv_fused(p, x, None, torch.float32).reshape(-1)


def gemv_mc(p: dict, x: torch.Tensor) -> torch.Tensor:
    """The multi-column arm: x [M, C] (bf16 or fp32), 1 <= M <= MC_MAX ->
    fp32 [M, R] (CompressedLinear._gemv_cols's contract; the epilogue adds
    the bias and rounds once)."""
    M = x.shape[0]
    if not 1 <= M <= MC_MAX:
        raise ValueError(f"radix gemv_mc: M={M} outside 1..{MC_MAX}")
    R = int(p["R"])
    x = x.contiguous()
    nb, tiles, splits, warps = _splits(p, "mc")
    out = torch.empty((M, R), dtype=torch.float32, device=x.device)
    partial = out if splits == 1 else torch.empty((M * R, splits), dtype=torch.float32,
                                                  device=x.device)
    args = _args(p)
    _gemv_mc[(R * splits,)](partial, x, *args, tiles, splits, M, RAW=_raw(p),
                            num_warps=warps, enable_fp_fusion=FUSION)
    if splits != 1:
        _finish[(triton.cdiv(M * R, 128),)](out, partial, args[0], M * R, R, splits,
                                            triton.next_power_of_2(splits), False, num_warps=4)
    return out


def gemv_narrow(w: torch.Tensor, x: torch.Tensor, bias, block: int, num_warps: int) -> torch.Tensor:
    """swap.NarrowLinear's forward for 1 <= M <= MC_MAX rows: x [M, C] (bf16
    or fp32) against the row-major bf16 weight w [R, C] -> [M, R] in x's
    dtype, the bias (if any) added in fp32 before the one rounding. ONE
    launch of R programs of _gemv_narrow, `block`-wide chunks at
    `num_warps`."""
    M, C = x.shape
    if not 1 <= M <= MC_MAX:
        raise ValueError(f"gemv_narrow: M={M} outside 1..{MC_MAX}")
    R = w.shape[0]
    x = x.contiguous()
    out = torch.empty((M, R), dtype=x.dtype, device=x.device)
    has_bias = bias is not None
    _gemv_narrow[(R,)](out, x, w, bias if has_bias else w, C, M, block, has_bias, num_warps=num_warps)
    return out


def decode_bits(p: dict, num_warps: int = NUM_WARPS) -> torch.Tensor:
    """The dense arm's decode: every block once, the scheduled decoder, to
    the original uint16 bit patterns [R, C] — one _dense program per
    block, DEQUANT off (the bits are stored, not widened).
    `num_warps` is the gates' knob: the decoder is pinned bit-exact at every
    warp count the gemv schedule can run at (bench/radix_schedule_bitpin.py)."""
    R, C, B = int(p["R"]), int(p["C"]), int(p["block_size"])
    out = torch.empty((R, C), dtype=torch.uint16, device=p["rx_data"].device)
    _dense[(R * triton.cdiv(C, B),)](out, *_args(p), False, num_warps=num_warps)
    return out


def decode_weight(p: dict) -> torch.Tensor:
    """The transient bf16 weight [R, C] for F.linear."""
    return decode_bits(p).view(torch.bfloat16)


def read_bytes(p: dict) -> int:
    """Bytes one GEMV/MC call streams from the device: the payload (NW
    words — not the zero pad after it, which no well-formed block reads),
    the directory, the lookup tables and (gulp) the schedule."""
    words = int(p["NW"]) if "NW" in p else p["rx_data"].numel()
    return words * 4 + sum(p[k].numel() * p[k].element_size()
                           for k in ("rx_offsets", "rx_palette", "rx_schedule"))
