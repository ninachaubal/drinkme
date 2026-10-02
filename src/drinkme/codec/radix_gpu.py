"""The radix codec's block decoder on Triton: the bit-field reader, the
palette lookup, the reference (unscheduled) block decode, and the
bits -> bf16 weight step. Jit functions only; radix_kernel_gpu.py builds
the scheduled decoder and the launched kernels over these, and
radix_ops.py launches them for the served arms."""
from __future__ import annotations
import triton
import triton.language as tl
from .decode import _e4m3_to_f32


@triton.jit
def _read(data, pos, index, active, W: tl.constexpr):
    """Field `index` of the W-bit little-endian stream at word `pos`, per
    lane, masked by `active` alone. No load is masked on the block's end:
    the reader's reach is bounded instead. Every read of a block lies
    within radix.block_word_bounds(n)[1] words of its start — the literal
    plane, then each tier at its widest (every lane escaping), then a full
    terminal, each stream whole-word aligned; the directory is bounded by
    radix.validate (offsets monotone, inside the payload) — and
    swap.to_device_radix allocates that many zero words after the payload.
    A block short of its data-dependent streams therefore reads its
    neighbour's words or the pad — garbage — and never a word outside the
    allocation. A per-lane mask here (`word < end`, one compare per load)
    was measured at +2% on the sip GEMV on gfx1102 even on the terminal read
    alone (RX 7600 XT 16 GB, runs 1-3); the bound costs
    nothing: the kernel's loads stay unmasked."""
    bit = index * W
    word = tl.load(data + pos + bit // 32, mask=active, other=0)
    code = word >> (bit % 32)
    if 32 % W != 0:
        extra = tl.load(data + pos + bit // 32 + 1,
                        mask=active & (bit % 32 + W > 32), other=0)
        code |= extra << (32 - bit % 32)
    return code & ((1 << W) - 1)


@triton.jit
def _lookup(palette, table, code, W: tl.constexpr, B: tl.constexpr):
    value = tl.zeros((B,), tl.uint32)
    for word in tl.static_range(((1 << W) + 2) // 4):
        packed = tl.load(palette + table + word)
        value |= tl.where(code // 4 == word, packed >> (code % 4 * 8), 0)
    return value & 255


@triton.jit
def _decode_block(data, offsets, palette, block, n,
                  M: tl.constexpr, E: tl.constexpr, WIDTHS: tl.constexpr, B: tl.constexpr):
    lane = tl.arange(0, B)
    active = lane < n
    pos = tl.load(offsets + block).to(tl.int64)
    literal = _read(data, pos, lane, active, M + 1)
    pos += tl.cdiv(n * (M + 1), 32)
    rank = lane
    count = n
    exp = tl.zeros((B,), tl.uint32)
    base = 0
    table = 0
    for level in tl.static_range(len(WIDTHS) - 1):
        width = WIDTHS[level]
        escape = (1 << width) - 1
        code = _read(data, pos, rank, active, width)
        pos += tl.cdiv(count * width, 32)
        done = active & (code != escape)
        value = _lookup(palette, table, code, WIDTHS[level], B)
        exp |= tl.where(done, value, 0)
        table += ((1 << WIDTHS[level]) + 2) // 4
        active &= code == escape
        if level < len(WIDTHS) - 2:
            rank = tl.cumsum(active.to(tl.int32)) - 1
        count = tl.sum(active.to(tl.int32))
        base += escape
    if count > 0:
        rank = tl.cumsum(active.to(tl.int32)) - 1
        final = _read(data, pos, rank, active, WIDTHS[-1])
        exp |= final
    return (literal & ((1 << M) - 1)) | ((literal >> M) << (M + E)) | (exp << M)


@triton.jit
def _weight(bits, scale, row, col, C: tl.constexpr, E: tl.constexpr):
    if E == 8:
        return bits.to(tl.uint16).to(tl.bfloat16, bitcast=True).to(tl.float32)
    else:
        sf = tl.load(scale + row // 128 * triton.cdiv(C, 128) + col // 128,
                     mask=col < C, other=0)
        value = tl.where((bits & 127) == 127, float('nan'), _e4m3_to_f32(bits))
        return (value * sf).to(tl.bfloat16).to(tl.float32)

