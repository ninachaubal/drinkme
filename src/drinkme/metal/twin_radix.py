"""The RADIX decode in plain MLX ops, vectorised over a tensor's blocks: the
bench's MLX twin decodes each packed tensor ONCE with this at load
(serving/engine_mlx.RadixTwinLinear) instead of running the CPU oracle
on every forward.

THE CLAIM. decode(pack) is BITWISE the CPU oracle's decode
(radix_pack.decode_back_radix -> the native decoder or codec/radix.decode)
for every tensor the pack format admits: pinned on a toy in
tests/test_twin_radix.py (every bf16 pattern, ragged blocks, sip / balanced
/ gulp) and on every tensor of a real pack by bench/mlx_twin_bitpin.py.

WHY NOT dense_radix. dense_radix.py decodes with the fused kernel's own
chunk decoder (rx_decode_chunk). The twin is the correctness control the
fused path is measured against, so its decoder shares no code with that
kernel: a fault in the chunk decoder would otherwise show up in both arms
and in neither's difference. This one is written from the wire layout
(docs/pack-format.md; codec/radix.py `_put` / `_get`) in mlx's generic ops,
so it also runs where there is no Metal (the Linux dev box, mlx[cpu]).

THE LAYOUT, per block of n lanes at word `pos` (radix.pack_array):

  literals   n bytes: sign in bit 7, the 7 mantissa bits below (8-bit
             fields packed little-endian into words, so byte i of the
             block's words IS lane i's literal)
  tier 0     n codes of widths[0] bits; the largest code escapes
  tier k     one code per lane that escaped tier k-1, in lane order, at
             the lane's RANK among those lanes (an exclusive cumsum)
  terminal   one 8-bit exponent per lane that escaped every tier, in lane
             order — byte `rank` of the stream

every stream starting on a word. A lane's exponent is the palette entry
at (the tier's base + its code), or its terminal byte. Its bf16 pattern is
sign:15 | exponent:14..7 | mantissa:6..0.

EXACTNESS OF THE OPS. Everything is uint32 / int32: gathers, shifts,
masks, compares and a cumsum over 0/1 lanes. No float, so no backend
rounding can enter. A field that may straddle two words is read as
(lo >> sh) | ((hi << 1) << (31 - sh)): both shift counts stay in 0..31 (a
shift by 32 is undefined in C++ and in Metal, and the two backends would
be free to disagree on it).

STRICTNESS. Like the oracle, a block must consume its directory extent
exactly: the end of its terminal stream is its successor's offset. A block
that does not is refused by name ("radix block ... does not end where the
directory says"), never decoded to garbage.

MEMORY. Rows go through in chunks of about CHUNK_LANES lanes, each chunk's
graph evaluated before the next, so the transient beyond the output is one
chunk's planes (chunk_transient_bytes). The output is written in place into
one uint16 [R, C] array.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np

# lanes per evaluated chunk (whole rows; at least one row)
CHUNK_LANES = 1 << 21
# what one chunk's graph holds beyond the output, per lane: measured with mlx
# 0.32.2 (CPU) on Qwen3-0.6B's packs at 53 B (sip) and 83 B (gulp, two more
# tiers); charged with a margin, since a Metal graph may schedule differently
BYTES_PER_LANE = 128


def chunk_transient_bytes(R: int, C: int, block_size: int = 1024) -> int:
    """The decode's working set beyond its output, for the fit check: one
    chunk's lanes at BYTES_PER_LANE (256 MiB for a tensor of 2M weights or more)."""
    NB = -(-C // block_size)
    rows = _rows_per_chunk(NB, block_size, R)
    return BYTES_PER_LANE * rows * NB * block_size


def _rows_per_chunk(NB: int, B: int, R: int) -> int:
    return max(1, min(R, CHUNK_LANES // (NB * B)))


def _field(words: mx.array, start: mx.array, idx: mx.array, width: int) -> mx.array:
    """The `width`-bit little-endian field number `idx` of the stream at
    word `start` (radix._get's arithmetic): uint32, broadcast over
    start/idx. Indices past the chunk's words are clamped: those lanes
    never use the value."""
    bit = idx * width
    w = start + (bit >> 5)
    sh = bit & 31
    last = words.shape[0] - 1
    lo = mx.take(words, mx.minimum(w, last))
    hi = mx.take(words, mx.minimum(w + 1, last))
    v = mx.right_shift(lo, sh) | mx.left_shift(mx.left_shift(hi, 1), 31 - sh)
    return v & ((1 << width) - 1)


def _chunk(words: mx.array, byts: mx.array, offs: mx.array, palette: mx.array,
           n: mx.array, widths: tuple[int, ...], B: int):
    """One chunk of blocks, [nblk, B] lanes -> (bits uint32 [nblk, B],
    the blocks' computed end words [nblk], the directory's [nblk]).
    `offs` is the chunk's nblk + 1 directory entries, local to `words`."""
    pos = offs[:-1][:, None]
    lane = mx.arange(B, dtype=mx.uint32)[None, :]
    active = lane < n[:, None]
    lit = mx.take(byts, mx.minimum(pos * 4 + lane, byts.shape[0] - 1)).astype(mx.uint32)
    start = pos + ((n[:, None] * 8 + 31) >> 5)  # tier 0 follows the literal plane
    exp = mx.zeros(active.shape, dtype=mx.uint32)
    live = active                   # lanes still unresolved entering this tier
    rank = lane                     # tier 0: every lane has a code, at its own index
    count = n[:, None]              # codes in this tier's stream
    base = 0
    cover = palette.shape[0] - 1
    for w in widths[:-1]:
        code = _field(words, start, rank, w)
        esc = (1 << w) - 1
        idx = mx.minimum(base + code, cover)
        stays = live & (code != esc)
        exp = mx.where(stays, mx.take(palette, idx).astype(mx.uint32), exp)
        live = live & (code == esc)
        start = start + ((count * w + 31) >> 5)
        live32 = live.astype(mx.uint32)
        rank = mx.cumsum(live32, axis=1, inclusive=False)
        count = mx.sum(live32, axis=1, keepdims=True)
        base += esc
    term = mx.take(byts, mx.minimum(start * 4 + rank, byts.shape[0] - 1)).astype(mx.uint32)
    exp = mx.where(live, term, exp)
    end = (start + ((count * 8 + 31) >> 5))[:, 0]
    bits = (lit & 0x7F) | ((lit & 0x80) << 8) | (exp << 7)
    return bits, end, offs[1:]


def decode_arrays(palette: np.ndarray, offsets: np.ndarray, data: np.ndarray,
                  R: int, C: int, widths, block_size: int = 1024,
                  name: str = "radix tensor") -> mx.array:
    """The radix streams (numpy, as the pack stores them: palette uint8
    unpadded, directory uint32 [R*NB + 1], data uint32) -> the bf16 bit
    patterns uint16 [R, C] as an EVALUATED mx array (radix_native.decode's
    signature, on mlx)."""
    R, C, B = int(R), int(C), int(block_size)
    widths = tuple(int(w) for w in widths)
    if len(widths) < 2 or widths[-1] != 8 or any(w < 1 or w > 8 for w in widths[:-1]):
        raise ValueError(f"{name}: radix widths {widths}: bf16 needs 1..8-bit tiers ending in the "
                         "8-bit exponent")
    if B < 32 or B & (B - 1):
        raise ValueError(f"{name}: radix block_size {B} is not a power of two >= 32")
    NB = -(-C // B)
    offsets = np.ascontiguousarray(offsets, dtype=np.uint32).reshape(-1)
    data = np.ascontiguousarray(data, dtype=np.uint32).reshape(-1)
    palette = np.ascontiguousarray(palette, dtype=np.uint8).reshape(-1)
    if offsets.size != R * NB + 1 or int(offsets[0]) != 0 or int(offsets[-1]) != data.size:
        raise ValueError(f"{name}: radix directory does not match the streams ({offsets.size} "
                         f"entries for {R * NB} blocks, {data.size} words)")
    covered = sum((1 << w) - 1 for w in widths[:-1])
    if palette.size != covered:
        raise ValueError(f"{name}: radix palette has {palette.size} entries, widths {widths} "
                         f"need {covered}")
    pal = mx.array(palette)
    out = mx.zeros((R, C), dtype=mx.uint16)
    rows = _rows_per_chunk(NB, B, R)
    # the lanes of each block: B, but the row's last block holds C's remainder
    n_row = np.minimum(B, C - np.arange(NB, dtype=np.int64) * B).astype(np.uint32)
    for r0 in range(0, R, rows):
        r1 = min(R, r0 + rows)
        b0, b1 = r0 * NB, r1 * NB
        w0, w1 = int(offsets[b0]), int(offsets[b1])
        # the chunk's words plus one zero word, so the last straddling read has a neighbour
        words = mx.array(np.concatenate([data[w0:w1], np.zeros(1, dtype=np.uint32)]))
        byts = words.view(mx.uint8)
        offs = mx.array(offsets[b0:b1 + 1] - np.uint32(w0))
        n = mx.array(np.tile(n_row, r1 - r0))
        bits, end, want = _chunk(words, byts, offs, pal, n, widths, B)
        bits = bits.reshape(r1 - r0, NB * B)[:, :C].astype(mx.uint16)
        bad = mx.any(end != want)
        out[r0:r1] = bits
        mx.eval(out, bad)
        if bad.item():
            e, wn = np.array(end), np.array(want)
            i = int(np.flatnonzero(e != wn)[0])
            raise ValueError(f"{name}: radix block {b0 + i} does not end where the directory "
                             f"says (word {int(e[i]) + w0}, not {int(wn[i]) + w0})")
    return out


def decode(pack: dict, name: str = "radix tensor") -> mx.array:
    """A radix tensor dict (iter_pack_dir's, or gemv_radix.resident's) ->
    uint16 [R, C] bf16 bit patterns, evaluated: decode_arrays over its
    streams."""
    from ..codec import radix_pack as _rp

    if "data" in pack and "params" in pack:  # a resident dict: the streams back off the device
        return decode_arrays(pack["rx_palette_np"], np.array(pack["offsets"])[: pack["R"] * pack["NB"] + 1],
                             np.array(pack["data"])[: pack["NW"]], pack["R"], pack["C"],
                             pack["widths"], pack["B"], name)
    if not _rp.is_radix_pack(pack):
        raise ValueError(f"{name}: not a radix pack tensor (codec {pack.get('codec')!r})")
    return decode_arrays(_rp._palette_u8(pack), pack["rx_offsets"], _rp._payload(pack),
                         pack["R"], pack["C"], pack["widths"], pack["block_size"], name)
