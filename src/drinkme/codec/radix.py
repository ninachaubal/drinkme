"""Lossless exponent coding with independently decodable blocks — the BF16 codec.

Sign and mantissa are verbatim. A frequency palette assigns short radix
codes to common exponents; each tier's last code escapes to the next, narrower
stream. docs/pack-format.md has the wire layout.
This module is NumPy-only; accelerator operations live in radix_gpu.py.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

DEFAULT_BLOCK = 1024
PROFILES = {
    'gulp': {'bf16': (2, 2, 4, 8), 'fp8_e4m3': (2, 2, 4)},
    'balanced': {'bf16': (2, 3, 8), 'fp8_e4m3': (2, 2, 4)},
    'sip': {'bf16': (3, 8), 'fp8_e4m3': (3, 4)},
}


@dataclass(frozen=True)
class RadixPack:
    shape: tuple[int, int]
    dtype: str
    block_size: int
    widths: tuple[int, ...]
    palette: np.ndarray
    offsets: np.ndarray
    data: np.ndarray
    raw: bool = False

    @property
    def nbytes(self) -> int:
        """Resident array bytes, including the palette and block directory."""
        return self.palette.nbytes + self.offsets.nbytes + self.data.nbytes

    @property
    def bpw(self) -> float:
        return 8 * self.nbytes / (self.shape[0] * self.shape[1])


def _fields(dtype):
    if dtype == 'bf16':
        return 7, 8, np.dtype('<u2')
    if dtype == 'fp8_e4m3':
        return 3, 4, np.dtype('u1')
    raise ValueError(f'unsupported radix dtype: {dtype!r}')


def _put(values, width):
    """Pack little-endian fields into uint32 words (zero-padded last word)."""
    values = np.asarray(values, dtype=np.uint32)
    if not len(values):
        return np.empty(0, dtype='<u4')
    bit = np.arange(len(values), dtype=np.int64) * width
    out = np.zeros((len(values) * width + 31) // 32, dtype='<u4')
    np.bitwise_or.at(out, bit // 32, values << (bit % 32).astype(np.uint32))
    cross = bit % 32 + width > 32
    if np.any(cross):
        np.bitwise_or.at(out, bit[cross] // 32 + 1,
                         values[cross] >> (32 - bit[cross] % 32).astype(np.uint32))
    return out


def _get(data, start, count, width):
    bit = np.arange(count, dtype=np.int64) * width
    words = (count * width + 31) // 32
    if start < 0 or start + words > len(data):
        raise ValueError('truncated radix stream')
    value = data[start + bit // 32] >> (bit % 32).astype(np.uint32)
    cross = bit % 32 + width > 32
    if np.any(cross):
        value[cross] |= data[start + bit[cross] // 32 + 1] << (32 - bit[cross] % 32).astype(np.uint32)
    return value & ((1 << width) - 1), start + words


def pack_array(bits: np.ndarray, *, dtype: str | None = None,
               block_size: int = DEFAULT_BLOCK,
               widths: tuple[int, ...] | None = None,
               compression_profile: str = 'gulp') -> RadixPack:
    """Encode uint16 BF16 or uint8 E4M3 *bit patterns*, without float conversion.

    Every bit pattern, including signed zeros, subnormals and NaN payloads,
    round-trips. A tensor that would expand is stored verbatim instead.
    """
    bits = np.asarray(bits)
    if dtype is None:
        dtype = 'bf16' if bits.dtype == np.uint16 else 'fp8_e4m3' if bits.dtype == np.uint8 else ''
    mant, exponent, storage = _fields(dtype)
    if bits.dtype != storage or bits.ndim != 2 or min(bits.shape) <= 0:
        raise ValueError(f'expected a nonempty 2D {storage} array of {dtype} bit patterns')
    if type(block_size) is not int or block_size < 32 or block_size > 4096 or block_size & (block_size - 1):
        raise ValueError('block_size must be a power of two from 32 through 4096')
    if compression_profile not in PROFILES:
        raise ValueError(f'unknown profile: {compression_profile!r}')
    exp = ((bits >> mant) & ((1 << exponent) - 1)).astype(np.uint8)
    hist = np.bincount(exp.ravel(), minlength=1 << exponent)
    order = np.argsort(-hist, kind='stable')
    widths = tuple(PROFILES[compression_profile][dtype] if widths is None else widths)
    covered = _check_widths(widths, exponent)
    palette = order.astype(np.uint8)[:covered]
    inverse = np.full(1 << exponent, covered, dtype=np.uint16)
    inverse[palette] = np.arange(covered, dtype=np.uint16)
    rank = inverse[exp]
    literals = ((bits & ((1 << mant) - 1)) | ((bits >> (mant + exponent)) << mant)).astype(np.uint8)
    chunks, offsets = [], [0]
    for r in range(bits.shape[0]):
        for c in range(0, bits.shape[1], block_size):
            stop = min(c + block_size, bits.shape[1])
            chunks.append(_put(literals[r, c:stop], mant + 1))
            rr, ee = rank[r, c:stop], exp[r, c:stop]
            base = 0
            for width in widths[:-1]:
                escape = (1 << width) - 1
                code = np.minimum(rr - base, escape)
                chunks.append(_put(code, width))
                mask = code == escape
                rr, ee = rr[mask], ee[mask]
                base += escape
            chunks.append(_put(ee, widths[-1]))
            offsets.append(sum(x.size for x in chunks[-len(widths)-1:]) + offsets[-1])
    if offsets[-1] > np.iinfo(np.uint32).max:
        raise ValueError('radix payload exceeds the uint32 word directory')
    packed = RadixPack(tuple(bits.shape), dtype, block_size, widths, palette,
                       np.array(offsets, dtype='<u4'), np.concatenate(chunks))
    # GPU lookup tables pad each tier to whole uint32 words. Count that
    # small difference too, so a borderline tensor cannot expand at upload.
    gpu_bytes = packed.data.nbytes + packed.offsets.nbytes + sum(
        4 * (((1 << w) + 2) // 4) for w in widths[:-1])
    if max(packed.nbytes, gpu_bytes) >= bits.nbytes:
        packed = RadixPack(tuple(bits.shape), dtype, block_size, widths,
                           np.empty(0, dtype=np.uint8), np.empty(0, dtype='<u4'),
                           bits.copy(order='C'), raw=True)
    if not np.array_equal(decode(packed), bits):
        raise ValueError('radix round-trip verification failed')
    return packed


def _check_widths(widths, exponent):
    if (len(widths) < 2 or len(widths) > 8 or widths[-1] != exponent
            or any(type(w) is not int or w < 1 or w > 8 for w in widths)):
        raise ValueError('radix needs 2..8 widths in 1..8, ending with the exponent width')
    covered = sum((1 << w) - 1 for w in widths[:-1])
    if covered > 1 << exponent:
        raise ValueError('radix tiers exceed the exponent alphabet')
    return covered


def block_word_bounds(n, mant: int, exponent: int, widths) -> tuple:
    """The fewest and the most uint32 words a block of `n` weights can occupy
    (`n` a scalar or an array of block lengths), from the codec parameters
    alone. Every stream is padded to whole words:

        literal plane   ceil(n * (mant + 1) / 32)   FIXED: one literal per weight
        tier 0          ceil(n * widths[0] / 32)    FIXED: one code per weight
        tier k >= 1     ceil(e_k * widths[k] / 32)  DATA-DEPENDENT: e_k = the weights
                                                    that escaped tiers 0..k-1, 0..n
        terminal        ceil(e_T * exponent / 32)   DATA-DEPENDENT: e_T = the weights
                                                    that escaped every tier, 0..n

    (bf16 sip, n = 1024: 256 + 96 fixed, then 0..256 terminal words, so a
    block holds 352..608 words; gulp (2, 2, 4, 8): 256 + 64 fixed, then
    0..64 + 0..128 + 0..256, so 320..768.) The floor is what the directory
    must leave every block; the ceiling is what no block can exceed (each
    decoder requires exact consumption, so a longer block never decodes) —
    and, read the other way, the farthest word past a block's start any
    reader can ask for, which is what the device allocations pad by."""
    n = np.asarray(n, dtype=np.int64)
    fixed = (n * (mant + 1) + 31) // 32 + (n * widths[0] + 31) // 32
    most = fixed + sum((n * w + 31) // 32 for w in widths[1:-1]) + (n * exponent + 31) // 32
    return fixed, most


def validate(pack: RadixPack) -> None:
    """Check descriptors and array bounds before any accelerator allocation.

    What this proves, without reading a payload word: the directory is
    monotone from 0 to the payload end with one entry per block, and every
    block's extent holds at least its FIXED streams (the literal plane and
    tier 0, whose lengths follow from the block's weight count alone) and no
    more than every stream at its widest (block_word_bounds). What it cannot
    prove: that the block's DATA-DEPENDENT streams (tiers past the first and
    the terminal exponents, whose lengths follow from how many weights
    escaped) fit its extent — that takes decoding the block. A block short of
    those streams is refused by the CPU decoders ('truncated radix stream';
    the native decoder's 'short exponent stream' / 'terminal stream
    mismatch') and decodes to garbage on the device readers, whose reach
    past a block's start is at most block_word_bounds(B)[1] words and whose
    allocations carry that many zero words after the payload
    (swap.to_device_radix, metal/gemv_radix.pad_words) — with the
    directory bounded here, never a load outside the allocation.
    docs/pack-format.md, "Verification"."""
    mant, exponent, storage = _fields(pack.dtype)
    if (len(pack.shape) != 2 or any(type(n) is not int or n <= 0 for n in pack.shape)
            or type(pack.raw) is not bool):
        raise ValueError('invalid radix shape or raw flag')
    b = pack.block_size
    if type(b) is not int or b < 32 or b > 4096 or b & (b - 1):
        raise ValueError('invalid radix block size')
    covered = _check_widths(pack.widths, exponent)
    if pack.palette.ndim != 1 or pack.palette.dtype != np.uint8:
        raise ValueError('invalid radix palette')
    if pack.offsets.ndim != 1 or pack.offsets.dtype != np.dtype('<u4'):
        raise ValueError('invalid radix directory')
    if pack.raw:
        if (pack.data.shape != pack.shape or pack.data.dtype != storage
                or pack.offsets.size or pack.palette.size):
            raise ValueError('invalid raw radix tensor')
    else:
        nb = pack.shape[0] * ((pack.shape[1] + b - 1) // b)
        if (pack.palette.size != covered or len(np.unique(pack.palette)) != covered
                or np.any(pack.palette >= 1 << exponent)):
            raise ValueError('invalid radix palette entries')
        if pack.data.ndim != 1 or pack.data.dtype != np.dtype('<u4'):
            raise ValueError('invalid radix payload')
        if (pack.offsets.size != nb + 1 or pack.offsets[0] != 0
                or pack.offsets[-1] != pack.data.size
                or np.any(pack.offsets[1:] <= pack.offsets[:-1])):
            raise ValueError('invalid radix block offsets')
        # Each block's extent against the fixed streams and the widest
        # possible block (block_word_bounds): O(blocks) integer arithmetic on
        # the directory, no payload word read. A row's blocks are all full
        # (b weights) but its last, which holds the remainder.
        per_row = (pack.shape[1] + b - 1) // b
        n = np.minimum(b, pack.shape[1] - np.arange(nb, dtype=np.int64) % per_row * b)
        length = pack.offsets[1:].astype(np.int64) - pack.offsets[:-1].astype(np.int64)
        fewest, most = block_word_bounds(n, mant, exponent, pack.widths)
        short = np.flatnonzero(length < fewest)
        if short.size:
            i = int(short[0])
            raise ValueError(f'invalid radix block offsets: block {i} holds {int(length[i])} word(s), '
                             f'fewer than the {int(fewest[i])} its literal plane and first tier '
                             f'take for {int(n[i])} weights')
        long = np.flatnonzero(length > most)
        if long.size:
            i = int(long[0])
            raise ValueError(f'invalid radix block offsets: block {i} holds {int(length[i])} words, '
                             f'more than the {int(most[i])} any block of {int(n[i])} weights can use')


def decode(pack: RadixPack) -> np.ndarray:
    """CPU oracle, returning the original integer bit patterns."""
    validate(pack)
    mant, exponent, storage = _fields(pack.dtype)
    if pack.raw:
        if pack.data.shape != pack.shape or pack.data.dtype != storage:
            raise ValueError('invalid raw radix tensor')
        return pack.data.copy()
    out = np.empty(pack.shape, dtype=storage)
    block = 0
    for r in range(pack.shape[0]):
        for c in range(0, pack.shape[1], pack.block_size):
            count = min(pack.block_size, pack.shape[1] - c)
            start, end = map(int, pack.offsets[block:block+2])
            if end < start or end > len(pack.data):
                raise ValueError('invalid radix block offsets')
            data = pack.data[start:end]
            literal, pos = _get(data, 0, count, mant + 1)
            exps = np.empty(count, dtype=np.uint32)
            active = np.arange(count)
            base = 0
            for width in pack.widths[:-1]:
                code, pos = _get(data, pos, len(active), width)
                escape = (1 << width) - 1
                done = code != escape
                indices = base + code[done]
                if np.any(indices >= len(pack.palette)):
                    raise ValueError('invalid radix palette index')
                exps[active[done]] = pack.palette[indices]
                active = active[~done]
                base += escape
            exps[active], pos = _get(data, pos, len(active), pack.widths[-1])
            if pos != len(data):
                raise ValueError('radix block has trailing words')
            out[r, c:c+count] = ((literal & ((1 << mant) - 1)) |
                                ((literal >> mant) << (mant + exponent)) | (exps << mant))
            block += 1
    return out
