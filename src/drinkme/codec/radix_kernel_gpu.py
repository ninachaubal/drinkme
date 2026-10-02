"""Gulp/sip radix execution with precomputed stream lengths and fused output.

The checkpoint streams are unchanged. Gulp uses two extra metadata bytes
per block on the GPU; sip derives its terminal position from the directory.
The launch schedule per profile family is radix_schedule.py's measured
table; the acceptance gates are
bench/radix_gemv_bitpin.py and bench/radix_schedule_bitpin.py.
"""
import triton
import triton.language as tl
from .radix_gpu import _decode_block, _read, _lookup, _weight


@triton.jit
def _decode_predicated(data, offsets, palette, block, n, M: tl.constexpr, E: tl.constexpr,
                       WIDTHS: tl.constexpr, B: tl.constexpr):
    lane = tl.arange(0, B)
    active = lane < n
    pos = tl.load(offsets + block).to(tl.int64)
    literal = _read(data, pos, lane, active, M + 1)
    pos += tl.cdiv(n * (M + 1), 32)
    rank = lane
    count = n
    exponent = tl.zeros((B,), tl.uint32)
    table = 0
    for level in tl.static_range(len(WIDTHS) - 1):
        width = WIDTHS[level]
        escape = (1 << width) - 1
        code = _read(data, pos, rank, active, width)
        pos += tl.cdiv(count * width, 32)
        value = _lookup(palette, table, code, WIDTHS[level], B)
        exponent |= tl.where(active & (code != escape), value, 0)
        table += ((1 << WIDTHS[level]) + 2) // 4
        active &= code == escape
        rank = tl.cumsum(active.to(tl.int32)) - 1
        if level < len(WIDTHS) - 2:
            count = tl.sum(active.to(tl.int32))
    # The terminal loads already have per-lane masks. No block-wide escape
    # count or branch is needed, even when every terminal load is inactive.
    exponent |= _read(data, pos, rank, active, E)
    return (literal & ((1 << M) - 1)) | ((literal >> M) << (M + E)) | (exponent << M)


@triton.jit
def _prepare_schedule(data, offsets, schedule, C: tl.constexpr, M: tl.constexpr,
                      WIDTHS: tl.constexpr, B: tl.constexpr):
    block = tl.program_id(0)
    n = tl.minimum(C - block % triton.cdiv(C, B) * B, B)
    lane = tl.arange(0, B)
    active = lane < n
    rank = lane
    count = n
    pos = tl.load(offsets + block).to(tl.int64) + tl.cdiv(n * (M + 1), 32)
    header = tl.full((), 0, tl.uint32)
    for level in tl.static_range(len(WIDTHS) - 1):
        width = WIDTHS[level]
        words = tl.cdiv(count * width, 32)
        if level > 0:
            header |= words << ((level - 1) * 8)
        if level < len(WIDTHS) - 2:
            code = _read(data, pos, rank, active, width)
            active &= code == (1 << width) - 1
            rank = tl.cumsum(active.to(tl.int32)) - 1
            count = tl.sum(active.to(tl.int32))
        pos += words
    tl.store(schedule + block, header)


@triton.jit
def _decode_scheduled(data, offsets, palette, schedule, block, n, M: tl.constexpr,
                      E: tl.constexpr, WIDTHS: tl.constexpr, B: tl.constexpr):
    lane = tl.arange(0, B)
    active = lane < n
    pos = tl.load(offsets + block).to(tl.int64)
    end = tl.load(offsets + block + 1).to(tl.int64)
    header = tl.load(schedule + block).to(tl.uint32) if len(WIDTHS) > 2 else 0
    literal = _read(data, pos, lane, active, M + 1)
    pos += tl.cdiv(n * (M + 1), 32)
    rank = lane
    exponent = tl.zeros((B,), tl.uint32)
    table = 0
    for level in tl.static_range(len(WIDTHS) - 1):
        width = WIDTHS[level]
        code = _read(data, pos, rank, active, width)
        if level == 0:
            pos += tl.cdiv(n * width, 32)
        else:
            pos += (header >> ((level - 1) * 8)) & 255
        escape = (1 << width) - 1
        value = _lookup(palette, table, code, WIDTHS[level], B)
        exponent |= tl.where(active & (code != escape), value, 0)
        table += ((1 << WIDTHS[level]) + 2) // 4
        active &= code == escape
        if level < len(WIDTHS) - 2:
            rank = tl.cumsum(active.to(tl.int32)) - 1
    if end > pos:
        rank = tl.cumsum(active.to(tl.int32)) - 1
        exponent |= _read(data, pos, rank, active, E)
    return (literal & ((1 << M) - 1)) | ((literal >> M) << (M + E)) | (exponent << M)


@triton.jit
def _decode_sip_lean(data, offsets, palette, block, n, M: tl.constexpr, E: tl.constexpr,
                     WIDTHS: tl.constexpr, B: tl.constexpr):
    """_decode_scheduled for sip's two-tier streams (WIDTHS = (3, 8), an
    8-bit literal), returning the weight's fp32 bit pattern (the bf16 bits
    << 16) with fewer instructions per weight. The lanes are decoded in
    groups of 8 consecutive ones, a thread's own at B = 1024 and 4 warps:

      literal   an 8-bit field of a little-endian word stream is its byte:
                one uint8 load per lane at byte 4 * pos + lane
      tier 0    a group's 8 codes are 24 consecutive bits: the two words
                they can touch, joined into 64 bits and shifted once, give
                the group's chunk, one 3-bit field per lane
      escapes   SWAR over the chunk's 3-bit fields: one bit per escaping
                lane (code 7), then one multiply gives every lane's
                exclusive rank within the group at once
      prefix    one cumsum over one element per group (its escape count)
                gives each group's first terminal byte
      palette   the table's two words byte-reversed into 64 bits: a funnel
                shift by 8 * code places entry `code` at the exponent's
                bits; the escape code (7) indexes the table's zero pad byte
                (radix_pack.padded_palette_words), so no select
      terminal  a group's escapes are its terminal bytes first .. first +
                count - 1: the group's 8-byte window from `first` (three
                word loads), each escaping lane's byte placed at the
                exponent's bits by a shift like the palette's
      merge     sign and mantissa from the literal and the palette entry's
                exponent in one masked merge, the terminal byte ORed in

    Whether the block escapes at all is one 32-bit compare of its directory
    entries: radix.validate bounds the directory (monotone, inside the
    payload, each block holding at least its fixed streams), so the start
    plus the literal and tier-0 words never exceeds the next entry. The
    tier-0 and terminal loads are unmasked. A group's `first` is at most
    B - 8, so the window's third word can be word B / 4 of the terminal,
    one past a full terminal: one word past radix.block_word_bounds(B)[1]
    of the block's start. That word is inside the allocation: validate
    leaves every block start at least its fixed B / 4 + 3 * B / 32 words
    before the payload's end, so the word is at most B / 4 words past the
    payload, inside the block_word_bounds(B)[1] zero words
    swap.to_device_radix allocates after it (_read's docstring).

    The decoded bits are _decode_scheduled's: bench/radix_gemv_bitpin.py
    and radix_mc_bitpin.py gate every decoder the launch selects,
    radix_twin_bitpin.py the GEMV against its order-matched twin,
    bench/radix_lean_gate.py compares the two decoders tensor by tensor on
    a pack, and bench/radix_lean_interp.py runs both under the Triton
    interpreter on the CPU. radix_ops._decoder selects it on a CUDA build
    only (it was measured on NVIDIA only) and for whole-block rows only:
    the GEMV around it is bitwise the scheduled decoder's there, and on a
    ragged row it is not (radix_ops._decoder's docstring). On CUDA the sip
    GEMV was ALU-bound under the previous lean decoder: an L4 held its
    72 W cap at about 1.1 GHz and an H100 reached 0.20 of its read
    bandwidth (the nvidia-speed profile, 2026-09-28). bench/radix_sass_count.py
    counts 25.25 SASS instructions per weight per thread in the GEMV's tile
    loop on sm_89, against 43.5 for the previous lean decoder and 49.6 for
    the scheduled one, at 39 registers (48 of 48 warps resident)."""
    tl.static_assert(len(WIDTHS) == 2, "the lean decoder reads two-tier (sip) streams")
    tl.static_assert(M + 1 == 8 and E == 8, "the lean decoder reads 8-bit literal and terminal fields")
    tl.static_assert(WIDTHS[0] <= 3, "the lean decoder's palette is two words")
    W0: tl.constexpr = WIDTHS[0]
    tl.static_assert(W0 == 3, "the lean decoder counts escapes in 3-bit fields")
    lane = tl.arange(0, B)
    G: tl.constexpr = 8  # lanes per group
    k = lane % G
    sh = (k * W0).to(tl.uint32)
    start = tl.load(offsets + block)
    stop = tl.load(offsets + block + 1)
    pos = start.to(tl.int64)
    raw = data.to(tl.pointer_type(tl.uint8))
    literal = tl.load(raw + pos * 4 + lane, mask=lane < n, other=0).to(tl.uint32)
    pos += tl.cdiv(n * 8, 32)
    bit = lane // G * (G * W0)
    word = data + pos + bit // 32
    lo = tl.load(word).to(tl.uint64)
    hi = tl.load(word + 1).to(tl.uint64)
    chunk = ((lo | (hi << 32)) >> (bit % 32).to(tl.uint64)).to(tl.uint32)
    pos += tl.cdiv(n * W0, 32)
    # e: bit 3k set iff lane k's code is 7 (the escape)
    e = chunk & (chunk >> 1) & (chunk >> 2) & 0x249249
    t0 = tl.load(palette)
    t1 = tl.load(palette + 1)
    code8 = ((chunk << 3) >> sh) & 0x38
    # R: the table's bytes reversed, then >> 1 (byte 7, the pad, is
    # zero, so no bit is lost): byte c of the table is R's bits
    # 55 - 8c .. 62 - 8c, so hi32(R << 8c) holds it at 23..30
    r0 = (t0 >> 24) | ((t0 >> 8) & 0xFF00) | ((t0 & 0xFF00) << 8) | (t0 << 24)
    r1 = (t1 >> 24) | ((t1 >> 8) & 0xFF00) | ((t1 & 0xFF00) << 8) | (t1 << 24)
    rev = ((r0.to(tl.uint64) << 32) | r1.to(tl.uint64)) >> 1
    bpal = ((rev << code8.to(tl.uint64)) >> 32).to(tl.uint32)
    # one masked merge: the literal's low 16 bits are zero
    lit = literal * 0x01010000
    # a constant mask makes LLVM rewrite the merge as two ANDs and an OR,
    # narrowing the literal's constant by its known zero bits, and ptxas
    # then spends two LOP3: the mask ORs in the table's pad byte (zero), so
    # LLVM keeps the xor form and ptxas makes it one three-register LOP3
    exp_mask = 0x7F800000 | (t1 >> 24)
    out = lit ^ ((lit ^ bpal) & exp_mask)
    # in 32 bits: radix.validate bounds the directory (each block holds its
    # fixed streams), so start + the fixed words <= stop
    escapes = stop > start + (tl.cdiv(n * 8, 32) + tl.cdiv(n * W0, 32))
    if escapes:
        # in-group exclusive ranks in every 3-bit field at once (a field
        # counts at most 7 lanes, so none carries into the next)
        p = e * 0x249248
        total = ((p >> 21) & 7) + ((e >> 21) & 1)
        # one element per group: its cumsum is the group's inclusive
        # prefix, and broadcast back every lane of the group holds it
        per = tl.max(tl.reshape(total.to(tl.int32), [B // G, G]), axis=1)
        first = tl.reshape(tl.broadcast_to((tl.cumsum(per, 0) - per)[:, None], [B // G, G]), [B])
        # the group's escapes are terminal bytes first .. first + total - 1:
        # the 8 bytes from `first` as 64 bits
        qw = data + pos + (first >> 2)
        q0 = tl.load(qw).to(tl.uint64)
        q1 = tl.load(qw + 1).to(tl.uint64)
        q2 = tl.load(qw + 2).to(tl.uint64)
        s = ((first & 3) * 8).to(tl.uint64)
        qlo = ((q0 | (q1 << 32)) >> s).to(tl.uint32)
        qhi = ((q1 | (q2 << 32)) >> s).to(tl.uint32)
        # lanes 0..6 (rank <= 6): the window << 7, built in 32-bit halves,
        # byte r at bits 8r + 7 .. 8r + 14, so hi32(q7 << 8 * (6 - r)) holds
        # it at 23..30, and hi32(q7 << 56) holds zeros there (bits -1..6)
        q7 = ((qlo << 7).to(tl.uint64)
              | ((((qlo.to(tl.uint64) | (qhi.to(tl.uint64) << 32)) >> 25).to(tl.uint32)).to(tl.uint64) << 32))
        # fields 0..6: 6 - rank for an escape, 7 for any other lane
        # (field 7 too, but when all 8 lanes escape: 6 - 7 wraps to 7)
        g = (0xDB6DB6 - p) | ((e ^ 0x249249) * 7)
        s8 = ((g << 3) >> sh) & 0x38
        term = ((q7 << s8.to(tl.uint64)) >> 32).to(tl.uint32)
        # lane 7 of a group whose 8 lanes escape: the window's byte 7,
        # which q7 lost, is qhi's top byte
        term = tl.where((k == G - 1) & (total == G), qhi >> 1, term)
        out |= term & 0x7F800000
    return out


@triton.jit
def _flags2(w):
    """Bit 4r set iff 2-bit field r (bits 2r, 2r + 1) of w's low 16 bits
    is 3, the escape: the eight fields spread into nibbles, then each
    nibble's two low bits ANDed."""
    x = w & 0xFFFF
    x = (x | (x << 8)) & 0x00FF00FF
    x = (x | (x << 4)) & 0x0F0F0F0F
    x = (x | (x << 2)) & 0x33333333
    return x & (x >> 1) & 0x11111111


@triton.jit
def _flags4(w):
    """Bit 4r set iff 4-bit field r of w is 15, the escape."""
    t = w & (w >> 2)
    return t & (t >> 1) & 0x11111111


@triton.jit
def _prefix(flags):
    """Nibble k (k = 0..8) of the 64-bit result counts the set flags (bits
    4j) below field k: one multiply adds every flag into every higher
    nibble, and a nibble holds at most 8, so none carries."""
    return flags.to(tl.uint64) * tl.full((), 0x111111110, tl.uint64)


@triton.jit
def _nibble(p, k):
    """Nibble k of the 64-bit prefix p, as uint32."""
    return ((p >> (4 * k).to(tl.uint64)) & 15).to(tl.uint32)


@triton.jit
def _group_first(total, B: tl.constexpr, G: tl.constexpr):
    """Each lane's group's exclusive prefix of the groups' `total`s (every
    lane of a group holds its group's total): one cumsum over one element
    per group, broadcast back to the group's lanes."""
    per = tl.max(tl.reshape(total.to(tl.int32), [B // G, G]), axis=1)
    return tl.reshape(tl.broadcast_to((tl.cumsum(per, 0) - per)[:, None], [B // G, G]), [B])


@triton.jit
def _window(data, pos, first, W: tl.constexpr):
    """32 bits of the W-bit field stream at word `pos`, from field `first`
    on: the two words they can touch, joined into 64 bits and shifted
    once."""
    bit = first * W
    word = data + pos + bit // 32
    lo = tl.load(word).to(tl.uint64)
    hi = tl.load(word + 1).to(tl.uint64)
    return ((lo | (hi << 32)) >> (bit % 32).to(tl.uint64)).to(tl.uint32)


@triton.jit
def _decode_gulp_lean(data, offsets, palette, schedule, block, n, M: tl.constexpr, E: tl.constexpr,
                      WIDTHS: tl.constexpr, B: tl.constexpr, WHOLE: tl.constexpr):
    """_decode_scheduled for gulp's four-tier streams (WIDTHS = (2, 2, 4,
    8), an 8-bit literal), returning the weight's fp32 bit pattern (the
    bf16 bits << 16). WHOLE (constexpr) says n == B. The lanes are decoded
    in groups of 8 consecutive ones (a thread's own under the GEMV's
    8-wide layout), and every per-group quantity is a function of the
    group's words alone, so a thread computes it once for its 8 lanes:

      literal   one uint8 load per lane at byte 4 * pos + lane
      tier 0    the group's 8 2-bit codes are 16 bits of one word
      ranks     per tier, the group's escape flags as one bit per nibble
                (_flags2, _flags4) and their prefix in nibbles by one
                multiply (_prefix): nibble r is the exclusive rank of
                field r among the group's escapes, nibble `count` the
                group's total; one cumsum over one element per group
                gives each group's first field in the next tier's stream
      tiers 1-2 the group's codes of the next tier are consecutive fields
                from `first`: 32 bits of the stream from there
                (_window), a lane's code the field at its rank
      palette   a code's byte of the tier's table, the escape code
                indexing its zero pad byte (radix_pack.padded_palette_words);
                tier 2's 16 bytes as two 64-bit halves
      terminal  the group's escapes are terminal bytes first .. first +
                count - 1: the 8 bytes from `first` (three word loads),
                a lane's byte the one at its rank; read only when the
                directory says the block has a terminal stream

    The window, tier and terminal loads are unmasked. Each lies within
    radix.block_word_bounds(n)[1] words of the block's start, plus one
    word (a window's second, the terminal's third) or for a ragged
    block's tier-0 load at most B / 16 words: within the zero pad
    swap.to_device_radix allocates after the payload (_read's
    docstring)."""
    tl.static_assert(len(WIDTHS) == 4, "the gulp lean decoder reads four-tier (gulp) streams")
    tl.static_assert(WIDTHS[0] == 2 and WIDTHS[1] == 2 and WIDTHS[2] == 4 and WIDTHS[3] == 8,
                     "the gulp lean decoder reads widths (2, 2, 4, 8)")
    tl.static_assert(M + 1 == 8 and E == 8, "the gulp lean decoder reads 8-bit literal and terminal fields")
    G: tl.constexpr = 8  # lanes per group
    lane = tl.arange(0, B)
    k = (lane % G).to(tl.uint32)
    start = tl.load(offsets + block)
    stop = tl.load(offsets + block + 1)
    header = tl.load(schedule + block).to(tl.uint32)
    pos = start.to(tl.int64)
    raw = data.to(tl.pointer_type(tl.uint8))
    if WHOLE:
        literal = tl.load(raw + pos * 4 + lane).to(tl.uint32)
        pos += B // 4
        c0 = tl.load(data + pos + lane // 16) >> ((lane // G % 2) * 16).to(tl.uint32)
        pos += B // 16
    else:
        literal = tl.load(raw + pos * 4 + lane, mask=lane < n, other=0).to(tl.uint32)
        pos += tl.cdiv(n * 8, 32)
        c0 = tl.load(data + pos + lane // 16) >> ((lane // G % 2) * 16).to(tl.uint32)
        pos += tl.cdiv(n * 2, 32)
    # tier 0
    p0 = _prefix(_flags2(c0))
    if WHOLE:
        total0 = ((p0 >> 32) & 15).to(tl.uint32)
    else:
        total0 = _nibble(p0, tl.minimum(tl.maximum(n - lane // G * G, 0), G))
    code0 = (c0 >> (2 * k)) & 3
    esc0 = code0 == 3
    r1 = _nibble(p0, k)
    exponent = (tl.load(palette) >> (8 * code0)) & 255
    # tier 1
    w1 = _window(data, pos, _group_first(total0, B, G), 2)
    pos += header & 255
    p1 = _prefix(_flags2(w1))
    total1 = _nibble(p1, total0)
    code1 = (w1 >> (2 * r1)) & 3
    esc1 = esc0 & (code1 == 3)
    r2 = _nibble(p1, r1)
    exponent |= tl.where(esc0, (tl.load(palette + 1) >> (8 * code1)) & 255, 0)
    # tier 2
    w2 = _window(data, pos, _group_first(total1, B, G), 4)
    pos += (header >> 8) & 255
    p2 = _prefix(_flags4(w2))
    total2 = _nibble(p2, total1)
    code2 = (w2 >> (4 * r2)) & 15
    esc2 = esc1 & (code2 == 15)
    r3 = _nibble(p2, r2)
    lo = tl.load(palette + 2).to(tl.uint64) | (tl.load(palette + 3).to(tl.uint64) << 32)
    hi = tl.load(palette + 4).to(tl.uint64) | (tl.load(palette + 5).to(tl.uint64) << 32)
    half = tl.where(code2 >= 8, hi, lo)
    exponent |= tl.where(esc1, ((half >> (8 * (code2 & 7)).to(tl.uint64)) & 255).to(tl.uint32), 0)
    # terminal
    if stop.to(tl.int64) > pos:
        first = _group_first(total2, B, G)
        q = data + pos + (first >> 2)
        q0 = tl.load(q).to(tl.uint64)
        q1 = tl.load(q + 1).to(tl.uint64)
        q2 = tl.load(q + 2).to(tl.uint64)
        s = ((first & 3) * 8).to(tl.uint64)
        qlo = ((q0 | (q1 << 32)) >> s) & 0xFFFFFFFF
        qhi = ((q1 | (q2 << 32)) >> s) << 32
        term = (((qlo | qhi) >> (8 * r3).to(tl.uint64)) & 255).to(tl.uint32)
        exponent |= tl.where(esc2, term, 0)
    return ((literal * 0x01010000) & 0x807F0000) | (exponent << 23)


@triton.jit
def _rev_bytes(t):
    """The uint32 t with its four bytes reversed."""
    return (t >> 24) | ((t >> 8) & 0xFF00) | ((t & 0xFF00) << 8) | (t << 24)


@triton.jit
def _window32(data, pos, first, W: tl.constexpr):
    """_window in 32 bits: the two words, funnel-shifted right by the
    field's bit offset s (the high word's part as (hi << 1) << (31 - s), so
    s == 0 needs no shift by 32)."""
    bit = first.to(tl.uint32) * W
    word = data + pos + (bit >> 5)
    lo = tl.load(word)
    hi = tl.load(word + 1)
    s = bit & 31
    return (lo >> s) | ((hi << 1) << (31 - s))


@triton.jit
def _spread2(w):
    """w's low 16 bits as eight 2-bit fields, field r moved to bits 4r and
    4r + 1 (_flags2's spread)."""
    x = w & 0xFFFF
    x = (x | (x << 8)) & 0x00FF00FF
    x = (x | (x << 4)) & 0x0F0F0F0F
    return (x | (x << 2)) & 0x33333333


@triton.jit
def _funnel(lo, hi, s):
    """The low word of (hi:lo) >> s, for an s whose known bits keep it
    below 32: one funnel shift."""
    return ((lo.to(tl.uint64) | (hi.to(tl.uint64) << 32)) >> s.to(tl.uint64)).to(tl.uint32)


@triton.jit
def _fields_from(t):
    """Every bit of the 4-bit fields t .. 7 (t = 0..8): 0 - 2^(4t) in two
    shifts of 2t, neither reaching 32 (t == 8 wraps to zero)."""
    t2 = (2 * t).to(tl.uint32)
    return 0 - ((1 << t2) << t2)


@triton.jit
def _gulp_whole_cuda(data, offsets, palette, schedule, block, B: tl.constexpr):
    """_decode_gulp_lean_cuda's whole-block (n == B) body."""
    G: tl.constexpr = 8  # lanes per group
    lane = tl.arange(0, B)
    k = (lane % G).to(tl.uint32)
    start = tl.load(offsets + block)
    stop = tl.load(offsets + block + 1)
    header = tl.load(schedule + block).to(tl.uint32)
    pos = start.to(tl.int64)
    raw = data.to(tl.pointer_type(tl.uint8))
    literal = tl.load(raw + pos * 4 + lane).to(tl.uint32)
    pos += B // 4
    c0 = tl.load(data + pos + lane // 16) >> ((lane // G % 2) * 16).to(tl.uint32)
    pos += B // 16
    t0 = tl.load(palette)
    # tiers 0 and 1's tables byte-reversed, >> 1: entry c at bits 23 - 8c ..
    # 30 - 8c, so << 8c places it at the exponent's bits; the escape code
    # (3) indexes the pad byte, which is zero and loses only its bit 0
    r0 = _rev_bytes(t0) >> 1
    r1t = _rev_bytes(tl.load(palette + 1)) >> 1
    # tier 0
    x0 = _spread2(c0)
    f0 = x0 & (x0 >> 1) & 0x11111111
    p0 = f0 * 0x11111110  # nibble r: field r's exclusive rank among the escapes
    total0 = (p0 >> 28) + (f0 >> 28)
    i0 = ((c0 << 3) >> (2 * k)) & 0x18  # 8 * code0
    v0 = r0 << i0
    # a lane that stops at a tier reads field 7 of every later one, which
    # is forced to that tier's escape code and to terminal byte zero
    s1 = (((p0 | ((f0 ^ 0x11111111) * 7)) >> (4 * k)) << 2) & 0x1C  # 4 * r1
    # tier 1
    w1 = _window32(data, pos, _group_first(total0, B, G), 2)
    pos += header & 255
    x1 = _spread2(w1)
    hi1 = _fields_from(total0)
    f1 = x1 & (x1 >> 1) & (hi1 ^ 0x11111111)
    p1 = f1 * 0x11111110
    total1 = (p1 >> 28) + (f1 >> 28)
    x1 = x1 | (hi1 & 0x33333333)
    i1 = _funnel(x1 << 3, x1 >> 29, s1) & 0x18  # 8 * code1
    v1 = r1t << i1
    q1 = p1 | ((f1 ^ 0x11111111) * 7)
    s2 = _funnel(q1 << 2, q1 >> 30, s1) & 0x1C  # 4 * r2
    # tier 2
    w2 = _window32(data, pos, _group_first(total1, B, G), 4)
    pos += (header >> 8) & 255
    hi2 = _fields_from(total1)
    f2 = _flags4(w2) & (hi2 ^ 0x11111111)
    p2 = f2 * 0x11111110
    total2 = (p2 >> 28) + (f2 >> 28)
    code2 = ((w2 | hi2) >> s2) & 15
    # tier 2's 16-byte table (15 entries and the zero pad), a byte per lane
    deep = tl.load(palette.to(tl.pointer_type(tl.uint8)) + 8 + code2).to(tl.uint32)
    # terminal
    if stop.to(tl.int64) > pos:
        first = _group_first(total2, B, G).to(tl.uint32)
        q = data + pos + (first >> 2)
        q0 = tl.load(q)
        q1w = tl.load(q + 1)
        q2 = tl.load(q + 2)
        s = ((first & 3) * 8).to(tl.uint32)
        qlo = _funnel(q0, q1w, s)
        # byte 7 zero unless all 8 lanes reach the terminal
        qhi = _funnel(q1w, q2, s) & (0x00FFFFFF + (total2 >> 3) * 0xFF000000)
        q2r = p2 | ((f2 ^ 0x11111111) * 7)
        r3 = _funnel(q2r << 3, q2r >> 29, s2) & 0x38  # 8 * r3
        deep |= ((qlo.to(tl.uint64) | (qhi.to(tl.uint64) << 32)) >> r3.to(tl.uint64)).to(tl.uint32)
    bpal = v0 | v1 | (deep << 23)
    lit = literal * 0x01010000
    # the mask ORs in tier 0's pad byte (zero) so the merge stays one LOP3
    # (_decode_sip_lean's exp_mask)
    exp_mask = 0x7F800000 | (t0 >> 24)
    return lit ^ ((lit ^ bpal) & exp_mask)


@triton.jit
def _decode_gulp_lean_cuda(data, offsets, palette, schedule, block, n, M: tl.constexpr, E: tl.constexpr,
                           WIDTHS: tl.constexpr, B: tl.constexpr, WHOLE: tl.constexpr):
    """_decode_gulp_lean's bits (the weight's fp32 bit pattern, the bf16
    bits << 16) with fewer instructions per weight on NVIDIA, where the
    gulp GEMV is instruction-issue-bound. WHOLE (constexpr) says n == B;
    a ragged row (WHOLE false) runs _decode_gulp_lean itself. On a whole
    row (_gulp_whole_cuda) the lanes are decoded in groups of 8 as there,
    and everything per lane is 32-bit except one-instruction funnel shifts:

      ranks     32-bit nibble prefixes: the group's escape flags at bits 4r
                (for tiers 1 and 2 only the fields below the previous
                tier's count, _fields_from) times 0x11111110 (wrapping) put
                field r's exclusive rank in nibble r (each at most 7, so no
                carry); the group's count is nibble 7 plus flag 7
      fields    each lane's rank pre-scaled (4 * r) and each window shifted
                by it in one 64-bit funnel shift (_funnel): tier 1's codes
                spread into nibbles (_spread2), tier 2's 4-bit codes as they
                are, the next tier's rank from the prefix as 4 * r
      tiers 0-1 the table word byte-reversed, >> 1, shifted left by 8 * code
                (_decode_sip_lean's palette): the entry lands at the
                exponent's bits and the escape code indexes the zero pad
      tier 2    the lane's byte of the 16-byte table (15 entries and the
                pad), one uint8 load per lane (radix_pack.padded_palette_words:
                tier 2's table is bytes 8..23 of the palette words)
      terminal  the group's 8 bytes from `first` (three word loads, two
                funnel shifts per group), the lane's byte by one 64-bit
                shift right by 8 * rank
      no selects a lane that stops at a tier reads field 7 at every later
                one: a non-escaping field's rank is forced to 7, fields from
                the tier's count on are forced to its escape code (whose
                entry is the zero pad), and the terminal window's byte 7 is
                zeroed unless all 8 lanes reach the terminal. Field 7 is
                only ever real when the previous tier's count is 8, and
                then no lane stops before it. So every tier contributes
                zero where the lane did not end, and the tiers are ORed:
                tiers 0-1 placed, tier 2's or the terminal's byte << 23
      merge     sign and mantissa from the literal and the exponent in one
                masked merge (_decode_sip_lean's exp_mask)

    The literal, tier-0, window and terminal loads are unmasked, and none
    reaches further than _decode_gulp_lean's: the literal and tier-0 loads
    are its WHOLE ones; a group's `first` is at most B - 8, so tier 1's and
    tier 2's windows (_window32, the same two words as _window) end at most
    one word past that tier's stream at its widest (B / 16 and B / 8
    words), and the terminal's three words at word B / 4 of the terminal,
    one past a full terminal: each within radix.block_word_bounds(B)[1]
    words of the block's start plus one word, inside the zero pad
    swap.to_device_radix allocates after the payload (_read's docstring).
    The per-lane palette byte is at most byte 23 of the six palette words.

    bench/radix_lean_interp.py runs it under the Triton interpreter against
    the scheduled decoder. bench/radix_sass_count.py counts 51.00 SASS
    instructions per weight per thread on sm_89 at one warp and one block
    per program (C = 4096, 128 registers), against 68.25 for
    _decode_gulp_lean (157) and 122.00 for the scheduled decoder (225)."""
    tl.static_assert(len(WIDTHS) == 4, "the gulp lean decoder reads four-tier (gulp) streams")
    tl.static_assert(WIDTHS[0] == 2 and WIDTHS[1] == 2 and WIDTHS[2] == 4 and WIDTHS[3] == 8,
                     "the gulp lean decoder reads widths (2, 2, 4, 8)")
    tl.static_assert(M + 1 == 8 and E == 8, "the gulp lean decoder reads 8-bit literal and terminal fields")
    if WHOLE:
        return _gulp_whole_cuda(data, offsets, palette, schedule, block, B)
    else:
        return _decode_gulp_lean(data, offsets, palette, schedule, block, n, M, E, WIDTHS, B, WHOLE)


@triton.jit
def _decode(data, offsets, palette, schedule, block, n, M: tl.constexpr, E: tl.constexpr,
            WIDTHS: tl.constexpr, B: tl.constexpr, DECODER: tl.constexpr):
    """The block's element bits under DECODER 0, 1 or 2."""
    tl.static_assert(DECODER != 3 and DECODER != 4 and DECODER != 5,
                     "DECODER 3, 4 and 5 return fp32 bits: _decode_weight and _dense call them")
    if DECODER == 1:
        return _decode_predicated(data, offsets, palette, block, n, M, E, WIDTHS, B)
    elif DECODER == 2:
        return _decode_scheduled(data, offsets, palette, schedule, block, n, M, E, WIDTHS, B)
    return _decode_block(data, offsets, palette, block, n, M, E, WIDTHS, B)


@triton.jit
def _decode_weight(data, offsets, palette, schedule, scale, block, n, row, col, C: tl.constexpr,
                   M: tl.constexpr, E: tl.constexpr, WIDTHS: tl.constexpr, B: tl.constexpr,
                   DECODER: tl.constexpr):
    """The block's weights as fp32, the GEMV's operand: DECODER 3
    decodes the fp32 bit pattern itself (_decode_sip_lean), the others
    decode the element's bits for _weight to widen."""
    if DECODER == 3:
        return _decode_sip_lean(data, offsets, palette, block, n, M, E, WIDTHS, B).to(tl.float32, bitcast=True)
    elif DECODER == 4:
        return _decode_gulp_lean(data, offsets, palette, schedule, block, n, M, E, WIDTHS, B,
                                 C % B == 0).to(tl.float32, bitcast=True)
    elif DECODER == 5:
        return _decode_gulp_lean_cuda(data, offsets, palette, schedule, block, n, M, E, WIDTHS, B,
                                      C % B == 0).to(tl.float32, bitcast=True)
    else:
        bits = _decode(data, offsets, palette, schedule, block, n, M, E, WIDTHS, B, DECODER)
        return _weight(bits, scale, row, col, C, E)


@triton.jit
def _dense(out, data, offsets, palette, schedule, scale, C: tl.constexpr, M: tl.constexpr,
           E: tl.constexpr, WIDTHS: tl.constexpr, B: tl.constexpr, DECODER: tl.constexpr,
           DEQUANT: tl.constexpr):
    block = tl.program_id(0)
    row = block // triton.cdiv(C, B)
    col = block % triton.cdiv(C, B) * B + tl.arange(0, B)
    if DECODER == 3:
        bits = _decode_sip_lean(data, offsets, palette, block, tl.minimum(C - block % triton.cdiv(C, B) * B, B),
                                M, E, WIDTHS, B) >> 16
    elif DECODER == 4:
        bits = _decode_gulp_lean(data, offsets, palette, schedule, block,
                                 tl.minimum(C - block % triton.cdiv(C, B) * B, B), M, E, WIDTHS, B, C % B == 0) >> 16
    elif DECODER == 5:
        bits = _decode_gulp_lean_cuda(data, offsets, palette, schedule, block,
                                      tl.minimum(C - block % triton.cdiv(C, B) * B, B), M, E, WIDTHS, B,
                                      C % B == 0) >> 16
    else:
        bits = _decode(data, offsets, palette, schedule, block, tl.minimum(C - block % triton.cdiv(C, B) * B, B),
                       M, E, WIDTHS, B, DECODER)
    if DEQUANT:
        bits = _weight(bits, scale, row, col, C, E).to(tl.bfloat16)
    tl.store(out + row.to(tl.int64) * C + col, bits, mask=col < C)


@triton.jit
def _raw_weight(data, row, col, x, C: tl.constexpr):
    """RAW=True's weight: `data` is the untouched row-major bf16 weight
    [R, C], read at the (row, col) the decoder would have produced — the
    order-matched twin (swap.RadixTwinLinear). The decoder's own result is
    these bf16 bits widened to fp32 (radix_gpu._weight, E == 8, or the same
    fp32 bits built by _decode_sip_lean), so the loop around this call is
    the served kernel's, fed the same values.

    The load's vector is capped at the activation load's: 16 bytes of `x`
    (8 elements of a bf16 x, 4 of an fp32 one). tl.sum(acc) reduces
    through a reorderable reshape, so which elements each thread adds is
    the loop's register layout, and Triton takes that layout from the
    widest vectorized load feeding it. The decoder's loads are gathers, so
    in the served kernel x's load sets it; an uncapped 16-byte bf16 load
    here (8 per thread) outvoted an fp32 x's 4 and reordered the sum, so
    the fp32-x GEMV differed in the last bits at every C that lets both
    loads vectorize (measured on gfx1151; bf16 x, 8 and 8, matched).
    bench/radix_twin_bitpin.py --compile compares the two kernels' layouts
    at the device's pointer specialization.

    The loop body is much smaller than the decoder's, so the compiler may
    unroll it where it keeps the served loop rolled, and on a row whose C
    is not a multiple of B that changes which products it can prove in
    bounds and so contract into an FMA: bench/radix_twin_bitpin.py
    measures the result, and its --compile mode compares the two
    specializations' ISA on the CPU."""
    VEC: tl.constexpr = 128 // x.dtype.element_ty.primitive_bitwidth
    offset = tl.max_contiguous(row.to(tl.int64) * C + col, VEC)
    return tl.load(data + offset, mask=col < C, other=0).to(tl.float32)


@triton.jit
def _widen(value, DECODER: tl.constexpr):
    """The activation as fp32. Under DECODER 3 and 5 a bf16 one is widened as
    integers (its bits << 16, what the conversion computes exactly): ptxas
    then takes a pair's upper element with one AND instead of PRMT + IMAD."""
    if (DECODER == 3 or DECODER == 5) and value.dtype == tl.bfloat16:
        return (value.to(tl.uint16, bitcast=True).to(tl.uint32) << 16).to(tl.float32, bitcast=True)
    else:
        return value.to(tl.float32)


@triton.jit
def _gemv(out, x, bias, data, offsets, palette, schedule, scale, C: tl.constexpr, M: tl.constexpr,
          E: tl.constexpr, WIDTHS: tl.constexpr, B: tl.constexpr, DECODER: tl.constexpr,
          TILES: tl.constexpr, SPLITS: tl.constexpr, BIAS: tl.constexpr, RAW: tl.constexpr = False):
    program = tl.program_id(0)
    row = program // SPLITS
    start = program % SPLITS * TILES
    batch = tl.program_id(1)
    lane = tl.arange(0, B)
    acc = tl.zeros((B,), tl.float32)
    NB: tl.constexpr = triton.cdiv(C, B)
    for tile in range(start, tl.minimum(start + TILES, NB)):
        col = tile * B + lane
        if RAW:
            weight = _raw_weight(data, row, col, x, C)
        else:
            weight = _decode_weight(data, offsets, palette, schedule, scale, row * NB + tile,
                                    tl.minimum(C - tile * B, B), row, col, C, M, E, WIDTHS, B, DECODER)
        value = _widen(tl.load(x + batch.to(tl.int64) * C + col, mask=col < C, other=0), DECODER)
        acc += tl.where(col < C, value * weight, 0)
    value = tl.sum(acc)
    if BIAS and SPLITS == 1:
        value += tl.load(bias + row).to(tl.float32)
    tl.store(out + batch.to(tl.int64) * tl.num_programs(0) + program, value)


@triton.jit
def _finish(out, partial, bias, TOTAL: tl.constexpr, R: tl.constexpr,
            NB: tl.constexpr, K: tl.constexpr, BIAS: tl.constexpr):
    row = tl.program_id(0) * 128 + tl.arange(0, 128)
    tile = tl.arange(0, K)
    values = tl.load(partial + row[:, None] * NB + tile[None, :],
                     mask=(row[:, None] < TOTAL) & (tile[None, :] < NB), other=0)
    value = tl.sum(values, 1)
    if BIAS:
        value += tl.load(bias + row % R, mask=row < TOTAL, other=0).to(tl.float32)
    tl.store(out + row, value, mask=row < TOTAL)

