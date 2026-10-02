"""Fused decode+GEMV for the RADIX codec on Metal: y = W @ x, W never built.

The radix format (docs/pack-format.md; the CPU
oracle is codec/radix.py): 1024-weight blocks, each a packed literal stream
(8 bits per weight: 7 mantissa bits + the sign), the tier-0 code stream
(sip: 3-bit codes — 7 palette entries + one escape), then the terminal
stream (8-bit exponents for the escaped lanes, in the ORDER of the escaped
lanes — the rank of the escape flag among the block's lanes locates a
weight's exponent), every stream uint32-aligned; a uint32 word directory per
block; a per-tensor palette. Gulp / balanced chain more tiers between the
first and the terminal (each tier's largest code escapes to the next).

THE PROGRAM SHAPE mirrors what the parity work chose for the Triton arm
(radix_kernel_gpu._decode_scheduled inside the whole-row _gemv program:
one program per row, the row's blocks accumulated in registers, the bias
fused, no split-K, no second launch). On Metal the row's program is ONE
SIMD GROUP: 32 lanes, each holding 32 CONSECUTIVE weights of a block —

  * a thread's 32 literals are 8 consecutive words of the literal stream
    and its 32 tier-0 codes are exactly W0 consecutive words of the code
    stream (32 x 3 bits = 96 bits = 3 words for sip) — no field ever
    straddles into another thread's words, so a thread decodes its chunk
    from registers;
  * THE RANK of an escape flag — how many lanes before this one escaped,
    which is where the lane's exponent sits in the next stream — is a
    popcount within the thread's 32-bit escape mask plus a
    simd_prefix_exclusive_sum of the per-thread popcounts across the SIMD
    group: one SIMD scan per tier per block, no threadgroup memory, no
    barrier (the Triton kernel's tl.cumsum, at the width of one warp);
  * the terminal gather is then one byte load per escaped lane
    (data[stream + rank / 4] >> 8 * (rank % 4)), ~3% of lanes on a real
    tensor; a chained (gulp) tier reads its codes the same way at its own
    width, the thread's consecutive codes out of words read once (rx_tier),
    and gulp's 4-bit tier 2 takes them with no branch per lane, in groups
    of 8 lanes (rx_lean4, the ROCm lean gulp decoder's scheme).

The 32 products per thread go into four fp32 accumulators (lane mod 4),
folded (a0 + a1) + (a2 + a3) after the row's last block; the 32 lane sums
fold through a fixed simd_shuffle_down tree; the bias is added in fp32 and
the sum rounded ONCE to bf16 (RNE, in integer ops — the torch arm's
epilogue, radix_ops.gemv_fused). No atomics, no cross-threadgroup
accumulation: the same row sums in the same order every run —
DETERMINISTIC by construction. x is read as bf16
bits (the activation dtype the MLX engine runs in) and widened in-kernel;
an fp32 x is read as is.

Blocks of 32..1024 lanes (a power of two) are served: 1024 / B threads
share a block and 32 / that many blocks ride one SIMD step, the scan's
prefix taken relative to the block's first thread. Wider blocks and more
than four tiers are refused by name (the pack writes 1024 and sip / gulp
are two and four tiers; the Triton scheduled decoder draws the same line).

Bit-pinned on the Mac by bench/radix_bitpin_mlx.py (all 65,536 bf16
patterns through the streams at 32- and 1024-weight blocks, ragged shapes,
the GEMV against a float64 dot with the cancellation-aware bound, the
fused epilogue, and real Qwen3-8B tensors); the dense decode twin is
dense_radix.py over this module's resident dict. Needs Metal; a forced call
on a machine without one fails at dispatch with mlx's own "No Metal back-end".
`import drinkme.metal` imports none of this; import this module explicitly.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np

from drinkme.codec import radix as _radix, radix_pack as _rp
from drinkme.codec.pack import FORMAT_VERSION

LPT = 32            # lanes (weights) per thread
SIMD = 32           # threads per SIMD group = threads per (1024-lane) block step
MAX_BLOCK = LPT * SIMD
MAX_TIERS = 4       # sip (3, 8), balanced (2, 3, 8), gulp (2, 2, 4, 8)
ROWS_PER_TG = 8     # rows (row pairs, PAIR) per threadgroup in the GEMV
PAIR_TAILS = True   # the paired-rows GEMV where it applies (pairs_tails; _GEMV's PAIR)
SIMD_PER_ROW = 1    # SIMD groups per row (split-K inside the threadgroup; 1 = the whole row in one)
PAD_WORDS = 16      # the pad's floor (see pad_words): a ragged chunk's 8-word literal load and W0
                    # code words stay in the buffer even when the block is a few words long
MIN_BOUND = 8       # mlx binds an input under 8 elements in the constant address space

# ---------------------------------------------------------------- MSL ----
#
# rx_decode_chunk decodes lanes [32*sub, 32*sub + 32) of block `gb` (n lanes
# long; n = 0 marks an idle slot that still takes part in the SIMD scans)
# into 32 bf16 bit patterns and the chunk's active-lane mask. Every SIMD
# intrinsic is reached by all 32 threads: the callers keep the block loop
# uniform across the SIMD group.
_HEADER = r"""
#define LPT_ 32u

inline uint rx_sel4(uint4 p, uint i) {
    return i == 0u ? p.x : (i == 1u ? p.y : (i == 2u ? p.z : p.w));
}

// The even bits of x (bit 2i) gathered to bits 0..15 (bit i).
inline uint rx_even_bits(uint x) {
    x &= 0x55555555u;
    x = (x | (x >> 1)) & 0x33333333u;
    x = (x | (x >> 2)) & 0x0F0F0F0Fu;
    x = (x | (x >> 4)) & 0x00FF00FFu;
    return (x | (x >> 8)) & 0x0000FFFFu;
}

inline uint rx_bf16_bits(float f) {
    uint u = as_type<uint>(f);
    if ((u & 0x7F800000u) == 0x7F800000u && (u & 0x007FFFFFu) != 0u) {
        return ((u >> 16) | 0x40u) & 0xFFFFu;          // a NaN stays a (quiet) NaN
    }
    return (u + 0x7FFFu + ((u >> 16) & 1u)) >> 16;      // round to nearest even
}

// The rank scan of one tier over the SIMD group: c = this thread's escapes;
// r0 = the escapes before this thread's chunk in its block, tot = the
// block's. PAIRED (the GEMV's paired rows, _GEMV): a block spans `seg`
// lanes (16 for the side-by-side tails), and on a `rot` step lane L holds
// chunk L ^ 16 — the scan runs in chunk order and each lane takes its own
// chunk's prefix back.
template <int TPB, bool PAIRED>
inline void rx_scan(uint c, ushort lane, uint seg, bool rot, thread uint& r0, thread uint& tot) {
    uint pre, base;
    if (PAIRED) {
        uint cx = simd_shuffle_xor(c, (ushort)16);
        uint cc = rot ? cx : c;
        pre = simd_prefix_exclusive_sum(cc);
        ushort first = (ushort)((uint)lane & ~(seg - 1u));
        base = simd_shuffle(pre, first);
        tot = simd_shuffle(pre + cc, (ushort)(first + seg - 1u)) - base;
        uint px = simd_shuffle_xor(pre, (ushort)16);
        pre = rot ? px : pre;
    } else {
        pre = simd_prefix_exclusive_sum(c);
        ushort first = (ushort)((lane / TPB) * TPB);
        base = simd_shuffle(pre, first);
        tot = simd_shuffle(pre + c, (ushort)(first + TPB - 1)) - base;
    }
    r0 = pre - base;
}

// Whether lane i escaped: bit i of m, or (SIP3) bit 3i of the 96-bit
// e2:e1:e0 (rx_decode_chunk's sip test). i is a constant once unrolled.
template <bool SIP3>
inline uint rx_esc(uint i, uint m, uint e0, uint e1, uint e2) {
    if (SIP3) {
        uint bit = 3u * i;
        return ((bit < 32u ? e0 : (bit < 64u ? e1 : e2)) >> (bit & 31u)) & 1u;
    }
    return (m >> i) & 1u;
}

// The terminal exponents of this thread's escaped lanes (rx_esc). The
// escapes hold ranks r0, r0 + 1, ...: consecutive bytes of the exponent
// stream. When they all lie in the two words from r0's on (k = r0 % 4 plus
// the count at most 8: every thread but a rare one), the two words are
// read ONCE and each escaped lane takes its byte from them; otherwise the
// thread takes one load per escaped lane. Same bytes either way. The
// one-load-per-lane form alone cost the Qwen3-4B decode step 3.9% on an
// M4. The second word may be the next block's first
// (tests/test_radix_bounds.py models the reach); no escaped lane's byte is
// in it unless the stream reaches it.
template <bool SIP3>
inline void rx_term_window(const device uint* data, uint spos, uint r0, uint c, uint m,
                           uint e0, uint e1, uint e2, thread uint (&ex)[32]) {
    uint k = r0 & 3u;
    if (k + c <= 8u) {
        uint w0 = 0u, w1 = 0u;
        if (c != 0u) { w0 = data[spos + (r0 >> 2)]; w1 = data[spos + (r0 >> 2) + 1u]; }
        #pragma unroll
        for (uint i = 0u; i < 32u; ++i) {
            if (rx_esc<SIP3>(i, m, e0, e1, e2)) {
                ex[i] = (k < 4u ? (w0 >> (8u * k)) : (w1 >> (8u * (k - 4u)))) & 0xFFu;
                ++k;
            }
        }
    } else {
        uint r = r0;
        #pragma unroll
        for (uint i = 0u; i < 32u; ++i) {
            if (rx_esc<SIP3>(i, m, e0, e1, e2)) {
                ex[i] = (data[spos + (r >> 2)] >> (8u * (r & 3u))) & 0xFFu;
                ++r;
            }
        }
    }
}

// The W bits at bit p of w1:w0 (p + W <= 64). A W that divides 32 never
// straddles the two words: its codes start at multiples of W.
template <int W>
inline uint rx_bits(uint w0, uint w1, uint p) {
    uint s = p & 31u;
    uint v = (p < 32u ? w0 : w1) >> s;
    if (32 % (W > 0 ? W : 1) != 0) {                   // W = 0: a tier the profile does not have
        if (p < 32u && s + (uint)W > 32u) { v |= w1 << (32u - s); }
    }
    return v & ((1u << (uint)W) - 1u);
}

// One chained tier over the lanes still escaped (mask m): their code /
// exponent sits at their RANK in the tier's stream. TERM: the terminal
// exponent stream (W = 8, nothing escapes further).
//
// A thread's escaped lanes hold consecutive ranks r0, r0 + 1, ..., so their
// codes are the consecutive bits [W * r0, W * (r0 + c)) of the stream: the
// words that hold them are read ONCE, before the lanes, and each escaped
// lane takes the next W bits (a cursor). W <= 2: three words always hold
// them (k + W * c <= 31 + 64), and the tier's palette is one word, held in
// a register. W > 2: two words when they hold them (every thread but a
// rare one), else one load per escaped lane (rx_term_window's two forms);
// the palette byte is read per lane (four words held in registers cost
// more than the loads). On a Qwen3-4B gulp step (tiers 2, 2, 4, 8) on an
// M4, the per-lane loads cost the step 1.4x. IDX (gulp, rx_decode_chunk's
// LEAN4; W <= 2 only): a lane takes its code's palette INDEX, 4 * tbase + v
// (its byte's place in the palette words), not the byte.
template <int W, bool TERM, int TPB, bool PAIRED = false, bool IDX = false>
inline void rx_tier(const device uint* data, const device uint* palette, ushort lane,
                    thread uint& m, thread uint& spos, thread uint& tbase,
                    thread uint (&ex)[32], uint seg = (uint)TPB, bool rot = false) {
    uint c = popcount(m);
    uint r0, tot;
    rx_scan<TPB, PAIRED>(c, lane, seg, rot, r0, tot);
    uint nm = 0u;
    if (TERM && W == 8) {
        rx_term_window<false>(data, spos, r0, c, m, 0u, 0u, 0u, ex);
    } else {
        const uint MW = (1u << (uint)W) - 1u;
        uint b0 = r0 * (uint)W;
        uint k = b0 & 31u;                               // the cursor: the next code's bit in the window
        if (W <= 2) {
            uint pt = palette[tbase];
            uint w0 = 0u, w1 = 0u, w2 = 0u;
            if (c != 0u) { w0 = data[spos + (b0 >> 5)]; w1 = data[spos + (b0 >> 5) + 1u]; w2 = data[spos + (b0 >> 5) + 2u]; }
            // the thread's at most 64 code bits from bit 0: ca, then cb (one
            // select per lane, not two; +1.2% on the Qwen3-4B gulp step on an M4)
            uint ca = (w0 >> k) | ((w1 << 1u) << (31u - k));
            uint cb = (w1 >> k) | ((w2 << 1u) << (31u - k));
            k = 0u;
            #pragma unroll
            for (uint i = 0u; i < 32u; ++i) {
                if ((m >> i) & 1u) {
                    uint v = ((k < 32u ? ca : cb) >> (k & 31u)) & MW;
                    k += (uint)W;
                    if (v == MW) { nm |= 1u << i; } else { ex[i] = IDX ? 4u * tbase + v : extract_bits(pt, 8u * (v & 3u), 8u); }
                }
            }
        } else if (k + (uint)W * c <= 64u) {
            uint w0 = 0u, w1 = 0u;
            if (c != 0u) { w0 = data[spos + (b0 >> 5)]; w1 = data[spos + (b0 >> 5) + 1u]; }
            #pragma unroll
            for (uint i = 0u; i < 32u; ++i) {
                if ((m >> i) & 1u) {
                    uint v = rx_bits<W>(w0, w1, k);
                    k += (uint)W;
                    if (v == MW) { nm |= 1u << i; } else { ex[i] = (palette[tbase + (v >> 2)] >> (8u * (v & 3u))) & 0xFFu; }
                }
            }
        } else {
            #pragma unroll
            for (uint i = 0u; i < 32u; ++i) {
                if ((m >> i) & 1u) {
                    uint bit = b0 + (uint)W * popcount(m & ((1u << i) - 1u));
                    uint sh = bit & 31u;
                    uint v = data[spos + (bit >> 5)] >> sh;
                    if (sh + (uint)W > 32u) { v |= data[spos + (bit >> 5) + 1u] << (32u - sh); }
                    v &= MW;
                    if (v == MW) { nm |= 1u << i; } else { ex[i] = (palette[tbase + (v >> 2)] >> (8u * (v & 3u))) & 0xFFu; }
                }
            }
        }
    }
    m = nm;
    spos += (tot * (uint)W + 31u) >> 5;
    tbase += (uint)(((1 << W) + 2) / 4);
}

// The thread lanes [i & ~7, i): lane i's rank within its group of 8 is the
// popcount of the group's escaped lanes under this mask (a constant once
// unrolled).
#define RX_GROUP_BELOW_(i) (((1u << (i)) - 1u) & ~((1u << (8u * ((i) >> 3u))) - 1u))

// Gulp's 4-bit tier 2 with no branch per lane: the lean gulp decoder
// (radix_kernel_gpu._decode_gulp_lean) in this kernel's thread layout, in
// rx_decode_chunk's index form (LEAN4). m: the lanes that reached the tier.
// A thread's 32 lanes are four groups of 8, and a group's escaped lanes hold
// consecutive ranks, so its at most 8 codes are the 32 bits of the stream
// from its first one (gw[g], from two words read once). Every lane takes the
// code at its rank within its group, and an escaped lane keeps its palette
// index 4 * tbase + v. On a Qwen3-4B gulp step on an M4 only ~6% of lanes
// reach this tier, but 80-88% of lane positions have one somewhere in the
// SIMD group, so rx_tier's predicated pass ran nearly whole.
template <int TPB, bool PAIRED>
inline void rx_lean4(const device uint* data, ushort lane, uint m, thread uint& spos,
                     thread uint& tbase, thread uint (&ex)[32], uint seg, bool rot) {
    uint r0, tot;
    uint gw[4];
    rx_scan<TPB, PAIRED>(popcount(m), lane, seg, rot, r0, tot);
    #pragma unroll
    for (uint g = 0u; g < 4u; ++g) {
        uint bit = 4u * (r0 + popcount(m & ((1u << (8u * g)) - 1u)));
        uint s = bit & 31u;                              // a multiple of 4: a code never straddles the words
        uint lo = data[spos + (bit >> 5)], hi = data[spos + (bit >> 5) + 1u];
        gw[g] = (lo >> s) | ((hi << 1u) << (31u - s));
    }
    #pragma unroll
    for (uint i = 0u; i < 32u; ++i) {
        uint v = (gw[i >> 3] >> (4u * popcount(m & RX_GROUP_BELOW_(i)))) & 15u;
        ex[i] = ((m >> i) & 1u) ? 4u * tbase + v : ex[i];
    }
    spos += (tot * 4u + 31u) >> 5;
    tbase += 4u;
}

template <int NT, int W0, int W1, int W2, int TPB, bool PAIRED = false>
inline void rx_decode_chunk(const device uint* data, const device uint* offsets,
                            const device uint* palette, uint gb, uint n, uint sub,
                            ushort lane, thread uint (&lw)[8], thread uint (&ex)[32],
                            thread uint& amask, uint seg = (uint)TPB, bool rot = false) {
    uint lane0 = sub * LPT_;
    uint nact = (lane0 < n) ? min(LPT_, n - lane0) : 0u;
    amask = nact >= LPT_ ? 0xFFFFFFFFu : ((1u << nact) - 1u);
    // gulp (2, 2, 4, 8): tiers 0-2 leave each lane's palette INDEX, and one
    // byte load per lane reads them all after tier 2 (rx_lean4)
    const bool LEAN4 = NT == 4 && W0 == 2 && W1 == 2 && W2 == 4;
    uint pos = offsets[gb];
    uint pend = NT > 2 ? offsets[gb + 1u] : 0u;         // read early, for the terminal skip below
    uint cw[W0 + 1];
    #pragma unroll
    for (uint k = 0u; k < 8u; ++k) { lw[k] = 0u; }
    #pragma unroll
    for (uint k = 0u; k < (uint)W0 + 1u; ++k) { cw[k] = 0u; }
    uint cpos = pos + ((n * 8u + 31u) >> 5);           // the tier-0 code stream
    if (nact != 0u) {
        // 8 literal words (4-byte aligned: packed vectors, two loads) and W0 code words
        const device packed_uint4* L = (const device packed_uint4*)(data + pos + 8u * sub);
        uint4 la = uint4(L[0]);
        uint4 lb = uint4(L[1]);
        lw[0] = la.x; lw[1] = la.y; lw[2] = la.z; lw[3] = la.w;
        lw[4] = lb.x; lw[5] = lb.y; lw[6] = lb.z; lw[7] = lb.w;
        if (W0 == 3) {          // the sip profile: one packed load for the thread's three words
            uint3 c3 = uint3(*((const device packed_uint3*)(data + cpos + 3u * sub)));
            cw[0] = c3.x; cw[1] = c3.y; cw[2] = c3.z;
        } else {
            #pragma unroll
            for (uint k = 0u; k < (uint)W0; ++k) { cw[k] = data[cpos + (uint)W0 * sub + k]; }
        }
    }
    uint spos = cpos + ((n * (uint)W0 + 31u) >> 5);     // the next stream (tier 1 or terminal)
    const bool SIP3 = NT == 2 && W0 == 3;               // sip: tier 0, then the terminal
    const uint PW0 = (uint)(((1 << W0) + 2) / 4);
    uint4 p0 = uint4(palette[0], PW0 > 1u ? palette[1] : 0u,
                     PW0 > 2u ? palette[2] : 0u, PW0 > 3u ? palette[3] : 0u);
    uint m = 0u;                                        // lanes escaping tier 0
    #pragma unroll
    for (uint i = 0u; i < 32u; ++i) {
        uint bit = (uint)W0 * i;
        uint sh = bit & 31u;
        uint v;
        if (sh + (uint)W0 > 32u) {
            v = ((cw[bit >> 5] >> sh) | (cw[(bit >> 5) + 1u] << (32u - sh))) & ((1u << (uint)W0) - 1u);
        } else {
            v = extract_bits(cw[bit >> 5], sh, (uint)W0);
        }
        uint pw = (W0 <= 3) ? ((v & 4u) ? p0.y : p0.x) : rx_sel4(p0, v >> 2);
        ex[i] = LEAN4 ? v : extract_bits(pw, 8u * (v & 3u), 8u);
        if (!SIP3 && W0 != 2) { m |= (v == (1u << (uint)W0) - 1u) ? (1u << i) : 0u; }
    }
    if (SIP3) {
        // sip's escapes (code 7: all three bits set) for the 32 lanes at once:
        // bit 3i of e2:e1:e0 is lane i's flag, limited to the chunk's active
        // lanes (bits below 3 * nact) — the flags the per-lane test gives
        uint c0 = cw[0], c1 = cw[SIP3 ? 1 : 0], c2 = cw[SIP3 ? 2 : 0];   // in bounds for every W0
        uint e0 = c0 & ((c0 >> 1) | (c1 << 31)) & ((c0 >> 2) | (c1 << 30)) & 0x49249249u;
        uint e1 = c1 & ((c1 >> 1) | (c2 << 31)) & ((c1 >> 2) | (c2 << 30)) & 0x92492492u;
        uint e2 = c2 & (c2 >> 1) & (c2 >> 2) & 0x24924924u;
        uint L = 3u * nact;
        e0 &= L >= 32u ? 0xFFFFFFFFu : ((1u << L) - 1u);
        e1 &= L >= 64u ? 0xFFFFFFFFu : (L <= 32u ? 0u : ((1u << (L - 32u)) - 1u));
        e2 &= L >= 96u ? 0xFFFFFFFFu : (L <= 64u ? 0u : ((1u << (L - 64u)) - 1u));
        uint c = popcount(e0) + popcount(e1) + popcount(e2);
        uint r0, tot;
        rx_scan<TPB, PAIRED>(c, lane, seg, rot, r0, tot);
        rx_term_window<true>(data, spos, r0, c, 0u, e0, e1, e2, ex);
        return;
    }
    if (W0 == 2) {
        // gulp / balanced's tier-0 escapes (code 3: both bits set) for the 32
        // lanes at once: bit 2i of cw[1]:cw[0] is lane i's flag, gathered to
        // bit i (sip's SWAR test at two bits; the per-lane test's flags)
        m = rx_even_bits(cw[0] & (cw[0] >> 1)) | (rx_even_bits(cw[1] & (cw[1] >> 1)) << 16);
    }
    m &= amask;
    uint tbase = PW0;
    if (NT > 2) { rx_tier<W1, false, TPB, PAIRED, LEAN4>(data, palette, lane, m, spos, tbase, ex, seg, rot); }
    // A chained profile's terminal tier is all but empty (Qwen3-4B gulp: ~1e-5
    // of lanes, ~1% of blocks): a well-formed block whose streams end where
    // its terminal stream would start has no lane escaping to it. When that
    // holds for every block of the SIMD step (an idle slot holds none), the
    // terminal's scan and lanes are skipped together, so the scan stays
    // uniform; it was ~10% of the Qwen3-4B gulp decode step on an M4. A
    // malformed block skipped here decodes to other garbage, inside the buffer.
    if (LEAN4) {
        // Each lane's palette index (tier 0: its code; tier 1: 4 + code; tier
        // 2: 8 + code; an escape code indexes its tier's zero pad byte) becomes
        // its byte by ONE byte load. Index 23 is tier 2's escape: the terminal's
        // lanes, collected only when the step runs the terminal. On the Qwen3-4B
        // gulp step on an M4, one after another: rx_lean4 in place of rx_tier's
        // predicated pass +10%, the index form +8%, the terminal's lanes from
        // the indices (no windows held for it) +7%, the byte load in place of a
        // word load and a shift +12%.
        rx_lean4<TPB, PAIRED>(data, lane, m, spos, tbase, ex, seg, rot);
        bool skip = simd_all(n == 0u || pend == spos);
        uint nm = 0u;
        if (!skip) {
            #pragma unroll
            for (uint i = 0u; i < 32u; ++i) { nm |= ex[i] == 23u ? (1u << i) : 0u; }
        }
        const device uchar* pal8 = (const device uchar*)palette;
        #pragma unroll
        for (uint i = 0u; i < 32u; ++i) { ex[i] = pal8[ex[i]]; }
        if (skip) { return; }
        m = nm;
    } else {
        if (NT > 3) { rx_tier<W2, false, TPB, PAIRED>(data, palette, lane, m, spos, tbase, ex, seg, rot); }
        if (NT > 2 && simd_all(n == 0u || pend == spos)) { return; }
    }
    rx_tier<8, true, TPB, PAIRED>(data, palette, lane, m, spos, tbase, ex, seg, rot);
}

// lane i's bf16 bit pattern off the chunk's literal words and exponents:
// literal byte i = sign (bit 7) + 7 mantissa bits; the pattern is
// sign:15 | exponent:14..7 | mantissa:6..0. RX_F32_ is the same pattern
// already at bits 31..16 — the fp32 value's bits.
#define RX_LIT_(lw, i) (((lw)[(i) >> 2] >> (8u * ((i) & 3u))) & 0xFFu)
#define RX_BITS_(lw, ex, i) \
    ((RX_LIT_(lw, i) & 0x7Fu) | ((RX_LIT_(lw, i) & 0x80u) << 8) | (((ex)[i] & 0xFFu) << 7))
#define RX_F32_(lw, ex, i) \
    as_type<float>(((RX_LIT_(lw, i) & 0x7Fu) << 16) | ((RX_LIT_(lw, i) & 0x80u) << 24) | (((ex)[i] & 0xFFu) << 23))

// The chunk's 32 products into a0..a3 (lane mod 4) in lane order: x's lanes
// [col0, col0 + 32), bf16 bits when XBF. A chunk with no active lane (amask 0,
// the idle half of Qwen3-4B's last block at C = 2560) is skipped rather than
// added as +0.0f products: an accumulator starts at +0.0 and, rounding to
// nearest even, never becomes -0.0, so adding +0.0f changes no sum's bits.
template <bool XBF, typename XT>
inline void rx_acc(const device XT* x, uint col0, thread uint (&lw)[8], thread uint (&ex)[32],
                   uint amask, thread float& a0, thread float& a1, thread float& a2, thread float& a3) {
    if (amask == 0xFFFFFFFFu) {
        if (XBF) {
            // 32 bf16 = 64 bytes at a 64-byte-aligned offset (x is one row; col0 is a multiple of 32):
            // four uint4 loads, two activations per word
            const device uint4* x4 = ((const device uint4*)x) + (col0 >> 3);
            #pragma unroll
            for (uint q = 0u; q < 4u; ++q) {
                uint4 w = x4[q];
                uint i = 8u * q;
                a0 += RX_F32_(lw, ex, i) * as_type<float>(w.x << 16);
                a1 += RX_F32_(lw, ex, i + 1u) * as_type<float>(w.x & 0xFFFF0000u);
                a2 += RX_F32_(lw, ex, i + 2u) * as_type<float>(w.y << 16);
                a3 += RX_F32_(lw, ex, i + 3u) * as_type<float>(w.y & 0xFFFF0000u);
                a0 += RX_F32_(lw, ex, i + 4u) * as_type<float>(w.z << 16);
                a1 += RX_F32_(lw, ex, i + 5u) * as_type<float>(w.z & 0xFFFF0000u);
                a2 += RX_F32_(lw, ex, i + 6u) * as_type<float>(w.w << 16);
                a3 += RX_F32_(lw, ex, i + 7u) * as_type<float>(w.w & 0xFFFF0000u);
            }
        } else {
            #pragma unroll
            for (uint i = 0u; i < 32u; i += 4u) {
                a0 += RX_F32_(lw, ex, i) * (float)x[col0 + i];
                a1 += RX_F32_(lw, ex, i + 1u) * (float)x[col0 + i + 1u];
                a2 += RX_F32_(lw, ex, i + 2u) * (float)x[col0 + i + 2u];
                a3 += RX_F32_(lw, ex, i + 3u) * (float)x[col0 + i + 3u];
            }
        }
    } else if (amask != 0u) {
        #pragma unroll
        for (uint i = 0u; i < 32u; i += 4u) {
            float xv[4];
            #pragma unroll
            for (uint j = 0u; j < 4u; ++j) {
                bool on = ((amask >> (i + j)) & 1u) != 0u;
                float xi = 0.0f;
                if (on) { xi = XBF ? as_type<float>(((uint)((const device ushort*)x)[col0 + i + j]) << 16) : (float)x[col0 + i + j]; }
                xv[j] = xi;
            }
            a0 += (((amask >> i) & 1u) ? RX_F32_(lw, ex, i) : 0.0f) * xv[0];
            a1 += (((amask >> (i + 1u)) & 1u) ? RX_F32_(lw, ex, i + 1u) : 0.0f) * xv[1];
            a2 += (((amask >> (i + 2u)) & 1u) ? RX_F32_(lw, ex, i + 2u) : 0.0f) * xv[2];
            a3 += (((amask >> (i + 3u)) & 1u) ? RX_F32_(lw, ex, i + 3u) : 0.0f) * xv[3];
        }
    }
}
"""

# params = [R, C, B, NB, 0, 0, 0, 0]. One SIMD group per row, ROWS_PER_TG
# rows per threadgroup; the row index is uniform across the SIMD group, so
# the early return and the block loop never split a SIMD op.
#
# PAIR: TWO ROWS PER SIMD GROUP, for a tensor whose rows end in a block of
# at most 16 chunks (B = 1024 and C mod 1024 in (0, 512]: Qwen3-4B's 2560
# and 9728). One row's last block would leave half the SIMD group idle for
# a whole step (C = 2560: three steps for 2.5 blocks); here rows rA and rB
# run their full blocks one after the other and then their last blocks
# SIDE BY SIDE in one step, lanes 0-15 rA's and 16-31 rB's. Every chunk's
# products still go into one thread's four accumulators in block order — rB's
# chunk c lives on lane c ^ 16 throughout, the scan taking its prefix in
# chunk order (rx_tier's PAIRED) — and rB's per-lane sums move back to lane
# c before the unpaired kernel's fold, so each row's sum is that kernel's,
# bit for bit (bench/radix_bitpin_mlx.py pins the two against each other).
_GEMV = r"""
    const uint R = params[0];
    const uint C = params[1];
    const uint B = params[2];
    const uint NB = params[3];
    const uint BPS = 32u / (uint)TPB;                    // blocks per SIMD step
    ushort lane = thread_index_in_simdgroup;
    uint sg = (uint)simdgroup_index_in_threadgroup;
    if (PAIR) {
        uint g = threadgroup_position_in_grid.x * (threads_per_threadgroup.x / 32u) + sg;
        uint rA = 2u * g, rB = rA + 1u;
        uint ra = rA < R ? rA : R - 1u, rb = rB < R ? rB : R - 1u;  // past the end: a real row, stored nowhere
        float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f, b0 = 0.0f, b1 = 0.0f, b2 = 0.0f, b3 = 0.0f;
        uint lw[8];
        uint ex[32];
        uint amask;
        // ONE step loop (one inlined copy of the decode): rA's full blocks, rB's
        // full blocks (lane L on chunk L ^ 16), then both tails side by side
        const uint F = NB - 1u;                          // full blocks per row
        bool hb = lane >= 16;
        for (uint t = 0u; t < 2u * F + 1u; ++t) {
            uint blk, gb, n, sub, seg;
            bool rot, onb;
            if (t < F) {
                blk = t; gb = ra * NB + t; n = B; sub = (uint)lane; seg = 32u; rot = false; onb = false;
            } else if (t < 2u * F) {
                blk = t - F; gb = rb * NB + blk; n = B; sub = (uint)lane ^ 16u; seg = 32u; rot = true; onb = true;
            } else {
                blk = F; gb = (hb ? rb : ra) * NB + F; n = C - F * B; sub = (uint)lane & 15u; seg = 16u; rot = false; onb = hb;
            }
            rx_decode_chunk<NT, W0, W1, W2, TPB, true>(data, offsets, palette, gb, n, sub, lane, lw, ex, amask, seg, rot);
            float t0 = onb ? b0 : a0, t1 = onb ? b1 : a1, t2 = onb ? b2 : a2, t3 = onb ? b3 : a3;
            rx_acc<XBF>(x, blk * B + sub * LPT_, lw, ex, amask, t0, t1, t2, t3);
            if (onb) { b0 = t0; b1 = t1; b2 = t2; b3 = t3; } else { a0 = t0; a1 = t1; a2 = t2; a3 = t3; }
        }
        float accA = (a0 + a1) + (a2 + a3);
        float accB = simd_shuffle_xor((b0 + b1) + (b2 + b3), (ushort)16);
        for (uint off = 16u; off > 0u; off >>= 1u) {
            accA += simd_shuffle_down(accA, off);
            accB += simd_shuffle_down(accB, off);
        }
        if (lane == 0) {
            if (rA < R) {
                if (HAS_BIAS) { accA += bias[rA]; }
                if (OUT32) { ((device float*)y)[rA] = accA; } else { ((device ushort*)y)[rA] = (ushort)rx_bf16_bits(accA); }
            }
            if (rB < R) {
                if (HAS_BIAS) { accB += bias[rB]; }
                if (OUT32) { ((device float*)y)[rB] = accB; } else { ((device ushort*)y)[rB] = (ushort)rx_bf16_bits(accB); }
            }
        }
        return;
    }
    uint rtg = sg / (uint)SPR;                           // row within the threadgroup
    uint split = sg % (uint)SPR;                         // this SIMD group's share of the row's blocks
    uint r = threadgroup_position_in_grid.x * (threads_per_threadgroup.x / (32u * (uint)SPR)) + rtg;
    bool live = r < R;
    uint rr = live ? r : R - 1u;                         // a tail SIMD group computes a real row, stores nothing
    uint slot = (uint)lane / (uint)TPB;
    uint sub = (uint)lane % (uint)TPB;
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    uint lw[8];
    uint ex[32];
    uint amask;
    for (uint s = split * BPS; s < NB; s += BPS * (uint)SPR) {
        uint b = s + slot;
        bool valid = b < NB;
        uint bb = valid ? b : NB - 1u;
        uint n = valid ? min(B, C - bb * B) : 0u;
        rx_decode_chunk<NT, W0, W1, W2, TPB>(data, offsets, palette, rr * NB + bb, n, sub, lane, lw, ex, amask);
        uint col0 = bb * B + sub * LPT_;
        rx_acc<XBF>(x, col0, lw, ex, amask, a0, a1, a2, a3);
    }
    float acc = (a0 + a1) + (a2 + a3);
    for (uint off = 16u; off > 0u; off >>= 1u) { acc += simd_shuffle_down(acc, off); }
    if (SPR > 1) {
        threadgroup float part[RPT * SPR];
        if (lane == 0) { part[sg] = acc; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (split != 0u) { return; }
        acc = part[rtg * (uint)SPR];
        for (uint k = 1u; k < (uint)SPR; ++k) { acc += part[rtg * (uint)SPR + k]; }
    }
    if (lane == 0 && live) {
        if (HAS_BIAS) { acc += bias[r]; }
        if (OUT32) { ((device float*)y)[r] = acc; } else { ((device ushort*)y)[r] = (ushort)rx_bf16_bits(acc); }
    }
"""

_gemv_kernel = mx.fast.metal_kernel(
    name="drinkme_radix_gemv",
    input_names=["data", "offsets", "palette", "params", "x", "bias"],
    output_names=["y"],
    header=_HEADER,
    source=_GEMV,
)


def _bound(arr: mx.array) -> mx.array:
    """Every input a real device buffer: zero-pad anything under MIN_BOUND
    elements (mlx binds a tiny input in the constant address space, where a
    device pointer will not compile)."""
    if arr.size >= MIN_BOUND:
        return arr
    pad = mx.zeros((MIN_BOUND - arr.size,), dtype=arr.dtype)
    return mx.concatenate([arr.reshape(-1), pad]) if arr.size else pad


def _widths(pack: dict) -> tuple[int, ...]:
    return tuple(int(w) for w in pack["widths"])


def pad_words(B: int, widths) -> int:
    """Zero words after the streams, so no load of rx_decode_chunk leaves the
    buffer whatever the block holds: the chunk decoder's reach past a
    block's start is at most the widest block's word count
    (radix.block_word_bounds(B)[1] — the literal plane, each tier with
    every lane escaping, a full terminal; sip 608, gulp 768 words at
    B = 1024), the directory is bounded by radix.validate, and the kernels
    carry no per-load bound (the Triton reader measured a mask at +2%;
    codec/radix_gpu._read). A block short of its data-dependent streams
    reads its neighbour's words or the pad — garbage, inside the buffer.
    tests/test_radix_bounds.py models the chunk decoder's addresses."""
    return max(PAD_WORDS, int(_radix.block_word_bounds(B, 7, 8, widths)[1]))


def refuse_unsupported(pack: dict) -> None:
    """The lines the two kernels draw, by name: bf16 radix only (the
    terminal tier is the 8-bit exponent), 2..4 tiers, tier widths 1..4, a
    power-of-two block of 32..1024 lanes, and a tensor big enough to bind."""
    if not _rp.is_radix_pack(pack):
        raise ValueError(f"not a radix pack tensor (codec {pack.get('codec')!r}); "
                         "this module serves the radix codec only")
    widths = _widths(pack)
    if len(widths) < 2 or len(widths) > MAX_TIERS:
        raise ValueError(f"radix widths {widths}: the Metal kernels serve 2..{MAX_TIERS} tiers")
    if widths[-1] != 8:
        raise ValueError(f"radix widths {widths}: the terminal tier must be the 8-bit bf16 "
                         "exponent (bf16 only)")
    if any(w < 1 or w > 4 for w in widths[:-1]):
        raise ValueError(f"radix widths {widths}: nonterminal tiers of 1..4 bits only")
    B = int(pack["block_size"])
    if B < LPT or B > MAX_BLOCK or B & (B - 1):
        raise ValueError(f"radix block_size {B}: the Metal kernels serve a power of two "
                         f"from {LPT} to {MAX_BLOCK}")
    if int(pack["R"]) < 1 or int(pack["C"]) < 1:
        raise ValueError("radix tensor with an empty dimension")


def resident(pack: dict) -> dict:
    """A radix tensor dict (iter_pack_dir's: rx_palette / rx_offsets /
    rx_data + scalars) -> the arrays the two kernels read, as mx arrays in
    the kernels' dtypes, built ONCE — what the MLX engine's RadixLinear
    holds for the life of the model:

      data      uint32  the block streams verbatim, + pad_words(B, widths)
                        zero words (the chunk decoder's reach bound)
      offsets   uint32  the word directory verbatim
      palette   uint32  the per-tier lookup tables, each tier's entries
                        padded to whole words (radix_pack.padded_palette_words,
                        the Triton arm's construction), zero-padded to bind
      params    uint32  [R, C, B, NB, 0, 0, 0, 0]
      bias      None    (the engine attaches one as float32 [R] when the
                        checkpoint carries it)

    plus the scalars and the unpadded palette bytes (numpy) for the CPU
    reference decoder. An already-resident dict round-trips through itself."""
    if "data" in pack and "params" in pack:
        return pack
    refuse_unsupported(pack)
    R, C, B = int(pack["R"]), int(pack["C"]), int(pack["block_size"])
    NB = -(-C // B)
    widths = _widths(pack)
    data = np.ascontiguousarray(pack["rx_data"], dtype=np.uint32).reshape(-1)
    offsets = np.ascontiguousarray(pack["rx_offsets"], dtype=np.uint32).reshape(-1)
    if offsets.size != R * NB + 1 or int(offsets[-1]) != data.size:
        raise ValueError(f"radix directory does not match the streams: {offsets.size} entries "
                         f"for {R * NB} blocks, last {int(offsets[-1]) if offsets.size else '-'} "
                         f"vs {data.size} words")
    padded = np.concatenate([data, np.zeros(pad_words(B, widths), dtype=np.uint32)])
    return {
        "R": R, "C": C, "B": B, "NB": NB, "NT": len(widths), "widths": widths,
        "TPB": B // LPT, "NW": int(data.size),
        "data": mx.array(padded),
        # the directory has R*NB + 1 entries; a tiny tensor's (< 8) is
        # zero-padded so mlx binds it as a device pointer (_bound's rule)
        "offsets": _bound(mx.array(offsets)),
        "palette": _bound(mx.array(_rp.padded_palette_words(pack))),
        "params": mx.array(np.array([R, C, B, NB, 0, 0, 0, 0], dtype=np.uint32)),
        "bias": None,
        "rx_palette_np": _rp._palette_u8(pack),
        "codec": "radix", "format_version": pack.get("format_version", FORMAT_VERSION), "layout": 0,
        "profile": pack.get("profile"),
        "bpw": pack.get("bpw"),
    }


def resident_bytes(res: dict) -> int:
    """What one GEMV streams from the device per call: the streams, the
    directory and the lookup tables (the pad and the params are not
    weights, but they are resident; count them — a few dozen bytes)."""
    return sum(int(res[k].nbytes) for k in ("data", "offsets", "palette", "params"))


def pairs_tails(res: dict) -> bool:
    """Whether the GEMV runs two rows per SIMD group (_GEMV's PAIR): 1024-weight
    blocks and a last block of at most 16 chunks, which alone would idle half
    the SIMD group for a step. Same output bits either way; PAIR_TAILS = False
    turns it off."""
    tail = res["C"] - (res["NB"] - 1) * res["B"]
    return PAIR_TAILS and res["B"] == MAX_BLOCK and 0 < tail <= MAX_BLOCK // 2


def _template(res: dict, *, has_bias: bool, xbf: bool, out32: bool, spr: int, rpt: int,
              pair: bool = False) -> list:
    w = list(res["widths"]) + [0, 0, 0]
    return [("NT", int(res["NT"])), ("W0", int(w[0])), ("W1", int(w[1])), ("W2", int(w[2])),
            ("TPB", int(res["TPB"])), ("HAS_BIAS", bool(has_bias)), ("XBF", bool(xbf)),
            ("OUT32", bool(out32)), ("SPR", int(spr)), ("RPT", int(rpt)), ("PAIR", bool(pair))]


def gemv_radix_resident(res: dict, x: mx.array, bias=None, out_dtype=mx.bfloat16,
                        rows_per_tg: int = ROWS_PER_TG, simd_per_row: int = SIMD_PER_ROW) -> mx.array:
    """y = W @ x (+ bias) from a resident() dict: `x` an mx vector of length
    C — bf16 (read as bits, widened in-kernel) or float32 — the result an
    UNEVALUATED mx vector of length R in `out_dtype` (bf16: fp32 accumulate,
    bias in fp32, ONE rounding; float32: the fp32 sum, no rounding — the
    gate's view of the accumulator). `bias` overrides res["bias"] (an
    mx vector of length R, any float dtype; widened to fp32 once here)."""
    R, C = res["R"], res["C"]
    if x.ndim != 1 or x.shape[0] != C:
        raise ValueError(f"radix gemv: x has shape {x.shape}, want ({C},)")
    if x.dtype == mx.bfloat16:
        xs, xbf = x.view(mx.uint16), True
    elif x.dtype == mx.float32:
        xs, xbf = x, False
    else:
        raise ValueError(f"radix gemv: x must be bf16 or float32, not {x.dtype}")
    if out_dtype not in (mx.bfloat16, mx.float32):
        raise ValueError(f"radix gemv: out_dtype must be bf16 or float32, not {out_dtype}")
    b = res["bias"] if bias is None else bias
    if b is not None:
        if b.shape != (R,):
            raise ValueError(f"radix gemv: bias has shape {b.shape}, want ({R},)")
        b = b.astype(mx.float32) if b.dtype != mx.float32 else b
    has_bias = b is not None
    out32 = out_dtype == mx.float32
    spr, rpt = int(simd_per_row), int(rows_per_tg)
    if spr not in (1, 2, 4, 8) or rpt < 1 or spr * rpt > 32:
        raise ValueError(f"radix gemv: simd_per_row {spr} x rows_per_tg {rpt} is not a threadgroup")
    pair = spr == 1 and pairs_tails(res)
    units = -(-R // 2) if pair else R                   # SIMD groups: rows, or row pairs
    tg = SIMD * spr * rpt
    grid = -(-units // rpt) * tg
    (y,) = _gemv_kernel(
        inputs=[res["data"], res["offsets"], res["palette"], res["params"], _bound(xs),
                b if has_bias else res["params"]],
        template=_template(res, has_bias=has_bias, xbf=xbf, out32=out32, spr=spr, rpt=rpt, pair=pair),
        output_shapes=[(R,)],
        output_dtypes=[mx.float32 if out32 else mx.uint16],
        grid=(grid, 1, 1), threadgroup=(tg, 1, 1),
    )
    return y if out32 else y.view(mx.bfloat16)


def gemv_radix_step(res: dict, x: mx.array, out_shape: tuple) -> mx.array:
    """The decode step's call (engine_mlx.RadixLinear: fused, M = 1, bf16, no
    bias): `x` a contiguous bf16 array holding one row of C, whatever its
    leading ones, handed to the kernel as is (the kernel reads it as bits);
    the result bf16 in `out_shape`. ONE graph node per Linear, where
    gemv_radix_resident's general path wraps the kernel in a reshape, a
    slice and two views; the template, grid and inputs are built once per
    tensor and kept on `res`. The same kernel over the same bits as
    gemv_radix_resident(res, x.reshape(-1)), so the same output bits."""
    key = (ROWS_PER_TG, pairs_tails(res))
    k = res.get("_step")
    if k is None or k[0] != key:
        rpt, pair = key
        tg = SIMD * rpt
        units = -(-res["R"] // 2) if pair else res["R"]
        k = res["_step"] = (key, [res["data"], res["offsets"], res["palette"], res["params"]],
                            _template(res, has_bias=False, xbf=True, out32=False, spr=1, rpt=rpt, pair=pair),
                            (-(-units // rpt) * tg, 1, 1), (tg, 1, 1))
    _, ins, template, grid, tg3 = k
    (y,) = _gemv_kernel(inputs=[*ins, x, res["params"]], template=template,
                        output_shapes=[out_shape], output_dtypes=[mx.bfloat16],
                        grid=grid, threadgroup=tg3)
    return y


def decode_reference(res: dict) -> np.ndarray:
    """The CPU oracle over a resident dict's own bytes: the uint16 [R, C]
    bf16 patterns through radix_pack.decode_back_radix (the native decoder
    when a C++ compiler is at hand, else radix.decode). What the MLX
    engine's reference path multiplies by, and what the gates pin the
    kernels against on a real pack. Reads the streams back off the device
    (a copy; slow by design — it is the oracle, not a path). The native
    decoder speaks 1024-weight blocks only; any other block size (the
    gates' 32) goes through the research numpy decoder."""
    p = {"rx_palette": res["rx_palette_np"],
         "rx_offsets": np.array(res["offsets"])[: res["R"] * res["NB"] + 1].astype(np.uint32),
         "rx_data": np.array(res["data"])[: res["NW"]].astype(np.uint32),
         "R": res["R"], "C": res["C"], "widths": list(res["widths"]),
         "block_size": res["B"], "codec": "radix"}
    return _rp.decode_back_radix(p, encoder=None if res["B"] == _rp.RADIX_BLOCK else "numpy")
