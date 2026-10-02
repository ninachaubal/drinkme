"""Radix DECODE-TO-WEIGHT on Metal: the pack's streams -> the bf16 tensor,
as ONE `mx.fast.metal_kernel` dispatch. The dense arm (M > 1).

WHY IT EXISTS. gemv_radix is y = W @ x at M = 1, W never built — the decode
step. Prefill is M rows, and the MLX engine's shape for that (the torch
side's radix_ops.decode_weight) is a transient bf16 weight — one Linear at a time, 2 B/weight, dropped by the
graph as soon as mx.matmul has read it — and this is the radix codec's
instance of it: radix_kernel_gpu._dense folded into one dispatch.

THE CLAIM. The transient is BITWISE the CPU oracle's decode
(radix_pack.decode_back_radix -> codec/radix.decode) — the codec is
lossless, so bitwise the original bf16 tensor — pinned by
bench/radix_bitpin_mlx.py on all 65,536 bf16 patterns and on real
Qwen3-8B tensors (decoded bits == the safetensors bits, by digest). The
M > 1 forward is then mlx's own bf16 matmul over that weight: the same
expression (`x @ W.T`) the reference path runs over the oracle's bytes.

THE KERNEL. gemv_radix's chunk decoder (rx_decode_chunk: one SIMD group
per 1024-lane block, 32 consecutive lanes per thread, the escape rank by
popcount + SIMD prefix sum, the terminal gather by rank) with the store in
place of the multiply: each thread writes its 32 bf16 bit patterns (64
contiguous bytes) to out[r, col0 .. col0 + 32), masked on a ragged block.
Every element is stored exactly once, by one thread: DETERMINISTIC by
construction, and the matmul that consumes it is mlx's.

MEMORY. The transient is R*C*2 bytes for ONE Linear at a time (the 8B's
12288x4096 is 100.7 MB; its untied lm_head 1.24 GB — the reason it is per
call and never retained). The resident streams are ~1.45 B/weight at
profile sip and are shared with gemv_radix — one resident dict, two
kernels over it.

Needs Metal. `import drinkme.metal` imports none of this; import this
module explicitly.
"""

from __future__ import annotations

import mlx.core as mx

from . import gemv_radix

SIMD = gemv_radix.SIMD
TG = 256  # 8 SIMD groups per threadgroup

# params = [R, C, B, NB, ...] — gemv_radix.resident's own params array.
# Global SIMD group g decodes blocks [g*BPS, g*BPS + BPS) of the flattened
# (row, block) index; a thread past the last block idles through the SIMD
# scans with n = 0 and stores nothing.
_SOURCE = r"""
    const uint R = params[0];
    const uint C = params[1];
    const uint B = params[2];
    const uint NB = params[3];
    const uint BPS = 32u / (uint)TPB;
    ushort lane = thread_index_in_simdgroup;
    uint g = thread_position_in_grid.x / 32u;
    uint slot = (uint)lane / (uint)TPB;
    uint sub = (uint)lane % (uint)TPB;
    uint total = R * NB;
    uint gb = g * BPS + slot;
    bool valid = gb < total;
    uint gbb = valid ? gb : total - 1u;
    uint r = gbb / NB;
    uint b = gbb - r * NB;
    uint n = valid ? min(B, C - b * B) : 0u;
    uint lw[8];
    uint ex[32];
    uint amask;
    rx_decode_chunk<NT, W0, W1, W2, TPB>(data, offsets, palette, gbb, n, sub, lane, lw, ex, amask);
    if (amask == 0u) { return; }
    uint col0 = b * B + sub * LPT_;
    device uint16_t* o = out + (ulong)r * (ulong)C + col0;
    if (amask == 0xFFFFFFFFu) {
        // 32 patterns as 16 words (the row offset is 2*C*r bytes: word-aligned when C is even)
        if ((C & 1u) == 0u) {
            device uint* ow = (device uint*)o;
            #pragma unroll
            for (uint i = 0u; i < 32u; i += 2u) {
                ow[i >> 1] = RX_BITS_(lw, ex, i) | (RX_BITS_(lw, ex, i + 1u) << 16);
            }
        } else {
            #pragma unroll
            for (uint i = 0u; i < 32u; ++i) { o[i] = (uint16_t)RX_BITS_(lw, ex, i); }
        }
    } else {
        #pragma unroll
        for (uint i = 0u; i < 32u; ++i) {
            if ((amask >> i) & 1u) { o[i] = (uint16_t)RX_BITS_(lw, ex, i); }
        }
    }
"""

_kernel = mx.fast.metal_kernel(
    name="drinkme_radix_decode_dense",
    input_names=["data", "offsets", "palette", "params"],
    output_names=["out"],
    header=gemv_radix._HEADER,
    source=_SOURCE,
)


def _template(res: dict) -> list:
    w = list(res["widths"]) + [0, 0, 0]
    return [("NT", int(res["NT"])), ("W0", int(w[0])), ("W1", int(w[1])), ("W2", int(w[2])),
            ("TPB", int(res["TPB"]))]


def decode_dense_resident(res: dict) -> mx.array:
    """The bf16 bit patterns [R, C] as uint16, UNEVALUATED — the dense arm's
    transient weight (view it as bf16 and matmul; let it go). One dispatch
    of ceil(R*NB / blocks-per-SIMD-group) SIMD groups."""
    R, C, NB = res["R"], res["C"], res["NB"]
    bps = SIMD // int(res["TPB"])
    groups = -(-(R * NB) // bps)
    grid = -(-groups // (TG // SIMD)) * TG
    (out,) = _kernel(
        inputs=[res["data"], res["offsets"], res["palette"], res["params"]],
        template=_template(res),
        output_shapes=[(R, C)], output_dtypes=[mx.uint16],
        grid=(grid, 1, 1), threadgroup=(TG, 1, 1),
    )
    return out


def weight_bf16_resident(res: dict) -> mx.array:
    """decode_dense_resident viewed as bf16 — what `x @ W.T` takes."""
    return decode_dense_resident(res).view(mx.bfloat16)
