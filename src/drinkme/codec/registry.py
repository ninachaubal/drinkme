"""THE CAPABILITY REGISTRY: which (pack tensor kind, operation, kernel)
combinations are implemented, in ONE table — consulted BEFORE a path is built,
so an unsupported combination is a decision its caller makes (fall back,
refuse by name) rather than a KeyError three frames into a constructor.

The case that shapes it: the reduced-vocab draft projection
(serving/draft_vocab.make_projection) needs a row-subset kernel that not
every tensor kind has. Were installation to build it blind, a head whose
tensor has no such kernel would die in the constructor — DRINKME_MTP_DRAFT_VOCAB's
documented "falls back to the full vocabulary" would never get a chance,
an explicit DRINKME_MTP_DEPTH=4 would propagate the error out of the loader, and
AUTO would drop the whole head. So installation asks this table first.

THE KEY. (tensor_codec, layout, op, kernel):

    tensor_codec
             the tensor's kind, as CompressedLinear.tensor_codec / of_pack name it:
             "radix"  the radix codec (codec/radix_pack.py) — every bf16
                      tensor of a pack
             "raw"    the raw fallback (bf16 bits verbatim, swap.RawLinear)
             The pack format VERSION is not a key: a pack of another version
             is refused at load (codec/pack._check_format_version) and its
             tensors never reach this table.
    layout   always 0 (no code-plane layouts); kept in the key so a future
             intra-block variant has somewhere to go
    op       DENSE       decode the whole bf16 weight (radix_ops.decode_weight
                         / the CPU reference decoder) — the M >= GEMM_MIN_ROWS
                         prefill arm, and every bit gate
             GEMV        the M=1 kernel (radix_ops.gemv_fused)
             MC          the multi-column kernel, 2 <= M <= ops.MC_MAX
             ROW_SUBSET  the draft-vocab row selection (no codec has one;
                         the draft keeps the full head)
    kernel   the kernel family that would run it — not the torch/MLX
             RUNTIME and not the CUDA/ROCm/Metal hardware family, though
             on Metal the three coincide:
             TRITON      the torch accelerator arm (CUDA and ROCm both report
                         as torch's "cuda"): codec/radix_kernel_gpu.py and
                         radix_ops.py
             CPU         the torch CPU reference route: F.linear over the
                         decoded weight. GEMV / MC / ROW_SUBSET are not
                         separate paths there — every M is DENSE — so the
                         table says False for them rather than leaving the
                         key out, because "there is no such kernel on CPU"
                         is a known fact, not an unknown one
             METAL       the MLX arm (drinkme.metal): the radix tensor (metal/
                         gemv_radix's M=1 GEMV and metal/dense_radix's decode-
                         to-weight dispatch whose transient bf16 weight is the
                         M>1 dense arm — gated on the M4 by bench/radix_bitpin_
                         mlx.py) and the raw fallback (engine_mlx
                         .RawLinear: mlx's own bf16 matmul, DENSE only, as on
                         every kernel), nothing else

A key NOT in the table is unsupported. That is the conservative direction: a
new codec or kernel earns its True by having its gate run
(bench/radix_gemv_bitpin.py, bench/radix_mc_bitpin.py,
bench/radix_schedule_bitpin.py, bench/radix_bitpin_mlx.py),
not by being assumed — and the False rows are written out so that a reader
can tell "decided against" from "nobody has looked".
"""

from __future__ import annotations

DENSE, GEMV, MC, ROW_SUBSET = "dense", "gemv", "mc", "row_subset"
OPS = (DENSE, GEMV, MC, ROW_SUBSET)

TRITON, CPU, METAL = "triton", "cpu", "metal"
KERNELS = (TRITON, CPU, METAL)

# the radix codec (codec/radix_pack.py): `codec: "radix"` in the tensor's
# scalars. No code-plane layouts.
RADIX = "radix"
# the raw fallback (codec/radix_pack.py): a tensor radix would have expanded,
# stored as its bf16 bits — served by stock's own F.linear / matmul on every
# kernel, so it has no kernel rows of its own to earn
RAW = "raw"

# (tensor_codec, layout) pairs that exist: none has a code plane, so every layout is 0.
FORMATS = ((RADIX, 0), (RAW, 0))

# tensor_codec  layout  op  kernel  supported
TABLE: dict[tuple[str, int, str, str], bool] = {
    # ---- radix (exponent tiers, literal sign/mantissa, 1024-weight blocks;
    #      codec/radix_pack.py + radix_ops.py — THE bf16 codec) -------------
    # Every row starts False and is flipped ONLY by its gate's PASS: the CPU
    # dense row by tests/test_radix_pack.py, the three TRITON rows by
    # bench/radix_gemv_bitpin.py + bench/radix_mc_bitpin.py +
    # bench/radix_schedule_bitpin.py (every launch-table row).
    (RADIX, 0, DENSE, TRITON): True,   # radix_ops.decode_weight — bench/radix_gemv_bitpin.py
    #                                    PASS: all
    #                                    65,536 bf16 patterns, every profile,
    #                                    32/1024 blocks, dense-exact at warps 1/2/4
    (RADIX, 0, GEMV, TRITON): True,    # radix_ops.gemv_fused — same gates: float64 bound, fused
    #                                    epilogue == unfused + bias rounded once, at every
    #                                    schedule of the grid (radix_schedule_bitpin.py)
    (RADIX, 0, MC, TRITON): True,      # radix_ops.gemv_mc — bench/radix_mc_bitpin.py PASS:
    #                                    210 synthetic (tensor, M)
    #                                    cases + the real packs within the oracle bound
    (RADIX, 0, ROW_SUBSET, TRITON): False,  # no draft-vocab subset kernel: the
    #                                         draft keeps the full head
    (RADIX, 0, DENSE, CPU): True,   # tests/test_radix_pack.py: toy pack -> serve loader on
    #                                 the CPU, logits bitwise stock; every tensor decodes
    #                                 bit-exact through radix_pack.decode_back_radix
    (RADIX, 0, GEMV, CPU): False,
    (RADIX, 0, MC, CPU): False,
    (RADIX, 0, ROW_SUBSET, CPU): False,
    (RADIX, 0, DENSE, METAL): True,   # metal.dense_radix — bench/radix_bitpin_mlx.py PASS
    #                                   on an M4: all 65,536 bf16
    #                                   patterns, sip/balanced/gulp, 32/1024 blocks, eight
    #                                   real Qwen3-8B tensors bitwise the safetensors digests
    (RADIX, 0, GEMV, METAL): True,    # metal.gemv_radix (engine_mlx.RadixLinear, M = 1) — same
    #                                   gate: float64 bound, bf16-x == fp32-x, fused epilogue
    (RADIX, 0, MC, METAL): False,     # no multi-column kernel: 2 <= M is the dense arm (the
    #                                   transient decode, not a row loop), so nothing is
    #                                   lost by its absence
    (RADIX, 0, ROW_SUBSET, METAL): False,
    # ---- raw (the fallback: bf16 bits verbatim, swap.RawLinear on torch,
    #      engine_mlx.RawLinear on Metal) --------------------------------------
    # DENSE is the only op: every M is F.linear / mx.matmul over the resident
    # bf16 weight — stock's own arithmetic, nothing to gate.
    (RAW, 0, DENSE, TRITON): True,
    (RAW, 0, GEMV, TRITON): False,
    (RAW, 0, MC, TRITON): False,
    (RAW, 0, ROW_SUBSET, TRITON): False,
    (RAW, 0, DENSE, CPU): True,
    (RAW, 0, GEMV, CPU): False,
    (RAW, 0, MC, CPU): False,
    (RAW, 0, ROW_SUBSET, CPU): False,
    (RAW, 0, DENSE, METAL): True,   # engine_mlx.RawLinear: the bf16 bits through mx.matmul
    (RAW, 0, GEMV, METAL): False,
    (RAW, 0, MC, METAL): False,
    (RAW, 0, ROW_SUBSET, METAL): False,
}

_OP_NAMES = {DENSE: "dense decode", GEMV: "M=1 GEMV", MC: "multi-column GEMV",
             ROW_SUBSET: "row-subset (draft-vocab) projection"}


# the tensor_codec of a pack dict that names no kind (of_pack): a known-absent key,
# never a match — describe() names it and no row exists for it
UNKNOWN = "unknown"


def _codec_key(tensor_codec):
    """"radix" / "raw" as they are; anything else is UNKNOWN — a
    known-absent key, never a KeyError and never a match."""
    return tensor_codec if tensor_codec in (RADIX, RAW) else UNKNOWN


def supported(tensor_codec, layout: int, op: str, kernel: str) -> bool:
    """The one question. Unknown keys are False, never KeyError — a caller
    holding a pack this table has never heard of gets "no", which is the
    answer that cannot serve wrong bytes."""
    return TABLE.get((_codec_key(tensor_codec), int(layout), op, kernel), False)


def describe(tensor_codec, layout: int) -> str:
    if tensor_codec == RADIX:
        return "radix (exponent tiers, literal sign/mantissa, 1024-weight blocks)"
    if tensor_codec == RAW:
        return "raw (bf16 bits verbatim — the radix fallback)"
    return f"unknown tensor kind {tensor_codec!r} layout-{int(layout)} (no codec named)"


def unsupported_note(tensor_codec, layout: int, op: str, kernel: str) -> str:
    """The log-line half of a refusal: what was asked, of what, and where.
    Pure string — the caller decides whether it is a warning or a raise."""
    known = (_codec_key(tensor_codec), int(layout), op, kernel) in TABLE
    return (f"the {_OP_NAMES.get(op, op)} has no {kernel} kernel for a "
            f"{describe(tensor_codec, layout)} pack tensor "
            f"({'unsupported' if known else 'unknown combination'} in "
            "codec/registry.py)")


def kernel_of(device) -> str:
    """torch device (or its string) -> the kernel family that serves it.
    ROCm's HIP reports as torch's "cuda", so this is one test, not a vendor
    list; anything that is not the CPU is the triton arm — the MLX arm never
    holds a torch device and names METAL itself."""
    import torch

    return CPU if torch.device(device).type == "cpu" else TRITON


def of_pack(p: dict) -> tuple:
    """(tensor_codec, layout) off a pack dict — raw (numpy) or runtime (to_device's),
    both of which carry the discriminator the same way: the `codec`
    scalar (radix | raw). A dict without one (a `dtype` scalar, say — an
    FP8 pack's tensors) names no kind this table has a row for: UNKNOWN
    comes back so the note names it and no row matches (the loaders refuse
    such a tensor by name before it gets here — codec/pack._arrays_for)."""
    if p.get("codec") == RADIX:
        return RADIX, 0
    if p.get("codec") == RAW:
        return RAW, 0
    return UNKNOWN, int(p.get("layout", 0))


def refuse_unless_metal_dense(pack: dict, name: str) -> None:
    """The MLX/Metal loader's callsite (serving/engine_mlx.py): refuse a
    pack tensor BY NAME before it is handed to code that assumes one
    module's fields. The table's (tensor_codec, layout, DENSE, METAL) row is the
    question — the reference path IS the dense decode, so a tensor with no
    dense row on Metal has no path at all there. Today that passes the
    radix tensor (RadixLinear) and the raw fallback (RawLinear) and refuses
    a dict that names no kind with the registry's own note, so the refusal and the table cannot
    disagree; the loader's make_module_mlx then dispatches on the same
    answer."""
    tensor_codec, layout = of_pack(pack)
    if not supported(tensor_codec, layout, DENSE, METAL):
        raise ValueError(
            f"no MLX path for {name} ({describe(tensor_codec, layout)}) — "
            f"{unsupported_note(tensor_codec, layout, DENSE, METAL)}; serve this pack on "
            "the torch runtime")
