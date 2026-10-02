"""The serving codec: pack (compress once), decode (fused kernels), swap
(resident-compressed Linears).

  pack.py         — the on-disk pack container (docs/pack-format.md): the
                    transactional writer, the file hashes and manifest, the loaders
                    and the format gate; numpy-pure, no torch/triton at import
  radix_pack.py   — the bf16 codec's pack tensors: the profiles, the encoder
                    (radix_native.py's C++ path or radix.py's numpy path),
                    the bit-exact round trip, the raw fallback
  radix_ops.py    — the served kernels over a runtime dict: the M=1 GEMV,
                    the multi-column arm, the dense decode (triton, bound
                    lazily by swap.py); radix_kernel_gpu.py (the scheduled
                    decoder, the launched kernels) and radix_gpu.py (the
                    block decoder) hold the jit functions; radix_schedule.py
                    the launch table; RAW=True runs the gemv and mc kernels
                    over a raw bf16 weight for the bench's order-matched twin;
                    gemv_narrow is the narrow raw Linears' bf16 GEMV
  decode.py       — the e4m3 decode helper radix_gpu.py imports
  ops.py          — MC_MAX, importable without triton
  swap.py         — module surgery: nn.Linear -> RadixCompressedLinear /
                    RawLinear, or the order-matched twin; the device is a
                    parameter, not an assumption
  registry.py     — which (tensor kind, op, kernel) cells are implemented
  identity.py     — the checkpoint identity a pack binds to; torch-free

The Metal port of the same kernels is `drinkme.metal` (docs/metal.md).
"""
