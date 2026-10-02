"""drinkme on Apple silicon: the Metal kernels.

The radix codec's kernels (gated bit-exact on an M4 by
bench/radix_bitpin_mlx.py):

  gemv_radix.py  — the resident dict off a radix pack tensor (streams,
                   directory, palette) and the fused decode+GEMV at M=1:
                   one SIMD group per row, 32 consecutive weights per
                   thread, the escape rank by popcount + SIMD prefix sum,
                   bias fused, bf16 out. Needs Metal.
  dense_radix.py — the same chunk decoder storing instead of multiplying:
                   the bf16 transient for M>1, bitwise the CPU oracle.
                   Needs Metal.
  twin_radix.py  — the radix decode in generic mlx ops, vectorised over a
                   tensor's blocks and sharing no code with the kernels
                   above: the bench's twin decodes each tensor with it once
                   at load. Bitwise the CPU oracle; runs without Metal too.

The raw fallback (`codec: "raw"`) needs no kernel: serving/engine_mlx
.RawLinear runs the bf16 bits through mlx's own matmul.

NOTHING here imports mlx at package import: `import drinkme` and every
torch-side path must keep working on a machine that has never heard of mlx (the
Linux dev machines, the CUDA/ROCm lanes). Import the submodule explicitly.
docs/metal.md carries the port's numbers, the thermal caveat and the
hardware verification steps.
"""
