"""The e4m3 bit-layout decode radix_gpu._weight's E == 4 branch imports
(the radix codec's own dtype axis, constexpr-dead on the served bf16 path
where E == 8). Kept for that import alone: no fp8 tensor is packed, loaded
or served. The narrow raw bf16 Linears' GEMV is radix_ops.gemv_narrow.

Triton (CUDA + ROCm): this module needs triton at import (the @triton.jit
decorator), so import it only on a GPU machine.
"""

from __future__ import annotations

import triton
import triton.language as tl


@triton.jit
def _e4m3_to_f32(b):
    """int32 e4m3fn code (0..255) -> fp32 value, by the bit layout: sign(1)
    exp(4, bias 7) man(3). exp == 0 is subnormal (man * 2^-9); the format
    has no inf and its NaNs (0x7F/0xFF) are the caller's to mask. A normal
    value is rebuilt directly as fp32 bits: exponent rebias +120 (127 - 7),
    mantissa left-aligned by 20. Imported by radix_gpu._weight's E == 4
    branch (the module docstring says why it stays)."""
    sign = b >> 7
    exp = (b >> 3) & 0xF
    man = b & 0x7
    normal = ((sign << 31) | ((exp + 120) << 23) | (man << 20)).to(tl.float32, bitcast=True)
    # subnormal: man * 2^-9 (exact), sign injected into the BITS — `0 - x`
    # would turn code 0x80 (negative zero) into +0.0, and the decoded bytes
    # must be the bit layout's (0x8000), not merely equal as numbers
    sub_mag = (man.to(tl.float32) * 0.001953125).to(tl.int32, bitcast=True)
    sub = ((sign << 31) | sub_mag).to(tl.float32, bitcast=True)
    return tl.where(exp == 0, sub, normal)
