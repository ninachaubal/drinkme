"""Triton-free constants the codec's Python side reads on any machine:
MC_MAX, the multi-column kernel's column cap. The radix codec's ops are
codec/radix_ops.py.
"""

from __future__ import annotations

import os

REFERENCE = os.environ.get("DRINKME_REFERENCE") == "1"

# The radix multi-column kernel is written out to eight named accumulators
# (one per column, so each reduction is the M=1 kernel's own 1-D `tl.sum` —
# radix_ops._gemv_mc's docstring says why). Eight is therefore a STRUCTURAL
# cap, not a tuning knob: a ninth column would be silently dropped, so the
# kernel (`tl.static_assert`) and the ops refuse M > MC_MAX rather than
# compute garbage. Lives here because ops.py imports no triton and so is
# importable on any machine — swap.py's router reads it without dragging triton in.
MC_MAX = 8
