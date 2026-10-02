"""The multi-column GEMV arm: the parts that hold WITHOUT a GPU.

The numerics claim — every column within the float64 oracle's bound of the
M=1 kernel, on real pack tensors, M in 2..8 — belongs to
`bench/radix_mc_bitpin.py`, because it needs triton, a GPU and a pack. What
belongs here is the structure around it: the column cap the kernels are
written out to, the router's contract with that cap, and the fact that the
gate itself can still go red.

Why the cap gets its own test at all: radix_ops._gemv_mc holds ONE NAMED
ACCUMULATOR PER COLUMN (so each reduction is the M=1 kernel's own 1-D
`tl.sum` — see the kernel's docstring), which means a ninth column would
not be an error, it would be silently missing from the output. `ops.MC_MAX`
is the number the router, the ops and the kernel all have to agree on, and
agreement between a constant and eight lines of unrolled triton is exactly
the thing that rots quietly.
"""

import os
import subprocess
import sys

import pytest
import torch

from drinkme.codec.ops import MC_MAX
from drinkme.codec.pack import FORMAT_VERSION, default_pack_dir

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RADIX_OPS_SRC = os.path.join(REPO, "src", "drinkme", "codec", "radix_ops.py")
# the default pack directory of the pinned Qwen3-8B (`drinkme pack` with no -o)
PACK_8B = default_pack_dir("Qwen/Qwen3-8B", "b968826d9c46dd6066d109eabc6255188de91218")


# ------------------------------------------------------------- the cap ----


def test_the_mc_op_refuses_more_columns_than_the_kernel_has():
    """A ninth column would be dropped, not diagnosed — so the op refuses it
    (radix_ops.gemv_mc; triton at import, so read as text here — bench/
    radix_mc_bitpin.py's box runs it)."""
    src = open(RADIX_OPS_SRC).read()
    assert f'raise ValueError(f"radix gemv_mc: M={{M}} outside 1..{{MC_MAX}}")' in src


@pytest.mark.parametrize("src_path,start,end,acc", [
    (RADIX_OPS_SRC, "def _gemv_mc(", "def _args(", "tl.zeros((B,), tl.float32)"),
])
def test_kernels_are_written_out_to_exactly_mc_max_columns(src_path, start, end, acc):
    """MC_MAX vs the unrolled source, read as text: the kernel needs one
    accumulator, one reduce and one store per column, and the guards must
    cover 1..MC_MAX-1 with nothing above. Text, not import, so the check runs
    on a box with no triton."""
    src = open(src_path).read()
    body = src[src.index(start):src.index(end)]
    for n in range(MC_MAX):
        assert f"a{n} = {acc}" in body, n
        assert f"tl.sum(a{n})" in body, n
    assert f"a{MC_MAX} " not in body  # no accumulator past the cap
    guards = sorted({int(line.split("MC > ")[1].split(":")[0])
                     for line in body.splitlines() if "if MC > " in line})
    assert guards == list(range(1, MC_MAX)), guards
    assert f'tl.static_assert(MC <= {MC_MAX}' in body


def test_router_never_hands_the_kernel_more_than_it_holds():
    """The one invariant that turns a silent wrong answer into a slow one."""
    from drinkme.codec.swap import CompressedLinear

    route = CompressedLinear._route_rows  # reads no instance state
    for m in list(range(1, MC_MAX + 4)) + [64, 4096]:
        assert route(None, m) != "mc" or m <= MC_MAX, m


# ----------------------------------------------------- the gate's own gate --



def _have_pack(path: str) -> bool:
    """A pack this build reads at `path`: meta.json present and of this
    format (a pack of another version would be refused, not gated)."""
    try:
        import json

        with open(os.path.join(path, "meta.json")) as f:
            return json.load(f).get("formatVersion") == FORMAT_VERSION
    except (OSError, ValueError):
        return False


_HAVE_GPU = torch.cuda.is_available()
_HAVE_PACK = _have_pack(PACK_8B)
gpu_gate = pytest.mark.skipif(
    not (_HAVE_GPU and _HAVE_PACK),
    reason=f"needs a GPU and a Qwen3-8B pack of format {FORMAT_VERSION} at {PACK_8B} "
           "(bench/radix_mc_bitpin.py owns this claim)")


def _bitpin(*extra):
    return subprocess.run(
        [sys.executable, os.path.join(REPO, "bench", "radix_mc_bitpin.py"),
         "--pack-dir", PACK_8B, "--limit", "1", *extra],
        capture_output=True, text=True, cwd=REPO,
        env={**os.environ, "PYTHONPATH": os.path.join(REPO, "src")})


@gpu_gate
def test_bitpin_gate_passes_on_the_real_pack():
    r = _bitpin()
    assert r.returncode == 0, r.stdout + r.stderr
    assert "PASS:" in r.stdout


@gpu_gate
def test_bitpin_gate_can_actually_fail():
    """An unfailable gate is worse than no gate. --selftest corrupts one
    column on purpose; the run must report FAIL for the right reason."""
    r = _bitpin("--selftest")
    assert "SELFTEST PASSED" in r.stdout, r.stdout + r.stderr
    assert "FAIL:" in r.stdout
