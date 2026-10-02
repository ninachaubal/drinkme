"""The post-install self-test, run as a CHILD PROCESS by bootstrap.selftest.

    python -m drinkme.bootstrap_selftest --lane therock --gfx gfx1151 --workdir DIR

Seven rungs, in order, one line each, and a verdict line at the end. The
order is the order a broken lane fails in: the official PyTorch ROCm wheel
on a Strix Halo (gfx1151) imports, sees the device, allocates — and dies
with SIGSEGV at the FIRST KERNEL LAUNCH (rung 4), every time (measured
on gfx1151, kernel 7.0.0-31; docs/rocm.md). That is
why this runs in a subprocess: a segfault here is a verdict the parent
reads off the exit status (bootstrap.selftest: `lane_cannot_launch`), not
a crash of the installer.

  1 import          `import torch` — the wheel is installed and loads
  2 device          torch.cuda.is_available() (the API is torch.cuda on
                    ROCm too) — the driver, /dev/kfd and the runtime agree
  3 arch-list       torch.cuda.get_arch_list() carries the detected gfx
                    target (or the LLVM generic it belongs to) — the wheel
                    has kernels for this part; the list is quoted verbatim
  4 kernel-launch   torch.zeros(16, device="cuda").sum().item() == 0 — the
                    first kernel launch, where the official wheel dies on
                    gfx1151
  5 triton          `import triton` and one trivial @triton.jit kernel,
                    written to a .py file in --workdir (Triton refuses
                    kernels defined in a `<string>` source) — the JIT path
                    every radix kernel takes
  6 sdpa            sdpa.probe_backends at Qwen3.8-27B's head shape; MATH
                    only is a PASS with the one-line flag note from
                    docs/rocm.md, never a failure — drinkme does not set
                    that variable
  7 gemv-bitpin     the smallest case of bench/radix_gemv_bitpin.py's
                    oracle: one ragged bf16 tensor through the sip codec,
                    decode_bits bit-exact against the source, the M=1 GEMV
                    inside the float64 bound (2e-6 * |x|.|W| + 1e-7)

The exit status is NOT the verdict: TheRock's ROCm torch installs an
atexit `_exit(0)` once HIP is up (cli.entry's docstring), so the parent
reads the `SELFTEST PASS` / `SELFTEST FAIL <rung>` line and uses the exit
status only to recognise a signal death. This file ends through os._exit
for the same reason.

Nothing above rung 1 imports torch at module level: the module must be
importable by the parent (for RUNGS and the line format) on a machine with no
torch at all.
"""

from __future__ import annotations

import argparse
import os
import re
import sys

RUNGS = ("import", "device", "arch-list", "kernel-launch", "triton", "sdpa",
         "gemv-bitpin")

PASS_LINE = re.compile(r"^rung (\d+)/(\d+) ([\w-]+): (PASS|FAIL)(?: — (.*))?$")
VERDICT_LINE = re.compile(r"^SELFTEST (PASS|FAIL)(?: (.*))?$")

# LLVM's generic AMDGPU targets (AMDGPUUsage.rst, "Generic processors"): a
# wheel built for the generic runs every member of the family, so a target
# missing from the arch list by name may still be carried. Family -> generic.
GENERIC_FOR = (
    (r"gfx9(0[0-9a-f])", "gfx9-generic"),
    (r"gfx94\d", "gfx9-4-generic"),
    (r"gfx101\d", "gfx10-1-generic"),
    (r"gfx103\d", "gfx10-3-generic"),
    (r"gfx11[0-5]\d", "gfx11-generic"),
    (r"gfx120\d", "gfx12-generic"),
)


def arch_list_covers(target: str | None, arch_list: list[str]) -> tuple[bool, str]:
    """Does the wheel's arch list carry `target`? Exact name first, then
    the LLVM generic its family maps to. The second value quotes what the
    wheel reports, whichever way it goes, so a reader never has to guess
    what the list said."""
    quoted = ", ".join(arch_list) if arch_list else "(empty)"
    if target is None:
        return True, f"wheel reports [{quoted}]; no gfx target to check on this lane"
    if target in arch_list:
        return True, f"wheel reports [{quoted}]; carries {target}"
    for pattern, generic in GENERIC_FOR:
        if re.fullmatch(pattern, target) and generic in arch_list:
            return True, f"wheel reports [{quoted}]; carries {target} via {generic}"
    return False, f"wheel reports [{quoted}]; no kernels for {target}"


def _say(line: str) -> None:
    print(line, flush=True)


def _rung(i: int, name: str, ok: bool, detail: str) -> None:
    _say(f"rung {i}/{len(RUNGS)} {name}: {'PASS' if ok else 'FAIL'} — {detail}")


def _exc(e: BaseException) -> str:
    return f"{type(e).__name__}: {str(e).splitlines()[0][:200] if str(e) else ''}"


# ------------------------------------------------------------------ rungs --


def rung_import() -> tuple[bool, str]:
    import torch  # noqa: F401 — the rung IS the import

    built = (f"hip {torch.version.hip}" if getattr(torch.version, "hip", None)
             else f"cuda {torch.version.cuda}" if getattr(torch.version, "cuda", None)
             else "cpu build")
    return True, f"torch {torch.__version__} ({built}) from {os.path.dirname(torch.__file__)}"


def rung_device() -> tuple[bool, str]:
    import torch

    if not torch.cuda.is_available():
        built = ("a ROCm/HIP build" if getattr(torch.version, "hip", None)
                 else "a CUDA build" if getattr(torch.version, "cuda", None)
                 else "a CPU build")
        return False, f"torch.cuda.is_available() is False ({built} that sees zero devices)"
    n = torch.cuda.device_count()
    props = torch.cuda.get_device_properties(0)
    arch = getattr(props, "gcnArchName", None)
    return True, (f"{n} device{'s' if n != 1 else ''}: {props.name}"
                  f"{' (' + arch + ')' if arch else ''}, "
                  f"{props.total_memory / 1024**3:.1f} GiB")


def rung_arch_list(gfx: str | None) -> tuple[bool, str]:
    import torch

    arch_list = list(torch.cuda.get_arch_list())
    if gfx is None and not getattr(torch.version, "hip", None):
        # the cuda lane: name the device's own sm_XY beside the list
        try:
            major, minor = torch.cuda.get_device_capability(0)
            sm = f"sm_{major}{minor}"
        except Exception:  # noqa: BLE001 — a name is a nicety
            sm = None
        quoted = ", ".join(arch_list) if arch_list else "(empty)"
        if sm and sm in arch_list:
            return True, f"wheel reports [{quoted}]; carries this device's {sm}"
        if sm:
            return True, (f"wheel reports [{quoted}]; this device is {sm}, not listed "
                          "by name — the kernel-launch rung below is the measurement")
        return True, f"wheel reports [{quoted}]"
    return arch_list_covers(gfx, arch_list)


def rung_kernel_launch() -> tuple[bool, str]:
    import torch

    got = torch.zeros(16, device="cuda").sum().item()
    torch.cuda.synchronize()
    if got != 0:
        return False, f"torch.zeros(16, device='cuda').sum().item() returned {got!r}, not 0"
    return True, "torch.zeros(16, device='cuda').sum().item() == 0.0 (the first kernel launch)"


_TRITON_KERNEL = '''\
"""One trivial Triton kernel, written to disk because Triton's JIT reads the
kernel's source file and refuses a function defined in a <string>."""
import triton
import triton.language as tl


@triton.jit
def add_one(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0)
    tl.store(y_ptr + offs, x + 1, mask=mask)
'''


def rung_triton(workdir: str) -> tuple[bool, str]:
    import importlib.util

    import torch
    import triton

    path = os.path.join(workdir, "drinkme_selftest_triton_kernel.py")
    with open(path, "w") as f:
        f.write(_TRITON_KERNEL)
    spec = importlib.util.spec_from_file_location("drinkme_selftest_triton_kernel", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    n = 64
    x = torch.arange(n, device="cuda", dtype=torch.int32)
    y = torch.empty_like(x)
    mod.add_one[(1,)](x, y, n, BLOCK=64)
    torch.cuda.synchronize()
    if not torch.equal(y, x + 1):
        return False, f"triton {triton.__version__}: add-one kernel over {n} ints disagrees"
    return True, f"triton {triton.__version__}; a @triton.jit add-one kernel over {n} ints agrees"


def rung_sdpa() -> tuple[bool, str]:
    from . import sdpa

    shape = sdpa.AttnShape(256, 24, 4)  # Qwen3.8-27B's full-attention layers
    rep = sdpa.probe_backends(shape, device="cuda")
    if not rep.probed:
        return True, f"not probed — {rep.note}"
    if rep.has_efficient:
        flag = f" ({sdpa.EXPERIMENTAL_ENV}={rep.flag})" if rep.flag_set else ""
        return True, f"{', '.join(rep.efficient)} available at {shape}{flag}"
    if rep.flag_set:
        return True, (f"only MATH at {shape} with {sdpa.EXPERIMENTAL_ENV}={rep.flag} "
                      "already set — long prompts will OOM (docs/rocm.md)")
    return True, (f"only MATH at {shape} — set {sdpa.EXPERIMENTAL_ENV}=1 before "
                  "serving long prompts (docs/rocm.md)")


def rung_gemv_bitpin() -> tuple[bool, str]:
    import numpy as np
    import torch

    from .codec import radix_ops, radix_pack as rp
    from .codec.swap import to_device_radix

    # bench/radix_gemv_bitpin.py's RAGGED (129, 260) case with its realistic
    # exponent distribution: finite bf16 bits so the dot has an oracle.
    R, C = 129, 260
    rng = np.random.default_rng(831)
    exps = rng.choice(np.arange(113, 122), size=(R, C),
                      p=np.array([.002, .003, .005, .01, .025, .055, .1, .3, .5]))
    bits = (exps.astype(np.uint16) << 7) | rng.integers(0, 128, (R, C), dtype=np.uint16)
    bits |= rng.integers(0, 2, (R, C), dtype=np.uint16) << 15
    # numpy, explicitly: this rung validates the GPU decode/GEMV kernels, not
    # the encoder choice — a machine with no C++ compiler yet
    # must not read as "the accelerator install is broken" here
    p = rp.pack_array_radix(bits, "sip", encoder="numpy")
    if p is None:
        return False, "the toy tensor did not pack (radix would expand it) — the oracle changed"
    rt = to_device_radix(p, "cuda")
    got = radix_ops.decode_bits(rt).view(torch.int16).cpu().numpy().view(np.uint16)
    if not np.array_equal(got, bits):
        where = np.argwhere(got != bits)[0]
        return False, f"decode_bits differs from the source at {tuple(int(i) for i in where)}"
    W = torch.from_numpy(bits.view(np.int16).copy()).view(torch.bfloat16).cuda()
    g = torch.Generator(device="cuda").manual_seed(916)
    x = torch.randn((C,), device="cuda", dtype=torch.float32, generator=g)
    y = radix_ops.gemv(rt, x)
    torch.cuda.synchronize()
    oracle = x.double() @ W.double().T
    bound = 2e-6 * (x.double().abs() @ W.double().abs().T) + 1e-7
    err = float(((y.double() - oracle).abs() / bound).max())
    if not err <= 1.0:
        return False, f"GEMV exceeds the float64 bound: err/bound {err:.3f}"
    return True, (f"{R}x{C} sip, bpw {p['bpw']:.3f}: {R * C:,} weights decode bit-exact; "
                  f"gemv err/bound {err:.3f}")


# ------------------------------------------------------------------- main --


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="drinkme's post-install self-test (child process)")
    ap.add_argument("--lane", required=True)
    ap.add_argument("--gfx", default=None, help="the detected AMD gfx target, if any")
    ap.add_argument("--workdir", required=True, help="a scratch directory for the triton kernel file")
    a = ap.parse_args(argv)
    gfx = a.gfx or None

    _say(f"selftest lane={a.lane} gfx={gfx or '-'} python={sys.executable}")
    steps = (
        ("import", rung_import),
        ("device", rung_device),
        ("arch-list", lambda: rung_arch_list(gfx)),
        ("kernel-launch", rung_kernel_launch),
        ("triton", lambda: rung_triton(a.workdir)),
        ("sdpa", rung_sdpa),
        ("gemv-bitpin", rung_gemv_bitpin),
    )
    assert tuple(n for n, _ in steps) == RUNGS
    for i, (name, fn) in enumerate(steps, 1):
        try:
            ok, detail = fn()
        except BaseException as e:  # noqa: BLE001 — "it raised" IS the rung's answer
            ok, detail = False, _exc(e)
        _rung(i, name, ok, detail)
        if not ok:
            _say(f"SELFTEST FAIL {name}")
            return 1
    _say("SELFTEST PASS")
    return 0


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(rc)
