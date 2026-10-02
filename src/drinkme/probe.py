"""Measured device bandwidth — the denominator every result carries.

Two numbers: `copy_gb_s` (STREAM copy convention, read+write traffic, kept as
reference) and `read_gb_s`, measured DIRECTLY by a full reduction — reads every
byte, writes one scalar. That is the decode-relevant direction: batch-1 decode
reads every weight once per token and writes almost nothing.

Why read is measured and not taken as copy/2: in a copy, reads and writes
SHARE the bus, so copy/2 charges the read stream for write traffic decode
never generates, while a pure read stream may use the whole bus (on an L4,
stock decode ran at 1.8x a copy/2 "ceiling", which is physically
impossible). The probe runs on an empty device before any model loads so the probe
tensor defeats cache.

torch is imported lazily: detection and the menu must work pre-install.

UNITS: everything this module returns is in BYTES and BYTES PER SECOND —
`probe_bytes`, `read_bytes_s`, `copy_bytes_s`. The conversion to decimal
GB/s happens once, at the record boundary (bench._gb_s), where the unit
string lives.
"""

from __future__ import annotations

import os
import time

GIB = 1024 ** 3  # the probe SIZES in binary (a quarter of free memory); it REPORTS in bytes

# THE HOST'S LOAD while an arm is timed. Other work on the host slows the arms
# unevenly: on an APU the CPU and GPU share one power budget and one memory
# bus, and a launch-bound decode (a small model) waits on the host. bench
# records the 1-minute load average before and after each arm's timed passes,
# with the CPU count (raw.host_load), and warns above HOST_LOAD_WARN_PER_CPU
# x the CPU count: 6.4 on Strix Halo's 32 threads. The line comes from its
# runs of Qwen3-0.6B, -1.7B and -8B (`uptime` before and after
# each bench run), each
# against the quiet window's repeats of the same tree, which agreed within
# 0.1%. The two runs whose load stayed at or under 6.4 (1.9-6.4) were within
# 3% on every arm. The first run past it (5.6 -> 8.2) lost 3.4% on the
# compressed arm. Runs starting at 8-11 lost from nothing to 9% on one arm,
# and 21% on the 0.6B's host-bound twin. The 8B's four runs at 11-27 lost
# 11-17% on the compressed arm against 2-10% on the twin and under 6% on
# stock; the 0.6B at 16 lost 14-29% on every arm; the 8B beside a 16-vCPU
# VM lost 18% (compressed), 6% (twin) and 4% (stock). A ratio from a loaded
# host is biased, on the 8B against the compressed arm.
# The warning never refuses a record: the numbers are the contributor's.
HOST_LOAD_WARN_PER_CPU = 0.2

# THE WARM-UP REP: one UNTIMED rep of each timed shape (decode, prefill,
# TTFT) before its timed reps — compile, first touch, allocator growth — so
# no sample is a cold run. Spelled here, torch-free, because both runtimes'
# arms need the SAME rule for the SAME env var: arms.py (torch) imports
# torch at module level, so a constant it defined could not be imported by
# arms_mlx.py, which must not import torch at all
# (tests/test_mlx_tests_collect_without_torch.py). raw.warmup_rep says
# whether the untimed rep ran; DRINKME_BENCH_WARMUP=0 opts out (a run whose
# sample 0 is deliberately cold) on both runtimes at once; any other value,
# or none, is the default.
WARMUP_REP = os.environ.get("DRINKME_BENCH_WARMUP", "").strip() != "0"


def load1() -> float | None:
    """The host's 1-minute load average (os.getloadavg), or None on an OS
    without one."""
    try:
        return round(os.getloadavg()[0], 2)
    except (AttributeError, OSError):
        return None


def record_host_load(report: dict, arm: str, before: float | None) -> None:
    """raw.host_load: {"cpu_count": n, "load1": {arm: [before, after]}}, the
    host's 1-minute load average before and after `arm`'s timed passes."""
    host = report.setdefault("host_load", {"cpu_count": os.cpu_count(), "load1": {}})
    host["load1"][arm] = [before, load1()]


def host_load_warning(host: dict | None) -> str | None:
    """The summary's warning when any arm was timed with the 1-minute load
    above HOST_LOAD_WARN_PER_CPU x the CPU count, else None."""
    if not host or not host.get("cpu_count"):
        return None
    limit = HOST_LOAD_WARN_PER_CPU * host["cpu_count"]
    busy = {arm: max(v for v in pair if v is not None)
            for arm, pair in (host.get("load1") or {}).items() if any(v is not None for v in pair)}
    busy = {arm: v for arm, v in busy.items() if v > limit}
    if not busy:
        return None
    return (f"the host was busy while {', '.join(busy)} {'was' if len(busy) == 1 else 'were'} timed "
            f"(1-minute load up to {max(busy.values()):.1f}, over {limit:.1f} = "
            f"{HOST_LOAD_WARN_PER_CPU:g} x {host['cpu_count']} CPUs). Other work slows the arms "
            "unevenly, the compressed arm most, so this ratio is not comparable with a quiet run's: "
            "stop other work and run again for a comparable record")


def pick_device() -> str:
    """cuda covers NVIDIA and ROCm builds; mps is the Metal path."""
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    raise RuntimeError(
        "no accelerator: need a CUDA/ROCm build of torch (or MPS on Apple silicon)"
    )


def probe_size_for(device: str, requested: int = 4 * GIB) -> int:
    """Pick a probe size (BYTES) that FITS. The probe needs TWO buffers (copy
    source and destination), so it costs 2x its size in resident memory — on
    an 8 GB card a 4 GiB request is an instant OOM, and 8 GB cards are a tier
    the model menu explicitly targets. Take at most a quarter of free memory,
    floor 256 MiB, rounded down to whole MiB."""
    import torch

    if device == "cuda":
        free, _total = torch.cuda.mem_get_info()
    else:  # mps has no mem_get_info; recommendedMaxWorkingSetSize is not exposed
        free = int(requested * 2)
    return probe_size_from_free(free, requested)


def probe_size_from_free(free: int, requested: int = 4 * GIB) -> int:
    """probe_size_for's rule on a number: a quarter of `free` bytes, whole
    MiB, at most `requested`, at least 256 MiB. Torch-free, so the mlx
    runtime's probe (arms_mlx.measure_bandwidth) sizes by the same rule."""
    budget = (int(free) // 4) // (1 << 20) * (1 << 20)
    return max(GIB // 4, min(int(requested), budget))


def bandwidth_from_timings(nbytes: int, reps: int, copy_seconds: float,
                           read_seconds: float) -> dict:
    """The probe's arithmetic, pure: `reps` copies of `nbytes` (read + write,
    the STREAM copy convention, so 2x the bytes cross the bus) in
    `copy_seconds`, `reps` full reductions of `nbytes` in `read_seconds` ->
    {read_bytes_s, copy_bytes_s}, whole bytes per second. Split out so the
    unit can be pinned without an accelerator: 4 GiB read once in 1 s is
    4,294,967,296 B/s, which the record boundary turns into 4.295 GB/s — not
    4.0, which would be GiB labelled as GB."""
    return {
        "read_bytes_s": int(round(reps * nbytes / read_seconds)),
        "copy_bytes_s": int(round(reps * 2 * nbytes / copy_seconds)),
    }


def measure_bandwidth(probe_bytes: int | None = None, device: str | None = None) -> dict:
    """{probe_bytes, read_bytes_s, copy_bytes_s, device[, gpu_count, gpus]} —
    bytes and bytes/second (module docstring). `device="cpu"` runs the same
    loop on the host with no synchronize: the plumbing is testable without
    an accelerator, and the number it gives is a host number, never
    published as a device one (bench never asks for it)."""
    import torch

    device = device or pick_device()
    probe_bytes = probe_size_for(device) if probe_bytes is None else int(probe_bytes)
    if device == "cuda":
        sync = torch.cuda.synchronize
    elif device == "mps":
        sync = torch.mps.synchronize
    else:
        sync = lambda: None  # noqa: E731 — host memory needs no fence

    n = probe_bytes // 4
    a = torch.empty(n, dtype=torch.float32, device=device)
    a.uniform_()
    b = torch.empty_like(a)
    sync()
    b.copy_(a)
    a.sum()
    sync()  # warm both paths
    reps = 5
    t0 = time.time()
    for _ in range(reps):
        b.copy_(a)
    sync()
    copy_seconds = time.time() - t0
    t0 = time.time()
    for _ in range(reps):
        a.sum()
    sync()
    read_seconds = time.time() - t0
    del a, b
    if device == "cuda":
        torch.cuda.empty_cache()
    elif device == "mps":
        torch.mps.empty_cache()

    out = {
        "probe_bytes": n * 4,
        **bandwidth_from_timings(n * 4, reps, copy_seconds, read_seconds),
        "device": device,
    }
    if device == "cuda":
        out["gpu_count"] = torch.cuda.device_count()
        out["gpus"] = [
            torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
        ]
    return out


def ceiling_tok_s(read_bytes_s: float, weight_bytes: float) -> float:
    """The BF16 bandwidth ceiling for a model: every token reads every
    weight. A ratio of two byte quantities, unit-free — the same number the
    old GiB/s over GiB gave, because the conversion cancelled there."""
    return round(read_bytes_s / weight_bytes, 2)


if __name__ == "__main__":
    import json

    print(json.dumps(measure_bandwidth(), indent=2))
