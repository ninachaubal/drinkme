"""Hardware identity + memory budget — the detect step of `drinkme bench`.

Identity is CLASS-first (how owners self-identify: "4090", "M3 Max", "the 395").
Decisions this module encodes (docs/models.md):

- `device_class` (the record's `deviceClass`) — normalized class string from standard utilities:
    Apple:            `sysctl -n machdep.cpu.brand_string`  ("Apple M3 Max")
    Linux/NVIDIA:     nvidia-smi GPU name
    Linux/AMD dGPU:   GPU name
    Linux unified APU: the CPU brand string from /proc/cpuinfo — the GPU name
    LIES on APUs ("Radeon 8060S Graphics" identifies nothing retail).
- `memoryBytes` — classes ship in multiple sizes (3080 10/12GB, 4060 Ti 8/16GB);
  capacity does real work even when the class string is unambiguous. The
  record carries the bytes; `memory_gb` (decimal GB, bytes / 1e9) is the
  menu's unit.
- `memoryKind` — "vram" | "unified". Linux APU detection is a HEURISTIC
  (GTT-backed, "Graphics" in the GPU name, tiny VRAM carveout) → the CLI asks
  for a one-keystroke confirm; this module only reports its guess + evidence.
- `cpuInfo` / `gpuInfo` — VERBATIM raw strings kept under the normalized fields
  (llama-bench lesson): a normalization mistake stays repairable without re-runs.

Budget rule (for the model menu): Apple = the default GPU working-set limit
(~75% of RAM, lower on small machines); discrete = VRAM; Linux unified = GTT.

No torch at import time — detection must run before the heavy install.
"""

from __future__ import annotations

import glob
import os
import platform
import re
import socket
import subprocess
from dataclasses import dataclass, field


def force_ipv4() -> None:
    """Prefer AF_INET for every subsequent lookup in this process.

    Some residential networks blackhole IPv6 to big CDNs and Python HANGS
    rather than falling back (observed against Hugging Face's CDN:
    an 8GB download moved 4.4MB in 7 minutes on v6, 10 of 13 files in 13
    seconds v4-forced). Harmless on healthy networks; call it before any
    HF traffic.
    """
    if getattr(socket.getaddrinfo, "_drinkme_v4", False):
        return  # idempotent: build_engine and check may both call this
    orig = socket.getaddrinfo

    def v4_first(host, port, family=0, type=0, proto=0, flags=0):
        return orig(host, port, socket.AF_INET, type, proto, flags)

    v4_first._drinkme_v4 = True
    socket.getaddrinfo = v4_first


@dataclass
class Hardware:
    device_class: str | None  # normalized class string, None = could not normalize
    # DECIMAL GB (bytes / 1e9) — capacity of the memory the model must live
    # in, the same unit as suggest.Sizes.bf16_gb/comp_gb (decimal GB);
    # packs.py carries its own GiB-vs-GB boundary
    # conversion to compensate (its own docstring still names that hazard).
    # Every detector below now converts bytes -> decimal GB once, at its own
    # boundary (`memory_bytes`/`budget_bytes` carry the untouched bytes for
    # a caller that needs the exact physical figure back).
    memory_gb: float | None
    memory_kind: str | None  # "vram" | "unified"
    budget_gb: float | None  # usable working set for weights (menu input), decimal GB
    cpu_info: str | None  # verbatim
    gpu_info: str | None  # verbatim
    os_info: str = ""
    heuristic: bool = False  # True -> CLI must confirm memory_kind
    evidence: list[str] = field(default_factory=list)  # why we think so
    # How `device_class` was named on an AMD discrete card:
    # "table" (_AMD_PCI_NAMES hit) | "sysfs" (product_name/rocminfo/rocm-smi)
    # | "fallback" ("amdgpu 0x----", never a name) | None everywhere else
    # (Apple/NVIDIA/AMD-unified already carry a real identity by construction
    # — see this module's docstring). bench.py refuses to write a record on
    # "fallback" unless told not to.
    device_source: str | None = None
    # The raw byte counts behind memory_gb/budget_gb, untouched by rounding —
    # sysctl hw.memsize, nvidia-smi's MiB reading, or sysfs's mem_info_*_total,
    # each converted to bytes at its own detector before decimal GB is taken.
    # Carried so a caller needing an exact physical ceiling (drinkme.fit's
    # bytes-throughout primitives) never has to re-inflate a rounded decimal
    # float back into bytes with the wrong constant.
    memory_bytes: int | None = None
    budget_bytes: int | None = None

    def as_record_env(self) -> dict:
        """The environment block a measurement record carries — the lexicon's
        `#environment` shape (lexicons/wtf.petrichor.drinkme.measurement.json):
        `deviceClass` (this `device_class`), `memoryBytes` (the detector's own byte count, an
        integer — never a rounded decimal GB re-inflated), `memoryKind`,
        `cpuInfo`/`gpuInfo`/`os` verbatim. Optional fields are omitted when unknown rather than
        written as null, so a lexicon check reads them as absent. Nothing
        the lexicon does not declare is here: the detector's working notes
        are `as_record_detector`, which bench puts in the record's `raw`, and
        so is `device_source` (bench's own refusal input, not a property of
        the hardware). (A plain
        `asdict(self)` — snake_case, floats — would pass no lexicon
        validator.)"""
        d = {
            "deviceClass": self.device_class,
            "memoryBytes": self.memory_bytes,
            "memoryKind": self.memory_kind,
            "cpuInfo": self.cpu_info,
            "gpuInfo": self.gpu_info,
            "os": self.os_info or None,
        }
        return {k: v for k, v in d.items() if v is not None}

    def as_record_detector(self) -> dict:
        """The detector's working notes, for the record's `raw.detector`:
        the weight budget it derived (`budgetBytes`, bytes — not the stock
        arm's ceiling, which is `stock.budgetBytes`), whether memory_kind was
        a guess (`heuristic`), why it thinks so (`evidence`) and the
        detector's own `version`. Kept so a normalization mistake stays
        repairable without re-runs."""
        d = {
            "budgetBytes": self.budget_bytes,
            "heuristic": self.heuristic,
            "evidence": list(self.evidence),
            "version": 2,
        }
        return {k: v for k, v in d.items() if v is not None}


def _run(cmd: list[str]) -> str | None:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        return out.stdout.strip() if out.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def _read(path: str) -> str | None:
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return None


_GB = 1_000_000_000  # decimal GB — every detector's bytes -> memory_gb/budget_gb edge


# ---------------------------------------------------------------- Apple ----

# Default GPU working-set budget by RAM (GB). Metal's recommendedMaxWorkingSetSize
# is ~75% of RAM on big machines but tighter on small ones — the 8GB figure is
# the measured ~5.4GB default that makes the 4B fit-cliff experiment what it is.
_APPLE_BUDGET = {8: 5.4}
_APPLE_BUDGET_FRACTION = 0.75


def _detect_darwin() -> Hardware:
    brand = _run(["sysctl", "-n", "machdep.cpu.brand_string"])
    mem = _run(["sysctl", "-n", "hw.memsize"])
    mem_bytes = int(mem) if mem and mem.isdigit() else None
    mem_gb = round(mem_bytes / _GB, 2) if mem_bytes is not None else None
    budget = None
    budget_bytes = None
    if mem_gb:
        # _APPLE_BUDGET is keyed by a truncated GB count (int(mem_gb)): an
        # 8 GiB machine's decimal mem_gb is 8.59, and int() of that is still
        # 8, so the measured-default lookup still resolves for the one
        # machine class it covers.
        budget = _APPLE_BUDGET.get(int(mem_gb), round(mem_gb * _APPLE_BUDGET_FRACTION, 2))
        budget_bytes = round(budget * _GB)
    return Hardware(
        device_class=brand,
        memory_gb=mem_gb,
        memory_bytes=mem_bytes,
        memory_kind="unified",
        budget_gb=budget,
        budget_bytes=budget_bytes,
        cpu_info=brand,
        gpu_info=brand,  # on Apple silicon the SoC string IS the GPU identity
        os_info=f"macOS {platform.mac_ver()[0]}",
        evidence=["Darwin + Apple silicon: unified by construction"],
    )


# ---------------------------------------------------------------- NVIDIA ----


def _detect_nvidia() -> Hardware | None:
    out = _run(
        ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"]
    )
    if not out:
        return None
    name, mem_mib = [x.strip() for x in out.splitlines()[0].split(",")[:2]]
    mem_bytes = round(float(mem_mib) * 1024**2)  # nvidia-smi reports MiB
    mem_gb = round(mem_bytes / _GB, 2)
    # GB10 (DGX Spark) and Jetson are unified CUDA parts; everything else is vram.
    unified = any(k in name for k in ("GB10", "Jetson", "Orin"))
    return Hardware(
        device_class=name,
        memory_gb=mem_gb,
        memory_bytes=mem_bytes,
        memory_kind="unified" if unified else "vram",
        budget_gb=mem_gb,
        budget_bytes=mem_bytes,
        cpu_info=_cpu_brand_linux(),
        gpu_info=out.splitlines()[0],
        os_info=_os_linux(),
        heuristic=unified,  # unified-CUDA classification is name-based -> confirm
        evidence=[f"nvidia-smi: {out.splitlines()[0]}"],
    )


# ------------------------------------------------------------------- AMD ----


def _cpu_brand_linux() -> str | None:
    cpuinfo = _read("/proc/cpuinfo") or ""
    m = re.search(r"^model name\s*:\s*(.+)$", cpuinfo, re.M)
    return m.group(1).strip() if m else None


def _os_linux() -> str:
    osr = _read("/etc/os-release") or ""
    m = re.search(r'^PRETTY_NAME="?([^"\n]+)', osr, re.M)
    return m.group(1) if m else platform.platform()


def amd_is_unified(vram_gb: float | None, gtt_gb: float | None) -> bool:
    """The APU heuristic, pure: unified parts run a tiny VRAM carveout and a
    huge GTT (Strix Halo ships 0.5GB carveout / 124GB GTT); a discrete card is
    the opposite. Always heuristic=True upstream -> the CLI confirms."""
    return bool(vram_gb is not None and gtt_gb is not None and gtt_gb > 2 * vram_gb)


# A discrete AMD card's PCI id is NOT a name. The
# amdgpu kernel driver stopped shipping a per-device id table for anything
# RDNA3 or newer — checked directly (2026-09-14) against
# drivers/gpu/drm/amd/amdgpu/amdgpu_drv.c in torvalds/linux@master: every
# entry from Navi3x onward is one wildcard match on PCI class
# (PCI_CLASS_DISPLAY_VGA/_OTHER/_ACCELERATOR_PROCESSING) with
# driver_data = CHIP_IP_DISCOVERY — the chip identifies itself to the driver
# at runtime via IP-discovery tables in its own vbios, not a static id list.
# So there is no kernel table left to read a name out of; the two tables
# below are sourced from userspace instead, the same data Mesa/ROCm read to
# print names like "Radeon RX 7600 XT" in the first place:
#
# _AMD_PCI_NAMES: PCI device id -> marketing name (or, where one device id
#   covers several SKUs off the same die, {revision hex: name, "default":
#   name}). Source: libdrm's own id table, data/amdgpu.ids —
#   gitlab.freedesktop.org/mesa/drm/-/raw/main/data/amdgpu.ids, fetched
#   2026-09-14. Format there is "device_id,\trevision_id,\tproduct_name".
# _AMD_GFX_TARGET: PCI device id -> LLVM/ROCm gfx target. Source: LLVM's
#   AMDGPUUsage.rst "Processors" tables —
#   llvm/llvm-project, llvm/docs/AMDGPUUsage.rst, fetched 2026-09-14
#   (matched by chip family, e.g. every Navi33 id -> gfx1102).
#
# Not exhaustive — the common RDNA3/RDNA4 dGPUs, the RDNA3.5 APUs this
# project ships on (Strix Point/Halo) and the Phoenix/Hawk Point 780M APU,
# and the CDNA Instinct accelerators likely to show up on a Strix Halo-class
# machine. An id that misses both tables falls through to the sysfs/rocminfo
# read, then to the pci-id fallback below — it never crashes, it degrades.
_AMD_PCI_NAMES: dict[str, str | dict[str, str]] = {
    # RDNA3 dGPU — Navi31 (gfx1100)
    "0x744c": {"default": "Radeon RX 7900 XTX", "c8": "Radeon RX 7900 XTX",
               "cc": "Radeon RX 7900 XT", "ce": "Radeon RX 7900 GRE",
               "cf": "Radeon RX 7900M"},
    "0x7448": "Radeon Pro W7900",
    "0x7449": "Radeon Pro W7800 48GB",
    "0x744a": "Radeon Pro W7900 Dual Slot",
    "0x744b": "Radeon Pro W7900D",
    "0x745e": "Radeon Pro W7800",
    # RDNA3 dGPU — Navi32 (gfx1101)
    "0x7470": "Radeon Pro W7700",
    "0x747e": {"default": "Radeon RX 7800 XT", "c8": "Radeon RX 7800 XT",
               "d8": "Radeon RX 7800M", "db": "Radeon RX 7700",
               "ff": "Radeon RX 7700 XT"},
    # RDNA3 dGPU — Navi33 (gfx1102)
    "0x7480": {"default": "Radeon RX 7600 XT", "00": "Radeon Pro W7600",
               "c0": "Radeon RX 7600 XT", "c1": "Radeon RX 7700S",
               "c2": "Radeon RX 7650 GRE", "c3": "Radeon RX 7600S",
               "c7": "Radeon RX 7600M XT", "cf": "Radeon RX 7600"},
    "0x7483": "Radeon RX 7600M",
    # RDNA3.5 APU — Strix Point (gfx1150) / Strix Halo (gfx1151)
    "0x150e": {"default": "Radeon 890M", "c1": "Radeon 890M",
               "c4": "Radeon 880M", "c5": "Radeon 890M", "c6": "Radeon 890M",
               "c7": "Radeon 890M", "d1": "Radeon 890M", "d2": "Radeon 880M",
               "d3": "Radeon 890M"},
    "0x1586": {"default": "Radeon 8060S / Strix Halo 395",
               "c1": "Radeon 8060S / Strix Halo 395", "c2": "Radeon 8050S",
               "c3": "Radeon 8060S / Strix Halo 395", "c4": "Radeon 8050S",
               "c6": "Radeon 8060S / Strix Halo 395",
               "d1": "Radeon 8060S / Strix Halo 395", "d2": "Radeon 8050S",
               "d4": "Radeon 8050S", "d5": "Radeon 8040S"},
    # RDNA3 APU — Phoenix/Hawk Point (gfx1103)
    "0x15bf": "Radeon 780M",
    "0x1900": "Radeon 780M",
    # RDNA4 dGPU — Navi48 (gfx1201) / Navi44 (gfx1200)
    "0x7550": {"default": "Radeon RX 9070 XT", "c0": "Radeon RX 9070 XT",
               "c2": "Radeon RX 9070 GRE", "c3": "Radeon RX 9070"},
    "0x7590": {"default": "Radeon RX 9060 XT", "c0": "Radeon RX 9060 XT",
               "c1": "Radeon RX 9060 XT LP", "c7": "Radeon RX 9060"},
    # CDNA — Instinct accelerators
    "0x66a1": "Instinct MI60 / MI50",  # gfx906
    "0x738c": "Instinct MI100",  # gfx908
    "0x7408": "Instinct MI250X",  # gfx90a
    "0x740c": "Instinct MI250X / MI250",  # gfx90a
    "0x740f": "Instinct MI210",  # gfx90a
    "0x74a0": "Instinct MI300A",  # gfx942
    "0x74a1": "Instinct MI300X",  # gfx942
    "0x74a2": "Instinct MI308X",  # gfx942
    "0x74a5": "Instinct MI325X",  # gfx942
    "0x74a9": "Instinct MI300X HF",  # gfx942
    "0x74b5": "Instinct MI300X VF",  # gfx942
    "0x74b6": "Instinct MI308X",  # gfx942
    "0x74bd": "Instinct MI300X HF",  # gfx942
}

_AMD_GFX_TARGET: dict[str, str] = {
    "0x744c": "gfx1100", "0x7448": "gfx1100", "0x7449": "gfx1100",
    "0x744a": "gfx1100", "0x744b": "gfx1100", "0x745e": "gfx1100",
    "0x7470": "gfx1101", "0x747e": "gfx1101",
    "0x7480": "gfx1102", "0x7483": "gfx1102",
    "0x150e": "gfx1150",
    "0x1586": "gfx1151",
    "0x15bf": "gfx1103", "0x1900": "gfx1103",
    "0x7550": "gfx1201",
    "0x7590": "gfx1200",
    "0x66a1": "gfx906",
    "0x738c": "gfx908",
    "0x7408": "gfx90a", "0x740c": "gfx90a", "0x740f": "gfx90a",
    "0x74a0": "gfx942", "0x74a1": "gfx942", "0x74a2": "gfx942",
    "0x74a5": "gfx942", "0x74a9": "gfx942", "0x74b5": "gfx942",
    "0x74b6": "gfx942", "0x74bd": "gfx942",
}


def _norm_hex(raw: str | None) -> str:
    return (raw or "").strip().lower().removeprefix("0x")


def _table_name(pci_id: str | None, revision: str | None) -> tuple[str, str] | None:
    """(name, gfx target) from the PCI-id tables above, or None on a miss.
    Revision disambiguates a die shared by several SKUs (0x7480 alone covers
    the W7600 workstation card and five consumer/mobile ones); the untagged
    "default" is used when the revision is absent or itself unrecognized."""
    key = _norm_hex(pci_id)
    if not key:
        return None
    key = f"0x{key}"
    entry = _AMD_PCI_NAMES.get(key)
    if entry is None:
        return None
    gfx = _AMD_GFX_TARGET.get(key, "")
    if isinstance(entry, str):
        return entry, gfx
    return entry.get(_norm_hex(revision), entry["default"]), gfx


def _sysfs_product_name(dev: str) -> str | None:
    """`/sys/class/drm/cardN/device/product_name` — read-only, no sudo;
    populated from the card's own vbios VPD where the vendor ships one
    (common on workstation/Instinct cards, often absent on consumer ones,
    which is exactly why the table above exists)."""
    return _read(os.path.join(dev, "product_name")) or None


def _rocminfo_marketing_name() -> str | None:
    """rocminfo's `Marketing Name:` line — read-only, no sudo; slower than
    the sysfs reads above (it enumerates agents) so it is only tried after
    them."""
    out = _run(["rocminfo"])
    if not out:
        return None
    m = re.search(r"Marketing Name:\s*(.+)", out)
    return m.group(1).strip() if m else None


def _amd_card_dirs() -> list[str]:
    dirs = []
    for dev in sorted(glob.glob("/sys/class/drm/card[0-9]*/device")):
        if (_read(os.path.join(dev, "vendor")) or "").lower() == "0x1002":
            dirs.append(dev)
    return dirs


def _detect_amd() -> Hardware | None:
    cards = _amd_card_dirs()
    if not cards:
        return None
    dev = cards[0]
    vram = _read(os.path.join(dev, "mem_info_vram_total"))
    gtt = _read(os.path.join(dev, "mem_info_gtt_total"))
    vram_bytes = int(vram) if vram and vram.isdigit() else None
    gtt_bytes = int(gtt) if gtt and gtt.isdigit() else None
    vram_gb = round(vram_bytes / _GB, 2) if vram_bytes is not None else None
    gtt_gb = round(gtt_bytes / _GB, 2) if gtt_bytes is not None else None

    # GPU marketing name, best effort: rocm-smi if present, else the raw ids.
    gpu_name = None
    smi = _run(["rocm-smi", "--showproductname"])
    if smi:
        m = re.search(r"Card series:\s*(.+)", smi)
        gpu_name = m.group(1).strip() if m else None
    pci_id = _read(os.path.join(dev, "device"))
    revision = _read(os.path.join(dev, "revision"))
    gpu_verbatim = f"amdgpu {pci_id or '?'} vram={vram_gb}GB gtt={gtt_gb}GB" + (
        f" name={gpu_name}" if gpu_name else ""
    )

    evidence = [f"vram_total={vram_gb}GB", f"gtt_total={gtt_gb}GB"]
    unified = amd_is_unified(vram_gb, gtt_gb)
    if unified:
        evidence.append("gtt >> vram carveout -> unified APU")
        cpu = _cpu_brand_linux()
        mem_bytes = gtt_bytes + (vram_bytes or 0)
        return Hardware(
            device_class=cpu,  # the CPU brand string IS the retail identity on APUs
            memory_gb=round(mem_bytes / _GB, 2),
            memory_bytes=mem_bytes,
            memory_kind="unified",
            budget_gb=gtt_gb,
            budget_bytes=gtt_bytes,
            cpu_info=cpu,
            gpu_info=gpu_verbatim,
            os_info=_os_linux(),
            heuristic=True,
            evidence=evidence,
            device_source="cpu-brand",
        )
    evidence.append("vram-dominant -> discrete")

    # A discrete card must never come out of here with device_class=None — a
    # `bench` run on a card neither table nor sysfs can name would otherwise
    # write `<model>_unknown-device_<date>.json` (RX 7600 XT:
    # gpuInfo carries the pci id fine, `device_class` does not). Priority: the
    # static table (most specific — SKU-exact where the die is shared) beats
    # a live system read (sysfs product_name, then rocminfo, then rocm-smi's
    # own "Card series" above) beats the pci-id fallback, which is always
    # present because `pci_id` came straight off sysfs above.
    table = _table_name(pci_id, revision)
    pci_norm = f"0x{_norm_hex(pci_id)}" if pci_id else None
    if table:
        name, gfx = table
        source = "table"
        evidence.append(f"pci id table: {pci_norm} rev {_norm_hex(revision) or '?'} "
                        f"-> {name} ({gfx})")
    else:
        name = _sysfs_product_name(dev) or _rocminfo_marketing_name() or gpu_name
        if name:
            source = "sysfs"
            evidence.append(f"sysfs/rocminfo/rocm-smi name: {name}")
        else:
            name = f"amdgpu {pci_norm}" if pci_norm else "amdgpu unknown-pci-id"
            source = "fallback"
            evidence.append(f"no table or live name for {pci_norm} -> pci-id fallback "
                            "(add it to detect._AMD_PCI_NAMES)")
    return Hardware(
        device_class=name,  # never None: table hit, live name, or the pci-id fallback
        memory_gb=vram_gb,
        memory_bytes=vram_bytes,
        memory_kind="vram",
        budget_gb=vram_gb,
        budget_bytes=vram_bytes,
        cpu_info=_cpu_brand_linux(),
        gpu_info=gpu_verbatim,
        os_info=_os_linux(),
        heuristic=(source == "fallback"),
        evidence=evidence,
        device_source=source,
    )


# ----------------------------------------------------------------- entry ----


def detect() -> Hardware:
    """Best-effort hardware identity. Never raises; unknowns stay None.

    The CLI layer owns the confirm step (heuristic=True) and the manual
    fallback — this module reports, it does not interrogate the user.
    """
    if platform.system() == "Darwin":
        return _detect_darwin()
    hw = _detect_nvidia()
    if hw:
        return hw
    hw = _detect_amd()
    if hw:
        return hw
    return Hardware(
        device_class=None,
        memory_gb=None,
        memory_kind=None,
        budget_gb=None,
        cpu_info=_cpu_brand_linux(),
        gpu_info=None,
        os_info=_os_linux(),
        heuristic=True,
        evidence=["no accelerator found by any detector"],
    )


# ------------------------------------------------------- live free memory ----

_MEMINFO = "/proc/meminfo"


def meminfo_available_bytes() -> int | None:
    """/proc/meminfo's MemAvailable in bytes: what Linux can hand a process
    now without swapping, and on a unified APU the pool its GTT window draws
    from. None when the file or the line can't be read."""
    try:
        with open(_MEMINFO) as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024  # kB -> bytes
    except (OSError, ValueError, IndexError):
        return None
    return None


def live_free_bytes(hw: Hardware) -> tuple[int | None, str]:
    """What the memory `hw`'s model must live in has free RIGHT NOW, and what
    that reading is called: the live counterpart of hw.budget_bytes, which is
    a capacity. Linux only (a Mac reads fit.host_available_bytes), torch-free,
    and on the same device detect() chose — NVIDIA first, then the first AMD
    card:
      unified (AMD APU, GB10/Jetson): MemAvailable;
      vram, NVIDIA: nvidia-smi's memory.free for the first GPU;
      vram, AMD: sysfs mem_info_vram_total - mem_info_vram_used.
    (None, why) when it can't be read; the caller keeps the capacity and says
    the reading was missing, never "all of it is free"."""
    if hw.memory_kind == "unified":
        free = meminfo_available_bytes()
        return (free, "MemAvailable") if free is not None else (None, "MemAvailable unread")
    if hw.memory_kind != "vram":
        return None, "no free-memory reading"
    if any(e.startswith("nvidia-smi:") for e in hw.evidence):
        out = _run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"])
        try:
            return round(float(out.splitlines()[0].strip()) * 1024**2), "free VRAM"  # MiB
        except (AttributeError, ValueError, IndexError):
            return None, "free VRAM unread"
    cards = _amd_card_dirs()
    if cards:
        total = _read(os.path.join(cards[0], "mem_info_vram_total"))
        used = _read(os.path.join(cards[0], "mem_info_vram_used"))
        if total and used and total.isdigit() and used.isdigit():
            return max(0, int(total) - int(used)), "free VRAM"
    return None, "free VRAM unread"


if __name__ == "__main__":
    import json

    print(json.dumps(detect().as_record_env(), indent=2))
