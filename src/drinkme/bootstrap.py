"""Make one command enough: read the machine, install the matching torch, prove
it launches, carry on.

torch is not a base dependency — it lives in the conflicting
`cuda`/`rocm`/`rocm-official`/`metal` dependency groups so one lock can serve
an NVIDIA machine, an AMD one and a Mac (pyproject's ACCELERATOR LANES
comment explains it). That fork is right, but left to the user it leaks its
implementation: pick the wrong group and you get a wheel that reports zero
devices, which fails by being *slow* rather than by stopping, and a
quickstart that names one group cannot work on the very hardware the
project is built around.

So the machine picks. `drinkme detect` already reads the hardware with nothing but
stdlib, and this module turns that reading into the one decision uv needs —
and then, for the torch lanes, checks the decision in a subprocess before
believing it (`selftest`).

THE TWO AMD LANES (docs/rocm.md carries the table):

  official  the official PyTorch ROCm wheel, `torch 2.13.0+rocm7.1` from
            https://download.pytorch.org/whl/rocm7.1 — the `rocm-official`
            group. It BUNDLES the ROCm runtime (~25 .so in torch/lib; runs
            with no /opt/rocm at all — measured on the RX 7600 XT machine) and
            carries kernels for fifteen targets (`torch.cuda.get_arch_list()`
            on that machine, quoted in LANES below). Bit-exact on gfx1102 with
            `triton-rocm` 3.7.1 from the same index (torch 2.13.0+rocm7.1's
            own requirement; the older `pytorch-triton-rocm` name stops at
            3.5.1 and belongs to torch <= 2.10). The index is PINNED to
            rocm7.1: pytorch.org's "stable" quick-start now points at
            rocm7.14, which does not bundle the runtime (it expects TheRock
            `_rocm_sdk_*` sibling packages) and which is untested.
  therock   AMD's per-target nightly indexes (rocm.nightlies.amd.com/v2).
            gfx1151 (Strix Halo) is the one pinned group (`rocm`, unchanged)
            and it stays there: the official wheel imports, sees the device
            and allocates on that machine — and SEGFAULTS at the first kernel
            launch, every time. Three more index families carry both a Linux
            torch and a Linux triton wheel today (gfx120X-all, gfx94X-dcgpu,
            gfx950-dcgpu); for those the bootstrap prints the exact install
            line and stops, because a pyproject group needs a pinned version
            per index and we have none. The other TheRock indexes
            (gfx110X-dgpu: torch ten months stale and no triton;
            gfx1150, gfx900/906/908/90a, gfx103X, gfx101X: no torch at all)
            are not install targets: a hint naming one would send a Strix
            Point or MI200 owner to an index with nothing to install.

  Order for an AMD part: gfx1151 -> therock. Any other target the official
  wheel lists -> official. Else a therock family with wheels -> the line.
  Else refuse by name. AMD's own support matrix is not a gate (gfx1102 and
  gfx1151 both ran while unlisted); HSA_OVERRIDE_GFX_VERSION is never set.

BEFORE anything downloads, `probe_host` reads what a machine tells us for free:
Linux; the gfx target from the KFD topology in sysfs (and the PCI table);
amdgpu loaded; /dev/kfd and /dev/dri/renderD* readable and writable by this
user — the group-membership problem is refused with its fix before a wheel
is fetched. AFTER an install, `selftest` runs bootstrap_selftest's seven
rungs in a child interpreter: import, device, arch list, first kernel
launch, triton, SDPA backends, a toy gemv bit-pin. A signal death is a typed
verdict (`lane_cannot_launch`), not a crash.

The `metal` lane sits beside the torch ones, for Apple
silicon (docs/metal.md). It is decided by the platform alone — Darwin on
arm64 — and the thing it checks is mlx, not torch. It installs torch as
well, PyPI's CPU wheel, because `drinkme pack` reads checkpoints through it;
the MLX runtime does not use it, so there is no torch self-test.

WHY THIS IS SAFE TO DO AUTOMATICALLY, measured rather than assumed. With `default-groups = []` (pyproject), a bare `uv run` is a no-op for
the accelerator while uv.lock is current — it syncs inexactly, so it neither
installs nor prunes a group's packages (a bare `uv sync` is exact and does
prune them; pyproject's [tool.uv] comment). That is
what makes healing idempotent: we act only when torch is absent or broken, and
nothing uv does behind our back re-breaks it afterwards.

Deliberately narrow:
  - We never touch a working environment. `ok` returns immediately.
  - We never guess. An AMD part outside both lanes gets an honest refusal by
    name and no install.
  - DRINKME_NO_AUTO_DEPS=1 turns the whole thing off; DRINKME_DEVICE=cpu means
    the operator already decided.
  - We only drive uv when this really is a uv project we can see.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from . import exitcodes
from .bootstrap_selftest import PASS_LINE, RUNGS, VERDICT_LINE  # stdlib-only module

# ------------------------------------------------------------- the lanes ----

OFFICIAL_INDEX = "https://download.pytorch.org/whl/rocm7.1"
THEROCK_BASE = "https://rocm.nightlies.amd.com/v2"

# The AMD lane table. One dict; everything the AMD branch of choose_lane
# decides is read off it. Every set here was READ, not guessed:
#   official.targets   `torch.cuda.get_arch_list()` of torch 2.13.0+rocm7.1,
#                      run on an RX 7600 XT
#   official.verified  bit-exact gates on that machine (gfx1102)
#   therock.indexes    the TheRock index folders that carried BOTH a Linux
#                      torch and a Linux triton wheel when read, 2026-09-19
#   therock.groups     the pinned pyproject group per target — gfx1151 only
#   therock.verified   packs served and gates passed on a Strix Halo
LANES = {
    "official": {
        "group": "rocm-official",
        "index": OFFICIAL_INDEX,
        "packages": ("torch", "triton-rocm", "accelerate"),
        "targets": frozenset({
            "gfx900", "gfx906", "gfx908", "gfx90a", "gfx942", "gfx950",
            "gfx1030",
            "gfx1100", "gfx1101", "gfx1102", "gfx1103",
            "gfx1150", "gfx1151",
            "gfx1200", "gfx1201",
        }),
        "verified": frozenset({"gfx1102"}),
    },
    "therock": {
        "indexes": (
            (r"gfx1151", "gfx1151"),
            (r"gfx120\d", "gfx120X-all"),
            (r"gfx94\d", "gfx94X-dcgpu"),
            (r"gfx950", "gfx950-dcgpu"),
        ),
        "packages": ("torch", "rocm[libraries]", "rocm-sdk-core",
                     "rocm-sdk-libraries-{libs}", "triton", "accelerate"),
        "groups": {"gfx1151": "rocm"},
        "verified": frozenset({"gfx1151"}),
    },
}

# Targets that must NEVER take the official lane, and why (measured).
OFFICIAL_REFUSED = {
    "gfx1151": ("the official wheel imports, sees the device and allocates on "
                "this target, then segfaults at the first kernel launch — "
                "reproducible on every run"),
}

# Retail names for the refusal line, keyed by gfx target, for parts whose
# PCI id detect.py may not have seen. detect's table wins when it has a hit.
GFX_NAMES = {
    "gfx1150": "Strix Point (Radeon 880M/890M)",
    "gfx1152": "Krackan Point",
    "gfx1153": "Gorgon Point",
    "gfx1010": "RX 5700 series (RDNA1)",
    "gfx1011": "RDNA1",
    "gfx1012": "RX 5500 series (RDNA1)",
    "gfx900": "Instinct MI25 / Vega 10",
    "gfx906": "Instinct MI50 / Radeon VII",
    "gfx1030": "RX 6800/6900 series (RDNA2)",
}

_MARKER = "DRINKME_BOOTSTRAPPED"  # re-exec guard: heal at most once per process


def gfx_libs_name(gfx: str) -> str:
    """The rocm-sdk-libraries-* suffix for a target. Follows the index slug and
    is LOWERCASE — the gfx1151 index spells it `gfx1151`, the RDNA4 one
    spells it `gfx120x-all` with a small x. Read off the listings, not
    inferred."""
    slug = therock_slug(gfx)
    return slug.lower() if slug else gfx


def therock_slug(gfx: str | None) -> str | None:
    """The TheRock index slug for a target — only for the families whose
    index carries both wheels today (LANES['therock']['indexes'])."""
    if not gfx:
        return None
    for pattern, slug in LANES["therock"]["indexes"]:
        if re.fullmatch(pattern, gfx):
            return slug
    return None


def therock_index(gfx: str | None) -> str | None:
    """The wheel index for an AMD target, or None: this target has no
    TheRock index with a torch AND a triton wheel on it."""
    slug = therock_slug(gfx)
    return f"{THEROCK_BASE}/{slug}/" if slug else None


def therock_line(gfx: str) -> str | None:
    """The exact `uv pip install` line for a therock family target, or
    None where there is no index with both wheels."""
    index = therock_index(gfx)
    if index is None:
        return None
    pkgs = " ".join(p.format(libs=gfx_libs_name(gfx)) for p in LANES["therock"]["packages"])
    return f"uv pip install --index-url {index} {pkgs}"


def gfx_name(gfx: str | None) -> str | None:
    """A retail name for a target, from detect's PCI table (the tables are
    keyed by PCI id, so any id that maps to this target names it) or the
    short list above."""
    if not gfx:
        return None
    from .detect import _AMD_GFX_TARGET, _AMD_PCI_NAMES

    for pci, target in _AMD_GFX_TARGET.items():
        if target == gfx:
            entry = _AMD_PCI_NAMES.get(pci)
            if isinstance(entry, dict):
                return entry["default"]
            if entry:
                return entry
    return GFX_NAMES.get(gfx)


@dataclass
class LaneChoice:
    """What the machine gets. `group` is the pyproject dependency group to sync
    (None = nothing is installed automatically); `status` says how much we
    have measured on this target:

      verified  a real card ran pack + serve and the bit-exact gates
      carried   the official wheel's arch list names this target; nobody
                here has run it — the post-install self-test is the
                measurement
      line      no pinned group; TheRock's index has both wheels; the exact
                install line is `command`
      refused   no lane drinkme can install
      lane      cuda / metal / the CPU wheel — not an AMD decision
    """

    lane: str | None          # "official" | "therock" | "cuda" | "metal" | None
    group: str | None
    status: str
    reason: str
    gfx: str | None = None
    command: str | None = None    # the print-the-line install for `line`
    fallback: str | None = None   # the therock line to try after `lane_cannot_launch`
    selftest: bool = False        # run bootstrap_selftest after the sync: the torch
                                  # lanes with a device to launch on (not the CPU
                                  # wheel, whose rung 2 would fail by design; not metal)

    def message(self) -> str:
        """The paragraph a person reads, with the command where there is one."""
        if self.status == "line":
            return (f"{self.reason}\n    {self.command}\n"
                    "  then re-run with DRINKME_NO_AUTO_DEPS=1. If it works, that is "
                    "a lane we would genuinely like to hear about.")
        return self.reason


def choose_lane(system: str, has_nvidia: bool, has_amd: bool, gfx: str | None,
                machine: str | None = None, name: str | None = None) -> LaneChoice:
    """Which lane this machine takes. Pure, so it can be tested for every machine
    we do not own.

    Naming caveat worth stating out loud: the group is called `cuda` but it is
    really "the PyPI wheel", which is also the correct answer on an Intel Mac
    and on a machine with no accelerator at all (CPU). Renaming it touches
    the lock and every doc, so the name stays and this function carries the
    truth.

    `machine` is platform.machine(): on Darwin it decides between the
    `metal` lane (arm64 — Apple silicon serves through serving/engine_mlx.py
    over mlx, and torch is there only to pack; docs/metal.md) and the PyPI
    wheel (x86_64, CPU).
    Ignored elsewhere; None on Darwin means we cannot tell and take the
    wheel, which fails slow rather than broken.

    `name` is the card's retail name when the detector read one (the PCI
    table, sysfs, or the CPU brand string on an APU); the refusal and lane
    lines say it beside the gfx target. Without one, gfx_name's family
    name stands in.
    """
    if system == "Darwin":
        if machine in ("arm64", "aarch64"):
            return LaneChoice("metal", "metal", "lane", "Apple silicon — mlx (the Metal lane)")
        return LaneChoice("cuda", "cuda", "lane",
                          "macOS on Intel — the PyPI wheel, which runs on CPU")
    if has_nvidia:
        return LaneChoice("cuda", "cuda", "lane", "NVIDIA — the PyPI CUDA wheel", selftest=True)
    if has_amd:
        if system != "Linux":
            return LaneChoice(None, None, "refused",
                              f"AMD on {system}: the ROCm lanes are Linux-only — no "
                              "ROCm torch wheel exists for this platform.")
        if gfx:
            return _amd_lane(gfx, name)
        return LaneChoice(None, None, "refused",
                          "an AMD GPU is present but its gfx target could not be read "
                          "(no KFD topology in sysfs, unrecognized PCI id, no rocminfo). "
                          "Install the torch build that matches your card, then re-run "
                          "with DRINKME_NO_AUTO_DEPS=1.")
    return LaneChoice("cuda", "cuda", "lane",
                      "no accelerator detected — the PyPI wheel, which runs on CPU")


def amd_label(gfx: str, name: str | None = None) -> str:
    """`AMD Radeon RX 7600 XT (gfx1102)` — the part as a person names it,
    then the target; a name that already says AMD (the CPU brand string on
    an APU) is not prefixed twice."""
    name = name or gfx_name(gfx)
    if not name:
        return f"AMD {gfx}"
    if name.upper().startswith("AMD"):
        return f"{name} ({gfx})"
    return f"AMD {name} ({gfx})"


def _amd_lane(gfx: str, name: str | None = None) -> LaneChoice:
    official = LANES["official"]
    therock = LANES["therock"]
    label = amd_label(gfx, name)

    # 1. a pinned TheRock group (gfx1151): measured, and the official wheel
    #    is measured NOT to work there
    if gfx in therock["groups"]:
        why = OFFICIAL_REFUSED.get(gfx)
        return LaneChoice(
            "therock", therock["groups"][gfx], "verified", gfx=gfx, selftest=True,
            reason=(f"{label} — AMD's TheRock per-target wheels (the "
                    f"`{therock['groups'][gfx]}` group): measured on this target"
                    + (f". Not the official PyTorch ROCm wheel: {why}" if why else "")))

    # 2. the official wheel lists it
    if gfx in official["targets"] and gfx not in OFFICIAL_REFUSED:
        fallback = therock_line(gfx)
        if gfx in official["verified"]:
            status, measured = "verified", "measured on this target (bit-exact gates)"
        else:
            status, measured = "carried", ("the wheel carries its kernels (arch list read "
                                           "on a gfx1102 machine); not run by us on this "
                                           "target — the self-test after install is the "
                                           "measurement")
        return LaneChoice(
            "official", official["group"], status, gfx=gfx, fallback=fallback, selftest=True,
            reason=(f"{label} — the official PyTorch ROCm wheel (rocm7.1, the "
                    f"`{official['group']}` group): {measured}"))

    # 3. a TheRock family whose index carries both wheels: the line, no install
    line = therock_line(gfx)
    if line:
        return LaneChoice(
            "therock", None, "line", gfx=gfx, command=line,
            reason=(f"{label}: no pinned group for this target, and the official "
                    f"PyTorch ROCm wheel does not list it; AMD's TheRock index "
                    f"{therock_slug(gfx)} carried a torch and a triton wheel when read "
                    "(2026-09-19; not run by us). This should be it:"))

    # 4. nothing drinkme can install
    return LaneChoice(
        None, None, "refused", gfx=gfx,
        reason=(f"{label}: no lane drinkme can install; the official PyTorch ROCm "
                "wheel does not carry its kernels and TheRock publishes no torch for "
                "it. See docs/rocm.md. Install a torch that matches your card, "
                "then re-run with DRINKME_NO_AUTO_DEPS=1."))


def choose_group(system: str, has_nvidia: bool, has_amd: bool,
                 gfx: str | None, machine: str | None = None) -> tuple[str | None, str]:
    """(group, reason) — choose_lane flattened. group None means WE WILL NOT
    INSTALL — the reason is written to be shown to a person."""
    c = choose_lane(system, has_nvidia, has_amd, gfx, machine=machine)
    return c.group, c.message()


def classify_torch(mod) -> tuple[str, str]:
    """`ok` | `broken`, and why. Caller handles the missing case without
    importing anything.

    `broken` is specifically an accelerator-flavoured wheel that sees no
    device, which serves at CPU speed instead of stopping. A plain CPU wheel is `ok`: it is a
    real choice, and serve.resolve_device warns about it separately.
    """
    if mod.cuda.is_available():
        return "ok", f"torch {mod.__version__} sees a device"
    built_for = ("CUDA" if mod.version.cuda
                 else "ROCm/HIP" if mod.version.hip else None)
    if built_for:
        return "broken", (f"torch {mod.__version__} is a {built_for} build but "
                          "sees zero devices")
    return "ok", f"torch {mod.__version__} is a CPU build"


def classify_mlx(mod) -> tuple[str, str]:
    """The metal lane's classify_torch: `ok` | `broken`, and why, for an
    imported mlx.core. `broken` is an mlx that sees no Metal device on a machine
    the detector says is Apple silicon — the `mlx[cpu]` Linux-style build
    installed by hand, or a wheel that lost its Metal library — which would
    serve every token through the CPU reference path at a Mac's expense and
    call it Metal in the log."""
    try:
        available = bool(mod.metal.is_available())
    except AttributeError:
        available = False
    version = getattr(mod, "__version__", "?")
    if available:
        return "ok", f"mlx {version} sees Metal"
    return "broken", f"mlx {version} is installed but sees no Metal device"


# ------------------------------------------------------ the host probe -----


def decode_gfx_target_version(v: int | str | None) -> str | None:
    """KFD's `gfx_target_version` (sysfs, /sys/class/kfd/kfd/topology/nodes/
    N/properties) -> the LLVM target name: `gfx{v // 10000}{v // 100 % 100}
    {v % 100:x}` — 110501 -> gfx1151, 110002 -> gfx1102, 90010 -> gfx90a.
    Confirmed against the Strix Halo box's own node (110501). 0 is the CPU
    node (no GPU) -> None."""
    try:
        v = int(str(v).strip())
    except (TypeError, ValueError):
        return None
    if v <= 0:
        return None
    return f"gfx{v // 10000}{(v // 100) % 100}{v % 100:x}"


def _kfd_targets(sys_root: str) -> list[tuple[str, str]]:
    """[(node dir, gfx)] for every GPU node KFD enumerated, in sysfs order.
    Empty when amdgpu/KFD is not up — which means the driver is not loaded,
    not that there is no GPU."""
    import glob

    out = []
    for props in sorted(glob.glob(os.path.join(sys_root, "class/kfd/kfd/topology/nodes/*/properties"))):
        try:
            with open(props) as f:
                text = f.read()
        except OSError:
            continue
        m = re.search(r"^gfx_target_version\s+(\d+)\s*$", text, re.M)
        gfx = decode_gfx_target_version(m.group(1)) if m else None
        if gfx:
            out.append((os.path.dirname(props), gfx))
    return out


def _pci_gfx(card_dirs: list[str]) -> tuple[str | None, str | None]:
    """(gfx, pci id) from detect's PCI table for the first AMD card whose id
    it knows."""
    from .detect import _AMD_GFX_TARGET, _read

    for dev in card_dirs:
        pci = (_read(os.path.join(dev, "device")) or "").strip().lower()
        if pci in _AMD_GFX_TARGET:
            return _AMD_GFX_TARGET[pci], pci
    return None, None


def _node_access(path: str) -> tuple[bool, str]:
    """(usable, description) for a device node: readable AND writable by this
    user, else the owner/group/mode so the fix can name the group."""
    import grp
    import pwd
    import stat

    try:
        st = os.stat(path)
    except OSError:
        return False, "missing"
    try:
        group = grp.getgrgid(st.st_gid).gr_name
    except KeyError:
        group = str(st.st_gid)
    try:
        owner = pwd.getpwuid(st.st_uid).pw_name
    except KeyError:
        owner = str(st.st_uid)
    mode = stat.S_IMODE(st.st_mode)
    if os.access(path, os.R_OK | os.W_OK):
        return True, f"rw ({owner}:{group} {mode:04o})"
    return False, f"no access ({owner}:{group} {mode:04o})"


def _distro(os_release: str | None) -> tuple[str, str]:
    """(id, id_like) from /etc/os-release, lowercased, '' when unknown."""
    text = os_release or ""
    m = re.search(r'^ID="?([^"\n]+)"?', text, re.M)
    like = re.search(r'^ID_LIKE="?([^"\n]+)"?', text, re.M)
    return (m.group(1).strip().lower() if m else "",
            like.group(1).strip().lower() if like else "")


@dataclass
class HostProbe:
    """What the machine says before a byte is downloaded (probe_host). `ok` is
    the gate: False means stop, and `refusal` + `fix` say why and what to
    do. Everything else is the evidence, printed line by line."""

    system: str
    ok: bool = True
    refusal: str | None = None
    fix: list[str] = field(default_factory=list)
    gfx: str | None = None
    gfx_source: str | None = None      # "sysfs kfd" | "PCI table" | None
    gfx_all: list[str] = field(default_factory=list)   # every KFD GPU node's target
    pci_gfx: str | None = None
    pci_id: str | None = None
    amdgpu_loaded: bool = False
    kfd: str = "missing"
    render_nodes: dict[str, str] = field(default_factory=dict)
    distro: str = ""
    user: str = ""
    groups: list[str] = field(default_factory=list)
    has_amd: bool = False

    def lines(self) -> list[str]:
        """The `host:` / `target:` lines the bootstrap prints, and the
        refusal + fix when there is one."""
        out = []
        if self.system != "Linux":
            out.append(f"host: {self.system} — {self.refusal}")
            return out
        parts = [f"host: {self.system}"]
        if self.distro:
            parts[0] += f" ({self.distro})"
        parts.append("amdgpu loaded" if self.amdgpu_loaded else "amdgpu NOT loaded")
        parts.append(f"/dev/kfd {self.kfd}")
        for node, how in self.render_nodes.items():
            parts.append(f"{node} {how}")
        if not self.render_nodes:
            parts.append("no /dev/dri/renderD*")
        parts.append(f"user {self.user or '?'}: {', '.join(self.groups) or 'no groups read'}")
        out.append(", ".join(parts))
        if self.gfx:
            src = self.gfx_source
            agree = ""
            if src == "sysfs kfd":
                if self.pci_gfx and self.pci_gfx == self.gfx:
                    agree = f"; PCI table {self.pci_id} agrees"
                elif self.pci_gfx:
                    agree = f"; PCI table {self.pci_id} says {self.pci_gfx} — sysfs wins"
                else:
                    agree = "; PCI id not in detect's table"
            elif src == "PCI table":
                agree = f" ({self.pci_id}); no KFD topology in sysfs"
            out.append(f"target: {self.gfx} ({src}{agree})")
            if len(set(self.gfx_all)) > 1:
                out.append(f"target: {len(self.gfx_all)} GPU nodes in KFD "
                           f"({', '.join(self.gfx_all)}) — the first decides the lane; "
                           "HIP_VISIBLE_DEVICES picks the one to serve on")
        elif self.has_amd:
            out.append("target: an AMD GPU is present but no gfx target could be read "
                       "(no KFD topology in sysfs, PCI id not in detect's table)")
        if not self.ok:
            out.append(f"refused before downloading anything: {self.refusal}")
            out.extend(f"  {line}" for line in self.fix)
        return out

    def as_dict(self) -> dict:
        return {
            "system": self.system, "ok": self.ok, "refusal": self.refusal,
            "fix": list(self.fix), "gfx": self.gfx, "gfxSource": self.gfx_source,
            "gfxAll": list(self.gfx_all), "pciGfx": self.pci_gfx, "pciId": self.pci_id,
            "amdgpuLoaded": self.amdgpu_loaded, "kfd": self.kfd,
            "renderNodes": dict(self.render_nodes), "distro": self.distro,
            "user": self.user, "groups": list(self.groups), "hasAmd": self.has_amd,
        }


def probe_host(sys_root: str = "/sys", dev_root: str = "/dev", system: str | None = None,
               os_release: str | None = None, card_dirs: list[str] | None = None,
               user: str | None = None, groups: list[str] | None = None) -> HostProbe:
    """The pre-install probe: no ROCm, no torch, nothing but sysfs, /dev and
    /etc/os-release. Every input is injectable so a fake tree under tmp_path
    stands in for a machine we do not own.

    Reads, in order: the platform (the ROCm lanes are Linux-only); every KFD
    GPU node's `gfx_target_version` (preferred — it is what the driver
    itself reports) and detect's PCI table (used when KFD has nothing;
    quoted for agreement when it does); `/sys/module/amdgpu`; and whether
    this user can open /dev/kfd and /dev/dri/renderD* for read and write.
    A device node this user cannot open is refused with the fix, BEFORE
    anything downloads: a torch wheel is hundreds of megabytes and the
    result would be a ROCm build that sees zero devices.
    """
    import glob
    import platform

    system = system or platform.system()
    if system != "Linux":
        return HostProbe(system=system, ok=False,
                         refusal="the ROCm lanes are Linux-only — no ROCm torch wheel "
                                 f"exists for {system}")

    if os_release is None:
        try:
            with open("/etc/os-release") as f:
                os_release = f.read()
        except OSError:
            os_release = ""
    distro_id, distro_like = _distro(os_release)
    if user is None:
        try:
            import pwd

            user = pwd.getpwuid(os.getuid()).pw_name
        except Exception:  # noqa: BLE001 — a name is a nicety
            user = os.environ.get("USER", "")
    if groups is None:
        try:
            import grp

            groups = [grp.getgrgid(g).gr_name for g in os.getgroups()]
        except Exception:  # noqa: BLE001
            groups = []

    probe = HostProbe(system=system, distro=distro_id, user=user or "", groups=list(groups))

    if card_dirs is None:
        from .detect import _amd_card_dirs

        card_dirs = _amd_card_dirs()
    probe.has_amd = bool(card_dirs)
    probe.amdgpu_loaded = os.path.isdir(os.path.join(sys_root, "module/amdgpu"))

    nodes = _kfd_targets(sys_root)
    probe.gfx_all = [g for _, g in nodes]
    probe.pci_gfx, probe.pci_id = _pci_gfx(card_dirs)
    if nodes:
        probe.gfx, probe.gfx_source = nodes[0][1], "sysfs kfd"
    elif probe.pci_gfx:
        probe.gfx, probe.gfx_source = probe.pci_gfx, "PCI table"
    if probe.gfx or probe.pci_gfx:
        probe.has_amd = True

    kfd_ok, probe.kfd = _node_access(os.path.join(dev_root, "kfd"))
    render_ok = False
    for node in sorted(glob.glob(os.path.join(dev_root, "dri/renderD*"))):
        ok, how = _node_access(node)
        probe.render_nodes[node.replace(dev_root, "/dev", 1)] = how
        render_ok = render_ok or ok

    if not probe.has_amd:
        return probe  # nothing AMD to gate; the lane decision is not ours

    if not probe.amdgpu_loaded:
        probe.ok = False
        probe.refusal = ("the amdgpu kernel driver is not loaded (/sys/module/amdgpu "
                         "is absent) — ROCm cannot see a GPU without it")
        probe.fix = ["Ryzen APUs and every card this project runs use the kernel's own "
                     "inbox amdgpu driver (no DKMS): check `dmesg | grep amdgpu` and that "
                     "the running kernel is recent enough to support this part."]
        return probe
    if not (kfd_ok and render_ok):
        # the group that owns the node is the group to join; read it rather
        # than assume `render`
        owners = []
        for path in [os.path.join(dev_root, "kfd")] + sorted(glob.glob(os.path.join(dev_root, "dri/renderD*"))):
            try:
                import grp

                owners.append(grp.getgrgid(os.stat(path).st_gid).gr_name)
            except (OSError, KeyError):
                pass
        wanted = ",".join(dict.fromkeys(owners)) or "render,video"
        missing = [w for w in wanted.split(",") if w not in groups]
        probe.ok = False
        what = ("/dev/kfd " + probe.kfd if not kfd_ok else
                "no /dev/dri/renderD* this user can open")
        if probe.kfd == "missing":
            probe.refusal = ("/dev/kfd is missing — the amdgpu driver is loaded but KFD "
                             "(the compute half ROCm talks to) is not up")
            probe.fix = ["Check `dmesg | grep -i kfd`; on a Ryzen APU, KFD needs a "
                         "kernel recent enough to support the part."]
            return probe
        probe.refusal = (f"{what}: user {user} is not in the group that owns the GPU "
                         f"device nodes ({wanted})" if missing else
                         f"{what}: this user cannot open the GPU device nodes")
        if distro_id in ("arch", "cachyos", "endeavouros", "manjaro") or "arch" in distro_like:
            probe.fix = [
                f"{distro_id or 'Arch'} normally ships /dev/kfd and /dev/dri/renderD* "
                "world-rw (crw-rw-rw-) through udev, so this machine has a stricter rule; "
                f"either join the group — `sudo usermod -aG {wanted} $USER`, then log "
                "out and back in — or restore the udev default.",
            ]
        else:
            probe.fix = [
                f"sudo usermod -aG {wanted} $USER",
                "then log out and back in (a new login shell is not enough for the "
                "group to apply to the desktop session), and re-run.",
            ]
            if distro_id in ("ubuntu", "debian", "pop", "linuxmint") or "debian" in distro_like:
                probe.fix.append("(Debian/Ubuntu gate the nodes by group; AMD's own "
                                 "prerequisites page names `video,render`.)")
        return probe
    return probe


# ----------------------------------------------------------------- probing --


def amd_gfx_target(runner=None, card_dirs=None, sys_root: str = "/sys") -> tuple[str | None, str]:
    """(gfx target, how we know it). Injectable for tests.

    KFD's topology in sysfs first — the driver's own word, readable with no
    ROCm installed at all — then detect's PCI table, then rocminfo (which
    belongs to a SYSTEM ROCm install a fresh machine deliberately will not have;
    the whole point of both wheel lanes is that they bring their own).
    """
    from .detect import _AMD_GFX_TARGET, _amd_card_dirs, _read, _run

    nodes = _kfd_targets(sys_root)
    if nodes:
        return nodes[0][1], "sysfs kfd"
    for dev in (card_dirs if card_dirs is not None else _amd_card_dirs()):
        pci = (_read(os.path.join(dev, "device")) or "").strip().lower()
        if pci in _AMD_GFX_TARGET:
            return _AMD_GFX_TARGET[pci], f"PCI id {pci}"
    runner = runner or _run
    out = runner(["rocminfo"])
    if out:
        m = re.search(r"\bgfx\d+[a-z]?\b", out)
        if m:
            return m.group(0), "rocminfo"
    return None, "no KFD topology, unrecognized PCI id, no rocminfo"


def detect_lane(gfx_override: str | None = None) -> LaneChoice:
    """choose_lane() fed from the real machine. `gfx_override` pretends the
    AMD target is another one (the CLI's --gfx, dry runs only)."""
    import platform

    from .detect import _amd_card_dirs, _detect_amd, _detect_nvidia

    system = platform.system()
    if system == "Darwin":
        return choose_lane(system, False, False, None, machine=platform.machine())
    has_amd = bool(_amd_card_dirs()) or bool(gfx_override)
    gfx = gfx_override or (amd_gfx_target()[0] if has_amd else None)
    name = None
    if has_amd and not gfx_override:
        hw = _detect_amd()
        if hw is not None and hw.device_source in ("table", "sysfs", "cpu-brand"):
            name = hw.device_class
    return choose_lane(system, _detect_nvidia() is not None, has_amd, gfx, name=name)


def project_root() -> Path | None:
    """The uv project we are running out of, or None.

    A uv venv lives at <project>/.venv, so sys.prefix's parent is the project —
    checked against pyproject.toml's own name so we never drive uv in someone
    else's directory.
    """
    root = Path(sys.prefix).parent
    pp = root / "pyproject.toml"
    try:
        if pp.is_file() and re.search(r'^name\s*=\s*"drinkme"',
                                      pp.read_text(), re.M):
            return root
    except OSError:
        pass
    return None


# ---------------------------------------------------------------- selftest --

# Exit statuses that mean the child died of a signal: subprocess reports a
# signal death as -N; a shell in between reports 128+N; some launchers report
# 135/139 for SIGBUS/SIGSEGV. All three spellings are read.
_SIGNAL_NAMES = {s.value: s.name for s in signal.Signals}


def _signal_of(rc: int | None) -> str | None:
    if rc is None:
        return None
    if rc < 0:
        return _SIGNAL_NAMES.get(-rc, f"signal {-rc}")
    if rc > 128 and (rc - 128) in _SIGNAL_NAMES:
        return _SIGNAL_NAMES[rc - 128]
    return None


@dataclass
class SelfTestResult:
    """bootstrap.selftest's answer — a value, never an exception.

      ok                  every rung printed PASS and the verdict line agreed
      rung_failed         a rung printed FAIL (`failed_rung`, `detail`)
      lane_cannot_launch  the child died of a signal (`signal`) — the
                          official-wheel-on-gfx1151 shape: import, device
                          and allocation fine, SIGSEGV at the first launch;
                          `failed_rung` is the rung in flight
      timeout             the child did not finish inside `timeout` seconds
      no_interpreter      the lane's python could not be started
    """

    verdict: str
    lane: str
    gfx: str | None
    python: str
    rungs: list[str] = field(default_factory=list)   # the child's rung lines, verbatim
    failed_rung: str | None = None
    detail: str = ""
    exit_code: int | None = None
    signal: str | None = None
    output: str = ""   # everything the child printed (stdout + stderr)

    @property
    def ok(self) -> bool:
        return self.verdict == "ok"

    def message(self) -> str:
        """One line, the verdict and what was measured."""
        n = len(self.rungs)
        if self.verdict == "ok":
            return f"self-test passed: {n}/{len(RUNGS)} rungs on {self.lane} ({self.python})"
        if self.verdict == "lane_cannot_launch":
            return (f"self-test: lane {self.lane} cannot launch on this machine — the child "
                    f"interpreter died of {self.signal} (exit {self.exit_code}) during "
                    f"rung {self.failed_rung!r} after {n} rung{'s' if n != 1 else ''} passed")
        if self.verdict == "rung_failed":
            return (f"self-test failed at rung {self.failed_rung!r} on {self.lane}: "
                    f"{self.detail or 'no detail'}")
        if self.verdict == "timeout":
            return (f"self-test: the child interpreter did not finish (killed during rung "
                    f"{self.failed_rung!r} after {n} passed) — a hang at that rung")
        return f"self-test could not run: {self.detail}"


def lane_python(root: Path | None = None) -> str:
    """The lane's interpreter: the project venv's python when we are in a
    checkout, else this one."""
    root = root if root is not None else project_root()
    if root is not None:
        cand = root / ".venv" / "bin" / "python"
        if cand.is_file():
            return str(cand)
    return sys.executable


def _child_env() -> dict:
    """The child sees THIS checkout's drinkme (PYTHONPATH to the package's
    parent, so a worktree tests its own code against the shared venv) and
    never recurses into the bootstrap."""
    env = dict(os.environ)
    src = str(Path(__file__).resolve().parent.parent)
    env["PYTHONPATH"] = src + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env["DRINKME_NO_AUTO_DEPS"] = "1"
    env.setdefault("PYTHONUNBUFFERED", "1")
    return env


def selftest(lane: str, gfx: str | None = None, python: str | None = None,
             timeout: float = 600.0, child: list[str] | None = None,
             stream=None) -> SelfTestResult:
    """Run bootstrap_selftest in a child interpreter and type the outcome.

    `child` replaces the whole child command (tests hand in a stub that
    exits 139 or prints a failing rung). Every rung line the child prints
    is echoed to `stream` as it arrives when a stream is given, so a person
    watching sees where a hang or a segfault lands.
    """
    python = python or lane_python()
    result = SelfTestResult(verdict="ok", lane=lane, gfx=gfx, python=python)
    with tempfile.TemporaryDirectory(prefix="drinkme-selftest-") as workdir:
        cmd = child or [python, "-m", "drinkme.bootstrap_selftest", "--lane", lane,
                        "--gfx", gfx or "", "--workdir", workdir]
        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    text=True, env=_child_env(), cwd=workdir)
        except OSError as e:
            result.verdict, result.detail = "no_interpreter", f"{cmd[0]}: {e}"
            return result
        lines: list[str] = []
        verdict: tuple[str, str] | None = None
        try:
            import threading

            def pump():
                for line in proc.stdout:
                    line = line.rstrip("\n")
                    lines.append(line)
                    if stream is not None:
                        print(f"[drinkme]   {line}", file=stream, flush=True)

            t = threading.Thread(target=pump, daemon=True)
            t.start()
            try:
                rc = proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
                rc = None
            t.join(timeout=5)
        finally:
            proc.stdout.close()

    result.output = "\n".join(lines)
    for line in lines:
        m = PASS_LINE.match(line)
        if m:
            result.rungs.append(line)
            if m.group(4) == "FAIL":
                result.failed_rung, result.detail = m.group(3), m.group(5) or ""
        v = VERDICT_LINE.match(line)
        if v:
            verdict = (v.group(1), v.group(2) or "")
    result.exit_code = rc
    passed = sum(1 for r in result.rungs if ": PASS" in r)
    in_flight = RUNGS[passed] if passed < len(RUNGS) else None

    if rc is None:
        result.verdict = "timeout"
        result.failed_rung = result.failed_rung or in_flight
        return result
    result.signal = _signal_of(rc)
    if result.signal and verdict is None:
        result.verdict = "lane_cannot_launch"
        result.failed_rung = result.failed_rung or in_flight
        return result
    if verdict is not None and verdict[0] == "PASS" and result.failed_rung is None:
        result.verdict = "ok"
        return result
    # a FAIL verdict, or no verdict at all (the child ended without one:
    # an import-time crash, a traceback, the atexit clobber) — a failed run
    # either way, named by the rung that did not pass
    result.verdict = "rung_failed"
    if result.failed_rung is None:
        result.failed_rung = in_flight or "unknown"
        tail = [ln for ln in lines if ln.strip()][-1:] if lines else []
        result.detail = result.detail or (f"the child ended without a verdict (exit {rc})"
                                          + (f": {tail[0][:200]}" if tail else ""))
    return result


def after_selftest(choice: LaneChoice, res: SelfTestResult) -> list[str]:
    """What to print after a self-test that did not pass: the verdict, then
    the next thing to try — the TheRock line for an official-lane target
    whose family carries wheels, else the refusal by name."""
    out = [res.message()]
    if res.ok:
        return out
    if res.verdict == "lane_cannot_launch" and choice.lane == "official" and choice.fallback:
        out += [
            f"the official PyTorch ROCm wheel (rocm7.1) cannot launch a kernel on "
            f"{choice.gfx} here (measured just now: {res.signal} at rung {res.failed_rung!r}). "
            f"AMD's TheRock index {therock_slug(choice.gfx)} carried a torch and a triton "
            "wheel when read (2026-09-19; not run by us). The next thing to try:",
            f"    {choice.fallback}",
            "  then re-run with DRINKME_NO_AUTO_DEPS=1.",
        ]
    elif res.verdict in ("lane_cannot_launch", "rung_failed", "timeout"):
        label = amd_label(choice.gfx) if choice.gfx else (choice.gfx or choice.lane)
        if choice.lane in ("official", "therock"):
            out.append(f"{label}: no other lane drinkme can install for this target "
                       "(docs/rocm.md). The rungs above say what was measured.")
        else:
            out.append(f"{choice.lane}: the rungs above say what was measured; "
                       "serve will refuse a torch that sees no device.")
    return out


# ------------------------------------------------------------------- heal ----


@dataclass
class BootstrapOutcome:
    """ensure_accelerator's answer — a value the CLI dispatches on, never an
    exception, and never a bare return: the command that asked must know
    whether the runtime it is about to read weights through can be trusted.

      ok         the environment was already fine, or the operator opted out
                 (DRINKME_NO_AUTO_DEPS / DRINKME_DEVICE): nothing was touched.
      installed  a lane was installed and PROVEN in this process's venv —
                 the self-test passed, or the lane has nothing to launch on
                 (the CPU wheel, metal) — and a fresh import will find it.
                 (When a broken module was already imported the process
                 re-execs instead, and never returns here.)
      failed     the runtime is not one the command may read weights through,
                 and `reason` names why: the self-test's verdict
                 (`lane_cannot_launch` / SIGSEGV, a failed rung, a timeout),
                 a sync that failed, no uv or no checkout to drive it in, a
                 host the probe refused, a target with no lane to install,
                 or a re-exec the marker stopped. The diagnostic and the
                 print-the-next-lane line have already been printed; the CLI
                 exits nonzero BEFORE any model work, never calling serve.run
                 over a runtime that just died at its first kernel launch.
    """

    verdict: str            # "ok" | "installed" | "failed"
    reason: str | None = None

    @property
    def ok(self) -> bool:
        return self.verdict != "failed"


def _ok() -> BootstrapOutcome:
    return BootstrapOutcome("ok")


def _installed() -> BootstrapOutcome:
    return BootstrapOutcome("installed")


def _failed(reason: str) -> BootstrapOutcome:
    return BootstrapOutcome("failed", reason)


def ensure_accelerator(*, stream=None) -> BootstrapOutcome:
    """No-op when the environment is fine; otherwise install the right torch
    and prove it launches. Returns a BootstrapOutcome (above) — the CLI stops
    on `failed`.

    Called from the CLI before any command that needs weights. Never raises for
    an ordinary "we could not help": it explains, and its outcome names the
    failure so the caller can refuse instead of reading weights through a
    runtime that was just shown not to launch.

    `stream` resolves at CALL time, not as a default argument: a
    `stream=sys.stderr` default binds the object that existed at import, so
    anything that replaces sys.stderr afterwards (a test harness, a wrapper
    that tees the log) gets silently bypassed. Caught by these tests the first
    time they ran — the same evening a buffered print cost 2h37m, which is not
    a coincidence so much as a family resemblance.
    """
    stream = sys.stderr if stream is None else stream
    if os.environ.get("DRINKME_NO_AUTO_DEPS") or os.environ.get("DRINKME_DEVICE"):
        return _ok()

    if is_metal_box():
        # The metal lane (docs/metal.md): the thing to have is mlx, not torch.
        # Same shape as the torch path below — never touch a working
        # environment, re-exec only when the wrong module is already in the
        # process — with mlx.core in torch's seat.
        spec = importlib.util.find_spec("mlx")
        if spec is not None:
            import mlx.core as mx
            state, why = classify_mlx(mx)
            if state == "ok":
                return _ok()
            print(f"[drinkme] {why}.", file=stream, flush=True)
            needs_reexec = True
        else:
            print("[drinkme] mlx is not installed yet.", file=stream, flush=True)
            needs_reexec = False
        choice = choose_lane("Darwin", False, False, None, machine="arm64")
        synced = _sync_group(choice.group, choice.reason, needs_reexec, stream, reexec=False)
        if synced is not True:
            return _failed(synced)
        if not _reexec_or_refresh(needs_reexec, stream):
            return _failed("still not right after a sync (re-exec stopped by the marker)")
        return _installed()

    spec = importlib.util.find_spec("torch")
    if spec is not None:
        import torch
        state, why = classify_torch(torch)
        if state == "ok":
            return _ok()
        print(f"[drinkme] {why}.", file=stream, flush=True)
        needs_reexec = True  # a broken torch is already imported; can't swap live
    else:
        print("[drinkme] torch is not installed yet.", file=stream, flush=True)
        needs_reexec = False

    choice = detect_lane()
    if choice.lane in ("official", "therock"):
        # the free checks first: a device node this user cannot open is
        # refused with its fix before a wheel is fetched
        host = probe_host()
        for line in host.lines():
            print(f"[drinkme] {line}", file=stream, flush=True)
        if not host.ok:
            return _failed(f"host refused before downloading anything: {host.refusal}")
    if choice.group is None:
        print(f"[drinkme] {choice.message()}", file=stream, flush=True)
        return _failed(f"no lane installed ({choice.status}): {choice.reason.splitlines()[0]}")
    synced = _sync_group(choice.group, choice.reason, needs_reexec, stream, reexec=False)
    if synced is not True:
        return _failed(synced)
    if choice.selftest:
        res = selftest(choice.lane, choice.gfx, stream=stream)
        for line in after_selftest(choice, res):
            print(f"[drinkme] {line}", file=stream, flush=True)
        if not res.ok:
            return _failed(f"self-test {res.verdict}: {res.message()}")
    if not _reexec_or_refresh(needs_reexec, stream):
        return _failed("still not right after a sync (re-exec stopped by the marker)")
    return _installed()


def is_metal_box() -> bool:
    """Darwin on Apple silicon — the one machine whose lane is decided by the
    platform alone, ahead of any torch probe. Injectable for tests by
    monkeypatching this name."""
    import platform

    return platform.system() == "Darwin" and platform.machine() in ("arm64", "aarch64")


def sync_command(group: str) -> list[str]:
    return ["uv", "sync", "--no-default-groups", "--group", group]


def _sync_group(group: str, reason: str, needs_reexec: bool, stream,
                reexec: bool = True):
    """Drive `uv sync --no-default-groups --group <group>` in the checkout we
    are running out of, then (when `reexec`) re-exec if a broken module is
    already imported. The uv-driving half of ensure_accelerator, shared by
    the torch lanes, the metal lane and `drinkme bootstrap` so there is one
    banner, one opt-out line, one re-exec guard. Returns True when the sync
    ran and succeeded, else the one-line reason it did not (a str — so
    `is not True` is the failure test, and the reason rides up into
    ensure_accelerator's outcome)."""
    root = project_root()
    uv = shutil.which("uv")
    if root is None or uv is None:
        missing = "uv is not on PATH" if root else "this is not a uv checkout"
        print(f"[drinkme] {missing}, so I can't install it for you. Run:\n"
              f"    {' '.join(sync_command(group))}",
              file=stream, flush=True)
        return f"{missing}; run {' '.join(sync_command(group))} by hand"

    cmd = [uv] + sync_command(group)[1:]
    print(f"[drinkme] {reason}. Installing: {' '.join(cmd[1:])}\n"
          "[drinkme] (one time; DRINKME_NO_AUTO_DEPS=1 to skip this forever)",
          file=stream, flush=True)
    try:
        rc = subprocess.call(cmd, cwd=str(root))
    except OSError as e:
        print(f"[drinkme] could not run uv: {e}", file=stream, flush=True)
        return f"could not run uv: {e}"
    if rc != 0:
        print(f"[drinkme] that sync failed (exit {rc}) — running it by hand "
              "will show you why.", file=stream, flush=True)
        return f"uv sync --group {group} failed (exit {rc})"
    importlib.invalidate_caches()  # a fresh import in this process now finds it
    if reexec:
        _reexec_or_refresh(needs_reexec, stream)
    return True


def _reexec_or_refresh(needs_reexec: bool, stream) -> bool:
    """After a successful sync: nothing to do when no broken module was
    imported (True — a fresh import finds the new one); otherwise re-exec
    (never returns) or, when the process cannot be re-executed, ask for a
    re-run (SystemExit 0). False = the marker stopped a second re-exec: the
    sync did not fix it, and the runtime in this process is still the
    broken one."""
    if needs_reexec:
        # The wrong torch (or mlx) is in this process's module table already,
        # so the only way to pick up the new one is a new process. The marker stops a
        # sync that didn't actually help from looping forever.
        if os.environ.get(_MARKER):
            print("[drinkme] still not right after a sync — stopping rather "
                  "than looping.", file=stream, flush=True)
            return False
        argv0 = sys.argv[0]
        if os.access(argv0, os.X_OK):
            os.execve(argv0, sys.argv, {**os.environ, _MARKER: "1"})
        print("[drinkme] installed — re-run the same command to pick it up.",
              file=stream, flush=True)
        raise SystemExit(exitcodes.OK)
    return True


# ----------------------------------------------------- `drinkme bootstrap` --


def plan_lines(choice: LaneChoice) -> list[str]:
    """The `lane:` / `plan:` lines: what would be installed and checked."""
    status = f" ({choice.status})" if choice.status != "lane" else ""
    out = [f"lane: {choice.lane or 'none'}{status} — {choice.reason}"]
    if choice.status == "line":
        out.append(f"plan: no automatic install for this target; run by hand:")
        out.append(f"    {choice.command}")
        out.append("  then re-run with DRINKME_NO_AUTO_DEPS=1")
    elif choice.status == "refused":
        out.append("plan: nothing to install")
    else:
        out.append(f"plan: {' '.join(sync_command(choice.group))} (into the project venv)")
        if choice.selftest:
            out.append(f"plan: then the self-test in a child interpreter — "
                       f"{len(RUNGS)} rungs: {', '.join(RUNGS)}")
        if choice.fallback:
            out.append(f"plan: if the self-test says lane_cannot_launch, the next thing to "
                       f"try is TheRock's {therock_slug(choice.gfx)} index:")
            out.append(f"    {choice.fallback}")
        elif choice.lane == "official":
            out.append(f"plan: if the self-test says lane_cannot_launch there is no other "
                       f"lane for this target (TheRock publishes no torch for {choice.gfx}) "
                       "— the verdict is a refusal by name")
    return out


def run_bootstrap(dry_run: bool = False, detect_only: bool = False,
                  gfx: str | None = None, stream=None) -> int:
    """`drinkme bootstrap`: probe -> lane -> install -> self-test. With
    --dry-run the plan is printed and nothing is installed or run; with
    --detect-only only the probe and the lane. `gfx` (the CLI's --gfx)
    pretends the machine is another AMD target — dry runs only, so a wrong
    pretence can never install anything."""
    stream = sys.stderr if stream is None else stream

    def say(line: str) -> None:
        print(f"[drinkme] {line}", file=stream, flush=True)

    if gfx and not (dry_run or detect_only):
        say("--gfx pretends the machine is another target; it is only honoured with "
            "--dry-run or --detect-only, so a pretence can never install anything.")
        return exitcodes.USAGE

    choice = detect_lane(gfx_override=gfx)
    if choice.lane == "metal":
        say(f"lane: metal — {choice.reason}")
        say(f"plan: {' '.join(sync_command('metal'))} (mlx, mlx-lm, and torch's CPU "
            "wheel for packing); the check is mlx.core importing and seeing a Metal "
            "device (no torch self-test on this lane)")
        if dry_run or detect_only:
            say("dry run: nothing installed")
            return exitcodes.OK
        ok = _sync_group("metal", choice.reason, False, stream, reexec=False)
        return exitcodes.OK if ok is True else exitcodes.CANT_RUN_HERE

    host = None
    if choice.lane in ("official", "therock") or gfx or (choice.lane is None):
        host = probe_host()
        for line in host.lines():
            say(line)
        if gfx:
            say(f"target: --gfx {gfx} overrides what the machine reads "
                f"({host.gfx or 'nothing'}) for this dry run")
    for line in plan_lines(choice):
        say(line)
    if host is not None and not host.ok and choice.group is not None:
        say("verdict: refused before downloading anything (above)")
        return exitcodes.CANT_RUN_HERE
    if detect_only or dry_run:
        say("verdict: dry run — nothing installed, no self-test run"
            if dry_run else "verdict: detect only — nothing installed")
        return (exitcodes.OK if choice.group is not None or choice.status == "line"
                else exitcodes.CANT_RUN_HERE)
    if choice.group is None:
        say(f"verdict: {choice.status} — nothing installed")
        return exitcodes.CANT_RUN_HERE

    if _sync_group(choice.group, choice.reason, False, stream, reexec=False) is not True:
        return exitcodes.CANT_RUN_HERE
    if not choice.selftest:
        say("verdict: installed; no self-test on this lane (nothing to launch on)")
        return exitcodes.OK
    res = selftest(choice.lane, choice.gfx, stream=stream)
    for line in after_selftest(choice, res):
        say(line)
    say(f"verdict: {res.verdict}")
    return exitcodes.OK if res.ok else exitcodes.CANT_RUN_HERE
