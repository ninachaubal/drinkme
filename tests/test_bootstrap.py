"""bootstrap: pick the lane, refuse to guess, prove it launches.

The decision function is pure precisely so every box we do NOT own is
testable — what this module exists to prevent is "the wrong wheel
installed silently", which never reproduces on the machine that has the
right one. The two AMD lanes (official wheel / TheRock) and their order
come from the index listings and the gfx1151 probe (docs/rocm.md); the self-test runs in a child so a
segfault is a verdict here, not a crash.
"""

import os
import signal
import sys
import textwrap
import types

import pytest

from drinkme import bootstrap
from drinkme.bootstrap import LaneChoice, SelfTestResult
from drinkme.bootstrap_selftest import RUNGS, arch_list_covers


# ------------------------------------------------------------ choose_lane --


@pytest.fixture(autouse=True)
def _not_a_mac(monkeypatch):
    """Every torch-lane test below assumes the box is NOT Apple silicon, so
    that a developer running the suite on a Mac exercises the same branches
    the Linux boxes do; the metal-lane tests flip this themselves."""
    monkeypatch.setattr(bootstrap, "is_metal_box", lambda: False)


@pytest.mark.parametrize("system,machine,nvidia,amd,gfx,expect", [
    ("Darwin", "arm64", False, False, None, "metal"),   # Apple silicon: mlx (torch only to pack)
    ("Darwin", "x86_64", False, False, None, "cuda"),   # Intel Mac: the PyPI wheel (CPU)
    ("Linux", "x86_64", True, False, None, "cuda"),
    ("Linux", "x86_64", False, True, "gfx1151", "rocm"),          # Strix Halo: TheRock
    ("Linux", "x86_64", False, True, "gfx1102", "rocm-official"),  # RX 7600 XT: the official wheel
    ("Linux", "x86_64", False, False, None, "cuda"),    # no accelerator: CPU is honest
])
def test_the_boxes_we_can_answer_for(system, machine, nvidia, amd, gfx, expect):
    group, reason = bootstrap.choose_group(system, nvidia, amd, gfx, machine=machine)
    assert group == expect
    assert reason  # never a silent choice; the reason is shown to a person


def test_darwin_with_no_machine_reading_takes_the_wheel_not_the_metal_lane():
    """The metal group pins mlx, which has no wheel for an Intel Mac; if we
    cannot read the machine we fail slow (CPU wheel) rather than broken."""
    assert bootstrap.choose_group("Darwin", False, False, None)[0] == "cuda"


def test_the_lane_table_is_one_dict_with_the_measured_sets():
    """The fifteen targets are the official wheel's own arch list (read on
    the gfx1102 box); the therock index families are the four that carried
    both wheels; gfx1151 is the one pinned TheRock group."""
    official, therock = bootstrap.LANES["official"], bootstrap.LANES["therock"]
    assert len(official["targets"]) == 15
    assert {"gfx1102", "gfx1151", "gfx1150", "gfx1201", "gfx942", "gfx950",
            "gfx900", "gfx1030", "gfx1103"} <= official["targets"]
    assert "gfx1010" not in official["targets"]
    assert official["verified"] == {"gfx1102"}
    assert official["group"] == "rocm-official"
    assert official["index"] == "https://download.pytorch.org/whl/rocm7.1"  # pinned, not "stable"
    assert therock["groups"] == {"gfx1151": "rocm"}
    assert [slug for _, slug in therock["indexes"]] == [
        "gfx1151", "gfx120X-all", "gfx94X-dcgpu", "gfx950-dcgpu"]


def test_strix_halo_takes_therock_and_never_the_official_wheel():
    """Measured: the official wheel imports, sees the device,
    allocates, and segfaults at the first kernel launch on gfx1151. The
    lane says so, and gfx1151 is in the official list only as a fact about
    the wheel's arch list, never as a route."""
    c = bootstrap.choose_lane("Linux", False, True, "gfx1151")
    assert (c.lane, c.group, c.status) == ("therock", "rocm", "verified")
    assert "segfaults at the first kernel launch" in c.reason
    assert c.fallback is None and c.command is None


def test_the_rx_7600_xt_takes_the_official_wheel_verified():
    c = bootstrap.choose_lane("Linux", False, True, "gfx1102")
    assert (c.lane, c.group, c.status) == ("official", "rocm-official", "verified")
    assert "Radeon RX 7600 XT (gfx1102)" in c.reason
    assert "measured on this target" in c.reason
    assert c.fallback is None  # gfx110X-dgpu has no triton and a stale torch: no line


@pytest.mark.parametrize("gfx,slug", [
    ("gfx1201", "gfx120X-all"),   # RX 9070 XT
    ("gfx1200", "gfx120X-all"),   # RX 9060 XT
    ("gfx942", "gfx94X-dcgpu"),   # MI300
    ("gfx950", "gfx950-dcgpu"),   # MI350
])
def test_a_carried_target_with_a_therock_family_gets_the_wheel_and_a_fallback(gfx, slug):
    """The wheel's arch list names it, nobody here has run it: the official
    lane installs ("carried", not "verified" — the reason says so), and the
    TheRock line rides along as the next thing to try after a
    lane_cannot_launch verdict."""
    c = bootstrap.choose_lane("Linux", False, True, gfx)
    assert (c.lane, c.group, c.status) == ("official", "rocm-official", "carried")
    assert "not run by us" in c.reason
    assert c.fallback == bootstrap.therock_line(gfx)
    assert f"https://rocm.nightlies.amd.com/v2/{slug}/" in c.fallback
    assert f"rocm-sdk-libraries-{slug.lower()}" in c.fallback
    assert "triton" in c.fallback


@pytest.mark.parametrize("gfx", ["gfx1100", "gfx1101", "gfx1103", "gfx1150", "gfx90a",
                                 "gfx908", "gfx906", "gfx900", "gfx1030"])
def test_a_carried_target_without_a_therock_family_gets_the_wheel_alone(gfx):
    """RDNA3 discrete, the 780M/Strix Point APUs, RDNA2 and the older
    Instinct parts: the wheel carries their kernels; TheRock's matching
    index has no torch (or no triton), so there is no line to fall back
    to — the plan says so rather than pointing at an empty index."""
    c = bootstrap.choose_lane("Linux", False, True, gfx)
    assert (c.lane, c.status) == ("official", "carried")
    assert c.fallback is None
    assert bootstrap.therock_index(gfx) is None
    plan = "\n".join(bootstrap.plan_lines(c))
    assert "no other lane for this target" in plan
    assert "rocm.nightlies" not in plan


def test_strix_point_is_carried_not_refused_up_front():
    """gfx1150 is in the wheel's arch list; its sibling APU (gfx1151) fails
    the official lane at the first launch, so the self-test decides — and a
    failure there is a refusal by name with no next step (no TheRock torch
    for gfx1150)."""
    c = bootstrap.choose_lane("Linux", False, True, "gfx1150")
    assert (c.lane, c.status, c.fallback) == ("official", "carried", None)
    assert "Radeon 890M (gfx1150)" in c.reason


def test_a_therock_only_family_member_is_handed_the_line():
    """gfx941 (an early MI300 stepping): not in the official wheel's list,
    but the gfx94X-dcgpu index carries both wheels — the exact command,
    no install."""
    c = bootstrap.choose_lane("Linux", False, True, "gfx941")
    assert (c.lane, c.group, c.status) == ("therock", None, "line")
    assert c.command.startswith("uv pip install --index-url https://rocm.nightlies.amd.com/v2/gfx94X-dcgpu/ ")
    assert "rocm-sdk-libraries-gfx94x-dcgpu" in c.command and " triton " in c.command
    group, reason = bootstrap.choose_group("Linux", False, True, "gfx941")
    assert group is None and c.command in reason and "DRINKME_NO_AUTO_DEPS" in reason


@pytest.mark.parametrize("gfx,name", [
    ("gfx1010", "RX 5700 series (RDNA1)"),
    ("gfx1012", "RX 5500 series (RDNA1)"),
    ("gfx1152", "Krackan Point"),
    ("gfx777", None),
])
def test_everything_else_amd_is_refused_by_name(gfx, name):
    """No official kernels, no TheRock torch: the refusal names the part and
    both reasons, and invents no URL."""
    c = bootstrap.choose_lane("Linux", False, True, gfx)
    assert (c.lane, c.group, c.status) == (None, None, "refused")
    assert gfx in c.reason
    if name:
        assert name in c.reason
    assert "does not carry its kernels" in c.reason
    assert "TheRock publishes no torch" in c.reason
    assert "rocm.nightlies" not in c.reason
    assert "DRINKME_NO_AUTO_DEPS" in c.reason


def test_the_detectors_card_name_is_used_and_not_prefixed_twice():
    c = bootstrap.choose_lane("Linux", False, True, "gfx1151",
                              name="AMD RYZEN AI MAX+ 395 w/ Radeon 8060S")
    assert c.reason.startswith("AMD RYZEN AI MAX+ 395 w/ Radeon 8060S (gfx1151) —")
    c = bootstrap.choose_lane("Linux", False, True, "gfx1102", name="Radeon RX 7600")
    assert c.reason.startswith("AMD Radeon RX 7600 (gfx1102) —")


def test_an_amd_box_with_no_readable_target_is_refused_without_a_guess():
    c = bootstrap.choose_lane("Linux", False, True, None)
    assert c.group is None and "gfx target could not be read" in c.reason


def test_amd_outside_linux_is_refused_in_one_line():
    c = bootstrap.choose_lane("Windows", False, True, "gfx1102")
    assert c.group is None and "Linux-only" in c.reason


def test_nvidia_wins_over_a_present_amd_igpu():
    """A discrete NVIDIA card beside an AMD APU is a real desktop. The CUDA
    wheel is right there and the rocm lane would be wrong."""
    assert bootstrap.choose_group("Linux", True, True, "gfx1151")[0] == "cuda"


@pytest.mark.parametrize("gfx,slug,libs", [
    ("gfx1201", "gfx120X-all", "gfx120x-all"),     # RDNA4
    ("gfx942", "gfx94X-dcgpu", "gfx94x-dcgpu"),    # MI300
    ("gfx950", "gfx950-dcgpu", "gfx950-dcgpu"),    # MI350
    ("gfx1151", "gfx1151", "gfx1151"),             # ours
])
def test_the_four_therock_families_with_wheels_get_a_real_index(gfx, slug, libs):
    """Slugs and the LOWERCASE libraries suffix are read off the live
    listings, not inferred — the index is spelled gfx120X-all while its
    package is rocm-sdk-libraries-gfx120x-all, small x."""
    assert bootstrap.therock_index(gfx) == f"https://rocm.nightlies.amd.com/v2/{slug}/"
    assert bootstrap.gfx_libs_name(gfx) == libs


@pytest.mark.parametrize("gfx", ["gfx1102", "gfx1100", "gfx90a", "gfx1150", "gfx1030", "gfx1010"])
def test_the_indexes_with_nothing_to_install_are_no_longer_offered(gfx):
    """Read 2026-09-19: gfx110X-dgpu's torch is ten months stale with no
    triton beside it; gfx1150, gfx900/906/908/90a, gfx103X, gfx101X have no
    torch at all. A hint to those indexes would send a Strix Point or MI200
    owner to an index with nothing on it."""
    assert bootstrap.therock_index(gfx) is None
    assert bootstrap.therock_line(gfx) is None


# ---------------------------------------------------------- classify_torch --


def _torch(version, cuda=None, hip=None, available=False):
    m = types.SimpleNamespace()
    m.__version__ = version
    m.version = types.SimpleNamespace(cuda=cuda, hip=hip)
    m.cuda = types.SimpleNamespace(is_available=lambda: available)
    return m


def test_an_accelerator_wheel_with_no_device_is_broken():
    """The signature exactly: PyPI cu130 on an AMD box."""
    state, why = bootstrap.classify_torch(_torch("2.13.0+cu130", cuda="13.0"))
    assert state == "broken" and "CUDA" in why


def test_a_plain_cpu_wheel_is_not_broken():
    """Someone may have meant it. serve.resolve_device warns separately;
    this module must not go install a GPU torch over their choice."""
    assert bootstrap.classify_torch(_torch("2.13.0"))[0] == "ok"


def test_a_working_gpu_is_ok():
    assert bootstrap.classify_torch(
        _torch("2.12.0a0+rocm7.13", hip="7.13", available=True))[0] == "ok"


# ------------------------------------------------------------ classify_mlx --


def _mlx(version="0.32.2", available=False):
    m = types.SimpleNamespace()
    m.__version__ = version
    m.metal = types.SimpleNamespace(is_available=lambda: available)
    return m


def test_an_mlx_that_sees_metal_is_ok():
    assert bootstrap.classify_mlx(_mlx(available=True))[0] == "ok"


def test_an_mlx_with_no_metal_device_is_broken():
    """The mlx[cpu] build (what the Linux dev box installs by hand) on a box
    the detector calls Apple silicon: every token would take the CPU
    reference path under a Metal label — a CPU fallback behind a GPU label,
    in mlx."""
    state, why = bootstrap.classify_mlx(_mlx())
    assert state == "broken" and "no Metal" in why


# ------------------------------------------------------- the host probe ----


def _fake_box(tmp_path, gfx_versions=(110501,), amdgpu=True, kfd_mode=0o660,
              kfd_group=None, render=("renderD128",), render_mode=0o660,
              card_pci=None):
    """A sysfs + /dev tree under tmp_path. Device nodes are plain files:
    os.access answers for them the same way it does for a character
    device, which is the question the probe asks."""
    sys_root = tmp_path / "sys"
    dev_root = tmp_path / "dev"
    (sys_root / "class/kfd/kfd/topology/nodes/0").mkdir(parents=True)
    (sys_root / "class/kfd/kfd/topology/nodes/0/properties").write_text(
        "cpu_cores_count 32\nsimd_count 0\ngfx_target_version 0\n")
    for i, v in enumerate(gfx_versions, 1):
        d = sys_root / f"class/kfd/kfd/topology/nodes/{i}"
        d.mkdir(parents=True)
        (d / "properties").write_text(f"cpu_cores_count 0\nsimd_count 80\n"
                                      f"gfx_target_version {v}\nvendor_id 4098\n")
    if amdgpu:
        (sys_root / "module/amdgpu").mkdir(parents=True)
    (dev_root / "dri").mkdir(parents=True)
    kfd = dev_root / "kfd"
    kfd.write_text("")
    kfd.chmod(kfd_mode)
    for name in render:
        node = dev_root / "dri" / name
        node.write_text("")
        node.chmod(render_mode)
    cards = []
    if card_pci:
        card = tmp_path / "card0/device"
        card.mkdir(parents=True)
        (card / "device").write_text(card_pci + "\n")
        (card / "vendor").write_text("0x1002\n")
        cards.append(str(card))
    return str(sys_root), str(dev_root), cards


@pytest.mark.parametrize("v,gfx", [
    (110501, "gfx1151"), (110002, "gfx1102"), (110000, "gfx1100"), (90010, "gfx90a"),
    (90402, "gfx942"), (100300, "gfx1030"), (120001, "gfx1201"), (0, None), ("", None),
])
def test_gfx_target_version_decodes_the_kfd_encoding(v, gfx):
    """gfx{v/10000}{v/100%100}{v%100:x} — 110501 is the Strix Halo box's own
    node; 0 is the CPU node."""
    assert bootstrap.decode_gfx_target_version(v) == gfx


def test_a_healthy_box_probes_ok_from_sysfs_and_says_the_pci_table_agrees(tmp_path):
    sys_root, dev_root, cards = _fake_box(tmp_path, card_pci="0x1586")
    p = bootstrap.probe_host(sys_root, dev_root, system="Linux",
                             os_release='ID=ubuntu\nID_LIKE=debian\n', card_dirs=cards,
                             user="alice", groups=["alice", "render", "video"])
    assert p.ok and p.refusal is None
    assert (p.gfx, p.gfx_source, p.pci_gfx) == ("gfx1151", "sysfs kfd", "gfx1151")
    assert p.amdgpu_loaded and p.kfd.startswith("rw")
    assert p.render_nodes == {"/dev/dri/renderD128": p.render_nodes["/dev/dri/renderD128"]}
    text = "\n".join(p.lines())
    assert "target: gfx1151 (sysfs kfd; PCI table 0x1586 agrees)" in text
    assert "amdgpu loaded" in text and "REFUSED" not in text


def test_sysfs_wins_over_the_pci_table_and_says_so(tmp_path):
    sys_root, dev_root, cards = _fake_box(tmp_path, gfx_versions=(110002,), card_pci="0x1586")
    p = bootstrap.probe_host(sys_root, dev_root, system="Linux", os_release="", card_dirs=cards,
                             user="u", groups=["render"])
    assert p.gfx == "gfx1102" and p.pci_gfx == "gfx1151"
    assert "PCI table 0x1586 says gfx1151 — sysfs wins" in "\n".join(p.lines())


def test_without_kfd_topology_the_pci_table_names_the_target(tmp_path):
    sys_root, dev_root, cards = _fake_box(tmp_path, gfx_versions=(), card_pci="0x7480")
    p = bootstrap.probe_host(sys_root, dev_root, system="Linux", os_release="", card_dirs=cards,
                             user="u", groups=["render"])
    assert (p.gfx, p.gfx_source) == ("gfx1102", "PCI table")
    assert "target: gfx1102 (PCI table (0x7480); no KFD topology in sysfs)" in "\n".join(p.lines())


def test_two_gpu_nodes_are_reported_and_the_first_decides(tmp_path):
    sys_root, dev_root, cards = _fake_box(tmp_path, gfx_versions=(110501, 120001))
    p = bootstrap.probe_host(sys_root, dev_root, system="Linux", os_release="", card_dirs=cards,
                             user="u", groups=["render"])
    assert p.gfx == "gfx1151" and p.gfx_all == ["gfx1151", "gfx1201"]
    assert "2 GPU nodes in KFD (gfx1151, gfx1201)" in "\n".join(p.lines())
    assert "HIP_VISIBLE_DEVICES" in "\n".join(p.lines())


def test_a_user_outside_the_render_group_is_refused_with_the_debian_fix(tmp_path):
    """The one that costs a stranger an afternoon: the wheel installs, torch
    imports, and is_available() is False because /dev/kfd is root:render
    0660. Refused BEFORE the download, with the usermod line and the
    re-login note."""
    sys_root, dev_root, cards = _fake_box(tmp_path, kfd_mode=0o000, render_mode=0o000)
    p = bootstrap.probe_host(sys_root, dev_root, system="Linux",
                             os_release='ID=ubuntu\nID_LIKE=debian\n', card_dirs=cards,
                             user="stranger", groups=["stranger"])
    assert not p.ok
    assert "/dev/kfd no access" in p.refusal
    assert "stranger is not in the group" in p.refusal
    fix = "\n".join(p.fix)
    assert fix.startswith("sudo usermod -aG ")
    assert "$USER" in fix and "log out and back in" in fix
    assert "Debian/Ubuntu" in fix
    text = "\n".join(p.lines())
    assert "refused before downloading anything" in text


def test_arch_gets_its_own_wording_because_it_ships_the_nodes_world_rw(tmp_path):
    sys_root, dev_root, cards = _fake_box(tmp_path, kfd_mode=0o000, render_mode=0o000)
    p = bootstrap.probe_host(sys_root, dev_root, system="Linux",
                             os_release='ID=cachyos\nID_LIKE="arch"\n', card_dirs=cards,
                             user="u", groups=["u"])
    assert not p.ok
    fix = "\n".join(p.fix)
    assert "world-rw" in fix and "udev" in fix and "sudo usermod -aG" in fix


def test_a_missing_amdgpu_module_is_refused_by_name(tmp_path):
    sys_root, dev_root, cards = _fake_box(tmp_path, gfx_versions=(), amdgpu=False, card_pci="0x7480")
    p = bootstrap.probe_host(sys_root, dev_root, system="Linux", os_release="", card_dirs=cards,
                             user="u", groups=["render"])
    assert not p.ok and "amdgpu kernel driver is not loaded" in p.refusal


def test_a_missing_dev_kfd_is_refused_by_name(tmp_path):
    sys_root, dev_root, cards = _fake_box(tmp_path)
    os.remove(os.path.join(dev_root, "kfd"))
    p = bootstrap.probe_host(sys_root, dev_root, system="Linux", os_release="", card_dirs=cards,
                             user="u", groups=["render"])
    assert not p.ok and "/dev/kfd is missing" in p.refusal


def test_a_non_linux_host_is_refused_in_one_line(tmp_path):
    p = bootstrap.probe_host(str(tmp_path), str(tmp_path), system="Darwin")
    assert not p.ok and "Linux-only" in p.refusal
    assert p.lines() == [f"host: Darwin — {p.refusal}"]


def test_a_box_with_no_amd_part_is_not_gated(tmp_path):
    """No AMD card, no KFD nodes: the probe has nothing to refuse — the lane
    (cuda, or the CPU wheel) is not its call."""
    sys_root, dev_root, _ = _fake_box(tmp_path, gfx_versions=(), amdgpu=False, kfd_mode=0o000)
    p = bootstrap.probe_host(sys_root, dev_root, system="Linux", os_release="", card_dirs=[],
                             user="u", groups=["u"])
    assert p.ok and not p.has_amd and p.gfx is None


# ------------------------------------------------------------ gfx probing ---


def test_kfd_sysfs_is_read_first(tmp_path):
    sys_root, _, _ = _fake_box(tmp_path, gfx_versions=(110002,))
    got, how = bootstrap.amd_gfx_target(runner=lambda cmd: "  Name:  gfx1151\n",
                                        card_dirs=[], sys_root=sys_root)
    assert got == "gfx1102" and how == "sysfs kfd"


def test_pci_id_carries_a_box_with_no_kfd_topology_and_no_system_rocm(tmp_path):
    """A fresh box has no rocminfo — both wheel lanes bring their own ROCm,
    which is the point. The PCI table has to be enough."""
    dev = tmp_path / "device"
    dev.write_text("0x7480\n")
    got, how = bootstrap.amd_gfx_target(runner=lambda cmd: None, card_dirs=[str(tmp_path)],
                                        sys_root=str(tmp_path / "nosys"))
    assert got == "gfx1102" and "0x7480" in how


def test_rocminfo_is_the_last_resort(tmp_path):
    out = "  Name:  gfx1151\n      Name:  amdgcn-amd-amdhsa--gfx1151\n"
    got, how = bootstrap.amd_gfx_target(runner=lambda cmd: out, card_dirs=[],
                                        sys_root=str(tmp_path / "nosys"))
    assert got == "gfx1151" and how == "rocminfo"


def test_an_unknown_amd_part_reports_unknown_rather_than_guessing(tmp_path):
    (tmp_path / "device").write_text("0xdead\n")
    got, _ = bootstrap.amd_gfx_target(runner=lambda cmd: None, card_dirs=[str(tmp_path)],
                                      sys_root=str(tmp_path / "nosys"))
    assert got is None


# ---------------------------------------------------------- arch_list_covers --


@pytest.mark.parametrize("target,arch_list,ok", [
    ("gfx1151", ["gfx1151"], True),
    ("gfx1102", ["gfx900", "gfx1102", "gfx1201"], True),
    ("gfx1103", ["gfx11-generic"], True),       # the LLVM generic its family maps to
    ("gfx1201", ["gfx12-generic"], True),
    ("gfx942", ["gfx9-4-generic"], True),
    ("gfx1151", ["gfx900", "gfx1102"], False),
    ("gfx1010", ["gfx10-3-generic"], False),    # gfx10.1 is not gfx10.3
    (None, ["sm_80", "sm_90"], True),           # the cuda lane: nothing to check
])
def test_arch_list_covers_exact_names_and_generics(target, arch_list, ok):
    got, detail = arch_list_covers(target, arch_list)
    assert got is ok
    assert "wheel reports [" in detail  # the list is always quoted


# ------------------------------------------------------------- selftest -----


def _stub(tmp_path, body: str) -> list[str]:
    path = tmp_path / "stub_child.py"
    path.write_text(textwrap.dedent(body))
    return [sys.executable, str(path)]


def test_a_segfault_at_the_first_launch_is_the_typed_verdict_not_a_crash(tmp_path):
    """The official-wheel-on-gfx1151 shape exactly: three rungs pass, the
    child dies of SIGSEGV at the fourth. The parent says which rung was in
    flight, and the message quotes the signal."""
    child = _stub(tmp_path, '''
        import os, signal
        print("rung 1/7 import: PASS — torch 2.13.0+rocm7.1 (hip 7.1)", flush=True)
        print("rung 2/7 device: PASS — 1 device: Radeon 8060S Graphics (gfx1151)", flush=True)
        print("rung 3/7 arch-list: PASS — wheel reports [gfx900, gfx1151]; carries gfx1151", flush=True)
        os.kill(os.getpid(), signal.SIGSEGV)
    ''')
    res = bootstrap.selftest("official", "gfx1151", child=child)
    assert res.verdict == "lane_cannot_launch"
    assert res.signal == "SIGSEGV" and res.exit_code == -signal.SIGSEGV
    assert res.failed_rung == "kernel-launch" and len(res.rungs) == 3
    assert "died of SIGSEGV" in res.message() and "'kernel-launch'" in res.message()


def test_exit_139_reads_as_sigsegv_too(tmp_path):
    """A shell or a launcher in between reports 128+N instead of -N."""
    child = _stub(tmp_path, '''
        import sys
        print("rung 1/7 import: PASS — torch", flush=True)
        sys.exit(139)
    ''')
    res = bootstrap.selftest("official", "gfx1150", child=child)
    assert res.verdict == "lane_cannot_launch" and res.signal == "SIGSEGV"
    assert res.failed_rung == "device"


def test_a_failing_rung_is_named_with_its_detail(tmp_path):
    child = _stub(tmp_path, '''
        import sys
        print("rung 1/7 import: PASS — torch 2.13.0+rocm7.1", flush=True)
        print("rung 2/7 device: FAIL — torch.cuda.is_available() is False (a ROCm/HIP build that sees zero devices)", flush=True)
        print("SELFTEST FAIL device", flush=True)
        sys.exit(1)
    ''')
    res = bootstrap.selftest("official", "gfx1102", child=child)
    assert res.verdict == "rung_failed" and res.failed_rung == "device"
    assert "sees zero devices" in res.detail
    assert "failed at rung 'device'" in res.message()


def test_a_pass_verdict_is_read_off_the_line_not_the_exit_status(tmp_path):
    """TheRock's atexit _exit(0) clobbers exit codes once HIP is up, so the
    verdict line is the contract — and a child that exits 0 WITHOUT one is
    not a pass."""
    lines = "".join(f'print("rung {i}/7 {n}: PASS — ok", flush=True)\n' for i, n in enumerate(RUNGS, 1))
    child = _stub(tmp_path, "import sys\n" + lines + 'print("SELFTEST PASS", flush=True)\nsys.exit(0)\n')
    res = bootstrap.selftest("therock", "gfx1151", child=child)
    assert res.ok and len(res.rungs) == len(RUNGS)
    assert res.message().startswith("self-test passed: 7/7 rungs")

    child = _stub(tmp_path, 'import sys\nprint("rung 1/7 import: PASS — ok", flush=True)\nsys.exit(0)\n')
    res = bootstrap.selftest("therock", "gfx1151", child=child)
    assert res.verdict == "rung_failed" and res.failed_rung == "device"
    assert "without a verdict" in res.detail


def test_a_hung_child_is_killed_and_typed_as_a_timeout(tmp_path):
    child = _stub(tmp_path, '''
        import time
        print("rung 1/7 import: PASS — ok", flush=True)
        time.sleep(60)
    ''')
    res = bootstrap.selftest("official", "gfx1102", child=child, timeout=1.0)
    assert res.verdict == "timeout" and res.failed_rung == "device"


def test_an_interpreter_that_cannot_start_is_a_result_not_an_exception(tmp_path):
    res = bootstrap.selftest("official", "gfx1102", child=[str(tmp_path / "no-such-python")])
    assert res.verdict == "no_interpreter" and "no-such-python" in res.detail


def test_after_a_cannot_launch_verdict_a_carried_target_with_a_family_gets_the_therock_line():
    choice = bootstrap.choose_lane("Linux", False, True, "gfx1201")
    res = SelfTestResult(verdict="lane_cannot_launch", lane="official", gfx="gfx1201",
                         python="py", signal="SIGSEGV", exit_code=-11, failed_rung="kernel-launch",
                         rungs=["r1", "r2", "r3"])
    text = "\n".join(bootstrap.after_selftest(choice, res))
    assert "died of SIGSEGV" in text
    assert "measured just now: SIGSEGV at rung 'kernel-launch'" in text
    assert "https://rocm.nightlies.amd.com/v2/gfx120X-all/" in text
    assert "DRINKME_NO_AUTO_DEPS=1" in text


def test_after_a_cannot_launch_verdict_strix_point_is_refused_by_name():
    choice = bootstrap.choose_lane("Linux", False, True, "gfx1150")
    res = SelfTestResult(verdict="lane_cannot_launch", lane="official", gfx="gfx1150",
                         python="py", signal="SIGSEGV", exit_code=139, failed_rung="kernel-launch")
    text = "\n".join(bootstrap.after_selftest(choice, res))
    assert "Radeon 890M (gfx1150): no other lane drinkme can install" in text
    assert "rocm.nightlies" not in text


def test_after_a_pass_there_is_only_the_pass_line():
    choice = bootstrap.choose_lane("Linux", False, True, "gfx1151")
    res = SelfTestResult(verdict="ok", lane="therock", gfx="gfx1151", python="py",
                         rungs=[f"rung {i}/7 {n}: PASS — ok" for i, n in enumerate(RUNGS, 1)])
    assert bootstrap.after_selftest(choice, res) == [res.message()]


def _gpu_visible() -> bool:
    """The real end-to-end below needs a device the suite's own interpreter
    can see; a masked HIP/CUDA_VISIBLE_DEVICES (the CPU-only suite) or a box
    with no accelerator skips it rather than failing it."""
    if os.environ.get("HIP_VISIBLE_DEVICES") == "" or os.environ.get("CUDA_VISIBLE_DEVICES") == "":
        return False
    return os.path.exists("/dev/kfd") or os.path.exists("/dev/nvidiactl")


@pytest.mark.skipif(not _gpu_visible(), reason="no visible accelerator for the real self-test")
def test_the_real_self_test_passes_on_this_box():
    """End to end against the interpreter running the suite (the project
    venv per AGENTS.md): every rung, the way `drinkme bootstrap` runs it.
    A torch-less or device-less interpreter skips at its first rung; any
    later failure is a real one."""
    lane = "therock" if os.path.exists("/dev/kfd") else "cuda"
    gfx = bootstrap.amd_gfx_target()[0] if lane == "therock" else None
    res = bootstrap.selftest(lane, gfx, python=sys.executable)
    if res.verdict == "rung_failed" and res.failed_rung in ("import", "device"):
        pytest.skip(f"this interpreter cannot run the self-test: {res.detail}")
    assert res.ok, res.output
    assert len(res.rungs) == len(RUNGS)
    for i, name in enumerate(RUNGS, 1):
        assert res.rungs[i - 1].startswith(f"rung {i}/{len(RUNGS)} {name}: PASS — ")
    assert "kernel-launch: PASS" in res.rungs[3]
    assert "decode bit-exact" in res.rungs[6]


# ----------------------------------------------- ensure_accelerator: metal --


def _metal_box(monkeypatch, tmp_path):
    monkeypatch.delenv("DRINKME_NO_AUTO_DEPS", raising=False)
    monkeypatch.delenv("DRINKME_DEVICE", raising=False)
    monkeypatch.setattr(bootstrap, "is_metal_box", lambda: True)
    monkeypatch.setattr(bootstrap, "project_root", lambda: tmp_path)
    monkeypatch.setattr(bootstrap.shutil, "which", lambda n: "/usr/bin/uv")
    monkeypatch.setattr(bootstrap, "detect_lane",
                        lambda **k: pytest.fail("the metal lane never probes for torch"))


def test_a_mac_with_no_mlx_installs_the_metal_group_and_never_looks_at_torch(
        monkeypatch, capsys, tmp_path):
    """The lane is decided by the platform; torch's presence or absence is
    not consulted at all — a Mac with a stray torch must not be 'healed'
    onto the cuda group."""
    _metal_box(monkeypatch, tmp_path)
    seen = {}
    monkeypatch.setattr(bootstrap.importlib.util, "find_spec", lambda n: None)
    monkeypatch.setattr(bootstrap.subprocess, "call",
                        lambda cmd, cwd=None: seen.update(cmd=cmd, cwd=cwd) or 0)
    monkeypatch.setattr(bootstrap.os, "execve",
                        lambda *a: pytest.fail("re-exec was not needed"))
    bootstrap.ensure_accelerator()
    assert seen["cmd"][1:] == ["sync", "--no-default-groups", "--group", "metal"]
    assert seen["cwd"] == str(tmp_path)
    err = capsys.readouterr().err
    assert "mlx is not installed yet" in err
    assert "Apple silicon" in err and "DRINKME_NO_AUTO_DEPS" in err  # the banner + the door


def test_a_mac_with_a_working_mlx_is_left_alone(monkeypatch, capsys, tmp_path):
    _metal_box(monkeypatch, tmp_path)
    monkeypatch.setattr(bootstrap.importlib.util, "find_spec",
                        lambda n: object() if n == "mlx" else None)
    monkeypatch.setitem(sys.modules, "mlx", types.ModuleType("mlx"))
    monkeypatch.setitem(sys.modules, "mlx.core", _mlx(available=True))
    monkeypatch.setattr(bootstrap.subprocess, "call",
                        lambda *a, **k: pytest.fail("touched uv on a good Mac"))
    bootstrap.ensure_accelerator()
    assert capsys.readouterr().err == ""


def test_the_metal_lane_honours_the_opt_out(monkeypatch, tmp_path):
    _metal_box(monkeypatch, tmp_path)
    monkeypatch.setenv("DRINKME_NO_AUTO_DEPS", "1")
    monkeypatch.setattr(bootstrap.importlib.util, "find_spec",
                        lambda n: pytest.fail("probed despite opt-out"))
    monkeypatch.setattr(bootstrap.subprocess, "call",
                        lambda *a, **k: pytest.fail("ran uv despite opt-out"))
    bootstrap.ensure_accelerator()


_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_APPLE_SILICON = "platform_machine == 'arm64' and sys_platform == 'darwin'"


def _toml(name):
    tomllib = pytest.importorskip("tomllib")
    with open(os.path.join(_ROOT, name), "rb") as f:
        return tomllib.load(f)


def test_the_metal_group_carries_torch_for_packing():
    """`serve` packs on first use and pack_model reads the checkpoint
    through torch, so the lane a Mac's first run installs must carry it,
    or the quickstart stops at "packing needs torch". Every entry is marked
    Apple-silicon-only, so the group resolves to nothing elsewhere."""
    from packaging.requirements import Requirement

    reqs = [Requirement(r) for r in _toml("pyproject.toml")["dependency-groups"]["metal"]]
    assert sorted(r.name for r in reqs) == ["mlx", "mlx-lm", "torch"]
    for r in reqs:
        for machine, system, expect in (("arm64", "Darwin", True), ("x86_64", "Darwin", False),
                                        ("aarch64", "Linux", False), ("x86_64", "Linux", False)):
            env = {"platform_machine": machine, "sys_platform": system.lower()}
            assert r.marker.evaluate(env) is expect, (r, machine, system)


def test_the_lock_matches_the_metal_group():
    """A lock behind pyproject makes a stranger's first `uv sync` re-resolve
    the shared lock; the committed one must already carry the metal group's
    torch, as the same PyPI build the cuda group locks."""
    lock = _toml("uv.lock")
    me = next(p for p in lock["package"] if p["name"] == "drinkme")
    assert "torch" in {d["name"] for d in me["metadata"]["requires-dev"]["metal"]}
    locked = [d for d in me["dev-dependencies"]["metal"] if d["name"] == "torch"]
    assert [d["marker"] for d in locked] == [_APPLE_SILICON]
    cuda = [d for d in me["dev-dependencies"]["cuda"] if d["name"] == "torch"]
    assert {(d.get("version"), d["source"]["registry"]) for d in locked} \
        == {(d.get("version"), d["source"]["registry"]) for d in cuda}


# ------------------------------------------------------- ensure_accelerator --


def _healthy_host():
    return bootstrap.HostProbe(system="Linux", ok=True, gfx="gfx1151", gfx_source="sysfs kfd",
                               amdgpu_loaded=True, kfd="rw (root:render 0660)", has_amd=True)


def _passing_selftest(lane, gfx=None, **k):
    return SelfTestResult(verdict="ok", lane=lane, gfx=gfx, python="py",
                          rungs=[f"rung {i}/7 {n}: PASS — ok" for i, n in enumerate(RUNGS, 1)])


def _fresh_box(monkeypatch, tmp_path, choice: LaneChoice, host=None, selftest=None):
    """A Linux box (pinned: on a Mac ensure_accelerator takes the metal
    lane before it asks detect_lane) that has not installed `choice` yet."""
    from hosts import pin_linux_host

    pin_linux_host(monkeypatch)
    monkeypatch.delenv("DRINKME_NO_AUTO_DEPS", raising=False)
    monkeypatch.delenv("DRINKME_DEVICE", raising=False)
    monkeypatch.setattr(bootstrap.importlib.util, "find_spec", lambda n: None)
    monkeypatch.setattr(bootstrap, "detect_lane", lambda **k: choice)
    monkeypatch.setattr(bootstrap, "probe_host", lambda **k: host or _healthy_host())
    monkeypatch.setattr(bootstrap, "selftest", selftest or _passing_selftest)
    monkeypatch.setattr(bootstrap, "project_root", lambda: tmp_path)
    monkeypatch.setattr(bootstrap.shutil, "which", lambda n: "/usr/bin/uv")
    monkeypatch.setattr(bootstrap.os, "execve",
                        lambda *a: pytest.fail("re-exec was not needed"))


def test_opt_outs_are_honored_before_anything_is_probed(monkeypatch):
    """Both env doors must short-circuit ahead of any import or subprocess —
    an operator who said 'don't' gets no side effects at all."""
    for var in ("DRINKME_NO_AUTO_DEPS", "DRINKME_DEVICE"):
        monkeypatch.delenv("DRINKME_NO_AUTO_DEPS", raising=False)
        monkeypatch.delenv("DRINKME_DEVICE", raising=False)
        monkeypatch.setenv(var, "1")
        monkeypatch.setattr(bootstrap, "detect_lane",
                            lambda **k: pytest.fail("probed despite opt-out"))
        monkeypatch.setattr(bootstrap.subprocess, "call",
                            lambda *a, **k: pytest.fail("ran uv despite opt-out"))
        bootstrap.ensure_accelerator(stream=sys.stderr)


def test_a_healthy_environment_is_left_completely_alone(monkeypatch, capsys):
    """The normal case, and the one that must never cost anything: no uv, no
    output, no re-exec, no self-test."""
    monkeypatch.delenv("DRINKME_NO_AUTO_DEPS", raising=False)
    monkeypatch.delenv("DRINKME_DEVICE", raising=False)
    monkeypatch.setattr(bootstrap, "classify_torch", lambda m: ("ok", "fine"))
    monkeypatch.setattr(bootstrap.subprocess, "call",
                        lambda *a, **k: pytest.fail("touched uv on a good box"))
    monkeypatch.setattr(bootstrap, "selftest",
                        lambda *a, **k: pytest.fail("self-tested a good box"))
    bootstrap.ensure_accelerator()
    assert capsys.readouterr().err == ""


def test_it_refuses_to_drive_uv_outside_a_drinkme_checkout(monkeypatch, capsys, tmp_path):
    """project_root() checks pyproject's own name — we must never run
    `uv sync` in whatever directory the user happens to be standing in."""
    _fresh_box(monkeypatch, tmp_path, bootstrap.choose_lane("Linux", False, True, "gfx1151"))
    monkeypatch.setattr(bootstrap, "project_root", lambda: None)
    monkeypatch.setattr(bootstrap.subprocess, "call",
                        lambda *a, **k: pytest.fail("ran uv outside a checkout"))
    bootstrap.ensure_accelerator()
    err = capsys.readouterr().err
    assert "not a uv checkout" in err
    assert "--group rocm" in err  # still tells them the command to run


def test_a_refused_box_explains_and_installs_nothing(monkeypatch, capsys, tmp_path):
    _fresh_box(monkeypatch, tmp_path, bootstrap.choose_lane("Linux", False, True, "gfx1010"))
    monkeypatch.setattr(bootstrap.subprocess, "call",
                        lambda *a, **k: pytest.fail("installed a guess"))
    bootstrap.ensure_accelerator()
    err = capsys.readouterr().err
    assert "no lane drinkme can install" in err and "gfx1010" in err


def test_a_line_target_is_probed_then_handed_the_command(monkeypatch, capsys, tmp_path):
    _fresh_box(monkeypatch, tmp_path, bootstrap.choose_lane("Linux", False, True, "gfx941"))
    monkeypatch.setattr(bootstrap.subprocess, "call",
                        lambda *a, **k: pytest.fail("installed a guess"))
    bootstrap.ensure_accelerator()
    err = capsys.readouterr().err
    assert "host: Linux" in err
    assert "uv pip install --index-url https://rocm.nightlies.amd.com/v2/gfx94X-dcgpu/" in err


def test_the_host_probe_refuses_before_the_sync(monkeypatch, capsys, tmp_path):
    """A user outside the render group: the wheel is never fetched."""
    host = bootstrap.HostProbe(system="Linux", ok=False, gfx="gfx1102", gfx_source="sysfs kfd",
                               amdgpu_loaded=True, kfd="no access (root:render 0660)", has_amd=True,
                               refusal="/dev/kfd no access: user u is not in the group (render)",
                               fix=["sudo usermod -aG render $USER"])
    _fresh_box(monkeypatch, tmp_path, bootstrap.choose_lane("Linux", False, True, "gfx1102"), host=host)
    monkeypatch.setattr(bootstrap.subprocess, "call",
                        lambda *a, **k: pytest.fail("downloaded before the probe passed"))
    bootstrap.ensure_accelerator()
    err = capsys.readouterr().err
    assert "refused before downloading anything" in err
    assert "sudo usermod -aG render $USER" in err


def test_missing_torch_installs_the_lane_then_self_tests_and_does_not_reexec(
        monkeypatch, capsys, tmp_path):
    """torch was never imported, so a fresh import finds the new one —
    re-exec is reserved for the broken case, where the wrong module is
    already in the process's table. The self-test runs after the sync, in
    a child, and its pass line is printed."""
    seen = {}
    _fresh_box(monkeypatch, tmp_path, bootstrap.choose_lane("Linux", False, True, "gfx1102"))
    monkeypatch.setattr(bootstrap.subprocess, "call",
                        lambda cmd, cwd=None: seen.update(cmd=cmd, cwd=cwd) or 0)
    bootstrap.ensure_accelerator()
    assert seen["cmd"][1:] == ["sync", "--no-default-groups", "--group", "rocm-official"]
    assert seen["cwd"] == str(tmp_path)
    err = capsys.readouterr().err
    assert "self-test passed: 7/7 rungs" in err


def test_a_cannot_launch_verdict_after_the_sync_prints_the_next_lane_and_stops(
        monkeypatch, capsys, tmp_path):
    """The flow the two lanes exist for: the official wheel installs on an
    RDNA4 card, the child segfaults at the first launch, and the TheRock
    line is the next thing printed — with the verdict quoted, no re-exec,
    no second install."""
    def crashing(lane, gfx=None, **k):
        return SelfTestResult(verdict="lane_cannot_launch", lane=lane, gfx=gfx, python="py",
                              signal="SIGSEGV", exit_code=-11, failed_rung="kernel-launch",
                              rungs=["rung 1/7 import: PASS — t", "rung 2/7 device: PASS — d",
                                     "rung 3/7 arch-list: PASS — a"])
    _fresh_box(monkeypatch, tmp_path, bootstrap.choose_lane("Linux", False, True, "gfx1201"),
               selftest=crashing)
    calls = []
    monkeypatch.setattr(bootstrap.subprocess, "call", lambda cmd, cwd=None: calls.append(cmd) or 0)
    bootstrap.ensure_accelerator()
    assert len(calls) == 1
    err = capsys.readouterr().err
    assert "died of SIGSEGV" in err
    assert "https://rocm.nightlies.amd.com/v2/gfx120X-all/" in err


def test_a_cpu_box_installs_the_wheel_and_runs_no_self_test(monkeypatch, capsys, tmp_path):
    """No accelerator: the PyPI wheel is the honest lane and there is
    nothing to launch on — a self-test would fail at rung 2 by design and
    say nothing true about the install."""
    choice = bootstrap.choose_lane("Linux", False, False, None)
    assert choice.group == "cuda" and choice.selftest is False
    _fresh_box(monkeypatch, tmp_path, choice)
    monkeypatch.setattr(bootstrap.subprocess, "call", lambda cmd, cwd=None: 0)
    monkeypatch.setattr(bootstrap, "selftest",
                        lambda *a, **k: pytest.fail("self-tested a CPU box"))
    bootstrap.ensure_accelerator()
    assert "--group cuda" in capsys.readouterr().err


def test_a_failed_sync_says_so_and_stops(monkeypatch, capsys, tmp_path):
    _fresh_box(monkeypatch, tmp_path, bootstrap.choose_lane("Linux", True, False, None))
    monkeypatch.setattr(bootstrap.subprocess, "call", lambda cmd, cwd=None: 2)
    monkeypatch.setattr(bootstrap, "selftest",
                        lambda *a, **k: pytest.fail("self-tested after a failed sync"))
    bootstrap.ensure_accelerator()
    assert "exit 2" in capsys.readouterr().err


def test_the_marker_stops_a_reexec_loop(monkeypatch, capsys, tmp_path):
    """If a sync doesn't actually fix it, re-execing forever is worse than
    stopping — the loop would be invisible and eat the machine."""
    _fresh_box(monkeypatch, tmp_path, bootstrap.choose_lane("Linux", False, True, "gfx1151"))
    monkeypatch.setenv(bootstrap._MARKER, "1")
    monkeypatch.setattr(bootstrap.importlib.util, "find_spec", lambda n: object())
    monkeypatch.setitem(sys.modules, "torch", _torch("2.13.0+cu130", cuda="13.0"))
    monkeypatch.setattr(bootstrap.subprocess, "call", lambda cmd, cwd=None: 0)
    monkeypatch.setattr(bootstrap.os, "execve",
                        lambda *a: pytest.fail("looped after the marker was set"))
    bootstrap.ensure_accelerator()
    assert "rather than looping" in capsys.readouterr().err


def test_project_root_rejects_a_foreign_pyproject(monkeypatch, tmp_path):
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "someone-else"\n')
    monkeypatch.setattr(bootstrap.sys, "prefix", str(tmp_path / ".venv"))
    assert bootstrap.project_root() is None


def test_project_root_finds_a_real_drinkme_checkout(monkeypatch, tmp_path):
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "drinkme"\n')
    monkeypatch.setattr(bootstrap.sys, "prefix", str(tmp_path / ".venv"))
    assert bootstrap.project_root() == tmp_path


# ----------------------------------------------------- `drinkme bootstrap` --


def _cli_box(monkeypatch, gfx="gfx1151", host=None):
    """The real detect_lane, over a fake box: the gfx override rides
    through --gfx, the host probe is a healthy fake. The box is Linux on
    x86_64, pinned where detect_lane reads it: on a Mac it answers the
    metal lane before it looks at any AMD part."""
    from hosts import pin_linux_host

    from drinkme import detect

    pin_linux_host(monkeypatch)
    monkeypatch.setattr(detect, "_amd_card_dirs", lambda: ["/fake/card0/device"])
    monkeypatch.setattr(detect, "_detect_nvidia", lambda: None)
    monkeypatch.setattr(detect, "_detect_amd", lambda: None)
    monkeypatch.setattr(bootstrap, "amd_gfx_target", lambda *a, **k: (gfx, "sysfs kfd"))
    monkeypatch.setattr(bootstrap, "probe_host", lambda **k: host or _healthy_host())
    monkeypatch.setattr(bootstrap.subprocess, "call",
                        lambda *a, **k: pytest.fail("a dry run installed something"))
    monkeypatch.setattr(bootstrap, "selftest",
                        lambda *a, **k: pytest.fail("a dry run ran the self-test"))


@pytest.mark.parametrize("gfx,rc,lane_line,extra", [
    ("gfx1102", 0, "lane: official (verified)", "--group rocm-official"),
    ("gfx1201", 0, "lane: official (carried)", "rocm.nightlies.amd.com/v2/gfx120X-all/"),
    ("gfx1150", 0, "lane: official (carried)", "no other lane for this target"),
    ("gfx1010", 4, "lane: none (refused)", "TheRock publishes no torch"),
])
def test_dry_run_prints_the_plan_for_the_four_fake_boxes_and_installs_nothing(
        monkeypatch, capsys, gfx, rc, lane_line, extra):
    from drinkme.cli import main

    _cli_box(monkeypatch)
    assert main(["bootstrap", "--dry-run", "--gfx", gfx]) == rc
    err = capsys.readouterr().err
    assert lane_line in err
    assert extra in err
    assert f"--gfx {gfx} overrides" in err
    assert "verdict: dry run — nothing installed" in err


def test_gfx_without_a_dry_run_is_refused_so_a_pretence_cannot_install(monkeypatch, capsys):
    from drinkme.cli import main

    _cli_box(monkeypatch)
    assert main(["bootstrap", "--gfx", "gfx1201"]) == 2
    assert "only honoured with --dry-run" in capsys.readouterr().err


def test_detect_only_prints_the_probe_and_the_lane(monkeypatch, capsys):
    from drinkme.cli import main

    _cli_box(monkeypatch)
    assert main(["bootstrap", "--detect-only"]) == 0
    err = capsys.readouterr().err
    assert "host: Linux" in err and "lane: therock (verified)" in err
    assert "verdict: detect only" in err


def test_a_refusing_probe_stops_a_dry_run_too(monkeypatch, capsys):
    from drinkme.cli import main

    host = bootstrap.HostProbe(system="Linux", ok=False, gfx="gfx1151", gfx_source="sysfs kfd",
                               amdgpu_loaded=True, kfd="no access (root:render 0660)", has_amd=True,
                               refusal="/dev/kfd no access", fix=["sudo usermod -aG render $USER"])
    _cli_box(monkeypatch, host=host)
    assert main(["bootstrap", "--dry-run"]) == 4
    err = capsys.readouterr().err
    assert "refused before downloading anything" in err
    assert "verdict: refused before downloading anything" in err


def test_the_full_flow_installs_then_self_tests(monkeypatch, capsys, tmp_path):
    from drinkme.cli import main

    _cli_box(monkeypatch)
    calls = []
    monkeypatch.setattr(bootstrap, "project_root", lambda: tmp_path)
    monkeypatch.setattr(bootstrap.shutil, "which", lambda n: "/usr/bin/uv")
    monkeypatch.setattr(bootstrap.subprocess, "call", lambda cmd, cwd=None: calls.append(cmd) or 0)
    monkeypatch.setattr(bootstrap, "selftest", _passing_selftest)
    assert main(["bootstrap"]) == 0
    assert calls[0][1:] == ["sync", "--no-default-groups", "--group", "rocm"]
    err = capsys.readouterr().err
    assert "self-test passed: 7/7 rungs" in err and "verdict: ok" in err
