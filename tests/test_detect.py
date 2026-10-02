import socket

from drinkme import detect
from drinkme.detect import Hardware, _APPLE_BUDGET, amd_is_unified, force_ipv4


# ------------------------------------------------------- discrete naming --
#
# An RX 7600 XT (gfx1102): `gpuInfo` carried the pci id fine but
# rocm-smi returned nothing on that box, so a `device=gpu_name` read from
# rocm-smi alone comes out None and `bench` writes a record slugged
# "unknown-device". These
# fake a `/sys/class/drm/cardN/device` tree (never touching the real one) so
# the three paths — table hit, live-system hit, and the pci-id fallback —
# are each reachable without a GPU.

_DEV = "/fake/card0/device"


def _fake_amd_sysfs(monkeypatch, files: dict, rocm_smi: str | None = None,
                    rocminfo: str | None = None):
    monkeypatch.setattr(detect, "_amd_card_dirs", lambda: [_DEV])
    monkeypatch.setattr(detect, "_read", lambda path: files.get(path))

    def fake_run(cmd):
        if cmd[0] == "rocm-smi":
            return rocm_smi
        if cmd[0] == "rocminfo":
            return rocminfo
        return None

    monkeypatch.setattr(detect, "_run", fake_run)
    monkeypatch.setattr(detect, "_cpu_brand_linux", lambda: "AMD Ryzen 7 5700X 8-Core Processor")
    monkeypatch.setattr(detect, "_os_linux", lambda: "CachyOS")


def _rx7600xt_files(device="0x7480", revision="0xc0"):
    # 16GB VRAM, 31.4GB GTT: vram-dominant -> discrete (amd_is_unified false).
    return {
        f"{_DEV}/vendor": "0x1002",
        f"{_DEV}/mem_info_vram_total": str(16 * 1024**3),
        f"{_DEV}/mem_info_gtt_total": str(int(31.4 * 1024**3)),
        f"{_DEV}/device": device,
        f"{_DEV}/revision": revision,
    }


def test_amd_discrete_table_hit_names_the_card(monkeypatch):
    """The exact RX 7600 XT id (0x7480, RX 7600 XT die) with a revision the
    table recognizes: named from the static table, not a guess."""
    _fake_amd_sysfs(monkeypatch, _rx7600xt_files())
    hw = detect._detect_amd()
    assert hw.device_class == "Radeon RX 7600 XT"
    assert hw.device_source == "table"
    assert hw.heuristic is False
    assert "pci id table: 0x7480" in " ".join(hw.evidence)


def test_amd_discrete_table_hit_unknown_revision_uses_default(monkeypatch):
    _fake_amd_sysfs(monkeypatch, _rx7600xt_files(revision="0xff"))
    hw = detect._detect_amd()
    assert hw.device_class == "Radeon RX 7600 XT"  # the table's "default" for 0x7480
    assert hw.device_source == "table"


def test_amd_discrete_sysfs_hit_when_pci_id_is_not_in_the_table(monkeypatch):
    """An id the static table has never seen, but the running system can
    still name it — sysfs product_name wins over rocminfo/rocm-smi."""
    files = _rx7600xt_files(device="0xbeef", revision="0x00")
    files[f"{_DEV}/product_name"] = "AMD Radeon Test Card XT"
    _fake_amd_sysfs(monkeypatch, files)
    hw = detect._detect_amd()
    assert hw.device_class == "AMD Radeon Test Card XT"
    assert hw.device_source == "sysfs"
    assert hw.heuristic is False


def test_amd_discrete_rocminfo_marketing_name_when_sysfs_is_silent(monkeypatch):
    files = _rx7600xt_files(device="0xbeef", revision="0x00")
    _fake_amd_sysfs(monkeypatch, files, rocminfo="Agent 1\n  Marketing Name:  Radeon Test Card XT  \n")
    hw = detect._detect_amd()
    assert hw.device_class == "Radeon Test Card XT"
    assert hw.device_source == "sysfs"


def test_amd_discrete_falls_back_to_pci_id_never_none(monkeypatch):
    """The RX 7600 XT bug, reproduced: an id neither table nor any live source
    can name. `device` must still be a non-None string carrying the pci id
    (never the word 'unknown') and `heuristic` must flag the uncertainty."""
    _fake_amd_sysfs(monkeypatch, _rx7600xt_files(device="0xdead", revision="0x00"))
    hw = detect._detect_amd()
    assert hw.device_class == "amdgpu 0xdead"
    assert hw.device_source == "fallback"
    assert hw.heuristic is True
    assert "unknown" not in hw.device_class.lower()


def test_amd_unified_apu_device_source_is_cpu_brand(monkeypatch):
    files = {
        f"{_DEV}/vendor": "0x1002",
        f"{_DEV}/mem_info_vram_total": str(int(0.5 * 1024**3)),
        f"{_DEV}/mem_info_gtt_total": str(124 * 1024**3),
        f"{_DEV}/device": "0x1586",
        f"{_DEV}/revision": "0xc1",
    }
    _fake_amd_sysfs(monkeypatch, files)
    hw = detect._detect_amd()
    assert hw.memory_kind == "unified"
    assert hw.device_source == "cpu-brand"
    assert hw.device_class == "AMD Ryzen 7 5700X 8-Core Processor"


def test_table_name_pure_lookup():
    assert detect._table_name("0x7480", "0xc0") == ("Radeon RX 7600 XT", "gfx1102")
    assert detect._table_name("0x1586", None) == ("Radeon 8060S / Strix Halo 395", "gfx1151")
    assert detect._table_name("0xffff", None) is None
    assert detect._table_name(None, None) is None


def test_apu_heuristic_strix():
    # Strix Halo, post-BIOS-reflash: 0.5GB carveout, 124GB GTT.
    assert amd_is_unified(0.5, 124.0) is True


def test_apu_heuristic_discrete():
    # RX 7600 XT: 16GB VRAM, GTT ~ half of system RAM but smaller than 2x VRAM
    # on typical boxes; the carveout pattern is what marks unified.
    assert amd_is_unified(16.0, 15.9) is False


def test_apu_heuristic_missing_data_defaults_discrete():
    assert amd_is_unified(None, 124.0) is False
    assert amd_is_unified(16.0, None) is False


def test_mac_8gb_budget_is_the_measured_default():
    # The ~5.4GB default working set is what makes the 4B cliff experiment.
    assert _APPLE_BUDGET[8] == 5.4


def test_record_env_carries_verbatim_and_version():
    hw = Hardware(
        device_class="AMD RYZEN AI MAX+ 395 w/ Radeon 8060S",
        memory_gb=128.0,
        memory_kind="unified",
        budget_gb=124.0,
        memory_bytes=128_000_000_000,
        budget_bytes=124_000_000_000,
        cpu_info="AMD RYZEN AI MAX+ 395 w/ Radeon 8060S",
        gpu_info="amdgpu 0x1586 vram=0.5GB gtt=124.0GB",
        heuristic=True,
    )
    env = hw.as_record_env()
    assert env["gpuInfo"].startswith("amdgpu")  # verbatim survives
    # The lexicon's #environment shape, not the dataclass's: memoryBytes is
    # the detector's own byte count (an integer), memoryKind the enum, and an
    # unknown optional (os_info was left empty here) is absent rather than
    # null. device_source is not here (bench keeps it in raw).
    assert env["memoryBytes"] == 128_000_000_000 and type(env["memoryBytes"]) is int
    assert "memoryGB" not in env and "device_source" not in env and "deviceSource" not in env
    assert env["memoryKind"] == "unified"
    assert "os" not in env and "memory_gb" not in env
    # the detector's working notes are not lexicon fields: they go to raw
    for k in ("budgetGB", "heuristic", "evidence", "detectorVersion"):
        assert k not in env
    notes = hw.as_record_detector()
    assert notes == {"budgetBytes": 124_000_000_000, "heuristic": True, "evidence": [], "version": 2}


def test_force_ipv4_pins_family(monkeypatch):
    seen = {}
    real = socket.getaddrinfo

    def spy(host, port, family=0, type=0, proto=0, flags=0):
        seen["family"] = family
        return real(host, port, family, type, proto, flags)

    monkeypatch.setattr(socket, "getaddrinfo", spy)
    force_ipv4()
    socket.getaddrinfo("localhost", 80)
    assert seen["family"] == socket.AF_INET
    monkeypatch.setattr(socket, "getaddrinfo", real)
