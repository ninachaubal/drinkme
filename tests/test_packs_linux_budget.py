"""packs.hardware_budget on Linux: the smaller of detect()'s capacity
(GTT on an APU, VRAM on a discrete card) and what is free of it right now
(detect.live_free_bytes), so a box already running a model is not offered a
model sized for an empty one. Seen 09-28 on a Strix Halo with a 27B resident:
'budget: GTT 124.0 GiB' and a 98 GiB 72B proposed. Every reading is mocked."""

import pytest

from drinkme import detect, packs
from drinkme.detect import Hardware

pytestmark = pytest.mark.usefixtures("linux_host")

GIB = 1024 ** 3


def apu(gtt_gib=124.0):
    b = int(gtt_gib * GIB)
    return Hardware(device_class="AMD RYZEN AI MAX+ 395 w/ Radeon 8060S", memory_gb=b / 1e9,
                    memory_kind="unified", budget_gb=b / 1e9, budget_bytes=b, cpu_info=None,
                    gpu_info=None, device_source="cpu-brand", evidence=["vram_total=0.54GB"])


def nvidia(vram_gib=48.0):
    b = int(vram_gib * GIB)
    return Hardware(device_class="NVIDIA L40S", memory_gb=b / 1e9, memory_kind="vram",
                    budget_gb=b / 1e9, budget_bytes=b, cpu_info=None, gpu_info=None,
                    evidence=["nvidia-smi: NVIDIA L40S, 46068"])


def amd_dgpu(vram_gib=16.0):
    b = int(vram_gib * GIB)
    return Hardware(device_class="Radeon RX 7600 XT", memory_gb=b / 1e9, memory_kind="vram",
                    budget_gb=b / 1e9, budget_bytes=b, cpu_info=None, gpu_info=None,
                    device_source="table", evidence=["vram_total=17.16GB"])


def budget_with(monkeypatch, hw, free):
    monkeypatch.setattr(detect, "detect", lambda: hw)
    monkeypatch.setattr(detect, "live_free_bytes", lambda h: free)
    return packs.hardware_budget()


# ---------------------------------------------------------------- the budget --

def test_a_busy_apu_budgets_what_is_free_and_says_so(monkeypatch):
    gib, source = budget_with(monkeypatch, apu(124.0), (int(61.5 * GIB), "MemAvailable"))
    assert gib == pytest.approx(61.5)
    assert source == "MemAvailable 61.5 GiB, of GTT 124.0 GiB"


def test_an_idle_box_keeps_its_capacity_and_shows_the_reading(monkeypatch):
    gib, source = budget_with(monkeypatch, nvidia(48.0), (int(47.9 * GIB), "free VRAM"))
    assert gib == pytest.approx(47.9)  # a driver's own reservation is still less than total
    gib, source = budget_with(monkeypatch, nvidia(48.0), (int(48.0 * GIB), "free VRAM"))
    assert gib == pytest.approx(48.0)
    assert source == "VRAM 48.0 GiB (free VRAM 48.0 GiB)"


def test_an_unread_reading_keeps_the_capacity_and_names_the_miss(monkeypatch):
    gib, source = budget_with(monkeypatch, amd_dgpu(16.0), (None, "free VRAM unread"))
    assert gib == pytest.approx(16.0)
    assert source == "VRAM 16.0 GiB (free VRAM unread)"


def test_a_unified_nvidia_part_is_not_called_gtt(monkeypatch):
    hw = nvidia(119.0)
    hw.memory_kind = "unified"  # GB10 / Jetson
    _, source = budget_with(monkeypatch, hw, (int(100 * GIB), "MemAvailable"))
    assert source == "MemAvailable 100.0 GiB, of unified memory 119.0 GiB"


def cand(name, gib):
    return packs.Candidate(name=name, hf_repo=f"Qwen/{name}", revision=None, resident_gib=gib,
                           packed=False, pack_dir=None, measured=False, config=None,
                           mtp_head_gib=None)


def test_the_picker_offers_a_smaller_model_when_memory_is_taken(monkeypatch):
    """The 09-28 case through rank_fits: the 72B (98.45 GiB, from that day's
    summary) led on the empty 124 GiB box and no longer fits with ~61 GiB
    free; the 32B (44.36 GiB) leads instead."""
    menu = [cand("Qwen2.5-72B", 98.45), cand("Qwen3-32B", 44.36), cand("Qwen3-8B", 11.09)]
    idle, _ = budget_with(monkeypatch, apu(124.0), (int(124.0 * GIB), "MemAvailable"))
    assert [f.candidate.name for f in packs.rank_fits(idle, menu, 8192, 1) if f.fits][0] == "Qwen2.5-72B"
    busy, _ = budget_with(monkeypatch, apu(124.0), (int(61.5 * GIB), "MemAvailable"))
    assert [f.candidate.name for f in packs.rank_fits(busy, menu, 8192, 1) if f.fits][0] == "Qwen3-32B"


# ----------------------------------------------------------- live_free_bytes --

def test_unified_reads_memavailable(monkeypatch, tmp_path):
    p = tmp_path / "meminfo"
    p.write_text("MemTotal:       131072000 kB\nMemFree:          1000 kB\n"
                 "MemAvailable:    64000000 kB\n")
    monkeypatch.setattr(detect, "_MEMINFO", str(p))
    assert detect.live_free_bytes(apu()) == (64000000 * 1024, "MemAvailable")


def test_unified_without_the_line_is_unread(monkeypatch, tmp_path):
    p = tmp_path / "meminfo"
    p.write_text("MemTotal:       131072000 kB\n")
    monkeypatch.setattr(detect, "_MEMINFO", str(p))
    assert detect.live_free_bytes(apu()) == (None, "MemAvailable unread")


def test_nvidia_reads_nvidia_smi_memory_free_in_mib(monkeypatch):
    calls = []
    monkeypatch.setattr(detect, "_run", lambda cmd: calls.append(cmd) or "40960\n40960")
    assert detect.live_free_bytes(nvidia()) == (40960 * 1024 ** 2, "free VRAM")
    assert calls == [["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"]]


def test_nvidia_smi_failing_is_unread(monkeypatch):
    monkeypatch.setattr(detect, "_run", lambda cmd: None)
    assert detect.live_free_bytes(nvidia()) == (None, "free VRAM unread")


def test_amd_dgpu_reads_total_minus_used_on_detects_card(monkeypatch, tmp_path):
    card = tmp_path / "card1" / "device"
    card.mkdir(parents=True)
    (card / "mem_info_vram_total").write_text(str(16 * GIB) + "\n")
    (card / "mem_info_vram_used").write_text(str(5 * GIB) + "\n")
    monkeypatch.setattr(detect, "_amd_card_dirs", lambda: [str(card)])
    monkeypatch.setattr(detect, "_run", lambda cmd: pytest.fail("no nvidia-smi on an AMD card"))
    assert detect.live_free_bytes(amd_dgpu()) == (11 * GIB, "free VRAM")


def test_amd_dgpu_without_the_used_file_is_unread(monkeypatch, tmp_path):
    card = tmp_path / "card0" / "device"
    card.mkdir(parents=True)
    (card / "mem_info_vram_total").write_text(str(16 * GIB))
    monkeypatch.setattr(detect, "_amd_card_dirs", lambda: [str(card)])
    assert detect.live_free_bytes(amd_dgpu()) == (None, "free VRAM unread")
