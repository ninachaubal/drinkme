"""probe's host-load sampler and the bench summary's warning threshold
(probe.HOST_LOAD_WARN_PER_CPU: the comment above it has the Strix Halo
runs it comes from). CPU only."""

from __future__ import annotations

import os

import pytest

from drinkme import probe


def test_load1_is_the_one_minute_average_or_none(monkeypatch):
    monkeypatch.setattr(os, "getloadavg", lambda: (3.14159, 2.0, 1.0))
    assert probe.load1() == 3.14

    def no_loadavg():
        raise OSError("no load average here")
    monkeypatch.setattr(os, "getloadavg", no_loadavg)
    assert probe.load1() is None


def test_record_host_load_keeps_each_arm_before_and_after(monkeypatch):
    loads = iter([4.0, 2.5])
    monkeypatch.setattr(probe, "load1", lambda: next(loads))
    monkeypatch.setattr(os, "cpu_count", lambda: 32)
    report = {}
    probe.record_host_load(report, "stock", 1.25)
    probe.record_host_load(report, "compressed", 3.0)
    assert report["host_load"] == {"cpu_count": 32, "load1": {"stock": [1.25, 4.0], "compressed": [3.0, 2.5]}}


@pytest.mark.parametrize("load1, warned", [
    ({"stock": [1.4, 1.6], "compressed": [6.4, 6.4]}, None),          # at the line: quiet
    ({"stock": [1.4, 1.6], "compressed": [5.6, 8.2]}, ["compressed"]),  # the first biased run
    ({"stock": [14.5, 25.0], "twin": [27.5, 21.0]}, ["stock", "twin"]),
    ({"stock": [None, None]}, None),                                  # an OS without a load average
])
def test_the_warning_fires_above_a_fifth_of_the_cpus(load1, warned):
    host = {"cpu_count": 32, "load1": load1}
    w = probe.host_load_warning(host)
    if warned is None:
        assert w is None
    else:
        assert w is not None and f"{', '.join(warned)} {'was' if len(warned) == 1 else 'were'} timed" in w
        assert "over 6.4 = 0.2 x 32 CPUs" in w


def test_the_line_scales_with_the_cpu_count():
    host = {"cpu_count": 8, "load1": {"stock": [1.0, 2.0]}}
    assert probe.host_load_warning(host) is not None  # 2.0 > 0.2 x 8
    assert probe.host_load_warning({"cpu_count": 8, "load1": {"stock": [1.0, 1.6]}}) is None
    assert probe.host_load_warning(None) is None and probe.host_load_warning({"cpu_count": None}) is None
