"""The running version is `drinkme.__version__` (bench's records, the engine
string, `drinkme --version` and the site build read it), repeated where
packaging reads it: pyproject.toml's project version and uv.lock's entry for
drinkme (docs/versioning.md). A bump that misses one of them fails here.
"""

from __future__ import annotations

import pathlib

import pytest

from drinkme import __version__

ROOT = pathlib.Path(__file__).resolve().parents[1]
tomllib = pytest.importorskip("tomllib")  # Python 3.11+


def test_pyproject_declares_the_running_version():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    assert project["version"] == __version__


def test_the_lock_declares_the_running_version():
    lock = tomllib.loads((ROOT / "uv.lock").read_text())
    [entry] = [p for p in lock["package"] if p["name"] == "drinkme"]
    assert entry["version"] == __version__
