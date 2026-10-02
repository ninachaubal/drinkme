"""publish/store.py: the session file is 0600 from its first byte, the
directory 0700, a file readable by others is refused, --logout deletes."""

import os
import stat

import pytest

from drinkme.publish import store


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    d = tmp_path / "cfg"
    monkeypatch.setenv("DRINKME_CONFIG_DIR", str(d))
    return d


def test_session_is_0600_and_dir_0700(cfg):
    path = store.save_session({"access_token": "A", "dpop_key": {"d": "secret"}})
    assert path == str(cfg / "session.json")
    assert stat.S_IMODE(os.stat(path).st_mode) == 0o600
    assert stat.S_IMODE(os.stat(cfg).st_mode) == 0o700
    assert store.load_session()["access_token"] == "A"
    assert not [p for p in os.listdir(cfg) if p.startswith(".tmp-")]  # no temp left behind


def test_group_or_world_readable_session_is_refused(cfg):
    path = store.save_session({"access_token": "A"})
    os.chmod(path, 0o644)
    with pytest.raises(PermissionError, match="readable by others"):
        store.load_session()


def test_logout_deletes_and_reports(cfg):
    assert store.delete_session() is False
    store.save_session({"access_token": "A"})
    assert store.delete_session() is True
    assert store.load_session() is None
    assert store.delete_session() is False


def test_config_remembers_identity_and_skips_none(cfg):
    store.save_config(handle="alice.example", did="did:plc:x", plc=None)
    assert store.load_config() == {"handle": "alice.example", "did": "did:plc:x"}
    store.save_config(plc="https://plc.example")
    assert store.load_config()["plc"] == "https://plc.example"
    assert store.load_config()["handle"] == "alice.example"


def test_save_config_clear_sentinel_removes_the_field_none_leaves_it(cfg):
    """None and CLEAR must not do the same thing — None
    leaves whatever is stored alone, CLEAR removes the key outright."""
    store.save_config(handle="alice.example", did="did:plc:x")
    store.save_config(handle=None, did="did:plc:y")
    assert store.load_config() == {"handle": "alice.example", "did": "did:plc:y"}
    store.save_config(handle=store.CLEAR, did="did:plc:z")
    assert store.load_config() == {"did": "did:plc:z"}
    assert "handle" not in store.load_config()
    store.save_config(handle=store.CLEAR)  # clearing an absent field is a no-op, not an error
    assert store.load_config() == {"did": "did:plc:z"}


def test_config_dir_defaults_under_xdg_or_home(monkeypatch, tmp_path):
    monkeypatch.delenv("DRINKME_CONFIG_DIR", raising=False)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    assert store.config_dir() == str(tmp_path / "xdg" / "drinkme")
    monkeypatch.delenv("XDG_CONFIG_HOME")
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    assert store.config_dir() == str(tmp_path / "home" / ".config" / "drinkme")


def test_corrupt_session_reads_as_none(cfg):
    store.save_session({"a": 1})
    with open(store.session_path(), "w") as f:
        f.write("{not json")
    assert store.load_session() is None
