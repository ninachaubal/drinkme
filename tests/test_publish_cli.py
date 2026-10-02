"""The publish verb's wiring and refusals, no network: --logout, a record
that fails the lexicon is refused with the field named, an incomplete run
is refused, no identity stops before any network, the newest file under
./measurements/ is the default, and a stale stored session is refreshed."""

import json
import os
import time

import pytest

from drinkme import bench, publish
from drinkme.cli import main
from drinkme.publish import oauth, store
from drinkme.publish.jwt import Es256Key
from test_publish_validate import good


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    monkeypatch.setenv("DRINKME_CONFIG_DIR", str(tmp_path / "cfg"))
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_logout_alone_exits_zero_and_says_what_happened(cfg, capsys):
    assert main(["publish", "--logout"]) == 0
    assert "was not present" in capsys.readouterr().out
    store.save_session({"access_token": "A"})
    assert main(["publish", "--logout"]) == 0
    assert "removed" in capsys.readouterr().out
    assert store.load_session() is None


def test_invalid_record_is_refused_with_the_field(cfg, capsys):
    r = good()
    del r["environment"]["memoryKind"]
    p = cfg / "bad.json"
    p.write_text(json.dumps(r))
    assert main(["publish", str(p), "--handle", "alice.example"]) == 3
    err = capsys.readouterr().err
    assert "not a valid wtf.petrichor.drinkme.measurement record" in err
    assert "environment.memoryKind: required, missing" in err


def test_incomplete_run_is_refused(cfg, capsys):
    r = good()
    r["incomplete"] = "no-ratio-model-fits"
    p = cfg / "inc.json"
    p.write_text(json.dumps(r))
    assert main(["publish", str(p), "--handle", "alice.example"]) == 3
    assert "incomplete run" in capsys.readouterr().err


def test_no_identity_stops_before_any_network(cfg, capsys, monkeypatch):
    monkeypatch.setattr(publish.identity, "resolve", lambda **k: (_ for _ in ()).throw(AssertionError("network")))
    p = cfg / "ok.json"
    p.write_text(json.dumps(good()))
    assert main(["publish", str(p)]) == 2
    assert "who are you" in capsys.readouterr().err


def test_default_is_the_newest_measurement(cfg, capsys, monkeypatch):
    first = bench.write_record(good())
    time.sleep(0.01)
    second = bench.write_record(good())
    os.utime(first, (time.time() + 100, time.time() + 100))  # newest by mtime, not by name
    seen = {}

    def fake_resolve(**k):
        seen["resolved"] = True
        raise publish.identity.IdentityError("stop here")
    monkeypatch.setattr(publish.identity, "resolve", fake_resolve)
    assert main(["publish", "--handle", "alice.example"]) == 4
    out = capsys.readouterr()
    assert first in out.out and "(newest under ./measurements/)" in out.out
    assert second not in out.out and seen["resolved"]


def test_no_measurements_dir_says_run_bench(cfg, capsys):
    assert main(["publish", "--handle", "a.b"]) == 2
    assert "run `drinkme bench` first" in capsys.readouterr().err


def test_stale_session_is_refreshed_not_reauthorized(cfg, monkeypatch):
    key = Es256Key.generate()
    store.save_session({"did": "did:plc:x", "issuer": "https://as", "pds": "https://pds",
                        "client_id": "http://localhost?x", "scope": "atproto repo:y?action=create",
                        "access_token": "OLD", "refresh_token": "R1",
                        "expires_at": int(time.time()) - 10, "dpop_key": key.to_jwk(private=True),
                        "nonces": {"https://as": "n"}})
    calls = {}

    def fake_refresh(client, meta, *, cid, refresh_token):
        calls["cid"], calls["rt"] = cid, refresh_token
        assert client.key.to_jwk() == key.to_jwk()  # the same DPoP key, as the AS requires
        return {"access_token": "NEW", "refresh_token": "R2", "expires_in": 300,
                "sub": "did:plc:x", "token_type": "DPoP", "scope": "atproto repo:y?action=create"}
    monkeypatch.setattr(oauth, "refresh_tokens", fake_refresh)
    s, client = publish._session_for({"did": "did:plc:x", "pds": "https://pds"}, {"issuer": "https://as"})
    assert calls == {"cid": "http://localhost?x", "rt": "R1"}
    assert s["access_token"] == "NEW" and s["refresh_token"] == "R2"
    assert store.load_session()["access_token"] == "NEW"


def test_session_for_another_did_or_issuer_is_ignored(cfg):
    key = Es256Key.generate()
    store.save_session({"did": "did:plc:x", "issuer": "https://as", "access_token": "A",
                        "expires_at": int(time.time()) + 900, "dpop_key": key.to_jwk(private=True)})
    assert publish._session_for({"did": "did:plc:other", "pds": "p"}, {"issuer": "https://as"}) == (None, None)
    assert publish._session_for({"did": "did:plc:x", "pds": "p"}, {"issuer": "https://elsewhere"}) == (None, None)
    s, c = publish._session_for({"did": "did:plc:x", "pds": "p"}, {"issuer": "https://as"})
    assert s["access_token"] == "A" and isinstance(c, oauth.DpopClient)


def test_did_only_switch_clears_the_old_handle_for_the_next_invocation(cfg, monkeypatch):
    """A successful --did-only publish must not leave the
    previous account's handle as the next default. The ruler is the *next*
    invocation's resolved identity, not the raw stored field."""
    store.save_config(handle="old.example", did="did:plc:old")

    p = cfg / "rec.json"
    p.write_text(json.dumps(good()))

    key = Es256Key.generate()
    store.save_session({"did": "did:plc:new", "issuer": "https://as.example",
                        "pds": "https://pds.example", "client_id": "http://localhost?x",
                        "scope": "atproto repo:y?action=create", "access_token": "FAKE-TOK",
                        "refresh_token": "FAKE-RT", "expires_at": int(time.time()) + 900,
                        "dpop_key": key.to_jwk(private=True), "nonces": {}})

    monkeypatch.setattr(publish.identity, "resolve", lambda **k: {
        "did": "did:plc:new", "handle": None, "pds": "https://pds.example", "doc": {}})
    monkeypatch.setattr(publish.oauth, "discover", lambda pds: {"issuer": "https://as.example"})
    monkeypatch.setattr(publish.pds, "create_record", lambda client, session, collection, record: {
        "uri": "at://did:plc:new/wtf.petrichor.drinkme.measurement/abc"})
    monkeypatch.setattr(publish.pds, "get_record",
                        lambda client, session, uri: {"value": good()})

    assert main(["publish", str(p), "--did", "did:plc:new"]) == 0
    # the stored field itself: handle is gone, not merely unchanged
    assert store.load_config() == {"did": "did:plc:new"}

    # the ruler: the *next* invocation's identity resolution, not the field
    seen = {}

    def fake_resolve(**k):
        seen.update(k)
        raise publish.identity.IdentityError("stop before any network")
    monkeypatch.setattr(publish.identity, "resolve", fake_resolve)
    assert main(["publish", str(p)]) == 4
    assert seen == {"handle": None, "did": "did:plc:new", "plc": None}


def test_canonical_compare_is_the_verdict():
    from drinkme.publish import pds
    a = {"b": 1, "a": {"y": [1, 2], "x": "é"}}
    b = {"a": {"x": "é", "y": [1, 2]}, "b": 1}
    assert pds.canonical(a) == pds.canonical(b) == b'{"a":{"x":"\xc3\xa9","y":[1,2]},"b":1}'
    assert pds.canonical({"b": 1.0}) != pds.canonical({"b": 1})  # 124.0 vs 124: a real difference
    assert pds.parse_at_uri("at://did:plc:x/wtf.petrichor.drinkme.measurement/3abc") == (
        "did:plc:x", "wtf.petrichor.drinkme.measurement", "3abc")
    with pytest.raises(ValueError):
        pds.parse_at_uri("https://nope")
