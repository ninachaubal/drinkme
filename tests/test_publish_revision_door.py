"""model.revision is REQUIRED and `drinkme publish` is the door.

A public record with no immutable source identity cannot be grouped or
reproduced: `model.revision` is the resolved hub commit every arm loaded,
required by the lexicon, and `drinkme publish` refuses a record without
one BY NAME, with the fix in the reason — per record when a set is given
(the others still go). The operator never loses a bench: `drinkme bench`
against a local checkpoint directory (no hub commit, only a structural
digest) still writes its record locally, marks it `raw.unpublishable` and
says so on its last line. bench never validates against the lexicon at
write time (it never imports publish), so the door is publish's alone.
"""

from __future__ import annotations

import json
import time

import pytest

from drinkme import arms, bench, publish
from drinkme.cli import main
from drinkme.publish import store
from drinkme.publish import validate as V
from drinkme.publish.jwt import Es256Key
from test_publish_validate import good

SHA = "b968826d9c46dd6066d109eabc6255188de91218"
LOCAL = "local-0123456789abcdef"  # checkpoint.resolved_revision's form for a directory


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    monkeypatch.setenv("DRINKME_CONFIG_DIR", str(tmp_path / "cfg"))
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _boom(**k):
    raise AssertionError("identity must not be resolved for a record the door refuses")


def _fake_network(monkeypatch, created: list):
    """A stored session and a fake PDS: identity, discovery, createRecord
    (appending what was sent to `created`) and a byte-equal getRecord."""
    key = Es256Key.generate()
    store.save_session({"did": "did:plc:new", "issuer": "https://as.example",
                        "pds": "https://pds.example", "client_id": "http://localhost?x",
                        "scope": "atproto repo:y?action=create", "access_token": "FAKE-TOK",
                        "refresh_token": "FAKE-RT", "expires_at": int(time.time()) + 900,
                        "dpop_key": key.to_jwk(private=True), "nonces": {}})
    monkeypatch.setattr(publish.identity, "resolve", lambda **k: {
        "did": "did:plc:new", "handle": None, "pds": "https://pds.example", "doc": {}})
    monkeypatch.setattr(publish.oauth, "discover", lambda pds: {"issuer": "https://as.example"})

    def create_record(client, session, collection, record):
        created.append(record)
        return {"uri": f"at://did:plc:new/{collection}/rec{len(created)}"}

    monkeypatch.setattr(publish.pds, "create_record", create_record)
    monkeypatch.setattr(publish.pds, "get_record",
                        lambda client, session, uri: {"value": created[int(uri[-1]) - 1]})


# ------------------------------------------------------------ the lexicon --


def test_the_lexicon_requires_model_revision_and_says_why():
    lex = V.load_lexicon()
    model = lex["defs"]["model"]
    assert "revision" in model["required"], model["required"]
    desc = model["properties"]["revision"]["description"]
    assert "commit" in desc and "not publishable" in desc, desc


def test_validate_refuses_a_record_without_revision_by_path():
    r = good()
    del r["model"]["revision"]
    errors = V.validate(r)
    assert any(e.startswith("model.revision:") for e in errors), errors


def test_revision_verdict_is_none_for_a_commit_and_names_the_fix_otherwise():
    r = good()
    r["model"]["revision"] = SHA
    assert V.revision_verdict(r) is None
    del r["model"]["revision"]
    v = V.revision_verdict(r)
    assert v and "model.revision" in v and "hub repo" in v, v
    assert "\n" not in v  # one reason line
    # a local directory's structural digest is not a resolved hub commit
    r["model"]["revision"] = LOCAL
    v = V.revision_verdict(r)
    assert v and LOCAL in v and "hub repo" in v, v
    r["model"]["revision"] = "main"  # a coordinate, not what resolved
    assert V.revision_verdict(r)


# ---------------------------------------------------------------- the door --


def test_a_record_with_a_revision_publishes_and_the_same_one_without_is_refused_by_name(
        cfg, capsys, monkeypatch):
    created: list = []
    _fake_network(monkeypatch, created)
    r = good()
    r["model"]["revision"] = SHA
    p = cfg / "ok.json"
    p.write_text(json.dumps(r))
    assert main(["publish", str(p), "--did", "did:plc:new"]) == 0
    out = capsys.readouterr().out
    assert "your point is published" in out and len(created) == 1

    del r["model"]["revision"]
    q = cfg / "norev.json"
    q.write_text(json.dumps(r))
    monkeypatch.setattr(publish.identity, "resolve", _boom)  # refused before any network
    assert main(["publish", str(q), "--did", "did:plc:new"]) == 3
    err = capsys.readouterr().err
    assert str(q) in err and "model.revision" in err and "hub repo" in err, err
    assert len(created) == 1  # nothing else was sent


def test_the_refusal_is_typed(cfg, monkeypatch):
    r = good()
    del r["model"]["revision"]
    with pytest.raises(publish.RecordRefused) as e:
        publish.check_record(r, "norev.json")
    assert e.value.path == "norev.json" and "model.revision" in e.value.reason
    assert isinstance(e.value, publish.PublishError)
    # the older refusals are the same type, so a set can catch them per record
    r = good()
    r["model"]["revision"] = SHA
    r["incomplete"] = "no-ratio-model-fits"
    with pytest.raises(publish.RecordRefused):
        publish.check_record(r, "inc.json")


def test_a_mixed_set_publishes_the_good_ones_and_lists_the_refused(cfg, capsys, monkeypatch):
    created: list = []
    _fake_network(monkeypatch, created)
    a, b, c = good(), good(), good()
    a["model"]["revision"] = SHA
    a["model"]["name"] = "A"
    del b["model"]["revision"]
    c["model"]["revision"] = SHA
    c["model"]["name"] = "C"
    paths = []
    for name, rec in (("a", a), ("b", b), ("c", c)):
        p = cfg / f"{name}.json"
        p.write_text(json.dumps(rec))
        paths.append(str(p))
    rc = main(["publish", *paths, "--did", "did:plc:new"])
    out, err = capsys.readouterr()
    # the good ones went, in order, and each has its uri on stdout
    assert [x["model"]["name"] for x in created] == ["A", "C"]
    assert "at://did:plc:new/wtf.petrichor.drinkme.measurement/rec1" in out
    assert "at://did:plc:new/wtf.petrichor.drinkme.measurement/rec2" in out
    assert "published 2 of 3" in out
    # the refused one is listed by path with the reason, and the run says so
    assert rc == 3
    assert paths[1] in err and "model.revision" in err and "hub repo" in err, err
    assert paths[0] not in err and paths[2] not in err


def test_a_set_with_nothing_publishable_stops_before_identity(cfg, capsys, monkeypatch):
    monkeypatch.setattr(publish.identity, "resolve", _boom)
    a, b = good(), good()
    del a["model"]["revision"]
    b["model"]["revision"] = LOCAL
    pa, pb = cfg / "a.json", cfg / "b.json"
    pa.write_text(json.dumps(a))
    pb.write_text(json.dumps(b))
    assert main(["publish", str(pa), str(pb), "--did", "did:plc:new"]) == 3
    err = capsys.readouterr().err
    assert str(pa) in err and str(pb) in err and LOCAL in err


# ------------------------------------------------- the local bench still writes --


def _local_raw():
    from test_bench_record_shape import _fake_raw

    raw = _fake_raw()
    raw["revision"] = None  # a directory has no coordinate to pin
    raw["resolved_revision"] = LOCAL
    return raw


def test_a_local_bench_writes_its_record_marks_it_and_says_so(linux_host, tmp_path, monkeypatch, capsys):
    from test_bench_record_shape import _fake_hw, _fake_suggestion

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(bench, "detect", _fake_hw)
    monkeypatch.setattr(bench, "suggest", lambda *a, **k: _fake_suggestion())
    monkeypatch.setattr(arms, "run_arms", lambda *a, **k: _local_raw())
    bench.main_json(None, None, no_gemma=True)
    out = capsys.readouterr().out
    files = sorted((tmp_path / "measurements").glob("*.json"))
    assert len(files) == 1, "the operator's own numbers are still written"
    on_disk = json.loads(files[0].read_text())
    assert "revision" not in on_disk["model"]  # the typed field is the hub commit or nothing
    assert on_disk["raw"]["resolved_revision"] == LOCAL  # the digest stays, in raw
    assert on_disk["raw"]["unpublishable"].startswith("no resolved revision")
    # the last lines say so, and do not promise a publish
    tail = out.strip().splitlines()[-2:]
    assert files[0].name in tail[0]  # printed relative to the working directory
    assert "drinkme publish" in tail[1] and "refuse" in tail[1], tail
    assert "sends exactly those bytes" not in out
    # and the door refuses that very file, by name
    monkeypatch.setenv("DRINKME_CONFIG_DIR", str(tmp_path / "cfg"))
    monkeypatch.setattr(publish.identity, "resolve", _boom)
    assert main(["publish", str(files[0]), "--handle", "alice.example"]) == 3
    err = capsys.readouterr().err
    assert "model.revision" in err and "hub repo" in err


def test_a_hub_bench_carries_the_commit_and_no_unpublishable_note(monkeypatch, capsys, tmp_path):
    from test_bench_record_shape import _build_record

    monkeypatch.chdir(tmp_path)
    r = _build_record(monkeypatch)
    assert V.validate(bench.lexicon_safe(r)) == []
    assert "unpublishable" not in r["raw"]
    assert V.revision_verdict(r) is None
    bench.main_json(None, None, no_gemma=True)
    assert "sends exactly those bytes" in capsys.readouterr().out
