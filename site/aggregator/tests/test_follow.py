"""follow and reconcile, against a fake Jetstream (a local websocket server fed
canned events) and a mocked getRecord."""
import copy
import json
import threading
import time
import urllib.parse

import pytest
from websockets.sync.server import serve

from aggregator import follow, sources
from aggregator.follow import Follower
from aggregator.pipeline import NSID
from aggregator.sources import FetchError, Snapshot, Source
from records import DID_A, DID_B, rec, uri


def v2(seq, did, rkey, op="create", record=None, collection=NSID):
    p = {"$type": "network.bsky.jetstream.subscribeEvents#commit", "did": did, "collection": collection,
         "operation": op, "rkey": rkey, "seq": seq, "time": "2026-09-23T21:00:00Z"}
    if record is not None:
        p["record"] = record
    return json.dumps({"$type": "message", "payload": p})


def identity(seq, did):
    return json.dumps({"$type": "message", "payload": {"$type": "network.bsky.jetstream.subscribeEvents#identity",
                                                       "did": did, "seq": seq}})


class FakeJetstream:
    """A local websocket server. `scripts[i]` is what the i-th connection is
    sent; a connection whose script ends in None is closed by the server
    after sending, otherwise it stays open. Every request path is kept."""

    def __init__(self, scripts):
        self.scripts, self.paths, self.lock = list(scripts), [], threading.Lock()
        self.server = serve(self.handler, "127.0.0.1", 0)
        self.url = f"ws://127.0.0.1:{self.server.socket.getsockname()[1]}" \
                   "/xrpc/network.bsky.jetstream.subscribeEvents"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def handler(self, ws):
        with self.lock:
            i = len(self.paths)
            self.paths.append(ws.request.path)
        script = self.scripts[i] if i < len(self.scripts) else []
        for msg in script:
            if msg is None:
                return  # the server hangs up
            ws.send(msg)
        try:
            for _ in ws:
                pass
        except Exception:
            pass

    def query(self, i):
        return dict(urllib.parse.parse_qsl(urllib.parse.urlsplit(self.paths[i]).query))

    def close(self):
        self.server.shutdown()


class PDS:
    """Mocked getRecord (and backfill): the records each PDS actually holds."""

    def __init__(self, monkeypatch, records=(), backfill=None):
        self.records = {r["uri"]: r for r in records}
        self.reads = []
        monkeypatch.setattr(sources, "get_record", self.get_record)
        monkeypatch.setattr(sources, "backfill", backfill or self.backfill)

    def get_record(self, u, src):
        self.reads.append(u)
        if u == "fail":
            raise FetchError("down")
        r = self.records.get(u)
        return copy.deepcopy(r) if r else None

    def backfill(self, src):
        return Snapshot(list(copy.deepcopy(list(self.records.values()))), [DID_A, DID_B], [])


def until(cond, timeout=10.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(0.02)
    raise AssertionError("timed out waiting")


@pytest.fixture
def run_follower(tmp_path):
    """Start a Follower in a thread; stop and join them all at teardown."""
    running = []

    def start(jet, state="state", **kw):
        f = Follower(Source(), tmp_path / "data" / "points.json", tmp_path / state, jetstream=jet.url,
                     backoff_min=0.05, recv_timeout=0.05, **kw)
        stop = threading.Event()
        t = threading.Thread(target=f.run, args=(stop,), daemon=True)
        t.start()
        running.append((stop, t))
        return f, stop, t

    yield start
    for stop, t in running:
        stop.set()
        t.join(10)


def points(tmp_path):
    p = tmp_path / "data" / "points.json"
    return json.loads(p.read_text()) if p.exists() else None


def tok(doc, u):
    for r in doc["records"]:
        if u in r["site"]["provenance"]:
            return {m["name"]: m["value"] for m in r["metrics"]}["compressed_decode_tok_s"]
    return None


def saved_cursor(tmp_path):
    p = tmp_path / "state" / "cursor.json"
    return json.loads(p.read_text())["cursor"] if p.exists() else None


def settled(tmp_path, cursor):
    d = points(tmp_path)
    return d is not None and d["cursor"] == cursor


# ---------------------------------------------------------------- pieces --

def test_the_subscription_filters_to_our_collection_on_either_endpoint():
    q = lambda url: dict(urllib.parse.parse_qsl(urllib.parse.urlsplit(url).query))
    assert q(follow.subscribe_url(follow.JETSTREAM)) == {"collections": NSID}
    assert q(follow.subscribe_url(follow.JETSTREAM, 42)) == {"collections": NSID, "cursor": "42"}
    assert q(follow.subscribe_url("wss://jetstream2.us-east.bsky.network/subscribe", 7)) == {
        "wantedCollections": NSID, "cursor": "7"}


def test_both_wire_formats_parse_to_one_event():
    a = follow.parse_event(v2(5, DID_A, "3a", "update"))
    b = follow.parse_event(json.dumps({"did": DID_A, "time_us": 5, "kind": "commit",
                                       "commit": {"operation": "update", "collection": NSID, "rkey": "3a"}}))
    assert a == b == {"cursor": 5, "kind": "commit", "did": DID_A, "collection": NSID,
                      "operation": "update", "rkey": "3a"}
    assert follow.parse_event("not json") is None and follow.parse_event("[1]") is None


# ------------------------------------------------------------- authority --

def test_a_forged_event_payload_loses_to_the_pds_record(tmp_path, monkeypatch, run_follower):
    """The event claims 99 tok/s (and would fail F3); the PDS holds 16.05:
    16.05 is admitted. An event for a record its PDS does not have admits nothing."""
    genuine = rec(DID_A, "3real")
    forged_body = copy.deepcopy(genuine["value"])
    forged_body["metrics"][0]["value"] = "99"
    server = PDS(monkeypatch, [genuine], backfill=lambda src: Snapshot([], [], []))  # the reconcile sees nothing
    jet = FakeJetstream([[identity(1, DID_B), v2(2, DID_A, "3real", record=forged_body),
                          v2(3, DID_B, "3ghost", record=rec(DID_B, "3ghost")["value"])]])
    f, stop, t = run_follower(jet)
    until(lambda: uri(DID_B, "3ghost") in server.reads)
    assert server.reads == [uri(DID_A, "3real"), uri(DID_B, "3ghost")]
    assert f.state.cursor == 3
    doc = points(tmp_path)
    assert doc["cursor"] == 2  # the last change
    assert tok(doc, uri(DID_A, "3real")) == "16.05"
    assert [r["site"]["source"] for r in doc["records"]] == [uri(DID_A, "3real")]
    assert doc["excluded"] == []
    jet.close()


def test_an_unreadable_pds_leaves_the_record_set_alone(tmp_path, monkeypatch):
    PDS(monkeypatch, [rec(DID_A, "3a")])
    f = Follower(Source(), tmp_path / "p.json", tmp_path / "s")
    monkeypatch.setattr(sources, "get_record", lambda u, s: (_ for _ in ()).throw(FetchError("timed out")))
    assert f.handle(follow.parse_event(v2(9, DID_A, "3a"))) is False
    assert f.state.records == {} and f.state.cursor == 9


def test_events_of_other_collections_are_ignored(tmp_path, monkeypatch):
    server = PDS(monkeypatch, [rec(DID_A, "3a")])
    f = Follower(Source(), tmp_path / "p.json", tmp_path / "s")
    assert f.handle(follow.parse_event(v2(4, DID_A, "3a", collection="app.bsky.feed.post"))) is False
    assert server.reads == [] and f.state.cursor == 4


# -------------------------------------------------- creates, updates, deletes --

def test_create_update_and_delete_rewrite_points_and_repeats_change_nothing(tmp_path, monkeypatch, run_follower):
    server = PDS(monkeypatch, [])
    first, second = rec(DID_A, "3a"), rec(DID_A, "3a", comp="17.0")
    jet = FakeJetstream([[]])
    f, stop, t = run_follower(jet)
    until(lambda: points(tmp_path) is not None)
    # apply events directly through the same handler the stream uses
    server.records[first["uri"]] = first
    assert f.handle(follow.parse_event(v2(10, DID_A, "3a"))) is True
    assert f.handle(follow.parse_event(v2(10, DID_A, "3a"))) is False       # the inclusive cursor's repeat
    server.records[first["uri"]] = second
    assert f.handle(follow.parse_event(v2(11, DID_A, "3a", "update"))) is True
    assert f.state.records[first["uri"]]["value"]["metrics"][0]["value"] == "17.0"
    assert f.handle(follow.parse_event(v2(12, DID_A, "3a", "delete"))) is True
    assert f.handle(follow.parse_event(v2(12, DID_A, "3a", "delete"))) is False
    assert f.state.records == {}
    jet.close()


def test_a_delete_from_the_stream_drops_the_record_from_points(tmp_path, monkeypatch, run_follower):
    keep, gone = rec(DID_A, "3keep"), rec(DID_B, "3gone", comp="15.0")
    PDS(monkeypatch, [keep, gone])
    jet = FakeJetstream([[v2(20, DID_B, "3gone", "delete")]])
    f, stop, t = run_follower(jet)
    until(lambda: settled(tmp_path, 20))
    doc = points(tmp_path)
    assert [r["site"]["source"] for r in doc["records"]] == [keep["uri"]]
    assert json.loads((tmp_path / "state" / "records.json").read_text()).keys() == {keep["uri"]}
    jet.close()


# ------------------------------------------------- cursor and reconnection --

def test_the_cursor_persists_and_a_restart_resumes_from_it(tmp_path, monkeypatch, run_follower):
    PDS(monkeypatch, [rec(DID_A, "3a")])
    jet = FakeJetstream([[v2(30, DID_A, "3a", "update"), identity(31, DID_B)], []])
    f, stop, t = run_follower(jet, cursor_every=0)
    until(lambda: saved_cursor(tmp_path) == 31)
    stop.set()
    t.join(10)
    assert "cursor" not in jet.query(0)
    f2, stop2, t2 = run_follower(jet)
    until(lambda: len(jet.paths) == 2)
    assert jet.query(1) == {"collections": NSID, "cursor": "31"}
    assert f2.state.cursor == 31
    jet.close()


def test_a_dropped_stream_reconnects_with_backoff_from_the_cursor(tmp_path, monkeypatch, run_follower):
    a, b = rec(DID_A, "3a"), rec(DID_B, "3b", comp="15.0")
    PDS(monkeypatch, [a, b], backfill=lambda src: Snapshot([], [], []))
    jet = FakeJetstream([[v2(40, DID_A, "3a"), None],      # sends one event, then hangs up
                         [None],                            # hangs up at once
                         [v2(41, DID_B, "3b")]])
    f, stop, t = run_follower(jet)
    until(lambda: settled(tmp_path, 41))
    assert f.connections == 3
    assert "cursor" not in jet.query(0)
    assert jet.query(1)["cursor"] == "40" and jet.query(2)["cursor"] == "40"
    assert {r["site"]["source"] for r in points(tmp_path)["records"]} == {a["uri"], b["uri"]}
    jet.close()


# ------------------------------------------------------------- reconcile --

def test_reconcile_replaces_the_record_set(tmp_path, monkeypatch):
    stale, fresh = rec(DID_A, "3stale"), rec(DID_A, "3fresh", name="Qwen3-4B")
    PDS(monkeypatch, [fresh])
    f = Follower(Source(), tmp_path / "p.json", tmp_path / "s")
    f.state.records = {stale["uri"]: stale}
    assert f.reconcile() is True
    assert set(f.state.records) == {fresh["uri"]}
    doc = json.loads((tmp_path / "p.json").read_text())
    assert [r["site"]["source"] for r in doc["records"]] == [fresh["uri"]]


def test_reconcile_keeps_the_records_of_a_repo_it_could_not_read(tmp_path, monkeypatch):
    held, new = rec(DID_B, "3held"), rec(DID_A, "3new")
    PDS(monkeypatch, backfill=lambda src: Snapshot([new], [DID_A], [{"did": DID_B, "reason": "timed out"}]))
    f = Follower(Source(), tmp_path / "p.json", tmp_path / "s")
    f.state.records = {held["uri"]: held}
    assert f.reconcile() is True
    assert set(f.state.records) == {held["uri"], new["uri"]}
    assert json.loads((tmp_path / "p.json").read_text())["skipped"] == [{"did": DID_B, "reason": "timed out"}]


def test_a_failed_discovery_replaces_nothing_and_retries_later(tmp_path, monkeypatch):
    held = rec(DID_B, "3held")

    def down(src):
        raise FetchError("every relay failed")
    PDS(monkeypatch, backfill=down)
    now = [1000.0]
    f = Follower(Source(), tmp_path / "p.json", tmp_path / "s", clock=lambda: now[0], retry_every=600)
    f.state.records = {held["uri"]: held}
    assert f.reconcile_due() and f.reconcile() is False
    assert set(f.state.records) == {held["uri"]}
    assert not f.reconcile_due()
    now[0] += 600
    assert f.reconcile_due()


def test_reconcile_runs_on_its_timer_inside_follow(tmp_path, monkeypatch, run_follower):
    calls = []
    server = PDS(monkeypatch, [rec(DID_A, "3a")])
    real = server.backfill
    monkeypatch.setattr(sources, "backfill", lambda src: (calls.append(1), real(src))[1])
    now = [0.0]
    jet = FakeJetstream([[]])
    f, stop, t = run_follower(jet, reconcile_every=100, clock=lambda: now[0])
    until(lambda: len(calls) == 1 and len(jet.paths) == 1)   # the first run reconciles
    time.sleep(0.2)
    assert len(calls) == 1
    now[0] = 100.0                                            # a day later, in miniature
    until(lambda: len(calls) == 2)
    # reconcile stamps reconciledAt after its backfill returns, so wait for the stamp, not the call
    cursor = tmp_path / "state" / "cursor.json"

    def stamped():
        try:
            return json.loads(cursor.read_text())["reconciledAt"] == 100.0
        except (OSError, ValueError):
            return False
    assert until(stamped)
    jet.close()


def test_the_reconcile_command_uses_the_follow_state(tmp_path, monkeypatch):
    from aggregator import __main__ as cli
    PDS(monkeypatch, [rec(DID_A, "3a")])
    out, state = tmp_path / "points.json", tmp_path / "state"
    assert cli.main(["reconcile", "-o", str(out), "--state", str(state)]) == 0
    assert len(json.loads(out.read_text())["records"]) == 1
    assert uri(DID_A, "3a") in json.loads((state / "records.json").read_text())
