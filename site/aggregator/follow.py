"""Keeping the record set current: Jetstream's live tail, and a daily reconcile.

`follow` subscribes to Jetstream for our collection alone and persists the
stream cursor, so a restart resumes where it stopped. Jetstream is not
self-authenticating, so an event is only a hint: on a create or update the
record is fetched from its author's PDS (`sources.get_record`, the PDS found
through the DID document) and THAT body is admitted, never the event's
payload; a record the PDS no longer has is dropped. A delete drops the at-uri.
Every change re-runs the pipeline and rewrites points.json atomically.
Applying an event twice changes nothing (the cursor is inclusive), a dropped
connection reconnects with backoff from the cursor, and once a day
`reconcile` replaces the record set with a full backfill, catching missed
events and deletes the relay stopped listing.

Two Jetstream wire formats are read. The v2 endpoint
(`/xrpc/network.bsky.jetstream.subscribeEvents`) filters on `collections=`
and wraps each event as {"$type": "message", "payload": {..., "seq"}}; its
cursor is `seq`. The v1 endpoint (`/subscribe`) filters on
`wantedCollections=` (it ignores `collections=`, and sends every event of
the network) and its cursor is `time_us`. Both resume with `cursor=`. Events
of other collections are ignored whatever the server sends.

State, in the state directory: `cursor.json` (the stream cursor and when the
last reconcile ran) and `records.json` (every record by at-uri), each
written through a temporary file and a rename.
"""
from __future__ import annotations

import datetime as _dt
import json
import pathlib
import threading
import time
import urllib.parse

from . import pipeline, sources
from .pipeline import NSID, Settings
from .sources import FetchError, Source, log

JETSTREAM = "wss://jetstream.us-east.bsky.network/xrpc/network.bsky.jetstream.subscribeEvents"
DAY = 24 * 3600.0


def subscribe_url(base: str, cursor=None) -> str:
    """The subscription URL for our collection, with the filter parameter
    the endpoint understands, resuming from `cursor`."""
    parts = urllib.parse.urlsplit(base)
    name = "collections" if parts.path.rstrip("/").endswith("subscribeEvents") else "wantedCollections"
    q = urllib.parse.parse_qsl(parts.query) + [(name, NSID)]
    if cursor is not None:
        q.append(("cursor", str(cursor)))
    return urllib.parse.urlunsplit(parts._replace(query=urllib.parse.urlencode(q)))


def parse_event(msg) -> dict | None:
    """A Jetstream message (either format) -> {cursor, kind, did, collection,
    operation, rkey}; None if it is not an event."""
    try:
        e = json.loads(msg)
    except (TypeError, ValueError):
        return None
    if not isinstance(e, dict):
        return None
    if e.get("$type") == "message" and isinstance(e.get("payload"), dict):  # v2
        p = e["payload"]
        kind = str(p.get("$type", "")).rpartition("#")[2]
        return {"cursor": p.get("seq"), "kind": kind, "did": p.get("did"), "collection": p.get("collection"),
                "operation": p.get("operation"), "rkey": p.get("rkey")}
    if "time_us" in e:  # v1
        c = e.get("commit") if isinstance(e.get("commit"), dict) else {}
        return {"cursor": e.get("time_us"), "kind": e.get("kind"), "did": e.get("did"),
                "collection": c.get("collection"), "operation": c.get("operation"), "rkey": c.get("rkey")}
    return None


class State:
    def __init__(self, path):
        self.dir = pathlib.Path(path)
        self.dir.mkdir(parents=True, exist_ok=True)
        cur = self._load("cursor.json", {})
        self.cursor = cur.get("cursor")
        self.reconciled_at = cur.get("reconciledAt")  # epoch seconds
        self.records = self._load("records.json", {})

    def _load(self, name, default):
        p = self.dir / name
        return json.loads(p.read_text()) if p.exists() else default

    def save_cursor(self):
        pipeline._replace(self.dir / "cursor.json",
                          json.dumps({"cursor": self.cursor, "reconciledAt": self.reconciled_at}) + "\n")

    def save_records(self):
        pipeline._replace(self.dir / "records.json", json.dumps(self.records) + "\n")


class Follower:
    def __init__(self, src: Source, out, state_dir, settings: Settings = Settings(),
                 jetstream: str = JETSTREAM, reconcile_every: float = DAY, retry_every: float = 600.0,
                 backoff_min: float = 1.0, backoff_max: float = 60.0, recv_timeout: float = 1.0,
                 cursor_every: float = 10.0, clock=time.time):
        self.src, self.out, self.settings = src, out, settings
        self.state = State(state_dir)
        self.jetstream, self.reconcile_every, self.retry_every = jetstream, reconcile_every, retry_every
        self.backoff_min, self.backoff_max = backoff_min, backoff_max
        self.recv_timeout, self.cursor_every, self.clock = recv_timeout, cursor_every, clock
        self.next_try = 0.0  # after a failed reconcile, not before this
        self.skipped: list = []
        self.connections = 0  # how many times the stream connected (tests read it)
        self.messages = 0     # messages received, of any kind

    # ------------------------------------------------------------ output --
    def publish(self):
        points, excluded = pipeline.run(self.state.records.values(), self.settings)
        doc = pipeline.document(points, excluded, _dt.datetime.now(_dt.timezone.utc),
                                cursor=self.state.cursor, skipped=self.skipped)
        pipeline.write_atomic(self.out, doc)
        log(f"wrote {self.out}: {len(points)} points, {len(excluded)} excluded, cursor {self.state.cursor}")

    # --------------------------------------------------------- reconcile --
    def reconcile(self) -> bool:
        """A full backfill replaces the record set. A repository that could
        not be read keeps the records it had, and if discovery itself fails
        nothing is replaced and it is retried after `retry_every` seconds."""
        try:
            snap = sources.backfill(self.src)
        except FetchError as exc:
            self.next_try = self.clock() + self.retry_every
            log(f"reconcile failed, keeping {len(self.state.records)} records: {exc}")
            return False
        kept = {s["did"] for s in snap.skipped}
        records = {r["uri"]: r for r in snap.records}
        for uri, r in self.state.records.items():
            if pipeline.parse_uri(uri) and pipeline.parse_uri(uri)[0] in kept:
                records.setdefault(uri, r)
        self.state.records, self.skipped = records, snap.skipped
        self.state.reconciled_at = self.clock()
        self.state.save_records()
        self.state.save_cursor()
        log(f"reconciled: {len(records)} records from {len(snap.read)} repos, {len(snap.skipped)} skipped")
        self.publish()
        return True

    def reconcile_due(self) -> bool:
        now = self.clock()
        return now >= self.next_try and (self.state.reconciled_at is None
                                         or now - self.state.reconciled_at >= self.reconcile_every)

    # ------------------------------------------------------------ events --
    def handle(self, ev: dict) -> bool:
        """Apply one event; True if the record set changed."""
        if ev.get("cursor") is not None:
            self.state.cursor = ev["cursor"]
        if ev.get("kind") != "commit" or ev.get("collection") != NSID:
            return False
        did, rkey, op = ev.get("did"), ev.get("rkey"), ev.get("operation")
        uri = f"at://{did}/{NSID}/{rkey}"
        if pipeline.parse_uri(uri) is None:
            return False
        if op == "delete":
            return self.state.records.pop(uri, None) is not None
        if op not in ("create", "update"):
            return False
        try:
            rec = sources.get_record(uri, self.src)
        except FetchError as exc:
            log(f"{op} {uri}: could not read it from its PDS ({exc}); the next reconcile retries")
            return False
        if rec is None:  # the PDS no longer has it
            return self.state.records.pop(uri, None) is not None
        if self.state.records.get(uri) == rec:
            return False
        self.state.records[uri] = rec
        log(f"{op} {uri}: admitted the PDS's record")
        return True

    # -------------------------------------------------------------- loop --
    def run(self, stop: threading.Event):
        from websockets.exceptions import WebSocketException
        from websockets.sync.client import connect

        if self.reconcile_due():
            if not self.reconcile():
                self.publish()
        else:
            self.publish()
        backoff, saved = self.backoff_min, self.clock()
        while not stop.is_set():
            url = subscribe_url(self.jetstream, self.state.cursor)
            try:
                with connect(url, open_timeout=10, close_timeout=2, max_size=1 << 22) as ws:
                    self.connections += 1
                    log(f"subscribed: {url}")
                    while not stop.is_set():
                        if self.reconcile_due():
                            self.reconcile()
                        try:
                            msg = ws.recv(timeout=self.recv_timeout)
                        except TimeoutError:
                            msg = None
                        if msg is not None:
                            self.messages += 1
                            backoff = self.backoff_min
                            ev = parse_event(msg)
                            if ev is not None and self.handle(ev):
                                self.state.save_records()
                                self.state.save_cursor()
                                saved = self.clock()
                                self.publish()
                        if self.clock() - saved >= self.cursor_every:
                            self.state.save_cursor()
                            saved = self.clock()
            except (WebSocketException, OSError, TimeoutError) as exc:
                log(f"stream dropped ({type(exc).__name__}: {exc}); reconnecting in {backoff:g}s from cursor "
                    f"{self.state.cursor}")
                self.state.save_cursor()
                stop.wait(backoff)
                backoff = min(backoff * 2, self.backoff_max)
        self.state.save_cursor()
        log(f"stopped: {self.messages} messages on {self.connections} connections, cursor {self.state.cursor}")
