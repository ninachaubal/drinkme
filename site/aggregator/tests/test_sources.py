"""Discovery, DID resolution and listing, with the HTTP layer mocked."""
import io
import json
import urllib.error
import urllib.parse

import pytest

from aggregator import __main__ as cli
from aggregator import pipeline, sources
from aggregator.pipeline import NSID
from aggregator.sources import FetchError, Source
from records import DID_A, DID_B, rec, uri

EAST, WEST = sources.RELAYS
PDS_A, PDS_B = "https://pds-a.example", "https://pds-b.example"


def did_doc(did, pds):
    return {"id": did, "service": [{"id": "#atproto_pds", "type": "AtprotoPersonalDataServer",
                                    "serviceEndpoint": pds}]}


class FakeHttp:
    """Routes GETs by URL prefix to canned answers: a dict, a callable of the
    params, or a FetchError to raise. Records every call."""

    def __init__(self, routes):
        self.routes, self.calls = routes, []

    def get_json(self, url, params=None):
        self.calls.append((url, dict(params or {})))
        for prefix, answer in self.routes.items():
            if url == prefix or url.startswith(prefix):
                if callable(answer):
                    answer = answer(dict(params or {}))
                if isinstance(answer, Exception):
                    raise answer
                return answer
        raise FetchError(f"HTTP 404 from {url}", 404)


def pages(items, key, size):
    """A paginated listing: `size` items a page, the cursor an offset."""
    def answer(params):
        start = int(params.get("cursor") or 0)
        out = {key: items[start:start + size]}
        if start + size < len(items):
            out["cursor"] = str(start + size)
        return out
    return answer


def world(**extra):
    routes = {
        f"{EAST}/xrpc/com.atproto.sync.listReposByCollection":
            pages([{"did": DID_A}, {"did": DID_B}], "repos", 1),
        f"{sources.PLC}/{DID_A}": did_doc(DID_A, PDS_A),
        "https://bench.example.com/.well-known/did.json": did_doc(DID_B, PDS_B),
        f"{PDS_A}/xrpc/com.atproto.repo.listRecords": pages([rec(DID_A, f"3a{i}") for i in range(5)], "records", 2),
        f"{PDS_B}/xrpc/com.atproto.repo.listRecords": pages([rec(DID_B, "3b0")], "records", 100),
    }
    routes.update(extra)
    return FakeHttp(routes)


def test_discovery_pages_the_relay_for_our_collection():
    http = world()
    assert sources.discover(Source(http=http)) == [DID_A, DID_B]
    relay_calls = [p for u, p in http.calls if "listReposByCollection" in u]
    assert [p.get("cursor") for p in relay_calls] == [None, "1"]
    assert all(p["collection"] == NSID and p["limit"] == 2000 for p in relay_calls)


def test_discovery_falls_back_to_the_next_relay():
    http = world(**{f"{EAST}/xrpc/com.atproto.sync.listReposByCollection": FetchError("timed out"),
                    f"{WEST}/xrpc/com.atproto.sync.listReposByCollection": {"repos": [{"did": DID_B}]}})
    assert sources.discover(Source(http=http)) == [DID_B]


def test_discovery_fails_only_when_every_relay_fails():
    http = FakeHttp({EAST: FetchError("down"), WEST: FetchError("down")})
    with pytest.raises(FetchError, match="every relay failed"):
        sources.discover(Source(http=http))


def test_did_plc_resolves_through_plc_directory_and_did_web_through_well_known():
    assert sources.did_document_url(DID_A) == f"{sources.PLC}/{DID_A}"
    assert sources.did_document_url(DID_B) == "https://bench.example.com/.well-known/did.json"
    assert sources.did_document_url("did:web:host.example%3A8443:u:alice") == "https://host.example:8443/u/alice/did.json"
    with pytest.raises(FetchError):
        sources.did_document_url("did:key:z6Mk")
    http = world()
    assert sources.resolve_pds(DID_A, Source(http=http)) == PDS_A
    assert sources.resolve_pds(DID_B, Source(http=http)) == PDS_B


@pytest.mark.parametrize("doc, needle", [
    (did_doc("did:plc:someoneelse", PDS_A), "is for"),
    (did_doc(DID_A, "http://pds-a.example"), "not https"),
    ({"id": DID_A, "service": []}, "no #atproto_pds"),
])
def test_a_did_document_that_cannot_name_a_pds_is_refused(doc, needle):
    with pytest.raises(FetchError, match=needle):
        sources.pds_of(doc, DID_A)


def test_backfill_discovers_resolves_and_pages_every_repo():
    snap = sources.backfill(Source(http=world()))
    assert [r["uri"] for r in snap.records] == [uri(DID_A, f"3a{i}") for i in range(5)] + [uri(DID_B, "3b0")]
    assert snap.read == [DID_A, DID_B] and snap.skipped == []


def test_repos_skips_discovery_and_pds_skips_resolution():
    http = world()
    snap = sources.backfill(Source(http=http, repos=(DID_A,), pds=PDS_A))
    assert len(snap.records) == 5
    assert not any("listReposByCollection" in u or "plc.directory" in u for u, _ in http.calls)


def test_a_failing_pds_is_skipped_with_its_reason_and_the_rest_are_read():
    http = world(**{f"{PDS_A}/xrpc/com.atproto.repo.listRecords": FetchError("timed out after 10s")})
    snap = sources.backfill(Source(http=http))
    assert [r["uri"] for r in snap.records] == [uri(DID_B, "3b0")]
    assert snap.skipped == [{"did": DID_A, "reason": "timed out after 10s"}]


def test_a_slow_pds_hits_the_repo_deadline_and_is_skipped(monkeypatch):
    clock = iter(range(0, 1000, 40))
    monkeypatch.setattr(sources.time, "monotonic", lambda: next(clock))
    snap = sources.backfill(Source(http=world(), repos=(DID_A,), repo_deadline=60))
    assert snap.records == [] and "over 60s" in snap.skipped[0]["reason"]


def test_a_cursor_that_never_ends_stops_at_max_pages():
    http = world(**{f"{PDS_A}/xrpc/com.atproto.repo.listRecords":
                    lambda p: {"records": [rec(DID_A, "3loop")], "cursor": str(int(p.get("cursor") or 0) + 1)}})
    snap = sources.backfill(Source(http=http, repos=(DID_A,), max_pages=5))
    assert len(snap.records) == 5


def test_a_pds_cannot_list_records_for_another_did():
    http = world(**{f"{PDS_A}/xrpc/com.atproto.repo.listRecords":
                    {"records": [rec(DID_A, "3mine"), rec(DID_B, "3forged"),
                                 {**rec(DID_A, "3x"), "uri": f"at://{DID_A}/app.bsky.feed.post/3x"}]}})
    snap = sources.backfill(Source(http=http, repos=(DID_A,)))
    assert [r["uri"] for r in snap.records] == [uri(DID_A, "3mine")]


def test_get_record_reads_the_authors_pds_and_a_missing_record_is_none():
    body = rec(DID_A, "3a0")
    http = world(**{f"{PDS_A}/xrpc/com.atproto.repo.getRecord":
                    lambda p: body if p["rkey"] == "3a0" else FetchError("HTTP 400", 400, "RecordNotFound")})
    src = Source(http=http)
    assert sources.get_record(uri(DID_A, "3a0"), src)["value"] == body["value"]
    assert sources.get_record(uri(DID_A, "3gone"), src) is None
    http.routes[f"{PDS_A}/xrpc/com.atproto.repo.getRecord"] = FetchError("HTTP 502", 502)
    with pytest.raises(FetchError):
        sources.get_record(uri(DID_A, "3a0"), src)


# ----------------------------------------------------- the real HTTP layer --

class _Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


def test_http_sends_a_timeout_caps_the_body_and_names_xrpc_errors(monkeypatch):
    seen = {}

    def urlopen(req, timeout):
        seen["timeout"], seen["url"] = timeout, req.full_url
        q = urllib.parse.parse_qs(urllib.parse.urlsplit(req.full_url).query)
        if q.get("big"):
            return _Resp(b"x" * 200)
        if q.get("err"):
            raise urllib.error.HTTPError(req.full_url, 400, "Bad", {}, io.BytesIO(b'{"error":"RecordNotFound"}'))
        return _Resp(json.dumps({"ok": True}).encode())
    monkeypatch.setattr(sources.urllib.request, "urlopen", urlopen)
    http = sources.Http(timeout=3.5, max_bytes=100)
    assert http.get_json("https://x.example/a", {"q": 1, "cursor": None}) == {"ok": True}
    assert seen["timeout"] == 3.5 and seen["url"] == "https://x.example/a?q=1"
    with pytest.raises(FetchError, match="over 100 bytes"):
        http.get_json("https://x.example/a", {"big": 1})
    with pytest.raises(FetchError) as exc:
        http.get_json("https://x.example/a", {"err": 1})
    assert exc.value.status == 400 and exc.value.error == "RecordNotFound"


# ------------------------------------------------------------------ CLI --

def test_backfill_cli_writes_points_and_reports_exclusions(tmp_path, monkeypatch, capsys):
    http = world(**{f"{PDS_B}/xrpc/com.atproto.repo.listRecords":
                    {"records": [rec(DID_B, "3short", verified=252, swapped=253)]}})
    monkeypatch.setattr(sources, "Http", lambda timeout: http)
    out = tmp_path / "points.json"
    assert cli.main(["backfill", "-o", str(out)]) == 0
    doc = json.loads(out.read_text())
    assert len(doc["records"]) == 1 and len(doc["records"][0]["site"]["provenance"]) == 5
    assert [e["filter"] for e in doc["excluded"]] == ["F2"] and doc["cursor"] is None
    printed = capsys.readouterr().out
    assert "read 6 records; 5 admitted, collapsed to 1 points; 1 excluded" in printed
    assert "excluded [F2]" in printed


def test_the_f5_knobs_are_flags_defaulting_to_settings(monkeypatch):
    seen = []
    monkeypatch.setattr(cli, "cmd_backfill", lambda a: seen.append(cli.settings_of(a)) or 0)
    cli.main(["backfill", "-o", "x"])
    cli.main(["backfill", "-o", "x", "--baseline-floor", "0.2", "--baseline-relative", "0.5",
              "--baseline-min-contributors", "4", "--baseline-arm", "stock"])
    d, s = seen
    assert (d.baseline_arm, d.baseline_floor, d.baseline_relative, d.baseline_min_contributors) == (
        pipeline.Settings.baseline_arm, pipeline.Settings.baseline_floor,
        pipeline.Settings.baseline_relative, pipeline.Settings.baseline_min_contributors)
    assert (s.baseline_arm, s.baseline_floor, s.baseline_relative, s.baseline_min_contributors) == (
        "stock", 0.2, 0.5, 4)
    assert d.baseline_arm == "stock"
