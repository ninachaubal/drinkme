"""Where published records are found: the relay, DID documents, and the authors' PDSes.

Read-only GETs, every one with a timeout and a size cap (`Http`). Discovery
is `com.atproto.sync.listReposByCollection` on a relay, trying each relay in
turn; each DID's PDS comes from its DID document (plc.directory for did:plc,
`/.well-known/did.json` for did:web), and its records from
`com.atproto.repo.listRecords` there. A PDS that is slow, down, or answers
nonsense is skipped with its reason and never stops the run; `backfill`
reports what it skipped. A record is accepted only under the at-uri of the
repository it was asked for, so one PDS cannot speak for another DID.
"""
from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field

from .pipeline import NSID, parse_uri

RELAYS = ("https://relay1.us-east.bsky.network", "https://relay1.us-west.bsky.network")
PLC = "https://plc.directory"
USER_AGENT = "drinkme-aggregator (+https://tangled.org/ninachaubal.com/drinkme)"


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


class FetchError(Exception):
    """A GET that failed: unreachable, timed out, non-2xx, too big, or not JSON."""

    def __init__(self, msg: str, status: int | None = None, error: str | None = None):
        super().__init__(msg)
        self.status = status
        self.error = error  # the XRPC error name, when the server sent one


@dataclass
class Http:
    """GET JSON with a timeout on every call, a wall-clock limit on reading
    the body (a server dripping bytes under the socket timeout still stops),
    and a cap on its size."""
    timeout: float = 10.0
    max_bytes: int = 16 << 20

    def _read(self, resp, url: str) -> bytes:
        chunks, size, deadline = [], 0, time.monotonic() + 3 * self.timeout
        while True:
            chunk = resp.read(1 << 16)
            if not chunk:
                return b"".join(chunks)
            chunks.append(chunk)
            size += len(chunk)
            if size > self.max_bytes:
                raise FetchError(f"{url}: response over {self.max_bytes} bytes")
            if time.monotonic() > deadline:
                raise FetchError(f"{url}: body took over {3 * self.timeout:g}s")

    def get_json(self, url: str, params: dict | None = None) -> dict:
        if params:
            url += "?" + urllib.parse.urlencode({k: v for k, v in params.items() if v is not None})
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = self._read(resp, url)
        except urllib.error.HTTPError as exc:
            try:
                error = json.loads(exc.read(4096)).get("error")
            except Exception:
                error = None
            raise FetchError(f"HTTP {exc.code}{' ' + error if error else ''} from {url}", exc.code, error) from None
        except (urllib.error.URLError, OSError, ValueError) as exc:
            raise FetchError(f"{type(exc).__name__} fetching {url}: {getattr(exc, 'reason', exc)}") from None
        try:
            doc = json.loads(body)
        except ValueError:
            raise FetchError(f"{url}: not JSON") from None
        if not isinstance(doc, dict):
            raise FetchError(f"{url}: not a JSON object")
        return doc


@dataclass
class Source:
    """How to reach the network. `repos` skips discovery; `pds` skips DID
    resolution (every repo is read from that one PDS: for development)."""
    http: Http = field(default_factory=Http)
    relays: tuple = RELAYS
    plc: str = PLC
    repos: tuple | None = None
    pds: str | None = None
    repo_deadline: float = 60.0  # seconds one repository may take across all its pages
    max_pages: int = 1000        # pages of one listing, so a cursor that never ends cannot loop


# ------------------------------------------------------------ discovery --

def discover(src: Source) -> list[str]:
    """Every DID the relay lists as holding our collection; the next relay
    if one fails. Raises FetchError if every relay fails."""
    errors = []
    for relay in src.relays:
        try:
            dids, cursor, seen = [], None, set()
            for _ in range(src.max_pages):
                page = src.http.get_json(f"{relay.rstrip('/')}/xrpc/com.atproto.sync.listReposByCollection",
                                         {"collection": NSID, "limit": 2000, "cursor": cursor})
                for repo in page.get("repos") or []:
                    did = repo.get("did") if isinstance(repo, dict) else None
                    if isinstance(did, str) and did.startswith("did:") and did not in seen:
                        seen.add(did)
                        dids.append(did)
                cursor = page.get("cursor")
                if not cursor or not page.get("repos"):
                    break
            log(f"relay {relay}: {len(dids)} repos hold {NSID}")
            return dids
        except FetchError as exc:
            log(f"relay {relay} failed: {exc}")
            errors.append(str(exc))
    raise FetchError("every relay failed: " + "; ".join(errors))


def did_document_url(did: str, plc: str = PLC) -> str:
    if did.startswith("did:plc:"):
        return f"{plc.rstrip('/')}/{did}"
    if did.startswith("did:web:"):
        # did:web:host[:path...], the host percent-encoded (a port is %3A)
        host, *path = did[len("did:web:"):].split(":")
        host = urllib.parse.unquote(host)
        if not host or "/" in host:
            raise FetchError(f"{did}: not a did:web host")
        tail = "/".join(urllib.parse.unquote(p) for p in path)
        return f"https://{host}/{tail}/did.json" if tail else f"https://{host}/.well-known/did.json"
    raise FetchError(f"{did}: unsupported DID method")


def pds_of(doc: dict, did: str) -> str:
    """The `#atproto_pds` service endpoint of a DID document. HTTPS only."""
    if doc.get("id") != did:
        raise FetchError(f"{did}: the DID document is for {doc.get('id')!r}")
    for svc in doc.get("service") or []:
        if not isinstance(svc, dict):
            continue
        if str(svc.get("id", "")).endswith("#atproto_pds") and svc.get("type") == "AtprotoPersonalDataServer":
            ep = svc.get("serviceEndpoint")
            if isinstance(ep, str) and ep.startswith("https://"):
                return ep.rstrip("/")
            raise FetchError(f"{did}: #atproto_pds endpoint {ep!r} is not https")
    raise FetchError(f"{did}: no #atproto_pds service in its DID document")


def resolve_pds(did: str, src: Source) -> str:
    if src.pds:
        return str(src.pds).rstrip("/")
    return pds_of(src.http.get_json(did_document_url(did, src.plc)), did)


# -------------------------------------------------------------- records --

def _own(rec, did: str) -> bool:
    """A listed record is the asked repository's own record in our collection."""
    if not isinstance(rec, dict):
        return False
    parsed = parse_uri(rec.get("uri"))
    return parsed is not None and parsed[0] == did and parsed[1] == NSID


def list_records(did: str, src: Source) -> list[dict]:
    """Every record of our collection in `did`'s repository, as {uri, cid, value}."""
    pds = resolve_pds(did, src)
    out, cursor, start, foreign = [], None, time.monotonic(), 0
    for _ in range(src.max_pages):
        if time.monotonic() - start > src.repo_deadline:
            raise FetchError(f"{did}: listing took over {src.repo_deadline:g}s")
        page = src.http.get_json(f"{pds}/xrpc/com.atproto.repo.listRecords",
                                 {"repo": did, "collection": NSID, "limit": 100, "cursor": cursor})
        recs = page.get("records") or []
        for rec in recs:
            if _own(rec, did):
                out.append({"uri": rec["uri"], "cid": rec.get("cid"), "value": rec.get("value")})
            else:
                foreign += 1
        new = page.get("cursor")
        if not new or not recs or new == cursor:
            break
        cursor = new
    if foreign:
        log(f"{did}: ignored {foreign} listed records not under at://{did}/{NSID}/")
    return out


def get_record(uri: str, src: Source) -> dict | None:
    """The record at `uri`, read from its author's PDS; None if it is gone."""
    did, collection, rkey = parse_uri(uri)
    pds = resolve_pds(did, src)
    try:
        rec = src.http.get_json(f"{pds}/xrpc/com.atproto.repo.getRecord",
                                {"repo": did, "collection": collection, "rkey": rkey})
    except FetchError as exc:
        # a deleted record: the XRPC error RecordNotFound (a 400), or a 404
        if exc.status == 404 or exc.error == "RecordNotFound":
            return None
        raise
    if rec.get("uri") != uri:
        raise FetchError(f"{uri}: the PDS answered with {rec.get('uri')!r}")
    return {"uri": uri, "cid": rec.get("cid"), "value": rec.get("value")}


@dataclass
class Snapshot:
    records: list        # {uri, cid, value}
    read: list           # DIDs read in full
    skipped: list        # {did, reason}


def backfill(src: Source) -> Snapshot:
    """Every record of every repository: discovered on the relay, or `src.repos`."""
    dids = list(src.repos) if src.repos else discover(src)
    records, read, skipped = [], [], []
    for did in dids:
        try:
            recs = list_records(did, src)
        except FetchError as exc:
            log(f"skipped {did}: {exc}")
            skipped.append({"did": did, "reason": str(exc)})
            continue
        log(f"{did}: {len(recs)} records")
        records.extend(recs)
        read.append(did)
    return Snapshot(records, read, skipped)
