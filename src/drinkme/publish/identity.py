"""Handle -> DID -> DID document -> PDS, the atproto identity step the OAuth
spec makes mandatory ("Identity Authentication": resolve independently,
verify the handle bidirectionally, then confirm the server that issues the
token is the one the DID document names).

Handle resolution runs both blessed methods in the specified order: the
`_atproto.<handle>` DNS TXT record, then `https://<handle>/.well-known/
atproto-did`. DNS is a small stdlib client (UDP to the resolvers in
/etc/resolv.conf, TCP on truncation; DNS-over-HTTPS when the machine has no
resolver we can read — the spec names DoH as the alternative for exactly
that case) so `publish` adds no resolver dependency.

DID documents: did:plc from a PLC directory (`--plc`; default
https://plc.directory; and when that directory has never heard of the DID
but the handle's own host served the well-known answer, that host is asked
for the document through com.atproto.repo.describeRepo — a PDS on a private
PLC, which is what the internal stack is, knows its own directory); did:web
from `/.well-known/did.json` on the DID's host.
"""

from __future__ import annotations

import json
import random
import socket
import struct
import urllib.parse
import urllib.request

from . import transport

PLC_DIRECTORY = "https://plc.directory"
DNS_TIMEOUT = 4.0
HTTP_TIMEOUT = 15.0
_DOH = ("https://cloudflare-dns.com/dns-query", "https://dns.google/resolve")


class IdentityError(Exception):
    pass


# ------------------------------------------------------------------ DNS ----

def _dns_query(name: str, qtype: int = 16) -> bytes:
    qid = random.randrange(1 << 16)
    labels = b"".join(struct.pack("B", len(p)) + p.encode("ascii")
                      for p in name.rstrip(".").split("."))
    return struct.pack(">HHHHHH", qid, 0x0100, 1, 0, 0, 0) + labels + b"\0" + struct.pack(">HH", qtype, 1)


def _read_name(msg: bytes, off: int) -> tuple[str, int]:
    parts, jumped, end = [], False, 0
    for _ in range(128):  # bounded: a pointer loop cannot spin us
        n = msg[off]
        if n & 0xC0 == 0xC0:
            if not jumped:
                end = off + 2
            off = ((n & 0x3F) << 8) | msg[off + 1]
            jumped = True
            continue
        off += 1
        if n == 0:
            break
        parts.append(msg[off:off + n].decode("ascii", "replace"))
        off += n
    return ".".join(parts), (end if jumped else off)


def parse_txt_answers(msg: bytes) -> list[str]:
    """TXT rdata strings out of a DNS response, each record's character-
    strings joined. Raises IdentityError on a non-zero RCODE."""
    qid, flags, qd, an, _ns, _ar = struct.unpack(">HHHHHH", msg[:12])
    rcode = flags & 0xF
    if rcode not in (0, 3):  # NXDOMAIN is simply "no record"
        raise IdentityError(f"DNS RCODE {rcode}")
    off = 12
    for _ in range(qd):
        _, off = _read_name(msg, off)
        off += 4
    out = []
    for _ in range(an):
        _, off = _read_name(msg, off)
        rtype, _rclass, _ttl, rdlen = struct.unpack(">HHIH", msg[off:off + 10])
        off += 10
        rdata = msg[off:off + rdlen]
        off += rdlen
        if rtype != 16:
            continue
        chunks, i = [], 0
        while i < len(rdata):
            n = rdata[i]
            chunks.append(rdata[i + 1:i + 1 + n])
            i += 1 + n
        out.append(b"".join(chunks).decode("utf-8", "replace"))
    return out


def _resolvers() -> list[str]:
    try:
        with open("/etc/resolv.conf") as f:
            return [ln.split()[1] for ln in f if ln.startswith("nameserver") and len(ln.split()) > 1]
    except OSError:
        return []


def _dns_txt_local(name: str) -> list[str] | None:
    """None = no resolver answered at all (distinct from 'no record')."""
    for ns in _resolvers():
        fam = socket.AF_INET6 if ":" in ns else socket.AF_INET
        q = _dns_query(name)
        try:
            with socket.socket(fam, socket.SOCK_DGRAM) as s:
                s.settimeout(DNS_TIMEOUT)
                s.sendto(q, (ns, 53))
                msg, _ = s.recvfrom(4096)
            if struct.unpack(">H", msg[:2])[0] != struct.unpack(">H", q[:2])[0]:
                continue
            if struct.unpack(">H", msg[2:4])[0] & 0x0200:  # TC: ask again over TCP
                with socket.create_connection((ns, 53), timeout=DNS_TIMEOUT) as t:
                    t.sendall(struct.pack(">H", len(q)) + q)
                    hdr = t.recv(2)
                    need = struct.unpack(">H", hdr)[0]
                    msg = b""
                    while len(msg) < need:
                        chunk = t.recv(need - len(msg))
                        if not chunk:
                            break
                        msg += chunk
            return parse_txt_answers(msg)
        except (OSError, IdentityError, struct.error):
            continue
    return None


def _dns_txt_doh(name: str) -> list[str]:
    for base in _DOH:
        url = f"{base}?{urllib.parse.urlencode({'name': name, 'type': 'TXT'})}"
        try:
            req = urllib.request.Request(url, headers={"accept": "application/dns-json"})
            with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as r:
                doc = json.loads(r.read())
        except (OSError, ValueError):
            continue
        return [a["data"].strip('"').replace('""', "") for a in doc.get("Answer", [])
                if a.get("type") == 16]
    return []


def dns_txt(name: str) -> list[str]:
    local = _dns_txt_local(name)
    return local if local is not None else _dns_txt_doh(name)


# ------------------------------------------------------------- handles ----

def _http_get(url: str, accept: str = "*/*") -> tuple[int, bytes]:
    req = urllib.request.Request(url, headers={"accept": accept, "user-agent": "drinkme-publish"})
    try:
        with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def resolve_handle(handle: str) -> tuple[str, str | None]:
    """-> (did, well_known_host). DNS first, then the well-known URL; the
    second element is the host that answered over https (None when DNS
    did), a hint for a private PLC later on."""
    handle = handle.strip().lstrip("@").lower().rstrip(".")
    if not handle or "." not in handle:
        raise IdentityError(f"{handle!r} is not a handle (want e.g. alice.example.com)")
    dids = sorted({t[4:].strip() for t in dns_txt(f"_atproto.{handle}") if t.startswith("did=")})
    if len(dids) > 1:
        raise IdentityError(f"{handle}: {len(dids)} conflicting _atproto TXT records")
    if dids and dids[0].startswith("did:"):
        return dids[0], None
    try:
        status, body = _http_get(f"https://{handle}/.well-known/atproto-did", "text/plain")
    except OSError as e:
        raise IdentityError(f"{handle}: no _atproto DNS record, and https://{handle}/"
                            f".well-known/atproto-did failed: {e}") from None
    did = body.decode("utf-8", "replace").strip().splitlines()[0].strip() if body.strip() else ""
    if status != 200 or not did.startswith("did:"):
        raise IdentityError(f"{handle}: no _atproto DNS record, and the well-known "
                            f"lookup answered {status} {body[:80]!r}")
    return did, handle


# ------------------------------------------------------------------ DIDs ----

def did_web_url(did: str) -> str:
    """did:web:host[:path...] -> the did.json URL (W3C did:web §3.2)."""
    parts = did.split(":")[2:]
    host = urllib.parse.unquote(parts[0])
    if len(parts) == 1:
        return f"https://{host}/.well-known/did.json"
    return f"https://{host}/{'/'.join(urllib.parse.unquote(p) for p in parts[1:])}/did.json"


def resolve_did(did: str, plc: str | None = None, hint_host: str | None = None) -> dict:
    if did.startswith("did:web:"):
        status, body = _http_get(did_web_url(did), "application/json")
        if status != 200:
            raise IdentityError(f"{did}: {did_web_url(did)} answered {status}")
        doc = json.loads(body)
    elif did.startswith("did:plc:"):
        directory = (plc or PLC_DIRECTORY).rstrip("/")
        status, body = _http_get(f"{directory}/{did}", "application/json")
        if status == 200:
            doc = json.loads(body)
        elif status == 404 and hint_host and not plc:
            # The public directory never heard of it and nobody named a
            # private one; the host that vouched for the handle is a PDS
            # (or fronts one) and can hand over the document it holds.
            q = urllib.parse.urlencode({"repo": did})
            s2, b2 = _http_get(f"https://{hint_host}/xrpc/com.atproto.repo.describeRepo?{q}",
                               "application/json")
            if s2 != 200:
                raise IdentityError(f"{did}: not in {directory} (404) and "
                                    f"https://{hint_host} describeRepo answered {s2}; "
                                    f"pass --plc <directory url>")
            doc = json.loads(b2).get("didDoc") or {}
        else:
            raise IdentityError(f"{did}: {directory} answered {status}; is that the "
                                f"right PLC directory? (--plc URL)")
    else:
        raise IdentityError(f"{did}: only did:plc and did:web are supported")
    if doc.get("id") != did:
        raise IdentityError(f"DID document id {doc.get('id')!r} != {did}")
    return doc


def pds_endpoint(doc: dict) -> str:
    for svc in doc.get("service") or []:
        if (svc.get("id", "").endswith("#atproto_pds")
                and svc.get("type") == "AtprotoPersonalDataServer"):
            url = svc.get("serviceEndpoint", "")
            if isinstance(url, str) and url.startswith(("https://", "http://")):
                url = url.rstrip("/")
                try:
                    return transport.require_secure(url, "PDS")
                except transport.InsecureEndpointError as e:
                    raise IdentityError(str(e)) from None
    raise IdentityError(f"{doc.get('id')}: DID document names no #atproto_pds service")


def handle_claimed(doc: dict, handle: str) -> bool:
    return f"at://{handle.lower()}" in [a.lower() for a in doc.get("alsoKnownAs") or []]


def resolve(handle: str | None = None, did: str | None = None,
            plc: str | None = None) -> dict:
    """The whole chain: -> {did, handle, pds, doc}. With a handle, the DID
    document must claim it back (bidirectional check); with only a DID, the
    handle shown is whatever the document claims."""
    hint = None
    if handle:
        did_found, hint = resolve_handle(handle)
        if did and did != did_found:
            raise IdentityError(f"{handle} resolves to {did_found}, not {did}")
        did = did_found
    if not did:
        raise IdentityError("need a handle or a DID")
    doc = resolve_did(did, plc, hint)
    if handle and not handle_claimed(doc, handle.strip().lstrip("@").lower().rstrip(".")):
        raise IdentityError(f"{did} does not claim {handle} (alsoKnownAs: "
                            f"{doc.get('alsoKnownAs')})")
    aka = [a[5:] for a in doc.get("alsoKnownAs") or [] if a.startswith("at://")]
    return {"did": did, "handle": (handle or (aka[0] if aka else None)),
            "pds": pds_endpoint(doc), "doc": doc}
