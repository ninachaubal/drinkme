"""The three repo calls publish makes, over the DPoP-bound session:
createRecord (validate:false — the PDS holds no copy of our lexicon, the
validator in validate.py already ran), getRecord for the read-back, and
deleteRecord (the e2e driver's cleanup). Plus the canonical-JSON compare
that decides success: the value the PDS hands back must be byte-equal to
what was sent after both are serialised with sorted keys and no
whitespace — a mismatch is a failure, not a warning.
"""

from __future__ import annotations

import json
import urllib.parse

from .oauth import DpopClient, OAuthError


def canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False).encode("utf-8")


def _xrpc(pds: str, nsid: str, params: dict | None = None) -> str:
    url = f"{pds.rstrip('/')}/xrpc/{nsid}"
    return f"{url}?{urllib.parse.urlencode(params)}" if params else url


def create_record(client: DpopClient, session: dict, collection: str, record: dict) -> dict:
    resp = client.request("POST", _xrpc(session["pds"], "com.atproto.repo.createRecord"),
                          json_body={"repo": session["did"], "collection": collection,
                                     "record": record, "validate": False},
                          access_token=session["access_token"])
    body = resp.json()
    if resp.status != 200 or not body.get("uri"):
        raise OAuthError(f"createRecord answered {resp.status} {body.get('error', '')} "
                         f"{body.get('message', '')} {resp.headers.get('www-authenticate', '')}".strip())
    return body


def get_record(client: DpopClient, session: dict, uri: str) -> dict:
    did, collection, rkey = parse_at_uri(uri)
    resp = client.request("GET", _xrpc(session["pds"], "com.atproto.repo.getRecord",
                                       {"repo": did, "collection": collection, "rkey": rkey}),
                          access_token=session["access_token"])
    body = resp.json()
    if resp.status != 200 or "value" not in body:
        raise OAuthError(f"getRecord answered {resp.status} {body.get('error', '')} "
                         f"{body.get('message', '')}".strip())
    return body


def delete_record(client: DpopClient, session: dict, uri: str) -> dict:
    did, collection, rkey = parse_at_uri(uri)
    resp = client.request("POST", _xrpc(session["pds"], "com.atproto.repo.deleteRecord"),
                          json_body={"repo": did, "collection": collection, "rkey": rkey},
                          access_token=session["access_token"])
    if resp.status != 200:
        body = resp.json()
        raise OAuthError(f"deleteRecord answered {resp.status} {body.get('error', '')} "
                         f"{body.get('message', '')}".strip())
    return resp.json()


def parse_at_uri(uri: str) -> tuple[str, str, str]:
    if not uri.startswith("at://"):
        raise ValueError(f"{uri!r} is not an at:// uri")
    parts = uri[5:].split("/")
    if len(parts) != 3 or not all(parts):
        raise ValueError(f"{uri!r} is not at://<did>/<collection>/<rkey>")
    return parts[0], parts[1], parts[2]
