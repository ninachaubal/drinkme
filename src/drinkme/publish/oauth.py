"""The atproto OAuth loopback flow for a CLI, per https://atproto.com/specs/oauth
(read 2026-09-12) — the public-client, "localhost client development" shape:

- client_id is `http://localhost?redirect_uri=<...>&scope=<...>`: the spec's
  virtual client metadata for a client that publishes none (hostname exactly
  `localhost`, no port, empty path; `redirect_uri` and `scope` as query
  parameters). The redirect_uri declared there is `http://127.0.0.1/callback`
  — the spec matches loopback redirect paths and ignores ports, so the
  actual callback uses whatever random port the listener got, and the
  client_id stays the same string from run to run (a refresh has to present
  the client_id the token was issued to). Verified against a self-hosted
  PDS: PAR 201 with this client_id; the consent page shows the client
  as "Development client".
- PKCE S256, PAR (mandatory), DPoP with an ephemeral ES256 key on every
  request to the authorization server and the PDS, server nonces harvested
  from every response and one retry on `use_dpop_nonce` (400 + JSON error at
  the AS; 401 + WWW-Authenticate at the resource server).
- Scope: `atproto` plus the narrowest write the server will grant. The
  granular permission for one collection (permissions spec:
  `repo:<nsid>?action=create`) is tried first at PAR; a server that answers
  invalid_scope gets the transitional `transition:generic` instead, and the
  consent-prep line says which it was and what it means.
- The token response is checked: `sub` is the DID we resolved, `scope`
  contains `atproto`, the issuer is the one the PDS's metadata named.
"""

from __future__ import annotations

import http.server
import json
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

from .. import MEASUREMENT
from . import transport
from .jwt import Es256Key, dpop_proof

HTTP_TIMEOUT = 30.0
CALLBACK_PATH = "/callback"
DECLARED_REDIRECT = f"http://127.0.0.1{CALLBACK_PATH}"
GRANULAR_SCOPE = f"repo:{MEASUREMENT}?action=create"
TRANSITION_SCOPE = "transition:generic"


class OAuthError(Exception):
    pass


# ------------------------------------------------------------- client id ----

def client_id(scope: str) -> str:
    """`http://localhost?redirect_uri=...&scope=...` — the loopback client_id
    (spec: "Localhost Client Development"). Encoded with quote_via=quote so
    the space in the scope is %20 and the colon in a scope value %3A, which
    is what both the reference PDS and a second, self-hosted PDS parse back."""
    q = urllib.parse.urlencode({"redirect_uri": DECLARED_REDIRECT, "scope": scope},
                               quote_via=urllib.parse.quote)
    return f"http://localhost?{q}"


def scope_explained(scope: str) -> str:
    """The consent-prep line: what the requested scope lets us do."""
    if GRANULAR_SCOPE in scope.split():
        return (f"creates records in {MEASUREMENT} and nothing else — the consent "
                f"screen should say exactly that")
    if TRANSITION_SCOPE in scope.split():
        return ("this server offers no per-collection scope, so this grants GENERAL WRITE "
                "access to your repo (any record type, blobs, preferences) — the "
                "\"App Password\" level; drinkme only ever writes one "
                f"{MEASUREMENT} record with it")
    return "authentication only"


# ------------------------------------------------------------- discovery ----

def _get_json(url: str) -> tuple[int, dict, dict]:
    req = urllib.request.Request(url, headers={"accept": "application/json",
                                               "user-agent": "drinkme-publish"})
    try:
        with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as r:
            return r.status, {k.lower(): v for k, v in r.headers.items()}, json.loads(r.read())
    except urllib.error.HTTPError as e:
        body = e.read()
        try:
            return e.code, {k.lower(): v for k, v in e.headers.items()}, json.loads(body)
        except ValueError:
            return e.code, {}, {"error": body[:200].decode("utf-8", "replace")}


def _origin(url: str) -> str:
    p = urllib.parse.urlsplit(url)
    return f"{p.scheme}://{p.netloc}"


def _secure(url: str, what: str) -> str:
    """require_secure, retyped as OAuthError so callers need only catch
    the one exception this module already raises."""
    try:
        return transport.require_secure(url, what)
    except transport.InsecureEndpointError as e:
        raise OAuthError(str(e)) from None


def discover(pds: str) -> dict:
    """PDS -> its authorization server's metadata, both documents checked
    for what the atproto profile requires. Returns the AS metadata with
    `_resource` (the PDS origin) added."""
    pds = _secure(pds.rstrip("/"), "PDS")
    status, _, res = _get_json(f"{pds}/.well-known/oauth-protected-resource")
    if status != 200:
        raise OAuthError(f"{pds}/.well-known/oauth-protected-resource answered {status}")
    servers = res.get("authorization_servers") or []
    if len(servers) != 1:
        raise OAuthError(f"resource metadata names {len(servers)} authorization servers; want 1")
    issuer = _secure(servers[0].rstrip("/"), "authorization server")
    status, _, meta = _get_json(f"{issuer}/.well-known/oauth-authorization-server")
    if status != 200:
        raise OAuthError(f"{issuer}/.well-known/oauth-authorization-server answered {status}")
    if meta.get("issuer", "").rstrip("/") != issuer:
        raise OAuthError(f"AS metadata issuer {meta.get('issuer')!r} != {issuer}")
    for key in ("authorization_endpoint", "token_endpoint", "pushed_authorization_request_endpoint"):
        if not meta.get(key):
            raise OAuthError(f"AS metadata lacks {key}")
        _secure(meta[key], f"AS {key}")
    if "S256" not in (meta.get("code_challenge_methods_supported") or []):
        raise OAuthError("AS does not support PKCE S256")
    if "ES256" not in (meta.get("dpop_signing_alg_values_supported") or []):
        raise OAuthError("AS does not support ES256 DPoP")
    if "atproto" not in (meta.get("scopes_supported") or []):
        raise OAuthError("AS does not advertise the atproto scope")
    meta["_resource"] = _origin(pds)
    return meta


def preferred_scope() -> str:
    """Tried first, always: `scopes_supported` cannot enumerate a
    parameterised permission (a PDS can list only atproto and the
    transition:* scopes yet enforce repo: scopes — seen on a self-hosted
    development PDS), so
    the PAR answer is the probe. invalid_scope -> fallback_scope()."""
    return f"atproto {GRANULAR_SCOPE}"


def fallback_scope() -> str:
    return f"atproto {TRANSITION_SCOPE}"


# ------------------------------------------------------------- DPoP http ----

class Response:
    def __init__(self, status: int, headers: dict, body: bytes):
        self.status, self.headers, self.body = status, headers, body

    def json(self) -> dict:
        try:
            v = json.loads(self.body) if self.body else {}
        except ValueError:
            return {"error": self.body[:300].decode("utf-8", "replace")}
        return v if isinstance(v, dict) else {"value": v}


_REDIRECT_CODES = frozenset({301, 302, 303, 307, 308})


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Refuses to reissue the request elsewhere — `redirect_request`
    returning None makes urllib raise HTTPError(code) instead of
    following, so a 3xx surfaces as an error carrying the target URL
    rather than a second, credentialed request to it."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


_NO_REDIRECT_OPENER = urllib.request.build_opener(_NoRedirect)


class DpopClient:
    """urllib with a DPoP proof on every request and the per-origin nonce
    dance: the nonce in any response is kept; a `use_dpop_nonce` answer is
    retried once with the nonce it delivered. Every request goes through
    an opener that never follows a redirect — see _NoRedirect — because
    the Authorization/DPoP headers above are minted for this URL and must
    not reach whatever origin a 3xx names."""

    def __init__(self, key: Es256Key, nonces: dict | None = None):
        self.key = key
        self.nonces: dict[str, str] = dict(nonces or {})

    def request(self, method: str, url: str, *, form: dict | None = None,
                json_body: dict | None = None, access_token: str | None = None,
                headers: dict | None = None) -> Response:
        origin = _origin(url)
        for attempt in (1, 2):
            resp = self._once(method, url, form, json_body, access_token, headers,
                              self.nonces.get(origin))
            nonce = resp.headers.get("dpop-nonce")
            if nonce:
                self.nonces[origin] = nonce
            if attempt == 1 and nonce and _wants_nonce(resp):
                continue
            return resp
        return resp  # unreachable; keeps type checkers calm

    def _once(self, method, url, form, json_body, access_token, headers, nonce) -> Response:
        data = None
        hdrs = {"accept": "application/json", "user-agent": "drinkme-publish"}
        if form is not None:
            data = urllib.parse.urlencode(form).encode("ascii")
            hdrs["content-type"] = "application/x-www-form-urlencoded"
        elif json_body is not None:
            data = json.dumps(json_body, ensure_ascii=False).encode("utf-8")
            hdrs["content-type"] = "application/json"
        hdrs["DPoP"] = dpop_proof(self.key, method, url, nonce=nonce, access_token=access_token)
        if access_token:
            hdrs["authorization"] = f"DPoP {access_token}"
        hdrs.update(headers or {})
        req = urllib.request.Request(url, data=data, method=method.upper(), headers=hdrs)
        try:
            with _NO_REDIRECT_OPENER.open(req, timeout=HTTP_TIMEOUT) as r:
                return Response(r.status, {k.lower(): v for k, v in r.headers.items()}, r.read())
        except urllib.error.HTTPError as e:
            if e.code in _REDIRECT_CODES:
                target = e.headers.get("location", "<no location>")
                raise OAuthError(
                    f"refusing to follow a redirect from {url} to {target}: the "
                    f"Authorization/DPoP proof above was minted for {url} and cannot be "
                    f"reissued to a different origin") from None
            return Response(e.code, {k.lower(): v for k, v in e.headers.items()}, e.read())


def _wants_nonce(resp: Response) -> bool:
    if resp.status == 400 and resp.json().get("error") == "use_dpop_nonce":
        return True
    if resp.status == 401 and "use_dpop_nonce" in resp.headers.get("www-authenticate", ""):
        return True
    return False


# ------------------------------------------------------------------- PAR ----

def par_request(client: DpopClient, meta: dict, *, scope: str, redirect_uri: str,
                state: str, code_challenge: str, login_hint: str | None) -> dict:
    """POST the authorization request; -> {request_uri, expires_in,
    client_id, scope}. Raises OAuthError('invalid_scope') when the server
    rejects the scope, so the caller can fall back."""
    cid = client_id(scope)
    form = {"client_id": cid, "response_type": "code", "redirect_uri": redirect_uri,
            "scope": scope, "state": state, "code_challenge": code_challenge,
            "code_challenge_method": "S256"}
    if login_hint:
        form["login_hint"] = login_hint
    resp = client.request("POST", meta["pushed_authorization_request_endpoint"], form=form)
    body = resp.json()
    if resp.status not in (200, 201) or not body.get("request_uri"):
        err = body.get("error", "?")
        if err == "invalid_scope":
            raise OAuthError("invalid_scope")
        raise OAuthError(f"PAR answered {resp.status} {err} "
                         f"{body.get('error_description', '')}".strip())
    return {"request_uri": body["request_uri"], "expires_in": body.get("expires_in"),
            "client_id": cid, "scope": scope}


def authorize_url(meta: dict, cid: str, request_uri: str) -> str:
    q = urllib.parse.urlencode({"client_id": cid, "request_uri": request_uri})
    return f"{meta['authorization_endpoint']}?{q}"


# -------------------------------------------------------------- loopback ----

class LoopbackCallback:
    """A one-shot listener on 127.0.0.1:<random> for the redirect. Only a
    request whose `state` matches completes it; anything else gets a 400
    and the listener keeps waiting, so a stray hit on the port cannot end
    the flow or smuggle a code in. An `error` from the server (the user
    pressed Reject) completes it as a failure."""

    _PAGE = ("<!doctype html><meta charset=utf-8><title>drinkme</title>"
             "<body style='font:16px system-ui;margin:3em'><p>{msg}</p>"
             "<p>You can close this tab.</p></body>")

    def __init__(self, state: str, path: str = CALLBACK_PATH):
        self.state, self.path = state, path
        self.result: dict | None = None
        self._done = threading.Event()
        outer = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a):  # quiet
                pass

            def do_GET(self):
                url = urllib.parse.urlsplit(self.path)
                q = {k: v[0] for k, v in urllib.parse.parse_qs(url.query).items()}
                if url.path != outer.path:
                    return self._reply(404, "not the callback path")
                if q.get("state") != outer.state:
                    return self._reply(400, "state mismatch — this response is not for the "
                                            "flow drinkme is running; ignored")
                if outer.result is None:
                    outer.result = q
                    outer._done.set()
                if "error" in q:
                    return self._reply(200, f"Authorization did not complete: {q['error']}")
                self._reply(200, "Authorized. drinkme is finishing up in the terminal.")

            def _reply(self, code: int, msg: str):
                body = outer._PAGE.format(msg=_html(msg)).encode()
                self.send_response(code)
                self.send_header("content-type", "text/html; charset=utf-8")
                self.send_header("content-length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self._server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    @property
    def port(self) -> int:
        return self._server.server_address[1]

    @property
    def redirect_uri(self) -> str:
        return f"http://127.0.0.1:{self.port}{self.path}"

    def start(self) -> "LoopbackCallback":
        self._thread.start()
        return self

    def wait(self, timeout: float) -> dict:
        if not self._done.wait(timeout):
            raise OAuthError(f"no authorization callback within {timeout:.0f}s")
        assert self.result is not None
        return self.result

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()


def _html(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


# ---------------------------------------------------------------- tokens ----

def exchange_code(client: DpopClient, meta: dict, *, cid: str, code: str,
                  redirect_uri: str, code_verifier: str) -> dict:
    form = {"grant_type": "authorization_code", "code": code, "redirect_uri": redirect_uri,
            "client_id": cid, "code_verifier": code_verifier}
    resp = client.request("POST", meta["token_endpoint"], form=form)
    body = resp.json()
    if resp.status != 200 or not body.get("access_token"):
        raise OAuthError(f"token exchange answered {resp.status} {body.get('error', '')} "
                         f"{body.get('error_description', '')}".strip())
    return body


def refresh_tokens(client: DpopClient, meta: dict, *, cid: str, refresh_token: str) -> dict:
    form = {"grant_type": "refresh_token", "refresh_token": refresh_token, "client_id": cid}
    resp = client.request("POST", meta["token_endpoint"], form=form)
    body = resp.json()
    if resp.status != 200 or not body.get("access_token"):
        raise OAuthError(f"token refresh answered {resp.status} {body.get('error', '')} "
                         f"{body.get('error_description', '')}".strip())
    return body


def check_token_response(tokens: dict, *, expected_did: str, issuer: str,
                         requested_scope: str) -> str:
    """The atproto-profile checks on a token response; returns the granted
    scope string. `sub` must be the DID the flow started with; the scope
    must include atproto; the token must be DPoP-bound."""
    if tokens.get("sub") != expected_did:
        raise OAuthError(f"token response is for {tokens.get('sub')!r}, not {expected_did} "
                         f"— the authorization server {issuer} did not authorize the "
                         f"account this flow started with")
    if str(tokens.get("token_type", "")).lower() != "dpop":
        raise OAuthError(f"token_type {tokens.get('token_type')!r} is not DPoP")
    granted = tokens.get("scope")
    if not isinstance(granted, str) or "atproto" not in granted.split():
        raise OAuthError(f"granted scope {granted!r} lacks atproto; rejecting the session")
    want = set(requested_scope.split()) - {"atproto"}
    if want and not (want & set(granted.split())):
        raise OAuthError(f"server granted {granted!r}, not the requested {requested_scope!r}; "
                         f"a write would be refused, so stopping here")
    return granted


def session_from_tokens(tokens: dict, *, key: Es256Key, did: str, pds: str, issuer: str,
                        cid: str, nonces: dict) -> dict:
    return {
        "did": did, "pds": pds, "issuer": issuer, "client_id": cid,
        "scope": tokens.get("scope"), "token_type": tokens.get("token_type"),
        "access_token": tokens["access_token"], "refresh_token": tokens.get("refresh_token"),
        "expires_at": int(time.time()) + int(tokens.get("expires_in") or 300),
        "dpop_key": key.to_jwk(private=True), "nonces": nonces,
        "created_at": int(time.time()),
    }
