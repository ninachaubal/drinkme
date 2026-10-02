"""publish/oauth.py without a network: the loopback client_id form, the
PAR form, the loopback listener (a wrong state is refused and the flow
keeps waiting), the DPoP nonce dance against a local server that checks
every proof with the key from its header, and the atproto-profile checks
on a token response."""

import http.server
import json
import threading
import urllib.parse
import urllib.request

import pytest

from drinkme import MEASUREMENT
from drinkme.publish import jwt, oauth


# ------------------------------------------------------------ client id ----

def test_client_id_is_the_localhost_form_with_redirect_and_scope():
    cid = oauth.client_id("atproto transition:generic")
    u = urllib.parse.urlsplit(cid)
    assert u.scheme == "http" and u.netloc == "localhost" and u.path == ""  # no port, no path
    q = urllib.parse.parse_qs(u.query)
    assert q == {"redirect_uri": ["http://127.0.0.1/callback"], "scope": ["atproto transition:generic"]}
    # %20 for the space, %3A for the colon: what the servers parse back
    assert "scope=atproto%20transition%3Ageneric" in cid
    assert "redirect_uri=http%3A%2F%2F127.0.0.1%2Fcallback" in cid


def test_client_id_is_stable_across_runs_no_port_in_it():
    assert oauth.client_id("atproto") == oauth.client_id("atproto")
    assert ":4" not in oauth.client_id("atproto")


def test_scopes():
    assert oauth.preferred_scope() == f"atproto repo:{MEASUREMENT}?action=create"
    assert oauth.fallback_scope() == "atproto transition:generic"
    assert "nothing else" in oauth.scope_explained(oauth.preferred_scope())
    assert "GENERAL WRITE" in oauth.scope_explained(oauth.fallback_scope())


# ----------------------------------------------------------------- PAR ----

class _Capture(oauth.DpopClient):
    def __init__(self, status=201, body=None):
        super().__init__(jwt.Es256Key.generate())
        self.calls = []
        self._status, self._body = status, body or {"request_uri": "urn:x", "expires_in": 60}

    def request(self, method, url, **kw):
        self.calls.append((method, url, kw))
        return oauth.Response(self._status, {}, json.dumps(self._body).encode())


META = {"issuer": "https://as.example", "pushed_authorization_request_endpoint": "https://as.example/par",
        "authorization_endpoint": "https://as.example/authorize", "token_endpoint": "https://as.example/token"}


def test_par_form_carries_the_profile_fields():
    c = _Capture()
    out = oauth.par_request(c, META, scope="atproto x", redirect_uri="http://127.0.0.1:5555/callback",
                            state="S", code_challenge="C", login_hint="alice.example")
    (method, url, kw), = c.calls
    assert method == "POST" and url == "https://as.example/par"
    form = kw["form"]
    assert form == {"client_id": oauth.client_id("atproto x"), "response_type": "code",
                    "redirect_uri": "http://127.0.0.1:5555/callback", "scope": "atproto x",
                    "state": "S", "code_challenge": "C", "code_challenge_method": "S256",
                    "login_hint": "alice.example"}
    assert "client_secret" not in form and kw.get("access_token") is None
    assert out["request_uri"] == "urn:x" and out["client_id"] == form["client_id"]
    url = oauth.authorize_url(META, out["client_id"], out["request_uri"])
    q = urllib.parse.parse_qs(urllib.parse.urlsplit(url).query)
    assert q == {"client_id": [form["client_id"]], "request_uri": ["urn:x"]}
    assert url.startswith("https://as.example/authorize?")


def test_par_invalid_scope_is_a_distinct_error_for_the_fallback():
    c = _Capture(400, {"error": "invalid_scope", "error_description": "no"})
    with pytest.raises(oauth.OAuthError, match="^invalid_scope$"):
        oauth.par_request(c, META, scope="atproto repo:x", redirect_uri="r", state="s",
                          code_challenge="c", login_hint=None)
    c = _Capture(400, {"error": "invalid_request", "error_description": "bad redirect"})
    with pytest.raises(oauth.OAuthError, match="invalid_request bad redirect"):
        oauth.par_request(c, META, scope="atproto", redirect_uri="r", state="s",
                          code_challenge="c", login_hint=None)


# ------------------------------------------------------------ loopback ----

def _get(url):
    try:
        with urllib.request.urlopen(url, timeout=5) as r:
            return r.status, r.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()


def test_loopback_rejects_wrong_state_then_accepts_the_right_one():
    lb = oauth.LoopbackCallback("GOOD").start()
    try:
        assert lb.redirect_uri == f"http://127.0.0.1:{lb.port}/callback"
        st, body = _get(f"{lb.redirect_uri}?code=EVIL&state=BAD")
        assert st == 400 and "state mismatch" in body
        assert lb.result is None  # nothing accepted, still waiting
        st, _ = _get(f"http://127.0.0.1:{lb.port}/elsewhere?code=x&state=GOOD")
        assert st == 404 and lb.result is None
        st, body = _get(f"{lb.redirect_uri}?code=C1&state=GOOD&iss=https%3A%2F%2Fas.example")
        assert st == 200 and "Authorized" in body
        assert lb.wait(1) == {"code": "C1", "state": "GOOD", "iss": "https://as.example"}
        # a second hit cannot overwrite the first
        _get(f"{lb.redirect_uri}?code=C2&state=GOOD")
        assert lb.result["code"] == "C1"
    finally:
        lb.close()


def test_loopback_binds_127_0_0_1_only_and_times_out():
    lb = oauth.LoopbackCallback("S").start()
    try:
        assert lb._server.server_address[0] == "127.0.0.1"
        with pytest.raises(oauth.OAuthError, match="no authorization callback"):
            lb.wait(0.05)
    finally:
        lb.close()


def test_loopback_error_from_the_server_completes_as_failure():
    lb = oauth.LoopbackCallback("S").start()
    try:
        st, body = _get(f"{lb.redirect_uri}?error=access_denied&state=S")
        assert st == 200 and "did not complete" in body
        assert lb.wait(1)["error"] == "access_denied"
    finally:
        lb.close()


# ---------------------------------------------------------- DPoP nonce ----

class _NonceServer:
    """Answers use_dpop_nonce until the proof carries the current nonce;
    verifies every proof's signature with the JWK in its own header, and
    records the claims it saw."""

    def __init__(self, resource_style: bool):
        outer = self
        self.nonce = "n-1"
        self.seen = []

        class H(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_POST(self):
                self.do_GET()

            def do_GET(self):
                n = int(self.headers.get("content-length") or 0)
                self.rfile.read(n)
                proof = self.headers.get("DPoP")
                header, payload = jwt.verify(proof, json.loads(
                    jwt.b64url_decode(proof.split(".")[0]))["jwk"])
                outer.seen.append((header, payload, self.headers.get("authorization")))
                ok = payload.get("nonce") == outer.nonce
                if resource_style:
                    code = 200 if ok else 401
                    body = b'{"ok":true}' if ok else b'{"error":"use_dpop_nonce"}'
                else:
                    code = 200 if ok else 400
                    body = b'{"ok":true}' if ok else b'{"error":"use_dpop_nonce"}'
                self.send_response(code)
                self.send_header("content-type", "application/json")
                self.send_header("DPoP-Nonce", outer.nonce)
                if not ok and resource_style:
                    self.send_header("WWW-Authenticate", 'DPoP error="use_dpop_nonce"')
                self.send_header("content-length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self.server = http.server.HTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()


@pytest.mark.parametrize("resource_style", [False, True])
def test_dpop_nonce_retry_once_then_remembers(resource_style):
    srv = _NonceServer(resource_style)
    try:
        client = oauth.DpopClient(jwt.Es256Key.generate())
        r = client.request("POST", f"{srv.url}/xrpc/thing?q=1", json_body={"a": 1},
                           access_token="TOK" if resource_style else None)
        assert r.status == 200 and r.json() == {"ok": True}
        assert len(srv.seen) == 2  # one refusal, one retry — not a loop
        first, second = srv.seen[0][1], srv.seen[1][1]
        assert "nonce" not in first and second["nonce"] == "n-1"
        assert second["htm"] == "POST" and second["htu"] == f"{srv.url}/xrpc/thing"
        assert first["jti"] != second["jti"]
        if resource_style:
            assert srv.seen[1][2] == "DPoP TOK" and "ath" in second
        else:
            assert srv.seen[1][2] is None and "ath" not in second
        # the nonce is remembered per origin: the next call needs no retry
        r = client.request("GET", f"{srv.url}/xrpc/other", access_token="TOK" if resource_style else None)
        assert r.status == 200 and len(srv.seen) == 3
        assert client.nonces[srv.url] == "n-1"
        # a rotated nonce costs exactly one more retry
        srv.nonce = "n-2"
        r = client.request("GET", f"{srv.url}/xrpc/other", access_token="TOK" if resource_style else None)
        assert r.status == 200 and len(srv.seen) == 5 and client.nonces[srv.url] == "n-2"
    finally:
        srv.close()


# --------------------------------------------------------------- tokens ----

def test_token_response_checks():
    good = {"sub": "did:plc:abc", "token_type": "DPoP", "scope": "atproto repo:x?action=create",
            "access_token": "a"}
    assert oauth.check_token_response(good, expected_did="did:plc:abc", issuer="i",
                                      requested_scope="atproto repo:x?action=create") == good["scope"]
    with pytest.raises(oauth.OAuthError, match="not did:plc:abc"):
        oauth.check_token_response({**good, "sub": "did:plc:other"}, expected_did="did:plc:abc",
                                   issuer="i", requested_scope="atproto")
    with pytest.raises(oauth.OAuthError, match="not DPoP"):
        oauth.check_token_response({**good, "token_type": "Bearer"}, expected_did="did:plc:abc",
                                   issuer="i", requested_scope="atproto")
    with pytest.raises(oauth.OAuthError, match="lacks atproto"):
        oauth.check_token_response({**good, "scope": "transition:generic"}, expected_did="did:plc:abc",
                                   issuer="i", requested_scope="atproto")
    with pytest.raises(oauth.OAuthError, match="granted 'atproto', not the requested"):
        oauth.check_token_response({**good, "scope": "atproto"}, expected_did="did:plc:abc",
                                   issuer="i", requested_scope="atproto repo:x?action=create")


def test_session_from_tokens_keeps_the_private_key_and_expiry():
    key = jwt.Es256Key.generate()
    s = oauth.session_from_tokens({"access_token": "A", "refresh_token": "R", "expires_in": 300,
                                   "scope": "atproto", "token_type": "DPoP"},
                                  key=key, did="did:plc:x", pds="https://pds", issuer="https://as",
                                  cid="http://localhost?x", nonces={"https://as": "n"})
    assert s["dpop_key"] == key.to_jwk(private=True)
    assert s["expires_at"] > s["created_at"] and s["refresh_token"] == "R"
    assert s["client_id"] == "http://localhost?x" and s["nonces"] == {"https://as": "n"}


# --------------------------------------------------------- redirects ----

def test_dpop_client_refuses_to_follow_a_redirect_across_origins():
    """A 302 to a different origin must not carry the
    Authorization/DPoP headers with it — the proof was minted for the
    source URL. Plain urlopen follows the redirect: through it, this test
    fails with 'DID NOT RAISE' and `hits` holds the leaked headers."""
    hits = []

    class H2(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            hits.append({"authorization": self.headers.get("authorization"),
                        "dpop": self.headers.get("dpop")})
            body = b'{"ok":true}'
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    srv2 = http.server.HTTPServer(("127.0.0.1", 0), H2)
    threading.Thread(target=srv2.serve_forever, daemon=True).start()
    target = f"http://127.0.0.1:{srv2.server_address[1]}/elsewhere"

    class H1(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            self.send_response(302)
            self.send_header("location", target)
            self.send_header("content-length", "0")
            self.end_headers()

    srv1 = http.server.HTTPServer(("127.0.0.1", 0), H1)
    threading.Thread(target=srv1.serve_forever, daemon=True).start()
    source = f"http://127.0.0.1:{srv1.server_address[1]}/start"

    try:
        client = oauth.DpopClient(jwt.Es256Key.generate())
        with pytest.raises(oauth.OAuthError, match="refusing to follow a redirect") as ei:
            client.request("GET", source, access_token="FAKE-REVIEW-TOKEN")
        assert source in str(ei.value) and target in str(ei.value)
        assert hits == []  # the second origin never even saw the request
    finally:
        srv1.shutdown()
        srv1.server_close()
        srv2.shutdown()
        srv2.server_close()


# --------------------------------------------------------- discovery transport ----

class _JsonServer:
    """A loopback http server that answers a fixed map of path -> JSON body."""

    def __init__(self):
        self.routes: dict = {}
        outer = self

        class H(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                found = outer.routes.get(self.path)
                if found is None:
                    self.send_response(404)
                    self.end_headers()
                    return
                data = json.dumps(found).encode()
                self.send_response(200)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self.server = http.server.HTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()


def _as_metadata(as_url: str) -> dict:
    return {"issuer": as_url, "authorization_endpoint": f"{as_url}/authorize",
            "token_endpoint": f"{as_url}/token", "pushed_authorization_request_endpoint": f"{as_url}/par",
            "code_challenge_methods_supported": ["S256"], "dpop_signing_alg_values_supported": ["ES256"],
            "scopes_supported": ["atproto", "transition:generic"]}


def test_discover_still_works_against_a_loopback_http_pds_and_as():
    """A development PDS on http://127.0.0.1:8081 — loopback http
    must keep working end to end through discover()."""
    pds_srv, as_srv = _JsonServer(), _JsonServer()
    try:
        as_srv.routes["/.well-known/oauth-authorization-server"] = _as_metadata(as_srv.url)
        pds_srv.routes["/.well-known/oauth-protected-resource"] = {"authorization_servers": [as_srv.url]}
        meta = oauth.discover(pds_srv.url)
        assert meta["issuer"] == as_srv.url and meta["_resource"] == pds_srv.url
    finally:
        pds_srv.close()
        as_srv.close()


def test_discover_refuses_an_insecure_pds_before_any_network():
    with pytest.raises(oauth.OAuthError, match="PDS 'http://remote.example' is not https"):
        oauth.discover("http://remote.example")


def test_discover_refuses_an_insecure_authorization_server():
    pds_srv = _JsonServer()
    try:
        pds_srv.routes["/.well-known/oauth-protected-resource"] = {
            "authorization_servers": ["http://remote-as.example"]}
        with pytest.raises(oauth.OAuthError, match="authorization server 'http://remote-as.example'"):
            oauth.discover(pds_srv.url)
    finally:
        pds_srv.close()


def test_discover_refuses_an_insecure_as_endpoint():
    pds_srv, as_srv = _JsonServer(), _JsonServer()
    try:
        meta = _as_metadata(as_srv.url)
        meta["token_endpoint"] = "http://remote-token.example/token"
        as_srv.routes["/.well-known/oauth-authorization-server"] = meta
        pds_srv.routes["/.well-known/oauth-protected-resource"] = {"authorization_servers": [as_srv.url]}
        with pytest.raises(oauth.OAuthError, match="AS token_endpoint 'http://remote-token.example/token'"):
            oauth.discover(pds_srv.url)
    finally:
        pds_srv.close()
        as_srv.close()


def test_discover_escape_hatch_allows_insecure_pds_with_a_warning(monkeypatch, capsys):
    monkeypatch.setenv("DRINKME_PUBLISH_ALLOW_INSECURE", "1")
    monkeypatch.setattr(oauth, "_get_json", lambda url: (
        (200, {}, {"authorization_servers": ["https://as.example"]}) if "protected-resource" in url
        else (200, {}, _as_metadata("https://as.example"))))
    meta = oauth.discover("http://remote.example")
    assert meta["issuer"] == "https://as.example"
    err = capsys.readouterr().err
    assert "warning" in err and "allowing insecure" in err and "http://remote.example" in err
