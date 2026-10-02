"""publish/identity.py without a network: the DNS TXT parser on a crafted
response (compression pointers, split character-strings), the did:web
URL mapping, and the resolution chain with the fetchers faked — including
the bidirectional handle check and the private-PLC fallback."""

import struct

import pytest

from drinkme.publish import identity as I


def _dns_response(qname: str, txts: list[bytes], rcode: int = 0) -> bytes:
    q = I._dns_query(qname)
    qid = q[:2]
    header = qid + struct.pack(">H", 0x8180 | rcode) + struct.pack(">HHHH", 1, len(txts), 0, 0)
    question = q[12:]
    answers = b""
    for t in txts:
        # name as a compression pointer to offset 12 (the question name)
        rdata = b"".join(struct.pack("B", len(c)) + c for c in [t[:200], t[200:]] if c or t == b"")
        answers += b"\xc0\x0c" + struct.pack(">HHIH", 16, 1, 300, len(rdata)) + rdata
    return header + question + answers


def test_txt_parser_handles_pointers_and_split_strings():
    msg = _dns_response("_atproto.alice.example", [b"did=did:plc:abc", b"x" * 250])
    got = I.parse_txt_answers(msg)
    assert got == ["did=did:plc:abc", "x" * 250]


def test_txt_parser_nxdomain_is_empty_and_servfail_raises():
    assert I.parse_txt_answers(_dns_response("a.b", [], rcode=3)) == []
    with pytest.raises(I.IdentityError):
        I.parse_txt_answers(_dns_response("a.b", [], rcode=2))


def test_did_web_url():
    assert I.did_web_url("did:web:example.com") == "https://example.com/.well-known/did.json"
    assert I.did_web_url("did:web:example.com%3A8443:u:alice") == "https://example.com:8443/u/alice/did.json"


DOC = {"id": "did:plc:abc", "alsoKnownAs": ["at://alice.example"],
       "service": [{"id": "#atproto_pds", "type": "AtprotoPersonalDataServer",
                    "serviceEndpoint": "https://pds.example/"}]}


def test_resolve_via_dns_then_plc(monkeypatch):
    monkeypatch.setattr(I, "dns_txt", lambda name: ["did=did:plc:abc"] if name == "_atproto.alice.example" else [])
    fetched = []

    def get(url, accept="*/*"):
        fetched.append(url)
        assert url == "https://plc.directory/did:plc:abc"
        return 200, I.json.dumps(DOC).encode()
    monkeypatch.setattr(I, "_http_get", get)
    r = I.resolve(handle="Alice.Example")
    assert r == {"did": "did:plc:abc", "handle": "Alice.Example", "pds": "https://pds.example", "doc": DOC}


def test_resolve_via_well_known_and_private_plc_fallback(monkeypatch):
    monkeypatch.setattr(I, "dns_txt", lambda name: [])
    calls = []

    def get(url, accept="*/*"):
        calls.append(url)
        if url == "https://alice.example/.well-known/atproto-did":
            return 200, b"did:plc:abc\n"
        if url == "https://plc.directory/did:plc:abc":
            return 404, b"not found"
        if url.startswith("https://alice.example/xrpc/com.atproto.repo.describeRepo?repo=did%3Aplc%3Aabc"):
            return 200, I.json.dumps({"didDoc": DOC}).encode()
        raise AssertionError(url)
    monkeypatch.setattr(I, "_http_get", get)
    r = I.resolve(handle="alice.example")
    assert r["did"] == "did:plc:abc" and r["pds"] == "https://pds.example"
    assert calls[1] == "https://plc.directory/did:plc:abc"


def test_explicit_plc_is_used_and_not_fallen_back_from(monkeypatch):
    monkeypatch.setattr(I, "dns_txt", lambda name: ["did=did:plc:abc"])

    def get(url, accept="*/*"):
        assert url == "https://plc.internal/did:plc:abc"
        return 404, b""
    monkeypatch.setattr(I, "_http_get", get)
    with pytest.raises(I.IdentityError, match="plc.internal answered 404"):
        I.resolve(handle="alice.example", plc="https://plc.internal/")


def test_handle_must_be_claimed_back(monkeypatch):
    monkeypatch.setattr(I, "dns_txt", lambda name: ["did=did:plc:abc"])
    doc = dict(DOC, alsoKnownAs=["at://someone.else"])
    monkeypatch.setattr(I, "_http_get", lambda url, accept="*/*": (200, I.json.dumps(doc).encode()))
    with pytest.raises(I.IdentityError, match="does not claim alice.example"):
        I.resolve(handle="alice.example")


def test_conflicting_txt_records_fail_and_did_mismatch_fails(monkeypatch):
    monkeypatch.setattr(I, "dns_txt", lambda name: ["did=did:plc:a", "did=did:plc:b"])
    with pytest.raises(I.IdentityError, match="conflicting"):
        I.resolve_handle("alice.example")
    monkeypatch.setattr(I, "dns_txt", lambda name: ["did=did:plc:a"])
    with pytest.raises(I.IdentityError, match="resolves to did:plc:a, not did:plc:z"):
        I.resolve(handle="alice.example", did="did:plc:z")


def test_did_web_document_and_pds_extraction(monkeypatch):
    doc = {"id": "did:web:pds.example", "alsoKnownAs": [],
           "service": [{"id": "#atproto_pds", "type": "AtprotoPersonalDataServer",
                        "serviceEndpoint": "https://pds.example"}]}
    monkeypatch.setattr(I, "_http_get", lambda url, accept="*/*": (
        200, I.json.dumps(doc).encode()) if url == "https://pds.example/.well-known/did.json" else (500, b""))
    r = I.resolve(did="did:web:pds.example")
    assert r["pds"] == "https://pds.example" and r["handle"] is None
    with pytest.raises(I.IdentityError, match="names no #atproto_pds"):
        I.pds_endpoint({"id": "x", "service": []})


# ------------------------------------------------------ PDS transport ----

def _svc_doc(endpoint: str) -> dict:
    return {"id": "did:plc:abc", "service": [{"id": "#atproto_pds", "type": "AtprotoPersonalDataServer",
                                              "serviceEndpoint": endpoint}]}


def test_pds_selection_refuses_a_remote_http_endpoint():
    with pytest.raises(I.IdentityError, match="PDS 'http://remote.example' is not https"):
        I.pds_endpoint(_svc_doc("http://remote.example"))


def test_pds_selection_allows_loopback_http():
    assert I.pds_endpoint(_svc_doc("http://127.0.0.1:8081")) == "http://127.0.0.1:8081"


def test_pds_selection_escape_hatch_allows_and_warns(monkeypatch, capsys):
    monkeypatch.setenv("DRINKME_PUBLISH_ALLOW_INSECURE", "1")
    assert I.pds_endpoint(_svc_doc("http://remote.example")) == "http://remote.example"
    err = capsys.readouterr().err
    assert "warning" in err and "http://remote.example" in err
