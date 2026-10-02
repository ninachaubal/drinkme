"""publish/jwt.py: the ES256 signer produces what a verifier expects (the
signature is raw r||s, the header says what it is), DPoP proofs carry the
claims RFC 9449 wants and nothing more, PKCE is S256."""

import base64
import hashlib
import json

import pytest

from drinkme.publish import jwt


def _b64d(s):
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


def test_sign_then_verify_with_the_public_jwk():
    key = jwt.Es256Key.generate()
    token = key.sign({"alg": "ES256", "typ": "JWT"}, {"hello": "world", "n": 1})
    header, payload = jwt.verify(token, key.to_jwk())
    assert header == {"alg": "ES256", "typ": "JWT"}
    assert payload == {"hello": "world", "n": 1}
    # raw r||s, 64 bytes — not DER
    assert len(_b64d(token.split(".")[2])) == 64


def test_verify_rejects_a_tampered_payload_and_a_wrong_key():
    key, other = jwt.Es256Key.generate(), jwt.Es256Key.generate()
    token = key.sign({"alg": "ES256"}, {"a": 1})
    h, p, s = token.split(".")
    forged = jwt.b64url(json.dumps({"a": 2}, separators=(",", ":")).encode())
    with pytest.raises(ValueError):
        jwt.verify(f"{h}.{forged}.{s}", key.to_jwk())
    with pytest.raises(ValueError):
        jwt.verify(token, other.to_jwk())


def test_private_jwk_round_trips_and_public_jwk_has_no_d():
    key = jwt.Es256Key.generate()
    again = jwt.Es256Key.from_jwk(key.to_jwk(private=True))
    assert again.to_jwk() == key.to_jwk()
    assert "d" not in key.to_jwk() and "d" in key.to_jwk(private=True)
    assert key.thumbprint() == again.thumbprint()


def test_thumbprint_is_rfc7638():
    key = jwt.Es256Key.generate()
    pub = key.to_jwk()
    canon = json.dumps({"crv": pub["crv"], "kty": "EC", "x": pub["x"], "y": pub["y"]},
                       separators=(",", ":"), sort_keys=True).encode()
    assert key.thumbprint() == jwt.b64url(hashlib.sha256(canon).digest())


def test_dpop_proof_claims():
    key = jwt.Es256Key.generate()
    proof = jwt.dpop_proof(key, "post", "https://pds.example/xrpc/x?y=1#z", nonce="N1",
                           access_token="tok", now=1_700_000_000)
    header, payload = jwt.verify(proof, key.to_jwk())
    assert header["typ"] == "dpop+jwt" and header["alg"] == "ES256"
    assert header["jwk"] == key.to_jwk()  # public only, in the header
    assert payload["htm"] == "POST"
    assert payload["htu"] == "https://pds.example/xrpc/x"  # no query, no fragment
    assert payload["iat"] == 1_700_000_000
    assert payload["nonce"] == "N1"
    assert payload["ath"] == jwt.b64url(hashlib.sha256(b"tok").digest())
    assert len(payload["jti"]) >= 16


def test_dpop_proof_without_token_or_nonce_omits_those_claims():
    key = jwt.Es256Key.generate()
    _, p1 = jwt.verify(jwt.dpop_proof(key, "GET", "https://as.example/par"), key.to_jwk())
    assert "ath" not in p1 and "nonce" not in p1
    _, p2 = jwt.verify(jwt.dpop_proof(key, "GET", "https://as.example/par"), key.to_jwk())
    assert p1["jti"] != p2["jti"]  # unique per proof


def test_pkce_pair_is_s256():
    verifier, challenge = jwt.pkce_pair()
    assert 43 <= len(verifier) <= 128
    assert challenge == jwt.b64url(hashlib.sha256(verifier.encode()).digest())
    assert jwt.pkce_pair()[0] != verifier
