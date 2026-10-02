"""ES256 keys, compact JWTs, DPoP proofs and PKCE for the OAuth flow — the
only place `cryptography` is touched. No pyjwt: a JWT is
base64url(header).base64url(payload).base64url(signature), and the ES256
signature is the raw 64-byte r||s (RFC 7518 §3.4), not the DER that
`cryptography` hands back, so the conversion lives here in the open.

Every proof and key here is per-session and per-request: the DPoP key is
generated on the first `drinkme publish` and bound to the token set the PDS
issues (RFC 9449); it is never reused across accounts.
"""

from __future__ import annotations

import base64
import hashlib
import json
import secrets
import time
import urllib.parse

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.asymmetric.utils import (
    decode_dss_signature, encode_dss_signature)

_CURVE = ec.SECP256R1()
_COORD = 32  # P-256 coordinate / scalar size in bytes


def b64url(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def b64url_decode(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


def _compact(obj) -> bytes:
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


class Es256Key:
    """One P-256 keypair. `to_jwk(private=True)` is what the session store
    keeps (mode 0600); `to_jwk()` is what goes in the DPoP header."""

    def __init__(self, private: ec.EllipticCurvePrivateKey):
        if not isinstance(private.curve, ec.SECP256R1):
            raise ValueError("ES256 needs a P-256 key")
        self._private = private

    @classmethod
    def generate(cls) -> "Es256Key":
        return cls(ec.generate_private_key(_CURVE))

    @classmethod
    def from_jwk(cls, jwk: dict) -> "Es256Key":
        if jwk.get("kty") != "EC" or jwk.get("crv") != "P-256" or "d" not in jwk:
            raise ValueError("not a private P-256 JWK")
        d = int.from_bytes(b64url_decode(jwk["d"]), "big")
        x = int.from_bytes(b64url_decode(jwk["x"]), "big")
        y = int.from_bytes(b64url_decode(jwk["y"]), "big")
        pub = ec.EllipticCurvePublicNumbers(x, y, _CURVE)
        return cls(ec.EllipticCurvePrivateNumbers(d, pub).private_key())

    def to_jwk(self, private: bool = False) -> dict:
        nums = self._private.public_key().public_numbers()
        jwk = {"kty": "EC", "crv": "P-256",
               "x": b64url(nums.x.to_bytes(_COORD, "big")),
               "y": b64url(nums.y.to_bytes(_COORD, "big"))}
        if private:
            jwk["d"] = b64url(self._private.private_numbers().private_value
                              .to_bytes(_COORD, "big"))
        return jwk

    def thumbprint(self) -> str:
        """RFC 7638 JWK thumbprint (base64url SHA-256 of the canonical public
        members) — what the PDS binds a token to as `jkt`."""
        pub = self.to_jwk()
        canon = json.dumps({k: pub[k] for k in ("crv", "kty", "x", "y")},
                           separators=(",", ":"), sort_keys=True).encode()
        return b64url(hashlib.sha256(canon).digest())

    def sign(self, header: dict, payload: dict) -> str:
        head = b64url(_compact(header))
        body = b64url(_compact(payload))
        signing_input = f"{head}.{body}".encode("ascii")
        der = self._private.sign(signing_input, ec.ECDSA(hashes.SHA256()))
        r, s = decode_dss_signature(der)
        sig = r.to_bytes(_COORD, "big") + s.to_bytes(_COORD, "big")
        return f"{head}.{body}.{b64url(sig)}"


def verify(token: str, public_jwk: dict) -> tuple[dict, dict]:
    """Check an ES256 compact JWT against a public JWK; returns (header,
    payload) or raises ValueError. Used by the tests to prove the signer
    above produces what a verifier expects; the CLI itself never verifies
    (the PDS does)."""
    try:
        head, body, sig = token.split(".")
    except ValueError:
        raise ValueError("not a compact JWT (need three dot-separated parts)") from None
    header = json.loads(b64url_decode(head))
    if header.get("alg") != "ES256":
        raise ValueError(f"alg is {header.get('alg')!r}, not ES256")
    raw = b64url_decode(sig)
    if len(raw) != 2 * _COORD:
        raise ValueError("ES256 signature must be 64 bytes (r||s)")
    r = int.from_bytes(raw[:_COORD], "big")
    s = int.from_bytes(raw[_COORD:], "big")
    x = int.from_bytes(b64url_decode(public_jwk["x"]), "big")
    y = int.from_bytes(b64url_decode(public_jwk["y"]), "big")
    pub = ec.EllipticCurvePublicNumbers(x, y, _CURVE).public_key()
    try:
        pub.verify(encode_dss_signature(r, s), f"{head}.{body}".encode("ascii"),
                   ec.ECDSA(hashes.SHA256()))
    except InvalidSignature:
        raise ValueError("signature does not verify") from None
    return header, json.loads(b64url_decode(body))


def htu(url: str) -> str:
    """The DPoP `htu` claim: the target URI without query or fragment (RFC
    9449 §4.2)."""
    parts = urllib.parse.urlsplit(url)
    return urllib.parse.urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))


def dpop_proof(key: Es256Key, method: str, url: str, nonce: str | None = None,
               access_token: str | None = None, now: int | None = None) -> str:
    """One DPoP proof JWT (RFC 9449 §4): typ dpop+jwt, the public JWK in the
    header, a fresh jti, htm/htu, iat; the server's nonce when we have one;
    `ath` (base64url SHA-256 of the access token) when the proof rides with
    an access token to the resource server — and only then."""
    payload = {"jti": secrets.token_urlsafe(16), "htm": method.upper(), "htu": htu(url),
               "iat": int(time.time()) if now is None else now}
    if nonce:
        payload["nonce"] = nonce
    if access_token:
        payload["ath"] = b64url(hashlib.sha256(access_token.encode("ascii")).digest())
    return key.sign({"typ": "dpop+jwt", "alg": "ES256", "jwk": key.to_jwk()}, payload)


def pkce_pair() -> tuple[str, str]:
    """(code_verifier, code_challenge) for S256 (RFC 7636): a 43-char
    URL-safe verifier and base64url(SHA-256(verifier))."""
    verifier = b64url(secrets.token_bytes(32))
    challenge = b64url(hashlib.sha256(verifier.encode("ascii")).digest())
    return verifier, challenge


def random_state() -> str:
    return b64url(secrets.token_bytes(24))
