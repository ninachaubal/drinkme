"""Transport policy for every remote endpoint the publish flow touches:
discovery results (the auth server and its advertised endpoints) and the
chosen PDS. Plain HTTP leaks the DPoP-bound access token and proof to
anyone on the path, so it is refused — except for loopback hosts (a
development PDS on http://127.0.0.1:8081 must keep working) or the
explicit escape hatch below, which is meant to be noticed, not forgotten.
"""

from __future__ import annotations

import os
import sys
import urllib.parse

ALLOW_INSECURE_ENV = "DRINKME_PUBLISH_ALLOW_INSECURE"
_LOOPBACK_HOSTS = {"127.0.0.1", "::1", "localhost"}


class InsecureEndpointError(Exception):
    pass


def _is_loopback(host: str) -> bool:
    return host.lower().rstrip(".") in _LOOPBACK_HOSTS


def require_secure(url: str, what: str) -> str:
    """Raise InsecureEndpointError unless `url` is https, a loopback http
    URL, or DRINKME_PUBLISH_ALLOW_INSECURE=1 is set (which still prints a
    loud warning). Returns `url` unchanged, for chaining."""
    p = urllib.parse.urlsplit(url)
    if p.scheme == "https":
        return url
    if p.scheme == "http" and _is_loopback(p.hostname or ""):
        return url
    if os.environ.get(ALLOW_INSECURE_ENV) == "1":
        print(f"drinkme: warning: {ALLOW_INSECURE_ENV}=1 — allowing insecure {what} "
             f"{url!r}; credentials can be sent in cleartext", file=sys.stderr, flush=True)
        return url
    raise InsecureEndpointError(
        f"{what} {url!r} is not https, and is not a loopback address — refusing to send "
        f"credentials over plain HTTP; set {ALLOW_INSECURE_ENV}=1 to override")
