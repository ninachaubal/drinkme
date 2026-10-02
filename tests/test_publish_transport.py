"""publish/transport.py: https is always fine, plain http is refused
except for loopback hosts, and the escape hatch allows it anyway while
printing a warning."""

import pytest

from drinkme.publish import transport


def test_https_is_always_fine():
    assert transport.require_secure("https://pds.example/xrpc/x", "PDS") == \
        "https://pds.example/xrpc/x"


@pytest.mark.parametrize("url", [
    "http://127.0.0.1:8081", "http://127.0.0.1", "http://localhost:3000",
    "http://[::1]:9000",
])
def test_loopback_http_is_allowed(url):
    assert transport.require_secure(url, "PDS") == url


@pytest.mark.parametrize("url", [
    "http://remote.example", "http://93.184.216.34", "http://pds.internal:8080",
])
def test_remote_http_is_refused(url):
    with pytest.raises(transport.InsecureEndpointError, match="not https"):
        transport.require_secure(url, "PDS")


def test_escape_hatch_allows_and_warns(monkeypatch, capsys):
    monkeypatch.setenv(transport.ALLOW_INSECURE_ENV, "1")
    assert transport.require_secure("http://remote.example", "authorization server") == \
        "http://remote.example"
    err = capsys.readouterr().err
    assert "warning" in err and "http://remote.example" in err and "authorization server" in err


def test_escape_hatch_must_be_exactly_1(monkeypatch):
    monkeypatch.setenv(transport.ALLOW_INSECURE_ENV, "true")
    with pytest.raises(transport.InsecureEndpointError):
        transport.require_secure("http://remote.example", "PDS")
