"""bench/serve_drive.py's served-ruler check (docs/bench.md): a served number
is taken against the server AS SHIPPED, so the driver must refuse to record
one against a server whose /health reports the prefix cache off, unless the
caller explicitly says --allow-cache-off — and then the label carries the
fact rather than hiding it.

Only check_shipped() and the /health fetch are exercised; nothing here
starts a real drinkme server, loads a model, or touches a GPU. The fake
/health server follows tests/test_serving_http.py's fake-server pattern:
a real socket, a real HTTP response, torn down after the test.
"""

import http.server
import importlib.util
import json
import os
import threading

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
SERVE_DRIVE = os.path.join(HERE, "..", "bench", "serve_drive.py")


def _load():
    spec = importlib.util.spec_from_file_location("serve_drive_under_test", SERVE_DRIVE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def serve_drive():
    return _load()


@pytest.fixture
def fake_health():
    """fake_health(body_dict) -> port; a GET-anything server that always
    answers /health with the given JSON. Shut down after the test."""
    ports = []

    def make(body: dict):
        payload = json.dumps(body).encode()

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, fmt, *args):
                pass

        srv = http.server.HTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        ports.append(srv)
        return srv.server_address[1]

    yield make
    for srv in ports:
        srv.shutdown()
        srv.server_close()


def test_refuses_a_served_label_when_the_cache_is_off(serve_drive, fake_health):
    port = fake_health({"ok": True, "device": "cuda:0", "prefix_cache": {"slots": 0}})
    h = serve_drive.health(port)
    with pytest.raises(SystemExit, match="prefix_cache.slots=0"):
        serve_drive.check_shipped(h, "sip", require_shipped=True, allow_cache_off=False)


def test_allow_cache_off_marks_the_label_instead_of_refusing(serve_drive, fake_health):
    port = fake_health({"ok": True, "device": "cuda:0", "prefix_cache": {"slots": 0}})
    h = serve_drive.health(port)
    label = serve_drive.check_shipped(h, "sip", require_shipped=True, allow_cache_off=True)
    assert label == "sip-cache-off"


def test_a_shipped_cache_passes_unmarked(serve_drive, fake_health):
    port = fake_health({"ok": True, "device": "cuda:0", "prefix_cache": {"slots": 1}})
    h = serve_drive.health(port)
    label = serve_drive.check_shipped(h, "sip", require_shipped=True, allow_cache_off=False)
    assert label == "sip"


def test_a_server_that_does_not_expose_the_field_passes_unmarked(serve_drive, fake_health):
    # docs/bench.md's rule is conditional on the server exposing the field;
    # an older server or an engine with no concept of a prefix cache is not
    # evidence the cache is off.
    port = fake_health({"ok": True, "device": "cuda:0"})
    h = serve_drive.health(port)
    label = serve_drive.check_shipped(h, "sip", require_shipped=True, allow_cache_off=False)
    assert label == "sip"


def test_no_require_shipped_skips_the_check_entirely(serve_drive, fake_health):
    port = fake_health({"ok": True, "device": "cuda:0", "prefix_cache": {"slots": 0}})
    h = serve_drive.health(port)
    label = serve_drive.check_shipped(h, "sip", require_shipped=False, allow_cache_off=False)
    assert label == "sip"
