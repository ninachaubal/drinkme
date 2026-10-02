"""http(s) image URLs and file:// paths (CPU, no external network).

llama.cpp parity: http(s) `image_url`s are downloaded by default (10s
timeout, the 20 MiB cap enforced mid-stream, a small bound on redirects),
and `file://` paths resolve under `--media-path DIR` when one is set. There
is NO address filtering (the server runs on its operator's own machine and
fetches what that operator's client asks for) — nothing here checks for a
private or loopback address, on purpose.

Every network test runs against a `ThreadingHTTPServer` bound to
127.0.0.1:0, in-process, torn down at the end of the test — no external
host is ever contacted. FETCH_TIMEOUT_S/MAX_ENCODED_BYTES are monkeypatched
down so the timeout/oversize paths cost milliseconds, not the real
10s/20MiB.
"""

from __future__ import annotations

import http.server
import io
import os
import threading
import time

import pytest
from PIL import Image

from drinkme.serving import vision


def _png_bytes(seed: int = 0) -> bytes:
    im = Image.new("RGB", (8, 8), (seed % 256, (seed * 7) % 256, 0))
    buf = io.BytesIO()
    im.save(buf, "PNG")
    return buf.getvalue()


PNG = _png_bytes(1)

# The concurrency test's shared counter: how many /slotN.png handlers are
# in flight at once, across worker threads.
_CONCURRENCY_LOCK = threading.Lock()
_CONCURRENCY = {"active": 0, "max": 0}


class _Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *a):  # keep test output quiet
        pass

    def _send(self, status: int, body: bytes = b"", headers: dict | None = None) -> None:
        self.send_response(status)
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if body:
            self.wfile.write(body)

    def do_GET(self):
        if self.path == "/ok.png":
            self._send(200, PNG, {"Content-Type": "image/png"})
        elif self.path == "/missing.png":
            self._send(404)
        elif self.path == "/slow.png":
            time.sleep(5)  # far past every shrunk test timeout below
            self._send(200, PNG, {"Content-Type": "image/png"})
        elif self.path == "/big.png":
            # No Content-Length: the client must abort on its OWN cap, not
            # wait for a declared length. ~16 MB total if never aborted.
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.end_headers()
            try:
                for _ in range(4000):
                    self.wfile.write(b"\x89PNG\r\n\x1a\n" + b"\0" * 4096)
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass  # the client aborted mid-stream -- exactly the point
        elif self.path == "/redirect.png":
            self._send(302, headers={"Location": "/ok.png"})
        elif self.path == "/not-an-image.png":
            self._send(200, b"hello, this is not an image", {"Content-Type": "text/plain"})
        elif self.path.startswith("/slot"):
            n = int(self.path[len("/slot"):].split(".")[0])
            with _CONCURRENCY_LOCK:
                _CONCURRENCY["active"] += 1
                _CONCURRENCY["max"] = max(_CONCURRENCY["max"], _CONCURRENCY["active"])
            time.sleep(0.2)
            with _CONCURRENCY_LOCK:
                _CONCURRENCY["active"] -= 1
            self._send(200, _png_bytes(n), {"Content-Type": "image/png"})
        else:
            self._send(404)


@pytest.fixture
def server():
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    srv.daemon_threads = True
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield srv
    srv.shutdown()
    t.join(timeout=5)


@pytest.fixture(autouse=True)
def _fast_and_reset(monkeypatch):
    monkeypatch.setattr(vision, "FETCH_TIMEOUT_S", 0.5)
    _CONCURRENCY["active"] = 0
    _CONCURRENCY["max"] = 0


def _base(server) -> str:
    return f"http://127.0.0.1:{server.server_address[1]}"


# --------------------------------------------------------------- fetching --


def test_a_url_is_fetched_and_decoded(server):
    enc = vision.parse_image_url(f"{_base(server)}/ok.png", where="x", fetch_urls=True)
    assert (enc.format, enc.data) == ("png", PNG)


def test_a_404_is_refused_by_name(server):
    with pytest.raises(vision.ImageError) as e:
        vision.parse_image_url(f"{_base(server)}/missing.png", where="x", fetch_urls=True)
    assert e.value.code == "image_fetch_status" and "404" in str(e.value)


def test_a_slow_handler_times_out_rather_than_hanging(server):
    t0 = time.monotonic()
    with pytest.raises(vision.ImageError) as e:
        vision.parse_image_url(f"{_base(server)}/slow.png", where="x", fetch_urls=True)
    assert e.value.code == "image_fetch_timeout"
    assert time.monotonic() - t0 < 4.0  # aborted well short of the handler's 5s sleep


def test_an_oversize_download_is_aborted_mid_stream(server, monkeypatch):
    monkeypatch.setattr(vision, "MAX_ENCODED_BYTES", 10_000)
    t0 = time.monotonic()
    with pytest.raises(vision.ImageError) as e:
        vision.parse_image_url(f"{_base(server)}/big.png", where="x", fetch_urls=True)
    assert e.value.code == "image_too_large"
    assert time.monotonic() - t0 < 2.0  # not the whole ~16 MB body: aborted early


def test_a_redirect_is_followed(server):
    enc = vision.parse_image_url(f"{_base(server)}/redirect.png", where="x", fetch_urls=True)
    assert enc.data == PNG


def test_too_many_redirects_is_refused_by_name(server, monkeypatch):
    # /redirect.png -> /ok.png is one hop; with NO hops allowed at all, it
    # cannot clear even that one.
    monkeypatch.setattr(vision, "FETCH_MAX_REDIRECTS", 0)
    with pytest.raises(vision.ImageError) as e:
        vision.parse_image_url(f"{_base(server)}/redirect.png", where="x", fetch_urls=True)
    assert e.value.code == "image_fetch_status" and "redirects" in str(e.value)


def test_a_non_image_response_is_refused_by_name(server):
    with pytest.raises(vision.ImageError) as e:
        vision.parse_image_url(f"{_base(server)}/not-an-image.png", where="x", fetch_urls=True)
    assert e.value.code == "image_format"


def test_several_images_in_one_batch_fetch_concurrently(server):
    urls = [(f"{_base(server)}/slot{i}.png", f"x[{i}]") for i in range(4)]
    t0 = time.monotonic()
    encs = vision.parse_image_urls(urls, fetch_urls=True)
    elapsed = time.monotonic() - t0
    assert len(encs) == 4 and all(e.format == "png" for e in encs)
    assert _CONCURRENCY["max"] >= 2, "the four /slotN.png fetches never overlapped"
    assert elapsed < 4 * 0.2, "took as long as four SERIAL 0.2s fetches would"


def test_a_single_url_batch_skips_the_pool(server):
    # len(pairs) <= 1 takes the plain loop in parse_image_urls — cheap to
    # confirm it still round-trips correctly.
    (enc,) = vision.parse_image_urls([(f"{_base(server)}/ok.png", "x")], fetch_urls=True)
    assert enc.data == PNG


def test_fetching_off_never_touches_the_network(server):
    # fetch_urls=False must refuse before any connection — proven by
    # pointing at a path this handler doesn't serve at all; if the code
    # tried to connect it would get a 404, not the off-switch's own code.
    with pytest.raises(vision.ImageError) as e:
        vision.parse_image_url(f"{_base(server)}/does-not-exist-either-way.png",
                               where="x", fetch_urls=False)
    assert e.value.code == "image_url_fetch_off"


# ------------------------------------------------------------- file:// ----


def test_file_url_reads_a_path_relative_to_media_path(tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "pic.png").write_bytes(PNG)
    enc = vision.parse_image_url("file://sub/pic.png", where="x", media_path=str(tmp_path))
    assert (enc.format, enc.data) == ("png", PNG)


def test_file_url_is_refused_by_name_with_no_media_path_set():
    with pytest.raises(vision.ImageError) as e:
        vision.parse_image_url("file://pic.png", where="x")
    assert e.value.code == "image_file_off"
    assert "a base64 data URL" in str(e.value)


def test_file_url_refuses_an_absolute_path(tmp_path):
    (tmp_path / "passwd").write_bytes(PNG)
    with pytest.raises(vision.ImageError) as e:
        vision.parse_image_url("file:///etc/passwd", where="x", media_path=str(tmp_path))
    assert e.value.code == "image_file_path" and "absolute" in str(e.value)


def test_file_url_refuses_a_dotdot_segment(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    (tmp_path / "secret.png").write_bytes(PNG)
    with pytest.raises(vision.ImageError) as e:
        vision.parse_image_url("file://../secret.png", where="x", media_path=str(root))
    assert e.value.code == "image_file_path"


def test_file_url_refuses_a_symlink_that_escapes_media_path(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside.png"
    outside.write_bytes(PNG)
    os.symlink(outside, root / "escape.png")
    with pytest.raises(vision.ImageError) as e:
        vision.parse_image_url("file://escape.png", where="x", media_path=str(root))
    assert e.value.code == "image_file_path" and "outside" in str(e.value)


def test_file_url_refuses_a_missing_file(tmp_path):
    with pytest.raises(vision.ImageError) as e:
        vision.parse_image_url("file://nope.png", where="x", media_path=str(tmp_path))
    assert e.value.code == "image_file_path" and "no such file" in str(e.value)


def test_file_url_checks_format_and_size_like_any_other_source(tmp_path, monkeypatch):
    (tmp_path / "text.png").write_bytes(b"not a real image, just text")
    with pytest.raises(vision.ImageError) as e:
        vision.parse_image_url("file://text.png", where="x", media_path=str(tmp_path))
    assert e.value.code == "image_format"
    monkeypatch.setattr(vision, "MAX_ENCODED_BYTES", 4)
    (tmp_path / "big.png").write_bytes(PNG)
    with pytest.raises(vision.ImageError) as e:
        vision.parse_image_url("file://big.png", where="x", media_path=str(tmp_path))
    assert e.value.code == "image_too_large"
