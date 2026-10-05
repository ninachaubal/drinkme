"""Image input: from the bytes a request carries to the arrays a vision
tower reads. Torch-free (numpy and Pillow), so the dialect handlers can run
it outside the generation lock and the count routes can use it without a
model. The engine's side, the tower itself (`Tower`, `bounded_attention`),
sits at the end and imports torch inside its functions.

Three stages, and the seam between them:

1. THE WIRE -> EncodedImage. `parse_image_url` takes an OpenAI-style
   `image_url` string (also Anthropic's `source.type: "url"` and Responses'
   `image_url`): a `data:` URL decodes inline; an http(s) URL is
   DOWNLOADED — llama.cpp parity: a 10 s timeout, the download aborted
   mid-stream the moment it passes the 20 MiB cap (never buffered then
   checked), and a small bound on redirects. There is NO address
   filtering — no SSRF guard, no private-range check: the server runs on
   its operator's own machine, fetching what its operator's own client
   asks for, as llama.cpp's does.
   An operator who exposes the server past their own machine turns
   fetching off (`--no-image-urls` / DRINKME_IMAGE_URLS=0); a `file://`
   path resolves under `--media-path DIR` when one is set (llama
   semantics: disabled unless set, the path relative to DIR, an absolute
   path / `..` / a symlink escape all refused by name). `parse_image_urls`
   fetches every `image_url` in one turn CONCURRENTLY (a thread pool over
   the network read only — decoding after it is CPU-bound and stays
   sequential). `parse_base64` takes an Anthropic-style base64 block. Every
   refusal names what IS accepted, never just what is off (`sources_
   message`). The format comes from the magic bytes (PNG, JPEG, WebP,
   GIF), never from the declared media type. The declared type only has to
   name one of those four.
2. DECODE, shared by every architecture. Pillow opens the header and the
   bomb guard reads the pixel count there, before anything is allocated.
   Then comes `load()` (a GIF or animated WebP/PNG yields its first frame),
   `ImageOps.exif_transpose`, and `convert("RGB")`. This is transformers'
   `image_utils.load_image`, so an RGBA image's alpha is dropped, not
   composited.
3. THE ARCHITECTURE'S PREPROCESSOR, looked up in a registry by the
   checkpoint's `model_type`. It turns the RGB image into `pixel_values` and
   a `grid_thw`, and it can also plan the same image from its header alone
   (`Vision.count`: the token count for count_tokens and /tokenize,
   without decoding pixels). Qwen3.5's (Qwen3.8-27B, MiMo-9B), gemma-4's
   (gemma-4-31B-it) and Muse-Glimmer's (Muse-Glimmer-30B) are implemented,
   each a class that satisfies `Preprocessor` and a factory registered
   under its model type (`@register(...)`). No caller changes: engines
   hold a `Vision`, and the dialects call its `count` and `prepare`.

Qwen3.5's preprocessor reproduces transformers' `Qwen2VLImageProcessorPil`
bit for bit (pinned by tests/test_serving_vision.py over a fixture set):
smart_resize to multiples of patch x merge, a Pillow resize on uint8,
rescale and normalize with the reference's exact float64 -> float32
arithmetic (folded into a 256-entry table per channel, which gives the same
bits), and patchify with the single frame duplicated along T. gemma-4's
reproduces `Gemma4ImageProcessorPil` bit for bit the same way
(tests/test_serving_gemma_vision.py): an aspect-preserving resize to
multiples of 48 px under a patch budget, rescale to [0, 1], patchify, and
zero rows up to the budget's patch count. Muse-Glimmer's reproduces
`MuseGlimmerImageProcessor`, whose only backend is torchvision's
(tests/test_serving_glimmer_vision.py, against a torchvision run): the
grid of 28-px cells nearest the aspect ratio under a token cap, torch's
own LANCZOS resize (Pillow's differs by a level on some pixels), the fused
normalize, and patchify, all numpy but the resize.

The pixel budget: the server cap (DRINKME_IMAGE_MAX_PIXELS or `drinkme
serve --image-max-pixels`, default 2560x1440) bounds every image. A
request's `detail: "low"` asks for 512x512 instead, and `high`/`auto` take
the cap; nothing a request sends can raise it. The architecture applies the
budget inside its own resize rule, so the token count is always what the
model's own algorithm gives for that budget (Qwen: min(checkpoint max,
budget) as smart_resize's max_pixels; gemma-4: the budget lowers
max_soft_tokens to what it covers; Muse-Glimmer: the budget lowers
max_image_tokens to the 28 x 28 cells it covers).

Every refusal is an ImageError: a ValueError whose message starts with the
part's location (`messages[1].content[0]`) and carries a stable `code`, so
each dialect renders it as its own invalid_request_error 400.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import http.client
import itertools
import json
import math
import os
import re
import sys
import time
import urllib.parse
import warnings
from dataclasses import dataclass
from io import BytesIO
from typing import Callable, Protocol

import numpy as np
from PIL import ExifTags, Image, ImageOps

# ------------------------------------------------------------- the limits --
# drinkme's own choices. The request body's own cap (DRINKME_MAX_BODY_BYTES,
# 128 MiB) is enforced before any of this runs.

MAX_ENCODED_BYTES = 20 * 1024 * 1024  # one image, after base64 decoding OR fetched/read raw
MAX_SOURCE_PIXELS = 50_000_000        # the decompression-bomb guard, from the header
MAX_IMAGES = 32                       # per request
DEFAULT_MAX_PIXELS = 2560 * 1440      # the server cap: 3,600 Qwen3.5 tokens at most
LOW_DETAIL_PIXELS = 512 * 512         # `detail: "low"`: 256 Qwen3.5 tokens at most
FORMATS = ("png", "jpeg", "webp", "gif")
DETAILS = ("auto", "low", "high")

# http(s) fetch (llama.cpp parity): module globals, read at CALL time
# (never bound as a default parameter value) so a test can monkeypatch them
# down to exercise the timeout/oversize paths in milliseconds instead of
# the real 10s/20MiB.
FETCH_TIMEOUT_S = 10.0        # wall clock, covers every redirect hop and the whole body read
FETCH_MAX_REDIRECTS = 5       # a small bound, not curl's default chain
FETCH_MAX_WORKERS = 8         # parse_image_urls: at most this many concurrent downloads

IMAGE_URLS_ENV = "DRINKME_IMAGE_URLS"  # "0" turns off http(s) fetching; default on
MEDIA_PATH_ENV = "DRINKME_MEDIA_PATH"  # a directory; unset/empty = file:// stays refused

# Pillow reads the pixel count in Image.open, before it allocates anything.
# At MAX_IMAGE_PIXELS it only warns (it raises at twice that), so the warning
# is made an error for this one category. _open also checks the count
# itself, which still holds if something resets the warning filters.
Image.MAX_IMAGE_PIXELS = MAX_SOURCE_PIXELS
warnings.filterwarnings("error", category=Image.DecompressionBombWarning)

_PIL_FORMAT = {"png": "PNG", "jpeg": "JPEG", "webp": "WEBP", "gif": "GIF"}
_MEDIA_TYPES = {"image/png": "png", "image/jpeg": "jpeg", "image/jpg": "jpeg",
                "image/webp": "webp", "image/gif": "gif"}


class ImageError(ValueError):
    """An image refused, phrased for a 400 (invalid_request_error). The
    message starts with `where`, the request part it came from. `code` is a
    stable name for tests and logs:

      image_url_fetch_off      an http(s) URL, with fetching turned off
                               (--no-image-urls / DRINKME_IMAGE_URLS=0)
      image_file_off           a file:// URL, with no --media-path set
      image_file_path          a file:// path that is absolute, contains
                               `..`, resolves outside --media-path
                               (symlinks followed), or does not exist
      image_files_api          Anthropic `source.type: "file"` or Responses
                               `file_id`: a provider Files-API id, not a
                               path — always refused, both dialects
      image_fetch_unreachable  the http(s) fetch could not connect (DNS,
                               refused, reset, ...)
      image_fetch_timeout      the http(s) fetch exceeded FETCH_TIMEOUT_S
      image_fetch_status       a non-2xx/redirect HTTP status, a redirect
                               with no Location, or too many redirects
      image_url                not http(s), file:// or a data URL
      image_data_url           a malformed data URL, or one that is not base64
      image_media_type         a declared type other than PNG/JPEG/WebP/GIF
      image_base64             the payload is not valid base64
      image_too_large          over MAX_ENCODED_BYTES (decoded, fetched, or read)
      image_format             the bytes are none of the four formats
      image_too_many_pixels    over MAX_SOURCE_PIXELS (the bomb guard)
      image_aspect_ratio       long side over 200x the short side (smart_resize)
      image_decode             the decoder failed (truncated, corrupt)
      image_detail             `detail` not one of DETAILS
      too_many_images          over MAX_IMAGES in one request
    """

    def __init__(self, where: str, code: str, message: str):
        self.where, self.code = where, code
        super().__init__(f"{where}: {message}")


# ------------------------------------------------------ 1. the wire -------

@dataclass(frozen=True)
class EncodedImage:
    """An image's bytes as the request carried them, format already
    sniffed from the magic bytes. Nothing is decoded yet."""

    data: bytes
    format: str  # one of FORMATS


def sniff_format(data: bytes) -> str | None:
    """The format the magic bytes name, or None."""
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if data.startswith(b"\xff\xd8\xff"):
        return "jpeg"
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return "gif"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "webp"
    return None


def _looks_like(data: bytes) -> str | None:
    """A name for a common format this server refuses, for the message."""
    head = data[:1024].lstrip()
    if head.startswith(b"<?xml") or b"<svg" in head:
        return "SVG"
    if data[4:8] == b"ftyp":
        return "HEIC/AVIF"
    if data[:4] in (b"II*\x00", b"MM\x00*"):
        return "TIFF"
    if data[:2] == b"BM":
        return "BMP"
    return None


def _check_media_type(media_type, where: str) -> None:
    if not isinstance(media_type, str) or media_type.strip().lower() not in _MEDIA_TYPES:
        raise ImageError(where, "image_media_type",
                         f"media type {media_type!r} is not supported; send PNG, JPEG, "
                         "WebP or GIF (image/png, image/jpeg, image/webp, image/gif)")


def parse_base64(data, media_type: str | None = None, *, where: str) -> EncodedImage:
    """A base64 payload (Anthropic's `source.data`, or a data URL's body)
    -> EncodedImage. `media_type`, when given, must name one of the four
    formats. The bytes decide which one it is. ASCII whitespace (MIME
    line breaks) is ignored and missing `=` padding is restored. Anything
    else that is not base64 is refused."""
    if media_type is not None:
        _check_media_type(media_type, where)
    return _decoded_bytes(base64_bytes(data, where=where), where=where, noun="image")


def base64_bytes(data, *, where: str, kind: "MediaKind | None" = None) -> bytes:
    """A base64 string -> its bytes, for an image (the default) or a video
    (serving/video.py): ASCII whitespace ignored, missing `=` padding
    restored, and a payload over the kind's byte cap refused from its
    length, before it is decoded."""
    kind = kind or IMAGE
    err, noun, cap = kind.error, kind.noun, kind.max_bytes()
    if not isinstance(data, str):
        raise err(where, f"{noun}_base64", f"{noun} data must be a base64 string")
    s = "".join(data.split())
    if len(s) // 4 * 3 > cap + 3:  # refused before decoding it
        raise err(where, f"{noun}_too_large",
                  f"{noun} is over {cap // (1024 * 1024)} MiB once decoded")
    s += "=" * (-len(s) % 4)
    try:
        return base64.b64decode(s, validate=True)
    except (binascii.Error, ValueError):
        raise err(where, f"{noun}_base64", f"{noun} data is not valid base64") from None


def sources_message(*, fetch_urls: bool, media_path: str | None) -> str:
    """A noun phrase naming what IS currently accepted, given this server's
    (fetch_urls, media_path) — a refusal names the alternative, never just
    the thing that's off. Callers prefix it (e.g. "send " + this)."""
    opts = ["a base64 data URL"]
    if fetch_urls:
        opts.append("an http(s) URL")
    if media_path:
        opts.append("a file:// path under the media directory")
    if len(opts) == 1:
        return opts[0]
    if len(opts) == 2:
        return f"{opts[0]} or {opts[1]}"
    return f"{', '.join(opts[:-1])}, or {opts[-1]}"


def _decoded_bytes(data: bytes, *, where: str, noun: str) -> EncodedImage:
    """The bytes a fetch or a file read produced -> EncodedImage: the same
    magic-byte sniff and size cap `parse_base64` applies to a decoded
    payload (nothing about how the bytes arrived changes what happens to
    them next)."""
    if len(data) > MAX_ENCODED_BYTES:
        raise ImageError(where, "image_too_large",
                         f"{noun} is {len(data)} bytes; the limit is {MAX_ENCODED_BYTES} "
                         f"({MAX_ENCODED_BYTES // (1024 * 1024)} MiB)")
    fmt = sniff_format(data)
    if fmt is None:
        seen = _looks_like(data)
        raise ImageError(where, "image_format",
                         (f"{seen} images are not supported" if seen else
                          f"{noun} is not PNG, JPEG, WebP or GIF")
                         + "; send PNG, JPEG, WebP or GIF")
    return EncodedImage(data, fmt)


@dataclass(frozen=True)
class MediaKind:
    """What differs between fetching or reading an image and a video
    (serving/video.py) through the one set of source rules below: the noun
    the refusal messages and codes use (`image_fetch_timeout`,
    `video_fetch_timeout`, ...), the error class, the byte cap and the
    fetch deadline (callables, so the module globals behind them are read
    at call time and a test can monkeypatch them), the Accept header, and
    the data URL a refusal shows as the example."""

    noun: str
    error: type
    max_bytes: Callable[[], int]
    timeout_s: Callable[[], float]
    accept: str
    example: str


IMAGE = MediaKind("image", ImageError, lambda: MAX_ENCODED_BYTES, lambda: FETCH_TIMEOUT_S,
                  "image/*", "data:image/png;base64,...")


def _fetch_url_bytes(url: str, *, where: str, kind: MediaKind = IMAGE) -> bytes:
    """GET url over http(s), stdlib only (http.client — no new dependency):
    one wall-clock deadline (FETCH_TIMEOUT_S for an image) covers every
    redirect hop and the whole body read; at most FETCH_MAX_REDIRECTS hops;
    the read is ABORTED the instant it passes the kind's byte cap
    (MAX_ENCODED_BYTES for an image), never buffered then checked. Both
    are read from the module globals at call time (not bound as default
    parameters) so a test can monkeypatch them down. NO address filtering:
    the server fetches whatever the operator's own client asked it to,
    from the operator's own network position (module docstring)."""
    err, noun = kind.error, kind.noun
    timeout_s, cap = kind.timeout_s(), kind.max_bytes()
    deadline = time.monotonic() + timeout_s
    for _ in range(FETCH_MAX_REDIRECTS + 1):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise err(where, f"{noun}_fetch_timeout",
                      f"fetching {url!r} timed out after {timeout_s:g}s")
        parsed = urllib.parse.urlsplit(url)
        if parsed.scheme not in ("http", "https"):
            raise err(where, f"{noun}_url",
                      f"{url!r}: only http(s), file:// and data: URLs are accepted")
        if not parsed.hostname:
            raise err(where, f"{noun}_url", f"{url!r}: no host")
        cls = http.client.HTTPSConnection if parsed.scheme == "https" else http.client.HTTPConnection
        conn = cls(parsed.hostname, parsed.port, timeout=remaining)
        target = urllib.parse.urlunsplit(("", "", parsed.path or "/", parsed.query, ""))
        try:
            conn.request("GET", target, headers={"User-Agent": "drinkme/1.0",
                                                  "Accept": kind.accept})
            resp = conn.getresponse()
        except TimeoutError:
            conn.close()
            raise err(where, f"{noun}_fetch_timeout",
                      f"fetching {url!r} timed out after {timeout_s:g}s") from None
        except OSError as e:
            conn.close()
            raise err(where, f"{noun}_fetch_unreachable",
                      f"could not fetch {url!r} ({e})") from None
        if resp.status in (301, 302, 303, 307, 308):
            location = resp.getheader("Location")
            resp.read()
            conn.close()
            if not location:
                raise err(where, f"{noun}_fetch_status",
                          f"{url!r} redirected ({resp.status}) with no Location header")
            url = urllib.parse.urljoin(url, location)
            continue
        if not (200 <= resp.status < 300):
            resp.read()
            conn.close()
            raise err(where, f"{noun}_fetch_status", f"fetching {url!r} got HTTP {resp.status}")
        data = bytearray()
        try:
            while True:
                if time.monotonic() > deadline:
                    raise err(where, f"{noun}_fetch_timeout",
                              f"fetching {url!r} timed out after {timeout_s:g}s")
                chunk = resp.read(65536)
                if not chunk:
                    break
                data += chunk
                if len(data) > cap:
                    raise err(where, f"{noun}_too_large",
                              f"fetching {url!r}: over {cap // (1024 * 1024)} MiB")
        except TimeoutError:
            raise err(where, f"{noun}_fetch_timeout",
                      f"fetching {url!r} timed out after {timeout_s:g}s") from None
        finally:
            conn.close()
        return bytes(data)
    raise err(where, f"{noun}_fetch_status", f"{url!r}: too many redirects")


def _read_media_bytes(url: str, media_path: str, *, where: str,
                      kind: MediaKind = IMAGE) -> tuple[str, bytes]:
    """`file://` + `media_path` (llama.cpp's --media-path semantics) ->
    (the relative path, at most the kind's byte cap plus one byte of the
    file): the path after `file://` is relative to `media_path`, never
    absolute, never containing a `..` segment, and its resolved real path
    (symlinks followed) must stay inside `media_path` — every one of those
    is a named 400, not a silent clamp."""
    err, code = kind.error, f"{kind.noun}_file_path"
    raw = urllib.parse.unquote(url[len("file://"):])
    if os.path.isabs(raw):
        raise err(where, code, "file:// paths must be relative to the configured media "
                  "directory; an absolute path is refused")
    segments = raw.split("/")
    if not raw or any(seg in ("", "..") for seg in segments):
        raise err(where, code, f"{raw!r} is not a valid relative path under the media "
                  "directory ('..' and empty segments are refused)")
    root = os.path.realpath(media_path)
    candidate = os.path.realpath(os.path.join(root, raw))
    if candidate != root and not candidate.startswith(root + os.sep):
        raise err(where, code, f"{raw!r} resolves outside the media directory")
    if not os.path.isfile(candidate):
        raise err(where, code, f"no such file: {raw!r}")
    try:
        with open(candidate, "rb") as f:
            data = f.read(kind.max_bytes() + 1)
    except OSError as e:
        raise err(where, code, f"could not read {raw!r} ({e})") from None
    return raw, data


def _read_media_file(url: str, media_path: str, *, where: str) -> EncodedImage:
    """An image at a `file://` path under `media_path` (_read_media_bytes)."""
    raw, data = _read_media_bytes(url, media_path, where=where)
    return _decoded_bytes(data, where=where, noun=f"{raw!r}")


def parse_media_url(url, *, where: str, fetch_urls: bool, media_path: str | None,
                    kind: MediaKind, from_bytes: Callable, from_base64: Callable):
    """The source rules every media URL follows, image or video: a data:
    URL decodes inline (`from_base64(body, media_type, where=)`); an
    http(s) URL is fetched (`_fetch_url_bytes`) unless `fetch_urls` is
    False; a file:// path resolves under `media_path`
    (`_read_media_bytes`) when one is given. Fetched and read bytes go
    through `from_bytes(data, where=, noun=)`, the kind's own size cap and
    magic-byte sniff. Every refusal names what IS accepted
    (`sources_message`)."""
    err, noun = kind.error, kind.noun
    if not isinstance(url, str):
        raise err(where, f"{noun}_url", f"{noun} URL must be a string")
    head = url[:16].lower()
    if head.startswith(("http://", "https://")):
        if not fetch_urls:
            raise err(where, f"{noun}_url_fetch_off",
                      f"{noun} URLs are not fetched on this server; send "
                      + sources_message(fetch_urls=False, media_path=media_path))
        data = _fetch_url_bytes(url, where=where, kind=kind)
        return from_bytes(data, where=where, noun=f"the fetched {noun} ({url!r})")
    if head.startswith("file://"):
        if not media_path:
            raise err(where, f"{noun}_file_off",
                      f"file:// {noun} paths are not accepted on this server (no "
                      "--media-path is configured); send "
                      + sources_message(fetch_urls=fetch_urls, media_path=None))
        raw, data = _read_media_bytes(url, media_path, where=where, kind=kind)
        return from_bytes(data, where=where, noun=f"{raw!r}")
    if not head.startswith("data:"):
        raise err(where, f"{noun}_url",
                  f"{noun} URL must be http(s), file:// or a data URL; send "
                  + sources_message(fetch_urls=fetch_urls, media_path=media_path))
    comma = url.find(",", 0, 256)
    if comma < 0:
        raise err(where, f"{noun}_data_url", "malformed data URL (no ',' after the media type)")
    params = [p.strip().lower() for p in url[5:comma].split(";")]
    if "base64" not in params[1:]:
        raise err(where, f"{noun}_data_url",
                  f"data URL must be base64-encoded ({kind.example})")
    return from_base64(url[comma + 1:], params[0], where=where)


def parse_image_url(url, *, where: str, fetch_urls: bool = True,
                    media_path: str | None = None) -> EncodedImage:
    """An `image_url` string (OpenAI Chat's `image_url.url`, Responses'
    `input_image.image_url`, and Anthropic's `source.type: "url"` value)
    -> EncodedImage, by parse_media_url's source rules: a data: URL
    decodes inline (`parse_base64`); an http(s) URL is fetched unless
    `fetch_urls` is False; a file:// path resolves under `media_path` when
    one is given."""
    return parse_media_url(url, where=where, fetch_urls=fetch_urls, media_path=media_path,
                           kind=IMAGE, from_bytes=_decoded_bytes, from_base64=parse_base64)


def parse_many(pairs: list[tuple[str, str]], parse_one: Callable) -> list:
    """`parse_one(url, where=)` over several (url, where) pairs at once:
    every http(s) URL's DOWNLOAD runs concurrently in a thread pool
    (network I/O only — the GIL releases during a socket read; decoding
    stays out of the pool since it is CPU-bound), so several images or
    videos in one request or one turn fetch at once rather than one after
    another. Returns the results in the SAME order as `pairs`; the first
    one (in that order) to raise is what propagates, matching
    one-at-a-time semantics for the client. A single pair skips the pool
    entirely."""
    if len(pairs) <= 1:
        return [parse_one(u, where=w) for u, w in pairs]
    from concurrent.futures import ThreadPoolExecutor

    ex = ThreadPoolExecutor(max_workers=min(FETCH_MAX_WORKERS, len(pairs)))
    try:
        futures = [ex.submit(parse_one, u, where=w) for u, w in pairs]
        return [f.result() for f in futures]
    finally:
        ex.shutdown(wait=False)


def parse_image_urls(pairs: list[tuple[str, str]], *, fetch_urls: bool = True,
                     media_path: str | None = None) -> list[EncodedImage]:
    """`parse_image_url` over several (url, where) pairs at once, the
    downloads concurrent (`parse_many`). Returns EncodedImages in the SAME
    order as `pairs`."""
    def one(u, *, where):
        return parse_image_url(u, where=where, fetch_urls=fetch_urls, media_path=media_path)

    return parse_many(pairs, one)


def check_image_count(n: int, *, where: str = "request") -> None:
    """Refuse a request carrying more than MAX_IMAGES images."""
    if n > MAX_IMAGES:
        raise ImageError(where, "too_many_images",
                         f"{n} images in one request; the limit is {MAX_IMAGES}")


# ------------------------------------------------------ 2. decode ---------

def _open(image: EncodedImage, where: str) -> Image.Image:
    """Pillow on the header only. The pixel count is checked here, before
    load() allocates anything."""
    try:
        im = Image.open(BytesIO(image.data), formats=[_PIL_FORMAT[image.format]])
    except (Image.DecompressionBombError, Image.DecompressionBombWarning):
        raise ImageError(where, "image_too_many_pixels",
                         f"image has more than {MAX_SOURCE_PIXELS:,} pixels") from None
    except Exception as e:  # noqa: BLE001 — untrusted bytes: any decoder failure is a 400
        raise ImageError(where, "image_decode",
                         f"could not read the {image.format.upper()} image "
                         f"({type(e).__name__}: {e})") from None
    w, h = im.size
    if w * h > MAX_SOURCE_PIXELS:
        raise ImageError(where, "image_too_many_pixels",
                         f"image is {w}x{h} = {w * h:,} pixels; the limit is {MAX_SOURCE_PIXELS:,}")
    if w < 1 or h < 1:
        raise ImageError(where, "image_decode", f"image is {w}x{h}")
    return im


# Chunks that can carry the EXIF orientation (eXIf; "Raw profile type exif"
# and XMP live in text chunks). Pillow reads the ones after IDAT only in load().
_PNG_METADATA = (b"eXIf", b"iTXt", b"tEXt", b"zTXt")


def _png_metadata_after_pixels(data: bytes) -> bool:
    """Does a PNG carry a metadata chunk after its first IDAT? Walks the
    chunk headers only, never inflating anything."""
    pos, seen_idat = 8, False
    while pos + 8 <= len(data):
        length = int.from_bytes(data[pos:pos + 4], "big")
        kind = data[pos + 4:pos + 8]
        if kind == b"IDAT":
            seen_idat = True
        elif kind == b"IEND":
            return False
        elif seen_idat and kind in _PNG_METADATA:
            return True
        pos += 12 + length
    return False


def _oriented_size(im: Image.Image, image: EncodedImage) -> tuple[int, int]:
    """(height, width) after exif_transpose, from the header. This is the
    same orientation lookup exif_transpose makes: EXIF, then the XMP
    fallback, in Image.getexif. The base class's getexif is called directly
    because PngImageFile's override load()s the pixels whenever the header
    had no eXIf. That load is needed only when metadata follows IDAT."""
    if image.format == "png" and _png_metadata_after_pixels(image.data):
        im.load()
    orientation = Image.Image.getexif(im).get(ExifTags.Base.Orientation, 1)
    w, h = im.size
    return (w, h) if orientation in (5, 6, 7, 8) else (h, w)


def decode(image: EncodedImage, *, where: str) -> Image.Image:
    """EncodedImage -> an RGB PIL image, oriented. This is transformers'
    load_image (exif_transpose, then convert("RGB")) with the bomb guard in
    front. A GIF or animated WebP/PNG gives its first frame."""
    im = _open(image, where)
    try:
        im.load()
        im = ImageOps.exif_transpose(im)
        return im if im.mode == "RGB" else im.convert("RGB")
    except Exception as e:  # noqa: BLE001 — untrusted bytes: any decoder failure is a 400
        raise ImageError(where, "image_decode",
                         f"could not decode the {image.format.upper()} image "
                         f"({type(e).__name__}: {e})") from None


# ------------------------------------------------------ the results -------

@dataclass(frozen=True)
class ImagePlan:
    """What an image costs, known from its header: what `Vision.count`
    returns, and the shape half of a PreparedImage."""

    architecture: str                  # the preprocessor's registry key (config.json model_type)
    source_format: str                 # one of FORMATS, from the magic bytes
    source_size: tuple[int, int]       # (height, width), EXIF orientation applied
    size: tuple[int, int]              # (height, width) the image is resized to
    grid_thw: tuple[int, int, int]     # (t, h, w) in ViT patches
    tokens: int                        # placeholder tokens the prompt carries for it


@dataclass(frozen=True, eq=False)
class PreparedImage:
    """One image, ready for the vision tower: what `Vision.prepare` returns
    and what GenerationRequest.images holds.

    `pixel_values` is float32, C-contiguous and read-only, with shape
    [t*h*w, C*T*P*P] in the architecture's own layout. For Qwen3.5 that is
    [gh*gw, 3*2*16*16 = 1536], rows merge-block-major, exactly what
    `Qwen3_5Model.get_image_features(pixel_values, image_grid_thw)` takes
    once the rows of every image are concatenated. `digest` is sha256 hex
    over the pixel_values bytes, then the three grid_thw values as
    little-endian int64. Equal digests mean the tower sees equal input, so
    two PreparedImages compare equal by (architecture, digest)."""

    architecture: str
    source_format: str
    source_size: tuple[int, int]
    size: tuple[int, int]
    grid_thw: tuple[int, int, int]
    tokens: int
    pixel_values: np.ndarray
    digest: str

    @property
    def plan(self) -> ImagePlan:
        return ImagePlan(self.architecture, self.source_format, self.source_size, self.size,
                         self.grid_thw, self.tokens)

    @property
    def released(self) -> bool:
        return self.pixel_values is None

    def release(self) -> None:
        """Drop pixel_values. The engine calls this once the vision tower
        has read them, or as soon as it knows the tower never will (the
        image sits inside a reused prefix). One image is 88 MB at the
        default cap and a request may carry MAX_IMAGES of them, so the
        host copy lives until the prefill reaches the image, not for the
        whole generation. The plan, digest and prefix_key stay; a second
        generation that needs the tower to read this image again is
        refused by Tower.features (prepare the image again)."""
        object.__setattr__(self, "pixel_values", None)

    @property
    def prefix_key(self) -> int:
        """The id that stands for this image's placeholder run in a prefix
        cache key. The pad token id is the same for every image, so two
        same-size images render to the same ids and would otherwise share
        KV, a wrong answer served with a 200.
        The value is negative, so it can never equal a vocabulary id, and
        it fits int64 (slotstore packs keys as array("q")). It is the
        digest's first 63 bits. The image token id does not need to be
        mixed in: a key only ever meets keys from the same pack."""
        return -1 - (int(self.digest[:16], 16) >> 1)

    def __eq__(self, other) -> bool:
        if not isinstance(other, PreparedImage):
            return NotImplemented
        return (self.architecture, self.digest) == (other.architecture, other.digest)

    def __hash__(self) -> int:
        return hash((self.architecture, self.digest))

    def __repr__(self) -> str:
        return (f"PreparedImage({self.architecture}, {self.source_format} "
                f"{self.source_size[1]}x{self.source_size[0]} -> {self.size[1]}x{self.size[0]}, "
                f"grid={self.grid_thw}, tokens={self.tokens}, digest={self.digest[:12]}…)")


def image_digest(pixel_values: np.ndarray, grid_thw) -> str:
    """sha256 over the pixel_values bytes, then grid_thw as three
    little-endian int64."""
    h = hashlib.sha256()
    h.update(memoryview(np.ascontiguousarray(pixel_values, dtype="<f4")).cast("B"))
    h.update(np.asarray(grid_thw, dtype="<i8").tobytes())
    return h.hexdigest()


# ------------------------------------------ 3. the per-architecture seam --

class Preprocessor(Protocol):
    """One architecture's image preprocessing. The registry maps a
    checkpoint's `model_type` to a factory that builds one of these from
    the checkpoint's processor config."""

    architecture: str
    # Literal strings the tokenizer reads as a modality's special tokens:
    # EVERY image, video and audio marker it defines, whether or not this
    # server feeds that modality. The tokenizer maps them to the special ids
    # even inside user text, so a request's text that carries one is refused
    # by the dialects (capability.check_injection). Pinned against each
    # architecture's real tokenizer where it is cached
    # (tests/test_serving_glimmer_vision.py).
    reserved_text: tuple[str, ...]
    # Tokens the prompt carries for one image beyond its `tokens`: the
    # markers the placeholder's expansion adds around the image's run.
    # Qwen3.5 0 (its template renders the markers itself), gemma-4 2
    # (`<|image>` and `<image|>`), Muse-Glimmer 2 (`<|image_start|>` and
    # `<|image_end|>`), both added by transformers' processor.
    wrap: int

    def plan(self, height: int, width: int, max_pixels: int, *,
             where: str) -> tuple[tuple[int, int], tuple[int, int, int], int]:
        """(resized (h, w), grid_thw, tokens) for a source of height x
        width under a `max_pixels` budget. Header facts only."""
        ...

    def pixel_values(self, rgb: Image.Image, size: tuple[int, int],
                     max_pixels: int) -> np.ndarray:
        """The oriented RGB image resized to `size` (h, w) -> pixel_values,
        float32, C-contiguous. `max_pixels` is the budget `plan` was given
        (gemma-4 pads its patches to the budget's patch count)."""
        ...


# The registry: config.json `model_type` -> factory(processor config dict,
# or None when the checkpoint carries none) -> Preprocessor, or None when
# that checkpoint cannot take images.
_REGISTRY: dict[str, Callable[[dict | None], Preprocessor | None]] = {}


def register(*model_types: str):
    """Decorator: the factory for these model types."""
    def deco(factory):
        for t in model_types:
            _REGISTRY[t] = factory
        return factory
    return deco


def architectures() -> tuple[str, ...]:
    """The model types with a registered preprocessor."""
    return tuple(sorted(_REGISTRY))


def preprocessor_for(model_type: str, processor_config: dict | None) -> Preprocessor | None:
    """The registered preprocessor for `model_type`, built from its
    processor config. None if no preprocessor is registered, or the
    factory declines (for Qwen3.5, a checkpoint without a processor
    config). A config the factory cannot honor raises ValueError: that is
    a load-time error, not a request's."""
    factory = _REGISTRY.get(model_type)
    return None if factory is None else factory(processor_config)


def read_processor_config(checkpoint_dir: str) -> dict | None:
    """The image-processor dict transformers' from_pretrained would load
    from `checkpoint_dir`: processor_config.json's nested `image_processor`
    when it has one, else preprocessor_config.json, else None.
    Both files are identity files (codec/identity.IDENTITY_FILES), and a
    pack embeds both."""
    nested = os.path.join(checkpoint_dir, "processor_config.json")
    if os.path.isfile(nested):
        with open(nested) as f:
            cfg = json.load(f)
        if isinstance(cfg.get("image_processor"), dict):
            return cfg["image_processor"]
    flat = os.path.join(checkpoint_dir, "preprocessor_config.json")
    if os.path.isfile(flat):
        with open(flat) as f:
            return json.load(f)
    return None


# ------------------------------------------------------- the pixel cap ----

def parse_pixels(text: str) -> int:
    """'3686400' or '2560x1440' -> a positive pixel count. ValueError
    otherwise."""
    t = str(text).strip().lower()
    m = re.fullmatch(r"(\d+)\s*[x×]\s*(\d+)", t)
    n = int(m[1]) * int(m[2]) if m else int(t)
    if n < 1:
        raise ValueError(f"{text!r} is not a positive pixel count")
    return n


def max_pixels_from_env(explicit: int | None = None) -> int:
    """The server's pixel cap. `explicit` (`drinkme serve
    --image-max-pixels`) wins, then DRINKME_IMAGE_MAX_PIXELS, then
    DEFAULT_MAX_PIXELS. A garbage environment value warns and falls back to
    the default rather than taking the server down (serve.py's posture for
    every such variable)."""
    if explicit is not None:
        if explicit < 1:
            raise ValueError(f"image max pixels {explicit} is not a positive pixel count")
        return int(explicit)
    raw = os.environ.get("DRINKME_IMAGE_MAX_PIXELS", "").strip()
    if not raw:
        return DEFAULT_MAX_PIXELS
    try:
        return parse_pixels(raw)
    except ValueError:
        print(f"[drinkme] DRINKME_IMAGE_MAX_PIXELS={raw!r} is not a pixel count "
              f"(N or WxH) — using the default {DEFAULT_MAX_PIXELS}",
              file=sys.stderr, flush=True)
        return DEFAULT_MAX_PIXELS


def fetch_urls_from_env(explicit: bool | None = None) -> bool:
    """The server's http(s) fetch switch: `explicit` (--no-image-urls,
    already inverted to fetch_urls=False by the caller) wins, then
    DRINKME_IMAGE_URLS ("0" turns it off), else on — the default, llama.cpp
    parity."""
    if explicit is not None:
        return explicit
    return os.environ.get(IMAGE_URLS_ENV, "").strip() != "0"


def media_path_from_env(explicit: str | None = None) -> str | None:
    """The server's file:// media root: `explicit` (--media-path, already
    validated as an existing directory by the CLI's argparse type=) wins,
    then DRINKME_MEDIA_PATH — a bad environment value warns and leaves
    file:// disabled rather than taking the server down
    (max_pixels_from_env's posture); unset/empty leaves it None, meaning
    file:// stays refused (llama.cpp semantics: disabled unless a media
    path is set)."""
    if explicit is not None:
        return os.path.realpath(explicit)
    raw = os.environ.get(MEDIA_PATH_ENV, "").strip()
    if not raw:
        return None
    resolved = os.path.realpath(raw)
    if not os.path.isdir(resolved):
        print(f"[drinkme] DRINKME_MEDIA_PATH={raw!r} is not a directory — "
              "file:// image paths stay disabled", file=sys.stderr, flush=True)
        return None
    return resolved


# ------------------------------------------------------- the engine side --

@dataclass(frozen=True)
class Vision:
    """An engine's image input: one architecture's preprocessor, the
    server's pixel cap, and its http(s)/file:// posture. The dialects call
    `count` and `prepare`, and read `fetch_urls`/`media_path` to pass to
    `parse_image_url(s)`. The engine reads `architecture` and
    `reserved_text`.

    `video` is the same engine's video input (serving/video.py), which
    rides on the image input because the same tower reads it and the same
    source posture fetches it: a video.VideoInput, or None with
    `video_reason` saying why (no video processor for the architecture or
    in the checkpoint, PyAV not installed)."""

    preprocessor: Preprocessor
    max_pixels: int = DEFAULT_MAX_PIXELS
    fetch_urls: bool = True
    media_path: str | None = None
    video: "video.VideoInput | None" = None
    video_reason: str | None = "this server has no video input for the model"

    @property
    def architecture(self) -> str:
        return self.preprocessor.architecture

    @property
    def reserved_text(self) -> tuple[str, ...]:
        return self.preprocessor.reserved_text

    def prompt_tokens(self, image) -> int:
        """The tokens one image (an ImagePlan or PreparedImage) takes in the
        prompt: its `tokens` and the markers around them
        (Preprocessor.wrap). The chat template renders it as ONE
        placeholder, so a rendered prompt's count grows by this less one
        per image (count_tokens, /tokenize)."""
        return image.tokens + self.preprocessor.wrap

    def expansion(self, images, videos=()) -> int:
        """What expanding `images`' and `videos`' placeholders adds to a
        rendered prompt's token count (a video's own
        video.PreparedVideo.expansion: its timestamps, markers and runs,
        less the three ids the template rendered it as)."""
        return (sum(self.prompt_tokens(img) - 1 for img in images)
                + sum(v.expansion for v in videos))

    def budget(self, detail=None, *, where: str = "image") -> int:
        """A request's `detail` -> its pixel budget. None, "auto" and "high"
        give the server cap. "low" gives LOW_DETAIL_PIXELS, and never more
        than the cap."""
        if detail is None or detail in ("auto", "high"):
            return self.max_pixels
        if detail == "low":
            return min(LOW_DETAIL_PIXELS, self.max_pixels)
        raise ImageError(where, "image_detail",
                         f"detail {detail!r} is not one of {', '.join(DETAILS)}")

    def _plan(self, image: EncodedImage, source_hw, detail, where) -> ImagePlan:
        size, grid, tokens = self.preprocessor.plan(*source_hw, self.budget(detail, where=where),
                                                    where=where)
        return ImagePlan(self.architecture, image.format, tuple(source_hw), tuple(size),
                         tuple(grid), int(tokens))

    def count(self, image: EncodedImage, *, detail=None, where: str = "image") -> ImagePlan:
        """The image's plan from its header alone: no pixel decode (except
        a PNG whose EXIF or XMP follows its pixel data). Equal to
        `prepare(...).plan` for the same image and detail."""
        im = _open(image, where)
        try:
            hw = _oriented_size(im, image)
        except Exception as e:  # noqa: BLE001 — untrusted bytes: any decoder failure is a 400
            raise ImageError(where, "image_decode",
                             f"could not read the {image.format.upper()} image's orientation "
                             f"({type(e).__name__}: {e})") from None
        return self._plan(image, hw, detail, where)

    def prepare(self, image: EncodedImage, *, detail=None, where: str = "image") -> PreparedImage:
        """Decode, resize, normalize, patchify and digest: the one image the
        vision tower will read."""
        rgb = decode(image, where=where)
        plan = self._plan(image, (rgb.height, rgb.width), detail, where)
        pv = self.preprocessor.pixel_values(rgb, plan.size, self.budget(detail, where=where))
        pv.flags.writeable = False
        return PreparedImage(plan.architecture, plan.source_format, plan.source_size, plan.size,
                             plan.grid_thw, plan.tokens, pv, image_digest(pv, plan.grid_thw))


def load(model_type: str, checkpoint_dir: str, *, max_pixels: int | None = None,
        fetch_urls: bool | None = None, media_path: str | None = None,
        encode: Callable[[str], list[int]] | None = None) -> Vision | None:
    """The Vision for a checkpoint (a snapshot, or a pack's embedded
    checkpoint/), or None when its architecture has no registered
    preprocessor or the factory declines. `max_pixels`/`fetch_urls`/
    `media_path` None reads max_pixels_from_env/fetch_urls_from_env/
    media_path_from_env. `encode` is the engine tokenizer's text -> ids
    (no special tokens), which a video's timestamps are written with; the
    Vision's `video` is serving/video.load's answer for the checkpoint."""
    pre = preprocessor_for(model_type, read_processor_config(checkpoint_dir))
    if pre is None:
        return None
    from . import video

    vid, why = video.load(model_type, checkpoint_dir, encode)
    return Vision(pre, max_pixels_from_env(max_pixels), fetch_urls_from_env(fetch_urls),
                 media_path_from_env(media_path), vid, why)


# ---------------------------------------------------------- Qwen3.5 -------

# Qwen2VLImageProcessorPil's class defaults, which apply to any key a
# checkpoint's processor config leaves out.
_OPENAI_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
_OPENAI_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def smart_resize(height: int, width: int, factor: int, min_pixels: int,
                 max_pixels: int) -> tuple[int, int]:
    """transformers' qwen2_vl smart_resize, line for line: both sides a
    multiple of `factor` (Python's round is banker's rounding), the area
    brought inside [min_pixels, max_pixels] at the source's aspect ratio,
    with max_pixels checked first. ValueError past an aspect ratio of 200."""
    if max(height, width) / min(height, width) > 200:
        raise ValueError(
            f"absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}")
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


@dataclass(frozen=True)
class QwenVLPreprocessor:
    """Qwen2-VL-family preprocessing as Qwen3.5 checkpoints configure it
    (`Qwen2VLImageProcessor[Fast]` in preprocessor_config.json), equal bit
    for bit to transformers' `Qwen2VLImageProcessorPil`.

    `min_pixels`/`max_pixels` are the config's size.shortest_edge and
    size.longest_edge: the checkpoint's own bounds. A request's budget can
    only lower max_pixels."""

    architecture: str
    patch_size: int = 14
    temporal_patch_size: int = 2
    merge_size: int = 2
    min_pixels: int = 56 * 56
    max_pixels: int = 28 * 28 * 1280
    resample: int = Image.Resampling.BICUBIC
    rescale_factor: float = 1 / 255
    image_mean: tuple[float, ...] = _OPENAI_CLIP_MEAN
    image_std: tuple[float, ...] = _OPENAI_CLIP_STD
    # the image and video placeholders, the vision markers and pad, and the
    # audio markers and pad the Qwen3.5 tokenizer also defines
    reserved_text: tuple[str, ...] = ("<|image_pad|>", "<|vision_start|>", "<|vision_end|>",
                                      "<|video_pad|>", "<|vision_pad|>", "<|audio_start|>",
                                      "<|audio_end|>", "<|audio_pad|>")
    wrap = 0  # the template renders <|vision_start|>/<|vision_end|> itself (not a field)

    @classmethod
    def from_config(cls, architecture: str, cfg: dict) -> "QwenVLPreprocessor":
        """From a processor-config dict (read_processor_config), with the
        same key precedence as the transformers class: `size`, then
        `min_pixels`/`max_pixels` over it. A switched-off step (do_resize
        and friends set false) or another processor class is refused by
        name, not approximated."""
        kind = cfg.get("image_processor_type")
        if kind is not None and not str(kind).startswith("Qwen2VLImageProcessor"):
            raise ValueError(f"{architecture}: image_processor_type {kind!r} is not a "
                             "Qwen2VLImageProcessor; no preprocessor for it")
        for flag in ("do_resize", "do_rescale", "do_normalize", "do_convert_rgb"):
            if cfg.get(flag, True) is not True:
                raise ValueError(f"{architecture}: processor config sets {flag}="
                                 f"{cfg[flag]!r}; only the default (true) is implemented")
        size = cfg.get("size") or {}
        kw = {}
        for key, field_, src in (("shortest_edge", "min_pixels", size), ("longest_edge", "max_pixels", size),
                                 ("min_pixels", "min_pixels", cfg), ("max_pixels", "max_pixels", cfg)):
            if src.get(key) is not None:
                kw[field_] = int(src[key])
        for key in ("patch_size", "temporal_patch_size", "merge_size", "resample"):
            if cfg.get(key) is not None:
                kw[key] = int(cfg[key])
        if cfg.get("rescale_factor") is not None:
            kw["rescale_factor"] = float(cfg["rescale_factor"])
        for key in ("image_mean", "image_std"):
            if cfg.get(key) is not None:
                v = cfg[key]
                kw[key] = tuple(float(x) for x in (v if isinstance(v, (list, tuple)) else [v] * 3))
        return cls(architecture, **kw)

    @property
    def factor(self) -> int:
        return self.patch_size * self.merge_size

    def plan(self, height, width, max_pixels, *, where):
        try:
            h, w = smart_resize(height, width, self.factor, self.min_pixels,
                                min(self.max_pixels, max_pixels))
        except ValueError:
            raise ImageError(where, "image_aspect_ratio",
                             f"image is {width}x{height}; the long side may be at most 200 "
                             "times the short side") from None
        gh, gw = h // self.patch_size, w // self.patch_size
        return (h, w), (1, gh, gw), gh * gw // (self.merge_size ** 2)

    def _table(self) -> np.ndarray:
        """[3, 256] float32: each uint8 value per channel through the
        reference's rescale (uint8 -> float64, times rescale_factor, to
        float32) and normalize ((x - mean) / std in float32). These are
        elementwise IEEE operations, so looking a pixel up here gives the
        bits the reference computes for it."""
        v = (np.arange(256, dtype=np.uint8).astype(np.float64) * self.rescale_factor).astype(np.float32)
        mean = np.array(self.image_mean, dtype=np.float32)[:, None]
        std = np.array(self.image_std, dtype=np.float32)[:, None]
        return (v[None, :] - mean) / std

    def pixel_values(self, rgb, size, max_pixels=None):
        h, w = size
        a = np.asarray(rgb.resize((w, h), resample=Image.Resampling(self.resample)))
        P, m, T = self.patch_size, self.merge_size, self.temporal_patch_size
        gh, gw, C = h // P, w // P, a.shape[2]
        # (H, W, C) -> (gh/m, gw/m, m, m, C, P, P): the reference's row order
        # (merge blocks, then the 2x2 inside one) and its per-row layout
        a = a.reshape(gh // m, m, P, gw // m, m, P, C).transpose(0, 3, 1, 4, 6, 2, 5)
        a = a.reshape(gh * gw, C, 1, P * P)
        vals = self._table()[np.arange(C)[None, :, None, None], a]  # float32 (N, C, 1, P*P)
        out = np.empty((gh * gw, C, T, P * P), dtype=np.float32)
        out[...] = vals  # the one frame, duplicated along T
        return out.reshape(gh * gw, C * T * P * P)


@register("qwen3_5")
def _qwen3_5(cfg: dict | None) -> QwenVLPreprocessor | None:
    # No processor config means no image input: the class defaults are
    # Qwen2-VL's (patch 14), not Qwen3.5's (patch 16).
    return None if cfg is None else QwenVLPreprocessor.from_config("qwen3_5", cfg)


# ---------------------------------------------------------- gemma-4 -------

# The soft-token budgets Gemma4ImageProcessorPil accepts for its
# `max_soft_tokens` (it refuses any other value).
GEMMA4_SOFT_TOKENS = (70, 140, 280, 560, 1120)


def aspect_ratio_preserving_size(height: int, width: int, patch_size: int, max_patches: int,
                                 pooling_kernel_size: int) -> tuple[int, int]:
    """transformers' gemma4 `get_aspect_ratio_preserving_size`, line for
    line: the largest (h, w) at the source's aspect ratio with both sides
    multiples of pooling_kernel_size x patch_size and at most `max_patches`
    patches. A side that floors to 0 becomes one multiple and the other is
    capped at the budget's longest side. ValueError when both floor to 0,
    or the result is over budget."""
    total_px = height * width
    target_px = max_patches * (patch_size ** 2)
    factor = math.sqrt(target_px / total_px)
    ideal_height = factor * height
    ideal_width = factor * width
    side_mult = pooling_kernel_size * patch_size
    target_height = int(math.floor(ideal_height / side_mult)) * side_mult
    target_width = int(math.floor(ideal_width / side_mult)) * side_mult
    if target_height == 0 and target_width == 0:
        raise ValueError(f"resizing {height}x{width} would give a 0 x 0 image")
    max_side_length = (max_patches // pooling_kernel_size ** 2) * side_mult
    if target_height == 0:
        target_height = side_mult
        target_width = min(int(math.floor(width / height)) * side_mult, max_side_length)
    elif target_width == 0:
        target_width = side_mult
        target_height = min(int(math.floor(height / width)) * side_mult, max_side_length)
    if target_height * target_width > target_px:
        raise ValueError(f"resizing {height}x{width} to {target_height}x{target_width} "
                         f"exceeds {max_patches} patches")
    return target_height, target_width


def patch_positions(grid_thw, rows: int) -> np.ndarray:
    """[rows, 2] int64: gemma-4's `image_position_ids` for one image, the
    (x, y) of each of its t*h*w patches in row-major order, then (-1, -1)
    for every padding row (Gemma4ImageProcessorPil's meshgrid and pad)."""
    _t, gh, gw = grid_thw
    out = np.full((rows, 2), -1, dtype=np.int64)
    y, x = np.divmod(np.arange(gh * gw, dtype=np.int64), gw)
    out[:gh * gw, 0], out[:gh * gw, 1] = x, y
    return out


@dataclass(frozen=True)
class Gemma4Preprocessor:
    """gemma-4's image preprocessing, equal bit for bit to transformers'
    `Gemma4ImageProcessorPil` (the PIL backend; no torchvision). The
    gemma-4-31B-it snapshot ships no processor config, so the class
    defaults below are the ones that apply (a processor config's keys
    override them, as in transformers).

    Each image is resized at its own aspect ratio to the most pixels whose
    sides are multiples of pooling_kernel_size x patch_size (48 px) and
    whose patches number at most max_soft_tokens x pooling_kernel_size^2
    (280 x 9 = 2,520): a Pillow resize on uint8 (skipped when the size is
    already right, as the reference skips it), then [0, 1] by the
    reference's float64 -> float32 rescale, with no normalization (the
    tower maps its input to [-1, 1] itself). Patchified to
    [patches, P*P*C], channels last within a patch, and padded with zero
    rows to the budget's patch count, which is what the tower reads: its
    pooling makes one soft token of every 3 x 3 patches, so an image takes
    grid_h * grid_w / 9 tokens, at most max_soft_tokens. The padding's
    (-1, -1) positions are patch_positions(grid_thw, rows).

    The budget: a request's max_pixels lowers max_soft_tokens to what fits
    in it (max_pixels // (P^2 k^2), at least 1), the way Qwen3.5 takes
    min(checkpoint max, budget) inside its own resize rule. At the default
    server cap gemma's own 645,120-pixel budget always wins; `detail:
    "low"` (512 x 512) gives at most 113 tokens."""

    architecture: str
    patch_size: int = 16
    max_soft_tokens: int = 280
    pooling_kernel_size: int = 3
    resample: int = Image.Resampling.BICUBIC
    rescale_factor: float = 1 / 255
    # the template's placeholder, the markers the processor wraps its run
    # in, the video placeholder, and the audio placeholder and its markers
    reserved_text: tuple[str, ...] = ("<|image|>", "<|image>", "<image|>", "<|video|>",
                                      "<|audio|>", "<|audio>", "<audio|>")
    wrap = 2  # <|image> + the soft tokens + <image|> (not a field)

    @classmethod
    def from_config(cls, architecture: str, cfg: dict | None) -> "Gemma4Preprocessor":
        """From a processor-config dict, or the class defaults when the
        checkpoint carries none. Another processor class, a switched-off
        step, normalization switched on, or a soft-token budget
        transformers does not accept is refused by name, not approximated."""
        if cfg is None:
            return cls(architecture)
        kind = cfg.get("image_processor_type")
        if kind is not None and not str(kind).startswith("Gemma4ImageProcessor"):
            raise ValueError(f"{architecture}: image_processor_type {kind!r} is not a "
                             "Gemma4ImageProcessor; no preprocessor for it")
        for flag, default in (("do_resize", True), ("do_rescale", True),
                              ("do_convert_rgb", True), ("do_normalize", False)):
            if cfg.get(flag, default) is not default:
                raise ValueError(f"{architecture}: processor config sets {flag}={cfg[flag]!r}; "
                                 f"only the default ({str(default).lower()}) is implemented")
        kw = {}
        for key in ("patch_size", "max_soft_tokens", "pooling_kernel_size", "resample"):
            if cfg.get(key) is not None:
                kw[key] = int(cfg[key])
        if cfg.get("rescale_factor") is not None:
            kw["rescale_factor"] = float(cfg["rescale_factor"])
        pre = cls(architecture, **kw)
        if pre.max_soft_tokens not in GEMMA4_SOFT_TOKENS:
            raise ValueError(f"{architecture}: max_soft_tokens {pre.max_soft_tokens} is not one "
                             f"of {GEMMA4_SOFT_TOKENS}")
        return pre

    def soft_tokens(self, max_pixels: int) -> int:
        """The soft-token budget under `max_pixels`: the checkpoint's own,
        lowered to what the pixels cover."""
        per_token = self.patch_size ** 2 * self.pooling_kernel_size ** 2
        return max(1, min(self.max_soft_tokens, max_pixels // per_token))

    def plan(self, height, width, max_pixels, *, where):
        k = self.pooling_kernel_size
        try:
            h, w = aspect_ratio_preserving_size(height, width, self.patch_size,
                                                self.soft_tokens(max_pixels) * k * k, k)
        except ValueError:
            raise ImageError(where, "image_aspect_ratio",
                             f"image is {width}x{height}; it cannot be resized to a "
                             f"{k * self.patch_size}-pixel grid at its aspect ratio") from None
        gh, gw = h // self.patch_size, w // self.patch_size
        return (h, w), (1, gh, gw), gh * gw // (k * k)

    def pixel_values(self, rgb, size, max_pixels):
        h, w = size
        if (w, h) != rgb.size:
            rgb = rgb.resize((w, h), resample=Image.Resampling(self.resample))
        a = np.asarray(rgb)
        P = self.patch_size
        gh, gw, C = h // P, w // P, a.shape[2]
        # (H, W, C) -> (gh, gw, P, P, C): convert_image_to_patches' layout
        a = a.reshape(gh, P, gw, P, C).transpose(0, 2, 1, 3, 4).reshape(gh * gw, P * P * C)
        # the reference's rescale (uint8 -> float64, times rescale_factor,
        # to float32), elementwise, so a 256-entry lookup gives its bits
        table = (np.arange(256, dtype=np.uint8).astype(np.float64)
                 * self.rescale_factor).astype(np.float32)
        k = self.pooling_kernel_size
        out = np.zeros((self.soft_tokens(max_pixels) * k * k, P * P * C), dtype=np.float32)
        out[:gh * gw] = table[a]
        return out


@register("gemma4")
def _gemma4(cfg: dict | None) -> Gemma4Preprocessor:
    # No processor config (the 31B's snapshot ships none): transformers'
    # class defaults, which is what its own processor would use.
    return Gemma4Preprocessor.from_config("gemma4", cfg)


# ------------------------------------------------------ Muse-Glimmer -------

def glimmer_smart_resize(height: int, width: int, patch_size: int,
                         max_tokens: int) -> tuple[int, int]:
    """transformers' muse_glimmer `smart_resize`, line for line: the integer
    grid of `patch_size` cells (patch x merge, 28 px) closest to the
    source's aspect ratio with at most `max_tokens` cells. The candidates
    are built and ordered exactly as the reference builds them (a set of
    floor/ceil pairs, listed), so where two grids have the same aspect
    error min() keeps the one the reference keeps. When no candidate is
    within the budget the rounded ideal grid is returned, which can exceed
    it (the caller refuses that)."""
    ideal_patches_height = height / patch_size
    ideal_patches_width = width / patch_size
    ratio = ideal_patches_width / ideal_patches_height if ideal_patches_height > 0 else 1.0
    if ideal_patches_height * ideal_patches_width > max_tokens:
        ideal_patches_height = (max_tokens / ratio) ** 0.5
        ideal_patches_width = ideal_patches_height * ratio
    candidates = list(set(itertools.product(
        [math.floor(ideal_patches_height), math.ceil(ideal_patches_height)],
        [math.floor(ideal_patches_width), math.ceil(ideal_patches_width)])))
    candidates = [(ph, pw) for ph, pw in candidates
                  if ph >= 1 and pw >= 1 and ph * pw <= max_tokens]
    if not candidates:
        candidates = [(max(1, round(ideal_patches_height)), max(1, round(ideal_patches_width)))]
    ph, pw = min(candidates, key=lambda grid: abs(grid[0] / grid[1] - height / width))
    return ph * patch_size, pw * patch_size


def lanczos_resize(a: np.ndarray, height: int, width: int) -> np.ndarray:
    """An (H, W, C) uint8 image resized to (height, width), uint8: torch's
    own antialiased LANCZOS kernel on the CPU (F.interpolate, mode
    "lanczos"), the call torchvision's resize makes for the Muse-Glimmer
    processor, on the tensor layout its pil_to_tensor hands it (a CHW view
    of the HWC pixels). Pillow's LANCZOS is not the same kernel: it
    differs by one level on some pixels (tests/test_serving_glimmer_vision.py
    pins the difference), so the resize is torch's, imported here, and the
    rest of the preprocessing stays numpy."""
    import torch

    t = torch.from_numpy(np.array(a, copy=True)).permute(2, 0, 1)[None]
    out = torch.nn.functional.interpolate(t, size=[height, width], mode="lanczos",
                                          align_corners=False, antialias=True)
    return out[0].permute(1, 2, 0).contiguous().numpy()


@dataclass(frozen=True)
class MuseGlimmerPreprocessor:
    """Muse-Glimmer's image preprocessing, equal bit for bit to transformers'
    `MuseGlimmerImageProcessor` (processor_config.json's image_processor),
    whose only implementation is torchvision's backend. torchvision is not
    installed, so this reproduces what that backend computes on the CPU:

    - the size: `glimmer_smart_resize` to a grid of 28-px cells (patch 14 x
      merge 2) at the source's aspect ratio, at most max_image_tokens (4,096)
      cells, one token each; small images are upscaled;
    - the resize: `lanczos_resize`, torch's antialiased LANCZOS on uint8,
      skipped when the size is already right (torchvision returns the input
      then);
    - rescale and normalize fused as the reference fuses them: mean and std
      scaled by 1 / rescale_factor in float32, then (x - mean) / std in
      float32, a 256-entry lookup per channel;
    - patchify: patches in grid row-major order, each [T, C, P, P] with the
      one frame duplicated along T, [gh*gw, 2*3*14*14 = 1176]. The ViT
      merges 2 x 2 patches into a token itself (pixel_shuffle), so the rows
      are not merge-block-major as Qwen3.5's are.

    The budget: a request's max_pixels lowers max_image_tokens to the cells
    it covers (max_pixels // 28^2, at least 1). At the default server cap
    (2560x1440, 4,702 cells) the checkpoint's own 4,096 wins; `detail:
    "low"` (512 x 512) gives at most 334 tokens. An image whose aspect
    ratio is so extreme that the reference's grid overruns the budget is
    refused by name (image_aspect_ratio), not served past the cap."""

    architecture: str
    patch_size: int = 14
    temporal_patch_size: int = 2
    merge_size: int = 2
    max_image_tokens: int = 4096
    rescale_factor: float = 1 / 255
    image_mean: tuple[float, ...] = (0.5, 0.5, 0.5)  # IMAGENET_STANDARD_MEAN
    image_std: tuple[float, ...] = (0.5, 0.5, 0.5)   # IMAGENET_STANDARD_STD
    # every modality marker the Muse-Glimmer tokenizer defines: the
    # template's image placeholder (<|patch|>, also the image token id), the
    # markers the processor wraps an image's run in, the unused <|image|>,
    # and the video placeholder and its frame markers (it has no audio)
    reserved_text: tuple[str, ...] = ("<|patch|>", "<|image_start|>", "<|image_end|>",
                                      "<|image|>", "<|video|>", "<|vid_start|>",
                                      "<|vid_end|>", "<|vid_frame_separator|>")
    wrap = 2  # <|image_start|> + the image's tokens + <|image_end|> (not a field)

    @classmethod
    def from_config(cls, architecture: str, cfg: dict) -> "MuseGlimmerPreprocessor":
        """From processor_config.json's image_processor dict, with the class
        defaults for any key it leaves out. Another processor class, a
        switched-off step, or a resample other than LANCZOS (1, the class
        default and the 30B's) is refused by name, not approximated."""
        kind = cfg.get("image_processor_type")
        if kind is not None and not str(kind).startswith("MuseGlimmerImageProcessor"):
            raise ValueError(f"{architecture}: image_processor_type {kind!r} is not a "
                             "MuseGlimmerImageProcessor; no preprocessor for it")
        for flag in ("do_resize", "do_rescale", "do_normalize", "do_convert_rgb"):
            if cfg.get(flag, True) is not True:
                raise ValueError(f"{architecture}: processor config sets {flag}="
                                 f"{cfg[flag]!r}; only the default (true) is implemented")
        if cfg.get("resample", Image.Resampling.LANCZOS) != Image.Resampling.LANCZOS:
            raise ValueError(f"{architecture}: processor config sets resample="
                             f"{cfg['resample']!r}; only LANCZOS (1) is implemented")
        kw = {}
        for key in ("patch_size", "temporal_patch_size", "merge_size", "max_image_tokens"):
            if cfg.get(key) is not None:
                kw[key] = int(cfg[key])
        if cfg.get("rescale_factor") is not None:
            kw["rescale_factor"] = float(cfg["rescale_factor"])
        for key in ("image_mean", "image_std"):
            if cfg.get(key) is not None:
                v = cfg[key]
                kw[key] = tuple(float(x) for x in (v if isinstance(v, (list, tuple)) else [v] * 3))
        return cls(architecture, **kw)

    @property
    def factor(self) -> int:
        return self.patch_size * self.merge_size

    def max_tokens(self, max_pixels: int) -> int:
        """The token budget under `max_pixels`: the checkpoint's own,
        lowered to the 28 x 28 cells the pixels cover."""
        return max(1, min(self.max_image_tokens, max_pixels // self.factor ** 2))

    def unavailable(self) -> str | None:
        """Why this server cannot run the preprocessing (its torch has no
        LANCZOS resize, which arrived in torch 2.12), or None."""
        try:
            lanczos_resize(np.zeros((2, 2, 3), dtype=np.uint8), 3, 3)
        except Exception as e:  # noqa: BLE001 — any failure means no LANCZOS here
            return (f"this server's torch has no LANCZOS resize ({type(e).__name__}); "
                    "Muse-Glimmer's image preprocessing needs torch 2.12 or later")
        return None

    def plan(self, height, width, max_pixels, *, where):
        budget = self.max_tokens(max_pixels)
        h, w = glimmer_smart_resize(height, width, self.factor, budget)
        tokens = (h // self.factor) * (w // self.factor)
        if tokens > budget:
            raise ImageError(where, "image_aspect_ratio",
                             f"image is {width}x{height}; at its aspect ratio it cannot be "
                             f"resized under the {budget}-token budget")
        return (h, w), (1, h // self.patch_size, w // self.patch_size), tokens

    def _table(self) -> np.ndarray:
        """[3, 256] float32: each uint8 value per channel through the
        reference's fused rescale and normalize (torchvision backend's
        rescale_and_normalize): mean and std times 1 / rescale_factor in
        float32, then (x - mean) / std in float32. Elementwise IEEE
        operations, so the lookup gives the reference's bits."""
        scale = np.float32(1.0 / self.rescale_factor)
        mean = (np.array(self.image_mean, dtype=np.float32) * scale)[:, None]
        std = (np.array(self.image_std, dtype=np.float32) * scale)[:, None]
        return (np.arange(256, dtype=np.float32)[None, :] - mean) / std

    def pixel_values(self, rgb, size, max_pixels=None):
        h, w = size
        a = np.asarray(rgb)
        if a.shape[:2] != (h, w):
            a = lanczos_resize(a, h, w)
        P, T = self.patch_size, self.temporal_patch_size
        gh, gw, C = h // P, w // P, a.shape[2]
        # (H, W, C) -> (gh, gw, C, P, P): patchify's grid row-major rows,
        # channel-major within a patch
        a = a.reshape(gh, P, gw, P, C).transpose(0, 2, 4, 1, 3).reshape(gh * gw, 1, C, P * P)
        vals = self._table()[np.arange(C)[None, None, :, None], a]  # float32 (N, 1, C, P*P)
        out = np.empty((gh * gw, T, C, P * P), dtype=np.float32)
        out[...] = vals  # the one frame, duplicated along T (T before C)
        return out.reshape(gh * gw, T * C * P * P)


@register("muse_glimmer")
def _muse_glimmer(cfg: dict | None) -> MuseGlimmerPreprocessor | None:
    # No processor config means no image input: the snapshot ships one
    # (processor_config.json), and transformers' AutoProcessor would find
    # nothing to build without it.
    return None if cfg is None else MuseGlimmerPreprocessor.from_config("muse_glimmer", cfg)


# --------------------------------------------------------- the vision tower --
# The engine's side: where the tower sits in the served tree, how one image
# goes through it, and the bound on its attention dispatches. These
# functions import torch themselves, so importing this module stays
# torch-free for the dialects and the count routes.

VISION_ENV = "DRINKME_VISION"  # "0": no tower is built, and images are refused by name
BOUNDED_ENV = "DRINKME_VISION_BOUNDED"  # "0": one attention call per image (a bit-pin reference)
BOUNDED_NAME = "drinkme_vision_bounded"
UNBOUNDED_NAME = "drinkme_vision_one_call"

# Query-key pairs per ViT attention dispatch. The ViT's attention over one
# image is ONE call over all N patches (N^2 pairs), and the text path's
# longest prefill dispatch is held near 60 ms (docs/serve.md "Chunked
# prefill" has the 250 ms budget and why). Measured on gfx1151 with
# bench/vision_dispatch.py (rocprofv3, AOTriton flash with the head padded
# to 80), the longest dispatch of a whole tower forward, stock
# and compressed, 27B and MiMo: 12 ms at 1920x1080, 13-14 ms at 2560x1440
# (the default cap; 7 dispatches per layer), 16-17 ms at 3840x2176. One
# unbounded call at 1920x1080 already takes 22.5 ms, and it grows as N^2.
# 2**26 halves the dispatch count for 4% less tower time at the cap (2.70 s
# against 2.82 s) and doubles the longest dispatch (24 ms), so the bound
# stays at 2**25.
WORK_V = 1 << 25
# AOTriton's attention on gfx1151 (TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1,
# ROCm 7.13) returns wrong values at the Qwen3.5 ViT's head width, 72, on
# real tower activations: flash is off by up to 1,240 on outputs under 13 in
# magnitude, and mem-efficient returns NaN (bench/
# vision_attention_check.py). The tower's output was all NaN. Zero-padded to
# a multiple of 16 (80), flash matches an fp32 reference to bf16 rounding and
# runs no slower. The padding is exact: the zero columns add nothing to q.k,
# and the output's zero columns are dropped. On ROCm devices only; the CPU
# and CUDA calls are unchanged.
HEAD_ALIGN = 16
# The fewest query rows a dispatch carries before the KEYS are split too
# (and the pieces combined by log-sum-exp): below this the per-dispatch
# overhead outweighs the attention it bounds. At WORK_V that is an image of
# over 131,072 patches, past any checkpoint's own pixel limit.
MIN_ROWS = 256

# The attention classes bounded_attention replaces, by architecture.
_BOUNDED_TYPES = {"qwen3_5": ("Qwen3_5VisionAttention",),
                  "gemma4": ("Gemma4VisionAttention",),
                  "muse_glimmer": ("MuseGlimmerVisionAttention",)}
# The patch embeddings route_patch_embed replaces, by architecture: a Conv3d
# whose kernel IS its stride, over input already cut into patches.
_PATCH_EMBED_TYPES = {"qwen3_5": ("Qwen3_5VisionPatchEmbed",)}


def enabled_from_env() -> bool:
    """DRINKME_VISION: "0" builds no vision tower (the tower's resident
    bytes go back to the KV cache, and every image is refused by name).
    Anything else, or unset, serves images wherever the model can."""
    return os.environ.get(VISION_ENV, "").strip() != "0"


@dataclass(frozen=True)
class Tower:
    """One architecture's vision tower as the engine drives it: where
    arms.skeleton put it in the served tree (`path`, the ViT; `paths`, every
    subtree the tower spans, the ViT's alone unless given), the placeholder
    id the chat template renders an image as, the spatial merge its grid
    shrinks by, and whether the text model takes M-RoPE positions (Qwen3.5)
    or plain 1-D ones (gemma-4, Muse-Glimmer). `boi`/`eoi` are the ids the
    placeholder's expansion puts around the image's run (gemma-4's
    `<|image>`/`<image|>`; None where the template renders its own
    markers). `video_token_id` is the id a video's runs carry and
    `vision_start_id`/`vision_end_id` the markers the template renders a
    video's placeholder between, for a tower that reads video (Qwen3.5;
    serving/video.py); None elsewhere.

    `bidirectional` is how the text model attends among an image's run
    during prefill, the one fact that decides how a run may be prefilled
    (serving/image_prompt.py). False, CAUSAL (Qwen3.5, Muse-Glimmer): a row
    sees the rows before it, as text does, so a prefill span may cut a run
    and a reused prefix may end inside one. True (gemma-4's
    `use_bidirectional_attention: "vision"`): on the sliding layers the run
    attends forward within itself, so a run is never split, and a reuse
    point inside one prefills cold. The module itself is looked up in the
    model per call, so a level-2 sleep that drops the model drops the tower
    with it."""

    architecture: str
    path: str
    image_token_id: int
    merge_size: int
    mrope: bool = True
    paths: tuple[str, ...] = ()
    boi: int | None = None
    eoi: int | None = None
    bidirectional: bool = False
    video_token_id: int | None = None
    vision_start_id: int | None = None
    vision_end_id: int | None = None

    def __post_init__(self):
        if not self.paths:
            object.__setattr__(self, "paths", (self.path,))

    def module(self, model):
        return model.get_submodule(self.path)

    def _pixels(self, image: PreparedImage, device):
        import torch

        if image.released:
            raise RuntimeError(f"{image!r}: its pixel_values were released after an earlier "
                               "generation read them; prepare the image again")
        with warnings.catch_warnings():
            # pixel_values is read-only (vision.PreparedImage); the tower
            # never writes its input
            warnings.filterwarnings("ignore", message=".*not writable.*")
            return torch.from_numpy(image.pixel_values).to(device=device)

    def _checked(self, out, image: PreparedImage):
        if out.shape[0] != image.tokens:
            raise RuntimeError(f"{image!r}: the tower made {out.shape[0]} rows for "
                               f"{image.tokens} placeholder tokens")
        return out

    def features(self, model, image: PreparedImage):
        """[image.tokens, text hidden]: what the tower makes of one image,
        the rows its placeholder run is replaced by. Qwen3.5's is
        transformers' `get_image_features` for one image: pixel_values cast
        to the tower's dtype, the grid as a [1, 3] tensor, the merger's
        output (`pooler_output`)."""
        import torch

        vit = self.module(model)
        p = next(vit.parameters())
        pv = self._pixels(image, p.device).type(vit.dtype)
        grid = torch.tensor([image.grid_thw], dtype=torch.long, device=p.device)
        return self._checked(vit(pv, grid_thw=grid).pooler_output, image)


@dataclass(frozen=True)
class Gemma4Tower(Tower):
    """gemma-4's tower: the ViT at `path` (model.vision_tower: patch
    embedding, 2-D RoPE, the encoder, 3 x 3 pooling, standardization), then
    the projection into the text width at paths[1] (model.embed_vision: a
    norm and one Linear)."""

    def features(self, model, image: PreparedImage):
        """transformers' `Gemma4Model.get_image_features` for one image:
        the padded pixel_values and their positions (patch_positions) as a
        batch of one, the ViT's unpadded rows through embed_vision. The
        pixels stay float32, as the processor makes them: the patch
        embedder maps them to [-1, 1] and only then casts to its weights'
        dtype."""
        import torch

        vit = self.module(model)
        project = model.get_submodule(self.paths[1])
        p = next(vit.parameters())
        pv = self._pixels(image, p.device)[None]
        pos = torch.from_numpy(patch_positions(image.grid_thw, pv.shape[1])).to(p.device)[None]
        hidden = vit(pixel_values=pv, pixel_position_ids=pos).last_hidden_state
        return self._checked(project(inputs_embeds=hidden), image)


# The markers MuseGlimmerProcessor wraps an image's run in (its
# image_start_token/image_end_token). The config carries no ids for them:
# the processor looks them up in the tokenizer, and so does tower_for.
GLIMMER_IMAGE_MARKERS = ("<|image_start|>", "<|image_end|>")


def glimmer_patch_embed(module, pixel_values, grid_thw=None, **kwargs):
    """MuseGlimmerVisionPatchEmbedder.forward for a tower whose
    patch_embedding Linear is packed (codec/swap.CompressedLinear: 1176 x
    1536 clears the codec's bar). The stock forward reads
    `self.patch_embedding.weight.dtype` to cast the pixels, and a packed
    Linear has no `.weight`; this is the same forward, operation for
    operation, with the dtype read off the position table, which a served
    tree holds in the same dtype (bf16). Pinned equal to the stock forward
    and, packed, to transformers' get_image_features
    (tests/test_serving_glimmer_vision.py)."""
    from transformers.models.muse_glimmer.modeling_muse_glimmer import (
        get_vision_bilinear_indices_and_weights)

    batch_sequence_len = pixel_values.shape[0]
    target_dtype = module.position_embedding_table.weight.dtype
    patch_embeds = module.patch_embedding(pixel_values.to(dtype=target_dtype))
    embeddings = patch_embeds.flatten(-2).squeeze(-1)
    embeddings = embeddings.reshape(batch_sequence_len, -1)
    bilinear_indices, bilinear_weights = get_vision_bilinear_indices_and_weights(
        grid_thw, num_grid_per_side=module.num_grid_per_side, spatial_merge_size=1,
        kwargs=kwargs)
    pos_embeds = (module.position_embedding_table(bilinear_indices)
                  * bilinear_weights[:, :, None]).sum(0)
    return embeddings + pos_embeds.to(embeddings.dtype)


@dataclass(frozen=True)
class GlimmerTower(Tower):
    """Muse-Glimmer's tower: the ViT at `path` (model.vision_tower: a
    Linear patch embedding, 2-D RoPE, window attention over 32 x 32-patch
    windows with every fourth layer and the last one full, then the 2 x 2
    pixel shuffle), the adapter (gelu(fc2(gelu(fc1)))) and the projection
    into the text width, at paths[1:], and the weightless RMSNorm the
    wrapper applies last. Its text side is causal with 1-D positions, so an
    image is its rows' embeddings substituted and nothing else."""

    def features(self, model, image: PreparedImage):
        """transformers' `MuseGlimmerModel.get_image_features` for one
        image, called on the wrapper's own model (the module that holds
        the tower): pixel_values as the processor makes them (float32; the
        patch embedder casts them), the grid as a [1, 3] tensor, the one
        image's rows of `pooler_output`. A packed patch embedding is routed
        through glimmer_patch_embed first."""
        import torch

        owner = model.get_submodule(self.path.rpartition(".")[0])
        embedder = owner.vision_tower.patch_embedder
        if not isinstance(embedder.patch_embedding, torch.nn.Linear) \
                and "forward" not in vars(embedder):
            import types

            embedder.forward = types.MethodType(glimmer_patch_embed, embedder)
        p = next(owner.vision_tower.parameters())
        pv = self._pixels(image, p.device)
        grid = torch.tensor([image.grid_thw], dtype=torch.long, device=p.device)
        return self._checked(owner.get_image_features(pv, grid).pooler_output[0], image)


def tower_for(model_type: str, cfg, paths: tuple[str, ...], tokenizer=None) -> Tower:
    """The Tower for a checkpoint's config (the composite's, with a
    vision_config) whose tower sits at `paths` in the served tree
    (arms.vision_tower_paths). Muse-Glimmer's image markers are looked up
    in `tokenizer` (the checkpoint's), as its processor looks them up; a
    tokenizer that lacks either is refused by name."""
    vc = cfg.vision_config
    if model_type == "gemma4":
        text = cfg.get_text_config()
        return Gemma4Tower(model_type, paths[0], int(cfg.image_token_id),
                           int(vc.pooling_kernel_size), mrope=False, paths=tuple(paths),
                           boi=int(cfg.boi_token_id), eoi=int(cfg.eoi_token_id),
                           bidirectional=getattr(text, "use_bidirectional_attention",
                                                 None) == "vision")
    if model_type == "muse_glimmer":
        if tokenizer is None:
            raise ValueError("muse_glimmer: the tower needs the checkpoint's tokenizer to "
                             f"find its image markers {GLIMMER_IMAGE_MARKERS}")
        ids = []
        for marker in GLIMMER_IMAGE_MARKERS:
            i = tokenizer.convert_tokens_to_ids(marker)
            if not isinstance(i, int) or tokenizer.convert_ids_to_tokens(i) != marker:
                raise ValueError(f"muse_glimmer: the tokenizer has no {marker!r} token, "
                                 "which the processor wraps each image's run in")
            ids.append(i)
        return GlimmerTower(model_type, paths[0], int(cfg.image_token_id),
                            int(vc.merge_size), mrope=False, paths=tuple(paths),
                            boi=ids[0], eoi=ids[1])
    ids = [getattr(cfg, k, None) for k in ("video_token_id", "vision_start_token_id",
                                           "vision_end_token_id")]
    if any(i is None for i in ids):
        ids = [None, None, None]
    return Tower(model_type, paths[0], int(cfg.image_token_id), int(vc.spatial_merge_size),
                 paths=tuple(paths), video_token_id=ids[0], vision_start_id=ids[1],
                 vision_end_id=ids[2])


def bounded_attention(module, query, key, value, attention_mask=None, dropout=0.0,
                      scaling=None, is_causal=False, work: int | None = None, **kwargs):
    """The ViT's attention, in transformers' attention-function shape, with
    no dispatch over `work` (WORK_V) query-key pairs. query, key, value are
    [1, H, N, D] for one image; returns ([1, N, H, D], None).

    The attention is bidirectional over one image, so each block of query
    rows sees every key and nothing needs combining: the rows are split
    into blocks of work // N, each one the stock call (transformers'
    sdpa_attention_forward) on its rows. Only when a block would fall below
    MIN_ROWS rows are the keys split as well, into segments combined by
    log-sum-exp in fp32 (serving/segmented_attention's combine, without its
    causal square). At or under `work` pairs this IS the stock call, except
    on a ROCm device, where a head width that is not a multiple of
    HEAD_ALIGN is zero-padded first (AOTriton's kernels are wrong at 72,
    the width of both Qwen3.5's ViT and gemma-4's).

    A masked call (gemma-4's padded patches: [1, 1, N, N] or a broadcast
    row) splits its queries only, each block with its rows of the mask.
    At gemma-4's largest soft-token budget (1,120 tokens, 10,080 patches) a
    block is 3,328 rows."""
    import torch
    from transformers.integrations.sdpa_attention import sdpa_attention_forward

    from .segmented_attention import _partial

    d = query.shape[-1]
    if d % HEAD_ALIGN and _rocm(query):
        pad = (0, HEAD_ALIGN - d % HEAD_ALIGN)
        scale = 1.0 / math.sqrt(d) if scaling is None else scaling
        out, _ = bounded_attention(module, *(torch.nn.functional.pad(x, pad)
                                             for x in (query, key, value)),
                                   attention_mask, dropout=dropout, scaling=scale,
                                   is_causal=is_causal, work=work, **kwargs)
        return out[..., :d].contiguous(), None
    work = WORK_V if work is None else work
    n_q, n_k = query.shape[-2], key.shape[-2]
    if is_causal or n_q * n_k <= work:
        return sdpa_attention_forward(module, query, key, value, attention_mask, dropout=dropout,
                                      scaling=scaling, is_causal=is_causal, **kwargs)
    rows = work // n_k
    if rows >= MIN_ROWS or attention_mask is not None:
        rows = max(1, rows)

        def mask(i):
            if attention_mask is None or attention_mask.shape[-2] == 1:
                return attention_mask
            return attention_mask[..., i:i + rows, :]

        outs = [sdpa_attention_forward(module, query[:, :, i:i + rows], key, value, mask(i),
                                       dropout=dropout, scaling=scaling, is_causal=False,
                                       **kwargs)[0]
                for i in range(0, n_q, rows)]
        return torch.cat(outs, dim=1), None
    rows = MIN_ROWS
    seg = max(1, work // rows)
    scale = 1.0 / math.sqrt(query.shape[-1]) if scaling is None else scaling
    outs = []
    for i in range(0, n_q, rows):
        q = query[:, :, i:i + rows]
        acc = lse = None
        for a in range(0, n_k, seg):
            o, l = _partial(q, key[:, :, a:a + seg], value[:, :, a:a + seg], False, scale)
            l = l.unsqueeze(-1)
            if acc is None:
                acc, lse = o.float(), l
                continue
            new = torch.logaddexp(lse, l)
            acc.mul_(torch.exp(lse - new)).addcmul_(o, torch.exp(l - new))
            lse = new
        outs.append(acc.to(query.dtype).transpose(1, 2))
    return torch.cat(outs, dim=1).contiguous(), None


def _rocm(x) -> bool:
    """Is x on a ROCm device (where a Conv3d goes to MIOpen and SDPA to
    AOTriton)?"""
    import torch

    return x.device.type == "cuda" and torch.version.hip is not None


def linear_patch_embed(module, hidden_states):
    """Qwen3_5VisionPatchEmbed.forward on a ROCm device, as ONE GEMM: the
    Conv3d's kernel is its stride, so each output row is one flattened patch
    dotted with the flattened kernel, i.e. F.linear(x [N, C*T*P*P],
    weight.view(E, C*T*P*P), bias). The same math in another accumulation
    order, so not the stock conv's bits (bench/vision_bitpin.py measures
    the difference against transformers' tower).

    Why: on ROCm a Conv3d goes to MIOpen, which runs a Find for every input
    shape it has not seen, and every image size is a new shape. On gfx1151
    (rocprofv3) that Find ran MIOpen's naive fp64-accumulating
    kernel (naive_conv_ab_nonpacked_fwd_ncdhw_ushort_double_ushort) for
    4.3 s per dispatch, 8 times, on a 640x480 image, with the time growing
    with the patch count. That is past amdgpu's ring timeout (docs/serve.md
    "Chunked prefill"). The GEMM's longest dispatch is well under 1 ms at
    that size. Off ROCm, the stock forward."""
    import torch.nn.functional as F

    if not _rocm(hidden_states):
        return type(module).forward(module, hidden_states)
    w = module.proj.weight
    x = hidden_states.to(dtype=w.dtype).reshape(-1, w[0].numel())
    return F.linear(x, w.reshape(w.shape[0], -1), module.proj.bias)


def route_patch_embed(tower, model) -> int:
    """Route the tower's patch embedding through linear_patch_embed (a
    no-op off ROCm, decided per call by the input's device). Not switched
    off by DRINKME_VISION_BOUNDED=0: the stock conv's first call per image
    size is itself a dispatch past the desktop's budget on ROCm. Returns how
    many modules were routed."""
    import types

    kinds = _PATCH_EMBED_TYPES.get(tower.architecture, ())
    n = 0
    for mod in tower.module(model).modules():
        if type(mod).__name__ in kinds:
            mod.forward = types.MethodType(linear_patch_embed, mod)
            n += 1
    return n


def one_call_attention(module, query, key, value, *args, **kwargs):
    """bounded_attention with no bound: one call per image, still through
    its ROCm head padding (DRINKME_VISION_BOUNDED=0)."""
    return bounded_attention(module, query, key, value, *args, work=sys.maxsize, **kwargs)


def bound(tower, model) -> int:
    """Route the tower's attention through bounded_attention, or with
    DRINKME_VISION_BOUNDED=0 through one_call_attention. Each attention
    module gets its own copy of the config naming the function
    (serving/suffix_attention.install's pattern), so the text model's
    attention is untouched; calling it again re-routes. Returns how many
    modules are bounded (0 with the switch off)."""
    import copy

    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    ALL_ATTENTION_FUNCTIONS.register(BOUNDED_NAME, bounded_attention)
    ALL_ATTENTION_FUNCTIONS.register(UNBOUNDED_NAME, one_call_attention)
    off = os.environ.get(BOUNDED_ENV, "").strip() == "0"
    name = UNBOUNDED_NAME if off else BOUNDED_NAME
    kinds = _BOUNDED_TYPES.get(tower.architecture, ())
    n = 0
    for mod in tower.module(model).modules():
        if type(mod).__name__ not in kinds:
            continue
        impl = mod.config._attn_implementation
        if impl in (None, "sdpa", "eager", BOUNDED_NAME, UNBOUNDED_NAME):
            mod.config = copy.copy(mod.config)
            mod.config._attn_implementation = impl = name
        n += impl == BOUNDED_NAME
    return n


# The tower's attention self-test (attention_self_test) runs one call of
# SELFTEST_ROWS queries over SELFTEST_KEYS keys per head: 8.4M query-key
# pairs, a quarter of a bounded dispatch (WORK_V), so at 16 heads the MATH
# fallback's fp32 score matrix is sdpa.SELFTEST_BUDGET (512 MiB).
SELFTEST_ROWS = 2048
SELFTEST_KEYS = 4096
# The architectures whose ViT attention carries a mask (gemma-4's padding
# patches, create_bidirectional_mask over the valid ones). A masked SDPA call
# cannot take flash, so the self-test runs that form as well: on gfx1151 the
# other kernel, mem-efficient, returned NaN at head 72 (HEAD_ALIGN).
_MASKED_TYPES = {"gemma4"}


@dataclass(frozen=True)
class AttentionCheck:
    """attention_self_test's verdict: the shape it ran (`shape`), one
    sdpa.SelfTest per call form the tower makes (`tests`: "unmasked", and
    "masked" where its ViT masks), or the error its attention raised."""

    shape: str
    tests: tuple = ()
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and all(t.ok for _form, t in self.tests)

    @property
    def reason(self) -> str:
        """Why images are refused, for the dialects' named 400."""
        worst = "; ".join(f"{form}: max|diff| {t.max_abs:.2e} (tolerance {t.tol_max:.2e})"
                          for form, t in self.tests if not t.ok)
        return ("the vision tower's attention failed its boot self-test on this device "
                f"({self.shape}; {self.error or worst})")

    def lines(self) -> list[str]:
        from .. import sdpa

        head = f"image input: attention self-test @ {self.shape}, against fp32"
        if self.ok:
            return [head + ": " + "; ".join(f"{form} {t.line()}" for form, t in self.tests)
                    + " — AGREES"]
        out = [sdpa.RULE, "Vision tower attention self-test failed — image input refused.",
               f"  {self.shape}"]
        if self.error:
            out.append(f"  raised {self.error}")
        out += [f"  {form}: {t.line()}" + ("" if t.ok else " — DISAGREES")
                for form, t in self.tests]
        return out + [
            "  The attention kernel this device runs for the vision tower does",
            "  not match an fp32 reference at the tower's shape, so its image",
            "  features would be wrong (on gfx1151 AOTriton returned NaN at head",
            "  width 72, which the Qwen3.5 and gemma-4 ViTs both have;",
            "  serving/vision.py HEAD_ALIGN). Text is served; every image is",
            "  refused by name.",
            sdpa.RULE]


def _on_accelerator(x) -> bool:
    """Is x on a device whose attention kernels attention_self_test checks
    (anything but the CPU)?"""
    return x.device.type != "cpu"


def fp32_attention(query, key, value, attention_mask=None, scaling=None, block: int = 256):
    """softmax(q k^T scaling + mask) v in float32 by matmul, which no SDPA
    kernel computes: query [B, H, N, D], key and value [B, H/g, M, D] (kv
    heads repeated g times), out [B, H, N, D] float32. A bool mask is True
    where a key is attended (the sdpa form); any other mask is added to the
    scores (the eager form). `block` query rows at a time, so the score
    matrix held is block x M per head."""
    import torch

    groups = query.shape[1] // key.shape[1]
    kf = key.float().repeat_interleave(groups, dim=1)
    vf = value.float().repeat_interleave(groups, dim=1)
    scaling = query.shape[-1] ** -0.5 if scaling is None else scaling
    out = torch.empty(query.shape, dtype=torch.float32, device=query.device)
    for i in range(0, query.shape[-2], block):
        sc = (query[:, :, i:i + block].float() @ kf.transpose(-1, -2)) * scaling
        if attention_mask is not None:
            m = attention_mask if attention_mask.shape[-2] == 1 else \
                attention_mask[..., i:i + block, :]
            sc = (sc.masked_fill(~m, float("-inf")) if m.dtype == torch.bool
                  else sc + m.float())
        out[:, :, i:i + block] = torch.softmax(sc, -1) @ vf
    return out


def attention_error(got, query, key, value, attention_mask=None, scaling=None,
                    backend: str = ""):
    """An attention kernel's output `got` ([B, H, N, D]) on query, key and
    value, against fp32_attention on the same inputs rounded to the served
    dtype (query's), as the kernel's own output is: what is left is the
    kernel's own error, the quantity sdpa.self_test's tolerances were sized
    against. So the tolerances are those: sdpa.MAX_ULPS ulps of the
    reference's peak magnitude on the max |diff|, one bf16 eps on the mean.
    A NaN anywhere fails (the comparisons are False). The convention of
    attention_self_test (random inputs) and of the vision gates'
    --fp32-check (the tower's own activations, every call)."""
    from .. import sdpa

    ref = fp32_attention(query, key, value, attention_mask, scaling)
    diff = (got.float() - ref.to(query.dtype).float()).abs()
    max_abs, mean_abs = float(diff.max()), float(diff.mean())
    ref_max = float(ref.abs().max())
    tol_max = sdpa.MAX_ULPS * sdpa.BF16_EPS * max(1.0, ref_max)
    return sdpa.SelfTest(backend=backend, seq_len=key.shape[-2], max_abs=max_abs,
                         mean_abs=mean_abs, ref_max=ref_max, tol_max=tol_max,
                         tol_mean=sdpa.MEAN_TOL,
                         ok=max_abs <= tol_max and mean_abs <= sdpa.MEAN_TOL)


def attention_self_test(tower, model, rows: int | None = None,
                        keys: int | None = None) -> AttentionCheck | None:
    """The tower's attention as its modules call it (the function bound()
    routed them to: bounded_attention, head-padded on ROCm), on random
    inputs of its shape (heads, head width, scaling, kv groups; bf16 as
    served), against the same attention in fp32 by matmul and softmax,
    which no SDPA kernel computes (attention_error: sdpa.self_test's
    tolerances).
    The query and key scale gives logits of standard deviation ~9, the scale
    at which AOTriton's head-72 failure showed on synthetic inputs.
    HFEngine refuses images when the check is not ok, so a
    ROCm wheel whose kernel is wrong at the tower's shape refuses images at
    boot instead of serving NaN features.

    None when there is nothing to check: a tower on the CPU (its SDPA is
    not the kernel at issue, and the check would cost seconds there), no
    attention module of a bounded type, or one routed to a function that
    is not registered (eager: matmul and softmax already)."""
    import torch
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    kinds = _BOUNDED_TYPES.get(tower.architecture, ())
    vit = tower.module(model)
    mod = next((m for m in vit.modules() if type(m).__name__ in kinds), None)
    p = next(vit.parameters(), None)
    if mod is None or p is None or not _on_accelerator(p):
        return None
    impl = mod.config._attn_implementation
    if impl not in ALL_ATTENTION_FUNCTIONS:
        return None
    fn = ALL_ATTENTION_FUNCTIONS[impl]
    rows = SELFTEST_ROWS if rows is None else rows
    keys = SELFTEST_KEYS if keys is None else keys
    h = getattr(mod, "num_heads", None) or mod.config.num_attention_heads
    d = mod.head_dim
    groups = max(1, getattr(mod, "num_key_value_groups", 1))
    scaling = getattr(mod, "scaling", None) or d ** -0.5
    dev, dtype = p.device, p.dtype
    shape = f"{h} heads x {d}"
    if d % HEAD_ALIGN and _rocm(p):
        shape += f" (padded to {d + -d % HEAD_ALIGN})"
    shape += f", {rows:,} queries x {keys:,} keys, bidirectional {str(dtype).rpartition('.')[2]}"
    gen = torch.Generator(device=dev).manual_seed(20260925)
    amp = math.sqrt(9.0 / (scaling * math.sqrt(d)))

    def rand(heads, n, a=1.0):
        return (torch.randn(1, heads, n, d, generator=gen, device=dev) * a).to(dtype)

    forms = [("unmasked", None)]
    if tower.architecture in _MASKED_TYPES:
        valid = torch.arange(keys, device=dev) < keys - keys // 8  # the padding patches
        forms.append(("masked", valid.expand(1, 1, rows, keys)))
    tests = []
    try:
        with torch.inference_mode():
            q, k, v = rand(h, rows, amp), rand(h // groups, keys, amp), rand(h // groups, keys)
            for form, mask in forms:
                got = fn(mod, q, k, v, mask, dropout=0.0, scaling=scaling,
                         is_causal=False)[0].transpose(1, 2)
                tests.append((form, attention_error(got, q, k, v, mask, scaling, impl)))
    except Exception as e:  # noqa: BLE001 — an attention that raises here raises per image
        return AttentionCheck(shape, tuple(tests), f"{type(e).__name__}: "
                              f"{str(e).splitlines()[0][:160] if str(e) else ''}")
    finally:
        if dev.type == "cuda":
            torch.cuda.empty_cache()
    return AttentionCheck(shape, tuple(tests))
