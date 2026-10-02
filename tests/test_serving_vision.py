"""serving/vision.py: image bytes -> what the vision tower reads.

The preprocessing half is pinned BIT FOR BIT against transformers'
Qwen2VLImageProcessorPil, fed through transformers' own load_image (the
EXIF transpose and RGB conversion it applies to a base64 image). The
fixtures are generated in-test: sizes on each side of every smart_resize
rounding edge, EXIF- and XMP-rotated images, palette/LA/1/I;16/RGBA/CMYK
modes, GIF and WebP (animated too), and the pixel cap on and off. The
header-only count must equal what preprocessing produces for every one of
them.

The wire half (data URLs, base64, the limits and the named refusals) and
the registry seam need no transformers at all.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import struct
import zlib

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFile, WebPImagePlugin

from drinkme.serving import vision
from drinkme.serving.engine import GenerationRequest, SampleParams

# The two launch checkpoints' processor configs, verbatim: MiMo-9B's
# preprocessor_config.json (every key spelled out) and Qwen3.8-27B's (no
# rescale_factor or resample: the class defaults apply).
MIMO = {
    "do_convert_rgb": True, "do_normalize": True, "do_rescale": True, "do_resize": True,
    "image_mean": [0.5, 0.5, 0.5], "image_processor_type": "Qwen2VLImageProcessor",
    "image_std": [0.5, 0.5, 0.5], "merge_size": 2, "patch_size": 16, "resample": 3,
    "rescale_factor": 0.00392156862745098,
    "size": {"longest_edge": 16777216, "shortest_edge": 65536}, "temporal_patch_size": 2,
}
QWEN38 = {
    "size": {"longest_edge": 16777216, "shortest_edge": 65536}, "patch_size": 16,
    "temporal_patch_size": 2, "merge_size": 2, "image_mean": [0.5, 0.5, 0.5],
    "image_std": [0.5, 0.5, 0.5], "processor_class": "Qwen3VLProcessor",
    "image_processor_type": "Qwen2VLImageProcessorFast",
}
CKPT_MAX = 16777216
UNCAPPED = 1 << 40


def _vision(cfg=MIMO, max_pixels=UNCAPPED) -> vision.Vision:
    return vision.Vision(vision.preprocessor_for("qwen3_5", cfg), max_pixels)


# --------------------------------------------------------------- fixtures --

def _picture(w: int, h: int, seed: int = 0) -> Image.Image:
    """A deterministic RGB test image: a gradient, noise, a few shapes and
    some drawn text, so resampling has edges and texture to work on."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:h, 0:w]
    a = np.stack([(x * 255 // max(w - 1, 1)), (y * 255 // max(h - 1, 1)),
                  ((x + y) * 7) % 256], axis=-1).astype(np.int16)
    a += rng.integers(-24, 25, size=a.shape, dtype=np.int16)
    im = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8), "RGB")
    d = ImageDraw.Draw(im)
    d.rectangle([w // 5, h // 5, w // 2, h // 2], outline=(250, 10, 10), width=max(1, w // 100))
    d.text((min(4, w - 1), min(4, h - 1)), "PELICAN-7391 drinkme", fill=(0, 0, 0))
    return im


def _save(im: Image.Image, fmt: str, **kw) -> bytes:
    buf = io.BytesIO()
    im.save(buf, fmt, **kw)
    return buf.getvalue()


def _png_chunk(kind: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))


def _png_chunks(data: bytes) -> list[tuple[bytes, bytes]]:
    out, pos = [], 8
    while pos < len(data):
        n = int.from_bytes(data[pos:pos + 4], "big")
        out.append((data[pos + 4:pos + 8], data[pos:pos + 12 + n]))
        pos += 12 + n
    return out


def _exif_after_idat(png: bytes) -> bytes:
    """The same PNG with its eXIf chunk moved to just before IEND: Pillow
    reads it only in load()."""
    chunks = _png_chunks(png)
    exif = [raw for kind, raw in chunks if kind == b"eXIf"]
    assert len(exif) == 1
    rest = [raw for kind, raw in chunks if kind != b"eXIf"]
    return png[:8] + b"".join(rest[:-1]) + exif[0] + rest[-1]


def _png_header_only(w: int, h: int) -> bytes:
    """A PNG that declares w x h and carries no pixel data at all."""
    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + _png_chunk(b"IHDR", ihdr) + _png_chunk(b"IEND", b"")


def _exif(orientation: int) -> Image.Exif:
    e = Image.Exif()
    e[0x0112] = orientation
    return e


_XMP6 = ('<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF xmlns:rdf="http://www.w3.org/1999/'
         '02/22-rdf-syntax-ns#"><rdf:Description xmlns:tiff="http://ns.adobe.com/tiff/1.0/" '
         'tiff:Orientation="6"/></rdf:RDF></x:xmpmeta>')


def _fixtures() -> dict[str, bytes]:
    fx = {}
    # smart_resize's rounding edges (factor 32, banker's rounding): heights
    # k*32 + 16 +- 1 at a width that keeps the rounded area above min_pixels
    for h in (47, 48, 49, 79, 80, 81, 111, 112, 113):
        fx[f"round-h{h}"] = _save(_picture(2048, h, h), "PNG")
    fx["min-exact-256"] = _save(_picture(256, 256, 1), "PNG")        # exactly min_pixels
    fx["min-under-200x300"] = _save(_picture(300, 200, 2), "PNG")    # the ceil path
    fx["min-tiny-5x7"] = _save(_picture(7, 5, 3), "PNG")
    fx["aspect-200"] = _save(_picture(400, 2, 4), "PNG")             # exactly 200: allowed
    fx["cap-exact-2560x1440"] = _save(_picture(2560, 1440, 5), "JPEG", quality=90)
    fx["cap-plus-2560x1441"] = _save(_picture(2560, 1441, 6), "JPEG", quality=90)
    fx["cap-over-2560x1456"] = _save(_picture(2560, 1456, 7), "JPEG", quality=90)
    fx["retina-2880x1800"] = _save(_picture(2880, 1800, 8), "JPEG", quality=85)
    fx["1080p"] = _save(_picture(1920, 1080, 9), "PNG")
    # orientation: EXIF on JPEG, PNG (before and after the pixel data) and
    # WebP; XMP on PNG
    base = _picture(300, 200, 10)
    for o in (3, 5, 6, 8):
        fx[f"jpeg-exif-{o}"] = _save(base, "JPEG", exif=_exif(o), quality=92)
    fx["png-exif-6"] = _save(base, "PNG", exif=_exif(6))
    fx["png-exif-6-after-idat"] = _exif_after_idat(fx["png-exif-6"])
    fx["webp-exif-8"] = _save(base, "WEBP", exif=_exif(8), lossless=True)
    from PIL.PngImagePlugin import PngInfo

    info = PngInfo()
    info.add_itxt("XML:com.adobe.xmp", _XMP6)
    fx["png-xmp-6"] = _save(base, "PNG", pnginfo=info)
    # modes
    src = _picture(333, 250, 11)
    fx["png-P"] = _save(src.convert("P"), "PNG")
    pt = src.convert("P")
    fx["png-P-transparency"] = _save(pt, "PNG", transparency=3)
    fx["png-L"] = _save(src.convert("L"), "PNG")
    fx["png-LA"] = _save(src.convert("LA"), "PNG")
    fx["png-1"] = _save(src.convert("1"), "PNG")
    fx["png-RGBA"] = _save(src.convert("RGBA"), "PNG")
    wide = (np.asarray(src.convert("L"), dtype=np.uint16) * 257)
    fx["png-I16"] = _save(Image.fromarray(wide), "PNG")
    fx["jpeg-CMYK"] = _save(src.convert("CMYK"), "JPEG", quality=90)
    fx["jpeg-L"] = _save(src.convert("L"), "JPEG", quality=90)
    fx["webp-lossy"] = _save(src, "WEBP", quality=80)
    fx["webp-RGBA"] = _save(src.convert("RGBA"), "WEBP", lossless=True)
    # animation: the first frame is the image
    f2 = _picture(333, 250, 12)
    fx["gif-animated"] = _save(src.convert("P"), "GIF", save_all=True, append_images=[f2.convert("P")],
                               duration=100, loop=0)
    fx["gif-transparency"] = _save(pt, "GIF", transparency=3)
    fx["webp-animated"] = _save(src, "WEBP", save_all=True, append_images=[f2], lossless=True)
    fx["png-animated"] = _save(src, "PNG", save_all=True, append_images=[f2])
    return fx


FIXTURES = _fixtures()


def _encoded(data: bytes) -> vision.EncodedImage:
    return vision.parse_base64(base64.b64encode(data).decode(), where="t")


def _reference(data: bytes, cfg: dict, max_pixels: int | None = None):
    """transformers' answer: load_image on the base64 string (exif_transpose,
    convert RGB), then Qwen2VLImageProcessorPil built from the config, its
    longest_edge lowered to the budget exactly as vision applies it."""
    pil_mod = pytest.importorskip("transformers.models.qwen2_vl.image_processing_pil_qwen2_vl")
    from transformers.image_utils import load_image

    cfg = json.loads(json.dumps(cfg))
    if max_pixels is not None:
        cfg["size"]["longest_edge"] = min(cfg["size"]["longest_edge"], max_pixels)
    proc = pil_mod.Qwen2VLImageProcessorPil.from_dict(cfg)
    out = proc(load_image(base64.b64encode(data).decode()), return_tensors="np")
    return out["pixel_values"], tuple(int(v) for v in out["image_grid_thw"][0])


def _assert_bit_equal(img: vision.PreparedImage, ref_pv, ref_grid) -> None:
    assert img.grid_thw == ref_grid
    assert img.pixel_values.dtype == np.float32 and ref_pv.dtype == np.float32
    assert img.pixel_values.shape == ref_pv.shape
    assert img.pixel_values.tobytes() == np.ascontiguousarray(ref_pv).tobytes()


# ------------------------------------------------------------ bit parity --


@pytest.mark.parametrize("name", sorted(FIXTURES))
def test_preprocessing_equals_transformers_bit_for_bit(name):
    """No cap (the checkpoint's own 16.7 MP ceiling): pixel_values bytes and
    grid_thw equal the reference's, and the header-only count equals
    the plan preprocessing produced."""
    data = FIXTURES[name]
    v = _vision(MIMO)
    enc = _encoded(data)
    img = v.prepare(enc)
    _assert_bit_equal(img, *_reference(data, MIMO))
    assert img.tokens == img.grid_thw[1] * img.grid_thw[2] // 4
    assert v.count(enc) == img.plan


@pytest.mark.parametrize("name", ["cap-exact-2560x1440", "cap-plus-2560x1441", "cap-over-2560x1456",
                                  "retina-2880x1800", "1080p", "round-h80", "jpeg-exif-6"])
@pytest.mark.parametrize("cap", [vision.DEFAULT_MAX_PIXELS, vision.LOW_DETAIL_PIXELS, 1000 * 700])
def test_preprocessing_under_a_pixel_cap_equals_transformers(name, cap):
    """The cap is smart_resize's max_pixels, min(checkpoint, budget): the
    reference built with that longest_edge gives the same bytes."""
    data = FIXTURES[name]
    v = _vision(MIMO, max_pixels=cap)
    enc = _encoded(data)
    img = v.prepare(enc)
    _assert_bit_equal(img, *_reference(data, MIMO, cap))
    assert img.size[0] * img.size[1] <= cap
    assert v.count(enc) == img.plan


@pytest.mark.parametrize("name", ["round-h49", "min-under-200x300", "png-LA", "jpeg-exif-8",
                                  "gif-animated", "retina-2880x1800"])
def test_the_27b_config_with_its_class_defaults_equals_transformers(name):
    """Qwen3.8-27B's config leaves rescale_factor and resample to the class
    defaults: from_config fills them the way transformers does."""
    data = FIXTURES[name]
    img = _vision(QWEN38).prepare(_encoded(data))
    _assert_bit_equal(img, *_reference(data, QWEN38))


def test_detail_low_is_the_512x512_budget_and_never_above_the_cap():
    data = FIXTURES["1080p"]
    enc = _encoded(data)
    v = _vision(MIMO, max_pixels=vision.DEFAULT_MAX_PIXELS)
    low = v.prepare(enc, detail="low")
    _assert_bit_equal(low, *_reference(data, MIMO, vision.LOW_DETAIL_PIXELS))
    assert low.tokens <= 256
    assert v.count(enc, detail="low") == low.plan
    for detail in (None, "auto", "high"):
        assert v.count(enc, detail=detail).grid_thw == (1, 68, 120)  # 1920x1088: under the cap
    tight = _vision(MIMO, max_pixels=100_000)
    assert tight.budget("low") == tight.budget("high") == 100_000
    with pytest.raises(vision.ImageError) as e:
        v.count(enc, detail="medium", where="messages[0].content[1]")
    assert e.value.code == "image_detail" and str(e.value).startswith("messages[0].content[1]: ")


def test_the_first_frame_is_the_image():
    """An animated GIF/WebP/PNG reads as its first frame: the same pixels
    as that frame saved alone (lossless formats; GIF's palette is the
    first frame's own)."""
    src = _picture(333, 250, 11)
    v = _vision()
    assert v.prepare(_encoded(FIXTURES["webp-animated"])) == \
        v.prepare(_encoded(_save(src, "WEBP", lossless=True)))
    assert v.prepare(_encoded(FIXTURES["png-animated"])) == v.prepare(_encoded(_save(src, "PNG")))
    first = Image.open(io.BytesIO(FIXTURES["gif-animated"]))
    first.load()
    assert v.prepare(_encoded(FIXTURES["gif-animated"])) == \
        v.prepare(_encoded(_save(first.convert("RGB"), "PNG")))


def test_orientation_swaps_the_grid_and_the_count_sees_it():
    v = _vision()
    upright = v.count(_encoded(_save(_picture(300, 200, 10), "PNG")))
    for name in ("jpeg-exif-5", "jpeg-exif-6", "jpeg-exif-8", "png-exif-6", "png-exif-6-after-idat",
                 "webp-exif-8", "png-xmp-6"):
        plan = v.count(_encoded(FIXTURES[name]))
        assert plan.source_size == (300, 200), name
        assert plan.grid_thw == (1, upright.grid_thw[2], upright.grid_thw[1]), name
    assert v.count(_encoded(FIXTURES["jpeg-exif-3"])).source_size == (200, 300)


# ------------------------------------------------------ smart_resize table --


def test_smart_resize_equals_transformers_over_a_sweep():
    ref = pytest.importorskip("transformers.models.qwen2_vl.image_processing_pil_qwen2_vl").smart_resize
    rng = np.random.default_rng(3)
    sizes = [(int(h), int(w)) for h, w in rng.integers(1, 6000, size=(400, 2))
             if max(h, w) / min(h, w) <= 200]
    sizes += [(k * 32 + d, 1024) for k in range(1, 40) for d in (15, 16, 17)]
    for budget in (CKPT_MAX, vision.DEFAULT_MAX_PIXELS, vision.LOW_DETAIL_PIXELS, 50_000):
        for h, w in sizes:
            assert vision.smart_resize(h, w, 32, 65536, budget) == ref(h, w, 32, 65536, budget), (h, w)


@pytest.mark.parametrize("w, h, grid, tokens", [
    (512, 512, (1, 32, 32), 256),
    (1280, 800, (1, 50, 80), 1000),
    (1920, 1080, (1, 68, 120), 2040),
    (2560, 1440, (1, 90, 160), 3600),
    (2880, 1800, (1, 112, 180), 5040),
    (3840, 2160, (1, 136, 240), 8160),
    (5120, 2880, (1, 180, 320), 14400),
])
def test_the_token_table_at_the_checkpoints_own_limits(w, h, grid, tokens):
    """Screen sizes at the checkpoint's own 16.7 MP ceiling (WxH in,
    grid (t, h, w) out)."""
    pre = vision.preprocessor_for("qwen3_5", MIMO)
    _, g, n = pre.plan(h, w, UNCAPPED, where="t")
    assert (g, n) == (grid, tokens)


@pytest.mark.parametrize("w, h, size, tokens", [
    (1920, 1080, (1088, 1920), 2040),   # untouched by the cap
    (2560, 1440, (1440, 2560), 3600),   # exactly the cap
    (2880, 1800, (1504, 2400), 3525),   # Retina: resized down
    (3840, 2160, (1440, 2560), 3600),   # 4K: resized to the cap
])
def test_the_default_cap(w, h, size, tokens):
    pre = vision.preprocessor_for("qwen3_5", MIMO)
    s, _, n = pre.plan(h, w, vision.DEFAULT_MAX_PIXELS, where="t")
    assert (s, n) == (size, tokens)


def test_aspect_ratio_over_200_is_refused_by_name():
    v = _vision()
    assert v.count(_encoded(FIXTURES["aspect-200"])).source_size == (2, 400)
    with pytest.raises(vision.ImageError) as e:
        v.prepare(_encoded(_save(_picture(402, 2), "PNG")), where="messages[2].content[0]")
    assert e.value.code == "image_aspect_ratio" and "402x2" in str(e.value)


# -------------------------------------------------------- the prepared image --


def test_digest_is_sha256_of_the_pixel_bytes_then_the_grid():
    img = _vision().prepare(_encoded(FIXTURES["png-P"]))
    want = hashlib.sha256(img.pixel_values.tobytes()
                          + np.asarray(img.grid_thw, dtype="<i8").tobytes()).hexdigest()
    assert img.digest == want == vision.image_digest(img.pixel_values, img.grid_thw)
    assert not img.pixel_values.flags.writeable and img.pixel_values.flags.c_contiguous
    assert (img.architecture, img.source_format, img.source_size) == ("qwen3_5", "png", (250, 333))


def test_same_size_different_images_differ_in_digest_and_prefix_key():
    """The prefix-cache trap: two same-size images expand to the same
    placeholder run. Their keys must differ, and one
    image resent in another container must key the same."""
    v = _vision()
    a = v.prepare(_encoded(_save(_picture(640, 480, 1), "PNG")))
    b = v.prepare(_encoded(_save(_picture(640, 480, 2), "PNG")))
    again = v.prepare(_encoded(_save(_picture(640, 480, 1), "WEBP", lossless=True)))
    assert a.grid_thw == b.grid_thw and a.tokens == b.tokens
    assert a.digest != b.digest and a.prefix_key != b.prefix_key and a != b
    assert again == a and again.prefix_key == a.prefix_key and hash(again) == hash(a)
    for k in (a.prefix_key, b.prefix_key):
        assert -(2 ** 63) <= k < 0


def test_generation_request_carries_images_in_order():
    s = SampleParams()
    assert GenerationRequest([{"role": "user", "content": "hi"}], s).images == ()
    v = _vision()
    imgs = (v.prepare(_encoded(FIXTURES["png-L"])), v.prepare(_encoded(FIXTURES["jpeg-L"])))
    msgs = [{"role": "user", "content": [{"type": "text", "text": "compare"}, {"type": "image"},
                                         {"type": "image"}]}]
    req = GenerationRequest(msgs, s, images=imgs)
    assert req.images == imgs and req == GenerationRequest(msgs, s, images=imgs)


# ----------------------------------------------------------- header only --


def test_count_reads_the_header_only(monkeypatch):
    """No pixel decode for a count: load() raises here, and every format
    still counts. A PNG whose EXIF follows its pixel data is the one
    exception (Pillow reads trailing chunks in load())."""
    def no_load(self):
        raise AssertionError("count decoded pixels")

    monkeypatch.setattr(ImageFile.ImageFile, "load", no_load)
    monkeypatch.setattr(WebPImagePlugin.WebPImageFile, "load", no_load)
    v = _vision()
    for name in ("1080p", "jpeg-exif-6", "png-exif-6", "png-xmp-6", "webp-exif-8", "webp-animated",
                 "gif-animated", "png-P", "jpeg-CMYK"):
        v.count(_encoded(FIXTURES[name]))
    with pytest.raises(vision.ImageError, match="count decoded pixels"):
        v.count(_encoded(FIXTURES["png-exif-6-after-idat"]))


def test_count_of_a_header_with_no_pixels_and_prepare_refuses_it():
    """10000x5000 is exactly the bomb guard's limit: the header opens and
    counts at the cap, and decoding a file with no pixel data is a named
    400, not a 500."""
    v = _vision(max_pixels=vision.DEFAULT_MAX_PIXELS)
    enc = vision.EncodedImage(_png_header_only(10000, 5000), "png")
    plan = v.count(enc)
    assert plan.source_size == (5000, 10000) and plan.tokens <= 3600
    with pytest.raises(vision.ImageError) as e:
        v.prepare(enc, where="messages[0].content[0]")
    assert e.value.code == "image_decode"


@pytest.mark.parametrize("w, h", [(10001, 5000), (20000, 6000)])
@pytest.mark.parametrize("bomb_warning", ["error", "ignore"])
def test_the_bomb_guard_refuses_from_the_header(w, h, bomb_warning):
    """Over 50 MP: refused at open, before load() allocates. Up to 100 MP
    Pillow only warns: vision's import makes that warning an error
    ("error"), and its own header check refuses even when the filters were
    reset ("ignore", which is also what pytest's per-test filters amount
    to). Past 100 MP Pillow raises by itself. Every path is the same named
    400."""
    import warnings

    v = _vision()
    enc = vision.EncodedImage(_png_header_only(w, h), "png")
    for call in (v.count, v.prepare):
        with warnings.catch_warnings():
            warnings.filterwarnings(bomb_warning, category=Image.DecompressionBombWarning)
            with pytest.raises(vision.ImageError) as e:
                call(enc, where="input[0].content[1]")
        assert e.value.code == "image_too_many_pixels"
        assert str(e.value).startswith("input[0].content[1]: ")


def test_a_truncated_or_corrupt_image_is_a_named_400():
    data = FIXTURES["1080p"][:5000]
    with pytest.raises(vision.ImageError) as e:
        _vision().prepare(_encoded(data), where="messages[0].content[0]")
    assert e.value.code == "image_decode"
    # corrupt pixel data under a trailing eXIf: even the count must decode
    # to find the orientation, and its failure is the same 400
    png = bytearray(FIXTURES["png-exif-6-after-idat"])
    idat = png.index(b"IDAT") + 4
    png[idat + 10:idat + 60] = bytes(50)
    for call in (_vision().count, _vision().prepare):
        with pytest.raises(vision.ImageError) as e:
            call(_encoded(bytes(png)), where="messages[0].content[0]")
        assert e.value.code == "image_decode"


# ------------------------------------------------------------- the wire --


def _data_url(data: bytes, mt: str = "image/png") -> str:
    return f"data:{mt};base64," + base64.b64encode(data).decode()


def _refused(fn, *a, code: str, **kw) -> vision.ImageError:
    with pytest.raises(vision.ImageError) as e:
        fn(*a, where="messages[3].content[1]", **kw)
    assert e.value.code == code, str(e.value)
    assert str(e.value).startswith("messages[3].content[1]: ")
    return e.value


def test_data_urls_decode_and_the_bytes_decide_the_format():
    png, jpeg = FIXTURES["png-L"], FIXTURES["jpeg-L"]
    assert vision.parse_image_url(_data_url(png), where="x") == vision.EncodedImage(png, "png")
    # a declared type that is one of the four but wrong: the magic bytes win
    assert vision.parse_image_url(_data_url(jpeg, "image/png"), where="x").format == "jpeg"
    assert vision.parse_image_url("DATA:Image/PNG;BASE64," + base64.b64encode(png).decode(),
                                  where="x").format == "png"
    assert vision.parse_image_url(_data_url(jpeg, "image/jpg"), where="x").format == "jpeg"
    for name, fmt in (("gif-animated", "gif"), ("webp-lossy", "webp")):
        assert vision.parse_image_url(_data_url(FIXTURES[name], f"image/{fmt}"), where="x").format == fmt


def test_base64_tolerates_line_breaks_and_missing_padding():
    png = FIXTURES["png-1"]
    b64 = base64.encodebytes(png).decode()  # MIME: a newline every 76 characters
    assert "\n" in b64
    assert vision.parse_base64(b64, "image/png", where="x").data == png
    stripped = base64.b64encode(png).decode().rstrip("=")
    assert vision.parse_base64(stripped, None, where="x").data == png


def test_urls_are_fetched_by_default_and_off_or_malformed_ones_are_refused_by_name():
    """llama.cpp parity: http(s) fetching is ON by
    default — the fetch itself, against a local server, is
    test_serving_vision_urls.py's job. Here: the off-switch's refusal, and
    every URL shape parse_image_url still rejects regardless of the
    fetch/media-path posture."""
    for url in ("https://example.com/cat.png", "HTTP://10.0.0.1/x.png"):
        e = _refused(vision.parse_image_url, url, code="image_url_fetch_off", fetch_urls=False)
        assert "a base64 data URL" in str(e)  # says what IS accepted, not a toggle
    # file:// is off unless --media-path is set (llama semantics)
    e = _refused(vision.parse_image_url, "file:///etc/passwd", code="image_file_off")
    assert "a base64 data URL" in str(e)
    e = _refused(vision.parse_image_url, "file:///etc/passwd", code="image_file_path",
                 media_path="/tmp")
    assert "absolute" in str(e)
    _refused(vision.parse_image_url, base64.b64encode(FIXTURES["png-1"]).decode(), code="image_url")
    _refused(vision.parse_image_url, {"url": "data:..."}, code="image_url")
    _refused(vision.parse_image_url, "data:image/png;base64" + "A" * 400, code="image_data_url")
    _refused(vision.parse_image_url, "data:image/png,%89PNG", code="image_data_url")


def test_sources_message_names_what_is_accepted():
    assert vision.sources_message(fetch_urls=False, media_path=None) == "a base64 data URL"
    assert vision.sources_message(fetch_urls=True, media_path=None) == \
        "a base64 data URL or an http(s) URL"
    assert vision.sources_message(fetch_urls=False, media_path="/x") == \
        "a base64 data URL or a file:// path under the media directory"
    assert vision.sources_message(fetch_urls=True, media_path="/x") == \
        "a base64 data URL, an http(s) URL, or a file:// path under the media directory"


def test_fetch_urls_from_env(monkeypatch):
    monkeypatch.delenv("DRINKME_IMAGE_URLS", raising=False)
    assert vision.fetch_urls_from_env() is True  # default: on
    monkeypatch.setenv("DRINKME_IMAGE_URLS", "0")
    assert vision.fetch_urls_from_env() is False
    monkeypatch.setenv("DRINKME_IMAGE_URLS", "1")  # anything else: on
    assert vision.fetch_urls_from_env() is True
    assert vision.fetch_urls_from_env(False) is False  # the flag wins either way


def test_media_path_from_env(monkeypatch, capsys, tmp_path):
    monkeypatch.delenv("DRINKME_MEDIA_PATH", raising=False)
    assert vision.media_path_from_env() is None  # default: off (llama semantics)
    monkeypatch.setenv("DRINKME_MEDIA_PATH", str(tmp_path))
    assert vision.media_path_from_env() == os.path.realpath(str(tmp_path))
    monkeypatch.setenv("DRINKME_MEDIA_PATH", str(tmp_path / "nope"))
    assert vision.media_path_from_env() is None
    assert "DRINKME_MEDIA_PATH" in capsys.readouterr().err
    assert vision.media_path_from_env(str(tmp_path)) == os.path.realpath(str(tmp_path))  # explicit wins


def test_formats_other_than_the_four_are_refused_by_name():
    svg = b'<?xml version="1.0"?><svg xmlns="http://www.w3.org/2000/svg" width="4" height="4"/>'
    e = _refused(vision.parse_image_url, _data_url(svg, "image/svg+xml"), code="image_media_type")
    assert "image/svg+xml" in str(e)
    e = _refused(vision.parse_image_url, _data_url(svg, "image/png"), code="image_format")
    assert "SVG" in str(e)
    tiff = _save(_picture(8, 8), "TIFF")
    assert "TIFF" in str(_refused(vision.parse_base64, base64.b64encode(tiff).decode(), "image/png",
                                  code="image_format"))
    bmp = _save(_picture(8, 8), "BMP")
    assert "BMP" in str(_refused(vision.parse_base64, base64.b64encode(bmp).decode(),
                                 code="image_format"))
    heic = b"\x00\x00\x00\x18ftypheic" + b"\x00" * 64
    assert "HEIC" in str(_refused(vision.parse_base64, base64.b64encode(heic).decode(),
                                  code="image_format"))
    _refused(vision.parse_base64, "not base64 at all!", "image/png", code="image_base64")
    _refused(vision.parse_base64, b"bytes", "image/png", code="image_base64")
    _refused(vision.parse_base64, "AAAA", "image/heic", code="image_media_type")
    _refused(vision.parse_base64, "AAAA", None, code="image_format")


def test_over_20_mib_is_refused_before_and_after_decoding():
    big = "A" * ((vision.MAX_ENCODED_BYTES // 3 + 2) * 4)
    _refused(vision.parse_base64, big, "image/png", code="image_too_large")
    just = base64.b64encode(b"\x89PNG\r\n\x1a\n" + b"\0" * (vision.MAX_ENCODED_BYTES - 8)).decode()
    assert len(vision.parse_base64(just, "image/png", where="x").data) == vision.MAX_ENCODED_BYTES
    over = base64.b64encode(b"\x89PNG\r\n\x1a\n" + b"\0" * (vision.MAX_ENCODED_BYTES - 7)).decode()
    _refused(vision.parse_base64, over, "image/png", code="image_too_large")


def test_image_count_limit():
    vision.check_image_count(vision.MAX_IMAGES)
    with pytest.raises(vision.ImageError) as e:
        vision.check_image_count(vision.MAX_IMAGES + 1, where="messages")
    assert e.value.code == "too_many_images" and "33" in str(e.value)


# ------------------------------------------------------------ the cap -----


@pytest.mark.parametrize("raw, want", [("", vision.DEFAULT_MAX_PIXELS), ("1000000", 1_000_000),
                                       ("2560x1440", 3_686_400), (" 1920 X 1080 ", 2_073_600),
                                       ("0", None), ("-5", None), ("lots", None), ("2560x", None)])
def test_max_pixels_from_env(monkeypatch, capsys, raw, want):
    if raw:
        monkeypatch.setenv("DRINKME_IMAGE_MAX_PIXELS", raw)
    else:
        monkeypatch.delenv("DRINKME_IMAGE_MAX_PIXELS", raising=False)
    got = vision.max_pixels_from_env()
    if want is None:
        assert got == vision.DEFAULT_MAX_PIXELS
        assert "DRINKME_IMAGE_MAX_PIXELS" in capsys.readouterr().err
    else:
        assert got == want
    monkeypatch.setenv("DRINKME_IMAGE_MAX_PIXELS", "4096")
    assert vision.max_pixels_from_env(777) == 777  # the flag wins


# ---------------------------------------------------------- the registry --


def test_the_registry_holds_qwen3_5_and_reads_the_processor_config(tmp_path):
    assert "qwen3_5" in vision.architectures()
    assert vision.preprocessor_for("llama", MIMO) is None
    assert vision.preprocessor_for("qwen3_5", None) is None  # no processor config: no images
    assert vision.load("qwen3_5", str(tmp_path)) is None
    (tmp_path / "preprocessor_config.json").write_text(json.dumps(QWEN38))
    v = vision.load("qwen3_5", str(tmp_path), max_pixels=123_456)
    assert v.architecture == "qwen3_5" and v.max_pixels == 123_456
    assert v.preprocessor.patch_size == 16 and v.preprocessor.max_pixels == CKPT_MAX
    assert "<|image_pad|>" in v.reserved_text
    # processor_config.json's nested image_processor wins, as in transformers
    (tmp_path / "processor_config.json").write_text(
        json.dumps({"image_processor": dict(MIMO, patch_size=14), "processor_class": "X"}))
    assert vision.load("qwen3_5", str(tmp_path)).preprocessor.patch_size == 14


def test_from_config_takes_transformers_key_precedence_and_refuses_what_it_does_not_implement():
    pre = vision.preprocessor_for("qwen3_5", dict(MIMO, min_pixels=1000, max_pixels=50_000))
    assert (pre.min_pixels, pre.max_pixels) == (1000, 50_000)
    with pytest.raises(ValueError, match="do_normalize"):
        vision.preprocessor_for("qwen3_5", dict(MIMO, do_normalize=False))
    with pytest.raises(ValueError, match="SiglipImageProcessor"):
        vision.preprocessor_for("qwen3_5", dict(MIMO, image_processor_type="SiglipImageProcessor"))


def test_a_second_architecture_plugs_in_without_touching_callers(monkeypatch, tmp_path):
    """The seam gemma-4 used and Muse-Glimmer will: a class with `plan`,
    `pixel_values` and `wrap`, and a factory registered under its
    model_type. Vision, count and prepare are unchanged."""
    class Toy:
        architecture = "toy"
        reserved_text = ("<|image|>",)
        wrap = 0

        def plan(self, height, width, max_pixels, *, where):
            return (48, 48), (1, 3, 3), 9

        def pixel_values(self, rgb, size, max_pixels):
            return np.asarray(rgb.resize(size[::-1]), dtype=np.float32).reshape(9, -1) / 255

    monkeypatch.setitem(vision._REGISTRY, "toy", lambda cfg: Toy())
    v = vision.load("toy", str(tmp_path))
    img = v.prepare(_encoded(FIXTURES["png-L"]))
    assert (img.architecture, img.grid_thw, img.tokens, img.pixel_values.shape) == \
        ("toy", (1, 3, 3), 9, (9, 768))
    assert v.count(_encoded(FIXTURES["png-L"])) == img.plan
