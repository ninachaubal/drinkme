"""Video input's request half (serving/video.py): CPU, no downloads, no
external network.

THE REFERENCE is transformers' own Qwen3-VL video path, run here:
`read_video_pyav` decodes, `Qwen3VLVideoProcessor` samples, resizes,
normalizes and patchifies, and `Qwen3VLProcessor.replace_video_token`
writes the timestamped prompt string. The processor's only backend is
torchvision's, which is not installed (its wheels must match the torch
build), so `reference` below hands it the three torchvision functions it
calls, transcribed from torchvision's CPU kernels: resize (bicubic uint8
natively, antialiased), normalize, grayscale_to_rgb. Everything else the
reference runs is its own code: frame sampling, smart_resize, the fused
rescale/normalize, patchify, the timestamps. drinkme's bicubic_resize is
the same torch call as the transcribed resize, so the resize KERNEL is not
independently checked here, only its layout and arguments.

The pins:
  * the frames sampled, the grid, the pixel_values bit for bit and the
    prompt string, against the reference, over clips that exercise the
    budget, the upscale of a small source, an odd frame count and
    min_frames;
  * smart_resize and the sampling rule line for line;
  * timestamps from the frames' presentation times: a constant-rate clip
    gives the reference's index / fps, a variable-rate one the times the
    frames were really shown;
  * the wire (sniffing, data URLs, the source switches, the byte cap) and
    every limit, each refused by name;
  * the checkpoint's own files, where the Qwen3.8-27B snapshot is cached:
    its video config read as written, its template rendering a video as
    the three ids the engine expands, its tokenizer writing the
    timestamps.

PyAV is the optional `drinkme[video]` extra: every test that decodes is
skipped without it (`needs_av`).
"""

from __future__ import annotations

import base64
import builtins
import enum
import http.server
import io
import os
import threading
import types
from fractions import Fraction

import numpy as np
import pytest

from drinkme.serving import video, vision

needs_av = pytest.mark.skipif(not video.available(),
                              reason="PyAV (the drinkme[video] extra) is not installed")

# Qwen3.8-27B's video_preprocessor_config.json, as the snapshot ships it
QWEN38_VIDEO = {"size": {"longest_edge": 25165824, "shortest_edge": 4096}, "patch_size": 16,
                "temporal_patch_size": 2, "merge_size": 2, "image_mean": [0.5, 0.5, 0.5],
                "image_std": [0.5, 0.5, 0.5], "processor_class": "Qwen3VLProcessor",
                "video_processor_type": "Qwen3VLVideoProcessor"}
QWEN38 = ("Qwen/Qwen3.8-27B", "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0")


def clip(frames: int = 25, fps=10, w: int = 64, h: int = 48, *, codec: str = "libx264",
         fmt: str = "mp4", seed: int = 0, times=None) -> bytes:
    """A synthetic clip made with PyAV: `frames` frames of noise (distinct
    per seed) at `fps`, or at the presentation `times` (seconds) given."""
    import av

    rng = np.random.default_rng(seed)
    buf = io.BytesIO()
    with av.open(buf, "w", format=fmt) as c:
        s = c.add_stream(codec, rate=fps)
        s.width, s.height, s.pix_fmt = w, h, "yuv420p"
        if times is not None:
            s.codec_context.time_base = s.time_base = Fraction(1, 1000)
        for i in range(frames if times is None else len(times)):
            f = av.VideoFrame.from_ndarray(rng.integers(0, 256, (h, w, 3), dtype=np.uint8),
                                           format="rgb24")
            if times is not None:
                f.pts, f.time_base = int(round(times[i] * 1000)), Fraction(1, 1000)
            for p in s.encode(f):
                c.mux(p)
        for p in s.encode():
            c.mux(p)
    return buf.getvalue()


def data_url(data: bytes, mt: str = "video/mp4") -> str:
    return f"data:{mt};base64," + base64.b64encode(data).decode()


def ids(text: str) -> list[int]:
    """A stand-in tokenizer for the timestamps: one id per character."""
    return [ord(ch) for ch in text]


def video_input(**cfg) -> video.VideoInput:
    pre = video.QwenVideoPreprocessor.from_config("qwen3_5", dict(QWEN38_VIDEO, **cfg))
    return video.VideoInput(pre, ids)


def encoded(data: bytes) -> video.EncodedVideo:
    return video.parse_video_url(data_url(data), where="test")


# ------------------------------------------------------- THE REFERENCE --

class _Interpolation(enum.Enum):
    NEAREST = "nearest"
    BILINEAR = "bilinear"
    BICUBIC = "bicubic"
    LANCZOS = "lanczos"


def _tv_resize(image, size, interpolation=_Interpolation.BILINEAR, antialias=True):
    """torchvision.transforms.v2.functional.resize_image on the CPU: the
    input returned when the size is already right, else F.interpolate
    over [-1, C, H, W] with align_corners False, on uint8 itself for
    bicubic (torchvision resizes bicubic uint8 natively on the CPU)."""
    import torch

    shape = image.shape
    c, h, w = shape[-3:]
    if tuple(size) == (h, w):
        return image
    out = torch.nn.functional.interpolate(image.reshape(-1, c, h, w), size=list(size),
                                          mode=interpolation.value, align_corners=False,
                                          antialias=antialias)
    return out.reshape(shape[:-3] + (c, *size))


def _tv_normalize(t, mean, std):
    """torchvision's normalize on a float tensor: (t - mean) / std."""
    import torch

    mean = torch.as_tensor(mean, dtype=t.dtype)[:, None, None]
    std = torch.as_tensor(std, dtype=t.dtype)[:, None, None]
    return t.sub(mean).div(std)


@pytest.fixture
def reference(monkeypatch):
    """transformers' Qwen3VLVideoProcessor for Qwen3.8-27B's config (budget
    lowered by `cfg`), with the torchvision functions it calls supplied
    (module docstring)."""
    import transformers.image_processing_backends as backends
    import transformers.video_processing_utils as vpu
    from transformers.image_utils import PILImageResampling
    from transformers.models.qwen3_vl.video_processing_qwen3_vl import Qwen3VLVideoProcessor

    tvf = types.SimpleNamespace(resize=_tv_resize, normalize=_tv_normalize,
                                InterpolationMode=_Interpolation,
                                grayscale_to_rgb=lambda v: v)
    monkeypatch.setattr(backends, "tvF", tvf, raising=False)
    monkeypatch.setattr(vpu, "tvF", tvf, raising=False)
    monkeypatch.setattr(backends, "pil_torch_interpolation_mapping",
                        {PILImageResampling.BICUBIC: _Interpolation.BICUBIC}, raising=False)

    def make(**cfg):
        keys = {k: v for k, v in dict(QWEN38_VIDEO, **cfg).items()
                if k not in ("processor_class", "video_processor_type")}
        return Qwen3VLVideoProcessor(**keys)

    return make


def _ref_prompt(processor, out) -> str:
    """Qwen3VLProcessor.replace_video_token for the reference's output: the
    processor's own method, on a stand-in for the processor object (the
    real one cannot be built without torchvision's video processor)."""
    from transformers.models.qwen3_vl.processing_qwen3_vl import Qwen3VLProcessor

    p = types.SimpleNamespace(video_processor=processor, vision_start_token="<|vision_start|>",
                              vision_end_token="<|vision_end|>", video_token="<|video_pad|>")
    p._calculate_timestamps = types.MethodType(Qwen3VLProcessor._calculate_timestamps, p)
    return Qwen3VLProcessor.replace_video_token(
        p, {"video_grid_thw": out["video_grid_thw"], "video_metadata": out["video_metadata"]}, 0)


def _reference_run(processor, data: bytes):
    """read_video_pyav's every frame and metadata, then the processor's own
    sampling and preprocessing over them, as AutoProcessor runs it."""
    from transformers.video_utils import read_video_pyav

    frames, meta = read_video_pyav(io.BytesIO(data),
                                   lambda metadata, **k: np.arange(metadata.total_num_frames))
    meta.frames_indices = None
    return processor(videos=[frames], video_metadata=[meta], return_metadata=True)


# (frames, fps, w, h, budget): the budget binding; a source smaller than the
# 32-px factor upscaled under min_pixels; an odd frame count (the last frame
# repeated); min_frames raising a short clip's count; a non-integer rate
CLIPS = [(25, 10, 96, 72, 64 * 64 * 6), (7, 3, 50, 130, None), (3, 1, 20, 24, None),
         (61, 24, 160, 90, 64 * 64 * 6), (16, Fraction(30000, 1001), 64, 64, None)]


@needs_av
@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("frames, fps, w, h, budget", CLIPS)
def test_the_video_is_the_references_bit_for_bit(reference, frames, fps, w, h, budget):
    """The frames sampled, the grid, the pixel values and the prompt string
    are the reference's, for the same bytes."""
    cfg = {} if budget is None else {"size": {"longest_edge": budget, "shortest_edge": 4096}}
    proc = reference(**cfg)
    data = clip(frames, fps, w, h, seed=frames)
    out = _reference_run(proc, data)
    got = video_input(**cfg).prepare(encoded(data), where="test")
    meta = out["video_metadata"][0]
    assert got.frame_indices == tuple(int(i) for i in meta.frames_indices)
    assert got.fps == meta.fps and got.source_frames == meta.total_num_frames
    assert got.grid_thw == tuple(out["video_grid_thw"][0].tolist())
    want = out["pixel_values_videos"].numpy()
    assert got.pixel_values.shape == want.shape and np.array_equal(got.pixel_values, want)
    assert got.prompt_text() == _ref_prompt(proc, out)
    assert got.tokens == got.groups * got.group_tokens == \
        int(np.prod(got.grid_thw)) // 4 == got.prompt_text().count("<|video_pad|>")


@needs_av
def test_the_27b_budget_and_rate_give_its_token_counts():
    """At Qwen3.8-27B's own config: 2 frames per second from the clip's
    rate, every frame resized together under 25,165,824 pixels, one token
    per 32 x 32 pixels per pair of frames; the frames the processor keeps
    are evenly spread over the whole clip."""
    vi = video_input()
    got = vi.prepare(encoded(clip(40, 8, 320, 240)), where="test")  # 5 s at 8 fps
    assert got.frame_indices == (0, 4, 9, 13, 17, 22, 26, 30, 35, 39)  # 10 frames: 2 per second
    assert got.size == (256, 320) and got.grid_thw == (5, 16, 20)
    assert got.tokens == 5 * 8 * 10
    assert got.timestamps == pytest.approx([0.25, 1.375, 2.4375, 3.5, 4.625])
    assert got.prompt_text().startswith("<0.2 seconds><|vision_start|><|video_pad|>")


def test_smart_resize_is_the_references_line_for_line():
    from transformers.models.qwen3_vl.video_processing_qwen3_vl import smart_resize

    for t in (2, 3, 4, 5, 20, 767, 768):
        for h, w in ((1080, 1920), (1920, 1080), (720, 1280), (31, 400), (8, 8), (480, 640),
                     (2160, 3840), (100, 10000)):
            for lo, hi in ((4096, 25165824), (128 * 32 * 32, 32 * 32 * 768), (64, 4096)):
                assert video.video_smart_resize(t, h, w, 2, 32, lo, hi) == \
                    smart_resize(t, h, w, 2, 32, lo, hi), (t, h, w, lo, hi)
    for args in ((1, 64, 64), (4, 10, 4000)):
        with pytest.raises(ValueError):
            smart_resize(*args, 2, 32, 4096, 25165824)
        with pytest.raises(ValueError):
            video.video_smart_resize(*args, 2, 32, 4096, 25165824)


@pytest.mark.filterwarnings("ignore")
def test_the_sampling_rule_is_the_references_line_for_line():
    """Qwen3VLVideoProcessor.sample_frames (the processor's own method, on
    a stand-in carrying its fps and frame bounds), against
    QwenVideoPreprocessor.sample for the source's real frame rate."""
    from transformers.models.qwen3_vl.video_processing_qwen3_vl import Qwen3VLVideoProcessor
    from transformers.video_utils import VideoMetadata

    pre = video.QwenVideoPreprocessor("qwen3_5")
    stand_in = types.SimpleNamespace(fps=pre.fps, min_frames=pre.min_frames,
                                     max_frames=pre.max_frames)
    for total in (2, 3, 4, 5, 7, 25, 60, 61, 300, 1001, 23040):
        for fps in (1.0, 2.0, 7.5, 10.0, 24.0, 29.97002997002997, 30.0, 60.0, 240.0):
            want = Qwen3VLVideoProcessor.sample_frames(
                stand_in, VideoMetadata(total_num_frames=total, fps=fps), fps=None)
            assert pre.sample(total, fps).tolist() == np.asarray(want).tolist(), (total, fps)


def test_the_group_times_are_the_references_timestamps():
    """Qwen3VLProcessor._calculate_timestamps over index / fps, against
    group_times over the same times."""
    from transformers.models.qwen3_vl.processing_qwen3_vl import Qwen3VLProcessor

    pre = video.QwenVideoPreprocessor("qwen3_5")
    for idx, fps in (([0, 6, 12, 18, 24], 10.0), ([0, 1, 2], 1.0), (list(range(0, 300, 16)), 30.0),
                     ([0, 7, 15, 22, 30, 37], 29.97)):
        want = Qwen3VLProcessor._calculate_timestamps(None, list(idx), fps, 2)
        assert pre.group_times([i / fps for i in idx]) == want
    assert video.timestamp_text(0.25) == "<0.2 seconds>"  # Python's round-half-even, as .1f gives
    assert video.timestamp_text(383.75) == "<383.8 seconds>"


@needs_av
def test_a_variable_rate_clip_is_timed_by_when_its_frames_are_shown():
    """A screen-recording-like clip whose frames are spaced unevenly: the
    timestamps are the sampled frames' presentation times, where
    index / average_rate (the reference's value) would put them elsewhere.
    A constant-rate clip's are index / fps exactly."""
    times = [0.0, 0.1, 0.2, 1.0, 1.1, 2.5, 2.6, 2.7, 3.9, 4.0]
    vi = video_input()
    got = vi.prepare(encoded(clip(times=times, w=32, h=32)), where="test")
    shown = [times[i] for i in got.frame_indices]
    assert got.timestamps == pytest.approx(vi.preprocessor.group_times(shown))
    by_index = vi.preprocessor.group_times([i / got.fps for i in got.frame_indices])
    assert got.timestamps != pytest.approx(by_index)
    steady = vi.prepare(encoded(clip(25, 10)), where="test")
    assert steady.timestamps == tuple(
        vi.preprocessor.group_times([i / 10 for i in steady.frame_indices]))


@needs_av
@pytest.mark.parametrize("codec, fmt, mt", [("libx264", "mp4", "video/mp4"),
                                            ("mpeg4", "mov", "video/quicktime"),
                                            ("libvpx-vp9", "webm", "video/webm"),
                                            ("libx264", "matroska", "video/x-matroska")])
def test_every_container_decodes_to_the_same_plan(codec, fmt, mt):
    """MP4, MOV, WebM and MKV are read, by their magic bytes; WebM and MKV
    list no frame count, so their packets are counted instead."""
    data = clip(25, 10, codec=codec, fmt=fmt)
    enc = video.parse_video_url(data_url(data, mt), where="test")
    assert enc.format == ("webm" if fmt in ("webm", "matroska") else "mp4")
    src = video.probe(enc, where="test")
    assert (src.frames, src.fps, src.height, src.width) == (25, 10.0, 48, 64)
    got = video_input().prepare(enc, where="test")
    assert got.frame_indices == (0, 6, 12, 18, 24) and got.timestamps == (0.3, 1.5, 2.4)


@needs_av
def test_a_container_that_overstates_its_frames_is_sampled_over_the_frames_it_has(monkeypatch):
    """An MP4 whose header lists more frames than decode (an edit list
    trimming some does this): the frames are sampled again over the ones
    the stream has, rather than the clip refused; a stream that falls short
    twice is refused by name."""
    import dataclasses

    data = clip(25, 10)
    want = video_input().prepare(encoded(data), where="t")
    real = video.probe
    monkeypatch.setattr(video, "probe", lambda v, *, where: dataclasses.replace(
        real(v, where=where), frames=31))
    got = video_input().prepare(encoded(data), where="t")
    assert got == want and got.frame_indices == want.frame_indices == (0, 6, 12, 18, 24)
    assert got.source_frames == 25 and got.timestamps == want.timestamps

    def short(*a, **k):
        raise video.Shortfall(20)

    monkeypatch.setattr(video, "read_frames", short)
    with pytest.raises(video.VideoError) as e:
        video_input().prepare(encoded(data), where="t")
    assert e.value.code == "video_decode" and "decoded to 20 frames" in str(e.value)


@needs_av
def test_two_clips_differ_by_digest_and_one_clip_is_stable():
    vi = video_input()
    a, a2 = (vi.prepare(encoded(clip(seed=1)), where="t") for _ in range(2))
    b = vi.prepare(encoded(clip(seed=2)), where="t")
    assert a == a2 and a.digest == a2.digest and a != b
    assert a.prefix_key < 0 and a.prefix_key != b.prefix_key
    assert a.digest != vision.image_digest(a.pixel_values, a.grid_thw)  # never an image's
    assert not a.pixel_values.flags.writeable
    a.release()
    assert a.released and a.digest == a2.digest


@needs_av
def test_the_prompt_count_is_the_timestamps_markers_and_runs():
    """prompt_tokens: each group's timestamp text as `encode` writes it,
    its two markers and its run; expansion: that, less the three ids the
    template rendered."""
    got = video_input().prepare(encoded(clip(25, 10)), where="test")
    stamps = [video.timestamp_text(t) for t in got.timestamps]
    assert got.timestamp_ids == tuple(tuple(ids(s)) for s in stamps)
    assert got.prompt_tokens == sum(map(len, stamps)) + 2 * got.groups + got.tokens
    assert got.expansion == got.prompt_tokens - 3
    pre = vision.QwenVLPreprocessor("qwen3_5")
    veng = vision.Vision(pre, video=video_input())
    assert veng.expansion((), (got, got)) == 2 * got.expansion


# ------------------------------------------------------------- LIMITS --

@needs_av
def test_a_clip_longer_than_the_processors_frames_at_its_rate_is_refused():
    """max_frames / fps: past it the processor would read fewer than fps
    frames a second. Lowered here to 8 frames (4 s at 2 fps)."""
    vi = video_input(max_frames=8)
    assert vi.preprocessor.max_seconds == 4.0
    ok = vi.prepare(encoded(clip(40, 10)), where="t")  # 4.0 s: 8 frames, exactly the limit
    assert len(ok.frame_indices) == 8
    with pytest.raises(video.VideoError) as e:
        vi.prepare(encoded(clip(45, 10)), where="messages[0].content[1]")
    assert e.value.code == "video_too_long"
    assert str(e.value) == ("messages[0].content[1]: video is 4.5 s; at 2 frames per second "
                            "the model reads at most 8 frames, so the limit is 4 s; send a "
                            "shorter clip")
    assert video.QwenVideoPreprocessor("qwen3_5").max_seconds == 384.0


@needs_av
def test_a_one_frame_clip_is_refused_as_too_short():
    with pytest.raises(video.VideoError) as e:
        video_input().prepare(encoded(clip(1, 10)), where="t")
    assert e.value.code == "video_too_short" and "image_url" in str(e.value)


@needs_av
def test_a_frame_over_the_source_pixel_cap_is_refused_before_decoding(monkeypatch):
    monkeypatch.setattr(vision, "MAX_SOURCE_PIXELS", 64 * 48 - 1)

    def no_decode(*a, **k):
        raise AssertionError("decoded a clip the probe should have refused")

    monkeypatch.setattr(video, "read_frames", no_decode)
    with pytest.raises(video.VideoError) as e:
        video_input().prepare(encoded(clip(4, 10)), where="t")
    assert e.value.code == "video_too_many_pixels"


@needs_av
def test_truncated_or_corrupt_bytes_are_a_named_decode_error():
    data = clip(25, 10)
    for bad in (data[:len(data) // 3], data[:40] + bytes(2000)):
        with pytest.raises(video.VideoError) as e:
            video_input().prepare(video.EncodedVideo(bad, "mp4"), where="t")
        assert e.value.code == "video_decode", bad[:16]


@needs_av
def test_an_aspect_ratio_past_200_is_refused_by_name():
    with pytest.raises(video.VideoError) as e:
        video_input().prepare(encoded(clip(4, 10, w=4096, h=16)), where="t")
    assert e.value.code == "video_aspect_ratio"


def test_too_many_videos_is_refused():
    video.check_video_count(video.MAX_VIDEOS)
    with pytest.raises(video.VideoError) as e:
        video.check_video_count(video.MAX_VIDEOS + 1, where="messages")
    assert e.value.code == "too_many_videos" and str(e.value).startswith("messages: 5 videos")


def test_without_pyav_a_video_is_refused_with_the_install_line(monkeypatch, tmp_path):
    """No PyAV: the decoder refuses by name, and `load` gives no video
    input, with the reason the dialects quote."""
    real = builtins.__import__

    def no_av(name, *a, **k):
        if name == "av" or name.startswith("av."):
            raise ImportError("No module named 'av'")
        return real(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", no_av)
    with pytest.raises(video.VideoError) as e:
        video.probe(video.EncodedVideo(b"\0\0\0\x18ftypisom" + bytes(16), "mp4"), where="t")
    assert e.value.code == "video_decoder_missing" and "drinkme[video]" in str(e.value)
    monkeypatch.setattr(video, "available", lambda: False)
    got, why = video.load("qwen3_5", _snapshot_dir_with(tmp_path, QWEN38_VIDEO), ids)
    assert got is None and "PyAV is not installed" in why and "drinkme[video]" in why


# ------------------------------------------------------------- THE WIRE --

def test_the_container_comes_from_the_magic_bytes():
    assert video.sniff_format(b"\0\0\0\x18ftypisom" + bytes(8)) == "mp4"
    assert video.sniff_format(b"\0\0\0\x14ftypqt  " + bytes(8)) == "mp4"
    assert video.sniff_format(b"\0\0\0\x08moov" + bytes(8)) == "mp4"
    assert video.sniff_format(b"\x1a\x45\xdf\xa3" + bytes(8)) == "webm"
    assert video.sniff_format(b"\0\0\0\x18ftypheic" + bytes(8)) is None  # a still image
    for data, seen in ((b"RIFF\0\0\0\0AVI LIST", "AVI videos"), (b"FLV\x01" + bytes(8), "FLV"),
                       (b"\0\0\0\x18ftypavif" + bytes(8), "HEIC/AVIF"),
                       (b"\x89PNG\r\n\x1a\n" + bytes(8), "an image"),
                       (b"hello world, not a video", "not an MP4/MOV or WebM/MKV")):
        with pytest.raises(video.VideoError) as e:
            video.parse_base64(base64.b64encode(data).decode(), "video/mp4", where="t")
        assert e.value.code == "video_format" and seen in str(e.value), (data, str(e.value))


def test_data_urls_and_media_types():
    mp4 = b"\0\0\0\x18ftypisom" + bytes(32)
    for mt in ("video/mp4", "video/quicktime", "video/webm", "VIDEO/X-MATROSKA"):
        assert video.parse_video_url(data_url(mp4, mt), where="t").format == "mp4"
    cases = [(data_url(mp4, "image/png"), "video_media_type"),
             (data_url(mp4, "application/octet-stream"), "video_media_type"),
             ("data:video/mp4," + "AAAA", "video_data_url"),
             ("data:video/mp4;base64", "video_data_url"),
             ("data:video/mp4;base64,not-base64!!", "video_base64"),
             ("ftp://example.com/x.mp4", "video_url"), (7, "video_url")]
    for url, code in cases:
        with pytest.raises(video.VideoError) as e:
            video.parse_video_url(url, where="messages[0].content[0]")
        assert e.value.code == code, (url, e.value)
        assert str(e.value).startswith("messages[0].content[0]: ")
    with pytest.raises(video.VideoError) as e:
        video.parse_video_url("data:video/mp4,AAAA", where="t")
    assert "data:video/mp4;base64," in str(e.value)


def test_the_byte_cap_is_checked_before_and_after_decoding(monkeypatch):
    monkeypatch.setattr(video, "MAX_VIDEO_BYTES", 64)
    mp4 = b"\0\0\0\x18ftypisom" + bytes(40)
    assert video.parse_video_url(data_url(mp4), where="t").data == mp4
    # from the base64 length, before decoding; then from the decoded bytes
    # (66 bytes: a length the base64 check lets through)
    for extra in (64, 14):
        with pytest.raises(video.VideoError) as e:
            video.parse_video_url(data_url(mp4 + bytes(extra)), where="t")
        assert e.value.code == "video_too_large", extra


def test_fetching_off_and_file_paths_follow_the_image_rules(tmp_path):
    with pytest.raises(video.VideoError) as e:
        video.parse_video_url("https://example.com/a.mp4", where="t", fetch_urls=False)
    assert e.value.code == "video_url_fetch_off" and "base64 data URL" in str(e.value)
    with pytest.raises(video.VideoError) as e:
        video.parse_video_url("file://a.mp4", where="t")
    assert e.value.code == "video_file_off"
    mp4 = b"\0\0\0\x18ftypisom" + bytes(32)
    (tmp_path / "clips").mkdir()
    (tmp_path / "clips" / "a.mp4").write_bytes(mp4)
    got = video.parse_video_url("file://clips/a.mp4", where="t", media_path=str(tmp_path))
    assert got.data == mp4
    for bad in ("file:///etc/passwd", "file://../x.mp4", "file://clips/none.mp4"):
        with pytest.raises(video.VideoError) as e:
            video.parse_video_url(bad, where="t", media_path=str(tmp_path))
        assert e.value.code == "video_file_path", bad


class _ClipHandler(http.server.BaseHTTPRequestHandler):
    body = b""

    def log_message(self, fmt, *a):
        pass

    def do_GET(self):
        if self.path == "/missing.mp4":
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Type", "video/mp4")
        self.send_header("Content-Length", str(len(self.body)))
        self.end_headers()
        self.wfile.write(self.body)


@pytest.fixture
def clip_server():
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _ClipHandler)
    srv.daemon_threads = True
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield srv
    srv.shutdown()
    t.join(timeout=5)


def test_http_urls_are_fetched_with_the_video_cap(clip_server, monkeypatch):
    mp4 = b"\0\0\0\x18ftypisom" + bytes(1000)
    _ClipHandler.body = mp4
    base = f"http://127.0.0.1:{clip_server.server_address[1]}"
    a, b = video.parse_video_urls([(f"{base}/a.mp4", "p0"), (f"{base}/b.mp4", "p1")])
    assert a.data == b.data == mp4
    with pytest.raises(video.VideoError) as e:
        video.parse_video_url(f"{base}/missing.mp4", where="t")
    assert e.value.code == "video_fetch_status"
    monkeypatch.setattr(video, "MAX_VIDEO_BYTES", 100)
    with pytest.raises(video.VideoError) as e:
        video.parse_video_url(f"{base}/a.mp4", where="t")
    assert e.value.code == "video_too_large"


# ------------------------------------------------- THE CHECKPOINT CONFIG --

def _snapshot_dir_with(tmp_path, cfg: dict, name: str = "video_preprocessor_config.json") -> str:
    import json

    d = tmp_path / name.split(".")[0]
    d.mkdir()
    (d / name).write_text(json.dumps(cfg))
    return str(d)


def test_the_27b_config_reads_as_written():
    pre = video.QwenVideoPreprocessor.from_config("qwen3_5", QWEN38_VIDEO)
    assert (pre.patch_size, pre.temporal_patch_size, pre.merge_size) == (16, 2, 2)
    assert (pre.min_pixels, pre.max_pixels) == (4096, 25165824)
    assert (pre.fps, pre.min_frames, pre.max_frames) == (2.0, 4, 768)  # the class defaults
    assert pre.max_pixels // (pre.factor ** 2 * pre.temporal_patch_size) == 12288  # tokens at most
    assert video.QwenVideoPreprocessor.from_config("qwen3_5", {}) == \
        video.QwenVideoPreprocessor("qwen3_5")


@pytest.mark.parametrize("bad, needle", [
    ({"video_processor_type": "LlavaNextVideoProcessor"}, "not a Qwen3VLVideoProcessor"),
    ({"do_sample_frames": False}, "do_sample_frames"),
    ({"do_normalize": False}, "do_normalize"),
    ({"resample": 2}, "BICUBIC"),
    ({"num_frames": 16}, "num_frames"),
    ({"fps": 0}, "samples nothing"),
])
def test_what_the_preprocessor_does_not_implement_is_refused_by_name(bad, needle):
    with pytest.raises(ValueError, match=needle):
        video.QwenVideoPreprocessor.from_config("qwen3_5", dict(QWEN38_VIDEO, **bad))


def test_load_reads_the_checkpoints_video_config_or_says_why_not(monkeypatch, tmp_path):
    monkeypatch.setattr(video, "available", lambda: True)
    flat = _snapshot_dir_with(tmp_path, QWEN38_VIDEO)
    got, why = video.load("qwen3_5", flat, ids)
    assert why is None and got.preprocessor == video.QwenVideoPreprocessor.from_config(
        "qwen3_5", QWEN38_VIDEO)
    nested = _snapshot_dir_with(tmp_path, {"video_processor": dict(QWEN38_VIDEO, fps=4)},
                                name="processor_config.json")
    assert video.load("qwen3_5", nested, ids)[0].preprocessor.fps == 4.0
    (tmp_path / "empty").mkdir()
    assert video.load("qwen3_5", str(tmp_path / "empty"), ids) == (
        None, "the checkpoint carries no video processor config")
    assert video.load("gemma4", flat, ids) == (None, "this server has no video input for gemma4")
    assert video.load("qwen3_5", flat, None) == (
        None, "no tokenizer to write the video timestamps with")


def _cached(repo: str, revision: str, filename: str) -> str:
    """The snapshot directory in the local HF cache, or a clean skip."""
    from huggingface_hub import try_to_load_from_cache

    hit = try_to_load_from_cache(repo, filename, revision=revision)
    if not isinstance(hit, str):
        pytest.skip(f"{repo}@{revision[:12]} is not in the local HF cache ({filename})")
    return os.path.dirname(hit)


def test_the_cached_27b_snapshot_reads_its_video_config_and_renders_the_placeholder():
    """Qwen3.8-27B's own files, read-only from the local cache: its video
    config is QWEN38_VIDEO; its chat template renders a video part as the
    three ids the engine expands; and its tokenizer writes a timestamp in
    a handful of plain-text ids, none of them special."""
    import json

    from transformers import AutoConfig, AutoTokenizer

    snap = _cached(*QWEN38, "video_preprocessor_config.json")
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json",
                 "chat_template.jinja"):
        if not os.path.isfile(os.path.join(snap, name)):
            pytest.skip(f"{QWEN38[0]}: {name} is not in the local HF cache")
    with open(os.path.join(snap, "video_preprocessor_config.json")) as f:
        assert json.load(f) == QWEN38_VIDEO
    assert video.read_video_processor_config(snap) == QWEN38_VIDEO
    cfg = AutoConfig.from_pretrained(snap)
    assert (cfg.video_token_id, cfg.vision_start_token_id, cfg.vision_end_token_id) == \
        (248057, 248053, 248054)
    tok = AutoTokenizer.from_pretrained(snap)
    msgs = [{"role": "user", "content": [{"type": "text", "text": "When does it change?"},
                                         {"type": "video"}]}]
    out = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True)
    rendered = out if isinstance(out, list) else out["input_ids"]
    at = rendered.index(248057)
    assert rendered[at - 1:at + 2] == [248053, 248057, 248054]
    assert rendered.count(248057) == 1
    for t in (0.25, 1.5, 12.0, 383.8):
        stamp = tok.encode(video.timestamp_text(t), add_special_tokens=False)
        assert tok.decode(stamp) == video.timestamp_text(t)
        assert not set(stamp) & set(tok.all_special_ids)
        assert 5 <= len(stamp) <= 10


@needs_av
def test_the_cached_27b_snapshot_gets_video_input_from_the_loader():
    """engines._vision_for over Qwen3.8-27B's own config, processor configs
    and tokenizer (read-only, no weights): the Vision carries a VideoInput
    built from the checkpoint's video config, whose timestamps the
    checkpoint's tokenizer writes, and the tower names the video ids."""
    from transformers import AutoConfig, AutoTokenizer

    from drinkme.serving.engines import _vision_for

    snap = _cached(*QWEN38, "video_preprocessor_config.json")
    for name in ("config.json", "preprocessor_config.json", "tokenizer.json"):
        if not os.path.isfile(os.path.join(snap, name)):
            pytest.skip(f"{QWEN38[0]}: {name} is not in the local HF cache")
    tok = AutoTokenizer.from_pretrained(snap)
    vis, tower, why = _vision_for(AutoConfig.from_pretrained(snap), snap, QWEN38[0],
                                  tokenizer=lambda: tok)
    assert why is None and vis.video is not None and vis.video_reason is None
    assert vis.video.preprocessor == video.QwenVideoPreprocessor.from_config(
        "qwen3_5", QWEN38_VIDEO)
    assert (tower.video_token_id, tower.vision_start_id, tower.vision_end_id) == \
        (248057, 248053, 248054)
    assert vis.video.encode("<1.5 seconds>") == tok.encode("<1.5 seconds>",
                                                            add_special_tokens=False)
    no_tok, _tower, _ = _vision_for(AutoConfig.from_pretrained(snap), snap, QWEN38[0])
    assert no_tok.video is None
    assert no_tok.video_reason == "no tokenizer to write the video timestamps with"
