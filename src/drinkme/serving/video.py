"""Video input: from the bytes a `video_url` part carries to the arrays a
vision tower reads and the prompt text around them, for an architecture
whose processor takes video (Qwen3.5: Qwen3.8-27B). Torch-free at import,
like serving/vision.py: PyAV (`av`, the `drinkme[video]` extra) is imported
when a video is decoded, torch only for the resize.

Four stages:

1. THE WIRE -> EncodedVideo. vLLM's part, `{"type": "video_url",
   "video_url": {"url": ...}}`, follows the image path's source rules
   (vision.parse_media_url): a data: URL decodes inline, an http(s) URL is
   fetched unless `--no-image-urls` turned fetching off, and a file:// path
   resolves under `--media-path`. A video's own caps: MAX_VIDEO_BYTES and
   VIDEO_FETCH_TIMEOUT_S. The container comes from the magic bytes (MP4/MOV
   or WebM/Matroska), and FFmpeg is only ever handed that one demuxer; a
   declared media type only has to name one of the two.
2. DECODE (PyAV, `read_frames`). The source's frame count (the container's
   own, or its demuxed packets when it lists none), frame rate
   (`average_rate`, what transformers' read_video_pyav reads) and frame
   size are read before any frame is decoded, so a clip over a limit is
   refused first. Then the frames the processor samples are converted to
   RGB (`to_ndarray(format="rgb24")`, as read_video_pyav converts them) and
   resized as they come off the decoder, so a long or large source never
   holds its frames at full size. Rotation metadata is not applied, as
   read_video_pyav does not apply it.
3. THE PROCESSOR (`QwenVideoPreprocessor`), transformers'
   Qwen3VLVideoProcessor step for step: the frames sampled at the
   processor's fps (2) from the source's real frame rate, at least
   min_frames and at most max_frames; one size for every frame by the
   video smart_resize, which budgets the pixels of ALL frames together
   (25,165,824 on Qwen3.8-27B: 12,288 tokens at most); torchvision's uint8
   bicubic antialiased resize, made by the torch call torchvision makes on
   the CPU; rescale and normalize fused; and patchify into groups of
   temporal_patch_size frames, the last frame repeated to fill the last
   group. One video token is 32 x 32 pixels over 2 frames.
4. THE PROMPT. The chat template renders a video part as
   `<|vision_start|><|video_pad|><|vision_end|>`. Each temporal group
   becomes `<{t:.1f} seconds><|vision_start|>` + its tokens +
   `<|vision_end|>`, with t the group's time (Qwen3VLProcessor.
   replace_video_token), and the group's run is positioned by M-RoPE as an
   image of one frame (Qwen3_5Model.get_rope_index splits a video's grid
   per group). The template's three ids are replaced whole, the layout of
   the Qwen3-VL release processor (transformers 4.57) and of vLLM;
   transformers 5.15.1's generic replacement swaps only the pad and keeps
   an outer `<|vision_start|>`/`<|vision_end|>` pair around the groups.
   Each timestamp's text is tokenized on its own (vLLM does the same), by
   the engine's tokenizer (`VideoInput.encode`).

TIMESTAMPS. A group's time is the mean of its first and last frame's
times, as Qwen3VLProcessor._calculate_timestamps computes it. A frame's
time is its presentation time relative to the first frame. For a
constant-frame-rate clip that is index / fps, the reference's own value;
for a variable-rate one (a phone's screen recording) it is when the frame
was really shown, where index / average_rate drifts. Without real
metadata transformers falls back to 24 fps and every timestamp is wrong:
here the rate always comes from the container, and a stream with none is
refused.

LIMITS, refused by name rather than truncated: MAX_VIDEOS per request,
MAX_VIDEO_BYTES per video, vision.MAX_SOURCE_PIXELS per frame, fewer than
temporal_patch_size frames, and a clip longer than max_frames / fps (384 s
on Qwen3.8-27B): past it the reference samples fewer than fps frames per
second, so the timestamps would be spaced wider than the model's own rate
without anyone having asked for that.

Every refusal is a VideoError: vision.ImageError's subclass, so every
handler that renders an image refusal renders this one the same way (a 400
naming the part, with a stable `code`).
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
from dataclasses import dataclass
from io import BytesIO
from typing import Callable

import numpy as np

from . import vision

# ------------------------------------------------------------- the limits --

MAX_VIDEO_BYTES = 64 * 1024 * 1024  # one video, after base64 decoding OR fetched/read raw
MAX_VIDEOS = 4                      # per request
VIDEO_FETCH_TIMEOUT_S = 30.0        # wall clock: every redirect hop and the whole body read
FORMATS = ("mp4", "webm")           # ISO BMFF (MP4, MOV, M4V) and Matroska (WebM, MKV)
# FFmpeg's demuxer for each: av.open(..., format=) takes any one name of the
# demuxer's comma-separated list ("mov,mp4,m4a,3gp,3g2,mj2", "matroska,webm")
_DEMUXER = {"mp4": "mov", "webm": "matroska"}
_MEDIA_TYPES = {"video/mp4": "mp4", "video/quicktime": "mp4", "video/x-m4v": "mp4",
                "video/webm": "webm", "video/x-matroska": "webm", "video/matroska": "webm"}
# ISO BMFF brands that are still images, not video (HEIF, AVIF)
_IMAGE_BRANDS = (b"heic", b"heix", b"hevc", b"hevx", b"heim", b"heis", b"mif1", b"msf1",
                 b"avif", b"avis")
# QuickTime files from before the ftyp box open with one of these atoms
_QT_ATOMS = (b"moov", b"mdat", b"wide", b"free", b"skip")
INSTALL_HINT = ("install PyAV, the drinkme[video] extra (`uv pip install av` in the "
                "server's environment), and restart the server")


class VideoError(vision.ImageError):
    """A video refused, phrased for a 400 (invalid_request_error). The
    message starts with `where`, the request part it came from. `code`:

      video_url_fetch_off      an http(s) URL, with fetching turned off
                               (--no-image-urls / DRINKME_IMAGE_URLS=0)
      video_file_off           a file:// URL, with no --media-path set
      video_file_path          a file:// path that is absolute, contains
                               `..`, resolves outside --media-path, or
                               does not exist
      video_fetch_unreachable  the http(s) fetch could not connect
      video_fetch_timeout      the http(s) fetch exceeded VIDEO_FETCH_TIMEOUT_S
      video_fetch_status       a non-2xx/redirect status, a redirect with
                               no Location, or too many redirects
      video_url                not http(s), file:// or a data URL
      video_data_url           a malformed data URL, or one that is not base64
      video_media_type         a declared type other than MP4/MOV/WebM/MKV
      video_base64             the payload is not valid base64
      video_too_large          over MAX_VIDEO_BYTES (decoded, fetched, or read)
      video_format             the bytes are neither container
      video_too_many_pixels    a frame over vision.MAX_SOURCE_PIXELS
      video_aspect_ratio       long side over 200x the short side (smart_resize)
      video_too_short          fewer frames than one temporal group
      video_too_long           longer than the processor's max_frames / fps
      video_decode             no video stream, no frame rate, or the
                               decoder failed (truncated, corrupt)
      video_decoder_missing    PyAV is not installed (the drinkme[video] extra)
      too_many_videos          over MAX_VIDEOS in one request
    """


VIDEO = vision.MediaKind("video", VideoError, lambda: MAX_VIDEO_BYTES,
                         lambda: VIDEO_FETCH_TIMEOUT_S, "video/*", "data:video/mp4;base64,...")


def available() -> bool:
    """Is PyAV importable here? Looked up without importing it."""
    return importlib.util.find_spec("av") is not None


def _av(where: str):
    """PyAV, imported on first use; a server without it refuses the video
    by name, with the install line."""
    try:
        import av
    except ImportError:
        raise VideoError(where, "video_decoder_missing",
                         "video input needs PyAV, which this server does not have; "
                         + INSTALL_HINT) from None
    return av


# ------------------------------------------------------ 1. the wire -------

@dataclass(frozen=True)
class EncodedVideo:
    """A video's bytes as the request carried them, the container already
    sniffed from the magic bytes. Nothing is decoded yet."""

    data: bytes
    format: str  # one of FORMATS


def sniff_format(data: bytes) -> str | None:
    """The container the magic bytes name, or None."""
    if data[:4] == b"\x1a\x45\xdf\xa3":  # EBML: Matroska and WebM
        return "webm"
    if data[4:8] == b"ftyp":
        return None if data[8:12] in _IMAGE_BRANDS else "mp4"
    if data[4:8] in _QT_ATOMS:
        return "mp4"
    return None


def _looks_like(data: bytes) -> str | None:
    """A name for a common format this server refuses, for the message."""
    if data[4:8] == b"ftyp":
        return "HEIC/AVIF"
    if data[:4] == b"RIFF" and data[8:12] == b"AVI ":
        return "AVI"
    if data[:3] == b"FLV":
        return "FLV"
    if data[:4] == b"OggS":
        return "Ogg"
    if data[:1] == b"\x47" and data[188:189] == b"\x47":
        return "MPEG-TS"
    if vision.sniff_format(data):
        return "image"
    return None


def _check_media_type(media_type, where: str) -> None:
    if not isinstance(media_type, str) or media_type.strip().lower() not in _MEDIA_TYPES:
        raise VideoError(where, "video_media_type",
                         f"media type {media_type!r} is not supported; send MP4, MOV, WebM or "
                         "MKV (video/mp4, video/quicktime, video/webm, video/x-matroska)")


def _decoded_bytes(data: bytes, *, where: str, noun: str) -> EncodedVideo:
    """The bytes a payload, fetch or file read produced -> EncodedVideo:
    the size cap, then the magic-byte sniff."""
    if len(data) > MAX_VIDEO_BYTES:
        raise VideoError(where, "video_too_large",
                         f"{noun} is {len(data)} bytes; the limit is {MAX_VIDEO_BYTES} "
                         f"({MAX_VIDEO_BYTES // (1024 * 1024)} MiB)")
    fmt = sniff_format(data)
    if fmt is None:
        seen = _looks_like(data)
        if seen == "image":
            msg = f"{noun} is an image; send it as an image_url part"
        else:
            msg = ((f"{seen} videos are not supported" if seen else
                    f"{noun} is not an MP4/MOV or WebM/MKV video")
                   + "; send MP4, MOV, WebM or MKV")
        raise VideoError(where, "video_format", msg)
    return EncodedVideo(data, fmt)


def parse_base64(data, media_type: str | None = None, *, where: str) -> EncodedVideo:
    """A base64 payload (a data URL's body) -> EncodedVideo. `media_type`,
    when given, must name MP4/MOV or WebM/MKV; the bytes decide which."""
    if media_type is not None:
        _check_media_type(media_type, where)
    return _decoded_bytes(vision.base64_bytes(data, where=where, kind=VIDEO),
                          where=where, noun="video")


def parse_video_url(url, *, where: str, fetch_urls: bool = True,
                    media_path: str | None = None) -> EncodedVideo:
    """A `video_url.url` string -> EncodedVideo, by the image path's
    source rules (vision.parse_media_url) with a video's caps."""
    return vision.parse_media_url(url, where=where, fetch_urls=fetch_urls,
                                  media_path=media_path, kind=VIDEO,
                                  from_bytes=_decoded_bytes, from_base64=parse_base64)


def parse_video_urls(pairs: list[tuple[str, str]], *, fetch_urls: bool = True,
                     media_path: str | None = None) -> list[EncodedVideo]:
    """`parse_video_url` over several (url, where) pairs, the downloads
    concurrent (vision.parse_many); results in the order of `pairs`."""
    def one(u, *, where):
        return parse_video_url(u, where=where, fetch_urls=fetch_urls, media_path=media_path)

    return vision.parse_many(pairs, one)


def check_video_count(n: int, *, where: str = "request") -> None:
    """Refuse a request carrying more than MAX_VIDEOS videos."""
    if n > MAX_VIDEOS:
        raise VideoError(where, "too_many_videos",
                         f"{n} videos in one request; the limit is {MAX_VIDEOS}")


# ------------------------------------------------- 3. the processor -------

def video_smart_resize(num_frames: int, height: int, width: int, temporal_factor: int,
                       factor: int, min_pixels: int, max_pixels: int) -> tuple[int, int]:
    """transformers' qwen3_vl video `smart_resize`, line for line: both
    sides a multiple of `factor` after a source smaller than `factor` is
    scaled up, and the pixels of all the frames (their count rounded to a
    multiple of `temporal_factor`) brought inside [min_pixels,
    max_pixels] at the source's aspect ratio. ValueError for fewer frames
    than `temporal_factor` or an aspect ratio past 200."""
    if num_frames < temporal_factor:
        raise ValueError(f"t:{num_frames} must be larger than temporal_factor:{temporal_factor}")
    if height < factor or width < factor:
        scale = max(factor / height, factor / width)
        height = int(height * scale)
        width = int(width * scale)
    if max(height, width) / min(height, width) > 200:
        raise ValueError(
            f"absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}")
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    t_bar = round(num_frames / temporal_factor) * temporal_factor
    if t_bar * h_bar * w_bar > max_pixels:
        beta = math.sqrt((num_frames * height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif t_bar * h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (num_frames * height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def bicubic_resize(frame: np.ndarray, height: int, width: int) -> np.ndarray:
    """A [C, H, W] uint8 frame resized to (height, width), uint8: torch's
    antialiased bicubic kernel on uint8, which is the call torchvision's
    resize makes for a uint8 tensor on the CPU (it resizes bicubic uint8
    natively there, without a float round trip), on the contiguous
    [N, C, H, W] layout transformers' video processor hands it. A frame
    already the right size is returned as it is, as torchvision returns
    it. torchvision itself is not installed (its wheels must match the
    torch build), so the kernel is torch's, imported here, and the rest of
    the preprocessing stays numpy."""
    if frame.shape[1:] == (height, width):
        return frame
    import torch

    t = torch.from_numpy(np.ascontiguousarray(frame))[None]
    out = torch.nn.functional.interpolate(t, size=[height, width], mode="bicubic",
                                          align_corners=False, antialias=True)
    return out[0].numpy()


@dataclass(frozen=True)
class QwenVideoPreprocessor:
    """Qwen3-VL-family video preprocessing as Qwen3.5 checkpoints configure
    it (`Qwen3VLVideoProcessor` in video_preprocessor_config.json), step for
    step with transformers' class (module docstring, stage 3). The class
    defaults apply to any key the config leaves out; Qwen3.8-27B's sets
    only the patch sizes, the pixel bounds (shortest_edge 4,096,
    longest_edge 25,165,824) and mean and std (0.5).

    `rendered` is how many ids the chat template renders a video as
    (`<|vision_start|><|video_pad|><|vision_end|>`), `wrap` the markers
    around each temporal group's run (`<|vision_start|>`,
    `<|vision_end|>`)."""

    architecture: str
    patch_size: int = 16
    temporal_patch_size: int = 2
    merge_size: int = 2
    min_pixels: int = 128 * 32 * 32
    max_pixels: int = 32 * 32 * 768
    fps: float = 2.0
    min_frames: int = 4
    max_frames: int = 768
    rescale_factor: float = 1 / 255
    image_mean: tuple[float, ...] = (0.5, 0.5, 0.5)  # IMAGENET_STANDARD_MEAN
    image_std: tuple[float, ...] = (0.5, 0.5, 0.5)   # IMAGENET_STANDARD_STD
    rendered = 3  # (not a field)
    wrap = 2      # (not a field)

    @classmethod
    def from_config(cls, architecture: str, cfg: dict) -> "QwenVideoPreprocessor":
        """From a video-processor config dict (read_video_processor_config),
        `size` then `min_pixels`/`max_pixels` over it as the image config
        reads them. Another processor class, a switched-off step, a resample
        other than BICUBIC, or a fixed `num_frames` (the processor would
        sample a fixed count instead of by fps) is refused by name, not
        approximated."""
        kind = cfg.get("video_processor_type")
        if kind is not None and not str(kind).startswith("Qwen3VLVideoProcessor"):
            raise ValueError(f"{architecture}: video_processor_type {kind!r} is not a "
                             "Qwen3VLVideoProcessor; no video preprocessor for it")
        for flag in ("do_resize", "do_rescale", "do_normalize", "do_convert_rgb",
                     "do_sample_frames"):
            if cfg.get(flag, True) is not True:
                raise ValueError(f"{architecture}: video processor config sets {flag}="
                                 f"{cfg[flag]!r}; only the default (true) is implemented")
        if cfg.get("resample", 3) != 3:
            raise ValueError(f"{architecture}: video processor config sets resample="
                             f"{cfg['resample']!r}; only BICUBIC (3) is implemented")
        if cfg.get("num_frames") is not None:
            raise ValueError(f"{architecture}: video processor config sets num_frames="
                             f"{cfg['num_frames']!r}; only sampling by fps is implemented")
        size = cfg.get("size") or {}
        kw: dict = {}
        for key, field_, src in (("shortest_edge", "min_pixels", size),
                                 ("longest_edge", "max_pixels", size),
                                 ("min_pixels", "min_pixels", cfg),
                                 ("max_pixels", "max_pixels", cfg)):
            if src.get(key) is not None:
                kw[field_] = int(src[key])
        for key in ("patch_size", "temporal_patch_size", "merge_size", "min_frames",
                    "max_frames"):
            if cfg.get(key) is not None:
                kw[key] = int(cfg[key])
        for key in ("fps", "rescale_factor"):
            if cfg.get(key) is not None:
                kw[key] = float(cfg[key])
        for key in ("image_mean", "image_std"):
            if cfg.get(key) is not None:
                v = cfg[key]
                kw[key] = tuple(float(x) for x in (v if isinstance(v, (list, tuple)) else [v] * 3))
        pre = cls(architecture, **kw)
        if pre.fps <= 0 or pre.min_frames < 1 or pre.max_frames < pre.min_frames:
            raise ValueError(f"{architecture}: video processor config fps={pre.fps}, "
                             f"min_frames={pre.min_frames}, max_frames={pre.max_frames} "
                             "samples nothing")
        return pre

    @property
    def factor(self) -> int:
        return self.patch_size * self.merge_size

    @property
    def max_seconds(self) -> float:
        """The longest clip served at the processor's own fps: max_frames /
        fps (384 s at Qwen3.8-27B's 768 frames and 2 fps)."""
        return self.max_frames / self.fps

    def check_length(self, total: int, fps: float, *, where: str) -> None:
        """Refuse a source of `total` frames at `fps` that is too short for
        one temporal group, or so long that the reference's sampling rule
        would read fewer than `self.fps` frames per second (module
        docstring, LIMITS)."""
        if total < self.temporal_patch_size:
            raise VideoError(where, "video_too_short",
                             f"video has {total} frame{'s' if total != 1 else ''}; the model "
                             f"reads video {self.temporal_patch_size} frames at a time, so "
                             "send one frame as an image_url part instead")
        if int(total / fps * self.fps) > self.max_frames:
            raise VideoError(where, "video_too_long",
                             f"video is {total / fps:.1f} s; at {self.fps:g} frames per second "
                             f"the model reads at most {self.max_frames} frames, so the limit "
                             f"is {self.max_seconds:g} s; send a shorter clip")

    def sample(self, total: int, fps: float) -> np.ndarray:
        """The source frame indices the processor reads from `total` frames
        at `fps`: Qwen3VLVideoProcessor.sample_frames with the video's real
        frame rate, line for line."""
        n = int(total / fps * self.fps)
        n = min(max(n, self.min_frames), self.max_frames, total)
        return np.linspace(0, total - 1, n).round().astype(int)

    def plan(self, frames: int, height: int, width: int, *,
             where: str) -> tuple[tuple[int, int], tuple[int, int, int], int]:
        """(resized (h, w), grid_thw, tokens) for `frames` sampled frames of
        height x width: the video smart_resize, the frame count rounded up
        to whole temporal groups, and one token per merge block per group."""
        try:
            h, w = video_smart_resize(frames, height, width, self.temporal_patch_size,
                                      self.factor, self.min_pixels, self.max_pixels)
        except ValueError:
            if frames < self.temporal_patch_size:
                raise VideoError(where, "video_too_short",
                                 f"video gives {frames} frame(s); the model reads video "
                                 f"{self.temporal_patch_size} frames at a time") from None
            raise VideoError(where, "video_aspect_ratio",
                             f"video is {width}x{height}; the long side may be at most 200 "
                             "times the short side") from None
        groups = -(-frames // self.temporal_patch_size)
        gh, gw = h // self.patch_size, w // self.patch_size
        return (h, w), (groups, gh, gw), groups * gh * gw // (self.merge_size ** 2)

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

    def pixel_values(self, frames: np.ndarray) -> np.ndarray:
        """[T, C, h, w] uint8 frames, already resized -> pixel_values
        [groups*gh*gw, C*temporal*P*P] float32: the last frame repeated up
        to whole temporal groups, normalized, and cut into the reference's
        rows (group, merge blocks, the 2x2 inside one), each row channel,
        then frame, then the patch."""
        T, C, h, w = frames.shape
        tp, m, P = self.temporal_patch_size, self.merge_size, self.patch_size
        if pad := -T % tp:
            frames = np.concatenate([frames, np.repeat(frames[-1:], pad, axis=0)])
            T += pad
        G, gh, gw = T // tp, h // P, w // P
        vals = self._table()[np.arange(C)[None, :, None, None], frames]  # float32 [T, C, h, w]
        a = vals.reshape(G, tp, C, gh // m, m, P, gw // m, m, P)
        a = a.transpose(0, 3, 6, 4, 7, 2, 1, 5, 8)
        return np.ascontiguousarray(a.reshape(G * gh * gw, C * tp * P * P))

    def group_times(self, times) -> list[float]:
        """One time per temporal group from the sampled frames' times:
        Qwen3VLProcessor._calculate_timestamps, the last time repeated up
        to whole groups and each group's first and last averaged."""
        t = [float(x) for x in times]
        tp = self.temporal_patch_size
        if len(t) % tp:
            t.extend(t[-1] for _ in range(tp - len(t) % tp))
        return [(t[i] + t[i + tp - 1]) / 2 for i in range(0, len(t), tp)]


def timestamp_text(seconds: float) -> str:
    """A temporal group's text in the prompt, Qwen3VLProcessor's format."""
    return f"<{seconds:.1f} seconds>"


# ----------------------------------------------------- 2. decode ---------

@dataclass(frozen=True)
class Source:
    """What a container says about its video stream, before any frame is
    decoded: the frame count, the frame rate, and the frame size."""

    frames: int
    fps: float
    height: int
    width: int


def _open(av, video: EncodedVideo, where: str):
    try:
        container = av.open(BytesIO(video.data), format=_DEMUXER[video.format])
    except Exception as e:  # noqa: BLE001 — untrusted bytes: any demuxer failure is a 400
        raise VideoError(where, "video_decode",
                         f"could not read the {video.format.upper()} container "
                         f"({type(e).__name__}: {e})") from None
    if not container.streams.video:
        container.close()
        raise VideoError(where, "video_decode", "the container has no video stream")
    return container, container.streams.video[0]


def probe(video: EncodedVideo, *, where: str) -> Source:
    """The video stream's frame count, rate and size, from the container:
    `stream.frames` (what read_video_pyav reads), or, for a container that
    lists none (WebM and MKV carry no count), its demuxed packets, counted
    without decoding any. The rate is `average_rate` (read_video_pyav's),
    else FFmpeg's guess; a stream with neither is refused rather than
    given transformers' 24 fps fallback. A frame over
    vision.MAX_SOURCE_PIXELS is refused here, before anything is
    decoded."""
    av = _av(where)
    container, stream = _open(av, video, where)
    try:
        w, h = int(stream.codec_context.width or 0), int(stream.codec_context.height or 0)
        if w < 1 or h < 1:
            raise VideoError(where, "video_decode", f"the video stream's frames are {w}x{h}")
        if w * h > vision.MAX_SOURCE_PIXELS:
            raise VideoError(where, "video_too_many_pixels",
                             f"video frames are {w}x{h} = {w * h:,} pixels; the limit is "
                             f"{vision.MAX_SOURCE_PIXELS:,}")
        rate = stream.average_rate or stream.guessed_rate
        fps = float(rate) if rate else 0.0
        if not fps > 0:
            raise VideoError(where, "video_decode", "the video stream has no frame rate")
        frames = int(stream.frames or 0)
        if frames <= 0:
            try:
                frames = sum(1 for p in container.demux(stream) if p.size)
            except Exception as e:  # noqa: BLE001 — untrusted bytes
                raise VideoError(where, "video_decode",
                                 f"could not read the video's packets "
                                 f"({type(e).__name__}: {e})") from None
        if frames <= 0:
            raise VideoError(where, "video_decode", "the video stream has no frames")
        return Source(frames, fps, h, w)
    finally:
        container.close()


class Shortfall(Exception):
    """read_frames reached the end of the stream before its last index:
    the container listed more frames than decode (an MP4 edit list that
    trims frames does this). `decoded` is how many there were."""

    def __init__(self, decoded: int):
        self.decoded = decoded
        super().__init__(f"the stream decoded to {decoded} frames")


def read_frames(video: EncodedVideo, indices, size: tuple[int, int], source: Source, *,
                where: str) -> tuple[np.ndarray, list[float]]:
    """Decode the frames at `indices` (positions in the decoded stream, in
    presentation order, as read_video_pyav counts them) -> ([n, 3, h, w]
    uint8 resized to `size`, each one's time in seconds). A frame is
    converted to RGB and resized the moment it is decoded, and decoding
    stops after the last index. A frame's time is its presentation time
    relative to the stream's first frame, or index / fps for a frame
    without one (module docstring, TIMESTAMPS). Raises Shortfall when the
    stream ends before the last index."""
    av = _av(where)
    h, w = size
    want: dict[int, list[int]] = {}
    for k, i in enumerate(int(i) for i in indices):
        want.setdefault(i, []).append(k)
    last = max(want)
    out = np.empty((len(indices), 3, h, w), dtype=np.uint8)
    times = [0.0] * len(indices)
    container, stream = _open(av, video, where)
    stream.thread_type = "AUTO"
    seen, t0 = 0, None
    try:
        for i, frame in enumerate(container.decode(stream)):
            seen = i + 1
            if i == 0:
                t0 = frame.time
            if i in want:
                if (frame.height, frame.width) != (source.height, source.width):
                    raise VideoError(where, "video_decode",
                                     f"frame {i} is {frame.width}x{frame.height}; the stream's "
                                     f"frames are {source.width}x{source.height}")
                rgb = frame.to_ndarray(format="rgb24")
                small = bicubic_resize(np.ascontiguousarray(rgb.transpose(2, 0, 1)), h, w)
                t = (frame.time - t0 if frame.time is not None and t0 is not None
                     else i / source.fps)
                for k in want[i]:
                    out[k] = small
                    times[k] = t
            if i >= last:
                break
    except VideoError:
        raise
    except Exception as e:  # noqa: BLE001 — untrusted bytes: any decoder failure is a 400
        raise VideoError(where, "video_decode",
                         f"could not decode the {video.format.upper()} video at frame {seen} "
                         f"({type(e).__name__}: {e})") from None
    finally:
        container.close()
    if seen <= last:
        raise Shortfall(seen)
    return out, times


# ------------------------------------------------------ the results -------

def video_digest(pixel_values: np.ndarray, grid_thw) -> str:
    """sha256 over b"video", the pixel_values bytes, then grid_thw as three
    little-endian int64: vision.image_digest's bytes behind a tag, so a
    video and an image never share a digest."""
    h = hashlib.sha256(b"video")
    h.update(memoryview(np.ascontiguousarray(pixel_values, dtype="<f4")).cast("B"))
    h.update(np.asarray(grid_thw, dtype="<i8").tobytes())
    return h.hexdigest()


@dataclass(frozen=True, eq=False)
class PreparedVideo:
    """One video, ready for the vision tower and the prompt: what
    `VideoInput.prepare` returns and what GenerationRequest.videos holds.

    `pixel_values` is float32 [groups*gh*gw, 3*2*16*16 = 1536] for Qwen3.5,
    what `Qwen3_5Model.get_video_features(pixel_values_videos,
    video_grid_thw)` takes. `tokens` is the tower's rows (every group's run
    together), `timestamps` one time per temporal group and
    `timestamp_ids` each one's text (timestamp_text) as the engine's
    tokenizer encodes it. `prompt_tokens` is what the video takes in the
    prompt (the runs, the timestamps, two markers per group) and
    `rendered_tokens` what the template rendered it as. Like a
    PreparedImage it compares equal by (architecture, digest), keys a
    prefix by `prefix_key`, and drops its pixels with `release`."""

    architecture: str
    source_format: str
    source_size: tuple[int, int]     # (height, width) of the stream's frames
    size: tuple[int, int]            # (height, width) every frame is resized to
    fps: float                       # the source's frame rate
    source_frames: int               # frames in the source
    frame_indices: tuple[int, ...]   # the source frames read, in order
    grid_thw: tuple[int, int, int]   # (temporal groups, h, w) in ViT patches
    tokens: int                      # video placeholder tokens: the tower's rows
    timestamps: tuple[float, ...]    # seconds, one per temporal group
    timestamp_ids: tuple[tuple[int, ...], ...]
    prompt_tokens: int
    rendered_tokens: int
    pixel_values: np.ndarray
    digest: str

    @property
    def groups(self) -> int:
        return self.grid_thw[0]

    @property
    def group_tokens(self) -> int:
        """Tokens in one temporal group's run."""
        return self.tokens // self.grid_thw[0]

    @property
    def expansion(self) -> int:
        """What expanding the template's placeholder adds to a rendered
        prompt's token count."""
        return self.prompt_tokens - self.rendered_tokens

    @property
    def released(self) -> bool:
        return self.pixel_values is None

    def release(self) -> None:
        """Drop pixel_values once the tower has read them (or never will):
        vision.PreparedImage.release's rule. At Qwen3.8-27B's budget one
        video is up to 302 MB of float32."""
        object.__setattr__(self, "pixel_values", None)

    @property
    def prefix_key(self) -> int:
        """The id every video token of this video stands as in a prefix
        cache key: vision.PreparedImage.prefix_key's rule over the video's
        digest."""
        return -1 - (int(self.digest[:16], 16) >> 1)

    def prompt_text(self, video_token: str = "<|video_pad|>",
                    start: str = "<|vision_start|>", end: str = "<|vision_end|>") -> str:
        """The text the template's placeholder becomes: per temporal group,
        its timestamp, then `start`, its run of `video_token`, `end`
        (Qwen3VLProcessor.replace_video_token's string)."""
        run = video_token * self.group_tokens
        return "".join(timestamp_text(t) + start + run + end for t in self.timestamps)

    def __eq__(self, other) -> bool:
        if not isinstance(other, PreparedVideo):
            return NotImplemented
        return (self.architecture, self.digest) == (other.architecture, other.digest)

    def __hash__(self) -> int:
        return hash((self.architecture, self.digest))

    def __repr__(self) -> str:
        return (f"PreparedVideo({self.architecture}, {self.source_format} "
                f"{self.source_size[1]}x{self.source_size[0]} @ {self.fps:g} fps, "
                f"{len(self.frame_indices)} of {self.source_frames} frames -> "
                f"{self.size[1]}x{self.size[0]}, grid={self.grid_thw}, tokens={self.tokens}, "
                f"digest={self.digest[:12]}…)")


# ------------------------------------------------------- the engine side --

@dataclass(frozen=True)
class VideoInput:
    """An engine's video input (vision.Vision.video): one architecture's
    video preprocessor and the engine tokenizer's `encode` (text -> ids,
    no special tokens added), which the timestamps are tokenized with. The
    dialects call `prepare`."""

    preprocessor: QwenVideoPreprocessor
    encode: Callable[[str], list[int]]

    @property
    def architecture(self) -> str:
        return self.preprocessor.architecture

    def prepare(self, video: EncodedVideo, *, where: str = "video") -> PreparedVideo:
        """Probe, sample, decode, resize, normalize, patchify, time and
        digest: the one video the tower and the prompt will read. A stream
        that decodes to fewer frames than its container lists is sampled
        again over the frames it really has (one more decode pass), and
        refused if it falls short again."""
        pre = self.preprocessor
        src = probe(video, where=where)
        for attempt in range(2):
            pre.check_length(src.frames, src.fps, where=where)
            idx = pre.sample(src.frames, src.fps)
            size, grid, tokens = pre.plan(len(idx), src.height, src.width, where=where)
            try:
                frames, times = read_frames(video, idx, size, src, where=where)
                break
            except Shortfall as e:
                if attempt or e.decoded < 1:
                    raise VideoError(where, "video_decode",
                                     f"the video decoded to {e.decoded} frames; its container "
                                     f"lists {src.frames}") from None
                src = Source(e.decoded, src.fps, src.height, src.width)
        pv = pre.pixel_values(frames)
        pv.flags.writeable = False
        stamps = tuple(pre.group_times(times))
        ids = tuple(tuple(int(i) for i in self.encode(timestamp_text(t))) for t in stamps)
        prompt = sum(len(i) for i in ids) + grid[0] * pre.wrap + tokens
        return PreparedVideo(pre.architecture, video.format, (src.height, src.width), size,
                             src.fps, src.frames, tuple(int(i) for i in idx), grid, tokens,
                             stamps, ids, prompt, pre.rendered, pv, video_digest(pv, grid))

    def block(self) -> dict:
        """What /v1/models announces as `capabilities.videoInput`."""
        pre = self.preprocessor
        return {"fps": pre.fps, "maxFrames": pre.max_frames, "maxSeconds": pre.max_seconds,
                "maxPixels": pre.max_pixels, "maxBytes": MAX_VIDEO_BYTES,
                "formats": list(FORMATS)}


# The registry: config.json `model_type` -> factory(video processor config
# dict) -> preprocessor. An architecture absent here has no video input.
_REGISTRY: dict[str, Callable[[dict], QwenVideoPreprocessor]] = {
    "qwen3_5": lambda cfg: QwenVideoPreprocessor.from_config("qwen3_5", cfg),
}


def read_video_processor_config(checkpoint_dir: str) -> dict | None:
    """The video-processor dict transformers' from_pretrained would load
    from `checkpoint_dir`: processor_config.json's nested
    `video_processor` when it has one, else video_preprocessor_config.json,
    else None. A pack embeds both (codec/pack.py's `*.json`)."""
    nested = os.path.join(checkpoint_dir, "processor_config.json")
    if os.path.isfile(nested):
        with open(nested) as f:
            cfg = json.load(f)
        if isinstance(cfg.get("video_processor"), dict):
            return cfg["video_processor"]
    flat = os.path.join(checkpoint_dir, "video_preprocessor_config.json")
    if os.path.isfile(flat):
        with open(flat) as f:
            return json.load(f)
    return None


def load(model_type: str, checkpoint_dir: str,
         encode: Callable[[str], list[int]] | None) -> tuple[VideoInput | None, str | None]:
    """(the VideoInput for a checkpoint, None), or (None, why not): no
    video preprocessor for the architecture, no video processor config in
    the checkpoint, no tokenizer to write the timestamps with, or PyAV not
    installed. A config the preprocessor cannot honor raises ValueError, a
    load-time error like vision.preprocessor_for's."""
    factory = _REGISTRY.get(model_type)
    if factory is None:
        return None, f"this server has no video input for {model_type}"
    cfg = read_video_processor_config(checkpoint_dir)
    if cfg is None:
        return None, "the checkpoint carries no video processor config"
    pre = factory(cfg)
    if encode is None:
        return None, "no tokenizer to write the video timestamps with"
    if not available():
        return None, "PyAV is not installed; " + INSTALL_HINT
    return VideoInput(pre, encode), None
