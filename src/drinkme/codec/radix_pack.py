"""The radix codec as a PACK TENSOR — THE bf16 codec of the pack
(docs/pack-format.md): the research encoder's three arrays (codec/radix.py's
RadixPack — palette, block directory, streams) carried inside this repo's
pack container, one npz per tensor, hashed and manifested like every tensor.

numpy-pure like pack.py: no torch at import, so packing and its bit-exact
verification run anywhere.

WHAT A RADIX TENSOR'S NPZ CARRIES:

  arrays   rx_palette  uint8  [covered]      exponents in code order, UNPADDED
                                             (radix.RadixPack.palette)
           rx_offsets  uint32 [R*NBK + 1]    word directory, final sentinel
           rx_data     uint32 [words]        the concatenated block streams
  scalars  R, C, bpw   as every codec tensor
           codec       "radix"   — the per-tensor discriminator
                                   (pack._arrays_for keys on it; registry
                                   .of_pack maps it to ("radix", 0))
           profile     "sip" | "balanced" | "gulp"
           widths      the exponent tier widths: [3, 8] / [2, 3, 8] / [2, 2, 4, 8]
           block_size  1024

A pack's profile is one for the whole pack and rides in meta.json
(`profile`, `profileWidths`, `blockSize`) — the profile's name ("sip" /
"gulp") is the pack's name; the loader dispatches per tensor off the
`codec` scalar all the same.

THE RAW FALLBACK: radix.pack_array stores a tensor that would EXPAND as a
raw copy of the bits. Here that tensor is stored as a RAW pack tensor —
`codec: "raw"`, one array `raw_bits` (uint16 [R, C], the bf16 bit patterns
verbatim), 16 bpw — so a pack is never larger than its source anywhere and
the loader serves it as a plain bf16 Linear (swap.RawLinear). meta.json
counts them (rawFallbackTensorCount). No real checkpoint has produced one
yet (0 of 253 on Qwen3-8B at every profile); the toy in
tests/test_radix_pack.py does.

THE PROFILES: sip (3,8) is the default and the speed profile; gulp
(2,2,4,8) is the fit profile (the 27B on a 48 GB card); balanced (2,3,8) measured between the two on bytes and closer to gulp
on speed and is not offered on the CLI —
DRINKME_COMPRESSION_PROFILE=balanced selects it. resolve_compression_profile is the one place
the choice is made.

VERIFICATION: the encoder round-trips on the CPU before anything is
returned (radix.pack_array does it itself; the native path decodes back
with the native decoder and compares — radix_native.encode). The file
hash then covers the files, as for every other tensor.
"""

from __future__ import annotations

import os
import sys

import numpy as np

from . import radix as _radix
from . import radix_native as _native

RADIX = "radix"  # the per-tensor `codec` scalar, and registry.RADIX
RAW = "raw"      # the fallback's `codec` scalar, and registry.RAW

# the profiles (radix.PROFILES' bf16 widths); PUBLIC_PROFILES is what
# `drinkme pack` offers as flags (--sip / --gulp), DEFAULT_PROFILE what it
# packs unasked
PROFILES = ("sip", "balanced", "gulp")
PUBLIC_PROFILES = ("sip", "gulp")
DEFAULT_PROFILE = "sip"
RADIX_BLOCK = _radix.DEFAULT_BLOCK  # 1024 weights per block, the research default

_ARRAYS_RADIX = ("rx_palette", "rx_offsets", "rx_data")
_SCALARS_RADIX = ("R", "C", "bpw", "codec", "profile", "widths", "block_size")
_ARRAYS_RAW = ("raw_bits",)
_SCALARS_RAW = ("R", "C", "bpw", "codec")


def is_radix_pack(p: dict) -> bool:
    return p.get("codec") == RADIX


def is_raw_pack(p: dict) -> bool:
    return p.get("codec") == RAW


def resolve_compression_profile(explicit: str | None = None) -> str:
    """The profile a pack is written at: the explicit one (`--sip` /
    `--gulp`), else DRINKME_COMPRESSION_PROFILE (the hidden door for balanced),
    else sip. Unknown names are refused here, by name."""
    name = explicit or os.environ.get("DRINKME_COMPRESSION_PROFILE", "").strip() or DEFAULT_PROFILE
    if name not in PROFILES:
        raise ValueError(f"unknown profile {name!r}; one of {PROFILES}")
    return name


def widths_of(compression_profile: str) -> tuple[int, ...]:
    return tuple(_radix.PROFILES[compression_profile]["bf16"])


def _encoder(lead_in: str = "drinkme") -> str:
    """DRINKME_RADIX_ENCODER = native | numpy; unset (or "native") REFUSES
    (raises exitcodes.CantRunHere with radix_native.refusal_message —
    fixable by installing a compiler or retrying, not by changing the
    command) rather than silently drop to numpy when no compiler built the
    native encoder — packing does not fall back. The numpy encoder is
    reached only by an explicit DRINKME_RADIX_ENCODER=numpy. Both write the
    same bytes (pinned in tests/test_radix_pack.py); native is the one that
    packs an 8B in minutes rather than hours. `lead_in` is the caller's
    `drinkme <verb>` prefix, threaded to refusal_message."""
    raw = os.environ.get("DRINKME_RADIX_ENCODER", "").strip().lower()
    if raw not in ("", "native", "numpy"):
        raise ValueError(f"DRINKME_RADIX_ENCODER={raw!r}: expected native or numpy")
    if raw == "numpy":
        return "numpy"
    if not _native.available():
        from .. import exitcodes

        raise exitcodes.CantRunHere(_native.refusal_message(lead_in=lead_in))
    return "native"


def _decoder() -> str:
    """DRINKME_RADIX_ENCODER = native | numpy for decode_back_radix — DECODE
    IS OUT OF SCOPE for the encoder refusal (a served pack's CPU reference
    path, swap.py's dense/CPU forward, and the MLX engine's reference decode
    all call decode_back_radix with no compiler required): unset picks
    native when available, else numpy, SILENTLY — the old _encoder()
    behavior, kept here so serving an existing pack never needs a
    compiler."""
    raw = os.environ.get("DRINKME_RADIX_ENCODER", "").strip().lower()
    if raw in ("native", "numpy"):
        return raw
    if raw:
        raise ValueError(f"DRINKME_RADIX_ENCODER={raw!r}: expected native or numpy")
    return "native" if _native.available() else "numpy"


def encoder_status() -> tuple[str, _native.Diagnosis | None]:
    """What _encoder() would resolve to RIGHT NOW, without raising or
    printing — `drinkme check`'s prediction, never an action:
      ("native", an ok Diagnosis)     the default path packing will take
      ("numpy", None)                 DRINKME_RADIX_ENCODER=numpy, explicit
      ("refuse", a failing Diagnosis) what _encoder() would raise over
    An explicit numpy opt-out is never a problem to report, whatever this
    machine's compiler situation is."""
    raw = os.environ.get("DRINKME_RADIX_ENCODER", "").strip().lower()
    if raw == "numpy":
        return "numpy", None
    diag = _native.diagnose()
    return ("native", diag) if diag.ok else ("refuse", diag)


def refuse_unless_encoder_available(lead_in: str = "drinkme") -> None:
    """The front door: `drinkme pack`, bench's in-memory pack and serve's
    auto-pack call this before touching the Hub or a checkpoint — the same
    decision _encoder() makes per tensor, made once and early, so a
    machine with no compiler (or a compiler that fails to build the
    encoder) refuses before paying for a download, not partway through
    one. Also prints the numpy notice ONCE, here, rather than once per
    tensor from inside _encoder(). `lead_in` is the caller's `drinkme <verb>`
    prefix (e.g. "drinkme pack"), threaded to _encoder/refusal_message."""
    if _encoder(lead_in) == "numpy":
        print("[drinkme] DRINKME_RADIX_ENCODER=numpy: packing with the slow pure-Python "
              "radix encoder (no native C++ encoder is in use) — much slower than native, "
              "hours for an 8B model", file=sys.stderr)


def _would_expand(palette, offsets, data, widths, nbytes: int) -> bool:
    """radix.pack_array's raw criterion, verbatim: the stored bytes OR the
    device bytes (palette padded to whole words per tier) reaching the raw
    plane's size means no compression."""
    stored = palette.nbytes + offsets.nbytes + data.nbytes
    device = data.nbytes + offsets.nbytes + _native.padded_table_bytes(widths)
    return max(stored, device) >= nbytes


def pack_array_radix(U: np.ndarray, compression_profile: str, encoder: str | None = None) -> dict | None:
    """uint16 [R, C] bf16 bit patterns -> the radix tensor dict, or None when
    radix would not compress this tensor (the caller stores it raw instead).
    The CPU round trip runs inside whichever encoder is used."""
    U = np.ascontiguousarray(U, dtype=np.uint16)
    if U.ndim != 2 or min(U.shape) < 1:
        raise ValueError("pack_array_radix: need a nonempty 2D uint16 array")
    widths = widths_of(compression_profile)
    enc = encoder or _encoder()
    if enc == "native":
        palette, offsets, data = _native.encode(U, widths)
        if _would_expand(palette, offsets, data, widths, U.nbytes):
            return None
    else:
        rp = _radix.pack_array(U, compression_profile=compression_profile, block_size=RADIX_BLOCK)
        if rp.raw:
            return None
        palette, offsets, data = rp.palette, rp.offsets, rp.data
    R, C = U.shape
    stored = palette.nbytes + offsets.nbytes + data.nbytes
    return {
        "rx_palette": np.ascontiguousarray(palette, dtype=np.uint8),
        "rx_offsets": np.ascontiguousarray(offsets, dtype=np.uint32),
        "rx_data": np.ascontiguousarray(data, dtype=np.uint32),
        "R": int(R), "C": int(C),
        "bpw": 8.0 * stored / (R * C),  # the stored payload; device adds gulp's schedule
        "codec": RADIX,
        "profile": compression_profile,
        "widths": list(widths),
        "block_size": RADIX_BLOCK,
    }


def raw_dict(U: np.ndarray) -> dict:
    """The raw fallback tensor: the bf16 bit patterns verbatim."""
    U = np.ascontiguousarray(U, dtype=np.uint16)
    if U.ndim != 2 or min(U.shape) < 1:
        raise ValueError("raw_dict: need a nonempty 2D uint16 array")
    R, C = U.shape
    return {"raw_bits": U, "R": int(R), "C": int(C), "bpw": 16.0, "codec": RAW}


def pack_weight_radix(w_bf16, compression_profile: str, name: str | None = None,
                      encoder: str | None = None) -> dict:
    """ONE tensor for `drinkme pack`: the radix dict at `compression_profile`, or — when
    radix would expand it — the raw dict (the fallback, so the pack is never
    larger than the source anywhere). Eligibility is the caller's
    (pack_model / eligible_linears); this only encodes — and encodes the
    bf16 bits it is handed, never a cast of them: a tensor of any other
    dtype is refused by name (the source-dtype contract, codec/pack.py
    SOURCE_DTYPE; pack._pack_one already said so with the checkpoint's
    line, this is the codec's own guard)."""
    import torch

    if w_bf16.dtype != torch.bfloat16:
        raise TypeError(f"{name or 'tensor'}: {w_bf16.dtype} is not bf16 — the codec packs bf16 "
                        "bits as released and never converts a source tensor")
    U = w_bf16.contiguous().view(torch.int16).numpy().view(np.uint16)
    p = pack_array_radix(U, compression_profile, encoder)
    if p is None:
        return raw_dict(U)
    return p


def _palette_u8(p: dict) -> np.ndarray:
    """The UNPADDED palette of a raw or runtime dict. A runtime dict's
    rx_palette is the padded per-tier word table (to_device_radix), from
    which the unpadded palette is not a prefix once there is more than one
    nonterminal tier — so the runtime dict keeps the original bytes under
    rx_palette_np (caught on the gulp toy: the CPU reference decoded
    wrong exponents while sip, one tier, happened to work)."""
    if "rx_palette_np" in p:
        return np.ascontiguousarray(p["rx_palette_np"], dtype=np.uint8)
    return _np(p["rx_palette"], np.uint8)


def _payload(p: dict) -> np.ndarray:
    """The block streams of a raw or runtime dict as uint32 — a runtime
    dict's rx_data carries swap.to_device_radix's zero pad after them, and
    NW says where the payload ends."""
    data = _np(p["rx_data"], np.uint32)
    return data[: int(p["NW"])] if "NW" in p else data


def as_radixpack(p: dict) -> _radix.RadixPack:
    """The encoder's dataclass (radix.RadixPack) over a radix tensor dict
    (raw or runtime), for radix.decode / radix.validate — the CPU oracle
    and the pack-time check."""
    palette, offsets, data = (_palette_u8(p), _np(p["rx_offsets"], np.uint32), _payload(p))
    return _radix.RadixPack((int(p["R"]), int(p["C"])), "bf16", int(p["block_size"]),
                            tuple(int(w) for w in p["widths"]), palette,
                            offsets.astype("<u4"), data.astype("<u4"))


def _np(a, dtype):
    """numpy view of a pack array that may be a torch tensor (a runtime dict)
    — torch has no uint32/uint16 numpy round trip for every dtype, so go via
    the same-width signed view where needed."""
    if isinstance(a, np.ndarray):
        return np.ascontiguousarray(a).view(dtype)
    import torch

    if a.dtype == torch.uint32:
        return a.view(torch.int32).cpu().numpy().view(np.uint32).view(dtype)
    if a.dtype in (torch.uint16, torch.bfloat16):
        return a.contiguous().view(torch.int16).cpu().numpy().view(np.uint16).view(dtype)
    return a.cpu().numpy().view(dtype)


def decode_back_radix(p: dict, encoder: str | None = None) -> np.ndarray:
    """The CPU oracle: a radix tensor dict -> the original uint16 [R, C]
    bits. Native when available (seconds on an 8B tensor), else the
    research numpy decoder (radix.decode, which also validates the
    directory and every stream length). Decode is out of scope for the
    encoder refusal (_decoder(), never _encoder()): serving an existing
    pack needs no compiler."""
    enc = encoder or _decoder()
    if enc == "native":
        return _native.decode(_palette_u8(p), _np(p["rx_offsets"], np.uint32), _payload(p),
                              int(p["R"]), int(p["C"]), tuple(int(w) for w in p["widths"]))
    return _radix.decode(as_radixpack(p))


def verify_pack_radix(p: dict, U: np.ndarray) -> None:
    """Round-trip gate: the stored streams decode to exactly U, and the
    research validator accepts the descriptors (palette, directory, widths)."""
    _radix.validate(as_radixpack(p))
    back = decode_back_radix(p)
    if back.shape != U.shape or not np.array_equal(back, U):
        n = int((back != U).sum()) if back.shape == U.shape else -1
        raise ValueError(f"radix pack does not reconstruct the source bits ({n} words differ)")


def decode_back_raw(p: dict) -> np.ndarray:
    """A raw tensor's bits, as uint16 [R, C] (a runtime dict holds them as a
    bf16 torch tensor)."""
    if "raw_bits" in p:
        return np.ascontiguousarray(_np(p["raw_bits"], np.uint16)).reshape(int(p["R"]), int(p["C"]))
    return _np(p["weight"], np.uint16).reshape(int(p["R"]), int(p["C"]))


def verify_pack_raw(p: dict, U: np.ndarray) -> None:
    back = decode_back_raw(p)
    if back.shape != U.shape or not np.array_equal(back, U):
        raise ValueError("raw pack tensor does not hold the source bits")


def decode_back(p: dict, encoder: str | None = None) -> np.ndarray:
    """The CPU oracle for any bf16 pack tensor dict (raw or runtime): the
    original uint16 [R, C] bits — radix through decode_back_radix, the raw
    fallback through its own bits."""
    if is_raw_pack(p):
        return decode_back_raw(p)
    if is_radix_pack(p):
        return decode_back_radix(p, encoder)
    raise ValueError("decode_back: not a radix or raw pack tensor")


def padded_palette_words(p: dict) -> np.ndarray:
    """The GPU lookup tables: for each nonterminal tier (every width but the
    last), its (1 << w) - 1 palette bytes copied into a zero-filled uint8
    buffer of ceil(n / 4) * 4 bytes, viewed as uint32 words; the tiers'
    words concatenated in order. radix_gpu._lookup reads a code's byte as
    `word[code // 4] >> (code % 4 * 8)` from the tier's table at offset
    sum(((1 << w) + 2) // 4) over the tiers before it — the padding is what
    makes every tier start on a word."""
    palette = _palette_u8(p)
    widths = [int(w) for w in p["widths"]]
    tables, base = [], 0
    for w in widths[:-1]:
        n = (1 << w) - 1
        t = np.zeros(((n + 3) // 4) * 4, dtype=np.uint8)
        t[:n] = palette[base:base + n]
        tables.append(t.view(np.uint32))
        base += n
    return np.concatenate(tables) if tables else np.empty(0, dtype=np.uint32)


def _count_escapes(data: np.ndarray, pos: np.ndarray, count: np.ndarray, w: int, B: int) -> np.ndarray:
    """For each block b: how many of the first count[b] little-endian w-bit
    fields of the stream starting at word pos[b] equal the escape code
    (1 << w) - 1. radix._get's field arithmetic, over a chunk of blocks at
    once; positions >= count[b] never count. A word past the payload reads
    as zero — the device pad radix_kernel_gpu._prepare_schedule reads
    there (swap.to_device_radix), so a block short of its data-dependent
    streams counts the same escapes on both."""
    W = (B * w + 31) // 32 + 1  # +1: a crossing field may touch one more word
    idx = pos[:, None] + np.arange(W)[None, :]
    valid = idx < len(data)
    words = np.where(valid, data[np.minimum(idx, len(data) - 1)], 0).astype(np.uint32)
    bit = np.arange(B, dtype=np.int64) * w
    wi, sh = bit // 32, (bit % 32).astype(np.uint32)
    vals = words[:, wi] >> sh[None, :]
    cross = (bit % 32 + w) > 32
    if cross.any():
        vals[:, cross] |= words[:, wi[cross] + 1] << (32 - sh[cross])[None, :]
    esc = (vals & np.uint32((1 << w) - 1)) == np.uint32((1 << w) - 1)
    live = np.arange(B)[None, :] < count[:, None]
    return (esc & live).sum(axis=1)


def schedule_np(p: dict) -> np.ndarray | None:
    """radix_kernel_gpu._prepare_schedule on the CPU: one uint16 per block
    whose byte (level-1) is the WORD LENGTH of nonterminal tier `level`'s
    stream for level >= 1 — what lets the scheduled decoder skip the
    per-tier escape-count reductions. None for a two-tier profile (sip),
    whose only nonterminal stream has a fixed length. Vectorized over
    chunks of blocks; pinned equal to the GPU kernel's output in
    bench/radix_gemv_bitpin.py."""
    widths = [int(w) for w in p["widths"]]
    L = len(widths)
    if L <= 2:
        return None
    R, C, B = int(p["R"]), int(p["C"]), int(p["block_size"])
    offsets = _np(p["rx_offsets"], np.uint32).astype(np.int64)
    data = _payload(p)
    nb = (C + B - 1) // B
    nblk = R * nb
    hdr = np.zeros(nblk, dtype=np.uint32)
    CH = 8192
    for s in range(0, nblk, CH):
        e = min(nblk, s + CH)
        bi = np.arange(s, e) % nb
        n = np.minimum(C - bi * B, B).astype(np.int64)
        pos = offsets[s:e] + (n * 8 + 31) // 32  # past the sign/mantissa literals (M+1 = 8 bits)
        count = n.copy()
        h = np.zeros(e - s, dtype=np.uint32)
        for level in range(L - 1):
            w = widths[level]
            words = (count * w + 31) // 32
            if level > 0:
                h |= words.astype(np.uint32) << np.uint32((level - 1) * 8)
            if level < L - 2:
                count = _count_escapes(data, pos, count, w, B)
            pos = pos + words
        hdr[s:e] = h
    return hdr.astype(np.uint16)
