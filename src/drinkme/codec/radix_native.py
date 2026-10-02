"""ctypes wrapper over radix_native.cpp — the C++ encoder,
JIT-compiled once with `c++ -O3` into $DRINKME_HOME/radix-native/<sha>.so.

Why it exists here: radix.pack_array is a per-block numpy loop and packs a
Qwen3-8B tensor in tens of seconds; 253 of them is hours. The native
encoder writes the SAME streams (same palette order — frequency, ties by
increasing exponent — same little-endian field packing, same per-block
layout; tests/test_radix_pack.py pins native == numpy byte for byte on a
toy tensor) in seconds. Every tensor it encodes is decoded back by the
native decoder and compared to the source bits before it is accepted
(radix_pack.pack_array_radix). None when no C++ compiler is available, or
when one is but fails to build the encoder — in which case
radix_pack._encoder refuses rather than silently drop to the numpy path
(the front door, docs/cli.md's DRINKME_RADIX_ENCODER row)."""

from __future__ import annotations

import ctypes as ct
import hashlib
import os
import shutil
import subprocess
from dataclasses import dataclass

import numpy as np

_SOURCE = os.path.join(os.path.dirname(__file__), "radix_native.cpp")
_LIB = None  # the loaded CDLL, or False once loading failed
_DIAGNOSIS = None  # the Diagnosis behind _LIB, set at the same two points

CANDIDATES = ("c++", "g++", "clang++")  # the compilers library() looks for, in order

INSTALL_LINES = (
    ("Debian/Ubuntu", "sudo apt install g++"),
    ("Fedora", "sudo dnf install gcc-c++"),
    ("Arch", "sudo pacman -S gcc"),
    ("macOS", "xcode-select --install"),
)

ESCAPE_HATCH = ("escape hatch: DRINKME_RADIX_ENCODER=numpy packs with the pure-Python "
               "encoder instead — much slower, hours for an 8B model")


@dataclass(frozen=True)
class Diagnosis:
    """Why the native encoder is, or is not, available — library()'s own
    reasoning, cached alongside _LIB so diagnose() never recompiles."""
    ok: bool
    compiler: str | None = None       # the path found on PATH
    version: str | None = None        # `<compiler> --version`'s first line, when ok
    stderr_tail: str | None = None    # a failed compile's last ~20 lines
    reason: str | None = None         # "no_compiler" | "compile_failed", when not ok


def _compiler_version(cxx: str) -> str | None:
    try:
        out = subprocess.run([cxx, "--version"], capture_output=True, text=True, timeout=10)
        lines = out.stdout.splitlines() or out.stderr.splitlines()
        return lines[0].strip() if lines else None
    except (OSError, subprocess.SubprocessError):
        return None


def _stderr_tail(raw: bytes, n: int = 20) -> str:
    lines = raw.decode(errors="replace").splitlines()
    return "\n".join(lines[-n:])


def refusal_message(diag: Diagnosis | None = None, lead_in: str = "drinkme") -> str:
    """The refusal `drinkme pack` / `drinkme bench` / `drinkme serve` print
    when the native encoder is unavailable and DRINKME_RADIX_ENCODER has not
    opted into numpy: case (a) no compiler on PATH, naming the three this
    looked for and how to install one; case (b) a compiler that FAILED to
    build the encoder, naming its path and the tail of its stderr. Either
    way, the escape hatch.

    `lead_in` is the caller's own prefix (`drinkme pack`, `drinkme bench`,
    `drinkme serve`, or `drinkme check`'s longer "this machine can't pack it
    yet") — one `drinkme <verb>: ` per the error-copy rule, applied
    once here rather than by each caller re-wrapping the string."""
    d = diag if diag is not None else diagnose()
    if d.ok:
        raise ValueError("refusal_message: the native encoder is available, nothing to refuse")
    if d.reason == "no_compiler":
        lines = "\n".join(f"  {os_name:<14} {cmd}" for os_name, cmd in INSTALL_LINES)
        return (f"{lead_in}: no C++ compiler on PATH (looked for {', '.join(CANDIDATES)}) — "
                "the native radix encoder needs one to pack, and packing refuses rather than "
                "silently drop to the numpy encoder (hours for an 8B model). Install one:\n"
                f"{lines}\n{ESCAPE_HATCH}")
    tail = d.stderr_tail or "(no stderr captured)"
    return (f"{lead_in}: the C++ compiler at {d.compiler} failed to build the native radix "
            f"encoder. Its stderr, last {len(tail.splitlines())} lines:\n{tail}\n{ESCAPE_HATCH}")


def _cache_dir() -> str:
    root = os.environ.get("DRINKME_HOME", os.path.expanduser("~/.cache/drinkme"))
    return os.path.join(root, "radix-native")


def source_sha256() -> str:
    with open(_SOURCE, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def library():
    """The compiled library, or None (no compiler / compile failed). Cached
    per process; the .so is keyed on the source digest so an edit recompiles.
    diagnose() explains a None the same moment this decides it."""
    global _LIB, _DIAGNOSIS
    if _LIB is not None:
        return _LIB or None
    cxx = next((p for p in (shutil.which(c) for c in CANDIDATES) if p), None)
    if cxx is None:
        _LIB = False
        _DIAGNOSIS = Diagnosis(False, reason="no_compiler")
        return None
    cache = _cache_dir()
    os.makedirs(cache, exist_ok=True)
    so = os.path.join(cache, source_sha256() + ".so")
    if not os.path.exists(so):
        tmp = f"{so}.{os.getpid()}.tmp"
        try:
            subprocess.run([cxx, "-O3", "-std=c++17", "-Wall", "-Wextra", "-Werror",
                            "-fPIC", "-shared", _SOURCE, "-o", tmp], check=True,
                           capture_output=True)
            os.replace(tmp, so)
        except subprocess.CalledProcessError as e:
            if os.path.exists(tmp):
                os.unlink(tmp)
            _LIB = False
            _DIAGNOSIS = Diagnosis(False, compiler=cxx, reason="compile_failed",
                                   stderr_tail=_stderr_tail(e.stderr or b""))
            return None
        except OSError as e:
            if os.path.exists(tmp):
                os.unlink(tmp)
            _LIB = False
            _DIAGNOSIS = Diagnosis(False, compiler=cxx, reason="compile_failed",
                                   stderr_tail=str(e))
            return None
    lib = ct.CDLL(so)
    fn = lib.radix_native
    fn.argtypes = [ct.c_uint32, ct.c_void_p, ct.c_uint64, ct.c_uint64,
                   ct.c_uint64, ct.c_uint64, ct.c_uint32, ct.c_uint32,
                   ct.c_void_p, ct.c_uint32, ct.c_void_p, ct.c_void_p,
                   ct.c_void_p, ct.c_uint64, ct.c_uint64, ct.c_uint64,
                   ct.c_void_p, ct.c_void_p, ct.c_uint64]
    fn.restype = ct.c_int
    _LIB = lib
    _DIAGNOSIS = Diagnosis(True, compiler=cxx, version=_compiler_version(cxx))
    return lib


def available() -> bool:
    return library() is not None


def diagnose() -> Diagnosis:
    """Why the native encoder is or isn't available, for `drinkme check` and
    the pack/bench/serve refusal — library()'s own reasoning, without
    forcing a caller that only wants the library to import dataclasses."""
    library()
    return _DIAGNOSIS


def _call(op: int, value, R: int, C: int, widths, tables, offsets, data=None,
          first: int = 0, rows: int = 0) -> int:
    lib = library()
    if lib is None:
        raise RuntimeError("radix native encoder unavailable (no C++ compiler)")
    w = np.ascontiguousarray(np.asarray(widths, dtype=np.uint32))
    err = ct.create_string_buffer(1024)
    res = ct.c_uint64()
    ptr = lambda a: None if a is None else a.ctypes.data  # noqa: E731
    # row_group = R, block_group = nb: ONE palette region per tensor (the
    # per-tensor palette radix.pack_array writes)
    nb = (C + 1023) // 1024
    status = lib.radix_native(op, ptr(value), R, C, R, nb, 7, 8, ptr(w), len(w),
                              ptr(tables), ptr(offsets), ptr(data),
                              0 if data is None else data.size, first, rows,
                              ct.byref(res), err, len(err))
    if status:
        raise ValueError("native radix: " + err.value.decode())
    return res.value


def padded_table_bytes(widths) -> int:
    """The GPU palette layout's size in bytes: each nonterminal tier's
    (1 << w) - 1 entries padded to whole uint32 words, i.e. 4 * (((1 << w)
    + 2) // 4) per tier — radix_pack.padded_palette_words's layout, and
    native's `stride`."""
    return sum(4 * (((1 << w) + 2) // 4) for w in widths[:-1])


def encode(U: np.ndarray, widths) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """uint16 [R, C] bf16 bits -> (palette uint8 UNPADDED, offsets uint32,
    data uint32) — radix.RadixPack's three arrays, 1024-weight blocks.
    Raises if the native round trip (encode -> decode -> compare) fails."""
    U = np.ascontiguousarray(U, dtype=np.uint16)
    R, C = U.shape
    nb = (C + 1023) // 1024
    tables = np.zeros(padded_table_bytes(widths), dtype=np.uint8)
    offsets = np.zeros(R * nb + 1, dtype=np.uint32)
    words = _call(0, U, R, C, widths, tables, offsets)
    data = np.zeros(words, dtype=np.uint32)
    _call(1, U, R, C, widths, tables, offsets, data)
    back = np.empty_like(U)
    _call(2, back, R, C, widths, tables, offsets, data, 0, R)
    if not np.array_equal(back, U):
        raise ValueError("native radix: CPU round trip differs from the source bits")
    # padded per-tier tables -> the unpadded palette radix.pack_array stores
    parts, at = [], 0
    for w in widths[:-1]:
        n = (1 << w) - 1
        parts.append(tables[at:at + n])
        at += ((n + 3) // 4) * 4
    return np.concatenate(parts).astype(np.uint8), offsets, data


def decode(palette: np.ndarray, offsets: np.ndarray, data: np.ndarray, R: int, C: int,
           widths, workers: int = 1) -> np.ndarray:
    """The native decoder over RadixPack-shaped arrays -> uint16 [R, C].
    The library reads the directory by the block count it derives from
    (R, C) and the palette by the widths, so their sizes are checked here
    (the library sees pointers, not lengths); every stream length is the
    library's own check, a clean error, never a read past `data`.

    `workers` > 1 decodes row ranges on that many threads (every block
    decodes independently, and ctypes releases the GIL for the call);
    the output is the same array."""
    nb = (C + 1023) // 1024
    if offsets.ndim != 1 or offsets.size != R * nb + 1:
        raise ValueError("native radix: invalid radix block offsets")
    if palette.ndim != 1 or palette.size != sum((1 << w) - 1 for w in widths[:-1]):
        raise ValueError("native radix: invalid radix palette entries")
    tables = np.zeros(padded_table_bytes(widths), dtype=np.uint8)
    at = base = 0
    for w in widths[:-1]:
        n = (1 << w) - 1
        tables[at:at + n] = palette[base:base + n]
        at += ((n + 3) // 4) * 4
        base += n
    out = np.empty((R, C), dtype=np.uint16)
    offsets = np.ascontiguousarray(offsets, dtype=np.uint32)
    data = np.ascontiguousarray(data, dtype=np.uint32)
    if workers <= 1 or R < 2 * workers:
        _call(2, out, R, C, widths, tables, offsets, data, 0, R)
        return out
    from concurrent.futures import ThreadPoolExecutor

    # the library writes row `first` at the pointer it is handed, so each
    # range gets the view that starts at its own first row
    step = -(-R // (4 * workers))
    ranges = [(a, min(step, R - a)) for a in range(0, R, step)]
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for _ in ex.map(lambda r: _call(2, out[r[0]:], R, C, widths, tables, offsets, data,
                                        r[0], r[1]), ranges):
            pass
    return out
