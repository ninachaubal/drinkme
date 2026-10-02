"""Pack: a BF16 checkpoint's eligible Linears -> the radix exponent-tier
codec (codec/radix_pack.py), in this repo's container — pack format 1.

CPU-side and numpy-pure — no torch/triton at import, so packing (and its
bit-exact verification) runs anywhere, and the disk format is platform-neutral
(the torch runtime on CUDA and ROCm and the mlx runtime load the same pack —
codec/registry.py).

A PACK holds two kinds of tensor, each one npz, discriminated by its
scalars blob (docs/pack-format.md):

  codec: "radix"   the exponent-tier codec at one profile for the whole
                   pack (rx_palette / rx_offsets / rx_data + profile,
                   widths, block_size) — every eligible bf16 Linear
  codec: "raw"     THE FALLBACK: the bf16 bits verbatim (raw_bits), for a
                   tensor radix would EXPAND (radix_pack.pack_weight_radix's
                   own criterion). Counted in meta.json as
                   rawFallbackTensorCount; never seen on a real checkpoint
                   yet (0 of 253 on Qwen3-8B at every profile), kept so the
                   pack is never larger than the source anywhere

There is ONE bf16 codec and ONE format. An FP8 checkpoint (any float8 plane
in its shards, or an fp8 quantization_config) is refused by name at every
door — refusal_for_checkpoint below, the one line pack, check, serve
and bench all print. A pack that declares any other
format version (or none) is refused at load with one line naming the
version seen and the re-pack command (_check_format_version); there is no
other reader and no migration.

Every pack is verified by full decode == source, bit-for-bit, before it is
returned or written. A pack that does not round-trip does not exist.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil

import numpy as np

from . import identity as _identity
from . import radix as _radix
from . import radix_pack as _rx

FORMAT_VERSION = 1  # radix tensors + raw fallbacks, one codec
SUPPORTED_FORMAT_VERSIONS = (FORMAT_VERSION,)


def weighted_bpw(bpws, numels) -> float:
    """Bits per weight over a POPULATION of packed tensors, weight-weighted:
    total encoded payload bits (each tensor's bpw x its R*C — bpw is
    bits/n by construction in pack_array/_blockify) over
    total weights. The plain mean of per-tensor bpw (meanBpw, kept) gives a
    2048x1024 projection and a 151936x4096 head equal say; this gives the
    head its weight (1000 weights at 12 bpw + 100 at 16 is 14 by tensor,
    12.364 by weight). The population is the
    caller's: pack_model's is the trunk's packed tensors only — the raw
    remainder and the MTP head are NOT in it."""
    bpws, numels = list(bpws), list(numels)
    if len(bpws) != len(numels) or not bpws:
        raise ValueError("weighted_bpw: one bpw per numel, at least one tensor")
    bits = sum(float(b) * int(n) for b, n in zip(bpws, numels))
    return bits / sum(int(n) for n in numels)


def bpw_of_pack_dir(path: str) -> tuple[float, float]:
    """(mean over tensors, weight-weighted) bits per weight of a pack dir's
    packed population, read off each tensor's own scalars blob (bpw, R, C)
    — the loaded artifact's figures, not meta.json's copy of them. Opens
    each npz for its `scalars` member only."""
    with open(os.path.join(path, "meta.json")) as f:
        meta = json.load(f)
    bpws, numels = [], []
    for name, fn in meta["tensors"].items():
        with np.load(_tensor_file(path, name, fn)) as z:
            sc = json.loads(str(z["scalars"][0]))
        bpws.append(float(sc["bpw"]))
        numels.append(int(sc["R"]) * int(sc["C"]))
    return float(np.mean(bpws)), weighted_bpw(bpws, numels)


# The meta.json fields a bench record's raw.pack carries when the compressed
# arm loaded a pack from disk (arms.run_arms with --pack-dir, every
# arms_mlx.run_arms); the record's compression block publishes two of them
# as packedBytes / residentBytes (lexicons/wtf.petrichor.drinkme.measurement.json).
# `vision`, when the pack has the block, says the served tree's tower is
# already inside those totals: the lexicon has no typed field for it yet,
# so this rides in raw, which
# the lexicon declares `unknown` — a reader can tell a vision-capable
# record's residentBytes includes the tower without reopening the pack.
PACK_RECORD_KEYS = ("formatVersion", "profile", "profileWidths", "dtype", "tensorCount",
                    "rawTensorCount", "rawResidentBytes", "residentBytes", "packedBytes",
                    "meanBpw", "weightedBpw", "radixTensorCount", "radixBytes",
                    "rawFallbackTensorCount", "rawFallbackBytes", "vision")


def pack_record_fields(meta: dict) -> dict:
    """The subset of a pack's meta.json a bench record carries under
    raw.pack — torch-free, so the mlx arms can call it too."""
    return {k: meta.get(k) for k in PACK_RECORD_KEYS}



def pack_weight_served(w_bf16, name: str | None = None, compression_profile: str | None = None) -> dict:
    """ONE tensor exactly as `drinkme pack` writes it — the radix codec at
    the resolved profile (radix_pack.resolve_compression_profile: the explicit one, else
    DRINKME_COMPRESSION_PROFILE, else sip), or the raw fallback when radix would
    expand it. swap.make_compressed (the bench's in-memory compressed arm)
    calls this so the arm times the writer's own bytes."""
    return _pack_one(w_bf16, _rx.resolve_compression_profile(compression_profile), name=name)


# ------------------------------------------------ the container ---------


def _arrays_for(version: int, dtype: str | None = None, name: str | None = None,
                codec: str | None = None):
    """Which arrays a tensor's npz carries, from its scalars blob: the
    `codec` scalar (radix | raw). A codec this build does not know is
    refused BY NAME here — falling through would die on a bare KeyError
    three frames into np.load. A `dtype` scalar names a tensor kind this
    release does not serve (a pack cut from an FP8 checkpoint says
    `fp8_e4m3`): refused by the one line."""
    if version != FORMAT_VERSION:
        raise ValueError(f"pack format {version} tensors are not readable by this build "
                         f"(reads {FORMAT_VERSION})")
    if codec == _rx.RADIX:
        return _rx._ARRAYS_RADIX
    if codec == _rx.RAW:
        return _rx._ARRAYS_RAW
    if codec is not None:
        raise ValueError(
            f"pack tensor {name or '<unnamed>'} declares codec {codec!r}, which "
            f"this build does not implement (knows {_rx.RADIX!r} and {_rx.RAW!r}) "
            "— refusing to load.")
    if dtype is not None:
        raise UnsupportedCheckpoint(
            f"pack tensor {name or '<unnamed>'} (dtype {dtype!r}): {refusal_for_dtype(dtype)}")
    raise ValueError(
        f"pack tensor {name or '<unnamed>'} names no codec ({_rx.RADIX!r} | "
        f"{_rx.RAW!r}) — not a pack format {FORMAT_VERSION} tensor; "
        "refusing to load.")


def _scalars_for(pack: dict) -> tuple[str, ...]:
    if _rx.is_radix_pack(pack):
        return _rx._SCALARS_RADIX
    if _rx.is_raw_pack(pack):
        return _rx._SCALARS_RAW
    raise ValueError("PackWriter.add: a pack dict must be a radix or raw tensor")


# The largest tensor the loader will allocate for: 2^33 weights (16 GiB of
# bf16). Every tensor this build serves is under 2^31 (the 27B's lm_head is
# ~1e9); the cap exists so a meta.json that lies about R x C cannot make
# _check_member_sizes wave through a stream the size of the disk.
MAX_TENSOR_WEIGHTS = 1 << 33
_NPY_HEADER_SLACK = 4096  # the .npy header + zip framing above the array's own bytes


def _shape_of(name: str, sc: dict, kind: str) -> tuple[int, int]:
    R, C = sc.get("R"), sc.get("C")
    if type(R) is not int or type(C) is not int or R <= 0 or C <= 0:
        raise ValueError(f"pack tensor {name} ({kind}): R, C must be positive ints, got {R!r}, {C!r}")
    if R * C > MAX_TENSOR_WEIGHTS:
        raise ValueError(f"pack tensor {name} ({kind}): R x C = {R:,} x {C:,} is {R * C:,} weights, "
                         f"more than the {MAX_TENSOR_WEIGHTS:,} this loader will allocate for")
    return R, C


def _kind_of(sc: dict) -> str:
    return sc.get("codec") or "?"


def _check_member_sizes(z, name: str, sc: dict) -> None:
    """The allocation cap, read off the ZIP DIRECTORY before np.load
    materialises anything (a 100 GB rx_data would otherwise OOM the loader).
    A radix tensor's streams are each smaller than its bf16 plane (2 R C
    bytes) by construction — pack_array_radix stores the tensor raw when they
    would not be — so a member larger than that is not a radix stream this
    writer produced; the raw fallback's raw_bits is exactly 2 R C. `z` is
    the zipfile (np.load's NpzFile.zip)."""
    kind = _kind_of(sc)
    R, C = _shape_of(name, sc, kind)
    sizes = {i.filename: i.file_size for i in z.infolist()}

    def member(key: str) -> int:
        fn = key + ".npy"
        if fn not in sizes:
            raise ValueError(f"pack tensor {name} ({kind}): declares R x C = {R} x {C} but the npz "
                             f"holds no {fn} (members: {sorted(sizes)})")
        return sizes[fn]

    def cap(key: str, most: int, what: str) -> None:
        got = member(key)
        if got > most + _NPY_HEADER_SLACK:
            raise ValueError(f"pack tensor {name} ({kind}): {key}.npy is {got:,} bytes, more than "
                             f"{what} — refusing to allocate it")

    if kind == _rx.RADIX:
        for key in _rx._ARRAYS_RADIX:
            cap(key, 2 * R * C, f"the 2 x R x C = {2 * R * C:,} the tensor's bf16 plane takes")
    elif kind == _rx.RAW:
        cap("raw_bits", 2 * R * C, f"the 2 x R x C = {2 * R * C:,} bytes of bf16 bits it declares")


def validate_tensor(name: str, p: dict) -> None:
    """THE descriptor gate at load, hashes or not: the
    one place every loader — engines.load_compressed, mtp.install_head_pack,
    engine_mlx.load_compressed_mlx — goes through, because iter_pack_dir
    (and load_pack_dir) run it on every tensor they yield. Pure numpy, O(the
    directory) per tensor: milliseconds against verify_hashes' seconds.

      radix  radix.validate over the dict (offsets monotone from 0, the
             sentinel == data.size, the directory covers R x ceil(C/B)
             blocks, palette size/uniqueness/range, widths well formed), and
             the SERVED line on top of the research class: block_size 1024
             (the blocks the kernels walk), at most four tiers and
             nonterminal widths <= 4 (the scheduled decoder's per-block
             header is a uint16 holding one byte per nonterminal tier past
             the first; a byte fits widths <= 4 and the header three tiers —
             a crafted `widths: [2,2,2,2,8]` decoded silently wrong before)
      raw    raw_bits is uint16 [R, C]

    The file hashes and manifest digest are self-attested — a downloaded pack's
    meta.json was written by whoever wrote the streams — so it proves the
    files are the files, not that the descriptors are sane; this does."""
    kind = _kind_of(p)
    R, C = _shape_of(name, p, kind)
    if kind == _rx.RADIX:
        widths = p.get("widths")
        if (not isinstance(widths, (list, tuple)) or len(widths) < 2
                or any(isinstance(w, bool) or not isinstance(w, (int, np.integer)) for w in widths)):
            raise ValueError(f"pack tensor {name} (radix): widths {widths!r} is not a list of 2+ ints")
        widths = tuple(int(w) for w in widths)
        B = p.get("block_size")
        if B != _rx.RADIX_BLOCK:
            raise ValueError(f"pack tensor {name} (radix): block_size {B!r} — the served kernels walk "
                             f"{_rx.RADIX_BLOCK}-weight blocks only")
        if len(widths) > 4:
            raise ValueError(f"pack tensor {name} (radix): widths {list(widths)} is {len(widths)} tiers; "
                             "the scheduled decoder's block header holds at most four (three nonterminal)")
        if any(w > 4 or w < 1 for w in widths[:-1]):
            raise ValueError(f"pack tensor {name} (radix): widths {list(widths)} — a nonterminal tier "
                             "must be 1..4 bits (one byte of the block header per tier)")
        if widths[-1] != 8:
            raise ValueError(f"pack tensor {name} (radix): widths {list(widths)} — the terminal tier is "
                             "the 8-bit bf16 exponent")
        if p.get("profile") not in _rx.PROFILES:
            # a pack written under other profile names (or none) is not a
            # pack this build reads: refused by the name seen, like an
            # unknown codec; re-packing is the fix
            raise ValueError(f"pack tensor {name} (radix): profile {p.get('profile')!r} is not one this "
                             f"drinkme packs or serves ({' | '.join(_rx.PUBLIC_PROFILES)}); re-pack "
                             f"with `{repack_command()}`")
        for key in _rx._ARRAYS_RADIX:
            if key not in p:
                raise ValueError(f"pack tensor {name} (radix): no {key} array")
        try:
            _radix.validate(_rx.as_radixpack(p))
        except ValueError as e:
            raise ValueError(f"pack tensor {name} (radix): {e}") from None
    elif kind == _rx.RAW:
        bits = p.get("raw_bits")
        if bits is None:
            raise ValueError(f"pack tensor {name} (raw): no raw_bits array")
        bits = np.asarray(bits)
        if bits.dtype != np.uint16:
            raise ValueError(f"pack tensor {name} (raw): raw_bits dtype {bits.dtype}, not uint16")
        if bits.shape != (R, C):
            raise ValueError(f"pack tensor {name} (raw): raw_bits is {bits.shape}, not [R, C] = ({R}, {C})")


def repack_command(repo=None, replace: bool = True) -> str:
    """The exact command that replaces a pack this build does not read:
    `drinkme pack --model <repo> --replace` — the repo off the pack's own
    meta.json when the caller has it, a placeholder otherwise; `--replace`
    because the directory is not empty."""
    cmd = f"drinkme pack --model {repo if isinstance(repo, str) and repo else '<repo>'}"
    return cmd + (" --replace" if replace else "")


def refusal_for_version(v, path: str | None = None, repo: str | None = None) -> str:
    """The one-line refusal for a pack whose meta.json declares a format
    version this build does not read (or none at all): the path, the
    version seen, the version read, and the exact re-pack command. Every
    door (the loaders, the pack menu, serve's front door, the bench's,
    `drinkme verify`) prints this same line. Keeps the substring "pack
    format" (pinned by tests)."""
    where = path or "<pack>"
    # an int as is; anything else (a string "1", say) with its quotes, so
    # the line cannot read "unsupported pack format 1 (... reads format 1)"
    seen = "(none declared)" if v is None else (str(v) if isinstance(v, int) else repr(v))
    return (f"{where}: unsupported pack format {seen} (this drinkme reads format "
            f"{FORMAT_VERSION}); re-pack with `{repack_command(repo)}`")


def refusal_for_compression_profile(seen, path: str | None = None, repo: str | None = None) -> str:
    """The one-line refusal for a pack that names a profile this build does
    not pack or serve: the name seen, the names known, the re-pack command.
    The same shape as refusal_for_version, and printed at the same doors;
    the tensor descriptors are checked again per tensor in iter_pack_dir."""
    where = path or "<pack>"
    return (f"{where}: pack profile {seen!r} is not one this drinkme packs or serves "
            f"({' | '.join(_rx.PUBLIC_PROFILES)}); re-pack with `{repack_command(repo)}`")


def refusal_for_meta(meta: dict, path: str | None = None) -> str | None:
    """None when meta.json is one this build reads, else the one-line
    refusal: a pack cut from an FP8 checkpoint (its `sourceDtype` / `dtype`
    says `fp8_e4m3`), a format version other than the one implemented (or
    none), or — at that version — a `profile` name this build does not
    know. A tool-written meta without those fields is judged by its tensors
    (iter_pack_dir). Keys this build does not read (a format-1 pack's
    `method`, say) are ignored, not refused."""
    if is_fp8_pack_meta(meta):
        return f"{meta.get('hfRepo') or path or '<pack>'}: {FP8_REFUSAL}"
    v = meta.get("formatVersion")
    if v not in SUPPORTED_FORMAT_VERSIONS:
        return refusal_for_version(v, path, meta.get("hfRepo"))
    prof = meta.get("profile")
    if prof is not None and prof not in _rx.PROFILES:
        return refusal_for_compression_profile(prof, path, meta.get("hfRepo"))
    return None


def _check_format_version(meta: dict, path: str | None = None) -> int:
    """The refusal gate: a build reads the format version it implements and
    nothing else — no other reader, no migration — and, at that version,
    only the profile names it knows (refusal_for_meta)."""
    line = refusal_for_meta(meta, path)
    if line is None:
        return int(meta["formatVersion"])
    if is_fp8_pack_meta(meta):
        raise UnsupportedCheckpoint(line)
    raise ValueError(line)


# the one filename shape the writer emits (PackWriter: t0000.npz, t0001.npz, …)
_TENSOR_FILE = re.compile(r"t\d{4,}\.npz\Z")


def _tensor_file(path: str, name: str, fn) -> str:
    r"""The on-disk path of a tensor's npz, from meta.json's `tensors` map —
    refused BY NAME unless `fn` is the bare basename the writer emits
    (`t\d{4,}\.npz`: no directory part, no `..`, no absolute path) AND the
    file it resolves to (symlinks followed) is inside the pack directory.
    Otherwise a downloaded pack's meta.json could name `../outside.npz` or
    `/etc/anything`, and `verify_hashes` would hash it and `iter_pack_dir`
    parse it (read-only, `allow_pickle=False`, but any file the user can
    read)."""
    if not isinstance(fn, str) or not _TENSOR_FILE.fullmatch(fn) or os.path.basename(fn) != fn:
        raise ValueError(f"{path}/meta.json: tensor {name!r} names file {fn!r}; a pack file is a "
                         "bare basename of the form t0000.npz — refusing to open it")
    root = os.path.realpath(path)
    fp = os.path.realpath(os.path.join(path, fn))
    if os.path.dirname(fp) != root:
        raise ValueError(f"{path}/meta.json: tensor {name!r}'s file {fn!r} resolves to {fp}, outside "
                         "the pack directory — refusing to open it")
    return os.path.join(path, fn)


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


class _PackLock:
    """One writer per destination. A sibling `<dest>.lock`, flock'd
    non-blocking: a second writer refuses immediately, naming the lock,
    rather than waiting hours behind a pack in progress and then finding
    the destination taken. flock dies with its process, so a crash never
    leaves the destination locked; the file itself is unlinked on release
    (with the inode re-check that makes unlink-on-release safe). Where
    fcntl does not exist the lock is an O_EXCL file, released by unlink."""

    def __init__(self, path: str) -> None:
        self.path = path
        self.fd: int | None = None
        try:
            import fcntl
        except ImportError:  # not POSIX
            try:
                self.fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o644)
            except FileExistsError:
                raise RuntimeError(self._held()) from None
            return
        while True:
            fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                os.close(fd)
                raise RuntimeError(self._held()) from None
            try:
                same = os.fstat(fd).st_ino == os.stat(path).st_ino
            except FileNotFoundError:
                same = False
            if same:
                self.fd = fd
                return
            os.close(fd)  # the file we locked was unlinked under us: again

    def _held(self) -> str:
        return (f"another pack writer holds {self.path} — a pack into this "
                "destination is already in progress (or its process is still "
                "alive). Wait for it, or point --out elsewhere.")

    def release(self) -> None:
        if self.fd is None:
            return
        try:
            os.unlink(self.path)
        except FileNotFoundError:
            pass
        os.close(self.fd)  # drops the flock
        self.fd = None



class PackWriter:
    """Streaming counterpart to save_pack_dir — same dir layout, same
    sorted-name filename assignment, but each tensor hits disk the moment it
    is packed and is then released, so peak host RAM is ONE tensor's pack,
    not one model's. The tensor SET is declared up front (filenames must be
    stable regardless of arrival order). meta.json is written LAST by
    finish() and is the commit point: a crashed or incomplete pack has no
    meta.json and therefore does not exist to the loader.

    TRANSACTIONAL: nothing is written at `path` until finish() completes.
    Every file goes to a fresh sibling `<path>.staging-<pid>`, and finish()
    publishes it by ONE rename — so a crash, an incomplete tensor set or a
    failed verification leaves the destination exactly as it was: absent,
    or the previous pack, byte for byte and still verifying (a re-pack that
    overwrote files in place would leave old metadata over a mixture of old
    and new files). A non-empty destination is REFUSED unless
    the caller passes replace=True; replace stages the whole new pack first,
    then swaps the directories, and the old pack is removed only after the
    new one is in place. A `<path>.lock` sibling (_PackLock) serialises
    competing writers. meta.json itself is written via a temp file +
    os.replace, never truncated in place. abort() discards the staging dir
    and releases the lock; finish() on an incomplete set does the same
    before raising.

    THE HASHES: each npz is sha256-hashed from DISK the moment it lands
    (read-back, not the buffer we asked to write — on a FUSE mount those
    can differ), and finish() records the
    hashes in meta.json. Verification against the true source happened at
    pack time; the hashes extend it forward: a loader that checks the hash
    is serving the exact bytes that passed. Load-time proof drops from a
    full CPU decode of the model (×2) to hashing the files.

    THE MANIFEST: the file hashes alone do not bind a tensor NAME to its
    file — unhashed, that map could have two same-shaped entries swapped
    and every file would still hash clean. finish() also records
    per-tensor shape/dtype/layout (`tensorInfo`) and the digest of the
    canonical manifest over name -> file -> hash -> descriptor + the source
    identity (`manifestSha256`, codec/identity.py); verify_hashes recomputes
    it from the live maps."""

    def __init__(self, path: str, names, format_version: int = FORMAT_VERSION,
                 replace: bool = False) -> None:
        if format_version != FORMAT_VERSION:
            raise ValueError(f"PackWriter: this build writes pack format {FORMAT_VERSION} only, "
                             f"not {format_version}")
        path = os.path.normpath(path)
        parent = os.path.dirname(os.path.abspath(path))
        os.makedirs(parent, exist_ok=True)
        self.path = path
        self.replace = replace
        self.format_version = format_version
        self.names = {n: f"t{i:04d}.npz" for i, n in enumerate(sorted(names))}
        self.written: set[str] = set()
        self.hashes: dict[str, str] = {}
        self.info: dict[str, dict] = {}
        self.staging = f"{path}.staging-{os.getpid()}"
        self._lock: _PackLock | None = _PackLock(path + ".lock")
        try:
            if os.path.isdir(path) and os.listdir(path) and not replace:
                raise FileExistsError(
                    f"pack destination is not empty: {path} — refusing to write "
                    "over it. Pass replace=True (`drinkme pack --replace`) to "
                    "stage a new pack and swap it in whole, or choose another "
                    "--out.")
            if os.path.exists(path) and not os.path.isdir(path):
                raise FileExistsError(f"pack destination {path} exists and is not a directory")
            if os.path.isdir(self.staging):
                shutil.rmtree(self.staging)  # ours by pid: a dead run's leftover
            os.makedirs(self.staging)
        except BaseException:
            self._lock.release()
            self._lock = None
            raise

    def file_path(self, name: str) -> str:
        """Where a tensor's npz is RIGHT NOW: in staging until finish()
        publishes, then under path. Callers that size files must ask here
        rather than assume the destination."""
        fn = self.names[name]
        root = self.staging if os.path.isdir(self.staging) else self.path
        return os.path.join(root, fn)

    def add(self, name: str, pack: dict) -> None:
        fn = self.names[name]
        fp = os.path.join(self.staging, fn)
        arrays = _arrays_for(self.format_version, pack.get("dtype"), name=name,
                             codec=pack.get("codec"))
        scalars = _scalars_for(pack)
        np.savez(
            fp,
            **{k: pack[k] for k in arrays},
            scalars=np.array([json.dumps({k: pack[k] for k in scalars})]),
        )
        self.hashes[fn] = _sha256_file(fp)
        self.info[name] = _identity.tensor_info(pack, name=name)
        self.written.add(name)

    def abort(self) -> None:
        """Discard the staging dir and release the lock; the destination is
        untouched. Idempotent; safe after finish()."""
        if os.path.isdir(self.staging):
            shutil.rmtree(self.staging, ignore_errors=True)
        if self._lock is not None:
            self._lock.release()
            self._lock = None

    def finish(self, meta: dict) -> None:
        missing = sorted(set(self.names) - self.written)
        if missing:
            self.abort()
            raise ValueError(
                f"pack incomplete — {len(missing)} declared tensors never "
                f"arrived, meta.json withheld: {missing[:8]}"
            )
        full = {"formatVersion": self.format_version, "tensors": self.names,
                "sha256": dict(sorted(self.hashes.items())), **meta}
        full["tensorInfo"] = {n: self.info[n] for n in sorted(self.info)}
        full["manifestSha256"] = _identity.manifest_digest(full)
        _write_meta(self.staging, full)
        try:
            self._publish()
        except BaseException:
            self.abort()
            raise
        if self._lock is not None:
            self._lock.release()
            self._lock = None

    def _publish(self) -> None:
        """staging -> path by rename. An empty existing destination is
        removed first; a non-empty one (replace=True) is moved aside, the
        new pack renamed in, and only then is the old one deleted — if the
        second rename fails the old pack is put back."""
        if os.path.isdir(self.path):
            if not os.listdir(self.path):
                os.rmdir(self.path)
            else:
                old = f"{self.path}.replaced-{os.getpid()}"
                if os.path.isdir(old):
                    shutil.rmtree(old)
                os.rename(self.path, old)
                try:
                    os.rename(self.staging, self.path)
                except BaseException:
                    os.rename(old, self.path)
                    raise
                shutil.rmtree(old, ignore_errors=True)
                return
        os.rename(self.staging, self.path)



def _write_meta(path: str, meta: dict) -> None:
    """meta.json via a temp file + os.replace: the commit point is one
    rename, never a truncate-in-place that an interruption could leave
    half-written."""
    target = os.path.join(path, "meta.json")
    tmp = target + ".tmp"
    if os.path.lexists(tmp):
        os.unlink(tmp)  # a new inode, never a write through a hard link (`cp -al` copies)
    with open(tmp, "w") as f:
        json.dump(meta, f, indent=2)
    os.replace(tmp, target)


def save_pack_dir(path: str, tensors: dict[str, dict], meta: dict,
                  format_version: int = FORMAT_VERSION, replace: bool = False) -> None:
    """Write a model pack: <path>/meta.json + one .npz per tensor. In-memory
    batch form of PackWriter, for tests and tools that already hold the packs;
    pack_model streams instead so a model's worth is never resident.

    meta must identify the source exactly (hf repo, revision, dtype) — a pack
    whose provenance is unknown is not servable. `profile` is filled in
    (off the first radix tensor; the default profile for an all-raw pack)
    when the caller left it out, so a test pack loads the way a written
    one does.
    """
    if "profile" not in meta:
        rx = next((p for p in tensors.values() if _rx.is_radix_pack(p)), None)
        meta = {**meta, "profile": str(rx["profile"]) if rx else _rx.DEFAULT_PROFILE}
    w = PackWriter(path, list(tensors), format_version=format_version, replace=replace)
    try:
        for name, p in tensors.items():
            w.add(name, p)
        w.finish(meta)
    except BaseException:
        w.abort()
        raise


def load_pack_dir(path: str) -> tuple[dict[str, dict], dict]:
    """Load EVERY tensor into host RAM at once. Fine for tests and tools;
    model loaders must use iter_pack_dir instead — this function's peak is a
    whole model resident in host RAM, which is exactly what the fit check
    forbids (the machine that needs the pack cannot hold the model)."""
    with open(os.path.join(path, "meta.json")) as f:
        meta = json.load(f)
    v = _check_format_version(meta, path)
    tensors = {name: _read_tensor(path, v, name, fn) for name, fn in meta["tensors"].items()}
    return tensors, meta


def _read_tensor(path: str, v: int, name: str, fn) -> dict:
    """One tensor of a pack directory as iter_pack_dir yields it: its npz
    opened and released, the arrays its codec names materialised after
    their sizes were checked against R x C, and the descriptor validated."""
    with np.load(_tensor_file(path, name, fn)) as z:
        sc = json.loads(str(z["scalars"][0]))
        # which arrays to pull is known per tensor (the codec/dtype
        # scalars ride in the blob); their sizes are checked before any
        # is read, and they are materialised BEFORE the file closes
        keys = _arrays_for(v, sc.get("dtype"), name=name, codec=sc.get("codec"))
        _check_member_sizes(z.zip, name, sc)
        p = {k: np.asarray(z[k]) for k in keys}
        p.update(sc)
    p["format_version"] = v
    validate_tensor(name, p)
    return p


def read_pack_tensor(path: str, meta: dict, name: str) -> dict:
    """One named tensor of the pack at `path` (its meta.json already read),
    through the same format, size and descriptor gates as iter_pack_dir:
    for a reader that needs the tensors in another order than meta.json's
    (upstream.py rebuilds the source shards in header order)."""
    return _read_tensor(path, _check_format_version(meta, path), name, meta["tensors"][name])


def under(name: str, paths) -> bool:
    """Is `name` (a tensor or module name) one of the subtree `paths`, or
    inside one? A subtree can itself be ONE Linear, whose pack tensor is
    named the path exactly: Muse-Glimmer's model.vision_projection, a
    vision-tower subtree (arms._VISION_TOWERS) that a "path." prefix
    misses."""
    return any(name == p or name.startswith(p + ".") for p in paths)


def iter_pack_dir(path: str, skip: tuple[str, ...] | None = None):
    """Yield (tensor_name, pack_dict) ONE tensor at a time, lazily.

    THE loader entry point. Each .npz is opened only when its tensor is
    pulled and is fully released before the next, so peak host RAM is one
    tensor, not one model — a loader consumes this, moves the arrays to the
    device, and lets them drop. Same meta.json + format check as
    load_pack_dir; meta order (sorted tensor names) is the yield order.

    Every yielded pack dict carries "format_version" (1): the version this pack
    directory declared, echoed onto each tensor so downstream consumers
    can dispatch without re-reading meta.json; the tensor's own kind is its
    `codec` (radix | raw) scalar.

    Every yielded tensor has passed validate_tensor (the descriptor gate,
    hashed or not) and _check_member_sizes (the allocation
    cap against R x C, read off the zip directory before np.load
    materialises a member); a tensor that fails is refused by name here,
    before any loader builds a module over it.

    `skip`: subtree paths (`under`) whose tensors are neither read nor
    yielded — a vision tower the loader is not building (DRINKME_VISION=0,
    the bench's text arms). Their files still passed the hash gate."""
    with open(os.path.join(path, "meta.json")) as f:
        meta = json.load(f)
    v = _check_format_version(meta, path)
    for name, fn in meta["tensors"].items():
        if skip is not None and under(name, skip):
            continue
        yield name, _read_tensor(path, v, name, fn)


def iter_pack_descriptors(path: str):
    """Yield (tensor_name, scalars, {member: bytes}) ONE tensor at a time,
    reading NO array: meta.json, each npz's `scalars` blob (a few hundred
    bytes) and its zip directory — the member sizes np.load would allocate
    for. The memory-estimate half of iter_pack_dir: what a loader needs to
    charge a pack's resident bytes BEFORE it materialises a tensor
    (serving/engine_mlx.estimate_load_bytes), through the same format,
    codec and allocation-cap gates, so a tensor iter_pack_dir would refuse
    is refused here first, by name. Sizes are the zip directory's
    uncompressed member sizes, keyed by array name (`rx_data`, ...) — the
    .npy header's ~128 bytes ride along; the resident module's pad words
    and padded palette are about as many the other way."""
    with open(os.path.join(path, "meta.json")) as f:
        meta = json.load(f)
    v = _check_format_version(meta, path)
    for name, fn in meta["tensors"].items():
        with np.load(_tensor_file(path, name, fn)) as z:
            sc = json.loads(str(z["scalars"][0]))
            _arrays_for(v, sc.get("dtype"), name=name, codec=sc.get("codec"))
            _check_member_sizes(z.zip, name, sc)
            sizes = {i.filename[:-4]: i.file_size for i in z.zip.infolist()
                     if i.filename.endswith(".npy")}
        yield name, dict(sc), sizes


def packed_resident_estimate(sc: dict, sizes: dict) -> int:
    """The bytes a loader will hold resident for one pack tensor, off its
    descriptor alone (iter_pack_descriptors' scalars + member sizes): the
    radix streams, directory and palette, or the raw fallback's bf16 bits —
    the arrays the module keeps (gemv_radix.resident / swap.to_device_*)
    plus the radix reach-bound pad those allocations carry — within a
    header's worth of the measured figure."""
    codec = sc.get("codec")
    keys = _rx._ARRAYS_RADIX if codec == _rx.RADIX else _rx._ARRAYS_RAW
    total = sum(int(sizes.get(k, 0)) for k in keys)
    if codec == _rx.RADIX:
        # The device allocation carries block_word_bounds(B)[1] zero words
        # after the payload (swap.to_device_radix, metal/gemv_radix.pad_words:
        # the readers' reach bound, sip 608 / gulp 768 words at B = 1024) and
        # resident_bytes counts them — so does the estimate.
        from .radix import block_word_bounds
        widths = tuple(int(w) for w in sc["widths"])
        total += 4 * int(block_word_bounds(int(sc["block_size"]), 7, 8, widths)[1])
    return total


def _say(msg: str) -> None:
    print(f"[drinkme] {msg}", flush=True)


def verify_hashes(path: str) -> bool:
    """Load-time integrity gate: every file matches the per-file sha256 in
    meta.json, and the recorded manifest digest recomputes from the live maps. Returns
    True; raises on anything else — a pack that fails verification does not get
    served, ever.

    The hash check is the cheap gate that stands in for per-tensor decode-verify
    at serve time: pack time verifies, load time checks the hash. Every
    pack records its hashes, manifest and
    all, at pack time (PackWriter.finish), so a meta.json WITHOUT the
    per-file hashes or the manifest digest is not a pack this drinkme wrote
    — or was edited since — and is refused by name here, not served on a
    weaker check.

    THE MANIFEST: `manifestSha256` is recomputed from the
    live meta.json maps first — name -> file -> hash -> shape/dtype/layout
    + source — so a reassigned tensor name fails here even though every
    file it names hashes clean."""
    with open(os.path.join(path, "meta.json")) as f:
        meta = json.load(f)
    # the tensor map's file names first: a name outside the
    # pack directory is refused before anything is hashed or even stat'ed
    files = {name: _tensor_file(path, name, fn) for name, fn in meta["tensors"].items()}
    want = meta.get("sha256")
    fix = repack_command(meta.get("hfRepo"))
    if not isinstance(want, dict):
        raise ValueError(
            f"not verifiable: {path}/meta.json records no per-file sha256 — every pack "
            "records them at pack time, so this is not a pack this drinkme wrote, or "
            f"its meta.json was edited. Refusing to serve; re-pack: `{fix}`")
    recorded = meta.get("manifestSha256")
    if not recorded:
        raise ValueError(
            f"no manifest: {path}/meta.json records no manifestSha256 — every pack "
            "records its tensor manifest digest at pack time, so this is not a pack this "
            f"drinkme wrote, or its meta.json was edited. Refusing to serve; re-pack: `{fix}`")
    got = _identity.manifest_digest(meta)
    if got != recorded:
        raise ValueError(
            f"manifest broken: {path}/meta.json's tensor manifest digests "
            f"{got[:16]}…, recorded {recorded[:16]}… — the tensor-name -> "
            "file assignment, a shape/layout descriptor or the source "
            "identity has changed since the pack was written, so the files "
            "may hash clean and still not be the tensors they claim to be. "
            "Refusing to serve; re-pack or restore meta.json.")
    items = []
    for name, fn in meta["tensors"].items():
        fp = files[name]
        if fn not in want:
            raise ValueError(f"hashes incomplete: no recorded hash for {fn} ({name})")
        if not os.path.exists(fp):
            raise ValueError(f"pack file missing: {fn} ({name})")
        items.append((name, fn, fp, want[fn]))
    # the embedded checkpoint (a self-contained pack): each file against the
    # sha256 its `embedded.files` entry records — the map itself is in the
    # manifest checked above
    for fn, digest in _embedded_files(path, meta).items():
        fp = _embedded_file(path, fn)
        if not os.path.isfile(fp):
            raise ValueError(f"pack file missing: {EMBED_SUBDIR}/{fn} (embedded checkpoint)")
        items.append(("embedded checkpoint", f"{EMBED_SUBDIR}/{fn}", fp, digest))

    def _one(item) -> None:
        name, fn, fp, recorded_hash = item
        got = _sha256_file(fp)
        if got != recorded_hash:
            raise ValueError(
                f"hash mismatch: {fn} ({name}) hashes {got[:16]}…, meta.json "
                f"recorded {recorded_hash[:16]}… — the pack on disk is not the pack "
                "that passed verification. Refusing to serve; re-pack or "
                "restore the file."
            )

    # 4 workers: hashlib releases the GIL and sha256 saturates this class of
    # NVMe at ~4 threads (measured 2.1x over serial).
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=4) as ex:
        for _ in ex.map(_one, items):
            pass
    return True


def verify_pack(path: str, progress=print) -> None:
    """`drinkme verify`: verify a pack in place and say so. Every pack
    records its hashes, manifest and all, at pack time (PackWriter.finish), so there
    is nothing to add: this runs the loader's own gate — the format version
    first, then verify_hashes (every file against its recorded sha256, the
    manifest digest against the live maps) — and reports, without loading a
    model. A pack that fails is refused with the gate's own line; a `mtp/`
    head sub-pack is verified the same way. meta.json is never rewritten."""
    with open(os.path.join(path, "meta.json")) as f:
        meta = json.load(f)
    # the format gate FIRST, so a pack of another version gets the one
    # refusal line rather than a hash verdict
    _check_format_version(meta, path)
    verify_hashes(path)  # raises on a mismatched or missing hash
    embedded = _embedded_files(path, meta)
    if embedded:
        # the embedded files hash clean; they must also be the checkpoint
        # the pack names (the loader's check, run here without loading)
        check_embedded_identity(path, meta)
    progress(f"verified, manifest and all: {path} — {len(meta['tensors'])} files verified, "
             f"manifest {meta['manifestSha256'][:16]}…"
             + (f"; embedded checkpoint: {len(embedded)} files verified, identity matches "
                f"{_identity.describe(meta.get('source'))}" if embedded
                else "; no embedded checkpoint (refused at load: repack it)"
                if not meta.get("module") else ""))
    if has_mtp_pack(path):
        verify_pack(mtp_pack_dir(path), progress)


# ------------------------------------------------ the MTP head sub-pack --
# The trunk is served compressed; a raw bf16 MTP draft head would cost
# residency the fit check pays for and, at batch 1, stream its own 0.79 GiB
# every draft step. The head's eligible Linears pack with the SAME codec,
# the SAME eligibility rule and the SAME writer as the trunk's — the only
# difference is WHERE they live.
#
# WHERE: a nested pack dir, `<pack>/mtp/`, with its own meta.json, rather
# than `mtp.*` keys in the trunk's tensor map: the trunk's meta.json lists
# exactly the text model's tensors, so engines.load_compressed's
# `model.get_submodule(name)` over it never meets a name the text skeleton
# lacks, and the head loader (serving/mtp.py) reads the subdir on its own.
# The subdir is a complete pack dir, so PackWriter/iter_pack_dir/verify_hashes/
# verify_pack all apply to it unchanged, and its tensor names are MTPHead
# submodule paths ("fc", "layers.0.mlp.gate_proj") exactly as the trunk's are
# text-model paths: pack names == what the loader asks for, by construction.
MTP_SUBDIR = "mtp"


def mtp_pack_dir(pack_dir: str) -> str:
    """The head sub-pack's directory inside a model pack."""
    return os.path.join(pack_dir, MTP_SUBDIR)


def has_mtp_pack(pack_dir: str) -> bool:
    """Does this pack carry a packed MTP head? meta.json is the commit point
    for the sub-pack exactly as it is for the trunk, so its presence — not
    the directory's — is the question."""
    return os.path.isfile(os.path.join(mtp_pack_dir(pack_dir), "meta.json"))


def read_mtp_meta(pack_dir: str) -> dict | None:
    """The head sub-pack's meta.json, or None when there is no head pack."""
    if not has_mtp_pack(pack_dir):
        return None
    with open(os.path.join(mtp_pack_dir(pack_dir), "meta.json")) as f:
        return json.load(f)


# ------------------------------------------- the embedded checkpoint --
# A pack's packed Linears are half a model; the other half (embeddings,
# norms, biases, sub-bar projections, the MTP head's norms) and the files
# the skeleton, tokenizer and sampler defaults are built from are the
# other half, so every pack carries them in `<pack>/checkpoint/`
# (docs/pack-format.md#the-embedded-checkpoint): the small files verbatim,
# the unpacked tensors copied byte for byte into one safetensors file, and
# the shard headers the source identity was digested from. That directory
# is itself a checkpoint the loaders already read — config.json, a
# tokenizer, *.safetensors — so resolve_pack_source hands it to them in
# place of the snapshot and nothing downstream changes. meta.json's
# `embedded` block records each file's sha256; the manifest binds the
# block, verify_hashes checks it. A pack without the block is refused at
# load: repack it.
#
# Why safetensors and not the npz container: the remainder is every dtype
# and rank a checkpoint ships (1-D norms, F32 biases, int buffers), while
# the npz raw kind is uint16 [R, C] by definition; a byte-range copy keeps
# the source's own dtype, so the loaders' existing cast rule (float -> bf16
# at load, exactly from_pretrained's) runs on the same bytes as before and
# the served weights are identical by construction; and every loader, both
# runtimes, reads it today (safe_open, mx.load) without a new reader.
EMBED_SUBDIR = "checkpoint"
REMAINDER_FILE = "remainder.safetensors"
# The small files carried along: the patterns snapshot_dir fetches beside
# the weights (config, tokenizer, templates, generation defaults, processor
# configs), plus the license and model card, since a self-contained pack
# redistributes the model. Top level only; never a weight file or the
# shard index (identity keeps the index's bytes instead).
_EMBED_PATTERNS = ("*.json", "*.jinja", "*.txt", "*.model", "*.md", "LICENSE*", "NOTICE*")
_EMBED_FILE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")
_COPY_CHUNK = 16 << 20


def embedded_dir(pack_dir: str) -> str:
    """Where a self-contained pack keeps its checkpoint."""
    return os.path.join(pack_dir, EMBED_SUBDIR)


def is_self_contained(meta: dict) -> bool:
    """Does this pack's meta.json declare an embedded checkpoint? The
    `embedded` block is the commit point: a `checkpoint/` directory that
    meta.json does not name (an interrupted write) is ignored. False means
    the pack carries no embedded checkpoint; the loaders refuse to serve it
    (serving/checkpoint.resolve_pack_source)."""
    return isinstance(meta.get("embedded"), dict)


def _embedded_files(path: str, meta: dict) -> dict:
    if not is_self_contained(meta):
        return {}
    files = meta["embedded"].get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError(f"{path}/meta.json: `embedded` names no files — not a pack this "
                         f"drinkme wrote. Re-pack: `{repack_command(meta.get('hfRepo'))}`")
    return files


def _embedded_file(path: str, fn) -> str:
    """The on-disk path of one embedded file, refused BY NAME unless `fn` is
    a bare basename that resolves inside `<pack>/checkpoint/` — the same
    rule _tensor_file holds meta.json's tensor map to."""
    if not isinstance(fn, str) or not _EMBED_FILE.fullmatch(fn):
        raise ValueError(f"{path}/meta.json: embedded file {fn!r} is not a bare file name — "
                         "refusing to open it")
    root = os.path.realpath(embedded_dir(path))
    fp = os.path.realpath(os.path.join(root, fn))
    if os.path.dirname(fp) != root:
        raise ValueError(f"{path}/meta.json: embedded file {fn!r} resolves to {fp}, outside "
                         f"{EMBED_SUBDIR}/ — refusing to open it")
    return os.path.join(embedded_dir(path), fn)


def check_embedded_identity(path: str, meta: dict) -> str:
    """The embedded checkpoint IS the one `source` names: its digest,
    recomputed from the embedded config, tokenizer files, image processor
    configs and the recorded shard headers (identity.embedded_digest),
    equals the recorded one.
    Returns the embedded directory; raises ValueError naming both
    identities otherwise."""
    src = meta.get("source")
    where = embedded_dir(path)
    if not isinstance(src, dict) or not src.get("digest"):
        raise ValueError(f"{path}: carries an embedded checkpoint but records no source "
                         "identity to check it against — not a pack this drinkme wrote. "
                         f"Re-pack: `{repack_command(meta.get('hfRepo'))}`")
    if src.get("identityVersion") != _identity.IDENTITY_VERSION:
        raise ValueError(f"{path}: records its source with identity version "
                         f"{src.get('identityVersion')}; this build computes version "
                         f"{_identity.IDENTITY_VERSION}, so the embedded checkpoint cannot "
                         "be checked. Re-pack.")
    try:
        got = _identity.embedded_digest(where)
    except (OSError, ValueError, KeyError, TypeError) as e:
        raise ValueError(f"{path}: embedded checkpoint unreadable ({type(e).__name__}: {e})") from None
    if got != src["digest"]:
        raise ValueError(
            f"embedded checkpoint mismatch: {where} digests {got[:16]}…, but the pack is bound "
            f"to {_identity.describe(src)} — its config, tokenizer files, image processor "
            "configs or shard headers are not the ones the pack was cut from. Refusing to "
            "load; restore the pack, or re-pack.")
    return where


def _shard_layout(fpath: str) -> tuple[int, dict]:
    """(byte offset of the data section, header dict) of one safetensors file."""
    import struct

    with open(fpath, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    return 8 + n, header


def tied_output_weight(model) -> str | None:
    """The skeleton name of the output embedding's weight when the model
    ties it to the input embedding (one Parameter under two names), else
    None. The loaders never read it: stream_checkpoint's tie_weights
    points it back at the embedding (and the MLX skeleton has no lm_head to
    hold it), so a checkpoint that ships the redundant copy anyway (Qwen3-
    0.6B does, 311 MB) does not have it embedded."""
    out, inp = model.get_output_embeddings(), model.get_input_embeddings()
    if out is None or inp is None or out.weight is not inp.weight:
        return None
    for name, mod in model.named_modules():
        if mod is out:
            return f"{name}.weight"
    return None


def embedded_tensor_names(snap: str, to_skel, trunk_weights: set, head_weights: set,
                          tied: str | None = None) -> list[str]:
    """The checkpoint tensors a self-contained pack must carry: exactly the
    ones the loaders read raw. `to_skel` is arms.ckpt_to_skel(cfg);
    `trunk_weights` the skeleton names (`<linear>.weight`) the trunk packed;
    `head_weights` the checkpoint keys (`mtp.<linear>.weight`) the `mtp/`
    sub-pack packed.

      trunk  every tensor with a place in the text skeleton that the pack
             does not carry (stream_checkpoint's rule: a packed Linear's
             weight is skipped, its bias is read)
      head   every mtp.* tensor the head sub-pack does not carry (its norms;
             the whole raw head when the pack has no sub-pack, which is
             what mtp.load_head then reads)

    A tower the served tree never builds (arms._NON_TEXT_TOWERS: a
    text-config Qwen3.5 checkpoint's model.visual; `to_skel` says None) is
    not carried: no loader reads it; neither is a tied output
    embedding (`tied`, tied_output_weight), which the loaders re-tie. A
    served vision tower (arms._VISION_TOWERS) is part of the tree, so its
    raw tensors are carried like the text model's."""
    names = []
    for tname in _shard_headers(snap):
        if tname.startswith("mtp."):
            if tname not in head_weights:
                names.append(tname)
            continue
        sname = to_skel(tname)
        if sname is not None and sname not in trunk_weights and sname != tied:
            names.append(tname)
    return names


def _embed_small_files(snap: str) -> list[str]:
    import fnmatch

    out = []
    for name in sorted(os.listdir(snap)):
        if (name.endswith(".safetensors") or name == _identity.INDEX_FILE
                or not os.path.isfile(os.path.join(snap, name))):
            continue
        if any(fnmatch.fnmatch(name, pat) for pat in _EMBED_PATTERNS):
            if not _EMBED_FILE.fullmatch(name):
                continue  # a name meta.json could not record (spaces, a leading dot)
            if name in (_identity.EMBEDDED_IDENTITY_FILE, REMAINDER_FILE):
                raise ValueError(f"{snap}/{name}: the checkpoint ships a file under a name the "
                                 "embedded checkpoint reserves; cannot embed it")
            out.append(name)
    return out


def write_embedded(snap: str, dest: str, tensor_names, progress=print) -> dict:
    """Write the embedded checkpoint for `snap` into `dest` (a fresh
    directory inside a staging area; the caller publishes it) and return
    meta.json's `embedded` block. Torch-free.

      the small files   copied verbatim (_EMBED_PATTERNS)
      the tensors       `tensor_names`, each copied as the byte range its
                        shard header names, with its dtype and shape, into
                        one safetensors file (remainder.safetensors) —
                        never decoded or cast
      the identity      source-identity.json: the shard headers and index
                        content_digest reads, so the digest recomputes

    Verified before it returns: every tensor re-read from the written file
    hashes equal to its source bytes, and the digest recomputed from `dest`
    equals content_digest(snap). A failure raises; the caller discards
    `dest`."""
    os.makedirs(dest)
    for name in _embed_small_files(snap):
        shutil.copyfile(os.path.join(snap, name), os.path.join(dest, name))
    with open(os.path.join(dest, _identity.EMBEDDED_IDENTITY_FILE), "w") as f:
        json.dump(_identity.embedded_identity_record(snap), f, indent=1)

    # which shard holds each wanted tensor, and where
    want = set(tensor_names)
    entries = []  # (tname, dtype, shape, shard path, absolute start, nbytes)
    import glob

    for fpath in sorted(glob.glob(os.path.join(snap, "*.safetensors"))):
        base, header = _shard_layout(fpath)
        for tname, info in sorted(((k, v) for k, v in header.items() if k != "__metadata__"),
                                  key=lambda kv: kv[1]["data_offsets"][0]):
            if tname in want:
                start, end = (int(x) for x in info["data_offsets"])
                entries.append((tname, info["dtype"], list(info["shape"]), fpath,
                                base + start, end - start))
    missing = sorted(want - {e[0] for e in entries})
    if missing:
        raise ValueError(f"{snap}: the checkpoint holds none of {missing[:8]} — cannot embed")

    header, off = {}, 0
    for tname, dtype, shape, _fp, _start, nbytes in entries:
        header[tname] = {"dtype": dtype, "shape": shape, "data_offsets": [off, off + nbytes]}
        off += nbytes
    header["__metadata__"] = {"format": "pt"}
    hb = json.dumps(header, separators=(",", ":")).encode("utf-8")
    hb += b" " * ((8 - len(hb) % 8) % 8)
    out_path = os.path.join(dest, REMAINDER_FILE)
    source_hashes = {}
    with open(out_path, "wb") as out:
        out.write(len(hb).to_bytes(8, "little"))
        out.write(hb)
        for tname, _dtype, _shape, fpath, start, nbytes in entries:
            h = hashlib.sha256()
            with open(fpath, "rb") as src:
                src.seek(start)
                left = nbytes
                while left:
                    chunk = src.read(min(left, _COPY_CHUNK))
                    if not chunk:
                        raise ValueError(f"{fpath}: truncated inside {tname}")
                    h.update(chunk)
                    out.write(chunk)
                    left -= len(chunk)
            source_hashes[tname] = h.hexdigest()

    # the proof, read back from disk: every tensor's bytes are its source's
    base, written = _shard_layout(out_path)
    with open(out_path, "rb") as f:
        for tname, digest in source_hashes.items():
            start, end = written[tname]["data_offsets"]
            f.seek(base + start)
            h = hashlib.sha256()
            left = end - start
            while left:
                chunk = f.read(min(left, _COPY_CHUNK))
                h.update(chunk)
                left -= len(chunk)
            if h.hexdigest() != digest:
                raise ValueError(f"embedded tensor {tname} does not read back as its source bytes")
    got, want_digest = _identity.embedded_digest(dest), _identity.content_digest(snap)
    if got != want_digest:
        raise ValueError(f"embedded checkpoint digests {got[:16]}…, the source {want_digest[:16]}… "
                         "— an identity file was not carried")

    files = {name: _sha256_file(os.path.join(dest, name)) for name in sorted(os.listdir(dest))}
    nbytes = sum(os.path.getsize(os.path.join(dest, name)) for name in files)
    progress(f"  embedded checkpoint: {len(entries)} tensors ({off / 2**20:.1f} MiB) + "
             f"{len(files) - 2} files, {nbytes / 2**20:.1f} MiB in {EMBED_SUBDIR}/")
    return {"files": files, "tensorCount": len(entries), "tensorBytes": off, "bytes": nbytes}


def resident_bytes(runtime: dict) -> int:
    """Device-resident bytes of a runtime dict (swap.to_device_radix /
    to_device_raw).

    The residency charge the fit guard needs is NOT the npz's size: the
    radix runtime dict carries the padded palette tables and (3+ tiers) the
    per-block schedule that the file does not, so what sits on the device is
    a little more than what sat on disk. Measured here rather than estimated,
    from the one object that knows — the dict the loader is about to hand
    the module."""
    import torch

    return sum(v.numel() * v.element_size() for v in runtime.values()
               if torch.is_tensor(v))


# safetensors header dtype strings -> resident width. A floating tensor's
# resident width is ALWAYS 2 (load_compressed's raw-tensor loop converts
# every one to bf16, `.to(dtype=bf16 if is_floating_point else None)`,
# regardless of what it was on disk); a non-floating one keeps its own.
_RESIDENT_FLOAT_DTYPES = {"F64", "F32", "F16", "BF16"}
_RESIDENT_NONFLOAT_NBYTES = {"I64": 8, "I32": 4, "I16": 2, "I8": 1,
                            "U64": 8, "U32": 4, "U16": 2, "U8": 1, "BOOL": 1}


def raw_resident_bytes(dtype: str, shape) -> int:
    """Bytes ONE raw (unpacked) checkpoint tensor costs resident, from its
    safetensors header dtype and shape: a float plane at bf16 width (every
    raw loader converts to bf16 — engines.load_compressed's raw-tensor
    loop, engine_mlx._stream_checkpoint's astype), a non-float at its own.
    Pure arithmetic, torch-free: the packer's measurer below and the MLX
    loader's pre-materialisation estimate charge the same rule."""
    n = 1
    for d in shape:
        n *= int(d)
    width = 2 if dtype in _RESIDENT_FLOAT_DTYPES else _RESIDENT_NONFLOAT_NBYTES.get(dtype, 2)
    return n * width


def _raw_resident_bytes(sf, tname: str) -> int:
    """raw_resident_bytes off an open safetensors file (get_slice —
    shape+dtype, zero tensor bytes touched): the same charge
    load_compressed's raw-tensor loop actually pays for it."""
    sl = sf.get_slice(tname)
    return raw_resident_bytes(sl.get_dtype(), sl.get_shape())



def mtp_head_names(cfg, trunk):
    """{checkpoint key -> head-pack tensor name} for the MTP head's eligible
    Linears, decided from a zero-byte MTPHead skeleton exactly as pack_model
    decides the trunk's from a zero-byte model skeleton. Empty dict when the
    family has no head. The head skeleton is built at the TRUNK's dtype, which
    is what makes swap.eligible's bf16 test mean the same thing here as it
    does at load."""
    from ..serving.mtp import MTPHead, is_supported

    if not is_supported(cfg):
        return {}
    text_cfg = cfg.get_text_config() if hasattr(cfg, "get_text_config") else cfg
    head = MTPHead(text_cfg, trunk)
    from .swap import eligible_linears

    return {f"mtp.{name}.weight": name for name, _, _, _ in eligible_linears(head)}


def default_pack_dir(hf_repo: str, revision: str | None, compression_profile: str | None = None) -> str:
    """`<DRINKME_HOME>/packs/<org>--<model>@<rev>` — the default profile's
    (sip's) pack, the one `serve` / `verify` / `bench` resolve unasked. A pack
    at another profile defaults to `<org>--<model>-<profile>@<rev>` beside
    it (`-gulp@`), which packs.local_packs lists as non-canonical: served
    via --pack-dir only."""
    root = os.environ.get("DRINKME_HOME", os.path.expanduser("~/.cache/drinkme"))
    slug = hf_repo.replace("/", "--")
    if compression_profile and compression_profile != _rx.DEFAULT_PROFILE:
        slug = f"{slug}-{compression_profile}"
    return os.path.join(root, "packs", f"{slug}@{(revision or 'main')[:12]}")


# torch dtype -> the meta.json spelling of what was read (sourceDtype). One
# entry: the codec packs bf16 and nothing else; _pack_one refuses the rest.
_META_DTYPE_OF = {"torch.bfloat16": "bf16"}
_SAFETENSORS_DTYPE_OF = {"torch.bfloat16": "BF16", "torch.float16": "F16", "torch.float32": "F32",
                         "torch.float64": "F64"}


def _pack_one(w, compression_profile: str, name: str | None = None) -> dict:
    """One checkpoint tensor -> one verified pack dict: the radix tensor at
    `compression_profile`, or the raw fallback when radix would expand it
    (radix_pack.pack_weight_radix; both round-trip on the CPU before they
    are returned). THE PACKING BOUNDARY of the source-dtype contract (see
    SOURCE_DTYPE): a tensor that is not bf16 is refused by name here, never
    cast — refuse_checkpoint already said so from the headers at
    pack_model's door; this is the last line, shared by the trunk and the
    MTP head so the two cannot drift on it. A float8 plane never reaches
    here for the same reason."""
    import torch

    if w.dtype != torch.bfloat16:
        seen = _SAFETENSORS_DTYPE_OF.get(str(w.dtype), str(w.dtype))
        reason = source_dtype_refusal_reason([(name or "tensor", seen)], lambda _n: True)
        raise UnsupportedCheckpoint(reason)
    return _rx.pack_weight_radix(w, compression_profile, name=name)


def _source_dtype_meta(seen: set) -> str:
    """meta.json's `sourceDtype`, DERIVED from the torch dtypes the packer
    read (a set of str(dtype)) — never stamped. One dtype, and one the codec
    packs, or the pack is refused. An EMPTY packed population (a toy whose
    every Linear is under the eligibility bar) read nothing: the field then
    names the one dtype the contract admits, since there is no tensor for
    it to disagree with."""
    if not seen:
        return _META_DTYPE_OF["torch.bfloat16"]
    if len(seen) != 1:
        raise UnsupportedCheckpoint(
            f"packed tensors were read at {len(seen)} dtypes ({sorted(seen)}); the codec packs one")
    (dtype,) = seen
    if dtype not in _META_DTYPE_OF:
        raise UnsupportedCheckpoint(unsupported_format(f"a source tensor stored as {dtype}"))
    return _META_DTYPE_OF[dtype]


def _shard_headers(snap: str) -> dict[str, tuple[str, list[int], str]]:
    """{tensor name: (safetensors dtype, shape, shard path)} across every
    shard, from the JSON headers alone (the 8-byte length prefix + header —
    zero weight bytes touched)."""
    import glob
    import struct

    out = {}
    for fpath in sorted(glob.glob(os.path.join(snap, "*.safetensors"))):
        with open(fpath, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(n))
        for key, info in header.items():
            if key == "__metadata__":
                continue
            out[key] = (info["dtype"], list(info["shape"]), fpath)
    return out


def _runtime_dict(p: dict) -> dict:
    """The dict the loader would hand the module for this raw pack dict,
    built on the CPU — the residentBytes measurer (read off this, never
    off the file)."""
    from .swap import to_device_radix, to_device_raw

    if _rx.is_radix_pack(p):
        return to_device_radix(p, "cpu")
    if _rx.is_raw_pack(p):
        return to_device_raw(p, "cpu")
    raise ValueError("_runtime_dict: not a radix or raw pack dict")



class _HeadPacker:
    """The MTP head's sub-pack, written tensor by tensor as head tensors go
    past. Fed by pack_model out of its own shard pass (so a full pack reads
    the shards once) and by pack_mtp_head out of a pass of its own (so an
    ALREADY-PACKED model can gain a head without repacking 401 trunk tensors
    — the retrofit case, and the reason this is a class and not eight lines
    inline).

    The writer is created on the FIRST head tensor, never in __init__:
    is_supported(cfg) says the FAMILY can carry a head, the checkpoint says
    whether this one does, and a checkpoint that does not must leave no empty
    sub-pack behind."""

    def __init__(self, cfg, trunk, out: str, compression_profile: str, progress=print,
                 replace: bool = False) -> None:
        self.want = mtp_head_names(cfg, trunk)
        self.out = mtp_pack_dir(out)  # inside the trunk's STAGING dir when fed by pack_model
        self.compression_profile, self.progress = compression_profile, progress
        self.replace = replace
        self.writer = None
        self.bpws, self.bpw_by_name, self.numels = [], {}, []
        self.raw_bytes = self.packed_bytes = self.resident_bytes = 0
        self.unpacked_bytes = self.unpacked_count = 0
        self.radix_count = self.raw_fallback_count = 0
        self.encode_seconds = 0.0
        self.source_dtypes: set[str] = set()  # str(torch dtype) of every tensor packed

    def offer(self, tname: str, sf) -> bool:
        """Consume `tname` if it belongs to the head. True = handled (the
        caller must not also pack it), False = not ours."""
        name = self.want.get(tname)
        if name is None:
            if not tname.startswith("mtp."):
                return False
            # a head tensor the codec does not touch (the norms, and any
            # projection under the eligibility bar): raw at load exactly as
            # the trunk's are, but still resident, so the residency charge
            # must carry it
            w = sf.get_tensor(tname)
            self.unpacked_bytes += w.numel() * 2
            self.unpacked_count += 1
            del w
            return True
        if self.writer is None:
            self.writer = PackWriter(self.out, list(self.want.values()), replace=self.replace)
        import time

        t0 = time.time()
        w = sf.get_tensor(tname)
        self.source_dtypes.add(str(w.dtype))
        p = _pack_one(w, self.compression_profile, name=tname)
        del w
        self.encode_seconds += time.time() - t0
        if _rx.is_radix_pack(p):
            self.radix_count += 1
        else:
            self.raw_fallback_count += 1
        self.writer.add(name, p)
        self.bpws.append(p["bpw"])
        self.numels.append(int(p["R"]) * int(p["C"]))
        self.bpw_by_name[name] = round(p["bpw"], 4)
        # rawBytes: what the head costs served raw at the checkpoint's own
        # width, bf16
        self.raw_bytes += 2 * p["R"] * p["C"]
        self.packed_bytes += os.path.getsize(self.writer.file_path(name))
        # what the fit guard will be charged, measured on the very dict the
        # loader hands CompressedLinear
        self.resident_bytes += resident_bytes(_runtime_dict(p))
        self.progress(f"  packed mtp/{name}  {_describe(p)}  bpw={p['bpw']:.3f}")
        del p
        return True

    def finish(self, hf_repo: str, revision: str | None,
               source: dict | None = None) -> dict | None:
        """Commit the sub-pack's meta.json (its own commit point, exactly as
        the trunk's is its own). None when the checkpoint carried no head.
        `source` is the trunk's checkpoint identity: the
        head's norms stream raw from the same snapshot the trunk's raw half
        does, so the sub-pack is bound to it the same way."""
        if self.writer is None:
            return None
        meta = {
            "module": MTP_SUBDIR,
            "hfRepo": hf_repo,
            "revision": revision,
            "source": source,
            "dtype": "bf16",
            "sourceDtype": _source_dtype_meta(self.source_dtypes),  # what offer() read
            "tensorCount": len(self.want),
            "meanBpw": round(float(np.mean(self.bpws)), 3),
            "weightedBpw": round(weighted_bpw(self.bpws, self.numels), 3),  # the head's packed population
            "bpwByTensor": self.bpw_by_name,
            # the three numbers the diet is judged on, all measured:
            "rawBytes": self.raw_bytes,            # bf16 bytes of what we packed
            "packedBytes": self.packed_bytes,      # the npz files on disk
            "residentBytes": self.resident_bytes,  # what to_device puts on the GPU
            # mtp.* tensors the codec does not touch: raw at load, so the
            # residency charge adds them back
            "unpackedBytes": self.unpacked_bytes,
            "unpackedTensorCount": self.unpacked_count,
            **_profile_meta(self.compression_profile),
            "radixTensorCount": self.radix_count,
            "rawFallbackTensorCount": self.raw_fallback_count,
            "radixEncodeSeconds": round(self.encode_seconds, 1),
        }
        self.writer.finish(meta)
        self.progress(
            f"   mtp head: {len(self.want)} tensors packed "
            f"({self.raw_bytes / 2**20:.1f} MiB bf16 -> "
            f"{self.resident_bytes / 2**20:.1f} MiB resident, mean "
            f"{meta['meanBpw']} bpw) + {self.unpacked_count} raw "
            f"({self.unpacked_bytes / 2**10:.1f} KiB)")
        return meta

    def abort(self) -> None:
        """Discard a half-written sub-pack (its writer's staging dir)."""
        if self.writer is not None:
            self.writer.abort()


def _head_meta_keys(meta: dict, head_meta: dict) -> None:
    """The head's headline numbers, echoed into the TRUNK's meta.json for
    provenance. Additive keys only: the loader reads meta.json for
    formatVersion/tensors/sha256 and the manifest and ignores the rest."""
    meta["mtpTensorCount"] = head_meta["tensorCount"]
    meta["mtpMeanBpw"] = head_meta["meanBpw"]
    meta["mtpResidentBytes"] = head_meta["residentBytes"] + head_meta["unpackedBytes"]
    meta["mtpRawBytes"] = head_meta["rawBytes"] + head_meta["unpackedBytes"]



def pack_mtp_head(hf_repo: str, revision: str | None = None,
                  out: str | None = None, progress=print,
                  compression_profile: str | None = None, replace: bool = False) -> str | None:
    """Write ONLY the `mtp/` sub-pack, into an existing (or new) pack dir.

    The retrofit path: the shipped 8B and 27B packs cost hours to build and
    are sha256-hashed tensor by tensor; giving them a head must not mean
    re-packing the trunk. This reads the checkpoint's `mtp.*` tensors alone,
    writes `<out>/mtp/`, and updates the trunk's meta.json with the head's
    headline numbers if there is one to update — the trunk's own npz files
    and their recorded hashes are never opened for writing.

    Returns the sub-pack path, or None when the checkpoint carries no head.
    The sub-pack is staged and published whole (PackWriter); an existing
    `mtp/` is refused unless `replace`.
    """
    import gc
    import glob
    import time

    from ..serving.checkpoint import snapshot_dir
    from .identity import checkpoint_identity

    t0 = time.time()
    compression_profile = _rx.resolve_compression_profile(compression_profile)
    # ONE snapshot: resolve it first, and read the config
    # from it — never AutoConfig(hf_repo, revision) over the network, which
    # can name a different commit than the shards below on the day a branch
    # moves.
    snap = snapshot_dir(hf_repo, revision)
    refuse_checkpoint(hf_repo, snap)  # torch-free, before torch is imported below
    from safetensors import safe_open
    from transformers import AutoConfig

    from ..arms import skeleton

    source = checkpoint_identity(snap, hf_repo)
    cfg = AutoConfig.from_pretrained(snap)
    model = skeleton(cfg)
    head = _HeadPacker(cfg, model, out or default_pack_dir(hf_repo, revision, compression_profile),
                       compression_profile, progress, replace=replace)
    del model
    gc.collect()
    if not head.want:
        progress(f"{hf_repo}: {getattr(cfg, 'model_type', '?')} has no MTP head")
        return None
    out = out or default_pack_dir(hf_repo, revision, compression_profile)
    files = sorted(glob.glob(os.path.join(snap, "*.safetensors")))
    if not files:
        raise FileNotFoundError(f"no *.safetensors under {snap}")
    try:
        for fpath in files:
            with safe_open(fpath, framework="pt") as sf:
                for tname in sf.keys():
                    if tname.startswith("mtp."):
                        head.offer(tname, sf)
            gc.collect()
        head_meta = head.finish(hf_repo, revision, source=source)
    except BaseException:
        head.abort()
        raise
    if head_meta is None:
        progress(f"{hf_repo} carries no mtp.* tensors — no sub-pack written")
        return None
    trunk_meta_path = os.path.join(out, "meta.json")
    if os.path.isfile(trunk_meta_path):
        with open(trunk_meta_path) as f:
            meta = json.load(f)
        _head_meta_keys(meta, head_meta)
        _write_meta(out, meta)
    progress(f"-> {head.out}  ({head_meta['tensorCount']} tensors, mean "
             f"{head_meta['meanBpw']} bpw, {time.time() - t0:.1f}s)")
    return head.out



def pack_model(hf_repo: str, revision: str | None = None, out: str | None = None,
               progress=print, compression_profile: str | None = None,
               mtp: bool = True, replace: bool = False, embed: bool = True) -> str:
    """`drinkme pack`: stream a BF16 HF checkpoint into a verified pack dir.

    `compression_profile` (`--sip` / `--gulp`): the profile every eligible tensor is
    encoded at — sip (the default; radix_pack.resolve_compression_profile also reads
    DRINKME_COMPRESSION_PROFILE), gulp (the fit profile), balanced (hidden).
    The eligible population is eligible_linears'; a tensor radix would
    expand is stored raw and counted (rawFallbackTensorCount). An FP8
    checkpoint is refused by name before torch is imported
    (refuse_checkpoint), and so is one that stores an eligible tensor as
    anything but BF16 (the source-dtype contract, SOURCE_DTYPE: refused by
    tensor name, never cast). The MTP head sub-pack (when a checkpoint has one)
    is packed the same way — same profile, same fallback, same writer.

    THE MEMORY CONTRACT: peak host RAM scales with the largest SHARD, never
    the model — HF checkpoints shard at ~4-5 GB by convention, so packing
    stays under ~9 GB at ANY model size (measured: Qwen3-4B peaks at 5.0
    GiB = one 3.9 GB shard's mmap residency + one tensor's transients; the
    27B at 8.9 GB). A meta-device skeleton (zero weight
    bytes) decides eligibility and naming, weights stream one tensor at a
    time straight from the safetensors shards, and every pack tensor hits
    disk the moment it verifies (the encoder raises on any round-trip
    failure).

    Eligibility and names come from the SAME skeleton load_compressed builds,
    so pack keys == what the loader will ask for, by construction; the
    key-set check below turns "checkpoint lacks an eligible weight" into a
    named refusal instead of a half-pack. Only demand a skip when the source
    shipped something to skip: a tied lm_head is simply absent from `want`,
    so a checkpoint that ships the redundant copy is skipped silently and one
    that omits it owes nothing (never assert an export detail as a
    correctness property).

    THE MTP HEAD (`mtp=True`): a checkpoint that carries a trained
    MTP draft head gets that head's eligible Linears packed too, into the
    `mtp/` sub-pack (see MTP_SUBDIR above) — same codec, same eligibility
    rule, same writer, one pass over the same shards. `mtp=False` writes
    the default pack exactly. Either way the TRUNK's files are untouched by the
    option: PackWriter assigns filenames by sorted position within its own
    name set, and the head's names live in a different writer, so t0000.npz
    means the same tensor with the head on or off.

    TRANSACTIONAL: the whole pack — trunk and `mtp/` —
    is staged in `<out>.staging-<pid>` and published by one rename when
    everything verified; an existing pack at `out` is refused unless
    `replace`, and then kept until the new one is in place. Nothing at
    `out` changes on any failure.

    SELF-CONTAINED (`embed=True`, the only supported shape): the pack also
    carries the checkpoint's small files and every tensor the loaders read
    raw, in `checkpoint/` (write_embedded), so it loads with no snapshot
    and no network. `embed=False` exists only for tests that need to build
    a pack without one, to prove the loaders refuse it — there is no CLI
    flag for it, and the loaders refuse a pack without `embedded` at load
    (serving/checkpoint.resolve_pack_source): there is no snapshot for a
    refused pack to fall back to.

    Pure CPU — no GPU needed to pack. Returns the pack path."""
    import gc
    import glob
    import time

    from ..serving.checkpoint import snapshot_dir
    from .identity import checkpoint_identity

    # THE FRONT DOOR, first: a machine with no C++ compiler (or one that
    # fails to build the native encoder) refuses here, before the Hub
    # snapshot below — never partway through a multi-GB download. See
    # radix_pack.refuse_unless_encoder_available.
    _rx.refuse_unless_encoder_available(lead_in="drinkme pack")
    t0 = time.time()
    # ONE immutable snapshot: resolved before anything
    # else, config read FROM it, its identity recorded as meta.json's
    # `source` — for a hub repo the commit sha the cache actually resolved
    # (never the caller's branch or None), for a local dir a content digest
    # (codec/identity.py). The loader streams the raw half from exactly this
    # snapshot, or refuses.
    compression_profile = _rx.resolve_compression_profile(compression_profile)  # raises on an unknown profile
    snap = snapshot_dir(hf_repo, revision)
    # THE DOOR: an FP8 checkpoint (any float8 plane in the shard headers,
    # or an fp8 quantization_config), or one storing a packable tensor as
    # anything but BF16 (the header estimate of eligibility; the exact set
    # is checked again below once the skeleton names it), is refused by
    # name here, from config.json and the headers alone — before torch,
    # transformers or a shard is imported or opened, so `drinkme pack` on
    # one prints the line and nothing else.
    refuse_checkpoint(hf_repo, snap)
    from transformers import AutoConfig

    from ..arms import ckpt_to_skel, skeleton, vision_tower_paths
    from .swap import eligible_linears

    source = checkpoint_identity(snap, hf_repo)
    cfg = AutoConfig.from_pretrained(snap)
    # the served tree, vision tower included where the architecture serves
    # one (arms._VISION_TOWERS): its Linears are packed like any other, the
    # rest embedded raw, and meta.json's `vision` block records it
    model = skeleton(cfg, vision=True)
    to_skel = ckpt_to_skel(cfg, vision=True)  # wrapped checkpoints name text weights differently
    # checkpoint key -> pack name (identical strings; kept as a mapping
    # so the pack name stays the loader-side contract if they ever diverge)
    want = {f"{name}.weight": name for name, _, _, _ in eligible_linears(model)}
    # the source-dtype contract over the skeleton's EXACT eligible set (the
    # door above applied it over the header estimate): a checkpoint that
    # stores any of these as anything but BF16 is refused by name here,
    # before a shard is opened — never cast (SOURCE_DTYPE).
    refuse_checkpoint(hf_repo, snap, eligible=lambda tname: (to_skel(tname) or "") in want)
    out = out or default_pack_dir(hf_repo, revision, compression_profile)
    for stale in sorted(glob.glob(f"{out}.staging-*")):
        if stale != f"{out}.staging-{os.getpid()}":
            progress(f"note: leftover staging dir from an earlier run: {stale} "
                     "— safe to delete")
    writer = PackWriter(out, list(want.values()), replace=replace)
    head = None
    try:
        # The head's eligible Linears, from the head skeleton the loader
        # builds (empty for every family without an MTP head — i.e.
        # everything but qwen3_5, so this costs those checkpoints nothing).
        # Its sub-pack is written INSIDE the trunk's staging dir, so the
        # trunk's one publishing rename carries it.
        head = (_HeadPacker(cfg, model, writer.staging, compression_profile, progress)
                if mtp else None)
        tied = tied_output_weight(model)
        paths = vision_tower_paths(cfg)
        vision = None if paths is None else {"tower": paths[0], "paths": list(paths),
                                             "imageTokenId": int(cfg.image_token_id)}
        del model
        gc.collect()
        return _pack_model_into(writer, head, snap, source, want, to_skel, hf_repo,
                                revision, compression_profile, progress, t0, embed, tied,
                                vision)
    except BaseException:
        if head is not None:
            head.abort()
        writer.abort()
        raise



# ------------------------------------------------ what is not served ------
# THE one statement of what this release serves, and the one line it says
# when handed anything else. Every by-name refusal for a quantization
# layout or dtype this build does not implement — an int4/fp4 family
# quant_method, an unnamed quantized checkpoint, an unrecognized per-tensor
# `dtype` scalar in a pack — ends with unsupported_format()'s exact wording
# (check, pack and serve all route through it), so what IS supported is
# stated once and cannot drift between call sites. No "yet", no promise of
# more.
SUPPORTED_FORMATS = ("bf16 (served losslessly)",)

# An FP8 checkpoint gets its own line — the same line at every door it can
# reach (`drinkme pack`, the raw-trunk loader, `check`, `serve --model`,
# `bench`), prefixed by the checkpoint's name: nothing fp8 may pack, load
# or serve in this release.
FP8_REFUSAL = "FP8 checkpoints are not supported in this release (bf16 only)"
FP8_SAFETENSORS_DTYPES = ("F8_E4M3", "F8_E5M2")  # the float8 dtype strings a shard header can carry
_FP8_WORDS = ("fp8", "float8", "e4m3", "e5m2")     # how a quantization_config spells fp8, any vendor
FP8_PACK_DTYPE = "fp8_e4m3"                        # an FP8 pack's tensor/meta dtype scalar


class UnsupportedCheckpoint(ValueError):
    """The one-line refusal for a checkpoint (or a pack cut from one) this
    release does not serve; `str(e)` is the line, nothing else."""


def unsupported_format(seen: str) -> str:
    """The one unsupported-format message. `seen` names the condition (a
    dtype, a quant_method, a layout detail) — the caller's own message
    should already have said so too, in its own words; this is the shared
    tail every refusal ends with, not a replacement for the specifics."""
    return (f"{seen} is not a format this build supports. Supported: "
            f"{'; '.join(SUPPORTED_FORMATS)}. Nothing else is supported.")


def is_fp8_quantization_config(qc) -> bool:
    """Does config.json's quantization_config describe an fp8 checkpoint,
    in any vendor's spelling — `quant_method: "fp8"` (Qwen), `fbgemm_fp8`
    (Meta), a `fmt: e4m3`, a `float8` anywhere in it? A layout the words do
    not name (compressed-tensors' `type: float, num_bits: 8`) is still
    caught: its shards carry F8_* planes, and the door checks those too."""
    if not isinstance(qc, dict) or not qc:
        return False
    text = json.dumps(qc, sort_keys=True, default=str).lower()
    return any(word in text for word in _FP8_WORDS)


def checkpoint_refusal_reason(config, dtypes) -> str | None:
    """The reason (no name prefix) a checkpoint is refused, or None: FP8 —
    any F8_* plane among `dtypes` (its safetensors header dtype strings) or
    an fp8 quantization_config in `config` (config.json as a dict) — says
    FP8_REFUSAL; any other quantization_config names its quant_method in
    the same shape."""
    if any(d in FP8_SAFETENSORS_DTYPES for d in dtypes):
        return FP8_REFUSAL
    qc = (config or {}).get("quantization_config") if isinstance(config, dict) else None
    if is_fp8_quantization_config(qc):
        return FP8_REFUSAL
    if isinstance(qc, dict) and qc:
        method = qc.get("quant_method")
        return (f"quantized checkpoints ({'quant_method=' + repr(str(method)) if method else 'a quantization_config'}) "
                "are not supported in this release (bf16 only)")
    return None


def refusal_for_checkpoint(name: str, config, dtypes) -> str | None:
    """`<name>: <reason>` — the line — or None when the checkpoint is a
    bf16 one this release packs and serves."""
    reason = checkpoint_refusal_reason(config, dtypes)
    return None if reason is None else f"{name}: {reason}"


def refusal_for_dtype(dtype) -> str:
    """The line for a pack tensor whose scalars carry a `dtype` (the bf16
    codec's tensors carry none): an FP8 pack's `fp8_e4m3` is FP8_REFUSAL,
    anything else the shared unsupported-format tail."""
    if dtype == FP8_PACK_DTYPE:
        return FP8_REFUSAL
    return unsupported_format(f"dtype {dtype!r}")


def is_fp8_pack_meta(meta: dict) -> bool:
    """Was this pack cut from an FP8 checkpoint?
    Its meta says so in `sourceDtype` / `dtype`."""
    return meta.get("sourceDtype") == FP8_PACK_DTYPE or meta.get("dtype") == FP8_PACK_DTYPE


# The source-dtype contract (docs/pack-format.md): every tensor the codec
# packs is read as the BF16 bits the checkpoint released, and nothing is ever
# cast on the way in — a wider or narrower source would be rounded, and a
# rounded tensor served under a "lossless" label is the one thing this
# product must not do. So an eligible tensor stored as anything else refuses
# the checkpoint BY NAME (tensor, dtype, checkpoint), at every door: `drinkme check`
# (from the hub's headers), refuse_checkpoint (every loader and packer), and
# _pack_one (the packing boundary itself, the last line). Tensors the codec
# does not pack — embeddings, norms, biases, sub-bar Linears — are outside
# the contract: they stream raw and take the loader's bf16 rule, exactly as
# from_pretrained(dtype=bf16) would.
SOURCE_DTYPE = "BF16"  # the safetensors header dtype string of a packable tensor


def header_eligible(name: str, shape, config: dict) -> bool:
    """The header-only ESTIMATE of swap.eligible_linears — which tensors the
    codec will pack — from a tensor's name and shape alone (no torch, no
    skeleton): 2D, cols % 4 == 0, min dim >= 1024, not an embedding table,
    not a tied lm_head. dtype is deliberately NOT a term: the source-dtype
    rule is applied to exactly what this names. check.assess estimates
    its packable mass with this; refuse_checkpoint applies the dtype rule
    with it when the caller has nothing more exact (pack_model does, and
    checks the skeleton's own set as well)."""
    if len(shape) != 2:
        return False
    r, c = int(shape[0]), int(shape[1])
    if c % 4 != 0 or min(r, c) < 1024:
        return False
    low = name.lower()
    if "embed" in low:
        return False  # embedding tables: lookup, not GEMV — pack skips them
    if (config or {}).get("tie_word_embeddings") and "lm_head" in low:
        return False  # the tied-head rule (swap.py)
    return True


def source_dtype_refusal_reason(tensors, eligible) -> str | None:
    """The reason (no name prefix) the source-dtype contract refuses a
    checkpoint, or None. `tensors`: (name, safetensors dtype string) pairs;
    `eligible(name)`: whether the codec would pack that tensor. The FIRST
    eligible tensor stored as anything but BF16 is named, with its dtype —
    the one decision check.assess, refuse_checkpoint and the packing boundary
    all read."""
    for tname, dtype in tensors:
        if dtype != SOURCE_DTYPE and eligible(tname):
            return (f"{tname} is stored as {dtype}, not {SOURCE_DTYPE} — the codec packs a "
                    f"BF16 release's bits as released and never converts a source tensor. "
                    f"{unsupported_format(f'a source tensor stored as {dtype}')}")
    return None


def read_config_json(snap: str) -> dict:
    """A snapshot's config.json as a dict — json alone, no transformers."""
    with open(os.path.join(snap, "config.json")) as f:
        return json.load(f)


def refuse_checkpoint(name: str, snap: str, eligible=None) -> None:
    """The door: raise UnsupportedCheckpoint with the line for a local
    snapshot this release does not serve, from its config.json and the
    shard headers (kilobytes; zero weight bytes; no torch). `name` is what
    the caller called it — the HF repo id or the local path.

    Two rules, in order: the FP8/quantization door (refusal_for_checkpoint),
    then the source-dtype contract (source_dtype_refusal_reason) over the
    tensors `eligible(name)` says the codec packs — the header estimate
    (header_eligible) when the caller passes nothing, which is what every
    loader's door uses; pack_model passes its skeleton's exact set too."""
    config = read_config_json(snap) if os.path.isfile(os.path.join(snap, "config.json")) else {}
    headers = _shard_headers(snap)
    dtypes = [dtype for dtype, _shape, _fpath in headers.values()]
    line = refusal_for_checkpoint(name, config, dtypes)
    if line is not None:
        raise UnsupportedCheckpoint(line)
    if eligible is None:
        def eligible(tname: str) -> bool:
            return header_eligible(tname, headers[tname][1], config)
    reason = source_dtype_refusal_reason(((t, d) for t, (d, _s, _f) in headers.items()), eligible)
    if reason is not None:
        raise UnsupportedCheckpoint(f"{name}: {reason}")


def _profile_meta(compression_profile: str) -> dict:
    """The three profile fields every bf16 pack's meta.json (trunk and mtp/)
    carries: the profile IS the pack's name (`sip` / `gulp`; the bf16 codec
    has no name of its own), and a bench record's compression.profile is
    read off the loaded arm (arms.arm_compression_profile) or the loaded pack, never
    typed. A pack's raw fallbacks are counted in meta.json rather than
    hidden in the name."""
    return {"profile": compression_profile, "profileWidths": list(_rx.widths_of(compression_profile)),
            "blockSize": _rx.RADIX_BLOCK}


def _describe(p: dict) -> str:
    """One tensor's kind for the pack log: `sip widths=[3, 8]` /
    `RAW (coding would expand it)`."""
    if _rx.is_radix_pack(p):
        return f"{p['profile']} widths={list(p['widths'])}"
    return "RAW (coding would expand it)"


def _pack_model_into(writer, head, snap, source, want, to_skel, hf_repo, revision,
                     compression_profile, progress, t0, embed: bool = True,
                     tied: str | None = None, vision: dict | None = None) -> str:
    """pack_model's streaming body, under its transactional guard. Every
    tensor in `want` (skeleton sname -> pack name, keyed the way
    eligible_linears always has been) goes through the radix codec at
    `compression_profile`. `vision` ({"tower", "paths", "imageTokenId"})
    names the served vision tower when the tree holds one (`tower` the ViT,
    `paths` every subtree it spans: gemma-4's projection sits beside its
    ViT): its tensors are tallied as they pass, into meta.json's `vision`
    block."""
    import gc
    import glob
    import time

    from safetensors import safe_open

    out = writer.path
    files = sorted(glob.glob(os.path.join(snap, "*.safetensors")))
    if not files:
        raise FileNotFoundError(f"no *.safetensors under {snap}")
    bpws, numels = [], []
    packed_disk_bytes = 0  # the trunk's own npz sum, tallied as written
    # what to_device actually puts on the device for each packed tensor
    # — NOT the npz size (the radix runtime dict carries the padded palette
    # tables and, for 3+ tiers, the per-block schedule). Same measurer
    # _HeadPacker.offer uses.
    packed_resident_bytes = 0
    raw_resident_bytes_total = 0  # what the loader's raw-tensor loop pays
    raw_tensor_count = 0
    # counts + bytes broken out by what each packed tensor actually is —
    # radix or the raw fallback — so meta.json says which rather than one
    # blended number
    radix_tensor_count = radix_disk_bytes = 0
    raw_fallback_count = raw_fallback_bytes = 0
    encode_seconds = 0.0
    source_dtypes: set[str] = set()  # str(torch dtype) of every tensor packed
    # the vision tower's share of the above: tensors (packed and raw) and
    # the resident bytes the loader holds for them
    vpaths = None if vision is None else tuple(vision["paths"])
    v_count = v_packed = v_bytes = 0
    for fpath in files:
        with safe_open(fpath, framework="pt") as sf:
            for tname in sf.keys():
                sname = to_skel(tname)
                # looked up by SKELETON name (eligible_linears')
                name = want.get(sname) if sname else None
                if name is None:
                    # head.offer's return says whether IT resident-charges
                    # this tensor (a packed or raw-but-unpacked head tensor);
                    # everything else that reaches here with a real skeleton
                    # name (sname is not None — a tower the served tree
                    # never builds, arms._NON_TEXT_TOWERS', is never
                    # loaded at all) is
                    # exactly what load_compressed's raw-tensor loop streams.
                    consumed = head.offer(tname, sf) if head is not None else False
                    if not consumed and sname is not None:
                        nbytes = _raw_resident_bytes(sf, tname)
                        raw_resident_bytes_total += nbytes
                        raw_tensor_count += 1
                        if vpaths is not None and under(sname, vpaths):
                            v_count += 1
                            v_bytes += nbytes
                    continue  # raw tensor: embedded (write_embedded) or streamed at load
                w = sf.get_tensor(tname)
                source_dtypes.add(str(w.dtype))
                t_rx = time.time()
                p = _pack_one(w, compression_profile, name=tname)
                encode_seconds += time.time() - t_rx
                writer.add(name, p)
                disk_bytes = os.path.getsize(writer.file_path(name))
                packed_disk_bytes += disk_bytes
                nbytes = resident_bytes(_runtime_dict(p))
                packed_resident_bytes += nbytes
                if vpaths is not None and under(name, vpaths):
                    v_count += 1
                    v_packed += 1
                    v_bytes += nbytes
                bpws.append(p["bpw"])
                numels.append(int(p["R"]) * int(p["C"]))
                if _rx.is_radix_pack(p):
                    radix_tensor_count += 1
                    radix_disk_bytes += disk_bytes
                    progress(f"  packed {name}  {_describe(p)}  bpw={p['bpw']:.3f}  {p['R']}x{p['C']}")
                else:
                    raw_fallback_count += 1
                    raw_fallback_bytes += disk_bytes
                    progress(f"  packed {name}  {_describe(p)}  bpw={p['bpw']:.3f}  {p['R']}x{p['C']}")
                del w, p
        gc.collect()  # one shard's transients released before the next opens
    missing = sorted(set(want.values()) - writer.written)
    if missing:
        raise ValueError(
            f"checkpoint is missing {len(missing)} eligible weights the "
            f"loader will ask for: {missing[:8]}"
        )
    meta = {
        "hfRepo": hf_repo,
        "revision": revision,  # the caller's coordinate (names the pack dir)
        "source": source,      # the resolved, immutable one
        # the pack's coded dtype, and the checkpoint's released one it
        # serves exactly: bf16, both — the lossless codec serves a bf16
        # release as released. Kept apart so a codec over another source
        # dtype, if one is ever built, can say so. sourceDtype is DERIVED
        # from the dtypes the packer read (never stamped): _pack_one refused
        # anything but bf16, so this can only ever say bf16 or raise.
        "dtype": "bf16",
        "sourceDtype": _source_dtype_meta(source_dtypes),
        # the profile the pack's tensors (trunk and head) are encoded at —
        # the pack's name; docs/pack-format.md
        **_profile_meta(compression_profile),
        "tensorCount": len(want),
        "meanBpw": round(float(np.mean(bpws)), 3) if bpws else None,
        # weightedBpw: total encoded payload
        # bits over total weights of the PACKED population — the trunk's
        # radix + raw-fallback tensors, NOT the raw remainder and NOT the
        # MTP head (mtp/meta.json carries its own). meanBpw above is the
        # unweighted mean over tensors. The record and /v1/models carry both
        # by the lexicon's names: weightedBpw as bitsPerWeight, meanBpw as
        # meanTensorBitsPerWeight (docs/pack-format.md).
        "weightedBpw": round(weighted_bpw(bpws, numels), 3) if bpws else None,
        "packSeconds": round(time.time() - t0, 1),
        # the trunk's true resident footprint — what to_device puts on
        # the GPU for every packed tensor plus every tensor the loader
        # streams raw (untied embed_tokens, norms, anything below the
        # eligibility bar) — the same meaning as the MTP sub-pack's
        # residentBytes (mtp/meta.json). packedBytes is the disk figure.
        "packedBytes": packed_disk_bytes,
        "residentBytes": packed_resident_bytes + raw_resident_bytes_total,
        # what is radix vs raw-fallback vs streamed raw, by count and by
        # bytes — radix/raw-fallback bytes are the on-disk npz sums
        # (packedBytes' split), rawResidentBytes is what the loader's
        # raw-tensor loop pays reading the checkpoint directly (the same
        # number folded into residentBytes above).
        "radixTensorCount": radix_tensor_count,
        "radixBytes": radix_disk_bytes,
        "rawFallbackTensorCount": raw_fallback_count,
        "rawFallbackBytes": raw_fallback_bytes,
        "rawTensorCount": raw_tensor_count,
        "rawResidentBytes": raw_resident_bytes_total,
        "radixEncoder": _rx._encoder(),
        "radixEncodeSeconds": round(encode_seconds, 1),
    }
    if vpaths is not None and v_count:
        # THE commit point for image input, like `embedded` for the raw
        # half: a pack serves images only if it says it carries the tower,
        # and the loader refuses to build a tower for one without the
        # block (re-pack). The tower's bytes are inside residentBytes above; this is their share,
        # which the fit leaves out when images are off (fit.served_resident_bytes).
        meta["vision"] = {**vision, "tensorCount": v_count, "packedTensorCount": v_packed,
                          "residentBytes": v_bytes}
    head_meta = head.finish(hf_repo, revision, source=source) if head else None
    if head_meta is not None:
        _head_meta_keys(meta, head_meta)
    if embed:
        # inside the trunk's staging dir, so the one publishing rename
        # carries it with the rest; the head's packed keys are the ones
        # _HeadPacker wrote, none when the checkpoint has no head
        head_weights = set(head.want) if head_meta is not None else set()
        meta["embedded"] = write_embedded(
            snap, embedded_dir(writer.staging),
            embedded_tensor_names(snap, to_skel, set(want), head_weights, tied), progress)
    writer.finish(meta)
    progress(f"-> {out}  ({len(want)} tensors: {radix_tensor_count} {compression_profile} + "
             f"{raw_fallback_count} raw fallback, {raw_tensor_count} raw, "
             f"{meta['weightedBpw']} bpw weighted ({meta['meanBpw']} mean over tensors), {meta['packSeconds']}s; encoder {meta['radixEncoder']}, "
             f"{meta['radixEncodeSeconds']}s encoding)")
    return out
