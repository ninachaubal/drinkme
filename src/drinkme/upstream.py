"""Upstream verification: a pack rebuilt into its source checkpoint's own
files, byte for byte, each compared with the SHA-256 the Hugging Face Hub
publishes for that file at the pack's bound commit
(docs/pack-format.md#upstream-verification).

The lossless guarantee is established at pack time, on the packer's
machine: the encoder decodes every tensor back and compares it with the
source. Loading rechecks the pack's own file hashes, not the source. A pack
downloaded from someone else would otherwise ask its user to trust the
packer. It does not have to: the Hub records the SHA-256 of every LFS file
at every commit (a safetensors shard is always LFS), and a pack carries
everything a shard is made of — the shard's header bytes
(`checkpoint/source-identity.json`), its packed Linears (decoded here) and
every other tensor byte for byte (`checkpoint/remainder.safetensors`).
Rebuilt in header order and streamed into sha256, nothing written out, a
shard either hashes to the publisher's digest or it does not. The embedded
small files (config, tokenizer, chat template) are compared the same way:
an LFS file by its SHA-256, any other by its git blob id.

Per file the verdict is MATCH, MISMATCH, or NOT COVERED with the reason and
the exact tensors (a tensor the pack does not carry, say). Only a verdict
of MATCH for every file is a pass. The result is written to the pack
directory as `upstream-verified.json` (RECEIPT_FILE), bound to the pack's
manifest digests, so it runs once per pack: a pack whose file hashes
changed since no longer has a current receipt.

Torch-free: the native decoder when it is built (radix_native), else
NumPy's (radix.decode).
"""

from __future__ import annotations

import datetime
import hashlib
import json
import os
import struct
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field

import numpy as np

MATCH, MISMATCH, NOT_COVERED = "MATCH", "MISMATCH", "NOT COVERED"
RECEIPT_FILE = "upstream-verified.json"
RECEIPT_VERSION = 1
# one Hub metadata request with every file's LFS hash; a large repo's
# listing takes longer than suggest.HUB_TIMEOUT_S's single probe
HUB_TIMEOUT_S = 30.0
_CHUNK = 16 << 20

# model_type -> (checkpoint prefix, skeleton prefix): arms._TEXT_WRAP, copied
# rather than imported because arms imports torch at module scope and this
# module runs on the MLX lane too; tests/test_upstream.py pins the two equal.
_TEXT_WRAP: dict[str, tuple[str, str]] = {
    "qwen3_5": ("model.language_model.", "model."),
    "qwen3_5_text": ("model.language_model.", "model."),
}


class UpstreamUnavailable(RuntimeError):
    """The Hub's file listing for the source commit could not be read:
    offline, HF_HUB_OFFLINE, a timeout, a gated or removed repository."""


@dataclass
class FileVerdict:
    name: str
    verdict: str        # MATCH | MISMATCH | NOT COVERED
    kind: str           # "weights" (a rebuilt shard) or "file" (an embedded small file)
    size: int | None
    upstream: str | None  # the publisher's digest: sha256, or "git:<blob id>"
    rebuilt: str | None   # the same digest over the pack's bytes
    reason: str | None = None
    tensors: list[str] = field(default_factory=list)  # the tensors a NOT COVERED names
    note: str | None = None

    def line(self) -> str:
        head = f"{self.verdict:<11} {self.name}"
        if self.verdict == MATCH:
            return head + (f"  sha256 {self.upstream[:16]}…" if self.kind == "weights"
                           else "") + (f" ({self.note})" if self.note else "")
        return f"{head} — {self.reason}"


@dataclass
class Verdict:
    repo: str
    commit: str
    files: list[FileVerdict]
    seconds: float
    decoder: str

    @property
    def verdict(self) -> str:
        states = {f.verdict for f in self.files}
        if MISMATCH in states:
            return MISMATCH
        if NOT_COVERED in states or not self.weights():
            return NOT_COVERED
        return MATCH

    @property
    def ok(self) -> bool:
        return self.verdict == MATCH

    def weights(self) -> list[FileVerdict]:
        return [f for f in self.files if f.kind == "weights"]

    def mismatched(self) -> list[FileVerdict]:
        return [f for f in self.files if f.verdict == MISMATCH]

    def summary(self) -> str:
        """The one verdict line: how many weight files matched the
        publisher's hashes, and the first file that did not."""
        w = self.weights()
        matched = sum(1 for f in w if f.verdict == MATCH)
        nbytes = sum(f.size or 0 for f in w)
        head = (f"upstream {self.verdict}: {matched} of {len(w)} weight files "
                f"({nbytes / 1e9:.2f} GB) rebuilt from the pack hash to "
                f"{self.repo}@{self.commit[:12]}'s published sha256")
        others = [f for f in self.files if f.kind == "file"]
        if others:
            head += (f"; {sum(1 for f in others if f.verdict == MATCH)} of {len(others)} "
                     "embedded files match")
        bad = next((f for f in self.files if f.verdict != MATCH), None)
        if bad is not None:
            head += f"; {bad.verdict} {bad.name}: {bad.reason}"
        return head + f" ({self.seconds:.1f}s, {self.decoder} decoder)"


# ------------------------------------------------------------- the Hub --

def hub_listing(repo: str, commit: str, timeout: float = HUB_TIMEOUT_S) -> dict[str, dict]:
    """{file name -> {"size", "sha256" (LFS files; else None), "blobId"}}
    for every file of `repo` at `commit`, from one Hub metadata request
    (model_info with files_metadata: the listing `hf` itself reads).
    Raises UpstreamUnavailable when the Hub cannot answer."""
    from .codec.identity import is_commit_sha

    if not is_commit_sha(commit):
        raise UpstreamUnavailable(f"{repo}: the pack records no source commit to check against")
    try:
        from huggingface_hub import HfApi

        info = HfApi().model_info(repo, revision=commit, files_metadata=True, timeout=timeout)
    except Exception as e:  # noqa: BLE001 — offline, gated, gone: all the same verdict here
        raise UpstreamUnavailable(_why(e)) from None
    out = {}
    for s in info.siblings or ():
        lfs = getattr(s, "lfs", None)
        sha = lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)
        out[s.rfilename] = {"size": s.size, "sha256": sha, "blobId": s.blob_id}
    return out


def _why(e: Exception) -> str:
    """One short clause for a failed Hub request: the error's class and the
    start of its first line, never the whole URL-laden message."""
    if type(e).__name__ == "OfflineModeIsEnabled" or os.environ.get("HF_HUB_OFFLINE", "") not in ("", "0"):
        return "offline: HF_HUB_OFFLINE is set"
    first = str(e).splitlines()[0] if str(e) else ""
    return f"{type(e).__name__}: {first[:100]}" if first else type(e).__name__


def git_blob_id(path: str) -> str:
    """The git blob id of a file: sha1 over `blob <size>\\0` and its bytes,
    the id the Hub lists for a file stored in git rather than LFS."""
    h = hashlib.sha1(f"blob {os.path.getsize(path)}\0".encode())
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


# ------------------------------------------------------- the rebuild --

def _layout(path: str) -> tuple[int, dict]:
    """(data section offset, header dict) of one safetensors file."""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return 8 + n, json.loads(f.read(n))


@dataclass
class _Part:
    tensor: str
    dtype: str
    shape: list
    nbytes: int
    source: tuple  # ("trunk" | "mtp", pack name) or ("embedded", remainder name)
    note: str | None = None


class _Pack:
    """What one pack can put into a rebuilt shard: its trunk and `mtp/`
    tensors by pack name, and the remainder's tensors by checkpoint name."""

    def __init__(self, pack_dir: str, meta: dict):
        from .codec.pack import (EMBED_SUBDIR, REMAINDER_FILE, embedded_dir, mtp_pack_dir,
                                 read_mtp_meta)

        self.dir, self.meta = pack_dir, meta
        self.trunk = set(meta.get("tensors") or ())
        self.mtp_meta = read_mtp_meta(pack_dir)
        self.mtp_dir = mtp_pack_dir(pack_dir)
        self.mtp = set((self.mtp_meta or {}).get("tensors") or ())
        self.remainder_path = os.path.join(embedded_dir(pack_dir), REMAINDER_FILE)
        self.remainder_base, hdr = (_layout(self.remainder_path)
                                    if os.path.isfile(self.remainder_path) else (0, {}))
        self.remainder = {k: v for k, v in hdr.items() if k != "__metadata__"}
        self.embed_subdir = EMBED_SUBDIR
        with open(os.path.join(embedded_dir(pack_dir), "config.json")) as f:
            cfg = json.load(f)
        self.model_type = cfg.get("model_type")
        text = cfg.get("text_config") or cfg
        self.tied = bool(text.get("tie_word_embeddings", cfg.get("tie_word_embeddings", False)))

    def _pack_name(self, tname: str) -> tuple | None:
        """The pack tensor a checkpoint tensor was packed as, if any: the
        `mtp/` sub-pack's for an mtp.* weight, else the trunk's, under the
        checkpoint name or the text wrap's skeleton name (arms._TEXT_WRAP)."""
        if not tname.endswith(".weight"):
            return None
        stem = tname[: -len(".weight")]
        if stem.startswith("mtp.") and stem[len("mtp."):] in self.mtp:
            return ("mtp", stem[len("mtp."):])
        wrap = _TEXT_WRAP.get(self.model_type)
        if wrap and stem.startswith(wrap[0]):
            stem = wrap[1] + stem[len(wrap[0]):]
        return ("trunk", stem) if stem in self.trunk else None

    def _tie_source(self, tname: str, info: dict) -> str | None:
        """For a tied output embedding the pack leaves out (the loaders re-tie
        it: codec/pack.tied_output_weight), the remainder tensor it is tied
        to: the input embedding of the same dtype and shape. The rebuilt
        shard's hash is what then shows the checkpoint's copy is
        byte-identical to it."""
        if not self.tied or not tname.endswith("lm_head.weight"):
            return None
        for name, r in self.remainder.items():
            if (name.endswith("embed_tokens.weight") and r["dtype"] == info["dtype"]
                    and list(r["shape"]) == list(info["shape"])):
                return name
        return None

    def plan(self, header: dict) -> tuple[list[_Part], list[str]]:
        """The header's tensors in byte order, each with where its bytes
        come from, and the tensors the pack cannot supply."""
        parts, missing = [], []
        entries = sorted(((k, v) for k, v in header.items() if k != "__metadata__"),
                         key=lambda kv: kv[1]["data_offsets"][0])
        for tname, info in entries:
            start, end = (int(x) for x in info["data_offsets"])
            nbytes = end - start
            if tname in self.remainder:
                parts.append(_Part(tname, info["dtype"], info["shape"], nbytes, ("embedded", tname)))
                continue
            packed = self._pack_name(tname)
            if packed is not None:
                parts.append(_Part(tname, info["dtype"], info["shape"], nbytes, packed))
                continue
            tie = self._tie_source(tname, info)
            if tie is not None:
                parts.append(_Part(tname, info["dtype"], info["shape"], nbytes, ("embedded", tie),
                                   note=f"{tname} rebuilt from {tie}, the tied input embedding"))
                continue
            missing.append(tname)
        return parts, missing

    def tensor_bytes(self, part: _Part, workers: int):
        """The part's bytes as the checkpoint stores them: a buffer (a
        decoded pack tensor, little-endian BF16, row-major) or a list of
        chunks read off the remainder."""
        kind, name = part.source
        if kind == "embedded":
            r = self.remainder[name]
            start, end = (int(x) for x in r["data_offsets"])
            return ("range", self.remainder_base + start, end - start)
        from .codec.pack import read_pack_tensor

        where, meta = (self.dir, self.meta) if kind == "trunk" else (self.mtp_dir, self.mtp_meta)
        p = read_pack_tensor(where, meta, name)
        if part.dtype != "BF16" or [int(p["R"]), int(p["C"])] != [int(x) for x in part.shape]:
            raise _Differs(f"{part.tensor}: the pack holds {name} as BF16 "
                           f"[{int(p['R'])}, {int(p['C'])}], the header says {part.dtype} "
                           f"{list(part.shape)}")
        bits = _decode(p, workers)
        return ("buffer", np.ascontiguousarray(bits.astype("<u2", copy=False)).reshape(-1).view(np.uint8))


class _Differs(ValueError):
    """The pack's tensor cannot be the header's (dtype, shape): a MISMATCH
    with a reason rather than a hash."""


def _decode(p: dict, workers: int) -> np.ndarray:
    from .codec import radix_native
    from .codec.radix_pack import _decoder, _palette_u8, decode_back, is_radix_pack

    if is_radix_pack(p) and _decoder() == "native":
        return radix_native.decode(_palette_u8(p), np.ascontiguousarray(p["rx_offsets"], dtype=np.uint32),
                                   np.ascontiguousarray(p["rx_data"], dtype=np.uint32),
                                   int(p["R"]), int(p["C"]), tuple(int(w) for w in p["widths"]),
                                   workers=workers)
    return decode_back(p)


def _workers() -> int:
    return max(1, min(8, os.cpu_count() or 1))


def _rebuild_sha256(pack: _Pack, header_bytes: bytes, parts: list[_Part]) -> str:
    """sha256 of the shard these parts rebuild: the 8-byte header length,
    the header bytes, then every tensor's bytes in header order. The next
    pack tensor decodes on a second thread while this one hashes (both
    release the GIL)."""
    h = hashlib.sha256()
    h.update(struct.pack("<Q", len(header_bytes)))
    h.update(header_bytes)
    workers = _workers()
    with ThreadPoolExecutor(max_workers=1) as ex, open(pack.remainder_path, "rb") as rem:
        pending = ex.submit(pack.tensor_bytes, parts[0], workers) if parts else None
        for i, part in enumerate(parts):
            got = pending.result()
            pending = ex.submit(pack.tensor_bytes, parts[i + 1], workers) if i + 1 < len(parts) else None
            if got[0] == "buffer":
                buf = got[1]
                if buf.nbytes != part.nbytes:
                    raise _Differs(f"{part.tensor}: the pack's tensor is {buf.nbytes} bytes, "
                                   f"the header's range {part.nbytes}")
                h.update(buf)
                del buf, got
                continue
            _, start, nbytes = got
            if nbytes != part.nbytes:
                raise _Differs(f"{part.tensor}: the embedded tensor is {nbytes} bytes, the "
                               f"header's range {part.nbytes}")
            rem.seek(start)
            left = nbytes
            while left:
                chunk = rem.read(min(left, _CHUNK))
                if not chunk:
                    raise _Differs(f"{pack.remainder_path}: truncated inside {part.tensor}")
                h.update(chunk)
                left -= len(chunk)
    return h.hexdigest()


def _name_tensors(names: list[str], limit: int = 6) -> str:
    shown = ", ".join(names[:limit])
    return shown + (f" and {len(names) - limit} more" if len(names) > limit else "")


def _verify_shard(pack: _Pack, shard: dict, up: dict | None) -> FileVerdict:
    name, size = shard["name"], int(shard["size"])
    header_bytes = shard["header"].encode("utf-8")
    if up is None:
        return FileVerdict(name, NOT_COVERED, "weights", size, None, None,
                           reason="the pack records this shard, but the source commit's file "
                                  "listing does not have it")
    want = up.get("sha256")
    if not want:
        return FileVerdict(name, NOT_COVERED, "weights", size, None, None,
                           reason="the Hub lists no LFS sha256 for it")
    if up.get("size") is not None and int(up["size"]) != size:
        return FileVerdict(name, MISMATCH, "weights", size, want, None,
                           reason=f"the pack records it as {size} bytes, the Hub {up['size']}")
    header = json.loads(header_bytes)
    parts, missing = pack.plan(header)
    if missing:
        return FileVerdict(name, NOT_COVERED, "weights", size, want, None,
                           reason=f"{len(missing)} tensor(s) the pack does not carry: "
                                  f"{_name_tensors(missing)}", tensors=missing)
    # the data section must be exactly the tensors, end to end: a byte
    # between two ranges is in no tensor, so in nothing a pack holds
    at = 0
    for kv in sorted((v["data_offsets"] for k, v in header.items() if k != "__metadata__"),
                     key=lambda o: o[0]):
        if int(kv[0]) != at:
            return FileVerdict(name, NOT_COVERED, "weights", size, want, None,
                               reason=f"its header leaves bytes {at}..{int(kv[0])} of the data "
                                      "section outside every tensor")
        at = int(kv[1])
    if 8 + len(header_bytes) + at != size:
        return FileVerdict(name, NOT_COVERED, "weights", size, want, None,
                           reason=f"{size - 8 - len(header_bytes) - at} bytes after the last "
                                  "tensor are in no tensor")
    try:
        got = _rebuild_sha256(pack, header_bytes, parts)
    except _Differs as e:
        return FileVerdict(name, MISMATCH, "weights", size, want, None, reason=str(e))
    notes = [p.note for p in parts if p.note]
    if got != want:
        return FileVerdict(name, MISMATCH, "weights", size, want, got,
                           reason=f"rebuilt sha256 {got[:16]}…, the publisher's {want[:16]}…"
                                  + "".join(f"; {n}" for n in notes))
    return FileVerdict(name, MATCH, "weights", size, want, got, note="; ".join(notes) or None)


def _verify_small_file(path: str, name: str, up: dict | None) -> FileVerdict:
    size = os.path.getsize(path)
    if up is None:
        return FileVerdict(name, MISMATCH, "file", size, None, None,
                           reason="the pack embeds it as the checkpoint's, but the source "
                                  "commit has no such file")
    if up.get("sha256"):
        got = _sha256_file(path)
        want = up["sha256"]
    else:
        got, want = "git:" + git_blob_id(path), "git:" + str(up.get("blobId"))
    if got != want:
        return FileVerdict(name, MISMATCH, "file", size, want, got,
                           reason=f"the embedded copy hashes {got[:20]}…, the publisher's "
                                  f"{want[:20]}…")
    return FileVerdict(name, MATCH, "file", size, want, got)


def verify_upstream(pack_dir: str, listing: dict | None = None,
                    write_receipt: bool = True) -> Verdict:
    """Rebuild every source shard the pack records from the pack, hash it,
    and compare with the Hub's published sha256 for that file at the pack's
    bound commit; compare each embedded small file the same way. `listing`
    is hub_listing's result when the caller already has it; otherwise it is
    fetched, and UpstreamUnavailable propagates. Writes RECEIPT_FILE unless
    `write_receipt` is False. The pack's own file hashes are the caller's
    gate (codec/pack.verify_pack); this reads the files as they are."""
    from . import __version__
    from .codec import identity
    from .codec.pack import _check_format_version, embedded_dir, is_self_contained

    with open(os.path.join(pack_dir, "meta.json")) as f:
        meta = json.load(f)
    _check_format_version(meta, pack_dir)
    src = meta.get("source") or {}
    if src.get("kind") != "hub" or not identity.is_commit_sha(src.get("revision")):
        raise ValueError(f"{pack_dir}: bound to {identity.describe(src)}, not a Hub commit — "
                         "there is no published hash to compare with")
    if not is_self_contained(meta):
        raise ValueError(f"{pack_dir}: carries no embedded checkpoint, so no source shard can "
                         "be rebuilt from it")
    repo, commit = src["repo"], src["revision"]
    t0 = time.perf_counter()
    if listing is None:
        listing = hub_listing(repo, commit)
    edir = embedded_dir(pack_dir)
    with open(os.path.join(edir, identity.EMBEDDED_IDENTITY_FILE)) as f:
        shards = json.load(f)["shards"]
    pack = _Pack(pack_dir, meta)
    files = []
    recorded = {s["name"] for s in shards}
    for shard in shards:
        files.append(_verify_shard(pack, shard, listing.get(shard["name"])))
    for name in sorted(listing):
        if name.endswith(".safetensors") and "/" not in name and name not in recorded:
            files.append(FileVerdict(name, NOT_COVERED, "weights", listing[name].get("size"),
                                     listing[name].get("sha256"), None,
                                     reason="a weight file of the source commit the pack "
                                            "records no header for"))
    own = {identity.EMBEDDED_IDENTITY_FILE, "remainder.safetensors"}
    for name in sorted((meta.get("embedded") or {}).get("files") or {}):
        if name not in own:
            files.append(_verify_small_file(os.path.join(edir, name), name, listing.get(name)))
    from .codec.radix_pack import _decoder

    v = Verdict(repo, commit, files, time.perf_counter() - t0, _decoder())
    if write_receipt:
        write_receipt_file(pack_dir, meta, v, __version__)
    return v


# ------------------------------------------------------- the receipt --

def _pack_binding(pack_dir: str, meta: dict) -> dict:
    """What a receipt is bound to: the manifest digests, recomputed from the
    live meta.json maps (trunk and `mtp/`), which cover every file hash, the
    embedded checkpoint and the source identity."""
    from .codec.identity import manifest_digest
    from .codec.pack import read_mtp_meta

    mtp = read_mtp_meta(pack_dir)
    return {"manifestSha256": manifest_digest(meta),
            "mtpManifestSha256": manifest_digest(mtp) if mtp is not None else None,
            "formatVersion": meta.get("formatVersion"), "profile": meta.get("profile")}


def write_receipt_file(pack_dir: str, meta: dict, v: Verdict, version: str) -> str:
    rec = {
        "receiptVersion": RECEIPT_VERSION,
        "repo": v.repo,
        "commit": v.commit,
        "verdict": v.verdict,
        "drinkmeVersion": version,
        "verifiedAt": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "seconds": round(v.seconds, 2),
        "decoder": v.decoder,
        "pack": _pack_binding(pack_dir, meta),
        "files": [{k: val for k, val in asdict(f).items() if val not in (None, [])}
                  for f in v.files],
    }
    path = os.path.join(pack_dir, RECEIPT_FILE)
    tmp = f"{path}.tmp-{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(rec, f, indent=1)
        f.write("\n")
    os.replace(tmp, path)
    return path


def read_receipt(pack_dir: str) -> dict | None:
    try:
        with open(os.path.join(pack_dir, RECEIPT_FILE)) as f:
            rec = json.load(f)
    except (OSError, ValueError):
        return None
    return rec if isinstance(rec, dict) else None


def receipt_current(pack_dir: str, meta: dict | None = None) -> dict | None:
    """The pack's receipt when it records a MATCH for the pack as it is now
    (the same manifest digests, so the same file hashes), else None. The
    files themselves are the load-time hash check's (verify_hashes)."""
    rec = read_receipt(pack_dir)
    if rec is None or rec.get("receiptVersion") != RECEIPT_VERSION or rec.get("verdict") != MATCH:
        return None
    if meta is None:
        with open(os.path.join(pack_dir, "meta.json")) as f:
            meta = json.load(f)
    try:
        if rec.get("pack") != _pack_binding(pack_dir, meta):
            return None
    except (OSError, ValueError):
        return None
    src = meta.get("source") or {}
    if (rec.get("repo"), rec.get("commit")) != (src.get("repo"), src.get("revision")):
        return None
    return rec
