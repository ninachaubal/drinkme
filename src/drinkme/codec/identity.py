"""Pack identity, torch-free: which checkpoint a pack was cut from.

The codec packs a model's eligible Linears; embeddings, norms, biases, the
small projections and the MTP head's norms are the raw half, copied into
every pack's `checkpoint/` (codec/pack.write_embedded) and read back from
there at load. The file hashes prove the packed half is the exact bytes
that passed verification; the `source` block below proves the raw half
comes from the same checkpoint — recomputed from the embedded files
(embedded_digest), or, at pack time and for `drinkme bench`'s stock and
twin arms (which always read the real checkpoint), from the snapshot
(content_digest). Without it the loader would
read the raw tensors from wherever the caller's coordinate (None, a branch,
whatever the CLI was given) resolved on the serving machine — a different
snapshot the day the branch moves and a different model the day someone
passes the wrong --model with --pack-dir — and compatible shapes would let
the mixed engine load and serve with a 200 status.

This module is the identity the pack binds to and the loader checks:

  checkpoint_identity(snap) -> {"kind", "revision", "digest", ...}

  kind "hub":   the snapshot is `<hf cache>/models--*/snapshots/<sha>/`,
                and `revision` is that commit sha — immutable by the hub's
                construction, and the cache's own name for what it holds.
  kind "local": a plain directory; `revision` is None.
  digest:       BOTH kinds carry a content digest, sha256 over the
                safetensors HEADERS (every tensor's name, dtype, shape and
                byte range, plus the file's size), config.json, the
                tokenizer files and the image processor configs — the
                serving-critical kilobytes, never the gigabytes, so it costs
                milliseconds at every load and still pins the model's whole
                structure, vocabulary and image input. Modifying a local
                checkpoint's values in place with identical shapes is the one
                edit it cannot see; a different repo, revision, shard layout,
                config, tokenizer or image preprocessing it does.

The loader-side comparison (serving/checkpoint.resolve_pack_source) lives
next to snapshot_dir; this file is pure so the mlx runtime (no torch —
docs/metal.md) and the CLI's pre-torch front door can both use it.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import struct

IDENTITY_VERSION = 1

# The serving-critical small files: what the skeleton, the sampler defaults,
# the tokenizer and the image preprocessing are built from. Whichever of
# these exist are digested, by name, so a missing one changes the identity
# too. The two processor configs are the image input's semantics (patch and
# merge size, pixel bounds, mean and std: serving/vision.read_processor_config
# reads them) as much as config.json is the skeleton's. They joined at
# version 1, before any pack had shipped.
# generation_config.json is deliberately NOT here: it is sampling defaults
# and the EOS set, read from the same snapshot at load (gen_config.load) but
# not part of what makes two checkpoints the same model — an operator tuning
# it under a local checkpoint has not changed the weights the pack binds.
IDENTITY_FILES = (
    "config.json", "model.safetensors.index.json",
    "tokenizer_config.json", "tokenizer.json", "tokenizer.model", "vocab.json",
    "merges.txt", "special_tokens_map.json", "added_tokens.json",
    "chat_template.jinja", "chat_template.json", "spiece.model", "vocab.txt",
    "preprocessor_config.json", "processor_config.json",
)

_SHA_RE = re.compile(r"^[0-9a-f]{40}$")


def is_commit_sha(rev) -> bool:
    return isinstance(rev, str) and bool(_SHA_RE.match(rev))


def hub_revision_of(snap: str) -> str | None:
    """The commit sha a hub-cache snapshot dir is named by, or None for a
    directory that is not one (`.../snapshots/<40 hex>` is huggingface_hub's
    layout; nothing else names a dir that way)."""
    path = os.path.normpath(snap)
    leaf, parent = os.path.basename(path), os.path.basename(os.path.dirname(path))
    if parent == "snapshots" and is_commit_sha(leaf):
        return leaf
    return None


def _safetensors_header(path: str) -> bytes:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return f.read(n)


def source_shards(snap: str) -> list[tuple[str, int, bytes]]:
    """(name, file size, header bytes) for every *.safetensors in `snap`,
    sorted by name: the shard half of what content_digest hashes. A
    self-contained pack keeps exactly this (embedded_identity_record) so
    the digest can be recomputed without the shards."""
    out = []
    for name in sorted(n for n in os.listdir(snap) if n.endswith(".safetensors")
                       and os.path.isfile(os.path.join(snap, n))):
        p = os.path.join(snap, name)
        out.append((name, os.path.getsize(p), _safetensors_header(p)))
    return out


def digest_of(shards, files: dict[str, bytes]) -> str:
    """content_digest's arithmetic over parts already in hand: `shards` as
    source_shards returns them, `files` the IDENTITY_FILES members that
    exist, by name. content_digest reads both off a checkpoint directory;
    embedded_digest reads them off a self-contained pack."""
    h = hashlib.sha256()
    h.update(f"drinkme-checkpoint-identity-v{IDENTITY_VERSION}\0".encode())
    for name, size, header in sorted(shards):
        h.update(f"safetensors\0{name}\0{size}\0{len(header)}\0".encode())
        h.update(header)
    for name in IDENTITY_FILES:
        blob = files.get(name)
        if blob is None:
            continue
        h.update(f"file\0{name}\0{len(blob)}\0".encode())
        h.update(blob)
    return h.hexdigest()


def _identity_files(snap: str) -> dict[str, bytes]:
    files = {}
    for name in IDENTITY_FILES:
        p = os.path.join(snap, name)
        if os.path.isfile(p):
            with open(p, "rb") as f:
                files[name] = f.read()
    return files


def content_digest(snap: str) -> str:
    """sha256 over the checkpoint's serving-critical bytes: every
    *.safetensors header (+ the file's size), and each IDENTITY_FILES
    member that exists, whole. File names are folded in, so the set of
    files is part of the identity, not only their contents."""
    return digest_of(source_shards(snap), _identity_files(snap))


# ------------------------------------------- the embedded checkpoint --
# A self-contained pack (docs/pack-format.md#the-embedded-checkpoint)
# carries the checkpoint's small files and its unpacked tensors in
# `<pack>/checkpoint/`, but not the original shards, and not the shard
# index (it names shards the directory does not hold, and
# missing_from_snapshot would demand them). What content_digest read from
# those two is kept in `checkpoint/source-identity.json`, so the digest
# the pack recorded as `source` recomputes from the pack alone.

EMBEDDED_IDENTITY_FILE = "source-identity.json"
INDEX_FILE = "model.safetensors.index.json"


def embedded_identity_record(snap: str) -> dict:
    """What source-identity.json holds for the checkpoint at `snap`: each
    shard's name, size and header (safetensors headers are UTF-8 JSON by
    the format's definition), and the shard index, if there is one."""
    index = os.path.join(snap, INDEX_FILE)
    rec = {
        "identityVersion": IDENTITY_VERSION,
        "shards": [{"name": n, "size": size, "header": header.decode("utf-8")}
                   for n, size, header in source_shards(snap)],
        "index": None,
    }
    if os.path.isfile(index):
        with open(index, "rb") as f:
            rec["index"] = f.read().decode("utf-8")
    return rec


def embedded_digest(embed_dir: str) -> str:
    """The source digest recomputed from a pack's embedded checkpoint: the
    shard headers and index from source-identity.json, every other
    IDENTITY_FILES member from the directory itself. Equal to the
    `source.digest` a self-contained pack records, or the embedded files
    are not the ones the pack was cut from."""
    with open(os.path.join(embed_dir, EMBEDDED_IDENTITY_FILE)) as f:
        rec = json.load(f)
    if rec.get("identityVersion") != IDENTITY_VERSION:
        raise ValueError(f"{embed_dir}/{EMBEDDED_IDENTITY_FILE}: identity version "
                         f"{rec.get('identityVersion')!r}; this build computes {IDENTITY_VERSION}")
    shards = [(s["name"], int(s["size"]), s["header"].encode("utf-8")) for s in rec["shards"]]
    files = _identity_files(embed_dir)
    files.pop(INDEX_FILE, None)
    if rec.get("index") is not None:
        files[INDEX_FILE] = rec["index"].encode("utf-8")
    return digest_of(shards, files)


def checkpoint_identity(snap: str, repo: str | None = None) -> dict:
    """The identity block a pack records as meta.json's `source` and the
    loader recomputes from the snapshot it is about to stream from.

    `repo` is the coordinate the caller named (an HF repo id, or the local
    path) — carried for the refusal message and for the loader's repo
    check, never part of the digest."""
    rev = hub_revision_of(snap)
    return {
        "identityVersion": IDENTITY_VERSION,
        "kind": "hub" if rev else "local",
        "repo": repo,
        "revision": rev,
        "digest": content_digest(snap),
    }


def describe(ident: dict | None) -> str:
    """One line naming an identity, for refusals: repo@revision plus the
    digest's head."""
    if not ident:
        return "<no source recorded>"
    rev = ident.get("revision") or "local"
    digest = str(ident.get("digest") or "?")[:16]
    return f"{ident.get('repo')}@{rev} (digest {digest}…)"


# ------------------------------------------------------------ the manifest --
# meta.json hashes each .npz BY FILENAME; the map from tensor NAME to
# filename is what the manifest binds. Were that map unhashed, swapping two
# same-shaped entries of meta['tensors'] would change what the loader
# decodes as `a` and `b` while every hashed file stayed identical — the
# file hashes would pass, and a cache id folded from the file->hash map
# alone would keep KV computed against the old assignment eligible. The
# manifest is the canonical statement of what a pack IS: name -> file -> content hash, each
# tensor's shape/dtype/layout, the format version, and the checkpoint
# identity above. Its digest is recorded at pack time; verify_hashes recomputes it from the
# live maps, and the cache id derives from it.

MANIFEST_VERSION = 1


def tensor_info(pack: dict, name: str | None = None) -> dict:
    """The per-tensor descriptor the manifest carries, from a pack dict
    (pack_weight_radix's output, or an npz's scalars blob): shape, dtype
    and the codec's own descriptors (radix: profile, widths, block size).

    A `dtype` scalar (the bf16 codec's own tensors carry none; an FP8
    pack's carry "fp8_e4m3") is refused BY NAME rather than defaulted
    into the `"dtype": "bf16"` this dict starts from: a manifest that
    mislabels a plane bf16 would record a lie (codec/pack.py's `_arrays_for`
    is the load-time twin of this refusal)."""
    dtype = pack.get("dtype")
    if dtype is not None:
        from .pack import refusal_for_dtype

        raise ValueError(
            f"pack tensor {name or '<unnamed>'} declares dtype {dtype!r} — refusing "
            f"to describe it in the manifest rather than label it bf16. {refusal_for_dtype(dtype)}")
    info = {"shape": [int(pack["R"]), int(pack["C"])], "dtype": "bf16"}
    if pack.get("codec") == "radix":
        # the radix codec (codec/radix_pack.py): the manifest binds the tier
        # widths and block size — the two descriptors that decide how the
        # streams decode — beside the shape, so a same-shaped tensor
        # re-encoded at another profile fails the manifest, not just the file hashes
        info["codec"] = "radix"
        info["profile"] = str(pack["profile"])
        info["widths"] = [int(w) for w in pack["widths"]]
        info["blockSize"] = int(pack["block_size"])
        return info
    if pack.get("codec") == "raw":
        # the raw fallback: the bf16 bits verbatim, named as such so a
        # manifest cannot pass off a raw tensor as a coded one or vice versa
        info["codec"] = "raw"
        return info
    if "layout" in pack:
        info["layout"] = int(pack["layout"])
    return info


def pack_manifest(meta: dict) -> dict:
    """The canonical manifest, built from the LIVE meta.json maps the loader
    reads (`tensors`, `sha256`, `tensorInfo`, `source`, and `embedded` when
    the pack carries its checkpoint) — so any edit to any of them changes
    the digest, which is the whole point."""
    hashes = meta.get("sha256") or {}
    info = meta.get("tensorInfo") or {}
    tensors = {}
    for name in sorted(meta.get("tensors") or {}):
        fn = meta["tensors"][name]
        entry = {"file": fn, "sha256": hashes.get(fn)}
        entry.update(info.get(name) or {})
        tensors[name] = entry
    manifest = {
        "manifestVersion": MANIFEST_VERSION,
        "formatVersion": meta.get("formatVersion"),
        "source": meta.get("source"),
        "tensors": tensors,
    }
    # a self-contained pack's embedded checkpoint (file -> sha256): bound
    # like the tensors, and absent from the manifest of a pack without one
    # (a tool-written pack, save_pack_dir)
    if meta.get("embedded") is not None:
        manifest["embedded"] = meta["embedded"]
    return manifest


def manifest_digest(meta: dict) -> str:
    """sha256 of the manifest's canonical JSON (sorted keys, no whitespace,
    ASCII) — what PackWriter.finish records as meta.json's manifestSha256
    and verify_hashes recomputes."""
    blob = json.dumps(pack_manifest(meta), sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()



def pack_id_from_meta(pack_meta: dict) -> str:
    """A stable id for the exact bytes — and the exact tensor assignment —
    a pack serves: "manifest" + the recorded manifest digest. Keys the on-disk
    prefix-slot store's directory (serving/slotstore.py imports it from
    here: this module has no torch, which is what lets the mlx runtime's
    compressed loader reach it).

    The digest is RECOMPUTED from the live maps, never read
    back from the recorded field, so a meta.json whose tensor map was edited
    without re-hashing lands in a different directory even before
    verify_hashes refuses it. Every pack records its manifest digest at pack time; a
    meta.json without one is not a pack this drinkme wrote and has no id."""
    if not pack_meta.get("manifestSha256"):
        raise ValueError("pack_id_from_meta: meta.json records no manifestSha256 — "
                         "not a verifiable pack (every pack records its manifest digest at pack time)")
    return "manifest" + manifest_digest(pack_meta)[:16]
