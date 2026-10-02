"""The checkpoint-side helpers every engine needs and none of which need torch.

Three things every engine needs — where the checkpoint snapshot is, how its
tokenizer loads, how the context window is clamped — live here rather than in
arms.py or engines.py: both import torch at the top, and the mlx runtime's
code never imports torch (pyproject's `metal` group carries it only to pack, and a
metal environment made before that has none; docs/metal.md).
serving/engine_mlx.py shares them byte for byte instead of
carrying a second opinion; arms.py and engines.py import them under their
own names.
"""

from __future__ import annotations

import functools
import json
import os
import sys

# StaticCache preallocates its full length, so an unbounded ctx is an
# unbounded KV allocation on a unified-memory machine (Qwen3-8B at its native 40k
# would preallocate ~6GB). Cap by default; DRINKME_CTX raises it deliberately.
# The MLX engine's KVCache grows instead of preallocating, but the SAME cap
# applies: it is the window the server advertises and checks requests against.
CTX_CAP = 8192


# What a local snapshot must hold before it counts as a hit. The weights
# (every shard the index names, or a bare model.safetensors), the config the
# skeleton is built from, and the tokenizer files the loader reads —
# tokenizer_config.json is the exact file tokenizer() below gates on, plus
# one vocabulary payload in whichever spelling the family ships.
_VOCAB_FILES = ("tokenizer.json", "tokenizer.model", "vocab.json", "spiece.model",
                "vocab.txt")


def missing_from_snapshot(local: str) -> list[str]:
    """The files a checkpoint snapshot is missing, by name; [] means the
    snapshot is complete enough to pack or serve from.

    A partial cache is the normal shape of an interrupted download: the HF
    cache commits each file as it lands, so a two-shard checkpoint whose pull
    died after shard one has an index, config.json and exactly one
    *.safetensors — every earlier "is there a weight file?" test called that
    a hit, and the packer then refused on the missing
    shard while every retry took the same local-only path and never
    repaired it. When an index exists, every shard it references must
    exist; without one, some *.safetensors must."""
    import glob

    missing: list[str] = []
    if not os.path.isfile(os.path.join(local, "config.json")):
        missing.append("config.json")
    index = os.path.join(local, "model.safetensors.index.json")
    if os.path.isfile(index):
        try:
            with open(index) as f:
                shards = sorted(set(json.load(f).get("weight_map", {}).values()))
        except (OSError, ValueError):
            shards = []
            missing.append("model.safetensors.index.json (unreadable)")
        missing.extend(s for s in shards if not os.path.isfile(os.path.join(local, s)))
    elif not glob.glob(os.path.join(local, "*.safetensors")):
        missing.append("*.safetensors")
    if not os.path.isfile(os.path.join(local, "tokenizer_config.json")):
        missing.append("tokenizer_config.json")
    if not any(os.path.isfile(os.path.join(local, v)) for v in _VOCAB_FILES):
        missing.append(" or ".join(_VOCAB_FILES))
    return missing


# What snapshot_dir fetches: the weights and the small serving-critical
# files, never the rest of a repo (README, LICENSE, images).
SNAPSHOT_PATTERNS = ("*.safetensors", "*.safetensors.index.json", "*.json",
                     "*.jinja", "*.txt", "tokenizer.model")


@functools.lru_cache(maxsize=None)
def snapshot_dir(repo: str, revision: str | None) -> str:
    """Local checkpoint dir: the repo path itself, or the HF cache snapshot
    (a cache hit whenever the checkpoint was already pulled — COMPLETELY:
    missing_from_snapshot decides, and an incomplete hit falls through to
    the normal download, which is the only path that repairs it).

    lru_cached — the MTP head loader calls this a second time per boot with
    identical arguments, which was a second hub round-trip (the hot-loop
    audit). Local-first for the same reason: with revision=None the hub
    must resolve "main" over the network even when every byte is cached;
    network only on a genuine local miss."""
    if os.path.isdir(repo):
        return repo
    from huggingface_hub import snapshot_download

    # Weights AND the small serving-critical files (tokenizer/config/
    # template — kilobytes beside gigabytes). A weights-only snapshot makes a
    # fresh machine serve a template-less fabricated
    # tokenizer (transformers builds one from NOTHING under
    # local_files_only rather than raising — see tokenizer() below).
    pats = list(SNAPSHOT_PATTERNS)
    local, missing = None, []
    try:
        local = snapshot_download(repo, revision=revision, allow_patterns=pats,
                                  local_files_only=True)
        # local_files_only=True SUCCEEDS on a partial cache — AutoConfig
        # (called before us in pack_model) seeds the snapshot with just
        # config.json, and the "hit" then holds zero weight files.
        # Reproduced on a fresh cache; invisible on any machine whose models
        # are fully cached. A
        # local snapshot only counts as a hit if it holds what the patterns
        # asked for — ALL of it (one shard of two is not a
        # checkpoint either).
        missing = missing_from_snapshot(local)
        if not missing:
            return local
    except Exception:  # LocalEntryNotFoundError et al. — a genuine miss
        pass
    _announce_download(repo, revision, pats, local, missing)
    try:
        return snapshot_download(repo, revision=revision, allow_patterns=pats)
    except Exception as e:
        if local is None:
            raise
        # Offline with a partial cache: name what is missing, so the fix is
        # "get these files" rather than a hub traceback three frames down.
        raise RuntimeError(
            f"checkpoint {repo}@{revision or 'main'} is incomplete in the local "
            f"cache ({local}) — missing {missing} — and fetching it failed "
            f"({type(e).__name__}: {e}). Reconnect and retry, or copy the "
            "missing files into that snapshot directory.") from e


def _announce_download(repo: str, revision: str | None, pats: list[str],
                       local: str | None, missing: list[str]) -> None:
    """One line before snapshot_dir goes to the Hub: which checkpoint, why
    (not cached, or which files a partial cache lacks), and how much it
    is — the files the patterns select, sized from the Hub's own listing
    when it answers. A multi-gigabyte download never happens silently."""
    import fnmatch

    why = (f"missing locally: {', '.join(missing[:4])}{' …' if len(missing) > 4 else ''}"
           if local is not None else "not in the local Hugging Face cache")
    size = ""
    try:
        from huggingface_hub import HfApi

        info = HfApi().model_info(repo, revision=revision, files_metadata=True)
        picked = [f for f in info.siblings or []
                  if any(fnmatch.fnmatch(f.rfilename, p) for p in pats)]
        total = sum(int(f.size or 0) for f in picked)
        size = f": {len(picked)} files, {total / 1e9:.2f} GB (files already cached are kept)"
    except Exception:  # noqa: BLE001 — offline or unknown: the download itself will say
        size = " (size unknown: the Hub did not answer)"
    _warn(f"downloading checkpoint {repo}@{revision or 'main'} from the Hugging Face Hub, "
          f"{why}{size}")


def resolve_commit(repo: str) -> str | None:
    """An off-menu hub repo's default branch -> the commit sha this machine
    would read, or None when it cannot be told (a local directory, offline
    with nothing cached, an unknown repo).

    Local-first, like snapshot_dir with revision=None: a cached snapshot is
    what the loader would stream from, so its sha is the one a pack of it
    binds to (the cache's refs/main, read through the snapshot path it
    names). Only on a miss does the hub answer, which is where snapshot_dir
    would download from. The caller keys the pack directory and reports the
    revision by this sha, so a later push to the branch is a different pack,
    never a silent reuse of the old one."""
    if os.path.isdir(repo):
        return None
    from ..codec.identity import hub_revision_of, is_commit_sha

    try:
        from huggingface_hub import try_to_load_from_cache

        hit = try_to_load_from_cache(repo, "config.json")
        if isinstance(hit, str):
            rev = hub_revision_of(os.path.dirname(hit))
            if rev:
                return rev
    except Exception:  # noqa: BLE001 — a cache we cannot read is a miss
        pass
    try:
        from huggingface_hub import HfApi

        sha = HfApi().model_info(repo).sha
        return sha if is_commit_sha(sha) else None
    except Exception:  # noqa: BLE001 — offline, unknown repo: nothing to pin
        return None


def pack_revision(pack_meta: dict) -> str | None:
    """The revision a served pack reports (/v1/models): the one it was keyed
    by, or, for a pack keyed `@main` (cli.pin_revision could not resolve the
    commit), the commit its source block is bound to — never null for a hub
    pack that knows its commit."""
    return pack_meta.get("revision") or (pack_meta.get("source") or {}).get("revision")


def cached_snapshot_dir(repo: str, revision: str | None) -> str | None:
    """snapshot_dir's local-only half: the complete snapshot if this machine
    already has it, else None — never the network. For the tools that must
    not download (`drinkme bench`'s fit check)."""
    if os.path.isdir(repo):
        return repo
    from huggingface_hub import snapshot_download

    try:
        # the files snapshot_dir fetches, no more: asked for the whole repo,
        # huggingface_hub calls a snapshot missing its README incomplete,
        # and every snapshot snapshot_dir downloaded would read as a miss
        local = snapshot_download(repo, revision=revision, local_files_only=True,
                                  allow_patterns=list(SNAPSHOT_PATTERNS))
    except Exception:  # LocalEntryNotFoundError et al. — a miss
        return None
    return local if not missing_from_snapshot(local) else None


def largest_tensor_bytes(snap: str) -> int | None:
    """The byte size of the LARGEST tensor across a checkpoint snapshot's
    safetensors shards, from the JSON headers alone (dtype x shape; zero
    weight bytes read — codec.pack._shard_headers). None when the directory
    holds no readable shard.

    This is the streaming stock loader's host transient
    (fit.stock_transient_bytes `tensor_bytes`): serving.engines.
    stream_checkpoint holds one tensor beyond the resident model, so the
    fit arithmetic charges resident + this. A caller whose snapshot is not on
    disk yet (bench's fit check before the download) passes None and the
    arithmetic charges fit.STAGING_SHARD_BYTES instead. Torch-free, so
    `bench --dry-run` can read it."""
    import math

    from ..codec.pack import _shard_headers
    from ..check import _dtype_nbytes

    try:
        headers = _shard_headers(snap)
    except (OSError, ValueError):
        return None
    if not headers:
        return None
    return max(_dtype_nbytes(dtype) * math.prod(shape or [0])
               for dtype, shape, _path in headers.values())


def checkpoint_tensor_bytes(snap: str) -> int | None:
    """The bytes every tensor across a snapshot's safetensors shards stores
    (dtype x shape, summed), from the JSON headers alone, like
    largest_tensor_bytes. A BF16 checkpoint's BF16 size: what the menu
    derives a model's sizes from (suggest.checkpoint_bytes). None when the
    directory holds no readable shard."""
    import math

    from ..codec.pack import _shard_headers
    from ..check import _dtype_nbytes

    try:
        headers = _shard_headers(snap)
    except (OSError, ValueError):
        return None
    if not headers:
        return None
    return sum(_dtype_nbytes(dtype) * math.prod(shape or [0])
               for dtype, shape, _path in headers.values())


def resolved_revision(snap: str) -> str:
    """The immutable identity of the weights in a resolved checkpoint
    directory — what a persisted KV cache is keyed on (`--revision main`
    today and `main` next month are different weights
    under one name, and the store must not serve one's keys to the other).

    A hub snapshot names it already: the HF cache lays a checkpoint out as
    `models--Org--Name/snapshots/<commit sha>/`, and the sha IS the
    revision. A local directory has no such name, so it gets a digest of
    what is there: config.json whole, and for every weight shard (and the
    shard index) its name, size, mtime and the first MiB of bytes — the
    safetensors JSON header (every tensor's name, dtype, shape, offset)
    plus the head of the first tensor's data. Not a full-content hash (a
    16 GiB checkpoint is seconds of hashing at every boot for a derived
    artifact that costs a prefill to rebuild) but not a name either: a
    rewrite of the weights in place moves it. Prefixed `local-` so a header
    says which kind it holds."""
    import hashlib
    import re

    m = re.search(r"/snapshots/([0-9a-f]{40})(?:/|$)", snap)
    if m:
        return m.group(1)
    h = hashlib.sha256()
    try:
        with open(os.path.join(snap, "config.json"), "rb") as f:
            h.update(f.read())
    except OSError:
        h.update(b"<no config.json>")
    names = sorted(n for n in os.listdir(snap)
                   if n.endswith(".safetensors") or n.endswith(".safetensors.index.json"))
    for name in names:
        full = os.path.join(snap, name)
        st = os.stat(full)
        h.update(f"{name}\0{st.st_size}\0{st.st_mtime_ns}\0".encode("utf-8"))
        with open(full, "rb") as f:
            h.update(f.read(1 << 20))
    return "local-" + h.hexdigest()[:16]


def tokenizer(repo: str, revision: str | None):
    """The checkpoint's tokenizer, local-first.

    Everything is on disk after the first pull, and the hub HEAD-revalidates
    ~6 files per boot otherwise (~1s good network, unbounded on a bad one).
    Network only on a miss. GATED on the cache actually
    holding tokenizer_config.json: under local_files_only,
    transformers FABRICATES a tokenizer from a partial cache — no tokenizer
    files at all — instead of raising; the degenerate object has no
    chat_template and serves 500s on a fresh machine. A snapshot ref is
    not a tokenizer."""
    from transformers import AutoTokenizer

    kw = {"revision": revision} if revision else {}
    if os.path.isdir(repo):  # local checkpoint dir — no hub cache involved
        return AutoTokenizer.from_pretrained(repo, **kw)
    from huggingface_hub import try_to_load_from_cache

    cached = try_to_load_from_cache(
        repo, "tokenizer_config.json", **({"revision": revision} if revision else {})
    )
    if isinstance(cached, str):
        try:
            return AutoTokenizer.from_pretrained(repo, local_files_only=True, **kw)
        except (OSError, ValueError):
            pass
    return AutoTokenizer.from_pretrained(repo, **kw)


def resolve_ctx(native: int, ctx: int | None) -> int:
    """Resolve the allocated context window from the model's native one.
    An EXPLICIT ask (--ctx or DRINKME_CTX) is honored but clamped to the
    native window: past max_position_embeddings the positions are outside
    the trained RoPE range, and StaticCache would happily preallocate the
    whole ask — measured as a single 81 GiB allocation attempt
    from an over-window ctx. The clamp says so on stderr rather than letting
    the OOM be the error message."""
    native = int(native or 0)
    asked = ctx or int(os.environ.get("DRINKME_CTX", 0) or 0)
    if asked:
        if native and asked > native:
            print(f"[drinkme] ctx {asked} exceeds the model's native window "
                  f"{native} — clamped to {native}", file=sys.stderr)
            return native
        return asked
    return min(native or CTX_CAP, CTX_CAP)


# ------------------------------------------------- the pack's ONE snapshot --


class PackSourceMismatch(ValueError):
    """The pack and the checkpoint the caller named are not the same model."""


def _warn(msg: str) -> None:
    print(f"[drinkme] {msg}", flush=True)


def check_pack_repo(repo: str, pack_meta: dict, pack_dir: str | None = None) -> None:
    """The no-I/O half of resolve_pack_source: a pack cut from one hub repo
    served under another hub id is refused on the names alone — every pack
    records hfRepo, so this holds for a pack without a `source` block too.
    Local paths are not
    compared as strings (a symlink or a relative spelling is the same dir);
    their identity is the content digest."""
    packed_repo = pack_meta.get("hfRepo")
    if (not os.path.isdir(repo) and isinstance(packed_repo, str)
            and not os.path.isdir(packed_repo) and packed_repo != repo):
        raise PackSourceMismatch(
            f"refusing to load: pack {pack_dir or '<pack>'} was cut from "
            f"{packed_repo}, but the checkpoint named is {repo} — a pack's packed "
            "Linears and the checkpoint's raw tensors must come from one model. "
            "Serve the pack with its own model, or pack this one.")


def _cached_commit(repo: str, revision: str) -> str | None:
    """The commit a branch or tag names in the local HF cache, or None —
    never the network (a self-contained pack's revision check)."""
    from ..codec.identity import hub_revision_of

    try:
        from huggingface_hub import try_to_load_from_cache

        hit = try_to_load_from_cache(repo, "config.json", revision=revision)
    except Exception:  # noqa: BLE001 — an unreadable cache is a miss
        return None
    return hub_revision_of(os.path.dirname(hit)) if isinstance(hit, str) else None


def _resolve_embedded(repo: str, revision: str | None, pack_meta: dict, pack_dir: str) -> str:
    """resolve_pack_source for a self-contained pack: the pack's own
    `checkpoint/` directory, after the same refusals the snapshot path
    makes, none of which needs the checkpoint or the network.

      the name      check_pack_repo (already run by the caller)
      a revision    an explicit commit must be the pack's; a branch or tag
                    is resolved in the local cache only, and refused when
                    it resolves elsewhere or cannot be resolved offline
      a local dir   a --model naming an existing directory must digest to
                    the pack's source, as it would have to if it were read
      the files     the embedded config, tokenizer, image processor configs
                    and shard headers recompute the recorded digest
                    (codec/pack.check_embedded_identity)"""
    from ..codec import identity
    from ..codec.pack import check_embedded_identity

    src = pack_meta.get("source") or {}
    want = src.get("revision")
    if revision and revision != want and not os.path.isdir(repo):
        if identity.is_commit_sha(revision) or not want:
            raise PackSourceMismatch(
                f"refusing to load: pack {pack_dir} is bound to {identity.describe(src)}, "
                f"but revision {revision} was asked for. Serve the pack at its own "
                "revision (or pass no revision: the pack's binding wins), or pack that "
                "revision.")
        other = _cached_commit(repo, revision)
        if other != want:
            raise PackSourceMismatch(
                f"refusing to load: pack {pack_dir} is bound to {identity.describe(src)}, "
                f"but {repo}@{revision} "
                + (f"resolves to commit {other} on this machine" if other else
                   "cannot be resolved to a commit without the network")
                + ". Pass no revision (the pack's binding wins) or its commit.")
    if os.path.isdir(repo):
        got = identity.checkpoint_identity(repo, repo)
        if got["digest"] != src.get("digest"):
            raise PackSourceMismatch(
                f"refusing to load: pack {pack_dir} is bound to {identity.describe(src)}, "
                f"but the checkpoint at {repo} is {identity.describe(got)} — its "
                "safetensors headers, config, tokenizer files or image processor configs "
                "differ from the ones the pack was cut from. Serve the pack with the "
                "checkpoint it names, or re-pack this checkpoint.")
    try:
        return check_embedded_identity(pack_dir, pack_meta)
    except ValueError as e:
        raise PackSourceMismatch(f"refusing to load: {e}") from None


def resolve_pack_source(repo: str, revision: str | None, pack_meta: dict,
                        pack_dir: str | None = None, embedded: bool = True) -> str:
    """The ONE checkpoint snapshot a pack's raw half streams from — resolved
    from the identity the pack recorded (meta.json `source`), never from
    the caller's coordinates alone — or a PackSourceMismatch
    naming both identities, raised before anything has been allocated.

    Every pack `drinkme pack` writes is SELF-CONTAINED (meta.json
    `embedded`, codec/pack.write_embedded): it resolves to its own
    `checkpoint/` directory, checked against the same identity
    (_resolve_embedded) — no snapshot, no network. A pack without that
    block is refused outright, by name, here: there is no snapshot for it
    to fall back to.

    `embedded=False` is the one exception, for callers that need the real
    checkpoint regardless of what the pack carries — the bench's stock and
    twin arms read the packed Linears' source weights too (resolve_source).
    There the rule is: the pack pins a commit (hub) or a content digest
    (local). A caller's `revision=None` means "whatever the pack is bound
    to", so a branch that moved on the hub since the pack was cut changes
    nothing — the pinned snapshot is fetched by sha if the cache lost it. An
    EXPLICIT revision has to agree: a different commit sha refuses outright;
    a branch/tag name is resolved and its commit compared. The snapshot the
    call returns is then digested and compared with the pack's, so a local
    directory with different contents (or a hub snapshot whose files
    differ) refuses too, whatever its name.

    A pack without a `source` block (a tool-written pack — save_pack_dir
    with no identity; `drinkme pack` always records one) resolves the
    caller's coordinates, with one warning naming the pack; its hfRepo is
    still held to the caller's when both are hub ids, since that comparison
    needs no snapshot to be right."""
    from ..codec import identity
    from ..codec.pack import is_self_contained, repack_command

    src = pack_meta.get("source")
    where = pack_dir or "<pack>"
    check_pack_repo(repo, pack_meta, pack_dir)
    if embedded:
        if pack_dir is None or not is_self_contained(pack_meta):
            raise PackSourceMismatch(
                f"refusing to load: {where} carries no embedded checkpoint (meta.json "
                "has no `embedded` block), so nothing holds its unpacked tensors. Repack it: `"
                + repack_command(pack_meta.get("hfRepo") or repo) + "`.")
        return _resolve_embedded(repo, revision, pack_meta, pack_dir)
    if not src:
        _warn(f"unbound pack (no source identity recorded): {where} — raw tensors "
              f"stream from {repo}@{revision or 'main'} as resolved on this machine, "
              "unpinned. `drinkme pack` records the checkpoint identity; re-pack "
              "to bind the pack to the checkpoint it was cut from.")
        return snapshot_dir(repo, revision)

    if os.path.isdir(repo):
        snap = repo
    elif src.get("kind") == "hub" and src.get("revision"):
        want = src["revision"]
        if revision and revision != want:
            if identity.is_commit_sha(revision):
                raise PackSourceMismatch(
                    f"refusing to load: pack {where} is bound to "
                    f"{identity.describe(src)}, but revision {revision} was "
                    "asked for. Serve the pack at its own revision (or pass no "
                    "revision: the pack's binding wins), or pack that revision.")
            other = identity.hub_revision_of(snapshot_dir(repo, revision))
            if other != want:
                raise PackSourceMismatch(
                    f"refusing to load: pack {where} is bound to "
                    f"{identity.describe(src)}, but {repo}@{revision} resolves to "
                    f"commit {other} on this machine. Pass no revision (the pack's "
                    "binding wins), or pack that revision.")
        snap = snapshot_dir(repo, want)  # snapshot_dir announces any download itself
    else:
        snap = snapshot_dir(repo, revision)  # a local-dir pack served by hub id

    got = identity.checkpoint_identity(snap, repo)
    if src.get("identityVersion") != got["identityVersion"]:
        raise PackSourceMismatch(
            f"refusing to load: pack {where} records its source with identity "
            f"version {src.get('identityVersion')}; this build computes version "
            f"{got['identityVersion']}, so the two cannot be compared. Run "
            "`drinkme verify --pack-dir ...` on a build matching the pack, or "
            "re-pack.")
    if got["digest"] != src.get("digest"):
        raise PackSourceMismatch(
            f"refusing to load: pack {where} is bound to {identity.describe(src)}, "
            f"but the checkpoint at {snap} is {identity.describe(got)} — its "
            "safetensors headers, config, tokenizer files or image processor configs "
            "differ from the ones the pack was cut from. Serve the pack with the "
            "checkpoint it names, or re-pack this checkpoint.")
    return snap


def resolve_source(repo: str, revision: str | None, pack_dir: str | None) -> tuple[str, str]:
    """The ONE snapshot a benchmark loads EVERY arm and the tokenizer from,
    and its immutable revision — resolved once, before any arm loads.

    With a pack, the pack's bound source IS the source (resolve_pack_source:
    a caller's revision=None means "the pack's"; a pinned revision that
    disagrees refuses with both identities named). Without one, the
    caller's coordinates resolve here, once, and every arm reads the
    directory this returns — so stock and twin cannot follow the current
    default ref while the compressed arm follows the pack's older commit
    (with revision=None the two resolution paths returned different
    snapshots whenever a pack was bound to an older commit than the local
    default). The revision is
    resolved_revision(snap) — the commit sha for a hub snapshot, the
    `local-` digest for a directory — what a record's model.revision
    carries, never the menu's coordinate."""
    if pack_dir is not None:
        with open(os.path.join(pack_dir, "meta.json")) as f:
            pack_meta = json.load(f)
        # the real snapshot even for a self-contained pack: the stock and
        # twin arms read the weights the pack compressed
        snap = resolve_pack_source(repo, revision, pack_meta, pack_dir, embedded=False)
    else:
        snap = snapshot_dir(repo, revision)
    return snap, resolved_revision(snap)
