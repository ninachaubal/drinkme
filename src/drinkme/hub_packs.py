"""Published packs: `drinkme serve` with no local pack fetches one someone
published on the Hugging Face Hub, instead of downloading the BF16
checkpoint and packing it, and checks it against the upstream publisher's
own hashes before serving it (docs/serve.md#published-packs).

WHERE A PACK LIVES: one Hub repo per model, `<namespace>/<model>-drinkme`
(<model> is the source repo's name: Qwen3-8B), and in it one directory per
pack, `v<format>/<profile>/<source commit>/` (the pack format's version, which by
the versioning policy is drinkme's major), holding the pack directory as
`drinkme pack` writes it. The reasons:

  - the source commit is in the path, all 40 hex digits: the lookup asks for
    the commit this machine resolved, so a moved upstream branch finds
    another path or nothing, never a pack cut from another commit; and the
    pack's own meta.json `source` is held to that commit before the rest of
    it downloads (and again by serve.check_pack_dir's identity refusal)
  - the profile and the pack format version are in the path: sip and gulp
    sit side by side, and so will a later format, which this build never
    asks for (it reads format 1 only)
  - one repo per model, because the model's licence governs its pack: the
    repo's card carries the upstream licence and its text, and the Hub
    keeps licence metadata per repo
  - directories on one branch rather than a branch per pack: every pack is
    in the repo's file list, the card links each, and a hand download is
    `hf download <repo> --include '<path>/*'`

THE NAMESPACE is PACK_HUB_NAMESPACE, `drinkme-packs` (None
means no lookup). DRINKME_PACK_HUB overrides it: a namespace
(`org`), or a whole repo id (`org/name`), which then holds every model's
packs at the same paths (a mirror, or a test repo); `off` disables the
lookup. DRINKME_NO_HUB_PACK=1 is `serve --no-hub-pack`.

WHAT SERVE DOES WITH ONE: download into a staging directory beside the
cache entry, run the pack's own gates (format, identity, every file hash),
rebuild every upstream weight file from it and compare with the Hub's
published sha256 (upstream.py), and only then rename it into the cache,
with the receipt and `hub-pack.json` (where it came from). A MISMATCH is
refused loudly and nothing of the pack is kept; any other failure is one
line, and serve packs locally as it would have without a hub. When the Hub
cannot be asked for the upstream hashes, the pack is served on its file-hash
check, said so, and checked again at the next serve.
"""

from __future__ import annotations

import datetime
import json
import os
import shutil
import sys
import time

# the Hub namespace published packs live under; None: no lookup
PACK_HUB_NAMESPACE: str | None = "drinkme-packs"
PACK_HUB_ENV = "DRINKME_PACK_HUB"
NO_HUB_PACK_ENV = "DRINKME_NO_HUB_PACK"
PACK_REPO_SUFFIX = "-drinkme"
ORIGIN_FILE = "hub-pack.json"
HUB_TIMEOUT_S = 30.0
# files fetched at once. A pack is one file per tensor (206 files for
# Qwen3-0.6B, 3 MB each on average), and each costs a round trip or two
# before its bytes: from Hawaii, the 0.95 GB 0.6B pack took 55 s at
# huggingface_hub's default 8 and 34-48 s at 32-64 (measured 09-29).
DOWNLOAD_WORKERS = 32


def pack_hub() -> str | None:
    """DRINKME_PACK_HUB, else PACK_HUB_NAMESPACE; None when neither names
    one (or the variable says `off`)."""
    raw = os.environ.get(PACK_HUB_ENV, "").strip()
    if raw.lower() in ("off", "0", "none"):
        return None
    return raw or PACK_HUB_NAMESPACE or None


def disabled_by_env() -> bool:
    return os.environ.get(NO_HUB_PACK_ENV, "").strip() not in ("", "0")


def pack_repo(hub: str, repo: str) -> str:
    """The Hub repo that holds `repo`'s packs: `<hub>/<model>-drinkme`, or
    `hub` itself when it is a whole repo id."""
    if "/" in hub:
        return hub
    return f"{hub}/{repo.rstrip('/').split('/')[-1]}{PACK_REPO_SUFFIX}"


def pack_path(commit: str, profile: str, version: int | None = None) -> str:
    """The directory in the pack repo holding the pack of `commit` at `profile`."""
    from .codec.pack import FORMAT_VERSION

    return f"v{FORMAT_VERSION if version is None else version}/{profile}/{commit}"


def is_hub_pack(pack_dir: str) -> bool:
    return os.path.isfile(os.path.join(pack_dir, ORIGIN_FILE))


def _say(msg: str) -> None:
    print(f"[drinkme] {msg}", flush=True)


def _loud(msg: str) -> None:
    print(f"[drinkme] {msg}", file=sys.stderr, flush=True)


def _gb(n: int) -> str:
    return f"{n / 1e9:.2f} GB"


class _Skip(Exception):
    """Not this pack: the one line saying why, and serve packs locally."""


def _listing(api, prepo: str, ppath: str):
    """(pack repo commit, {path in repo: size}) for the pack directory at
    `ppath`, or _Skip naming why there is none."""
    from huggingface_hub.errors import (EntryNotFoundError, RepositoryNotFoundError,
                                        RevisionNotFoundError)
    from huggingface_hub.hf_api import RepoFile

    try:
        rev = api.model_info(prepo, timeout=HUB_TIMEOUT_S).sha
        files = {f.path: int(f.size or 0) for f in
                 api.list_repo_tree(prepo, path_in_repo=ppath, recursive=True, revision=rev)
                 if isinstance(f, RepoFile)}
    except (RepositoryNotFoundError, RevisionNotFoundError):
        raise _Skip(f"no published pack repo {prepo}") from None
    except EntryNotFoundError:
        files = {}
    except Exception as e:  # noqa: BLE001 — offline, timeout: the full flow may still work
        raise _Skip(f"the pack hub did not answer ({type(e).__name__})") from None
    if f"{ppath}/meta.json" not in files:
        raise _Skip(f"no published pack at {prepo}/{ppath}")
    return rev, files


def _check_meta(meta: dict, repo: str, commit: str, profile: str, where: str) -> None:
    """The published meta.json names this checkpoint, commit and profile,
    in a format this build reads — before the rest of the pack downloads."""
    from .codec import identity
    from .codec.pack import refusal_for_meta

    refusal = refusal_for_meta(meta, where)
    if refusal is not None:
        raise _Skip(f"the published pack is not one this drinkme reads ({refusal})")
    src = meta.get("source") or {}
    if (src.get("kind"), src.get("repo"), src.get("revision")) != ("hub", repo, commit) \
            or meta.get("hfRepo") != repo:
        raise _Skip(f"the published pack at {where} is bound to {identity.describe(src)}, "
                    f"not {repo}@{commit[:12]}")
    if meta.get("profile") != profile:
        raise _Skip(f"the published pack at {where} is {meta.get('profile')}, not {profile}")


def _bf16_bytes(listing: dict | None) -> int | None:
    if not listing:
        return None
    n = sum(int(v.get("size") or 0) for k, v in listing.items()
            if k.endswith(".safetensors") and "/" not in k)
    return n or None


def fetch(repo: str, commit: str | None, dest: str, profile: str) -> str | None:
    """Install the published pack of `repo`@`commit` at `profile` as `dest`
    and return `dest`; None, after one line saying why (none when no pack
    hub is set), when serve should pack locally instead."""
    hub = pack_hub()
    if hub is None or os.path.isdir(repo):  # a local checkpoint has nothing published
        return None
    from .codec import identity

    if not identity.is_commit_sha(commit):
        _say(f"no published pack lookup: {repo} is not pinned to a commit here — packing locally")
        return None
    t0 = time.time()
    prepo = pack_repo(hub, repo)
    ppath = pack_path(commit, profile)
    staging = f"{dest}.hub-{os.getpid()}"
    try:
        return _fetch(repo, commit, dest, profile, prepo, ppath, staging, t0)
    except _Skip as e:
        _say(f"{e} — packing locally")
        return None
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def _fetch(repo, commit, dest, profile, prepo, ppath, staging, t0) -> str | None:
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download

    from . import upstream

    api = HfApi()
    rev, files = _listing(api, prepo, ppath)
    os.makedirs(staging, exist_ok=True)
    try:
        mp = hf_hub_download(prepo, f"{ppath}/meta.json", revision=rev, local_dir=staging)
        with open(mp) as f:
            meta = json.load(f)
    except Exception as e:  # noqa: BLE001
        raise _Skip(f"could not read the published pack's meta.json ({type(e).__name__})") from None
    _check_meta(meta, repo, commit, profile, f"{prepo}/{ppath}")
    try:
        listing, unavailable = upstream.hub_listing(repo, commit), None
    except upstream.UpstreamUnavailable as e:
        listing, unavailable = None, str(e)
    skip = {f"{ppath}/{upstream.RECEIPT_FILE}", f"{ppath}/{ORIGIN_FILE}"}
    pack_bytes = sum(n for p, n in files.items() if p not in skip)
    bf16 = _bf16_bytes(listing)
    _say(f"no pack for {repo} yet — downloading the published {profile} pack from {prepo}: "
         f"{_gb(pack_bytes)}"
         + (f", {pack_bytes / bf16:.0%} of the {_gb(bf16)} BF16 checkpoint" if bf16 else ""))
    try:
        snapshot_download(prepo, revision=rev, local_dir=staging, allow_patterns=[f"{ppath}/*"],
                          ignore_patterns=sorted(skip), max_workers=DOWNLOAD_WORKERS)
    except Exception as e:  # noqa: BLE001
        raise _Skip(f"the pack download failed ({type(e).__name__}: {str(e)[:120]})") from None
    got = os.path.join(staging, *ppath.split("/"))
    shutil.rmtree(os.path.join(got, ".cache"), ignore_errors=True)
    _gate(repo, commit, got)
    if listing is None:
        _loud(f"upstream verification unavailable ({unavailable}) — serving on the pack's "
              "file-hash check alone; it is checked against the upstream hashes at the next "
              "serve, or run `drinkme verify --upstream`")
    else:
        n = sum(1 for k in listing if k.endswith(".safetensors") and "/" not in k)
        _say(f"checking it against {repo}@{commit[:12]}'s published sha256: rebuilding "
             f"{n} weight file{'s' if n != 1 else ''} from the pack")
        try:
            v = upstream.verify_upstream(got, listing=listing)
        except (ValueError, OSError) as e:
            raise _Skip(f"upstream verification could not run ({e})") from None
        _judge(v, f"{prepo}/{ppath}")
    _install(got, dest, {"repo": prepo, "path": ppath, "revision": rev})
    _say(f"downloaded {'and verified ' if listing is not None else ''}in "
         f"{time.time() - t0:.0f}s -> {dest}")
    return dest


def _gate(repo: str, commit: str, pack_dir: str) -> None:
    """The pack's own gates, as a load would run them: format and identity
    (serve.check_pack_dir), then every file hash and the manifest
    (verify_pack), trunk and `mtp/`."""
    from . import exitcodes
    from .codec.pack import verify_pack
    from .serve import check_pack_dir

    try:
        check_pack_dir(repo, commit, pack_dir)
        verify_pack(pack_dir, progress=lambda *_: None)
    except (exitcodes.DrinkmeExit, ValueError, OSError) as e:
        raise _Skip(f"the published pack failed its own checks ({e})") from None


def _judge(v, where: str) -> None:
    """MATCH: the summary line. MISMATCH: refused, loudly, naming the
    file. Anything else (NOT COVERED): _Skip with the summary."""
    from . import upstream

    if v.verdict == upstream.MISMATCH:
        bad = v.mismatched()[0]
        _loud(f"REFUSED the published pack {where}: {bad.name} does not rebuild to "
              f"the file {v.repo}@{v.commit[:12]} publishes ({bad.reason}). Deleted it.")
        raise _Skip("the published pack is refused")
    if not v.ok:
        raise _Skip(f"the published pack is not verified: {v.summary()}")
    _say(v.summary())


def _install(got: str, dest: str, origin: dict) -> None:
    """Rename the checked pack into the cache, with where it came from."""
    origin = {**origin, "downloadedAt": datetime.datetime.now(datetime.timezone.utc)
              .strftime("%Y-%m-%dT%H:%M:%SZ")}
    with open(os.path.join(got, ORIGIN_FILE), "w") as f:
        json.dump(origin, f, indent=1)
        f.write("\n")
    if os.path.isdir(dest) and not os.listdir(dest):
        os.rmdir(dest)
    try:
        os.rename(got, dest)
    except OSError as e:
        if os.path.isfile(os.path.join(dest, "meta.json")):
            return  # another drinkme installed this pack meanwhile; its checks ran there
        raise _Skip(f"could not move the pack into {dest} ({e})") from None


def recheck(repo: str, pack_dir: str) -> bool:
    """A cached pack that came from the pack hub without a current receipt
    (the Hub could not be asked when it arrived): verify it against the
    upstream hashes now. True: serve it (MATCH, or the Hub still cannot be
    asked, said so). False: it did not verify and has been deleted, and
    serve packs locally."""
    from . import upstream

    with open(os.path.join(pack_dir, "meta.json")) as f:
        src = json.load(f).get("source") or {}
    try:
        listing = upstream.hub_listing(src.get("repo"), src.get("revision"))
    except upstream.UpstreamUnavailable as e:
        _loud(f"upstream verification unavailable ({e}) — serving {pack_dir} on its "
              "file-hash check alone")
        return True
    _say(f"checking the published pack against {src.get('repo')}@{str(src.get('revision'))[:12]}"
         "'s published sha256")
    try:
        v = upstream.verify_upstream(pack_dir, listing=listing)
    except (ValueError, OSError) as e:
        _loud(f"upstream verification could not run ({e}) — serving {pack_dir} on its "
              "file-hash check alone")
        return True
    try:
        _judge(v, pack_dir)
    except _Skip as e:
        shutil.rmtree(pack_dir, ignore_errors=True)
        _say(f"{e} — packing locally")
        return False
    return True
