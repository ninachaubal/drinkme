"""SSD-persisted prefix slots: the cold tier under engines.py's live slots.

The prefix cache gave the engine N whole cache states in GPU memory. This file gives those
states somewhere to go when they are evicted, and somewhere to come back
from after a bounce — so a prefix the machine has already paid for (Claude Code's
22k system prompt, a long agent transcript) is read off an SSD instead of
recomputed. oMLX's framing, which is the borrow:
"Coding agents invalidate the KV cache dozens of times per session. oMLX
persists every cache block to SSD — so when the agent circles back to a
previous prefix, it's restored from disk in milliseconds, not recomputed from
scratch."

THREE TIERS, ONE RULE. live (GPU, engines._Slot) -> warm (host RAM, the
staging copy an eviction has to make anyway) -> cold (safetensors on disk).
The rule they all obey is the DeltaNet law from the prefix cache, unchanged: reuse only
when the incoming prompt EXTENDS the whole stored state. The recurrent state
`[B, v_heads, k_dim, v_dim]` has NO sequence dimension, so it cannot be
rewound to an earlier position — not in RAM, not on disk, not by any amount
of block bookkeeping. A stored slot is usable WHOLE or not at all, and this
file never truncates one.

KEYING IS CHAIN-HASHED BLOCKS, vLLM's (quoting vLLM's own docs/design/prefix_caching.md):
"hash(tuple[components])... Parent
hash value... Block tokens... Extra hashes". Each stored slot carries the
chain of its FULL 256-token blocks; block h_d is hash(h_{d-1}, tokens). Two
prefixes share block d iff they share every token up to (d+1)*256, so
testing one hash tests the whole walk, and finding the deepest shared block
is a walk down the chain rather than an LCP scan over every stored prefix
(engines.pick_slot's per-token scan is right for N live slots and the wrong
shape for a disk index). The tail — the at-most-255 tokens past the last
block boundary — rides in the header as plain ids, so the final "is this
really a prefix of the prompt" test is exact and costs no tensor read.

ONE FILE PER SLOT, not per block: slots are few and large (a 27B slot at ctx
32768 is 2.14 GiB, of which the whole DeltaNet state is 147.75 MiB: the
boot line's 154,927,232 B of slot state, the recurrent state in float32),
and the block chain is the index, not the storage layout. A slot's context
checkpoints (serving/ctx_checkpoints.py) are a second file beside it.

WHAT A SLOT FILE IS. One directory per stored prefix:

    <root>/<model>@<pack_id or arm>@<revision>-<attention digest>/<n_ids>-<digest>/
        header.json         identity, chain, tail, sizes, LRU stamp
        cache.safetensors   every tensor the cache's layers hold, + the ids

The tree is PRIVATE — directories 0700, files 0600, from creation and
regardless of the umask, and an existing store is tightened when it is
opened: the header's `tail` is the user's prompt in plain
ids, and the public tokenizer decodes it.

The header's identity block (format_version, model_id, pack_id, arm, dtype,
transformers version, revision, attention) is what makes a stale file
SKIPPED rather than loaded: the tensors are a transformers cache layout
serving one exact set of weights under one positional transformation, and a
file that does not name the weights being served has nothing to say about
them. `pack_id` is the pack's hashes — the sha256 map meta.json already records per
tensor, folded to one digest — so it changes when the served bytes change
and only then. `revision` is the resolved, immutable revision of the
checkpoint the weights were read from (the hub commit sha, or a digest of
a local directory — checkpoint.resolved_revision), for BOTH arms: the hashes
cover a pack's compressed tensors, not the embeddings and norms streamed
from the checkpoint beside them, and a stock engine has no pack hashes at all.
`attention` is the effective attention configuration — rope_theta, the
full rope_parameters/rope_scaling dict (YaRN factors included), the
window, sliding_window, layer_types — because keys in a cache were rotated
under ONE of those and are wrong under any other. The
directory name carries all of it too, so two identities never share an LRU.
A header that lacks any identity field is refused, never read as "probably
the same".

RESTORE IS VERIFIED BEFORE IT IS APPLIED. The engine builds a fresh cache at
the saved allocation and drives ONE token through it (transformers-5 layers
allocate lazily; a fresh cache honestly holds only its counters), which gives
a live cache whose tensor set, shapes and dtypes are the truth of THIS
process. The file has to match that exactly or it is refused with a log line.
Then, and only then, the saved tensors are written in — including
`cumulative_length`, the internal counter transformers-5's static layers
actually write at, which the caller re-checks against the id count it is
about to reuse. A sliding-window layer ALSO carries its python
`cumulative_length_int` alongside that tensor: the
tensor freezes once the ring is full, the int is what the mask actually
reads (serving/mtp.py's module docstring, surface 1b), and a restore that only
wrote the tensor handed the layer back a fresh layer's own int instead.

The read stages through anonymous memory: `get_tensor` hands back a view onto
the MAP_PRIVATE mapping, and ROCm's pageable H2D from file-backed pages ran at
~33 MB/s (the same fix engines.load_stock makes). clone() first, .to(device)
second, one tensor at a time, so peak extra host memory is one tensor rather
than one slot.

WRITES ARE ASYNC, AND THEY COMPLETE. An eviction copies the cache to host RAM
on the calling thread (it has to: the GPU tensors are about to be reused) and
hands the copy to one writer thread. `flush()` waits for the queue, and both
serve.py's SIGTERM handler and an atexit belt call it — "the write must
complete before the process exits" is the requirement, so nothing here is
fire-and-forget.

Off by env: DRINKME_SLOT_DIR=off. Capped by DRINKME_SLOT_DISK_GIB (LRU on
disk). The host-RAM warm tier is OFF by default (DRINKME_SLOT_RAM_GIB=0):
this machine has UNIFIED memory, so host RAM held here is memory the model cannot
have, and a warm tier that quietly competes with the weights is the fit check
losing an argument it was never shown.
"""

from __future__ import annotations

import atexit
import hashlib
import json
import os
import queue
import shutil
import sys
import threading
import time
from array import array

import torch

# The pack-id helper lives in the torch-free codec/identity.py (the Metal
# lane's compressed loader needs it and must not import torch through this
# module); re-exported here for the engines and tests that import it from
# slotstore.
from ..codec.identity import pack_id_from_meta  # noqa: F401

# Bumped when the on-disk shape changes in a way an older reader would get
# wrong. A file whose version is not this one is skipped, never migrated:
# a KV cache is a derived artifact, and recomputing it is always available.
FORMAT_VERSION = 1

# vLLM's default block size, and the granularity the chain hashes at. Only
# FULL blocks are hashed; the tail rides in the header as ids.
BLOCK_TOKENS = 256

# LRU cap on the cold tier. Generous because a slot is large and an SSD is
# not: two 27B slots at ctx 32768 are 4.15 GiB, and the whole point is to
# keep more prefixes than fit in the accelerator.
DEFAULT_DISK_GIB = 64.0
DEFAULT_RAM_GIB = 0.0

_GIB = 1024 ** 3
DEFAULT_ROOT = "~/.cache/drinkme/slots"
# a slot's context checkpoints (serving/ctx_checkpoints.py), beside
# cache.safetensors: a reader from before them never opens it, and a slot
# stored without them has none
CKPT_FILE = "checkpoints.safetensors"


def _log(msg: str, err: bool = False) -> None:
    print(f"[drinkme.slots] {msg}", file=sys.stderr if err else sys.stdout,
          flush=True)


# The tree's modes. A header carries `tail` — up to 255 real
# token ids past the last block boundary, decodable with the public
# tokenizer — so a slot file is the user's prompt, and the store keeps it the
# way ~/.ssh keeps a key: directories 0700, files 0600, set at creation and
# never left to the umask. An HTTP API key does not protect a disk artifact.
DIR_MODE = 0o700
FILE_MODE = 0o600


def _mkdir_private(path: str) -> None:
    """os.makedirs with every level THIS call creates at DIR_MODE, chmod'd
    after the mkdir so the umask has no say. Levels that already exist are
    left alone here — a pre-existing root may be a directory the operator
    owns for other reasons (DRINKME_SLOT_DIR=/some/shared/place); what is
    ours, and gets tightened, is the store's own tree (`_tighten`)."""
    todo = []
    cur = os.path.abspath(path)
    while not os.path.isdir(cur):
        todo.append(cur)
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent
    for d in reversed(todo):
        try:
            os.mkdir(d, DIR_MODE)
        except FileExistsError:
            continue  # raced with the writer thread; the chmod still applies
        os.chmod(d, DIR_MODE)


def _open_private(path: str):
    """A text file for writing, created at FILE_MODE (or, if the name is
    already there — a stale .tmp — truncated and chmod'd to it)."""
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, FILE_MODE)
    try:
        os.fchmod(fd, FILE_MODE)
    except OSError:
        os.close(fd)
        raise
    return os.fdopen(fd, "w")


def slot_root() -> str | None:
    """Where the cold tier lives, or None for "no cold tier".

    Unset = the default cache path, on purpose: a prefix the
    machine has already paid for should survive a bounce without anyone having
    configured anything. DRINKME_SLOT_DIR=off (or "", or "0") turns it off
    for a run that must be cold in its own terms — the benches say it that
    way, and engines.reset_prefix_cache() says it in code."""
    raw = os.environ.get("DRINKME_SLOT_DIR")
    if raw is None:
        return os.path.expanduser(DEFAULT_ROOT)
    raw = raw.strip()
    if raw.lower() in ("", "0", "off", "none"):
        return None
    return os.path.expanduser(raw)


def _gib_from_env(name: str, default: float) -> float:
    """A GiB budget from the environment. Garbage warns and falls back rather
    than dying — engines.slots_from_env's posture, for the same reason: a typo
    in a size must not take the server down."""
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        val = float(raw)
    except ValueError:
        val = -1.0
    if val < 0:
        _log(f"{name}={raw!r} is not a size in GiB — using {default}", err=True)
        return default
    return val


def disk_cap_bytes() -> int:
    return int(_gib_from_env("DRINKME_SLOT_DISK_GIB", DEFAULT_DISK_GIB) * _GIB)


def ram_cap_bytes() -> int:
    return int(_gib_from_env("DRINKME_SLOT_RAM_GIB", DEFAULT_RAM_GIB) * _GIB)


# ------------------------------------------------------------- the keying --


def _digest(parent: str, tokens) -> str:
    """One link of the chain: hash(parent_hash, block_tokens). The ids go in
    as their raw 8-byte machine representation rather than text — same content,
    no formatting to agree on, and a block hashes in microseconds."""
    h = hashlib.sha256()
    h.update(parent.encode("ascii"))
    h.update(b"|")
    h.update(array("q", tokens).tobytes())
    return h.hexdigest()[:32]


def block_chain(ids, block: int = BLOCK_TOKENS) -> list[str]:
    """vLLM's chain: one hash per FULL block, each folded into the next.

    Because h_d covers h_{d-1}, two id sequences agree on h_d iff they agree
    on every token up to (d+1)*block — so a shared hash at depth d IS the
    walk, and the deepest shared hash is the longest shared block boundary.
    A partial trailing block is deliberately not hashed: it is not a boundary
    anything can be shared at, and it rides in the header as ids instead."""
    out: list[str] = []
    parent = ""
    for i in range(0, len(ids) - block + 1, block):
        parent = _digest(parent, ids[i:i + block])
        out.append(parent)
    return out


def prefix_key(ids) -> str:
    """The directory name for a stored prefix: its length and a digest of
    the whole id sequence. Deterministic, so re-storing the same prefix
    overwrites its own file instead of growing a second copy."""
    h = hashlib.sha256(array("q", ids).tobytes()).hexdigest()[:24]
    return f"{len(ids)}-{h}"


# The attributes of a text config that decide what a cached key/value
# MEANS. Read by attention_identity, present-or-None so a header
# written by a config that lacks one still compares equal to itself and
# unequal to one that has it. rope_parameters is transformers-5's canonical
# dict (rope_theta, rope_type, YaRN's factor / original_max_position_
# embeddings / attention_factor / beta_fast / beta_slow, partial_rotary_
# factor, and on gemma-4 one dict per layer type); rope_scaling is its alias
# and rope_theta its old top-level home — all three carried so a config that
# only has one of them is still named. The rest are the window and which
# layers see how much of it. NOT carried: _attn_implementation (sdpa vs
# eager rotate identically and lay the StaticCache out identically) and
# anything structural (heads, head_dim, layer count) — those change tensor
# SHAPES, and state_mismatch refuses a shape change against a live cache.
ATTENTION_FIELDS = ("rope_theta", "rope_parameters", "rope_scaling",
                    "max_position_embeddings", "sliding_window", "layer_types",
                    "use_sliding_window", "max_window_layers",
                    "partial_rotary_factor", "rope_local_base_freq",
                    "attention_chunk_size")


def attention_identity(config) -> dict:
    """The effective attention configuration of a loaded model, as the
    store keys on it: ATTENTION_FIELDS read off the TEXT config (a
    multimodal wrapper's get_text_config(), the config itself otherwise),
    after whatever --rope-scaling did to it, JSON-normalised so the dict in
    memory equals the dict a header round-trips."""
    tcfg = config.get_text_config() if hasattr(config, "get_text_config") else config
    out = {}
    for name in ATTENTION_FIELDS:
        val = getattr(tcfg, name, None)
        out[name] = json.loads(json.dumps(val, sort_keys=True, default=str))
    return out


def _safe_name(s: str) -> str:
    return "".join(c if c.isalnum() or c in "._-" else "-" for c in s)


def store_ident(model_id: str, arm: str, pack_id: str | None, revision: str,
                attention: dict) -> str:
    """One directory per (weights, arm, revision, attention configuration).
    Slots computed against uncompressed weights and slots computed against
    a pack are the same numbers by the project's own claim, but they are
    not the same CLAIM, and an A/B that quietly shared a KV cache would
    stop measuring the codec. The revision and a digest of the attention
    configuration are in the name too, so two identities
    never share an LRU or a disk cap — the header check is the truth, the
    directory is what keeps the truths apart."""
    safe = model_id.replace("/", "--").replace(os.sep, "--")
    attn = hashlib.sha256(json.dumps(attention, sort_keys=True).encode("utf-8")).hexdigest()[:12]
    return f"{safe}@{pack_id or arm}@{_safe_name(revision)[:40]}-{attn}"


# ---------------------------------------------------- the cache's tensors --


def _is_sliding_layer(layer) -> bool:
    """transformers' StaticSlidingWindowLayer surface: a python
    `cumulative_length_int` beside the tensor `cumulative_length` — the same
    two-counter surface serving/mtp.py's `_is_sliding_layer` rewinds for a
    rejected draft. Recognized by the surface, never
    by model name, so a family this file has never heard of that grows the
    surface is still carried correctly."""
    return hasattr(layer, "cumulative_length_int") and hasattr(layer, "cumulative_length")


def cache_state(cache) -> tuple[dict[str, torch.Tensor], dict[str, bool], dict[str, int]]:
    """Every tensor a cache's layers hold, flattened to safetensors keys, the
    booleans that say which of them are live, and the
    one int a restore cannot do without: a sliding-window layer's python
    `cumulative_length_int`.

    Walks vars() rather than naming fields — engines.cache_bytes' rule, for
    the same reason: a transformers bump that adds a cache surface gets
    CARRIED rather than silently dropped, which is the direction a serializer
    has to fail in. On the hybrid this picks up the StaticLayers' keys,
    values and `cumulative_length` (the internal write counter, a 0-dim
    tensor, and the thing extends-only reuse actually turns on) together with
    the LinearAttentionLayers' conv_states and recurrent_states dicts,
    without knowing that either kind exists.

    Ints, dtypes and devices are otherwise deliberately NOT carried: they are
    structural, fixed by the construction of the cache a state is restored
    into, and a stale one would describe a cache that is not there.
    `cumulative_length_int` is the one exception, because it is not
    structural — it is the mask's q_offset (mtp.py's docstring, surface 1b)
    and a fresh layer's own value (the prime token's 1) is simply wrong."""
    tensors: dict[str, torch.Tensor] = {}
    flags: dict[str, bool] = {}
    ints: dict[str, int] = {}
    for i, layer in enumerate(cache.layers):
        for name, value in vars(layer).items():
            items = value.items() if isinstance(value, dict) else ((None, value),)
            for k, v in items:
                key = f"{i}.{name}" if k is None else f"{i}.{name}.{k}"
                if isinstance(v, torch.Tensor):
                    tensors[key] = v
                elif isinstance(v, bool):
                    flags[key] = v
        if _is_sliding_layer(layer):
            ints[f"{i}.cumulative_length_int"] = int(layer.cumulative_length_int)
    return tensors, flags, ints


def state_mismatch(cache, tensors: dict, flags: dict, ints: dict) -> str | None:
    """None when a saved state can be written into this cache, else the
    reason, phrased for a log line. Compared against a LIVE cache — one the
    caller has already driven a token through — because that is the only
    honest description of what this process's transformers allocates.

    A sliding layer with no `cumulative_length_int` in `ints` (a slot file
    without the field) lands in `missing` the same as any
    other absent surface — refused here, never silently restored with the
    fresh layer's own counter."""
    want_t, want_f, want_i = cache_state(cache)
    missing = (sorted(set(want_t) - set(tensors)) + sorted(set(want_f) - set(flags))
              + sorted(set(want_i) - set(ints)))
    extra = (sorted(set(tensors) - set(want_t)) + sorted(set(flags) - set(want_f))
            + sorted(set(ints) - set(want_i)))
    if missing or extra:
        return f"cache layout differs (missing {missing[:3]}, extra {extra[:3]})"
    for key, live in want_t.items():
        saved = tensors[key]
        if tuple(saved.shape) != tuple(live.shape) or saved.dtype != live.dtype:
            return (f"{key}: saved {tuple(saved.shape)}/{saved.dtype} != live "
                    f"{tuple(live.shape)}/{live.dtype}")
    return None


def apply_state(cache, tensors: dict, flags: dict, ints: dict) -> None:
    """Write a verified state into a live cache, one tensor at a time.

    Each tensor is cloned out of whatever backing it arrived on before it
    moves: a safetensors `get_tensor` is a view onto the MAP_PRIVATE mapping,
    and ROCm's pageable H2D from file-backed pages measured ~33 MB/s
    (engines.load_stock's own fix) — page-cache -> anon is a
    memcpy, anon -> device is a staged DMA. Peak extra host memory is one
    tensor, never one slot."""
    from .kvcache import LiveStaticLayer

    for i, layer in enumerate(cache.layers):
        for name, value in list(vars(layer).items()):
            if isinstance(value, dict):
                for k in list(value):
                    key = f"{i}.{name}.{k}"
                    if key in tensors:
                        live = value[k]
                        value[k] = tensors[key].clone().to(live.device)
                    elif key in flags:
                        value[k] = flags[key]
            elif isinstance(value, torch.Tensor):
                key = f"{i}.{name}"
                if key in tensors:
                    setattr(layer, name, tensors[key].clone().to(value.device))
            elif isinstance(value, bool):
                key = f"{i}.{name}"
                if key in flags:
                    setattr(layer, name, flags[key])
        if _is_sliding_layer(layer):
            key = f"{i}.cumulative_length_int"
            if key in ints:
                layer.cumulative_length_int = ints[key]
        if isinstance(layer, LiveStaticLayer):
            # the live-window int follows the counter tensor just restored
            # (kvcache.py); derived, so no slot file carries it
            layer.resync()


def written_tokens(cache) -> int | None:
    """How many tokens the cache's layers say they hold — `get_seq_length()`,
    not the raw `cumulative_length` tensor: on a
    sliding-window layer that tensor FREEZES at the window once the ring is
    full (mtp.py's docstring, surface 1b) while the python
    `cumulative_length_int` — what `get_seq_length()` returns — keeps
    counting, so past the window the tensor under-reports. None when no
    layer keeps a counter at all (a purely recurrent family); a
    DISAGREEMENT between layers is a corrupt state and reads as -1 so a
    caller can refuse it."""
    seen = set()
    for layer in cache.layers:
        if hasattr(layer, "get_seq_length"):
            # the real surface: correct for both a StaticLayer (the tensor)
            # and a StaticSlidingWindowLayer (the int the tensor cannot see
            # past the window)
            seen.add(int(layer.get_seq_length()))
            continue
        c = getattr(layer, "cumulative_length", None)
        if isinstance(c, torch.Tensor):
            seen.add(int(c.reshape(-1)[0]) if c.numel() else 0)
        elif isinstance(c, int):
            seen.add(c)
    if not seen:
        return None
    return seen.pop() if len(seen) == 1 else -1


# ------------------------------------------------------------- the store --


# An entry's place in the writer's life cycle. The
# transitions are one-way and the sweeps read them:
#
#     PENDING ---- _write lands tensors + header ----> COMMITTED (evictable)
#        |                                                |
#        +-- write fails / entry forgotten --> FAILED     +-- disk cap --> EVICTED
#
# PENDING: put() queued it; the ONLY copy of the snapshot is `entry.ram`, and
# nothing may drop it — not the RAM cap (it is not a warm-tier copy, it is
# the payload), not the disk cap (there is no file yet). _write meeting a
# pending entry with no payload is a bug in this file, raised, never a quiet
# return. COMMITTED: the file is on disk; the entry is EVICTABLE — its file
# by the disk cap, its RAM copy (if the warm tier kept one) by the RAM cap.
# FAILED: the write raised, or the entry was forgotten before it landed;
# it is out of the index and counted in `failed`, which is what flush()
# reports. EVICTED: the disk cap removed a committed file. An entry the
# boot index read off disk starts COMMITTED. Only COMMITTED means "a
# caller may expect the file to be there".
PENDING, COMMITTED, FAILED, EVICTED = "pending", "committed", "failed", "evicted"


class _Entry:
    """One stored prefix, as the index knows it. `ram` holds the snapshot
    while it is pending and the host-RAM copy after that while the warm tier
    keeps it; `path` is where the write lands (or landed)."""

    __slots__ = ("key", "path", "n_ids", "alloc", "chain", "tail", "nbytes",
                 "used", "ram", "state", "ckpts", "free")

    def __init__(self, key: str, n_ids: int, alloc: int, chain: list[str],
                 tail: list[int], nbytes: int, used: float,
                 path: str | None = None, ram=None, state: str = COMMITTED,
                 ckpts: list[int] | None = None, free: bool = False):
        self.key, self.n_ids, self.alloc = key, n_ids, alloc
        self.chain, self.tail = chain, tail
        self.nbytes, self.used, self.path, self.ram = nbytes, used, path, ram
        self.state = state
        # the slot's context checkpoints' positions, and whether its cache
        # reaches any earlier position by length (serving/ctx_checkpoints.py)
        self.ckpts, self.free = list(ckpts or ()), free

    @property
    def pending(self) -> bool:
        return self.state == PENDING

    @property
    def committed(self) -> bool:
        return self.state == COMMITTED

    @property
    def evictable(self) -> bool:
        """What the caps may act on: a file that exists and a RAM copy that
        is a COPY. Only a committed entry is either."""
        return self.state == COMMITTED

    def extends_into(self, ids: list[int], block: int) -> bool:
        """Is this stored prefix a STRICT prefix of `ids`? The chain has
        already said the full blocks agree; this is the tail, exactly, plus
        the prefix cache's rule that at least one prompt token must be left to compute
        fresh logits from."""
        if not (0 < self.n_ids < len(ids)):
            return False
        start = len(self.chain) * block
        return self.tail == ids[start:self.n_ids]

    def reach(self, ids: list[int], chain: list[str], block: int, fit=None) -> int:
        """How many of `ids` (whose block chain is `chain`) this stored slot
        can serve: all of it when it extends into them, else what a rewind
        by length or one of its context checkpoints reaches at or below the
        prefix the two certainly share — every full block both chains agree
        on, and the tail too when every one of the slot's blocks agrees —
        leaving one prompt token to compute (ctx_checkpoints.Checkpoints.
        reach, with the prefix known only to the block where it is not
        exact). The engine recomputes it from the restored ids."""
        if self.extends_into(ids, block):
            return self.n_ids
        if not (self.free or self.ckpts):
            return 0
        d = 0
        while d < min(len(self.chain), len(chain)) and self.chain[d] == chain[d]:
            d += 1
        shared = d * block
        if d == len(self.chain):
            for a, b in zip(self.tail, ids[shared:]):
                if a != b:
                    break
                shared += 1
        target = min(shared, len(ids) - 1)
        if target <= 0:
            return 0
        if self.free:
            return fit(target) if fit is not None else target
        best = [c for c in self.ckpts if c <= target and (fit is None or fit(c) == c)]
        return max(best, default=0)


class SlotStore:
    """The warm+cold tiers for ONE (weights, arm) identity.

    Not thread-safe by accident: the engine calls into it under the HTTP
    layer's single generation lock, and the writer thread touches only the
    queue and the index under `_lock`."""

    def __init__(self, root: str, ident: str, header: dict,
                 cap_bytes: int | None = None, ram_bytes: int | None = None,
                 block: int = BLOCK_TOKENS):
        self.dir = os.path.join(root, ident)
        self.header = dict(header)
        self.cap = disk_cap_bytes() if cap_bytes is None else cap_bytes
        self.ram_cap = ram_cap_bytes() if ram_bytes is None else ram_bytes
        self.block = block
        self.enabled = True
        self._entries: dict[str, _Entry] = {}
        self._lock = threading.RLock()
        self._q: queue.Queue = queue.Queue()
        self._pending = 0
        self._idle = threading.Condition()
        self._worker: threading.Thread | None = None
        # saved: writes that reached disk. failed: writes that did not —
        # raised, or found their payload gone. The two are different words
        # on purpose: a cap EVICTION is the store choosing, a write FAILURE
        # is the store losing. `_unflushed_failures` is the slice of
        # `failed` the next flush() has to answer for.
        self.hits = self.misses = self.saved = self.failed = 0
        self._unflushed_failures = 0

    # -- the boot index ----------------------------------------------------

    def index(self) -> tuple[int, int]:
        """Scan the store, keep the headers that name these weights, and say
        so. (count, bytes). NOTHING is loaded here: N slots' worth of GPU
        memory at boot is exactly what the live tier already refuses to
        allocate, so the tensors wait for a request that matches."""
        found, bad, total = {}, 0, 0
        self._tighten()
        try:
            names = sorted(os.listdir(self.dir))
        except FileNotFoundError:
            names = []
        except OSError as e:
            _log(f"cannot read {self.dir} ({e}) — serving without a cold tier",
                 err=True)
            self.enabled = False
            return 0, 0
        for name in names:
            path = os.path.join(self.dir, name)
            entry, why = self._read_header(path, name)
            if entry is None:
                if why:
                    _log(f"skipping {name}: {why}", err=True)
                    bad += 1
                continue
            found[name] = entry
            total += entry.nbytes
        with self._lock:
            self._entries = found
        note = f", {bad} skipped" if bad else ""
        _log(f"cold tier {self.dir}: {len(found)} slot(s), "
             f"{total / _GIB:.2f} GiB{note} (cap {self.cap / _GIB:.0f} GiB)")
        return len(found), total

    def _tighten(self) -> None:
        """Every directory of the store's tree to DIR_MODE, every file to
        FILE_MODE — a store written by an older drinkme,
        or loosened by hand, is not assumed fixed by the new defaults. One
        log line when anything changed; best effort, since a mode we cannot
        set is not a reason to stop serving."""
        import stat

        if not os.path.isdir(self.dir):
            return  # nothing stored yet; the first write creates it private
        fixed, failed = 0, 0
        paths = [(self.dir, DIR_MODE)]
        for parent, dirs, files in os.walk(self.dir):
            paths += [(os.path.join(parent, d), DIR_MODE) for d in dirs]
            paths += [(os.path.join(parent, f), FILE_MODE) for f in files]
        for full, want in paths:
            if os.path.islink(full):
                continue  # not ours to follow
            try:
                if stat.S_IMODE(os.stat(full).st_mode) != want:
                    os.chmod(full, want)
                    fixed += 1
            except OSError:
                failed += 1
        if fixed or failed:
            note = f" ({failed} could not be changed)" if failed else ""
            _log(f"tightened permissions on {fixed} path(s) under {self.dir} "
                 f"to {DIR_MODE:04o}/{FILE_MODE:04o}{note}", err=bool(failed))

    def _read_header(self, path: str, name: str) -> tuple[_Entry | None, str]:
        """One stored directory -> an index entry, or (None, why). Every
        refusal is a skip: a slot file is a derived artifact, so the worst a
        broken one may cost is the prefill it would have saved."""
        hpath = os.path.join(path, "header.json")
        try:
            with open(hpath) as f:
                head = json.load(f)
        except FileNotFoundError:
            # a write that never finished (the tensors land before the
            # header): not an error, just nothing to index
            return None, "" if os.path.isdir(path) else "not a slot directory"
        except (OSError, ValueError) as e:
            return None, f"unreadable header ({type(e).__name__}: {e})"
        why = self._identity_mismatch(head)
        if why:
            return None, why
        tpath = os.path.join(path, "cache.safetensors")
        try:
            size = os.path.getsize(tpath)
        except OSError:
            return None, "header without tensors"
        if size != head.get("bytes"):
            return None, (f"tensor file is {size} B, header says "
                          f"{head.get('bytes')} B")
        ckpts = head.get("checkpoints") or []
        if ckpts:
            # a slot's context checkpoints ride in a second file, sized in
            # the header like the first (serving/ctx_checkpoints.py)
            try:
                ck_size = os.path.getsize(os.path.join(path, CKPT_FILE))
            except OSError:
                return None, "header lists checkpoints without their file"
            if ck_size != head.get("checkpoint_bytes"):
                return None, (f"checkpoint file is {ck_size} B, header says "
                              f"{head.get('checkpoint_bytes')} B")
            size += ck_size
        try:
            entry = _Entry(name, int(head["n_ids"]), int(head["alloc"]),
                           list(head["chain"]), [int(t) for t in head["tail"]],
                           size, float(head.get("used", 0.0)), path=path,
                           ckpts=[int(c["n"]) for c in ckpts],
                           free=bool(head.get("free", False)))
        except (KeyError, TypeError, ValueError) as e:
            return None, f"malformed header ({type(e).__name__}: {e})"
        if entry.n_ids <= 0 or entry.alloc <= 0:
            return None, "header describes an empty slot"
        if head.get("block") != self.block:
            return None, f"block size {head.get('block')} != {self.block}"
        return entry, ""

    def _identity_mismatch(self, head: dict) -> str:
        """What this file serves vs what we serve. A file that does not name
        the exact weights, revision, arm, dtype, attention configuration and
        cache library in play has nothing to say about them — skip it, never
        load it. A field ABSENT from the header is a refusal in its own
        right: a header that lacks a field does not get to match it by
        default."""
        if head.get("format_version") != FORMAT_VERSION:
            return (f"format_version {head.get('format_version')} "
                    f"!= {FORMAT_VERSION}")
        for field, mine in self.header.items():
            if field not in head:
                return f"header lacks {field}"
            theirs = head[field]
            if theirs != mine:
                return f"{field} {theirs!r} != {mine!r}"
        return ""

    # -- lookup ------------------------------------------------------------

    def lookup(self, ids: list[int], need: int, ctx: int, partial: bool = False,
               fit=None, floor: int = 0) -> _Entry | None:
        """The deepest stored block boundary this prompt shares, as an entry
        that EXTENDS-ONLY into it — or None.

        The walk is vLLM's: compute this prompt's chain, then try the deepest
        hash first. A stored slot whose last block hash equals the prompt's
        hash at that depth agrees with the prompt on every token up to that
        boundary (that is what chaining buys), so all that is left is the
        tail. Entries with no full block at all sit in the depth-0 bucket and
        are checked last, which is also longest-match-first: a deeper boundary
        is strictly more tokens.

        `partial` (context checkpoints on) also takes a stored slot the
        prompt parts from, as llama.cpp's host prompt cache does: the entry
        whose rewind or checkpoints serve the most of the prompt, and more
        than `floor` of it (_Entry.reach; `fit` keeps a bidirectional image
        run whole)."""
        if not self.enabled:
            return None
        chain = block_chain(ids, self.block)
        if partial:
            with self._lock:
                best, best_n = None, floor
                for e in self._entries.values():
                    if e.alloc < need or e.alloc > ctx:
                        continue
                    n = e.reach(ids, chain, self.block, fit)
                    if n > best_n:
                        best, best_n = e, n
                return best
        with self._lock:
            buckets: dict[str, list[_Entry]] = {}
            for e in self._entries.values():
                buckets.setdefault(e.chain[-1] if e.chain else "", []).append(e)
            for depth in range(len(chain), -1, -1):
                key = chain[depth - 1] if depth else ""
                best = None
                for e in buckets.get(key, ()):
                    if len(e.chain) != depth or e.alloc < need or e.alloc > ctx:
                        continue
                    if not e.extends_into(ids, self.block):
                        continue
                    if best is None or e.n_ids > best.n_ids:
                        best = e
                if best is not None:
                    return best
        return None

    def restore(self, entry: _Entry, make_cache, prime) -> tuple[object, list[int]] | None:
        """A live cache holding this stored slot, and the ids it holds — or
        None, having said why.

        `make_cache(alloc)` builds an empty cache at the saved width;
        `prime(cache)` drives ONE token through it so transformers-5's lazy
        layers materialize. Priming costs a single-token forward and buys the
        only honest description of what this process allocates, which is what
        the saved state is then checked against. The token it writes is
        overwritten by every tensor the file carries, `cumulative_length`
        (and, for a sliding-window layer, `cumulative_length_int`)
        included."""
        tensors = flags = ints = None
        try:
            flags = self._flags(entry)
            ints = self._ints(entry)
            tensors, ids = self._read_tensors(entry)
        except Exception as e:  # noqa: BLE001 — a bad file must cost a prefill
            _log(f"restore {entry.key} failed to read "
                 f"({type(e).__name__}: {e}) — prefilling cold", err=True)
            self.forget_entry(entry)
            return None
        if ids is None or len(ids) != entry.n_ids:
            _log(f"restore {entry.key}: ids are {0 if ids is None else len(ids)} "
                 f"long, header says {entry.n_ids} — prefilling cold", err=True)
            self.forget_entry(entry)
            return None
        cache = make_cache(entry.alloc)
        prime(cache)
        why = state_mismatch(cache, tensors, flags, ints)
        if why:
            _log(f"restore {entry.key}: {why} — prefilling cold", err=True)
            self.forget_entry(entry)
            return None
        apply_state(cache, tensors, flags, ints)
        held = written_tokens(cache)
        if held is not None and held != entry.n_ids:
            _log(f"restore {entry.key}: cache counter says {held} tokens, "
                 f"header says {entry.n_ids} — prefilling cold", err=True)
            self.forget_entry(entry)
            return None
        self.hits += 1
        self.touch(entry)
        return cache, ids

    def _flags(self, entry: _Entry) -> dict:
        ram = entry.ram  # read once: the writer thread may let go of it
        if ram is not None:
            return ram[1]
        with open(os.path.join(entry.path, "header.json")) as f:
            return {k: bool(v) for k, v in json.load(f)["flags"].items()}

    def _ints(self, entry: _Entry) -> dict:
        """Sliding-window layers' `cumulative_length_int`, keyed the same way
        as `_flags` (warm tier first, else the header). Absent entirely on a
        file without the field — `{}` there, which
        `state_mismatch` turns into a refusal for any cache that DOES have a
        sliding layer, and into no-op agreement for one that does not."""
        ram = entry.ram
        if ram is not None:
            return ram[2]
        with open(os.path.join(entry.path, "header.json")) as f:
            return {k: int(v) for k, v in json.load(f).get("ints", {}).items()}

    def _read_tensors(self, entry: _Entry):
        """(tensors, ids). From the warm tier when it still holds the copy,
        else from disk as mmap views — nothing is resident until apply_state
        clones it, one tensor at a time."""
        ram = entry.ram  # read once, as above
        if ram is not None:
            raw = dict(ram[0])
        else:
            from safetensors import safe_open

            with safe_open(os.path.join(entry.path, "cache.safetensors"),
                           framework="pt") as f:
                raw = {k: f.get_tensor(k) for k in f.keys()}
        ids_t = raw.pop("slot_ids", None)
        ids = None if ids_t is None else [int(t) for t in ids_t.reshape(-1)]
        return raw, ids

    def restore_checkpoints(self, entry: _Entry, cache, device) -> list:
        """The context checkpoints stored with `entry`, rebuilt on `device`
        (the cache's) and each checked against it (ctx_checkpoints.mismatch) — call
        after restore() has written the slot's own state into `cache`. One
        that does not fit is dropped with a line; a file that cannot be read
        costs the checkpoints, never the slot."""
        from . import ctx_checkpoints

        if not entry.ckpts:
            return []
        try:
            ram = entry.ram
            if ram is not None and ram[3] is not None:
                raw, metas = dict(ram[3][0]), ram[3][1]
            else:
                from safetensors import safe_open

                with open(os.path.join(entry.path, "header.json")) as f:
                    metas = json.load(f)["checkpoints"]
                with safe_open(os.path.join(entry.path, CKPT_FILE), framework="pt") as f:
                    raw = {k: f.get_tensor(k) for k in f.keys()}
            out = []
            for meta in metas:
                ck = ctx_checkpoints.unflatten(raw, meta, device)
                why = ctx_checkpoints.mismatch(cache, ck)
                if why:
                    _log(f"restore {entry.key}: checkpoint at {ck.n} dropped ({why})", err=True)
                    continue
                out.append(ck)
            return out
        except Exception as e:  # noqa: BLE001 — the checkpoints are the cost, not the slot
            _log(f"restore {entry.key}: checkpoints unreadable ({type(e).__name__}: {e}) "
                 "— the slot is restored without them", err=True)
            return []

    # -- eviction ----------------------------------------------------------

    def put(self, ids: list[int], alloc: int, cache, ckpts=None) -> _Entry | None:
        """Take a copy of a live slot on THIS thread and queue it for the
        writer. The copy is synchronous on purpose: the GPU tensors are about
        to be reused by whatever evicted this slot, so a background thread
        reading them later would serialize a cache that has moved on. Peak
        extra host memory is one slot — which is also, exactly, the warm
        tier's entry."""
        if not self.enabled or cache is None or len(ids) < self.block:
            # A prefix shorter than one block has no chain to be found by and
            # saves a prefill nobody notices; storing it would pay a
            # device->host copy of the WHOLE allocation for it.
            return None
        tensors, flags, ints = cache_state(cache)
        if not tensors:
            return None
        host = {k: v.detach().to("cpu", copy=True) for k, v in tensors.items()}
        host["slot_ids"] = torch.tensor(ids, dtype=torch.int64)
        nbytes = sum(t.numel() * t.element_size() for t in host.values())
        # the slot's context checkpoints go with it (llama.cpp's host prompt
        # cache keeps a prompt's checkpoints too), copied on this thread for
        # the same reason as the slot
        ck, positions, free = None, [], False
        if ckpts is not None and ckpts.on:
            from . import ctx_checkpoints

            free = bool(ckpts.free)
            if len(ckpts):
                ck_host, metas = {}, []
                for j, c in enumerate(ckpts.items):
                    t, meta = ctx_checkpoints.flatten(c, f"ck{j}.")
                    ck_host.update({k: v.detach().to("cpu", copy=True) for k, v in t.items()})
                    metas.append(meta)
                    positions.append(c.n)
                ck = (ck_host, metas)
                nbytes += sum(t.numel() * t.element_size() for t in ck_host.values())
        chain = block_chain(ids, self.block)
        entry = _Entry(prefix_key(ids), len(ids), alloc, chain,
                       list(ids[len(chain) * self.block:]), nbytes, time.time(),
                       path=os.path.join(self.dir, prefix_key(ids)),
                       ram=(host, flags, ints, ck), state=PENDING,
                       ckpts=positions, free=free)
        with self._lock:
            self._entries[entry.key] = entry
        # outside the index lock on purpose: the writer thread takes that lock
        # from _sweep, and a queue hand-off is not worth teaching two locks an
        # order they have to keep forever.
        self._pending_write(entry)
        return entry

    def _pending_write(self, entry: _Entry) -> None:
        if self._worker is None:
            self._worker = threading.Thread(target=self._drain, daemon=True,
                                            name="drinkme-slots")
            self._worker.start()
            atexit.register(self.flush)
        with self._idle:
            self._pending += 1
        self._q.put(entry)

    def _drain(self) -> None:
        while True:
            entry = self._q.get()
            try:
                if entry is None:
                    return
                self._write(entry)
            except Exception as e:  # noqa: BLE001 — a failed write is not a
                # failed generation; the prefix is still correct in the
                # live slot. But it IS a failed write: said so, counted,
                # and the entry leaves the index rather than standing in it
                # as a slot that was never there.
                _log(f"write {entry.key} failed ({type(e).__name__}: {e}) "
                     "— this slot will prefill cold", err=True)
                self.forget_entry(entry)
                with self._idle:
                    self.failed += 1
                    self._unflushed_failures += 1
            finally:
                with self._idle:
                    self._pending -= 1
                    self._idle.notify_all()

    def _write(self, entry: _Entry) -> None:
        """Tensors first, header second, both through a temp name. A crash
        between them leaves a directory the boot index skips silently, which
        is the only failure mode a derived artifact is allowed."""
        from safetensors.torch import save_file

        if not entry.pending:
            # forgotten (restore refused it, or a caller dropped it) while
            # it sat in the queue: nothing to write and nothing lost
            return
        ram = entry.ram
        if ram is None:
            # A pending entry's payload is the only copy there is, and no
            # sweep may take it (the state machine above). If it is gone
            # anyway, that is this file's bug, and the drain loop's handler
            # says so — not a silent return that flush() then calls success.
            raise RuntimeError("pending entry has no payload — the snapshot "
                               "was dropped before it reached disk")
        tensors, flags, ints, ck = ram
        _mkdir_private(entry.path)

        def save(name: str, what: dict) -> int:
            tpath = os.path.join(entry.path, name)
            tmp = tpath + ".tmp"
            # the tensor file is created here at FILE_MODE and only then handed
            # to safetensors, which truncates into it and keeps the mode — not
            # left to whatever mode that writer picks in whatever version
            _open_private(tmp).close()
            save_file({k: v.contiguous() for k, v in what.items()}, tmp)
            os.chmod(tmp, FILE_MODE)
            os.replace(tmp, tpath)
            return os.path.getsize(tpath)

        entry.nbytes = save("cache.safetensors", tensors)
        head = dict(self.header)
        head.update(format_version=FORMAT_VERSION, n_ids=entry.n_ids,
                    alloc=entry.alloc, block=self.block, chain=entry.chain,
                    tail=entry.tail, bytes=entry.nbytes, used=entry.used,
                    created=entry.used, flags=flags)
        if ints:
            # omitted entirely when empty (every model with no sliding layer,
            # e.g. Qwen) so the header of a slot without them carries no
            # trace of the field.
            head["ints"] = ints
        if entry.free:
            head["free"] = True
        if ck is not None:
            # like `ints`, absent from a slot that has none; a reader from
            # before them skips the fields and the file
            ck_bytes = save(CKPT_FILE, ck[0])
            head.update(checkpoints=ck[1], checkpoint_bytes=ck_bytes)
            entry.nbytes += ck_bytes
        hpath = os.path.join(entry.path, "header.json")
        htmp = hpath + ".tmp"
        with _open_private(htmp) as f:
            json.dump(head, f)
        os.replace(htmp, hpath)
        # the header is the last thing to land and the first thing a boot
        # index reads: the entry is on disk from here, and only from here
        entry.state = COMMITTED
        self.saved += 1
        _log(f"saved slot {entry.key} ({entry.n_ids} tokens, "
             f"{entry.nbytes / _GIB:.2f} GiB)")
        self._sweep(entry)

    # -- the caps ----------------------------------------------------------

    def _sweep(self, keep: _Entry) -> None:
        """LRU down to the disk cap, then down to the RAM cap. `keep` is the
        entry just written: it is the most recently used thing in the store,
        so LRU never picks it unless it alone is over the cap — in which case
        it goes too, loudly, because a slot bigger than the whole budget is a
        misconfiguration and not a cache.

        Only EVICTABLE entries are candidates for either cap. A pending
        entry is not on disk, so it is not the disk cap's
        to count or remove; and its `ram` is the snapshot itself, not a
        warm-tier copy, so it is not the RAM cap's to let go. Each pending
        write runs its own sweep when it lands, which is when its bytes
        start to count."""
        with self._lock:
            entries = sorted((e for e in self._entries.values() if e.evictable),
                             key=lambda e: e.used)
            total = sum(e.nbytes for e in entries)
            for e in entries:
                if total <= self.cap:
                    break
                total -= e.nbytes
                self._evict_file(e, "disk cap")
            if keep.key in self._entries and self._entries[keep.key].nbytes > self.cap:
                self._evict_file(keep, "larger than the whole disk cap")
            ram = [e for e in sorted(self._entries.values(), key=lambda e: e.used)
                   if e.evictable and e.ram is not None]
            held = sum(e.nbytes for e in ram)
            for e in ram:
                if held <= self.ram_cap:
                    break
                held -= e.nbytes
                e.ram = None  # on disk already; the warm tier just lets go

    def _evict_file(self, entry: _Entry, why: str) -> None:
        self._entries.pop(entry.key, None)
        entry.state = EVICTED
        entry.ram = None
        if entry.path and os.path.isdir(entry.path):
            shutil.rmtree(entry.path, ignore_errors=True)
        _log(f"evicted slot {entry.key} ({entry.nbytes / _GIB:.2f} GiB): {why}")

    def forget_entry(self, entry: _Entry) -> None:
        """Drop an entry from the index without deleting anything a
        concurrent reader might hold. Used when a file turns out to be
        unusable: the prefill it would have saved is the whole cost. A
        pending entry forgotten here never reaches disk (its write becomes
        a no-op) and reads as FAILED — it did not persist."""
        with self._lock:
            self._entries.pop(entry.key, None)
            if entry.pending:
                entry.state = FAILED
        entry.ram = None

    def touch(self, entry: _Entry) -> None:
        """Mark an entry most-recently-used, on disk too, so LRU survives a
        bounce. Best effort: an unwritable header costs an eviction order,
        not a served request."""
        entry.used = time.time()
        if not entry.path:
            return
        hpath = os.path.join(entry.path, "header.json")
        try:
            with open(hpath) as f:
                head = json.load(f)
            head["used"] = entry.used
            tmp = hpath + ".tmp"
            with _open_private(tmp) as f:
                json.dump(head, f)
            os.replace(tmp, hpath)
        except (OSError, ValueError):
            pass

    # -- lifecycle ---------------------------------------------------------

    def flush(self, timeout: float = 300.0) -> bool:
        """Wait for every queued write. True only if the queue drained AND
        every write queued since the previous flush committed — a drained
        queue with a failure in it is not "the writes completed", it is one
        write fewer than was promised. Called by serve.py's
        SIGTERM handler and by an atexit belt: "async is fine but the write
        must complete before the process exits". The failure count is
        reported once, to the flush that covers it, so a long-lived server's
        next SIGTERM answers for its own writes."""
        with self._idle:
            drained = self._idle.wait_for(lambda: self._pending == 0, timeout)
            failed, self._unflushed_failures = self._unflushed_failures, 0
            return drained and failed == 0

    def disable(self) -> None:
        """Stop answering and stop storing, for the rest of the process.

        engines.reset_prefix_cache() says this: a harness that asked to start
        cold in its own terms must not be handed yesterday's prefix off an
        SSD, and a bench arm's evictions must not land in a real server's
        store. Files already written stay — this is about what THIS process
        will do."""
        with self._lock:
            self.enabled = False
            self._entries = {}


def open_store(model_id: str, arm: str, meta: dict, ctx: int,
               dtype: str, block: int = BLOCK_TOKENS,
               attention: dict | None = None) -> SlotStore | None:
    """The store for one engine, indexed and announced — or None when the
    cold tier is off, its directory cannot be made, or the engine cannot
    name what it serves: `meta["resolvedRevision"]` (the loaders set it —
    checkpoint.resolved_revision for stock, the pack's recorded source
    revision or the snapshot's for compressed) and `attention`
    (attention_identity(model.config)) are both REQUIRED.
    An engine built without them — a bare HFEngine(...) — serves with no
    cold tier rather than with one keyed on less than the weights."""
    root = slot_root()
    if root is None:
        return None
    revision = meta.get("resolvedRevision")
    if not isinstance(revision, str) or not revision:
        _log("no resolved weights revision in the engine's meta — a cache "
             "keyed on less than the weights is not a cache; serving without "
             "a cold tier", err=True)
        return None
    if not isinstance(attention, dict):
        _log("no attention configuration for the store's identity — serving "
             "without a cold tier", err=True)
        return None
    pack_id = meta.get("packId")
    attention = json.loads(json.dumps(attention, sort_keys=True, default=str))
    header = {"model_id": model_id, "arm": arm, "pack_id": pack_id,
              "dtype": dtype, "transformers": _transformers_version(),
              "revision": revision, "attention": attention}
    store = SlotStore(root, store_ident(model_id, arm, pack_id, revision, attention),
                      header, block=block)
    try:
        _mkdir_private(store.dir)
    except OSError as e:
        _log(f"cannot create {store.dir} ({e}) — serving without a cold tier",
             err=True)
        return None
    store.index()
    return store


def _transformers_version() -> str:
    """The cache layout a slot file was written against. Not decoration: the
    tensors ARE transformers' internal cache state, and the honest posture
    for a library that renames a surface is to skip the file, not to guess."""
    try:
        import transformers

        return str(transformers.__version__)
    except Exception:  # noqa: BLE001
        return "unknown"
