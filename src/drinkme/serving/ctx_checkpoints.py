"""Context checkpoints: a prefix slot is reused when a prompt SHARES a prefix
with it, not only when the prompt extends all of it. This mirrors llama.cpp's
`--ctx-checkpoints` (tools/server/server-context.cpp at 1e6f9e4a5:
create_checkpoint, the restore search in the SLOT_STATE_STARTED branch, and
the checkpoint offsets in the prompt-batch loop; common/common.h's defaults).

THE PROBLEM. A slot holds the ids of the last prompt and of the tokens the
model generated after it. A chat template re-renders that history differently
from what the model wrote: gemma-4 drops the empty `<|channel>thought\\n
<channel|>` its thinking-off generation prompt ends in and that the model
writes after a tool response, Qwen3.5 trims a reply's trailing newline, and
Muse-Glimmer's `<|start|>assistant` is followed by the model's ` to=self`
channel but re-rendered as ` to=user`. So the next prompt shares a prefix with
the slot and then parts from it. Before this module the slot was reused only
when the prompt extended everything it held, because three kinds of state
cannot be rewound by length: a sliding-window layer's ring
(transformers' StaticSlidingWindowLayer shifts its oldest rows out once the
window is full), and a gated-DeltaNet layer's recurrent and conv states,
which have no sequence dimension at all. Full-attention KV can be rewound
by length (kvcache.LiveStaticLayer.rewind, the same call mtp.py uses).

WHAT A CHECKPOINT IS. A snapshot, at one position n, of only the state that
cannot be rewound to n: every sliding layer's ring (its first min(n, window)
rows of keys and values, and both of its counters) and every linear-attention
layer's conv and recurrent states. Full-attention layers are not copied; a
restore rewinds them to n by length. A snapshot is a device copy beside the
slot's own cache.

llama.cpp's rules, and drinkme's version of each:

| llama.cpp (1e6f9e4a5) | drinkme |
|---|---|
| `--ctx-checkpoints N`, default 32 per slot; 0 disables | `--ctx-checkpoints N` / DRINKME_CTX_CHECKPOINTS, default 32; 0 is today's extends-only rule, byte for byte |
| taken only for models whose memory cannot be truncated (recurrent, hybrid, SWA) | the same: a cache with no sliding or linear layer takes none, and a rewind by length reaches any shared prefix |
| taken BEFORE a batch is decoded: the state at the batch's first token | taken after the span ending at n is written, so the prefill is split at every checkpoint position; or, when the prefill leaves every ring unshifted and the cache has no linear layer, copied out of the rings after an unsplit prefill (`ring_holds`) |
| two near the end of the prompt: at n_prompt - 4 - n_ubatch and n_prompt - 4 (#20288) | the same two, with n_ubatch = 512 (llama.cpp's default) |
| none at n_prompt: the last batch must compute the logits row | also one at n_prompt, where a history that keeps the whole generation prompt parts from the slot (gemma's tool loop, Qwen's trailing newline); it costs no extra forward |
| any batch start within n_ubatch of the prompt's end | not taken: those positions follow llama.cpp's batch geometry |
| one at the start of the last user message (#24176, found by per-template delimiters) | the same, found by rendering the history before that message: its ids must be a prefix of the prompt's, or none is taken |
| one at each earlier user-message start more than `checkpoint_min_step` apart | not taken (see below) |
| none right after an image chunk, and none inside one (an image chunk is atomic) | none inside an image run that a prefill span may not cut (image_prompt.ImagePrompt.whole): the position moves back to the run's start; right after a run is allowed |
| create: erase other requests' checkpoints within `checkpoint_min_step` (8192) of the previous kept one (#25472), then the oldest while N are held | the same, by request |
| restore: needed only when the memory cannot reach the common prefix; the newest checkpoint at or below it, leaving at least one prompt token to compute | the same; a sliding cache whose rings have not wrapped, and a cache with only full attention, rewinds by length instead |
| erase checkpoints past the restore point | the same |
| a slot is picked by prompt similarity (common prefix / prompt) above `--slot-prompt-similarity` 0.1, else the least recently used | picked by what it serves (the reuse a rewind or checkpoint reaches), above 0.1 of the prompt unless the prompt extends it; on a tie an extension, then the least recently used (engines.pick_slot) |
| — | a slot that would keep less than half its tokens is not taken while an untouched slot is free: N slots keep N conversations warm |
| a slot about to keep less than half its tokens (f_keep < 0.5) is saved to the prompt cache first, which is then asked for a better match | the same with the cold tier: the slot is put there, then a stored slot that serves more replaces it |
| host-memory prompt cache (`--cache-ram`) keeps a prompt's checkpoints with it and finds a cached prompt by common prefix | the cold tier (slotstore.py) keeps a slot's checkpoints with it, and a stored slot is found by the prefix its checkpoints reach; so a sleep, which parks slots there, keeps them too |
| `/slots` save and restore drop checkpoints | drinkme has no such route |
| speculative state (a draft model's, EAGLE's) stashed with each checkpoint | nothing to stash: the MTP head's KV and the n-gram table are rebuilt per request from the prompt (mtp.Speculator.open) |

Earlier user-message starts are not taken: drinkme renders a prompt through
the family's own template and has no table of role delimiters in tokens
(llama.cpp's common/chat.cpp `message_delimiters`), and finding each one
would render the history once per message. They are MIN_STEP (8192) tokens
apart, so at the default ctx of 8192 llama.cpp takes none of them either.
The last user message's start is where a conversation that shares only a
system prompt, or a regenerated last message, parts from the slot; the
checkpoints near the end serve every re-render the reports measured.

THE BOUND. Other requests' checkpoints are more than MIN_STEP apart and lie
in [1, ctx - 1]; the current request adds at most four. So a slot holds at
most min(N, floor((ctx - 2) / (MIN_STEP + 1)) + 5) (`max_held`), five at the
default ctx of 8192. The fit charge is that count times `config_bytes`.

WHEN THE PREFILL IS SPLIT. A split costs one more forward over the whole
model, for a few tokens: on gemma-4-31B on gfx1151 about 0.6 s each, which
took a 41-token cold turn's time to first token from 0.99 s to 2.26 s. It is
needed only for state the prefill overwrites. A recurrent state changes
with every token, and a sliding ring shifts its oldest row out once it
holds a window. But while a ring has not shifted, its rows 0..p-1 with its
counters at p are the ring at p (the prefill writes row i for position i
and never touches it again), and a full layer rewinds by length. So when a
prefill to n_prompt leaves every ring unshifted (n_prompt at most every
sliding layer's window) and the cache has no linear layer (`ring_holds`),
the prefill runs in the spans of `--ctx-checkpoints 0`, and each checkpoint
is copied out of the rings after it (`snapshot`'s `at`). The slot gets the
same positions with the same shapes, holding the rows a rewind of this
cache to p would keep, so it keeps every checkpoint a split prefill gave
it: a ring that wraps later, in the reply or in a later turn, still has
them. A prompt longer than the window, and every model with a linear layer
(Qwen3.5), split as before.

NUMERICS. A split prefill runs in more spans than the unsplit one:
logits agree to reduction order, the chunked-prefill class engines.py
assigns to warm-vs-cold already. Restored state is the snapshot byte for
byte; the full-attention rows below the restore point are the ones the
earlier request wrote.
"""

from __future__ import annotations

import os
import sys

ENV = "DRINKME_CTX_CHECKPOINTS"
# llama.cpp common/common.h: n_ctx_checkpoints = 32, checkpoint_min_step = 8192
DEFAULT_MAX = 32
MIN_STEP = 8192
# server-context.cpp's checkpoint_offsets {4 + n_ubatch, 4}, with llama.cpp's
# default n_ubatch of 512
TAIL = 4
UBATCH = 512
# get_available_slot: a slot the prompt does not extend is taken only when
# what it serves is more than SIMILARITY of the prompt (common.h
# slot_prompt_similarity = 0.1), and a slot about to keep less than KEEP of
# its tokens is saved to the prompt cache first, which is then asked for a
# better match (f_keep < 0.5 -> update_cache)
SIMILARITY = 0.1
KEEP = 0.5

FULL, SLIDING, LINEAR = "full", "sliding", "linear"


def max_from_env(explicit: int | None = None) -> int:
    """`explicit` (--ctx-checkpoints) wins over DRINKME_CTX_CHECKPOINTS, which
    wins over DEFAULT_MAX: engines.slots_from_env's precedence. 0 turns
    checkpoints off. Garbage or a negative count warns and falls back to the
    default, the same posture as the slot count."""
    if explicit is not None:
        if explicit < 0:
            print(f"[drinkme] --ctx-checkpoints {explicit} is not a count of 0 or "
                  f"more — using {DEFAULT_MAX}", file=sys.stderr, flush=True)
            return DEFAULT_MAX
        return explicit
    raw = os.environ.get(ENV, "").strip()
    if not raw:
        return DEFAULT_MAX
    try:
        n = int(raw)
    except ValueError:
        n = -1
    if n < 0:
        print(f"[drinkme] {ENV}={raw!r} is not a count of 0 or more — using "
              f"{DEFAULT_MAX}", file=sys.stderr, flush=True)
        return DEFAULT_MAX
    return n


def max_held(n_max: int, ctx: int, min_step: int = MIN_STEP) -> int:
    """The most checkpoints one slot can hold under these rules (module
    docstring, THE BOUND)."""
    if n_max <= 0:
        return 0
    return min(n_max, max(0, (ctx - 2) // (min_step + 1)) + 5)


# ------------------------------------------------------- the cache layers --


def layer_kind(layer) -> str | None:
    """How a slot cache layer reaches an earlier position: FULL (rewind by
    length: kvcache.LiveStaticLayer), SLIDING (transformers'
    StaticSlidingWindowLayer surface: two counters beside a ring), LINEAR (a
    linear-attention layer's conv and recurrent states), or None for a layer
    this module does not know, which turns checkpoints off for the cache.
    Recognized by surface, as slotstore and mtp.py recognize a sliding
    layer."""
    from .kvcache import LiveStaticLayer

    if hasattr(layer, "conv_states") and hasattr(layer, "recurrent_states"):
        # transformers' hybrid classes also carry an attention half
        return None if hasattr(layer, "cumulative_length") else LINEAR
    if hasattr(layer, "cumulative_length_int") and hasattr(layer, "cumulative_length"):
        return SLIDING
    if isinstance(layer, LiveStaticLayer):
        return FULL
    return None


def kinds(cache) -> list[str | None]:
    return [layer_kind(layer) for layer in cache.layers]


def supported(cache) -> bool:
    """Can this cache be checkpointed and rewound at all?"""
    return all(k is not None for k in kinds(cache))


def needs_checkpoints(cache) -> bool:
    """Does this cache hold any state a rewind by length cannot reach? False
    for a cache of full-attention layers only: it rewinds to any position,
    and llama.cpp takes no checkpoints for such a model either."""
    ks = kinds(cache)
    return all(k is not None for k in ks) and any(k in (SLIDING, LINEAR) for k in ks)


def length(cache) -> int:
    """The tokens the full and sliding layers hold (python ints, no device
    read). -1 when they disagree."""
    seen = set()
    for layer, kind in zip(cache.layers, kinds(cache)):
        if kind == FULL:
            seen.add(layer.live)
        elif kind == SLIDING:
            seen.add(int(layer.cumulative_length_int))
    if not seen:
        return 0
    return seen.pop() if len(seen) == 1 else -1


def rewinds_freely(cache) -> bool:
    """Can this cache reach ANY earlier position by length alone? True when
    it has no linear layer and no sliding ring has shifted a row out yet
    (cumulative_length_int <= window): llama.cpp's `pos_min < pos_min_thold`
    case, where the memory still holds every position a rewind needs."""
    for layer, kind in zip(cache.layers, kinds(cache)):
        if kind is None or kind == LINEAR and _linear_live(layer):
            return False
        if kind == SLIDING and layer.cumulative_length_int > layer.max_cache_len:
            return False
    return True


def ring_holds(cache, n: int) -> bool:
    """Will this cache, prefilled to position n, still hold the state at
    every earlier position? True when it has no linear layer (a recurrent
    state has no rows to go back to) and no sliding ring shifts a row out on
    the way to n (n at most its window): then each checkpoint position up to
    n is copied out of the rings after the prefill (`snapshot`'s `at`), and
    the prefill is not split for it (module docstring, WHEN THE PREFILL IS
    SPLIT)."""
    for layer, kind in zip(cache.layers, kinds(cache)):
        if kind is None or kind == LINEAR:
            return False
        if kind == SLIDING and n > layer.max_cache_len:
            return False
    return True


def _linear_live(layer) -> bool:
    return any(layer.is_conv_states_initialized.values()) or any(
        layer.is_recurrent_states_initialized.values())


def rewind(cache, n: int) -> None:
    """Rewind a cache that `rewinds_freely` to position n: the full layers by
    length, the sliding layers' two counters (rows below n are the ring's
    rows 0..n-1, since it never shifted; the rows past n are masked until
    they are written again, as after mtp.py's rewind)."""
    for layer, kind in zip(cache.layers, kinds(cache)):
        if kind == FULL:
            layer.rewind(layer.live - n)
        elif kind == SLIDING:
            if not n <= layer.cumulative_length_int <= layer.max_cache_len:
                raise ValueError(f"sliding layer at {layer.cumulative_length_int} "
                                 f"(window {layer.max_cache_len}) cannot rewind to {n}")
            layer.cumulative_length_int = n
            layer.cumulative_length.fill_(n)
        elif kind == LINEAR and _linear_live(layer):
            raise ValueError("a linear-attention layer cannot rewind by length")


class Checkpoint:
    """The non-rewindable state of one cache at position `n`, taken during
    request `task`. `layers` maps a layer index to what that layer needs
    back: a sliding layer's (keys, values, cumulative_length, int), a linear
    layer's {state index: (conv, recurrent, has_previous_state)}."""

    __slots__ = ("n", "task", "layers", "nbytes")

    def __init__(self, n: int, task: int, layers: dict, nbytes: int):
        self.n, self.task, self.layers, self.nbytes = n, task, layers, nbytes

    def __repr__(self) -> str:
        return f"Checkpoint(n={self.n}, task={self.task}, {self.nbytes} B)"


def _nbytes(t) -> int:
    return t.numel() * t.element_size()


def snapshot(cache, task: int, at: int | None = None) -> Checkpoint:
    """The checkpoint of `cache` where it stands, or at an earlier position
    `at` its rings still hold (`ring_holds`): its non-rewindable state,
    copied on the device (no host read). At `at`, a ring is its first `at`
    rows and both counters at `at`, the shapes a snapshot taken there has."""
    n = length(cache)
    if n < 0:
        raise ValueError("cache layers disagree about their length")
    if at is not None and not (0 < at <= n and ring_holds(cache, n)):
        raise ValueError(f"a cache at {n} does not hold the state at {at}")
    layers: dict = {}
    total = 0
    for i, (layer, kind) in enumerate(zip(cache.layers, kinds(cache))):
        if kind == SLIDING:
            if not layer.is_initialized:
                continue
            if at is None:
                n_int = int(layer.cumulative_length_int)
                c = layer.cumulative_length.clone()
            else:
                n_int = at
                c = layer.cumulative_length.clone().fill_(at)
            rows = min(n_int, layer.max_cache_len)
            k = layer.keys[:, :, :rows].clone()
            v = layer.values[:, :, :rows].clone()
            layers[i] = (k, v, c, n_int)
            total += _nbytes(k) + _nbytes(v) + _nbytes(c)
        elif kind == LINEAR:
            states = {}
            for s in layer.conv_states:
                conv = (layer.conv_states[s].clone()
                        if layer.is_conv_states_initialized[s] else None)
                rec = (layer.recurrent_states[s].clone()
                       if layer.is_recurrent_states_initialized[s] else None)
                if conv is None and rec is None:
                    continue
                states[s] = (conv, rec, bool(layer.has_previous_state[s]))
                total += sum(_nbytes(t) for t in (conv, rec) if t is not None)
            if states:
                layers[i] = states
        elif kind is None:
            raise ValueError(f"cache layer {i} ({type(layer).__name__}) cannot be checkpointed")
    return Checkpoint(n if at is None else at, task, layers, total)


def restore(cache, ck: Checkpoint) -> None:
    """Put `cache` back at checkpoint `ck`: its snapshot copied into the live
    tensors in place (their static addresses survive, as mtp.py's rewind
    requires), the full layers rewound to ck.n by length."""
    for i, (layer, kind) in enumerate(zip(cache.layers, kinds(cache))):
        if kind == FULL:
            if layer.live < ck.n:
                raise ValueError(f"full layer {i} holds {layer.live} rows, "
                                 f"checkpoint is at {ck.n}")
            layer.rewind(layer.live - ck.n)
        elif kind == SLIDING and i in ck.layers:
            k, v, c, n_int = ck.layers[i]
            rows = k.shape[2]
            layer.keys[:, :, :rows].copy_(k)
            layer.values[:, :, :rows].copy_(v)
            layer.cumulative_length.copy_(c)
            layer.cumulative_length_int = n_int
        elif kind == LINEAR and i in ck.layers:
            for s, (conv, rec, prev) in ck.layers[i].items():
                if conv is not None:
                    layer.conv_states[s].copy_(conv)
                if rec is not None:
                    layer.recurrent_states[s].copy_(rec)
                layer.has_previous_state[s] = prev
    got = length(cache)
    if got != ck.n:
        raise ValueError(f"restored cache holds {got} tokens, checkpoint is at {ck.n}")


# ------------------------------------------------- the cold tier's form --


def flatten(ck: Checkpoint, prefix: str) -> tuple[dict, dict]:
    """(tensors keyed under `prefix`, the rest as JSON) — what slotstore.py
    writes beside a slot's own tensors. Tensors are returned as they are;
    the caller copies them to host."""
    tensors: dict = {}
    layers: dict = {}
    for i, entry in ck.layers.items():
        if isinstance(entry, tuple):
            k, v, c, n_int = entry
            tensors[f"{prefix}{i}.k"], tensors[f"{prefix}{i}.v"] = k, v
            tensors[f"{prefix}{i}.c"] = c
            layers[str(i)] = {"kind": SLIDING, "int": int(n_int)}
        else:
            states = {}
            for s, (conv, rec, prev) in entry.items():
                if conv is not None:
                    tensors[f"{prefix}{i}.{s}.conv"] = conv
                if rec is not None:
                    tensors[f"{prefix}{i}.{s}.rec"] = rec
                states[str(s)] = {"prev": bool(prev), "conv": conv is not None,
                                  "rec": rec is not None}
            layers[str(i)] = {"kind": LINEAR, "states": states}
    return tensors, {"n": ck.n, "prefix": prefix, "layers": layers}


def unflatten(tensors: dict, meta: dict, device) -> Checkpoint:
    """flatten's inverse, each tensor cloned out of its backing (a
    safetensors mmap view, slotstore.apply_state's reason) and moved to
    `device`. The request that took it is gone, so its task is -1: no live
    request's thinning exempts it."""
    prefix = meta["prefix"]

    def get(key):
        return tensors[prefix + key].clone().to(device)

    layers: dict = {}
    total = 0
    for i, lay in meta["layers"].items():
        if lay["kind"] == SLIDING:
            k, v, c = get(f"{i}.k"), get(f"{i}.v"), get(f"{i}.c")
            layers[int(i)] = (k, v, c, int(lay["int"]))
            total += _nbytes(k) + _nbytes(v) + _nbytes(c)
        elif lay["kind"] == LINEAR:
            states = {}
            for s, st in lay["states"].items():
                conv = get(f"{i}.{s}.conv") if st["conv"] else None
                rec = get(f"{i}.{s}.rec") if st["rec"] else None
                states[int(s)] = (conv, rec, bool(st["prev"]))
                total += sum(_nbytes(t) for t in (conv, rec) if t is not None)
            layers[int(i)] = states
        else:
            raise ValueError(f"checkpoint layer {i}: unknown kind {lay['kind']!r}")
    return Checkpoint(int(meta["n"]), -1, layers, total)


def mismatch(cache, ck: Checkpoint) -> str | None:
    """None when checkpoint `ck` can be restored into this (primed) cache,
    else why not: every sliding and live linear layer of the cache must be
    in it, with this process's shapes and dtypes."""
    for i, (layer, kind) in enumerate(zip(cache.layers, kinds(cache))):
        entry = ck.layers.get(i)
        if kind == SLIDING:
            if not isinstance(entry, tuple):
                return f"layer {i}: no sliding-window state"
            k, v, c, n_int = entry
            live = layer.keys
            if (k.dtype != live.dtype or k.shape[:2] != live.shape[:2]
                    or k.shape[3:] != live.shape[3:] or v.shape[:2] != layer.values.shape[:2]
                    or v.shape[3:] != layer.values.shape[3:] or v.dtype != layer.values.dtype
                    or k.shape[2] != v.shape[2] or k.shape[2] > layer.max_cache_len
                    or k.shape[2] != min(n_int, layer.max_cache_len) or n_int != ck.n):
                return f"layer {i}: sliding state {tuple(k.shape)} does not fit {tuple(live.shape)}"
        elif kind == LINEAR and _linear_live(layer):
            if not isinstance(entry, dict):
                return f"layer {i}: no linear-attention state"
            for s, (conv, rec, _prev) in entry.items():
                for saved, live in ((conv, layer.conv_states.get(s)),
                                    (rec, layer.recurrent_states.get(s))):
                    if (saved is None) != (live is None):
                        return f"layer {i}: state {s} present on one side only"
                    if saved is not None and (saved.shape != live.shape
                                              or saved.dtype != live.dtype):
                        return (f"layer {i}: state {tuple(saved.shape)}/{saved.dtype} "
                                f"!= {tuple(live.shape)}/{live.dtype}")
        elif entry is not None and kind not in (SLIDING, LINEAR):
            return f"layer {i}: state for a {kind} layer"
    return None


# ------------------------------------------------------ one slot's list --


class Checkpoints:
    """One slot's checkpoints, oldest (lowest position) first, and the rules
    llama.cpp keeps them by (module docstring). `on` is False when the knob is
    0 or the slot's cache cannot be checkpointed; `free` is True when the
    slot's cache reaches any earlier position by length (rewinds_freely),
    decided when the slot was last written."""

    __slots__ = ("n_max", "min_step", "items", "free", "on")

    def __init__(self, n_max: int = 0, min_step: int = MIN_STEP):
        self.n_max, self.min_step = n_max, min_step
        self.items: list[Checkpoint] = []
        self.free = False
        self.on = n_max > 0

    def __len__(self) -> int:
        return len(self.items)

    @property
    def nbytes(self) -> int:
        return sum(c.nbytes for c in self.items)

    def positions(self) -> list[int]:
        return [c.n for c in self.items]

    def clear(self) -> None:
        self.items = []
        self.free = False

    def at(self, n: int) -> Checkpoint | None:
        for c in self.items:
            if c.n == n:
                return c
        return None

    def reach(self, lcp: int, n_ids: int, n_prompt: int, fit=None) -> int:
        """How many of a prompt's tokens this slot can serve, given their
        common prefix `lcp` with the slot's `n_ids` ids: the whole slot when
        the prompt extends it (the extends-only rule), else the furthest
        point a rewind or a checkpoint reaches at or below the prefix and
        before the prompt's last token, else 0. `fit(n)` is the largest
        reuse point at or below n that cuts no image run (image_prompt), or
        None for a text prompt."""
        if 0 < lcp == n_ids and lcp < n_prompt:
            return lcp
        if not self.on:
            return 0
        target = min(lcp, n_prompt - 1)
        if target <= 0:
            return 0
        if self.free:
            return max(0, fit(target) if fit is not None else target)
        for c in reversed(self.items):
            if c.n <= target and (fit is None or fit(c.n) == c.n):
                return c.n
        return 0

    def drop_beyond(self, n: int) -> None:
        """Erase the checkpoints past position n: the tokens after it are about
        to be overwritten."""
        self.items = [c for c in self.items if c.n <= n]

    def add(self, ck: Checkpoint) -> list[Checkpoint]:
        """llama.cpp's create_checkpoint: erase other requests' checkpoints
        within min_step of the previous kept one, then the oldest until there
        is room, then keep `ck`. Returns what was erased."""
        gone, kept, last = [], [], -1
        for c in self.items:
            if c.task != ck.task and last >= 0 and c.n <= last + self.min_step:
                gone.append(c)
                continue
            kept.append(c)
            last = c.n
        while kept and len(kept) >= self.n_max:
            gone.append(kept.pop(0))
        kept.append(ck)
        kept.sort(key=lambda c: c.n)
        self.items = kept
        return gone


def positions(n_prompt: int, start: int, whole=(), user: int | None = None) -> list[int]:
    """Where a prefill from `start` takes checkpoints: at `user`, the start of
    the last user message when it is known, llama.cpp's two near the end,
    and drinkme's own at n_prompt (module docstring). Each lies after
    `start`, since the state at start is the slot's already. One inside a run
    `whole` lists ([s, e) image runs a prefill span may not cut) moves back
    to the run's start."""
    out: list[int] = []
    near = (n_prompt - TAIL - UBATCH, n_prompt - TAIL, n_prompt)
    for p in near if user is None else (user, *near):
        for s, e in whole:
            if s < p < e:
                p = s
                break
        if p > start and p >= 1 and p not in out:
            out.append(p)
    return sorted(out)


# ---------------------------------------------------------- the fit charge --


# A gated-DeltaNet layer's recurrent state is float32 whatever the model's
# dtype: transformers' torch path computes it in float32 and never casts it
# back (modeling_qwen3_5.torch_chunk_gated_delta_rule), and the served 27B's
# boot line measures 154,927,232 B of slot state for its 48 linear layers,
# which is 48 x (48 x 128 x 128 x 4 B + a bf16 conv state) + 16 int64
# counters.
STATE_BYTES = 4
# a StaticLayer's `cumulative_length`: a 0-dim int64 tensor
COUNTER_BYTES = 8


def config_bytes(config: dict, dtype_bytes: int = 2, ctx: int | None = None,
                 state_bytes: int = STATE_BYTES) -> int:
    """Bytes of ONE full checkpoint for a model, from its config.json alone
    (torch-free, for the no-model picker): every sliding layer's keys and
    values at the whole window and its counter, every linear-attention
    layer's conv state (the model's dtype) and recurrent state (float32).
    0 for a model with neither, which takes no checkpoints. `ctx` caps the
    window as the slot's allocation does (StaticSlidingWindowLayer holds
    min(window, max_cache_len) rows). tests/test_serving_ctx_checkpoints.py
    holds this equal to a real snapshot's bytes on the gemma-4, Qwen3.5 and
    Muse-Glimmer toys."""
    tc = config.get("text_config", config)
    types = tc.get("layer_types") or []
    heads = int(tc.get("num_attention_heads") or 1)
    head_dim = int(tc.get("head_dim") or int(tc.get("hidden_size", 0)) // heads)
    kv_heads = int(tc.get("num_key_value_heads") or heads)
    window = tc.get("sliding_window")
    total = 0
    for t in types:
        if t in ("sliding_attention", "chunked_attention") and window:
            rows = int(window) if ctx is None else min(int(window), ctx)
            total += 2 * kv_heads * rows * head_dim * dtype_bytes + COUNTER_BYTES
        elif t == "linear_attention":
            dims = [tc.get(k) for k in ("linear_num_key_heads", "linear_num_value_heads",
                                        "linear_key_head_dim", "linear_value_head_dim",
                                        "linear_conv_kernel_dim")]
            if None in dims:
                continue  # a config without the gated-DeltaNet fields: nothing to size
            k_heads, v_heads, k_dim, v_dim, kernel = (int(d) for d in dims)
            conv_dim = 2 * k_heads * k_dim + v_heads * v_dim
            total += conv_dim * kernel * dtype_bytes
            total += v_heads * k_dim * v_dim * state_bytes
    return total
