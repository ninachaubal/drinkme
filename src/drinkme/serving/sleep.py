"""SLEEP / WAKE (vLLM borrow 2): park a loaded model in host RAM and
resume it, instead of tearing the process down and reloading the pack.

vLLM's docs/features/sleep_mode.md: "temporarily release most GPU memory used by a model, including
model weights and KV cache, without stopping the server... Level 1 sleep will
offload the model weights and discard the KV cache... The model weights are
backed up in CPU memory." `POST /sleep?level=1`, `POST /wake_up`.

WHY HERE. A Strix Halo box (gfx1151, 128 GB unified) holds ONE big model at
a time, so two model servers sharing it (two systemd units in Conflict=, say)
swap by stopping one, and drinkme comes back through a full cold load. A
level-1 sleep
turns that swap into a resume: the pack is already decoded, the escape arrays
are already built, the tokenizer is already parsed, and all that has to happen
is a host->device copy of bytes that never leave the machine.

WHAT MOVES. Everything the accelerator holds for this model, in one ordered
pass over the module tree:
  - the trunk's parameters and buffers (the raw tensors: embeddings, norms,
    rotary inv_freq, any Linear the codec found ineligible);
  - every CompressedLinear's `p` dict — the packed streams, directory,
    palette and schedule that codec/swap.py's to_device_radix built (the
    raw weight for a raw-fallback tensor)
    (these are NOT registered parameters; they are dict entries on a plain
    attribute, which is why this file walks vars() and not state_dict());
  - the MTP draft head (serving/mtp.py) and its own packed sub-pack, plus a
    reduced-vocab draft projection if one is installed.
The walk is vars()-based on purpose — engines.cache_bytes and
slotstore.cache_state take the same posture for the same reason: a tensor
this file fails to NAME must still be CARRIED, because a surface we forget is
a surface that stays resident on the device we promised to give back.

WHAT DOES NOT MOVE. The tokenizer (host-side already, and keeping it means
/tokenize, /detokenize, /tokenizer_info and count_tokens keep answering while
the model sleeps), the capability probe, the pack metadata, the cold-tier
index, and CompressedLinear's `_w_cpu` CPU reference weight — a tensor that
was never on the device is not this pass's business. The prefix slots do not
move either: they are PERSISTED (on-disk prefix slots' cold tier, the exact path SIGTERM
takes) and then dropped, which is vLLM's "discard the KV cache" with the
drinkme twist that discarding is not the same as losing.

PINNING IS OFF BY DEFAULT. `DRINKME_SLEEP_PIN=1` opts in. On a unified-memory
machine (gfx1151, 124 GiB shared) pinned host RAM is memory the NEXT model cannot
have, which is the entire point of going to sleep: parking 16 GiB in pages the
allocator may not reclaim would hand the next model a smaller machine than it had
before. Pinning buys a faster wake H2D and costs the thing sleep exists to
give, so it is an explicit ask.

WHERE THE REST OF SLEEP/WAKE LIVES. This file is the TENSOR pass and nothing else.
`SleepState` (the bookkeeping /health reports) and `SleepError` sit on the
engine seam in serving/engine.py, and the idle timer sits in serving/http.py,
because serving/http.py imports neither weights nor torch — its own docstring
says so, and DRINKME_FAKE_ENGINE=1 promises a curl-able endpoint with "no
model, no pack, no GPU". A state object that dragged torch across that seam
would quietly break both.
"""

from __future__ import annotations

import os
import sys

import torch

# nn.Module's own storage dicts: walked explicitly (parameters keep their
# Parameter identity, buffers keep their slot), so the generic vars() sweep
# below must not walk them a second time and replace a Parameter with a plain
# tensor. Everything else in vars() — hook OrderedDicts, `training`, the
# codec's `p` dict, CompressedLinear's `bias` — is fair game; the hook dicts
# simply hold no tensors.
_MODULE_STORAGE = frozenset({"_parameters", "_buffers", "_modules"})

# The inventory buckets the report counts by count + GiB per class.
# "head" wins over the other two: an MTP head served
# from a pack has packed planes of its own, and "how big is the head" is the
# question a reader of the residency arithmetic is actually asking.
PACKED, RAW, HEAD = "packed", "raw", "head"

_GIB = 1024 ** 3


def _log(msg: str, err: bool = False) -> None:
    print(f"[drinkme.sleep] {msg}", file=sys.stderr if err else sys.stdout,
          flush=True)


def human(nbytes: float) -> str:
    """GiB once it is GiB, MiB below that — engines._human's rule, and the
    exact byte count is always printed beside it."""
    return (f"{nbytes / _GIB:.2f} GiB" if nbytes >= _GIB
            else f"{nbytes / (1024 ** 2):.1f} MiB")


def pin_from_env() -> bool:
    """DRINKME_SLEEP_PIN=1 -> stage the host copies in PINNED memory.

    Default OFF, and the module docstring carries the reason: on a unified-
    memory machine pinned RAM is memory the next model cannot have. Anything
    other than "1" is off, deliberately without a warning — this is a
    performance opt-in, not a config whose typo could change what is served.
    """
    return os.environ.get("DRINKME_SLEEP_PIN") == "1"


# ------------------------------------------------------------- the tensors --


class _Ref:
    """One movable tensor and the slot it has to go back into.

    Three kinds, because the three places a model keeps a tensor need three
    different write-backs and only one of them is a plain attribute:
      - "param": the owner IS the Parameter; `p.data = ...` moves it while
        keeping the Parameter object, so a TIED weight (lm_head sharing
        embed_tokens' Parameter) moves once and stays tied;
      - "item":  the owner is a dict (a module's `_buffers`, or a
        CompressedLinear's `p`);
      - "attr":  the owner is any object with `__dict__` (CompressedLinear's
        `bias`, the draft projection's row selector).
    `cls` is the inventory bucket; `name` is for log lines only."""

    __slots__ = ("owner", "key", "kind", "cls", "name")

    def __init__(self, owner, key, kind: str, cls: str, name: str):
        self.owner, self.key, self.kind, self.cls, self.name = (
            owner, key, kind, cls, name)

    def get(self):
        if self.kind == "param":
            return self.owner.data
        if self.kind == "item":
            return self.owner[self.key]
        return getattr(self.owner, self.key)

    def put(self, t) -> None:
        if self.kind == "param":
            self.owner.data = t
        elif self.kind == "item":
            self.owner[self.key] = t
        else:
            setattr(self.owner, self.key, t)


def _here(t, device) -> bool:
    """Is this tensor on the device sleep is emptying?

    The filter that keeps CompressedLinear's `_w_cpu` (the lazy CPU reference
    weight, built by the codec's CPU branch) out of the pass: it was never on
    the accelerator, so parking it is a no-op and WAKING it would move a
    host-side reference weight onto the GPU — memory nobody asked for and a
    tensor the CPU reference route expects to find on the host. Compared by
    device TYPE, not index: drinkme is single-device by construction and
    `cuda` vs `cuda:0` is not a distinction this pass should be inventing."""
    return device is None or t.device.type == torch.device(device).type


def _module_refs(mod, cls: str, prefix: str, seen: set, out: list,
                 device) -> None:
    """Every tensor ONE module owns directly, in a stable order: parameters,
    then buffers, then whatever else vars() is holding.

    Deduplicated by tensor identity across the whole walk, so a tied weight is
    moved once and a `p` dict shared with a draft projection is moved once.
    The dedupe is safe against id() reuse because every object it records is
    still reachable from the live model for the whole walk.

    A Parameter is deduped on the PARAMETER, never on `p.data`: `.data`
    returns a fresh detached view on every access, so its id is a new number
    each time and a tied lm_head/embed_tokens weight would have been walked
    twice — one wasted device->host copy, and a byte count that overstates
    what was parked by a whole embedding table."""
    for name, p in list(mod._parameters.items()):
        if isinstance(p, torch.Tensor) and id(p) not in seen and _here(p, device):
            seen.add(id(p))
            out.append(_Ref(p, name, "param", cls, f"{prefix}{name}"))
    for name, b in list(mod._buffers.items()):
        if isinstance(b, torch.Tensor) and id(b) not in seen and _here(b, device):
            seen.add(id(b))
            out.append(_Ref(mod._buffers, name, "item", cls, f"{prefix}{name}"))
    _object_refs(mod, cls, prefix, seen, out, device, skip=_MODULE_STORAGE)


def _object_refs(obj, cls: str, prefix: str, seen: set, out: list, device,
                 skip=frozenset()) -> None:
    """vars(obj), one level deep: tensor attributes, dict-of-tensor attributes
    (the codec's `p`), and tuple/list-of-tensor attributes (the draft
    projection's escape triple).

    One level, never recursive: an object graph walk would follow
    MTPHead._trunk back into the trunk it deliberately does not own, and
    would need a cycle check to survive it. Every container drinkme actually
    keeps device tensors in is reachable from a module or from the two
    explicit hops walk() makes below."""
    for name, value in list(vars(obj).items()):
        if name in skip:
            continue
        if isinstance(value, torch.Tensor):
            if id(value) not in seen and _here(value, device):
                seen.add(id(value))
                out.append(_Ref(obj, name, "attr", cls, f"{prefix}{name}"))
        elif isinstance(value, dict):
            sub = PACKED if cls != HEAD and name == "p" else cls
            for k, t in list(value.items()):
                if isinstance(t, torch.Tensor) and id(t) not in seen and _here(t, device):
                    seen.add(id(t))
                    out.append(_Ref(value, k, "item", sub, f"{prefix}{name}.{k}"))
        elif isinstance(value, (tuple, list)) and value and all(
                isinstance(t, torch.Tensor) and _here(t, device) for t in value):
            # rebuilt whole rather than written through: a tuple has no slot
            # to assign into, and the sequences drinkme keeps here are short.
            # All-or-nothing on the device test: a mixed sequence is not something this file
            # should be guessing about, and skipping it costs residency, not
            # correctness.
            out.append(_SeqRef(obj, name, cls, f"{prefix}{name}"))


class _SeqRef(_Ref):
    """A tuple/list attribute holding tensors: get() returns the tensors,
    put() rebuilds the sequence around the moved ones. Kept as a _Ref so the
    mover has exactly one shape to handle."""

    __slots__ = ()

    def __init__(self, owner, key, cls, name):
        super().__init__(owner, key, "seq", cls, name)

    def get(self):
        return [t for t in getattr(self.owner, self.key)
                if isinstance(t, torch.Tensor)]

    def put(self, moved) -> None:
        old = getattr(self.owner, self.key)
        it = iter(moved)
        rebuilt = [next(it) if isinstance(t, torch.Tensor) else t for t in old]
        setattr(self.owner, self.key,
                tuple(rebuilt) if isinstance(old, tuple) else rebuilt)


def walk(model, head=None, device=None) -> list[_Ref]:
    """Every tensor of one loaded model that sits on `device`, in module order.

    The order is the module tree's own, which is deterministic for a given
    config — so a sleep and the wake that reverses it touch the same tensors
    in the same sequence, and the peak extra memory of either pass is ONE
    tensor (the same discipline engines.load_compressed's streaming fit path
    keeps: peak host memory is one tensor, never one model).

    `device=None` takes everything, which is what a test asserting on the
    inventory wants; the engine always passes its own."""
    seen: set[int] = set()
    refs: list[_Ref] = []
    if model is not None:
        for name, mod in model.named_modules():
            _module_refs(mod, RAW, f"{name}." if name else "", seen, refs, device)
    if head is not None:
        for name, mod in head.named_modules():
            _module_refs(mod, HEAD, f"mtp.{name}." if name else "mtp.", seen,
                         refs, device)
        # the reduced-vocab draft projection (serving/draft_vocab.py) is a
        # plain attribute holding a NON-Module on purpose (mtp.MTPHead's own
        # docstring), so named_modules() never reaches it.
        proj = getattr(head, "draft_proj", None)
        if proj is not None:
            _object_refs(proj, HEAD, "mtp.draft_proj.", seen, refs, device)
    return refs


def _nbytes(t) -> int:
    return t.numel() * t.element_size()


def inventory(refs: list[_Ref]) -> tuple[int, int, dict[str, int]]:
    """(tensors, bytes, bytes per inventory class) for a ref list, WITHOUT
    moving anything — so a level-2 sleep, which frees rather than copies, can
    still say how much it gave back."""
    total = 0
    by_class: dict[str, int] = {}
    n = 0
    for ref in refs:
        got = ref.get()
        for t in (got if isinstance(got, list) else (got,)):
            if not isinstance(t, torch.Tensor):
                continue
            n += 1
            b = _nbytes(t)
            total += b
            by_class[ref.cls] = by_class.get(ref.cls, 0) + b
    return n, total, by_class


def _to_host(t, pin: bool):
    if not pin:
        return t.detach().to("cpu", copy=True)
    try:
        out = torch.empty(t.shape, dtype=t.dtype, device="cpu", pin_memory=True)
    except RuntimeError as e:  # no pinned allocator (CPU-only build)
        _log(f"pinned host memory unavailable ({e}) — parking pageable", err=True)
        return t.detach().to("cpu", copy=True)
    out.copy_(t.detach())
    return out


def move(refs: list[_Ref], target: str, pin: bool = False) -> int:
    """Move every ref's tensor to `target`, in list order, writing each one
    back before the next is read — so the old copy is unreferenced (and
    reclaimable) one tensor at a time rather than all at the end. Returns the
    bytes moved.

    `pin` only ever applies to a move TO host: pinning a device tensor is not
    a thing, and a wake that pinned would be pinning on the way out."""
    to_host = target == "cpu"
    total = 0
    for ref in refs:
        got = ref.get()
        if isinstance(got, list):
            moved = [(_to_host(t, pin) if to_host else t.detach().to(target))
                     for t in got]
            total += sum(_nbytes(t) for t in moved)
            ref.put(moved)
            continue
        if not isinstance(got, torch.Tensor):
            continue
        total += _nbytes(got)
        ref.put(_to_host(got, pin) if to_host else got.detach().to(target))
    return total


def release(device) -> None:
    """Give the freed blocks back to the driver.

    synchronize() FIRST: with pinned host staging the D2H copies can still be
    in flight, and empty_cache() on a stream that has not finished reading a
    block is how a "returned 16 GiB" number becomes a corrupted parked weight.
    Never fatal — a cache that will not empty is a memory report that reads
    high, not a server that has to stop.

    The BLAS workspaces go too (measured: with every
    model tensor gone, one 32 MiB `active_allocated` segment stayed behind on
    the toy, 36 MiB on the real 4B at level 1). torch hands each cuBLAS /
    hipBLASLt handle a scratch workspace out of the caching allocator on the
    first GEMM and keeps it in a C++ pool no Python object owns, so
    empty_cache() cannot see it as free. _cuda_clearCublasWorkspaces is the
    hook torch's own tests use to drop that pool; the next GEMM (the wake's
    prime) re-allocates a workspace of the same configured size, so the
    kernels chosen after a wake are the ones chosen before it. Private API,
    hence the getattr: a build without it keeps the 32 MiB and says so."""
    if not str(device).startswith("cuda"):
        return
    try:
        torch.cuda.synchronize()
        clear_ws = getattr(torch._C, "_cuda_clearCublasWorkspaces", None)
        if clear_ws is not None:
            clear_ws()
        else:
            _log("torch has no _cuda_clearCublasWorkspaces — the BLAS "
                 "workspace (tens of MiB) stays reserved", err=True)
        torch.cuda.empty_cache()
    except Exception as e:  # noqa: BLE001 — containment is the point
        _log(f"empty_cache failed ({type(e).__name__}: {e}) — the copies "
             "landed; the allocator kept its blocks", err=True)
