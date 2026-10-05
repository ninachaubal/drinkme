"""The vision tower's output, kept in host RAM between requests.

WHY. The prefix cache skips the tower for an image or a video only when the
request reuses the slot through that media's end
(image_prompt.ImagePrompt.begin). The same media in another conversation
(another system prompt, other turns before it, the media at another
position) shares no prefix with any slot, so the tower ran again over the
same pixels. A TowerCache keeps what the tower made of each media item,
and a request that carries the same item again copies it back instead of
running the tower.

THE KEY is (`identity`, the item's kind, its `digest`, its `grid_thw`).
- `digest` is sha256 over the exact tensor the tower reads: the
  pixel_values bytes, then the grid (vision.image_digest, and
  video.video_digest behind a b"video" tag). Everything that decides the
  tower's input is inside it: for a video the frames sampled (the sampling
  plan), the size they were resized to (the grid), and the processor's
  resize and normalization; for an image the same, from the pixel cap and
  `detail`. Two items whose inputs agree byte for byte get one entry, and
  any difference in the input is a different key. Keying on the source
  bytes instead would also need the processor config and the pixel cap in
  the key, and would still not see a change to drinkme's own resize.
- `grid_thw` is inside the digest already, and is kept in the key so an
  entry can be read without the digest's preimage.
- `identity` is the engine's: model id, arm, the tower's architecture and
  path, and its parameters' dtype (HFEngine builds it). One cache serves
  one engine and one tower. The identity is in the key so that a cache
  shared between engines one day could not serve the stock arm's output
  to the compressed arm, whose kernels may round differently.

The decode still runs on every request: the dialects prepare a video
(serving/video.py) before the engine sees it, and its digest is over the
decoded pixels. The cache saves the tower's forward pass, which is what
reran on a follow-up.

THE CAP. An entry is the tower's output, [tokens, text hidden] at the
tower's dtype: on Qwen3.8-27B (hidden 5,120, bf16) 10,240 B per token,
27.0 MB for a 2,640-token clip, 126 MB for a video at the 12,288-token
budget, 36.9 MB for a 3,600-token image at the default pixel cap. The cap
is DRINKME_TOWER_CACHE_GIB (`drinkme serve --tower-cache-gib`), default
DEFAULT_GIB = 0.5: 19 such clips, 14 such images, or 4 videos at the
budget. It is host RAM, and on a unified-memory machine host RAM is memory
the model cannot have (slotstore.py keeps its host-RAM tier off by default
for that reason), so the default is small. Entries are evicted least
recently used first. An entry larger than the whole cap is not kept. 0
turns the cache off.

EXACTNESS. A hit is the tower's own output: copied to host when it was
made, and back to the device byte for byte. The prompt's embeddings, and
so every logit, are the ones a miss computes.

Single-flight: generate() runs under the HTTP layer's generation lock, and
the lock here only keeps the counters and the order consistent if a caller
does not hold it.
"""

from __future__ import annotations

import os
import sys
import threading
from collections import OrderedDict

ENV = "DRINKME_TOWER_CACHE_GIB"
DEFAULT_GIB = 0.5
_GIB = 1024 ** 3


def cap_from_env(explicit: float | None = None) -> int:
    """The cap in bytes: `explicit` GiB (--tower-cache-gib) over
    DRINKME_TOWER_CACHE_GIB over DEFAULT_GIB. 0 turns the cache off. Garbage
    or a negative size warns and falls back to the default, the posture of
    slotstore's GiB knobs."""
    if explicit is not None:
        if explicit < 0:
            print(f"[drinkme] --tower-cache-gib {explicit} is not a size of 0 or more — "
                  f"using {DEFAULT_GIB}", file=sys.stderr, flush=True)
            return int(DEFAULT_GIB * _GIB)
        return int(explicit * _GIB)
    raw = os.environ.get(ENV, "").strip()
    if not raw:
        return int(DEFAULT_GIB * _GIB)
    try:
        val = float(raw)
    except ValueError:
        val = -1.0
    if not val >= 0:  # also NaN
        print(f"[drinkme] {ENV}={raw!r} is not a size in GiB — using {DEFAULT_GIB}",
              file=sys.stderr, flush=True)
        return int(DEFAULT_GIB * _GIB)
    return int(val * _GIB)


def _nbytes(t) -> int:
    return t.numel() * t.element_size()


def human(nbytes: float) -> str:
    """GiB once it is GiB, MiB below that (engines._human's rule)."""
    return (f"{nbytes / _GIB:.2f} GiB" if nbytes >= _GIB
            else f"{nbytes / (1024 ** 2):.1f} MiB")


class TowerCache:
    """The tower's outputs for one engine, least recently used first, at
    most `cap` bytes (module docstring)."""

    def __init__(self, cap: int, identity: tuple = ()):
        self.cap = max(0, int(cap))
        self.identity = tuple(identity)
        self._items: OrderedDict = OrderedDict()
        self.nbytes = 0
        self.hits = self.misses = self.evictions = 0
        self._lock = threading.Lock()

    @property
    def on(self) -> bool:
        return self.cap > 0

    def __len__(self) -> int:
        return len(self._items)

    def key(self, item) -> tuple:
        """A PreparedImage's or PreparedVideo's key (module docstring, THE
        KEY)."""
        return (*self.identity, type(item).__name__, item.digest,
                tuple(int(g) for g in item.grid_thw))

    def get(self, item, device):
        """The tower's output for `item` on `device`, or None. A hit becomes
        the most recently used entry."""
        if not self.on:
            return None
        k = self.key(item)
        with self._lock:
            f = self._items.get(k)
            if f is None:
                self.misses += 1
                return None
            self._items.move_to_end(k)
            self.hits += 1
        return f.to(device)

    def put(self, item, features) -> bool:
        """Keep a host copy of the tower's output for `item`, evicting the
        least recently used entries until it fits. False when the cache is
        off or the entry alone is larger than the cap."""
        size = _nbytes(features)
        if not self.on or size > self.cap:
            return False
        k = self.key(item)
        host = features.detach().to("cpu", copy=True)
        with self._lock:
            old = self._items.pop(k, None)
            if old is not None:
                self.nbytes -= _nbytes(old)
            while self._items and self.nbytes + size > self.cap:
                _k, gone = self._items.popitem(last=False)
                self.nbytes -= _nbytes(gone)
                self.evictions += 1
            self._items[k] = host
            self.nbytes += size
        return True

    def clear(self) -> None:
        with self._lock:
            self._items.clear()
            self.nbytes = 0

    def describe(self) -> str:
        """The boot line's half after "tower cache: "."""
        if not self.on:
            return f"off ({ENV}=0)"
        return (f"up to {human(self.cap)} of the tower's output in host RAM, least "
                f"recently used out first ({ENV} or --tower-cache-gib; 0 turns it off)")
