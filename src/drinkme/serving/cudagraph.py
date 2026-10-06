"""CUDA graphs for the decode step: CUDA only, on by default where verified,
experimental (docs/serve-kernels.md#cuda-graphs).

On NVIDIA an eager decode step is bound by the host: about 2,000 kernel
launches per Qwen3-8B token, 9-17 ms of host time over the GPU's own on an
H100. One `torch.cuda.CUDAGraph` per step replays those launches from the
device instead: H100 8B stock 18.9 -> 10.6 ms/step in the spike.

WHAT IS CAPTURED. One forward of `rows` tokens (the M = 1 decode step, or
a speculative verify of M rows) over a LiveStaticCache (serving/kvcache.py),
with every input on the device:

  - the tokens are a static [1, rows] buffer the caller fills before replay;
  - the positions are the cache's own device counter (`cumulative_length`,
    before this step's write) plus 0..rows-1, passed as `position_ids`, so
    transformers never reads the Python `live` int into a position (that
    int would be baked into the graph at capture);
  - attention reads the WHOLE allocation, with the live length in a device
    tensor: the mask mapping carries a `LiveLength` marker, and
    `attention_forward` calls torch's flash kernel in its varlen form with
    `seqused_k`, so the kernel reads only the live rows and never needs a
    mask (no `repeat_kv`, native grouped-query heads, bottom-right causal
    alignment for M > 1 rows). This is flash-decoding over the allocation;
    the cost that a masked StaticCache carried (kvcache.py's docstring,
    88.6 against 60.2 ms/token at 4096) does not come back, because the
    kernel's work follows `seqused_k`, not the allocation. Measured with
    bench/cuda_graph_gate.py (Qwen3-8B stock on an L4, where the step is
    GPU-bound): the static step run eagerly at 4096 and 16384 live tokens
    (allocations 8192 and 32768) is 1.001x and 1.002x today's eager step,
    so length-bucketed graphs were not needed.

Replay writes the KV rows and advances the device counter inside the
graph; the Python `live` int on each LiveStaticLayer is advanced by the
host after each replay (`StaticStep.replay`), so `rewind`, `resync` and the
context checkpoints keep seeing the counter and the int agree.

WHO USES IT. Serve's serial step (serving/engines.py) and bench's decode
loop (arms.timed_decode), through `decide()`:

  - `DRINKME_CUDA_GRAPHS=0` forces eager;
  - ROCm, Metal and the CPU never graph (TheRock's HIP graph capture has
    segfaulted on the pinned build; nothing is announced there);
  - a model family and mode not on `VERIFIED` decodes eager, with one line
    saying so. A (family, mode) goes on the list only after
    bench/cuda_graph_gate.py's verdicts on it: graph replay matched the
    eager static step bit for bit, and serve's own generate in graph mode
    (against eager, speculation off and on, and across arms) parted from
    the other runs nowhere or only at a near-tie;
  - a capture that fails decodes eager, with one line saying so.

MEMORY. Every graph captured against one cache (the decode step, the MTP
re-arm step and each verify width) allocates from ONE private pool
(`pool_for`), and warms up and captures on one stream per device
(`capture_stream`), so lazily created per-stream state (cuBLAS workspaces)
lands outside the pool once instead of inside each graph's. Sharing is safe
because only one graph runs at a time (the HTTP layer serializes
generations) and everything a replay produces is read before the next
replay of any graph in the pool: the step's outputs are cloned at once
(`decode_step`'s caller, mtp.Speculator._trunk), and a verify's DeltaNet
rows are read by the same cycle's rewind (mtp._restore_rows) or by
`finish` before the loop replays again. The KV rows, the DeltaNet states
and every static input live outside the pool (allocated before capture).
On a DeltaNet hybrid the verify graphs write each row's recurrent state
into one buffer per cache (`row_states`, sized for the widest width),
where each graph used to hold its own: the 27B's graphs went from about
2.4 GiB per cache to the pool plus that buffer. What a cache's graphs
cost, and the fit checks' charge for it, is serving/cudagraph_fit.py's.
`SHARED = False` is the per-graph-pool layout (a private pool per graph, a new side
stream per capture, per-graph row copies), kept for the gate's
before/after measurement.

Numerics: graph replay is the eager static step's own kernels, bitwise
(the gate's first check). Against today's eager step, the attention kernel
differs (varlen flash over the allocation against SDPA over the live
window): same softmax, possibly a different split of the keys, so greedy
transcripts can differ at a near-tie, the class docs/method.md describes.
"""

from __future__ import annotations

import sys
import weakref
from dataclasses import dataclass
from typing import Callable, NamedTuple

import torch

from .cudagraph_fit import ENV, VERIFIED  # noqa: F401  (the policy's table, torch-free)

# the gate's before/after seam (module docstring, MEMORY): True shares one
# pool per cache, one capture stream per device and one row buffer per cache
SHARED = True

# warm-up forwards before a capture: the lazy state (Triton compiles and
# autotunes, cuBLAS workspaces, the flash kernel's first launch) lands
# outside the graph. Each is rewound.
WARMUP = 2


class Decision(NamedTuple):
    """Whether this model decodes in graph mode here, and the line to print
    (None: nothing to announce, e.g. on ROCm)."""

    on: bool
    line: str | None


def platform(device=None) -> str:
    """cuda | rocm | metal | cpu, for the device the model runs on."""
    if getattr(torch.version, "hip", None):
        return "rocm"
    dev = torch.device(device) if device is not None else None
    if dev is not None and dev.type == "mps":
        return "metal"
    if dev is not None and dev.type != "cuda":
        return "cpu"
    if dev is None and not torch.cuda.is_available():
        return "cpu"
    return "cuda"


def family(model) -> str | None:
    """The model's `model_type` (the VERIFIED key): the top-level config's,
    which is what the loaders and mtp.trunk_ids key on too."""
    cfg = getattr(model, "config", None)
    return getattr(cfg, "model_type", None)


def env_off() -> bool:
    from .cudagraph_fit import env_off as off

    return off()


def decide(model, device=None, mode: str = "decode") -> Decision:
    """The policy (module docstring): env override, platform, verified list."""
    if platform(device) != "cuda":
        return Decision(False, None)
    if env_off():
        return Decision(False, f"decode step: eager ({ENV}=0)")
    fam = family(model)
    if (fam, mode) not in VERIFIED:
        return Decision(False, f"decode step: eager; CUDA graphs are not verified for "
                               f"{fam or 'this model'} ({mode})")
    return Decision(True, f"decode step: CUDA graphs (experimental; {ENV}=0 for eager)")


def verified(model, mode: str) -> bool:
    return (family(model), mode) in VERIFIED


# ------------------------------------------------------------- attention --


@dataclass
class LiveLength:
    """The attention-mask marker a static step passes for its full-attention
    layers: `rows` query rows over an allocation of `alloc` keys, of which
    the first `live[0]` (int32, on the device) are written."""

    rows: int
    alloc: int
    live: torch.Tensor
    cu_q: torch.Tensor
    cu_k: torch.Tensor


def _flash(query, key, value, m: LiveLength, scaling):
    # [1, H, M, D] -> [M, H, D]; [1, Hk, L, D] -> [L, Hk, D] as views (the
    # kernel takes any row/head stride with a unit last stride)
    q = query[0].transpose(0, 1)
    k = key[0].transpose(0, 1)
    v = value[0].transpose(0, 1)
    out = torch.ops.aten._flash_attention_forward(
        q, k, v, m.cu_q, m.cu_k, m.rows, m.alloc, 0.0, m.rows > 1, False,
        scale=scaling, seqused_k=m.live)[0]
    return out.unsqueeze(0)


def _reference(query, key, value, m: LiveLength, scaling):
    """The same attention through SDPA with a mask built from the device
    length: the CPU tests' path, and the numerics the flash call must meet."""
    alloc, rows = key.shape[-2], query.shape[-2]
    cols = torch.arange(alloc, device=key.device)
    last = m.live.to(torch.long) - rows + torch.arange(rows, device=key.device)
    mask = (cols.view(1, -1) <= last.view(-1, 1)).view(1, 1, rows, alloc)
    out = torch.nn.functional.scaled_dot_product_attention(
        query, key, value, attn_mask=mask, scale=scaling,
        enable_gqa=query.shape[1] != key.shape[1])
    return out.transpose(1, 2).contiguous()


def attention_forward(module, query, key, value, attention_mask, dropout=0.0,
                      scaling=None, **kwargs):
    """The marker's path, in transformers' attention-function shape."""
    m = attention_mask
    if query.shape[-2] != m.rows or key.shape[-2] != m.alloc:
        raise ValueError("static step: the query rows or the KV allocation do not match the marker")
    if query.device.type == "cuda" and not getattr(torch.version, "hip", None):
        return _flash(query, key, value, m, scaling), None
    return _reference(query, key, value, m, scaling), None


def _dispatch(inner):
    def sdpa_or_live(module, query, key, value, attention_mask, *args, **kwargs):
        if isinstance(attention_mask, LiveLength):
            return attention_forward(module, query, key, value, attention_mask, *args, **kwargs)
        return inner(module, query, key, value, attention_mask, *args, **kwargs)

    sdpa_or_live.__wrapped__ = inner
    return sdpa_or_live


def register() -> None:
    """Wrap transformers' "sdpa" attention function so the marker reaches
    `attention_forward`; every other mask passes through. Idempotent."""
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    fn = ALL_ATTENTION_FUNCTIONS["sdpa"]
    if getattr(fn, "__name__", "") != "sdpa_or_live":
        ALL_ATTENTION_FUNCTIONS.register("sdpa", _dispatch(fn))


# ----------------------------------------------------------- the step --


class Unsupported(Exception):
    """This cache or model cannot take a static step (the caller decodes eager)."""


def _layer_kinds(model) -> list | None:
    cfg = getattr(model, "config", None)
    text = cfg.get_text_config() if hasattr(cfg, "get_text_config") else cfg
    return getattr(text, "layer_types", None)


def mask_mapping(model, marker: LiveLength) -> dict:
    """The attention-mask mapping a static step passes: the marker for the
    full-attention layers, None for linear attention (DeltaNet takes none)."""
    kinds = _layer_kinds(model) or ["full_attention"]
    out = {}
    for kind in set(kinds):
        if kind == "full_attention":
            out[kind] = marker
        elif kind == "linear_attention":
            out[kind] = None
        else:
            raise Unsupported(f"a {kind} layer")
    return out


class StaticStep:
    """One forward of `rows` tokens over `cache` with every input on the
    device (module docstring), run eagerly or captured and replayed.

    `forward(ids, position_ids, mask)` is the caller's forward over the
    model and `cache`, returning the tensor (or tuple of tensors) the step
    produces; it runs eagerly, and once more inside the capture. `ids` is
    the static token buffer the caller fills before each run."""

    def __init__(self, model, cache, rows: int, forward: Callable, max_rows: int | None = None):
        from transformers.cache_utils import LinearAttentionCacheLayerMixin

        from .kvcache import LiveStaticLayer

        layers = [layer for layer in cache.layers if isinstance(layer, LiveStaticLayer)]
        linear = [layer for layer in cache.layers if isinstance(layer, LinearAttentionCacheLayerMixin)]
        others = [layer for layer in cache.layers
                  if not isinstance(layer, (LiveStaticLayer, LinearAttentionCacheLayerMixin))]
        if not layers or others:
            raise Unsupported("a cache layer other than LiveStaticLayer or DeltaNet's: "
                              + ", ".join(sorted({type(layer).__name__ for layer in others})))
        if not all(layer.is_initialized for layer in layers):
            raise Unsupported("an unwritten cache")
        # DeltaNet's conv and recurrent states are updated with copy_ (static
        # addresses: a captured graph keeps reading and writing the same
        # tensors); a layer recording its past re-binds them instead
        if any(layer.record_past or layer.conv_states.get(0) is None
               or layer.recurrent_states.get(0) is None or not layer.has_previous_state.get(0)
               for layer in linear):
            raise Unsupported("a DeltaNet state that is unwritten or re-bound per step")
        # the cache weakly: the cache keeps its steps (_step), and a strong
        # reference back would make the pair a cycle that outlives the slot
        # that dropped it, graphs and KV included, until the cyclic GC runs
        self._cache = weakref.ref(cache)
        self.model, self.rows, self.forward = model, rows, forward
        # the widest verify step this cache will capture (row_states' size)
        self.max_rows = max(rows, max_rows or rows)
        self.layers = layers
        self.linear = linear
        self.counter = layers[0].cumulative_length
        dev = self.counter.device
        alloc = layers[0].max_cache_len
        self.ids = torch.zeros((1, rows), dtype=torch.long, device=dev)
        self.offsets = torch.arange(rows, device=dev)
        self.marker = LiveLength(rows, alloc, torch.zeros(1, dtype=torch.int32, device=dev),
                                 torch.tensor([0, rows], dtype=torch.int32, device=dev),
                                 torch.tensor([0, alloc], dtype=torch.int32, device=dev))
        self.mask = mask_mapping(model, self.marker)
        self.graph = None
        self.out = None
        register()

    @property
    def cache(self):
        return self._cache()

    def _run(self):
        pos = self.counter + self.offsets
        self.marker.live.copy_(pos[-1:] + 1)
        for layer in self.layers:
            layer.whole = True
        try:
            return self.forward(self.ids, pos.view(1, -1), self.mask)
        finally:
            for layer in self.layers:
                layer.whole = False

    def _lengths(self) -> list[int]:
        return [layer.live for layer in self.layers]

    def snapshot(self) -> tuple:
        """What a run changes: the full-attention lengths, and a copy of
        each DeltaNet layer's conv and recurrent state (they cannot rewind)."""
        return (self._lengths(), [(layer.conv_states[0].clone(), layer.recurrent_states[0].clone())
                                  for layer in self.linear])

    def restore(self, snap: tuple) -> None:
        """Back to `snap`: counter and int per full-attention layer (the rows
        past them stay in the buffer, outside every window), DeltaNet states
        copied back in place."""
        lengths, states = snap
        for layer, n in zip(self.layers, lengths):
            layer.cumulative_length.fill_(n)
            layer.live = n
        for layer, (conv, rec) in zip(self.linear, states):
            layer.conv_states[0].copy_(conv)
            layer.recurrent_states[0].copy_(rec)

    def eager(self):
        """The static step, not captured: the gate's reference for replay."""
        return self._run()

    def capture(self, shared: bool | None = None) -> None:
        """Warm up (each run rewound), then capture one step. Leaves the
        cache as it found it; raises whatever capture raised. `shared`
        (default SHARED): into the cache's pool, warmed up and captured on
        the device's capture stream, a hybrid's verify rows into the
        cache's row buffer (module docstring, MEMORY); False is the
        per-graph private pool and fresh side stream."""
        shared = SHARED if shared is None else shared
        pool = stream = None
        if shared:
            pool, stream = pool_for(self.cache), capture_stream(self.counter.device)
            if self.rows > 1 and self.linear:
                row_states(self.cache, self.max_rows)
        before = self.snapshot()
        side = stream if stream is not None else torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        try:
            with torch.cuda.stream(side):
                for _ in range(WARMUP):
                    self._run()
                    self.restore(before)
            torch.cuda.current_stream().wait_stream(side)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=pool, stream=stream, capture_error_mode="thread_local"):
                out = self._run()
        finally:
            torch.cuda.current_stream().wait_stream(side)
            self.restore(before)
        self.graph, self.out = graph, out
        if shared:
            # the warm-ups' blocks sit cached on the capture stream, where
            # nothing else can reuse them (1.0 GiB after the 27B's widest
            # verify, bench/cuda_graph_gate.py --memory): hand them back, as
            # torch.cuda.graph does before each capture
            del before
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

    def replay(self):
        """One step: the graph, then the host's side of the bookkeeping."""
        self.graph.replay()
        for layer in self.layers:
            layer.live += self.rows
        return self.out


# ------------------------------------------------------------ memory --

_STREAMS: dict = {}


def capture_stream(device) -> "torch.cuda.Stream":
    """The one stream a device's warm-ups and captures run on (module
    docstring, MEMORY)."""
    dev = torch.device(device)
    idx = dev.index if dev.index is not None else torch.cuda.current_device()
    if idx not in _STREAMS:
        with torch.cuda.device(idx):
            _STREAMS[idx] = torch.cuda.Stream()
    return _STREAMS[idx]


def pool_for(cache):
    """The private pool every graph of `cache` allocates from; it lives as
    long as the cache (and its graphs) do."""
    pool = cache.__dict__.get("_drinkme_pool")
    if pool is None:
        pool = cache.__dict__["_drinkme_pool"] = torch.cuda.graph_pool_handle()
    return pool


def row_states(cache, rows: int) -> dict:
    """{layer index: [rows - 1, *state]} per DeltaNet layer of `cache`: the
    buffer a captured verify of up to `rows` rows writes each row's
    recurrent state into (mtp._deltanet_forward_capturing reads it off the
    cache while capturing), in the cache state's dtype, allocated outside
    any pool. The last row is never stored: a verify that keeps every row
    needs no rewind. A width wider than the buffer holds gets a new one;
    the old stays alive with the cache, since the graphs captured before
    still write it (callers size it for their widest width, so this does
    not happen in serve)."""
    from transformers.cache_utils import LinearAttentionCacheLayerMixin

    have = cache.__dict__.get("_drinkme_row_states")
    if have is not None and cache.__dict__.get("_drinkme_row_cap", 0) >= rows - 1:
        return have
    if have is not None:
        cache.__dict__.setdefault("_drinkme_row_old", []).append(have)
    n = max(rows - 1, 1)
    bufs = {}
    for i, layer in enumerate(cache.layers):
        if isinstance(layer, LinearAttentionCacheLayerMixin) and layer.recurrent_states.get(0) is not None:
            st = layer.recurrent_states[0]
            bufs[i] = torch.empty((n, *st.shape), dtype=st.dtype, device=st.device)
    cache.__dict__["_drinkme_row_states"] = bufs
    cache.__dict__["_drinkme_row_cap"] = n
    return bufs


def graph_bytes(cache) -> int:
    """What `cache`'s graphs hold right now, by the allocator: its pool's
    reserved segments plus the row buffer (0 before any capture)."""
    pool = cache.__dict__.get("_drinkme_pool")
    total = 0
    if pool is not None:
        for seg in torch.cuda.memory_snapshot():
            if tuple(seg.get("segment_pool_id", (0, 0))) == tuple(pool):
                total += seg["total_size"]
    for bufs in [cache.__dict__.get("_drinkme_row_states") or {},
                 *cache.__dict__.get("_drinkme_row_old", [])]:
        total += sum(buf.numel() * buf.element_size() for buf in bufs.values())
    return total


# ---------------------------------------------------- serve's M = 1 step --

_SAID: set = set()


def say_once(key: str, line: str) -> None:
    if key not in _SAID:
        _SAID.add(key)
        print(f"[drinkme] {line}", file=sys.stderr, flush=True)


def _step(model, cache, key: str, rows: int, forward, max_rows: int | None = None) -> StaticStep | None:
    """The captured step `key` for `cache`, built on first use and kept on
    the cache (whose tensors it replays against; it dies with the cache), or
    None when this cache cannot take one or its capture failed. Either is
    said once; the caller then decodes eager."""
    steps = cache.__dict__.setdefault("_drinkme_steps", {})
    if key in steps:
        return steps[key]
    try:
        step = StaticStep(model, cache, rows, forward, max_rows)
        step.capture()
    except Unsupported as e:
        say_once(f"unsupported:{e}", f"decode step: eager for this cache ({e})")
        step = None
    except Exception as e:  # noqa: BLE001 — a failed capture decodes eager, never fails the request
        say_once("capture", f"decode step: CUDA graph capture failed ({type(e).__name__}: "
                            f"{str(e)[:200]}); decoding eager")
        step = None
    steps[key] = step
    return step


def decode_step(model, cache) -> StaticStep | None:
    """The serial loop's M = 1 step: the wrapper's forward, output the last
    row's logits [V] as the loop reads them. The forwards below reach the
    cache through a weak reference (StaticStep.__init__'s reason); they run
    only while it lives (warm-up and capture)."""
    ref = weakref.ref(cache)

    def forward(ids, position_ids, mask):
        return model(input_ids=ids, position_ids=position_ids, attention_mask=mask,
                     past_key_values=ref(), use_cache=True).logits[0, -1]

    return _step(model, cache, "decode", 1, forward)


def verify_step(model, cache, rows: int, max_rows: int | None = None) -> StaticStep | None:
    """A speculative verify of `rows` rows (serving/mtp.py's cycle): the
    trunk through mtp.forward_with_hidden, output (hidden [1, rows, H],
    logits [rows, V]). One graph per width; `max_rows` is the widest the
    caller will ask for (row_states' size)."""
    from .mtp import forward_with_hidden

    ref = weakref.ref(cache)

    def forward(ids, position_ids, mask):
        return forward_with_hidden(model, ids, ref(), None, attention_mask=mask,
                                   position_ids=position_ids)

    return _step(model, cache, f"verify{rows}", rows, forward, max_rows)


def hidden_step(model, cache) -> StaticStep | None:
    """mtp.Speculator.serial_step's M = 1 forward (the hidden row kept for
    the head's re-arm): output (hidden [1, 1, H], logits [1, V])."""
    from .mtp import forward_with_hidden

    ref = weakref.ref(cache)

    def forward(ids, position_ids, mask):
        return forward_with_hidden(model, ids, ref(), None, last_row_only=True,
                                   attention_mask=mask, position_ids=position_ids)

    return _step(model, cache, "hidden", 1, forward)
