"""The measurement arms on the mlx runtime: mlx-lm bf16 -> twin ->
fused compressed. arms.py's protocol (three isolated passes, 64 greedy
tokens, one untimed warm-up rep then timed 3x, median + samples) with the
MLX engine under each arm, over any pack the engine serves: radix bf16
with its raw fallbacks (serving/engine_mlx dispatches per tensor).

  stock       mlx-lm's own bf16 forward of the checkpoint — every tensor as
              the Mac community already runs it. serving/engine_mlx.load_stock_mlx
              is that model behind drinkme's server; the arm times the model.
  twin        the MLX engine over the pack on the TWIN path
              (engine_mlx.RadixTwinLinear): every radix tensor decoded
              ONCE at load by metal/twin_radix (plain mlx ops, bitwise the
              CPU oracle, no code shared with the fused kernel) and held as
              bf16, then the reference path's own matmul over it — its
              outputs are the reference path's, bit for bit. A CORRECTNESS
              twin, not a kernel-quality one — a different animal from
              arms.py's torch twin, which reads raw bf16 through the
              compressed arm's own kernels: its decode speed is measured
              (it IS the exactness check — the pack must decode and run),
              but it is mlx's matmul over bf16, not a bandwidth control,
              and it is never published alongside a torch twin's number,
              which would mislead by column. The decode is paid at load,
              outside the timed passes (the reference path pays it on every
              forward: 1.17 tok/s on an M4's Qwen3-0.6B). No prefill/TTFT
              timing on this arm either, same reasoning as arms.py's
              twin: it has nothing to isolate at M>1.
  compressed  the MLX engine over the pack on the FUSED path: metal/gemv_radix,
              W never built. The plotted number.

MIN-OF-N WITH A DRIFT BAND (docs/metal.md): a fanless Apple machine throttles
under sustained load — the M1 Air drifted +43% across six identical runs —
so beside the median each arm reports its minimum and its spread, and the
headline ratio is computed on the minima. A spread wider than the ratio is
a thermal fact, not a kernel fact, and the record says so.

On a machine without Metal (`fused=False`, the dry run tests/test_bench.py
drives on the toy) the compressed arm runs the reference path too and the
report names it (`compressed_path`), so a CPU number can never wear the
fused label.

The round trip is a write-time GATE here too: a compressed engine that
swapped zero Linears (nothing eligible got the pack's verify-at-load
treatment) refuses the whole run rather than publish a record with an
unverified — or absent — compressed arm. No fidelity metric is published
(arms.py's module docstring says why).
"""

from __future__ import annotations

import json
import os
import time

import numpy as np

from . import exitcodes
from .probe import WARMUP_REP

DECODE_REPS = 3
N_NEW = 64
PREFILL_LEN = 512

# arms.py's own constant, duplicated rather than imported: that module pulls
# in torch at import time, and this one runs on the mlx runtime, without torch.
PREFILL_PASSAGE = (
    "The key idea of lossless weight compression is to store exactly the "
    "bits a model was trained with, using fewer bytes, so that decoding "
    "reproduces the original weights bit for bit. "
)


GIB = 1024 ** 3
# probe.measure_bandwidth's repeat count, so the two runtimes time the same protocol
PROBE_REPS = 5
# the probe buffer asked for: 2 GiB, what probe.probe_size_for gives on Apple
# silicon (its mps branch) and the size the M4's 106 GB/s bf16 read was measured
# at; a Mac with less than 8 GiB of working set gets a quarter of it instead
PROBE_REQUEST = 2 * GIB


def _return_cache(mx) -> None:
    """Hand MLX's buffer cache back to macOS. synchronize() first: freed
    buffers are still owned by in-flight command buffers, and Metal's
    completion handler returns them to MLX's cache AFTER a bare
    clear_cache() has run (4 GiB after the 2 GiB probe on an M4).
    Called only outside timed passes."""
    mx.synchronize()
    mx.clear_cache()
    _settle_host()


# macOS's free count lags a hand-back: on an M4 (09-28) vm_stat went 9.27 -> 13.01 GiB within
# 0.52 s of the probe's synchronize + clear, and the stock arm's fit check, which reads it at once,
# refused the 4B at 8.70 GiB "host available".
SETTLE_MIN_S = 0.75
SETTLE_MAX_S = 3.0
SETTLE_AGREE_BYTES = 64 * 2**20


def _settle_host(min_s: float = SETTLE_MIN_S, max_s: float = SETTLE_MAX_S,
                 poll_s: float = 0.1, clock=time.monotonic, sleep=time.sleep) -> None:
    """Wait until macOS's free count has caught up with a cache hand-back: at least `min_s`,
    then until two vm_stat reads agree within SETTLE_AGREE_BYTES, never past `max_s`. Off
    macOS, or when vm_stat can't be read, it returns at once. Outside timed passes only."""
    import platform

    if platform.system() != "Darwin":
        return
    from .fit import host_available_bytes

    start = clock()
    prev = host_available_bytes()
    if prev is None:
        return
    sleep(min_s)
    while clock() - start < max_s:
        cur = host_available_bytes()
        if cur is None or abs(cur - prev) <= SETTLE_AGREE_BYTES:
            return
        prev = cur
        sleep(poll_s)


def measure_bandwidth(probe_bytes: int | None = None) -> dict:
    """probe.measure_bandwidth's protocol with mx in torch's seat, on the
    empty device BEFORE any model is loaded: one bf16 buffer of
    `probe_bytes` (default: probe_size_for_mlx), both paths warmed once,
    then PROBE_REPS timed full reductions (the read) and PROBE_REPS timed
    copies. Returns {probe_bytes, read_bytes_s, copy_bytes_s, device}:
    BYTES and bytes/second; the record boundary converts to decimal GB/s.

    The read is mx.sum over the bf16 buffer itself, so it reads each byte
    once and holds nothing beyond the buffer. A cast to fp32 before the sum
    materialises a 2x copy every rep: on an M4 (24 GB, MLX 0.32.2) that
    read 14.3 GB/s inside bench where the bf16 sum reads 106. The copy is
    `buf + 0` (read + write, the STREAM convention), so the probe peaks at
    two buffers. Afterwards mx.synchronize() then mx.clear_cache() hands the
    freed buffers back; without the clear MLX kept them cached (8 GiB on that
    M4) through every arm, and without the synchronize Metal's completion
    handler put 4 GiB back into the cache after the clear (the same order
    engine_mlx._return_cache uses)."""
    import mlx.core as mx

    from .probe import bandwidth_from_timings

    if probe_bytes is None:
        probe_bytes = probe_size_for_mlx()
    n = int(probe_bytes) // 2
    buf = mx.ones((n,), dtype=mx.bfloat16)
    zero = mx.array(0, dtype=mx.bfloat16)
    mx.eval(buf)
    mx.eval(mx.sum(buf))  # warm both paths
    mx.eval(buf + zero)
    t0 = time.perf_counter()
    for _ in range(PROBE_REPS):
        mx.eval(mx.sum(buf))
    read_seconds = time.perf_counter() - t0
    t0 = time.perf_counter()
    for _ in range(PROBE_REPS):
        mx.eval(buf + zero)
    copy_seconds = time.perf_counter() - t0
    del buf, zero  # freed before clear_cache, so no buffer of the probe's stays cached
    _return_cache(mx)
    return {"probe_bytes": n * 2,
            **bandwidth_from_timings(n * 2, PROBE_REPS, copy_seconds, read_seconds),
            "device": _device()}


def probe_size_for_mlx(requested: int = PROBE_REQUEST) -> int:
    """probe.probe_size_for's rule (a quarter of free memory, whole MiB, at
    most `requested`, at least 256 MiB) on the mlx runtime, where the free
    memory is Metal's recommended working set less what MLX already holds.
    Without Metal there is no working set to bound it, and the probe takes
    the request (PROBE_REQUEST) itself."""
    import mlx.core as mx

    from .probe import probe_size_from_free

    ws = None
    if mx.metal.is_available():
        try:
            ws = int(mx.device_info()["max_recommended_working_set_size"])
        except Exception:  # noqa: BLE001 — a probe size must not take the run down
            ws = None
    free = 4 * int(requested) if ws is None else ws - int(mx.get_active_memory())
    return probe_size_from_free(free, requested)


def _device() -> str:
    from .serving.engine_mlx import device_name

    return device_name()


def greedy(engine, ids: list[int], n_new: int) -> list[int]:
    """n_new greedy tokens through the engine's model with a fresh mlx-lm
    cache — arms.greedy's loop on this side. Every step heads the last
    position only (engine_mlx.last_row_logits, serve's own path), the first,
    full-prompt step included."""
    import mlx.core as mx
    from mlx_lm.models.cache import make_prompt_cache

    from .serving.engine_mlx import last_row_logits

    cache = make_prompt_cache(engine.model)
    out = list(ids)
    step = ids
    for _ in range(n_new):
        nxt = mx.argmax(last_row_logits(engine.model, step, cache))
        mx.eval(nxt)
        tok = int(nxt)
        out.append(tok)
        step = [tok]
    return out


def timed_decode(engine, ids: list[int], n_new: int = N_NEW, reps: int = DECODE_REPS):
    """reps timed greedy decodes (identical greedy path -> identical tokens);
    returns (tokens, tok_s samples). Prefill is inside the timing, as it is
    in arms.timed_decode; at 64 new tokens over a short prompt it is noise.
    WARMUP_REP (probe.py's, shared with arms.timed_decode) runs one untimed
    decode first — mlx-lm's lazy graph builds its kernels and the allocator
    grows on the first call, the same reason as the torch side's warm-up;
    its tokens are discarded, never entering `samples`."""
    samples = []
    out = None
    if WARMUP_REP:
        greedy(engine, ids, n_new)  # untimed: graph build, first-touch, allocator growth
    for _ in range(reps):
        t0 = time.perf_counter()
        out = greedy(engine, ids, n_new)
        samples.append(round(n_new / (time.perf_counter() - t0), 2))
    return out, samples


def build_prefill_ids(tok, length: int = PREFILL_LEN) -> list[int]:
    """A deterministic `length`-token prompt: arms.py's build_prefill_ids,
    restated on this side's tokenizer call (`tok.encode`, not the HF
    `__call__` convention)."""
    ids = tok.encode(PREFILL_PASSAGE)
    if not ids:
        raise ValueError("prefill passage tokenized to zero ids")
    while len(ids) < length:
        ids = ids + ids
    return ids[:length]


def timed_prefill(engine, ids: list[int], reps: int = DECODE_REPS) -> list[float]:
    """`reps` timed forward passes over the full `ids` sequence, no cache,
    the output head over the last position only — engine_mlx.last_row_logits,
    the path serve runs and the counterpart of arms.timed_prefill's
    logits_to_keep=1. Returns tok/s samples over
    len(ids) tokens per call. WARMUP_REP (probe.py's, shared with
    arms.timed_prefill) times one extra rep first and discards it, the same
    gate and the same discard-by-index as the torch side."""
    import mlx.core as mx

    from .serving.engine_mlx import last_row_logits

    n = len(ids)
    samples = []
    for i in range(reps + int(WARMUP_REP)):
        t0 = time.perf_counter()
        mx.eval(last_row_logits(engine.model, ids))
        if WARMUP_REP and i == 0:
            continue  # the untimed rep
        samples.append(round(n / (time.perf_counter() - t0), 2))
    return samples


def timed_ttft(engine, ids: list[int], reps: int = DECODE_REPS) -> list[float]:
    """`reps` timings of wall time from call to the first generated token on
    the SAME prompt as timed_prefill: one prefill forward plus one greedy
    decode step (greedy()'s own shape, n_new=1, a fresh cache each rep).
    Returns second samples (a latency, not a rate). WARMUP_REP (probe.py's,
    shared with arms.timed_ttft) times one extra rep first and discards it,
    the same gate and the same discard-by-index as the torch side. The head
    is over the last position only (engine_mlx.last_row_logits)."""
    import mlx.core as mx
    from mlx_lm.models.cache import make_prompt_cache

    from .serving.engine_mlx import last_row_logits

    samples = []
    for i in range(reps + int(WARMUP_REP)):
        cache = make_prompt_cache(engine.model)
        t0 = time.perf_counter()
        nxt = mx.argmax(last_row_logits(engine.model, ids, cache))
        mx.eval(nxt)
        if WARMUP_REP and i == 0:
            continue  # the untimed rep
        samples.append(round(time.perf_counter() - t0, 4))
    return samples


def _stats(samples: list[float]) -> dict:
    lo, hi = min(samples), max(samples)
    return {"median": round(float(np.median(samples)), 2), "min": round(lo, 2),
            "max": round(hi, 2), "drift_pct": round(100 * (hi - lo) / lo, 1) if lo else None}


def refuse_unless_verified(model_id: str, swapped: int) -> None:
    """THE GATE, this engine's version of arms.refuse_unless_verified:
    load_compressed_mlx already verifies the pack's hashes at load time, so
    by the time this runs the
    compressed engine either served exactly the pack's verified bytes or
    never loaded. `swapped == 0` is the one case left to catch here — nothing
    eligible went through that verification at all, which is not a valid
    compressed point. Raises a named SystemExit (nonzero, printed) rather
    than returning; pure Python, no mlx import, so it is callable from tests
    without Metal or even mlx installed."""
    if swapped:
        return
    raise exitcodes.Refused(
        f"drinkme: refusing to write the record: {model_id} — the compressed "
        "MLX engine swapped zero Linears to a packed module, so no verified "
        "tensor was served compressed; bench refuses rather than publish a "
        "record with no compressed tensor in it.")


def run_arms(model_id: str, revision: str | None, prompt: str,
             pack_dir: str | None = None, fused: bool | None = None,
             n_new: int = N_NEW, reps: int = DECODE_REPS,
             prefill_len: int = PREFILL_LEN) -> dict:
    """The full three-pass measurement on the MLX engine. Returns arms.run_arms's
    field names where the quantity exists here, plus the min/drift table.

    `fused=None` = whatever the machine has (mx.metal.is_available()); False is
    the dry run — every arm on the reference path, which is the only thing a
    machine without Metal can do, and the report's `compressed_path` says so.
    `prefill_len` defaults to arms.py's fixed 512; it is a parameter
    only so a toy model with a small max_position_embeddings (tests) can
    override it — production callers should leave it at the default so the
    number is comparable across machines."""
    import mlx.core as mx

    from .codec.pack import bpw_of_pack_dir, default_pack_dir, pack_record_fields
    from .probe import load1, record_host_load
    from .serving import checkpoint
    from .serving.engine_mlx import TWIN, load_compressed_mlx, load_stock_mlx

    if fused is None:
        fused = mx.metal.is_available()
    if pack_dir is None:
        pack_dir = default_pack_dir(model_id, revision)
        if revision is None and not os.path.exists(os.path.join(pack_dir, "meta.json")):
            # an unpinned repo's pack is keyed by its commit (cli.pin_revision)
            sha = checkpoint.resolve_commit(model_id)
            pack_dir = default_pack_dir(model_id, sha) if sha else pack_dir
    # ONE SOURCE, FIRST (arms.run_arms' rule): the pack's bound snapshot,
    # resolved before the tokenizer or any arm loads, handed to every
    # loader as `snap` — stock and compressed cannot read two commits.
    snap, resolved = checkpoint.resolve_source(model_id, revision, pack_dir)
    tok = checkpoint.tokenizer(snap, None)
    ids = tok.encode(prompt)
    prefill_ids = build_prefill_ids(tok, prefill_len)
    report: dict = {"model": model_id, "revision": revision, "resolved_revision": resolved,
                    "gpu": _device(), "runtime": "mlx",
                    "compressed_path": "fused" if fused else "reference",
                    "prefill_prompt_len": len(prefill_ids)}
    report["warmup_rep"] = WARMUP_REP  # one untimed rep per timed shape; False = DRINKME_BENCH_WARMUP=0
    report["bandwidth"] = measure_bandwidth()  # BEFORE any load: empty device

    # --- pass 1: stock bf16 (mlx-lm's own model) ---
    eng = load_stock_mlx(model_id, revision, snap=snap)
    # vram_* on this lane is the ENGINE'S resident-bytes accounting (weights
    # the engine holds, packed or raw — MLXEngine.resident_bytes), not an
    # allocator's view; bytes here, decimal GB at the record boundary.
    report["vram_bf16_bytes"] = int(eng.resident_bytes)
    report["stock_outcome"] = "measured"  # this lane always runs stock (no fit check, no twin skip)
    report["bandwidth"]["total_param_bytes_bf16"] = int(eng.resident_bytes)
    report["bandwidth"]["ceiling_tok_s_bf16"] = round(
        report["bandwidth"]["read_bytes_s"] / eng.resident_bytes, 2)
    load0 = load1()
    _, s_stock = timed_decode(eng, ids, n_new, reps)
    report["stock_decode_tok_s"] = _stats(s_stock)["median"]
    report["stock_decode_samples"] = s_stock
    report["stock_decode_stats"] = _stats(s_stock)
    s_stock_pre = timed_prefill(eng, prefill_ids, reps)
    report["stock_prefill_tok_s"] = _stats(s_stock_pre)["median"]
    report["stock_prefill_samples"] = s_stock_pre
    s_stock_ttft = timed_ttft(eng, prefill_ids, reps)
    report["stock_ttft_s"] = round(float(np.median(s_stock_ttft)), 4)
    report["stock_ttft_samples"] = s_stock_ttft
    record_host_load(report, "stock", load0)
    del eng
    _return_cache(mx)  # the arm's weights out of MLX's cache before the next fit check and load

    # --- pass 2: the twin (the pack decoded once at load, the reference matmul) ---
    # Decode only: on this lane "twin" is a different animal from the
    # torch twin — the reference path's arithmetic over weights decoded
    # once (engine_mlx.RadixTwinLinear), a correctness check (does the pack
    # decode to bf16 the fused kernel can be trusted against) rather than a
    # comparable kernel-quality number. Its decode still runs (that IS the
    # check); bench.py never publishes its tok/s, and there is no
    # prefill/TTFT timing to publish either — same reasoning as the torch
    # twin (arms.py), restated: this path is bench's own correctness
    # exercise, not a prefill kernel.
    eng = load_compressed_mlx(model_id, revision, pack_dir, path=TWIN, snap=snap)
    load0 = load1()
    _, s_twin = timed_decode(eng, ids, n_new, reps)
    record_host_load(report, "twin", load0)
    report["twin_decode_tok_s"] = _stats(s_twin)["median"]
    report["twin_decode_samples"] = s_twin
    report["twin_decode_stats"] = _stats(s_twin)
    report["twin_outcome"] = "measured"
    # bits per weight off the loaded artifact's own per-tensor scalars
    # (pack.bpw_of_pack_dir): the tensor mean (meanBpw's statistic) and the
    # weight-weighted figure the record publishes as bitsPerWeight
    mean_bpw, w_bpw = bpw_of_pack_dir(pack_dir)
    report["mean_bpw"], report["weighted_bpw"] = round(mean_bpw, 3), round(w_bpw, 3)
    del eng
    _return_cache(mx)

    # --- pass 3: resident-compressed, the fused kernel ---
    eng = load_compressed_mlx(model_id, revision, pack_dir,
                              path="fused" if fused else "reference", snap=snap)
    report["vram_compressed_bytes"] = int(eng.resident_bytes)
    # compression.profile off the artifact this arm actually loaded — the
    # pack writer's own `profile` in meta.json, which is what the manifest
    # digest binds; never a literal in bench.py.
    if not eng.meta.get("profile"):
        raise exitcodes.Refused(f"drinkme: refusing to write the record: {pack_dir}/meta.json carries no "
                         "`profile`, so the record cannot name what it measured; re-pack "
                         "(`drinkme pack --replace`).")
    report["compression_profile"] = eng.meta["profile"]
    # raw.pack, the same subset arms.run_arms copies off a --pack-dir pack:
    # the record's compression.packedBytes / residentBytes come from here
    with open(os.path.join(pack_dir, "meta.json")) as f:
        report["pack"] = pack_record_fields(json.load(f))
    load0 = load1()
    _, s_comp = timed_decode(eng, ids, n_new, reps)
    report["compressed_decode_tok_s"] = _stats(s_comp)["median"]
    report["compressed_decode_samples"] = s_comp
    report["compressed_decode_stats"] = _stats(s_comp)
    swapped = sum(1 for _ in _packed_modules(eng.model))
    refuse_unless_verified(model_id, swapped)
    report["swapped_linears"] = swapped
    report["verified_tensors"] = swapped
    report["verification"] = "pack"  # every MLX arm is a pack off the disk: its hashes matched at load
    s_comp_pre = timed_prefill(eng, prefill_ids, reps)
    report["compressed_prefill_tok_s"] = _stats(s_comp_pre)["median"]
    report["compressed_prefill_samples"] = s_comp_pre
    s_comp_ttft = timed_ttft(eng, prefill_ids, reps)
    report["compressed_ttft_s"] = round(float(np.median(s_comp_ttft)), 4)
    report["compressed_ttft_samples"] = s_comp_ttft
    record_host_load(report, "compressed", load0)
    del eng
    _return_cache(mx)

    # the headline on the MINIMA (least-throttled samples), beside the medians
    report["ratio_min_of_n"] = round(_stats(s_comp)["min"] / _stats(s_stock)["min"], 3)
    report["drift_band_pct"] = max(_stats(s)["drift_pct"] or 0 for s in (s_stock, s_twin, s_comp))
    return report


def _packed_modules(model):
    from .serving.engine_mlx import RadixLinear, RawLinear

    packed = (RadixLinear, RawLinear)
    for layer in model.model.layers:
        for sub in (layer.self_attn, layer.mlp):
            for name in ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"):
                m = getattr(sub, name, None)
                if isinstance(m, packed):
                    yield m
    if isinstance(getattr(model, "lm_head", None), packed):
        yield model.lm_head
