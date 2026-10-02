"""ONE fit primitive: resident + kv, headroom-multiplied, against a
budget — the arithmetic `suggest()`/`serve_pick()` (bench's ratio/fit points)
and `packs.rank_fits` (the serve picker) each apply to their own menu.

Bytes end to end. Each caller converts its own native unit to bytes at its
own boundary — suggest.py's decimal-GB `Sizes.bf16_gb`/`comp_gb` via `GB`,
packs.py's GiB pack measurements via its own `GIB` (already defined there;
not re-exported from here to avoid two names for one constant) — and GiB/GB
reappear only where a caller formats a line for a human. No torch, no
transformers: this is pure arithmetic (and the vm_stat read a Mac's two budgets share),
importable from either runtime.
"""

from __future__ import annotations

import re
import subprocess

GIB = 1024 ** 3
GB = 1_000_000_000  # decimal GB — suggest.py's Sizes.bf16_gb/comp_gb unit


def fits(resident_bytes: float, kv_bytes: float, budget_bytes: float,
        headroom: float) -> bool:
    """(resident + kv) * headroom <= budget, in bytes throughout."""
    return (resident_bytes + kv_bytes) * headroom <= budget_bytes


def served_resident_bytes(meta: dict, vision: bool) -> int | None:
    """What a pack holds resident once served, from its meta.json:
    `residentBytes`, which charges a served vision tower like any other
    resident tensor (packed Linears at their runtime size, the rest raw),
    less the tower's own share (the `vision` block's residentBytes, ~0.65
    GB on Qwen3.8-27B) when the server builds no tower (`vision` False:
    DRINKME_VISION=0). None when the pack records no residentBytes."""
    total = meta.get("residentBytes")
    if total is None:
        return None
    block = meta.get("vision")
    if not vision and isinstance(block, dict):
        total -= int(block.get("residentBytes") or 0)
    return int(total)


# ------------------------------------------------------ the stock-arm fit check --
#
# suggest()'s ratio/fit split and packs.rank_fits both charge a model's FINAL
# resident footprint — correct for the compressed arm (it is the only copy
# that ever exists) but not for a stock pass through transformers' own
# loader, whose transient is larger than the model.
#
# On a DISCRETE card, CPU RAM and VRAM are separate physical pools: a host
# copy is freed once the VRAM copy exists, and only the VRAM copy ever
# competes with the budget being checked — so the "resident + kv" charge is
# correct there (RX 7600 XT 16 GB: Qwen3-8B's stock arm 15.26 GB * 1.15 =
# 17.55 GB > 16 GB, predicted not to fit, and it does not).
#
# On UNIFIED memory, CPU RAM and the GPU's GTT window are the SAME physical
# pool, and transformers' from_pretrained holds ~2x the model in host
# memory while it loads WHETHER OR NOT it is asked to load straight to the
# device (the file pages get pinned for the GTT upload; they are not
# reclaimable while it runs): MEASURED on a Strix Halo
# (gfx1151, 124 GB unified, ROCm), `from_pretrained(device_map="cuda",
# low_cpu_mem_usage=True)` on gemma-4-31B-it (58.25 GB bf16) took
# MemAvailable from 107 GB to 5 GB within ~30 s of "Loading weights" — a
# watchdog killed it before the kernel could, and unguarded the same load
# drove a GLOBAL kernel OOM (not a graceful CUDA/HIP one) that also took
# three browser tabs and an unrelated process: 2 * 58.25 GB * RATIO_HEADROOM
# (1.15) = 133.98 GB against the 124 GB physical ceiling. That 2x is the
# from_pretrained path's cost and is selectable here as `direct=False` — what
# `arms.load_cpu(device_map="cuda")` charges when an operator picks it
# (`bench --stock-loader from_pretrained`).
#
# The STREAMING stock loader (arms.load_stock_streaming, the default)
# is `direct=True`: the module tree is built empty on the meta device and
# each tensor is read from its shard, moved to the device and released
# before the next is read — the same walker `drinkme serve` streams its raw
# remainder with (serving/engines.stream_checkpoint). Its peak host charge
# beyond the resident model is ONE tensor, so the term is resident + the
# largest tensor in the checkpoint (`tensor_bytes`, read off the safetensors
# headers by serving.checkpoint.largest_tensor_bytes — cheap, zero weight
# bytes) or, when the snapshot is not on disk yet to read, STAGING_SHARD_BYTES:
# HF's own max shard size, an upper bound on any tensor that lives inside a
# shard (the suite's largest is gemma's 2.82 GB embedding). `direct` must
# only be True when the streaming loader is the one that will run — the
# default is the conservative 2x, so an unqualified call can never approve
# the load that OOM'd the machine. The Strix Halo's numbers under each charge:
# gemma 2 * 58.25 * 1.15 = 134 GB against a 124 GiB = 133 GB pool (OOM);
# streaming (58.25 + 2.82) * 1.15 = 70.2 GB (fits, a 3-arm ratio point);
# the 72B 145.4 GB is over the pool before any transient (a fit point either way).
STAGING_SHARD_BYTES = 5 * 1024**3  # HF's default max_shard_size is 5GB

# The two stock-arm loaders, by name — the selector for `direct` below.
# Spelled here (torch-free) because bench.py (no torch on the --dry-run
# path) and arms.py (torch) both need the one spelling; arms.run_arms
# dispatches pass 1 on it and bench's `--stock-loader` flag picks it.
STOCK_LOADER_STREAM = "stream"                    # arms.load_stock_streaming: direct=True
STOCK_LOADER_FROM_PRETRAINED = "from_pretrained"  # arms.load_cpu(device_map="cuda"): direct=False
STOCK_LOADERS = (STOCK_LOADER_STREAM, STOCK_LOADER_FROM_PRETRAINED)


def streamed_transient_bytes(resident_bytes: float, kv_bytes: float, memory_kind: str,
                             tensor_bytes: float | None = None) -> float:
    """The peak a STREAMED arm charges against the budget: its resident
    footprint plus, on unified memory, ONE tensor — the host copy of the
    tensor being read off its shard before it lands on the device
    (`tensor_bytes`, the checkpoint's largest, else STAGING_SHARD_BYTES).
    On discrete it is the resident footprint alone: the host-side tensor
    lives in a pool VRAM never sees.

    All three of the bench's bf16 arms stream (arms.load_stock_streaming,
    load_compressed_streaming, load_twin_streaming), so this
    is the one term each arm's fit check and live guard charge:
      stock       resident = the bf16 weights
      twin        resident = the bf16 weights (the order-matched twin
                  holds them as they are) — one copy, never two
      compressed  resident = the pack's resident bytes (suggest's
                  Sizes.comp_gb). The packer's host scratch over ONE tensor
                  is ~10x that tensor (MEASURED on Strix Halo:
                  +4.73 GB VmHWM packing a 0.484 GB 72B down_proj-shaped
                  tensor, 9.8x, linear across shapes), and an untied
                  lm_head is itself eligible (the 72B's: 2.49 GB -> ~27 GB
                  of scratch). The streaming loader swaps LARGEST FIRST
                  (arms._stream_swapped) so that scratch lands while the
                  resident set is smallest, and the raw remainder — the
                  embedding, the checkpoint's largest tensor — streams
                  last with everything resident: one tensor is then the
                  peak beyond resident, and this term is honest for it.
                  (72B on Strix Halo, largest first: ~27 GB at the start,
                  ~60 + 5.3 GB through the MLP tensors, 102.7 + 2.49 GB
                  at the end.)
    Before this the compressed and twin arms were built by materialising
    the whole bf16 checkpoint on the host (arms.load_cpu) and swapping
    Linears on that tree — a charge no fit arithmetic here ever named, and
    the reason the 72B fit point (145 GB bf16, 102.71 GB compressed) could
    not be re-taken by `drinkme bench` on a 124 GiB machine."""
    staged = resident_bytes
    if memory_kind == "unified":
        staged += tensor_bytes if tensor_bytes else STAGING_SHARD_BYTES
    return staged + kv_bytes


def streamed_arm_fits(resident_bytes: float, kv_bytes: float, ceiling_bytes: float,
                      memory_kind: str, headroom: float,
                      tensor_bytes: float | None = None) -> bool:
    """Would a streamed arm's load fit `ceiling_bytes` with `headroom`
    (streamed_transient_bytes x headroom <= ceiling)? The twin is judged
    with the stock arm's RATIO_HEADROOM (it holds the same bf16 bytes),
    the compressed arm with suggest's FIT_HEADROOM."""
    return fits(streamed_transient_bytes(resident_bytes, kv_bytes, memory_kind, tensor_bytes),
                0.0, ceiling_bytes, headroom)


def stock_transient_bytes(resident_bytes: float, kv_bytes: float, memory_kind: str,
                          direct: bool = False, tensor_bytes: float | None = None) -> float:
    """The real peak a stock-arm load charges against the budget — not just
    the final resident footprint. `memory_kind` is `Hardware.memory_kind`
    ("unified" | "vram"). On unified memory: 2x resident for the
    from_pretrained loader (`direct=False`, the default; measured, see
    above), resident + one tensor for the streaming loader (`direct=True`;
    streamed_transient_bytes — `tensor_bytes` = the checkpoint's largest
    tensor, else STAGING_SHARD_BYTES). On discrete it is resident (VRAM)
    under either loader — the host side never touches the VRAM pool."""
    if direct:
        return streamed_transient_bytes(resident_bytes, kv_bytes, memory_kind, tensor_bytes)
    if memory_kind != "unified":
        staged = resident_bytes
    else:
        staged = 2 * resident_bytes
    return staged + kv_bytes


def stock_arm_fits(resident_bytes: float, kv_bytes: float, ceiling_bytes: float,
                   memory_kind: str, headroom: float, direct: bool = False,
                   tensor_bytes: float | None = None) -> bool:
    """Would a stock-arm load fit `ceiling_bytes`, honoring the real
    transient of the loader that will run (`direct`/`tensor_bytes`, see
    stock_transient_bytes)? `ceiling_bytes` is the caller's choice of
    ceiling — physical memory on unified (the OOM killer's own limit), VRAM
    on discrete (a pool the host side never touches, so `Hardware.budget_gb`
    is already the right number there)."""
    return fits(stock_transient_bytes(resident_bytes, kv_bytes, memory_kind, direct, tensor_bytes),
                0.0, ceiling_bytes, headroom)


# The four vm_stat counters parse_vm_stat adds up, by vm_stat's own names.
VM_STAT_AVAILABLE = ("free", "inactive", "speculative", "purgeable")


def parse_vm_stat(text: str) -> int | None:
    """Bytes macOS would hand a process now, from `vm_stat`'s output
    (`memory_pressure` prints the same counters under the same names): the
    free, inactive, speculative and purgeable pages times the page size the
    header states. None when the page size or any of the four is missing.

    Free pages are unused; speculative ones are read-ahead file pages; the
    purgeable ones belong to volatile objects discarded on demand; inactive
    pages are the ones macOS reclaims next — file-backed ones dropped,
    anonymous ones compressed — before it swaps. So this is what can be
    taken without swapping, not without touching anyone's pages: an M4
    that showed ~12 GiB of free + inactive before a request let the server
    reach ~12 GB by compressing its neighbour's inactive pages, with no
    swap growth. Purgeable pages also sit in the active and inactive
    queues, so a purgeable inactive page counts twice; on the M4's recorded
    outputs purgeable is 0.2-4.5% of the sum."""
    page = re.search(r"page size of (\d+)", text)
    counts = dict(re.findall(r"^Pages (free|inactive|speculative|purgeable):\s+(\d+)", text, re.M))
    if page is None or set(counts) != set(VM_STAT_AVAILABLE):
        return None
    return int(page.group(1)) * sum(int(v) for v in counts.values())


def host_available_bytes() -> int | None:
    """parse_vm_stat of `vm_stat` (macOS; no sudo): what macOS would hand a
    process now. None when vm_stat is missing, fails, times out or prints
    what parse_vm_stat cannot read. The caller decides whether the host is a
    Mac."""
    try:
        out = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=10,
                             check=True).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    return parse_vm_stat(out)
