"""`drinkme serve` with no `--model`: what is already packed on this machine, and
what fits it — the discovery + KV-aware fit arithmetic behind the no-args
picker (cli.py's `serve` branch).

UNITS: this module's arithmetic is GiB-native (`GIB`, `GB_TO_GIB`,
`kv_gib_per_token`, `resident_gib`). Decimal-GB figures are converted at
their edge: suggest.Sizes.comp_gb (a menu row's estimate) through
`GB_TO_GIB`, and the box's budget in `hardware_budget()` from
`detect.Hardware.budget_bytes` (`detect`'s `budget_gb` is bytes / 1e9,
~7.4% bigger than the same bytes' GiB count, and is never charged as GiB here —
`test_packs.py::test_detect_budget_is_treated_as_gib_not_gb`).

No torch, no transformers, no mlx at import time: this module runs ahead of
`build_engine`, on every runtime, including MLX on Metal, whose code never
imports torch (pyproject's `metal` group carries it only to pack, and a metal
environment made before that has none). `serving.checkpoint` is torch-free for
exactly this reason (its own docstring) and is reused here for
`resolve_ctx`; `serving.engines.slots_from_env` and `serving.mtp.head_pack_gib`
both import torch at module scope, so their shapes are copied rather than
imported — `slots_estimate` and `_mtp_head_gib` say so at each call site.
"""

from __future__ import annotations

import glob
import json
import os
import sys
from dataclasses import dataclass

from . import runtimes
from .runtimes import model_type_of
from .fit import fits as _fits
from .fit import served_resident_bytes

GIB = 1024**3
GB_TO_GIB = 1e9 / GIB  # decimal GB (suggest.Sizes) -> GiB, at the edge


@dataclass(frozen=True)
class LocalPack:
    hf_repo: str
    revision: str | None
    pack_dir: str
    packed_bytes: int  # sum of the trunk's *.npz on disk (stat only, no read)
    canonical: bool  # pack_dir == this identity's default_pack_dir
    menu: object | None  # the matching suggest.Model, if this identity is curated
    # meta.json's own residentBytes (npz + the raw-streamed tensors the
    # loader also holds — embed_tokens when untied, norms, anything below the
    # eligibility bar); packed_bytes when a pack has no such key (a
    # tool-written one — `drinkme pack` always measures it), with
    # resident_bytes_estimated=True so a caller never mistakes the fallback
    # for a measurement.
    resident_bytes: int = 0
    resident_bytes_estimated: bool = True


def packs_root(root: str | None = None) -> str:
    """`$DRINKME_HOME/packs` — same default as codec.pack.default_pack_dir,
    reimplemented (not imported) so a caller's explicit `root` and the
    canonical-dir check below always agree on which root they mean, even
    when `root` differs from the environment (as a test's tmp_path does)."""
    base = root or os.environ.get("DRINKME_HOME", os.path.expanduser("~/.cache/drinkme"))
    return os.path.join(base, "packs")


def _default_pack_dir(root: str, hf_repo: str, revision: str | None) -> str:
    """codec.pack.default_pack_dir's exact formula, against `root` (this
    scan's own root) rather than re-reading DRINKME_HOME from the
    environment — see packs_root's docstring."""
    slug = hf_repo.replace("/", "--")
    return os.path.join(root, "packs", f"{slug}@{(revision or 'main')[:12]}")


def local_packs(root: str | None = None) -> list[LocalPack]:
    """Scan `$DRINKME_HOME/packs/*/meta.json`. Cheap: stat the *.npz sizes,
    never open one. Identity is meta.json's hfRepo/revision, never the
    directory name — a `pack --gulp` directory (`...-gulp@main`) has
    real hfRepo/revision but a name that does not match its own
    default_pack_dir, so `canonical` comes back False and callers must not
    offer it as the named menu model (servable only via --pack-dir).

    Unreadable or malformed meta.json: skipped with one stderr line, never
    raised — a scan must not be able to crash the pick. A pack of a format
    version this build does not read is skipped the same way, with the
    loader's refusal line. An empty directory (no meta.json at all, e.g.
    a `pack` that died before its commit point) is simply not matched by
    the glob."""
    base = root or os.environ.get("DRINKME_HOME", os.path.expanduser("~/.cache/drinkme"))
    proot = os.path.join(base, "packs")
    from .codec.pack import refusal_for_meta
    from .serving.vision import enabled_from_env
    from .suggest import MODELS

    by_key = {(m.hf_repo, m.revision): m for m in MODELS}
    vision_on = enabled_from_env()  # DRINKME_VISION=0: no tower resident
    out: list[LocalPack] = []
    for meta_path in sorted(glob.glob(os.path.join(proot, "*", "meta.json"))):
        pack_dir = os.path.dirname(meta_path)
        try:
            with open(meta_path) as f:
                meta = json.load(f)
            hf_repo = meta["hfRepo"]
        except Exception as e:  # noqa: BLE001 — a bad meta.json must not kill the pick
            print(f"[drinkme] packs: skipping unreadable {meta_path} "
                  f"({type(e).__name__}: {e})", file=sys.stderr, flush=True)
            continue
        refusal = refusal_for_meta(meta, pack_dir)
        if refusal is not None:
            # a pack of a format version this build does not read, or one
            # naming a profile it does not know: not offered on the menu —
            # the loader would refuse it by name — and said once, with the fix
            print(f"[drinkme] packs: not offering — {refusal}", file=sys.stderr, flush=True)
            continue
        revision = meta.get("revision")
        packed_bytes = sum(os.path.getsize(f)
                           for f in glob.glob(os.path.join(pack_dir, "*.npz")))
        canonical = pack_dir == _default_pack_dir(base, hf_repo, revision)
        # the vision tower's share left out when images are off (fit.py)
        resident_bytes = served_resident_bytes(meta, vision_on)
        resident_estimated = resident_bytes is None
        if resident_bytes is None:
            resident_bytes = packed_bytes  # no residentBytes key: npz sum, marked estimated
        out.append(LocalPack(hf_repo, revision, pack_dir, packed_bytes, canonical,
                             by_key.get((hf_repo, revision)), resident_bytes,
                             resident_estimated))
    return out


def _local_config(hf_repo: str, revision: str | None, pack_dir: str) -> dict | None:
    """config.json for a locally-packed model: a self-contained pack's own
    (`<pack>/checkpoint/config.json`, docs/pack-format.md), else the HF hub
    cache, local-only — never a network hit from the picker. None when
    neither has it; the caller charges weights-only and says so."""
    from .codec.pack import embedded_dir

    p = os.path.join(embedded_dir(pack_dir), "config.json")
    if os.path.isfile(p):
        try:
            with open(p) as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError):
            pass
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return None
    kw = {"revision": revision} if revision else {}
    cached = try_to_load_from_cache(hf_repo, "config.json", **kw)
    if not isinstance(cached, str):
        return None
    try:
        with open(cached) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def _mtp_head_gib(pack_dir: str) -> float | None:
    """GiB the packed MTP head will occupy at serve time — serving.mtp.
    head_pack_gib's exact shape, copied rather than imported: that module
    imports torch at the top, and this one must stay importable with none
    (module docstring). None when there is no head sub-pack, or the diet
    is off (DRINKME_MTP_DIET=0 serves the raw bf16 head instead — a
    residency charge this estimator does not attempt)."""
    if os.environ.get("DRINKME_MTP_DIET") == "0":
        return None
    from .codec.pack import mtp_pack_dir, read_mtp_meta

    meta = read_mtp_meta(pack_dir)
    if meta is None:
        return None
    resident = meta.get("residentBytes")
    if resident is None:
        hp = mtp_pack_dir(pack_dir)
        resident = sum(os.path.getsize(f) for f in glob.glob(os.path.join(hp, "*.npz")))
    return (resident + meta.get("unpackedBytes", 0)) / GIB


@dataclass(frozen=True)
class Candidate:
    name: str  # menu name when curated, else the bare hf_repo
    hf_repo: str
    revision: str | None
    resident_gib: float  # weights only: real residentBytes, npz-sum estimate, or suggest.estimate
    packed: bool  # True = a canonical local pack backs this number
    pack_dir: str | None
    measured: bool  # True = resident_gib is a real measurement, not an estimate
    config: dict | None  # None -> KV cannot be charged, and the fit says so
    mtp_head_gib: float | None
    # True = `packed` but the pack carries no residentBytes key, so
    # resident_gib fell back to the npz sum — a real measurement of packed
    # bytes, but an ESTIMATE of what will actually be resident (misses the
    # raw-streamed tensors). format_summary's line must say so.
    resident_estimated: bool = False
    # config.json's top-level model_type — the menu row's, or the local
    # config's — what runtimes.refusal decides runtime support from; None
    # when neither knows the family.
    model_type: str | None = None


def build_candidates(local: list[LocalPack], gemma_ok: bool = False) -> list[Candidate]:
    """The union suggest.serve_pick never had to reason about: locally-packed
    canonical identities (real bytes, real config when cached) plus the menu
    (estimated bytes, no KV — weights-only until it is actually packed).
    Gated models (gemma) are excluded unless the caller already knows the
    gate is open, exactly as serve_pick refuses to probe it — including when
    gemma is ALREADY packed locally: a canonical pack on disk is not consent
    to serve it by surprise, so it is filtered out of the fallback loop below
    by the same `gated_repos` test, not just out of the curated menu loop.
    A row not marked suggest.Model.auto_eligible is left out the same
    two ways: never in the menu loop, and a local pack of it is not a
    candidate for the no-arg pick either — `--model` names it.

    An unpacked row is charged suggest.estimate: its checkpoint's bytes x
    the profile's ratio, tower included. A row whose checkpoint cannot be
    sized (not cached, and the Hub did not answer) is not a candidate, and
    one stderr line names it: it could not be downloaded either."""
    from .suggest import MODELS, estimates

    gated_repos = {m.hf_repo for m in MODELS if m.gated}
    not_auto_repos = {m.hf_repo for m in MODELS if not m.auto_eligible}
    menu = [m for m in MODELS if m.auto_eligible and (not m.gated or gemma_ok)]
    canonical_by_key = {(p.hf_repo, p.revision): p for p in local if p.canonical}

    unpacked = [m for m in menu if (m.hf_repo, m.revision) not in canonical_by_key]
    est_by_name = dict(zip((m.name for m in unpacked), estimates(unpacked)))

    out: list[Candidate] = []
    seen: set[tuple[str, str | None]] = set()
    unsized = []
    for m in menu:
        key = (m.hf_repo, m.revision)
        p = canonical_by_key.get(key)
        if p is not None:
            cfg = _local_config(m.hf_repo, m.revision, p.pack_dir)
            out.append(Candidate(m.name, m.hf_repo, m.revision, p.resident_bytes / GIB,
                                 True, p.pack_dir, True, cfg, _mtp_head_gib(p.pack_dir),
                                 p.resident_bytes_estimated,
                                 model_type=m.model_type or model_type_of(cfg)))
        else:
            est = est_by_name[m.name]
            if est is None:
                unsized.append(m.name)
                continue
            out.append(Candidate(m.name, m.hf_repo, m.revision, est.comp_gb * GB_TO_GIB,
                                 False, None, False, None, None, model_type=m.model_type))
        seen.add(key)
    if unsized:
        print(f"[drinkme] packs: not offering {', '.join(unsized)} — the checkpoint's size could "
              "not be read (not in the local Hugging Face cache, and the Hub did not answer)",
              file=sys.stderr, flush=True)
    for p in local:
        key = (p.hf_repo, p.revision)
        if not p.canonical or key in seen or (p.hf_repo in gated_repos and not gemma_ok):
            continue
        if p.hf_repo in not_auto_repos:
            continue
        cfg = _local_config(p.hf_repo, p.revision, p.pack_dir)
        out.append(Candidate(p.hf_repo, p.hf_repo, p.revision, p.resident_bytes / GIB,
                             True, p.pack_dir, True, cfg, _mtp_head_gib(p.pack_dir),
                             p.resident_bytes_estimated, model_type=model_type_of(cfg)))
        seen.add(key)
    return out


def filter_for_runtime(candidates: list[Candidate],
                       runtime: str) -> tuple[list[Candidate], list[tuple[Candidate, str]]]:
    """The candidates `runtime` can serve, and (candidate, reason) for each
    one it cannot — runtimes.refusal, the one runtime-capability predicate,
    applied BEFORE the fit ranking so the no-model pick never proposes a
    family the runtime's loader would refuse after the download and the
    pack (for example Qwen3.8-27B to a 48 GiB Mac). The torch
    runtime excludes nothing here."""
    kept, excluded = [], []
    for c in candidates:
        reason = runtimes.refusal(runtime, c.model_type)
        (kept if reason is None else excluded).append(c if reason is None else (c, reason))
    return kept, excluded


def kv_gib_per_token(config: dict) -> float:
    """K and V, bf16, per token — engine_mlx.kv_bytes_per_token's shape
    (module docstring: 'borrow this shape'), in GiB, and extended for hybrid
    DeltaNet configs: when `layer_types` distinguishes attention layers, only
    `full_attention` layers hold a KV cache (DeltaNet's linear-attention
    layers carry their recurrent state, not a growing KV); otherwise every
    layer is charged, the dense formula's honest upper bound. `text_config`
    is unwrapped first — Qwen3.8-27B's config.json nests the language-model
    fields there (a `Qwen3_5ForConditionalGeneration` composite config),
    exactly what `cfg.get_text_config()` does for the live transformers
    objects elsewhere in this tree; here the config is a plain dict read off
    disk, so it's unwrapped by hand."""
    tc = config.get("text_config", config)
    head_dim = int(tc.get("head_dim") or tc["hidden_size"] // tc["num_attention_heads"])
    layer_types = tc.get("layer_types")
    n_layers = (sum(1 for t in layer_types if t == "full_attention") if layer_types
               else int(tc["num_hidden_layers"]))
    return (2 * n_layers * int(tc["num_key_value_heads"]) * head_dim * 2) / GIB


def ctx_estimate(explicit: int | None) -> int:
    """The ctx the no-model KV charge assumes: serving.checkpoint.resolve_ctx
    with an unknown native window (0 — no model is picked yet, so no
    max_position_embeddings to clamp against), which reduces exactly to
    --ctx or DRINKME_CTX if given, else CTX_CAP. The real load, once a model
    IS picked, resolves ctx again against that model's actual native window
    (engines.py) — this is the estimate the confirm prompt shows, not a
    second source of truth."""
    from .serving.checkpoint import resolve_ctx

    return resolve_ctx(0, explicit)


def slots_estimate(explicit: int | None) -> int:
    """serving.engines.slots_from_env's precedence (CLI wins over
    DRINKME_PREFIX_SLOTS; 0 (the prefix cache off) still allocates one
    per-request cache, and garbage or < 1 counts as 1), reimplemented
    rather than imported: that module imports torch at the top (module
    docstring). The real load still goes through slots_from_env and prints
    its own warning on garbage input; this is a silent estimate only."""
    if explicit is not None:
        return explicit if explicit >= 1 else 1
    raw = os.environ.get("DRINKME_PREFIX_SLOTS", "").strip()
    if not raw:
        return 1
    try:
        n = int(raw)
    except ValueError:
        n = 0
    return n if n >= 1 else 1


def checkpoints_estimate(explicit: int | None) -> int:
    """The context checkpoints per slot the no-model charge assumes:
    serving.ctx_checkpoints.max_from_env's precedence (--ctx-checkpoints over
    DRINKME_CTX_CHECKPOINTS over llama.cpp's 32), silent on garbage like
    slots_estimate. That module is torch-free, so it is read, not copied."""
    from .serving import ctx_checkpoints

    if explicit is not None:
        return explicit if explicit >= 0 else ctx_checkpoints.DEFAULT_MAX
    raw = os.environ.get(ctx_checkpoints.ENV, "").strip()
    try:
        n = int(raw) if raw else ctx_checkpoints.DEFAULT_MAX
    except ValueError:
        n = ctx_checkpoints.DEFAULT_MAX
    return n if n >= 0 else ctx_checkpoints.DEFAULT_MAX


def checkpoint_gib(config: dict, ctx: int, slots: int, n_max: int) -> float:
    """What a model's context checkpoints can hold at most, in GiB: the
    count the rules allow at this ctx (ctx_checkpoints.max_held) times one
    checkpoint's bytes (ctx_checkpoints.config_bytes: bf16 rings and conv
    states, float32 recurrent states), per slot. 0 for a model that takes
    none (full attention only)."""
    from .serving import ctx_checkpoints

    one = ctx_checkpoints.config_bytes(config, 2, ctx)
    return one * ctx_checkpoints.max_held(n_max, ctx) * slots / GIB


def hardware_budget() -> tuple[float | None, str]:
    """The usable working-set budget in GiB, and a one-line description of
    where it came from. Darwin prefers the smaller of the MLX working-set
    ceiling and what macOS would hand a process now (fit.host_available_bytes),
    the two numbers engine_mlx.py's fit_check takes the min of, over detect's
    sysctl x 0.75 guess, when mlx actually imports; the description names
    the one that binds. Never torch.cuda.mem_get_info on a Mac (mtp.py:
    returns None on mps). On Linux, the smaller of detect()'s capacity (GTT
    on an APU, VRAM on a discrete card) and what is free of it right now
    (detect.live_free_bytes), so a box already running a model is not
    offered a model sized for an empty one."""
    import platform

    from .detect import detect

    hw = detect()
    if hw.budget_gb is None:
        return None, "could not detect a memory budget on this machine"
    # detect's budget_gb is decimal GB (bytes / 1e9); this module charges GiB,
    # so the budget enters as its bytes (every detector sets budget_bytes
    # beside budget_gb; a hand-built Hardware without it converts the GB).
    budget_bytes = hw.budget_bytes if hw.budget_bytes is not None else round(hw.budget_gb * 1e9)
    budget_gib = budget_bytes / GIB
    if platform.system() == "Darwin":
        try:
            import mlx.core as mx

            if mx.metal.is_available():
                ws = mx.device_info().get("max_recommended_working_set_size")
                if ws:
                    from .fit import host_available_bytes

                    host = host_available_bytes()
                    if host is None:
                        return ws / GIB, (f"mlx working set {ws / GIB:.1f} GiB "
                                          "(host available unread)")
                    bound = "host available" if host < ws else "mlx working set"
                    return min(ws, host) / GIB, (
                        f"{bound} {min(ws, host) / GIB:.1f} GiB, the smaller of mlx working "
                        f"set {ws / GIB:.1f} GiB and host available {host / GIB:.1f} GiB")
        except Exception:  # noqa: BLE001 — no mlx, or no Metal device: fall through
            pass
        return budget_gib, f"sysctl hw.memsize x 0.75 = {budget_gib:.1f} GiB"
    # Linux: the capacity detect() read, or what is free of it right now when
    # that is less (another model, a desktop, a browser); the same idea as the
    # Mac branch above and as bench's live load guard (arms.live_available_bytes).
    from .detect import live_free_bytes

    if hw.memory_kind == "unified":
        capacity = "GTT" if hw.device_source == "cpu-brand" else "unified memory"
    elif hw.memory_kind == "vram":
        capacity = "VRAM"
    else:
        return budget_gib, f"{budget_gib:.1f} GiB"
    free, what = live_free_bytes(hw)
    if free is None:
        return budget_gib, f"{capacity} {budget_gib:.1f} GiB ({what})"
    if free >= budget_bytes:
        return budget_gib, f"{capacity} {budget_gib:.1f} GiB ({what} {free / GIB:.1f} GiB)"
    return free / GIB, f"{what} {free / GIB:.1f} GiB, of {capacity} {budget_gib:.1f} GiB"


def graph_platform(runtime: str | None) -> str | None:
    """"cuda" when the serve about to be picked for runs on an NVIDIA card
    through torch — the only place CUDA graphs run (serving/cudagraph.py) —
    else None: the picker then charges no graph memory. Read off detect()'s
    evidence (nvidia-smi answered), torch-free."""
    if runtime == "mlx":
        return None
    from .detect import detect

    hw = detect()
    return "cuda" if any(e.startswith("nvidia-smi:") for e in hw.evidence) else None


@dataclass(frozen=True)
class Fit:
    candidate: Candidate
    kv_gib: float
    total_gib: float  # resident + kv + checkpoints + mtp head, NOT headroom-multiplied
    charged_gib: float  # total_gib * FIT_HEADROOM — what the budget check uses
    note: str | None  # e.g. "KV not counted — config not cached"
    fits: bool
    ckpt_gib: float = 0.0  # the context checkpoints' most (checkpoint_gib)
    graph_gib: float = 0.0  # CUDA graphs, per slot (serving/cudagraph_fit.py)


def rank_fits(budget_gib: float, candidates: list[Candidate], ctx: int,
             slots: int, ckpts: int | None = None, platform: str | None = None) -> list[Fit]:
    """Per candidate: (resident + kv_per_token(config) x ctx x slots +
    checkpoint_gib(config, ctx, slots, ckpts) + mtp_head_gib + CUDA graphs)
    x FIT_HEADROOM <= budget_gib — suggest.FIT_HEADROOM, the one number three call sites
    already share; this is not a second headroom. `ckpts` is the context
    checkpoints per slot (checkpoints_estimate; None = llama.cpp's 32). The
    graphs are serving/cudagraph_fit.charge_bytes per slot, charged only
    when `platform` is "cuda" (graph_platform) and the family graphs. The
    per-slot FIXED state cost (serving/checkpoint.py's own module docstring)
    is measurable only post-load, so it is left uncounted here, same as the
    per-token charge omits it.

    Sorted once, by total resident GiB descending with already-packed
    preferred on a tie — 'largest that fits comfortably' ranks by what serve
    will actually put in memory, not by menu curation order. Filtering this
    single list by `.fits` afterwards preserves that order in both the
    fitting and the non-fitting halves, so no separate sort is needed for
    either."""
    from .serving import ctx_checkpoints, cudagraph_fit
    from .suggest import FIT_HEADROOM

    n_ckpt = ctx_checkpoints.DEFAULT_MAX if ckpts is None else ckpts
    scored = []
    for c in candidates:
        if c.config is not None:
            kv_gib = kv_gib_per_token(c.config) * ctx * slots
            ckpt_gib = checkpoint_gib(c.config, ctx, slots, n_ckpt)
            note = None
        else:
            kv_gib = ckpt_gib = 0.0
            note = "KV not counted — config not cached"
        graph_gib = cudagraph_fit.charge_bytes(c.config, platform, slots) / GIB
        total = c.resident_gib + kv_gib + ckpt_gib + (c.mtp_head_gib or 0.0) + graph_gib
        charged = total * FIT_HEADROOM
        # the fit verdict itself goes through the one shared primitive
        # (drinkme.fit) — GiB converted to bytes at this edge, same as
        # suggest.py converts its own decimal-GB numbers at its edge.
        kv_and_head_gib = kv_gib + ckpt_gib + (c.mtp_head_gib or 0.0) + graph_gib
        ok = _fits(c.resident_gib * GIB, kv_and_head_gib * GIB, budget_gib * GIB,
                  FIT_HEADROOM)
        scored.append(Fit(c, kv_gib, total, charged, note, ok, ckpt_gib, graph_gib))
    return sorted(scored, key=lambda f: (-f.total_gib, 0 if f.candidate.packed else 1))


def format_summary(budget_gib: float, source: str, ranked: list[Fit],
                   local: list[LocalPack], root: str, ctx: int, slots: int,
                   runtime: str | None = None,
                   excluded: list[tuple[Candidate, str]] | None = None) -> list[str]:
    """The stderr lines cli.py prints ahead of the confirm prompt — pure, so
    the arithmetic and the wording are each testable without capturing
    stdio. No 'drinkme serve: ' prefix here; the caller adds it per line,
    matching every other diagnostic in cli.py. `excluded` names every
    candidate filter_for_runtime dropped, with its reason's first clause —
    a model that fits but cannot be served here is said, not hidden."""
    from .suggest import FIT_HEADROOM

    lines = [f"no --model given — budget: {source}" + (f", runtime: {runtime}" if runtime else ""),
            f"assuming ctx {ctx}, {slots} prefix slot(s) ({FIT_HEADROOM}x headroom)"]
    for c, reason in excluded or ():
        lines.append(f"not offered on the {runtime} runtime: {c.name} — {reason.split('. ')[0]}")
    fits = [f for f in ranked if f.fits]
    non_fits = [f for f in ranked if not f.fits]
    if fits:
        lines.append("candidates (ranked, largest resident first):")
        for i, f in enumerate(fits):
            marker = "->" if i == 0 else "  "
            c = f.candidate
            state = "packed" if c.packed else "will download+pack"
            # a pack without residentBytes reports the npz sum under
            # that name, which is not what it means — say so, every
            # time, rather than let an estimate pass as a measurement;
            # and an unpacked row's size is the checkpoint x the ratio
            estimate_note = (" (estimated — repack to measure)" if c.resident_estimated
                             else "" if c.measured else " (estimate)")
            kv_part = f"({f.note})" if f.note else f"+ KV {f.kv_gib:.2f} GiB @ ctx {ctx}"
            if f.ckpt_gib:
                kv_part += f" + checkpoints {f.ckpt_gib:.2f} GiB"
            if f.graph_gib:
                kv_part += f" + CUDA graphs {f.graph_gib:.2f} GiB"
            lines.append(f"  {marker} {c.name}: {c.resident_gib:.2f} GiB{estimate_note} "
                        f"{kv_part} = {f.total_gib:.2f} GiB ({state})")
        if non_fits:
            lines.append(f"{len(non_fits)} more don't fit at this ctx")
    if local:
        n_canon = sum(1 for p in local if p.canonical)
        n_exp = len(local) - n_canon
        extra = (f" ({n_canon} canonical, {n_exp} experiment — not offered by "
                f"name; use --pack-dir)" if n_exp else "")
        lines.append(f"{len(local)} pack(s) found under {root}{extra}")
    return lines
