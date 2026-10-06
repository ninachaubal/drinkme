"""The model menu — capability-filtered, Qwen3 spine + gemma when runnable.

Rule per detected budget M (docs/models.md):

  RATIO model — largest with BF16_size x 1.15 <= M. Both arms run; the point
      lands on the curve.
  FIT model — the next size up (not a ratio candidate) whose compressed size
      fits: clean when comp x 1.1 <= M; KNIFE-EDGE when comp <= M without the
      headroom (attempt, warned); HONEST-NEGATIVE CANDIDATE when comp exceeds
      the default budget but not physical unified memory (the Mac-8GB 4B cliff
      — raiseable via iogpu.wired_limit_mb, a documented power-user flag, off
      by default).

A menu row carries no sizes; they are read when a pick needs them (`sizes`).
BF16_size is the checkpoint's own safetensors byte count
(`checkpoint_bytes`). The compressed size is the pack's own `residentBytes`
once a pack of the row exists on this machine, and before that an estimate,
BF16_size x COMP_RATIO[profile], which says it is one wherever it is shown.
A row whose checkpoint cannot be sized (not cached here, and the Hub did not
answer) is left out of the pick, with a note; it could not be downloaded
either.

gemma-4-31B-it joins the menu only when its HF gate is open for this user
(HEAD probe; account + one license click). It NEVER blocks the Qwen path.
"""

from __future__ import annotations

import json
import os
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from .fit import GB, fits

# What a pack holds resident (its meta.json `residentBytes`) as a fraction of
# the checkpoint's safetensors bytes: the median per profile over the Strix
# Halo records of these models (site/data/points.json carries the same
# per-model ratios), compression.residentBytes / the checkpoint's bytes at
# the record's revision.
#   sip, 10 records: Qwen3.8-27B 0.7063, Qwen2.5-72B 0.7106, Qwen3-32B 0.7147,
#     gemma-4-31B-it 0.7179, Muse-Glimmer-30B 0.7204, Qwen3-8B 0.7329,
#     Qwen3-4B 0.7359, MiMo-V2.6-Distill-Qwen-9B 0.7362, Qwen3-1.7B 0.7947,
#     Qwen3-0.6B 0.8286
#   gulp, 10 records: Qwen3.8-27B 0.6803, Qwen2.5-72B 0.6812, Qwen3-32B 0.6881,
#     gemma-4-31B-it 0.6909, Muse-Glimmer-30B 0.6925, Qwen3-8B 0.7066,
#     MiMo-V2.6-Distill-Qwen-9B 0.7115, Qwen3-4B 0.7126, Qwen3-1.7B 0.7761,
#     Qwen3-0.6B 0.8133
# The vision tower stays in both numbers: the four vision models' ratios sit
# among the text-only models'. The two smallest models run above the median,
# so their estimate is low by up to a tenth; FIT_HEADROOM covers that, and a
# pack's own size replaces the estimate once the pack exists.
COMP_RATIO = {"sip": 0.727, "gulp": 0.700}
RATIO_HEADROOM = 1.15  # stock arm needs activations + runtime on top of weights
FIT_HEADROOM = 1.10  # compressed arm headroom
HUB_TIMEOUT_S = 5.0  # one Hub metadata request (checkpoint_bytes)


def comp_ratio(profile: str) -> float:
    """COMP_RATIO for `profile`. balanced (DRINKME_COMPRESSION_PROFILE's
    hidden door) has no record; its widths lie between gulp's and sip's, so
    it is charged sip's ratio, the larger."""
    return COMP_RATIO.get(profile, COMP_RATIO["sip"])


@dataclass(frozen=True)
class Model:
    name: str
    hf_repo: str
    revision: str | None  # pinned; None = pin at first validated bench
    gated: bool = False  # HF license gate (gemma)
    # May the automatic pick choose this row? suggest() and serve_pick()
    # (and packs.build_candidates, the no-arg serve picker) consider only
    # rows marked auto_eligible. Every row is on the menu either way —
    # `bench --model` / `serve --model` resolve it by name. Opt-in, default
    # False, so a new row joins the automatic pick only when marked. Left
    # off: coverage rows (Muse-Glimmer-30B) that would otherwise take a
    # showcase seat just by being a little larger.
    auto_eligible: bool = False
    # config.json's top-level model_type (the wrapper's for a composite),
    # read off the pinned checkpoint — what runtimes.refusal decides runtime
    # support from, so the no-model picker can filter the menu with no
    # config on disk (a row without one is treated as an unknown family).
    model_type: str | None = None


@dataclass(frozen=True)
class Sizes:
    """What a menu row weighs, read when a pick needs it (`sizes`)."""
    bf16_gb: float  # the checkpoint's safetensors bytes, decimal GB
    comp_gb: float  # what the compressed model holds resident, decimal GB
    profile: str  # the compression profile comp_gb is for
    pack_dir: str | None = None  # the pack comp_gb was read off; None: the estimate

    @property
    def estimated(self) -> bool:
        """comp_gb is BF16 x COMP_RATIO, not a pack's own residentBytes."""
        return self.pack_dir is None

    def comp_label(self) -> str:
        """comp_gb as a line shows it: an estimate says so and how it was made."""
        if self.estimated:
            return (f"{self.comp_gb:.2f} GB (estimate: {self.bf16_gb:.2f} GB checkpoint x "
                    f"{comp_ratio(self.profile)}, the {self.profile} ratio)")
        return f"{self.comp_gb:.2f} GB (the {self.profile} pack's own size)"


_CHECKPOINT_BYTES: dict = {}


def checkpoint_bytes(hf_repo: str, revision: str | None) -> int | None:
    """The checkpoint's safetensors bytes (every tensor's dtype x shape,
    summed: a BF16 checkpoint's BF16 size), read from its metadata, never its
    weights. Local first: the shard headers of a snapshot this machine holds
    (serving.checkpoint.checkpoint_tensor_bytes), or the shard index's
    total_size when only the index is cached. Else one Hub metadata request
    for that revision: its safetensors parameter counts by dtype, asked
    anonymously (no token), HUB_TIMEOUT_S. None when neither answers.
    Remembered per process once known."""
    key = (hf_repo, revision)
    if key not in _CHECKPOINT_BYTES:
        n = _local_checkpoint_bytes(hf_repo, revision)
        if n is None:
            n = _hub_checkpoint_bytes(hf_repo, revision)
        if n is None:
            return None
        _CHECKPOINT_BYTES[key] = n
    return _CHECKPOINT_BYTES[key]


def _local_checkpoint_bytes(hf_repo: str, revision: str | None) -> int | None:
    from .serving.checkpoint import cached_snapshot_dir, checkpoint_tensor_bytes

    try:
        snap = cached_snapshot_dir(hf_repo, revision)
    except Exception:  # noqa: BLE001 — an unreadable cache is a miss
        snap = None
    if snap:
        return checkpoint_tensor_bytes(snap)
    try:
        from huggingface_hub import try_to_load_from_cache

        kw = {"revision": revision} if revision else {}
        index = try_to_load_from_cache(hf_repo, "model.safetensors.index.json", **kw)
        if not isinstance(index, str):
            return None
        with open(index) as f:
            total = (json.load(f).get("metadata") or {}).get("total_size")
        return int(float(total)) if total else None
    except Exception:  # noqa: BLE001 — no cache, or an index without a total
        return None


def _hub_checkpoint_bytes(hf_repo: str, revision: str | None) -> int | None:
    from .check import _dtype_nbytes

    try:
        from huggingface_hub import HfApi

        info = HfApi().model_info(hf_repo, revision=revision, expand=["safetensors"],
                                  timeout=HUB_TIMEOUT_S, token=False)
    except Exception:  # noqa: BLE001 — offline, unknown repo: not sized
        return None
    params = getattr(info.safetensors, "parameters", None) if info.safetensors else None
    if not params:
        return None
    return sum(int(n) * _dtype_nbytes(dtype) for dtype, n in params.items())


def estimate(m: Model, profile: str | None = None) -> Sizes | None:
    """A row's sizes before a pack of it exists: the checkpoint's bytes, and
    those bytes x comp_ratio(profile) as the compressed size, marked an
    estimate. `profile` defaults to the one a pack would be written at
    (radix_pack.resolve_compression_profile: DRINKME_COMPRESSION_PROFILE,
    else sip). The whole checkpoint counts, vision tower included, so under
    DRINKME_VISION=0 the estimate errs large by the tower's share. None
    when the checkpoint cannot be sized."""
    from .codec.radix_pack import resolve_compression_profile

    profile = resolve_compression_profile(profile)
    n = checkpoint_bytes(m.hf_repo, m.revision)
    if n is None:
        return None
    return Sizes(n / GB, n * comp_ratio(profile) / GB, profile)


def sizes(m: Model, profile: str | None = None, vision: bool = True,
          pack_dir: str | None = None) -> Sizes | None:
    """A row's sizes: the checkpoint's bytes, and as the compressed size the
    pack's own residentBytes when a pack of it exists (`pack_dir`, else the
    profile's default pack directory, codec.pack.default_pack_dir), less
    the vision tower's share when `vision` is False (fit.
    served_resident_bytes); else `estimate`. None when the checkpoint
    cannot be sized."""
    from .fit import served_resident_bytes

    est = estimate(m, profile)
    if est is None:
        return None
    meta, where = _pack_meta(m, est.profile, pack_dir)
    resident = served_resident_bytes(meta, vision) if meta is not None else None
    if resident is None:
        return est
    return Sizes(est.bf16_gb, resident / GB, est.profile, where)


def _pack_meta(m: Model, profile: str, pack_dir: str | None) -> tuple[dict | None, str | None]:
    """(meta.json, its directory) of the pack a row's compressed size is read
    off: `pack_dir` as given, or the default pack directory of this row, its
    revision (a row pinned to no revision: the commit the local HF cache has
    for main) and `profile`, when that holds a pack of this repo at this
    profile that this build reads. (None, None) otherwise."""
    from .codec.pack import default_pack_dir, refusal_for_meta
    from .codec.radix_pack import DEFAULT_PROFILE

    d = pack_dir
    if d is None:
        rev = m.revision
        if rev is None:
            from .serving.checkpoint import _cached_commit

            rev = _cached_commit(m.hf_repo, "main")
        d = default_pack_dir(m.hf_repo, rev, profile)
    try:
        with open(os.path.join(d, "meta.json")) as f:
            meta = json.load(f)
    except (OSError, ValueError):
        return None, None
    if not isinstance(meta, dict) or refusal_for_meta(meta, d) is not None:
        return None, None
    if pack_dir is None and (meta.get("hfRepo") != m.hf_repo
                             or (meta.get("profile") or DEFAULT_PROFILE) != profile):
        return None, None
    return meta, d


def _each(fn, menu: list[Model]) -> list:
    """fn over the rows, in threads: a row not cached here costs a Hub request."""
    with ThreadPoolExecutor(max_workers=max(1, min(8, len(menu)))) as ex:
        return list(ex.map(fn, menu))


def estimates(menu: list[Model]) -> list[Sizes | None]:
    """`estimate` for each row, the Hub requests in parallel."""
    return _each(estimate, menu)


def size_menu(menu: list[Model], vision: bool = True) -> tuple[list[tuple[Model, Sizes]], list[Model]]:
    """`sizes` for each row, the Hub requests in parallel: ([(row, its
    sizes)], [the rows that could not be sized])."""
    found = _each(lambda m: sizes(m, vision=vision), menu)
    return ([(m, sz) for m, sz in zip(menu, found) if sz is not None],
            [m for m, sz in zip(menu, found) if sz is None])


def unsized_note(rows: list[Model]) -> str:
    return (f"not considered: {', '.join(m.name for m in rows)} — the checkpoint's size could "
            "not be read (not in the local Hugging Face cache, and the Hub did not answer)")


# The spine. Revisions pinned from the validated runs; None means no
# validated run has touched this size yet.
#
# ORDER (the showcase lineup): the first three are
# PRESENTATION, not selection — Qwen3.8-27B is the main showcase, Qwen3-8B
# and gemma beside it.
# This is the order a reader browsing MODELS or
# an unknown-model error message (cli.resolve_model, bench.run) sees first.
# suggest()/serve_pick() never read list position — both filter and re-sort
# by each row's sizes (`sizes`) — so reordering here is pure curation with
# zero effect on which model a budget actually picks.
MODELS = [
    # The showcase model: hybrid DeltaNet 27B, validated on a Strix Halo
    # (Ryzen AI Max+ 395) — the stock A/B, tools, think and MTP.
    Model("Qwen3.8-27B", "Qwen/Qwen3.8-27B", "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0", auto_eligible=True, model_type="qwen3_5"),
    Model("Qwen3-8B", "Qwen/Qwen3-8B", "b968826d9c46dd6066d109eabc6255188de91218", auto_eligible=True, model_type="qwen3"),
    Model("gemma-4-31B-it", "google/gemma-4-31B-it", "842da3794eaa0b77d5f08bae87a17459d91ff475", gated=True, auto_eligible=True, model_type="gemma4"),
    # Coverage tier: Meta's Apache-2.0 composite VLM, whose
    # text skeleton is the wrapper itself (arms._VISION_TOWERS). Not
    # auto_eligible: see Model.auto_eligible.
    Model("Muse-Glimmer-30B", "meta-models/Muse-Glimmer-30B", "a4e59da52a7bc87ae7251dd5545c0dd437c44b68", model_type="muse_glimmer"),
    # Fit-point candidate for a 16 GB card (doesn't fit uncompressed, fits
    # compressed): IBM's
    # granite-4.2-8b (2026-08-07, Apache-2.0, GraniteForCausalLM, 8.79B, untied
    # 100k head). Not benchmarked.
    Model("granite-4.2-8b", "ibm-granite/granite-4.2-8b", "f8de16cdcdbc6c779ca517604e050d82cc119e44", model_type="granite"),
    # Coverage tier: Xiaomi's MiMo-V2.6-Distill-Qwen-9B, an SFT
    # distill on the Qwen3.5 architecture (hybrid DeltaNet, MIT); the server
    # reads its images. config.json says
    # mtp_num_hidden_layers: 1 but the checkpoint ships no mtp.* tensors, so
    # AUTO speculation is the n-gram lookup. Tools: its chat template renders
    # the tools with no call format, but writes an assistant tool call back as
    # qwen-xml, and capability's history probe reads that: tool_format
    # "qwen-xml", and a get_weather request on Strix Halo came back
    # as a parsed tool_calls, finish_reason "tool_calls". Not auto_eligible.
    Model("MiMo-V2.6-Distill-Qwen-9B", "XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B", "2367e865d009c13ac81713a2878291d33ab28177", model_type="qwen3_5"),
    # The rest of the spine, by size.
    Model("Qwen3-0.6B", "Qwen/Qwen3-0.6B", None, auto_eligible=True, model_type="qwen3"),
    Model("Qwen3-1.7B", "Qwen/Qwen3-1.7B", "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e", auto_eligible=True, model_type="qwen3"),
    Model("Qwen3-4B", "Qwen/Qwen3-4B", "1cfa9a7208912126459214e8b04321603b3df60c", auto_eligible=True, model_type="qwen3"),
    Model("Qwen3-14B", "Qwen/Qwen3-14B", None, auto_eligible=True, model_type="qwen3"),
    Model("Qwen3-32B", "Qwen/Qwen3-32B", None, auto_eligible=True, model_type="qwen3"),
    Model("Qwen2.5-72B", "Qwen/Qwen2.5-72B-Instruct", "495f39366efef23836d0cfae4fbe635880d2be31", auto_eligible=True, model_type="qwen2"),
]


@dataclass
class Suggestion:
    ratio: Model | None  # both arms; the curve point
    ratio_extra: list[Model]  # additional ratio-capable big models (gemma on 128GB)
    fit: Model | None  # the "does not run stock, runs compressed" point
    fit_knife_edge: bool  # comp fits, headroom does not — attempt with a warning
    fit_honest_negative: bool  # comp exceeds the default budget; the cliff experiment
    notes: list[str]
    sizes: dict[str, Sizes] = field(default_factory=dict)  # by row name, every row the pick sized


def gemma_gate_open(timeout: float = 5.0) -> bool:
    """HEAD-probe the gemma gate. Any failure -> False, never an exception —
    the gate must not be able to block the Qwen path (or the whole bench)."""
    url = "https://huggingface.co/google/gemma-4-31B-it/resolve/main/config.json"
    token = os.environ.get("HF_TOKEN")
    if not token:
        try:
            from huggingface_hub import get_token

            token = get_token()
        except Exception:
            token = None
    try:
        req = urllib.request.Request(url, method="HEAD")
        if token:
            req.add_header("Authorization", f"Bearer {token}")
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status == 200
    except Exception:
        return False


def graph_bytes(m: Model) -> int:
    """What the compressed arm's CUDA graphs hold for a menu row, on a CUDA
    card (serving/cudagraph_fit.charge_bytes, one cache): read off the
    row's config.json from a pack of it or the local Hugging Face cache
    (never the network); 0 when neither has it or the family does not graph."""
    from . import packs
    from .serving import cudagraph_fit

    pack_dir = packs._default_pack_dir(packs.packs_root(), m.hf_repo, m.revision)
    return cudagraph_fit.charge_bytes(packs._local_config(m.hf_repo, m.revision, pack_dir), "cuda", 1)


def suggest(
    budget_gb: float,
    memory_gb: float | None = None,
    memory_kind: str | None = None,
    gemma_ok: bool = False,
    graphs: bool = False,
) -> Suggestion:
    """Pick the menu for a budget. The gate probe result is an input; each
    row's sizes are read (`size_menu`), with bench's view of the tower: its
    arms load it (arms._TOWER_BY_DEFAULT), so a pack is charged whole.
    `graphs` (an NVIDIA card on the torch runtime): the fit point's
    compressed charge includes its CUDA graphs (`graph_bytes`)."""
    menu = [m for m in MODELS if m.auto_eligible and (not m.gated or gemma_ok)]
    sized, unsized = size_menu(menu, vision=True)
    by = {m.name: sz for m, sz in sized}
    notes: list[str] = [unsized_note(unsized)] if unsized else []

    # fits() takes bytes throughout (drinkme.fit); budget_gb and the sizes
    # are both decimal GB here, so the GB conversion below cancels — it
    # exists to route this module's arithmetic through the one primitive
    # the serve picker (packs.rank_fits) also uses, not to change a result.
    ratio_all = [m for m, sz in sized if fits(sz.bf16_gb * GB, 0.0, budget_gb * GB, RATIO_HEADROOM)]
    ratio = max(ratio_all, key=lambda m: by[m.name].bf16_gb) if ratio_all else None
    ratio_extra = [m for m in ratio_all if m is not ratio and m.gated]

    fit = None
    knife = False
    negative = False
    for m, sz in sorted(sized, key=lambda ms: ms[1].comp_gb):
        if m in ratio_all:
            continue  # already runs both arms; not a fit-limited point
        extra = graph_bytes(m) if graphs else 0
        if fits(sz.comp_gb * GB, extra, budget_gb * GB, FIT_HEADROOM):
            fit = m
            break
        if fits(sz.comp_gb * GB, extra, budget_gb * GB, 1.0):
            fit, knife = m, True
            notes.append(
                f"{m.name}: compressed fits ({_amount(sz)} into {budget_gb:.2f}GB) but "
                f"without {FIT_HEADROOM}x headroom — attempting; it may run out of memory"
            )
            break
        if (
            memory_kind == "unified"
            and memory_gb
            and sz.comp_gb <= memory_gb * 0.9
        ):
            fit, negative = m, True
            notes.append(
                f"{m.name}: compressed ({_amount(sz)}) exceeds the default working-set "
                f"budget ({budget_gb}GB) but not physical memory ({memory_gb}GB) — "
                "the fit-cliff experiment; on macOS iogpu.wired_limit_mb can raise "
                "the budget (documented power-user flag, off by default)"
            )
            break
        break  # everything above only gets bigger

    if fit is not None and by[fit.name].estimated and not (knife or negative):
        notes.append(f"{fit.name}: its compressed size is an estimate from its checkpoint until a "
                     "pack of it exists here")
    return Suggestion(ratio, ratio_extra, fit, knife, negative, notes, by)


def _amount(sz: Sizes) -> str:
    """A note's compressed size, marked when it is the estimate."""
    return f"{'an estimated ' if sz.estimated else ''}{sz.comp_gb:.2f}GB"


def serve_pick(budget_gb: float, gemma_ok: bool = False) -> tuple[Model | None, list[str]]:
    """`drinkme serve` with no --model: the largest model whose COMPRESSED
    size fits the budget with headroom. A different question from bench's
    ratio/fit split — serve doesn't care whether stock would also fit; it
    wants the most model this machine comfortably holds. Gated models are
    excluded unless the caller already probed the gate (serve never
    HEAD-probes with the user's token by surprise; bench owns that probe
    and its --no-gemma opt-out). Rows not marked auto_eligible never pick themselves."""
    from .serving.vision import enabled_from_env

    vision = enabled_from_env()  # a server without images holds no tower (a pack's `vision` block)
    menu = [m for m in MODELS if m.auto_eligible and (not m.gated or gemma_ok)]
    sized, unsized = size_menu(menu, vision=vision)
    notes = [unsized_note(unsized)] if unsized else []
    fitting = [(m, sz) for m, sz in sized
               if fits(sz.comp_gb * GB, 0.0, budget_gb * GB, FIT_HEADROOM)]
    if not fitting:
        if not sized:
            return None, [*notes, "no menu model could be sized — pass --model to override"]
        small, sz = min(sized, key=lambda ms: ms[1].comp_gb)
        return None, [
            f"nothing on the menu fits {budget_gb}GB with {FIT_HEADROOM}x headroom "
            f"(smallest is {small.name} at {sz.comp_label()} compressed) — pass "
            "--model to override", *notes
        ]
    pick, sz = max(fitting, key=lambda ms: ms[1].comp_gb)
    if sz.estimated:
        notes.append(f"{pick.name}: compressed size {sz.comp_label()}")
    return pick, notes


if __name__ == "__main__":
    for m in MODELS:
        sz = sizes(m)
        print(f"{m.name:28s} " + (f"bf16 {sz.bf16_gb:7.2f} GB  compressed {sz.comp_label()}"
                                  if sz else "not sized"))
    for budget in (8, 12, 16, 24, 48, 110):
        s = suggest(budget, gemma_ok=True)
        print(
            f"{budget:>4}GB  ratio={s.ratio.name if s.ratio else '—':12s}"
            f" fit={s.fit.name if s.fit else '—':16s}"
            f"{' KNIFE' if s.fit_knife_edge else ''}{' NEG' if s.fit_honest_negative else ''}"
        )
