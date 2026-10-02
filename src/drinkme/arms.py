"""The measurement arms: stock BF16 -> resident-compressed -> the order-matched twin.

The twin is the compressed arm's own modules and kernels over the
uncompressed weight (swap.RadixTwinLinear: the served radix kernels with
the decode replaced by a raw bf16 load, at the same launch schedule; a raw
fallback slot is the same RawLinear), so compressed / twin isolates the
weight read (docs/bench.md, "Torch benchmark scope"): the diagnostic we
tune against. Its decode is published as data; the results site divides
the compressed arm by stock.

Three ISOLATED passes (fresh model load each) so peak VRAM = one
representation. Every pass STREAMS: the module tree is built
empty on the meta device and materialised one checkpoint tensor at a time
straight onto the device — the stock arm raw (load_stock_streaming),
the twin and compressed arms with each eligible Linear swapped as its tensor
is read (load_twin_streaming / load_compressed_streaming) — so the host
never holds the whole bf16 model, and each arm's fit charge is its resident
bytes plus one tensor (fit.streamed_transient_bytes). That is what lets the
Qwen2.5-72B fit point (145 GB bf16, 102.71 GB compressed) be taken by
`drinkme bench` on a 122 GB machine.

Suite v1 protocol: decode timed 3x per arm (median + all samples reported),
64 new tokens greedy; prefill and TTFT timed 3x on the same FIXED
512-token prompt (build_prefill_ids, comparable across models regardless of
the tokenizer's take on PROMPT's dozen words) for the stock and compressed
arms ONLY — not the twin. The twin routes by M as the compressed arm does,
so at a 512-token prefill it runs the dense arm: stock's own F.linear over
its raw weight, a stock prefill, not a comparison. Its decode is timed and
published beside stock's as twin_decode_tok_s, with its footprint
(vram_twin_bytes, the record's twin_weights_gb): the same bf16 bytes
through the compressed arm's kernels, so compressed / twin is what reading
fewer bytes buys, decode included. Bandwidth is probed on the EMPTY device
before any load.

The per-tensor bitwise-roundtrip check (the radix encoder's own, run
inside make_compressed) is a GATE here, not a metric: a failure raises
before any decode timing happens, so a record is never written for a
compressed arm that didn't verify. See run_arms's pass 2 for where that
gate lives.

Every arm is ROUTED ALIKE before it is timed:
serving/kernel_route.route_kernels — the one call engines.load_stock and
load_compressed make — binds the DeltaNet decode recurrence kernel and
adopts the narrow Linears on each arm's tree, and its record rides in raw
as `<arm>_routing`. "Stock" is therefore drinkme's runtime over the raw
bf16 weights (the same recurrence kernel, the same narrow GEMV, no codec),
not transformers' eager path. A run whose arms routed differently writes
NO record (refuse_unless_routed_alike, the second gate): the 27B's stock
arm on the torch recurrence against a compressed arm on fla's fused kernel
was ~7% in the compressed arm's favour, and that is a runtime difference,
not a weight-read one.

No fidelity metric is measured or published: the invariant is bit-identical
weights, and greedy transcripts across arms can diverge at near-ties because
the kernels accumulate in different orders (docs/dev-environment.md, "Known behavior"). A
logits comparison taken at a prefill-shaped M would not measure the M=1
kernel anyway — at M >= CompressedLinear.GEMM_MIN_ROWS the compressed arm
decodes the verified weight and runs stock's own F.linear, so it is bitwise
stock by construction.
"""

from __future__ import annotations

import json
import os
import time

import numpy as np
import torch

from . import exitcodes
from .codec.pack import (UnsupportedCheckpoint, pack_record_fields, refuse_checkpoint,
                         source_dtype_refusal_reason, weighted_bpw)
from .codec.swap import make_compressed, make_twin
from .probe import WARMUP_REP, ceiling_tok_s, load1, measure_bandwidth, record_host_load
from .serving.kernel_route import describe as describe_route, route_kernels

DECODE_REPS = 3
N_NEW = 64
PREFILL_LEN = 512
# WARMUP_REP: probe.py's (torch-free, shared with arms_mlx.py — see that
# module's import of the same name for why it lives there and not here).
# The record's `samples` are the timed reps only; raw.warmup_rep says
# whether the untimed rep ran.

# A fixed passage (never the arbitrary user PROMPT below, whose token count
# varies by tokenizer) tokenized once and repeated to build a deterministic
# PREFILL_LEN-token prompt: same text, same length, every model, every run.
PREFILL_PASSAGE = (
    "The key idea of lossless weight compression is to store exactly the "
    "bits a model was trained with, using fewer bytes, so that decoding "
    "reproduces the original weights bit for bit. "
)


def vram_bytes() -> int:
    """The torch caching allocator's live allocation, in BYTES — the scope of
    every vram_* figure on the torch runtime (allocator usage, not the
    driver's view and not the process's). Bytes, not GiB: the decimal-GB
    conversion happens once, at the record boundary (bench._gb)."""
    return int(torch.cuda.memory_allocated())


def free_all() -> None:
    import gc

    gc.collect()
    torch.cuda.empty_cache()


def load_cpu(model_id: str, revision: str | None = None, config=None,
            device_map: str | None = None, snap: str | None = None,
            vision: bool | None = None):
    """config (serving/engines.py's rope-scaling path) hands from_pretrained
    an already-resolved, already-mutated config instance instead of letting it
    re-resolve config.json itself — transformers uses a PreTrainedConfig
    instance as-is (deepcopied, values intact) rather than re-fetching. None
    (every other caller) is the default behaviour exactly.

    `device_map`: pass "cuda" to load STRAIGHT to the device,
    shard-by-shard, `low_cpu_mem_usage=True` — the bench's stock arm under
    `--stock-loader from_pretrained` only; no other caller needs it (the
    bench's twin/compressed passes stream — load_twin_streaming /
    load_compressed_streaming — and never call this). Without it the name
    means what it says: a CPU-resident model the caller moves with its own
    `.cuda()`. NOTE that device_map="cuda" does NOT remove from_pretrained's
    ~2x host transient on unified memory, where CPU RAM and the GPU's GTT
    window are the same physical pool — MEASURED on
    a Strix Halo: gemma-4-31B-it (58.25 GB bf16) took MemAvailable 107 -> 5 GB
    in ~30 s under exactly this call, and the plain path drove a GLOBAL
    kernel OOM. The bench's stock arm therefore streams by default
    (load_stock_streaming) and charges one tensor of transient, not two
    models.

    THE DOOR: an FP8 (or otherwise quantized) checkpoint is refused by name
    here, from its config.json and shard headers, before transformers'
    own quantized-loading path gets a look at it (`drinkme serve --stock`
    on such a repo prints the one line and nothing else).

    `snap`: the already-resolved snapshot directory to load from (the
    bench's one source for every arm, serving.checkpoint.resolve_source);
    None resolves `model_id`@`revision` here, as every other caller does.

    `vision`: whether a vision tower the class builds and loads (gemma-4,
    Muse-Glimmer) stays on the tree, skeleton()'s tri-state: None is the
    bench's tree (_TOWER_BY_DEFAULT), True keeps it (`drinkme serve --stock`
    with images), False prunes it. A tower the class does not build
    (Qwen3.5's) is never attached here; engines.load_stock does that."""
    refuse_checkpoint(model_id, snap or snapshot_dir(model_id, revision))
    from transformers import AutoModelForCausalLM

    src = snap or model_id
    kw = {"revision": revision} if revision and snap is None else {}
    if config is not None:
        kw["config"] = config
    if device_map is not None:
        kw["device_map"] = device_map
        kw["low_cpu_mem_usage"] = True

    def _load(cls):
        try:
            try:
                m = cls.from_pretrained(src, dtype=torch.bfloat16, **kw).eval()
            except TypeError:  # pre-5.x arg name
                m = cls.from_pretrained(src, torch_dtype=torch.bfloat16, **kw).eval()
        except (ImportError, ValueError) as e:
            # transformers raises an ImportError from its quantizers and a
            # ValueError ("Using a `device_map` ... requires `accelerate`")
            # from the device_map path (the direct-to-device stock load) —
            # both mean the same missing package. MEASURED
            # on an RX 7600 XT 16 GB: the ValueError fell into _load's caller's
            # ValueError/KeyError multimodal retry and surfaced as
            # "Unrecognized configuration class Qwen3Config for
            # AutoModelForImageTextToText" — the wrong error for a missing
            # dependency. Name it here, before that retry can see it.
            if "accelerate" not in str(e).lower():
                raise
            # refuse by name instead, naming the fix (--stock's device_map
            # path needs accelerate; the compressed arm needs neither).
            # RuntimeError, not ValueError: _load's own caller catches
            # ValueError/KeyError to retry a ForConditionalGeneration class
            # for unified multimodal checkpoints, which this refusal must
            # not be swept into.
            raise RuntimeError(
                f"{model_id}: --stock needs the `accelerate` package to load "
                "this checkpoint (transformers' own quantized-loading path "
                f"requires it) — {e}. It ships in this project's cuda/rocm "
                "dependency groups; `uv sync` (or, for a one-off without "
                "resolving the whole lock, `uv pip install accelerate`) "
                "installs it. The compressed arm does not need it.") from e
        for p in m.parameters():
            p.requires_grad_(False)
        return m

    try:
        model = _load(AutoModelForCausalLM)
    except (ValueError, KeyError):
        # Unified multimodal checkpoints (Gemma 4, Muse Glimmer) are absent
        # from the causal-LM mapping; transformers routes them through
        # image-text-to-text. MEASURED on 5.15.0: muse_glimmer
        # appears ONLY in MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES, so ask
        # the auto class that owns the mapping.
        model = _load(_image_text_auto())
    model = _prune_non_text(model, model.config)
    return model if _wants_tower(model.config, vision) else drop_vision_tower(model, model.config)


# ---------------- composite (wrapped) checkpoints: text tower vs the rest ----
#
# ONE table, TWO consumers, because the halves must agree by construction:
#   * skeleton()/load_cpu() PRUNE these submodules, so the served tree never
#     holds a tower's parameters — and eligible_linears never names one, so the
#     pack never contains one;
#   * ckpt_to_skel() SKIPS the matching checkpoint keys, so no tensor is
#     offered a home the tree no longer has.
# Editing one side only is exactly the bug the shared table prevents: pruned
# but not skipped and the loader refuses ("checkpoint tensor has no module");
# skipped but not pruned and it refuses the other way ("meta parameters after
# streaming"). Both refusals are correct — this is how they stay unreachable.
# A vision tower that IS served is not here: it is _VISION_TOWERS' (below),
# under the same two-sided rule, with the build decided per call.
_NON_TEXT_TOWERS: dict[str, tuple[str, ...]] = {
    # Qwen3.8-27B: 333 vision tensors under model.visual.*
    # (served: _VISION_TOWERS), 15 MTP under mtp.*. AutoModelForCausalLM
    # already builds the text-only Qwen3_5ForCausalLM, which has no mtp
    # submodule — so the entry is skip-only and the prune is a measured
    # no-op.
    "qwen3_5": ("mtp",),
    # Both spellings: the WRAPPER config (qwen3_5, what Qwen3.8-27B ships) and
    # the TEXT config (qwen3_5_text) a text-only checkpoint of the same family
    # would carry. mtp.is_supported already accepts both, so a head-carrying
    # text-config checkpoint reached engines.load_compressed's get_submodule
    # with `mtp.fc.weight` and died on "has no module" (reproduced with
    # such a toy).
    "qwen3_5_text": ("model.visual", "mtp"),
}

# ---------------- the vision tower, served beside the text model ------------
#
# model_type -> where the ViT sits in the served tree: a SIBLING of the text
# model, at the path the checkpoint already names it by, so ckpt_to_skel
# passes its keys through unrenamed and the rest follows with no new code —
# eligible_linears names its Linears (the pack carries them compressed),
# stream_checkpoint attaches the rest, a self-contained pack embeds that
# rest raw, and sleep parks all of it like any other parameter. The text
# model's call shape is untouched: the tree stays Qwen3_5ForCausalLM, never
# the ForConditionalGeneration wrapper.
#
# Asked for per build (skeleton(cfg, vision=...)): True, the tower is in
# the tree (the packer always; the server unless DRINKME_VISION=0); False,
# it is not (the server otherwise, and DRINKME_VISION=0); None, the tree the
# bench's arms measure
# (_TOWER_BY_DEFAULT, below) — every architecture that serves a tower,
# so a measurement always describes what the served model actually
# carries. Whether the tower is there decides what ckpt_to_skel
# skips, the same two-sided rule as _NON_TEXT_TOWERS.
#
# A tower spans one subtree or several, all listed; the first is the ViT
# itself. Qwen3.5's text skeleton has no tower, so it is attached when
# asked. gemma-4's causal auto-mapping builds the whole
# Gemma4ForConditionalGeneration, and Muse-Glimmer's image-text-to-text
# class builds MuseGlimmerForConditionalGeneration (_CLASS_BUILDS_TOWER),
# so their towers are there already and are pruned when not asked for.
_VISION_TOWERS: dict[str, tuple[str, ...]] = {
    "qwen3_5": ("model.visual",),  # Qwen3_5VisionModel(cfg.vision_config)
    "gemma4": ("model.vision_tower", "model.embed_vision"),
    # Muse-Glimmer-30B (transformers 5.15.0). MEASURED off the
    # snapshot's 1436 shard keys: 626 text under model.language_model.*, 806
    # vision under model.vision_tower.*, 2 under model.vision_adapter.*, 1
    # model.vision_projection.weight, lm_head.weight at root.
    # There is NO text-only causal class to build — 5.15.0 ships
    # MuseGlimmerTextModel (headless) and MuseGlimmerForConditionalGeneration
    # (text + vision + head), and muse_glimmer is registered ONLY in the
    # image-text-to-text mapping. So the text skeleton IS the wrapper, and
    # the text keys need no renaming: the wrapper names them exactly as the
    # checkpoint does (model.language_model.*, lm_head). Keeping the wrapper
    # class also keeps its logit path — output_multiplier + tanh softcap live
    # in ForConditionalGeneration.forward, not in the text tower.
    # The wrapper's fourth stage, model.perception_emb_norm (a weightless
    # RMSNorm over the projection's output), holds no tensor and is never
    # pruned. The FOURTH registered config, muse_glimmer_assistant, is NOT a
    # tower of this checkpoint: it is a separate top-level arch (its own
    # MuseGlimmerAssistantConfig/-Model, Exaone4-derived) and ZERO shard keys
    # mention it.
    "muse_glimmer": (
        "model.vision_tower",       # 806 keys, 1.853B params — the ViT
        "model.vision_adapter",     # 2 keys — fc1/fc2 over the ViT output
        "model.vision_projection",  # 1 key — ViT hidden -> text hidden
    ),
}
_CLASS_BUILDS_TOWER = frozenset({"gemma4", "muse_glimmer"})
# The tree `vision=None` builds (the bench's arms) holds the tower for
# every architecture _VISION_TOWERS names: a served model carries its
# tower, so a bench measurement of it does too — weights GB, resident GB,
# bits per weight and the fit check all follow from the tree the arms
# actually build, so they agree with what `drinkme serve` holds and with
# docs/pack-format.md's meta.json `vision` block. DRINKME_VISION=0 (serving/vision.py) still turns
# the tower off for a server that does not want it; that is a serve-time
# choice, never a measurement default.
_TOWER_BY_DEFAULT = frozenset(_VISION_TOWERS)


def _wants_tower(cfg, vision: bool | None) -> bool:
    """`vision` as skeleton()/ckpt_to_skel()/load_cpu() were asked, None
    resolved to the architecture's default tree (_TOWER_BY_DEFAULT)."""
    if vision is None:
        return getattr(cfg, "model_type", None) in _TOWER_BY_DEFAULT
    return bool(vision)


def vision_tower_paths(cfg) -> tuple[str, ...] | None:
    """Where this checkpoint's vision tower sits in a served tree, the ViT
    first, or None: an architecture whose tower is not served, or a config
    with no vision config at all."""
    paths = _VISION_TOWERS.get(getattr(cfg, "model_type", None))
    return paths if paths and getattr(cfg, "vision_config", None) is not None else None


def vision_tower_path(cfg) -> str | None:
    """The ViT's own path (vision_tower_paths' first), or None."""
    paths = vision_tower_paths(cfg)
    return None if paths is None else paths[0]


def has_vision_tower(model, cfg) -> bool:
    """Does this tree hold cfg's vision tower, all of it (skeleton(cfg,
    vision=True), attach_vision_tower, or a class that builds its own)?"""
    paths = vision_tower_paths(cfg)
    if paths is None:
        return False
    try:
        return all(model.get_submodule(p) is not None for p in paths)
    except AttributeError:  # absent, or pruned to None
        return False


def tower_included(model, snap) -> bool:
    """raw.vision_tower_included: does the measured tree hold the vision
    tower of the checkpoint it was built from? Judged against the snapshot's
    own config.json, as the loaders build the tree, not against
    model.config: a Qwen3.5 tree (MiMo-9B, Qwen3.8-27B) is its text class,
    Qwen3_5ForCausalLM, whose model.config is the text config, with no
    vision_config even when the tower sits at model.visual. A source that
    holds no config.json (the tests' stubbed sources) is judged against the
    tree's own config."""
    from transformers import AutoConfig

    if os.path.isfile(os.path.join(str(snap), "config.json")):
        return has_vision_tower(model, AutoConfig.from_pretrained(snap))
    return has_vision_tower(model, model.config)


def attach_vision_tower(model, cfg) -> str | None:
    """Build cfg's vision tower EMPTY (meta device, bf16), as the composite
    class builds it (AutoModel.from_config(cfg.vision_config)), and attach
    it at its served path. A tree that holds it already is left as it is.
    Returns the ViT's path (vision_tower_paths' first), or None when the
    architecture serves no tower."""
    from transformers import AutoModel

    paths = vision_tower_paths(cfg)
    if paths is None or has_vision_tower(model, cfg):
        return None if paths is None else paths[0]
    if cfg.model_type in _CLASS_BUILDS_TOWER or len(paths) != 1:
        raise ValueError(f"{cfg.model_type}: its vision tower is built by its own class "
                         f"({', '.join(paths)}), never attached")
    with torch.device("meta"):
        try:
            tower = AutoModel.from_config(cfg.vision_config, dtype=torch.bfloat16).eval()
        except TypeError:  # pre-5.x arg name
            tower = AutoModel.from_config(cfg.vision_config, torch_dtype=torch.bfloat16).eval()
    parent_path, _, leaf = paths[0].rpartition(".")
    setattr(model.get_submodule(parent_path) if parent_path else model, leaf, tower)
    return paths[0]


def drop_vision_tower(model, cfg):
    """Prune cfg's vision tower from a tree whose class built it (gemma-4,
    Muse-Glimmer), in place, the way _prune_non_text prunes: each subtree assigned None.
    A tree without it passes through untouched."""
    for path in vision_tower_paths(cfg) or ():
        parent_path, _, leaf = path.rpartition(".")
        try:
            parent = model.get_submodule(parent_path) if parent_path else model
        except AttributeError:
            continue
        if getattr(parent, leaf, None) is not None:
            setattr(parent, leaf, None)
    return model


# model_type -> (checkpoint prefix, skeleton prefix) for the text tower. Only
# arches whose text skeleton renames the wrap need an entry.
_TEXT_WRAP: dict[str, tuple[str, str]] = {
    # the text-only Qwen3_5ForCausalLM calls them model.*, the checkpoint
    # calls them model.language_model.*
    "qwen3_5": ("model.language_model.", "model."),
    "qwen3_5_text": ("model.language_model.", "model."),  # see _NON_TEXT_TOWERS
}


def _image_text_auto():
    """The auto class that owns composite VLM configs. Resolved at CALL time,
    not import time, so a transformers too old to have it fails on the one
    checkpoint that needs it (AttributeError naming the missing class) instead
    of making this whole module unimportable."""
    import transformers

    return transformers.AutoModelForImageTextToText


def _prune_non_text(model, cfg):
    """Drop the towers _NON_TEXT_TOWERS names for this arch, in place.

    drinkme serves text. A vision tower left on the tree costs its parameters
    twice over — resident VRAM at serve, and (because eligible_linears would
    name its Linears) a pack of weights nothing reads. Assigning None keeps the
    attribute present and LOUD: an image request hits "NoneType is not
    callable" instead of a wrong answer. Absent submodules are skipped, so an
    arch whose text skeleton never built the tower (qwen3_5) passes through
    untouched."""
    for path in _NON_TEXT_TOWERS.get(getattr(cfg, "model_type", None), ()):
        parent_path, _, leaf = path.rpartition(".")
        try:
            parent = model.get_submodule(parent_path) if parent_path else model
        except AttributeError:
            continue  # the wrap this tower hangs off does not exist here
        if getattr(parent, leaf, None) is not None:
            setattr(parent, leaf, None)
    return model


def skeleton(cfg, vision: bool | None = None):
    """Meta-device skeleton: shapes and module tree, zero bytes of weights.
    Mirrors load_cpu's accommodations (transformers-5 dtype rename, the
    image-text-to-text mapping for composite VLMs, the same tower prune) so
    packer, loader, and measurement arms all walk the SAME tree — pack names
    == loader asks, by construction. `vision` True puts the served vision
    tower (_VISION_TOWERS) in the tree where the architecture has one (the
    packer's tree, and the server's unless images are off), False leaves it
    out, and None builds the tree the bench's arms measure (_TOWER_BY_DEFAULT:
    every architecture with a served tower — gemma-4, Qwen3.5, Muse-Glimmer —
    so a measurement always describes the full served model); a loader
    without it skips the pack's tower tensors by name."""
    from transformers import AutoModelForCausalLM

    def _build(cls):
        with torch.device("meta"):
            try:
                return cls.from_config(cfg, dtype=torch.bfloat16).eval()
            except TypeError:  # pre-5.x arg name
                return cls.from_config(cfg, torch_dtype=torch.bfloat16).eval()

    try:
        model = _build(AutoModelForCausalLM)
    except (ValueError, KeyError):
        # See load_cpu: only the Auto classes define `from_config` (a plain
        # model class raises AttributeError — "type object
        # 'Gemma3ForConditionalGeneration' has no attribute 'from_config'",
        # muse_glimmer), so ask the auto class.
        model = _build(_image_text_auto())
    model = _prune_non_text(model, cfg)
    if _wants_tower(cfg, vision):
        attach_vision_tower(model, cfg)
    else:
        drop_vision_tower(model, cfg)
    return model


def ckpt_to_skel(cfg, vision: bool | None = None):
    """Checkpoint-key -> skeleton-name translation; None = skip the tensor.

    Multimodal wrappers ship text weights under a wrapped prefix while the
    text skeleton may name them differently, plus whole towers the text path
    never serves (vision, MTP). This mirrors transformers' own per-arch
    conversion_mapping + _keys_to_ignore_on_load_unexpected — one explicit
    entry per arch we serve, identity for everything else — so "pack names ==
    loader asks" stays true for wrapped checkpoints too.

    The skip list is _NON_TEXT_TOWERS, shared with the skeleton prune so the
    two can never disagree (see that table); the rename, when an arch needs
    one, is _TEXT_WRAP. qwen3_5 (Qwen3.8-27B) needs both:
    model.language_model.* -> model.*, mtp.* dropped, and model.visual.*
    passed through as it is named when the tree holds the tower (`vision`,
    as skeleton() was asked; None, the bench's tree, which holds it too),
    dropped when it does not. gemma4 needs no rename: its class builds the
    tower, so its model.vision_tower.* and model.embed_vision.* pass through
    unless the tree was built without them.
    muse_glimmer (Muse-Glimmer-30B) needs no rename either: its
    text keys already match the wrapper skeleton name-for-name, and its
    tower's (model.vision_tower/vision_adapter/vision_projection.*) pass
    through whenever the tree holds it, which the bench's tree does too.
    """
    mt = getattr(cfg, "model_type", None)
    skip = _NON_TEXT_TOWERS.get(mt, ())
    held = _wants_tower(cfg, vision)
    if mt in _VISION_TOWERS and not (held and vision_tower_paths(cfg)):
        skip += _VISION_TOWERS[mt]
    towers = tuple(t + "." for t in skip)
    wrap = _TEXT_WRAP.get(mt)
    if not towers and not wrap:
        return lambda key: key

    def translate(key: str):
        if towers and key.startswith(towers):
            return None
        if wrap and key.startswith(wrap[0]):
            return wrap[1] + key[len(wrap[0]):]
        return key

    return translate


# Moved to serving/checkpoint.py (torch-free, shared with the MLX engine);
# the name stays importable from here.
from .serving.checkpoint import resolve_source, snapshot_dir  # noqa: E402,F401


# the loader names bench dispatches on (fit.py owns the spelling: torch-free)
from .fit import STOCK_LOADER_STREAM, STOCK_LOADERS  # noqa: E402


def load_stock_streaming(model_id: str, revision: str | None = None, device: str = "cuda",
                         snap: str | None = None, vision: bool | None = None):
    """The bench's stock arm, loaded by STREAMING: the
    same module tree load_cpu builds (AutoModelForCausalLM, or the
    image-text-to-text class for a composite VLM, with the non-text towers
    pruned — skeleton() mirrors load_cpu accommodation for accommodation),
    built EMPTY on the meta device in bf16, then materialized one tensor at
    a time straight onto `device` from the checkpoint's safetensors shards
    by serving.engines.stream_checkpoint — the walker `drinkme serve`
    already streams its raw remainder with; NOT a second walker. Each host
    copy is released before the next tensor is read, so the host transient
    beyond the resident model is one tensor (fit.stock_transient_bytes,
    `direct=True`).

    Why: transformers' from_pretrained holds ~2x the model in host memory
    while it loads, WITH or WITHOUT device_map="cuda" — MEASURED
    on a Strix Halo (gfx1151, 124 GB unified): gemma-4-31B-it
    (58.25 GB bf16) took MemAvailable 107 -> 5 GB in ~30 s of "Loading
    weights" before a watchdog killed it, and unguarded the same load drove
    a global kernel OOM. On unified memory that 2x is charged against
    the one pool the GPU also draws from, so the 27B and Muse-Glimmer-30B
    stock arms (55.6 / 55.7 GB) were refused by the live guard on a 124 GiB
    machine. Streaming charges resident + one tensor and gets those stock
    numbers safely.

    forward-identical to from_pretrained's tree: tests/test_stock_stream_loader.py
    asserts logits BITWISE equal (torch.equal) between the two loaders on the
    CPU over tied-embedding, multi-shard, gemma-4 (computed embed_scale and
    per-layer-type inv_freq) and vision-pruned muse_glimmer toys. Tied
    weights: the checkpoint carries no lm_head, tie_weights() supplies it.
    Buffers not in the checkpoint (rotary inv_freq, embed_scale) are
    CONSTRUCTED by the module, never loaded (engines._rebuild_computed_buffers).
    dtype: bf16 as stored, floats cast exactly as from_pretrained(dtype=bf16).

    An FP8 (or otherwise quantized) checkpoint is NOT this loader's and is
    refused by name before a tensor is read (_refuse_quantized: the one
    line): the bf16 cast would otherwise store unscaled codes as the
    weight. Only the bench's bf16 stock arm calls this; `drinkme serve
    --stock` keeps engines.load_stock (from_pretrained on the CPU, then a
    per-tensor move).

    `snap`: the already-resolved snapshot directory (run_arms' one source
    for every arm); None resolves `model_id`@`revision` here. `vision`:
    skeleton()'s, None (the bench's tree) unless an image acceptance tool
    asks for the tower (bench/glimmer_vision_gate.py)."""
    from transformers import AutoConfig

    from .serving.engines import stream_checkpoint

    snap = snap or snapshot_dir(model_id, revision)
    _refuse_quantized(model_id, snap)
    cfg = AutoConfig.from_pretrained(snap)  # the snapshot's own config.json
    model = skeleton(cfg, vision=vision)
    stream_checkpoint(model, snap, cfg, device, name=model_id)
    return model


def _refuse_quantized(model_id: str, snap: str) -> None:
    """The bf16 streaming loaders read checkpoints AS STORED and never
    dequantize: an FP8 checkpoint (a float8 plane in the shard headers, or
    an fp8 quantization_config) — or any other quantization_config — is
    refused by name before any tensor is read, with the one line
    (codec/pack.refuse_checkpoint). Without this the bf16 cast would store
    unscaled codes as the weight."""
    refuse_checkpoint(model_id, snap)


def _stream_swapped(model_id: str, revision: str | None, device: str, make, arm: str,
                    snap: str | None = None, per_slot: dict | None = None):
    """The bench's compressed and twin arms, built by STREAMING: the same
    module tree load_cpu builds (skeleton() mirrors it accommodation for
    accommodation — the auto class, the tower prune, bf16), EMPTY on the
    meta device; then every eligible Linear (swap.eligible_linears over the
    skeleton — the same walk `drinkme pack` decides eligibility with, so
    the swapped population is the pack's) is built from its own checkpoint
    tensor as that tensor is read off its shard: `make(w, None, device)` is
    swap.make_compressed (pack_weight_served -> make_module, the per-tensor
    roundtrip gate inside) or swap.make_twin (RadixTwinLinear or RawLinear
    over the bf16 bytes), the module lands on `device`, the host copy is released
    before the next tensor is read. The remainder — embeddings, norms,
    sub-bar Linears, and the swapped Linears' biases (bias-after-swap) —
    then streams through the ONE walker (serving.engines.stream_checkpoint,
    which skips a swapped Linear's weight by module type and attaches its
    bias).

    ORDER: the eligible Linears are swapped LARGEST FIRST (by element count,
    off the shard headers — not in shard order). The packer's host scratch
    over one tensor is ~10x that tensor (MEASURED on Strix Halo,
    pack_weight_served on the CPU: VmHWM +4.73 GB over a Qwen2.5-72B
    down_proj-shaped [8192, 29568] bf16 tensor of 0.484 GB, 9.8x; 10.3x /
    9.8x / 9.8x at three other shapes — linear), and the 72B's UNTIED
    lm_head ([152064, 8192], 2.49 GB) is itself eligible: packing it needs
    ~27 GB of host, which is fine with nothing resident and fatal with 100
    GB already on a unified device. Largest first puts that scratch where
    the resident set is smallest; by the time the raw remainder streams
    (the walker, last — the 2.49 GB embedding with everything resident)
    the transient is one tensor, which is what fit.streamed_transient_bytes
    charges. The order changes nothing in the result: each tensor's pack
    depends on its own bytes alone, and the stats are sorted back into
    eligible_linears order below.

    Why streaming and not load_cpu(model_id) + swap_linears + `.cuda()`: on
    a 122 GB machine a whole-model host stage cannot exist for Qwen2.5-72B (145
    GB bf16), and the fit point the site carries (the 72B compressed at
    102.71 GB) is one `drinkme bench` has to be able to take.

    The tree is IDENTICAL in structure and numerics to load_cpu +
    swap_linears's: the same class at every slot, the same packed dict /
    the same twin weight per tensor (the input bytes are the same bytes), the
    same bias, the same raw remainder — tests/test_stream_compressed_arms.py
    pins logits BITWISE equal (torch.equal) between the two paths on the
    CPU for both arms, and the stats list equal in content AND order
    (sorted here into eligible_linears order, which is swap_linears's).
    An eligible tensor the checkpoint stores as anything but bf16 is
    refused by name at the door (codec/pack.refuse_checkpoint, the
    source-dtype contract), never cast. Returns
    (model, stats) — stats in make_compressed's shape (empty for the twin,
    whose make_twin returns no stat). `snap`: the already-resolved snapshot
    directory (run_arms' one source for every arm); None resolves
    `model_id`@`revision` here. `per_slot` ({module name: make's keyword
    arguments}, twin_plan's) is the per-tensor half of `make`'s call: the
    twin arm's codec and widths per slot, which only the compressed arm's
    pack knows; a slot the plan does not name is refused."""
    import functools
    import gc
    import math

    from safetensors import safe_open
    from transformers import AutoConfig

    from .codec.pack import _shard_headers
    from .codec.swap import eligible_linears
    from .serving.engines import stream_checkpoint

    snap = snap or snapshot_dir(model_id, revision)
    _refuse_quantized(model_id, snap)
    cfg = AutoConfig.from_pretrained(snap)
    model = skeleton(cfg)
    to_skel = ckpt_to_skel(cfg)
    order = {name: i for i, (name, _, _, _) in enumerate(eligible_linears(model))}
    want = {f"{name}.weight": name for name in order}  # skeleton key -> module name
    headers = _shard_headers(snap)  # {checkpoint key: (dtype, shape, shard)} — zero weight bytes
    if not headers:
        raise FileNotFoundError(f"no *.safetensors under {snap}")
    plan = []
    for tname, (_dtype, shape, fpath) in headers.items():
        sname = to_skel(tname)
        name = want.get(sname) if sname else None
        if name is not None:  # else: the raw remainder, a bias, or a pruned tower
            plan.append((math.prod(shape), tname, fpath, name))
    plan.sort(key=lambda t: (-t[0], t[1]))  # LARGEST FIRST — see the docstring
    stats: list[dict] = []
    installed: set[str] = set()
    for _numel, tname, fpath, name in plan:
        with safe_open(fpath, framework="pt") as sf:  # mmap + header parse: cheap per tensor
            w = sf.get_tensor(tname)
            if not w.is_floating_point():  # a float8 plane was refused at the door above
                raise ValueError(f"checkpoint tensor {tname} is {w.dtype}, not a bf16 "
                                 "(or wider float) weight the codec can take")
            if w.dtype != torch.bfloat16:  # refused at the door above; never cast here
                reason = source_dtype_refusal_reason([(tname, str(w.dtype))], lambda _n: True)
                raise UnsupportedCheckpoint(f"{model_id}: {reason}")
            if per_slot is None:
                make_one = make
            elif name in per_slot:
                make_one = functools.partial(make, **per_slot[name])
            else:
                raise ValueError(f"the {arm} arm's plan names no tensor {name!r}: the "
                                 "compressed arm did not swap it")
            _install_swapped(model, name, w, device, make_one, stats)
            installed.add(name)
            del w
        gc.collect()  # this tensor's host copy and the packer's scratch, gone before the next
    missing = sorted(set(order) - installed, key=order.get)
    if missing:
        raise ValueError(f"checkpoint is missing {len(missing)} weights the {arm} arm needs: "
                         f"{missing[:8]}")
    stats.sort(key=lambda st: order[st["name"]])  # swap_linears's order, exactly
    stream_checkpoint(model, snap, cfg, device, name=model_id)  # the remainder + the biases
    return model, stats


def load_compressed_streaming(model_id: str, revision: str | None, device: str = "cuda",
                              snap: str | None = None):
    """The bench's COMPRESSED arm, streamed (_stream_swapped over
    swap.make_compressed): every eligible Linear packed EXACTLY as
    `drinkme pack` writes it (pack_weight_served: radix at the default profile) and installed through
    the loader's own constructor (make_module) straight from its checkpoint
    tensor, so the whole bf16 model never exists on the host. THE GATE
    stays where it was: pack_weight_served -> the CPU round trip raises on any
    single tensor's roundtrip failure, inside the load, before that tensor
    is even installed; run_arms's refuse_unless_verified sits right after
    the load, before any timing. Returns (model, stats) — the stats
    swap_linears(model, make_compressed) would have returned, in its
    order."""
    return _stream_swapped(model_id, revision, device, make_compressed, "compressed", snap=snap)


def twin_plan(stats: list[dict]) -> dict:
    """{module name: make_twin's per-tensor keywords} off the compressed
    arm's stats (make_compressed's, or pack_stat's under --pack-dir): each
    slot's codec and tier widths, so the twin serves every tensor the way
    the compressed arm does — a radix tensor through the same kernels at
    the same launch-table row, a raw fallback through the same F.linear."""
    return {st["name"]: {"codec": st["codec"], "widths": st.get("widths")} for st in stats}


def load_twin_streaming(model_id: str, revision: str | None, device: str = "cuda",
                        snap: str | None = None, plan: dict | None = None):
    """The bench's TWIN arm, streamed (_stream_swapped over swap.make_twin):
    at each eligible slot the module the compressed arm has there, over the
    UNCOMPRESSED bf16 weight — swap.RadixTwinLinear for a radix tensor (the
    served radix kernels with the decode replaced by a raw bf16 load, at
    the same launch schedule), swap.RawLinear for a raw fallback. `plan` is
    twin_plan(the compressed arm's stats): the codec and widths per slot.
    Each module is built from the checkpoint tensor as it is read, so the
    arm holds ONE bf16 copy and never a second on the host. Returns
    (model, []) — make_twin has no stat."""
    if plan is None:
        raise ValueError("the twin arm needs the compressed arm's plan (arms.twin_plan)")
    return _stream_swapped(model_id, revision, device, make_twin, "twin", snap=snap, per_slot=plan)


def greedy(model, ids, n_new: int):
    out = ids
    past = None
    with torch.no_grad():
        for _ in range(n_new):
            res = model(
                out if past is None else out[:, -1:], past_key_values=past, use_cache=True
            )
            past = res.past_key_values
            nxt = res.logits[:, -1, :].argmax(-1, keepdim=True)
            out = torch.cat([out, nxt], -1)
    return out


class GraphDecode:
    """greedy()'s decode in graph mode (serving/cudagraph.py): the prompt
    prefilled eagerly into a LiveStaticCache sized to the run, then each of
    the n_new - 1 steps one replay of a captured M = 1 step that feeds its
    own argmax back into its input on the device, as greedy() does with
    torch.cat. Built (prefill, warm-up, capture) untimed; `run()` is one
    timed decode, the same n_new forwards greedy() makes."""

    def __init__(self, model, ids, n_new: int):
        from .serving.cudagraph import StaticStep
        from .serving.kvcache import LiveStaticCache

        self.model, self.ids, self.n_new = model, ids, n_new
        self.P = ids.shape[1]
        self.cache = LiveStaticCache(config=model.config, max_cache_len=self.P + n_new)
        self.tokens = torch.zeros(self.P + n_new, dtype=torch.long, device=ids.device)
        cache, tokens = self.cache, self.tokens

        def forward(tok, position_ids, mask):
            logits = model(input_ids=tok, position_ids=position_ids, attention_mask=mask,
                           past_key_values=cache, use_cache=True).logits[0, -1]
            nxt = logits.argmax(-1).view(1)
            tokens.index_copy_(0, position_ids.view(-1) + 1, nxt)
            tok.copy_(nxt.view(1, 1))
            return logits

        self._prefill()
        self.step = StaticStep(model, cache, 1, forward)
        self.step.capture()

    def _prefill(self):
        self.cache.reset()
        first = self.model(self.ids, past_key_values=self.cache, use_cache=True).logits[0, -1].argmax(-1)
        self.tokens[:self.P] = self.ids[0]
        self.tokens[self.P] = first
        return first

    def run(self):
        first = self._prefill()
        self.step.ids.copy_(first.view(1, 1))
        for _ in range(self.n_new - 1):
            self.step.replay()
        return self.tokens.view(1, -1)


def decode_step(model) -> str:
    """The step path `drinkme bench` times this arm's decode on:
    "cudagraph" where serving/cudagraph.decide() says so (CUDA, the
    family on its verified list, DRINKME_CUDA_GRAPHS not 0), else
    "eager". Every arm of a run asks it of the same checkpoint, so the
    arms agree unless a capture fails (timed_decode's `ran`)."""
    from .serving import cudagraph

    d = cudagraph.decide(model, next(model.parameters()).device) if isinstance(
        model, torch.nn.Module) else cudagraph.Decision(False, None)
    if d.line:
        cudagraph.say_once("bench:" + d.line, d.line)
    return "cudagraph" if d.on else "eager"


def timed_decode(model, ids, n_new: int = N_NEW, reps: int = DECODE_REPS,
                 step: str = "eager", ran: dict | None = None):
    """reps timed greedy decodes (identical greedy path -> identical tokens);
    returns (tokens, tok_s samples). `step` "cudagraph" times GraphDecode
    instead of greedy(); a capture that fails times greedy() after one
    line saying so, and `ran["step"]` (when given) says which ran."""
    samples = []
    out = None
    run = None
    if step == "cudagraph":
        try:
            run = GraphDecode(model, ids, n_new).run
        except Exception as e:  # noqa: BLE001 — a failed capture times eager, never fails the run
            from .serving import cudagraph
            cudagraph.say_once("bench:capture", f"decode step: CUDA graph capture failed "
                                                f"({type(e).__name__}: {str(e)[:200]}); timing eager")
    if ran is not None:
        ran["step"] = "eager" if run is None else "cudagraph"
    if run is None:
        def run():
            return greedy(model, ids, n_new)
    if WARMUP_REP:
        run()  # untimed: compile, first-touch, allocator growth
        torch.cuda.synchronize()
    for _ in range(reps):
        torch.cuda.synchronize()
        t0 = time.time()
        out = run()
        torch.cuda.synchronize()
        samples.append(round(n_new / (time.time() - t0), 2))
    return out, samples


def build_prefill_ids(tok, length: int = PREFILL_LEN) -> list[int]:
    """A deterministic `length`-token prompt: PREFILL_PASSAGE tokenized once
    (no special tokens — this is a synthetic timing prompt, not a real chat
    turn) and repeated until long enough, then truncated to exactly `length`
    ids. Same tokenizer, same text in -> same ids out, every run, so
    prefill_tok_s/ttft_s are comparable across models despite each
    tokenizer cutting PREFILL_PASSAGE into a different number of pieces."""
    ids = tok(PREFILL_PASSAGE, add_special_tokens=False).input_ids
    if not ids:
        raise ValueError("prefill passage tokenized to zero ids")
    while len(ids) < length:
        ids = ids + ids
    return ids[:length]


def timed_prefill(model, ids, reps: int = DECODE_REPS):
    """`reps` timed forward passes over the full `ids` sequence, each with
    logits_to_keep=1 where the arm supports it (only the last position's
    logits get materialized — the actual prefill shape, distinct from
    timed_decode's KV-cached single-row steps). Returns tok/s samples over
    len(ids) tokens per call."""
    n = ids.shape[1]
    samples = []
    with torch.no_grad():
        for i in range(reps + int(WARMUP_REP)):
            torch.cuda.synchronize()
            t0 = time.time()
            try:
                model(ids, logits_to_keep=1)
            except TypeError:  # arm/model build doesn't take the kwarg
                model(ids)
            torch.cuda.synchronize()
            if WARMUP_REP and i == 0:
                continue  # the untimed rep
            samples.append(round(n / (time.time() - t0), 2))
    return samples


def timed_ttft(model, ids, reps: int = DECODE_REPS):
    """`reps` timings of wall-clock time from call to the first generated
    token on the SAME prompt as timed_prefill: one prefill forward plus one
    greedy decode step. Returns second samples (a latency, not a rate)."""
    samples = []
    with torch.no_grad():
        for i in range(reps + int(WARMUP_REP)):
            torch.cuda.synchronize()
            t0 = time.time()
            out = model(ids, use_cache=True)
            out.logits[:, -1, :].argmax(-1, keepdim=True)
            torch.cuda.synchronize()
            if WARMUP_REP and i == 0:
                continue  # the untimed rep
            samples.append(round(time.time() - t0, 4))
    return samples


def _median(xs, ndigits: int = 2):
    return round(float(np.median(xs)), ndigits)


def stat_evidence(stat: dict) -> str | None:
    """What a compressed-arm stat's bytes rest on: "roundtrip" (swap.
    make_compressed — the tensor was encoded and compared with its source
    bits in this process) or "pack" (pack_stat — the pack verified at
    load, trust in the pack-time comparison); None for a stat with neither,
    which the gate refuses."""
    if stat.get("verified"):
        return "roundtrip"
    if stat.get("pack_verified"):
        return "pack"
    return None


def refuse_unless_verified(model_id: str, stats: list[dict]) -> None:
    """THE GATE: a round-trip result can only ever be published true, so
    it is not a field to write — it is a condition to refuse on. Raises a
    named SystemExit (nonzero, printed) rather than returning when any
    compressed tensor carries no evidence (stat_evidence: a fresh round
    trip for the in-memory arm, the pack hashes for the pack arm), or when nothing
    was eligible to compress at all. make_compressed's own
    pack_weight_served already RAISES on a round-trip mismatch for any
    single tensor (codec/radix_pack.py), and build_compressed_model on a pack that fails verify_hashes, so
    `not stats` is the one case this still catches in practice — kept as a
    real check, not a dead one, because a model with zero eligible Linears is
    a real (if unlikely) input. Pure Python, no torch: callable from tests
    without a GPU."""
    if stats and all(stat_evidence(s) for s in stats):
        return
    n_bad = sum(1 for s in stats if not stat_evidence(s))
    raise exitcodes.Refused(
        f"drinkme: refusing to write the record: {model_id} — "
        f"{n_bad} of {len(stats)} compressed tensors carry no evidence — "
        "neither the encoder's own bitwise round trip (the in-memory arm) "
        "nor a pack that verified at load (the pack arm) — or no tensor was eligible for "
        "compression at all. bench writes a record only when every compressed "
        "tensor was verified; it refuses rather than publish an unverified one.")


def refuse_unless_routed_alike(model_id: str, routes: dict) -> None:
    """THE OTHER GATE: every arm this run measured
    must have routed the same way — the same recurrence and prefill convolution,
    the same narrow GEMV and count (serving/kernel_route.route_kernels's
    record, `routes` = {arm: record} for the arms routed so far) — or the
    ratio compares two runtimes, not two weight-reads (the 27B's stock arm
    on the torch recurrence against a compressed arm on fla: ~7% in the
    compressed arm's favour). Raises a named SystemExit, every arm's route
    in the message, rather than returning; called after EACH arm routes,
    so a mismatch refuses before that arm is timed. Pure Python, no torch:
    callable from tests without a GPU."""
    if len({json.dumps(r, sort_keys=True) for r in routes.values()}) <= 1:
        return
    each = "; ".join(f"{arm} {describe_route(r) if r else 'unrouted'}" for arm, r in routes.items())
    raise exitcodes.Refused(
        f"drinkme: refusing to write the record: {model_id} — the arms did not route alike "
        f"({each}). A bench that compares arms must route both or neither: the same "
        "DeltaNet recurrence, prefill convolution, and narrow GEMV on every arm, or the "
        "ratio measures the runtime and not the weight-read. Look at the [drinkme] "
        "deltanet recurrence / prefill convolution / narrow gemv lines above for "
        "which arm fell back and why; DRINKME_DELTANET_KERNEL / DRINKME_DELTANET_CONV / "
        "DRINKME_NARROW_GEMV pin a route for every arm.")


def _route_arm(model, arm: str, routes: dict) -> dict | None:
    """route_kernels over one arm's loaded tree — the SAME call
    engines.load_stock / load_compressed make, so what the bench times is
    what `drinkme serve` runs — said on one line beside the arm's pass,
    kept under `routes[arm]`, and checked against the arms before it
    (refuse_unless_routed_alike). None for a tree that is not an
    nn.Module (the tests' stubs): nothing to route, nothing claimed."""
    if not isinstance(model, torch.nn.Module):
        return None
    rec = route_kernels(model, "cuda")
    print(f"  {arm} arm: {describe_route(rec)}", flush=True)
    routes[arm] = rec
    return rec


def arm_compression_profile(stats: list[dict]) -> str:
    """The record's compression.profile, read OFF THE LOADED ARM: every
    swapped tensor's codec/profile (make_compressed's stat) — the same
    string meta.json's writer carries as `profile` (`sip` / `gulp`) — so
    the label names what was timed, never a literal the bench types
    (tests/test_bench_arm_wiring.py pins that no profile string is a
    literal in bench.py). An arm is one profile's radix tensors plus its
    raw fallbacks (codec/radix_pack.py) — one artifact, named by the
    profile; an arm that mixes profiles, or holds a tensor with a `dtype`
    scalar (an FP8 pack's), is refused by name."""
    if not stats:
        raise ValueError("arm_compression_profile: no compressed tensors — nothing was timed")
    dtypes = {s.get("dtype") for s in stats}
    if dtypes != {None}:
        raise ValueError(f"arm_compression_profile: the compressed arm holds a dtype this build does not name "
                         f"({sorted(map(str, dtypes))})")
    codecs = {s.get("codec") for s in stats}
    if not codecs <= {"radix", "raw"}:
        raise ValueError(f"arm_compression_profile: the compressed arm mixes encodings (codec {sorted(map(str, codecs))}) "
                         "— one profile cannot name it")
    compression_profiles = {s.get("profile") for s in stats if s.get("codec") == "radix"}
    if len(compression_profiles) != 1:
        raise ValueError(f"arm_compression_profile: the compressed arm holds {len(compression_profiles)} profiles "
                         f"({sorted(map(str, compression_profiles))}); one artifact is one profile")
    return str(compression_profiles.pop())


def stock_load_failure(e: BaseException) -> bool:
    """Is `e` the stock arm failing to LOAD — as opposed to a bug? True for
    the live guard's own refusal (exitcodes.LoadRefused, decided by type),
    an accelerator out-of-memory (torch.OutOfMemoryError, or the
    RuntimeError HIP/CUDA raise naming "out of memory") and a host
    MemoryError. Only these become `stock_outcome: failed_load` in run_arms
    (an observed failure is a different evidence category from a
    predicted non-fit, and neither should take
    the compressed arm's measurement down with it); anything else still
    crashes the run — a missing `accelerate`, a config the loader cannot
    build, a bug — because a record must not quietly call those "did not
    fit". Pure Python, no device."""
    if isinstance(e, SystemExit):
        return isinstance(e, exitcodes.LoadRefused)
    if isinstance(e, MemoryError):
        return True
    oom = getattr(torch, "OutOfMemoryError", None) or getattr(torch.cuda, "OutOfMemoryError", ())
    if isinstance(e, oom):
        return True
    return isinstance(e, RuntimeError) and "out of memory" in str(e).lower()


def bpw_stats(stats: list[dict]) -> tuple[float, float]:
    """(mean over tensors, weight-weighted) bits per weight of the
    compressed arm's packed population — the swapped Linears only (the raw
    remainder the arm never touches is not in either number), off
    make_compressed's stats. The record publishes the weighted one as
    compression.bitsPerWeight and the tensor mean as
    meanTensorBitsPerWeight. Pure Python."""
    mean = round(float(np.mean([s["bpw"] for s in stats])), 3)
    weighted = round(weighted_bpw([s["bits"] / s["numel"] for s in stats],
                                  [s["numel"] for s in stats]), 3)
    return mean, weighted


def live_available_bytes(memory_kind: str) -> float | None:
    """A LIVE read of what's actually free right now, not the detect()
    snapshot bench took whenever it started (which can be stale by the time
    a multi-minute bandwidth probe and any earlier pass finish — on a Strix
    Halo three browser tabs and an unrelated process ate into the budget
    between detect and this load). unified: `/proc/meminfo`'s
    MemAvailable — the same physical pool the GPU's GTT window draws from.
    discrete: `torch.cuda.mem_get_info()`'s free byte count. None when
    neither read succeeds — the caller must treat that as "unknown", not as
    "plenty free"."""
    if memory_kind == "unified":
        from .detect import meminfo_available_bytes

        return meminfo_available_bytes()
    try:
        free, _total = torch.cuda.mem_get_info()
        return float(free)
    except Exception:  # noqa: BLE001 — a probe failure must not crash the refusal itself
        return None


def refuse_if_stock_wont_fit(model_id: str, bf16_bytes: float, memory_kind: str,
                             available_bytes: float | None, direct: bool = False,
                             tensor_bytes: float | None = None) -> None:
    """Defense in depth behind bench.py's own fit check (drinkme.fit's
    stock_arm_fits, run against the detect()-time snapshot): a LIVE check
    right before the stock arm's big allocation, using `available_bytes`
    (live_available_bytes's reading, passed in rather than read here so this
    stays pure-Python-testable — see test_bench_record_shape.py's style for
    why). Raises a named SystemExit rather than letting a doomed load run
    into a global OOM (as on a Strix Halo).

    `direct` / `tensor_bytes` are the loader's term:
    True + the checkpoint's largest tensor for the streaming loader
    (load_stock_streaming: resident + one tensor), False for from_pretrained
    (2x resident on unified, measured) — the SAME arithmetic bench's
    fit check ran (fit.stock_transient_bytes), so the two guards can only
    disagree on the live reading, never on the charge."""
    from .fit import GB, STAGING_SHARD_BYTES, stock_arm_fits
    from .suggest import RATIO_HEADROOM

    if available_bytes is None:
        raise exitcodes.LoadRefused(
            f"drinkme: refusing to load: {model_id} — could not read live "
            f"{'system memory' if memory_kind == 'unified' else 'VRAM'} "
            "right before the stock arm's allocation (see arms.live_available_bytes). "
            "An unreadable machine is not a known-safe one; fix the read or route "
            "this model through the fit-point path (compressed arm only).")
    if stock_arm_fits(bf16_bytes, 0.0, available_bytes, memory_kind, RATIO_HEADROOM,
                      direct=direct, tensor_bytes=tensor_bytes):
        return
    if memory_kind != "unified":
        extra = ""
    elif direct:
        one = tensor_bytes if tensor_bytes else STAGING_SHARD_BYTES
        extra = (f" + one tensor ({one / GB:.2f} GB, the streaming loader's host "
                 f"transient{'' if tensor_bytes else ' — the 5 GiB shard bound, headers unread'})")
    else:
        extra = " x2 (from_pretrained's transient on unified memory, measured)"
    raise exitcodes.LoadRefused(
        f"drinkme: refusing to load: {model_id} — the stock arm's bf16 weights "
        f"({bf16_bytes / GB:.2f} GB{extra}) do not fit what is live right now "
        f"({available_bytes / GB:.2f} GB "
        f"{'system memory' if memory_kind == 'unified' else 'VRAM'} available), which "
        "even before KV or activations. On unified memory the loader's staging and "
        "the device copy are the same physical pool, so this load would exhaust "
        "system memory. Free memory or let "
        "bench route this model to a fit-point record instead (compressed arm only).")


def refuse_if_compressed_wont_fit(model_id: str, compressed_bytes: float, memory_kind: str,
                                  available_bytes: float | None,
                                  tensor_bytes: float | None = None) -> None:
    """The compressed arm's live guard, right before its streaming load —
    the same live reading (live_available_bytes) and the same one-tensor
    term (fit.streamed_transient_bytes) as the stock arm's guard, over the
    pack's resident bytes (bench's suggest.sizes: a pack's own residentBytes
    or the checkpoint-ratio estimate, never a fact learned by loading). Two lines, because this arm has no fallback — a
    refusal ends the run with NO record, which is the honest outcome, so
    it refuses only at the hard line:
      * resident + one tensor > what is live right now: SystemExit, named
        (on unified memory this is a global OOM — the
        load would take the desktop with it);
      * it fits, but not with suggest's FIT_HEADROOM: printed as a
        knife-edge attempt (suggest()'s own stance for such a fit point:
        "attempting, may OOM honestly") and the load proceeds.
    An unreadable live figure refuses, as the stock guard does: an
    unreadable machine is not a known-safe one."""
    from .fit import GB, STAGING_SHARD_BYTES, streamed_transient_bytes
    from .suggest import FIT_HEADROOM

    pool = "system memory" if memory_kind == "unified" else "VRAM"
    if available_bytes is None:
        raise exitcodes.LoadRefused(
            f"drinkme: refusing to load: {model_id} — could not read live {pool} right "
            "before the compressed arm's allocation (see arms.live_available_bytes). An "
            "unreadable machine is not a known-safe one; fix the read and re-run.")
    charge = streamed_transient_bytes(compressed_bytes, 0.0, memory_kind, tensor_bytes)
    if memory_kind == "unified":
        one = tensor_bytes if tensor_bytes else STAGING_SHARD_BYTES
        term = (f" + one tensor ({one / GB:.2f} GB, the streaming loader's host transient"
                f"{'' if tensor_bytes else ' — the 5 GiB shard bound, headers unread'})")
    else:
        term = ""
    if charge > available_bytes:
        raise exitcodes.LoadRefused(
            f"drinkme: refusing to load: {model_id} — the compressed arm's resident bytes "
            f"({compressed_bytes / GB:.2f} GB{term} = {charge / GB:.2f} GB) exceed what is "
            f"live right now ({available_bytes / GB:.2f} GB {pool} available). This arm "
            "has no fallback: no record is written. Free memory (on unified memory the "
            "desktop's own resident set is part of the pool) and re-run; a smaller "
            "compressed pack or a bigger machine are the other options.")
    if charge * FIT_HEADROOM > available_bytes:
        print(f"[drinkme] compressed arm: {charge / GB:.2f} GB charged against "
              f"{available_bytes / GB:.2f} GB live {pool} — fits, but not with the "
              f"x{FIT_HEADROOM} headroom; attempting (knife-edge: it may run out of memory, "
              "and then the run ends without a record)", flush=True)


def run_arms(model_id: str, revision: str | None, prompt: str, run_stock: bool = True,
            stock_memory_kind: str = "vram", stock_bf16_bytes: float | None = None,
            run_twin: bool = True, stock_loader: str = STOCK_LOADER_STREAM,
            stock_tensor_bytes: float | None = None,
            compressed_bytes: float | None = None,
            pack_dir: str | None = None, spec: bool = True) -> dict:
    """The full measurement — three passes (stock, compressed, twin) when
    `run_stock` (default), two (compressed, twin) when bench's own fit check
    (drinkme.fit.stock_arm_fits) predicted the stock arm would not fit, one
    (compressed) when the twin's single bf16 copy was predicted not to fit
    either (`run_twin`). The twin runs last because it is built from the
    compressed arm's stats (twin_plan).
    `stock_memory_kind` (the box's memory kind — every arm's live guard
    reads it) / `stock_bf16_bytes` feed the live pre-allocation guard right
    before the stock pass's load — the checkpoint's safetensors bytes
    (bench's suggest.sizes), not a fact learned by loading;
    `compressed_bytes` (the pack's resident bytes, the same sizes' comp_gb)
    feeds the compressed arm's own live guard
    (refuse_if_compressed_wont_fit) the same way. `stock_loader`
    names HOW pass 1 loads: "stream" (default —
    load_stock_streaming, one tensor of host transient; `stock_tensor_bytes`
    is the checkpoint's largest tensor for every streaming guard's charge,
    None = the 5 GiB shard bound) or "from_pretrained"
    (load_cpu(device_map="cuda"), the transformers loader, 2x resident on
    unified memory — measured). Passes 2 and 3 ALWAYS stream
    (load_twin_streaming / load_compressed_streaming: one tensor of host
    transient beyond the resident arm, never the whole bf16 model on the
    host). The record's raw carries the stock loader as `stock_loader`.
    When `run_stock` is False the stock_* fields are OMITTED from the
    returned dict (not null — the lexicon's own convention for a fit point,
    see bench.py), and `report["stock_error"]` names why.
    `report["stock_outcome"]` / `report["twin_outcome"]` name each arm's outcome
    (measured | skipped_predicted_nonfit | failed_load, the last stock-only
    — see pass 1). Returns the flat result dict (same field names as the
    research harness, so curve tooling reads both).

    `spec` (False under `drinkme bench --no-spec`): after each measured
    stock and compressed arm's plain timings, the speculation pass over the
    same loaded tree (time_speculation; spec_pass.py), into `report["spec"]`
    (spec_pass.assemble); False records {"skipped": "--no-spec"}. The plain
    timings come first, so nothing the pass loads or wraps (the head, the
    DeltaNet capture) is on the tree while they run.

    ONE SOURCE, FIRST: the snapshot every arm and the tokenizer load from
    is resolved once here (serving.checkpoint.resolve_source) — the pack's
    bound commit under --pack-dir, else the caller's coordinates resolved
    now — and handed to each loader as `snap`; no arm resolves its own.
    `resolved_revision` (the commit, not the coordinate) rides in the
    report for the record's model.revision."""
    from .serving.checkpoint import tokenizer as load_tokenizer

    snap, resolved = resolve_source(model_id, revision, pack_dir)
    tok = load_tokenizer(snap, None)  # the one snapshot's, not a re-resolve
    ids = tok(prompt, return_tensors="pt").input_ids.cuda()
    prefill_ids = torch.tensor([build_prefill_ids(tok)], device="cuda")
    report: dict = {
        "model": model_id,
        "revision": revision,            # the coordinate asked for (the menu's)
        "resolved_revision": resolved,   # the commit every arm loaded — model.revision
        "gpu": torch.cuda.get_device_name(0),
        "prefill_prompt_len": prefill_ids.shape[1],
    }
    report["warmup_rep"] = WARMUP_REP  # one untimed rep per timed shape; False = DRINKME_BENCH_WARMUP=0
    # no pack_dir in raw: a record is published as written, and a local path
    # names the operator's home. That the arm ran off a pack on disk is
    # already in the record (compression.packedBytes, raw.pack).
    report["bandwidth"] = measure_bandwidth()  # BEFORE any load: empty device
    pack_meta = None
    # {arm: what route_kernels did to its tree} for every arm that loaded
    # (raw carries each as `<arm>_routing`); the arms must agree or no
    # record is written (refuse_unless_routed_alike, after each arm routes)
    routes: dict = {}
    # {arm: time_speculation's result} for each measured arm the pass ran over
    spec_results: dict = {}

    with torch.inference_mode():
        # --- pass 1: stock bf16 ---
        # `stock_outcome` names the OUTCOME as
        # an evidence category: "measured" (pass ran, stock_* fields
        # present), "skipped_predicted_nonfit" (bench's fit check stepped
        # aside — a prediction, never an observation) or "failed_load"
        # (attempted, and the live guard refused or the load OOM'd — an
        # observation, the message kept). bench.py publishes it as
        # stock.outcome, with the budget or the error beside it.
        if stock_loader not in STOCK_LOADERS:
            raise ValueError(f"unknown stock loader {stock_loader!r}; one of {STOCK_LOADERS}")
        report["stock_loader"] = stock_loader
        if run_stock:
            try:
                streaming = stock_loader == STOCK_LOADER_STREAM
                if stock_bf16_bytes is not None:
                    refuse_if_stock_wont_fit(model_id, stock_bf16_bytes, stock_memory_kind,
                                             live_available_bytes(stock_memory_kind),
                                             direct=streaming,
                                             tensor_bytes=stock_tensor_bytes if streaming else None)
                # Two loaders: STREAMING (the default) builds the tree
                # empty on the meta device and moves one tensor at a time
                # from the shards — host transient = one tensor; the tree is
                # forward-identical to from_pretrained's (bitwise, pinned in
                # tests/test_stock_stream_loader.py). from_pretrained with
                # device_map="cuda" is the selectable alternative and
                # the reason the streaming one exists: on unified memory it
                # holds ~2x the model in host memory while loading, straight
                # to the device or not (MEASURED on a Strix Halo:
                # gemma-4-31B-it took MemAvailable 107 -> 5 GB in ~30 s).
                if streaming:
                    model = load_stock_streaming(model_id, revision, device="cuda", snap=snap)
                else:
                    model = load_cpu(model_id, revision, device_map="cuda", snap=snap)
                # the runtime's kernel routes over the raw bf16 tree — the
                # same recurrence kernel and narrow GEMV the other arms and
                # `drinkme serve` get; "stock" is drinkme's runtime over the
                # raw weights, not transformers' eager path (docs/bench.md)
                report["stock_routing"] = _route_arm(model, "stock", routes)
                report["vram_bf16_bytes"] = vram_bytes()
                if isinstance(model, torch.nn.Module):  # the tests' stubs are not
                    # the stock arm's bandwidth bound: read_gb_s / stock_decode_read_gb
                    report["stock_bytes_per_token"] = decode_read_bytes(model, tower_paths(model, snap))
                pbytes = sum(p.numel() for p in model.parameters()) * 2
                report["bandwidth"]["total_param_bytes_bf16"] = pbytes
                report["bandwidth"]["ceiling_tok_s_bf16"] = ceiling_tok_s(
                    report["bandwidth"]["read_bytes_s"], pbytes)
                load0 = load1()
                ran: dict = {}
                _, s_stock = timed_decode(model, ids, step=decode_step(model), ran=ran)
                report["stock_decode_step"] = ran.get("step", "eager")
                report["stock_decode_tok_s"] = _median(s_stock)
                report["stock_decode_samples"] = s_stock
                s_stock_pre = timed_prefill(model, prefill_ids)
                report["stock_prefill_tok_s"] = _median(s_stock_pre)
                report["stock_prefill_samples"] = s_stock_pre
                s_stock_ttft = timed_ttft(model, prefill_ids)
                report["stock_ttft_s"] = _median(s_stock_ttft, 4)
                report["stock_ttft_samples"] = s_stock_ttft
                record_host_load(report, "stock", load0)
            except BaseException as e:  # noqa: BLE001 — re-raised unless it is a load failure
                if not stock_load_failure(e):
                    raise
                # The observed failure IS the result for this arm: keep the
                # message, drop any half-measured stock field, free the
                # device, and let the twin/compressed passes run — the
                # compressed arm is the point of a machine the stock arm does
                # not fit. Before this, one OOM took the whole run down.
                model = None
                free_all()
                for k in [k for k in report if (k.startswith("stock_") and k != "stock_loader")
                          or k == "vram_bf16_bytes"]:
                    del report[k]
                report.get("host_load", {}).get("load1", {}).pop("stock", None)
                routes.pop("stock", None)  # no stock arm ran: nothing to hold the others to
                report["stock_outcome"] = "failed_load"
                report["stock_error"] = f"{type(e).__name__}: {e}"
                print(f"drinkme: stock arm failed to load ({type(e).__name__}); recorded as "
                      "failed_load, continuing with the twin/compressed passes", flush=True)
            else:
                report["stock_outcome"] = "measured"
                if spec:  # after the arm's timings, outside the load-failure net above
                    _speculate(spec_results, model, "stock", model_id, tok, snap, None)
                model = None
                free_all()
        else:
            # The fit-point path. bench.py's own fit check already
            # decided the naive load would not fit — stock_* fields are
            # OMITTED (not null; matches the lexicon's fit-point shape, see
            # bench.py's record-building) rather than attempted and crashed.
            report["stock_outcome"] = "skipped_predicted_nonfit"
            report["stock_error"] = ("skipped by bench's stock-arm fit check (predicted OOM "
                                     f"under the {stock_loader} loader's transient)")
            if stock_bf16_bytes is not None:
                report["bandwidth"]["total_param_bytes_bf16"] = int(stock_bf16_bytes)
                report["bandwidth"]["ceiling_tok_s_bf16"] = ceiling_tok_s(
                    report["bandwidth"]["read_bytes_s"], stock_bf16_bytes)

        # --- pass 2: resident-compressed ---
        # STREAMED: each eligible Linear is packed
        # (pack_weight_served, the writer's own call — the CPU round trip is
        # the per-tensor GATE inside it and raises before the tensor is
        # installed) and put on the device as its tensor is read; the raw
        # remainder follows through the one walker. The bf16 model never
        # exists whole on the host, which is what lets the 72B fit point
        # (145 GB bf16, 102.71 GB compressed) be taken on a 122 GB machine.
        # refuse_unless_verified stays right after the load, before any
        # timing — the same place in the flow it has always had.
        if compressed_bytes is not None:
            refuse_if_compressed_wont_fit(model_id, compressed_bytes, stock_memory_kind,
                                          live_available_bytes(stock_memory_kind),
                                          tensor_bytes=stock_tensor_bytes)
        if pack_dir is not None:
            # radix spike: the artifact off the disk, through the serve loader
            model, stats, pack_meta = load_pack_compressed(model_id, revision, pack_dir,
                                                           device="cuda", snap=snap)
        else:
            model, stats = load_compressed_streaming(model_id, revision, device="cuda", snap=snap)
        refuse_unless_verified(model_id, stats)
        report["compressed_routing"] = _route_arm(model, "compressed", routes)
        refuse_unless_routed_alike(model_id, routes)  # before the compressed arm is timed
        free_all()
        report["swapped_linears"] = len(stats)
        report["vram_compressed_bytes"] = vram_bytes()
        report["verified_tensors"] = len(stats)
        # what verified_tensors rests on: "roundtrip" for the in-memory arm
        # (each tensor freshly compared with its source), "pack" under
        # --pack-dir (the pack's hashes matched at load; the comparison was the
        # packer's) — the record says which, the count alone cannot
        report["verification"] = "pack" if pack_dir is not None else "roundtrip"
        if hasattr(model, "modules"):  # the tests' stubs are not nn.Modules
            report["compressed_bytes_per_token"] = decode_read_bytes(model, tower_paths(model, snap))
            report["compressed_linear_kinds"] = _module_kinds(model)
            # Whether this measurement's tree carries a vision tower — False
            # for a text-only architecture, True for a vision-capable one
            # (the bench's default, _TOWER_BY_DEFAULT), False
            # only when the caller asked for `vision=False`. The lexicon has
            # no typed field for this yet; it rides in raw,
            # which the lexicon declares `unknown`, so a record always says
            # plainly whether its weights/resident/bpw figures include the
            # tower rather than leaving a reader to infer it.
            report["vision_tower_included"] = tower_included(model, snap)
        load0 = load1()
        ran = {}
        _, s_comp = timed_decode(model, ids, step=decode_step(model), ran=ran)
        report["compressed_decode_step"] = ran.get("step", "eager")
        report["compressed_decode_tok_s"] = _median(s_comp)
        report["compressed_decode_samples"] = s_comp
        s_comp_pre = timed_prefill(model, prefill_ids)
        report["compressed_prefill_tok_s"] = _median(s_comp_pre)
        report["compressed_prefill_samples"] = s_comp_pre
        s_comp_ttft = timed_ttft(model, prefill_ids)
        report["compressed_ttft_s"] = _median(s_comp_ttft, 4)
        report["compressed_ttft_samples"] = s_comp_ttft
        record_host_load(report, "compressed", load0)
        if spec:
            _speculate(spec_results, model, "compressed", model_id, tok, snap, pack_dir)
        model = None
        free_all()

        # --- pass 3: the order-matched twin (the decode comparator) ---
        # AFTER the compressed arm, because it is built from that arm's
        # stats: each slot's codec and widths (twin_plan) decide its module
        # (RadixTwinLinear or RawLinear) and its launch-table row, and only
        # the compressed arm's pack knows them — which tensors fell back to
        # raw is data-dependent, and under --pack-dir the profile is the
        # pack's. Re-packing here to learn them would double the CPU pack
        # pass and its ~10x-a-tensor host scratch.
        # Decode only, as before: the twin now has the compressed arm's
        # routes at every M (dense F.linear over its raw weight at M >= 9),
        # but its prefill would be stock's F.linear, and no metric is
        # published for it.
        # The twin holds the SAME bf16 bytes as stock, so where a single bf16
        # copy does not fit the budget (RX 7600 XT 16 GB: the 8B fit
        # point — stock fit check routed correctly, then the TWIN pass OOM'd
        # the 16 GB card the same way) there is no twin either: a fit point
        # is compressed-only. `run_twin` is bench's single-copy fit check.
        if run_twin:
            # STREAMED: each slot's module is built from its checkpoint
            # tensor as it is read — one bf16 copy resident, one tensor of
            # host transient — never load_cpu's whole-model host stage
            # followed by a swap and a `.cuda()`.
            model, _ = load_twin_streaming(model_id, revision, device="cuda", snap=snap,
                                           plan=twin_plan(stats))
            report["twin_routing"] = _route_arm(model, "twin", routes)
            refuse_unless_routed_alike(model_id, routes)  # before the twin is timed
            report["vram_twin_bytes"] = vram_bytes()  # the twin's footprint: twin_weights_gb
            if isinstance(model, torch.nn.Module):  # the tests' stubs are not
                # the twin's bandwidth bound: read_gb_s / twin_decode_read_gb
                report["twin_bytes_per_token"] = decode_read_bytes(model, tower_paths(model, snap))
            free_all()
            load0 = load1()
            ran = {}
            _, s_twin = timed_decode(model, ids, step=decode_step(model), ran=ran)
            report["twin_decode_step"] = ran.get("step", "eager")
            record_host_load(report, "twin", load0)
            report["twin_decode_tok_s"] = _median(s_twin)
            report["twin_decode_samples"] = s_twin
            report["twin_outcome"] = "measured"
            model = None
            free_all()
        else:
            report["twin_decode_tok_s"] = None
            report["twin_decode_samples"] = []
            report["twin_outcome"] = "skipped_predicted_nonfit"  # a prediction, named as one
            report["twin_error"] = "twin skipped: a single bf16 copy does not fit the budget (fit point is compressed-only)"

    from . import spec_pass

    report["spec"] = (spec_pass.assemble(spec_results, DECODE_REPS) if spec
                      else {"skipped": spec_pass.SKIP_FLAG})
    report["compression_profile"] = arm_compression_profile(stats)  # off the arm, never a literal
    report["mean_bpw"], report["weighted_bpw"] = bpw_stats(stats)
    if pack_meta is not None:
        if report["compression_profile"] != pack_meta.get("profile"):
            raise ValueError(f"the loaded arm says {report['compression_profile']!r} but the pack's "
                             f"meta.json says {pack_meta.get('profile')!r}")
        report["pack"] = pack_record_fields(pack_meta)
        report["pack"]["fallback_tensors"] = sum(1 for st in stats if st.get("codec") == "raw")
    return report


# ---------------- the speculation pass (spec_pass.py: what it is and why) ----


class _spec_env:
    """DRINKME_SPEC set to `mode` for the pass, the caller's value (or its
    absence) back afterwards: `drinkme serve --spec` writes the same
    variable, and the head load (ngram.wants_head) and the engine's first
    request (mtp.spec_plan) both read it."""

    def __init__(self, mode: str):
        self.mode = mode

    def __enter__(self):
        self.before = os.environ.get("DRINKME_SPEC")
        os.environ["DRINKME_SPEC"] = self.mode

    def __exit__(self, *exc):
        if self.before is None:
            os.environ.pop("DRINKME_SPEC", None)
        else:
            os.environ["DRINKME_SPEC"] = self.before


def _diet_in_memory(head, device: str) -> int:
    """The head diet for the in-memory compressed arm: `drinkme pack` writes
    the head's eligible Linears into the pack's mtp/ sub-pack (codec/pack.
    _HeadPacker, the same _pack_one as the trunk) and `drinkme serve` loads
    them from there compressed (mtp.install_head_pack), so the arm that
    re-packs its trunk in memory packs its head the same way — swap.
    make_compressed over each tensor, the round trip gate inside — and
    times the head serve would load. Off under DRINKME_MTP_DIET=0, as
    serve's is. Returns how many Linears were packed."""
    from .codec.swap import make_compressed, swap_linears
    from .serving import mtp

    if not mtp.diet_enabled():
        return 0
    return len(swap_linears(head, lambda w, bias: make_compressed(w, bias, device), device))


def _sync(device) -> None:
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize()


def _spec_rep(engine, text: str, n_tokens: int) -> dict:
    """One greedy request through the engine, timed as bench/ngram_gpu_ab.py
    times it: decode tok/s is the tokens after the first over the time from
    the first delta to the end, so the prompt's prefill is not in it. The
    engine's own Speculator.stats() (engine.last_spec_stats) says the head
    proposed; a request it served any other way is refused rather than
    recorded as a speculation sample."""
    from . import spec_pass
    from .serving.engine import GenerationRequest, SampleParams, complete

    first = None

    def on_delta(_text):
        nonlocal first
        if first is None:
            first = time.perf_counter()
        return True

    req = GenerationRequest([{"role": "user", "content": text}],
                            SampleParams(temperature=0.0, max_tokens=n_tokens))
    _sync(engine.device)
    r = complete(engine, req, on_delta)
    _sync(engine.device)
    end = time.perf_counter()
    stats = engine.last_spec_stats
    if not stats or stats.get("mode") != spec_pass.MODE:
        raise ValueError(f"the speculation pass asked for {spec_pass.MODE!r} and the engine "
                         f"served {(stats or {}).get('mode', 'serially')!r}")
    if r.completion_tokens < 2 or first is None or end <= first:
        raise ValueError(f"the speculation pass decoded {r.completion_tokens} token(s): "
                         "no decode rate to time")
    return {"tok_s": round((r.completion_tokens - 1) / (end - first), 2),
            "prompt_tokens": r.prompt_tokens, "tokens": r.completion_tokens,
            "cycles": stats["cycles"], "accepted": stats["accepted"]}


def time_speculation(model, arm: str, model_id: str, tok, snap: str,
                     pack_dir: str | None = None, reps: int = DECODE_REPS,
                     device: str = "cuda") -> dict:
    """The speculation pass over one arm's loaded tree (spec_pass.py),
    after its plain timings. The head is the one `drinkme serve` would load
    for this arm, by the decision serve makes: engines._mtp_head under
    DRINKME_SPEC=mtp (ngram.wants_head, the residency guard, mtp.load_head:
    none for a family or checkpoint without one), then ngram.resolve_mode
    over whether it loaded. The stock arm's is the raw bf16 head (serve
    --stock), the compressed arm's the pack's under --pack-dir and the same
    head packed in memory otherwise (_diet_in_memory). The engine is
    serve's HFEngine over the arm's own model, at its default context,
    prefix cache off (each rep prefills cold), thinking off (the template
    ngram_gpu_ab measured with).

    Returns {"skipped": spec_pass.SKIP_NO_HEAD} when no head loaded, else
    {"ctx", "prompts": {name: {sha256, tokens}}, "timed": {name: {samples,
    accepted_per_step}}}: `samples` each timed rep's tok/s, and
    `accepted_per_step` the drafts the head proposed that verify accepted,
    over the verify steps, pooled across the timed reps."""
    from . import spec_pass
    from .serving import engines, gen_config, ngram

    if not isinstance(model, torch.nn.Module):  # the tests' stubs: no tree to load a head onto
        return {"skipped": spec_pass.SKIP_NO_HEAD}
    t0 = time.perf_counter()
    with _spec_env(spec_pass.MODE):
        head = engines._mtp_head(model, snap, None, device, pack_dir)
        if ngram.resolve_mode(head is not None) != spec_pass.MODE:
            return {"skipped": spec_pass.SKIP_NO_HEAD}
        if arm == "compressed" and pack_dir is None:
            n = _diet_in_memory(head, device)
            print(f"  {arm} arm: MTP head diet: {n} Linears packed in memory, as the pack's "
                  "mtp/ would carry them" if n else f"  {arm} arm: MTP head diet off "
                  "(DRINKME_MTP_DIET=0): the raw head", flush=True)
        engine = engines.HFEngine(model, tok, model_id=model_id, arm=arm, meta={},
                                  ctx=engines._ctx(model.config, None),
                                  template_kwargs={"enable_thinking": False},
                                  mtp_head=head, prefix_slots=0,
                                  gen_defaults=gen_config.load(model, snap))
        print(f"  {arm} arm: speculation pass ({spec_pass.MODE}, ctx {engine.ctx}): "
              f"{len(spec_pass.BENCH_PROMPTS)} prompts x {reps} timed reps"
              f"{' after one untimed' if WARMUP_REP else ''}, {spec_pass.TOKENS} tokens each",
              flush=True)
        prompts, timed = {}, {}
        for name, key in spec_pass.BENCH_PROMPTS.items():
            text = spec_pass.PROMPTS[key]
            if WARMUP_REP:
                _spec_rep(engine, text, spec_pass.TOKENS)  # untimed: compiles, first-touch
            runs = [_spec_rep(engine, text, spec_pass.TOKENS) for _ in range(reps)]
            cycles = sum(r["cycles"] for r in runs)
            prompts[name] = {"sha256": spec_pass.sha256(text), "tokens": runs[0]["prompt_tokens"]}
            timed[name] = {"samples": [r["tok_s"] for r in runs],
                           "accepted_per_step": (round(sum(r["accepted"] for r in runs) / cycles, 3)
                                                 if cycles else 0.0)}
            print(f"  {arm} speculation {name}: {timed[name]['samples']} tok/s, "
                  f"{timed[name]['accepted_per_step']} accepted/step "
                  f"({prompts[name]['tokens']}-token prompt)", flush=True)
        ctx = engine.ctx
    print(f"  {arm} arm: speculation pass took {time.perf_counter() - t0:.0f} s", flush=True)
    return {"ctx": ctx, "prompts": prompts, "timed": timed}


def _speculate(results: dict, model, arm: str, model_id: str, tok, snap: str,
               pack_dir: str | None) -> None:
    """run_arms' call of time_speculation for a measured arm, into
    `results[arm]`. Running out of memory is this pass's outcome, never the
    arm's: its plain metrics are already measured, so the arm keeps them and
    the pass records why it has none (stock_load_failure's test); anything
    else is a bug and raises. The allocator's cache goes first, so the head's
    residency guard reads the device as serve's does right after its load."""
    free_all()
    try:
        results[arm] = time_speculation(model, arm, model_id, tok, snap, pack_dir)
    except BaseException as e:  # noqa: BLE001 — re-raised unless it ran out of memory
        if not stock_load_failure(e) or isinstance(e, SystemExit):
            raise
        results[arm] = {"skipped": f"{type(e).__name__}: {e}"[:512]}
        print(f"drinkme: the {arm} arm's speculation pass ran out of memory "
              f"({type(e).__name__}); its plain metrics stand", flush=True)
    finally:
        free_all()


def _module_kinds(model) -> dict:
    """{class name: count} over the swapped Linears — which module class
    served each tensor (RadixCompressedLinear vs RawLinear in a pack with
    raw fallbacks)."""
    out: dict = {}
    for m in model.modules():
        if linear_kind(m) in ("drinkme_codec", "drinkme_raw", "drinkme_twin"):
            out[type(m).__name__] = out.get(type(m).__name__, 0) + 1
    return out


def _install_swapped(model, name: str, w: torch.Tensor, device: str, make, stats: list) -> None:
    """Replace the Linear at `name` on a meta skeleton with `make(w, None,
    device)`'s module for weight `w` — swap.make_compressed (pack_weight_served
    -> make_module, verified inside) or swap.make_twin (RadixTwinLinear / RawLinear).
    bias=None here: the bias streams from the checkpoint afterwards, exactly
    as the serve loader's bias-after-swap (engines.stream_checkpoint attaches
    it to any swap.SWAPPED_LINEAR_TYPES module). A stat, when `make` returns
    one, is named and appended."""
    parent_path, _, child = name.rpartition(".")
    parent = model.get_submodule(parent_path) if parent_path else model
    old = getattr(parent, child)
    if not isinstance(old, torch.nn.Linear):
        raise ValueError(f"{name} is not a Linear in this config — refusing to install a "
                         "swapped module there")
    mod, stat = make(w, None, device)
    setattr(parent, child, mod)
    if stat is not None:
        stat["name"] = name
        stats.append(stat)


def pack_stat(name: str, pack: dict) -> dict:
    """make_compressed's stat shape, for a tensor read back OFF A PACK DIR
    (iter_pack_dir's raw dict: its scalars blob carries R, C, bpw, the
    formatVersion echoed from meta.json as format_version, the codec/profile). `bits`
    is bpw x numel exactly as make_compressed forms it. Where
    make_compressed's stat says `verified` — that tensor was encoded and
    compared with its source bits just now — this one says `pack_verified`: the
    pack's hashes matched at load (engines.build_compressed_model raises on a
    mismatch before any tensor is installed), which is trust in the
    pack-time comparison, not a fresh one (docs/pack-format.md,
    "Verification"). Pure Python."""
    R, C = int(pack["R"]), int(pack["C"])
    bpw = float(pack["bpw"])
    return {
        "name": name,
        "shape": [R, C],
        "numel": R * C,
        "bits": int(round(bpw * R * C)),
        "bpw": round(bpw, 4),
        "format_version": pack["format_version"],
        "dtype": pack.get("dtype"),      # None: the bf16 codec's tensors carry no dtype scalar
        "codec": pack.get("codec"),      # "radix" | "raw"
        "profile": pack.get("profile"),  # its radix profile, else None
        "widths": [int(x) for x in pack["widths"]] if pack.get("widths") is not None else None,
        "pack_verified": True,  # the pack's hashes matched at load; not a fresh source comparison
    }


def load_pack_compressed(model_id: str, revision: str | None, pack_dir: str,
                         device: str = "cuda", snap: str | None = None):
    """The bench's compressed arm OFF A PACK DIR (`drinkme bench
    --pack-dir`): the artifact `drinkme pack` wrote, through the loader
    `drinkme serve` runs (engines.build_compressed_model: hash gate, bound
    snapshot, every packed tensor installed by swap.make_module —
    RadixCompressedLinear / RawLinear by the tensor's own scalars — and the
    raw remainder streamed). Nothing is re-packed in memory, so the sip and
    gulp arms are two packs through one loader. `snap`: the pack's bound
    snapshot when the caller already resolved it (run_arms, for every
    arm); None resolves it in the loader (resolve_pack_source).
    Returns (model, stats, pack_meta)."""
    from .serving.engines import build_compressed_model

    stats: list[dict] = []
    model, _cfg, _snap, pack_meta = build_compressed_model(
        model_id, revision, pack_dir, device, snap=snap,
        on_tensor=lambda name, pack: stats.append(pack_stat(name, pack)))
    return model, stats, pack_meta


# the arrays a radix Linear's decode step reads:
# the payload stream, and the directory + padded palette + the per-block
# schedule (empty for sip) — swap.to_device_radix's resident dict
_CODEC_PLANE_KEYS = ("rx_data",)
_CODEC_TABLE_KEYS = ("rx_offsets", "rx_palette", "rx_schedule")


def linear_kind(mod) -> str:
    """drinkme_codec | drinkme_raw | drinkme_twin | bf16 | other, by
    duck-typing. drinkme_raw is the pack's raw fallback (swap.RawLinear): a
    pack tensor served as its bf16 bits."""
    p = getattr(mod, "p", None)
    if isinstance(p, dict) and any(k in p for k in _CODEC_PLANE_KEYS):
        return "drinkme_codec"
    if isinstance(p, dict) and p.get("codec") == "raw" and torch.is_tensor(p.get("weight")):
        return "drinkme_raw"
    if isinstance(p, dict) and p.get("codec") == "twin" and torch.is_tensor(p.get("weight")):
        return "drinkme_twin"
    w = getattr(mod, "weight", None)
    if isinstance(mod, torch.nn.Linear) and w is not None:
        return "bf16"
    return "other"


def decode_read_bytes(model, skip: tuple[str, ...] = ()) -> dict:
    """Bytes ONE decode step reads from the weights, by kind, off the
    RESIDENT tree: a radix Linear reads its payload stream and the
    directory, palette and schedule whole; the order-matched twin its raw
    bf16 weight; a plain bf16 Linear — and a pack's raw-fallback tensor,
    which IS one — two bytes a weight, under raw_linear_bytes; and each
    token embedding table one row (embedding_row_bytes: a step looks up
    one token; a tied lm_head is a Linear and reads the whole table).
    Modules under `skip` — the vision tower's subtrees (tower_paths): a
    text decode step never runs them — are not counted, and neither are
    the unused embedding rows, norms or the KV cache: those count toward
    memory (`<arm>_weights_gb`), not toward what a step reads.
    `total_bytes` x decode tok/s is the achieved read rate, to hold
    against the probe's read_bytes_s (bench.bound_fractions)."""
    out = {"packed_bytes": 0, "raw_linear_bytes": 0, "embedding_row_bytes": 0, "counts": {}}
    prefixes = tuple(p + "." for p in skip)
    for name, mod in model.named_modules():
        if name in skip or name.startswith(prefixes):
            continue
        if isinstance(mod, torch.nn.Embedding):
            out["embedding_row_bytes"] += mod.weight.shape[1] * mod.weight.element_size()
            continue
        kind = linear_kind(mod)
        if kind == "other":
            continue
        out["counts"][kind] = out["counts"].get(kind, 0) + 1
        if kind == "drinkme_codec":
            for k in _CODEC_PLANE_KEYS + _CODEC_TABLE_KEYS:
                v = mod.p.get(k)
                if torch.is_tensor(v):
                    out["packed_bytes"] += v.numel() * v.element_size()
        elif kind == "drinkme_twin":
            out["packed_bytes"] += mod.p["weight"].numel() * mod.p["weight"].element_size()
        elif kind == "drinkme_raw":
            out["raw_linear_bytes"] += mod.p["weight"].numel() * mod.p["weight"].element_size()
        else:
            out["raw_linear_bytes"] += mod.weight.numel() * mod.weight.element_size()
    out["total_bytes"] = out["packed_bytes"] + out["raw_linear_bytes"] + out["embedding_row_bytes"]
    if skip:
        out["skipped"] = list(skip)
    return out


def tower_paths(model, snap) -> tuple[str, ...]:
    """The vision tower's subtrees in this measured tree (vision_tower_paths
    for the snapshot's own config, as tower_included judges it), the ones
    the tree holds; () for a text-only tree. decode_read_bytes skips them."""
    from transformers import AutoConfig

    cfg = (AutoConfig.from_pretrained(snap) if os.path.isfile(os.path.join(str(snap), "config.json"))
           else model.config)
    held = []
    for path in vision_tower_paths(cfg) or ():
        try:
            if model.get_submodule(path) is not None:
                held.append(path)
        except AttributeError:  # absent, or pruned to None
            continue
    return tuple(held)

