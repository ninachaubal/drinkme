"""Runtime x model family: the ONE runtime-capability predicate.

Two RUNTIMES serve (serve.resolve_runtime: `--runtime`, DRINKME_RUNTIME, then
the host): `torch` (serving/engines.py — CUDA, ROCm, CPU) and `mlx`
(serving/engine_mlx.py — Apple silicon, docs/metal.md). They do not serve
the same families: the MLX engine serves dense Qwen3 text models only, and
refuses the hybrid DeltaNet lineage (Qwen3.5 / Qwen3.8 / qwen3_next) and
everything else by name. That refusal once
lived only inside the MLX loader, so the no-model picker (packs.build_
candidates) offered Qwen3.8-27B to a 48 GiB Mac and Qwen2.5-72B to a 120 GiB
one, and `serve --model` could download and pack a checkpoint the loader
would then refuse.

Every selection door reads THIS module and nothing else:
  - suggest.Model rows carry a `model_type`, so the menu can be filtered
    without a config on disk;
  - packs.filter_for_runtime drops unsupported candidates from the no-model
    pick and names each one in the summary;
  - serve.run refuses an explicitly requested unsupported model BEFORE any
    download or pack (refusal_for_repo, from config.json alone);
  - engine_mlx.refuse_unsupported is this predicate at the loader.

Torch-free and mlx-free by design: the picker runs before either is
imported. The tables here are the MLX engine's own (engine_mlx imports
them from here, so the two cannot drift). The torch runtime's family
support is transformers' (arms.skeleton): nothing is pre-filtered for it.
"""

from __future__ import annotations

TORCH, MLX = "torch", "mlx"
RUNTIMES = (TORCH, MLX)

# mlx-lm model_type -> the model class module the MLX engine serves. One
# entry, on purpose: adding a family means its packed Linears have been
# walked through engine_mlx's toy gates and its real checkpoint through the
# hardware verification in docs/metal.md. Dense Llama-shaped families would slot
# in the same way; the hybrids would not.
MLX_SUPPORTED_MODEL_TYPES = {"qwen3": "mlx_lm.models.qwen3"}

# Families the MLX engine knows the NAME of and refuses out loud. A
# model_type outside both tables is refused too, with a less specific line.
MLX_HYBRID_MODEL_TYPES = frozenset({"qwen3_5", "qwen3_5_text", "qwen3_next", "qwen3_5_moe"})

_MLX_SERVES = ("The mlx runtime serves DENSE Qwen3 text models (Qwen3-0.6B / 1.7B / 4B "
               "/ 8B, from radix bf16 packs) only.")


def model_type_of(config: dict | None) -> str | None:
    """The top-level `model_type` of a config.json dict (the wrapper's, for a
    composite — `qwen3_5`, not the nested text config's `qwen3_5_text` —
    which is what the MLX loader reads), or None when there is no config."""
    if not isinstance(config, dict):
        return None
    mt = config.get("model_type")
    return str(mt) if mt is not None else None


def refusal(runtime: str, model_type: str | None) -> str | None:
    """None when `runtime` serves `model_type`; else the reason (no name
    prefix — the caller names the model). `model_type` None means the
    family is not known to the caller (no config cached): the torch runtime
    lets the loader decide; the MLX runtime refuses rather than download
    on a guess, and says what would settle it."""
    if runtime == TORCH:
        return None
    if runtime != MLX:
        raise ValueError(f"unknown runtime {runtime!r}; runtimes: {', '.join(RUNTIMES)}")
    if model_type in MLX_SUPPORTED_MODEL_TYPES:
        return None
    if model_type is None:
        return (f"model family unknown (no config.json cached) — {_MLX_SERVES} "
                "Name a Qwen3 with --model, or fetch the config first.")
    if model_type in MLX_HYBRID_MODEL_TYPES:
        return (f"model_type {model_type!r} is a hybrid DeltaNet family (the Qwen3.5 / "
                "qwen3_next lineage, e.g. the 27B): its linear-attention layers "
                "run on the torch runtime's fla path and the mlx runtime has no "
                f"kernel for them. {_MLX_SERVES}")
    return (f"model_type {model_type!r} is not one the mlx runtime serves (it serves "
            f"{sorted(MLX_SUPPORTED_MODEL_TYPES)}). Dense Qwen3 text models only.")


def supported(runtime: str, model_type: str | None) -> bool:
    return refusal(runtime, model_type) is None


def refusal_for_repo(runtime: str, repo: str, revision: str | None,
                     config: dict | None = None) -> str | None:
    """`serve --model`'s door, before a download or a pack: `<repo>: <reason>`
    or None. The family comes from `config` when given, else the menu row
    for this repo (suggest.MODELS carries model_type), else config.json —
    the HF hub cache first, then the hub itself (kilobytes, never a weight
    file). A config that cannot be read at all falls toward the working
    path (the loader refuses later, by name) — an unmeasurable condition
    must not imitate a failed check."""
    if runtime == TORCH:
        return None
    mt = model_type_of(config)
    if mt is None:
        mt = _menu_model_type(repo)
    if mt is None:
        cfg = _config_json(repo, revision)
        if cfg is None:
            return None
        mt = model_type_of(cfg)
    reason = refusal(runtime, mt)
    return None if reason is None else f"{repo}: {reason}"


def _menu_model_type(repo: str) -> str | None:
    from .suggest import MODELS

    for m in MODELS:
        if m.hf_repo == repo and m.model_type:
            return m.model_type
    return None


def _config_json(repo: str, revision: str | None) -> dict | None:
    """config.json as a dict: a local directory, the hub cache, or one
    small hub download. None when none of those can be read."""
    import json
    import os

    if os.path.isdir(repo):
        p = os.path.join(repo, "config.json")
    else:
        try:
            from huggingface_hub import hf_hub_download, try_to_load_from_cache
        except ImportError:
            return None
        kw = {"revision": revision} if revision else {}
        p = try_to_load_from_cache(repo, "config.json", **kw)
        if not isinstance(p, str):
            try:
                from .detect import force_ipv4

                force_ipv4()
                p = hf_hub_download(repo, "config.json", **kw)
            except Exception:  # noqa: BLE001 — offline, gated, missing: the loader decides later
                return None
    try:
        with open(p) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError, TypeError):
        return None
