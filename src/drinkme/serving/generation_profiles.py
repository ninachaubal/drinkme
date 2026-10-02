"""Alias ids (`--served-model-name`, vLLM's `--served-model-name`) and
GENERATION PROFILES: named sampling/template-kwarg overlays (`--profile`,
borrowed from oMLX) exposed as `<id>:<profile>` on /v1/models, with
`drinkme.sampling.profile` naming the one an entry carries. Not the pack's
compression profile (sip / gulp — `compressionProfile` on the same entry,
`compression_profile` in code).

ONE engine. Aliases are extra accepted names for it (the default `model ==
eng.model_id` check stays true; this only widens what else is accepted — the
hard 404 on an unknown id is unchanged, 4xx-reason logging's reason line still fires).
Profiles are a named overlay of SampleParams sampling fields + chat-template
kwargs, applied per request on that SAME engine — no reload, no second
residency. Precedence, everywhere a value could come from: request field >
profile > generation_config.json default (gen_config.py) > SampleParams'
own OpenAI default.

THE THINKING HAZARD (bench/prefix_slots_verify.py's asymmetry note): a profile that flips `enable_thinking` between turns of ONE
conversation changes how the history re-renders — Qwen3.8 re-renders a prior
plain-content turn with an EMPTY `<think>\\n\\n</think>\\n\\n` pair when
thinking is on (so thinking-on history only extends the prefix cache when
reasoning_content is preserved) and Qwen3-8B does the mirror (its
enable_thinking=False is the shape that does NOT extend). A profile is
therefore something a CLIENT commits to for a conversation, not something to
toggle turn by turn — documented here, not enforced: there is no way to see
"this is the same conversation as three requests ago" from inside one POST.
"""

from __future__ import annotations

import json
import os

from .. import exitcodes
from . import gen_config


def served_names_from_env(explicit: list[str] | None) -> list[str]:
    """--served-model-name (CLI, repeatable/space-separated) or
    DRINKME_SERVED_MODEL_NAMES (comma- or space-separated); CLI wins when
    both are set — the standing precedent (engines.slots_from_env,
    engines._ctx). Neither set = no aliases, the default one id, unchanged."""
    if explicit:
        return list(explicit)
    raw = os.environ.get("DRINKME_SERVED_MODEL_NAMES", "").strip()
    if not raw:
        return []
    return [n for n in raw.replace(",", " ").split() if n]


def _coerce(v: str):
    """'true'/'false' -> bool, else int, else float, else the string
    itself — enough for every field a profile actually carries (sampling
    numbers, enable_thinking, reasoning_effort)."""
    low = v.lower()
    if low in ("true", "false"):
        return low == "true"
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        pass
    return v


def parse_generation_profile_flag(spec: str) -> tuple[str, dict]:
    """'name=key:val,key:val' -> (name, {key: coerced value})."""
    name, sep, body = spec.partition("=")
    if not sep or not name:
        raise ValueError(f"--profile {spec!r}: expected 'name=key:val,key:val'")
    overlay = {}
    for pair in body.split(","):
        if not pair:
            continue
        key, sep2, val = pair.partition(":")
        if not sep2 or not key:
            raise ValueError(f"--profile {spec!r}: {pair!r} is not 'key:val'")
        overlay[key] = _coerce(val)
    return name, overlay


def check_generation_profile(name: str, overlay: dict) -> None:
    """A generation profile's sampling fields through THE validator
    (engine.check_sampling) at boot, so a bad profile is refused by name
    before it can 400 every request that picks it — or, worse, pass the
    parser and fail in the sampler after the prefill. Non-sampling keys
    (enable_thinking, reasoning_effort) are template kwargs, not judged
    here. Raises SystemExit with the profile's name."""
    from .engine import SampleParamError, check_sampling

    if not isinstance(overlay, dict):
        # a malformed --profile flag or profiles.json entry: exitcodes.Usage
        raise exitcodes.Usage(f"drinkme: refusing to serve: profile {name!r} must be an object "
                              f"of key: value, not {type(overlay).__name__}")
    try:
        check_sampling({f: v for f, v in overlay.items() if f in gen_config.SAMPLING_FIELDS})
    except SampleParamError as e:
        raise exitcodes.Usage(f"drinkme: refusing to serve: profile {name!r}: {e}") from None


def generation_profiles_from_sources(explicit: list[str] | None, pack_dir: str | None) -> dict:
    """profiles.json beside the pack (if any) first, then --profile flags
    layered on top by name — CLI is the more specific ask, same precedent as
    served_names_from_env. {} when neither source names anything. Every
    profile's sampling fields are validated here (check_generation_profile): a
    profile the samplers could not take is refused at boot, by name."""
    out: dict = {}
    if pack_dir:
        path = os.path.join(pack_dir, "profiles.json")
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
            if isinstance(data, dict):
                out.update(data)
    for spec in explicit or ():
        name, overlay = parse_generation_profile_flag(spec)
        out[name] = overlay
    for name, overlay in out.items():
        check_generation_profile(name, overlay)
    return out


def resolve_model(requested: str, known_ids: set, generation_profiles: dict) -> tuple[str, str | None] | None:
    """`requested` -> (base id, generation profile name or None), or None when it names
    neither a known id nor a known '<id>:<profile>' pair — the caller 404s."""
    if requested in known_ids:
        return requested, None
    base, sep, prof = requested.rpartition(":")
    if sep and base in known_ids and prof in generation_profiles:
        return base, prof
    return None


def unknown_model_message(requested: str, known_ids: set, generation_profiles: dict) -> str:
    """4xx-reason logging's reason line, worth actually reading: every id this server will
    answer to, and one worked `<id>:<profile>` example if profiles exist."""
    ids = sorted(known_ids)
    msg = f"The model {requested!r} does not exist. Known: {', '.join(ids)}"
    if generation_profiles:
        msg += (f" (profiles: {', '.join(sorted(generation_profiles))}, e.g. "
                f"{ids[0]}:{sorted(generation_profiles)[0]})")
    return msg + "."


def effective_defaults(base_defaults: dict, generation_profile: dict) -> dict:
    """base_defaults (an engine's full effective table, generation_config.json
    already folded in) with a profile's sampling fields overlaid — a profile
    wins over generation_config.json, and a subsequent request field (applied
    by the caller via gen_config.resolve_sampling on THIS result) wins over
    the profile."""
    overlay = {f: generation_profile.get(f) for f in gen_config.SAMPLING_FIELDS}
    return gen_config.resolve_sampling(overlay, base_defaults)


def template_kwargs(generation_profile: dict) -> dict:
    """A generation profile's non-sampling keys (enable_thinking, reasoning_effort, ...)
    — everything that is not one of SampleParams' own fields."""
    return {k: v for k, v in generation_profile.items() if k not in gen_config.SAMPLING_FIELDS}
