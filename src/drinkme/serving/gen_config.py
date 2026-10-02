"""generation_config.json as the request-defaults source (vLLM borrow).
vLLM: "By default, the server
applies generation_config.json from the Hugging Face model repository if it
exists." Reading `model.generation_config` only for the EOS id would leave
the rest of the file unread — Qwen3.8's thinking/template defaults sit in
the checkpoint and would never apply.

EOS_IDS covers the EOS id itself: engines.py's
union still reads `model.generation_config.eos_token_id` too, and for the
compressed arm that object is the same manufactured-from-config.json
GenerationConfig this module exists to route around — gemma-4's config.json
carries only [1, 106] of the file's [1, 106, 50] (its `<|tool_response>`
end-of-calls token), and Qwen3-8B's config.json carries only its `<|im_end>`,
missing `<|endoftext|>`. `eos_ids` is read from THIS module's own file (int
or list, both shapes occur in the wild), normalised once, and unioned in on
top — so the two arms agree on the stop set by construction instead of by
accident of which config.json fields happen to overlap the real file.

SampleParams' own field defaults are the OpenAI ones; a checkpoint's
generation_config.json may override the SAMPLING fields per-field. Only a
field the file actually sets is applied — a repo silent on
`repetition_penalty` leaves SampleParams' own default in place, it is never
invented. A per-request field always wins over both.

Read from the FILE, not `model.generation_config`: engines.load_compressed
builds its model with `AutoModelForCausalLM.from_config` on the meta device,
which never sees the repo's generation_config.json at all — transformers
manufactures a bare default GenerationConfig from config.json instead
(`GenerationConfig.from_model_config`), so `model.generation_config` would
lie for exactly the arm that needs this most: it would report library
defaults (e.g. top_k=50) as if the checkpoint had asked for them. The file is
also strictly simpler to reason about — it holds only what the repo author
actually wrote, arbitrary keys included (`chat_template_kwargs`), where the
in-memory object mixes those with every field HF's GenerationConfig class
defaults regardless of the file.

`model.generation_config` is used only as a last-resort fallback when no file
can be found on disk (a load that genuinely has none) — via `to_diff_dict()`,
never `to_dict()`, because `to_dict()` returns the FULL class-default table
and would reintroduce exactly the false-positive this module exists to avoid;
`to_diff_dict()` returns only what differs from a bare `GenerationConfig()`,
which is the same "only what was actually asked for" property the file has.

DRINKME_IGNORE_GENERATION_CONFIG=1 restores OpenAI-default SAMPLING and
template behavior: generation_config.json's sampling fields and
chat_template_kwargs (and the model fallback for those) are not read.
eos_ids is exempt from the env var — it is not a sampling knob, and skipping
it would reopen that exact bug behind a flag documented as being about
temperature/top_p/top_k, silently diverging the compressed arm's stop set
from stock's again for anyone who sets it.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field

from .engine import SampleParamError, SampleParams, check_sampling

IGNORE_ENV = "DRINKME_IGNORE_GENERATION_CONFIG"

# generation_config.json keys this module understands, 1:1 with SampleParams'
# own field names. Present-only: a repo silent on a field leaves SampleParams'
# OpenAI default in place.
SAMPLING_FIELDS = ("temperature", "top_p", "top_k", "repetition_penalty",
                   "presence_penalty", "frequency_penalty")

# The one non-sampling key this file understands: an object of chat-template
# kwargs the checkpoint author recommends (e.g. {"enable_thinking": false}).
# Not a standard generation_config.json field on either target model —
# support is here because generation profiles need it and it costs nothing: an
# absent key is simply {}.
_TEMPLATE_KEY = "chat_template_kwargs"


@dataclass
class GenDefaults:
    """What one engine's generation_config.json (or its absence) resolved
    to, computed once at engine construction."""

    sampling: dict = field(default_factory=dict)        # SampleParams field -> value
    template_kwargs: dict = field(default_factory=dict)  # e.g. {"enable_thinking": False}
    eos_ids: tuple = ()                                  # sorted ints; () when absent
    source: str = "none read"                            # for the boot log


def _from_file(path: str) -> dict | None:
    """The raw JSON at `path`, or None on a missing/unreadable file — never
    raises, the same posture as every other load-time probe in this codebase
    (a broken diagnostic must not take the server down)."""
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError) as e:
        print(f"[drinkme] generation_config.json at {path} unreadable "
              f"({type(e).__name__}: {e}) — falling back", file=sys.stderr,
              flush=True)
        return None


def _from_model(model) -> dict:
    """model.generation_config, diffed against a bare GenerationConfig —
    only fields something actually set, not the class's blanket defaults.
    {} when the model cannot generate or carries nothing unusual."""
    gc = getattr(model, "generation_config", None)
    if gc is None or not hasattr(gc, "to_diff_dict"):
        return {}
    try:
        return gc.to_diff_dict()
    except Exception:  # noqa: BLE001 — a fallback that itself fails is still {}
        return {}


def _eos_ids(raw) -> tuple:
    """`eos_token_id` normalised to a sorted tuple of ints — int or list both
    occur in the wild (Qwen3-8B's file is a list, a single-EOS family's may be
    a bare int) — empty when the key is absent or neither shape."""
    v = raw.get("eos_token_id")
    if isinstance(v, int):
        return (v,)
    if isinstance(v, (list, tuple)):
        return tuple(sorted(int(e) for e in v))
    return ()


def load(model, snapshot_dir: str) -> GenDefaults:
    """`snapshot_dir`/generation_config.json -> request defaults. `model` is
    consulted only when no file is found there (see module docstring).

    `eos_ids` is read regardless of `DRINKME_IGNORE_GENERATION_CONFIG`: it is
    a structural identity between the stock and compressed arms, not a
    sampling default, and the env var's contract is "OpenAI sampling
    defaults," never "compressed's stop set may drift from stock's again."""
    path = os.path.join(snapshot_dir, "generation_config.json")
    raw = _from_file(path)
    source = path
    if raw is None:
        raw = _from_model(model)
        source = f"model.generation_config (no file at {path})"
    eos_ids = _eos_ids(raw)
    if not eos_ids:
        # A file that exists but carries no eos_token_id: transformers
        # REPLACES the stock arm's generation config with
        # the file's contents (eos -> None) while the compressed skeleton keeps
        # config.json's, so the two arms diverged. The fallback is config.json's
        # own eos_token_id — the same object both arms carry, and the same set
        # the no-file path already yields.
        eos_ids = _eos_ids(_from_file(os.path.join(snapshot_dir, "config.json")) or {})
    if os.environ.get(IGNORE_ENV) == "1":
        return GenDefaults(eos_ids=eos_ids,
                           source=f"ignored ({IGNORE_ENV}=1) — OpenAI sampling "
                                  "defaults, eos_ids still read")
    sampling = {f: raw[f] for f in SAMPLING_FIELDS if raw.get(f) is not None}
    # THE validator (engine.check_sampling) at load: a checkpoint file
    # asking for a value the samplers cannot take (top_k -1, top_p 0) would
    # otherwise 400 every request naming a field the client never sent.
    # Refused here instead, with the file and the way out.
    try:
        check_sampling(sampling)
    except SampleParamError as e:
        raise ValueError(f"{source}: {e} — fix the file, or {IGNORE_ENV}=1 serves the "
                         "OpenAI sampling defaults instead") from None
    tkw = raw.get(_TEMPLATE_KEY)
    return GenDefaults(sampling, dict(tkw) if isinstance(tkw, dict) else {},
                       eos_ids, source)


def effective_defaults(overrides: dict) -> dict:
    """Every SampleParams sampling field, `overrides` (typically an engine's
    GenDefaults.sampling, or a profile's sampling keys) beating
    SampleParams' own OpenAI default — the FULL table /v1/models advertises,
    not just the override subset."""
    return {f: overrides.get(f, getattr(SampleParams, f)) for f in SAMPLING_FIELDS}


def resolve_sampling(requested: dict, defaults: dict) -> dict:
    """One request's sampling fields, resolved. `requested`: {field: value}
    for whatever THIS wire dialect read off the request — a dialect with no
    such field (Anthropic has no presence_penalty) simply omits the key,
    which reads the same as an explicit JSON null: "whatever you'd otherwise
    pick." `defaults` is normally model_meta()["sampling"]["defaults"] (already
    the full effective table), so missing keys there still fall through to
    SampleParams' own default."""
    return {f: requested.get(f) if requested.get(f) is not None
            else defaults.get(f, getattr(SampleParams, f))
            for f in SAMPLING_FIELDS}
