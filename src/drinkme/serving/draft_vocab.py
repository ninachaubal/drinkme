"""The draft diet, lever 1: let a draft step argmax over a SUBSET of the vocab.

A speculative cycle's read budget is not just the trunk. Each
draft step runs the shared vocab projection, and on the 27B that is
`lm_head` — 248,320 x 5,120, 2.54 GB bf16 / ~1.9 GB packed — so a depth-4
cycle reads it FOUR times for the drafts alone, on top of once for the verify.
Roughly a third of the cycle's bytes go to proposing tokens, not to deciding
them.

THE LEVER, and why it is lossless: a draft is a PROPOSAL. The verify pass runs
the full lm_head over every row and picks with the engine's own sampler, so
the emitted stream is the serial greedy stream whatever the draft proposed. A
draft that cannot reach a token simply drafts something else, the verify pass
rejects it, and the cycle loses its tail. That costs ACCEPTANCE — measurable,
and expected to be small because the tokens outside a top-K subset are the
rare ones a draft head misses anyway — and it cannot cost a token.

So: pick K rows of the projection once, read only those. Off by default.

    DRINKME_MTP_DRAFT_VOCAB=<K>     the K lowest token ids, plus the config's
                                    special ids. Zero setup, and CRUDE: it
                                    leans on BPE id order being a rough
                                    frequency order, which is true-ish and
                                    unmeasured.
    DRINKME_MTP_DRAFT_VOCAB=<path>  a JSON file {"ids": [...]} built from the
                                    model's own output statistics —
                                    bench/draft_vocab_build.py, which also
                                    measures held-out coverage (what fraction
                                    of real argmax picks the subset contains).

ON THE COMPRESSED PATH THE LEVER IS DORMANT: no codec has a row-subset
kernel — the capability registry (codec/registry.py, ROW_SUBSET rows all
False) says so, and make_projection turns that into the
documented fallback: one line, the full-vocabulary head kept. When a radix
row-subset kernel lands it earns its registry row and a projection class
here; gathering K rows into a private pack is NOT the way (K/R of the pack
again in resident memory — ~250 MB at K=32k on the 27B — the fit check paying
for a speed lever). The raw-`nn.Linear` fallback below DOES gather, because
there is no row-selecting matmul to hand it to; that path is for tied/stock
heads and its cost is stated at build time.
"""

from __future__ import annotations

import json
import os

import torch


def _special_ids(config) -> list[int]:
    """EOS/BOS/PAD from the config. A draft that cannot propose end-of-turn
    would mispredict the last token of every generation — cheap to prevent,
    and the tokenizer is not in scope here (the head loader has no tokenizer;
    the builder script, which does, writes them into its file too)."""
    text = config.get_text_config() if hasattr(config, "get_text_config") else config
    out: list[int] = []
    for attr in ("eos_token_id", "bos_token_id", "pad_token_id"):
        v = getattr(text, attr, None) or getattr(config, attr, None)
        if isinstance(v, int):
            out.append(v)
        elif isinstance(v, (list, tuple)):
            out += [int(i) for i in v if isinstance(i, int)]
    return out


def ids_from_env(config, vocab_size: int, warn) -> tuple[torch.Tensor, str] | None:
    """DRINKME_MTP_DRAFT_VOCAB -> (sorted unique ids, provenance) or None.

    Never fatal and never silent: a typo must not take the server down, and it
    must not quietly change decode either — it says so and serves the full
    vocab, which is the default behaviour exactly."""
    raw = os.environ.get("DRINKME_MTP_DRAFT_VOCAB", "").strip()
    if not raw:
        return None
    if raw.lstrip("+-").isdigit():
        k = int(raw)
        if k <= 0:
            return None
        if k >= vocab_size:
            warn("dvbig", f"DRINKME_MTP_DRAFT_VOCAB={k} >= vocab {vocab_size} — "
                          "nothing to cut, drafting over the full vocab")
            return None
        ids = list(range(k))
        source = (f"first {k} token ids + config specials (CRUDE: assumes BPE id "
                  f"order tracks frequency; build a measured subset with "
                  f"bench/draft_vocab_build.py)")
    else:
        try:
            with open(os.path.expanduser(raw)) as f:
                blob = json.load(f)
            ids = [int(i) for i in blob["ids"]]
        except (OSError, ValueError, KeyError, TypeError) as e:
            warn("dvfile", f"DRINKME_MTP_DRAFT_VOCAB={raw!r}: {type(e).__name__}: {e} "
                           "— drafting over the full vocab")
            return None
        if not ids:
            warn("dvempty", f"{raw} lists no ids — drafting over the full vocab")
            return None
        source = f"{raw} ({blob.get('method', 'unknown method')})"
    ids = sorted({i for i in ids + _special_ids(config) if 0 <= i < vocab_size})
    return torch.tensor(ids, dtype=torch.long), source


class SubsetDense:
    """The same restriction for a raw `nn.Linear` head (tied, or the stock
    arm): there is no row-selecting matmul to hand it to, so this GATHERS the
    K rows. That is K/V of the head duplicated in resident memory — stated at
    build time, and the reason the compressed path does it the other way."""

    def __init__(self, parent, ids: torch.Tensor):
        w = parent.weight
        self.ids = ids.to(w.device)
        self.weight = w[self.ids].contiguous()
        self.bias = None if parent.bias is None else parent.bias[self.ids].contiguous()
        self.R = w.shape[0]

    @property
    def read_ratio(self) -> float:
        return self.ids.numel() / self.R

    def __call__(self, x):
        if x.numel() // x.shape[-1] != 1:
            raise ValueError("SubsetDense is the DRAFT projection and takes one row")
        return torch.nn.functional.linear(x, self.weight, self.bias)


def _pack_device(p: dict):
    """Where a runtime pack lives: any tensor in it says (they all share it);
    read that way rather than by a key one codec carries and another does
    not — a fixed key is exactly the lookup that would raise here."""
    return next(v.device for v in p.values() if isinstance(v, torch.Tensor))


def make_projection(lm_head, ids: torch.Tensor):
    """(projection, note) for whatever kind of head this is, or (None, why).

    A CompressedLinear is asked of the capability registry FIRST
    (codec/registry.py): a packed head whose codec has no
    row-subset kernel for this kernel family — every codec, on every kernel,
    today — gets (None, why) here, which install_draft_vocab
    turns into the documented fallback: one line, the full-vocabulary head
    kept. (Building a projection unconditionally would die on a dict
    lookup in its constructor, and an explicit DRINKME_MTP_DEPTH=4 would
    propagate that out of the loader.)"""
    from ..codec import registry
    from ..codec.swap import CompressedLinear

    if isinstance(lm_head, CompressedLinear):
        kernel = registry.kernel_of(_pack_device(lm_head.p))
        if not registry.supported(lm_head.tensor_codec, lm_head.layout,
                                  registry.ROW_SUBSET, kernel):
            return None, registry.unsupported_note(
                lm_head.tensor_codec, lm_head.layout, registry.ROW_SUBSET, kernel)
        # a registry row flipped True without a projection class to serve it
        # is a bug in this file, not a served fallback
        raise NotImplementedError(
            f"codec/registry.py says {registry.describe(lm_head.tensor_codec, lm_head.layout)} has a "
            f"row-subset kernel on {kernel}, but serving/draft_vocab.py has no projection "
            "class for it")
    if isinstance(lm_head, torch.nn.Linear):
        p = SubsetDense(lm_head, ids)
        mb = p.weight.numel() * p.weight.element_size() / 1e6
        return p, (f"{ids.numel()} of {p.R} rows ({100 * p.read_ratio:.1f}%), "
                   f"GATHERED ({mb:.0f} MB resident — raw head, no row-selecting "
                   f"matmul to use)")
    return None, f"lm_head is a {type(lm_head).__name__}, not a supported projection"
