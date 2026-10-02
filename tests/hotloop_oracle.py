"""The hot loop's unoptimized reference — the equivalence oracle.

Both functions below are VERBATIM copies of the unoptimized
`serving/sampling.py:sample_next` and `serving/constrain.py:pick_token`,
with three kinds of edit, each marked ORACLE-EDIT:
the imports point at this module's copy instead of the package's, one type
annotation that referenced a package-private name is dropped, and status()
calls go to status_reference() — the recursive descent this file's vintage
had, so a comparison is old-path-end-to-end and not half the new one. They are here
so tests/test_hotloop_equivalence.py can run OLD and NEW over the same inputs
in the same process and compare — bitwise on logits, id-for-id on tokens.

DO NOT "improve" anything in this file. Slowness is the point: it is what the
new code has to agree with. If a change to serving/ makes a test here fail,
the new code is wrong until proven otherwise.
"""

from __future__ import annotations

from typing import Callable, Iterable

from drinkme.serving.constrain import COMPLETE, FAIL, JsonConstraint  # verdict names


def old_sample_next(logits, params, generator=None,  # ORACLE-EDIT: annotation
                prev_ids: Iterable[int] | None = None,
                gen_ids: Iterable[int] | None = None) -> int:
    """One decode step: logits [V] float -> token id.

    Order matters and is the HF/vLLM convention: repetition penalty on raw
    logits (over prev_ids = prompt + generated), then the additive OpenAI
    presence/frequency penalties (over gen_ids = generated only), then
    temperature, then top_k, then top_p, then sample. Determinism comes from
    the caller's seeded torch.Generator; temperature == 0 is greedy and
    ignores the generator entirely.
    """
    import torch  # local: keep serving importable without torch (see module docstring)

    logits = torch.as_tensor(logits).to(torch.float32).clone()
    if prev_ids is not None and params.repetition_penalty != 1.0:
        ids = sorted({int(i) for i in prev_ids})
        if ids:
            idx = torch.tensor(ids, dtype=torch.long)
            picked = logits[idx]
            # HF convention: DIVIDE positive logits, MULTIPLY negative — both
            # directions push a seen token down; a plain divide would push a
            # negative logit UP.
            logits[idx] = torch.where(picked > 0,
                                      picked / params.repetition_penalty,
                                      picked * params.repetition_penalty)
    if gen_ids is not None and (params.presence_penalty or params.frequency_penalty):
        counts: dict[int, int] = {}
        for i in gen_ids:
            counts[int(i)] = counts.get(int(i), 0) + 1
        if counts:
            ids = sorted(counts)
            idx = torch.tensor(ids, dtype=torch.long)
            cnt = torch.tensor([counts[i] for i in ids], dtype=torch.float32)
            # OpenAI semantics: presence is a flat tax on any seen token,
            # frequency scales with how often it appeared
            logits[idx] -= params.presence_penalty + params.frequency_penalty * cnt
    if params.temperature == 0:
        return int(torch.argmax(logits))
    logits = logits / params.temperature
    if 0 < params.top_k < logits.numel():
        kth = torch.topk(logits, params.top_k).values[-1]
        logits[logits < kth] = float("-inf")  # ties at the kth value survive (HF-same)
    if params.top_p < 1.0:
        srt, idx = torch.sort(logits, descending=True)
        probs = torch.softmax(srt, dim=-1)
        cum = torch.cumsum(probs, dim=-1)
        # cum - probs is the mass BEFORE each token: drop a token only when the
        # threshold was already reached without it, so the crossing token stays
        # and at least one token always survives.
        srt[cum - probs > params.top_p] = float("-inf")
        logits = torch.full_like(logits, float("-inf")).scatter(0, idx, srt)
    return int(torch.multinomial(torch.softmax(logits, dim=-1), 1, generator=generator))


def old_pick_token(logits, params, generator, prev_ids, gen_ids: list[int],
               eos_ids, decode: Callable[[list[int]], str],
               constraint: JsonConstraint) -> int:
    """One constrained decode step: sample through the normal pipeline, ban
    candidates whose decoded text goes FAIL, resample. EOS allowed iff the
    current text is COMPLETE. A candidate that adds no visible text (a
    special token decoding to nothing) is banned — invisible tokens would
    loop forever. Raises RuntimeError on the unreachable all-banned state."""
    import torch

    sample_next = old_sample_next  # ORACLE-EDIT: this module's copy

    logits = torch.as_tensor(logits).to(torch.float32).clone()
    cur = decode(gen_ids)
    while True:
        if not torch.isfinite(logits).any():
            raise RuntimeError(
                "json constraint dead-end: every token banned (validator bug — "
                "please report the schema and partial output)")
        tok = sample_next(logits, params, generator, prev_ids=prev_ids,
                          gen_ids=gen_ids)
        if tok in eos_ids:
            # ORACLE-EDIT: the recursive-descent verdict, so this really is
            # the old path end to end (old sampler, old parser, old decode).
            if constraint.status_reference(cur) == COMPLETE:
                return tok
        else:
            cand = decode(gen_ids + [tok])
            if len(cand) > len(cur) and constraint.status_reference(cand) != FAIL:  # ORACLE-EDIT
                return tok
        logits[tok] = float("-inf")


def old_penalties(logits, params, prev_ids=None, gen_ids=None):
    """The penalty block of old_sample_next, hoisted verbatim (ORACLE-EDIT:
    hoist only — not one character of the arithmetic is different) so a test
    can compare PROCESSED LOGITS bitwise instead of only the token they
    happened to choose."""
    import torch

    logits = torch.as_tensor(logits).to(torch.float32).clone()
    if prev_ids is not None and params.repetition_penalty != 1.0:
        ids = sorted({int(i) for i in prev_ids})
        if ids:
            idx = torch.tensor(ids, dtype=torch.long)
            picked = logits[idx]
            # HF convention: DIVIDE positive logits, MULTIPLY negative — both
            # directions push a seen token down; a plain divide would push a
            # negative logit UP.
            logits[idx] = torch.where(picked > 0,
                                      picked / params.repetition_penalty,
                                      picked * params.repetition_penalty)
    if gen_ids is not None and (params.presence_penalty or params.frequency_penalty):
        counts: dict[int, int] = {}
        for i in gen_ids:
            counts[int(i)] = counts.get(int(i), 0) + 1
        if counts:
            ids = sorted(counts)
            idx = torch.tensor(ids, dtype=torch.long)
            cnt = torch.tensor([counts[i] for i in ids], dtype=torch.float32)
            # OpenAI semantics: presence is a flat tax on any seen token,
            # frequency scales with how often it appeared
            logits[idx] -= params.presence_penalty + params.frequency_penalty * cnt
    return logits
