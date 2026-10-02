"""`drinkme bench`'s speculation pass, the torch-free half: the prompts it
times, its settings, the reasons it does not run, and the record's metrics
and summary lines off `raw.spec`. The half that runs a model is
arms.time_speculation.

The pass times decode with speculation on, as `drinkme serve` runs it for a
checkpoint with an MTP head (`--spec mtp`: the head proposing): greedy,
TOKENS new tokens per rep, one untimed warm-up per arm and prompt, then as
many timed reps as the plain decode (arms.DECODE_REPS). It runs on each
measured stock and compressed arm, on the arm's own loaded model after its
plain timings, and only when the engine loaded a head
(engines._mtp_head, the decision `drinkme serve` makes). The mlx engine does
not speculate (serving/engine_mlx.py), so a metal record carries none.

The prompts are shared with bench/ngram_gpu_ab.py, whose runs fed the
results site's speculation ranges before bench measured them. The classes
separate the cases a proposer can meet: an agent-shaped transcript with a
tool result re-sent verbatim (where n-gram lookup should win), a plain chat
turn (where it should be about neutral), a code edit, and one synthetic
repetitive prompt (the ceiling). Bench times the first three; the
repetitive one is ngram_gpu_ab's alone.
"""

from __future__ import annotations

import hashlib
import statistics

# Agent-shaped: a tool result re-sent verbatim, which is what n-gram speculation exists for.
_TOOL = "\n".join(
    f"  src/drinkme/serving/{n}.py   {200 + i * 37} lines   modified 2026-09-0{i % 7 + 1}"
    for i, n in enumerate(["http", "engine", "engines", "mtp", "ngram", "sampling",
                           "constrain", "template", "detok", "think", "tools",
                           "metrics", "slotstore", "capability", "tool_formats"]))
AGENT = (
    "Here is the output of `ls -l src/drinkme/serving`:\n\n" + _TOOL +
    "\n\nI ran it again after my edit and it is unchanged:\n\n" + _TOOL +
    "\n\nAnd once more, for the record:\n\n" + _TOOL +
    "\n\nWalk through that listing file by file and say, for each one, what "
    "you would expect it to contain. Repeat the file name and line count "
    "before each answer.")

# ngram_gpu_ab's names, in its default order
PROMPTS = {
    "agent-transcript": AGENT,
    "chat": "Write a detailed, multi-paragraph explanation of why the sky is blue.",
    "code": "Write a Python function that parses an ISO-8601 timestamp string into a "
            "datetime, handling optional fractional seconds and timezone offsets, with tests.",
    "repetitive": "Repeat the following line exactly 40 times, one per line, "
                  "with no other text: the quick brown fox jumps over the lazy dog.",
}

# The record's prompt names (the suffix of <arm>_spec_decode_tok_s_<prompt>)
# -> PROMPTS' keys, in the order bench runs and prints them.
BENCH_PROMPTS = {"agent": "agent-transcript", "chat": "chat", "code": "code"}

MODE = "mtp"   # the proposer: ngram.resolve_mode's name for the head alone
TOKENS = 256   # new tokens per rep
ARMS = ("stock", "compressed")  # the twin is a diagnostic of the weight read; no pass

# raw.spec.skipped: why the pass did not run (the lexicon's raw description names all three)
SKIP_NO_HEAD = "no MTP head"
SKIP_RUNTIME = "runtime does not speculate"
SKIP_FLAG = "--no-spec"


def metric_name(arm: str, prompt: str) -> str:
    return f"{arm}_spec_decode_tok_s_{prompt}"


def sha256(text: str) -> str:
    """raw.spec.prompts' identity for a prompt's text (UTF-8)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def median(samples: list) -> float:
    """A metric's value: the median of its timed reps, 2 places, as
    arms._median rounds every other decode metric."""
    return round(float(statistics.median(float(x) for x in samples)), 2)


def assemble(results: dict, reps: int) -> dict:
    """raw.spec from each measured arm's arms.time_speculation result
    ({arm: result}). The pass ran when any arm timed it:
    {mode, tokens, reps, ctx, prompts, arms}, with `skipped_arms` naming why
    a measured arm has no entry (its engine loaded no head, or the pass ran
    out of memory). When no arm timed it because no engine loaded a head,
    {"skipped": SKIP_NO_HEAD}."""
    timed = {arm: r for arm, r in results.items() if "timed" in r}
    skipped = {arm: r["skipped"] for arm, r in results.items() if "timed" not in r}
    if not timed and set(skipped.values()) <= {SKIP_NO_HEAD}:
        return {"skipped": SKIP_NO_HEAD}
    spec: dict = {"mode": MODE, "tokens": TOKENS, "reps": reps}
    first = next(iter(timed.values()), None)
    if first is not None:
        spec["ctx"] = first["ctx"]
        spec["prompts"] = first["prompts"]
    spec["arms"] = {arm: r["timed"] for arm, r in timed.items()}
    if skipped:
        spec["skipped_arms"] = skipped
    return spec


def metrics(spec: dict | None, stock_ran: bool) -> list[tuple[str, float, list]]:
    """(name, value, samples) for every arm and prompt raw.spec timed, in
    ARMS x BENCH_PROMPTS order: `stock_*` only when the stock arm was
    measured, nothing when the pass did not run. A timed arm missing a
    prompt is refused: the six names are all or none per arm."""
    out: list[tuple[str, float, list]] = []
    arms = (spec or {}).get("arms") or {}
    for arm in ARMS:
        if arm not in arms or (arm == "stock" and not stock_ran):
            continue
        for prompt in BENCH_PROMPTS:
            cell = arms[arm].get(prompt)
            if not cell or not cell.get("samples"):
                raise ValueError(f"raw.spec timed the {arm} arm without its {prompt!r} prompt")
            out.append((metric_name(arm, prompt), median(cell["samples"]), list(cell["samples"])))
    return out


def summary_lines(spec: dict | None, stock_ran: bool) -> list[str]:
    """What bench prints under an arm's decode line: one line per arm the
    pass timed (its median tok/s per prompt and its accepted drafts per
    verify step, averaged over the prompts), or why it did not run."""
    if not spec:
        return []
    if "skipped" in spec:
        return [f"speculation: skipped ({spec['skipped']})"]
    lines = []
    arms = spec.get("arms") or {}
    for arm in ARMS:
        if arm in arms and (arm != "stock" or stock_ran):
            cells = arms[arm]
            rates = " · ".join(f"{p} {median(cells[p]['samples']):.1f}" for p in BENCH_PROMPTS)
            per_step = statistics.mean(float(cells[p]["accepted_per_step"]) for p in BENCH_PROMPTS)
            lines.append(f"{arm} speculation ({spec['mode']}): {rates} tok/s, "
                         f"{per_step:.1f} accepted/step")
        elif arm in (spec.get("skipped_arms") or {}):
            lines.append(f"{arm} speculation: skipped ({spec['skipped_arms'][arm]})")
    return lines
