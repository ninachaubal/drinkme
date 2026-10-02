#!/usr/bin/env python3
"""N-gram speculation on the real box: identity, acceptance, decode tok/s.

The GPU half of the n-gram speculation gate. bench/ngram_proposals.py counts what the lookup
PROPOSES (exact on CPU, no model); this file measures what those proposals are
WORTH, which is bandwidth and therefore only measurable here.

Three arms per prompt, toggled per REQUEST on one engine build:

    off     the serial loop
    ngram   prompt lookup (serving/ngram.py)
    mtp     the checkpoint's MTP head — only when the checkpoint has one
    ngram+mtp   chained, likewise

and one gate plus one informational report:

  1. AGREEMENT (informational, never PASS/FAIL): greedy text per arm vs the
     `off` arm's text. Token-exact between speculation on and off is not a
     property this repo asserts — a batched verify forward and
     a serial single-token one can round differently in accumulation order,
     and a near-tie can flip an argmax for reasons that have nothing to do
     with a bug (bench/mtp_gpu_acceptance.py's note applies verbatim; never
     added to `fails` below). The ngram arm has a STRONGER prior than the
     MTP one — it does not touch prefill at all — so a fork here still
     deserves a second look, just not an auto-fail.
  2. SPEED (the gate): decode tok/s per arm, plus the engine's own acceptance
     and lookup counters, over prompt classes chosen to separate the cases:
     an agent-shaped transcript with a re-sent tool result (where lookup
     should win), a plain chat prompt (where it should be ~neutral), and one
     synthetic repetitive prompt (the ceiling).

ARM TOGGLE. DRINKME_SPEC is read once per ENGINE, into `_spec_plan`, and
dropped by the same `_mtp_depth = None` seam bench/mtp_gpu_acceptance.set_arm
uses — see that file's STALE-INSTRUMENT note, which is what this one is
written to not repeat. `set_spec` below drops BOTH, and the receipt records
the mode the engine actually reported per request, not the mode we asked for.

    # 8B, no MTP head: off vs ngram
    uv run --no-sync python bench/ngram_gpu_ab.py \\
        --model Qwen/Qwen3-8B \\
        --pack ~/.cache/drinkme/packs/Qwen--Qwen3-8B@b968826d9c46 \\
        --arms off,ngram --json /tmp/ngram_ab_8b.json

    # 27B with the head: all four
    uv run --no-sync python bench/ngram_gpu_ab.py \\
        --model Qwen/Qwen3.8-27B --revision 1d4bf0f2ff60 \\
        --pack ~/.cache/drinkme/packs/Qwen--Qwen3.8-27B@1d4bf0f2ff60 \\
        --arms off,mtp,ngram,ngram+mtp --json /tmp/ngram_ab_27b.json

    # 27B BF16, two prompts, three reps each on one engine build
    uv run --no-sync python bench/ngram_gpu_ab.py \\
        --model Qwen/Qwen3.8-27B --revision 1d4bf0f2ff60 --stock \\
        --arms off,mtp --prompts chat,code --reps 3 --json /tmp/ngram_ab_27b_bf16.json

REPS AND PROMPTS. `--prompts a,b` runs only those prompts (names from PROMPTS
below; default all four), in the order given. `--reps N` builds the engine
ONCE and runs every prompt x arm N times, interleaved (rep 1 of every arm,
then rep 2, ...) so a drift in the box lands on every arm alike. A cell keeps
the single-run receipt's fields, with `decode_toks_per_s` the MEDIAN over the
reps and `speedup` the ratio of medians; the other fields (tokens, spec,
audit, identity, ...) come from the run whose tok/s is the lower median.
Beside them, `decode_toks_per_s_runs` and `spread` (min, max, max-min over
the median) summarize the reps, and `runs` keeps every run's raw numbers, its
identity against the same rep's `off` run included. The BF16 27B's build
plus four prompts takes longer than ten minutes, so `--prompts` is also how a
run fits a command-time limit.

WARM-UP. Before the timed reps, each arm serves ONE untimed request (the first
prompt given, the full --tokens). The first request of an arm pays its Triton
compiles, which a timed run would count as decode time: 17.6 against 143 tok/s
on the 0.6B, 8.10 against about 9.2 for the 27B's MTP arm. A median over three
reps absorbs one such run, a single rep does not. The receipt's `warmup` names
the prompt and keeps each arm's untimed tok/s, so the compile cost stays
visible; no cell, median or speedup reads it.

VERDICT LINES GO IN THE OUTPUT, never in `$?` (AGENTS.md: TheRock ROCm torch
_exit(0)s over the real status once HIP initializes).
"""

import json
import os
import sys
import time

# The prompts live in the package (drinkme/spec_pass.py), so `drinkme bench`'s
# speculation pass times the same text: one copy.
from drinkme.spec_pass import PROMPTS

USAGE = ("usage: ngram_gpu_ab.py [--model REPO] [--revision SHA] [--pack DIR] "
         "[--stock] [--arms a,b] [--prompts a,b] [--reps N] [--tokens N] "
         "[--ctx N] [--ngram-min N] [--json PATH]")


def parse_args(argv: list[str]) -> dict:
    """The command line -> the run's settings. Pure: no environment, no
    imports, so the CPU suite can hold it (tests/test_ngram_gpu_ab_args.py).
    Raises SystemExit with the usage line on anything it does not know."""
    argv = list(argv)

    def opt(flag, default=None):
        if flag in argv:
            i = argv.index(flag)
            if i + 1 >= len(argv):
                raise SystemExit(f"{flag} needs a value\n{USAGE}")
            v = argv[i + 1]
            del argv[i:i + 2]
            return v
        return default

    def flag(name):
        if name in argv:
            argv.remove(name)
            return True
        return False

    cfg = {
        "model": opt("--model", "Qwen/Qwen3-8B"),
        "revision": opt("--revision"),
        "pack": os.path.expanduser(opt("--pack", "~/.cache/drinkme/packs/Qwen--Qwen3-8B@b968826d9c46")),
        "arms": opt("--arms", "off,ngram").split(","),
        "prompts": opt("--prompts", ",".join(PROMPTS)).split(","),
        "reps": opt("--reps", "1"),
        "n_tokens": int(opt("--tokens", "256")),
        "ctx": int(opt("--ctx", "16384")),
        "out_json": opt("--json"),
        "ngram_min": opt("--ngram-min"),
        "stock": flag("--stock"),
    }
    if argv:
        raise SystemExit(f"unknown arguments: {' '.join(argv)}\n{USAGE}")
    try:
        cfg["reps"] = int(cfg["reps"])
    except ValueError:
        raise SystemExit(f"--reps takes a whole number, got {cfg['reps']!r}")
    if cfg["reps"] < 1:
        raise SystemExit(f"--reps must be at least 1, got {cfg['reps']}")
    unknown = [p for p in cfg["prompts"] if p not in PROMPTS]
    if unknown or not cfg["prompts"]:
        raise SystemExit(f"unknown prompt(s) {unknown}; choose from {','.join(PROMPTS)}")
    if len(set(cfg["prompts"])) != len(cfg["prompts"]):
        raise SystemExit(f"--prompts names a prompt twice: {','.join(cfg['prompts'])}")
    return cfg


_LAST_STATS = {}


def install_stats_tap(mtp) -> None:
    """Keep the last Speculator.stats() the engine computed, per request."""
    real = mtp.Speculator.stats

    def _stats(self):
        s = real(self)
        _LAST_STATS.clear()
        _LAST_STATS.update(s)
        return s

    mtp.Speculator.stats = _stats


def set_spec(engine, mode: str) -> None:
    """Point the engine at `mode` for the NEXT request. Both caches go: the
    speculation plan AND the depth it hangs off (module docstring)."""
    os.environ["DRINKME_SPEC"] = mode
    engine._spec_plan = None
    engine._mtp_depth = None


def free_gib():
    import torch

    if not torch.cuda.is_available():
        return None
    return round(torch.cuda.mem_get_info()[0] / 1024 ** 3, 3)


def run_once(engine, prompt, n_tokens):
    from drinkme.serving.engine import GenerationRequest, SampleParams, complete

    deltas, t_first = [], None
    t0 = time.monotonic()

    def on_delta(s):
        nonlocal t_first
        if t_first is None:
            t_first = time.monotonic()
        deltas.append(s)
        return True

    _LAST_STATS.clear()
    p = SampleParams(temperature=0.0, max_tokens=n_tokens)
    r = complete(engine, GenerationRequest([{"role": "user", "content": prompt}], p), on_delta)
    wall = time.monotonic() - t0
    ttft = (t_first - t0) if t_first else wall
    dec = ((r.completion_tokens - 1) / (wall - ttft)
           if wall > ttft and r.completion_tokens > 1 else 0.0)
    s = {k: v for k, v in _LAST_STATS.items() if k != "audit"}
    return {"text": r.text, "finish": r.finish_reason, "tokens": r.completion_tokens,
            "wall_s": round(wall, 2), "ttft_s": round(ttft, 2),
            "decode_toks_per_s": round(dec, 3), "spec": s,
            "audit": _LAST_STATS.get("audit")}


def warm_up(engine, arms: list[str], prompt_name: str, n_tokens: int, has_head: bool,
            run=None) -> dict:
    """One untimed request per arm before any timed rep (module docstring,
    WARM-UP): the receipt's `warmup` block. An arm that needs the MTP head
    is skipped without one, as the timed loop skips it."""
    run = run or run_once
    out = {"requests_per_arm": 1, "prompt": prompt_name, "n_tokens": n_tokens,
           "timed": False, "decode_toks_per_s": {}}
    for a in arms:
        if a in ("mtp", "ngram+mtp") and not has_head:
            continue
        set_spec(engine, a)
        r = run(engine, PROMPTS[prompt_name], n_tokens)
        out["decode_toks_per_s"][a] = r["decode_toks_per_s"]
        print(f"[warm-up] {a:10s} {r['decode_toks_per_s']:7.2f} tok/s (untimed: pays the first "
              "request's compiles)", flush=True)
    return out


def identity(t_off: str, t_on: str):
    """"IDENTICAL", or the FORK record: where the two texts first differ."""
    if t_on == t_off:
        return "IDENTICAL"
    div = next((j for j, (x, y) in enumerate(zip(t_off, t_on)) if x != y),
               min(len(t_off), len(t_on)))
    return {"verdict": "FORK", "first_divergence_char": div,
            "off_context": t_off[max(0, div - 60):div + 40],
            "on_context": t_on[max(0, div - 60):div + 40]}


def median(xs: list[float]) -> float:
    xs = sorted(xs)
    n = len(xs)
    return xs[n // 2] if n % 2 else round((xs[n // 2 - 1] + xs[n // 2]) / 2, 3)


def summarize_cell(runs: list[dict]) -> dict:
    """N runs of one prompt x arm -> the cell. The single-run fields come
    from the lower-median run (a real run, so its counters and its tok/s
    belong together), except `decode_toks_per_s`, which is the median; the
    runs themselves stay under `runs` (module docstring)."""
    tps = [r["decode_toks_per_s"] for r in runs]
    med = median(tps)
    rep = sorted(runs, key=lambda r: r["decode_toks_per_s"])[(len(runs) - 1) // 2]
    cell = {k: v for k, v in rep.items() if k != "text"}
    cell["decode_toks_per_s"] = med
    cell["decode_toks_per_s_runs"] = tps
    cell["spread"] = {"min": min(tps), "max": max(tps),
                      "rel_range": round((max(tps) - min(tps)) / med, 4) if med else None}
    cell["runs"] = [{k: v for k, v in r.items() if k != "text"} for r in runs]
    return cell


def main(argv=None):
    cfg = parse_args(sys.argv[1:] if argv is None else argv)
    arms, reps, stock = cfg["arms"], cfg["reps"], cfg["stock"]
    # The head must be RESIDENT for the mtp/ngram+mtp arms to be togglable at all
    # (a mode that cannot use a head does not load one — ngram.wants_head), so the
    # build asks for it and the per-request toggle does the rest.
    os.environ["DRINKME_SPEC"] = "auto"
    os.environ["DRINKME_PREFIX_SLOTS"] = "0"  # cold caches for a like-for-like A/B (the shipped default reuses them)
    if cfg["ngram_min"] is not None:
        os.environ["DRINKME_NGRAM_MIN"] = cfg["ngram_min"]

    from drinkme.serve import build_engine
    from drinkme.serving import mtp

    install_stats_tap(mtp)
    arm = "stock bf16" if stock else "compressed"
    print(f"[ngram-ab] building {arm} engine for {cfg['model']}; arms={','.join(arms)}; "
          f"prompts={','.join(cfg['prompts'])}; reps={reps}", flush=True)
    before = free_gib()
    engine = build_engine(cfg["model"], cfg["revision"], None if stock else cfg["pack"],
                          stock=stock, ctx=cfg["ctx"])
    engine.template_kwargs = {"enable_thinking": False}
    has_head = engine.mtp_head is not None
    print(f"[ngram-ab] MTP head resident: {has_head}; free VRAM "
          f"{before} -> {free_gib()} GiB", flush=True)
    report = {"model": cfg["model"], "revision": cfg["revision"], "pack": cfg["pack"],
              "arm": engine.arm, "meta": engine.meta, "arms": arms,
              "head_resident": has_head, "n_tokens": cfg["n_tokens"], "ctx": cfg["ctx"],
              "ngram_min": cfg["ngram_min"], "prompt_names": cfg["prompts"], "reps": reps,
              "prompts": {}}
    fails, notes = [], []
    report["warmup"] = warm_up(engine, arms, cfg["prompts"][0], cfg["n_tokens"], has_head)
    for name in cfg["prompts"]:
        prompt = PROMPTS[name]
        runs: dict[str, list[dict]] = {}
        for rep in range(reps):
            for a in arms:
                if a in ("mtp", "ngram+mtp") and not has_head:
                    if rep == 0:
                        notes.append(f"{name}: arm {a} skipped — no MTP head resident")
                    continue
                set_spec(engine, a)
                r = run_once(engine, prompt, cfg["n_tokens"])
                r["rep"] = rep + 1
                runs.setdefault(a, []).append(r)
                got = r["spec"].get("mode") if r["spec"] else "off"
                print(f"[{name}] rep {rep + 1}/{reps} {a:10s} mode={got or 'off':10s} "
                      f"{r['decode_toks_per_s']:7.2f} tok/s  ttft {r['ttft_s']:.2f}s  "
                      f"{r['spec'].get('accepted', 0)}/{r['spec'].get('drafted', 0)} "
                      f"accepted  lookup={r['spec'].get('lookup')}", flush=True)
                if a == "off" and got not in (None, "off"):
                    fails.append(f"{name} rep {rep + 1}: the `off` arm reported mode {got!r} — "
                                 "the toggle did not take (stale plan cache?)")
                if a != "off" and got != a:
                    fails.append(f"{name} rep {rep + 1}: asked for {a!r}, engine served {got!r}")
        # identity per rep, against the same rep's off run
        offs = runs.get("off")
        for a, rs in runs.items():
            if offs is None or a == "off":
                continue
            for r, o in zip(rs, offs):
                r["identity"] = identity(o["text"], r["text"])
                if r["identity"] != "IDENTICAL":
                    notes.append(f"{name}/{a} rep {r['rep']}: FORK at char "
                                 f"{r['identity']['first_divergence_char']} — near-tie "
                                 "class, judge against the twin noise floor")
        entry = {a: summarize_cell(rs) for a, rs in runs.items()}
        base = entry.get("off")
        for a, e in entry.items():
            if base is None or a == "off":
                continue
            e["speedup"] = (round(e["decode_toks_per_s"] / base["decode_toks_per_s"], 3)
                            if base["decode_toks_per_s"] else None)
            print(f"[{name}] {a:10s} identity="
                  f"{e['identity'] if isinstance(e['identity'], str) else 'FORK'} "
                  f"median {e['decode_toks_per_s']} tok/s (off {base['decode_toks_per_s']}) "
                  f"speedup={e['speedup']}x spread {e['spread']}", flush=True)
            for r in e["runs"]:
                au = r.get("audit")
                if au and au.get("violations"):
                    fails.append(f"{name}/{a} rep {r['rep']}: {au['violations']} AUDIT "
                                 "VIOLATIONS — the accept rule is wrong, stop here")
        report["prompts"][name] = entry

    print("\n=== VERDICT ===")
    for n in notes:
        print(f"  note: {n}")
    for f in fails:
        print(f"  FAIL: {f}")
    print("PASS: every arm served the mode it was asked for, no audit "
          "violations" if not fails else f"FAIL: {len(fails)} criteria")
    print("(speed is a JUDGEMENT, not a threshold: read the speedup column "
          "per prompt class — ngram should win big on agent-transcript and "
          "repetitive, and be ~1.0x on chat.)")
    report["verdict"] = "PASS" if not fails else "FAIL"
    report["notes"] = notes
    report["fails"] = fails
    print("=== RECEIPT JSON ===")
    print(json.dumps(report, indent=1))
    if cfg["out_json"]:
        with open(cfg["out_json"], "w") as f:
            json.dump(report, f, indent=1)
        print(f"wrote {cfg['out_json']}")
    return 1 if fails else 0


if __name__ == "__main__":
    import os as _os

    code = 1
    try:
        code = main()
    except Exception:
        import traceback

        traceback.print_exc()
    sys.stdout.flush()
    sys.stderr.flush()
    _os._exit(code)  # os._exit discipline (AGENTS.md): never trust $? alone
