"""Where one decode step goes on a CUDA card, and what a CUDA graph buys:
ONE arm per process (stock | compressed | twin), loaded and routed as
`drinkme bench` loads and routes it, then

  1. eager, the bench's own loop (arms.greedy: DynamicCache, one
     model() call per token) — ms/step over --n-new tokens, --reps times;
  2. torch.profiler (CPU + CUDA) over --profile-steps of that loop after
     warm-up, each step inside a `decode_step` range: per step the host
     time (the range), the GPU period (first kernel of this step to the
     first of the next), the GPU busy time (union of kernel intervals) and
     the gap between them; per kernel family (by kernel name) count and
     time per step;
  3. the same, ATTRIBUTED: forward hooks put a profiler range around every
     Linear (its kind, shape class and the bytes it reads), norm, rotary,
     attention and MLP module for --attr-steps steps, and each kernel is
     charged to the innermost range its launch sat in — the radix GEMV per
     shape class, stock cuBLAS per shape class, attention, norms, rotary,
     elementwise; for the GEMVs bytes / kernel time = GB/s against the
     probe's read bandwidth (probe.measure_bandwidth, on the empty device);
  4. the CUDA-graph spike: a transformers StaticCache (max length
     --max-cache), the decode step written so nothing in it reads the host
     (the position is the cache's own device counter, the causal mask is
     built from it on the device and handed to the model as a prepared
     mask dict, the next token is fed back into a static input), timed
     eager and then captured into one torch.cuda.CUDAGraph and replayed;
     tokens and final logits compared bitwise between the two; the replays
     profiled like 2.; and last, what breaks capture when the step is left
     as transformers writes it (no position_ids, no prepared mask).

The hooks of 3. cost host time, so 3.'s wall is not a result; 2.'s is.
Output: one JSON summary (--json); raw traces stay in the container.

    python bench/decode_step_profile.py --model Qwen3-8B --arm compressed \\
        --pack-dir /vol/packs/Qwen3-8B-sip --json out/compressed.json

Verdicts from the printed lines (DECODE_STEP_PROFILE ... / GRAPH ...),
never the exit status.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from drinkme import arms  # noqa: E402
from drinkme.bench import PROMPT  # noqa: E402
from drinkme.codec.radix_schedule import shape_class  # noqa: E402
from drinkme.probe import measure_bandwidth  # noqa: E402
from drinkme.serving.checkpoint import resolve_source, tokenizer as load_tokenizer  # noqa: E402
from drinkme.serving.kernel_route import describe, route_kernels  # noqa: E402


# ------------------------------------------------------------ the model ----

def menu_model(name: str):
    from drinkme.suggest import MODELS

    by = {m.name: m for m in MODELS} | {m.hf_repo: m for m in MODELS}
    if name not in by:
        raise SystemExit(f"unknown model {name!r}")
    return by[name]


def load_arm(arm: str, repo: str, revision, pack_dir: str | None, snap: str):
    """The arm's tree as run_arms builds it: stock streamed raw; compressed
    off the pack through the serve loader (or packed in memory without
    one); the twin from the compressed arm's plan. Then route_kernels."""
    if arm == "stock":
        model = arms.load_stock_streaming(repo, revision, device="cuda", snap=snap)
    elif arm == "compressed":
        if pack_dir:
            model, stats, _meta = arms.load_pack_compressed(repo, revision, pack_dir, device="cuda", snap=snap)
        else:
            model, stats = arms.load_compressed_streaming(repo, revision, device="cuda", snap=snap)
        arms.refuse_unless_verified(repo, stats)
    elif arm == "twin":
        if not pack_dir:
            raise SystemExit("--arm twin needs --pack-dir (its plan is the pack's)")
        _m, stats, _meta = arms.load_pack_compressed(repo, revision, pack_dir, device="cuda", snap=snap)
        del _m
        arms.free_all()
        model, _ = arms.load_twin_streaming(repo, revision, device="cuda", plan=arms.twin_plan(stats), snap=snap)
    else:
        raise SystemExit(f"unknown arm {arm!r}")
    route = route_kernels(model, "cuda")
    print(f"[profile] {arm}: routed {describe(route)}", flush=True)
    model.eval()
    return model, route


def module_bytes(mod) -> int:
    """Bytes the module's weight read costs one decode step — the rule of
    arms.decode_read_bytes, per module."""
    kind = arms.linear_kind(mod)
    if kind == "drinkme_codec":
        return sum(mod.p[k].numel() * mod.p[k].element_size()
                   for k in ("rx_data", "rx_offsets", "rx_palette", "rx_schedule") if torch.is_tensor(mod.p.get(k)))
    if kind in ("drinkme_twin", "drinkme_raw"):
        return mod.p["weight"].numel() * mod.p["weight"].element_size()
    if kind == "bf16":
        return mod.weight.numel() * mod.weight.element_size()
    return 0


def module_shape(mod) -> tuple[int, int]:
    if hasattr(mod, "R") and hasattr(mod, "C"):
        return int(mod.R), int(mod.C)
    return int(mod.out_features), int(mod.in_features)


def sync():
    torch.cuda.synchronize()


class GpuSampler:
    """nvidia-smi in the background at 50 ms: SM/memory clocks, power,
    utilization, temperature and the active clock-event (throttle) reasons
    while a timed loop runs — is a kernel slower because of the code or
    because the card lowered its clocks?"""

    FIELDS = ["clocks.sm", "clocks.mem", "power.draw", "utilization.gpu", "temperature.gpu"]
    REASON_FIELDS = ("clocks_event_reasons.active", "clocks_throttle_reasons.active")

    def __init__(self):
        self.proc = None
        self.lines: list[str] = []
        self.fields = list(self.FIELDS)

    def __enter__(self):
        import subprocess
        import threading

        try:
            for reason in self.REASON_FIELDS + (None,):
                fields = self.FIELDS + ([reason] if reason else [])
                probe = subprocess.run(["nvidia-smi", f"--query-gpu={','.join(fields)}",
                                        "--format=csv,noheader,nounits"], capture_output=True, text=True)
                if probe.returncode == 0:
                    self.fields = fields
                    break
            self.proc = subprocess.Popen(["nvidia-smi", f"--query-gpu={','.join(self.fields)}",
                                          "--format=csv,noheader,nounits", "-lms", "50"],
                                         stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        except OSError:
            self.proc = None
            return self

        def reader():
            for line in self.proc.stdout:
                self.lines.append(line.strip())

        self.thread = threading.Thread(target=reader, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *exc):
        if self.proc is not None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except Exception:  # noqa: BLE001
                self.proc.kill()
            self.thread.join(timeout=2)
        return False

    def summary(self) -> dict:
        rows = []
        for line in self.lines:
            parts = [x.strip() for x in line.split(",")]
            if len(parts) != len(self.fields):
                continue
            rows.append(parts)
        if not rows:
            return {"samples": 0}
        out: dict = {"samples": len(rows)}
        for i, f in enumerate(self.fields):
            if "reasons" in f:
                vals: dict = {}
                for r in rows:
                    vals[r[i]] = vals.get(r[i], 0) + 1
                out["reasons"] = vals
                continue
            xs = []
            for r in rows:
                try:
                    xs.append(float(r[i]))
                except ValueError:
                    pass
            if xs:
                xs.sort()
                out[f] = {"median": xs[len(xs) // 2], "min": xs[0], "max": xs[-1]}
        return out


# --------------------------------------------------------- eager (bench) ---

def bench_steps(model, ids, n: int):
    """arms.greedy's loop, returning the cache and ids after `n` steps from
    a fresh prefill (the prefill is outside `n`)."""
    res = model(ids, use_cache=True)
    past = res.past_key_values
    out = torch.cat([ids, res.logits[:, -1, :].argmax(-1, keepdim=True)], -1)
    return past, out


def eager_timing(model, ids, n_new: int, reps: int) -> dict:
    """ms/step of arms.greedy (prefill included, as the bench times it) and
    of the decode steps alone (prefill outside the clock)."""
    arms.greedy(model, ids, n_new)
    sync()
    bench_ms, step_ms = [], []
    toks = None
    for _ in range(reps):
        sync()
        t0 = time.perf_counter()
        toks = arms.greedy(model, ids, n_new)
        sync()
        bench_ms.append((time.perf_counter() - t0) * 1e3 / n_new)
    with GpuSampler() as smp:
        for _ in range(reps):
            past, out = bench_steps(model, ids, 0)
            sync()
            t0 = time.perf_counter()
            for _ in range(n_new):
                res = model(out[:, -1:], past_key_values=past, use_cache=True)
                past = res.past_key_values
                out = torch.cat([out, res.logits[:, -1, :].argmax(-1, keepdim=True)], -1)
            sync()
            step_ms.append((time.perf_counter() - t0) * 1e3 / n_new)
    return {"gpu_samples": smp.summary(),
            "bench_greedy_ms_per_token": [round(x, 3) for x in bench_ms],
            "bench_greedy_tok_s": [round(1e3 / x, 2) for x in bench_ms],
            "decode_only_ms_per_step": [round(x, 3) for x in step_ms],
            "tokens": toks[0, ids.shape[1]:].tolist()}


# ------------------------------------------------------------- the trace ---

def family(name: str) -> str:
    """A kernel's family by its name (the unattributed view)."""
    n = name
    lo = n.lower()
    if n.startswith("_gemv_narrow"):
        return "narrow_gemv (triton)"
    if n.startswith("_gemv_mc"):
        return "radix_mc (triton)"
    if n.startswith("_gemv"):
        return "radix_gemv (triton)"
    if n.startswith("_finish"):
        return "radix_finish (triton)"
    if n.startswith("_dense") or n.startswith("_prepare_schedule"):
        return "radix_dense (triton)"
    if any(k in lo for k in ("fmha", "flash", "attention", "attn", "efficient")):
        return "attention (sdpa)"
    if any(k in lo for k in ("gemv", "gemm", "xmma", "cutlass", "cublas", "splitk", "sm80_", "sm86_", "sm89_",
                             "sm90_", "ampere_", "hopper", "ada_", "nvjet")):
        return "stock gemm/gemv (cublas)"
    if "reduce" in lo:
        return "reduction"
    if "cat" in lo and "catarray" in lo.replace("_", ""):
        return "cat"
    if "index" in lo or "scatter" in lo or "gather" in lo:
        return "index/copy"
    if "elementwise" in lo or "vectorized" in lo or "unrolled" in lo:
        return "elementwise"
    if "copy" in lo or "memcpy" in lo or "memset" in lo:
        return "copy/memset"
    return "other"


def load_trace(path: str) -> dict:
    with open(path) as f:
        tr = json.load(f)
    ev = tr["traceEvents"] if isinstance(tr, dict) else tr
    kernels, launches, annos = [], {}, []
    for e in ev:
        if e.get("ph") != "X":
            continue
        cat = e.get("cat")
        args = e.get("args") or {}
        if cat in ("kernel", "gpu_memcpy", "gpu_memset"):
            kernels.append({"name": e.get("name", ""), "ts": float(e["ts"]), "dur": float(e.get("dur", 0)),
                            "corr": args.get("correlation"), "cat": cat})
        elif cat in ("cuda_runtime", "cuda_driver"):
            c = args.get("correlation")
            if c is not None:
                launches[c] = {"name": e.get("name", ""), "ts": float(e["ts"]), "dur": float(e.get("dur", 0)),
                               "tid": e.get("tid")}
        elif cat == "user_annotation":
            annos.append({"name": e.get("name", ""), "ts": float(e["ts"]), "dur": float(e.get("dur", 0)),
                          "tid": e.get("tid")})
    kernels.sort(key=lambda k: k["ts"])
    return {"kernels": kernels, "launches": launches, "annos": annos}


def innermost(annos: list, points: list[float]) -> list:
    """For each time in `points` (sorted ascending), the innermost range of
    `annos` (properly nested, one thread) containing it, or None."""
    items = sorted(annos, key=lambda a: (a["ts"], -a["dur"]))
    out, stack, i = [], [], 0
    for t in points:
        while i < len(items) and items[i]["ts"] <= t:
            while stack and stack[-1]["ts"] + stack[-1]["dur"] < items[i]["ts"]:
                stack.pop()
            stack.append(items[i])
            i += 1
        while stack and stack[-1]["ts"] + stack[-1]["dur"] < t:
            stack.pop()
        out.append(stack[-1] if stack else None)
    return out


def union_us(intervals: list[tuple[float, float]]) -> float:
    total, end = 0.0, None
    start = None
    for a, b in sorted(intervals):
        if end is None or a > end:
            if end is not None:
                total += end - start
            start, end = a, b
        else:
            end = max(end, b)
    if end is not None:
        total += end - start
    return total


def per_step(tr: dict, step_prefix: str, attr: dict | None = None) -> dict:
    """Split the trace into the `step_prefix` ranges and summarize: host
    time, GPU period, busy, gap, launches; per family (name or attributed
    scope) count and time, averaged per step. `attr`: {range name ->
    (family, bytes)} for the attributed run."""
    steps = sorted((a for a in tr["annos"] if a["name"].startswith(step_prefix)), key=lambda a: a["ts"])
    if not steps:
        return {"error": f"no {step_prefix!r} ranges in the trace"}
    tid = steps[0]["tid"]
    # each kernel -> its launch's CPU time -> the step (and scope) it belongs to
    ks = []
    for k in tr["kernels"]:
        la = tr["launches"].get(k["corr"])
        if la is None:
            continue
        ks.append((la["ts"], k))
    ks.sort(key=lambda x: x[0])
    step_of = innermost(steps, [t for t, _ in ks])
    scope_of = None
    if attr is not None:
        scoped = [a for a in tr["annos"] if a["tid"] == tid and a["name"] in attr]
        scope_of = innermost(scoped, [t for t, _ in ks])
    by_step: dict = {}
    for j, (t, k) in enumerate(ks):
        s = step_of[j]
        if s is None:
            continue
        rec = by_step.setdefault(s["name"], {"anno": s, "kernels": []})
        # the range instance (name, start): one Linear call, whichever layer
        scope = ((scope_of[j]["name"], scope_of[j]["ts"]) if scope_of is not None and scope_of[j] is not None
                 else None)
        rec["kernels"].append((k, scope))
    names = [s["name"] for s in steps if s["name"] in by_step]
    rows = []
    fam_time: dict = {}
    fam_count: dict = {}
    gemv_bytes: dict = {}
    for i, name in enumerate(names):
        rec = by_step[name]
        kk = rec["kernels"]
        first = min(k["ts"] for k, _ in kk)
        last_end = max(k["ts"] + k["dur"] for k, _ in kk)
        nxt = (min(k["ts"] for k, _ in by_step[names[i + 1]]["kernels"]) if i + 1 < len(names) else None)
        busy = union_us([(k["ts"], k["ts"] + k["dur"]) for k, _ in kk])
        period = (nxt - first) if nxt is not None else (last_end - first)
        rows.append({"step": name, "host_us": round(rec["anno"]["dur"], 1), "gpu_period_us": round(period, 1),
                     "gpu_busy_us": round(busy, 1), "gap_us": round(period - busy, 1),
                     "kernel_sum_us": round(sum(k["dur"] for k, _ in kk), 1), "kernels": len(kk),
                     "has_next": nxt is not None})
        for k, scope in kk:
            if attr is not None:
                fam, nbytes = attr.get(scope[0], ("unscoped", 0)) if scope else ("unscoped", 0)
                kf = family(k["name"])
                if fam.startswith("linear"):
                    # a Linear's own GEMV vs the glue around it (casts, reshapes)
                    is_gemv = kf.startswith(("radix_gemv", "radix_finish", "narrow_gemv", "stock gemm",
                                             "radix_mc"))
                    fam = fam if is_gemv else fam.split("|")[0] + "|glue"
                    if is_gemv:
                        g = gemv_bytes.setdefault(fam, {"bytes": 0, "us": 0.0, "calls": set()})
                        g["us"] += k["dur"]
                        if (name, scope) not in g["calls"]:
                            g["calls"].add((name, scope))
                            g["bytes"] += nbytes
                elif fam == "attention":
                    fam = "attention (sdpa kernel)" if kf == "attention (sdpa)" else "attention glue (rotary apply, kv write, repeat_kv, casts)"
                else:
                    fam = fam
            else:
                fam = family(k["name"])
            fam_time[fam] = fam_time.get(fam, 0.0) + k["dur"]
            fam_count[fam] = fam_count.get(fam, 0) + 1
    n = len(rows)
    steady = [r for r in rows if r["has_next"]] or rows

    def med(key):
        xs = sorted(r[key] for r in steady)
        return xs[len(xs) // 2]

    fams = {f: {"count_per_step": round(fam_count[f] / n, 1), "us_per_step": round(fam_time[f] / n, 1)}
            for f in sorted(fam_time, key=lambda f: -fam_time[f])}
    out = {"steps": n, "median": {k: med(k) for k in ("host_us", "gpu_period_us", "gpu_busy_us", "gap_us",
                                                        "kernel_sum_us", "kernels")},
           "mean": {k: round(sum(r[k] for r in steady) / len(steady), 1)
                    for k in ("host_us", "gpu_period_us", "gpu_busy_us", "gap_us", "kernel_sum_us")},
           "families": fams, "rows": rows}
    if attr is not None:
        out["gemv_read"] = {f: {"bytes_per_step": round(g["bytes"] / n), "us_per_step": round(g["us"] / n, 1),
                                "gb_s": round(g["bytes"] / (g["us"] * 1e-6) / 1e9, 1) if g["us"] else None,
                                "calls_per_step": round(len(g["calls"]) / n, 1)}
                            for f, g in sorted(gemv_bytes.items(), key=lambda x: -x[1]["us"])}
    return out


def profile(fn, steps: int, prefix: str, path: str, attr: dict | None = None, warm: int = 3) -> dict:
    """fn(i) is one step; warm-up `warm` steps, then `steps` profiled, each
    in a `<prefix>#i` range; wall per step with a sync at both ends."""
    from torch.profiler import ProfilerActivity, profile as tprofile, record_function

    for i in range(warm):
        fn(i)
    sync()
    with tprofile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=False,
                  with_stack=False) as prof:
        sync()
        t0 = time.perf_counter()
        for i in range(steps):
            with record_function(f"{prefix}#{i}"):
                fn(warm + i)
        sync()
        wall = (time.perf_counter() - t0) * 1e3 / steps
    prof.export_chrome_trace(path)
    size = os.path.getsize(path)
    summary = per_step(load_trace(path), prefix + "#", attr)
    summary["wall_ms_per_step_under_profiler"] = round(wall, 3)
    summary["trace_bytes"] = size
    return summary


# ------------------------------------------------------- attribution hooks --

def install_scopes(model) -> tuple[list, dict]:
    """Forward pre/post hooks that open and close a profiler range per
    module of interest; returns (handles, {range name -> (family, bytes)})."""
    from torch.autograd.profiler import record_function

    attr: dict = {}
    handles = []
    stack: list = []

    def scope_name(mod, qual) -> tuple[str, str, int] | None:
        cls = type(mod).__name__
        kind = arms.linear_kind(mod)
        if kind != "other":
            R, C = module_shape(mod)
            sc = shape_class(R, C)
            fam = f"linear:{kind}:{sc}"
            return f"{fam}|{R}x{C}|{qual}", fam + "|gemv", module_bytes(mod)
        if "RMSNorm" in cls or cls.endswith("Norm"):
            return "norm", "norm (rmsnorm)", 0
        if "Rotary" in cls:
            return "rotary", "rotary (cos/sin)", 0
        if cls.endswith("Attention"):
            return "attention", "attention", 0
        if cls.endswith("MLP"):
            return "mlp", "mlp elementwise (silu*up)", 0
        if cls.endswith("DecoderLayer"):
            return "layer", "residual adds / layer glue", 0
        if isinstance(mod, torch.nn.Embedding):
            return "embed", "embedding", 0
        if "Activation" in cls or cls in ("SiLU", "SiLUActivation"):
            return "act", "mlp elementwise (silu*up)", 0
        return None

    for qual, mod in model.named_modules():
        got = scope_name(mod, qual)
        if got is None:
            continue
        name, fam, nbytes = got
        # one range name per (family, shape): the per-layer instances share
        # it, the bytes are per call
        key = name.rsplit("|", 1)[0] if name.startswith("linear:") else name
        attr[key] = (fam, nbytes)

        def pre(m, args, _key=key):
            rf = record_function(_key)
            rf.__enter__()
            stack.append(rf)

        def post(m, args, out):
            stack.pop().__exit__(None, None, None)

        handles.append(mod.register_forward_pre_hook(pre))
        handles.append(mod.register_forward_hook(post))
    # linear families keep the kind and class; the bytes differ per shape
    fixed = {}
    for key, (fam, nbytes) in attr.items():
        if key.startswith("linear:"):
            fixed[key] = (fam.replace("|gemv", "") + "|" + key.split("|")[1], nbytes)
        else:
            fixed[key] = (fam, nbytes)
    return handles, fixed


# ------------------------------------------------------------ the graph ---

def graph_spike(model, ids, n_new: int, max_cache: int, reps: int, trace_dir: str, arm: str,
                profile_steps: int, bench_tokens: list | None = None) -> dict:
    from transformers import StaticCache

    out: dict = {"max_cache_len": max_cache}
    cfg = model.config.get_text_config(decoder=True) if hasattr(model.config, "get_text_config") else model.config
    layer_types = set(getattr(cfg, "layer_types", None) or ["full_attention"])
    out["layer_types"] = sorted(layer_types)
    if layer_types != {"full_attention"}:
        out["error"] = f"the prepared-mask step handles full_attention only, not {sorted(layer_types)}"
        return out
    P = ids.shape[1]
    if P + n_new + 8 > max_cache:
        raise SystemExit(f"--max-cache {max_cache} < prompt {P} + {n_new} + 8")
    cache = StaticCache(config=model.config, max_cache_len=max_cache)
    res = model(ids, past_key_values=cache, use_cache=True)
    first = res.logits[:, -1, :].argmax(-1, keepdim=True)
    layers = cache.layers
    static_ids = first.clone()
    ar = torch.arange(max_cache, device="cuda")
    tokens = torch.zeros(max_cache, dtype=torch.long, device="cuda")

    def rewind():
        for layer in layers:
            layer.cumulative_length.fill_(P)
        static_ids.copy_(first)
        tokens.zero_()

    def step():
        # every input is on the device: the position is the cache's own
        # counter (before this step's write), the mask is built from it, the
        # next token goes back into the static input
        pos = layers[0].cumulative_length.clone()
        mask = (ar <= pos).view(1, 1, 1, max_cache)
        o = model(input_ids=static_ids, position_ids=pos.view(1, 1),
                  attention_mask={"full_attention": mask}, past_key_values=cache, use_cache=True)
        logits = o.logits[:, -1, :]
        nxt = logits.argmax(-1, keepdim=True)
        tokens.index_copy_(0, pos.view(1), nxt.view(1))
        static_ids.copy_(nxt)
        return logits

    # eager over the static cache
    rewind()
    for _ in range(4):
        step()
    eager_ms = []
    last_eager = None
    eager_tokens = None
    with GpuSampler() as smp:
        for _ in range(reps):
            rewind()
            sync()
            t0 = time.perf_counter()
            for _ in range(n_new):
                last_eager = step()
            sync()
            eager_ms.append((time.perf_counter() - t0) * 1e3 / n_new)
            # the sequence the bench loop reports: prefill's token, then each step's
            eager_tokens = [int(first)] + tokens[P:P + n_new - 1].tolist()
    out["eager_static_gpu_samples"] = smp.summary()
    last_eager = last_eager.clone()
    out["eager_static_ms_per_step"] = [round(x, 3) for x in eager_ms]
    out["eager_static_tokens"] = eager_tokens
    if bench_tokens is not None:
        # the bench loop's DynamicCache attends over the live length with no
        # mask; this one over the allocation with one: near-ties may differ
        out["tokens_equal_static_vs_bench_loop"] = eager_tokens == bench_tokens[:len(eager_tokens)]

    # capture: warm on a side stream (lazy cuBLAS/Triton state), then one step
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            rewind()
            step()
    torch.cuda.current_stream().wait_stream(s)
    sync()
    rewind()
    sync()
    g = torch.cuda.CUDAGraph()
    t0 = time.perf_counter()
    try:
        with torch.cuda.graph(g):
            static_logits = step()
    except Exception as e:  # noqa: BLE001 — the blocker is the finding
        out["capture_error"] = f"{type(e).__name__}: {e}"[:2000]
        out["capture_traceback"] = traceback.format_exc()[-4000:]
        print(f"GRAPH {arm}: capture FAILED: {out['capture_error'][:300]}", flush=True)
        return out
    out["capture_s"] = round(time.perf_counter() - t0, 3)
    graph_ms, graph_tokens, samples = time_graph(g, rewind, n_new, reps, tokens, first, P)
    out["graph_gpu_samples"] = samples
    out["graph_ms_per_step"] = [round(x, 3) for x in graph_ms]
    out["graph_tokens"] = graph_tokens
    out["tokens_equal_graph_vs_eager_static"] = graph_tokens == eager_tokens
    out["final_logits_bitwise_equal"] = bool(torch.equal(static_logits, last_eager))
    em = sorted(eager_ms)[len(eager_ms) // 2]
    gm = sorted(graph_ms)[len(graph_ms) // 2]
    out["speedup_graph_over_eager_static"] = round(em / gm, 3)
    print(f"GRAPH {arm}: eager-static {em:.2f} ms/step, graph {gm:.2f} ms/step ({em / gm:.2f}x); tokens equal "
          f"{out['tokens_equal_graph_vs_eager_static']}; final logits bitwise {out['final_logits_bitwise_equal']}",
          flush=True)

    # the replays under the profiler (steps stay inside the cache)
    rewind()
    sync()
    try:
        out["graph_profile"] = profile(lambda i: g.replay(), min(profile_steps, n_new - 4), "graph_step",
                                       os.path.join(trace_dir, f"{arm}_graph.json"), warm=2)
        out["graph_profile"].pop("rows", None)
        rewind()
        out["eager_static_profile"] = profile(lambda i: step(), min(profile_steps, n_new - 4), "static_step",
                                              os.path.join(trace_dir, f"{arm}_static.json"), warm=2)
        out["eager_static_profile"].pop("rows", None)
    except Exception as e:  # noqa: BLE001
        out["graph_profile_error"] = f"{type(e).__name__}: {e}"
    del g
    return out


def time_graph(g, rewind, n_new: int, reps: int, tokens, first, P: int):
    """reps x n_new replays from the rewound cache; (ms/step list, the
    last rep's token sequence, the sampler's summary)."""
    ms, toks = [], None
    with GpuSampler() as smp:
        for _ in range(reps):
            rewind()
            sync()
            t0 = time.perf_counter()
            for _ in range(n_new):
                g.replay()
            sync()
            ms.append((time.perf_counter() - t0) * 1e3 / n_new)
            toks = [int(first)] + tokens[P:P + n_new - 1].tolist()
    return [round(x, 3) for x in ms], toks, smp.summary()


def parse_config(spec: str) -> dict:
    """`t<tiles|row>/w<warps>[/s<stages>]` -> a gemv launch override."""
    parts = spec.split("/")
    tiles = parts[0][1:]
    cfg = {"gemv_tiles": 0 if tiles == "row" else int(tiles), "gemv_warps": int(parts[1][1:])}
    for extra in parts[2:]:
        if extra.startswith("s"):
            cfg["gemv_stages"] = int(extra[1:])
    return cfg


_ORIG_GEMV_FUSED = None


def stages_gemv_fused(p, x, bias, out_dtype=torch.bfloat16):
    """radix_ops.gemv_fused with the launch's optional `gemv_stages` passed
    to Triton as num_stages — the one knob the schedule table does not
    carry. Everything else is the shipped function's (same kernels, same
    arguments)."""
    import triton

    from drinkme.codec import radix_ops as ro

    stages = (p.get("rx_launch") or {}).get("gemv_stages")
    if not stages:
        return _ORIG_GEMV_FUSED(p, x, bias, out_dtype)
    R = int(p["R"])
    x = x.reshape(1, -1).contiguous()
    nb, tiles, splits, warps = ro._splits(p, "gemv")
    out = torch.empty((1, R), dtype=out_dtype, device=x.device)
    partial = out if splits == 1 else torch.empty((R, splits), dtype=torch.float32, device=x.device)
    has_bias = bias is not None
    args = ro._args(p)
    b = bias.contiguous() if has_bias else args[0]
    ro._gemv[(R * splits, 1)](partial, x, b, *args, tiles, splits, has_bias, RAW=ro._raw(p),
                              num_warps=warps, num_stages=int(stages), enable_fp_fusion=ro.FUSION)
    if splits != 1:
        ro._finish[(triton.cdiv(R, 128),)](out, partial, b, R, R, splits,
                                           triton.next_power_of_2(splits), has_bias, num_warps=4)
    return out


def graph_config_sweep(model, ids, n_new: int, max_cache: int, reps: int, configs: list[str],
                       classes: list[str] | None) -> dict:
    """The compressed step captured once per launch config and replayed:
    end-to-end ms/step under sustained load (no host gaps to cool the card
    between kernels) with the clocks sampled. Each config overrides the
    gemv fields of every radix Linear of `classes` (all when None) — the
    runtime dict's rx_launch, read at every call — and is restored after."""
    global _ORIG_GEMV_FUSED
    from transformers import StaticCache

    from drinkme.codec import radix_ops as ro

    if _ORIG_GEMV_FUSED is None:
        _ORIG_GEMV_FUSED = ro.gemv_fused
        ro.gemv_fused = stages_gemv_fused
    mods = [m for m in model.modules() if arms.linear_kind(m) == "drinkme_codec"]
    saved = [dict(m.p["rx_launch"]) for m in mods]
    P = ids.shape[1]
    cache = StaticCache(config=model.config, max_cache_len=max_cache)
    res = model(ids, past_key_values=cache, use_cache=True)
    first = res.logits[:, -1, :].argmax(-1, keepdim=True)
    layers = cache.layers
    static_ids = first.clone()
    ar = torch.arange(max_cache, device="cuda")
    tokens = torch.zeros(max_cache, dtype=torch.long, device="cuda")

    def rewind():
        for layer in layers:
            layer.cumulative_length.fill_(P)
        static_ids.copy_(first)
        tokens.zero_()

    def step():
        pos = layers[0].cumulative_length.clone()
        mask = (ar <= pos).view(1, 1, 1, max_cache)
        o = model(input_ids=static_ids, position_ids=pos.view(1, 1),
                  attention_mask={"full_attention": mask}, past_key_values=cache, use_cache=True)
        nxt = o.logits[:, -1, :].argmax(-1, keepdim=True)
        tokens.index_copy_(0, pos.view(1), nxt.view(1))
        static_ids.copy_(nxt)

    results = {}
    base_tokens = None
    env0 = os.environ.get(ro.DECODER_ENV)
    for spec in ["table"] + list(configs):
        # a spec repeated in the list (an A/B cycled in rounds) keys its
        # later runs spec#1, spec#2, ...
        key = spec if spec not in results else f"{spec}#{sum(k.split('#')[0] == spec for k in results)}"
        try:
            # `dec=<scheduled|lean>`: the table's launches under that decoder
            # (radix_ops._decoder, read at every launch through _args)
            os.environ[ro.DECODER_ENV] = spec[4:] if spec.startswith("dec=") else (env0 or "")
            ro._decoder.cache_clear()
            for m, sv in zip(mods, saved):
                m.p["rx_launch"] = dict(sv)
                if spec != "table" and not spec.startswith("dec=") and (classes is None or sv["shape_class"] in classes):
                    m.p["rx_launch"].update(parse_config(spec))
            rewind()
            step()
            step()
            sync()
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                rewind()
                step()
            torch.cuda.current_stream().wait_stream(s)
            sync()
            rewind()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                step()
            ms, toks, samples = time_graph(g, rewind, n_new, reps, tokens, first, P)
            if base_tokens is None:
                base_tokens = toks
            med = sorted(ms)[len(ms) // 2]
            results[key] = {"ms_per_step": ms, "median": med, "gpu_samples": samples,
                            "tokens_equal_table": toks == base_tokens}
            clk = samples.get("clocks.sm", {}).get("median")
            pw = samples.get("power.draw", {}).get("median")
            print(f"SWEEP {spec:14s} {med:8.2f} ms/step  sm {clk} MHz  power {pw} W  tokens=table "
                  f"{toks == base_tokens}", flush=True)
            del g
        except Exception as e:  # noqa: BLE001 — a config that fails is a result
            results[key] = {"error": f"{type(e).__name__}: {e}"[:800]}
            print(f"SWEEP {spec:14s} ERROR {type(e).__name__}: {str(e)[:200]}", flush=True)
    for m, sv in zip(mods, saved):
        m.p["rx_launch"] = sv
    if env0 is None:
        os.environ.pop(ro.DECODER_ENV, None)
    else:
        os.environ[ro.DECODER_ENV] = env0
    ro._decoder.cache_clear()
    return {"classes": classes or "all", "results": results}


def naive_capture(model, ids, max_cache: int) -> dict:
    """The step as transformers writes it — input_ids and the cache only —
    captured: what breaks. Runs LAST (a failed capture can leave the
    context unusable)."""
    from transformers import StaticCache

    cache = StaticCache(config=model.config, max_cache_len=max_cache)
    res = model(ids, past_key_values=cache, use_cache=True)
    static_ids = res.logits[:, -1, :].argmax(-1, keepdim=True).clone()

    def step():
        o = model(input_ids=static_ids, past_key_values=cache, use_cache=True)
        return o.logits[:, -1, :].argmax(-1, keepdim=True)

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        step()
    torch.cuda.current_stream().wait_stream(s)
    sync()
    g = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(g):
            step()
        return {"captured": True}
    except Exception as e:  # noqa: BLE001
        return {"captured": False, "error": f"{type(e).__name__}: {e}"[:1500],
                "traceback": traceback.format_exc()[-5000:]}


# ------------------------------------------------------------------ main ---

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen3-8B")
    ap.add_argument("--arm", required=True, choices=("stock", "compressed", "twin"))
    ap.add_argument("--pack-dir", default=None)
    ap.add_argument("--json", required=True)
    ap.add_argument("--trace-dir", default="/tmp/decode_traces")
    ap.add_argument("--n-new", type=int, default=64)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--profile-steps", type=int, default=20)
    ap.add_argument("--attr-steps", type=int, default=5)
    ap.add_argument("--max-cache", type=int, default=128)
    ap.add_argument("--no-graph", action="store_true")
    ap.add_argument("--no-naive", action="store_true")
    ap.add_argument("--only-sweep", action="store_true", help="skip 1-4, run the graph config sweep only")
    ap.add_argument("--sweep", nargs="*", default=[],
                    help="graph-mode launch configs for the radix gemv, t<tiles|row>/w<warps>[/s<stages>]")
    ap.add_argument("--sweep-classes", nargs="*", default=None, help="shape classes the sweep overrides")
    args = ap.parse_args()
    os.makedirs(args.trace_dir, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)

    report: dict = {"arm": args.arm, "model": args.model, "pack_dir": args.pack_dir,
                    "device": torch.cuda.get_device_name(0), "capability": list(torch.cuda.get_device_capability()),
                    "torch": torch.__version__, "env": {k: v for k, v in os.environ.items()
                                                         if k.startswith("DRINKME_")}}
    try:
        import transformers
        import triton
        report["transformers"], report["triton"] = transformers.__version__, triton.__version__
    except Exception:  # noqa: BLE001
        pass

    def save():
        with open(args.json, "w") as f:
            json.dump(report, f, indent=1)

    m = menu_model(args.model)
    snap, resolved = resolve_source(m.hf_repo, m.revision, args.pack_dir)
    report["revision"] = resolved
    tok = load_tokenizer(snap, None)
    ids = tok(PROMPT, return_tensors="pt").input_ids.cuda()
    report["prompt_tokens"] = ids.shape[1]
    report["bandwidth"] = measure_bandwidth()
    bw = report["bandwidth"]["read_bytes_s"]
    print(f"[profile] probe read {bw / 1e9:.1f} GB/s on {report['device']}", flush=True)

    with torch.inference_mode():
        t0 = time.perf_counter()
        model, route = load_arm(args.arm, m.hf_repo, m.revision, args.pack_dir, snap)
        report["load_s"] = round(time.perf_counter() - t0, 1)
        report["routing"] = route
        try:
            from drinkme.codec import radix_ops as ro
            report["radix_decoder"] = {"env": os.environ.get(ro.DECODER_ENV), "sip": ro._decoder((3, 8))}
        except Exception as e:  # noqa: BLE001
            report["radix_decoder"] = f"{type(e).__name__}: {e}"

        report["read_bytes"] = arms.decode_read_bytes(model)
        total = report["read_bytes"]["total_bytes"]
        report["bound_ms_per_step"] = round(total / bw * 1e3, 3)
        save()

        if args.only_sweep:
            report["sweep"] = graph_config_sweep(model, ids, args.n_new, args.max_cache, args.reps, args.sweep,
                                                 args.sweep_classes)
            save()
            print(f"DECODE_STEP_PROFILE {args.arm}: DONE (sweep only) -> {args.json}", flush=True)
            sys.stdout.flush()
            os._exit(0)

        # 1. eager timing, the bench's loop
        et = eager_timing(model, ids, args.n_new, args.reps)
        report["eager"] = et
        med = sorted(et["decode_only_ms_per_step"])[len(et["decode_only_ms_per_step"]) // 2]
        report["eager"]["fraction_of_bound"] = round(report["bound_ms_per_step"] / med, 3)
        print(f"DECODE_STEP_PROFILE {args.arm}: eager (bench loop) {med:.2f} ms/step; bytes {total / 1e9:.3f} GB -> "
              f"bound {report['bound_ms_per_step']:.2f} ms ({report['eager']['fraction_of_bound']:.2f} of bound)",
              flush=True)
        save()

        # 2. clean profile of the bench loop
        state = {}

        def bench_step(i):
            if i == 0 or "past" not in state:
                state["past"], state["out"] = bench_steps(model, ids, 0)
            res = model(state["out"][:, -1:], past_key_values=state["past"], use_cache=True)
            state["past"] = res.past_key_values
            state["out"] = torch.cat([state["out"], res.logits[:, -1, :].argmax(-1, keepdim=True)], -1)

        state.clear()
        prof = profile(bench_step, args.profile_steps, "decode_step",
                       os.path.join(args.trace_dir, f"{args.arm}_eager.json"))
        report["profile_eager"] = prof
        md = prof["median"]
        print(f"DECODE_STEP_PROFILE {args.arm}: profiled {prof['steps']} steps: host {md['host_us'] / 1e3:.2f} ms, "
              f"GPU period {md['gpu_period_us'] / 1e3:.2f} ms, busy {md['gpu_busy_us'] / 1e3:.2f} ms, "
              f"gap {md['gap_us'] / 1e3:.2f} ms, {md['kernels']} kernels", flush=True)
        for f, v in list(prof["families"].items())[:12]:
            print(f"    {f:40s} {v['count_per_step']:7.1f}/step {v['us_per_step'] / 1e3:8.3f} ms/step", flush=True)
        save()

        # 3. attributed profile
        handles, attr = install_scopes(model)
        try:
            state.clear()
            ap_ = profile(bench_step, args.attr_steps, "decode_step",
                          os.path.join(args.trace_dir, f"{args.arm}_attr.json"), attr=attr, warm=2)
        finally:
            for h in handles:
                h.remove()
        ap_.pop("rows", None)
        report["profile_attributed"] = ap_
        print(f"DECODE_STEP_PROFILE {args.arm}: attributed families (ms/step):", flush=True)
        for f, v in ap_["families"].items():
            print(f"    {f:60s} {v['count_per_step']:7.1f}/step {v['us_per_step'] / 1e3:8.3f}", flush=True)
        for f, v in ap_.get("gemv_read", {}).items():
            frac = (v["gb_s"] * 1e9 / bw) if v["gb_s"] else None
            print(f"    GEMV {f:55s} {v['bytes_per_step'] / 1e6:9.1f} MB {v['us_per_step'] / 1e3:7.3f} ms "
                  f"{v['gb_s']} GB/s ({frac:.2f} of probe)" if frac else f"    GEMV {f}: {v}", flush=True)
        save()

        # 4. the graph spike
        if not args.no_graph:
            try:
                report["graph"] = graph_spike(model, ids, args.n_new, args.max_cache, args.reps, args.trace_dir,
                                              args.arm, args.profile_steps, et["tokens"])
            except Exception as e:  # noqa: BLE001
                report["graph"] = {"error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()[-5000:]}
                print(f"GRAPH {args.arm}: ERROR {type(e).__name__}: {e}", flush=True)
            save()
            if not args.no_naive:
                try:
                    report["naive_capture"] = naive_capture(model, ids, args.max_cache)
                except Exception as e:  # noqa: BLE001
                    report["naive_capture"] = {"captured": False, "error": f"{type(e).__name__}: {e}"}
                nc = report["naive_capture"]
                print(f"GRAPH {args.arm}: naive capture (transformers' own step) captured={nc.get('captured')} "
                      f"{(nc.get('error') or '')[:300]}", flush=True)
                save()
        if args.sweep and args.arm == "compressed":
            report["sweep"] = graph_config_sweep(model, ids, args.n_new, args.max_cache, args.reps, args.sweep,
                                                 args.sweep_classes)
            save()
    print(f"DECODE_STEP_PROFILE {args.arm}: DONE -> {args.json}", flush=True)
    save()
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
