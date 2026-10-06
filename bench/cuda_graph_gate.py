"""The correctness bar for CUDA-graph decode (serving/cudagraph.py), ONE arm
per process, loaded and routed as `drinkme bench` loads and routes it:

  0. attention: the static step's attention (torch's flash kernel over the
     whole allocation, the live length in `seqused_k`) against SDPA over
     the same keys with a mask built from that length, at the model's head
     shape, 1..8 query rows; and the flash output bitwise unchanged when
     every key row past the live length is overwritten with noise;
  1. the bench loop: arms.greedy (eager, DynamicCache) against
     arms.GraphDecode (the captured step), tokens and ms/step;
  2. graph replay == eager execution of the same static step, bitwise:
     tokens and the final logits over --n-new steps, for each --rows
     (M = 1 is the decode step; M > 1 a verify-shaped step whose rows'
     argmaxes feed the next step);
  3. at each --lengths live length (a prompt of that many tokens,
     prefilled into a cache allocated by serve's rule): today's eager serve
     step (LiveStaticLayer's live window, SDPA) against the eager static
     step and its graph, ms/step, and the bitwise check of 2. again.

  5. `--memory ROWS`: what one serve cache's graphs hold (the decode step,
     `--memory-hidden`'s MTP re-arm step, and every verify width 1..ROWS),
     captured the way serve captures them, first with the per-graph-pool layout
     (cudagraph.SHARED = False: a private pool per graph) and then shared
     (one pool per cache, one row buffer), by torch.cuda.memory_snapshot's
     per-pool segments; the shared layout is captured on a second cache too
     (capture time without first-use compiles).

With more than one --rows, every width is captured into ONE cache's shared
pool, and a second pass re-checks each width's replay against its eager
step with every other width replayed (and rewound) before each of its
replays: the aliasing a shared pool could introduce fails that pass.

  4. serve's own generate (--engine-specs), eager and graph, speculation
     off and on: each run's emitted rows (token, top-1 minus top-2 logit),
     and where two runs part — graph against eager, speculation against
     serial — with both runs' margins there and agreement.label's verdict
     (agree, near-tie, or above agreement.NEAR_TIE_MARGIN).

`--compare stock.json compressed.json` reads two arms' JSONs and prints
where the compressed arm's tokens part from stock's: the bench loop's,
graph and eager, with the static step's margins at that index, and serve's
generate per setting, with both runs' margins and the near-tie verdict.
Two arms over the same weights agree or part at a near-tie
(docs/method.md, "Numerical behavior").

    python bench/cuda_graph_gate.py --model Qwen3-8B --arm compressed \\
        --pack-dir /vol/packs/Qwen3-8B-sip --json out/gate_compressed.json

Verdicts from the printed VERDICT lines, never the exit status.
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
from drinkme.serving import cudagraph  # noqa: E402
from drinkme.serving.checkpoint import resolve_source, tokenizer as load_tokenizer  # noqa: E402
from drinkme.serving.kvcache import LiveStaticCache  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from agreement import describe, fork, label  # noqa: E402
from decode_step_profile import load_arm, menu_model  # noqa: E402


def sync():
    torch.cuda.synchronize()


def median(xs):
    xs = sorted(xs)
    return xs[len(xs) // 2]


def text_cfg(model):
    cfg = model.config
    return cfg.get_text_config() if hasattr(cfg, "get_text_config") else cfg


# ------------------------------------------------------------ 0. attention --

def attention_check(model) -> dict:
    cfg = text_cfg(model)
    H, Hk = cfg.num_attention_heads, cfg.num_key_value_heads
    D = getattr(cfg, "head_dim", None) or cfg.hidden_size // H
    g = torch.Generator(device="cuda").manual_seed(0)
    worst, cases, ignored = 0.0, 0, True
    rows_out = []
    for alloc in (256, 8192):
        for live in (1, 7, 129, 250) if alloc == 256 else (300, 4096, 8000):
            for rows in range(1, 9):
                if rows > live:
                    continue
                q = torch.randn(1, H, rows, D, device="cuda", dtype=torch.bfloat16, generator=g)
                k = torch.randn(1, Hk, alloc, D, device="cuda", dtype=torch.bfloat16, generator=g)
                v = torch.randn(1, Hk, alloc, D, device="cuda", dtype=torch.bfloat16, generator=g)
                m = cudagraph.LiveLength(rows, alloc, torch.tensor([live], dtype=torch.int32, device="cuda"),
                                         torch.tensor([0, rows], dtype=torch.int32, device="cuda"),
                                         torch.tensor([0, alloc], dtype=torch.int32, device="cuda"))
                a = cudagraph._flash(q, k, v, m, D ** -0.5)
                ref = cudagraph._reference(q.float(), k.float(), v.float(), m, D ** -0.5)
                d = (a.float() - ref).abs().max().item()
                worst = max(worst, d)
                cases += 1
                if live < alloc:
                    k2, v2 = k.clone(), v.clone()
                    k2[:, :, live:] = 1e4
                    v2[:, :, live:] = float("nan")
                    b = cudagraph._flash(q, k2, v2, m, D ** -0.5)
                    same = bool(torch.equal(a, b))
                    ignored &= same
                rows_out.append({"alloc": alloc, "live": live, "rows": rows, "max_abs": round(d, 6)})
    ok = worst < 2e-2 and ignored
    print(f"VERDICT attention: flash varlen (seqused_k) vs fp32 masked SDPA: max |diff| {worst:.3g} over {cases} "
          f"cases (H {H}, Hk {Hk}, D {D}, rows 1..8); rows past the live length ignored bitwise: {ignored} -> "
          f"{'PASS' if ok else 'FAIL'}", flush=True)
    return {"max_abs": worst, "cases": cases, "rows_past_live_ignored": ignored, "pass": ok, "detail": rows_out}


# --------------------------------------------------------- the static step --

def make_step(model, cache, rows: int, tokens: torch.Tensor, margins: torch.Tensor | None = None,
              max_rows: int | None = None):
    """A static step of `rows` rows whose forward feeds each row's argmax
    into the next step's input (row i -> input row i), and records the last
    row's argmax at its position + 1 in `tokens`. The forward is serve's:
    the wrapper at M = 1 (cudagraph.decode_step), the verify's
    mtp.forward_with_hidden above it (cudagraph.verify_step)."""
    from drinkme.serving.mtp import forward_with_hidden

    from drinkme.serving import mtp

    def forward(ids, position_ids, mask):
        if rows == 1:
            logits = model(input_ids=ids, position_ids=position_ids, attention_mask=mask,
                           past_key_values=cache, use_cache=True).logits[0]
        else:
            # the cycle's own conditions: a DeltaNet capture active, so a
            # hybrid's verify rows take the capturing (unrolled) forward
            mtp._ACTIVE = mtp._Capture()
            try:
                logits = forward_with_hidden(model, ids, cache, None, attention_mask=mask,
                                             position_ids=position_ids)[1]
            finally:
                mtp._ACTIVE = None
        nxt = logits.argmax(-1)
        tokens.index_copy_(0, position_ids.view(-1)[-1:] + 1, nxt[-1:])
        if margins is not None:
            top = logits[-1].float().topk(2).values
            margins.index_copy_(0, position_ids.view(-1)[-1:] + 1, (top[0] - top[1]).view(1))
        ids.copy_(nxt.view(1, -1))
        return logits

    return cudagraph.StaticStep(model, cache, rows, forward, max_rows)


def alloc_of(cache) -> int:
    from drinkme.serving.kvcache import LiveStaticLayer
    return next(layer.max_cache_len for layer in cache.layers if isinstance(layer, LiveStaticLayer))


def prefill(model, cache, ids, chunk: int = 2048):
    cache.reset()
    P = ids.shape[1]
    logits = None
    for a in range(0, P, chunk):
        logits = model(ids[:, a:a + chunk], past_key_values=cache, use_cache=True, logits_to_keep=1).logits
    return logits[0, -1]


def graph_vs_eager(model, cache, ids, rows: int, n: int, max_rows: int | None = None,
                   step=None, others=()) -> tuple[dict, object]:
    """Eager static steps, then the graph's replays, from the same
    prefilled state and the same first input: tokens and final logits.
    `step` given: its graph is reused (not captured again), and before each
    replay every step in `others` replays once and the cache is rewound."""
    P = ids.shape[1]
    last = prefill(model, cache, ids)
    first = last.argmax(-1)
    top = last.float().topk(2).values
    first_margin = float(top[0] - top[1])
    capture_s = graph_bytes = graph_reserved = 0
    if step is None:
        tokens = torch.zeros(alloc_of(cache) + 8, dtype=torch.long, device="cuda")
        margins = torch.zeros(alloc_of(cache) + 8, dtype=torch.float32, device="cuda")
        step = make_step(model, cache, rows, tokens, margins, max_rows)
        step.gate_bufs = (tokens, margins)
        sync()
        mem0, res0 = torch.cuda.memory_allocated(), torch.cuda.memory_reserved()
        t0 = time.perf_counter()
        step.capture()
        capture_s = time.perf_counter() - t0
        sync()
        # what this graph added: its static outputs and the capture's
        # allocations (a pool shared with the widths before it grows by less)
        graph_bytes = torch.cuda.memory_allocated() - mem0
        graph_reserved = torch.cuda.memory_reserved() - res0
    tokens, margins = step.gate_bufs
    base = step.snapshot()
    seed = torch.full((1, rows), int(first), dtype=torch.long, device="cuda")
    step.ids.copy_(seed)
    tokens.zero_()
    out = None
    for _ in range(n):
        out = step.eager()
    sync()
    eager_tokens = tokens.clone()
    eager_logits = out.clone()
    eager_live = step._lengths()
    step.restore(base)
    step.ids.copy_(seed)
    tokens.zero_()
    for _ in range(n):
        if others:
            snap = step.snapshot()
            for o in others:
                o.replay()
            step.restore(snap)
        out = step.replay()
    sync()
    graph_live = step._lengths()
    counters = [int(layer.cumulative_length) for layer in step.layers]
    same_tokens = bool(torch.equal(eager_tokens, tokens))
    same_logits = bool(torch.equal(eager_logits, out))
    return {"rows": rows, "steps": n, "capture_s": round(capture_s, 3), "graph_bytes": graph_bytes,
            "graph_reserved_bytes": graph_reserved, "tokens_equal": same_tokens,
            "final_logits_bitwise": same_logits, "live_after": graph_live[0],
            "live_and_counter_agree": graph_live == counters == eager_live,
            "others_replayed": len(others),
            "tokens": tokens[tokens.nonzero().view(-1)].tolist()[:n],
            # top-1 minus top-2 logit of each generated token (the first from prefill)
            "margins": ([round(first_margin, 4)] + [round(x, 4) for x in margins[P + 1:P + n].tolist()]
                        if rows == 1 else None)}, step


# --------------------------------------------------------- 5. memory --

def pools_now() -> dict:
    """{pool id: [reserved, allocated]} over the allocator's segments."""
    out: dict = {}
    for seg in torch.cuda.memory_snapshot():
        pid = tuple(seg.get("segment_pool_id", (0, 0)))
        o = out.setdefault(pid, [0, 0])
        o[0] += seg["total_size"]
        o[1] += seg["allocated_size"]
    return out


def capture_all(model, ids, rows_max: int, hidden: bool, shared: bool) -> tuple[object, dict]:
    """One serve slot's cache (KV_FLOOR = 4096), the prompt prefilled, then
    every step serve replays against it captured the way serve captures
    them (engines.HFEngine._precapture, mtp.Speculator.precapture)."""
    import gc

    from drinkme.serving import mtp

    cudagraph.SHARED = shared
    cache = LiveStaticCache(config=model.config, max_cache_len=4096)
    prefill(model, cache, ids)
    sync()
    gc.collect()
    torch.cuda.empty_cache()
    p0, r0, a0 = pools_now(), torch.cuda.memory_reserved(), torch.cuda.memory_allocated()
    times = {}
    t = time.perf_counter()
    ok = cudagraph.decode_step(model, cache) is not None
    times["decode"] = round(time.perf_counter() - t, 3)
    if hidden:
        t = time.perf_counter()
        ok &= cudagraph.hidden_step(model, cache) is not None
        times["hidden"] = round(time.perf_counter() - t, 3)
    for rows in range(1, rows_max + 1):
        cap = mtp._Capture()
        mtp._ACTIVE = cap
        t = time.perf_counter()
        try:
            st = cudagraph.verify_step(model, cache, rows, rows_max)
        finally:
            mtp._ACTIVE = None
        times[f"verify{rows}"] = round(time.perf_counter() - t, 3)
        ok &= st is not None
        if st is not None and getattr(st, "cap", None) is None:
            st.cap = cap  # what mtp keeps alive (the DeltaNet rows' views)
    sync()
    p1, r1, a1 = pools_now(), torch.cuda.memory_reserved(), torch.cuda.memory_allocated()
    graph_pools = {k: v for k, v in p1.items() if k != (0, 0) and k not in p0}
    rows_buf = sum(b.numel() * b.element_size()
                   for d in [cache.__dict__.get("_drinkme_row_states") or {},
                             *cache.__dict__.get("_drinkme_row_old", [])] for b in d.values())
    res = {"shared": shared, "rows_max": rows_max, "hidden": hidden, "captured_all": ok,
           "graphs": len(cache.__dict__.get("_drinkme_steps", {})), "capture_s": times,
           "capture_s_total": round(sum(times.values()), 3),
           "pools": len(graph_pools),
           "pool_reserved_bytes": sum(v[0] for v in graph_pools.values()),
           "pool_allocated_bytes": sum(v[1] for v in graph_pools.values()),
           "row_buffer_bytes": rows_buf,
           "default_pool_reserved_delta": p1.get((0, 0), [0, 0])[0] - p0.get((0, 0), [0, 0])[0],
           "reserved_delta_bytes": r1 - r0, "allocated_delta_bytes": a1 - a0,
           "graph_bytes_fn": cudagraph.graph_bytes(cache) if shared else None}
    return cache, res


def memory_phase(model, ids, rows_max: int, hidden: bool) -> dict:
    import gc

    from drinkme.serving import mtp

    if rows_max > 1:
        mtp.install_deltanet_capture(model)
    out = {}
    for name, shared in (("before", False), ("after", True), ("after_second_cache", True)):
        try:
            cache, r = capture_all(model, ids, rows_max, hidden, shared)
            del cache
        except Exception as e:  # noqa: BLE001
            r = {"error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()[-3000:]}
        cudagraph.SHARED = True
        gc.collect()
        sync()
        torch.cuda.empty_cache()
        out[name] = r
        mib = 2 ** 20
        if "error" in r:
            print(f"MEMORY {name}: ERROR {r['error']}", flush=True)
        else:
            print(f"MEMORY {name}: {r['graphs']} graphs (decode{' + re-arm' if hidden else ''} + verify 1..{rows_max}) "
                  f"in {r['pools']} pool(s): pools reserve {r['pool_reserved_bytes'] / mib:.1f} MiB "
                  f"({r['pool_allocated_bytes'] / mib:.1f} allocated), row buffer {r['row_buffer_bytes'] / mib:.1f} MiB; "
                  f"reserved +{r['reserved_delta_bytes'] / mib:.1f} MiB, allocated +{r['allocated_delta_bytes'] / mib:.1f} MiB "
                  f"over the capture; capture {r['capture_s_total']} s {r['capture_s']}", flush=True)
    return out


def time_steps(fn, n: int, reps: int = 3) -> list[float]:
    out = []
    for _ in range(reps):
        sync()
        t0 = time.perf_counter()
        for _ in range(n):
            fn()
        sync()
        out.append((time.perf_counter() - t0) * 1e3 / n)
    return out


def serve_alloc(need: int, floor: int = 4096) -> int:
    size = floor
    while size < need:
        size *= 2
    return size


def long_live(model, tok, L: int, n: int, max_new: int) -> dict:
    """Today's eager serve step against the static step (eager and graph)
    at a live length of L."""
    ids = torch.tensor([arms.build_prefill_ids(tok, L)], device="cuda")
    alloc = serve_alloc(L + max_new)
    cache = LiveStaticCache(config=model.config, max_cache_len=alloc)
    last = prefill(model, cache, ids)
    first = last.argmax(-1)
    tokens = torch.zeros(alloc + 8, dtype=torch.long, device="cuda")
    step = make_step(model, cache, 1, tokens)
    t0 = time.perf_counter()
    step.capture()
    capture_s = time.perf_counter() - t0
    base = step.snapshot()
    step_in = torch.zeros((1, 1), dtype=torch.long, device="cuda")

    def today():
        logits = model(step_in, past_key_values=cache, use_cache=True).logits[0, -1]
        step_in.copy_(logits.argmax(-1).view(1, 1))

    def reset(buf):
        step.restore(base)
        buf.fill_(int(first))

    res = {"live": L, "alloc": alloc, "steps": n, "capture_s": round(capture_s, 3)}
    times = {"today_eager": [], "static_eager": [], "graph": []}
    for _round in range(2):
        for name, buf, fn in (("today_eager", step_in, today), ("static_eager", step.ids, step.eager),
                              ("graph", step.ids, step.replay)):
            for _ in range(3):
                reset(buf)
                ms = time_steps(fn, n, reps=1)[0]
                times[name].append(round(ms, 3))
    for k, v in times.items():
        res[f"{k}_ms"] = v
        res[f"{k}_median_ms"] = round(median(v), 3)
    # bitwise at this length, and the noise check past the live window
    reset(step.ids)
    tokens.zero_()
    out = None
    for _ in range(n):
        out = step.eager()
    sync()
    e_tok, e_log = tokens.clone(), out.clone()
    reset(step.ids)
    tokens.zero_()
    for _ in range(n):
        out = step.replay()
    sync()
    res["tokens_equal"] = bool(torch.equal(e_tok, tokens))
    res["final_logits_bitwise"] = bool(torch.equal(e_log, out))
    reset(step.ids)
    for layer in step.layers:
        layer.keys[:, :, L + n + 1:].normal_()
        layer.values[:, :, L + n + 1:].normal_()
    tokens.zero_()
    for _ in range(n):
        out = step.replay()
    sync()
    res["noise_past_live_changes_nothing"] = bool(torch.equal(e_tok, tokens) and torch.equal(e_log, out))
    del step, cache
    arms.free_all()
    return res


# ------------------------------------------------------- 4. the engine --

ENGINE_PROMPT = ("Write a Python function that returns the n-th Fibonacci number iteratively, then "
                 "repeat the function's body once more with the variable names in uppercase.")


def rows_fork(a: list, b: list) -> dict:
    """Where two runs' emitted rows ([token, top-1 minus top-2 logit] per
    row, engines.sample_next wrapped) first part, and each run's margin
    there (agreement.fork)."""
    return fork([x[0] for x in a], [y[0] for y in b], [x[1] for x in a], [y[1] for y in b])


def text_and_rows(a: dict, b: dict) -> str:
    """Two engine runs side by side: the same text, or the character where
    their texts part and the emitted row where their picks part."""
    if a["text"] == b["text"]:
        return f"same text ({len(a['text'])} chars)"
    at = next((i for i, (p, q) in enumerate(zip(a["text"], b["text"])) if p != q),
              min(len(a["text"]), len(b["text"])))
    rows = describe(rows_fork(a["rows"], b["rows"]), "emitted row") if "rows" in a and "rows" in b else "no rows"
    return f"text parts at char {at}; {rows}"


def engine_runs(model, tok, snap, arm: str, model_id: str, pack_dir, n_tokens: int, specs: list) -> dict:
    """serve's own generate() over this arm (HFEngine, prefix slots 0,
    thinking off), one greedy request per (graphs, speculation) setting:
    the text, tok/s, the speculator's stats, and every emitted row's token
    and top-1 minus top-2 logit (the serial loop and the verify's accept
    loop both pick through engines.sample_next), so a divergence can be read
    against the margin where it happens."""
    from drinkme.serving import engines

    inner = engines.sample_next
    rec: list = []

    def sample_next(logits, *a, **k):
        t = inner(logits, *a, **k)
        top = logits.float().topk(2).values
        rec.append([int(t), round(float(top[0] - top[1]), 4)])
        return t

    engines.sample_next = sample_next
    try:
        return _engine_runs(model, tok, snap, arm, model_id, pack_dir, n_tokens, specs, rec)
    finally:
        engines.sample_next = inner


def _engine_runs(model, tok, snap, arm, model_id, pack_dir, n_tokens, specs, rec) -> dict:
    from drinkme.serving import engines, gen_config, ngram
    from drinkme.serving.engine import GenerationRequest, SampleParams, complete

    out = {}
    for spec in specs:
        os.environ["DRINKME_SPEC"] = spec
        head = None
        if ngram.wants_head():
            # the stock arm's head is the raw one (serve --stock), as bench's pass loads it
            head = engines._mtp_head(model, snap, None, "cuda", pack_dir if arm == "compressed" else None)
        engine = engines.HFEngine(model, tok, model_id=model_id, arm=arm, meta={},
                                  ctx=engines._ctx(model.config, None),
                                  template_kwargs={"enable_thinking": False}, mtp_head=head,
                                  prefix_slots=0, gen_defaults=gen_config.load(model, snap))
        on = engine._graphs
        for graphs in ((False, True) if on else (False,)):
            engine._graphs = graphs
            engine._scratch_kv, engine._scratch_alloc = None, 0
            runs = []
            for _ in range(2):
                first = [None]

                def on_delta(_t):
                    if first[0] is None:
                        first[0] = time.perf_counter()
                    return True

                req = GenerationRequest([{"role": "user", "content": ENGINE_PROMPT}],
                                        SampleParams(temperature=0.0, max_tokens=n_tokens))
                rec.clear()
                sync()
                r = complete(engine, req, on_delta)
                sync()
                end = time.perf_counter()
                st = engine.last_spec_stats or {}
                runs.append({"text": r.text, "tokens": r.completion_tokens,
                             "tok_s": round((r.completion_tokens - 1) / (end - first[0]), 2) if first[0] else None,
                             "mode": st.get("mode", "serial"), "cycles": st.get("cycles"),
                             "accepted": st.get("accepted"), "drafted": st.get("drafted"),
                             "rows": [list(x) for x in rec]})
            key = f"{'graph' if graphs else 'eager'}/{spec}"
            out[key] = {"text": runs[-1]["text"], "tokens": runs[-1]["tokens"], "runs": runs,
                        "rows": runs[-1]["rows"], "repeatable": runs[0]["text"] == runs[1]["text"]}
            print(f"ENGINE {arm} {key}: {runs[-1]['tokens']} tokens at {[x['tok_s'] for x in runs]} tok/s, "
                  f"{runs[-1]['mode']} (cycles {runs[-1]['cycles']}, accepted {runs[-1]['accepted']}/"
                  f"{runs[-1]['drafted']}); two runs agree {out[key]['repeatable']}", flush=True)
        del engine, head
        arms.free_all()
    os.environ.pop("DRINKME_SPEC", None)
    base = out.get("graph/off")
    for key, r in out.items():
        if key.startswith("graph/") and key != "graph/off" and base:
            print(f"VERDICT engine {arm} {key} vs graph/off (speculation against serial in graph mode): "
                  f"{text_and_rows(r, base)} -> {label(rows_fork(r['rows'], base['rows']))}", flush=True)
    if "eager/off" in out and base:
        print(f"VERDICT engine {arm} graph/off vs eager/off (today's serve): "
              f"{text_and_rows(base, out['eager/off'])} -> "
              f"{label(rows_fork(base['rows'], out['eager/off']['rows']))}", flush=True)
    return out


def warm_runs(model, tok, snap, arm: str, model_id: str, n_tokens: int) -> dict:
    """serve's own repeat: one prefix slot (the server's default), the same
    greedy request three times (cold, then warm off the slot through a
    rewind or a context checkpoint), eager and then graph, speculation at
    serve's default; every sampled row's top-1 minus top-2 logit recorded
    (engines.sample_next wrapped), so a divergence between two runs can be
    read against the margin where it happens."""
    from drinkme.serving import engines, gen_config
    from drinkme.serving.engine import GenerationRequest, SampleParams, complete
    from serve_drive import PROMPT as SERVE_PROMPT

    os.environ.pop("DRINKME_SPEC", None)
    inner = engines.sample_next
    rec: list = []

    def sample_next(logits, *a, **k):
        t = inner(logits, *a, **k)
        top = logits.float().topk(2).values
        rec.append((int(t), round(float(top[0] - top[1]), 4)))
        return t

    engines.sample_next = sample_next
    out = {}
    try:
        for graphs in (False, True):
            engine = engines.HFEngine(model, tok, model_id=model_id, arm=arm, meta={},
                                      ctx=8192, template_kwargs={"enable_thinking": False},
                                      prefix_slots=1, gen_defaults=gen_config.load(model, snap))
            engine._graphs = graphs and engine._graphs
            key = "graph" if engine._graphs else "eager"
            runs = []
            for _ in range(3):
                rec.clear()
                req = GenerationRequest([{"role": "user", "content": SERVE_PROMPT}],
                                        SampleParams(temperature=0.0, max_tokens=n_tokens))
                r = complete(engine, req, lambda _t: True)
                runs.append({"text": r.text, "tokens": [t for t, _m in rec], "margins": [m for _t, m in rec],
                             "cached": r.cached_tokens, "spec": (engine.last_spec_stats or {}).get("mode")})
            out[key] = runs
            for i in (1, 2):
                f = fork(runs[0]["tokens"], runs[i]["tokens"], runs[0]["margins"], runs[i]["margins"])
                print(f"VERDICT warm {arm} {key} run {i} (reused {runs[i]['cached']} prompt tokens, "
                      f"{runs[i]['spec']}) vs cold run 0: {describe(f, 'sampled row')}", flush=True)
            small = sorted(runs[0]["margins"])[:3]
            print(f"  warm {arm} {key}: the cold run's three smallest margins {small}", flush=True)
            del engine
            arms.free_all()
    finally:
        engines.sample_next = inner
    return out


# ------------------------------------------------------------------ main --

def compare(a_path: str, b_path: str) -> None:
    a, b = json.load(open(a_path)), json.load(open(b_path))

    def first_diff(x, y):
        for i, (p, q) in enumerate(zip(x, y)):
            if p != q:
                return i
        return None if len(x) == len(y) else min(len(x), len(y))

    for key, what in (("graph_tokens", "graph-mode bench loop"), ("eager_tokens", "eager bench loop")):
        x, y = a.get(key), b.get(key)
        if x is None or y is None:
            print(f"VERDICT {b['arm']} vs {a['arm']} {what}: MISSING", flush=True)
            continue
        d = first_diff(x, y)
        gap = ""
        if d is not None:
            g = d - a.get("prompt_tokens", 0)
            ma = next((r.get("margins") for r in a.get("bitwise", []) if r.get("rows") == 1), None)
            mb = next((r.get("margins") for r in b.get("bitwise", []) if r.get("rows") == 1), None)
            if ma and mb and g < len(ma):
                gap = (f"; top-2 logit margin there in each arm's static-step run: "
                       f"{a['arm']} {ma[g]}, {b['arm']} {mb[g]}")
        print(f"VERDICT {b['arm']} vs {a['arm']} {what} tokens: "
              + ("agree" if d is None else f"part at generated token {d - a.get('prompt_tokens', 0)}")
              + f" ({len(x)} ids){gap}", flush=True)
    ea, eb = a.get("engine") or {}, b.get("engine") or {}
    for key in sorted(set(ea) & set(eb)):
        if isinstance(ea[key], dict) and "text" in ea[key]:
            verdict = (f" -> {label(rows_fork(eb[key]['rows'], ea[key]['rows']))}"
                       if "rows" in ea[key] and "rows" in eb[key] else "")
            print(f"VERDICT {b['arm']} vs {a['arm']} engine {key}: "
                  f"{text_and_rows(eb[key], ea[key])}{verdict}", flush=True)
    for L in sorted({r["live"] for r in a.get("long", []) if "live" in r}):
        ra = next(r for r in a["long"] if r.get("live") == L)
        rb = next((r for r in b.get("long", []) if r.get("live") == L), None)
        if rb:
            print(f"  long L={L}: {a['arm']} today {ra['today_eager_median_ms']} / static {ra['static_eager_median_ms']}"
                  f" / graph {ra['graph_median_ms']} ms; {b['arm']} today {rb['today_eager_median_ms']} / static "
                  f"{rb['static_eager_median_ms']} / graph {rb['graph_median_ms']} ms", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen3-8B")
    ap.add_argument("--arm", choices=("stock", "compressed", "twin"))
    ap.add_argument("--pack-dir", default=None)
    ap.add_argument("--json")
    ap.add_argument("--n-new", type=int, default=64)
    ap.add_argument("--rows", type=int, nargs="*", default=[1])
    ap.add_argument("--lengths", type=int, nargs="*", default=[4096, 16384])
    ap.add_argument("--long-steps", type=int, default=32)
    ap.add_argument("--max-new", type=int, default=256, help="serve's allocation rule: live + max_new, doubled from 4096")
    ap.add_argument("--compare", nargs=2, default=None)
    ap.add_argument("--engine-specs", nargs="*", default=[], help="DRINKME_SPEC values for the engine runs (off ngram mtp)")
    ap.add_argument("--engine-tokens", type=int, default=192)
    ap.add_argument("--only-warm", action="store_true", help="only warm_runs (serve's repeated request)")
    ap.add_argument("--memory", type=int, default=0,
                    help="5.: capture decode + verify widths 1..N into one serve cache, the per-graph-pool layout then shared")
    ap.add_argument("--memory-hidden", action="store_true", help="5.: the MTP re-arm step too (a model with a head)")
    ap.add_argument("--only-memory", action="store_true")
    ap.add_argument("--assume-verified", action="store_true",
                    help="put this family's decode and verify modes on cudagraph.VERIFIED in this process "
                         "(the gate is what verifies them)")
    args = ap.parse_args()
    if args.compare:
        compare(*args.compare)
        return
    os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
    report: dict = {"arm": args.arm, "model": args.model, "device": torch.cuda.get_device_name(0),
                    "torch": torch.__version__,
                    "env": {k: v for k, v in os.environ.items() if k.startswith("DRINKME_")}}

    def save():
        with open(args.json, "w") as f:
            json.dump(report, f, indent=1)

    m = menu_model(args.model)
    snap, resolved = resolve_source(m.hf_repo, m.revision, args.pack_dir)
    tok = load_tokenizer(snap, None)
    ids = tok(PROMPT, return_tensors="pt").input_ids.cuda()
    report["prompt_tokens"] = ids.shape[1]
    with torch.inference_mode():
        model, route = load_arm(args.arm, m.hf_repo, m.revision, args.pack_dir, snap)
        report["routing"] = route
        report["family"] = cudagraph.family(model)
        if args.assume_verified:
            cudagraph.VERIFIED = cudagraph.VERIFIED | {(report["family"], "decode"), (report["family"], "verify")}
            report["assumed_verified"] = sorted(cudagraph.VERIFIED)
        if args.only_warm:
            report["warm"] = warm_runs(model, tok, snap, args.arm, m.hf_repo, 256)
            save()
            print(f"CUDA_GRAPH_GATE {args.arm}: DONE (warm only) -> {args.json}", flush=True)
            sys.stdout.flush()
            os._exit(0)
        if args.memory:
            report["memory"] = memory_phase(model, ids, args.memory, args.memory_hidden)
            save()
            if args.only_memory:
                print(f"CUDA_GRAPH_GATE {args.arm}: DONE (memory only) -> {args.json}", flush=True)
                sys.stdout.flush()
                os._exit(0)
        try:
            report["attention"] = attention_check(model)
        except Exception as e:  # noqa: BLE001
            report["attention"] = {"error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()[-3000:]}
            print(f"VERDICT attention: ERROR {type(e).__name__}: {e}", flush=True)
        save()
        # 1. the bench loop, eager then graph
        eager_out, eager_s = arms.timed_decode(model, ids, args.n_new, 3, step="eager")
        report["eager_tokens"] = eager_out[0].tolist()
        report["eager_tok_s"] = eager_s
        ran = {}
        graph_out, graph_s = arms.timed_decode(model, ids, args.n_new, 3, step="cudagraph", ran=ran)
        report["graph_ran"] = ran.get("step")
        report["graph_tokens"] = graph_out[0].tolist()
        report["graph_tok_s"] = graph_s
        same = report["graph_tokens"] == report["eager_tokens"]
        print(f"BENCH LOOP {args.arm}: eager {median(eager_s)} tok/s, {report['graph_ran']} {median(graph_s)} tok/s "
              f"({median(graph_s) / median(eager_s):.2f}x); graph tokens agree with the eager bench loop's: {same}",
              flush=True)
        save()
        # 2. graph == eager static, per rows
        report["bitwise"] = []
        if any(r > 1 for r in args.rows):
            from drinkme.serving import mtp
            mtp.install_deltanet_capture(model)  # what the engine does once it speculates
        # every width into ONE cache and its shared pool (serve's layout)
        maxr = max(args.rows)
        cache = LiveStaticCache(config=model.config, max_cache_len=ids.shape[1] + maxr * args.n_new + 16)
        kept = {}

        def verdict(r, rows, label):
            ok = r.get("tokens_equal") and r.get("final_logits_bitwise") and r.get("live_and_counter_agree")
            print(f"VERDICT {label} {args.arm} M={rows}: "
                  + (f"ERROR {r['error']}" if "error" in r else
                     f"tokens {'EQUAL' if r['tokens_equal'] else 'DIFFER'} ({r['steps']} steps), final logits "
                     f"{'BITWISE EQUAL' if r['final_logits_bitwise'] else 'DIFFER'}, live==counter "
                     f"{r['live_and_counter_agree']} ("
                     + (f"{r['others_replayed']} other widths replayed before each replay"
                        if r["others_replayed"] else
                        f"capture {r['capture_s']} s, the shared pool +{r['graph_reserved_bytes'] / 2**20:.1f} MiB "
                        f"reserved, +{r['graph_bytes'] / 2**20:.1f} MiB allocated")
                     + f") -> {'PASS' if ok else 'FAIL'}"),
                  flush=True)

        for rows in args.rows:
            try:
                r, kept[rows] = graph_vs_eager(model, cache, ids, rows, args.n_new, max_rows=maxr)
            except Exception as e:  # noqa: BLE001
                r = {"rows": rows, "error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()[-4000:]}
            report["bitwise"].append(r)
            verdict(r, rows, "graph==eager-static")
            save()
        # the aliasing pass: each width again, every other width replayed first
        if len(kept) > 1:
            report["bitwise_shared"] = []
            for rows in reversed(list(kept)):
                try:
                    r, _ = graph_vs_eager(model, cache, ids, rows, args.n_new, step=kept[rows],
                                          others=[kept[w] for w in kept if w != rows])
                except Exception as e:  # noqa: BLE001
                    r = {"rows": rows, "error": f"{type(e).__name__}: {e}",
                         "traceback": traceback.format_exc()[-4000:]}
                report["bitwise_shared"].append(r)
                verdict(r, rows, "shared-pool graph==eager-static")
                save()
        report["shared_cache_graph_bytes"] = cudagraph.graph_bytes(cache)
        print(f"  the bitwise cache's {len(kept)} graphs hold {report['shared_cache_graph_bytes'] / 2**20:.1f} MiB "
              f"(cudagraph.graph_bytes)", flush=True)
        del cache, kept
        arms.free_all()
        # 4. serve's own generate, eager and graph, speculation off and on
        if args.engine_specs:
            try:
                report["engine"] = engine_runs(model, tok, snap, args.arm, m.hf_repo, args.pack_dir,
                                               args.engine_tokens, args.engine_specs)
            except Exception as e:  # noqa: BLE001
                report["engine"] = {"error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()[-4000:]}
                print(f"VERDICT engine {args.arm}: ERROR {type(e).__name__}: {e}", flush=True)
            save()
        # 3. long live lengths
        report["long"] = []
        for L in args.lengths:
            try:
                r = long_live(model, tok, L, args.long_steps, args.max_new)
            except Exception as e:  # noqa: BLE001
                r = {"live": L, "error": f"{type(e).__name__}: {str(e)[:500]}"}
                arms.free_all()
            report["long"].append(r)
            if "error" in r:
                print(f"VERDICT long {args.arm} L={L}: ERROR {r['error']}", flush=True)
            else:
                ratio = r["static_eager_median_ms"] / r["today_eager_median_ms"]
                gratio = r["graph_median_ms"] / r["today_eager_median_ms"]
                ok = r["tokens_equal"] and r["final_logits_bitwise"] and r["noise_past_live_changes_nothing"]
                # the bar is the step as it runs (the graph) against today's
                # eager step; the static step run eagerly is reported beside
                # it (its extra launches cost host time where eager is host-bound)
                print(f"VERDICT long {args.arm} L={L} (alloc {r['alloc']}): today's eager step "
                      f"{r['today_eager_median_ms']} ms, graph {r['graph_median_ms']} ms ({gratio:.3f}x), "
                      f"static eager {r['static_eager_median_ms']} ms ({ratio:.3f}x); graph==static tokens "
                      f"{r['tokens_equal']}, logits {r['final_logits_bitwise']}; noise past live changes nothing "
                      f"{r['noise_past_live_changes_nothing']} -> {'PASS' if ok and gratio <= 1.0 else 'FAIL'}",
                      flush=True)
            save()
    save()
    print(f"CUDA_GRAPH_GATE {args.arm}: DONE -> {args.json}", flush=True)
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
