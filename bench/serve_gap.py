"""Where `drinkme serve`'s decode rate goes, against the engine's own: ONE
process, `drinkme serve`'s own engine (serve.build_engine and serve._warmup,
the pack's compressed arm, --ctx as given), the same greedy request
(bench/serve_drive.py's prompt and body) driven several ways, each timed as
serve_drive times a stream ((tokens - 1) / first-to-last delta):

  http      start_server on a spare port (never 3215), the request streamed
            by a client subprocess: what `drinkme serve` measures;
  direct    engine.complete() in this process, on the same engine: the
            HTTP layer removed;
  variants  HFEngine rebuilt around the same weights and head (as
            bench/cuda_graph_gate.py's engine runs build it): prefix slots
            0, context checkpoints off, speculation off, and the gate's own
            ENGINE_PROMPT, so each piece of serve's configuration is
            measured apart;
  timed     one direct request under host timers, CUDA-synchronized at each
            boundary so each bucket holds its GPU time: per speculation
            cycle the head's drafts, the trunk's verify, the greedy picks,
            the rewind and the head's rebuild; per token the detok and the
            scanners; prefill; the rest of the loop;
  profile   cProfile over one direct request (top functions by own time).

    python bench/serve_gap.py --model Qwen3.8-27B --pack-dir /vol/packs/Qwen3.8-27B-sip \\
        --json out/serve_gap.json [--ctx 8192] [--n 3]

Readings from the printed SERVE_GAP lines, never the exit status.
"""
from __future__ import annotations

import argparse
import cProfile
import io
import json
import os
import pstats
import subprocess
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402

from serve_drive import PROMPT  # noqa: E402

PORT = 3292
HERE = os.path.dirname(os.path.abspath(__file__))


def request(prompt: str, max_tokens: int):
    from drinkme.serving.engine import GenerationRequest, SampleParams

    return GenerationRequest([{"role": "user", "content": prompt}],
                             SampleParams(temperature=0.0, max_tokens=max_tokens),
                             template_kwargs={"enable_thinking": False})


def direct(engine, prompt: str, max_tokens: int) -> dict:
    """complete() with serve_drive's clock: first to last delta."""
    from drinkme.serving.engine import complete

    stamps = []

    def on_delta(_t):
        stamps.append(time.perf_counter())
        return True

    t0 = time.perf_counter()
    r = complete(engine, request(prompt, max_tokens), on_delta)
    torch.cuda.synchronize()
    ct = r.completion_tokens
    span = stamps[-1] - stamps[0] if len(stamps) > 1 else None
    st = engine.last_spec_stats or {}
    return {"text": r.text, "completion_tokens": ct, "ttft_s": stamps[0] - t0 if stamps else None,
            "decode_tok_s": (ct - 1) / span if span else None, "deltas": len(stamps),
            "cached": r.cached_tokens, "mode": st.get("mode", "serial"), "cycles": st.get("cycles"),
            "accepted": st.get("accepted"), "drafted": st.get("drafted")}


def over_http(engine, max_tokens: int, n: int) -> list:
    from drinkme.serving.http import start_server

    srv = start_server(engine, "127.0.0.1", PORT)
    try:
        out = []
        for _ in range(n):
            code = ("import json, sys; sys.path.insert(0, %r); from serve_drive import request; "
                    "print(json.dumps(request(%d, %d)))" % (HERE, PORT, max_tokens))
            p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=900)
            out.append(json.loads(p.stdout.strip().splitlines()[-1]) if p.returncode == 0
                       else {"error": p.stderr[-2000:]})
        return out
    finally:
        srv.shutdown()


def in_thread(engine, prompt: str, max_tokens: int) -> dict:
    """direct(), on a fresh thread (the HTTP handler's situation)."""
    import threading

    out: dict = {}
    th = threading.Thread(target=lambda: out.update(direct(engine, prompt, max_tokens)))
    th.start()
    th.join()
    return out


def thread_state() -> dict:
    """What torch keeps per thread, read in the calling thread."""
    s = torch.cuda.current_stream()
    return {"stream": hex(s.cuda_stream), "device": torch.cuda.current_device(),
            "grad": torch.is_grad_enabled(), "inference": torch.is_inference_mode_enabled(),
            "num_threads": torch.get_num_threads(), "interop": torch.get_num_interop_threads()}


def sync_probe(n: int = 200) -> dict:
    """ms per (one small kernel + a device-to-host read), and per launch
    without a read: the two host costs a speculation cycle repeats."""
    x = torch.randn(4096, device="cuda")
    for _ in range(10):
        int(x.argmax())
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(n):
        int(x.argmax())
    read = (time.perf_counter() - t) * 1e3 / n
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(n):
        x.add_(0.0)
    torch.cuda.synchronize()
    launch = (time.perf_counter() - t) * 1e3 / n
    return {"argmax_item_ms": round(read, 4), "launch_ms": round(launch, 4)}


def sdpa_probe(heads: int = 24, kv_heads: int = 4, dim: int = 256, n: int = 24) -> dict:
    """torch's backend choice and ms per call for one decode query row over
    n NEW key lengths (a fresh shape each call, as a growing KV gives),
    default backends and with cuDNN excluded, in the calling thread."""
    from torch.nn.attention import SDPBackend, sdpa_kernel

    out = {}
    q = torch.randn(1, heads, 1, dim, device="cuda", dtype=torch.bfloat16)
    base = 300 + 1000 * len(_PROBED)
    _PROBED.append(1)
    for label, ctx in (("default", None),
                       ("no cudnn", [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION,
                                     SDPBackend.MATH])):
        ks = [torch.randn(1, kv_heads, base + i + (500 if ctx else 0), dim, device="cuda",
                          dtype=torch.bfloat16) for i in range(n)]
        import contextlib

        names = {int(v): k for k, v in SDPBackend.__members__.items()}
        cm = sdpa_kernel(ctx) if ctx else contextlib.nullcontext()
        with cm:
            choice = torch._fused_sdp_choice(q, ks[0], ks[0], None, 0.0, False, scale=None, enable_gqa=True)
            torch.cuda.synchronize()
            t = time.perf_counter()
            for k in ks:
                torch.nn.functional.scaled_dot_product_attention(q, k, k, enable_gqa=True)
            torch.cuda.synchronize()
        out[label] = {"backend": names.get(int(choice), choice), "ms_per_new_length": round((time.perf_counter() - t) * 1e3 / n, 3)}
    return out


_PROBED: list = []


def on_thread(fn):
    import threading

    out: dict = {}
    th = threading.Thread(target=lambda: out.update(v=fn()))
    th.start()
    th.join()
    return out["v"]


def http_timed(engine, max_tokens: int) -> dict:
    """One request over HTTP with the handler's time split: inside the
    engine's generator (every send), in _sse (JSON and the socket write),
    and the whole _generate."""
    from drinkme.serving import http as H

    T: dict = defaultdict(float)
    N: dict = defaultdict(int)
    orig_gen, orig_sse, orig_g = engine.generate, H._Handler._sse, H._Handler._generate

    def gen(req):
        g = orig_gen(req)

        class W:
            def __iter__(self):
                return self

            def __next__(self):
                return self.send(None)

            def send(self, v):
                t = time.perf_counter()
                try:
                    return g.send(v)
                finally:
                    T["engine generator (send)"] += time.perf_counter() - t
                    N["engine generator (send)"] += 1

            def close(self):
                g.close()

        return W()

    def sse(self, obj):
        t = time.perf_counter()
        try:
            return orig_sse(self, obj)
        finally:
            T["_sse (json + socket write)"] += time.perf_counter() - t
            N["_sse (json + socket write)"] += 1

    def g2(self, eng, req, emit=None):
        t = time.perf_counter()
        try:
            return orig_g(self, eng, req, emit)
        finally:
            T["_generate (whole)"] += time.perf_counter() - t
            N["_generate (whole)"] += 1

    engine.generate = gen
    H._Handler._sse, H._Handler._generate = sse, g2
    try:
        run = over_http(engine, max_tokens, 1)[0]
    finally:
        del engine.generate
        H._Handler._sse, H._Handler._generate = orig_sse, orig_g
    run.pop("text", None)
    run["buckets_s"] = {k: round(v, 4) for k, v in T.items()}
    run["buckets_n"] = dict(N)
    return run


def variant(base, *, slots: int | None = None, ckpts: int | None = None):
    """An HFEngine around base's model and head, as the gate builds one."""
    from drinkme.serving import engines

    return engines.HFEngine(base.model, base.tok, model_id=base.model_id, arm=base.arm, meta=base.meta,
                            ctx=base.ctx, template_kwargs=None, mtp_head=base.mtp_head,
                            prefix_slots=slots, n_ctx_checkpoints=ckpts,
                            gen_defaults=base.gen_defaults)


class Timers:
    """Inclusive wall per bucket, CUDA-synchronized at entry and exit."""

    def __init__(self):
        self.t = defaultdict(float)
        self.n = defaultdict(int)
        self.undo = []

    def wrap(self, owner, name: str, label: str):
        fn = getattr(owner, name)
        t, n = self.t, self.n

        def w(*a, **k):
            torch.cuda.synchronize()
            s = time.perf_counter()
            try:
                return fn(*a, **k)
            finally:
                torch.cuda.synchronize()
                t[label] += time.perf_counter() - s
                n[label] += 1

        setattr(owner, name, w)
        self.undo.append((owner, name, fn))

    def restore(self):
        for owner, name, fn in reversed(self.undo):
            setattr(owner, name, fn)
        self.undo.clear()


def timed(engine, prompt: str, max_tokens: int) -> dict:
    from drinkme.serving import detok, engines, mtp, prefill, sampling

    T = Timers()
    T.wrap(mtp.Speculator, "cycle", "cycle")
    T.wrap(mtp.Speculator, "_trunk", "trunk (verify / re-arm step)")
    T.wrap(mtp, "_restore_rows", "rewind (_restore_rows)")
    if engine.mtp_head is not None:
        head = type(engine.mtp_head)
        for name in ("draft", "run", "crop"):
            if hasattr(head, name):
                T.wrap(head, name, f"head.{name}")
    T.wrap(engines, "sample_next", "pick (sample_next)")
    T.wrap(prefill, "run", "prefill")
    T.wrap(detok.IncrementalDetok, "push", "detok.push")
    if hasattr(detok, "SuffixWindow"):
        T.wrap(detok.SuffixWindow, "full", "detok window")
    T.wrap(sampling.StopScanner, "feed", "stop scanner")
    try:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        r = direct(engine, prompt, max_tokens)
        wall = time.perf_counter() - t0
    finally:
        T.restore()
    r.pop("text", None)
    r["wall_s"] = wall
    r["buckets_s"] = {k: round(v, 4) for k, v in sorted(T.t.items(), key=lambda kv: -kv[1])}
    r["buckets_n"] = dict(T.n)
    return r


def profiled(engine, prompt: str, max_tokens: int, top: int = 35) -> str:
    pr = cProfile.Profile()
    pr.enable()
    direct(engine, prompt, max_tokens)
    pr.disable()
    s = io.StringIO()
    pstats.Stats(pr, stream=s).sort_stats("tottime").print_stats(top)
    return s.getvalue()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen3.8-27B")
    ap.add_argument("--pack-dir", required=True)
    ap.add_argument("--json", required=True)
    ap.add_argument("--ctx", type=int, default=8192)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--skip", nargs="*", default=[], help="phases to skip: http variants timed profile")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
    from drinkme import serve
    from drinkme.suggest import MODELS

    from cuda_graph_gate import ENGINE_PROMPT

    m = next(x for x in MODELS if x.name == args.model)
    report: dict = {"model": args.model, "device": torch.cuda.get_device_name(0), "ctx": args.ctx,
                    "env": {k: v for k, v in os.environ.items() if k.startswith("DRINKME_")}}

    def save():
        with open(args.json, "w") as f:
            json.dump(report, f, indent=1)

    engine = serve.build_engine(m.hf_repo, m.revision, args.pack_dir, stock=False, ctx=args.ctx)
    serve._warmup(engine)
    report["graphs"] = engine._graphs

    def line(label, runs):
        rates = [round(x["decode_tok_s"], 2) for x in runs if x.get("decode_tok_s")]
        extra = ""
        r0 = runs[-1]
        if r0.get("cycles"):
            extra = f"; {r0['mode']} {r0['cycles']} cycles, {r0['accepted']}/{r0['drafted']} accepted"
        print(f"SERVE_GAP {label}: decode tok/s {rates}; {r0.get('completion_tokens')} tokens{extra}", flush=True)

    # serve's own engine, the HTTP layer on and off, alternated
    if "http" not in args.skip:
        report["http"] = over_http(engine, args.max_tokens, args.n)
        line("http (serve's engine, streamed over HTTP)", report["http"])
        save()
    report["direct"] = [direct(engine, PROMPT, args.max_tokens) for _ in range(args.n)]
    line("direct (serve's engine, complete() in-process)", report["direct"])
    report["direct_thread"] = [in_thread(engine, PROMPT, args.max_tokens) for _ in range(args.n)]
    line("direct, on a fresh thread", report["direct_thread"])
    if "http" not in args.skip:
        from drinkme.serving import http as H

        report["http_timed"] = http_timed(engine, args.max_tokens)
        r = report["http_timed"]
        print(f"SERVE_GAP http, timed: decode tok/s {r.get('decode_tok_s')}; "
              + "; ".join(f"{k} {v:.3f} s x{r['buckets_n'][k]}" for k, v in r["buckets_s"].items()),
              flush=True)
        H._Handler.disable_nagle_algorithm = True
        try:
            report["http_nodelay"] = over_http(engine, args.max_tokens, args.n)
        finally:
            H._Handler.disable_nagle_algorithm = False
        line("http with TCP_NODELAY on the handler's socket", report["http_nodelay"])
        save()
    report["direct_engine_prompt"] = [direct(engine, ENGINE_PROMPT, 192) for _ in range(2)]
    line("direct, the gate's ENGINE_PROMPT at 192 tokens", report["direct_engine_prompt"])
    if report.get("http") and report["direct"]:
        same = report["http"][0].get("text") == report["direct"][0]["text"]
        print(f"  http text == direct text: {same}", flush=True)
    save()
    if "variants" not in args.skip:
        for label, kw in (("prefix slots 0", {"slots": 0}), ("context checkpoints 0", {"ckpts": 0}),
                          ("speculation off", {"spec": "off"})):
            spec = kw.pop("spec", None)
            if spec is not None:
                os.environ["DRINKME_SPEC"] = spec  # read at the engine's first generation
            try:
                eng = variant(engine, **kw)
                runs = [direct(eng, PROMPT, args.max_tokens) for _ in range(args.n)]
                del eng
            except Exception as e:  # noqa: BLE001
                runs = [{"error": f"{type(e).__name__}: {e}"}]
                print(f"SERVE_GAP {label}: ERROR {runs[0]['error']}", flush=True)
            else:
                line(f"direct, {label}", runs)
            finally:
                if spec is not None:
                    os.environ.pop("DRINKME_SPEC", None)
                torch.cuda.empty_cache()
            report[f"variant {label}"] = runs
            save()
    if "nocudnn" not in args.skip:
        # cuDNN attention off for the whole process (the eager trunk's decode
        # and prefill too): a numerics change for the trunk, measured, not shipped
        torch.backends.cuda.enable_cudnn_sdp(False)
        try:
            report["nocudnn_http"] = over_http(engine, args.max_tokens, args.n)
            line("http, cuDNN SDPA off in the whole process", report["nocudnn_http"])
            report["nocudnn_thread"] = [in_thread(engine, PROMPT, args.max_tokens) for _ in range(args.n)]
            line("direct on a fresh thread, cuDNN SDPA off", report["nocudnn_thread"])
            if report.get("http"):
                same = report["http"][0].get("text") == report["nocudnn_http"][0].get("text")
                print(f"  http text with cuDNN off == http text with it on: {same}", flush=True)
        finally:
            torch.backends.cuda.enable_cudnn_sdp(True)
        save()
    if "probe" not in args.skip:
        for where, call in (("main", lambda f: f()), ("thread", on_thread)):
            st = call(thread_state)
            pr = call(sync_probe)
            try:
                sp = call(sdpa_probe)
            except Exception as e:  # noqa: BLE001
                sp = {"error": f"{type(e).__name__}: {e}"}
            report[f"probe_{where}"] = {"state": st, "sync": pr, "sdpa": sp}
            print(f"SERVE_GAP probe {where}: {st}; {pr}; sdpa {sp}", flush=True)
        save()
    if "timed" not in args.skip:
        for where, call in (("main thread", lambda f: f()), ("fresh thread", on_thread)):
            r = call(lambda: timed(engine, PROMPT, args.max_tokens))
            report[f"timed {where}"] = r
            print(f"SERVE_GAP timed on the {where} (synchronized buckets, inclusive): wall {r['wall_s']:.2f} s, "
                  f"{r['completion_tokens']} tokens, {r.get('cycles')} cycles", flush=True)
            for k, v in r["buckets_s"].items():
                print(f"  {k:32s} {v:8.3f} s  x{r['buckets_n'][k]}", flush=True)
            save()
    if "profile" not in args.skip:
        for where, call in (("main thread", lambda f: f()), ("fresh thread", on_thread)):
            report[f"profile {where}"] = call(lambda: profiled(engine, PROMPT, args.max_tokens, 25))
            print(f"SERVE_GAP profile on the {where}:\n" + report[f"profile {where}"][:5000], flush=True)
            save()
    save()
    print(f"SERVE_GAP: DONE -> {args.json}", flush=True)
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
