"""The serving fixes on a GPU: the prefill backend measurement, eager token
identity, and served requests over HTTP, each run against whichever source
tree `DRINKME_SRC` names (bench/modal_serve_fixes.py mounts main's `src/` as
the "before" tree beside the branch's).

  --what ttft      serve's engine in-process (serve.build_engine, the pack's
                   compressed arm, DRINKME_CUDA_GRAPHS=0 so no capture sits
                   in the first token), one request per (length, backends),
                   each on a FRESH thread as serve runs a request, the
                   prefix cache reset before each so the whole prompt
                   prefills: max_tokens 1, wall to the result. Backends:
                   torch's own choice, and cuDNN SDPA excluded process-wide
                   (torch.backends.cuda.enable_cudnn_sdp(False)); interleaved,
                   `--reps` of each. Also torch's pick at the prefill shape.
  --what identity  serve's engine in-process, one arm (`--arm stock|compressed`),
                   eager, speculation off, prefix slots 0: greedy requests,
                   the token ids and every row's top-1 minus top-2 logit
                   (engines.sample_next wrapped). `--compare A.json B.json`
                   prints where two such runs part, with the margins there.
  --what serve     `drinkme serve` as a subprocess on a spare port (never
                   3215) under PYTHONPATH=$DRINKME_SRC: the serve_drive prompt
                   three times on one slot, then a second turn that extends
                   the first (the first answer as the assistant turn); per
                   request the text, decode tok/s, TTFT, and the server's
                   [drinkme.spec] and prefix-cache lines.

Verdicts from the printed lines, never the exit status.
"""
from __future__ import annotations

import argparse
import http.client
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.environ.get("DRINKME_SRC") or os.path.join(HERE, "..", "src")
sys.path.insert(0, SRC)
sys.path.insert(0, HERE)

from cuda_graph_serve import long_prompt  # noqa: E402
from serve_drive import PROMPT, SPEC_RE, health  # noqa: E402

PORT = 3293
TURN2 = "Now do the same with this sentence instead: 'A small green frog sat on a lily pad and sang.'"


# ------------------------------------------------------------ in-process --

def _request(messages, max_tokens: int):
    from drinkme.serving.engine import GenerationRequest, SampleParams

    return GenerationRequest(messages, SampleParams(temperature=0.0, max_tokens=max_tokens),
                             template_kwargs={"enable_thinking": False})


def _engine(model: str, pack_dir: str | None, stock: bool, ctx: int, slots: int | None = None):
    from drinkme import serve
    from drinkme.suggest import MODELS

    m = next(x for x in MODELS if x.name == model)
    eng = serve.build_engine(m.hf_repo, m.revision, None if stock else pack_dir, stock=stock, ctx=ctx,
                             prefix_slots=slots)
    serve._warmup(eng)
    return eng


def _on_thread(fn):
    out: dict = {}

    def run():
        try:
            out["v"] = fn()
        except BaseException as e:  # noqa: BLE001 — reported, not raised across threads
            out["e"] = e

    th = threading.Thread(target=run)
    th.start()
    th.join()
    if "e" in out:
        raise out["e"]
    return out["v"]


def sdpa_pick(shape, length: int, rows: int | None = None) -> str:
    """torch's backend, cuDNN allowed, for a causal prefill-shaped call at
    this length, or with `rows` for a decode-shaped one (that many query
    rows, no mask) over `length` keys."""
    import torch
    from torch.nn.attention import SDPBackend

    names = {int(v): k for k, v in SDPBackend.__members__.items()}
    q = torch.zeros(1, shape.n_heads, rows or length, shape.head_dim, device="cuda", dtype=torch.bfloat16)
    k = torch.zeros(1, shape.n_kv_heads, length, shape.head_dim, device="cuda", dtype=torch.bfloat16)
    prev = torch.backends.cuda.cudnn_sdp_enabled()
    torch.backends.cuda.enable_cudnn_sdp(True)
    try:
        c = torch._fused_sdp_choice(q, k, k, None, 0.0, rows is None, scale=None,
                                    enable_gqa=shape.n_kv_heads != shape.n_heads)
    finally:
        torch.backends.cuda.enable_cudnn_sdp(prev)
    return names.get(int(c), str(c))


def ttft(args) -> dict:
    import torch

    from drinkme import sdpa
    from drinkme.serving.engine import complete

    eng = _engine(args.model, args.pack_dir, False, args.ctx)
    shapes = sdpa.shapes_from_config(eng.model.config)
    report: dict = {"model": args.model, "device": torch.cuda.get_device_name(0), "ctx": args.ctx,
                    "torch": torch.__version__, "cudnn_sdp_enabled_at_load": torch.backends.cuda.cudnn_sdp_enabled(),
                    "runs": [], "picks": {}}
    arms = (("torch's choice", True), ("cuDNN excluded", False))
    for length in args.lengths:
        prompt = long_prompt(length)
        n = eng.count_tokens(_request([{"role": "user", "content": prompt}], 1))
        if shapes:
            report["picks"][str(length)] = {str(s): {"prefill": sdpa_pick(s, n), "decode": sdpa_pick(s, n, 1)}
                                            for s, _ in shapes}
        for rep in range(args.reps):
            order = arms if rep % 2 == 0 else arms[::-1]
            for label, cudnn in order:
                eng.reset_prefix_cache()
                torch.cuda.synchronize()
                prev = torch.backends.cuda.cudnn_sdp_enabled()
                torch.backends.cuda.enable_cudnn_sdp(cudnn)
                try:
                    def one():
                        t0 = time.perf_counter()
                        r = complete(eng, _request([{"role": "user", "content": prompt}], 1))
                        torch.cuda.synchronize()
                        return time.perf_counter() - t0, r
                    dt, r = _on_thread(one)
                finally:
                    torch.backends.cuda.enable_cudnn_sdp(prev)
                row = {"length": length, "prompt_tokens": r.prompt_tokens, "arm": label, "rep": rep,
                       "ttft_s": round(dt, 4)}
                report["runs"].append(row)
                print(f"TTFT {args.model} {r.prompt_tokens} prompt tokens, {label}, rep {rep}: "
                      f"{dt * 1e3:.1f} ms", flush=True)
    for length in args.lengths:
        rows = [r for r in report["runs"] if r["length"] == length]
        med = {}
        for label, _ in arms:
            xs = sorted(r["ttft_s"] for r in rows if r["arm"] == label)
            med[label] = xs[len(xs) // 2] if xs else None
        a, b = med[arms[0][0]], med[arms[1][0]]
        print(f"TTFT_SUMMARY {args.model} ~{length} ({rows[0]['prompt_tokens']} tokens): median "
              f"torch's choice {a * 1e3:.1f} ms, cuDNN excluded {b * 1e3:.1f} ms "
              f"({(a / b - 1) * 100:+.1f}% for torch's choice); picks {report['picks'].get(str(length))}",
              flush=True)
    return report


def identity(args) -> dict:
    import torch

    from drinkme.serving import engines
    from drinkme.serving.engine import complete

    os.environ["DRINKME_SPEC"] = "off"
    stock = args.arm == "stock"
    eng = _engine(args.model, args.pack_dir, stock, args.ctx, slots=0)
    rows: list = []
    inner = engines.sample_next

    def wrapped(logits, *a, **k):
        tok = inner(logits, *a, **k)
        top = torch.topk(logits.float(), 2)
        rows.append((int(tok), float(top.values[0] - top.values[1])))
        return tok

    engines.sample_next = wrapped
    report: dict = {"model": args.model, "arm": args.arm, "device": torch.cuda.get_device_name(0),
                    "cudnn_sdp_enabled": torch.backends.cuda.cudnn_sdp_enabled(), "src": SRC, "runs": {}}
    prompts = {"serve_drive": (PROMPT, 256), "long4k": (long_prompt(4096), 128)}
    try:
        for name, (prompt, n) in prompts.items():
            rows.clear()
            r = _on_thread(lambda: complete(eng, _request([{"role": "user", "content": prompt}], n)))
            report["runs"][name] = {"text": r.text, "ids": [t for t, _ in rows],
                                    "margins": [round(m, 4) for _, m in rows], "prompt_tokens": r.prompt_tokens}
            print(f"IDENTITY {args.model} {args.arm} {name}: {len(rows)} tokens, {len(r.text)} chars, "
                  f"cuDNN SDPA enabled {torch.backends.cuda.cudnn_sdp_enabled()}", flush=True)
    finally:
        engines.sample_next = inner
    return report


def compare(a_path: str, b_path: str, what: str) -> None:
    a, b = json.load(open(a_path)), json.load(open(b_path))
    for name in a["runs"]:
        ra, rb = a["runs"][name], b["runs"].get(name)
        if rb is None:
            print(f"VERDICT {what} {a['model']} {name}: MISSING", flush=True)
            continue
        ia, ib = ra["ids"], rb["ids"]
        at = next((i for i, (x, y) in enumerate(zip(ia, ib)) if x != y), None)
        if at is None and len(ia) == len(ib):
            print(f"VERDICT {what} {a['model']} eager {name}: EQUAL ({len(ia)} ids)", flush=True)
        else:
            at = min(len(ia), len(ib)) if at is None else at
            print(f"VERDICT {what} {a['model']} eager {name}: DIFFERS at token {at} of {len(ia)}/{len(ib)} "
                  f"(top-2 margin there {ra['margins'][at] if at < len(ra['margins']) else None} / "
                  f"{rb['margins'][at] if at < len(rb['margins']) else None})", flush=True)


# ---------------------------------------------------------------- served --

def chat(port: int, messages, max_tokens: int) -> dict:
    """serve_drive.request with a message list."""
    body = json.dumps({"messages": messages, "temperature": 0, "max_tokens": max_tokens, "stream": True,
                       "stream_options": {"include_usage": True},
                       "chat_template_kwargs": {"enable_thinking": False}})
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=900)
    t0 = time.perf_counter()
    c.request("POST", "/v1/chat/completions", body, {"Content-Type": "application/json"})
    r = c.getresponse()
    if r.status != 200:
        return {"error": f"{r.status} {r.read()[:400]!r}"}
    t_first = t_last = None
    text, usage, buf = [], None, b""
    while True:
        chunk = r.read1(65536)
        if not chunk:
            break
        now = time.perf_counter()
        buf += chunk
        while b"\n\n" in buf:
            frame, buf = buf.split(b"\n\n", 1)
            if not frame.startswith(b"data: ") or frame == b"data: [DONE]":
                continue
            obj = json.loads(frame[6:])
            usage = obj.get("usage") or usage
            for ch in obj.get("choices", []):
                d = ch.get("delta", {}).get("content")
                if d:
                    t_first = t_first or now
                    t_last = now
                    text.append(d)
    c.close()
    u = usage or {}
    ct = int(u.get("completion_tokens") or 0)
    tps = (ct - 1) / (t_last - t_first) if ct > 1 and t_first and t_last and t_last > t_first else None
    return {"text": "".join(text), "completion_tokens": ct, "prompt_tokens": u.get("prompt_tokens"),
            "ttft_s": None if t_first is None else round(t_first - t0, 4),
            "decode_tok_s": None if tps is None else round(tps, 2)}


def served(args) -> dict:
    env = {**os.environ, "PYTHONPATH": SRC, "DRINKME_SLOT_DIR": os.path.join(args.out, "slots")}
    cmd = [sys.executable, "-c", "import sys; from drinkme.cli import entry; sys.exit(entry())", "serve",
           "--model", args.model, "--port", str(PORT), "--yes",
           *(["--stock"] if args.stock else ["--pack-dir", args.pack_dir]),
           *(["--ctx", str(args.ctx)] if args.ctx else [])]
    log_path = os.path.join(args.out, "server.log")
    where = subprocess.run([sys.executable, "-c", "import drinkme; print(drinkme.__file__)"], env=env,
                           capture_output=True, text=True).stdout.strip()
    print(f"  [{args.label}] the server imports drinkme from {where}", flush=True)
    with open(log_path, "w") as lf:
        server = subprocess.Popen(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT, start_new_session=True)
    report: dict = {"model": args.model, "label": args.label, "src": SRC, "drinkme": where, "cmd": cmd,
                    "env": {k: v for k, v in os.environ.items() if k.startswith("DRINKME_")}, "runs": []}
    try:
        t0 = time.time()
        while time.time() - t0 < 1500:
            if server.poll() is not None:
                report["error"] = f"server exited {server.returncode} before /health"
                break
            if health(PORT):
                break
            time.sleep(2)
        else:
            report["error"] = "no /health in 1500 s"
        if "error" not in report:
            report["boot_s"] = round(time.time() - t0, 1)
            first = [{"role": "user", "content": PROMPT}]
            plan = [("repeat 1", first), ("repeat 2", first), ("repeat 3", first)]
            for i in range(len(plan) + 1):
                if i == len(plan):
                    t1 = report["runs"][0].get("text", "")
                    label, msgs = "turn 2", first + [{"role": "assistant", "content": t1},
                                                     {"role": "user", "content": TURN2}]
                else:
                    label, msgs = plan[i]
                off = os.path.getsize(log_path)
                try:
                    r = chat(PORT, msgs, args.max_tokens)
                except Exception as e:  # noqa: BLE001
                    r = {"error": f"{type(e).__name__}: {str(e)[:400]}"}
                time.sleep(0.5)
                with open(log_path, errors="replace") as f:
                    f.seek(off)
                    new = f.read()
                r["label"] = label
                r["spec_line"] = next((ln for ln in new.splitlines() if "[drinkme.spec]" in ln), None)
                r["cache_lines"] = [ln for ln in new.splitlines() if "prefix cache" in ln]
                m = SPEC_RE.search(r["spec_line"] or "")
                if m:
                    r["cycles"], r["accepted"], r["drafted"] = int(m.group(3)), int(m.group(4)), int(m.group(5))
                    r["acceptance"] = round(r["accepted"] / r["drafted"], 3) if r["drafted"] else None
                report["runs"].append(r)
                print(f"SERVED {args.model} [{args.label}] {label}: "
                      + (f"ERROR {r['error']}" if "error" in r else
                         f"{r['decode_tok_s']} tok/s, ttft {r['ttft_s']} s, {r['completion_tokens']} tokens, "
                         f"{r.get('prompt_tokens')} prompt tokens"
                         + (f", {r['accepted']}/{r['drafted']} drafts accepted ({r['acceptance']:.0%}), "
                            f"{r['cycles']} cycles" if m and r.get("drafted") else "")
                         + (f"; {r['cache_lines'][-1].split('] ', 1)[-1]}" if r["cache_lines"] else "")),
                      flush=True)
    finally:
        try:
            os.killpg(server.pid, signal.SIGTERM)
            server.wait(90)
        except Exception:  # noqa: BLE001
            os.killpg(server.pid, signal.SIGKILL)
        with open(log_path, errors="replace") as f:
            log = f.read()
        report["boot_lines"] = [ln for ln in log.splitlines()
                                if ln.startswith("[drinkme]") and any(w in ln for w in (
                                    "decode step", "prefix cache", "SDPA", "attention", "speculat", "MTP"))][:30]
        report["log_tail"] = log[-8000:]
        for ln in report["boot_lines"]:
            print(f"  [{args.label}] server: {ln}", flush=True)
        if "error" in report:
            print(f"SERVED {args.model} [{args.label}]: ERROR {report['error']}\n{log[-4000:]}", flush=True)
    return report


def compare_served(a_path: str, b_path: str) -> None:
    a, b = json.load(open(a_path)), json.load(open(b_path))
    for ra, rb in zip(a.get("runs", []), b.get("runs", [])):
        ta, tb = ra.get("text"), rb.get("text")
        if ta is None or tb is None:
            print(f"VERDICT served {a['model']} {ra.get('label')}: ERROR", flush=True)
            continue
        at = next((i for i, (x, y) in enumerate(zip(ta, tb)) if x != y), min(len(ta), len(tb)))
        print(f"VERDICT served {a['model']} {ra['label']}: [{b['label']}] text "
              + ("IDENTICAL to" if ta == tb else f"DIFFERS at char {at} from") + f" [{a['label']}] "
              f"({len(tb)} chars, {rb.get('completion_tokens')} tokens)", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--what", choices=("ttft", "identity", "serve"))
    ap.add_argument("--model", default="Qwen3-8B")
    ap.add_argument("--pack-dir", default=None)
    ap.add_argument("--stock", action="store_true")
    ap.add_argument("--arm", default="compressed")
    ap.add_argument("--ctx", type=int, default=8192)
    ap.add_argument("--lengths", type=int, nargs="*", default=[1024, 4096, 16384])
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--label", default="")
    ap.add_argument("--out", default="out")
    ap.add_argument("--json")
    ap.add_argument("--compare", nargs=2)
    ap.add_argument("--compare-served", nargs=2)
    ap.add_argument("--verdict", default="compressed==stock")
    args = ap.parse_args()
    if args.compare:
        compare(*args.compare, args.verdict)
        return
    if args.compare_served:
        compare_served(*args.compare_served)
        return
    os.makedirs(args.out, exist_ok=True)
    import drinkme

    print(f"SERVE_FIXES {args.what}: drinkme from {drinkme.__file__}", flush=True)
    fn = {"ttft": ttft, "identity": identity, "serve": served}[args.what]
    report = fn(args)
    with open(args.json or os.path.join(args.out, f"{args.what}.json"), "w") as f:
        json.dump(report, f, indent=1)
    print(f"SERVE_FIXES {args.what}: DONE", flush=True)
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
