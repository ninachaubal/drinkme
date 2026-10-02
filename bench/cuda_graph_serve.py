"""A served request in CUDA-graph mode against the same request eager:
`drinkme serve` over a pack on a spare port (never 3215), once under
DRINKME_CUDA_GRAPHS=0 and once with the default, each answering the same
greedy streamed OpenAI chat request (bench/serve_drive.py's request: one
fixed prompt, thinking off, temperature 0) three times; the first answer's
text is compared across the two servers, and the server's own startup
line naming the decode mode is quoted.

    python bench/cuda_graph_serve.py --model Qwen3-8B --pack-dir /vol/packs/Qwen3-8B-sip \\
        --json out/serve.json [--spec off]

`--long N` (the fit acceptance): each server answers a streamed request
whose prompt is about N tokens, then the short request on the same slot;
`--ctx 0` serves the model's own window (serve's default). Each answer's
text is compared across the two servers, the device memory in use is read
after each request, and a server that dies or answers an error is said.

Verdicts from the printed VERDICT lines, never the exit status.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from serve_drive import health, request  # noqa: E402

PORT = 3291


def long_prompt(tokens: int) -> str:
    """About `tokens` tokens of numbered lines (about 21 Qwen tokens each), then
    a question only the lines answer."""
    colors = ("red", "blue", "green", "grey", "white", "black", "amber", "violet")
    animals = ("fox", "owl", "heron", "otter", "badger", "wren", "hare", "stoat", "newt")
    places = ("river", "mill", "bridge", "orchard", "quarry", "harbour", "chapel")
    lines = [f"Line {i}: the {colors[i % 8]} {animals[i % 9]} counted {i * 7} stones by the "
             f"{places[i % 7]}." for i in range(1, tokens // 21 + 1)]
    k = len(lines) * 3 // 5
    return ("Here is a long log.\n" + "\n".join(lines) + f"\n\nWhich animal is on line {k}, and how many "
            f"stones did it count? Answer in one sentence, then copy lines {k} to {k + 4} exactly.")


def gpu_used_mib() -> int | None:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=30).stdout
        return int(out.split()[0])
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return None


def serve_once(model: str, pack_dir: str, env: dict, log_path: str, max_tokens: int, n: int,
               ctx: int = 8192, long: int = 0) -> dict:
    cmd = ["uv", "run", "--no-sync", "drinkme", "serve", "--model", model, "--pack-dir", pack_dir,
           "--port", str(PORT), *(["--ctx", str(ctx)] if ctx else []), "--yes"]
    with open(log_path, "w") as lf:
        server = subprocess.Popen(cmd, env={**os.environ, **env}, stdout=lf, stderr=subprocess.STDOUT,
                                  start_new_session=True)
    out: dict = {"cmd": cmd, "env": env}
    try:
        t0 = time.time()
        while time.time() - t0 < 900:
            if server.poll() is not None:
                out["error"] = f"server exited {server.returncode} before /health"
                return out
            if health(PORT):
                break
            time.sleep(2)
        else:
            out["error"] = "no /health in 900 s"
            return out
        out["boot_s"] = round(time.time() - t0, 1)
        out["used_mib_boot"] = gpu_used_mib()
        prompts = [long_prompt(long), None] if long else [None] * n
        out["runs"] = []
        for prompt in prompts:
            try:
                r = request(PORT, max_tokens) if prompt is None else request(PORT, max_tokens, prompt)
            except Exception as e:  # noqa: BLE001 — an OOM'd server answers 500 or drops the socket
                r = {"error": f"{type(e).__name__}: {str(e)[:500]}", "server_alive": server.poll() is None}
            r["used_mib_after"] = gpu_used_mib()
            out["runs"].append(r)
    finally:
        try:
            os.killpg(server.pid, signal.SIGTERM)
            server.wait(60)
        except Exception:  # noqa: BLE001
            os.killpg(server.pid, signal.SIGKILL)
        with open(log_path, errors="replace") as f:
            log = f.read()
        out["decode_lines"] = [ln for ln in log.splitlines() if "decode step" in ln]
        out["fit_lines"] = [ln for ln in log.splitlines()
                            if any(w in ln for w in ("prefix cache:", "context checkpoints:", "warning",
                                                     "OutOfMemory", "out of memory", "refusing", "capture"))][:20]
        out["spec_lines"] = [ln for ln in log.splitlines() if "[drinkme.spec]" in ln][:6]
        out["log_tail"] = log[-6000:]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen3-8B")
    ap.add_argument("--pack-dir", required=True)
    ap.add_argument("--json", required=True)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--spec", default=None, help="DRINKME_SPEC for both servers (unset = serve's default)")
    ap.add_argument("--ctx", type=int, default=8192, help="--ctx for both servers; 0 = the model's own window")
    ap.add_argument("--long", type=int, default=0, help="a ~N-token request, then the short one, per server")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
    base = {} if args.spec is None else {"DRINKME_SPEC": args.spec}
    report: dict = {"model": args.model, "spec": args.spec or "<default>"}
    for mode, env in (("eager", {"DRINKME_CUDA_GRAPHS": "0"}), ("graph", {})):
        # each server its own cold tier: the second must prefill, not restore the first's slot
        slots = os.path.join(os.path.dirname(os.path.abspath(args.json)), f"slots_{mode}")
        r = serve_once(args.model, args.pack_dir, {**base, **env, "DRINKME_SLOT_DIR": slots},
                       os.path.join(os.path.dirname(os.path.abspath(args.json)), f"serve_{mode}.log"),
                       args.max_tokens, args.n, args.ctx, args.long)
        report[mode] = r
        for ln in r.get("decode_lines", []) + (r.get("fit_lines", []) if args.long else []):
            print(f"  [{mode}] server: {ln}", flush=True)
        if "error" in r:
            print(f"SERVE {mode}: ERROR {r['error']}", flush=True)
            print(r.get("log_tail", "")[-3000:], flush=True)
        else:
            for i, x in enumerate(r["runs"]):
                if "error" in x:
                    print(f"SERVE {mode} request {i}: ERROR {x['error']} (server alive {x.get('server_alive')})",
                          flush=True)
                    print(r.get("log_tail", "")[-3000:], flush=True)
            ok = [x for x in r["runs"] if "error" not in x]
            print(f"SERVE {mode}: boot {r['boot_s']} s; decode tok/s "
                  f"{[round(x['decode_tok_s'], 2) for x in ok if x.get('decode_tok_s')]}; prompt tokens "
                  f"{[x.get('prompt_tokens') for x in ok]}; completion tokens {[x['completion_tokens'] for x in ok]}; "
                  f"ttft {[round(x['ttft_s'], 3) for x in ok if x.get('ttft_s') is not None]}; GPU MiB in use: "
                  f"boot {r.get('used_mib_boot')}, after each {[x.get('used_mib_after') for x in r['runs']]}",
                  flush=True)
        with open(args.json, "w") as f:
            json.dump(report, f, indent=1)
    e, g = report["eager"].get("runs"), report["graph"].get("runs")
    if args.long and e and g:
        for i, what in enumerate((f"~{args.long}-token request", "short request after it, same slot")):
            if i >= len(e) or i >= len(g) or any("error" in x[i] or not x[i].get("completion_tokens")
                                                 for x in (e, g)):
                print(f"VERDICT serve {args.model} {what}: ERROR (eager "
                      f"{e[i].get('error', 'no tokens') if i < len(e) else 'missing'}; graph "
                      f"{g[i].get('error', 'no tokens') if i < len(g) else 'missing'})", flush=True)
                continue
            te, tg = e[i]["text"], g[i]["text"]
            at = next((j for j, (a, b) in enumerate(zip(te, tg)) if a != b), min(len(te), len(tg)))
            print(f"VERDICT serve {args.model} {what} ({g[i].get('prompt_tokens')} prompt tokens): graph-mode "
                  "streamed text " + ("IDENTICAL to eager" if te == tg else f"DIFFERS from eager at char {at}")
                  + f" ({len(tg)} chars, {g[i]['completion_tokens']} tokens)", flush=True)
        with open(args.json, "w") as f:
            json.dump(report, f, indent=1)
        return
    if e and g:
        te, tg = e[0]["text"], g[0]["text"]
        same = te == tg
        at = next((i for i, (a, b) in enumerate(zip(te, tg)) if a != b), min(len(te), len(tg)))
        repeat = all(x["text"] == tg for x in g)
        print(f"VERDICT serve {args.model} (spec {report['spec']}): graph-mode streamed text "
              + ("IDENTICAL to eager" if same else f"DIFFERS from eager at char {at}")
              + f" ({len(tg)} chars, {g[0]['completion_tokens']} tokens); graph runs agree with each other: {repeat}",
              flush=True)
        report["same_text"], report["graph_runs_agree"] = same, repeat
        with open(args.json, "w") as f:
            json.dump(report, f, indent=1)


if __name__ == "__main__":
    main()
