"""The speculative-decoding range on the L40S (48 GB): Qwen3.8-27B sip on
the tree this is run from, bench/ngram_gpu_ab.py's arms off and mtp over
its four prompts (a range reads agent-transcript, chat and code;
repetitive is recorded and left out of it), 256 tokens, 3 reps interleaved
on one engine build, nvidia-smi memory sampled through every GPU step.
Compressed only: the 27B's BF16 does not fit 48 GB.

One container, `spec_L40S`, on bench/modal_radix.py's image (the cuda
environment prebuilt from the lock, g++ for the radix native encoder) with
bench/modal_nvidia2.py's Cell (the capped log, the deadline reaper, the
identity step, the GPU memory sampler):

    0. identity: nvidia-smi; `uv sync --frozen --group cuda` against the
       prebuilt env; bootstrap.ensure_accelerator; torch identity (the
       device must witness the L40S, drinkme must import from the mounted
       checkout, the native encoder must compile); `drinkme --version`
    1. `drinkme pack --model <M> --sip -o <container disk>`: the checkpoint
       comes from HF inside the container and the pack is this tree's own
       cut (no volume's pack is read)
    2. bench/ngram_gpu_ab.py --pack <that pack> --arms off,mtp --reps 3
       --tokens 256, ctx the tool's default (16384, as the Strix range's
       runs) -> its receipt JSON, memory sampled
    3. the served default: `drinkme serve --pack-dir <pack>` with no --ctx
       (serve's own cap) and speculation as shipped, driven by
       bench/serve_drive.py, 3 runs x 256 tokens, memory sampled: the peak
       GPU memory at the served context. Its tok/s is serve_drive's
       repeat-this-sentence prompt: informational, never
       in the range.

The local side is this file run as a script with the modal tool's python,
after `modal deploy bench/modal_spec_range.py`: `spawn`, then `collect` in
bounded waits (each command ends within ten minutes). Receipts, verbatim
as the container returned them, under --out: call.json, cell.json (the
report minus the big texts), cell.log, ngram_ab.json, pack.meta.json,
serve.json, serve.server.log.

    PY=~/.local/share/uv/tools/modal/bin/python
    modal deploy bench/modal_spec_range.py
    $PY bench/modal_spec_range.py spawn --out <dir>
    $PY bench/modal_spec_range.py collect --out <dir> --wait 540

Verdicts come from OUTPUT (the tool's VERDICT block, the pack's summary,
the identity JSON), never from exit codes.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time

import modal

HERE = os.path.dirname(os.path.abspath(__file__))
# locally this file sits beside modal_nvidia2.py; in the container Modal
# mounts the entry module at /root/ and the checkout at /root/drinkme
sys.path.insert(0, HERE)
sys.path.insert(0, "/root/drinkme/bench")
from modal_nvidia2 import Cell  # noqa: E402
from modal_radix import HOME_DIR, OUT_DIR, REMOTE_REPO, SERVE_PORT, _read, image  # noqa: E402

APP_NAME = "drinkme-spec-range"
app = modal.App(APP_NAME)

TIMEOUT = 3600
MODEL = "Qwen3.8-27B"          # the menu name: `drinkme pack` resolves repo + revision
REPO_ID = "Qwen/Qwen3.8-27B"
REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"  # suggest.MENU's pin for the 27B
PROMPTS = ("agent-transcript", "chat", "code", "repetitive")
RANGE_PROMPTS = ("agent-transcript", "chat", "code")


def summarize_ab(receipt: dict) -> dict:
    """Per prompt and arm: the median tok/s, each rep's, the lower-median
    run's accepted/drafted and cycles, and tokens per step for mtp
    (cycles + accepted over cycles: one verify step emits its accepted
    drafts plus one token); the range over RANGE_PROMPTS' mtp medians."""
    out = {"prompts": {}, "verdict": receipt.get("verdict"), "fails": receipt.get("fails"),
           "notes": receipt.get("notes"), "warmup": (receipt.get("warmup") or {}).get("decode_toks_per_s")}
    for name, cells in (receipt.get("prompts") or {}).items():
        row = {}
        for arm, cell in cells.items():
            spec = cell.get("spec") or {}
            cyc, acc = spec.get("cycles"), spec.get("accepted")
            row[arm] = {"median": cell.get("decode_toks_per_s"), "runs": cell.get("decode_toks_per_s_runs"),
                        "speedup": cell.get("speedup"), "tokens": cell.get("tokens"),
                        "accepted": acc, "drafted": spec.get("drafted"), "cycles": cyc,
                        "tokens_per_step": round((cyc + acc) / cyc, 3) if cyc else None,
                        "identity": cell.get("identity") if isinstance(cell.get("identity"), str)
                        else (cell.get("identity") or {}).get("verdict")}
        out["prompts"][name] = row
    mtp = [out["prompts"][p]["mtp"]["median"] for p in RANGE_PROMPTS
           if "mtp" in out["prompts"].get(p, {})]
    if len(mtp) == len(RANGE_PROMPTS):
        out["mtp_range"] = [min(mtp), max(mtp)]
    return out


def served_default(c: Cell, pack_dir: str, runs: int, max_tokens: int) -> None:
    """`drinkme serve` as shipped (no --ctx, speculation and prefix cache
    at their defaults) + bench/serve_drive.py, memory sampled through the
    boot and the drive; the server log's ctx and spec lines kept."""
    slog, sout = f"{OUT_DIR}/serve.server.log", f"{OUT_DIR}/serve.json"
    cmd = ["uv", "run", "--no-sync", "drinkme", "serve", "--model", REPO_ID, "--pack-dir", pack_dir,
           "--port", str(SERVE_PORT), "--yes"]
    senv = dict(c.env)
    senv.pop("DRINKME_SPEC", None)
    c.log(f"\n=== [{c.elapsed():7.1f}s] serve (as shipped): {' '.join(cmd)} -> {slog}\n")
    t_boot = time.time()
    with open(slog, "w") as lf:
        server = subprocess.Popen(cmd, cwd=REMOTE_REPO, env=senv, stdout=lf, stderr=subprocess.STDOUT,
                                  start_new_session=True)
    wait = max(60, int(min(900, c.remaining() - 120)))
    step = "drive served default"
    rc = c.run(step, ["uv", "run", "--no-sync", "python", "bench/serve_drive.py", "--port", str(SERVE_PORT),
                      "--label", "L40S_27b-sip_served-default", "--server-log", slog, "--out", sout,
                      "--runs", str(runs), "--max-tokens", str(max_tokens), "--wait", str(wait)],
               tail_cap=8000, sample_gpu=True)
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(server.pid, sig)
        except ProcessLookupError:
            break
        for _ in range(15):
            if server.poll() is not None:
                break
            time.sleep(1)
        if server.poll() is not None:
            break
    server_log = _read(slog) or ""
    c.report["served"] = {
        "drive_exit": rc, "boot_and_drive_s": round(time.time() - t_boot, 1), "json": _read(sout),
        "server_log": server_log[-400_000:], "server_exit": server.returncode,
        "ctx_lines": [l for l in server_log.splitlines() if "ctx" in l.lower()][:12],
        "spec_lines": [l for l in server_log.splitlines() if "[drinkme.spec]" in l][-8:],
        "gpu_mem": c.report["steps"][step].get("gpu_mem")}
    c.log(f"--- served default: drive exit {rc}; gpu mem {json.dumps(c.report['served']['gpu_mem'])}\n")


@app.function(image=image, gpu="L40S", timeout=TIMEOUT, memory=96 * 1024, cpu=16, single_use_containers=True)
def spec_L40S(hf_token: str = "", commit: str = "", reps: int = 3, tokens: int = 256,
              served_runs: int = 3) -> dict:
    c = Cell("L40S", "the speculative-decoding range: the 27B sip, ngram_gpu_ab off/mtp x 4 prompts, "
                     "+ the served default's memory", TIMEOUT, hf_token, commit)
    # the shape this function was given (Cell's is modal_radix's L40S row, 48 GiB)
    c.report.update(model=MODEL, revision=REVISION, reps=reps, tokens=tokens,
                    shape={"gpu": "L40S", "memory": 96 * 1024, "cpu": 16, "timeout": TIMEOUT})
    err = c.identity()
    if err:
        return c.done(err)
    c.run("drinkme --version", ["uv", "run", "--no-sync", "drinkme", "--version"])
    c.report["drinkme_version"] = (c.report["steps"]["drinkme --version"]["tail"] or "").strip().splitlines()[-1:]
    pack_dir = f"{HOME_DIR}/packs/Qwen--{MODEL}-sip"
    err = c.pack(MODEL, "sip", pack_dir)
    if err:
        return c.done(err)
    tail = c.report["steps"]["drinkme pack --sip"]["tail"]
    c.report["pack_summary_lines"] = [l for l in tail.splitlines()
                                      if "bit-exact" in l or "verified" in l or "bpw" in l][-6:]
    ab = f"{OUT_DIR}/ngram_ab.json"
    rc = c.run("ngram_gpu_ab off,mtp", ["uv", "run", "--no-sync", "python", "bench/ngram_gpu_ab.py",
                                        "--model", REPO_ID, "--revision", REVISION, "--pack", pack_dir,
                                        "--arms", "off,mtp", "--prompts", ",".join(PROMPTS),
                                        "--reps", str(reps), "--tokens", str(tokens), "--json", ab],
               tail_cap=20000, sample_gpu=True)
    text = _read(ab)
    c.report["ngram_ab"] = {"exit": rc, "json": text,
                            "gpu_mem": c.report["steps"]["ngram_gpu_ab off,mtp"].get("gpu_mem")}
    if not text:
        return c.done(f"ngram_gpu_ab wrote no receipt (exit {rc})")
    c.report["ngram_ab"]["summary"] = summarize_ab(json.loads(text))
    c.log(f"\n--- ngram_gpu_ab summary: {json.dumps(c.report['ngram_ab']['summary'])}\n")
    if c.remaining() < 480:
        c.report["served"] = {"error": "skipped: under 480 s to the deadline"}
        c.log("\n--- served default: SKIPPED, under 480 s to the deadline\n")
    else:
        served_default(c, pack_dir, served_runs, tokens)
    return c.done()


# ----------------------------------------------------------- the local side --


def _commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=os.path.join(HERE, ".."), capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:  # noqa: BLE001
        return ""


def _token() -> str:
    token = os.environ.get("HF_TOKEN", "")
    tp = os.path.expanduser("~/.cache/huggingface/token")
    if not token and os.path.exists(tp):
        token = open(tp).read().strip()
    return token


def _write(path: str, text: str | None) -> None:
    if text is not None:
        with open(path, "w") as f:
            f.write(text)


def _now() -> str:
    import datetime

    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def local_main(argv: list[str]) -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="verb", required=True)
    sp = sub.add_parser("spawn")
    sp.add_argument("--out", required=True)
    sp.add_argument("--reps", type=int, default=3)
    sp.add_argument("--tokens", type=int, default=256)
    co = sub.add_parser("collect")
    co.add_argument("--out", required=True)
    co.add_argument("--wait", type=int, default=540)
    a = ap.parse_args(argv)
    cpath = os.path.join(a.out, "call.json")
    if a.verb == "spawn":
        if os.path.exists(cpath):
            print(f"[spec] {cpath} exists: one spawn per directory", flush=True)
            return 2
        os.makedirs(a.out, exist_ok=True)
        fn = modal.Function.from_name(APP_NAME, "spec_L40S")
        kw = {"hf_token": _token(), "commit": _commit(), "reps": a.reps, "tokens": a.tokens}
        fc = fn.spawn(**kw)
        rec = {"call_id": fc.object_id, "function": "spec_L40S", "card": "L40S", "spawned": _now(),
               "spawned_epoch": time.time(), "kwargs": {k: v for k, v in kw.items() if k != "hf_token"},
               "hf_token_passed": bool(kw["hf_token"])}
        _write(cpath, json.dumps(rec, indent=1))
        print(f"[spec] spawned spec_L40S -> {fc.object_id} · {cpath}", flush=True)
        return 0
    rec = json.load(open(cpath))
    t0 = time.time()
    try:
        report = modal.FunctionCall.from_id(rec["call_id"]).get(timeout=a.wait)
    except TimeoutError:
        print(f"[spec] still running after {time.time() - t0:.0f}s of waiting "
              f"({time.time() - rec['spawned_epoch']:.0f}s since the spawn)", flush=True)
        return 3
    except Exception as e:  # noqa: BLE001
        _write(os.path.join(a.out, "FAILED.cell.json"),
               json.dumps({"error": f"the function raised {type(e).__name__}: {e}", "call": rec,
                           "collected": _now()}, indent=1))
        print(f"[spec] FAILED: the function raised {type(e).__name__}: {e}", flush=True)
        return 1
    _write(os.path.join(a.out, "cell.log"), report.pop("log", None))
    ab = report.get("ngram_ab") or {}
    _write(os.path.join(a.out, "ngram_ab.json"), ab.pop("json", None))
    sv = report.get("served") or {}
    _write(os.path.join(a.out, "serve.json"), sv.pop("json", None))
    _write(os.path.join(a.out, "serve.server.log"), sv.pop("server_log", None))
    for codec, sub_ in (report.get("packs") or {}).items():
        _write(os.path.join(a.out, f"{codec}.pack.meta.json"), sub_.pop("pack_meta_text", None))
    for step in (report.get("steps") or {}).values():
        step.pop("tail", None)  # the full log holds every line
    report.update(collected=_now(), spawn_to_collect_s=round(time.time() - rec["spawned_epoch"], 1), call=rec)
    _write(os.path.join(a.out, "cell.json"), json.dumps(report, indent=1))
    print(f"[spec] collected -> {a.out}: wall {report.get('wall_s')} s"
          + (f" · ERROR {report['error']}" if report.get("error") else ""), flush=True)
    print(json.dumps({"steps": {s: (report['steps'][s]['seconds'], report['steps'][s]['exit'],
                                    report['steps'][s].get('gpu_mem'))
                                for s in report.get("step_order", [])},
                      "version": report.get("drinkme_version"), "device": report.get("device"),
                      "pack": report.get("pack_summary_lines"), "summary": ab.get("summary"),
                      "served_gpu_mem": sv.get("gpu_mem"), "served_ctx_lines": sv.get("ctx_lines")},
                     indent=1), flush=True)
    return 0 if not report.get("error") else 1


if __name__ == "__main__":
    raise SystemExit(local_main(sys.argv[1:]))
