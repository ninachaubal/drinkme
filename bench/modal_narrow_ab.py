"""The narrow GEMV on NVIDIA: bench/narrow_linear_knobs.py's A/B and
bench/narrow_linear_numerics.py --synthetic on ONE L4, where F.linear is
cuBLAS (the knobs' `rocblas` route is PyTorch's default library, cuBLAS
on a CUDA build; `hipblaslt` is cuBLASLt). No checkpoint is read: the
knobs time random weights, and the numerics check seeded ones at the three
narrow shapes.

One container, `narrow_ab_L4`, on bench/modal_radix.py's image (the cuda
environment prebuilt from the lock, g++ for the radix native encoder) with
bench/modal_nvidia2.py's Cell (the capped log, the deadline reaper, the
identity step):

    0. identity: nvidia-smi; `uv sync --frozen --group cuda` against the
       prebuilt env; bootstrap.ensure_accelerator; torch identity (the
       device must witness the L4, drinkme must import from the mounted
       checkout)
    1. bench/narrow_linear_numerics.py --synthetic 8 -> numerics.json
    2. bench/narrow_linear_knobs.py --routes <routes>, `passes` times in a
       row -> knobs-<i>.json

The local side is this file run as a script with the modal tool's python,
after `modal deploy bench/modal_narrow_ab.py`: `spawn`, then `collect` in
bounded waits (each command ends within ten minutes), then stop the app.
Receipts, verbatim as the container returned them, under --out: call.json,
cell.json (the report minus the big texts), cell.log, numerics.json,
knobs-<i>.json.

    PY=~/.local/share/uv/tools/modal/bin/python
    modal deploy bench/modal_narrow_ab.py
    $PY bench/modal_narrow_ab.py spawn --out <dir>
    $PY bench/modal_narrow_ab.py collect --out <dir> --wait 540
    modal app stop drinkme-narrow-ab --yes

The function's timeout is 25 minutes, so one call costs at most that many
L4 minutes. Verdicts come from OUTPUT (each tool's verdict line, the
identity JSON), never from exit codes.
"""

from __future__ import annotations

import json
import os
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
from modal_radix import OUT_DIR, _read, image  # noqa: E402

APP_NAME = "drinkme-narrow-ab"
app = modal.App(APP_NAME)

TIMEOUT = 1500
ROUTES = ("triton", "rocblas", "hipblaslt")


@app.function(image=image, gpu="L4", timeout=TIMEOUT, memory=16 * 1024, cpu=16, single_use_containers=True)
def narrow_ab_L4(commit: str = "", passes: int = 3, routes: list | None = None) -> dict:
    routes = list(routes or ROUTES)
    c = Cell("L4", "the narrow GEMV A/B: narrow_linear_numerics --synthetic, narrow_linear_knobs x passes",
             TIMEOUT, "", commit)
    c.report.update(passes=passes, routes=routes,
                    shape={"gpu": "L4", "memory": 16 * 1024, "cpu": 16, "timeout": TIMEOUT})
    err = c.identity()
    if err:
        return c.done(err)
    out = f"{OUT_DIR}/numerics.json"
    c.run("narrow_linear_numerics --synthetic 8", ["uv", "run", "--no-sync", "python",
                                                   "bench/narrow_linear_numerics.py", "--synthetic", "8",
                                                   "--json", out], tail_cap=8000)
    c.report["numerics"] = {"json": _read(out), "verdict": c.verdict_line(
        c.report["steps"]["narrow_linear_numerics --synthetic 8"]["tail"], "NARROW_LINEAR_NUMERICS")}
    c.report["knobs"] = []
    for i in range(passes):
        if c.remaining() < 360:
            c.log(f"\n--- knobs pass {i + 1}: SKIPPED, under 360 s to the deadline\n")
            break
        out = f"{OUT_DIR}/knobs-{i + 1}.json"
        label = f"narrow_linear_knobs pass {i + 1}"
        c.run(label, ["uv", "run", "--no-sync", "python", "bench/narrow_linear_knobs.py",
                      "--routes", *routes, "--json", out], tail_cap=12000)
        c.report["knobs"].append({"json": _read(out), "verdict": c.verdict_line(
            c.report["steps"][label]["tail"], "NARROW_LINEAR_KNOBS")})
    return c.done()


# ----------------------------------------------------------- the local side --


def _commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=os.path.join(HERE, ".."), capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:  # noqa: BLE001
        return ""


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
    sp.add_argument("--passes", type=int, default=3)
    sp.add_argument("--routes", nargs="*", default=list(ROUTES))
    co = sub.add_parser("collect")
    co.add_argument("--out", required=True)
    co.add_argument("--wait", type=int, default=540)
    a = ap.parse_args(argv)
    cpath = os.path.join(a.out, "call.json")
    if a.verb == "spawn":
        if os.path.exists(cpath):
            print(f"[narrow] {cpath} exists: one spawn per directory", flush=True)
            return 2
        os.makedirs(a.out, exist_ok=True)
        fn = modal.Function.from_name(APP_NAME, "narrow_ab_L4")
        kw = {"commit": _commit(), "passes": a.passes, "routes": a.routes}
        fc = fn.spawn(**kw)
        rec = {"call_id": fc.object_id, "function": "narrow_ab_L4", "card": "L4", "spawned": _now(),
               "spawned_epoch": time.time(), "kwargs": kw}
        _write(cpath, json.dumps(rec, indent=1))
        print(f"[narrow] spawned narrow_ab_L4 -> {fc.object_id} · {cpath}", flush=True)
        return 0
    rec = json.load(open(cpath))
    t0 = time.time()
    try:
        report = modal.FunctionCall.from_id(rec["call_id"]).get(timeout=a.wait)
    except TimeoutError:
        print(f"[narrow] still running after {time.time() - t0:.0f}s of waiting "
              f"({time.time() - rec['spawned_epoch']:.0f}s since the spawn)", flush=True)
        return 3
    except Exception as e:  # noqa: BLE001
        _write(os.path.join(a.out, "FAILED.cell.json"),
               json.dumps({"error": f"the function raised {type(e).__name__}: {e}", "call": rec,
                           "collected": _now()}, indent=1))
        print(f"[narrow] FAILED: the function raised {type(e).__name__}: {e}", flush=True)
        return 1
    _write(os.path.join(a.out, "cell.log"), report.pop("log", None))
    num = report.get("numerics") or {}
    _write(os.path.join(a.out, "numerics.json"), num.pop("json", None))
    for i, k in enumerate(report.get("knobs") or []):
        _write(os.path.join(a.out, f"knobs-{i + 1}.json"), k.pop("json", None))
    for step in (report.get("steps") or {}).values():
        step.pop("tail", None)  # the full log holds every line
    report.update(collected=_now(), spawn_to_collect_s=round(time.time() - rec["spawned_epoch"], 1), call=rec)
    _write(os.path.join(a.out, "cell.json"), json.dumps(report, indent=1))
    print(f"[narrow] collected -> {a.out}: wall {report.get('wall_s')} s"
          + (f" · ERROR {report['error']}" if report.get("error") else ""), flush=True)
    print(json.dumps({"steps": {s: (report['steps'][s]['seconds'], report['steps'][s]['exit'])
                                for s in report.get("step_order", [])},
                      "device": report.get("device"), "numerics": num.get("verdict"),
                      "knobs": [k.get("verdict") for k in report.get("knobs") or []]}, indent=1), flush=True)
    return 0 if not report.get("error") else 1


if __name__ == "__main__":
    raise SystemExit(local_main(sys.argv[1:]))
