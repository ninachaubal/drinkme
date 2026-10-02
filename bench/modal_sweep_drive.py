"""The local side of bench/modal_sweep.py for a driver whose commands must
each end within ten minutes (a coding agent's foreground ceiling; a 27B
cell runs half an hour or more). The app is DEPLOYED once per card, a cell
is SPAWNED against it, and the result is COLLECTED by call id in as many
bounded waits as it takes. A collected cell is written by modal_sweep's
own write_cell, exactly as `modal run bench/modal_sweep.py` writes it: the
record bytes verbatim under bench's name (the _2/_3 rule, never an
overwrite), the container log and a meta JSON beside it.

    PY=~/.local/share/uv/tools/modal/bin/python
    AB_GPU=A100-80GB modal deploy bench/modal_sweep.py
    $PY bench/modal_sweep_drive.py spawn --card A100-80GB --model Qwen3.8-27B [--profile gulp] --log-dir <logs>
    $PY bench/modal_sweep_drive.py collect --call <logs>/<stem>.call.json \\
        --out-dir <records> --log-dir <logs> --wait 540      # repeat until collected

The deployed function's card is whatever AB_GPU said at deploy time. The
spawn passes --card as the label, and the container's torch identity must
witness it, so a label that is not the deployed card fails there, named,
before bench runs. --profile (sip, the default, or gulp) rides in the call
JSON, and the call's name carries it (modal_sweep.cell_stem), so a sip and a
gulp cell of one model and card never collide. Collect exits 0 with a
record, 1 without one, 3 while
the cell is still running; verdicts still come from the record and the
log, never from these codes.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
import time

import modal

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import modal_sweep as ms  # noqa: E402


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def spawn(card: str, model: str, log_dir: str, profile: str = "sip") -> int:
    if card not in ms.SHAPES:
        print(f"[drive] --card must be one of {sorted(ms.SHAPES)}", flush=True)
        return 2
    if profile not in ms.PROFILES:
        print(f"[drive] --profile must be one of {ms.PROFILES}", flush=True)
        return 2
    os.makedirs(log_dir, exist_ok=True)
    fn = modal.Function.from_name(ms.app.name, "sweep")
    commit = ms.local_commit()
    token = ms.local_token()
    stamp = ms.utc_stamp()
    fc = fn.spawn(model, token, card, commit, ms.SHAPES[card]["timeout"], profile)
    rec = {"call_id": fc.object_id, "app": ms.app.name, "card": card, "model": model, "profile": profile,
           "commit": commit, "shape": ms.SHAPES[card], "hf_token_passed": bool(token), "spawned": _now(),
           "spawned_epoch": time.time(), "stamp": stamp}
    path = os.path.join(log_dir, f"{ms.cell_stem(model, card, profile, stamp)}.call.json")
    with open(path, "w") as f:
        json.dump(rec, f, indent=1)
    print(f"[drive] spawned sweep({model}, {card}, {profile}) -> {fc.object_id} · {path}", flush=True)
    return 0


def collect(call: str, out_dir: str, log_dir: str, wait: int) -> int:
    rec = json.load(open(call))
    profile = rec.get("profile", "sip")  # a call JSON from before --profile was a sip cell
    fc = modal.FunctionCall.from_id(rec["call_id"])
    t0 = time.time()
    try:
        report = fc.get(timeout=wait)
    except TimeoutError:
        print(f"[drive] {rec['model']} {profile} on {rec['card']}: still running after {time.time() - t0:.0f}s of "
              f"waiting ({time.time() - rec['spawned_epoch']:.0f}s since the spawn at {rec['spawned']})",
              flush=True)
        return 3
    except Exception as e:  # noqa: BLE001 — the function raised / timed out / died
        report = {"gpu": rec["card"], "model": rec["model"], "profile": profile, "commit": rec["commit"],
                  "shape": rec["shape"],
                  "steps": {}, "step_order": [], "error": f"the function raised {type(e).__name__}: {e}",
                  "log": f"(no log came back: {type(e).__name__}: {e})\n"}
    report["collected"] = _now()
    report["spawn_to_collect_s"] = round(time.time() - rec["spawned_epoch"], 1)
    report["call"] = rec
    report["app_id"] = rec["app"]
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    return ms.write_cell(report, rec["model"], rec["card"], out_dir, log_dir, rec["stamp"], profile)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="verb", required=True)
    sp = sub.add_parser("spawn")
    sp.add_argument("--card", required=True)
    sp.add_argument("--model", required=True)
    sp.add_argument("--profile", default="sip", help="sip (bench's in-memory pack) or gulp (a cut pack)")
    sp.add_argument("--log-dir", required=True)
    co = sub.add_parser("collect")
    co.add_argument("--call", required=True)
    co.add_argument("--out-dir", required=True)
    co.add_argument("--log-dir", required=True)
    co.add_argument("--wait", type=int, default=540)
    a = ap.parse_args()
    if a.verb == "spawn":
        return spawn(a.card, a.model, a.log_dir, a.profile)
    return collect(a.call, a.out_dir, a.log_dir, a.wait)


if __name__ == "__main__":
    raise SystemExit(main())
