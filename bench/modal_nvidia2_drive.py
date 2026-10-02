"""The local side of bench/modal_nvidia2.py: foreground
commands that each finish well under the 10-minute ceiling. The app is
DEPLOYED once (`modal deploy bench/modal_nvidia2.py`); cells are spawned
against the deployed functions and collected later by call id.

    PY=~/.local/share/uv/tools/modal/bin/python
    $PY bench/modal_nvidia2_drive.py spawn --card A10G          # -> nvidia2/A10G/run-N/call.json
    $PY bench/modal_nvidia2_drive.py collect --card A10G --run run-1 --wait 540
    $PY bench/modal_nvidia2_drive.py spawn27pack --codec sip
    $PY bench/modal_nvidia2_drive.py collect27pack --codec sip --wait 540
    $PY bench/modal_nvidia2_drive.py spawn27 --card-table nvidia2/L40S/run-1/card_table.json
    $PY bench/modal_nvidia2_drive.py collect --card L40S-27b --run run-1 --wait 540
    $PY bench/modal_nvidia2_drive.py status

Receipts, verbatim as the container returned them, under nvidia2/<card>/run-N/
(nvidia2/L40S-27b/run-N/ for the 27B; one spawn per directory, never
overwritten — the A10G pool needs several spawns to draw both silicons):

    call.json                       the spawn (call id, time, arguments)
    cell.json                       the cell report minus the big texts
    cell.log                        the container's full log
    wall.json                       the bandwidth wall (3 readings)
    gates/<name>.{json,log}         the schedule gate (+selftest), the mc gate on the sip pack
    rotation.json / rotation.log    the sweep (rotation27.* for the 27B) and the step's tail
    card_table.json                 the card's row per class (Q1); *_table.json (Q2)
    <codec>.pack.meta.json          the packs' meta.json
    <arm>.bench.json                the bench record, bench's own bytes
    <arm>_spec-{off,on}.json        bench/serve_drive.py's JSON per served run
    <arm>_spec-{off,on}.server.log

A cell whose function raised (Modal timeout, container death) still gets a
FAILED.cell.json with the exception. Verdicts come from the receipts,
never from this script's exit code.
"""

from __future__ import annotations

import argparse
import datetime
import glob
import json
import os
import subprocess
import sys
import time

import modal

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
from modal_nvidia2 import APP_NAME  # noqa: E402

RECEIPTS = os.path.join(REPO, "nvidia2")
BIG = ("log", "record_text", "pack_meta_text", "json", "server_log", "tail", "wall_json")


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()
    except Exception:  # noqa: BLE001
        return ""


def _token() -> str:
    token = os.environ.get("HF_TOKEN", "")
    tp = os.path.expanduser("~/.cache/huggingface/token")
    if not token and os.path.exists(tp):
        token = open(tp).read().strip()
    return token


def _write(path: str, text: str | None) -> None:
    if text is None:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(text)


def _next_run(card_dir: str) -> str:
    os.makedirs(card_dir, exist_ok=True)
    n = 1
    while os.path.exists(os.path.join(card_dir, f"run-{n}")):
        n += 1
    d = os.path.join(card_dir, f"run-{n}")
    os.makedirs(d)
    return d


def spawn(card: str, fn_name: str, kwargs: dict, card_dir: str) -> str:
    fn = modal.Function.from_name(APP_NAME, fn_name)
    kwargs = {"hf_token": _token(), "commit": _commit(), **kwargs}
    fc = fn.spawn(**kwargs)
    d = _next_run(card_dir)
    rec = {"call_id": fc.object_id, "function": fn_name, "card": card, "spawned": _now(),
           "kwargs": {k: v for k, v in kwargs.items() if k != "hf_token"}, "hf_token_passed": bool(kwargs["hf_token"])}
    with open(os.path.join(d, "call.json"), "w") as f:
        json.dump(rec, f, indent=1)
    print(f"[drive] spawned {fn_name}({json.dumps(rec['kwargs'])}) -> {fc.object_id} · {d}", flush=True)
    return d


def collect(card: str, run: str, wait: int) -> int:
    d = os.path.join(RECEIPTS, card, run)
    cpath = os.path.join(d, "call.json")
    if not os.path.exists(cpath):
        print(f"[drive] no spawn record at {cpath}", flush=True)
        return 2
    rec = json.load(open(cpath))
    fc = modal.FunctionCall.from_id(rec["call_id"])
    t0 = time.time()
    try:
        report = fc.get(timeout=wait)
    except TimeoutError:
        print(f"[drive] {card}/{run}: still running after {time.time() - t0:.0f}s of waiting "
              f"(spawned {rec['spawned']}, now {_now()})", flush=True)
        return 3
    except Exception as e:  # noqa: BLE001 — the function raised / timed out / died
        report = {"gpu": card, "error": f"the function raised {type(e).__name__}: {e}", "steps": {}, "step_order": [],
                  "log": f"(no log came back: {type(e).__name__}: {e})\n"}
        _write(os.path.join(d, "FAILED.cell.json"), json.dumps({**report, "collected": _now(), "call": rec}, indent=1))
        _write(os.path.join(d, "FAILED.cell.log"), report["log"])
        print(f"[drive] {card}/{run}: FAILED — {report['error']} -> {d}/FAILED.cell.json", flush=True)
        return 1
    summary = summarize(report)
    _write_receipts(d, report, rec)
    print(f"[drive] {card}/{run}: collected -> {d}", flush=True)
    print(summary, flush=True)
    return 0


def _write_receipts(d: str, report: dict, rec: dict) -> None:
    _write(os.path.join(d, "cell.log"), report.pop("log", None))
    _write(os.path.join(d, "wall.json"), report.pop("wall_json", None))
    for codec, sub in (report.get("packs") or {}).items():
        _write(os.path.join(d, f"{codec}.pack.meta.json"), sub.pop("pack_meta_text", None))
    for name, g in (report.get("gates") or {}).items():
        _write(os.path.join(d, "gates", f"{name}.json"), g.pop("json", None))
        _write(os.path.join(d, "gates", f"{name}.log"), g.pop("tail", None))
    for label, r in (report.get("rotations") or {}).items():
        _write(os.path.join(d, f"{label}.json"), r.pop("json", None))
        step = (report.get("steps") or {}).get(label) or {}
        _write(os.path.join(d, f"{label}.log"), step.get("tail"))
    for name in ("card_table", "sip_27b_table", "gulp_27b_table", "card_table_from_q1"):
        if report.get(name) is not None:
            _write(os.path.join(d, f"{name}.json"), json.dumps(report[name], indent=1) + "\n")
    for label, arm in (report.get("arms") or {}).items():
        _write(os.path.join(d, f"{label}.bench.json"), arm.pop("record_text", None))
        for spec, s in (arm.get("served") or {}).items():
            if not isinstance(s, dict):
                continue
            _write(os.path.join(d, f"{label}_spec-{spec}.json"), s.pop("json", None))
            _write(os.path.join(d, f"{label}_spec-{spec}.server.log"), s.pop("server_log", None))
    for step in (report.get("steps") or {}).values():
        step.pop("tail", None)  # the full log holds every line
    slim = {k: v for k, v in report.items() if k not in ("log",)}
    slim["collected"] = _now()
    slim["call"] = rec
    _write(os.path.join(d, "cell.json"), json.dumps(slim, indent=1))


def summarize(report: dict) -> str:
    lines = [f"  nvidia-smi {report.get('nvidia_smi')!r} · device {report.get('device')!r} · "
             f"torch {(report.get('torch_identity') or {}).get('torch')}",
             f"  wall {report.get('wall_s')} s · steps: " + ", ".join(
                 f"{s} {report['steps'][s]['seconds']}s exit {report['steps'][s]['exit']}"
                 + (" KILLED" if report['steps'][s].get('killed_at_deadline') else "")
                 for s in report.get("step_order", []))]
    if report.get("error"):
        lines.append(f"  ERROR: {report['error']}")
    lines.append(f"  wall: {report.get('wall_verdict')}")
    for name, g in (report.get("gates") or {}).items():
        lines.append(f"  gate {name}: {g.get('verdict')!r} / {g.get('selftest_line')!r} exit {g.get('exit')}")
    for codec, sub in (report.get("packs") or {}).items():
        lines.append(f"  pack {codec}: wall {sub.get('pack_wall_s')} s · du {sub.get('pack_du_bytes')}")
    for label, r in (report.get("rotations") or {}).items():
        lines.append(f"  {label}: {r.get('verdict')} exit {r.get('exit')}")
        try:
            rot = json.loads(r["json"])
            # the rotation's per-class picks (parity_rotation.py's `classes` /
            # `mc_classes`): the best config against the `spike` schedule (t1/w2)
            def _vs_spike(p):
                return f" = {p['best_vs_spike']:.3f}x spike" if "best_vs_spike" in p else ""

            for cls, pk in rot.get("classes", {}).items():
                extra = "".join(f"; {arm} best {p['best']}{_vs_spike(p)}"
                                for arm, p in (pk.get("picks") or {}).items()
                                if p.get("best") != pk.get("best"))
                lines.append(f"    {cls:7s} best {pk['best']}{_vs_spike(pk)}; "
                             f"within 1%: {pk.get('within_1pct')}{extra}")
            for cls, mp in (rot.get("mc_classes") or {}).items():
                lines.append(f"    mc {cls:7s} best {mp['best']}{_vs_spike(mp)}")
        except (ValueError, KeyError, TypeError) as e:
            lines.append(f"    (rotation JSON unreadable: {e})")
    for key in ("card_table", "card_table_same_as_gfx1151", "sip_27b_table", "gulp_27b_table", "arm_plan", "arms_skipped"):
        if key in report:
            lines.append(f"  {key}: {json.dumps(report[key])}")
    for label in report.get("arm_order", []):
        arm = report["arms"][label]
        rt = arm.get("record_text")
        if rt:
            try:
                r = json.loads(rt)
                met = {m["name"]: m["value"] for m in r.get("metrics", [])}
                raw = r.get("raw") or {}
                lines.append(f"  [{label}] bench: stock {met.get('stock_decode_tok_s')} · twin (raw) {raw.get('twin_decode_tok_s')}"
                             f" · compressed {met.get('compressed_decode_tok_s')} {raw.get('compressed_decode_samples')}"
                             f" · read {met.get('read_gb_s')} GB/s · vram_compressed {raw.get('vram_compressed_bytes')}"
                             f" · verified {raw.get('verified_tensors')}/{raw.get('swapped_linears')}"
                             f" · gpu mem {json.dumps(arm.get('bench_gpu_mem'))}")
            except ValueError:
                lines.append(f"  [{label}] bench: record is not JSON")
        else:
            lines.append(f"  [{label}] bench: NO RECORD ({arm.get('bench_error')})")
        for spec, s in (arm.get("served") or {}).items():
            if not isinstance(s, dict) or not s.get("json"):
                lines.append(f"  [{label}] served spec-{spec}: NO JSON ({(s or {}).get('error')}; drive exit "
                             f"{(s or {}).get('drive_exit')}; device {(s or {}).get('device_line')!r})")
                continue
            try:
                j = json.loads(s["json"])
                sm = j["summary"]
                lines.append(f"  [{label}] served spec-{spec}: decode {sm['decode_tok_s_median']} {sm['decode_tok_s_samples']}"
                             f" · ttft {sm['ttft_s_median']} · tokens/step {sm['tokens_per_step_median']}"
                             f" · acceptance {sm['metrics_acceptance']} · sha {sm['text_sha_first_run']}"
                             f" · identical {sm['text_identical_across_runs']} · gpu mem {json.dumps(s.get('gpu_mem'))}")
            except (ValueError, KeyError) as e:
                lines.append(f"  [{label}] served spec-{spec}: JSON unreadable ({e})")
    return "\n".join(lines)


def status() -> int:
    for cpath in sorted(glob.glob(os.path.join(RECEIPTS, "*", "*", "call.json")) + glob.glob(os.path.join(RECEIPTS, "*", "pack27_*.call.json"))):
        rec = json.load(open(cpath))
        d = os.path.dirname(cpath)
        stem = os.path.basename(cpath).replace(".call.json", "")
        have = os.path.exists(os.path.join(d, "cell.json" if stem == "call" else f"{stem}.json"))
        failed = os.path.exists(os.path.join(d, "FAILED.cell.json" if stem == "call" else f"FAILED_{stem}.json"))
        state = "collected" if have else ("FAILED" if failed else "?")
        if state == "?":
            try:
                modal.FunctionCall.from_id(rec["call_id"]).get(timeout=0)
                state = "done, not collected"
            except TimeoutError:
                state = "running"
            except Exception as e:  # noqa: BLE001
                state = f"raised {type(e).__name__}"
        print(f"{os.path.relpath(d, RECEIPTS):>22s} {rec['function']:14s} {rec['call_id']} spawned {rec['spawned']} · {state}",
              flush=True)
    return 0


def collect27_pack(codec: str, wait: int) -> int:
    d = os.path.join(RECEIPTS, "L40S-27b")
    cpath = os.path.join(d, f"pack27_{codec}.call.json")
    rec = json.load(open(cpath))
    fc = modal.FunctionCall.from_id(rec["call_id"])
    try:
        report = fc.get(timeout=wait)
    except TimeoutError:
        print(f"[drive] pack27/{codec}: still running", flush=True)
        return 3
    except Exception as e:  # noqa: BLE001
        _write(os.path.join(d, f"FAILED_pack27_{codec}.json"), json.dumps({"error": f"{type(e).__name__}: {e}", "call": rec}, indent=1))
        print(f"[drive] pack27/{codec}: FAILED {type(e).__name__}: {e}", flush=True)
        return 1
    _write(os.path.join(d, f"pack27_{codec}.log"), report.pop("log", None))
    _write(os.path.join(d, f"pack27_{codec}.pack.meta.json"), report.pop("pack_meta_text", None))
    _write(os.path.join(d, f"pack27_{codec}.json"), json.dumps({**report, "collected": _now(), "call": rec}, indent=1))
    print(f"[drive] pack27/{codec}: wall {report.get('wall_s')} s · pack wall {report.get('pack_wall_s')} s "
          f"exit {report.get('pack_exit')} · du {report.get('pack_du_bytes')} · volume {report.get('volume_committed')}"
          f" · hf {report.get('hf_snapshot_du')}", flush=True)
    for s in report.get("step_order", []):
        st = report["steps"][s]
        print(f"   {s}: {st['seconds']}s exit {st['exit']} killed {st['killed_at_deadline']}", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="verb", required=True)
    sp = sub.add_parser("spawn")
    sp.add_argument("--card", required=True, choices=["L4", "A10G", "L40S", "H100"])
    sp.add_argument("--passes", type=int, default=20)
    sp.add_argument("--repeats", type=int, default=3)
    co = sub.add_parser("collect")
    co.add_argument("--card", required=True)
    co.add_argument("--run", required=True)
    co.add_argument("--wait", type=int, default=540)
    sub.add_parser("status")
    s27p = sub.add_parser("spawn27pack")
    s27p.add_argument("--codec", required=True)
    s27p.add_argument("--model", default="Qwen3.8-27B")
    c27p = sub.add_parser("collect27pack")
    c27p.add_argument("--codec", required=True)
    c27p.add_argument("--wait", type=int, default=540)
    s27 = sub.add_parser("spawn27")
    s27.add_argument("--card-table", default="", help="the L40S row from Q1 (a card_table.json); none = the gfx1151 table")
    s27.add_argument("--arms", default="", help="explicit arm list (comma-separated) instead of the planned one")
    s27.add_argument("--passes", type=int, default=20)
    s27.add_argument("--repeats", type=int, default=3)
    s27.add_argument("--serve-ctx", type=int, default=8192)
    a = ap.parse_args()

    if a.verb == "spawn":
        spawn(a.card, f"nv2_{a.card}", {"passes": a.passes, "repeats": a.repeats}, os.path.join(RECEIPTS, a.card))
        return 0
    if a.verb == "collect":
        return collect(a.card, a.run, a.wait)
    if a.verb == "status":
        return status()
    if a.verb == "spawn27pack":
        d = os.path.join(RECEIPTS, "L40S-27b")
        os.makedirs(d, exist_ok=True)
        fn = modal.Function.from_name(APP_NAME, "pack27")
        fc = fn.spawn(model=a.model, codec=a.codec, hf_token=_token(), commit=_commit())
        rec = {"call_id": fc.object_id, "function": "pack27", "card": "cpu", "codec": a.codec, "model": a.model,
               "spawned": _now()}
        with open(os.path.join(d, f"pack27_{a.codec}.call.json"), "w") as f:
            json.dump(rec, f, indent=1)
        print(f"[drive] spawned pack27({a.model}, {a.codec}) -> {fc.object_id}", flush=True)
        return 0
    if a.verb == "collect27pack":
        return collect27_pack(a.codec, a.wait)
    if a.verb == "spawn27":
        kwargs = {"passes": a.passes, "repeats": a.repeats, "serve_ctx": a.serve_ctx}
        if a.card_table:
            kwargs["card_table"] = json.load(open(a.card_table))
        if a.arms:
            kwargs["arms"] = a.arms.split(",")
        spawn("L40S-27b", "nv2_27B_L40S", kwargs, os.path.join(RECEIPTS, "L40S-27b"))
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
