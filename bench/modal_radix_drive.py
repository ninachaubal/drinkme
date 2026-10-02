"""The local side of bench/modal_radix.py, split into foreground commands
that each finish well under the 10-minute ceiling (never
background-and-wait). The app is DEPLOYED once (`modal deploy
bench/modal_radix.py`); cells are spawned against the deployed functions
and collected later by call id, so a 25-minute cell keeps running on
Modal's side between two short local calls.

    PY=~/.local/share/uv/tools/modal/bin/python     # the modal tool's interpreter
    $PY bench/modal_radix_drive.py spawn   --card L4 --codec sip --gates
    $PY bench/modal_radix_drive.py collect --card L4 --codec sip --wait 540
    $PY bench/modal_radix_drive.py status
    $PY bench/modal_radix_drive.py spawn27 --codec gulp --stage pack|cell
    $PY bench/modal_radix_drive.py collect --card L40S --codec gulp --model Qwen3.8-27B ...

Receipts land under cuda/<card>/ (cuda/<card>-27b/ for the stretch),
verbatim as the container returned them:

    <codec>.call.json            the spawn (call id, time, arguments)
    <codec>.cell.json            the cell report minus the big texts
    <codec>.cell.log             the container's full log
    <codec>.bench.json           the bench record, bench's own bytes
    <codec>_spec-{off,on}.json   bench/serve_drive.py's JSON per served run
    <codec>_spec-{off,on}.server.log
    <codec>.pack.meta.json       the pack's meta.json
    gates/<name>.{json,log}      gate receipts (the --gates cell) and the
                                 real-pack mc gate (every radix cell)

A cell whose function raised (Modal timeout, container death) still gets a
FAILED_<codec>.cell.json with the exception — a detached dispatch must not
smile. Verdicts come from the receipts, never from this script's exit code.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import subprocess
import sys
import time

import modal

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
from modal_radix import APP_NAME, CODECS, SHAPES  # noqa: E402

RECEIPTS = os.path.join(REPO, "cuda")
BIG = ("log", "record_text", "pack_meta_text")


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:  # noqa: BLE001
        return ""


def _token() -> str:
    token = os.environ.get("HF_TOKEN", "")
    tp = os.path.expanduser("~/.cache/huggingface/token")
    if not token and os.path.exists(tp):
        token = open(tp).read().strip()
    return token


def _dir(card: str, model: str) -> str:
    d = os.path.join(RECEIPTS, card if model == "Qwen3-8B" else f"{card}-{model.lower()}")
    os.makedirs(d, exist_ok=True)
    return d


def _write(path: str, text: str | None) -> None:
    if text is None:
        return
    with open(path, "w") as f:
        f.write(text)


def spawn(card: str, codec: str, gates: bool, model: str, fn_name: str | None = None,
          extra: dict | None = None, out_dir: str | None = None) -> str:
    fn = modal.Function.from_name(APP_NAME, fn_name or f"cell_{card}")
    kwargs = {"model": model, "codec": codec, "gates": gates, "hf_token": _token(),
              "commit": _commit(), **(extra or {})}
    fc = fn.spawn(**kwargs)
    rec = {"call_id": fc.object_id, "function": fn_name or f"cell_{card}", "card": card,
           "codec": codec, "model": model, "gates": gates, "spawned": _now(),
           "kwargs": {k: v for k, v in kwargs.items() if k != "hf_token"},
           "hf_token_passed": bool(kwargs["hf_token"])}
    d = out_dir or _dir(card, model)
    path = os.path.join(d, f"{codec}.call.json")
    with open(path, "w") as f:
        json.dump(rec, f, indent=1)
    print(f"[drive] spawned {rec['function']}({model}, {codec}, gates={gates}) -> {fc.object_id} · {path}",
          flush=True)
    return fc.object_id


def collect(card: str, codec: str, model: str, wait: int, out_dir: str | None = None) -> int:
    d = out_dir or _dir(card, model)
    cpath = os.path.join(d, f"{codec}.call.json")
    if not os.path.exists(cpath):
        print(f"[drive] no spawn record at {cpath}", flush=True)
        return 2
    rec = json.load(open(cpath))
    fc = modal.FunctionCall.from_id(rec["call_id"])
    t0 = time.time()
    try:
        report = fc.get(timeout=wait)
    except TimeoutError:
        print(f"[drive] {card}/{codec}: still running after {time.time() - t0:.0f}s of waiting "
              f"(spawned {rec['spawned']}, now {_now()})", flush=True)
        return 3
    except Exception as e:  # noqa: BLE001 — the function raised / timed out / died
        report = {"gpu": card, "codec": codec, "model": model, "error":
                  f"the function raised {type(e).__name__}: {e}", "steps": {}, "step_order": [],
                  "log": f"(no log came back: {type(e).__name__}: {e})\n"}
        stem = f"FAILED_{codec}"
        _write(os.path.join(d, f"{stem}.cell.json"), json.dumps({**report, "collected": _now(),
                                                                  "call": rec}, indent=1))
        _write(os.path.join(d, f"{stem}.cell.log"), report["log"])
        print(f"[drive] {card}/{codec}: FAILED — {report['error']} -> {d}/{stem}.cell.json", flush=True)
        return 1

    if "per_codec" in report:
        # codec="all": one container, three packs — each codec's receipts
        # under triple-N/ in the single-cell layout, the container's own
        # log/report as triple-N/all.*
        n = 1
        while os.path.exists(os.path.join(d, f"triple-{n}")):
            n += 1
        td = os.path.join(d, f"triple-{n}")  # one same-host run per directory, never overwritten
        os.makedirs(td)
        _write(os.path.join(td, "all.cell.log"), report.get("log"))
        summaries = []
        for cdc, sub in report["per_codec"].items():
            merged = {**{k: v for k, v in report.items() if k not in ("per_codec", "log")}, **sub,
                      "gates": {**report.get("gates", {}), **sub.get("gates", {})}}
            summaries.append(f"[{cdc}]\n" + summarize(merged))
            _write_receipts(td, cdc, merged, rec)
        slim = {k: v for k, v in report.items() if k not in ("per_codec", "log")}
        slim["per_codec_keys"] = list(report["per_codec"])
        slim["collected"] = _now()
        slim["call"] = rec
        _write(os.path.join(td, "all.cell.json"), json.dumps(slim, indent=1))
        print(f"[drive] {card}/all: collected -> {td}", flush=True)
        print("\n".join(summaries), flush=True)
        return 0

    # the receipts, verbatim (the summary is printed off the report BEFORE
    # the big texts are popped out of it below)
    summary = summarize(report)
    _write_receipts(d, codec, report, rec)
    print(f"[drive] {card}/{codec}: collected -> {d}", flush=True)
    print(summary, flush=True)
    return 0


def _write_receipts(d: str, codec: str, report: dict, rec: dict) -> None:
    """One codec's receipts in the layout the module docstring gives; the
    big texts are popped out of `report` as they are written."""
    _write(os.path.join(d, f"{codec}.cell.log"), report.get("log"))
    _write(os.path.join(d, f"{codec}.bench.json"), report.get("record_text"))
    _write(os.path.join(d, f"{codec}.pack.meta.json"), report.get("pack_meta_text"))
    served = report.get("served") or {}
    for spec, s in served.items():
        if not isinstance(s, dict):
            continue
        _write(os.path.join(d, f"{codec}_spec-{spec}.json"), s.pop("json", None))
        _write(os.path.join(d, f"{codec}_spec-{spec}.server.log"), s.pop("server_log", None))
    gates = report.get("gates") or {}
    if gates:
        gd = os.path.join(d, "gates")
        os.makedirs(gd, exist_ok=True)
        for name, g in gates.items():
            stem = f"{name}_{codec}" if name == "radix_mc_bitpin_pack" else name
            _write(os.path.join(gd, f"{stem}.json"), g.pop("json", None))
            _write(os.path.join(gd, f"{stem}.log"), g.pop("tail", None))
    slim = {k: v for k, v in report.items() if k not in BIG}
    slim["collected"] = _now()
    slim["call"] = rec
    slim["record_bytes"] = len(report.get("record_text") or "")
    _write(os.path.join(d, f"{codec}.cell.json"), json.dumps(slim, indent=1))


def summarize(report: dict) -> str:
    lines = [f"  nvidia-smi {report.get('nvidia_smi')!r} · torch {report.get('torch_identity')}",
             f"  wall {report.get('wall_s')} s · steps: " + ", ".join(
                 f"{s} {report['steps'][s]['seconds']}s exit {report['steps'][s]['exit']}"
                 + (" KILLED" if report['steps'][s].get('killed_at_deadline') else "")
                 for s in report.get("step_order", []))]
    if report.get("error"):
        lines.append(f"  ERROR: {report['error']}")
    for name, g in (report.get("gates") or {}).items():
        lines.append(f"  gate {name}: {g.get('verdict')!r} / {g.get('selftest_line')!r} exit {g.get('exit')}")
    lines.append(f"  pack: wall {report.get('pack_wall_s')} s · du {report.get('pack_du_bytes')} · "
                 f"{json.dumps(report.get('pack_facts'))}")
    rt = report.get("record_text")
    if rt:
        try:
            r = json.loads(rt)
            met = {m["name"]: m["value"] for m in r.get("metrics", [])}
            raw = r.get("raw") or {}
            lines.append(f"  bench: stock {met.get('stock_decode_tok_s')} · twin (raw) {raw.get('twin_decode_tok_s')}"
                         f" · compressed {met.get('compressed_decode_tok_s')} {raw.get('compressed_decode_samples')}"
                         f" · prefill {met.get('compressed_prefill_tok_s')} · ttft {met.get('compressed_ttft_s')}"
                         f" · read {met.get('read_gb_s')} GB/s · vram_compressed {raw.get('vram_compressed_bytes')}"
                         f" · verified {raw.get('verified_tensors')}/{raw.get('swapped_linears')}"
                         f" · stock.outcome {(r.get('stock') or {}).get('outcome')} raw.twin_outcome {raw.get('twin_outcome')}")
        except ValueError:
            lines.append("  bench: record is not JSON")
    else:
        lines.append(f"  bench: NO RECORD ({report.get('bench_error')})")
    for spec, s in (report.get("served") or {}).items():
        if not isinstance(s, dict) or not s.get("json"):
            lines.append(f"  served spec-{spec}: NO JSON ({(s or {}).get('error')}; drive exit "
                         f"{(s or {}).get('drive_exit')}; device {(s or {}).get('device_line')!r})")
            continue
        try:
            j = json.loads(s["json"])
            sm = j["summary"]
            lines.append(f"  served spec-{spec}: decode {sm['decode_tok_s_median']} {sm['decode_tok_s_samples']}"
                         f" · ttft {sm['ttft_s_median']} · tokens/step {sm['tokens_per_step_median']}"
                         f" · acceptance {sm['metrics_acceptance']} · sha {sm['text_sha_first_run']}"
                         f" · identical {sm['text_identical_across_runs']} · device {s.get('device_line')!r}")
        except (ValueError, KeyError) as e:
            lines.append(f"  served spec-{spec}: JSON unreadable ({e})")
    return "\n".join(lines)


def status() -> int:
    import glob

    for cpath in sorted(glob.glob(os.path.join(RECEIPTS, "*", "*.call.json"))):
        rec = json.load(open(cpath))
        d = os.path.dirname(cpath)
        codec = rec["codec"]
        have = os.path.exists(os.path.join(d, f"{codec}.cell.json"))
        failed = os.path.exists(os.path.join(d, f"FAILED_{codec}.cell.json"))
        state = "collected" if have else ("FAILED" if failed else "?")
        if state == "?":
            try:
                modal.FunctionCall.from_id(rec["call_id"]).get(timeout=0)
                state = "done, not collected"
            except TimeoutError:
                state = "running"
            except Exception as e:  # noqa: BLE001
                state = f"raised {type(e).__name__}"
        print(f"{os.path.basename(d):>22s} {codec:14s} {rec['call_id']} spawned {rec['spawned']} · {state}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="verb", required=True)
    sp = sub.add_parser("spawn")
    sp.add_argument("--card", required=True, choices=sorted(SHAPES))
    sp.add_argument("--codec", required=True, choices=(*CODECS, "all"),
                    help="one codec per container, or `all`: the three on one host, receipts under triple-N/")
    sp.add_argument("--gates", action="store_true")
    sp.add_argument("--model", default="Qwen3-8B")
    co = sub.add_parser("collect")
    co.add_argument("--card", required=True)
    co.add_argument("--codec", required=True, choices=(*CODECS, "all"))
    co.add_argument("--model", default="Qwen3-8B")
    co.add_argument("--wait", type=int, default=540)
    sub.add_parser("status")
    s27 = sub.add_parser("spawn27")
    s27.add_argument("--codec", required=True, choices=(*CODECS, "all"))
    s27.add_argument("--stage", required=True, choices=["pack", "cell"])
    s27.add_argument("--model", default="Qwen3.8-27B")
    s27.add_argument("--serve-ctx", type=int, default=8192)
    s27.add_argument("--codecs", default="", help="codec=all: the packs on the volume to run, in order (comma-separated)")
    a = ap.parse_args()

    if a.verb == "spawn":
        spawn(a.card, a.codec, a.gates, a.model)
        return 0
    if a.verb == "collect":
        return collect(a.card, a.codec, a.model, a.wait)
    if a.verb == "status":
        return status()
    if a.verb == "spawn27":
        d = _dir("L40S", a.model)
        if a.stage == "pack":
            fn = modal.Function.from_name(APP_NAME, "pack27")
            fc = fn.spawn(model=a.model, codec=a.codec, hf_token=_token(), commit=_commit())
            rec = {"call_id": fc.object_id, "function": "pack27", "card": "cpu", "codec": a.codec,
                   "model": a.model, "spawned": _now()}
            path = os.path.join(d, f"pack27_{a.codec}.call.json")
            with open(path, "w") as f:
                json.dump(rec, f, indent=1)
            print(f"[drive] spawned pack27({a.model}, {a.codec}) -> {fc.object_id} · {path}", flush=True)
        else:
            extra = {"serve_ctx": a.serve_ctx}
            if a.codecs:
                extra["codecs"] = a.codecs.split(",")
            spawn("L40S", a.codec, False, a.model, fn_name="cell27_L40S", extra=extra, out_dir=d)
        return 0
    return 2


def collect27_pack(codec: str, model: str, wait: int) -> int:
    """The pack27 stage's receipt (a different report shape: no bench, no serve)."""
    d = _dir("L40S", model)
    cpath = os.path.join(d, f"pack27_{codec}.call.json")
    rec = json.load(open(cpath))
    fc = modal.FunctionCall.from_id(rec["call_id"])
    try:
        report = fc.get(timeout=wait)
    except TimeoutError:
        print(f"[drive] pack27/{codec}: still running", flush=True)
        return 3
    except Exception as e:  # noqa: BLE001
        _write(os.path.join(d, f"FAILED_pack27_{codec}.json"),
               json.dumps({"error": f"{type(e).__name__}: {e}", "call": rec}, indent=1))
        print(f"[drive] pack27/{codec}: FAILED {type(e).__name__}: {e}", flush=True)
        return 1
    _write(os.path.join(d, f"pack27_{codec}.log"), report.pop("log", None))
    _write(os.path.join(d, f"pack27_{codec}.pack.meta.json"), report.pop("pack_meta_text", None))
    _write(os.path.join(d, f"pack27_{codec}.json"), json.dumps({**report, "collected": _now(), "call": rec},
                                                                indent=1))
    print(f"[drive] pack27/{codec}: wall {report.get('wall_s')} s · pack wall {report.get('pack_wall_s')} s "
          f"exit {report.get('pack_exit')} · du {report.get('pack_du_bytes')} · volume {report.get('volume_committed')}"
          f" · hf {report.get('hf_snapshot_du')}", flush=True)
    for s in report.get("step_order", []):
        st = report["steps"][s]
        print(f"   {s}: {st['seconds']}s exit {st['exit']} killed {st['killed_at_deadline']}", flush=True)
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "collect27pack":
        ap = argparse.ArgumentParser()
        ap.add_argument("collect27pack")
        ap.add_argument("--codec", required=True)
        ap.add_argument("--model", default="Qwen3.8-27B")
        ap.add_argument("--wait", type=int, default=540)
        a = ap.parse_args()
        raise SystemExit(collect27_pack(a.codec, a.model, a.wait))
    raise SystemExit(main())
