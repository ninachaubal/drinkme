"""NVIDIA with the schedule table: per card, in ONE container, the rotation
sweep of the launch-config grid on Qwen3-8B's layer-0 tensors + lm_head
(bench/parity_rotation.py, the gfx1151 protocol: primer, the `spike`
schedule t1/w2 as the in-session control, after a 3x bandwidth probe), the
card's own table row per shape class read off it, then the same-host
model-level pair — sip at the gfx1151 table / sip at THIS card's row — as
`drinkme bench --pack-dir` (warm-up rep) + served spec-off / spec-on
(bench/serve_drive.py, the same 128-token protocol as the receipts of
record). And the 27B on the L40S off the volume
`drinkme-radix-27b`: the rotation on its own shapes (DeltaNet projections,
the 248K-vocab lm_head) for sip / gulp at every config, then the
model-level arms sip / gulp with VRAM headroom per pack.

Built on bench/modal_radix.py (the image, the shapes, the mount, the pack27
body) — its app `drinkme-radix-cuda` is left deployed and untouched; this
is a second app, `drinkme-radix-nvidia2`, so no container of the old
deployment can be warm when this one deploys.

The cell (a card's function, `nv2_<card>`):
    0. identity: nvidia-smi, torch/triton, the device name (the A10G pool
       hands out A10 and A10G silicon; get_device_name witnesses which);
       `uv sync --frozen` against the env prebuilt into the image
    1. the wall: bench/roofline_wall.py (drinkme.probe.measure_bandwidth x3)
    2. the sip pack on the container CPU from HF — gulp is not
       in Q1's pair, so it is not built
    3. gates: bench/radix_schedule_bitpin.py (+ --selftest; --no-real: the
       rotation repeats the real-tensor checks in-session on the timed
       buffers), bench/radix_mc_bitpin.py --pack-dir on the sip pack
    4. the rotation: --layers 0 --lm-head, tiles {1, 2, row} x warps
       {1, 2, 4, 8}, 20 passes x 3 repeats, --primer, the mc arm at
       M in {2, 4, 8} for t1 at every warp count; every arm's decoded bits
       against the checkpoint and every config's GEMV against the float64
       oracle before timing (PARITY_ROTATION PASS is the receipt)
    5. the card's table: per class the gemv pick and the mc pick ->
       card_table.json (radix_schedule's @file override)
    6. the pair on this host: sip with DRINKME_RADIX_SCHEDULE=table
       (gfx1151's row); sip with
       DRINKME_RADIX_SCHEDULE=@card_table.json — each `drinkme bench
       --pack-dir` (DRINKME_BENCH_WARMUP=1) then served spec-off / spec-on,
       nvidia-smi memory sampled through every GPU step (peak used ->
       headroom)

Every step is killed DEADLINE_MARGIN_S before the container timeout; the
function returns receipts (records, JSONs, logs, the rotation JSON, the
table) and bench/modal_nvidia2_drive.py writes them under
nvidia2/<card>/run-N/ verbatim. Verdicts come from OUTPUT (the gates'
verdict lines, PARITY_ROTATION PASS, the record's fields, the driver's
summary) — never from exit codes.

Timeouts: the Q1 cells 4800 s (the L4 builds its pack in ~8 min and runs
two bench + four served runs after the sweep), the 27B cell 5400 s (five
model-level arms off the volume at ~14 min each), pack27 3600 s. Spend is
by wall, not by timeout.

The receipts under nvidia2/ were taken
with an earlier protocol that also carried the prior codec's kernel as an
in-session control arm and a third model-level arm over its pack; that
codec is not on this tree, and the reader of those receipts lives beside
them in the companion repo.
"""

import glob
import json
import os
import re
import shutil
import signal
import subprocess
import threading
import time

import modal

HERE = os.path.dirname(os.path.abspath(__file__))
import sys  # noqa: E402
# locally this file sits beside modal_radix.py; in the container Modal
# mounts the entry module at /root/ and the checkout at /root/drinkme
sys.path.insert(0, HERE)
sys.path.insert(0, "/root/drinkme/bench")
from modal_radix import (  # noqa: E402
    DEADLINE_MARGIN_S, HOME_DIR, LOG_CAP, OUT_DIR, PROFILE_OF, REMOTE_MEASUREMENTS, REMOTE_REPO, SERVE_PORT, SHAPES,
    VENV, VOL_NAME, VOL_PATH, collapse_cr, image, pack_argv, run_pack27, _read,
)

APP_NAME = "drinkme-radix-nvidia2"
app = modal.App(APP_NAME)
vol27 = modal.Volume.from_name(VOL_NAME, create_if_missing=True)

Q1_TIMEOUT = 4800
Q2_TIMEOUT = 5400
GRID_TILES = ("1", "2", "0")          # blocks per program: 1, 2, the whole row
GRID_WARPS = ("1", "2", "4", "8")
MC_MS = ("2", "4", "8")
MC_CONFIGS = ("t1/w2", "t1/w1", "t1/w4", "t1/w8")
CLASSES = ("head", "narrow", "long", "wide", "square")


class GpuMem:
    """nvidia-smi memory.used sampled in a thread through a step: the peak
    is the pack's VRAM footprint under that step (bench / a server), the
    total the card's; headroom = total - peak."""

    def __init__(self, interval=2.0):
        self.interval = interval
        self.used = []
        self.total = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.is_set():
            try:
                out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
                                     capture_output=True, text=True, timeout=5).stdout.strip().split(",")
                self.used.append(int(out[0]))
                self.total = int(out[1])
            except Exception:  # noqa: BLE001 — a missed sample is not a failure
                pass
            self._stop.wait(self.interval)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()

    def summary(self):
        if not self.used:
            return {"samples": 0}
        peak = max(self.used)
        return {"samples": len(self.used), "peak_used_mib": peak, "total_mib": self.total,
                "headroom_mib": (self.total - peak) if self.total else None, "first_used_mib": self.used[0]}


class Cell:
    """The container side's helpers (modal_radix.run_cell's closures as a
    class): a capped log, a step runner with the deadline reaper, receipts."""

    def __init__(self, card: str, gate: str, timeout: int, hf_token: str, commit: str, extra_env: dict | None = None):
        self.card = card
        self.shape = SHAPES[card]
        self.env = dict(os.environ, PATH="/root/.local/bin:" + os.environ["PATH"],
                        UV_PROJECT_ENVIRONMENT=VENV, DRINKME_HOME=HOME_DIR,
                        HF_HUB_DISABLE_PROGRESS_BARS="1", TOKENIZERS_PARALLELISM="false",
                        PYTHONUNBUFFERED="1", DRINKME_NO_AUTO_DEPS="1",
                        OMP_NUM_THREADS=str(self.shape["cpu"]), MKL_NUM_THREADS=str(self.shape["cpu"]),
                        **(extra_env or {}))
        if hf_token:
            self.env["HF_TOKEN"] = hf_token
        self.report = {"gate": gate, "gpu": card, "commit": commit, "shape": {**self.shape, "timeout": timeout},
                       "steps": {}, "step_order": [],
                       "started": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())}
        self.T0 = time.time()
        self.deadline = self.T0 + timeout - DEADLINE_MARGIN_S
        self.log_parts: list[str] = []
        self.log_len = 0
        os.makedirs(OUT_DIR, exist_ok=True)
        os.makedirs(HOME_DIR, exist_ok=True)

    def log(self, text: str) -> None:
        print(text, end="" if text.endswith("\n") else "\n", flush=True)
        self.log_parts.append(text if text.endswith("\n") else text + "\n")
        self.log_len += len(self.log_parts[-1])
        if self.log_len > 2 * LOG_CAP:
            joined = "".join(self.log_parts)[-LOG_CAP:]
            self.log_parts[:] = [joined]
            self.log_len = len(joined)

    def elapsed(self) -> float:
        return time.time() - self.T0

    def remaining(self) -> float:
        return self.deadline - time.time()

    def run(self, label, cmd, cwd=REMOTE_REPO, extra_env=None, tail_cap=6000, sample_gpu=False):
        """Stream a step's output as it runs, keep the tail, kill the whole
        process group at the deadline; the env the step was given beyond
        the cell's (the schedule override, the pack paths) is in the
        receipt."""
        extra_env = dict(extra_env or {})
        shown = {k: v for k, v in extra_env.items() if k != "HF_TOKEN"}
        self.log(f"\n=== [{self.elapsed():7.1f}s] {label}: {' '.join(cmd)}" + (f"  env {json.dumps(shown)}" if shown else "") + "\n")
        t0 = time.time()
        p = subprocess.Popen(cmd, cwd=cwd, env={**self.env, **extra_env}, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True, errors="replace", start_new_session=True)
        killed = {"at": None}

        def reaper():
            while p.poll() is None:
                if time.time() >= self.deadline:
                    killed["at"] = round(self.elapsed(), 1)
                    try:
                        os.killpg(p.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    return
                time.sleep(2)

        threading.Thread(target=reaper, daemon=True).start()
        mem = GpuMem() if sample_gpu else None
        if mem:
            mem.__enter__()
        tail = ""
        for line in p.stdout:
            line = collapse_cr(line)
            self.log(line)
            tail = (tail + line)[-2 * tail_cap:]
        p.wait()
        if mem:
            mem.__exit__(None, None, None)
        dt = time.time() - t0
        verdict = (f"KILLED at the deadline ({killed['at']}s into the container)"
                   if killed["at"] is not None else f"exit {p.returncode}")
        self.log(f"=== [{self.elapsed():7.1f}s] {label}: {verdict} after {dt:.1f}s\n")
        self.report["steps"][label] = {"exit": p.returncode, "seconds": round(dt, 1), "killed_at_deadline": killed["at"],
                                       "env": shown, "tail": tail[-tail_cap:],
                                       **({"gpu_mem": mem.summary()} if mem else {})}
        self.report["step_order"].append(label)
        return p.returncode

    def done(self, error=None) -> dict:
        if error:
            self.report["error"] = error
            self.log(f"\nFAILED: {error}\n")
        self.report["wall_s"] = round(self.elapsed(), 1)
        self.report["log"] = "".join(self.log_parts)[-LOG_CAP:]
        return self.report

    @staticmethod
    def verdict_line(text: str, prefix: str) -> str | None:
        for line in (text or "").splitlines():
            if line.startswith(prefix):
                return line.strip()
        return None

    # ---- the steps every cell shares ----------------------------------------

    def identity(self) -> str | None:
        """nvidia-smi + torch identity; the device name must witness the
        card. Returns an error string or None."""
        r = self.report
        try:
            smi = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                                  "--format=csv,noheader"], capture_output=True, text=True, timeout=30).stdout.strip()
        except Exception as e:  # noqa: BLE001
            smi = f"nvidia-smi unavailable: {e}"
        r["nvidia_smi"] = smi
        try:
            r["cpu_count_visible"] = os.cpu_count()
            with open("/proc/cpuinfo") as f:
                models = [l.split(":", 1)[1].strip() for l in f if l.startswith("model name")]
            r["cpu_model"] = models[0] if models else None
        except Exception:  # noqa: BLE001
            pass
        try:
            r["disk_free_gb"] = round(shutil.disk_usage("/root").free / 1e9, 1)
        except Exception:  # noqa: BLE001
            pass
        self.log(f"nvidia-smi: {smi}\ncard (the calling function's literal): {self.card}; commit {r['commit'] or '?'}; "
                 f"cpu visible {r.get('cpu_count_visible')} ({r.get('cpu_model')}); disk free {r.get('disk_free_gb')} GB\n")
        if self.run("uv sync --frozen --group cuda (prebuilt env)",
                    ["uv", "sync", "--frozen", "--no-default-groups", "--group", "cuda"]):
            return "uv sync failed"
        if self.run("bootstrap.ensure_accelerator", ["uv", "run", "--no-sync", "python", "-c",
                                                     "from drinkme.bootstrap import ensure_accelerator; ensure_accelerator()"]):
            return "bootstrap.ensure_accelerator failed"
        code = self.run("torch identity", ["uv", "run", "--no-sync", "python", "-c", (
            "import json, torch, triton, numpy, transformers, drinkme, os; "
            "from drinkme.codec import radix_native; "
            "p = torch.cuda.get_device_properties(0) if torch.cuda.is_available() else None; "
            "print(json.dumps({'torch': torch.__version__, 'triton': triton.__version__, "
            "'numpy': numpy.__version__, 'transformers': transformers.__version__, "
            "'cuda': torch.cuda.is_available(), "
            "'device': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, "
            "'capability': list(torch.cuda.get_device_capability()) if torch.cuda.is_available() else None, "
            "'sms': p.multi_processor_count if p else None, 'total_mem': p.total_memory if p else None, "
            "'cuda_runtime': torch.version.cuda, "
            "'drinkme_file': os.path.abspath(drinkme.__file__), "
            "'radix_native_available': radix_native.available()}))")])
        ident = {}
        for line in r["steps"]["torch identity"]["tail"].splitlines():
            if line.startswith("{"):
                try:
                    ident = json.loads(line)
                except ValueError:
                    pass
        r["torch_identity"] = ident
        r["device"] = ident.get("device")
        if code or not ident.get("cuda") or not re.search(self.shape["witness"], ident.get("device") or ""):
            return f"torch identity is not a CUDA {self.card}: {ident}"
        if not ident.get("drinkme_file", "").startswith(REMOTE_REPO):
            return f"drinkme imports from {ident.get('drinkme_file')!r}, not the mounted checkout"
        if not ident.get("radix_native_available"):
            return "the radix native encoder does not compile here"
        return None

    def wall(self) -> str | None:
        path = f"{OUT_DIR}/wall.json"
        rc = self.run("wall (roofline_wall x3)", ["uv", "run", "--no-sync", "python", "bench/roofline_wall.py", path])
        self.report["wall_json"] = _read(path)
        self.report["wall_verdict"] = self.verdict_line(self.report["steps"]["wall (roofline_wall x3)"]["tail"], "ROOFLINE_WALL")
        if rc or not (self.report["wall_verdict"] or "").startswith("ROOFLINE_WALL PASS"):
            return "the bandwidth wall did not PASS"
        return None

    def pack(self, model: str, codec: str, pack_dir: str) -> str | None:
        t = time.time()
        try:
            argv = pack_argv(model, codec, pack_dir)  # a label that is not a profile refuses here
        except ValueError as e:
            return str(e)
        rc = self.run(f"drinkme pack {PROFILE_OF[codec]}", argv, tail_cap=12000)
        self.report.setdefault("packs", {})[codec] = {"pack_dir": pack_dir, "pack_wall_s": round(time.time() - t, 1)}
        if rc or not os.path.exists(os.path.join(pack_dir, "meta.json")):
            return f"drinkme pack {PROFILE_OF[codec]} failed (exit {rc})"
        return self.pack_facts(codec, pack_dir)

    def pack_facts(self, codec: str, pack_dir: str) -> str | None:
        sub = self.report.setdefault("packs", {}).setdefault(codec, {"pack_dir": pack_dir})
        if not os.path.exists(os.path.join(pack_dir, "meta.json")):
            return f"no pack at {pack_dir}"
        try:
            sub["pack_du_bytes"] = int(subprocess.run(["du", "-sb", pack_dir], capture_output=True, text=True).stdout.split()[0])
        except Exception as e:  # noqa: BLE001
            sub["pack_du_bytes"] = f"du failed: {e}"
        sub["pack_meta_text"] = _read(os.path.join(pack_dir, "meta.json"))
        try:
            meta = json.loads(sub["pack_meta_text"] or "{}")
            sub["pack_facts"] = {k: meta.get(k) for k in (
                "formatVersion", "profile", "profileWidths", "tensorCount", "meanBpw", "weightedBpw",
                "packSeconds", "packedBytes", "residentBytes", "rawResidentBytes", "rawTensorCount", "radixTensorCount",
                "radixBytes", "rawFallbackTensorCount", "rawFallbackBytes", "radixEncoder", "radixEncodeSeconds",
                "manifestSha256", "hfRepo", "revision", "mtpResidentBytes")}
        except Exception as e:  # noqa: BLE001
            sub["pack_facts"] = {"error": f"{type(e).__name__}: {e}"}
        self.log(f"--- pack facts ({codec}): du {sub['pack_du_bytes']} · {json.dumps(sub.get('pack_facts'))}\n")
        return None

    def gate(self, name: str, script: str, prefix: str, flags=(), selftest=False, tail_cap=20000) -> None:
        gate_dir = f"{OUT_DIR}/gates"
        os.makedirs(gate_dir, exist_ok=True)
        jpath = f"{gate_dir}/{name}.json"
        cmd = ["uv", "run", "--no-sync", "python", f"bench/{script}.py", "--json", jpath, *flags]
        if selftest:
            cmd.append("--selftest")
        rc = self.run(f"gate {name}", cmd, tail_cap=tail_cap)
        tail = self.report["steps"][f"gate {name}"]["tail"]
        self.report.setdefault("gates", {})[name] = {
            "exit": rc, "verdict": self.verdict_line(tail, prefix), "selftest_line": self.verdict_line(tail, "SELFTEST"),
            "json": _read(jpath), "tail": tail}
        self.log(f"--- gate {name}: verdict {self.report['gates'][name]['verdict']!r}; "
                 f"selftest {self.report['gates'][name]['selftest_line']!r}; exit {rc}\n")

    def rotation(self, label: str, packs: dict, args: list, hf_home: str | None = None) -> str | None:
        """bench/parity_rotation.py on the packs (name -> dir); the JSON
        comes back in the report under `label`."""
        path = f"{OUT_DIR}/{label}.json"
        env = {"DRINKME_PARITY_PACK_SIP": packs["sip"]}
        for codec, var in (("balanced", "DRINKME_PARITY_PACK_BALANCED"), ("gulp", "DRINKME_PARITY_PACK_GULP")):
            if codec in packs:
                env[var] = packs[codec]
        if hf_home:
            env["HF_HOME"] = hf_home
        rc = self.run(label, ["uv", "run", "--no-sync", "python", "bench/parity_rotation.py", "--json", path,
                              "--wall", f"{OUT_DIR}/wall.json", *args], extra_env=env, tail_cap=60000, sample_gpu=True)
        text = _read(path)
        self.report.setdefault("rotations", {})[label] = {
            "json": text, "verdict": self.verdict_line(self.report["steps"][label]["tail"], "PARITY_ROTATION"), "exit": rc}
        if rc or not text or self.report["rotations"][label]["verdict"] != "PARITY_ROTATION PASS":
            return f"{label} did not PASS (exit {rc})"
        return None

    def arm(self, label: str, model: str, pack_dir: str, env: dict, serve_ctx: int = 8192, bench: bool = True) -> dict:
        """One model-level arm on this host: `drinkme bench --pack-dir`
        (warm-up rep) then served spec-off / spec-on, nvidia-smi memory
        sampled through each; `env` carries the schedule override."""
        sub = {"label": label, "pack_dir": pack_dir, "env": env}
        self.report.setdefault("arms", {})[label] = sub
        self.report.setdefault("arm_order", []).append(label)
        if bench:
            if self.remaining() < 420:
                sub["bench_error"] = "skipped: under 420 s to the deadline"
            else:
                os.makedirs(REMOTE_MEASUREMENTS, exist_ok=True)
                before = set(glob.glob(f"{REMOTE_MEASUREMENTS}/*.json"))
                step = f"bench {label}"
                code = self.run(step, ["uv", "run", "--no-sync", "drinkme", "bench", "--model", model, "--pack-dir", pack_dir],
                                extra_env={**env, "DRINKME_BENCH_WARMUP": "1"}, tail_cap=12000, sample_gpu=True)
                sub["bench_exit"] = code
                sub["bench_gpu_mem"] = self.report["steps"][step].get("gpu_mem")
                written = sorted(set(glob.glob(f"{REMOTE_MEASUREMENTS}/*.json")) - before, key=os.path.getmtime)
                if written:
                    text = _read(written[-1])
                    try:
                        json.loads(text)
                        sub["record_basename"] = os.path.basename(written[-1])
                        sub["record_text"] = text
                        self.log(f"\n=== record written (in the container): {sub['record_basename']} ({len(text)} bytes)\n")
                    except ValueError as e:
                        sub["bench_error"] = f"the record is not JSON: {e}"
                else:
                    sub["bench_error"] = f"drinkme bench wrote no record (exit {code}) — read the bench step's tail"
                    self.log(f"\n{sub['bench_error']}\n")
        sub["served"] = {}
        drive = "bench/serve_drive.py"
        for spec in ("off", "on"):
            if self.remaining() < 300:
                sub["served"][spec] = {"error": "skipped: under 300 s to the deadline"}
                self.log(f"\n=== served {label} spec-{spec}: SKIPPED, under 300 s to the deadline\n")
                continue
            slabel = f"{label}_spec-{spec}"
            slog = f"{OUT_DIR}/{slabel}.server.log"
            sout = f"{OUT_DIR}/{slabel}.json"
            senv = {**self.env, **env, "DRINKME_PREFIX_SLOTS": "0", "DRINKME_SLOT_DIR": "off"}
            senv.pop("DRINKME_SPEC", None)
            if spec == "off":
                senv["DRINKME_SPEC"] = "off"
            cmd = ["uv", "run", "--no-sync", "drinkme", "serve", "--model", model, "--pack-dir", pack_dir,
                   "--port", str(SERVE_PORT), "--ctx", str(serve_ctx), "--yes"]
            self.log(f"\n=== [{self.elapsed():7.1f}s] serve {slabel}: {' '.join(cmd)} "
                     f"(DRINKME_SPEC={senv.get('DRINKME_SPEC', '<unset>')}, DRINKME_RADIX_SCHEDULE="
                     f"{senv.get('DRINKME_RADIX_SCHEDULE', '<unset>')}) -> {slog}\n")
            t_boot = time.time()
            with open(slog, "w") as lf:
                server = subprocess.Popen(cmd, cwd=REMOTE_REPO, env=senv, stdout=lf, stderr=subprocess.STDOUT,
                                          start_new_session=True)
            wait = max(60, int(min(900, self.remaining() - 120)))
            step = f"drive {slabel}"
            rc = self.run(step, ["uv", "run", "--no-sync", "python", drive, "--port", str(SERVE_PORT), "--label", slabel,
                                 "--server-log", slog, "--out", sout, "--runs", "5", "--max-tokens", "128",
                                 "--wait", str(wait)],
                          extra_env={k: senv[k] for k in ("DRINKME_SPEC", "DRINKME_PREFIX_SLOTS") if k in senv},
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
            dev_line = next((l for l in server_log.splitlines() if "[drinkme] device:" in l), None)
            spec_lines = [l for l in server_log.splitlines() if "[drinkme.spec]" in l]
            sub["served"][spec] = {
                "label": slabel, "drive_exit": rc, "boot_and_drive_s": round(time.time() - t_boot, 1),
                "json": _read(sout), "server_log": server_log[-400_000:], "device_line": dev_line,
                "spec_lines": spec_lines[-8:], "server_exit": server.returncode,
                "gpu_mem": self.report["steps"][step].get("gpu_mem")}
            self.log(f"--- served {slabel}: drive exit {rc}; device line {dev_line!r}; {len(spec_lines)} spec lines; "
                     f"json {'present' if sub['served'][spec]['json'] else 'MISSING'}; "
                     f"gpu mem {json.dumps(sub['served'][spec]['gpu_mem'])}\n")
        return sub


def card_table_from_rotation(rot: dict) -> dict:
    """The card's table: per shape class present in the rotation, the gemv
    pick (tiles, warps) and the mc pick; classes the sweep did not see keep
    the gfx1151 row (radix_schedule.TABLES["gfx1151"]) by omission."""
    table = {}
    for cls, pk in rot.get("classes", {}).items():
        row = {"gemv_tiles": int(pk["tiles"]), "gemv_warps": int(pk["warps"])}
        mc = (rot.get("mc_classes") or {}).get(cls)
        if mc:
            row.update(mc_tiles=int(mc["mc_tiles"]), mc_warps=int(mc["mc_warps"]))
        table[cls] = row
    return table


def gulp_table_from_rotation(rot: dict) -> dict:
    """gulp's own best row per class off a rotation that timed
    gulp at the grid (--gulp-configs grid); mc as sip's pick."""
    table = {}
    for cls, pk in rot.get("classes", {}).items():
        cp = (pk.get("picks") or {}).get("gulp")  # parity_rotation's per-arm picks
        if cp is None and "compact_tiles" in pk:      # the earlier receipts' spelling
            cp = {"tiles": pk["compact_tiles"], "warps": pk["compact_warps"]}
        if cp is None:
            continue
        row = {"gemv_tiles": int(cp["tiles"]), "gemv_warps": int(cp["warps"])}
        mc = (rot.get("mc_classes") or {}).get(cls)
        if mc:
            row.update(mc_tiles=int(mc["mc_tiles"]), mc_warps=int(mc["mc_warps"]))
        table[cls] = row
    return table


GFX1151 = {cls: {"gemv_tiles": 0, "gemv_warps": 1, "mc_tiles": 1, "mc_warps": 1} for cls in CLASSES}
# The gfx1151 row by name: radix_schedule's table is per box class, so
# "table" on CUDA means the CUDA row, and this arm names the gfx1151 row.
GFX1151_SPEC = "table=gfx1151"


def same_as_gfx1151(table: dict) -> bool:
    return all({**GFX1151[cls], **row} == GFX1151[cls] for cls, row in table.items())


# ------------------------------------------------------------------ Q1 ----

def run_q1(card: str, model: str = "Qwen3-8B", hf_token: str = "", commit: str = "", passes: int = 20,
           repeats: int = 3, serve_ctx: int = 8192) -> dict:
    c = Cell(card, "nvidia2 Q1: wall / pack / schedule gate / rotation sweep / card table / same-host pair "
                   "(sip@gfx1151, sip@card)", Q1_TIMEOUT, hf_token, commit)
    c.report.update(model=model, question="Q1")
    err = c.identity()
    if err:
        return c.done(err)
    err = c.wall()
    if err:
        return c.done(err)
    packs = {}
    for codec in ("sip",):
        pack_dir = f"{HOME_DIR}/packs/Qwen--{model}-{codec}"
        err = c.pack(model, codec, pack_dir)
        if err:
            return c.done(err)
        packs[codec] = pack_dir
    # gates: the schedule gate's bits + math sections (the rotation checks
    # the real tensors on the timed buffers), the mc gate on the sip pack
    for selftest in (False, True):
        c.gate("radix_schedule_bitpin" + ("_selftest" if selftest else ""), "radix_schedule_bitpin", "RADIX_SCHEDULE_BITPIN",
               flags=("--no-real",), selftest=selftest)
    c.gate("radix_mc_bitpin_pack_sip", "radix_mc_bitpin", "RADIX_MC_BITPIN", flags=("--pack-dir", packs["sip"]))
    # the rotation sweep
    err = c.rotation("rotation", packs, ["--layers", "0", "--lm-head", "--passes", str(passes), "--repeats", str(repeats),
                                        "--tiles", *GRID_TILES, "--warps", *GRID_WARPS, "--mc-Ms", *MC_MS,
                                        "--mc-configs", *MC_CONFIGS, "--primer"])
    if err:
        return c.done(err)
    rot = json.loads(c.report["rotations"]["rotation"]["json"])
    table = card_table_from_rotation(rot)
    c.report["card_table"] = table
    c.report["card_table_same_as_gfx1151"] = same_as_gfx1151(table)
    tpath = f"{OUT_DIR}/card_table.json"
    with open(tpath, "w") as f:
        json.dump(table, f, indent=1)
    c.log(f"\n--- the card's table: {json.dumps(table)} (same as gfx1151's: {c.report['card_table_same_as_gfx1151']})\n")
    # the same-host pair
    c.arm("sip-gfx1151", model, packs["sip"], {"DRINKME_RADIX_SCHEDULE": GFX1151_SPEC}, serve_ctx)
    c.arm("sip-card", model, packs["sip"], {"DRINKME_RADIX_SCHEDULE": "@" + tpath}, serve_ctx)
    return c.done()


def _fn(card, timeout):
    s = SHAPES[card]
    return app.function(image=image, gpu=s["gpu"], timeout=timeout, memory=s["memory"], cpu=s["cpu"])


@_fn("L4", Q1_TIMEOUT)
def nv2_L4(hf_token: str = "", commit: str = "", **kw) -> dict:
    return run_q1("L4", hf_token=hf_token, commit=commit, **kw)


@_fn("A10G", Q1_TIMEOUT)
def nv2_A10G(hf_token: str = "", commit: str = "", **kw) -> dict:
    return run_q1("A10G", hf_token=hf_token, commit=commit, **kw)


@_fn("L40S", Q1_TIMEOUT)
def nv2_L40S(hf_token: str = "", commit: str = "", **kw) -> dict:
    return run_q1("L40S", hf_token=hf_token, commit=commit, **kw)


@_fn("H100", Q1_TIMEOUT)
def nv2_H100(hf_token: str = "", commit: str = "", **kw) -> dict:
    return run_q1("H100", hf_token=hf_token, commit=commit, **kw)


# ------------------------------------------------------------------ Q2 ----

@app.function(image=image, timeout=3600, memory=128 * 1024, cpu=32, volumes={VOL_PATH: vol27})
def pack27(model: str, codec: str, hf_token: str = "", commit: str = "") -> dict:
    """modal_radix.pack27 under this app: a 27B pack onto the volume (the
    snapshot is already there)."""
    return run_pack27(model, codec, hf_token, commit, vol27)


def run_q2(model: str, hf_token: str, commit: str, card_table: dict | None, passes: int, repeats: int,
           serve_ctx: int, arms: list | None) -> dict:
    """The 27B on the L40S off the volume: the rotation on its own shapes
    (a DeltaNet layer, a full-attention layer, lm_head; sip at the grid /
    gulp at the grid; mc), then the model-level arms — sip at the L40S
    row from Q1 (`card_table`; the gfx1151 table when none differs),
    gulp at its own best row off this rotation, and
    with wall left: sip at the gfx1151 table if it differs from the L40S
    row, gulp at the shipped table (what a gulp pack is served at
    today)."""
    hf_home = f"{VOL_PATH}/hf"
    c = Cell("L40S", "nvidia2 Q2: the 27B on the L40S off the volume — rotation on its shapes + model-level "
                     "sip / gulp with VRAM headroom", Q2_TIMEOUT, hf_token, commit,
             extra_env={"HF_HOME": hf_home})
    c.report.update(model=model, question="Q2", card_table_from_q1=card_table)
    err = c.identity()
    if err:
        return c.done(err)
    err = c.wall()
    if err:
        return c.done(err)
    packs = {}
    for codec in ("sip", "gulp"):
        pack_dir = f"{VOL_PATH}/drinkme-home/packs/Qwen--{model}-{codec}"
        err = c.pack_facts(codec, pack_dir)
        if err:
            return c.done(err)
        packs[codec] = pack_dir
    # the tensors: the sip pack's meta names layer 0 (DeltaNet) and the
    # first full-attention layer; lm_head
    meta = json.loads(c.report["packs"]["sip"]["pack_meta_text"])
    names = [n for n in meta["tensors"] if n.startswith("model.layers.0.")]
    attn_layers = sorted({int(n.split(".")[2]) for n in meta["tensors"] if ".self_attn." in n})
    if attn_layers:
        names += [n for n in meta["tensors"] if n.startswith(f"model.layers.{attn_layers[0]}.")]
    names.append("lm_head")
    c.report["rotation_tensors"] = names
    err = c.rotation("rotation27", packs, ["--tensors", *names, "--passes", str(passes), "--repeats", str(repeats),
                                          "--tiles", *GRID_TILES, "--warps", *GRID_WARPS, "--mc-Ms", *MC_MS,
                                          "--mc-configs", *MC_CONFIGS, "--gulp", "--gulp-configs", "grid",
                                          "--primer"], hf_home=hf_home)
    if err:
        return c.done(err)
    rot = json.loads(c.report["rotations"]["rotation27"]["json"])
    own = card_table_from_rotation(rot)        # the 27B's own sip picks on this host
    gulp_own = gulp_table_from_rotation(rot)
    c.report["sip_27b_table"] = own
    c.report["gulp_27b_table"] = gulp_own
    tables = {"l40s": card_table or {}, "sip27": own, "gulp27": gulp_own}
    for name, t in tables.items():
        with open(f"{OUT_DIR}/{name}_table.json", "w") as f:
            json.dump(t, f, indent=1)
    l40s_is_gfx = same_as_gfx1151(card_table or {})
    c.log(f"\n--- L40S row from Q1: {json.dumps(card_table)} (same as gfx1151: {l40s_is_gfx}); "
          f"the 27B's own sip picks: {json.dumps(own)}; gulp's own: {json.dumps(gulp_own)}\n")
    plan = {
        "sip-l40s": (packs["sip"], {"DRINKME_RADIX_SCHEDULE": GFX1151_SPEC if l40s_is_gfx else f"@{OUT_DIR}/l40s_table.json"}),
        "gulp-own": (packs["gulp"], {"DRINKME_RADIX_SCHEDULE": f"@{OUT_DIR}/gulp27_table.json"}),
        "sip-gfx1151": (packs["sip"], {"DRINKME_RADIX_SCHEDULE": GFX1151_SPEC}),
        "gulp-table": (packs["gulp"], {"DRINKME_RADIX_SCHEDULE": GFX1151_SPEC}),
        "sip-own": (packs["sip"], {"DRINKME_RADIX_SCHEDULE": f"@{OUT_DIR}/sip27_table.json"}),
    }
    order = list(arms) if arms else ["sip-l40s", "gulp-own"]
    if not arms:
        if not l40s_is_gfx:
            order.append("sip-gfx1151")
        if same_as_gfx1151(gulp_own):
            pass  # gulp-own IS the table's row: nothing to add
        else:
            order.append("gulp-table")
    c.report["arm_plan"] = order
    for label in order:
        if c.remaining() < 600:
            c.report.setdefault("arms_skipped", []).append(label)
            c.log(f"\n--- arm {label}: SKIPPED, under 600 s to the deadline\n")
            continue
        pack_dir, env = plan[label]
        c.arm(label, model, pack_dir, env, serve_ctx)
    return c.done()


@app.function(image=image, gpu="L40S", timeout=Q2_TIMEOUT, memory=64 * 1024, cpu=16, volumes={VOL_PATH: vol27})
def nv2_27B_L40S(model: str = "Qwen3.8-27B", hf_token: str = "", commit: str = "", card_table: dict | None = None,
                 passes: int = 20, repeats: int = 3, serve_ctx: int = 8192, arms: list | None = None) -> dict:
    return run_q2(model, hf_token, commit, card_table, passes, repeats, serve_ctx, arms)
