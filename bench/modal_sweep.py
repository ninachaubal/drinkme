"""The publishing sweep's Modal leg: `drinkme bench` on a rented NVIDIA card,
the record kept exactly as bench wrote it.

One cell of the grid per invocation — one card, one menu model, one
container, one measurement record. The container does what a stranger does
(bench/modal_fresh_box.py's flow, which this file is built from):

  1. `uv sync`                       the base environment, default groups
                                     (empty by design: no torch yet)
  2. bootstrap.ensure_accelerator    the box picks its own group — the same
                                     call `drinkme bench` makes first thing;
                                     on this box it drives
                                     `uv sync --no-default-groups --group cuda`
                                     (accelerate rides in that group)
  3. torch identity                  cuda must be available and the device
                                     must be the card this run is about
  4. `uv run --no-sync drinkme bench --model <M>`
                                     NO `-o`: bench names the file itself
                                     (bench.measurement_path) under the
                                     container's ./measurements/, so the
                                     record reaches the local side with
                                     bench's own name and bench's own bytes

That is the `sip` profile (the default): bench packs in memory at the
default compression profile. `profile="gulp"` makes a gulp point the way
the Strix sweep does: between
steps 3 and 4 the container cuts the pack and checks it in place, and the
bench times its compressed arm off that pack:

  3a. `drinkme pack --model <M> --gulp -o <container disk>`
                                     the checkpoint is pulled from HF into
                                     the container's cache; pack and pull
                                     both stay on container disk
  3b. `drinkme verify --pack-dir <that pack>`
                                     every file against its recorded
                                     sha256; the cell fails without its
                                     "verified, manifest and all" line
  4.  `drinkme bench --model <M> --pack-dir <that pack>`
                                     bench names a gulp record `…_gulp.json`
                                     itself (bench.measurement_path)

Either way the record's compression.profile must be the profile asked for,
or the cell fails and no record is kept (the bytes ride in the meta).

and returns {record_text, record_basename, log, steps, ...}. The local side
writes the record bytes VERBATIM into the main checkout's measurements/
(bench's `_2`, `_3` rule if the name is taken — an existing record is never
overwritten) and the full container log beside a small meta JSON under
--log-dir. Nothing is reshaped between bench and publish. A run with no
record exits 1 (a detached dispatch must not smile).

THE BARE `uv sync` BELOW RUNS INSIDE A DISPOSABLE MODAL CONTAINER AND MUST
NEVER BE COPIED TO A DEVELOPMENT BOX (AGENTS.md: on a real machine it
replaces the pinned accelerator torch and everything afterwards silently
runs on CPU).

The card comes from the LOCAL side, twice: `AB_GPU` is read at module level
because `@app.function` takes gpu/memory/cpu/timeout at import time, AND
the label is passed into the function as an argument — a module global
re-evaluated inside the container reads an unset env and silently says
"L4" (bench/modal_serve_ab.py's 2026-08-13 A100/H100 receipts carry that
wrong field). Shapes: L4 -> 32 GiB / 8 cpu / 2700 s; A100-80GB and H100 ->
96 GiB / 16 cpu / 3600 s (the 27B is a 54 GB HF pull + a CPU pack + three
arms per invocation; no volumes, nothing persists on Modal). Never above
3600.

Usage (from the checkout; the worktree you run it from is what gets
uploaded, so harness edits ride along):
    AB_GPU=L4        uv tool run modal run bench/modal_sweep.py --model Qwen3-8B
    AB_GPU=A100-80GB uv tool run modal run bench/modal_sweep.py --model Qwen3.8-27B \\
                         --out-dir /path/to/drinkme/measurements --log-dir /path/to/logs
    AB_GPU=L4        uv tool run modal run bench/modal_sweep.py --model Qwen3-8B --profile gulp

A driver whose commands must each end within ten minutes deploys this app
per card and spawns / collects its cells instead
(bench/modal_sweep_drive.py; the receipts are written by the same
write_cell).

Verdicts come from OUTPUT — the record's fields, bench's own gate line
("bit-exact gate: N/N tensors verified by a fresh round trip") — never
from exit codes.
"""

import json
import os
import re
import signal
import subprocess
import threading
import time

import modal

# ---------------------------------------------------------------- the card --
# Read at import for the decorator (Modal fixes gpu/memory/cpu/timeout when
# the function is defined), and passed into the function again as `gpu_label`
# so the container never has to know an env it does not have.
GPU = os.environ.get("AB_GPU", "L4")
SHAPES = {
    # memory is MiB (Modal's unit); timeout seconds.
    "L4": {"memory": 32 * 1024, "cpu": 8, "timeout": 2700},
    "A100-80GB": {"memory": 96 * 1024, "cpu": 16, "timeout": 3600},
    "H100": {"memory": 96 * 1024, "cpu": 16, "timeout": 3600},
    # 48 GB: the 27B's BF16 does not fit, so its stock arm is skipped_predicted_nonfit and the record is a
    # fit point. Same host shape as bench/modal_spec_range.py's L40S cell.
    "L40S": {"memory": 96 * 1024, "cpu": 16, "timeout": 3600},
}
if GPU not in SHAPES:
    raise SystemExit(f"modal_sweep: AB_GPU must be one of {sorted(SHAPES)} — refusing {GPU!r}")
SHAPE = SHAPES[GPU]
assert SHAPE["timeout"] <= 3600, "the cost ceiling: timeout <= 3600 on every function"
# the substring torch's device name must carry for the run to be about this card
CARD_WITNESS = {"L4": "L4", "A100-80GB": "A100", "H100": "H100", "L40S": "L40S"}

app = modal.App("drinkme-sweep")

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
REMOTE_REPO = "/root/drinkme"
REMOTE_MEASUREMENTS = f"{REMOTE_REPO}/measurements"  # bench.MEASUREMENTS_DIR under cwd
# a gulp cell's pack, on container disk and outside the uploaded tree
REMOTE_PACKS = "/root/packs"
# the compression profiles a cell can ask for (`drinkme pack --sip / --gulp`);
# sip is bench's in-memory default
PROFILES = ("sip", "gulp")
# meta.json facts a gulp cell's report carries about its pack (codec/pack.py
# writes them; the full meta.json stays in the container)
PACK_FACT_KEYS = ("formatVersion", "profile", "profileWidths", "tensorCount", "meanBpw", "weightedBpw",
                  "packSeconds", "packedBytes", "residentBytes", "radixTensorCount", "rawFallbackTensorCount",
                  "rawTensorCount", "radixEncoder", "radixEncodeSeconds", "manifestSha256", "mtpTensorCount")

# Same deliberately boring base as modal_fresh_box.py: not a torch image,
# not a CUDA devel image — `uv sync` on a bare box must produce the working
# environment by itself (nvidia drivers come from Modal's GPU runtime). The
# ignore list is modal_serve_ab.py's, RECURSIVE globs (the bare forms match
# only the tree's top level): the venv, git, caches, the other worktrees,
# and — the point of this sweep — the local measurements/ and verification/
# collections, none of which a stranger has; and nothing git does not track
# (_untracked), so private working files beside the tree stay off the image.
# g++ is the one addition: bench packs in memory with the radix encoder, which
# JIT-compiles codec/radix_native.cpp and falls back to the numpy encoder
# without a C++ compiler — hours for an 8B (codec/radix_native.py), past
# every timeout below. modal_radix.py's image carries it for the same reason.


def _untracked(repo: str) -> list[str]:
    """Mount-ignore globs for every path in the checkout that git does not
    track, ignored or not, untracked directories collapsed (`git ls-files
    --others --directory`). Run where the image is built, never in the
    container (no .git there); a checkout git cannot read raises rather
    than being uploaded whole."""
    out = subprocess.run(["git", "-C", repo, "ls-files", "-z", "--others", "--directory"],
                         capture_output=True, text=True, check=True).stdout
    globs = []
    for p in filter(None, out.split("\0")):
        p = p.rstrip("/")
        globs += [p, p + "/**"]
    return globs


image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "curl", "g++")
    .run_commands("curl -LsSf https://astral.sh/uv/install.sh | sh")
    .add_local_dir(
        REPO,
        remote_path=REMOTE_REPO,
        ignore=["**/.venv/**", "**/.git/**", "**/.git", "**/__pycache__/**", "**/*.pyc",
                "**/.pytest_cache/**", "**/.worktrees/**", "**/verification/**",
                "**/measurements/**"] + (_untracked(REPO) if modal.is_local() else []),
    )
)

# the log the container returns: the last LOG_CAP characters, carriage-return
# progress lines collapsed to their final state (bench's HF download bars)
LOG_CAP = 3_000_000


def collapse_cr(line: str) -> str:
    """A `\\r`-rewritten progress line keeps only its final state (plus the
    newline it ended with) — the local log then reads like the terminal did,
    not like the byte stream."""
    nl = "\n" if line.endswith("\n") else ""
    body = line[:-1] if nl else line
    return body.rsplit("\r", 1)[-1] + nl


# single_use_containers: one cell, one container, then it is gone. A deployed
# app redeployed for another card handed a new call to the previous cell's
# still-warm container (2026-09-27: an A100-80GB spawn ran on the L4 that had
# just benched, its env already synced), and a warm container is not the
# stranger's fresh box whatever card it is on.
@app.function(image=image, gpu=GPU, timeout=SHAPE["timeout"], memory=SHAPE["memory"], cpu=SHAPE["cpu"],
              single_use_containers=True)
def sweep(model: str, hf_token: str = "", gpu_label: str = "", commit: str = "",
          timeout_s: int = 0, profile: str = "sip") -> dict:
    import glob

    env = dict(os.environ, PATH="/root/.local/bin:" + os.environ["PATH"],
               HF_HUB_DISABLE_PROGRESS_BARS="1", TOKENIZERS_PARALLELISM="false",
               PYTHONUNBUFFERED="1")
    if hf_token:
        env["HF_TOKEN"] = hf_token  # gemma is gated
    if commit:
        env["DRINKME_COMMIT"] = commit

    # gpu_label comes FROM THE LOCAL SIDE (module docstring): `GPU`/`SHAPE`
    # here would re-read an env the container does not have and say "L4" —
    # so nothing below reads them; the shape is looked up by the label.
    shape = SHAPES.get(gpu_label)
    report = {"gate": "drinkme bench on a rented NVIDIA card",
              "gpu": gpu_label or "UNLABELED (the local side passed no card — do not trust)",
              "model": model, "profile": profile, "commit": commit, "shape": shape, "steps": {},
              "step_order": [],
              "started": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())}
    T0 = time.time()
    # bench is killed 150 s before the container timeout so the function can
    # still RETURN its log and a named error instead of vanishing mid-arm
    # (an A100 27B cell, 2026-09-16: Modal's FunctionTimeoutError, no
    # log, no step table — the 90 s reaper below killed `uv run` but not the
    # `drinkme bench` python under it, which kept the pipe open and the read
    # loop blocked; hence the process GROUP kill now)
    deadline = T0 + (timeout_s or (shape or {}).get("timeout", 2700)) - 150
    log_parts: list[str] = []
    log_len = 0

    def log(text: str) -> None:
        nonlocal log_len
        print(text, end="" if text.endswith("\n") else "\n", flush=True)
        log_parts.append(text if text.endswith("\n") else text + "\n")
        log_len += len(log_parts[-1])
        if log_len > 2 * LOG_CAP:  # trim rarely, keep the tail
            joined = "".join(log_parts)[-LOG_CAP:]
            log_parts[:] = [joined]
            log_len = len(joined)

    def run(label, cmd, cwd=REMOTE_REPO):
        """Stream a step's output as it runs (a step that dies with the
        container still leaves its lines in the local `modal run` output),
        keep the tail, kill it at the deadline."""
        log(f"\n=== [{time.time() - T0:7.1f}s] {label}: {' '.join(cmd)}\n")
        t0 = time.time()
        # its own session, so the deadline kill reaches the whole tree: `uv
        # run` execs nothing — the bench python is its child, and killing
        # the parent alone leaves the child writing into our pipe
        p = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True, errors="replace",
                             start_new_session=True)
        killed = {"at": None}

        def reaper():
            while p.poll() is None:
                if time.time() >= deadline:
                    killed["at"] = round(time.time() - T0, 1)
                    try:
                        os.killpg(p.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    return
                time.sleep(2)

        threading.Thread(target=reaper, daemon=True).start()
        tail = ""
        for line in p.stdout:
            line = collapse_cr(line)
            log(line)
            tail = (tail + line)[-6000:]
        p.wait()
        dt = time.time() - t0
        verdict = (f"KILLED at the deadline ({killed['at']}s into the container)"
                   if killed["at"] is not None else f"exit {p.returncode}")
        log(f"=== [{time.time() - T0:7.1f}s] {label}: {verdict} after {dt:.1f}s\n")
        report["steps"][label] = {"exit": p.returncode, "seconds": round(dt, 1),
                                  "killed_at_deadline": killed["at"], "tail": tail[-3000:]}
        report["step_order"].append(label)
        return p.returncode

    def done(error=None):
        if error:
            report["error"] = error
            log(f"\nFAILED: {error}\n")
        report["wall_s"] = round(time.time() - T0, 1)
        report["log"] = "".join(log_parts)[-LOG_CAP:]
        return report

    try:
        smi = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                              "--format=csv,noheader"], capture_output=True, text=True,
                             timeout=30).stdout.strip()
    except Exception as e:  # noqa: BLE001
        smi = f"nvidia-smi unavailable: {e}"
    report["nvidia_smi"] = smi
    log(f"nvidia-smi: {smi}\n")
    log(f"card label (from the local side): {report['gpu']}; model {model}; profile {profile}; "
        f"commit {commit or '?'}\n")
    if profile not in PROFILES:
        return done(f"profile must be one of {PROFILES}, not {profile!r}")

    # 1. the stranger's sync: base deps only (default-groups = [])
    if run("uv sync (default groups)", ["uv", "sync"]):
        return done("uv sync failed")

    # 2. the box picks its accelerator group — the exact call `drinkme bench`
    #    makes before anything else (cli.py -> bootstrap.ensure_accelerator),
    #    made here on its own so the step is named in the report. torch is
    #    absent at this point, so no re-exec happens.
    if run("bootstrap.ensure_accelerator", [
            "uv", "run", "--no-sync", "python", "-c",
            "from drinkme.bootstrap import ensure_accelerator; ensure_accelerator()"]):
        return done("bootstrap.ensure_accelerator failed")

    # 3. torch identity: the card this run is about, or nothing; plus the
    #    version bench will stamp and whether the native radix encoder
    #    compiles (without it bench's in-memory pack takes hours — the image
    #    comment above)
    code = run("torch identity", ["uv", "run", "--no-sync", "python", "-c",
                                  "import json, torch, drinkme; from drinkme.codec import radix_native; "
                                  "print(json.dumps({'torch': torch.__version__, "
                                  "'cuda': torch.cuda.is_available(), 'device': torch.cuda.get_device_name(0) "
                                  "if torch.cuda.is_available() else None, 'capability': "
                                  "list(torch.cuda.get_device_capability()) if torch.cuda.is_available() "
                                  "else None, 'drinkme': drinkme.__version__, "
                                  "'radix_native_available': radix_native.available()}))"])
    ident = {}
    for line in report["steps"]["torch identity"]["tail"].splitlines():
        if line.startswith("{"):
            try:
                ident = json.loads(line)
            except ValueError:
                pass
    report["torch_identity"] = ident
    witness = CARD_WITNESS.get(gpu_label, gpu_label)
    if code or not ident.get("cuda") or witness not in (ident.get("device") or ""):
        return done(f"torch identity is not a CUDA {gpu_label or '?'}: {ident}")
    if not ident.get("radix_native_available"):
        return done(f"the radix native encoder does not compile here: {ident}")

    bench_cmd = ["uv", "run", "--no-sync", "drinkme", "bench", "--model", model]
    if profile == "gulp":
        # 3a/3b. the gulp point: cut the pack, check it in place,
        #    then time the compressed arm off it (module docstring)
        pack_dir = f"{REMOTE_PACKS}/{_slug(model)}-gulp"
        code = run("drinkme pack --gulp", ["uv", "run", "--no-sync", "drinkme", "pack", "--model", model,
                                           "--gulp", "-o", pack_dir])
        if code or not os.path.exists(os.path.join(pack_dir, "meta.json")):
            return done(f"drinkme pack --gulp made no pack at {pack_dir} (exit {code})")
        report["pack"] = pack_facts(pack_dir)
        log(f"--- pack facts: {json.dumps(report['pack'])}\n")
        if report["pack"].get("profile") != profile:
            return done(f"the pack at {pack_dir} is profile {report['pack'].get('profile')!r}, not {profile!r}")
        code = run("drinkme verify", ["uv", "run", "--no-sync", "drinkme", "verify", "--pack-dir", pack_dir])
        if not pack_verified(report["steps"]["drinkme verify"]["tail"]):
            return done(f"drinkme verify did not verify the pack at {pack_dir} (exit {code})")
        bench_cmd += ["--pack-dir", pack_dir]

    # 4. THE RUN. No -o: bench names the record itself under ./measurements/
    #    (empty in this container — the local one is excluded from the
    #    upload), and the confirms take their defaults off a non-TTY stdin
    #    (bench._confirm; the one confirm, memory kind, is only asked on a
    #    unified CUDA part and defaults to yes).
    os.makedirs(REMOTE_MEASUREMENTS, exist_ok=True)
    before = set(glob.glob(f"{REMOTE_MEASUREMENTS}/*.json"))
    code = run("drinkme bench", bench_cmd)
    report["bench_exit"] = code
    written = sorted(set(glob.glob(f"{REMOTE_MEASUREMENTS}/*.json")) - before, key=os.path.getmtime)
    if not written:
        return done(f"drinkme bench wrote no record (exit {code}) — read the bench step's tail")
    if len(written) > 1:
        report["extra_records"] = [os.path.basename(p) for p in written[:-1]]
    path = written[-1]
    with open(path) as f:
        text = f.read()
    try:
        record = json.loads(text)
    except ValueError as e:
        return done(f"the record at {path} is not JSON: {e}")
    report["record_basename"] = os.path.basename(path)
    wrong = profile_mismatch(record, profile)
    if wrong:
        # not a record of this cell: its bytes ride in the meta, never into records/
        report["refused_record_text"] = text
        return done(wrong)
    report["record_text"] = text
    log(f"\n=== record written (in the container): {report['record_basename']} "
        f"({len(text)} bytes)\n")
    return done()


def pack_facts(pack_dir: str) -> dict:
    """A gulp cell's pack as its meta.json describes it (PACK_FACT_KEYS),
    plus its size on disk."""
    try:
        with open(os.path.join(pack_dir, "meta.json")) as f:
            meta = json.load(f)
        facts = {k: meta.get(k) for k in PACK_FACT_KEYS if k in meta}
    except (OSError, ValueError) as e:
        facts = {"error": f"meta.json unreadable: {type(e).__name__}: {e}"}
    try:
        facts["du_bytes"] = int(subprocess.run(["du", "-sb", pack_dir], capture_output=True,
                                               text=True).stdout.split()[0])
    except Exception as e:  # noqa: BLE001
        facts["du_bytes"] = f"du failed: {e}"
    facts["pack_dir"] = pack_dir
    return facts


def pack_verified(tail: str) -> bool:
    """Whether `drinkme verify`'s output passed: a line that
    starts "verified, manifest and all: ", and no refusal, mismatch or
    traceback anywhere."""
    return (any(line.startswith("verified, manifest and all: ") for line in tail.splitlines())
            and not re.search(r"REFUSED|mismatch|Traceback", tail, re.IGNORECASE))


def profile_mismatch(record: dict, profile: str) -> str | None:
    """None when the record's compression.profile is the profile the cell
    asked for, else the cell's error. bench reads the profile off the
    loaded arm (arms.arm_compression_profile), so this is what was timed."""
    got = (record.get("compression") or {}).get("profile")
    if got == profile:
        return None
    return f"the record's compression.profile is {got!r}, not the {profile!r} this cell asked for"


# ----------------------------------------------------------- the local side --


def free_record_path(out_dir: str, basename: str) -> str:
    """bench.measurement_path's rule for a name that already exists in
    `out_dir`: the first free of `<stem>.json`, `<stem>_2.json`,
    `<stem>_3.json` … — never overwrite. `basename` is the container's,
    which is always the `n=1` form (its measurements/ started empty)."""
    stem = basename[:-5] if basename.endswith(".json") else basename
    n = 1
    while True:
        path = os.path.join(out_dir, f"{stem}{'' if n == 1 else f'_{n}'}.json")
        if not os.path.exists(path):
            return path
        n += 1


def main_checkout() -> str:
    """The main checkout of this repo even when run from a worktree — the
    records are ONE collection under the main checkout's measurements/, not
    one per branch. `git rev-parse --git-common-dir` is the main .git."""
    try:
        common = subprocess.run(["git", "rev-parse", "--git-common-dir"], cwd=REPO,
                                capture_output=True, text=True, check=True).stdout.strip()
        common = os.path.abspath(os.path.join(REPO, common))
        if os.path.basename(common) == ".git":
            return os.path.dirname(common)
    except Exception:  # noqa: BLE001
        pass
    return REPO


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9.]+", "-", text.lower()).strip("-.") or "x"


def summarize(record: dict, report: dict, app_id) -> str:
    """The lines a detached operator greps for: the fields the gate reads."""
    env = record.get("environment") or {}
    raw = record.get("raw") or {}
    met = {m["name"]: m["value"] for m in record.get("metrics", [])}
    lines = [
        f"deviceClass {env.get('deviceClass')!r} · memoryKind {env.get('memoryKind')} · platform {env.get('platform')}"
        f" · memoryBytes {env.get('memoryBytes')} · cpuInfo {env.get('cpuInfo')!r}",
        f"model {(record.get('model') or {}).get('name')} · compression.profile "
        f"{(record.get('compression') or {}).get('profile')} · bpw {(record.get('compression') or {}).get('bitsPerWeight')}",
        f"stock.outcome {(record.get('stock') or {}).get('outcome')} · raw.twin_outcome {raw.get('twin_outcome')}"
        + (f" · stock.error {str((record.get('stock') or {}).get('error'))[:160]!r}"
           if (record.get("stock") or {}).get("error") else ""),
        f"decode tok/s: stock {met.get('stock_decode_tok_s')} · twin (raw) {raw.get('twin_decode_tok_s')} · "
        f"compressed {met.get('compressed_decode_tok_s')} · read {met.get('read_gb_s')} GB/s",
        f"roundtrip gate: {raw.get('verified_tensors')}/{raw.get('swapped_linears')} tensors verified "
        f"-> {'PASS' if raw.get('verified_tensors') is not None and raw.get('verified_tensors') == raw.get('swapped_linears') else 'CHECK'}",
        f"container wall {report.get('wall_s')} s · steps "
        + ", ".join(f"{s} {report['steps'][s]['seconds']}s exit {report['steps'][s]['exit']}"
                    for s in report.get("step_order", []))
        + f" · app {app_id}",
    ]
    return "\n".join(lines)


def local_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:  # noqa: BLE001
        return ""


def local_token() -> str:
    """The HF token the local side passes into the container (gemma is
    gated): $HF_TOKEN, else the hub's token file; no Modal secret."""
    token = os.environ.get("HF_TOKEN", "")
    tp = os.path.expanduser("~/.cache/huggingface/token")
    if not token and os.path.exists(tp):
        token = open(tp).read().strip()
    return token


def utc_stamp() -> str:
    import datetime

    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")


def cell_stem(model: str, gpu: str, profile: str, stamp: str) -> str:
    """A cell's name before it has a record (the spawn's call JSON, a
    failure's log and meta): model, card, profile and the spawn's stamp, so
    a sip and a gulp cell of one model and card never share a file."""
    return f"{_slug(model)}_{_slug(gpu)}_{_slug(profile)}_{stamp}"


def write_cell(report: dict, model: str, gpu: str, out_dir: str, log_dir: str, stamp: str,
               profile: str = "sip") -> int:
    """The local side's receipts for one returned cell (the entrypoint's and
    bench/modal_sweep_drive.py's alike): the record bytes verbatim under
    out_dir, the log and a meta JSON under log_dir. 0 with a record, 1
    without one (a detached dispatch must not smile). A record's log and
    meta take the record's own stem, which carries a non-default profile by
    bench's rule (`…_gulp`); a failure's take FAILED_ + cell_stem."""
    record_text = report.pop("record_text", None)
    log_text = report.pop("log", "")
    if record_text is not None:
        path = free_record_path(out_dir, report["record_basename"])
        stem = os.path.basename(path)[:-5]
    else:
        path = None
        stem = f"FAILED_{cell_stem(model, gpu, profile, stamp)}"
    log_path = os.path.join(log_dir, f"{stem}.log")
    with open(log_path, "w") as f:
        f.write(log_text)
    meta_path = os.path.join(log_dir, f"{stem}.meta.json")
    with open(meta_path, "w") as f:
        json.dump({**report, "record_path": path, "log_path": log_path}, f, indent=1)
    print(f"[sweep] log -> {log_path}\n[sweep] meta -> {meta_path}", flush=True)
    print("[sweep] steps: " + ", ".join(
        f"{s} {report['steps'][s]['seconds']}s exit {report['steps'][s]['exit']}"
        for s in report.get("step_order", [])), flush=True)
    if "error" in report:
        print(f"[sweep] ERROR: {report['error']}", flush=True)

    if record_text is None:
        # no record = the run FAILED, whatever the step tails say
        print("[sweep] NO RECORD — read the log", flush=True)
        return 1

    # the bytes bench wrote, verbatim — nothing is reshaped between bench and publish
    with open(path, "w") as f:
        f.write(record_text)
    record = json.loads(record_text)
    print(f"[sweep] record written -> {path}"
          + (f" (the container's name {report['record_basename']} was taken)"
             if os.path.basename(path) != report["record_basename"] else ""), flush=True)
    print(summarize(record, report, report.get("app_id")), flush=True)
    return 0


@app.local_entrypoint()
def main(model: str, out_dir: str = "", log_dir: str = "", profile: str = "sip"):
    out_dir = out_dir or os.path.join(main_checkout(), "measurements")
    log_dir = log_dir or os.path.join(REPO, "verification", "modal-sweep")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    commit = local_commit()
    token = local_token()

    stamp = utc_stamp()
    if profile not in PROFILES:
        raise SystemExit(f"modal_sweep: --profile must be one of {PROFILES}, not {profile!r}")
    print(f"[sweep] card {GPU} ({SHAPE}) · model {model} · profile {profile} · commit {commit[:12] or '?'} · "
          f"records -> {out_dir} · logs -> {log_dir}", flush=True)
    t0 = time.time()
    try:
        report = sweep.remote(model, token, GPU, commit, SHAPE["timeout"], profile)
    except Exception as e:  # noqa: BLE001 — a timeout/crash still gets a local receipt
        report = {"gpu": GPU, "model": model, "profile": profile, "commit": commit, "shape": SHAPE, "steps": {},
                  "step_order": [], "error": f"sweep.remote raised {type(e).__name__}: {e}",
                  "log": f"(no log came back: {type(e).__name__}: {e} — the streamed `modal run` "
                         "output is the only trace)\n"}
    report["local_wall_s"] = round(time.time() - t0, 1)
    report["app_id"] = app.app_id
    if write_cell(report, model, GPU, out_dir, log_dir, stamp, profile):
        # exit nonzero so a detached dispatch reports failure instead of smiling
        raise SystemExit(1)
