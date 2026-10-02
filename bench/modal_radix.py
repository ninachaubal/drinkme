"""The bf16 codec on NVIDIA: pack, `drinkme bench --pack-dir`, served
spec-off / spec-on — on rented Modal cards, one (card, model, codec) cell
per container.

Built from bench/modal_sweep.py (the publishing sweep's leg, left
byte-identical for its own cells); the differences are the shape of a cell
and how it is driven:

  cell = (card, model, codec ∈ {sip, gulp})
    0. nvidia-smi identity; `uv sync --frozen --no-default-groups --group
       cuda` against the env PREBUILT INTO THE IMAGE (below) — seconds, not
       the sweep's per-container torch download; bootstrap.ensure_accelerator
       as the CLI would call it (a no-op on a working env); torch/triton
       identity — the device name must witness the card.
    1. gates (first cell of a card only): bench/radix_gemv_bitpin.py,
       bench/radix_mc_bitpin.py and bench/radix_twin_bitpin.py --no-real
       (the bench twin against the compressed module), each plus --selftest
       (must print FAIL). A
       Triton compile failure on CUDA is a FINDING — recorded, never fixed
       here.
    2. `drinkme pack --model <M> --sip|--gulp -o <dir>` on the container
       CPU (the checkpoint comes from HF inside Modal — never from the
       local box); pack facts: wall, `du -sb`, meta.json.
    3. `DRINKME_BENCH_WARMUP=1 drinkme bench --model <M> --pack-dir <dir>`,
       no -o (bench names the record; the bytes come back verbatim).
    4. served pair: `drinkme serve --pack-dir <dir> --ctx 8192` driven by
       bench/serve_drive.py in the same container — the 76-token prompt,
       128 tokens, 1 warm-up + 5 timed; spec off = DRINKME_SPEC=off,
       spec on = default (ngram on Qwen3-8B). Server and driver both live
       in this process's care; the server is killed by process group.

  Every step is killed 150 s before the container timeout (the sweep's
  reaper, process-group kill) so the function returns receipts rather than
  dying silent. The cell returns the bench record bytes, the served JSONs,
  the server logs, the gate receipts, the pack facts, the identities and
  the full log; the local side (bench/modal_radix_drive.py) writes them
  under cuda/<card>/ verbatim.

THE ENVIRONMENT IS BUILT INTO THE IMAGE: pyproject.toml + uv.lock (+ the
README/LICENSE/lexicon hatchling's build reads and a stub package) are copied
in and `uv sync --no-default-groups --group cuda` runs AT IMAGE BUILD into
/opt/venv (UV_PROJECT_ENVIRONMENT). Modal caches that layer, so the ~3 GB
torch install is paid once per lock, not per cell; the runtime sync in
the mounted checkout only re-links the editable `drinkme` at
/root/drinkme/src. THE BARE `uv sync` HERE RUNS INSIDE A DISPOSABLE MODAL
IMAGE BUILD AND MUST NEVER BE COPIED TO A DEVELOPMENT BOX (AGENTS.md).
g++ is in the image: the radix native encoder JIT-compiles
codec/radix_native.cpp; without a compiler the numpy encoder packs an 8B
in hours.

THE CARD IS THE FUNCTION. modal_sweep.py's AB_GPU module global (read at
import for the decorator, re-read inside the container as an unset env ->
"L4") is gone: there is one @app.function per card, each with its shape
fixed in its own decorator and its label passed into run_cell as a
literal, so nothing in the container ever reads an env to learn which
card it is on. The device name torch reports must still witness it.

Shapes (memory MiB, Modal's unit; timeout <= 3600, the cost ceiling,
asserted): L4 / A10G / L40S 48 GiB, 16 cpu; H100 96 GiB, 16 cpu.

Driving (the local side, under the 10-minute foreground ceiling):
    ~/.local/bin/uv tool run modal deploy bench/modal_radix.py
    <modal tool python> bench/modal_radix_drive.py spawn --card L4 --codec sip --gates
    <modal tool python> bench/modal_radix_drive.py collect --card L4 --codec sip --wait 540

The 27B stretch (shape (b), a Volume): pack27 (CPU only, no GPU) downloads
and packs onto the volume `drinkme-radix-27b`; cell27_L40S benches and
serves from it. Only attempted after the twelve 8B cells are in.

Verdicts come from OUTPUT — the gates' verdict lines, the record's fields,
the driver's summary — never from exit codes.
"""

import json
import os
import re
import signal
import subprocess
import threading
import time

import modal

APP_NAME = "drinkme-radix-cuda"
app = modal.App(APP_NAME)

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
REMOTE_REPO = "/root/drinkme"
REMOTE_MEASUREMENTS = f"{REMOTE_REPO}/measurements"  # bench.MEASUREMENTS_DIR under cwd
VENV = "/opt/venv"                # UV_PROJECT_ENVIRONMENT, built into the image
LOCK_DIR = "/opt/drinkme-lock"    # the image-build project: lock + metadata + stub
HOME_DIR = "/root/drinkme-home"   # DRINKME_HOME: packs, the native encoder's .so
OUT_DIR = "/root/out"             # gate receipts, served JSONs, server logs
SERVE_PORT = 3299
DEADLINE_MARGIN_S = 150           # steps are killed this long before the container timeout

# memory is MiB (Modal's unit); timeout seconds. cpu 16 everywhere: the
# pack is a CPU job (Strix Halo: sip 54 s / gulp 199 s on 32 threads)
# and 16 budgets 2-3x that.
SHAPES = {
    "L4":   {"gpu": "L4",   "memory": 48 * 1024, "cpu": 16, "timeout": 3600, "witness": r"\bL4\b"},
    "A10G": {"gpu": "A10G", "memory": 48 * 1024, "cpu": 16, "timeout": 3600, "witness": r"\bA10G?\b"},
    "L40S": {"gpu": "L40S", "memory": 48 * 1024, "cpu": 16, "timeout": 3600, "witness": r"\bL40S\b"},
    "H100": {"gpu": "H100", "memory": 96 * 1024, "cpu": 16, "timeout": 3600, "witness": r"\bH100\b"},
}
for _card, _shape in SHAPES.items():
    assert _shape["timeout"] <= 3600, f"the cost ceiling: timeout <= 3600 on every function ({_card})"

# The cell vocabulary: a codec label IS a profile name, and names the
# `drinkme pack` flag — "sip" -> --sip, "gulp" -> --gulp. (The receipts
# under cuda/ predate the names and carry the old labels.)
PROFILE_OF = {"sip": "--sip", "gulp": "--gulp"}
CODECS = tuple(PROFILE_OF)


def pack_argv(model: str, codec: str, pack_dir: str) -> list[str]:
    """The `drinkme pack` command for a cell's codec label, or a ValueError
    naming a label that is not a profile."""
    if codec not in PROFILE_OF:
        raise ValueError(f"codec {codec!r} is not a profile (one of {list(PROFILE_OF)})")
    return ["uv", "run", "--no-sync", "drinkme", "pack", "--model", model,
            PROFILE_OF[codec], "-o", pack_dir, "--replace"]

# The mount: the checkout minus everything a cell does not need and
# everything that must never ride a phone hotspot (tens of MB, not
# hundreds — measurements/, packs, *.npz, .worktrees/, receipts, the site). The
# globs are RECURSIVE (the bare forms match only the tree's top level).
# Nothing git does not track rides along (_untracked): private working
# files a checkout keeps beside the tree stay off the image, whatever they
# are named. The served step's driver is bench/serve_drive.py.


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


MOUNT_IGNORE = [
    "**/.venv/**", "**/.git/**", "**/.git", "**/__pycache__/**", "**/*.pyc",
    "**/.pytest_cache/**", "**/.ruff_cache/**", "**/.mypy_cache/**", "**/.worktrees/**",
    "**/verification/**", "**/measurements/**", "**/results/**", "**/cuda/**",
    "**/node_modules/**",
    "**/*.npz", "**/*.log", "**/*.pt", "**/*.safetensors", "**/*.png", "**/*.jpg",
    "**/site/**", "**/tests/**",
    # local measurement output directories
    "**/parity/**", "**/v4/**", "**/nvidia2/**",
] + (_untracked(REPO) if modal.is_local() else [])

# Same deliberately boring base as the sweep (not a torch image, not a CUDA
# devel image: the lock produces the environment by itself; drivers come
# from Modal's GPU runtime), plus g++ for the radix native encoder, with
# the accelerator environment resolved AT BUILD TIME from the lock alone.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "curl", "g++")
    .run_commands("curl -LsSf https://astral.sh/uv/install.sh | sh")
    .env({"UV_PROJECT_ENVIRONMENT": VENV})
    .add_local_file(f"{REPO}/pyproject.toml", f"{LOCK_DIR}/pyproject.toml", copy=True)
    .add_local_file(f"{REPO}/uv.lock", f"{LOCK_DIR}/uv.lock", copy=True)
    .add_local_file(f"{REPO}/README.md", f"{LOCK_DIR}/README.md", copy=True)
    .add_local_file(f"{REPO}/LICENSE", f"{LOCK_DIR}/LICENSE", copy=True)
    # the wheel force-includes the lexicon (pyproject [tool.hatch...force-include])
    .add_local_file(f"{REPO}/lexicons/wtf.petrichor.drinkme.measurement.json",
                    f"{LOCK_DIR}/lexicons/wtf.petrichor.drinkme.measurement.json", copy=True)
    .run_commands(
        f"mkdir -p {LOCK_DIR}/src/drinkme && touch {LOCK_DIR}/src/drinkme/__init__.py",
        # THE BARE `uv sync` OF A DISPOSABLE IMAGE BUILD (module docstring)
        f"cd {LOCK_DIR} && /root/.local/bin/uv sync --frozen --no-default-groups --group cuda",
        f"{VENV}/bin/python -c 'import torch, triton; print(torch.__version__, triton.__version__)'",
    )
    .add_local_dir(REPO, remote_path=REMOTE_REPO, ignore=MOUNT_IGNORE)
)

# the log the container returns: the last LOG_CAP characters, carriage-return
# progress lines collapsed to their final state
LOG_CAP = 3_000_000


def collapse_cr(line: str) -> str:
    """A `\\r`-rewritten progress line keeps only its final state (plus the
    newline it ended with)."""
    nl = "\n" if line.endswith("\n") else ""
    body = line[:-1] if nl else line
    return body.rsplit("\r", 1)[-1] + nl


def _read(path: str, cap: int = LOG_CAP) -> str | None:
    try:
        with open(path, errors="replace") as f:
            return f.read()[-cap:]
    except OSError:
        return None


def run_cell(card: str, model: str, codec: str, gates: bool, hf_token: str = "",
             commit: str = "", pack_dir_override: str = "", skip_pack: bool = False,
             skip_bench: bool = False, serve_ctx: int = 8192, codecs: list | None = None) -> dict:
    """One cell, inside the container. `card` is a literal from the calling
    @app.function — never an env. Returns the receipts (module docstring).
    `codecs` narrows/orders codec="all"; `pack_dir_override` may carry a
    `{codec}` placeholder (the 27B pair off the volume)."""
    import glob
    import shutil

    shape = SHAPES[card]
    env = dict(os.environ, PATH="/root/.local/bin:" + os.environ["PATH"],
               UV_PROJECT_ENVIRONMENT=VENV, DRINKME_HOME=HOME_DIR,
               HF_HUB_DISABLE_PROGRESS_BARS="1", TOKENIZERS_PARALLELISM="false",
               PYTHONUNBUFFERED="1",
               OMP_NUM_THREADS=str(shape["cpu"]), MKL_NUM_THREADS=str(shape["cpu"]))
    if hf_token:
        env["HF_TOKEN"] = hf_token
    py = f"{VENV}/bin/python"

    report = {"gate": "radix on NVIDIA: pack / bench --pack-dir / served spec off+on",
              "gpu": card, "model": model, "codec": codec, "gates_requested": gates,
              "commit": commit, "shape": shape, "steps": {}, "step_order": [],
              "started": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())}
    T0 = time.time()
    deadline = T0 + shape["timeout"] - DEADLINE_MARGIN_S
    log_parts: list[str] = []
    log_len = 0

    def log(text: str) -> None:
        nonlocal log_len
        print(text, end="" if text.endswith("\n") else "\n", flush=True)
        log_parts.append(text if text.endswith("\n") else text + "\n")
        log_len += len(log_parts[-1])
        if log_len > 2 * LOG_CAP:
            joined = "".join(log_parts)[-LOG_CAP:]
            log_parts[:] = [joined]
            log_len = len(joined)

    def run(label, cmd, cwd=REMOTE_REPO, extra_env=None, tail_cap=6000):
        """Stream a step's output as it runs, keep the tail, kill the whole
        process group at the deadline (modal_sweep.py's pattern)."""
        log(f"\n=== [{time.time() - T0:7.1f}s] {label}: {' '.join(cmd)}\n")
        t0 = time.time()
        p = subprocess.Popen(cmd, cwd=cwd, env={**env, **(extra_env or {})},
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                             errors="replace", start_new_session=True)
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
            tail = (tail + line)[-2 * tail_cap:]
        p.wait()
        dt = time.time() - t0
        verdict = (f"KILLED at the deadline ({killed['at']}s into the container)"
                   if killed["at"] is not None else f"exit {p.returncode}")
        log(f"=== [{time.time() - T0:7.1f}s] {label}: {verdict} after {dt:.1f}s\n")
        report["steps"][label] = {"exit": p.returncode, "seconds": round(dt, 1),
                                  "killed_at_deadline": killed["at"], "tail": tail[-tail_cap:]}
        report["step_order"].append(label)
        return p.returncode

    def done(error=None):
        if error:
            report["error"] = error
            log(f"\nFAILED: {error}\n")
        report["wall_s"] = round(time.time() - T0, 1)
        report["log"] = "".join(log_parts)[-LOG_CAP:]
        return report

    def verdict_line(text: str, prefix: str) -> str | None:
        for line in (text or "").splitlines():
            if line.startswith(prefix):
                return line.strip()
        return None

    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(HOME_DIR, exist_ok=True)

    # 0. identity
    try:
        smi = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                              "--format=csv,noheader"], capture_output=True, text=True,
                             timeout=30).stdout.strip()
    except Exception as e:  # noqa: BLE001
        smi = f"nvidia-smi unavailable: {e}"
    report["nvidia_smi"] = smi
    try:
        report["cpu_count_visible"] = os.cpu_count()
        with open("/proc/cpuinfo") as f:
            models = [l.split(":", 1)[1].strip() for l in f if l.startswith("model name")]
        report["cpu_model"] = models[0] if models else None
    except Exception:  # noqa: BLE001
        pass
    log(f"nvidia-smi: {smi}\n")
    log(f"card (the calling function's literal): {card}; model {model}; codec {codec}; "
        f"gates {gates}; commit {commit or '?'}; cpu visible {report.get('cpu_count_visible')} "
        f"({report.get('cpu_model')})\n")

    # the runtime sync: the image holds the resolved cuda environment at
    # /opt/venv; this re-links the editable drinkme at /root/drinkme/src
    if run("uv sync --frozen --group cuda (prebuilt env)",
           ["uv", "sync", "--frozen", "--no-default-groups", "--group", "cuda"]):
        return done("uv sync failed")
    if run("bootstrap.ensure_accelerator", [
            "uv", "run", "--no-sync", "python", "-c",
            "from drinkme.bootstrap import ensure_accelerator; ensure_accelerator()"]):
        return done("bootstrap.ensure_accelerator failed")

    code = run("torch identity", ["uv", "run", "--no-sync", "python", "-c", (
        "import json, torch, triton, numpy, transformers, drinkme, os; "
        "from drinkme.codec import radix_native; "
        "print(json.dumps({'torch': torch.__version__, 'triton': triton.__version__, "
        "'numpy': numpy.__version__, 'transformers': transformers.__version__, "
        "'cuda': torch.cuda.is_available(), "
        "'device': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, "
        "'capability': list(torch.cuda.get_device_capability()) if torch.cuda.is_available() else None, "
        "'cuda_runtime': torch.version.cuda, "
        "'drinkme_file': os.path.abspath(drinkme.__file__), "
        "'radix_native_available': radix_native.available()}))")])
    ident = {}
    for line in report["steps"]["torch identity"]["tail"].splitlines():
        if line.startswith("{"):
            try:
                ident = json.loads(line)
            except ValueError:
                pass
    report["torch_identity"] = ident
    if code or not ident.get("cuda") or not re.search(shape["witness"], ident.get("device") or ""):
        return done(f"torch identity is not a CUDA {card}: {ident}")
    if not ident.get("drinkme_file", "").startswith(REMOTE_REPO):
        return done(f"drinkme imports from {ident.get('drinkme_file')!r}, not the mounted checkout")
    if not ident.get("radix_native_available"):
        # record the compiler's own words; the numpy encoder would take hours
        cxx = shutil.which("c++") or shutil.which("g++") or "c++"
        run("radix native compile (diagnostic)", [cxx, "-O3", "-std=c++17", "-Wall", "-Wextra",
                                                  "-Werror", "-fPIC", "-shared",
                                                  f"{REMOTE_REPO}/src/drinkme/codec/radix_native.cpp",
                                                  "-o", "/tmp/radix_native_diag.so"])
        return done("the radix native encoder does not compile here (see the diagnostic step)")

    # 1. gates, once per card
    report["gates"] = {}
    if gates:
        gate_dir = f"{OUT_DIR}/gates"
        os.makedirs(gate_dir, exist_ok=True)
        # the gemv gate runs as shipped AND with --contiguous-bias (the CUDA
        # finding, in its module comment): the first is the
        # verdict on the gate as shipped, the second the verdict on
        # the arms once the reference holds the bias the module holds
        for script, prefix, flags in (("radix_gemv_bitpin", "RADIX_GEMV_BITPIN", ()),
                                      ("radix_gemv_bitpin", "RADIX_GEMV_BITPIN", ("--contiguous-bias",)),
                                      ("radix_mc_bitpin", "RADIX_MC_BITPIN", ()),
                                      ("radix_twin_bitpin", "RADIX_TWIN_BITPIN", ("--no-real",))):
            for selftest in (False, True):
                name = (script + ("_contiguous-bias" if "--contiguous-bias" in flags else "")
                        + ("_selftest" if selftest else ""))
                jpath = f"{gate_dir}/{name}.json"
                cmd = ["uv", "run", "--no-sync", "python", f"bench/{script}.py", "--json", jpath, *flags]
                if selftest:
                    cmd.append("--selftest")
                rc = run(f"gate {name}", cmd, tail_cap=20000)
                tail = report["steps"][f"gate {name}"]["tail"]
                report["gates"][name] = {
                    "exit": rc,
                    "verdict": verdict_line(tail, prefix),
                    "selftest_line": verdict_line(tail, "SELFTEST"),
                    "json": _read(jpath),
                    "tail": tail,
                }
                log(f"--- gate {name}: verdict {report['gates'][name]['verdict']!r}; "
                    f"selftest {report['gates'][name]['selftest_line']!r}; exit {rc}\n")

    # 2-4. pack, bench, served pair — per codec. A single-codec cell runs
    # one; codec="all" runs both in ONE container (sip, then gulp, so a
    # deadline kill costs the least), on one host: H100 cells have
    # landed on hosts whose stock/twin drift controls differed
    # by 25%, which no cross-container comparison can absorb.
    codecs = (list(codecs) if codecs else list(CODECS)) if codec == "all" else [codec]
    report["codecs"] = codecs
    report["per_codec"] = {}

    def one_codec(cdc: str, sub: dict) -> str | None:
        """Pack + mc gate + bench + served pair for one codec into `sub`;
        returns an error string or None."""
        pack_dir = (pack_dir_override.replace("{codec}", cdc) if pack_dir_override
                    else f"{HOME_DIR}/packs/Qwen--{model}-{cdc}")
        sub["pack_dir"] = pack_dir
        if not skip_pack:
            t_pack = time.time()
            try:
                argv = pack_argv(model, cdc, pack_dir)
            except ValueError as e:
                return str(e)
            rc = run(f"drinkme pack {PROFILE_OF[cdc]}", argv, tail_cap=12000)
            sub["pack_wall_s"] = round(time.time() - t_pack, 1)
            if rc or not os.path.exists(os.path.join(pack_dir, "meta.json")):
                return f"drinkme pack {PROFILE_OF[cdc]} failed (exit {rc}) — read the pack step's tail"
        try:
            du = subprocess.run(["du", "-sb", pack_dir], capture_output=True, text=True).stdout.split()[0]
            sub["pack_du_bytes"] = int(du)
        except Exception as e:  # noqa: BLE001
            sub["pack_du_bytes"] = f"du failed: {e}"
        sub["pack_meta_text"] = _read(os.path.join(pack_dir, "meta.json"))
        try:
            meta = json.loads(sub["pack_meta_text"] or "{}")
            sub["pack_facts"] = {k: meta.get(k) for k in (
                "formatVersion", "profile", "profileWidths", "tensorCount", "meanBpw",
                "weightedBpw", "packSeconds", "packedBytes", "residentBytes", "rawResidentBytes",
                "rawTensorCount", "radixTensorCount", "radixBytes", "rawFallbackTensorCount",
                "rawFallbackBytes", "radixEncoder", "radixEncodeSeconds", "manifestSha256")}
            src = meta.get("source")
            sub["pack_facts"]["sourceRevision"] = src.get("revision") if isinstance(src, dict) else None
            npz = 0
            for p in glob.glob(os.path.join(pack_dir, "**", "*.npz"), recursive=True):
                npz += os.path.getsize(p)
            sub["pack_facts"]["npz_bytes"] = npz
        except Exception as e:  # noqa: BLE001
            sub["pack_facts"] = {"error": f"{type(e).__name__}: {e}"}
        log(f"--- pack facts ({cdc}): du {sub['pack_du_bytes']} · {json.dumps(sub.get('pack_facts'))}\n")
        # the mc gate on the real pack, every radix cell (the first 8 tensors
        # of each pack): a per-card lossless receipt on
        # the bytes the bench and the server are about to read
        sub["gates"] = {}
        os.makedirs(f"{OUT_DIR}/gates", exist_ok=True)
        jpath = f"{OUT_DIR}/gates/radix_mc_bitpin_pack_{cdc}.json"
        step = f"gate radix_mc_bitpin --pack-dir ({cdc})"
        rc = run(step, ["uv", "run", "--no-sync", "python", "bench/radix_mc_bitpin.py", "--json", jpath,
                        "--pack-dir", pack_dir], tail_cap=20000)
        tail = report["steps"][step]["tail"]
        sub["gates"]["radix_mc_bitpin_pack"] = {
            "exit": rc, "verdict": verdict_line(tail, "RADIX_MC_BITPIN"),
            "selftest_line": None, "json": _read(jpath), "tail": tail}

        # 3. bench
        if not skip_bench:
            os.makedirs(REMOTE_MEASUREMENTS, exist_ok=True)
            before = set(glob.glob(f"{REMOTE_MEASUREMENTS}/*.json"))
            code = run(f"drinkme bench --pack-dir ({cdc})", ["uv", "run", "--no-sync", "drinkme", "bench",
                                                            "--model", model, "--pack-dir", pack_dir],
                       extra_env={"DRINKME_BENCH_WARMUP": "1"}, tail_cap=12000)
            sub["bench_exit"] = code
            written = sorted(set(glob.glob(f"{REMOTE_MEASUREMENTS}/*.json")) - before, key=os.path.getmtime)
            if written:
                path = written[-1]
                text = _read(path)
                try:
                    json.loads(text)
                    sub["record_basename"] = os.path.basename(path)
                    sub["record_text"] = text
                    log(f"\n=== record written (in the container): {sub['record_basename']} "
                        f"({len(text)} bytes)\n")
                except ValueError as e:
                    sub["bench_error"] = f"the record at {path} is not JSON: {e}"
            else:
                sub["bench_error"] = f"drinkme bench wrote no record (exit {code}) — read the bench step's tail"
                log(f"\n{sub['bench_error']}\n")

        # 4. served pair — server and driver in this container
        sub["served"] = {}
        drive = "bench/serve_drive.py"
        for spec in ("off", "on"):
            if time.time() > deadline - 300:
                sub["served"][spec] = {"error": "skipped: under 300 s to the deadline"}
                log(f"\n=== served {cdc} spec-{spec}: SKIPPED, under 300 s to the deadline\n")
                continue
            label = f"{cdc}_spec-{spec}"
            slog = f"{OUT_DIR}/{label}.server.log"
            sout = f"{OUT_DIR}/{label}.json"
            senv = {**env, "DRINKME_PREFIX_SLOTS": "0", "DRINKME_SLOT_DIR": "off"}
            senv.pop("DRINKME_SPEC", None)
            if spec == "off":
                senv["DRINKME_SPEC"] = "off"
            cmd = ["uv", "run", "--no-sync", "drinkme", "serve", "--model", model, "--pack-dir", pack_dir,
                   "--port", str(SERVE_PORT), "--ctx", str(serve_ctx), "--yes"]
            log(f"\n=== [{time.time() - T0:7.1f}s] serve {label}: {' '.join(cmd)} "
                f"(DRINKME_SPEC={senv.get('DRINKME_SPEC', '<unset>')}) -> {slog}\n")
            t_boot = time.time()
            with open(slog, "w") as lf:
                server = subprocess.Popen(cmd, cwd=REMOTE_REPO, env=senv, stdout=lf,
                                          stderr=subprocess.STDOUT, start_new_session=True)
            wait = max(60, int(min(900, deadline - time.time() - 120)))
            rc = run(f"drive {label}", ["uv", "run", "--no-sync", "python", drive, "--port", str(SERVE_PORT),
                                        "--label", label, "--server-log", slog, "--out", sout,
                                        "--runs", "5", "--max-tokens", "128", "--wait", str(wait)],
                     extra_env={k: senv[k] for k in ("DRINKME_SPEC", "DRINKME_PREFIX_SLOTS") if k in senv},
                     tail_cap=8000)
            # kill the server by process group; confirm it is gone
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
                "label": label, "drive_exit": rc, "boot_and_drive_s": round(time.time() - t_boot, 1),
                "json": _read(sout), "server_log": server_log[-400_000:],
                "device_line": dev_line, "spec_lines": spec_lines[-8:],
                "server_exit": server.returncode,
            }
            log(f"--- served {label}: drive exit {rc}; device line {dev_line!r}; "
                f"{len(spec_lines)} spec lines; json {'present' if sub['served'][spec]['json'] else 'MISSING'}\n")
        if len(codecs) > 1 and not skip_pack:
            # three packs + the snapshot would sit on the container disk
            # together; this codec's receipts are in hand, its pack can go
            shutil.rmtree(pack_dir, ignore_errors=True)
            log(f"--- removed {pack_dir} (codec=all: one pack on disk at a time)\n")
        return None

    error = None
    for cdc in codecs:
        sub: dict = {}
        report["per_codec"][cdc] = sub
        try:
            err = one_codec(cdc, sub)
        except Exception as e:  # noqa: BLE001 — one codec's crash must not lose the others' receipts
            err = f"{type(e).__name__}: {e}"
        if err:
            sub["error"] = err
            log(f"\n--- {cdc}: {err}\n")
            error = error or err
    if len(codecs) == 1:
        # the single-codec shape the collector read from the first cells:
        # the sub-report flattened into the report, the pack's mc gate
        # beside the synthetic gates
        sub = report.pop("per_codec")[codecs[0]]
        gates_all = {**report.get("gates", {}), **sub.get("gates", {})}
        report.update(sub)
        report["gates"] = gates_all
    return done(error)


# ------------------------------------------------------------ the cards --
# One function per card, the shape fixed in its decorator, the label a
# literal in the body: the container never reads an env to learn its card.

def _fn(card):
    s = SHAPES[card]
    return app.function(image=image, gpu=s["gpu"], timeout=s["timeout"], memory=s["memory"], cpu=s["cpu"])


@_fn("L4")
def cell_L4(model: str, codec: str, gates: bool, hf_token: str = "", commit: str = "", **kw) -> dict:
    return run_cell("L4", model, codec, gates, hf_token, commit, **kw)


@_fn("A10G")
def cell_A10G(model: str, codec: str, gates: bool, hf_token: str = "", commit: str = "", **kw) -> dict:
    return run_cell("A10G", model, codec, gates, hf_token, commit, **kw)


@_fn("L40S")
def cell_L40S(model: str, codec: str, gates: bool, hf_token: str = "", commit: str = "", **kw) -> dict:
    return run_cell("L40S", model, codec, gates, hf_token, commit, **kw)


@_fn("H100")
def cell_H100(model: str, codec: str, gates: bool, hf_token: str = "", commit: str = "", **kw) -> dict:
    return run_cell("H100", model, codec, gates, hf_token, commit, **kw)


# ---------------------------------------------------- the 27B stretch (b) --
# A Volume carries the HF snapshot and the pack between two containers:
# pack27 (CPU only — `drinkme pack` never touches a GPU) and cell27_L40S.
VOL_NAME = "drinkme-radix-27b"
VOL_PATH = "/vol"
vol27 = modal.Volume.from_name(VOL_NAME, create_if_missing=True)


@app.function(image=image, timeout=3600, memory=128 * 1024, cpu=32, volumes={VOL_PATH: vol27})
def pack27(model: str, codec: str, hf_token: str = "", commit: str = "") -> dict:
    """Download + pack onto the volume. HF_HOME on the volume so the
    snapshot (tokenizer, config, shard headers) is there for the bench and
    the server in cell27_L40S. No GPU; the shape is the pack's."""
    return run_pack27(model, codec, hf_token, commit, vol27)


def run_pack27(model: str, codec: str, hf_token: str, commit: str, vol) -> dict:
    """pack27's body, shared with bench/modal_nvidia2.py's app (same volume,
    another deployment): `vol` is the calling app's Volume handle."""
    env = dict(os.environ, PATH="/root/.local/bin:" + os.environ["PATH"],
               UV_PROJECT_ENVIRONMENT=VENV, DRINKME_HOME=f"{VOL_PATH}/drinkme-home",
               HF_HOME=f"{VOL_PATH}/hf", HF_HUB_DISABLE_PROGRESS_BARS="1",
               TOKENIZERS_PARALLELISM="false", PYTHONUNBUFFERED="1",
               OMP_NUM_THREADS="32", MKL_NUM_THREADS="32", DRINKME_NO_AUTO_DEPS="1")
    if hf_token:
        env["HF_TOKEN"] = hf_token
    T0 = time.time()
    deadline = T0 + 3600 - DEADLINE_MARGIN_S
    report = {"gate": "27B stretch: download + pack onto the volume", "model": model, "codec": codec,
              "commit": commit, "steps": {}, "step_order": [], "log": ""}
    lines: list[str] = []

    def run(label, cmd):
        lines.append(f"\n=== [{time.time() - T0:7.1f}s] {label}: {' '.join(cmd)}\n")
        print(lines[-1], flush=True)
        t0 = time.time()
        p = subprocess.Popen(cmd, cwd=REMOTE_REPO, env=env, stdout=subprocess.PIPE,
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
            print(line, end="", flush=True)
            lines.append(line)
            tail = (tail + line)[-20000:]
        p.wait()
        dt = time.time() - t0
        report["steps"][label] = {"exit": p.returncode, "seconds": round(dt, 1),
                                  "killed_at_deadline": killed["at"], "tail": tail[-10000:]}
        report["step_order"].append(label)
        lines.append(f"=== {label}: exit {p.returncode} after {dt:.1f}s (killed {killed['at']})\n")
        return p.returncode

    os.makedirs(f"{VOL_PATH}/hf", exist_ok=True)
    os.makedirs(f"{VOL_PATH}/drinkme-home/packs", exist_ok=True)
    run("uv sync --frozen --group cuda (prebuilt env)",
        ["uv", "sync", "--frozen", "--no-default-groups", "--group", "cuda"])
    pack_dir = f"{VOL_PATH}/drinkme-home/packs/Qwen--{model}-{codec}"
    report["pack_dir"] = pack_dir
    t_pack = time.time()
    rc = run(f"drinkme pack {PROFILE_OF.get(codec, '?')}", pack_argv(model, codec, pack_dir))
    report["pack_wall_s"] = round(time.time() - t_pack, 1)
    report["pack_exit"] = rc
    try:
        vol.commit()
        report["volume_committed"] = True
    except Exception as e:  # noqa: BLE001
        report["volume_committed"] = f"{type(e).__name__}: {e}"
    if os.path.exists(os.path.join(pack_dir, "meta.json")):
        report["pack_meta_text"] = _read(os.path.join(pack_dir, "meta.json"))
        du = subprocess.run(["du", "-sb", pack_dir], capture_output=True, text=True).stdout.split()
        report["pack_du_bytes"] = int(du[0]) if du else None
    try:
        report["hf_snapshot_du"] = subprocess.run(["du", "-sh", f"{VOL_PATH}/hf"], capture_output=True,
                                                  text=True).stdout.strip()
    except Exception:  # noqa: BLE001
        pass
    report["wall_s"] = round(time.time() - T0, 1)
    report["log"] = "".join(lines)[-LOG_CAP:]
    return report


@app.function(image=image, gpu="L40S", timeout=3600, memory=64 * 1024, cpu=16, volumes={VOL_PATH: vol27})
def cell27_L40S(model: str, codec: str, gates: bool = False, hf_token: str = "", commit: str = "",
                serve_ctx: int = 8192, codecs: list | None = None) -> dict:
    """Bench + serve the pack(s) pack27 left on the volume (skip_pack; codec
    "all" + `codecs` = the packs named, on one host). HF_HOME is the
    volume's, where the snapshot already is."""
    os.environ["HF_HOME"] = f"{VOL_PATH}/hf"
    os.environ["DRINKME_NO_AUTO_DEPS"] = "1"
    return run_cell("L40S", model, codec, gates, hf_token, commit,
                    pack_dir_override=f"{VOL_PATH}/drinkme-home/packs/Qwen--{model}-{{codec}}",
                    skip_pack=True, serve_ctx=serve_ctx, codecs=codecs)
