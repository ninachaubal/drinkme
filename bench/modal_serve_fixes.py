"""The serving fixes on Modal: bench/serve_fixes.py on an H100 or an L4, against
main's source tree ("before") and the checkout's ("after") in one container.

The image is bench/modal_cuda_graphs.py's (the cuda environment resolved
from the lock at image build, the checkout mounted minus what git does not
track), plus a "before" `src/` mounted at /root/before/src: the local
directory $SERVE_FIXES_BEFORE names (required; made
with `git archive <main> src | tar -x -C <dir>`).
Volumes: `drinkme-nvidia-speed` (the Qwen3-8B snapshot and sip pack, the
Qwen3.8-27B sip pack), `drinkme-radix-27b` (the 27B's snapshot) and
`drinkme-gemma4-31b` (gemma-4-31B-it's snapshot, staged by `--what stage-gemma`
with the local HF token: the repo is gated).

    uv tool run modal run bench/modal_serve_fixes.py --what stage-gemma
    uv tool run modal run bench/modal_serve_fixes.py --card H100 --model Qwen3-8B \\
        --steps ttft,identity-before,serve-before-eager --tag hb8 --out-dir DIR

Steps (comma-separated, in order, one container): ttft[-before|-after],
identity-before|identity-after (stock then compressed), serve-before|
serve-after (the model's default decode mode), serve-before-eager|
serve-after-eager (DRINKME_CUDA_GRAPHS=0). Every step is killed 150 s before
the container timeout so the function returns receipts. Verdicts come from
the printed lines, never exit codes.
"""

import json
import os
import signal
import subprocess
import threading
import time

import modal

APP_NAME = "drinkme-serve-fixes"
app = modal.App(APP_NAME)

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
REMOTE_REPO = "/root/drinkme"
BEFORE_LOCAL = os.environ.get("SERVE_FIXES_BEFORE")
if not BEFORE_LOCAL:
    raise SystemExit("set SERVE_FIXES_BEFORE to the \"before\" src/ directory (see the module docstring)")
BEFORE_SRC = "/root/before/src"
AFTER_SRC = f"{REMOTE_REPO}/src"
VENV = "/opt/venv"
LOCK_DIR = "/opt/drinkme-lock"
OUT_DIR = "/root/out"
VOL_PATH, VOL27_PATH, VOLG_PATH = "/vol", "/vol27", "/volg"
DEADLINE_MARGIN_S = 150
vol = modal.Volume.from_name("drinkme-nvidia-speed", create_if_missing=False)
vol27 = modal.Volume.from_name("drinkme-radix-27b", create_if_missing=False)
volg = modal.Volume.from_name("drinkme-gemma4-31b", create_if_missing=True)
VOLUMES = {VOL_PATH: vol, VOL27_PATH: vol27, VOLG_PATH: volg}

SHAPES = {
    "L4": {"gpu": "L4", "memory": 32 * 1024, "cpu": 8, "timeout": 3600, "witness": "L4"},
    "H100": {"gpu": "H100!", "memory": 128 * 1024, "cpu": 16, "timeout": 3600, "witness": "H100"},
}

MODELS = {
    # name: (HF_HOME, pack dir or None to serve --stock, ctx for serve)
    "Qwen3-8B": (f"{VOL_PATH}/hf", f"{VOL_PATH}/packs/Qwen3-8B-sip", 8192),
    "Qwen3.8-27B": (f"{VOL27_PATH}/hf", f"{VOL_PATH}/packs/Qwen3.8-27B-sip", 8192),
    "gemma-4-31B-it": (f"{VOLG_PATH}/hf", None, 8192),
}
GEMMA = ("google/gemma-4-31B-it", "842da3794eaa0b77d5f08bae87a17459d91ff475")


def _untracked(repo: str) -> list[str]:
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
    "**/node_modules/**", "**/*.npz", "**/*.log", "**/*.pt", "**/*.safetensors", "**/*.png", "**/*.jpg",
    "**/site/**", "**/tests/**",
] + [g for g in (_untracked(REPO) if modal.is_local() else []) if not g.startswith("bench/serve_fixes")
     and not g.startswith("bench/modal_serve_fixes")]

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "curl", "g++")
    .run_commands("curl -LsSf https://astral.sh/uv/install.sh | sh")
    .env({"UV_PROJECT_ENVIRONMENT": VENV})
    .add_local_file(f"{REPO}/pyproject.toml", f"{LOCK_DIR}/pyproject.toml", copy=True)
    .add_local_file(f"{REPO}/uv.lock", f"{LOCK_DIR}/uv.lock", copy=True)
    .add_local_file(f"{REPO}/README.md", f"{LOCK_DIR}/README.md", copy=True)
    .add_local_file(f"{REPO}/LICENSE", f"{LOCK_DIR}/LICENSE", copy=True)
    .add_local_file(f"{REPO}/lexicons/wtf.petrichor.drinkme.measurement.json",
                    f"{LOCK_DIR}/lexicons/wtf.petrichor.drinkme.measurement.json", copy=True)
    .run_commands(
        f"mkdir -p {LOCK_DIR}/src/drinkme && touch {LOCK_DIR}/src/drinkme/__init__.py",
        # THE BARE `uv sync` OF A DISPOSABLE IMAGE BUILD (bench/modal_radix.py)
        f"cd {LOCK_DIR} && /root/.local/bin/uv sync --frozen --no-default-groups --group cuda",
        f"{VENV}/bin/python -c 'import torch, triton; print(torch.__version__, triton.__version__)'",
    )
    .add_local_dir(BEFORE_LOCAL, remote_path=BEFORE_SRC, ignore=["**/__pycache__/**", "**/*.pyc"])
    .add_local_dir(REPO, remote_path=REMOTE_REPO, ignore=MOUNT_IGNORE)
)

LOG_CAP = 2_000_000
PY = f"{VENV}/bin/python"


def run_steps(card: str | None, steps: list, timeout: int, name: str = "") -> dict:
    """bench/modal_cuda_graphs.run_steps: run `steps` [(label, argv, env)] in
    the mounted checkout, streaming and keeping the log; returns the log,
    per-step exits and every file under OUT_DIR."""
    env = dict(os.environ, PATH="/root/.local/bin:" + os.environ["PATH"], UV_PROJECT_ENVIRONMENT=VENV,
               DRINKME_HOME="/root/dmh", HF_HUB_DISABLE_PROGRESS_BARS="1",
               TOKENIZERS_PARALLELISM="false", PYTHONUNBUFFERED="1", DRINKME_NO_AUTO_DEPS="1",
               HF_HUB_OFFLINE="1")
    T0 = time.time()
    deadline = T0 + timeout - DEADLINE_MARGIN_S
    report = {"card": card, "steps": {}, "step_order": [],
              "started": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())}
    parts: list[str] = []

    def log(text: str) -> None:
        print(text, end="" if text.endswith("\n") else "\n", flush=True)
        parts.append(text if text.endswith("\n") else text + "\n")

    def run(label, cmd, extra=None):
        log(f"\n=== [{time.time() - T0:7.1f}s] {label}: {' '.join(cmd)} {json.dumps(extra or {})}\n")
        t0 = time.time()
        p = subprocess.Popen(cmd, cwd=REMOTE_REPO, env={**env, **(extra or {})}, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True, errors="replace", start_new_session=True)
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
        for line in p.stdout:
            log(line.rsplit("\r", 1)[-1])
        p.wait()
        dt = time.time() - t0
        log(f"=== [{time.time() - T0:7.1f}s] {label}: exit {p.returncode} after {dt:.1f}s"
            f"{' KILLED at the deadline' if killed['at'] else ''}\n")
        report["steps"][label] = {"exit": p.returncode, "seconds": round(dt, 1), "killed": killed["at"]}
        report["step_order"].append(label)

    os.makedirs(OUT_DIR, exist_ok=True)
    try:
        first = open("/proc/cpuinfo").read().split("\n\n", 1)[0]
        fields = {k.strip(): v.strip() for k, v in (ln.split(":", 1) for ln in first.splitlines() if ":" in ln)}
        report["host_cpu"] = {"model": fields.get("model name"), "vendor": fields.get("vendor_id"),
                              "family": fields.get("cpu family"), "model_id": fields.get("model"),
                              "nproc": os.cpu_count()}
    except (OSError, ValueError) as e:
        report["host_cpu"] = {"error": str(e)}
    log(f"host cpu: {json.dumps(report['host_cpu'])}")
    if card:
        smi = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                              "--format=csv,noheader"], capture_output=True, text=True).stdout
        report["nvidia_smi"] = smi.strip()
        log(f"nvidia-smi: {smi}")
        if SHAPES[card]["witness"] not in smi:
            report["error"] = f"not a {card}: {smi}"
            report["log"] = "".join(parts)[-LOG_CAP:]
            return report
    run("uv sync (prebuilt env)", ["uv", "sync", "--frozen", "--no-default-groups", "--group", "cuda"])
    for label, argv, extra in steps:
        run(label, argv, extra)
    files = {}
    for root, _dirs, names in os.walk(OUT_DIR):
        for n in names:
            path = os.path.join(root, n)
            if os.path.getsize(path) <= 8_000_000:
                with open(path, errors="replace") as f:
                    files[os.path.relpath(path, OUT_DIR)] = f.read()
    report["files"] = files
    report["wall_s"] = round(time.time() - T0, 1)
    report["log"] = "".join(parts)[-LOG_CAP:]
    return report


@app.function(image=image, cpu=8, memory=16 * 1024, timeout=3600, volumes=VOLUMES)
def stage_gemma(token: str) -> dict:
    """gemma-4-31B-it's snapshot at the menu's revision onto its volume
    (huggingface_hub from the image's venv)."""
    t0 = time.time()
    code = ("import json, os, sys; from huggingface_hub import snapshot_download; "
            "p = snapshot_download(%r, revision=%r, token=os.environ['HF_TOKEN'], "
            "allow_patterns=['*.json', '*.safetensors', '*.jinja', '*.model', '*.txt']); "
            "print('PATH ' + p)" % GEMMA)
    p = subprocess.run([PY, "-c", code], env=dict(os.environ, HF_HOME=f"{VOLG_PATH}/hf", HF_TOKEN=token,
                                                    HF_HUB_DISABLE_PROGRESS_BARS="1"),
                       capture_output=True, text=True)
    volg.commit()
    path = next((ln[5:] for ln in p.stdout.splitlines() if ln.startswith("PATH ")), None)
    size = (sum(os.path.getsize(os.path.join(r, n)) for r, _d, ns in os.walk(path) for n in ns)
            if path else None)
    return {"path": path, "bytes": size, "seconds": round(time.time() - t0, 1), "exit": p.returncode,
            "stderr": p.stderr[-3000:], "files": sorted(os.listdir(path)) if path else None}


def _gpu(card: str):
    s = SHAPES[card]
    return app.function(image=image, gpu=s["gpu"], cpu=s["cpu"], memory=s["memory"], timeout=s["timeout"],
                        volumes=VOLUMES)


@_gpu("L4")
def gpu_L4(steps: list, name: str = "") -> dict:
    return run_steps("L4", steps, SHAPES["L4"]["timeout"], name)


@_gpu("H100")
def gpu_H100(steps: list, name: str = "") -> dict:
    return run_steps("H100", steps, SHAPES["H100"]["timeout"], name)


def steps_for(model: str, names: list) -> list:
    hf, pd, ctx = MODELS[model]
    base = {"HF_HOME": hf}
    tool = [PY, "bench/serve_fixes.py", "--model", model]
    pack = ["--pack-dir", pd] if pd else ["--stock"]
    out = []
    for step in names:
        code = "before" if "before" in step else "after"
        src = {"DRINKME_SRC": BEFORE_SRC if code == "before" else AFTER_SRC}
        if step.startswith("ttft"):
            out.append((step, [*tool, "--what", "ttft", *pack, "--ctx", "32768", "--out", f"{OUT_DIR}/{step}"],
                        {**base, **src, "DRINKME_CUDA_GRAPHS": "0"}))
        elif step.startswith("identity"):
            for arm in ("stock", "compressed"):
                out.append((f"{step} {arm}", [*tool, "--what", "identity", "--arm", arm, *(pack if pd else []),
                                              "--ctx", str(ctx), "--out", f"{OUT_DIR}/{step}",
                                              "--json", f"{OUT_DIR}/{step}/{arm}.json"],
                            {**base, **src, "DRINKME_CUDA_GRAPHS": "0"}))
            out.append((f"{step} compare", [*tool, "--compare", f"{OUT_DIR}/{step}/stock.json",
                                            f"{OUT_DIR}/{step}/compressed.json", "--verdict",
                                            f"compressed==stock ({code})"], base))
        elif step.startswith("serve"):
            # -eager: DRINKME_CUDA_GRAPHS=0; -cudnn: DRINKME_CUDNN_SDPA=1 (torch's
            # backend choice under the after tree, so the MTP head fix is read apart from the prefill one)
            knobs = {**({"DRINKME_CUDA_GRAPHS": "0"} if "-eager" in step else {}),
                     **({"DRINKME_CUDNN_SDPA": "1"} if "-cudnn" in step else {})}
            out.append((step, [*tool, "--what", "serve", *pack, "--ctx", str(ctx), "--label", step,
                               "--out", f"{OUT_DIR}/{step}", "--json", f"{OUT_DIR}/{step}/serve.json"],
                        {**base, **src, **knobs}))
        else:
            raise SystemExit(f"unknown step {step!r}")
    # before/after comparisons of whatever pairs this container ran
    served = [n for n in names if n.startswith("serve")]
    base_served = next((n for n in served if "before" in n), None)
    for b in served:
        if base_served and b != base_served:
            out.append((f"compare {base_served} {b}", [*tool, "--compare-served",
                                                        f"{OUT_DIR}/{base_served}/serve.json",
                                                        f"{OUT_DIR}/{b}/serve.json"], base))
    for a in names:
        if "before" in a:
            b = a.replace("before", "after")
            if b in names and a.startswith("identity"):
                for arm in ("stock", "compressed"):
                    out.append((f"compare {a} {b} {arm}", [*tool, "--compare", f"{OUT_DIR}/{a}/{arm}.json",
                                                          f"{OUT_DIR}/{b}/{arm}.json", "--verdict",
                                                          f"after==before ({arm})"], base))
    return out


def local_token() -> str:
    token = os.environ.get("HF_TOKEN", "")
    tp = os.path.expanduser("~/.cache/huggingface/token")
    if not token and os.path.exists(tp):
        token = open(tp).read().strip()
    return token


@app.local_entrypoint()
def main(what: str = "run", card: str = "H100", model: str = "Qwen3-8B", steps: str = "",
         out_dir: str = "", tag: str = ""):
    stamp = time.strftime("%Y%m%dT%H%M%S", time.gmtime())
    out_dir = out_dir or os.path.join(REPO, "verification", "serve-fixes")
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()
    if what == "stage-gemma":
        rep = stage_gemma.remote(local_token())
        print(f"[serve-fixes] gemma staged: {json.dumps(rep)}", flush=True)
        with open(os.path.join(out_dir, f"stage-gemma_{stamp}.json"), "w") as f:
            json.dump(rep, f, indent=1)
        return
    fn = {"L4": gpu_L4, "H100": gpu_H100}[card]
    name = f"{card}_{model}_{tag or stamp}"
    rep = fn.remote(steps_for(model, steps.split(",")), name)
    rep["local_wall_s"] = round(time.time() - t0, 1)
    rep["app_id"] = app.app_id
    files = rep.pop("files", {}) or {}
    log = rep.pop("log", "")
    with open(os.path.join(out_dir, name + ".log"), "w") as f:
        f.write(log)
    for rel, text in files.items():
        with open(os.path.join(out_dir, f"{name}__{rel.replace('/', '__')}"), "w") as f:
            f.write(text)
    with open(os.path.join(out_dir, name + ".meta.json"), "w") as f:
        json.dump({**rep, "files": sorted(files)}, f, indent=1)
    print(f"[serve-fixes] {name}: {json.dumps(rep.get('steps'))} wall {rep.get('wall_s')} s "
          f"(local {rep['local_wall_s']} s) -> {out_dir}", flush=True)
