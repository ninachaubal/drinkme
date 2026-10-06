"""CUDA graphs for the decode step (serving/cudagraph.py) on Modal: the
correctness bar per (family, mode), `drinkme bench` eager against graph,
and a served request over HTTP, on an L4 and an H100.

The image is bench/modal_nvidia_speed.py's (the cuda environment resolved
from the lock at image build, the checkout mounted minus what git does not
track). Two volumes: `drinkme-nvidia-speed` holds the Qwen3-8B snapshot
and the verified packs, one per (model, compression profile), at
`/vol/packs/<model>-<profile>`; `drinkme-radix-27b` holds the 27B's HF
snapshot at the menu's revision, mounted as that model's HF_HOME.
`--profile` (sip | gulp, default sip) picks the pack every step reads.

    uv tool run modal run bench/modal_cuda_graphs.py --what prep --model Qwen3.8-27B --profile gulp
        CPU only: `drinkme pack --<profile>` of the model onto the volume
        (skipped when the pack's meta.json is there), then `drinkme verify`
    uv tool run modal run bench/modal_cuda_graphs.py --what gate --card L4 --out-dir DIR
        bench/cuda_graph_gate.py per arm (graph == eager static step,
        compressed vs stock tokens, the step at 4k/16k live)
    uv tool run modal run bench/modal_cuda_graphs.py --what bench --card L4 --out-dir DIR
        `drinkme bench --pack-dir` under DRINKME_CUDA_GRAPHS=0, then the
        default, in one container
    uv tool run modal run bench/modal_cuda_graphs.py --what serve --card L4 --out-dir DIR
        `drinkme serve` over the pack, one streamed OpenAI chat request,
        eager then graph, the texts compared

    uv tool run modal run bench/modal_cuda_graphs.py --what servegap --card H100 \
            --model Qwen3.8-27B --out-dir DIR
        bench/serve_gap.py: serve's engine over HTTP and in-process, its
        configuration taken apart, host timers and a profile
    uv tool run modal run bench/modal_cuda_graphs.py --what codec --card L4 --profile gulp --out-dir DIR
        the lean decoder's codec gates on the pack: bench/radix_lean_gate.py
        and its selftest, bench/radix_twin_bitpin.py over the profile's
        fixtures, synthetic widths and the pack's real tensors (Qwen3-8B),
        and its selftest
    uv tool run modal run bench/modal_cuda_graphs.py --what abbench --card L4 --profile gulp \
            --arms before,after,before,after --out-dir DIR
        `drinkme bench --pack-dir --no-spec` in graph mode (the default)
        once per --arms entry, in order, in one container: `before` is the
        scheduled decoder at the scheduled row's launch (gulp on CUDA before
        the lean gulp decoder), `after` the default, `scheduled` and `lean`
        the DRINKME_RADIX_DECODER values
    uv tool run modal run bench/modal_cuda_graphs.py --what sweep --card L4 --profile gulp \
            --sweep "dec=scheduled dec=lean t1/w1 trow/w4" --out-dir DIR
        bench/decode_step_profile.py's graph-mode sweep on the compressed arm

`--what` takes several steps (comma-separated) in one container, on
`--card` L4 | L40S | H100; `--extra` passes further arguments to the gate,
`--sextra` to cuda_graph_serve.py and serve_gap.py; `--bench-modes` picks
bench's eager and/or graph run; `--env` a JSON object of extra environment. Every step is killed 150 s before the container timeout
so the function returns receipts. Verdicts come from the printed lines,
never exit codes.
"""

import json
import os
import signal
import subprocess
import threading
import time

import modal

APP_NAME = "drinkme-gulp-graphs"
app = modal.App(APP_NAME)

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
REMOTE_REPO = "/root/drinkme"
VENV = "/opt/venv"
LOCK_DIR = "/opt/drinkme-lock"
OUT_DIR = "/root/out"
VOL_PATH = "/vol"
VOL27_PATH = "/vol27"
DEADLINE_MARGIN_S = 150
vol = modal.Volume.from_name("drinkme-nvidia-speed", create_if_missing=False)
vol27 = modal.Volume.from_name("drinkme-radix-27b", create_if_missing=False)
VOLUMES = {VOL_PATH: vol, VOL27_PATH: vol27}

SHAPES = {
    "L4": {"gpu": "L4", "memory": 32 * 1024, "cpu": 8, "timeout": 3600, "witness": "L4"},
    # "H100!": exactly an H100 (Modal may otherwise hand out an H200, which the witness refuses)
    "H100": {"gpu": "H100!", "memory": 96 * 1024, "cpu": 16, "timeout": 3600, "witness": "H100"},
    "L40S": {"gpu": "L40S", "memory": 80 * 1024, "cpu": 8, "timeout": 3600, "witness": "L40S"},
}

MODELS = {
    # name: (HF_HOME, pack dir stem (the profile appended), repo, revision)
    "Qwen3-8B": (f"{VOL_PATH}/hf", f"{VOL_PATH}/packs/Qwen3-8B", "Qwen/Qwen3-8B",
                 "b968826d9c46dd6066d109eabc6255188de91218"),
    "Qwen3.8-27B": (f"{VOL27_PATH}/hf", f"{VOL_PATH}/packs/Qwen3.8-27B", "Qwen/Qwen3.8-27B",
                    "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"),
    # no pack on a volume: downloaded into the container, the compressed arm packed in memory
    "MiMo-V2.6-Distill-Qwen-9B": ("/root/hf", None, "XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B", None),
    "Qwen3-4B": ("/root/hf", None, "Qwen/Qwen3-4B", None),
}
# `drinkme pack`'s compression profiles
PROFILES = ("sip", "gulp")
# the checkpoint a pack on the volume was cut from (bench/parity_common.py's
# DRINKME_PARITY_SNAPSHOT): the twin gate's real tensors read both
SNAPSHOTS = {"Qwen3-8B": f"{VOL_PATH}/hf/hub/models--Qwen--Qwen3-8B/snapshots/{MODELS['Qwen3-8B'][3]}"}
# abbench's arms: the environment each `drinkme bench` runs under
AB_ARMS = {
    "before": {"DRINKME_RADIX_DECODER": "scheduled",
               "DRINKME_RADIX_SCHEDULE": "tiles=1,warps=1,mc_tiles=1,mc_warps=1"},
    "after": {},
    "scheduled": {"DRINKME_RADIX_DECODER": "scheduled"},
    "lean": {"DRINKME_RADIX_DECODER": "lean"},
}


def pack_dir(model: str, profile: str) -> str | None:
    """The model's pack at `profile` on the volume, or None (packed in memory)."""
    if profile not in PROFILES:
        raise SystemExit(f"--profile {' | '.join(PROFILES)}, not {profile!r}")
    stem = MODELS[model][1]
    return f"{stem}-{profile}" if stem else None


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
] + (_untracked(REPO) if modal.is_local() else [])

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
    .add_local_dir(REPO, remote_path=REMOTE_REPO, ignore=MOUNT_IGNORE)
)

LOG_CAP = 2_000_000
PY = f"{VENV}/bin/python"


def run_steps(card: str | None, steps: list, timeout: int, name: str = "") -> dict:
    """Run `steps` [(label, argv, extra_env)] in the mounted checkout,
    streaming and keeping the log; returns the log, per-step exits, the
    host's CPU model and every file under OUT_DIR."""
    env = dict(os.environ, PATH="/root/.local/bin:" + os.environ["PATH"], UV_PROJECT_ENVIRONMENT=VENV,
               HF_HOME=f"{VOL_PATH}/hf", DRINKME_HOME="/root/dmh", HF_HUB_DISABLE_PROGRESS_BARS="1",
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
        return p.returncode

    os.makedirs(OUT_DIR, exist_ok=True)
    try:
        text = open("/proc/cpuinfo").read()
        first = text.split("\n\n", 1)[0]
        fields = dict(ln.split(":", 1) for ln in first.splitlines() if ":" in ln)
        fields = {k.strip(): v.strip() for k, v in fields.items()}
        report["host_cpu"] = {"model": fields.get("model name"), "vendor": fields.get("vendor_id"),
                              "family": fields.get("cpu family"), "model_id": fields.get("model"),
                              "stepping": fields.get("stepping"), "mhz": fields.get("cpu MHz"),
                              "logical": text.count("processor\t"), "nproc": os.cpu_count(),
                              "affinity": len(os.sched_getaffinity(0))}
        report["cpuinfo_first"] = first[:3000]
        lscpu = subprocess.run(["lscpu"], capture_output=True, text=True).stdout
        report["lscpu"] = lscpu[:4000]
    except (OSError, ValueError) as e:
        report["host_cpu"] = {"error": str(e)}
    log(f"host cpu: {json.dumps(report['host_cpu'])}")
    if card:
        smi = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version,clocks.max.sm,power.limit",
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
    if name:
        # the receipts on the volume too, so a detached run is collected with
        # `modal volume get drinkme-nvidia-speed cg-out/<name>`
        dst = f"{VOL_PATH}/cg-out/{name}"
        os.makedirs(dst, exist_ok=True)
        for rel, text in files.items():
            with open(os.path.join(dst, rel.replace("/", "__")), "w") as f:
                f.write(text)
        with open(os.path.join(dst, "run.log"), "w") as f:
            f.write(report["log"])
        with open(os.path.join(dst, "meta.json"), "w") as f:
            json.dump({k: v for k, v in report.items() if k not in ("files", "log")}, f, indent=1)
        vol.commit()
    return report


@app.function(image=image, cpu=32, memory=160 * 1024, timeout=3600, volumes=VOLUMES)
def prep(model: str, profile: str) -> dict:
    hf = MODELS[model][0]
    pd = pack_dir(model, profile)
    os.makedirs(os.path.dirname(pd), exist_ok=True)
    steps = []
    if not os.path.exists(os.path.join(pd, "meta.json")):
        steps.append((f"drinkme pack --{profile}", ["uv", "run", "--no-sync", "drinkme", "pack", "--model", model,
                                                    f"--{profile}", "-o", pd, "--replace"],
                      {"OMP_NUM_THREADS": "32", "HF_HOME": hf}))
    steps.append(("drinkme verify", ["uv", "run", "--no-sync", "drinkme", "verify", "--pack-dir", pd], {"HF_HOME": hf}))
    rep = run_steps(None, steps, 3600)
    vol.commit()
    rep["pack_dir"] = pd
    meta = os.path.join(pd, "meta.json")
    if os.path.exists(meta):
        with open(meta) as f:
            m = json.load(f)
        rep["pack"] = {k: m.get(k) for k in ("formatVersion", "profile", "profileWidths", "tensorCount", "meanBpw",
                                             "weightedBpw", "packSeconds", "packedBytes", "residentBytes",
                                             "radixTensorCount", "rawFallbackTensorCount", "rawTensorCount",
                                             "manifestSha256", "mtpTensorCount", "mtpMeanBpw")}
    rep["du"] = subprocess.run(["du", "-sb", pd], capture_output=True, text=True).stdout.strip()
    return rep


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


@_gpu("L40S")
def gpu_L40S(steps: list, name: str = "") -> dict:
    return run_steps("L40S", steps, SHAPES["L40S"]["timeout"], name)


def model_env(model: str) -> dict:
    hf, stem = MODELS[model][:2]
    return {"HF_HOME": hf, **({} if stem else {"HF_HUB_OFFLINE": "0"})}


def pack_args(model: str, profile: str) -> list:
    pd = pack_dir(model, profile)
    return ["--pack-dir", pd] if pd else []


def gate_steps(model: str, extra: list, env: dict, profile: str = "sip") -> list:
    """bench/cuda_graph_gate.py: stock then compressed, one process each,
    then the token comparison across the two JSONs."""
    e = {**model_env(model), **env}
    arm_list = ("compressed",) if "--only-warm" in extra else ("stock", "compressed")
    steps = [(f"cuda_graph_gate {arm}", [PY, "bench/cuda_graph_gate.py", "--model", model, "--arm", arm,
                                         *pack_args(model, profile), "--json", f"{OUT_DIR}/gate_{arm}.json",
                                         *extra], e)
             for arm in arm_list]
    if "--only-warm" in extra:
        return steps
    steps.append(("cuda_graph_gate compare", [PY, "bench/cuda_graph_gate.py", "--compare",
                                              f"{OUT_DIR}/gate_stock.json", f"{OUT_DIR}/gate_compressed.json"], e))
    return steps


def bench_steps(model: str, env: dict, spec: bool = False, tag: str = "",
                modes: tuple = ("eager", "graph"), profile: str = "sip") -> list:
    """`drinkme bench --pack-dir` eager (DRINKME_CUDA_GRAPHS=0), then the
    default (graph mode where verified), in one container."""
    pd = pack_dir(model, profile)
    flags = [] if spec else ["--no-spec"]
    return [(f"drinkme bench ({mode}{tag})", ["uv", "run", "--no-sync", "drinkme", "bench", "--model", model,
                                              "--pack-dir", pd, *flags, "-o", f"{OUT_DIR}/bench_{mode}{tag}.json"],
             {**model_env(model), "DRINKME_BENCH_WARMUP": "1", **cg, **env})
            for mode, cg in (("eager", {"DRINKME_CUDA_GRAPHS": "0"}), ("graph", {})) if mode in modes]


def serve_steps(model: str, env: dict, extra: list, tag: str = "", profile: str = "sip") -> list:
    """bench/cuda_graph_serve.py: `drinkme serve` over the pack on a spare
    port, one streamed chat request under DRINKME_CUDA_GRAPHS=0 and then
    the default, the texts compared."""
    pd = pack_dir(model, profile)
    return [(f"serve e2e{tag}", [PY, "bench/cuda_graph_serve.py", "--model", model, "--pack-dir", pd,
                                 "--json", f"{OUT_DIR}/serve{tag}/serve.json", *extra], {**model_env(model), **env})]


def gap_steps(model: str, env: dict, extra: list, profile: str = "sip") -> list:
    """bench/serve_gap.py over the pack."""
    pd = pack_dir(model, profile)
    return [("serve gap", [PY, "bench/serve_gap.py", "--model", model, "--pack-dir", pd,
                           "--json", f"{OUT_DIR}/serve_gap.json", *extra], {**model_env(model), **env})]


def codec_steps(model: str, env: dict, profile: str = "gulp") -> list:
    """The lean decoder's codec gates over the model's pack at `profile`
    (docs/checks.md): bench/radix_lean_gate.py on every radix tensor (dense,
    M=1 and mc against the scheduled decoder bitwise at the tensor's launch,
    the checkpoint's own bits, per-class timing) after its selftest (must
    FAIL), then bench/radix_twin_bitpin.py over the profile's fixtures,
    synthetic widths and the pack's real tensors (layer 0 + lm_head against
    the checkpoint), and its selftest."""
    pd = pack_dir(model, profile)
    g = f"{OUT_DIR}/gates"
    e = {**model_env(model), **env}
    parity = {f"DRINKME_PARITY_PACK_{profile.upper()}": pd, "DRINKME_PARITY_SNAPSHOT": SNAPSHOTS[model]}
    twin = [PY, "bench/radix_twin_bitpin.py", "--profiles", profile, "--real-arms", profile]
    return [
        ("mkdir gates", ["mkdir", "-p", g], {}),
        ("lean gate selftest", [PY, "bench/radix_lean_gate.py", "--model", model, "--pack-dir", pd, "--limit", "2",
                                "--no-timing", "--selftest", "--json", f"{g}/radix_lean_gate_selftest.json"], e),
        ("lean gate", [PY, "bench/radix_lean_gate.py", "--model", model, "--pack-dir", pd, "--seconds", "2.5",
                       "--json", f"{g}/radix_lean_gate.json"], e),
        (f"gate radix_twin_bitpin real {profile}", [*twin, "--json", f"{g}/radix_twin_bitpin_real_{profile}.json"],
         {**e, **parity}),
        (f"gate radix_twin_bitpin real {profile} selftest",
         [*twin, "--fixtures-only", "--selftest", "--json", f"{g}/radix_twin_bitpin_selftest.json"], {**e, **parity})]


def sweep_steps(model: str, env: dict, specs: list, profile: str = "gulp") -> list:
    """bench/decode_step_profile.py's graph-mode sweep on the compressed arm:
    the step captured once per spec (`dec=<scheduled|lean>`, or a launch
    `t<tiles|row>/w<warps>` for every radix Linear under the process's
    decoder) and replayed, ms/step with the clocks sampled."""
    pd = pack_dir(model, profile)
    return [("graph sweep", [PY, "bench/decode_step_profile.py", "--model", model, "--arm", "compressed",
                             "--pack-dir", pd, "--json", f"{OUT_DIR}/sweep.json", "--only-sweep", "--sweep", *specs],
             {**model_env(model), **env})]


MERGED_HF = "/root/hfm"


def abbench_steps(model: str, env: dict, arms: list, profile: str = "gulp") -> list:
    """`drinkme bench --pack-dir --no-spec` in the default (graph) mode once
    per arm of `arms` (AB_ARMS' names), in order, in one container. A model
    whose snapshot is not on the 8B's volume benches over a merged HF cache
    (symlinks to both volumes' snapshots): bench's menu needs a ratio model
    it can size from the local cache before it runs a fit point, and on an
    L40S the 27B is the fit point and the 8B the ratio one."""
    pd = pack_dir(model, profile)
    e = model_env(model)
    steps = []
    if e["HF_HOME"] != f"{VOL_PATH}/hf":
        steps.append(("merged HF cache", ["sh", "-c", f"mkdir -p {MERGED_HF}/hub && ln -sf {VOL_PATH}/hf/hub/models--* "
                                                      f"{e['HF_HOME']}/hub/models--* {MERGED_HF}/hub/ && ls -l {MERGED_HF}/hub"],
                      {}))
        e = {**e, "HF_HOME": MERGED_HF}
    return steps + [(f"drinkme bench ({arm} #{i})", ["uv", "run", "--no-sync", "drinkme", "bench", "--model", model,
                                                     "--pack-dir", pd, "--no-spec", "-o",
                                                     f"{OUT_DIR}/bench_{arm}_{i}.json"],
                     {**e, "DRINKME_BENCH_WARMUP": "1", **AB_ARMS[arm], **env})
                    for i, arm in enumerate(arms)]


@app.local_entrypoint()
def main(what: str = "gate", card: str = "L4", model: str = "Qwen3-8B", out_dir: str = "",
         extra: str = "", env: str = "", tag: str = "", spec: bool = False, sextra: str = "",
         bench_modes: str = "eager,graph", profile: str = "sip", arms: str = "before,after,before,after",
         sweep: str = ""):
    stamp = time.strftime("%Y%m%dT%H%M%S", time.gmtime())
    out_dir = out_dir or os.path.join(REPO, "verification", "cuda-graphs")
    os.makedirs(out_dir, exist_ok=True)
    extra_argv = extra.split() if extra else []
    sextra_argv = sextra.split() if sextra else []
    extra_env = json.loads(env) if env else {}
    t0 = time.time()
    pack_dir(model, profile)  # refuses an unknown profile before anything runs
    if what == "prep":
        rep = prep.remote(model, profile)
    else:
        steps = []
        for w in what.split(","):
            if w == "gate":
                steps += gate_steps(model, [a for a in extra_argv if not a.startswith("--spec")], extra_env,
                                    profile)
            elif w == "bench":
                steps += bench_steps(model, extra_env, spec, modes=tuple(bench_modes.split(",")), profile=profile)
            elif w == "serve":
                steps += serve_steps(model, extra_env,
                                     [a for a in extra_argv if a.startswith("--spec")] + sextra_argv,
                                     profile=profile)
            elif w == "servegap":
                steps += gap_steps(model, extra_env, sextra_argv, profile)
            elif w == "codec":
                steps += codec_steps(model, extra_env, profile)
            elif w == "abbench":
                steps += abbench_steps(model, extra_env, arms.split(","), profile)
            elif w == "sweep":
                steps += sweep_steps(model, extra_env, sweep.split(), profile)
            else:
                raise SystemExit(f"--what prep|gate|bench|serve|servegap|codec|abbench|sweep (comma-separated), "
                                 f"not {w!r}")
        fn = {"L4": gpu_L4, "H100": gpu_H100, "L40S": gpu_L40S}[card]
        rep = fn.remote(steps, f"{what.replace(',', '+')}_{card}_{model}-{profile}_{tag or stamp}")
    rep["local_wall_s"] = round(time.time() - t0, 1)
    rep["app_id"] = app.app_id
    name = (f"{what.replace(',', '+')}_{card if what != 'prep' else 'cpu'}_{model}-{profile}"
            f"{'_' + tag if tag else ''}_{stamp}")
    files = rep.pop("files", {}) or {}
    log = rep.pop("log", "")
    with open(os.path.join(out_dir, name + ".log"), "w") as f:
        f.write(log)
    for rel, text in files.items():
        with open(os.path.join(out_dir, f"{name}__{rel.replace('/', '__')}"), "w") as f:
            f.write(text)
    with open(os.path.join(out_dir, name + ".meta.json"), "w") as f:
        json.dump({**rep, "files": sorted(files)}, f, indent=1)
    print(f"[cuda-graphs] {name}: {json.dumps(rep.get('steps'))} host {json.dumps(rep.get('host_cpu'))} "
          f"wall {rep.get('wall_s')} s (local {rep['local_wall_s']} s) -> {out_dir}", flush=True)
