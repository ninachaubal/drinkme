"""The NVIDIA decode-speed spike on Modal: bench/decode_step_profile.py
(where a decode step goes; eager vs CUDA graph; the graph-mode radix
launch-config and decoder sweeps) and bench/radix_lean_gate.py (the lean
sip decoder's pack gate and per-class timing), on an L4 and one H100.

The image is bench/modal_radix.py's (the cuda environment resolved from
the lock at image build into /opt/venv, g++ for the radix native encoder,
the checkout mounted minus everything git does not track). Unlike the
publishing sweep, the checkpoint and the pack live on a Volume
(`drinkme-nvidia-speed`, HF_HOME and the pack directory) so each GPU
container starts at the load, not at a 16 GB download and a CPU pack:

    uv tool run modal run bench/modal_nvidia_speed.py --what prep
        CPU only: `drinkme pack --model Qwen3-8B --sip` onto the volume
        (the checkpoint downloads into the volume's HF cache), then
        `drinkme verify` on it
    uv tool run modal run bench/modal_nvidia_speed.py --what profile --card L4 --out-dir DIR
        each arm (--arms stock,compressed[,twin]) through
        decode_step_profile.py, one process each; `--extra "--sweep
        trow/w2 t1/w4 ..."` adds the graph-mode launch-config sweep
    uv tool run modal run bench/modal_nvidia_speed.py --what lean --card L4 --out-dir DIR
        the lean decoder's pack gate, the repo's radix device gates (each
        plus --selftest), the graph-mode decoder A/B and the eager bench
        loop under each decoder; `--what lean2` is the shorter second
        round (the twin gate's fixtures under each decoder, the gemv and
        pack gates, the eager A/B alternating twice); `--what bench` is
        `drinkme bench --pack-dir --no-spec` under each decoder; `--what
        checks` the lean gate's selftest and the twin gate on the real sip
        tensors
    uv tool run modal run bench/modal_nvidia_speed.py --what v1checks --card L4 --out-dir DIR
        docs/checks.md's serve-level kernel checks under the default and
        the scheduled decoder: logits_row_identity.py, greedy_ab_smoke.sh
        and the hotloop acceptance pairs
    uv tool run modal run bench/modal_nvidia_speed.py --what abdec --card L4 --out-dir DIR
        the decoder A/B (--decoders, DRINKME_RADIX_DECODER values,
        scheduled,lean by default): the eager bench loop and the
        graph-mode step each alternating them twice, then `drinkme bench`
        under each (`--extra --no-bench` skips it)
    uv tool run modal run bench/modal_nvidia_speed.py --what v2gates --card L4 --out-dir DIR
        the default lean decoder through the codec gates: the lean pack
        gate (+ selftest), radix_gemv_bitpin.py, radix_twin_bitpin.py
        (synthetic, real sip, selftest) and radix_mc_bitpin.py on the pack
    uv tool run modal run bench/modal_nvidia_speed.py --what v2meas --card L4 --out-dir DIR
        abdec over scheduled and lean

`--extra` passes further script arguments; `--env` a JSON object of extra
environment. Every step is killed 150 s before the container timeout so the
function returns receipts; the local side writes them only when the
function returns, so a `modal run` killed locally leaves only its streamed
output. Verdicts from the printed lines, never exit codes.
"""

import json
import os
import signal
import subprocess
import threading
import time

import modal

APP_NAME = "drinkme-nvidia-speed"
app = modal.App(APP_NAME)

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
REMOTE_REPO = "/root/drinkme"
VENV = "/opt/venv"
LOCK_DIR = "/opt/drinkme-lock"
OUT_DIR = "/root/out"
VOL_NAME = "drinkme-nvidia-speed"
VOL_PATH = "/vol"
HF_HOME = f"{VOL_PATH}/hf"
DRINKME_HOME = f"{VOL_PATH}/drinkme-home"
DEADLINE_MARGIN_S = 150
vol = modal.Volume.from_name(VOL_NAME, create_if_missing=True)

SHAPES = {
    "L4": {"gpu": "L4", "memory": 32 * 1024, "cpu": 8, "timeout": 3600, "witness": "L4"},
    "H100": {"gpu": "H100", "memory": 64 * 1024, "cpu": 16, "timeout": 3600, "witness": "H100"},
}
for _s in SHAPES.values():
    assert _s["timeout"] <= 3600


def _untracked(repo: str) -> list[str]:
    """bench/modal_radix.py's: mount-ignore globs for what git does not track."""
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


def pack_dir_for(model: str) -> str:
    return f"{VOL_PATH}/packs/{model}-sip"


def run_steps(card: str | None, steps: list, timeout: int) -> dict:
    """Run `steps` [(label, argv, extra_env)] in the mounted checkout,
    streaming and keeping the log; returns the log, per-step exits and
    every file under OUT_DIR."""
    env = dict(os.environ, PATH="/root/.local/bin:" + os.environ["PATH"], UV_PROJECT_ENVIRONMENT=VENV,
               HF_HOME=HF_HOME, DRINKME_HOME=DRINKME_HOME, HF_HUB_DISABLE_PROGRESS_BARS="1",
               TOKENIZERS_PARALLELISM="false", PYTHONUNBUFFERED="1", DRINKME_NO_AUTO_DEPS="1")
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

    os.makedirs(f"{OUT_DIR}/gates", exist_ok=True)
    if card:
        smi = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version,clocks.max.sm,power.limit,power.max_limit,"
                              "clocks.max.mem", "--format=csv,noheader"], capture_output=True, text=True).stdout
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


@app.function(image=image, cpu=16, memory=64 * 1024, timeout=3600, volumes={VOL_PATH: vol})
def prep(model: str, hf_token: str = "") -> dict:
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
    pd = pack_dir_for(model)
    os.makedirs(HF_HOME, exist_ok=True)
    os.makedirs(os.path.dirname(pd), exist_ok=True)
    steps = []
    if not os.path.exists(os.path.join(pd, "meta.json")):
        steps.append(("drinkme pack --sip", ["uv", "run", "--no-sync", "drinkme", "pack", "--model", model, "--sip",
                                             "-o", pd, "--replace"], {"OMP_NUM_THREADS": "16"}))
    steps.append(("drinkme verify", ["uv", "run", "--no-sync", "drinkme", "verify", "--pack-dir", pd], None))
    rep = run_steps(None, steps, 3600)
    vol.commit()
    rep["pack_dir"] = pd
    return rep


def _gpu(card: str):
    s = SHAPES[card]
    return app.function(image=image, gpu=s["gpu"], cpu=s["cpu"], memory=s["memory"], timeout=s["timeout"],
                        volumes={VOL_PATH: vol})


@_gpu("L4")
def gpu_L4(steps: list) -> dict:
    return run_steps("L4", steps, SHAPES["L4"]["timeout"])


@_gpu("H100")
def gpu_H100(steps: list) -> dict:
    return run_steps("H100", steps, SHAPES["H100"]["timeout"])


PY = f"{VENV}/bin/python"


def profile_steps(model: str, arms: list, extra: list, env: dict) -> list:
    pd = pack_dir_for(model)
    return [(f"decode_step_profile {arm}", [PY, "bench/decode_step_profile.py", "--model", model, "--arm", arm,
                                            "--pack-dir", pd, "--json", f"{OUT_DIR}/profile_{arm}.json", *extra], env)
            for arm in arms]


def lean_steps(model: str) -> list:
    """The lean sip decoder on this card: its pack gate (selftest first,
    must FAIL), the repo's device gates with the lean decoder selected
    (the CUDA default) each plus --selftest, the mc gate on the pack, then
    end to end: the graph-mode step alternating the decoders, and the bench
    loop eager under each."""
    pd = pack_dir_for(model)
    g = f"{OUT_DIR}/gates"
    steps = [("lean gate selftest", [PY, "bench/radix_lean_gate.py", "--model", model, "--pack-dir", pd, "--limit", "2",
                                     "--no-timing", "--selftest", "--json", f"{g}/radix_lean_gate_selftest.json"], {}),
             ("lean gate", [PY, "bench/radix_lean_gate.py", "--model", model, "--pack-dir", pd,
                            "--json", f"{g}/radix_lean_gate.json"], {})]
    for script, flags in (("radix_gemv_bitpin", []), ("radix_gemv_bitpin", ["--contiguous-bias"]),
                          ("radix_mc_bitpin", []), ("radix_schedule_bitpin", ["--no-real"]),
                          ("radix_twin_bitpin", ["--no-real"])):
        for selftest in (False, True):
            name = script + ("_contiguous-bias" if "--contiguous-bias" in flags else "") + ("_selftest" if selftest else "")
            steps.append((f"gate {name}", [PY, f"bench/{script}.py", "--json", f"{g}/{name}.json", *flags,
                                           *(["--selftest"] if selftest else [])], {}))
    steps.append(("gate radix_mc_bitpin --pack-dir", [PY, "bench/radix_mc_bitpin.py", "--json",
                                                      f"{g}/radix_mc_bitpin_pack.json", "--pack-dir", pd], {}))
    steps.append(("graph sweep decoders", [PY, "bench/decode_step_profile.py", "--model", model, "--arm", "compressed",
                                           "--pack-dir", pd, "--json", f"{OUT_DIR}/sweep_decoders.json", "--only-sweep",
                                           "--sweep", "dec=scheduled", "dec=lean", "dec=scheduled", "dec=lean"], {}))
    for mode in ("scheduled", "lean"):
        steps.append((f"eager {mode}", [PY, "bench/decode_step_profile.py", "--model", model, "--arm", "compressed",
                                        "--pack-dir", pd, "--json", f"{OUT_DIR}/profile_compressed_{mode}.json",
                                        "--no-naive"], {"DRINKME_RADIX_DECODER": mode}))
    return steps


def lean2_steps(model: str) -> list:
    """After the whole-block restriction: the twin gate's fixtures under the
    scheduled decoder (the baseline) and the default (lean where it
    applies), each selftested; the gemv gate and the lean pack gate again;
    the bench loop eager under each decoder, alternating."""
    pd = pack_dir_for(model)
    g = f"{OUT_DIR}/gates"
    steps = []
    for tag, env in (("scheduled", {"DRINKME_RADIX_DECODER": "scheduled"}), ("default", {})):
        steps.append((f"gate radix_twin_bitpin fixtures ({tag})",
                      [PY, "bench/radix_twin_bitpin.py", "--no-real", "--fixtures-only", "--json",
                       f"{g}/radix_twin_bitpin_fixtures_{tag}.json"], env))
    steps.append(("gate radix_twin_bitpin fixtures selftest (default)",
                  [PY, "bench/radix_twin_bitpin.py", "--no-real", "--fixtures-only", "--selftest", "--json",
                   f"{g}/radix_twin_bitpin_fixtures_default_selftest.json"], {}))
    steps.append(("gate radix_gemv_bitpin (default)", [PY, "bench/radix_gemv_bitpin.py", "--json",
                                                       f"{g}/radix_gemv_bitpin.json"], {}))
    steps.append(("lean gate", [PY, "bench/radix_lean_gate.py", "--model", model, "--pack-dir", pd, "--seconds", "4",
                                "--json", f"{g}/radix_lean_gate.json"], {}))
    for i, mode in enumerate(("scheduled", "lean", "scheduled", "lean")):
        steps.append((f"eager {mode} #{i}", [PY, "bench/decode_step_profile.py", "--model", model, "--arm", "compressed",
                                             "--pack-dir", pd, "--json", f"{OUT_DIR}/eager_{mode}_{i}.json",
                                             "--no-graph", "--no-naive", "--profile-steps", "10", "--attr-steps", "3"],
                      {"DRINKME_RADIX_DECODER": mode}))
    return steps


def bench_steps(model: str) -> list:
    """`drinkme bench --pack-dir --no-spec` twice in one container: the
    scheduled decoder, then the default (lean on whole-block sip rows) —
    bench's own record of the kernel change, the same host for both."""
    pd = pack_dir_for(model)
    return [(f"drinkme bench ({tag})", ["uv", "run", "--no-sync", "drinkme", "bench", "--model", model,
                                         "--pack-dir", pd, "--no-spec", "-o", f"{OUT_DIR}/bench_{tag}.json"],
             {"DRINKME_BENCH_WARMUP": "1", **env})
            for tag, env in (("scheduled", {"DRINKME_RADIX_DECODER": "scheduled"}), ("default", {}))]


SNAPSHOT_8B = f"{HF_HOME}/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218"


def checks_steps(model: str) -> list:
    """The lean decoder's remaining kernel checks (docs/checks.md): its pack
    gate's selftest, and the twin gate over the real Qwen3-8B sip tensors
    (layer 0 + lm_head, parity_common's pack and snapshot from the volume)
    with the sip fixtures and synthetic widths, plus its selftest."""
    pd = pack_dir_for(model)
    g = f"{OUT_DIR}/gates"
    parity = {"DRINKME_PARITY_PACK_SIP": pd, "DRINKME_PARITY_SNAPSHOT": SNAPSHOT_8B}
    return [("lean gate selftest", [PY, "bench/radix_lean_gate.py", "--model", model, "--pack-dir", pd, "--limit", "2",
                                    "--no-timing", "--selftest", "--json", f"{g}/radix_lean_gate_selftest.json"], {}),
            ("gate radix_twin_bitpin real sip", [PY, "bench/radix_twin_bitpin.py", "--profiles", "sip", "--real-arms",
                                                 "sip", "--json", f"{g}/radix_twin_bitpin_real_sip.json"], parity),
            ("gate radix_twin_bitpin real sip selftest", [PY, "bench/radix_twin_bitpin.py", "--profiles", "sip",
                                                          "--real-arms", "sip", "--fixtures-only", "--selftest",
                                                          "--json", f"{g}/radix_twin_bitpin_real_sip_selftest.json"],
             parity)]


REV_8B = "b968826d9c46dd6066d109eabc6255188de91218"
HOTLOOP_COMPARE = r"""
import json, sys
old, new, tag = json.load(open(sys.argv[1])), json.load(open(sys.argv[2])), sys.argv[3]
bad = []
for k in old["cells"]:
    o, n = old["cells"][k], new["cells"][k]
    same = o["text"] == n["text"]
    if not same:
        bad.append(k)
    print(f"  {tag}/{k}: {o['tokens']}t @{o['decode_toks_per_s']} | {n['tokens']}t @{n['decode_toks_per_s']}"
          f"{' valid_json=' + str(n.get('valid_json')) if 'json' in k else ''}{'' if same else ' *** DIFFERS ***'}")
print(f"VERDICT {tag}: " + ("ALL CELLS TOKEN-IDENTICAL" if not bad else f"{len(bad)} cells differ: {bad}"))
"""


def v1checks_steps(model: str) -> list:
    """docs/checks.md's serve-level kernel checks the lean decoder had not
    run (round 1's report, section 6), on this card: logits_row_identity.py
    on the pack; greedy_ab_smoke.sh (compressed against --stock) under the
    default decoder and again under the scheduled one, then the two
    compressed transcripts compared; hotloop_gpu_acceptance.py's 8B pair
    (hotloop_gpu_ab.sh's old / new arms, the default decoder) and the new
    arm again under the scheduled decoder, each pair compared cell by cell.
    The smoke serves the pack from a container-local DRINKME_HOME whose
    canonical pack directory links to the volume's."""
    assert model == "Qwen3-8B", "the serve-level checks are wired for the volume's Qwen3-8B pack"
    pd = pack_dir_for(model)
    o = OUT_DIR
    home = {"DRINKME_HOME": "/root/dmh"}
    link = (f"mkdir -p /root/dmh/packs && ln -sfn {pd} /root/dmh/packs/Qwen--Qwen3-8B@{REV_8B[:12]}")
    # the gate writes its receipt beside its own bench/ directory: run a copy
    steps = [("logits_row_identity", ["sh", "-c", f"mkdir -p /root/lri/bench && cp bench/logits_row_identity.py "
                                                  f"/root/lri/bench/ && {PY} /root/lri/bench/logits_row_identity.py "
                                                  f"--pack {pd}; cp /root/lri/verification/*.json {o}/"], {})]
    for tag, env in (("default", {}), ("scheduled", {"DRINKME_RADIX_DECODER": "scheduled"})):
        steps.append((f"greedy_ab_smoke ({tag})",
                      ["sh", "-c", f"{link} && bash bench/greedy_ab_smoke.sh Qwen/Qwen3-8B; mkdir -p {o}/greedy_{tag} "
                                   f"&& cp /tmp/qsmoke_* {o}/greedy_{tag}/"], {**home, **env}))
    steps.append(("greedy compressed default vs scheduled",
                  ["sh", "-c", f"cmp {o}/greedy_default/qsmoke_compressed.txt {o}/greedy_scheduled/qsmoke_compressed.txt "
                               f"&& echo 'VERDICT decoder: compressed greedy transcripts BYTE-IDENTICAL' "
                               f"|| echo 'VERDICT decoder: compressed greedy transcripts DIFFER'"], {}))
    arms = (("old", {"DRINKME_HOTLOOP_OFF": "1"}), ("new", {"DRINKME_DETOK_VERIFY": "1"}),
            ("new_scheduled", {"DRINKME_DETOK_VERIFY": "1", "DRINKME_RADIX_DECODER": "scheduled"}))
    for tag, env in arms:
        steps.append((f"hotloop {tag}", [PY, "bench/hotloop_gpu_acceptance.py", "Qwen/Qwen3-8B", pd,
                                         f"{o}/hotloop_8b_{tag}.json", REV_8B], env))
    for a, b, tag in (("old", "new", "8b"), ("new_scheduled", "new", "8b-decoder")):
        steps.append((f"hotloop compare {tag}", [PY, "-c", HOTLOOP_COMPARE, f"{o}/hotloop_8b_{a}.json",
                                                 f"{o}/hotloop_8b_{b}.json", tag], {}))
    return steps


def abdec_steps(model: str, decoders: list, bench: bool = True) -> list:
    """The decoders A/B'd in one container: the eager bench loop
    (decode_step_profile --no-graph, arms.greedy) alternating the decoders
    twice, the graph-mode step alternating them twice, then `drinkme bench
    --pack-dir --no-spec` once under each (stock and the twin ride along
    unchanged). `decoders` are DRINKME_RADIX_DECODER values."""
    pd = pack_dir_for(model)
    steps = []
    for i, mode in enumerate(decoders * 2):
        steps.append((f"eager {mode} #{i}", [PY, "bench/decode_step_profile.py", "--model", model, "--arm", "compressed",
                                             "--pack-dir", pd, "--json", f"{OUT_DIR}/eager_{mode}_{i}.json",
                                             "--no-graph", "--no-naive", "--profile-steps", "10", "--attr-steps", "3"],
                      {"DRINKME_RADIX_DECODER": mode}))
    steps.append(("graph sweep decoders", [PY, "bench/decode_step_profile.py", "--model", model, "--arm", "compressed",
                                           "--pack-dir", pd, "--json", f"{OUT_DIR}/sweep_decoders.json", "--only-sweep",
                                           "--sweep", *[f"dec={d}" for d in decoders * 2]], {}))
    if bench:
        for mode in decoders:
            steps.append((f"drinkme bench ({mode})", ["uv", "run", "--no-sync", "drinkme", "bench", "--model", model,
                                                      "--pack-dir", pd, "--no-spec", "-o", f"{OUT_DIR}/bench_{mode}.json"],
                          {"DRINKME_BENCH_WARMUP": "1", "DRINKME_RADIX_DECODER": mode}))
    return steps


def v2gates_steps(model: str) -> list:
    """The default lean decoder (radix_ops._decoder, no override) through
    the codec gates on this card: the lean pack gate (every tensor: dense,
    M=1 and mc against the scheduled decoder, the checkpoint's bits,
    per-class timing) and its selftest, the gemv gate (all 65,536
    patterns), the twin gate's sip synthetic widths, the twin gate on the
    real sip tensors (+ selftest) and the mc gate on the pack."""
    pd = pack_dir_for(model)
    g = f"{OUT_DIR}/gates"
    parity = {"DRINKME_PARITY_PACK_SIP": pd, "DRINKME_PARITY_SNAPSHOT": SNAPSHOT_8B}
    return [
        ("lean gate selftest", [PY, "bench/radix_lean_gate.py", "--model", model, "--pack-dir", pd, "--limit", "2",
                                "--no-timing", "--selftest", "--json", f"{g}/radix_lean_gate_selftest.json"], {}),
        ("lean gate", [PY, "bench/radix_lean_gate.py", "--model", model, "--pack-dir", pd, "--seconds", "2.5",
                       "--json", f"{g}/radix_lean_gate.json"], {}),
        ("gate radix_gemv_bitpin", [PY, "bench/radix_gemv_bitpin.py", "--json", f"{g}/radix_gemv_bitpin.json"], {}),
        ("gate radix_twin_bitpin sip synthetic", [PY, "bench/radix_twin_bitpin.py", "--profiles", "sip", "--no-real",
                                                  "--json", f"{g}/radix_twin_bitpin_sip.json"], {}),
        ("gate radix_twin_bitpin real sip", [PY, "bench/radix_twin_bitpin.py", "--profiles", "sip", "--real-arms",
                                             "sip", "--json", f"{g}/radix_twin_bitpin_real_sip.json"], parity),
        ("gate radix_twin_bitpin selftest", [PY, "bench/radix_twin_bitpin.py", "--profiles", "sip", "--no-real",
                                             "--selftest", "--json", f"{g}/radix_twin_bitpin_selftest.json"], {}),
        ("gate radix_mc_bitpin --pack-dir", [PY, "bench/radix_mc_bitpin.py", "--json",
                                             f"{g}/radix_mc_bitpin_pack.json", "--pack-dir", pd], {})]


def v2meas_steps(model: str, bench: bool = True) -> list:
    """The scheduled and lean decoders measured in one container: abdec's
    steps over DRINKME_RADIX_DECODER=scheduled,lean (the eager bench loop
    and the graph-mode step each cycling them twice, then `drinkme bench
    --pack-dir --no-spec` under each, its 253/253 gate and stock / twin
    alongside)."""
    return abdec_steps(model, ["scheduled", "lean"], bench)


@app.local_entrypoint()
def main(what: str = "profile", card: str = "L4", model: str = "Qwen3-8B", out_dir: str = "",
         arms: str = "stock,compressed", extra: str = "", env: str = "", tag: str = "",
         decoders: str = ""):
    stamp = time.strftime("%Y%m%dT%H%M%S", time.gmtime())
    out_dir = out_dir or os.path.join(REPO, "verification", "nvidia-speed")
    os.makedirs(out_dir, exist_ok=True)
    token = os.environ.get("HF_TOKEN", "")
    extra_argv = extra.split() if extra else []
    extra_env = json.loads(env) if env else {}
    t0 = time.time()
    if what == "prep":
        rep = prep.remote(model, token)
    else:
        if what == "profile":
            steps = profile_steps(model, arms.split(","), extra_argv, extra_env)
        elif what == "lean":
            steps = lean_steps(model)
        elif what == "lean2":
            steps = lean2_steps(model)
        elif what == "bench":
            steps = bench_steps(model)
        elif what == "checks":
            steps = checks_steps(model)
        elif what == "v1checks":
            steps = v1checks_steps(model)
        elif what == "abdec":
            steps = abdec_steps(model, (decoders or "scheduled,lean").split(","), "--no-bench" not in extra_argv)
        elif what == "v2gates":
            steps = v2gates_steps(model)
        elif what == "v2meas":
            steps = v2meas_steps(model, "--no-bench" not in extra_argv)
        else:
            raise SystemExit(f"--what prep|profile|lean|lean2|bench|checks|v1checks|abdec|v2gates|v2meas, not {what!r}")
        fn = {"L4": gpu_L4, "H100": gpu_H100}[card]
        rep = fn.remote(steps)
    rep["local_wall_s"] = round(time.time() - t0, 1)
    rep["app_id"] = app.app_id
    name = f"{what}_{card if what != 'prep' else 'cpu'}{'_' + tag if tag else ''}_{stamp}"
    files = rep.pop("files", {}) or {}
    log = rep.pop("log", "")
    with open(os.path.join(out_dir, name + ".log"), "w") as f:
        f.write(log)
    for rel, text in files.items():
        dst = os.path.join(out_dir, f"{name}__{rel.replace('/', '__')}")
        with open(dst, "w") as f:
            f.write(text)
    with open(os.path.join(out_dir, name + ".meta.json"), "w") as f:
        json.dump({**rep, "files": sorted(files)}, f, indent=1)
    print(f"[nvidia-speed] {name}: {json.dumps(rep.get('steps'))} wall {rep.get('wall_s')} s "
          f"(local {rep['local_wall_s']} s) -> {out_dir}", flush=True)
