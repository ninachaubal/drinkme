"""The publication sweep: serve_timing_ab on a Modal NVIDIA GPU — one card, one receipt.

The consolidation rule: every published number
is the drinkme SERVE PATH, uniformly — the research twin harness found the
direction and stays as preliminary work; the artifact's own A/B is the
result. So each curve point re-runs as: fresh box, `uv sync` (accelerator
fork: PyPI CUDA torch must arrive), `drinkme pack`, the radix bit-pin gates
(`radix_gemv_bitpin.py` — every bf16 bit pattern through the streams, the
fallback, the float64-bound math — and `radix_mc_bitpin.py --pack-dir` on the
pack's own tensors; the kernels have to prove themselves on every NEW vendor
before any number is kept), then `serve_timing_ab.py`.

THE BARE `uv sync` BELOW RUNS INSIDE A DISPOSABLE MODAL CONTAINER AND MUST
NEVER BE COPIED TO A DEVELOPMENT BOX — on a real machine it replaces the
pinned accelerator torch in place and everything afterwards silently runs on
CPU (AGENTS.md). Here, arriving at PyPI CUDA torch IS the acceptance test.

Usage (from the repo root; GPU is fixed at import time, so it rides an env):
    AB_GPU=L4   uv tool run modal run bench/modal_serve_ab.py --model Qwen3-8B
    AB_GPU=A10G uv tool run modal run bench/modal_serve_ab.py --model Qwen3-8B

Receipt lands locally at verification/serve_timing_ab_MODAL-<gpu>_<date>.json
(the remote hostname is a container id; the filename carries the card).

Cost guardrails: one GPU container, hard timeout, no volumes, nothing
persists but the returned JSON. Cheapest card first; read its receipt before
spending on the next.
"""

import json
import os

import modal

GPU = os.environ.get("AB_GPU", "L4")

app = modal.App("drinkme-serve-ab")

# Same deliberately boring base as modal_fresh_box.py: uv sync on a bare box
# must produce the working environment by itself.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "curl")
    .run_commands("curl -LsSf https://astral.sh/uv/install.sh | sh")
    .add_local_dir(
        os.path.join(os.path.dirname(__file__), ".."),
        remote_path="/root/drinkme",
        ignore=[".venv", ".git", "__pycache__", "*.pyc", ".pytest_cache",
                ".worktrees"],
    )
)


# cpu=8 is PINNED so every point shares one container shape. Known
# consequence: compressed LOAD is CPU-bound in the container — 152s vs
# 39.9s on a Strix Halo (gfx1151, 128 GB unified); timing claims are
# unaffected (load is untimed warmup).
@app.function(image=image, gpu=GPU, timeout=3600, memory=32768, cpu=8)
def serve_ab(model: str, hf_token: str = "",
             turns: int = 4, max_tokens: int = 192, ctx: int = 8192,
             gpu_label: str = "") -> dict:
    import glob
    import subprocess

    env = dict(os.environ, PATH="/root/.local/bin:" + os.environ["PATH"])
    if hf_token:
        env["HF_TOKEN"] = hf_token

    # gpu_label is passed FROM THE LOCAL SIDE: the module-global GPU
    # re-evaluates inside the container (where AB_GPU is unset) and silently
    # reads "L4"; the receipt's bandwidth_probe.gpus is the authoritative
    # device witness.
    report = {"gpu": gpu_label or GPU, "model": model, "steps": {}}

    def run(label, cmd, **kw):
        print(f"\n=== {label}: {' '.join(cmd)}", flush=True)
        r = subprocess.run(cmd, cwd="/root/drinkme", env=env,
                           capture_output=True, text=True, **kw)
        tail = (r.stdout + r.stderr)[-4000:]
        print(tail, flush=True)
        report["steps"][label] = {"exit": r.returncode, "tail": tail[-2500:]}
        return r.returncode

    if run("uv sync", ["uv", "sync"]):
        return report
    if run("pack", ["uv", "run", "--no-sync", "drinkme", "pack",
                    "--model", model]):
        return report

    packs = glob.glob("/root/.cache/drinkme/packs/*")
    if len(packs) != 1:
        report["error"] = f"expected exactly one pack, found {packs}"
        return report
    pack_dir = packs[0]
    report["pack_dir"] = pack_dir

    # THE GATES: the radix dense/gemv gate (every bf16 bit pattern, no pack
    # needed) and the multi-column gate on this pack's tensors, on this
    # vendor. Exit code carries the verdict (NVIDIA: honest); a mismatch
    # ends the run before any timing number exists to be quoted.
    os.makedirs("/root/drinkme/verification", exist_ok=True)
    if run("radix gemv bit-pin", ["uv", "run", "--no-sync", "python", "bench/radix_gemv_bitpin.py",
                                  "--json", "/root/drinkme/verification/radix_gemv_bitpin.json"]):
        report["error"] = "radix gemv bit-pin FAILED on this vendor"
        return report
    if run("radix mc bit-pin", ["uv", "run", "--no-sync", "python", "bench/radix_mc_bitpin.py",
                                "--pack-dir", pack_dir,
                                "--json", "/root/drinkme/verification/radix_mc_bitpin.json"]):
        report["error"] = "radix mc bit-pin FAILED on this vendor"
        return report

    env["DRINKME_AB_PACK"] = pack_dir
    env["PYTHONPATH"] = "src"
    if run("serve timing A/B", ["uv", "run", "--no-sync", "python",
                                "bench/serve_timing_ab.py",
                                "--turns", str(turns),
                                "--max-tokens", str(max_tokens),
                                "--ctx", str(ctx)]):
        return report

    receipts = sorted(glob.glob("/root/drinkme/verification/serve_timing_ab_*.json"),
                      key=os.path.getmtime)
    if receipts:
        report["receipt"] = json.load(open(receipts[-1]))
    return report


@app.local_entrypoint()
def main(model: str = "Qwen3-8B", out: str = "",
         turns: int = 4, max_tokens: int = 192, ctx_len: int = 8192):
    # `ctx_len`, not `ctx`: modal's local_entrypoint is Click-wrapped and Click
    # owns the parameter name `ctx` (its own context object) — a param named
    # ctx dies with "got multiple values for argument 'ctx'" before launch.
    import datetime

    report = serve_ab.remote(model, os.environ.get("HF_TOKEN", ""),
                             turns, max_tokens, ctx_len, GPU)
    date = datetime.date.today().isoformat()
    path = out or os.path.join(
        os.path.dirname(__file__), "..", "verification",
        f"serve_timing_ab_MODAL-{GPU}_{date}.json")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(report, f, indent=1)
    print(f"\nwrote {path}")
    if "receipt" in report:
        print(json.dumps(report["receipt"].get("summary", {}), indent=1))
    else:
        # no receipt = the run FAILED, whatever the step tails say — exit
        # nonzero so a detached dispatch reports failure instead of smiling
        # (an unfailable gate — exit 0 with no receipt — is worse than none)
        print("NO RECEIPT — read the step tails above")
        raise SystemExit(1)
