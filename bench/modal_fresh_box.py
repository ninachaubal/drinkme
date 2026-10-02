"""The fresh-box test: drinkme's stranger flow on a Modal NVIDIA GPU.

What a stranger does, verbatim, on hardware none of our code has touched:
copy the repo tree in, `uv sync` (the DEFAULT groups — this is the
accelerator-fork acceptance test: PyPI CUDA torch must arrive, never the
gfx1151 TheRock pin), then `drinkme bench` end to end — detect, fit check,
download, pack (the dieted streaming packer), both arms through the serve
path, bit-exact verification, receipt.

THE BARE `uv sync` BELOW RUNS INSIDE A DISPOSABLE MODAL CONTAINER AND MUST
NEVER BE COPIED TO A DEVELOPMENT BOX. It is the one place in this repo where
a default-group sync is the intended behaviour — being a virgin NVIDIA box IS
the test. On a real machine the same command replaces the pinned accelerator
torch in place and everything afterwards silently runs on CPU (AGENTS.md).

Usage (from the repo root; runs remotely, prints the receipt):
    uv tool run modal run bench/modal_fresh_box.py            # L4, Qwen3-1.7B
    uv tool run modal run bench/modal_fresh_box.py --model Qwen3-8B

Cost guardrails: one GPU container, hard timeout, no volumes, nothing
persists but the returned JSON.
"""

import json
import os

import modal

app = modal.App("drinkme-fresh-box")

# A deliberately boring base: NOT a torch image, NOT a CUDA devel image —
# the whole point is that `uv sync` on a bare box must produce the working
# environment by itself. (nvidia drivers come from Modal's GPU runtime.)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "curl")
    .run_commands("curl -LsSf https://astral.sh/uv/install.sh | sh")
    .add_local_dir(
        os.path.join(os.path.dirname(__file__), ".."),
        remote_path="/root/drinkme",
        # RECURSIVE globs. The bare `__pycache__` / `*.pyc` forms here matched
        # only the tree's TOP level, so every src/drinkme/__pycache__/*.pyc was
        # still being hashed — and a build begun while someone edits the source
        # dies with "was modified during build process". The
        # worktrees are excluded for size and because they hold other branches.
        ignore=["**/.venv/**", "**/.git/**", "**/__pycache__/**", "**/*.pyc",
                "**/.pytest_cache/**", "**/.worktrees/**"],
    )
)


@app.function(image=image, gpu="L4", timeout=2700, memory=24576, cpu=8)
def fresh_box(model: str, hf_token: str = "") -> dict:
    import subprocess

    env = dict(os.environ, PATH="/root/.local/bin:" + os.environ["PATH"])
    if hf_token:
        env["HF_TOKEN"] = hf_token

    def run(label, cmd, **kw):
        print(f"\n=== {label}: {' '.join(cmd)}")
        r = subprocess.run(cmd, cwd="/root/drinkme", env=env,
                           capture_output=True, text=True, **kw)
        tail = (r.stdout + r.stderr)[-4000:]
        print(tail)
        return r.returncode, tail

    report = {"gpu": "L4", "model": model, "steps": {}}

    def torch_identity(label):
        code, out = run(label, [
            "uv", "run", "--no-sync", "python", "-c",
            "import json;\n"
            "try:\n"
            "    import torch\n"
            "    print(json.dumps({'present': True, 'version': torch.__version__,\n"
            "        'cuda': torch.cuda.is_available(),\n"
            "        'device': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}))\n"
            "except ModuleNotFoundError:\n"
            "    print(json.dumps({'present': False}))",
        ])
        line = [l for l in out.strip().splitlines() if l.startswith("{")]
        return json.loads(line[-1]) if line else {"raw": out[-300:]}

    # 1. the stranger's sync. This installs BASE DEPS ONLY —
    #    `default-groups = []`, so uv never picks an accelerator for anyone.
    code, out = run("uv sync (default groups)", ["uv", "sync"])
    report["steps"]["sync"] = {"exit": code}
    if code != 0:
        report["steps"]["sync"]["tail"] = out
        return report

    # 2. torch must be ABSENT here. This is the inverted acceptance line: the
    #    old contract asserted a CUDA wheel had arrived, which is precisely the
    #    behaviour that swapped this project's pinned ROCm torch twice. Now the
    #    fork stays unresolved until something that knows the hardware decides.
    report["steps"]["torch_after_sync"] = torch_identity("torch after sync")
    report["steps"]["default_groups_are_empty"] = (
        report["steps"]["torch_after_sync"].get("present") is False)

    # 3. ONE COMMAND. bootstrap.ensure_accelerator reads the box, installs the
    #    matching group itself, and the bench runs. Nothing here names an
    #    accelerator — that is the whole claim being tested.
    code, out = run("drinkme bench", [
        "uv", "run", "--no-sync", "drinkme", "bench",
        "--model", model, "-o", "/tmp/receipt.json",
    ])
    report["steps"]["bench"] = {"exit": code, "tail": out[-2500:]}

    # 4. what bootstrap chose, after the fact
    report["steps"]["torch_after_bootstrap"] = torch_identity("torch after bootstrap")
    report["steps"]["bootstrap_picked_cuda"] = bool(
        report["steps"]["torch_after_bootstrap"].get("cuda"))

    if os.path.exists("/tmp/receipt.json"):
        report["receipt"] = json.load(open("/tmp/receipt.json"))
    return report


@app.local_entrypoint()
def main(model: str = "Qwen3-1.7B", out: str = ""):
    token = ""
    tp = os.path.expanduser("~/.cache/huggingface/token")
    if os.path.exists(tp):
        token = open(tp).read().strip()
    result = fresh_box.remote(model, token)
    text = json.dumps(result, indent=2)
    print(text)
    if out:
        with open(out, "w") as f:
            f.write(text)
