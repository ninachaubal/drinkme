"""Longest single GPU dispatch of one vision-tower (ViT) forward, per kernel
family, as a function of the image size: the ViT variant of
bench/prefill_dispatch.py, and the dispatch gate that calibrates
serving/vision.WORK_V (the bound on one ViT attention dispatch).

Why: amdgpu resets a graphics ring whose job waits past its lockup timeout,
and a compute dispatch that holds the CUs that long takes the desktop's
compositor with it (serving/prefill.py's docstring; the text path's longest
prefill dispatch is held near 60 ms). The stock ViT makes ONE attention call
over all N patches of an image (N^2 query-key pairs); bounded_attention
splits it at WORK_V pairs.

WORKER (`--size WxH`): the served tower alone, without the text model:
  * a root module holding only `model.visual`, built empty by
    arms.attach_vision_tower and filled by engines.stream_checkpoint (the
    served walker) from the snapshot, or with `--pack DIR` its Linears from
    the pack through swap.make_module (a hard-linked subset of the pack's
    tower tensors; nothing written into the pack) and the rest from the
    pack's embedded checkpoint;
  * serving/kernel_route.route_kernels, vision.bound (bounded attention;
    `--unbounded` is DRINKME_VISION_BOUNDED=0, one call per image, `--work N` sets
    vision.WORK_V) and vision.route_patch_embed, as HFEngine installs them;
  * a synthetic screenshot (bench/vision_screens.py) through vision.load's
    preprocessing at `--max-pixels` (the server cap), then Tower.features
    (the engine's call) `--reps` times. Rep 0 compiles and autotunes; a
    dispatch's intrinsic time is its minimum over reps 1.. .
Run with TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1, as the served units are.

DRIVER (`--ladder 640x480,1920x1080,...`): each size's worker under
`rocprofv3 --kernel-trace`, smallest first. Before each step it predicts the
step's longest dispatch from the last measured one (x2 for a first dispatch)
and refuses to launch a step predicted over --launch-ms; it stops the ladder
when a measured dispatch passes --stop-ms (60) or when the kernel log shows a
`ring gfx_0.0.0 timeout` since the step began. One summary.json per step and
<label>.json for the ladder under --out.

ENGINE (`--engine PACK --ladder ...`): the whole served path instead of the
tower alone: engines via serve.build_engine from the pack, one greedy
one-token request per size through complete() (the tower, then the text
prefill, where an image run longer than the prefill chunk is a span of its
own, serving/prefill.spans). Each step is its own rocprofv3 process with its
cap (`--max-pixels`), and the longest dispatch inside the request's window is
reported with the family split, the same ladder guards applied.

Hold `flock -w 3600 /tmp/drinkme-gpu.lock` around the driver.
"""
from __future__ import annotations

import argparse
import base64
import glob
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from prefill_dispatch import PY, ROCPROF, mem_available_gb, read_trace, summarize  # noqa: E402

HUB = os.path.expanduser("~/.cache/huggingface/hub")
MODELS = {"27b": "models--Qwen--Qwen3.8-27B",
          "mimo": "models--XiaomiMiMo--MiMo-V2.6-Distill-Qwen-9B"}
REPS = 4


def snapshot(model: str) -> str:
    snaps = sorted(glob.glob(os.path.join(HUB, MODELS[model], "snapshots", "*")))
    if not snaps:
        raise SystemExit(f"no snapshot for {model} under {HUB}")
    return snaps[-1]


# --------------------------------------------------------------- worker --

def _subset_pack(pack: str, tmp: str, prefix: str) -> str:
    """The pack's tensors under `prefix`, hard-linked into tmp with a
    rewritten meta.json. Read-only on the pack."""
    meta = json.load(open(os.path.join(pack, "meta.json")))
    keep = {k: v for k, v in meta["tensors"].items() if k.startswith(prefix)}
    if not keep:
        raise SystemExit(f"{pack}: no tensor under {prefix} (a pack cut before the vision block?)")
    for fn in set(keep.values()):
        os.link(os.path.join(pack, fn), os.path.join(tmp, fn))
    json.dump(dict(meta, tensors=keep), open(os.path.join(tmp, "meta.json"), "w"))
    return tmp


def build(snap: str, pack: str | None, device: str = "cuda"):
    import torch
    from transformers import AutoConfig

    from drinkme.arms import attach_vision_tower
    from drinkme.serving.engines import stream_checkpoint
    from drinkme.serving.kernel_route import route_kernels

    cfg = AutoConfig.from_pretrained(snap)
    root = torch.nn.Module()
    root.model = torch.nn.Module()
    root.tie_weights = lambda: None  # stream_checkpoint reties; nothing is tied here
    path = attach_vision_tower(root, cfg)
    raw_from = snap
    if pack:
        from drinkme.codec.pack import embedded_dir, iter_pack_dir
        from drinkme.codec.swap import make_module

        scratch = os.path.dirname(os.path.abspath(pack))  # same filesystem as the pack
        with tempfile.TemporaryDirectory(dir=scratch, prefix="vision-dispatch-") as tmp:
            for name, p in iter_pack_dir(_subset_pack(pack, tmp, path + ".")):
                parent, _, child = name.rpartition(".")
                assert isinstance(root.get_submodule(name), torch.nn.Linear), name
                setattr(root.get_submodule(parent), child, make_module(p, None, device))
        if os.path.isdir(embedded_dir(pack)):
            raw_from = embedded_dir(pack)
    stream_checkpoint(root, raw_from, cfg, device, only=path + ".")
    route = route_kernels(root, device)
    return root, cfg, path, route


def engine_worker(args) -> None:
    import torch

    from drinkme.serve import build_engine
    from drinkme.serving import vision
    from drinkme.serving.engine import GenerationRequest, SampleParams, complete
    from vision_screens import png_bytes, screenshot

    os.environ.update(DRINKME_SPEC="off", DRINKME_PREFIX_SLOTS="0", DRINKME_SLOT_DIR="off",
                      HF_HUB_OFFLINE="1", DRINKME_NO_AUTO_DEPS="1",
                      DRINKME_IMAGE_MAX_PIXELS=str(args.max_pixels))
    meta = json.load(open(os.path.join(args.engine, "meta.json")))
    eng = build_engine(meta["hfRepo"], None, args.engine, stock=False, ctx=16384)
    assert torch.cuda.is_available() and str(eng.device) != "cpu", "REFUSING: no accelerator"
    w, h = (int(x) for x in args.size.lower().split("x"))
    data = png_bytes(screenshot(w, h, scale=args.scale))
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text",
                                                               "text": "What is shown here?"}]}]
    walls, windows, tokens = [], [], None
    for _ in range(args.reps):
        img = eng.vision.prepare(vision.parse_base64(base64.b64encode(data).decode(), "image/png",
                                                     where="dispatch"))
        tokens = img.tokens
        req = GenerationRequest(msgs, SampleParams(temperature=0.0, max_tokens=1), images=(img,),
                                template_kwargs={"enable_thinking": False})
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        m0 = time.monotonic_ns()
        complete(eng, req)
        torch.cuda.synchronize()
        walls.append(time.perf_counter() - t0)
        windows.append((m0, time.monotonic_ns()))
    n = img.grid_thw[1] * img.grid_thw[2]
    print("VERDICT " + json.dumps({
        "model": meta["hfRepo"], "size": [w, h], "resized": list(img.size),
        "grid": list(img.grid_thw), "patches": n, "tokens": tokens, "pairs": n * n,
        "work": vision.WORK_V, "query_rows_per_dispatch": n if n * n <= vision.WORK_V
        else max(vision.MIN_ROWS, vision.WORK_V // n), "prefill_chunk": eng._prefill_chunk,
        "compressed": True, "route": None, "wall_s": walls, "windows": windows,
        "peak_gib": torch.cuda.max_memory_allocated() / 2**30}), flush=True)


def worker(args) -> None:
    import torch

    from drinkme.serving import vision
    from vision_screens import png_bytes, screenshot

    assert torch.cuda.is_available(), "REFUSING: no accelerator"
    snap = snapshot(args.model)
    root, cfg, path, route = build(snap, args.pack)
    tower = vision.Tower(cfg.model_type, path, int(cfg.image_token_id),
                         int(cfg.vision_config.spatial_merge_size))
    if args.work:
        vision.WORK_V = args.work
    if args.unbounded:
        os.environ[vision.BOUNDED_ENV] = "0"  # one call per image, still head-padded
    routed = vision.bound(tower, root)
    vision.route_patch_embed(tower, root)  # as HFEngine installs it
    vis = vision.load(cfg.model_type, snap, max_pixels=args.max_pixels)
    w, h = (int(x) for x in args.size.lower().split("x"))
    data = png_bytes(screenshot(w, h, scale=args.scale))
    img = vis.prepare(vision.parse_base64(base64.b64encode(data).decode(), "image/png",
                                          where="dispatch"))
    n = img.grid_thw[1] * img.grid_thw[2]
    walls, windows = [], []
    for _ in range(args.reps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        m0 = time.monotonic_ns()  # rocprofv3's timestamps are CLOCK_MONOTONIC
        with torch.inference_mode():
            out = tower.features(root, img)
        torch.cuda.synchronize()
        walls.append(time.perf_counter() - t0)
        windows.append((m0, time.monotonic_ns()))
        del out
    work = vision.WORK_V
    rows = n if routed == 0 or n * n <= work else max(vision.MIN_ROWS, work // n)
    print("VERDICT " + json.dumps({
        "model": args.model, "size": [w, h], "resized": list(img.size), "grid": list(img.grid_thw),
        "patches": n, "tokens": img.tokens, "pairs": n * n, "work": work,
        "bounded_modules": routed, "query_rows_per_dispatch": rows,
        "attention_dispatches_per_layer": -(-n // rows),
        "compressed": bool(args.pack), "route": route, "wall_s": walls, "windows": windows,
        "peak_gib": torch.cuda.max_memory_allocated() / 2**30}), flush=True)


# --------------------------------------------------------------- driver --

def ring_timeouts(since: str) -> int:
    p = subprocess.run(["journalctl", "-k", "--since", since, "--no-pager"],
                       capture_output=True, text=True)
    return sum("ring gfx_0.0.0 timeout" in ln for ln in p.stdout.splitlines())


def run_one(args, size: str, outdir: Path) -> dict:
    floor = 50 if args.engine else 12
    if mem_available_gb() < floor:
        raise SystemExit(f"REFUSING: MemAvailable {mem_available_gb():.1f} GB < {floor}")
    arm = "comp" if args.pack else "stock"
    if args.engine:
        arm = "engine"
    tag = (f"{args.model}_{size}_{arm}{'_unbounded' if args.unbounded else ''}"
           f"{'_w' + str(args.work) if args.work else ''}_cap{args.max_pixels}")
    d = outdir / tag
    d.mkdir(parents=True, exist_ok=True)
    cmd = [ROCPROF, "--kernel-trace", "-f", "csv", "-d", str(d), "-o", "trace", "--",
           PY, __file__, "--model", args.model, "--size", size, "--reps", str(args.reps),
           "--max-pixels", str(args.max_pixels), "--scale", str(args.scale)]
    if args.pack:
        cmd += ["--pack", args.pack]
    if args.engine:
        cmd += ["--engine", args.engine]
    if args.unbounded:
        cmd.append("--unbounded")
    if args.work:
        cmd += ["--work", str(args.work)]
    env = dict(os.environ, TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL="1",
               OMP_NUM_THREADS="8", MKL_NUM_THREADS="8")
    since = time.strftime("%Y-%m-%d %H:%M:%S")
    p = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=1800)
    (d / "run.log").write_text(p.stdout + "\n--- stderr ---\n" + p.stderr)
    timeouts = ring_timeouts(since)
    verdict = [ln for ln in p.stdout.splitlines() if ln.startswith("VERDICT ")]
    if not verdict:
        raise SystemExit(f"worker {tag} gave no verdict; see {d/'run.log'} "
                         f"(ring timeouts since start: {timeouts})")
    res = json.loads(verdict[0][8:])
    res["ring_timeouts"] = timeouts
    rows = read_trace(str(d), res["windows"])
    by_rep = [sorted((r for r in rows if r["rep"] == i), key=lambda r: r["t0"])
              for i in range(len(res["windows"]))]
    steady = by_rep[1:]
    if len({len(x) for x in steady}) != 1:
        raise SystemExit(f"worker {tag}: reps 1.. dispatched {[len(x) for x in steady]} kernels")
    intrinsic = [{"name": col[0]["name"], "ms": min(r["ms"] for r in col)} for col in zip(*steady)]
    res["n_dispatch"] = len(intrinsic)
    res["rep_max_ms"] = [max((r["ms"] for r in x), default=0.0) for x in by_rep]
    res["families"] = summarize(intrinsic)
    res["families_observed"] = summarize(rows)
    res["max_ms"] = max(r["ms"] for r in intrinsic)
    res["max_kernel"] = max(intrinsic, key=lambda r: r["ms"])["name"][:160]
    res["observed_max_ms"] = max(r["ms"] for r in rows)
    res["observed_max_kernel"] = max(rows, key=lambda r: r["ms"])["name"][:160]
    # the whole-process max too: model build and preprocessing included
    allrows = read_trace(str(d))
    res["process_max_ms"] = max(r["ms"] for r in allrows)
    res["process_max_kernel"] = max(allrows, key=lambda r: r["ms"])["name"][:160]
    for fn in glob.glob(str(d / "**" / "*kernel_trace.csv"), recursive=True):
        subprocess.run(["gzip", "-f", fn])  # the raw per-dispatch receipt, compressed
    (d / "summary.json").write_text(json.dumps(res, indent=1))
    return res


def predict(last: dict, patches: int, work: int, bounded: bool) -> float:
    """The next step's longest dispatch from the last one: attention grows
    with the pairs ONE dispatch carries (min(N^2, work) bounded, N^2 not),
    everything else with N; x2 for a first dispatch."""
    n0 = last["patches"]
    per = (lambda n: min(n * n, work)) if bounded else (lambda n: n * n)
    ratio = max(patches / n0, per(patches) / per(n0), 1.0)
    return 2 * last["observed_max_ms"] * ratio


def patches_for(args, size: str) -> int:
    from drinkme.serving import vision

    w, h = (int(x) for x in size.lower().split("x"))
    vis = vision.load("qwen3_5", snapshot(args.model), max_pixels=args.max_pixels)
    _, grid, _ = vis.preprocessor.plan(h, w, args.max_pixels, where="ladder")
    return grid[1] * grid[2]


def ladder(args) -> list[dict]:
    from drinkme.serving import vision

    work = args.work or vision.WORK_V
    outdir = args.out
    outdir.mkdir(parents=True, exist_ok=True)
    label = (f"ladder_{args.model}_{'engine' if args.engine else 'comp' if args.pack else 'stock'}"
             f"{'_unbounded' if args.unbounded else ''}{'_w' + str(args.work) if args.work else ''}")
    done: list[dict] = []
    for size in args.ladder.split(","):
        n = patches_for(args, size)
        if done:
            last = done[-1]
            if last["observed_max_ms"] > args.stop_ms:
                print(f"[{label}] STOP: measured {last['observed_max_ms']:.1f} ms > {args.stop_ms}",
                      flush=True)
                break
            if n < last["patches"]:
                print(f"[{label}] ladder not increasing at {size}", flush=True)
                break
            pred = predict(last, n, work, not args.unbounded)
            if pred > args.launch_ms:
                print(f"[{label}] NOT LAUNCHING {size} ({n} patches): predicted {pred:.0f} ms "
                      f"> {args.launch_ms}", flush=True)
                break
        r = run_one(args, size, outdir)
        top = sorted(r["families"].items(), key=lambda kv: -kv[1]["max_ms"])[:4]
        print(f"[{label}] {size:>10} -> {r['resized'][1]}x{r['resized'][0]} N={r['patches']:>6} "
              f"rows/dispatch={r['query_rows_per_dispatch']:>6}: intrinsic max {r['max_ms']:7.2f} ms"
              f" ({r['max_kernel'][:50]}), observed max {r['observed_max_ms']:7.2f} ms, "
              + "  ".join(f"{k} {v['max_ms']:.2f}" for k, v in top)
              + f"  wall {r['wall_s'][-1] * 1e3:.0f} ms  ring timeouts {r['ring_timeouts']}",
              flush=True)
        done.append(r)
        if r["ring_timeouts"]:
            print(f"[{label}] STOP: {r['ring_timeouts']} ring gfx_0.0.0 timeout(s) in the kernel log",
                  flush=True)
            break
    (outdir / f"{label}.json").write_text(json.dumps(done, indent=1))
    return done


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=sorted(MODELS), default="27b")
    ap.add_argument("--size", help="worker: one image, WxH")
    ap.add_argument("--ladder", help="driver: WxH sizes, smallest first, comma-separated")
    ap.add_argument("--pack", help="the compressed arm: a pack directory with a vision block")
    ap.add_argument("--unbounded", action="store_true", help="DRINKME_VISION_BOUNDED=0")
    ap.add_argument("--engine", help="the whole served path from this pack (ENGINE above)")
    ap.add_argument("--work", type=int, default=0, help="vision.WORK_V for this run")
    ap.add_argument("--max-pixels", type=int, default=2560 * 1440, help="the server's pixel cap")
    ap.add_argument("--scale", type=float, default=1.0, help="the screenshot's UI scale")
    ap.add_argument("--reps", type=int, default=REPS)
    ap.add_argument("--out", type=Path, default=Path("/tmp/vision_dispatch"))
    ap.add_argument("--stop-ms", type=float, default=60.0)
    ap.add_argument("--launch-ms", type=float, default=250.0)
    args = ap.parse_args()
    if args.ladder:
        return ladder(args)
    return engine_worker(args) if args.engine else worker(args)


if __name__ == "__main__":
    main()
