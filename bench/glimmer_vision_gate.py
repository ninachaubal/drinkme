"""GPU acceptance for Muse-Glimmer image input, for Muse-Glimmer-30B: the
tower's bit-pin, the first-token logits row, a screenshot over the wire and
the dispatch gate, one process per step. Each step prints one
`VERDICT {json}` line; read that, not the exit status (TheRock ROCm torch
can exit 0 after a failure). The steps and their flags are
gemma_vision_gate.py's; what differs is the tower and the reference.

  tower    the ViT bit-pin, and the dispatch gate's worker
           under rocprofv3. The served tower ALONE (model.vision_tower,
           model.vision_adapter, model.vision_projection, ~3.84 GB bf16) in
           the served tree's own wrapper with its text model and lm_head
           pruned, streamed from the snapshot by the served walker, and with
           --pack a second copy whose Linears come from the pack
           (swap.make_module over a hard-linked subset of its tower
           tensors; nothing is written into the pack) and the rest from its
           embedded checkpoint. One synthetic screenshot through the served
           preprocessing at --max-pixels, then --reps times each:
             (a) transformers' own MuseGlimmerModel.get_image_features over
                 the stock arm's tree with drinkme's attention route
                 installed (vision.bound, as HFEngine installs it, on every
                 arm before any run: bench/vision_bitpin.py's
                 "ref_served"). Past WORK_V pairs a full layer is split into
                 query blocks, and on gfx1151 flash's bits depend on the
                 block's row count (the Qwen3.8-27B tower's bounded and
                 one-call outputs were not byte-equal: max |diff| 22, mean
                 0.0045), so an unrouted (a) could not be byte-equal;
             (b) drinkme's GlimmerTower.features on the same tree,
             (c) (b) over the pack's Linears (the packed patch embedding
                 through vision.glimmer_patch_embed).
           PASS when a == b (== c) byte for byte. That cannot see a wrong
           attention kernel (every arm calls the same one; AOTriton was
           wrong at the Qwen3.5 and gemma-4 ViTs' head width, 72, and
           Glimmer's is 96, a multiple of HEAD_ALIGN and so unpadded), so
           --fp32-check holds every ViT attention call of one more
           (untimed) forward per arm against fp32 on that call's own inputs,
           with the boot self-test's tolerances, and runs the stock tower in
           float32 on the CPU with transformers' eager attention for the
           features' finiteness and the printed row cosines
           (gemma_vision_gate.fp32_verdict). The CPU tower holds 7.7 GB of
           fp32 weights on the host, and its eager full layer a
           16 x patches^2 fp32 score matrix (17 GB at the cap), so run it at
           a small size. --no-fp32-tower leaves it out: the per-call check
           alone, which is what decides, with each arm's own features read
           for finiteness. That runs at the cap, where the full layers see
           16,320 keys (on gfx1151 AOTriton's head-72 fault showed from
           4,096 keys and not at 1,024, so a small image cannot stand in).
           Glimmer's token count grows
           with the image up to its 4,096 cap, so the dispatch ladder is over
           sizes: 640x480 (391 tokens, 1,564 patches), 1280x720 (1,196 /
           4,784), 1920x1080 (2,691 / 10,764), 2560x1440 (4,080 / 16,320,
           the cap: a full-attention layer is 266M query-key pairs, eight
           bounded dispatches).
  row      the first-token logits row of an image prompt and a
           short greedy answer. --arm stock streams the wrapper WITH its
           tower (arms.load_stock_streaming(vision=True)) and runs transformers' own
           forward(input_ids, pixel_values, image_grid_thw), then drinkme's
           engine over the same tree. --arm compressed loads the pack
           through engines.load_compressed. Each writes <out>/<arm>.pt.
  compare  <out>/stock.pt against <out>/compressed.pt (gemma_vision_gate.compare).
  trace    the longest dispatches in a rocprofv3 --kernel-trace directory.
  smoke    a screenshot carrying a nonce, sent to a running
           `drinkme serve` on --port through both dialects. Glimmer writes
           its reasoning message first (returned as reasoning, not in the
           reply's text), so the smoke asks for low reasoning strength in a
           system turn and 256 tokens, and reads the nonce in the answer.

Hold `flock -w 3600 /tmp/drinkme-gpu.lock` around every GPU step.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gemma_vision_gate as g  # noqa: E402

TOWER = ("model.vision_tower", "model.vision_adapter", "model.vision_projection")
NEED_GB = {"tower": 12, "stock": 75, "compressed": 55}
# the steps that are the same for every architecture
compare, trace, smoke = g.compare, g.trace, g.smoke


def guard(kind: str, args) -> None:
    if args.need_gb is None:
        args.need_gb = NEED_GB[kind]
    g.guard(kind, args)


def processor_batch(img, device):
    """The image as transformers' processor hands it to the model:
    pixel_values [patches, 1176] float32 and image_grid_thw [1, 3]
    (bit-pinned against MuseGlimmerImageProcessor in
    tests/test_serving_glimmer_vision.py)."""
    import numpy as np
    import torch

    pv = torch.from_numpy(np.array(img.pixel_values)).to(device)
    return pv, torch.tensor([list(img.grid_thw)], dtype=torch.long, device=device)


# --------------------------------------------------------------- tower --

def _subset_pack(pack: str, tmp: str) -> str:
    """The pack's tower tensors, hard-linked into tmp with a rewritten
    meta.json. Read-only on the pack."""
    from drinkme.codec.pack import under

    meta = json.load(open(os.path.join(pack, "meta.json")))
    # under(): model.vision_projection is one Linear, named the path itself
    keep = {k: v for k, v in meta["tensors"].items() if under(k, TOWER)}
    if not keep or not meta.get("vision"):
        raise SystemExit(f"{pack}: no vision block or tower tensors (re-pack it)")
    for fn in set(keep.values()):
        os.link(os.path.join(pack, fn), os.path.join(tmp, fn))
    json.dump(dict(meta, tensors=keep), open(os.path.join(tmp, "meta.json"), "w"))
    return tmp


def tower_root(snap: str, cfg, device: str, pack: str | None = None):
    """The served tree's wrapper holding only Glimmer's tower (the text
    model and lm_head pruned to None), built empty and filled by the served
    walker (stream_checkpoint)."""
    import torch

    from drinkme.arms import skeleton
    from drinkme.serving.engines import stream_checkpoint

    root = skeleton(cfg, vision=True)
    root.model.language_model = None
    root.lm_head = None
    root.tie_weights = lambda: None  # nothing is tied here
    raw_from = snap
    if pack:
        from drinkme.codec.pack import embedded_dir, iter_pack_dir
        from drinkme.codec.swap import make_module

        scratch = os.path.dirname(os.path.abspath(pack))  # the pack's filesystem
        with tempfile.TemporaryDirectory(dir=scratch, prefix="glimmer-gate-") as tmp:
            for name, p in iter_pack_dir(_subset_pack(pack, tmp)):
                parent, _, child = name.rpartition(".")
                assert isinstance(root.get_submodule(name), torch.nn.Linear), name
                setattr(root.get_submodule(parent), child, make_module(p, None, device))
        if os.path.isdir(embedded_dir(pack)):
            raw_from = embedded_dir(pack)
    stream_checkpoint(root, raw_from, cfg, device, only=tuple(p + "." for p in TOWER))
    return root


def fp32_check(snap: str, cfg, img, got: dict, runs: dict, cpu_tower: bool = True) -> dict:
    """gemma_vision_gate.fp32_verdict: each arm's attention calls in one
    more forward (`runs`, untimed) against fp32 (g.attention_calls), and
    each arm's features (`got`) against Glimmer's stock tower in float32 on
    the CPU, from the same snapshot, its attention transformers' eager
    function. Without `cpu_tower`, each arm's features are only read for
    finiteness (--no-fp32-tower)."""
    import torch
    from transformers.models.muse_glimmer.modeling_muse_glimmer import MuseGlimmerModel

    with torch.inference_mode():
        calls = {arm: g.attention_calls(run) for arm, run in runs.items()}
    if not cpu_tower:
        return g.fp32_verdict({arm: {"finite": bool(torch.isfinite(o).all())}
                               for arm, o in got.items()}, calls)
    root = tower_root(snap, cfg, "cpu").float()
    root.model.vision_tower.config._attn_implementation = "eager"
    pv, grid = processor_batch(img, "cpu")
    with torch.inference_mode():
        want = MuseGlimmerModel.get_image_features(root.model, pv, grid).pooler_output[0]
    del root
    return g.fp32_verdict(g.fp32_rows(got, want), calls)


def tower(args) -> None:
    import torch
    from transformers import AutoConfig
    from transformers.models.muse_glimmer.modeling_muse_glimmer import MuseGlimmerModel

    from drinkme.arms import vision_tower_paths
    from drinkme.serving import vision
    from drinkme.serving.checkpoint import tokenizer
    from drinkme.serving.kernel_route import route_kernels

    guard("tower", args)
    snap = args.snap
    cfg = AutoConfig.from_pretrained(snap)
    vis = vision.load(cfg.model_type, snap, max_pixels=args.max_pixels)
    gt = vision.tower_for(cfg.model_type, cfg, vision_tower_paths(cfg), tokenizer(snap, None))
    arms = {"stock": tower_root(snap, cfg, args.device)}
    if args.pack:
        arms["compressed"] = tower_root(snap, cfg, args.device, args.pack)
    for root in arms.values():
        route_kernels(root, args.device)
        # before (a): transformers' forward takes drinkme's attention too
        # (the module docstring's ref_served)
        routed = vision.bound(gt, root)
    img = g.prepared(vis, args.size)
    pv, grid = processor_batch(img, args.device)
    out, walls, windows = {}, [], []

    def timed(fn):
        res = None
        fresh = [g.prepared(vis, args.size) for _ in range(args.reps)]  # the tower releases each
        for i in range(args.reps):
            g.sync(args.device)
            t0, m0 = time.perf_counter(), time.monotonic_ns()
            with torch.inference_mode():
                res = fn(fresh[i])
            g.sync(args.device)
            walls.append(time.perf_counter() - t0)
            windows.append((m0, time.monotonic_ns()))
        return res

    fns = {"reference": lambda _img: MuseGlimmerModel.get_image_features(
        arms["stock"].model, pv, grid).pooler_output[0]}
    for arm, root in arms.items():
        fns[arm] = lambda im, root=root: gt.features(root, im)
    for arm, fn in fns.items():
        out[arm] = timed(fn)
    sha = {k: g.digest(v) for k, v in out.items()}
    ok = len(set(sha.values())) == 1
    diff = {k: float((v.float() - out["reference"].float()).abs().max()) for k, v in out.items()}
    patches = int(pv.shape[0])
    checked = fp32_check(snap, cfg, img, out, {
        arm: lambda fn=fn: fn(g.prepared(vis, args.size)) for arm, fn in fns.items()},
        cpu_tower=not getattr(args, "no_fp32_tower", False)) if args.fp32_check else {}
    print("VERDICT " + json.dumps({**checked,
        "step": "tower", "verdict": "PASS" if ok else "FAIL", "size": args.size,
        "max_pixels": args.max_pixels, "resized": list(img.size), "grid": list(img.grid_thw),
        "tokens": img.tokens, "patches": patches,
        "pairs_per_full_attention_call": patches ** 2, "work_v": vision.WORK_V,
        "bounded_modules": routed, "sha256": sha, "max_abs_diff_vs_reference": diff,
        "wall_s": walls, "windows": windows, "device": args.device,
        "peak_gib": (torch.cuda.max_memory_allocated() / 2**30
                     if args.device == "cuda" else None)}), flush=True)


# ----------------------------------------------------------------- row --

def row(args) -> None:
    import torch
    from transformers import AutoConfig

    from drinkme.serving.engines import HFEngine, _vision_for, load_compressed

    guard(args.arm, args)
    os.makedirs(args.out, exist_ok=True)
    snap = args.snap
    cfg = AutoConfig.from_pretrained(snap)
    receipt = {"arm": args.arm, "size": args.size, "prompt": g.PROMPT}
    if args.arm == "stock":
        from drinkme.arms import load_stock_streaming
        from drinkme.serving.checkpoint import tokenizer

        model = load_stock_streaming(args.model, None, args.device, snap=snap, vision=True)
        tok = tokenizer(snap, None)
        vis, gt, why = _vision_for(cfg, snap, args.model, tokenizer=lambda: tok)
        if vis is None:
            raise SystemExit(f"REFUSING: no image input on the stock arm ({why})")
        eng = HFEngine(model, tok, model_id=args.model, arm="stock", meta={}, ctx=args.ctx,
                       vision=vis, tower=gt)
    else:
        eng = load_compressed(args.model, None, args.pack, args.device, ctx=args.ctx)
        if eng.vision is None:
            raise SystemExit(f"REFUSING: no image input on the compressed arm "
                             f"({eng.vision_reason})")
        vis = eng.vision

    def img():
        return g.prepared(vis, args.size)

    t0 = time.perf_counter()
    ip, drow, res, greedy = g._engine_row(eng, img, args.device, args.n)
    receipt.update(ids=ip.ids, drinkme_row=drow, text=res.text, greedy=greedy,
                   prompt_tokens=res.prompt_tokens, drinkme_s=time.perf_counter() - t0)
    if args.arm == "stock":
        # transformers' own forward over the same weights and the same
        # expanded ids: its processor's pixel_values and grid
        pv, grid = processor_batch(img(), args.device)
        t = torch.tensor([ip.ids], device=args.device)
        with torch.inference_mode():
            ref = eng.model(input_ids=t, pixel_values=pv, image_grid_thw=grid, use_cache=False,
                            logits_to_keep=1).logits[0, -1]
        receipt["reference_row"] = ref.float().cpu()
    torch.save(receipt, os.path.join(args.out, f"{args.arm}.pt"))
    summary = {k: v for k, v in receipt.items() if k not in ("ids", "drinkme_row",
                                                            "reference_row", "greedy")}
    if "reference_row" in receipt:
        summary.update(g._rows(receipt["reference_row"], drow, "reference", "drinkme"))
    print("VERDICT " + json.dumps({"step": "row", **summary}), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="step", required=True)
    for name in ("tower", "row"):
        p = sub.add_parser(name)
        p.add_argument("--snap", required=True, help="the checkpoint snapshot directory")
        p.add_argument("--pack", help="a pack with a vision block (tower: arm (c))")
        p.add_argument("--size", default="1920x1080", help="the screenshot, WxH")
        p.add_argument("--device", default="cuda")
        p.add_argument("--need-gb", type=float, help="MemAvailable to insist on "
                       "(default: 12 for tower, 75 stock, 55 compressed)")
    tw = sub.choices["tower"]
    tw.add_argument("--max-pixels", type=int, default=2560 * 1440, help="the server's cap")
    tw.add_argument("--reps", type=int, default=4, help="rep 0 compiles and autotunes")
    tw.add_argument("--fp32-check", action="store_true",
                    help="also compare with the tower in float32 on the CPU (host RAM: 7.7 GB)")
    tw.add_argument("--no-fp32-tower", action="store_true",
                    help="with --fp32-check: every attention call against fp32 and each arm's "
                         "features' finiteness, without the CPU tower (runs at the cap)")
    rw = sub.choices["row"]
    rw.add_argument("--arm", choices=["stock", "compressed"], required=True)
    rw.add_argument("--model", required=True, help="the repo id (the engine's model_id)")
    rw.add_argument("--ctx", type=int, default=8192)
    rw.add_argument("--n", type=int, default=24, help="greedy tokens")
    rw.add_argument("--out", required=True)
    cp = sub.add_parser("compare")
    cp.add_argument("--out", required=True)
    tr = sub.add_parser("trace")
    tr.add_argument("dir")
    tr.add_argument("--verdict", help="a file holding a tower step's VERDICT line")
    tr.add_argument("--stop-ms", type=float, default=60.0)
    sm = sub.add_parser("smoke")
    sm.add_argument("--port", type=int, required=True)
    sm.add_argument("--served", default="Muse-Glimmer-30B", help="the model name to send")
    sm.add_argument("--size", default="2560x1440")
    sm.add_argument("--max-tokens", type=int, default=256)
    sm.add_argument("--system", default="Reasoning strength: low.",
                    help="the system turn (Glimmer's template reads its reasoning strength)")
    args = ap.parse_args()
    {"tower": tower, "row": row, "compare": compare, "trace": trace,
     "smoke": smoke}[args.step](args)


if __name__ == "__main__":
    main()
