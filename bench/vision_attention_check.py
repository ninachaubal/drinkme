"""Is the vision tower's attention right on this GPU? Each SDPA backend at the
ViT's head width, on the tower's own activations, against an fp32 reference.

AOTriton on gfx1151 (TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1, TheRock
ROCm 7.13) returned wrong values at the Qwen3.5 ViT's head width
of 72: flash was off by up to 1,240 on outputs under 13 in magnitude and
mem-efficient returned NaN, so the whole tower's output was NaN.
serving/vision.bounded_attention zero-pads the head to a multiple of
vision.HEAD_ALIGN on ROCm devices. This instrument re-measures that on any
box, for any wheel.

The stock tower (bench/vision_dispatch.build) takes one synthetic screenshot
(--size, default 640x480). Every ViT attention call is intercepted and
computed six ways on the same q, k, v: flash, mem-efficient and math at the
head's own width, flash with the head padded to 80 and 128, and the served
function (vision.bounded_attention). Each is compared with an fp32
reference (softmax(q k^T s) v in float32). The reference's output feeds the
next layer, so every layer sees the activations the tower really makes.
A kernel is "right" at a layer when its max |error| is at most 2x the
reference's own bf16 rounding error. The error it makes on that layer's
inputs is attributed to it alone.

VERDICT PASS: the served function is right at every layer. The other rows
are printed, per layer and as a summary, so a wheel that fixes (or breaks)
a backend shows up here.

Hold `flock -w 3600 /tmp/drinkme-gpu.lock`. One 640x480 image; every
dispatch is the size bench/vision_dispatch.py measured.
"""
from __future__ import annotations

import argparse
import base64
import copy
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="27b")
    ap.add_argument("--size", default="640x480")
    ap.add_argument("--out", help="write the per-layer table here (JSON)")
    args = ap.parse_args()

    import torch
    import torch.nn.functional as F
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    import vision_dispatch as vd
    from drinkme.serving import vision
    from vision_screens import png_bytes, screenshot

    assert torch.cuda.is_available(), "REFUSING: no accelerator"
    snap = vd.snapshot(args.model)
    root, cfg, path, _ = vd.build(snap, None, "cuda")
    tower = vision.Tower(cfg.model_type, path, int(cfg.image_token_id),
                         int(cfg.vision_config.spatial_merge_size))
    vision.route_patch_embed(tower, root)
    vis = vision.load(cfg.model_type, snap)
    w, h = (int(x) for x in args.size.lower().split("x"))
    img = vis.prepare(vision.parse_base64(base64.b64encode(png_bytes(screenshot(w, h))).decode(),
                                          "image/png", where="check"))
    rows = []

    def ref32(q, k, v, scale):
        out = torch.empty(q.shape, dtype=torch.float32, device=q.device)
        for i in range(0, q.shape[2], 1024):
            s = (q[:, :, i:i + 1024].float() @ k.float().transpose(-1, -2)) * scale
            out[:, :, i:i + 1024] = torch.softmax(s, -1) @ v.float()
        return out

    def err(o, r):
        o = o.float()
        return float("nan") if torch.isnan(o).any() else float((o - r).abs().max())

    def probe(module, q, k, v, attention_mask=None, dropout=0.0, scaling=None, is_causal=False,
              **kw):
        d = q.shape[-1]
        scale = scaling if scaling is not None else 1 / math.sqrt(d)
        r = ref32(q, k, v, scale)
        rec = {"layer": len(rows), "head": d, "rounding": float((r.bfloat16().float() - r).abs().max())}
        for name, be in (("flash", SDPBackend.FLASH_ATTENTION),
                         ("efficient", SDPBackend.EFFICIENT_ATTENTION), ("math", SDPBackend.MATH)):
            try:
                with sdpa_kernel([be]):
                    rec[name] = err(F.scaled_dot_product_attention(q, k, v, scale=scale), r)
            except RuntimeError as e:  # a backend that refuses the shape
                rec[name] = f"refused: {str(e)[:60]}"
        for dp in (80, 128):
            pad = (0, dp - d)
            with sdpa_kernel([SDPBackend.FLASH_ATTENTION]):
                o = F.scaled_dot_product_attention(F.pad(q, pad), F.pad(k, pad), F.pad(v, pad),
                                                   scale=scale)[..., :d]
            rec[f"flash_pad{dp}"] = err(o, r)
        o, _ = vision.bounded_attention(module, q, k, v, attention_mask, dropout=dropout,
                                        scaling=scaling, is_causal=is_causal, **kw)
        rec["served"] = err(o.transpose(1, 2), r)
        rows.append(rec)
        return r.to(q.dtype).transpose(1, 2).contiguous(), None

    ALL_ATTENTION_FUNCTIONS.register("drinkme_vision_check", probe)
    for m in root.model.visual.modules():
        if type(m).__name__ in vision._BOUNDED_TYPES[cfg.model_type]:
            m.config = copy.copy(m.config)
            m.config._attn_implementation = "drinkme_vision_check"
    with torch.inference_mode():
        tower.features(root, img)
    torch.cuda.synchronize()

    kinds = ["flash", "efficient", "math", "flash_pad80", "flash_pad128", "served"]

    def right(rec, kind):
        x = rec[kind]
        return isinstance(x, float) and x == x and x <= 2 * rec["rounding"]

    print(f"{'layer':>5} {'bf16 rnd':>9} " + " ".join(f"{k:>13}" for k in kinds))
    for rec in rows:
        print(f"{rec['layer']:>5} {rec['rounding']:>9.4f} "
              + " ".join(f"{rec[k]:>13.4f}" if isinstance(rec[k], float) else f"{'refused':>13}"
                         for k in kinds))
    summary = {k: sum(right(r, k) for r in rows) for k in kinds}
    print("layers right (max |err| <= 2x bf16 rounding): "
          + ", ".join(f"{k} {v}/{len(rows)}" for k, v in summary.items()))
    verdict = "PASS" if summary["served"] == len(rows) else "FAIL"
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"model": args.model, "size": args.size, "grid": list(img.grid_thw),
                       "torch": torch.__version__, "rows": rows, "summary": summary,
                       "verdict": verdict}, f, indent=1)
    print(f"VERDICT {verdict} served attention right at {summary['served']}/{len(rows)} layers",
          flush=True)


if __name__ == "__main__":
    main()
