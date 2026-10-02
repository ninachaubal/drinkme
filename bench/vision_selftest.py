"""The vision tower's boot self-test (vision.attention_self_test), alone, on
this device, for a checkpoint's ViT: what HFEngine runs when it builds a
tower, without loading a model. The self-test reads only the attention
modules' shape (heads, head width, scaling, kv groups), so the ViT is built
from the snapshot's config.json with one layer and uninitialized weights
(about 100 MB bf16); no weight is read.

  --no-pad   turns ROCm's head padding off (vision.HEAD_ALIGN = 1). On
             gfx1151 at head width 72 (Qwen3.5, gemma-4) that is AOTriton's
             wrong kernel, so the check should FAIL: the negative control
             that shows the boot check can see the failure it exists for.
             An AGREES here means this wheel's kernel is right at 72 at the
             self-test's shape (SELFTEST_ROWS x SELFTEST_KEYS), or that the
             shape does not reproduce the fault; either way, say so.

One `VERDICT {json}` line: verdict PASS when the check agrees. Read it, not
the exit status. Hold `flock -w 3600 /tmp/drinkme-gpu.lock`; every dispatch
is at most SELFTEST_ROWS x SELFTEST_KEYS query-key pairs per head (8.4M,
a quarter of a bounded tower dispatch).
"""
from __future__ import annotations

import argparse
import copy
import json


def run(snap: str, device: str, no_pad: bool = False, rows: int | None = None,
        keys: int | None = None) -> dict:
    import torch
    from transformers import AutoConfig, AutoModel

    from drinkme.serving import vision

    cfg = AutoConfig.from_pretrained(snap)
    vc = copy.deepcopy(cfg.vision_config)
    for k in ("depth", "num_hidden_layers"):  # Qwen3.5's name, then the others'
        if getattr(vc, k, None):
            setattr(vc, k, 1)
    with torch.device("meta"):
        vit = AutoModel.from_config(vc, dtype=torch.bfloat16)
    root = torch.nn.Module()
    root.vit = vit.to_empty(device=device).eval()
    tower = vision.Tower(cfg.model_type, "vit", 0, 1)
    routed = vision.bound(tower, root)
    align = vision.HEAD_ALIGN
    if no_pad:
        vision.HEAD_ALIGN = 1
    try:
        check = vision.attention_self_test(tower, root, rows=rows, keys=keys)
    finally:
        vision.HEAD_ALIGN = align
    res = {"step": "selftest", "model_type": cfg.model_type, "device": device,
           "no_pad": no_pad, "routed_modules": routed,
           "torch": torch.__version__, "hip": torch.version.hip}
    if check is None:
        return {**res, "verdict": "NOT CHECKED",
                "why": "no accelerator, or no routed attention module"}
    return {**res, "verdict": "PASS" if check.ok else "FAIL", "shape": check.shape,
            "error": check.error, "lines": check.lines(),
            "tests": {form: t.to_dict() for form, t in check.tests}}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--snap", required=True, help="the checkpoint snapshot (its config.json)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-pad", action="store_true", help="vision.HEAD_ALIGN = 1")
    args = ap.parse_args()
    print("VERDICT " + json.dumps(run(args.snap, args.device, args.no_pad)), flush=True)


if __name__ == "__main__":
    main()
