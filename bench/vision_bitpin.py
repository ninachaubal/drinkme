"""The vision tower's bit-pin and the first token's logits row with an image
in the prompt, on the GPU: transformers' Qwen3_5ForConditionalGeneration
against drinkme's stock and compressed arms (docs/checks.md).

One synthetic screenshot (bench/vision_screens.py, 1920x1080 by default:
8,160 patches, so bounded attention splits it into two query blocks and the
unbounded call is one 24 ms dispatch on gfx1151, bench/vision_dispatch.py).
Each arm is its own process, run in turn, writing <out>/<arm>.pt:

  stock  engines.load_stock. The tower's features with DRINKME_VISION_BOUNDED=0
         (transformers' one-call attention) and then with vision.bound
         installed (the served attention); the first token's logits row
         from the engine's own generate() (prefill.run's return, spec off,
         no prefix slots), once per attention. Writes the prompt ids and
         the prepared image the other arms reuse.
  ref    transformers alone: Qwen3_5ForConditionalGeneration.from_pretrained,
         its model.get_image_features and its forward(input_ids,
         pixel_values, image_grid_thw, logits_to_keep=1) over the stock arm's
         ids and pixels (also checked against transformers' own processor).
         Two ops cannot run stock on gfx1151. The patch embedding's Conv3d:
         MIOpen's first call per shape runs a naive kernel for seconds, and
         with MIOpen off PyTorch loops over the patches at ~2.5 ms each, so
         the reference computes it with PyTorch's CPU Conv3d in fp32 and
         rounds to bf16 (MIOpen stays off for the rest of the process). The
         ViT's SDPA at head 72: AOTriton's kernels are wrong there
         (bench/vision_attention_check.py), so the ViT runs transformers'
         eager attention (longest dispatch 42 ms at 1920x1080). Then again
         with drinkme's two routes installed on transformers' model
         (vision.route_patch_embed, vision.bound): "ref_served".
  comp   engines.load_compressed from --pack: what the stock arm measures.
  compare  reads the three files and prints the verdict.

PASS: every feature and logit finite; ref_served == stock == comp byte-equal,
bounded and unbounded; the served tower's mean |error| against the same tower
in fp32 at most 1.5x transformers' own bf16 tower's (ref);
no two logits rows part on the first token with a gap above
agreement.NEAR_TIE_MARGIN (the rows come from different attention routes
and arms, so they agree or part at a near-tie; a near-tie part is reported);
ids and pixels agree with transformers' processor. Every number is printed
and kept in <out>/verdict.json.

Hold `flock -w 3600 /tmp/drinkme-gpu.lock` around each arm. Set
TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1, as the served units are.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from agreement import NEAR_TIE_MARGIN, above_margin, row_fork  # noqa: E402

MODELS = {"27b": ("Qwen/Qwen3.8-27B", "models--Qwen--Qwen3.8-27B"),
          "mimo": ("XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B",
                   "models--XiaomiMiMo--MiMo-V2.6-Distill-Qwen-9B")}
QUESTION = "What is the deploy token shown on the last output line of the terminal?"
CTX = 8192


def mem_gb() -> float:
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1e6
    return 0.0


def _snapshot(model: str) -> str:
    import glob

    hub = os.path.expanduser("~/.cache/huggingface/hub")
    return sorted(glob.glob(os.path.join(hub, MODELS[model][1], "snapshots", "*")))[-1]


def _env():
    os.environ.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")
    os.environ.update(DRINKME_SPEC="off", DRINKME_PREFIX_SLOTS="0", DRINKME_SLOT_DIR="off",
                      HF_HUB_OFFLINE="1", DRINKME_NO_AUTO_DEPS="1")


def _png(args) -> bytes:
    from vision_screens import png_bytes, screenshot

    w, h = (int(x) for x in args.size.lower().split("x"))
    return png_bytes(screenshot(w, h))


def _messages():
    return [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": QUESTION}]}]


def _engine_arm(args, eng) -> dict:
    """Features (unbounded, then bounded) and the logits row per attention,
    through the engine's own calls."""
    import torch

    from drinkme.serving import prefill, vision
    from drinkme.serving.engine import GenerationRequest, SampleParams, complete

    assert str(eng.device) != "cpu", "REFUSING to measure on the CPU"
    data = _png(args)

    def prepared():
        return eng.vision.prepare(vision.parse_base64(base64.b64encode(data).decode(),
                                                      "image/png", where="bitpin"))

    seen = {}
    orig = prefill.run

    def capture(model, ids, start, *a, **k):
        out = orig(model, ids, start, *a, **k)
        seen["ids"], seen["start"], seen["row"] = list(ids), start, out.float().cpu()
        return out

    prefill.run = capture
    res = {}
    try:
        for label in ("unbounded", "bounded"):
            if label == "bounded":
                os.environ.pop(vision.BOUNDED_ENV, None)
                assert vision.bound(eng._tower, eng.model) > 0
            img = prepared()
            with torch.inference_mode():
                feats = eng._tower.features(eng.model, img)
            torch.cuda.synchronize()
            res[f"features_{label}"] = feats.cpu()
            out = complete(eng, GenerationRequest(_messages(),
                                                  SampleParams(temperature=0.0, max_tokens=1),
                                                  images=(prepared(),)))
            assert seen.get("start") == 0, "the prompt did not prefill from 0"
            res[f"logits_{label}"] = seen["row"]
            res[f"token_{label}"] = out.text
        res["ids"] = seen["ids"]
        img = prepared()
        res["pixel_values"] = torch.from_numpy(img.pixel_values.copy())
        res["grid"] = list(img.grid_thw)
        res["digest"] = img.digest
    finally:
        prefill.run = orig
    res["patch_embed_routed"] = (eng.model.get_submodule(eng._tower.path).patch_embed.forward
                                 .__func__ is vision.linear_patch_embed)
    return res


def arm_stock(args):
    import torch

    from drinkme.serving import vision
    from drinkme.serving.engines import load_stock

    os.environ[vision.BOUNDED_ENV] = "0"  # constructed unbounded; bound() later
    t0 = time.time()
    eng = load_stock(MODELS[args.model][0], None, "cuda", ctx=CTX)
    res = _engine_arm(args, eng)
    res["load_s"] = time.time() - t0
    torch.save(res, os.path.join(args.out, "stock.pt"))
    print("VERDICT " + json.dumps({"arm": "stock", "load_s": res["load_s"], "grid": res["grid"],
                                   "prompt_tokens": len(res["ids"]),
                                   "tokens": [res["token_unbounded"], res["token_bounded"]]}),
          flush=True)


def arm_comp(args):
    import torch

    from drinkme.serving import vision
    from drinkme.serving.engines import load_compressed

    os.environ[vision.BOUNDED_ENV] = "0"
    t0 = time.time()
    eng = load_compressed(MODELS[args.model][0], None, args.pack, "cuda", ctx=CTX)
    res = _engine_arm(args, eng)
    res["load_s"] = time.time() - t0
    torch.save(res, os.path.join(args.out, "comp.pt"))
    print("VERDICT " + json.dumps({"arm": "comp", "load_s": res["load_s"], "grid": res["grid"],
                                   "prompt_tokens": len(res["ids"]),
                                   "tokens": [res["token_unbounded"], res["token_bounded"]]}),
          flush=True)


def arm_ref(args):
    import io
    import itertools
    import types

    import torch
    from PIL import Image
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration
    from transformers.models.qwen2_vl.image_processing_pil_qwen2_vl import (
        Qwen2VLImageProcessorPil)

    from drinkme.serving import vision

    torch.backends.cudnn.enabled = False  # no MIOpen: see the module docstring
    stock = torch.load(os.path.join(args.out, "stock.pt"))
    snap = _snapshot(args.model)
    res = {}
    # transformers' own processor over the same PNG: the ids and pixels the
    # engine computed must be its (drinkme's preprocessing is bit-pinned
    # against it on the CPU; this is the same claim on the real snapshot)
    # (the image processor and the tokenizer's template with
    # Qwen3VLProcessor.replace_image_token's expansion: AutoProcessor also
    # builds the video processor, which needs torchvision)
    ip = Qwen2VLImageProcessorPil.from_pretrained(snap)
    enc = ip(images=[Image.open(io.BytesIO(_png(args))).convert("RGB")], return_tensors="pt")
    tok = AutoTokenizer.from_pretrained(snap)
    text = tok.apply_chat_template(_messages(), add_generation_prompt=True, tokenize=False)
    n = int(enc["image_grid_thw"][0].prod()) // ip.merge_size ** 2
    text = text.replace("<|image_pad|>", "<|image_pad|>" * n)
    res["processor_pixels_equal"] = bool(torch.equal(enc["pixel_values"].float(),
                                                     stock["pixel_values"]))
    res["processor_grid"] = enc["image_grid_thw"][0].tolist()
    res["processor_ids_equal"] = (tok(text, add_special_tokens=False)["input_ids"]
                                  == stock["ids"])
    t0 = time.time()
    model = Qwen3_5ForConditionalGeneration.from_pretrained(snap, dtype=torch.bfloat16,
                                                            attn_implementation="sdpa")
    # engines.load_stock's clone-then-move: ROCm's H2D from mmap'd pages crawls
    for p in itertools.chain(model.parameters(), model.buffers()):
        p.data = p.data.clone().to("cuda")
    model = model.to("cuda").eval()
    res["load_s"] = time.time() - t0
    vit = model.model.visual
    # the ViT's attention in transformers' eager form (bf16 matmuls, fp32
    # softmax): its SDPA at head 72 is the AOTriton kernel that is wrong on
    # gfx1151 (bench/vision_attention_check.py)
    vit.config._attn_implementation = "eager"
    assert all(b.attn.config is vit.config for b in vit.blocks)
    pe = vit.patch_embed

    def cpu_conv(self, x):  # the stock Conv3d, on the CPU in fp32
        y = torch.nn.functional.conv3d(
            x.float().cpu().view(-1, self.in_channels, self.temporal_patch_size,
                                 self.patch_size, self.patch_size),
            self.proj.weight.float().cpu(), self.proj.bias.float().cpu(), stride=self.proj.stride)
        return y.view(-1, self.embed_dim).to(device=x.device, dtype=self.proj.weight.dtype)

    pe.forward = types.MethodType(cpu_conv, pe)
    ids = torch.tensor([stock["ids"]], device="cuda")
    mm = (ids == model.config.image_token_id).long()  # the processor's mm_token_type_ids
    pv = stock["pixel_values"].to("cuda")
    grid = torch.tensor([stock["grid"]], device="cuda")

    def run(label, logits=True):
        with torch.inference_mode():
            f = model.model.get_image_features(pv, grid).pooler_output[0]
            if logits:
                out = model(input_ids=ids, pixel_values=pv, image_grid_thw=grid,
                            mm_token_type_ids=mm, logits_to_keep=1)
        torch.cuda.synchronize()
        res[f"features_{label}"] = f.cpu()
        if logits:
            res[f"logits_{label}"] = out.logits[0, -1].float().cpu()

    run("ref")
    # transformers' model with drinkme's two routes installed (the patch
    # embedding's replaces the CPU conv), and nothing else of drinkme: its
    # tower must be drinkme's arms' bit for bit
    tower = vision.Tower("qwen3_5", "model.visual", int(model.config.image_token_id),
                         int(model.config.vision_config.spatial_merge_size))
    assert vision.route_patch_embed(tower, model) == 1
    os.environ[vision.BOUNDED_ENV] = "0"
    vision.bound(tower, model)
    run("ref_served_unbounded", logits=False)
    os.environ.pop(vision.BOUNDED_ENV)
    assert vision.bound(tower, model) > 0
    run("ref_served_bounded")  # MIOpen stays off: the DeltaNet conv1d would reach it
    # the same tower in fp32 (eager attention, the CPU conv): the yardstick
    # for "within bf16 rounding" (last: .float() is not undone)
    for b in vit.blocks:
        b.attn.config = vit.config  # the shared config: eager
    pe.forward = types.MethodType(cpu_conv, pe)
    vit.float()
    with torch.inference_mode():
        res["features_fp32"] = model.model.get_image_features(pv, grid).pooler_output[0].cpu()
    torch.save(res, os.path.join(args.out, "ref.pt"))
    print("VERDICT " + json.dumps({"arm": "ref", "load_s": res["load_s"],
                                   "processor_pixels_equal": res["processor_pixels_equal"],
                                   "processor_ids_equal": res["processor_ids_equal"]}), flush=True)


def _cmp(a, b) -> dict:
    import torch

    a, b = a.double(), b.double()
    d = (a - b).abs()
    return {"equal": bool(torch.equal(a, b)), "max_abs": float(d.max()),
            "mean_abs": float(d.mean()), "max_ref": float(a.abs().max()),
            "cos": float(torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0))}


def _row(r) -> dict:
    import torch

    top = torch.topk(r, 2)
    return {"argmax": int(top.indices[0]), "top2_gap": float(top.values[0] - top.values[1])}


def compare(args):
    import torch

    s = torch.load(os.path.join(args.out, "stock.pt"))
    c = torch.load(os.path.join(args.out, "comp.pt"))
    r = torch.load(os.path.join(args.out, "ref.pt"))
    v = {"model": args.model, "grid": s["grid"], "prompt_tokens": len(s["ids"]),
         "ids_equal_stock_comp": s["ids"] == c["ids"], "digest_equal": s["digest"] == c["digest"],
         "processor_pixels_equal": r["processor_pixels_equal"],
         "processor_ids_equal": r["processor_ids_equal"],
         "patch_embed_routed": [s["patch_embed_routed"], c["patch_embed_routed"]]}
    feats = {"ref": r["features_ref"],
             "ref_served_unbounded": r["features_ref_served_unbounded"],
             "ref_served_bounded": r["features_ref_served_bounded"],
             "stock_unbounded": s["features_unbounded"], "stock_bounded": s["features_bounded"],
             "comp_unbounded": c["features_unbounded"], "comp_bounded": c["features_bounded"]}
    v["features_finite"] = {k: bool(torch.isfinite(x.float()).all()) for k, x in feats.items()}
    v["features"] = {
        "ref_served_vs_stock_unbounded": _cmp(feats["ref_served_unbounded"],
                                              feats["stock_unbounded"]),
        "ref_served_vs_stock_bounded": _cmp(feats["ref_served_bounded"], feats["stock_bounded"]),
        "stock_vs_comp_unbounded": _cmp(feats["stock_unbounded"], feats["comp_unbounded"]),
        "stock_vs_comp_bounded": _cmp(feats["stock_bounded"], feats["comp_bounded"]),
        "stock_unbounded_vs_bounded": _cmp(feats["stock_unbounded"], feats["stock_bounded"]),
        "ref_vs_stock_bounded": _cmp(feats["ref"], feats["stock_bounded"]),
        "ref_vs_ref_served_bounded": _cmp(feats["ref"], feats["ref_served_bounded"]),
    }
    rows = {"ref": r["logits_ref"], "ref_served": r["logits_ref_served_bounded"],
            "stock_unbounded": s["logits_unbounded"], "stock_bounded": s["logits_bounded"],
            "comp_unbounded": c["logits_unbounded"], "comp_bounded": c["logits_bounded"]}
    v["logits_finite"] = {k: bool(torch.isfinite(x).all()) for k, x in rows.items()}
    v["logits"] = {k: _row(x) for k, x in rows.items()}
    v["logits_vs_ref"] = {k: _cmp(rows["ref"], x) for k, x in rows.items() if k != "ref"}
    v["logits_stock_vs_comp_bounded"] = _cmp(rows["stock_bounded"], rows["comp_bounded"])
    v["tokens"] = {"stock": [s["token_unbounded"], s["token_bounded"]],
                   "comp": [c["token_unbounded"], c["token_bounded"]]}
    f = v["features"]
    bits = all(f[k]["equal"] for k in ("ref_served_vs_stock_unbounded",
                                       "ref_served_vs_stock_bounded",
                                       "stock_vs_comp_unbounded", "stock_vs_comp_bounded"))
    # "within bf16 rounding": drinkme's served tower is no further from the
    # fp32 tower than transformers' own bf16 tower is (mean |error|, 1.5x)
    fp = r["features_fp32"]
    v["vs_fp32"] = {k: _cmp(fp, x) for k, x in feats.items()}
    near = (v["vs_fp32"]["stock_bounded"]["mean_abs"]
            <= 1.5 * v["vs_fp32"]["ref"]["mean_abs"])
    argmax = {x["argmax"] for x in v["logits"].values()}
    v["same_first_token"] = len(argmax) == 1
    v["first_token_parts"] = {}  # every pair of rows whose first tokens differ
    names = list(rows)
    for i, p in enumerate(names):
        for q in names[i + 1:]:
            f = row_fork(rows[p], rows[q])
            if not f["agree"]:
                v["first_token_parts"][f"{p}_vs_{q}"] = f
    v["near_tie_margin"] = NEAR_TIE_MARGIN
    v["pass"] = {"finite": all(v["features_finite"].values()) and all(v["logits_finite"].values()),
                 "tower_bits": bits, "tower_near_reference": near,
                 "first_token_agrees_or_near_tie": not any(above_margin(f) for f in
                                                           v["first_token_parts"].values()),
                 "ids_and_pixels": v["ids_equal_stock_comp"] and v["processor_pixels_equal"]
                 and v["processor_ids_equal"] and all(v["patch_embed_routed"])}
    v["verdict"] = "PASS" if all(v["pass"].values()) else "FAIL"
    with open(os.path.join(args.out, "verdict.json"), "w") as fh:
        json.dump(v, fh, indent=1)
    print(json.dumps(v, indent=1))
    print(f"VERDICT {v['verdict']} {json.dumps(v['pass'])}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("arm", choices=["stock", "ref", "comp", "compare"])
    ap.add_argument("--model", choices=sorted(MODELS), default="27b")
    ap.add_argument("--pack", help="comp: the pack directory (with a vision block)")
    ap.add_argument("--size", default="1920x1080")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    _env()
    if args.arm != "compare" and mem_gb() < (60 if args.model == "27b" else 25):
        raise SystemExit(f"REFUSING: MemAvailable {mem_gb():.1f} GB")
    {"stock": arm_stock, "ref": arm_ref, "comp": arm_comp, "compare": compare}[args.arm](args)
    return 0


if __name__ == "__main__":
    main()
