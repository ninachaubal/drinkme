"""GPU acceptance for gemma-4 image input, for gemma-4-31B-it: the tower's
bit-pin, the first-token logits row, a screenshot over the wire and the
dispatch gate, one process per step. Each step prints one
`VERDICT {json}` line; read that, not the exit status (TheRock ROCm torch
can exit 0 after a failure).

  tower    the ViT bit-pin, and the dispatch gate's worker
           under rocprofv3. The served tower ALONE (model.vision_tower and
           model.embed_vision, ~1.2 GB bf16), streamed from the snapshot by
           the served walker, and with --pack a second copy whose Linears
           come from the pack (swap.make_module over a hard-linked subset of
           its tower tensors; nothing is written into the pack) and the rest
           from its embedded checkpoint. One synthetic screenshot through
           the served preprocessing at --max-pixels, then --reps times each:
             (a) transformers' own Gemma4Model.get_image_features over the
                 stock arm's tree with drinkme's attention route installed
                 (vision.bound, as HFEngine installs it, on every arm before
                 any run): bench/vision_bitpin.py's "ref_served". Without
                 it (a) would be the stock SDPA at the ViT's head width, 72,
                 which on gfx1151 is AOTriton's wrong kernel
                 (serving/vision.py HEAD_ALIGN);
             (b) drinkme's Gemma4Tower.features on the same tree,
             (c) (b) over the pack's Linears.
           PASS when a == b (== c) byte for byte. That cannot see a wrong
           attention kernel (every arm calls the same one), so --fp32-check
           runs each arm once more, untimed, and holds every ViT attention
           call it makes against fp32 matmul and softmax on that call's own
           inputs, with the boot self-test's tolerances
           (attention_calls, vision.attention_error); and it runs the
           stock tower in float32 on the CPU with transformers' eager
           attention, whose finiteness is judged and whose row cosines are
           printed (fp32_rows). fp32_verdict PASS when every call of every
           arm is within tolerance and every arm is finite (fp32_verdict).
           gemma resizes every image
           to its soft-token budget, so the dispatch ladder is over budgets,
           not image sizes: --max-pixels 161280 / 322560 / the default cap
           give 70 / 140 / 280 tokens (630 / 1,260 / 2,520 padded patches).
  row      the first-token logits row of an image prompt and a
           short greedy answer. --arm stock streams the transformers class
           (arms.load_stock_streaming, bitwise from_pretrained with one
           tensor of host transient; never engines.load_stock, whose
           from_pretrained held ~2x the 31B on unified memory, fit.py) and
           runs transformers' own forward(input_ids, pixel_values,
           image_position_ids, mm_token_type_ids), then drinkme's engine
           over the same tree. --arm compressed loads the pack through
           engines.load_compressed. Each writes <out>/<arm>.pt.
  compare  <out>/stock.pt against <out>/compressed.pt: one first-token
           argmax, and greedy answers equal or parting at a near-tie.
  trace    the longest dispatches in a rocprofv3 --kernel-trace directory,
           per kernel family: every dispatch of the process, and those
           inside the forwards' windows when --verdict names a tower run's
           output.
  smoke    a screenshot carrying a nonce, sent to a running
           `drinkme serve` on --port as a data URL (/v1/chat/completions)
           and as a base64 block (/v1/messages); PASS when both replies
           carry the nonce.

Hold `flock -w 3600 /tmp/drinkme-gpu.lock` around every GPU step.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

TOWER = ("model.vision_tower", "model.embed_vision")
PROMPT = "What does this screenshot show? Answer in one sentence."
NEED_GB = {"tower": 8, "stock": 75, "compressed": 55}


def mem_available_gb() -> float:
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1e6
    return 0.0


def guard(kind: str, args) -> None:
    import torch

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("REFUSING: --device cuda and no accelerator")
    need = NEED_GB[kind] if args.need_gb is None else args.need_gb
    if mem_available_gb() < need:
        raise SystemExit(f"REFUSING: MemAvailable {mem_available_gb():.1f} GB < "
                         f"{need} GB for a {kind} step")


def sync(device: str) -> None:
    import torch

    if device == "cuda":
        torch.cuda.synchronize()


# --------------------------------------------------------- the picture --

def screenshot(w: int, h: int, nonce: str = "PELICAN-7391"):
    """A dark terminal pane of monospace lines, one carrying `nonce`, and a
    light panel with a small table: drawn here, nothing downloaded."""
    from PIL import Image, ImageDraw, ImageFont

    path = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
    px = max(12, h // 40)
    font = (ImageFont.truetype(path, px) if os.path.exists(path)
            else ImageFont.load_default(size=px))
    im = Image.new("RGB", (w, h), (30, 30, 36))
    d = ImageDraw.Draw(im)
    lines = ["$ drinkme serve --model gemma-4-31B-it --port 3299",
             "[drinkme] image input: gemma4 tower at model.vision_tower",
             "$ cat /etc/motd", f"CODE: {nonce}", "$ uptime",
             " 14:07:11 up 3 days,  2:41,  1 user,  load average: 0.42"]
    for i, line in enumerate(lines):
        d.text((px, px + i * int(px * 1.5)), line, fill=(220, 220, 220), font=font)
    x0, y0 = w * 3 // 5, h // 2
    d.rectangle([x0, y0, w - px, h - px], fill=(240, 240, 244))
    for r, row in enumerate([("model", "tokens"), ("gemma-4", "280"), ("qwen", "3600")]):
        for c, cell in enumerate(row):
            d.text((x0 + px + c * 8 * px, y0 + px + r * 2 * px), cell, fill=(10, 10, 10),
                   font=font)
    return im


def png_bytes(im) -> bytes:
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()


def prepared(vis, size: str, nonce: str = "PELICAN-7391"):
    from drinkme.serving import vision

    w, h = (int(x) for x in size.lower().split("x"))
    data = base64.b64encode(png_bytes(screenshot(w, h, nonce))).decode()
    return vis.prepare(vision.parse_base64(data, "image/png", where="gate"))


def processor_batch(img, device):
    """The image as transformers' processor hands it to the model: a batch
    of one padded pixel_values [1, patches, 768] float32 and its positions
    [1, patches, 2] (vision.patch_positions; bit-pinned against
    Gemma4ImageProcessorPil in tests/test_serving_gemma_vision.py)."""
    import numpy as np
    import torch

    from drinkme.serving import vision

    pv = torch.from_numpy(np.array(img.pixel_values))[None].to(device)
    pos = torch.from_numpy(vision.patch_positions(img.grid_thw, pv.shape[1]))[None].to(device)
    return pv, pos


def digest(t) -> str:
    return hashlib.sha256(t.detach().float().cpu().contiguous().numpy().tobytes()).hexdigest()


# --------------------------------------------------------------- tower --

def _subset_pack(pack: str, tmp: str) -> str:
    """The pack's tower tensors, hard-linked into tmp with a rewritten
    meta.json. Read-only on the pack."""
    meta = json.load(open(os.path.join(pack, "meta.json")))
    keep = {k: v for k, v in meta["tensors"].items()
            if k.startswith(tuple(p + "." for p in TOWER))}
    if not keep or not meta.get("vision"):
        raise SystemExit(f"{pack}: no vision block or tower tensors (re-pack it)")
    for fn in set(keep.values()):
        os.link(os.path.join(pack, fn), os.path.join(tmp, fn))
    json.dump(dict(meta, tensors=keep), open(os.path.join(tmp, "meta.json"), "w"))
    return tmp


def tower_root(snap: str, cfg, device: str, pack: str | None = None):
    """A root holding only gemma's tower, at the paths the served tree has
    it, built empty and filled by the served walker (stream_checkpoint)."""
    import torch
    from transformers import AutoModel
    from transformers.models.gemma4.modeling_gemma4 import Gemma4MultimodalEmbedder

    from drinkme.serving.engines import stream_checkpoint

    root = torch.nn.Module()
    root.model = torch.nn.Module()
    root.tie_weights = lambda: None  # nothing is tied here
    with torch.device("meta"):
        root.model.vision_tower = AutoModel.from_config(cfg.vision_config,
                                                        dtype=torch.bfloat16).eval()
        root.model.embed_vision = Gemma4MultimodalEmbedder(
            cfg.vision_config, cfg.text_config).to(torch.bfloat16).eval()
    raw_from = snap
    if pack:
        from drinkme.codec.pack import embedded_dir, iter_pack_dir
        from drinkme.codec.swap import make_module

        scratch = os.path.dirname(os.path.abspath(pack))  # the pack's filesystem
        with tempfile.TemporaryDirectory(dir=scratch, prefix="gemma-gate-") as tmp:
            for name, p in iter_pack_dir(_subset_pack(pack, tmp)):
                parent, _, child = name.rpartition(".")
                assert isinstance(root.get_submodule(name), torch.nn.Linear), name
                setattr(root.get_submodule(parent), child, make_module(p, None, device))
        if os.path.isdir(embedded_dir(pack)):
            raw_from = embedded_dir(pack)
    stream_checkpoint(root, raw_from, cfg, device, only=tuple(p + "." for p in TOWER))
    root.model.config = cfg  # what Gemma4Model.get_image_features reads
    return root


def attention_calls(run) -> list:
    """Every call the routed ViT attention (vision.bound's functions) makes
    while run() runs, each one's output against fp32 on its own inputs
    (vision.attention_error, the boot self-test's convention and
    tolerances): one (masked, sdpa.SelfTest) per call, masked when the call
    carried an attention mask (on ROCm a masked call takes mem-efficient,
    an unmasked one flash). The fp32 reference is matmul and softmax on the
    same device, in 256-row blocks."""
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    from drinkme.serving import vision

    tests = []
    names = (vision.BOUNDED_NAME, vision.UNBOUNDED_NAME)
    real = {n: ALL_ATTENTION_FUNCTIONS[n] for n in names}

    def spy(name):
        def call(module, query, key, value, attention_mask=None, *a, **kw):
            out = real[name](module, query, key, value, attention_mask, *a, **kw)
            tests.append((attention_mask is not None, vision.attention_error(
                out[0].transpose(1, 2), query, key, value, attention_mask, kw.get("scaling"),
                name)))
            return out
        return call

    for n in names:
        ALL_ATTENTION_FUNCTIONS.register(n, spy(n))
    try:
        run()
    finally:
        for n in names:
            ALL_ATTENTION_FUNCTIONS.register(n, real[n])
    return tests


def fp32_rows(got: dict, want) -> dict:
    """Each arm's features in `got` against `want`, the stock tower's in
    float32 on the CPU with transformers' eager attention: whether every
    value is finite, and, as diagnostics only, the row cosines and the
    absolute differences. Cosine is scale-invariant (a uniform x3 passes
    it), and a whole bf16 tower is not within one kernel's tolerance of
    fp32 (the served Qwen3.8-27B tower's mean |error| measured 0.00685 on
    gfx1151, over sdpa.MEAN_TOL), so the verdict is fp32_verdict's."""
    import torch

    rows = {}
    for arm, out in got.items():
        o = out.float().cpu()
        cos = torch.nn.functional.cosine_similarity(o, want, dim=-1)
        rows[arm] = {"finite": bool(torch.isfinite(o).all()),
                     "min_row_cosine": float(cos.min()), "mean_row_cosine": float(cos.mean()),
                     "max_abs_diff": float((o - want).abs().max()),
                     "mean_abs_diff": float((o - want).abs().mean())}
    return rows


def fp32_verdict(rows: dict, calls: dict) -> dict:
    """fp32_verdict PASS when every arm's features are finite (fp32_rows)
    and every attention call each arm made is within the boot self-test's
    tolerance of fp32 on its own inputs (attention_calls), with at least
    one call per arm (a tower whose attention was not routed checks
    nothing). Per arm: the calls, how many carried a mask, how many
    failed, and the worst, as a multiple of its tolerance."""
    import math

    def over(t):  # the larger of the two errors over its tolerance; NaN is worst
        r = max(t.max_abs / t.tol_max, t.mean_abs / t.tol_mean)
        return r if r == r else math.inf

    summary = {}
    for arm, pairs in calls.items():
        tests = [t for _masked, t in pairs]
        worst = max(tests, key=over, default=None)
        summary[arm] = {"calls": len(tests), "masked": sum(m for m, _t in pairs),
                        "failed": sum(not t.ok for t in tests),
                        "worst": worst.to_dict() if worst else None,
                        "worst_over_tolerance": over(worst) if worst else None}
    ok = all(r["finite"] for r in rows.values()) and all(
        s["calls"] and not s["failed"] for s in summary.values())
    return {"fp32_verdict": "PASS" if ok else "FAIL", "attention_vs_fp32": summary,
            "vs_fp32_cpu": rows}


def fp32_check(snap: str, cfg, img, got: dict, runs: dict) -> dict:
    """fp32_verdict: each arm's attention calls in one more forward
    (`runs`, untimed) against fp32, and each arm's features (`got`)
    against gemma's stock tower in float32 on the CPU, from the same
    snapshot (host RAM: about 2.3 GB for the 31B's tower)."""
    import torch
    from transformers.models.gemma4.modeling_gemma4 import Gemma4Model

    with torch.inference_mode():
        calls = {arm: attention_calls(run) for arm, run in runs.items()}
    root = tower_root(snap, cfg, "cpu").float()
    root.model.vision_tower.config._attn_implementation = "eager"
    pv, pos = processor_batch(img, "cpu")
    with torch.inference_mode():
        want = Gemma4Model.get_image_features(root.model, pv, pos).pooler_output[0]
    del root
    return fp32_verdict(fp32_rows(got, want), calls)


def tower(args) -> None:
    import torch
    from transformers import AutoConfig
    from transformers.models.gemma4.modeling_gemma4 import Gemma4Model

    from drinkme.arms import vision_tower_paths
    from drinkme.serving import vision
    from drinkme.serving.kernel_route import route_kernels

    guard("tower", args)
    snap = args.snap
    cfg = AutoConfig.from_pretrained(snap)
    vis = vision.load(cfg.model_type, snap, max_pixels=args.max_pixels)
    gt = vision.tower_for(cfg.model_type, cfg, vision_tower_paths(cfg))
    arms = {"stock": tower_root(snap, cfg, args.device)}
    if args.pack:
        arms["compressed"] = tower_root(snap, cfg, args.device, args.pack)
    for root in arms.values():
        route_kernels(root, args.device)
        # before (a): transformers' forward takes drinkme's attention too
        # (the module docstring's ref_served)
        routed = vision.bound(gt, root)
    img = prepared(vis, args.size)
    pv, pos = processor_batch(img, args.device)
    out, walls, windows = {}, [], []

    def timed(fn):
        res = None
        fresh = [prepared(vis, args.size) for _ in range(args.reps)]  # the tower releases each
        for i in range(args.reps):
            sync(args.device)
            t0, m0 = time.perf_counter(), time.monotonic_ns()
            with torch.inference_mode():
                res = fn(fresh[i])
            sync(args.device)
            walls.append(time.perf_counter() - t0)
            windows.append((m0, time.monotonic_ns()))
        return res

    fns = {"reference": lambda _img: Gemma4Model.get_image_features(
        arms["stock"].model, pv, pos).pooler_output[0]}
    for arm, root in arms.items():
        fns[arm] = lambda im, root=root: gt.features(root, im)
    for arm, fn in fns.items():
        out[arm] = timed(fn)
    sha = {k: digest(v) for k, v in out.items()}
    ok = len(set(sha.values())) == 1
    diff = {k: float((v.float() - out["reference"].float()).abs().max()) for k, v in out.items()}
    checked = fp32_check(snap, cfg, img, out, {
        arm: lambda fn=fn: fn(prepared(vis, args.size)) for arm, fn in fns.items()}
    ) if args.fp32_check else {}
    print("VERDICT " + json.dumps({**checked,
        "step": "tower", "verdict": "PASS" if ok else "FAIL", "size": args.size,
        "max_pixels": args.max_pixels, "resized": list(img.size), "grid": list(img.grid_thw),
        "tokens": img.tokens, "padded_patches": int(pv.shape[1]),
        "pairs_per_attention_call": int(pv.shape[1]) ** 2, "work_v": vision.WORK_V,
        "bounded_modules": routed, "sha256": sha, "max_abs_diff_vs_reference": diff,
        "wall_s": walls, "windows": windows, "device": args.device,
        "peak_gib": (torch.cuda.max_memory_allocated() / 2**30
                     if args.device == "cuda" else None)}), flush=True)


# ----------------------------------------------------------------- row --

def _request(n: int):
    from drinkme.serving.engine import GenerationRequest, SampleParams

    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]}]
    return msgs, lambda img: GenerationRequest(
        msgs, SampleParams(temperature=0.0, max_tokens=n), images=(img,))


TOP_K = 8  # logits kept per greedy position, for compare's near-tie rule


def _engine_row(eng, img, device, n):
    """drinkme's first-token row (the engine's serial prefill, its own
    chunking), its greedy answer, and each emitted position's pick with
    its row's TOP_K ids and logits (taken at engines.sample_next, the seam
    every decode path shares: tests/spec_agree.py's capture), over a fresh
    copy of the image each."""
    import torch

    from drinkme.serving import engines, prefill
    from drinkme.serving.engine import complete

    _msgs, req = _request(n)
    ip = eng._image_prompt(req(img()), eng._render(req(img())).ids)
    with torch.inference_mode():
        row = prefill.run(eng.model, ip.ids, 0, eng._cache(ip.n + n), device,
                          eng._prefill_chunk, image=ip)
    picks, real = {}, engines.sample_next

    def rec(logits, params, generator=None, prev_ids=None, gen_ids=None, **kw):
        t = real(logits, params, generator, prev_ids=prev_ids, gen_ids=gen_ids, **kw)
        top = torch.topk(logits.detach().float().reshape(-1), TOP_K)
        picks[len(gen_ids) if gen_ids is not None else len(picks)] = (
            int(t), top.indices.tolist(), top.values.tolist())
        return t

    engines.sample_next = rec
    try:
        res = complete(eng, req(img()))
    finally:
        engines.sample_next = real
    return ip, row.float().cpu(), res, [picks[i] for i in sorted(picks)]


def row(args) -> None:
    import torch
    from transformers import AutoConfig

    from drinkme.serving import vision
    from drinkme.serving.engines import HFEngine, _vision_for, load_compressed

    guard(args.arm, args)
    os.makedirs(args.out, exist_ok=True)
    snap = args.snap
    cfg = AutoConfig.from_pretrained(snap)
    receipt = {"arm": args.arm, "size": args.size, "prompt": PROMPT}
    if args.arm == "stock":
        from drinkme.arms import load_stock_streaming
        from drinkme.serving.checkpoint import tokenizer

        model = load_stock_streaming(args.model, None, args.device, snap=snap)
        vis, gt, why = _vision_for(cfg, snap, args.model)
        if vis is None:
            raise SystemExit(f"REFUSING: no image input on the stock arm ({why})")
        eng = HFEngine(model, tokenizer(snap, None), model_id=args.model, arm="stock",
                       meta={}, ctx=args.ctx, vision=vis, tower=gt)
    else:
        eng = load_compressed(args.model, None, args.pack, args.device, ctx=args.ctx)
        if eng.vision is None:
            raise SystemExit(f"REFUSING: no image input on the compressed arm "
                             f"({eng.vision_reason})")
        vis = eng.vision

    def img():
        return prepared(vis, args.size)

    t0 = time.perf_counter()
    ip, drow, res, greedy = _engine_row(eng, img, args.device, args.n)
    receipt.update(ids=ip.ids, drinkme_row=drow, text=res.text, greedy=greedy,
                   prompt_tokens=res.prompt_tokens, drinkme_s=time.perf_counter() - t0)
    if args.arm == "stock":
        # transformers' own forward over the same weights and the same
        # expanded ids: its processor's pixel batch, its mm_token_type_ids
        pv, pos = processor_batch(img(), args.device)
        t = torch.tensor([ip.ids], device=args.device)
        with torch.inference_mode():
            ref = eng.model(input_ids=t, pixel_values=pv, image_position_ids=pos,
                            mm_token_type_ids=(t == cfg.image_token_id).int(),
                            use_cache=False, logits_to_keep=1).logits[0, -1]
        receipt["reference_row"] = ref.float().cpu()
    torch.save(receipt, os.path.join(args.out, f"{args.arm}.pt"))
    summary = {k: v for k, v in receipt.items() if k not in ("ids", "drinkme_row",
                                                            "reference_row", "greedy")}
    if "reference_row" in receipt:
        summary.update(_rows(receipt["reference_row"], drow, "reference", "drinkme"))
    print("VERDICT " + json.dumps({"step": "row", **summary}), flush=True)


def _rows(a, b, na: str, nb: str) -> dict:
    import torch

    top = lambda r: set(torch.topk(r, 5).indices.tolist())  # noqa: E731
    return {f"{na}_vs_{nb}": {"bytes_equal": bool(torch.equal(a, b)),
                              "max_abs_diff": float((a - b).abs().max()),
                              "argmax": [int(a.argmax()), int(b.argmax())],
                              "top5_overlap": len(top(a) & top(b))}}


def bf16_step(x: float) -> float:
    """One bf16 step (ulp: 8 significant bits) at magnitude |x|, at least 1."""
    import math

    return 2.0 ** (math.floor(math.log2(max(abs(x), 1.0))) - 7)


def near_tie(s: dict, c: dict) -> dict:
    """Where two arms' greedy picks (their receipts' `greedy`) first differ,
    and whether that is a near-tie: on EACH arm the two picks' logits there
    are at most one bf16 step apart at that arm's top logit, i.e. equal or
    adjacent bf16 values. Two arms over the same weights round differently
    (the compressed arm's GEMMs accumulate in another order,
    docs/serve-kernels.md), so they may pick differently only where the row
    cannot tell the two apart; a pick outside the other arm's TOP_K is no
    near-tie. first_difference None: the picks agree."""
    sp, cp = ([p[0] for p in r.get("greedy", ())] for r in (s, c))
    i = next((k for k, (a, b) in enumerate(zip(sp, cp)) if a != b), None)
    if i is None:
        return {"first_difference": None, "near_tie": None}
    a, b = sp[i], cp[i]
    out, ok = {"first_difference": i, "tokens": [a, b]}, True
    for arm, r in (("stock", s), ("compressed", c)):
        _pick, ids, vals = r["greedy"][i]
        top = dict(zip(ids, vals))
        margin = abs(top[a] - top[b]) if a in top and b in top else None
        out[arm] = {"margin": margin, "bf16_step": bf16_step(vals[0])}
        ok = ok and margin is not None and margin <= bf16_step(vals[0])
    return {**out, "near_tie": ok}


def compare(args) -> None:
    """PASS when both arms saw the same prompt, the three first-token rows
    (transformers', stock, compressed) have one argmax, and the greedy
    answers are equal or first part at a near_tie (AGENTS.md: greedy
    transcripts can differ at near-ties)."""
    import torch

    s = torch.load(os.path.join(args.out, "stock.pt"))
    c = torch.load(os.path.join(args.out, "compressed.pt"))
    same_prompt = s["ids"] == c["ids"]
    res = {**_rows(s["reference_row"], s["drinkme_row"], "reference", "stock"),
           **_rows(s["drinkme_row"], c["drinkme_row"], "stock", "compressed")}
    argmax = {int(s["reference_row"].argmax()), int(s["drinkme_row"].argmax()),
              int(c["drinkme_row"].argmax())}
    equal = s["text"] == c["text"]
    tie = near_tie(s, c)
    ok = same_prompt and len(argmax) == 1 and (equal or bool(tie["near_tie"]))
    print("VERDICT " + json.dumps({
        "step": "compare", "verdict": "PASS" if ok else "FAIL", "same_prompt": same_prompt,
        "prompt_tokens": [s["prompt_tokens"], c["prompt_tokens"]],
        "greedy_equal": equal, "greedy_divergence": tie,
        "text": {"stock": s["text"], "compressed": c["text"]}, **res}), flush=True)


# --------------------------------------------------------------- trace --

def trace(args) -> None:
    from prefill_dispatch import read_trace, summarize

    everything = read_trace(args.dir)
    res = {"step": "trace", "dir": args.dir, "dispatches": len(everything),
           "process_max_ms": max(r["ms"] for r in everything),
           "process_max_kernel": max(everything, key=lambda r: r["ms"])["name"][:160],
           "families": summarize(everything)}
    if args.verdict:
        with open(args.verdict) as f:
            v = json.loads(next(ln for ln in f if ln.startswith("VERDICT "))[8:])
        rows = read_trace(args.dir, v["windows"])
        res["forward_max_ms"] = max(r["ms"] for r in rows)
        res["forward_max_kernel"] = max(rows, key=lambda r: r["ms"])["name"][:160]
    res["verdict"] = "PASS" if res["process_max_ms"] <= args.stop_ms else "FAIL"
    print("VERDICT " + json.dumps(res), flush=True)


# --------------------------------------------------------------- smoke --

def _post(port: int, path: str, body: dict) -> dict:
    import http.client

    c = http.client.HTTPConnection("127.0.0.1", port, timeout=600)
    c.request("POST", path, json.dumps(body), {"Content-Type": "application/json"})
    r = c.getresponse()
    data = json.loads(r.read())
    c.close()
    if r.status != 200:
        raise SystemExit(f"{path}: HTTP {r.status}: {data}")
    return data


def smoke(args) -> None:
    from drinkme.serve import DEFAULT_PORT

    if args.port == DEFAULT_PORT:
        raise SystemExit(f"REFUSING: port {DEFAULT_PORT} is drinkme serve's default, where a "
                         "working server may be listening; serve the pack under test on another --port")
    nonce = f"PELICAN-{int(time.time()) % 10000:04d}"
    w, h = (int(x) for x in args.size.lower().split("x"))
    data = base64.b64encode(png_bytes(screenshot(w, h, nonce))).decode()
    ask = "What is the code after 'CODE:' in the terminal? Reply with the code only."
    # another model's tool may ask for more tokens and a system prompt
    # (glimmer_vision_gate.py: its reasoning channel comes first)
    n, system = getattr(args, "max_tokens", 32), getattr(args, "system", None)
    chat = _post(args.port, "/v1/chat/completions", {
        "model": args.served, "max_tokens": n, "temperature": 0,
        "messages": ([{"role": "system", "content": system}] if system else []) + [
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": "data:image/png;base64," + data}},
                {"type": "text", "text": ask}]}]})
    msgs = _post(args.port, "/v1/messages", {
        "model": args.served, "max_tokens": n, "temperature": 0,
        **({"system": system} if system else {}),
        "messages": [{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                         "data": data}},
            {"type": "text", "text": ask}]}]})
    got = {"chat": chat["choices"][0]["message"]["content"],
           "messages": "".join(b.get("text", "") for b in msgs["content"])}
    ok = all(nonce in t for t in got.values())
    print("VERDICT " + json.dumps({
        "step": "smoke", "verdict": "PASS" if ok else "FAIL", "nonce": nonce, "size": args.size,
        "replies": got, "prompt_tokens": [chat["usage"]["prompt_tokens"],
                                          msgs["usage"]["input_tokens"]]}), flush=True)


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
                       "(default: 8 for tower, 75 stock, 55 compressed)")
    tw = sub.choices["tower"]
    tw.add_argument("--max-pixels", type=int, default=2560 * 1440, help="the server's cap")
    tw.add_argument("--reps", type=int, default=4, help="rep 0 compiles and autotunes")
    tw.add_argument("--fp32-check", action="store_true",
                    help="also compare with the tower in float32 on the CPU (host RAM: 2.3 GB)")
    rw = sub.choices["row"]
    rw.add_argument("--arm", choices=["stock", "compressed"], required=True)
    rw.add_argument("--model", required=True, help="the repo id (the engine's model_id)")
    rw.add_argument("--ctx", type=int, default=4096)
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
    sm.add_argument("--served", default="gemma-4-31B-it", help="the model name to send")
    sm.add_argument("--size", default="2560x1440")
    args = ap.parse_args()
    {"tower": tower, "row": row, "compare": compare, "trace": trace, "smoke": smoke}[args.step](args)


if __name__ == "__main__":
    main()
