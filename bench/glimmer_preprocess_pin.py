"""Muse-Glimmer's image preprocessing (serving/vision.MuseGlimmerPreprocessor)
against transformers' MuseGlimmerImageProcessor, bit for bit.

transformers ships that processor with ONE backend, torchvision's, and
torchvision is in no drinkme environment (its wheels must match the torch
build; pyproject.toml). So the reference runs in a THROWAWAY venv that has
it, never the shared one:

    uv venv --python 3.12 /tmp/tv
    uv pip install --python /tmp/tv/bin/python --index-strategy unsafe-best-match \\
        "torch==2.12.1+cpu" "torchvision==0.27.1+cpu" "pillow==12.3.0" "numpy==2.5.1" \\
        "transformers==5.15.1" --index-url https://download.pytorch.org/whl/cpu \\
        --extra-index-url https://pypi.org/simple
    PYTHONPATH=src /tmp/tv/bin/python bench/glimmer_preprocess_pin.py reference \\
        --out /tmp/ref.json [--files DIR]
    PYTHONPATH=src .venv/bin/python bench/glimmer_preprocess_pin.py drinkme \\
        --out /tmp/ours.json [--files DIR]
    .venv/bin/python bench/glimmer_preprocess_pin.py compare /tmp/ref.json /tmp/ours.json

  reference  in the throwaway venv: for every case, transformers' processor
             (built from the Muse-Glimmer-30B snapshot's image_processor
             config, max_image_tokens lowered to the case's budget) and
             drinkme's preprocessor side by side in one process, compared
             element for element; also a Pillow-LANCZOS variant of drinkme's,
             the port a torch-free resize would have been. Prints a VERDICT
             line and writes every case's digest (the reference's) to --out.
  drinkme    drinkme's preprocessor alone, in any venv with drinkme (the
             served torch's own LANCZOS kernel): the digests to --out.
  compare    two --out files: PASS when every case's grid, token count and
             pixel_values digest agree.

The cases: SYNTHETIC ones, an RGB array per case from a fixed numpy seed
(tests/test_serving_glimmer_vision.py pins drinkme against these cases'
reference digests, so no decoder version can move them), and with --files
DIR every file in DIR as a request's bytes, decoded by each side its own way
(transformers' load_image; drinkme's vision.decode).
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import sys

# the 30B's processor_config.json image_processor (the snapshot's, verbatim)
SNAPSHOT_CONFIG = {
    "do_convert_rgb": True, "do_normalize": True, "do_rescale": True, "do_resize": True,
    "image_mean": [0.5, 0.5, 0.5], "image_processor_type": "MuseGlimmerImageProcessor",
    "image_std": [0.5, 0.5, 0.5], "max_image_tokens": 4096, "merge_size": 2, "patch_size": 14,
    "resample": 1, "rescale_factor": 0.00392156862745098, "temporal_patch_size": 2,
}
DEFAULT_CAP = 2560 * 1440  # vision.DEFAULT_MAX_PIXELS
LOW = 512 * 512            # vision.LOW_DETAIL_PIXELS

# (name, width, height, seed, pixel budget): the synthetic cases. Sizes
# already on the 28-px grid (no resize), upscaled small images, downscaled
# large ones, the token cap, extreme aspect ratios, grids whose floor/ceil
# candidates tie on aspect error, and budgets under the checkpoint's own.
SYNTHETIC = [
    ("grid-560x392", 560, 392, 1, DEFAULT_CAP),
    ("tiny-5x7", 5, 7, 2, DEFAULT_CAP),
    ("small-37x53", 37, 53, 3, DEFAULT_CAP),
    ("tie-100x100", 100, 100, 4, DEFAULT_CAP),
    ("odd-641x479", 641, 479, 5, DEFAULT_CAP),
    ("1080p-1920x1080", 1920, 1080, 6, DEFAULT_CAP),
    ("cap-2560x1440", 2560, 1440, 7, DEFAULT_CAP),
    ("wide-3000x3", 3000, 3, 8, DEFAULT_CAP),
    ("tall-3x1500", 3, 1500, 9, DEFAULT_CAP),
    ("square-1000", 1000, 1000, 10, DEFAULT_CAP),
    ("low-1280x720", 1280, 720, 11, LOW),
    ("one-token-90x70", 90, 70, 12, 28 * 28),
    ("budget-40-777x333", 777, 333, 13, 40 * 28 * 28),
]


def synthetic(w: int, h: int, seed: int):
    """A deterministic RGB PIL image: gradients, noise and two filled
    rectangles (edges for the resampler), from numpy alone."""
    import numpy as np
    from PIL import Image

    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:h, 0:w]
    a = np.stack([(x * 255 // max(w - 1, 1)), (y * 255 // max(h - 1, 1)),
                  ((x + y) * 7) % 256], axis=-1).astype(np.int16)
    a += rng.integers(-24, 25, size=a.shape, dtype=np.int16)
    a[h // 5:h // 2 + 1, w // 5:w // 2 + 1] = (250, 10, 10)
    a[2 * h // 3:, 2 * w // 3:] = (5, 5, 240)
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8), "RGB")


def preprocessor():
    from drinkme.serving import vision

    return vision.preprocessor_for("muse_glimmer", dict(SNAPSHOT_CONFIG))


def ours(pre, rgb, budget: int, pillow: bool = False):
    """drinkme's (pixel_values, grid, tokens) for an RGB image; `pillow`
    swaps the resize for Pillow's LANCZOS (the negative control)."""
    import numpy as np
    from PIL import Image

    size, grid, tokens = pre.plan(rgb.height, rgb.width, budget, where="pin")
    if pillow and (rgb.height, rgb.width) != tuple(size):
        rgb = rgb.resize((size[1], size[0]), resample=Image.Resampling.LANCZOS)
    return np.ascontiguousarray(pre.pixel_values(rgb, size, budget)), tuple(grid), tokens


def reference(rgb_or_b64, budget: int):
    """transformers' MuseGlimmerImageProcessor (torchvision backend) on one
    image: (pixel_values, grid). A base64 string goes through load_image
    first, as a request's bytes would."""
    from transformers.image_utils import load_image
    from transformers.models.muse_glimmer.image_processing_muse_glimmer import (
        MuseGlimmerImageProcessor)

    cfg = {k: v for k, v in SNAPSHOT_CONFIG.items() if k != "image_processor_type"}
    proc = MuseGlimmerImageProcessor(**cfg)
    img = load_image(rgb_or_b64) if isinstance(rgb_or_b64, str) else rgb_or_b64
    out = proc(img, max_image_tokens=pre_tokens(budget), return_tensors="pt")
    return (out["pixel_values"].numpy(),
            tuple(int(v) for v in out["image_grid_thw"][0]))


def pre_tokens(budget: int) -> int:
    return preprocessor().max_tokens(budget)


def digest(pv, grid) -> str:
    import numpy as np

    h = hashlib.sha256(np.ascontiguousarray(pv, dtype="<f4").tobytes())
    h.update(np.asarray(grid, dtype="<i8").tobytes())
    return h.hexdigest()


def cases(files: str | None):
    """(name, source, budget): source is an RGB image (synthetic) or a
    base64 string (a file's bytes, decoded by each side its own way)."""
    for name, w, h, seed, budget in SYNTHETIC:
        yield name, synthetic(w, h, seed), budget
    if files:
        for fn in sorted(os.listdir(files)):
            with open(os.path.join(files, fn), "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
            for budget in (DEFAULT_CAP, LOW):
                yield f"file:{fn}@{budget}", b64, budget


def _decoded(src):
    from drinkme.serving import vision

    if not isinstance(src, str):
        return src
    return vision.decode(vision.parse_base64(src, where="pin"), where="pin")


def run_reference(args) -> None:
    import numpy as np

    pre = preprocessor()
    rows, bad, pillow_off = {}, [], []
    for name, src, budget in cases(args.files):
        rgb = _decoded(src)
        try:
            pv, grid, tokens = ours(pre, rgb, budget)
        except ValueError as e:  # an ImageError: drinkme refuses what the reference serves
            rpv, rgrid = reference(src, budget)
            rows[name] = {"refused": str(e), "reference_grid": list(rgrid)}
            continue
        rpv, rgrid = reference(src, budget)
        same = grid == rgrid and pv.shape == rpv.shape and pv.tobytes() == rpv.tobytes()
        diff = (float(np.abs(pv - rpv).max()) if pv.shape == rpv.shape else None)
        ppv = ours(pre, rgb, budget, pillow=True)[0]
        pillow_same = ppv.shape == rpv.shape and ppv.tobytes() == rpv.tobytes()
        rows[name] = {"grid": list(grid), "tokens": tokens, "shape": list(pv.shape),
                      "sha256": digest(rpv, rgrid), "bit_equal": same,
                      "max_abs_diff": diff, "pillow_lanczos_bit_equal": pillow_same}
        if not same:
            bad.append(name)
        if not pillow_same:
            pillow_off.append(name)
    with open(args.out, "w") as f:
        json.dump(rows, f, indent=1, sort_keys=True)
    import torch
    import torchvision

    print("VERDICT " + json.dumps({
        "step": "reference", "verdict": "PASS" if not bad else "FAIL",
        "cases": len(rows), "bit_equal": len(rows) - len(bad), "mismatched": bad,
        "refused": [k for k, v in rows.items() if "refused" in v],
        "pillow_lanczos_differs": len(pillow_off), "torch": torch.__version__,
        "torchvision": torchvision.__version__}), flush=True)


def run_drinkme(args) -> None:
    import torch

    pre = preprocessor()
    rows = {}
    for name, src, budget in cases(args.files):
        try:
            pv, grid, tokens = ours(pre, _decoded(src), budget)
        except ValueError as e:
            rows[name] = {"refused": str(e)}
            continue
        rows[name] = {"grid": list(grid), "tokens": tokens, "shape": list(pv.shape),
                      "sha256": digest(pv, grid)}
    with open(args.out, "w") as f:
        json.dump(rows, f, indent=1, sort_keys=True)
    print("VERDICT " + json.dumps({"step": "drinkme", "cases": len(rows),
                                   "torch": torch.__version__}), flush=True)


def run_compare(args) -> None:
    a, b = (json.load(open(p)) for p in (args.a, args.b))
    keys = ("grid", "tokens", "sha256", "refused")

    def row(side, k):
        return {x: side.get(k, {}).get(x) for x in keys}

    off = sorted(k for k in set(a) | set(b) if row(a, k) != row(b, k))
    print("VERDICT " + json.dumps({"step": "compare", "verdict": "PASS" if not off else "FAIL",
                                   "cases": len(set(a) | set(b)), "differ": off}), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="step", required=True)
    for name in ("reference", "drinkme"):
        p = sub.add_parser(name)
        p.add_argument("--out", required=True)
        p.add_argument("--files", help="a directory of image files to add as cases")
    cp = sub.add_parser("compare")
    cp.add_argument("a")
    cp.add_argument("b")
    args = ap.parse_args()
    {"reference": run_reference, "drinkme": run_drinkme, "compare": run_compare}[args.step](args)


if __name__ == "__main__":
    sys.exit(main())
