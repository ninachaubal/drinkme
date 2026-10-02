#!/usr/bin/env python3
"""Cut Alice (The Nursery Alice, 1890, Tenniel hand-colored, PD) out of the
full-res Commons scan by REGISTERING the curated Commons transparency
(Alice_drink_me_transparent.png, 491x900 — its original size) onto the
1900x3484 scan (Alice_drink_me.jpg) and reusing its alpha at scale.

Why not flood fill: the wall behind Alice is a green wash over engraved
crosshatching — dark lines everywhere, so color-region methods can't tell
wall from figure. The curated mask already made the editorial calls
(table removed, monogram removed); registration borrows them at high res.
A 3.7x-upscaled mask has ~4px edge error on a 3484px image — sub-pixel at
any size the site will ever display.
"""
import json
import os
import sys
import urllib.request

import numpy as np
from PIL import Image, ImageFilter

# `make-cutout.py --bands <cutout.png|webp>` prints the page's band array for a cutout: the image's
# aspect (width / height) and, for 64 rows top to bottom, the [left, right] extent of its opaque pixels
# (alpha > 64) as fractions of its width. index.html inlines these as ALICE and RABBIT for the text that
# flows around each figure; caterpillar-bands.json is the Caterpillar's (from caterpillar-cutout.png), not
# inlined, since no text runs beside him. A row with nothing opaque takes the next row's extent.
if sys.argv[1:2] == ["--bands"]:
    a = np.asarray(Image.open(sys.argv[2]).convert("RGBA"))[..., 3] > 64
    H, W = a.shape
    rows = []
    for r in range(64):
        cols = np.nonzero(a[r * H // 64:(r + 1) * H // 64].any(0))[0]
        rows.append([round(cols.min() / W, 4), round((cols.max() + 1) / W, 4)] if len(cols) else None)
    for i, row in enumerate(rows):
        if row is None:
            rows[i] = next((x for x in rows[i:] + rows[::-1] if x), [0.5, 0.5])
    print(json.dumps({"aspect": round(W / H, 5), "rows": rows}, separators=(",", ":")))
    sys.exit(0)

BIG = "alice-drink-me-full.jpg"
SMALL = "alice-drink-me-transparent-491.png"
OUT = "alice-hero.png"

# both PD, on Wikimedia Commons; fetched on demand (gitignored locally)
SOURCES = {
    BIG: "https://upload.wikimedia.org/wikipedia/commons/0/0d/Alice_drink_me.jpg",
    SMALL: "https://upload.wikimedia.org/wikipedia/commons/b/b4/Alice_drink_me_transparent.png",
}
for path, url in SOURCES.items():
    if not os.path.exists(path):
        print(f"fetching {path} from Commons...")
        req = urllib.request.Request(url, headers={"User-Agent": "drinkme-site-build/1.0"})
        with urllib.request.urlopen(req) as r, open(path, "wb") as f:
            f.write(r.read())

big = Image.open(BIG).convert("RGB")
small = Image.open(SMALL).convert("RGBA")
W, H = big.size
bg = np.asarray(big.convert("L"), dtype=np.float64)

def ncc_at(sg, sa, bg8, s, step):
    """Best offset of scaled-small over big at this pyramid step; returns
    (score, dy, dx). sg/sa = scaled small gray/alpha at 1/step. bg8 = big
    gray at 1/step."""
    h, w = sg.shape
    Hb, Wb = bg8.shape
    if h > Hb or w > Wb:
        return (-2, 0, 0)
    best = (-2, 0, 0)
    m = sa > 0.5
    sgm = sg[m]
    sgn = (sgm - sgm.mean()) / (sgm.std() + 1e-9)
    for dy in range(0, Hb - h + 1):
        for dx in range(0, Wb - w + 1):
            win = bg8[dy:dy+h, dx:dx+w][m]
            wn = (win - win.mean()) / (win.std() + 1e-9)
            score = (sgn * wn).mean()
            if score > best[0]:
                best = (score, dy, dx)
    return best

def scaled_arrays(s, step):
    sw, sh = round(small.width * s), round(small.height * s)
    sc = small.resize((sw, sh), Image.LANCZOS)
    g = np.asarray(sc.convert("L"), dtype=np.float64)[::step, ::step]
    a = np.asarray(sc.split()[3], dtype=np.float64)[::step, ::step] / 255
    return g, a, (sw, sh)

# --- coarse: 1/16, scan scales ---
STEP = 16
bg16 = bg[::STEP, ::STEP]
results = []
for s in np.arange(3.45, 3.95, 0.05):
    g, a, _ = scaled_arrays(s, STEP)
    score, dy, dx = ncc_at(g, a, bg16, s, STEP)
    results.append((score, s, dy * STEP, dx * STEP))
    print(f"s={s:.2f} ncc={score:.4f} at ({dx*STEP},{dy*STEP})")
score, s0, y0, x0 = max(results)

# --- refine scale at 1/8 around winner ---
STEP = 8
bg8 = bg[::STEP, ::STEP]
results = []
for s in np.arange(s0 - 0.06, s0 + 0.061, 0.015):
    g, a, _ = scaled_arrays(s, STEP)
    sc_, dy, dx = ncc_at(g, a, bg8, s, STEP)
    results.append((sc_, s, dy * STEP, dx * STEP))
score, s1, y1, x1 = max(results)
print(f"refined: s={s1:.3f} ncc={score:.4f} at ({x1},{y1})")

# --- final: full-res local offset search (±8px) at the chosen scale ---
sw, sh = round(small.width * s1), round(small.height * s1)
scf = small.resize((sw, sh), Image.LANCZOS)
gf = np.asarray(scf.convert("L"), dtype=np.float64)
af = np.asarray(scf.split()[3], dtype=np.float64) / 255
m = af > 0.5
gfm = gf[m]
gfn = (gfm - gfm.mean()) / (gfm.std() + 1e-9)
best = (-2, y1, x1)
for dy in range(max(0, y1 - 8), min(H - sh, y1 + 8) + 1, 2):
    for dx in range(max(0, x1 - 8), min(W - sw, x1 + 8) + 1, 2):
        win = bg[dy:dy+sh, dx:dx+sw][m]
        wn = (win - win.mean()) / (win.std() + 1e-9)
        sc_ = (gfn * wn).mean()
        if sc_ > best[0]:
            best = (sc_, dy, dx)
score, dy, dx = best
print(f"final: ncc={score:.4f} at ({dx},{dy}), scale {s1:.3f}")

# --- build output: big RGB + registered alpha ---
alpha_full = np.zeros((H, W), dtype=np.uint8)
alpha_full[dy:dy+sh, dx:dx+sw] = (af * 255).astype(np.uint8)
alpha = Image.fromarray(alpha_full, "L").filter(ImageFilter.GaussianBlur(2.0))
out = big.copy()
out.putalpha(alpha)
a = np.asarray(alpha)
ys, xs = np.nonzero(a > 8)
pad = 4
box = (max(xs.min()-pad, 0), max(ys.min()-pad, 0),
       min(xs.max()+pad, W), min(ys.max()+pad, H))
out = out.crop(box)
out.save(OUT)
print(f"saved {OUT} {out.size}, crop box {box}")

prev = out.copy(); prev.thumbnail((480, 900))
bgc = Image.new("RGB", prev.size, (247, 241, 227))
bgc.paste(prev, mask=prev.split()[3])
bgc.save("/tmp/alice-cutout-preview.png")
