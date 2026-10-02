#!/usr/bin/env python3
"""Cut the Caterpillar plate (The Nursery Alice, 1890, Tenniel hand-coloured,
PD) off its paper by processing alone, and write the webps #start uses.

Commons has no transparent version of this plate, so the alpha comes from
the paper, not from a curated mask:

1. Flat-field the paper. The scan's paper is warm white (about 247,248,241)
   and drifts across the sheet. Its tone is the median of the paper-like
   pixels (light, unsaturated) in each 48 px block, blocks with none are
   filled from their neighbours, and the field is smoothed. Dividing by it
   makes the paper white everywhere.
2. Paper to alpha (the inverse of printing on white paper): a pixel's alpha
   is how far its darkest channel falls below white, and its colour is what,
   laid over white at that alpha, gives the pixel back. So the plate reads
   as printed on whatever paper it sits on, and there is no plate edge
   anywhere, above all around the smoke, where the drawn rings continue it.
   Paper grain (alpha under 0.05) goes to 0 on a short ramp.
3. Two hand-set regions go to alpha 0: the rule across the top of the page
   (above y=100 in the scan) and Tenniel's monogram at the bottom left, as
   the hero cutout's curated mask also removed it.
4. Crop to what's left, and write the webps at the widths the page needs.

It prints the smoke's measured colour and line width (as scanned, and as
the cutout carries it: ink colour and alpha), and the open top of the
right-hand plume in the cropped image's fractions: the page's smoke script
(index.html, SMOKE) draws its rings with that ink and starts them there.
"""
import json
import os
import urllib.request

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

SRC = "caterpillar-nursery-c06543-03.jpg"      # gitignored: fetched on demand, like the hero's scan
URL = ("https://upload.wikimedia.org/wikipedia/commons/8/8a/"
       "John_Tenniel_-_Illustration_from_The_Nursery_Alice_%281890%29_-_c06543_03.jpg")
SHA1 = "92c594ee16efaec4668363e1bf6d53e246cd26ac"   # Commons imageinfo sha1 of the original
WIDTHS = (400, 700)                               # it shows at most ~340 CSS px wide: 1x and 2x

if not os.path.exists(SRC):
    print(f"fetching {SRC} from Commons...")
    req = urllib.request.Request(URL, headers={"User-Agent": "drinkme-site-build/1.0 (https://tangled.org/ninachaubal.com/drinkme)"})
    with urllib.request.urlopen(req) as r, open(SRC, "wb") as f:
        f.write(r.read())
import hashlib
assert hashlib.sha1(open(SRC, "rb").read()).hexdigest() == SHA1, "not the Commons original"

rgb = np.asarray(Image.open(SRC).convert("RGB"), dtype=np.float64)
H, W, _ = rgb.shape

# 1. the paper's tone, block by block
B = 48
lum = rgb.mean(2)
sat = rgb.max(2) - rgb.min(2)
paperish = (lum > 225) & (sat < 22)
gh, gw = -(-H // B), -(-W // B)
field = np.full((gh, gw, 3), np.nan)
for j in range(gh):
    for i in range(gw):
        m = paperish[j*B:(j+1)*B, i*B:(i+1)*B]
        if m.sum() > B * B * 0.2:
            field[j, i] = np.median(rgb[j*B:(j+1)*B, i*B:(i+1)*B][m], axis=0)
for _ in range(200):                                   # fill empty blocks from their neighbours
    gaps = np.isnan(field[..., 0])
    if not gaps.any():
        break
    pad = np.pad(field, ((1, 1), (1, 1), (0, 0)), constant_values=np.nan)
    near = np.stack([pad[:-2, 1:-1], pad[2:, 1:-1], pad[1:-1, :-2], pad[1:-1, 2:]])
    with np.errstate(all="ignore"):
        fill = np.nanmean(near, axis=0)
    field[gaps] = fill[gaps]
tone = Image.fromarray(np.clip(field, 0, 255).astype(np.uint8), "RGB").resize((W, H), Image.BICUBIC)
tone = np.asarray(tone.filter(ImageFilter.GaussianBlur(40)), dtype=np.float64)
flat = np.clip(rgb / np.maximum(tone, 1) * 255, 0, 255)

# 2. paper to alpha, against white
a = (255 - flat.min(2)) / 255
a = np.clip(a, 0, 1)
col = np.where(a[..., None] > 1e-3, (flat - 255 * (1 - a[..., None])) / np.maximum(a[..., None], 1e-3), 0)
ramp = np.clip((a - 0.05) / 0.05, 0, 1)                # paper grain out, lines and washes kept
alpha = a * ramp

# 3. the hand-set regions: the page's top rule, and the monogram
cut = Image.new("L", (W, H), 255)
d = ImageDraw.Draw(cut)
d.rectangle((0, 0, W, 100), fill=0)
d.polygon([(135, 1750), (232, 1750), (232, 1842), (135, 1842)], fill=0)
alpha = alpha * (np.asarray(cut.filter(ImageFilter.GaussianBlur(3)), dtype=np.float64) / 255)

out = np.dstack([np.clip(col, 0, 255), alpha * 255]).round().astype(np.uint8)
img = Image.fromarray(out, "RGBA")

# 4. crop to the art
ink = alpha > 0.25
rows, cols = np.nonzero(ink.sum(1) >= 3)[0], np.nonzero(ink.sum(0) >= 3)[0]   # a line, not a speck of scan dust
pad = 8
box = (max(cols.min() - pad, 0), max(rows.min() - pad, 0), min(cols.max() + pad + 1, W), min(rows.max() + pad + 1, H))
img = img.crop(box)
cw, ch = img.size
print(f"crop box {box}, {cw}x{ch}, aspect {cw/ch:.5f}")
img.save("caterpillar-cutout.png")                     # gitignored working file
for w in WIDTHS:
    h = round(ch * w / cw)
    img.resize((w, h), Image.LANCZOS).save(f"caterpillar-{w}.webp", quality=72, alpha_quality=70, method=6)
    print(f"caterpillar-{w}.webp {w}x{h}")

# the smoke, measured: the right-hand plume's lines between its open top and the hookah's pipe
x0, y0, x1, y1 = 1000, 150, 1240, 440
reg, pap = rgb[y0:y1, x0:x1], np.median(rgb[y0:y1, x0:x1].reshape(-1, 3), axis=0)
dark = pap.mean() - reg.mean(2)
line = reg[dark > 80]
runs = []
for row in dark > 40:
    n = 0
    for v in row:
        if v: n += 1
        elif n: runs.append(n); n = 0
fl = np.median(flat[y0:y1, x0:x1][dark > 80], axis=0)  # the line's core, on the flattened paper
ink_a = (255 - fl.min()) / 255                          # and as the cutout carries it: colour and alpha
ink = (fl - 255 * (1 - ink_a)) / ink_a
top = [(1035, 172), (1215, 170)]                       # the plume's two open line ends, read off the scan
print(json.dumps({
    "smoke_rgb": [round(v) for v in np.median(line, axis=0)],
    "smoke_line_px_at_1440": float(np.median(runs)),
    "smoke_ink_rgb": [round(v) for v in ink], "smoke_ink_alpha": round(ink_a, 3),
    "smoke_line_frac_of_width": round(float(np.median(runs)) / cw, 5),
    "plume_open_top": [[round((x - box[0]) / cw, 4), round((y - box[1]) / ch, 4)] for x, y in top],
    "aspect": round(cw / ch, 5),
}))
