"""Synthetic screenshots for the vision instruments, drawn with Pillow at run
time (nothing downloaded, nothing checked in): a dark terminal pane of
monospace shell output with a nonce on one line, and a light UI panel with a
small table whose one cell holds a second nonce.

`screenshot(w, h, nonce=..., cell=..., scale=1.0)` draws at `scale` x the
1x layout's pixel sizes, so `scale=2.0` at 2880x1800 is a Retina capture of a
1440x900 desktop: the text is as many points as at 1x, twice the pixels.
`small_text_screenshot` draws one line per font size, each carrying its own
code, for the readability ladder (the pixel-cap decision).

Fonts: DejaVu Sans Mono / DejaVu Sans from /usr/share/fonts when present,
else Pillow's own scalable default.
"""
from __future__ import annotations

import io
import os
import random

from PIL import Image, ImageDraw, ImageFont

_MONO = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
_SANS = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


def font(px: int, mono: bool = True):
    path = _MONO if mono else _SANS
    if os.path.exists(path):
        return ImageFont.truetype(path, px)
    return ImageFont.load_default(size=px)


_SHELL = [
    "$ git log --oneline -6",
    "3f2a9c1 docs: a note on the pixel cap",
    "a7d04e2 panel: keep the cursor after a resize",
    "9b1e6f0 Merge branch 'ui-panel' into main",
    "4c88d21 tests: the panel renders its table",
    "e05b7a3 build: refresh the font cache",
    "$ ls -la src/drinkme/serving",
    "-rw-r--r-- 1 dev dev  41203 Sep 25 10:12 engines.py",
    "-rw-r--r-- 1 dev dev  12877 Sep 25 10:12 image_prompt.py",
    "-rw-r--r-- 1 dev dev   9310 Sep 25 10:12 mrope.py",
    "-rw-r--r-- 1 dev dev  38116 Sep 25 10:12 vision.py",
    "$ drinkme status",
]


def screenshot(w: int, h: int, *, nonce: str = "PELICAN-7391", cell: str = "OSPREY-2846",
               scale: float = 1.0, seed: int = 0) -> Image.Image:
    """A w x h screenshot: terminal on the left 60%, a UI table on the right.
    The nonce is on the terminal's last output line ("deploy token: <nonce>"),
    `cell` in the table's third row, 'Token' column."""
    rng = random.Random(seed)
    img = Image.new("RGB", (w, h), (236, 238, 241))
    d = ImageDraw.Draw(img)
    s = scale
    bar = int(28 * s)
    d.rectangle([0, 0, w, bar], fill=(210, 212, 216))
    d.text((int(12 * s), int(6 * s)), "Terminal — dev@workstation: ~/drinkme",
           font=font(int(14 * s), mono=False), fill=(40, 40, 40))
    tw = int(w * 0.6)
    d.rectangle([0, bar, tw, h], fill=(24, 26, 30))
    f = font(int(15 * s))
    lh = int(22 * s)
    y = bar + int(10 * s)
    lines = list(_SHELL)
    while len(lines) * lh < (h - bar) * 0.55:
        lines.append(f"[drinkme.engine] images: 1 in the prompt ({rng.randint(200, 3600)} "
                     f"image tokens); the tower ran 1x, 0 inside the reused prefix")
    lines.append(f"deploy token: {nonce}")
    lines.append("$ _")
    for ln in lines:
        if y + lh > h:
            break
        col = (120, 220, 120) if ln.startswith("$") else (215, 218, 222)
        if ln.startswith("deploy token"):
            col = (255, 214, 102)
        d.text((int(14 * s), y), ln, font=f, fill=col)
        y += lh
    # the UI panel: a small table
    x0, y0 = tw + int(24 * s), bar + int(24 * s)
    d.text((x0, y0), "Deployments", font=font(int(20 * s), mono=False), fill=(20, 20, 20))
    y0 += int(40 * s)
    cols = ["Service", "Region", "Token", "Status"]
    cw = (w - x0 - int(24 * s)) // len(cols)
    rows = [["api", "us-west", "HERON-1177", "ok"],
            ["worker", "eu-central", "EGRET-5520", "ok"],
            ["bottle", "local", cell, "degraded"],
            ["site", "us-east", "IBIS-0934", "ok"]]
    rh = int(30 * s)
    ft = font(int(13 * s), mono=False)
    d.rectangle([x0, y0, x0 + cw * len(cols), y0 + rh], fill=(200, 205, 214))
    for j, c in enumerate(cols):
        d.text((x0 + j * cw + int(8 * s), y0 + int(8 * s)), c, font=ft, fill=(10, 10, 10))
    for i, r in enumerate(rows):
        yy = y0 + rh * (i + 1)
        d.rectangle([x0, yy, x0 + cw * len(cols), yy + rh],
                    fill=(255, 255, 255) if i % 2 == 0 else (245, 246, 248), outline=(220, 220, 225))
        for j, c in enumerate(r):
            d.text((x0 + j * cw + int(8 * s), yy + int(8 * s)), c, font=ft, fill=(30, 30, 30))
    return img


def small_text_screenshot(w: int, h: int, sizes: list[int], codes: list[str],
                          scale: float = 1.0) -> Image.Image:
    """One line per font size (points at 1x; drawn at size * scale pixels):
    'size N px: code <CODE>', light UI background, the rest of the pane
    filled with filler UI text at the smallest size."""
    img = Image.new("RGB", (w, h), (248, 248, 250))
    d = ImageDraw.Draw(img)
    y = int(20 * scale)
    for sz, code in zip(sizes, codes):
        px = max(4, int(round(sz * scale)))
        d.text((int(20 * scale), y), f"label {sz}: code {code}", font=font(px, mono=False),
               fill=(30, 30, 35))
        y += int(px * 1.8)
    fill = font(max(4, int(round(min(sizes) * scale))), mono=False)
    while y < h - int(30 * scale):
        d.text((int(20 * scale), y), "Settings  General  Appearance  Accounts  Privacy  "
               "Notifications  Keyboard  Displays  Sound  Network", font=fill, fill=(120, 120, 128))
        y += int(min(sizes) * scale * 2.2)
    return img


def png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()
