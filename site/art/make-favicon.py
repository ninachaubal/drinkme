"""Make the site's icons from the DRINK ME bottle in the hero plate.

The bottle is cut from art/alice-hero-1400.png (Tenniel, The Nursery "Alice", 1890,
public domain; provenance in alice-hero-1400.webp.json): a polygon keeps the tag and
the bottle and cuts away Alice's hand along the bottle's edge, a colour pass drops the
last specks of finger, and the result is stood upright and set on a rounded lilac tile
so it reads on light and dark tab strips.

    python3 site/art/make-favicon.py      # writes site/favicon.ico, site/apple-touch-icon.png,
                                          # site/art/favicon-32.png
"""
import colorsys
import pathlib

from PIL import Image, ImageDraw

ART = pathlib.Path(__file__).resolve().parent
SITE = ART.parent
TILE = (0xf3, 0xe7, 0xf4, 255)  # the page theme-color, #f3e7f4

# The bottle's box in the hero cutout, and the outline that keeps tag + bottle (box coords).
BOX = (5, 232, 232, 452)
KEEP = [(0, 185), (0, 120), (60, 70), (150, 25), (190, 5), (215, 12), (210, 58), (192, 92),
        (174, 128), (157, 172), (147, 212), (120, 220), (40, 210)]
UPRIGHT_DEG = 22


def skin(r, g, b):
    h, s, v = colorsys.rgb_to_hsv(r / 255, g / 255, b / 255)
    return 0.0 <= h <= 0.12 and 0.12 <= s <= 0.6 and v >= 0.68


def bottle() -> Image.Image:
    plate = Image.open(ART / "alice-hero-1400.png").convert("RGBA").crop(BOX)
    w, h = plate.size
    keep = Image.new("L", (w, h), 0)
    ImageDraw.Draw(keep).polygon(KEEP, fill=255)
    plate.putalpha(Image.composite(plate.split()[3], Image.new("L", (w, h), 0), keep))
    px = plate.load()
    for y in range(h):
        for x in range(w):
            r, g, b, a = px[x, y]
            if a and x > 115 and skin(r, g, b):  # finger specks right of the tag
                px[x, y] = (r, g, b, 0)
    up = plate.crop(plate.getbbox()).rotate(UPRIGHT_DEG, resample=Image.BICUBIC, expand=True)
    return up.crop(up.getbbox())


def tile(src: Image.Image, size: int, pad: float = 0.05, radius: float = 0.22) -> Image.Image:
    t = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    m = Image.new("L", (size, size), 0)
    ImageDraw.Draw(m).rounded_rectangle((0, 0, size - 1, size - 1), int(size * radius), fill=255)
    t.paste(Image.new("RGBA", (size, size), TILE), (0, 0), m)
    s = src.copy()
    s.thumbnail((int(size * (1 - 2 * pad)), int(size * (1 - 2 * pad))), Image.LANCZOS)
    t.alpha_composite(s, ((size - s.width) // 2, (size - s.height) // 2))
    return t


def main() -> None:
    b = bottle()
    tile(b, 256).save(SITE / "favicon.ico", sizes=[(16, 16), (32, 32), (48, 48)])
    tile(b, 32).save(ART / "favicon-32.png", optimize=True)
    tile(b, 180, radius=0.0).convert("RGB").save(SITE / "apple-touch-icon.png", optimize=True)


if __name__ == "__main__":
    main()
