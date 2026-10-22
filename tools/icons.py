"""Draw the icons of the Grasshopper components: one flat set, 24 by 24 pixels.

Each glyph is drawn at four times the size and scaled down; it uses its subcategory's accent colour and dark grey.
``python tools/icons.py`` writes ``csharp/CompasAcoustics.Grasshopper/Components/Specs/<CA_name>/icon.png``.
"""

import json
import math
from pathlib import Path

from PIL import Image
from PIL import ImageDraw

SPECS = Path(__file__).resolve().parents[1] / "csharp" / "CompasAcoustics.Grasshopper" / "Components" / "Specs"
SIZE = 24
SCALE = 4
S = SIZE * SCALE
W = 7
DARK = (60, 60, 60, 255)

ACCENTS = {
    "00 Elements": (31, 119, 180),
    "01 Model": (44, 160, 44),
    "02 Simulation": (214, 39, 40),
    "03 Analysis": (148, 103, 189),
    "04 Auralisation": (255, 127, 14),
    "05 Display": (23, 160, 190),
    "06 Util": (110, 110, 110),
    "07 Optimization": (170, 150, 20),
}


def _dot(d, x, y, r, fill):
    d.ellipse((x - r, y - r, x + r, y + r), fill=fill)


def _ring(d, x, y, r, fill, width=W):
    d.ellipse((x - r, y - r, x + r, y + r), outline=fill, width=width)


def _line(d, points, fill, width=W):
    d.line(points, fill=fill, width=width, joint="curve")


def _hatch(image, box, fill, step=14):
    layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    for k in range(-S, S, step):
        d.line((k, S, k + S, 0), fill=fill, width=W - 2)
    mask = Image.new("L", image.size, 0)
    ImageDraw.Draw(mask).rectangle(box, fill=255)
    image.alpha_composite(Image.composite(layer, Image.new("RGBA", image.size, (0, 0, 0, 0)), mask))


def _speaker(d, x, y, scale, fill):
    body = (x, y - 10 * scale, x + 14 * scale, y + 10 * scale)
    d.rectangle(body, fill=fill)
    d.polygon([(x + 14 * scale, y - 10 * scale), (x + 30 * scale, y - 24 * scale), (x + 30 * scale, y + 24 * scale), (x + 14 * scale, y + 10 * scale)], fill=fill)


def source(image, d, accent):
    _speaker(d, 10, 48, 1.0, DARK)
    for r in (18, 32):
        d.arc((44 - r, 48 - r, 44 + r, 48 + r), -45, 45, fill=accent, width=W)


def receiver(image, d, accent):
    _ring(d, 48, 42, 24, accent)
    _dot(d, 22, 44, 7, DARK)
    _dot(d, 74, 44, 7, DARK)
    _line(d, [(48, 68), (48, 86)], DARK)
    d.polygon([(42, 18), (54, 18), (48, 6)], fill=DARK)


def material(image, d, accent):
    _hatch(image, (18, 18, 78, 78), accent)
    d.rectangle((16, 16, 80, 80), outline=DARK, width=W)


def deconstruct_material(image, d, accent):
    _hatch(image, (12, 18, 42, 78), accent)
    _hatch(image, (54, 18, 84, 78), accent)
    d.rectangle((10, 16, 44, 80), outline=DARK, width=W)
    d.rectangle((52, 16, 86, 80), outline=DARK, width=W)


def library(image, d, accent):
    d.rectangle((20, 12, 34, 84), fill=accent)
    d.rectangle((20, 12, 78, 84), outline=DARK, width=W)
    for y in (32, 46, 60):
        _line(d, [(44, y), (68, y)], DARK, W - 2)


def wall(image, d, accent):
    d.rectangle((10, 22, 86, 74), outline=DARK, width=W)
    _line(d, [(10, 48), (86, 48)], accent)
    for x in (36, 62):
        _line(d, [(x, 22), (x, 48)], accent)
    for x in (24, 50, 74):
        _line(d, [(x, 48), (x, 74)], accent)


def room(image, d, accent):
    front = (12, 34, 62, 84)
    back = (34, 12, 84, 62)
    d.rectangle(back, outline=DARK, width=W - 2)
    for (x0, y0), (x1, y1) in zip(
        [(front[0], front[1]), (front[2], front[1]), (front[0], front[3]), (front[2], front[3])],
        [(back[0], back[1]), (back[2], back[1]), (back[0], back[3]), (back[2], back[3])],
    ):
        _line(d, [(x0, y0), (x1, y1)], DARK, W - 2)
    d.rectangle(front, outline=accent, width=W)


def rays(image, d, accent):
    d.rectangle((8, 8, 88, 88), outline=DARK, width=W - 2)
    _line(d, [(24, 68), (80, 38), (56, 16), (16, 44)], accent)
    _dot(d, 24, 68, 8, DARK)


def hybrid(image, d, accent):
    _line(d, [(12, 70), (30, 18), (46, 56)], accent)
    _dot(d, 12, 70, 7, DARK)
    for i, x in enumerate(range(54, 90, 8)):
        h = 46 * math.exp(-0.45 * i)
        _line(d, [(x, 84), (x, 84 - h)], DARK, W - 1)


def graph(image, d, accent):
    for y in (24, 72):
        _line(d, [(22, y), (74, 48)], accent)
    _dot(d, 22, 24, 11, DARK)
    _dot(d, 22, 72, 11, DARK)
    _dot(d, 74, 48, 13, accent)


def decay(image, d, accent):
    _line(d, [(14, 10), (14, 84), (88, 84)], DARK)
    points = [(16 + 70 * t, 16 + 64 * (1 - math.exp(-4 * t))) for t in [i / 40 for i in range(41)]]
    _line(d, points, accent)


def bars(image, d, accent):
    for x, h in zip((16, 36, 56, 76), (40, 62, 50, 28)):
        d.rectangle((x - 6, 82 - h, x + 6, 82), fill=accent)
    _line(d, [(6, 84), (90, 84)], DARK)


def chart(image, d, accent):
    _line(d, [(12, 8), (12, 84), (90, 84)], DARK)
    _line(d, [(16, 70), (34, 44), (52, 56), (70, 22), (88, 34)], accent)


def auralise(image, d, accent):
    _speaker(d, 6, 48, 0.75, DARK)
    points = [(40 + i, 48 - 22 * math.sin(i / 8) * math.exp(-i / 60)) for i in range(0, 52)]
    _line(d, points, accent)


def play(image, d, accent):
    _ring(d, 48, 48, 40, DARK, W - 2)
    d.polygon([(36, 26), (36, 70), (72, 48)], fill=accent)


def channel(image, d, accent):
    for y in (22, 58):
        d.rectangle((14, y, 82, y + 16), fill=accent)
    _line(d, [(8, 10), (8, 86)], DARK)


def target(image, d, accent):
    _ring(d, 48, 48, 38, DARK, W - 2)
    _ring(d, 48, 48, 24, accent)
    _dot(d, 48, 48, 9, accent)


def reflections(image, d, accent):
    _line(d, [(74, 8), (74, 88)], DARK, W + 2)
    _line(d, [(16, 18), (70, 48), (16, 78)], accent)
    _dot(d, 16, 18, 7, DARK)
    _dot(d, 16, 78, 7, DARK)


def fan(image, d, accent):
    for y in (10, 30, 48, 66, 86):
        _line(d, [(16, 48), (88, y)], accent, W - 1)
    _dot(d, 16, 48, 10, DARK)


def billiard(image, d, accent):
    _line(d, [(10, 70), (40, 18), (66, 74), (88, 30)], DARK, W - 2)
    _dot(d, 53, 46, 12, accent)


def gear(image, d, accent):
    teeth = []
    for i in range(16):
        t = 2 * math.pi * i / 16
        r = 40 if i % 2 == 0 else 30
        for dt in (-0.12, 0.12):
            teeth.append((48 + r * math.cos(t + dt), 48 + r * math.sin(t + dt)))
    d.polygon(teeth, fill=DARK)
    _dot(d, 48, 48, 12, accent)


GLYPHS = {
    "CA_Acoustic_Object": wall,
    "CA_Analyse_IR": decay,
    "CA_Audio_Player": play,
    "CA_Auralise": auralise,
    "CA_Channel": channel,
    "CA_Construct_Material": material,
    "CA_Deconstruct_IR": bars,
    "CA_Deconstruct_IR_Graph": graph,
    "CA_Deconstruct_Material": deconstruct_material,
    "CA_Material_Library": library,
    "CA_Model": room,
    "CA_Plot_2D": chart,
    "CA_Receiver": receiver,
    "CA_Regularized_Fitness": target,
    "CA_Simulate_Hybrid": hybrid,
    "CA_Simulate_Ray_Tracing": rays,
    "CA_Source": source,
    "CA_Visualise_Billiard": billiard,
    "CA_Visualise_Rays": fan,
    "CA_Visualise_Reflections": reflections,
    "CA_Worker": gear,
}


def draw(name):
    """The icon of the spec folder ``name``, 24 by 24 RGBA."""
    subcategory = json.loads((SPECS / name / "metadata.json").read_text(encoding="utf-8"))["subcategory"]
    accent = ACCENTS[subcategory] + (255,)
    image = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    GLYPHS[name](image, ImageDraw.Draw(image), accent)
    return image.resize((SIZE, SIZE), Image.LANCZOS)


def main():
    for name in sorted(GLYPHS):
        draw(name).save(SPECS / name / "icon.png")
        print("wrote", name)


if __name__ == "__main__":
    main()
