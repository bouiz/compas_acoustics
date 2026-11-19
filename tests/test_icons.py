"""The Grasshopper components' icons: one generated set (tools/icons.py), current on disk."""

import importlib.util
import json
from pathlib import Path

import pytest

PIL = pytest.importorskip("PIL")

ROOT = Path(__file__).resolve().parents[1]
SPECS = ROOT / "src" / "compas_acoustics_ghcsharp" / "CompasAcoustics.Grasshopper" / "Components" / "Specs"


def _icons():
    spec = importlib.util.spec_from_file_location("icons", ROOT / "tools" / "icons.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _names():
    return sorted(p.parent.name for p in SPECS.glob("*/metadata.json"))


def test_every_spec_has_a_glyph_and_every_subcategory_an_accent():
    icons = _icons()
    assert sorted(icons.GLYPHS) == _names()
    subcategories = {json.loads((SPECS / name / "metadata.json").read_text())["subcategory"] for name in _names()}
    assert subcategories <= set(icons.ACCENTS)


def test_every_spec_has_a_generated_icon():
    from PIL import Image

    icons = _icons()
    drawn = {}
    for name in _names():
        image = Image.open(SPECS / name / "icon.png")
        assert image.size == (24, 24)
        assert image.mode == "RGBA"
        assert sum(1 for a in image.getchannel("A").tobytes() if a > 128) >= 40, name
        assert image.tobytes() == icons.draw(name).tobytes(), f"{name}: run python tools/icons.py"
        drawn[name] = image.tobytes()
    assert len(set(drawn.values())) == len(drawn)


def test_icons_are_reproducible():
    icons = _icons()
    for name in _names():
        assert icons.draw(name).tobytes() == icons.draw(name).tobytes()


def test_the_docs_have_each_icon_at_four_times_the_size():
    from PIL import Image

    icons = _icons()
    for name in _names():
        image = Image.open(ROOT / "docs" / "_images" / "icons" / (name + ".png"))
        assert image.size == (96, 96), name
        assert image.tobytes() == icons.draw(name, size=96).tobytes(), f"{name}: run python tools/icons.py"
