"""The sound-library README matches the package."""

import pathlib
import re

ROOT = pathlib.Path(__file__).parents[1]


def test_the_sound_library_readme_lists_the_files_the_package_has():
    sounds = ROOT / "src/compas_acoustics/anechoic_soundfiles_library"
    listed = set(re.findall(r"`(\w+\.wav)`", (sounds / "README.md").read_text(encoding="utf-8")))
    assert listed == {p.name for p in sounds.glob("*.wav")}
