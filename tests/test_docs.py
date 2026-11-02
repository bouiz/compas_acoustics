"""The docs and the sound-library README match the package."""

import pathlib
import re

import compas_acoustics

ROOT = pathlib.Path(__file__).parents[1]


def read(rel):
    return (ROOT / rel).read_text(encoding="utf-8")


def test_the_api_reference_lists_every_public_name():
    listed = set(re.findall(r"^\s{4}(\w+)\s*$", read("docs/api/compas_acoustics.rst"), re.M))
    assert set(compas_acoustics.__all__) | {"Binaural", "Ambisonics"} <= listed


EXAMPLES = ["simple_simulation_shoebox", "obj", "rays", "scattered_rays", "specular_reflections"]


def test_every_example_has_a_docs_page_and_the_tutorial_has_content():
    for name in EXAMPLES:
        assert f"/../examples/{name}.py" in read(f"docs/python/examples/{name}.rst")
    assert read("docs/python/tutorial.rst").count(".. code-block:: python") >= 4


def test_the_sound_library_readme_lists_the_files_the_package_has():
    sounds = ROOT / "src/compas_acoustics/anechoic_soundfiles_library"
    listed = set(re.findall(r"`(\w+\.wav)`", read("src/compas_acoustics/anechoic_soundfiles_library/README.md")))
    assert listed == {p.name for p in sounds.glob("*.wav")}
