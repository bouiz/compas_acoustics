"""What a first-time visitor reads: package metadata, docs and repository files carry no template leftovers."""

import pathlib
import re

import compas_acoustics

ROOT = pathlib.Path(__file__).parents[1]


def read(rel):
    return (ROOT / rel).read_text(encoding="utf-8")


def test_the_license_names_its_copyright_holder():
    assert "Copyright (c) 2025 Gramazio Kohler Research, ETH Zurich" in read("LICENSE")


def test_pyproject_has_description_keywords_and_classifiers():
    text = read("pyproject.toml")
    assert 'description = ""' not in text and "keywords = []" not in text
    assert "License :: OSI Approved :: MIT License" in text
    assert "Operating System :: MacOS" in text
    assert "Development Status :: 4 - Beta" in text
    assert "[tool.black]" not in text


def test_the_docs_copyright_is_set():
    assert 'copyright = ""' not in read("docs/conf.py")


def test_no_black_and_no_tools_leftovers():
    assert "black" not in read("requirements-dev.txt")
    assert "Tools" not in read("src/compas_acoustics/__init__.py")


LEAD = "compas_acoustics simulates, analyses and auralises the acoustics of rooms modelled with COMPAS"


def test_readme_and_docs_open_with_the_lead():
    assert LEAD in read("README.md").split("## Installation")[0]
    assert LEAD in read("docs/index.rst")
    assert ".. .. figure::" not in read("docs/index.rst")


def test_the_api_reference_lists_every_public_name():
    listed = set(re.findall(r"^\s{4}(\w+)\s*$", read("docs/api/compas_acoustics.rst"), re.M))
    assert set(compas_acoustics.__all__) | {"Binaural", "Ambisonics"} <= listed


EXAMPLES = ["shoebox", "obj", "rays", "scattered_rays", "specular_reflections"]


def test_every_example_has_a_docs_page_and_the_tutorial_has_content():
    for name in EXAMPLES:
        assert f"../../examples/{name}.py" in read(f"docs/examples/{name}.rst")
    assert read("docs/tutorial.rst").count(".. code-block:: python") >= 4


def test_installation_docs_match_how_the_package_is_distributed():
    text = read("docs/installation.rst")
    assert "compas_tna" not in text
    assert "compas_acoustics_installer.zip" in text
    assert "pip install compas_acoustics\n" not in text  # not on PyPI for 0.1.0
