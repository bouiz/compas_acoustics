"""What a first-time visitor reads: package metadata, docs and repository files carry no template leftovers."""

import pathlib

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
