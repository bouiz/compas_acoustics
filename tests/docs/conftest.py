"""Helpers for the tests of the docs generator in docs/_ext/ghcomponents."""

import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).parents[2]
sys.path.insert(0, str(ROOT / "docs" / "_ext"))


@pytest.fixture
def fixtures():
    return pathlib.Path(__file__).parent / "fixtures"
