"""The Grasshopper plugin and the package it talks to carry one version and one schema."""

import re
from pathlib import Path

import compas_acoustics
from compas_acoustics.pb import schema_hash

ROOT = Path(__file__).parents[1]
PLUGIN = ROOT / "src" / "compas_acoustics_ghcsharp" / "CompasAcoustics.Grasshopper"


def test_the_assembly_version_equals_the_package_version():
    version = re.search(r"<Version>([^<]+)</Version>", (PLUGIN / "CompasAcoustics.Grasshopper.csproj").read_text()).group(1)
    assert version == compas_acoustics.__version__


def test_the_plugins_schema_hash_is_the_packages():
    # regenerate both with `invoke generate` after a change to the .proto files
    generated = re.search(r'Hash = "([0-9a-f]{64})"', (PLUGIN / "Generated" / "Schema.cs").read_text()).group(1)
    assert generated == schema_hash()


def test_bumpversion_updates_the_plugin_too():
    assert "src/compas_acoustics_ghcsharp/CompasAcoustics.Grasshopper/CompasAcoustics.Grasshopper.csproj" in (ROOT / "pyproject.toml").read_text()
