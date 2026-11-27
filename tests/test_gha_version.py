"""The Grasshopper plugin works with the package of its compas_acoustics range and carries its schema."""

import re
from pathlib import Path

import compas_acoustics
from compas_acoustics.pb import schema_hash

ROOT = Path(__file__).parents[1]
PLUGIN = ROOT / "src" / "compas_acoustics_ghcsharp" / "CompasAcoustics.Grasshopper"


def test_the_plugin_requires_the_package_version():
    # CompasAcousticsVersion is the minimum the plugin accepts: always the current release (bump-my-version keeps it in step)
    csproj = (PLUGIN / "CompasAcoustics.Grasshopper.csproj").read_text()
    minimum = re.search(r"<CompasAcousticsVersion>([^<]+)</", csproj).group(1)
    assert minimum == compas_acoustics.__version__


def test_the_plugin_has_the_package_version():
    # the plugin is released with the library: bump-my-version keeps <Version> in step
    csproj = (PLUGIN / "CompasAcoustics.Grasshopper.csproj").read_text()
    assert re.search(r"<Version>([^<]+)</Version>", csproj).group(1) == compas_acoustics.__version__


def test_the_plugins_schema_hash_is_the_packages():
    # regenerate both with `invoke generate` after a change to the .proto files
    generated = re.search(r'Hash = "([0-9a-f]{64})"', (PLUGIN / "Generated" / "Schema.cs").read_text()).group(1)
    assert generated == schema_hash()
