"""The Grasshopper plugin works with the package of its compas_acoustics range and carries its schema."""

import re
from pathlib import Path

import compas_acoustics
from compas_acoustics.pb import schema_hash

ROOT = Path(__file__).parents[1]
PLUGIN = ROOT / "src" / "compas_acoustics_ghcsharp" / "CompasAcoustics.Grasshopper"


def test_the_package_version_is_in_the_plugins_range():
    # semver, as Worker.Compatible: from CompasAcousticsVersion up to its next breaking release
    csproj = (PLUGIN / "CompasAcoustics.Grasshopper.csproj").read_text()
    minimum = tuple(map(int, re.search(r"<CompasAcousticsVersion>(\d+)\.(\d+)\.(\d+)</", csproj).groups()))
    version = tuple(map(int, re.match(r"(\d+)\.(\d+)\.(\d+)", compas_acoustics.__version__).groups()))
    below = (0, minimum[1] + 1, 0) if minimum[0] == 0 else (minimum[0] + 1, 0, 0)
    assert minimum <= version < below


def test_the_plugins_schema_hash_is_the_packages():
    # regenerate both with `invoke generate` after a change to the .proto files
    generated = re.search(r'Hash = "([0-9a-f]{64})"', (PLUGIN / "Generated" / "Schema.cs").read_text()).group(1)
    assert generated == schema_hash()
