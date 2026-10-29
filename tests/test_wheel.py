import importlib.util
import pathlib
import subprocess
import sys
import zipfile

import pytest

ROOT = pathlib.Path(__file__).parents[1]


@pytest.mark.slow
def test_the_wheel_ships_the_new_modules(tmp_path):
    # a ``build/`` folder left by an earlier build is importable as a namespace package: look for the tool itself
    if importlib.util.find_spec("build") is None or importlib.util.find_spec("build.__main__") is None:
        pytest.skip("needs the build package")
    subprocess.run([sys.executable, "-m", "build", "--wheel", "--outdir", str(tmp_path), str(ROOT)], check=True, capture_output=True)
    names = zipfile.ZipFile(next(tmp_path.glob("compas_acoustics-*.whl"))).namelist()
    assert "compas_acoustics/hrtfs_library/KEMAR_FreeFieldComp_48kHz.sofa" in names
    assert "compas_acoustics/hrtfs_library/README.md" in names
    assert "compas_acoustics/hybrid/diffraction.py" in names
    assert "compas_acoustics/simulation_methods/pra.py" in names
    assert "compas_acoustics/worker/service.py" in names
    # the schema and its Python bindings; the Grasshopper plugin ships in the installer bundle, not in the wheel
    assert "compas_acoustics/pb/protobuf_defs/compas_acoustics/pb/generated/acoustics.proto" in names
    assert "compas_acoustics/pb/generated/acoustics_pb2.py" in names
    entry_points = next(n for n in names if n.endswith(".dist-info/entry_points.txt"))
    assert "compas_acoustics.pb.conversions" in zipfile.ZipFile(next(tmp_path.glob("compas_acoustics-*.whl"))).read(entry_points).decode()
