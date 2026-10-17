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
    assert "compas_acoustics/hrtfs_library/KEMAR_Knowl_EarSim_SmallEars_FreeFieldComp_48kHz.sofa" in names
    assert "compas_acoustics/hybrid/diffraction.py" in names
    assert "compas_acoustics/methods/pra.py" in names
    assert not [n for n in names if n.startswith("compas_acoustics_ghpython/components/") and n.endswith("code.py")]
