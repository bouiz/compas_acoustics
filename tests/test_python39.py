"""What Rhino 8 (CPython 3.9, pyfar < 0.8, pyrato < 1.0, no spharpy) needs to keep working."""

import subprocess
import sys


def test_the_package_imports_without_optional_dependencies():
    code = (
        "import sys\n"
        "for name in ('spharpy', 'pyFDN', 'multislope'):\n"
        "    sys.modules[name] = None\n"
        "import compas_acoustics\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_binaural_without_spharpy_names_the_requirement():
    code = (
        "import sys\nsys.modules['spharpy'] = None\n"
        "import compas_acoustics.binaural as b\n"
        "try:\n    b.magls_filters(b.DEFAULT_HRTF, 48000, 1)\n"
        "except ImportError as e:\n    assert '3.11' in str(e), e\nelse:\n    raise AssertionError('no ImportError')\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_importing_the_package_does_not_import_the_hybrid_method():
    code = (
        "import sys\n"
        "import compas_acoustics\n"
        "import compas_acoustics.simulation_methods\n"
        "assert 'compas_acoustics.simulation_methods.hybrid' not in sys.modules\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
