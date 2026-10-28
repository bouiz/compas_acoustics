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


def test_importing_the_package_does_not_import_the_hybrid_method():
    code = (
        "import sys\n"
        "import compas_acoustics\n"
        "import compas_acoustics.simulation_methods\n"
        "assert 'compas_acoustics.simulation_methods.hybrid' not in sys.modules\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_a_star_import_needs_no_pyfdn_and_imports_no_hybrid_method():
    code = (
        "import sys\n"
        "for name in ('spharpy', 'pyFDN', 'multislope'):\n"
        "    sys.modules[name] = None\n"
        "from compas_acoustics import *\n"
        "from compas_acoustics.simulation_methods import *\n"
        "assert 'compas_acoustics.simulation_methods.hybrid' not in sys.modules\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
