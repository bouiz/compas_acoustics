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
