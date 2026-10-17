"""Create the worker's environment: a uv-managed CPython 3.12 with compas_acoustics, recorded in CONFIG.

Run with the Python whose Grasshopper components will use the worker (Rhino's), after
compas_acoustics is installed there; the installer does this when its ``worker`` input is set::

    python -m compas_acoustics.worker.install --wheel compas_acoustics-0.1.0-py3-none-any.whl [--hybrid]
"""

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from compas_acoustics.worker import CONFIG

WORKER_HOME = Path.home() / ".compas_acoustics" / "worker"


def _uv():
    target = Path(tempfile.mkdtemp(prefix="compas_acoustics_uv_"))
    subprocess.run([sys.executable, "-I", "-m", "pip", "--isolated", "install", "--disable-pip-version-check", "--target", str(target), "uv"], check=True)
    for candidate in ("bin/uv", "bin/uv.exe", "Scripts/uv.exe"):
        if (target / candidate).exists():
            return str(target / candidate)
    raise RuntimeError(f"uv was installed into {target}, but its executable was not found there.")


def install_worker(wheel, hybrid=False, home=WORKER_HOME, config=CONFIG, python_version="3.12"):
    """Create the worker environment in ``home`` and record its interpreter in ``config``; returns the interpreter."""
    uv = _uv()
    home, config = Path(home), Path(config)
    shutil.rmtree(home, ignore_errors=True)
    subprocess.run([uv, "venv", "--python", python_version, str(home)], check=True)
    python = home / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    requirement = "compas_acoustics{} @ {}".format("[hybrid]" if hybrid else "", Path(wheel).resolve().as_uri())
    subprocess.run([uv, "pip", "install", "--python", str(python), requirement], check=True)
    if hybrid:
        # flamo (via pyFDN) installs the abandoned PySoundFile over soundfile
        subprocess.run([uv, "pip", "install", "--python", str(python), "--reinstall", "--no-deps", "soundfile"], check=True)
    # The first import of the scientific stack takes about 30 s; do it here, not on the first Grasshopper call
    subprocess.run([str(python), "-c", "import compas_acoustics.methods as m; m.Hybrid" if hybrid else "import compas_acoustics.worker.api"], check=True)
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text(json.dumps({"python": str(python), "hybrid": hybrid}))
    return python


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--wheel", required=True)
    parser.add_argument("--hybrid", action="store_true", help="also install the hybrid extra (PyTorch, several GB)")
    parser.add_argument("--home", default=str(WORKER_HOME))
    parser.add_argument("--config", default=str(CONFIG))
    args = parser.parse_args()
    print("Worker Python:", install_worker(args.wheel, args.hybrid, args.home, args.config))
