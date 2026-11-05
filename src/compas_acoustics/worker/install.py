"""Create the worker's environment: a uv-managed CPython 3.12 with compas_acoustics, recorded in CONFIG.

For Python scripts; in Grasshopper the Install Worker component does this. Run with a Python that has
compas_acoustics, with the wheel to install::

    python -m compas_acoustics.worker.install --wheel compas_acoustics-0.1.0-py3-none-any.whl [--hybrid]
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from compas_acoustics.worker import CONFIG
from compas_acoustics.worker import stop_running_workers

__all__ = ["WORKER_HOME", "install_worker"]

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
    # Run from Rhino, PYTHONPATH is on its 3.9 packages; the worker's 3.12 must not load them
    env = {name: value for name, value in os.environ.items() if name not in ("PYTHONHOME", "PYTHONPATH")}
    # A running worker holds the files of the environment open, which Windows does not let go of
    stop_running_workers()
    shutil.rmtree(home, ignore_errors=True)
    subprocess.run([uv, "venv", "--python", python_version, str(home)], check=True, env=env)
    python = home / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    # On Python >= 3.11 compas_acoustics requires the hybrid method's packages too
    subprocess.run([uv, "pip", "install", "--python", str(python), "compas_acoustics @ {}".format(Path(wheel).resolve().as_uri())], check=True, env=env)
    # The first import of the scientific stack is slow; do it here, not on the first Grasshopper call
    subprocess.run([str(python), "-c", "import compas_acoustics.simulation_methods as m; m.Binaural" if hybrid else "import compas_acoustics.worker.api"], check=True, env=env)
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text(json.dumps({"python": str(python), "hybrid": hybrid}))
    return python


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--wheel", required=True)
    parser.add_argument("--hybrid", action="store_true", help="check that the binaural and ambisonics methods import (pyFDN comes with compas_acoustics on Python >= 3.11)")
    parser.add_argument("--home", default=str(WORKER_HOME))
    parser.add_argument("--config", default=str(CONFIG))
    args = parser.parse_args()
    print("Worker Python:", install_worker(args.wheel, args.hybrid, args.home, args.config))
