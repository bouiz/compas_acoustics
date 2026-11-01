"""Run one example script headless and keep what it shows: ``python docs/_tools/run_example.py <script> <out_dir>``.

Every matplotlib figure the script shows is saved as figure_<n>.png, what it prints as stdout.txt; a COMPAS viewer
it opens is not shown. ``invoke docs-images`` runs this for each example whose output is out of date.
"""

from __future__ import annotations

import contextlib
import io
import os
import random
import runpy
import sys
from pathlib import Path


def run_example(script: Path, out_dir: Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.iterdir():
        if old.is_file():
            old.unlink()

    os.environ["MPLBACKEND"] = "Agg"
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    random.seed(0)
    np.random.seed(0)

    count = [0]

    def save_figures(*args, **kwargs):
        for number in plt.get_fignums():
            count[0] += 1
            plt.figure(number).savefig(out_dir / "figure_{}.png".format(count[0]), dpi=150, bbox_inches="tight")
        plt.close("all")

    shown = plt.show
    plt.show = save_figures
    try:
        import compas_viewer

        viewer_show = compas_viewer.Viewer.show
        compas_viewer.Viewer.show = lambda self, *args, **kwargs: None
    except ImportError:
        compas_viewer = None

    stdout = io.StringIO()
    try:
        with contextlib.redirect_stdout(stdout):
            runpy.run_path(str(script), run_name="__main__")
        save_figures()
    finally:
        plt.show = shown
        if compas_viewer is not None:
            compas_viewer.Viewer.show = viewer_show
    (out_dir / "stdout.txt").write_text(stdout.getvalue(), encoding="utf-8")


if __name__ == "__main__":
    run_example(Path(sys.argv[1]), Path(sys.argv[2]))
