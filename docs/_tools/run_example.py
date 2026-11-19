"""Run one example script headless and keep what it shows: ``python docs/_tools/run_example.py <script> <out_dir>``.

Every matplotlib figure the script shows is saved as figure_<n>.png, every COMPAS viewer it opens as viewer_<n>.png
(the window opens for a moment, is captured and closes), and what it prints as stdout.txt. ``invoke docs-images`` runs this for each example whose output is out of date.
"""

from __future__ import annotations

import contextlib
import io
import os
import random
import runpy
import sys
from pathlib import Path

VIEWER_SIZE = (1600, 1000)
VIEWER_DELAY_MS = 1500  # long enough for the first frames to render
VIEWER_DISTANCE = 1.5  # camera distance, in diagonals of the scene's bounding box


def _points(item):
    """The points of a scene object's geometry, for framing the camera."""
    if hasattr(item, "vertices_attributes"):
        return item.vertices_attributes("xyz")
    if hasattr(item, "points"):
        return list(item.points)
    if hasattr(item, "start") and hasattr(item, "end"):
        return [item.start, item.end]
    if hasattr(item, "x") and hasattr(item, "z"):
        return [item]
    return []


def _frame(viewer):
    """Aim the camera at the centre of everything in the scene, from a distance that fits it."""
    points = [list(p) for obj in viewer.scene.objects for p in _points(getattr(obj, "item", None))]
    if not points:
        return
    low = [min(p[i] for p in points) for i in range(3)]
    high = [max(p[i] for p in points) for i in range(3)]
    diagonal = sum((h - lo) ** 2 for lo, h in zip(low, high)) ** 0.5
    camera = viewer.renderer.camera
    camera.target = [(lo + h) / 2 for lo, h in zip(low, high)]
    camera.distance = max(diagonal * VIEWER_DISTANCE, 1.0)


def _capturing_show(out_dir: Path):
    """A ``Viewer.show`` that renders the scene, saves it as viewer_<n>.png and closes the window."""
    count = [0]

    def show(self, *args, **kwargs):
        from PySide6.QtCore import QTimer

        def capture():
            count[0] += 1
            self.renderer.grabFramebuffer().save(str(out_dir / "viewer_{}.png".format(count[0])), "png")
            self.app.quit()

        self.running = True
        self.ui.init()
        self.ui.window.widget.resize(*VIEWER_SIZE)
        _frame(self)
        QTimer.singleShot(VIEWER_DELAY_MS, capture)
        self.app.exec()
        self.ui.window.widget.close()

    return show


def run_example(script: Path, out_dir: Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # only this script's outputs: a .ghx of the same name keeps its canvas.png and viewport.png here
    for old in list(out_dir.glob("figure_*.png")) + list(out_dir.glob("viewer_*.png")) + list(out_dir.glob("stdout.txt")):
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
        compas_viewer.Viewer.show = _capturing_show(out_dir)
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
