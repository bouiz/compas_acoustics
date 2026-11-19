"""Capture the Grasshopper images of the documentation. Run it in Rhino 8 (ScriptEditor), with the current plugin installed.

It captures what is missing or out of date, as docs/_images/fingerprints.json tells:

* each component on an empty canvas, to docs/_images/components/<CA_name>.png;
* each examples/<name>.ghx: its canvas (canvas.png) and the Rhino viewport with its preview (viewport.png), in
  docs/_images/examples/<name>/, once its solution and background runs have finished.

and records their fingerprints. Set ALL = True to capture everything. Your open Grasshopper document is restored at
the end. See docs/contributing/docs.md.
"""

import os
import sys
import tempfile
import time
import traceback
from pathlib import Path

import clr  # noqa: F401
import Grasshopper
import Rhino
import System
from System.Drawing import Color
from System.Drawing import PointF
from System.Drawing import RectangleF
from System.Drawing import Size
from System.Drawing.Imaging import ImageFormat

ALL = False

MARGIN = 20.0  # canvas units around what is captured
ZOOM = 2.0  # pixels per canvas unit, for a component
CANVAS_ZOOM = 1.0  # for an example's whole canvas, which is much larger
VIEWPORT = (1600, 1000)
SETTLE = 3.0  # seconds a solved .ghx must stay idle before its capture
# The Worker component shows the worker's Python, which must exist. To show a neutral path instead of this computer's,
# the capture points the variable at a relative path and runs in a temporary folder that has an empty file there.
SHOWN_PYTHON = "~/.compas_acoustics/workers/0.1.0/bin/python"
PYTHON_VARIABLE = "COMPAS_ACOUSTICS_WORKER_PYTHON"

try:
    REPO = Path(__file__).resolve().parents[2]
except NameError:  # run through lamcp, which has no __file__
    REPO = Path(os.environ["COMPAS_ACOUSTICS_REPO"])
sys.path.insert(0, str(REPO / "docs" / "_ext"))

from ghcomponents import examples  # noqa: E402
from ghcomponents import fingerprints  # noqa: E402
from ghcomponents import specs  # noqa: E402

IMAGES = REPO / "docs" / "_images"


def log(message):
    print(message)
    Rhino.RhinoApp.WriteLine("capture_grasshopper: " + message)


def _viewport(canvas, bounds, zoom):
    vp = Grasshopper.GUI.Canvas.GH_Viewport(canvas.Viewport)
    vp.Width = int((bounds.Width + 2 * MARGIN) * zoom)
    vp.Height = int((bounds.Height + 2 * MARGIN) * zoom)
    vp.Zoom = zoom
    vp.Tx = -(bounds.X - MARGIN) * zoom  # the translation is in pixels, after the zoom
    vp.Ty = -(bounds.Y - MARGIN) * zoom
    return vp


def _bounds(objects):
    rects = [RectangleF(o.Attributes.Bounds) for o in objects]
    result = rects[0]
    for r in rects[1:]:
        result = RectangleF.Union(result, r)
    return result


def render(canvas, objects, path, zoom=ZOOM):
    """Render the canvas area of ``objects`` to a PNG; the first render lays them out, the second is kept."""
    canvas.GenerateHiResImageTile(_viewport(canvas, _bounds(objects), zoom), Color.White).Dispose()
    bitmap = canvas.GenerateHiResImageTile(_viewport(canvas, _bounds(objects), zoom), Color.White)
    path.parent.mkdir(parents=True, exist_ok=True)
    bitmap.Save(str(path), ImageFormat.Png)
    bitmap.Dispose()


def capture_component(canvas, component, guid):
    document = Grasshopper.Kernel.GH_Document()
    obj = Grasshopper.Instances.ComponentServer.EmitObject(System.Guid(guid))
    obj.CreateAttributes()
    obj.Attributes.Pivot = PointF(100, 100)
    document.AddObject(obj, False)
    canvas.Document = document  # this solves it: clear what the solution left, its warning colour among it
    document.Enabled = False
    for item in [obj] + list(obj.Params):
        item.ClearData()
        item.ClearRuntimeMessages()
        item.Phase = Grasshopper.Kernel.GH_SolutionPhase.Blank
    obj.Message = None
    render(canvas, [obj], IMAGES / "components" / (component.key + ".png"))
    document.Dispose()


def _running(document):
    """Whether a component of ``document`` still waits for the worker."""
    flags = System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance
    for obj in document.Objects:
        runs = obj.GetType().GetProperty("Runs", flags)
        if runs is not None and runs.GetValue(obj).AnyRunning:
            return True
    return False


def _preview_box(document):
    box = Rhino.Geometry.BoundingBox.Empty
    for obj in document.Objects:
        if isinstance(obj, Grasshopper.Kernel.IGH_PreviewObject) and not obj.Hidden and obj.IsPreviewCapable:
            clip = obj.ClippingBox
            if clip.IsValid:
                box.Union(clip)
    return box


def capture_viewport(document, path):
    view = Rhino.RhinoDoc.ActiveDoc.Views.Find("Perspective", False) or Rhino.RhinoDoc.ActiveDoc.Views.ActiveView
    Rhino.RhinoDoc.ActiveDoc.Views.ActiveView = view
    shaded = Rhino.Display.DisplayModeDescription.FindByName("Shaded")
    if shaded is not None:
        view.ActiveViewport.DisplayMode = shaded
    box = _preview_box(document)
    if box.IsValid:
        view.ActiveViewport.ZoomBoundingBox(box)
    view.Redraw()
    bitmap = view.CaptureToBitmap(Size(*VIEWPORT))
    path.parent.mkdir(parents=True, exist_ok=True)
    bitmap.Save(str(path), ImageFormat.Png)
    bitmap.Dispose()


class ExampleQueue:
    """Opens each .ghx in turn and captures it once it has solved and stayed idle for SETTLE seconds."""

    def __init__(self, canvas, original, items, record):
        self.canvas, self.original, self.items, self.record = canvas, original, list(items), record
        self.document = None
        self.idle_since = None
        self.handler = System.EventHandler(self.on_idle)

    def start(self):
        Rhino.RhinoApp.Idle += self.handler
        self.next()

    def next(self):
        if not self.items:
            Rhino.RhinoApp.Idle -= self.handler
            self.canvas.Document = self.original
            log("done")
            return
        self.example = self.items.pop(0)
        io = Grasshopper.Kernel.GH_DocumentIO()
        if not io.Open(str(self.example.ghx)):
            log("could not open {}".format(self.example.ghx))
            return self.next()
        self.document = io.Document
        Grasshopper.Instances.DocumentServer.AddDocument(self.document)
        self.canvas.Document = self.document
        self.document.Enabled = True
        self.document.NewSolution(True)
        self.idle_since = None
        log("solving {}".format(self.example.name))

    def on_idle(self, sender, args):
        try:
            busy = self.document.SolutionState != Grasshopper.Kernel.GH_ProcessStep.PostProcess or _running(self.document)
            if busy:
                self.idle_since = None
                return
            if self.idle_since is None:
                self.idle_since = time.time()
                return
            if time.time() - self.idle_since < SETTLE:
                return
            folder = IMAGES / "examples" / self.example.name
            render(self.canvas, list(self.document.Objects), folder / "canvas.png", CANVAS_ZOOM)
            capture_viewport(self.document, folder / "viewport.png")
            fingerprints.record_file(self.record, "ghx", self.example.ghx)
            fingerprints.save(self.record)
            log("captured {}".format(self.example.name))
            Grasshopper.Instances.DocumentServer.RemoveDocument(self.document)
            self.document.Dispose()
            self.next()
        except Exception:
            Rhino.RhinoApp.Idle -= self.handler
            self.canvas.Document = self.original
            log(traceback.format_exc())


def main(capture_all=None):
    capture_all = ALL if capture_all is None else capture_all
    canvas = Grasshopper.Instances.ActiveCanvas
    original = canvas.Document
    record = fingerprints.load()
    components = specs.load_components(specs.SPECS_DIR)
    guids = specs.read_component_guids(specs.CSHARP_DIR)
    stale = {s.name for s in fingerprints.stale_components(components, record, IMAGES)}
    python = System.Environment.GetEnvironmentVariable(PYTHON_VARIABLE)
    cwd = System.Environment.CurrentDirectory
    stand_in = Path(tempfile.mkdtemp()) / SHOWN_PYTHON
    stand_in.parent.mkdir(parents=True)
    stand_in.touch()
    System.Environment.CurrentDirectory = str(stand_in.parents[len(Path(SHOWN_PYTHON).parts) - 1])
    System.Environment.SetEnvironmentVariable(PYTHON_VARIABLE, SHOWN_PYTHON)
    try:
        for component in components:
            if capture_all or component.key in stale:
                capture_component(canvas, component, guids[component.key])
                fingerprints.record_component(record, component)
                fingerprints.save(record)
                log("captured {}".format(component.key))
    finally:
        System.Environment.SetEnvironmentVariable(PYTHON_VARIABLE, python)
        System.Environment.CurrentDirectory = cwd
        canvas.Document = original

    ghx = [e for e in examples.discover(REPO / "examples") if e.ghx is not None]
    stale = {s.name for s in fingerprints.stale_files("ghx", [e.ghx for e in ghx], record, IMAGES)}
    queue = ExampleQueue(canvas, original, [e for e in ghx if capture_all or e.name in stale], record)
    queue.start()
    return queue


if __name__ == "__main__":
    main()
