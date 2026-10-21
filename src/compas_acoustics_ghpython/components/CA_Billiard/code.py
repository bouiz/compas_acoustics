import time

import Grasshopper
import Rhino
import scriptcontext as sc

from compas_acoustics_ghpython import ensure_inputs
from compas_acoustics_ghpython.display import POINT_STYLE
from compas_acoustics_ghpython.display import bounding_box
from compas_acoustics_ghpython.display import order_color
from compas_acoustics_ghpython.display import typed_list

FRAME_MS = 33
PARTICLE_RADIUS = 4


def _now():
    return time.time()


class BilliardComponent(Grasshopper.Kernel.GH_ScriptInstance):

    def RunScript(self, ray_field, time, play, speed):
        ensure_inputs(ghenv.Component, {"play": None})
        self._drawn = {}
        self._box = bounding_box([])

        if ray_field is None or len(ray_field) == 0:
            ghenv.Component.Message = ""
            text = "Input parameter 'ray_field' failed to collect data." if ray_field is None else "The ray field has no rays."
            ghenv.Component.AddRuntimeMessage(Grasshopper.Kernel.GH_RuntimeMessageLevel.Warning, text)
            return (None,) * 3

        duration = ray_field.duration_ms
        key = "compas_acoustics.billiard." + str(ghenv.Component.InstanceGuid)

        if speed is not None and speed <= 0:
            ghenv.Component.AddRuntimeMessage(
                Grasshopper.Kernel.GH_RuntimeMessageLevel.Remark, "speed must be positive; using the default."
            )
            speed = None
        # Resolved on every solve: a retraced field changes the default end and speed
        speed = float(duration / 10.0 if speed is None else speed)
        end = min(max(float(duration if time is None else time), 0.0), duration)

        # A Boolean Toggle stays True: only its change to True starts playback, as a button's press does
        pressed = bool(play) and not sc.sticky.get(key + ".play", False)
        sc.sticky[key + ".play"] = bool(play)
        if pressed:
            sc.sticky[key] = {"start": _now(), "offset": 0.0, "time": time, "speed": speed}

        state = sc.sticky.get(key)
        if state is not None and not pressed and time != state["time"]:
            # Moving the slider during or after playback stops it and scrubs
            del sc.sticky[key]
            state = None

        if state is not None:
            now = _now()
            if speed != state["speed"]:
                # A new speed runs on from the frame shown, not from 0 at the new rate
                state["offset"] = min(state["offset"] + (now - state["start"]) * state["speed"], end)
                state["start"], state["speed"] = now, speed
            t = min(max(state["offset"] + (now - state["start"]) * speed, 0.0), end)
            if t < end:
                self._schedule_next_frame()
        elif time is not None:
            t = min(max(float(time), 0.0), duration)
        else:
            t = 0.0

        if state is None and time is None:
            ghenv.Component.Message = "press Play or connect time"
        else:
            ghenv.Component.Message = "t = {:.1f} ms".format(t)

        points, _, orders = ray_field.particles(t)
        particles = [Rhino.Geometry.Point3d(*point) for point in points]
        for particle, order in zip(particles, orders):
            self._drawn.setdefault(order, []).append(particle)
        self._drawn = {order: typed_list(Rhino.Geometry.Point3d, group) for order, group in self._drawn.items()}
        # The whole field, not the particles: the box stays still while they move
        self._box = bounding_box([Rhino.Geometry.Point3d(*point) for ray in ray_field for point in ray.points])
        return particles, t, [order_color(order) for order in orders]

    @property
    def ClippingBox(self):
        box = getattr(self, "_box", None)
        return bounding_box([]) if box is None else box

    def DrawViewportWires(self, args):
        for order, particles in getattr(self, "_drawn", {}).items():
            args.Display.DrawPoints(particles, POINT_STYLE, PARTICLE_RADIUS, order_color(order))

    @staticmethod
    def _schedule_next_frame():
        component = ghenv.Component
        document = component.OnPingDocument()
        if document is None:
            return

        def next_frame(gh_document):
            component.ExpireSolution(False)

        document.ScheduleSolution(FRAME_MS, next_frame)
