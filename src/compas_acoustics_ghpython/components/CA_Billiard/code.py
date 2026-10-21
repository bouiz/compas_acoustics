import time

import Grasshopper
import Rhino
import scriptcontext as sc

from compas_acoustics_ghpython import ensure_inputs

FRAME_MS = 33


def _now():
    return time.time()


def _polyline(ray):
    polyline = Rhino.Geometry.Polyline()
    polyline.AddRange([Rhino.Geometry.Point3d(*point) for point in ray.points])
    return polyline


class BilliardComponent(Grasshopper.Kernel.GH_ScriptInstance):

    def RunScript(self, ray_field, time, play, speed):
        ensure_inputs(ghenv.Component, {"play": None})

        if ray_field is None or len(ray_field) == 0:
            ghenv.Component.AddRuntimeMessage(Grasshopper.Kernel.GH_RuntimeMessageLevel.Warning, "Connect a ray field.")
            return (None,) * 3

        duration = ray_field.duration_ms
        key = "compas_acoustics.billiard." + str(ghenv.Component.InstanceGuid)

        if speed is not None and speed <= 0:
            ghenv.Component.AddRuntimeMessage(
                Grasshopper.Kernel.GH_RuntimeMessageLevel.Remark, "speed must be positive; using the default."
            )
            speed = None

        if play:
            sc.sticky[key] = {
                "start": _now(),
                "end": float(duration if time is None else time),
                "speed": float(duration / 10.0 if speed is None else speed),
            }

        state = sc.sticky.get(key)
        if state is not None:
            end = min(state["end"], duration)
            t = min(max((_now() - state["start"]) * state["speed"], 0.0), end)
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

        points, _ = ray_field.particles(t)
        particles = [Rhino.Geometry.Point3d(*point) for point in points]
        trails = [_polyline(ray) for ray in ray_field.truncated(travel_time_ms=t)]
        return particles, trails, t

    @staticmethod
    def _schedule_next_frame():
        component = ghenv.Component
        document = component.OnPingDocument()
        if document is None:
            return

        def next_frame(gh_document):
            component.ExpireSolution(False)

        document.ScheduleSolution(FRAME_MS, next_frame)
