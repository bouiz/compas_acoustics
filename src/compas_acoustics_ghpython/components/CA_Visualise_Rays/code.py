import random
import warnings

import Grasshopper
import Rhino

from compas_acoustics import RayField
from compas_acoustics import ScatteredRay
from compas_acoustics_ghpython.display import bounding_box
from compas_acoustics_ghpython.display import order_color
from compas_acoustics_ghpython.display import typed_list

LINE_THICKNESS = 1


def _polyline(ray):
    polyline = Rhino.Geometry.Polyline()
    polyline.AddRange([Rhino.Geometry.Point3d(*point) for point in ray.points])
    return polyline


def _segments(ray):
    """Each segment of a ray with the reflection order it travels after."""
    # A scattered ray is born at its parent's hit after that many reflections
    first = ray.reflection_order if isinstance(ray, ScatteredRay) else 0
    points = [Rhino.Geometry.Point3d(*point) for point in ray.points]
    return [(first + k, Rhino.Geometry.Line(start, end)) for k, (start, end) in enumerate(zip(points, points[1:]))]


class VisualiseRays(Grasshopper.Kernel.GH_ScriptInstance):

    def RunScript(self, scene, source_idx, n_rays, max_order, direction, angle, scattering, seed, order):
        self._drawn = {}
        self._box = bounding_box([])

        empty = (None,) * 3
        if scene is None:
            ghenv.Component.AddRuntimeMessage(
                Grasshopper.Kernel.GH_RuntimeMessageLevel.Warning, "Input parameter 'scene' failed to collect data."
            )
            return empty

        count = len(scene.sources)
        if source_idx is not None and not 0 <= source_idx < count:
            ghenv.Component.AddRuntimeMessage(
                Grasshopper.Kernel.GH_RuntimeMessageLevel.Warning,
                "No source at index {}; the scene has {} source(s), 0 to {}.".format(source_idx, count, count - 1),
            )
            return empty

        if n_rays is None:
            n_rays = 500
        if max_order is None:
            max_order = 10
        if seed is None:
            # Only the scattered rays are random; an empty input draws a new seed each run
            seed = random.randrange(2**31)

        cone = {}
        if direction is not None:
            cone["direction"] = [direction.X, direction.Y, direction.Z]
        if angle is not None:
            cone["angle"] = angle

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                field = RayField.from_scene(
                    scene,
                    sources=None if source_idx is None else [scene.sources[source_idx]],
                    n_rays=n_rays,
                    max_reflection_order=max_order,
                    include_scattering=bool(scattering),
                    random_seed=seed,
                    **cone
                )
            except ValueError as e:
                ghenv.Component.AddRuntimeMessage(Grasshopper.Kernel.GH_RuntimeMessageLevel.Error, str(e))
                return empty
        # Sources outside the room
        for warning in caught:
            if issubclass(warning.category, UserWarning):
                ghenv.Component.AddRuntimeMessage(Grasshopper.Kernel.GH_RuntimeMessageLevel.Warning, str(warning.message))

        shown = field if order is None else field.truncated(reflection_order=int(order))

        ghenv.Component.Message = "{} rays, order={}, seed={}".format(
            len(field), max_order if order is None else int(order), seed
        )

        for ray in list(shown) + shown.scattered_rays:
            for k, line in _segments(ray):
                self._drawn.setdefault(k, []).append(line)
        self._drawn = {k: typed_list(Rhino.Geometry.Line, lines) for k, lines in self._drawn.items()}
        self._box = bounding_box([Rhino.Geometry.Point3d(*point) for ray in list(shown) + shown.scattered_rays for point in ray.points])

        return field, [_polyline(ray) for ray in shown], [_polyline(ray) for ray in shown.scattered_rays]

    @property
    def ClippingBox(self):
        box = getattr(self, "_box", None)
        return bounding_box([]) if box is None else box

    def DrawViewportWires(self, args):
        for k, lines in getattr(self, "_drawn", {}).items():
            args.Display.DrawLines(lines, order_color(k), LINE_THICKNESS)
