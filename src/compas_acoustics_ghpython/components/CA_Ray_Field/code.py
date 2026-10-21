import random
import warnings

import Grasshopper
import Rhino

from compas_acoustics import RayField


def _polyline(ray):
    polyline = Rhino.Geometry.Polyline()
    polyline.AddRange([Rhino.Geometry.Point3d(*point) for point in ray.points])
    return polyline


class RayFieldComponent(Grasshopper.Kernel.GH_ScriptInstance):

    def RunScript(self, scene, n_rays, max_order, direction, angle, scattering, seed, order):

        empty = (None,) * 3
        if scene is None:
            ghenv.Component.AddRuntimeMessage(
                Grasshopper.Kernel.GH_RuntimeMessageLevel.Warning, "Input parameter 'scene' failed to collect data."
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

        return field, [_polyline(ray) for ray in shown], [_polyline(ray) for ray in shown.scattered_rays]
