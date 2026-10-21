import warnings

import Grasshopper
from Grasshopper import DataTree
from Grasshopper.Kernel.Data import GH_Path
import Rhino
import System

from compas_acoustics import SpecularReflections


class SpecularReflectionsComponent(Grasshopper.Kernel.GH_ScriptInstance):

    def RunScript(self, scene, order, source_idx, receiver_idx):

        empty = (None,) * 6
        if scene is None:
            ghenv.Component.AddRuntimeMessage(
                Grasshopper.Kernel.GH_RuntimeMessageLevel.Warning, "Input parameter 'scene' failed to collect data."
            )
            return empty

        orders = sorted({int(o) for o in order if o is not None}) if order else []
        negative = [o for o in orders if o < 0]
        if negative:
            ghenv.Component.AddRuntimeMessage(
                Grasshopper.Kernel.GH_RuntimeMessageLevel.Warning,
                "Orders below 0 are left out: {}.".format(", ".join(str(o) for o in negative)),
            )
            orders = [o for o in orders if o >= 0]
        if not orders:
            orders = [1]

        for name, index, count in (("source", source_idx, len(scene.sources)), ("receiver", receiver_idx, len(scene.receivers))):
            if index is not None and not 0 <= index < count:
                ghenv.Component.AddRuntimeMessage(
                    Grasshopper.Kernel.GH_RuntimeMessageLevel.Warning,
                    "No {} at index {}; the scene has {} {}(s), 0 to {}.".format(name, index, count, name, count - 1),
                )

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                found = SpecularReflections.from_scene(scene, max_order=max(orders))
            except ValueError as e:
                ghenv.Component.AddRuntimeMessage(Grasshopper.Kernel.GH_RuntimeMessageLevel.Error, str(e))
                return empty
        # Sources and receivers outside the room
        for warning in caught:
            if issubclass(warning.category, UserWarning):
                ghenv.Component.AddRuntimeMessage(Grasshopper.Kernel.GH_RuntimeMessageLevel.Warning, str(warning.message))

        reflections = found.filter(source_index=source_idx, receiver_index=receiver_idx, order=orders)

        paths = []
        images = []
        walls = DataTree[object]()
        for i, path in enumerate(reflections):
            polyline = Rhino.Geometry.Polyline()
            polyline.AddRange([Rhino.Geometry.Point3d(*point) for point in path.points])
            paths.append(polyline)
            images.append(Rhino.Geometry.Point3d(*path.image))
            walls.AddRange(path.wall_names, GH_Path(i))

        counts = ["order {}: {}".format(o, sum(1 for path in reflections if path.order == o)) for o in reflections.orders]
        ghenv.Component.Message = ", ".join(counts) if counts else "no paths"

        return (
            reflections,
            paths,
            images,
            [path.order for path in reflections],
            walls,
            [path.travel_time for path in reflections],
        )
