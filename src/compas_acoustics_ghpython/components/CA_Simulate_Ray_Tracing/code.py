import random
import warnings

import Grasshopper

from compas_acoustics.simulation_methods import Pyroomacoustics
from compas_acoustics_ghpython import MethodGraph

# Inputs that only apply with ray tracing
RAY_TRACING_ONLY = ("n_rays", "energy_thres", "duration", "seed")


class SimulateRayTracing(Grasshopper.Kernel.GH_ScriptInstance):

    def RunScript(self, scene, ism_order, ray_tracing, n_rays, absorb_air, energy_thres, time_thres, seed, normalize):

        if scene is None:
            ghenv.Component.AddRuntimeMessage(
                Grasshopper.Kernel.GH_RuntimeMessageLevel.Warning, "Input parameter 'scene' failed to collect data."
            )
            return

        if ray_tracing is None:
            ray_tracing = True
        if absorb_air is None:
            absorb_air = True

        # Unset inputs are left out so that the method's own defaults apply
        # (image source order 3, 20000 rays, an Eyring-sized buffer).
        given = {
            "image_source_order": ism_order,
            "n_rays": n_rays,
            "air_absorption": absorb_air,
            "energy_thres": energy_thres,
            "duration": time_thres,
            "seed": seed,
            "normalize": normalize,
        }
        given = {key: value for key, value in given.items() if value is not None}
        if not ray_tracing:
            ignored = [key for key in RAY_TRACING_ONLY if key in given]
            if ignored:
                ghenv.Component.AddRuntimeMessage(
                    Grasshopper.Kernel.GH_RuntimeMessageLevel.Remark,
                    "Ray tracing is off, so {} are not used.".format(", ".join(ignored)),
                )
            given = {key: value for key, value in given.items() if key not in RAY_TRACING_ONLY}
        elif "seed" not in given:
            # The method's own default is a fixed seed; an empty input draws a new one each run
            given["seed"] = random.randrange(2**31)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                scene.simulate(Pyroomacoustics(ray_tracing=ray_tracing, **given))
            except ValueError as e:
                ghenv.Component.AddRuntimeMessage(Grasshopper.Kernel.GH_RuntimeMessageLevel.Error, str(e))
                return
        # Points outside the room, open rooms, mergeable faces, a slow image source search
        for warning in caught:
            if issubclass(warning.category, UserWarning):
                ghenv.Component.AddRuntimeMessage(Grasshopper.Kernel.GH_RuntimeMessageLevel.Warning, str(warning.message))

        ghenv.Component.Message = "{:g}Hz, {}, order={}, rays={}, air={}, seed={}".format(
            scene.sampling_rate,
            "ISM + ray tracing" if ray_tracing else "ISM",
            given.get("image_source_order", "default"),
            given.get("n_rays", "default") if ray_tracing else "-",
            given.get("air_absorption", "default"),
            given["seed"] if ray_tracing else "-",
        )

        return MethodGraph(scene.ir_graph, "pyroomacoustics")
