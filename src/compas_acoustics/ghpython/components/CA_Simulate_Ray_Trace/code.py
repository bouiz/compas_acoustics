# flake8: noqa
import Grasshopper

from compas_acoustics.simulator import Simulator

class SimulateRayTracing(Grasshopper.Kernel.GH_ScriptInstance):

    def RunScript(self, scene, sample_rate, max_order, n_rays, absorb_air, seed):

        if scene is None:
            ghenv.Component.AddRuntimeMessage(
                Grasshopper.Kernel.GH_RuntimeMessageLevel.Warning, "Input parameter 'scene' failed to collect data."
            )
            return

        if max_order is None:
            max_order = 2
        if n_rays is None:
            n_rays = 1000
        if absorb_air is None:
            absorb_air = True

        sim = Simulator(scene)
        if sample_rate is not None:
            sim.sampling_rate = sample_rate

        sim.rir_from_ray_tracing(
            image_source_order=max_order,
            n_rays=n_rays,
            air_absorption=absorb_air,
            random_seed=seed,
        )

        ghenv.Component.Message = "{}Hz, order={}, rays={}, air={}, seed={}".format(
            sim.sampling_rate, max_order, n_rays, absorb_air, seed if seed is not None else "random"
        )

        return sim.scene.ir_graph