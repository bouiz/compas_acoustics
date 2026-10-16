# flake8: noqa
import Grasshopper

from compas_acoustics.simulator import Simulator

class SimulateRayTracing(Grasshopper.Kernel.GH_ScriptInstance):

    def RunScript(self, scene, sample_rate, max_order, n_rays, absorb_air, energy_thres, time_thres, seed, normalize):

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
        if energy_thres is None:
            energy_thres = 1e-7
        if time_thres is None:
            time_thres = 10.0
        if normalize is None:
            normalize = True

        sim = Simulator(scene)
        if sample_rate is not None:
            sim.sampling_rate = sample_rate

        rirs = sim.rir_from_ray_tracing(
            image_source_order=max_order,
            n_rays=n_rays,
            air_absorption=absorb_air,
            energy_thres=energy_thres,
            time_thres=time_thres,
            random_seed=seed,
            normalize=normalize,
        )

        ghenv.Component.Message = "{}Hz, order={}, rays={}, air={}, seed={}, energy_thres={}, time_thres={}, normalized={}".format(
            # Report the sample rate of the generated impulse responses
            next(iter(rirs.values())).sample_rate if rirs else sim.sampling_rate,
            max_order, n_rays, absorb_air, seed if seed is not None else "random", energy_thres, time_thres, normalize
        )

        return sim.scene.ir_graph