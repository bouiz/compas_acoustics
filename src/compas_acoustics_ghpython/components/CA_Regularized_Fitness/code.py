import Grasshopper
import System
from compas_acoustics import regularized_fitness


class RegularizedFitness(Grasshopper.Kernel.GH_ScriptInstance):
    def RunScript(
            self,
            values,
            target,
            k
        ):
        if values is None:
            ghenv.Component.AddRuntimeMessage(
                Grasshopper.Kernel.GH_RuntimeMessageLevel.Warning,
                "Input parameter 'values' failed to collect data."
            )
            return
        if target is None:
            ghenv.Component.AddRuntimeMessage(
                Grasshopper.Kernel.GH_RuntimeMessageLevel.Warning,
                "Input parameter 'target' failed to collect data."
            )
            return
        if k is None:
            k = 1.0
        return regularized_fitness(values, target, k)