# flake8: noqa
import Grasshopper
from compas_acoustics import worker
from compas_acoustics_ghpython import AcousticParameter
from compas_acoustics import Defaults


class AnalyseIR(Grasshopper.Kernel.GH_ScriptInstance):
    def RunScript(self, ir):
        if ir is None:
            ghenv.Component.AddRuntimeMessage(Grasshopper.Kernel.GH_RuntimeMessageLevel.Warning, "Input parameter 'ir' failed to collect data.")
            return None

        # Pass the ImpulseResponse so its sample rate is used
        # STI and G come from the worker where Rhino's Python cannot compute them
        analyser = worker.analyse(ir)

        return [AcousticParameter().from_analyser(analyser, name) for name in Defaults().PARAM_NAMES]