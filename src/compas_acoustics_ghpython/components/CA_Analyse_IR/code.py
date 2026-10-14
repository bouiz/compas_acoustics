# flake8: noqa
import Grasshopper
from compas_acoustics.analyser import AcousticAnalyser
from compas_acoustics_ghpython import AcousticParameter
from compas_acoustics import Defaults


class AnalyseIR(Grasshopper.Kernel.GH_ScriptInstance):
    def RunScript(self, ir):
        if ir is None:
            ghenv.Component.AddRuntimeMessage(Grasshopper.Kernel.GH_RuntimeMessageLevel.Warning, "Input parameter 'ir' failed to collect data.")
            return None

        analyser = AcousticAnalyser(ir.ir)
        analyser.analyse()

        return [AcousticParameter().from_analyser(analyser, name) for name in Defaults().PARAM_NAMES]