# flake8: noqa
import Grasshopper
from compas_acoustics.analyser import AcousticAnalyser
from compas_acoustics import AcousticParameter
from compas_acoustics import Defaults


class DeconstructParam(Grasshopper.Kernel.GH_ScriptInstance):
    def RunScript(self, param):
        if param is None:
            self.AddRuntimeMessage(Grasshopper.Kernel.GH_RuntimeMessageLevel.Warning, "Input parameter 'param' failed to collect data.")
            return None
        
        return param.title, param.values, param.overall, param.frequencies