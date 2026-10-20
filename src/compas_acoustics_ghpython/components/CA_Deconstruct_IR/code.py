import Grasshopper
import System
from Grasshopper import DataTree
from Grasshopper.Kernel.Data import GH_Path

from compas_acoustics_ghpython import AcousticParameter
from compas_acoustics_ghpython import ensure_inputs

# First item is the default
SELECT = ["all", *AcousticParameter.NAMES]


class DeconstructParam(Grasshopper.Kernel.GH_ScriptInstance):
    def RunScript(self, param, select):
        ensure_inputs(ghenv.Component, {"select": SELECT})

        if not param:
            ghenv.Component.AddRuntimeMessage(Grasshopper.Kernel.GH_RuntimeMessageLevel.Warning, "Input parameter 'param' failed to collect data.")
            return None, None, None, None, None

        params = list(param)
        if select not in (None, "", "all"):
            params = [p for p in params if p.name == select]
            if not params:
                ghenv.Component.AddRuntimeMessage(
                    Grasshopper.Kernel.GH_RuntimeMessageLevel.Warning,
                    "The parameters connected do not include '{}'.".format(select),
                )
                return None, None, None, None, None

        if len(params) == 1:
            p = params[0]
            return p, p.title, p.values, p.overall, p.frequencies

        # Every parameter in its own branch, so that they stay apart; they share their frequency bands
        values = DataTree[object]()
        for i, p in enumerate(params):
            values.AddRange(p.values, GH_Path(i))
        return params, [p.title for p in params], values, [p.overall for p in params], params[0].frequencies
