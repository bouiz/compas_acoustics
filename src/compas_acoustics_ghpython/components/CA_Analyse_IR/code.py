import math
import warnings

import Grasshopper

from compas_acoustics import AcousticAnalyser
from compas_acoustics import worker
from compas_acoustics_ghpython import AcousticParameter
from compas_acoustics_ghpython import ensure_inputs

# First item is the default
TIME_SCALES = ["db", "linear"]


class AnalyseIR(Grasshopper.Kernel.GH_ScriptInstance):
    def message(self, level, text):
        ghenv.Component.AddRuntimeMessage(getattr(Grasshopper.Kernel.GH_RuntimeMessageLevel, level), text)

    def RunScript(self, ir, time_scale):
        ensure_inputs(ghenv.Component, {"time_scale": TIME_SCALES})

        if ir is None:
            self.message("Warning", "Input parameter 'ir' failed to collect data.")
            return None, None, None
        time_scale = time_scale or TIME_SCALES[0]
        if time_scale not in TIME_SCALES:
            self.message("Error", "Input parameter 'time_scale' must be one of {}.".format(TIME_SCALES))
            return None, None, None

        # STI and G need pyrato >= 1.0: on Rhino 8's Python 3.9 they come from the worker, when there is one
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                analyser, messages = worker.analyse(ir)
            except Exception as e:
                self.message("Warning", "STI and G could not be computed in the worker: {}".format(e))
                analyser, messages = AcousticAnalyser(ir), []
                analyser.analyse()
        messages = [str(w.message) for w in caught if issubclass(w.category, UserWarning)] + messages
        for text in dict.fromkeys(messages):
            if not text.startswith("STI is not available"):
                self.message("Warning", text)

        data = analyser.analysis_data
        names = [name for name in AcousticParameter.NAMES
                 if name not in worker.NEEDS_PYRATO_1 or not all(math.isnan(value) for value in data[name])]
        if len(names) < len(AcousticParameter.NAMES):
            self.message("Remark", "STI and G need pyrato >= 1.0, which Rhino's Python 3.9 cannot install: "
                                   "run the installer with worker 'hybrid' to compute them in the worker.")
        parameters = [AcousticParameter().from_analyser(analyser, name) for name in names]
        time = AcousticParameter.from_time_response(ir, db=time_scale == "db")
        frequency = AcousticParameter.from_frequency_response(ir)
        return parameters, time, frequency
