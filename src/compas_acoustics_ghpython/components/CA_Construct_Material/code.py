import Grasshopper
import System
from compas_acoustics.acoustic_material import AcousticMaterial


class ConstructMaterial(Grasshopper.Kernel.GH_ScriptInstance):

    def RunScript(self, absorption, scattering, name, category):
        if absorption is None:
            ghenv.Component.AddRuntimeMessage(
                Grasshopper.Kernel.GH_RuntimeMessageLevel.Warning,
                "Input parameter 'absorption' failed to collect data."
            )
            return
        if name is None:
            ghenv.Component.AddRuntimeMessage(
                Grasshopper.Kernel.GH_RuntimeMessageLevel.Warning,
                "Input parameter 'name' failed to collect data."
            )
            return

        return AcousticMaterial(
            name=name,
            absorption_coefficients=absorption,
            scattering_coefficients=scattering,
            category=category or None,
        )