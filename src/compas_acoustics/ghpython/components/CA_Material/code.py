# flake8: noqa
import Grasshopper
from compas_acoustics.acoustic_material import AcousticMaterial


class CreateMaterial(Grasshopper.Kernel.GH_ScriptInstance):

    def RunScript(self, absorption, scattering, name):
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

        absorption = [float(x) for x in absorption.split(",")]
        scattering = [float(x) for x in scattering.split(",")] if scattering is not None else None

        return AcousticMaterial(
            name=name,
            absorption_coefficients=absorption,
            scattering_coefficients=scattering,
        )