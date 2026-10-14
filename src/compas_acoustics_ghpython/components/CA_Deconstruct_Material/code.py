# flake8: noqa
import Grasshopper


class DeconstructMaterial(Grasshopper.Kernel.GH_ScriptInstance):

    def RunScript(self, material):
        if material is None:
            ghenv.Component.AddRuntimeMessage(
                Grasshopper.Kernel.GH_RuntimeMessageLevel.Warning,
                "Input parameter 'material' failed to collect data."
            )
            return
        return material.absorption_coefficients, material.scattering_coefficients, material.name, material.category.value, material.data_origin