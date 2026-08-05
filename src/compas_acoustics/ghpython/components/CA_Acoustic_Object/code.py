# flake8: noqa
import Grasshopper
import Rhino
from compas_acoustics.geometry import AcousticObject
from compas_rhino.conversions import mesh_to_compas

# TODO: Change the typeHintID in metadata.json from "mesh" to "geometrybase"

class CreateAcousticObject(Grasshopper.Kernel.GH_ScriptInstance):

    def RunScript(self, geometry, material, name):

        if geometry is None:
            ghenv.Component.AddRuntimeMessage(
                Grasshopper.Kernel.GH_RuntimeMessageLevel.Warning, "No geometry provided."
            )
            return
        if material is None:
            ghenv.Component.AddRuntimeMessage(
                Grasshopper.Kernel.GH_RuntimeMessageLevel.Warning, "No material provided."
            )
            return

        # Gemetry conversion from Rhino to Compas
        # TODO: Figure out how to handle other geometry types (e.g., Brep, Surface, etc.)
        if isinstance(geometry, Rhino.Geometry.Mesh):
            geometry = mesh_to_compas(geometry)

        return AcousticObject(acoustic_geometry=geometry, material=material, name=name)