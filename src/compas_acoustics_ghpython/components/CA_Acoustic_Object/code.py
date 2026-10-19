import Grasshopper
import Rhino
from compas_acoustics.geometry import AcousticObject
from compas_acoustics_ghpython import ensure_inputs
from compas_rhino.conversions import mesh_to_compas
from compas_rhino.conversions import polyline_to_compas

# TODO: Change the typeHintID in metadata.json from "mesh" to "geometrybase"

# First item is the default
ROLES = ["boundary", "obstacle"]


class CreateAcousticObject(Grasshopper.Kernel.GH_ScriptInstance):

    def RunScript(self, geometry, material, name, role):
        ensure_inputs(ghenv.Component, {"role": ROLES})

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
        elif isinstance(geometry, Rhino.Geometry.Polyline):
            geometry = polyline_to_compas(geometry)
        elif isinstance(geometry, Rhino.Geometry.PolylineCurve):
            geometry = polyline_to_compas(geometry.ToPolyline())
        else:
            ghenv.Component.AddRuntimeMessage(
                Grasshopper.Kernel.GH_RuntimeMessageLevel.Error, "Unsupported geometry type."
            )
            return
        role = "boundary" if role in (None, "") else role
        try:
            return AcousticObject(acoustic_geometry=geometry, material=material, name=name, role=role)
        except (TypeError, ValueError) as e:
            ghenv.Component.AddRuntimeMessage(Grasshopper.Kernel.GH_RuntimeMessageLevel.Error, str(e))
            return
