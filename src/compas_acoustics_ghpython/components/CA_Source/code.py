import Grasshopper
import Rhino
from compas_acoustics import Source


class CreateSource(Grasshopper.Kernel.GH_ScriptInstance):
    ITEMS = ["bidirectional", "hypercardioid", "supercardioid", "cardioid", "omnidirectional"]

    def RunScript(self, position, name, direction, directivity, sound_file):
        if position is None:
            position = Rhino.Geometry.Point3d(0, 0, 0)
        if direction is None:
            direction = Rhino.Geometry.Vector3d(1, 0, 0)

        if directivity is None:
            dirt = "omnidirectional"
        elif directivity not in self.ITEMS:
            ghenv.Component.AddRuntimeMessage(
                Grasshopper.Kernel.GH_RuntimeMessageLevel.Error, "Input parameter 'directivity' is not valid."
            )
            return None
        else:
            dirt = directivity

        if name is None:
            name = "my_source"

        return Source(
            name,
            *position,
            directivity=dirt,
            direction=[*direction],
            sound_file=sound_file
        )
