import Grasshopper
import Rhino
from compas.geometry import Frame
from compas_acoustics import Receiver

class CreateReceiver(Grasshopper.Kernel.GH_ScriptInstance):

    def RunScript(self, position, name, direction, plane):
        if name is None:
            name = "my_receiver"

        if plane is not None:
            # A plane sets position and orientation together: x axis facing, z axis up
            if position is not None or direction is not None:
                ghenv.Component.AddRuntimeMessage(
                    Grasshopper.Kernel.GH_RuntimeMessageLevel.Remark,
                    "'plane' sets the position and orientation; 'position' and 'direction' are not used.",
                )
            origin, xaxis, yaxis = plane.Origin, plane.XAxis, plane.YAxis
            receiver = Receiver.from_frame(name, Frame([origin.X, origin.Y, origin.Z], [xaxis.X, xaxis.Y, xaxis.Z], [yaxis.X, yaxis.Y, yaxis.Z]))
        else:
            if position is None:
                position = Rhino.Geometry.Point3d(0, 0, 1)
            # upright: the receiver's up stays +z
            receiver = Receiver(name, *position)
            try:
                if direction is not None:
                    receiver.direction = [*direction]
            except ValueError as e:
                ghenv.Component.AddRuntimeMessage(Grasshopper.Kernel.GH_RuntimeMessageLevel.Error, str(e))
                return None

        try:
            receiver.frame
        except ValueError as e:
            ghenv.Component.AddRuntimeMessage(Grasshopper.Kernel.GH_RuntimeMessageLevel.Error, str(e))
            return None

        return receiver
