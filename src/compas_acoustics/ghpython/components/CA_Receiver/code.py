# flake8: noqa
import Grasshopper
import Rhino
from compas_acoustics import Receiver

# TODO: Implement direction
class CreateReceiver(Grasshopper.Kernel.GH_ScriptInstance):

    def RunScript(self, position, name):
        if position is None:
            position = Rhino.Geometry.Point3d(0, 0, 1)
        if name is None:
            name = "my_receiver"

        return Receiver(name, *position)
