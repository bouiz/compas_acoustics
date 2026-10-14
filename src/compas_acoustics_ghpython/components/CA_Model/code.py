# flake8: noqa
import Grasshopper

import System
from compas_acoustics.scene import Scene
from compas_acoustics.source import Source
from compas_acoustics.receiver import Receiver


class CreateModelScene(Grasshopper.Kernel.GH_ScriptInstance):

    def RunScript(self, objects, sources, receivers, name):

        if objects is None:
            ghenv.Component.AddRuntimeMessage(
                Grasshopper.Kernel.GH_RuntimeMessageLevel.Warning, "No objects provided."
            )
            return
        if sources is None:
            ghenv.Component.AddRuntimeMessage(
                Grasshopper.Kernel.GH_RuntimeMessageLevel.Warning, "No sources provided."
            )
            return
        if receivers is None:
            ghenv.Component.AddRuntimeMessage(
                Grasshopper.Kernel.GH_RuntimeMessageLevel.Warning, "No receivers provided."
            )
            return

        scene = Scene(name=name)
        scene.add_acoustic_objects(objects)
        for src in sources:
            if isinstance(src, Source):
                scene.add_source(src)
            else:
                ghenv.Component.AddRuntimeMessage(
                    Grasshopper.Kernel.GH_RuntimeMessageLevel.Error,
                    f"Invalid source type: {type(src)}. Expected a Source object.",
                )
        for rcv in receivers:
            if isinstance(rcv, Receiver):
                scene.add_receiver(rcv)
            else:
                ghenv.Component.AddRuntimeMessage(
                    Grasshopper.Kernel.GH_RuntimeMessageLevel.Error,
                    f"Invalid receiver type: {type(rcv)}. Expected a Receiver object.",
                )

        return scene