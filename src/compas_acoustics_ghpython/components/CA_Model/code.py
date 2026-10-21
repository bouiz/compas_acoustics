import Grasshopper

import System
from compas_acoustics import AcousticObject
from compas_acoustics.scene import Scene
from compas_acoustics.source import Source
from compas_acoustics.receiver import Receiver


class CreateModelScene(Grasshopper.Kernel.GH_ScriptInstance):

    def RunScript(self, objects, sources, receivers, name, sampling_rate):

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

        # An upstream component that failed passes None; leave such items out instead of failing the scene
        given = list(objects)
        objects = [o for o in given if isinstance(o, AcousticObject)]
        if not objects:
            ghenv.Component.AddRuntimeMessage(
                Grasshopper.Kernel.GH_RuntimeMessageLevel.Warning,
                "No acoustic objects: the {} object(s) given are empty or invalid.".format(len(given)),
            )
            return
        if len(objects) < len(given):
            ghenv.Component.AddRuntimeMessage(
                Grasshopper.Kernel.GH_RuntimeMessageLevel.Warning,
                "{} of the {} objects are not acoustic objects (empty or invalid) and are left out.".format(
                    len(given) - len(objects), len(given)
                ),
            )

        try:
            scene = Scene(name=name, sampling_rate=sampling_rate)
        except ValueError as e:
            ghenv.Component.AddRuntimeMessage(Grasshopper.Kernel.GH_RuntimeMessageLevel.Error, str(e))
            return
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