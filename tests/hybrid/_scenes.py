"""Scenes shared by the hybrid tests."""

from compas.datastructures import Mesh

from compas_acoustics import AcousticMaterial
from compas_acoustics import AcousticObject
from compas_acoustics import AcousticObjectsLibrary
from compas_acoustics import Receiver
from compas_acoustics import Scene
from compas_acoustics import Source

WALL = AcousticMaterial("wall", 0.1, 0.25)


def reference_scene(panel=False, direction=(0, 1, 0)):
    scene = Scene("reference", sampling_rate=32000)
    scene.add_acoustic_objects(AcousticObjectsLibrary.shoebox_room(6, 5, 3, materials=WALL))
    if panel:
        # 2 m by 2 m, across the direct path from the source to the receiver
        mesh = Mesh.from_vertices_and_faces([[3, 1.5, 0.5], [3, 3.5, 0.5], [3, 3.5, 2.5], [3, 1.5, 2.5]], [[0, 1, 2, 3]])
        scene.add_acoustic_object(AcousticObject(mesh, WALL, name="panel", role="obstacle"))
    scene.add_source(Source("src", 1.5, 1.5, 1.5))
    scene.add_receiver(Receiver("rec", 4.5, 3.5, 1.2, direction=list(direction)))
    return scene
