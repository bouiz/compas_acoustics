import numpy as np

from compas_acoustics import AcousticMaterial
from compas_acoustics import AcousticObjectsLibrary
from compas_acoustics import Receiver
from compas_acoustics import Scene
from compas_acoustics.hybrid.directional import directional_absorption_filters
from compas_acoustics.hybrid.directional import scanned_absorption
from compas_acoustics.hybrid.late import fdn_directions
from compas_acoustics.hybrid.pair import Pair
from compas_acoustics.hybrid.scattering import _first_hit
from compas_acoustics.room.geometry import RoomGeometry

FS = 48000.0


def spec(direction=(1, 0, 0)):
    scene = Scene("test")
    wall, floor = AcousticMaterial("wall", 0.1, 0.1), AcousticMaterial("carpet", 0.6, 0.1)
    scene.add_acoustic_objects(AcousticObjectsLibrary.shoebox_room(7, 5, 3, materials={
        "floor": floor, "ceiling": wall, "left_wall": wall, "right_wall": wall, "front_wall": wall, "back_wall": wall,
    }))
    receiver = Receiver("r", 3.5, 2.5, 1.5, direction=list(direction))
    return Pair(RoomGeometry.from_scene(scene), [1.5, 1.5, 1.6], receiver.position, receiver.rotation, FS)


def test_a_downward_ray_hits_the_floor():
    faces = spec().geometry.faces_with_materials()
    hit = _first_hit(faces, np.array([3.5, 2.5, 1.5]), np.array([[0.0, 0.0, -1.0]]))
    assert faces[hit[0]][1] == "carpet"


def test_the_line_facing_the_floor_is_the_most_absorbing():
    directions = fdn_directions()
    alpha = scanned_absorption(spec(), directions)
    assert alpha.shape == (directions.cshape[0], 8)
    assert np.all((alpha >= 0.1 - 1e-12) & (alpha <= 0.6 + 1e-12))
    down = int(np.argmin(directions.cartesian[:, 2]))
    assert np.all(alpha[down] >= alpha.max(axis=0) - 1e-12)


def test_turning_the_receiver_about_z_keeps_the_floor_below():
    directions = fdn_directions()
    down = int(np.argmin(directions.cartesian[:, 2]))
    np.testing.assert_allclose(
        scanned_absorption(spec((0, 1, 0)), directions)[down], scanned_absorption(spec(), directions)[down]
    )


def test_one_filter_bank_per_line():
    directions = fdn_directions()
    sos = directional_absorption_filters(spec(), directions)
    assert sos.shape[1:] == (6, directions.cshape[0])
