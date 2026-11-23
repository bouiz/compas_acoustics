import numpy as np
import pyfar as pf
import pytest

from compas_acoustics import AcousticMaterial
from compas_acoustics import AcousticObjectsLibrary
from compas_acoustics import Receiver
from compas_acoustics import Scene
from compas_acoustics.hybrid.late import directional_delays
from compas_acoustics.hybrid.late import fdn_directions
from compas_acoustics.hybrid.pair import Pair
from compas_acoustics.room.geometry import RoomGeometry

FS = 48000.0
DIMS = np.array([7.0, 5.0, 3.0])


def pair(direction=(1, 0, 0)):
    scene = Scene("test")
    scene.add_acoustic_objects(AcousticObjectsLibrary.shoebox_room(*DIMS, materials=AcousticMaterial("wall", 0.2, 0.1)))
    receiver = Receiver("r", 5, 3, 1.5, direction=list(direction))
    return Pair(RoomGeometry.from_scene(scene), [1.5, 1.5, 1.6], receiver.position, receiver.rotation, FS, 343.0)


def jitter(n, rng=0):
    """The factor ``directional_delays`` draws for ``n`` lines: up to 0.1 either way."""
    return 1.0 + 0.1 * np.random.default_rng(rng).uniform(-1.0, 1.0, size=n)


def test_each_line_is_the_shoebox_free_path_along_it():
    directions = fdn_directions()
    unit = directions.cartesian / np.linalg.norm(directions.cartesian, axis=1, keepdims=True)
    expected = 1.0 / (np.abs(unit) / DIMS).sum(axis=1) / 343.0 * FS * jitter(unit.shape[0])
    delays = directional_delays(pair(), directions)
    np.testing.assert_allclose(delays, expected, rtol=0.031, atol=1.0)


def test_a_turned_receiver_reads_the_room_along_its_own_axes():
    ahead = pf.Coordinates(1.0, 0.0, 0.0)
    assert directional_delays(pair((1, 0, 0)), ahead)[0] == pytest.approx(7.0 / 343.0 * FS * jitter(1)[0], rel=0.031)
    assert directional_delays(pair((0, 1, 0)), ahead)[0] == pytest.approx(5.0 / 343.0 * FS * jitter(1)[0], rel=0.031)


def test_the_mean_free_path_is_cauchys():
    directions = fdn_directions()
    delays = directional_delays(pair(), directions) / jitter(directions.cshape[0])
    mean_free_path = 4.0 * DIMS.prod() / (2.0 * (DIMS[0] * DIMS[1] + DIMS[0] * DIMS[2] + DIMS[1] * DIMS[2]))
    assert np.mean(delays) == pytest.approx(mean_free_path / 343.0 * FS, rel=0.1)
