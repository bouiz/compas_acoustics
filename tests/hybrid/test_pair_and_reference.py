import numpy as np
import pyroomacoustics as pra
import pytest

from compas_acoustics import AcousticMaterial
from compas_acoustics import AcousticObjectsLibrary
from compas_acoustics import Scene
from compas_acoustics.hybrid.bands import BANDS
from compas_acoustics.hybrid.pair import Pair
from compas_acoustics.room.geometry import SYNTHESIS_BANDS
from compas_acoustics.room.geometry import RoomGeometry
from compas_acoustics.room.raytracing import _on_package_scale


def shoebox(leaves=6):
    scene = Scene("test")
    room = AcousticObjectsLibrary.shoebox_room(7, 5, 3, materials=AcousticMaterial("wall", 0.2, 0.1))
    scene.add_acoustic_objects(room[:leaves])
    return RoomGeometry.from_scene(scene)


def test_the_geometry_grid_is_pyfdns():
    np.testing.assert_allclose(BANDS, SYNTHESIS_BANDS)


def test_a_pair_takes_positions_as_arrays():
    pair = Pair(shoebox(), [1.5, 1.5, 1.6], [5, 3, 1.5], np.eye(3), 48000.0)
    assert pair.source.shape == (3,)
    assert pair.receiver.dtype == float


def test_a_pair_needs_a_closed_room():
    with pytest.warns(UserWarning):
        open_room = shoebox(leaves=5)
    with pytest.raises(ValueError, match="closed"):
        Pair(open_room, [1.5, 1.5, 1.6], [5, 3, 1.5], np.eye(3), 48000.0)


def test_a_pair_needs_a_proper_rotation():
    with pytest.raises(ValueError, match="rotation"):
        Pair(shoebox(), [1.5, 1.5, 1.6], [5, 3, 1.5], np.diag([1.0, -1.0, 1.0]), 48000.0)


def test_the_package_scale_removes_the_latency_and_the_1_over_r_convention():
    latency = pra.constants.get("frac_delay_length") // 2
    rir = np.zeros(1000)
    rir[latency + 10] = 4.0 * np.pi
    scaled = _on_package_scale(rir, latency, 1200)
    assert scaled.shape == (1200,)
    assert scaled[10] == pytest.approx(1.0)
    assert np.count_nonzero(scaled) == 1
