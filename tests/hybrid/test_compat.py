import pytest
import numpy as np
from _compat import Material, MeshGeometry, RoomSpec


def test_a_compat_mesh_is_a_closed_room():
    v = np.array([[0, 0, 0], [4, 0, 0], [4, 3, 0], [0, 3, 0], [0, 0, 2.5], [4, 0, 2.5], [4, 3, 2.5], [0, 3, 2.5]], float)
    f = ((0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7))
    g = MeshGeometry(v, f, ("wall",) * 6, {"wall": Material(np.full(8, 0.2))})
    spec = RoomSpec(g, [1, 1, 1], [3, 2, 1.2])
    assert g.is_closed and spec.fs == 48000.0


def test_a_compat_shoebox_is_the_same_room():
    from _compat import ShoeboxGeometry

    m = Material.from_scalar(0.1, 0.2)
    g = ShoeboxGeometry((8.0, 6.0, 3.0), {n: m for n in ShoeboxGeometry.WALL_NAMES})
    assert g.is_closed and g.volume == pytest.approx(144.0) and g.surface_area == pytest.approx(2 * (48 + 24 + 18))


def test_from_room_is_eyring_on_the_mesh():
    import pyrato
    from _compat import ShoeboxGeometry

    from compas_acoustics.hybrid.target import RoomTarget

    m = Material.from_scalar(0.1)
    spec = RoomSpec(ShoeboxGeometry((8.0, 6.0, 3.0), {n: m for n in ShoeboxGeometry.WALL_NAMES}), [2, 2, 1.5], [5, 4, 1.5],
                    air_absorption=False)
    expected = pyrato.parametric.reverberation_time_eyring(volume=144.0, surface_area=180.0, mean_absorption=np.full(8, 0.1),
                                                           speed_of_sound=343.0)
    np.testing.assert_allclose(RoomTarget.from_room(spec).rt[:, 0], expected)
