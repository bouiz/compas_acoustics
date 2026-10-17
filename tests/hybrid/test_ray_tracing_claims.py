import os

import numpy as np
import pyFDN
import pyroomacoustics as pra
import pytest

from compas_acoustics import AcousticMaterial
from compas_acoustics import AcousticObjectsLibrary
from compas_acoustics import Scene
from compas_acoustics.raytracing import MIN_RAYS
from compas_acoustics.raytracing import ray_traced_rirs
from compas_acoustics.room_geometry import RoomGeometry

pytestmark = [pytest.mark.slow, pytest.mark.skipif(not os.environ.get("COMPAS_ACOUSTICS_SLOW"), reason="set COMPAS_ACOUSTICS_SLOW=1")]

FS = 24000
WALL = [0.10, 0.08, 0.06, 0.05, 0.05, 0.05, 0.06]
FLOOR = [0.05, 0.10, 0.25, 0.45, 0.60, 0.65, 0.70]
CEILING = [0.25, 0.45, 0.70, 0.80, 0.85, 0.85, 0.80]


def uneven():
    s = [0.15] * 7
    wall = AcousticMaterial("wall", WALL, s)
    materials = {"floor": AcousticMaterial("floor", FLOOR, s), "ceiling": AcousticMaterial("ceiling", CEILING, s),
                 "left_wall": wall, "right_wall": wall, "front_wall": wall, "back_wall": wall}
    scene = Scene("uneven")
    scene.add_acoustic_objects(AcousticObjectsLibrary.shoebox_room(6, 4, 3, materials=materials))
    return RoomGeometry.from_scene(scene)


def rt_bands(g, max_order, n_rays, seed):
    ((rir,),) = ray_traced_rirs(
        g, [pra.SoundSource(np.array([1.5, 1.0, 1.2]))], np.array([[4.0], [3.0], [1.7]]), FS,
        max_order=max_order, duration=2.0, n_rays=n_rays, air_absorption=False, seed=seed,
    )
    return pyFDN.estimate_rt_bands(rir, FS)[0]


def test_more_rays_reduce_the_run_to_run_scatter():
    """Upstream: mid-band CV of T30 16 % at 2000 rays vs 3 % at MIN_RAYS (ten runs)."""
    g = uneven()

    def spread(n_rays, n=6):
        # Order 3, not upstream's 8: on a mesh room pyroomacoustics' image source search is exponential in the
        # order (upstream used a ShoeboxRoom). The claim is about the ray count, not the specular order.
        rt = np.stack([rt_bands(g, 3, n_rays, k) for k in range(n)])
        return rt.mean(axis=0), rt.std(axis=0) / rt.mean(axis=0)

    mean_few, cv_few = spread(2000)
    mean_many, cv_many = spread(MIN_RAYS)
    mid = slice(2, 7)
    assert np.mean(cv_many[mid]) < 0.8 * np.mean(cv_few[mid])
    np.testing.assert_allclose(mean_many[mid], mean_few[mid], rtol=0.25)


@pytest.mark.skip(reason="orders 6 and 30 on a mesh room: pyroomacoustics' general image source search is exponential "
                         "in the order and does not finish; upstream measures this claim on a ShoeboxRoom")
def test_image_source_order_lengthens_the_decay_of_an_unevenly_treated_room():
    """Upstream: specular energy substitutes for scattered energy and the decay
    lengthens without converging (0.94 s at order 6, 1.28 s at 30 in an office)."""
    g = uneven()

    def mid_band_rt(max_order):
        return float(np.mean([rt_bands(g, max_order, MIN_RAYS, k) for k in range(3)], axis=0)[2:7].mean())

    assert mid_band_rt(30) > 1.15 * mid_band_rt(6)
