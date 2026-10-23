import warnings

import numpy as np
import pytest
from compas.data import json_dumps
from compas.data import json_loads

from compas_acoustics import AcousticMaterial
from compas_acoustics import AcousticObjectsLibrary
from compas_acoustics import EnergyMap
from compas_acoustics import Receiver
from compas_acoustics import Scene
from compas_acoustics import Source
from compas_acoustics.defaults import Defaults

A_WEIGHTS = [-16.1, -8.6, -3.2, 0.0, 1.2, 1.0, -1.1]


def shoebox(dims, alpha, scattering, sources, spl=80.0):
    scene = Scene("map")
    scene.add_acoustic_objects(AcousticObjectsLibrary.shoebox_room(*dims, materials=AcousticMaterial("m", alpha, scattering)))
    for i, position in enumerate(sources):
        source = Source(f"s{i}", *position)
        source.spl = spl
        scene.add_source(source)
    return scene


def test_direct_sound_alone_gives_the_free_field_level():
    scene = shoebox((20.0, 20.0, 10.0), 0.99, 0.0, [(10.0, 10.0, 5.0)])
    m = EnergyMap.from_scene(scene, [[12.0, 10.0, 5.0]], n_rays=2000, air_absorption=False)
    assert m.spl()[0] == pytest.approx([80 - 6.02] * 7, abs=0.3)


def test_the_late_level_matches_the_pyroomacoustics_response_at_the_same_point():
    from compas_acoustics import Mono

    source, point = (1.5, 1.5, 1.6), (4.0, 3.0, 1.2)
    scene = shoebox((6.0, 5.0, 3.0), 0.1, 0.1, [source])
    m = EnergyMap.from_scene(scene, [point], n_rays=20000, air_absorption=False, seed=0)
    scene.add_receiver(Receiver("r", *point))
    (ir,) = scene.simulate(Mono(ray_tracing=True, image_source_order=1, air_absorption=False, seed=0)).values()
    start = ir.direct_sound_index() + int(0.08 * ir.sampling_rate)
    # a flat material: the response's broadband energy is each band's; package scale to SPL as Source.spl defines it
    expected = 80.0 + 20 * np.log10(4 * np.pi) + 10 * np.log10(np.sum(np.asarray(ir.denormalized(), float)[start:] ** 2))
    distance = np.linalg.norm(np.subtract(point, source))
    first = int(np.ceil((distance / Defaults().DEFAULT_SPEED_OF_SOUND + 0.08) / m.bin_size))
    assert 10 * np.log10(np.sum(m.energy[0, 3, first:])) == pytest.approx(expected, abs=1.0)


def test_points_outside_the_room_are_nan_and_warn():
    scene = shoebox((6.0, 5.0, 3.0), 0.3, 0.1, [(1.5, 1.5, 1.6)])
    with pytest.warns(UserWarning, match="1 of 3 points"):
        m = EnergyMap.from_scene(scene, [[3, 2, 1], [4, 3, 1.2], [100, 0, 0]], n_rays=1000)
    assert list(m.inside) == [True, True, False]
    assert np.all(np.isnan(m.spl()[2])) and np.isnan(m.spl_a()[2])
    assert np.all(np.isfinite(m.spl()[:2]))


def test_two_sources_add_incoherently():
    one = EnergyMap.from_scene(shoebox((6.0, 5.0, 3.0), 0.3, 0.1, [(1.5, 1.5, 1.6)]), [[4, 3, 1.2]], n_rays=4000, seed=1)
    two = EnergyMap.from_scene(shoebox((6.0, 5.0, 3.0), 0.3, 0.1, [(1.5, 1.5, 1.6)] * 2), [[4, 3, 1.2]], n_rays=4000, seed=1)
    assert two.spl() == pytest.approx(one.spl() + 10 * np.log10(2), abs=0.05)


def test_one_source_of_several():
    scene = shoebox((6.0, 5.0, 3.0), 0.3, 0.1, [(1.5, 1.5, 1.6), (5.0, 4.0, 1.6)])
    both = EnergyMap.from_scene(scene, [[3, 2.5, 1.2]], n_rays=4000, seed=1)
    first = EnergyMap.from_scene(scene, [[3, 2.5, 1.2]], source_index=0, n_rays=4000, seed=1)
    assert np.all(first.spl() < both.spl())


def test_spl_a_is_the_a_weighted_total_of_the_bands():
    m = EnergyMap.from_scene(shoebox((6.0, 5.0, 3.0), 0.3, 0.1, [(1.5, 1.5, 1.6)]), [[4, 3, 1.2]], n_rays=2000)
    expected = 10 * np.log10(np.sum(10 ** ((m.spl()[0] + A_WEIGHTS) / 10)))
    assert m.spl_a()[0] == pytest.approx(expected, abs=1e-6)


def test_frames_sum_to_the_steady_state_level():
    m = EnergyMap.from_scene(shoebox((6.0, 5.0, 3.0), 0.3, 0.1, [(1.5, 1.5, 1.6)]), [[4, 3, 1.2]], n_rays=2000)
    times, levels = m.frames()
    assert levels.shape == (1, times.size) and times[1] - times[0] == pytest.approx(3 * m.bin_size)
    assert m.frames(0.010)[0][1] == pytest.approx(3 * m.bin_size)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        total = 10 * np.log10(np.nansum(10 ** (levels / 10), axis=1))
    assert total == pytest.approx(10 * np.log10(np.sum(10 ** (m.spl() / 10), axis=1)), abs=0.01)


def test_the_map_round_trips_through_json():
    with pytest.warns(UserWarning, match="1 of 2 points"):
        m = EnergyMap.from_scene(shoebox((6.0, 5.0, 3.0), 0.3, 0.1, [(1.5, 1.5, 1.6)]), [[4, 3, 1.2], [100, 0, 0]], n_rays=1000)
    back = json_loads(json_dumps(m))
    assert isinstance(back, EnergyMap)
    assert np.allclose(back.spl(), m.spl(), equal_nan=True) and list(back.inside) == list(m.inside)
