import numpy as np
import pyroomacoustics as pra

from compas_acoustics import AcousticMaterial
from compas_acoustics import AcousticObjectsLibrary
from compas_acoustics import Scene
from compas_acoustics.progress import reporting
from compas_acoustics.room import raytracing
from compas_acoustics.room.geometry import RoomGeometry
from compas_acoustics.room.raytracing import directional_histograms
from compas_acoustics.room.raytracing import ray_traced_run

SOURCES = ([1.5, 1.5, 1.6], [4.0, 2.0, 1.2])
MICS = np.array([[5.0, 3.0, 1.5], [2.0, 4.0, 1.0]]).T
DIRECTIONS = [np.eye(3), -np.eye(3)]


def geometry():
    scene = Scene("counts")
    scene.add_acoustic_objects(AcousticObjectsLibrary.shoebox_room(7, 5, 3, materials=AcousticMaterial("wall", 0.2, 0.3)))
    return RoomGeometry.from_scene(scene)


def traced(n_rays=4000):
    sources = [pra.SoundSource(np.array(s)) for s in SOURCES]
    reports = []
    with reporting(lambda stage, done, of: reports.append((stage, done, of))):
        run = ray_traced_run(geometry(), sources, MICS, 48000, max_order=2, n_rays=n_rays, duration=0.3, seed=3, directions=DIRECTIONS)
    return run, reports


def test_rays_are_counted_as_traced(monkeypatch):
    _, reports = traced()
    rays = [(done, of) for stage, done, of in reports if stage == "ray tracing"]
    assert rays[0] == (0, 8000)
    assert rays[-1] == (8000, 8000)
    assert [done for done, _ in rays] == sorted(done for done, _ in rays)
    assert len(rays) == 2 * (raytracing.RAY_REPORT_PARTS + 1)


def test_counting_the_rays_leaves_the_responses_bit_identical(monkeypatch):
    counted, _ = traced()
    histograms = directional_histograms(geometry(), [pra.SoundSource(np.array(s)) for s in SOURCES], MICS, 48000, DIRECTIONS, max_order=2, n_rays=4000, duration=0.3, seed=3)
    monkeypatch.setattr(raytracing, "RAY_REPORT_PARTS", 1)
    single, reports = traced()
    single_histograms = directional_histograms(
        geometry(), [pra.SoundSource(np.array(s)) for s in SOURCES], MICS, 48000, DIRECTIONS, max_order=2, n_rays=4000, duration=0.3, seed=3
    )
    assert len([r for r in reports if r[0] == "ray tracing"]) == 2 * 2  # one call per source: before and after it
    for a, b in zip(counted.responses, single.responses):
        for x, y in zip(a, b):
            assert np.array_equal(x, y)
    for a, b in zip(counted.histograms + histograms, single.histograms + single_histograms):
        for x, y in zip(a, b):
            assert np.array_equal(x.energy, y.energy)


def test_an_odd_number_of_rays_goes_in_one_call():
    assert raytracing._report_parts(15000) == 8
    assert raytracing._report_parts(20000) == 16
    assert raytracing._report_parts(2001) == 1


def test_the_image_source_search_counts_the_reflections_it_finds():
    run, reports = traced()
    found = [done for stage, done, of in reports if stage == "image sources"]
    reflections = sum(int(np.count_nonzero(source.orders)) for source in run.room.sources)
    assert found == sorted(found)
    assert len(found) == len(SOURCES)
    assert found[-1] == reflections > 0
    assert all(of is None for stage, _, of in reports if stage == "image sources")
