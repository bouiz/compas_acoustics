"""The paths a run records (compas_acoustics.path_recorder) equal those a display search finds."""

import warnings

import numpy as np
import pytest

import compas_acoustics as ca
from compas_acoustics.path_recorder import active
from compas_acoustics.room.pra import source_to_pra
from compas_acoustics.room.raytracing import emitter_rirs
from compas_acoustics.simulation_methods import SimulationContext

WALL = ca.AcousticMaterial("wall", 0.15, 0.2)


def _scene(n_sources=1, n_receivers=1):
    scene = ca.Scene("recorded")
    scene.add_acoustic_objects(ca.AcousticObjectsLibrary.shoebox_room(7, 5, 3, materials=WALL))
    for k in range(n_sources):
        scene.add_source(ca.Source(f"s{k}", 1.2 + 1.5 * k, 1.2 + 0.5 * k, 1.6))
    for k in range(n_receivers):
        scene.add_receiver(ca.Receiver(f"r{k}", 6.0 - k, 4.0, 1.2))
    return scene


def _simulate(scene, max_order=1, **method):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with ca.record_paths(max_order) as recorded:
            scene.simulate(ca.Mono(**method))
    return recorded


def _match(recorded, expected, points, gain, atol=1e-4):
    """Pair every recorded path with one expected path of the same points (within ``atol``: a reversed path's hits
    are float32 crossings from the other side) and compare their gains."""
    left = list(expected)
    assert len(recorded) == len(left)
    for p in recorded:
        a = np.asarray(points(p), dtype=float)
        k = next((k for k, q in enumerate(left) if np.asarray(points(q)).shape == a.shape and np.allclose(points(q), a, atol=atol)), None)
        assert k is not None, f"no expected path through {a.tolist()}"
        np.testing.assert_allclose(gain(p), gain(left.pop(k)), rtol=1e-4)


def _assert_same_paths(recorded, expected):
    def points(p):
        return [[p.source_index, p.receiver_index, p.order]] + [list(q) for q in p.points]

    _match(recorded.paths, expected.paths, points, lambda p: p.band_gain)


def test_a_run_records_the_paths_a_display_search_finds():
    scene = _scene()
    recorded = _simulate(scene, image_source_order=1, ray_tracing=False).reflections(scene)
    expected = ca.SpecularReflections.from_scene(scene, max_order=1)
    assert len(recorded.paths) == 7  # the direct sound and one reflection per wall of the shoebox
    _assert_same_paths(recorded, expected)


def test_only_the_orders_asked_for_are_kept():
    scene = _scene()
    recorded = _simulate(scene, max_order=1, image_source_order=3, ray_tracing=False).reflections(scene)
    assert recorded.orders == [0, 1]
    _assert_same_paths(recorded, ca.SpecularReflections.from_scene(scene, max_order=1))


def test_a_ray_traced_run_records_its_image_sources_too():
    scene = _scene()
    recorded = _simulate(scene, image_source_order=1, n_rays=500, duration=0.3).reflections(scene)
    _assert_same_paths(recorded, ca.SpecularReflections.from_scene(scene, max_order=1))


def test_a_reciprocal_run_gives_its_paths_source_first():
    scene = _scene(n_sources=3)
    recorder = _simulate(scene, image_source_order=1, ray_tracing=False, reciprocity=True)
    recorded = recorder.reflections(scene)
    _assert_same_paths(recorded, ca.SpecularReflections.from_scene(scene, max_order=1))
    for p in recorded.paths:  # the image, mirrored back, lies the path's length from the receiver
        np.testing.assert_allclose(np.linalg.norm(np.subtract(list(p.image), list(p.points[-1]))), p.length, rtol=1e-6)


def test_the_processes_of_a_parallel_run_hand_their_paths_back():
    scene = _scene(n_sources=2)
    context = SimulationContext.from_scene(scene)
    sources = [source_to_pra(s, 48000) for s in context.sources]
    mics = np.array([r.position for r in context.receivers], dtype=float).T
    with ca.record_paths(1) as recorded:
        emitter_rirs(context.geometry, sources, mics, 48000, max_order=1, n_jobs=2)
    _assert_same_paths(recorded.reflections(scene), ca.SpecularReflections.from_scene(scene, max_order=1))


def test_recording_is_off_by_default_and_not_nestable():
    assert active() is None
    with ca.record_paths(0) as recorded:
        assert active() is recorded
        with pytest.raises(RuntimeError, match="already active"):
            with ca.record_paths(1):
                pass
    assert active() is None
    with pytest.raises(ValueError, match="max_order"):
        ca.PathRecorder(-1)


def test_an_unsearched_pair_has_no_path():
    scene = _scene()
    with ca.record_paths(1) as recorded:
        pass
    assert recorded.reflections(scene).paths == []
    assert recorded.diffracted(scene) == []


def test_a_diffracting_run_records_its_diffracted_paths():
    pytest.importorskip("spharpy")  # compas_acoustics.hybrid.diffraction needs it
    from compas.datastructures import Mesh

    from compas_acoustics.rays import diffracted_paths

    scene = ca.Scene("shadow")
    scene.add_acoustic_objects(ca.AcousticObjectsLibrary.shoebox_room(8, 6, 3, materials=WALL))
    panel = Mesh.from_vertices_and_faces([[4, 2, 0.0], [4, 4, 0.0], [4, 4, 3.0], [4, 2, 3.0]], [[0, 1, 2, 3]])
    scene.add_acoustic_object(ca.AcousticObject(panel, ca.AcousticMaterial("panel", 0.2, 0.1), name="panel", role="obstacle"))
    scene.add_source(ca.Source("s", 2, 3, 1.5))
    scene.add_receiver(ca.Receiver("r", 6, 3, 1.5))
    recorder = _simulate(scene, image_source_order=1, ray_tracing=False, diffraction=True)
    recorded = recorder.diffracted(scene)
    expected = diffracted_paths(scene, max_order=1)
    assert recorded

    _match(recorded, expected, lambda p: [[p.edge, p.order, p.distance]] + p.points, lambda p: p.band_gain)
    assert {p.zone for p in recorded} == {p.zone for p in expected}
