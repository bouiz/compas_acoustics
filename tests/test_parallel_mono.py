"""Mono over several processes, and with acoustic reciprocity."""

import warnings

import numpy as np
import pytest

import compas_acoustics as ca
from compas_acoustics.progress import Cancelled
from compas_acoustics.progress import reporting
from compas_acoustics.room.parallel import split_processes
from compas_acoustics.room.raytracing import directional_histograms
from compas_acoustics.simulation_methods import SimulationContext

RAYS = dict(image_source_order=1, n_rays=4000, duration=0.8)


def _scene(n_sources=3, n_receivers=1, directivity=None):
    scene = ca.Scene("parallel")
    scene.add_acoustic_objects(ca.AcousticObjectsLibrary.shoebox_room(7, 5, 3, materials=ca.AcousticMaterial("wall", 0.15, 0.2)))
    for k in range(n_sources):
        scene.add_source(ca.Source(f"s{k}", 1.2 + 1.5 * k, 1.2 + 0.5 * k, 1.6, directivity=directivity))
    for k in range(n_receivers):
        scene.add_receiver(ca.Receiver(f"r{k}", 6.0 - k, 4.0, 1.2))
    return scene


def _run(scene, **method):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rirs = scene.simulate(ca.Mono(**method))
    return {key: ir.denormalized() for key, ir in rirs.items()}, next(iter(rirs.values())).generation


def _level_db(a, b):
    return 10.0 * np.log10(np.sum(b**2) / np.sum(a**2))


@pytest.mark.parametrize("n_jobs, n_emitters, expected", [(1, 5, (1, 1)), (8, 1, (1, 8)), (8, 2, (2, 4)), (8, 3, (3, 2)), (8, 12, (8, 1)), (4, 4, (4, 1))])
def test_split_processes(n_jobs, n_emitters, expected):
    assert split_processes(n_jobs, n_emitters) == expected


def test_image_sources_over_processes_are_the_same():
    scene = _scene()
    serial, generation = _run(scene, image_source_order=3, ray_tracing=False)
    parallel, parallel_generation = _run(scene, image_source_order=3, ray_tracing=False, n_jobs=2)
    assert "n_jobs" not in generation and parallel_generation["n_jobs"] == 2
    for key in serial:
        np.testing.assert_array_equal(parallel[key], serial[key])


def test_reciprocity_swaps_only_when_cheaper_and_exact():
    scene = _scene()
    forward, generation = _run(scene, image_source_order=3, ray_tracing=False)
    swapped, swapped_generation = _run(scene, image_source_order=3, ray_tracing=False, reciprocity=True)
    assert "reciprocal" not in generation and swapped_generation["reciprocal"] is True
    assert set(swapped) == set(forward)
    for key in forward:
        np.testing.assert_allclose(swapped[key], forward[key], atol=1e-3 * np.abs(forward[key]).max())
    # a directional source, or no more sources than receivers, keeps the roles
    assert _run(_scene(directivity="cardioid"), image_source_order=1, ray_tracing=False, reciprocity=True)[1]["reciprocal"] is False
    assert _run(_scene(2, 2), image_source_order=1, ray_tracing=False, reciprocity=True)[1]["reciprocal"] is False


def test_reciprocal_ray_tracing_matches_in_level():
    scene = _scene()
    forward, _ = _run(scene, **RAYS)
    swapped, _ = _run(scene, reciprocity=True, **RAYS)
    for key in forward:
        assert abs(_level_db(forward[key], swapped[key])) < 1.0


@pytest.mark.parametrize("n_sources, n_jobs", [(3, 3), (1, 2)])  # emitters at once; one emitter's rays split
def test_ray_tracing_over_processes_matches_in_level_and_repeats(n_sources, n_jobs):
    scene = _scene(n_sources)
    serial, _ = _run(scene, **RAYS)
    parallel, _ = _run(scene, n_jobs=n_jobs, **RAYS)
    again, _ = _run(scene, n_jobs=n_jobs, **RAYS)
    for key in serial:
        assert parallel[key].shape == serial[key].shape
        assert abs(_level_db(serial[key], parallel[key])) < 1.0
        np.testing.assert_array_equal(again[key], parallel[key])


def test_a_cancelled_parallel_run_stops_and_the_next_one_runs():
    def stop(stage, done, of):
        if stage == "ray tracing" and done:
            raise Cancelled()

    with pytest.raises(Cancelled):
        with reporting(stop):
            _run(_scene(1), n_jobs=2, **RAYS)
    assert _run(_scene(1), n_jobs=2, **RAYS)[0]


def test_directional_histograms_over_processes_hold_the_same_energy():
    scene = _scene(1)
    context = SimulationContext.from_scene(scene)
    from compas_acoustics.room.pra import source_to_pra

    grid = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], dtype=float)
    kwargs = dict(max_order=1, duration=0.8, n_rays=4000, seed=3)
    args = (context.geometry, [source_to_pra(s) for s in context.sources], np.array([r.position for r in context.receivers]).T, 48000, [grid])
    serial = directional_histograms(*args, **kwargs)[0][0].energy
    parallel = directional_histograms(*args, n_jobs=2, **kwargs)[0][0].energy
    assert parallel.shape == serial.shape
    assert parallel.sum() == pytest.approx(serial.sum(), rel=0.1)
    np.testing.assert_allclose(parallel.sum(axis=(1, 2)), serial.sum(axis=(1, 2)), rtol=0.3)


def test_n_jobs_must_be_positive():
    with pytest.raises(ValueError):
        ca.Mono(n_jobs=0)
