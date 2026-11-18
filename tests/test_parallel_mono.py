"""Mono over several processes, and with acoustic reciprocity."""

import warnings

import numpy as np
import pyroomacoustics as pra
import pytest

import compas_acoustics as ca
from compas_acoustics.room.parallel import split_processes
from compas_acoustics.room.pra import source_to_pra
from compas_acoustics.room.raytracing import directional_histograms
from compas_acoustics.room.raytracing import image_source_rirs
from compas_acoustics.room.raytracing import ray_traced_run
from compas_acoustics.simulation_methods import SimulationContext

RAYS = dict(image_source_order=1, n_rays=4000, duration=0.8)
GRID = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], dtype=float)


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


def test_a_microphone_no_image_source_reaches_does_not_fail_over_processes():
    # outside the room no image source is visible, not even the direct path; the other microphone keeps its response
    context = SimulationContext.from_scene(_scene(2))
    sources = [source_to_pra(s) for s in context.sources]
    mics = np.array([[6.0, 4.0, 1.2], [9.0, 3.0, 1.5]], dtype=float).T
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ism = image_source_rirs(context.geometry, sources, mics, 48000, max_order=2, n_jobs=2)
    assert all(np.abs(r).max() > 0 for r in ism[0])
    assert all(r.size > 0 and not np.any(r) for r in ism[1])


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


def test_a_run_with_split_rays_keeps_its_room_and_sorts_its_rays_by_direction():
    """The hybrid's traced reference: one image source search, each source's rays split over processes, the
    directional copies of the receivers filled from the same slices."""
    context = SimulationContext.from_scene(_scene(1))
    sources = [source_to_pra(s) for s in context.sources]
    mics = np.array([r.position for r in context.receivers], dtype=float).T
    kwargs = dict(max_order=1, duration=0.8, n_rays=4000, seed=3, directions=[GRID])
    serial = ray_traced_run(context.geometry, sources, mics, 48000, **kwargs)
    parallel = ray_traced_run(context.geometry, sources, mics, 48000, n_jobs=2, **kwargs)
    assert isinstance(parallel.room, pra.Room) and parallel.room.simulator_state["ism_done"]
    assert parallel.responses[0][0].shape == serial.responses[0][0].shape
    assert abs(_level_db(serial.responses[0][0], parallel.responses[0][0])) < 1.0
    energy, serial_energy = parallel.histograms[0][0].energy, serial.histograms[0][0].energy
    assert energy.shape == serial_energy.shape
    assert energy.sum() == pytest.approx(serial_energy.sum(), rel=0.1)


def test_directional_histograms_over_processes_hold_the_same_energy():
    context = SimulationContext.from_scene(_scene(1))
    kwargs = dict(max_order=1, duration=0.8, n_rays=4000, seed=3)
    args = (context.geometry, [source_to_pra(s) for s in context.sources], np.array([r.position for r in context.receivers]).T, 48000, [GRID])
    serial = directional_histograms(*args, **kwargs)[0][0].energy
    parallel = directional_histograms(*args, n_jobs=2, **kwargs)[0][0].energy
    assert parallel.shape == serial.shape
    assert parallel.sum() == pytest.approx(serial.sum(), rel=0.1)
    np.testing.assert_allclose(parallel.sum(axis=(1, 2)), serial.sum(axis=(1, 2)), rtol=0.3)


def test_the_hybrid_methods_record_n_jobs():
    pytest.importorskip("pyFDN")
    method = ca.Ambisonics(sh_order=1, image_source_order=1, tail="eyring", n_jobs=2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ir = next(iter(_scene(1).simulate(method).values()))
    assert ir.generation["n_jobs"] == 2


def _split_scene(sources, receivers):
    scene = _scene(0, 0)
    for name, position in sources.items():
        scene.add_source(ca.Source(name, *position))
    for name, position in receivers.items():
        scene.add_receiver(ca.Receiver(name, *position))
    return scene


SPLIT_SOURCES = {"s0": (1.5, 2.5, 1.5), "s1": (4.5, 1.5, 1.2)}
SPLIT_RECEIVERS = {"r0": (3.0, 3.5, 1.5), "r1": (2.0, 1.0, 1.4)}


def _by_names(responses):
    return {(ir.source.name, ir.receiver.name): ir.denormalized() for ir in responses.values()}


@pytest.mark.parametrize("method", ["mono", "ambisonics"])
def test_a_pair_is_the_same_whichever_pairs_share_the_run(method):
    """With a seed, a pair's rays and tail restart for it: the run of every pair at once equals runs of some of them.

    So a caller may split the pairs over processes, by source or by receiver, and get the joint run's responses.
    """

    def run(sources, receivers):
        scene = _split_scene({k: SPLIT_SOURCES[k] for k in sources}, {k: SPLIT_RECEIVERS[k] for k in receivers})
        if method == "mono":
            simulation = ca.Mono(seed=5, per_pair_seeds=True, **RAYS)
        else:
            pytest.importorskip("pyFDN")
            reference = ca.Mono(image_source_order=1, n_rays=2000, duration=0.5, seed=5, normalize=False, per_pair_seeds=True)
            simulation = ca.Ambisonics(reference=reference, image_source_order=1, seed=5, normalize=False, per_pair_seeds=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return _by_names(scene.simulate(simulation))

    joint = run(SPLIT_SOURCES, SPLIT_RECEIVERS)
    parts = {}
    for name in SPLIT_SOURCES:
        parts.update(run([name], SPLIT_RECEIVERS))
    by_receiver = {}
    for name in SPLIT_RECEIVERS:
        by_receiver.update(run(SPLIT_SOURCES, [name]))
    assert set(parts) == set(by_receiver) == set(joint)
    for key, response in joint.items():
        for part in (parts, by_receiver):
            assert part[key].shape == response.shape
            np.testing.assert_allclose(part[key], response, rtol=1e-6, atol=1e-9)


@pytest.mark.parametrize("n_jobs", [1, 2])
def test_a_source_bins_are_the_same_whichever_sources_share_the_trace(n_jobs):
    from compas_acoustics.room.pra import source_to_pra

    context = SimulationContext.from_scene(_split_scene(SPLIT_SOURCES, SPLIT_RECEIVERS))
    sources = [source_to_pra(s) for s in context.sources]
    grid = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], dtype=float)
    mics = np.array([r.position for r in context.receivers]).T
    kwargs = dict(max_order=1, duration=0.5, n_rays=2000, seed=3, n_jobs=n_jobs, per_pair_seeds=True)
    joint = directional_histograms(context.geometry, sources, mics, 48000, [grid, grid], **kwargs)
    for s, source in enumerate(sources):
        alone = directional_histograms(context.geometry, [source], mics, 48000, [grid, grid], **kwargs)
        for m in range(mics.shape[1]):
            np.testing.assert_array_equal(alone[m][0].energy, joint[m][s].energy)


def test_n_jobs_must_be_positive():
    with pytest.raises(ValueError):
        ca.Mono(n_jobs=0)


def test_the_hybrid_n_jobs_must_be_positive():
    pytest.importorskip("pyFDN")
    with pytest.raises(ValueError):
        ca.Ambisonics(n_jobs=0)
