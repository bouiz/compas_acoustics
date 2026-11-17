"""One ray trace for the mono reference and the directional histogram, same seed and same parameters.

The separate calls (:func:`ray_traced_rirs`, :func:`directional_histograms`) are the before; :func:`traced_reference`
and :class:`Ambisonics` given that ``Mono`` are the after. With one seed and one set of ray parameters the
responses and the bins match.
"""

import numpy as np
import pyroomacoustics as pra

import compas_acoustics as ca
from compas_acoustics.hybrid.late import fdn_directions
from compas_acoustics.room.geometry import RoomGeometry
from compas_acoustics.room.raytracing import directional_histograms
from compas_acoustics.room.raytracing import traced_reference
from compas_acoustics.simulation_methods import Ambisonics
from compas_acoustics.simulation_methods import Mono

FS = 48000
SOURCE = np.array([6.0, 3.0, 1.5])
RECEIVER = np.array([2.0, 3.0, 1.5])
SEED = 7
RAYS = dict(max_order=1, duration=0.25, n_rays=400, receiver_radius=0.5, energy_threshold=1e-7, air_absorption=False, seed=SEED)


def geometry():
    scene = ca.Scene("unified", sampling_rate=FS)
    scene.add_acoustic_objects(ca.AcousticObjectsLibrary.shoebox_room(8, 6, 3, materials=ca.AcousticMaterial("wall", 0.2, 0.05)))
    return RoomGeometry.from_scene(scene)


def directions():
    unit = fdn_directions().cartesian
    return [unit / np.linalg.norm(unit, axis=1, keepdims=True)]


def test_one_trace_matches_the_separate_histogram_and_is_independent_of_the_other_sources():
    """The bins match a directional trace with the same seed. A source traced alone matches that source in a joint trace."""
    g = geometry()
    mics = RECEIVER[:, np.newaxis]
    alone_src = [pra.SoundSource(SOURCE)]
    other = np.array([1.0, 1.0, 1.2])
    both_src = [pra.SoundSource(SOURCE), pra.SoundSource(other)]
    dirs = directions()
    separate_hist = directional_histograms(g, alone_src, mics, FS, dirs, **RAYS)[0][0]
    alone_hist, alone_rir = traced_reference(g, alone_src, mics, FS, dirs, **RAYS)
    both_hist, both_rir = traced_reference(g, both_src, mics, FS, dirs, **RAYS)
    np.testing.assert_array_equal(alone_hist[0][0].energy, separate_hist.energy)
    np.testing.assert_array_equal(both_hist[0][0].energy, alone_hist[0][0].energy)
    np.testing.assert_allclose(both_rir[0][0], alone_rir[0][0], rtol=1e-5, atol=1e-8)


def test_a_different_seed_changes_the_trace():
    g = geometry()
    mics = RECEIVER[:, np.newaxis]
    sources = [pra.SoundSource(SOURCE)]
    dirs = directions()
    same, _ = traced_reference(g, sources, mics, FS, dirs, **RAYS)
    other, _ = traced_reference(g, sources, mics, FS, dirs, **dict(RAYS, seed=SEED + 1))
    assert not np.allclose(same[0][0].energy, other[0][0].energy)


def test_ambisonics_follows_the_reference_mono_not_the_defaults():
    """The late field uses the Mono's ray count and seed, traced once for the reference and the histogram."""
    scene = ca.Scene("unified", sampling_rate=FS)
    scene.add_acoustic_objects(ca.AcousticObjectsLibrary.shoebox_room(8, 6, 3, materials=ca.AcousticMaterial("wall", 0.2, 0.05)))
    scene.add_source(ca.Source("src", *SOURCE))
    scene.add_receiver(ca.Receiver("rec", *RECEIVER))
    mono = Mono(
        ray_tracing=True,
        image_source_order=1,
        n_rays=RAYS["n_rays"],
        duration=RAYS["duration"],
        receiver_radius=RAYS["receiver_radius"],
        energy_thres=RAYS["energy_threshold"],
        air_absorption=False,
        seed=SEED,
        normalize=False,
    )
    # The network's own seed is different on purpose: the rays must follow the Mono, not this.
    hybrid = Ambisonics(reference=mono, image_source_order=1, duration=RAYS["duration"], seed=SEED + 3, directional_absorption=False)
    (ir,) = scene.simulate(hybrid).values()
    (reference,) = hybrid.reference_rirs.values()
    g = RoomGeometry.from_scene(scene)
    traced_hist, traced_rir = traced_reference(
        g,
        [pra.SoundSource(SOURCE)],
        RECEIVER[:, np.newaxis],
        FS,
        directions(),
        max_order=1,
        duration=RAYS["duration"],
        n_rays=RAYS["n_rays"],
        receiver_radius=RAYS["receiver_radius"],
        energy_threshold=RAYS["energy_threshold"],
        air_absorption=False,
        seed=SEED,
    )
    np.testing.assert_allclose(reference.denormalized(), traced_rir[0][0], rtol=1e-5, atol=1e-8)
    assert ir.generation["n_rays"] == RAYS["n_rays"]
    assert ir.generation["ray_seed"] == SEED
    assert ir.generation["seed"] == SEED + 3

    used = next(iter(hybrid.results.values())).core.directional_envelope
    np.testing.assert_array_equal(used.energy, traced_hist[0][0].energy)
    # and not the old hardcoded 20_000-ray default: a fresh histogram at that count differs
    assert used.energy.shape[0] == traced_hist[0][0].energy.shape[0]
    assert int(ir.generation["n_rays"]) != 20_000
