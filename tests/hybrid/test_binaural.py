"""Binaural rendering of the hybrid: an HRTF per early path, the late field MagLS-decoded."""

import numpy as np
import pytest
from compas.datastructures import Mesh

from compas_acoustics import AcousticMaterial
from compas_acoustics import AcousticObject
from compas_acoustics import AcousticObjectsLibrary
from compas_acoustics import Receiver
from compas_acoustics import Scene
from compas_acoustics import Source
from compas_acoustics.methods import Hybrid

FS = 48000
WALL = AcousticMaterial("wall", [0.1, 0.12, 0.15, 0.2, 0.25, 0.3, 0.35], 0.1)
FAST = dict(tail="eyring", image_source_order=2, duration=0.3)


def scene(receiver=None, panel=False):
    s = Scene("binaural")
    s.add_acoustic_objects(AcousticObjectsLibrary.shoebox_room(10, 8, 4, materials=WALL))
    if panel:
        mesh = Mesh.from_vertices_and_faces([[5, 3, 0.5], [5, 5, 0.5], [5, 5, 2.5], [5, 3, 2.5]], [[0, 1, 2, 3]])
        s.add_acoustic_object(AcousticObject(mesh, WALL, name="panel", role="obstacle"))
    s.add_source(Source("src", 2.0, 4.0, 1.5))
    s.add_receiver(receiver or Receiver("rec", 8.0, 4.2, 1.5))
    return s


def run(method, **kwargs):
    s = scene(**kwargs)
    s.simulate(method)
    (result,) = method.results.values()
    return result


def test_the_result_keeps_the_early_images():
    result = run(Hybrid(**FAST))
    assert result.images is not None
    assert result.images.order.max() == result.max_order
    assert result.diffraction_paths == () and result.btms_contributions == ()


def test_the_result_keeps_the_diffracted_paths():
    result = run(Hybrid(diffraction=True, **FAST), panel=True)
    assert len(result.diffraction_paths) > 0


def test_magls_filters_are_the_auralisers():
    from compas_acoustics.auraliser import Auraliser
    from compas_acoustics.binaural import DEFAULT_HRTF, magls_filters

    a = Auraliser("room1", {"ir1": {"ir": np.ones((32, 4)), "sample_rate": FS, "ambisonic_order": 1,
                                     "channel_order": "acn", "normalisation": "sn3d"}},
                  {"ac1": {"filepath": "a.wav"}}, "binaural")
    np.testing.assert_array_equal(a.binaural_filters(1), magls_filters(DEFAULT_HRTF, FS, 1))


def test_the_nearest_measured_direction_is_picked():
    from compas_acoustics.binaural import DEFAULT_HRTF
    from compas_acoustics.binaural import HrtfSet

    hrtf = HrtfSet.from_sofa(DEFAULT_HRTF, FS)
    assert hrtf.hrirs.shape[1] == 2 and hrtf.directions.shape == (hrtf.hrirs.shape[0], 3)
    picked = hrtf.nearest(hrtf.directions[[0, 100, 500]] * 3.0)
    np.testing.assert_array_equal(picked, [0, 100, 500])


def test_a_set_at_another_rate_is_resampled():
    from compas_acoustics.binaural import DEFAULT_HRTF
    from compas_acoustics.binaural import HrtfSet

    native, resampled = HrtfSet.from_sofa(DEFAULT_HRTF, FS), HrtfSet.from_sofa(DEFAULT_HRTF, 44100)
    assert resampled.sample_rate == 44100
    assert resampled.hrirs.shape[2] == int(np.ceil(native.hrirs.shape[2] * 44100 / FS))  # pyfar rounds up


def test_magls_agrees_with_the_hrtf_it_was_fitted_to():
    """A plane wave through its HRIR and through order-3 MagLS: same ear levels, same arrival.

    Measured on this set: per-ear level differences of 0.46 dB on average (8.8 dB with the
    ears swapped), the decode 9 samples early at the median.
    """
    import pyfar as pf
    import spharpy
    from compas_acoustics.binaural import DEFAULT_HRTF, magls_filters
    from compas_acoustics.binaural import HrtfSet

    hrtf = HrtfSet.from_sofa(DEFAULT_HRTF, FS)
    filters = magls_filters(DEFAULT_HRTF, FS, 3)
    differences, lags = [], []
    for i in np.random.default_rng(0).choice(hrtf.directions.shape[0], 40, replace=False):
        x, y, z = hrtf.directions[i]
        y_sh = np.sqrt(4 * np.pi) * spharpy.spherical.spherical_harmonic_basis_real(
            3, pf.Coordinates(x, y, z), normalization="SN3D", channel_convention="ACN", condon_shortley=False)[0]
        decoded = np.einsum("c,nce->ne", y_sh, filters)
        direct = hrtf.hrirs[i].T
        differences.append(np.abs(10 * np.log10(np.sum(decoded ** 2, 0) / np.sum(direct ** 2, 0))))
        for ear in range(2):
            xc = np.correlate(np.pad(decoded[:, ear], (64, 64)), direct[:, ear], "valid")
            lags.append(int(np.argmax(np.abs(xc))) - 64)
    assert np.mean(differences) < 1.5
    assert abs(np.median(lags)) <= 12


def identity(n_dir=1):
    """An HRTF set that passes every direction through unchanged to both ears."""
    from compas_acoustics.binaural import HrtfSet

    hrirs = np.zeros((n_dir, 2, 8))
    hrirs[:, :, 0] = 1.0
    return HrtfSet(hrirs, np.tile([[1.0, 0.0, 0.0]], (n_dir, 1)), FS)


@pytest.mark.parametrize("diffraction", [False, True])
def test_with_a_transparent_hrtf_both_ears_hear_the_omnidirectional_early_part(diffraction):
    """Same paths, same band interpolation, same delays and taper: the W channel exactly."""
    from compas_acoustics.binaural import render_early_binaural

    result = run(Hybrid(diffraction=diffraction, **FAST), panel=diffraction)
    ears = render_early_binaural(result, identity())
    w = result.early[:, 0]
    np.testing.assert_allclose(ears[:, 0], w, atol=1e-9 * np.max(np.abs(w)))
    np.testing.assert_array_equal(ears[:, 0], ears[:, 1])


@pytest.mark.slow
def test_with_a_transparent_hrtf_btms_paths_match_too():
    from compas_acoustics.binaural import render_early_binaural

    result = run(Hybrid(diffraction="btms", **FAST), panel=True)
    assert len(result.btms_contributions) > 0
    w = result.early[:, 0]
    # encode_btms's render_signals sums blocks whose band gains agree to 0.01 dB under the first one's gain;
    # each path here keeps its own, so they may differ by that grouping: 10 ** (0.01 / 20) - 1 = 1.15e-3.
    # Measured: 4e-5 (fixed air table) and 1.2e-4 (ISO 9613-1 at 20 C, 50 %) of the peak.
    np.testing.assert_allclose(render_early_binaural(result, identity())[:, 0], w, atol=1.15e-3 * np.max(np.abs(w)))


def direct_level(ears, result, ear):
    """Level in dB of the 2 ms after the direct sound arrives, at one ear."""
    start = int(round(float(result.images.delay[result.images.order == 0][0]) * FS))
    return 10 * np.log10(np.sum(ears[start:start + 96, ear] ** 2))


def test_a_source_on_the_left_is_louder_in_the_left_ear():
    from compas_acoustics.binaural import DEFAULT_HRTF
    from compas_acoustics.binaural import HrtfSet, render_early_binaural

    # The source is at (2, 4, 1.5); a receiver at (2, 1, 1.5) facing +x has it on its left (+y)
    result = run(Hybrid(**FAST), receiver=Receiver("rec", 2.0, 1.0, 1.5))
    ears = render_early_binaural(result, HrtfSet.from_sofa(DEFAULT_HRTF, FS))
    assert direct_level(ears, result, 0) - direct_level(ears, result, 1) > 6.0


def test_the_ears_turn_with_the_receiver():
    from compas_acoustics.binaural import DEFAULT_HRTF
    from compas_acoustics.binaural import HrtfSet, render_early_binaural

    turned = Receiver("rec", 2.0, 1.0, 1.5, direction=[-1, 0, 0])
    result = run(Hybrid(**FAST), receiver=turned)
    ears = render_early_binaural(result, HrtfSet.from_sofa(DEFAULT_HRTF, FS))
    assert direct_level(ears, result, 1) - direct_level(ears, result, 0) > 6.0


def test_the_binaural_response_is_early_plus_the_magls_late_field():
    from scipy.signal import fftconvolve
    from compas_acoustics.binaural import DEFAULT_HRTF, magls_filters
    from compas_acoustics.binaural import HrtfSet, binaural_rir, render_early_binaural

    result = run(Hybrid(sh_order=1, **FAST))
    hrtf, filters = HrtfSet.from_sofa(DEFAULT_HRTF, FS), magls_filters(DEFAULT_HRTF, FS, 1)
    n = result.rir.shape[0]
    late = sum(fftconvolve(result.late[:, ch:ch + 1], filters[:, ch, :], axes=0) for ch in range(4))[:n]
    np.testing.assert_allclose(binaural_rir(result, hrtf, filters), render_early_binaural(result, hrtf) + late, atol=1e-12)


def test_the_hrtf_must_be_at_the_simulations_rate():
    from compas_acoustics.binaural import DEFAULT_HRTF
    from compas_acoustics.binaural import HrtfSet, render_early_binaural

    result = run(Hybrid(**FAST))
    with pytest.raises(ValueError, match="sample rate"):
        render_early_binaural(result, HrtfSet.from_sofa(DEFAULT_HRTF, 44100))


def test_hybrid_returns_binaural_responses():
    hybrid = Hybrid(**FAST)
    s = scene()
    ambisonic = s.simulate(hybrid)
    binaural = hybrid.binaural()
    assert binaural.keys() == ambisonic.keys()
    (ir,) = binaural.values()
    assert ir.n_channels == 2 and ir.ambisonic_order is None
    assert ir.generation["output"] == "binaural"
    assert ir.generation["hrtf"] == "KEMAR_Knowl_EarSim_SmallEars_FreeFieldComp_48kHz.sofa"
    assert ir.generation["method"] == "hybrid"
    assert ir.denormalized().shape == (next(iter(ambisonic.values())).denormalized().shape[0], 2)


def test_binaural_needs_a_run_first():
    with pytest.raises(ValueError, match="simulate"):
        Hybrid(**FAST).binaural()


def test_binaural_needs_a_directional_late_field():
    hybrid = Hybrid(sh_order=0, **FAST)
    scene().simulate(hybrid)
    with pytest.raises(ValueError, match="sh_order"):
        hybrid.binaural()


def test_n3d_and_sn3d_give_the_same_binaural_response():
    sn3d, n3d = Hybrid(seed=3, **FAST), Hybrid(normalization="N3D", seed=3, **FAST)
    scene().simulate(sn3d)
    scene().simulate(n3d)
    (a,), (b,) = sn3d.binaural().values(), n3d.binaural().values()
    np.testing.assert_allclose(a.denormalized(), b.denormalized(), atol=1e-12)
