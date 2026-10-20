"""Binaural rendering of the hybrid: an HRTF per early path and per delay line; MagLS decoding for the Auraliser."""

import numpy as np
import pytest
from compas.datastructures import Mesh

from compas_acoustics import AcousticMaterial
from compas_acoustics import AcousticObject
from compas_acoustics import AcousticObjectsLibrary
from compas_acoustics import Receiver
from compas_acoustics import Scene
from compas_acoustics import Source
from compas_acoustics.simulation_methods import Hybrid

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

    a = Auraliser("room1", {"ir1": {"ir": np.ones((32, 4)), "sampling_rate": FS, "ambisonic_order": 1,
                                     "channel_order": "acn", "normalization": "sn3d"}},
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
    assert resampled.sampling_rate == 44100
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


def test_the_binaural_response_is_early_plus_each_line_through_its_hrir():
    from scipy.signal import fftconvolve
    from compas_acoustics.binaural import DEFAULT_HRTF
    from compas_acoustics.binaural import HrtfSet, binaural_rir, render_early_binaural

    result = run(Hybrid(sh_order=1, **FAST))
    hrtf = HrtfSet.from_sofa(DEFAULT_HRTF, FS)
    n, lines = result.rir.shape[0], result.late_lines.shape[1]
    late = sum(fftconvolve(result.late_lines[:, i:i + 1] / np.sqrt(lines), hrtf.hrirs[hrtf.nearest(result.line_directions[i])[0]].T,
                           axes=0)[:n] for i in range(lines))
    np.testing.assert_allclose(binaural_rir(result, hrtf), render_early_binaural(result, hrtf) + late, atol=1e-12)


def test_the_hrtf_must_be_at_the_simulations_rate():
    from compas_acoustics.binaural import DEFAULT_HRTF
    from compas_acoustics.binaural import HrtfSet, render_early_binaural

    result = run(Hybrid(**FAST))
    with pytest.raises(ValueError, match="sample rate"):
        render_early_binaural(result, HrtfSet.from_sofa(DEFAULT_HRTF, 44100))


def test_hybrid_returns_binaural_responses():
    hybrid = Hybrid(output=["ambisonic", "binaural"], **FAST)
    s = scene()
    ambisonic = s.simulate(hybrid)
    binaural = hybrid.extra_responses["hybrid_binaural"]
    assert binaural.keys() == ambisonic.keys()
    (ir,) = binaural.values()
    assert ir.n_channels == 2 and ir.ambisonic_order is None
    assert ir.generation["output"] == "binaural"
    assert ir.generation["hrtf"] == "KEMAR_FreeFieldComp_48kHz.sofa"
    assert ir.generation["method"] == "hybrid"
    assert ir.denormalized().shape == (next(iter(ambisonic.values())).denormalized().shape[0], 2)


def test_the_binaural_output_does_not_depend_on_sh_order():
    """Every path and every line takes its own HRIR; the ambisonic order only sets the ambisonic output."""
    (zero,) = scene().simulate(Hybrid(sh_order=0, num_delays=16, seed=3, **FAST)).values()
    (first,) = scene().simulate(Hybrid(sh_order=1, num_delays=16, seed=3, **FAST)).values()
    np.testing.assert_allclose(zero.denormalized(), first.denormalized(), atol=1e-12)


def test_n3d_and_sn3d_give_the_same_binaural_response():
    (a,) = scene().simulate(Hybrid(seed=3, **FAST)).values()
    (b,) = scene().simulate(Hybrid(normalization="N3D", seed=3, **FAST)).values()
    np.testing.assert_allclose(a.denormalized(), b.denormalized(), atol=1e-12)


def _kemar_covariances(order, monkeypatch, delivered):
    """Diffuse-field binaural covariance per bin, 200 Hz to 16 kHz, of the KEMAR set and of its MagLS decoder.

    The decoder is the matched frequency response, or with ``delivered`` the filters
    ``magls_filters`` returns, which are cut to the HRIR length (about 0.07 % of the energy of
    the order 1 filters lies later, mostly pre-ringing that wraps around the FFT).
    """
    import pyfar as pf
    import spharpy
    from compas_acoustics import binaural
    from compas_acoustics.binaural import DEFAULT_HRTF, _quadrature_weights, magls_filters

    hrirs, directions, _ = pf.io.read_sofa(str(DEFAULT_HRTF), verbose=False)
    weights = _quadrature_weights(directions.azimuth, directions.colatitude)
    n_fft = 1024
    hrtfs = np.fft.rfft(hrirs.time, n=n_fft)
    Y = np.sqrt(4 * np.pi) * spharpy.spherical.spherical_harmonic_basis_real(
        order, directions, normalization="SN3D", channel_convention="ACN", condon_shortley=False)
    matched = []
    monkeypatch.setattr(binaural, "_BINAURAL_FILTERS_CACHE", {})
    monkeypatch.setattr(binaural, "_diffuse_covariance_matched",
                        lambda *args, original=binaural._diffuse_covariance_matched: matched.append(original(*args)) or matched[-1])
    filters = magls_filters(DEFAULT_HRTF, FS, order)
    decoder = np.fft.rfft(filters, n=n_fft, axis=0).transpose(1, 2, 0) if delivered else matched[0]
    h_decoded = np.einsum("dq,qek->ked", Y, decoder)
    h_target = hrtfs.transpose(2, 1, 0)
    c_target = np.einsum("ked,d,kfd->kef", h_target, weights, h_target.conj())
    c_decoded = np.einsum("ked,d,kfd->kef", h_decoded, weights, h_decoded.conj())
    freqs = np.fft.rfftfreq(n_fft, 1 / FS)
    band = (freqs >= 200) & (freqs <= 16000)
    return c_target[band], c_decoded[band]


def _coherence(c):
    return np.abs(c[:, 0, 1]) / np.sqrt(np.real(c[:, 0, 0]) * np.real(c[:, 1, 1]))


def _ear_energies(c):
    return np.real(np.stack([c[:, 0, 0], c[:, 1, 1]]))


@pytest.mark.parametrize("order", [1, 3])
def test_decoder_reproduces_the_diffuse_field_coherence(order, monkeypatch):
    pytest.importorskip("spharpy")
    c_target, c_decoded = _kemar_covariances(order, monkeypatch, delivered=False)
    assert np.max(np.abs(_coherence(c_decoded) - _coherence(c_target))) < 0.02


@pytest.mark.parametrize("order", [1, 3])
def test_decoder_reproduces_the_diffuse_field_ear_energies(order, monkeypatch):
    pytest.importorskip("spharpy")
    c_target, c_decoded = _kemar_covariances(order, monkeypatch, delivered=False)
    assert np.max(np.abs(10 * np.log10(_ear_energies(c_decoded) / _ear_energies(c_target)))) < 0.1


@pytest.mark.parametrize("order", [1, 3])
def test_the_returned_filters_keep_most_of_the_covariance_constraint(order, monkeypatch):
    """Cutting the matched filters to the HRIR length leaves up to 0.05 coherence and 1.1 dB of error at order 1 and 3."""
    pytest.importorskip("spharpy")
    c_target, c_decoded = _kemar_covariances(order, monkeypatch, delivered=True)
    assert np.max(np.abs(_coherence(c_decoded) - _coherence(c_target))) < 0.05
    assert np.max(np.abs(10 * np.log10(_ear_energies(c_decoded) / _ear_energies(c_target)))) < 1.1


def test_a_plane_wave_from_the_left_stays_left():
    spharpy = pytest.importorskip("spharpy")
    import pyfar as pf
    from compas_acoustics.binaural import DEFAULT_HRTF, magls_filters

    y = np.sqrt(4 * np.pi) * spharpy.spherical.spherical_harmonic_basis_real(
        1, pf.Coordinates.from_spherical_colatitude(np.pi / 2, np.pi / 2, 1.0),
        normalization="SN3D", channel_convention="ACN", condon_shortley=False)[0]
    ears = np.einsum("c,nce->ne", y, magls_filters(DEFAULT_HRTF, FS, 1))
    e_left, e_right = np.sum(ears ** 2, axis=0)
    assert 10 * np.log10(e_left / e_right) > 10.0


def test_partial_sphere_hrtf_set():
    import pyfar as pf
    from compas_acoustics.binaural import DEFAULT_HRTF, _quadrature_weights

    _, directions, _ = pf.io.read_sofa(str(DEFAULT_HRTF), verbose=False)
    upper = directions.elevation > 0
    w = _quadrature_weights(directions.azimuth[upper], directions.colatitude[upper])
    assert np.all(np.isfinite(w)) and w.sum() == pytest.approx(1.0)


def test_the_covariance_constraint_is_part_of_the_cache_key():
    pytest.importorskip("spharpy")
    from compas_acoustics.binaural import DEFAULT_HRTF, magls_filters

    constrained = magls_filters(DEFAULT_HRTF, FS, 1)
    plain = magls_filters(DEFAULT_HRTF, FS, 1, covariance_constraint=False)
    assert constrained is magls_filters(DEFAULT_HRTF, FS, 1, covariance_constraint=True)
    assert not np.allclose(constrained, plain)


def test_a_singular_covariance_zeroes_the_bin_as_saf_does_and_warns_once(monkeypatch, tmp_path):
    """Ears identical at DC make the 2x2 covariance singular there; the other bins keep their correction."""
    pytest.importorskip("spharpy")
    import pyfar as pf
    from compas_acoustics import binaural
    from compas_acoustics.binaural import DEFAULT_HRTF, magls_filters

    hrirs, directions, _ = pf.io.read_sofa(str(DEFAULT_HRTF), verbose=False)
    time = np.array(hrirs.time)
    time[:, 1] += ((time[:, 0].sum(axis=-1) - time[:, 1].sum(axis=-1)) / time.shape[-1])[:, None]
    synthetic = pf.Signal(time, hrirs.sampling_rate)
    monkeypatch.setattr(binaural.pf.io, "read_sofa", lambda *args, **kwargs: (synthetic, directions, None))
    monkeypatch.setattr(binaural, "_BINAURAL_FILTERS_CACHE", {})
    matched = []
    monkeypatch.setattr(binaural, "_diffuse_covariance_matched",
                        lambda *args, original=binaural._diffuse_covariance_matched: matched.append(original(*args)) or matched[-1])
    path = tmp_path / "dc_identical.sofa"

    with pytest.warns(UserWarning, match=r"1 bin") as caught:
        constrained = magls_filters(path, FS, 1)
    assert len([w for w in caught if "bin" in str(w.message)]) == 1
    plain = magls_filters(path, FS, 1, covariance_constraint=False)
    assert np.all(np.isfinite(constrained)) and not np.allclose(constrained, plain)
    np.testing.assert_array_equal(matched[0][..., 0], 0.0)
    assert np.all(np.max(np.abs(matched[0][..., 1:-1]), axis=(0, 1)) > 0.0)
