"""The early encoder: band gains, delays, ambisonic channel gains and the path renderers."""

import numpy as np
import pyfar as pf
import pyFDN
import pytest
from scipy.signal import sosfilt

from compas_acoustics.hybrid.bands import BANDS, N_BANDS
from compas_acoustics.hybrid.early import ImageSourceSet, encode, render_images, render_signals
from _rooms import box_room, images, pair
from compas_acoustics.room.geometry import SurfaceMaterial


def one_image(azimuth, elevation, delay=0.01, gain=1.0, fs=48000.0):
    x = np.cos(elevation) * np.cos(azimuth)
    y = np.cos(elevation) * np.sin(azimuth)
    z = np.sin(elevation)
    g = np.full((1, N_BANDS), gain) if np.isscalar(gain) else np.asarray(gain)[None, :]
    return ImageSourceSet(
        delay=np.array([delay]),
        gain=g,
        doa=pf.Coordinates(np.array([x]), np.array([y]), np.array([z])),
        order=np.array([0]),
        bands=BANDS.copy(),
        fs=fs,
    )


def measure_band_gains(signal, fs):
    """Realised per-band level, using an INDEPENDENT analysis filterbank.

    Deliberately order 8: the analysis filterbank must not share a failure mode
    with whatever the encoder does internally.
    """
    edges, _ = pyFDN.octave_bands()
    bank = pyFDN.octave_band_filterbank(edges, fs, filter_order=8)
    reference = np.zeros_like(signal)
    reference[len(signal) // 4] = 1.0
    got = np.array([np.sqrt(np.sum(sosfilt(s, signal) ** 2)) for s in bank])
    ref = np.array([np.sqrt(np.sum(sosfilt(s, reference) ** 2)) for s in bank])
    return got / ref


def test_output_shape_matches_the_ambisonic_order():
    for order, n_ch in [(0, 1), (1, 4), (2, 9), (3, 16)]:
        out = encode(one_image(0.0, 0.0), order, 4096)
        assert out.shape == (4096, n_ch)


def test_sloped_band_gains_are_reproduced_within_a_fraction_of_a_decibel():
    """A target sloping down 26 dB across the spectrum is what an absorptive room
    produces. Reproducing it is the encoder's entire job.
    """
    fs = 48000.0
    target = np.array([1.0, 0.9, 0.8, 0.6, 0.4, 0.25, 0.12, 0.05])
    out = encode(one_image(0.0, 0.0, delay=0.01, gain=target, fs=fs), 1, 16384)
    est = measure_band_gains(out[:, 0], fs)
    est = est / est[3] * target[3]  # align on the 500 Hz band
    err_db = 20 * np.log10(est / target)
    assert np.max(np.abs(err_db[:-1])) < 0.5
    assert np.abs(err_db[-1]) < 1.5  # top band sees the analysis edge


def test_flat_band_gains_stay_flat():
    fs = 48000.0
    out = encode(one_image(0.0, 0.0, delay=0.01, gain=1.0, fs=fs), 1, 16384)
    est = measure_band_gains(out[:, 0], fs)
    err_db = 20 * np.log10(est / est[3])
    assert np.max(np.abs(err_db[:-1])) < 0.5


def test_peak_lands_at_the_requested_delay():
    fs = 48000.0
    out = encode(one_image(0.0, 0.0, delay=480 / fs, fs=fs), 1, 4096)
    assert abs(int(np.argmax(np.abs(out[:, 0]))) - 480) <= 1


def test_fractional_delay_shifts_the_waveform():
    """A half-sample delay must change the waveform, not just round to a sample."""
    fs = 48000.0
    a = encode(one_image(0.0, 0.0, delay=480.0 / fs, fs=fs), 1, 4096)
    b = encode(one_image(0.0, 0.0, delay=480.5 / fs, fs=fs), 1, 4096)
    assert not np.allclose(a, b, atol=1e-9)
    centroid = lambda w: np.sum(np.arange(w.size) * w**2) / np.sum(w**2)
    assert 0.3 < centroid(b[:, 0]) - centroid(a[:, 0]) < 0.7


def test_front_source_puts_energy_in_x_and_none_in_y_or_z():
    """ACN: 0=W, 1=Y, 2=Z, 3=X. A source dead ahead is pure W and X, and in
    N3D X carries 2n + 1 = 3 times W's energy."""
    out = encode(one_image(0.0, 0.0), 1, 4096)
    energy = np.sum(out**2, axis=0)
    assert energy[3] == pytest.approx(3.0 * energy[0], rel=1e-6)
    assert energy[1] < 1e-12 * energy[0]
    assert energy[2] < 1e-12 * energy[0]


def test_left_source_puts_energy_in_y():
    out = encode(one_image(np.pi / 2, 0.0), 1, 4096)
    energy = np.sum(out**2, axis=0)
    assert energy[1] == pytest.approx(3.0 * energy[0], rel=1e-6)
    assert energy[3] < 1e-12 * energy[0]


def test_channel_ratios_equal_the_direction_cosines():
    """SN3D first order IS the Cartesian direction cosines; N3D, the default,
    is sqrt(3) times them."""
    azimuth, elevation = 0.7, 0.3
    expected = np.array(
        [
            np.cos(elevation) * np.cos(azimuth),
            np.cos(elevation) * np.sin(azimuth),
            np.sin(elevation),
        ]
    )
    for normalization, scale in (("NM", np.sqrt(3.0)), ("SNM", 1.0)):
        out = encode(one_image(azimuth, elevation), 1, 4096, normalization=normalization)
        peak = int(np.argmax(np.abs(out[:, 0])))
        got = np.array([out[peak, 3], out[peak, 1], out[peak, 2]]) / out[peak, 0]
        np.testing.assert_allclose(got, scale * expected, atol=1e-6)


def test_w_is_unit_gain_for_a_unit_image_not_one_over_root_four_pi():
    """Pins spharpy's SNM against its SN3D, which differ by 1/sqrt(4 pi).

    A ratio test cannot see this: the two conventions differ by one global
    scale. Only an absolute check catches a response that decodes 11 dB quiet.
    """
    fs = 48000.0
    out = encode(one_image(0.0, 0.0, delay=0.01, gain=1.0, fs=fs), 1, 16384)
    assert measure_band_gains(out[:, 0], fs)[3] == pytest.approx(1.0, rel=0.05)


def test_w_energy_matches_the_sum_of_image_energies():
    fs = 48000.0
    rng = np.random.default_rng(1)
    m = 20
    iss = ImageSourceSet(
        delay=np.sort(rng.uniform(0.005, 0.05, m)),
        gain=np.full((m, N_BANDS), 0.1),
        doa=pf.Coordinates(*rng.standard_normal((3, m))),
        order=np.zeros(m, dtype=int),
        bands=BANDS.copy(),
        fs=fs,
    )
    out = encode(iss, 1, 8192)
    np.testing.assert_allclose(np.sum(out[:, 0] ** 2), np.sum(iss.gain[:, 0] ** 2), rtol=0.1)


def test_images_beyond_the_buffer_are_dropped_not_wrapped():
    out = encode(one_image(0.0, 0.0, delay=1.0, fs=48000.0), 1, 4096)
    assert np.max(np.abs(out)) < 1e-12


def test_shoebox_end_to_end_produces_a_finite_response():
    spec = pair(
        geometry=box_room((5.0, 4.0, 3.0), SurfaceMaterial("wall", np.full(N_BANDS, 0.2))),
        source=np.array([1.0, 1.0, 1.0]),
        receiver=np.array([3.0, 2.0, 1.5]),
    )
    out = encode(images(spec, max_order=3), 1, 4800)
    assert out.shape == (4800, 4)
    assert np.all(np.isfinite(out))
    assert np.max(np.abs(out)) > 0.0


def several_images(n=7, seed=3, fs=48000.0):
    rng = np.random.default_rng(seed)
    azimuth, elevation = rng.uniform(-np.pi, np.pi, n), rng.uniform(-0.5, 0.5, n)
    return ImageSourceSet(
        delay=rng.uniform(0.005, 0.03, n),
        gain=rng.uniform(0.05, 1.0, (n, N_BANDS)),
        doa=pf.Coordinates(np.cos(elevation) * np.cos(azimuth), np.cos(elevation) * np.sin(azimuth), np.sin(elevation)),
        order=np.arange(n),
        bands=BANDS.copy(),
        fs=fs,
    )


def test_render_images_is_the_sum_of_every_image_times_its_weights():
    """One outer product per image, accumulated and finished as the renderer does, to 1e-12."""
    from compas_acoustics.hybrid.early import _band_magnitude_fn, _fft_grid, _finish

    iss, n = several_images(), 2048
    weights = np.random.default_rng(4).standard_normal((7, 5))
    nfft, freq = _fft_grid(n, iss.fs)
    magnitude = _band_magnitude_fn(freq, iss.bands, iss.fs)
    spectrum = np.zeros((freq.size, 5), dtype=complex)
    for m in range(7):
        spectrum += np.outer(magnitude(iss.gain[m]) * np.exp(-2j * np.pi * freq * iss.delay[m]), weights[m])
    expected = _finish(spectrum, freq, nfft, n)
    out = render_images(iss, weights, n)
    np.testing.assert_allclose(out, expected, rtol=0, atol=1e-12 * np.max(np.abs(expected)))


def test_render_transfers_is_the_sum_of_every_path_times_its_weights():
    from compas_acoustics.hybrid.early import _band_magnitude_fn, _fft_grid, _finish, render_transfers

    iss, n = several_images(), 2048
    weights = np.random.default_rng(5).standard_normal((7, 3))
    transfers = [lambda f, k=k: np.exp(-1j * 0.1 * k * f / iss.fs) / (1.0 + k) for k in range(7)]
    nfft, freq = _fft_grid(n, iss.fs)
    magnitude = _band_magnitude_fn(freq, iss.bands, iss.fs)
    spectrum = np.zeros((freq.size, 3), dtype=complex)
    for m in range(7):
        spectrum += np.outer(transfers[m](freq) * magnitude(iss.gain[m]) * np.exp(-2j * np.pi * freq * iss.delay[m]), weights[m])
    expected = _finish(spectrum, freq, nfft, n)
    out = render_transfers(iss.delay, transfers, iss.gain, weights, iss.bands, iss.fs, n)
    np.testing.assert_allclose(out, expected, rtol=0, atol=1e-12 * np.max(np.abs(expected)))


def test_render_signals_equals_render_images_for_impulses():
    iss = one_image(0.3, 0.1)
    n = 4096
    delay = round(iss.delay[0] * iss.fs)
    iss = ImageSourceSet(np.array([delay / iss.fs]), iss.gain, iss.doa, iss.order, iss.bands, iss.fs)
    weights = np.array([[1.0, 0.5, -0.25, 0.75]])
    a = render_images(iss, weights, n)
    b = render_signals([(delay, weights)], iss.gain, iss.bands, iss.fs, n)
    np.testing.assert_allclose(b, a, atol=1e-12 * np.max(np.abs(a)))
