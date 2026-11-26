"""The ray-traced late envelope imposed on the network output."""

import numpy as np
import pytest

from _rooms import box_room, images
from _rooms import pair as make_pair
from compas_acoustics.hybrid.synthesis import prepare
from compas_acoustics.hybrid.ambisonics import render_ambisonics
from compas_acoustics.hybrid.target import RoomTarget
from compas_acoustics.room.geometry import SurfaceMaterial
from compas_acoustics.hybrid.envelope import _smoothed_over_time, envelope_gains, octave_band_split, shape_late

FS = 48000.0
N = int(1.6 * FS)
T = np.arange(N) / FS


def parts(rt=0.8, onset=0.03, seed=0):
    """A few early impulses, and a decaying 4-channel noise from ``onset``."""
    rng = np.random.default_rng(seed)
    early = np.zeros((N, 4))
    for t, a in ((0.01, 1.0), (0.015, 0.5), (0.022, 0.4), (0.027, 0.3)):
        early[int(t * FS), 0] = a
    decay = 10 ** (-3 * np.clip(T - onset, 0.0, None) / rt) * (T >= onset)
    late = 0.02 * rng.standard_normal((N, 4)) * decay[:, None]
    return early, late


def energy_db(x, a, b):
    return 10 * np.log10(np.sum(x[int(a * FS) : int(b * FS)] ** 2))


# ----------------------------------------------------------------- the bank --
@pytest.mark.parametrize("n, columns", [(9600, 96), (4801, 3), (4799, 96), (20000, 1), (9600, 100)])
def test_the_smoothing_over_time_is_fftconvolve_bit_for_bit(n, columns):
    from scipy.signal import fftconvolve

    x = np.random.default_rng(2).random((n, columns)) ** 4
    window = np.hanning(4802)[1:-1]
    window /= window.sum()
    np.testing.assert_array_equal(_smoothed_over_time(x, window), fftconvolve(x, window[:, np.newaxis], mode="same", axes=0))


def test_the_band_split_is_pyfar_reconstructing_bank():
    """pyfar's linear-phase bank applied by pyfar itself, its delay removed, to 1e-12; one channel and several."""
    import pyfar as pf

    from compas_acoustics.hybrid.envelope import _N_FILT, _TOP_BAND_HZ

    x = np.random.default_rng(2).standard_normal((3000, 4)) * np.exp(-np.arange(3000) / 600.0)[:, None]
    padded = np.concatenate([x.T, np.zeros((4, _N_FILT))], axis=1)
    out = pf.dsp.filter.reconstructing_fractional_octave_bands(pf.Signal(padded, FS), num_fractions=1, frequency_range=(63, _TOP_BAND_HZ), n_samples=_N_FILT)
    filtered = out[0] if isinstance(out, tuple) else out
    delay = _N_FILT // 2
    expected = np.transpose(filtered.time[:, :, delay : delay + 3000], (0, 2, 1))
    bands = octave_band_split(x, FS)
    np.testing.assert_allclose(bands, expected, rtol=0, atol=1e-12 * np.max(np.abs(expected)))
    np.testing.assert_allclose(octave_band_split(x[:, 0], FS), expected[:, :, 0], rtol=0, atol=1e-12 * np.max(np.abs(expected)))


# ---------------------------------------------------------------- the gains --
def test_a_louder_reference_leaves_the_late_field_alone():
    early, late = parts()
    shaped = shape_late(late, early[:, 0], late[:, 0], early[:, 0] + 3.0 * late[:, 0], FS)
    np.testing.assert_allclose(shaped, late, atol=1e-6)


def test_a_quieter_reference_attenuates_by_the_difference():
    early, late = parts()
    reference = early[:, 0] + 10 ** (-6 / 20) * late[:, 0]
    shaped = shape_late(late, early[:, 0], late[:, 0], reference, FS)
    lost = energy_db(shaped[:, 0], 0.15, 0.6) - energy_db(late[:, 0], 0.15, 0.6)
    assert lost == pytest.approx(-6.0, abs=0.5)


def test_every_channel_gets_the_same_gain():
    early, late = parts()
    reference = early[:, 0] + 0.5 * late[:, 0]
    shaped = shape_late(late, early[:, 0], late[:, 0], reference, FS)
    ratio = [energy_db(shaped[:, c], 0.15, 0.6) - energy_db(late[:, c], 0.15, 0.6) for c in range(4)]
    assert np.ptp(ratio) < 0.5


def test_the_gain_is_held_once_the_reference_has_decayed_20_db():
    """A burst in the reference after the hold point does not reach the output."""
    early, late = parts()
    reference = early[:, 0] + 0.5 * late[:, 0]
    burst = reference.copy()
    i = int(1.0 * FS)
    burst[i : i + int(0.02 * FS)] *= 30.0
    plain = shape_late(late, early[:, 0], late[:, 0], reference, FS)
    bursty = shape_late(late, early[:, 0], late[:, 0], burst, FS)
    np.testing.assert_allclose(bursty[int(0.9 * FS) :], plain[int(0.9 * FS) :], atol=1e-9)


def test_a_short_reference_holds_its_last_gain():
    early, late = parts()
    reference = (early[:, 0] + 0.5 * late[:, 0])[: int(0.8 * FS)]
    shaped = shape_late(late, early[:, 0], late[:, 0], reference, FS)
    assert shaped.shape == late.shape
    tail = energy_db(shaped[:, 0], 1.0, 1.4) - energy_db(late[:, 0], 1.0, 1.4)
    assert tail == pytest.approx(-6.0, abs=1.0)


def test_a_reference_on_another_scale_warns():
    early, late = parts()
    reference = 100.0 * (early[:, 0] + late[:, 0])
    with pytest.warns(UserWarning, match="scale"):
        shape_late(late, early[:, 0], late[:, 0], reference, FS)


# -------------------------------------------------------------- the pipeline --
def shoebox():
    return make_pair(
        geometry=box_room((7.0, 5.0, 3.0), SurfaceMaterial("wall", np.full(8, 0.2))),
        source=np.array([1.5, 1.5, 1.6]),
        receiver=np.array([5.0, 3.0, 1.5]),
        fs=FS,
        air_absorption=False,
    )


def test_a_reference_shapes_only_the_late_field():
    pair = shoebox()
    plain = render_ambisonics(prepare(pair, images(pair, 3), RoomTarget.from_room(pair), max_order=2, duration_s=0.5))
    reference = plain.early[:, 0] + 0.5 * plain.late[:, 0]
    shaped = render_ambisonics(prepare(pair, images(pair, 3), RoomTarget.from_room(pair), max_order=2, duration_s=0.5, late_envelope=reference))
    np.testing.assert_array_equal(shaped.early, plain.early)
    np.testing.assert_allclose(shaped.rir, shaped.early + shaped.late)
    assert np.sum(shaped.late**2) < np.sum(plain.late**2)


def test_below_32_khz_the_bank_has_eight_bands_and_the_shaping_uses_only_those():
    """At 24 kHz the 16 kHz band is at or above Nyquist: eight bands, every one written and summed."""
    fs = 24000.0
    n = int(1.0 * fs)
    t = np.arange(n) / fs
    rng = np.random.default_rng(3)
    early_w = np.zeros(n)
    early_w[int(0.01 * fs)] = 1.0
    lines = 0.02 * rng.standard_normal((n, 16)) * (10 ** (-3 * t / 0.6) * (t >= 0.03))[:, None]
    late_w = lines.sum(axis=1) / 4.0
    reference = early_w + 0.5 * late_w

    bands = octave_band_split(lines, fs)
    assert bands.shape == (8, n, 16)
    np.testing.assert_allclose(bands.sum(axis=0), lines, atol=1e-6 * np.max(np.abs(lines)))
    gains = envelope_gains(reference, early_w, late_w, fs)
    assert gains.shape == (8, n)

    first = shape_late(lines, early_w, late_w, reference, fs)
    second = shape_late(lines, early_w, late_w, reference, fs)
    assert np.all(np.isfinite(first))
    assert np.array_equal(first, second)
    np.testing.assert_array_equal(first, np.einsum("bn,bnc->nc", gains, bands))
