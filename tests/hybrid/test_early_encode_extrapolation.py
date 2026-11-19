"""What ``encode`` does outside the eight bands it is given.

Material data stops at 62.5 Hz and 8 kHz, but the FFT grid runs from DC to
Nyquist. Holding the edge band's gain flat across both gaps would invent
content in two places that matter:

* At DC every image's phase ramp is exactly 1, so all gains add coherently
  into a DC component a source cannot radiate, which also biases the DRR.
* Above 8 kHz a flat hold leaves the top 1.5 octaves at 48 kHz with neither
  wall roll-off nor air absorption, while the FDN's own ``nyquist_factor``
  shortens its decay there.
"""

import numpy as np
import pyfar as pf

from compas_acoustics.hybrid.bands import BANDS, N_BANDS
from compas_acoustics.hybrid.early import ImageSourceSet, encode

FS = 48000.0


def images(gain, delays=(0.005, 0.011, 0.017, 0.023)):
    """Several images, so the coherent DC sum has something to sum."""
    delays = np.asarray(delays, dtype=float)
    g = np.tile(np.asarray(gain, dtype=float), (delays.size, 1)) if np.ndim(gain) == 1 else np.full((delays.size, N_BANDS), float(gain))
    az = np.linspace(0.0, 2.0, delays.size)
    return ImageSourceSet(
        delay=delays,
        gain=g,
        doa=pf.Coordinates(np.cos(az), np.sin(az), np.zeros(delays.size)),
        order=np.arange(delays.size),
        bands=BANDS.copy(),
        fs=FS,
    )


def one_image(gain, delay=0.01):
    """A single image, so the measured spectrum is the magnitude curve itself.

    Several images comb-filter each other, which swamps a one-decibel tilt
    measurement; only the DC tests need a sum to be coherent.
    """
    return images(gain, delays=(delay,))


def spectrum(signal):
    mag = np.abs(np.fft.rfft(signal))
    freq = np.fft.rfftfreq(signal.size, 1.0 / FS)
    return freq, mag


def test_the_extrapolation_has_no_coherent_dc():
    """The running sum stays small against the peak sample.

    Not asserted as zero. The taper's spectrum is zero at DC, so the sum over
    the *whole* transform is zero by construction, but the response is
    truncated to ``n_samples`` and the taper's own tail is cut with it.
    """
    out = encode(images(1.0), 1, 8192)
    assert abs(out[:, 0].sum()) < 0.25 * np.abs(out[:, 0]).max()


def test_infrasonic_energy_is_negligible():
    """Below the lowest band edge there is no material data and no radiation."""
    out = encode(images(1.0), 1, 8192)
    freq, mag = spectrum(out[:, 0])
    below = np.sum(mag[freq < 20.0] ** 2)
    total = np.sum(mag**2)
    assert below / total < 1e-3


def test_the_lowest_band_itself_is_not_eaten_by_the_roll_off():
    """The taper must clear the data, not encroach on the 62.5 Hz band."""
    out = encode(one_image(1.0), 1, 16384)
    freq, mag = spectrum(out[:, 0])

    def level(lo, hi):
        return np.sqrt(np.mean(mag[(freq >= lo) & (freq < hi)] ** 2))

    # flat band gains, so 62.5 Hz must sit within a decibel of 1 kHz
    tilt_db = 20.0 * np.log10(level(44.2, 88.4) / level(707.0, 1414.0))
    assert abs(tilt_db) < 1.0


def test_a_falling_gain_keeps_falling_above_the_top_band():
    """Extrapolate the measured slope, do not hold the last value flat."""
    falling = np.geomspace(1.0, 0.05, N_BANDS)  # -26 dB by 8 kHz
    out = encode(one_image(falling), 1, 16384)
    freq, mag = spectrum(out[:, 0])

    def level(lo, hi):
        return np.sqrt(np.mean(mag[(freq >= lo) & (freq < hi)] ** 2))

    top = level(5657.0, 11314.0)  # the 8 kHz octave
    beyond = level(12000.0, 20000.0)
    assert 20.0 * np.log10(beyond / top) < -3.0


def test_a_flat_gain_stays_flat_above_the_top_band():
    """A zero slope extrapolates to zero slope; nothing is invented."""
    out = encode(one_image(1.0), 1, 16384)
    freq, mag = spectrum(out[:, 0])

    def level(lo, hi):
        return np.sqrt(np.mean(mag[(freq >= lo) & (freq < hi)] ** 2))

    tilt_db = 20.0 * np.log10(level(12000.0, 20000.0) / level(5657.0, 11314.0))
    assert abs(tilt_db) < 1.0


def test_a_rising_gain_is_not_extrapolated_upward():
    """Guard against a runaway: reflection gains do not grow toward Nyquist.

    A material whose absorption falls with frequency would otherwise have its
    slope continued into the top 1.5 octaves, amplifying a region for which
    there is no data at all.
    """
    rising = np.geomspace(0.05, 1.0, N_BANDS)
    out = encode(one_image(rising), 1, 16384)
    freq, mag = spectrum(out[:, 0])

    def level(lo, hi):
        return np.sqrt(np.mean(mag[(freq >= lo) & (freq < hi)] ** 2))

    tilt_db = 20.0 * np.log10(level(12000.0, 20000.0) / level(5657.0, 11314.0))
    assert tilt_db < 1.0
