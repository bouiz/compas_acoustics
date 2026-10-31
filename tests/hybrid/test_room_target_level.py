"""``RoomTarget.level`` must be in pyFDN's units, on both construction paths.

pyFDN's ``level`` is the linear *amplitude* ``L`` of the per-octave-band decay
``L * 10 ** (-3 t / T)``, as returned by ``estimate_initial_level_bands``. That
is neither an energy nor a spectral density, so predicting it from
diffuse-field theory in ``from_room`` takes two conversions; these tests pin
them.
"""

import numpy as np
import pyFDN
import pytest
from scipy.signal import sosfilt

from compas_acoustics.hybrid.bands import BANDS, BANDWIDTHS
from _compat import Material, RoomSpec, ShoeboxGeometry
from compas_acoustics.hybrid.target import RoomTarget


def spec_with(alpha, dims=(8.0, 6.0, 3.0)):
    material = (
        Material.from_scalar(alpha)
        if np.isscalar(alpha)
        else Material(np.asarray(alpha, dtype=float), BANDS.copy())
    )
    geom = ShoeboxGeometry(dims, {n: material for n in ShoeboxGeometry.WALL_NAMES})
    return RoomSpec(
        geometry=geom,
        source=np.array([2.0, 2.0, 1.5]),
        receiver=np.array([5.0, 4.0, 1.5]),
    )


def test_a_spectrally_white_decay_has_level_proportional_to_root_bandwidth():
    """The reference measurement the other tests rest on.

    ``estimate_initial_level_bands`` matches total energy *within each octave
    band*, and a wide band collects proportionally more of a white spectrum.
    So the level of a spectrally flat decay is not flat: it rises as
    ``sqrt(bandwidth)``, 3 dB per octave. Any target that means "flat late
    field" must say so in these units.
    """
    fs = 48000.0
    rt_true = 0.5
    n = int(2 * rt_true * fs)
    t = np.arange(n) / fs
    rng = np.random.default_rng(0)
    white = rng.standard_normal(n) * 10 ** (-3 * t / rt_true)

    rt, _ = pyFDN.estimate_rt_bands(white, fs)
    level, _ = pyFDN.estimate_initial_level_bands(white, rt, fs)

    np.testing.assert_allclose(
        level / level[-1], np.sqrt(BANDWIDTHS / BANDWIDTHS[-1]), rtol=0.15
    )


def test_from_room_level_describes_a_spectrally_flat_late_field():
    """Frequency-flat absorption must mean a frequency-flat late field.

    A flat ``level`` array would not: in pyFDN's units that specifies a decay
    falling 3 dB per octave, 21 dB across this package's eight bands. The
    diffuse-field prediction has to carry the same ``sqrt(bandwidth)`` weight
    that the estimator applies.
    """
    target = RoomTarget.from_room(spec_with(0.25), air_absorption=False)
    shape = target.level[:, 0] / target.level[-1, 0]
    np.testing.assert_allclose(shape, np.sqrt(BANDWIDTHS / BANDWIDTHS[-1]), rtol=1e-12)


def test_from_room_level_is_an_amplitude_not_an_energy():
    """Doubling the absorption halves the diffuse energy, so level falls 3 dB.

    ``level`` is an amplitude, so the ``1 / A`` of diffuse-field theory enters
    under a square root. Treating it as an energy doubles every spectral tilt
    in decibels.
    """
    quiet = RoomTarget.from_room(spec_with(0.5), air_absorption=False)
    loud = RoomTarget.from_room(spec_with(0.25), air_absorption=False)
    ratio_db = 20.0 * np.log10(loud.level[:, 0] / quiet.level[:, 0])
    np.testing.assert_allclose(ratio_db, 3.0103, rtol=1e-6)


def test_from_room_level_tilt_matches_the_absorption_slope_in_amplitude():
    """A 16:1 absorption ratio is 12 dB of amplitude, not 24 dB."""
    alpha = np.array([0.05, 0.05, 0.05, 0.05, 0.8, 0.8, 0.8, 0.8])
    target = RoomTarget.from_room(spec_with(alpha), air_absorption=False)
    # remove the sqrt(bandwidth) weight to isolate the absorption term
    shape = target.level[:, 0] / np.sqrt(BANDWIDTHS)
    tilt_db = 20.0 * np.log10(shape[0] / shape[-1])
    assert tilt_db == pytest.approx(10.0 * np.log10(0.8 / 0.05), rel=1e-9)


def test_from_room_and_from_rir_agree_on_a_room_they_both_describe():
    """The two constructors must produce levels a caller can compare.

    ``from_rir`` measures with pyFDN's estimator, so it is in pyFDN's units by
    construction; ``from_room`` predicts. Synthesising a decay with the shape ``from_room``
    specifies and measuring it back must return that same shape.
    """
    fs = 48000.0
    spec = spec_with(np.array([0.05, 0.08, 0.12, 0.2, 0.3, 0.45, 0.6, 0.75]))
    target = RoomTarget.from_room(spec, air_absorption=False)

    rt = float(np.median(target.rt[:, 0]))
    n = int(2 * rt * fs)
    t = np.arange(n) / fs
    rng = np.random.default_rng(1)
    noise = rng.standard_normal(n) * 10 ** (-3 * t / rt)

    # shape the white decay to the target's spectral envelope. A band's share
    # of a white signal scales with sqrt(bandwidth), so the per-band weight
    # that realises a given level is level / sqrt(bandwidth).
    edges, _ = pyFDN.octave_bands()
    weights = target.level[:, 0] / np.sqrt(BANDWIDTHS)
    shaped = np.zeros(n)
    for weight, sos in zip(weights, pyFDN.octave_band_filterbank(edges, fs, 8)):
        shaped += weight * sosfilt(sos, noise)

    measured = RoomTarget.from_rir(shaped, fs)
    got = 20.0 * np.log10(measured.level[1:-1, 0] / measured.level[-2, 0])
    want = 20.0 * np.log10(target.level[1:-1, 0] / target.level[-2, 0])
    np.testing.assert_allclose(got, want, atol=2.0)
