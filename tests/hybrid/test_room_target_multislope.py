"""Multi-slope ``RoomTarget.from_rir`` and the slope selection of ``geq_bands``."""

import numpy as np
import pytest

from compas_acoustics.hybrid.bands import N_BANDS
from compas_acoustics.hybrid.target import RoomTarget

import importlib.util

needs_multislope = pytest.mark.skipif(
    importlib.util.find_spec("multislope") is None, reason="multislope is not installed"
)

FS = 48000.0


def decaying_noise(decay_times, offsets_db, seed=0, duration=6.0):
    """Noise whose ENERGY envelope is a sum of exponentials.

    Multi-slope decay is defined on the energy decay curve, so the envelope is
    the square root of a sum of energy exponentials. Summing amplitude
    envelopes instead introduces a cross term and is not a clean multi-slope
    signal.
    """
    rng = np.random.default_rng(seed)
    n = int(duration * FS)
    t = np.arange(n) / FS
    energy = sum(
        10 ** (-off / 10) * 10 ** (-6 * t / T)
        for T, off in zip(decay_times, offsets_db)
    )
    return rng.standard_normal(n) * np.sqrt(energy)


def test_single_slope_target_is_two_dimensional_with_one_column():
    target = RoomTarget.from_rir(decaying_noise([0.8], [0.0]), FS)
    assert target.rt.shape == (N_BANDS, 1)
    assert target.level.shape == (N_BANDS, 1)
    assert target.n_slopes == 1


def test_a_single_slope_is_pyfdns_estimate():
    """The n_slopes=1 path is pyFDN's single-slope estimate."""
    import pyFDN

    rir = decaying_noise([0.8], [0.0])
    target = RoomTarget.from_rir(rir, FS)
    expected, _ = pyFDN.estimate_rt_bands(rir, FS)
    np.testing.assert_allclose(target.rt[:, 0], expected)


@needs_multislope
def test_auto_detection_finds_one_slope_in_a_single_slope_room():
    target = RoomTarget.from_rir(decaying_noise([0.8], [0.0]), FS, n_slopes=0)
    assert target.n_slopes == 1
    assert np.median(target.rt[:, 0]) == pytest.approx(0.8, rel=0.2)


@needs_multislope
def test_auto_detection_finds_two_slopes_in_a_coupled_room():
    """0.4 s and 1.8 s, 30 dB apart: separated enough to be resolvable."""
    target = RoomTarget.from_rir(
        decaying_noise([0.4, 1.8], [0.0, 30.0]), FS, n_slopes=0
    )
    assert target.n_slopes == 2
    fast = np.median(target.rt.min(axis=1))
    slow = np.median(target.rt.max(axis=1))
    assert fast == pytest.approx(0.4, rel=0.4)
    assert slow == pytest.approx(1.8, rel=0.4)
    assert slow > 2 * fast


@needs_multislope
def test_slopes_are_ordered_fastest_first():
    target = RoomTarget.from_rir(
        decaying_noise([0.4, 1.8], [0.0, 30.0]), FS, n_slopes=0
    )
    assert np.all(np.diff(target.rt, axis=1) >= 0.0)


@needs_multislope
def test_inactive_slopes_are_dropped_not_carried_as_zeros():
    """DecayFitNet reports three columns; zero decay time means inactive."""
    target = RoomTarget.from_rir(decaying_noise([0.8], [0.0]), FS, n_slopes=0)
    assert np.all(target.rt > 0.0)


@needs_multislope
def test_geq_bands_selects_a_slope():
    target = RoomTarget.from_rir(
        decaying_noise([0.4, 1.8], [0.0, 30.0]), FS, n_slopes=0
    )
    fast = target.geq_bands(slope=0)
    slow = target.geq_bands(slope=1)
    assert fast.shape == (10,)
    assert slow.shape == (10,)
    assert slow[5] > fast[5]


def test_geq_bands_rejects_a_slope_that_does_not_exist():
    target = RoomTarget.from_rir(decaying_noise([0.8], [0.0]), FS)
    with pytest.raises(IndexError):
        target.geq_bands(slope=1)


def test_from_room_is_single_slope():
    """Geometry gives one decay per band; there is no second slope to predict."""
    from _rooms import box_room, pair

    from compas_acoustics.room.geometry import SurfaceMaterial

    spec = pair(
        geometry=box_room((8.0, 6.0, 3.0), SurfaceMaterial("wall", np.full(N_BANDS, 0.2))),
        source=np.array([2.0, 2.0, 1.5]),
        receiver=np.array([5.0, 4.0, 1.5]),
    )
    target = RoomTarget.from_room(spec)
    assert target.n_slopes == 1
    assert target.rt.shape == (N_BANDS, 1)


@needs_multislope
def test_forcing_two_slopes_on_a_single_slope_room_is_documented_as_harmful():
    """Not a bug, a warning: n_slopes=2 forced on one slope corrupts the fit.

    The estimator has no way to decline, so the caller must. This test pins the behaviour so the docstring
    stays honest.
    """
    forced = RoomTarget.from_rir(decaying_noise([0.8], [0.0]), FS, n_slopes=2)
    auto = RoomTarget.from_rir(decaying_noise([0.8], [0.0]), FS, n_slopes=0)
    assert forced.n_slopes == 2
    assert auto.n_slopes == 1
    assert abs(np.median(auto.rt) - 0.8) < abs(np.median(forced.rt) - 0.8)
