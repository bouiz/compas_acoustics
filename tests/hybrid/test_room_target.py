"""Ported from hybrid-rir (gpapadimitri/hybrid-rir @ a272da7, tests/test_room_target.py)."""

import numpy as np
import pyrato
import pytest

from compas_acoustics.hybrid.bands import BANDS, N_BANDS
from _compat import Material, RoomSpec, ShoeboxGeometry
from compas_acoustics.hybrid.target import RoomTarget
from compas_acoustics.diffuse import mean_absorption


def spec_with(alpha=0.2, dims=(8.0, 6.0, 3.0)):
    mat = Material.from_scalar(alpha)
    geom = ShoeboxGeometry(dims, {n: mat for n in ShoeboxGeometry.WALL_NAMES})
    return RoomSpec(
        geometry=geom,
        source=np.array([2.0, 2.0, 1.5]),
        receiver=np.array([5.0, 4.0, 1.5]),
    )


def test_from_room_without_air_absorption_equals_pyrato_directly():
    """from_room is an adapter; the model must be pyrato's, unmodified."""
    spec = spec_with()
    target = RoomTarget.from_room(spec, model="eyring", air_absorption=False)
    expected = pyrato.parametric.reverberation_time_eyring(
        volume=spec.geometry.volume,
        surface_area=spec.geometry.surface_area,
        mean_absorption=mean_absorption(spec.geometry),
        speed_of_sound=spec.speed_of_sound,
    )
    np.testing.assert_allclose(target.rt[:, 0], expected, rtol=1e-12)


def test_from_room_sabine_differs_from_eyring_and_is_longer():
    spec = spec_with(alpha=0.4)
    sabine = RoomTarget.from_room(spec, model="sabine", air_absorption=False)
    eyring = RoomTarget.from_room(spec, model="eyring", air_absorption=False)
    assert np.all(sabine.rt[:, 0] > eyring.rt[:, 0])


def test_from_room_rejects_an_unknown_model():
    with pytest.raises(ValueError, match="model"):
        RoomTarget.from_room(spec_with(), model="norris")


def test_air_absorption_shortens_rt_and_more_so_at_high_frequency():
    spec = spec_with(alpha=0.1)
    dry = RoomTarget.from_room(spec, air_absorption=False)
    wet = RoomTarget.from_room(spec, air_absorption=True)
    assert np.all(wet.rt[:, 0] <= dry.rt[:, 0])
    ratio = wet.rt[:, 0] / dry.rt[:, 0]
    # the correction grows with frequency, so the ratio falls monotonically
    assert np.all(np.diff(ratio) < 0.0)
    assert ratio[0] == pytest.approx(1.0, abs=0.01)
    assert ratio[-1] < 0.9


def test_from_room_shapes_and_bands():
    target = RoomTarget.from_room(spec_with())
    assert target.rt.shape == (N_BANDS, 1)
    assert target.level.shape == (N_BANDS, 1)
    np.testing.assert_allclose(target.bands, BANDS)


def test_from_rir_recovers_the_rt_of_a_synthetic_exponential_decay():
    fs = 48000.0
    rt_true = 0.8
    n = int(3 * rt_true * fs)
    rng = np.random.default_rng(0)
    t = np.arange(n) / fs
    rir = rng.standard_normal(n) * 10 ** (-3 * t / rt_true)
    target = RoomTarget.from_rir(rir, fs)
    # skip the edge bands, where the octave filters have too little support
    np.testing.assert_allclose(target.rt[1:-1, 0], rt_true, rtol=0.15)


def test_from_rir_decay_db_sets_the_fit_range():
    """A knee at -25 dB: T20 stays on the fast slope, T30 reaches the slow one."""
    fs = 48000.0
    n = int(4.0 * fs)
    t = np.arange(n) / fs
    rng = np.random.default_rng(0)
    fast = 10 ** (-3 * t / 0.8)
    slow = 10 ** (-25 / 20) * 10 ** (-3 * t / 3.0)
    rir = rng.standard_normal(n) * np.sqrt(fast**2 + slow**2)
    t20 = RoomTarget.from_rir(rir, fs, decay_db=20.0).rt[2:-1, 0]
    t30 = RoomTarget.from_rir(rir, fs).rt[2:-1, 0]
    np.testing.assert_allclose(t20, 0.8, rtol=0.2)
    assert np.all(t30 > 1.15 * t20)


def test_geq_bands_extends_eight_bands_to_ten():
    target = RoomTarget.from_room(spec_with())
    geq = target.geq_bands(dc_factor=0.9, nyquist_factor=0.5)
    assert geq.shape == (10,)
    np.testing.assert_allclose(geq[1:9], target.rt[:, 0])
    assert geq[0] == pytest.approx(0.9 * target.rt[0, 0])
    assert geq[9] == pytest.approx(0.5 * target.rt[-1, 0])


def test_geq_bands_defaults():
    target = RoomTarget.from_room(spec_with())
    geq = target.geq_bands()
    assert geq[0] == pytest.approx(target.rt[0, 0])
    assert geq[9] == pytest.approx(0.5 * target.rt[-1, 0])
