"""``RoomTarget`` from the diffuse-field models and from a response; the GEQ band extension."""

import numpy as np
import pytest

from compas_acoustics.hybrid.bands import BANDS, N_BANDS
from _rooms import box_room
from _rooms import pair as make_pair
from compas_acoustics.hybrid.target import RoomTarget
from compas_acoustics.room.geometry import SurfaceMaterial


def pair_with(alpha=0.2, dims=(8.0, 6.0, 3.0), air_absorption=True):
    return make_pair(
        geometry=box_room(dims, SurfaceMaterial("wall", np.full(N_BANDS, float(alpha)))),
        source=np.array([2.0, 2.0, 1.5]),
        receiver=np.array([5.0, 4.0, 1.5]),
        air_absorption=air_absorption,
    )


def test_from_room_sabine_differs_from_eyring_and_is_longer():
    pair = pair_with(alpha=0.4, air_absorption=False)
    sabine = RoomTarget.from_room(pair, model="sabine")
    eyring = RoomTarget.from_room(pair, model="eyring")
    assert np.all(sabine.rt[:, 0] > eyring.rt[:, 0])


def test_from_room_rejects_an_unknown_model():
    with pytest.raises(ValueError, match="model"):
        RoomTarget.from_room(pair_with(), model="norris")


def test_air_absorption_shortens_rt_and_more_so_at_high_frequency():
    dry = RoomTarget.from_room(pair_with(alpha=0.1, air_absorption=False))
    wet = RoomTarget.from_room(pair_with(alpha=0.1))
    assert np.all(wet.rt[:, 0] <= dry.rt[:, 0])
    ratio = wet.rt[:, 0] / dry.rt[:, 0]
    # the correction grows with frequency, so the ratio falls monotonically
    assert np.all(np.diff(ratio) < 0.0)
    assert ratio[0] == pytest.approx(1.0, abs=0.01)
    assert ratio[-1] < 0.9


def test_from_room_shapes_and_bands():
    target = RoomTarget.from_room(pair_with())
    assert target.rt.shape == (N_BANDS, 1)
    assert target.level is None
    np.testing.assert_allclose(target.bands, BANDS)


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
    target = RoomTarget.from_room(pair_with())
    geq = target.geq_bands()
    assert geq.shape == (10,)
    np.testing.assert_allclose(geq[1:9], target.rt[:, 0])
    assert geq[0] == pytest.approx(target.rt[0, 0])
    assert geq[9] == pytest.approx(0.5 * target.rt[-1, 0])
