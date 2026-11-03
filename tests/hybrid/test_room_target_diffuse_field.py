"""``RoomTarget.from_room`` must say when Sabine and Eyring do not apply.

Both models collapse the room to one area-weighted mean absorption and assume
a diffuse field. That assumption fails in two independent ways, in both of
which the returned reverberation time is too short with nothing in the return
value to show it:

- absorption concentrated on some surfaces (a carpeted, acoustically treated
  room);
- high absorption spread uniformly: the spread of reflection order at a given
  path length makes ``E[(1 - a) ** n]`` exceed ``(1 - a) ** E[n]``.

The models are pyrato's and are used correctly; what the target can do is
refuse to be silent.
"""

import warnings

import numpy as np
import pytest

from compas_acoustics.hybrid.bands import BANDS
from compas_acoustics.room.diffuse import DiffuseFieldWarning
from _rooms import box_room, pair
from compas_acoustics.hybrid.target import RoomTarget
from compas_acoustics.room.geometry import SurfaceMaterial

DIMS = (7.0, 5.0, 3.0)


def spec_with(walls, floor=None, ceiling=None):
    return pair(
        geometry=box_room(DIMS, walls, floor=floor, ceiling=ceiling),
        source=np.array([1.5, 1.5, 1.6]),
        receiver=np.array([5.0, 3.0, 1.5]),
        air_absorption=False,
    )


def uniform(alpha):
    return SurfaceMaterial("wall", np.full(BANDS.size, float(alpha)))


def treated_room():
    """Absorption on the floor and ceiling only, as in a carpeted office."""
    wall = np.array([0.10, 0.10, 0.08, 0.06, 0.05, 0.05, 0.05, 0.06])
    floor = np.array([0.05, 0.05, 0.10, 0.25, 0.45, 0.60, 0.65, 0.70])
    ceiling = np.array([0.20, 0.25, 0.45, 0.70, 0.80, 0.85, 0.85, 0.80])
    return dict(walls=SurfaceMaterial("wall", wall), floor=SurfaceMaterial("floor", floor),
                ceiling=SurfaceMaterial("ceiling", ceiling))


def test_a_uniformly_and_lightly_absorbing_room_warns_about_nothing():
    """The case both models are valid for must stay quiet, or the warning is noise."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", DiffuseFieldWarning)
        RoomTarget.from_room(spec_with(uniform(0.12)))


def test_absorption_concentrated_on_two_surfaces_warns():
    with pytest.warns(DiffuseFieldWarning, match="unevenly distributed"):
        RoomTarget.from_room(spec_with(**treated_room()))


def test_high_uniform_absorption_warns():
    with pytest.warns(DiffuseFieldWarning, match="too absorbing"):
        RoomTarget.from_room(spec_with(uniform(0.35)))


def test_the_uneven_warning_names_only_the_bands_it_applies_to():
    """A room can be uniform at 62.5 Hz and wildly uneven at 8 kHz.

    Porous absorbers do exactly this, so the check has to be per band or it
    either cries wolf across the whole spectrum or misses the top of it.
    """
    wall = uniform(0.10)
    # identical to the walls in the lowest band, far from them in the highest
    porous = SurfaceMaterial("porous", np.array([0.10, 0.12, 0.16, 0.25, 0.40, 0.55, 0.70, 0.80]))

    with pytest.warns(DiffuseFieldWarning, match="unevenly distributed") as record:
        RoomTarget.from_room(spec_with(wall, floor=porous, ceiling=porous))
    message = str(record[0].message)
    assert "8000" in message
    assert "62.5" not in message
    assert "125" not in message


def test_the_warning_points_at_from_rir():
    with pytest.warns(DiffuseFieldWarning, match="from_rir"):
        RoomTarget.from_room(spec_with(uniform(0.35)))


def test_sabine_is_warned_about_too_because_it_shares_the_assumption():
    with pytest.warns(DiffuseFieldWarning):
        RoomTarget.from_room(spec_with(uniform(0.35)), model="sabine")


def test_warning_does_not_change_the_returned_target():
    """The warning is a diagnostic, not a correction."""
    spec = spec_with(uniform(0.35))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DiffuseFieldWarning)
        warned = RoomTarget.from_room(spec)
        quiet = RoomTarget.from_room(spec_with(uniform(0.12)))
    assert warned.rt.shape == quiet.rt.shape
    # unchanged against the model called directly
    import pyrato

    from compas_acoustics.room.diffuse import mean_absorption

    expected = pyrato.parametric.reverberation_time_eyring(
        volume=spec.geometry.volume,
        surface_area=spec.geometry.surface_area,
        mean_absorption=mean_absorption(spec.geometry),
        speed_of_sound=spec.speed_of_sound,
    )
    np.testing.assert_allclose(warned.rt[:, 0], expected, rtol=1e-12)


def test_from_rir_never_warns_because_it_assumes_no_diffuse_field():
    fs = 48000.0
    t = np.arange(int(fs)) / fs
    rng = np.random.default_rng(0)
    decay = rng.standard_normal(t.size) * 10 ** (-3 * t / 0.6)
    with warnings.catch_warnings():
        warnings.simplefilter("error", DiffuseFieldWarning)
        RoomTarget.from_rir(decay, fs, n_slopes=1)
