"""Rooms that do not close, prepared with their air volume and openings: diffuse-field predictions and the hybrid."""

import warnings

import numpy as np
import pytest

import compas_acoustics as ca
from compas_acoustics.geometry import PrepareOptions
from compas_acoustics.room.diffuse import predicted_rt
from compas_acoustics.room.raytracing import default_duration

WALL = ca.AcousticMaterial("wall", 0.1, 0.1)


def _scene(open_top):
    scene = ca.Scene("open")
    objects = ca.AcousticObjectsLibrary.shoebox_room(8, 6, 3, materials=WALL)
    if open_top:
        objects = [o for o in objects if "ceiling" not in o.name]
    scene.add_acoustic_objects(objects)
    scene.add_source(ca.Source("s", 2, 2, 1.5))
    scene.add_receiver(ca.Receiver("r", 6, 4, 1.2))
    return scene


def _quiet(function, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return function(*args, **kwargs)


def test_a_closed_room_prepared_has_no_openings():
    geometry = ca.RoomGeometry.from_scene(_scene(False), prepare=PrepareOptions())
    drawn = _quiet(ca.RoomGeometry.from_scene, _scene(False))
    assert geometry.is_closed and geometry.has_volume and geometry.open_area == 0.0 and geometry.air_volume is None
    assert geometry.surface_area == pytest.approx(drawn.surface_area)
    np.testing.assert_allclose(predicted_rt(geometry, warn=False), predicted_rt(drawn, warn=False))


def test_an_open_room_takes_its_openings_as_absorption():
    closed = ca.RoomGeometry.from_scene(_scene(False), prepare=PrepareOptions())
    opened = ca.RoomGeometry.from_scene(_scene(True), prepare=PrepareOptions())
    assert not opened.is_closed and opened.has_volume and opened.open_area > 0.0
    # within the walk's diffuse-field estimates of the real 144 m^3 and 48 m^2
    assert 100.0 < opened.volume < 200.0 and 30.0 < opened.open_area < 100.0
    assert any(material.name == "open" for _, material in opened.surfaces)
    rt_open, rt_closed = predicted_rt(opened, warn=False), predicted_rt(closed, warn=False)
    assert np.all(rt_open < 0.5 * rt_closed)
    assert default_duration(opened, True) < default_duration(closed, True)


def test_an_unprepared_open_room_still_has_no_volume():
    geometry = _quiet(ca.RoomGeometry.from_scene, _scene(True))
    assert not geometry.has_volume
    with pytest.raises(ValueError, match="closed room"):
        predicted_rt(geometry)
    with pytest.raises(ValueError, match="prepared model"):
        _quiet(_scene(True).simulate, ca.Ambisonics(sh_order=1, image_source_order=1, tail="eyring"))


def test_the_hybrid_runs_in_a_prepared_open_room_and_says_so():
    scene = _scene(True)
    geometry = ca.RoomGeometry.from_scene(scene, prepare=PrepareOptions())
    with pytest.warns(UserWarning, match="does not close"):
        rirs = scene.simulate(ca.Ambisonics(sh_order=1, image_source_order=1, tail="eyring"), geometry=geometry)
    ir = next(iter(rirs.values()))
    assert ir.n_channels == 4 and np.all(np.isfinite(ir.ir))


def test_scan_directions_out_of_an_opening_absorb_all():
    from compas_acoustics.hybrid.directional import scan_directions
    from compas_acoustics.hybrid.directional import scanned_absorption
    from compas_acoustics.hybrid.pair import Pair

    geometry = ca.RoomGeometry.from_scene(_scene(True), prepare=PrepareOptions())
    pair = Pair(geometry=geometry, source=np.array([2.0, 2.0, 1.5]), receiver=np.array([6.0, 4.0, 1.2]), receiver_rotation=np.eye(3), fs=48000.0)
    directions = scan_directions()
    alpha = scanned_absorption(pair, directions)
    up = directions.cartesian[:, 2] / np.linalg.norm(directions.cartesian, axis=1)
    assert np.all(alpha[up > 0.9] > 0.9)  # looking up: the missing ceiling
    assert np.all(alpha[up < -0.9] < 0.2)  # looking down: the floor
