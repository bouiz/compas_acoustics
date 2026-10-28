"""The early-to-late transition: the last rendered order drives the network, into the opposite line.

Wendt et al. (2014, Sec. 1.3): the network input "must equal the total energy of the Nth-order
reflections", N the maximum image source order the ISM renders. RAZR 0.96.3 feeds that order
(``image_source_model.m``: "sigmat of the last loop iteration will be used for FDN input"), and so do
Kirsch et al. (2023) and Ewert et al. (2025). Ewert et al. (Sec. 3.1.2) map each specular image to the
channels of the spatially opposite hemisphere, "including the FDN delays, thus mimicking sound having
travelled once through the room and being reflected on the opposite side".
"""

import numpy as np
import pytest

from compas_acoustics import AcousticMaterial
from compas_acoustics import AcousticObjectsLibrary
from compas_acoustics import Receiver
from compas_acoustics import Scene
from compas_acoustics import Source
from compas_acoustics.hybrid.bands import BANDS
from compas_acoustics.hybrid.early import image_sources
from compas_acoustics.hybrid.early import route_to_directions
from compas_acoustics.hybrid.late import fdn_directions
from compas_acoustics.hybrid.pair import Pair
from compas_acoustics.hybrid.synthesis import prepare
from compas_acoustics.hybrid.ambisonics import render_ambisonics
from compas_acoustics.hybrid.target import RoomTarget
from compas_acoustics.simulation_methods import Ambisonics
from compas_acoustics.room.geometry import RoomGeometry

FS = 48000
DIMS = (10.0, 8.0, 4.0)
WALL = AcousticMaterial("wall", [0.1, 0.12, 0.15, 0.18, 0.2, 0.22, 0.25], 0.1)
TARGET = RoomTarget(rt=np.full(8, 0.6), level=np.ones(8), bands=BANDS.copy(), fs=float(FS))
SOURCE, RECEIVER = (2.0, 4.0, 1.5), (7.12, 5.07, 1.61)


def pair(order):
    scene = Scene("t")
    scene.add_acoustic_objects(AcousticObjectsLibrary.shoebox_room(*DIMS, materials=WALL))
    geometry = RoomGeometry.from_scene(scene)
    room = geometry.to_pra(FS, order)
    room.add_source(list(SOURCE))
    room.add_microphone(np.reshape(RECEIVER, (3, 1)))
    room.image_source_model()
    spec = Pair(geometry, np.asarray(SOURCE), np.asarray(RECEIVER), np.eye(3), float(FS))
    return spec, image_sources(room, 0, 0, spec)


def test_the_last_rendered_order_drives_the_network():
    spec, iss = pair(2)
    result = render_ambisonics(prepare(spec, iss, TARGET, max_order=2, duration_s=0.3))
    assert result.core.driving_images == int(np.sum(iss.order == 2))
    np.testing.assert_array_equal(np.unique(result.core.images.order), [0, 1, 2])


def test_images_above_the_rendered_order_are_ignored():
    spec, deep = pair(3)
    _, shallow = pair(2)
    a = render_ambisonics(prepare(spec, deep, TARGET, max_order=2, duration_s=0.3))
    b = render_ambisonics(prepare(spec, shallow, TARGET, max_order=2, duration_s=0.3))
    np.testing.assert_allclose(a.rir, b.rir, rtol=0, atol=1e-12 * np.abs(b.rir).max())


def test_the_network_is_heard_one_line_after_the_last_order():
    spec, iss = pair(2)
    result = render_ambisonics(prepare(spec, iss, TARGET, max_order=2, duration_s=0.3))
    first = iss.delay[iss.order == 2].min() * FS + np.asarray(result.core.builds[0].delays).min()
    energy = result.late[:, 0] ** 2
    # before that only the zero-phase band magnitude's pre-ringing: about -34 dB here, each image entering
    # some 48 lines of its opposite hemisphere (-51 dB when it entered one); a drive heard at once would
    # put the whole last order there, at 0 dB
    assert energy[: int(first) - 48].sum() < 1e-3 * energy.sum()


def unit(xyz):
    xyz = np.asarray(xyz)
    return xyz / np.linalg.norm(xyz, axis=1, keepdims=True)


def test_each_image_enters_the_line_opposite_its_arrival():
    directions = fdn_directions()
    _, iss = pair(1)
    weights = route_to_directions(iss, directions, opposite=True)
    expected = np.argmax(-unit(iss.doa.cartesian) @ unit(directions.cartesian).T, axis=1)
    np.testing.assert_array_equal(np.argmax(weights, axis=1), expected)
    np.testing.assert_array_equal(weights.sum(axis=1), 1.0)


def test_routing_defaults_to_the_direction_of_arrival():
    directions = fdn_directions()
    _, iss = pair(1)
    expected = np.argmax(unit(iss.doa.cartesian) @ unit(directions.cartesian).T, axis=1)
    np.testing.assert_array_equal(np.argmax(route_to_directions(iss, directions), axis=1), expected)


def test_the_hybrid_searches_only_the_orders_it_renders(monkeypatch):
    orders = []
    original = RoomGeometry.to_pra

    def spy(self, fs, max_order, *args, **kwargs):
        orders.append(max_order)
        return original(self, fs, max_order, *args, **kwargs)

    scene = Scene("t")
    scene.add_acoustic_objects(AcousticObjectsLibrary.shoebox_room(*DIMS, materials=WALL))
    scene.add_source(Source("s", *SOURCE))
    scene.add_receiver(Receiver("r", *RECEIVER))
    monkeypatch.setattr(RoomGeometry, "to_pra", spy)
    hybrid = Ambisonics(tail="eyring", image_source_order=2, duration=0.3)
    scene.simulate(hybrid)
    assert orders == [2]
    (result,) = hybrid.results.values()
    assert result.core.driving_images == int(np.sum(result.core.images.order == 2))


def room_with_a_panel(receiver):
    from compas.datastructures import Mesh

    from compas_acoustics import AcousticObject

    scene = Scene("t")
    scene.add_acoustic_objects(AcousticObjectsLibrary.shoebox_room(*DIMS, materials=WALL))
    panel = Mesh.from_vertices_and_faces([[5, 3, 0.5], [5, 5, 0.5], [5, 5, 2.5], [5, 3, 2.5]], [[0, 1, 2, 3]])
    scene.add_acoustic_object(AcousticObject(panel, WALL, name="panel", role="obstacle"))
    scene.add_source(Source("s", *SOURCE))
    scene.add_receiver(Receiver("r", *receiver))
    return scene


def test_at_order_zero_the_direct_sound_drives_the_network():
    """Wendt et al. (2014, Sec. 1.3): N = 0 is a valid choice. With an obstacle in the room pyroomacoustics
    returns one band of damping for the direct sound alone, which is 1 in every band."""
    scene = room_with_a_panel((7.12, 6.5, 1.61))
    hybrid = Ambisonics(tail="eyring", image_source_order=0, duration=0.3, air_absorption=False)
    scene.simulate(hybrid)
    (result,) = hybrid.results.values()
    assert result.core.driving_images == 1
    reflection_product = result.core.images.gain[0] * 4 * np.pi * result.core.images.delay[0] * 343.0
    np.testing.assert_allclose(reflection_product, 1.0, rtol=0.01)


def test_nothing_drives_the_late_field_when_the_last_order_is_blocked():
    """A panel between source and receiver blocks the direct sound: at order 0 no image reaches the receiver."""
    scene = room_with_a_panel(RECEIVER)
    hybrid = Ambisonics(tail="eyring", image_source_order=0, duration=0.3)
    with pytest.warns(UserWarning, match="drives the late field"):
        scene.simulate(hybrid)
    (result,) = hybrid.results.values()
    assert result.core.driving_images == 0
    assert np.all(np.isfinite(result.rir)) and not np.any(result.late)
