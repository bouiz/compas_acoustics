import numpy as np
import pytest
from compas.datastructures import Mesh

from compas_acoustics import AcousticMaterial
from compas_acoustics import AcousticObject
from compas_acoustics import AcousticObjectsLibrary
from compas_acoustics import Receiver
from compas_acoustics import Scene
from compas_acoustics import Source
from compas_acoustics.hybrid.early import image_sources
from compas_acoustics.hybrid.pair import Pair
from compas_acoustics.room.diffuse import air_attenuation
from compas_acoustics.room.pra import directivity_to_pra
from compas_acoustics.room.pra import receivers_to_pra_mic_array
from compas_acoustics.room.pra import source_to_pra
from compas_acoustics.room.geometry import RoomGeometry

FS = 48000
WALL = AcousticMaterial("wall", 0.2, 0.1)


def computed(sources, receivers, obstacles=(), order=2):
    scene = Scene("test")
    scene.add_acoustic_objects(AcousticObjectsLibrary.shoebox_room(7, 5, 3, materials=WALL))
    for obstacle in obstacles:
        scene.add_acoustic_object(obstacle)
    geometry = RoomGeometry.from_scene(scene)
    room = geometry.to_pra(FS, order)
    for source in sources:
        room.add_soundsource(source_to_pra(source))
    room.add_microphone_array(receivers_to_pra_mic_array(receivers, sampling_rate=FS))
    room.image_source_model()
    return geometry, room


def pair(geometry, source, receiver, air_absorption=False):
    return Pair(geometry, source.position, receiver.position, receiver.rotation, float(FS), 343.0, air_absorption)


def direct(iss):
    (index,) = np.flatnonzero(iss.order == 0)
    return index


def test_the_direct_sound_has_the_free_field_delay_and_gain():
    source, receiver = Source("s", 1.5, 1.5, 1.6), Receiver("r", 5, 3, 1.5)
    geometry, room = computed([source], [receiver])
    iss = image_sources(room, 0, 0, pair(geometry, source, receiver))
    d = np.linalg.norm(np.subtract(receiver.position, source.position))
    k = direct(iss)
    assert iss.delay[k] == pytest.approx(d / 343.0)
    np.testing.assert_allclose(iss.gain[k], 1.0 / (4.0 * np.pi * d))
    assert iss.order.max() == 2


def test_each_receiver_gets_only_the_images_it_sees():
    panel = AcousticObject(
        Mesh.from_vertices_and_faces([[3, 0.5, 0.5], [3, 0.5, 2.5], [3, 2.5, 2.5], [3, 2.5, 0.5]], [[0, 1, 2, 3]]),
        WALL, name="panel", role="obstacle",
    )
    source = Source("s", 1.5, 1.5, 1.6)
    hidden, seen = Receiver("hidden", 5, 1.5, 1.6), Receiver("seen", 5, 4.5, 1.5)
    geometry, room = computed([source], [hidden, seen], obstacles=[panel], order=1)
    behind = image_sources(room, 0, 0, pair(geometry, source, hidden))
    clear = image_sources(room, 0, 1, pair(geometry, source, seen))
    assert not np.any(behind.order == 0)
    assert np.sum(clear.order == 0) == 1


def test_a_shared_room_gives_each_pair_the_images_of_its_own_room():
    """Upstream builds one room per source and receiver; the per-pair visibility
    mask of a shared room must reproduce that, obstacles included."""
    panel = AcousticObject(
        Mesh.from_vertices_and_faces([[3, 0.5, 0.5], [3, 0.5, 2.5], [3, 2.5, 2.5], [3, 2.5, 0.5]], [[0, 1, 2, 3]]),
        WALL, name="panel", role="obstacle",
    )
    source = Source("s", 1.5, 1.5, 1.6)
    receivers = [Receiver("a", 5, 1.5, 1.6), Receiver("b", 5, 4.5, 1.5)]
    geometry, shared = computed([source], receivers, obstacles=[panel], order=2)
    for m, receiver in enumerate(receivers):
        _, alone = computed([source], [receiver], obstacles=[panel], order=2)
        together = image_sources(shared, 0, m, pair(geometry, source, receiver))
        single = image_sources(alone, 0, 0, pair(geometry, source, receiver))
        np.testing.assert_allclose(np.sort(together.delay), np.sort(single.delay))
        np.testing.assert_allclose(np.sort(together.gain.sum(axis=1)), np.sort(single.gain.sum(axis=1)))


def test_source_directivity_scales_each_image():
    omni = Source("s", 1.5, 1.5, 1.6)
    cardioid = Source("c", 1.5, 1.5, 1.6, directivity="cardioid", direction=[0, 1, 0])
    receiver = Receiver("r", 5, 1.5, 1.6)
    geometry, room = computed([omni], [receiver])
    spec = pair(geometry, omni, receiver)
    plain = image_sources(room, 0, 0, spec)
    shaped = image_sources(room, 0, 0, spec, directivity_to_pra(cardioid.directivity, cardioid.direction))
    k = direct(plain)
    np.testing.assert_allclose(shaped.gain[k], 0.5 * plain.gain[k])
    assert not np.allclose(shaped.gain, 0.5 * plain.gain)


def test_directions_of_arrival_are_in_the_receiver_frame():
    source = Source("s", 1.5, 1.5, 1.6)
    receiver = Receiver("r", 5, 1.5, 1.6, direction=[0, 1, 0])
    geometry, room = computed([source], [receiver])
    iss = image_sources(room, 0, 0, pair(geometry, source, receiver))
    # pyroomacoustics stores positions as float32
    np.testing.assert_allclose(iss.doa.cartesian[direct(iss)], [0.0, 3.5, 0.0], atol=1e-6)


def test_air_absorption_darkens_by_the_path_length():
    source, receiver = Source("s", 1.5, 1.5, 1.6), Receiver("r", 5, 3, 1.5)
    geometry, room = computed([source], [receiver])
    dry = image_sources(room, 0, 0, pair(geometry, source, receiver))
    wet = image_sources(room, 0, 0, pair(geometry, source, receiver, air_absorption=True))
    d = dry.delay * 343.0
    np.testing.assert_allclose(
        wet.gain / dry.gain, np.exp(-np.outer(d, air_attenuation()) / 2.0), rtol=1e-12
    )


def test_a_medium_sets_the_air_the_images_cross():
    from compas_acoustics import AirConditions
    from compas_acoustics.room.geometry import SYNTHESIS_BANDS

    source, receiver = Source("s", 1.5, 1.5, 1.6), Receiver("r", 5, 3, 1.5)
    geometry, room = computed([source], [receiver])
    air = AirConditions(temperature=20.0, humidity=10.0)
    dry = image_sources(room, 0, 0, pair(geometry, source, receiver))
    wet = image_sources(room, 0, 0, Pair(geometry, source.position, receiver.position, receiver.rotation,
                                         float(FS), 343.0, True, air))
    d = dry.delay * 343.0
    expected = np.exp(-np.outer(d, air.air_attenuation(SYNTHESIS_BANDS)) / 2.0)
    np.testing.assert_allclose(wet.gain / dry.gain, expected, rtol=1e-12)


def test_a_medium_without_air_absorption_takes_nothing():
    from compas_acoustics import AirConditions

    source, receiver = Source("s", 1.5, 1.5, 1.6), Receiver("r", 5, 3, 1.5)
    geometry, _ = computed([source], [receiver])
    spec = Pair(geometry, source.position, receiver.position, receiver.rotation, float(FS), 343.0, False, AirConditions())
    np.testing.assert_array_equal(spec.air_attenuation(), np.zeros(8))
