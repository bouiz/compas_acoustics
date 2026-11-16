"""One image source search read as another build's: the walls of every image and the float32 products along them."""

import numpy as np
import pyroomacoustics as pra
import pytest
from compas.datastructures import Mesh

from compas_acoustics import AcousticMaterial
from compas_acoustics import AcousticObject
from compas_acoustics import AcousticObjectsLibrary
from compas_acoustics import Receiver
from compas_acoustics import Scene
from compas_acoustics import Source
from compas_acoustics.hybrid.scattering import complementary_scattering
from compas_acoustics.room import reflections
from compas_acoustics.room.geometry import RoomGeometry
from compas_acoustics.room.geometry import pra_octave_grid
from compas_acoustics.room.pra import source_to_pra
from compas_acoustics.room.reflections import damping_as_searched

FS = 48000
SURFACES = ("floor", "right_wall", "front_wall", "left_wall", "back_wall", "ceiling")


def material(name, k):
    """Every surface its own absorption and scattering, so a wrong wall shows in the product."""
    return AcousticMaterial(name, list(np.round(np.linspace(0.04, 0.3, 7) * (1 + 0.15 * k), 4)), list(np.round(np.linspace(0.05, 0.5, 7) / (1 + 0.4 * k), 4)))


def scene(panel=True, height=None):
    """A shoebox with a material per surface and a panel obstacle; ``height`` puts every point at one height,
    whose paths graze the edges and are found from two wall orders."""
    s = Scene("reflections")
    s.add_acoustic_objects(AcousticObjectsLibrary.shoebox_room(9, 7, 3.5, materials={n: material(n, i) for i, n in enumerate(SURFACES)}))
    if panel:
        mesh = Mesh.from_vertices_and_faces([[5, 2.5, 0.4], [5, 4.5, 0.4], [5, 4.5, 2.2], [5, 2.5, 2.2]], [[0, 1, 2, 3]])
        s.add_acoustic_object(AcousticObject(mesh, material("panel", 6), name="panel", role="obstacle"))
    z = (1.5, 1.6, 1.2, 1.7) if height is None else (height,) * 4
    s.add_source(Source("a", 2.0, 3.5, z[0]))
    s.add_source(Source("b", 1.5, 5.5, z[1], directivity="cardioid", direction=[1, -0.3, 0]))
    s.add_receiver(Receiver("r1", 7.5, 3.4, z[2]))
    s.add_receiver(Receiver("r2", 6.5, 1.5, z[3]))
    return s


def built(geometry, s, order, ray_tracing):
    room = geometry.to_pra(FS, order, ray_tracing=ray_tracing)
    with pra_octave_grid():
        for source in s.sources:
            room.add_soundsource(source_to_pra(source))
        room.add_microphone_array(np.array([r.position for r in s.receivers], dtype=float).T)
    return room


def searched(geometry, s, order, ray_tracing):
    room = built(geometry, s, order, ray_tracing)
    with pra_octave_grid():
        room.image_source_model()
    return room


@pytest.mark.parametrize("panel, height", [(True, None), (False, 1.75)])
@pytest.mark.parametrize("order", [1, 2, 3])
def test_the_other_builds_damping_is_their_own_searchs_bit_for_bit(monkeypatch, panel, height, order):
    """The search for ray tracing (specular products) read as the plain room's and as the scattered parts' room."""
    s = scene(panel, height)
    geometry = RoomGeometry.from_scene(s)
    monkeypatch.setattr(reflections, "_searched", lambda room: pytest.fail("searched the other room"))
    room = searched(geometry, s, order, True)
    for other, ray_tracing in ((geometry, False), (complementary_scattering(geometry), True)):
        derived = damping_as_searched(room, built(other, s, order, ray_tracing))
        own = searched(other, s, order, ray_tracing)
        for d, source in zip(derived, own.sources):
            assert d.dtype == source.damping.dtype
            assert np.array_equal(d, source.damping)


def test_the_searchs_own_build_is_returned_unchanged():
    s = scene()
    geometry = RoomGeometry.from_scene(s)
    room = searched(geometry, s, 2, False)
    for d, source in zip(damping_as_searched(room, built(geometry, s, 2, False)), room.sources):
        assert np.array_equal(d, source.damping)


def test_a_path_whose_walls_are_not_singled_out_searches_the_other_room(monkeypatch):
    """At order 5 every point at one height sends paths through the corners of three walls, which the search
    accepts from two wall orders whose float32 products differ in the last bit."""
    s = scene(panel=False, height=1.75)
    geometry = RoomGeometry.from_scene(s)
    room = searched(geometry, s, 5, True)
    calls = []
    search = reflections._searched
    monkeypatch.setattr(reflections, "_searched", lambda other: calls.append(other) or search(other))
    derived = damping_as_searched(room, built(geometry, s, 5, False))
    own = searched(geometry, s, 5, False)
    assert len(calls) == 1
    for d, source in zip(derived, own.sources):
        assert np.array_equal(d, source.damping)


def test_a_source_no_receiver_sees_keeps_its_damping():
    """Order 0 behind the panel: the search exports no image of that source."""
    s = Scene("hidden")
    s.add_acoustic_objects(AcousticObjectsLibrary.shoebox_room(9, 7, 3.5, materials=material("wall", 0)))
    mesh = Mesh.from_vertices_and_faces([[5, 0, 0], [5, 7, 0], [5, 7, 3.5], [5, 0, 3.5]], [[0, 1, 2, 3]])
    s.add_acoustic_object(AcousticObject(mesh, material("panel", 3), name="panel", role="obstacle"))
    s.add_source(Source("a", 2.0, 3.5, 1.5))
    s.add_receiver(Receiver("r", 7.5, 3.4, 1.2))
    geometry = RoomGeometry.from_scene(s)
    room = searched(geometry, s, 0, True)
    (d,) = damping_as_searched(room, built(geometry, s, 0, False))
    assert np.array_equal(d, room.sources[0].damping)


def test_the_tracer_is_not_moved_by_reading_the_walls():
    """Reading the search draws no random numbers: the run after it is the same."""
    s = scene()
    geometry = RoomGeometry.from_scene(s)
    pra.random.seed(3)
    room = searched(geometry, s, 2, True)
    damping_as_searched(room, built(geometry, s, 2, False))
    after = pra.random.get_rng().random(4)
    pra.random.seed(3)
    searched(geometry, s, 2, True)
    built(geometry, s, 2, False)
    assert np.array_equal(after, pra.random.get_rng().random(4))
