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
from compas_acoustics.room.geometry import SurfaceMaterial
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
    for source in s.sources:
        room.add_soundsource(source_to_pra(source))
    room.add_microphone_array(np.array([r.position for r in s.receivers], dtype=float).T)
    return room


def searched(geometry, s, order, ray_tracing):
    room = built(geometry, s, order, ray_tracing)
    room.image_source_model()
    return room


@pytest.mark.parametrize("panel, height", [(True, None), (False, 1.75)])
@pytest.mark.parametrize("order", [1, 2, 3])
def test_the_other_builds_damping_is_their_own_searchs_bit_for_bit(monkeypatch, panel, height, order):
    """The search for ray tracing (specular products) read as the plain room's and as the scattered parts' room."""
    s = scene(panel, height)
    geometry = RoomGeometry.from_scene(s)
    room = searched(geometry, s, order, True)
    for other, ray_tracing in ((geometry, False), (complementary_scattering(geometry), True)):
        unsearched = built(other, s, order, ray_tracing)
        monkeypatch.setattr(unsearched, "image_source_model", lambda: pytest.fail("searched the other room"))
        derived = damping_as_searched(room, unsearched)
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


def test_paths_through_the_corners_of_three_walls_are_read_too():
    """At order 5 every point at one height sends paths through corners, which the search accepts from several wall
    orders; its own tests tell which of them it exported."""
    s = scene(panel=False, height=1.75)
    geometry = RoomGeometry.from_scene(s)
    room = searched(geometry, s, 5, True)
    derived = damping_as_searched(room, built(geometry, s, 5, False))
    for d, source in zip(derived, searched(geometry, s, 5, False).sources):
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


# ------------------------------------------------ paths the search's float32 decides --
def panelled_box(x, names):
    """A 6 x 5 x 3 m box whose walls are split along the planes of ``x`` and y = 2.5 into coplanar panels;
    ``names(index, axis, low)`` gives each panel's material, by its number, its wall's axis and side. "swap_a" and
    "swap_b" have the same specular product and different plain ones (absorption and scattering exchanged)."""
    axes = [np.array([0.0, *x, 6.0]), np.array([0.0, 2.5, 5.0]), np.array([0.0, 3.0])]
    vertices, index, faces, materials = [], {}, [], []

    def vertex(p):
        if tuple(p) not in index:
            index[tuple(p)] = len(vertices)
            vertices.append(p)
        return index[tuple(p)]

    for axis in range(3):
        u, v = [a for a in range(3) if a != axis]
        for coordinate in (axes[axis][0], axes[axis][-1]):
            for i in range(len(axes[u]) - 1):
                for j in range(len(axes[v]) - 1):
                    quad = []
                    for a, b in ((i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)):
                        p = [0.0, 0.0, 0.0]
                        p[axis], p[u], p[v] = float(coordinate), float(axes[u][a]), float(axes[v][b])
                        quad.append(vertex(p))
                    materials.append(names(len(faces), axis, coordinate == 0.0))
                    faces.append(tuple(quad))
    flat = np.ones(8)
    pool = {
        "swap_a": SurfaceMaterial("swap_a", 0.15 * flat, 0.6 * flat),
        "swap_b": SurfaceMaterial("swap_b", 0.6 * flat, 0.15 * flat),
        "dull": SurfaceMaterial("dull", np.linspace(0.1, 0.4, 8), np.linspace(0.3, 0.05, 8)),
        "hard": SurfaceMaterial("hard", np.linspace(0.02, 0.06, 8), 0.0 * flat),
    }
    return RoomGeometry.from_mesh(np.array(vertices), tuple(faces), tuple(materials), {n: pool[n] for n in set(materials)})


def strip_box():
    """Every panel another material in turn, a 1 cm strip among them."""
    return panelled_box((3.0, 3.01), lambda index, axis, low: ("swap_a", "swap_b", "dull", "hard")[index % 4])


def seam_scene(offset=0.0, y=(1.7, 1.7)):
    """Source and receivers at one height, mirrored about the seams: reflection points on them, ``offset`` beside."""
    s = Scene("seams")
    s.add_source(Source("a", 2.2, y[0], 1.5))
    s.add_receiver(Receiver("r1", 3.8 + offset, y[1], 1.5))
    s.add_receiver(Receiver("r2", 2.2, 5.0 - y[0] + offset, 1.5))
    return s


@pytest.mark.parametrize("offset", [0.0, 1e-6, -1e-5, 2e-4, -5e-4])
@pytest.mark.parametrize("order", [2, 3])
def test_reflection_points_on_the_seams_of_coplanar_panels(offset, order):
    """Where a path meets the border of a panel, the search's float32 tests decide which panels reflect it, on
    the seam and within fractions of a millimetre of it; the damping read from it is still the other room's own."""
    geometry, s = strip_box(), seam_scene(offset)
    room = searched(geometry, s, order, True)
    for other, ray_tracing in ((geometry, False), (complementary_scattering(geometry), True)):
        derived = damping_as_searched(room, built(other, s, order, ray_tracing))
        (own,) = searched(other, s, order, ray_tracing).sources
        assert np.array_equal(derived[0], own.damping)


def test_the_sequences_kept_are_the_images_the_search_exported():
    """Of every sequence of walls ending on an image's last, the search's tests keep as many as it exported at that
    position off that wall, and the proposals traced back from the receiver hold them all. A kept sequence being one
    of the search's, a path the proposals missed could cost a search of the other room and not a wrong value."""
    from itertools import product

    geometry, s = strip_box(), seam_scene()
    room = searched(geometry, s, 3, True)
    walls = reflections._Walls(room.room_engine.walls, 6e-4)
    search = reflections._Search(room.room_engine)
    (source,) = room.sources
    visible = np.asarray(room.visibility[0], dtype=bool)
    images, own = np.asarray(source.images, dtype=np.float32), np.asarray(source.damping)
    orders, last = np.asarray(source.orders).ravel(), np.asarray(source.walls).ravel()
    position, mics = np.asarray(source.position, dtype=float).ravel(), np.asarray(room.mic_array.R)
    shared = 0
    for k in np.flatnonzero(orders > 0):
        m = int(np.flatnonzero(visible[:, k])[0])
        every = [w + (int(last[k]),) for w in product(range(len(room.walls)), repeat=int(orders[k]) - 1)]
        exported = {w for w in every if search.exported(w, position, mics[:, m], images[:, k], own[:, k])}
        proposed = reflections._wall_sequences(walls, images[:, k].astype(float), int(orders[k]), int(last[k]), mics[:, m], position)
        twins = visible[m] & (last == last[k]) & np.all(images == images[:, [k]], axis=0) & np.all(own == own[:, [k]], axis=0)
        assert len(exported) == np.count_nonzero(twins)
        assert exported <= set(proposed)
        shared += len(exported) > 1
    assert shared > 0


def searched_other(monkeypatch, room, other):
    """The damping read from ``room`` for ``other``, and a record of each search of ``other`` it made."""
    calls = []
    search = other.image_source_model
    monkeypatch.setattr(other, "image_source_model", lambda: calls.append(other) or search())
    return damping_as_searched(room, other), calls


@pytest.mark.filterwarnings("ignore:.*lie in one plane")
def test_two_images_of_one_damping_and_different_walls_search_the_other_room(monkeypatch):
    """A path off the floor's seam and then a wall is two images, one per floor panel, at one position and off one
    last wall. The panels' specular products are equal and their plain ones are not, so the search's damping does not
    tell which image is which."""
    geometry = panelled_box((3.0,), lambda index, axis, low: ("swap_a" if index % 2 else "swap_b") if axis == 2 and low else "dull")
    s = seam_scene(y=(2.5, 1.0))
    room = searched(geometry, s, 2, True)
    derived, calls = searched_other(monkeypatch, room, built(geometry, s, 2, False))
    (own,) = searched(geometry, s, 2, False).sources
    assert len(calls) == 1
    assert np.array_equal(derived[0], own.damping)
    images, specular = np.asarray(room.sources[0].images), np.asarray(room.sources[0].damping)
    pairs = [(j, k) for j in range(images.shape[1]) for k in range(j) if np.array_equal(images[:, j], images[:, k]) and np.array_equal(specular[:, j], specular[:, k])]
    assert any(not np.array_equal(own.damping[:, j], own.damping[:, k]) for j, k in pairs)


def test_an_image_without_a_receiver_that_sees_it_searches_the_other_room(monkeypatch):
    """pyroomacoustics clears the visibility of a microphone it finds outside the room: no path to read."""
    s = scene()
    geometry = RoomGeometry.from_scene(s)
    room = searched(geometry, s, 2, True)
    k = int(np.flatnonzero(np.asarray(room.sources[0].orders).ravel() == 2)[0])
    room.visibility[0][:, k] = 0
    derived, calls = searched_other(monkeypatch, room, built(geometry, s, 2, False))
    assert len(calls) == 1
    for d, source in zip(derived, searched(geometry, s, 2, False).sources):
        assert np.array_equal(d, source.damping)
