from types import SimpleNamespace

import compas
import numpy as np
import pytest
from compas.datastructures import Mesh
from compas.geometry import Box
from compas.geometry import Frame
from scipy.spatial import cKDTree

from compas_acoustics import AcousticMaterial
from compas_acoustics import AcousticObject
from compas_acoustics import AcousticObjectsLibrary
from compas_acoustics import AirConditions
from compas_acoustics import Receiver
from compas_acoustics import Scene
from compas_acoustics import Source
from compas_acoustics.room.geometry import RoomGeometry
from compas_acoustics.rays import SpecularPath
from compas_acoustics.rays import SpecularReflections

WALL = AcousticMaterial("w", 0.2, 0.1)


def path(order=1, s=0, r=0):
    return SpecularPath([[0, 0, 0], [3, 4, 0], [6, 0, 0]], order=order, walls=[2] * order,
                        image=[0, 8, 0], source_index=s, receiver_index=r)


def test_path_length_and_travel_time():
    p = path()
    assert p.length == pytest.approx(10.0)
    assert p.travel_time == pytest.approx(10.0 / AirConditions().speed_of_sound * 1000)


def test_filter_by_pair_and_exact_order():
    field = SpecularReflections([path(1, 0, 0), path(2, 0, 0), path(1, 1, 0), path(1, 0, 1)])
    assert len(field.filter(source=0, receiver_index=0)) == 2
    assert [p.order for p in field.filter(order=2)] == [2]
    assert len(field.filter(order=[1, 2], source=0, receiver_index=0)) == 2
    assert isinstance(field.filter(order=1), SpecularReflections)
    assert field.orders == [1, 2]


def test_json_round_trip_keeps_every_attribute():
    back = compas.json_loads(compas.json_dumps(SpecularReflections([path(2, 1, 0)])))
    p = back.paths[0]
    assert (p.order, p.walls, p.source_index, p.receiver_index) == (2, [2, 2], 1, 0)
    assert list(p.image) == [0, 8, 0]


def shoebox(source=(1.5, 1.5, 1.6), receiver=(5, 3, 1.5)):
    s = Scene("t")
    s.add_acoustic_objects(AcousticObjectsLibrary.shoebox_room(7, 5, 3, materials=WALL))
    s.add_source(Source("s", *source))
    s.add_receiver(Receiver("r", *receiver))
    return s


def l_shaped():
    # the reflex corner at (5, 3) hides the receiver from the source
    outline = [(0, 0), (8, 0), (8, 3), (5, 3), (5, 6), (0, 6)]
    n = len(outline)
    vertices = [[x, y, 0.0] for x, y in outline] + [[x, y, 3.0] for x, y in outline]
    faces = [list(range(n))[::-1], list(range(n, 2 * n))] + [[k, (k + 1) % n, (k + 1) % n + n, k + n] for k in range(n)]
    s = Scene("l")
    s.add_acoustic_object(AcousticObject(Mesh.from_vertices_and_faces(vertices, faces), WALL, name="l"))
    s.add_source(Source("s", 7.0, 1.0, 1.5))
    s.add_receiver(Receiver("r", 4.0, 5.5, 1.5))
    return s


def obstacle():
    # a cabinet in the air on the direct path
    s = shoebox()
    s.add_acoustic_object(AcousticObject(Mesh.from_shape(Box(1.0, 1.0, 1.0, Frame([3.2, 2.2, 1.5]))), WALL,
                                         name="cabinet", role="obstacle"))
    return s


ROOMS = {"shoebox": shoebox, "l_shaped": l_shaped, "obstacle": obstacle}


def pra_visible_images(scene, max_order, s=0, m=0):
    """pyroomacoustics' own visible images for one pair, duplicates removed, as an (n, 3) array."""
    pytest.importorskip("spharpy")  # hybrid.early needs it, Python >= 3.11 only
    from compas_acoustics.hybrid.early import _first_of_each_position

    geometry = RoomGeometry.from_scene(scene)
    room = geometry.to_pra(48000, max_order)
    for source in scene.sources:
        room.add_source(source.position)
    room.add_microphone_array(np.array([r.position for r in scene.receivers], dtype=float).T)
    room.image_source_model()
    visible = np.array(room.visibility[s][m], dtype=bool)
    images = np.asarray(room.sources[s].images, dtype=float)[:, visible]
    return images[:, _first_of_each_position(images, SimpleNamespace(geometry=geometry))].T


def test_shoebox_has_one_direct_and_six_first_order_paths():
    field = SpecularReflections.from_scene(shoebox(), max_order=1)
    assert [p.order for p in field].count(0) == 1
    assert [p.order for p in field].count(1) == 6


def test_first_order_hit_points_are_the_mirror_solution():
    # floor z=0: image (1.5, 1.5, -1.6); hit = intersection of image->receiver with z=0
    p = next(p for p in SpecularReflections.from_scene(shoebox(), 1).filter(order=1) if abs(p.points[1][2]) < 1e-6)
    t = 1.6 / (1.6 + 1.5)
    assert list(p.points[1]) == pytest.approx([1.5 + t * 3.5, 1.5 + t * 1.5, 0.0], abs=1e-4)
    assert list(p.image) == pytest.approx([1.5, 1.5, -1.6], abs=1e-5)


@pytest.mark.parametrize("room", ["shoebox", "l_shaped", "obstacle"])
def test_images_match_pyroomacoustics_visible_images(room):
    scene = ROOMS[room]()
    field = SpecularReflections.from_scene(scene, max_order=3)
    ours = np.array([list(p.image) for p in field.filter(source=0, receiver_index=0)])
    theirs = pra_visible_images(scene, 3)
    assert len(ours) == len(theirs)
    assert cKDTree(theirs).query(ours)[0].max() < 1e-3


@pytest.mark.parametrize("room", ["l_shaped", "obstacle"])
def test_the_direct_sound_is_hidden_behind_the_corner_and_the_cabinet(room):
    assert 0 not in SpecularReflections.from_scene(ROOMS[room](), max_order=1).orders
    assert len(SpecularReflections.from_scene(ROOMS[room](), max_order=0)) == 0


@pytest.mark.parametrize("room", ["shoebox", "l_shaped", "obstacle"])
def test_each_path_runs_from_the_source_to_the_receiver_as_long_as_its_image_is_far(room):
    scene = ROOMS[room]()
    for p in SpecularReflections.from_scene(scene, max_order=3):
        assert len(p.points) == p.order + 2 and len(p.walls) == p.order
        assert list(p.points[0]) == pytest.approx(scene.sources[0].position, abs=1e-6)
        assert list(p.points[-1]) == pytest.approx(scene.receivers[0].position, abs=1e-6)
        assert p.length == pytest.approx(p.image.distance_to_point(p.points[-1]), abs=1e-4)


def test_edge_paths_are_counted_once():
    field = SpecularReflections.from_scene(shoebox(source=(2, 2.5, 1.5), receiver=(5, 2.5, 1.5)), max_order=2)
    images = np.array([list(p.image) for p in field])
    assert len(np.unique(images.round(4), axis=0)) == len(images)


def test_every_pair_in_scene_order():
    scene = shoebox()
    scene.add_source(Source("s2", 6, 4, 1.2))
    scene.add_receiver(Receiver("r2", 2, 4, 1.4))
    field = SpecularReflections.from_scene(scene, max_order=1)
    for s in range(2):
        for r in range(2):
            pair = field.filter(source=s, receiver_index=r)
            assert len(pair) == 7
            assert list(pair.filter(order=0).paths[0].points[0]) == pytest.approx(scene.sources[s].position)
            assert list(pair.filter(order=0).paths[0].points[-1]) == pytest.approx(scene.receivers[r].position)


def test_a_receiver_outside_the_room_warns_and_has_no_paths():
    with pytest.warns(UserWarning, match="outside"):
        field = SpecularReflections.from_scene(shoebox(receiver=(9, 3, 1.5)), 1)
    assert len(field) == 0


def test_a_source_outside_the_room_warns_and_has_no_paths():
    with pytest.warns(UserWarning, match="outside"):
        field = SpecularReflections.from_scene(shoebox(source=(9, 3, 1.5)), 1)
    assert len(field) == 0


def test_travel_time_uses_the_package_air():
    scene = shoebox()
    p = SpecularReflections.from_scene(scene, 0).paths[0]
    distance = np.linalg.norm(np.subtract(scene.receivers[0].position, scene.sources[0].position))
    assert p.travel_time == pytest.approx(distance / AirConditions().speed_of_sound * 1000)


def test_filter_takes_numpy_integers():
    field = SpecularReflections([path(1), path(2)])
    assert [p.order for p in field.filter(order=np.int64(2))] == [2]


def test_the_viewer_is_not_available_in_rhino(monkeypatch):
    monkeypatch.setattr(compas, "is_rhino", lambda: True)
    with pytest.raises(RuntimeError):
        SpecularReflections([path()]).display_in_viewer(show=False)


def edge():
    # source and receiver on the room's mid-height line, so second-order paths run through edges
    return shoebox(source=(2, 2.5, 1.5), receiver=(5, 2.5, 1.5))


def test_data_stored_with_wall_names_and_a_speed_of_sound_loads():
    data = path().__data__
    data.update(wall_names=["floor"], sound_speed=343.98)
    loaded = SpecularPath.__from_data__(data)
    assert (loaded.order, loaded.walls, loaded.receiver_index) == (1, [2], 0)
    stored = compas.json_dumps(SpecularReflections([path()])).replace('"rays":', '"paths":')
    assert len(compas.json_loads(stored)) == 1


def test_a_negative_order_is_a_value_error():
    with pytest.raises(ValueError, match="max_order"):
        SpecularReflections.from_scene(shoebox(), max_order=-1)


def test_the_search_warns_when_it_would_be_slow(monkeypatch):
    import compas_acoustics.simulation_methods.base as base

    calls = []
    monkeypatch.setattr(base, "_warn_if_the_search_is_slow", lambda geometry, order, *args, **kwargs: calls.append((geometry, order)))
    SpecularReflections.from_scene(shoebox(), max_order=2)
    assert len(calls) == 1
    assert isinstance(calls[0][0], RoomGeometry) and len(calls[0][0].faces) == 6 and calls[0][1] == 2


def test_an_image_traced_on_a_second_attempt_is_not_counted_as_untraced(monkeypatch):
    import compas_acoustics.rays as specular

    trace_back = specular._trace_back
    calls = {}

    def first_attempt_fails(walls, receiver, image, *args):
        key = tuple(np.round(image, 3))
        calls[key] = calls.get(key, 0) + 1
        return None if calls[key] == 1 else trace_back(walls, receiver, image, *args)

    monkeypatch.setattr(specular, "_trace_back", first_attempt_fails)
    with pytest.warns(UserWarning, match="could not be traced") as record:
        field = SpecularReflections.from_scene(edge(), max_order=2)
    alone = sum(1 for count in calls.values() if count == 1)
    if alone == len(calls):
        # both orders across the edge are visible only when pyroomacoustics breaks that tie their way (macOS, not Linux)
        pytest.skip("no image position is visible twice on this platform")
    assert 0 < alone < len(calls) and len(field) == len(calls) - alone
    assert [str(w.message) for w in record] == [f"{alone} visible image sources could not be traced back to their source."]


def _inside_polygon(point, polygon, normal, tolerance):
    # on the plane: inside when every edge sees the point on its inner side, or the point is on the boundary
    signs = [np.dot(np.cross(b - a, point - a), normal) for a, b in zip(polygon, np.roll(polygon, -1, axis=0))]
    lengths = [np.linalg.norm(b - a) for a, b in zip(polygon, np.roll(polygon, -1, axis=0))]
    return all(s / length >= -tolerance for s, length in zip(signs, lengths))


def _unit(vector):
    return vector / np.linalg.norm(vector)


@pytest.mark.parametrize("room", ["shoebox", "l_shaped", "obstacle", "edge"])
def test_each_hit_is_on_its_wall_and_reflects_specularly(room):
    scene = dict(ROOMS, edge=edge)[room]()
    geometry = RoomGeometry.from_scene(scene)
    field = SpecularReflections.from_scene(scene, max_order=3)
    assert max(field.orders) == 3
    for p in field:
        points = np.array([list(q) for q in p.points])
        normals = []
        for k, w in enumerate(p.walls):
            polygon = geometry.vertices[list(geometry.faces[w])]
            normal = _unit(np.cross(polygon[1] - polygon[0], polygon[2] - polygon[0]))
            normals.append(normal)
            hit = points[k + 1]
            assert abs(np.dot(hit - polygon[0], normal)) < 1e-4
            # the L-shaped room's floor and ceiling are not convex: test against their fan triangles
            triangles = [polygon[[0, j, j + 1]] for j in range(1, len(polygon) - 1)]
            assert any(_inside_polygon(hit, t, normal, 1e-4) for t in triangles)
        # successive hits at one point (an edge) reflect the ray in each of their walls in turn
        k = 0
        while k < p.order:
            direction = _unit(points[k + 1] - points[k])
            j = k
            while True:
                direction = direction - 2 * np.dot(direction, normals[j]) * normals[j]
                j += 1
                if j == p.order or np.linalg.norm(points[j + 1] - points[k + 1]) > 1e-4:
                    break
            assert np.linalg.norm(direction - _unit(points[j + 1] - points[k + 1])) < 1e-4
            k = j


def test_specular_reflections_are_a_ray_field_of_paths_that_end_at_the_receiver():
    from compas_acoustics import Ray
    from compas_acoustics import RayField

    scene = shoebox()
    field = SpecularReflections.from_scene(scene, max_order=2)
    assert isinstance(field, RayField) and all(isinstance(path, Ray) for path in field)
    assert [path.ray_index for path in field] == list(range(len(field)))
    assert all(path.source is scene.sources[path.source_index] for path in field)
    assert all(path.reflection_order == path.order for path in field)
    assert field.duration_ms == pytest.approx(max(path.travel_time for path in field))
    points, indices, orders = field.particles(0.0)
    assert len(points) == len(field) and set(orders) == {0}
    # cut paths are plain rays: they stop short of the receiver
    assert all(type(ray) is Ray for ray in field.truncated(reflection_order=1))
    back = compas.json_loads(compas.json_dumps(field))
    assert isinstance(back, SpecularReflections) and back.paths[3].source.name == scene.sources[0].name
    assert [path.walls for path in back] == [path.walls for path in field]
