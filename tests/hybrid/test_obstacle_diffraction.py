import numpy as np
import pytest
from compas.datastructures import Mesh
from compas.geometry import Box
from compas.geometry import Frame

from compas_acoustics import AcousticMaterial
from compas_acoustics import AcousticObject
from compas_acoustics import AcousticObjectsLibrary
from compas_acoustics import Scene
from compas_acoustics.hybrid.diffraction import diffraction_paths
from compas_acoustics.hybrid.diffraction import obstacle_edges
from compas_acoustics.hybrid.pair import Pair
from compas_acoustics.room_geometry import RoomGeometry

WALL = AcousticMaterial("wall", [0.05, 0.08, 0.1, 0.12, 0.14, 0.16, 0.18], 0.1)
SOURCE, RECEIVER = np.array([2.0, 4.0, 1.5]), np.array([8.0, 4.0, 1.5])
TOP = 2.5


def room(*obstacles):
    scene = Scene("test")
    scene.add_acoustic_objects(AcousticObjectsLibrary.shoebox_room(10, 8, 4, materials=WALL))
    for obstacle in obstacles:
        scene.add_acoustic_object(obstacle)
    return RoomGeometry.from_scene(scene)


def panel():
    mesh = Mesh.from_vertices_and_faces([[5, 3, 0.5], [5, 5, 0.5], [5, 5, TOP], [5, 3, TOP]], [[0, 1, 2, 3]])
    return AcousticObject(mesh, WALL, name="panel", role="obstacle")


def shelf():
    box = Box(4.0, 0.6, 1.523, Frame([3.0, 2.0, 1.523 / 2.0]))  # x 1..5, y 1.7..2.3, z 0..1.523
    return AcousticObject(Mesh.from_shape(box), WALL, name="shelf", role="obstacle")


def test_a_panel_has_four_knife_edges():
    edges = obstacle_edges(room(panel()))
    assert len(edges) == 4
    assert all(e.theta_w == pytest.approx(2 * np.pi) for e in edges)


def test_a_shelf_on_the_floor_has_four_top_and_four_vertical_edges():
    edges = obstacle_edges(room(shelf()))
    assert len(edges) == 8
    assert all(e.theta_w == pytest.approx(1.5 * np.pi) for e in edges)
    tops = [e for e in edges if np.allclose([e.start[2], e.end[2]], 1.523)]
    assert sorted(round(e.length, 6) for e in tops) == [0.6, 0.6, 4.0, 4.0]


def test_an_occluded_pair_diffracts_over_all_four_panel_edges():
    spec = Pair(room(panel()), SOURCE, RECEIVER, np.eye(3), 48000.0, 343.0, False)
    paths = [p for p in diffraction_paths(spec, max_order=0) if p.order == 0]
    assert len(paths) == 4
    assert all(p.zone == "shadow" for p in paths)
    apex = np.array([5.0, 4.0, TOP])
    (top,) = [p for p in paths if np.allclose(p.doa, apex - RECEIVER)]
    assert top.distance == pytest.approx(np.linalg.norm(SOURCE - apex) + np.linalg.norm(apex - RECEIVER))


def test_first_order_paths_reflect_off_the_enclosure():
    spec = Pair(room(panel()), SOURCE, RECEIVER, np.eye(3), 48000.0, 343.0, False)
    assert any(p.order == 1 for p in diffraction_paths(spec, max_order=1))


def test_the_doa_is_in_the_receiver_frame():
    g = room(panel())
    turned = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])  # facing +y
    plain = diffraction_paths(Pair(g, SOURCE, RECEIVER, np.eye(3), 48000.0, 343.0, False), max_order=0)
    rotated = diffraction_paths(Pair(g, SOURCE, RECEIVER, turned, 48000.0, 343.0, False), max_order=0)
    np.testing.assert_allclose([turned.T @ p.doa for p in plain], [p.doa for p in rotated], atol=1e-12)


def test_a_rectangular_room_without_obstacles_has_no_diffraction_paths():
    assert diffraction_paths(Pair(room(), SOURCE, RECEIVER, np.eye(3), 48000.0), max_order=0) == []
