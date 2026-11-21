"""Closed obstacles that are not convex, and the per-geometry edge cache."""

import warnings

import numpy as np
import pytest
from compas.datastructures import Mesh

pytest.importorskip("spharpy")  # compas_acoustics.hybrid.diffraction needs it

import compas_acoustics as ca  # noqa: E402
from compas_acoustics.hybrid.diffraction import ObstacleEdgeWarning  # noqa: E402
from compas_acoustics.hybrid.diffraction import diffraction_edges  # noqa: E402
from compas_acoustics.hybrid.diffraction import obstacle_edges  # noqa: E402

WALL = ca.AcousticMaterial("wall", 0.3, 0.1)


def _panel_scene():
    """A 2 m wide panel, as high as the room, at x = 4 between the source and the receiver."""
    scene = ca.Scene("shadow")
    scene.add_acoustic_objects(ca.AcousticObjectsLibrary.shoebox_room(8, 6, 3, materials=WALL))
    panel = Mesh.from_vertices_and_faces([[4, 2, 0.0], [4, 4, 0.0], [4, 4, 3.0], [4, 2, 3.0]], [[0, 1, 2, 3]])
    scene.add_acoustic_object(ca.AcousticObject(panel, ca.AcousticMaterial("panel", 0.2, 0.1), name="panel", role="obstacle"))
    scene.add_source(ca.Source("s", 2, 3, 1.5))
    scene.add_receiver(ca.Receiver("r", 6, 3.0, 1.5))
    return scene


def _l_prism(z0=0.5, z1=1.5):
    outline = [(3, 3), (5, 3), (5, 4), (4, 4), (4, 5), (3, 5)]  # re-entrant corner at (4, 4)
    n = len(outline)
    vertices = [[x, y, z0] for x, y in outline] + [[x, y, z1] for x, y in outline]
    faces = [list(range(n))[::-1], list(range(n, 2 * n))]
    faces += [[i, (i + 1) % n, (i + 1) % n + n, i + n] for i in range(n)]
    return Mesh.from_vertices_and_faces(vertices, faces)


def test_a_closed_obstacle_that_is_not_convex_diffracts_at_its_convex_edges():
    scene = ca.Scene("counter")
    scene.add_acoustic_objects(ca.AcousticObjectsLibrary.shoebox_room(8, 8, 3, materials=WALL))
    scene.add_acoustic_object(ca.AcousticObject(_l_prism(), ca.AcousticMaterial("counter", 0.1, 0.1), name="counter", role="obstacle"))
    with warnings.catch_warnings():
        warnings.simplefilter("error", ObstacleEdgeWarning)
        geometry = ca.RoomGeometry.from_scene(scene)
        edges = obstacle_edges(geometry)
    vertical = [e for e in edges if abs(e.axis[2]) > 0.99]
    corners = sorted((round(float(e.start[0]), 6), round(float(e.start[1]), 6)) for e in vertical)
    assert corners == [(3, 3), (3, 5), (4, 5), (5, 3), (5, 4)]  # not (4, 4)
    assert all(e.theta_w == pytest.approx(1.5 * np.pi) for e in edges)
    assert len(edges) == 5 + 12  # the outline's sides at the top and the bottom


def test_edges_are_kept_per_geometry():
    geometry = ca.RoomGeometry.from_scene(_panel_scene())
    first, second = diffraction_edges(geometry), diffraction_edges(geometry)
    assert first == second and first is not second
    assert len(first) > 0
