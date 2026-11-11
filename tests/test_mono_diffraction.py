"""Edge diffraction in the mono method, closed obstacles that are not convex, and the per-geometry edge cache."""

import warnings

import numpy as np
import pytest
from compas.datastructures import Mesh

import compas_acoustics as ca
from compas_acoustics.hybrid.diffraction import ObstacleEdgeWarning
from compas_acoustics.hybrid.diffraction import diffraction_edges
from compas_acoustics.hybrid.diffraction import obstacle_edges

FS = 48000
WALL = ca.AcousticMaterial("wall", 0.3, 0.1)


def _panel_scene(receiver_y):
    """A 2 m wide, 3 m tall-room-high panel at x = 4 between the source and a receiver behind it."""
    scene = ca.Scene("shadow")
    scene.add_acoustic_objects(ca.AcousticObjectsLibrary.shoebox_room(8, 6, 3, materials=WALL))
    panel = Mesh.from_vertices_and_faces([[4, 2, 0.0], [4, 4, 0.0], [4, 4, 3.0], [4, 2, 3.0]], [[0, 1, 2, 3]])
    scene.add_acoustic_object(ca.AcousticObject(panel, ca.AcousticMaterial("panel", 0.2, 0.1), name="panel", role="obstacle"))
    scene.add_source(ca.Source("s", 2, 3, 1.5))
    scene.add_receiver(ca.Receiver("r", 6, receiver_y, 1.5))
    return scene


def _direct_level(scene, diffraction, until=None):
    """Level in a window round the direct arrival (-0.3 to +0.6 ms), as the hybrid's continuity test reads it, or to a path length ``until``."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ir = next(iter(scene.simulate(ca.Mono(image_source_order=1, ray_tracing=False, diffraction=diffraction)).values())).denormalized()
    n0 = np.linalg.norm(np.subtract(scene.receivers[0].position, scene.sources[0].position)) / 343.98 * FS
    end = n0 + 0.6e-3 * FS if until is None else until / 343.98 * FS + 0.6e-3 * FS
    return 10 * np.log10(np.sum(ir[int(n0 - 0.3e-3 * FS) : int(end)] ** 2) + 1e-30)


def test_diffraction_fills_the_shadow_and_keeps_the_level_continuous():
    # the shadow boundary of the edge at (4, 2): the line from the source through it reaches y = 1 at x = 6
    shadow, lit = _panel_scene(1.005), _panel_scene(0.995)
    jump_without = _direct_level(lit, False) - _direct_level(shadow, False)
    jump_with = _direct_level(lit, True) - _direct_level(shadow, True)
    assert jump_without > 6.0
    assert abs(jump_with) < 0.5
    # deep in the shadow the sound round the panel's edges (4.47 m, before the floor's 5 m) is there
    deep = _panel_scene(3.0)
    assert _direct_level(deep, True, until=4.6) > _direct_level(deep, False, until=4.6) + 10.0


def test_mono_diffraction_is_opt_in_and_recorded():
    scene = _panel_scene(3.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plain = next(iter(scene.simulate(ca.Mono(image_source_order=1, ray_tracing=False)).values()))
        with_paths = next(iter(scene.simulate(ca.Mono(image_source_order=1, n_rays=2000, duration=0.5, diffraction=True)).values()))
    assert "diffraction" not in plain.generation and with_paths.generation["diffraction"] == "udfa"
    assert with_paths.ir.shape == (int(0.5 * FS),)
    with pytest.raises(ValueError, match="BTMS"):
        ca.Mono(diffraction="btms")


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
    geometry = ca.RoomGeometry.from_scene(_panel_scene(3.0))
    first, second = diffraction_edges(geometry), diffraction_edges(geometry)
    assert first == second and first is not second
    assert len(first) > 0
