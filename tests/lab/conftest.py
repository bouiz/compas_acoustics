"""Shapes for the scattering-sample tests, as fixtures that return the helper (the pattern of ``tests/conftest.py``)."""

import numpy as np
import pytest
from compas.datastructures import Mesh


def _grid(nx, ny, size, z=None, keep=None):
    """A quad grid of ``nx`` × ``ny`` cells of ``size``, heights ``z(i, j)``, cells ``keep(i, j)``."""
    vertices = [[i * size, j * size, z(i, j) if z else 0.0] for j in range(ny + 1) for i in range(nx + 1)]
    faces = []
    for j in range(ny):
        for i in range(nx):
            if keep is None or keep(i, j):
                a = j * (nx + 1) + i
                faces.append([a, a + 1, a + nx + 2, a + nx + 1])
    return Mesh.from_vertices_and_faces(vertices, faces)


def _sine_tile(period, depth, n=16):
    """One period of ``z = depth/2·(1 + sin(2πx/period))`` as an n×1 quad strip, 0.05 m deep in y."""
    vertices = []
    for y in (0.0, 0.05):
        for i in range(n + 1):
            x = period * i / n
            vertices.append([x, y, depth / 2 * (1 + np.sin(2 * np.pi * x / period))])
    faces = [[i, i + 1, i + n + 2, i + n + 1] for i in range(n)]
    return Mesh.from_vertices_and_faces(vertices, faces)


def _tile_with_hole():
    """A 3×3 quad grid, 0.1 m square, without its centre face."""
    return _grid(3, 3, 0.1 / 3, keep=lambda i, j: (i, j) != (1, 1))


def _open_l_shaped_relief():
    """An L-shaped 4×4-cell quad patch (cells with i < 2 or j < 2) with a 0.02 m bump at its inner vertex (1, 1)."""
    return _grid(4, 4, 0.1, z=lambda i, j: 0.02 if (i, j) == (1, 1) else 0.0, keep=lambda i, j: i < 2 or j < 2)


def _two_separate_patches():
    """Two quads that don't touch."""
    vertices = [[0, 0, 0], [0.1, 0, 0], [0.1, 0.1, 0], [0, 0.1, 0], [0.3, 0, 0], [0.4, 0, 0], [0.4, 0.1, 0], [0.3, 0.1, 0]]
    return Mesh.from_vertices_and_faces(vertices, [[0, 1, 2, 3], [4, 5, 6, 7]])


def _footprint_area(mesh):
    """The summed xy-projected area of the faces whose normal has z > 0."""
    area = 0.0
    for face in mesh.faces():
        points = np.array(mesh.face_coordinates(face))
        cross = np.cross(points[1:-1] - points[0], points[2:] - points[0])
        z = cross[:, 2].sum() / 2
        if z > 0:
            area += z
    return area


@pytest.fixture
def sine_tile():
    return _sine_tile


@pytest.fixture
def tile_with_hole():
    return _tile_with_hole


@pytest.fixture
def open_l_shaped_relief():
    return _open_l_shaped_relief


@pytest.fixture
def two_separate_patches():
    return _two_separate_patches


@pytest.fixture
def footprint_area():
    return _footprint_area
