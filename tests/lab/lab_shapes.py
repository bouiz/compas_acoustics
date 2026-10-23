"""Shapes shared by the lab tests, as plain functions (``pythonpath = ["tests/lab"]`` in pyproject.toml)."""

import numpy as np
from compas.datastructures import Mesh


def flat_tile(size=0.1):
    """A flat square quad of side ``size`` at z = 0."""
    return Mesh.from_vertices_and_faces([[0, 0, 0], [size, 0, 0], [size, size, 0], [0, size, 0]], [[0, 1, 2, 3]])


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


def sine_tile(period, depth, n=16):
    """One period of ``z = depth/2·(1 + sin(2πx/period))`` as an n×1 quad strip, 0.05 m deep in y."""
    vertices = []
    for y in (0.0, 0.05):
        for i in range(n + 1):
            x = period * i / n
            vertices.append([x, y, depth / 2 * (1 + np.sin(2 * np.pi * x / period))])
    faces = [[i, i + 1, i + n + 2, i + n + 1] for i in range(n)]
    return Mesh.from_vertices_and_faces(vertices, faces)


def tile_with_hole():
    """A 3×3 quad grid, 0.1 m square, without its centre face."""
    return _grid(3, 3, 0.1 / 3, keep=lambda i, j: (i, j) != (1, 1))


def open_l_shaped_relief():
    """An L-shaped 4×4-cell quad patch (cells with i < 2 or j < 2) with a 0.02 m bump at its inner vertex (1, 1)."""
    return _grid(4, 4, 0.1, z=lambda i, j: 0.02 if (i, j) == (1, 1) else 0.0, keep=lambda i, j: i < 2 or j < 2)


def two_separate_patches():
    """Two quads that don't touch."""
    vertices = [[0, 0, 0], [0.1, 0, 0], [0.1, 0.1, 0], [0, 0.1, 0], [0.3, 0, 0], [0.4, 0, 0], [0.4, 0.1, 0], [0.3, 0.1, 0]]
    return Mesh.from_vertices_and_faces(vertices, [[0, 1, 2, 3], [4, 5, 6, 7]])


def footprint_area(mesh):
    """The summed xy-projected area of the faces whose normal has z > 0."""
    area = 0.0
    for face in mesh.faces():
        points = np.array(mesh.face_coordinates(face))
        cross = np.cross(points[1:-1] - points[0], points[2:] - points[0])
        z = cross[:, 2].sum() / 2
        if z > 0:
            area += z
    return area


def step_tile(size=0.1, height=0.02):
    """One period of a step relief: z = 0 for x < size/2, z = height above, a vertical riser up at size/2 and down at size."""
    h, w = size / 2, size
    vertices = [[0, 0, 0], [h, 0, 0], [h, w, 0], [0, w, 0], [h, 0, height], [w, 0, height], [w, w, height], [h, w, height], [w, 0, 0], [w, w, 0]]
    return Mesh.from_vertices_and_faces(vertices, [[0, 1, 2, 3], [1, 4, 7, 2], [4, 5, 6, 7], [5, 8, 9, 6]])


def bumped_box(size=1.0, thickness=0.01, bump=0.02):
    """A closed box ``size`` × ``size`` × ``thickness``, its top a 2×2 quad grid with the centre vertex raised by ``bump``."""
    s, t = size / 2, thickness
    top = [[x, y, t + (bump if (x, y) == (s, s) else 0.0)] for y in (0, s, size) for x in (0, s, size)]
    vertices = top + [[0, 0, 0], [size, 0, 0], [size, size, 0], [0, size, 0]]
    faces = [[0, 1, 4, 3], [1, 2, 5, 4], [3, 4, 7, 6], [4, 5, 8, 7], [12, 11, 10, 9]]
    faces += [[9, 10, 2, 1, 0], [10, 11, 8, 5, 2], [11, 12, 6, 7, 8], [12, 9, 0, 3, 6]]
    return Mesh.from_vertices_and_faces(vertices, faces)
