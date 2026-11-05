"""Diffraction at the edges of the room's own boundary."""

import warnings
from itertools import pairwise

import numpy as np
import pytest

from _rooms import images, pair
from compas_acoustics.hybrid.synthesis import prepare
from compas_acoustics.hybrid.ambisonics import render_ambisonics
from compas_acoustics.hybrid.target import RoomTarget
from compas_acoustics.hybrid.diffraction import (
    TWO_PI,
    Edge,
    ShellEdgeWarning,
    along_plane,
    btms_paths,
    diffraction_edges,
    diffraction_paths,
    enclosure_edges,
    encode_btms,
    shared_face,
)
from compas_acoustics.room.geometry import RoomGeometry, SurfaceMaterial

C = 343.0
H = 3.0
TILT = np.linspace(0.8, 1.2, 8)
F = np.logspace(np.log10(20.0), np.log10(16000.0), 60)
L_OUTLINE = [(0, 0), (8, 0), (8, 3), (5, 3), (5, 6), (0, 6)]
PI_OUTLINE = [(0, 0), (4.5, 0), (4.5, 3.5), (5.5, 3.5), (5.5, 0), (10, 0), (10, 6), (0, 6)]


def materials(*names):
    # frequency-dependent on purpose: with every material flat, pyroomacoustics
    # returns a single band and image_sources refuses the mesh
    return {n: SurfaceMaterial(n, 0.1 * TILT) for n in names}


def prism_room(outline, height=H, rotation=None):
    """A room extruded from a counter-clockwise floor outline, with one floor and
    one ceiling polygon, which are non-convex for an L or a Pi."""
    n = len(outline)
    v = [[x, y, 0.0] for x, y in outline] + [[x, y, height] for x, y in outline]
    faces = [tuple(range(n))[::-1], tuple(range(n, 2 * n))]
    faces += [(k, (k + 1) % n, (k + 1) % n + n, k + n) for k in range(n)]
    v = np.array(v, float)
    if rotation is not None:
        v = v @ rotation.T
    return RoomGeometry.from_mesh(v, tuple(faces), tuple(["floor", "ceiling"] + ["wall"] * n), materials("floor", "ceiling", "wall"))


def grid_room(xs, ys, solid, height=H):
    """A room extruded from the cells of a rectilinear grid not in ``solid``: one
    floor and one ceiling tile per cell and one wall per cell side facing a solid
    cell or the outside, so floors and walls are full of coplanar seams."""
    index = {}

    def vid(x, y, z):
        return index.setdefault((x, y, z), len(index))

    nx, ny = len(xs) - 1, len(ys) - 1

    def air(i, j):
        return 0 <= i < nx and 0 <= j < ny and (i, j) not in solid

    faces, names = [], []
    for i in range(nx):
        for j in range(ny):
            if not air(i, j):
                continue
            x0, x1, y0, y1 = xs[i], xs[i + 1], ys[j], ys[j + 1]
            faces += [(vid(x0, y0, 0), vid(x0, y1, 0), vid(x1, y1, 0), vid(x1, y0, 0)), (vid(x1, y0, height), vid(x1, y1, height), vid(x0, y1, height), vid(x0, y0, height))]
            names += ["floor", "ceiling"]
            if not air(i + 1, j):
                faces.append((vid(x1, y0, 0), vid(x1, y1, 0), vid(x1, y1, height), vid(x1, y0, height)))
                names.append("wall")
            if not air(i - 1, j):
                faces.append((vid(x0, y1, 0), vid(x0, y0, 0), vid(x0, y0, height), vid(x0, y1, height)))
                names.append("wall")
            if not air(i, j + 1):
                faces.append((vid(x1, y1, 0), vid(x0, y1, 0), vid(x0, y1, height), vid(x1, y1, height)))
                names.append("wall")
            if not air(i, j - 1):
                faces.append((vid(x0, y0, 0), vid(x1, y0, 0), vid(x1, y0, height), vid(x0, y0, height)))
                names.append("wall")
    v = np.array(sorted(index, key=index.get), float)
    return RoomGeometry.from_mesh(v, tuple(faces), tuple(names), materials("floor", "ceiling", "wall"))


def vertical_at(edges, x, y):
    return [e for e in edges if np.allclose(e.start[:2], [x, y]) and np.allclose(e.end[:2], [x, y])]


def test_a_rectangular_room_has_no_room_edges():
    with warnings.catch_warnings():
        warnings.simplefilter("error", ShellEdgeWarning)
        assert enclosure_edges(prism_room([(0, 0), (8, 0), (8, 6), (0, 6)])) == []


@pytest.mark.parametrize("geometry", [prism_room(L_OUTLINE), grid_room([0, 5, 8], [0, 3, 6], {(1, 1)})], ids=["one-polygon floor", "tiled floor"])
def test_an_l_room_has_one_re_entrant_corner(geometry):
    edges = enclosure_edges(geometry)
    assert len(edges) == 1
    e = edges[0]
    assert len(vertical_at(edges, 5.0, 3.0)) == 1
    assert e.length == pytest.approx(H)
    assert e.theta_w == pytest.approx(1.5 * np.pi)
    assert len(e.regions) == 2 and e.regions[0] != e.regions[1]
    n_obstacle = len(geometry.obstacle_faces)
    assert all(f >= n_obstacle for f in e.faces)


def test_a_pi_room_has_exactly_its_two_stub_corners():
    edges = enclosure_edges(prism_room(PI_OUTLINE))
    assert len(edges) == 2
    assert len(vertical_at(edges, 4.5, 3.5)) == len(vertical_at(edges, 5.5, 3.5)) == 1
    assert all(e.theta_w == pytest.approx(1.5 * np.pi) for e in edges)


def t_junction_l(offset=0.0):
    """The L with its wall x = 5 split at mid-height: the corner's faces meet the
    edge in two pieces on one side and one on the other. ``offset`` moves the
    T-junction vertex off the corner line, along x."""
    n = len(L_OUTLINE)
    v = [[x, y, 0.0] for x, y in L_OUTLINE] + [[x, y, H] for x, y in L_OUTLINE]
    v += [[5.0 + offset, 3.0, H / 2], [5.0, 6.0, H / 2]]
    faces = [tuple(range(n))[::-1], tuple(range(n, 2 * n))]
    for k in range(n):
        a, b = k, (k + 1) % n
        if (L_OUTLINE[a], L_OUTLINE[b]) == ((5, 3), (5, 6)):
            faces += [(a, b, 2 * n + 1, 2 * n), (2 * n, 2 * n + 1, b + n, a + n)]
        else:
            faces.append((a, b, b + n, a + n))
    return RoomGeometry.from_mesh(np.array(v, float), tuple(faces), tuple(["floor", "ceiling"] + ["wall"] * (n + 1)), materials("floor", "ceiling", "wall"))


def test_a_corner_split_by_a_t_junction_is_one_edge():
    edges = enclosure_edges(t_junction_l())
    assert len(edges) == 1
    assert edges[0].length == pytest.approx(H)
    assert len(edges[0].faces) == 3


def test_a_t_junction_vertex_just_off_the_corner_keeps_one_edge():
    """5 um off the line: within the tolerance RoomGeometry.from_mesh splits sides at."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", ShellEdgeWarning)
        edges = enclosure_edges(t_junction_l(5e-6))
    assert len(edges) == 1
    assert edges[0].length == pytest.approx(H, abs=1e-4)


def banded_l(levels):
    """The L prism with every wall split into horizontal bands at ``levels``, so
    the corner is made of several pieces on both of its sides."""
    zs = [0.0, *levels, H]
    index = {}

    def vid(x, y, z):
        return index.setdefault((x, y, z), len(index))

    faces = [tuple(vid(x, y, 0.0) for x, y in L_OUTLINE)[::-1], tuple(vid(x, y, H) for x, y in L_OUTLINE)]
    for (x0, y0), (x1, y1) in zip(L_OUTLINE, L_OUTLINE[1:] + L_OUTLINE[:1]):
        faces += [(vid(x0, y0, z0), vid(x1, y1, z0), vid(x1, y1, z1), vid(x0, y0, z1)) for z0, z1 in pairwise(zs)]
    v = np.array(sorted(index, key=index.get), float)
    return RoomGeometry.from_mesh(v, tuple(faces), tuple(["floor", "ceiling"] + ["wall"] * (len(faces) - 2)), materials("floor", "ceiling", "wall"))


@pytest.mark.parametrize("seed", range(5))
def test_vertex_noise_on_a_corner_of_several_pieces_changes_nothing(seed):
    clean = banded_l([1.0, 2.0])
    noise = np.random.default_rng(seed).normal(0.0, 1e-6, clean.vertices.shape)
    noisy = RoomGeometry.from_mesh(clean.vertices + noise, clean.faces, clean.face_materials, clean.materials)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ShellEdgeWarning)
        edges = enclosure_edges(noisy)
    assert len(edges) == 1
    assert edges[0].length == pytest.approx(H, abs=1e-4)


def test_a_wall_tilted_by_a_microradian_next_to_a_long_floor_is_not_warned_about():
    """3 um off at the top of a 3 m wall: the corner is exact to the mesh
    tolerance, though the 60 m floor would turn the angle into 60 um."""
    v = np.array([[0, 0, 0], [60, 0, 0], [60, 6, 0], [0, 6, 0], [0, 0, H], [60 + 3e-6, 0, H], [60 + 3e-6, 6, H], [0, 6, H]], float)
    faces = ((0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7))
    geometry = RoomGeometry.from_mesh(v, faces, ("floor", "ceiling") + ("wall",) * 4, materials("floor", "ceiling", "wall"))
    with warnings.catch_warnings():
        warnings.simplefilter("error", ShellEdgeWarning)
        assert enclosure_edges(geometry) == []


def floating_panel_room():
    """An 8 x 6 m room with a 2 x 1 m panel of zero thickness hanging in it, both
    sides drawn as faces of the enclosure (a group not listed in ``obstacles``)."""
    box = prism_room([(0, 0), (8, 0), (8, 6), (0, 6)])
    n = len(box.vertices)
    v = np.vstack([box.vertices, [[3, 2, 1], [5, 2, 1], [5, 2, 2], [3, 2, 2]]])
    faces = box.faces + ((n, n + 1, n + 2, n + 3), (n + 3, n + 2, n + 1, n))
    return RoomGeometry.from_mesh(v, faces, box.face_materials + ("wall", "wall"), box.materials)


def test_a_panel_drawn_into_the_enclosure_gives_knife_edges():
    geometry = floating_panel_room()
    edges = enclosure_edges(geometry)
    assert len(edges) == 4
    assert sorted(round(e.length, 9) for e in edges) == [1.0, 1.0, 2.0, 2.0]
    for e in edges:
        assert e.theta_w == TWO_PI
        np.testing.assert_allclose(e.face_a, e.face_b)
        assert abs(e.face_a[1]) < 1e-12
    paths = diffraction_paths(spec_of(geometry, [4.0, 1.0, 1.5], [4.0, 3.0, 1.5]), max_order=1)
    assert any(p.order == 0 and p.zone == "shadow" for p in paths)


def test_obtuse_room_corners_are_left_out_with_a_warning():
    with pytest.warns(ShellEdgeWarning, match="2 room-boundary edge pieces"):
        edges = enclosure_edges(prism_room([(0, 0), (8, 0), (8, 4), (6, 6), (0, 6)]))
    assert edges == []


def test_a_room_box_has_four_corners():
    edges = enclosure_edges(grid_room([0, 4, 6, 10], [0, 3, 5, 8], {(1, 1)}))
    assert len(edges) == 4
    for x, y in ((4, 3), (6, 3), (6, 5), (4, 5)):
        assert len(vertical_at(edges, x, y)) == 1


def test_a_sliver_face_along_the_corner_keeps_the_corner():
    """Raw CAD triangulations leave thin triangles: here one 0.25 mm wide at
    mid-height runs along the L's corner, its apex splitting the ceiling side."""
    n = len(L_OUTLINE)
    v = [[x, y, 0.0] for x, y in L_OUTLINE] + [[x, y, H] for x, y in L_OUTLINE] + [[5.0, 3.0005, H]]
    faces = [tuple(range(n))[::-1], (6, 7, 8, 9, 12, 10, 11)]
    for k in range(n):
        a, b = k, (k + 1) % n
        if (L_OUTLINE[a], L_OUTLINE[b]) == ((5, 3), (5, 6)):
            faces += [(3, 12, 9), (3, 4, 10, 12)]
        else:
            faces.append((a, b, b + n, a + n))
    geometry = RoomGeometry.from_mesh(np.array(v, float), tuple(faces), tuple(["floor", "ceiling"] + ["wall"] * (n + 1)), materials("floor", "ceiling", "wall"))
    edges = enclosure_edges(geometry)
    assert len(edges) == 1
    assert edges[0].length == pytest.approx(H)
    assert edges[0].theta_w == pytest.approx(1.5 * np.pi)


def test_vertex_noise_below_the_mesh_tolerance_changes_nothing():
    clean = grid_room(list(range(9)), list(range(7)), {(i, j) for i in range(5, 8) for j in range(3, 6)})
    noise = np.random.default_rng(0).normal(0.0, 1e-6, clean.vertices.shape)
    noisy = RoomGeometry.from_mesh(clean.vertices + noise, clean.faces, clean.face_materials, clean.materials)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ShellEdgeWarning)
        edges = enclosure_edges(noisy)
    assert len(edges) == 1
    assert edges[0].theta_w == pytest.approx(1.5 * np.pi, abs=1e-4)


def test_two_corners_of_one_room_box_share_its_face():
    edges = enclosure_edges(grid_room([0, 4, 6, 10], [0, 3, 5, 8], {(1, 1)}))
    se, ne, sw = vertical_at(edges, 6, 3)[0], vertical_at(edges, 6, 5)[0], vertical_at(edges, 4, 3)[0]
    assert along_plane(se, ne) and along_plane(se, sw)
    assert not along_plane(sw, ne)


def test_coplanar_faces_of_two_room_boxes_are_not_one_surface():
    edges = enclosure_edges(grid_room([0, 2, 4, 6, 8, 10], [0, 3, 5, 8], {(1, 1), (3, 1)}))
    assert len(edges) == 8
    first_se, second_sw = vertical_at(edges, 4, 3)[0], vertical_at(edges, 6, 3)[0]
    first_ne, second_nw = vertical_at(edges, 4, 5)[0], vertical_at(edges, 6, 5)[0]
    assert not along_plane(first_se, second_sw)
    assert not along_plane(first_ne, second_nw)
    assert along_plane(first_se, vertical_at(edges, 2, 3)[0])


def test_a_room_edge_and_an_obstacle_edge_never_share_a_face():
    room = Edge(np.array([6.0, 3.0, 0.0]), np.array([6.0, 3.0, 3.0]), np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), 1.5 * np.pi, regions=(0, 1))
    panel = Edge(np.array([7.0, 3.0, 0.5]), np.array([7.0, 3.0, 2.5]), np.array([1.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]), 2 * np.pi, component=0)
    assert shared_face(room, panel) is None and shared_face(panel, room) is None


S_L = np.array([7.0, 1.0, 1.5])


def panel_l_room():
    """The L room as an 8 x 6 m box with the quadrant x > 5, y > 3
    closed off by two full-height obstacle panels."""
    v = [[0, 0, 0], [8, 0, 0], [8, 6, 0], [0, 6, 0], [0, 0, H], [8, 0, H], [8, 6, H], [0, 6, H]]
    faces = [(0, 3, 7, 4), (1, 5, 6, 2), (0, 4, 5, 1), (3, 2, 6, 7), (0, 1, 2, 3), (4, 7, 6, 5)]
    names = ["wall"] * 4 + ["floor", "ceiling"]
    for (x0, y0), (x1, y1) in (((5, 3), (5, 6)), ((5, 3), (8, 3))):
        n = len(v)
        v += [[x0, y0, 0], [x1, y1, 0], [x1, y1, H], [x0, y0, H]]
        faces.append((n, n + 1, n + 2, n + 3))
        names.append("panel")
    return RoomGeometry.from_mesh(np.array(v, float), tuple(faces), tuple(names), materials("wall", "floor", "ceiling", "panel"), obstacles=("panel",))


def box_obstacle_room(rotation=None):
    """A 10 x 8 m room with a full-height open box x 4-6, y 3-5 built from four
    obstacle panels: the room box of grid_room([0, 4, 6, 10], [0, 3, 5, 8], {(1, 1)})
    as an obstacle."""
    v = [[0, 0, 0], [10, 0, 0], [10, 8, 0], [0, 8, 0], [0, 0, H], [10, 0, H], [10, 8, H], [0, 8, H]]
    faces = [(0, 3, 7, 4), (1, 5, 6, 2), (0, 4, 5, 1), (3, 2, 6, 7), (0, 1, 2, 3), (4, 7, 6, 5)]
    names = ["wall"] * 4 + ["floor", "ceiling"]
    corners = [(4, 3), (6, 3), (6, 5), (4, 5)]
    for (x0, y0), (x1, y1) in zip(corners, corners[1:] + corners[:1]):
        n = len(v)
        v += [[x0, y0, 0], [x1, y1, 0], [x1, y1, H], [x0, y0, H]]
        faces.append((n, n + 1, n + 2, n + 3))
        names.append("box")
    v = np.array(v, float)
    if rotation is not None:
        v = v @ rotation.T
    return RoomGeometry.from_mesh(v, tuple(faces), tuple(names), materials("wall", "floor", "ceiling", "box"), obstacles=("box",))


def spec_of(geometry, source, receiver):
    return pair(geometry=geometry, source=np.asarray(source, float), receiver=np.asarray(receiver, float), air_absorption=False, speed_of_sound=C)


def test_paths_index_diffraction_edges():
    g = prism_room(L_OUTLINE)
    edges = diffraction_edges(g)
    assert len(edges) == 1
    assert [p.edge for p in diffraction_paths(spec_of(g, S_L, [4.0, 4.5, 1.5]), max_order=0)] == [0]


@pytest.mark.parametrize("receiver", [[4.0, 4.5, 1.5], [2.0, 4.5, 1.5], [4.0, 2.0, 2.5]], ids=["shadow", "view, upper leg", "view, lower leg"])
def test_a_room_corner_diffracts_as_the_same_corner_built_from_panels(receiver):
    shell = diffraction_paths(spec_of(prism_room(L_OUTLINE), S_L, receiver), max_order=0)
    panels = diffraction_paths(spec_of(panel_l_room(), S_L, receiver), max_order=0)
    assert len(shell) == len(panels) == 1
    a, b = shell[0], panels[0]
    assert a.zone == b.zone
    np.testing.assert_allclose([a.delay, a.distance], [b.delay, b.distance], rtol=1e-12)
    np.testing.assert_allclose(a.doa, b.doa, atol=1e-12)
    np.testing.assert_allclose(a.transfer(F), b.transfer(F), rtol=1e-9, atol=1e-12)


def test_btms_at_a_room_corner_equals_the_corner_built_from_panels():
    out = {
        name: encode_btms(btms_paths(spec_of(g, S_L, [4.0, 4.5, 1.5]), max_order=0), 0, 4800, 48000.0)
        for name, g in (("shell", prism_room(L_OUTLINE)), ("panels", panel_l_room()))
    }
    assert np.max(np.abs(out["panels"])) > 0.0
    np.testing.assert_allclose(out["shell"], out["panels"], atol=1e-9 * np.max(np.abs(out["panels"])))


def test_second_order_around_a_room_box_equals_the_box_built_from_panels():
    """The line from S to R crosses the box; no single corner sees both, so only
    second order over two corners along a face reaches R."""
    s, r = [2.0, 4.0, 1.5], [8.0, 4.0, 1.5]
    out, second = {}, {}
    for name, g in (("shell", grid_room([0, 4, 6, 10], [0, 3, 5, 8], {(1, 1)})), ("panels", box_obstacle_room())):
        paths = btms_paths(spec_of(g, s, r), max_order=0)
        second[name] = sum(p.order == 2 for p in paths)
        out[name] = encode_btms(paths, 0, 4800, 48000.0)
    assert second["shell"] == second["panels"] > 0
    np.testing.assert_allclose(out["shell"], out["panels"], atol=1e-9 * np.max(np.abs(out["panels"])))


def rotation_z(degrees):
    a = np.radians(degrees)
    return np.array([[np.cos(a), -np.sin(a), 0.0], [np.sin(a), np.cos(a), 0.0], [0.0, 0.0, 1.0]])


@pytest.mark.parametrize(
    ("source", "receiver"), [([3.0, 1.0, 1.5], [2.0, 5.0, 2.0]), ([3.0, 1.0, 1.5], [4.0, 4.5, 1.7]), (S_L, [4.0, 4.5, 1.7])], ids=["view", "reflection", "shadow"]
)
def test_a_rotated_room_diffracts_as_the_axis_aligned_room(source, receiver):
    """A room edge lies in the planes of its own faces, so an image in one of
    them reaches the edge with a zero-length wall leg; rounding must not turn
    that into a path in a room that is not axis-aligned."""
    rot = rotation_z(30.0)
    base = prism_room(L_OUTLINE)
    turned = RoomGeometry.from_mesh(base.vertices @ rot.T, base.faces, base.face_materials, base.materials)
    specs = (spec_of(base, source, receiver), spec_of(turned, rot @ np.asarray(source), rot @ np.asarray(receiver)))
    a, b = (sorted(diffraction_paths(s, max_order=1), key=lambda p: (round(p.distance, 9), p.order, p.zone)) for s in specs)
    assert len(a) == len(b) > 0
    assert [(p.order, p.zone, p.edge) for p in a] == [(p.order, p.zone, p.edge) for p in b]
    np.testing.assert_allclose([p.delay for p in b], [p.delay for p in a], rtol=1e-12)
    np.testing.assert_allclose([p.distance for p in b], [p.distance for p in a], rtol=1e-12)
    w = [encode_btms(btms_paths(s, max_order=1), 0, 4800, 48000.0)[:, 0] for s in specs]
    assert np.max(np.abs(w[0])) > 0.0
    np.testing.assert_allclose(w[1], w[0], atol=1e-9 * np.max(np.abs(w[0])))


def test_a_rectangular_room_has_no_room_edges_and_diffraction_changes_nothing():
    spec = spec_of(prism_room([(0, 0), (8, 0), (8, 6), (0, 6)]), [2.0, 2.0, 1.5], [6.0, 4.0, 1.7])
    off = render_ambisonics(prepare(spec, images(spec, 2), RoomTarget.from_room(spec), max_order=1, duration_s=0.2))
    for model in (True, "btms"):
        on = render_ambisonics(prepare(spec, images(spec, 2), RoomTarget.from_room(spec), max_order=1, duration_s=0.2, diffraction=model))
        np.testing.assert_array_equal(on.rir, off.rir)


def window_db(rir, r, fs):
    n0 = r / C * fs
    a, b = int(n0 - 0.3e-3 * fs), int(n0 + 0.6e-3 * fs)
    return 10 * np.log10(np.sum(rir[a:b, 0] ** 2))


def test_the_field_is_continuous_across_a_room_corner_shadow_boundary():
    # the line from S_L over the corner (5, 3) reaches y = 4.5 at x = 3.5
    levels = {}
    for flag in (False, True):
        for x in (3.495, 3.505):
            spec = spec_of(prism_room(L_OUTLINE), S_L, [x, 4.5, 1.5])
            rir = render_ambisonics(prepare(spec, images(spec, 2), RoomTarget.from_room(spec), max_order=1, duration_s=0.2, diffraction=flag)).early
            levels[flag, x] = window_db(rir, float(np.linalg.norm(spec.receiver - spec.source)), spec.fs)
    assert abs(levels[False, 3.505] - levels[False, 3.495]) > 6.0
    assert abs(levels[True, 3.505] - levels[True, 3.495]) < 0.5


def partition_room(as_obstacle):
    """An 8 x 6 x 3 m room with a 2 x 2 m partition of zero thickness standing on
    the floor at y = 2, x 3-5: drawn into the enclosure (both sides as room faces,
    the floor not split under it) or built as an obstacle panel."""
    outline = [(0, 0), (8, 0), (8, 6), (0, 6)]
    n = len(outline)
    v = [[x, y, 0.0] for x, y in outline] + [[x, y, H] for x, y in outline]
    v += [[3.0, 2.0, 0.0], [5.0, 2.0, 0.0], [5.0, 2.0, 2.0], [3.0, 2.0, 2.0]]
    faces = [tuple(range(n))[::-1], tuple(range(n, 2 * n))]
    faces += [(k, (k + 1) % n, (k + 1) % n + n, k + n) for k in range(n)]
    faces.append((8, 9, 10, 11))
    names = ["floor", "ceiling"] + ["wall"] * n + ["panel"]
    if as_obstacle:
        return RoomGeometry.from_mesh(np.array(v, float), tuple(faces), tuple(names), materials("floor", "ceiling", "wall", "panel"), obstacles=("panel",))
    faces.append((11, 10, 9, 8))
    names.append("panel")
    return RoomGeometry.from_mesh(np.array(v, float), tuple(faces), tuple(names), materials("floor", "ceiling", "wall", "panel"))


def test_a_partition_on_an_unsplit_floor_has_no_edge_along_the_floor():
    edges = enclosure_edges(partition_room(as_obstacle=False))
    assert len(edges) == 3
    assert all(e.theta_w == pytest.approx(2 * np.pi) for e in edges)
    assert not any(np.allclose([e.start[2], e.end[2]], 0.0) for e in edges)


@pytest.mark.parametrize("max_order", [0, 1])
def test_a_partition_in_the_room_shell_diffracts_as_the_same_partition_built_as_an_obstacle(max_order):
    s, r = [4.0, 1.0, 1.0], [4.0, 3.0, 1.2]

    def key(p):
        return (round(p.distance, 9), p.zone, tuple(np.round(p.doa, 9)))

    shell = sorted(diffraction_paths(spec_of(partition_room(False), s, r), max_order=max_order), key=key)
    panel = sorted(diffraction_paths(spec_of(partition_room(True), s, r), max_order=max_order), key=key)
    assert len(shell) == len(panel) > 0
    for a, b in zip(shell, panel):
        assert a.zone == b.zone
        np.testing.assert_allclose([a.delay, a.distance], [b.delay, b.distance], rtol=1e-12)
        np.testing.assert_allclose(a.doa, b.doa, atol=1e-12)
        np.testing.assert_allclose(a.band_gain, b.band_gain, rtol=1e-12)
        np.testing.assert_allclose(a.transfer(F), b.transfer(F), rtol=1e-9, atol=1e-12)
    w = {name: encode_btms(btms_paths(spec_of(partition_room(flag), s, r), max_order=max_order), 0, 4800, 48000.0) for name, flag in (("shell", False), ("panel", True))}
    np.testing.assert_allclose(w["shell"], w["panel"], atol=1e-9 * np.max(np.abs(w["panel"])))


def tilt(deg):
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


@pytest.mark.parametrize(
    "room, source, receiver",
    [
        ("l", [1.7, 2.61, 0.37], [1.37, 3.85, 1.85]),
        ("l", [1.7, 2.61, 0.37], [4.08, 5.25, 0.65]),
        ("l", [1.7, 2.61, 0.37], [2.81, 4.44, 1.03]),
        ("box", [2.0, 4.0, 1.5], [8.0, 4.0, 1.7]),
        ("box", [2.0, 4.0, 1.5], [8.0, 6.5, 0.6]),
    ],
    ids=["room corner 1", "room corner 2", "room corner 3", "obstacle box 1", "obstacle box 2"],
)
def test_a_tilted_room_diffracts_as_the_level_room(room, source, receiver):
    """Floor and ceiling tilted 20 deg about x: an apex clipped to an edge end on
    the floor must not gain a zero-length floor reflection (exact t = 1 when level).
    At these positions the tilted floor is hit at t = 1 - eps (most positions miss it)."""
    build = (lambda rot: prism_room(L_OUTLINE, rotation=rot)) if room == "l" else box_obstacle_room
    rot = tilt(20.0)
    level = spec_of(build(None), source, receiver)
    tilted = spec_of(build(rot), rot @ np.asarray(source), rot @ np.asarray(receiver))
    a = sorted(p.distance for p in diffraction_paths(level, max_order=1))
    b = sorted(p.distance for p in diffraction_paths(tilted, max_order=1))
    assert len(a) == len(b)
    np.testing.assert_allclose(a, b, rtol=1e-12)
    wa = encode_btms(btms_paths(level, max_order=1), 0, 4800, 48000.0)
    wb = encode_btms(btms_paths(tilted, max_order=1), 0, 4800, 48000.0)
    np.testing.assert_allclose(wa, wb, atol=1e-9 * np.max(np.abs(wa)))


def test_a_large_room_corner_with_a_t_junction_within_the_mesh_tolerance_is_one_edge():
    """A 40 x 30 m L: RoomGeometry.from_mesh accepts a T-junction up to 1e-6 x 40 m = 4e-5 m
    off a side; one 2e-5 m off the corner splits the wall x = 25 at mid-height."""
    outline = [(x * 5, y * 5) for x, y in L_OUTLINE]
    n = len(outline)
    v = [[x, y, 0.0] for x, y in outline] + [[x, y, H] for x, y in outline]
    v += [[25.0, 15.0 + 2e-5, H / 2], [25.0, 30.0, H / 2]]
    faces = [tuple(range(n))[::-1], tuple(range(n, 2 * n))]
    for k in range(n):
        a, b = k, (k + 1) % n
        if (outline[a], outline[b]) == ((25, 15), (25, 30)):
            faces += [(a, b, 2 * n + 1, 2 * n), (2 * n, 2 * n + 1, b + n, a + n)]
        else:
            faces.append((a, b, b + n, a + n))
    geometry = RoomGeometry.from_mesh(np.array(v, float), tuple(faces), tuple(["floor", "ceiling"] + ["wall"] * (n + 1)), materials("floor", "ceiling", "wall"))
    edges = enclosure_edges(geometry)
    assert len(edges) == 1
    assert edges[0].length == pytest.approx(H, abs=1e-4)
