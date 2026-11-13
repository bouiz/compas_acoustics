"""The sector detector wired into diffraction_edges: rooms of a prepared mesh take their edges from the walls' sides,
rooms built otherwise keep the legacy detectors, and a chosen subset of edges can be set per geometry."""

import warnings
from dataclasses import replace

import numpy as np
import pytest

pytest.importorskip("spharpy")  # compas_acoustics.hybrid.diffraction needs it

from compas.datastructures import Mesh  # noqa: E402

import compas_acoustics as ca  # noqa: E402
from compas_acoustics.geometry import PrepareOptions  # noqa: E402
from compas_acoustics.geometry import prepare_mesh  # noqa: E402
from compas_acoustics.hybrid.diffraction import TWO_PI  # noqa: E402
from compas_acoustics.hybrid.diffraction import AmbiguousEdgeWarning  # noqa: E402
from compas_acoustics.hybrid.diffraction import Occluders  # noqa: E402
from compas_acoustics.hybrid.diffraction import ShellEdgeWarning  # noqa: E402
from compas_acoustics.hybrid.diffraction import _inside_polygon  # noqa: E402
from compas_acoustics.hybrid.diffraction import _keep_selection  # noqa: E402
from compas_acoustics.hybrid.diffraction import clear_diffraction_edges  # noqa: E402
from compas_acoustics.hybrid.diffraction import diffraction_edges  # noqa: E402
from compas_acoustics.hybrid.diffraction import diffraction_paths  # noqa: E402
from compas_acoustics.hybrid.diffraction import enclosure_edges  # noqa: E402
from compas_acoustics.hybrid.diffraction import obstacle_edges  # noqa: E402
from compas_acoustics.hybrid.diffraction import set_diffraction_edges  # noqa: E402
from compas_acoustics.hybrid.pair import Pair  # noqa: E402
from compas_acoustics.room.geometry import DrawnSides  # noqa: E402
from compas_acoustics.room.geometry import RoomGeometry  # noqa: E402
from compas_acoustics.room.geometry import SurfaceMaterial  # noqa: E402
from compas_acoustics.room.geometry import _polygon_area_and_centroid  # noqa: E402

WALL = ca.AcousticMaterial("wall", 0.2, 0.1)
H = 3.0
# Box quads wound with outward normals, bottom first.
QUADS = [[0, 3, 2, 1], [4, 5, 6, 7], [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]]


class _Mesh:
    """A triangle soup by object, as prepare_mesh takes it."""

    def __init__(self):
        self.v, self.f, self.ranges = [], [], {}

    def add(self, name, vertices, quads):
        base, start = len(self.v), len(self.f)
        self.v += [[float(x) for x in p] for p in vertices]
        for q in quads:
            q = [base + i for i in q]
            self.f += [[q[0], q[k], q[k + 1]] for k in range(1, len(q) - 1)]
        self.ranges[name] = [start, len(self.f) - 1]
        return self

    def prepare(self, seeds, **options):
        return prepare_mesh(self.v, self.f, self.ranges, dict.fromkeys(range(len(self.f)), "wall"), None, seeds, PrepareOptions(**options))


def box(lo, hi):
    (x0, y0, z0), (x1, y1, z1) = lo, hi
    return [[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0], [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]], QUADS


def prism(outline, height=H, apex=0):
    """A room extruded from a counter-clockwise outline, wound outward; the floor and ceiling are fanned (by
    _Mesh.add) from outline[apex], which must see the whole outline."""
    n = len(outline)
    v = [[x, y, 0.0] for x, y in outline] + [[x, y, height] for x, y in outline]
    cap = [(apex + k) % n for k in range(n)]
    quads = [cap[:1] + cap[1:][::-1], [n + k for k in cap]] + [[k, (k + 1) % n, (k + 1) % n + n, k + n] for k in range(n)]
    return v, quads


def grid(xs, ys, solid, height=H, ceiling=True):
    """A room extruded from the grid cells not in ``solid``, one floor and ceiling tile per cell and a wall per cell
    side facing a solid cell or the outside, wound outward: a pillar in ``solid`` is fused into the shell."""
    index = {}

    def vid(x, y, z):
        return index.setdefault((x, y, z), len(index))

    nx, ny = len(xs) - 1, len(ys) - 1

    def air(i, j):
        return 0 <= i < nx and 0 <= j < ny and (i, j) not in solid

    quads = []
    for i in range(nx):
        for j in range(ny):
            if not air(i, j):
                continue
            x0, x1, y0, y1 = xs[i], xs[i + 1], ys[j], ys[j + 1]
            quads.append([vid(x0, y0, 0), vid(x0, y1, 0), vid(x1, y1, 0), vid(x1, y0, 0)])
            if ceiling:
                quads.append([vid(x1, y0, height), vid(x1, y1, height), vid(x0, y1, height), vid(x0, y0, height)])
            if not air(i + 1, j):
                quads.append([vid(x1, y0, 0), vid(x1, y1, 0), vid(x1, y1, height), vid(x1, y0, height)])
            if not air(i - 1, j):
                quads.append([vid(x0, y1, 0), vid(x0, y0, 0), vid(x0, y0, height), vid(x0, y1, height)])
            if not air(i, j + 1):
                quads.append([vid(x1, y1, 0), vid(x0, y1, 0), vid(x0, y1, height), vid(x1, y1, height)])
            if not air(i, j - 1):
                quads.append([vid(x0, y0, 0), vid(x1, y0, 0), vid(x1, y0, height), vid(x0, y0, height)])
    return sorted(index, key=index.get), quads


def room(prepared, **options):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return RoomGeometry.from_prepared(prepared, {"wall": WALL}, **options)


def edges_of(geometry):
    """The edges and the edge warnings diffraction_edges gave."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        edges = diffraction_edges(geometry)
    return edges, [w for w in caught if issubclass(w.category, (ShellEdgeWarning, AmbiguousEdgeWarning))]


def vertical_at(edges, x, y):
    return [e for e in edges if np.allclose(e.start[:2], [x, y]) and np.allclose(e.end[:2], [x, y])]


SHOEBOX_SEEDS = [[2.0, 2.0, 1.5], [8.0, 6.0, 1.5], [7.0, 2.0, 2.0]]


def shoebox(*objects):
    mesh = _Mesh().add("room", *box((0, 0, 0), (10, 8, 4)))
    for name, (v, quads) in objects:
        mesh.add(name, v, quads)
    return mesh


PILLAR_CORNERS = ((4, 3), (6, 3), (6, 5), (4, 5))
PILLAR_SEEDS = [[2.0, 1.5, 1.5], [8.0, 6.5, 1.5], [8.0, 1.5, 1.2]]


def open_pillar_room():
    """A 10 x 8 x 3 m room without its ceiling, a 2 x 2 m pillar fused into its shell (its walls welded to the floor)."""
    return _Mesh().add("room", *grid([0, 4, 6, 10], [0, 3, 5, 8], {(1, 1)}, ceiling=False)).prepare(PILLAR_SEEDS)


# ---------------------------------------------------------------------------------------------------------------
# Rooms of a prepared mesh
# ---------------------------------------------------------------------------------------------------------------


def test_a_prepared_closed_shoebox_has_no_edges():
    geometry = room(shoebox().prepare(SHOEBOX_SEEDS))
    assert geometry.drawn_sides is not None and geometry.is_closed
    edges, caught = edges_of(geometry)
    assert edges == [] and not caught


def test_a_prepared_closed_l_room_has_its_re_entrant_edge():
    v, quads = prism([(0, 0), (8, 0), (8, 3), (5, 3), (5, 6), (0, 6)], apex=3)
    geometry = room(_Mesh().add("room", v, quads).prepare([[2.0, 1.5, 1.5], [7.0, 1.5, 1.5], [2.0, 5.0, 1.5]]))
    edges, caught = edges_of(geometry)
    (e,) = edges
    assert vertical_at(edges, 5.0, 3.0) == [e] and not caught
    assert e.theta_w == pytest.approx(1.5 * np.pi) and e.length == pytest.approx(H)
    assert e.component == -1 and len(set(e.regions)) == 2


def test_a_prepared_open_room_diffracts_at_the_pillar_fused_into_its_shell():
    prepared = open_pillar_room()
    geometry = room(prepared)
    assert not geometry.is_closed
    edges, caught = edges_of(geometry)
    assert not caught  # no "boundary does not close" ShellEdgeWarning, nothing ambiguous
    assert len(edges) == 4
    for x, y in PILLAR_CORNERS:
        (e,) = vertical_at(edges, x, y)
        assert e.theta_w == pytest.approx(1.5 * np.pi) and e.length == pytest.approx(H)
        assert e.component == -1 and len(set(e.regions)) == 2
    # the role-based detectors find nothing there: the boundary does not close
    legacy = replace(geometry, drawn_sides=None)
    with pytest.warns(ShellEdgeWarning, match="does not close"):
        assert diffraction_edges(legacy) == []


def test_a_prepared_room_of_six_closed_slabs_has_no_edges():
    """Floor, ceiling and four walls, each a closed 0.2 m slab: all of them one-sided boundary (no obstacle), their
    outer skins never reached and their inner corners open pi / 2. Nothing escapes, so the room is closed; it does
    not enclose the air (the slabs' divergence volume is theirs), so its volume is the walk's."""
    t = 0.2
    mesh = _Mesh()
    for name, lo, hi in (
        ("floor", (-t, -t, -t), (6 + t, 4 + t, 0.0)),
        ("ceiling", (-t, -t, 3.0), (6 + t, 4 + t, 3 + t)),
        ("west", (-t, -t, 0.0), (0.0, 4 + t, 3.0)),
        ("east", (6.0, -t, 0.0), (6 + t, 4 + t, 3.0)),
        ("south", (0.0, -t, 0.0), (6.0, 0.0, 3.0)),
        ("north", (0.0, 4.0, 0.0), (6.0, 4 + t, 3.0)),
    ):
        mesh.add(name, *box(lo, hi))
    geometry = room(mesh.prepare([[1.0, 1.0, 1.5], [4.5, 3.0, 1.2], [3.0, 2.0, 2.5]]))
    assert geometry.is_closed and not geometry.encloses and "obstacle" not in geometry.face_roles
    assert geometry.air_volume is not None and geometry.open_area == 0.0
    edges, caught = edges_of(geometry)
    assert edges == [] and not caught


def test_a_free_standing_two_sided_panel_gives_its_knife_edges():
    panel = [[5, 3, 0.5], [5, 5, 0.5], [5, 5, 2.5], [5, 3, 2.5]]
    prepared = shoebox(("panel", (panel, [[0, 1, 2, 3]]))).prepare(SHOEBOX_SEEDS + [[3.0, 4.0, 1.5], [7.0, 4.0, 1.5]])
    assert prepared.two_sided[prepared.face_object == prepared.objects.index("panel")].all()
    edges, caught = edges_of(room(prepared))
    assert len(edges) == 4 and not caught
    assert all(e.theta_w == TWO_PI and e.component == 0 for e in edges)
    assert sorted(round(e.length, 6) for e in edges) == [2.0, 2.0, 2.0, 2.0]


def test_a_closed_box_standing_on_the_floor_gives_its_vertical_and_top_edges():
    prepared = shoebox(("cabinet", box((4.0, 3.0, 0.0), (5.0, 4.5, 1.5)))).prepare(SHOEBOX_SEEDS)
    edges, caught = edges_of(room(prepared))
    assert len(edges) == 8 and not caught
    # one-sided, so boundary: its edges are boundary edges (component -1, with regions)
    assert all(e.theta_w == pytest.approx(1.5 * np.pi) and e.component == -1 and len(set(e.regions)) == 2 for e in edges)
    assert sum(abs(e.axis[2]) > 0.99 for e in edges) == 4
    assert sum(np.allclose([e.start[2], e.end[2]], 1.5) for e in edges) == 4
    assert not any(np.allclose([e.start[2], e.end[2]], 0.0) for e in edges)  # none on the floor contact


def test_a_column_from_floor_to_ceiling_gives_its_four_vertical_edges():
    prepared = shoebox(("column", box((4.0, 3.0, 0.0), (5.0, 4.0, 4.0)))).prepare(SHOEBOX_SEEDS)
    edges, caught = edges_of(room(prepared))
    assert len(edges) == 4 and not caught
    assert all(abs(e.axis[2]) > 0.99 and e.length == pytest.approx(4.0) and e.component == -1 for e in edges)


def test_a_standing_solid_declared_an_obstacle_is_one_sided():
    """from_scene with prepare makes role='obstacle' surfaces two-sided; a closed one standing in the room still has
    its air outside only: wedges, not knife edges."""
    scene = ca.Scene("cabinet")
    scene.add_acoustic_objects(ca.AcousticObjectsLibrary.shoebox_room(10, 8, 4, materials=WALL))
    v, quads = box((4.0, 3.0, 0.0), (5.0, 4.5, 1.5))
    scene.add_acoustic_object(ca.AcousticObject(Mesh.from_vertices_and_faces(v, quads), WALL, name="cabinet", role="obstacle"))
    for k, p in enumerate(SHOEBOX_SEEDS):
        (scene.add_source if k == 0 else scene.add_receiver)((ca.Source if k == 0 else ca.Receiver)(f"p{k}", *p))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        geometry = RoomGeometry.from_scene(scene, prepare=PrepareOptions())
    assert geometry.drawn_sides is not None
    n_obstacle = len(geometry.obstacle_faces)
    assert n_obstacle and not geometry.drawn_sides.two_sided[:n_obstacle].any()
    edges, caught = edges_of(geometry)
    assert len(edges) == 8 and not caught
    assert all(e.theta_w == pytest.approx(1.5 * np.pi) for e in edges)


def test_the_thresholds_are_options_of_from_prepared():
    prepared = shoebox(("cabinet", box((4.0, 3.0, 0.0), (5.0, 4.5, 1.5)))).prepare(SHOEBOX_SEEDS)
    assert len(diffraction_edges(room(prepared, diffraction_min_length=1.2))) == 6  # the 1.5 m sides: 4 vertical, 2 on top
    assert diffraction_edges(room(prepared, diffraction_min_excess=np.pi / 2 + 0.1)) == []


def _on_polygon(x, polygon, tol):
    _, normal, _ = _polygon_area_and_centroid(polygon)
    if abs((x - polygon[0]) @ normal) > tol:
        return False
    if _inside_polygon(polygon, normal, x):
        return True
    for p, q in zip(polygon, np.roll(polygon, -1, axis=0)):
        d = q - p
        t = float(np.clip((x - p) @ d / (d @ d), 0.0, 1.0))
        if np.linalg.norm(x - p - t * d) <= tol:
            return True
    return False


@pytest.mark.parametrize(
    "build",
    [
        lambda: room(open_pillar_room()),
        lambda: room(shoebox(("cabinet", box((4.0, 3.0, 0.0), (5.0, 4.5, 1.5)))).prepare(SHOEBOX_SEEDS)),
        lambda: room(shoebox(("panel", ([[5, 3, 0.5], [5, 5, 0.5], [5, 5, 2.5], [5, 3, 2.5]], [[0, 1, 2, 3]]))).prepare(SHOEBOX_SEEDS + [[3.0, 4.0, 1.5], [7.0, 4.0, 1.5]])),
    ],
    ids=["open pillar", "cabinet", "panel"],
)
def test_edge_faces_index_the_occluder_polygons_that_hold_the_edge(build):
    geometry = build()
    occ = Occluders.from_geometry(geometry)
    assert len(geometry.drawn_sides.two_sided) == len(occ.polygons)
    edges = diffraction_edges(geometry)
    assert edges
    for e in edges:
        assert e.faces and all(0 <= k < len(occ.polygons) for k in e.faces)
        middle = (e.start + e.end) / 2.0
        assert all(_on_polygon(middle, occ.polygons[k], 1e-6) for k in e.faces)
        # each face direction runs into one of the edge's polygons
        for u in (e.face_a, e.face_b):
            assert any(_on_polygon(middle + 0.01 * u, occ.polygons[k], 1e-6) for k in e.faces)
        assert (e.component >= 0) == any(k < occ.n_obstacle for k in e.faces)


def test_drawn_sides_follow_the_faces_through_collapse_and_triangulation():
    """Polygon k is tagged by its normal (k + 1, 0, 0). A collapsed face is dropped, a warped one becomes two
    triangles; the sides come out in the room's drawn order, obstacles first."""
    polygons = [
        np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], float),  # 0 boundary
        np.array([[0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], float),  # 1 obstacle
        np.array([[2, 0, 0], [2, 0, 1e-9], [2, 1e-9, 0]], float),  # 2 boundary, collapses
        np.array([[0, 0, 2], [1, 0, 2], [1, 1, 2.005], [0, 1, 2]], float),  # 3 boundary, warped by 5 mm
        np.array([[0, 0, 3], [1, 0, 3], [1, 1, 3]], float),  # 4 obstacle
    ]
    roles = ["boundary", "obstacle", "boundary", "boundary", "obstacle"]
    groups = [(role, f"p{k}", "m") for k, role in enumerate(roles)]
    n = len(polygons)
    sides = DrawnSides(np.array([[k + 1.0, 0.0, 0.0] for k in range(n)]), np.arange(n) % 2 == 1, np.ones(n, bool))
    material = {"m": SurfaceMaterial("m", np.full(8, 0.1))}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        geometry = RoomGeometry._from_polygons(polygons, groups, material, False, 1e-6, trust_orientation=True, closed=False, sides=sides)
    assert geometry.drawn_sides.normals[:, 0].tolist() == [2.0, 5.0, 1.0, 4.0, 4.0]
    assert geometry.drawn_sides.two_sided.tolist() == [True, False, False, True, True]
    assert len(Occluders.from_geometry(geometry).polygons) == 5
    with pytest.raises(ValueError, match="merge_coplanar"):
        RoomGeometry._from_polygons(polygons, groups, material, True, 1e-6, sides=sides)


# ---------------------------------------------------------------------------------------------------------------
# End to end, and the edge selection
# ---------------------------------------------------------------------------------------------------------------


def test_mono_diffraction_renders_the_fused_pillar_of_an_open_room():
    """The pillar's corner (4, 3) hides the receiver from the source, both in its view: the direct sound (6.5 m) is
    blocked, and the sound round the corner (6.51 m) arrives only with diffraction."""
    source, receiver = [2.0, 4.0, 1.5], [8.0, 1.5, 1.5]
    prepared = _Mesh().add("room", *grid([0, 4, 6, 10], [0, 3, 5, 8], {(1, 1)}, ceiling=False)).prepare(PILLAR_SEEDS + [source, receiver])
    geometry = room(prepared)
    scene = ca.Scene("pillar")
    scene.add_source(ca.Source("s", *source))
    scene.add_receiver(ca.Receiver("r", *receiver))

    def response(diffraction):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rirs = scene.simulate(ca.Mono(image_source_order=1, ray_tracing=False, diffraction=diffraction), geometry=geometry)
        return next(iter(rirs.values())).denormalized()

    plain, diffracted = response(False), response(True)
    n = min(len(plain), len(diffracted))
    assert not np.allclose(plain[:n], diffracted[:n])
    # the first reflection (off the floor) travels 7.16 m
    start, end = (int(d / 343.0 * 48000) for d in (6.3, 6.8))
    assert np.sum(diffracted[start:end] ** 2) > 100.0 * (np.sum(plain[start:end] ** 2) + 1e-30)


def _pair(geometry, source, receiver):
    return Pair(geometry, np.asarray(source, float), np.asarray(receiver, float), np.eye(3), 48000, require_volume=False)


def _corner(e):
    return tuple(int(round(c)) for c in e.start[:2])


def test_a_selected_subset_of_edges_is_all_the_paths_use():
    """Source and receiver at opposite corners of the room, the pillar between: corners (6, 3) and (4, 5) see both."""
    geometry = room(open_pillar_room())
    detected = diffraction_edges(geometry)
    spec = _pair(geometry, [2.0, 1.5, 1.5], [8.0, 6.5, 1.5])
    every = diffraction_paths(spec, max_order=0)
    assert sorted(_corner(detected[p.edge]) for p in every) == [(4, 5), (6, 3)]

    chosen = [e for e in detected if e.start[1] < 4.0]  # the two corners at y = 3
    set_diffraction_edges(geometry, chosen)
    try:
        assert diffraction_edges(geometry) == chosen
        selected = diffraction_paths(spec, max_order=0)
        assert [_corner(chosen[p.edge]) for p in selected] == [(6, 3)]
        set_diffraction_edges(geometry, [])
        assert diffraction_edges(geometry) == [] and diffraction_paths(spec, max_order=0) == []
    finally:
        clear_diffraction_edges(geometry)
    assert diffraction_edges(geometry) == detected
    set_diffraction_edges(geometry, chosen[:1])
    set_diffraction_edges(geometry, None)  # None clears too
    assert diffraction_edges(geometry) == detected
    with pytest.raises(TypeError):
        set_diffraction_edges(geometry, [object()])


def test_a_selection_belongs_to_its_geometry():
    first, second = room(open_pillar_room()), room(open_pillar_room())
    set_diffraction_edges(first, [])
    try:
        assert diffraction_edges(first) == [] and len(diffraction_edges(second)) == 4
        # a copy starts without it; the hybrid methods' copy (materials capped) keeps it
        assert len(diffraction_edges(replace(first, materials=dict(first.materials)))) == 4
        clamped = replace(first, materials=dict(first.materials))
        _keep_selection(first, clamped)
        assert diffraction_edges(clamped) == []
    finally:
        clear_diffraction_edges(first)


# ---------------------------------------------------------------------------------------------------------------
# Rooms built otherwise keep the legacy detectors
# ---------------------------------------------------------------------------------------------------------------


def _materials(*names):
    return {n: SurfaceMaterial(n, np.full(8, 0.1)) for n in names}


def test_a_room_from_mesh_keeps_the_legacy_detectors():
    v, f = box((0, 0, 0), (10, 8, 4))
    cv, cf = box((4.0, 3.0, 0.0), (5.0, 4.5, 1.5))
    faces = [tuple(q) for q in f] + [tuple(8 + i for i in q) for q in cf]
    names = ["wall"] * 6 + ["cabinet"] * 6
    geometry = RoomGeometry.from_mesh(np.vstack([v, cv]), faces, names, _materials("wall", "cabinet"), obstacles=("cabinet",))
    assert geometry.drawn_sides is None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        legacy = obstacle_edges(geometry) + enclosure_edges(geometry)
    edges = diffraction_edges(geometry)
    assert len(edges) == len(legacy) == 8
    for a, b in zip(edges, legacy):
        assert np.array_equal(a.start, b.start) and np.array_equal(a.end, b.end) and a.theta_w == b.theta_w and a.faces == b.faces


def test_an_open_room_from_mesh_still_warns_that_its_boundary_is_not_searched():
    v, f = prism([(0, 0), (8, 0), (8, 3), (5, 3), (5, 6), (0, 6)])
    with pytest.warns(UserWarning):
        geometry = RoomGeometry.from_mesh(v, [tuple(q) for q in f[:-1]], ["wall"] * (len(f) - 1), _materials("wall"))
    assert geometry.drawn_sides is None and not geometry.is_closed
    with pytest.warns(ShellEdgeWarning, match="does not close"):
        assert diffraction_edges(geometry) == []
