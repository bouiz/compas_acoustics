"""The sector detector (compas_acoustics.geometry.edges) against the legacy diffraction_edges, and what only it finds.

Comparison: on scenes where the legacy detector is defined, the two must give the same segments (ends unordered,
1e-6 m), the same theta_w (1e-6 rad) and the same unordered pair of face directions (1e-6). ``faces``,
``component`` and ``regions`` are compared and the differences reported (printed with ``pytest -s``); where the
detectors disagree the test asserts the physically right result, with the reason.
"""

import os
import sys
import time
import warnings

import numpy as np
import pytest

pytest.importorskip("spharpy")  # compas_acoustics.hybrid.diffraction needs it

from compas_acoustics.geometry import PrepareOptions  # noqa: E402
from compas_acoustics.geometry import prepare_mesh  # noqa: E402
from compas_acoustics.geometry.edges import sector_edges  # noqa: E402
from compas_acoustics.geometry.edges import sector_edges_from_prepared  # noqa: E402
from compas_acoustics.geometry.edges import sector_edges_from_room  # noqa: E402
from compas_acoustics.hybrid.diffraction import TWO_PI  # noqa: E402
from compas_acoustics.hybrid.diffraction import Occluders  # noqa: E402
from compas_acoustics.hybrid.diffraction import ShellEdgeWarning  # noqa: E402
from compas_acoustics.hybrid.diffraction import _inside_polygon  # noqa: E402
from compas_acoustics.hybrid.diffraction import diffraction_edges  # noqa: E402
from compas_acoustics.hybrid.diffraction import enclosure_edges  # noqa: E402
from compas_acoustics.hybrid.diffraction import obstacle_components  # noqa: E402
from compas_acoustics.room.geometry import RoomGeometry  # noqa: E402
from compas_acoustics.room.geometry import SurfaceMaterial  # noqa: E402
from compas_acoustics.room.geometry import _polygon_area_and_centroid  # noqa: E402

H = 3.0
TILT = np.linspace(0.8, 1.2, 8)
EXAMPLES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "examples")


def materials(*names):
    # frequency-dependent, as the other diffraction tests build them
    return {n: SurfaceMaterial(n, 0.1 * TILT) for n in names}


# ---------------------------------------------------------------------------------------------------------------
# Scenes
# ---------------------------------------------------------------------------------------------------------------

ROOM_V = np.array([[0, 0, 0], [10, 0, 0], [10, 8, 0], [0, 8, 0], [0, 0, 4], [10, 0, 4], [10, 8, 4], [0, 8, 4]], dtype=float)
ROOM_F = [(0, 3, 7, 4), (1, 5, 6, 2), (0, 4, 5, 1), (3, 2, 6, 7), (0, 1, 2, 3), (4, 7, 6, 5)]
L_OUTLINE = [(0, 0), (8, 0), (8, 3), (5, 3), (5, 6), (0, 6)]


def box(lo, hi):
    """A box's eight corners and six quads, bottom first, wound outward."""
    (x0, y0, z0), (x1, y1, z1) = lo, hi
    v = np.array([[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0], [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]], dtype=float)
    f = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
    return v, f


def room_with(obstacle_v, obstacle_f, name="obstacle"):
    """The 10 x 8 x 4 m shoebox with obstacle polygons."""
    faces = ROOM_F + [tuple(8 + i for i in f) for f in obstacle_f]
    names = ["wall"] * 4 + ["floor", "ceiling"] + [name] * len(obstacle_f)
    return RoomGeometry.from_mesh(
        np.vstack([ROOM_V, np.asarray(obstacle_v, float).reshape(-1, 3)]), tuple(faces), tuple(names), materials("wall", "floor", "ceiling", name), obstacles=(name,)
    )


def shoebox():
    return RoomGeometry.from_mesh(ROOM_V, tuple(ROOM_F), ("wall",) * 4 + ("floor", "ceiling"), materials("wall", "floor", "ceiling"))


def prism_room(outline, height=H, drop=()):
    """A room extruded from a counter-clockwise outline, wound outward; ``drop`` leaves out those walls (side k
    runs from outline[k] to outline[k + 1])."""
    n = len(outline)
    v = np.array([[x, y, 0.0] for x, y in outline] + [[x, y, height] for x, y in outline], float)
    faces = [tuple(range(n))[::-1], tuple(range(n, 2 * n))]
    faces += [(k, (k + 1) % n, (k + 1) % n + n, k + n) for k in range(n) if k not in drop]
    return RoomGeometry.from_mesh(v, tuple(faces), tuple(["floor", "ceiling"] + ["wall"] * (len(faces) - 2)), materials("floor", "ceiling", "wall"))


def grid_room(xs, ys, solid, height=H, open_x0=False):
    """A room extruded from the grid cells not in ``solid``, one floor and ceiling tile per cell and a wall per cell
    side facing a solid cell or the outside (as tests/hybrid/test_shell_edges.py); ``open_x0`` leaves out the walls
    on x = xs[0]."""
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
            if not air(i - 1, j) and not (open_x0 and i == 0):
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


def room_in_room():
    """The shoebox with a closed 2 x 2 x 1 m box floating in it, drawn as part of the boundary (a room within a room)."""
    v, f = box((4.0, 3.0, 1.0), (6.0, 5.0, 2.0))
    faces = ROOM_F + [tuple(8 + i for i in q) for q in f]
    names = ("wall",) * 4 + ("floor", "ceiling") + ("wall",) * 6
    return RoomGeometry.from_mesh(np.vstack([ROOM_V, v]), tuple(faces), names, materials("wall", "floor", "ceiling"))


PANEL = [[5, 3, 0.5], [5, 5, 0.5], [5, 5, 2.5], [5, 3, 2.5]]


def floating_panel():
    return room_with(PANEL, [(0, 1, 2, 3)], "panel")


def standing_panel():
    return room_with([[5, 3, 0.0], [5, 5, 0.0], [5, 5, 2.5], [5, 3, 2.5]], [(0, 1, 2, 3)], "panel")


def floating_box():
    return room_with(*box((4.0, 3.0, 1.0), (5.0, 4.5, 2.0)))


def standing_box():
    """A closed box on the unsplit floor (its bottom in the floor's plane)."""
    return room_with(*box((4.0, 3.0, 0.0), (5.0, 4.5, 1.5)))


def shelf_above_floor():
    """The shelf of tests/hybrid/test_diffraction.py: closed, its bottom 0.2 mm above the floor."""
    return room_with(*box((4.0, 2.0, 0.0002), (4.6, 6.0, 1.523)))


def bottomless_box():
    v, f = box((4.0, 3.0, 0.0), (5.0, 4.5, 1.5))
    return room_with(v, f[1:])


def box_on_tabletop():
    top = np.array([[3, 1, 0.75], [7, 1, 0.75], [7, 7, 0.75], [3, 7, 0.75]], dtype=float)
    v, f = box((4.0, 2.0, 0.7502), (4.6, 6.0, 1.5))
    return room_with(np.vstack([top, v]), [(0, 1, 2, 3)] + [tuple(4 + i for i in q) for q in f])


def l_solid():
    """A closed L-shaped prism floating in the room: five convex vertical edges, one re-entrant (4, 4)."""
    outline = [(3, 3), (5, 3), (5, 4), (4, 4), (4, 5), (3, 5)]
    n = len(outline)
    v = [[x, y, 0.5] for x, y in outline] + [[x, y, 1.5] for x, y in outline]
    f = [tuple(range(n))[::-1], tuple(range(n, 2 * n))] + [(i, (i + 1) % n, (i + 1) % n + n, i + n) for i in range(n)]
    return room_with(v, f)


def cylinder(sides=24, radius=0.5, z0=0.5, z1=2.5, centre=(5.0, 4.0)):
    """A closed tessellated column floating in the room."""
    a = np.linspace(0.0, TWO_PI, sides, endpoint=False)
    ring = np.stack([centre[0] + radius * np.cos(a), centre[1] + radius * np.sin(a)], axis=1)
    v = [[x, y, z0] for x, y in ring] + [[x, y, z1] for x, y in ring]
    f = [tuple(range(sides))[::-1], tuple(range(sides, 2 * sides))] + [(i, (i + 1) % sides, (i + 1) % sides + sides, i + sides) for i in range(sides)]
    return room_with(v, f)


def partition_on_split_wall():
    """The shoebox with its wall x = 0 split at y = 3 and a two-sided obstacle partition (y = 3, x 0..4, z 0..2)
    meeting the split: the boundary still closes, so the legacy detector is defined."""
    v = np.vstack([ROOM_V, [[0, 3, 0], [0, 3, 4]], [[0, 3, 0], [4, 3, 0], [4, 3, 2], [0, 3, 2]]])
    faces = [(0, 8, 9, 4), (8, 3, 7, 9)] + ROOM_F[1:] + [(10, 11, 12, 13)]
    names = ("wall",) * 5 + ("floor", "ceiling", "panel")
    return RoomGeometry.from_mesh(v, tuple(faces), names, materials("wall", "floor", "ceiling", "panel"), obstacles=("panel",))


def hil():
    """examples/hil_d24_1.obj, loaded as examples/hil_d24_1.py loads it."""
    sys.path.insert(0, EXAMPLES)
    try:
        from hil_d24_1 import OBJ_PATH
        from hil_d24_1 import load_room

        import compas_acoustics as ca

        scene = ca.Scene("hil")
        scene.add_acoustic_objects(load_room(OBJ_PATH))
        return ca.RoomGeometry.from_scene(scene)
    finally:
        sys.path.remove(EXAMPLES)


# ---------------------------------------------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------------------------------------------


def _same_ends(a, b, tol=1e-6):
    return (np.allclose(a.start, b.start, atol=tol) and np.allclose(a.end, b.end, atol=tol)) or (np.allclose(a.start, b.end, atol=tol) and np.allclose(a.end, b.start, atol=tol))


def _same_faces(a, b, tol=1e-6):
    return (np.allclose(a.face_a, b.face_a, atol=tol) and np.allclose(a.face_b, b.face_b, atol=tol)) or (
        np.allclose(a.face_a, b.face_b, atol=tol) and np.allclose(a.face_b, b.face_a, atol=tol)
    )


def match(legacy, sector):
    """Pairs (legacy index, sector index) of geometrically equal edges, then the unmatched of each."""
    pairs, used = [], set()
    for i, a in enumerate(legacy):
        for j, b in enumerate(sector):
            if j not in used and _same_ends(a, b) and abs(a.theta_w - b.theta_w) <= 1e-6 and _same_faces(a, b):
                pairs.append((i, j))
                used.add(j)
                break
    matched = {i for i, _ in pairs}
    return pairs, [i for i in range(len(legacy)) if i not in matched], [j for j in range(len(sector)) if j not in used]


def _regions_by_direction(e):
    """{rounded face direction: region} for face A and face B (one entry for a knife edge, both regions)."""
    if not e.regions:
        return {}
    if e.theta_w == TWO_PI:
        return {tuple(np.round(e.face_a, 6)): tuple(sorted(e.regions))}
    return {tuple(np.round(e.face_a, 6)): (e.regions[0],), tuple(np.round(e.face_b, 6)): (e.regions[1],)}


def attribute_differences(legacy, sector, pairs):
    """Differences in faces, component and regions over the matched pairs, as text. Region labels differ between
    the detectors (legacy: a union-find root, sector: the smallest face index), so regions are compared through
    one label map per scene, aligned by face direction."""
    out, label_map = [], {}
    for i, j in pairs:
        a, b = legacy[i], sector[j]
        if set(a.faces) != set(b.faces):
            out.append(f"edge {i}: faces legacy {sorted(a.faces)} sector {sorted(b.faces)}")
        if a.component != b.component:
            out.append(f"edge {i}: component legacy {a.component} sector {b.component}")
        ra, rb = _regions_by_direction(a), _regions_by_direction(b)
        if bool(a.regions) != bool(b.regions) or set(ra) != set(rb):
            out.append(f"edge {i}: regions legacy {a.regions} sector {b.regions}")
            continue
        for key, labels in ra.items():
            for la, lb in zip(labels, rb[key]):
                if label_map.setdefault(la, lb) != lb:
                    out.append(f"edge {i}: legacy region {la} maps to sector regions {label_map[la]} and {lb}")
    if len(set(label_map.values())) != len(label_map):
        out.append(f"regions: two legacy regions map to one sector region {label_map}")
    return out


def compare(geometry):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        legacy = diffraction_edges(geometry)
        sector, diagnostics = sector_edges_from_room(geometry, return_diagnostics=True)
    pairs, only_legacy, only_sector = match(legacy, sector)
    return legacy, sector, diagnostics, pairs, only_legacy, only_sector, attribute_differences(legacy, sector, pairs)


AGREEING = {
    "shoebox": (shoebox, 0),
    "L room": (lambda: prism_room(L_OUTLINE), 1),
    "pillar in the shell": (lambda: grid_room([0, 4, 6, 10], [0, 3, 5, 8], {(1, 1)}), 4),
    "room in room": (room_in_room, 12),
    "floating panel": (floating_panel, 4),
    "standing panel": (standing_panel, 3),
    "floating box": (floating_box, 12),
    "standing box": (standing_box, 8),
    "shelf 0.2 mm above floor": (shelf_above_floor, 8),
    "bottomless box": (bottomless_box, 8),
    "box on tabletop": (box_on_tabletop, 12),
    "24-sided column": (cylinder, 72),
    "partition on split wall": (partition_on_split_wall, 2),
}


@pytest.mark.parametrize("name", list(AGREEING))
def test_the_sector_detector_finds_the_legacy_edges(name):
    build, expected = AGREEING[name]
    legacy, sector, diagnostics, pairs, only_legacy, only_sector, differences = compare(build())
    print(f"\n{name}: legacy {len(legacy)}, sector {len(sector)}, matched {len(pairs)}, {diagnostics}")
    for line in differences:
        print("  ", line)
    assert not only_legacy and not only_sector
    assert len(sector) == expected
    assert diagnostics.ambiguous == 0
    # the attributes agree too on these scenes
    assert not differences


def _into_its_face(geometry, e, u):
    """Does a point 1 cm from the edge's middle along ``u`` lie on one of the edge's faces?"""
    occ = Occluders.from_geometry(geometry)
    x = (e.start + e.end) / 2.0 + 0.01 * u
    return any(_on_polygon(x, occ.polygons[k], 1e-6) for k in e.faces)


def _on_polygon(x, polygon, tol):
    """Is x within ``tol`` of the polygon's plane and inside it or within ``tol`` of its outline?"""
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


def test_a_closed_l_solid_disagrees_only_in_the_legacy_face_directions_on_its_l_faces():
    """Disagreement: on the two convex edges along the re-entrant notch of the L's top and bottom faces, the
    legacy detector points face A or B AWAY from its face (``_into_face`` picks the side towards the face's
    vertex mean, which for an L lies on the edge's own line, so the sign falls the wrong way). The sector
    detector takes the direction from the face's winding and points into the face: it is right. The edges'
    segments and angles agree; 15 of 17 agree entirely."""
    geometry = l_solid()
    legacy, sector, diagnostics, pairs, only_legacy, only_sector, differences = compare(geometry)
    assert len(legacy) == len(sector) == 17 and len(pairs) == 15 and not differences
    assert len(only_legacy) == len(only_sector) == 2 and diagnostics.ambiguous == 0
    for i in only_legacy:
        a = legacy[i]
        (b,) = [sector[j] for j in only_sector if _same_ends(a, sector[j])]
        assert a.theta_w == pytest.approx(b.theta_w, abs=1e-9) == 1.5 * np.pi
        assert sorted(a.faces) == sorted(b.faces) and a.component == b.component
        assert not all(_into_its_face(geometry, a, u) for u in (a.face_a, a.face_b))
        assert all(_into_its_face(geometry, b, u) for u in (b.face_a, b.face_b))
    for e in sector:  # every sector edge's directions run into its faces
        assert all(_into_its_face(geometry, e, u) for u in (e.face_a, e.face_b))


def test_hil_d24_1():
    """The lecture room. Every legacy edge is found by the sector detector, except where a third polygon meets
    the legacy edge: there the legacy two-face wedge is not the geometry, and the sector detector is right.

    - The legacy detector skips the 22 chair components (open, not planar) with a warning. The sector detector
      takes their polygons as two-sided and finds their knife edges and folds: physically they diffract. These
      extra edges lie on the skipped components only.
    - A riser's top front edge (plywood) is covered in two stretches by boxes stacked flush on it (their front
      faces continue the riser's): a flat continuation there, no wedge. The legacy detector drops a piece only
      when its midpoint lies on another polygon and splits only at the component's own vertices, so it keeps
      the whole 4 m edge; the sector detector gives the three uncovered stretches.
    - Three carpet knife edges run along interior seams of the planar carpet: the T-junction vertex there lies
      7e-6 m off the long side, and the legacy split point (on the line) and the vertex round to different
      position keys, so the pieces are not matched and each looks free. The sector detector matches pieces by
      welded vertex: a coplanar seam, no edge.
    """
    try:
        geometry = hil()
    except Exception as error:  # noqa: BLE001 - the example needs the materials library and the OBJ
        pytest.skip(f"examples/hil_d24_1 does not load: {error}")
    t0 = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        legacy = diffraction_edges(geometry)
    t1 = time.perf_counter()
    sector, diagnostics = sector_edges_from_room(geometry, return_diagnostics=True)
    t2 = time.perf_counter()
    pairs, only_legacy, only_sector = match(legacy, sector)
    differences = attribute_differences(legacy, sector, pairs)
    print(f"\nHIL: legacy {len(legacy)} in {t1 - t0:.3f} s, sector {len(sector)} in {t2 - t1:.3f} s, matched {len(pairs)}, {diagnostics}")
    print("  ", [str(w.message) for w in caught if "neither closed-convex nor planar" in str(w.message)])
    for line in differences:
        print("  ", line)
    assert not differences and diagnostics.ambiguous == 0

    occ = Occluders.from_geometry(geometry)
    covered_by_pieces = set()
    for i in only_legacy:
        a = legacy[i]
        pieces = [
            j
            for j in only_sector
            if sector[j].component == a.component
            and abs(sector[j].theta_w - a.theta_w) < 1e-6
            and _same_faces(a, sector[j])
            and all(np.linalg.norm(np.cross(p - a.start, a.axis)) < 1e-6 for p in (sector[j].start, sector[j].end))
        ]
        covered_by_pieces.update(pieces)
        for z in np.linspace(0.0, 1.0, 201)[1:-1]:
            x = a.start + z * (a.end - a.start)
            in_piece = any(sector[j].contains(x) for j in pieces)
            third = [k for k in range(len(occ.polygons)) if k not in a.faces and _on_polygon(x, occ.polygons[k], 2e-3)]
            # wherever the legacy edge is not in the sector output, another polygon meets it
            assert in_piece or third, (i, z)
            # and nowhere both (strictly: the 2 mm contact band reaches past a covered stretch's end)
            assert not (in_piece and any(_on_polygon(x, occ.polygons[k], 1e-6) for k in third)), (i, z)
        print(f"   legacy-only edge {i}: component {a.component} ({geometry.obstacle_face_materials[a.faces[0]]}), {len(pieces)} sector pieces")
    covered = {e.component for e in legacy}
    extra = [j for j in only_sector if j not in covered_by_pieces]
    extra_components = {sector[j].component for j in extra}
    print(f"   {len(extra)} more sector edges, on components {sorted(extra_components)}")
    assert not extra_components & covered and -1 not in extra_components
    assert len(only_legacy) == 4
    materials = {geometry.obstacle_face_materials[obstacle_components(geometry)[c][0]] for c in extra_components}
    assert materials <= {"chairs_unoccupied_plastic", "chairs_medium_upholstered"}


# ---------------------------------------------------------------------------------------------------------------
# What the legacy detector cannot do
# ---------------------------------------------------------------------------------------------------------------


def vertical_at(edges, x, y):
    return [e for e in edges if np.allclose(e.start[:2], [x, y]) and np.allclose(e.end[:2], [x, y])]


def test_an_open_l_room_still_has_its_re_entrant_corner():
    """The L prism without its wall x = 0: the boundary does not close, so the legacy detector searches nothing."""
    with pytest.warns(UserWarning):
        geometry = prism_room(L_OUTLINE, drop=(5,))
    assert not geometry.is_closed
    with pytest.warns(ShellEdgeWarning):
        assert enclosure_edges(geometry) == []
    edges, diagnostics = sector_edges_from_room(geometry, return_diagnostics=True)
    assert len(edges) == 1
    (e,) = vertical_at(edges, 5.0, 3.0)
    assert e.length == pytest.approx(H) and e.theta_w == pytest.approx(1.5 * np.pi)
    walls = sorted(e.faces)
    assert len(walls) == 2 and e.component == -1 and len(set(e.regions)) == 2
    assert diagnostics.open_rims > 0 and diagnostics.ambiguous == 0  # the opening's rim gives nothing


def test_an_open_room_finds_the_corners_of_a_pillar_fused_into_its_shell():
    with pytest.warns(UserWarning):
        geometry = grid_room([0, 4, 6, 10], [0, 3, 5, 8], {(1, 1)}, open_x0=True)
    assert not geometry.is_closed
    edges, diagnostics = sector_edges_from_room(geometry, return_diagnostics=True)
    assert len(edges) == 4
    for x, y in ((4, 3), (6, 3), (6, 5), (4, 5)):
        (e,) = vertical_at(edges, x, y)
        assert e.theta_w == pytest.approx(1.5 * np.pi) and e.length == pytest.approx(H)
    assert diagnostics.ambiguous == 0 and diagnostics.open_rims > 0


def t_junction_polygons(partition_two_sided=True):
    """A 10 x 8 x 4 m box, its wall x = 0 split at y = 3 into two faces, and a partition (y = 3, x 0..4, z 0..2)
    whose side x = 0 is shared with both wall faces: three faces at one edge (not manifold)."""
    p = {
        "floor": [[0, 0, 0], [0, 8, 0], [10, 8, 0], [10, 0, 0]],
        "ceiling": [[0, 0, 4], [10, 0, 4], [10, 8, 4], [0, 8, 4]],
        "x0 south": [[0, 0, 0], [0, 0, 4], [0, 3, 4], [0, 3, 0]],
        "x0 north": [[0, 3, 0], [0, 3, 4], [0, 8, 4], [0, 8, 0]],
        "x10": [[10, 0, 0], [10, 8, 0], [10, 8, 4], [10, 0, 4]],
        "y0": [[0, 0, 0], [10, 0, 0], [10, 0, 4], [0, 0, 4]],
        "y8": [[0, 8, 0], [0, 8, 4], [10, 8, 4], [10, 8, 0]],
        "partition": [[0, 3, 0], [4, 3, 0], [4, 3, 2], [0, 3, 2]],
    }
    polygons = [np.array(q, float) for q in p.values()]
    centre = np.array([5.0, 4.0, 2.0])
    normals = []
    for q in polygons:
        n = np.cross(q[1] - q[0], q[2] - q[0])
        n /= np.linalg.norm(n)
        normals.append(n if n @ (q.mean(axis=0) - centre) > 0 else -n)  # away from the air
    two_sided = [False] * 7 + [partition_two_sided]
    return polygons, np.array(normals), two_sided


def test_a_non_manifold_t_junction_gives_no_edge_and_the_partition_its_knife_edges():
    polygons, normals, two_sided = t_junction_polygons()
    edges, diagnostics = sector_edges(polygons, normals, two_sided)
    assert diagnostics.ambiguous == 0
    assert len(edges) == 2
    assert all(e.theta_w == TWO_PI and e.faces == (7,) for e in edges)
    top = [e for e in edges if np.allclose([e.start[2], e.end[2]], 2.0)]
    free = vertical_at(edges, 4.0, 3.0)
    assert len(top) == len(free) == 1
    assert top[0].length == pytest.approx(4.0) and free[0].length == pytest.approx(2.0)


def test_a_one_sided_partition_at_a_t_junction_is_ambiguous_and_gives_nothing():
    """Drawn as a boundary face, a zero-thickness partition is one-sided: air on one side only, which the walls
    contradict at the junction. The legacy detector sees a non-manifold boundary and searches nothing."""
    polygons, normals, _ = t_junction_polygons()
    v = np.vstack(polygons)
    faces, k = [], 0
    for q in polygons:
        faces.append(tuple(range(k, k + len(q))))
        k += len(q)
    with pytest.warns(UserWarning):
        geometry = RoomGeometry.from_mesh(v, tuple(faces), ("wall",) * 8, materials("wall"), weld_tolerance=1e-6)
    assert not geometry.is_closed
    with pytest.warns(ShellEdgeWarning):
        assert enclosure_edges(geometry) == []
    edges, diagnostics = sector_edges(*t_junction_polygons(partition_two_sided=False))
    assert edges == [] and diagnostics.ambiguous >= 1 and diagnostics.open_rims >= 1


def slab(lo, hi, base):
    """A closed box as 12 outward triangles, vertex indices from ``base``."""
    v, quads = box(lo, hi)
    return v, [[base + a, base + b, base + c] for q in quads for a, b, c in ((q[0], q[1], q[2]), (q[0], q[2], q[3]))]


def test_a_room_of_six_closed_slabs_has_no_edges():
    """Floor, ceiling and four walls, each a closed 0.2 m slab, enclosing 6 x 4 x 3 m of air. The inner corners
    open pi/2 to the air and the slabs' outer skins are never reached, so nothing diffracts."""
    t = 0.2
    pieces = {
        "floor": ((-t, -t, -t), (6 + t, 4 + t, 0.0)),
        "ceiling": ((-t, -t, 3.0), (6 + t, 4 + t, 3 + t)),
        "west": ((-t, -t, 0.0), (0.0, 4 + t, 3.0)),
        "east": ((6.0, -t, 0.0), (6 + t, 4 + t, 3.0)),
        "south": ((0.0, -t, 0.0), (6.0, 0.0, 3.0)),
        "north": ((0.0, 4.0, 0.0), (6.0, 4 + t, 3.0)),
    }
    vertices, faces, ranges = [], [], {}
    for name, (lo, hi) in pieces.items():
        v, f = slab(lo, hi, len(vertices))
        ranges[name] = [len(faces), len(faces) + len(f) - 1]
        vertices += v.tolist()
        faces += f
    seeds = [[1.0, 1.0, 1.5], [4.5, 3.0, 1.2], [3.0, 2.0, 2.5]]
    prepared = prepare_mesh(vertices, faces, ranges, dict.fromkeys(range(len(faces)), "wall"), None, seeds, PrepareOptions())
    edges, diagnostics = sector_edges_from_prepared(prepared)
    hits = prepared.air_hits + prepared.other_hits
    unreached = sum(not np.any(hits[w.faces] > 0) for w in prepared.walls)
    print(f"\nslabs: {len(prepared.walls)} walls, {unreached} unreached, {diagnostics}")
    assert edges == []
    assert diagnostics.ambiguous == 0


def test_min_excess_drops_the_facets_of_a_tessellated_column():
    """A 24-sided column: its vertical facet edges open pi + 2 pi / 24 (0.26 rad beyond pi), its rims 3 pi / 2."""
    geometry = cylinder(24)
    facet = np.pi + TWO_PI / 24
    edges, diagnostics = sector_edges_from_room(geometry, return_diagnostics=True)
    vertical = [e for e in edges if abs(e.axis[2]) > 0.99]
    assert len(vertical) == 24 and all(e.theta_w == pytest.approx(facet) for e in vertical)
    assert len(edges) == 72 and diagnostics.dropped_excess == 0
    coarse, dropped = sector_edges_from_room(geometry, min_excess=0.3, return_diagnostics=True)
    assert len(coarse) == 48 and dropped.dropped_excess == 24
    assert all(e.theta_w == pytest.approx(1.5 * np.pi) for e in coarse)
    assert all(abs(e.axis[2]) < 1e-9 for e in coarse)


def test_min_length_drops_short_edges():
    edges, diagnostics = sector_edges_from_room(floating_box(), min_length=1.2, return_diagnostics=True)
    assert len(edges) == 4 and all(e.length == pytest.approx(1.5) for e in edges)
    assert diagnostics.dropped_length == 8


def test_obstacle_components_close_or_not():
    """The room adapter's reading of the obstacles: a closed box is one-sided, a bottomless one two-sided."""
    assert len(obstacle_components(floating_box())) == 1
    edges = sector_edges_from_room(bottomless_box())
    assert all(e.component == 0 for e in edges)
