"""Orientation of closed surfaces from where the seeds are, and the walk of an all-outdoor scene.

A closed surface holds the air when a source or receiver is inside it (its normals point out of it), and is a solid
standing in the air otherwise (its normals point into it), whether the walk reaches it or not. When every seed is
outside the model, the exterior is the air, and the walk from those seeds votes. The scenes:
  (a) a closed absorber slab in free field, seeds 3.5 m away (BRAS RS2): normals into the slab,
  (b) the same slab against a plate of loose faces (the RS2 layout),
  (c) a closed room with a seed inside: normals out of the room (unchanged),
  (d) a closed column in a closed room: into the column, out of the room,
  (e) all-outdoor scenes: a free-standing panel hit from both sides is two-sided, from one side one-sided,
  (f) mixed seeds (one in a closed building, one in the street): the building is a room, the street seed does not
      vote, and a block standing in the street that the walk never reaches is a solid,
  (g) a seed inside a solid, or inside a closed surface its rays escape: the conflict is reported,
  (h) a closed box of loose faces that nothing reaches, no seed inside: a solid.
"""

import numpy as np
import pytest
from test_geometry_prepare import _BOX_QUADS
from test_geometry_prepare import _Scene
from test_geometry_prepare import _box

from compas_acoustics.geometry import SEED_INSIDE
from compas_acoustics.geometry import SEED_OUTSIDE
from compas_acoustics.geometry import ReportOptions
from compas_acoustics.geometry import Seed
from compas_acoustics.geometry import analyze
from compas_acoustics.geometry.orientation import OrientationInputs
from compas_acoustics.geometry.orientation import decide_orientation
from compas_acoustics.geometry.orientation import points_inside_closed
from compas_acoustics.geometry.topology import build_edges
from compas_acoustics.geometry.topology import component_closed
from compas_acoustics.geometry.topology import connected_face_components
from compas_acoustics.geometry.topology import face_normals_and_areas
from compas_acoustics.geometry.topology import orient_consistently
from compas_acoustics.geometry.topology import signed_volume

# The BRAS RS2 absorber: 20 mm thick, 2 m wide, 1.33 m high, its front face at x = 0
SLAB = (-0.02, -1.0, 0.0, 0.0, 1.0, 1.33)
RS2_SEEDS = [[3.5, 2.0, 0.67], [3.5, -2.0, 0.67]]


def _volume(mesh, name):
    """Signed volume of an object's faces as prepared: negative when its normals point into it."""
    ids = np.flatnonzero(mesh.face_object == mesh.objects.index(name))
    return signed_volume(mesh.vertices, mesh.faces[ids])


def _normals_into(mesh, name, centre):
    """Whether each face of an object has its normal towards ``centre`` (into the object)."""
    ids = np.flatnonzero(mesh.face_object == mesh.objects.index(name))
    to_centre = np.asarray(centre, dtype=float) - mesh.vertices[mesh.faces[ids]].mean(axis=1)
    return np.einsum("ij,ij->i", mesh.normals[ids], to_centre) > 0


def _loose_box(x0, y0, z0, x1, y1, z1, shrink=0.005):
    """A box of six unwelded quads (each shrunk towards its centre), every second one wound inward."""
    cv, _ = _box(x0, y0, z0, x1, y1, z1)
    verts, faces = [], []
    for q, (a, b, c, d) in enumerate(_BOX_QUADS):
        quad = np.array([cv[a], cv[b], cv[c], cv[d]], dtype=float)
        quad = quad + shrink * (quad.mean(axis=0) - quad) / np.ptp(quad, axis=0).max()
        base = len(verts)
        verts += quad.tolist()
        tris = [[base, base + 1, base + 2], [base, base + 2, base + 3]]
        faces += [t[::-1] for t in tris] if q % 2 else tris
    return verts, faces


def _conflicts(mesh):
    return mesh.stats["orientation_conflicts"]


# ── the parity test ──────────────────────────────────────────────────────


def test_points_inside_closed_on_symmetric_and_non_convex_solids():
    v, f = _box(0, 0, 0, 1, 1, 1)
    # the centre and points level with edges, vertices and face diagonals, where a ray along an axis or a diagonal
    # would graze an edge shared by two triangles
    inside = [[0.5, 0.5, 0.5], [0.5, 0.5, 0.999], [0.001, 0.5, 0.5], [0.25, 0.25, 0.25], [0.5, 0.25, 0.75]]
    outside = [[1.5, 0.5, 0.5], [0.5, 0.5, 1.001], [2.0, 2.0, 2.0], [-0.5, 0.5, 0.5], [1.0 + 1e-3, 1.0, 1.0], [0.5, 0.5, -3.0]]
    got, certain = points_inside_closed(np.array(v, float), np.array(f), np.array(inside + outside, float))
    assert certain.all()
    assert got.tolist() == [True] * len(inside) + [False] * len(outside)

    # an L-shaped prism: the notch is outside although it lies within the bounding box
    outline = [(0, 0), (4, 0), (4, 1), (1, 1), (1, 4), (0, 4)]
    n = len(outline)
    lv = [[x, y, 0] for x, y in outline] + [[x, y, 1] for x, y in outline]
    lf = []
    for i in range(n):
        j = (i + 1) % n
        lf += [[i, j, j + n], [i, j + n, i + n]]
    fan = [[0, 1, 2], [0, 2, 3], [0, 3, 4], [0, 4, 5]]
    lf += [[a, c, b] for a, b, c in fan] + [[a + n, b + n, c + n] for a, b, c in fan]
    points = [[0.5, 3.0, 0.5], [3.0, 0.5, 0.5], [0.999, 0.999, 0.5], [3.0, 3.0, 0.5], [1.001, 1.001, 0.5]]
    got, certain = points_inside_closed(np.array(lv, float), np.array(lf), np.array(points, float))
    assert certain.all()
    assert got.tolist() == [True, True, True, False, False]
    # a point on the surface (here on the re-entrant edge) is left undecided
    _, certain = points_inside_closed(np.array(lv, float), np.array(lf), np.array([[1.0, 1.0, 0.5]]))
    assert not certain.any()


# ── (a), (b) the BRAS RS2 absorber ──────────────────────────────────────


@pytest.mark.parametrize("inward", [False, True])
def test_closed_slab_in_free_field_points_into_the_slab(inward):
    scene = _Scene()
    scene.add("slab", *_box(*SLAB, inward=inward))
    mesh = scene.prepare(RS2_SEEDS)
    assert [s.status for s in mesh.visibility.seeds] == [SEED_OUTSIDE, SEED_OUTSIDE]
    assert mesh.visibility.outdoor
    assert _volume(mesh, "slab") < 0
    assert _normals_into(mesh, "slab", [-0.01, 0.0, 0.665]).all()
    front = [f for f in range(len(mesh.faces)) if np.allclose(mesh.vertices[mesh.faces[f]][:, 0], 0.0)]
    assert len(front) == 2
    assert all(mesh.normals[f][0] < -0.99 for f in front)  # the face towards the seeds: -x, into the slab
    # the walk of the outdoor seeds reaches the front, from the air side only
    assert all(mesh.air_hits[f] > 0 and mesh.other_hits[f] == 0 for f in front)
    assert not mesh.two_sided.any()
    assert not _conflicts(mesh)


def test_closed_slab_against_a_loose_plate():
    scene = _Scene()
    scene.add("plate", *_loose_box(-0.039, -1.0, 0.0, -0.02, 1.0, 1.33))
    scene.add("slab", *_box(*SLAB))
    mesh = scene.prepare(RS2_SEEDS)
    assert mesh.visibility.outdoor
    assert _volume(mesh, "slab") < 0
    assert _normals_into(mesh, "slab", [-0.01, 0.0, 0.665]).all()
    assert mesh.objects.index("plate") in mesh.enclosing_objects
    assert _normals_into(mesh, "plate", [-0.0295, 0.0, 0.665]).all()
    assert not mesh.two_sided.any()


# ── (c), (d) rooms ──────────────────────────────────────────────────────


@pytest.mark.parametrize("inward", [False, True])
def test_closed_room_with_a_seed_inside_points_out(inward):
    scene = _Scene()
    scene.add("room", *_box(0, 0, 0, 6, 4, 3, inward=inward))
    mesh = scene.prepare([[3, 2, 1.5]])
    assert _volume(mesh, "room") > 0
    assert mesh.flipped.all() == inward and mesh.flipped.any() == inward
    assert not mesh.visibility.outdoor
    assert not _conflicts(mesh)


@pytest.mark.parametrize("inward", [False, True])
def test_closed_column_in_a_closed_room(inward):
    scene = _Scene()
    scene.add("room", *_box(0, 0, 0, 6, 4, 3))
    scene.add("column", *_box(2.8, 1.8, 0.5, 3.2, 2.2, 2.5, inward=inward))
    mesh = scene.prepare([[1, 1, 1.5], [5, 3, 1.5]])
    assert [s.status for s in mesh.visibility.seeds] == [SEED_INSIDE, SEED_INSIDE]
    assert _volume(mesh, "room") > 0
    assert _volume(mesh, "column") < 0
    assert _normals_into(mesh, "column", [3.0, 2.0, 1.5]).all()
    assert not mesh.two_sided.any()
    assert not _conflicts(mesh)


# ── (e) all-outdoor scenes ──────────────────────────────────────────────


def _panel(x=0.0, size=2.0):
    h = size / 2
    return [[x, -h, 0.0], [x, h, 0.0], [x, h, size], [x, -h, size]], [[0, 1, 2], [0, 2, 3]]


def test_outdoor_panel_hit_from_both_sides_is_two_sided():
    scene = _Scene()
    scene.add("panel", *_panel())
    mesh = scene.prepare([[3.0, 0.0, 1.0], [-3.0, 0.0, 1.0]])
    assert all(s.status == SEED_OUTSIDE for s in mesh.visibility.seeds)
    assert mesh.visibility.outdoor
    assert mesh.two_sided.all()
    assert (mesh.air_hits > 0).all() and (mesh.other_hits > 0).all()


@pytest.mark.parametrize("seed_x", [3.0, -3.0])
def test_outdoor_panel_hit_from_one_side_faces_away_from_it(seed_x):
    scene = _Scene()
    scene.add("panel", *_panel())
    mesh = scene.prepare([[seed_x, 0.0, 1.0]])
    assert mesh.visibility.outdoor
    assert not mesh.two_sided.any()
    assert (np.sign(mesh.normals[:, 0]) == -np.sign(seed_x)).all()


def test_outdoor_walk_leaves_the_leak_statistics_to_seeds_inside():
    scene = _Scene()
    scene.add("panel", *_panel())
    mesh = scene.prepare([[3.0, 0.0, 1.0], [-3.0, 0.0, 1.0]])
    vis = mesh.visibility
    # free field: the escaping rays are not leaks of a room, and give no mean free path
    assert vis.n_reemitted == 0 and vis.n_chords == 0 and len(vis.leak_origins) == 0
    assert mesh.stats["leak_fraction"] == 0.0
    assert mesh.air_volume_method != "mean_free_path"


# ── (f) a building and a street ────────────────────────────────────────


def test_mixed_seeds_keep_the_building_a_room():
    scene = _Scene()
    scene.add("building", *_box(0, 0, 0, 6, 4, 3, inward=True))
    scene.add("block", *_box(15, 1, 0, 16, 3, 1))  # stands in the street
    mesh = scene.prepare([[3, 2, 1.5], [25, 2, 1.5]])
    assert [s.status for s in mesh.visibility.seeds] == [SEED_INSIDE, SEED_OUTSIDE]
    assert not mesh.visibility.outdoor
    assert _volume(mesh, "building") > 0  # flipped to point out of the room
    # the street seed does not vote: the building's outside and the block are never reached
    building = np.flatnonzero(mesh.face_object == mesh.objects.index("building"))
    assert (mesh.other_hits[building] == 0).all()
    block = np.flatnonzero(mesh.face_object == mesh.objects.index("block"))
    assert (mesh.air_hits[block] + mesh.other_hits[block] == 0).all()
    assert _volume(mesh, "block") < 0  # no seed inside it: a solid
    assert not _conflicts(mesh)


# ── (g) conflicts ───────────────────────────────────────────────────────


def test_a_seed_inside_a_solid_is_reported():
    scene = _Scene()
    scene.add("room", *_box(0, 0, 0, 6, 4, 3))
    scene.add("column", *_box(2.5, 1.5, 0.5, 3.5, 2.5, 2.5))
    mesh = scene.prepare([[1, 1, 1.5], [3.0, 2.0, 1.5]])  # the second seed is inside the column
    assert _volume(mesh, "room") > 0
    assert _volume(mesh, "column") > 0  # a seed is inside: it holds air
    column = mesh.component[np.flatnonzero(mesh.face_object == mesh.objects.index("column"))[0]]
    (conflict,) = _conflicts(mesh)
    assert conflict["component"] == column and conflict["objects"] == ["column"]
    assert conflict["air_inside"] and conflict["seeds_inside"] == [1]
    assert conflict["other_hits"] > 0 and conflict["air_hits"] > 0
    seeds = [Seed("s", "source", [1, 1, 1.5]), Seed("r", "receiver", [3.0, 2.0, 1.5])]
    report = analyze(mesh, seeds, [("s", "r")], ReportOptions())
    issues = [i for i in report.issues if i.code == "orientation_conflict"]
    assert len(issues) == 1 and issues[0].severity == "warning" and issues[0].object_ids == ["column"]
    assert len(issues[0].face_ids) == 12


def test_a_seed_whose_rays_escape_a_closed_surface_around_it_is_reported():
    scene = _Scene()
    scene.add("room", *_box(0, 0, 0, 6, 4, 3))
    # every seed counts as outside however few rays escape: the room is closed around it all the same
    mesh = scene.prepare([[3, 2, 1.5]], seed_outside_escape_fraction=0.0)
    assert mesh.visibility.seeds[0].status == SEED_OUTSIDE
    assert _volume(mesh, "room") > 0  # the parity test decides: still a room
    (conflict,) = _conflicts(mesh)
    assert conflict["escaping_seeds"] == [0] and conflict["air_inside"]


# ── (h) a closed box of loose faces nothing reaches ─────────────────────


def _inputs(verts, faces, face_object, seeds, hits_normal=None, hits_back=None):
    verts, faces = np.asarray(verts, float), np.asarray(faces, np.int64)
    table = build_edges(faces)
    n_comp, labels = connected_face_components(table)
    closed, manifold = component_closed(table, labels, n_comp)
    bfs_flip, orientable = orient_consistently(table, labels, n_comp)
    normals, areas = face_normals_and_areas(verts, faces)
    zeros = np.zeros(len(faces), dtype=np.int64)
    return OrientationInputs(
        verts=verts,
        faces=faces,
        normals=normals,
        areas=areas,
        hits_normal=zeros if hits_normal is None else hits_normal,
        hits_back=zeros if hits_back is None else hits_back,
        labels=labels,
        closed=closed,
        manifold=manifold,
        orientable=orientable,
        bfs_flip=bfs_flip,
        face_object=np.asarray(face_object, np.int64),
        seeds=np.asarray(seeds, float).reshape(-1, 3),
        seed_outside=np.zeros(len(seeds), dtype=bool),
    )


def _decide(inp):
    return decide_orientation(inp, True, 2, 0.1, 1e-5, 4, 5000, 0.8)


@pytest.mark.parametrize("seed, solid", [([5.0, 5.0, 5.0], True), ([0.5, 0.5, 0.5], False)])
def test_unreached_loose_box_is_a_solid_unless_a_seed_is_inside(seed, solid):
    verts, faces = _loose_box(0, 0, 0, 1, 1, 1)
    inp = _inputs(verts, faces, np.zeros(len(faces)), [seed])
    assert not inp.closed.any()
    result = _decide(inp)
    assert result.enclosing_objects == [0]
    final = np.where(result.flip[:, None], inp.faces[:, ::-1], inp.faces)
    normals, _ = face_normals_and_areas(inp.verts, final)
    to_centre = 0.5 - inp.verts[final].mean(axis=1)
    into = np.einsum("ij,ij->i", normals, to_centre) > 0
    assert (into == solid).all()


@pytest.mark.parametrize("seed, solid", [([5.0, 5.0, 5.0], True), ([0.5, 0.5, 0.5], False)])
def test_unreached_closed_box_is_a_solid_unless_a_seed_is_inside(seed, solid):
    verts, faces = _box(0, 0, 0, 1, 1, 1)
    inp = _inputs(verts, faces, np.zeros(len(faces)), [seed])
    result = _decide(inp)
    final = np.where(result.flip[:, None], inp.faces[:, ::-1], inp.faces)
    assert (signed_volume(inp.verts, final) < 0) == solid
    assert not result.conflicts


def test_without_seeds_an_unreached_closed_component_keeps_pointing_out():
    verts, faces = _box(0, 0, 0, 1, 1, 1, inward=True)
    result = _decide(_inputs(verts, faces, np.zeros(len(faces)), np.zeros((0, 3))))
    final = np.where(result.flip[:, None], np.asarray(faces)[:, ::-1], np.asarray(faces))
    assert signed_volume(np.asarray(verts, float), final) > 0
