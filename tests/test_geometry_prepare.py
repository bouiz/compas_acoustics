"""Scene tests of compas_acoustics.geometry.prepare_mesh (ported from compas_soundscape).

Each scene reproduces a flaw of the former weld_mesh / fix_outward_winding
pipeline (or a case of the hierarchical orientation rules):
  - re-entrant walls of an L-shaped room must not be flipped,
  - an inward-wound room must be flipped entirely,
  - the shared wall of two adjoining rooms must survive (no pruning),
  - an open ceiling must be reported as a leak + hole,
  - a source outside the model must be detected,
  - loose-face furniture stays one-sided with normals into the solid,
  - a free-standing panel with air on both sides becomes two-sided,
  - coplanar triangles merge into polygon walls,
  - the preparation is deterministic.
"""

import numpy as np

from compas_acoustics.geometry import FACE_CLASS_INTERIOR
from compas_acoustics.geometry import SEED_OUTSIDE
from compas_acoustics.geometry import PrepareOptions
from compas_acoustics.geometry import prepare_mesh

# Box quads wound with outward normals (vertex order of _box).
_BOX_QUADS = [[0, 3, 2, 1], [4, 5, 6, 7], [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]]


def _box(x0, y0, z0, x1, y1, z1, base=0, inward=False):
    v = [[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0], [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]]
    f = []
    for a, b, c, d in _BOX_QUADS:
        tris = [[a, b, c], [a, c, d]]
        for t in tris:
            t = [i + base for i in t]
            f.append(t[::-1] if inward else t)
    return v, f


class _Scene:
    def __init__(self):
        self.v, self.f, self.ranges = [], [], {}

    def add(self, name, verts, faces_local):
        base = len(self.v)
        start = len(self.f)
        self.v += [list(p) for p in verts]
        self.f += [[i + base for i in t] for t in faces_local]
        self.ranges[name] = [start, len(self.f) - 1]
        return list(range(start, len(self.f)))

    def prepare(self, seeds, **settings):
        materials = {i: "rough_concrete" for i in range(len(self.f))}
        return prepare_mesh(self.v, self.f, self.ranges, materials, None, seeds, PrepareOptions(**settings))


def _room(scene, name="room", inward=False, size=(6, 4, 3)):
    v, f = _box(0, 0, 0, *size, inward=inward)
    return scene.add(name, v, f)


def _faces_of(mesh, name):
    return np.flatnonzero(mesh.face_object == mesh.objects.index(name))


def test_l_room_reentrant_walls_not_flipped():
    outline = [(0, 0), (10, 0), (10, 4), (4, 4), (4, 10), (0, 10)]
    n, h = len(outline), 3.0
    v = [[x, y, 0] for x, y in outline] + [[x, y, h] for x, y in outline]
    f = []
    for i in range(n):
        j = (i + 1) % n
        f += [[i, j, j + n], [i, j + n, i + n]]
    fan = [[0, 1, 2], [0, 2, 3], [0, 3, 4], [0, 4, 5]]
    f += [[a, c, b] for a, b, c in fan] + [[a + n, b + n, c + n] for a, b, c in fan]
    scene = _Scene()
    scene.add("room", v, f)
    mesh = scene.prepare([[2, 2, 1.5], [2, 8, 1.5], [8, 2, 1.5]])
    assert mesh.flipped.sum() == 0
    assert mesh.two_sided.sum() == 0


def test_inward_room_is_flipped():
    scene = _Scene()
    _room(scene, inward=True)
    mesh = scene.prepare([[3, 2, 1.5]])
    assert mesh.flipped.all()


def test_adjoining_rooms_keep_shared_wall():
    scene = _Scene()
    va, fa = _box(0, 0, 0, 3, 4, 3)
    vb, fb = _box(3, 0, 0, 6, 4, 3)
    scene.add("a", va, fa)
    scene.add("b", vb, fb)
    mesh = scene.prepare([[1.5, 2, 1.5], [4.5, 2, 1.5]])
    assert mesh.stats["welded_faces"] == 24  # nothing pruned
    # Each room keeps its own partition, one-sided and facing its own air
    # (normal away from the air): two coincident walls, one per room.
    for name, sign in (("a", 1.0), ("b", -1.0)):
        shared = [f for f in _faces_of(mesh, name) if np.allclose(mesh.vertices[mesh.faces[f]][:, 0], 3.0)]
        assert len(shared) == 2
        assert all(mesh.normals[f][0] * sign > 0.99 for f in shared)


def test_open_ceiling_leaks_and_has_hole():
    scene = _Scene()
    v, f = _box(0, 0, 0, 6, 4, 3)
    del f[2]  # one ceiling triangle
    scene.add("room", v, f)
    mesh = scene.prepare([[3, 2, 1.5]])
    assert mesh.visibility.leak_fraction > 0
    assert len(mesh.loops) == 1
    assert len(mesh.visibility.leak_origins) > 0


def test_source_outside_detected():
    scene = _Scene()
    _room(scene)
    mesh = scene.prepare([[3, 2, 1.5], [20, 2, 1.5]])
    assert mesh.visibility.seeds[0].status != SEED_OUTSIDE
    assert mesh.visibility.seeds[1].status == SEED_OUTSIDE
    assert mesh.flipped.sum() == 0  # the outside seed must not vote


def test_loose_face_furniture_is_one_sided_into_solid():
    scene = _Scene()
    _room(scene)
    # Unit cube at (2.5..3.5, 1.5..2.5, 0.5..1.5); each quad shrunk by 5 mm so
    # nothing welds, every second quad deliberately wound inward.
    cv, cf = _box(2.5, 1.5, 0.5, 3.5, 2.5, 1.5)
    center = np.array([3.0, 2.0, 1.0])
    loose_v, loose_f = [], []
    for q, (a, b, c, d) in enumerate(_BOX_QUADS):
        quad = np.array([cv[a], cv[b], cv[c], cv[d]], dtype=float)
        quad = quad + 0.005 * (quad.mean(axis=0) - quad)
        base = len(loose_v)
        loose_v += quad.tolist()
        tris = [[base, base + 1, base + 2], [base, base + 2, base + 3]]
        loose_f += [t[::-1] for t in tris] if q % 2 else tris
    scene.add("chair", loose_v, loose_f)
    mesh = scene.prepare([[1, 1, 1.5], [5, 3, 1.5]])
    chair = _faces_of(mesh, "chair")
    assert mesh.objects.index("chair") in mesh.enclosing_objects
    assert not mesh.two_sided[chair].any()
    centroids = mesh.vertices[mesh.faces[chair]].mean(axis=1)
    assert (np.einsum("ij,ij->i", mesh.normals[chair], center - centroids) > 0).all()


def test_free_standing_panel_is_two_sided():
    scene = _Scene()
    _room(scene)
    scene.add("panel", [[3, 1, 0.5], [3, 3, 0.5], [3, 3, 2.5], [3, 1, 2.5]], [[0, 1, 2], [0, 2, 3]])
    mesh = scene.prepare([[1.5, 2, 1.5], [4.5, 2, 1.5]])
    panel = _faces_of(mesh, "panel")
    assert mesh.two_sided[panel].all()
    assert (mesh.face_class[panel] == FACE_CLASS_INTERIOR).all()  # thin surface inside the room


def test_column_faces_are_interior():
    scene = _Scene()
    _room(scene)
    cv, cf = _box(2.8, 1.8, 0, 3.2, 2.2, 3)
    scene.add("column", cv, cf)
    mesh = scene.prepare([[1, 1, 1.5], [5, 3, 1.5]])
    column = _faces_of(mesh, "column")
    seen = (mesh.air_hits[column] + mesh.other_hits[column]) > 0
    assert (mesh.face_class[column][seen] == FACE_CLASS_INTERIOR).mean() > 0.8


def test_coplanar_box_merges_to_six_walls():
    scene = _Scene()
    _room(scene)
    mesh = scene.prepare([[3, 2, 1.5]])
    assert len(mesh.walls) == 6
    assert all(len(w.corners) == 4 for w in mesh.walls)


def test_merge_disabled_keeps_triangles():
    scene = _Scene()
    _room(scene)
    mesh = scene.prepare([[3, 2, 1.5]], merge_coplanar=False)
    assert len(mesh.walls) == 12


def test_unassigned_faces_dropped():
    scene = _Scene()
    _room(scene)
    materials = {i: "rough_concrete" for i in range(10)}
    mesh = prepare_mesh(scene.v, scene.f, scene.ranges, materials, None, [[3, 2, 1.5]])
    assert mesh.stats["dropped_unassigned_faces"] == 2
    assert len(mesh.faces) == 10


def test_deterministic():
    scene = _Scene()
    _room(scene)
    scene.add("panel", [[3, 1, 0.5], [3, 3, 0.5], [3, 3, 2.5], [3, 1, 2.5]], [[0, 1, 2], [0, 2, 3]])
    seeds = [[1.5, 2, 1.5], [4.5, 2, 1.5]]
    a, b = scene.prepare(seeds), scene.prepare(seeds)
    assert np.array_equal(a.faces, b.faces)
    assert np.array_equal(a.two_sided, b.two_sided)
    assert np.array_equal(a.face_class, b.face_class)
    assert a.air_volume == b.air_volume
