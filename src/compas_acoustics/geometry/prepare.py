"""Prepare a triangle mesh for simulation: weld, orient from the air, find two-sided surfaces, classify, merge.

Architectural models arrive open, non-manifold, unwelded and wound any way. :func:`prepare_mesh` decides which side
of every face touches the air the simulation happens in, from where the sources and receivers (the *seeds*) are,
by a diffuse random walk of rays from them (:mod:`.visibility`). Nothing is removed because of topology: open
edges, non-manifold edges and coincident faces are reported, never pruned. The result is engine-neutral:

- the geometric methods build their room with :meth:`compas_acoustics.RoomGeometry.from_prepared`, whose
  diffracting edges the sector detector (:mod:`.edges`) finds from the walls' normals, two-sided flags and hits;
- a wave solver voxelises :attr:`PreparedMesh.faces` (single-sheet triangles with a two-sided flag);
- :func:`compas_acoustics.geometry.report.analyze` turns it into user-facing issues.

The preparation is deterministic (a fixed random seed), so a preview of it shows exactly what is simulated.
Normals follow the pyroomacoustics convention: they point away from the air.

Every face becomes a wall, reached by the air or not. Whether the air reaches a face is a stochastic, seed-dependent
observation of the walk, so it never removes geometry (a face wrongly judged unreached would open a hole); the room
leaves unreached walls out of its surface accounting and its diffraction only. Objects none of whose faces the air
reaches (:meth:`PreparedMesh.unreached_objects`) are listed by :func:`~compas_acoustics.geometry.analyze`, for the
caller to leave out of the simulation if it wants a cheaper image source search.

In the room (:meth:`~compas_acoustics.RoomGeometry.from_prepared`) a wall is an obstacle, stored twice, exactly when it
is two-sided (:func:`obstacle_mask`); every one-sided wall is boundary, the shell, a wall slab and a column standing in
the room alike. :func:`standing_objects` only labels which faces belong to an object standing in the air, for display.

Ported from compas_soundscape (``backend/services/simulation_mesh_service.py``).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from dataclasses import field
from typing import Any
from typing import Dict
from typing import List
from typing import Mapping
from typing import Optional
from typing import Sequence

import numpy as np

from .coplanar import MergedWall
from .coplanar import merge_coplanar
from .orientation import OrientationInputs
from .orientation import decide_orientation
from .raycast import TriangleSet
from .raycast import first_hits
from .topology import BoundaryLoop
from .topology import boundary_loops
from .topology import build_edges
from .topology import component_closed
from .topology import connected_face_components
from .topology import dedupe_faces
from .topology import face_normals_and_areas
from .topology import orient_consistently
from .topology import signed_volume
from .topology import weld_vertices
from .visibility import AirVisibility
from .visibility import rays_for_budget
from .visibility import side_enclosures
from .visibility import walk_from_seeds

__all__ = [
    "FACE_CLASS_SHELL",
    "FACE_CLASS_INTERIOR",
    "FACE_CLASS_SKIPPED",
    "VISIBILITY_QUALITY",
    "PrepareOptions",
    "PreparedMesh",
    "prepare_mesh",
    "material_key",
    "standing_objects",
    "obstacle_mask",
    "boundary_closes",
]

# Face classes. Two-sidedness is orthogonal to the class (PreparedMesh.two_sided): a thin surface is shell when
# sound escapes the model from one of its sides, interior otherwise.
FACE_CLASS_SHELL = 0  # air-facing, backed by the exterior / solid mass
FACE_CLASS_INTERIOR = 1  # air-facing face of an object / partition inside the room
FACE_CLASS_SKIPPED = 2  # never reached from the air (kept, no acoustic effect)

#: Ray budget factor of each visibility quality.
VISIBILITY_QUALITY = {"fast": 0.4, "standard": 1.0, "thorough": 2.5}

# Mean-free-path volume needs enough chords to be meaningful.
_MIN_CHORDS_FOR_VOLUME = 100


@dataclass
class PrepareOptions:
    """How :func:`prepare_mesh` treats a mesh. Lengths in metres.

    Attributes
    ----------
    weld_tolerance : float
        Vertices closer than this merge.
    merge_coplanar : bool
        Join edge-adjacent coplanar triangles of one object and material into polygon walls (any simple polygon).
    coplanar_angle_deg, coplanar_distance : float
        How far two triangles may be from one plane and still merge.
    detect_two_sided : bool
        Make a surface two-sided when the air clearly reaches both of its sides.
    quality : float
        Scales the visibility walk's ray budget; see :data:`VISIBILITY_QUALITY`.
    interior_object_max_depth : float
        A face whose back looks, within this depth, at the back of another face reached from the same air volume
        bounds an object standing in the air (:data:`FACE_CLASS_INTERIOR`); beyond it only when both faces belong
        to one closed solid (a closed welded component or an enclosing object wound into itself). See
        :func:`_classify`.
    """

    weld_tolerance: float = 1e-4
    merge_coplanar: bool = True
    coplanar_angle_deg: float = 1.0
    coplanar_distance: float = 1e-3
    detect_two_sided: bool = True
    quality: float = 1.0
    # Internals of the walk and the decisions; the defaults are tuned on architectural models.
    min_triangle_area: float = 1e-8
    ray_seed: int = 1234
    ray_test_budget: float = 1.5e9
    rays_per_seed_min: int = 200
    rays_per_seed_max: int = 4000
    walk_bounces: int = 12
    ray_epsilon: float = 1e-5
    two_sided_min_hits: int = 2
    two_sided_min_fraction: float = 0.1
    seed_outside_escape_fraction: float = 0.5
    seed_leaky_escape_fraction: float = 0.02
    interior_object_max_depth: float = 1.0
    leak_rays_max: int = 60
    enclosing_object_min_faces: int = 4
    enclosing_object_max_faces: int = 5000
    enclosing_object_min_share: float = 0.8


@dataclass
class PreparedMesh:
    """A mesh ready to simulate. Per-face arrays follow :attr:`faces`, which keep only faces with a material."""

    vertices: np.ndarray  # (V, 3) welded, metres
    faces: np.ndarray  # (F, 3) final (simulated) winding
    normals: np.ndarray  # (F, 3) final normals, pointing away from the air
    areas: np.ndarray  # (F,)
    objects: List[str]  # object ids; faces index into this list
    face_object: np.ndarray  # (F,) object index (-1 = none)
    face_material: List[Any]  # (F,) material, as given
    face_scattering: np.ndarray  # (F,) scattering override (nan = the material's)
    two_sided: np.ndarray  # (F,)
    flipped: np.ndarray  # (F,) winding reversed vs. the model
    face_class: np.ndarray  # (F,) FACE_CLASS_*
    air_hits: np.ndarray  # (F,) hits on the reflecting (air) side
    other_hits: np.ndarray  # (F,) hits on the other side
    component: np.ndarray  # (F,) welded component id
    component_closed: np.ndarray  # (C,)
    component_manifold: np.ndarray  # (C,)
    loops: List[BoundaryLoop]
    non_manifold_edges: np.ndarray  # (K, 2) vertex pairs
    walls: List[MergedWall]  # simulated walls (merged polygons / triangles)
    visibility: Optional[AirVisibility]
    seeds: np.ndarray  # (S, 3) seed positions, in input order
    enclosing_objects: List[int]
    air_volume: float  # m^3
    air_volume_method: str  # "mean_free_path" | "divergence" | "bounding_box" | "none"
    origin: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))  # (F,) input face index
    stats: Dict[str, Any] = field(default_factory=dict)

    @property
    def air_surface_area(self) -> float:
        """Area of the faces the air reaches, each side counted once it is reached."""
        return float(self.areas[self.air_hits > 0].sum() + self.areas[self.other_hits > 0].sum())

    def unreached_objects(self) -> List[str]:
        """The ids of the objects none of whose faces the air reaches (all :data:`FACE_CLASS_SKIPPED`), in
        :attr:`objects` order; objects left without faces (no material) are not listed.

        They are walls of the room all the same, which reflect and cost image sources but are left out of its surface
        accounting and diffraction; a caller may leave them out of the simulation to make the image source search
        cheaper.
        """
        face_object = np.asarray(self.face_object, dtype=np.int64)
        owned = face_object >= 0
        present = np.zeros(len(self.objects), dtype=bool)
        present[face_object[owned]] = True
        reached = np.zeros(len(self.objects), dtype=bool)
        reached[face_object[owned & (np.asarray(self.face_class) != FACE_CLASS_SKIPPED)]] = True
        return [self.objects[o] for o in np.flatnonzero(present & ~reached)]


def material_key(value: Any) -> str:
    """A hashable, order-independent key of a material given as a name or a dict of coefficients."""
    return json.dumps(value, sort_keys=True, default=str)


def prepare_mesh(
    vertices,
    faces,
    object_face_ranges: Mapping[str, Sequence[int]],
    face_materials: Mapping[int, Any],
    face_scattering: Optional[Mapping[int, float]] = None,
    seeds=(),
    options: Optional[PrepareOptions] = None,
    force_two_sided=None,
) -> PreparedMesh:
    """Weld, orient from the air, classify and merge a triangle mesh.

    Parameters
    ----------
    vertices : (V, 3) array-like
        In metres.
    faces : (F, 3) array-like of int
        Triangles.
    object_face_ranges : mapping
        Object id to its ``[first, last]`` face index (inclusive); faces outside every range belong to no object.
    face_materials : mapping
        Face index to its material (a name, or anything the caller resolves later). Faces without one are dropped.
    face_scattering : mapping, optional
        Face index to a scattering coefficient that overrides its material's in every band.
    seeds : (S, 3) array-like
        The sources and receivers: where the air is.
    options : PrepareOptions, optional
    force_two_sided : (F,) array-like of bool, optional
        Faces that are two-sided whatever the walk finds (``role="obstacle"`` surfaces).
    """
    options = options or PrepareOptions()
    stats: Dict[str, Any] = {}
    verts_in = np.asarray(vertices, dtype=np.float64).reshape(-1, 3)
    tri = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    stats["input_vertices"], stats["input_faces"] = len(verts_in), len(tri)
    forced_in = np.zeros(len(tri), dtype=bool) if force_two_sided is None else np.asarray(force_two_sided, dtype=bool).reshape(-1)
    if forced_in.shape != (len(tri),):
        raise ValueError(f"force_two_sided must hold one flag per face ({len(tri)}), got {forced_in.shape}")

    objects = list(object_face_ranges.keys())
    face_obj = np.full(len(tri), -1, dtype=np.int64)
    for i, rng_ in enumerate(object_face_ranges.values()):
        start, end = int(rng_[0]), int(rng_[1])
        face_obj[start : end + 1] = i

    # ── Drop faces with no material ──────────────────────────────────────
    assigned = np.array([i in face_materials for i in range(len(tri))], dtype=bool)
    unassigned_objs = sorted({objects[o] for o in face_obj[~assigned] if o >= 0})
    stats["dropped_unassigned_faces"] = int((~assigned).sum())
    stats["dropped_unassigned_objects"] = unassigned_objs
    origin = np.flatnonzero(assigned)
    tri = tri[assigned]

    # ── Weld, drop degenerate / sliver faces, dedupe ─────────────────────
    welded, inv = weld_vertices(verts_in, options.weld_tolerance)
    tri = inv[tri]
    ok = (tri[:, 0] != tri[:, 1]) & (tri[:, 1] != tri[:, 2]) & (tri[:, 0] != tri[:, 2])
    stats["degenerate_faces"] = int((~ok).sum())
    tri, origin = tri[ok], origin[ok]
    _, areas = face_normals_and_areas(welded, tri)
    ok = areas >= options.min_triangle_area
    stats["sliver_faces"] = int((~ok).sum())
    tri, origin = tri[ok], origin[ok]

    dd = dedupe_faces(tri)
    stats["duplicate_faces"], stats["coincident_opposite_faces"] = dd.n_same, dd.n_opposite
    tri, origin = tri[dd.kept], origin[dd.kept]

    used, tri = np.unique(tri, return_inverse=True)
    tri = tri.reshape(-1, 3)
    verts = welded[used]
    stats["welded_vertices"], stats["welded_faces"] = len(verts), len(tri)

    face_object = face_obj[origin]
    face_material = [face_materials[int(o)] for o in origin]
    face_scat = np.array([(face_scattering or {}).get(int(o), np.nan) for o in origin], dtype=np.float64)

    # ── Topology ─────────────────────────────────────────────────────────
    table = build_edges(tri)
    n_comp, labels = connected_face_components(table)
    closed, manifold = component_closed(table, labels, n_comp)
    bfs_flip, orientable = orient_consistently(table, labels, n_comp)
    loops = boundary_loops(verts, table)
    normals, areas = face_normals_and_areas(verts, tri)

    # ── Air visibility walk ──────────────────────────────────────────────
    seed_arr = np.asarray(seeds, dtype=np.float64).reshape(-1, 3)
    tris = TriangleSet(verts, tri)
    rays = rays_for_budget(
        len(tri),
        len(seed_arr),
        options.walk_bounces,
        options.ray_test_budget,
        options.rays_per_seed_min,
        options.rays_per_seed_max,
        options.quality,
    )
    vis = walk_from_seeds(
        tris,
        normals,
        seed_arr,
        rays,
        options.walk_bounces,
        options.ray_epsilon,
        np.random.default_rng(options.ray_seed),
        options.seed_outside_escape_fraction,
        options.seed_leaky_escape_fraction,
        options.leak_rays_max,
    )

    # ── Orientation + two-sidedness ──────────────────────────────────────
    orient = decide_orientation(
        OrientationInputs(
            verts=verts,
            faces=tri,
            normals=normals,
            areas=areas,
            hits_normal=vis.hits_normal,
            hits_back=vis.hits_back,
            labels=labels,
            closed=closed,
            manifold=manifold,
            orientable=orientable,
            bfs_flip=bfs_flip,
            face_object=face_object,
        ),
        detect_two_sided=options.detect_two_sided,
        min_hits=options.two_sided_min_hits,
        min_frac=options.two_sided_min_fraction,
        eps=options.ray_epsilon,
        enclosing_min_faces=options.enclosing_object_min_faces,
        enclosing_max_faces=options.enclosing_object_max_faces,
        enclosing_min_share=options.enclosing_object_min_share,
    )
    flip = orient.flip
    two_sided = orient.two_sided | forced_in[origin]
    stats["forced_two_sided_faces"] = int((forced_in[origin] & ~orient.two_sided).sum())
    tri = np.where(flip[:, None], tri[:, ::-1], tri)
    normals = np.where(flip[:, None], -normals, normals)
    air_hits = np.where(flip, vis.hits_normal, vis.hits_back)
    other_hits = np.where(flip, vis.hits_back, vis.hits_normal)

    group, group_closed = _face_groups(labels, closed, face_object, orient.enclosing_objects)
    solid = group_closed & (_group_volumes(verts, tri, areas, group, len(group_closed)) < -_OBJECT_VOLUME_TOLERANCE)
    air_enclosure = side_enclosures(vis, len(tri))[np.arange(len(tri)), np.where(flip, 0, 1)]
    face_class = _classify(tris, tri, verts, normals, air_hits + other_hits, two_sided, air_enclosure, group, solid, options)
    volume, method = _air_volume(vis, verts, tri, areas, air_hits, other_hits, labels, closed)

    # ── Coplanar merge into walls ────────────────────────────────────────
    # Every face becomes a wall, reached by the air or not: whether the walk reached a face is a stochastic
    # observation, and dropping a face on it could open a hole.
    if options.merge_coplanar:
        key_index: Dict[tuple, int] = {}
        keys = np.array(
            [
                key_index.setdefault(
                    (
                        material_key(face_material[f]),
                        float(np.nan_to_num(face_scat[f], nan=-1.0)),
                        bool(two_sided[f]),
                        int(face_object[f]),
                    ),
                    len(key_index),
                )
                for f in range(len(tri))
            ],
            dtype=np.int64,
        )
        walls = merge_coplanar(verts, tri, normals, areas, keys, options.coplanar_angle_deg, options.coplanar_distance)
    else:
        walls = [MergedWall(verts[t].copy(), np.array([f])) for f, t in enumerate(tri)]

    stats.update(
        {
            "components": int(n_comp),
            "closed_components": int(closed.sum()),
            "non_manifold_edges": int((table.counts > 2).sum()),
            "boundary_edges": int((table.counts == 1).sum()),
            "flipped_faces": int(flip.sum()),
            "two_sided_faces": int(two_sided.sum()),
            "skipped_faces": int((face_class == FACE_CLASS_SKIPPED).sum()),
            "simulated_walls": len(walls),
            "rays_per_seed": int(rays),
            "leak_fraction": float(vis.leak_fraction),
        }
    )
    return PreparedMesh(
        vertices=verts,
        faces=tri,
        normals=normals,
        areas=areas,
        objects=objects,
        face_object=face_object,
        face_material=face_material,
        face_scattering=face_scat,
        two_sided=two_sided,
        flipped=flip,
        face_class=face_class,
        air_hits=air_hits,
        other_hits=other_hits,
        component=labels,
        component_closed=closed,
        component_manifold=manifold,
        loops=loops,
        non_manifold_edges=table.edges[table.counts > 2],
        walls=walls,
        visibility=vis,
        seeds=seed_arr,
        enclosing_objects=orient.enclosing_objects,
        air_volume=volume,
        air_volume_method=method,
        origin=origin,
        stats=stats,
    )


#: Signed volume, in m^3, below which a closed set of faces wound away from the air encloses a solid (its inside).
_OBJECT_VOLUME_TOLERANCE = 1e-9


def _face_groups(component, closed, face_object, enclosing_objects) -> tuple:
    """``(group, group_closed)``: each face's group, the welded components then each enclosing object's faces outside
    closed components (:attr:`PreparedMesh.enclosing_objects`), and whether each group encloses a volume."""
    component = np.asarray(component, dtype=np.int64)
    closed = np.asarray(closed, dtype=bool)
    face_object = np.asarray(face_object)
    group = component.copy()
    closes = list(closed)
    loose = ~closed[component] if len(component) else np.zeros(0, dtype=bool)
    for obj in enclosing_objects:
        sel = loose & (face_object == obj)
        if sel.any():
            group[sel] = len(closes)
            closes.append(True)
    return group, np.asarray(closes, dtype=bool)


def _group_volumes(verts, tri, areas, group, n: int) -> np.ndarray:
    """Signed volume of each group of faces as wound, about the group's centroid (exact for a closed group, whatever
    its position; the centroid keeps the error of a nearly closed one small)."""
    if n == 0:
        return np.zeros(0)
    points = np.asarray(verts, dtype=float)[np.asarray(tri, dtype=np.int64)]
    weight = np.bincount(group, areas, n)
    centroid = points.mean(axis=1)
    centre = np.stack([np.bincount(group, areas * centroid[:, k], n) for k in range(3)], axis=1) / np.maximum(weight, 1e-30)[:, None]
    q = points - centre[group][:, None, :]
    return np.bincount(group, np.einsum("ij,ij->i", q[:, 0], np.cross(q[:, 1], q[:, 2])) / 6.0, n)


def standing_objects(mesh) -> np.ndarray:
    """Whether each face belongs to an object standing in the air (furniture, a column, a partition), shape (F,).

    A label for user interfaces only: the room (:meth:`~compas_acoustics.RoomGeometry.from_prepared`) does not read it,
    a column standing in the room being boundary like a pillar fused into the shell. Each group of faces (a welded
    component; an enclosing object's loose faces, :attr:`PreparedMesh.enclosing_objects`) is an object when, among the
    faces the air reaches, those classed :data:`FACE_CLASS_INTERIOR` outweigh by area those classed
    :data:`FACE_CLASS_SHELL` (see :func:`_classify`). A room's shell, whose faces look out of the model or into
    another room's air, is no object; a slab whose outer skin the air never reaches is none either.

    Parameters
    ----------
    mesh : PreparedMesh
        Or anything with its ``areas``, ``face_class``, ``face_object``, ``component``, ``component_closed`` and
        ``enclosing_objects``.
    """
    face_class = np.asarray(mesh.face_class)
    if len(face_class) == 0:
        return np.zeros(0, dtype=bool)
    areas = np.asarray(mesh.areas, dtype=float)
    group, closes = _face_groups(mesh.component, mesh.component_closed, mesh.face_object, mesh.enclosing_objects)
    n = len(closes)
    interior = np.bincount(group, areas * (face_class == FACE_CLASS_INTERIOR), n)
    shell = np.bincount(group, areas * (face_class == FACE_CLASS_SHELL), n)
    return (interior > shell)[group]


def obstacle_mask(mesh) -> np.ndarray:
    """Which faces are obstacles, shape (F,): the two-sided ones (``mesh.two_sided``, which holds the surfaces forced
    two-sided, ``role="obstacle"``). The room stores an obstacle wall twice, one copy per side; every other wall is
    boundary, one-sided."""
    return np.asarray(mesh.two_sided, dtype=bool).copy()


def boundary_closes(mesh, obstacle: Optional[np.ndarray] = None) -> bool:
    """Whether the room's boundary (every face but the obstacles, :func:`obstacle_mask`) is watertight: each of its
    faces belongs to a closed welded component. Topological only: the room's ``is_closed`` is acoustic
    (:meth:`~compas_acoustics.RoomGeometry.from_prepared`), and this selects how its volume is found."""
    boundary = ~(obstacle_mask(mesh) if obstacle is None else np.asarray(obstacle, dtype=bool))
    if not boundary.any():
        return False
    return bool(np.all(np.asarray(mesh.component_closed)[np.asarray(mesh.component)[boundary]]))


def _classify(tris_before: TriangleSet, tri, verts, normals, total_hits, two_sided, air_enclosure, group, solid, options: PrepareOptions) -> np.ndarray:
    """Shell vs interior.

    One-sided faces: probe from the face away from the air. The face bounds an object standing in the air (column,
    furniture, partition: :data:`FACE_CLASS_INTERIOR`) when the probe meets the back of another face whose air side is
    reached from the same air volume as its own (``air_enclosure``, the walk's enclosures), and either within
    ``interior_object_max_depth`` or with both faces in one closed solid (``solid``, by ``group``: a closed component
    or enclosing object wound into itself, however thick). Otherwise it bounds the exterior, a solid mass or another
    room's wall (:data:`FACE_CLASS_SHELL`): a wall between two rooms that share no air is shell on both sides. The
    depth still limits an object that does not close (a bottomless box), since the space behind a face of a closed
    room's shell can reach the back of the same room's wall across the outside (the arms of a U-shaped room).

    Two-sided faces have air on both sides: probe both directions. If sound can leave the model from either side
    the surface is part of the outer shell (e.g. a single-sheet facade); otherwise it stands inside the room
    (panel, partition, ceiling cloud).
    """
    cls = np.full(len(tri), FACE_CLASS_SHELL, dtype=np.int8)
    seen = total_hits > 0
    cls[~seen] = FACE_CLASS_SKIPPED
    centroids = verts[tri].mean(axis=1)

    def probe(ids: np.ndarray, d: np.ndarray):
        return first_hits(tris_before, centroids[ids] + options.ray_epsilon * d, d, tmin=0.0)

    one = np.flatnonzero(seen & ~two_sided)
    if len(one):
        d = normals[one]
        t, g = probe(one, d)
        hit = g >= 0
        g_safe = np.where(hit, g, 0)
        back_of_seen = hit & (total_hits[g_safe] > 0) & (np.einsum("ij,ij->i", d, normals[g_safe]) < 0)
        same_air = (air_enclosure[g_safe] >= 0) & (air_enclosure[g_safe] == air_enclosure[one])
        enclosed = (group[g_safe] == group[one]) & solid[group[one]]
        cls[one[back_of_seen & same_air & ((t <= options.interior_object_max_depth) | enclosed)]] = FACE_CLASS_INTERIOR

    two = np.flatnonzero(seen & two_sided)
    if len(two):
        _, g_plus = probe(two, normals[two])
        _, g_minus = probe(two, -normals[two])
        escapes = (g_plus < 0) | (g_minus < 0)
        cls[two[~escapes]] = FACE_CLASS_INTERIOR
    return cls


def _air_volume(vis, verts, tri, areas, air_hits, other_hits, labels, closed) -> tuple:
    """Air volume from the mean free path (V = l S / 4), else geometric fallbacks."""
    if vis.n_chords >= _MIN_CHORDS_FOR_VOLUME:
        surface = float(areas[air_hits > 0].sum() + areas[other_hits > 0].sum())
        if surface > 0:
            return vis.mean_free_path * surface / 4.0, "mean_free_path"
    if len(closed) and closed.any():
        biggest = max(np.flatnonzero(closed), key=lambda c: areas[labels == c].sum())
        vol = abs(signed_volume(verts, tri[labels == biggest]))
        if vol > 0:
            return vol, "divergence"
    if len(verts) == 0:
        return 0.0, "none"
    extents = verts.max(axis=0) - verts.min(axis=0)
    return float(np.prod(np.maximum(extents, 0.0))), "bounding_box"
