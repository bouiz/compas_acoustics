"""Decide the simulated orientation of every face from where the seeds are and from air-visibility hits.

pyroomacoustics convention: a wall's normal points AWAY from the air
(out of the room / into solids), and only two-sided walls reflect from both
sides. The decision is hierarchical so that sparse or noisy ray hits never
produce inconsistent surfaces:

  1. Closed, orientable welded components (rooms, solids) are oriented as a
     whole from the seeds (sources and receivers): a component holds the air
     when a seed is inside it (parity test, :func:`points_inside_closed`), and
     its normals point out of it; otherwise it is a solid standing in the air,
     and they point into it, whether the walk reaches it or not. The walk's
     hits only cross-check this: a closed component reached clearly from the
     side the seeds make solid, or with a seed inside whose rays escape, is
     reported (:attr:`OrientationResult.conflicts`). Never two-sided.
  2. Objects made of loose (unwelded) faces that still enclose a volume
     ("closed boxes" such as furniture, or a room shell of loose panels) are
     oriented geometrically: normals point into the enclosed volume when it
     is solid, out of it when it holds the air (the side the hits arrive on;
     a seed inside it when the walk never reaches it); never two-sided.
  3. Open orientable manifold components (panels, sheets) are voted as a whole and
     become two-sided only when the air clearly reaches both sides.
  4. Non-manifold / non-orientable components fall back to per-face votes.
  5. Two-sidedness is smoothed per object (area-weighted majority).

Ported from compas_soundscape (``backend/utils/mesh_orientation.py``).
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field

import numpy as np

from .raycast import TriangleSet
from .raycast import ray_crossings
from .raycast import ray_triangle_t
from .topology import signed_volume

# Rays per batch for the closed-box parity test (bounds [batch x faces] buffers).
_PARITY_BATCH = 256
# Off-centre barycentric sample points voting in the closed-box parity test.
_PARITY_SAMPLES = np.array([[0.6, 0.25, 0.15], [0.15, 0.6, 0.25], [0.25, 0.15, 0.6]])
# Share of a closed component's area the walk reaches from the side its seeds make solid (or hold no air) at which
# the conflict is reported whatever the share of the hits.
_CONFLICT_COVERAGE = 0.5


def _parity_directions(n: int = 9) -> np.ndarray:
    """``n`` unit directions spread over the sphere and turned off the axes and the diagonals, so that no ray of the
    point-in-solid test runs along a modelled plane or edge (architectural models are aligned with the axes)."""
    k = np.arange(n) + 0.5
    z = 1.0 - 2.0 * k / n
    phi = k * np.pi * (3.0 - np.sqrt(5.0)) + 0.4
    d = np.stack([np.sqrt(1.0 - z * z) * np.cos(phi), np.sqrt(1.0 - z * z) * np.sin(phi), z], axis=1)
    axis = np.array([0.3, 0.5, 0.81])
    axis /= np.linalg.norm(axis)
    angle = 0.37
    # Rodrigues' rotation by ``angle`` about ``axis``
    return d * np.cos(angle) + np.cross(axis, d) * np.sin(angle) + np.outer(d @ axis, axis) * (1.0 - np.cos(angle))


_PARITY_DIRECTIONS = _parity_directions()


def points_inside_closed(verts: np.ndarray, faces: np.ndarray, points: np.ndarray, margin: float = 1e-7) -> tuple[np.ndarray, np.ndarray]:
    """
    Whether each point is inside a closed surface, by the parity of the crossings of rays from it.

    A ray from a point inside a closed surface crosses it an odd number of times. Several rays in fixed directions
    (:data:`_PARITY_DIRECTIONS`) vote, each ray passing within ``margin`` of an edge or a vertex (where a crossing
    may count twice or not at all), along a face's plane or from a point on a face abstaining
    (:func:`.raycast.ray_crossings`). A point outside the surface's bounding box is outside.

    Returns:
        ``(inside (P,), certain (P,))``; ``certain`` is False where no ray voted or the votes tied.
    """
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    inside = np.zeros(len(points), dtype=bool)
    certain = np.ones(len(points), dtype=bool)
    if len(points) == 0 or len(faces) == 0:
        return inside, certain
    corners = np.asarray(verts, dtype=np.float64)[np.asarray(faces, dtype=np.int64)].reshape(-1, 3)
    lo, hi = corners.min(axis=0), corners.max(axis=0)
    pad = margin * max(float(np.linalg.norm(hi - lo)), 1.0)
    boxed = np.flatnonzero(np.all((points >= lo - pad) & (points <= hi + pad), axis=1))
    if len(boxed) == 0:
        return inside, certain
    dirs = _PARITY_DIRECTIONS
    o = np.repeat(points[boxed], len(dirs), axis=0)
    d = np.tile(dirs, (len(boxed), 1))
    count, grazing = ray_crossings(TriangleSet(verts, faces), o, d, margin)
    odd = (count % 2 == 1).reshape(len(boxed), len(dirs))
    valid = ~grazing.reshape(len(boxed), len(dirs))
    votes_in = (odd & valid).sum(axis=1)
    votes = valid.sum(axis=1)
    inside[boxed] = 2 * votes_in > votes
    certain[boxed] = (votes > 0) & (2 * votes_in != votes)
    return inside, certain


@dataclass
class OrientationInputs:
    verts: np.ndarray
    faces: np.ndarray  # as welded
    normals: np.ndarray  # as welded (unit)
    areas: np.ndarray
    hits_normal: np.ndarray  # air hits on the normal side
    hits_back: np.ndarray  # air hits on the opposite side
    labels: np.ndarray  # component id per face
    closed: np.ndarray  # per component
    manifold: np.ndarray  # per component (no edge shared by 3+ faces)
    orientable: np.ndarray  # per component
    bfs_flip: np.ndarray  # per face, makes each component consistent
    face_object: np.ndarray  # object index per face (-1 = none)
    seeds: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))  # (S, 3) sources and receivers
    seed_outside: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))  # (S,) most primary rays escape


@dataclass
class OrientationResult:
    flip: np.ndarray  # (F,) reverse this face's winding
    two_sided: np.ndarray  # (F,)
    enclosing_objects: list[int]  # loose-face objects treated as closed boxes
    # closed components whose orientation from the seeds the walk contradicts: dicts of "component", "air_inside",
    # "seeds_inside", "escaping_seeds" (seeds inside whose rays mostly escape), "air_hits" and "other_hits"
    # (the hits on the side the decision puts the air on, and on the other side)
    conflicts: list[dict] = field(default_factory=list)


def _two_sided_vote(n: int, b: int, min_hits: int, min_frac: float) -> bool:
    total = n + b
    minority = min(n, b)
    return total > 0 and minority >= min_hits and minority / total >= min_frac


def enclosing_object_inward(verts: np.ndarray, faces: np.ndarray, normals: np.ndarray, face_ids: np.ndarray, eps: float) -> tuple[np.ndarray, float]:
    """
    Parity test on one object's faces.

    From points on each face, count crossings of the object's other faces
    along +normal and -normal. For a closed object exactly one direction
    crosses an odd number of surfaces: that direction points into the volume.
    Several off-centre sample points vote, because a ray through the centroid
    of a symmetric face often grazes the shared diagonal of the opposite quad
    (counted twice, flipping the parity).

    Returns:
        ``(inward_sign (k,), share)`` with +1 when the normal already points
        inward, -1 when it must flip; ``share`` is the fraction of faces the
        parity test decided (the rest fall back to the object centroid).
    """
    tris = TriangleSet(verts, faces[face_ids])
    corners = verts[faces[face_ids]]
    nrm = normals[face_ids]
    votes = np.zeros(len(face_ids), dtype=np.int64)
    for bary in _PARITY_SAMPLES:
        pts = np.einsum("k,fkj->fj", bary, corners)
        for a in range(0, len(face_ids), _PARITY_BATCH):
            sl = slice(a, a + _PARITY_BATCH)
            odd = []
            for direction in (1.0, -1.0):
                d = nrm[sl] * direction
                t = ray_triangle_t(tris, pts[sl] + eps * d, d)
                odd.append((np.isfinite(t) & (t > 0)).sum(axis=1) % 2 == 1)
            plus_odd, minus_odd = odd
            votes[sl] += np.where(plus_odd & ~minus_odd, 1, np.where(minus_odd & ~plus_odd, -1, 0))
    sign = np.sign(votes)
    share = float((sign != 0).mean()) if len(sign) else 0.0
    if (sign == 0).any():
        to_center = corners.reshape(-1, 3).mean(axis=0) - corners.mean(axis=1)
        fallback = np.where(np.einsum("ij,ij->i", nrm, to_center) > 0, 1, -1)
        sign = np.where(sign == 0, fallback, sign)
    return sign, share


def decide_orientation(
    inp: OrientationInputs,
    detect_two_sided: bool,
    min_hits: int,
    min_frac: float,
    eps: float,
    enclosing_min_faces: int,
    enclosing_max_faces: int,
    enclosing_min_share: float,
) -> OrientationResult:
    n_faces = len(inp.faces)
    flip = np.zeros(n_faces, dtype=bool)
    two_sided = np.zeros(n_faces, dtype=bool)
    handled = np.zeros(n_faces, dtype=bool)
    hn, hb = inp.hits_normal, inp.hits_back

    # Component-consistent frame: counts as seen once BFS flips are applied.
    n_cons = np.where(inp.bfs_flip, hb, hn)
    b_cons = np.where(inp.bfs_flip, hn, hb)

    seeds = np.asarray(inp.seeds, dtype=np.float64).reshape(-1, 3)
    seed_outside = np.asarray(inp.seed_outside, dtype=bool).reshape(-1)
    if len(seed_outside) != len(seeds):
        seed_outside = np.zeros(len(seeds), dtype=bool)
    conflicts: list[dict] = []

    # 1. Closed orientable components: the seeds decide, the hits cross-check.
    solid_comp = inp.closed & inp.orientable
    for c in np.flatnonzero(solid_comp):
        sel = inp.labels == c
        n, b = int(n_cons[sel].sum()), int(b_cons[sel].sum())
        cons = np.where(inp.bfs_flip[sel, None], inp.faces[sel][:, ::-1], inp.faces[sel])
        volume = signed_volume(inp.verts, cons)  # > 0: the consistent winding points out of the component
        inside, certain = points_inside_closed(inp.verts, cons, seeds)
        held = inside & certain
        if len(seeds) == 0 or volume == 0.0:
            # nothing to place: the walk's majority, else normals out of the component
            flip_root = n > b if n + b > 0 else volume < 0
        elif held.any() or certain.all():
            # a seed inside: it holds the air, normals out of it; none: a solid, normals into it
            flip_root = (volume > 0) != bool(held.any())
        else:  # a seed the parity test cannot place, none inside: the walk's majority, else a solid
            flip_root = n > b if n + b > 0 else volume > 0
        flip[sel] = inp.bfs_flip[sel] ^ flip_root
        handled[sel] = True
        air, other = (n, b) if flip_root else (b, n)  # the air is on the side opposite the final normal
        other_face = (b_cons if flip_root else n_cons)[sel] > 0
        # reached from the other side by a share of the hits, or over most of its area (a seed's walk trapped in a
        # small solid piles up its hits there, so the share alone would hide the air outside)
        coverage = inp.areas[sel][other_face].sum() / max(float(inp.areas[sel].sum()), 1e-30)
        clear = other >= min_hits and (other >= min_frac * (air + other) or coverage >= _CONFLICT_COVERAGE)
        escaping = held & seed_outside
        if escaping.any() or clear:
            conflicts.append(
                {
                    "component": int(c),
                    "objects": sorted({int(o) for o in inp.face_object[sel] if o >= 0}),
                    "air_inside": bool((volume > 0) != flip_root),
                    "seeds_inside": np.flatnonzero(held).tolist(),
                    "escaping_seeds": np.flatnonzero(escaping).tolist(),
                    "air_hits": air,
                    "other_hits": other,
                }
            )

    # 2. Loose-face objects that enclose a volume ("closed boxes").
    enclosing: list[int] = []
    for obj in np.unique(inp.face_object[~handled]):
        if obj < 0:
            continue
        ids = np.flatnonzero((inp.face_object == obj) & ~handled)
        if not (enclosing_min_faces <= len(ids) <= enclosing_max_faces):
            continue
        sign, share = enclosing_object_inward(inp.verts, inp.faces, inp.normals, ids, eps)
        if share < enclosing_min_share:
            continue
        enclosing.append(int(obj))
        # The enclosed volume is either solid (furniture: normals point in) or
        # air (a room shell modelled as loose faces: normals point out). The
        # side the air hits arrive on tells which; when the walk never reaches
        # it, whether a seed is inside it.
        air_inside = np.where(sign > 0, hn[ids], hb[ids]).sum()
        air_outside = np.where(sign > 0, hb[ids], hn[ids]).sum()
        if air_inside + air_outside == 0:
            inside, certain = points_inside_closed(inp.verts, inp.faces[ids], seeds)
            holds_air = bool((inside & certain).any())
        else:
            holds_air = air_inside > air_outside
        flip[ids] = (sign > 0) if holds_air else (sign < 0)
        handled[ids] = True

    # 3. Open orientable manifold components: whole-component vote. A
    #    non-manifold component (rooms welded to partitions, T-junctions)
    #    mixes surfaces with different air sides, so it is voted per face.
    for c in np.flatnonzero(inp.orientable & inp.manifold & ~inp.closed):
        sel = (inp.labels == c) & ~handled
        if not sel.any():
            continue
        n, b = int(n_cons[sel].sum()), int(b_cons[sel].sum())
        if n + b == 0:
            flip[sel] = False  # hidden: keep the modelled winding
        elif detect_two_sided and _two_sided_vote(n, b, min_hits, min_frac):
            two_sided[sel] = True
            flip[sel] = inp.bfs_flip[sel]
        else:
            flip[sel] = inp.bfs_flip[sel] ^ (n > b)
        handled[sel] = True

    # 4. Everything else (non-manifold / non-orientable): per-face votes.
    rest = np.flatnonzero(~handled)
    for f in rest:
        n, b = int(hn[f]), int(hb[f])
        if detect_two_sided and _two_sided_vote(n, b, min_hits, min_frac):
            two_sided[f] = True
        else:
            flip[f] = n > b

    # 5. Smooth two-sidedness per object (area-weighted majority of seen faces).
    if detect_two_sided:
        seen = (hn + hb) > 0
        enclosing_set = set(enclosing)
        for obj in np.unique(inp.face_object[two_sided]):
            if obj < 0 or int(obj) in enclosing_set:
                continue
            ids = np.flatnonzero((inp.face_object == obj) & seen & ~solid_comp[inp.labels])
            if len(ids) == 0:
                continue
            share = inp.areas[ids][two_sided[ids]].sum() / max(inp.areas[ids].sum(), 1e-30)
            two_sided[ids] = share >= 0.5

    return OrientationResult(flip=flip, two_sided=two_sided, enclosing_objects=enclosing, conflicts=conflicts)
