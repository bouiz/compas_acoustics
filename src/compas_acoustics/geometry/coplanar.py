"""Merge adjacent coplanar triangles into planar polygon walls.

pyroomacoustics treats every wall as an image-source generator, so the ISM
cost grows as walls^order. Architectural meshes triangulate each flat wall
into many triangles, which multiplies that cost and double counts image
sources along shared triangle edges. Merging coplanar regions that share the
same acoustic properties back into one polygon fixes both.

Only regions bounded by ONE simple loop become polygons (pyroomacoustics
walls cannot have holes); anything else stays as triangles.

Ported from compas_soundscape (``backend/utils/mesh_coplanar.py``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from .topology import build_edges
from .topology import edge_face_pairs

# Sine of the turning angle below which a polygon corner is considered collinear.
_COLLINEAR_SIN_TOL = 1e-6


@dataclass
class MergedWall:
    corners: np.ndarray  # (n, 3) planar corners, counter-clockwise around the normal
    faces: np.ndarray  # triangle ids merged into this wall


def _boundary_loop(faces: np.ndarray, region: np.ndarray) -> list[int] | None:
    """Ordered boundary vertices of a region, or None if not one simple loop."""
    tri = faces[region]
    he = np.stack([tri, np.roll(tri, -1, axis=1)], axis=2).reshape(-1, 2)
    keys = np.sort(he, axis=1)
    _, inv, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    border = he[counts[inv.reshape(-1)] == 1]
    if len(border) < 3:
        return None
    nxt: dict[int, int] = {}
    for u, v in border.tolist():
        if u in nxt:  # pinch vertex -> not a simple polygon
            return None
        nxt[u] = v
    start = int(border[0, 0])
    loop = [start]
    cur = nxt.get(start)
    while cur is not None and cur != start:
        loop.append(cur)
        if len(loop) > len(nxt):
            return None
        cur = nxt.get(cur)
    if cur != start or len(loop) != len(nxt):  # several loops (holes / islands)
        return None
    return loop


def _drop_collinear(pts: np.ndarray) -> np.ndarray:
    """Remove corners lying on the straight line through their neighbours."""
    keep = []
    n = len(pts)
    for i in range(n):
        u, v = pts[i] - pts[i - 1], pts[(i + 1) % n] - pts[i]
        denom = np.linalg.norm(u) * np.linalg.norm(v)
        if denom > 0 and np.linalg.norm(np.cross(u, v)) > _COLLINEAR_SIN_TOL * denom:
            keep.append(i)
    return pts[keep] if len(keep) >= 3 else pts


def merge_coplanar(
    verts: np.ndarray,
    faces: np.ndarray,
    normals: np.ndarray,
    areas: np.ndarray,
    keys: np.ndarray,
    angle_deg: float,
    distance_m: float,
) -> list[MergedWall]:
    """
    Group edge-adjacent triangles with equal ``keys`` whose planes agree within
    ``angle_deg`` / ``distance_m`` and return one wall per group.

    ``keys`` encodes everything that must be identical inside a wall (material,
    scattering, two-sidedness, source object). Triangles must already be
    consistently oriented; merged corners follow their winding.
    """
    n = len(faces)
    if n == 0:
        return []
    centroids = verts[faces].mean(axis=1)
    table = build_edges(faces)
    # Never merge across open or non-manifold edges.
    pairs = edge_face_pairs(table, mask_edges=table.counts == 2)
    if len(pairs):
        a, b = pairs[:, 0], pairs[:, 1]
        cos_tol = np.cos(np.radians(angle_deg))
        same_plane = (
            (keys[a] == keys[b])
            & (np.einsum("ij,ij->i", normals[a], normals[b]) >= cos_tol)
            & (np.abs(np.einsum("ij,ij->i", normals[a], centroids[b] - centroids[a])) <= distance_m)
            & (np.abs(np.einsum("ij,ij->i", normals[b], centroids[a] - centroids[b])) <= distance_m)
        )
        a, b = a[same_plane], b[same_plane]
    else:
        a = b = np.zeros(0, dtype=np.int64)

    graph = coo_matrix((np.ones(len(a), dtype=np.int8), (a, b)), shape=(n, n))
    n_regions, labels = connected_components(graph, directed=False)
    order = np.argsort(labels, kind="stable")
    bounds = np.flatnonzero(np.diff(labels[order])) + 1
    regions = np.split(order, bounds)

    walls: list[MergedWall] = []
    for region in regions:
        if len(region) == 1:
            walls.append(MergedWall(verts[faces[region[0]]].copy(), region))
            continue
        w = areas[region]
        c = (centroids[region] * w[:, None]).sum(axis=0) / max(w.sum(), 1e-30)
        nrm = (normals[region] * w[:, None]).sum(axis=0)
        nrm /= max(np.linalg.norm(nrm), 1e-30)
        region_verts = np.unique(faces[region])
        planar = np.abs((verts[region_verts] - c) @ nrm).max() <= distance_m
        loop = _boundary_loop(faces, region) if planar else None
        if loop is None:
            walls.extend(MergedWall(verts[faces[f]].copy(), np.array([f])) for f in region)
            continue
        pts = verts[loop]
        pts = pts - np.outer((pts - c) @ nrm, nrm)  # project onto the fitted plane
        walls.append(MergedWall(_drop_collinear(pts), region))
    return walls


def walls_off_hull(walls: list[MergedWall], rel_tol: float) -> np.ndarray:
    """
    Walls that do not lie in a supporting plane of the convex hull — the walls
    pyroomacoustics tests for obstruction (same rule as the patched
    ``find_non_convex_walls``).
    """
    from scipy.spatial import ConvexHull
    from scipy.spatial import QhullError

    if not walls:
        return np.zeros(0, dtype=bool)
    pts = np.concatenate([w.corners for w in walls])
    diag = float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0)))
    try:
        eq = ConvexHull(pts).equations
    except QhullError:
        return np.zeros(len(walls), dtype=bool)
    centroids = np.array([w.corners.mean(axis=0) for w in walls])
    dist = np.abs(centroids @ eq[:, :-1].T + eq[:, -1][None, :]).min(axis=1)
    return dist > rel_tol * max(diag, 1.0)
