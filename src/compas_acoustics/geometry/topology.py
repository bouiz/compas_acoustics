"""Stateless triangle-mesh topology helpers for the simulation mesh pipeline.

Everything here is pure numpy/scipy and side-effect free. Nothing removes
geometry because of topology: non-manifold edges, open boundaries and
coincident faces are REPORTED, never pruned. pyroomacoustics does not need a
manifold mesh, and pruning silently deletes real walls (e.g. the shared wall
of two adjoining rooms).

Ported from compas_soundscape (``backend/utils/mesh_topology.py``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

# ─── Vertices ─────────────────────────────────────────────────────────────────


def weld_vertices(vertices, tolerance: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Merge vertices closer than ``tolerance`` (metres).

    Uses a KD-tree pair query + connected components, so two points within
    tolerance always merge (grid snapping misses pairs straddling a cell
    boundary). Chains of close points merge transitively.

    Returns:
        ``(welded (V', 3) float64, inverse (V,) int)`` where ``inverse[i]`` is
        the welded index of input vertex ``i``. Welded positions are cluster
        means.
    """
    verts = np.asarray(vertices, dtype=np.float64).reshape(-1, 3)
    n = len(verts)
    if n == 0:
        return verts, np.zeros(0, dtype=np.int64)

    pairs = cKDTree(verts).query_pairs(max(tolerance, 0.0), output_type="ndarray")
    if len(pairs) == 0:
        return verts.copy(), np.arange(n, dtype=np.int64)

    graph = coo_matrix((np.ones(len(pairs), dtype=np.int8), (pairs[:, 0], pairs[:, 1])), shape=(n, n))
    n_clusters, labels = connected_components(graph, directed=False)
    sums = np.zeros((n_clusters, 3))
    np.add.at(sums, labels, verts)
    counts = np.bincount(labels, minlength=n_clusters)[:, None]
    return sums / counts, labels.astype(np.int64)


# ─── Faces ────────────────────────────────────────────────────────────────────


def face_normals_and_areas(verts: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Unit normals (zero for degenerate faces) and areas of triangles."""
    if len(faces) == 0:
        return np.zeros((0, 3)), np.zeros(0)
    p = verts[faces]
    cross = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
    norm = np.linalg.norm(cross, axis=1)
    normals = np.divide(cross, norm[:, None], out=np.zeros_like(cross), where=norm[:, None] > 0)
    return normals, 0.5 * norm


def winding_parity(faces: np.ndarray) -> np.ndarray:
    """
    Orientation parity of each triangle relative to its sorted vertex set.

    Two triangles over the same three vertices have the same winding iff their
    parities are equal.
    """
    if len(faces) == 0:
        return np.zeros(0, dtype=bool)
    rot = np.argmin(faces, axis=1)
    idx = np.arange(len(faces))
    nxt = faces[idx, (rot + 1) % 3]
    prv = faces[idx, (rot + 2) % 3]
    return nxt < prv


@dataclass
class FaceDedupe:
    kept: np.ndarray  # (K,) indices into the input faces, original order
    group: np.ndarray  # (F,) position in ``kept`` of each input face
    opposite_pair: np.ndarray  # (K,) a duplicate with opposite winding exists
    n_same: int  # same-winding duplicates dropped
    n_opposite: int  # opposite-winding duplicates dropped


def dedupe_faces(faces: np.ndarray) -> FaceDedupe:
    """
    Collapse faces spanning the same vertex set to the first occurrence.

    Opposite-winding coincident pairs (two solids touching, or a surface
    modelled twice) are flagged so the air-visibility pass can decide whether
    the kept face is one- or two-sided.
    """
    n = len(faces)
    if n == 0:
        empty = np.zeros(0, dtype=np.int64)
        return FaceDedupe(empty, empty, np.zeros(0, dtype=bool), 0, 0)

    keys = np.sort(faces, axis=1)
    _, first, inverse = np.unique(keys, axis=0, return_index=True, return_inverse=True)
    inverse = inverse.reshape(-1)
    order = np.argsort(first)
    kept = first[order]
    rank = np.empty_like(order)
    rank[order] = np.arange(len(order))
    group = rank[inverse]

    parity = winding_parity(faces)
    opposite = np.zeros(len(kept), dtype=bool)
    differs = parity != parity[kept[group]]
    opposite[group[differs]] = True
    n_dupes = n - len(kept)
    n_opposite = int(differs.sum())
    return FaceDedupe(kept, group, opposite, n_dupes - n_opposite, n_opposite)


# ─── Edges ────────────────────────────────────────────────────────────────────


@dataclass
class EdgeTable:
    edges: np.ndarray  # (E, 2) sorted vertex pairs
    face_edges: np.ndarray  # (F, 3) edge id of each half-edge (v0v1, v1v2, v2v0)
    he_forward: np.ndarray  # (F, 3) half-edge runs from the smaller to the larger vertex
    counts: np.ndarray  # (E,) number of incident faces


def build_edges(faces: np.ndarray) -> EdgeTable:
    """Undirected edge table with face incidence counts."""
    if len(faces) == 0:
        z = np.zeros((0, 3), dtype=np.int64)
        return EdgeTable(np.zeros((0, 2), dtype=np.int64), z, z.astype(bool), np.zeros(0, dtype=np.int64))
    a = faces
    b = np.roll(faces, -1, axis=1)
    he = np.stack([a, b], axis=2).reshape(-1, 2)
    keys = np.sort(he, axis=1)
    edges, inverse = np.unique(keys, axis=0, return_inverse=True)
    inverse = inverse.reshape(-1)
    counts = np.bincount(inverse, minlength=len(edges))
    return EdgeTable(edges, inverse.reshape(-1, 3), (a < b), counts)


def edge_face_pairs(table: EdgeTable, mask_edges: np.ndarray | None = None) -> np.ndarray:
    """(P, 3) [face_a, face_b, edge] pairs of consecutive faces sharing an edge."""
    n_faces = len(table.face_edges)
    if n_faces == 0:
        return np.zeros((0, 3), dtype=np.int64)
    flat_edges = table.face_edges.reshape(-1)
    flat_faces = np.repeat(np.arange(n_faces), 3)
    order = np.argsort(flat_edges, kind="stable")
    e = flat_edges[order]
    f = flat_faces[order]
    same = e[1:] == e[:-1]
    pairs = np.stack([f[:-1][same], f[1:][same], e[1:][same]], axis=1)
    if mask_edges is not None:
        pairs = pairs[mask_edges[pairs[:, 2]]]
    return pairs


def connected_face_components(table: EdgeTable) -> tuple[int, np.ndarray]:
    """Faces connected through any shared edge (non-manifold included)."""
    n = len(table.face_edges)
    if n == 0:
        return 0, np.zeros(0, dtype=np.int64)
    pairs = edge_face_pairs(table)
    graph = coo_matrix((np.ones(len(pairs), dtype=np.int8), (pairs[:, 0], pairs[:, 1])), shape=(n, n))
    n_comp, labels = connected_components(graph, directed=False)
    return n_comp, labels.astype(np.int64)


def component_closed(table: EdgeTable, labels: np.ndarray, n_comp: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Per-component flags ``(closed, manifold)``.

    closed: every edge of the component is shared by exactly two faces.
    manifold: no edge is shared by more than two faces.
    """
    closed = np.ones(n_comp, dtype=bool)
    manifold = np.ones(n_comp, dtype=bool)
    if n_comp == 0:
        return closed, manifold
    edge_comp = np.zeros(len(table.edges), dtype=np.int64)
    edge_comp[table.face_edges.reshape(-1)] = np.repeat(labels, 3)
    closed[edge_comp[table.counts != 2]] = False
    manifold[edge_comp[table.counts > 2]] = False
    return closed, manifold


def orient_consistently(table: EdgeTable, labels: np.ndarray, n_comp: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Flip parity that makes each component's winding consistent.

    BFS over manifold edges (exactly two faces): neighbours must traverse their
    shared edge in opposite directions. Returns ``(flip (F,) bool, orientable
    (C,) bool)``; a component is non-orientable when a cycle disagrees.
    """
    n = len(table.face_edges)
    flip = np.zeros(n, dtype=bool)
    orientable = np.ones(n_comp, dtype=bool)
    if n == 0:
        return flip, orientable

    pairs = edge_face_pairs(table, mask_edges=table.counts == 2)
    # Direction of each face along the shared edge.
    pos_a = np.argmax(table.face_edges[pairs[:, 0]] == pairs[:, 2:3], axis=1)
    pos_b = np.argmax(table.face_edges[pairs[:, 1]] == pairs[:, 2:3], axis=1)
    same_dir = table.he_forward[pairs[:, 0], pos_a] == table.he_forward[pairs[:, 1], pos_b]

    adjacency: list[list[tuple[int, bool]]] = [[] for _ in range(n)]
    for fa, fb, s in zip(pairs[:, 0].tolist(), pairs[:, 1].tolist(), same_dir.tolist()):
        adjacency[fa].append((fb, s))
        adjacency[fb].append((fa, s))

    visited = np.zeros(n, dtype=bool)
    for root in range(n):
        if visited[root]:
            continue
        visited[root] = True
        stack = [root]
        while stack:
            f = stack.pop()
            for g, s in adjacency[f]:
                expected = flip[f] ^ s
                if not visited[g]:
                    visited[g] = True
                    flip[g] = expected
                    stack.append(g)
                elif flip[g] != expected:
                    orientable[labels[f]] = False
    return flip, orientable


# ─── Boundaries ───────────────────────────────────────────────────────────────


@dataclass
class BoundaryLoop:
    edge_ids: np.ndarray  # boundary edge ids in this loop
    edges: np.ndarray  # (k, 2) vertex pairs of those edges
    vertex_ids: np.ndarray  # unique vertices
    length: float  # total edge length (m)
    centroid: np.ndarray  # (3,)
    extent: float  # bounding-box diagonal of the loop (m)


def boundary_loops(verts: np.ndarray, table: EdgeTable) -> list[BoundaryLoop]:
    """Group open (single-face) edges into connected boundary loops."""
    b_ids = np.flatnonzero(table.counts == 1)
    if len(b_ids) == 0:
        return []
    b_edges = table.edges[b_ids]
    uniq, local = np.unique(b_edges.reshape(-1), return_inverse=True)
    local = local.reshape(-1, 2)
    m = len(uniq)
    graph = coo_matrix((np.ones(len(local), dtype=np.int8), (local[:, 0], local[:, 1])), shape=(m, m))
    n_loops, vlabel = connected_components(graph, directed=False)
    edge_loop = vlabel[local[:, 0]]
    seg = verts[b_edges[:, 1]] - verts[b_edges[:, 0]]
    seg_len = np.linalg.norm(seg, axis=1)

    loops: list[BoundaryLoop] = []
    for k in range(n_loops):
        sel = edge_loop == k
        vids = uniq[vlabel == k]
        pts = verts[vids]
        loops.append(
            BoundaryLoop(
                edge_ids=b_ids[sel],
                edges=b_edges[sel],
                vertex_ids=vids,
                length=float(seg_len[sel].sum()),
                centroid=pts.mean(axis=0),
                extent=float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0))),
            )
        )
    return loops


def signed_volume(verts: np.ndarray, faces: np.ndarray) -> float:
    """Divergence-theorem volume; positive for outward-wound closed shells."""
    if len(faces) == 0:
        return 0.0
    p = verts[faces]
    return float(np.einsum("ij,ij->i", p[:, 0], np.cross(p[:, 1], p[:, 2])).sum() / 6.0)
