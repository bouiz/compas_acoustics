"""The scattering sample: a relief closed on a baseplate, with its flat reference (Heimes & Vorländer §3.2).

A Rhino-made relief becomes two closed, outward-oriented triangle bodies with the same xy footprint:

- the **sample**: the relief, its lowest point at z = 0, walled down to ``z = −baseplate`` and closed by a bottom;
- the **reference**: the same outline, flat at z = 0, on the same baseplate.

``ScatteringSample.disc`` repeats a tile and crops it to a disc (ISO 17497-1 §6.3.2 asks for a circular sample);
``ScatteringSample.as_drawn`` takes a mesh as it is. The arrays are handled with numpy and turned into COMPAS meshes at the end.
"""

import numpy as np
from compas.data import Data
from compas.datastructures import Mesh
from compas.geometry import Polygon
from compas.geometry import earclip_polygon
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import ConvexHull
from scipy.spatial import cKDTree

from compas_acoustics.defaults import _PackageDtype

__all__ = ["ScatteringSample"]

_WELD = 1e-6  # m
_SYMMETRIES = ("none", "rotational")
_OUTLINE = "the relief must have exactly one closed outline (found {}): fill its holes and join its parts"


class ScatteringSample(_PackageDtype, Data):
    """A scattering sample and its flat reference, both closed and triangulated (Heimes & Vorländer §3.2).

    Use :meth:`disc` or :meth:`as_drawn` to make one.

    Parameters
    ----------
    sample, reference : :class:`compas.datastructures.Mesh`
        The closed sample body and the flat reference body with the same footprint, outward-oriented triangles.
    mode : {"disc", "as_drawn"}
    diameter : float
        The footprint diameter in m: the disc's, or, as drawn, the largest xy distance between outline vertices.
    period : tuple of float, optional
        The tile period (x, y) in m, in disc mode.
    structural_depth : float
        The relief's height h in m: its maximum z, its minimum being at z = 0.
    feature_size : float, optional
        The relief's characteristic feature size in m, if known.
    symmetry : {"none", "rotational"}
    baseplate : float
        The thickness in m of the plate below z = 0 that closes both bodies.
    warnings : list of str
        Problems found: ISO 17497-1 §6.3.2's depth limit ``h ≤ d/16``, or a size that suggests millimetres.
    """

    def __init__(self, sample, reference, mode, diameter, period=None, structural_depth=0.0, feature_size=None, symmetry="none", baseplate=0.01, warnings=None, name=None):
        super().__init__(name=name)
        if symmetry not in _SYMMETRIES:
            raise ValueError("unknown symmetry {!r}; valid symmetries are 'none', 'rotational'".format(symmetry))
        self.sample = sample
        self.reference = reference
        self.mode = mode
        self.diameter = float(diameter)
        self.period = tuple(float(p) for p in period) if period is not None else None
        self.structural_depth = float(structural_depth)
        self.feature_size = feature_size
        self.symmetry = symmetry
        self.baseplate = float(baseplate)
        self.warnings = list(warnings or [])

    @property
    def __data__(self):
        return {
            "sample": self.sample,
            "reference": self.reference,
            "mode": self.mode,
            "diameter": self.diameter,
            "period": list(self.period) if self.period is not None else None,
            "structural_depth": self.structural_depth,
            "feature_size": self.feature_size,
            "symmetry": self.symmetry,
            "baseplate": self.baseplate,
            "warnings": self.warnings,
        }

    @classmethod
    def __from_data__(cls, data):
        return cls(**data)

    @classmethod
    def disc(cls, tile, diameter, period=None, baseplate=0.01, segments=128, offset=(0.0, 0.0), symmetry="none", feature_size=None):
        """A tile repeated over the plane and cropped to a disc of ``diameter`` (ISO 17497-1 §6.3.2, Heimes & Vorländer §3.2).

        Parameters
        ----------
        tile : :class:`compas.datastructures.Mesh`
            One period of the relief; quads, n-gons and unwelded seams are fine.
        diameter : float
            The disc diameter d in m.
        period : tuple of float, optional
            The tile period (x, y) in m; by default the tile's xy bounding box.
        baseplate : float, optional
            The plate thickness in m below z = 0.
        segments : int, optional
            The sides of the regular polygon, inscribed in the circle of radius d/2, that stands for the disc.
        offset : tuple of float, optional
            The shift (x, y) in m of the tiling relative to the disc centre at the origin.
        symmetry : {"none", "rotational"}, optional
        feature_size : float, optional

        Raises
        ------
        ValueError
            If the cropped relief has holes or parts, i.e. not exactly one closed outline.
        """
        V, T = _clean(tile)
        if period is None:
            period = tuple(V[:, :2].max(axis=0) - V[:, :2].min(axis=0))
        V, T = _tile(V, T, period, diameter, offset)
        V, T = _clip_to_polygon(V, T, diameter / 2.0, segments)
        sample, reference, depth = _close(V, T, baseplate)
        return cls(sample, reference, "disc", diameter, period, depth, feature_size, symmetry, baseplate, _warnings(depth, diameter))

    @classmethod
    def as_drawn(cls, mesh, baseplate=0.01, symmetry="none", feature_size=None):
        """A relief as drawn, closed on a baseplate unless it is closed already (Heimes & Vorländer §3.2).

        An open relief is walled down to ``z = −baseplate`` and closed by a bottom, like :meth:`disc`.
        A closed mesh is used as it is, its lowest point moved to z = 0; its reference is the outline of its bottom, flat at z = 0.

        Raises
        ------
        ValueError
            If the relief has not exactly one closed outline, or a closed mesh has no flat bottom.
        """
        V, T = _clean(mesh)
        _, counts = _edge_counts(T)
        if (counts == 1).any():
            sample, reference, depth = _close(V, T, baseplate)
        else:
            V = V - [0.0, 0.0, V[:, 2].min()]
            if _signed_volume(V, T) < 0:
                T = T[:, ::-1]
            depth = float(V[:, 2].max())
            sample = _to_mesh(V, T)
            reference = _flat_body(V[_bottom_loop(V, T), :2], baseplate)
        diameter = _outline_diameter(reference)
        return cls(sample, reference, "as_drawn", diameter, None, depth, feature_size, symmetry, baseplate, _warnings(depth, diameter))


def _clean(mesh):
    """Triangulate a mesh by fans per face and weld its vertices within 1e-6 m; returns (vertices, triangles)."""
    index = {key: i for i, key in enumerate(mesh.vertices())}
    V = np.array([mesh.vertex_coordinates(key) for key in mesh.vertices()], dtype=float)
    T = []
    for face in mesh.faces():
        ids = [index[key] for key in mesh.face_vertices(face)]
        T.extend([ids[0], ids[i], ids[i + 1]] for i in range(1, len(ids) - 1))
    return _weld(V, np.array(T, dtype=int).reshape(-1, 3))


def _weld(V, T, tolerance=_WELD):
    """Merge vertices closer than ``tolerance``, drop the triangles that collapse and the vertices left unused."""
    pairs = cKDTree(V).query_pairs(tolerance, output_type="ndarray")
    graph = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(V), len(V)))
    _, labels = connected_components(graph, directed=False)
    W = np.zeros((labels.max() + 1, 3))
    W[labels] = V
    T = labels[T]
    T = T[(T[:, 0] != T[:, 1]) & (T[:, 1] != T[:, 2]) & (T[:, 2] != T[:, 0])]
    used, T = np.unique(T, return_inverse=True)
    return W[used], T.reshape(-1, 3)


def _tile(V, T, period, diameter, offset):
    """Repeat a tile with ``period`` (x, y) over a grid that covers ``[−d/2, d/2]²`` shifted by ``offset``, then weld."""
    px, py = float(period[0]), float(period[1])
    if px <= 0 or py <= 0:
        raise ValueError("the tile period must be positive, not {}".format((px, py)))
    x0, y0 = V[:, :2].min(axis=0) + np.asarray(offset, dtype=float)
    r = diameter / 2.0
    i = np.arange(np.floor((-r - x0) / px), np.ceil((r - x0) / px))
    j = np.arange(np.floor((-r - y0) / py), np.ceil((r - y0) / py))
    shifts = np.array([[a * px + offset[0], b * py + offset[1], 0.0] for a in i for b in j])
    Vs = (V[None, :, :] + shifts[:, None, :]).reshape(-1, 3)
    Ts = (T[None, :, :] + len(V) * np.arange(len(shifts))[:, None, None]).reshape(-1, 3)
    return _weld(Vs, Ts)


def _clip_to_polygon(V, T, radius, segments, chunk=4096):
    """Clip triangles to the regular ``segments``-gon inscribed in the circle of ``radius`` (Sutherland–Hodgman, one plane at a time).

    Triangles inside every half-space are kept, those outside any one are dropped, and the rest are clipped and fanned.
    Intersections are computed from the lexicographically smaller end of each edge, so that neighbours get the same point.
    """
    angles = 2 * np.pi * (np.arange(segments) + 0.5) / segments
    normals = np.column_stack([np.cos(angles), np.sin(angles)])
    h = radius * np.cos(np.pi / segments)
    eps = 1e-12 * max(1.0, radius)
    keep, clip = [], []
    for start in range(0, len(T), chunk):
        tri = T[start : start + chunk]
        d = V[tri][:, :, :2] @ normals.T - h  # (triangles, 3, planes)
        out = d > eps
        inside = ~out.any(axis=(1, 2))
        dropped = out.all(axis=1).any(axis=1)
        keep.append(tri[inside])
        clip.extend(start + np.flatnonzero(~inside & ~dropped))
    points, triangles = [V], list(keep)
    n = len(V)
    for t in clip:
        polygon = [tuple(p) for p in V[T[t]]]
        for k in np.flatnonzero((V[T[t]][:, :2] @ normals.T - h > eps).any(axis=0)):
            polygon = _clip(polygon, normals[k], h, eps)
            if len(polygon) < 3:
                break
        if len(polygon) >= 3:
            points.append(np.array(polygon))
            triangles.append(np.array([[n, n + i, n + i + 1] for i in range(1, len(polygon) - 1)]))
            n += len(polygon)
    return _weld(np.vstack(points), np.vstack(triangles).astype(int))


def _clip(polygon, normal, h, eps):
    """Keep the part of a planar polygon where ``x·n ≤ h`` (one Sutherland–Hodgman step)."""
    result = []
    for a, b in zip(polygon, polygon[1:] + polygon[:1]):
        da, db = a[0] * normal[0] + a[1] * normal[1] - h, b[0] * normal[0] + b[1] * normal[1] - h
        if da <= eps:
            result.append(a)
        if (da <= eps) != (db <= eps):
            p, q, dp, dq = (a, b, da, db) if a < b else (b, a, db, da)
            s = dp / (dp - dq)
            result.append(tuple(pi + s * (qi - pi) for pi, qi in zip(p, q)))
    return result


def _edge_counts(T):
    """The undirected edges of a triangle mesh and how many triangles use each."""
    edges = np.sort(T[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2), axis=1)
    return np.unique(edges, axis=0, return_counts=True)


def _boundary_loops(T):
    """The boundary edges (used by one triangle) ordered into loops, each in the direction of its triangles."""
    directed = T[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2)
    edges, counts = _edge_counts(T)
    boundary = {tuple(e) for e in edges[counts == 1]}
    following = {}
    for a, b in directed:
        if (min(a, b), max(a, b)) in boundary:
            if a in following:
                raise ValueError(_OUTLINE.format("a pinched outline"))
            following[a] = b
    loops = []
    while following:
        start, loop = next(iter(following)), []
        v = start
        while v in following:
            loop.append(v)
            v = following.pop(v)
        if v != start:
            raise ValueError(_OUTLINE.format("an outline that does not close"))
        loops.append(np.array(loop))
    return loops


def _orient(V, T):
    """Make the triangles consistently oriented, facing +z on the whole."""
    directed = T[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2)
    if len(np.unique(directed, axis=0)) < len(directed):
        mesh = _to_mesh(V, T)
        mesh.unify_cycles()
        T = np.array([mesh.face_vertices(f) for f in mesh.faces()], dtype=int)
    if np.cross(V[T[:, 1]] - V[T[:, 0]], V[T[:, 2]] - V[T[:, 0]])[:, 2].sum() < 0:
        T = T[:, ::-1]
    return T


def _close(V, T, baseplate):
    """Close a relief: lowest point at z = 0, walls from its one outline down to ``z = −baseplate`` and a bottom.

    Returns the sample, its flat reference and the structural depth (ISO 17497-1 §6.3.2; Heimes & Vorländer §3.2).
    """
    T = _orient(V, T)
    loops = _boundary_loops(T)
    if len(loops) != 1:
        raise ValueError(_OUTLINE.format(len(loops)))
    V = V - [0.0, 0.0, V[:, 2].min()]
    loop = loops[0]
    sample = _to_mesh(*_walls_and_bottom(V, T, loop, baseplate))
    return sample, _flat_body(V[loop, :2], baseplate), float(V[:, 2].max())


def _walls_and_bottom(V, T, loop, baseplate):
    """Add vertical walls from ``loop`` (in its triangles' direction) down to ``z = −baseplate`` and a bottom facing −z."""
    n, m = len(V), len(loop)
    low = V[loop].copy()
    low[:, 2] = -baseplate
    a, b = loop, np.roll(loop, -1)
    a2, b2 = n + np.arange(m), n + np.roll(np.arange(m), -1)
    walls = np.vstack([np.column_stack([b, a, a2]), np.column_stack([b, a2, b2])])
    extra, cap = _cap(low[:, :2])
    extra = np.column_stack([extra, np.full(len(extra), -baseplate)]).reshape(-1, 3)
    bottom = (n + cap)[:, ::-1]
    return np.vstack([V, low, extra]), np.vstack([T, walls, bottom])


def _flat_body(outline, baseplate):
    """A flat body on ``outline`` (xy, counter-clockwise or not): top at z = 0, walls and bottom at ``z = −baseplate``."""
    if _signed_area(outline) < 0:
        outline = outline[::-1]
    extra, cap = _cap(outline)
    top = np.column_stack([np.vstack([outline, extra.reshape(-1, 2)]), np.zeros(len(outline) + len(extra))])
    return _to_mesh(*_walls_and_bottom(top, cap, np.arange(len(outline)), baseplate))


def _cap(outline):
    """Triangulate a polygon given counter-clockwise: a fan from its centroid if it sees every edge, otherwise ear clipping.

    Returns the extra points (the centroid, or none) and counter-clockwise triangles, indexing the outline then the extra points.
    """
    m = len(outline)
    a, b = outline, np.roll(outline, -1, axis=0)
    cross = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
    area = cross.sum() / 2
    centroid = ((a + b) * cross[:, None]).sum(axis=0) / (6 * area)
    da, db = a - centroid, b - centroid
    if (da[:, 0] * db[:, 1] - da[:, 1] * db[:, 0] > 1e-12 * abs(area)).all():
        return centroid[None, :], np.column_stack([np.full(m, m), np.arange(m), np.roll(np.arange(m), -1)])
    cap = np.array(earclip_polygon(Polygon([[x, y, 0.0] for x, y in outline])), dtype=int)
    p = outline[cap]
    flip = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]) < 0
    cap[flip] = cap[flip][:, ::-1]
    return np.zeros((0, 2)), cap


def _bottom_loop(V, T):
    """The outline of a closed mesh's lowest faces: those at z = 0 that face −z."""
    p = V[T]
    normals = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
    lowest = (np.abs(p[:, :, 2]) < _WELD).all(axis=1) & (normals[:, 2] < 0)
    if not lowest.any():
        raise ValueError("a closed mesh needs a flat bottom at its lowest point, to outline its reference")
    loops = _boundary_loops(T[lowest])
    if len(loops) != 1:
        raise ValueError(_OUTLINE.format(len(loops)))
    return loops[0]


def _signed_area(outline):
    return float((outline[:, 0] * np.roll(outline[:, 1], -1) - np.roll(outline[:, 0], -1) * outline[:, 1]).sum() / 2)


def _signed_volume(V, T):
    p = V[T]
    return float(np.einsum("ij,ij->i", p[:, 0], np.cross(p[:, 1], p[:, 2])).sum() / 6)


def _outline_diameter(reference):
    """The largest xy distance between a body's vertices, over their convex hull."""
    xy = np.array([reference.vertex_coordinates(v)[:2] for v in reference.vertices()])
    hull = xy[ConvexHull(xy).vertices]
    return float(np.sqrt(((hull[:, None, :] - hull[None, :, :]) ** 2).sum(axis=2)).max())


def _to_mesh(V, T):
    return Mesh.from_vertices_and_faces(V.tolist(), T.tolist())


def _warnings(depth, diameter):
    """ISO 17497-1 §6.3.2's depth limit ``h ≤ d/16``, and a size that suggests a model in millimetres."""
    warnings = []
    if depth > diameter / 16:
        warnings.append("ISO 17497-1 6.3.2: structural depth {:.4g} m exceeds d/16 = {:.4g} m".format(depth, diameter / 16))
    if diameter > 20:
        warnings.append("the sample is {:.4g} m across; is the model in millimetres?".format(diameter))
    return warnings
