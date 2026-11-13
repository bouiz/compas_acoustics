"""Diffracting edges found from the air around each edge (the sector detector), with numpy and scipy only.

:func:`sector_edges` needs no roles, no closed boundary and no manifold mesh. Every side of every polygon is split at
the vertices of the other polygons lying on it, so sides are matched by position, not by vertex index. Around each
piece the faces meeting there (and any polygon whose plane passes through it, a box standing on an unsplit floor)
cut the full turn into angular sectors, and each sector is air or solid as the faces bounding it say: a one-sided
face is solid on its normal's side, a two-sided face is air on both, a face the air never reaches is solid on both.
An air sector opening wider than pi is a diffracting wedge; a free edge of a two-sided face is a knife edge.

Two adapters feed it: :func:`sector_edges_from_room` (a :class:`~compas_acoustics.RoomGeometry`, giving
:class:`~compas_acoustics.hybrid.diffraction.Edge`) and :func:`sector_edges_from_prepared` (a
:class:`~compas_acoustics.geometry.PreparedMesh`, its sides read by :func:`prepared_wall_sides`). A room built by
:meth:`~compas_acoustics.RoomGeometry.from_prepared` carries those sides, and
:func:`~compas_acoustics.hybrid.diffraction.diffraction_edges` finds its edges here. Importing this module loads
neither pyroomacoustics nor COMPAS.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations
from itertools import product
from typing import TYPE_CHECKING
from typing import Dict
from typing import List
from typing import NamedTuple
from typing import Optional
from typing import Sequence
from typing import Tuple

import numpy as np
from scipy.spatial import cKDTree

if TYPE_CHECKING:
    from compas_acoustics.room.geometry import RoomGeometry

    from .prepare import PreparedMesh

__all__ = ["RawEdge", "SectorDiagnostics", "SectorEdges", "sector_edges", "sector_edges_from_room", "sector_edges_from_prepared", "prepared_wall_sides"]

TWO_PI = 2.0 * np.pi

#: Half-planes closer than this, in radians, are one (coincident faces); a sector within it of pi is a seam.
_COINCIDENT_ANGLE = 1e-5
#: Most combinations of coincident-face readings tried at one piece before the solid reading is taken for all.
_MAX_ASSIGNMENTS = 4096


@dataclass(frozen=True)
class RawEdge:
    """A diffracting edge of :func:`sector_edges`, in the conventions of
    :class:`~compas_acoustics.hybrid.diffraction.Edge`.

    ``face_a`` and ``face_b`` point from the edge into the two faces bounding the air sector, perpendicular to the
    edge; ``theta_w`` is the sector's angle from face A to face B (2 pi for a knife edge, whose faces coincide).
    ``faces_a`` and ``faces_b`` index the input polygons lying along face A and face B (several where faces
    coincide or where a face is split along the edge); they are equal on a knife edge.
    """

    start: np.ndarray
    end: np.ndarray
    face_a: np.ndarray
    face_b: np.ndarray
    theta_w: float
    faces_a: Tuple[int, ...]
    faces_b: Tuple[int, ...]

    @property
    def faces(self) -> Tuple[int, ...]:
        """Every polygon bounding the air sector, sorted."""
        return tuple(sorted(set(self.faces_a) | set(self.faces_b)))

    @property
    def length(self) -> float:
        return float(np.linalg.norm(self.end - self.start))


@dataclass
class SectorDiagnostics:
    """What :func:`sector_edges` left out, counted.

    Attributes
    ----------
    segments : int
        Pieces of polygon sides (after the split at T-junctions) examined.
    ambiguous : int
        Pieces where the faces bounding a sector disagree on whether it is air: no edge is emitted there.
    open_rims : int
        Pieces on a free side of a one-sided face (the rim of an opening): no edge.
    dropped_excess : int
        Merged edges opening wider than pi by no more than ``min_excess``.
    dropped_length : int
        Merged edges shorter than ``min_length``.
    """

    segments: int = 0
    ambiguous: int = 0
    open_rims: int = 0
    dropped_excess: int = 0
    dropped_length: int = 0


class SectorEdges(NamedTuple):
    """The edges of :func:`sector_edges` and its :class:`SectorDiagnostics`."""

    edges: List[RawEdge]
    diagnostics: SectorDiagnostics


def _weld(points: np.ndarray, tolerance: float) -> Tuple[np.ndarray, np.ndarray]:
    """Merge points closer than ``tolerance``, transitively, each cluster into its first point."""
    parent = np.arange(len(points))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, j in cKDTree(points).query_pairs(tolerance):
        a, b = root(i), root(j)
        if a != b:
            parent[max(a, b)] = min(a, b)
    roots = np.array([root(i) for i in range(len(points))])
    first, inverse = np.unique(roots, return_inverse=True)
    return points[first], inverse.reshape(-1)


def _newell(points: np.ndarray) -> Tuple[float, np.ndarray]:
    """Area and unit normal (Newell, from the winding) of a polygon; a zero normal for a degenerate one."""
    normal = 0.5 * np.cross(points, np.roll(points, -1, axis=0)).sum(axis=0)
    area = float(np.linalg.norm(normal))
    return area, (normal / area if area > 0.0 else np.zeros(3))


def _canonical_axis(d: np.ndarray) -> np.ndarray:
    """``d`` as a unit vector whose first non-zero component is positive."""
    axis = d / np.linalg.norm(d)
    first = int(np.flatnonzero(np.abs(axis) > 1e-9)[0])
    return axis if axis[first] > 0.0 else -axis


def _perpendicular(axis: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Two unit vectors completing ``axis`` to a right-handed frame."""
    helper = np.eye(3)[int(np.argmin(np.abs(axis)))]
    e1 = np.cross(axis, helper)
    e1 /= np.linalg.norm(e1)
    return e1, np.cross(axis, e1)


def _split_sides(vertices: np.ndarray, faces: Sequence[Sequence[int]], tolerance: float) -> Dict[Tuple[int, int], List[Tuple[int, int]]]:
    """Every side of the faces, split at the vertices lying on it: {(a, b) with a < b: [(face, +1 or -1)]}.

    +1 when the face runs from ``a`` to ``b`` in its winding. Vertices are found with a KD-tree, within
    ``tolerance`` of the side and strictly between its ends.
    """
    tree = cKDTree(vertices)
    out: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}
    for i, face in enumerate(faces):
        for a, b in zip(face, list(face[1:]) + list(face[:1])):
            p0, d = vertices[a], vertices[b] - vertices[a]
            length2 = float(d @ d)
            near = np.array(tree.query_ball_point(p0 + d / 2.0, np.sqrt(length2) / 2.0 + tolerance), dtype=np.int64)
            near = near[(near != a) & (near != b)]
            chain = [a, b]
            if near.size:
                offsets = vertices[near] - p0
                t = offsets @ d / length2
                off = np.linalg.norm(offsets - np.outer(t, d), axis=1)
                keep = (t > 0.0) & (t < 1.0) & (off <= tolerance)
                chain = [a, *near[keep][np.argsort(t[keep])].tolist(), b]
            for v, w in zip(chain[:-1], chain[1:]):
                if v == w:
                    continue
                key = (v, w) if v < w else (w, v)
                out.setdefault(key, []).append((i, 1 if v < w else -1))
    return out


def _plane_basis(normal: np.ndarray) -> np.ndarray:
    e1, e2 = _perpendicular(normal)
    return np.stack([e1, e2], axis=1)


def _strictly_inside(polygon2d: np.ndarray, points2d: np.ndarray, margin: float) -> np.ndarray:
    """Points inside the polygon (crossing number) and further than ``margin`` from its outline."""
    x, y = points2d[:, :1], points2d[:, 1:]
    x1, y1 = polygon2d[None, :, 0], polygon2d[None, :, 1]
    nxt = np.roll(polygon2d, -1, axis=0)
    x2, y2 = nxt[None, :, 0], nxt[None, :, 1]
    straddle = (y1 > y) != (y2 > y)
    with np.errstate(divide="ignore", invalid="ignore"):
        cross_x = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
    inside = (np.sum(straddle & (x < cross_x), axis=1) % 2) == 1
    d = nxt - polygon2d
    dd = np.einsum("ij,ij->i", d, d)
    rel = points2d[:, None, :] - polygon2d[None, :, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.clip(np.where(dd > 0.0, np.einsum("pij,ij->pi", rel, d) / dd, 0.0), 0.0, 1.0)
    distance = np.linalg.norm(rel - t[:, :, None] * d[None, :, :], axis=2).min(axis=1)
    return inside & (distance > margin)


def _through(
    polygons: List[np.ndarray],
    winding: np.ndarray,
    valid: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    owners: List[set],
    tolerance: float,
    contact: float,
) -> List[List[int]]:
    """For each piece, the polygons whose plane holds it (both ends within ``contact``) and whose interior holds its
    midpoint (further than ``tolerance`` from the outline), other than the polygons it is a side of."""
    out: List[List[int]] = [[] for _ in range(len(starts))]
    if not len(starts):
        return out
    mids = (starts + ends) / 2.0
    tree = cKDTree(mids)
    for i, polygon in enumerate(polygons):
        if not valid[i]:
            continue
        n = winding[i]
        centre = polygon.mean(axis=0)
        radius = float(np.max(np.linalg.norm(polygon - centre, axis=1)))
        near = np.array(tree.query_ball_point(centre, radius + contact), dtype=np.int64)
        if not near.size:
            continue
        offset = float(n @ centre)
        keep = (np.abs(starts[near] @ n - offset) <= contact) & (np.abs(ends[near] @ n - offset) <= contact)
        near = near[keep]
        near = np.array([s for s in near if i not in owners[s]], dtype=np.int64)
        if not near.size:
            continue
        basis = _plane_basis(n)
        inside = _strictly_inside((polygon - centre) @ basis, (mids[near] - centre) @ basis, tolerance)
        for s in near[inside]:
            out[int(s)].append(i)
    return out


@dataclass
class _HalfPlane:
    angle: float
    u: np.ndarray
    polygon: int
    cw_air: bool
    ccw_air: bool


def _cluster_options(members: List[_HalfPlane]) -> List[Tuple[bool, bool]]:
    """The (cw, ccw) air states a cluster of coincident half-planes can show its two neighbouring sectors.

    Coincident faces are a stack of infinitely thin layers. Every order of the stack in which neighbouring layers
    agree on what lies between them gives the states of its outer sides; and the faces may also touch (a box
    standing on a floor, whose bottom and the floor coincide), where a side is air only if every face says so.
    """
    patterns = sorted({(m.cw_air, m.ccw_air) for m in members})
    options = {(all(p[0] for p in patterns), all(p[1] for p in patterns))}
    for order in permutations(patterns):
        if all(lower[1] == upper[0] for lower, upper in zip(order[:-1], order[1:])):
            options.add((order[0][0], order[-1][1]))
    return sorted(options)


def _state(u: np.ndarray, axis: np.ndarray, polygon: int, side_normal: np.ndarray, two_sided: np.ndarray, reached: np.ndarray) -> Tuple[bool, bool]:
    """(cw, ccw) air of the face of ``polygon`` running from the edge along ``u``."""
    if not reached[polygon]:
        return False, False
    if two_sided[polygon]:
        return True, True
    ccw_on_normal_side = float(np.cross(axis, u) @ side_normal[polygon]) > 0.0
    return ccw_on_normal_side, not ccw_on_normal_side


def sector_edges(
    polygons: Sequence,
    normals,
    two_sided,
    reached=None,
    *,
    min_excess: float = 1e-4,
    min_length: float = 0.0,
    tolerance: Optional[float] = None,
    contact: Optional[float] = None,
) -> SectorEdges:
    """The diffracting edges of a set of planar polygons, from the air sectors around each of their sides.

    Parameters
    ----------
    polygons : sequence of (k, 3) array-like
        Planar polygons, in metres, wound any way; nothing is assumed welded, closed or manifold.
    normals : (n, 3) array-like
        For a one-sided polygon, its normal pointing AWAY from the air (into the solid); only its side relative
        to the polygon's plane is used.
    two_sided : (n,) array-like of bool
        Air on both sides of the polygon.
    reached : (n,) array-like of bool, optional
        False where the air never touches the polygon: both of its sides count as solid. Default all True.
    min_excess : float, default=1e-4
        In radians: an air sector must open wider than pi by more than this to diffract (a tessellated curve's
        facets open by 2 pi / sides).
    min_length : float, default=0
        In metres: shorter merged edges are dropped.
    tolerance : float, optional
        In metres: vertices closer than this weld, and a vertex this close to a side splits it. Default 1e-6 of
        the model's extent, as :class:`~compas_acoustics.RoomGeometry` splits sides at T-junctions.
    contact : float, optional
        In metres: a polygon whose plane passes within this of a side (both ends) and whose interior holds its
        midpoint stands in for two opposite half-planes there, as if it ran through the edge. Default
        ``tolerance``.

    Returns
    -------
    SectorEdges
        The :class:`RawEdge` list and the :class:`SectorDiagnostics`.

    Notes
    -----
    Around each piece the half-planes (one per polygon it is a side of, two per polygon it runs through) are
    sorted by angle about the piece; half-planes within ``1e-5`` rad of one another are one (coincident faces,
    see :func:`_cluster_options`). Each sector between neighbouring half-planes is air when the faces bounding it
    both say air. Where coincident faces can be read several ways, the reading under which every sector's two
    faces agree is taken, the one with the most air if several do. Where no reading agrees the piece is
    ambiguous and gives no edge. A piece that is the side of one face only gives a knife edge (2 pi) when that
    face is two-sided and reached, and nothing when it is one-sided (an opening's rim). Every air sector opening
    wider than ``pi + min_excess`` is a wedge. Collinear pieces that share an end and their two face directions
    and angle are merged into one straight edge.
    """
    polys = [np.asarray(p, dtype=float).reshape(-1, 3) for p in polygons]
    n = len(polys)
    diagnostics = SectorDiagnostics()
    if n == 0:
        return SectorEdges([], diagnostics)
    normals = np.asarray(normals, dtype=float).reshape(n, 3)
    two_sided = np.asarray(two_sided, dtype=bool).reshape(n)
    reached = np.ones(n, dtype=bool) if reached is None else np.asarray(reached, dtype=bool).reshape(n)
    points = np.vstack(polys)
    extent = float(np.ptp(points, axis=0).max())
    tolerance = max(1e-6 * extent if tolerance is None else float(tolerance), 1e-12)
    contact = tolerance if contact is None else max(float(contact), tolerance)

    vertices, inverse = _weld(points, tolerance)
    faces: List[List[int]] = []
    winding = np.zeros((n, 3))
    valid = np.zeros(n, dtype=bool)
    start = 0
    for i, polygon in enumerate(polys):
        ids = [int(v) for v in inverse[start : start + len(polygon)]]
        start += len(polygon)
        ids = [v for k, v in enumerate(ids) if v != ids[k - 1]]
        area, winding[i] = _newell(polygon)
        valid[i] = len(set(ids)) >= 3 and area > 0.0
        faces.append(ids if valid[i] else [])
    # the side of the polygon's own plane its normal points to, as a unit normal of the winding
    side_normal = winding * np.sign(np.einsum("ij,ij->i", winding, normals))[:, None]

    sides = _split_sides(vertices, faces, tolerance)
    keys = list(sides)
    diagnostics.segments = len(keys)
    starts = np.array([vertices[a] for a, _ in keys]).reshape(-1, 3)
    ends = np.array([vertices[b] for _, b in keys]).reshape(-1, 3)
    owners = [{i for i, _ in sides[k]} for k in keys]
    through = _through(polys, winding, valid, starts, ends, owners, tolerance, contact)

    records = []  # (a, b, axis, u_a, u_b, theta, faces_a, faces_b)
    for s, key in enumerate(keys):
        p, q = starts[s], ends[s]
        axis = _canonical_axis(q - p)
        e1, e2 = _perpendicular(axis)
        planes: List[_HalfPlane] = []

        def add(u, polygon):
            u = u - (u @ axis) * axis
            u = u / np.linalg.norm(u)
            cw, ccw = _state(u, axis, polygon, side_normal, two_sided, reached)
            planes.append(_HalfPlane(float(np.arctan2(u @ e2, u @ e1)) % TWO_PI, u, polygon, cw, ccw))

        direction = (q - p) / np.linalg.norm(q - p)
        for polygon, sense in sides[key]:
            add(np.cross(winding[polygon], sense * direction), polygon)
        for polygon in through[s]:
            u = np.cross(winding[polygon], axis)
            add(u, polygon)
            add(-u, polygon)
        planes.sort(key=lambda h: h.angle)

        clusters: List[List[_HalfPlane]] = [[planes[0]]]
        for h in planes[1:]:
            if h.angle - clusters[-1][-1].angle <= _COINCIDENT_ANGLE:
                clusters[-1].append(h)
            else:
                clusters.append([h])
        if len(clusters) > 1 and clusters[0][0].angle + TWO_PI - clusters[-1][-1].angle <= _COINCIDENT_ANGLE:
            clusters[0] = clusters.pop() + clusters[0]
        options = [_cluster_options(c) for c in clusters]
        members = [tuple(sorted({h.polygon for h in c})) for c in clusters]

        if len(clusters) == 1:
            if (True, True) in options[0]:
                u = clusters[0][0].u
                records.append((key[0], key[1], axis, u, u, TWO_PI, members[0], members[0]))
            elif (False, False) not in options[0]:
                diagnostics.open_rims += 1
            continue

        k = len(clusters)
        angles = [c[0].angle for c in clusters]
        widths = [(angles[(i + 1) % k] - angles[i]) % TWO_PI for i in range(k)]
        best = None
        n_assignments = int(np.prod([len(o) for o in options]))
        choices = product(*options) if n_assignments <= _MAX_ASSIGNMENTS else [tuple(o[0] for o in options)]
        for choice in choices:
            disagree, air = 0, 0.0
            for i in range(k):
                left, right = choice[i][1], choice[(i + 1) % k][0]
                if left != right:
                    disagree += 1
                elif left:
                    air += widths[i]
            score = (disagree, -air)
            if best is None or score < best[0]:
                best = (score, choice)
        (disagree, _), choice = best
        if disagree:
            diagnostics.ambiguous += 1
            continue
        for i in range(k):
            j = (i + 1) % k
            if choice[i][1] and choice[j][0] and widths[i] > np.pi + _COINCIDENT_ANGLE:
                records.append((key[0], key[1], axis, clusters[i][0].u, clusters[j][0].u, widths[i], members[i], members[j]))

    edges = []
    for group in _merge(vertices, records, tolerance):
        longest = max(group, key=lambda r: np.linalg.norm(vertices[r[1]] - vertices[r[0]]))
        axis = longest[2]
        ends_ = {v for r in group for v in r[:2]}
        along = {v: float(vertices[v] @ axis) for v in ends_}
        p, q = vertices[min(ends_, key=along.get)], vertices[max(ends_, key=along.get)]
        theta = float(longest[5])
        if theta <= np.pi + min_excess:
            diagnostics.dropped_excess += 1
            continue
        if np.linalg.norm(q - p) < max(min_length, 1e-12):
            diagnostics.dropped_length += 1
            continue
        final = (q - p) / np.linalg.norm(q - p)
        ua, ub = (u - (u @ final) * final for u in longest[3:5])
        ua, ub = ua / np.linalg.norm(ua), ub / np.linalg.norm(ub)
        faces_a = tuple(sorted({f for r in group for f in r[6]}))
        faces_b = tuple(sorted({f for r in group for f in r[7]}))
        edges.append(RawEdge(p.copy(), q.copy(), ua, ub, theta, faces_a, faces_b))
    return SectorEdges(edges, diagnostics)


def _merge(vertices: np.ndarray, records: list, tolerance: float) -> List[list]:
    """Records joined where they share an end, lie on one line and have the same two face directions and angle."""
    parent = list(range(len(records)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    at: Dict[int, List[int]] = {}
    for k, r in enumerate(records):
        at.setdefault(r[0], []).append(k)
        at.setdefault(r[1], []).append(k)
    cos_tol = np.cos(_COINCIDENT_ANGLE)
    for shared, ks in at.items():
        for x, k in enumerate(ks):
            for m in ks[x + 1 :]:
                a, b = records[k], records[m]
                if a[2] @ b[2] < 1.0 - 1e-10 or abs(a[5] - b[5]) > _COINCIDENT_ANGLE:
                    continue
                if a[3] @ b[3] < cos_tol or a[4] @ b[4] < cos_tol:
                    continue
                other = b[1] if b[0] == shared else b[0]
                v = vertices[other] - vertices[shared]
                if np.linalg.norm(v - (v @ a[2]) * a[2]) > tolerance:
                    continue
                parent[find(m)] = find(k)
    groups: Dict[int, list] = {}
    for k in range(len(records)):
        groups.setdefault(find(k), []).append(records[k])
    return list(groups.values())


def _into_solid(geometry: RoomGeometry, faces: List[int]) -> Optional[np.ndarray]:
    """Unit normals pointing into the volume an obstacle component closes, from its consistent winding; None when
    it closes none."""
    from compas_acoustics.room.geometry import _orient
    from compas_acoustics.room.geometry import _signed_volume

    polygons = [list(geometry.obstacle_faces[i]) for i in faces]
    flips, _, defect = _orient(geometry.vertices, polygons)
    if defect is not None:
        return None
    wound = [p[::-1] if flip else p for p, flip in zip(polygons, flips)]
    sign = 1.0 if _signed_volume(geometry.vertices, wound) > 0.0 else -1.0
    return np.array([-sign * _newell(geometry.vertices[w])[1] for w in wound])


def _boundary_regions(polygons: List[np.ndarray], normals: np.ndarray, tolerance: float, plane_tolerance: float) -> np.ndarray:
    """The coplanar connected surface of each boundary polygon, labelled by its smallest polygon index: polygons
    sharing a piece of a side, facing the same way and lying in one plane join."""
    if not polygons:
        return np.zeros(0, dtype=np.int64)
    points = np.vstack(polygons)
    vertices, inverse = _weld(points, tolerance)
    faces, start = [], 0
    for polygon in polygons:
        ids = [int(v) for v in inverse[start : start + len(polygon)]]
        start += len(polygon)
        faces.append([v for k, v in enumerate(ids) if v != ids[k - 1]])
    parent = list(range(len(polygons)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def coplanar(i, j):
        return normals[i] @ normals[j] > 0.0 and float(np.max(np.abs((polygons[j] - polygons[i][0]) @ normals[i]))) <= plane_tolerance

    for users in _split_sides(vertices, faces, tolerance).values():
        found = sorted({i for i, _ in users})
        for x, i in enumerate(found):
            for j in found[x + 1 :]:
                if coplanar(i, j):
                    a, b = find(i), find(j)
                    parent[max(a, b)] = min(a, b)
    return np.array([find(i) for i in range(len(polygons))], dtype=np.int64)


def sector_edges_from_room(
    geometry: RoomGeometry,
    *,
    min_excess: float = 1e-4,
    min_length: float = 0.0,
    tolerance: Optional[float] = None,
    contact: Optional[float] = None,
    return_diagnostics: bool = False,
):
    """The diffracting edges of a room by :func:`sector_edges`, as :class:`~compas_acoustics.hybrid.diffraction.Edge`.

    The polygons are those of :class:`~compas_acoustics.hybrid.diffraction.Occluders`, in its order: the obstacle
    polygons as drawn (``geometry.obstacle_faces``), then the boundary faces. A room built from a prepared mesh
    carries which sides of each the air touches (``geometry.drawn_sides``, see
    :meth:`~compas_acoustics.RoomGeometry.from_prepared`), and they are used as they are. Otherwise the sides are
    read from the roles: boundary faces are one-sided with their stored winding (normal out of the air; as drawn
    when the boundary does not close); an obstacle component
    (:func:`~compas_acoustics.hybrid.diffraction.obstacle_components`) that closes a volume is one-sided with its
    normals into that volume, and any other obstacle polygon is two-sided. Nothing depends on ``geometry.is_closed``.

    ``Edge.faces`` indexes those polygons; ``Edge.component`` is the obstacle component of the edge's faces (the
    smallest if they belong to several), -1 when they are all boundary faces; ``Edge.regions`` is set on edges whose
    faces are all boundary faces: the coplanar connected surface of face A and of face B (labelled by its smallest
    boundary-face index), as :func:`~compas_acoustics.hybrid.diffraction.enclosure_edges` sets it.

    ``tolerance`` defaults to the room's T-junction tolerance (1e-6 of its extent) and ``contact`` to twice
    ``OBSTACLE_OFFSET``, the distance at which :func:`~compas_acoustics.hybrid.diffraction.obstacle_edges` takes an
    edge to lie on another polygon. ``return_diagnostics=True`` returns ``(edges, SectorDiagnostics)``.
    """
    from compas_acoustics.hybrid.diffraction import Edge
    from compas_acoustics.hybrid.diffraction import obstacle_components
    from compas_acoustics.room.geometry import _MERGE_TOLERANCE
    from compas_acoustics.room.geometry import OBSTACLE_OFFSET

    n_obstacle = len(geometry.obstacle_faces)
    n_boundary = len(geometry.faces) - 2 * n_obstacle
    faces = list(geometry.obstacle_faces) + list(geometry.faces[:n_boundary])
    polygons = [geometry.vertices[list(f)] for f in faces]
    normals = np.array([_newell(p)[1] for p in polygons]).reshape(-1, 3)
    two_sided = np.zeros(len(polygons), dtype=bool)
    reached = None
    component = np.full(len(polygons), -1, dtype=np.int64)
    sides = getattr(geometry, "drawn_sides", None)
    for c, members in enumerate(obstacle_components(geometry) if n_obstacle else []):
        component[members] = c
        if sides is not None:
            continue
        inward = _into_solid(geometry, members)
        if inward is None:
            two_sided[members] = True
        else:
            normals[members] = inward
    if sides is not None:
        if len(sides.two_sided) != len(polygons):
            raise ValueError(f"drawn_sides holds {len(sides.two_sided)} polygons, the room draws {len(polygons)}")
        normals = np.asarray(sides.normals, dtype=float).reshape(-1, 3)
        two_sided = np.asarray(sides.two_sided, dtype=bool)
        reached = np.asarray(sides.reached, dtype=bool)
    if tolerance is None:
        tolerance = 1e-6 * float(np.ptp(geometry.vertices, axis=0).max())
    if contact is None:
        contact = 2.0 * OBSTACLE_OFFSET
    raw, diagnostics = sector_edges(polygons, normals, two_sided, reached, min_excess=min_excess, min_length=min_length, tolerance=tolerance, contact=contact)
    regions = _boundary_regions(polygons[n_obstacle:], normals[n_obstacle:], tolerance, _MERGE_TOLERANCE)

    def region(members):
        return int(regions[min(members) - n_obstacle])

    edges = []
    for e in raw:
        components = {int(component[i]) for i in e.faces if i < n_obstacle}
        owned: Tuple[int, ...] = ()
        if not components:
            if e.theta_w == TWO_PI:
                # a knife edge of two opposite boundary faces: one region per face, ordered by normal as enclosure_edges does
                labels = sorted({int(regions[i - n_obstacle]) for i in e.faces_a}, key=lambda r: tuple(np.round(normals[n_obstacle + r], 6)))
                owned = tuple(labels)
            else:
                owned = (region(e.faces_a), region(e.faces_b))
        edges.append(Edge(e.start, e.end, e.face_a, e.face_b, e.theta_w, e.faces, min(components) if components else -1, owned))
    return (edges, diagnostics) if return_diagnostics else edges


def sector_edges_from_prepared(
    prepared: PreparedMesh,
    *,
    min_excess: float = 1e-4,
    min_length: float = 0.0,
    tolerance: Optional[float] = None,
    contact: Optional[float] = None,
) -> SectorEdges:
    """The diffracting edges of a prepared mesh by :func:`sector_edges`, over its merged walls.

    Each wall (``prepared.walls``) is one polygon, its sides as :func:`prepared_wall_sides` reads them.
    ``RawEdge.faces_a`` and ``faces_b`` index ``prepared.walls``.
    """
    polygons = [np.asarray(w.corners, dtype=float) for w in prepared.walls]
    normals, two_sided, reached = prepared_wall_sides(prepared)
    return sector_edges(polygons, normals, two_sided, reached, min_excess=min_excess, min_length=min_length, tolerance=tolerance, contact=contact)


def prepared_wall_sides(prepared: PreparedMesh) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Which sides of each wall of a prepared mesh (``prepared.walls``) the air touches: ``(normals, two_sided,
    reached)``, as :func:`sector_edges` takes them.

    A wall's normal is its faces' (away from the air); it is two-sided when its faces are, except on an object
    standing in the air that closes a volume (:func:`~compas_acoustics.geometry.standing_objects` on a closed welded
    component), whose air is outside only, even when its faces were made two-sided (``force_two_sided``); it is
    reached when any of its faces was hit from either side by the visibility walk (``air_hits + other_hits > 0``).
    """
    from .prepare import standing_objects

    walls = prepared.walls
    first = np.array([int(w.faces[0]) for w in walls], dtype=np.int64)
    normals = np.asarray(prepared.normals, dtype=float)[first].reshape(-1, 3)
    solid = standing_objects(prepared) & np.asarray(prepared.component_closed, dtype=bool)[np.asarray(prepared.component, dtype=np.int64)]
    face_two_sided = np.asarray(prepared.two_sided, dtype=bool) & ~solid
    two_sided = np.array([bool(np.any(face_two_sided[w.faces])) for w in walls], dtype=bool)
    hits = np.asarray(prepared.air_hits) + np.asarray(prepared.other_hits)
    reached = np.array([bool(np.any(hits[w.faces] > 0)) for w in walls], dtype=bool)
    return normals, two_sided, reached
