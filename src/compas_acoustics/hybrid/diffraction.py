"""Edge diffraction around obstacles, as pseudo-sources of the early field.

Geometry first: which edges of a mesh's obstacles diffract
(:func:`obstacle_edges`), what blocks a path (:class:`Occluders`), and the
wedge-frame path over an edge (:func:`wedge_path`). The acoustics is in
:mod:`hybrid_rir.udfa`.

Every free edge of an obstacle diffracts by the universal diffraction filter
approximation (Kirsch and Ewert 2023), and an object is the sum of its
edges' paths, as the paper composes flat objects (Sec. VI). Higher-order
diffraction between edges is not rendered: the three-sided barrier of Kirsch
and Ewert (2024, :class:`hybrid_rir.udfa.DoubleEdgePath`) is defined for an
infinite double edge, and the papers give no rule for applying it to a
finite obstacle.

For ``max_order >= 1`` one enclosure-wall reflection is allowed before the
diffraction (the source's image in that wall) or after it (the receiver's
image).

Ported from hybrid-rir (gpapadimitri/hybrid-rir @ 8a70661, src/diffraction.py).
Reads a RoomGeometry, which carries upstream RoomGeometry's attribute names.
"""

from __future__ import annotations

import warnings
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial
from itertools import pairwise

import numpy as np
import pyfar as pf
import spharpy

from .btms import _coordtrans1, element_grid, wedge1st_ir, wedge2nd_ir
from .early import render_signals, render_transfers
from compas_acoustics.room_geometry import OBSTACLE_OFFSET, RoomGeometry, _polygon_area_and_centroid

from .bands import BANDS
from .pair import Pair
from .target import _air_absorption_coefficient
from .udfa import WedgePath, udfa_transfer

TWO_PI = 2.0 * np.pi

# Edges closer than this to an enclosure face or another obstacle are not free
# edges: a shelf on the floor has its bottom rim there, and the floor fills its
# open side.
ON_ENCLOSURE = 2.0 * OBSTACLE_OFFSET
_PARALLEL = 1.0 - 1e-8


@dataclass(frozen=True)
class Edge:
    """A straight rigid edge and its two faces.

    ``face_a`` and ``face_b`` point from the edge into each face,
    perpendicular to it. ``theta_w`` is the open (exterior) angle from face A
    to face B; for a knife edge the two faces coincide and it is 2 pi.
    ``faces`` indexes the polygons of :class:`Occluders` that meet at the edge.
    """

    start: np.ndarray
    end: np.ndarray
    face_a: np.ndarray
    face_b: np.ndarray
    theta_w: float
    faces: tuple[int, ...] = ()
    component: int = -1
    _w: np.ndarray = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        for name in ("start", "end", "face_a", "face_b"):
            object.__setattr__(self, name, np.asarray(getattr(self, name), dtype=float))
        if not np.pi < self.theta_w <= TWO_PI:
            raise ValueError("theta_w must lie in (pi, 2 pi]")
        if self.length <= 0.0:
            raise ValueError("an edge needs two distinct ends")
        w = np.cross(self.axis, self.face_a)
        angle = np.arctan2(self.face_b @ w, self.face_b @ self.face_a) % TWO_PI
        if not (np.isclose(angle, self.theta_w % TWO_PI, atol=1e-6)
                or np.isclose(self.theta_w, TWO_PI)):
            w = -w
        object.__setattr__(self, "_w", w)

    @property
    def length(self) -> float:
        return float(np.linalg.norm(self.end - self.start))

    @property
    def axis(self) -> np.ndarray:
        return (self.end - self.start) / self.length

    def point_at(self, z: float) -> np.ndarray:
        """The point at axial coordinate z, measured from the edge's centre."""
        return self.start + (z + self.length / 2.0) * self.axis

    def contains(self, point: np.ndarray) -> bool:
        """Does a point on the edge's line lie on the physical edge?"""
        z = float((np.asarray(point) - self.start) @ self.axis)
        return -1e-9 <= z <= self.length + 1e-9

    def cylindrical(self, point: np.ndarray) -> tuple[float, float, float]:
        """(r, z from the centre, open angle from face A) of a point."""
        v = np.asarray(point, dtype=float) - self.start
        along = float(v @ self.axis)
        v = v - along * self.axis
        r = float(np.linalg.norm(v))
        theta = float(np.arctan2(v @ self._w, v @ self.face_a)) % TWO_PI
        return r, along - self.length / 2.0, theta


def wedge_path(edge: Edge, source: np.ndarray, receiver: np.ndarray) -> WedgePath | None:
    """The wedge-frame path over ``edge``, or None if a point is in the solid.

    Angles are referenced to the face nearer the source (Kirsch and Ewert
    2023, Fig. 1): when the source is further from face A than the receiver
    is, both are measured from face B instead.
    """
    r_s, z_s, t_s = edge.cylindrical(source)
    r_r, z_r, t_r = edge.cylindrical(receiver)
    tol = 1e-9
    if r_s <= tol or r_r <= tol or t_s > edge.theta_w + tol or t_r > edge.theta_w + tol:
        return None
    t_s, t_r = min(t_s, edge.theta_w), min(t_r, edge.theta_w)
    if t_s > t_r:
        t_s, t_r = edge.theta_w - t_s, edge.theta_w - t_r
    return WedgePath(edge.theta_w, t_s, t_r, r_s, r_r, z_s, z_r, edge.length)


def toolbox_frame(edge: Edge) -> tuple[np.ndarray, np.ndarray]:
    """The edge's ends in EDtoolbox order and the reference face's normal into
    the air: x into face A, y = z cross x into the open region (R2)."""
    w = np.cross(edge.axis, edge.face_a)
    if w @ edge._w > 0.0:
        return np.array([edge.start, edge.end]), w
    return np.array([edge.end, edge.start]), np.cross(-edge.axis, edge.face_a)


def toolbox_cyl(edge: Edge, points: np.ndarray) -> np.ndarray:
    """(r, theta, z) of points in the toolbox's frame of the edge, (n, 3).

    Rounding puts points on face A's plane at theta = 2 pi - eps; on a wedge
    that is face A, theta = 0. On a knife edge 2 pi is the panel's other side.
    """
    ends, normal = toolbox_frame(edge)
    r, theta, z = _coordtrans1(np.atleast_2d(points), ends, normal)
    if edge.theta_w < TWO_PI - 1e-9:
        theta = np.where(theta > TWO_PI - 1e-9, 0.0, theta)
    return np.stack([r, theta, z], axis=1)


def edge_segments(edge: Edge, source: np.ndarray, receiver: np.ndarray) -> np.ndarray:
    """Segment limits (k, 2) along the edge in the toolbox's z, each
    subtending at most about 5 deg at the nearer of source and receiver, and
    at least 0.01 m long where a point is (nearly) on the edge's line."""
    ends, _ = toolbox_frame(edge)
    nearest = min(np.linalg.norm(np.cross(ends[1] - ends[0], ends[0] - p)) / edge.length
                  for p in (np.asarray(source), np.asarray(receiver)))
    n = max(1, int(np.ceil(edge.length / max(0.087 * nearest, 0.01))))
    z = np.linspace(0.0, edge.length, n + 1)
    return np.stack([z[:-1], z[1:]], axis=1)


def _face_planes(e: Edge) -> list[tuple[np.ndarray, float]]:
    """Unit normal and offset of the plane of each of the edge's two faces."""
    out = []
    for u in (e.face_a, e.face_b):
        n = np.cross(e.axis, u)
        n = n / np.linalg.norm(n)
        out.append((n, float(n @ e.start)))
    return out


def shared_face(e1: Edge, e2: Edge, tol: float = 1e-6) -> int | None:
    """The face of e1 (0 = A, 1 = B) whose plane is also a face plane of e2,
    or None. Edges of different obstacles share none. A face split into
    several polygons still counts, so the test is on the planes, not on the
    face indices; a knife edge's two faces are one plane, found as A."""
    if e1.component != e2.component:
        return None
    for k, (n1, d1) in enumerate(_face_planes(e1)):
        for n2, d2 in _face_planes(e2):
            sign = 1.0 if n1 @ n2 > 0.0 else -1.0
            if np.linalg.norm(n1 - sign * n2) < tol and abs(d1 - sign * d2) < tol:
                return k
    return None


def along_plane(e1: Edge, e2: Edge, tol: float = 1e-6) -> bool:
    """Do the two edges of one obstacle bound a common plane, so the path
    between them runs along a surface (R3)?"""
    return shared_face(e1, e2, tol) is not None


def coplanar_thetas(e1: Edge, e2: Edge) -> tuple[float, float]:
    """theta of e2 about e1 and of e1 about e2, for edges along a common
    plane, by face membership as EDed2geo.m assigns them: 0 in the reference
    face (A), theta_w in the other face. Not computed from points: an end on
    the other edge's line (a shared corner) has no angle, and on a knife edge
    rounding puts a point of the panel at 0 or 2 pi - eps.

    A knife edge's faces are one plane, and the toolbox gives every edge of a
    thin plate the same reference side (theta = 0 approached from the front
    plane's normal), so the path runs along one side of the panel. Here each
    edge's frame picks its side, so e1 sets it: 0 about e1, and about e2 0 if
    e2's frame has the same side, else 2 pi. Mirroring both frames leaves the
    second-order term unchanged."""
    def one(a, b):
        return 0.0 if shared_face(a, b) == 0 else a.theta_w

    if e1.theta_w < TWO_PI or e2.theta_w < TWO_PI:
        return one(e1, e2), one(e2, e1)
    same = toolbox_frame(e1)[1] @ toolbox_frame(e2)[1] > 0.0
    return 0.0, 0.0 if same else TWO_PI


def pair_valid(e1: Edge, e2: Edge, point1: np.ndarray, point2: np.ndarray) -> bool:
    """Is point2 (on e2) outside e1's solid and point1 (on e1) outside e2's?"""
    tol = 1e-9
    t2 = toolbox_cyl(e1, np.asarray(point2)[None, :])[0, 1]
    t1 = toolbox_cyl(e2, np.asarray(point1)[None, :])[0, 1]
    return bool(-tol <= t2 <= e1.theta_w + tol and -tol <= t1 <= e2.theta_w + tol)


class ObstacleEdgeWarning(UserWarning):
    """An obstacle component whose edges are not modelled."""


def _inside_polygon(points: np.ndarray, normal: np.ndarray, x: np.ndarray) -> bool:
    """Crossing-number test in the polygon's plane, dropping its largest normal axis."""
    keep = [i for i in range(3) if i != int(np.argmax(np.abs(normal)))]
    p, q = points[:, keep], np.asarray(x)[keep]
    inside = False
    for (x1, y1), (x2, y2) in zip(p, np.roll(p, -1, axis=0)):
        if (y1 > q[1]) != (y2 > q[1]) and q[0] < x1 + (q[1] - y1) * (x2 - x1) / (y2 - y1):
            inside = not inside
    return inside


@dataclass(frozen=True)
class Occluders:
    """Every polygon that can block a path: obstacles first, then the enclosure.

    Obstacle polygons keep the indices of ``RoomGeometry.obstacle_faces``;
    enclosure normals point out of the room.
    """

    polygons: tuple[np.ndarray, ...]
    normals: np.ndarray
    offsets: np.ndarray
    n_obstacle: int

    @classmethod
    def from_geometry(cls, geometry: RoomGeometry) -> Occluders:
        n_enclosure = len(geometry.faces) - 2 * len(geometry.obstacle_faces)
        faces = list(geometry.obstacle_faces) + list(geometry.faces[:n_enclosure])
        polygons = tuple(geometry.vertices[list(f)] for f in faces)
        normals = np.array([_polygon_area_and_centroid(p)[1] for p in polygons])
        offsets = np.einsum("ij,ij->i", normals, np.array([p[0] for p in polygons]))
        return cls(polygons, normals, offsets, len(geometry.obstacle_faces))

    def blocked(self, p: np.ndarray, q: np.ndarray, skip: frozenset[int] = frozenset()) -> bool:
        """Does the open segment p-q cross a polygon not in ``skip``?"""
        d = q - p
        denom = self.normals @ d
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (self.offsets - self.normals @ p) / denom
        hits = np.flatnonzero((np.abs(denom) > 1e-12) & (t > 1e-9) & (t < 1.0 - 1e-9))
        return any(
            int(i) not in skip and _inside_polygon(self.polygons[i], self.normals[i], p + t[i] * d)
            for i in hits
        )

    def on_enclosure(self, point: np.ndarray, tol: float) -> bool:
        return self.on_polygons(point, tol, range(self.n_obstacle, len(self.polygons)))

    def on_polygons(self, point: np.ndarray, tol: float, polygons) -> bool:
        """Does a point lie within ``tol`` of one of the given polygons?"""
        for i in polygons:
            if abs(self.normals[i] @ point - self.offsets[i]) < tol and _inside_polygon(
                self.polygons[i], self.normals[i], point
            ):
                return True
        return False


def _key(point: np.ndarray, scale: float) -> tuple[int, ...]:
    return tuple(np.round(point / scale).astype(np.int64))


def obstacle_components(geometry: RoomGeometry) -> list[list[int]]:
    """Obstacle polygons grouped by shared vertex positions."""
    scale = 1e-6 * float(np.ptp(geometry.vertices, axis=0).max())
    parent = list(range(len(geometry.obstacle_faces)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    owner: dict[tuple, int] = {}
    for i, face in enumerate(geometry.obstacle_faces):
        for v in face:
            k = _key(geometry.vertices[v], scale)
            if k in owner:
                parent[find(i)] = find(owner[k])
            else:
                owner[k] = i
    groups: dict[int, list[int]] = {}
    for i in range(len(parent)):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


def _segments(geometry: RoomGeometry, faces: list[int]):
    """Every side of the faces, split at T-junctions: {key: [p, q, owners]}.

    Only the component's own vertices split a side. The mesh's
    ``_split_at_t_junctions`` would also take the obstacle copies, whose
    corners sit OBSTACLE_OFFSET along a face normal and so exactly on the
    neighbouring face's edges.
    """
    scale = 1e-6 * float(np.ptp(geometry.vertices, axis=0).max())
    own = np.unique([v for i in faces for v in geometry.obstacle_faces[i]])
    points = geometry.vertices[own]
    out: dict[frozenset, list] = {}
    for i in faces:
        face = list(geometry.obstacle_faces[i])
        for a, b in zip(face, face[1:] + face[:1]):
            p0, d = geometry.vertices[a], geometry.vertices[b] - geometry.vertices[a]
            t = (points - p0) @ d / (d @ d)
            off = np.linalg.norm(points - p0 - np.outer(t, d), axis=1)
            inner = np.unique(t[(t > 1e-9) & (t < 1.0 - 1e-9) & (off < scale)])
            ts = np.concatenate([[0.0], inner, [1.0]])
            for t0, t1 in pairwise(ts):
                p, q = p0 + t0 * d, p0 + t1 * d
                if np.linalg.norm(q - p) < scale:
                    continue
                key = frozenset((_key(p, scale), _key(q, scale)))
                out.setdefault(key, [p, q, []])[2].append(i)
    return out


def _canonical_axis(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    axis = (q - p) / np.linalg.norm(q - p)
    first = np.flatnonzero(np.abs(axis) > 1e-9)[0]
    return axis if axis[first] > 0.0 else -axis


def _into_face(axis, normal, midpoint, centroid):
    u = np.cross(axis, normal)
    u /= np.linalg.norm(u)
    return u if (centroid - midpoint) @ u > 0.0 else -u


def _merge_collinear(pieces):
    """Join (p, q, owners) pieces of one line where they touch."""
    axis = _canonical_axis(pieces[0][0], pieces[0][1])
    origin = pieces[0][0] - (pieces[0][0] @ axis) * axis
    spans = sorted(
        (min(p @ axis, q @ axis), max(p @ axis, q @ axis), list(o)) for p, q, o in pieces
    )
    merged = [spans[0]]
    for lo, hi, owners in spans[1:]:
        if lo <= merged[-1][1] + 1e-9:
            merged[-1] = (merged[-1][0], max(merged[-1][1], hi), merged[-1][2] + owners)
        else:
            merged.append((lo, hi, owners))
    return [(origin + lo * axis, origin + hi * axis, tuple(sorted(set(o)))) for lo, hi, o in merged]


def obstacle_edges(geometry: RoomGeometry) -> list[Edge]:
    """The diffracting edges of a mesh's obstacles.

    Closed convex components (shelves, cabinets) give wedges whose exterior
    angle follows from the dihedral angle; planar components (panels,
    tabletops drawn as one surface) give knife edges along their outline.
    Anything else is skipped with :class:`ObstacleEdgeWarning`. Edges on an
    enclosure face or on another obstacle (a box on a tabletop) are dropped,
    and a rim lying there (a bottomless box on the floor) still counts as
    closed. ``Edge.component`` indexes :func:`obstacle_components`.
    """
    occ = Occluders.from_geometry(geometry)
    edges: list[Edge] = []
    for c, faces in enumerate(obstacle_components(geometry)):
        own = set(faces)
        support = [i for i in range(len(occ.polygons)) if i not in own]
        polys = {i: geometry.vertices[list(geometry.obstacle_faces[i])] for i in faces}
        info = {i: _polygon_area_and_centroid(p) for i, p in polys.items()}
        points = np.vstack(list(polys.values()))
        centre = points.mean(axis=0)
        normal = {i: (n if n @ (ctr - centre) > 0.0 else -n) for i, (_, n, ctr) in info.items()}
        n0 = normal[faces[0]]
        planar = all(abs(normal[i] @ n0) > _PARALLEL for i in faces) and bool(
            np.all(np.abs((points - info[faces[0]][2]) @ n0) < 1e-5))
        free = [s for s in _segments(geometry, faces).values()
                if not occ.on_polygons((s[0] + s[1]) / 2.0, ON_ENCLOSURE, support)]
        groups: dict[tuple, list] = {}
        if planar:
            for p, q, owners in free:
                if len(owners) != 1:
                    continue
                axis = _canonical_axis(p, q)
                u = _into_face(axis, n0, (p + q) / 2.0, info[owners[0]][2])
                key = (tuple(np.round(u, 6)), tuple(np.round(np.cross(axis, p), 6)))
                groups.setdefault(key, []).append((p, q, owners, u, u, TWO_PI))
        else:
            closed = all(len(owners) == 2 for _, _, owners in free)
            convex = closed and all(
                bool(np.all((points - info[i][2]) @ normal[i] < 1e-6)) for i in faces)
            if not convex:
                warnings.warn(
                    f"obstacle component {c} ({geometry.obstacle_face_materials[faces[0]]}, "
                    f"{len(faces)} faces) is neither closed-convex nor planar: no diffraction",
                    ObstacleEdgeWarning, stacklevel=2)
                continue
            for p, q, (i, j) in free:
                cos_ij = float(np.clip(normal[i] @ normal[j], -1.0, 1.0))
                if cos_ij > _PARALLEL:
                    continue
                if tuple(np.round(normal[i], 6)) > tuple(np.round(normal[j], 6)):
                    i, j = j, i
                axis = _canonical_axis(p, q)
                mid = (p + q) / 2.0
                ua = _into_face(axis, normal[i], mid, info[i][2])
                ub = _into_face(axis, normal[j], mid, info[j][2])
                key = (tuple(np.round(normal[i], 6)), tuple(np.round(normal[j], 6)),
                       tuple(np.round(np.cross(axis, p), 6)))
                groups.setdefault(key, []).append((p, q, [i, j], ua, ub, np.pi + np.arccos(cos_ij)))
        for pieces in groups.values():
            _, _, _, ua, ub, theta_w = pieces[0]
            for p, q, owners in _merge_collinear([(p, q, o) for p, q, o, *_ in pieces]):
                edges.append(Edge(p, q, ua, ub, theta_w, owners, c))
    return edges


@dataclass(frozen=True)
class DiffractionPath:
    """One diffracted contribution.

    Pressure on the package scale: ``band_gain`` (reflections, air) times
    ``transfer(f)`` times ``exp(-j 2 pi f delay) / (4 pi distance)``.
    ``doa`` is in the receiver's frame, ``edge`` indexes
    :func:`obstacle_edges`, ``order`` counts the wall
    reflections on the path (0 or 1) and ``zone`` is the receiver's zone
    around the edge (:attr:`hybrid_rir.udfa.WedgePath.zone`).
    """

    delay: float
    distance: float
    doa: np.ndarray
    band_gain: np.ndarray
    transfer: Callable[[np.ndarray], np.ndarray]
    order: int
    zone: str
    edge: int


# A reflection point this close to a polygon's outline counts as on it, so a
# point on a seam between two faces of one wall is accepted exactly once.
ON_OUTLINE = 1e-6


@dataclass(frozen=True)
class _End:
    """The source or the receiver, or its image in one enclosure plane.

    An image belongs to a plane, not to a face: CAD exports split one wall
    into coplanar faces, and an image per face counts a reflection point on a
    seam twice, or, with a point-in-polygon test that is ambiguous on a shared
    side, not at all.
    """

    position: np.ndarray  # the point or its image
    origin: np.ndarray  # the real point
    faces: tuple[int, ...] = ()  # Occluders indices of the plane's faces; () for the point itself


def _ends(point: np.ndarray, occ: Occluders, images: bool) -> list[_End]:
    point = np.asarray(point, dtype=float)
    out = [_End(point, point)]
    if not images:
        return out
    planes: dict[tuple, list[int]] = {}
    for i in range(occ.n_obstacle, len(occ.polygons)):
        key = (tuple(np.round(occ.normals[i], 6)), round(float(occ.offsets[i]), 6))
        planes.setdefault(key, []).append(i)
    for faces in planes.values():
        n, off = occ.normals[faces[0]], occ.offsets[faces[0]]
        height = n @ point - off
        if height < 0.0:
            out.append(_End(point - 2.0 * height * n, point, tuple(faces)))
    return out


def _on_face(polygon: np.ndarray, normal: np.ndarray, x: np.ndarray) -> bool:
    if _inside_polygon(polygon, normal, x):
        return True
    for p, q in zip(polygon, np.roll(polygon, -1, axis=0)):
        d = q - p
        t = float(np.clip((x - p) @ d / (d @ d), 0.0, 1.0))
        if np.linalg.norm(x - p - t * d) < ON_OUTLINE:
            return True
    return False


def _wall_hit(occ: Occluders, end: _End, towards: np.ndarray):
    """(point, face) where the line from an image to ``towards`` meets its plane."""
    n, off = occ.normals[end.faces[0]], occ.offsets[end.faces[0]]
    d = towards - end.position
    if abs(n @ d) < 1e-12:
        return None
    t = (off - n @ end.position) / (n @ d)
    if not 0.0 < t < 1.0:
        return None
    hit = end.position + t * d
    for i in end.faces:
        if _on_face(occ.polygons[i], n, hit):
            return hit, i
    return None


def _polyline(occ: Occluders, src: _End, apexes: list[tuple[np.ndarray, frozenset[int]]],
              rec: _End):
    """The real path source (- wall) - apexes (- wall) - receiver, or None.

    ``apexes`` pairs each apex with its edge's faces. Returns the points, the
    walls hit, and for each point the faces it lies on.
    """
    points, walls, on = [src.origin], [], [frozenset()]
    if src.faces:
        hit = _wall_hit(occ, src, apexes[0][0])
        if hit is None:
            return None
        points.append(hit[0])
        walls.append(hit[1])
        on.append(frozenset(src.faces))
    points += [a for a, _ in apexes]
    on += [faces for _, faces in apexes]
    if rec.faces:
        hit = _wall_hit(occ, rec, apexes[-1][0])
        if hit is None:
            return None
        points.append(hit[0])
        walls.append(hit[1])
        on.append(frozenset(rec.faces))
    return points + [rec.origin], walls, on + [frozenset()]


def _clear(occ: Occluders, points: list[np.ndarray], on: list[frozenset[int]]) -> bool:
    """Is every leg free? A leg passes the faces its two ends lie on."""
    return not any(occ.blocked(a, b, sa | sb) for (a, sa), (b, sb) in pairwise(zip(points, on)))


def diffraction_paths(spec: Pair, *, max_order: int = 1) -> list[DiffractionPath]:
    """Every diffracted path, with at most one wall reflection before or after it.

    For ``max_order >= 1`` the source's image in a wall gives the paths that
    reflect before they diffract, and the receiver's image those that reflect
    after, which is the same construction by reciprocity.
    """
    geometry = spec.geometry
    if not geometry.obstacle_faces:
        raise ValueError("diffraction needs surfaces with role='obstacle'")
    c = spec.speed_of_sound
    receiver = np.asarray(spec.receiver, dtype=float)
    occ = Occluders.from_geometry(geometry)
    edges = obstacle_edges(geometry)
    air = _air_absorption_coefficient() if spec.air_absorption else np.zeros(BANDS.size)

    def reflection(face):
        material = geometry.materials[geometry.face_materials[face - occ.n_obstacle]]
        return np.sqrt(1.0 - np.asarray(material.absorption, dtype=float))

    def make(line, length, transfer, zone, edge):
        points, walls, _ = line
        doa = points[-2] - receiver
        if spec.receiver_rotation is not None:
            doa = spec.receiver_rotation.T @ doa
        gain = np.exp(-air * length / 2.0)
        for face in walls:
            gain = gain * reflection(face)
        return DiffractionPath(length / c, length, doa, gain, transfer, len(walls), zone, edge)

    sources = _ends(spec.source, occ, max_order >= 1)
    receivers = _ends(receiver, occ, max_order >= 1)
    pairs = [(s, receivers[0]) for s in sources] + [(sources[0], r) for r in receivers[1:]]
    out: list[DiffractionPath] = []
    for src, rec in pairs:
        for k, e in enumerate(edges):
            wp = wedge_path(e, src.position, rec.position)
            if wp is None:
                continue
            half = wp.length / 2.0
            apex = e.point_at(float(np.clip(wp.z_apex, -half, half)))
            pl = _polyline(occ, src, [(apex, frozenset(e.faces))], rec)
            if pl is None or not _clear(occ, pl[0], pl[2]):
                continue
            out.append(make(pl, wp.d_constrained,
                            partial(udfa_transfer, wp, c=c), wp.zone, k))
    return [p for p in out if np.any(p.band_gain > 0.0)]


def encode_diffraction(paths: list[DiffractionPath], sh_order: int, n_samples: int, fs: float,
                       *, normalization: str = "NM") -> np.ndarray:
    """Encode diffracted paths at their arrival directions, as :func:`encode` does images."""
    n_ch = (sh_order + 1) ** 2
    if not paths:
        return np.zeros((n_samples, n_ch))
    doa = np.array([p.doa for p in paths])
    sh = spharpy.spherical.spherical_harmonic_basis_real(
        sh_order, pf.Coordinates(doa[:, 0], doa[:, 1], doa[:, 2]),
        normalization=normalization, channel_convention="ACN")
    gains = np.array([p.band_gain / (4.0 * np.pi * p.distance) for p in paths])
    return render_transfers(np.array([p.delay for p in paths]), [p.transfer for p in paths],
                            gains, sh, BANDS, fs, n_samples)


@dataclass(frozen=True)
class BtmsContribution:
    """One BTMS segment (first order) or segment pair (second order).

    ``ir`` is the toolbox's response (free field 1/r) starting at sample
    ``start``; ``doa`` in the receiver frame; ``edges`` indexes
    :func:`obstacle_edges`. Order 0 is the scaled geometric component of a
    path exactly on a zone boundary of edge ``edges[0]``.
    """

    start: int
    ir: np.ndarray
    doa: np.ndarray
    band_gain: np.ndarray
    order: int
    edges: tuple[int, ...]
    singular: bool


def segment_distance(point: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    """Distance from a point to the segment a-b."""
    ab = b - a
    t = np.clip((point - a) @ ab / (ab @ ab), 0.0, 1.0)
    return float(np.linalg.norm(point - (a + t * ab)))


def second_order_start(source, receiver, seg1, seg2, c: float, fs: float) -> int:
    """A sample before the first arrival over any point of segment 1 then any
    point of segment 2: the path is at least |SR| and at least the distance
    from S to segment 1 plus that from segment 2 to R. One sample of margin
    keeps rounding at an exact bound from putting the first slot below 1."""
    shortest = max(np.linalg.norm(receiver - source),
                   segment_distance(source, *seg1) + segment_distance(receiver, *seg2))
    return max(int(np.floor(shortest / c * fs)) - 1, 0)


def btms_paths(spec: Pair, *, max_order: int = 1, second_order: bool = True) -> list[BtmsContribution]:
    """First-order BTMS over every edge segment, with at most one wall
    reflection before or after; second order (direct paths) when
    ``second_order``."""
    geometry = spec.geometry
    if not geometry.obstacle_faces:
        raise ValueError("diffraction needs surfaces with role='obstacle'")
    c, fs = spec.speed_of_sound, spec.fs
    receiver = np.asarray(spec.receiver, dtype=float)
    occ = Occluders.from_geometry(geometry)
    edges = obstacle_edges(geometry)
    air = _air_absorption_coefficient() if spec.air_absorption else np.zeros(BANDS.size)

    def reflection(face):
        material = geometry.materials[geometry.face_materials[face - occ.n_obstacle]]
        return np.sqrt(1.0 - np.asarray(material.absorption, dtype=float))

    def gain_of(length, walls):
        gain = np.exp(-air * length / 2.0)
        for face in walls:
            gain = gain * reflection(face)
        return gain

    def doa_of(point):
        d = point - receiver
        return spec.receiver_rotation.T @ d if spec.receiver_rotation is not None else d

    def face_reflection(e, u):
        face = next((i for i in e.faces if abs(occ.normals[i] @ u) < 1e-6), e.faces[0])
        material = geometry.materials[geometry.obstacle_face_materials[face]]
        return np.sqrt(1.0 - np.asarray(material.absorption, dtype=float))

    sources = _ends(spec.source, occ, max_order >= 1)
    receivers = _ends(receiver, occ, max_order >= 1)
    pairs = [(s, receivers[0]) for s in sources] + [(sources[0], r) for r in receivers[1:]]
    out: list[BtmsContribution] = []
    halves: set[tuple] = set()
    ends_seen = Counter(_key(p, 1e-6) for e in edges for p in (e.start, e.end))
    corners = {p for p, n in ends_seen.items() if n > 1}
    for src, rec in pairs:
        for k, e in enumerate(edges):
            ends, _ = toolbox_frame(e)
            (rs, ts, zs), (rr, tr, zr) = toolbox_cyl(e, np.array([src.position, rec.position]))
            if rs < 1e-9 or rr < 1e-9 or ts > e.theta_w + 1e-9 or tr > e.theta_w + 1e-9:
                continue
            unit = (ends[1] - ends[0]) / e.length
            for z0, z1 in edge_segments(e, src.position, rec.position):
                mid = ends[0] + (z0 + z1) / 2.0 * unit
                pl = _polyline(occ, src, [(mid, frozenset(e.faces))], rec)
                if pl is None or not _clear(occ, pl[0], pl[2]):
                    continue
                apex = np.clip((rs * zr + rr * zs) / (rs + rr), z0, z1)
                shortest = min(np.linalg.norm(p - src.position) + np.linalg.norm(rec.position - p)
                               for p in (ends[0] + z * unit for z in (z0, z1, apex)))
                start = max(int(np.floor(shortest / c * fs)) - 1, 0)
                ir, _, singular, _ = wedge1st_ir(fs, c, TWO_PI - e.theta_w, rs, min(ts, e.theta_w), zs,
                                                 rr, min(tr, e.theta_w), zr, np.array([z0, z1]), start * c / fs)
                length = np.linalg.norm(mid - src.position) + np.linalg.norm(rec.position - mid)
                out.append(BtmsContribution(start, ir, doa_of(pl[0][-2]), gain_of(length, pl[1]),
                                            1, (k,), bool(np.any(singular))))
                # On a zone boundary the toolbox drops the singular term and scales the
                # geometric component, which keeps the total field continuous (Svensson
                # and Calamia 2006): 1/2 for an edge hit (Svensson et al. 2025, Eqs. 2-4),
                # and for a corner hit 2/3 for the direct sound and 1/3 for a specular
                # reflection (EDfindconvexGApaths.m).
                # The image sources drop that component there (pyroomacoustics occludes a
                # path within about 1 mm of an obstacle's edge), so it is added here, as
                # EDmakefirstorderirs.m places a pulse: split linearly between the two
                # samples around its arrival. Once per component: the edges meeting at a
                # corner, an apex on a segment end, and a wall image before or after the
                # edge would otherwise count it again.
                apex_length = float(np.hypot(rs + rr, zr - zs))
                za = (rs * zr + rr * zs) / (rs + rr)
                end = ends[0] if abs(za) < 1e-9 else ends[1] if abs(za - e.length) < 1e-9 else None
                corner = end is not None and _key(end, 1e-6) in corners
                for term in np.flatnonzero(singular):
                    face = None if term in (1, 2) else (e.face_b if term == 0 else e.face_a)
                    plane = None
                    if face is not None:
                        normal = np.cross(e.axis, face)
                        normal = normal / np.linalg.norm(normal)
                        normal = normal if normal[np.flatnonzero(np.abs(normal) > 1e-9)[0]] > 0 else -normal
                        plane = (_key(normal, 1e-6), round(float(normal @ e.start), 6))
                    key = (tuple(sorted(pl[1])), plane, round(apex_length, 9))
                    if key in halves:
                        continue
                    halves.add(key)
                    scale = (1.0 / 3.0 if face is not None else 2.0 / 3.0) if corner else 0.5
                    slot = (apex_length - start * c / fs) * fs / c
                    n = int(np.floor(slot))
                    half = np.zeros(n + 2)
                    half[n:] = scale / apex_length * np.array([1.0 - (slot - n), slot - n])
                    gain = gain_of(apex_length, pl[1])
                    if face is not None:
                        gain = gain * face_reflection(e, face)
                    out.append(BtmsContribution(start, half, doa_of(pl[0][-2]), gain, 0, (k,), True))

    if second_order:
        src, rec = sources[0], receivers[0]

        def prepare(e):
            ends, _ = toolbox_frame(e)
            cyl_s = toolbox_cyl(e, src.position[None, :])[0]
            cyl_r = toolbox_cyl(e, rec.position[None, :])[0]
            ok_s = cyl_s[0] >= 1e-9 and cyl_s[1] <= e.theta_w + 1e-9
            ok_r = cyl_r[0] >= 1e-9 and cyl_r[1] <= e.theta_w + 1e-9
            z = edge_segments(e, src.position, rec.position)
            lo = ends[0] + z[:, :1] * (ends[1] - ends[0]) / e.length
            hi = ends[0] + z[:, 1:] * (ends[1] - ends[0]) / e.length
            mid = (lo + hi) / 2.0
            skip = frozenset(e.faces)
            from_source = np.array([ok_s and not occ.blocked(src.position, m, skip) for m in mid])
            to_receiver = np.array([ok_r and not occ.blocked(m, rec.position, skip) for m in mid])
            return cyl_s, cyl_r, z, lo, hi, mid, from_source, to_receiver

        prep = [prepare(e) for e in edges]
        for i, e1 in enumerate(edges):
            cyl_s, _, z1, lo1, hi1, mid1, from_source, _ = prep[i]
            if not from_source.any():
                continue
            for j, e2 in enumerate(edges):
                _, cyl_r, z2, lo2, hi2, mid2, _, to_receiver = prep[j]
                if i == j or not to_receiver.any():
                    continue
                along = along_plane(e1, e2)
                if along:
                    theta_2_about_1, theta_1_about_2 = coplanar_thetas(e1, e2)
                # the equal path along the panel's rear side (EDmakeHODirs.m, boostfactor 2)
                thin_plate = along and e1.theta_w == TWO_PI and e2.theta_w == TWO_PI
                nu = np.pi / np.array([e1.theta_w, e2.theta_w])
                skip = frozenset(e1.faces) | frozenset(e2.faces)
                r2_about_1 = toolbox_cyl(e1, mid2)[:, 0]
                for k in np.flatnonzero(from_source):
                    p0, p1, m1 = lo1[k], hi1[k], mid1[k]
                    r1_about_2 = toolbox_cyl(e2, m1[None, :])[0, 0]
                    for n in np.flatnonzero(to_receiver):
                        q0, q1, m2 = lo2[n], hi2[n], mid2[n]
                        if r1_about_2 < 1e-9 or r2_about_1[n] < 1e-9:
                            continue
                        if not pair_valid(e1, e2, m1, m2) or occ.blocked(m1, m2, skip):
                            continue
                        limits = np.array([z1[k], z2[n]])
                        steps, dz = element_grid(limits[:, 1] - limits[:, 0], c, fs)
                        start = second_order_start(src.position, rec.position, (p0, p1), (q0, q1), c, fs)
                        e2_about_1 = toolbox_cyl(e1, np.array([q0, q1]))
                        e1_about_2 = toolbox_cyl(e2, np.array([p0, p1]))
                        if along:
                            e2_about_1[:, 1], e1_about_2[:, 1] = theta_2_about_1, theta_1_about_2
                        ir = wedge2nd_ir(cyl_s, cyl_r, e2_about_1, e1_about_2, nu, limits, dz, steps,
                                         along, start * c / fs, c, fs)
                        if thin_plate:
                            ir = 2.0 * ir
                        length = (np.linalg.norm(m1 - src.position) + np.linalg.norm(m2 - m1)
                                  + np.linalg.norm(rec.position - m2))
                        out.append(BtmsContribution(start, ir, doa_of(m2), gain_of(length, ()),
                                                    2, (i, j), False))
    return out


def encode_btms(contributions: list[BtmsContribution], sh_order: int, n_samples: int, fs: float,
                *, normalization: str = "NM") -> np.ndarray:
    """Encode BTMS contributions at their directions; free field 1/r to 1/(4 pi r)."""
    n_ch = (sh_order + 1) ** 2
    if not contributions:
        return np.zeros((n_samples, n_ch))
    doa = np.array([p.doa for p in contributions])
    sh = spharpy.spherical.spherical_harmonic_basis_real(
        sh_order, pf.Coordinates(doa[:, 0], doa[:, 1], doa[:, 2]),
        normalization=normalization, channel_convention="ACN")
    signals = [(p.start, np.outer(p.ir / (4.0 * np.pi), sh[m])) for m, p in enumerate(contributions)]
    return render_signals(signals, np.array([p.band_gain for p in contributions]), BANDS, fs, n_samples)
