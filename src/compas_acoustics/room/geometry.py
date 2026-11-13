"""A scene's surfaces as one validated room: welded, merged, checked closed, wound outward, obstacles two-sided.

Two views are kept. ``faces`` holds each obstacle twice, ``OBSTACLE_OFFSET`` either side, as the geometric
methods read it; the drawn model is the boundary faces plus ``obstacle_faces``, which a volume mesher (a wave
solver) must use instead of the offset copies.
"""

from __future__ import annotations

import threading
import warnings
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass
from dataclasses import replace
from functools import cached_property
from typing import Dict
from typing import Iterator
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
import pyroomacoustics as pra
from compas.datastructures import Mesh
from compas.geometry import Point
from compas.geometry import Polyline
from scipy.spatial import cKDTree

from compas_acoustics.room.air import AirConditions
from compas_acoustics.room.air import air_or_default

__all__ = ["RoomGeometry", "SurfaceMaterial", "DrawnSides", "SYNTHESIS_BANDS", "OBSTACLE_OFFSET", "DEFAULT_SCATTERING", "PRA_STATE_LOCK", "OPEN_MATERIAL", "pra_octave_grid"]

# The synthesis grid; materials start at 125 Hz and 62.5 Hz takes the 125 Hz value
SYNTHESIS_BANDS = (62.5, 125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0)

# Half the gap between the two copies of an obstacle face, in metres: 100 times _MERGE_TOLERANCE
OBSTACLE_OFFSET = 1e-3

# Scattering coefficient, every band, of a material that gives none
DEFAULT_SCATTERING = 0.05


#: Held while compas_acoustics reads or changes pyroomacoustics' process-wide state: its constants (the octave
#: grid) and its random generator (seeded, then drawn by the ray tracer). Re-entrant. Threads that simulate at
#: once therefore run their pyroomacoustics sections one at a time, with the same result as run alone; for
#: parallel speed use processes (``Mono(n_jobs=...)``).
PRA_STATE_LOCK = threading.RLock()


@contextmanager
def pra_octave_grid() -> Iterator[None]:
    """Start pyroomacoustics' octave grid, process-wide state read when a room is built, at ``SYNTHESIS_BANDS[0]``, then restore it.

    At its default of 125 Hz a 48 kHz room has 8 bands against 7 wall coefficients. Holds :data:`PRA_STATE_LOCK`,
    so another thread can neither see the changed grid nor change it meanwhile.
    """
    key = "octave_bands_base_freq"
    with PRA_STATE_LOCK:
        previous = pra.constants.get(key)
        pra.constants.set(key, SYNTHESIS_BANDS[0])
        try:
            yield
        finally:
            pra.constants.set(key, previous)


def _extended(coefficients) -> Optional[np.ndarray]:
    if coefficients is None:
        return None
    values = [float(c) for c in coefficients]
    return np.asarray([values[0]] + values, dtype=float)


@dataclass(frozen=True, eq=False)
class SurfaceMaterial:
    """An acoustic material on ``SYNTHESIS_BANDS``."""

    name: str
    absorption: np.ndarray
    scattering: Optional[np.ndarray] = None
    #: Placeholder for a wave-based solver's boundary condition; unused by the geometric methods.
    impedance: Optional[np.ndarray] = None

    @classmethod
    def from_acoustic_material(cls, material) -> SurfaceMaterial:
        bands = [float(f) for f in material.frequency_bands]
        if bands != list(SYNTHESIS_BANDS[1:]):
            raise ValueError(f"Material '{material.name}' is on bands {bands}, expected {list(SYNTHESIS_BANDS[1:])}.")
        surface = cls(
            name=material.name,
            absorption=_extended(material.absorption_coefficients),
            scattering=_extended(material.scattering_coefficients),
        )
        # a coefficient outside [0, 1] is an input error, not something to absorb silently
        for kind, values in (("absorption", surface.absorption), ("scattering", surface.scattering)):
            if values is not None and (np.any(values < 0.0) or np.any(values > 1.0)):
                raise ValueError(f"Material '{material.name}' has {kind} coefficients outside [0, 1].")
        return surface

    def same_coefficients(self, other: SurfaceMaterial) -> bool:
        if not np.array_equal(self.absorption, other.absorption):
            return False
        if self.scattering is None or other.scattering is None:
            return self.scattering is None and other.scattering is None
        return bool(np.array_equal(self.scattering, other.scattering))


@dataclass(frozen=True, eq=False)
class DrawnSides:
    """Which sides of each drawn polygon the air touches, for the sector edge detector
    (:func:`compas_acoustics.geometry.sector_edges`).

    :meth:`RoomGeometry.from_prepared` sets it from the prepared mesh's walls. In a room the polygons are those of
    :class:`~compas_acoustics.hybrid.diffraction.Occluders`, in its order: ``obstacle_faces``, then the boundary faces.

    Attributes
    ----------
    normals : (n, 3) ndarray
        For a one-sided polygon, its normal pointing away from the air; only its side of the polygon's plane is used.
    two_sided : (n,) ndarray of bool
        Air on both sides.
    reached : (n,) ndarray of bool
        False where the air never touches the polygon: both of its sides are solid.
    min_excess : float
        In radians: an air sector must open wider than pi by more than this to diffract.
    min_length : float
        In metres: shorter edges are dropped.
    """

    normals: np.ndarray
    two_sided: np.ndarray
    reached: np.ndarray
    min_excess: float = 1e-4
    min_length: float = 0.0

    def take(self, indices) -> DrawnSides:
        """The sides of the polygons ``indices``, in that order."""
        indices = np.asarray(list(indices), dtype=np.int64)
        return replace(self, normals=self.normals[indices].reshape(-1, 3), two_sided=self.two_sided[indices], reached=self.reached[indices])


#: The material of :attr:`RoomGeometry.open_area`: an opening reflects nothing.
OPEN_MATERIAL = SurfaceMaterial(name="open", absorption=np.ones(len(SYNTHESIS_BANDS)), scattering=np.zeros(len(SYNTHESIS_BANDS)))


def _polygon_area_and_centroid(points: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
    """Area, unit normal (Newell) and vertex mean of one planar polygon in 3-D."""
    rolled = np.roll(points, -1, axis=0)
    normal = 0.5 * np.cross(points, rolled).sum(axis=0)
    area = float(np.linalg.norm(normal))
    if area <= 0.0:
        raise ValueError("a face has zero area; check for repeated vertices")
    return area, normal / area, points.mean(axis=0)


def _signed_volume(vertices: np.ndarray, faces) -> float:
    """Divergence theorem, ``V = sum_f (c_f . n_f) A_f / 3``; any point of the face's plane serves as ``c_f``."""
    total = 0.0
    for face in faces:
        area, normal, centroid = _polygon_area_and_centroid(vertices[list(face)])
        total += float(centroid @ normal) * area
    return total / 3.0


def _object_polygons(acoustic_object) -> List[np.ndarray]:
    """The faces of one acoustic object as (k, 3) arrays; a polyline is closed, its last point repeating its first."""
    geometry = acoustic_object.acoustic_geometry
    if isinstance(geometry, Mesh):
        return [np.array(geometry.face_coordinates(f), dtype=float) for f in geometry.faces()]
    if isinstance(geometry, Polyline):
        return [np.array([list(p) for p in geometry.points[:-1]], dtype=float)]
    if isinstance(geometry, list) and geometry and isinstance(geometry[0], Point):
        return [np.array([list(p) for p in geometry], dtype=float)]
    raise ValueError(f"Unsupported acoustic geometry type on '{acoustic_object.name}'.")


def _weld(points: np.ndarray, tolerance: float) -> Tuple[np.ndarray, np.ndarray]:
    """Merge points closer than ``tolerance``, transitively, each cluster into its first point.

    CAD exports write every surface's vertices separately.
    """
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


def _split_at_t_junctions(vertices: np.ndarray, a: int, b: int) -> List[int]:
    """The edge ``a -> b`` as the chain of vertices lying on it, within 1e-6 of the model's extent.

    CAD exports subdivide one side of a seam and not the other, so edges are compared segment by segment. The
    tolerance is far below the weld's: a sliver face can put a vertex a fraction of a millimetre off an edge,
    and splitting the edge there makes it non-manifold.
    """
    start = vertices[a]
    direction = vertices[b] - start
    offsets = vertices - start
    t = offsets @ direction / (direction @ direction)
    distance = np.linalg.norm(offsets - np.outer(t, direction), axis=1)
    tolerance = 1e-6 * float(np.ptp(vertices, axis=0).max())
    inner = np.flatnonzero((t > 0.0) & (t < 1.0) & (distance < tolerance))
    # t of the endpoints themselves can land one ulp inside (0, 1): BLAS rounds the matrix product differently by platform
    inner = inner[(inner != a) & (inner != b)]
    return [a, *inner[np.argsort(t[inner])].tolist(), b]


def _segments(vertices: np.ndarray, face) -> List[Tuple[int, int]]:
    face = list(face)
    out: List[Tuple[int, int]] = []
    for a, b in zip(face, face[1:] + face[:1]):
        if a == b:
            raise ValueError("a face repeats a vertex")
        chain = _split_at_t_junctions(vertices, a, b)
        out.extend(zip(chain[:-1], chain[1:]))
    return out


def _without_collapsed(faces, groups):
    """Faces without consecutive repeated vertices, dropping those left with fewer than three; and how many were dropped."""
    kept_faces, kept_groups = [], []
    for face, group in zip(faces, groups):
        cleaned = [v for k, v in enumerate(face) if v != face[k - 1]]
        if len(set(cleaned)) >= 3:
            kept_faces.append(cleaned)
            kept_groups.append(group)
    return kept_faces, kept_groups, len(faces) - len(kept_faces)


def _triangulated_where_warped(vertices: np.ndarray, faces, groups):
    """Faces whose corners lie off their own plane, as fans of triangles of the same group and winding.

    A wall's corners must lie within ``_MERGE_TOLERANCE`` of a plane, and a CAD quad with one corner a few
    millimetres out is still a surface.
    """
    out_faces, out_groups = [], []
    for face, group in zip(faces, groups):
        if len(face) > 3:
            points = vertices[face]
            _, normal, centroid = _polygon_area_and_centroid(points)
            if np.max(np.abs((points - centroid) @ normal)) > _MERGE_TOLERANCE:
                for k in range(1, len(face) - 1):
                    out_faces.append([face[0], face[k], face[k + 1]])
                    out_groups.append(group)
                continue
        out_faces.append(face)
        out_groups.append(group)
    return out_faces, out_groups


def _orient(vertices: np.ndarray, faces):
    """Flip flags that make every shared segment run opposite ways.

    Surfaces drawn as separate objects have no common winding, so each face's
    neighbours across every segment are flipped to agree with it. Returns
    ``(flips, component, defect)``: ``component`` labels each connected
    surface, and ``defect`` is ``None`` when every segment is shared by exactly
    two faces that can be made to agree, else such a segment ``(a, b)``.
    """
    segments = [_segments(vertices, face) for face in faces]
    directed = [set(segs) for segs in segments]
    users: Dict[frozenset, List[int]] = {}
    for index, segs in enumerate(segments):
        for p, q in segs:
            users.setdefault(frozenset((p, q)), []).append(index)
    for edge, owners in users.items():
        if len(owners) != 2:
            return None, None, tuple(sorted(edge))

    flips: List[Optional[bool]] = [None] * len(faces)
    component = [-1] * len(faces)
    for start in range(len(faces)):
        if flips[start] is not None:
            continue
        flips[start] = False
        component[start] = start
        queue = deque([start])
        while queue:
            i = queue.popleft()
            for p, q in segments[i]:
                first, second = users[frozenset((p, q))]
                j = second if first == i else first
                if j == i:
                    return None, None, (p, q)
                required = flips[i] != ((p, q) in directed[j])
                if flips[j] is None:
                    flips[j] = required
                    component[j] = start
                    queue.append(j)
                elif flips[j] != required:
                    return None, None, (p, q)
    return flips, component, None


def _orient_outward(vertices: np.ndarray, faces):
    """Boundary faces wound consistently, their largest volume facing out.

    pyroomacoustics drops image sources silently unless the normals point out of the air. Any further closed volume (a
    pillar) bounds the air from outside, so it is wound into itself. Returns ``(faces, defect)``; on a defect
    the faces come back as drawn.
    """
    flips, component, defect = _orient(vertices, faces)
    if defect is not None:
        return [list(f) for f in faces], defect
    oriented = [list(f)[::-1] if flip else list(f) for f, flip in zip(faces, flips)]
    volumes: Dict[int, float] = {}
    for face, label in zip(oriented, component):
        volumes[label] = volumes.get(label, 0.0) + _signed_volume(vertices, [face])
    outer = max(volumes, key=lambda label: abs(volumes[label]))
    for index, label in enumerate(component):
        if (volumes[label] > 0.0) != (label == outer):
            oriented[index] = oriented[index][::-1]
    return oriented, None


@dataclass(frozen=True, eq=False)
class RoomGeometry:
    """Every surface of a scene as one room: welded, oriented, materials on ``SYNTHESIS_BANDS``.

    Faces are listed boundary first, then the front and then the back copy of every obstacle face.
    ``is_closed`` says whether the boundary faces close a volume; ``open_edge`` holds the end points of the
    first edge that prevents it.
    """

    vertices: np.ndarray
    faces: Tuple[Tuple[int, ...], ...]
    face_materials: Tuple[str, ...]
    face_names: Tuple[str, ...]
    face_roles: Tuple[str, ...]
    materials: Dict[str, SurfaceMaterial]
    is_closed: bool
    open_edge: Optional[Tuple[Tuple[float, ...], Tuple[float, ...]]] = None
    # The obstacle polygons as drawn, before the two offset copies: the edges
    # that diffract are theirs (compas_acoustics.hybrid.diffraction).
    obstacle_faces: Tuple[Tuple[int, ...], ...] = ()
    obstacle_face_materials: Tuple[str, ...] = ()
    #: How many faces fewer the room would have with ``merge_coplanar=True`` (0 once merged).
    mergeable_faces: int = 0
    #: Whether :meth:`is_inside` tests points; False takes every point as in the air (a prepared, open model).
    check_inside: bool = True
    #: The air volume in m^3 when known otherwise than from a closed boundary (a prepared model's mean free path).
    air_volume: Optional[float] = None
    #: Area of the openings in m^2, fully absorbing in :attr:`surfaces` and :attr:`surface_area` (a prepared, open model).
    open_area: float = 0.0
    #: Which sides of each drawn polygon the air touches (a prepared model): when set,
    #: :func:`~compas_acoustics.hybrid.diffraction.diffraction_edges` finds the edges by the sector detector from it,
    #: otherwise by the role-based legacy detectors.
    drawn_sides: Optional[DrawnSides] = None

    @classmethod
    def from_scene(cls, scene, merge_coplanar: bool = True, weld_tolerance: float = 1e-3, prepare=None) -> RoomGeometry:
        """Collect, weld, orient and validate every surface of ``scene``.

        Boundary faces that close a volume are wound outward; otherwise they are kept as drawn, with a
        ``UserWarning``. Obstacle faces are stored twice, ``OBSTACLE_OFFSET`` either side and wound opposite,
        since a wall reflects off one side only. ``merge_coplanar`` joins coplanar faces of one object and
        material into larger walls (a triangulated CAD mesh); ``False`` keeps them and warns where it could.
        ``weld_tolerance``, in metres, is a CAD model's usual absolute tolerance.

        ``prepare``, a :class:`compas_acoustics.geometry.PrepareOptions`, instead orients every face from the side
        the scene's sources and receivers are on, which also handles models that do not close, and finds thin
        surfaces with air on both sides (:func:`compas_acoustics.geometry.prepare_mesh`; ``merge_coplanar`` and
        ``weld_tolerance`` are then the options'). Obstacles stay two-sided.
        """
        if prepare is not None:
            return cls._from_scene_prepared(scene, prepare)
        polygons: List[np.ndarray] = []
        groups: List[Tuple[str, str, str]] = []  # (role, object name, material name)
        materials: Dict[str, SurfaceMaterial] = {}
        for acoustic_object in scene.acoustic_objects:
            if acoustic_object.material is None:
                raise ValueError(f"Acoustic object '{acoustic_object.name}' has no material.")
            material = SurfaceMaterial.from_acoustic_material(acoustic_object.material)
            known = materials.setdefault(material.name, material)
            if not known.same_coefficients(material):
                raise ValueError(f"Two different materials are named '{material.name}'; material names must be unique.")
            for polygon in _object_polygons(acoustic_object):
                polygons.append(polygon)
                groups.append((acoustic_object.role, str(acoustic_object.name), material.name))
        return cls._from_polygons(polygons, groups, materials, merge_coplanar, weld_tolerance)

    @classmethod
    def from_mesh(cls, vertices, faces, face_materials, materials, obstacles=(), merge_coplanar: bool = False, weld_tolerance: float = 1e-3) -> RoomGeometry:
        """A room from shared vertices and faces; ``materials`` maps names to :class:`SurfaceMaterial`, those in ``obstacles`` are obstacles."""
        vertices = np.asarray(vertices, dtype=float)
        polygons = [vertices[list(face)] for face in faces]
        groups = [("obstacle" if name in obstacles else "boundary", name, name) for name in face_materials]
        unknown = set(obstacles) - set(face_materials)
        if unknown:
            raise ValueError(f"obstacles names groups no face has: {sorted(unknown)}")
        missing = set(face_materials) - set(materials)
        if missing:
            raise ValueError(f"materials has no entry for {sorted(missing)}")
        return cls._from_polygons(polygons, groups, dict(materials), merge_coplanar, weld_tolerance)

    @classmethod
    def from_prepared(
        cls,
        prepared,
        materials,
        check_inside: bool = False,
        air_volume: Optional[float] = None,
        *,
        diffraction_min_excess: float = 1e-4,
        diffraction_min_length: float = 0.0,
    ) -> RoomGeometry:
        """The room of a :class:`compas_acoustics.geometry.PreparedMesh`, its orientation and two-sided faces kept.

        Every merged wall of ``prepared`` becomes a face: two-sided walls and the walls of objects standing in the air
        (furniture, columns: :func:`compas_acoustics.geometry.standing_objects`) are obstacles, stored twice as usual,
        and the others are the boundary, wound as prepared (their normals point away from the air). The room is
        closed when every boundary face belongs to a closed welded component.

        The diffracting edges (:func:`~compas_acoustics.hybrid.diffraction.diffraction_edges`) are found by the sector
        detector (:func:`compas_acoustics.geometry.sector_edges`) from what the preparation knows of each wall
        (:attr:`drawn_sides`, :func:`compas_acoustics.geometry.prepared_wall_sides`), not from the roles: a room that
        does not close still diffracts at its re-entrant corners and at the pillars fused into its shell, and a wall
        the air never reaches gives no edge.

        Parameters
        ----------
        prepared : compas_acoustics.geometry.PreparedMesh
            Whose ``face_material`` holds material names.
        materials : mapping
            Material name to :class:`~compas_acoustics.AcousticMaterial` or :class:`SurfaceMaterial`. A face whose
            ``face_scattering`` is set gets a copy of its material with that scattering in every band.
        check_inside : bool, default=False
            False takes every point as in the air: the preparation already reports seeds outside the model
            (:func:`compas_acoustics.geometry.analyze`).
        air_volume : float, optional
            In m^3; default the prepared mesh's (from the mean free path) when the room does not close.
        diffraction_min_excess : float, default=1e-4
            In radians: an edge diffracts when its air sector opens wider than pi by more than this (the facets of a
            tessellated curve open by 2 pi / sides).
        diffraction_min_length : float, default=0
            In metres: shorter diffracting edges are dropped.
        """
        from compas_acoustics.geometry import boundary_closes
        from compas_acoustics.geometry import obstacle_mask
        from compas_acoustics.geometry import prepared_wall_sides

        surface: Dict[str, SurfaceMaterial] = {}
        obstacle = obstacle_mask(prepared)

        def resolve(name, scattering):
            if not isinstance(name, str):
                raise ValueError(f"face materials must be names, got {name!r}; resolve them before RoomGeometry.from_prepared")
            if name not in materials:
                raise ValueError(f"materials has no entry for '{name}'")
            base = materials[name]
            base = base if isinstance(base, SurfaceMaterial) else SurfaceMaterial.from_acoustic_material(base)
            if np.isnan(scattering):
                surface.setdefault(name, base)
                return name
            if not 0.0 <= scattering <= 1.0:
                raise ValueError(f"scattering {scattering} outside [0, 1]")
            derived = f"{name} (s={scattering:g})"
            flat = np.full(len(SYNTHESIS_BANDS), float(scattering))
            surface.setdefault(derived, SurfaceMaterial(name=derived, absorption=base.absorption, scattering=flat, impedance=base.impedance))
            return derived

        polygons: List[np.ndarray] = []
        groups: List[Tuple[str, str, str]] = []
        for wall in prepared.walls:
            f0 = int(wall.faces[0])
            obj = int(prepared.face_object[f0])
            name = prepared.objects[obj] if obj >= 0 else "unnamed"
            material = resolve(prepared.face_material[f0], float(prepared.face_scattering[f0]))
            polygons.append(np.asarray(wall.corners, dtype=float))
            groups.append(("obstacle" if obstacle[f0] else "boundary", str(name), material))
        if not polygons:
            raise ValueError("The prepared mesh has no walls (check that its faces have materials).")
        normals, two_sided, reached = prepared_wall_sides(prepared)
        sides = DrawnSides(normals, two_sided, reached, float(diffraction_min_excess), float(diffraction_min_length))
        closed = boundary_closes(prepared, obstacle)
        given_volume = air_volume is not None
        if not given_volume and not closed and prepared.air_volume > 0.0:
            air_volume = float(prepared.air_volume)
        # the walls are welded already; this only joins corners the merge projected a hair apart
        geometry = cls._from_polygons(
            polygons,
            groups,
            surface,
            merge_coplanar=False,
            weld_tolerance=1e-7,
            trust_orientation=True,
            closed=closed,
            check_inside=check_inside,
            air_volume=air_volume,
            sides=sides,
        )
        leak = float(getattr(prepared.visibility, "leak_fraction", 0.0)) if prepared.visibility is not None else 0.0
        if not closed and leak > 0.0:
            # In a diffuse field the share of reflections that escape is the openings' share of the boundary, so
            # an opening of S f / (1 - f) absorbs what leaks through the walls the model lacks. The walk's mean free
            # path runs between walls and openings, 4 V / (S + S_open), while the preparation's volume took it over
            # the walls alone: scale it up by the same share. Both are diffuse-field estimates.
            leak = min(leak, 0.99)
            walls = geometry.surface_area
            open_area = walls * leak / (1.0 - leak)
            volume = geometry.air_volume if given_volume else geometry.air_volume * (walls + open_area) / walls
            geometry = replace(geometry, open_area=open_area, air_volume=volume)
        return geometry

    @classmethod
    def _from_scene_prepared(cls, scene, options) -> RoomGeometry:
        from compas_acoustics.geometry import prepare_mesh

        vertices: List[np.ndarray] = []
        faces: List[List[int]] = []
        ranges: Dict[str, List[int]] = {}
        face_materials: Dict[int, str] = {}
        forced: List[bool] = []
        materials: Dict[str, SurfaceMaterial] = {}
        n = 0
        for acoustic_object in scene.acoustic_objects:
            if acoustic_object.material is None:
                raise ValueError(f"Acoustic object '{acoustic_object.name}' has no material.")
            material = SurfaceMaterial.from_acoustic_material(acoustic_object.material)
            known = materials.setdefault(material.name, material)
            if not known.same_coefficients(material):
                raise ValueError(f"Two different materials are named '{material.name}'; material names must be unique.")
            first = len(faces)
            for polygon in _object_polygons(acoustic_object):
                vertices.append(polygon)
                for k in range(1, len(polygon) - 1):
                    face_materials[len(faces)] = material.name
                    faces.append([n, n + k, n + k + 1])
                    forced.append(acoustic_object.role == "obstacle")
                n += len(polygon)
            name = str(acoustic_object.name)
            while name in ranges:  # object names need not be unique
                name += "'"
            ranges[name] = [first, len(faces) - 1]
        if not faces:
            raise ValueError("The scene has no acoustic surfaces.")
        seeds = [item.position for item in list(scene.sources) + list(scene.receivers)]
        prepared = prepare_mesh(np.vstack(vertices), faces, ranges, face_materials, None, seeds, options, force_two_sided=forced)
        return cls.from_prepared(prepared, materials)

    @classmethod
    def _from_polygons(
        cls,
        polygons,
        groups,
        materials,
        merge_coplanar,
        weld_tolerance,
        trust_orientation: bool = False,
        closed: Optional[bool] = None,
        check_inside: bool = True,
        air_volume: Optional[float] = None,
        sides: Optional[DrawnSides] = None,
    ) -> RoomGeometry:
        """``sides``, one entry per input polygon, follows each polygon through the weld, the collapsed faces dropped
        and the warped ones triangulated, and is stored in the room's drawn order (:attr:`drawn_sides`)."""
        if trust_orientation and closed is None:
            raise ValueError("trust_orientation needs closed")
        if not polygons:
            raise ValueError("The scene has no acoustic surfaces.")
        if sides is not None and merge_coplanar:
            raise ValueError("sides cannot follow faces through merge_coplanar")
        # every face carries the index of the polygon it comes from, through the steps that drop or split faces
        groups = [(group, k) for k, group in enumerate(groups)]

        vertices, inverse = _weld(np.vstack(polygons), weld_tolerance)
        faces: List[List[int]] = []
        start = 0
        for polygon in polygons:
            faces.append([int(i) for i in inverse[start : start + len(polygon)]])
            start += len(polygon)

        extent = float(np.ptp(vertices, axis=0).max())
        if extent > 500.0:
            warnings.warn(
                f"The model is {extent:g} units across. Coordinates are read as metres; a model drawn in millimetres needs scaling by 0.001 first.",
                UserWarning,
                stacklevel=3,
            )

        faces, groups, collapsed = _without_collapsed(faces, groups)
        if collapsed:
            warnings.warn(
                f"{collapsed} faces collapsed to a line or a point when vertices within {weld_tolerance:g} m were welded, and were dropped.",
                UserWarning,
                stacklevel=3,
            )
        if not faces:
            raise ValueError("Every face of the scene collapsed when its vertices were welded; check the model's units.")
        faces, groups = _triangulated_where_warped(vertices, faces, groups)
        source = [k for _, k in groups]
        groups = [group for group, _ in groups]

        mergeable = 0
        if merge_coplanar:
            faces, groups = _merged(vertices, faces, groups)
        else:
            mergeable = len(faces) - len(_merged(vertices, [list(f) for f in faces], list(groups))[0])
            if mergeable:
                warnings.warn(
                    f"{mergeable} faces lie in one plane with a neighbour of the same object and material, as in a "
                    "triangulated surface. Image sources are lost at the seams between them; "
                    "merge_coplanar=True joins them into larger walls.",
                    UserWarning,
                    stacklevel=3,
                )

        boundary = [i for i, group in enumerate(groups) if group[0] == "boundary"]
        defect = None
        oriented = [faces[i] for i in boundary]
        if boundary and not trust_orientation:
            oriented, defect = _orient_outward(vertices, oriented)
        is_closed = bool(closed) if trust_orientation else bool(boundary) and defect is None
        open_edge = None
        # a prepared mesh reports its open edges itself (compas_acoustics.geometry.analyze)
        if not is_closed and not trust_orientation:
            if defect is not None:
                open_edge = (tuple(vertices[defect[0]].tolist()), tuple(vertices[defect[1]].tolist()))
            where = f" (first open or non-manifold edge from {open_edge[0]} to {open_edge[1]})" if open_edge else ""
            if boundary:
                message = (
                    f"The boundary surfaces do not close a volume{where}. They are kept as drawn, so "
                    "reflections off their back sides may be missed. Mark free-standing "
                    "surfaces with role='obstacle'. "
                )
            else:
                message = "The scene has no boundary surfaces (role='boundary'). "
            warnings.warn(
                message + "Every point outside the obstacles is simulated (an open field); the hybrid method needs a closed room.",
                UserWarning,
                stacklevel=3,
            )

        out_faces = [tuple(face) for face in oriented]
        out_groups = [groups[i] for i in boundary]

        all_vertices = [vertices]
        n = vertices.shape[0]
        fronts, backs, obstacle_groups = [], [], []
        for face, group in zip(faces, groups):
            if group[0] != "obstacle":
                continue
            points = vertices[face]
            _, normal, _ = _polygon_area_and_centroid(points)
            all_vertices += [points - OBSTACLE_OFFSET * normal, points + OBSTACLE_OFFSET * normal]
            k = len(face)
            fronts.append(tuple(range(n, n + k)))
            backs.append(tuple(reversed(range(n + k, n + 2 * k))))
            obstacle_groups.append(group)
            n += 2 * k
        out_faces += fronts + backs
        out_groups += obstacle_groups + obstacle_groups
        drawn_sides = None
        if sides is not None:
            # the drawn order of Occluders: the obstacle polygons, then the boundary faces
            drawn_sides = sides.take([k for k, group in zip(source, groups) if group[0] == "obstacle"] + [source[i] for i in boundary])

        return cls(
            vertices=np.vstack(all_vertices),
            faces=tuple(out_faces),
            face_materials=tuple(g[2] for g in out_groups),
            face_names=tuple(g[1] for g in out_groups),
            face_roles=tuple(g[0] for g in out_groups),
            materials={name: materials[name] for name in dict.fromkeys(g[2] for g in out_groups)},
            is_closed=is_closed,
            open_edge=open_edge,
            obstacle_faces=tuple(tuple(face) for face, group in zip(faces, groups) if group[0] == "obstacle"),
            obstacle_face_materials=tuple(group[2] for group in groups if group[0] == "obstacle"),
            mergeable_faces=mergeable,
            check_inside=check_inside,
            air_volume=air_volume,
            drawn_sides=drawn_sides,
        )

    def _face_geometry(self) -> List[Tuple[float, np.ndarray, np.ndarray]]:
        return [_polygon_area_and_centroid(self.vertices[list(face)]) for face in self.faces]

    @property
    def has_volume(self) -> bool:
        """Whether :attr:`volume` is defined: the room closes, or its air volume is given."""
        return self.is_closed or self.air_volume is not None

    @property
    def volume(self) -> float:
        """The air volume: :attr:`air_volume` when given, else the boundary's, less the volume each closed obstacle
        encloses, defined only for a closed room."""
        if self.air_volume is not None:
            return float(self.air_volume)
        if not self.is_closed:
            raise ValueError("The room volume is undefined: the boundary surfaces do not close a volume.")
        boundary = [f for f, role in zip(self.faces, self.face_roles) if role == "boundary"]
        drawn = self._drawn_vertices
        enclosed = 0.0
        for faces in self._closed_obstacles.values():
            flips = _orient(drawn, faces)[0]
            wound = [list(f)[::-1] if flip else list(f) for f, flip in zip(faces, flips)]
            enclosed += abs(_signed_volume(drawn, wound))
        return _signed_volume(self.vertices, boundary) - enclosed

    def _exposed_faces(self) -> List[int]:
        """Indices into ``faces`` of the faces the air touches.

        The boundary faces; of an obstacle that closes a volume (a cabinet) its
        front copies only, since the back copies face its inside; of any other
        obstacle (a panel) both copies. An obstacle copy lying on a boundary
        face (a cabinet's base on the floor) is not exposed, and the boundary
        under a standing obstacle stays counted.
        """
        boundary = [i for i, role in enumerate(self.face_roles) if role == "boundary"]
        count = len(self.obstacle_faces)
        first = len(boundary)
        closed = set(self._closed_obstacles)
        copies = [first + k for k in range(count)]
        copies += [first + count + k for k in range(count) if self.face_names[first + k] not in closed]
        if not copies:
            return boundary
        geometry = self._face_geometry()
        on_boundary = _lying_on(
            np.array([geometry[i][2] for i in copies]),
            [(self.vertices[list(self.faces[i])], geometry[i][1], geometry[i][2]) for i in boundary],
        )
        return boundary + [i for i, lying in zip(copies, on_boundary) if not lying]

    @property
    def surfaces(self) -> List[Tuple[float, SurfaceMaterial]]:
        """``(area, material)`` per material over the faces exposed to the air (see :meth:`_exposed_faces`)."""
        geometry = self._face_geometry()
        areas: Dict[str, float] = {}
        for i in self._exposed_faces():
            name = self.face_materials[i]
            areas[name] = areas.get(name, 0.0) + geometry[i][0]
        surfaces = [(areas[name], self.materials[name]) for name in areas]
        if self.open_area > 0.0:
            surfaces.append((self.open_area, OPEN_MATERIAL))
        return surfaces

    @property
    def surface_area(self) -> float:
        """Total area exposed to the air: Sabine's and Eyring's ``S``, and the ``S`` of the mean free path ``4 V / S``."""
        geometry = self._face_geometry()
        return float(sum(geometry[i][0] for i in self._exposed_faces())) + float(self.open_area)

    def area_normals(self) -> np.ndarray:
        """The unit normal of every exposed face scaled by its area, shape (n_exposed, 3); the boundary faces come first."""
        geometry = self._face_geometry()
        return np.array([geometry[i][0] * geometry[i][1] for i in self._exposed_faces()]).reshape(-1, 3)

    def faces_with_materials(self) -> List[Tuple[np.ndarray, str]]:
        """Every face as ``(vertices (k, 3), material name)``, obstacle copies included."""
        return [(self.vertices[list(face)], name) for face, name in zip(self.faces, self.face_materials)]

    def is_inside(self, point) -> bool:
        """Whether ``point`` is in the air: inside a closed boundary (anywhere in an open scene), outside every closed obstacle.

        Decided by generalised winding numbers over the drawn faces, so the obstacle copies play no part. Always
        True when :attr:`check_inside` is False.
        """
        if not self.check_inside:
            return True
        point = np.asarray(point, dtype=float).reshape(3)
        if self.is_closed and abs(_winding_number(point, self._triangles["boundary"])) < 0.5:
            return False
        return not any(abs(_winding_number(point, triangles)) > 0.5 for triangles in self._triangles["obstacles"])

    @cached_property
    def _drawn_vertices(self) -> np.ndarray:
        """The welded vertices of the drawn faces, without the obstacle copies, which follow them in ``vertices``."""
        first = sum(1 for role in self.face_roles if role == "boundary")
        return self.vertices[: min(self.faces[first])] if self.obstacle_faces else self.vertices

    @cached_property
    def _closed_obstacles(self) -> Dict[str, List[Tuple[int, ...]]]:
        """The drawn faces of each obstacle object whose faces close a volume, by object name."""
        first = sum(1 for role in self.face_roles if role == "boundary")  # the front copies follow, in obstacle_faces order
        by_object: Dict[str, List[Tuple[int, ...]]] = {}
        for k, face in enumerate(self.obstacle_faces):
            by_object.setdefault(self.face_names[first + k], []).append(face)
        return {name: faces for name, faces in by_object.items() if _orient(self._drawn_vertices, faces)[2] is None}

    @cached_property
    def _triangles(self) -> Dict[str, object]:
        """Fan triangles ``(n, 3, 3)`` of the boundary, and of each obstacle object that closes a volume."""
        boundary = [f for f, role in zip(self.faces, self.face_roles) if role == "boundary"]
        return {
            "boundary": _fan_triangles(self.vertices, boundary),
            "obstacles": [_fan_triangles(self.vertices, faces) for faces in self._closed_obstacles.values()],
        }

    def without_obstacles(self) -> RoomGeometry:
        """The enclosure alone."""
        keep = [i for i, role in enumerate(self.face_roles) if role == "boundary"]
        names = tuple(self.face_materials[i] for i in keep)
        n_obstacle = len(self.obstacle_faces)
        sides = None if self.drawn_sides is None else self.drawn_sides.take(range(n_obstacle, n_obstacle + len(keep)))
        return RoomGeometry(
            vertices=self.vertices,
            faces=tuple(self.faces[i] for i in keep),
            face_materials=names,
            face_names=tuple(self.face_names[i] for i in keep),
            face_roles=tuple(self.face_roles[i] for i in keep),
            materials={name: self.materials[name] for name in dict.fromkeys(names)},
            is_closed=self.is_closed,
            open_edge=self.open_edge,
            check_inside=self.check_inside,
            drawn_sides=sides,
        )

    def to_pra(
        self,
        fs: int,
        max_order: int,
        ray_tracing: bool = False,
        air_absorption: bool = False,
        enclosure_only: bool = False,
        air: Optional[AirConditions] = None,
    ) -> pra.Room:
        """The pyroomacoustics room, one wall per face, materials resampled to its octave grid (nine bands at 48 kHz).

        Bands above 8 kHz hold the 8 kHz value, where linear extrapolation could pass 1. A material without
        scattering scatters ``DEFAULT_SCATTERING``. ``enclosure_only`` leaves the obstacles out. The room
        answers ``is_inside`` with :meth:`is_inside`. ``air`` (default ``AirConditions()``) sets the speed of
        sound (Cramer) and, with ``air_absorption``, the ISO 9613-1 absorption, in place of pyroomacoustics'
        own 343 m/s and absorption table.
        """
        with pra_octave_grid():
            octave_bands = pra.acoustics.AntoniOctaveFilterBank(
                fs=int(fs),
                n_fft=pra.constants.get("octave_bands_n_fft"),
                base_frequency=SYNTHESIS_BANDS[0],
            )
            above = [float(f) for f in octave_bands.centers if f > SYNTHESIS_BANDS[-1]]
            freqs = list(SYNTHESIS_BANDS) + above

            def held(values):
                return [float(v) for v in values] + [float(values[-1])] * len(above)

            coefficients = {}
            for name, material in self.materials.items():
                scattering = material.scattering if material.scattering is not None else np.full(len(SYNTHESIS_BANDS), DEFAULT_SCATTERING)
                resampled = pra.Material(
                    energy_absorption={"coeffs": held(material.absorption), "center_freqs": freqs},
                    scattering={"coeffs": held(scattering), "center_freqs": freqs},
                )
                resampled.resample(octave_bands)
                coefficients[name] = (
                    np.asarray(resampled.energy_absorption["coeffs"], dtype=float),
                    np.asarray(resampled.scattering["coeffs"], dtype=float),
                )
            walls = []
            for index, (face, material, face_name, role) in enumerate(zip(self.faces, self.face_materials, self.face_names, self.face_roles)):
                if enclosure_only and role == "obstacle":
                    continue
                absorption, scattering = coefficients[material]
                try:
                    wall = pra.wall_factory(
                        self.vertices[list(face)].T,
                        absorption,
                        scattering,
                        name=f"{face_name}_{index}",
                    )
                except Exception as error:  # RuntimeError for corners off a plane
                    raise ValueError(f"pyroomacoustics cannot build a wall from face {index} of '{face_name}' ({len(face)} corners): {error}") from error
                walls.append(wall)
            try:
                room = pra.Room(
                    walls,
                    fs=int(fs),
                    max_order=max_order,
                    ray_tracing=ray_tracing,
                    air_absorption=air_absorption,
                )
            except Exception as error:  # RuntimeError, ValueError or scipy's QhullError
                raise ValueError(
                    f"pyroomacoustics cannot build a room from these {len(walls)} walls (obstacles count twice): it "
                    f"needs at least four walls that span three dimensions. ({type(error).__name__}: {error})"
                ) from error
            air = air_or_default(air)
            room.set_sound_speed(air.speed_of_sound)
            if air_absorption:
                centres = [float(f) for f in room.octave_bands.centers]
                room.set_air_absorption(coefficients=air.air_attenuation(centres).tolist(), center_freqs=centres)
        # pyroomacoustics counts wall crossings, which the obstacle copies and an open boundary confuse,
        # and refuses sources and hides microphones it calls outside
        crossings = room.is_inside

        def is_inside(p, include_borders=True):
            # its own test still runs, unused: it draws from the random generator, and skipping
            # the draws would change every seeded ray-traced response
            crossings(p, include_borders)
            return self.is_inside(p)

        room.is_inside = is_inside
        if self.air_volume is not None:
            # the ray-traced tail's echo density follows the air volume, which pyroomacoustics' own
            # divergence sum misreads on a model that does not close
            volume = float(self.air_volume)
            room.get_volume = lambda: volume
        return room


# Coplanarity tolerance in metres: pyroomacoustics stores corners as float32 and rejects a wall whose
# corners are further than this from a plane
_MERGE_TOLERANCE = 1e-5


def _merge_coplanar(
    vertices: np.ndarray,
    faces: list[list[int]],
    names: list[str],
    obstacles: tuple[str, ...],
) -> tuple[list[list[int]], list[str]]:
    """Join neighbouring coplanar faces of one group into convex polygons.

    Every extra wall multiplies the image source search. Two faces are joined only when they belong to the
    same group, share exactly one edge no other face uses, lie in one plane within ``_MERGE_TOLERANCE`` and
    give a convex polygon, a valid wall that cannot close a ring around a hole. Vertices in the middle of a
    straight side are dropped; the closedness check reads any T-junction that leaves. Obstacle faces may join
    across opposite windings, being two-sided anyway; boundary faces must already agree.
    """
    faces = [list(f) for f in faces]
    alive = [True] * len(faces)

    def normal_of(face):
        return _polygon_area_and_centroid(vertices[face])[1]

    def clean(poly, normal):
        """Drop vertices on a straight side; None if the result is not convex."""
        changed = True
        while changed and len(poly) > 3:
            changed = False
            for k in range(len(poly)):
                u, v, w = (vertices[poly[k - 1]], vertices[poly[k]], vertices[poly[(k + 1) % len(poly)]])
                side = w - u
                if np.linalg.norm(np.cross(v - u, side)) <= _MERGE_TOLERANCE * np.linalg.norm(side):
                    del poly[k]
                    changed = True
                    break
        pts = vertices[poly]
        turn = np.cross(pts - np.roll(pts, 1, axis=0), np.roll(pts, -1, axis=0) - pts) @ normal
        return poly if len(poly) >= 3 and np.all(turn > 0) else None

    def merge(i, j, a, b):
        """Faces i and j across edge a-b, or None."""
        fa, fb = faces[i], faces[j]
        na, nb = normal_of(fa), normal_of(fb)
        if na @ nb < 0:
            if names[i] not in obstacles:
                return None
            fb, nb = fb[::-1], -nb
        if na @ nb < 1.0 - 1e-8:
            return None
        if np.max(np.abs((vertices[fb] - vertices[fa[0]]) @ na)) > _MERGE_TOLERANCE:
            return None
        if len(set(fa) & set(fb)) != 2:
            return None
        ka = fa.index(a)
        if fa[(ka + 1) % len(fa)] != b:
            fa = fa[::-1]
            fb = fb[::-1]
            na = -na
            ka = fa.index(a)
        kb = fb.index(b)
        if fb[(kb + 1) % len(fb)] != a:
            return None
        # a -> b in fa, b -> a in fb: walk fa from b round to a, then fb's
        # vertices strictly between a and b.
        path_a = [fa[(ka + 1 + k) % len(fa)] for k in range(len(fa))]  # b ... a
        path_b = [fb[(kb + 2 + k) % len(fb)] for k in range(len(fb) - 2)]
        poly = clean(path_a + path_b, na)
        if poly is None:
            return None
        # keep the winding face i had, which is what the enclosure relies on
        return poly if normal_of(poly) @ normal_of(faces[i]) > 0 else poly[::-1]

    merged_any = True
    while merged_any:
        merged_any = False
        users: dict[frozenset[int], list[int]] = {}
        for i, face in enumerate(faces):
            if alive[i]:
                for a, b in zip(face, face[1:] + face[:1]):
                    users.setdefault(frozenset((a, b)), []).append(i)
        touched = set()
        for edge, owners in users.items():
            if len(owners) != 2:
                continue
            i, j = owners
            if i in touched or j in touched or names[i] != names[j]:
                continue
            a, b = tuple(edge)
            poly = merge(i, j, a, b)
            if poly is None:
                continue
            faces[i], alive[j] = poly, False
            touched |= {i, j}
            merged_any = True

    keep = [i for i in range(len(faces)) if alive[i]]
    return [faces[i] for i in keep], [names[i] for i in keep]


def _lying_on(points: np.ndarray, polygons) -> List[bool]:
    """Whether each point is within ``2 * OBSTACLE_OFFSET`` of a polygon's plane and inside the polygon.

    ``polygons`` are ``(corners (k, 3), unit normal, centroid)``. The test inside is an even-odd
    crossing count in the polygon's plane.
    """
    if not polygons:
        return [False] * len(points)
    normals = np.array([normal for _, normal, _ in polygons])
    centroids = np.array([centroid for _, _, centroid in polygons])
    near = np.abs(np.einsum("pfk,fk->pf", points[:, None, :] - centroids[None], normals)) < 2.0 * OBSTACLE_OFFSET
    lying = []
    for p, candidates in zip(points, near):
        found = False
        for j in np.flatnonzero(candidates):
            corners, normal, _ = polygons[j]
            u = corners[1] - corners[0]
            u = (u - (u @ normal) * normal) / np.linalg.norm(u - (u @ normal) * normal)
            v = np.cross(normal, u)
            x, y = (corners - p) @ u, (corners - p) @ v
            x2, y2 = np.roll(x, -1), np.roll(y, -1)
            straddles = (y > 0.0) != (y2 > 0.0)
            with np.errstate(divide="ignore", invalid="ignore"):
                crossing = x + (0.0 - y) * (x2 - x) / (y2 - y)
            if int(np.count_nonzero(straddles & (crossing > 0.0))) % 2 == 1:
                found = True
                break
        lying.append(found)
    return lying


def _fan_triangles(vertices: np.ndarray, faces) -> np.ndarray:
    """Every polygon as a fan of triangles, shape (n, 3, 3)."""
    triangles = [(vertices[face[0]], vertices[face[k]], vertices[face[k + 1]]) for face in faces for k in range(1, len(face) - 1)]
    return np.asarray(triangles, dtype=float).reshape(-1, 3, 3)


def _winding_number(point: np.ndarray, triangles: np.ndarray) -> float:
    """Generalised winding number of a triangle soup around ``point``, from the Van Oosterom and Strackee (1983) solid angles.

    About +-1 inside a closed, consistently wound surface and 0 outside, whichever way it is wound.
    """
    if triangles.size == 0:
        return 0.0
    a, b, c = (triangles[:, k] - point for k in range(3))
    la, lb, lc = (np.linalg.norm(x, axis=1) for x in (a, b, c))
    numerator = np.einsum("ij,ij->i", a, np.cross(b, c))
    denominator = la * lb * lc + np.einsum("ij,ij->i", a, b) * lc + np.einsum("ij,ij->i", a, c) * lb + np.einsum("ij,ij->i", b, c) * la
    return float(np.sum(2.0 * np.arctan2(numerator, denominator)) / (4.0 * np.pi))


def _merged(vertices, faces, groups):
    """Coplanar faces joined within one object, role and material."""
    keys = ["\x00".join(group) for group in groups]
    obstacles = tuple(sorted({key for key, group in zip(keys, groups) if group[0] == "obstacle"}))
    faces, keys = _merge_coplanar(vertices, faces, keys, obstacles)
    return faces, [tuple(key.split("\x00")) for key in keys]
