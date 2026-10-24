"""Ray casting against a closed triangulated body, for the geometric scattering method.

:class:`RayCaster` finds, for many rays at once, the first triangle each one hits. It wraps ``trimesh``'s ray intersector, which uses
Embree (the ``embreex`` package) when it is installed; ``trimesh`` is imported when a :class:`RayCaster` is built, never at module level,
so :mod:`compas_acoustics.lab` still imports where it is missing (Rhino, Python 3.9).

Embree works in single precision (relative resolution 1.2e-7), which shapes two details. The body is shifted to the centre of its bounding
box before it is handed over, so the coordinates Embree sees are of the order of the body's size whatever its position. Each ray starts
1e-5 of the body's bounding size beyond its origin, so a ray leaving a bounce point does not hit the face it just left (an offset of
1e-9, below single precision, leaves the origin on that face and the ray hits it at distance zero). The returned hit points are
recomputed in double precision from the ray and the plane of the hit triangle, so they are not limited to single precision.

Throughput, measured on 2026-10-24 (macOS arm64, Python 3.13, trimesh 5.1.1, embreex 4.4.0.post1): about 2.7e6 rays/s (2.6 to 2.8e6
over runs) for 200 000 random rays aimed at the centre of an icosphere of 20 480 faces (``test_throughput_is_recorded``,
``pytest -m slow -s``).
"""

import numpy as np

__all__ = ["RayCaster"]

_INSTALL_HINT = "ray casting needs trimesh and an Embree backend; install them with: pip install compas_acoustics[lab]"


class RayCaster:
    """First-hit ray casting against a closed triangulated body.

    Parameters
    ----------
    body : :class:`compas.datastructures.Mesh`
        A closed body (as ``ScatteringSample.sample``); polygonal faces are triangulated.

    Attributes
    ----------
    normals : :class:`numpy.ndarray`
        The outward unit normals of the triangles, shape (n_triangles, 3); zero for a zero-area triangle.
        The triangles are numbered as ``body.to_vertices_and_faces(triangulated=True)`` lists them.

    Raises
    ------
    ImportError
        If ``trimesh`` or its ray backend is missing.
    """

    def __init__(self, body):
        try:
            import trimesh
        except ImportError as error:
            raise ImportError(_INSTALL_HINT) from error
        vertices, faces = body.to_vertices_and_faces(triangulated=True)
        self._vertices = np.asarray(vertices, dtype=float)
        self._faces = np.asarray(faces, dtype=int).reshape(-1, 3)
        self._centre = 0.5 * (self._vertices.min(axis=0) + self._vertices.max(axis=0))
        self._mesh = trimesh.Trimesh(self._vertices - self._centre, self._faces, process=False)
        try:
            self._mesh.ray  # noqa: B018  # builds the intersector, which imports the backend
        except ImportError as error:
            raise ImportError(_INSTALL_HINT) from error
        triangles = self._vertices[self._faces]
        cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
        twice = np.linalg.norm(cross, axis=1)
        self.normals = np.where(twice[:, None] > 0, cross / np.where(twice > 0, twice, 1.0)[:, None], 0.0)
        self._offset = 1e-5 * float(np.ptp(self._vertices, axis=0).max())

    def first_hits(self, origins, directions, exclude=None):
        # type: (np.ndarray, np.ndarray, object) -> tuple
        """The first triangle each ray hits, and where.

        Parameters
        ----------
        origins : :class:`numpy.ndarray`
            Ray origins, shape (n, 3).
        directions : :class:`numpy.ndarray`
            Unit ray directions, shape (n, 3).
        exclude : :class:`numpy.ndarray`, optional
            Shape (n,) int: the triangle each ray has just left (-1 for none), for rays that start on the body. A hit on it is never
            returned for that ray: a ray leaving a planar triangle cannot meet its plane again, so such a hit is a numerical self-hit.
            Without it, a ray leaving a face at a grazing angle (sine of the elevation below about 1e-2) can report that face.

        Returns
        -------
        tuple of :class:`numpy.ndarray`
            ``(triangle, point)``: ``triangle`` shape (n,) int, the index of the first triangle hit, -1 where the ray hits nothing;
            ``point`` shape (n, 3), the hit point (zeros where the ray misses).

        Notes
        -----
        A ray starts ``offset`` (1e-5 of the body's size) along its direction. That lifts it above the plane of the face it leaves by only
        ``offset · |d · n|``, below single precision for a grazing ray. The start of a ray with an ``exclude`` triangle is therefore also
        lifted ``offset`` along that triangle's normal, to the side the ray leaves it, a displacement of the ray of the order of ``offset``.
        A ray still reporting its excluded triangle is lifted ten times as far, up to three times; one that still does is reported as a miss.
        """
        origins = np.asarray(origins, dtype=float).reshape(-1, 3)
        directions = np.asarray(directions, dtype=float).reshape(-1, 3)
        n = len(origins)
        triangle = np.full(n, -1, dtype=int)
        point = np.zeros((n, 3))
        if n == 0:
            return triangle, point
        if exclude is None:
            exclude = np.full(n, -1, dtype=int)
        exclude = np.asarray(exclude, dtype=int).reshape(-1)
        if len(exclude) != n:
            raise ValueError("exclude has {} entries for {} rays".format(len(exclude), n))
        starts = origins - self._centre + self._offset * directions
        has = exclude >= 0
        side = np.ones(n)
        side[has] = np.where(np.einsum("ij,ij->i", directions[has], self.normals[exclude[has]]) < 0, -1.0, 1.0)
        lift = np.zeros((n, 3))
        lift[has] = side[has, None] * self.normals[exclude[has]]
        found = np.full(n, -1, dtype=int)
        pending = np.arange(n)
        for attempt in range(4):
            ray_index, tri_index = self._first_hit_ids(starts[pending] + self._offset * 10.0**attempt * lift[pending], directions[pending])
            found[pending] = -1
            found[pending[ray_index]] = tri_index
            again = pending[has[pending] & (found[pending] == exclude[pending])]
            if len(again) == 0:
                break
            pending = again
        found[has & (found == exclude)] = -1
        hit = np.flatnonzero(found >= 0)
        if len(hit) == 0:
            return triangle, point
        # the hit point in double precision: the ray against the plane of the triangle it hit
        tri_index = found[hit]
        o = origins[hit]
        d = directions[hit]
        normal = self.normals[tri_index]
        corner = self._vertices[self._faces[tri_index, 0]]
        denominator = np.einsum("ij,ij->i", d, normal)
        usable = np.abs(denominator) > 1e-300
        t = np.einsum("ij,ij->i", corner - o, normal) / np.where(usable, denominator, 1.0)
        triangle[hit] = tri_index
        point[hit] = o + t[:, None] * d
        return triangle, point

    def _first_hit_ids(self, starts, directions):
        """The indices of the rays that hit, and of the triangle each hit first (Embree's nearest hit)."""
        tri_index, ray_index = self._mesh.ray.intersects_id(starts, directions, multiple_hits=False, return_locations=False)[:2]
        return np.asarray(ray_index, dtype=int), np.asarray(tri_index, dtype=int)
