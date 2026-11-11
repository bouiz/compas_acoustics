"""Ray / triangle queries for the simulation mesh pipeline.

``first_hits`` is the hot path (the air-visibility random walk shoots
hundreds of thousands of rays). It uses a numba-parallel brute-force
Möller–Trumbore kernel when numba is available and a chunked numpy kernel
otherwise. The remaining helpers handle a handful of queries and are plain
vectorized numpy.

Ported from compas_soundscape (``backend/utils/mesh_raycast.py``).
"""

from __future__ import annotations

import numpy as np

try:  # numba is optional: a pure-numpy path runs without it.
    from numba import njit
    from numba import prange

    _HAS_NUMBA = True
except ImportError:  # pragma: no cover - exercised only without numba
    _HAS_NUMBA = False

# Determinant below which a ray is considered parallel to a triangle.
_PARALLEL_EPS = 1e-12
# Rays per chunk in the numpy fallback (bounds the [chunk x n_faces] buffers).
_NUMPY_CHUNK = 256


class TriangleSet:
    """Pre-computed edge vectors of a triangle soup (float64)."""

    def __init__(self, verts: np.ndarray, faces: np.ndarray):
        p = np.asarray(verts, dtype=np.float64)[np.asarray(faces, dtype=np.int64)]
        self.v0 = np.ascontiguousarray(p[:, 0])
        self.e1 = np.ascontiguousarray(p[:, 1] - p[:, 0])
        self.e2 = np.ascontiguousarray(p[:, 2] - p[:, 0])
        self.centroids = p.mean(axis=1)

    def __len__(self) -> int:
        return len(self.v0)


if _HAS_NUMBA:

    @njit(parallel=True, cache=True)
    def _first_hits_numba(v0, e1, e2, origins, dirs, tmin, out_t, out_f):  # pragma: no cover - jit
        n_rays = origins.shape[0]
        n_faces = v0.shape[0]
        for r in prange(n_rays):
            ox, oy, oz = origins[r, 0], origins[r, 1], origins[r, 2]
            dx, dy, dz = dirs[r, 0], dirs[r, 1], dirs[r, 2]
            best = np.inf
            best_f = -1
            for f in range(n_faces):
                e1x, e1y, e1z = e1[f, 0], e1[f, 1], e1[f, 2]
                e2x, e2y, e2z = e2[f, 0], e2[f, 1], e2[f, 2]
                px = dy * e2z - dz * e2y
                py = dz * e2x - dx * e2z
                pz = dx * e2y - dy * e2x
                det = e1x * px + e1y * py + e1z * pz
                if -_PARALLEL_EPS < det < _PARALLEL_EPS:
                    continue
                inv = 1.0 / det
                tx = ox - v0[f, 0]
                ty = oy - v0[f, 1]
                tz = oz - v0[f, 2]
                u = (tx * px + ty * py + tz * pz) * inv
                if u < 0.0 or u > 1.0:
                    continue
                qx = ty * e1z - tz * e1y
                qy = tz * e1x - tx * e1z
                qz = tx * e1y - ty * e1x
                v = (dx * qx + dy * qy + dz * qz) * inv
                if v < 0.0 or u + v > 1.0:
                    continue
                t = (e2x * qx + e2y * qy + e2z * qz) * inv
                if tmin < t < best:
                    best = t
                    best_f = f
            out_t[r] = best
            out_f[r] = best_f


def ray_triangle_t(tris: TriangleSet, o: np.ndarray, d: np.ndarray) -> np.ndarray:
    """
    Hit distance of rays against every triangle, numpy-vectorized.

    Args:
        o, d: (R, 3) origins and directions (directions need not be unit).
    Returns:
        (R, F) array of ``t`` (``inf`` where there is no hit).
    """
    pvec = np.cross(d[:, None, :], tris.e2[None, :, :])
    det = np.einsum("fk,rfk->rf", tris.e1, pvec)
    ok = np.abs(det) > _PARALLEL_EPS
    inv = np.divide(1.0, det, out=np.zeros_like(det), where=ok)
    tvec = o[:, None, :] - tris.v0[None, :, :]
    u = np.einsum("rfk,rfk->rf", tvec, pvec) * inv
    qvec = np.cross(tvec, tris.e1[None, :, :])
    v = np.einsum("rk,rfk->rf", d, qvec) * inv
    t = np.einsum("fk,rfk->rf", tris.e2, qvec) * inv
    hit = ok & (u >= 0) & (u <= 1) & (v >= 0) & (u + v <= 1)
    return np.where(hit, t, np.inf)


def first_hits(tris: TriangleSet, origins: np.ndarray, dirs: np.ndarray, tmin: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Closest hit of each ray beyond ``tmin``.

    Returns:
        ``(t (R,), face (R,))``; ``face == -1`` (and ``t == inf``) on a miss.
    """
    origins = np.ascontiguousarray(origins, dtype=np.float64)
    dirs = np.ascontiguousarray(dirs, dtype=np.float64)
    n = len(origins)
    out_t = np.full(n, np.inf)
    out_f = np.full(n, -1, dtype=np.int64)
    if n == 0 or len(tris) == 0:
        return out_t, out_f

    if _HAS_NUMBA:
        _first_hits_numba(tris.v0, tris.e1, tris.e2, origins, dirs, float(tmin), out_t, out_f)
        return out_t, out_f

    for a in range(0, n, _NUMPY_CHUNK):
        t = ray_triangle_t(tris, origins[a : a + _NUMPY_CHUNK], dirs[a : a + _NUMPY_CHUNK])
        t[t <= tmin] = np.inf
        f = np.argmin(t, axis=1)
        tt = t[np.arange(len(f)), f]
        out_t[a : a + _NUMPY_CHUNK] = tt
        out_f[a : a + _NUMPY_CHUNK] = np.where(np.isfinite(tt), f, -1)
    return out_t, out_f


def segment_crossings(tris: TriangleSet, a, b, end_margin: float) -> np.ndarray:
    """
    Faces strictly crossed by the segment ``a → b``.

    Hits within ``end_margin`` metres of either endpoint are ignored, so a
    source or receiver resting on a surface does not count as blocked.
    """
    a = np.asarray(a, dtype=np.float64).reshape(1, 3)
    b = np.asarray(b, dtype=np.float64).reshape(1, 3)
    d = b - a
    length = float(np.linalg.norm(d))
    if length == 0 or len(tris) == 0:
        return np.zeros(0, dtype=np.int64)
    t = ray_triangle_t(tris, a, d)[0]
    margin = end_margin / length
    return np.flatnonzero((t > margin) & (t < 1.0 - margin))


def point_triangle_distances(tris: TriangleSet, point) -> np.ndarray:
    """
    Distance from ``point`` to every triangle (closest-point on triangle,
    Ericson, Real-Time Collision Detection §5.1.5), vectorized over faces.
    """
    p = np.asarray(point, dtype=np.float64)
    a = tris.v0
    ab = tris.e1
    ac = tris.e2
    ap = p - a
    d1 = np.einsum("ij,ij->i", ab, ap)
    d2 = np.einsum("ij,ij->i", ac, ap)
    bp = ap - ab
    d3 = np.einsum("ij,ij->i", ab, bp)
    d4 = np.einsum("ij,ij->i", ac, bp)
    cp = ap - ac
    d5 = np.einsum("ij,ij->i", ab, cp)
    d6 = np.einsum("ij,ij->i", ac, cp)

    va = d3 * d6 - d5 * d4
    vb = d5 * d2 - d1 * d6
    vc = d1 * d4 - d3 * d2
    denom = va + vb + vc
    safe = np.where(np.abs(denom) > 0, denom, 1.0)
    v = vb / safe
    w = vc / safe
    closest = a + ab * v[:, None] + ac * w[:, None]  # interior region

    def _edge(mask, start, direction, num, den):
        tt = np.clip(np.divide(num, den, out=np.zeros_like(num), where=den != 0), 0.0, 1.0)
        closest[mask] = (start + direction * tt[:, None])[mask]

    # Edge regions (BC, AC, AB) then vertex regions; later assignments win.
    _edge((va <= 0) & (d4 - d3 >= 0) & (d5 - d6 >= 0), a + ab, ac - ab, d4 - d3, (d4 - d3) + (d5 - d6))
    _edge((vb <= 0) & (d2 >= 0) & (d6 <= 0), a, ac, d2, d2 - d6)
    _edge((vc <= 0) & (d1 >= 0) & (d3 <= 0), a, ab, d1, d1 - d3)
    closest[(d6 >= 0) & (d5 <= d6)] = (a + ac)[(d6 >= 0) & (d5 <= d6)]
    closest[(d3 >= 0) & (d4 <= d3)] = (a + ab)[(d3 >= 0) & (d4 <= d3)]
    closest[(d1 <= 0) & (d2 <= 0)] = a[(d1 <= 0) & (d2 <= 0)]
    return np.linalg.norm(closest - p, axis=1)


def uniform_sphere_directions(n: int, rng: np.random.Generator) -> np.ndarray:
    """``n`` unit vectors uniformly distributed on the sphere."""
    v = rng.normal(size=(n, 3))
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def cosine_hemisphere_directions(normals: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Lambertian (cosine-weighted) directions around each unit normal."""
    n = len(normals)
    u1 = rng.random(n)
    u2 = rng.random(n)
    r = np.sqrt(u1)
    phi = 2.0 * np.pi * u2
    local = np.stack([r * np.cos(phi), r * np.sin(phi), np.sqrt(np.maximum(0.0, 1.0 - u1))], axis=1)
    # Orthonormal basis around each normal.
    helper = np.where(np.abs(normals[:, :1]) < 0.9, [[1.0, 0.0, 0.0]], [[0.0, 1.0, 0.0]])
    t1 = np.cross(normals, helper)
    t1 /= np.linalg.norm(t1, axis=1, keepdims=True)
    t2 = np.cross(normals, t1)
    return local[:, :1] * t1 + local[:, 1:2] * t2 + local[:, 2:3] * normals
