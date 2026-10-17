"""Ray casts against the faces of a room.

Ported from hybrid-rir (gpapadimitri/hybrid-rir @ 8a70661, src/scattering.py).
Only ``_unit_normal`` and ``_first_hit``, which :mod:`directional` uses; the
scattered image sources of that module are not ported.
"""

from __future__ import annotations

import numpy as np


def _unit_normal(vertices: np.ndarray) -> np.ndarray:
    normal = 0.5 * np.cross(vertices, np.roll(vertices, -1, axis=0)).sum(axis=0)
    return normal / np.linalg.norm(normal)


def _first_hit(
    faces: list[tuple[np.ndarray, str]], origin: np.ndarray, directions: np.ndarray
) -> np.ndarray:
    """Index of the first face each ray from ``origin`` meets, -1 for none.

    Points within a small tolerance of a face outline count as on it, because
    image sources of rooms with round coordinates routinely aim exactly at the
    seam between two faces.
    """
    directions = directions / np.linalg.norm(directions, axis=1, keepdims=True)
    extent = max(float(np.ptp(np.vstack([v for v, _ in faces]), axis=0).max()), 1.0)
    tol = 1e-6 * extent
    best_t = np.full(directions.shape[0], np.inf)
    best = np.full(directions.shape[0], -1, dtype=int)
    for index, (vertices, _) in enumerate(faces):
        normal = _unit_normal(vertices)
        denom = directions @ normal
        with np.errstate(divide="ignore", invalid="ignore"):
            t = ((vertices[0] - origin) @ normal) / denom
        candidate = (np.abs(denom) > 1e-12) & (t > tol) & (t < best_t - tol)
        if not np.any(candidate):
            continue
        rows = np.flatnonzero(candidate)
        points = origin + t[rows, np.newaxis] * directions[rows]
        keep = [i for i in range(3) if i != int(np.argmax(np.abs(normal)))]
        q = points[:, keep]
        poly = vertices[:, keep]
        inside = np.zeros(rows.size, dtype=bool)
        near = np.zeros(rows.size, dtype=bool)
        for (ax, ay), (bx, by) in zip(poly, np.roll(poly, -1, axis=0)):
            crosses = (ay > q[:, 1]) != (by > q[:, 1])
            with np.errstate(divide="ignore", invalid="ignore"):
                x_cross = (bx - ax) * (q[:, 1] - ay) / (by - ay) + ax
            inside ^= crosses & (q[:, 0] < x_cross)
            edge = np.array([bx - ax, by - ay])
            along = np.clip(((q - [ax, ay]) @ edge) / float(edge @ edge), 0.0, 1.0)
            closest = np.array([ax, ay]) + along[:, np.newaxis] * edge
            near |= np.linalg.norm(q - closest, axis=1) <= tol
        hit = rows[inside | near]
        best_t[hit] = t[hit]
        best[hit] = index
    return best
