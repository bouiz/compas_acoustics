"""Direction-dependent boundary absorption on the delay line outputs.

Wendt et al. (2014, Sec. 1.2) weight "the signal of each incidence direction ... with the
reflection coefficient of the room wall corresponding to the respective parallel cube side face". Kirsch et al.
(2023, Sec. 1.3, Eqs. 4-5) generalise it to arbitrary geometry, which holds for a mesh: rays from
the receiver sample the absorption of the first face they meet, and with fewer than their 96 scanned directions
each line takes a weighted mean,

    w_s,k = max(v_s . v_k, 0) ** beta,   beta = K / 6,
    a_k   = sum_s w_s,k alpha_s / sum_s w_s,k,

``v_s`` the scanned directions, ``v_k`` the line's, ``K`` the number of lines (the paper gives a -3 dB lobe of 90
degrees for K = 6). With 96 lines each takes the absorption of its own scanned direction there; the weighted mean
at every ``K``, 96 included, is the package's choice (the scan is fixed in the room, the lines turn with the
receiver). The line's output filter has the magnitude ``sqrt(1 - a_k)`` per band. It sits outside the loop, so
every line still decays at ``target.rt``; the late field loses one reflection's absorption,
``10 log10(mean_k (1 - a_k))`` dB, left uncompensated; neither paper describes a compensation.
"""

from __future__ import annotations

import numpy as np
import pyfar as pf
import pyFDN

from compas_acoustics.room.geometry import _polygon_area_and_centroid

from .bands import N_BANDS
from .late import fdn_directions
from .pair import Pair


def scanned_absorption(spec: Pair, directions: pf.Coordinates) -> np.ndarray:
    """Absorption coefficient per delay line and band, shape (n_lines, 8).

    Each scan direction takes the absorption of the first face its ray from the receiver meets, which holds in
    a concave room too. The scan is the 96-point grid of the lines, in room coordinates, as the 96 scanned
    directions of Kirsch et al. (2023, Sec. 1.3).
    """
    faces = spec.geometry.faces_with_materials()
    scan = fdn_directions(96).cartesian
    scan = scan / np.linalg.norm(scan, axis=1, keepdims=True)

    hit = _first_hit(faces, spec.receiver, scan)
    if np.any(hit < 0):
        raise RuntimeError(f"{int(np.sum(hit < 0))} of {hit.size} scanned directions meet no boundary; the room is not closed around the receiver")
    alpha = np.stack([spec.geometry.materials[name].absorption for _, name in faces])
    alpha = alpha[hit][:, :N_BANDS]  # (96, 8)

    unit = directions.cartesian
    unit = unit / np.linalg.norm(unit, axis=1, keepdims=True)
    unit = unit @ spec.receiver_rotation.T  # the scan is in room coordinates
    beta = unit.shape[0] / 6.0
    weights = np.clip(scan @ unit.T, 0.0, None) ** beta  # (96, n_lines)
    return (weights.T @ alpha) / weights.sum(axis=0)[:, np.newaxis]


def _first_hit(faces: list[tuple[np.ndarray, str]], origin: np.ndarray, directions: np.ndarray) -> np.ndarray:
    """Index of the first face each ray from ``origin`` meets, -1 for none.

    Points within a small tolerance of a face outline count as on it, because the receiver's scan directions in a
    room with round coordinates routinely aim exactly at the seam between two faces.
    """
    directions = directions / np.linalg.norm(directions, axis=1, keepdims=True)
    extent = max(float(np.ptp(np.vstack([v for v, _ in faces]), axis=0).max()), 1.0)
    tol = 1e-6 * extent
    best_t = np.full(directions.shape[0], np.inf)
    best = np.full(directions.shape[0], -1, dtype=int)
    for index, (vertices, _) in enumerate(faces):
        normal = _polygon_area_and_centroid(vertices)[1]
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


def absorption_filters(absorption: np.ndarray, fs: float) -> np.ndarray:
    """Output filters of magnitude ``sqrt(1 - a)``, shape (n_sections, 6, n_lines).

    One graphic EQ per delay line on the band grid, its two extrapolated edges held at the outermost bands.
    """
    absorption = np.asarray(absorption, dtype=float)
    if absorption.ndim != 2 or absorption.shape[1] != N_BANDS:
        raise ValueError(f"absorption must be (n_lines, {N_BANDS})")
    if np.any(absorption < 0.0) or np.any(absorption >= 1.0):
        raise ValueError("absorption coefficients must lie in [0, 1)")

    db = 10.0 * np.log10(1.0 - absorption)  # pressure, 20 log10 sqrt(1 - a)
    return pyFDN.gain_to_graphic_eq(np.hstack((db[:, :1], db, db[:, -1:])).T, float(fs))


def directional_absorption_filters(spec: Pair, directions: pf.Coordinates) -> np.ndarray:
    """The per-line output filters of this room, the ``line_eq`` of :func:`~compas_acoustics.hybrid.late.render`."""
    return absorption_filters(scanned_absorption(spec, directions), spec.fs)
