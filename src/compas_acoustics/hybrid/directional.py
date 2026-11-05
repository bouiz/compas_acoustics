"""Direction-dependent boundary absorption on the delay line outputs.

Wendt, van de Par and Ewert (2014, Sec. 1.2) weight "the signal of each incidence direction ... with the
reflection coefficient of the room wall corresponding to the respective parallel cube side face". Kirsch, Wendt
and van de Par (2023, Sec. 1.3, Eqs. 4-5) generalise it to arbitrary geometry, which holds for a mesh: rays from
the receiver sample the absorption of the first face they meet, and each line takes a weighted mean,

    w_s,k = max(v_s . v_k, 0) ** beta,   beta = K / 6,
    a_k   = sum_s w_s,k alpha_s / sum_s w_s,k,

``v_s`` the scanned directions, ``v_k`` the line's, ``K`` the number of lines (a -3 dB lobe of 90 degrees at six).
The line's output filter has the magnitude ``sqrt(1 - a_k)`` per band. It sits outside the loop, so every line
still decays at ``target.rt``; the late field loses one reflection's absorption, ``10 log10(mean_k (1 - a_k))``
dB, left uncompensated as in both papers.
"""

from __future__ import annotations

import numpy as np
import pyfar as pf
import pyFDN
import spharpy

from .bands import N_BANDS
from .pair import Pair
from .scattering import _first_hit

SCAN_DIRECTIONS = 96  # the highest resolution of Kirsch et al. (2023), as the scan grid


def scan_directions(n_scan: int = SCAN_DIRECTIONS) -> pf.Coordinates:
    """The directions the boundary is sampled along, shape (n_scan,)."""
    directions = spharpy.samplings.equal_area(n_max=0, n_points=n_scan)
    if directions.cshape[0] != n_scan:
        raise ValueError(f"equal_area returned {directions.cshape[0]} directions for n_scan={n_scan}; pass a count it can produce exactly")
    return directions


def scanned_absorption(spec: Pair, directions: pf.Coordinates, *, n_scan: int = SCAN_DIRECTIONS) -> np.ndarray:
    """Absorption coefficient per delay line and band, shape (n_lines, 8).

    Each scan direction takes the absorption of the first face its ray from the receiver meets, which holds in
    a concave room too.
    """
    faces = spec.geometry.faces_with_materials()
    scan = scan_directions(n_scan).cartesian
    scan = scan / np.linalg.norm(scan, axis=1, keepdims=True)

    hit = _first_hit(faces, spec.receiver, scan)
    if np.any(hit < 0):
        raise RuntimeError(f"{int(np.sum(hit < 0))} of {hit.size} scanned directions meet no boundary; the room is not closed around the receiver")
    alpha = np.stack([spec.geometry.materials[name].absorption for _, name in faces])
    alpha = alpha[hit][:, :N_BANDS]  # (n_scan, 8)

    unit = directions.cartesian
    unit = unit / np.linalg.norm(unit, axis=1, keepdims=True)
    if spec.receiver_rotation is not None:
        unit = unit @ spec.receiver_rotation.T  # the scan is in room coordinates
    beta = unit.shape[0] / 6.0
    weights = np.clip(scan @ unit.T, 0.0, None) ** beta  # (n_scan, n_lines)
    return (weights.T @ alpha) / weights.sum(axis=0)[:, np.newaxis]


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
    return pyFDN.gain_to_geq(np.hstack((db[:, :1], db, db[:, -1:])).T, float(fs))


def directional_absorption_filters(spec: Pair, directions: pf.Coordinates, *, n_scan: int = SCAN_DIRECTIONS) -> np.ndarray:
    """The per-line output filters of this room, the ``line_eq`` of :func:`~compas_acoustics.hybrid.late.render`."""
    return absorption_filters(scanned_absorption(spec, directions, n_scan=n_scan), spec.fs)
