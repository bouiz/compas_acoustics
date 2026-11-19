"""Surface scattering of image sources, and ray casts against the faces of a room.

After Ewert, Goessling, Buttler, van de Par and Hu, "Computationally-efficient rendering of diffuse reflections
for geometrical acoustics based room simulation", Acta Acustica 9, 9 (2025).
:func:`~compas_acoustics.hybrid.early.image_sources` splits each image into a specular and a scattered part;
here are the room that split needs (:func:`complementary_scattering`), the scattered part's temporal spread
(Eqs. 12-13) and its weights on the delay lines.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pyfar as pf
from scipy.signal import lfilter

from ..room.geometry import DEFAULT_SCATTERING


def complementary_scattering(geometry):
    """``geometry`` with every material's scattering ``s`` replaced by ``1 - s``.

    Built for ray tracing, an image's damping takes ``sqrt(1 - s)`` for every wall of non-zero scattering it
    reflects off, so the room gives the specular part, ``prod sqrt(1 - s_i)``, and this one the scattered part,
    ``prod sqrt(s_i)``, both exact at ``s = 0`` and 1. A material without coefficients scatters ``DEFAULT_SCATTERING``.
    """
    return replace(
        geometry,
        materials={
            name: replace(
                material,
                scattering=1.0 - (np.full(len(material.absorption), DEFAULT_SCATTERING) if material.scattering is None else np.asarray(material.scattering, dtype=float)),
            )
            for name, material in geometry.materials.items()
        },
    )


def scattering_decay_time(distance: float, speed_of_sound: float) -> float:
    """Local decay time ``T_s`` of an ideal diffuse reflection, in seconds.

    Ewert et al. (2025), Sec. 2.1.2: ``T_s = 6 (10**(1/5) - 1) 2R / c``, about ``7 R / c``, for a reflector at
    distance ``R``; for a general room "the mean free path length".
    """
    return 6.0 * (10.0**0.2 - 1.0) * 2.0 * distance / speed_of_sound


def allpass_delays(decay_time: float, fs: float) -> np.ndarray:
    """Delays in samples of the four-allpass cascade, Ewert et al. (2025) Eq. 13.

    ``tau_s = T_s log10(1/g) / 3`` and ``tau_i = tau_s fs / eta**(i-1)``, ``g = 1/sqrt(2)``, ``eta = pi``, rounded,
    never below one sample.
    """
    tau_s = decay_time * np.log10(np.sqrt(2.0)) / 3.0
    delays = tau_s * fs / np.pi ** np.arange(4)
    return np.maximum(np.round(delays).astype(int), 1)


def smear(signal: np.ndarray, decay_time: float, fs: float) -> np.ndarray:
    """The surface-scattering allpass cascade along axis 0 of ``signal``.

    Four Schroeder allpasses ``(g + z^-tau) / (1 + g z^-tau)``, ``g = 1/sqrt(2)``
    (Ewert et al. 2025, Eq. 12), with the delays of :func:`allpass_delays`.
    """
    out = np.asarray(signal, dtype=float)
    for tau in allpass_delays(decay_time, fs):
        b = np.zeros(tau + 1)
        a = np.zeros(tau + 1)
        b[0], b[-1] = 1.0 / np.sqrt(2.0), 1.0
        a[0], a[-1] = 1.0, 1.0 / np.sqrt(2.0)
        out = lfilter(b, a, out, axis=0)
    return out


def scattering_weights(doa: pf.Coordinates, directions: pf.Coordinates, opposite: bool = False) -> np.ndarray:
    """Weights of each image on the delay lines of a hemisphere, shape (M, n_lines).

    Ewert et al. (2025, Sec. 3.1.2) map a scattered part onto the "FDN channels on the same hemisphere" by a dot
    product (Sec. 2.1.2), and the last order's specular part onto "those FDN channels representing the spatially
    opposite hemisphere": ``w = clip(v . u, 0) / ||clip(v . u, 0)||_2``, ``v`` the image's direction of arrival
    (``-v`` with ``opposite``) and ``u`` a line's, in one frame. The unit l2 norm weighs every image alike; an
    image with no line in its hemisphere has no weights.
    """
    lines = directions.cartesian
    lines = lines / np.linalg.norm(lines, axis=1, keepdims=True)
    if doa.cshape == (0,):
        return np.zeros((0, lines.shape[0]), dtype=float)  # an empty Coordinates set has no cartesian
    arrival = doa.cartesian.reshape(-1, 3)
    arrival = arrival / np.linalg.norm(arrival, axis=1, keepdims=True)
    if opposite:
        arrival = -arrival
    weights = np.clip(arrival @ lines.T, 0.0, None)
    norm = np.linalg.norm(weights, axis=1, keepdims=True)
    return weights / np.where(norm > 0.0, norm, 1.0)


def _unit_normal(vertices: np.ndarray) -> np.ndarray:
    normal = 0.5 * np.cross(vertices, np.roll(vertices, -1, axis=0)).sum(axis=0)
    return normal / np.linalg.norm(normal)


def _first_hit(faces: list[tuple[np.ndarray, str]], origin: np.ndarray, directions: np.ndarray) -> np.ndarray:
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
