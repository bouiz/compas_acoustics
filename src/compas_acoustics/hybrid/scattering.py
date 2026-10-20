"""Surface scattering of image sources, and ray casts against the faces of a room.

After Ewert, Goessling, Buttler, van de Par and Hu, "Computationally-efficient
rendering of diffuse reflections for geometrical acoustics based room
simulation", Acta Acustica 9, 9 (2025), as RAZR 0.96.3, the authors'
implementation, does it (read as a black box: ``tests/hybrid/data/razr_scattering``).
:func:`~compas_acoustics.hybrid.early.image_sources` splits each image into a
specular and a scattered part; here are the room that split needs
(:func:`complementary_scattering`), the temporal spread of the scattered part
(Eqs. 12-13) and its weights on the delay lines.

``allpass_delays``, ``smear``, ``_unit_normal`` and ``_first_hit`` are ported
from hybrid-rir (gpapadimitri/hybrid-rir @ a272da7, src/scattering.py).
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pyfar as pf
from scipy.signal import lfilter

from ..room_geometry import DEFAULT_SCATTERING

_APC_GAIN = 1.0 / np.sqrt(2.0)
_APC_DELAY_RATIO = np.pi
_APC_STAGES = 4


def complementary_scattering(geometry):
    """``geometry`` with every material's scattering ``s`` replaced by ``1 - s``.

    Built with ``ray_tracing=True``, pyroomacoustics multiplies an image's
    damping by ``sqrt(1 - s)`` for every wall it reflects off whose scattering
    is above zero (``libroom_src/room.cpp``, ``image_sources_dfs``). The room
    itself so gives the specular part, ``prod sqrt(1 - s_i)``, and this one the
    scattered part, ``prod sqrt(s_i)``, both exact at ``s = 0`` and ``s = 1``:
    a wall that scatters nothing is skipped, which is the factor 1 either
    wants. A material without coefficients scatters ``DEFAULT_SCATTERING``.
    """
    return replace(geometry, materials={
        name: replace(
            material,
            scattering=1.0 - (np.full(len(material.absorption), DEFAULT_SCATTERING)
                              if material.scattering is None else np.asarray(material.scattering, dtype=float)),
        )
        for name, material in geometry.materials.items()
    })


def scattering_decay_time(distance: float, speed_of_sound: float) -> float:
    """Local decay time ``T_s`` of an ideal diffuse reflection, in seconds.

    Ewert et al. (2025), Sec. 2.1.2: ``T_s = 6 (10**(1/5) - 1) 2R / c``, about
    ``7 R / c``, for a reflector at distance ``R``. RAZR takes the receiver's
    mean distance to the six walls of its shoebox (``get_ism_setup.m``,
    ``0.0205 * mDist``, which is this at c = 342.4 m/s); for any other room
    the paper names "the mean free path length", which the hybrid uses.
    """
    return 6.0 * (10.0 ** 0.2 - 1.0) * 2.0 * distance / speed_of_sound


def allpass_delays(decay_time: float, fs: float) -> np.ndarray:
    """Delays in samples of the four-allpass cascade, Ewert et al. (2025) Eq. 13.

    ``tau_s = T_s log10(1/g) / 3`` and ``tau_i = tau_s fs / eta**(i-1)``,
    rounded, never below one sample. RAZR's time-spread cascade has these
    delays for the same ``T_s`` (oracle, ``allpass_delays``).
    """
    tau_s = decay_time * np.log10(1.0 / _APC_GAIN) / 3.0
    delays = tau_s * fs / _APC_DELAY_RATIO ** np.arange(_APC_STAGES)
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
        b[0], b[-1] = _APC_GAIN, 1.0
        a[0], a[-1] = 1.0, _APC_GAIN
        out = lfilter(b, a, out, axis=0)
    return out


def scattering_weights(doa: pf.Coordinates, directions: pf.Coordinates) -> np.ndarray:
    """Weights of each image's scattered part on each delay line, shape (M, n_lines).

    The paper maps a scattered part onto the "FDN channels on the same
    hemisphere" (Sec. 3.1.2) by a dot product (Sec. 2.1.2). RAZR does so with
    ``op.fdn_smart_ch_mapping = 1``, and its FDN input then holds exactly
    (oracle, ``coupling``): ``w = clip(v . u, 0) / ||clip(v . u, 0)||_2``,
    with ``v`` the image's direction of arrival, not the wall's normal, and
    ``u`` a line's direction; the unit l2 norm gives each scattered part the
    weight of one last-order specular image. ``doa`` and ``directions`` are in
    the same frame. A part with no line in its hemisphere has no weights.
    """
    lines = directions.cartesian
    lines = lines / np.linalg.norm(lines, axis=1, keepdims=True)
    arrival = doa.cartesian.reshape(-1, 3)
    arrival = arrival / np.linalg.norm(arrival, axis=1, keepdims=True)
    weights = np.clip(arrival @ lines.T, 0.0, None)
    norm = np.linalg.norm(weights, axis=1, keepdims=True)
    return weights / np.where(norm > 0.0, norm, 1.0)


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
