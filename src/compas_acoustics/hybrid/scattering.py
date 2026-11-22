"""Surface scattering of image sources.

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
from ..room.geometry import DEFAULT_SCATTERING


def complementary_scattering(geometry):
    """``geometry`` with every material's scattering ``s`` replaced by ``1 - s``.

    In a room built for ray tracing an image's damping takes ``sqrt(1 - s)`` at every wall of non-zero scattering it
    reflects off: the specular part, ``prod sqrt(1 - s_i)``. Built from this geometry it is the scattered part,
    ``prod sqrt(s_i)``. The two are power-complementary for a single reflection, as the decomposition filters of
    Ewert et al. (2025, Sec. 2.1.3) are. A material without coefficients scatters ``DEFAULT_SCATTERING``.
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

    Ewert et al. (2025, Sec. 2.1.2): ``T_s = 6 (10**(1/5) - 1) 2R / c``, about ``7 R / c``, for a reflector at
    distance ``R``; for a general room "the mean free path length".
    """
    return 6.0 * (10.0**0.2 - 1.0) * 2.0 * distance / speed_of_sound


def allpass_delays(decay_time: float, fs: float) -> np.ndarray:
    """Delays in samples of the four-allpass cascade, Ewert et al. (2025, Eq. 13).

    ``tau_s = T_s log10(1/g) / 3`` and ``tau_i = tau_s fs / eta**(i-1)``, ``g = 1/sqrt(2)``, ``eta = pi``, rounded,
    never below one sample.
    """
    tau_s = decay_time * np.log10(np.sqrt(2.0)) / 3.0
    delays = tau_s * fs / np.pi ** np.arange(4)
    return np.maximum(np.round(delays).astype(int), 1)


def smear(signal: np.ndarray, decay_time: float, fs: float) -> np.ndarray:
    """The surface-scattering allpass cascade along axis 0 of ``signal``.

    Four allpasses ``(g + z^-tau) / (1 + g z^-tau)``, ``g = 1/sqrt(2)`` (Ewert et al. 2025, Eq. 12), with the delays
    of :func:`allpass_delays`.
    """
    out = np.asarray(signal, dtype=float)
    for tau in allpass_delays(decay_time, fs):
        out = _allpass(out, int(tau), 1.0 / np.sqrt(2.0))
    return out


def _allpass(x: np.ndarray, tau: int, g: float) -> np.ndarray:
    """``(g + z^-tau) / (1 + g z^-tau)`` along axis 0, ``tau`` rows at a time.

    ``y[n] = g x[n] + x[n - tau] - g y[n - tau]``: every row of a block depends only on the block before it, so a
    block is one vector operation. It agrees with ``scipy.signal.lfilter`` to about 2e-16 of the peak.
    """
    y = np.empty_like(x)
    y[:tau] = g * x[:tau]
    for start in range(tau, x.shape[0], tau):
        stop = min(start + tau, x.shape[0])
        y[start:stop] = g * x[start:stop] + (x[start - tau : stop - tau] - g * y[start - tau : stop - tau])
    return y


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
