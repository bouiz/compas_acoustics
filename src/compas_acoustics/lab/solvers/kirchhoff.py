r"""The Kirchhoff (physical optics) free-field solver, with closed-form integrals over triangles.

Physical optics on a rigid body (Beckmann & Spizzichino 1963, ch. 3): on the faces lit by the incident plane wave the surface pressure is
twice the incident one and its normal derivative vanishes; the faces in shadow carry no field. For a source direction û (from the sample
towards the source, so the plane wave travels along −û), a far-field receiver direction r̂ and k = 2πf/c, the scattered pressure is

.. math::

    p(\hat r) = \frac{j k}{4\pi} \sum_{T:\ \hat n_T \cdot \hat u > 0} \hat n_T \cdot (\hat u + \hat r)\ I_T(q), \qquad
    q = -k(\hat u + \hat r), \qquad I_T(q) = \int_T e^{j q \cdot x}\, dS.

The obliquity factor :math:`\hat n \cdot (\hat u + \hat r)` is the reciprocal Beckmann form (spec decision 2026-10-24): with
:math:`\hat k_i = -\hat u` and :math:`\hat k_s = \hat r` it is :math:`-\hat n \cdot (\hat k_i - \hat k_s)`. Swapping û and r̂ leaves the
integrand unchanged, so the pressures are reciprocal wherever the lit set is the same for both directions; the 1/4π prefactor gives the
rigid form's amplitude at the specular direction. Every face of the closed body counts, the rim walls included, so the rim terms of a
sample and its reference match. The integral over each triangle is closed-form (Gordon 1975), see :func:`triangle_phase_integral`.
"""

import math
from typing import List

import numpy as np

from .base import Pressures
from .base import ScatteringSolver

__all__ = ["KirchhoffSolver", "triangle_phase_integral"]

_SERIES_SPREAD = 1.0  # rad: below this spread of a triangle's vertex phases the Taylor series is used
_SERIES_TERMS = 20  # enough for spreads below 1 rad to double precision
_SHADOW_PAIRS = 2_000_000  # (centroid, face) pairs per block of the shadow test
_PAIRS = 1_000_000  # (receiver, face) pairs per triangle-integral call: about 100 MB of temporaries
_STEEP_TILT = 45.0  # degrees
_STEEP_SHARE = 0.10
_STEEP = "kirchhoff: steep relief; multiple reflections are ignored"
_BELOW = "kirchhoff: below {:.0f} Hz the features ({:.4g} m) are smaller than the wavelength; s is unreliable there"


def triangle_phase_integral(triangles, q):
    r"""The integral :math:`\int_T e^{j q \cdot x}\, dS` over each triangle, in closed form (Gordon 1975).

    With the vertex phases α, β, γ = q·a, q·b, q·c and the area A, the Hermite–Genocchi formula for the second divided difference gives

    .. math::

        I = -2A \left( \frac{e^{j\alpha}}{(\alpha-\beta)(\alpha-\gamma)} + \frac{e^{j\beta}}{(\beta-\alpha)(\beta-\gamma)}
            + \frac{e^{j\gamma}}{(\gamma-\alpha)(\gamma-\beta)} \right),

    whose limit at α = β = γ is :math:`A e^{j\alpha}`. It is evaluated without cancellation: when the phases spread over more than 1 rad, as
    the difference of two first divided differences, each written with a sinc, divided by the largest phase difference; below that, as the
    Taylor series :math:`2A e^{j\bar\phi} \sum_n j^n h_n(\phi - \bar\phi) / (n+2)!` about the mean phase, with :math:`h_n` the complete
    homogeneous symmetric polynomials. Triangles with zero area give 0.

    Parameters
    ----------
    triangles : numpy.ndarray
        Vertices, shape (n_t, 3, 3).
    q : numpy.ndarray
        Wave vectors in rad/m, shape (..., 3).

    Returns
    -------
    numpy.ndarray
        Complex, shape (..., n_t).
    """
    triangles = np.asarray(triangles, dtype=float)
    q = np.asarray(q, dtype=float)
    a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    area = np.linalg.norm(np.cross(b - a, c - a), axis=1) / 2
    phases = np.einsum("...k,tvk->...tv", q, triangles)
    return 2 * area * _simplex_integral(phases)


def _simplex_integral(phases):
    """∫ exp(j λ·φ) dλ over the unit simplex {λ1, λ2 ≥ 0, λ1 + λ2 ≤ 1} (area 1/2), for vertex phases φ along the last axis."""
    phases = np.sort(phases, axis=-1)
    lo, mid, hi = phases[..., 0], phases[..., 1], phases[..., 2]
    spread = hi - lo
    small = spread < _SERIES_SPREAD
    # first divided differences of e^{jt}: (e^{ja} - e^{jb}) / (a - b) = j e^{j(a+b)/2} sin(d/2) / (d/2)
    d_lo = 1j * np.exp(0.5j * (lo + mid)) * np.sinc((mid - lo) / (2 * np.pi))
    d_hi = 1j * np.exp(0.5j * (mid + hi)) * np.sinc((hi - mid) / (2 * np.pi))
    out = (d_lo - d_hi) / np.where(small, 1.0, spread)  # -f[lo, mid, hi], the integral, where the spread is large
    if small.any():
        out[small] = _simplex_series(phases[small])
    return out


def _simplex_series(phases):
    """The simplex integral as a Taylor series about the mean phase, for phases (n, 3) that spread less than about 1 rad.

    ∫ (λ·x)^n dλ = n! h_n(x) / (n + 2)! (Dirichlet), and with Σx = 0 the h_n follow h_n = −e2 h_{n−2} + e3 h_{n−3}.
    """
    mean = phases.mean(axis=1)
    x = phases - mean[:, None]
    e2 = x[:, 0] * x[:, 1] + x[:, 0] * x[:, 2] + x[:, 1] * x[:, 2]
    e3 = x[:, 0] * x[:, 1] * x[:, 2]
    h = [np.ones_like(mean), np.zeros_like(mean), -e2]
    total = 0.5 + e2 / 24.0 + 0j  # n = 0 and n = 2: h_1 = 0, h_2 = -e2, j^2 = -1, 4! = 24
    for n in range(3, _SERIES_TERMS):
        h.append(-e2 * h[n - 2] + e3 * h[n - 3])
        total = total + (1j**n / math.factorial(n + 2)) * h[n]
    return np.exp(1j * mean) * total


def _arrays(mesh):
    """A triangulated mesh as triangles (n, 3, 3), unit normals (zero for zero-area faces) and areas."""
    vertices, faces = mesh.to_vertices_and_faces()
    V = np.asarray(vertices, dtype=float)
    T = np.asarray(faces, dtype=int).reshape(-1, 3)
    triangles = V[T]
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    twice = np.linalg.norm(cross, axis=1)
    normals = np.where(twice[:, None] > 0, cross / np.where(twice > 0, twice, 1.0)[:, None], 0.0)
    return triangles, normals, twice / 2


def _occluded(triangles, faces, direction):
    """Whether the ray from each face centroid of ``faces`` (indices) towards ``direction`` hits another triangle (Möller & Trumbore 1997)."""
    v0 = triangles[:, 0]
    e1 = triangles[:, 1] - v0
    e2 = triangles[:, 2] - v0
    pvec = np.cross(direction, e2)
    det = np.einsum("ij,ij->i", e1, pvec)
    scale = np.abs(triangles).max() if triangles.size else 1.0
    usable = np.abs(det) > 1e-12 * max(scale, 1e-300) ** 2
    inv = np.where(usable, 1.0 / np.where(usable, det, 1.0), 0.0)
    t_min = 1e-9 * max(scale, 1e-300)
    centroids = triangles[faces].mean(axis=1)
    hidden = np.zeros(len(faces), dtype=bool)
    block = max(1, _SHADOW_PAIRS // max(len(triangles), 1))
    for start in range(0, len(faces), block):
        origin = centroids[start : start + block]
        tvec = origin[:, None, :] - v0[None, :, :]
        u = np.einsum("mnk,nk->mn", tvec, pvec) * inv
        qvec = np.cross(tvec, e1[None, :, :])
        v = np.einsum("mnk,k->mn", qvec, direction) * inv
        t = np.einsum("mnk,nk->mn", qvec, e2) * inv
        hit = usable & (u >= 0) & (v >= 0) & (u + v <= 1) & (t > t_min)
        hit[np.arange(len(origin)), faces[start : start + block]] = False
        hidden[start : start + block] = hit.any(axis=1)
    return hidden


def _pressure(triangles, normals, u, receivers, k):
    """(j k / 4π) Σ_T n̂·(û + r̂) I_T(−k(û + r̂)) over the given (lit) faces, for each receiver direction r̂, in blocks of faces."""
    w = u + receivers
    total = np.zeros(len(receivers), dtype=complex)
    block = max(1, _PAIRS // max(len(receivers), 1))
    for start in range(0, len(triangles), block):
        integrals = triangle_phase_integral(triangles[start : start + block], -k * w)
        total += np.einsum("mt,mt->m", w @ normals[start : start + block].T, integrals)
    return (1j * k / (4 * np.pi)) * total


class KirchhoffSolver(ScatteringSolver):
    """Physical-optics (Kirchhoff) free-field pressures of a rigid sample and its reference (Beckmann & Spizzichino 1963).

    See the module docstring for the formula. Its obliquity factor is the reciprocal Beckmann form n̂·(û + r̂) (spec decision 2026-10-24);
    each face's phase integral is closed-form (Gordon 1975). The receivers are in the far field: only their directions count.

    Parameters
    ----------
    shadowing : bool, optional
        Also leave out a lit face when the ray from its centroid towards the source hits another face of the same body.
    chunk : int, optional
        The receivers evaluated at once. The faces are taken in blocks of about 10⁶ / ``chunk``, so memory stays near 100 MB.

    Notes
    -----
    **Known accuracy** (measured 2026-10-24). Against :class:`~compas_acoustics.lab.solvers.bem.BemSolver` on Heimes & Vorländer's
    simulated sine surface (Λ = 80 mm across x, h = 0.051/2.5 m, d = 0.8 m, c = 346.18 m/s), one plane wave each at colatitude 31.5° and
    61.74°, azimuth 0 (the incidence plane across the grooves), ``EXTENDED_BAND_RANGE``, the same receivers for both methods. The max over
    the two angles of |s_kirchhoff − s_bem| per octave (the octave value, the mean of its one-third-octave bands):

    ========  ==========  ==========================================================================
    octave    max |Δs|    note
    ========  ==========  ==========================================================================
    125 Hz    (0.466)     not a Kirchhoff error: the BEM runs did not converge below ~700 Hz
    250 Hz    (0.315)     as above (see below)
    500 Hz    (0.227)     as above; Kirchhoff gives s ≤ 0.015 here, the physical expectation
    1 kHz     0.022
    2 kHz     0.209       Λ/λ ≈ 0.37…0.58: Kirchhoff misses the onset of the first grating lobe
    4 kHz     0.247       one-third octaves up to 0.335 (3150 Hz, 31.5°); Kirchhoff lower than BEM
    8 kHz     0.115
    ========  ==========  ==========================================================================

    The bracketed octaves are BEM failures: below about 700 Hz the BEM runs did not converge, the sample in the 125, 250 and 500 Hz
    octaves and at 125 Hz also the reference (every step stopped at NumCalc's cap of 250 iterations, final residuals 2.6e-5…11 for the
    sample and 0.024…183 for the reference); see :class:`~compas_acoustics.lab.solvers.bem.BemSolver`. The BEM itself sits 0.03–0.10
    below Heimes & Vorländer's s in 11 of the 12 checked one-third octaves (Λ/λ ≥ 0.5, both angles), so Kirchhoff's error against an independent
    reference is likely larger than the gap stated here.

    So for Λ/λ ≳ 0.4 expect Kirchhoff's s to be off by up to about 0.25 per octave (0.33 per one-third octave), mostly too low, and use
    BEM where that matters. Reproduce with the Zenodo 15673498 results (see ``tests/lab/test_validation_heimes.py``)::

        HEIMES_RESULTS=<folder> HEIMES_BEM_CACHE=<cache> python -m pytest tests/lab/test_validation_heimes.py -v -s -k gap
    """

    name = "kirchhoff"

    def __init__(self, shadowing=False, chunk=2048):
        self.shadowing = bool(shadowing)
        self.chunk = int(chunk)
        if self.chunk < 1:
            raise ValueError("chunk must be at least 1, got {}".format(chunk))

    def solve(self, sample, sources, receivers, frequencies, c):
        from compas_acoustics.worker import jobs

        frequencies = np.atleast_1d(np.asarray(frequencies, dtype=float))
        bodies = [_arrays(sample.sample), _arrays(sample.reference)]
        u_all = np.asarray(sources.directions, dtype=float)
        r_all = np.asarray(receivers.directions, dtype=float)
        n_src, n_rec, n_freq = len(u_all), len(r_all), len(frequencies)
        out = [np.zeros((n_src, n_rec, n_freq), dtype=complex) for _ in bodies]
        done, of = 0, n_src * n_freq
        jobs.progress("kirchhoff", done, of)
        for i, u in enumerate(u_all):
            lit_sets = []
            for triangles, normals, _ in bodies:
                lit = np.flatnonzero(normals @ u > 0)
                if self.shadowing and len(lit):
                    lit = lit[~_occluded(triangles, lit, u)]
                lit_sets.append((triangles[lit], normals[lit]))
            for f_index, f in enumerate(frequencies):
                k = 2 * np.pi * f / c
                for (triangles, normals), pressures in zip(lit_sets, out):
                    for start in range(0, n_rec, self.chunk):
                        pressures[i, start : start + self.chunk, f_index] = _pressure(triangles, normals, u, r_all[start : start + self.chunk], k)
                done += 1
                jobs.progress("kirchhoff", done, of)
        generation = {"method": self.name, "shadowing": self.shadowing, "faces": {"sample": len(bodies[0][0]), "reference": len(bodies[1][0])}}
        return Pressures(out[0], out[1], frequencies, generation)

    def warnings(self, sample, nominal_bands, exact_bands, c) -> List[str]:
        """Where physical optics is unreliable: wavelengths longer than the features, and steep relief (multiple reflections).

        The feature size is ``sample.feature_size``, else the smaller tile period; without either the first check is skipped.
        The relief is steep when faces tilted more than 45° make up more than 10 % of its top area (the faces above the baseplate).
        """
        messages = []
        feature = sample.feature_size or (min(sample.period) if sample.period else None)
        if feature:
            exact = np.asarray(exact_bands, dtype=float)
            if (c / exact > feature).any():
                messages.append(_BELOW.format(c / feature, feature))
        triangles, normals, areas = _arrays(sample.sample)
        top = triangles[:, :, 2].min(axis=1) > -sample.baseplate / 2
        total = areas[top].sum()
        steep = top & (normals[:, 2] < np.cos(np.radians(_STEEP_TILT)))
        if total > 0 and areas[steep].sum() > _STEEP_SHARE * total:
            messages.append(_STEEP)
        return messages
