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
import os
import warnings
from typing import List

import numpy as np

from .base import Pressures
from .base import ScatteringSolver

__all__ = ["KirchhoffSolver", "triangle_phase_integral"]

_SERIES_SPREAD = 1.0  # rad: below this spread of a triangle's vertex phases the Taylor series is used
_SERIES_TERMS = 20  # enough for spreads below 1 rad to double precision
_SHADOW_PAIRS = 2_000_000  # (centroid, face) pairs per block of the shadow test
_PAIRS = 16384  # (receiver, face) pairs per block: their factors (1.3 MB) and a frequency's temporaries stay in the core's cache
_SECONDS_PER_PAIR = 2.4e-8  # s per (face, receiver, source, frequency) in one process, measured 2026-10-24 (see KirchhoffSolver)
_GAIN_PER_WORKER = 0.38  # the speed-up each worker beyond the first added, measured with 14 workers (5.9 x) on the same machine
_MIN_PARALLEL_SECONDS = 5.0  # runs estimated below this in one process stay in it: starting the worker processes takes about 1 s
_POLL = 1.0  # s between the progress reports while the workers run
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


def _pair_factors(triangles, normals, areas, u, receivers):
    """The frequency-independent factors of the (receiver, face) pairs of one source, flattened and sorted by phase spread.

    With w = û + r̂ the phases at the vertices are −k (w·x), so their order, the half sums and differences the closed form takes, the
    spread and the symmetric functions the series takes are k times those of w·x (:func:`_simplex_integral`, :func:`_simplex_series`):
    they are computed here once per source and receiver chunk and scaled by k for every frequency. The pairs are sorted by spread, so
    the pairs evaluated with the series at a frequency (spread < 1 rad) are a prefix. Pairs with a zero weight (zero-area faces, or
    n̂·w = 0) are left out: they add 0.

    Returns
    -------
    dict of numpy.ndarray
        Per pair: ``receiver`` (its index in ``receivers``), ``weight`` 2A n̂·w, ``spread`` b₂ − b₀, ``half_hi`` (b₁ + b₂)/2,
        ``half_lo`` (b₀ + b₁)/2, ``gap_hi`` b₂ − b₁, ``gap_lo`` b₁ − b₀, ``mean``, ``e2`` and ``e3`` of the centred b, for the sorted
        b₀ ≤ b₁ ≤ b₂ of w·x.
    """
    w = u + receivers
    weight = (w @ normals.T) * (2 * areas)  # (m, t)
    b = np.sort(np.einsum("mk,tvk->mtv", w, triangles), axis=-1)  # (m, t, 3)
    keep = weight != 0
    receiver = np.broadcast_to(np.arange(len(receivers))[:, None], weight.shape)[keep]
    weight, b = weight[keep], b[keep]
    b0, b1, b2 = b[:, 0], b[:, 1], b[:, 2]
    spread = b2 - b0
    order = np.argsort(spread, kind="stable")
    mean = b.mean(axis=1)
    x = b - mean[:, None]
    factors = {
        "receiver": receiver,
        "weight": weight,
        "spread": spread,
        "half_hi": 0.5 * (b1 + b2),
        "half_lo": 0.5 * (b0 + b1),
        "gap_hi": b2 - b1,
        "gap_lo": b1 - b0,
        "mean": mean,
        "e2": x[:, 0] * x[:, 1] + x[:, 0] * x[:, 2] + x[:, 1] * x[:, 2],
        "e3": x[:, 0] * x[:, 1] * x[:, 2],
    }
    return {name: value[order] for name, value in factors.items()}


def _series_terms(spread):
    """The terms of :func:`_simplex_series` that matter for phases spread at most ``spread`` (< 1 rad): |h_n| ≤ C(n+2, 2) spreadⁿ.

    A term is at most spreadⁿ / (2 n!); the series stops before the first below 1e-18 (all 20 at a spread of 1, as the plain series).
    """
    n = 3
    while n < _SERIES_TERMS and spread**n / (2.0 * math.factorial(n)) >= 1e-18:
        n += 1
    return n


def _accumulate(factors, k, n_receivers):
    """Σ over the pairs of each receiver of weight · ∫ exp(j λ·φ) dλ, for the phases φ = −k (w·x): complex, shape (n_receivers,).

    The same evaluation as :func:`triangle_phase_integral` (closed form where the spread is at least 1 rad, the Taylor series below), in
    real arithmetic on the factors of :func:`_pair_factors`.
    """
    n_series = int(np.searchsorted(factors["spread"], 1.0 / k, side="left"))
    receiver = factors["receiver"]
    re = np.zeros(n_receivers)
    im = np.zeros(n_receivers)
    if n_series < len(receiver):
        big = slice(n_series, None)
        # (d_lo - d_hi) / spread with d = j e^{jθ} sinc, θ = -k half, as in _simplex_integral
        s_hi = np.sinc(k * factors["gap_hi"][big] / (2 * np.pi))
        s_lo = np.sinc(k * factors["gap_lo"][big] / (2 * np.pi))
        a_hi = k * factors["half_hi"][big]
        a_lo = k * factors["half_lo"][big]
        scale = factors["weight"][big] / (k * factors["spread"][big])
        re += np.bincount(receiver[big], (np.sin(a_hi) * s_hi - np.sin(a_lo) * s_lo) * scale, minlength=n_receivers)
        im += np.bincount(receiver[big], (np.cos(a_hi) * s_hi - np.cos(a_lo) * s_lo) * scale, minlength=n_receivers)
    if n_series:
        small = slice(0, n_series)
        # the series about the mean phase -k mean, with the centred phases -k x: e2 -> k² e2, e3 -> -k³ e3
        e2 = k * k * factors["e2"][small]
        e3 = -(k**3) * factors["e3"][small]
        h = [np.ones_like(e2), np.zeros_like(e2), -e2]
        total_re = 0.5 + e2 / 24.0
        total_im = np.zeros_like(e2)
        for n in range(3, _series_terms(k * factors["spread"][n_series - 1])):
            h.append(-e2 * h[n - 2] + e3 * h[n - 3])
            term = h[n] / math.factorial(n + 2)
            sign = 1.0 if (n // 2) % 2 == 0 else -1.0  # j^n = (-1)^(n/2) for even n, j (-1)^((n-1)/2) for odd n
            if n % 2:
                total_im = total_im + sign * term
            else:
                total_re = total_re + sign * term
        phase = k * factors["mean"][small]
        cos, sin = np.cos(phase), np.sin(phase)  # e^{-j k mean} = cos - j sin
        weight = factors["weight"][small]
        re += np.bincount(receiver[small], (cos * total_re + sin * total_im) * weight, minlength=n_receivers)
        im += np.bincount(receiver[small], (cos * total_im - sin * total_re) * weight, minlength=n_receivers)
    return re + 1j * im


def _lit(body, u, shadowing):
    """The faces of a body lit from û (and not shadowed, with ``shadowing``): triangles, normals and areas."""
    triangles, normals, areas = body
    lit = np.flatnonzero(normals @ u > 0)
    if shadowing and len(lit):
        lit = lit[~_occluded(triangles, lit, u)]
    return triangles[lit], normals[lit], areas[lit]


def _source_pressures(bodies, u, receivers, ks, chunk, shadowing, tick=None):
    """The pressures of each body for one source direction û, complex (n_receivers, n_frequencies) per body.

    (j k / 4π) Σ_T n̂·(û + r̂) I_T(−k(û + r̂)) over the lit faces (the module docstring), with the geometry of each block of pairs
    computed once (:func:`_pair_factors`) and reused for every frequency. ``tick()`` is called after each block.
    """
    n_rec = len(receivers)
    out = []
    for body in bodies:
        triangles, normals, areas = _lit(body, u, shadowing)
        pressures = np.zeros((n_rec, len(ks)), dtype=complex)
        for r0 in range(0, n_rec, chunk):
            r_chunk = receivers[r0 : r0 + chunk]
            block = max(1, _PAIRS // max(len(r_chunk), 1))
            for t0 in range(0, len(triangles), block):
                t1 = t0 + block
                factors = _pair_factors(triangles[t0:t1], normals[t0:t1], areas[t0:t1], u, r_chunk)
                for f_index, k in enumerate(ks):
                    pressures[r0 : r0 + chunk, f_index] += _accumulate(factors, k, len(r_chunk))
                if tick is not None:
                    tick()
        out.append(pressures * (1j * ks / (4 * np.pi)))
    return out


_STATE = {}  # the bodies, receivers and wavenumbers of a pool's worker process


def _init_worker(state):
    _STATE.clear()
    _STATE.update(state)


def _solve_source(index, u):
    """One source in a worker process: ``(index, [sample, reference])``."""
    s = _STATE
    return index, _source_pressures(s["bodies"], u, s["receivers"], s["ks"], s["chunk"], s["shadowing"])


def _kill_pool(executor):
    """Stop a process pool now: cancel its queued sources and terminate its processes (the running sources are lost)."""
    processes = list(getattr(executor, "_processes", {}).values())  # private, but the only handle on the processes
    executor.shutdown(wait=False, cancel_futures=True)
    for process in processes:
        try:
            process.terminate()
        except Exception:  # noqa: BLE001 - already gone
            pass


class KirchhoffSolver(ScatteringSolver):
    """Physical-optics (Kirchhoff) free-field pressures of a rigid sample and its reference (Beckmann & Spizzichino 1963).

    See the module docstring for the formula. Its obliquity factor is the reciprocal Beckmann form n̂·(û + r̂) (spec decision 2026-10-24);
    each face's phase integral is closed-form (Gordon 1975). The receivers are in the far field: only their directions count.

    The sources run in parallel worker processes (``concurrent.futures.ProcessPoolExecutor``, started with "spawn"); a script that calls
    the solver at its top level then needs an ``if __name__ == "__main__":`` guard, or ``workers=1``. Per source, the geometry of each
    block of (receiver, face) pairs is computed once and reused for every frequency. :meth:`solve_batches` gives the pressures source
    batch by source batch, for callers that reduce them as they go (``measure_scattering``).

    Parameters
    ----------
    shadowing : bool, optional
        Also leave out a lit face when the ray from its centroid towards the source hits another face of the same body.
    chunk : int, optional
        The receivers evaluated at once. The faces are taken in blocks of about 16384 / ``chunk``, so that a block's factors stay in
        the core's cache.
    workers : int, optional
        The worker processes; ``os.cpu_count()`` by default. ``1``, a single source, or a run estimated under a few seconds run in this
        process.

    Notes
    -----
    **Runtime** (measured 2026-10-24 on a 14-core M-series Mac, 10 performance and 4 efficiency cores, Python 3.13). On the 0.8 m disc
    of ``sine_tile(0.0708, 0.0204)`` (``tests/lab/lab_shapes.py``; 7964 + 3504 faces), 1316 receivers and the 54 frequencies of the ISO
    range, one source takes 20-24 s in one process (80 s before the factors were reused across frequencies), and 2.9-3.4 s per source
    with 14 workers (about 6 times faster, not 14: the cores share memory bandwidth and 4 of them are efficiency cores). The ``"fine"``
    preset (864 sources) then takes about 42-49 min. :meth:`estimate_seconds` scales these: 2.4e-8 s per (face, receiver, source, frequency) in one process,
    faces counted over both bodies, and 0.38 of a process gained per further worker.

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
    batchable = True

    def __init__(self, shadowing=False, chunk=2048, workers=None):
        self.shadowing = bool(shadowing)
        self.chunk = int(chunk)
        if self.chunk < 1:
            raise ValueError("chunk must be at least 1, got {}".format(chunk))
        self.workers = int(workers) if workers is not None else (os.cpu_count() or 1)
        if self.workers < 1:
            raise ValueError("workers must be at least 1, got {}".format(workers))

    def estimate_seconds(self, sample, sources, receivers, frequencies, workers=None) -> float:
        """The expected wall time of :meth:`solve` in s (see Notes: Runtime).

        ``_SECONDS_PER_PAIR`` × faces of both bodies × receivers × sources × frequencies in one process, over ``1 + 0.38 (w − 1)`` for
        ``w`` workers (``self.workers`` by default, at most one per source).
        """
        faces = sum(len(_arrays(body)[0]) for body in (sample.sample, sample.reference))
        n_src = int(sources.n)
        serial = _SECONDS_PER_PAIR * faces * len(receivers.directions) * n_src * len(np.atleast_1d(frequencies))
        workers = max(1, min(self.workers if workers is None else int(workers), n_src))
        return serial / (1.0 + _GAIN_PER_WORKER * (workers - 1))

    def solve(self, sample, sources, receivers, frequencies, c):
        batches = list(self.solve_batches(sample, sources, receivers, frequencies, c, batch_size=max(1, sources.n)))
        return batches[0][1]

    def solve_batches(self, sample, sources, receivers, frequencies, c, batch_size):
        """The pressures in consecutive batches of at most ``batch_size`` sources, with one pool of worker processes for all of them.

        Progress is reported over all sources (``jobs.progress("kirchhoff", done, of)`` in source × frequency steps); a cancel stops the
        workers.

        Yields
        ------
        tuple
            ``(indices, pressures)``: the indices of the batch's sources in ``sources`` and their :class:`Pressures`.
        """
        from compas_acoustics.worker import jobs

        frequencies = np.atleast_1d(np.asarray(frequencies, dtype=float))
        ks = 2 * np.pi * frequencies / c
        bodies = [_arrays(sample.sample), _arrays(sample.reference)]
        u_all = np.asarray(sources.directions, dtype=float)
        r_all = np.asarray(receivers.directions, dtype=float)
        n_src, n_freq = len(u_all), len(frequencies)
        batch_size = max(1, int(batch_size))
        generation = {"method": self.name, "shadowing": self.shadowing, "faces": {"sample": len(bodies[0][0]), "reference": len(bodies[1][0])}}
        state = {"bodies": bodies, "receivers": r_all, "ks": ks, "chunk": self.chunk, "shadowing": self.shadowing}
        workers = min(self.workers, n_src)
        if workers > 1 and self.estimate_seconds(sample, sources, receivers, frequencies, workers=1) < _MIN_PARALLEL_SECONDS:
            workers = 1
        progress = {"done": 0, "of": n_src * n_freq}

        def report():
            jobs.progress("kirchhoff", progress["done"], progress["of"])

        report()
        executor = None
        if workers > 1:
            import multiprocessing
            from concurrent.futures import ProcessPoolExecutor

            executor = ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("spawn"), initializer=_init_worker, initargs=(state,))
        try:
            for start in range(0, n_src, batch_size):
                indices = np.arange(start, min(start + batch_size, n_src))
                out = [np.zeros((len(indices), len(r_all), n_freq), dtype=complex) for _ in bodies]
                pending = list(indices)
                if executor is not None:
                    pending = self._run_parallel(executor, u_all, indices, start, out, progress, report)
                    if pending:
                        executor = None
                for i in pending:
                    for p, target in zip(_source_pressures(bodies, u_all[i], r_all, ks, self.chunk, self.shadowing, tick=report), out):
                        target[i - start] = p
                    progress["done"] += n_freq
                    report()
                yield indices, Pressures(out[0], out[1], frequencies, dict(generation))
        finally:
            if executor is not None:
                _kill_pool(executor)

    def _run_parallel(self, executor, u_all, indices, start, out, progress, report):
        """Run a batch's sources on the pool; return the sources left to run here if the pool broke (warned)."""
        from concurrent.futures import FIRST_COMPLETED
        from concurrent.futures import wait
        from concurrent.futures.process import BrokenProcessPool

        n_freq = out[0].shape[2]
        futures = {executor.submit(_solve_source, int(i), u_all[i]) for i in indices}
        remaining = set(int(i) for i in indices)
        try:
            while futures:
                finished, futures = wait(futures, timeout=_POLL, return_when=FIRST_COMPLETED)
                for future in finished:
                    i, pressures = future.result()
                    for p, target in zip(pressures, out):
                        target[i - start] = p
                    remaining.discard(i)
                    progress["done"] += n_freq
                report()
        except BrokenProcessPool as error:
            _kill_pool(executor)
            warnings.warn("kirchhoff: the worker processes failed ({}); the remaining sources run in this process".format(error), stacklevel=4)
            return sorted(remaining)
        return []

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
