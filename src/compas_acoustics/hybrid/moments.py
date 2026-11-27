"""The ambisonic late field as Gaussian noise shaped by spherical harmonic energy moments.

An order-N ambisonic tail has ``K = (N + 1) ** 2`` channels. Per octave band and time bin it is described by its
channel covariance

    C_ij = sum_r e_r Y_i(d_r) Y_j(d_r),

the energy ``e_r`` of every ray detected in the bin, from direction ``d_r``, ``Y`` the real spherical harmonics (N3D
with W = 1, ACN). A product ``Y_i Y_j`` of two harmonics of order at most N is a sum of harmonics of order at most
2N, ``Y_i Y_j = sum_k G_ijk Y_k`` with the real Gaunt coefficients ``G`` (:func:`gaunt`), so

    C_ij = sum_k G_ijk M_k,   M_k = sum_r e_r Y_k(d_r),

and the ``(2N + 1) ** 2`` moments ``M`` of the energy's directional distribution carry everything the covariance
holds. ``C`` is positive semi-definite by construction, a sum of ``e y y^T``. The tail is white Gaussian noise
split into octave bands, each band mixed per bin by a square root of its covariance (Cholesky) and scaled to its
energy per sample (:func:`synthesize`).

This reproduces every channel's energy decay and the correlation between channels: all of the late field's
spatial information that an order-N signal can carry (its diffuseness, its anisotropy, the direction of the late
energy). A dense late tail is close to Gaussian, so these second-order statistics describe it. The early
reflections stay exact image sources; only the stochastic tail is synthesized here.

The tracer (pyroomacoustics' libroom) sorts every detected ray to the nearest direction of a receiver's grid in its
own code, so the moments are accumulated from the bins of a fine grid (:data:`MOMENT_GRID_POINTS` directions) rather
than per hit: the grid moves the covariance by under 0.5 % up to third order, against a spread of 20 to 45 % between
ray seeds in one 4 ms bin at the package's ray counts.

Where the energy and the shape come from (:func:`~compas_acoustics.hybrid.synthesis.prepare` with ``late_field="moments"``):

- Energy: the rays' own (``M_0``), when the late field follows a reference traced by the same run at the method's
  image source order: a ray is logged only after that order, so its bins hold exactly what the image sources do not.
  Otherwise the energy a feedback delay network driven by the last order's images would radiate
  (:func:`network_energy`), and with a reference the late envelope then brings it per band to the reference
  (:func:`~compas_acoustics.hybrid.envelope.shape_late`).
- Shape: the rays' moments, in the ``"directional"`` late envelope; otherwise the boundary's reflection in each
  direction, ``1 - a(u)``, scanned as for the network's lines (:func:`absorption_shape`), or isotropic without
  directional absorption.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache

import numpy as np
import pyfar as pf
import spharpy
from scipy.signal import fftconvolve

from .bands import N_BANDS
from .envelope import SMOOTH_MS
from .envelope import _octave_bands

__all__ = [
    "MOMENT_GRID_POINTS",
    "ENERGY_SMOOTH_MS",
    "LateField",
    "sh_basis",
    "gaunt",
    "moment_grid",
    "moments_from_hits",
    "histogram_moments",
    "covariance",
    "absorption_shape",
    "network_energy",
    "ray_late_field",
    "pair_seed",
    "synthesize",
]

#: Directions of the grid the tracer sorts the rays on for the moments, an equal-area grid. At 480 the grid
#: moves the covariance by 0.2 % (first order) to 0.4 % (third order) from the exact per-hit moments.
MOMENT_GRID_POINTS = 480

#: The Hann window over which the rays' energy per bin is averaged. Short, as the energy is an expected value
#: per bin and not a noise realisation; it keeps the onset of the late field where the rays put it.
ENERGY_SMOOTH_MS = 20.0

#: A band's late field is held, energy decaying at the target's reverberation time and shape frozen, once its
#: rays have decayed this far (the rays thin out towards the end of the trace; as the directional shares do).
HOLD_DB = 60.0

#: The rays before the hold that set the level the continuation starts from, and over which it fades in: the last
#: bins hold few rays, and one of them would carry its noise into the whole continuation.
HOLD_FIT_MS = 150.0

# 13.8155...: an energy that decays 60 dB in T seconds decays by exp(-RT_RATE / T) per second
RT_RATE = 6.0 * np.log(10.0)


def pair_seed(seed: int, *coords) -> int:
    """A noise seed from ``seed`` and a pair's coordinates (quantized to 0.1 mm), the same whichever other pairs are
    rendered with it."""
    mixed = int(seed) & 0xFFFFFFFFFFFFFFFF
    for coord in coords:
        for value in np.asarray(coord, dtype=np.float64).reshape(-1):
            mixed = (mixed * 0x9E3779B185EBCA87 + (round(float(value) * 1e4) & 0xFFFFFFFFFFFFFFFF)) & 0xFFFFFFFFFFFFFFFF
    return mixed % (2**63 - 1) or 1


def sh_basis(order: int, unit: np.ndarray) -> np.ndarray:
    """Real spherical harmonics up to ``order`` at ``unit`` (M, 3), (M, (order + 1) ** 2): N3D with W = 1 (``"NM"``), ACN.

    Orthogonal with ``integral Y_i Y_j dOmega = 4 pi delta_ij``, so a direction's W is 1.
    """
    unit = np.asarray(unit, dtype=float).reshape(-1, 3)
    coordinates = pf.Coordinates(unit[:, 0], unit[:, 1], unit[:, 2])
    return spharpy.spherical.spherical_harmonic_basis_real(order, coordinates, normalization="NM", channel_convention="ACN")


@cache
def gaunt(sh_order: int) -> np.ndarray:
    """Real Gaunt coefficients ``G_ijk``, (K, K, (2N + 1) ** 2) with ``K = (N + 1) ** 2``: ``Y_i Y_j = sum_k G_ijk Y_k``.

    ``G_ijk = (1 / 4 pi) integral Y_i Y_j Y_k dOmega``, exact on a Gaussian grid integrating order 4N.
    """
    if sh_order < 0:
        raise ValueError("sh_order must be non-negative")
    quadrature = spharpy.samplings.gaussian(n_max=2 * sh_order + 1)
    weights = quadrature.weights / np.sum(quadrature.weights)  # sums to one: the (1 / 4 pi) integral
    unit = quadrature.cartesian / np.linalg.norm(quadrature.cartesian, axis=1, keepdims=True)
    Y = sh_basis(2 * sh_order, unit)
    Yn = Y[:, : (sh_order + 1) ** 2]
    G = np.einsum("q,qi,qj,qk->ijk", weights, Yn, Yn, Y)
    G[np.abs(G) < 1e-12] = 0.0
    return G


@cache
def moment_grid(n_points: int = MOMENT_GRID_POINTS) -> np.ndarray:
    """The directions the tracer sorts the rays on for the moments, (n, 3) unit vectors in the receiver's frame."""
    grid = spharpy.samplings.equal_area(n_max=0, n_points=n_points).cartesian
    return grid / np.linalg.norm(grid, axis=1, keepdims=True)


def moments_from_hits(directions: np.ndarray, energies: np.ndarray, order: int) -> np.ndarray:
    """The exact moments ``M_k = sum_r e_r Y_k(d_r)`` of ray hits, ``(..., (order + 1) ** 2)``.

    ``directions`` (R, 3) are the hits' directions of arrival, ``energies`` (R, ...) their energy per band or bin.
    """
    Y = sh_basis(order, np.asarray(directions, dtype=float) / np.linalg.norm(directions, axis=1, keepdims=True))
    return np.tensordot(np.asarray(energies, dtype=float), Y, axes=(0, 0))


def histogram_moments(histogram, order: int, rotation: np.ndarray | None = None) -> np.ndarray:
    """The moments of a :class:`~compas_acoustics.room.raytracing.DirectionalHistogram`, (n_bands, n_bins, (order + 1) ** 2).

    Each bin's energy counts at its grid direction, turned into the receiver's frame by ``rotation`` (its columns
    the receiver's axes in room coordinates).
    """
    unit = np.asarray(histogram.directions, dtype=float)
    unit = unit / np.linalg.norm(unit, axis=1, keepdims=True)
    if rotation is not None:
        unit = unit @ np.asarray(rotation, dtype=float)
    return np.einsum("gbt,gk->btk", np.asarray(histogram.energy, dtype=float), sh_basis(order, unit))


def covariance(moments: np.ndarray, sh_order: int) -> np.ndarray:
    """The channel covariance ``C_ij = sum_k G_ijk M_k`` of moments of order at least ``2 sh_order``, (..., K, K)."""
    G = gaunt(sh_order)
    return np.einsum("ijk,...k->...ij", G, np.asarray(moments, dtype=float)[..., : G.shape[2]])


def absorption_shape(absorption: np.ndarray | None, directions: np.ndarray, order: int) -> tuple[np.ndarray, np.ndarray]:
    """``(shape, level)``: the normalised moments (N_BANDS + 1, (order + 1) ** 2) of the distribution ``1 - a(u)``.

    ``absorption`` (n_dir, 8) is the boundary's per direction (:func:`~compas_acoustics.hybrid.directional.scanned_absorption`)
    on the equal-area grid ``directions``, whose equal weights integrate the moments; None is isotropic. ``level``
    (N_BANDS + 1,) is the mean ``1 - a``, the one reflection's absorption the network's directional filters take
    from the late field. The 16 kHz row repeats the 8 kHz one.
    """
    Y = sh_basis(order, directions)
    if absorption is None:
        shape = np.zeros((N_BANDS + 1, Y.shape[1]))
        shape[:, 0] = 1.0
        return shape, np.ones(N_BANDS + 1)
    reflection = 1.0 - np.asarray(absorption, dtype=float)  # (n_dir, 8)
    reflection = np.hstack([reflection, reflection[:, -1:]])
    moments = reflection.T @ Y / Y.shape[0]  # (9, n_moments)
    level = moments[:, 0].copy()
    return moments / level[:, np.newaxis], level


def _hann(n: int) -> np.ndarray:
    window = np.hanning(max(int(n), 1) + 2)[1:-1]
    return window / window.sum()


def _rt_top(rt: np.ndarray, fs: float) -> np.ndarray:
    """``rt`` (8, S) with a 16 kHz row: the 8 kHz decay shortened as the network's attenuation is towards Nyquist,
    halved at Nyquist and interpolated on a log-frequency axis."""
    rt = np.atleast_2d(np.asarray(rt, dtype=float))
    octaves = np.log2(min(16000.0, fs / 2.0) / 8000.0) / max(np.log2(fs / 2.0 / 8000.0), 1e-9)
    return np.vstack([rt, rt[-1:] * 0.5**octaves])


def _slope_weights(level: np.ndarray | None, n_bands: int) -> np.ndarray:
    """Each slope's share of the energy per band, (n_bands + 1, S); a single slope has no level and takes it all."""
    if level is None:
        return np.ones((n_bands + 1, 1))
    level = np.atleast_2d(np.asarray(level, dtype=float))
    weights = level**2 / np.sum(level**2, axis=1, keepdims=True)
    return np.vstack([weights, weights[-1:]])


def network_energy(
    delays: np.ndarray,
    drive: tuple[np.ndarray, np.ndarray],
    target,
    n_samples: int,
    fs: float,
    *,
    tap: tuple[np.ndarray, np.ndarray] | None = None,
) -> np.ndarray:
    """The W energy per sample, (N_BANDS + 1, n), that the late network of Wendt et al. (2014) radiates in expectation.

    ``drive`` is ``(delay_s, energy)`` of the images driving it (M,) and (M, 8); ``tap`` the same for scattered parts
    that enter skipping the delays (Ewert et al. 2025). ``delays`` are the lines' lengths in samples.

    A unit of energy entering every line through a lossless mixing matrix leaves the network at the rate
    ``1 / L`` per sample, ``L`` the lines' mean length, as the energy spreads over all their samples, and decays at
    ``target.rt``: the impulse response ``F(t) exp(-13.8 t / T) / L``, ``F`` the share of lines whose delay has
    passed by ``t``, the onset of the first pass. A multi-slope target sums its slopes weighted by their levels, as the
    network's output filters do. A tapped part is heard at once, one pass's attenuation later.
    """
    delays = np.sort(np.asarray(delays, dtype=float))
    mean_length = float(np.mean(delays))
    t = np.arange(n_samples, dtype=float)
    onset = np.searchsorted(delays, t, side="right") / delays.size
    rt = _rt_top(target.rt, fs)  # (9, S)
    weights = _slope_weights(target.level, np.asarray(target.rt).shape[0])
    decay = np.einsum("bs,bst->bt", weights, np.exp(-RT_RATE / (rt[:, :, np.newaxis] * fs) * t))  # (9, n)

    def impulses(times_s, energy):
        x = np.zeros((N_BANDS + 1, n_samples))
        index = np.round(np.asarray(times_s, dtype=float) * fs).astype(int)
        keep = (index >= 0) & (index < n_samples)
        energy = np.asarray(energy, dtype=float).reshape(-1, N_BANDS)
        energy = np.hstack([energy, energy[:, -1:]])[keep]
        for b in range(N_BANDS + 1):
            np.add.at(x[b], index[keep], energy[:, b])
        return x

    out = np.zeros((N_BANDS + 1, n_samples))
    if np.asarray(drive[0]).size:
        out += fftconvolve(impulses(*drive), decay * onset / mean_length, axes=1)[:, :n_samples]
    if tap is not None and np.asarray(tap[0]).size:
        first_pass = np.einsum("bs,bs->b", weights, np.exp(-RT_RATE / (rt * fs) * mean_length))
        out += fftconvolve(impulses(*tap), decay * (first_pass / mean_length)[:, np.newaxis], axes=1)[:, :n_samples]
    return np.maximum(out, 0.0)


def ray_late_field(histogram, sh_order: int, rotation: np.ndarray | None, target, n_samples: int, fs: float, *, shape: bool = True) -> tuple[np.ndarray, np.ndarray | None, int]:
    """``(energy, shape, bin_samples)`` of the rays in a :class:`~compas_acoustics.room.raytracing.DirectionalHistogram`.

    ``energy`` (N_BANDS + 1, n) is the W energy per sample, the bins' over ``ENERGY_SMOOTH_MS``; ``shape`` (N_BANDS + 1,
    n_bins, (2N + 1) ** 2) the moments over ``SMOOTH_MS`` divided by their W, per bin, or None without ``shape``.
    Once a band's rays have decayed ``HOLD_DB`` (or the histogram ends) its energy decays at the target's slowest
    reverberation time from there, from the level of its last ``HOLD_FIT_MS`` of rays and faded in over them, and its
    shape is held. Bins before the rays arrive take the first shape. The
    16 kHz band is the histogram's own row where it has one (the tracer's at 48 kHz); otherwise the 8 kHz row,
    decaying from it at the network's 16 kHz decay.
    """
    order = 2 * sh_order if shape else 0
    moments = histogram_moments(histogram, order, rotation)  # (n_bands, n_bins, n_moments)
    bin_samples = max(int(round(histogram.bin_size * fs)), 1)
    n_bands, n_bins = moments.shape[:2]
    if n_bands < N_BANDS:
        raise ValueError(f"the histogram holds {n_bands} bands; the late field needs the {N_BANDS} from 62.5 Hz to 8 kHz")
    rt = _rt_top(target.rt, fs).max(axis=1)  # (9,)
    if n_bands > N_BANDS:
        moments = moments[: N_BANDS + 1]
    else:
        extra = np.exp(-RT_RATE * (1.0 / rt[-1] - 1.0 / rt[-2]) * (np.arange(n_bins) + 0.5) * histogram.bin_size)
        moments = np.concatenate([moments, moments[-1:] * extra[np.newaxis, :, np.newaxis]], axis=0)

    fit_bins = max(int(round(HOLD_FIT_MS * 1e-3 / histogram.bin_size)), 1)
    smooth_e = _hann(ENERGY_SMOOTH_MS * 1e-3 / histogram.bin_size)
    energy_bins = np.maximum(fftconvolve(moments[:, :, 0], smooth_e[np.newaxis], mode="same", axes=1), 0.0)  # (9, n_bins)
    raw = moments[:, :, 0]

    n_bins_out = int(np.ceil(n_samples / bin_samples)) + 1
    times = (np.arange(n_bins_out) + 0.5) * bin_samples
    energy = np.zeros((N_BANDS + 1, n_samples))
    holds = np.zeros(N_BANDS + 1, dtype=int)
    for b in range(N_BANDS + 1):
        edc = np.cumsum(raw[b, ::-1])[::-1]
        below = np.flatnonzero(edc <= 10 ** (-HOLD_DB / 10.0) * edc[0]) if edc[0] > 0.0 else np.array([0])
        hold = int(below[0]) if below.size else n_bins - 1
        hold = min(max(hold, 1), n_bins - 1)
        holds[b] = hold
        per_bin = np.zeros(n_bins_out)
        m = min(hold, n_bins_out)
        per_bin[:m] = energy_bins[b, :m]
        # the continuation's level at the hold: the trend of the last HOLD_FIT_MS of rays, not one bin's. Their own
        # decay per bin from the energy of the window's two halves (sums: empty bins count), the level from all of
        # it; faded in over the window along that trend, then the target's decay from the hold
        half = (hold - max(hold - fit_bins, 0)) // 2
        first = hold - 2 * half
        target_decay = np.exp(-RT_RATE / rt[b] * (np.arange(n_bins_out) - (hold - 1)) * histogram.bin_size)
        start = float(energy_bins[b, hold - 1])
        older, newer = (float(np.sum(raw[b, first : first + half])), float(np.sum(raw[b, first + half : hold]))) if half else (0.0, 0.0)
        if older > 0.0 and newer > 0.0:
            per_bin_decay = min((newer / older) ** (1.0 / half), 1.0)
            trend = per_bin_decay ** (np.arange(n_bins_out) - (hold - 1.0))
            start = float(np.sum(raw[b, first:hold]) / np.sum(trend[first:hold]))
            fade = np.arange(first, m)
            weight = (fade - first + 1.0) / (hold - first)
            per_bin[fade] = (1.0 - weight) * energy_bins[b, fade] + weight * start * trend[fade]
        if m < n_bins_out:
            per_bin[m:] = start * target_decay[m:]
        energy[b] = np.interp(np.arange(n_samples), times, per_bin / bin_samples, left=per_bin[0] / bin_samples)

    if not shape:
        return energy, None, bin_samples

    window = _hann(SMOOTH_MS * 1e-3 / histogram.bin_size)
    smoothed = fftconvolve(moments, window[np.newaxis, :, np.newaxis], mode="same", axes=1)  # (9, n_bins, n_moments)
    out = np.zeros((N_BANDS + 1, n_bins_out, smoothed.shape[2]))
    for b in range(N_BANDS + 1):
        w = smoothed[b, :, 0]
        valid = w > 1e-12 * max(float(w.max()), 1e-300)
        normalised = np.zeros_like(smoothed[b])
        normalised[:, 0] = 1.0
        normalised[valid] = smoothed[b, valid] / w[valid, np.newaxis]
        idx = np.flatnonzero(valid)
        if idx.size:
            normalised[: idx[0]] = normalised[idx[0]]  # before the rays: their first shape
            last = min(holds[b], idx[-1])
            normalised[last + 1 :] = normalised[last]  # after the hold, or the rays' end: held
        m = min(n_bins, n_bins_out)
        out[b, :m] = normalised[:m]
        out[b, m:] = normalised[m - 1]
    return energy, out, bin_samples


@dataclass(frozen=True)
class LateField:
    """The ambisonic late field of one pair before synthesis.

    ``energy`` (N_BANDS + 1, n) is the W energy per sample in each octave band, 63 Hz to 16 kHz; ``shape`` the
    moments per band normalised to W = 1, either (N_BANDS + 1, n_moments) for all time or (N_BANDS + 1, n_bins,
    n_moments) per bin of ``bin_samples``; their order ``moment_order`` must be at least twice the output's.
    ``follows_reference`` marks an energy of the network's (:func:`network_energy`) that the late envelope must
    still bring to the reference; ``seed`` seeds the noise.
    """

    energy: np.ndarray
    shape: np.ndarray
    bin_samples: int
    moment_order: int
    follows_reference: bool
    seed: int
    source: str  # "rays" or "network": where the energy came from

    @property
    def n_samples(self) -> int:
        return int(self.energy.shape[1])


def _square_roots(C: np.ndarray) -> np.ndarray:
    """Lower Cholesky factors of covariances (..., K, K), PSD by construction; a rank-deficient one gets a
    relative jitter of 1e-10 of its trace, raised until it factors."""
    C = 0.5 * (C + np.swapaxes(C, -1, -2))
    K = C.shape[-1]
    scale = np.trace(C, axis1=-2, axis2=-1)[..., np.newaxis, np.newaxis] / K
    scale = np.where(scale > 0.0, scale, 1.0)
    jitter = 1e-10
    while True:
        try:
            return np.linalg.cholesky(C + jitter * scale * np.eye(K))
        except np.linalg.LinAlgError:
            jitter *= 100.0
            if jitter > 1e-2:
                values, vectors = np.linalg.eigh(C)
                return vectors * np.sqrt(np.maximum(values, 0.0))[..., np.newaxis, :]


def synthesize(field: LateField, sh_order: int, fs: float) -> np.ndarray:
    """The late field as an ambisonic signal (n, (sh_order + 1) ** 2), ACN, N3D with W = 1 for a unit plane wave.

    One white Gaussian noise per channel, seeded by ``field.seed``, is split into the bands of the reconstructing
    octave bank, which sum back to it. Band ``b`` is mixed by the Cholesky factor of its covariance
    (:func:`covariance` of ``field.shape``), linearly interpolated between bins, and scaled by the square root of
    ``field.energy[b]``. ``energy`` is on the scale of the image sources' band gains and the rays' bins: the energy
    per sample of a white signal at that band's level, of which the band holds its share of the spectrum. A field
    with one energy and shape in every band is therefore white noise of that energy per sample, with that
    covariance; the covariance's W is one.
    """
    if field.moment_order < 2 * sh_order:
        raise ValueError(f"the late field's moments are of order {field.moment_order}; an order-{sh_order} output needs {2 * sh_order}")
    n = field.n_samples
    K = (sh_order + 1) ** 2
    noise = np.random.default_rng(field.seed).standard_normal((n, K))

    shape = np.asarray(field.shape, dtype=float)
    per_bin = shape.ndim == 3
    n_moments = (2 * sh_order + 1) ** 2
    out = np.zeros((n, K))
    B = int(field.bin_samples)
    for b, band in enumerate(_octave_bands(noise, fs)):
        amplitude = np.sqrt(np.maximum(field.energy[b], 0.0))[:, np.newaxis]
        if not per_bin:
            L = _square_roots(covariance(shape[b, :n_moments], sh_order))
            out += amplitude * (band @ L.T)
            continue
        # knots at the bins' centres; each segment between two knots blends their factors linearly
        L = _square_roots(covariance(shape[b, :, :n_moments], sh_order))  # (n_bins, K, K)
        n_knots = L.shape[0]
        position = np.clip((np.arange(n) + 0.5) / B - 0.5, 0.0, n_knots - 1.0)
        left = np.minimum(position.astype(int), n_knots - 1)
        frac = (position - left)[:, np.newaxis]
        mixed = np.empty((n, K))
        bounds = np.searchsorted(left, np.arange(n_knots + 1))
        for k in range(n_knots):
            a, z = bounds[k], bounds[k + 1]
            if a == z:
                continue
            here = band[a:z]
            mixed[a:z] = (1.0 - frac[a:z]) * (here @ L[k].T) + frac[a:z] * (here @ L[min(k + 1, n_knots - 1)].T)
        out += amplitude * mixed
    return out
