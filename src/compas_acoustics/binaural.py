"""Binaural rendering: MagLS decoding filters, HRTF sets, and the hybrid with an HRTF per path and per delay line.

:func:`magls_filters` are the decoding filters of :meth:`Auraliser.decode_binaural`.

The ambisonic output of :class:`~compas_acoustics.simulation_methods.Hybrid` is limited
to its ``sh_order``: MagLS decodes it with the right magnitudes but a spatially
smoothed image. The hybrid's binaural output is rendered from the paths and
lines themselves instead: every image source and diffracted path is filtered
with the head-related transfer function measured nearest its direction of
arrival, as RAVEN renders its image sources, and every delay line of the late
field with the one nearest the line's direction, as RAZR and RAVEN render
their feedback delay networks.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyfar as pf
from scipy.signal import fftconvolve

from compas_acoustics.defaults import Defaults

__all__ = ["DEFAULT_HRTF", "HrtfSet", "magls_filters", "render_early_binaural", "binaural_rir"]

# MagLS binaural decoding transition frequency per ambisonic order, in Hz
_MAGLS_TRANSITION_PER_ORDER = 600

# Binaural decoding filters keyed by (HRTF path, sample rate, order, covariance constraint), shared across instances
_BINAURAL_FILTERS_CACHE = {}

# The HRTF set every binaural rendering uses unless given another
DEFAULT_HRTF = Path(Defaults().HRTF_LIBRARY_PATH) / "KEMAR_FreeFieldComp_48kHz.sofa"


def _quadrature_weights(azimuth, colatitude):
    """Quadrature weights of a spherical sampling grid by pseudo-inverse, normalised to sum to 1.

    A port of ``spaudiopy.grids.calculate_grid_weights`` (Hold, MIT): the weights are
    ``sqrt(4 pi)`` times the first row of the pseudo-inverse of the real orthonormal
    (N3D) spherical harmonic matrix at the highest order the grid supports, searched as
    spaudiopy does by the condition number of ``Y^H Y`` against ``2 (order + 1)``. A grid that
    does not even support order 1 (a hemisphere, say) gets order 0, i.e. equal weights, where
    spaudiopy stops on an assertion; equal weights are also what SAF uses without weights.

    References
    ----------
    Fornberg, B., & Martel, J. M. (2014). On spherical harmonics based numerical quadrature
    over the surface of a sphere. Advances in Computational Mathematics, 40.
    """
    import spharpy

    azimuth = np.asarray(azimuth, dtype=float).ravel()
    coordinates = pf.Coordinates.from_spherical_colatitude(
        azimuth, np.asarray(colatitude, dtype=float).ravel(), np.ones_like(azimuth))

    def basis(order):
        return spharpy.spherical.spherical_harmonic_basis_real(
            order, coordinates, normalization='N3D', channel_convention='ACN', condon_shortley=False)

    for order in range(1, 100):
        Y = basis(order)
        if np.linalg.cond(Y.T @ Y) > 2 * (order + 1):
            order -= 1
            break
    else:
        raise ValueError("could not find the spherical harmonic order the grid supports")
    weights = np.sqrt(4 * np.pi) * np.linalg.pinv(basis(order))[0]
    if np.abs(np.sum(weights) - 4 * np.pi) > 0.01 or np.any(weights < 0):
        warnings.warn("Could not calculate the quadrature weights of the grid", stacklevel=2)
    return weights / np.sum(weights)


def _diffuse_covariance_matched(decoder, hrtfs, Y, weights):
    """The decoder corrected per bin so its binaural covariance in a diffuse field equals the HRTFs'.

    ``decoder`` is (Q, 2, K), ACN/SN3D with W = 1; ``hrtfs`` (D, 2, K); ``Y`` (D, Q) the same
    convention's harmonics at the HRTF directions; ``weights`` (D,) their quadrature weights.
    Every bin but the last (Nyquist) is corrected. A port of ``applyDiffCovMatching`` in
    Spatial_Audio_Framework (McCormack, ISC), after Zaunschirm et al. (2018): SAF's
    ``utility_cchol`` returns the upper factor, ``C = X^H X``, and ``utility_csvd`` the
    ``V`` of ``A = U S V^H``.

    References
    ----------
    Zaunschirm, M., Schoerkhuber, C., & Hoeldrich, R. (2018). Binaural rendering of Ambisonic
    signals by head-related impulse response time alignment and a diffuseness constraint.
    The Journal of the Acoustical Society of America, 143(6).
    """
    def hermitian(a):
        return a.conj().swapaxes(-1, -2)

    def upper_cholesky(h):
        covariance = (h * weights) @ hermitian(h)
        index = np.arange(2)
        covariance[..., index, index] = covariance[..., index, index].real
        factors = np.zeros_like(covariance)
        positive_definite = np.ones(len(covariance), dtype=bool)
        try:
            factors = hermitian(np.linalg.cholesky(covariance))
        except np.linalg.LinAlgError:
            for k, c in enumerate(covariance):
                try:
                    factors[k] = hermitian(np.linalg.cholesky(c))
                except np.linalg.LinAlgError:
                    positive_definite[k] = False
        return factors, positive_definite

    bins = slice(0, decoder.shape[2] - 1)
    B = decoder[..., bins].transpose(2, 0, 1)  # (K, Q, 2)
    X, X_ok = upper_cholesky(hrtfs[..., bins].transpose(2, 1, 0))
    X_ambi, X_ambi_ok = upper_cholesky(np.einsum('kqe,dq->ked', B, Y))
    # SAF's utility_cchol zeroes the factor of a matrix that is not positive definite, which
    # leaves that bin's decoder zeroed by the correction (it warns only in debug builds)
    ok = X_ok & X_ambi_ok
    U, _, Vh = np.linalg.svd(hermitian(X_ambi[ok]) @ X[ok])
    M = np.linalg.solve(X_ambi[ok], hermitian(Vh) @ hermitian(U) @ X[ok])
    matched = decoder.copy()
    corrected = np.zeros_like(B)
    corrected[ok] = B[ok] @ M.conj()
    matched[..., bins] = corrected.transpose(1, 2, 0)
    if not ok.all():
        warnings.warn(
            f"The diffuse-field covariance of {np.count_nonzero(~ok)} bin(s) is not positive definite "
            "(for example ears identical at DC): the decoder is zero there, as in Spatial_Audio_Framework.",
            stacklevel=3,
        )
    return matched


def magls_filters(hrtf_path, sampling_rate, order, covariance_constraint=True):
    """Magnitude least-squares (MagLS) binaural decoding filters up to ``order``, in ACN/SN3D.

    Below a transition frequency of ``order * 600`` Hz the filters are the least-squares
    spherical harmonic fit of the HRTFs. Above it only the HRTF magnitudes are fitted,
    with the phase continued from the frequency below using the spherically averaged
    group delay. This avoids the high-frequency loss of a plain least-squares fit at
    low orders. Follows ``spaudiopy.decoder.magls_bin`` (``hf_cont='avg'``).

    With ``covariance_constraint`` every bin is then corrected so that the decoder's binaural
    covariance in a diffuse field, the ear energies and their coherence, equals the HRTF set's
    (Zaunschirm et al. 2018), as the SPARTA binaural decoder does: the port of
    ``applyDiffCovMatching`` in Spatial_Audio_Framework, with quadrature weights by
    pseudo-inverse as ``spaudiopy.grids.calculate_grid_weights``.

    References
    ----------
    Hold, C., Meyer-Kahlen, N., & Pulkki, V. (2023). Magnitude-Least-Squares Binaural
    Ambisonic Rendering with Phase Continuation. Fortschritte der Akustik - DAGA.

    Zaunschirm, M., Schoerkhuber, C., & Hoeldrich, R. (2018). Binaural rendering of Ambisonic
    signals by head-related impulse response time alignment and a diffuseness constraint.
    The Journal of the Acoustical Society of America, 143(6).

    Zotter, F., & Frank, M. (2019). Ambisonics. Springer Topics in Signal Processing.

    Returns
    -------
    np.ndarray
        Filters shaped (samples, (order + 1)**2, 2), with ears ordered left, right.
    """
    try:
        import spharpy
    except ImportError as error:
        raise ImportError(
            "Binaural decoding needs spharpy, which needs Python >= 3.11 (Rhino 8 runs 3.9); "
            "use output_format='stereo' or 'mono' there."
        ) from error

    key = (str(Path(hrtf_path).resolve()), sampling_rate, order, bool(covariance_constraint))
    if key in _BINAURAL_FILTERS_CACHE:
        return _BINAURAL_FILTERS_CACHE[key]

    hrirs, directions, _ = pf.io.read_sofa(str(hrtf_path), verbose=False)
    if hrirs.sampling_rate != sampling_rate:
        hrirs = pf.dsp.resample(hrirs, sampling_rate, match_amplitude='freq')
    n_samples = hrirs.n_samples
    n_fft = max(1024, n_samples)
    hrtfs = np.fft.rfft(hrirs.time, n=n_fft)  # (directions, 2, bins)
    freqs = np.fft.rfftfreq(n_fft, 1 / sampling_rate)

    # spharpy's SN3D includes 1/sqrt(4 pi); scale to the ambisonic SN3D convention (W = 1)
    Y = np.sqrt(4 * np.pi) * spharpy.spherical.spherical_harmonic_basis_real(
        order, directions, normalization='SN3D', channel_convention='ACN', condon_shortley=False)
    Y_pinv = np.linalg.pinv(Y)

    # Least-squares fit below the transition frequency
    k_trans = int(np.argmin(np.abs(freqs - _MAGLS_TRANSITION_PER_ORDER * order)))
    hrtfs_nm = np.zeros((Y.shape[1], 2, len(freqs)), dtype=complex)
    hrtfs_nm[..., :k_trans] = np.einsum('qd,dek->qek', Y_pinv, hrtfs[..., :k_trans])

    # Phase step per bin from the spherical average (W channel) just below the transition
    n_delta = 5
    phase_w = np.unwrap(np.angle(hrtfs_nm[0, :, :k_trans]), axis=-1)
    delta_phase = np.mean(np.diff(phase_w, axis=-1)[:, k_trans - n_delta - 1:k_trans - 1], axis=-1)

    # Magnitude-only fit above, continuing the phase of the previous bin
    for k in range(k_trans, len(freqs)):
        phase = np.angle(Y @ hrtfs_nm[:, :, k - 1]) + delta_phase
        hrtfs_nm[:, :, k] = Y_pinv @ (np.abs(hrtfs[:, :, k]) * np.exp(1j * phase))

    if covariance_constraint:
        weights = _quadrature_weights(directions.azimuth, directions.colatitude)
        hrtfs_nm = _diffuse_covariance_matched(hrtfs_nm, hrtfs, Y, weights)

    filters = np.fft.irfft(hrtfs_nm, n=n_fft)[..., :n_samples]
    # Taper the start and end against truncation
    filters[..., -32:] *= np.hanning(64)[32:]
    filters[..., :4] *= np.hanning(8)[:4]

    _BINAURAL_FILTERS_CACHE[key] = filters.transpose(2, 0, 1)
    return _BINAURAL_FILTERS_CACHE[key]


@dataclass(frozen=True)
class HrtfSet:
    """Head-related impulse responses and the directions they were measured from.

    ``hrirs`` is (directions, 2, taps), ears left then right; ``directions`` are unit
    vectors in the listener's frame (x front, y left, z up), as SOFA stores them.
    """

    hrirs: np.ndarray
    directions: np.ndarray
    sampling_rate: int

    @classmethod
    def from_sofa(cls, path, sampling_rate: int) -> HrtfSet:
        """Read a SOFA HRIR set, resampled to ``sampling_rate`` as :func:`magls_filters` resamples it."""
        hrirs, directions, _ = pf.io.read_sofa(str(path), verbose=False)
        if hrirs.sampling_rate != sampling_rate:
            hrirs = pf.dsp.resample(hrirs, sampling_rate, match_amplitude="freq")
        xyz = np.asarray(directions.cartesian, dtype=float)
        return cls(np.asarray(hrirs.time, dtype=float), xyz / np.linalg.norm(xyz, axis=1, keepdims=True), int(sampling_rate))

    def nearest(self, doa) -> np.ndarray:
        """Index of the measured direction nearest each direction of arrival, by angle."""
        doa = np.asarray(doa, dtype=float).reshape(-1, 3)
        doa = doa / np.linalg.norm(doa, axis=1, keepdims=True)
        return np.argmax(doa @ self.directions.T, axis=1)


def _terms(result, freq: np.ndarray, nfft: int, n_samples: int) -> list:
    """``(direction of arrival, spectrum thunk)`` for every early path that arrives within ``n_samples``.

    The spectra are those :func:`render_images`, :func:`render_transfers` and
    :func:`render_signals` accumulate, before the shared radiation taper.
    """
    from compas_acoustics.hybrid.bands import BANDS
    from compas_acoustics.hybrid.early import _band_magnitude_fn

    fs = result.fs
    terms = []
    iss = result.images
    if iss is not None and np.asarray(iss.delay).size:
        magnitude = _band_magnitude_fn(freq, iss.bands, fs)
        doa = iss.doa.cartesian
        for m in np.flatnonzero(iss.delay * fs < n_samples):
            terms.append((doa[m], lambda m=m: magnitude(iss.gain[m]) * np.exp(-2j * np.pi * freq * iss.delay[m])))
    if result.diffraction_paths or result.btms_contributions:
        magnitude = _band_magnitude_fn(freq, BANDS, fs)
    for p in result.diffraction_paths:
        if p.delay * fs < n_samples:
            gain = p.band_gain / (4.0 * np.pi * p.distance)
            terms.append((p.doa, lambda p=p, gain=gain: p.transfer(freq) * magnitude(gain) * np.exp(-2j * np.pi * freq * p.delay)))
    for p in result.btms_contributions:
        if p.start < n_samples:
            def spectrum(p=p):
                buffer = np.zeros(nfft)
                stop = min(p.start + p.ir.size, nfft)
                buffer[p.start:stop] = p.ir[: stop - p.start] / (4.0 * np.pi)
                return np.fft.rfft(buffer) * magnitude(p.band_gain)
            terms.append((p.doa, spectrum))
    return terms


def render_early_binaural(result, hrtf: HrtfSet, n_samples: int | None = None) -> np.ndarray:
    """The early part of a :class:`~compas_acoustics.hybrid.synthesis.HybridResult` at the two ears, (n, 2).

    Each path keeps the band magnitude, fractional delay and radiation taper of
    its ambisonic encoding; instead of the spherical harmonics of its direction
    it takes the HRIR measured nearest to it. Paths sharing a measured direction
    are summed first, so each HRIR is applied once.
    """
    # pyFDN, on Python >= 3.11 only; the Auraliser imports this module without it
    from compas_acoustics.hybrid.early import _fft_grid
    from compas_acoustics.hybrid.early import _finish

    if result.images is None:
        raise ValueError("this HybridResult does not hold its early paths; synthesise it again")
    if hrtf.sampling_rate != int(result.fs):
        raise ValueError(f"the HRTF set has sample rate {hrtf.sampling_rate}; the response has {result.fs}")
    n = result.early.shape[0] if n_samples is None else int(n_samples)
    nfft, freq = _fft_grid(n, result.fs)
    out = np.zeros((n, 2))
    terms = _terms(result, freq, nfft, n)
    if not terms:
        return out
    nearest = hrtf.nearest(np.array([doa for doa, _ in terms]))
    for index in np.unique(nearest):
        spectrum = np.zeros(freq.size, dtype=complex)
        for k in np.flatnonzero(nearest == index):
            spectrum += terms[k][1]()
        mono = _finish(spectrum[:, np.newaxis], freq, nfft, n)
        out += fftconvolve(mono, hrtf.hrirs[index].T, axes=0)[:n]
    return out


def binaural_rir(result, hrtf: HrtfSet) -> np.ndarray:
    """The hybrid at the two ears, (n, 2): every early path and every delay line with the HRIR nearest its direction.

    The early part is :func:`render_early_binaural`. The late field is rendered
    per delay line, as RAZR and RAVEN render their FDN outputs: line ``i``,
    ``result.late_lines[:, i] / sqrt(N)``, through the HRIR measured nearest
    ``result.line_directions[i]``, lines sharing an HRIR summed first.
    ``1 / sqrt(N)`` is the W row of the ambisonic output matrix, so each ear
    receives the plane waves the ambisonic late field's W describes.
    """
    if result.late_lines is None or result.line_directions is None:
        raise ValueError("this HybridResult does not hold its delay lines; synthesise it again")
    n = result.rir.shape[0]
    out = render_early_binaural(result, hrtf, n)
    lines = np.asarray(result.late_lines, dtype=float) / np.sqrt(result.late_lines.shape[1])
    nearest = hrtf.nearest(result.line_directions)
    for index in np.unique(nearest):
        line = lines[:, nearest == index].sum(axis=1)
        out += fftconvolve(line[:, np.newaxis], hrtf.hrirs[index].T, axes=0)[:n]
    return out
