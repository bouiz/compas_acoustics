"""Binaural rendering: MagLS decoding filters, HRTF sets, and the hybrid's early part with an HRTF per path.

:func:`magls_filters` are the decoding filters of :meth:`Auraliser.decode_binaural`.

The ambisonic early part of :class:`~compas_acoustics.simulation_methods.Hybrid` is limited
to its ``sh_order``: MagLS decodes it with the right magnitudes but a spatially
smoothed image. Here every image source and diffracted path is filtered with
the head-related transfer function measured nearest its direction of arrival,
as RAVEN renders its image sources; the late field, which is diffuse, is
MagLS-decoded as :meth:`Auraliser.decode_binaural` does.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyfar as pf
from scipy.signal import fftconvolve

from compas_acoustics.defaults import Defaults

__all__ = ["DEFAULT_HRTF", "HrtfSet", "magls_filters", "render_early_binaural", "binaural_rir"]

# MagLS binaural decoding transition frequency per ambisonic order, in Hz
_MAGLS_TRANSITION_PER_ORDER = 600

# Binaural decoding filters keyed by (HRTF path, sample rate, order), shared across instances
_BINAURAL_FILTERS_CACHE = {}

# The HRTF set every binaural rendering uses unless given another
DEFAULT_HRTF = Path(Defaults().HRTF_LIBRARY_PATH) / "KEMAR_Knowl_EarSim_SmallEars_FreeFieldComp_48kHz.sofa"


def magls_filters(hrtf_path, sample_rate, order):
    """Magnitude least-squares (MagLS) binaural decoding filters up to ``order``, in ACN/SN3D.

    Below a transition frequency of ``order * 600`` Hz the filters are the least-squares
    spherical harmonic fit of the HRTFs. Above it only the HRTF magnitudes are fitted,
    with the phase continued from the frequency below using the spherically averaged
    group delay. This avoids the high-frequency loss of a plain least-squares fit at
    low orders. Follows ``spaudiopy.decoder.magls_bin`` (``hf_cont='avg'``).

    References
    ----------
    Hold, C., Meyer-Kahlen, N., & Pulkki, V. (2023). Magnitude-Least-Squares Binaural
    Ambisonic Rendering with Phase Continuation. Fortschritte der Akustik - DAGA.

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

    key = (str(Path(hrtf_path).resolve()), sample_rate, order)
    if key in _BINAURAL_FILTERS_CACHE:
        return _BINAURAL_FILTERS_CACHE[key]

    hrirs, directions, _ = pf.io.read_sofa(str(hrtf_path), verbose=False)
    if hrirs.sampling_rate != sample_rate:
        hrirs = pf.dsp.resample(hrirs, sample_rate, match_amplitude='freq')
    n_samples = hrirs.n_samples
    n_fft = max(1024, n_samples)
    hrtfs = np.fft.rfft(hrirs.time, n=n_fft)  # (directions, 2, bins)
    freqs = np.fft.rfftfreq(n_fft, 1 / sample_rate)

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
    sample_rate: int

    @classmethod
    def from_sofa(cls, path, sample_rate: int) -> HrtfSet:
        """Read a SOFA HRIR set, resampled to ``sample_rate`` as :func:`magls_filters` resamples it."""
        hrirs, directions, _ = pf.io.read_sofa(str(path), verbose=False)
        if hrirs.sampling_rate != sample_rate:
            hrirs = pf.dsp.resample(hrirs, sample_rate, match_amplitude="freq")
        xyz = np.asarray(directions.cartesian, dtype=float)
        return cls(np.asarray(hrirs.time, dtype=float), xyz / np.linalg.norm(xyz, axis=1, keepdims=True), int(sample_rate))

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
    # the hybrid extra (pyFDN); the Auraliser imports this module without it
    from compas_acoustics.hybrid.early import _fft_grid
    from compas_acoustics.hybrid.early import _finish

    if result.images is None:
        raise ValueError("this HybridResult does not hold its early paths; synthesise it again")
    if hrtf.sample_rate != int(result.fs):
        raise ValueError(f"the HRTF set has sample rate {hrtf.sample_rate}; the response has {result.fs}")
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


def binaural_rir(result, hrtf: HrtfSet, filters: np.ndarray) -> np.ndarray:
    """Early part with an HRTF per path plus the late field MagLS-decoded, (n, 2).

    ``filters`` are :func:`magls_filters` of the
    result's ambisonic order, from the same HRTF set.
    """
    from compas_acoustics.hybrid.synthesis import _from_n3d

    n = result.rir.shape[0]
    late = result.late if result.normalization == "SN3D" else _from_n3d(result.late, "SN3D")
    if filters.shape[1] != late.shape[1]:
        raise ValueError(f"MagLS filters for {filters.shape[1]} channels; the late field has {late.shape[1]}")
    decoded = np.zeros((n, 2))
    for ch in range(late.shape[1]):
        decoded += fftconvolve(late[:, ch:ch + 1], filters[:, ch, :], axes=0)[:n]
    return render_early_binaural(result, hrtf, n) + decoded
