"""Binaural rendering of the hybrid: an HRTF per path and per delay line.

The hybrid's binaural output (:class:`~compas_acoustics.simulation_methods.Hybrid`,
``output="binaural"``) is rendered from the paths and lines themselves, without
spherical harmonics: every image source and diffracted path is filtered with the
head-related transfer function measured nearest its direction of arrival, as
RAVEN renders its image sources, and every delay line of the late field with the
one nearest the line's direction, as RAZR renders its feedback delay network
and RAVEN the directivity groups of its ray-tracing histogram. compas_acoustics has no ambisonic-to-binaural decoder: ambisonic
responses are saved or convolved as they are, for a decoder elsewhere.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyfar as pf
from scipy.signal import fftconvolve

from compas_acoustics.defaults import Defaults

__all__ = ["DEFAULT_HRTF", "HrtfSet", "render_early_binaural", "binaural_rir"]

# The HRTF set every binaural rendering uses unless given another
DEFAULT_HRTF = Path(Defaults().HRTF_LIBRARY_PATH) / "KEMAR_FreeFieldComp_48kHz.sofa"


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
        """Read a SOFA HRIR set, resampled to ``sampling_rate`` with its frequency response kept."""
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
    per delay line, as RAZR renders its FDN output and RAVEN its directivity groups: line ``i``,
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
