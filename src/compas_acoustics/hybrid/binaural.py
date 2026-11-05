"""Binaural rendering of the hybrid: every early path and every delay line through the HRIR measured nearest its direction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyfar as pf
from scipy.signal import fftconvolve

from compas_acoustics.defaults import Defaults

__all__ = ["DEFAULT_HRTF", "HrtfSet", "render_binaural"]

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


def hrir_groups(hrtf: HrtfSet, doa) -> tuple[np.ndarray, np.ndarray]:
    """The measured directions nearest the directions of arrival ``doa`` (M, 3), and the (M, G) one-hot
    weights that sum the paths sharing one onto a channel per measured direction."""
    nearest = hrtf.nearest(doa)
    groups = np.unique(nearest)
    return groups, (nearest[:, np.newaxis] == groups[np.newaxis, :]).astype(float)


def render_groups(monos: np.ndarray, groups: np.ndarray, hrtf: HrtfSet, n_samples: int) -> np.ndarray:
    """The two ears, (n, 2): channel ``g`` of ``monos`` (n, G) through the HRIR of measured direction ``groups[g]``."""
    out = np.zeros((n_samples, 2))
    for g, index in enumerate(groups):
        out += fftconvolve(monos[:, g : g + 1], hrtf.hrirs[index].T, axes=0)[:n_samples]
    return out


def render_binaural(core, hrtf: HrtfSet):
    """A :class:`~compas_acoustics.hybrid.synthesis.HybridCore` at the two ears, (n, 2), as a
    :class:`~compas_acoustics.hybrid.synthesis.HybridResult`.

    Early part: every image, diffracted path and BTMS contribution has the band magnitude, fractional delay
    and radiation taper of :func:`~compas_acoustics.hybrid.early.render_images` and goes through the HRIR
    measured nearest its direction; paths sharing an HRIR are summed first. Late part: line ``i`` after the late envelope
    (:meth:`~compas_acoustics.hybrid.synthesis.HybridCore.shaped_lines`, fitted against the paths' omnidirectional
    sum), over ``sqrt(N)`` (the W row of the ambisonic output matrix), through the HRIR nearest
    ``core.line_directions[i]``.
    """
    # pyFDN, on Python >= 3.11 only; the Auraliser imports this module without it
    from compas_acoustics.hybrid.diffraction import encode_btms
    from compas_acoustics.hybrid.diffraction import encode_diffraction
    from compas_acoustics.hybrid.early import render_images
    from compas_acoustics.hybrid.synthesis import HybridResult

    if hrtf.sampling_rate != int(core.fs):
        raise ValueError(f"the HRTF set has sample rate {hrtf.sampling_rate}; the response has {core.fs}")
    n = core.n_samples
    images, paths, contributions = core.images, list(core.diffraction_paths), list(core.btms_contributions)
    # an empty Coordinates set has no cartesian
    image_doa = np.asarray(images.doa.cartesian, dtype=float).reshape(-1, 3) if np.asarray(images.delay).size else np.zeros((0, 3))
    doa = np.vstack([image_doa] + [np.array([p.doa for p in group], dtype=float).reshape(-1, 3) for group in (paths, contributions)])
    early, early_w = np.zeros((n, 2)), np.zeros(n)
    if doa.shape[0]:
        groups, onehot = hrir_groups(hrtf, doa)
        i, j = image_doa.shape[0], image_doa.shape[0] + len(paths)
        # channel 0 is W, the omnidirectional early field the late envelope compares the lines with; then one per HRIR
        channels = render_images(images, np.hstack([np.ones((i, 1)), onehot[:i]]), n) if i else np.zeros((n, 1 + groups.size))
        if paths:
            channels = channels + encode_diffraction(paths, 0, n, core.fs, extra_weights=onehot[i:j])
        if contributions:
            channels = channels + encode_btms(contributions, 0, n, core.fs, extra_weights=onehot[j:])
        early_w, early = channels[:, 0], render_groups(channels[:, 1:], groups, hrtf, n)
    shaped = core.shaped_lines(early_w)
    lines = shaped / np.sqrt(shaped.shape[1])
    nearest = hrtf.nearest(core.line_directions)
    late = np.zeros((n, 2))
    for index in np.unique(nearest):
        line = lines[:, nearest == index].sum(axis=1)
        late += fftconvolve(line[:, np.newaxis], hrtf.hrirs[index].T, axes=0)[:n]
    return HybridResult(core=core, rir=early + late, early=early, late=late, late_lines=shaped, normalization=None)
