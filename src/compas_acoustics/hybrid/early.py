"""Image sources of one pair, read from a search over the whole scene, and their encoding to ambisonics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pyfar as pf
import spharpy
from scipy.signal import butter, freqs_zpk
from scipy.spatial import cKDTree

from .bands import BANDS, N_BANDS, check_bands
from .pair import Pair


@dataclass(frozen=True)
class ImageSourceSet:
    """Delay, band gains and direction of every image source, independent of the output format."""

    delay: np.ndarray  # (M,) seconds
    gain: np.ndarray  # (M, 8) linear pressure gain
    doa: pf.Coordinates  # (M,) receiver-relative direction of arrival
    order: np.ndarray  # (M,) reflection order
    bands: np.ndarray  # (8,)
    fs: float
    # (M, 8) linear pressure gain of the scattered part; None without scattering
    scattered: np.ndarray | None = None
    wall: np.ndarray | None = None  # (M,) the generating face index, -1 for order 0

    def __post_init__(self) -> None:
        check_bands(self.bands)
        m = np.asarray(self.delay).size
        if self.gain.shape != (m, N_BANDS):
            raise ValueError(f"gain must have shape ({m}, {N_BANDS})")
        if self.doa.cshape != (m,):
            raise ValueError(f"doa must hold {m} directions")
        if self.scattered is not None and self.scattered.shape != (m, N_BANDS):
            raise ValueError(f"scattered must have shape ({m}, {N_BANDS})")


def _band_damping(source) -> np.ndarray:
    """A per-band search's reflection products on the package grid, (M, 8)."""
    damping = np.asarray(source.damping, dtype=float)
    n_images = np.asarray(source.images).shape[1]
    if damping.ndim != 2 or damping.shape[1] != n_images:
        raise RuntimeError(f"expected damping of shape (n_bands, {n_images}), got {damping.shape}: the search did not return a per-band reflection product")
    if damping.shape[0] == 1 and np.all(np.asarray(source.orders) == 0):
        # searched to order 0 in a room with obstacles, one band is kept: the direct sound is 1 in every band
        damping = np.repeat(damping, N_BANDS, axis=0)
    if damping.shape[0] < N_BANDS:
        raise RuntimeError(f"the search returned {damping.shape[0]} bands, fewer than the {N_BANDS} this package needs; its octave grid did not align")
    # rows above the grid (16 kHz at 48 kHz) are dropped; `encode` extrapolates above the top band
    return damping[:N_BANDS].T


def _first_of_each_position(images: np.ndarray, spec: Pair) -> np.ndarray:
    """Mask over ``images`` (3, M) keeping the first of every group at one position.

    The tolerance, 1e-5 of the room's extent, is well above the float32 rounding of one image reached twice
    and far below the distance between two distinct images.
    """
    keep = np.ones(images.shape[1], dtype=bool)
    if images.shape[1] < 2:
        return keep
    tolerance = 1e-5 * max(float(np.ptp(spec.geometry.vertices, axis=0).max()), 1.0)
    for i, j in cKDTree(images.T).query_pairs(tolerance):
        keep[max(i, j)] = False
    return keep


def image_sources(room, source_index: int, mic_index: int, spec: Pair, directivity=None, scattered_room=None) -> ImageSourceSet:
    """The image sources of one source at one microphone of a computed room.

    ``room`` has been searched with every source and microphone of the scene; an image is kept when any
    microphone sees it, so this pair's are those visible at ``mic_index``. Delays, per-band gains and
    receiver-relative directions of arrival; the direct sound is order zero; ``wall`` is each image's
    generating face. The room must be built without air absorption, applied here from the path length.
    ``directivity`` is evaluated along each image's emission direction from the real source.

    An image found more than once is kept once: a path passing exactly through an edge is accepted from both
    wall orderings, which grid layouts (source and receiver at one height) hit routinely.

    Surface scattering (Ewert et al. 2025, Sec. 2.1.3) splits each image per band into a specular part,
    ``prod sqrt(1 - s_i)``, and a scattered part, ``prod sqrt(s_i)``, over its reflections. ``room`` is then
    built with ``ray_tracing=True``, whose damping holds the specular product, and ``scattered_room`` the same
    way from :func:`~compas_acoustics.hybrid.scattering.complementary_scattering`; ``scattered`` is the gain
    of its images, matched by position, zero for the direct sound.
    """
    source = room.sources[source_index]
    visible = np.array(room.visibility[source_index][mic_index], dtype=bool)
    visible[visible] = _first_of_each_position(np.asarray(source.images, dtype=float)[:, visible], spec)
    images = np.asarray(source.images, dtype=float)[:, visible]  # (3, M)
    order = np.asarray(source.orders, dtype=int).ravel()[visible]
    wall = np.asarray(source.walls, dtype=int).ravel()[visible]
    wall_gain = _band_damping(source)[visible]

    offset = images - spec.receiver[:, np.newaxis]  # (3, M)
    distance = np.linalg.norm(offset, axis=0)  # (M,)
    delay = distance / spec.speed_of_sound
    spreading = 1.0 / (4.0 * np.pi * distance)

    # m is an energy attenuation coefficient, so pressure loses exp(-m d/2)
    # over the image's own path length; zero without air absorption.
    wall_gain = wall_gain * np.exp(-np.outer(distance, spec.air_attenuation()) / 2.0)

    gain = wall_gain * spreading[:, np.newaxis]

    if directivity is not None:
        emitted = np.asarray(source.directions[mic_index], dtype=float)[:, visible]
        azimuth = np.arctan2(emitted[1], emitted[0])
        colatitude = np.arccos(np.clip(emitted[2] / np.linalg.norm(emitted, axis=0), -1.0, 1.0))
        response = directivity.get_response(azimuth=azimuth, colatitude=colatitude, degrees=False)
        gain = gain * np.asarray(response, dtype=float).reshape(-1, 1)

    # the rotation's columns are the receiver's axes, so the transpose takes a world vector to its frame
    offset = spec.receiver_rotation.T @ offset

    scattered = None
    if scattered_room is not None:
        scattered = np.zeros_like(gain)
        other = image_sources(scattered_room, source_index, mic_index, spec, directivity)
        if other.delay.size != order.size:
            raise RuntimeError("the scattered room's image sources are not those of the room")
        if order.size:
            tolerance = 1e-5 * max(float(np.ptp(spec.geometry.vertices, axis=0).max()), 1.0)
            gap, match = cKDTree(other.doa.cartesian).query(offset.T)
            if np.any(gap > tolerance):
                raise RuntimeError("the scattered room's image sources are not those of the room")
            scattered = np.where((order > 0)[:, np.newaxis], other.gain[match], 0.0)

    return ImageSourceSet(
        delay=delay,
        gain=gain,
        doa=pf.Coordinates(offset[0], offset[1], offset[2]),
        order=order,
        bands=BANDS.copy(),
        fs=spec.fs,
        scattered=scattered,
        wall=wall,
    )


def _radiation_taper(freq: np.ndarray) -> np.ndarray:
    """Causal fourth-order Butterworth high-pass at 35 Hz, exactly zero at DC.

    Band gains start at 62.5 Hz and are held flat to f = 0, where every image's phase ramp is 1, so the image
    set sums coherently there; that DC and infrasonic energy would bias the FDN drive and every convolution.
    The taper costs 0.04 dB at 62.5 Hz and 0.6 dB at the band's 44 Hz lower edge.

    It is the analog filter's complex response, minimum phase, so it rings only after each arrival and no
    energy precedes the direct sound.
    """
    zeros, poles, gain = butter(4, 2.0 * np.pi * 35.0, "high", analog=True, output="zpk")
    return freqs_zpk(zeros, poles, gain, worN=2.0 * np.pi * np.asarray(freq, dtype=float))[1]


def _extrapolated_top_db(gain_db: np.ndarray, octaves: float) -> float:
    """Gain at Nyquist, continuing the last octave's slope, never upward.

    The top 1.5 octaves at 48 kHz keep the wall and air roll-off, consistent with the FDN's decay, which
    shortens there too (:meth:`~compas_acoustics.hybrid.target.RoomTarget.geq_bands`). A rising slope (absorption falling with
    frequency) is clamped to flat: there is no data to amplify.
    """
    slope_db_per_octave = min(float(gain_db[-1] - gain_db[-2]), 0.0)
    return float(gain_db[-1] + slope_db_per_octave * octaves)


def encode(
    iss: ImageSourceSet,
    sh_order: int,
    n_samples: int,
    *,
    normalization: str = "NM",
) -> np.ndarray:
    """An image source set as an ambisonic impulse response, ACN, N3D by default (``"NM"``: W = 1 for a unit plane wave).

    The band gains are interpolated onto the FFT grid in log-frequency and dB, the delay applied as an exact
    phase ramp, and the images accumulated in the frequency domain. Above the top band the magnitude continues
    its slope (:func:`_extrapolated_top_db`), below the lowest it is tapered to zero at DC (:func:`_radiation_taper`).
    """
    if sh_order < 0:
        raise ValueError("sh_order must be non-negative")
    if n_samples < 1:
        raise ValueError("n_samples must be positive")
    # an empty set is legitimate (no orders below the driving one at max_order=0), but spharpy cannot take it
    if np.asarray(iss.delay).size == 0:
        return np.zeros((n_samples, (sh_order + 1) ** 2), dtype=float)

    sh = spharpy.spherical.spherical_harmonic_basis_real(sh_order, iss.doa, normalization=normalization, channel_convention="ACN")  # (M, n_ch)
    return render_images(iss, sh, n_samples)


def render_images(iss: ImageSourceSet, weights: np.ndarray, n_samples: int) -> np.ndarray:
    """Image sources accumulated onto output channels, (n, n_ch), with ``weights`` (M, n_ch) per image.

    ``weights`` are spherical harmonics for ambisonic channels or a routing onto delay lines or HRIRs; every
    image gets the same band magnitude, exact fractional delay and radiation taper (:func:`encode`).
    """
    if n_samples < 1:
        raise ValueError("n_samples must be positive")
    weights = np.asarray(weights, dtype=float)
    if weights.ndim != 2 or weights.shape[0] != np.asarray(iss.delay).size:
        raise ValueError(f"weights must have shape ({np.asarray(iss.delay).size}, n_channels), got {weights.shape}")
    n_ch = weights.shape[1]

    # images after the buffer's end would wrap round to its start
    inside = iss.delay * iss.fs < n_samples
    if not np.any(inside):
        return np.zeros((n_samples, n_ch), dtype=float)

    nfft, freq = _fft_grid(n_samples, iss.fs)
    magnitude = _band_magnitude_fn(freq, iss.bands, iss.fs)
    spectrum = _accumulate(
        lambda m: magnitude(iss.gain[m]) * np.exp(-2j * np.pi * freq * iss.delay[m]),
        np.flatnonzero(inside),
        weights,
        freq.size,
    )
    return _finish(spectrum, freq, nfft, n_samples)


def _accumulate(spectrum_of, paths: np.ndarray, weights: np.ndarray, n_freq: int) -> np.ndarray:
    """``sum_m outer(spectrum_of(m), weights[m])`` over ``paths``, as matrix products over chunks of 16 paths.

    Each chunk is one pass over the bins; 16 complex spectra at a million bins take 256 MB, which bounds the chunk.
    """
    weights = np.asarray(weights, dtype=complex)
    spectrum = np.zeros((n_freq, weights.shape[1]), dtype=complex)
    for start in range(0, paths.size, 16):
        chunk = paths[start : start + 16]
        # the spectra as columns, so the product reads them contiguously
        columns = np.empty((n_freq, chunk.size), dtype=complex)
        for k, m in enumerate(chunk):
            columns[:, k] = spectrum_of(m)
        spectrum += columns @ weights[chunk]
    return spectrum


def _fft_grid(n_samples: int, fs: float) -> tuple[int, np.ndarray]:
    # headroom: ringing past the buffer's end and the pre-ringing of a kernel near t = 0 fall into the
    # discarded tail and do not alias into the response
    nfft = 1 << (int(np.ceil(np.log2(max(n_samples, 2)))) + 1)
    return nfft, np.fft.rfftfreq(nfft, 1.0 / fs)


def _band_magnitude_fn(freq: np.ndarray, bands: np.ndarray, fs: float):
    """Band gains to a magnitude on ``freq``, interpolated in log f and dB."""
    log_freq = np.log2(np.clip(freq, bands[0] / 2.0, None))
    nyquist = fs / 2.0
    log_bands = np.append(np.log2(bands), np.log2(nyquist))
    top_octaves = np.log2(nyquist / bands[-1])

    def magnitude(gain: np.ndarray) -> np.ndarray:
        gain_db = 20.0 * np.log10(np.maximum(gain, 1e-300))
        gain_db = np.append(gain_db, _extrapolated_top_db(gain_db, top_octaves))
        return 10.0 ** (np.interp(log_freq, log_bands, gain_db) / 20.0)

    return magnitude


def _finish(spectrum: np.ndarray, freq: np.ndarray, nfft: int, n_samples: int) -> np.ndarray:
    spectrum *= _radiation_taper(freq)[:, np.newaxis]
    return np.fft.irfft(spectrum, nfft, axis=0)[:n_samples]


def render_transfers(
    delays: np.ndarray,
    transfers,
    gains: np.ndarray,
    weights: np.ndarray,
    bands: np.ndarray,
    fs: float,
    n_samples: int,
) -> np.ndarray:
    """As :func:`render_images`, for paths whose band magnitude ``gains[m]`` is multiplied by ``transfers[m](freq)``."""
    delays = np.asarray(delays, dtype=float)
    weights = np.asarray(weights, dtype=float).reshape(delays.size, -1)
    nfft, freq = _fft_grid(n_samples, fs)
    magnitude = _band_magnitude_fn(freq, bands, fs)
    spectrum = _accumulate(
        lambda m: transfers[m](freq) * magnitude(gains[m]) * np.exp(-2j * np.pi * freq * delays[m]),
        np.flatnonzero(delays * fs < n_samples),
        weights,
        freq.size,
    )
    return _finish(spectrum, freq, nfft, n_samples)


def render_signals(
    signals: list[tuple[int, np.ndarray]],
    gains: np.ndarray,
    bands: np.ndarray,
    fs: float,
    n_samples: int,
) -> np.ndarray:
    """As :func:`render_images`, for sampled responses ``(start_sample, block (length, n_ch))`` weighted per channel.

    Blocks whose band gains agree to 0.01 dB are summed in the time domain and filtered once.
    """
    if n_samples < 1:
        raise ValueError("n_samples must be positive")
    if not signals:
        return np.zeros((n_samples, 0))
    gains = np.asarray(gains, dtype=float)
    nfft, freq = _fft_grid(n_samples, fs)
    magnitude = _band_magnitude_fn(freq, bands, fs)
    n_ch = signals[0][1].shape[1]
    keys = np.round(20.0 * np.log10(np.maximum(gains, 1e-300)) / 0.01).astype(np.int64)
    groups: dict[bytes, list[int]] = {}
    for m, key in enumerate(keys):
        groups.setdefault(key.tobytes(), []).append(m)
    spectrum = np.zeros((freq.size, n_ch), dtype=complex)
    for members in groups.values():
        buffer = np.zeros((nfft, n_ch))
        for m in members:
            start, block = signals[m]
            if start >= n_samples:
                continue
            stop = min(start + block.shape[0], nfft)
            buffer[start:stop] += block[: stop - start]
        spectrum += np.fft.rfft(buffer, axis=0) * magnitude(gains[members[0]])[:, np.newaxis]
    return _finish(spectrum, freq, nfft, n_samples)
