"""Image sources and their encoding to ambisonics.

Ported from hybrid-rir (gpapadimitri/hybrid-rir @ 8a70661, src/early.py).
The image source search moved to the caller: ``image_sources`` reads one pair
out of a pyroomacoustics room computed for every source and receiver of a
scene, so per-band gains always come from pyroomacoustics' damping (the
shoebox closed form is not ported).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pyfar as pf
import spharpy

from .bands import BANDS, N_BANDS, check_bands
from .pair import Pair
from .target import _air_absorption_coefficient


@dataclass(frozen=True)
class ImageSourceSet:
    """Geometry and material data for every image source.

    Deliberately knows nothing about ambisonics: ``encode`` turns this into a
    signal, and this stays testable against synthetic image lists.
    """

    delay: np.ndarray  # (M,) seconds
    gain: np.ndarray  # (M, 8) linear pressure gain
    doa: pf.Coordinates  # (M,) receiver-relative direction of arrival
    order: np.ndarray  # (M,) reflection order
    bands: np.ndarray  # (8,)
    fs: float

    def __post_init__(self) -> None:
        check_bands(self.bands)
        m = np.asarray(self.delay).size
        if self.gain.shape != (m, N_BANDS):
            raise ValueError(f"gain must have shape ({m}, {N_BANDS})")
        if self.doa.cshape != (m,):
            raise ValueError(f"doa must hold {m} directions")


def _band_damping(source) -> np.ndarray:
    """A per-band search's reflection products on the package grid, (M, 8)."""
    damping = np.asarray(source.damping, dtype=float)
    n_images = np.asarray(source.images).shape[1]
    if damping.ndim != 2 or damping.shape[1] != n_images:
        raise RuntimeError(
            f"expected damping of shape (n_bands, {n_images}), got "
            f"{damping.shape}: pyroomacoustics did not return a per-band "
            "reflection product"
        )
    if damping.shape[0] < N_BANDS:
        raise RuntimeError(
            f"pyroomacoustics returned {damping.shape[0]} bands, fewer than the "
            f"{N_BANDS} this package needs; its octave grid did not align"
        )
    # Rows beyond the package grid (16 kHz at 48 kHz sampling) are discarded;
    # `encode` extrapolates above the top band from its measured slope.
    return damping[:N_BANDS].T


def image_sources(room, source_index: int, mic_index: int, spec: Pair, directivity=None) -> ImageSourceSet:
    """The image sources of one source at one microphone of a computed room.

    ``room`` has had ``image_source_model()`` run with every source and
    microphone of the scene. pyroomacoustics keeps an image when any
    microphone sees it, so the images of this pair are those visible at
    ``mic_index``. Delays, per-band gains and receiver-relative directions of
    arrival; the direct sound is order zero. The room must be built without
    air absorption, which is applied here from the path length.

    ``directivity`` is the source's pyroomacoustics directivity. It is
    evaluated along each image's emission direction from the real source,
    ``source.directions``, which is what pyroomacoustics itself uses.
    """
    source = room.sources[source_index]
    visible = np.asarray(room.visibility[source_index][mic_index], dtype=bool)
    images = np.asarray(source.images, dtype=float)[:, visible]  # (3, M)
    order = np.asarray(source.orders, dtype=int).ravel()[visible]
    wall_gain = _band_damping(source)[visible]

    offset = images - spec.receiver[:, np.newaxis]  # (3, M)
    distance = np.linalg.norm(offset, axis=0)  # (M,)
    delay = distance / spec.speed_of_sound
    spreading = 1.0 / (4.0 * np.pi * distance)

    if spec.air_absorption:
        # m is an energy attenuation coefficient, so pressure loses exp(-m d/2)
        # over the image's own path length.
        wall_gain = wall_gain * np.exp(
            -np.outer(distance, _air_absorption_coefficient()) / 2.0
        )

    gain = wall_gain * spreading[:, np.newaxis]

    if directivity is not None:
        emitted = np.asarray(source.directions[mic_index], dtype=float)[:, visible]
        azimuth = np.arctan2(emitted[1], emitted[0])
        colatitude = np.arccos(
            np.clip(emitted[2] / np.linalg.norm(emitted, axis=0), -1.0, 1.0)
        )
        response = directivity.get_response(
            azimuth=azimuth, colatitude=colatitude, degrees=False
        )
        gain = gain * np.asarray(response, dtype=float).reshape(-1, 1)

    # The rotation's columns are the receiver's axes in world coordinates, so
    # a world vector goes into the receiver frame by the transpose; the matrix
    # itself would mirror the scene left to right.
    offset = spec.receiver_rotation.T @ offset

    return ImageSourceSet(
        delay=delay,
        gain=gain,
        doa=pf.Coordinates(offset[0], offset[1], offset[2]),
        order=order,
        bands=BANDS.copy(),
        fs=spec.fs,
    )


# Corner and order of the taper that removes DC from the encoded response.
# The corner sits below the lowest band's lower edge (44 Hz), so the taper
# costs 0.05 dB at 62.5 Hz and 0.65 dB at that edge. Fourth order because
# gentler slopes do not actually clear the region: measured on the shoebox
# below, first order at 20 Hz still left 4.5 per cent of the energy under
# 20 Hz, where fourth order leaves 0.03 per cent.
_RADIATION_HIGHPASS_HZ = 35.0
_RADIATION_HIGHPASS_ORDER = 4


def _radiation_taper(freq: np.ndarray) -> np.ndarray:
    """Butterworth high-pass magnitude, exactly zero at DC.

    Band gains are only defined from 62.5 Hz up, and ``np.interp`` holds the
    lowest band's value flat all the way down to f = 0. There every image's
    phase ramp is exactly 1, so the whole image set sums coherently: measured
    on a shoebox with realistic absorption, a DC value 26 times the peak sample
    and a quarter of the early field's energy below 20 Hz. A source cannot
    radiate DC, and that content biases the FDN drive, ``drr``
    and anything the response is convolved with. With the taper the same room
    keeps 0.03 per cent below 20 Hz.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(
            freq > 0.0,
            (freq / _RADIATION_HIGHPASS_HZ) ** _RADIATION_HIGHPASS_ORDER,
            0.0,
        )
    return ratio / np.sqrt(1.0 + ratio**2)


def _extrapolated_top_db(gain_db: np.ndarray, octaves: float) -> float:
    """Gain at Nyquist, continuing the last octave's slope. Never upward.

    Holding the 8 kHz value flat to Nyquist leaves the top 1.5 octaves at
    48 kHz with neither wall roll-off nor air absorption, while the FDN's
    ``nyquist_factor`` shortens its own decay there. Continuing the measured
    slope is the least-assumption extension of the data.

    The clamp matters: a material whose absorption falls with frequency has a
    rising slope, and continuing that would amplify a region for which there is
    no data at all. Reflection gains do not grow toward Nyquist.
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
    """Encode an image source set to an ambisonic impulse response.

    ACN channels, N3D by default (spharpy ``"NM"``: a unit plane wave gives
    W = 1), the normalization the package computes in; :func:`synthesize`
    converts once at the end.

    Each image's eight band gains are interpolated onto the FFT grid in
    log-frequency and decibels, giving a smooth magnitude response; the
    propagation delay is applied as a phase ramp, which is exact for a
    fractional delay and so needs no interpolating kernel. Images are
    accumulated in the frequency domain and one inverse transform per channel
    produces the response.

    Outside the eight bands the magnitude is not held flat: above the top band
    it continues that band's slope to Nyquist (see
    :func:`_extrapolated_top_db`), and below the lowest it is tapered to zero
    at DC (see :func:`_radiation_taper`).
    """
    if sh_order < 0:
        raise ValueError("sh_order must be non-negative")
    if n_samples < 1:
        raise ValueError("n_samples must be positive")
    # An empty set is a legitimate input, not an error: the driven-FDN path
    # renders the orders below the driving one, and at max_order=0 there are
    # none. spharpy cannot take an empty coordinate set, so answer first.
    if np.asarray(iss.delay).size == 0:
        return np.zeros((n_samples, (sh_order + 1) ** 2), dtype=float)

    sh = spharpy.spherical.spherical_harmonic_basis_real(
        sh_order, iss.doa, normalization=normalization, channel_convention="ACN"
    )  # (M, n_ch)
    return render_images(iss, sh, n_samples)


def render_images(
    iss: ImageSourceSet, weights: np.ndarray, n_samples: int
) -> np.ndarray:
    """Accumulate image sources onto arbitrary output channels, shape (n, n_ch).

    ``weights`` is ``(M, n_ch)``: row ``m`` says how image ``m`` contributes to
    each output channel. :func:`encode` passes the spherical harmonic basis, so
    the channels are ambisonic. The driven-FDN path passes a routing matrix, so
    the channels are delay lines. Both need identical treatment of the band
    magnitudes, the fractional delay and the radiation taper, and that is the
    reason this is one function rather than two.
    """
    if n_samples < 1:
        raise ValueError("n_samples must be positive")
    weights = np.asarray(weights, dtype=float)
    if weights.ndim != 2 or weights.shape[0] != np.asarray(iss.delay).size:
        raise ValueError(
            f"weights must have shape ({np.asarray(iss.delay).size}, n_channels), "
            f"got {weights.shape}"
        )
    n_ch = weights.shape[1]

    # Drop images arriving after the buffer ends rather than letting the
    # circular transform wrap them back to the start.
    inside = iss.delay * iss.fs < n_samples
    if not np.any(inside):
        return np.zeros((n_samples, n_ch), dtype=float)

    nfft, freq = _fft_grid(n_samples, iss.fs)
    magnitude = _band_magnitude_fn(freq, iss.bands, iss.fs)
    spectrum = np.zeros((freq.size, n_ch), dtype=complex)
    for m in np.flatnonzero(inside):
        spectrum += np.outer(
            magnitude(iss.gain[m]) * np.exp(-2j * np.pi * freq * iss.delay[m]), weights[m]
        )
    return _finish(spectrum, freq, nfft, n_samples)


def _fft_grid(n_samples: int, fs: float) -> tuple[int, np.ndarray]:
    # Headroom so that a kernel centred near t = 0 loses its pre-ringing off
    # the discarded tail instead of aliasing into the response.
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
    # One shared taper, so it factors out of the loop.
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
    """Like :func:`render_images`, for paths with a complex transfer function.

    ``transfers[m](freq)`` multiplies path m's band magnitude ``gains[m]``,
    so diffracted paths share the fractional delay, the band interpolation
    and the radiation taper of the image sources exactly.
    """
    delays = np.asarray(delays, dtype=float)
    weights = np.asarray(weights, dtype=float).reshape(delays.size, -1)
    nfft, freq = _fft_grid(n_samples, fs)
    magnitude = _band_magnitude_fn(freq, bands, fs)
    spectrum = np.zeros((freq.size, weights.shape[1]), dtype=complex)
    for m in np.flatnonzero(delays * fs < n_samples):
        h = transfers[m](freq) * magnitude(gains[m]) * np.exp(-2j * np.pi * freq * delays[m])
        spectrum += np.outer(h, weights[m])
    return _finish(spectrum, freq, nfft, n_samples)


def render_signals(
    signals: list[tuple[int, np.ndarray]],
    gains: np.ndarray,
    bands: np.ndarray,
    fs: float,
    n_samples: int,
) -> np.ndarray:
    """Like :func:`render_images`, for sampled responses already weighted per
    channel. ``signals[m]`` is ``(start_sample, block)`` with block shape
    (length, n_ch). Blocks whose band gains agree to 0.01 dB are summed in the
    time domain and filtered once."""
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


def route_to_directions(iss: ImageSourceSet, directions: pf.Coordinates) -> np.ndarray:
    """One-hot routing of images onto directions, shape (M, n_directions).

    Each image is assigned to the direction closest to its own direction of
    arrival, by cosine similarity. This generalises the mapping of delay lines
    to the six faces of a cube in Wendt et al. (2014, Sec. 1.2) to the
    resolution this package's network already has, since every delay line is
    given a direction anyway to build the ambisonic output matrix.

    One-hot rather than a soft projection deliberately: a soft assignment would
    put every image into every delay line, which spreads each reflection over
    the whole network's delay spread and smears its arrival time. The cost is
    that the routing is lumpy in energy, which is worth revisiting.
    """
    unit_dir = directions.cartesian
    unit_dir = unit_dir / np.linalg.norm(unit_dir, axis=1, keepdims=True)
    if np.asarray(iss.delay).size == 0:
        return np.zeros((0, unit_dir.shape[0]), dtype=float)
    doa = iss.doa.cartesian
    doa = doa / np.linalg.norm(doa, axis=1, keepdims=True)

    nearest = np.argmax(doa @ unit_dir.T, axis=1)
    weights = np.zeros((doa.shape[0], unit_dir.shape[0]), dtype=float)
    weights[np.arange(doa.shape[0]), nearest] = 1.0
    return weights
