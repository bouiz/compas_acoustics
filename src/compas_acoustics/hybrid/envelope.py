"""A ray-traced energy envelope imposed on the network's delay lines.

In a room with unevenly distributed absorption the late field does not decay exponentially from the transition
order, while the network of Wendt et al. (2014) assumes a diffuse field from its input on. In the hybrid approach
for such rooms (Vorlaender 2020, Secs. 11.9 and 16.5.5) ray tracing estimates the envelope M^2(dt, f) and a
synthetic fine structure is shaped to it (Heinz 1993; Schroeder 2011, Sec. 5.3.4). Here the fine structure is the
network's output. Per octave band b, with E the energies of the omnidirectional signals on the package scale,
smoothed with a Hann window of ``SMOOTH_MS``,

    g_b(t) = min(1, sqrt(max(E_ref,b - E_early,b, 0) / E_late,b))

scales every line. A ray is logged only after the image source order, so the reference minus the image sources is
what the lines must carry. The gain is held from where the reference band's energy decay curve is 20 dB down, never
before the early images end (:func:`_early_end`) plus the window. The bank has nine bands, 63 Hz to 16 kHz; the
16 kHz band keeps the energy above about 11.2 kHz out of the 8 kHz estimate and takes the 8 kHz gain.

With a directional histogram (``late_envelope="directional"``) each line is also scaled, per band, by
``sqrt(p_i / q_i)``: its direction's share of the rays' energy over its share of the lines'
(:func:`directional_shares`). The histogram is the detection sphere divided into directivity groups, each with a
histogram per frequency band (Lentz et al. 2007, Sec. 2.4 and Fig. 6; Schroeder 2011, Sec. 5.3.4). Schroeder
averages the bands into one hit probability per group; here each band keeps its own, a per-line, per-band gain as
the boundary filters of Kirsch et al. (2023, Eqs. 4-5) are.

The package's own choices, in none of these sources:

- Shaping the lines of a feedback delay network, by the omnidirectional gains and by the directional shares:
  Schroeder steers a Poisson noise process with one probability per group, and Kirsch's gains are fixed boundary
  filters.
- The late target ``E_ref - E_early``, and gains capped at 1, so a hybrid quieter than its reference stays so.
- The end of the early images (:func:`_early_end`): 13 dB below the reference, both smoothed over 10 ms.
- Pooling each time bin over neighbouring directions with the weights of Kirsch et al. (2023, Eq. 4),
  ``max(u_i . u_j, 0) ** (N / 6)``, which they use to map their 96 scanned directions onto fewer sources; with 96
  sources each takes its own scanned direction (Sec. 1.3).
- ``SMOOTH_MS`` and the holds, 20 dB down for the gains and 60 dB down for the shares.
- The ray count (:func:`histogram_rays`): the 1 dB that Schroeder (2011, Sec. 12.1) takes for a response's
  envelope, applied to one direction's share.
"""

from __future__ import annotations

import warnings
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

import numpy as np
import pyfar as pf
import scipy.fft

SMOOTH_MS = 100.0
# a reference this far from the hybrid over its first 50 ms is on another scale; 6 dB, 50 ms and the onset at 1e-3
# of the peak are the package's choice
SCALE_WARN_DB = 6.0
# the standard deviation of a direction's energy over one slot that sets the ray count of the reference and its
# directional histogram (the package's choice); a reference given with more rays has less
SHARE_ERROR_DB = 1.0

_TOP_BAND_HZ = 16000.0
_N_FILT = 2**12  # reconstructing bank length; its delay is half of it


@lru_cache(maxsize=None)
def _reconstructing_bank(fs: float) -> np.ndarray:
    """The amplitude-preserving octave bank, 63 Hz to 16 kHz, as FIR coefficients (9, _N_FILT)."""
    with warnings.catch_warnings():
        # the second return value (frequencies) is deprecated; both forms are handled
        warnings.filterwarnings("ignore", message="Return parameter 'frequencies'")
        out = pf.dsp.filter.reconstructing_fractional_octave_bands(
            None,
            num_fractions=1,
            frequency_range=(63, _TOP_BAND_HZ),
            n_samples=_N_FILT,
            sampling_rate=fs,
        )
    bank = out[0] if isinstance(out, tuple) else out
    return np.asarray(bank.coefficients, dtype=float)


@lru_cache(maxsize=8)
def _bank_response(fs: float, n_fft: int) -> np.ndarray:
    """The bank's spectrum on an ``n_fft`` grid, (9, n_fft // 2 + 1), shared by the splits of one call."""
    return scipy.fft.rfft(_reconstructing_bank(fs), n_fft, axis=1)


def octave_band_split(x: np.ndarray, fs: float) -> np.ndarray:
    """Octave bands 63 Hz to 16 kHz that sum back to ``x``, shape (9, n[, C]).

    The amplitude-preserving linear-phase bank with its delay removed, as one FFT per channel and one inverse
    per band, zero-padded past ``n + _N_FILT - 1`` so the circular product is the linear convolution.
    """
    x = np.asarray(x, dtype=float)
    columns = x.reshape(x.shape[0], -1)
    bands = np.empty((9, x.shape[0], columns.shape[1]))
    for b, band in enumerate(_octave_bands(columns, fs)):
        bands[b] = band
    return bands[..., 0] if x.ndim == 1 else bands


def _octave_bands(columns: np.ndarray, fs: float):
    """The bands of :func:`octave_band_split` of ``columns`` (n, C), one (n, C) array at a time."""
    n_fft = scipy.fft.next_fast_len(columns.shape[0] + _N_FILT - 1, real=True)
    spectrum = scipy.fft.rfft(columns, n_fft, axis=0, workers=-1)
    response = _bank_response(float(fs), n_fft)
    delay = _N_FILT // 2
    for b in range(response.shape[0]):
        yield scipy.fft.irfft(spectrum * response[b][:, np.newaxis], n_fft, axis=0, workers=-1)[delay : delay + columns.shape[0]]


def _hann(fs: float, ms: float) -> np.ndarray:
    """A Hann window of ``ms`` milliseconds, at least one sample, with unit sum."""
    n = max(round(ms * 1e-3 * fs), 1)
    window = np.hanning(n + 2)[1:-1]
    return window / window.sum()


def _smoothed_energy(bands: np.ndarray, fs: float, ms: float) -> np.ndarray:
    """The energy of ``bands`` (n_bands, n) over a Hann window of ``ms``, same shape."""
    return np.maximum(_smoothed_over_time((bands**2).T, _hann(fs, ms)), 0.0).T


def _smoothed_over_time(x: np.ndarray, window: np.ndarray) -> np.ndarray:
    """``x`` (n, K) convolved column by column with ``window``, centred and cut to ``n`` samples.

    The arithmetic of ``fftconvolve(x, window[:, np.newaxis], mode="same", axes=0)``, its FFT length included, on
    all cores. The columns go in groups of eight, a multiple of the number scipy's transforms vectorise over, so each
    column is rounded as in one call; ``workers=-1`` splits them anywhere and moves the shares by ~1e-12.
    """
    n, width = x.shape[0], window.size
    n_fft = scipy.fft.next_fast_len(n + width - 1, real=True)
    spectrum = scipy.fft.rfft(window[:, np.newaxis], n_fft, axis=0)
    start = (width - 1) // 2

    def smoothed(columns):
        return scipy.fft.irfft(scipy.fft.rfft(x[:, columns], n_fft, axis=0) * spectrum, n_fft, axis=0)[start : start + n]

    with ThreadPoolExecutor() as pool:
        return np.concatenate(list(pool.map(smoothed, [slice(i, i + 8) for i in range(0, x.shape[1], 8)])), axis=1)


def _hold(energy: np.ndarray, db: float, earliest: int, window: int) -> tuple[int, int]:
    """The sample from which a gain is held and the start of the ``window`` before it.

    The hold starts where the energy decay curve of ``energy`` is ``db`` down, never before ``earliest`` and at
    most at the last sample.
    """
    edc = np.cumsum(energy[::-1])[::-1]
    below = np.flatnonzero(edc <= 10 ** (-db / 10) * edc[0])
    last = energy.size - 1
    hold = int(below[0]) if below.size else last
    hold = min(max(hold, earliest), last)
    return hold, max(hold - window, 0)


def _early_end(ref_b: np.ndarray, early_b: np.ndarray, fs: float) -> int:
    """First sample after which the early images stay 13 dB below the reference in every band, over 10 ms."""
    e_ref = _smoothed_energy(ref_b, fs, 10.0)
    e_early = _smoothed_energy(early_b, fs, 10.0)
    present = e_early > 1e-6 * e_early.max(axis=1, keepdims=True)
    above = present & (e_early > 10 ** (-13.0 / 10) * e_ref)
    last = [np.flatnonzero(a)[-1] + 1 if a.any() else 0 for a in above]
    return int(max(last))


def envelope_gains(reference: np.ndarray, early_w: np.ndarray, late_w: np.ndarray, fs: float) -> np.ndarray:
    """Per-band gains in [0, 1] for the late field, shape (9, len(late_w)).

    ``reference`` is an omnidirectional response of the same pair on the package's scale; ``early_w`` and ``late_w``
    are the W channels of the hybrid's two parts. A reference shorter than the late field has its last gain held.
    """
    n = np.asarray(late_w).size
    m = min(np.asarray(reference).size, n)
    ref_b = octave_band_split(np.asarray(reference, dtype=float)[:m], fs)
    early_b = octave_band_split(np.asarray(early_w, dtype=float)[:m], fs)
    late_b = octave_band_split(np.asarray(late_w, dtype=float)[:m], fs)

    e_ref = _smoothed_energy(ref_b, fs, SMOOTH_MS)
    e_early = _smoothed_energy(early_b, fs, SMOOTH_MS)
    e_late = _smoothed_energy(late_b, fs, SMOOTH_MS)
    target = np.maximum(e_ref - e_early, 0.0)

    gains = np.ones_like(e_late)
    active = e_late > 1e-12 * e_late.max(axis=1, keepdims=True)
    gains[active] = np.sqrt(target[active] / e_late[active])
    gains = np.minimum(gains, 1.0)

    window = round(SMOOTH_MS * 1e-3 * fs)
    earliest = _early_end(ref_b[:8], early_b[:8], fs) + window
    for k in range(gains.shape[0]):
        hold, start = _hold(ref_b[k] ** 2, 20.0, earliest, window)
        late_energy = e_late[k, start:hold].sum()
        if late_energy > 0.0:
            held = np.sqrt(target[k, start:hold].sum() / late_energy)
            gains[k, hold:] = min(held, 1.0)
    gains[8] = gains[7]  # above about 11.2 kHz: no estimate of its own

    return np.pad(gains, ((0, 0), (0, n - m)), mode="edge")


def shape_late(signals: np.ndarray, early_w: np.ndarray, late_w: np.ndarray, reference: np.ndarray, fs: float, histogram=None) -> np.ndarray:
    """The network's lines ``signals`` (n, K) with the gains of :func:`envelope_gains` on every column.

    ``late_w`` is the lines' sum through ``C``, ``early_w`` the early field's W channel, and ``reference`` is on the
    same scale; if ``early_w + late_w`` and the reference differ by more than ``SCALE_WARN_DB`` over the first 50 ms,
    a warning says the reference is on another scale. ``histogram``, a
    :class:`~compas_acoustics.room.raytracing.DirectionalHistogram` on the lines' directions, also gives every band
    of the lines its :func:`directional_shares`. Its bands are the octaves from 62.5 Hz (the bank's to within 2 %),
    at least to 8 kHz; without a 16 kHz row that band takes the 8 kHz one.
    """
    signals = np.asarray(signals, dtype=float)
    if signals.ndim != 2:
        raise ValueError("signals must be the network's lines, (n, n_lines)")
    if histogram is not None:
        rows = np.asarray(histogram.bands, dtype=float)
        if not 8 <= rows.size <= 9 or not np.allclose(rows, 62.5 * 2.0 ** np.arange(rows.size), rtol=0.02):
            raise ValueError(f"the histogram's bands must be the octaves from 62.5 Hz to 8 or 16 kHz, got {rows.tolist()}")
    early_w = np.asarray(early_w, dtype=float).ravel()
    late_w = np.asarray(late_w, dtype=float).ravel()
    reference = np.asarray(reference, dtype=float).ravel()

    hybrid_w = early_w + late_w
    loud = np.flatnonzero(np.abs(hybrid_w) > 1e-3 * np.abs(hybrid_w).max())
    if loud.size:
        a = loud[0]
        b = a + int(0.05 * fs)
        e_h = np.sum(hybrid_w[a:b] ** 2)
        e_r = np.sum(reference[a:b] ** 2)
        if e_r <= 0.0 or abs(10 * np.log10(e_r / e_h)) > SCALE_WARN_DB:
            warnings.warn(
                "the reference differs from the hybrid by more than "
                f"{SCALE_WARN_DB:.0f} dB over the first 50 ms; it must be on the "
                "package's scale (free-field direct sound 1 / (4 pi r)), as "
                "Mono returns it",
                UserWarning,
                stacklevel=2,
            )

    gains = envelope_gains(reference, early_w, late_w, fs)
    if histogram is None:
        return np.einsum("bn,bnc->nc", gains, octave_band_split(signals, fs))
    shaped = np.zeros_like(signals)
    for b, band in enumerate(_octave_bands(signals, fs)):
        share = directional_shares(histogram, band, fs, min(b, rows.size - 1))
        share *= gains[b][:, np.newaxis]
        share *= band
        shaped += share
    return shaped


def _pooling_weights(directions: np.ndarray) -> np.ndarray:
    """``max(u_i . u_j, 0) ** (N / 6)`` (Kirsch et al. 2023, Eq. 4) with rows summing to one, (N, N)."""
    unit = np.asarray(directions, dtype=float)
    unit = unit / np.linalg.norm(unit, axis=1, keepdims=True)
    weights = np.clip(unit @ unit.T, 0.0, None) ** (unit.shape[0] / 6.0)
    return weights / weights.sum(axis=1, keepdims=True)


def smooth_over_directions(energy: np.ndarray, directions: np.ndarray) -> np.ndarray:
    """``energy`` (n_directions, ...) pooled over neighbouring directions, same shape.

    Direction ``i`` takes the mean of all directions weighted by ``max(u_i . u_j, 0) ** (N / 6)``, the weights
    of Kirsch et al. (2023, Eq. 4); pooling a grid onto itself with them is the package's choice. The rows of
    the kernel are normalised, so on a grid whose columns sum alike (an equal-area grid) the energy of every bin
    is kept to within the grid's unevenness.
    """
    energy = np.asarray(energy, dtype=float)
    weights = _pooling_weights(directions)
    if weights.shape[0] != energy.shape[0]:
        raise ValueError(f"energy has {energy.shape[0]} directions, directions {weights.shape[0]}")
    return np.tensordot(weights, energy, axes=(1, 0))


def histogram_rays(volume: float, receiver_radius: float, speed_of_sound: float, directions: np.ndarray) -> int:
    """The rays at which a direction's energy over one slot has a standard deviation of ``SHARE_ERROR_DB``.

    Vorlaender (2020, Sec. 11.3.7): the hits of a detection sphere of radius r_d within dt are Poisson
    distributed, N pi r_d^2 c dt / V of them on average, so the level of their energy has the standard deviation
    4.34 / sqrt(hits) dB (Eq. 11.20). One of n directions holds 1 / n of the hits. The Hann window of
    ``SMOOTH_MS`` and :func:`smooth_over_directions` weight them, and a weighted sum of Poisson counts has the
    error of ``sum(w) ** 2 / sum(w ** 2)`` equal ones: 2/3 of the window, and the mean of that ratio over the
    kernel's rows. The methods trace their own reference with this count in both late-envelope modes, so that
    ``"omni"`` and ``"directional"`` shape the late field to one reference.
    """
    weights = _pooling_weights(directions)
    pooled = float(np.mean(1.0 / np.sum(weights**2, axis=1)))
    slot = SMOOTH_MS * 1e-3 * 2.0 / 3.0
    hits = (10.0 / np.log(10.0) / SHARE_ERROR_DB) ** 2
    return int(np.ceil(hits * volume * weights.shape[0] / (np.pi * receiver_radius**2 * speed_of_sound * slot * pooled)))


def directional_shares(histogram, lines: np.ndarray, fs: float, band: int) -> np.ndarray:
    """The factor on every line, (n, n_lines), that gives one band of the lines the rays' distribution over directions.

    ``histogram`` is a :class:`~compas_acoustics.room.raytracing.DirectionalHistogram` on the lines' directions;
    ``lines`` (n, n_lines) are one octave band of the network's lines as heard and ``band`` the histogram's row of
    that band; both start at the emission. Per time slot ``p_i`` is direction ``i``'s share of the rays' energy in
    the band and ``q_i`` the line's share of the lines', both after :func:`smooth_over_directions` and over
    ``SMOOTH_MS``. The shares are held once the band's rays have decayed 60 dB (detections per bin do not thin
    out with the energy, as a reference's late part does), never before the lines start.
    """
    lines = np.asarray(lines, dtype=float)
    if lines.ndim != 2:
        raise ValueError("lines must be (n, n_lines)")
    n, n_lines = lines.shape
    energy = np.asarray(histogram.energy, dtype=float)
    if energy.ndim != 3 or energy.shape[0] != n_lines:
        raise ValueError(f"the histogram must hold one direction per line ({n_lines}), got {energy.shape}")
    bin_samples = max(int(round(histogram.bin_size * fs)), 1)
    hann = _hann(fs, SMOOTH_MS)
    window = hann.size

    rays = smooth_over_directions(energy[:, band], histogram.directions)  # (n_lines, n_bins)
    m = min(rays.shape[1] * bin_samples, n)
    rays = np.repeat(rays, bin_samples, axis=1)[:, :m].T  # (m, n_lines)
    e_rays = np.maximum(_smoothed_over_time(rays, hann), 0.0)
    e_lines = np.maximum(_smoothed_over_time(lines[:m] ** 2, hann), 0.0)
    e_lines = smooth_over_directions(e_lines.T, histogram.directions).T
    p = e_rays / np.where(e_rays.sum(axis=1, keepdims=True) > 0.0, e_rays.sum(axis=1, keepdims=True), 1.0)
    q = e_lines / np.where(e_lines.sum(axis=1, keepdims=True) > 0.0, e_lines.sum(axis=1, keepdims=True), 1.0)

    shares = np.ones((m, n_lines))
    active = (q > 1e-12 * q.max()) & (p.sum(axis=1, keepdims=True) > 0.0)
    shares[active] = np.sqrt(p[active] / q[active])

    total = e_lines.sum(axis=1)
    present = np.flatnonzero(total > 1e-6 * total.max())
    earliest = (int(present[0]) if present.size else 0) + window
    hold, start = _hold(rays.sum(axis=1), 60.0, earliest, window)
    held_rays = e_rays[start:hold].sum(axis=0)
    held_lines = e_lines[start:hold].sum(axis=0)
    if held_rays.sum() > 0.0 and held_lines.sum() > 0.0:
        held = np.sqrt((held_rays / held_rays.sum()) / np.where(held_lines > 0.0, held_lines / held_lines.sum(), 1.0))
        shares[hold:] = np.where(held_lines > 0.0, held, 1.0)

    return np.pad(shares, ((0, n - m), (0, 0)), mode="edge")
