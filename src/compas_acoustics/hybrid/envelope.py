"""A ray-traced energy envelope imposed on the network output.

The late field of a room whose absorption is unevenly distributed does not
decay exponentially from the transition order on: the part of the field that
reaches the absorbing surfaces dies early. The network of Wendt et al. (2014)
assumes the opposite,
a diffuse field from its input onwards, and so starts at the full energy of
the last image order and decays at the target rate from there. In the AVARIG
offices that leaves the late field 5 dB over Odeon at 200-800 ms (median of
60 pairs) and the EDT at twice Odeon's.

The remedy here is the hybrid approach Vorlaender describes for such
"non-Sabinean" rooms (Auralization, 2nd ed., 2020, Sec. 11.9 and 16.5.5):
ray tracing estimates the late impulse response envelope M^2(dt, f), and a
synthetic fine structure is shaped to it ("noise shaping", Heinz 1993;
Schroeder 2011), with M^2(dt) = H_j^2 n_j (his Eq. 11.52). Here the fine
structure is the network's own output, which is dense (normalized echo
density about 1) and carries the directions, and only its per-band energy
over time is changed. Using the network as the carrier is this package's
choice; no precedent for it was found.

Per octave band b, with E the smoothed energies of the omnidirectional (W)
signals and the reference on the package's scale,

    g_b(t) = min(1, sqrt(max(E_ref,b - E_early,b, 0) / E_late,b)).

pyroomacoustics logs a specular ray only after ``ism_order`` reflections
(``libroom_src/room.cpp``), so its reference minus the image sources is the
energy the image sources do not carry. The same gain goes on every channel.

The constants below are design choices, measured on 60 source-receiver pairs
in six furnished offices against Odeon, not taken from a reference:

- ``SMOOTH_MS``: the reference envelope fluctuates by 6-10 dB at the 10 ms
  Vorlaender cites (a few strong late ray arrivals; more rays did not reduce
  it), and following it puts audible swells into the tail. 100 ms leaves 3-4 dB
  while still following the correction, which builds up over 100-300 ms.
- Attenuate only: allowing the gain above 1 brightened the onset of the tail
  by up to 13 dB at 8 kHz in some pairs, an audible step right after the early
  reflections, and up to 20 dB in every band where the image sources are
  empty. Without boosts nothing was lost in accuracy. The price: a hybrid that
  is quieter than its reference stays so (a diffuse shoebox, 4 dB).
- ``HOLD_DB``: once the reference band has decayed 20 dB its gain is frozen
  and the network decays at its own target rate, at the corrected level. The
  held value is a ratio of energies over the preceding ``SMOOTH_MS``, and the
  hold is never earlier than the end of the early images plus that window, so
  it always comes from late-field energy. Holding later, or on the late field's
  own decay, brought the swells back; holding at -10 dB of the late field gave
  outliers of 10 dB.
- Nine bands, 63 Hz to 16 kHz: a reconstructing bank puts everything above
  its top band's lower edge into that band, and the reference has energy up to
  Nyquist that the hybrid, extrapolating above 8 kHz, does not. With an
  8 kHz top band that mismatch read as a 15 dB boost. The 16 kHz band takes the
  8 kHz gain.

Ported from hybrid-rir (gpapadimitri/hybrid-rir @ a272da7, src/envelope.py).
``shape_late`` takes the W channels apart from the signals it shapes, so that
it shapes the delay lines, from which both outputs are rendered; the gains are
upstream's.

The directional envelope
------------------------
:func:`shape_late_directional` adds RAVEN's directional histogram (Lentz,
Schroeder and Vorlaender 2007, Sec. 2.2: the detection sphere divided into
"directivity groups", each ray's energy added to the group of its time and
direction of arrival; Schroeder 2011, Sec. 5.3.4), which pyroomacoustics keeps
per receiver direction (:func:`compas_acoustics.room.raytracing.directional_histograms`).
RAVEN shapes the level of each time slot with the omni energy spectral density
and uses the directional histogram only as the distribution of directions the
impulses of that slot are allotted to, one distribution for all bands. Here
the level is the omni envelope above, unchanged, and every delay line is then
scaled by ``sqrt(p_i / q_i)``, its direction's share of the rays' energy over
its share of the lines' energy (:func:`directional_shares`), so the lines take
the rays' distribution and keep their total: a direction facing an absorbing
wall decays faster, as the rays say it does, which one gain on every line
cannot express. Shaping an FDN carrier this way has no precedent, as the omni
envelope has none.

Before that the bins are pooled over neighbouring directions with the weights
of Kirsch, Wendt and van de Par (2023, Eq. 4), ``max(u_i . u_j, 0) ** (N /
6)``, the kernel the directional absorption uses: at 96 lines a bin of 4 ms
holds a handful of detections per direction (RAVEN's rule of thumb, ``N = V /
((sigma / 4.34) ** 2 pi r ** 2 c dt)`` particles for sigma dB per bin, asks
for about a hundred times the rays of the omni histogram at the same
precision), and the kernel pools roughly the nearest ring.
"""

from __future__ import annotations

import warnings
from functools import lru_cache

import numpy as np
import pyfar as pf
import scipy.fft
from scipy.signal import fftconvolve

SMOOTH_MS = 100.0
HOLD_DB = -20.0
# the directional shares follow the rays until their energy has decayed this far: RAVEN traces its
# particles until they have lost 63 dB (itaRavenProject, energyLoss_DetectionSphere = 63), and a ray
# tracer's detections per bin do not thin out as the energy decays, as a reference response's late part does
SHARES_HOLD_DB = -60.0
# the early images have ended once they stay this far below the reference
EARLY_END_DB = -13.0
# a reference this far from the hybrid over its first 50 ms is on another scale
SCALE_WARN_DB = 6.0

_TOP_BAND_HZ = 16000.0
_N_FILT = 2**12  # reconstructing bank length; its delay is half of it


@lru_cache(maxsize=None)
def _reconstructing_bank(fs: float) -> np.ndarray:
    """pyfar's amplitude-preserving octave bank, 63 Hz to 16 kHz, as FIR coefficients (9, _N_FILT)."""
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
    """The bank's spectrum on an ``n_fft`` grid, (9, n_fft // 2 + 1); the four splits of one run share it."""
    return scipy.fft.rfft(_reconstructing_bank(fs), n_fft, axis=1)


def octave_band_split(x: np.ndarray, fs: float) -> np.ndarray:
    """Octave bands 63 Hz to 16 kHz that sum back to ``x``, shape (9, n[, C]).

    pyfar's amplitude-preserving bank, linear phase, with its delay removed.
    The bank is applied as one FFT per channel and one inverse FFT per band,
    zero-padded past ``n + _N_FILT - 1`` so the circular product is the linear
    convolution pyfar computes.
    """
    x = np.asarray(x, dtype=float)
    columns = x.reshape(x.shape[0], -1)
    n_fft = scipy.fft.next_fast_len(columns.shape[0] + _N_FILT - 1, real=True)
    spectrum = scipy.fft.rfft(columns, n_fft, axis=0, workers=-1)
    response = _bank_response(float(fs), n_fft)
    delay = _N_FILT // 2
    bands = np.empty((response.shape[0], x.shape[0], columns.shape[1]))
    for b in range(response.shape[0]):
        bands[b] = scipy.fft.irfft(spectrum * response[b][:, np.newaxis], n_fft, axis=0, workers=-1)[delay:delay + x.shape[0]]
    return bands[..., 0] if x.ndim == 1 else bands


def _smoothed_energy(bands: np.ndarray, fs: float, ms: float) -> np.ndarray:
    n = max(round(ms * 1e-3 * fs), 1)
    window = np.hanning(n + 2)[1:-1]
    window /= window.sum()
    return np.maximum(fftconvolve(bands**2, window[np.newaxis, :], mode="same", axes=1), 0.0)


def _early_end(ref_b: np.ndarray, early_b: np.ndarray, fs: float) -> int:
    """First sample after which the early images stay below the reference."""
    e_ref = _smoothed_energy(ref_b, fs, 10.0)
    e_early = _smoothed_energy(early_b, fs, 10.0)
    present = e_early > 1e-6 * e_early.max(axis=1, keepdims=True)
    above = present & (e_early > 10 ** (EARLY_END_DB / 10) * e_ref)
    last = [np.flatnonzero(a)[-1] + 1 if a.any() else 0 for a in above]
    return int(max(last))


def envelope_gains(
    reference: np.ndarray, early_w: np.ndarray, late_w: np.ndarray, fs: float
) -> np.ndarray:
    """Per-band gains in [0, 1] for the late field, shape (9, len(late_w)).

    ``reference`` is an omnidirectional response of the same source and
    receiver on the package's scale; ``early_w`` and ``late_w`` are the W
    channels of the hybrid's two parts. A reference shorter than the late
    field has its last gain held.
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
        edc = np.cumsum((ref_b[k] ** 2)[::-1])[::-1]
        below = np.flatnonzero(edc <= 10 ** (HOLD_DB / 10) * edc[0])
        hold = int(below[0]) if below.size else m - 1
        hold = min(max(hold, earliest), m - 1)
        start = max(hold - window, 0)
        late_energy = e_late[k, start:hold].sum()
        if late_energy > 0.0:
            held = np.sqrt(target[k, start:hold].sum() / late_energy)
            gains[k, hold:] = min(held, 1.0)
    gains[8] = gains[7]  # above 11.3 kHz: no estimate of its own

    if m < n:
        gains = np.concatenate([gains, np.repeat(gains[:, -1:], n - m, axis=1)], axis=1)
    return gains


def shape_late(
    signals: np.ndarray, early_w: np.ndarray, late_w: np.ndarray, reference: np.ndarray, fs: float
) -> np.ndarray:
    """``signals`` with the reference's envelope imposed, same shape as ``signals``.

    ``signals`` are the late field's parts, (n,) or (n, K): its ambisonic
    channels, or the network's lines, whose sum through ``C`` is ``late_w``.
    ``early_w`` and ``late_w`` are the W channels of the hybrid's two parts,
    (n,); ``reference`` is (n,) on the same scale. The gains of
    :func:`envelope_gains` go on every column alike. Warns if ``early_w +
    late_w`` and the reference disagree by more than ``SCALE_WARN_DB`` over the
    first 50 ms, which means the reference is on another scale (normalized, or
    with a different source level).
    """
    signals = np.asarray(signals, dtype=float)
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
                "RoomSpec.ray_traced_rir returns it",
                UserWarning,
                stacklevel=2,
            )

    gains = envelope_gains(reference, early_w, late_w, fs)
    bands = octave_band_split(signals, fs)
    if signals.ndim == 1:
        return np.einsum("bn,bn->n", gains, bands)
    return np.einsum("bn,bnc->nc", gains, bands)


# ------------------------------------------------------ the directional envelope --
def smooth_over_directions(energy: np.ndarray, directions: np.ndarray) -> np.ndarray:
    """``energy`` (n_directions, ...) pooled over neighbouring directions, same shape.

    Kirsch et al. (2023, Eq. 4): direction ``i`` takes the mean of all
    directions weighted by ``max(u_i . u_j, 0) ** (N / 6)``. The rows of the
    kernel are normalised, so on a grid whose columns sum alike (an equal-area
    grid) the energy of every bin is kept to within the grid's unevenness.
    """
    energy = np.asarray(energy, dtype=float)
    unit = np.asarray(directions, dtype=float)
    unit = unit / np.linalg.norm(unit, axis=1, keepdims=True)
    if unit.shape[0] != energy.shape[0]:
        raise ValueError(f"energy has {energy.shape[0]} directions, directions {unit.shape[0]}")
    weights = np.clip(unit @ unit.T, 0.0, None) ** (unit.shape[0] / 6.0)
    weights /= weights.sum(axis=1, keepdims=True)
    return np.tensordot(weights, energy, axes=(1, 0))




def directional_shares(histogram, lines: np.ndarray, fs: float) -> np.ndarray:
    """The factor on every delay line that gives the lines the rays' distribution over directions, (n, n_lines).

    ``histogram`` is a :class:`~compas_acoustics.room.raytracing.DirectionalHistogram`
    whose directions are the lines', in the same order; ``lines`` are the
    network's lines as heard, (n, n_lines); both start at the emission. Per
    time slot, ``p_i`` is direction ``i``'s share of the rays' energy summed
    over the bands (RAVEN's "ray probability for each directivity group", one
    distribution for all bands) and ``q_i`` the line's share of the lines'
    energy, both after :func:`smooth_over_directions`; the factor is
    ``sqrt(p_i / q_i)``, both smoothed over ``SMOOTH_MS``, so the lines' total
    energy is kept and only its distribution over directions changes. The
    shares are held once the rays' energy has decayed ``SHARES_HOLD_DB``,
    RAVEN's particle cutoff, and never before the lines have started.
    """
    lines = np.asarray(lines, dtype=float)
    if lines.ndim != 2:
        raise ValueError("lines must be (n, n_lines)")
    n, n_lines = lines.shape
    energy = np.asarray(histogram.energy, dtype=float)
    if energy.ndim != 3 or energy.shape[0] != n_lines:
        raise ValueError(f"the histogram must hold one direction per line ({n_lines}), got {energy.shape}")
    bin_samples = max(int(round(histogram.bin_size * fs)), 1)
    window = max(round(SMOOTH_MS * 1e-3 * fs), 1)
    hann = np.hanning(window + 2)[1:-1]
    hann /= hann.sum()

    rays = smooth_over_directions(energy.sum(axis=1), histogram.directions)  # (n_lines, n_bins)
    m = min(rays.shape[1] * bin_samples, n)
    rays = np.repeat(rays, bin_samples, axis=1)[:, :m].T  # (m, n_lines)
    e_rays = np.maximum(fftconvolve(rays, hann[:, np.newaxis], mode="same", axes=0), 0.0)
    e_lines = np.maximum(fftconvolve(lines[:m] ** 2, hann[:, np.newaxis], mode="same", axes=0), 0.0)
    e_lines = smooth_over_directions(e_lines.T, histogram.directions).T
    p = e_rays / np.where(e_rays.sum(axis=1, keepdims=True) > 0.0, e_rays.sum(axis=1, keepdims=True), 1.0)
    q = e_lines / np.where(e_lines.sum(axis=1, keepdims=True) > 0.0, e_lines.sum(axis=1, keepdims=True), 1.0)

    shares = np.ones((m, n_lines))
    active = (q > 1e-12 * q.max()) & (p.sum(axis=1, keepdims=True) > 0.0)
    shares[active] = np.sqrt(p[active] / q[active])

    total = e_lines.sum(axis=1)
    present = np.flatnonzero(total > 1e-6 * total.max())
    earliest = (int(present[0]) if present.size else 0) + window
    edc = np.cumsum(rays.sum(axis=1)[::-1])[::-1]
    below = np.flatnonzero(edc <= 10 ** (SHARES_HOLD_DB / 10) * edc[0])
    hold = int(below[0]) if below.size else m - 1
    hold = min(max(hold, earliest), m - 1)
    start = max(hold - window, 0)
    held_rays = e_rays[start:hold].sum(axis=0)
    held_lines = e_lines[start:hold].sum(axis=0)
    if held_rays.sum() > 0.0 and held_lines.sum() > 0.0:
        held = np.sqrt((held_rays / held_rays.sum()) / np.where(held_lines > 0.0, held_lines / held_lines.sum(), 1.0))
        shares[hold:] = np.where(held_lines > 0.0, held, 1.0)

    if m < n:
        shares = np.concatenate([shares, np.repeat(shares[-1:], n - m, axis=0)])
    return shares


def shape_late_directional(
    lines: np.ndarray, early_w: np.ndarray, late_w: np.ndarray, reference: np.ndarray, histogram, fs: float
) -> np.ndarray:
    """``lines`` (n, n_lines) shaped to the reference's envelope and to the rays' distribution over directions.

    :func:`shape_late` with the omni reference, then every line times its
    :func:`directional_shares`. RAVEN's two uses of its histogram: the energy
    envelope per band, and the directions the impulses of each time slot
    come from.
    """
    return shape_late(lines, early_w, late_w, reference, fs) * directional_shares(histogram, lines, fs)
