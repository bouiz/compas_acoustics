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
structure is the network's own output, which is dense (normalised echo
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
No changes.
"""

from __future__ import annotations

import warnings

import numpy as np
import pyfar as pf
from scipy.signal import fftconvolve

SMOOTH_MS = 100.0
HOLD_DB = -20.0
# the early images have ended once they stay this far below the reference
EARLY_END_DB = -13.0
# a reference this far from the hybrid over its first 50 ms is on another scale
SCALE_WARN_DB = 6.0

_TOP_BAND_HZ = 16000.0
_N_FILT = 2**12  # reconstructing bank length; its delay is half of it


def octave_band_split(x: np.ndarray, fs: float) -> np.ndarray:
    """Octave bands 63 Hz to 16 kHz that sum back to ``x``, shape (9, n[, C]).

    pyfar's amplitude-preserving bank, linear phase, with its delay removed.
    """
    x = np.asarray(x, dtype=float)
    columns = x.reshape(x.shape[0], -1)
    padded = np.concatenate([columns.T, np.zeros((columns.shape[1], _N_FILT))], axis=1)
    with warnings.catch_warnings():
        # the second return value (frequencies) is deprecated; both forms are handled
        warnings.filterwarnings("ignore", message="Return parameter 'frequencies'")
        out = pf.dsp.filter.reconstructing_fractional_octave_bands(
            pf.Signal(padded, fs),
            num_fractions=1,
            frequency_range=(63, _TOP_BAND_HZ),
            n_samples=_N_FILT,
        )
    filtered = out[0] if isinstance(out, tuple) else out
    delay = _N_FILT // 2
    bands = np.transpose(filtered.time[:, :, delay:delay + x.shape[0]], (0, 2, 1))
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
    late: np.ndarray, early: np.ndarray, reference: np.ndarray, fs: float
) -> np.ndarray:
    """The late field with the reference's envelope imposed, same shape as ``late``.

    ``late`` and ``early`` are the hybrid's parts, (n, C) with W first;
    ``reference`` is (n,) on the same scale. Warns if the two disagree by more
    than ``SCALE_WARN_DB`` over the first 50 ms, which means the reference is
    on another scale (normalised, or with a different source level).
    """
    late = np.asarray(late, dtype=float)
    early = np.asarray(early, dtype=float)
    reference = np.asarray(reference, dtype=float).ravel()
    late_w = late.reshape(late.shape[0], -1)[:, 0]
    early_w = early.reshape(early.shape[0], -1)[:, 0]

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
    bands = octave_band_split(late, fs)
    if late.ndim == 1:
        return np.einsum("bn,bn->n", gains, bands)
    return np.einsum("bn,bnc->nc", gains, bands)
