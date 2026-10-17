"""Direction-dependent boundary absorption on the delay line outputs.

As published by Wendt, van de Par and Ewert (JAES 62(11), 2014, Sec. 1.2): "by
the channel-wise application of reflection filters H_j^r, the signal of each
incidence direction is weighted with the reflection coefficient of the room
wall corresponding to the respective parallel cube side face". On by default in
:func:`hybrid_rir.synthesize`.

The implementation follows Kirsch, Wendt and van de Par, "Computationally-Efficient Simulation of
Late Reverberation for Inhomogeneous Boundary Conditions and Coupled Rooms",
JAES 71(4), 186-201 (2023), which extends that method so that a room with
unevenly distributed absorption radiates its late field anisotropically. In
their words, the simulation was "extended by a spatial sampling routine to calculate the absorption
filters applied at the output of each delay line of the FDN that maps to the
VRS", one virtual reverberation source per line.

Their "Generalized Approach for Arbitrary Geometry" is the one implemented
here, because it is the one that holds for a mesh: the absorption around the
receiver is sampled by casting rays at the boundary and taking the absorption
of the nearest face hit, and each delay line then takes a weighted mean of
those samples,

    w_s,k = max(v_s . v_k, 0) ** beta,   beta = K / 6,
    a_k   = sum_s w_s,k alpha_s / sum_s w_s,k,

with ``v_s`` the scanned directions, ``v_k`` the line's direction and ``K`` the
number of lines (their Sec. 1.3, Eqs. 4-5; ``beta = K / 6`` gives a -3 dB lobe
width of 90 degrees at six lines). The line's output filter then has the
magnitude of that direction's reflection, ``sqrt(1 - a_k)`` per band.

Two consequences worth stating plainly, both inherited from the paper:

- the decay is not touched. The filter sits at the output, outside the loop, so
  every line still decays at ``target.rt``. What changes is how much energy, and
  with what colour, arrives from each direction.
- the late field loses ``10 log10(mean_k (1 - a_k))`` dB overall, one
  reflection's worth of absorption. In Wendt et al. (2014) the decay also
  comes entirely from the target (Sabine's T60 in the absorption filters, Eqs. 2-3),
  so the loss sits on top of the target there too, by design. It is left in,
  as in both papers, and measured in
  ``examples/04_directional_absorption.ipynb`` instead of being compensated.

The anisotropy it creates has not been validated against a directional
reference; every reference in this repository is monaural.

Ported from hybrid-rir (gpapadimitri/hybrid-rir @ 8a70661, src/directional.py).
"""

from __future__ import annotations

import numpy as np
import pyfar as pf
import pyFDN
import spharpy

from .bands import N_BANDS
from .pair import Pair
from .scattering import _first_hit

SCAN_DIRECTIONS = 96  # the paper's highest VRS resolution, used as its scan grid


def scan_directions(n_scan: int = SCAN_DIRECTIONS) -> pf.Coordinates:
    """The directions the boundary is sampled along, shape (n_scan,)."""
    directions = spharpy.samplings.equal_area(n_max=0, n_points=n_scan)
    if directions.cshape[0] != n_scan:
        raise ValueError(
            f"equal_area returned {directions.cshape[0]} directions for "
            f"n_scan={n_scan}; pass a count it can produce exactly"
        )
    return directions


def scanned_absorption(
    spec: Pair, directions: pf.Coordinates, *, n_scan: int = SCAN_DIRECTIONS
) -> np.ndarray:
    """Absorption coefficient per delay line and band, shape (n_lines, 8).

    Kirsch et al. (2023), Sec. 1.3: rays from the receiver take the absorption
    of the first boundary face they meet, and each line averages them with the
    weights above. A concave room needs the *first* face, which is why this
    casts rays rather than projecting onto face normals.
    """
    faces = spec.geometry.faces_with_materials()
    scan = scan_directions(n_scan).cartesian
    scan = scan / np.linalg.norm(scan, axis=1, keepdims=True)

    hit = _first_hit(faces, spec.receiver, scan)
    if np.any(hit < 0):
        raise RuntimeError(
            f"{int(np.sum(hit < 0))} of {hit.size} scanned directions meet no "
            "boundary; the room is not closed around the receiver"
        )
    alpha = np.stack([spec.geometry.materials[name].absorption for _, name in faces])
    alpha = alpha[hit][:, :N_BANDS]  # (n_scan, 8)

    unit = directions.cartesian
    unit = unit / np.linalg.norm(unit, axis=1, keepdims=True)
    if spec.receiver_rotation is not None:
        unit = unit @ spec.receiver_rotation.T  # the scan is in room coordinates
    beta = unit.shape[0] / 6.0
    weights = np.clip(scan @ unit.T, 0.0, None) ** beta  # (n_scan, n_lines)
    return (weights.T @ alpha) / weights.sum(axis=0)[:, np.newaxis]


def absorption_filters(absorption: np.ndarray, fs: float) -> np.ndarray:
    """Output filters of magnitude ``sqrt(1 - a)``, shape (n_sections, 6, n_lines).

    One GEQ per delay line, designed on the package band grid with its two
    extrapolated edges held at the outermost bands, in the layout
    :class:`pyFDN.td.SOSBank` expects.
    """
    absorption = np.asarray(absorption, dtype=float)
    if absorption.ndim != 2 or absorption.shape[1] != N_BANDS:
        raise ValueError(f"absorption must be (n_lines, {N_BANDS})")
    if np.any(absorption < 0.0) or np.any(absorption >= 1.0):
        raise ValueError("absorption coefficients must lie in [0, 1)")

    db = 10.0 * np.log10(1.0 - absorption)  # pressure, 20 log10 sqrt(1 - a)
    return pyFDN.gain_to_geq(np.hstack((db[:, :1], db, db[:, -1:])).T, float(fs))


def directional_absorption_filters(
    spec: Pair, directions: pf.Coordinates, *, n_scan: int = SCAN_DIRECTIONS
) -> np.ndarray:
    """The per-line output filters of this room, ready for :func:`render`."""
    return absorption_filters(
        scanned_absorption(spec, directions, n_scan=n_scan), spec.fs
    )
