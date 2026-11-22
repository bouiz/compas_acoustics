"""The directional feedback delay network: one delay line per direction."""

from __future__ import annotations

from collections.abc import Sequence
from functools import cache
from math import gcd

import numpy as np
import pyfar as pf
import pyFDN
import spharpy
from pyFDN.td import SOSBank
from scipy.signal._sosfilt import _sosfilt

from .pair import Pair
from .target import RoomTarget


class _SOSBank(SOSBank):
    """pyFDN's cascade per line with the kernel of ``scipy.signal.sosfilt`` called directly, on the same state.

    The network filters every line once per block, at most its shortest delay long, where ``sosfilt``'s checks of
    its arguments take longer than the filtering; the output is bit-identical.
    """

    def process_block(self, block):
        out = np.array(np.asarray(block, dtype=float).T, order="C")
        for i in range(self.in_channels):
            _sosfilt(self.sos[i], out[i : i + 1], self._state[i : i + 1])
        return out.T


#: Fewest delay lines at any ambisonic order: Kirsch et al. (2023, Secs. 1.2 and 3.3) use 96 for offline simulation
#: whatever the rendered directions, so the tail's character does not change with the count and a small network's
#: recurrent patterns are avoided. The 96-point equal-area grid supports orders 0 to 3.
MIN_DELAYS = 96


@cache
def fdn_directions(num_delays: int = MIN_DELAYS) -> pf.Coordinates:
    """The direction each delay line radiates from, an equal-area grid of ``num_delays`` points.

    The one grid the image sources are routed onto, :func:`directional_delays` reads the room along, the
    directional histogram bins the rays on, and the renderers project the lines from.
    """
    return spharpy.samplings.equal_area(n_max=0, n_points=num_delays)


def _nearest_coprime(targets: np.ndarray) -> np.ndarray:
    """Distinct integers near their targets, pairwise coprime where that fits.

    Coprimality is sought within 3 % of each target (the package's choice), else the nearest unused integer is
    taken, so no length strays far from its target.
    """
    chosen: list[int] = []
    out = np.empty(targets.size, dtype=int)
    for idx in np.argsort(targets)[::-1]:
        centre = max(int(round(targets[idx])), 2)
        reach = max(2, int(0.03 * centre))
        offsets = [0] + [s * k for k in range(1, reach + 1) for s in (1, -1)]
        candidates = [centre + o for o in offsets if centre + o >= 2]
        pick = next(
            (c for c in candidates if all(gcd(c, other) == 1 for other in chosen)),
            None,
        )
        if pick is None:
            pick = next(centre + s * k for k in range(centre) for s in (1, -1) if centre + s * k >= 2 and centre + s * k not in chosen)
        chosen.append(pick)
        out[idx] = pick
    return out


def directional_delays(
    pair: Pair,
    directions: pf.Coordinates,
    *,
    rng: int | np.random.Generator = 0,
) -> np.ndarray:
    """Delay line lengths in samples, each the room's free path along its direction.

    The room-dimension delays of Kirsch et al. (2021, Sec. VI.B; 2023, Sec. 3.3), taken per direction and for any
    closed geometry: a uniform field along ``u`` meets the boundary at the rate ``c sum_f A_f |n_f . u| / (2 V)``,
    so line ``i`` is ``2 V / sum_f A_f |n_f . u_i|`` long, in a shoebox
    ``1 / (|u_x| / L_x + |u_y| / L_y + |u_z| / L_z)``. The rates average over the sphere to ``c S / (4 V)``, the
    mean free path ``4 V / S``.

    ``u`` and ``-u`` share a free path, so each length is jittered uniformly by up to 0.1 of it, drawn from
    ``rng``, before the lengths are made coprime (:func:`_nearest_coprime`), lest opposite lines ring together. The
    jitter and the coprime lengths are the package's choice; Wendt et al. (2014, Eq. 1) add up to 0.1 of the mean
    room dimension. ``directions`` are in the receiver's frame.
    """
    unit = directions.cartesian
    unit = unit / np.linalg.norm(unit, axis=1, keepdims=True)
    unit = unit @ pair.receiver_rotation.T
    projected = np.abs(unit @ pair.geometry.area_normals().T).sum(axis=1)
    free_path_samples = 2.0 * pair.geometry.volume / projected / pair.speed_of_sound * pair.fs
    generator = np.random.default_rng(rng)
    jitter = 1.0 + 0.1 * generator.uniform(-1.0, 1.0, size=free_path_samples.size)
    return _nearest_coprime(free_path_samples * jitter)


def design_fdn(
    target: RoomTarget,
    delays: np.ndarray | Sequence[np.ndarray],
    *,
    rng: int = 0,
) -> list[tuple[pyFDN.DSS, dict]]:
    """One network per decay slope of ``target``, as ``(pyFDN.DSS, filters)`` pairs.

    ``delays`` (samples) is one array for every slope or one per slope. Each network has a random orthogonal
    feedback matrix seeded ``rng + slope``, the attenuation ``filters["post_delay"]`` following ``target.rt``
    (:meth:`~compas_acoustics.hybrid.target.RoomTarget.geq_bands`) and, as ``C``, the lines' omnidirectional sum,
    ``1 / sqrt(N)`` each, which the late envelope compares with its reference; the renderers project the lines
    themselves.

    With several slopes the networks share their input, and ``filters["post_output"]`` weights each per band by
    ``level_k / sqrt(sum_j level_j ** 2)``, which keeps the slopes' level ratios and one network's initial energy;
    with one slope it is None. One network per slope is the package's choice; Kirsch et al. (2023, Sec. 2.1.2)
    superpose one network per room of two coupled rooms.
    """
    n_slopes = target.n_slopes
    delays = np.asarray(delays, dtype=int)
    if delays.ndim == 1:
        delays = np.tile(delays, (n_slopes, 1))
    if delays.shape[0] != n_slopes:
        raise ValueError(f"delays must be one array, or one per slope ({n_slopes}), got {delays.shape[0]}")
    num_delays = delays.shape[1]
    if num_delays < 1:
        raise ValueError("delays must hold at least one length")
    C = np.full((1, num_delays), 1.0 / np.sqrt(num_delays))

    weights_db = None
    if n_slopes > 1:
        level = np.asarray(target.level, dtype=float)
        weights_db = 20.0 * np.log10(level / np.sqrt(np.sum(level**2, axis=1, keepdims=True)))

    builds = []
    for slope in range(n_slopes):
        dss, filters = pyFDN.fdn_dss_gallery(
            num_delays,
            fs=target.fs,
            delays=delays[slope],
            num_outputs=C.shape[0],
            io_type="normalized",
            direct_gain=0.0,
            rt=None,
            rng=rng + slope,
        )
        dss = dss._replace(C=C)
        filters = dict(
            filters,
            post_delay=pyFDN.decay_to_graphic_eq(
                target.geq_bands(slope),
                delays[slope],
                target.fs,
            ),
        )
        if weights_db is not None:
            db = weights_db[:, slope]
            sos = pyFDN.gain_to_graphic_eq(np.concatenate(([db[0]], db, [db[-1]])), target.fs)
            filters["post_output"] = sos[:, :, np.newaxis]
        builds.append((dss, filters))
    return builds


def run_network(
    builds: Sequence[tuple[pyFDN.DSS, dict]],
    drive: np.ndarray,
    *,
    tap: np.ndarray | None = None,
    line_eq: np.ndarray | None = None,
) -> np.ndarray:
    """Run the networks with one input per delay line; the lines' outputs, (n, num_delays), summed over the slopes.

    ``drive`` (n, num_delays) enters line ``i`` in column ``i``. ``line_eq``, an SOS bank (n_sections, 6,
    num_delays), and the slope weights (``post_output``) filter each line after the feedback is taken, so they
    colour a line without changing its decay: the directional boundary absorption of Kirsch et al. (2023,
    Sec. 1.2). The ambisonic late field is the result times ``C.T``.

    ``tap`` (n, num_delays) enters each line skipping its delay (Ewert et al. 2025, Sec. 3.1.2): with ``f`` the
    attenuation filters, ``f(tap)`` is heard at once and ``A f(tap)`` enters the lines, which needs a static ``A``.
    The scattered parts of :mod:`compas_acoustics.hybrid.scattering` enter there.
    """
    builds = list(builds)
    if not builds:
        raise ValueError("builds must contain at least one network")
    drive = np.asarray(drive, dtype=float)
    if drive.ndim != 2:
        raise ValueError("drive must be 2-D (n_samples, num_delays)")

    n_samples, n_delays = drive.shape
    for dss, _ in builds:
        if np.asarray(dss.delays).size != n_delays:
            raise ValueError(f"drive must have one column per delay line: got {n_delays} columns for a network of {np.asarray(dss.delays).size}")
    if tap is not None:
        tap = np.asarray(tap, dtype=float)
        if tap.shape != drive.shape:
            raise ValueError(f"tap must have the shape of drive, {drive.shape}")
        if any(np.asarray(dss.A).ndim != 2 for dss, _ in builds):
            raise ValueError("tap needs a static feedback matrix")
    if line_eq is not None:
        line_eq = np.asarray(line_eq, dtype=float)
        if line_eq.ndim != 3 or line_eq.shape[1:] != (6, n_delays):
            raise ValueError(f"line_eq must be (n_sections, 6, {n_delays}), got {line_eq.shape}")

    identity = np.eye(n_delays)
    no_direct = np.zeros((n_delays, n_delays))

    def heard_from_the_lines(out, filters):
        if line_eq is not None:
            out = _SOSBank(line_eq).process_signal(out)
        if filters["post_output"] is not None:
            weight = np.asarray(filters["post_output"], dtype=float)[:, :, :1]
            out = _SOSBank(np.repeat(weight, n_delays, axis=2)).process_signal(out)
        return out

    total = np.zeros((n_samples, n_delays), dtype=float)
    for dss, filters in builds:
        post_delay = _SOSBank(filters["post_delay"]) if filters["post_delay"] is not None else None
        heard = None
        if tap is not None:
            heard = _SOSBank(filters["post_delay"]).process_signal(tap) if filters["post_delay"] is not None else tap
        line_input = drive if heard is None else drive + heard @ np.asarray(dss.A).T
        network = pyFDN.dss_to_td(dss.delays, dss.A, identity, identity, no_direct, post_delay=post_delay)
        total += heard_from_the_lines(np.asarray(network.process_signal(line_input), dtype=float), filters)
        if heard is not None:
            total += heard_from_the_lines(heard, filters)
    return total
