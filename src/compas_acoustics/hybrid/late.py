"""The directional feedback delay network: one delay line per direction."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from functools import cache
from math import gcd

import numpy as np
import pyfar as pf
import pyFDN
import spharpy
from pyFDN.td import SOSBank

from .pair import Pair
from .target import RoomTarget


#: Fewest delay lines at any ambisonic order: Kirsch, Wendt and van de Par (2023, Secs. 1.2 and 3.3) use 96
#: for offline simulation whatever the rendered directions, so the tail's character does not change with the
#: count and a small network's recurrent patterns are avoided. The 96-point equal-area grid supports orders 0
#: to 3.
MIN_DELAYS = 96


@cache
def fdn_directions(num_delays: int = MIN_DELAYS) -> pf.Coordinates:
    """The direction each delay line radiates from, an equal-area grid of ``num_delays`` points.

    The one grid the image sources are routed onto, :func:`directional_delays` reads the room along, the
    directional histogram bins the rays on, and the renderers project the lines from.
    """
    return spharpy.samplings.equal_area(n_max=0, n_points=num_delays)


def _nearest_coprime(targets: np.ndarray, tolerance: float = 0.03) -> np.ndarray:
    """Distinct integers near their targets, pairwise coprime where that fits.

    Coprimality is sought within ``tolerance`` of each target, else the nearest unused integer is taken, so
    no length strays far from its target.
    """
    chosen: list[int] = []
    out = np.empty(targets.size, dtype=int)
    for idx in np.argsort(targets)[::-1]:
        centre = max(int(round(targets[idx])), 2)
        reach = max(2, int(tolerance * centre))
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
    spec: Pair,
    directions: pf.Coordinates,
    *,
    spread: float = 0.1,
    rng: int | np.random.Generator = 0,
) -> np.ndarray:
    """Delay line lengths in samples, each the room's free path along its direction.

    The room-dimension delays of Kirsch, Poppitz and Wendt (2021) and Kirsch, Wendt and van de Par (2023), per
    direction and for any closed geometry: a uniform field along ``u`` meets the boundary at the rate
    ``c sum_f A_f |n_f . u| / (2 V)``, so line ``i`` takes ``2 V / sum_f A_f |n_f . u_i|``. For a shoebox that is
    ``1 / (|u_x| / L_x + |u_y| / L_y + |u_z| / L_z)``; its mean over the sphere is Cauchy's ``4 V / S``.

    ``u`` and ``-u`` share a free path, so ``spread`` jitters each length uniformly by up to that fraction,
    drawn from ``rng``, before the lengths are made coprime, lest opposite lines ring together. ``directions``
    are in the receiver's frame and are rotated back to the room's before projecting.
    """
    unit = directions.cartesian
    unit = unit / np.linalg.norm(unit, axis=1, keepdims=True)
    if spec.receiver_rotation is not None:
        unit = unit @ spec.receiver_rotation.T
    projected = np.abs(unit @ spec.geometry.area_normals().T).sum(axis=1)
    free_path_samples = 2.0 * spec.geometry.volume / projected / spec.speed_of_sound * spec.fs
    generator = np.random.default_rng(rng)
    jitter = 1.0 + spread * generator.uniform(-1.0, 1.0, size=free_path_samples.size)
    return _nearest_coprime(free_path_samples * jitter)


def design_fdn(
    target: RoomTarget,
    delays: np.ndarray | Sequence[np.ndarray],
    *,
    matrix: str = "orthogonal",
    dc_factor: float = 1.0,
    nyquist_factor: float = 0.5,
    rng: int = 0,
) -> list[pyFDN.FDNBuild]:
    """Design one FDN per decay slope of ``target``, with one output: the lines' omnidirectional sum.

    ``delays`` (samples) is one array shared by every slope or one per slope. ``C`` is the W row of the
    ambisonic output matrix, ``1 / sqrt(N)`` per line, which the late envelope compares with its reference;
    renderers project the lines themselves (:func:`render`). Each slope's attenuation follows ``target.rt``
    (:meth:`~compas_acoustics.hybrid.target.RoomTarget.geq_bands` with ``dc_factor`` and ``nyquist_factor``)
    and its feedback matrix is seeded ``rng + slope``; any ``matrix`` other than ``"orthogonal"`` is taken
    from ``pyFDN.fdn_matrix_gallery``, which draws from NumPy's global state.

    A multi-exponential decay sums independent networks with one input, each weighted per band by
    ``level_k / sqrt(sum_j level_j ** 2)`` as an output filter, which keeps the slopes' level ratios and one
    network's initial energy. A single slope has no output filter.
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
        # the seeded entry point; the matrix galleries draw from NumPy's global state
        build = pyFDN.fdn_build_gallery(
            num_delays,
            fs=target.fs,
            delays=delays[slope],
            num_outputs=C.shape[0],
            io_type="normalized",
            direct_gain=0.0,
            rt=None,
            rng=rng + slope,
        )
        if matrix != "orthogonal":
            build = replace(build, A=pyFDN.fdn_matrix_gallery(num_delays, matrix))
        build = replace(
            build,
            C=C,
            post_delay=pyFDN.decay_to_geq(
                target.geq_bands(slope, dc_factor=dc_factor, nyquist_factor=nyquist_factor),
                delays[slope],
                target.fs,
            ),
        )
        if weights_db is not None:
            db = weights_db[:, slope]
            sos = pyFDN.gain_to_geq(np.concatenate(([db[0]], db, [db[-1]])), target.fs)
            build = replace(build, post_output=sos[:, :, np.newaxis])
        builds.append(build)
    return builds


def render(
    builds: Sequence[pyFDN.FDNBuild],
    drive: np.ndarray,
    *,
    tap: np.ndarray | None = None,
    line_eq: np.ndarray | None = None,
) -> np.ndarray:
    """Run the network with a separate input per delay line; the lines' outputs, (n, num_delays).

    ``drive`` (n_samples, num_delays) enters line ``i`` in column ``i`` (``B`` identity, ``D`` zero). Column ``i`` of
    the result is what line ``i`` radiates, after ``line_eq`` and the slope weights, summed over the slopes'
    networks; the ambisonic late field is it times ``C.T``.

    ``line_eq``, an SOS bank (n_sections, 6, num_delays), filters each line on its way out, after the feedback
    is taken, so it colours a direction without touching the decay: the directional boundary absorption of
    Kirsch et al. (2023). The lines go through an identity output matrix and are filtered here, as are the
    builds' slope weights (``post_output``).

    ``tap``, of the same shape, enters each line skipping its delay, "initially skipping the FDN delays" (Ewert
    et al. 2025, Sec. 3.1.2): it passes the attenuation filters ``f``, is heard at once and recirculates, exactly
    ``f(tap)`` at the outputs plus ``A f(tap)`` into the lines with a static ``A``. The scattered parts of
    :mod:`compas_acoustics.hybrid.scattering` enter there.
    """
    builds = list(builds)
    if not builds:
        raise ValueError("builds must contain at least one FDNBuild")
    drive = np.asarray(drive, dtype=float)
    if drive.ndim != 2:
        raise ValueError("drive must be 2-D (n_samples, num_delays)")

    n_samples, n_delays = drive.shape
    for build in builds:
        if np.asarray(build.delays).size != n_delays:
            raise ValueError(f"drive must have one column per delay line: got {n_delays} columns for a network of {np.asarray(build.delays).size}")
    if tap is not None:
        tap = np.asarray(tap, dtype=float)
        if tap.shape != drive.shape:
            raise ValueError(f"tap must have the shape of drive, {drive.shape}")
        if any(np.asarray(b.A).ndim != 2 for b in builds):
            raise ValueError("tap needs a static feedback matrix")
    if line_eq is not None:
        line_eq = np.asarray(line_eq, dtype=float)
        if line_eq.ndim != 3 or line_eq.shape[1:] != (6, n_delays):
            raise ValueError(f"line_eq must be (n_sections, 6, {n_delays}), got {line_eq.shape}")

    identity = np.eye(n_delays)
    no_direct = np.zeros((n_delays, n_delays))

    def heard_from_the_lines(out, build):
        if line_eq is not None:
            out = SOSBank(line_eq).process_signal(out)
        if build.post_output is not None:
            weight = np.asarray(build.post_output, dtype=float)[:, :, :1]
            out = SOSBank(np.repeat(weight, n_delays, axis=2)).process_signal(out)
        return out

    total = np.zeros((n_samples, n_delays), dtype=float)
    for build in builds:
        post_delay = SOSBank(build.post_delay) if build.post_delay is not None else None
        heard = None
        if tap is not None:
            heard = SOSBank(build.post_delay).process_signal(tap) if build.post_delay is not None else tap
        line_input = drive if heard is None else drive + heard @ np.asarray(build.A).T
        out = pyFDN.process_dss(
            line_input,
            build.delays,
            build.A,
            identity,
            identity,
            no_direct,
            post_delay=post_delay,
        )
        # process_dss squeezes its output; restore the lines' axis every time
        total += heard_from_the_lines(np.asarray(out, dtype=float).reshape(n_samples, n_delays), build)
        if heard is not None:
            total += heard_from_the_lines(heard, build)
    return total
