"""The ambisonic feedback delay network: one delay line per direction.

Ported from hybrid-rir (gpapadimitri/hybrid-rir @ a272da7, src/late.py).
The free paths read the faces of a RoomGeometry.
"""

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


def ambisonic_output_matrix(
    sh_order: int,
    directions: pf.Coordinates,
    *,
    normalization: str = "NM",
    max_leakage: float = 0.05,
) -> np.ndarray:
    """Output matrix C mapping delay lines to ambisonic channels.

    Column ``i`` is the spherical harmonic vector of direction ``i``, so each
    delay line radiates from its own direction and the late field is diffuse in
    expectation. Scaled by ``1 / sqrt(N)`` so total output energy does not grow
    with the number of delay lines.

    ``normalization="NM"`` is spharpy's audio N3D (a unit plane wave gives
    W = 1), not the physics "N3D", which keeps the 1/sqrt(4 pi). The package
    computes in N3D: its basis is orthonormal, so ``C @ C.T`` of an ideal grid
    is a multiple of the identity and the leakage below compares like with
    like across orders. In SN3D the diagonal falls as 1 / (2n + 1) per order.

    The diffuse-field assumption only holds if ``C @ C.T`` is diagonal: its
    off-diagonal entries are the cross-correlation between ambisonic channels,
    and a large one means the late field carries a directional bias rather than
    being isotropic. That depends entirely on how well the direction grid
    supports ``sh_order``, so it is measured here rather than assumed.
    Measured: a default ``spiral_points`` grid of 16 directions leaks 21.5 per
    cent -- about -13 dB of channel cross-talk -- while ``equal_area`` at the
    same size leaks 2e-14 per cent.
    """
    sh = spharpy.spherical.spherical_harmonic_basis_real(
        sh_order, directions, normalization=normalization, channel_convention="ACN"
    )
    C = sh.T / np.sqrt(sh.shape[0])

    gram = C @ C.T
    off_diagonal = gram - np.diag(np.diag(gram))
    leakage = np.max(np.abs(off_diagonal)) / np.mean(np.abs(np.diag(gram)))
    if leakage > max_leakage:
        raise ValueError(
            f"direction grid leaks {leakage:.1%} between ambisonic channels, "
            f"above the {max_leakage:.0%} limit: it does not support sh_order="
            f"{sh_order}. Leave num_delays unset to have default_num_delays"
            f"({sh_order}) pick a count that works, or pass a better grid."
        )
    return C


@cache
def fdn_directions(sh_order: int, num_delays: int | None = None) -> pf.Coordinates:
    """The direction each delay line radiates from, shape (num_delays,).

    One grid, used three times: :func:`ambisonic_output_matrix` turns it into
    ``C``, the image sources are routed onto it, and :func:`directional_delays`
    reads the room along it. They must be the same grid, so this is the only
    place it is constructed.
    """
    if num_delays is None:
        num_delays = default_num_delays(sh_order)
    directions = spharpy.samplings.equal_area(n_max=sh_order, n_points=num_delays)
    if directions.cshape[0] != num_delays:
        raise ValueError(
            f"equal_area returned {directions.cshape[0]} directions for "
            f"num_delays={num_delays}; the output matrix needs exactly one "
            f"direction per delay line. Leave num_delays unset to have "
            f"default_num_delays({sh_order}) choose a count that works."
        )
    return directions


def default_num_delays(sh_order: int) -> int:
    """Smallest workable number of delay lines for an ambisonic order.

    ``num_delays`` and ``sh_order`` are coupled twice: the direction grid must
    hold exactly one point per delay line, and it must support the order well
    enough for :func:`ambisonic_output_matrix` to accept its channel leakage.
    A fixed 16 satisfied both only at first order -- second order leaked 7.8
    per cent, and at third order ``equal_area`` returns 17 points when asked
    for 16, so the network could not be built at all.

    Searched rather than computed, because the counts that satisfy both
    conditions are irregular: 20, 23, 24, 27, 28 ... at second order. The
    search starts at ``4 (N + 1) ** 2``: more lines than the order strictly
    needs, for echo density.
    """
    if sh_order < 0:
        raise ValueError("sh_order must be non-negative")
    start = max(16, 4 * (sh_order + 1) ** 2)
    for num_delays in range(start, 4 * start + 1):
        directions = spharpy.samplings.equal_area(n_max=sh_order, n_points=num_delays)
        if directions.cshape[0] != num_delays:
            continue
        try:
            ambisonic_output_matrix(sh_order, directions)
        except ValueError:
            continue
        return num_delays
    raise ValueError(  # pragma: no cover - no order in range hits this
        f"no equal_area grid between {start} and {4 * start} points supports "
        f"sh_order={sh_order}; pass num_delays explicitly"
    )


def _nearest_coprime(targets: np.ndarray, tolerance: float = 0.03) -> np.ndarray:
    """Distinct integers near their targets, pairwise coprime where that fits.

    Coprimality is only sought within ``tolerance`` of each target, and the
    nearest unused integer is taken where none is found. Unbounded, it destroys
    the lengths it is meant to protect: 64 pairwise-coprime values between 340
    and 650 samples do not exist, and the search walked lines down to 181.
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
            pick = next(
                centre + s * k
                for k in range(centre)
                for s in (1, -1)
                if centre + s * k >= 2 and centre + s * k not in chosen
            )
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

    Kirsch, Poppitz and Wendt (2021) and Kirsch, Wendt and van de Par (2023)
    derive the delay lines from the room's dimensions. This does that per
    direction, in a form that holds for any closed geometry. A uniform field
    travelling along ``u`` meets the boundary at the rate
    ``c sum_f A_f |n_f . u| / (2 V)``, so line ``i`` takes the reciprocal,
    ``2 V / sum_f A_f |n_f . u_i|``. For a shoebox that is exactly
    ``1 / (|u_x| / L_x + |u_y| / L_y + |u_z| / L_z)``: the dimensions, with an
    axis-aligned line taking that axis' length. Its average over the sphere is
    Cauchy's mean free path ``4 V / S``. A polygon or mesh needs no separate
    rule, only its faces.

    Directions ``u`` and ``-u`` have the same free path, and a symmetric grid
    holds both. ``spread`` jitters each length uniformly by that fraction
    before the lengths are made coprime, so opposite lines do not end up a
    sample or two apart and ring together.

    ``directions`` are the delay lines' directions in the receiver's frame,
    which is also the frame the images are routed in and the output matrix is
    built in. The faces are in room coordinates, so a rotated receiver has its
    grid rotated back before the two are projected onto each other; without
    that, a listener turned 90 degrees would read the room's length along the
    line that points at its width.
    """
    unit = directions.cartesian
    unit = unit / np.linalg.norm(unit, axis=1, keepdims=True)
    if spec.receiver_rotation is not None:
        unit = unit @ spec.receiver_rotation.T
    projected = np.abs(unit @ spec.geometry.area_normals().T).sum(axis=1)
    free_path_samples = (
        2.0 * spec.geometry.volume / projected / spec.speed_of_sound * spec.fs
    )
    generator = np.random.default_rng(rng)
    jitter = 1.0 + spread * generator.uniform(-1.0, 1.0, size=free_path_samples.size)
    return _nearest_coprime(free_path_samples * jitter)


def design_fdn(
    target: RoomTarget,
    delays: np.ndarray | Sequence[np.ndarray],
    *,
    sh_order: int = 1,
    matrix: str = "orthogonal",
    dc_factor: float = 1.0,
    nyquist_factor: float = 0.5,
    rng: int = 0,
) -> list[pyFDN.FDNBuild]:
    """Design one ambisonic FDN per decay slope of ``target``.

    ``delays`` is one array of lengths in samples, one per delay line, shared
    by every slope, or one such array per slope. The number of lines sets the
    direction grid (:func:`fdn_directions`), which gives the output matrix.
    Each slope's attenuation filters follow ``target.rt`` for that slope, and
    its feedback matrix is seeded ``rng + slope``.

    A multi-exponential decay is realised by summing independent FDNs, one per
    slope, as in pyFDN's multi-slope example. All of them take the same input,
    so each is weighted by its share of the initial level: slope ``k`` gets
    ``level_k / sqrt(sum_j level_j ** 2)`` per band, as an output filter. That
    keeps the ratios of ``target.level`` between slopes and leaves the summed
    initial energy equal to one network's. A single slope gets no output
    filter at all.
    """
    n_slopes = target.n_slopes
    delays = np.asarray(delays, dtype=int)
    if delays.ndim == 1:
        delays = np.tile(delays, (n_slopes, 1))
    if delays.shape[0] != n_slopes:
        raise ValueError(
            f"delays must be one array, or one per slope ({n_slopes}), got "
            f"{delays.shape[0]}"
        )
    num_delays = delays.shape[1]
    if num_delays < 1:
        raise ValueError("delays must hold at least one length")
    directions = fdn_directions(sh_order, num_delays)
    C = ambisonic_output_matrix(sh_order, directions)

    weights_db = None
    if n_slopes > 1:
        level = np.asarray(target.level, dtype=float)
        weights_db = 20.0 * np.log10(
            level / np.sqrt(np.sum(level**2, axis=1, keepdims=True))
        )

    builds = []
    for slope in range(n_slopes):
        # fdn_build_gallery is pyFDN's seeded entry point; fdn_matrix_gallery and
        # random_orthogonal would draw from NumPy's global state
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
                target.geq_bands(
                    slope, dc_factor=dc_factor, nyquist_factor=nyquist_factor
                ),
                delays[slope],
                target.fs,
            ),
        )
        if weights_db is not None:
            db = weights_db[:, slope]
            sos = pyFDN.gain_to_geq(np.concatenate(([db[0]], db, [db[-1]])), target.fs)
            build = replace(
                build, post_output=np.repeat(sos[:, :, np.newaxis], C.shape[0], axis=2)
            )
        builds.append(build)
    return builds


def render(
    builds: Sequence[pyFDN.FDNBuild],
    drive: np.ndarray,
    *,
    tap: np.ndarray | None = None,
    line_eq: np.ndarray | None = None,
) -> np.ndarray:
    """Run the network with a separate input per delay line, shape (n, n_ch).

    ``drive`` is ``(n_samples, num_delays)``: column ``i`` is the signal
    entering delay line ``i``, so ``B`` is the identity and ``D`` is zero. The
    slope networks are summed.

    ``line_eq``, an SOS bank of shape ``(n_sections, 6, num_delays)``, filters
    each line on its way out, after the feedback is taken, so it colours what
    is heard from that direction without touching the decay. That is where the
    direction-dependent boundary absorption of :mod:`hybrid_rir.directional`
    sits (Kirsch et al. 2023). It is applied by rendering the lines themselves
    and projecting with ``C`` afterwards, because ``pyFDN.process_dss`` filters
    either inside the loop (``post_delay``) or after ``C`` (``post_output``).

    ``tap``, the same shape, is added at the delay line outputs, after the
    attenuation filters: it is heard at once through ``C`` and then
    recirculates through ``A``. That is where the scattered reflections of
    :mod:`hybrid_rir.scattering` enter, "initially skipping the FDN delays"
    (Ewert et al. 2025, Sec. 3.1.2). With a static ``A`` it is exactly
    ``C tap`` at the output plus ``A tap`` into the delay lines.

    Not ``pyFDN.build_to_impz``: that renders an impulse through ``B``.
    ``pyFDN.process_dss`` takes a multichannel input and replaces ``B``, ``C``
    and ``D`` per call.
    """
    builds = list(builds)
    if not builds:
        raise ValueError("builds must contain at least one FDNBuild")
    drive = np.asarray(drive, dtype=float)
    if drive.ndim != 2:
        raise ValueError("drive must be 2-D (n_samples, num_delays)")

    n_samples, n_delays = drive.shape
    n_ch = builds[0].C.shape[0]
    for build in builds:
        if np.asarray(build.delays).size != n_delays:
            raise ValueError(
                f"drive must have one column per delay line: got {n_delays} "
                f"columns for a network of {np.asarray(build.delays).size}"
            )
        if build.C.shape[0] != n_ch:
            raise ValueError("every build must have the same number of channels")
    if tap is not None:
        tap = np.asarray(tap, dtype=float)
        if tap.shape != drive.shape:
            raise ValueError(f"tap must have the shape of drive, {drive.shape}")
        if any(np.asarray(b.A).ndim != 2 for b in builds):
            raise ValueError("tap needs a static feedback matrix")
    if line_eq is not None:
        line_eq = np.asarray(line_eq, dtype=float)
        if line_eq.ndim != 3 or line_eq.shape[1:] != (6, n_delays):
            raise ValueError(
                f"line_eq must be (n_sections, 6, {n_delays}), got {line_eq.shape}"
            )

    identity = np.eye(n_delays)
    no_direct = np.zeros((n_ch, n_delays))

    total = np.zeros((n_samples, n_ch), dtype=float)
    for build in builds:
        post_delay = SOSBank(build.post_delay) if build.post_delay is not None else None
        post_output = (
            SOSBank(build.post_output) if build.post_output is not None else None
        )
        line_input = drive if tap is None else drive + tap @ np.asarray(build.A).T
        out = pyFDN.process_dss(
            line_input,
            build.delays,
            build.A,
            identity,
            identity if line_eq is not None else build.C,
            np.zeros((n_delays, n_delays)) if line_eq is not None else no_direct,
            post_delay=post_delay,
            post_output=None if line_eq is not None else post_output,
        )
        # process_dss squeezes its output; restore the outputs' axis every time
        out = np.asarray(out, dtype=float).reshape(
            n_samples, n_delays if line_eq is not None else n_ch
        )
        if line_eq is not None:
            out = SOSBank(line_eq).process_signal(out) @ build.C.T
            if post_output is not None:
                out = post_output.process_signal(out)
        total += out
        if tap is not None:
            heard = tap
            if line_eq is not None:
                heard = SOSBank(line_eq).process_signal(heard)
            heard = heard @ build.C.T
            if build.post_output is not None:
                heard = SOSBank(build.post_output).process_signal(heard)
            total += heard
    return total
