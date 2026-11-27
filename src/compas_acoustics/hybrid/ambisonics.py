"""Ambisonic rendering of a :class:`~compas_acoustics.hybrid.synthesis.HybridCore`.

Early paths are encoded at their directions and delay lines projected through the output matrix. The order sets the
number of delay lines (:func:`default_num_delays`) and their projection (:func:`ambisonic_output_matrix`).
"""

from __future__ import annotations

from functools import cache
from typing import Literal

import numpy as np
import pyfar as pf
import spharpy

from .diffraction import encode_btms
from .diffraction import encode_diffraction
from .early import encode
from .envelope import MOMENTS_MAX_GAIN
from .envelope import shape_late
from .late import MIN_DELAYS
from .late import fdn_directions
from .moments import synthesize
from .synthesis import HybridCore
from .synthesis import HybridResult

__all__ = ["render_ambisonics", "ambisonic_output_matrix", "default_num_delays"]


def render_ambisonics(core: HybridCore, sh_order: int = 1, normalization: Literal["SN3D", "N3D"] = "SN3D") -> HybridResult:
    """The core as an ambisonic response, ACN channels at ``sh_order``.

    Every early path is encoded at its direction and every line projected through the output matrix
    (:func:`ambisonic_output_matrix`). Everything is computed in N3D; ``normalization`` sets the output, SN3D
    (AmbiX) or N3D, both with W = 1 for a unit plane wave. An order the core's grid cannot carry raises.

    A core prepared with ``late_field="moments"`` has no lines: its late field is synthesized at ``sh_order``
    (:func:`~compas_acoustics.hybrid.moments.synthesize`) and, where its energy is the network's and not the rays',
    brought to the reference by the late envelope as the lines would be, raised up to ``MOMENTS_MAX_GAIN`` too.
    """
    if normalization not in ("SN3D", "N3D"):
        raise ValueError(f"normalization must be 'SN3D' or 'N3D', not {normalization!r}")
    n = core.n_samples
    early = _early(core, sh_order, n)
    if core.late is not None:
        late = synthesize(core.late, sh_order, core.fs)
        if core.late.follows_reference and core.late_envelope is not None:
            late = shape_late(late, early[:, 0], late[:, 0], core.late_envelope, core.fs, max_gain=MOMENTS_MAX_GAIN)
        early, late = (_from_n3d(x, normalization) for x in (early, late))
        return HybridResult(core=core, rir=early + late, early=early, late=late, late_lines=None, normalization=normalization)
    C = ambisonic_output_matrix(sh_order, fdn_directions(core.lines.shape[1]))
    lines = core.shaped_lines(early[:, 0])
    early, late = (_from_n3d(x, normalization) for x in (early, lines @ C.T))
    return HybridResult(core=core, rir=early + late, early=early, late=late, late_lines=lines, normalization=normalization)


def _early(core: HybridCore, sh_order: int, n: int) -> np.ndarray:
    """The early paths encoded at their directions, N3D, (n, (sh_order + 1) ** 2)."""
    early = encode(core.images, sh_order, n)
    if core.diffraction_paths:
        early = early + encode_diffraction(list(core.diffraction_paths), sh_order, n, core.fs)
    if core.btms_contributions:
        early = early + encode_btms(list(core.btms_contributions), sh_order, n, core.fs)
    return early


def _from_n3d(x: np.ndarray, normalization: str) -> np.ndarray:
    """ACN channels on the last axis from N3D to ``normalization``; SN3D is N3D / sqrt(2n + 1) per order n."""
    if normalization == "N3D":
        return x
    order = np.floor(np.sqrt(np.arange(x.shape[-1]))).astype(int)
    return x / np.sqrt(2 * order + 1)


def ambisonic_output_matrix(sh_order: int, directions: pf.Coordinates) -> np.ndarray:
    """Output matrix ``C`` (n_channels, N) from delay lines to ambisonic channels, ACN.

    Column ``i`` is the spherical harmonic vector of direction ``i`` over ``sqrt(N)``, so the output energy does not
    grow with the number of lines. The basis is spharpy's ``"NM"``, N3D with W = 1 for a unit plane wave,
    orthonormal: on an ideal grid ``C @ C.T`` is a multiple of the identity. Its largest off-diagonal entry over its
    mean diagonal is the leakage between channels; a grid leaking more than 5 % (the package's choice) does not
    support ``sh_order`` and raises ``ValueError``.
    """
    sh = spharpy.spherical.spherical_harmonic_basis_real(sh_order, directions, normalization="NM", channel_convention="ACN")
    C = sh.T / np.sqrt(sh.shape[0])

    gram = C @ C.T
    off_diagonal = gram - np.diag(np.diag(gram))
    leakage = np.max(np.abs(off_diagonal)) / np.mean(np.abs(np.diag(gram)))
    if leakage > 0.05:
        raise ValueError(
            f"direction grid leaks {leakage:.1%} between ambisonic channels, "
            f"above the 5% limit: it does not support sh_order="
            f"{sh_order}; default_num_delays({sh_order}) gives a number of "
            f"delay lines whose grid does."
        )
    return C


@cache
def default_num_delays(sh_order: int) -> int:
    """Smallest workable number of delay lines for an ambisonic order.

    The grid must hold exactly one point per line and support the order within the leakage limit of
    :func:`ambisonic_output_matrix`. The counts that do are irregular (20, 23, 24, 27, 28 ... at second order), so
    they are searched from ``4 (N + 1) ** 2``, for echo density (the package's choice), and never below
    :data:`MIN_DELAYS`: 96 lines at orders 0 to 3, 100 at fourth order.
    """
    if sh_order < 0:
        raise ValueError("sh_order must be non-negative")
    start = max(MIN_DELAYS, 4 * (sh_order + 1) ** 2)
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
        f"no equal_area grid between {start} and {4 * start} points supports sh_order={sh_order}"
    )
