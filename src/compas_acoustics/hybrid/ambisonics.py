"""Ambisonic rendering of a :class:`~compas_acoustics.hybrid.synthesis.HybridCore`: paths encoded at their directions, delay lines through the output matrix.

The order sets the number of delay lines (:func:`default_num_delays`) and their projection (:func:`ambisonic_output_matrix`).
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
from .late import MIN_DELAYS
from .late import fdn_directions
from .synthesis import HybridCore
from .synthesis import HybridResult

__all__ = ["render_ambisonics", "ambisonic_output_matrix", "default_num_delays"]


def render_ambisonics(core: HybridCore, sh_order: int = 1, normalization: Literal["SN3D", "N3D"] = "SN3D") -> HybridResult:
    """The core as an ambisonic response: every early path encoded at its direction, every line projected through
    the output matrix of its direction (:func:`ambisonic_output_matrix`), ACN channels at ``sh_order``.

    Everything is computed in N3D; ``normalization`` sets the output, SN3D (AmbiX) or N3D, both with W = 1 for a
    unit plane wave. An order the core's grid cannot carry raises (the output matrix's leakage check).
    """
    if normalization not in ("SN3D", "N3D"):
        raise ValueError(f"normalization must be 'SN3D' or 'N3D', not {normalization!r}")
    C = ambisonic_output_matrix(sh_order, fdn_directions(core.lines.shape[1]))
    n = core.n_samples
    early = encode(core.images, sh_order, n)
    if core.diffraction_paths:
        early = early + encode_diffraction(list(core.diffraction_paths), sh_order, n, core.fs)
    if core.btms_contributions:
        early = early + encode_btms(list(core.btms_contributions), sh_order, n, core.fs)
    lines = core.shaped_lines(early[:, 0])
    early, late = (_from_n3d(x, normalization) for x in (early, lines @ C.T))
    return HybridResult(core=core, rir=early + late, early=early, late=late, late_lines=lines, normalization=normalization)


def _from_n3d(x: np.ndarray, normalization: str) -> np.ndarray:
    """ACN channels on the last axis, from N3D to ``normalization``: SN3D is
    N3D / sqrt(2n + 1) per order n, so W is untouched."""
    if normalization == "N3D":
        return x
    order = np.floor(np.sqrt(np.arange(x.shape[-1]))).astype(int)
    return x / np.sqrt(2 * order + 1)


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

    ``normalization="NM"`` is N3D with W = 1 for a unit plane wave (without the 1/sqrt(4 pi)). Its basis is
    orthonormal, so ``C @ C.T`` of an ideal grid is a multiple of the identity and the leakage compares like
    with like across orders (in SN3D the diagonal falls as 1 / (2n + 1)).

    The late field is isotropic only if ``C @ C.T`` is diagonal; its off-diagonal entries are the channels'
    cross-correlation, which depends on how well the grid supports ``sh_order``. The largest of them over the
    mean diagonal is the leakage; a grid leaking more than ``max_leakage`` raises ``ValueError``.
    """
    sh = spharpy.spherical.spherical_harmonic_basis_real(sh_order, directions, normalization=normalization, channel_convention="ACN")
    C = sh.T / np.sqrt(sh.shape[0])

    gram = C @ C.T
    off_diagonal = gram - np.diag(np.diag(gram))
    leakage = np.max(np.abs(off_diagonal)) / np.mean(np.abs(np.diag(gram)))
    if leakage > max_leakage:
        raise ValueError(
            f"direction grid leaks {leakage:.1%} between ambisonic channels, "
            f"above the {max_leakage:.0%} limit: it does not support sh_order="
            f"{sh_order}; default_num_delays({sh_order}) gives a number of "
            f"delay lines whose grid does."
        )
    return C


@cache
def default_num_delays(sh_order: int, minimum: int | None = None) -> int:
    """Smallest workable number of delay lines for an ambisonic order.

    The grid must hold exactly one point per line and support the order within the leakage limit of
    :func:`ambisonic_output_matrix`. The counts that do are irregular (20, 23, 24, 27, 28 ... at second
    order), so they are searched from ``4 (N + 1) ** 2``, for echo density, and never below ``minimum``
    (:data:`MIN_DELAYS` when None): 96 lines at orders 0 to 3, 100 at fourth order.
    """
    if sh_order < 0:
        raise ValueError("sh_order must be non-negative")
    start = max(MIN_DELAYS if minimum is None else minimum, 4 * (sh_order + 1) ** 2)
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
