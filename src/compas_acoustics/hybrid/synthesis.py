"""The hybrid ambisonic room impulse response of one source-receiver pair.

Ported by hand from hybrid-rir (gpapadimitri/hybrid-rir, src/synthesis.py, at
8a70661: the btms branch included). The image
source search moved to the caller, which runs one search for every source and
receiver of a scene: ``iss`` holds the pair's images up to ``max_order + 1``.
Upstream's scattering and obstacle-free drive are not ported.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace
from typing import Literal

import numpy as np
import pyfar as pf
import pyFDN

from .diffraction import btms_paths
from .diffraction import diffraction_paths
from .diffraction import encode_btms
from .diffraction import encode_diffraction
from .directional import directional_absorption_filters
from .early import ImageSourceSet
from .early import encode
from .early import render_images
from .early import route_to_directions
from .envelope import shape_late
from .late import default_num_delays
from .late import design_fdn
from .late import directional_delays
from .late import fdn_directions
from .late import render
from .pair import Pair
from .target import RoomTarget


@dataclass(frozen=True)
class HybridResult:
    """A synthesised response and everything needed to reproduce it."""

    rir: np.ndarray  # (n, (sh_order + 1) ** 2), ACN, in ``normalization``
    early: np.ndarray
    late: np.ndarray
    builds: list[pyFDN.FDNBuild]
    target: RoomTarget
    max_order: int
    driving_images: int
    drive_gain: float
    fs: float
    normalization: str  # "SN3D" or "N3D", audio convention (W = 1 for a unit plane wave)


def synthesize(
    spec: Pair,
    iss: ImageSourceSet,
    target: RoomTarget,
    *,
    max_order: int,
    sh_order: int = 1,
    duration_s: float | None = None,
    num_delays: int | None = None,
    drive_gain: float | None = None,
    directional_absorption: bool = True,
    late_envelope: np.ndarray | None = None,
    diffraction: bool | Literal["udfa", "btms"] = False,
    normalization: Literal["SN3D", "N3D"] = "SN3D",
    rng: int = 0,
) -> HybridResult:
    """Synthesise a hybrid ambisonic room impulse response.

    After Wendt, van de Par and Ewert (2014) and Kirsch et al. (2021, 2023),
    with ambisonic output:

    1. Images up to ``max_order``, encoded at their directions of arrival, are
       the early field.
    2. An FDN with one delay line per direction of an equal-area grid; its
       output matrix holds the spherical harmonics of those directions, its
       attenuation filters follow ``target.rt``, and each line is the room's
       free path along its direction (:func:`directional_delays`).
    3. Images of order ``max_order + 1`` enter the delay line nearest their
       direction of arrival.
    4. Each line's output is weighted with the reflection of the boundary in its
       direction (``directional_absorption``), and early field and network
       output are summed.

    ``drive_gain`` defaults to ``sqrt(num_delays)``. ``late_envelope`` is an
    omnidirectional reference on the package scale, normally the pair's
    ray-traced response; the network output is attenuated per band to follow
    its late energy (Vorlaender 2020, Sec. 11.9). ``diffraction=True`` adds
    edge diffraction around the obstacles to the early field (Kirsch and Ewert
    2023), for the direct sound and, when ``max_order >= 1``, one enclosure
    reflection before or after the edge; the late field is not affected.
    ``diffraction="btms"`` renders first- and second-order diffraction with the
    BTMS model (Svensson et al. 1999, kernels ported from EDtoolbox v0.504):
    first order with at most one wall reflection, second order between any two
    visible obstacle edges, direct paths only; slow, minutes per source-receiver
    pair. ``True`` is ``"udfa"``.
    Everything is computed in N3D; ``normalization`` sets the output, SN3D
    (AmbiX) or N3D, both with W = 1 for a unit plane wave.
    """
    if normalization not in ("SN3D", "N3D"):
        raise ValueError(f"normalization must be 'SN3D' or 'N3D', not {normalization!r}")
    model = {False: None, True: "udfa", "udfa": "udfa", "btms": "btms"}.get(diffraction, "invalid")
    if model == "invalid":
        raise ValueError(f"diffraction must be False, True, 'udfa' or 'btms', not {diffraction!r}")
    if max_order < 0:
        raise ValueError("max_order must be non-negative")
    if duration_s is None:
        duration_s = 1.5 * float(target.rt.max())
    if duration_s <= 0.0:
        raise ValueError(f"duration_s must be positive, got {duration_s!r}")
    n_samples = int(np.ceil(duration_s * spec.fs))

    if num_delays is None:
        num_delays = default_num_delays(sh_order)
    directions = fdn_directions(sh_order, num_delays)
    delays = [
        directional_delays(spec, directions, rng=rng + slope)
        for slope in range(target.n_slopes)
    ]
    builds = design_fdn(target, delays, sh_order=sh_order, rng=rng)
    if drive_gain is None:
        drive_gain = float(np.sqrt(num_delays))

    driving = iss.order == max_order + 1
    early = encode(_select(iss, ~driving), sh_order, n_samples)
    driving_set = _select(iss, driving)
    if model == "udfa":
        paths = diffraction_paths(spec, max_order=min(max_order, 1))
        early = early + encode_diffraction(paths, sh_order, n_samples, spec.fs)
    elif model == "btms":
        contributions = btms_paths(spec, max_order=min(max_order, 1))
        early = early + encode_btms(contributions, sh_order, n_samples, spec.fs)
    drive = render_images(
        driving_set, route_to_directions(driving_set, directions), n_samples
    )
    line_eq = (
        directional_absorption_filters(spec, directions)
        if directional_absorption
        else None
    )
    late = render(builds, drive * drive_gain, line_eq=line_eq)
    if late_envelope is not None:
        late = shape_late(late, early, late_envelope, spec.fs)

    early, late = (_from_n3d(x, normalization) for x in (early, late))
    return HybridResult(
        rir=early + late,
        early=early,
        late=late,
        builds=builds,
        target=target,
        max_order=max_order,
        driving_images=int(driving_set.delay.size),
        drive_gain=float(drive_gain),
        fs=spec.fs,
        normalization=normalization,
    )


def _from_n3d(x: np.ndarray, normalization: str) -> np.ndarray:
    """ACN channels on the last axis, from N3D to ``normalization``: SN3D is
    N3D / sqrt(2n + 1) per order n, so W is untouched."""
    if normalization == "N3D":
        return x
    order = np.floor(np.sqrt(np.arange(x.shape[-1]))).astype(int)
    return x / np.sqrt(2 * order + 1)


def _select(iss: ImageSourceSet, mask: np.ndarray) -> ImageSourceSet:
    """The subset of an image source set selected by a boolean mask."""
    xyz = iss.doa.cartesian[mask]
    return replace(
        iss,
        delay=iss.delay[mask],
        gain=iss.gain[mask],
        doa=pf.Coordinates(xyz[:, 0], xyz[:, 1], xyz[:, 2]),
        order=iss.order[mask],
    )
