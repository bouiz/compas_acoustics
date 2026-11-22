"""The format-neutral core of one pair's hybrid response: its early paths and delay lines (:func:`prepare`).

A renderer, :mod:`~compas_acoustics.hybrid.ambisonics` or :mod:`~compas_acoustics.hybrid.binaural`, turns it into
a response.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from dataclasses import replace
from typing import Literal

import numpy as np
import pyfar as pf
import pyFDN

from compas_acoustics.room.raytracing import DirectionalHistogram

from .diffraction import BtmsContribution
from .diffraction import DiffractionPath
from .diffraction import btms_paths
from .diffraction import diffraction_paths
from .directional import directional_absorption_filters
from .early import ImageSourceSet
from .early import render_images
from .scattering import scattering_decay_time
from .scattering import scattering_weights
from .scattering import smear
from .envelope import shape_late
from .late import MIN_DELAYS
from .late import design_fdn
from .late import directional_delays
from .late import fdn_directions
from .late import run_network
from .pair import Pair
from .target import RoomTarget


@dataclass(frozen=True)
class HybridCore:
    """The hybrid of one pair before it has a format: its early paths and its delay lines.

    The renderer evaluates each early path once; the late envelope compares the lines with the omnidirectional
    early field, every renderer's first channel, so it is applied there (:meth:`shaped_lines`).
    """

    pair: Pair
    n_samples: int
    fs: float
    max_order: int
    target: RoomTarget
    images: ImageSourceSet  # orders 0 .. max_order
    diffraction_paths: tuple[DiffractionPath, ...]
    btms_contributions: tuple[BtmsContribution, ...]
    builds: list[tuple[pyFDN.DSS, dict]]  # (network, filters) per decay slope
    lines: np.ndarray  # (n, num_delays), each line as heard from its direction, before the late envelope
    line_directions: np.ndarray  # (num_delays, 3), unit vectors in the receiver's frame
    late_envelope: np.ndarray | None
    directional_envelope: DirectionalHistogram | None
    drive_gain: float
    driving_images: int

    def shaped_lines(self, early_w: np.ndarray) -> np.ndarray:
        """The lines after the late envelope, given ``early_w``, the omnidirectional (N3D W) early field."""
        if self.late_envelope is None:
            return self.lines
        w = self.builds[0][0].C[0]
        return shape_late(self.lines, early_w, self.lines @ w, self.late_envelope, self.fs, histogram=self.directional_envelope)


@dataclass(frozen=True)
class HybridResult:
    """A rendered response of a :class:`HybridCore`, in the renderer's format."""

    core: HybridCore
    rir: np.ndarray  # (n, channels): ACN in ``normalization``, or the two ears
    early: np.ndarray
    late: np.ndarray
    late_lines: np.ndarray  # (n, num_delays), the shaped lines the late field was rendered from
    normalization: str | None  # "SN3D" or "N3D" (W = 1 for a unit plane wave); None for the ears


def prepare(
    pair: Pair,
    iss: ImageSourceSet,
    target: RoomTarget,
    *,
    max_order: int,
    duration_s: float,
    num_delays: int = MIN_DELAYS,
    directional_absorption: bool = True,
    late_envelope: np.ndarray | None = None,
    directional_envelope: DirectionalHistogram | None = None,
    diffraction: Literal["udfa", "btms"] | None = None,
    rng: int = 0,
    scattering: bool = False,
) -> HybridCore:
    """The format-neutral part of the hybrid for one pair: its early paths and its delay lines.

    After Wendt et al. (2014) and Kirsch et al. (2021, 2023):

    1. Early field: the images of ``iss`` up to ``max_order`` and, with ``diffraction``, the diffracted paths, kept
       path by path.
    2. Network: one delay line per direction of :func:`fdn_directions`, as long as the room's free path along it
       (:func:`directional_delays`), decaying at ``target.rt``.
    3. Drive: the images of order ``max_order``, each spread over the lines of the hemisphere opposite its arrival
       (:func:`~compas_acoustics.hybrid.scattering.scattering_weights` with ``opposite``; Ewert et al. 2025,
       Sec. 3.1.2) and scaled by ``sqrt(num_delays)``: the input "must equal the total energy of the Nth-order
       reflections" (Wendt et al. 2014, Sec. 1.3).
    4. With ``directional_absorption``, each line is weighted with the reflection of the boundary in its direction
       (:mod:`~compas_acoustics.hybrid.directional`).

    ``duration_s`` is the response length in seconds. ``rng`` seeds the delay jitter and the feedback matrices,
    ``rng + slope`` for each slope.

    ``late_envelope``, a package-scale omnidirectional reference, attenuates the lines per band to its late energy:
    the envelope of Vorlaender (2020, Sec. 11.9) imposed on the network's lines, the package's choice.
    ``directional_envelope``, a :class:`~compas_acoustics.room.raytracing.DirectionalHistogram` on the lines'
    directions in room coordinates, gives each band of them the rays' distribution
    (:mod:`~compas_acoustics.hybrid.envelope`). The renderer applies both (:meth:`HybridCore.shaped_lines`).

    ``diffraction="udfa"`` (Kirsch and Ewert 2023) covers obstacles and room edges wider than pi
    (:func:`~compas_acoustics.hybrid.diffraction.diffraction_edges`), for the direct sound and, when
    ``max_order >= 1``, one enclosure reflection before or after the edge. ``"btms"`` (Svensson et al. 1999) is
    first order with at most one wall reflection and second order between visible edges, direct paths only, and
    far slower. The late field is not affected.

    ``scattering`` (Ewert et al. 2025, Sec. 3.1.2) needs ``iss.scattered``; ``iss.gain`` is then the specular part.
    The scattered part of every image of orders 1 to ``max_order`` enters the lines of its own hemisphere skipping
    their delays (:func:`run_network`'s ``tap``), through the allpass cascade of Eqs. 12-13 with ``R`` the mean free
    path, weighted as a driving image.
    """
    n_samples = int(np.ceil(duration_s * pair.fs))

    directions = fdn_directions(num_delays)
    delays = [directional_delays(pair, directions, rng=rng + slope) for slope in range(target.n_slopes)]
    builds = design_fdn(target, delays, rng=rng)
    drive_gain = float(np.sqrt(num_delays))

    paths, contributions = (), ()
    if diffraction == "udfa":
        paths = tuple(diffraction_paths(pair, max_order=max_order))
    elif diffraction == "btms":
        contributions = tuple(btms_paths(pair, max_order=max_order))
    if directional_envelope is not None:
        if late_envelope is None:
            raise ValueError("directional_envelope needs late_envelope, the omni reference that sets the level")
        if directional_envelope.energy.shape[0] != num_delays:
            raise ValueError(f"the directional histogram holds {directional_envelope.energy.shape[0]} directions; the network has {num_delays} lines")

    driving_set = _select(iss, iss.order == max_order)
    if driving_set.delay.size == 0:
        warnings.warn(
            f"no image source of order {max_order} reaches the receiver, so nothing drives the late field and "
            "the response has no tail; raise image_source_order, or move the source or receiver out of the "
            "shadow of an obstacle",
            UserWarning,
            stacklevel=2,
        )
    drive = render_images(driving_set, scattering_weights(driving_set.doa, directions, opposite=True), n_samples)
    line_eq = directional_absorption_filters(pair, directions) if directional_absorption else None
    tap = None
    if scattering:
        if iss.scattered is None:
            raise ValueError("scattering needs image sources with scattered parts: pass image_sources the scattered damping")
        reflected = _select(iss, (iss.order >= 1) & (iss.order <= max_order))
        if reflected.delay.size:
            free_path = 4.0 * pair.geometry.volume / pair.geometry.surface_area
            tap = (
                smear(
                    render_images(
                        replace(reflected, gain=reflected.scattered),
                        scattering_weights(reflected.doa, directions),
                        n_samples,
                    ),
                    scattering_decay_time(free_path, pair.speed_of_sound),
                    pair.fs,
                )
                * drive_gain
            )
    return HybridCore(
        pair=pair,
        n_samples=n_samples,
        fs=pair.fs,
        max_order=max_order,
        target=target,
        images=_select(iss, iss.order <= max_order),
        diffraction_paths=paths,
        btms_contributions=contributions,
        builds=builds,
        lines=run_network(builds, drive * drive_gain, tap=tap, line_eq=line_eq),
        line_directions=directions.cartesian / np.linalg.norm(directions.cartesian, axis=1, keepdims=True),
        late_envelope=late_envelope,
        directional_envelope=directional_envelope,
        drive_gain=float(drive_gain),
        driving_images=int(driving_set.delay.size),
    )


def _select(iss: ImageSourceSet, mask: np.ndarray) -> ImageSourceSet:
    """The subset of an image source set selected by a boolean mask."""
    if np.asarray(iss.delay).size == 0:
        return iss  # an empty Coordinates set has no cartesian
    xyz = iss.doa.cartesian[mask]
    return replace(
        iss,
        delay=iss.delay[mask],
        gain=iss.gain[mask],
        doa=pf.Coordinates(xyz[:, 0], xyz[:, 1], xyz[:, 2]),
        order=iss.order[mask],
        scattered=None if iss.scattered is None else iss.scattered[mask],
        wall=None if iss.wall is None else iss.wall[mask],
    )
