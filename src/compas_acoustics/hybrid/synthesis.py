"""The hybrid ambisonic room impulse response of one source-receiver pair.

Ported by hand from hybrid-rir (gpapadimitri/hybrid-rir, src/synthesis.py, at
4b42bfd: the btms branch included). The image
source search moved to the caller, which runs one search for every source and
receiver of a scene: ``iss`` holds the pair's images up to ``max_order``.
Upstream's obstacle-free drive is not ported; its scattering is replaced by
surface scattering after Ewert et al. (2025) as RAZR 0.96.3 implements it. ``HybridResult`` also
keeps the early paths (images, UDFA paths, BTMS contributions), which upstream does not.

Deliberately different from upstream: the network is driven by the last
rendered order into the opposite line (see :func:`synthesize`, step 3), after
Wendt et al. (2014), RAZR and Ewert et al. (2025). Checked against the measured
responses of BRAS CR2 (Aspoeck et al.) and against RAZR on exact shoeboxes.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from dataclasses import replace
from typing import Literal

import numpy as np
import pyfar as pf
import pyFDN

from compas_acoustics.simulation_methods.raytracing import DirectionalHistogram

from .diffraction import BtmsContribution
from .diffraction import DiffractionPath
from .diffraction import btms_paths
from .diffraction import diffraction_paths
from .diffraction import encode_btms
from .diffraction import encode_diffraction
from .directional import directional_absorption_filters
from .early import ImageSourceSet
from .early import encode
from .early import render_images
from .scattering import scattering_decay_time
from .scattering import scattering_weights
from .scattering import smear
from .envelope import shape_late
from .envelope import shape_late_directional
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
    late: np.ndarray  # late_lines @ C.T, in ``normalization``
    builds: list[pyFDN.FDNBuild]
    target: RoomTarget
    max_order: int
    driving_images: int
    drive_gain: float
    fs: float
    normalization: str  # "SN3D" or "N3D", audio convention (W = 1 for a unit plane wave)
    # The early paths, for renderers that need each path rather than its encoding (compas_acoustics.hybrid.binaural)
    images: ImageSourceSet | None = None  # orders 0 .. max_order
    diffraction_paths: tuple[DiffractionPath, ...] = ()
    btms_contributions: tuple[BtmsContribution, ...] = ()
    # The late field before the output matrix, for renderers that need each line's direction (compas_acoustics.hybrid.binaural)
    late_lines: np.ndarray | None = None  # (n, num_delays), each line as heard from its direction
    line_directions: np.ndarray | None = None  # (num_delays, 3), unit vectors in the receiver's frame


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
    directional_envelope: DirectionalHistogram | None = None,
    diffraction: bool | Literal["udfa", "btms"] = False,
    normalization: Literal["SN3D", "N3D"] = "SN3D",
    rng: int = 0,
    scattering: bool = False,
) -> HybridResult:
    """Synthesise a hybrid ambisonic room impulse response.

    After Wendt, van de Par and Ewert (2014) and Kirsch et al. (2021, 2023),
    with ambisonic output:

    1. Images up to ``max_order``, encoded at their directions of arrival, are
       the early field. Images of higher orders in ``iss`` are ignored.
    2. An FDN with one delay line per direction of an equal-area grid; its
       output matrix holds the spherical harmonics of those directions, its
       attenuation filters follow ``target.rt``, and each line is the room's
       free path along its direction (:func:`directional_delays`).
    3. The images of order ``max_order``, the last order rendered, also enter
       the network, each spread over the delay lines of the hemisphere opposite
       its direction of arrival, weighted by the cosine to each line
       (:func:`~compas_acoustics.hybrid.scattering.scattering_weights` with
       ``opposite``), so the network's first output is the next order, one
       free path later, off the far side of the room. Wendt et al. (2014,
       Sec. 1.3): the network input "must equal the total energy of the
       Nth-order reflections"; RAZR 0.96.3 feeds that order
       (``image_source_model.m``), as do Kirsch et al. (2023) and Ewert et al.
       (2025, Sec. 3.1.2), whose mapping this is (RAZR's
       ``fdn_smart_ch_mapping``; its default, Wendt's, is by image index).
       hybrid-rir @ a272da7 fed order ``max_order + 1`` without rendering it,
       into the one line nearest the opposite direction; neither has a
       reference, and the extra order costs one more order of image source
       search.
    4. Each line's output is weighted with the reflection of the boundary in its
       direction (``directional_absorption``), and early field and network
       output are summed. The lines are kept (``late_lines``, with
       ``line_directions``): the ambisonic late field is their projection
       through the output matrix, and a binaural renderer filters each with
       the HRIR of its direction instead
       (:func:`compas_acoustics.hybrid.binaural.binaural_rir`).

    ``drive_gain`` defaults to ``sqrt(num_delays)``. ``late_envelope`` is an
    omnidirectional reference on the package scale, normally the pair's
    ray-traced response; the network output is attenuated per band to follow
    its late energy (Vorlaender 2020, Sec. 11.9), every line by the gains of the
    late field's W channel. ``directional_envelope``, a
    :class:`~compas_acoustics.simulation_methods.raytracing.DirectionalHistogram` binned on the
    lines' directions in room coordinates (their grid turned by the receiver's
    rotation), then gives the lines the rays' distribution over directions
    (:func:`~compas_acoustics.hybrid.envelope.shape_late_directional`, RAVEN's
    directivity groups); it needs ``late_envelope``. ``diffraction=True`` adds
    edge diffraction to the early field (Kirsch and Ewert 2023) around the
    obstacles and at the edges of the room's boundary that open wider than
    pi (see :func:`compas_acoustics.hybrid.diffraction.diffraction_edges`), for
    the direct sound and, when ``max_order >= 1``, one enclosure reflection
    before or after the edge; the late field is not affected.
    ``diffraction="btms"`` renders first- and second-order diffraction with the
    BTMS model (Svensson et al. 1999, kernels ported from EDtoolbox v0.504):
    first order with at most one wall reflection, second order between any two
    visible edges (obstacle or room boundary), direct paths only; slow, minutes
    per source-receiver pair. ``True`` is ``"udfa"``.
    ``scattering`` adds surface scattering after Ewert et al. (2025), as RAZR
    0.96.3 implements it (``tests/hybrid/data/razr_scattering``): ``iss`` must
    carry ``scattered`` (see :func:`~compas_acoustics.hybrid.early.image_sources`),
    and its gains, which render the early field and drive the network, are
    then the specular parts. The scattered part of every image of orders 1 to
    ``max_order``, the last included, enters "those FDN channels on the same
    hemisphere (initially skipping the FDN delays ...)" (Sec. 3.1.2):
    smeared by the allpass cascade of Eqs. 12-13, ``R`` the mean free path,
    and weighted onto the lines by
    :func:`~compas_acoustics.hybrid.scattering.scattering_weights`, RAZR's
    ``fdn_smart_ch_mapping`` rule, whose unit l2 norm is that of one driving
    image, times the same ``drive_gain``. It is ``render``'s ``tap``: heard
    through each line's attenuation filters and then recirculating, as RAZR's
    input placed one delay early is. The specular drive takes the same rule
    with the opposite hemisphere (step 3), so both inputs enter as RAZR's
    smart mapping places them.
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

    early_set = _select(iss, iss.order <= max_order)
    early = encode(early_set, sh_order, n_samples)
    driving_set = _select(iss, iss.order == max_order)
    if driving_set.delay.size == 0:
        warnings.warn(
            f"no image source of order {max_order} reaches the receiver, so nothing drives the late field and "
            "the response has no tail; raise image_source_order, or move the source or receiver out of the "
            "shadow of an obstacle",
            UserWarning,
            stacklevel=2,
        )
    paths, contributions = (), ()
    if model == "udfa":
        paths = tuple(diffraction_paths(spec, max_order=min(max_order, 1)))
        early = early + encode_diffraction(list(paths), sh_order, n_samples, spec.fs)
    elif model == "btms":
        contributions = tuple(btms_paths(spec, max_order=min(max_order, 1)))
        early = early + encode_btms(list(contributions), sh_order, n_samples, spec.fs)
    drive = render_images(
        driving_set, scattering_weights(driving_set.doa, directions, opposite=True), n_samples
    )
    line_eq = (
        directional_absorption_filters(spec, directions)
        if directional_absorption
        else None
    )
    tap = None
    if scattering:
        if iss.scattered is None:
            raise ValueError("scattering needs image sources with scattered parts: pass image_sources a scattered_room")
        reflected = _select(iss, (iss.order >= 1) & (iss.order <= max_order))
        if reflected.delay.size:
            free_path = 4.0 * spec.geometry.volume / spec.geometry.surface_area
            tap = smear(
                render_images(
                    replace(reflected, gain=reflected.scattered),
                    scattering_weights(reflected.doa, directions),
                    n_samples,
                ),
                scattering_decay_time(free_path, spec.speed_of_sound),
                spec.fs,
            ) * drive_gain
    lines = render(builds, drive * drive_gain, tap=tap, line_eq=line_eq)
    C = builds[0].C
    if directional_envelope is not None:
        if late_envelope is None:
            raise ValueError("directional_envelope needs late_envelope, the omni reference that sets the level")
        if directional_envelope.energy.shape[0] != num_delays:
            raise ValueError(
                f"the directional histogram holds {directional_envelope.energy.shape[0]} directions; the network has {num_delays} lines"
            )
        lines = shape_late_directional(lines, early[:, 0], lines @ C[0], late_envelope, directional_envelope, spec.fs)
    elif late_envelope is not None:
        lines = shape_late(lines, early[:, 0], lines @ C[0], late_envelope, spec.fs)

    early, late = (_from_n3d(x, normalization) for x in (early, lines @ C.T))
    unit = directions.cartesian / np.linalg.norm(directions.cartesian, axis=1, keepdims=True)
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
        images=early_set,
        diffraction_paths=paths,
        btms_contributions=contributions,
        late_lines=lines,
        line_directions=unit,
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
    if np.asarray(iss.delay).size == 0:
        return iss  # pyfar cannot read the coordinates of an empty set
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
