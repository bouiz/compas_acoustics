"""Diffuse-field prediction of a room's reverberation: Sabine and Eyring through pyrato.

Used for the length of a ray-traced simulation (:func:`compas_acoustics.simulation_methods.raytracing.default_duration`)
and, in the hybrid method, for a decay target that needs no reference response
(``Hybrid(tail="eyring")``). Base package: pyrato only, no pyFDN.

``DiffuseFieldWarning`` through ``_warn_if_not_diffuse`` and ``_air_absorption_coefficient``:
Ported from hybrid-rir (gpapadimitri/hybrid-rir @ 8a70661, src/room.py). Two warning texts name this package's API instead of upstream's.
``predicted_rt`` is the reverberation-time part of upstream ``RoomTarget.from_room``.
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import pyrato

from compas_acoustics.room.air import AirConditions
from compas_acoustics.room.air import air_or_default
from compas_acoustics.room.air import speed_of_sound as _speed_of_sound
from compas_acoustics.room.geometry import SYNTHESIS_BANDS

__all__ = ["DiffuseFieldWarning", "air_attenuation", "mean_absorption", "predicted_rt"]

BANDS = np.array(SYNTHESIS_BANDS)
Geometry = object  # upstream's union of room types; here a RoomGeometry


class DiffuseFieldWarning(UserWarning):
    """The room violates the assumptions behind Sabine and Eyring."""


# Both thresholds were calibrated against pyroomacoustics' image source model at
# order 20 plus ray tracing, over four room shapes and scattering coefficients
# from 0 to 0.6, at 1 kHz.
#
# _UNEVEN_CV: holding the area-weighted mean absorption fixed and moving it onto
# the floor and ceiling, Eyring's error grows monotonically with the spread of
# absorption across surfaces -- -1 % at a coefficient of variation of 0.20,
# -14 % at 0.41, -29 % at 0.76, -47 % at 0.96. It crosses 10 % near 0.3.
#
# _HIGH_ABSORPTION: with absorption uniform, Eyring is within 3 % of the
# reference near alpha = 0.12 in every room shape tested, and falls away on both
# sides. Only the high side is warned about, deliberately. Below alpha = 0.1 the
# discrepancy is dominated by the scattering coefficient the reference assumes
# -- at alpha = 0.05 the reference spans 3.13 s at scattering 0 down to 2.08 s
# at 0.3, straddling Eyring's 2.32 s -- and Eyring takes no scattering
# coefficient to account for it. At alpha = 0.25 and above the predicted
# RT is at least 18 % low for every scattering coefficient tested, so there the
# warning holds regardless of that assumption.
_UNEVEN_CV = 0.3
_HIGH_ABSORPTION = 0.25


def mean_absorption(geometry: Geometry, *, weighting: str = "area") -> np.ndarray:
    """Mean absorption coefficient per band, shape (8,).

    This is the adapter between this package's geometry types and pyrato's
    parametric reverberation models, which take a scalar or array of mean
    absorption. The reverberation model itself is not implemented here.

    ``weighting="area"`` weights each surface by its area, which is what
    Sabine and Eyring are derived with. ``weighting="surface"`` takes the plain
    mean over the surfaces; on a shoebox that is the mean over the six walls. It is not the same quantity, and it differs
    whenever the absorbing surfaces are not the largest ones.
    """
    surfaces = geometry.surfaces
    areas = np.array([area for area, _ in surfaces], dtype=float)
    alphas = np.stack([mat.absorption for _, mat in surfaces], axis=0)
    if weighting == "area":
        return (areas @ alphas) / areas.sum()
    if weighting == "surface":
        return alphas.mean(axis=0)
    raise ValueError(f"weighting must be 'area' or 'surface', got {weighting!r}")


def _absorption_uniformity(geometry: Geometry) -> np.ndarray:
    """Area-weighted coefficient of variation of absorption per band, shape (8,).

    Zero when every surface shares one coefficient, which is the case both
    diffuse-field models are derived for.
    """
    surfaces = geometry.surfaces
    areas = np.array([area for area, _ in surfaces], dtype=float)
    alphas = np.stack([mat.absorption for _, mat in surfaces], axis=0)
    mean = mean_absorption(geometry)
    variance = (areas @ (alphas - mean) ** 2) / areas.sum()
    return np.where(
        mean > 0.0, np.sqrt(variance) / np.where(mean > 0.0, mean, 1.0), 0.0
    )


def _band_list(mask: np.ndarray) -> str:
    return ", ".join(f"{b:g}" for b in BANDS[mask])


def _warn_if_not_diffuse(geometry: Geometry, model: str) -> None:
    """Warn when the room breaks the assumption both models rest on.

    Neither failure is fixable here: the models are pyrato's and they are being
    used correctly. What is fixable is that ``from_room`` used to return a
    confidently wrong number with nothing in the return value to show it, and
    the rest of the pipeline then works perfectly on that number -- the FDN
    reproduces it to under 1 %, and the result
    looks entirely plausible.
    """
    mean = mean_absorption(geometry)
    uniformity = _absorption_uniformity(geometry)
    name = model.capitalize()

    uneven = uniformity > _UNEVEN_CV
    if np.any(uneven):
        warnings.warn(
            f"{name} assumes a diffuse field, but this room's absorption is "
            f"unevenly distributed across its surfaces in the "
            f"{_band_list(uneven)} Hz band(s) (area-weighted coefficient of "
            f"variation up to {uniformity[uneven].max():.2f}, warning above "
            f"{_UNEVEN_CV}). Measured against a ray tracer, the predicted "
            f"reverberation time is then up to 47 % low, and the synthesised "
            f"T30 with it. Hybrid(tail='ray_tracing') simulates the decay "
            f"from this same geometry instead and lands within 4 %; "
            f"Hybrid(reference=...) fits it (RoomTarget.from_rir) to a response you already have.",
            DiffuseFieldWarning,
            stacklevel=3,
        )

    absorbing = mean >= _HIGH_ABSORPTION
    if np.any(absorbing):
        warnings.warn(
            f"this room is too absorbing for {name} in the "
            f"{_band_list(absorbing)} Hz band(s) (mean absorption up to "
            f"{mean[absorbing].max():.2f}, warning at {_HIGH_ABSORPTION} and "
            f"above). A diffuse-field model treats every reflection as drawing "
            f"the mean absorption, but the spread of reflection order at a "
            f"given path length makes the real decay slower: measured 18 to "
            f"31 % slower than predicted, for every scattering coefficient "
            f"tested. Use Hybrid(tail='ray_tracing'), which needs the same "
            f"geometry and no measurement, or Hybrid(reference=...) "
            f"(RoomTarget.from_rir) if you have a response already.",
            DiffuseFieldWarning,
            stacklevel=3,
        )


def _air_absorption_coefficient() -> np.ndarray:
    """Energy attenuation coefficient m per metre on ``SYNTHESIS_BANDS``, of the default air.

    Upstream's name for the air when none is given. It was a fixed table, ISO 9613-1
    at 10 C and 70 % rounded, documented as 20 C and 50 %; it is now ISO 9613-1 at
    20 C and 50 % (:class:`~compas_acoustics.room.air.AirConditions`), as every method uses.
    """
    return air_attenuation(None)


def air_attenuation(air: Optional[AirConditions] = None) -> np.ndarray:
    """Energy attenuation coefficient of the air per metre on ``SYNTHESIS_BANDS``, shape (8,).

    ISO 9613-1 at the air's conditions, ``AirConditions()`` (20 C, 50 %) without one.
    """
    return air_or_default(air).air_attenuation(BANDS)


def _infinite_without_absorption(reverberation_time, mean_absorption):
    return np.where(np.isclose(mean_absorption, 0, atol=1e-10, rtol=1e-10), np.inf, reverberation_time)


def _sabine(volume, surface_area, mean_absorption, speed_of_sound):
    mean_absorption = np.asarray(mean_absorption, dtype=float)
    with np.errstate(divide="ignore"):
        rt = 24 * np.log(10) / speed_of_sound * volume / (surface_area * mean_absorption)
    return _infinite_without_absorption(rt, mean_absorption)


def _eyring(volume, surface_area, mean_absorption, speed_of_sound):
    mean_absorption = np.asarray(mean_absorption, dtype=float)
    with np.errstate(divide="ignore"):
        rt = -24 * np.log(10) / speed_of_sound * volume / (surface_area * np.log(1 - mean_absorption))
    return _infinite_without_absorption(rt, mean_absorption)


# pyrato < 1.0 (Python 3.9, Rhino 8) has no `parametric` module
_CLOSED_FORM = {"sabine": _sabine, "eyring": _eyring}


def predicted_rt(
    geometry,
    *,
    model: str = "eyring",
    weighting: str = "area",
    air_absorption: bool = True,
    speed_of_sound: Optional[float] = None,
    air: Optional[AirConditions] = None,
    warn: bool = True,
) -> np.ndarray:
    """Reverberation time per band on ``SYNTHESIS_BANDS``, shape (8,), from Sabine or Eyring.

    The RT of upstream ``RoomTarget.from_room``: pyrato's model on the mean
    absorption (:func:`mean_absorption`), and, with ``air_absorption``, the
    ``4 m V`` term applied in closed form. Warns with :class:`DiffuseFieldWarning`
    where the room breaks the diffuse-field assumption, unless ``warn`` is False.
    ``air`` sets the air's attenuation (see :func:`air_attenuation`) and, unless
    ``speed_of_sound`` is given, the speed of sound.
    The mean absorption is held below 1 so that a fully absorbing room gives a
    finite (zero) time rather than a division error.
    """
    if hasattr(pyrato, "parametric"):
        models = {
            "sabine": pyrato.parametric.reverberation_time_sabine,
            "eyring": pyrato.parametric.reverberation_time_eyring,
        }
    else:
        models = _CLOSED_FORM
    if model not in models:
        raise ValueError(f"model must be one of {sorted(models)}, got {model!r}")
    if not geometry.is_closed:
        raise ValueError("A diffuse-field prediction needs a closed room; the boundary surfaces do not close a volume.")
    alpha = mean_absorption(geometry, weighting=weighting)
    if speed_of_sound is None:
        speed_of_sound = _speed_of_sound(air)
    if warn:
        _warn_if_not_diffuse(geometry, model)
    rt = np.asarray(
        models[model](
            volume=geometry.volume,
            surface_area=geometry.surface_area,
            mean_absorption=np.minimum(alpha, 1.0 - 1e-12),
            speed_of_sound=speed_of_sound,
        ),
        dtype=float,
    )
    if air_absorption:
        # RT = K V / (A + 4 m V) with K = 24 ln(10) / c, so
        # 1/RT_total = 1/RT_surfaces + 4 m / K.
        k = 24.0 * np.log(10.0) / speed_of_sound
        with np.errstate(divide="ignore"):
            rt = 1.0 / (1.0 / rt + 4.0 * air_attenuation(air) / k)
    return rt
