"""Diffuse-field prediction of a room's reverberation, Sabine and Eyring (Vorlaender 2020, Sec. 4.5.1, Eqs. 4.28 to 4.34).

Sizes ray-traced simulations (:func:`compas_acoustics.room.raytracing.default_duration`) and gives the hybrid
methods a decay target without a reference (``tail="eyring"``, ``"sabine"``).
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import pyrato

from compas_acoustics.room.air import AirConditions
from compas_acoustics.room.air import air_or_default
from compas_acoustics.room.geometry import SYNTHESIS_BANDS

__all__ = ["DiffuseFieldWarning", "air_attenuation", "mean_absorption", "predicted_rt"]


class DiffuseFieldWarning(UserWarning):
    """The room violates the assumptions behind Sabine and Eyring."""


def mean_absorption(geometry, *, weighting: str = "area") -> np.ndarray:
    """Mean absorption coefficient per band, shape (8,).

    ``weighting="area"`` weights each surface by its area, as Sabine and Eyring are derived; ``"surface"`` takes
    the plain mean over the surfaces, which differs whenever the absorbing surfaces are not the largest.
    """
    surfaces = geometry.surfaces
    areas = np.array([area for area, _ in surfaces], dtype=float)
    alphas = np.stack([mat.absorption for _, mat in surfaces], axis=0)
    if weighting == "area":
        return (areas @ alphas) / areas.sum()
    if weighting == "surface":
        return alphas.mean(axis=0)
    raise ValueError(f"weighting must be 'area' or 'surface', got {weighting!r}")


def _absorption_uniformity(geometry) -> np.ndarray:
    """Area-weighted coefficient of variation of absorption per band, shape (8,); zero for one coefficient everywhere."""
    surfaces = geometry.surfaces
    areas = np.array([area for area, _ in surfaces], dtype=float)
    alphas = np.stack([mat.absorption for _, mat in surfaces], axis=0)
    mean = mean_absorption(geometry)
    variance = (areas @ (alphas - mean) ** 2) / areas.sum()
    return np.where(mean > 0.0, np.sqrt(variance) / np.where(mean > 0.0, mean, 1.0), 0.0)


def _band_list(mask: np.ndarray) -> str:
    return ", ".join(f"{b:g}" for b in np.array(SYNTHESIS_BANDS)[mask])


def _warn_if_not_diffuse(geometry, model: str) -> None:
    """Warn with :class:`DiffuseFieldWarning` in the bands where the diffuse-field assumption is doubtful.

    The late field reproduces the predicted decay, right or wrong. The thresholds are the package's own: an
    area-weighted coefficient of variation of absorption above 0.3, or a mean absorption of 0.25 or more.
    """
    mean = mean_absorption(geometry)
    uniformity = _absorption_uniformity(geometry)
    name = model.capitalize()
    uneven_cv, high_absorption = 0.3, 0.25

    uneven = uniformity > uneven_cv
    if np.any(uneven):
        warnings.warn(
            f"{name} assumes a diffuse field, but this room's absorption is unevenly distributed in the "
            f"{_band_list(uneven)} Hz band(s) (area-weighted coefficient of variation up to "
            f"{uniformity[uneven].max():.2f}, warning above {uneven_cv}): the predicted reverberation time may be "
            f"too short. With Binaural or Ambisonics, use tail='ray_tracing' or a reference.",
            DiffuseFieldWarning,
            stacklevel=3,
        )

    absorbing = mean >= high_absorption
    if np.any(absorbing):
        warnings.warn(
            f"this room is too absorbing for {name} in the {_band_list(absorbing)} Hz band(s) (mean absorption up to "
            f"{mean[absorbing].max():.2f}, warning at {high_absorption} and above): the predicted reverberation time may be "
            f"too short. With Binaural or Ambisonics, use tail='ray_tracing' or a reference.",
            DiffuseFieldWarning,
            stacklevel=3,
        )


def air_attenuation(air: Optional[AirConditions] = None) -> np.ndarray:
    """Energy attenuation coefficient ``m`` per metre on ``SYNTHESIS_BANDS`` (8,), ISO 9613-1; default ``AirConditions()``."""
    return air_or_default(air).air_attenuation(np.array(SYNTHESIS_BANDS))


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
    """Reverberation time per band on ``SYNTHESIS_BANDS`` (8,), Sabine or Eyring on :func:`mean_absorption`.

    ``weighting`` is that of :func:`mean_absorption`. With ``air_absorption`` the ``4 m V`` term (Vorlaender 2020,
    Eqs. 4.32 to 4.34) is added in closed form. Warns with :class:`DiffuseFieldWarning` where the room is not diffuse, unless ``warn`` is False.
    ``air`` sets the attenuation and, unless ``speed_of_sound`` is given, the speed of sound. The mean absorption
    is held below 1, so a fully absorbing room gives a finite time rather than a division error.
    """
    if hasattr(pyrato, "parametric"):
        models = {
            "sabine": pyrato.parametric.reverberation_time_sabine,
            "eyring": pyrato.parametric.reverberation_time_eyring,
        }
    else:
        # pyrato < 1.0 (Python 3.9) has no `parametric` module
        models = {"sabine": _sabine, "eyring": _eyring}
    if model not in models:
        raise ValueError(f"model must be one of {sorted(models)}, got {model!r}")
    if not geometry.is_closed:
        raise ValueError("A diffuse-field prediction needs a closed room; the boundary surfaces do not close a volume.")
    alpha = mean_absorption(geometry, weighting=weighting)
    if speed_of_sound is None:
        speed_of_sound = air_or_default(air).speed_of_sound
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
