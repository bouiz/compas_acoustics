"""Diffuse-field prediction of a room's reverberation, Sabine and Eyring.

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
    return np.where(
        mean > 0.0, np.sqrt(variance) / np.where(mean > 0.0, mean, 1.0), 0.0
    )


def _band_list(mask: np.ndarray) -> str:
    return ", ".join(f"{b:g}" for b in np.array(SYNTHESIS_BANDS)[mask])


def _warn_if_not_diffuse(geometry, model: str) -> None:
    """Warn when the room breaks the diffuse-field assumption, since the FDN reproduces a wrong target to under 1 %.

    The thresholds were calibrated at 1 kHz against image sources of order 20 with ray tracing, over four room
    shapes and scattering 0 to 0.6. Moving a fixed mean absorption onto floor and ceiling, Eyring's error grows
    with the spread of absorption: -1 % at a coefficient of variation of 0.20, -14 % at 0.41, -29 % at 0.76, -47 %
    at 0.96, crossing 10 % near 0.3. With uniform absorption Eyring is within 3 % near alpha = 0.12; below 0.1 the
    error is dominated by the reference's scattering (at 0.05 it spans 3.13 s at s = 0 to 2.08 s at 0.3 around
    Eyring's 2.32 s), so only the high side warns: from 0.25 the prediction is at least 18 % low at every scattering.
    """
    mean = mean_absorption(geometry)
    uniformity = _absorption_uniformity(geometry)
    name = model.capitalize()
    uneven_cv, high_absorption = 0.3, 0.25

    uneven = uniformity > uneven_cv
    if np.any(uneven):
        warnings.warn(
            f"{name} assumes a diffuse field, but this room's absorption is "
            f"unevenly distributed across its surfaces in the "
            f"{_band_list(uneven)} Hz band(s) (area-weighted coefficient of "
            f"variation up to {uniformity[uneven].max():.2f}, warning above "
            f"{uneven_cv}). Measured against a ray tracer, the predicted "
            f"reverberation time is then up to 47 % low, and the synthesised "
            f"T30 with it. tail='ray_tracing' (Binaural, Ambisonics) simulates the decay "
            f"from this same geometry instead and lands within 4 %; "
            f"reference=... fits it (RoomTarget.from_rir) to a response you already have.",
            DiffuseFieldWarning,
            stacklevel=3,
        )

    absorbing = mean >= high_absorption
    if np.any(absorbing):
        warnings.warn(
            f"this room is too absorbing for {name} in the "
            f"{_band_list(absorbing)} Hz band(s) (mean absorption up to "
            f"{mean[absorbing].max():.2f}, warning at {high_absorption} and "
            f"above). A diffuse-field model treats every reflection as drawing "
            f"the mean absorption, but the spread of reflection order at a "
            f"given path length makes the real decay slower: measured 18 to "
            f"31 % slower than predicted, for every scattering coefficient "
            f"tested. Use tail='ray_tracing' (Binaural, Ambisonics), which needs the same "
            f"geometry and no measurement, or reference=... "
            f"(RoomTarget.from_rir) if you have a response already.",
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

    With ``air_absorption`` the ``4 m V`` term is added in closed form. Warns with :class:`DiffuseFieldWarning`
    where the room is not diffuse, unless ``warn`` is False. ``air`` sets the attenuation and, unless
    ``speed_of_sound`` is given, the speed of sound. The mean absorption is held below 1, so a fully absorbing
    room gives zero rather than a division error.
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
