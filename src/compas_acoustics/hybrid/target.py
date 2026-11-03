"""The decay the late field must reproduce."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pyFDN

from compas_acoustics.room.diffuse import mean_absorption
from compas_acoustics.room.diffuse import predicted_rt

from .bands import BANDS
from .bands import BANDWIDTHS
from .bands import N_BANDS
from .bands import check_bands

if TYPE_CHECKING:
    from .pair import Pair


@dataclass(frozen=True)
class RoomTarget:
    """What the late field must reproduce: decay and spectral shape per band.

    ``level`` is a relative spectral envelope that only weights the slopes of a multi-slope target against each
    other; the late field's absolute level comes from the image sources driving the network.
    """

    rt: np.ndarray
    level: np.ndarray
    bands: np.ndarray
    fs: float

    def __post_init__(self) -> None:
        check_bands(self.bands)
        for name in ("rt", "level"):
            value = np.asarray(getattr(self, name), dtype=float)
            if value.ndim == 1:
                value = value[:, np.newaxis]
            if value.ndim != 2 or value.shape[0] != N_BANDS:
                raise ValueError(
                    f"{name} must have shape ({N_BANDS},) or ({N_BANDS}, n_slopes)"
                )
            object.__setattr__(self, name, value)
        if self.rt.shape != self.level.shape:
            raise ValueError("rt and level must have the same shape")
        if np.any(self.rt <= 0.0):
            raise ValueError("rt must be positive in every band and slope")

    @property
    def n_slopes(self) -> int:
        return int(self.rt.shape[1])

    @classmethod
    def from_rir(
        cls, rir: np.ndarray, fs: float, *, n_slopes: int = 1, decay_db: float = 30.0
    ) -> RoomTarget:
        """Estimate the target from a reference impulse response.

        ``n_slopes=1`` fits one Schroeder slope from -5 dB over ``decay_db`` (30: T30, 20: T20). The shorter
        range stays above the late tail, where a reference whose decay is carried by few long-lived rays is least
        repeatable. Any other ``n_slopes`` uses DecayFitNet (Goetz et al. 2022): 0 infers one to three slopes per
        band, as coupled rooms need, and 2 or 3 force that many; ``decay_db`` is then ignored.

        A forced count is returned even on a single-slope decay: the estimator cannot decline, so the caller must.
        Slopes are returned fastest first; slopes close in level are not resolved and come back near-identical.
        """
        rir = np.asarray(rir, dtype=float).ravel()
        if n_slopes == 1:
            rt, centres = pyFDN.estimate_rt_bands(rir, fs, decay_db=decay_db)
            level, _ = pyFDN.estimate_initial_level_bands(rir, rt, fs)
            check_bands(centres)
            return cls(
                rt=rt[:, np.newaxis],
                level=level[:, np.newaxis],
                bands=centres,
                fs=float(fs),
            )

        from multislope import DecayFitNet

        fit = DecayFitNet(
            n_slopes=n_slopes, sample_rate=float(fs), filter_frequencies=list(BANDS)
        ).estimate(rir)
        decay_time = np.asarray(fit.t, dtype=float)
        amplitude = (
            np.asarray(fit.a, dtype=float)
            * np.asarray(fit.norm_vals, dtype=float)[0][:, np.newaxis]
        )
        level = pyFDN.slope_amplitude_to_level(amplitude, decay_time, fs)

        # always three columns; a zero decay time marks a slope inactive, kept only if active in every band
        active = np.all(decay_time > 0.0, axis=0)
        decay_time = decay_time[:, active]
        level = level[:, active]

        order = np.argsort(np.median(decay_time, axis=0))
        return cls(
            rt=decay_time[:, order],
            level=level[:, order],
            bands=BANDS.copy(),
            fs=float(fs),
        )

    @classmethod
    def from_room(
        cls,
        spec: Pair,
        *,
        model: str = "eyring",
        weighting: str = "area",
        air_absorption: bool | None = None,
    ) -> RoomTarget:
        """Predict a single-slope target from geometry and materials (:func:`~compas_acoustics.room.diffuse.predicted_rt`).

        ``model`` and ``weighting`` are those of :func:`~compas_acoustics.room.diffuse.predicted_rt` and
        :func:`~compas_acoustics.room.diffuse.mean_absorption`. ``air_absorption`` defaults to
        ``spec.air_absorption``, which keeps the decay and the image source gains consistent.
        """
        if air_absorption is None:
            air_absorption = spec.air_absorption
        alpha = mean_absorption(spec.geometry, weighting=weighting)
        rt = predicted_rt(
            spec.geometry,
            model=model,
            weighting=weighting,
            air_absorption=air_absorption,
            speed_of_sound=spec.speed_of_sound,
            air=spec.air,
        )

        # The diffuse-field energy density goes as 1 / A; ``level`` is a band's decay amplitude, so the square
        # root turns energy into amplitude and BANDWIDTHS a density into a band's energy (without it a flat room
        # would fall 3 dB per octave). Only the shape is used.
        level = np.sqrt(BANDWIDTHS / (alpha * spec.geometry.surface_area))
        return cls(
            rt=rt[:, np.newaxis],
            level=level[:, np.newaxis],
            bands=BANDS.copy(),
            fs=spec.fs,
        )

    def geq_bands(
        self,
        slope: int = 0,
        *,
        dc_factor: float = 1.0,
        nyquist_factor: float = 0.5,
    ) -> np.ndarray:
        """One slope's RT on the ten-band attenuation grid: 31.25 Hz, the eight octaves, 16 kHz.

        The outer bands, held to DC and Nyquist, are scaled by ``dc_factor`` and ``nyquist_factor``; the default
        shortens the top, where air and boundary absorption dominate.
        """
        rt = self.rt[:, slope]
        return np.concatenate(([dc_factor * rt[0]], rt, [nyquist_factor * rt[-1]]))
