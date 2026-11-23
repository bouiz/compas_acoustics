"""The decay the late field must reproduce."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pyFDN

from compas_acoustics.room.diffuse import predicted_rt

from .bands import BANDS
from .bands import N_BANDS
from .bands import check_bands

if TYPE_CHECKING:
    from .pair import Pair


@dataclass(frozen=True)
class RoomTarget:
    """What the late field must reproduce: decay and spectral shape per band.

    ``level`` is a relative spectral envelope that only weights the slopes of a multi-slope target against each
    other, so a single slope needs none; the late field's absolute level comes from the image sources driving the
    network.
    """

    rt: np.ndarray
    bands: np.ndarray
    fs: float
    level: np.ndarray | None = None

    def __post_init__(self) -> None:
        check_bands(self.bands)
        for name in ("rt", "level"):
            if getattr(self, name) is None:
                continue
            value = np.asarray(getattr(self, name), dtype=float)
            if value.ndim == 1:
                value = value[:, np.newaxis]
            if value.ndim != 2 or value.shape[0] != N_BANDS:
                raise ValueError(f"{name} must have shape ({N_BANDS},) or ({N_BANDS}, n_slopes)")
            object.__setattr__(self, name, value)
        if self.level is None and self.rt.shape[1] > 1:
            raise ValueError("a target of several slopes needs their level")
        if self.level is not None and self.rt.shape != self.level.shape:
            raise ValueError("rt and level must have the same shape")
        if np.any(self.rt <= 0.0):
            raise ValueError("rt must be positive in every band and slope")

    @property
    def n_slopes(self) -> int:
        return int(self.rt.shape[1])

    @classmethod
    def from_rir(cls, rir: np.ndarray, fs: float, *, n_slopes: int = 1, decay_db: float = 30.0) -> RoomTarget:
        """Estimate the target from a reference impulse response.

        ``n_slopes=1`` fits one slope to the backward-integrated energy decay curve from -5 dB over ``decay_db``
        (30: T30, 20: T20). The shorter range stays above the late tail, where a reference whose decay is carried by
        few long-lived rays is least repeatable. Any other ``n_slopes`` uses DecayFitNet (Goetz et al. 2022): 0
        infers the count per band and keeps the slopes every band has, as coupled rooms need, and 2 or 3 force that
        many; ``decay_db`` is then ignored.

        A forced count is returned even on a single-slope decay: the estimator cannot decline, so the caller must.
        Slopes are returned fastest first; slopes close in level are not resolved and come back near-identical.
        """
        rir = np.asarray(rir, dtype=float).ravel()
        if n_slopes == 1:
            rt, centres = pyFDN.estimate_rt_bands(rir, fs, decay_db=decay_db)
            return cls(rt=rt[:, np.newaxis], bands=centres, fs=float(fs))

        from multislope import DecayFitNet

        fit = DecayFitNet(n_slopes=n_slopes, sample_rate=float(fs), filter_frequencies=list(BANDS)).estimate(rir)
        decay_time = np.asarray(fit.t, dtype=float)
        amplitude = np.asarray(fit.a, dtype=float) * np.asarray(fit.norm_vals, dtype=float)[0][:, np.newaxis]
        level = pyFDN.slope_amplitude_to_level(amplitude, decay_time, fs)

        # three columns for 0, else the forced count; a zero decay time marks a slope inactive, kept only if active
        # in every band. multislope sorts each band's slopes by decay time, so the kept columns are fastest first.
        active = np.all(decay_time > 0.0, axis=0)
        return cls(
            rt=decay_time[:, active],
            level=level[:, active],
            bands=BANDS.copy(),
            fs=float(fs),
        )

    @classmethod
    def from_room(
        cls,
        pair: Pair,
        *,
        model: str = "eyring",
    ) -> RoomTarget:
        """A single-slope target predicted from geometry and materials.

        ``model`` is that of :func:`~compas_acoustics.room.diffuse.predicted_rt`. Air absorption follows
        ``pair.air_absorption``, which keeps the decay and the image source gains consistent.
        """
        rt = predicted_rt(
            pair.geometry,
            model=model,
            air_absorption=pair.air_absorption,
            speed_of_sound=pair.speed_of_sound,
            air=pair.air,
        )
        return cls(rt=rt[:, np.newaxis], bands=BANDS.copy(), fs=pair.fs)

    def geq_bands(self, slope: int = 0) -> np.ndarray:
        """One slope's RT on the ten-band attenuation grid: 31.25 Hz, the eight octaves, 16 kHz.

        The outer bands are held to DC and Nyquist, the top one halved, where air and boundary absorption dominate
        (the package's choice).
        """
        rt = self.rt[:, slope]
        return np.concatenate(([rt[0]], rt, [0.5 * rt[-1]]))
