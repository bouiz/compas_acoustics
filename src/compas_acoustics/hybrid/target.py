"""The decay the late field must reproduce.

Ported from hybrid-rir (gpapadimitri/hybrid-rir @ 8a70661, src/room.py).
``RoomTarget`` without ``from_ray_tracing`` (the Hybrid method runs its own
reference instead); ``from_room`` takes its reverberation time from
:func:`compas_acoustics.diffuse.predicted_rt`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pyFDN
import pyroomacoustics as pra

from compas_acoustics.diffuse import _air_absorption_coefficient  # noqa: F401  (upstream's import path)
from compas_acoustics.diffuse import mean_absorption
from compas_acoustics.diffuse import predicted_rt

from .bands import BANDS
from .bands import BANDWIDTHS
from .bands import N_BANDS
from .bands import check_bands

if TYPE_CHECKING:
    from .pair import Pair as RoomSpec


@dataclass(frozen=True)
class RoomTarget:
    """What the late field must reproduce: decay and spectral shape per band.

    ``level`` is always a **relative** spectral envelope. The absolute level of
    the late field comes from the image sources that drive the network, never
    from here; ``level`` only weights the slopes of a multi-slope target
    against each other.
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

        ``n_slopes=1`` uses pyFDN's single-slope Schroeder fit, from -5 dB over
        ``decay_db``: 30 is T30, 20 is T20. The shorter range stays above the
        late tail, which in a room whose decay is carried by few long-lived
        rays is where a simulated reference is least repeatable (measured on
        Odeon runs of one room at 20k and 40k rays: T30 moved 16-18 % at
        250-1000 Hz, T20 5-7 %). ``decay_db`` is ignored for more slopes. ``n_slopes=0``
        lets DecayFitNet infer one to three slopes per band, which is what the
        original py-ISM-FDN pipeline did and what coupled rooms need.

        Do not force ``n_slopes=2`` on a room that has one slope. Measured on a
        synthetic 0.8 s decay, forcing two slopes returns 0.626 and 0.661 s:
        it invents a second slope and corrupts the real one. Auto-detection
        returns a single 0.76 s slope. The estimator cannot decline, so the
        caller must.

        Slopes are returned fastest first. Note that two slopes closer than
        about 20 dB in level are not resolvable — the fast one is buried within
        15 dB of the peak — and DecayFitNet will return two near-identical
        values, degenerating to a single slope.
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

        # DecayFitNet always returns three columns; a zero decay time marks an
        # inactive slope. Keep the columns active in every band.
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
        spec: RoomSpec,
        *,
        model: str = "eyring",
        weighting: str = "area",
        air_absorption: bool | None = None,
    ) -> RoomTarget:
        """Predict the target from geometry and materials.

        Always single-slope: a Sabine or Eyring model describes one diffuse
        field, so there is no second decay to predict. Multi-slope targets come
        from :meth:`from_rir`.

        The reverberation model is pyrato's; this method computes the mean
        absorption it needs (:func:`mean_absorption`, area-weighted by default)
        and, when air absorption is active, applies the ``4 m V`` term as a
        closed-form correction to pyrato's output rather than forking the model.

        ``weighting="surface"`` takes the plain mean over the walls. It is not a
        better Eyring: measured against a ray tracer it happens to be closer in the two
        example rooms, because it under-weights the large absorbing surfaces.

        ``air_absorption`` defaults to ``spec.air_absorption``, so one switch
        keeps the predicted decay and the image source gains consistent.
        Passing it explicitly overrides the spec, for isolating the term.
        """
        if air_absorption is None:
            air_absorption = spec.air_absorption
        alpha = mean_absorption(spec.geometry, weighting=weighting)
        # Upstream's RT computation, moved to the base package so that the
        # ray-tracing duration shares it (compas_acoustics.diffuse.predicted_rt).
        rt = predicted_rt(
            spec.geometry,
            model=model,
            weighting=weighting,
            air_absorption=air_absorption,
            speed_of_sound=spec.speed_of_sound,
        )

        # Diffuse-field energy density is proportional to 1 / A, but ``level``
        # is neither an energy nor a spectral density: it is the linear
        # amplitude of one octave band's decay, as pyFDN's
        # ``estimate_initial_level_bands`` reports it. Two conversions follow
        # from that. The square root turns energy into amplitude. The
        # ``BANDWIDTHS`` factor turns a density into a per-band energy,
        # because the estimator integrates over the band and a wide band
        # collects proportionally more of a flat spectrum -- without it, a
        # frequency-flat room would be told to build a late field falling
        # 3 dB per octave, 21 dB across the eight bands. Only the shape is
        # used.
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
        """One slope's RT on the ten-band grid ``pyFDN.decay_to_geq`` expects.

        ``decay_to_geq`` targets 31.25 Hz, the eight octave bands, and 16 kHz,
        holding the outer values to DC and Nyquist. The two extrapolated edges
        are scaled by ``dc_factor`` and ``nyquist_factor``; the default shortens
        the top, where air and boundary absorption dominate.
        """
        rt = self.rt[:, slope]
        return np.concatenate(([dc_factor * rt[0]], rt, [nyquist_factor * rt[-1]]))


def reference_scale(rir: np.ndarray, n_samples: int) -> np.ndarray:
    """A pyroomacoustics response on this package's scale, ``n_samples`` long.

    pyroomacoustics spreads as ``1 / r`` and delays its response by half its
    fractional-delay kernel; this package spreads as ``1 / (4 pi r)`` with the
    direct sound on its geometric sample (upstream ``RoomSpec.ray_traced_rir``).
    """
    latency = pra.constants.get("frac_delay_length") // 2
    rir = np.asarray(rir, dtype=float).ravel()[latency:] / (4.0 * np.pi)
    return np.pad(rir, (0, max(0, n_samples - rir.size)))[:n_samples]
