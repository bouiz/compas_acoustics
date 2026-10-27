"""The mono simulation method: image sources, optionally with the ray tracer."""

from __future__ import annotations

import warnings
from typing import Optional
from typing import Union

import numpy as np

from compas_acoustics.ir import ImpulseResponse
from compas_acoustics.room.air import AirConditions
from compas_acoustics.room.air import air_or_default
from compas_acoustics.room.pra import source_to_pra
from compas_acoustics.room.raytracing import DURATION_RT_FACTOR
from compas_acoustics.room.raytracing import image_source_rirs
from compas_acoustics.room.raytracing import ray_traced_rirs

from .base import SimulationContext
from .base import SimulationMethod
from .base import _seed
from .base import _warn_if_the_search_is_slow

__all__ = ['Mono']


class Mono(SimulationMethod):
    """Image sources up to ``image_source_order``, and the stochastic ray tracer past it if ``ray_tracing``.

    Parameters
    ----------
    image_source_order : int, default=3
        Last specular order. With ray tracing a room with uneven absorption stops depending on it from 6 to 10;
        3 is chosen for the cost of the search: T30 did not move between 3 and 6, EDT by up to 15 % and C80 1 dB.
    ray_tracing : bool, default=True
        Materials without scattering coefficients scatter 0.05 in every band.
    air_absorption : bool, default=True
    n_rays : int or None, default=20000
        Per source; None: a heuristic of at least 50000. The T30 spread across seeds is about 3.9 % at 20000
        and 2.6 % at 50000.
    energy_thres : float, default=1e-7
        The product of its reflections in its least absorbed band at which a ray stops; the logged energy is
        that over the squared distance (Schroeder 2011, eq. 5.54), so the tail has decayed much further. No
        parameter moved beyond the seed spread between 1e-8 and 1e-4; 1e-8 cost 25 % more in a reverberant room.
    duration : float, optional
        Of the ray-traced responses, in seconds; default 1.5 times the longest Eyring reverberation time, within
        0.5 to 12 s (10 s for a scene that closes no volume), with a warning at the cap. Without ray tracing the
        responses have their natural length.
    receiver_radius : float, default=0.5
        In metres.
    seed : bool or int, default=True
        True seeds with 42, an int with that value, False or None leaves the run unseeded
        (``generation["seed"] = None``).
    normalize : bool, default=True
        Whether the responses are peak-normalized views.
    air : AirConditions, optional
        Its speed of sound and, with ``air_absorption``, its ISO 9613-1 absorption; default ``AirConditions()``.

    Ray-tracing settings (``n_rays`` to ``seed``) changed with ``ray_tracing=False`` raise ``ValueError``.

    Examples
    --------
    >>> import compas_acoustics as ca
    >>> scene = ca.Scene("office")
    >>> scene.add_acoustic_objects(ca.AcousticObjectsLibrary.shoebox_room(6, 5, 3, materials=ca.AcousticMaterial("wall", 0.2, 0.1)))
    >>> scene.add_source(ca.Source("talker", 1.5, 1.5, 1.6))
    >>> scene.add_receiver(ca.Receiver("listener", 4.0, 3.0, 1.2))
    >>> responses = scene.simulate(ca.Mono(image_source_order=2, ray_tracing=False))
    >>> ir = next(iter(responses.values()))
    >>> ir.sampling_rate, ir.n_channels, ir.generation["method"]
    (48000, 1, 'mono')

    Ray tracing, on by default, adds the late reflections; the seed makes a run repeatable.

    >>> responses = scene.simulate(ca.Mono(n_rays=2000, duration=0.5))
    >>> ir = next(iter(responses.values()))
    >>> ir.ir.shape, ir.generation["seed"]
    ((24000,), 42)
    """

    name = "mono"
    output = "mono"

    _RAY_TRACING_DEFAULTS = {
        "n_rays": 20_000,
        "energy_thres": 1e-7,
        "duration": None,
        "receiver_radius": 0.5,
        "seed": 42,
    }

    def __init__(self, image_source_order: int = 3, ray_tracing: bool = True, air_absorption: bool = True,
                 n_rays: Optional[int] = 20_000, energy_thres: float = 1e-7, duration: Optional[float] = None,
                 receiver_radius: float = 0.5, seed: Union[bool, int] = True, normalize: bool = True,
                 air: Optional[AirConditions] = None):
        seed = _seed(seed)
        given = {"n_rays": n_rays, "energy_thres": energy_thres, "duration": duration,
                 "receiver_radius": receiver_radius, "seed": seed}
        if not ray_tracing:
            changed = sorted(k for k, v in given.items() if v != self._RAY_TRACING_DEFAULTS[k])
            if changed:
                raise ValueError(f"{changed} only apply with ray_tracing=True")
        if image_source_order < 0:
            raise ValueError(f"image_source_order must be non-negative, got {image_source_order!r}")
        self.image_source_order = int(image_source_order)
        self.ray_tracing = bool(ray_tracing)
        self.air_absorption = bool(air_absorption)
        self.n_rays = n_rays
        self.energy_thres = energy_thres
        self.duration = duration
        self.receiver_radius = receiver_radius
        self.seed = seed
        self.normalize = normalize
        self.air = air

    @property
    def generation(self) -> dict:
        generation = {
            "method": self.name,
            "scale": "package",
            "ray_tracing": self.ray_tracing,
            "image_source_order": self.image_source_order,
            "air_absorption": self.air_absorption,
            "valid_range_hz": None,
            "air": air_or_default(self.air).to_dict(),
            "speed_of_sound": air_or_default(self.air).speed_of_sound,
        }
        if self.ray_tracing:
            generation.update(n_rays=self.n_rays, energy_thres=self.energy_thres, duration=self.duration,
                              receiver_radius=self.receiver_radius, seed=self.seed)
        return generation

    def compute(self, context: SimulationContext) -> dict:
        if not context.sources or not context.receivers:
            return {}
        fs = context.sampling_rate
        _warn_if_the_search_is_slow(context.geometry, self.image_source_order)
        sources = [source_to_pra(source) for source in context.sources]
        mic_positions = np.array([receiver.position for receiver in context.receivers], dtype=float).T
        from compas_acoustics.worker import jobs

        jobs.progress("ray tracing" if self.ray_tracing else "image sources")
        if self.ray_tracing:
            responses = ray_traced_rirs(
                context.geometry, sources, mic_positions, fs,
                max_order=self.image_source_order,
                duration=self.duration,
                n_rays=self.n_rays,
                receiver_radius=self.receiver_radius,
                energy_threshold=self.energy_thres,
                air_absorption=self.air_absorption,
                seed=self.seed,
                air=self.air,
            )
        else:
            responses = image_source_rirs(context.geometry, sources, mic_positions, fs, max_order=self.image_source_order,
                                          air_absorption=self.air_absorption, air=self.air)

        generation = dict(self.generation, scene=getattr(context.scene, "name", None))
        rirs = {}
        for m, receiver in enumerate(context.receivers):
            for s, source in enumerate(context.sources):
                jobs.progress("responses", m * len(context.sources) + s, len(context.sources) * len(context.receivers))
                key = context.key(source, receiver)
                rirs[key] = ImpulseResponse(
                    name=key, source=source, receiver=receiver, ir=responses[m][s], sampling_rate=fs,
                    normalized=self.normalize,
                    generation=generation,
                )
        if self.ray_tracing and rirs:
            self._warn_if_cut_short(rirs, fs)
        return rirs

    @staticmethod
    def _warn_if_cut_short(rirs: dict, fs: int) -> None:
        """Warn when a response ends before 2/3 of its own longest T30.

        The buffer comes from Eyring, which a non-diffuse room can underestimate several times over (0.64 s for
        a furnished office decaying in 2 s). T30's fit ends 35 dB down, at 0.58 of the reverberation time; a decay
        cut at 2/3 of it reads T30 2.4 % short (0.7 % at 0.76, 5 % at 0.6), and cuts the hybrid's decay fit.
        """
        from compas_acoustics.analyser import AcousticAnalyser

        longest, length = 0.0, 0.0
        for ir in rirs.values():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                t30 = np.asarray(AcousticAnalyser(ir).reverberation_times()["T30"], dtype=float)
            if np.any(np.isfinite(t30)):
                longest = max(longest, float(np.nanmax(t30)))
            length = ir.ir.shape[0] / fs
        if longest > 0.0 and length < 2.0 / 3.0 * longest:
            warnings.warn(
                f"The responses are {length:.1f} s long, less than 2/3 of their reverberation time ({longest:.1f} s): T30 reads "
                f"short and the hybrid's decay fit would be cut. Rerun with a duration of at least "
                f"{DURATION_RT_FACTOR * longest:.1f} s; the default buffer comes from Eyring, which this room's decay outruns.",
                UserWarning,
                stacklevel=4,
            )
