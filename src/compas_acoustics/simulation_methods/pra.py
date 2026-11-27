"""The mono simulation method: image sources, optionally with the ray tracer."""

from __future__ import annotations

import warnings
from typing import Optional
from typing import Union

import numpy as np

from compas_acoustics.ir import ImpulseResponse
from compas_acoustics.room.air import AirConditions
from compas_acoustics.room.air import air_or_default
from compas_acoustics.room.pra import points_to_array
from compas_acoustics.room.pra import source_to_pra
from compas_acoustics.room.raytracing import DURATION_RT_FACTOR
from compas_acoustics.room.raytracing import MAX_DURATION
from compas_acoustics.room.raytracing import RayTracedRun
from compas_acoustics.room.raytracing import image_source_rirs
from compas_acoustics.room.raytracing import ray_traced_run

from .base import SimulationContext
from .base import SimulationMethod
from .base import _seed
from .base import _warn_if_the_search_is_slow

__all__ = ["Mono"]


class Mono(SimulationMethod):
    """Image sources up to ``image_source_order`` (pyroomacoustics: Scheibler et al. 2018, Sec. 3) and, if
    ``ray_tracing``, pyroomacoustics' stochastic ray tracer.

    Parameters
    ----------
    image_source_order : int, default=3
        Last image source order. With ``ray_tracing`` the rays also hold this order, the later ones and the
        scattered energy (:mod:`compas_acoustics.room.raytracing`). The search cost grows steeply with the order
        (:func:`~compas_acoustics.simulation_methods.image_source_search_seconds`).
    ray_tracing : bool, default=True
        Materials without scattering coefficients scatter 0.05 in every band. Without it, the settings ``n_rays``
        to ``seed`` raise ``ValueError`` when changed.
    air_absorption : bool, default=True
    n_rays : int, optional
        Per source; default :func:`~compas_acoustics.room.raytracing.default_rays`, from the room's volume (at least
        20000).
    energy_thres : float, default=1e-7
        The product of its reflections in its least absorbed band at which a ray stops; the logged energy is that
        over the squared distance (Schroeder 2011, Eq. 5.54), so the tail has decayed much further.
    duration : float, optional
        Of the ray-traced responses, in seconds. Default: :func:`~compas_acoustics.room.raytracing.default_duration`,
        traced again longer while the responses end before 2/3 of their T30 (up to 12 s). A given duration is kept
        and warns when it cuts the decay. Without ray tracing the responses have their natural length.
    receiver_radius : float, default=0.5
        In metres.
    seed : bool or int, default=True
        True seeds with 42, an int with that value, False or None leaves the run unseeded
        (``generation["seed"] = None``).
    normalize : bool, default=True
        Whether the responses are peak-normalized views.
    air : AirConditions, optional
        Its speed of sound and, with ``air_absorption``, its ISO 9613-1 absorption; default ``AirConditions()``.

    Examples
    --------
    >>> import compas_acoustics as ca
    >>> scene = ca.Scene("office")
    >>> wall = ca.AcousticMaterial("wall", 0.2, 0.1)
    >>> scene.add_acoustic_objects(ca.AcousticObjectsLibrary.shoebox_room(6, 5, 3, materials=wall))
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
        "n_rays": None,
        "energy_thres": 1e-7,
        "duration": None,
        "receiver_radius": 0.5,
        "seed": 42,
    }

    def __init__(
        self,
        image_source_order: int = 3,
        ray_tracing: bool = True,
        air_absorption: bool = True,
        n_rays: Optional[int] = None,
        energy_thres: float = 1e-7,
        duration: Optional[float] = None,
        receiver_radius: float = 0.5,
        seed: Union[bool, int] = True,
        normalize: bool = True,
        air: Optional[AirConditions] = None,
    ):
        seed = _seed(seed)
        given = {"n_rays": n_rays, "energy_thres": energy_thres, "duration": duration, "receiver_radius": receiver_radius, "seed": seed}
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
            "air": air_or_default(self.air).to_dict(),
            "speed_of_sound": air_or_default(self.air).speed_of_sound,
        }
        if self.ray_tracing:
            generation.update(n_rays=self.n_rays, energy_thres=self.energy_thres, duration=self.duration, receiver_radius=self.receiver_radius, seed=self.seed)
        return generation

    def compute(self, context: SimulationContext) -> dict:
        return self._run(context)[0]

    def _run(self, context: SimulationContext, directions=None, lengthen=None) -> tuple[dict, Optional[RayTracedRun]]:
        """The responses and, with ray tracing, the run they came from; ``directions`` as :func:`ray_traced_run`'s.

        With ``lengthen`` (default: when ``duration`` is None) the rays are traced again, at 1.5 times the longest T30
        read, while the responses end before 2/3 of it, up to ``MAX_DURATION``: the default length comes from Eyring,
        which a room with unevenly distributed absorption outlasts. The rule is the package's choice.
        """
        if not context.sources or not context.receivers:
            return {}, None
        fs = context.sampling_rate
        _warn_if_the_search_is_slow(context.geometry, self.image_source_order)
        sources = [source_to_pra(source) for source in context.sources]
        mic_positions = points_to_array([receiver.position for receiver in context.receivers])
        from compas_acoustics.worker import jobs

        if not self.ray_tracing:
            jobs.progress("simulating")
            responses = image_source_rirs(context.geometry, sources, mic_positions, fs, max_order=self.image_source_order, air_absorption=self.air_absorption, air=self.air)
            return self._responses(context, responses, fs), None

        lengthen = self.duration is None if lengthen is None else lengthen
        duration = self.duration
        while True:
            jobs.progress("simulating")
            run = ray_traced_run(
                context.geometry,
                sources,
                mic_positions,
                fs,
                max_order=self.image_source_order,
                duration=duration,
                n_rays=self.n_rays,
                receiver_radius=self.receiver_radius,
                energy_threshold=self.energy_thres,
                air_absorption=self.air_absorption,
                seed=self.seed,
                air=self.air,
                directions=directions,
            )
            rirs = self._responses(context, run.responses, fs)
            length, longest = self._length_and_rt(rirs, fs)
            if not lengthen or length >= 2.0 / 3.0 * longest or (duration or 0.0) >= MAX_DURATION:
                break
            duration = min(DURATION_RT_FACTOR * longest, MAX_DURATION)
        self._warn_if_cut_short(length, longest, lengthen)
        return rirs, run

    def _responses(self, context: SimulationContext, responses, fs: int) -> dict:
        """The :class:`ImpulseResponse` of every pair from ``responses[mic][source]``, keyed by
        :meth:`SimulationContext.key`."""
        from compas_acoustics.worker import jobs

        generation = dict(self.generation, scene=getattr(context.scene, "name", None))
        rirs = {}
        for m, receiver in enumerate(context.receivers):
            for s, source in enumerate(context.sources):
                jobs.progress("simulating", m * len(context.sources) + s, len(context.sources) * len(context.receivers))
                key = context.key(source, receiver)
                rirs[key] = ImpulseResponse(
                    name=key,
                    source=source,
                    receiver=receiver,
                    ir=responses[m][s],
                    sampling_rate=fs,
                    normalized=self.normalize,
                    generation=generation,
                )
        return rirs

    @staticmethod
    def _length_and_rt(rirs: dict, fs: int) -> tuple[float, float]:
        """The responses' length in seconds and their longest T30 (0 when none is finite)."""
        from compas_acoustics.analyser import Analyser

        # one run: every response has its length
        length = next(iter(rirs.values())).ir.shape[0] / fs
        longest = 0.0
        for ir in rirs.values():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                t30 = np.asarray(Analyser(ir).reverberation_times()["T30"], dtype=float)
            if np.any(np.isfinite(t30)):
                longest = max(longest, float(np.nanmax(t30)))
        return length, longest

    @staticmethod
    def _warn_if_cut_short(length: float, longest: float, lengthened: bool) -> None:
        """Warn when the responses end before 2/3 of their longest T30, the package's margin over the 0.58 of it
        that T30's fit range (to -35 dB) needs; ``lengthened`` responses only end so at ``MAX_DURATION``."""
        if longest > 0.0 and length < 2.0 / 3.0 * longest:
            advice = (
                f"The default length stops at {MAX_DURATION:g} s; pass a longer duration to keep the whole decay."
                if lengthened
                else f"Rerun with a duration of at least {DURATION_RT_FACTOR * longest:.1f} s."
            )
            warnings.warn(
                f"The responses are {length:.1f} s long, less than 2/3 of their reverberation time ({longest:.1f} s): T30 reads "
                f"short and the hybrid's decay fit would be cut. {advice}",
                UserWarning,
                stacklevel=5,
            )
