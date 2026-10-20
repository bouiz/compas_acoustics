"""The pyroomacoustics simulation method: image sources, optionally with the ray tracer."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pyroomacoustics as pra

from compas_acoustics.air import AirConditions
from compas_acoustics.air import air_or_default
from compas_acoustics.ir import ImpulseResponse
from compas_acoustics.pra_converter import source_to_pra
from compas_acoustics.raytracing import _on_package_scale
from compas_acoustics.raytracing import ray_traced_rirs
from compas_acoustics.room_geometry import pra_octave_grid

from .base import SimulationContext
from .base import SimulationMethod
from .base import _warn_if_the_search_is_slow

__all__ = ['Pyroomacoustics']


class Pyroomacoustics(SimulationMethod):
    """pyroomacoustics: image sources up to ``image_source_order``, plus ray tracing if ``ray_tracing``.

    There is no separate image-source method: like pyroomacoustics itself, this
    runs the image source model, and adds the stochastic ray tracer for
    everything past ``image_source_order`` when ``ray_tracing`` is True. The
    ray tracer is hybrid-rir's (:mod:`compas_acoustics.raytracing`).

    Parameters
    ----------
    image_source_order : int, default=3
        Last specular order. With ray tracing, hybrid-rir measured 6 to 10 as
        the range where the decay of a room with uneven absorption stops
        depending on it. The default 3 is chosen for the cost of the image
        source search: with ray tracing, T30 did not move between orders 3 and
        6 in the evaluation, EDT by up to 15 % and C80 by 1 dB.
    ray_tracing : bool, default=True
        Add the ray tracer; ``False`` runs the image sources alone. Materials without scattering coefficients scatter 0.05 in every band.
    air_absorption : bool, default=True
        Air absorption in the simulation.
    n_rays : int or None, default=20000
        Rays per source; ``None`` takes pyroomacoustics' heuristic, at least
        50000. The T30 spread across seeds is about 3.9 % at 20000 rays and
        2.6 % at 50000.
    energy_thres : float, default=1e-8
        Energy at which a ray is stopped.
    duration : float, optional
        Length of the ray-traced responses in seconds; by default 2.5 times the
        longest Eyring reverberation time, within 0.5 to 12 s (10 s for a scene
        that does not close a volume). Without ray tracing the responses have
        their natural length.
    receiver_radius : float, default=0.5
        Radius of the ray-tracing receivers in metres.
    seed : int, default=42
        Seeds pyroomacoustics' generator, so that the same scene gives the same responses (``None`` is 42 too).
        The T30 spread between seeds is the ray tracer's own variance: change the seed to see it.
    normalize : bool, default=True
        Whether the returned responses are peak-normalized views.
    air : AirConditions, optional
        The air: its speed of sound and, with ``air_absorption``, its ISO 9613-1
        absorption. Without one, ``AirConditions()``: 20 C, 50 %, 101325 Pa.

    The ray-tracing settings (``n_rays`` to ``seed``) given away from their
    defaults with ``ray_tracing=False`` raise a ``ValueError`` rather than
    being ignored.

    Examples
    --------
    >>> import compas_acoustics as ca
    >>> scene = ca.Scene("office")
    >>> scene.add_acoustic_objects(ca.AcousticObjectsLibrary.shoebox_room(6, 5, 3, materials=ca.AcousticMaterial("wall", 0.2, 0.1)))
    >>> scene.add_source(ca.Source("talker", 1.5, 1.5, 1.6))
    >>> scene.add_receiver(ca.Receiver("listener", 4.0, 3.0, 1.2))
    >>> responses = scene.simulate(ca.Pyroomacoustics(image_source_order=2, ray_tracing=False))
    >>> ir = next(iter(responses.values()))
    >>> ir.sampling_rate, ir.n_channels, ir.generation["method"]
    (48000, 1, 'pyroomacoustics')

    Ray tracing, on by default, adds the late reflections; the seed makes a run repeatable.

    >>> responses = scene.simulate(ca.Pyroomacoustics(n_rays=2000, duration=0.5))
    >>> ir = next(iter(responses.values()))
    >>> ir.ir.shape, ir.generation["seed"]
    ((24000,), 42)
    """

    name = "pyroomacoustics"
    output = "mono"

    _RAY_TRACING_DEFAULTS = {
        "n_rays": 20_000,
        "energy_thres": 1e-8,
        "duration": None,
        "receiver_radius": 0.5,
        "seed": 42,
    }

    def __init__(self, image_source_order: int = 3, ray_tracing: bool = True, air_absorption: bool = True,
                 n_rays: Optional[int] = 20_000, energy_thres: float = 1e-8, duration: Optional[float] = None,
                 receiver_radius: float = 0.5, seed: int = 42, normalize: bool = True,
                 air: Optional[AirConditions] = None):
        seed = 42 if seed is None else int(seed)
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
            responses = self._image_sources(context.geometry, sources, mic_positions, fs)

        generation = dict(self.generation, scene=getattr(context.scene, "name", None))
        rirs = {}
        for m, receiver in enumerate(context.receivers):
            for s, source in enumerate(context.sources):
                key = context.key(source, receiver)
                rirs[key] = ImpulseResponse(
                    name=key, source=source, receiver=receiver, ir=responses[m][s], sampling_rate=fs,
                    normalized=self.normalize,
                    generation=generation,
                )
        return rirs

    def _image_sources(self, geometry, sources, mic_positions, fs) -> list:
        """Image-source responses ``[mic][source]`` on the package scale, at their natural length."""
        with pra_octave_grid():
            room = geometry.to_pra(fs, self.image_source_order, air_absorption=self.air_absorption, air=self.air)
            for source in sources:
                room.add_soundsource(source)
            room.add_microphone_array(pra.MicrophoneArray(mic_positions, fs))
            room.image_source_model()
            room.compute_rir()
        latency = pra.constants.get("frac_delay_length") // 2
        return [
            [_on_package_scale(room.rir[m][s], latency, len(room.rir[m][s]) - latency) for s in range(len(sources))]
            for m in range(mic_positions.shape[1])
        ]
