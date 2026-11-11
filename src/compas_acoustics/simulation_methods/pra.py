"""The mono simulation method: image sources, optionally with the ray tracer."""

from __future__ import annotations

import warnings
from typing import Optional
from typing import Union

import numpy as np

from compas_acoustics.directivity import is_omnidirectional
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

__all__ = ["Mono"]


class Mono(SimulationMethod):
    """Image sources up to ``image_source_order``, and the stochastic ray tracer past it if ``ray_tracing``.

    Parameters
    ----------
    image_source_order : int, default=3
        Last specular order; with ``ray_tracing`` the ray tracer covers what follows. The search cost grows steeply with the order
        (:func:`~compas_acoustics.simulation_methods.image_source_search_seconds`).
    ray_tracing : bool, default=True
        Materials without scattering coefficients scatter 0.05 in every band.
    air_absorption : bool, default=True
    n_rays : int or None, default=20000
        Per source; None: a heuristic of at least 50000.
    energy_thres : float, default=1e-7
        The product of its reflections in its least absorbed band at which a ray stops; the logged energy is
        that over the squared distance (Schroeder 2011, eq. 5.54), so the tail has decayed much further.
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
    n_jobs : int, default=1
        Processes. Above 1, every source is simulated in a room of its own, several at once, and the processes left
        over split each source's rays (:func:`~compas_acoustics.room.parallel.split_processes`). Image-source
        responses are unchanged; ray-traced ones differ from a single-process run by Monte-Carlo noise and repeat
        for a given ``seed`` and ``n_jobs``.
    diffraction : bool or str, default=False
        True or ``'udfa'`` adds the first-order edge diffraction of the obstacles' edges and the room's re-entrant
        corners (UDFA, Kirsch and Ewert 2023), with at most one wall reflection before or after the edge, as the
        hybrid methods do. Needs spharpy (Python >= 3.11).
    reciprocity : bool, default=False
        With more sources than receivers, all omnidirectional, swap their roles: the receivers emit, which costs one
        room per receiver instead of one per source. Exact for the image sources; the ray-traced part is reciprocal
        statistically (its detection sphere moves to the other end). ``generation["reciprocal"]`` says whether it
        was used.

    Notes
    -----
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

    def __init__(
        self,
        image_source_order: int = 3,
        ray_tracing: bool = True,
        air_absorption: bool = True,
        n_rays: Optional[int] = 20_000,
        energy_thres: float = 1e-7,
        duration: Optional[float] = None,
        receiver_radius: float = 0.5,
        seed: Union[bool, int] = True,
        normalize: bool = True,
        air: Optional[AirConditions] = None,
        n_jobs: int = 1,
        reciprocity: bool = False,
        diffraction=False,
    ):
        seed = _seed(seed)
        model = {False: None, None: None, True: "udfa", "udfa": "udfa"}.get(diffraction, "invalid")
        if model == "invalid":
            raise ValueError(f"diffraction must be False, True or 'udfa' for the mono method (BTMS is the hybrid's), not {diffraction!r}")
        self.diffraction = model
        if int(n_jobs) < 1:
            raise ValueError(f"n_jobs must be at least 1, got {n_jobs!r}")
        self.n_jobs = int(n_jobs)
        self.reciprocity = bool(reciprocity)
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
            "valid_range_hz": None,
            "air": air_or_default(self.air).to_dict(),
            "speed_of_sound": air_or_default(self.air).speed_of_sound,
        }
        if self.ray_tracing:
            generation.update(n_rays=self.n_rays, energy_thres=self.energy_thres, duration=self.duration, receiver_radius=self.receiver_radius, seed=self.seed)
        if self.n_jobs != 1:
            generation["n_jobs"] = self.n_jobs
        if self.diffraction:
            generation["diffraction"] = self.diffraction
        return generation

    def _reciprocal(self, context: SimulationContext) -> bool:
        """Whether the receivers emit: asked for, cheaper (more sources than receivers) and exact (omnidirectional sources)."""
        return self.reciprocity and len(context.sources) > len(context.receivers) and all(is_omnidirectional(s.directivity) for s in context.sources)

    def compute(self, context: SimulationContext) -> dict:
        if not context.sources or not context.receivers:
            return {}
        fs = context.sampling_rate
        _warn_if_the_search_is_slow(context.geometry, self.image_source_order)
        reciprocal = self._reciprocal(context)
        if reciprocal:
            import pyroomacoustics as pra

            from compas_acoustics.room.pra import points_to_array

            emitters = [pra.SoundSource(position=points_to_array(receiver.position)) for receiver in context.receivers]
            mic_positions = np.array([source.position for source in context.sources], dtype=float).T
        else:
            emitters = [source_to_pra(source, fs) for source in context.sources]
            mic_positions = np.array([receiver.position for receiver in context.receivers], dtype=float).T
        from compas_acoustics.worker import jobs

        jobs.progress("ray tracing" if self.ray_tracing else "image sources")
        if self.ray_tracing:
            responses = ray_traced_rirs(
                context.geometry,
                emitters,
                mic_positions,
                fs,
                max_order=self.image_source_order,
                duration=self.duration,
                n_rays=self.n_rays,
                receiver_radius=self.receiver_radius,
                energy_threshold=self.energy_thres,
                air_absorption=self.air_absorption,
                seed=self.seed,
                air=self.air,
                n_jobs=self.n_jobs,
            )
        else:
            responses = image_source_rirs(
                context.geometry, emitters, mic_positions, fs, max_order=self.image_source_order, air_absorption=self.air_absorption, air=self.air, n_jobs=self.n_jobs
            )
        if reciprocal:
            # responses[source][receiver] -> [receiver][source]
            responses = [[responses[s][m] for s in range(len(context.sources))] for m in range(len(context.receivers))]

        if self.diffraction:
            jobs.progress("diffraction")
            responses = self._with_diffraction(context, responses)

        generation = dict(self.generation, scene=getattr(context.scene, "name", None))
        if self.reciprocity:
            generation["reciprocal"] = reciprocal
        rirs = {}
        for m, receiver in enumerate(context.receivers):
            for s, source in enumerate(context.sources):
                jobs.progress("responses", m * len(context.sources) + s, len(context.sources) * len(context.receivers))
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
        if self.ray_tracing and rirs:
            self._warn_if_cut_short(rirs, fs)
        return rirs

    def _with_diffraction(self, context: SimulationContext, responses):
        """``responses[m][s]`` plus every first-order UDFA path (Kirsch and Ewert 2023), as the hybrid renders them.

        The geometric sound and its reflections come from the image sources, which switch off behind an edge; the
        diffracted paths make up that step on both sides of each shadow and reflection boundary, so the sum is
        continuous. Paths have at most one wall reflection, before or after the edge.
        """
        from compas_acoustics.directivity import band_gains
        from compas_acoustics.directivity import is_omnidirectional
        from compas_acoustics.directivity import to_pra
        from compas_acoustics.hybrid.bands import BANDS
        from compas_acoustics.hybrid.diffraction import diffraction_paths
        from compas_acoustics.hybrid.early import render_transfers
        from compas_acoustics.hybrid.pair import Pair

        fs = context.sampling_rate
        air = air_or_default(self.air)
        out = [list(row) for row in responses]
        for s, source in enumerate(context.sources):
            directivity = None if is_omnidirectional(source.directivity) else to_pra(source.directivity, list(source.direction), fs)
            source_gain = None
            if directivity is not None:

                def source_gain(direction, directivity=directivity):
                    return np.broadcast_to(band_gains(directivity, np.asarray(direction, dtype=float).reshape(3, 1))[0], (len(BANDS),))

            for m, receiver in enumerate(context.receivers):
                spec = Pair(
                    geometry=context.geometry,
                    source=np.asarray(source.position, dtype=float),
                    receiver=np.asarray(receiver.position, dtype=float),
                    receiver_rotation=np.eye(3),
                    fs=float(fs),
                    speed_of_sound=air.speed_of_sound,
                    air_absorption=self.air_absorption,
                    air=self.air,
                    require_volume=False,
                )
                paths = diffraction_paths(spec, max_order=min(self.image_source_order, 1), source_gain=source_gain)
                if not paths:
                    continue
                base = np.asarray(out[m][s], dtype=float)
                n = base.size
                if not self.ray_tracing:  # natural length: room for the latest path and its filters
                    n = max(n, int(np.ceil(max(p.delay for p in paths) * fs)) + 1024)
                gains = np.array([p.band_gain / (4.0 * np.pi * p.distance) for p in paths])
                rendered = render_transfers(np.array([p.delay for p in paths]), [p.transfer for p in paths], gains, np.ones((len(paths), 1)), BANDS, fs, n)[:, 0]
                rendered[: base.size] += base
                out[m][s] = rendered
        return out

    @staticmethod
    def _warn_if_cut_short(rirs: dict, fs: int) -> None:
        """Warn when a response ends before 2/3 of its own longest T30.

        The buffer comes from Eyring, which a non-diffuse room can underestimate several times over. T30's fit
        ends 35 dB down, at 0.58 of the reverberation time, so a response shorter than 2/3 of it biases T30 and
        cuts the hybrid's decay fit.
        """
        from compas_acoustics.analyser import Analyser

        longest, length = 0.0, 0.0
        for ir in rirs.values():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                t30 = np.asarray(Analyser(ir).reverberation_times()["T30"], dtype=float)
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
