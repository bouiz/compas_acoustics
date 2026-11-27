""":class:`Binaural` and :class:`Ambisonics`: image sources, optional edge diffraction, a directional FDN.

One image source search serves every pair. A ray-tracing tail's reference shares that search and, with
``late_envelope="directional"``, its ray trace gives the directional histograms; a given ray-tracing ``Mono``
reference at the same image source order shares its search too. Each pair is prepared by
:func:`compas_acoustics.hybrid.synthesis.prepare` and rendered by
:func:`~compas_acoustics.hybrid.binaural.render_binaural` or
:func:`~compas_acoustics.hybrid.ambisonics.render_ambisonics`.
"""

from __future__ import annotations

import os
import threading
import warnings
from abc import abstractmethod
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from dataclasses import replace
from functools import partial
from pathlib import Path
from typing import Callable
from typing import Optional

import numpy as np

from compas_acoustics.directivity import band_gains
from compas_acoustics.directivity import is_omnidirectional
from compas_acoustics.hybrid.ambisonics import default_num_delays
from compas_acoustics.hybrid.ambisonics import render_ambisonics
from compas_acoustics.hybrid.binaural import DEFAULT_HRTF
from compas_acoustics.hybrid.binaural import HrtfSet
from compas_acoustics.hybrid.binaural import render_binaural
from compas_acoustics.hybrid.diffraction import _keep_selection
from compas_acoustics.hybrid.early import ImageSourceSet
from compas_acoustics.hybrid.early import image_sources
from compas_acoustics.hybrid.envelope import histogram_rays
from compas_acoustics.hybrid.late import MIN_DELAYS
from compas_acoustics.hybrid.late import fdn_directions
from compas_acoustics.hybrid.pair import Pair
from compas_acoustics.hybrid.scattering import complementary_scattering
from compas_acoustics.hybrid.synthesis import HybridResult
from compas_acoustics.hybrid.synthesis import prepare
from compas_acoustics.hybrid.target import RoomTarget
from compas_acoustics.ir import ImpulseResponse
from compas_acoustics.receiver import Receiver
from compas_acoustics.room.air import AirConditions
from compas_acoustics.room.air import air_or_default
from compas_acoustics.room.diffuse import predicted_rt
from compas_acoustics.room.pra import directivity_to_pra
from compas_acoustics.room.pra import points_to_array
from compas_acoustics.room.pra import source_to_pra
from compas_acoustics.room.raytracing import DURATION_RT_FACTOR
from compas_acoustics.room.raytracing import MIN_RAYS
from compas_acoustics.room.raytracing import DirectionalHistogram
from compas_acoustics.room.raytracing import default_duration
from compas_acoustics.room.raytracing import default_rays
from compas_acoustics.room.reflections import damping_as_searched
from compas_acoustics.source import Source

from .base import SimulationContext
from .base import SimulationMethod
from .base import _seed
from .base import _warn_if_the_search_is_slow
from .pra import Mono

__all__ = ["Binaural", "Ambisonics", "TAILS"]

_HRTF_LOCK = threading.Lock()

#: Where the decay target of the late field comes from.
TAILS = ("reference", "ray_tracing", "eyring", "sabine")


@dataclass(frozen=True)
class _PairInputs:
    """What one source-receiver pair is rendered from."""

    key: str
    source: Source
    receiver: Receiver
    pair: Pair
    iss: ImageSourceSet
    target: RoomTarget
    reference: Optional[np.ndarray]
    histogram: Optional[DirectionalHistogram]
    duration: float
    # the source pattern's amplitude per band towards a direction, for the diffracted paths; None when omnidirectional
    source_gain: Optional[Callable[[np.ndarray], np.ndarray]] = None


def _without_full_absorption(geometry):
    """``geometry`` with every absorption coefficient at most 1 - 1e-6.

    A fully absorbing surface (an open window, alpha = 1) has no reflection, which the directional absorption filters
    cannot represent when a delay line sees nothing else; 1 - 1e-6 reflects -60 dB.
    """
    materials = {}
    for name, material in geometry.materials.items():
        materials[name] = replace(material, absorption=np.minimum(material.absorption, 1.0 - 1e-6))
    return replace(geometry, materials=materials)


def _emitted_gain(directivity, direction):
    """The source pattern's amplitude per band (8,) towards ``direction`` (3,), for the diffracted paths."""
    return np.broadcast_to(band_gains(directivity, np.asarray(direction, dtype=float).reshape(3, 1))[0], (8,))


class _HybridMethod(SimulationMethod):
    """References, image sources, network and directional histograms; a subclass sets ``num_delays`` and renders."""

    num_delays: int

    def __init__(
        self,
        reference=None,
        tail=None,
        image_source_order=3,
        n_slopes=1,
        decay_db=30.0,
        duration=None,
        n_rays=None,
        directional_absorption=True,
        late_envelope=None,
        diffraction=False,
        air_absorption=None,
        seed=True,
        normalize=True,
        air=None,
        scattering=False,
        n_jobs=1,
        per_pair_seeds=False,
    ):
        if tail is not None and tail not in TAILS:
            raise ValueError(f"tail must be None or one of {TAILS}, not {tail!r}")
        if reference is not None and not isinstance(reference, (str, dict, Mono)):
            raise ValueError("reference must be a method name, a dict of impulse responses or a Mono method")
        if tail == "reference" and reference is None:
            raise ValueError("tail='reference' needs a reference")
        if tail in ("ray_tracing", "eyring", "sabine") and reference is not None:
            raise ValueError(f"tail={tail!r} takes no reference; pass tail='reference' or leave tail out to use it")
        if late_envelope is not None and late_envelope is not False and late_envelope not in ("directional", "omni"):
            raise ValueError(f"late_envelope must be 'directional', 'omni' or False, not {late_envelope!r}")
        if tail in ("eyring", "sabine"):
            if n_slopes != 1:
                raise ValueError(f"an {tail} tail has one decay slope; n_slopes must be 1, not {n_slopes!r}")
            if late_envelope:
                raise ValueError(f"late_envelope follows a reference; an {tail} tail has none")
        model = {False: None, True: "udfa", "udfa": "udfa", "btms": "btms"}.get(diffraction, "invalid")
        if model == "invalid":
            raise ValueError(f"diffraction must be False, True, 'udfa' or 'btms', not {diffraction!r}")
        if image_source_order < 0:
            raise ValueError("image_source_order must be non-negative")
        if duration is not None and duration <= 0:
            raise ValueError("duration must be positive")

        self.reference = reference
        self.tail = tail if tail is not None else ("reference" if reference is not None else "ray_tracing")
        self.image_source_order = int(image_source_order)
        self.n_slopes = n_slopes
        self.decay_db = decay_db
        self.duration = duration
        self.directional_absorption = directional_absorption
        if late_envelope is None:
            late_envelope = False if self.tail in ("eyring", "sabine") else "omni"
        self.late_envelope = late_envelope
        if n_rays is not None and (self.tail in ("eyring", "sabine") or (self.tail == "reference" and late_envelope != "directional")):
            raise ValueError(
                "n_rays sets the rays the method traces itself, the reference of a ray_tracing tail or the directional "
                f"histograms of a given reference; with tail={self.tail!r} and late_envelope={late_envelope!r} it traces none"
            )
        self.n_rays = n_rays
        self.diffraction = model
        self.air_absorption = air_absorption
        self.seed = _seed(seed)
        self.normalize = normalize
        self.air = air
        self.scattering = bool(scattering)
        if int(n_jobs) < 1:
            raise ValueError(f"n_jobs must be at least 1, got {n_jobs!r}")
        self.n_jobs = int(n_jobs)
        self.per_pair_seeds = bool(per_pair_seeds)
        self.reference_rirs = None
        self.results = {}

    @abstractmethod
    def _response(self, core, key, source, receiver, generation) -> tuple[HybridResult, ImpulseResponse]:
        """``core`` rendered in the method's format, and that rendering as the pair's response."""

    def compute(self, context: SimulationContext) -> dict:
        self._check(context)
        # read before any work, so a receiver facing along its up vector (no orientation) fails early
        rotations = [receiver.rotation for receiver in context.receivers]
        geometry = _without_full_absorption(context.geometry)
        _keep_selection(context.geometry, geometry)  # the same faces: a diffraction edge selection still holds
        self.results = {}
        if not context.sources or not context.receivers:
            return {}

        _warn_if_the_search_is_slow(geometry, self.image_source_order)
        from compas_acoustics.worker import jobs

        # With a late envelope the method's own reference is traced with the histogram's ray count in both modes;
        # only "directional" bins the rays on the lines' directions.
        grid = directions = None
        if self.late_envelope:
            grid = fdn_directions(self.num_delays).cartesian
            grid = grid / np.linalg.norm(grid, axis=1, keepdims=True)
            if self.late_envelope == "directional":
                directions = [grid @ rotation.T for rotation in rotations]
        jobs.progress("references")
        references, air_absorption, air, description, run = self._references(context, geometry, grid, directions)
        self.reference_rirs = references

        jobs.progress("image sources")
        room, damping, scattered = self._searched_room(geometry, context, air, run)
        speed_of_sound = float(room.c)
        # The network needs an int, ``rng + slope``: without a seed, a fresh one from the OS
        rng = self.seed if self.seed is not None else int(np.random.default_rng().integers(2**31))
        generation = {
            "method": self.name,
            "scale": "package",
            "tail": self.tail,
            "reference": description,
            "air_absorption": air_absorption,
            "air": air.to_dict(),
            "speed_of_sound": speed_of_sound,
            "image_source_order": self.image_source_order,
            "n_slopes": self.n_slopes,
            "decay_db": self.decay_db,
            "diffraction": self.diffraction,
            "late_envelope": self.late_envelope,
            "directional_absorption": self.directional_absorption,
            "scattering": self.scattering,
            "num_delays": self.num_delays,
            "histogram_rays": None,
            "seed": self.seed,
            "scene": getattr(context.scene, "name", None),
        }
        if self.n_jobs != 1:
            generation["n_jobs"] = self.n_jobs
        if self.per_pair_seeds:
            generation["per_pair_seeds"] = True
        histograms, generation["histogram_rays"] = self._histograms(context, geometry, room, run, grid, directions, air_absorption, air)

        inputs = self._pair_inputs(context, geometry, room, rotations, references, damping, scattered, histograms, air_absorption, air)
        return self._rendered(inputs, rng, generation, context.sampling_rate)

    @staticmethod
    def _check(context: SimulationContext) -> None:
        fs = context.sampling_rate
        geometry = context.geometry
        if fs <= 2.0 * 8000.0 * np.sqrt(2.0):
            raise ValueError(f"the hybrid method needs a sample rate above 22627 Hz, got {fs}: the 8 kHz octave's upper edge must lie below Nyquist for the decay estimator")
        if not geometry.has_volume:
            where = f"; first open edge from {geometry.open_edge[0]} to {geometry.open_edge[1]}" if geometry.open_edge else ""
            raise ValueError(
                f"the hybrid method needs boundary surfaces that close a volume{where}, or a prepared model with its air volume "
                "(RoomGeometry.from_prepared, or RoomGeometry.from_scene(scene, prepare=PrepareOptions()))."
            )
        # Topology alone (unwelded seams, non-manifold edges) is not an acoustic leak: open_area is derived from the
        # preparation's escaping-ray share. Warn only when openings actually absorb.
        if not geometry.is_closed and geometry.open_area > 0.0:
            warnings.warn(
                f"The room does not close: the late field takes the prepared air volume ({geometry.volume:.0f} m^3) and the openings "
                f"({geometry.open_area:.1f} m^2) as fully absorbing. Check the model's leaks before trusting the decay.",
                UserWarning,
                stacklevel=4,
            )
        if context.outside:
            # in a closed room a point outside is a modelling error
            raise ValueError(f"the hybrid method: {context.outside} lie outside the room")

    def _searched_room(self, geometry, context, air, run):
        """``(room, damping, scattered)``: the searched room whose images the pairs take, and per source the specular
        and scattered damping where they are not the room's own (None otherwise)."""
        # A room built with ray_tracing=True holds each image's specular product, prod sqrt(1 - s); built on
        # complementary_scattering it holds the scattered one. Where a reference run has searched the room, its search
        # is reused and the other builds' damping is derived from it.
        early_room = self._built(geometry, context, air, ray_tracing=self.scattering)
        if run is None:
            early_room.image_source_model()
            room, damping = early_room, None
        else:
            room, damping = run.room, damping_as_searched(run.room, early_room)
        scattered = None
        if self.scattering:
            scattered = damping_as_searched(room, self._built(complementary_scattering(geometry), context, air, ray_tracing=True))
        return room, damping, scattered

    def _histograms(self, context, geometry, room, run, grid, directions, air_absorption, air):
        """``(histograms [mic][source] or None, their ray count or None)``: the reference run's, or traced here for
        ``late_envelope="directional"`` with a given reference."""
        if run is not None and run.histograms is not None:
            return run.histograms, int(run.room.rt_args["n_rays"])
        if self.late_envelope != "directional":
            return None, None
        from compas_acoustics.room.raytracing import directional_histograms
        from compas_acoustics.worker import jobs

        jobs.progress("directional histograms")
        tracer = Mono._RAY_TRACING_DEFAULTS
        n_rays = self._rays(room.get_volume(), float(room.c), grid)
        histograms = directional_histograms(
            geometry,
            [source_to_pra(source) for source in context.sources],
            points_to_array([receiver.position for receiver in context.receivers]),
            context.sampling_rate,
            directions,
            max_order=self.image_source_order,
            duration=self.duration,
            n_rays=n_rays,
            receiver_radius=tracer["receiver_radius"],
            energy_threshold=tracer["energy_thres"],
            air_absorption=air_absorption,
            seed=self.seed,
            air=air,
            n_jobs=self.n_jobs,
            per_pair_seeds=self.per_pair_seeds,
        )
        return histograms, n_rays

    def _rays(self, volume, speed_of_sound, grid) -> int:
        """The rays of a trace the method makes itself: ``n_rays``, else those at which a direction of ``grid`` has its
        energy to 1 dB (:func:`~compas_acoustics.hybrid.envelope.histogram_rays`), at least ``MIN_RAYS``; without a
        late envelope on the 96-line grid (:func:`~compas_acoustics.room.raytracing.default_rays`), as ``Mono``."""
        if self.n_rays is not None:
            return int(self.n_rays)
        radius = Mono._RAY_TRACING_DEFAULTS["receiver_radius"]
        if grid is None:
            return default_rays(volume, radius, speed_of_sound)
        return max(histogram_rays(volume, radius, speed_of_sound, grid), MIN_RAYS)

    def _pair_inputs(self, context, geometry, room, rotations, references, damping, scattered, histograms, air_absorption, air):
        """Every source-receiver pair's :class:`_PairInputs`, sources outer, receivers inner."""
        fs = context.sampling_rate
        # The longest Eyring RT, against which a reference's length is checked
        eyring_rt = None if references is None else float(predicted_rt(geometry, air_absorption=air_absorption, air=air, warn=False).max())
        inputs = []
        for s, source in enumerate(context.sources):
            directivity = None if is_omnidirectional(source.directivity) else directivity_to_pra(source.directivity, source.direction, fs)
            source_gain = None if directivity is None else partial(_emitted_gain, directivity)
            if directivity is not None and self.diffraction == "btms":
                warnings.warn(
                    f"Source '{source.name}' is directional, but BTMS diffraction radiates omnidirectionally; diffraction='udfa' applies the pattern.",
                    UserWarning,
                    stacklevel=3,
                )
            for m, receiver in enumerate(context.receivers):
                key = context.key(source, receiver)
                pair = Pair(
                    geometry=geometry,
                    source=np.asarray(source.position, dtype=float),
                    receiver=np.asarray(receiver.position, dtype=float),
                    receiver_rotation=rotations[m],
                    fs=float(fs),
                    speed_of_sound=float(room.c),
                    air_absorption=air_absorption,
                    air=air,
                )
                reference = None
                if references is not None:
                    # Decay target and late envelope need the absolute (package) level
                    reference = np.asarray(references[key].denormalized(), dtype=float)
                    if not np.any(reference):
                        raise ValueError(
                            f"the reference of '{receiver.name}' and '{source.name}' is silent, so it has no decay to fit: image sources "
                            "alone give a receiver that sees none of them no energy. Simulate the reference with ray tracing."
                        )
                    target = RoomTarget.from_rir(reference, fs, n_slopes=self.n_slopes, decay_db=self.decay_db)
                    self._check_reference_length(reference.size / fs, target, eyring_rt, key)
                else:
                    target = RoomTarget.from_room(pair, model=self.tail)
                iss = image_sources(
                    room,
                    s,
                    m,
                    pair,
                    directivity,
                    damping=None if damping is None else damping[s],
                    scattered=None if scattered is None else scattered[s],
                )
                inputs.append(
                    _PairInputs(
                        key=key,
                        source=source,
                        receiver=receiver,
                        pair=pair,
                        iss=iss,
                        target=target,
                        reference=reference,
                        histogram=None if histograms is None else histograms[m][s],
                        # the package's choice of default length: 1.5 times the pair's longest target RT
                        duration=self.duration if self.duration is not None else 1.5 * float(target.rt.max()),
                        source_gain=source_gain,
                    )
                )
        return inputs

    def _rendered(self, inputs, rng, generation, fs) -> dict:
        """The pairs rendered on threads, the responses by key in the pairs' order; :attr:`results` filled."""
        from compas_acoustics.worker import jobs

        def rendered(item):
            core = prepare(
                item.pair,
                item.iss,
                item.target,
                max_order=self.image_source_order,
                duration_s=item.duration,
                num_delays=self.num_delays,
                directional_absorption=self.directional_absorption,
                late_envelope=item.reference if self.late_envelope else None,
                directional_envelope=item.histogram,
                diffraction=self.diffraction,
                rng=rng,
                scattering=self.scattering,
                source_gain=item.source_gain,
            )
            return self._response(core, item.key, item.source, item.receiver, generation)

        # As many threads as keep the pairs' peak memory under 4 GB (about 17 arrays of (n_samples, num_delays) per
        # pair; cap and estimate are the package's); results do not depend on the thread count and keep the pairs'
        # order.
        longest = max(item.duration for item in inputs)
        peak = 17 * np.ceil(longest * fs) * self.num_delays * 8
        workers = max(1, min(len(inputs), os.cpu_count() or 1, int(4e9 // peak)))
        rirs = {}
        with ThreadPoolExecutor(workers) as pool:
            futures = [pool.submit(rendered, item) for item in inputs]
            try:
                for i, (item, future) in enumerate(zip(inputs, futures)):
                    jobs.progress("simulating", i, len(inputs))
                    self.results[item.key], rirs[item.key] = future.result()
            except BaseException:
                pool.shutdown(cancel_futures=True)
                raise
        return rirs

    def _built(self, geometry, context, air, ray_tracing: bool):
        """The room for the image source search, with every source and receiver, not yet searched."""
        room = geometry.to_pra(context.sampling_rate, self.image_source_order, ray_tracing=ray_tracing, air_absorption=False, air=air)
        for source in context.sources:
            room.add_soundsource(source_to_pra(source))
        room.add_microphone_array(points_to_array([receiver.position for receiver in context.receivers]))
        return room

    def _check_reference_length(self, length, target, eyring_rt, key):
        """Warn when a reference ends before 2/3 of its reverberation time (the fitted or Eyring one, the longer).

        The decay fit then shrinks to what the response has and the late envelope reads a cut tail.
        """
        rt = max(float(np.max(target.rt)), float(eyring_rt))
        if length < 2.0 / 3.0 * rt:
            warnings.warn(
                f"The reference of {key} is {length:.1f} s long, less than 2/3 of its reverberation time ({rt:.1f} s): the "
                f"decay fit and the late envelope read a cut tail. Simulate the reference with a duration of at "
                f"least {DURATION_RT_FACTOR * rt:.1f} s.",
                UserWarning,
                stacklevel=3,
            )

    def _references(self, context, geometry, grid, directions):
        """``(references or None, air_absorption, air, description, run)`` for :attr:`tail`.

        ``run`` is the :class:`~compas_acoustics.room.raytracing.RayTracedRun` whose image source search the early
        field reuses: that of the reference the method traces itself, or of a given ray-tracing ``Mono`` at the
        method's image source order; else None. The method's own reference is ``Mono(ray_tracing=True)`` at its image
        source order, seed and air, at first as long as the longer of ``duration`` and the mono default and traced again
        longer while it ends before 2/3 of its T30 (``Mono._run``), and, with a late envelope, traced with the
        histogram's ray count. ``directions`` (per receiver, room coordinates) make that
        trace return the histograms.
        """
        if self.tail in ("eyring", "sabine"):
            absorption = True if self.air_absorption is None else bool(self.air_absorption)
            return None, absorption, air_or_default(self.air), None, None

        run = None
        if self.tail == "ray_tracing":
            absorption = True if self.air_absorption is None else bool(self.air_absorption)
            duration = default_duration(context.geometry, absorption, air=self.air)
            if self.duration is not None:
                duration = max(duration, self.duration)
            volume_room = geometry.to_pra(context.sampling_rate, 0, air=air_or_default(self.air))
            n_rays = self._rays(volume_room.get_volume(), float(volume_room.c), grid)
            method = Mono(
                self.image_source_order,
                ray_tracing=True,
                air_absorption=absorption,
                n_rays=n_rays,
                duration=duration,
                seed=self.seed,
                air=self.air,
                n_jobs=self.n_jobs,
                per_pair_seeds=self.per_pair_seeds,
            )
            references, run = method._run(context, directions, lengthen=True)
            description = method.generation
        elif isinstance(self.reference, str):
            references, description = context.scene.impulse_responses(self.reference), self.reference
        elif isinstance(self.reference, Mono):
            references, run = self.reference._run(context)
            description = self.reference.generation
            if self.reference.image_source_order != self.image_source_order:
                run = None
        else:
            references, description = self.reference, "dict"

        fs = context.sampling_rate
        for source in context.sources:
            for receiver in context.receivers:
                key = context.key(source, receiver)
                if key not in references:
                    hint = (
                        f"; run a method named {self.reference!r} on this scene first"
                        if isinstance(self.reference, str)
                        else "; references are matched by source and receiver, so they must come from this scene"
                    )
                    raise ValueError(f"the reference has no response for '{receiver.name}' and '{source.name}'{hint}")
                response = references[key]
                if response.sampling_rate != fs:
                    raise ValueError(f"reference '{key}' has sample rate {response.sampling_rate}; the simulation runs at {fs}")
                if response.n_channels != 1:
                    raise ValueError(f"reference '{key}' is not a mono response; pass a Mono result")
                if response.generation.get("scale") != "package":
                    raise ValueError(f"reference '{key}' is not on the package scale; pass a Mono result")
        pairs = [references[context.key(s, r)] for s in context.sources for r in context.receivers]
        if any(r.generation.get("method") == "mono" and not r.generation.get("ray_tracing") for r in pairs):
            warnings.warn(
                "the reference is image sources only, without ray tracing: it has no diffuse decay to fit. Use Mono(ray_tracing=True), or tail='eyring'.",
                UserWarning,
                stacklevel=3,
            )

        used = {r.generation.get("air_absorption") for r in pairs}
        if self.air_absorption is None:
            if len(used) != 1 or None in used:
                raise ValueError(f"the references disagree about air_absorption ({used}); pass air_absorption")
            absorption = used.pop()
        else:
            absorption = bool(self.air_absorption)
            if used != {absorption}:
                raise ValueError(f"the references were simulated with air_absorption={used}, not {absorption}; the early field and the decay target must agree")
        conditions = {air_or_default(AirConditions.from_dict(r.generation.get("air"))) for r in pairs}
        if self.air is None:
            if len(conditions) != 1:
                raise ValueError(f"the references disagree about the air ({conditions}); pass air")
            air = conditions.pop()
        else:
            air = self.air
            if conditions != {air}:
                raise ValueError(f"the references were simulated in the air {conditions}, not {air}; the early field and the decay target must share it")
        return references, absorption, air, description, run


_ENGINE_PARAMETERS = """
    tail : {None, "reference", "ray_tracing", "eyring", "sabine"}
        Where the decay target comes from: fitted to ``reference``; fitted to the method's own
        ``Mono(ray_tracing=True)`` run at its ``image_source_order``, ``seed`` and air, kept in :attr:`reference_rirs`;
        or predicted from volume, area and mean absorption (warns where the room is not diffuse). None:
        ``"reference"`` with a reference, else ``"ray_tracing"``.
    image_source_order : int, default=3
        The last order of the early field; its images drive the network.
    n_slopes : int, default=1
        Decay slopes fitted to the reference; 0 lets DecayFitNet (Goetz et al. 2022) decide (needs ``multislope``).
        An Eyring or Sabine tail has one.
    decay_db : float, default=30.0
        Single-slope fit range from -5 dB: 30 is T30, 20 is T20.
    duration : float, optional
        In seconds; default 1.5 times the longest RT of each pair's target (the package's choice). The method's own
        reference lasts at least the mono default and is traced again longer while it ends before 2/3 of its T30; a
        directional histogram traced for a given reference lasts ``duration``, or the mono default. A reference
        shorter than 2/3 of its reverberation time warns.
    n_rays : int, optional
        The rays of every trace the method makes itself: the reference of a ray-tracing tail and the directional
        histograms of a given reference. Default the directions' rule below, at least 20000, which a ``Mono``
        reference also takes by default. Refused when the method traces nothing itself.
    directional_absorption : bool, default=True
        Weight each delay line with the reflection of the boundary it faces.
    late_envelope : {"omni", "directional", False}, optional
        How the late field follows the reference.
        ``"omni"``: each band is attenuated to the reference's late energy; the directions stay the network's.
        ``"directional"``: as ``"omni"``, and per octave band each delay line takes its direction's share of the ray
        energy (directional histogram: Lentz et al. 2007, Sec. 2.4; Schroeder 2011). The rays are traced with
        ``Mono(ray_tracing=True)``'s settings at ``image_source_order``, also with a given reference; their number
        gives a direction's energy a standard deviation of 1 dB (Vorlaender 2020, Eq. 11.20; the 1 dB is the
        package's choice), at least 20000, or is ``n_rays``. A ray-tracing tail traces its reference with that number
        in both modes, so both follow one reference.
        ``False``: the network's own decay to the target, no shaping.
        Default ``"omni"`` with a reference or a ray-tracing tail (the package's choice); with an Eyring or Sabine
        tail the default is ``False`` and the other two raise.
    diffraction : {False, True, "udfa", "btms"}, default=False
        Edge diffraction around obstacles and at room edges wider than 180 deg: ``True`` or ``"udfa"`` (Kirsch and
        Ewert 2023), or ``"btms"``, the BTMS model (Svensson et al. 1999), far slower.
    air_absorption : bool, optional
        On the image sources and diffracted paths. Default the reference's setting, or True; it must agree with the
        reference's, since the early field and the decay target share it.
    seed : bool or int, default=True
        For the feedback matrices, the delay jitter and the rays the method traces: True seeds with 42, an int with
        that value, False or None leaves them unseeded (``generation["seed"] = None``). Binaural and Ambisonics up to
        third order have the same 96 delay lines, and one seed gives them the same network.
    normalize : bool, default=True
        Whether the responses are peak-normalized views.
    air : AirConditions, optional
        Speed of sound and ISO 9613-1 absorption of the image sources, diffracted paths, predicted decay and the
        method's own reference. Default the references' air, else ``AirConditions()``; it must agree with theirs.
    scattering : bool, default=False
        Surface scattering after Ewert et al. (2025): each image keeps its specular part, ``prod sqrt(1 - s)``, in the
        early field; its scattered part, ``prod sqrt(s)``, enters the delay lines of its hemisphere through an allpass
        cascade. Both come from the one image source search.
    n_jobs : int, default=1
        Processes for the reference run (``Mono(n_jobs=...)``, when the method runs it) and the directional
        histograms, whose rays it splits.
    per_pair_seeds : bool, default=False
        Seed the reference run and the directional histograms per source and pair (``Mono(per_pair_seeds=...)``), so
        a pair's response does not depend on the other sources and receivers of the run.

    Attributes
    ----------
    reference_rirs : dict or None
        After :meth:`compute`, the references used: given, read from the graph, or run.
    results : dict
        After :meth:`compute`, the :class:`~compas_acoustics.hybrid.synthesis.HybridResult` per pair.
    num_delays : int
        The delay lines of the network."""


class Binaural(_HybridMethod):
    """Binaural room impulse responses: image sources, optional edge diffraction, a directional FDN.

    The early field is the image sources up to ``image_source_order`` and, with ``diffraction``, the diffracted
    paths. The images of the last order drive a feedback delay network of 96 delay lines, one per direction (the
    count Kirsch et al. 2023, Secs. 1.2 and 3.3, use offline), whose decay follows a target (Wendt et al. 2014,
    Sec. 1.2). Every early path and delay line is filtered with the HRIR measured nearest its direction. Needs
    Python >= 3.11 and a closed room. Responses are ears left then right, stored under ``'binaural'``;
    ``generation["hrtf"]`` names the set.

    Parameters
    ----------
    reference : str, dict, Mono or None
        A package-scale mono response per pair for the decay target and the late envelope: a method's name on
        the IR graph (``"mono"``, ``"data"``), a dict as :meth:`Scene.simulate` returns, or a
        :class:`~compas_acoustics.simulation_methods.Mono` to run first.
    hrtf : str or path, optional
        ``"ku100"`` (None), the Neumann KU 100 set of the HRTF library, or a SOFA file.

    Examples
    --------
    >>> import compas_acoustics as ca
    >>> scene = ca.Scene("office")
    >>> wall = ca.AcousticMaterial("wall", 0.2, 0.1)
    >>> scene.add_acoustic_objects(ca.AcousticObjectsLibrary.shoebox_room(6, 5, 3, materials=wall))
    >>> scene.add_source(ca.Source("talker", 1.5, 1.5, 1.6))
    >>> scene.add_receiver(ca.Receiver("listener", 4.0, 3.0, 1.2))
    >>> ir = next(iter(scene.simulate(ca.Binaural(tail="eyring")).values()))  # doctest: +SKIP
    >>> ir.n_channels, ir.generation["hrtf"]  # doctest: +SKIP
    (2, 'KU100_HRIR_L2702.sofa')

    With a ray-traced reference the decay is fitted to a response you already have.

    >>> reference = scene.simulate(ca.Mono(ray_tracing=True))  # doctest: +SKIP
    >>> responses = scene.simulate(ca.Binaural(reference="mono"))  # doctest: +SKIP
    """

    name = "binaural"
    output = "binaural"
    num_delays = MIN_DELAYS

    def __init__(
        self,
        reference=None,
        hrtf=None,
        *,
        tail=None,
        image_source_order=3,
        n_slopes=1,
        decay_db=30.0,
        duration=None,
        n_rays=None,
        directional_absorption=True,
        late_envelope=None,
        diffraction=False,
        air_absorption=None,
        seed=True,
        normalize=True,
        air=None,
        scattering=False,
        n_jobs=1,
        per_pair_seeds=False,
    ):
        hrtf = "ku100" if hrtf is None else hrtf
        if hrtf != "ku100" and not Path(hrtf).is_file():
            raise ValueError(f"hrtf must be 'ku100' or a SOFA file, not {hrtf!r}")
        super().__init__(
            reference=reference,
            tail=tail,
            image_source_order=image_source_order,
            n_slopes=n_slopes,
            decay_db=decay_db,
            duration=duration,
            n_rays=n_rays,
            directional_absorption=directional_absorption,
            late_envelope=late_envelope,
            diffraction=diffraction,
            air_absorption=air_absorption,
            seed=seed,
            normalize=normalize,
            air=air,
            scattering=scattering,
            n_jobs=n_jobs,
            per_pair_seeds=per_pair_seeds,
        )
        self.hrtf = hrtf
        self._hrtf_set = None
        self._hrtf_loaded = None

    def _response(self, core, key, source, receiver, generation):
        path = DEFAULT_HRTF if self.hrtf == "ku100" else Path(self.hrtf)
        # loaded once per run by the first pair to get here, again only when the set or the rate changed since
        with _HRTF_LOCK:
            if self._hrtf_loaded != (path, int(core.fs)):
                self._hrtf_set, self._hrtf_loaded = HrtfSet.from_sofa(path, int(core.fs)), (path, int(core.fs))
        result = render_binaural(core, self._hrtf_set)
        return result, ImpulseResponse(
            name=key, source=source, receiver=receiver, ir=result.rir, sampling_rate=int(core.fs), normalized=self.normalize, generation=dict(generation, hrtf=path.name)
        )


class Ambisonics(_HybridMethod):
    """Ambisonic room impulse responses: image sources, optional edge diffraction, a directional FDN.

    As :class:`Binaural`, with every early path encoded at its direction and every delay line projected from
    its direction at ``sh_order``. Responses are ACN in ``normalization``, stored under ``'ambisonics'``.

    Parameters
    ----------
    reference : str, dict, Mono or None
        As :class:`Binaural`.
    sh_order : int, default=1
        The network has 96 delay lines, the count Kirsch et al. (2023, Secs. 1.2 and 3.3) use offline, up to third
        order, which the 96-point grid supports; above that the smallest equal-area grid supporting the order. The
        input gain ``sqrt(num_delays)`` offsets the output's ``1 / sqrt(num_delays)``, so the late level does not
        depend on the count.
    normalization : {"SN3D", "N3D"}, default="SN3D"
        ACN channel order, W = 1 for a unit plane wave.

    Examples
    --------
    >>> import compas_acoustics as ca
    >>> scene = ca.Scene("office")
    >>> wall = ca.AcousticMaterial("wall", 0.2, 0.1)
    >>> scene.add_acoustic_objects(ca.AcousticObjectsLibrary.shoebox_room(6, 5, 3, materials=wall))
    >>> scene.add_source(ca.Source("talker", 1.5, 1.5, 1.6))
    >>> scene.add_receiver(ca.Receiver("listener", 4.0, 3.0, 1.2))
    >>> ir = next(iter(scene.simulate(ca.Ambisonics(tail="eyring", sh_order=2)).values()))  # doctest: +SKIP
    >>> ir.n_channels, ir.ambisonic_order  # doctest: +SKIP
    (9, 2)
    """

    name = "ambisonics"
    output = "ambisonic"

    def __init__(
        self,
        reference=None,
        sh_order=1,
        normalization="SN3D",
        *,
        tail=None,
        image_source_order=3,
        n_slopes=1,
        decay_db=30.0,
        duration=None,
        n_rays=None,
        directional_absorption=True,
        late_envelope=None,
        diffraction=False,
        air_absorption=None,
        seed=True,
        normalize=True,
        air=None,
        scattering=False,
        n_jobs=1,
        per_pair_seeds=False,
    ):
        if normalization not in ("SN3D", "N3D"):
            raise ValueError(f"normalization must be 'SN3D' or 'N3D', not {normalization!r}")
        if sh_order < 0:
            raise ValueError("sh_order must be non-negative")
        super().__init__(
            reference=reference,
            tail=tail,
            image_source_order=image_source_order,
            n_slopes=n_slopes,
            decay_db=decay_db,
            duration=duration,
            n_rays=n_rays,
            directional_absorption=directional_absorption,
            late_envelope=late_envelope,
            diffraction=diffraction,
            air_absorption=air_absorption,
            seed=seed,
            normalize=normalize,
            air=air,
            scattering=scattering,
            n_jobs=n_jobs,
            per_pair_seeds=per_pair_seeds,
        )
        self.sh_order = int(sh_order)
        self.normalization = normalization

    @property
    def num_delays(self) -> int:
        """96 up to third order, more above (:func:`~compas_acoustics.hybrid.ambisonics.default_num_delays`)."""
        return default_num_delays(self.sh_order)

    def _response(self, core, key, source, receiver, generation):
        result = render_ambisonics(core, self.sh_order, self.normalization)
        return result, ImpulseResponse(
            name=key,
            source=source,
            receiver=receiver,
            ir=result.rir,
            sampling_rate=int(core.fs),
            normalized=self.normalize,
            generation=dict(generation, sh_order=self.sh_order),
            ambisonic_order=self.sh_order,
            normalization=self.normalization.lower(),
            channel_order="acn",
        )


# under python -OO the docstrings are None
for _method in (Binaural, Ambisonics):
    if _method.__doc__ is not None:
        _method.__doc__ = _method.__doc__.replace("\n\n    Examples\n", _ENGINE_PARAMETERS + "\n\n    Examples\n", 1)
del _method
