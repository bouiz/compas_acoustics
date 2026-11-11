"""Monaural responses from image sources alone or with ray tracing, and the rays' energy per direction.

The backend of :class:`~compas_acoustics.simulation_methods.Mono` and of the hybrid methods' reference runs and
directional histograms. With ray tracing, specular reflections up to ``max_order`` come from the image sources and
everything later from pyroomacoustics' stochastic ray tracer, which also carries the scattered energy. Responses are
on the package scale: spreading ``1 / (4 pi r)``, the direct sound on its geometric sample (pyroomacoustics spreads
as ``1 / r`` and delays by half its fractional-delay kernel).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import List
from typing import Optional
from typing import Sequence

import numpy as np
import pyroomacoustics as pra

from .diffuse import predicted_rt
from .geometry import SYNTHESIS_BANDS
from .geometry import RoomGeometry
from .geometry import pra_octave_grid

__all__ = ["MIN_RAYS", "MAX_DURATION", "DURATION_RT_FACTOR", "DirectionalHistogram", "default_duration", "directional_histograms", "image_source_rirs", "ray_traced_rirs"]

# Floor on the heuristic ray count, applied only with n_rays=None
MIN_RAYS = 50_000
# A response lasts this many times the longest Eyring RT, within 0.5 s and MAX_DURATION. The margin over 1 keeps the
# T30 evaluation range inside the response where Eyring underestimates the decay; a room whose decay outlasts it is
# caught by the methods' length warnings.
DURATION_RT_FACTOR = 1.5
MAX_DURATION = 12.0
_HIST_BIN = 0.004


def default_duration(geometry: RoomGeometry, air_absorption: bool, speed_of_sound: Optional[float] = None, air=None, factor: float = DURATION_RT_FACTOR) -> float:
    """How long to simulate this room: ``factor`` times the longest Eyring RT, within 0.5 s and ``MAX_DURATION``.

    Eyring only sizes the buffer, so its diffuse-field check is skipped. A scene that closes no volume gets 10 s.
    Hitting ``MAX_DURATION`` warns: the rays then live for hundreds of reflections, and the cause is usually a
    large surface without low-frequency absorption.
    """
    if not geometry.is_closed:
        return 10.0
    rt = predicted_rt(geometry, air_absorption=air_absorption, speed_of_sound=speed_of_sound, air=air, warn=False)
    longest = int(np.argmax(rt))
    if factor * rt[longest] > MAX_DURATION:
        # the 62.5 Hz band holds the 125 Hz coefficients
        band = f"{SYNTHESIS_BANDS[longest]:g} Hz" + (" (the 125 Hz coefficients held)" if longest == 0 else "")
        warnings.warn(
            f"Predicted reverberation time {rt[longest]:.1f} s at {band}: the response is capped at "
            f"{MAX_DURATION:g} s and the ray tracing will take seconds per source. Check the low-frequency absorption of the "
            "large surfaces, or pass duration= to size the response yourself.",
            UserWarning,
            stacklevel=2,
        )
    return float(np.clip(factor * rt[longest], 0.5, MAX_DURATION))


def _on_package_scale(rir, latency: int, n_samples: int) -> np.ndarray:
    rir = np.asarray(rir, dtype=float)[latency:] / (4.0 * np.pi)
    return np.pad(rir, (0, max(0, n_samples - rir.size)))[:n_samples]


def image_source_rirs(
    geometry: RoomGeometry, sources, mic_positions, fs: int, *, max_order: int, air_absorption: bool = True, air=None, n_jobs: int = 1
) -> List[List[np.ndarray]]:
    """Image-source responses ``[mic][source]`` on the package scale at their natural length; arguments as :func:`ray_traced_rirs`.

    With ``n_jobs`` > 1 and several sources, each source's image sources are searched in a process of its own; the
    responses are the same as in one process.
    """
    mic_positions = np.asarray(mic_positions, dtype=float).reshape(3, -1)
    if n_jobs > 1 and len(sources) > 1:
        return _by_emitter(geometry, sources, mic_positions, fs, max_order, air_absorption, air, None, None, n_jobs)
    with pra_octave_grid():
        room = geometry.to_pra(fs, max_order, air_absorption=air_absorption, air=air)
        for source in sources:
            room.add_soundsource(source)
        room.add_microphone_array(pra.MicrophoneArray(mic_positions, fs))
        room.image_source_model()
        room.compute_rir()
    latency = pra.constants.get("frac_delay_length") // 2
    return [[_on_package_scale(room.rir[m][s], latency, len(room.rir[m][s]) - latency) for s in range(len(sources))] for m in range(mic_positions.shape[1])]


def ray_traced_rirs(
    geometry: RoomGeometry,
    sources,
    mic_positions,
    fs: int,
    *,
    max_order: int = 6,
    duration: Optional[float] = None,
    n_rays: Optional[int] = 20000,
    receiver_radius: float = 0.5,
    energy_threshold: float = 1e-7,
    air_absorption: bool = True,
    seed: Optional[int] = None,
    air=None,
    n_jobs: int = 1,
) -> List[List[np.ndarray]]:
    """Monaural responses of every source at every microphone, ``[mic][source]``.

    ``sources`` are ``pra.SoundSource``; ``mic_positions`` is ``(3, M)``. ``max_order`` splits specular from
    ray-traced energy. ``duration`` defaults to :func:`default_duration`; ``n_rays=None`` takes a heuristic raised to ``MIN_RAYS``;
    ``seed`` None draws a fresh one from the OS. ``air`` sets the speed of sound and the air absorption.

    ``n_jobs`` > 1 runs the sources in processes of their own and splits each source's rays over the processes left
    (:func:`~compas_acoustics.room.parallel.split_processes`). Each source then draws its own seed from ``seed``,
    so the responses differ from a run in one process by Monte-Carlo noise only, and repeat for a given ``seed``
    and ``n_jobs``.
    """
    if duration is None:
        duration = default_duration(geometry, air_absorption, air=air)
    if duration <= 0.0:
        raise ValueError(f"duration must be positive, got {duration!r}")
    mic_positions = np.asarray(mic_positions, dtype=float).reshape(3, -1)
    options = {
        "receiver_radius": receiver_radius,
        "energy_thres": energy_threshold,
        "time_thres": float(duration),
        "hist_bin_size": _HIST_BIN,
    }
    if n_jobs > 1:
        tracing = {"n_rays": n_rays, "options": options, "n_samples": round(duration * fs)}
        return _by_emitter(geometry, sources, mic_positions, fs, max_order, air_absorption, air, tracing, seed, n_jobs)
    with pra_octave_grid():
        # The numpy seed only: pyroomacoustics derives the tracer's seed from it. None reseeds both from the OS.
        # Inside the lock, so that no other thread reseeds or draws between the seed and the tracing.
        pra.random.seed(seed)
        room = geometry.to_pra(fs, max_order, ray_tracing=True, air_absorption=air_absorption, air=air)
        for source in sources:
            room.add_soundsource(source)
        room.add_microphone_array(pra.MicrophoneArray(mic_positions, fs))
        room.set_ray_tracing(n_rays=n_rays, **options)
        if n_rays is None and room.rt_args["n_rays"] < MIN_RAYS:
            room.set_ray_tracing(n_rays=MIN_RAYS, **options)
        room.compute_rir()

    latency = pra.constants.get("frac_delay_length") // 2
    n_samples = round(duration * fs)
    return [[_on_package_scale(room.rir[m][s], latency, n_samples) for s in range(len(sources))] for m in range(mic_positions.shape[1])]


def _emitter_rirs(geometry, source, mic_positions, fs, max_order, air_absorption, air, tracing, seed, ray_processes, progress=None):
    """One source's package-scale responses at every microphone, ``[mic]``: its own room, its rays maybe over processes."""
    from .parallel import SLICES_PER_PROCESS
    from .parallel import engine_spec
    from .parallel import room_responses
    from .parallel import source_rays
    from .parallel import trace_parallel

    with pra_octave_grid():
        pra.random.seed(seed)
        room = geometry.to_pra(fs, max_order, ray_tracing=tracing is not None, air_absorption=air_absorption, air=air)
        room.add_soundsource(source)
        room.add_microphone_array(pra.MicrophoneArray(mic_positions, fs))
        latency = pra.constants.get("frac_delay_length") // 2
        if tracing is None:
            room.image_source_model()
            room.compute_rir()
            return [_on_package_scale(r[0], latency, len(r[0]) - latency) for r in room.rir]
        room.set_ray_tracing(n_rays=tracing["n_rays"], **tracing["options"])
        if tracing["n_rays"] is None and room.rt_args["n_rays"] < MIN_RAYS:
            room.set_ray_tracing(n_rays=MIN_RAYS, **tracing["options"])
        if ray_processes <= 1:
            room.compute_rir()
            rirs = [r[0] for r in room.rir]
        else:
            room.image_source_model()
            n_bands = len(room.octave_bands.get_bw())
            directions, energies = source_rays(room.sources[0], room.rt_args["n_rays"], n_bands)
            seeds = np.random.default_rng(seed).integers(0, 2**62, size=ray_processes * SLICES_PER_PROCESS)
            mics = [(room.mic_array.R[:, m], None) for m in range(room.mic_array.M)]
            spec = engine_spec(room)
    if tracing is not None and ray_processes > 1:
        # the lock is not needed while the workers trace: they have their own pyroomacoustics
        histograms = trace_parallel(spec, mics, directions, energies, room.sources[0].position, ray_processes, seeds, progress)
        with pra_octave_grid():
            rirs = [r[0] for r in room_responses(room, [[h] for h in histograms])]
    return [_on_package_scale(r, latency, tracing["n_samples"]) for r in rirs]


def _by_emitter(geometry, sources, mic_positions, fs, max_order, air_absorption, air, tracing, seed, n_jobs) -> List[List[np.ndarray]]:
    """``[mic][source]`` with every source in a room of its own, over ``n_jobs`` processes (:func:`_emitter_rirs`)."""
    from compas_acoustics.worker import jobs

    from .parallel import process_pool
    from .parallel import split_processes

    at_once, ray_processes = split_processes(n_jobs, len(sources))
    if tracing is None:
        ray_processes = 1
    # a seed per source, drawn from the run's: with ray tracing the run repeats for a given seed and n_jobs
    seeds = np.random.default_rng(seed).integers(0, 2**31 - 1, size=len(sources))
    args = [(geometry, source, mic_positions, fs, max_order, air_absorption, air, tracing, int(seeds[s]), ray_processes) for s, source in enumerate(sources)]
    stage = "ray tracing" if tracing is not None else "image sources"
    per_source: List = [None] * len(sources)
    if at_once == 1:
        for s, a in enumerate(args):
            jobs.progress(stage, s, len(sources))

            def rays(done, total, s=s):
                jobs.progress(stage, s * total + done, len(sources) * total)

            per_source[s] = _emitter_rirs(*a, progress=rays)
    else:
        from concurrent.futures import as_completed

        jobs.progress(stage, 0, len(sources))
        with process_pool(at_once) as pool:
            futures = {pool.submit(_emitter_rirs, *a): s for s, a in enumerate(args)}
            for done, future in enumerate(as_completed(futures), 1):
                per_source[futures[future]] = future.result()
                jobs.progress(stage, done, len(sources))
    return [[per_source[s][m] for s in range(len(sources))] for m in range(mic_positions.shape[1])]


@dataclass(frozen=True)
class DirectionalHistogram:
    """The ray-traced energy of one source at one receiver, per direction of arrival, band and time bin.

    ``energy`` is ``(n_directions, n_bands, n_bins)``: the package-scale energy arriving within each bin of
    ``bin_size`` seconds from emission, from the one of ``directions`` (unit vectors in room coordinates) nearest
    its last leg, in each band of ``bands``. Summed over the directions it is the omnidirectional histogram;
    over the bins, the energy the rays carry past the run's image source order.
    """

    energy: np.ndarray
    bin_size: float
    directions: np.ndarray
    bands: np.ndarray


def directional_histograms(
    geometry: RoomGeometry,
    sources,
    mic_positions,
    fs: int,
    directions: Sequence[np.ndarray],
    *,
    max_order: int = 6,
    duration: Optional[float] = None,
    n_rays: Optional[int] = 20000,
    receiver_radius: float = 0.5,
    energy_threshold: float = 1e-7,
    air_absorption: bool = True,
    seed: Optional[int] = None,
    air=None,
    n_jobs: int = 1,
) -> List[List[DirectionalHistogram]]:
    """Ray-traced energy per direction of arrival of every source at every microphone, ``[mic][source]``.

    The detection sphere divided into directivity groups (Lentz, Schroeder and Vorlaender 2007; Schroeder 2011,
    Sec. 5.3.4): every detected ray is binned by the direction nearest its last leg. ``directions`` holds one
    ``(n, 3)`` array of unit vectors in room coordinates per microphone. Arguments and rays are those of
    :func:`ray_traced_rirs`: a ray is logged only after ``max_order`` reflections, so the bins hold what the image
    sources do not. The tracing engine is driven directly, since the room's own call would reset an
    omnidirectional receiver to one bin. Air absorption is applied per band on the bins' travel distance.
    ``n_jobs`` > 1 splits each source's rays over that many processes (:func:`~compas_acoustics.room.parallel.trace_parallel`).
    """
    from .parallel import MIN_RAYS_PER_PROCESS
    from .parallel import SLICES_PER_PROCESS
    from .parallel import engine_spec
    from .parallel import trace_parallel

    if duration is None:
        duration = default_duration(geometry, air_absorption, air=air)
    if duration <= 0.0:
        raise ValueError(f"duration must be positive, got {duration!r}")
    mic_positions = np.asarray(mic_positions, dtype=float).reshape(3, -1)
    if len(directions) != mic_positions.shape[1]:
        raise ValueError(f"directions must hold one array per microphone ({mic_positions.shape[1]}), got {len(directions)}")
    unit = []
    for d in directions:
        d = np.asarray(d, dtype=float)
        if d.ndim != 2 or d.shape[1] != 3 or d.shape[0] < 1:
            raise ValueError("each microphone's directions must be (n, 3)")
        unit.append(d / np.linalg.norm(d, axis=1, keepdims=True))
    options = {
        "receiver_radius": receiver_radius,
        "energy_thres": energy_threshold,
        "time_thres": float(duration),
        "hist_bin_size": _HIST_BIN,
    }
    from compas_acoustics.worker import jobs

    with pra_octave_grid():
        # inside the lock, as in ray_traced_rirs
        pra.random.seed(seed)
        room = geometry.to_pra(fs, max_order, ray_tracing=True, air_absorption=air_absorption, air=air)
        for source in sources:
            room.add_soundsource(source)
        room.add_microphone_array(pra.MicrophoneArray(mic_positions, fs))
        room.set_ray_tracing(n_rays=n_rays, **options)
        if n_rays is None and room.rt_args["n_rays"] < MIN_RAYS:
            room.set_ray_tracing(n_rays=MIN_RAYS, **options)
        n_rays = room.rt_args["n_rays"]
        n_bands = len(room.octave_bands.get_bw())
        bands = np.asarray(room.octave_bands.centers, dtype=float)
        for m, d in enumerate(unit):
            room.room_engine.microphones[m].set_directions(np.asarray(d, dtype=np.float32))
        bin_size = float(room.rt_args["hist_bin_size"])
        n_bins = int(np.ceil(duration / bin_size))
        distance = (np.arange(n_bins) + 0.5) * bin_size * room.c
        damping = np.exp(-np.asarray(room.air_absorption, dtype=float)[:, np.newaxis] * distance[np.newaxis, :]) if air_absorption else np.ones((n_bands, n_bins))
        histograms = [[] for _ in unit]
        parallel = n_jobs > 1 and n_rays >= 2 * MIN_RAYS_PER_PROCESS
        if parallel:
            spec = engine_spec(room)
            mics = [(mic_positions[:, m], np.asarray(d, dtype=np.float32)) for m, d in enumerate(unit)]
        for s, source in enumerate(room.sources):
            jobs.progress("directional histograms", s, len(room.sources))
            if source.directivity is None:
                ray_directions = pra.fibonacci_spherical_sampling(n_rays).T
                ray_energies = np.ones((n_rays, n_bands))
            else:
                ray_directions, ray_energies = source.directivity.sample_rays(n_rays)
                ray_energies = np.broadcast_to(ray_energies, (n_rays, n_bands))
            if parallel:
                seeds = np.random.default_rng(None if seed is None else (seed, s)).integers(0, 2**62, size=n_jobs * SLICES_PER_PROCESS)
                traced = trace_parallel(spec, mics, ray_directions, ray_energies, source.position, n_jobs, seeds)
            else:
                room.room_engine.ray_tracing(ray_directions, ray_energies, source.position)
            for m, d in enumerate(unit):
                # every line's histogram is sized for the default 10 s and doubled only for lines hit later
                if parallel:
                    lines = [np.asarray(h, dtype=float) for h in traced[m]]
                else:
                    lines = [np.asarray(h.get_hist(), dtype=float) for h in room.room_engine.microphones[m].histograms]
                width = max(n_bins, *(line.shape[1] for line in lines))
                energy = np.stack([np.pad(line, ((0, 0), (0, width - line.shape[1]))) for line in lines])[:, :, :n_bins]
                histograms[m].append(
                    DirectionalHistogram(
                        energy=energy * damping / (4.0 * np.pi) ** 2,
                        bin_size=bin_size,
                        directions=d,
                        bands=bands,
                    )
                )
            if not parallel:
                room.room_engine.reset_mics()
    return histograms
