"""Ray-traced monaural room impulse responses: image sources plus ray tracing.

Ported by hand from hybrid-rir (gpapadimitri/hybrid-rir, src/room.py:
``RoomSpec.ray_traced_rir``, ``_trace_rays``, ``_traced_in_parallel`` and
``_default_simulation_duration``) to trace every source and microphone of a
scene in one room, without pyFDN: this is the response the Grasshopper
components use.

pyroomacoustics runs in its hybrid mode: specular reflections up to
``max_order`` from the image source model, everything later from a stochastic
ray tracer that also carries the scattered energy. Responses are returned on
the package scale, spreading ``1 / (4 pi r)`` and the direct sound on its
geometric sample (pyroomacoustics spreads as ``1 / r`` and delays by half its
fractional-delay kernel).
"""

from __future__ import annotations

import warnings
from concurrent.futures import Executor
from concurrent.futures import ProcessPoolExecutor
from typing import List
from typing import Optional

import numpy as np
import pyroomacoustics as pra
from pyroomacoustics.doa import fibonacci_spherical_sampling

from compas_acoustics.diffuse import DiffuseFieldWarning
from compas_acoustics.diffuse import predicted_rt
from compas_acoustics.room_geometry import RoomGeometry
from compas_acoustics.room_geometry import pra_octave_grid

__all__ = ['MIN_RAYS', 'OPEN_ROOM_DURATION', 'default_duration', 'ray_traced_rirs']

# Floor on pyroomacoustics' own ray count, applied only with n_rays=None.
MIN_RAYS = 50_000
# Buffer for a scene that does not close a volume and so has no Eyring estimate.
OPEN_ROOM_DURATION = 10.0
_HIST_BIN = 0.004


def default_duration(geometry: RoomGeometry, air_absorption: bool, speed_of_sound: Optional[float] = None, air=None) -> float:
    """How long to simulate this room, from the Eyring prediction (upstream ``_default_simulation_duration``).

    Eyring is not accurate enough to be a decay target, but it sizes a buffer:
    2.5 times the longest predicted RT leaves the ray tracer room to be wrong
    by a factor of two and still capture 60 dB of decay, within 0.5 to 12 s.
    Its diffuse-field warning is suppressed because nothing acoustic is being
    claimed. A scene that does not close a volume has no Eyring estimate and
    gets ``OPEN_ROOM_DURATION``.
    """
    if not geometry.is_closed:
        return OPEN_ROOM_DURATION
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DiffuseFieldWarning)
        rt = float(np.max(predicted_rt(geometry, air_absorption=air_absorption, speed_of_sound=speed_of_sound,
                                        air=air)))
    return float(np.clip(2.5 * rt, 0.5, 12.0))


def _on_package_scale(rir, latency: int, n_samples: int) -> np.ndarray:
    rir = np.asarray(rir, dtype=float)[latency:] / (4.0 * np.pi)
    return np.pad(rir, (0, max(0, n_samples - rir.size)))[:n_samples]


def _trace_slice(geometry, fs, max_order, air_absorption, options, n_rays, mic_positions,
                 position, directions, energies, seed, air=None):
    """Receiver histograms of one slice of one source's rays, ``[mic][direction]``.

    ``Room.ray_tracing`` with its loop over sources taken out; the engine
    normalises each ray's energy by the ``n_rays`` it is configured with.
    """
    if seed is not None:
        pra.random.seed(seed)
    with pra_octave_grid():
        room = geometry.to_pra(fs, max_order, ray_tracing=True, air_absorption=air_absorption, air=air)
        room.add_microphone_array(pra.MicrophoneArray(mic_positions, fs))
        room.set_ray_tracing(n_rays=n_rays, **options)
    for mic in room.room_engine.microphones:
        mic.make_omnidirectional()
    room.room_engine.ray_tracing(directions, energies, position)
    return [[h.get_hist() for h in mic.histograms] for mic in room.room_engine.microphones]


def _merged(parts, slices, n_rays):
    """One microphone's histograms from every slice, weighted by its share of the rays."""
    merged = []
    for d in range(len(parts[0])):
        total = np.zeros((parts[0][d].shape[0], max(p[d].shape[-1] for p in parts)))
        for (a, b), p in zip(slices, parts):
            total[:, : p[d].shape[-1]] += (b - a) / n_rays * p[d]
        merged.append(total)
    return merged


def _traced_in_parallel(room, geometry, fs, max_order, air_absorption, options, mic_positions,
                        n_jobs, seed, executor, air=None):
    """``rt_histograms`` of every source, ``[mic][source][direction]``, in ``n_jobs`` slices.

    Each source's rays are the directions and energies ``Room.ray_tracing``
    would use, cut into contiguous slices; slice ``k`` draws from ``seed + k``.
    """
    n_rays = room.rt_args["n_rays"]
    n_bands = len(room.octave_bands.get_bw())
    edges = np.linspace(0, n_rays, n_jobs + 1).astype(int).tolist()
    slices = list(zip(edges[:-1], edges[1:]))
    histograms = [[] for _ in range(mic_positions.shape[1])]
    pool = executor if executor is not None else ProcessPoolExecutor(n_jobs)
    try:
        for source in room.sources:
            if source.directivity is None:
                directions = fibonacci_spherical_sampling(n_rays).T
                energies = np.ones((n_rays, n_bands))
            else:
                directions, energies = source.directivity.sample_rays(n_rays)
                energies = np.broadcast_to(energies, (n_rays, n_bands))
            position = np.asarray(source.position, dtype=float).ravel()
            jobs = [
                (geometry, fs, max_order, air_absorption, options, n_rays, mic_positions, position,
                 np.ascontiguousarray(directions[a:b]), np.ascontiguousarray(energies[a:b]),
                 None if seed is None else seed + k, air)
                for k, (a, b) in enumerate(slices)
            ]
            parts = list(pool.map(_trace_slice, *zip(*jobs)))
            for m in range(len(histograms)):
                histograms[m].append(_merged([p[m] for p in parts], slices, n_rays))
    finally:
        if executor is None:
            pool.shutdown()
    return histograms


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
    energy_threshold: float = 1e-8,
    air_absorption: bool = True,
    seed: Optional[int] = None,
    n_jobs: int = 1,
    executor: Optional[Executor] = None,
    air=None,
) -> List[List[np.ndarray]]:
    """Monaural responses of every source at every microphone, ``[mic][source]``.

    ``sources`` are ``pra.SoundSource`` objects; ``mic_positions`` is ``(3, M)``.
    Every material must carry scattering. ``max_order`` splits specular from
    ray-traced energy; hybrid-rir measured 6 to 10 as the range where the
    decay stops depending on it. ``duration`` defaults to
    :func:`default_duration`. ``n_rays=None`` takes pyroomacoustics' heuristic,
    raised to ``MIN_RAYS``. ``seed`` seeds pyroomacoustics' generators; without
    scattering nothing is random. ``n_jobs > 1`` traces each source's rays in
    that many processes: not inside Rhino, where ``sys.executable`` is Rhino,
    and from a script only under ``if __name__ == "__main__":``, because the
    workers are spawned and re-import it. Without scattering the result equals
    the single-process one to about 1e-8 of the peak; with scattering slice
    ``k`` draws from ``seed + k``, an equivalent draw, not the same one.
    ``air`` (:class:`AirConditions`) sets the speed of sound and the air absorption (see :meth:`RoomGeometry.to_pra`).
    """
    if n_jobs < 1:
        raise ValueError(f"n_jobs must be at least 1, got {n_jobs!r}")
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
    if seed is not None:
        # numpy only, as hybrid-rir: seeding libroom as well draws different
        # noise for the ray-traced tail (same envelope, 0.1 dB per window),
        # which breaks sample parity with upstream. Repeatable either way.
        pra.random.seed(seed)
    with pra_octave_grid():
        room = geometry.to_pra(fs, max_order, ray_tracing=True, air_absorption=air_absorption, air=air)
        for source in sources:
            room.add_soundsource(source)
        room.add_microphone_array(pra.MicrophoneArray(mic_positions, fs))
        room.set_ray_tracing(n_rays=n_rays, **options)
        if n_rays is None and room.rt_args["n_rays"] < MIN_RAYS:
            room.set_ray_tracing(n_rays=MIN_RAYS, **options)
        if n_jobs > 1:
            room.rt_histograms = _traced_in_parallel(
                room, geometry, fs, max_order, air_absorption, options, mic_positions, n_jobs, seed, executor, air
            )
            room.rt_directional_responses = [
                {"response": np.ones(1), "is_impulse_response": False} for _ in range(mic_positions.shape[1])
            ]
            room.simulator_state["rt_done"] = True
        room.compute_rir()

    latency = pra.constants.get("frac_delay_length") // 2
    n_samples = round(duration * fs)
    return [
        [_on_package_scale(room.rir[m][s], latency, n_samples) for s in range(len(sources))]
        for m in range(mic_positions.shape[1])
    ]
