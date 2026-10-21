"""Ray-traced monaural room impulse responses: image sources plus ray tracing.

Ported by hand from hybrid-rir (gpapadimitri/hybrid-rir, src/room.py:
``RoomSpec.ray_traced_rir``, ``_trace_rays`` and
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
from typing import List
from typing import Optional

import numpy as np
import pyroomacoustics as pra

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
    air=None,
) -> List[List[np.ndarray]]:
    """Monaural responses of every source at every microphone, ``[mic][source]``.

    ``sources`` are ``pra.SoundSource`` objects; ``mic_positions`` is ``(3, M)``.
    Every material must carry scattering. ``max_order`` splits specular from
    ray-traced energy; hybrid-rir measured 6 to 10 as the range where the
    decay stops depending on it. ``duration`` defaults to
    :func:`default_duration`. ``n_rays=None`` takes pyroomacoustics' heuristic,
    raised to ``MIN_RAYS``. ``seed`` seeds pyroomacoustics' generators; None
    draws a fresh seed from the OS. Without scattering nothing is random.
    ``air`` (:class:`AirConditions`) sets the speed of sound and the air absorption (see :meth:`RoomGeometry.to_pra`).
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
    # numpy only, as hybrid-rir: seeding libroom as well draws different
    # noise for the ray-traced tail (same envelope, 0.1 dB per window),
    # which breaks sample parity with upstream. Repeatable either way.
    # None reseeds from the OS rather than continuing an earlier run's stream.
    pra.random.seed(seed)
    with pra_octave_grid():
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
    return [
        [_on_package_scale(room.rir[m][s], latency, n_samples) for s in range(len(sources))]
        for m in range(mic_positions.shape[1])
    ]
