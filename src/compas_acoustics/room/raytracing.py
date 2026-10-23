"""pyroomacoustics runs on a room: monaural responses from image sources alone or with ray tracing, and the rays' energy per direction.

The backend of :class:`~compas_acoustics.simulation_methods.Mono` (:func:`image_source_rirs`,
:func:`ray_traced_rirs`) and of the binaural and ambisonics methods (their reference run, their directional histograms
and the length of both, :func:`default_duration`). The ray tracing is ported by hand from hybrid-rir
(gpapadimitri/hybrid-rir, src/room.py: ``RoomSpec.ray_traced_rir``, ``_trace_rays`` and
``_default_simulation_duration``) to trace every source and microphone of a scene in one room, without pyFDN.

With ray tracing pyroomacoustics runs in its hybrid mode: specular reflections up to
``max_order`` from the image source model, everything later from a stochastic
ray tracer that also carries the scattered energy. Responses are returned on
the package scale, spreading ``1 / (4 pi r)`` and the direct sound on its
geometric sample (pyroomacoustics spreads as ``1 / r`` and delays by half its
fractional-delay kernel).
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

__all__ = ['MIN_RAYS', 'OPEN_ROOM_DURATION', 'MAX_DURATION', 'DURATION_RT_FACTOR', 'DirectionalHistogram', 'default_duration', 'directional_histograms',
           'image_source_rirs', 'ray_traced_rirs']

# Floor on pyroomacoustics' own ray count, applied only with n_rays=None.
MIN_RAYS = 50_000
# A response lasts this many times the longest Eyring RT, clipped to the range below, in seconds, for the
# mono method and for the runs the hybrid makes itself. ODEON asks for at least 2/3 of the longest RT
# and RAZR takes 1.0 (op.len_rt_factor); 1.5 keeps T30 where Eyring underestimates the decay by 18 to 31 %
# (diffuse.py). Measured over five shoeboxes: the hybrid's response is identical at 1.5, 2.0 and 2.5 (nothing
# reads past 1.5), and 1.0 moves T30 at 125 Hz by 9 to 20 %. Where a room outruns Eyring altogether (a furnished
# office: 2 s against 0.64 s) no factor helps, and the methods warn about the length instead.
DURATION_RT_FACTOR = 1.5
MIN_DURATION = 0.5
MAX_DURATION = 12.0
# Buffer for a scene that does not close a volume and so has no Eyring estimate.
OPEN_ROOM_DURATION = 10.0
_HIST_BIN = 0.004


def default_duration(geometry: RoomGeometry, air_absorption: bool, speed_of_sound: Optional[float] = None, air=None,
                     factor: float = DURATION_RT_FACTOR) -> float:
    """How long to simulate this room, from the Eyring prediction.

    ``factor`` times the longest predicted RT, within ``MIN_DURATION`` and
    ``MAX_DURATION``. Eyring is not accurate enough to be a decay target, but it
    sizes a buffer (``DURATION_RT_FACTOR``). The diffuse-field check is skipped because nothing acoustic is being claimed.
    A scene that does not close a volume has no Eyring estimate and gets
    ``OPEN_ROOM_DURATION``.

    A prediction long enough to hit ``MAX_DURATION`` warns: the rays of such a
    room live for hundreds of reflections and the run takes seconds per source,
    and more often than not it is a material with no low-frequency absorption on
    a large surface rather than the room intended.
    """
    if not geometry.is_closed:
        return OPEN_ROOM_DURATION
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
    return float(np.clip(factor * rt[longest], MIN_DURATION, MAX_DURATION))


def _on_package_scale(rir, latency: int, n_samples: int) -> np.ndarray:
    rir = np.asarray(rir, dtype=float)[latency:] / (4.0 * np.pi)
    return np.pad(rir, (0, max(0, n_samples - rir.size)))[:n_samples]


def image_source_rirs(geometry: RoomGeometry, sources, mic_positions, fs: int, *, max_order: int, air_absorption: bool = True,
                      air=None) -> List[List[np.ndarray]]:
    """Image-source responses of every source at every microphone, ``[mic][source]``, on the package scale, at their
    natural length. The arguments are those of :func:`ray_traced_rirs`."""
    mic_positions = np.asarray(mic_positions, dtype=float).reshape(3, -1)
    with pra_octave_grid():
        room = geometry.to_pra(fs, max_order, air_absorption=air_absorption, air=air)
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


@dataclass(frozen=True)
class DirectionalHistogram:
    """The ray-traced energy of one source at one receiver, per direction of arrival, band and time bin.

    ``energy`` is ``(n_directions, n_bands, n_bins)``: the energy of the pressure
    response on the package scale (``1 / (4 pi r)``) that arrives within each
    bin of ``bin_size`` seconds, from the direction of ``directions`` (unit
    vectors in room coordinates) nearest to its last leg, in the octave band of
    ``bands``. Time runs from the emission, as :func:`ray_traced_rirs` returns
    it. The sum over the directions is pyroomacoustics' own histogram on the
    package scale; the sum over the bins is the energy the rays carry beyond the
    image source order of the run.
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
) -> List[List[DirectionalHistogram]]:
    """Ray-traced energy per direction of arrival of every source at every microphone, ``[mic][source]``.

    RAVEN's detection sphere divided into "directivity groups" (Lentz, Schroeder
    and Vorlaender 2007; Schroeder 2011, Sec. 5.3.4): pyroomacoustics' receiver
    keeps one histogram per direction and bins every detected ray by the
    direction, nearest in angle, its last leg arrives from
    (``libroom_src/microphone.hpp``, ``get_dir_bin``). ``directions`` holds one
    ``(n, 3)`` array of unit vectors in room coordinates per microphone; a
    receiver's own frame is the caller's business. The arguments are those of
    :func:`ray_traced_rirs`, and the rays are the same: pyroomacoustics logs a
    specular ray only after ``max_order`` reflections, so the bins hold what
    the image sources of that order do not. ``Room.ray_tracing`` would reset an
    omnidirectional receiver to one bin, so the engine is driven directly, as
    that method does. Air absorption is applied per band on the bins' travel
    distance, as pyroomacoustics applies it to its tail.
    """
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
    pra.random.seed(seed)
    with pra_octave_grid():
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
        damping = (
            np.exp(-np.asarray(room.air_absorption, dtype=float)[:, np.newaxis] * distance[np.newaxis, :])
            if air_absorption else np.ones((n_bands, n_bins))
        )
        histograms = [[] for _ in unit]
        for source in room.sources:
            if source.directivity is None:
                ray_directions = pra.fibonacci_spherical_sampling(n_rays).T
                ray_energies = np.ones((n_rays, n_bands))
            else:
                ray_directions, ray_energies = source.directivity.sample_rays(n_rays)
                ray_energies = np.broadcast_to(ray_energies, (n_rays, n_bands))
            room.room_engine.ray_tracing(ray_directions, ray_energies, source.position)
            for m, d in enumerate(unit):
                # libroom sizes every line's histogram for its default 10 s and doubles only the lines hit later
                lines = [np.asarray(h.get_hist(), dtype=float) for h in room.room_engine.microphones[m].histograms]
                width = max(n_bins, *(line.shape[1] for line in lines))
                energy = np.stack([np.pad(line, ((0, 0), (0, width - line.shape[1]))) for line in lines])[:, :, :n_bins]
                histograms[m].append(DirectionalHistogram(
                    energy=energy * damping / (4.0 * np.pi) ** 2, bin_size=bin_size, directions=d, bands=bands,
                ))
            room.room_engine.reset_mics()
    return histograms
