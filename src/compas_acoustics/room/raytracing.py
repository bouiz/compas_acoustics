# _pair_responses, _ray_tracing and the ray loop of directional_histograms are adapted from pyroomacoustics 0.10.1,
# Room.compute_rir and Room.ray_tracing (pyroomacoustics/room.py), which carries this notice:
#
# Copyright (C) 2019  Robin Scheibler, Ivan Dokmanic, Sidney Barthe, Cyril Cadoux
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Monaural responses from image sources alone or with ray tracing, and the rays' energy per direction.

The backend of :class:`~compas_acoustics.simulation_methods.Mono` and of the hybrid methods' reference runs and
directional histograms. With ray tracing, specular reflections up to ``max_order`` come from the image sources and the
rest from pyroomacoustics' stochastic ray tracer, which also carries the scattered energy of every reflection. The
tracer logs a ray's specular arrivals from its ``max_order``-th reflection on, so order ``max_order`` is in both the
image sources and the rays. Responses are on the package scale: spreading ``1 / (4 pi r)``, the direct sound on its
geometric sample (pyroomacoustics spreads as ``1 / r`` and delays by half its fractional-delay kernel).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import List
from typing import Optional
from typing import Sequence

import numpy as np
import pyroomacoustics as pra
from pyroomacoustics.simulation import compute_ism_rir
from pyroomacoustics.simulation import compute_rt_rir
from pyroomacoustics.utilities import design_highpass_filter_sos
from scipy.signal import sosfiltfilt

from .diffuse import predicted_rt
from .geometry import SYNTHESIS_BANDS
from .geometry import RoomGeometry

__all__ = [
    "MIN_RAYS",
    "MAX_DURATION",
    "DURATION_RT_FACTOR",
    "DirectionalHistogram",
    "RayTracedRun",
    "default_duration",
    "default_rays",
    "directional_histograms",
    "image_source_rirs",
    "ray_traced_run",
]

# Floor on the default ray count (the package's choice)
MIN_RAYS = 20_000
# n / (dt pooled) of hybrid.envelope.histogram_rays on the hybrid's 96-line grid, in 1/s: written out because the grid
# needs spharpy, which Python < 3.11 lacks; a test checks it against the grid
_GRID_RAYS = 271.1566980495139
# A response lasts this many times the longest Eyring RT, within 0.5 s and MAX_DURATION (the three numbers are the
# package's choices). The margin over 1 keeps the T30 evaluation range inside the response where Eyring underestimates
# the decay; a room whose decay outlasts it is caught by the methods' length warnings.
DURATION_RT_FACTOR = 1.5
MAX_DURATION = 12.0
# pyroomacoustics' default histogram bin
_HIST_BIN = 0.004


def default_rays(volume: float, receiver_radius: float, speed_of_sound: float) -> int:
    """Rays per source by default: those at which a direction's energy over one slot of the hybrid's late envelope has
    a standard deviation of 1 dB, on its 96-line grid (Vorlaender 2020, Sec. 11.3.7, Eq. 11.20;
    :func:`compas_acoustics.hybrid.envelope.histogram_rays`), and at least ``MIN_RAYS``.

    One count for every method, so that a ``Mono`` reference traces what ``Binaural`` and ``Ambisonics`` trace
    themselves; the floor is the package's choice.
    """
    hits = (10.0 / np.log(10.0)) ** 2
    return max(int(np.ceil(hits * volume * _GRID_RAYS / (np.pi * receiver_radius**2 * speed_of_sound))), MIN_RAYS)


def default_duration(geometry: RoomGeometry, air_absorption: bool, air=None, factor: float = DURATION_RT_FACTOR) -> float:
    """How long to simulate this room: ``factor`` times the longest Eyring RT, within 0.5 s and ``MAX_DURATION``.

    Eyring only sizes the buffer, so its diffuse-field check is skipped. A scene that closes no volume and has no
    given air volume gets 10 s.
    Hitting ``MAX_DURATION`` warns: the rays then live for hundreds of reflections, and the cause is usually a
    large surface without low-frequency absorption.
    """
    if not getattr(geometry, "has_volume", geometry.is_closed):
        return 10.0
    rt = predicted_rt(geometry, air_absorption=air_absorption, air=air, warn=False)
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


def _pair_responses(room) -> List[List[np.ndarray]]:
    """The room's responses ``[mic][source]`` as pyroomacoustics composes them, except that a pair without a visible
    image has no image part (the room's own call fails on it, or renders the blocked direct sound when the source is
    hidden from every microphone)."""
    if room.simulator_state["ism_needed"] and not room.simulator_state["ism_done"]:
        room.image_source_model()
    if room.simulator_state["rt_needed"] and not room.simulator_state["rt_done"]:
        room.ray_tracing()
    hpf = None
    if pra.constants.get("rir_hpf_enable"):
        hpf = design_highpass_filter_sos(room.fs, pra.constants.get("rir_hpf_fc"), **pra.constants.get("rir_hpf_kwargs"))
    fdl = pra.constants.get("frac_delay_length")
    volume = room.get_volume()
    responses = []
    for m, mic in enumerate(room.mic_array.R.T):
        responses.append([])
        for s, source in enumerate(room.sources):
            parts = []
            visible = np.asarray(room.visibility[s][m, :], dtype=bool) if room.simulator_state["ism_needed"] else None
            if visible is not None and visible.any():
                parts.append(
                    compute_ism_rir(
                        source,
                        mic,
                        room.mic_array.directivity[m],
                        source.directions[m, :, :],
                        visible,
                        fdl,
                        room.c,
                        room.fs,
                        room.octave_bands,
                        min_phase=room.min_phase,
                        air_abs_coeffs=room.air_absorption,
                    )
                )
            if room.simulator_state["rt_needed"]:
                t0 = np.linalg.norm(source.position - mic) / room.c
                parts.append(
                    compute_rt_rir(
                        t0.item(),
                        room.rt_histograms[m][s],
                        room.rt_directional_responses[m],
                        room.rt_args["hist_bin_size"],
                        room.rt_args["hist_bin_size_samples"],
                        volume,
                        fdl,
                        room.c,
                        room.fs,
                        room.octave_bands,
                        air_abs_coeffs=room.air_absorption,
                    )
                )
            if not parts:
                responses[m].append(np.zeros(fdl))
                continue
            if len(parts) == 1:
                rir = parts[0]
            else:
                rir = np.zeros(max(p.shape[0] for p in parts))
                for p in parts:
                    rir[: p.shape[0]] += p
            if hpf is not None:
                rir = np.ascontiguousarray(sosfiltfilt(hpf, rir))
            responses[m].append(rir)
    return responses


def image_source_rirs(geometry: RoomGeometry, sources, mic_positions, fs: int, *, max_order: int, air_absorption: bool = True, air=None) -> List[List[np.ndarray]]:
    """Image-source responses ``[mic][source]`` on the package scale at their natural length; arguments as
    :func:`ray_traced_run`."""
    mic_positions = np.asarray(mic_positions, dtype=float).reshape(3, -1)
    room = geometry.to_pra(fs, max_order, air_absorption=air_absorption, air=air)
    for source in sources:
        room.add_soundsource(source)
    room.add_microphone_array(pra.MicrophoneArray(mic_positions, fs))
    rirs = _pair_responses(room)
    latency = pra.constants.get("frac_delay_length") // 2
    return [[_on_package_scale(rir, latency, len(rir) - latency) for rir in row] for row in rirs]


def ray_traced_run(
    geometry: RoomGeometry,
    sources,
    mic_positions,
    fs: int,
    *,
    max_order: int = 6,
    duration: Optional[float] = None,
    n_rays: Optional[int] = None,
    receiver_radius: float = 0.5,
    energy_threshold: float = 1e-7,
    air_absorption: bool = True,
    seed: Optional[int] = None,
    air=None,
    directions: Optional[Sequence[np.ndarray]] = None,
) -> RayTracedRun:
    """Monaural responses of every source at every microphone from one image source search and one ray trace.

    ``sources`` are ``pra.SoundSource``; ``mic_positions`` is ``(3, M)``. ``max_order`` is the last image source
    order; the rays hold it too (module docstring). ``duration`` None: :func:`default_duration`. ``n_rays`` None:
    :func:`default_rays`. ``seed`` None: unseeded. ``air`` sets the speed of sound and
    the air absorption. ``directions``, one ``(n, 3)`` array of unit vectors in room coordinates per microphone, also
    returns the detected rays' energy by the direction nearest their last leg (``histograms``); the responses are
    unchanged.
    """
    room, duration, unit = _traced_room(
        geometry, sources, mic_positions, fs, directions, max_order, duration, n_rays, receiver_radius, energy_threshold, air_absorption, seed, air
    )
    room.image_source_model()
    lines = _ray_tracing(room, unit)
    rirs = _pair_responses(room)
    histograms = None
    if unit is not None:
        histograms = [[_directional_histogram(room, line, d, duration, air_absorption) for line in per_source] for per_source, d in zip(lines, unit)]

    latency = pra.constants.get("frac_delay_length") // 2
    n_samples = round(duration * fs)
    return RayTracedRun(room=room, responses=[[_on_package_scale(rir, latency, n_samples) for rir in row] for row in rirs], histograms=histograms)


def _traced_room(geometry, sources, mic_positions, fs, directions, max_order, duration, n_rays, receiver_radius, energy_threshold, air_absorption, seed, air):
    """``(room, duration, unit directions or None)``: the room built for ray tracing with every source and
    microphone and its tracer set, not yet searched or traced; the arguments are those of :func:`ray_traced_run`."""
    if duration is None:
        duration = default_duration(geometry, air_absorption, air=air)
    if duration <= 0.0:
        raise ValueError(f"duration must be positive, got {duration!r}")
    mic_positions = np.asarray(mic_positions, dtype=float).reshape(3, -1)
    unit = None if directions is None else _unit_directions(directions, mic_positions.shape[1])
    options = {
        "receiver_radius": receiver_radius,
        "energy_thres": energy_threshold,
        "time_thres": float(duration),
        "hist_bin_size": _HIST_BIN,
    }
    # The numpy seed only: pyroomacoustics derives the tracer's seed from it. None reseeds both from the OS.
    pra.random.seed(seed)
    room = geometry.to_pra(fs, max_order, ray_tracing=True, air_absorption=air_absorption, air=air)
    for source in sources:
        room.add_soundsource(source)
    room.add_microphone_array(pra.MicrophoneArray(mic_positions, fs))
    room.set_ray_tracing(n_rays=default_rays(room.get_volume(), receiver_radius, room.c) if n_rays is None else n_rays, **options)
    return room, duration, unit


def _each_source_traced(room):
    """Traces the rays of every source in turn, as pyroomacoustics' ``Room.ray_tracing`` does, yielding while the
    receivers hold that source's histograms and resetting them after."""
    engine = room.room_engine
    n_rays = room.rt_args["n_rays"]
    n_bands = len(room.octave_bands.get_bw()) if room.is_multi_band else 1
    for source in room.sources:
        if source.directivity is None:
            ray_directions = pra.fibonacci_spherical_sampling(n_rays).T
            ray_energies = np.ones((n_rays, n_bands))
        else:
            ray_directions, ray_energies = source.directivity.sample_rays(n_rays)
            ray_energies = np.broadcast_to(ray_energies, (n_rays, n_bands))
        engine.ray_tracing(ray_directions, ray_energies, source.position)
        yield
        engine.reset_mics()


def _ray_tracing(room, directions=None) -> Optional[List[List[list]]]:
    """pyroomacoustics' ``Room.ray_tracing`` for omnidirectional microphones, call for call, so its random draws and
    responses are the same; with ``directions`` a copy of every microphone, added after the image source search, sorts
    the same rays by direction. Returns the copies' raw histograms ``[mic][source]``, else None."""
    engine = room.room_engine
    n_mics = room.mic_array.M
    room.rt_directional_responses = []
    for m in range(n_mics):
        engine.microphones[m].make_omnidirectional()
        room.rt_directional_responses.append({"response": np.ones(1), "is_impulse_response": False})
    if directions is not None:
        for m in range(n_mics):
            engine.add_mic(room.mic_array.R[:, m])
        # A microphone with directions holds a search tree on its own memory and is not safe to copy: all copies
        # are added before any gets its directions, so no later addition moves one (MSVC grows the list by half).
        for m, d in enumerate(directions):
            engine.microphones[n_mics + m].set_directions(np.asarray(d, dtype=np.float32))
    room.rt_histograms = [[] for _ in range(n_mics)]
    copies = None if directions is None else [[] for _ in range(n_mics)]
    for _ in _each_source_traced(room):
        for m in range(n_mics):
            room.rt_histograms[m].append([h.get_hist() for h in engine.microphones[m].histograms])
            if copies is not None:
                copies[m].append([np.asarray(h.get_hist(), dtype=float) for h in engine.microphones[n_mics + m].histograms])
    room.simulator_state["rt_done"] = True
    return copies


def _unit_directions(directions, n_mics: int) -> List[np.ndarray]:
    if len(directions) != n_mics:
        raise ValueError(f"directions must hold one array per microphone ({n_mics}), got {len(directions)}")
    unit = []
    for d in directions:
        d = np.asarray(d, dtype=float)
        if d.ndim != 2 or d.shape[1] != 3 or d.shape[0] < 1:
            raise ValueError("each microphone's directions must be (n, 3)")
        unit.append(d / np.linalg.norm(d, axis=1, keepdims=True))
    return unit


def _directional_histogram(room, lines: list, directions: np.ndarray, duration: float, air_absorption: bool) -> DirectionalHistogram:
    """One microphone's raw histograms of one source on the package scale, air absorption applied per bin."""
    bin_size = float(room.rt_args["hist_bin_size"])
    n_bins = int(np.ceil(duration / bin_size))
    n_bands = len(room.octave_bands.get_bw())
    distance = (np.arange(n_bins) + 0.5) * bin_size * room.c
    damping = np.exp(-np.asarray(room.air_absorption, dtype=float)[:, np.newaxis] * distance[np.newaxis, :]) if air_absorption else np.ones((n_bands, n_bins))
    # every line's histogram is sized for the default 10 s and doubled only for lines hit later
    width = max(n_bins, *(line.shape[1] for line in lines))
    energy = np.stack([np.pad(line, ((0, 0), (0, width - line.shape[1]))) for line in lines])[:, :, :n_bins]
    return DirectionalHistogram(
        energy=energy * damping / (4.0 * np.pi) ** 2,
        bin_size=bin_size,
        directions=directions,
        bands=np.asarray(room.octave_bands.centers, dtype=float),
    )


@dataclass(frozen=True)
class DirectionalHistogram:
    """The ray-traced energy of one source at one receiver, per direction of arrival, band and time bin.

    ``energy`` is ``(n_directions, n_bands, n_bins)``: the package-scale energy arriving within each bin of
    ``bin_size`` seconds from emission, from the one of ``directions`` (unit vectors in room coordinates) nearest
    its last leg, in each band of ``bands``. Summed over the directions it is the omnidirectional histogram;
    over the bins, the energy the rays log: their specular arrivals from the run's image source order on and the
    scattered energy of every reflection.
    """

    energy: np.ndarray
    bin_size: float
    directions: np.ndarray
    bands: np.ndarray


@dataclass(frozen=True)
class RayTracedRun:
    """One image source search and one ray trace of a scene (:func:`ray_traced_run`).

    ``responses[mic][source]`` are on the package scale; ``histograms[mic][source]`` hold the rays' energy per
    direction where directions were asked for, else None; ``room`` is the searched room, built for ray tracing, whose
    image sources the hybrid methods take up.
    """

    room: pra.Room
    responses: List[List[np.ndarray]]
    histograms: Optional[List[List[DirectionalHistogram]]]


def directional_histograms(
    geometry: RoomGeometry,
    sources,
    mic_positions,
    fs: int,
    directions: Sequence[np.ndarray],
    *,
    max_order: int = 6,
    duration: Optional[float] = None,
    n_rays: Optional[int] = None,
    receiver_radius: float = 0.5,
    energy_threshold: float = 1e-7,
    air_absorption: bool = True,
    seed: Optional[int] = None,
    air=None,
) -> List[List[DirectionalHistogram]]:
    """Ray-traced energy per direction of arrival, ``[mic][source]``.

    The detection sphere divided into directivity groups (Lentz et al. 2007, Sec. 2.4; Schroeder 2011, Sec. 5.3.4):
    every detected ray is binned by the direction nearest its last leg. ``directions`` holds one ``(n, 3)`` array of
    unit vectors in room coordinates per microphone; the other arguments and the rays are those of
    :func:`ray_traced_run`, so order ``max_order`` is in the bins and in the image sources. Air absorption is applied
    per band over each bin's travel distance.
    """
    room, duration, unit = _traced_room(
        geometry, sources, mic_positions, fs, directions, max_order, duration, n_rays, receiver_radius, energy_threshold, air_absorption, seed, air
    )
    # the engine is driven directly: the room's own ray_tracing would reset an omnidirectional receiver to one bin
    for m, d in enumerate(unit):
        room.room_engine.microphones[m].set_directions(np.asarray(d, dtype=np.float32))
    histograms = [[] for _ in unit]
    for _ in _each_source_traced(room):
        for m, d in enumerate(unit):
            lines = [np.asarray(h.get_hist(), dtype=float) for h in room.room_engine.microphones[m].histograms]
            histograms[m].append(_directional_histogram(room, lines, d, duration, air_absorption))
    return histograms
