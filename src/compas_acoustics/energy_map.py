"""The sound pressure level of a scene's sources over many points.

Per point, the energy per octave band and 4 ms bin of the direct sound and the image sources up to
``image_source_order``, and of the rays past that order, weighted by each source's level at 1 m
(:attr:`compas_acoustics.Source.spl`). No impulse response is synthesised.
"""

from __future__ import annotations

import math
import numbers
import warnings

import numpy as np
from compas.data import Data
from scipy.spatial import cKDTree

from compas_acoustics.defaults import Defaults
from compas_acoustics.defaults import _PackageDtype

__all__ = ["EnergyMap", "map_rays"]



def map_rays(volume: float, duration: float) -> int:
    """The rays per source that give every point of a map 16000 hits, from 20000 to 250000.

    A ray of length ``c T`` crosses a receiver sphere of radius ``r = 0.5`` m about ``c T pi r^2 / V`` times in a
    diffuse field (Vorlaender 2008, Eq. 11.12, with the response length for the bin).
    """
    c = Defaults().DEFAULT_SPEED_OF_SOUND
    n = math.ceil(16000 * volume / (math.pi * 0.5 ** 2 * c * duration))
    return int(min(max(n, 20000), 250000))


class EnergyMap(_PackageDtype, Data):
    """The energy of a scene's sources at many points, per octave band and time bin, in Pa^2 s / p0^2.

    ``10 log10`` of the sum over the bins is the sound pressure level in dB re 20 uPa (:meth:`spl`).

    Parameters
    ----------
    points : array-like
        The points, (n, 3).
    inside : array-like
        Whether each point is in the air of the room, (n,); the others hold no energy and read NaN.
    bands : array-like
        The octave bands' centres in Hz, (b,).
    bin_size : float
        The length of a time bin in seconds; bin ``k`` starts ``k * bin_size`` after the emission.
    energy : array-like
        (n, b, bins).
    name : str, optional
    """

    def __init__(self, points, inside, bands, bin_size, energy, name=None):
        super().__init__(name=name)
        self.points = np.asarray(points, dtype=float).reshape(-1, 3)
        self.inside = np.asarray(inside, dtype=bool).reshape(-1)
        self.bands = np.asarray(bands, dtype=float).reshape(-1)
        self.bin_size = float(bin_size)
        self.energy = np.asarray(energy, dtype=float).reshape(len(self.points), len(self.bands), -1)

    @property
    def __data__(self):
        return {
            "points": self.points.tolist(),
            "inside": [bool(v) for v in self.inside],
            "bands": self.bands.tolist(),
            "bin_size": self.bin_size,
            "energy": self.energy.tolist(),
        }

    @classmethod
    def __from_data__(cls, data):
        return cls(data["points"], data["inside"], data["bands"], data["bin_size"], data["energy"])

    def spl(self) -> np.ndarray:
        """The sound pressure level per point and band, (n, b), in dB; NaN outside the room."""
        with np.errstate(divide="ignore"):
            levels = 10.0 * np.log10(self.energy.sum(axis=2))
        levels[~self.inside] = np.nan
        return levels

    def spl_a(self) -> np.ndarray:
        """The A-weighted total of the bands per point, (n,), in dB(A) (IEC 61672-1 weights at the band centres)."""
        from compas_acoustics.analyser import _A_WEIGHTS

        weights = np.array([_A_WEIGHTS[int(round(f))] for f in self.bands])
        with np.errstate(divide="ignore"):
            levels = 10.0 * np.log10(np.sum(self.energy.sum(axis=2) * 10.0 ** (weights / 10.0), axis=1))
        levels[~self.inside] = np.nan
        return levels

    def frames(self, frame_s: float = 0.012):
        """The broadband level per point in frames of ``frame_s`` seconds, for an animation of the energy arriving.

        A frame is a whole number of bins: ``frame_s`` is rounded up to one.

        Returns
        -------
        tuple of (np.ndarray, np.ndarray)
            The frames' start times (k,), in seconds, and the levels (n, k) in dB: -inf before any energy arrives,
            NaN outside the room.
        """
        per = max(1, math.ceil(frame_s / self.bin_size - 1e-9))
        broadband = self.energy.sum(axis=1)
        count = math.ceil(broadband.shape[1] / per)
        padded = np.pad(broadband, ((0, 0), (0, count * per - broadband.shape[1])))
        with np.errstate(divide="ignore"):
            levels = 10.0 * np.log10(padded.reshape(len(self.points), count, per).sum(axis=2))
        levels[~self.inside] = np.nan
        return np.arange(count) * per * self.bin_size, levels

    @classmethod
    def from_scene(cls, scene, points, source_index=None, n_rays=None, image_source_order=1,
                   duration=None, air_absorption=True, seed=True, ray_spacing=1.0) -> "EnergyMap":
        """The energy map of ``scene``'s sources at ``points``.

        Parameters
        ----------
        scene : :class:`compas_acoustics.Scene`
        points : iterable of xyz
            The points, in the room's coordinates.
        source_index : int, optional
            One source by its index in ``scene.sources``; by default every source, their energies added.
        n_rays : int, optional
            Rays per source; by default :func:`map_rays` of the room's volume and the response's length.
        image_source_order : int, default=1
            Image sources up to this order for every point, the rays past it. At least 1: at order 0 the ray
            tracer logs the direct path too, which would count it twice.
        duration : float, optional
            Seconds after the emission; by default 1.5 times the longest Eyring reverberation time, as the
            Mono method.
        air_absorption : bool, default=True
        seed : bool or int, default=True
            The rays' seed: True seeds with 42, an int with that value, False leaves them unseeded.
        ray_spacing : float, optional
            Default 1 m. The rays are traced to one receiver per cell of this size, the point of the cell nearest its centre, and
            their energy is interpolated to the other points; None traces them to every point. A run costs rays times
            receivers, the late energy varies slowly and a 0.5 m receiver sphere resolves nothing finer; the image
            sources stay exact at every point.

        Points outside the room, or inside a closed obstacle, are left out with one ``UserWarning``. The ray
        tracer's scattering over-estimates the late energy at high scattering coefficients.
        """
        from compas_acoustics.room.geometry import RoomGeometry
        from compas_acoustics.room.raytracing import default_duration
        from compas_acoustics.simulation_methods.base import _seed

        if image_source_order < 1:
            raise ValueError(f"image_source_order must be 1 or more, got {image_source_order!r}")
        if source_index is None:
            sources = list(scene.sources)
        else:
            if not isinstance(source_index, numbers.Integral) or not 0 <= source_index < len(scene.sources):
                raise ValueError(f"No source at index {source_index}; the scene has {len(scene.sources)} source(s).")
            sources = [scene.sources[source_index]]
        if not sources:
            raise ValueError("The scene has no source.")

        geometry = RoomGeometry.from_scene(scene)
        points = np.asarray([list(p) for p in points], dtype=float).reshape(-1, 3)
        inside = np.array([geometry.is_inside(p) for p in points], dtype=bool)
        if not inside.all():
            warnings.warn(f"{int((~inside).sum())} of {len(points)} points are outside the room or inside an obstacle and are left out.",
                          UserWarning, stacklevel=2)
        if duration is None:
            duration = default_duration(geometry, air_absorption)
        bands = np.asarray(Defaults().DEFAULT_FREQUENCY_BANDS, dtype=float)

        if n_rays is None:
            n_rays = map_rays(geometry.volume, duration)
        fs = int(scene.sampling_rate)
        mapped = points[inside]
        receivers = mapped if ray_spacing is None else _ray_receivers(mapped, ray_spacing)
        rays = _ray_energy(geometry, sources, receivers, fs, image_source_order, duration, n_rays, air_absorption, _seed(seed), bands)
        if receivers is not mapped:
            rays = _interpolate(receivers, rays, mapped)
        early = _image_source_energy(geometry, sources, mapped, fs, image_source_order, air_absorption, bands, rays.shape[2], 0.004)
        energy = np.zeros((len(points), len(bands), rays.shape[2]))
        energy[inside] = rays + early
        result = cls(points, inside, bands, 0.004, energy)
        # the rays traced, for a caller's message; not data of the map
        result.n_rays = int(n_rays)
        return result



def _ray_receivers(points: np.ndarray, spacing: float) -> np.ndarray:
    """One point per cell of ``spacing``: the point of the cell nearest the mean of the cell's points, in their order."""
    if not len(points):
        return points
    cells = np.floor(points / spacing).astype(np.int64)
    _, cell = np.unique(cells, axis=0, return_inverse=True)
    cell = np.asarray(cell).reshape(-1)
    chosen = []
    for c in range(cell.max() + 1):
        members = np.flatnonzero(cell == c)
        centre = points[members].mean(axis=0)
        chosen.append(members[np.argmin(np.linalg.norm(points[members] - centre, axis=1))])
    return points[np.sort(chosen)]


def _interpolate(receivers: np.ndarray, energy: np.ndarray, points: np.ndarray) -> np.ndarray:
    """The energy at ``points`` (n, 3) from the receivers' (r, b, bins): the four nearest by inverse squared distance."""
    if not len(points):
        return np.zeros((0,) + energy.shape[1:])
    k = min(4, len(receivers))
    distance, index = cKDTree(receivers).query(points, k=k)
    distance = np.asarray(distance, dtype=float).reshape(len(points), k)
    index = np.asarray(index).reshape(len(points), k)
    with np.errstate(divide="ignore"):
        weights = 1.0 / distance ** 2
    # a point on a receiver takes the receiver's energy alone
    exact = distance[:, 0] < 1e-9
    weights[exact] = 0.0
    weights[exact, 0] = 1.0
    weights /= weights.sum(axis=1, keepdims=True)
    return np.einsum("nk,nkbt->nbt", weights, energy[index])


def _level_weights(source, bands) -> np.ndarray:
    """Each band's factor from the package scale to Pa^2 s / p0^2: the free field at 1 m on axis, 1 / (16 pi^2), is spl."""
    spl = np.asarray(source.spl, dtype=float)
    by_band = dict(zip(Defaults().DEFAULT_FREQUENCY_BANDS, spl))
    return np.array([10.0 ** ((by_band[int(f)] + 20.0 * np.log10(4.0 * np.pi)) / 10.0) for f in bands])


def _ray_energy(geometry, sources, mics, fs, order, duration, n_rays, air_absorption, seed, bands) -> np.ndarray:
    """The rays' energy past ``order`` at every microphone, weighted by each source's level, (m, b, bins)."""
    from compas_acoustics.room.pra import source_to_pra
    from compas_acoustics.room.raytracing import directional_histograms
    from compas_acoustics.worker import jobs

    n_bins = int(np.ceil(duration / 0.004))
    total = np.zeros((len(mics), len(bands), n_bins))
    if not len(mics):
        return total
    jobs.progress("energy map: rays")
    histograms = directional_histograms(
        geometry, [source_to_pra(s) for s in sources], mics.T, fs, [np.array([[1.0, 0.0, 0.0]])] * len(mics),
        max_order=order, duration=duration, n_rays=n_rays, air_absorption=air_absorption, seed=seed,
    )
    for m, per_source in enumerate(histograms):
        for source, histogram in zip(sources, per_source):
            index = [int(np.argmin(np.abs(histogram.bands - f))) for f in bands]
            energy = histogram.energy.sum(axis=0)[index, :n_bins]
            total[m, :, :energy.shape[1]] += energy * _level_weights(source, bands)[:, np.newaxis]
    return total


def _image_source_energy(geometry, sources, mics, fs, order, air_absorption, bands, n_bins, bin_size) -> np.ndarray:
    """The direct sound and image sources up to ``order`` at every microphone, weighted by each source's level, (m, b, bins).

    The room is built for ray tracing, as the rays' run is, so an image carries its specular part ``prod sqrt(1 - s)``
    and the rays the scattered rest.
    """
    import pyroomacoustics as pra

    from compas_acoustics.room.air import AirConditions
    from compas_acoustics.room.geometry import pra_octave_grid
    from compas_acoustics.room.pra import source_to_pra
    from compas_acoustics.worker import jobs

    total = np.zeros((len(mics), len(bands), n_bins))
    if not len(mics):
        return total
    jobs.progress("energy map: image sources")
    with pra_octave_grid():
        room = geometry.to_pra(fs, order, ray_tracing=True, air_absorption=False)
        pra_sources = [source_to_pra(s) for s in sources]
        for source in pra_sources:
            room.add_soundsource(source)
        room.add_microphone_array(pra.MicrophoneArray(mics.T, fs))
        room.image_source_model()
        centres = np.asarray(room.octave_bands.centers, dtype=float)
    index = [int(np.argmin(np.abs(centres - f))) for f in bands]
    air = AirConditions().air_attenuation(bands) if air_absorption else np.zeros(len(bands))
    c = float(room.c)
    tolerance = 1e-5 * max(float(np.ptp(geometry.vertices, axis=0).max()), 1.0)
    for s, (source, pra_source) in enumerate(zip(sources, room.sources)):
        images = np.asarray(pra_source.images, dtype=float)
        damping = np.asarray(pra_source.damping, dtype=float)
        if damping.shape[0] == 1:
            damping = np.repeat(damping, len(centres), axis=0)
        weights = _level_weights(source, bands)
        for m, mic in enumerate(mics):
            visible = np.flatnonzero(np.asarray(room.visibility[s][m], dtype=bool))
            if not visible.size:
                continue
            # an image reached through an edge is found from both wall orderings: keep it once
            keep = np.ones(visible.size, dtype=bool)
            for i, j in cKDTree(images[:, visible].T).query_pairs(tolerance):
                keep[max(i, j)] = False
            visible = visible[keep]
            distance = np.linalg.norm(images[:, visible] - mic[:, np.newaxis], axis=0)
            gain = damping[index][:, visible] ** 2
            if pra_source.directivity is not None:
                emitted = np.asarray(pra_source.directions[m], dtype=float)[:, visible]
                response = pra_source.directivity.get_response(
                    azimuth=np.arctan2(emitted[1], emitted[0]),
                    colatitude=np.arccos(np.clip(emitted[2] / np.linalg.norm(emitted, axis=0), -1.0, 1.0)),
                    degrees=False,
                )
                gain = gain * np.asarray(response, dtype=float).reshape(1, -1) ** 2
            energy = gain * np.exp(-np.outer(air, distance)) / (16.0 * np.pi ** 2 * distance ** 2)
            bins = np.minimum((distance / c / bin_size).astype(int), n_bins - 1)
            for b in range(len(bands)):
                np.add.at(total[m, b], bins, energy[b] * weights[b])
    return total
