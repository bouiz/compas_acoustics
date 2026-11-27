"""The specular and diffracted paths a simulation finds, kept for display at no extra search.

A display of a run's paths (:meth:`~compas_acoustics.rays.SpecularReflections.from_scene`,
:func:`~compas_acoustics.rays.diffracted_paths`) searches the image sources and the diffracting edges a second time,
which on a room with many faces, or with diffraction on, costs as much as the run's own search. While
:func:`record_paths` is active, every image source search of a run
(:func:`~compas_acoustics.room.raytracing.search_images`) traces its visible images up to ``max_order`` back to
their source, and every diffraction search (:func:`~compas_acoustics.hybrid.diffraction.diffraction_paths`) keeps
its paths, so they come out of the run:

>>> import compas_acoustics as ca
>>> with ca.record_paths(max_order=1) as recorded:  # doctest: +SKIP
...     results = scene.simulate(ca.Mono())
>>> reflections = recorded.reflections(scene)  # doctest: +SKIP

The recorder is one per process (the hybrid renders its pairs on threads); the per-source processes of
:func:`~compas_acoustics.room.raytracing.emitter_rirs` record in their own and hand their paths back. A pair searched
twice (a reference run and its renderer) keeps its first record. A reciprocal run, whose receivers emit, gives its
paths reversed, source first.
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from typing import Dict
from typing import Iterator
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np

__all__ = ["PathRecorder", "record_paths"]

# Positions are matched to this many decimals (metres): a source or receiver is placed once per run.
_KEY_DECIMALS = 6

_lock = threading.Lock()
_active: Optional["PathRecorder"] = None

_Key = Tuple[float, float, float]


def _key(position) -> _Key:
    p = np.round(np.asarray(position, dtype=float).reshape(3), _KEY_DECIMALS)
    return (float(p[0]), float(p[1]), float(p[2]))


def active() -> Optional["PathRecorder"]:
    """The recorder of the enclosing :func:`record_paths`, or None."""
    return _active


class PathRecorder:
    """The paths recorded during :func:`record_paths`, by pair of positions.

    Parameters
    ----------
    max_order : int
        The last reflection order of the specular paths kept (0 = the direct sound only).
    """

    def __init__(self, max_order: int = 1):
        if max_order < 0:
            raise ValueError(f"max_order must be 0 or more, not {max_order}.")
        self.max_order = int(max_order)
        # (emitter, microphone) -> [(points, order, walls, image)], points from the emitter
        self._specular: Dict[Tuple[_Key, _Key], list] = {}
        # (source, receiver) -> [(points, delay, distance, band gain without spreading, zone, order, edge)]
        self._diffracted: Dict[Tuple[_Key, _Key], list] = {}

    # ── recording ──────────────────────────────────────────────────────────────────────────────────────────────────

    def add_room(self, room) -> None:
        """Trace the visible images of a searched pyroomacoustics ``room`` up to :attr:`max_order`, for each of its
        source-microphone pairs not recorded yet."""
        from compas_acoustics.rays import room_paths

        corners = np.concatenate([np.asarray(w.corners, dtype=float).reshape(3, -1) for w in room.walls], axis=1)
        tolerance = 1e-5 * max(float(np.ptp(corners, axis=1).max()), 1.0)  # room.reflections.position_tolerance
        mics = np.asarray(room.mic_array.R, dtype=float)
        for s, source in enumerate(room.sources):
            for m in range(mics.shape[1]):
                key = (_key(source.position), _key(mics[:, m]))
                with _lock:
                    if key in self._specular:
                        continue
                found, _ = room_paths(room, s, m, self.max_order, tolerance)
                with _lock:
                    self._specular.setdefault(key, found)

    def add_diffracted(self, source, receiver, paths) -> None:
        """Keep a pair's :class:`~compas_acoustics.hybrid.diffraction.DiffractionPath` list (its first record)."""
        rows = [
            (
                [[float(v) for v in p] for p in path.points],
                float(path.delay),
                float(path.distance),
                [float(g) for g in np.asarray(path.band_gain, dtype=float).ravel()],
                str(path.zone),
                int(path.order),
                int(path.edge),
            )
            for path in paths
        ]
        with _lock:
            self._diffracted.setdefault((_key(source), _key(receiver)), rows)

    def state(self) -> dict:
        """The records, picklable: what a worker process hands back (:meth:`merge`)."""
        with _lock:
            return {"specular": dict(self._specular), "diffracted": dict(self._diffracted)}

    def merge(self, state: Optional[dict]) -> None:
        """Add another recorder's :meth:`state`, keeping the pairs already recorded here."""
        if not state:
            return
        with _lock:
            for key, rows in state.get("specular", {}).items():
                self._specular.setdefault(key, rows)
            for key, rows in state.get("diffracted", {}).items():
                self._diffracted.setdefault(key, rows)

    # ── reading ────────────────────────────────────────────────────────────────────────────────────────────────────

    def _specular_rows(self, source, receiver) -> Optional[list]:
        """A pair's rows with points from the source; None when the pair was not searched."""
        s, r = _key(source), _key(receiver)
        if (s, r) in self._specular:
            return self._specular[(s, r)]
        if (r, s) in self._specular:  # a reciprocal run: the receiver emitted
            return [(points[::-1], order, walls[::-1], None) for points, order, walls, _ in self._specular[(r, s)]]
        return None

    def reflections(self, scene, geometry=None, air=None):
        """The recorded specular paths of ``scene``'s pairs, as :meth:`SpecularReflections.from_scene` gives them.

        ``geometry`` (default ``RoomGeometry.from_scene(scene)``) gives the walls' materials, ``air`` (default
        ``AirConditions()``) the air attenuation of :attr:`SpecularPath.band_gain`. A pair the run did not search has
        no path. The image of a reversed path is mirrored back from the source side.
        """
        from compas_acoustics.rays import SpecularPath
        from compas_acoustics.rays import SpecularReflections
        from compas_acoustics.rays import specular_band_gain
        from compas_acoustics.room.air import AirConditions
        from compas_acoustics.room.geometry import SYNTHESIS_BANDS
        from compas_acoustics.room.geometry import RoomGeometry

        geometry = RoomGeometry.from_scene(scene) if geometry is None else geometry
        attenuation = (air or AirConditions()).air_attenuation(list(SYNTHESIS_BANDS))
        paths = []
        for s, source in enumerate(scene.sources):
            for r, receiver in enumerate(scene.receivers):
                for points, order, walls, image in self._specular_rows(source.position, receiver.position) or []:
                    points = [np.asarray(p, dtype=float) for p in points]
                    if image is None:  # reversed: the image of the source seen from the receiver
                        image = _image_from(points)
                    paths.append(
                        SpecularPath(
                            points,
                            order=int(order),
                            walls=walls,
                            image=image,
                            receiver_index=r,
                            band_gain=specular_band_gain(geometry, points, walls, attenuation),
                            source=source,
                            source_index=s,
                            ray_index=len(paths),
                        )
                    )
        return SpecularReflections(paths)

    def diffracted(self, scene) -> list:
        """The recorded diffracted paths of ``scene``'s pairs, as :func:`~compas_acoustics.rays.diffracted_paths`
        gives them (the run's source pattern included in ``band_gain``)."""
        from compas_acoustics.rays import DiffractedPath

        out = []
        for s, source in enumerate(scene.sources):
            for r, receiver in enumerate(scene.receivers):
                for points, delay, distance, gain, zone, order, edge in self._diffracted.get((_key(source.position), _key(receiver.position)), []):
                    out.append(
                        DiffractedPath(
                            points=points,
                            source_index=s,
                            receiver_index=r,
                            delay=delay,
                            distance=distance,
                            band_gain=[g / (4.0 * np.pi * distance) for g in gain],
                            zone=zone,
                            order=order,
                            edge=edge,
                        )
                    )
        return out


def _image_from(points: List[np.ndarray]) -> np.ndarray:
    """The image of a path's source: along the last leg's line from the receiver, the path's length away."""
    end = points[-1]
    length = float(sum(np.linalg.norm(b - a) for a, b in zip(points[:-1], points[1:])))
    leg = points[-2] - end
    norm = float(np.linalg.norm(leg))
    return end + leg / norm * length if norm > 0 else end


@contextmanager
def record_paths(max_order: int = 1) -> Iterator[PathRecorder]:
    """Record the paths of the simulations run inside the block (one recorder per process; not nestable)."""
    global _active
    recorder = PathRecorder(max_order)
    with _lock:
        if _active is not None:
            raise RuntimeError("record_paths is already active in this process")
        _active = recorder
    try:
        yield recorder
    finally:
        with _lock:
            _active = None
