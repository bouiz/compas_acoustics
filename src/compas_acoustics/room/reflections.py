"""The walls of every image source, read back from one search, and the damping another build of the room gives them.

The image source search walks the tree of reflections depth first and exports, per visible image, only the wall that
generated it. Its gain per band is a float32 product along all its walls from the source outward: ``sqrt(1 - a)`` per
wall, times ``sqrt(1 - s)`` where the wall scatters and the room is built for ray tracing. A room built with other
coefficients or without ray tracing finds the same images, so one search serves several builds once each image's walls
are known.
"""

from __future__ import annotations

from typing import List
from typing import Optional

import numpy as np
import pyroomacoustics as pra

__all__ = ["damping_as_searched"]

# Generous on purpose: a candidate path only has to pass these to be tried; the search's own damping decides.
_SEGMENT_TOLERANCE = 1e-6
_POLYGON_TOLERANCE = 1e-4
_SOURCE_TOLERANCE = 1e-4


def damping_as_searched(searched: pra.Room, other: pra.Room) -> List[np.ndarray]:
    """Per source, the damping ``(n_bands, n_images)`` that ``other``'s search would give the images of ``searched``.

    Both rooms hold the same walls, sources and microphones; they may differ in their coefficients and in
    ``ray_tracing``, neither of which moves the search. Each image's walls are traced back from a receiver that sees
    it (:func:`_wall_sequences`) and kept only where their product reproduces ``searched``'s own damping bit for bit;
    ``other``'s product along them is then that search's, operation for operation. Where an image's walls are not
    singled out, as on a path through a corner of three walls that the search finds from two wall orders, ``other``
    is searched itself.
    """
    walls = _Walls(searched.room_engine.walls)
    mine = _Coefficients(searched.room_engine)
    theirs = _Coefficients(other.room_engine)
    tolerance = _SOURCE_TOLERANCE * max(float(np.ptp(np.hstack([w.corners for w in searched.walls]), axis=1).max()), 1.0)
    mics = np.asarray(searched.mic_array.R, dtype=float)
    damping = []
    for s, source in enumerate(searched.sources):
        result = _damping_of_source(source, np.asarray(searched.visibility[s], dtype=bool), mics, walls, mine, theirs, tolerance)
        if result is None:
            return _searched(other)
        damping.append(result)
    return damping


def _searched(room: pra.Room) -> List[np.ndarray]:
    room.image_source_model()
    return [np.asarray(source.damping) for source in room.sources]


def _damping_of_source(source, visible: np.ndarray, mics: np.ndarray, walls, mine, theirs, tolerance) -> Optional[np.ndarray]:
    own = np.asarray(source.damping)
    out = np.array(own, copy=True)
    if not visible.any():
        return out
    images = np.asarray(source.images, dtype=float)
    orders = np.asarray(source.orders, dtype=int).ravel()
    last = np.asarray(source.walls, dtype=int).ravel()
    origin = np.asarray(source.position, dtype=float).ravel()
    if own.shape[0] != mine.n_bands and np.any(orders > 0):
        return None
    for k in np.flatnonzero(orders > 0):
        mic = mics[:, int(np.flatnonzero(visible[:, k])[0])]
        fits = [w for w in _wall_sequences(walls, images[:, k], int(orders[k]), int(last[k]), mic, origin, tolerance) if np.array_equal(mine.damping(w), own[:, k])]
        values = {theirs.damping(w).tobytes() for w in fits}
        if len(values) != 1:
            return None
        out[:, k] = theirs.damping(fits[0])
    return out


class _Walls:
    """The search's walls as planes and polygons, in float64."""

    def __init__(self, walls):
        self.normal = np.array([np.asarray(w.normal, dtype=float) for w in walls])
        self.origin = np.array([np.asarray(w.origin, dtype=float) for w in walls])
        self.basis = [np.asarray(w.basis, dtype=float) for w in walls]
        self.flat = [np.asarray(w.flat_corners, dtype=float) for w in walls]

    def crossing(self, index, start, end):
        """Where the segment from ``start`` to ``end`` meets the plane of wall ``index``, as ``(t, point)``, or None."""
        direction = end - start
        across = self.normal[index] @ direction
        if abs(across) < 1e-12:
            return None
        t = self.normal[index] @ (self.origin[index] - start) / across
        return t, start + t * direction

    def contains(self, index, point) -> bool:
        """Whether ``point``, on the wall's plane, lies in its polygon or within ``_POLYGON_TOLERANCE`` of its border."""
        x, y = self.basis[index].T @ (point - self.origin[index])
        corners = self.flat[index]
        start = corners
        end = np.roll(corners, -1, axis=1)
        side = end - start
        along = np.clip(((x - start[0]) * side[0] + (y - start[1]) * side[1]) / np.sum(side**2, axis=0), 0.0, 1.0)
        if np.min(np.hypot(start[0] + along * side[0] - x, start[1] + along * side[1] - y)) < _POLYGON_TOLERANCE:
            return True
        straddles = (start[1] > y) != (end[1] > y)
        with np.errstate(divide="ignore", invalid="ignore"):
            cuts = x < start[0] + (y - start[1]) * side[0] / side[1]
        return bool(np.count_nonzero(straddles & cuts) % 2)

    def reflect(self, index, point):
        return point - 2.0 * (self.normal[index] @ (point - self.origin[index])) * self.normal[index]


def _wall_sequences(walls: _Walls, image, order: int, last: int, mic, source, tolerance) -> list:
    """Every sequence of walls, from the source outward, whose path from ``mic`` to ``image`` the search could accept.

    The search accepts an image when the segment from the receiver to it crosses its generating wall and, from that
    point, the segment to its parent crosses the parent's, down to the source. Read backwards, each step mirrors the
    image across its wall into its parent and tries every wall that could have generated that parent: one the parent
    lies in front of, crossed by the segment from the last reflection point to the parent.
    """
    found = []

    def walk(point, image, wall, remaining, sequence):
        hit = walls.crossing(wall, point, image)
        if hit is None or not (-_SEGMENT_TOLERANCE <= hit[0] <= 1.0 + _SEGMENT_TOLERANCE) or not walls.contains(wall, hit[1]):
            return
        parent = walls.reflect(wall, image)
        sequence = sequence + (wall,)
        if remaining == 1:
            if np.linalg.norm(parent - source) < tolerance:
                found.append(sequence[::-1])
            return
        in_front = np.flatnonzero((walls.normal * (parent - walls.origin)).sum(axis=1) > 0.0)
        for candidate in in_front[in_front != wall]:
            walk(hit[1], parent, int(candidate), remaining - 1, sequence)

    walk(np.asarray(mic, dtype=float), np.asarray(image, dtype=float), last, order, ())
    return found


class _Coefficients:
    """A built room's per-wall factors, as its search forms them in float32."""

    def __init__(self, engine):
        one = np.float32(1.0)
        walls = engine.walls
        self.transmission = [np.sqrt(one - np.asarray(w.absorption, dtype=np.float32)) for w in walls]
        self.specular = [np.sqrt(one - np.asarray(w.scatter, dtype=np.float32)) if engine.is_hybrid_sim and np.asarray(w.scatter).max() > 0.0 else None for w in walls]
        self.n_bands = self.transmission[0].size if walls else 0

    def damping(self, sequence) -> np.ndarray:
        attenuation = np.ones(self.n_bands, dtype=np.float32)
        for wall in sequence:
            attenuation = attenuation * self.transmission[wall]
            if self.specular[wall] is not None:
                attenuation = attenuation * self.specular[wall]
        return attenuation
