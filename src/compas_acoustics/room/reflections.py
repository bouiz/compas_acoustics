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
from scipy.spatial import cKDTree

__all__ = ["damping_as_searched", "first_of_each_position", "position_tolerance"]


def position_tolerance(geometry) -> float:
    """The distance within which two images of ``geometry``'s search are one position: 1e-5 of its extent, at least 1.

    The package's choice: well above the float32 rounding of one image reached twice and far below the distance
    between two distinct images. ``geometry`` is a :class:`~compas_acoustics.room.geometry.RoomGeometry`.
    """
    return 1e-5 * max(float(np.ptp(geometry.vertices, axis=0).max()), 1.0)


def first_of_each_position(images: np.ndarray, geometry) -> np.ndarray:
    """Mask over ``images`` (3, M) keeping the first of every group within :func:`position_tolerance`.

    A path passing exactly through an edge is found from both wall orderings.
    """
    keep = np.ones(images.shape[1], dtype=bool)
    if images.shape[1] < 2:
        return keep
    for i, j in cKDTree(images.T).query_pairs(position_tolerance(geometry)):
        keep[max(i, j)] = False
    return keep


def damping_as_searched(searched: pra.Room, other: pra.Room) -> List[np.ndarray]:
    """Per source, the damping ``(n_bands, n_images)`` that ``other``'s search would give the images of ``searched``.

    Both rooms hold the same walls, sources and microphones; they may differ in their coefficients and in
    ``ray_tracing``, neither of which moves the search. For each image, wall sequences are proposed by tracing its path
    back from a receiver that sees it (:func:`_wall_sequences`) and kept where the search's own functions accept them:
    its mirrorings give the image's position, its visibility test passes at that receiver, and its float32 product
    gives the image's damping, each bit for bit (:class:`_Search`). A kept sequence is therefore an image the search
    exported there. When as many are kept as the search exported at that position off that wall with that damping, the image's own
    is among them, and if ``other``'s product is the same along all of them it is that search's damping, operation for
    operation. Otherwise, as where a path through a corner of three walls is found from two wall orders whose
    products round differently, ``other`` is searched itself.
    """
    extent = max(float(np.ptp(np.hstack([w.corners for w in searched.walls]), axis=1).max()), 1.0)
    # How far outside the search's own tests a path may fall and still be proposed, 1e-4 of the room's extent. It
    # only has to be wide; what is proposed is then put to the search's tests themselves (_Search).
    walls = _Walls(searched.room_engine.walls, 1e-4 * extent)
    search = _Search(searched.room_engine)
    theirs = _Coefficients(other.room_engine)
    mics = np.asarray(searched.mic_array.R)
    damping = []
    for s, source in enumerate(searched.sources):
        result = _damping_of_source(source, np.asarray(searched.visibility[s], dtype=bool), mics, walls, search, theirs)
        if result is None:
            other.image_source_model()
            return [np.asarray(source.damping) for source in other.sources]
        damping.append(result)
    return damping


def _damping_of_source(source, visible: np.ndarray, mics: np.ndarray, walls, search, theirs) -> Optional[np.ndarray]:
    own = np.asarray(source.damping)
    out = np.array(own, copy=True)
    if not visible.any():
        return out
    images = np.asarray(source.images, dtype=np.float32)
    orders = np.asarray(source.orders, dtype=int).ravel()
    last = np.asarray(source.walls, dtype=int).ravel()
    position = np.asarray(source.position, dtype=float).ravel()
    if own.shape[0] != search.coefficients.n_bands and np.any(orders > 0):
        return None
    for k in np.flatnonzero(orders > 0):
        seen = np.flatnonzero(visible[:, k])
        if not seen.size:
            # found from a microphone that pyroomacoustics then declared outside the room
            return None
        mic = mics[:, int(seen[0])]
        proposed = _wall_sequences(walls, images[:, k].astype(float), int(orders[k]), int(last[k]), mic.astype(float), position)
        kept = [w for w in proposed if search.exported(w, position, mic, images[:, k], own[:, k])]
        # the images a kept sequence can be: the search's at this position off this last wall, with this damping
        twins = visible[seen[0]] & (last == last[k]) & np.all(images == images[:, [k]], axis=0) & np.all(own == own[:, [k]], axis=0)
        values = {theirs.damping(w).tobytes() for w in kept}
        if len(kept) != np.count_nonzero(twins) or len(values) != 1:
            return None
        out[:, k] = theirs.damping(kept[0])
    return out


class _Walls:
    """The search's walls as planes and polygons, in float64, with its tests widened by ``tolerance`` metres.

    The search takes a segment to cross a plane for a parameter within its epsilon of [0, 1], and a point
    to lie on a polygon's edge where the cross product with the edge is below that epsilon, so up to epsilon over the
    edge's length outside it. In float32 it also accepts points past a corner of a slender polygon, which no margin
    here is sure to cover: a path these tests miss costs a search of the other room, never a wrong value.
    """

    def __init__(self, walls, tolerance: float):
        self.tolerance = tolerance
        self.epsilon = 2.0 * float(pra.libroom.get_eps())
        self.normal = np.array([np.asarray(w.normal, dtype=float) for w in walls])
        self.origin = np.array([np.asarray(w.origin, dtype=float) for w in walls])
        self.basis = [np.asarray(w.basis, dtype=float) for w in walls]
        self.flat = [np.asarray(w.flat_corners, dtype=float) for w in walls]

    def crossing(self, index, start, end):
        """Where the segment from ``start`` to ``end`` crosses the plane of wall ``index``, or None."""
        direction = end - start
        across = self.normal[index] @ direction
        if abs(across) < 1e-12:
            return None
        t = self.normal[index] @ (self.origin[index] - start) / across
        slack = self.epsilon + self.tolerance / max(float(np.linalg.norm(direction)), 1e-12)
        return start + t * direction if -slack <= t <= 1.0 + slack else None

    def contains(self, index, point) -> bool:
        """Whether ``point``, on the wall's plane, lies in its polygon or near an edge."""
        x, y = self.basis[index].T @ (point - self.origin[index])
        start = self.flat[index]
        end = np.roll(start, -1, axis=1)
        side = end - start
        with np.errstate(divide="ignore", invalid="ignore"):
            length = np.hypot(side[0], side[1])
            along = np.clip(((x - start[0]) * side[0] + (y - start[1]) * side[1]) / length**2, 0.0, 1.0)
            near = np.hypot(start[0] + along * side[0] - x, start[1] + along * side[1] - y) < np.maximum(self.tolerance, self.epsilon / length)
            cuts = x < start[0] + (y - start[1]) * side[0] / side[1]
        straddles = (start[1] > y) != (end[1] > y)
        return bool(np.any(near) or np.count_nonzero(straddles & cuts) % 2)

    def in_front(self, point) -> np.ndarray:
        """The walls ``point`` can be the mirror image across: those whose normal side it lies on."""
        return np.flatnonzero(((point - self.origin) * self.normal).sum(axis=1) > -self.tolerance)

    def reflect(self, index, point):
        return point - 2.0 * (self.normal[index] @ (point - self.origin[index])) * self.normal[index]


def _wall_sequences(walls: _Walls, image, order: int, last: int, mic, source) -> list:
    """Sequences of walls, from the source outward, whose path from ``mic`` to ``image`` the search may have accepted.

    The search accepts an image when the segment from the receiver to it crosses its generating wall and, from that
    point, the segment to its parent crosses the parent's, down to the source. Read backwards, each step mirrors the
    image across its wall into its parent and tries every wall that could have generated that parent: one the parent
    lies in front of, crossed by the segment from the last reflection point to the parent.
    """
    found = []

    def walk(point, image, wall, remaining, sequence):
        hit = walls.crossing(wall, point, image)
        if hit is None or not walls.contains(wall, hit):
            return
        parent = walls.reflect(wall, image)
        sequence = sequence + (wall,)
        if remaining == 1:
            if np.linalg.norm(parent - source) < walls.tolerance:
                found.append(sequence[::-1])
            return
        for candidate in walls.in_front(parent):
            if candidate != wall:
                walk(hit, parent, int(candidate), remaining - 1, sequence)

    walk(mic, image, last, order, ())
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


class _Search:
    """The search's own verdict on a wall sequence, from its walls' functions and in its float32.

    The search mirrors the source across each wall in turn, where the point lies behind it, and takes the image as
    visible from a receiver when, level by level from the image down to the source, nothing obstructs the segment
    from the point reached to the level's image and that segment crosses the level's generating wall.
    """

    def __init__(self, engine):
        self.walls = engine.walls
        self.obstructing = [int(w) for w in engine.obstructing_walls]
        self.coefficients = _Coefficients(engine)

    def exported(self, sequence, source, mic, image, damping) -> bool:
        """Whether the search made this sequence the image at ``image`` with ``damping``, seen from ``mic``."""
        if not np.array_equal(self.coefficients.damping(sequence), damping):
            return False
        levels = [np.asarray(source, dtype=np.float32)]
        for wall in sequence:
            mirrored = np.zeros(3, dtype=np.float32)
            if self.walls[wall].reflect(levels[-1], mirrored) <= 0:
                return False
            levels.append(mirrored)
        if not np.array_equal(levels[-1], image):
            return False
        point = np.asarray(mic, dtype=np.float32)
        for level in range(len(sequence), -1, -1):
            generating = sequence[level - 1] if level else -1
            if self._obstructed(point, levels[level], generating):
                return False
            if level:
                crossing = np.zeros(3, dtype=np.float32)
                if self.walls[generating].intersection(point, levels[level], crossing) < 0:
                    return False
                point = crossing
        return True

    def _obstructed(self, point, image, generating: int) -> bool:
        crossing = np.zeros(3, dtype=np.float32)
        for wall in self.obstructing:
            # 0: within the segment and the polygon, 2: on the polygon's border
            if wall == generating or self.walls[wall].intersection(image, point, crossing) not in (0, 2):
                continue
            if generating < 0:
                return True
            # an obstruction on the image's side of its own wall lies behind it, as in a corner
            side = self.walls[generating].side(crossing)
            if side != 0 and side != self.walls[generating].side(image):
                return True
        return False
