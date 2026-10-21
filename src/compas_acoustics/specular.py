import numbers
import warnings
from typing import List
from typing import Optional
from typing import Union

import numpy as np
from compas.geometry import Point

from compas_acoustics.ray import Ray
from compas_acoustics.ray import RayField
from compas_acoustics.ray import _inside

__all__ = ['SpecularPath', 'SpecularReflections']


class SpecularPath(Ray):
    """A specular (image-source) reflection path from a source to a receiver: a ray that ends at the receiver.

    The vertices are ``[source_position, hit_1, ..., hit_order, receiver_position]``.

    Parameters
    ----------
    points : list
        The polyline vertices, from the source to the receiver.
    order : int
        The reflection order, the number of walls the path is reflected by.
    walls : list of int
        The wall index of each reflection, in path order: an index into the faces of
        ``RoomGeometry.from_scene(scene)``, obstacle copies included (:attr:`RoomGeometry.faces`).
    image : :class:`compas.geometry.Point` or list
        The image source of the path.
    receiver_index : int
        The index of the receiver in the scene.
    **kwargs
        Forwarded to :class:`compas_acoustics.Ray`: ``source``, ``source_index`` (the source's index in the
        scene), ``ray_index``, ``name``.
    """

    def __init__(self, points, order: int, walls: List[int], image, receiver_index: int, **kwargs):
        super().__init__(points, **kwargs)
        self.order = order
        self.walls = list(walls)
        self.image = Point(*image)
        self.receiver_index = receiver_index

    @property
    def __data__(self):
        data = super().__data__
        data.update(order=self.order, walls=self.walls, image=list(self.image), receiver_index=self.receiver_index)
        return data

    @classmethod
    def __from_data__(cls, data):
        data = dict(data)
        data.pop("wall_names", None)  # stored by the first version
        return super().__from_data__(data)

    @property
    def reflection_order(self) -> int:
        """The number of walls the path is reflected by: its last segment ends at the receiver, not a wall."""
        return self.order

    def _clone_with_points(self, points) -> Ray:
        # A cut path no longer reaches its receiver: a plain ray
        return Ray(points, source=self.source, source_index=self.source_index, ray_index=self.ray_index,
                   direction=self.direction, name=self.name)


class SpecularReflections(RayField):
    """The specular reflection paths between sources and receivers: a ray field whose rays are :class:`SpecularPath`.

    As a :class:`compas_acoustics.RayField` it plays in a billiard (:meth:`particles`), truncates and displays
    like traced rays.

    Parameters
    ----------
    paths : list of :class:`SpecularPath`, optional
        The reflection paths.
    name : str, optional
        The name of the collection.
    """

    def __init__(self, paths: Optional[List[SpecularPath]] = None, name: Optional[str] = None):
        super().__init__(paths, name=name)

    @classmethod
    def __from_data__(cls, data):
        data = dict(data)
        if "rays" not in data:  # stored by the first version
            data["rays"] = data.pop("paths")
        return super().__from_data__(data)

    @property
    def paths(self) -> List[SpecularPath]:
        """The reflection paths, the field's rays."""
        return self.rays

    @classmethod
    def from_scene(cls, scene, max_order: int = 3) -> "SpecularReflections":
        """The specular paths from every source to every receiver of ``scene``, orders 0 to ``max_order``.

        The paths are those of the image sources pyroomacoustics finds visible for each pair, searched in the
        room :meth:`compas_acoustics.room_geometry.RoomGeometry.to_pra` builds. pyroomacoustics records only
        the last wall of each image, so each path is traced back from the receiver: through the image's
        last wall, across it to the parent image, and on through whichever wall leads back to the source.
        An image found more than once (a path through an edge, from both wall orderings) is kept once.
        Sources and receivers outside the room are left out with a ``UserWarning``.

        Parameters
        ----------
        scene : :class:`compas_acoustics.Scene`
            The scene whose room, sources and receivers the paths are found for.
        max_order : int, default=3
            The highest reflection order, 0 or more.

        Returns
        -------
        :class:`SpecularReflections`

        Raises
        ------
        ValueError
            If ``max_order`` is negative, or pyroomacoustics cannot build the room.
        """
        from compas_acoustics.defaults import Defaults
        from compas_acoustics.room_geometry import RoomGeometry
        from compas_acoustics.simulation_methods.base import _warn_if_the_search_is_slow

        if max_order < 0:
            raise ValueError(f"max_order must be 0 or more, not {max_order}.")
        geometry = RoomGeometry.from_scene(scene)
        sources = _inside(geometry, enumerate(scene.sources), "Source '{name}' is outside the room and has no specular paths.")
        receivers = _inside(geometry, enumerate(scene.receivers), "Receiver '{name}' is outside the room and has no specular paths.")
        if not sources or not receivers:
            return cls()

        _warn_if_the_search_is_slow(geometry, max_order)
        room = geometry.to_pra(Defaults().SAMPLING_RATE, max_order, ray_tracing=False, air_absorption=False)
        for _, source in sources:
            room.add_source(np.asarray(source.position, dtype=float))
        room.add_microphone_array(np.array([receiver.position for _, receiver in receivers], dtype=float).T)
        room.image_source_model()

        walls = room.room_engine.walls
        tolerance = 1e-5 * max(float(np.ptp(geometry.vertices, axis=0).max()), 1.0)
        paths = []
        untraced = 0
        for s, (source_index, source) in enumerate(sources):
            pra_source = room.sources[s]
            images = np.asarray(pra_source.images, dtype=float).T
            orders = np.asarray(pra_source.orders, dtype=int).ravel()
            last_walls = np.asarray(pra_source.walls, dtype=int).ravel()
            start = np.asarray(source.position, dtype=float)
            for m, (receiver_index, receiver) in enumerate(receivers):
                end = np.asarray(receiver.position, dtype=float)
                kept = []
                failed = []
                for i in np.flatnonzero(np.asarray(room.visibility[s][m], dtype=bool)):
                    # the first of each group at one position, as hybrid.early._first_of_each_position
                    if _near(kept, images[i], tolerance):
                        continue
                    traced = _trace_back(walls, end, images[i], int(last_walls[i]), int(orders[i]), start, tolerance)
                    if traced is None:
                        if not _near(failed, images[i], tolerance):
                            failed.append(images[i])
                        continue
                    kept.append(images[i])
                    hits, path_walls = traced
                    paths.append(SpecularPath(
                        [start, *hits, end],
                        order=int(orders[i]),
                        walls=path_walls,
                        image=images[i],
                        receiver_index=receiver_index,
                        source=source,
                        source_index=source_index,
                        ray_index=len(paths),
                    ))
                # another image of the group may have traced a position whose first image did not
                untraced += sum(1 for image in failed if not _near(kept, image, tolerance))
        if untraced:
            warnings.warn(f"{untraced} visible image sources could not be traced back to their source.", UserWarning, stacklevel=2)
        return cls(paths)

    @property
    def orders(self) -> List[int]:
        """The sorted distinct reflection orders present."""
        return sorted({path.order for path in self.paths})

    def filter(
        self,
        source=None,
        ray_index: Optional[Union[int, List[int]]] = None,
        receiver_index: Optional[int] = None,
        order: Optional[Union[int, List[int]]] = None,
    ) -> "SpecularReflections":
        """Return a new :class:`SpecularReflections` keeping only the paths matching the filters.

        Parameters
        ----------
        source : :class:`compas_acoustics.Source` or int, optional
            Keep only paths from this source (matched by object or scene index).
        ray_index : int or list of int, optional
            Keep only paths with these ray indices.
        receiver_index : int, optional
            Keep only paths to this receiver.
        order : int or list of int, optional
            Keep only paths of exactly these reflection orders.

        Returns
        -------
        :class:`SpecularReflections`
        """
        orders = None
        if order is not None:
            orders = {order} if isinstance(order, numbers.Integral) else set(order)
        kept = [
            path
            for path in super().filter(source=source, ray_index=ray_index)
            if (receiver_index is None or path.receiver_index == receiver_index)
            and (orders is None or path.order in orders)
        ]
        return SpecularReflections(kept, name=self.name)


def _near(images: list, image, tolerance: float) -> bool:
    """Whether ``image`` is within ``tolerance`` of any of ``images``."""
    return bool(images) and float(np.min(np.linalg.norm(np.asarray(images) - image, axis=1))) <= tolerance


def _crossing(wall, start, end) -> Optional[np.ndarray]:
    """Where the segment ``start`` to ``end`` crosses ``wall``, by libroom's polygon test, or None."""
    out = np.zeros(3, dtype=np.float32)
    if wall.intersection(start.astype(np.float32), end.astype(np.float32), out) < 0:
        return None
    normal = np.asarray(wall.normal, dtype=float)
    direction = end - start
    t = normal @ (np.asarray(wall.origin, dtype=float) - start) / (normal @ direction)
    return start + t * direction


def _mirror(wall, point) -> np.ndarray:
    normal = np.asarray(wall.normal, dtype=float)
    normal = normal / np.linalg.norm(normal)
    return point + 2.0 * (normal @ (np.asarray(wall.origin, dtype=float) - point)) * normal


def _trace_back(walls, receiver, image, last_wall: int, order: int, source, tolerance: float):
    """The hit points and walls, source side first, of the path from ``source`` to ``receiver`` via ``image``.

    ``image`` is a visible image of ``order`` generated by ``last_wall``. Its parent is its mirror image
    in that wall, and the parent's own generating wall is one the segment from the hit point to the parent
    crosses, with the parent behind it; at an edge more than one does, so each is tried until the mirror
    images lead back to the source. None if none does.
    """
    if order == 0:
        return ([], []) if np.linalg.norm(image - source) <= tolerance else None

    def back(point, image, wall_index, order):
        hit = _crossing(walls[wall_index], point, image)
        if hit is None:
            return None
        parent = _mirror(walls[wall_index], image)
        if order == 1:
            return ([hit], [wall_index]) if np.linalg.norm(parent - source) <= tolerance else None
        for candidate in range(len(walls)):
            if candidate == wall_index or walls[candidate].side(parent.astype(np.float32)) <= 0:
                continue
            found = back(hit, parent, candidate, order - 1)
            if found is not None:
                return found[0] + [hit], found[1] + [wall_index]
        return None

    return back(receiver, image, last_wall, order)
