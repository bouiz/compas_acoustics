import numbers
import warnings
from typing import List
from typing import Optional
from typing import Union

import numpy as np
from compas.data import Data
from compas.geometry import Point
from compas.geometry import Polyline

__all__ = ['SpecularPath', 'SpecularReflections']

DEFAULT_SOUND_SPEED = 343.98


class SpecularPath(Polyline):
    """A specular (image-source) reflection path from a source to a receiver, as a polyline.

    The vertices are ``[source_position, hit_1, ..., hit_order, receiver_position]``.

    Parameters
    ----------
    points : list
        The polyline vertices, from the source to the receiver.
    order : int
        The reflection order, the number of walls the path is reflected by.
    walls : list of int
        The wall index of each reflection, in path order.
    image : :class:`compas.geometry.Point` or list
        The image source of the path.
    source_index : int
        The index of the source in the scene.
    receiver_index : int
        The index of the receiver in the scene.
    sound_speed : float
        The speed of sound in m/s used to convert the path length to travel time.
    name : str, optional
        The name of the path.
    """

    def __init__(
        self,
        points,
        order: int,
        walls: List[int],
        image,
        source_index: int,
        receiver_index: int,
        sound_speed: float,
        name: Optional[str] = None,
    ):
        super().__init__(points, name=name)
        self.order = order
        self.walls = list(walls)
        self.image = Point(*image)
        self.source_index = source_index
        self.receiver_index = receiver_index
        self.sound_speed = sound_speed

    @property
    def __data__(self):
        data = super().__data__
        data.update(
            order=self.order,
            walls=self.walls,
            image=list(self.image),
            source_index=self.source_index,
            receiver_index=self.receiver_index,
            sound_speed=self.sound_speed,
        )
        return data

    @classmethod
    def __from_data__(cls, data):
        return cls(**data)

    @property
    def travel_time(self) -> float:
        """The travel time of the path in milliseconds."""
        return self.length / self.sound_speed * 1000.0


class SpecularReflections(Data):
    """A collection of specular reflection paths between sources and receivers.

    Parameters
    ----------
    paths : list of :class:`SpecularPath`, optional
        The reflection paths.
    sound_speed : float, optional
        The speed of sound in m/s.
    name : str, optional
        The name of the collection.
    """

    def __init__(self, paths: Optional[List[SpecularPath]] = None, sound_speed: float = DEFAULT_SOUND_SPEED, name: Optional[str] = None):
        super().__init__(name=name)
        self.paths = list(paths) if paths else []
        self.sound_speed = sound_speed

    @property
    def __data__(self):
        return {"paths": self.paths, "sound_speed": self.sound_speed}

    @classmethod
    def __from_data__(cls, data):
        return cls(data["paths"], sound_speed=data["sound_speed"])

    def __len__(self) -> int:
        return len(self.paths)

    def __iter__(self):
        return iter(self.paths)

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
            The highest reflection order.

        Returns
        -------
        :class:`SpecularReflections`
        """
        from compas_acoustics.air import AirConditions
        from compas_acoustics.defaults import Defaults
        from compas_acoustics.room_geometry import RoomGeometry

        sound_speed = AirConditions().speed_of_sound
        geometry = RoomGeometry.from_scene(scene)
        sources = _inside(geometry, scene.sources, "Source")
        receivers = _inside(geometry, scene.receivers, "Receiver")
        if not sources or not receivers:
            return cls(sound_speed=sound_speed)

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
                for i in np.flatnonzero(np.asarray(room.visibility[s][m], dtype=bool)):
                    # the first of each group at one position, as hybrid.early._first_of_each_position
                    if kept and np.min(np.linalg.norm(np.asarray(kept) - images[i], axis=1)) <= tolerance:
                        continue
                    traced = _trace_back(walls, end, images[i], int(last_walls[i]), int(orders[i]), start, tolerance)
                    if traced is None:
                        untraced += 1
                        continue
                    kept.append(images[i])
                    hits, path_walls = traced
                    paths.append(SpecularPath(
                        [start, *hits, end],
                        order=int(orders[i]),
                        walls=path_walls,
                        image=images[i],
                        source_index=source_index,
                        receiver_index=receiver_index,
                        sound_speed=sound_speed,
                    ))
        if untraced:
            warnings.warn(f"{untraced} visible image sources could not be traced back to their source.", UserWarning, stacklevel=2)
        return cls(paths, sound_speed=sound_speed)

    def display_in_viewer(self, scene=None, order=None, color=None, linewidth: float = 1.0, show: bool = True):
        """Display the paths (and optionally the scene) in a compas viewer.

        Parameters
        ----------
        scene : :class:`compas_acoustics.Scene`, optional
            If provided, the acoustic geometry, sources and receivers are drawn as context.
        order : int or list of int, optional
            Only display the paths of these reflection orders.
        color : :class:`compas.colors.Color`, optional
            The colour of the paths.
        linewidth : float, optional
            The width of the drawn path lines.
        show : bool, optional
            Whether to open the viewer. If ``False`` the configured viewer is returned without showing.

        Returns
        -------
        The configured viewer instance.
        """
        import compas

        if compas.is_rhino():
            raise RuntimeError("Displaying specular reflections in the viewer is not supported in Rhino.")

        from compas.colors import Color
        from compas_viewer import Viewer

        from compas_acoustics.ray import RayField

        if color is None:
            color = Color.blue()

        viewer = Viewer()
        if scene is not None:
            RayField._add_scene_to_viewer(viewer, scene)

        for index, path in enumerate(self.filter(order=order)):
            name = f"specular_{path.source_index}_{path.receiver_index}_{path.order}_{index}"
            viewer.scene.add(path, name=name, linecolor=color, linewidth=linewidth)

        if show:
            viewer.show()
        return viewer

    @property
    def orders(self) -> List[int]:
        """The sorted distinct reflection orders present."""
        return sorted({path.order for path in self.paths})

    def filter(
        self,
        source_index: Optional[int] = None,
        receiver_index: Optional[int] = None,
        order: Optional[Union[int, List[int]]] = None,
    ) -> "SpecularReflections":
        """Return a new :class:`SpecularReflections` keeping only the paths matching the filters.

        Parameters
        ----------
        source_index : int, optional
            Keep only paths from this source.
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
            for path in self.paths
            if (source_index is None or path.source_index == source_index)
            and (receiver_index is None or path.receiver_index == receiver_index)
            and (orders is None or path.order in orders)
        ]
        return SpecularReflections(kept, sound_speed=self.sound_speed, name=self.name)


def _inside(geometry, items, kind: str) -> list:
    """``(scene index, item)`` of the items inside the room, warning for the others."""
    inside = []
    for index, item in enumerate(items):
        if geometry.is_inside(item.position):
            inside.append((index, item))
        else:
            warnings.warn(f"{kind} '{item.name}' is outside the room and has no specular paths.", UserWarning, stacklevel=3)
    return inside


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
