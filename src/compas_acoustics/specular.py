from typing import List
from typing import Optional
from typing import Union

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
            orders = {order} if isinstance(order, int) else set(order)

        kept = [
            path
            for path in self.paths
            if (source_index is None or path.source_index == source_index)
            and (receiver_index is None or path.receiver_index == receiver_index)
            and (orders is None or path.order in orders)
        ]
        return SpecularReflections(kept, sound_speed=self.sound_speed, name=self.name)
