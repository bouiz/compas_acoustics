import numpy as np
from compas.geometry import Frame
from compas.geometry import Point
from compas.geometry import Vector
from compas.tolerance import TOL

from compas_acoustics.defaults import _PackageDtype

__all__ = ['Receiver']


class Receiver(_PackageDtype, Point):
    """A receiver with an orientation, the frame of its ambisonic responses: x is ``direction``, y its left, z ``up``.

    Parameters
    ----------
    name : str
        Lowercase, without spaces.
    x, y, z : float
    direction : :class:`compas.geometry.Vector`, optional
        Where it faces; default +x.
    radius : float, optional
        In metres.
    up : :class:`compas.geometry.Vector`, optional
        Default +z; only its component orthogonal to ``direction`` is used.
    """

    def __init__(self, name: str, x:float, y:float, z:float, direction: Vector = None, radius=0.5, up: Vector = None):
        super().__init__(x, y, z)
        self.name = name
        self.direction = direction if direction is not None else Vector(1, 0, 0)
        self.up = up if up is not None else Vector(0, 0, 1)
        self.radius = radius

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        if " " in name:
            raise ValueError("Receiver name cannot contain spaces.")
        if any(char.isupper() for char in name):
            raise ValueError("Receiver name must be lowercase.")
        self._name = name

    @property
    def id(self):
        return self.guid

    @property
    def position(self):
        return [self.x, self.y, self.z]

    @position.setter
    def position(self, position:list):
        if len(position) != 3:
            raise ValueError("Position must be a list of three float values representing x, y, z coordinates.")
        self.x = position[0]
        self.y = position[1]
        self.z = position[2]
        return self.position

    @property
    def direction(self):
        return self._direction

    @direction.setter
    def direction(self, direction):
        self._direction = _vector(direction, "Direction")

    @property
    def up(self):
        return self._up

    @up.setter
    def up(self, up):
        self._up = _vector(up, "Up")

    @property
    def rotation(self):
        """The orientation, a 3x3 proper rotation whose columns are the facing, left and up axes in world coordinates.

        Its transpose takes a world vector to the receiver frame.
        """
        facing = np.asarray(list(self.direction), dtype=float)
        length = np.linalg.norm(facing)
        if TOL.is_zero(length):
            raise ValueError("Receiver direction must be a non-zero vector.")
        facing = facing / length
        up = np.asarray(list(self.up), dtype=float)
        up = up - (up @ facing) * facing
        if TOL.is_zero(np.linalg.norm(up)):
            raise ValueError("Receiver direction is parallel to its up vector; the orientation is undefined.")
        up = up / np.linalg.norm(up)
        return np.column_stack([facing, np.cross(up, facing), up])

    @property
    def frame(self):
        """:class:`compas.geometry.Frame` : The position and the axes of :attr:`rotation`; setting it sets ``direction`` and ``up``."""
        rotation = self.rotation
        return Frame(self.position, Vector(*rotation[:, 0]), Vector(*rotation[:, 1]))

    @frame.setter
    def frame(self, frame):
        self.position = list(frame.point)
        self.direction = Vector(*frame.xaxis)
        self.up = Vector(*frame.zaxis)

    @classmethod
    def from_frame(cls, name: str, frame, radius=0.5):
        """A receiver at ``frame.point``, facing ``frame.xaxis`` with ``frame.zaxis`` up."""
        return cls(name, *frame.point, direction=Vector(*frame.xaxis), radius=radius, up=Vector(*frame.zaxis))

    def transform(self, T):
        """Transform the receiver: its position and its orientation."""
        super().transform(T)
        self.direction = self.direction.transformed(T)
        self.up = self.up.transformed(T)

    @property
    def __data__(self):
        return {
            "name": self.name,
            "point": [float(v) for v in self.position],
            "direction": [float(v) for v in self.direction],
            "up": [float(v) for v in self.up],
            "radius": self.radius,
        }

    @classmethod
    def __from_data__(cls, data):
        return cls(data["name"], *data["point"], direction=data.get("direction"), radius=data.get("radius", 0.5), up=data.get("up"))

    def __str__(self):
        return f"Receiver(name={self.name}, position={self.position})"


def _vector(value, label):
    """A non-zero :class:`Vector` from a Vector or three numbers."""
    if isinstance(value, Vector):
        vector = value
    elif isinstance(value, (list, tuple)) and len(value) == 3:
        vector = Vector(*value)
    else:
        raise ValueError(f"{label} must be a compas.geometry.Vector object or a list/tuple of three float values.")
    if TOL.is_zero(vector.length):
        raise ValueError(f"{label} must be a non-zero vector.")
    return vector
