import math

import numpy as np
from compas.geometry import Frame
from compas.geometry import Point
from compas.geometry import Vector
from compas.tolerance import TOL

__all__ = ['Receiver']

class Receiver(Point):
    """
    Receiver class representing a receiver in an acoustic simulation.

    A receiver has an orientation, ``direction`` (where it faces) and ``up``,
    which sets the frame of ambisonic responses: their x axis is ``direction``,
    y is to the receiver's left and z is ``up``. :attr:`frame` gives the same
    orientation as a compas Frame. Copies, JSON and transformations keep it.
    
    Parameters
    ----------
    name : str
        The name of the receiver. Must be lowercase and contain no spaces.
    x : float
        The x-coordinate of the receiver's position.
    y : float
        The y-coordinate of the receiver's position.
    z : float
        The z-coordinate of the receiver's position.
    direction : :class:`compas.geometry.Vector`, optional
        The direction the receiver faces. Defaults to +x.
    radius : float, optional
        The receiver radius in metres, used by the SPL estimates.
    up : :class:`compas.geometry.Vector`, optional
        The receiver's up direction. Defaults to +z. Only its component
        orthogonal to ``direction`` is used.
    """

    def __init__(self, name: str, x:float, y:float, z:float, direction: Vector = None, radius=0.5, up: Vector = None):
        super().__init__(x, y, z)
        self.name=name
        self.direction = direction if direction is not None else Vector(1, 0, 0)
        self.up = up if up is not None else Vector(0, 0, 1)
        self.radius = radius

        self.received_spls = []  # Placeholder after running trace rays simulation

    @property
    def name(self):
        return self._name
    
    @name.setter
    def name(self, name):
        if " " in name:
            raise ValueError("Source name cannot contain spaces.")
        if any(char.isupper() for char in name):
            raise ValueError("Source name must be lowercase.")
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
        """
        Set the direction of the receiver.
        
        Parameters
        ----------
        direction : list or Vector
            A list of three float values representing the direction vector or a compas.geometry.Vector object.
        """
        self._direction = _vector(direction, "Direction")


    @property
    def up(self):
        return self._up


    @up.setter
    def up(self, up):
        self._up = _vector(up, "Up")


    @property
    def rotation(self):
        """The receiver's orientation, a 3x3 proper rotation.

        Its columns are the receiver's own axes in world coordinates: facing,
        left, up. A world vector is expressed in the receiver frame by the
        transpose; the matrix itself would mirror the scene left to right.
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
        """:class:`compas.geometry.Frame` : The receiver's position and orientation.

        Its x axis is the facing direction, its y axis points to the receiver's
        left and its z axis is up: the axes of the receiver's ambisonic
        responses. Setting it sets the position, ``direction`` and ``up``.
        """
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
        # Vectors take the linear part of the transformation, not the translation
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

    @property
    def total_spl(self):
        if not self.received_spls:
            return 0.0

        return 10.0 * math.log10(sum(10 ** (spl / 10) for spl in self.received_spls))  # NOTE: Validate code


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
