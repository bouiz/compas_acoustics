from compas.geometry import Point
from compas.geometry import Vector
import math

__all__ = ['Receiver']

class Receiver(Point):
    """
    Receiver class representing a receiver in an acoustic simulation.
    
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
    """

    def __init__(self, name: str, x:float, y:float, z:float, direction: Vector = None, radius=0.5):
        super().__init__(x, y, z)
        self.name=name
        self.direction = direction if direction is not None else Vector(0, 0, 1)
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
        if isinstance(direction, (list, tuple)) and len(direction) == 3:
            self._direction = Vector(*direction)
        elif isinstance(direction, Vector):
            self._direction = direction
        else:
            raise ValueError("Direction must be a compas.geometry.Vector object or a list/tuple of three float values.")
    
    
    def __str__(self):
        return f"Receiver(name={self.name}, position={self.position})"

    @property
    def total_spl(self):
        if not self.received_spls:
            return 0.0

        return 10.0 * math.log10(sum(10 ** (spl / 10) for spl in self.received_spls))  # NOTE: Validate code