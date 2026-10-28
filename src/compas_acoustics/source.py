import numbers

import numpy as np
from compas.geometry import Point
from compas.geometry import Vector

from compas_acoustics.defaults import Defaults
from compas_acoustics.defaults import _PackageDtype

__all__ = ['Source']


class Source(_PackageDtype, Point):
    """A sound source: position, directivity and facing, an optional sound file, and its level.

    ``spl`` is the free-field level on its axis at 1 m per octave band, 125 Hz to 8 kHz, in dB re 20 uPa (the
    voice presets give the talker levels of ANSI S3.5-1997). On the package scale the direct sound is
    ``1 / (4 pi r)`` with gain 1 on axis, so ``Lp = spl + G - 20 dB`` (:meth:`Analyser.sound_pressure_level`).
    """

    def __init__(self, name: str, x: float, y: float, z: float, directivity: str = None, direction: Vector = None, sound_file: str = None):
        super().__init__(x, y, z)
        self.name = name
        self.description = None
        self.directivity = directivity if directivity is not None else "omnidirectional"
        self.direction = direction if direction is not None else Vector(1, 0, 0)
        self.spl = 120.0
        self.sound_file = sound_file

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
    def direction(self):
        return self._direction

    @direction.setter
    def direction(self, direction):
        if isinstance(direction, (list, tuple)) and len(direction) == 3:
            self._direction = Vector(*direction)
        elif isinstance(direction, Vector):
            self._direction = direction
        else:
            raise ValueError("Direction must be a compas.geometry.Vector object or a list/tuple of three float values.")

    @property
    def spl(self):
        return self._spl

    @spl.setter
    def spl(self, spl):
        count = Defaults().NUMBER_OF_FREQUENCY_BANDS
        # any iterable: Grasshopper passes a .NET List[Double]
        values = [float(spl)] if isinstance(spl, numbers.Real) else [float(v) for v in np.asarray(list(spl), dtype=float).reshape(-1)]
        if len(values) == 1:
            values = values * count
        if len(values) != count:
            raise ValueError(f"SPL must be one value or a list of {count} float values corresponding to frequency bands.")
        self._spl = values

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
    def sound_file(self):
        return self._sound_file

    @sound_file.setter
    def sound_file(self, file_path: str):
        if file_path is None:
            self._sound_file = None
            return

        from pathlib import Path

        if not Path(file_path).exists():
            raise ValueError(f"Sound file {file_path} does not exist.")

        self._sound_file = str(file_path)

    @property
    def id(self):
        return self.guid

    @classmethod
    def omni(cls, name: str, x:float, y:float, z:float):
        source = cls(name, x, y, z)
        source.description = "omni"
        return source

    @classmethod
    def _voice(cls, name, x, y, z, effort, levels):
        # levels at 62.5 Hz to 8 kHz (ANSI S3.5-1997)
        source = cls(name, x, y, z)
        source.description = f"voice {effort}"
        source.directional = True
        by_band = dict(zip([62.5, 125, 250, 500, 1000, 2000, 4000, 8000], levels))
        source.spl = [by_band[f] for f in Defaults().DEFAULT_FREQUENCY_BANDS]
        return source

    @classmethod
    def voice_normal(cls, name: str, x:float, y:float, z:float):
        return cls._voice(name, x, y, z, "normal", [45.0, 55.0, 65.0, 69.0, 63.0, 55.8, 50.0, 44.0])

    @classmethod
    def voice_raised(cls, name: str, x:float, y:float, z:float):
        return cls._voice(name, x, y, z, "raised", [48.0, 60.0, 70.0, 75.0, 72.0, 64.0, 57.0, 49.0])

    @classmethod
    def voice_loud(cls, name: str, x:float, y:float, z:float):
        return cls._voice(name, x, y, z, "loud", [52.0, 64.0, 72.0, 80.0, 80.0, 73.0, 66.0, 55.0])

    @classmethod
    def voice_shouting(cls, name: str, x:float, y:float, z:float):
        return cls._voice(name, x, y, z, "shouting", [52.0, 65.0, 73.0, 84.0, 89.0, 82.0, 75.0, 64.0])

    @property
    def __data__(self):
        return {
            "name": self.name,
            "point": [float(v) for v in self.position],
            "directivity": self.directivity,
            "direction": [float(v) for v in self.direction],
            "sound_file": self.sound_file,
            "spl": list(self.spl),
            "description": self.description,
        }

    @classmethod
    def __from_data__(cls, data):
        source = cls(data["name"], *data["point"], directivity=data.get("directivity"), direction=data.get("direction"))
        # Set directly: the file may not exist on this machine
        source._sound_file = data.get("sound_file")
        if data.get("spl") is not None:
            source.spl = data["spl"]
        source.description = data.get("description")
        return source

    def transform(self, T):
        """Transform the source: its position and its direction."""
        super().transform(T)
        self.direction = self.direction.transformed(T)

    def __str__(self):
        return f"Source(name={self.name}, description={self.description}, position={self.position})"
