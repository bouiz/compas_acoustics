from compas.geometry import Point
from compas.geometry import Vector

from compas_acoustics.defaults import Defaults

__all__ = ['Source']

# directivity = ["bidirectional", "hypercardioid", "supercardioid", "cardioid", "omnidirectional", "custom",]

class Source(Point):
    
    def __init__(self, name: str, x: float, y: float, z: float, directivity: str = None, direction: Vector = None, sound_file: str = None):
        super().__init__(x, y, z)
        self.name = name
        self.description = None
        self.directivity = directivity if directivity is not None else "omnidirectional"
        self.direction = direction if direction is not None else Vector(0, 0, 1)
        self.spl = 120.0 # Default SPL value for omni source TODO: swl?
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
    def description(self):
        return self._description
    
    
    @description.setter
    def description(self, description):
        self._description = description
        
        
    @property
    def directional(self):
        return self._directional
    
    
    @directional.setter
    def directional(self, directional: bool):
        self._directional = directional

    
    @property
    def direction(self):
        return self._direction


    @direction.setter
    def direction(self, direction):
        """
        Set the direction of the source.
        
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
    
    
    @property
    def spl(self):
        return self._spl
    
    
    @spl.setter
    def spl(self, spl):
        if isinstance(spl, int) or isinstance(spl, float):
            spl = [spl] * Defaults().NUMBER_OF_FREQUENCY_BANDS
        elif isinstance(spl, list) or isinstance(spl, tuple):
            if len(spl) != Defaults().NUMBER_OF_FREQUENCY_BANDS:
                raise ValueError(f"SPL must be a list of {Defaults().NUMBER_OF_FREQUENCY_BANDS} float values corresponding to frequency bands.")
            else:
                spl = list(spl)
        self._spl = spl

        
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
        """
        Set the sound file for the source. Automatically checks if the file 
        is stereo/multi-channel and downmixes it to 1D mono to prevent convolution errors.
        """
        if file_path is None:
            self._sound_file = None
            return

        from pathlib import Path

        import numpy as np
        import soundfile as sf

        path = Path(file_path)
        if not path.exists():
            raise ValueError(f"Sound file {file_path} does not exist.")

        data, sr = sf.read(str(path))

        if data.ndim > 1:
            data = np.mean(data, axis=1)

            mono_filename = f"mono_{path.name}"
            mono_path = path.parent / mono_filename
            sf.write(str(mono_path), data, sr)
            self._sound_file = str(mono_path)
        else:
            self._sound_file = str(path)
    
    @property
    def id(self):
        return self.guid
    
    
    #TODO
    #Point receiver to a source
    
    
    @classmethod
    def omni(cls, name: str, x:float, y:float, z:float):
        source = cls(name, x, y, z)
        source.description = "omni"
        return source
    
    
    # SPL values for a human speaker at 1m distance in dB after
    # ANSI 3.5-1997. Methods for Calculation of the Speech Intelligibility Index. American National Standard.
    
    @classmethod
    def voice_normal(cls, name: str, x:float, y:float, z:float):
        source = cls(name, x, y, z)
        source.description = "voice normal"
        source.directional = True
        source.spl = [45.0, 55.0, 65.0, 69.0, 63.0, 55.8, 50.0, 44.0]
        return source
    
    
    @classmethod
    def voice_raised(cls, name: str, x:float, y:float, z:float):
        source = cls(name, x, y, z)
        source.description = "voice raised"
        source.directional = True
        source.spl = [48.0, 60.0, 70.0, 75.0, 72.0, 64.0, 57.0, 49.0]
        return source
    
    
    @classmethod
    def voice_loud(cls, name: str, x:float, y:float, z:float):
        source = cls(name, x, y, z)
        source.description = "voice loud"
        source.directional = True
        source.spl = [52.0, 64.0, 72.0, 80.0, 80.0, 73.0, 66.0, 55.0]
        return source
    
    
    @classmethod
    def voice_shouting(cls, name: str, x:float, y:float, z:float):
        source = cls(name, x, y, z)
        source.description = "voice shouting"
        source.directional = True
        source.spl = [52.0, 65.0, 73.0, 84.0, 89.0, 82.0, 75.0, 64.0]
        return source
    
    
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
        # Already mono when it was stored, and it may not exist on this machine
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