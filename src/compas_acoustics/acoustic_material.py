import math
import numbers
import os
import re
import warnings
from enum import Enum

import numpy as np
from compas.data import Data

from compas_acoustics.defaults import Defaults

__all__ = ['AcousticMaterial', 'MaterialCategory']


class MaterialCategory(str, Enum):
    """Broad groups used to organise acoustic materials, e.g. in the materials library."""

    CONCRETE = "Concrete"
    MASONRY = "Masonry"
    STONE_AND_TILES = "Stone & Tiles"
    PLASTER_AND_GYPSUM = "Plaster & Gypsum"
    WOOD = "Wood"
    GLASS = "Glass"
    METAL = "Metal"
    FLOOR_COVERINGS = "Floor Coverings"
    TEXTILES_AND_CURTAINS = "Textiles & Curtains"
    SEATING = "Seating"
    AUDIENCE = "Audience"
    ABSORBERS = "Absorbers"
    OTHER = "Other"

    @classmethod
    def parse(cls, value):
        """Get a category from a member, its value (``"Stone & Tiles"``) or its name (``"STONE_AND_TILES"``)."""
        if value is None:
            return cls.OTHER
        if isinstance(value, cls):
            return value
        for member in cls:
            if value in (member.value, member.name):
                return member
        raise ValueError(f"Unknown material category '{value}'. Valid categories: {', '.join(m.value for m in cls)}.")


class AcousticMaterial(Data):
    """Class representing an acoustic material with its properties.
    
    Parameters
    ----------
    name : str
        The name of the acoustic material.
    description : str
        A text describing the material.
    category : :class:`MaterialCategory` | str, optional
        The broad group the material belongs to. Defaults to ``MaterialCategory.OTHER``.
    absorption_coefficients : list
        A list of absorption coefficients corresponding to frequency bands, or a single value for every band.
    scattering_coefficients : list
        A list of scattering coefficients corresponding to frequency bands, or a single value for every band. Optional.
    transmission_coefficients : list
        A list of transmission coefficients corresponding to frequency bands, or a single value for every band. Optional.
    """
    
    def __init__(self, name: str, absorption_coefficients, scattering_coefficients: list=None, transmission_coefficients: list=None, description: str=None, category=None, data_origin: str=None): # noqa: E501
        super().__init__()
        self.name = name
        self.description = description
        self.category = category
        self.absorption_coefficients = absorption_coefficients
        self.scattering_coefficients = scattering_coefficients
        self.transmission_coefficients = transmission_coefficients
        self.data_origin = data_origin or "custom"
        self.frequency_bands = Defaults().DEFAULT_FREQUENCY_BANDS
        
    
    @property
    def name(self):
        return self._name
    
    
    @name.setter
    def name(self, name):
        if not name or not name.strip():
            raise ValueError("Material name cannot be empty.")
        self._name = name.strip()

    @property
    def slug(self):
        """File-system friendly version of the name, used as file name in the materials library."""
        return re.sub(r"[^a-z0-9.]+", "_", self.name.lower()).strip("_")

    @property
    def category(self):
        return self._category

    @category.setter
    def category(self, category):
        self._category = MaterialCategory.parse(category)

        
    @property
    def description(self):
        return self._description
    
    
    @description.setter
    def description(self, description):
        self._description = description

    
    @property
    def absorption_coefficients(self):
        return self._absorption_coefficients
    
    
    @absorption_coefficients.setter
    def absorption_coefficients(self, coefficients):
        coefficients = self._sanitize_coefficients(coefficients, "Absorption")
        if coefficients is None:
            raise ValueError("You need to provide absorption coefficients.")
        self._absorption_coefficients = coefficients
        
    
    @property
    def scattering_coefficients(self):
        return self._scattering_coefficients
    
    
    @scattering_coefficients.setter
    def scattering_coefficients(self, coefficients):
        self._scattering_coefficients = self._sanitize_coefficients(coefficients, "Scattering")
    
    
    @property
    def transmission_coefficients(self):
        return self._transmission_coefficients
    
    
    @transmission_coefficients.setter
    def transmission_coefficients(self, coefficients):
        self._transmission_coefficients = self._sanitize_coefficients(coefficients, "Transmission")
            
    
    @property
    def frequency_bands(self):
        return self._frequency_bands
    
    
    @frequency_bands.setter
    def frequency_bands(self, frequency_bands):
        # Coefficients are validated against these bands (_sanitize_coefficients),
        # and the JSON form does not store them, so any other grid would be silently
        # misread.
        bands = Defaults().DEFAULT_FREQUENCY_BANDS
        if frequency_bands and [float(f) for f in frequency_bands] != [float(f) for f in bands]:
            raise ValueError(f"Material frequency bands must be {bands} Hz, got {list(frequency_bands)}.")
        self._frequency_bands = bands
            
    
    @property
    def data_origin(self):
        return self._data_origin
    
    
    @data_origin.setter
    def data_origin(self, origin):
        if origin:
            self._data_origin = origin
        else:
            self._data_origin = "user"
    
        
    def scattering_coefficients_from_depth(self, depth, s_max=0.8, c = Defaults().DEFAULT_SPEED_OF_SOUND):
        """
        Compute scattering coefficients for octave bands
        based on surface depth using a wavelength-dependent model.
        
        Based on research from:
        1. Dalenbäck, B.-I. L. (1995). Room acoustic prediction and auralization—How close can we get to reality?
        Proceedings of the 15th International Congress on Acoustics (ICA), Trondheim, Norway.
        2. Pätynen, J., & Lokki, T. (2010). Directivities of symphony orchestra instruments and their effect on recorded spatial sound.
        Acta Acustica united with Acustica, 96(1).
        
        Parameters
        ----------
        depth : float
            Characteristic surface depth or roughness (in meters).
        s_max : float, optional
            Maximum scattering coefficient (default = 0.8).
        c : float, optional
            Speed of sound in air (m/s), default = 343.

        Returns
        -------
        list of float
            List of scattering coefficients.
        """
        if depth > 2:
            warnings.warn(f"Depth value should be in meters! Calculating scattering coefficients for depth {depth} meters.")
        
        freq_bands = self.frequency_bands
        
        coeffs = []
        for f in freq_bands:
            k = 2 * math.pi * f / c
            s = s_max * (1 - math.exp(-(k * depth)**2))
            coeffs.append((round(s, 3)))
            
        self.scattering_coefficients = coeffs

        return self.scattering_coefficients   
            
            
    def _sanitize_coefficients(self, coefficients, coeff_type:str):
        """One float per frequency band; a single value applies to every band; None for None or an empty list."""
        if coefficients is None:
            return None
        count = Defaults().NUMBER_OF_FREQUENCY_BANDS
        if isinstance(coefficients, (str, numbers.Real)):
            values = [float(coefficients)] * count
        else:
            # Any iterable, not only list/tuple: Grasshopper passes System.Collections.Generic.List[Double]
            values = np.asarray(list(coefficients), dtype=float).reshape(-1).tolist()
            if not values:
                return None
            if len(values) == 1:
                values = values * count
            if len(values) != count:
                raise ValueError(f"{coeff_type} coefficients list must have {count} values corresponding to the frequency bands.")
        if not all(0.0 <= v <= 1.0 for v in values):
            raise ValueError(f"{coeff_type} coefficients must lie in [0, 1], got {values}.")
        return values


    def add_to_library(self):
        """
        Add the AcousticMaterial to the library.
        """
        #TODO check if material with same name exists
        #TODO check if material with same properties exists
        path = os.path.join(Defaults().MATERIALS_LIBRARY_PATH, f"{self.slug}.json")
        self.to_json(path, compact=True, minimal=True)


    @property
    def __data__(self):
        return{
            "name": self.name,
            "description": self.description or "",
            "category": self.category.value,
            "absorption_coefficients": self.absorption_coefficients,
            "scattering_coefficients": self.scattering_coefficients or [],
            "transmission_coefficients": self.transmission_coefficients or [],
            "data_origin": self.data_origin,
        }
        
  
    def __str__(self):
        return f"AcousticMaterial(name={self.name}, category={self.category.value}, data_origin={self.data_origin}, absorption_coefficients={self.absorption_coefficients}, scattering_coefficients={self.scattering_coefficients}, transmission_coefficients={self.transmission_coefficients})" # noqa: E501
    