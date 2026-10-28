"""Acoustic materials: absorption, scattering and transmission per band, and the library of them."""

import math
import numbers
import os
import re
import warnings
from enum import Enum

import numpy as np
from compas.data import Data
from compas.data import json_load

from compas_acoustics.defaults import Defaults
from compas_acoustics.defaults import _PackageDtype

__all__ = ['AcousticMaterial', 'MaterialCategory', 'MaterialsLibrary']


class MaterialCategory(str, Enum):
    """Broad groups of acoustic materials, as the library is organised."""

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


class AcousticMaterial(_PackageDtype, Data):
    """Absorption, scattering and transmission coefficients per octave band of :attr:`Defaults.DEFAULT_FREQUENCY_BANDS`.

    Parameters
    ----------
    name : str
    absorption_coefficients : float or list of float
        One per band, or one for every band.
    scattering_coefficients : float or list of float, optional
    transmission_coefficients : float or list of float, optional
    description : str, optional
    category : :class:`MaterialCategory` or str, optional
        Default ``MaterialCategory.OTHER``.
    data_origin : str, optional
        Default ``"custom"``.
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
        """The name as the library's file name."""
        return re.sub(r"[^a-z0-9.]+", "_", self.name.lower()).strip("_")

    @property
    def category(self):
        return self._category

    @category.setter
    def category(self, category):
        self._category = MaterialCategory.parse(category)

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
        # The coefficients are validated against these bands and the JSON form does not store them,
        # so any other grid would be silently misread
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
        """Set the scattering coefficients from a surface depth: ``s = s_max (1 - exp(-(k d)^2))`` per band.

        After Dalenbaeck (1995), Room acoustic prediction and auralization - how close can we get to
        reality?, ICA Trondheim; and Paetynen and Lokki (2010), Acta Acust. united Ac. 96(1).

        Parameters
        ----------
        depth : float
            The surface's characteristic depth, in metres.
        s_max : float, optional
        c : float, optional
            Speed of sound in m/s; default that of ``AirConditions()``.

        Returns
        -------
        list of float
        """
        if depth > 2:
            warnings.warn(f"Depth value should be in meters! Calculating scattering coefficients for depth {depth} meters.")
        self.scattering_coefficients = [round(s_max * (1 - math.exp(-(2 * math.pi * f / c * depth) ** 2)), 3)
                                        for f in self.frequency_bands]
        return self.scattering_coefficients

    def _sanitize_coefficients(self, coefficients, coeff_type:str):
        """One float per frequency band; a single value applies to every band; None for None or an empty list."""
        if coefficients is None:
            return None
        count = Defaults().NUMBER_OF_FREQUENCY_BANDS
        if isinstance(coefficients, (str, numbers.Real)):
            values = [float(coefficients)] * count
        else:
            # any iterable: Grasshopper passes a .NET List[Double]
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
        """Write the material to the library folder."""
        #TODO check if material with same name exists
        #TODO check if material with same properties exists
        path = os.path.join(Defaults().MATERIALS_LIBRARY_PATH, f"{self.slug}.json")
        self.to_json(path, compact=True, minimal=True)

    @property
    def __data__(self):
        return {
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


class MaterialsLibrary:
    """The materials of a folder of JSON files, one per material, looked up by name."""

    def __init__(self, library_path: str = None):
        self._library_path = library_path or Defaults().MATERIALS_LIBRARY_PATH
        self._index = None
        self._categories = None

    @property
    def index(self) -> dict:
        """Material names to JSON file paths, read on first access."""
        if self._index is None:
            self._index = {}
            self._categories = {}
            for filename in sorted(os.listdir(self._library_path)):
                if not filename.endswith('.json'):
                    continue
                path = os.path.join(self._library_path, filename)
                material = self.material_from_json(path)
                self._index[material.name] = path
                self._categories[material.name] = material.category
        return self._index

    def material_from_name(self, material_name: str) -> AcousticMaterial:
        return self.material_from_json(self.path_from_material_name(material_name))

    def material_from_json(self, path: str) -> AcousticMaterial:
        return json_load(path)

    def path_from_material_name(self, material_name: str) -> str:
        """The JSON file of a material, or where it would be stored when not in the library."""
        if material_name in self.index:
            return self.index[material_name]
        return os.path.join(self._library_path, f"{material_name}.json")

    @property
    def material_names(self) -> list:
        """list : The names of the library's materials."""
        return list(self.index.keys())

    @property
    def materials_by_category(self) -> dict:
        """dict : Sorted material names per :class:`MaterialCategory` that has any, in the categories' order."""
        groups = {}
        for name in self.index:
            groups.setdefault(self._categories[name], []).append(name)
        return {category: sorted(groups[category]) for category in MaterialCategory if category in groups}
