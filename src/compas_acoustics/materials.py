"""Acoustic materials: absorption, scattering and transmission per band, and the library of them."""

import numbers
import os
import re
from enum import Enum

import numpy as np
from compas.data import Data
from compas.data import json_load

from compas_acoustics.defaults import Defaults
from compas_acoustics.defaults import _PackageDtype
from compas_acoustics.defaults import _per_band

__all__ = ["AcousticMaterial", "MaterialCategory", "MaterialsLibrary", "pyroomacoustics_material_names"]


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
    """Absorption, scattering and transmission coefficients per octave band, 125 Hz to 8 kHz.

    Parameters
    ----------
    name : str
    absorption_coefficients : float or list of float
        One for every band, or one per band; so are the other coefficients.
    scattering_coefficients : float or list of float, optional
    transmission_coefficients : float or list of float, optional
        Stored and serialised only: the simulation methods do not use them.
    description : str, optional
    category : :class:`MaterialCategory` or str, optional
        Default ``MaterialCategory.OTHER``.
    data_origin : str, optional
        Default ``"custom"``.
    """

    def __init__(
        self,
        name: str,
        absorption_coefficients,
        scattering_coefficients: list = None,
        transmission_coefficients: list = None,
        description: str = None,
        category=None,
        data_origin: str = None,
    ):
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

    @classmethod
    def from_pyroomacoustics(cls, name: str) -> "AcousticMaterial":
        """A material of pyroomacoustics' database (``pra.Material(name)``) on the package bands.

        Its absorption, and its scattering when the database has it, at 125 Hz to 8 kHz; a database entry stopping
        at 4 kHz holds its last value above, as pyroomacoustics does. ``data_origin`` says where it came from.
        """
        entry = _pra_material_table().get(name)
        if entry is None:
            raise ValueError(f"pyroomacoustics has no material named '{name}'")
        scattering = _on_package_bands(entry["scattering"]) if "scattering" in entry else None
        return cls(
            name=name,
            absorption_coefficients=_on_package_bands(entry["absorption"]),
            scattering_coefficients=scattering,
            description=entry["absorption"].get("description"),
            data_origin="pyroomacoustics",
        )

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
        # The coefficients are validated against these bands and the JSON form does not store them, so any other grid
        # would be silently misread
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

    def _sanitize_coefficients(self, coefficients, coeff_type: str):
        """One float per frequency band; a single value applies to every band; None for None or an empty list."""
        if coefficients is None:
            return None
        if isinstance(coefficients, str):
            coefficients = float(coefficients)
        elif not isinstance(coefficients, numbers.Real):
            coefficients = list(coefficients)
            if not np.size(coefficients):
                return None
        values = _per_band(coefficients, f"{coeff_type} coefficients")
        if not all(0.0 <= v <= 1.0 for v in values):
            raise ValueError(f"{coeff_type} coefficients must lie in [0, 1], got {values}.")
        return values

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
        return f"AcousticMaterial(name={self.name}, category={self.category.value}, data_origin={self.data_origin}, absorption_coefficients={self.absorption_coefficients}, scattering_coefficients={self.scattering_coefficients}, transmission_coefficients={self.transmission_coefficients})"  # noqa: E501


def _pra_material_table() -> dict:
    """pyroomacoustics' material database, flattened: name to its absorption (and scattering, when it has one) entry."""
    global _PRA_MATERIALS
    if _PRA_MATERIALS is None:
        import json

        import pyroomacoustics as pra

        path = os.path.join(os.path.dirname(pra.__file__), "data", "materials.json")
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        table = {}

        def walk(node, kind):
            for key, value in node.items():
                if isinstance(value, dict) and "coeffs" in value:
                    table.setdefault(key, {})[kind] = value
                elif isinstance(value, dict):
                    walk(value, kind)

        walk(data.get("absorption", {}), "absorption")
        walk(data.get("scattering", {}), "scattering")
        _PRA_MATERIALS = {k: v for k, v in table.items() if "absorption" in v}
    return _PRA_MATERIALS


_PRA_MATERIALS = None


def pyroomacoustics_material_names() -> list:
    """The names of pyroomacoustics' materials (``pra.Material(name)``), which
    :meth:`MaterialsLibrary.material_from_name` also finds."""
    return sorted(_pra_material_table())


def _on_package_bands(entry) -> list:
    """Coefficients at 125 Hz to 8 kHz from a pyroomacoustics entry: the matching centres, the last band held above."""
    centres = [float(f) for f in entry.get("center_freqs", [125, 250, 500, 1000, 2000, 4000, 8000][: len(entry["coeffs"])])]
    coeffs = [float(c) for c in entry["coeffs"]]
    out = []
    for band in Defaults().DEFAULT_FREQUENCY_BANDS:
        nearest = int(np.argmin([abs(np.log2(band / c)) for c in centres]))
        out.append(coeffs[nearest])
    return out


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
                if not filename.endswith(".json"):
                    continue
                path = os.path.join(self._library_path, filename)
                material = self.material_from_json(path)
                self._index[material.name] = path
                self._categories[material.name] = material.category
        return self._index

    def material_from_name(self, material_name: str) -> AcousticMaterial:
        """The library's material of that name, else pyroomacoustics'
        (:meth:`AcousticMaterial.from_pyroomacoustics`)."""
        if material_name not in self.index and _pra_material_table().get(material_name) is not None:
            return AcousticMaterial.from_pyroomacoustics(material_name)
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
