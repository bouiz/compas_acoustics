import os

from compas.data import Data
from compas.data import json_load

from compas_acoustics.acoustic_material import AcousticMaterial
from compas_acoustics.acoustic_material import MaterialCategory
from compas_acoustics.defaults import Defaults

__all__ = ['MaterialsLibrary']


class MaterialsLibrary(Data):
    """
    A class for handling acoustic materials.

    Materials are stored as one JSON file per material in the library folder.
    They are looked up by their name, the file name is just a file-system friendly version of it.
    """


    def __init__(self, library_path: str = None):
        self._library_path = library_path or Defaults().MATERIALS_LIBRARY_PATH
        self._index = None
        self._categories = None


    @property
    def index(self) -> dict:
        """Mapping of material names to JSON file paths, built on first access."""
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
        """Retrieve an AcousticMaterial instance by its name.

        Parameters
        ----------
        material_name : str
            The identifier of the acoustic material.

        Returns
        -------
        AcousticMaterial
            An instance of AcousticMaterial.
        """
        path = self.path_from_material_name(material_name)
        return self.material_from_json(path)


    def material_from_json(self, path: str) -> AcousticMaterial:
        """Create an AcousticMaterial instance from JSON data.

        Parameters
        ----------
        path : str
            The path to the JSON file containing material properties.

        Returns
        -------
        AcousticMaterial
            An instance of AcousticMaterial.
        """
        material = json_load(path)
        return material


    def path_from_material_name(self, material_name: str) -> str:
        """Get the path to the JSON file of an AcousticMaterial by its name.

        Parameters
        ----------
        material_name : str
            The identifier of the acoustic material.

        Returns
        -------
        str
            The path to the JSON file for the specified material.
        """
        if material_name in self.index:
            return self.index[material_name]
        # not in the library (yet): where it would be stored
        return os.path.join(self._library_path, f"{material_name}.json")


    @property
    def material_names(self) -> list:
        """List of all material names in the library.

        Returns
        -------
        list
            A list of material names.
        """
        return list(self.index.keys())


    @property
    def materials_by_category(self) -> dict:
        """Material names in the library grouped by category.

        Returns
        -------
        dict
            A dictionary mapping each :class:`MaterialCategory` that has materials to a sorted list of material names,
            in the order the categories are defined.
        """
        groups = {}
        for name in self.index:
            groups.setdefault(self._categories[name], []).append(name)
        return {category: sorted(groups[category]) for category in MaterialCategory if category in groups}
