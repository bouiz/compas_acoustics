from compas.data import Data
from compas.data import json_load

from compas_acoustics.defaults import Defaults
from compas_acoustics.acoustic_material import AcousticMaterial

import os


__all__ = ['MaterialsLibrary']


class MaterialsLibrary(Data):
    """
    A class for handling acoustic materials.
    """


    def __init__(self):
        self._library_path = Defaults().MATERIALS_LIBRARY_PATH
    
    
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
        """Construct the path to a JSON file for an AcousticMaterial by its name.
        
        Parameters
        ----------
        material_name : str
            The identifier of the acoustic material.
        
        Returns
        -------
        str
            The path to the JSON file for the specified material.
        """
        
        return os.path.join(self._library_path, f"{material_name}.json")


    @property
    def material_names(self) -> list:
        """List of all material names in the library.
        
        Returns
        -------
        list
            A list of material names.
        """
        material_files = [f for f in os.listdir(self._library_path) if f.endswith('.json')]
        material_names = [os.path.splitext(f)[0] for f in material_files]
        return material_names