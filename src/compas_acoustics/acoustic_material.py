from compas.data import Data
from compas_acoustics.defaults import Defaults

__all__ = ['AcousticMaterial']


class AcousticMaterial(Data):
    """Class representing an acoustic material with its properties.
    
    Parameters
    ----------
    name : str
        The name of the acoustic material. Lowercase and no spaces.
    description : str
        A text describing the material.
    absorption_coefficients : list
        A list of absorption coefficients corresponding to frequency bands.
    scattering_coefficients : list
        A list of scattering coefficients corresponding to frequency bands. Optional.
    transmission_coefficients : list
        A list of transmission coefficients corresponding to frequency bands. Optional.
    """
    
    def __init__(self, name: str, absorption_coefficients: list, scattering_coefficients: list = None, transmission_coefficients: list = None, description: str = None):
        super().__init__()
        self.name = name
        self.description = description
        self.absorption_coefficients = absorption_coefficients
        self.scattering_coefficients = scattering_coefficients
        self.transmission_coefficients = transmission_coefficients
        
    
    @property
    def name(self):
        return self._name
    
    
    @name.setter
    def name(self, name):
        if " " in name:
            raise ValueError("Material name cannot contain spaces.")
        if any(char.isupper() for char in name):
            raise ValueError("Material name must be lowercase.")
        self._name = name
        
        
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
        if coefficients:
            self._absorption_coefficients = self._sanitize_coefficients(coefficients, "Absorption")
        else:
            raise ValueError("You need to provide absorption coefficients.")
        
    
    @property
    def scattering_coefficients(self):
        return self._scattering_coefficients
    
    
    @scattering_coefficients.setter
    def scattering_coefficients(self, coefficients):
        if coefficients:
            self._scattering_coefficients = self._sanitize_coefficients(coefficients, "Scattering")
        else:
            self._scattering_coefficients = None
    
    
    @property
    def transmission_coefficients(self):
        return self._transmission_coefficients
    
    
    @transmission_coefficients.setter
    def transmission_coefficients(self, coefficients: list):
        if coefficients:
            self._transmission_coefficients = self._sanitize_coefficients(coefficients, "Transmission")
        else:
            self._transmission_coefficients = None
            
            
    def _sanitize_coefficients(self, coefficients, coeff_type:str):
        if isinstance(coefficients, int) or isinstance(coefficients, float):
                coefficients = [coefficients] * Defaults().NUMBER_OF_FREQUENCY_BANDS
        elif isinstance(coefficients, (list, tuple)):
            if len(coefficients) != Defaults().NUMBER_OF_FREQUENCY_BANDS:
                raise ValueError(f"{coeff_type} coefficients list must have {Defaults().NUMBER_OF_FREQUENCY_BANDS} values corresponding to the frequency bands.")
            else:
                coefficients = list(coefficients)
        return coefficients


    @property
    def __data__(self):
        return{
            "name": self.name,
            "description": self.description or "",
            "absorption_coefficients": self.absorption_coefficients,
            "scattering_coefficients": self.scattering_coefficients or [],
            "transmission_coefficients": self.transmission_coefficients or [],
        }
        
  
    def __str__(self):
        return f"AcousticMaterial(name={self.name}, absorption_coefficients={self.absorption_coefficients})"
    