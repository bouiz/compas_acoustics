from compas.data import Data
from compas_acoustics.defaults import Defaults

__all__ = ['AcousticMaterial']


class AcousticMaterial(Data):
    """Class representing an acoustic material with its properties.
    
    Parameters
    ----------
    id : str
        The identifier of the acoustic material. Lowercase and no spaces.
    description : str
        A text describing the material.
    absorption_coefficients : list
        A list of absorption coefficients corresponding to frequency bands.
    scattering_coefficients : list
        A list of scattering coefficients corresponding to frequency bands. Optional.
    transmission_coefficients : list
        A list of transmission coefficients corresponding to frequency bands. Optional.
    """
    
    def __init__(self, id: str, absorption_coefficients: list, scattering_coefficients: list = None, transmission_coefficients: list = None, description: str = None):
        super().__init__()
        self.id = id
        self.description = None
        self.number_of_frequency_bands = len(Defaults().DEFAULT_FREQUENCY_BANDS)
        self.absorption_coefficients = absorption_coefficients
        self.scattering_coefficients = scattering_coefficients
        self.transmission_coefficients = transmission_coefficients
        
    
    @property
    def id(self):
        return self._id
    
    
    @id.setter
    def id(self, id):
        if " " in id:
            raise ValueError("Material id cannot contain spaces.")
        if any(char.isupper() for char in id):
            raise ValueError("Material id must be lowercase.")
        self._id = id

    
    @property
    def absorption_coefficients(self):
        return self._absorption_coefficients
    
    
    @absorption_coefficients.setter
    def absorption_coefficients(self, coefficients: list):
        match len(coefficients):
            case 1:
                coefficients = coefficients * self.number_of_frequency_bands
                self._absorption_coefficients = coefficients
            case 8:
                self._absorption_coefficients = coefficients
            case _:
                raise ValueError(f"Absorption coefficients list must have {self.number_of_frequency_bands} values corresponding to the frequency bands.")
    
    
    @property
    def scattering_coefficients(self):
        return self._scattering_coefficients
    
    
    @scattering_coefficients.setter
    def scattering_coefficients(self, coefficients: list):
        if coefficients:
            match len(coefficients):
                case 1:
                    coefficients = coefficients * self.number_of_frequency_bands
                    self._scattering_coefficients = coefficients
                case 8:
                    self._scattering_coefficients = coefficients
                case _:
                    raise ValueError(f"Scattering coefficients list must have {self.number_of_frequency_bands} values corresponding to the frequency bands.")
        else:
            self._scattering_coefficients = coefficients
    
    
    @property
    def transmission_coefficients(self):
        return self._transmission_coefficients
    
    
    @transmission_coefficients.setter
    def transmission_coefficients(self, coefficients: list):
        if coefficients:
            match len(coefficients):
                case 1:
                    coefficients = coefficients * self.number_of_frequency_bands
                    self._transmission_coefficients = coefficients
                case 8:
                    self._transmission_coefficients = coefficients
                case _:
                    raise ValueError(f"Transmission coefficients list must have {self.number_of_frequency_bands} values corresponding to the frequency bands.")
        else:
            self._transmission_coefficients = coefficients


    @property
    def __data__(self):
        return{
            "id": self.id,
            "description": self.description or "",
            "absorption_coefficients": self.absorption_coefficients,
            "scattering_coefficients": self.scattering_coefficients or [],
            "transmission_coefficients": self.transmission_coefficients or [],
        }
      
  
    def __repr__(self):
        return f"AcousticMaterial(name={self.name}, absorption_coefficients={self.absorption_coefficients})"
    