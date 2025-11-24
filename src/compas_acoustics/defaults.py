import os

__all__ = ['Defaults']

class Defaults:
    _DEFAULT_FREQUENCY_BANDS = [63, 125, 250, 500, 1000, 2000, 4000, 8000]
    _DEFAULT_SCATTERING_COEFFICENTS = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    _DEFAULT_ABSORPTION_COEFFICIENTS = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    _DEFAULT_TRANSMISSION_COEFFICIENTS = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    _METERIALS_LIBRARY_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), 'materials_library'))
    
    @property
    def DEFAULT_FREQUENCY_BANDS(self):
        return self._DEFAULT_FREQUENCY_BANDS
    
    @property
    def NUMBER_OF_FREQUENCY_BANDS(self):
        return len(self._DEFAULT_FREQUENCY_BANDS)
    
    @property
    def DEFAULT_SCATTERING_COEFFICENTS(self):
        return self._DEFAULT_SCATTERING_COEFFICENTS

    @property
    def DEFAULT_ABSORPTION_COEFFICIENTS(self):
        return self._DEFAULT_ABSORPTION_COEFFICIENTS
    
    @property
    def DEFAULT_TRANSMISSION_COEFFICIENTS(self):
        return self._DEFAULT_TRANSMISSION_COEFFICIENTS
    
    @property
    def MATERIALS_LIBRARY_PATH(self):
        return self._METERIALS_LIBRARY_PATH