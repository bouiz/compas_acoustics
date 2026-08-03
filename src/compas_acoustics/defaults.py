import os

__all__ = ['Defaults']

class Defaults:
    _DEFAULT_FREQUENCY_BANDS = [125, 250, 500, 1000, 2000, 4000, 8000]
    _DEFAULT_SPEED_OF_SOUND = 343
    _MATERIALS_LIBRARY_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), 'materials_library'))
    _HRTF_LIBRARY_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), 'hrtfs_library'))
    _ANECHOIC_SOUNDFILES_LIBRARY_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), 'anechoic_soundfiles_library'))
    _SAMPLING_RATE = 48000

    _PARAM_NAMES = ['EDT', 'T20', 'T30', 'C50', 'C80', 'D50', 'DRR', 'Ts', 'STI',]

    @property
    def SAMPLING_RATE(self):
        return self._SAMPLING_RATE
    
    @property
    def DEFAULT_FREQUENCY_BANDS(self):
        return self._DEFAULT_FREQUENCY_BANDS
    
    @property
    def NUMBER_OF_FREQUENCY_BANDS(self):
        return len(self._DEFAULT_FREQUENCY_BANDS)
    
    @property
    def DEFAULT_SPEED_OF_SOUND(self) :
        return self._DEFAULT_SPEED_OF_SOUND
    
    @property
    def MATERIALS_LIBRARY_PATH(self):
        return self._MATERIALS_LIBRARY_PATH

    @property
    def HRTF_LIBRARY_PATH(self):
        return self._HRTF_LIBRARY_PATH

    @property
    def ANECHOIC_SOUNDFILES_LIBRARY_PATH(self):
        return self._ANECHOIC_SOUNDFILES_LIBRARY_PATH

    @property
    def PARAM_NAMES(self):
        return self._PARAM_NAMES