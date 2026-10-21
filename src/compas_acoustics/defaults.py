import functools
import os

__all__ = ['Defaults']


@functools.lru_cache(maxsize=None)
def _speed_of_sound() -> float:
    # The air every simulation runs in without one given; cached, since Cramer's formula is behind an import
    from compas_acoustics.air import AirConditions

    return AirConditions().speed_of_sound


class Defaults:
    _DEFAULT_FREQUENCY_BANDS = [125, 250, 500, 1000, 2000, 4000, 8000]
    _MATERIALS_LIBRARY_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), 'materials_library'))
    _HRTF_LIBRARY_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), 'hrtfs_library'))
    _ANECHOIC_SOUNDFILES_LIBRARY_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), 'anechoic_soundfiles_library'))
    _SAMPLING_RATE = 48000

    _PARAM_NAMES = ['EDT', 'T20', 'T30', 'C50', 'C80', 'D50', 'DRR', 'Ts', 'STI', 'G']

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
    def DEFAULT_SPEED_OF_SOUND(self):
        """The speed of sound of ``AirConditions()`` (20 deg C, 50 %, 101325 Pa), in m/s."""
        return _speed_of_sound()
    
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