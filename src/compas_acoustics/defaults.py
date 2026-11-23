import functools
import numbers
import os

import numpy as np

__all__ = ["Defaults"]


class _PackageDtype:
    """Names a class's data type ``compas_acoustics/<ClassName>``, so that renaming a module keeps its JSON readable."""

    @property
    def __dtype__(self):
        return "compas_acoustics/{}".format(type(self).__name__)


@functools.lru_cache(maxsize=None)
def _speed_of_sound() -> float:
    from compas_acoustics.room.air import AirConditions

    return AirConditions().speed_of_sound


def _per_band(values, label):
    """One float per frequency band from one number, or from any iterable of one or one per band.

    Any iterable, since Grasshopper passes a .NET List[Double].
    """
    count = Defaults.NUMBER_OF_FREQUENCY_BANDS
    values = [float(values)] if isinstance(values, numbers.Real) else np.asarray(list(values), dtype=float).reshape(-1).tolist()
    if len(values) == 1:
        values = values * count
    if len(values) != count:
        raise ValueError(f"{label} must be one value or {count}, one per frequency band, got {len(values)}.")
    return values


class Defaults:
    SAMPLING_RATE = 48000
    DEFAULT_FREQUENCY_BANDS = [125, 250, 500, 1000, 2000, 4000, 8000]
    NUMBER_OF_FREQUENCY_BANDS = len(DEFAULT_FREQUENCY_BANDS)
    MATERIALS_LIBRARY_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "materials_library"))
    HRTF_LIBRARY_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "hrtfs_library"))
    ANECHOIC_SOUNDFILES_LIBRARY_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "anechoic_soundfiles_library"))

    @property
    def DEFAULT_SPEED_OF_SOUND(self):
        """The speed of sound of ``AirConditions()`` (20 deg C, 50 %, 101325 Pa), in m/s."""
        return _speed_of_sound()
