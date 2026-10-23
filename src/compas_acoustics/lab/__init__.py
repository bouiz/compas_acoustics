"""Virtual lab: scattering coefficients of 3D surfaces (ISO 17497-1/-2)."""

from .bands import EXTENDED_BAND_RANGE
from .bands import ISO_BAND_RANGE
from .bands import THIRD_OCTAVE_CENTRES
from .bands import band_average
from .bands import fill_octaves
from .bands import fold_to_octaves
from .bands import simulation_frequencies
from .bands import third_octave_centres
from .coefficients import random_incidence
from .coefficients import scattering_freefield
from .grids import ReceiverGrid
from .grids import SourceGrid
from .grids import receiver_spacing

__all__ = [
    "EXTENDED_BAND_RANGE",
    "ISO_BAND_RANGE",
    "THIRD_OCTAVE_CENTRES",
    "ReceiverGrid",
    "SourceGrid",
    "band_average",
    "fill_octaves",
    "fold_to_octaves",
    "random_incidence",
    "receiver_spacing",
    "scattering_freefield",
    "simulation_frequencies",
    "third_octave_centres",
]
