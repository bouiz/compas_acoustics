"""Virtual lab: scattering coefficients of 3D surfaces (ISO 17497-1/-2)."""

from .bands import EXTENDED_BAND_RANGE
from .bands import ISO_BAND_RANGE
from .bands import THIRD_OCTAVE_CENTRES
from .bands import band_average
from .bands import fill_octaves
from .bands import fold_to_octaves
from .bands import simulation_frequencies
from .bands import third_octave_centres

__all__ = [
    "EXTENDED_BAND_RANGE",
    "ISO_BAND_RANGE",
    "THIRD_OCTAVE_CENTRES",
    "band_average",
    "fill_octaves",
    "fold_to_octaves",
    "simulation_frequencies",
    "third_octave_centres",
]
