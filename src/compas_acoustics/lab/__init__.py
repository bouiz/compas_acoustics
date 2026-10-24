"""Virtual lab: scattering coefficients of 3D surfaces (ISO 17497-1/-2)."""

from .bands import EXTENDED_BAND_RANGE
from .bands import ISO_BAND_RANGE
from .bands import THIRD_OCTAVE_CENTRES
from .bands import band_average
from .bands import fill_octaves
from .bands import fold_to_octaves
from .bands import simulation_frequencies
from .bands import third_octave_centres
from .coefficients import diffusion_coefficient
from .coefficients import normalised_diffusion
from .coefficients import random_incidence
from .coefficients import scattering_freefield
from .grids import ReceiverGrid
from .grids import SourceGrid
from .grids import receiver_spacing
from .results import ScatteringResult
from .sample import ScatteringSample
from .scattering import DEFAULT_PRESETS
from .scattering import measure_scattering

__all__ = [
    "DEFAULT_PRESETS",
    "EXTENDED_BAND_RANGE",
    "ISO_BAND_RANGE",
    "THIRD_OCTAVE_CENTRES",
    "ReceiverGrid",
    "ScatteringResult",
    "ScatteringSample",
    "SourceGrid",
    "band_average",
    "diffusion_coefficient",
    "fill_octaves",
    "fold_to_octaves",
    "measure_scattering",
    "normalised_diffusion",
    "random_incidence",
    "receiver_spacing",
    "scattering_freefield",
    "simulation_frequencies",
    "third_octave_centres",
]
