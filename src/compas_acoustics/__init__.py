from typing import TYPE_CHECKING

from .defaults import Defaults
from .room import AirConditions
from .materials import AcousticMaterial
from .materials import MaterialCategory
from .materials import MaterialsLibrary
from .source import Source
from .receiver import Receiver
from .acoustic_object import AcousticObject, AcousticObjectsLibrary
from .ir import ImpulseResponse
from .room import RoomGeometry
from .rays import Ray, ScatteredRay, RayField
from .rays import SpecularPath, SpecularReflections
from .scene import Scene
from .analyser import Analyser
from .energy_map import EnergyMap
from .simulation_methods import SimulationMethod, Mono
from .auraliser import Auraliser
from .audio_player import AudioPlayer
from .fitness import regularized_fitness


if TYPE_CHECKING:
    # for editors and type checkers; at runtime they are imported on first use (see __getattr__)
    from .simulation_methods.hybrid import Ambisonics  # noqa: F401
    from .simulation_methods.hybrid import Binaural  # noqa: F401


__version__ = "0.1.0"

__all__ = [
    'Defaults',
    'AirConditions',
    'AcousticMaterial',
    'MaterialCategory',
    'MaterialsLibrary',
    'Source',
    'Receiver',
    'AcousticObject',
    'AcousticObjectsLibrary',
    'Ray',
    'ScatteredRay',
    'RayField',
    'SpecularPath',
    'SpecularReflections',
    'Scene',
    'Analyser',
    'EnergyMap',
    'SimulationMethod',
    'Mono',
    'Auraliser',
    'AudioPlayer',
    'ImpulseResponse',
    'RoomGeometry',
    'regularized_fitness',
    ]


def __getattr__(name):
    # The binaural and ambisonics methods need pyFDN, which compas_acoustics requires on Python >= 3.11 only, so they are imported on first use
    if name in ("Binaural", "Ambisonics"):
        try:
            from . import simulation_methods
            return getattr(simulation_methods, name)
        except ImportError as error:
            raise ImportError(f"compas_acoustics.{name} needs Python >= 3.11, where compas_acoustics installs pyFDN") from error
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
