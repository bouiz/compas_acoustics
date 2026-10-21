from typing import TYPE_CHECKING

from .defaults import Defaults
# from .tools import Tools
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
from .analyser import AcousticAnalyser
from .simulation_methods import SimulationMethod, Pyroomacoustics
from .auraliser import Auraliser
from .audio_player import AudioPlayer
from .fitness import regularized_fitness


if TYPE_CHECKING:
    # for editors and type checkers; at runtime Hybrid is imported on first use (see __getattr__)
    from .simulation_methods.hybrid import Hybrid  # noqa: F401

__author__ = ["Achilleas Xydis"]
__copyright__ = "Gramazio Kohler Research"
__license__ = "MIT License"
__email__ = "xydis@arch.ethz.ch"
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
    'AcousticAnalyser',
    'SimulationMethod',
    'Pyroomacoustics',
    'Auraliser',
    'AudioPlayer',
    # 'Tools',
    'ImpulseResponse',
    'RoomGeometry',
    'regularized_fitness',
    ]


def __getattr__(name):
    # The hybrid method needs pyFDN, which compas_acoustics requires on Python >= 3.11 only, so it is imported on first use
    if name == "Hybrid":
        try:
            from .simulation_methods import Hybrid
        except ImportError as error:
            raise ImportError("compas_acoustics.Hybrid needs Python >= 3.11, where compas_acoustics installs pyFDN") from error
        return Hybrid
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
