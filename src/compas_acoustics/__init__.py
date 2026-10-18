from __future__ import print_function

from typing import TYPE_CHECKING

from .defaults import Defaults
# from .tools import Tools
from .air import AirConditions
from .acoustic_material import AcousticMaterial
from .acoustic_material import MaterialCategory
from .materials_library import MaterialsLibrary
from .source import Source
from .receiver import Receiver
from .geometry import AcousticObject, AcousticObjectsLibrary
from .ir import ImpulseResponse
from .room_geometry import RoomGeometry
from .ray import Ray, ScatteredRay, RayField
from .scene import Scene
from .analyser import AcousticAnalyser
from .simulation_methods import SimulationMethod, Pyroomacoustics
from .auraliser import Auraliser
from .audio_player import AudioPlayer
from .numerical.fitness import regularized_fitness
from . import pra_converter


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
    'pra_converter',
    ]


def __getattr__(name):
    # The hybrid method needs the optional 'hybrid' extra (pyFDN, torch; Python >= 3.11), so it is imported on first use
    if name == "Hybrid":
        try:
            from .simulation_methods import Hybrid
        except ImportError as error:
            raise ImportError("compas_acoustics.Hybrid needs Python >= 3.11 and the 'hybrid' extra: pip install compas_acoustics[hybrid]") from error
        return Hybrid
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
