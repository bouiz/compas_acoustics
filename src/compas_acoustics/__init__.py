"""Room acoustic simulation, analysis and auralisation for COMPAS.

The public names are imported on first use (PEP 562), so ``import compas_acoustics`` is cheap and a
subpackage that needs neither pyroomacoustics nor pyfar stays light.
``import compas_acoustics as ca; ca.Mono`` and ``from compas_acoustics import Mono`` work as before.
"""

from importlib import import_module
from typing import TYPE_CHECKING

__version__ = "0.2.0"

# public name -> module it is defined in
_EXPORTS = {
    "Defaults": ".defaults",
    "AirConditions": ".room",
    "AcousticMaterial": ".materials",
    "MaterialCategory": ".materials",
    "MaterialsLibrary": ".materials",
    "Source": ".source",
    "Receiver": ".receiver",
    "AcousticObject": ".acoustic_object",
    "AcousticObjectsLibrary": ".acoustic_object",
    "ImpulseResponse": ".ir",
    "RoomGeometry": ".room",
    "Ray": ".rays",
    "ScatteredRay": ".rays",
    "RayField": ".rays",
    "SpecularPath": ".rays",
    "SpecularReflections": ".rays",
    "Scene": ".scene",
    "Analyser": ".analyser",
    "EnergyMap": ".energy_map",
    "SimulationMethod": ".simulation_methods",
    "Mono": ".simulation_methods",
    "Auraliser": ".auraliser",
    "AudioPlayer": ".audio_player",
    "regularized_fitness": ".fitness",
    "ScatteringResult": ".lab",
    "ScatteringSample": ".lab",
}

# need pyFDN, which compas_acoustics installs on Python >= 3.11
_HYBRID = ("Binaural", "Ambisonics")

if TYPE_CHECKING:
    # for editors and type checkers
    from .acoustic_object import AcousticObject  # noqa: F401
    from .acoustic_object import AcousticObjectsLibrary  # noqa: F401
    from .analyser import Analyser  # noqa: F401
    from .audio_player import AudioPlayer  # noqa: F401
    from .auraliser import Auraliser  # noqa: F401
    from .defaults import Defaults  # noqa: F401
    from .energy_map import EnergyMap  # noqa: F401
    from .fitness import regularized_fitness  # noqa: F401
    from .ir import ImpulseResponse  # noqa: F401
    from .lab import ScatteringResult  # noqa: F401
    from .lab import ScatteringSample  # noqa: F401
    from .materials import AcousticMaterial  # noqa: F401
    from .materials import MaterialCategory  # noqa: F401
    from .materials import MaterialsLibrary  # noqa: F401
    from .rays import Ray  # noqa: F401
    from .rays import RayField  # noqa: F401
    from .rays import ScatteredRay  # noqa: F401
    from .rays import SpecularPath  # noqa: F401
    from .rays import SpecularReflections  # noqa: F401
    from .receiver import Receiver  # noqa: F401
    from .room import AirConditions  # noqa: F401
    from .room import RoomGeometry  # noqa: F401
    from .scene import Scene  # noqa: F401
    from .simulation_methods import Mono  # noqa: F401
    from .simulation_methods import SimulationMethod  # noqa: F401
    from .simulation_methods.hybrid import Ambisonics  # noqa: F401
    from .simulation_methods.hybrid import Binaural  # noqa: F401
    from .source import Source  # noqa: F401

# spelled out, so that documentation tools, which do not run this module, see the public names
__all__ = [
    "Defaults",
    "AirConditions",
    "AcousticMaterial",
    "MaterialCategory",
    "MaterialsLibrary",
    "Source",
    "Receiver",
    "AcousticObject",
    "AcousticObjectsLibrary",
    "ImpulseResponse",
    "RoomGeometry",
    "Ray",
    "ScatteredRay",
    "RayField",
    "SpecularPath",
    "SpecularReflections",
    "Scene",
    "Analyser",
    "EnergyMap",
    "SimulationMethod",
    "Mono",
    "Auraliser",
    "AudioPlayer",
    "regularized_fitness",
    "ScatteringResult",
    "ScatteringSample",
]


def __getattr__(name):
    module = _EXPORTS.get(name)
    if module is not None:
        value = getattr(import_module(module, __name__), name)
        globals()[name] = value  # later lookups skip this function
        return value
    if name in _HYBRID:
        try:
            from . import simulation_methods

            value = getattr(simulation_methods, name)
        except ImportError as error:
            raise ImportError(f"compas_acoustics.{name} needs Python >= 3.11, where compas_acoustics installs pyFDN") from error
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    # the hybrid methods are left out, as before the lazy import: where pyFDN is missing they raise ImportError, which
    # breaks tools that read every listed name (inspect.getmembers)
    return sorted(set(globals()) | set(_EXPORTS))
