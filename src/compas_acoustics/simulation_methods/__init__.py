"""Simulation methods: interchangeable ways to compute a scene's impulse responses.

See :mod:`compas_acoustics.simulation_methods.base` for the contract every method meets.
:class:`Mono` is always available; :class:`Binaural` and :class:`Ambisonics`, the
hybrid engine of :mod:`compas_acoustics.simulation_methods.hybrid`, need pyFDN (Python >= 3.11),
so they are imported on first use.
"""

from importlib import import_module
from typing import TYPE_CHECKING

from .base import SimulationContext
from .base import SimulationMethod
from .base import image_source_search_seconds
from .pra import Mono

if TYPE_CHECKING:
    # for editors and type checkers; at runtime they are imported on first use (see __getattr__)
    from .hybrid import TAILS  # noqa: F401
    from .hybrid import Ambisonics  # noqa: F401
    from .hybrid import Binaural  # noqa: F401

# Binaural and Ambisonics are left out: a star import would load pyFDN, or fail without it
__all__ = ["SimulationContext", "SimulationMethod", "Mono", "image_source_search_seconds"]

_HYBRID = ("Binaural", "Ambisonics", "TAILS")


def __getattr__(name):
    if name in _HYBRID:
        try:
            hybrid = import_module(f"{__name__}.hybrid")
        except ImportError as error:
            raise ImportError(f"{name} needs Python >= 3.11, where compas_acoustics installs pyFDN") from error
        return getattr(hybrid, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
