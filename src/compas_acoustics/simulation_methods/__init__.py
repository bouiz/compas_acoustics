"""Simulation methods: interchangeable ways to compute a scene's impulse responses.

See :mod:`compas_acoustics.simulation_methods.base` for the contract every method meets.
:class:`Pyroomacoustics` is always available; :class:`Hybrid` needs the
``hybrid`` extra (pyFDN, torch), so it is imported on first use.
"""

from importlib import import_module
from typing import TYPE_CHECKING

from .base import SimulationContext
from .base import SimulationMethod
from .base import image_source_search_seconds
from .pra import Pyroomacoustics

if TYPE_CHECKING:
    # for editors and type checkers; at runtime Hybrid is imported on first use (see __getattr__)
    from .hybrid import TAILS  # noqa: F401
    from .hybrid import Hybrid  # noqa: F401

__all__ = ["SimulationContext", "SimulationMethod", "Pyroomacoustics", "Hybrid", "image_source_search_seconds"]


def __getattr__(name):
    if name in ("Hybrid", "TAILS"):
        try:
            hybrid = import_module(f"{__name__}.hybrid")
        except ImportError as error:
            raise ImportError("the hybrid method needs Python >= 3.11 and the 'hybrid' extra: pip install compas_acoustics[hybrid]") from error
        return getattr(hybrid, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
