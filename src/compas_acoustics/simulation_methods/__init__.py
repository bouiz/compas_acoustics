"""Interchangeable ways to compute a scene's impulse responses (contract: :mod:`.base`).

:class:`Binaural` and :class:`Ambisonics` need pyFDN (Python >= 3.11), so they are imported on first use.
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


def __getattr__(name):
    if name in ("Binaural", "Ambisonics", "TAILS"):
        try:
            hybrid = import_module(f"{__name__}.hybrid")
        except ImportError as error:
            raise ImportError(f"{name} needs the hybrid dependencies: pip install compas_acoustics[hybrid] (Python >= 3.11)") from error
        return getattr(hybrid, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
