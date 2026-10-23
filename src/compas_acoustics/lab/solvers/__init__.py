"""Free-field solvers for the scattering lab: the sample's and the reference's pressures at the receivers.

The BEM solver lives in ``solvers.bem`` and is not imported here, so the lab imports on Python 3.9 without its dependencies.
"""

from .base import Pressures
from .base import ScatteringSolver
from .kirchhoff import KirchhoffSolver

__all__ = ["KirchhoffSolver", "Pressures", "ScatteringSolver"]
