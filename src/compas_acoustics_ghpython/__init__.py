"""Rhino/Grasshopper integration for compas_acoustics.

Everything that depends on RhinoCommon or Grasshopper lives here, so that
the core ``compas_acoustics`` package stays CAD-agnostic.
"""

from __future__ import print_function

import os

from .plotter import AcousticParameter
from .plotter import Plotter2D
from .plotter import Plotter2DSettings

__version__ = "0.1.0"

HERE = os.path.dirname(__file__)
USEROBJECTS_PATH = os.path.join(HERE, "components", "ghuser")

__all__ = [
    "AcousticParameter",
    "Plotter2D",
    "Plotter2DSettings",
    "USEROBJECTS_PATH",
]
