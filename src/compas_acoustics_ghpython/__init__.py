"""Rhino/Grasshopper integration for compas_acoustics.

Everything that depends on RhinoCommon or Grasshopper lives here, so that
the core ``compas_acoustics`` package stays CAD-agnostic.
"""

import os

from .ir_graph import MethodGraph
from .plotter import AcousticParameter
from .plotter import Plotter2D
from .plotter import Plotter2DSettings
from .value_list import ensure_inputs

__version__ = "0.1.0"

HERE = os.path.dirname(__file__)
USEROBJECTS_PATH = os.path.join(HERE, "components", "ghuser")

__all__ = [
    "AcousticParameter",
    "MethodGraph",
    "Plotter2D",
    "Plotter2DSettings",
    "ensure_inputs",
    "USEROBJECTS_PATH",
]
