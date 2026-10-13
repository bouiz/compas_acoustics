# flake8: noqa
import Grasshopper
from Grasshopper import DataTree
from Grasshopper.Kernel.Data import GH_Path
from compas_acoustics_ghpython import Plotter2DSettings


class Plot2DSettings(Grasshopper.Kernel.GH_ScriptInstance):
    def RunScript(self, title_height=None, axis_title_height=None, label_height=None):
        return Plotter2DSettings(
            title_height=title_height,
            axis_title_height=axis_title_height,
            label_height=label_height,
        )