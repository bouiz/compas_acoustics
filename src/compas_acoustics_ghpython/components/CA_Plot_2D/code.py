# flake8: noqa
import Grasshopper
from Grasshopper import DataTree
from Grasshopper.Kernel.Data import GH_Path
import Rhino
from compas_acoustics_ghpython import Plotter2D


class Plot2D(Grasshopper.Kernel.GH_ScriptInstance):
    def RunScript(self, param, plane, x_size, y_size, settings):
        if param is None:
            ghenv.Component.AddRuntimeMessage(Grasshopper.Kernel.GH_RuntimeMessageLevel.Warning, "Input parameter 'param' failed to collect data.")
            return None

        plotter = Plotter2D(param, settings)
        plotter.plot(plane=plane, width=x_size, height=y_size)

        title = plotter.title
        x_title = plotter.x_axis_title
        y_title = plotter.y_axis_title
        x_labels = plotter.x_tick_labels
        y_labels = plotter.y_tick_labels
        values = plotter.data_values

        points = plotter.data_points
        polyline = plotter.data_polyline

        # Draw grid
        grid = DataTree[object]()
        items = [
            plotter.x_axis_line,
            plotter.y_axis_line,
            plotter.x_axis_points,
            plotter.y_axis_points,
        ]

        for i, item in enumerate(items):
            path = GH_Path(i)
            if isinstance(item, list):
                grid.AddRange(item, path)
            else:
                grid.Add(item, path)
        
        return title, x_title, y_title, x_labels, y_labels, values, points, polyline, grid