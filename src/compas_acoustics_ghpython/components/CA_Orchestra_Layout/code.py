# flake8: noqa
import Grasshopper
import Rhino
from Rhino.Geometry import Point3d
from System.Windows.Forms import ToolStripSeparator

from compas_rhino.conversions import plane_to_compas_frame

from compas_acoustics.orchestra_layouts import (
    string_quartet,
    chamber_strings,
    medium_chamber_strings,
    symphonic_strings,
)


class OrchestraLayoutSelector(Grasshopper.Kernel.GH_ScriptInstance):
    LAYOUTS = {
        "string_quartet_4": string_quartet,
        "chamber_strings_16": chamber_strings,
        "chamber_strings_24": medium_chamber_strings,
        "symphonic_strings_50": symphonic_strings,
    }
    ITEMS = list(LAYOUTS.keys())

    def __init__(self):
        super().__init__()
        nickname = ghenv.Component.NickName
        self.layout_name = nickname if nickname in self.ITEMS else None

    def RunScript(self, plane):
        if self.layout_name not in self.ITEMS:
            ghenv.Component.AddRuntimeMessage(
                Grasshopper.Kernel.GH_RuntimeMessageLevel.Remark, "Right-click the icon to choose an orchestra layout."
            )
            ghenv.Component.Message = None
            return

        ghenv.Component.Message = self.layout_name
        frame = plane_to_compas_frame(plane) if plane is not None else None

        layout_fn = self.LAYOUTS[self.layout_name]
        points, names = layout_fn(frame)
        return [Point3d(*p) for p in points], names
    
    def AppendAdditionalMenuItems(self, menu):
        for name in self.ITEMS:
            item = menu.Items.Add(name, None, self.on_item_click)
            item.Checked = (name == self.layout_name)

        menu.Items.Add(ToolStripSeparator())

    def on_item_click(self, sender, event_info):
        name = str(sender)
        if name == self.layout_name:
            return
        self.layout_name = name
        ghenv.Component.NickName = name
        ghenv.Component.ExpireSolution(True)
