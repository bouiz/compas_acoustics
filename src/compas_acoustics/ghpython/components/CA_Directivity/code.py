# flake8: noqa
import Grasshopper
from System.Windows.Forms import ToolStripSeparator
from compas_acoustics.materials_library import MaterialsLibrary


class DirectivitySelector(Grasshopper.Kernel.GH_ScriptInstance):
    ITEMS = ["Bidirectional", "Hypercardioid", "Supercardioid", "Cardioid", "Omnidirectional"]

    def __init__(self):
        super().__init__()
        nickname = ghenv.Component.NickName
        self.directivity_name = nickname if nickname in self.ITEMS else None

    def RunScript(self):
        names = self.ITEMS
        if not names:
            ghenv.Component.AddRuntimeMessage(
                Grasshopper.Kernel.GH_RuntimeMessageLevel.Error, "No directivity patterns found."
            )
            return None

        if self.directivity_name not in names:
            ghenv.Component.AddRuntimeMessage(
                Grasshopper.Kernel.GH_RuntimeMessageLevel.Remark, "Right-click the icon to choose a directivity pattern."
            )
            ghenv.Component.Message = None
            return None

        ghenv.Component.Message = self.directivity_name
        return self.directivity_name.lower()

    def AppendAdditionalMenuItems(self, menu):
        for name in self.ITEMS:
            item = menu.Items.Add(name, None, self.on_item_click)
            item.Checked = (name == self.directivity_name)

        menu.Items.Add(ToolStripSeparator())

    def on_item_click(self, sender, event_info):
        name = str(sender)
        if name == self.directivity_name:
            return
        self.directivity_name = name
        ghenv.Component.NickName = name
        ghenv.Component.ExpireSolution(True)
