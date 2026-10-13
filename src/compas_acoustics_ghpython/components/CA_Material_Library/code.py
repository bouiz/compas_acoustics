# flake8: noqa
import Grasshopper
from System.Windows.Forms import ToolStripMenuItem
from System.Windows.Forms import ToolStripSeparator
from compas_acoustics.materials_library import MaterialsLibrary


class MaterialLibraryPicker(Grasshopper.Kernel.GH_ScriptInstance):

    def __init__(self):
        super().__init__()
        self.library = MaterialsLibrary()
        nickname = ghenv.Component.NickName
        self.material_name = nickname if nickname in self.library.material_names else None

    def RunScript(self):
        names = sorted(self.library.material_names)
        if not names:
            ghenv.Component.AddRuntimeMessage(
                Grasshopper.Kernel.GH_RuntimeMessageLevel.Error, "No material presets found in the materials library."
            )
            return None

        if self.material_name not in names:
            ghenv.Component.AddRuntimeMessage(
                Grasshopper.Kernel.GH_RuntimeMessageLevel.Remark, "Right-click the icon to choose a material."
            )
            ghenv.Component.Message = None
            return None

        ghenv.Component.Message = self.material_name
        return self.library.material_from_name(self.material_name)

    def AppendAdditionalMenuItems(self, menu):
        for category, names in self.library.materials_by_category.items():
            group = ToolStripMenuItem(category.value)
            group.Checked = self.material_name in names
            for name in names:
                item = group.DropDownItems.Add(name, None, self.on_item_click)
                item.Checked = (name == self.material_name)
            menu.Items.Add(group)

        menu.Items.Add(ToolStripSeparator())

    def on_item_click(self, sender, event_info):
        name = str(sender)
        if name == self.material_name:
            return
        self.material_name = name
        ghenv.Component.NickName = name
        ghenv.Component.ExpireSolution(True)
