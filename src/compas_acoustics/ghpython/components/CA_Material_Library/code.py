# flake8: noqa
import Grasshopper
from compas_acoustics.materials_library import MaterialsLibrary


class MaterialLibraryPicker(Grasshopper.Kernel.GH_ScriptInstance):

    def __init__(self):
        super(MaterialLibraryPicker, self).__init__()
        self.library = MaterialsLibrary()
        nickname = ghenv.Component.Params.Output[0].NickName
        self.material_name = nickname if nickname in self.library.material_names else None

    def RunScript(self):
        names = sorted(self.library.material_names)
        if not names:
            ghenv.Component.AddRuntimeMessage(
                Grasshopper.Kernel.GH_RuntimeMessageLevel.Error, "No material presets found in the materials library."
            )
            return None

        if self.material_name not in names:
            self.material_name = names[0]
            self._rename_output(self.material_name)
            ghenv.Component.AddRuntimeMessage(
                Grasshopper.Kernel.GH_RuntimeMessageLevel.Remark, "Right-click to choose a different material."
            )

        ghenv.Component.Message = self.material_name
        return self.library.material_from_name(self.material_name)

    def AppendAdditionalMenuItems(self, menu):
        for name in sorted(self.library.material_names):
            item = menu.Items.Add(name, None, self.on_item_click)
            item.Checked = (name == self.material_name)

    def on_item_click(self, sender, event_info):
        name = str(sender)
        if name == self.material_name:
            return
        self.material_name = name
        self._rename_output(name)
        ghenv.Component.ExpireSolution(True)

    def _rename_output(self, name):
        param = ghenv.Component.Params.Output[0]
        param.NickName = name
        param.Name = name
        param.Description = "AcousticMaterial: {}".format(name)
