# Regenerates ../../components/CA_Directivity.ghuser from metadata.json + icon.png.
#
# This is a native Grasshopper GH_ValueList (the stock "Value List" dropdown object),
# not a GHPython script component, so it cannot go through the `code.py`-based
# `invoke build-cpython-ghuser-components` pipeline (compas-actions.ghpython_components'
# componentizer hardcodes a script-component GUID and requires code.py/metadata.json's
# "ghpython" block). It also can't live as a folder directly under
# ghpython/components/, because that folder is scanned by the componentizer and any
# subfolder lacking code.py aborts the whole build. So this lives here instead, and the
# built artifact is committed as the loose file ghpython/components/CA_Directivity.ghuser
# (a sibling FILE, not a folder, so the componentizer's directory scan skips it and its
# rmtree-then-rebuild of ghuser/ never touches it).
#
# Run this from within Rhino 8 (Script Editor, or the LAMCP bridge's run_python_script),
# where `Grasshopper`/`System` are available. Edit metadata.json to change the preset
# list, then re-run.

import json
import os

import Grasshopper as gh
import System.Drawing as sd

HERE = os.path.dirname(os.path.abspath(__file__))
TARGET = os.path.abspath(os.path.join(HERE, "..", "..", "components", "CA_Directivity.ghuser"))

with open(os.path.join(HERE, "metadata.json")) as f:
    meta = json.load(f)

vl = gh.Kernel.Special.GH_ValueList()
vl.CreateAttributes()
vl.NickName = meta["nickname"]
vl.Name = meta["name"]
vl.Description = meta["description"]
vl.ListMode = getattr(gh.Kernel.Special.GH_ValueListMode, meta["listMode"])

vl.ListItems.Clear()
for item in meta["items"]:
    vl.ListItems.Add(gh.Kernel.Special.GH_ValueListItem(item["name"], item["expression"]))
vl.SelectItem(meta["defaultSelectedIndex"])

icon = sd.Bitmap(os.path.join(HERE, "icon.png"))

uo = gh.Kernel.GH_UserObject()
uo.Icon = icon
uo.Exposure = getattr(gh.Kernel.GH_Exposure, "primary")  # meta["exposure"] == 2
uo.BaseGuid = vl.ComponentGuid  # required: identifies the underlying GH type to instantiate
uo.Description.Name = meta["name"]
uo.Description.NickName = meta["nickname"]
uo.Description.Description = meta["description"]
uo.Description.Category = meta["category"]
uo.Description.SubCategory = meta["subcategory"]
uo.SetDataFromObject(vl)

uo.Path = TARGET
saved = uo.SaveToFile()
print("saved" if saved else "FAILED", TARGET)
