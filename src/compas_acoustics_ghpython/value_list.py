"""Value lists and buttons that Grasshopper components place next to their unconnected inputs."""

import scriptcontext as sc
import System
from Grasshopper.Kernel.Special import GH_ButtonObject
from Grasshopper.Kernel.Special import GH_ValueList
from Grasshopper.Kernel.Special import GH_ValueListItem

__all__ = ["ensure_inputs"]

# Distance, in canvas units, between the new object and the component
_OFFSET = 220


def _button(name):
    button = GH_ButtonObject()
    button.CreateAttributes()
    button.NickName = name.capitalize()
    return button


def _value_list(name, options):
    value_list = GH_ValueList()
    value_list.CreateAttributes()
    value_list.NickName = name
    value_list.ListItems.Clear()
    for option in options:
        value_list.ListItems.Add(GH_ValueListItem(option, '"{}"'.format(option)))
    return value_list


def ensure_inputs(component, inputs):
    """Give each unconnected input a value list or a button, once per component and Rhino session.

    The objects are added to the canvas after the current solution, wired to the input, and left alone from then on.
    An input the user connects, or removes the object from, is not touched again in the same session.

    Parameters
    ----------
    component : Grasshopper.Kernel.IGH_Component
        The component, as ``ghenv.Component``.
    inputs : dict
        Input name to the list of options for a value list, or to None for a button named after the input.

    Returns
    -------
    None
    """
    key = "compas_acoustics_inputs_offered_{}".format(component.InstanceGuid)
    if sc.sticky.get(key):
        return
    sc.sticky[key] = True

    doc = component.OnPingDocument()
    if doc is None:
        return

    targets = [(param.Name, inputs[param.Name]) for param in component.Params.Input if param.Name in inputs and param.SourceCount == 0]
    if not targets:
        return

    def deferred(gh_doc):
        for name, options in targets:
            param = next(p for p in component.Params.Input if p.Name == name)
            if param.SourceCount > 0:
                continue
            added = _button(name) if options is None else _value_list(name, options)
            added.Attributes.Pivot = System.Drawing.PointF(component.Attributes.Pivot.X - _OFFSET, param.Attributes.Pivot.Y - 10)
            gh_doc.AddObject(added, False)
            param.AddSource(added)
            added.ExpireSolution(True)

    doc.ScheduleSolution(1, deferred)
