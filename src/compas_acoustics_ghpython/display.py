"""Colours and viewport drawing shared by the display components."""

import Rhino
import System

__all__ = ["order_color", "bounding_box", "typed_list", "POINT_STYLE"]

# Order 0 (the direct sound) black, then matplotlib's tab10 from order 1, repeating after order 10
_ORDER_RGB = [
    (0, 0, 0),
    (214, 39, 40),
    (31, 119, 180),
    (44, 160, 44),
    (255, 127, 14),
    (148, 103, 189),
    (140, 86, 75),
    (227, 119, 194),
    (127, 127, 127),
    (188, 189, 34),
    (23, 190, 207),
]

POINT_STYLE = Rhino.Display.PointStyle.RoundSimple


def order_color(order):
    """The colour of a reflection order, the same in every display component."""
    order = int(order)
    rgb = _ORDER_RGB[0] if order <= 0 else _ORDER_RGB[1 + (order - 1) % (len(_ORDER_RGB) - 1)]
    return System.Drawing.Color.FromArgb(*rgb)


def typed_list(item_type, items):
    """A .NET List for the display pipeline's IEnumerable arguments, filled once per solve."""
    result = System.Collections.Generic.List[item_type]()
    for item in items:
        result.Add(item)
    return result


def bounding_box(points):
    """The box around what a component draws, so that zoom extents includes it."""
    points = list(points)
    if not points:
        return Rhino.Geometry.BoundingBox.Empty
    return Rhino.Geometry.BoundingBox(typed_list(Rhino.Geometry.Point3d, points))
