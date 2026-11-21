"""Helpers to build a scene from a Rhino OBJ export.

Rhino writes one ``g`` group per layer, named after the full layer path with spaces between the levels
(``Acoustics::plywood`` becomes ``g Acoustics plywood``), and point objects as ``p`` lines. These helpers
read the point groups as source and receiver positions, pick each surface group's material from the
materials library by its layer name, and turn a Y-up export into the Z-up frame compas_acoustics uses.
"""

import math

from compas.geometry import Rotation

import compas_acoustics as ca

# Rhino's "Y up" OBJ export writes (x, y, z) as (x, z, -y): a quarter turn about X undoes it.
Y_UP_TO_Z_UP = Rotation.from_axis_and_angle([1, 0, 0], math.pi / 2)


def layer_name(group):
    """The last level of a Rhino layer path as OBJ writes it: ``'Acoustics plywood'`` -> ``'plywood'``."""
    return group.split()[-1]


def y_up_to_z_up(point):
    """A point of a Y-up OBJ in the Z-up frame."""
    x, y, z = point
    return [x, -z, y]


def read_obj_point_groups(filepath, y_up=False):
    """The positions of the ``p`` (point) elements of an OBJ file, per group name.

    Parameters
    ----------
    filepath : str
    y_up : bool, default=False
        Convert the positions from a Y-up export to Z-up.

    Returns
    -------
    dict
        Group name to a list of ``[x, y, z]``, in file order.
    """
    vertices, groups = [], {}
    group = "default"
    with open(filepath, encoding="utf-8-sig") as f:
        for line in f:
            tokens = line.split()
            if not tokens:
                continue
            if tokens[0] == "v":
                vertices.append([float(value) for value in tokens[1:4]])
            elif tokens[0] == "g":
                group = " ".join(tokens[1:]) or "default"
            elif tokens[0] == "p":
                for token in tokens[1:]:
                    index = int(token.split("/")[0])
                    point = vertices[index if index < 0 else index - 1]
                    groups.setdefault(group, []).append(y_up_to_z_up(point) if y_up else list(point))
    return groups


def points_on_layer(point_groups, layer):
    """All points of the groups whose layer name (:func:`layer_name`) is ``layer``."""
    return [point for group, points in point_groups.items() if layer_name(group) == layer for point in points]


def materials_by_layer(groups, aliases=None, library=None):
    """A library material for every OBJ group, looked up by its layer name.

    Parameters
    ----------
    groups : iterable of str
        OBJ group names, e.g. ``[o.group for o in AcousticObject.from_obj(path)]``.
    aliases : dict, optional
        Layer name to library material name, for layers not named after a library material.
    library : :class:`compas_acoustics.MaterialsLibrary`, optional

    Returns
    -------
    dict
        Group name to :class:`compas_acoustics.AcousticMaterial`, as
        :meth:`compas_acoustics.AcousticObject.assign_materials_by_group` takes it.
    """
    aliases = aliases or {}
    library = library or ca.MaterialsLibrary()
    materials, missing = {}, []
    for group in groups:
        name = aliases.get(layer_name(group), layer_name(group))
        if name not in library.material_names:
            missing.append(f"{group!r} -> {name!r}")
            continue
        materials[group] = library.material_from_name(name)
    if missing:
        raise ValueError(f"No library material for {', '.join(missing)}; add them to the aliases")
    return materials


def set_scattering_by_layer(materials, scattering):
    """Override the scattering coefficients of the materials of the named layers.

    Parameters
    ----------
    materials : dict
        Group name to :class:`compas_acoustics.AcousticMaterial`, as :func:`materials_by_layer` returns it
        (one material instance per group, so overriding one group leaves the others alone).
    scattering : dict
        Layer name to one coefficient for every band, or one per band.

    Returns
    -------
    dict
        ``materials``, changed in place.
    """
    for group, material in materials.items():
        if layer_name(group) in scattering:
            material.scattering_coefficients = scattering[layer_name(group)]
    return materials
