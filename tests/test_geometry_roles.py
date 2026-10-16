import pytest
from compas.datastructures import Mesh

from compas_acoustics import AcousticMaterial
from compas_acoustics import AcousticObject
from compas_acoustics import AcousticObjectsGroup
from compas_acoustics import AcousticObjectsLibrary
from compas_acoustics import Scene


def panel(name, role=None):
    mesh = Mesh.from_vertices_and_faces([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], [[0, 1, 2, 3]])
    return AcousticObject(mesh, AcousticMaterial("m", 0.1), name=name, role=role)


def roles(*objects):
    scene = Scene("s")
    for obj in objects:
        scene.add_acoustic_object(obj)
    return {leaf.name: role for leaf, role in scene.iter_acoustic_objects_with_roles()}


def test_default_role_is_boundary():
    assert roles(panel("a")) == {"a": "boundary"}


def test_a_leaf_role_overrides_its_group():
    group = AcousticObjectsGroup("g", [panel("a"), panel("b", role="boundary")], role="obstacle")
    assert roles(group) == {"a": "obstacle", "b": "boundary"}


def test_the_nearest_group_wins():
    inner = AcousticObjectsGroup("i", [panel("a")], role="boundary")
    outer = AcousticObjectsGroup("o", [inner, panel("b")], role="obstacle")
    assert roles(outer) == {"a": "boundary", "b": "obstacle"}


def test_an_unknown_role_raises():
    with pytest.raises(ValueError, match="role"):
        panel("a", role="wall")
    with pytest.raises(ValueError, match="role"):
        AcousticObjectsGroup("g", [], role="wall")


def test_shoebox_room_surfaces_are_boundaries():
    resolved = roles(AcousticObjectsLibrary.shoebox_room(7, 5, 3))
    assert len(resolved) == 6
    assert set(resolved.values()) == {"boundary"}
