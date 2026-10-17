import pytest
from compas.datastructures import Mesh

from compas_acoustics import AcousticMaterial
from compas_acoustics import AcousticObject
from compas_acoustics import AcousticObjectsLibrary


def panel(name, role="boundary"):
    mesh = Mesh.from_vertices_and_faces([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], [[0, 1, 2, 3]])
    return AcousticObject(mesh, AcousticMaterial("m", 0.1), name=name, role=role)


def test_default_role_is_boundary():
    mesh = Mesh.from_vertices_and_faces([[0, 0, 0], [1, 0, 0], [1, 1, 0]], [[0, 1, 2]])
    assert AcousticObject(mesh, AcousticMaterial("m", 0.1)).role == "boundary"


def test_an_obstacle_keeps_its_role():
    assert panel("a", role="obstacle").role == "obstacle"


def test_an_unknown_role_raises():
    with pytest.raises(ValueError, match="role"):
        panel("a", role="wall")
    with pytest.raises(ValueError, match="role"):
        panel("a", role=None)


def test_shoebox_room_surfaces_are_boundaries():
    room = AcousticObjectsLibrary.shoebox_room(7, 5, 3)
    assert len(room) == 6
    assert {o.role for o in room} == {"boundary"}
