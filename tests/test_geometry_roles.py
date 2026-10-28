
import numpy as np
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


L_ROOM = ["v 0 0 0", "v 4 0 0", "v 4 2 0", "v 2 2 0", "v 2 4 0", "v 0 4 0",
          "v 0 0 2", "v 4 0 2", "v 4 2 2", "v 2 2 2", "v 2 4 2", "v 0 4 2"]


def write_obj(path, lines):
    path.write_text("\n".join(lines) + "\n")
    return str(path)


def test_objects_are_groups_when_there_are_no_groups(tmp_path):
    lines = L_ROOM + ["o floor", "f 1 6 5 4 3 2", "o ceiling", "f 7 8 9 10 11 12",
                      "o walls", "f 1 2 8 7", "f 2 3 9 8", "f 3 4 10 9", "f 4 5 11 10", "f 5 6 12 11", "f 6 1 7 12"]
    objs = AcousticObject.from_obj(write_obj(tmp_path / "l.obj", lines))
    assert sorted(o.group for o in objs) == ["ceiling", "floor", "walls"]
    assert {o.name for o in objs} == {"l_floor", "l_ceiling", "l_walls"}
    assert {o.group: o.acoustic_geometry.number_of_faces() for o in objs}["walls"] == 6


def test_negative_and_slashed_indices(tmp_path):
    lines = ["v 0 0 0", "v 1 0 0", "v 1 1 0", "v 0 1 0", "vt 0 0", "vn 0 0 1",
             "f -4/1/1 -1/1/1 -2/1/1 -3/1/1", "f 1//1 2//1 3//1", "f 1/1 3/1 4/1"]
    objs = AcousticObject.from_obj(write_obj(tmp_path / "q.obj", lines))
    assert len(objs) == 1 and objs[0].group == "default"
    mesh = objs[0].acoustic_geometry
    assert mesh.number_of_faces() == 3
    face = next(iter(mesh.faces()))
    assert np.allclose(mesh.face_coordinates(face), [[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0]])


def test_negative_indices_count_the_vertices_read_so_far(tmp_path):
    lines = ["v 0 0 0", "v 1 0 0", "v 1 1 0", "f -3 -2 -1", "v 5 5 5", "v 6 5 5", "v 6 6 5", "f -3 -2 -1"]
    mesh = AcousticObject.from_obj(write_obj(tmp_path / "r.obj", lines))[0].acoustic_geometry
    assert sorted(max(mesh.face_coordinates(f)[0]) for f in mesh.faces()) == [0, 5]


def test_g_wins_over_o(tmp_path):
    lines = L_ROOM + ["o room", "g floor", "f 1 6 5 4 3 2", "g ceiling", "f 7 8 9 10 11 12", "o other", "g floor", "f 1 2 8 7"]
    objs = AcousticObject.from_obj(write_obj(tmp_path / "g.obj", lines))
    assert {o.group for o in objs} == {"floor", "ceiling"}
    assert {o.group: o.acoustic_geometry.number_of_faces() for o in objs}["floor"] == 2


def test_vertices_are_not_rounded_and_degenerate_faces_are_skipped(tmp_path):
    lines = ["v 0 0 0", "v 1.0000004 0 0", "v 1 1 0", "g a", "f 1 2 3", "f 1 1 2", "f 1 2 1", "f 2 2 2"]
    mesh = AcousticObject.from_obj(write_obj(tmp_path / "d.obj", lines))[0].acoustic_geometry
    assert mesh.number_of_faces() == 1
    assert max(mesh.vertices_attribute("x")) == 1.0000004


def test_a_utf8_bom_and_non_ascii_names_are_read(tmp_path):
    path = tmp_path / "bom.obj"
    path.write_bytes("\ufeffv 0 0 0\nv 1 0 0\nv 1 1 0\ng St\u00fchle\nf 1 2 3\n".encode("utf-8"))
    (obj,) = AcousticObject.from_obj(str(path))
    assert obj.group == "St\u00fchle"
    assert np.allclose(obj.acoustic_geometry.face_coordinates(next(iter(obj.acoustic_geometry.faces()))), [[0, 0, 0], [1, 0, 0], [1, 1, 0]])
