import numpy as np
import pytest
from compas.datastructures import Mesh
from compas.geometry import Box
from compas.geometry import Frame

from compas_acoustics import AcousticMaterial
from compas_acoustics import AcousticObject
from compas_acoustics import AcousticObjectsLibrary
from compas_acoustics import Scene
from compas_acoustics.room_geometry import OBSTACLE_OFFSET
from compas_acoustics.room_geometry import SYNTHESIS_BANDS
from compas_acoustics.room_geometry import RoomGeometry

WALL = AcousticMaterial("wall", 0.2, 0.1)
CENTRE = np.array([3.5, 2.5, 1.5])


def scene_of(*objects):
    scene = Scene("test")
    for obj in objects:
        scene.add_acoustic_object(obj)
    return scene


def shoebox_leaves(materials=WALL):
    return AcousticObjectsLibrary.shoebox_room(7, 5, 3, materials=materials)


def quad(name, corners, material=WALL, role="boundary"):
    mesh = Mesh.from_vertices_and_faces([list(map(float, c)) for c in corners], [[0, 1, 2, 3]])
    return AcousticObject(mesh, material, name=name, role=role)


def corners_of(obj):
    mesh = obj.acoustic_geometry
    face = next(iter(mesh.faces()))
    return [mesh.vertex_coordinates(v) for v in mesh.face_vertices(face)]


def assert_outward(geometry, centre=CENTRE):
    for (vertices, _), area_normal, role in zip(
        geometry.faces_with_materials(), geometry.area_normals(), geometry.face_roles
    ):
        if role == "boundary":
            assert (vertices.mean(axis=0) - centre) @ area_normal > 0.0


def test_a_shoebox_welds_into_one_closed_room():
    g = RoomGeometry.from_scene(scene_of(*shoebox_leaves()))
    assert g.vertices.shape == (8, 3)
    assert len(g.faces) == 6
    assert g.is_closed
    assert g.volume == pytest.approx(105.0)
    assert g.surface_area == pytest.approx(142.0)
    assert set(g.face_roles) == {"boundary"}
    assert_outward(g)


def test_inconsistently_wound_surfaces_are_oriented():
    leaves = shoebox_leaves()
    leaves[0] = quad(leaves[0].name, corners_of(leaves[0])[::-1])
    leaves[3] = quad(leaves[3].name, corners_of(leaves[3])[::-1])
    g = RoomGeometry.from_scene(scene_of(*leaves))
    assert g.is_closed
    assert g.volume == pytest.approx(105.0)
    assert_outward(g)


def test_a_room_wound_inward_is_turned_outward():
    leaves = [quad(o.name, corners_of(o)[::-1]) for o in shoebox_leaves()]
    g = RoomGeometry.from_scene(scene_of(*leaves))
    assert g.volume == pytest.approx(105.0)
    assert_outward(g)


def test_a_missing_surface_warns_and_leaves_the_room_open():
    with pytest.warns(UserWarning, match="do not close a volume"):
        g = RoomGeometry.from_scene(scene_of(*shoebox_leaves()[:-1]))
    assert not g.is_closed
    assert g.open_edge is not None
    with pytest.raises(ValueError, match="volume"):
        g.volume


def test_a_t_junction_seam_still_closes():
    walls = [o for o in shoebox_leaves() if not o.name.endswith("floor")]
    floor_a = quad("floor_a", [[0, 0, 0], [0, 5, 0], [3.5, 5, 0], [3.5, 0, 0]])
    floor_b = quad("floor_b", [[3.5, 0, 0], [3.5, 5, 0], [7, 5, 0], [7, 0, 0]])
    g = RoomGeometry.from_scene(scene_of(*walls, floor_a, floor_b))
    assert g.is_closed
    assert g.volume == pytest.approx(105.0)


def test_near_coincident_vertices_weld():
    leaves = shoebox_leaves()
    ceiling = leaves[-1]
    leaves[-1] = quad(ceiling.name, [np.add(c, 1e-8) for c in corners_of(ceiling)])
    g = RoomGeometry.from_scene(scene_of(*leaves))
    assert g.is_closed
    assert g.vertices.shape == (8, 3)


def test_an_interior_closed_volume_is_a_hole():
    pillar = AcousticObject(Mesh.from_shape(Box(1, 1, 1, Frame([3.5, 2.5, 1.5]))), WALL, name="pillar")
    g = RoomGeometry.from_scene(scene_of(*shoebox_leaves(), pillar))
    assert g.is_closed
    assert g.volume == pytest.approx(104.0)
    pillar_faces = [i for i, name in enumerate(g.face_names) if name == "pillar"]
    for i in pillar_faces:
        vertices = g.faces_with_materials()[i][0]
        assert (vertices.mean(axis=0) - CENTRE) @ g.area_normals()[i] < 0.0


def test_an_obstacle_is_stored_on_both_sides():
    panel = quad("panel", [[3, 1, 0.5], [3, 1, 2.5], [3, 3, 2.5], [3, 3, 0.5]], role="obstacle")
    g = RoomGeometry.from_scene(scene_of(*shoebox_leaves(), panel))
    obstacle = [i for i, role in enumerate(g.face_roles) if role == "obstacle"]
    assert obstacle == [6, 7]
    xs = sorted(float(g.faces_with_materials()[i][0][:, 0].mean()) for i in obstacle)
    assert xs == pytest.approx([3.0 - OBSTACLE_OFFSET, 3.0 + OBSTACLE_OFFSET])
    normals = g.area_normals()[obstacle]
    assert normals[0] @ normals[1] < 0.0
    assert g.volume == pytest.approx(105.0)
    assert g.surface_area == pytest.approx(142.0 + 2 * 4.0)
    assert len(g.without_obstacles().faces) == 6


def test_materials_get_a_62_5_hz_band():
    graded = AcousticMaterial("graded", [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7], [0.05] * 7)
    g = RoomGeometry.from_scene(scene_of(*shoebox_leaves(graded)))
    assert len(SYNTHESIS_BANDS) == 8
    np.testing.assert_allclose(g.materials["graded"].absorption, [0.1, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    np.testing.assert_allclose(g.materials["graded"].scattering, [0.05] * 8)


def test_a_material_without_scattering_keeps_none():
    g = RoomGeometry.from_scene(scene_of(*shoebox_leaves(AcousticMaterial("bare", 0.2))))
    assert g.materials["bare"].scattering is None


def test_two_different_materials_with_one_name_raise():
    leaves = shoebox_leaves()
    leaves[0] = quad(leaves[0].name, corners_of(leaves[0]), AcousticMaterial("wall", 0.3, 0.1))
    with pytest.raises(ValueError, match="wall"):
        RoomGeometry.from_scene(scene_of(*leaves))


def test_coefficients_outside_0_1_raise():
    with pytest.raises(ValueError, match="outside"):
        RoomGeometry.from_scene(scene_of(*shoebox_leaves(AcousticMaterial("hot", 1.2, 0.1))))


def test_obstacle_polygons_are_kept_as_drawn():
    panel = quad("panel", [[3, 1, 0.5], [3, 1, 2.5], [3, 3, 2.5], [3, 3, 0.5]], role="obstacle")
    g = RoomGeometry.from_scene(scene_of(*shoebox_leaves(), panel))
    assert len(g.obstacle_faces) == 1
    assert g.obstacle_face_materials == ("wall",)
    np.testing.assert_allclose(g.vertices[list(g.obstacle_faces[0])][:, 0], 3.0)
    assert len(g.faces) - 2 * len(g.obstacle_faces) == 6
    assert g.without_obstacles().obstacle_faces == ()


def test_points_across_a_grid_line_weld():
    from compas_acoustics.room_geometry import _weld

    points = np.array([[1.000000499, 3.0, 0.0], [1.000000501, 3.0, 0.0], [5.0, 5.0, 5.0]])
    vertices, inverse = _weld(points, 1e-6)
    assert len(vertices) == 2 and inverse[0] == inverse[1] != inverse[2]


def test_welding_is_transitive():
    from compas_acoustics.room_geometry import _weld

    points = np.array([[0.0, 0, 0], [0.8e-6, 0, 0], [1.6e-6, 0, 0]])
    assert len(_weld(points, 1e-6)[0]) == 1


def test_from_mesh_matches_from_scene():
    leaves = shoebox_leaves()
    scene = RoomGeometry.from_scene(scene_of(*leaves))
    mesh = RoomGeometry.from_mesh(scene.vertices, scene.faces, scene.face_materials, scene.materials)
    assert mesh.is_closed and mesh.volume == pytest.approx(scene.volume)


def test_surfaces_volume_and_area_match_upstream_meshgeometry():
    panel = quad("panel", [[3, 1, 0.5], [3, 1, 2.5], [3, 3, 2.5], [3, 3, 0.5]], AcousticMaterial("felt", 0.6, 0.2), role="obstacle")
    g = RoomGeometry.from_scene(scene_of(*shoebox_leaves(), panel))
    surfaces = dict((material.name, area) for area, material in g.surfaces)
    assert surfaces == pytest.approx({"wall": 142.0, "felt": 8.0})
    assert g.surface_area == pytest.approx(150.0)
    assert g.volume == pytest.approx(105.0)


def test_from_mesh_names_obstacles_by_material():
    leaves = shoebox_leaves()
    panel = quad("panel", [[3, 1, 0.5], [3, 1, 2.5], [3, 3, 2.5], [3, 3, 0.5]], AcousticMaterial("felt", 0.6), role="obstacle")
    scene = RoomGeometry.from_scene(scene_of(*leaves, panel))
    drawn = [scene.faces[i] for i, role in enumerate(scene.face_roles) if role == "boundary"] + list(scene.obstacle_faces)
    names = [scene.face_materials[i] for i, role in enumerate(scene.face_roles) if role == "boundary"] + list(scene.obstacle_face_materials)
    mesh = RoomGeometry.from_mesh(scene.vertices, drawn, names, scene.materials, obstacles=("felt",))
    assert mesh.face_roles.count("obstacle") == 2
    assert mesh.volume == pytest.approx(105.0)
    with pytest.raises(ValueError, match="obstacles"):
        RoomGeometry.from_mesh(scene.vertices, drawn, names, scene.materials, obstacles=("glass",))


def triangulated(obj):
    corners = corners_of(obj)
    tri = Mesh.from_vertices_and_faces(corners, [[0, 1, 2], [0, 2, 3]])
    return AcousticObject(tri, obj.material, name=obj.name)


def test_merge_coplanar_joins_a_triangulated_room():
    leaves = [triangulated(o) for o in shoebox_leaves()]
    plain = RoomGeometry.from_scene(scene_of(*leaves), merge_coplanar=False)
    merged = RoomGeometry.from_scene(scene_of(*leaves), merge_coplanar=True)
    assert len(plain.faces) == 12
    assert len(merged.faces) == 6
    assert merged.is_closed
    assert merged.volume == pytest.approx(105.0)
    assert_outward(merged)


def test_from_scene_merges_coplanar_faces_by_default():
    import warnings

    leaves = [triangulated(o) for o in shoebox_leaves()]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        geometry = RoomGeometry.from_scene(scene_of(*leaves))
    assert len(geometry.faces) == 6
    assert geometry.mergeable_faces == 0


def test_a_triangulated_scene_simulates_merged_and_without_the_warning():
    import warnings

    from compas_acoustics import Receiver
    from compas_acoustics import Source
    from compas_acoustics.methods import Pyroomacoustics

    s = scene_of(*[triangulated(o) for o in shoebox_leaves()])
    s.add_source(Source("s", 1.5, 1.5, 1.6))
    s.add_receiver(Receiver("r", 4.0, 2.0, 1.5))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        rirs = s.simulate(Pyroomacoustics(image_source_order=2))
    assert len(rirs) == 1


def test_merge_coplanar_keeps_objects_apart():
    walls = [o for o in shoebox_leaves() if not o.name.endswith("floor")]
    floor_a = quad("floor_a", [[0, 0, 0], [0, 5, 0], [3.5, 5, 0], [3.5, 0, 0]])
    floor_b = quad("floor_b", [[3.5, 0, 0], [3.5, 5, 0], [7, 5, 0], [7, 0, 0]])
    merged = RoomGeometry.from_scene(scene_of(*walls, floor_a, floor_b), merge_coplanar=True)
    assert {"floor_a", "floor_b"} <= set(merged.face_names)


def test_mergeable_faces_are_counted_and_warned_about():
    leaves = [triangulated(o) for o in shoebox_leaves()]
    with pytest.warns(UserWarning, match="merge_coplanar=True"):
        plain = RoomGeometry.from_scene(scene_of(*leaves), merge_coplanar=False)
    assert plain.mergeable_faces == 6
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        merged = RoomGeometry.from_scene(scene_of(*leaves), merge_coplanar=True)
        assert RoomGeometry.from_scene(scene_of(*shoebox_leaves())).mergeable_faces == 0
    assert merged.mergeable_faces == 0


def test_an_obj_file_reads_one_object_per_group(tmp_path):
    obj = tmp_path / "box.obj"
    lines = ["v 0 0 0", "v 4 0 0", "v 4 3 0", "v 0 3 0", "v 0 0 2.5", "v 4 0 2.5", "v 4 3 2.5", "v 0 3 2.5",
             "g walls", "f 1 2 6 5", "f 2 3 7 6", "g floor", "f 1 4 3 2", "g ceiling", "f 5 6 7 8",
             "g walls", "f 3 4 8 7", "f 4 1 5 8"]  # a recurring group, as Rhino writes them
    obj.write_text("\n".join(lines) + "\n")
    carpet = AcousticMaterial("carpet", 0.3, 0.1)
    room = AcousticObject.from_obj(str(obj), {"floor": carpet, "ceiling": WALL, "walls": WALL})
    assert [(o.name, o.acoustic_geometry.number_of_faces()) for o in room] == [
        ("box_walls", 4), ("box_floor", 1), ("box_ceiling", 1)]
    assert room[1].material is carpet
    g = RoomGeometry.from_scene(scene_of(*room))
    assert g.is_closed and g.volume == pytest.approx(30.0)
    with pytest.raises(ValueError, match="walls"):
        AcousticObject.from_obj(str(obj), {"floor": carpet, "ceiling": WALL})
    furniture = AcousticObject.from_obj(str(obj), WALL, roles="obstacle")
    assert {o.role for o in furniture} == {"obstacle"}
    mixed = AcousticObject.from_obj(str(obj), WALL, roles={"floor": "obstacle"})
    assert [o.role for o in mixed] == ["boundary", "obstacle", "boundary"]


def test_assigning_scene_materials_by_surface_name_raises():
    scene = scene_of(*shoebox_leaves())
    assert list(scene.acoustic_materials) == ["wall"]
    with pytest.raises(AttributeError, match="shoebox_room"):
        scene.acoustic_materials = {"floor": WALL}
