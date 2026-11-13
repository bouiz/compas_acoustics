"""Rooms of a prepared mesh: obstacles are the two-sided walls only, the accounting counts what the air reaches,
``is_closed`` is acoustic, and unreached faces always stay walls."""

import warnings
from dataclasses import replace

import numpy as np
import pytest
from test_geometry_prepare import _box
from test_geometry_prepare import _faces_of
from test_geometry_prepare import _room
from test_geometry_prepare import _Scene

import compas_acoustics as ca
from compas_acoustics.geometry import FACE_CLASS_SHELL
from compas_acoustics.geometry import PrepareOptions
from compas_acoustics.geometry import ReportOptions
from compas_acoustics.geometry import Seed
from compas_acoustics.geometry import analyze
from compas_acoustics.geometry import obstacle_mask
from compas_acoustics.geometry import prepare_mesh
from compas_acoustics.geometry import standing_objects
from compas_acoustics.room.geometry import CLOSED_LEAK_TOLERANCE

WALL = ca.AcousticMaterial("wall", 0.2, 0.1)
SEEDS = [[1.0, 1.0, 1.5], [4.5, 3.0, 1.2], [4.5, 1.0, 1.6], [1.0, 3.2, 0.5]]
T = 0.2
# Floor and ceiling over the whole plan, the long walls between them, the short walls between the long ones: each a
# closed 0.2 m slab around a 6 x 4 x 3 m room. The air reaches each slab's inner face only.
SLABS = (
    ("floor", (-T, -T, -T, 6 + T, 4 + T, 0.0)),
    ("ceiling", (-T, -T, 3.0, 6 + T, 4 + T, 3 + T)),
    ("south", (-T, -T, 0.0, 6 + T, 0.0, 3.0)),
    ("north", (-T, 4.0, 0.0, 6 + T, 4 + T, 3.0)),
    ("west", (-T, 0.0, 0.0, 0.0, 4.0, 3.0)),
    ("east", (6.0, 0.0, 0.0, 6 + T, 4.0, 3.0)),
)
# the inner faces whole: floor and ceiling 6.4 x 4.4, long walls 6.4 x 3, short walls 4 x 3 (the parts under the
# adjoining slabs included, the known limit of merged walls)
SLAB_REACHED_AREA = 2 * 6.4 * 4.4 + 2 * 6.4 * 3 + 2 * 4 * 3


def _prepare(scene, seeds=SEEDS, **options):
    return prepare_mesh(scene.v, scene.f, scene.ranges, dict.fromkeys(range(len(scene.f)), "wall"), None, seeds, PrepareOptions(**options))


def _geometry(prepared, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return ca.RoomGeometry.from_prepared(prepared, {"wall": WALL}, **kwargs)


def _shell():
    scene = _Scene()
    _room(scene)
    return scene


def _slabs(ceiling=True):
    scene = _Scene()
    for name, bounds in SLABS:
        if ceiling or name != "ceiling":
            scene.add(name, *_box(*bounds))
    return scene


def _column(size=0.5):
    scene = _shell()
    h = size / 2.0
    scene.add("column", *_box(3 - h, 2 - h, 0.0, 3 + h, 2 + h, 3.0))
    return scene


def _edges(geometry):
    from compas_acoustics.hybrid.diffraction import diffraction_edges

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return diffraction_edges(geometry)


# ---------------------------------------------------------------------------------------------------------------
# Roles, accounting, closedness, volume
# ---------------------------------------------------------------------------------------------------------------


def test_a_thin_shell():
    prepared = _prepare(_shell())
    geometry = _geometry(prepared)
    assert geometry.face_roles.count("boundary") == 6 and not geometry.obstacle_faces and len(geometry.faces) == 6
    assert geometry.is_closed and geometry.encloses and geometry.air_volume is None and geometry.open_area == 0.0
    assert geometry.surface_area == pytest.approx(108.0) and geometry.volume == pytest.approx(72.0)


def test_six_closed_slabs_are_one_sided_boundary_and_the_room_is_acoustically_closed():
    prepared = _prepare(_slabs())
    assert prepared.stats["leak_fraction"] == 0.0
    assert not obstacle_mask(prepared).any()
    geometry = _geometry(prepared)
    # every wall is boundary and stored once: 6 walls per slab
    assert geometry.face_roles.count("boundary") == 36 and not geometry.obstacle_faces and len(geometry.faces) == 36
    assert geometry.is_closed and geometry.open_area == 0.0
    # watertight (every slab is closed) but the outer skins are never reached: the volume is the walk's
    assert not geometry.encloses and geometry.air_volume == pytest.approx(prepared.air_volume)
    assert geometry.has_volume and geometry.volume == pytest.approx(prepared.air_volume)
    assert prepared.air_volume_method == "mean_free_path" and 60.0 < geometry.volume < 90.0
    assert geometry.surface_area == pytest.approx(SLAB_REACHED_AREA)
    assert geometry.area_normals().shape == (6, 3)
    assert sum(area for area, _ in geometry.surfaces) == pytest.approx(SLAB_REACHED_AREA)
    assert _edges(geometry) == []  # the inner corners open pi / 2


def test_the_closedness_reads_the_stats_without_the_visibility():
    prepared = _prepare(_slabs())
    geometry = _geometry(replace(prepared, visibility=None))
    assert geometry.is_closed and geometry.open_area == 0.0 and geometry.volume == pytest.approx(prepared.air_volume)
    leaking = _geometry(replace(prepared, visibility=None, stats={**prepared.stats, "leak_fraction": 10 * CLOSED_LEAK_TOLERANCE}))
    assert not leaking.is_closed and leaking.open_area > 0.0
    within = _geometry(replace(prepared, visibility=None, stats={**prepared.stats, "leak_fraction": CLOSED_LEAK_TOLERANCE}))
    assert within.is_closed and within.open_area == 0.0


def test_five_slabs_without_a_ceiling_leak():
    prepared = _prepare(_slabs(ceiling=False))
    geometry = _geometry(prepared)
    assert prepared.stats["leak_fraction"] > CLOSED_LEAK_TOLERANCE
    assert not geometry.is_closed and not geometry.encloses and geometry.open_area > 0.0
    walls = SLAB_REACHED_AREA - 6.4 * 4.4
    assert geometry.surface_area == pytest.approx(walls + geometry.open_area)
    assert any(material.name == "open" for _, material in geometry.surfaces)


def test_a_column_is_boundary_and_leaves_the_volume():
    prepared = _prepare(_column())
    column = _faces_of(prepared, "column")
    assert standing_objects(prepared)[column].all() and not standing_objects(prepared)[_faces_of(prepared, "room")].any()
    geometry = _geometry(prepared)
    assert geometry.face_roles.count("boundary") == 12 and not geometry.obstacle_faces and len(geometry.faces) == 12
    # its ends on the floor and ceiling are never reached, but pressed against reached walls: the divergence holds
    assert geometry.is_closed and geometry.encloses and geometry.air_volume is None
    assert geometry.volume == pytest.approx(72.0 - 0.75)
    # the ends are out of the accounting; the floor and ceiling under them count whole (merged walls)
    assert geometry.surface_area == pytest.approx(108.0 + 4 * 0.5 * 3.0)
    edges = _edges(geometry)
    assert len(edges) == 4 and all(e.component == -1 and abs(e.axis[2]) > 0.99 for e in edges)
    assert sorted((round(float(e.start[0]), 6), round(float(e.start[1]), 6)) for e in edges) == [(2.75, 1.75), (2.75, 2.25), (3.25, 1.75), (3.25, 2.25)]


def test_the_inside_test_of_a_prepared_room():
    column = _geometry(_prepare(_column()), check_inside=True)
    assert column.is_inside([1.0, 1.0, 1.0]) and not column.is_inside([3.0, 2.0, 1.0]) and not column.is_inside([8.0, 2.0, 1.0])
    slabs = _geometry(_prepare(_slabs()), check_inside=True)
    assert slabs.is_inside([1.0, 1.0, 1.0]) and not slabs.is_inside([3.0, 2.0, -0.1]) and not slabs.is_inside([-0.1, 2.0, 1.0])


def test_the_enclosure_alone_keeps_the_air_volume():
    geometry = _geometry(_prepare(_slabs()))
    enclosure = geometry.without_obstacles()
    assert enclosure.is_closed and enclosure.volume == geometry.volume and enclosure.surface_area == pytest.approx(geometry.surface_area)


def test_a_two_sided_panel_is_the_only_obstacle_and_counts_both_sides():
    scene = _shell()
    scene.add("panel", [[3, 1, 0.5], [3, 3, 0.5], [3, 3, 2.5], [3, 1, 2.5]], [[0, 1, 2], [0, 2, 3]])
    prepared = _prepare(scene, [[1.5, 2, 1.5], [4.5, 2, 1.5]])
    assert np.array_equal(obstacle_mask(prepared), prepared.two_sided)
    assert prepared.two_sided[_faces_of(prepared, "panel")].all() and not prepared.two_sided[_faces_of(prepared, "room")].any()
    geometry = _geometry(prepared)
    assert len(geometry.obstacle_faces) == 1 and geometry.face_roles.count("obstacle") == 2 and geometry.face_roles.count("boundary") == 6
    assert geometry.drawn_sides.side_reached[0].all()
    assert geometry.is_closed and geometry.encloses and geometry.volume == pytest.approx(72.0)
    assert geometry.surface_area == pytest.approx(108.0 + 2 * 4.0)


def test_a_declared_closed_obstacle_counts_its_outside_only():
    scene = ca.Scene("cabinet")
    scene.add_acoustic_objects(ca.AcousticObjectsLibrary.shoebox_room(6, 4, 3, materials=WALL))
    from compas.datastructures import Mesh

    v, f = _box(2.0, 1.5, 0.0, 2.8, 2.1, 0.9)
    quads = [[f[2 * k][0], f[2 * k][1], f[2 * k][2], f[2 * k + 1][2]] for k in range(6)]
    scene.add_acoustic_object(ca.AcousticObject(Mesh.from_vertices_and_faces(v, quads), WALL, name="cabinet", role="obstacle"))
    scene.add_source(ca.Source("s", *SEEDS[0]))
    for k, p in enumerate(SEEDS[1:]):
        scene.add_receiver(ca.Receiver(f"r{k}", *p))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        geometry = ca.RoomGeometry.from_scene(scene, prepare=PrepareOptions())
    # forced two-sided, so an obstacle stored twice; its inside is never reached, and its base lies on the floor
    assert len(geometry.obstacle_faces) == 6 and geometry.face_roles.count("boundary") == 6
    assert geometry.volume == pytest.approx(72.0 - 0.8 * 0.6 * 0.9)
    assert geometry.surface_area == pytest.approx(108.0 + 0.8 * 0.6 + 2 * (0.8 + 0.6) * 0.9)


# ---------------------------------------------------------------------------------------------------------------
# The standing-object label and the face classes
# ---------------------------------------------------------------------------------------------------------------


def test_the_standing_label_column_slabs_and_thick_column():
    assert not standing_objects(_prepare(_slabs())).any()
    for size in (0.5, 1.5):  # 1.5 m is thicker than interior_object_max_depth: a closed solid is judged whole
        prepared = _prepare(_column(size))
        column = _faces_of(prepared, "column")
        assert standing_objects(prepared)[column].all(), size
        assert not standing_objects(prepared)[_faces_of(prepared, "room")].any(), size


def test_a_wall_between_rooms_that_share_no_air_is_shell():
    scene = _Scene()
    scene.add("a", *_box(0, 0, 0, 3, 4, 3))
    scene.add("b", *_box(3.2, 0, 0, 6, 4, 3))
    prepared = _prepare(scene, [[1.5, 2, 1.5], [4.5, 2, 1.5]])
    seen = (prepared.air_hits + prepared.other_hits) > 0
    assert (prepared.face_class[seen] == FACE_CLASS_SHELL).all()
    assert not standing_objects(prepared).any()


# ---------------------------------------------------------------------------------------------------------------
# Unreached faces stay walls; unreached objects are listed
# ---------------------------------------------------------------------------------------------------------------


def test_prepare_mesh_no_longer_takes_drop_unreached_faces():
    with pytest.raises(TypeError):
        PrepareOptions(drop_unreached_faces=True)
    prepared = _prepare(_slabs())
    assert "unreached_faces_dropped" not in prepared.stats
    assert sorted(int(f) for w in prepared.walls for f in w.faces) == list(range(len(prepared.faces)))


def test_objects_the_air_never_reaches_are_listed():
    scene = _shell()
    scene.add("buried", *_box(7.0, 1.0, 0.0, 8.0, 2.0, 1.0))  # outside the room, no seed near it
    scene.add("buried too", *_box(9.0, 1.0, 0.0, 10.0, 2.0, 1.0))
    seeds = [Seed("s", "source", [2.0, 2.0, 1.5]), Seed("r", "receiver", [4.0, 2.0, 1.5])]
    prepared = _prepare(scene, [s.position for s in seeds])
    assert prepared.unreached_objects() == ["buried", "buried too"]
    report = analyze(prepared, seeds, [("s", "r")])
    (issue,) = [i for i in report.issues if i.code == "skipped_faces"]
    assert issue.object_ids == ["buried", "buried too"] and "image-source search cheaper" in issue.detail
    limited = analyze(prepared, seeds, [("s", "r")], ReportOptions(max_listed_items=1))
    assert [i.object_ids for i in limited.issues if i.code == "skipped_faces"] == [["buried"]]
    # they stay walls of the room, out of its accounting
    geometry = _geometry(prepared)
    assert len(geometry.faces) == 18 and geometry.surface_area == pytest.approx(108.0)
    assert not _prepare(_slabs()).unreached_objects()


# ---------------------------------------------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("build", [_slabs, _column], ids=["slabs", "column"])
def test_mono_runs_in_the_room(build):
    source, receiver = [1.0, 1.0, 1.5], [4.5, 3.0, 1.2]
    geometry = _geometry(_prepare(build(), [source, receiver]))
    scene = ca.Scene("prepared")
    scene.add_source(ca.Source("s", *source))
    scene.add_receiver(ca.Receiver("r", *receiver))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rirs = scene.simulate(ca.Mono(image_source_order=2, ray_tracing=False, diffraction=True), geometry=geometry)
    rir = next(iter(rirs.values())).denormalized()
    assert np.isfinite(rir).all() and np.abs(rir).max() > 0.0
