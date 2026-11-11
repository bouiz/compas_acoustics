"""RoomGeometry from a prepared mesh: orientation and two-sided faces kept, open models simulated, points not dropped."""

import warnings

import numpy as np
import pytest
from test_geometry_prepare import _box
from test_geometry_prepare import _room
from test_geometry_prepare import _Scene

import compas_acoustics as ca
from compas_acoustics.geometry import PrepareOptions
from compas_acoustics.geometry import prepare_mesh

WALL = ca.AcousticMaterial("wall", 0.2, 0.1)
PANEL = ca.AcousticMaterial("panel", 0.1, 0.3)


def _prepared(scene, seeds, materials=None, scattering=None, **options):
    materials = materials or {i: "wall" for i in range(len(scene.f))}
    return prepare_mesh(scene.v, scene.f, scene.ranges, materials, scattering, seeds, PrepareOptions(**options))


def _ca_scene(source, receivers):
    scene = ca.Scene("prepared")
    scene.add_source(ca.Source("s", *source))
    for k, r in enumerate(receivers):
        scene.add_receiver(ca.Receiver(f"r{k}", *r))
    return scene


def _responses(scene, geometry, **method):
    method = dict(image_source_order=2, ray_tracing=False, **method)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rirs = scene.simulate(ca.Mono(**method), geometry=geometry)
    return [rirs[ca.simulation_methods.SimulationContext.key(scene.sources[0], r)].denormalized() for r in scene.receivers]


def test_a_prepared_box_simulates_as_the_drawn_one():
    source, receiver = [1.5, 1.2, 1.4], [4.2, 2.9, 1.6]
    scene = _ca_scene(source, [receiver])
    scene.add_acoustic_objects(ca.AcousticObjectsLibrary.shoebox_room(6, 4, 3, materials=WALL))
    drawn = _responses(scene, None)[0]

    box = _Scene()
    _room(box, inward=True)  # wound the wrong way: the preparation flips it
    geometry = ca.RoomGeometry.from_prepared(_prepared(box, [source, receiver]), {"wall": WALL})
    assert geometry.is_closed and geometry.air_volume is None
    assert len(geometry.faces) == 6 and geometry.volume == pytest.approx(72.0)
    np.testing.assert_allclose(_responses(scene, geometry)[0], drawn, atol=1e-6 * np.abs(drawn).max())  # float32 corners, another vertex order


def test_an_open_model_runs_with_the_prepared_air_volume():
    box = _Scene()
    v, f = _box(0, 0, 0, 6, 4, 3)
    del f[2:4]  # the whole ceiling quad
    box.add("room", v, f)
    source, receiver = [2.0, 2.0, 1.5], [4.5, 1.5, 1.2]
    prepared = _prepared(box, [source, receiver])
    geometry = ca.RoomGeometry.from_prepared(prepared, {"wall": WALL})
    assert not geometry.is_closed
    # the walk's volume, scaled for the mean free path that ends at the openings too
    assert geometry.open_area > 0 and prepared.air_volume > 0
    assert geometry.volume == pytest.approx(prepared.air_volume * geometry.surface_area / (geometry.surface_area - geometry.open_area))
    given = ca.RoomGeometry.from_prepared(prepared, {"wall": WALL}, air_volume=50.0)
    assert given.volume == 50.0
    rir = _responses(_ca_scene(source, [receiver]), geometry)[0]
    assert np.isfinite(rir).all() and np.abs(rir).max() > 0


def test_points_outside_are_kept_without_the_inside_test():
    box = _Scene()
    _room(box)
    outside = [8.0, 2.0, 1.5]
    prepared = _prepared(box, [[2.0, 2.0, 1.5], outside])
    scene = _ca_scene([2.0, 2.0, 1.5], [outside])
    kept = ca.RoomGeometry.from_prepared(prepared, {"wall": WALL})
    tested = ca.RoomGeometry.from_prepared(prepared, {"wall": WALL}, check_inside=True)
    assert kept.is_inside(outside) and not tested.is_inside(outside)
    context = ca.simulation_methods.SimulationContext.from_scene(scene, kept)
    assert [r.name for r in context.receivers] == ["r0"] and not context.outside


def test_a_detected_panel_reflects_like_a_declared_obstacle():
    source, receivers = [1.5, 2.0, 1.5], [[4.5, 2.0, 1.5], [1.0, 3.0, 2.0]]
    box = _Scene()
    _room(box)
    panel = [[3, 1, 0.5], [3, 3, 0.5], [3, 3, 2.5], [3, 1, 2.5]]
    box.add("panel", panel, [[0, 1, 2], [0, 2, 3]])
    materials = {i: ("panel" if i >= 12 else "wall") for i in range(len(box.f))}
    prepared = _prepared(box, [source] + receivers, materials)
    assert prepared.two_sided[prepared.face_object == prepared.objects.index("panel")].all()
    geometry = ca.RoomGeometry.from_prepared(prepared, {"wall": WALL, "panel": PANEL})
    assert geometry.face_roles.count("obstacle") == 2  # the merged panel, twice

    scene = _ca_scene(source, receivers)
    scene.add_acoustic_objects(ca.AcousticObjectsLibrary.shoebox_room(6, 4, 3, materials=WALL))
    scene.add_acoustic_object(ca.AcousticObject(_panel_mesh(panel), PANEL, name="panel", role="obstacle"))
    for ours, theirs in zip(_responses(scene, geometry), _responses(scene, None)):
        n = min(len(ours), len(theirs))
        np.testing.assert_allclose(ours[:n], theirs[:n], atol=1e-6 * np.abs(theirs).max())


def _panel_mesh(points):
    from compas.datastructures import Mesh

    return Mesh.from_vertices_and_faces(points, [[0, 1, 2, 3]])


def test_a_scattering_override_makes_its_own_material():
    box = _Scene()
    _room(box)
    prepared = _prepared(box, [[3, 2, 1.5]], scattering={0: 0.4, 1: 0.4})
    geometry = ca.RoomGeometry.from_prepared(prepared, {"wall": WALL})
    assert set(geometry.materials) == {"wall", "wall (s=0.4)"}
    np.testing.assert_array_equal(geometry.materials["wall (s=0.4)"].scattering, 0.4)
    np.testing.assert_array_equal(geometry.materials["wall (s=0.4)"].absorption, geometry.materials["wall"].absorption)


def test_from_scene_with_prepare_keeps_obstacles_two_sided_and_closes_loose_walls():
    scene = _ca_scene([1.5, 2.0, 1.5], [[4.5, 2.0, 1.5]])
    scene.add_acoustic_objects(ca.AcousticObjectsLibrary.shoebox_room(6, 4, 3, materials=WALL))  # six loose faces
    # a one-sided-looking box the user declares an obstacle
    scene.add_acoustic_object(ca.AcousticObject(_panel_mesh([[3, 3.5, 0.5], [3.5, 3.5, 0.5], [3.5, 3.5, 1.0], [3, 3.5, 1.0]]), PANEL, name="sign", role="obstacle"))
    geometry = ca.RoomGeometry.from_scene(scene, prepare=PrepareOptions(detect_two_sided=False))
    assert geometry.is_closed and geometry.volume == pytest.approx(72.0)
    assert geometry.face_roles.count("obstacle") == 2


def test_unknown_or_unresolved_materials_raise():
    box = _Scene()
    _room(box)
    with pytest.raises(ValueError, match="no entry"):
        ca.RoomGeometry.from_prepared(_prepared(box, [[3, 2, 1.5]]), {"other": WALL})
    with pytest.raises(ValueError, match="must be names"):
        ca.RoomGeometry.from_prepared(_prepared(box, [[3, 2, 1.5]], materials={i: {"coeffs": [0.1]} for i in range(12)}), {})
