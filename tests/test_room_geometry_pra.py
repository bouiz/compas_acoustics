import numpy as np
import pyroomacoustics as pra
import pytest
from compas.datastructures import Mesh

from compas_acoustics import AcousticMaterial
from compas_acoustics import AcousticObject
from compas_acoustics import AcousticObjectsLibrary
from compas_acoustics import Scene
from compas_acoustics.room.geometry import RoomGeometry

WALL = AcousticMaterial("wall", 0.2, 0.1)


def geometry(materials=WALL, extra=()):
    scene = Scene("test")
    scene.add_acoustic_objects(AcousticObjectsLibrary.shoebox_room(7, 5, 3, materials=materials))
    for obj in extra:
        scene.add_acoustic_object(obj)
    return RoomGeometry.from_scene(scene)


def panel():
    mesh = Mesh.from_vertices_and_faces([[3, 1, 0.5], [3, 1, 2.5], [3, 3, 2.5], [3, 3, 0.5]], [[0, 1, 2, 3]])
    return AcousticObject(mesh, WALL, name="panel", role="obstacle")


def test_builds_and_computes_at_48_khz_with_air_absorption():
    room = geometry().to_pra(48000, 3, air_absorption=True)
    room.add_source([1.5, 1.5, 1.6])
    room.add_microphone(np.array([[5.0], [3.0], [1.5]]))
    room.image_source_model()
    room.compute_rir()
    assert room.fs == 48000
    np.testing.assert_allclose(room.octave_bands.centers[:2], [62.5, 125.0])
    assert room.sources[0].damping.shape[0] == 9
    assert len(room.rir[0][0]) > 0


def test_the_octave_base_frequency_is_restored():
    before = pra.constants.get("octave_bands_base_freq")
    geometry().to_pra(48000, 1)
    assert pra.constants.get("octave_bands_base_freq") == before


def test_the_octave_base_frequency_is_restored_when_building_fails(monkeypatch):
    before = pra.constants.get("octave_bands_base_freq")

    def broken(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(pra, "Room", broken)
    with pytest.raises(ValueError, match="boom"):
        geometry().to_pra(48000, 1)
    assert pra.constants.get("octave_bands_base_freq") == before


def test_walls_are_named_after_their_objects():
    g = geometry()
    room = g.to_pra(48000, 1)
    assert [w.name for w in room.walls] == [f"{name}_{i}" for i, name in enumerate(g.face_names)]


def test_a_material_without_scattering_works_for_image_sources():
    room = geometry(AcousticMaterial("bare", 0.2)).to_pra(48000, 2)
    room.add_source([1.5, 1.5, 1.6])
    room.add_microphone(np.array([[5.0], [3.0], [1.5]]))
    room.image_source_model()
    assert room.sources[0].images.shape[1] > 1


def test_a_material_without_scattering_scatters_the_default_in_every_band():
    from compas_acoustics.room.geometry import DEFAULT_SCATTERING

    room = geometry(AcousticMaterial("bare", 0.2)).to_pra(48000, 2, ray_tracing=True)
    for wall in room.walls:
        np.testing.assert_allclose(wall.scatter, DEFAULT_SCATTERING)


def test_given_scattering_is_used_as_it_is():
    room = geometry(AcousticMaterial("rough", 0.2, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.6])).to_pra(48000, 2, ray_tracing=True)
    np.testing.assert_allclose(room.walls[0].scatter[[1, 4]], [0.1, 0.4])


@pytest.mark.parametrize("fs", [16000, 24000, 44100, 48000, 96000])
def test_the_top_band_is_held_at_every_rate(fs):
    steep = AcousticMaterial("steep", [0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 0.95], 0.1)
    room = geometry(steep).to_pra(fs, 1)
    for wall in room.walls:
        assert float(np.max(wall.absorption)) == pytest.approx(0.95)


def test_points_near_an_obstacle_are_inside():
    room = geometry(extra=[panel()]).to_pra(48000, 1)
    points = [[3.0, y, z] for y in (0.9, 1.0, 1.1, 2.9, 3.0, 3.1) for z in (0.4, 0.5, 0.6, 2.4, 2.5, 2.6)]
    assert all(room.is_inside(np.array(p)) for p in points)


def test_enclosure_only_leaves_obstacles_out():
    room = geometry(extra=[panel()]).to_pra(48000, 1, enclosure_only=True)
    assert len(room.walls) == 6


def pillar(role="obstacle"):
    from compas.geometry import Box
    from compas.geometry import Frame

    return AcousticObject(Mesh.from_shape(Box(1, 1, 3, Frame([3.5, 2.5, 1.5]))), WALL, name="pillar", role=role)


def test_a_point_inside_a_closed_obstacle_is_outside_the_room():
    g = geometry(extra=[pillar()])
    assert not g.is_inside([3.5, 2.5, 1.5])
    assert g.is_inside([1.0, 1.0, 1.0])
    assert not g.is_inside([20.0, 1.0, 1.0])
    room = g.to_pra(48000, 1)
    assert not room.is_inside(np.array([3.5, 2.5, 1.5])) and room.is_inside(np.array([1.0, 1.0, 1.0]))


def test_a_receiver_in_a_pillar_is_left_out_with_a_warning():
    from compas_acoustics import Receiver
    from compas_acoustics import Source
    from compas_acoustics.simulation_methods import Mono

    scene = Scene("pillar")
    scene.add_acoustic_objects(AcousticObjectsLibrary.shoebox_room(7, 5, 3, materials=WALL))
    scene.add_acoustic_object(pillar())
    scene.add_source(Source("s", 1, 1, 1.5))
    scene.add_receiver(Receiver("in_pillar", 3.5, 2.5, 1.5))
    scene.add_receiver(Receiver("free", 6, 4, 1.5))
    with pytest.warns(UserWarning, match="in_pillar"):
        rirs = scene.simulate(Mono(image_source_order=2, ray_tracing=False))
    assert [ir.receiver.name for ir in rirs.values()] == ["free"]


def test_an_open_panel_does_not_make_points_outside():
    g = geometry(extra=[panel()])
    assert g.is_inside([2.0, 2.0, 1.5]) and g.is_inside([4.0, 2.0, 1.5])


def test_a_scene_of_obstacles_only_is_an_open_field():
    from compas_acoustics import Receiver
    from compas_acoustics import Source
    from compas_acoustics.simulation_methods import Mono

    scene = Scene("field")
    for name, corners in (("a", [[0, 0, 0], [4, 0, 0], [4, 0, 3], [0, 0, 3]]), ("b", [[0, 0, 0], [0, 4, 0], [0, 4, 3], [0, 0, 3]])):
        scene.add_acoustic_object(AcousticObject(Mesh.from_vertices_and_faces(corners, [[0, 1, 2, 3]]), WALL, name=name, role="obstacle"))
    scene.add_source(Source("s", 1, 1, 1.5))
    scene.add_receiver(Receiver("r", 3, 2, 1.5))
    with pytest.warns(UserWarning, match="no boundary surfaces"):
        rirs = scene.simulate(Mono(image_source_order=2, ray_tracing=False))
    assert len(rirs) == 1


def test_too_few_walls_for_pyroomacoustics_is_a_clear_error():
    scene = Scene("ground")
    ground = Mesh.from_vertices_and_faces([[0, 0, 0], [10, 0, 0], [10, 10, 0], [0, 10, 0]], [[0, 1, 2, 3]])
    scene.add_acoustic_object(AcousticObject(ground, WALL, name="ground"))
    scene.add_acoustic_object(panel())
    with pytest.warns(UserWarning):
        g = RoomGeometry.from_scene(scene)
    with pytest.raises(ValueError, match="at least four walls"):
        g.to_pra(48000, 1)


def test_a_medium_sets_the_speed_of_sound_and_the_air():
    from compas_acoustics import AirConditions

    air = AirConditions(temperature=0.0, humidity=30.0)
    room = geometry().to_pra(48000, 1, air_absorption=True, air=air)
    assert room.c == pytest.approx(air.speed_of_sound)
    centres = [float(f) for f in room.octave_bands.centers]
    np.testing.assert_allclose(room.air_absorption, air.air_attenuation(centres), rtol=1e-12)


def test_a_medium_without_air_absorption_only_sets_the_speed():
    from compas_acoustics import AirConditions

    room = geometry().to_pra(48000, 1, air_absorption=False, air=AirConditions(temperature=0.0))
    assert room.c == pytest.approx(AirConditions(temperature=0.0).speed_of_sound)
    assert not room.simulator_state["air_abs_needed"]


def test_without_a_medium_the_room_is_in_the_default_medium():
    # Not pyroomacoustics' own air table (19.3 C, 0.3 % from 343 m/s), which the hybrid never used
    from compas_acoustics import AirConditions

    room = geometry().to_pra(48000, 1, air_absorption=True)
    assert room.c == pytest.approx(AirConditions().speed_of_sound)
    centres = [float(f) for f in room.octave_bands.centers]
    np.testing.assert_allclose(room.air_absorption, AirConditions().air_attenuation(centres), rtol=1e-12)


def test_a_warped_wall_simulates():
    from compas_acoustics import Receiver
    from compas_acoustics import Source
    from compas_acoustics.simulation_methods import Mono

    scene = Scene("warped")
    for obj in AcousticObjectsLibrary.shoebox_room(7, 5, 3, materials=WALL):
        mesh = obj.acoustic_geometry
        face = next(iter(mesh.faces()))
        corners = [np.add(mesh.vertex_coordinates(v), [0, 0, 2e-3]) if np.allclose(mesh.vertex_coordinates(v), [0, 0, 3])
                   else mesh.vertex_coordinates(v) for v in mesh.face_vertices(face)]
        mesh = Mesh.from_vertices_and_faces([list(map(float, c)) for c in corners], [[0, 1, 2, 3]])
        scene.add_acoustic_object(AcousticObject(mesh, WALL, name=obj.name))
    scene.add_source(Source("s", 1.5, 1.5, 1.6))
    scene.add_receiver(Receiver("r", 4.0, 2.0, 1.5))
    rirs = scene.simulate(Mono(ray_tracing=False, image_source_order=2))
    assert len(rirs) == 1


def test_wall_factory_errors_are_value_errors(monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError("corners are not coplanar")

    g = geometry()
    monkeypatch.setattr(pra, "wall_factory", broken)
    with pytest.raises(ValueError, match="face 0") as raised:
        g.to_pra(48000, 1)
    assert "corners are not coplanar" in str(raised.value)
    assert pra.constants.get("octave_bands_base_freq") == 125.0
