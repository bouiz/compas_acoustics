import numpy as np
import pytest

from compas_acoustics import AcousticMaterial
from compas_acoustics import AcousticObjectsLibrary
from compas_acoustics import ImpulseResponse
from compas_acoustics import Receiver
from compas_acoustics import RayField
from compas_acoustics import Scene
from compas_acoustics import Source
from compas_acoustics.room.pra import source_to_pra


def shoebox_scene():
    scene = Scene("test")
    scene.add_acoustic_objects(AcousticObjectsLibrary.shoebox_room(7, 5, 3, materials=AcousticMaterial("wall", 0.2, 0.1)))
    scene.add_source(Source("src", 1.5, 1.5, 1.6))
    scene.add_receiver(Receiver("rec", 5, 3, 1.5))
    return scene


def response(scene, data):
    return ImpulseResponse("k", scene.sources[0], scene.receivers[0], data, 48000)


def test_trace_rays_runs_on_room_geometry():
    field = RayField.from_scene(shoebox_scene(), n_rays=20, max_reflection_order=3)
    assert len(field.rays) == 20


def test_the_ray_field_seed_is_true_42_by_default_false_unseeded_or_an_int():
    scene = Scene("scattering")
    scene.add_acoustic_objects(AcousticObjectsLibrary.shoebox_room(7, 5, 3, materials=AcousticMaterial("wall", 0.2, 0.5)))
    scene.add_source(Source("src", 1.5, 1.5, 1.6))

    def scattered(**kwargs):
        field = RayField.from_scene(scene, n_rays=5, max_reflection_order=3, include_scattering=True, **kwargs)
        return np.concatenate([np.asarray(ray.direction, dtype=float) for ray in field.scattered_rays])

    default = scattered()
    np.testing.assert_array_equal(scattered(), default)
    np.testing.assert_array_equal(scattered(random_seed=True), default)
    np.testing.assert_array_equal(scattered(random_seed=42), default)
    assert not np.array_equal(scattered(random_seed=7), default)
    unseeded = scattered(random_seed=False)
    assert not np.array_equal(unseeded, default) and not np.array_equal(scattered(random_seed=None), unseeded)
    with pytest.raises(ValueError, match="seed"):
        scattered(random_seed=1.5)


def test_omnidirectional_sources_carry_no_pyroomacoustics_directivity():
    assert source_to_pra(Source("s", 1, 1, 1)).directivity is None
    assert source_to_pra(Source("c", 1, 1, 1, directivity="cardioid", direction=[1, 0, 0])).directivity is not None


def test_a_sources_sound_file_is_not_read(tmp_path):
    path = tmp_path / "not_audio.wav"
    path.write_bytes(b"")
    assert source_to_pra(Source("s", 1, 1, 1, sound_file=str(path))).signal is None


def test_mono_and_ambisonic_responses_share_an_edge():
    scene = shoebox_scene()
    mono, ambisonic = response(scene, np.zeros(10)), response(scene, np.zeros((10, 4)))
    scene.populate_ir_graph([mono])
    scene.populate_ir_graph([ambisonic], attribute="ambisonics")
    source, receiver = scene.sources[0], scene.receivers[0]
    assert scene.get_impulse_response(source, receiver) is mono
    assert scene.get_impulse_response(source, receiver, attribute="ambisonics") is ambisonic


def test_repopulating_one_attribute_leaves_the_other():
    scene = shoebox_scene()
    first, ambisonic, second = response(scene, np.zeros(10)), response(scene, np.zeros((10, 4))), response(scene, np.ones(10))
    scene.populate_ir_graph([first])
    scene.populate_ir_graph([ambisonic], attribute="ambisonics")
    scene.populate_ir_graph([second])
    source, receiver = scene.sources[0], scene.receivers[0]
    assert scene.get_impulse_response(source, receiver) is second
    assert scene.get_impulse_response(source, receiver, attribute="ambisonics") is ambisonic


def test_an_edge_without_responses_is_removed():
    scene = shoebox_scene()
    scene.populate_ir_graph([response(scene, np.zeros(10))])
    scene.populate_ir_graph([])
    assert list(scene.ir_graph.edges()) == []
    with pytest.raises(KeyError):
        scene.get_impulse_response(scene.sources[0], scene.receivers[0])


def test_sources_attached_to_a_library_room_are_scene_sources():
    scene = Scene("t")
    scene.add_acoustic_objects(AcousticObjectsLibrary.shoebox_room(7, 5, 3, sources=[Source("on_floor", 1, 1, 0.5)]))
    scene.add_source(Source("free", 2, 2, 1.5))
    assert len(scene.sources) == 2


def test_left_wall_is_on_the_receivers_left():
    # a receiver at the centre facing +x has +y on its left (receiver frame: x facing, y left, z up)
    width = 5
    room = AcousticObjectsLibrary.shoebox_room(7, width, 3)
    centroid_y = {w: next(o for o in room if o.name.endswith(w)).acoustic_geometry.centroid()[1] for w in ("left_wall", "right_wall", "front_wall")}
    assert centroid_y["left_wall"] == pytest.approx(width)
    assert centroid_y["right_wall"] == pytest.approx(0.0)
    assert centroid_y["front_wall"] == pytest.approx(width / 2)
