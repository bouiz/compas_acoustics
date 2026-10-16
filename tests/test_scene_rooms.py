import numpy as np
import pyroomacoustics as pra
import pytest

from compas_acoustics import AcousticMaterial
from compas_acoustics import AcousticObjectsLibrary
from compas_acoustics import ImpulseResponse
from compas_acoustics import Receiver
from compas_acoustics import RayField
from compas_acoustics import Scene
from compas_acoustics import Source
from compas_acoustics.pra_converter import scene_to_pra_room
from compas_acoustics.pra_converter import source_to_pra


def shoebox_scene():
    scene = Scene("test")
    scene.add_acoustic_object(AcousticObjectsLibrary.shoebox_room(7, 5, 3, materials=AcousticMaterial("wall", 0.2, 0.1)))
    scene.add_source(Source("src", 1.5, 1.5, 1.6))
    scene.add_receiver(Receiver("rec", 5, 3, 1.5))
    return scene


def response(scene, data):
    return ImpulseResponse("k", scene.sources[0], scene.receivers[0], data, 48000)


def test_scene_to_pra_room_keeps_its_signature():
    room = scene_to_pra_room(shoebox_scene(), 48000, 2, False, True)
    assert isinstance(room, pra.Room)
    assert len(room.walls) == 6
    assert room.fs == 48000


def test_trace_rays_runs_on_room_geometry():
    field = RayField.from_scene(shoebox_scene(), n_rays=20, max_reflection_order=3)
    assert len(field.rays) == 20


def test_ray_field_has_no_merge_coplanar_argument():
    with pytest.raises(TypeError):
        RayField.from_scene(shoebox_scene(), n_rays=5, merge_coplanar=True)


def test_omnidirectional_sources_carry_no_pyroomacoustics_directivity():
    assert source_to_pra(Source("s", 1, 1, 1)).directivity is None
    assert source_to_pra(Source("c", 1, 1, 1, directivity="cardioid", direction=[1, 0, 0])).directivity is not None


def test_mono_and_ambisonic_responses_share_an_edge():
    scene = shoebox_scene()
    mono, ambisonic = response(scene, np.zeros(10)), response(scene, np.zeros((10, 4)))
    scene.populate_ir_graph([mono])
    scene.populate_ir_graph([ambisonic], attribute="hybrid")
    source, receiver = scene.sources[0], scene.receivers[0]
    assert scene.get_impulse_response(source, receiver) is mono
    assert scene.get_impulse_response(source, receiver, attribute="hybrid") is ambisonic


def test_repopulating_one_attribute_leaves_the_other():
    scene = shoebox_scene()
    first, ambisonic, second = response(scene, np.zeros(10)), response(scene, np.zeros((10, 4))), response(scene, np.ones(10))
    scene.populate_ir_graph([first])
    scene.populate_ir_graph([ambisonic], attribute="hybrid")
    scene.populate_ir_graph([second])
    source, receiver = scene.sources[0], scene.receivers[0]
    assert scene.get_impulse_response(source, receiver) is second
    assert scene.get_impulse_response(source, receiver, attribute="hybrid") is ambisonic


def test_an_edge_without_responses_is_removed():
    scene = shoebox_scene()
    scene.populate_ir_graph([response(scene, np.zeros(10))])
    scene.populate_ir_graph([])
    assert list(scene.ir_graph.edges()) == []
    with pytest.raises(KeyError):
        scene.get_impulse_response(scene.sources[0], scene.receivers[0])


def test_number_of_sources_counts_attached_sources():
    scene = Scene("t")
    scene.add_acoustic_object(AcousticObjectsLibrary.shoebox_room(7, 5, 3, sources=[Source("on_floor", 1, 1, 0.5)]))
    scene.add_source(Source("free", 2, 2, 1.5))
    assert scene.number_of_sources == len(scene.sources) == 2
