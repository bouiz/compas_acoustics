"""compas_acoustics' compas_pb plugin: every Data class travels as its own protobuf message."""

import hashlib
from pathlib import Path

import pytest
from compas.datastructures import Mesh
from compas.geometry import Polyline
from compas_pb import pb_dump_bts
from compas_pb import pb_load_bts
from compas_pb.core import serialize_message

import compas_acoustics.pb
from compas_acoustics import AcousticMaterial
from compas_acoustics import AcousticObject
from compas_acoustics import AcousticObjectsLibrary
from compas_acoustics import MaterialCategory
from compas_acoustics import Receiver
from compas_acoustics import Scene
from compas_acoustics import Source


def roundtrip(obj):
    return pb_load_bts(pb_dump_bts(obj))


def arm(obj):
    """The AnyData arm the envelope carries ``obj`` in, and the type URL of a message."""
    data = serialize_message(obj).data
    which = data.WhichOneof("data")
    return which, data.message.type_url if which == "message" else None


def shoebox_scene():
    scene = Scene("room", sampling_rate=44100)
    scene.add_acoustic_objects(AcousticObjectsLibrary.shoebox_room(7, 5, 3, materials=AcousticMaterial("wall", [0.2] * 7, [0.1] * 7)))
    scene.add_source(Source("src", 1.5, 1.5, 1.6))
    scene.add_receiver(Receiver("rec", 5, 3, 1.5))
    return scene


def test_a_material_round_trips_as_a_message():
    material = AcousticMaterial("wall", [0.2] * 7, [0.1] * 7, category="Wood", data_origin="test")
    assert arm(material) == ("message", "type.googleapis.com/compas_acoustics.pb.AcousticMaterialData")
    loaded = roundtrip(material)
    assert isinstance(loaded, AcousticMaterial)
    assert loaded.category is MaterialCategory.WOOD
    assert loaded.name == "wall" and loaded.data_origin == "test"
    assert list(loaded.absorption_coefficients) == [0.2] * 7
    assert list(loaded.scattering_coefficients) == [0.1] * 7
    assert loaded.guid == material.guid


def test_a_source_keeps_guid_directivity_direction_and_sound_file(tmp_path):
    sound = tmp_path / "s.wav"
    sound.write_bytes(b"")
    source = Source("src", 1, 2, 3, directivity="cardioid", direction=[0, 1, 0])
    source._sound_file = str(sound)
    # a Source is a compas Point: without its own message it would travel as a PointData
    assert arm(source) == ("message", "type.googleapis.com/compas_acoustics.pb.SourceData")
    loaded = roundtrip(source)
    assert isinstance(loaded, Source)
    assert loaded.guid == source.guid
    assert loaded.name == "src" and list(loaded.position) == [1, 2, 3]
    assert loaded.directivity == "cardioid"
    assert list(loaded.direction) == [0, 1, 0]
    assert loaded.sound_file == str(sound)
    assert list(loaded.spl) == list(source.spl)


def test_a_source_without_a_sound_file_keeps_none():
    assert roundtrip(Source("src", 0, 0, 1)).sound_file is None


def test_a_receiver_keeps_direction_up_and_radius():
    receiver = Receiver("rec", 5, 3, 1.5, direction=[0, 1, 0], radius=0.3, up=[0, 0, 1])
    assert arm(receiver)[1] == "type.googleapis.com/compas_acoustics.pb.ReceiverData"
    loaded = roundtrip(receiver)
    assert isinstance(loaded, Receiver)
    assert loaded.guid == receiver.guid
    assert list(loaded.position) == [5, 3, 1.5]
    assert list(loaded.direction) == [0, 1, 0] and list(loaded.up) == [0, 0, 1]
    assert loaded.radius == 0.3


def test_an_acoustic_object_keeps_its_mesh_material_and_role():
    obj = shoebox_scene().acoustic_objects[0]
    obj.role = "obstacle"
    assert arm(obj)[1] == "type.googleapis.com/compas_acoustics.pb.AcousticObjectData"
    loaded = roundtrip(obj)
    assert isinstance(loaded, AcousticObject)
    assert isinstance(loaded.acoustic_geometry, Mesh)
    assert loaded.acoustic_geometry.number_of_faces() == obj.acoustic_geometry.number_of_faces()
    assert loaded.acoustic_geometry.number_of_vertices() == obj.acoustic_geometry.number_of_vertices()
    assert loaded.material.name == "wall"
    assert loaded.role == "obstacle"
    assert loaded.name == obj.name and loaded.guid == obj.guid


def test_an_acoustic_object_without_a_material_keeps_none():
    obj = shoebox_scene().acoustic_objects[0]
    obj.material = None
    assert roundtrip(obj).material is None


def test_a_polyline_object_round_trips():
    obj = AcousticObject(Polyline([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]), AcousticMaterial("m", [0.1] * 7))
    loaded = roundtrip(obj)
    assert isinstance(loaded.acoustic_geometry, Polyline)
    assert [list(p) for p in loaded.acoustic_geometry.points] == [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]


def test_a_scene_round_trips_with_its_sampling_rate_and_guids():
    scene = shoebox_scene()
    scene.description = "a shoebox"
    assert arm(scene)[1] == "type.googleapis.com/compas_acoustics.pb.SceneData"
    loaded = roundtrip(scene)
    assert isinstance(loaded, Scene)
    assert loaded.name == "room" and loaded.description == "a shoebox"
    assert loaded.sampling_rate == 44100
    assert loaded.guid == scene.guid
    assert len(loaded.acoustic_objects) == 6
    assert [s.guid for s in loaded.sources] == [s.guid for s in scene.sources]
    assert [r.guid for r in loaded.receivers] == [r.guid for r in scene.receivers]


def test_the_schema_hash_follows_the_proto_text():
    digest = hashlib.sha256()
    root = Path(compas_acoustics.pb.PROTOBUF_DEFS)
    for path in sorted(root.rglob("*.proto")):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes().replace(b"\r\n", b"\n"))
    assert compas_acoustics.pb.SCHEMA == digest.hexdigest()


def test_the_plugin_is_discovered_through_the_entry_point():
    from compas_pb.plugin import entry_points

    names = [ep.value for ep in entry_points(group="compas_pb.plugins")]
    assert "compas_acoustics.pb.conversions" in names


@pytest.mark.parametrize("value", [{"a": 1, "b": [1.5, "x", None]}, [1, 2.0, "three"]])
def test_plain_containers_travel(value):
    assert roundtrip(value) == value


# ---------------------------------------------------------------------------
# Responses, audio, rays and reflections
# ---------------------------------------------------------------------------

import numpy as np  # noqa: E402

from compas_acoustics import ImpulseResponse  # noqa: E402
from compas_acoustics import RayField  # noqa: E402
from compas_acoustics import ScatteredRay  # noqa: E402
from compas_acoustics import SpecularPath  # noqa: E402
from compas_acoustics import SpecularReflections  # noqa: E402
from compas_acoustics.pb import plain  # noqa: E402
from compas_acoustics.pb.types import Audio  # noqa: E402


def stereo_response(**kwargs):
    rng = np.random.default_rng(0)
    raw = rng.normal(size=(64, 2)) * 0.01
    return ImpulseResponse("ir", Source("src", 1, 1, 1), Receiver("rec", 2, 2, 1), raw, 48000,
                           generation={"method": "binaural", "seed": np.int64(3), "band": np.float64(0.5),
                                       "orders": np.array([1, 2])},
                           **kwargs)


def test_a_response_round_trips_its_raw_samples_and_generation():
    ir = stereo_response(normalized=False)
    assert arm(ir)[1] == "type.googleapis.com/compas_acoustics.pb.ImpulseResponseData"
    loaded = roundtrip(ir)
    assert isinstance(loaded, ImpulseResponse)
    assert loaded.denormalized().shape == (64, 2)
    assert np.array_equal(loaded.denormalized(), ir.denormalized())
    assert loaded.normalized is False and loaded.sampling_rate == 48000
    assert loaded.generation == {"method": "binaural", "seed": 3, "band": 0.5, "orders": [1, 2]}
    assert loaded.source.guid == ir.source.guid and loaded.receiver.guid == ir.receiver.guid
    assert loaded.ambisonic_order is None and loaded.normalization is None and loaded.channel_order is None
    assert loaded.guid == ir.guid


def test_a_mono_response_keeps_its_one_dimensional_shape():
    ir = ImpulseResponse("ir", Source("s", 0, 0, 1), Receiver("r", 1, 0, 1), np.linspace(1, 0, 10), 44100)
    assert roundtrip(ir).denormalized().shape == (10,)


def test_an_ambisonic_response_keeps_order_and_conventions():
    raw = np.zeros((16, 4))
    raw[0] = [1, 0.5, 0.25, 0.1]
    ir = ImpulseResponse("ir", Source("s", 0, 0, 1), Receiver("r", 1, 0, 1), raw, 48000,
                         ambisonic_order=1, normalization="n3d", channel_order="acn")
    loaded = roundtrip(ir)
    assert (loaded.ambisonic_order, loaded.normalization, loaded.channel_order) == (1, "n3d", "acn")
    assert loaded.n_channels == 4


def test_audio_round_trips():
    audio = Audio(np.arange(12.0).reshape(6, 2) / 12, 44100, "binaural")
    assert arm(audio)[1] == "type.googleapis.com/compas_acoustics.pb.AudioData"
    loaded = roundtrip(audio)
    assert isinstance(loaded, Audio)
    assert np.array_equal(loaded.samples, audio.samples)
    assert loaded.sampling_rate == 44100 and loaded.format == "binaural"


def test_a_ray_field_round_trips_rays_scattered_rays_and_sources():
    field = RayField.from_scene(shoebox_scene(), n_rays=20, max_reflection_order=3, include_scattering=True, random_seed=1)
    assert field.scattered_rays, "the test needs scattered rays"
    assert arm(field)[1] == "type.googleapis.com/compas_acoustics.pb.RayFieldData"
    loaded = roundtrip(field)
    assert isinstance(loaded, RayField)
    assert len(loaded.rays) == len(field.rays)
    assert len(loaded.scattered_rays) == len(field.scattered_rays)
    for a, b in zip(loaded.rays, field.rays):
        assert np.allclose(np.asarray(a.points, dtype=float), np.asarray(b.points, dtype=float))
        assert (a.source_index, a.ray_index, a.distance_offset) == (b.source_index, b.ray_index, b.distance_offset)
        assert a.source.guid == b.source.guid
    for a, b in zip(loaded.scattered_rays, field.scattered_rays):
        assert isinstance(a, ScatteredRay)
        assert (a.reflection_order, a.parent_ray_index, a.distance_offset) == (b.reflection_order, b.parent_ray_index, b.distance_offset)
        assert a.energy == pytest.approx(b.energy)
        assert a.source.guid == b.source.guid


def test_specular_reflections_round_trip_as_paths():
    found = SpecularReflections.from_scene(shoebox_scene(), max_order=2)
    assert arm(found)[1] == "type.googleapis.com/compas_acoustics.pb.SpecularReflectionsData"
    loaded = roundtrip(found)
    assert isinstance(loaded, SpecularReflections)
    assert len(loaded.paths) == len(found.paths)
    for a, b in zip(loaded.paths, found.paths):
        assert isinstance(a, SpecularPath)
        assert (a.order, a.walls, a.receiver_index) == (b.order, b.walls, b.receiver_index)
        assert np.allclose(list(a.image), list(b.image))
        assert a.travel_time == pytest.approx(b.travel_time)
    assert found.orders == loaded.orders


def test_plain_turns_numpy_into_python():
    assert plain({"a": np.float64(1.5), "b": np.array([[1, 2]]), "c": (np.int32(3),), "d": np.bool_(True)}) == {
        "a": 1.5, "b": [[1, 2]], "c": [3], "d": True}


FIXTURES = Path(__file__).parent / "pb_fixtures"


def _makers():
    import importlib.util

    spec = importlib.util.spec_from_file_location("pb_fixtures_generate", FIXTURES / "generate.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.FIXTURES


@pytest.mark.parametrize("name, kind", [("scene", Scene), ("source", Source), ("ir", ImpulseResponse), ("rayfield", RayField), ("audio", Audio)])
def test_the_interop_fixtures_decode_to_what_the_generator_makes(name, kind):
    import base64

    committed = pb_load_bts(base64.b64decode((FIXTURES / f"{name}.b64").read_text()))
    made = _makers()[name]()
    assert isinstance(committed, kind)
    # the generator fixes every guid, so the committed payload names the same objects as a fresh one
    assert committed.guid == made.guid
