import pickle

import numpy as np
import pytest
from compas.data import Data
from compas.data import json_dumps
from compas.data import json_loads

from compas_acoustics import AcousticAnalyser
from compas_acoustics import AcousticMaterial
from compas_acoustics import AcousticObjectsLibrary
from compas_acoustics import Auraliser
from compas_acoustics import MaterialsLibrary
from compas_acoustics import Mono
from compas_acoustics import RayField
from compas_acoustics import Receiver
from compas_acoustics import Scene
from compas_acoustics import Source
from compas_acoustics.simulation_methods import SimulationContext


def scene():
    s = Scene("room", "a test room")
    s.add_acoustic_objects(AcousticObjectsLibrary.shoebox_room(7, 5, 3, materials=AcousticMaterial("wall", 0.2, 0.1)))
    s.add_source(Source("src", 1.5, 1.5, 1.6))
    s.add_receiver(Receiver("rec", 5, 3, 1.5))
    return s


def keys(s):
    return sorted(SimulationContext.key(source, receiver) for source in s.sources for receiver in s.receivers)


def test_a_scene_survives_json():
    original = scene()
    copy = json_loads(json_dumps(original))
    assert (copy.name, copy.description) == (original.name, original.description)
    assert keys(copy) == keys(original)
    before = list(original.acoustic_objects)
    after = list(copy.acoustic_objects)
    assert [o.role for o in after] == [o.role for o in before]
    assert [o.material.absorption_coefficients for o in after] == [o.material.absorption_coefficients for o in before]
    for a, b in zip(before, after):
        np.testing.assert_allclose(b.acoustic_geometry.vertices_attributes("xyz"), a.acoustic_geometry.vertices_attributes("xyz"))


def test_a_round_tripped_scene_simulates_the_same():
    original = scene()
    copy = json_loads(json_dumps(original))
    a = original.simulate(Mono(image_source_order=2, ray_tracing=False))
    b = copy.simulate(Mono(image_source_order=2, ray_tracing=False))
    assert sorted(a) == sorted(b)
    for key in a:
        np.testing.assert_allclose(b[key].denormalized(), a[key].denormalized())


def test_the_sampling_rate_survives_the_round_trip():
    original = Scene("rate", sampling_rate=44100)
    assert json_loads(json_dumps(original)).sampling_rate == 44100


def traced_response():
    s = scene()
    return next(iter(s.simulate(Mono(ray_tracing=True, n_rays=1000, duration=0.5)).values()))


def test_an_impulse_response_survives_json():
    original = traced_response()
    copy = json_loads(json_dumps(original))
    np.testing.assert_array_equal(copy.denormalized(), original.denormalized())
    np.testing.assert_array_equal(copy.ir, original.ir)
    assert (copy.name, copy.sampling_rate, copy.normalized) == (original.name, original.sampling_rate, original.normalized)
    assert (copy.source.name, copy.receiver.name) == (original.source.name, original.receiver.name)
    assert copy.generation["method"] == "mono" and copy.generation["seed"] == 42


def test_an_ambisonic_impulse_response_keeps_its_convention():
    from compas_acoustics import ImpulseResponse

    ir = ImpulseResponse("a", Source("s", 1, 1, 1), Receiver("r", 2, 2, 1), np.random.default_rng(0).normal(size=(64, 4)), 48000,
                         ambisonic_order=1, normalization="n3d", channel_order="acn")
    copy = json_loads(json_dumps(ir))
    assert (copy.ambisonic_order, copy.normalization, copy.channel_order, copy.n_channels) == (1, "n3d", "acn", 4)
    np.testing.assert_array_equal(copy.denormalized(), ir.denormalized())


def scattering_scene():
    s = Scene("scattering")
    s.add_acoustic_objects(AcousticObjectsLibrary.shoebox_room(6, 5, 3, materials=AcousticMaterial("wall", 0.2, 0.5)))
    s.add_source(Source("src", 1.5, 1.5, 1.5))
    s.add_receiver(Receiver("rec", 4, 3, 1.4))
    return s


def test_a_ray_field_survives_json_with_its_scattered_rays_and_sources():
    original = RayField.from_scene(scattering_scene(), n_rays=5, max_reflection_order=3, include_scattering=True,
                                   scatter_density=10, scatter_max_bounces=1, random_seed=42)
    assert original.scattered_rays
    copy = json_loads(json_dumps(original))
    assert len(copy) == len(original) and len(copy.scattered_rays) == len(original.scattered_rays)
    for a, b in zip(original.rays, copy.rays):
        np.testing.assert_allclose(np.array(b.points), np.array(a.points))
        assert (b.ray_index, b.source_index, b.source.name) == (a.ray_index, a.source_index, a.source.name)
        np.testing.assert_allclose(b.direction, a.direction)
    first, again = original.scattered_rays[0], copy.scattered_rays[0]
    assert type(again).__name__ == "ScatteredRay"
    assert (again.parent_ray_index, again.reflection_order) == (first.parent_ray_index, first.reflection_order)
    assert again.energy == pytest.approx(first.energy) and again.distance_offset == pytest.approx(first.distance_offset)
    assert len({id(ray.source) for ray in copy.rays}) == 1


def test_the_services_are_plain_classes_that_pickle():
    ir = traced_response()
    analyser = AcousticAnalyser(ir)
    analyser.analyse()
    auraliser = Auraliser("a", {"s": {"ir": np.ones((10, 1)), "sampling_rate": 48000}}, {"s": {"filepath": "x.wav"}}, "mono")
    for service in (analyser, auraliser, MaterialsLibrary()):
        assert not isinstance(service, Data)
    assert pickle.loads(pickle.dumps(analyser)).analysis_data.keys() == analyser.analysis_data.keys()
    assert pickle.loads(pickle.dumps(auraliser)).name == "a"
    assert pickle.loads(pickle.dumps(MaterialsLibrary())).index == MaterialsLibrary().index


def test_every_class_is_stored_under_the_package_whatever_its_module():
    import inspect

    import compas
    from compas.data import Data

    import compas_acoustics as ca

    classes = [c for _, c in inspect.getmembers(ca, inspect.isclass) if issubclass(c, Data)]
    assert len(classes) == 11
    for cls in classes:
        assert cls.__dtype__.fget(object.__new__(cls)) == "compas_acoustics/" + cls.__name__
    scene = ca.Scene("dtype")
    scene.add_acoustic_objects(ca.AcousticObjectsLibrary.shoebox_room(7, 5, 3, materials=ca.AcousticMaterial("w", 0.2, 0.1)))
    scene.add_source(ca.Source("s", 1, 1, 1))
    scene.add_receiver(ca.Receiver("r", 3, 2, 1))
    text = compas.json_dumps(scene)
    assert '"compas_acoustics.' not in text
    back = compas.json_loads(text)
    assert isinstance(back, ca.Scene) and isinstance(back.acoustic_objects[0].material, ca.AcousticMaterial)


def test_every_library_material_loads():
    import compas_acoustics as ca

    library = ca.MaterialsLibrary()
    names = library.material_names
    assert len(names) == 95
    assert all(isinstance(library.material_from_name(name), ca.AcousticMaterial) for name in names)
