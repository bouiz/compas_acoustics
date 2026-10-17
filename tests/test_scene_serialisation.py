import numpy as np
from compas.data import json_dumps
from compas.data import json_loads

from compas_acoustics import AcousticMaterial
from compas_acoustics import AcousticObjectsLibrary
from compas_acoustics import Pyroomacoustics
from compas_acoustics import Receiver
from compas_acoustics import Scene
from compas_acoustics import Source
from compas_acoustics.methods import SimulationContext


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
    a = original.simulate(Pyroomacoustics(image_source_order=2))
    b = copy.simulate(Pyroomacoustics(image_source_order=2))
    assert sorted(a) == sorted(b)
    for key in a:
        np.testing.assert_allclose(b[key].denormalized(), a[key].denormalized())
