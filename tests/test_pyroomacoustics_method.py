import numpy as np
import pyroomacoustics as pra
import pytest

from compas_acoustics import AcousticMaterial
from compas_acoustics import AcousticObjectsLibrary
from compas_acoustics import Receiver
from compas_acoustics import Scene
from compas_acoustics import Source
from compas_acoustics.simulation_methods import Pyroomacoustics
from compas_acoustics.simulation_methods import SimulationContext
from compas_acoustics.raytracing import OPEN_ROOM_DURATION
from compas_acoustics.raytracing import default_duration
from compas_acoustics.raytracing import ray_traced_rirs
from compas_acoustics.room_geometry import RoomGeometry

FS = 48000
SOURCE = np.array([1.5, 1.5, 1.6])
RECEIVER = np.array([5.0, 3.0, 1.5])
WALL = AcousticMaterial("wall", 0.2, 0.1)
SMOOTH = AcousticMaterial("smooth", 0.2, [0.0] * 7)


def scene_of(materials=WALL, leaves=6):
    scene = Scene("test")
    room = AcousticObjectsLibrary.shoebox_room(7, 5, 3, materials=materials)
    scene.add_acoustic_objects(room[:leaves])
    return scene


def geometry(materials=WALL):
    return RoomGeometry.from_scene(scene_of(materials))


def traced(g, sources=(SOURCE,), receivers=(RECEIVER,), **kwargs):
    kwargs.setdefault("n_rays", 2000)
    kwargs.setdefault("duration", 0.5)
    return ray_traced_rirs(
        g, [pra.SoundSource(np.asarray(s, dtype=float)) for s in sources],
        np.array(receivers, dtype=float).T, FS, **kwargs,
    )


def test_one_response_per_microphone_and_source():
    rirs = traced(geometry(), sources=(SOURCE, [5.5, 1.0, 2.0]), receivers=(RECEIVER, [2.0, 4.0, 1.2]))
    assert len(rirs) == 2
    assert all(len(row) == 2 for row in rirs)
    assert all(r.shape == (24000,) for row in rirs for r in row)


def test_the_duration_sets_the_length():
    (rir,), = traced(geometry(), duration=0.25)
    assert rir.shape == (12000,)


def test_the_default_length_comes_from_eyring():
    g = geometry()
    expected = default_duration(g, air_absorption=True)
    assert 0.5 <= expected <= 12.0
    (rir,), = traced(g, duration=None, n_rays=500)
    assert rir.shape == (round(expected * FS),)


def test_an_open_room_gets_the_fixed_default():
    with pytest.warns(UserWarning):
        g = RoomGeometry.from_scene(scene_of(leaves=5))
    assert default_duration(g, air_absorption=True) == OPEN_ROOM_DURATION


def test_a_fully_absorbing_surface_has_a_finite_default():
    window = AcousticMaterial("window", 1.0, 0.1)
    materials = {"floor": WALL, "ceiling": window, "left_wall": WALL, "right_wall": WALL, "front_wall": WALL, "back_wall": WALL}
    assert 0.5 <= default_duration(geometry(materials), air_absorption=True) <= 12.0


def test_seeded_runs_repeat():
    (first,), = traced(geometry(), seed=7)
    (second,), = traced(geometry(), seed=7)
    np.testing.assert_array_equal(first, second)


def uniform(scattering=0.15):
    """Upstream's uniform_spec: 5 x 4 x 3 m, alpha 0.15, no air absorption."""
    material = AcousticMaterial("uniform", 0.15, [scattering] * 7)
    scene = Scene("uniform")
    scene.add_acoustic_objects(AcousticObjectsLibrary.shoebox_room(5, 4, 3, materials=material))
    return RoomGeometry.from_scene(scene)


def traced_uniform(g, **kwargs):
    kwargs.setdefault("n_rays", 2000)
    kwargs.setdefault("duration", 0.6)
    kwargs.setdefault("air_absorption", False)
    return traced(g, sources=([1.2, 1.0, 1.3],), receivers=([3.5, 2.8, 1.6],), **kwargs)[0][0]


def test_ray_tracing_is_stochastic_across_seeds():
    """Upstream: different seeds are different draws of the same room."""
    a = traced_uniform(uniform(), max_order=6, seed=1)
    b = traced_uniform(uniform(), max_order=6, seed=2)
    assert not np.array_equal(a, b)
    assert np.isclose(np.sum(a ** 2), np.sum(b ** 2), rtol=0.25)


def test_the_duration_must_be_positive():
    with pytest.raises(ValueError, match="duration"):
        traced_uniform(uniform(), duration=0.0)


def test_there_is_a_tail_beyond_the_image_source_order():
    """Upstream: the point of the ray tracer, energy where a low order would stop."""
    rir = traced_uniform(uniform(), max_order=4, duration=0.8)
    assert np.sum(rir[int(0.4 * FS):] ** 2) > 0.0


def test_a_material_without_scattering_ray_traces_with_the_default():
    ((rir,),) = traced(geometry(AcousticMaterial("bare", 0.2)), seed=1)
    assert np.all(np.isfinite(rir)) and np.sum(rir ** 2) > 0.0


def simulator(receiver=None, materials=WALL, leaves=6):
    scene = scene_of(materials, leaves)
    scene.add_source(Source("src", *SOURCE))
    scene.add_receiver(receiver or Receiver("rec", *RECEIVER))
    return scene


def test_ray_tracing_returns_mono_responses_at_48_khz():
    scene = simulator()
    rirs = scene.simulate(Pyroomacoustics(image_source_order=2, ray_tracing=True, n_rays=2000, seed=1, duration=0.5))
    (ir,) = rirs.values()
    assert ir.ir.shape == (24000,)
    expected = {"method": "pyroomacoustics", "scale": "package", "ray_tracing": True, "air_absorption": True, "image_source_order": 2}
    assert expected.items() <= ir.generation.items()
    assert ir.sampling_rate == FS
    assert scene.impulse_responses("data") == rirs
    assert scene.get_impulse_response(ir.source, ir.receiver) is ir


def test_a_vertical_receiver_still_ray_traces():
    scene = simulator(receiver=Receiver("rec", *RECEIVER, direction=[0, 0, 1]))
    assert len(scene.simulate(Pyroomacoustics(image_source_order=1, ray_tracing=True, n_rays=500, duration=0.2))) == 1


def test_an_open_scene_still_ray_traces():
    with pytest.warns(UserWarning):
        scene = simulator(leaves=5)
        rirs = scene.simulate(Pyroomacoustics(image_source_order=1, ray_tracing=True, n_rays=500, duration=0.2))
    assert all(np.all(np.isfinite(ir.ir)) for ir in rirs.values())


# ---------------------------------------------------------- the method --
@pytest.mark.parametrize("ray_tracing", [False, True])
def test_both_modes_meet_the_contract(assert_meets_contract, ray_tracing):
    scene = simulator()
    options = {"n_rays": 2000, "duration": 0.4, "seed": 1} if ray_tracing else {}
    method = Pyroomacoustics(image_source_order=2, ray_tracing=ray_tracing, **options)
    assert_meets_contract(scene.simulate(method), SimulationContext.from_scene(scene), method)


def test_ism_and_ray_tracing_share_the_direct_sound():
    scene = simulator()
    (ism,) = scene.simulate(Pyroomacoustics(image_source_order=2, air_absorption=False, ray_tracing=False)).values()
    (traced,) = scene.simulate(Pyroomacoustics(image_source_order=2, air_absorption=False, ray_tracing=True,
                                        n_rays=2000, duration=0.4, seed=1)).values()
    n = ism.direct_sound_index()
    assert n == traced.direct_sound_index()
    a, b = ism.denormalized(), traced.denormalized()
    assert np.argmax(np.abs(a[: n + 50])) == np.argmax(np.abs(b[: n + 50])) == n
    # the ray tracer adds its noise from t = 0, as in pyroomacoustics and upstream
    assert b[n] == pytest.approx(a[n], rel=0.1)


def test_ray_tracing_settings_without_ray_tracing_raise():
    with pytest.raises(ValueError, match="n_rays"):
        Pyroomacoustics(ray_tracing=False, n_rays=1000)
    with pytest.raises(ValueError, match="duration"):
        Pyroomacoustics(ray_tracing=False, duration=0.5)
    Pyroomacoustics(ray_tracing=False, n_rays=20000)  # the default, given explicitly, is fine


def test_image_sources_need_no_scattering():
    scene = simulator(materials=AcousticMaterial("bare", 0.2))
    assert len(scene.simulate(Pyroomacoustics(image_source_order=2, ray_tracing=False))) == 1


def test_default_duration_follows_the_predicted_rt():
    from compas_acoustics.diffuse import predicted_rt

    g = geometry()
    rt = float(np.max(predicted_rt(g, air_absorption=True, warn=False)))
    assert default_duration(g, air_absorption=True) == pytest.approx(np.clip(2.5 * rt, 0.5, 12.0))


def test_a_source_outside_the_room_warns_and_prints_nothing(capsys):
    scene = simulator()
    scene.add_source(Source("outside", 9.0, 1.5, 1.6))
    with pytest.warns(UserWarning, match="outside"):
        rirs = scene.simulate(Pyroomacoustics(image_source_order=1, ray_tracing=True, n_rays=500, duration=0.2))
    assert len(rirs) == 1
    assert capsys.readouterr().out == ""


def test_the_search_time_estimate_follows_the_measurements():
    from compas_acoustics.simulation_methods import image_source_search_seconds

    assert image_source_search_seconds(6, 6) < 0.1
    assert 0.1 < image_source_search_seconds(18, 6) < 3.0
    assert 40.0 < image_source_search_seconds(42, 6) < 400.0


def test_a_slow_image_source_search_warns_before_it_starts(monkeypatch):
    from compas.datastructures import Mesh
    from compas.geometry import Box
    from compas.geometry import Frame

    from compas_acoustics import AcousticObject
    from compas_acoustics.simulation_methods import Pyroomacoustics as Method

    scene = simulator()
    for i in range(3):
        scene.add_acoustic_object(AcousticObject(Mesh.from_shape(Box(0.8, 0.8, 1.0, Frame([2.0 + 1.5 * i, 4.0, 0.5]))), WALL,
                                                 name=f"box{i}", role="obstacle"))
    monkeypatch.setattr(Method, "_image_sources", lambda *a, **k: pytest.fail("searched"))
    with pytest.warns(UserWarning, match="image source search over 42 faces"), pytest.raises(pytest.fail.Exception):
        scene.simulate(Method(image_source_order=6, ray_tracing=False))


def test_ray_tracing_runs_in_the_medium():
    from compas_acoustics import AirConditions

    g = geometry(SMOOTH)
    (cold,), = traced(g, air=AirConditions(temperature=0.0), seed=42)
    (legacy,), = traced(g, seed=42)
    assert not np.allclose(cold, legacy)


def test_the_default_duration_follows_the_medium():
    from compas_acoustics import AirConditions

    g = geometry()
    assert default_duration(g, air_absorption=True, air=None) == default_duration(g, air_absorption=True)
    dry = default_duration(g, air_absorption=True, speed_of_sound=AirConditions(humidity=10.0).speed_of_sound,
                           air=AirConditions(humidity=10.0))
    assert dry != default_duration(g, air_absorption=True)
