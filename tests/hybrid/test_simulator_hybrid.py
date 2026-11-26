import warnings

import numpy as np
import pytest
from compas.datastructures import Mesh
from compas.geometry import Box

from compas_acoustics import AcousticMaterial
from compas_acoustics import AcousticObject
from compas_acoustics import AcousticObjectsLibrary
from compas_acoustics import Receiver
from compas_acoustics import Scene
from compas_acoustics import Source
from compas_acoustics.room.diffuse import DiffuseFieldWarning
from compas_acoustics.hybrid.diffraction import ShellEdgeWarning
from compas_acoustics.hybrid.diffraction import diffraction_edges
from compas_acoustics.hybrid.diffraction import enclosure_edges
from compas_acoustics.hybrid.diffraction import encode_diffraction
from compas_acoustics.simulation_methods import Ambisonics
from compas_acoustics.simulation_methods import Binaural
from compas_acoustics.simulation_methods import Mono
from compas_acoustics.simulation_methods import SimulationContext
from compas_acoustics.room.air import AirConditions
from compas_acoustics.room.geometry import RoomGeometry
from compas_acoustics.room.geometry import SurfaceMaterial
from compas_acoustics.room.raytracing import MIN_RAYS

FS = 48000
WALL = AcousticMaterial("wall", 0.2, 0.1)
# These tests read the ambisonic output; the binaural one is tested in test_hybrid_outputs.py and test_binaural.py
FAST = dict(image_source_order=2, duration=0.5)
FAST_RT = dict(image_source_order=3, n_rays=2000, duration=1.0, seed=1)


def make(receivers=None, materials=WALL, leaves=6, panel=False):
    scene = Scene("test")
    room = AcousticObjectsLibrary.shoebox_room(10, 8, 4, materials=materials)
    scene.add_acoustic_objects(room[:leaves])
    if panel:
        mesh = Mesh.from_vertices_and_faces([[5, 3, 0.5], [5, 5, 0.5], [5, 5, 2.5], [5, 3, 2.5]], [[0, 1, 2, 3]])
        scene.add_acoustic_object(AcousticObject(mesh, WALL, name="panel", role="obstacle"))
    scene.add_source(Source("src", 2.0, 4.0, 1.5))
    for receiver in receivers or [Receiver("rec", 8.0, 4.0, 1.5)]:
        scene.add_receiver(receiver)
    return scene


def references(scene, **kwargs):
    return scene.simulate(Mono(ray_tracing=True, **dict(FAST_RT, **kwargs)))


def uneven():
    carpet = AcousticMaterial("carpet", 0.9, 0.1)
    return {"floor": carpet, "ceiling": carpet, "left_wall": WALL, "right_wall": WALL, "front_wall": WALL, "back_wall": WALL}


# ------------------------------------------------------- ambisonic output --
def test_one_ambisonic_response_per_pair():
    scene = make([Receiver("a", 8.0, 4.0, 1.5), Receiver("b", 3.0, 6.0, 1.2, direction=[0, 1, 0])])
    hybrid = Ambisonics(references(scene), sh_order=1, **FAST)
    rirs = scene.simulate(hybrid)
    assert len(rirs) == 2
    for key, ir in rirs.items():
        assert ir.ir.shape == (int(0.5 * FS), 4)
        assert (ir.ambisonic_order, ir.normalization, ir.channel_order) == (1, "sn3d", "acn")
        assert ir.sampling_rate == FS
        assert scene.get_impulse_response(ir.source, ir.receiver, attribute="ambisonics") is ir
        assert hybrid.results[key].normalization == "SN3D"
    assert scene.impulse_responses("ambisonics") == rirs


def test_the_mono_responses_stay_in_the_graph():
    scene = make()
    mono = references(scene)
    scene.simulate(Ambisonics(mono, **FAST))
    (ir,) = mono.values()
    assert scene.get_impulse_response(ir.source, ir.receiver) is ir
    assert scene.get_impulse_response(ir.source, ir.receiver, attribute="mono") is ir
    assert scene.impulse_responses("data") == mono


def test_references_at_another_rate_raise():
    scene = make()
    refs = references(scene)
    for ir in refs.values():
        ir.sampling_rate = 16000
    with pytest.raises(ValueError, match="sample rate"):
        scene.simulate(Ambisonics(refs, **FAST))


def test_a_multichannel_reference_raises():
    scene = make()
    refs = references(scene)
    for ir in refs.values():
        ir.ir = np.stack([ir.denormalized(), ir.denormalized()], axis=1)
    with pytest.raises(ValueError, match="mono"):
        scene.simulate(Ambisonics(refs, **FAST))


def test_references_missing_a_pair_raise():
    scene = make()
    with pytest.raises(ValueError, match="no response"):
        scene.simulate(Ambisonics({}, **FAST))


def test_an_open_room_raises():
    with pytest.warns(UserWarning):
        scene = make(leaves=5)
        with pytest.raises(ValueError, match="close"):
            scene.simulate(Ambisonics({}, **FAST))


def test_a_vertical_receiver_raises_before_any_work(monkeypatch):
    scene = make([Receiver("rec", 8.0, 4.0, 1.5, direction=[0, 0, 1])])
    monkeypatch.setattr(Mono, "compute", lambda *a, **k: pytest.fail("worked first"))
    with pytest.raises(ValueError, match="parallel"):
        scene.simulate(Ambisonics(**FAST))


def test_a_fully_absorbing_surface_runs():
    window = AcousticMaterial("window", 1.0, 0.05)
    materials = {"floor": WALL, "ceiling": window, "left_wall": WALL, "right_wall": WALL, "front_wall": WALL, "back_wall": WALL}
    scene = make(materials=materials)
    (ir,) = scene.simulate(Ambisonics(references(scene), **FAST)).values()
    assert np.all(np.isfinite(ir.ir))


def l_room():
    outline = [(0, 0), (8, 0), (8, 3), (5, 3), (5, 6), (0, 6)]
    n = len(outline)
    vertices = [[x, y, 0.0] for x, y in outline] + [[x, y, 3.0] for x, y in outline]
    faces = [list(range(n))[::-1], list(range(n, 2 * n))]
    faces += [[k, (k + 1) % n, (k + 1) % n + n, k + n] for k in range(n)]
    scene = Scene("l_room")
    scene.add_acoustic_object(AcousticObject(Mesh.from_vertices_and_faces(vertices, faces), WALL, name="l_room"))
    scene.add_source(Source("src", 7.0, 1.0, 1.5))
    scene.add_receiver(Receiver("rec", 4.0, 5.5, 1.5))
    return scene


def test_diffraction_at_the_room_boundary_needs_no_obstacles():
    """The L's re-entrant corner diffracts: the paths are added to the early field, the late field is untouched."""
    scene = l_room()
    geometry = RoomGeometry.from_scene(scene)
    assert geometry.is_closed and "obstacle" not in geometry.face_roles
    off = Ambisonics(tail="eyring", normalization="N3D", diffraction=False, seed=1, **FAST)
    on = Ambisonics(tail="eyring", normalization="N3D", diffraction=True, seed=1, **FAST)
    scene.simulate(off)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ShellEdgeWarning)
        scene.simulate(on)
    (off,), (on,) = off.results.values(), on.results.values()
    assert off.core.diffraction_paths == () and len(on.core.diffraction_paths) > 0
    added = encode_diffraction(list(on.core.diffraction_paths), 1, off.early.shape[0], FS)
    np.testing.assert_allclose(on.early - off.early, added, atol=1e-12)
    np.testing.assert_array_equal(on.late, off.late)


def pillar_scene():
    scene = Scene("pillar")
    scene.add_acoustic_objects(AcousticObjectsLibrary.shoebox_room(10, 8, 3, materials=WALL))
    pillar = Mesh.from_shape(Box.from_corner_corner_height([4, 3, 0], [5, 4, 0], 3))
    scene.add_acoustic_object(AcousticObject(pillar, WALL, name="pillar"))
    return scene


def test_a_boundary_pillar_has_four_270_degree_corners():
    g = RoomGeometry.from_scene(pillar_scene())
    assert g.is_closed
    with warnings.catch_warnings():
        warnings.simplefilter("error", ShellEdgeWarning)
        edges = diffraction_edges(g)
    assert len(edges) == 4
    for e in edges:
        assert e.theta_w == 1.5 * np.pi
        assert e.start[2] == 0.0 and e.end[2] == 3.0
        np.testing.assert_array_equal(e.start[:2], e.end[:2])
    assert {tuple(e.start[:2]) for e in edges} == {(4.0, 3.0), (4.0, 4.0), (5.0, 3.0), (5.0, 4.0)}


def l_prism(ceiling):
    outline = [(0, 0), (8, 0), (8, 3), (5, 3), (5, 6), (0, 6)]
    n = len(outline)
    vertices = [[x, y, 0.0] for x, y in outline] + [[x, y, 3.0] for x, y in outline]
    faces = [list(range(n))[::-1]]
    names = ["floor"]
    if ceiling:
        faces.append(list(range(n, 2 * n)))
        names.append("ceiling")
    faces += [[k, (k + 1) % n, (k + 1) % n + n, k + n] for k in range(n)]
    names += ["wall"] * n
    materials = {name: SurfaceMaterial.from_acoustic_material(WALL) for name in set(names)}
    return RoomGeometry.from_mesh(vertices, faces, names, materials)


def test_room_edges_are_not_searched_in_a_boundary_that_does_not_close():
    g = l_prism(ceiling=True)
    assert g.is_closed
    assert len(enclosure_edges(g)) == 1
    with pytest.warns(UserWarning, match="do not close a volume"):
        g = l_prism(ceiling=False)
    assert not g.is_closed
    with pytest.warns(ShellEdgeWarning, match="does not close a volume"):
        assert enclosure_edges(g) == []


def test_diffraction_in_a_rectangular_room_changes_nothing():
    scene = make()
    refs = references(scene)
    (off,) = scene.simulate(Ambisonics(refs, diffraction=False, **FAST)).values()
    (on,) = scene.simulate(Ambisonics(refs, diffraction=True, **FAST)).values()
    np.testing.assert_allclose(on.ir[: int(0.1 * FS)], off.ir[: int(0.1 * FS)], atol=1e-12)


def test_references_not_from_a_package_scale_method_raise():
    scene = make()
    refs = references(scene)
    for ir in refs.values():
        ir.generation = {}
    with pytest.raises(ValueError, match="package scale"):
        scene.simulate(Ambisonics(refs, **FAST))


def test_air_absorption_comes_from_the_references():
    scene = make()
    dry = references(scene, air_absorption=False)
    hybrid = Ambisonics(dry, **FAST)
    (ir,) = scene.simulate(hybrid).values()
    assert ir.generation["air_absorption"] is False
    with pytest.raises(ValueError, match="air_absorption"):
        scene.simulate(Ambisonics(dry, air_absorption=True, **FAST))


def test_a_receiver_outside_the_room_raises():
    scene = make([Receiver("inside", 8.0, 4.0, 1.5), Receiver("outside", 12.0, 4.0, 1.5)])
    with pytest.warns(UserWarning, match="outside"):
        refs = references(scene)
    with pytest.warns(UserWarning, match="outside"), pytest.raises(ValueError, match="outside"):
        scene.simulate(Ambisonics(refs, **FAST))


def test_a_sampling_rate_too_low_for_the_8_khz_octave_raises():
    scene = make()
    scene.sampling_rate = 16000
    with pytest.raises(ValueError, match="22627"):
        scene.simulate(Ambisonics({}, **FAST))


def test_the_seed_is_true_42_by_default_false_unseeded_or_an_int():
    assert Ambisonics().seed == 42 and Ambisonics(seed=True).seed == 42 and Ambisonics(seed=1).seed == 1
    assert Ambisonics(seed=False).seed is None and Ambisonics(seed=None).seed is None

    def response(**kwargs):
        (ir,) = make().simulate(Ambisonics(tail="eyring", **FAST, **kwargs)).values()
        return ir

    default = response().denormalized()
    np.testing.assert_array_equal(response().denormalized(), default)
    np.testing.assert_array_equal(response(seed=42).denormalized(), default)
    assert not np.array_equal(response(seed=1).denormalized(), default)
    unseeded, unseeded_again = response(seed=False), response(seed=False)
    assert unseeded.generation["seed"] is None
    assert not np.array_equal(unseeded.denormalized(), default)
    assert not np.array_equal(unseeded_again.denormalized(), unseeded.denormalized())


def test_decay_db_reaches_the_target():
    scene = make()
    refs = references(scene)
    t30, t20 = Ambisonics(refs, **FAST), Ambisonics(refs, decay_db=20.0, **FAST)
    scene.simulate(t30)
    scene.simulate(t20)
    ((a,), (b,)) = ([r.core.target.rt for r in h.results.values()] for h in (t30, t20))
    assert not np.allclose(a, b)


def test_diffraction_and_the_late_envelope_run():
    scene = make(panel=True)
    refs = references(scene)
    for options in (dict(diffraction=True), dict(late_envelope="directional"), dict(late_envelope="omni")):
        (ir,) = scene.simulate(Ambisonics(refs, **FAST, **options)).values()
        assert np.all(np.isfinite(ir.ir))


# ------------------------------------------------------------------ tails --
@pytest.mark.parametrize("tail", ["reference", "ray_tracing"])
def test_every_tail_meets_the_contract(assert_meets_contract, tail):
    scene = make()
    reference = references(scene) if tail == "reference" else None
    hybrid = Ambisonics(reference=reference, tail=tail, **FAST)
    rirs = scene.simulate(hybrid)
    assert_meets_contract(rirs, SimulationContext.from_scene(scene), hybrid)
    assert all(ir.generation["tail"] == tail for ir in rirs.values())


def test_a_graph_name_a_dict_and_a_method_give_the_same_result():
    scene = make()
    refs = references(scene)
    by_name = scene.simulate(Ambisonics(reference="mono", seed=0, **FAST))
    by_dict = scene.simulate(Ambisonics(reference=refs, seed=0, **FAST))
    by_method = scene.simulate(Ambisonics(reference=Mono(ray_tracing=True, **FAST_RT), seed=0, **FAST))
    for key in by_name:
        np.testing.assert_allclose(by_name[key].denormalized(), by_dict[key].denormalized())
        np.testing.assert_allclose(by_method[key].denormalized(), by_dict[key].denormalized())


@pytest.mark.parametrize(
    "kwargs",
    [
        {"tail": "reference"},
        {"tail": "eyring", "reference": "mono"},
        {"tail": "ray_tracing", "reference": "mono"},
        {"tail": "eyring", "n_slopes": 2},
        {"tail": "air"},
        {"diffraction": "btm"},
        {"image_source_order": -1},
        {"duration": 0.0},
        {"normalization": "FuMa"},
    ],
)
def test_conflicting_settings_raise_at_construction(kwargs):
    with pytest.raises(ValueError):
        Ambisonics(**kwargs)


def test_diffraction_true_is_udfa():
    assert Ambisonics(diffraction=True).diffraction == "udfa"
    assert Ambisonics(diffraction=False).diffraction is None


def test_the_duration_defaults_to_one_and_a_half_times_the_pairs_longest_target_rt():
    method = Ambisonics(tail="eyring", image_source_order=1)
    (ir,) = make().simulate(method).values()
    (result,) = method.results.values()
    assert ir.ir.shape[0] == int(np.ceil(1.5 * float(result.core.target.rt.max()) * ir.sampling_rate))


def test_tail_defaults():
    assert Ambisonics().tail == "ray_tracing"
    assert Ambisonics(reference="mono").tail == "reference"
    assert Ambisonics().late_envelope == "omni" and Ambisonics(reference="data").late_envelope == "omni"
    assert Ambisonics(tail="eyring").late_envelope is False and Ambisonics(tail="sabine").late_envelope is False
    assert Ambisonics(late_envelope=False).late_envelope is False


def test_eyring_in_an_uneven_room_warns():
    scene = make(materials=uneven())
    with pytest.warns(DiffuseFieldWarning):
        scene.simulate(Ambisonics(tail="eyring", **FAST))


def test_an_eyring_tail_takes_no_simulation(monkeypatch):
    scene = make()
    monkeypatch.setattr(Mono, "compute", lambda *a, **k: pytest.fail("ran a reference"))
    hybrid = Ambisonics(tail="sabine", **FAST)
    (ir,) = scene.simulate(hybrid).values()
    assert hybrid.reference_rirs is None
    assert ir.generation["tail"] == "sabine" and ir.generation["air_absorption"] is True


def test_an_image_source_only_reference_warns():
    scene = make()
    scene.simulate(Mono(image_source_order=3, ray_tracing=False))
    with pytest.warns(UserWarning, match="no diffuse decay"):
        scene.simulate(Ambisonics(reference="mono", **FAST))


def test_normalized_references_give_the_raw_level_result():
    scene = make()
    refs = references(scene)
    normalized = scene.simulate(Ambisonics(reference=refs, late_envelope="directional", seed=0, **FAST))
    for ir in refs.values():
        ir.normalized = False
    raw = scene.simulate(Ambisonics(reference=refs, late_envelope="directional", seed=0, **FAST))
    for key in normalized:
        np.testing.assert_allclose(normalized[key].denormalized(), raw[key].denormalized())


def test_btms_runs_in_the_simulator():
    scene = make(panel=True)
    (ir,) = scene.simulate(Ambisonics(tail="eyring", image_source_order=0, duration=0.2, diffraction="btms")).values()
    assert np.all(np.isfinite(ir.ir))


# ---------------------------------------------------------- orientation --
def early_field(direction):
    scene = make([Receiver("rec", 8.0, 4.0, 1.5, direction=direction)])
    hybrid = Ambisonics(tail="eyring", **FAST)
    scene.simulate(hybrid)
    (result,) = hybrid.results.values()
    return result.early


def test_turning_the_receiver_turns_its_ambisonic_response():
    """Facing +y instead of +x: the receiver's X is world Y and its Y (left) is world -X (ACN: W, Y, Z, X)."""
    ahead, turned = early_field([1, 0, 0]), early_field([0, 1, 0])
    np.testing.assert_allclose(turned[:, 0], ahead[:, 0], atol=1e-12)
    np.testing.assert_allclose(turned[:, 2], ahead[:, 2], atol=1e-12)
    np.testing.assert_allclose(turned[:, 3], ahead[:, 1], atol=1e-9 * np.abs(ahead).max())
    np.testing.assert_allclose(turned[:, 1], -ahead[:, 3], atol=1e-9 * np.abs(ahead).max())


def test_the_orientation_is_in_the_responses_metadata(tmp_path):
    from compas_acoustics import ImpulseResponse

    scene = make([Receiver("rec", 8.0, 4.0, 1.5, direction=[0, 1, 0])])
    (ir,) = scene.simulate(Ambisonics(tail="eyring", **FAST)).values()
    loaded = ImpulseResponse.from_wav(ir.to_wav(tmp_path / "h.wav")[0])
    np.testing.assert_allclose(loaded.receiver.rotation, ir.receiver.rotation, atol=1e-12)
    assert (loaded.ambisonic_order, loaded.normalization) == (1, "sn3d")


def test_a_receiver_next_to_a_fully_absorbing_ceiling_runs():
    """Every scan direction of the upward delay line hits the alpha = 1 surface."""
    sky = AcousticMaterial("sky", 1.0, 0.1)
    materials = {"floor": WALL, "ceiling": sky, "left_wall": WALL, "right_wall": WALL, "front_wall": WALL, "back_wall": WALL}
    scene = make([Receiver("rec", 8.0, 4.0, 3.9)], materials=materials)
    (ir,) = scene.simulate(Ambisonics(tail="eyring", duration=0.3)).values()
    assert np.all(np.isfinite(ir.ir))


def test_the_medium_sets_the_hybrids_speed_and_air():
    from compas_acoustics import AirConditions

    air = AirConditions(temperature=0.0, humidity=20.0)
    scene = make()
    (ir,) = scene.simulate(Ambisonics(tail="eyring", air=air, **FAST)).values()
    assert ir.generation["air"] == air.to_dict()
    assert ir.generation["speed_of_sound"] == pytest.approx(air.speed_of_sound)
    distance = float(np.linalg.norm(np.subtract(ir.receiver.position, ir.source.position)))
    expected = round(distance / air.speed_of_sound * FS)
    # Up to 1 ms past the arrival: later, the network's tail has larger samples than the direct sound in this room
    w = ir.denormalized()[: expected + 48, 0]
    assert abs(int(np.argmax(np.abs(w))) - expected) <= 1


def test_the_medium_comes_from_the_references():
    from compas_acoustics import AirConditions

    air = AirConditions(temperature=0.0)
    scene = make()
    refs = references(scene, air=air)
    (ir,) = scene.simulate(Ambisonics(refs, **FAST)).values()
    assert ir.generation["air"] == air.to_dict()
    with pytest.raises(ValueError, match="air"):
        scene.simulate(Ambisonics(refs, air=AirConditions(temperature=25.0), **FAST))
    with pytest.raises(ValueError, match="air"):
        scene.simulate(Ambisonics(references(scene), air=air, **FAST))


def test_a_ray_tracing_tail_runs_its_reference_in_the_medium():
    from compas_acoustics import AirConditions

    air = AirConditions(temperature=0.0)
    scene = make()
    hybrid = Ambisonics(tail="ray_tracing", air=air, **FAST)
    scene.simulate(hybrid)
    assert all(r.generation["air"] == air.to_dict() for r in hybrid.reference_rirs.values())


def test_without_a_medium_the_hybrid_and_its_reference_share_the_default():
    from compas_acoustics import AirConditions

    scene = make()
    (ir,) = scene.simulate(Ambisonics(tail="eyring", **FAST)).values()
    assert ir.generation["air"] == AirConditions().to_dict()
    assert ir.generation["speed_of_sound"] == pytest.approx(AirConditions().speed_of_sound)
    (ir,) = scene.simulate(Ambisonics(references(scene), **FAST)).values()
    assert ir.generation["air"] == AirConditions().to_dict()


def test_the_hybrids_own_reference_lengthens_until_it_holds_its_decay():
    from compas_acoustics.room.raytracing import default_duration

    hard, soft = AcousticMaterial("hard", 0.02, 0.1), AcousticMaterial("soft", 0.9, 0.1)
    scene = make(materials={"floor": soft, "ceiling": soft, "left_wall": hard, "right_wall": hard, "front_wall": hard, "back_wall": hard})
    hybrid = Ambisonics(**FAST)
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        warnings.filterwarnings("ignore", category=DiffuseFieldWarning)
        scene.simulate(hybrid)
    (reference,) = hybrid.reference_rirs.values()
    assert reference.ir.shape[0] / FS > default_duration(SimulationContext.from_scene(scene).geometry, air_absorption=True)


def test_the_hybrids_own_reference_has_the_methods_default_length_whatever_the_output_duration():
    """A ray-traced reference carries its full decay (1.5 x the longest Eyring RT) even when the output is cut short."""
    from compas_acoustics.room.raytracing import default_duration

    scene = make()
    hybrid = Ambisonics(image_source_order=2, duration=0.3)
    scene.simulate(hybrid)
    g = SimulationContext.from_scene(scene).geometry
    expected = default_duration(g, air_absorption=True)
    assert expected > 0.3
    for ir in hybrid.reference_rirs.values():
        assert ir.ir.shape[0] == round(expected * FS)
    (result,) = hybrid.results.values()
    assert result.rir.shape[0] == round(0.3 * FS)


def test_a_reference_shorter_than_its_reverberation_time_warns():
    """pyFDN's fit shrinks its range to what the response has, silently; the hybrid says so."""
    scene = make()
    short = references(scene, duration=0.2)
    with pytest.warns(UserWarning, match="less than 2/3 of its reverberation time"):
        scene.simulate(Ambisonics(short, **FAST))
    scene = make()
    long_enough = references(scene, duration=None)  # the Mono default, 1.5 x the longest Eyring RT
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        warnings.filterwarnings("ignore", category=DiffuseFieldWarning)
        scene.simulate(Ambisonics(long_enough, **FAST))


def test_a_reference_of_two_thirds_of_its_reverberation_time_is_long_enough():
    """The decay fit, like T30, ends 35 dB down; the late envelope holds its gain from 20 dB down."""
    from types import SimpleNamespace

    target = SimpleNamespace(rt=np.array([2.0, 2.5]))
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        Ambisonics()._check_reference_length(1.9, target, 1.3, "rec, src")
    with pytest.warns(UserWarning, match=r"less than 2/3 of its reverberation time \(2.5 s\).* at least 3.8 s"):
        Ambisonics()._check_reference_length(1.5, target, 1.3, "rec, src")


HARD = AcousticMaterial("hard", 0.01, 0.05)


def test_an_eyring_tail_in_a_room_above_12_s_warns_nothing_about_ray_tracing():
    from compas_acoustics.room.diffuse import predicted_rt

    scene = make(materials=HARD)
    assert predicted_rt(RoomGeometry.from_scene(scene), warn=False).max() > 12.0
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        scene.simulate(Ambisonics(tail="eyring", image_source_order=1, duration=0.3))
    assert not [w for w in record if "ray tracing" in str(w.message)]


def test_a_reference_shorter_than_two_thirds_of_an_rt_above_12_s_warns_with_that_rt():
    import re

    from compas_acoustics.room.diffuse import predicted_rt

    scene = make(materials=HARD)
    short = references(scene, duration=1.0)
    with pytest.warns(UserWarning, match="less than 2/3 of its reverberation time") as record:
        scene.simulate(Ambisonics(short, image_source_order=1, duration=0.3))
    (message,) = [str(w.message) for w in record if "less than 2/3 of its reverberation time" in str(w.message)]
    rt = float(re.search(r"reverberation time \(([0-9.]+) s\)", message).group(1))
    assert rt >= round(float(predicted_rt(RoomGeometry.from_scene(scene), warn=False).max()), 1) > 12.0


def test_a_receiver_without_a_visible_image_beside_a_visible_one_renders():
    """A reference of order 0 has no image at the receiver behind the panel while the other receiver sees the source."""
    scene = make([Receiver("behind", 8.0, 4.0, 1.5), Receiver("in_view", 2.0, 7.0, 1.5)], panel=True)
    reference = Mono(image_source_order=0, n_rays=2000, duration=0.5, seed=1)
    rirs = scene.simulate(Ambisonics(reference, image_source_order=1, sh_order=1, duration=0.5))
    for ir in rirs.values():
        assert np.all(np.isfinite(ir.ir))
        assert np.sum(ir.ir[int(0.03 * FS) :, 0] ** 2) > 0.0


@pytest.mark.parametrize("given", [False, True])
def test_the_directional_histograms_trace_the_rays_of_a_1_db_share_error(monkeypatch, given):
    """Vorlaender (2020, Eq. 11.20) per direction, at least MIN_RAYS: in the reference the method traces itself, whose
    rays the histograms share, or in the histograms' own run with a given reference."""
    import compas_acoustics.room.raytracing as raytracing
    import compas_acoustics.simulation_methods.pra as mono
    from compas_acoustics.hybrid.envelope import histogram_rays
    from compas_acoustics.hybrid.late import fdn_directions

    class Traced(Exception):
        pass

    def spy(*args, **kwargs):
        if kwargs.get("directions", True) is not None:
            raise Traced(kwargs["n_rays"])
        return run(*args, **kwargs)

    run = mono.ray_traced_run
    monkeypatch.setattr(raytracing, "directional_histograms", spy)
    monkeypatch.setattr(mono, "ray_traced_run", spy)
    grid = fdn_directions(96).cartesian
    speed = AirConditions().speed_of_sound
    for size, below in (((10, 8, 4), True), ((30, 20, 8), False)):
        scene = Scene("test")
        scene.add_acoustic_objects(AcousticObjectsLibrary.shoebox_room(*size, materials=WALL))
        scene.add_source(Source("src", 2.0, 4.0, 1.5))
        scene.add_receiver(Receiver("rec", 8.0, 4.0, 1.5))
        with pytest.raises(Traced) as traced:
            scene.simulate(Ambisonics(Mono(n_rays=2000, duration=0.5) if given else None, sh_order=1, duration=0.5, late_envelope="directional", late_field="fdn"))
        wanted = histogram_rays(float(np.prod(size)), 0.5, speed, grid)
        assert (wanted < MIN_RAYS) == below
        assert traced.value.args[0] == max(wanted, MIN_RAYS)


def test_the_moments_take_the_rays_of_a_given_mono_at_the_methods_order(monkeypatch):
    """A Mono reference at the method's order traces its own rays sorted on the moments' grid, with its own count."""
    import compas_acoustics.simulation_methods.pra as mono
    from compas_acoustics.hybrid.moments import moment_grid

    seen = {}
    run = mono.ray_traced_run

    def spy(*args, **kwargs):
        seen.update(n_rays=kwargs["n_rays"], directions=kwargs.get("directions"))
        return run(*args, **kwargs)

    monkeypatch.setattr(mono, "ray_traced_run", spy)
    scene = make()
    (ir,) = scene.simulate(Ambisonics(Mono(n_rays=2000, duration=0.5), sh_order=1, duration=0.5, late_envelope="directional")).values()
    assert seen["n_rays"] == 2000 and len(seen["directions"]) == 1 and seen["directions"][0].shape == moment_grid().shape
    assert ir.generation["late_energy"] == "rays"


def test_the_generation_record_holds_the_histogram_rays():
    from compas_acoustics.hybrid.envelope import histogram_rays
    from compas_acoustics.hybrid.late import fdn_directions

    scene = make()
    (ir,) = scene.simulate(Ambisonics(references(scene), sh_order=1, duration=0.5, late_envelope="directional")).values()
    assert ir.generation["histogram_rays"] == max(histogram_rays(320.0, 0.5, AirConditions().speed_of_sound, fdn_directions(96).cartesian), MIN_RAYS)
    for mode in ("omni", False):
        (ir,) = scene.simulate(Ambisonics(references(scene), sh_order=1, duration=0.5, late_envelope=mode)).values()
        assert ir.generation["histogram_rays"] is None


def test_a_silent_reference_is_refused_by_the_names_of_its_pair():
    """Images alone leave the receiver behind the panel without any energy at order 0: no decay to fit."""
    scene = make([Receiver("behind", 8.0, 4.0, 1.5), Receiver("in_view", 2.0, 7.0, 1.5)], panel=True)
    reference = Mono(ray_tracing=False, image_source_order=0)
    with pytest.raises(ValueError, match="reference of 'behind' and 'src' is silent"):
        scene.simulate(Ambisonics(reference, image_source_order=1, sh_order=1, duration=0.5))


@pytest.mark.parametrize("scattering", [False, True])
def test_a_traced_reference_shares_its_search_and_rays_with_the_early_field_and_the_histograms(scattering):
    """One search and one trace give what a given reference with its own histogram run gives, bit for bit: the
    reference is Mono at the method's order, seed and histogram rays, its rays those of the histograms."""
    from compas_acoustics.hybrid.envelope import histogram_rays
    from compas_acoustics.hybrid.late import fdn_directions

    settings = dict(image_source_order=2, seed=7)
    scene = make(panel=True)
    shared = Ambisonics(sh_order=1, scattering=scattering, late_envelope="directional", **settings)
    rirs = scene.simulate(shared)
    n_rays = max(histogram_rays(320.0, 0.5, AirConditions().speed_of_sound, fdn_directions(96).cartesian), MIN_RAYS)
    given = Ambisonics(Mono(n_rays=n_rays, **settings), sh_order=1, scattering=scattering, late_envelope="directional", **settings)
    again = make(panel=True).simulate(given)
    (ir,) = rirs.values()
    (other,) = again.values()
    (reference,) = shared.reference_rirs.values()
    (expected,) = given.reference_rirs.values()
    assert reference.generation["image_source_order"] == 2 and reference.generation["seed"] == 7
    assert reference.generation["n_rays"] == ir.generation["histogram_rays"] == n_rays
    assert np.array_equal(reference.ir, expected.ir)
    assert np.array_equal(ir.ir, other.ir)


@pytest.mark.parametrize("scattering", [False, True])
def test_a_given_mono_reference_at_the_methods_order_lends_its_search_to_the_early_field(scattering):
    """Bit for bit what its responses given as a dict give, with which the early field searches its own room; the
    FDN's late field too. The moments' late field takes the Mono's rays, which the dict does not have."""
    settings = dict(sh_order=1, image_source_order=3, scattering=scattering, duration=0.5, seed=0)
    scene = make(panel=True)
    by_dict = scene.simulate(Ambisonics(references(scene), late_field="fdn", **settings))
    by_method = scene.simulate(Ambisonics(Mono(**FAST_RT), late_field="fdn", **settings))
    for key, ir in by_dict.items():
        assert np.array_equal(by_method[key].ir, ir.ir)
    dict_method, mono_method = Ambisonics(references(scene), **settings), Ambisonics(Mono(**FAST_RT), **settings)
    by_dict, by_method = scene.simulate(dict_method), scene.simulate(mono_method)
    for key, ir in by_dict.items():
        assert ir.generation["late_energy"] == "network" and by_method[key].generation["late_energy"] == "rays"
        np.testing.assert_array_equal(dict_method.results[key].early, mono_method.results[key].early)


@pytest.mark.parametrize("method", [Ambisonics, Binaural])
def test_pairs_rendered_together_on_threads_are_each_as_rendered_alone(method):
    sources = [Source("src", 2.0, 4.0, 1.5), Source("other", 6.0, 2.0, 1.2)]
    receivers = [Receiver("a", 8.0, 4.0, 1.5), Receiver("b", 3.0, 6.0, 1.2, direction=[0, 1, 0])]

    def scene_of(sources, receivers):
        scene = Scene("test")
        scene.add_acoustic_objects(AcousticObjectsLibrary.shoebox_room(10, 8, 4, materials=WALL))
        for item in sources:
            scene.add_source(item)
        for item in receivers:
            scene.add_receiver(item)
        return scene

    together = method(tail="eyring", seed=0, **FAST)
    rirs = scene_of(sources, receivers).simulate(together)
    keys = [SimulationContext.key(source, receiver) for source in sources for receiver in receivers]
    assert list(rirs) == list(together.results) == keys
    for source in sources:
        for receiver in receivers:
            (alone,) = scene_of([source], [receiver]).simulate(method(tail="eyring", seed=0, **FAST)).values()
            assert np.array_equal(rirs[SimulationContext.key(source, receiver)].ir, alone.ir)


@pytest.mark.parametrize("order, searches", [(3, 1), (2, 2)])
def test_the_early_field_searches_again_only_for_a_reference_at_another_order(monkeypatch, order, searches):
    import pyroomacoustics as pra

    calls = []
    search = pra.Room.image_source_model
    monkeypatch.setattr(pra.Room, "image_source_model", lambda room: calls.append(room) or search(room))
    make().simulate(Ambisonics(Mono(**FAST_RT), sh_order=1, image_source_order=order, duration=0.5))
    assert len(calls) == searches


def test_omni_traces_no_histogram(monkeypatch):
    """With a given reference no directional histogram is traced; with a ray-traced one the reference traces none."""
    import compas_acoustics.room.raytracing as raytracing

    def refuse(*args, **kwargs):
        raise AssertionError("a directional histogram was traced")

    scene = make()
    refs = references(scene)
    monkeypatch.setattr(raytracing, "directional_histograms", refuse)
    for late_field in ("fdn", "moments"):
        for method in (Ambisonics(refs, late_envelope="omni", late_field=late_field, **FAST), Ambisonics(tail="ray_tracing", late_envelope="omni", late_field=late_field, **FAST)):
            (ir,) = scene.simulate(method).values()
            (result,) = method.results.values()
            assert result.core.directional_envelope is None
            assert result.core.late_envelope is not None
            # the moments take the energy of their own reference's bins
            sorted_rays = late_field == "moments" and method.tail == "ray_tracing"
            assert (ir.generation["histogram_rays"] is None) != sorted_rays


def test_omni_keeps_the_early_field():
    scene = make()
    refs = references(scene)
    directional, omni = (Ambisonics(refs, late_envelope=mode, seed=0, **FAST) for mode in ("directional", "omni"))
    scene.simulate(directional)
    scene.simulate(omni)
    (d,), (o,) = directional.results.values(), omni.results.values()
    np.testing.assert_array_equal(d.early, o.early)
    fs = d.core.fs
    a = int(round((np.linalg.norm(d.core.pair.source - d.core.pair.receiver) / d.core.pair.speed_of_sound + 0.08) * fs))
    level = 10 * np.log10(np.sum(o.rir[a:, 0] ** 2) / np.sum(d.rir[a:, 0] ** 2))
    assert abs(level) < 1.0
    assert not np.allclose(d.late, o.late)


def test_the_default_ray_count_is_the_directions_rule_on_the_96_line_grid_at_least_min_rays():
    """Written out for Mono, which runs without the grid's packages: the hybrid's rule on its default grid."""
    from compas_acoustics.hybrid.envelope import histogram_rays
    from compas_acoustics.hybrid.late import fdn_directions
    from compas_acoustics.room.raytracing import default_rays

    grid = fdn_directions(96).cartesian
    grid = grid / np.linalg.norm(grid, axis=1, keepdims=True)
    speed = AirConditions().speed_of_sound
    for volume in (60.0, 320.0, 1000.0, 1100.0, 4800.0, 25000.0):
        assert default_rays(volume, 0.5, speed) == max(histogram_rays(volume, 0.5, speed, grid), MIN_RAYS)


def test_n_rays_sets_every_trace_the_method_makes_itself(monkeypatch):
    import compas_acoustics.room.raytracing as raytracing
    import compas_acoustics.simulation_methods.pra as mono

    seen = []
    run, histograms = mono.ray_traced_run, raytracing.directional_histograms

    def spy_run(*args, **kwargs):
        seen.append(kwargs["n_rays"])
        return run(*args, **kwargs)

    def spy_histograms(*args, **kwargs):
        seen.append(kwargs["n_rays"])
        return histograms(*args, **kwargs)

    monkeypatch.setattr(mono, "ray_traced_run", spy_run)
    monkeypatch.setattr(raytracing, "directional_histograms", spy_histograms)
    make().simulate(Ambisonics(sh_order=1, n_rays=2500, **FAST))
    scene = make()
    scene.simulate(Ambisonics(Mono(n_rays=2000, duration=0.5, seed=1), sh_order=1, n_rays=2500, late_envelope="directional", **FAST))
    assert seen == [2500, 2000, 2500]


@pytest.mark.parametrize("settings", [dict(tail="eyring"), dict(reference={}, late_envelope="omni"), dict(reference={}, late_envelope=False)])
def test_n_rays_without_a_trace_of_the_methods_own_is_refused(settings):
    with pytest.raises(ValueError, match="n_rays"):
        Ambisonics(n_rays=2500, **settings)


def test_omni_traces_its_reference_with_the_rays_of_the_directions_rule(monkeypatch):
    """The reference of "omni" takes the ray count of the directions' rule, as that of "directional" does, and no
    directions to sort the rays by."""
    import compas_acoustics.simulation_methods.pra as mono
    from compas_acoustics.hybrid.envelope import histogram_rays
    from compas_acoustics.hybrid.late import fdn_directions

    seen = {}
    run = mono.ray_traced_run

    def spy(*args, **kwargs):
        seen.update(n_rays=kwargs["n_rays"], directions=kwargs.get("directions"))
        return run(*args, **kwargs)

    monkeypatch.setattr(mono, "ray_traced_run", spy)
    scene = Scene("test")
    scene.add_acoustic_objects(AcousticObjectsLibrary.shoebox_room(5, 4, 3, materials=WALL))
    scene.add_source(Source("src", 1.0, 2.0, 1.5))
    scene.add_receiver(Receiver("rec", 4.0, 2.0, 1.5))
    scene.simulate(Ambisonics(sh_order=1, duration=0.5, late_envelope="omni", late_field="fdn"))
    assert seen["n_rays"] == max(histogram_rays(60.0, 0.5, AirConditions().speed_of_sound, fdn_directions(96).cartesian), MIN_RAYS)
    assert seen["directions"] is None
    # the moments take the energy of the same rays from the reference's own bins, unsorted
    (ir,) = scene.simulate(Ambisonics(sh_order=1, duration=0.5, late_envelope="omni")).values()
    assert seen["n_rays"] == max(histogram_rays(60.0, 0.5, AirConditions().speed_of_sound, fdn_directions(96).cartesian), MIN_RAYS)
    assert seen["directions"] is None and ir.generation["late_energy"] == "rays"


def test_omni_and_directional_share_their_reference():
    scene = Scene("test")
    scene.add_acoustic_objects(AcousticObjectsLibrary.shoebox_room(5, 4, 3, materials=WALL))
    scene.add_source(Source("src", 1.0, 2.0, 1.5))
    scene.add_receiver(Receiver("rec", 4.0, 2.0, 1.5))
    omni, directional = (Ambisonics(sh_order=1, duration=0.5, seed=3, late_envelope=mode) for mode in ("omni", "directional"))
    scene.simulate(omni)
    scene.simulate(directional)
    (a,), (b,) = omni.reference_rirs.values(), directional.reference_rirs.values()
    np.testing.assert_array_equal(a.ir, b.ir)
    assert a.generation["n_rays"] == b.generation["n_rays"]
