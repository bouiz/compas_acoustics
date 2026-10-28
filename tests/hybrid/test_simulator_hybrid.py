import warnings

import numpy as np
import pyFDN
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
from compas_acoustics.simulation_methods import Ambisonics
from compas_acoustics.simulation_methods import Mono
from compas_acoustics.simulation_methods import SimulationContext
from compas_acoustics.room.geometry import RoomGeometry
from compas_acoustics.room.geometry import SurfaceMaterial

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


# ------------------------------------------------------------ v1 Task 15 --
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


def test_second_order_n3d_has_nine_channels():
    scene = make()
    (ir,) = scene.simulate(Ambisonics(references(scene), sh_order=2, normalization="N3D", **FAST)).values()
    assert ir.ir.shape[1] == 9
    assert ir.normalization == "n3d"


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
    scene = l_room()
    assert RoomGeometry.from_scene(scene).is_closed
    refs = references(scene)
    (off,) = scene.simulate(Ambisonics(refs, diffraction=False, normalize=False, **FAST)).values()
    with warnings.catch_warnings():
        warnings.simplefilter("error", ShellEdgeWarning)
        (on,) = scene.simulate(Ambisonics(refs, diffraction=True, normalize=False, **FAST)).values()
    # the wall hides the direct sound, so the window starts at its geometric arrival
    start = int(np.linalg.norm(np.subtract(off.source.position, off.receiver.position)) / 343.0 * FS)
    window = slice(start, start + int(0.02 * FS))
    e_off, e_on = np.sum(off.ir[window] ** 2), np.sum(on.ir[window] ** 2)
    # raw level, seeded reference: the window energy rises by 21 % (1.98e-4 -> 2.41e-4); 10 % leaves half of that as margin
    assert e_on > 1.10 * e_off


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


def test_the_reference_is_on_the_scale_of_the_hybrid():
    """Upstream's run-time scale guard (envelope.SCALE_WARN_DB) stays silent."""
    scene = make()
    refs = references(scene)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        scene.simulate(Ambisonics(refs, late_envelope=True, **FAST))
    assert not [w for w in caught if "differs from the hybrid" in str(w.message)]


def test_diffraction_and_the_late_envelope_run():
    scene = make(panel=True)
    refs = references(scene)
    for options in (dict(diffraction=True), dict(late_envelope=True)):
        (ir,) = scene.simulate(Ambisonics(refs, **FAST, **options)).values()
        assert np.all(np.isfinite(ir.ir))


@pytest.mark.slow
def test_the_w_channel_decays_at_the_target_rate():
    scene = make()
    hybrid = Ambisonics(references(scene, n_rays=20000), sh_order=1, image_source_order=3)
    ((key, ir),) = scene.simulate(hybrid).items()
    rt_w = np.asarray(pyFDN.estimate_rt_bands(ir.denormalized()[:, 0], FS)[0])
    target = hybrid.results[key].core.target.rt[:, 0]
    # 6 %: the 250 Hz band reads 4.2-5.1 % short with the envelope off, with its level step alone and with the
    # directional shares (seed True), and the 125 Hz band moves 5 % between seeds; the network, not the envelope
    np.testing.assert_allclose(rt_w[2:7], target[2:7], rtol=0.06)


# ------------------------------------------------------- revision 2, tails --
@pytest.mark.parametrize("tail", ["reference", "ray_tracing", "eyring"])
def test_every_tail_meets_the_contract(assert_meets_contract, tail):
    scene = make()
    reference = references(scene) if tail == "reference" else None
    hybrid = Ambisonics(reference=reference, tail=tail, **FAST)
    rirs = scene.simulate(hybrid)
    assert_meets_contract(rirs, SimulationContext.from_scene(scene), hybrid)
    assert all(ir.generation["tail"] == tail for ir in rirs.values())


@pytest.mark.slow
def test_no_reference_runs_its_own_ray_tracing():
    scene = make()
    hybrid = Ambisonics(**FAST)
    scene.simulate(hybrid)
    assert hybrid.tail == "ray_tracing"
    assert all(ir.generation["ray_tracing"] and ir.generation["n_rays"] == 20000 for ir in hybrid.reference_rirs.values())
    assert "mono" not in scene.simulation_methods()  # not written to the graph


def test_a_graph_name_a_dict_and_a_method_give_the_same_result():
    scene = make()
    refs = references(scene)
    by_name = scene.simulate(Ambisonics(reference="mono", seed=0, **FAST))
    by_dict = scene.simulate(Ambisonics(reference=refs, seed=0, **FAST))
    by_method = scene.simulate(Ambisonics(reference=Mono(ray_tracing=True, **FAST_RT), seed=0, **FAST))
    for key in by_name:
        np.testing.assert_allclose(by_name[key].denormalized(), by_dict[key].denormalized())
        np.testing.assert_allclose(by_method[key].denormalized(), by_dict[key].denormalized())


@pytest.mark.parametrize("kwargs", [
    {"tail": "reference"},
    {"tail": "eyring", "reference": "mono"},
    {"tail": "ray_tracing", "reference": "mono"},
    {"tail": "eyring", "n_slopes": 2},
    {"tail": "eyring", "late_envelope": True},
    {"tail": "air"},
    {"diffraction": "btm"},
    {"normalization": "FuMa"},
])
def test_conflicting_settings_raise_at_construction(kwargs):
    with pytest.raises(ValueError):
        Ambisonics(**kwargs)


def test_tail_defaults():
    assert Ambisonics().tail == "ray_tracing"
    assert Ambisonics(reference="mono").tail == "reference"
    assert Ambisonics().late_envelope and Ambisonics(reference="data").late_envelope
    assert not Ambisonics(tail="eyring").late_envelope and not Ambisonics(tail="sabine").late_envelope
    assert not Ambisonics(late_envelope=False).late_envelope


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
    normalized = scene.simulate(Ambisonics(reference=refs, late_envelope=True, seed=0, **FAST))
    for ir in refs.values():
        ir.normalized = False
    raw = scene.simulate(Ambisonics(reference=refs, late_envelope=True, seed=0, **FAST))
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


# ------------------------------------------------------------ late level --
@pytest.mark.parametrize("dims, alpha, src, rec", [
    ((10, 8, 4), 0.15, (2.5, 3.0, 1.5), (7.5, 5.5, 1.2)),
    ((6, 5, 3), 0.1, (1.5, 1.5, 1.5), (4.5, 3.5, 1.2)),
])
def test_the_late_level_matches_an_exact_specular_room(dims, alpha, src, rec):
    """A shoebox with specular walls has an exact answer: its image sources to high order.

    Given that answer's decay, the hybrid's late energy (after 80 ms, 1 kHz band), averaged over four
    seeds of the network, must match it within 2 dB. The network is driven by the last rendered order,
    as in Wendt et al. (2014) and RAZR, so its level is that order's energy, which sits below the energy
    trend of the higher orders: measured over eight seeds, 0.5 dB below in the 10 x 8 x 4 m room and
    1.6 dB below in the 6 x 5 x 3 m room of alpha 0.1, where RAZR 0.96.3 itself is 3.7 dB below (seed
    spread of this network 0.1 to 0.4 dB). pyroomacoustics' ray tracer is not the yardstick: measured
    here it sits 1 to 2.6 dB below this answer without scattering, and 1.5 to 3.6 dB above Barron's
    revised theory with scattering 0.5.
    """
    import pyroomacoustics as pra
    from scipy.signal import butter
    from scipy.signal import sosfiltfilt

    from compas_acoustics import ImpulseResponse

    room = pra.ShoeBox(dims, fs=FS, materials=pra.Material(alpha), max_order=100, air_absorption=False)
    room.add_source(list(src))
    room.add_microphone(np.array(rec, dtype=float).reshape(3, 1))
    room.compute_rir()
    exact = room.rir[0][0][pra.constants.get("frac_delay_length") // 2:] / (4 * np.pi)

    scene = Scene("exact")
    scene.add_acoustic_objects(AcousticObjectsLibrary.shoebox_room(*dims, materials=AcousticMaterial("m", alpha, 0.0)))
    source, receiver = Source("s", *src), Receiver("r", *rec)
    scene.add_source(source)
    scene.add_receiver(receiver)
    reference = {f"{receiver.guid}_{source.guid}": ImpulseResponse("exact", source, receiver, exact, FS,
                                                                     generation={"scale": "package", "air_absorption": False})}
    sos = butter(4, [1000 / np.sqrt(2), 1000 * np.sqrt(2)], btype="band", fs=FS, output="sos")

    def late_energy(x, start):
        return np.sum(sosfiltfilt(sos, np.concatenate([x, np.zeros(FS // 2)]))[start:] ** 2)

    levels = []
    for seed in range(4):
        (ir,) = scene.simulate(Ambisonics(reference=reference, seed=seed)).values()
        start = ir.direct_sound_index() + int(0.08 * FS)
        levels.append(10 * np.log10(late_energy(ir.denormalized()[:, 0], start) / late_energy(exact, start)))
    assert np.mean(levels) == pytest.approx(0.0, abs=2.0)


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


def test_the_hybrid_runs_without_pytorch_and_flamo():
    # pyFDN requires flamo and PyTorch for its training modules only; the worker installs it without them
    import os
    import subprocess
    import sys

    script = """
import importlib.abc, sys, warnings
warnings.simplefilter("ignore")
class Missing(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name.split(".")[0] in ("torch", "torchaudio", "flamo", "nnAudio"):
            raise ImportError(name + " is not installed")
sys.meta_path.insert(0, Missing())
import compas_acoustics as ca
from compas_acoustics.simulation_methods import Ambisonics, Binaural
s = ca.Scene("b")
s.add_acoustic_objects(ca.AcousticObjectsLibrary.shoebox_room(6, 4, 3, materials=ca.AcousticMaterial("w", 0.2, 0.1)))
s.add_source(ca.Source("s", 1.5, 1.5, 1.6))
s.add_receiver(ca.Receiver("r", 4.5, 2.5, 1.2))
(binaural,) = s.simulate(Binaural(tail="eyring", image_source_order=1, diffraction="udfa")).values()
(ir,) = s.simulate(Ambisonics(tail="eyring", image_source_order=1, diffraction="udfa")).values()
print(binaural.n_channels)
print(ir.n_channels)
"""
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, env=dict(os.environ), timeout=600)
    assert result.returncode == 0, result.stderr[-2000:]
    assert result.stdout.strip().splitlines()[-2:] == ["2", "4"]



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
