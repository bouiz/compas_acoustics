import warnings

import numpy as np
import pyFDN
import pytest
from compas.datastructures import Mesh

from compas_acoustics import AcousticMaterial
from compas_acoustics import AcousticObject
from compas_acoustics import AcousticObjectsLibrary
from compas_acoustics import Receiver
from compas_acoustics import Scene
from compas_acoustics import Source
from compas_acoustics.diffuse import DiffuseFieldWarning
from compas_acoustics.methods import Hybrid
from compas_acoustics.methods import Pyroomacoustics
from compas_acoustics.methods import SimulationContext

FS = 48000
WALL = AcousticMaterial("wall", 0.2, 0.1)
FAST = dict(image_source_order=2, duration=0.5)
FAST_RT = dict(image_source_order=3, n_rays=2000, duration=1.0, seed=1)


def make(receivers=None, materials=WALL, leaves=6, panel=False):
    scene = Scene("test")
    room = AcousticObjectsLibrary.shoebox_room(10, 8, 4, materials=materials)
    for leaf in list(room.iter_acoustic_objects())[:leaves]:
        scene.add_acoustic_object(leaf)
    if panel:
        mesh = Mesh.from_vertices_and_faces([[5, 3, 0.5], [5, 5, 0.5], [5, 5, 2.5], [5, 3, 2.5]], [[0, 1, 2, 3]])
        scene.add_acoustic_object(AcousticObject(mesh, WALL, name="panel", role="obstacle"))
    scene.add_source(Source("src", 2.0, 4.0, 1.5))
    for receiver in receivers or [Receiver("rec", 8.0, 4.0, 1.5)]:
        scene.add_receiver(receiver)
    return scene


def references(scene, **kwargs):
    return scene.simulate(Pyroomacoustics(ray_tracing=True, **dict(FAST_RT, **kwargs)))


def uneven():
    carpet = AcousticMaterial("carpet", 0.9, 0.1)
    return {"floor": carpet, "ceiling": carpet, "left_wall": WALL, "right_wall": WALL, "front_wall": WALL, "back_wall": WALL}


# ------------------------------------------------------------ v1 Task 15 --
def test_one_ambisonic_response_per_pair():
    scene = make([Receiver("a", 8.0, 4.0, 1.5), Receiver("b", 3.0, 6.0, 1.2, direction=[0, 1, 0])])
    hybrid = Hybrid(references(scene), sh_order=1, **FAST)
    rirs = scene.simulate(hybrid)
    assert len(rirs) == 2
    for key, ir in rirs.items():
        assert ir.ir.shape == (int(0.5 * FS), 4)
        assert (ir.ambisonic_order, ir.normalisation, ir.channel_order) == (1, "sn3d", "acn")
        assert ir.sample_rate == FS
        assert scene.get_impulse_response(ir.source, ir.receiver, attribute="hybrid") is ir
        assert hybrid.results[key].normalization == "SN3D"
    assert scene.impulse_responses("hybrid") == rirs


def test_the_mono_responses_stay_in_the_graph():
    scene = make()
    mono = references(scene)
    scene.simulate(Hybrid(mono, **FAST))
    (ir,) = mono.values()
    assert scene.get_impulse_response(ir.source, ir.receiver) is ir
    assert scene.get_impulse_response(ir.source, ir.receiver, attribute="pyroomacoustics") is ir
    assert scene.impulse_responses("data") == mono


def test_second_order_n3d_has_nine_channels():
    scene = make()
    (ir,) = scene.simulate(Hybrid(references(scene), sh_order=2, normalization="N3D", **FAST)).values()
    assert ir.ir.shape[1] == 9
    assert ir.normalisation == "n3d"


def test_references_at_another_rate_raise():
    scene = make()
    refs = references(scene)
    for ir in refs.values():
        ir.sample_rate = 16000
    with pytest.raises(ValueError, match="sample rate"):
        scene.simulate(Hybrid(refs, **FAST))


def test_a_multichannel_reference_raises():
    scene = make()
    refs = references(scene)
    for ir in refs.values():
        ir.ir = np.stack([ir.denormalized(), ir.denormalized()], axis=1)
    with pytest.raises(ValueError, match="mono"):
        scene.simulate(Hybrid(refs, **FAST))


def test_references_missing_a_pair_raise():
    scene = make()
    with pytest.raises(ValueError, match="no response"):
        scene.simulate(Hybrid({}, **FAST))


def test_an_open_room_raises():
    with pytest.warns(UserWarning):
        scene = make(leaves=5)
        with pytest.raises(ValueError, match="close"):
            scene.simulate(Hybrid({}, **FAST))


def test_a_vertical_receiver_raises_before_any_work(monkeypatch):
    scene = make([Receiver("rec", 8.0, 4.0, 1.5, direction=[0, 0, 1])])
    monkeypatch.setattr(Pyroomacoustics, "compute", lambda *a, **k: pytest.fail("worked first"))
    with pytest.raises(ValueError, match="parallel"):
        scene.simulate(Hybrid(**FAST))


def test_a_fully_absorbing_surface_runs():
    window = AcousticMaterial("window", 1.0, 0.05)
    materials = {"floor": WALL, "ceiling": window, "left_wall": WALL, "right_wall": WALL, "front_wall": WALL, "back_wall": WALL}
    scene = make(materials=materials)
    (ir,) = scene.simulate(Hybrid(references(scene), **FAST)).values()
    assert np.all(np.isfinite(ir.ir))


def test_diffraction_needs_obstacles():
    scene = make()
    with pytest.raises(ValueError, match="obstacle"):
        scene.simulate(Hybrid({}, diffraction=True, **FAST))


def test_references_not_from_a_package_scale_method_raise():
    scene = make()
    refs = references(scene)
    for ir in refs.values():
        ir.generation = {}
    with pytest.raises(ValueError, match="package scale"):
        scene.simulate(Hybrid(refs, **FAST))


def test_air_absorption_comes_from_the_references():
    scene = make()
    dry = references(scene, air_absorption=False)
    hybrid = Hybrid(dry, **FAST)
    (ir,) = scene.simulate(hybrid).values()
    assert ir.generation["air_absorption"] is False
    with pytest.raises(ValueError, match="air_absorption"):
        scene.simulate(Hybrid(dry, air_absorption=True, **FAST))


def test_a_receiver_outside_the_room_raises():
    scene = make([Receiver("inside", 8.0, 4.0, 1.5), Receiver("outside", 12.0, 4.0, 1.5)])
    with pytest.warns(UserWarning, match="outside"):
        refs = references(scene)
    with pytest.warns(UserWarning, match="outside"), pytest.raises(ValueError, match="outside"):
        scene.simulate(Hybrid(refs, **FAST))


def test_a_sample_rate_too_low_for_the_8_khz_octave_raises():
    scene = make()
    with pytest.raises(ValueError, match="22627"):
        scene.simulate(Hybrid({}, **FAST), sampling_rate=16000)


def test_decay_db_reaches_the_target():
    scene = make()
    refs = references(scene)
    t30, t20 = Hybrid(refs, **FAST), Hybrid(refs, decay_db=20.0, **FAST)
    scene.simulate(t30)
    scene.simulate(t20)
    ((a,), (b,)) = ([r.target.rt for r in h.results.values()] for h in (t30, t20))
    assert not np.allclose(a, b)


def test_the_reference_is_on_the_scale_of_the_hybrid():
    """Upstream's run-time scale guard (envelope.SCALE_WARN_DB) stays silent."""
    scene = make()
    refs = references(scene)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        scene.simulate(Hybrid(refs, late_envelope=True, **FAST))
    assert not [w for w in caught if "differs from the hybrid" in str(w.message)]


def test_diffraction_and_the_late_envelope_run():
    scene = make(panel=True)
    refs = references(scene)
    for options in (dict(diffraction=True), dict(late_envelope=True)):
        (ir,) = scene.simulate(Hybrid(refs, **FAST, **options)).values()
        assert np.all(np.isfinite(ir.ir))


@pytest.mark.slow
def test_the_w_channel_decays_at_the_target_rate():
    scene = make()
    hybrid = Hybrid(references(scene, n_rays=20000), sh_order=1, image_source_order=3)
    ((key, ir),) = scene.simulate(hybrid).items()
    rt_w = np.asarray(pyFDN.estimate_rt_bands(ir.denormalized()[:, 0], FS)[0])
    target = hybrid.results[key].target.rt[:, 0]
    np.testing.assert_allclose(rt_w[2:7], target[2:7], rtol=0.05)


# ------------------------------------------------------- revision 2, tails --
@pytest.mark.parametrize("tail", ["reference", "ray_tracing", "eyring"])
def test_every_tail_meets_the_contract(assert_meets_contract, tail):
    scene = make()
    reference = references(scene) if tail == "reference" else None
    hybrid = Hybrid(reference=reference, tail=tail, **FAST)
    rirs = scene.simulate(hybrid)
    assert_meets_contract(rirs, SimulationContext.from_scene(scene), hybrid)
    assert all(ir.generation["tail"] == tail for ir in rirs.values())


@pytest.mark.slow
def test_no_reference_runs_its_own_ray_tracing():
    scene = make()
    hybrid = Hybrid(**FAST)
    scene.simulate(hybrid)
    assert hybrid.tail == "ray_tracing"
    assert all(ir.generation["ray_tracing"] and ir.generation["n_rays"] == 20000 for ir in hybrid.reference_rirs.values())
    assert "pyroomacoustics" not in scene.simulation_methods()  # not written to the graph


def test_a_graph_name_a_dict_and_a_method_give_the_same_result():
    scene = make()
    refs = references(scene)
    by_name = scene.simulate(Hybrid(reference="pyroomacoustics", seed=0, **FAST))
    by_dict = scene.simulate(Hybrid(reference=refs, seed=0, **FAST))
    by_method = scene.simulate(Hybrid(reference=Pyroomacoustics(ray_tracing=True, **FAST_RT), seed=0, **FAST))
    for key in by_name:
        np.testing.assert_allclose(by_name[key].denormalized(), by_dict[key].denormalized())
        np.testing.assert_allclose(by_method[key].denormalized(), by_dict[key].denormalized())


@pytest.mark.parametrize("kwargs", [
    {"tail": "reference"},
    {"tail": "eyring", "reference": "pyroomacoustics"},
    {"tail": "ray_tracing", "reference": "pyroomacoustics"},
    {"tail": "eyring", "n_slopes": 2},
    {"tail": "eyring", "late_envelope": True},
    {"tail": "air"},
    {"diffraction": "btm"},
    {"normalization": "FuMa"},
])
def test_conflicting_settings_raise_at_construction(kwargs):
    with pytest.raises(ValueError):
        Hybrid(**kwargs)


def test_tail_defaults():
    assert Hybrid().tail == "ray_tracing"
    assert Hybrid(reference="pyroomacoustics").tail == "reference"
    assert Hybrid().late_envelope and Hybrid(reference="data").late_envelope
    assert not Hybrid(tail="eyring").late_envelope and not Hybrid(tail="sabine").late_envelope
    assert not Hybrid(late_envelope=False).late_envelope


def test_eyring_in_an_uneven_room_warns():
    scene = make(materials=uneven())
    with pytest.warns(DiffuseFieldWarning):
        scene.simulate(Hybrid(tail="eyring", **FAST))


def test_an_eyring_tail_takes_no_simulation(monkeypatch):
    scene = make()
    monkeypatch.setattr(Pyroomacoustics, "compute", lambda *a, **k: pytest.fail("ran a reference"))
    hybrid = Hybrid(tail="sabine", **FAST)
    (ir,) = scene.simulate(hybrid).values()
    assert hybrid.reference_rirs is None
    assert ir.generation["tail"] == "sabine" and ir.generation["air_absorption"] is True


def test_an_image_source_only_reference_warns():
    scene = make()
    scene.simulate(Pyroomacoustics(image_source_order=3))
    with pytest.warns(UserWarning, match="no diffuse decay"):
        scene.simulate(Hybrid(reference="pyroomacoustics", **FAST))


def test_normalised_references_give_the_raw_level_result():
    scene = make()
    refs = references(scene)
    normalised = scene.simulate(Hybrid(reference=refs, late_envelope=True, seed=0, **FAST))
    for ir in refs.values():
        ir.normalized = False
    raw = scene.simulate(Hybrid(reference=refs, late_envelope=True, seed=0, **FAST))
    for key in normalised:
        np.testing.assert_allclose(normalised[key].denormalized(), raw[key].denormalized())


def test_btms_runs_in_the_simulator():
    scene = make(panel=True)
    (ir,) = scene.simulate(Hybrid(tail="eyring", image_source_order=0, duration=0.2, diffraction="btms")).values()
    assert np.all(np.isfinite(ir.ir))


# ---------------------------------------------------------- orientation --
def early_field(direction):
    scene = make([Receiver("rec", 8.0, 4.0, 1.5, direction=direction)])
    hybrid = Hybrid(tail="eyring", **FAST)
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
    (ir,) = scene.simulate(Hybrid(tail="eyring", **FAST)).values()
    loaded = ImpulseResponse.from_wav(ir.to_wav(tmp_path / "h.wav")[0])
    np.testing.assert_allclose(loaded.receiver.rotation, ir.receiver.rotation, atol=1e-12)
    assert (loaded.ambisonic_order, loaded.normalisation) == (1, "sn3d")


def test_a_receiver_next_to_a_fully_absorbing_ceiling_runs():
    """Every scan direction of the upward delay line hits the alpha = 1 surface."""
    sky = AcousticMaterial("sky", 1.0, 0.1)
    materials = {"floor": WALL, "ceiling": sky, "left_wall": WALL, "right_wall": WALL, "front_wall": WALL, "back_wall": WALL}
    scene = make([Receiver("rec", 8.0, 4.0, 3.9)], materials=materials)
    (ir,) = scene.simulate(Hybrid(tail="eyring", duration=0.3)).values()
    assert np.all(np.isfinite(ir.ir))


# ------------------------------------------------------------ late level --
@pytest.mark.parametrize("dims, alpha, src, rec", [
    ((10, 8, 4), 0.15, (2.5, 3.0, 1.5), (7.5, 5.5, 1.2)),
    ((6, 5, 3), 0.1, (1.5, 1.5, 1.5), (4.5, 3.5, 1.2)),
])
def test_the_late_level_matches_an_exact_specular_room(dims, alpha, src, rec):
    """A shoebox with specular walls has an exact answer: its image sources to high order.

    Given that answer's decay, the hybrid's late energy (after 80 ms, 1 kHz band) must match it.
    pyroomacoustics' ray tracer is not the yardstick: measured here it sits 1 to 2.6 dB below this
    answer without scattering, and 1.5 to 3.6 dB above Barron's revised theory with scattering 0.5.
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
    scene.add_acoustic_object(AcousticObjectsLibrary.shoebox_room(*dims, materials=AcousticMaterial("m", alpha, 0.0)))
    source, receiver = Source("s", *src), Receiver("r", *rec)
    scene.add_source(source)
    scene.add_receiver(receiver)
    reference = {f"{receiver.guid}_{source.guid}": ImpulseResponse("exact", source, receiver, exact, FS,
                                                                     generation={"scale": "package", "air_absorption": False})}
    (ir,) = scene.simulate(Hybrid(reference=reference)).values()

    sos = butter(4, [1000 / np.sqrt(2), 1000 * np.sqrt(2)], btype="band", fs=FS, output="sos")
    start = ir.direct_sound_index() + int(0.08 * FS)

    def late_energy(x):
        return np.sum(sosfiltfilt(sos, np.concatenate([x, np.zeros(FS // 2)]))[start:] ** 2)

    assert 10 * np.log10(late_energy(ir.denormalized()[:, 0]) / late_energy(exact)) == pytest.approx(0.0, abs=1.0)
