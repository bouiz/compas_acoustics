import json

import numpy as np
import pytest

from compas_acoustics.ir import ImpulseResponse
from compas_acoustics.receiver import Receiver
from compas_acoustics.source import Source

sample_rate_default = 48000


def make_ir(data, name="ir", normalized=True, **kwargs):
    return ImpulseResponse(name, Source("s", 1, 2, 3), Receiver("r", 4, 5, 6), np.asarray(data, dtype=float), sample_rate_default, "test",
                           normalized=normalized, **kwargs)


def test_ir_is_peak_normalised_by_default():
    ir = make_ir([0.0, -0.4, 0.2])
    np.testing.assert_allclose(ir.ir, [0.0, -1.0, 0.5])
    assert ir.normalization_factor == pytest.approx(0.4)
    assert ir.level_db == pytest.approx(20 * np.log10(0.4))


def test_denormalized_returns_the_raw_ir():
    ir = make_ir([0.0, -0.4, 0.2])
    raw = ir.denormalized()
    np.testing.assert_allclose(raw, [0.0, -0.4, 0.2])
    raw[0] = 1.0
    np.testing.assert_allclose(ir.denormalized(), [0.0, -0.4, 0.2])


def test_unnormalised_ir_returns_the_raw_ir():
    ir = make_ir([0.0, -0.4, 0.2], normalized=False)
    np.testing.assert_allclose(ir.ir, [0.0, -0.4, 0.2])
    ir.normalized = True
    np.testing.assert_allclose(ir.ir, [0.0, -1.0, 0.5])


def test_multichannel_ir_uses_one_factor_for_all_channels():
    ir = make_ir([[0.2, 0.1], [0.0, -0.4]])
    assert ir.normalization_factor == pytest.approx(0.4)
    np.testing.assert_allclose(ir.ir, [[0.5, 0.25], [0.0, -1.0]])


def test_silent_ir_is_left_unscaled():
    ir = make_ir(np.zeros(4))
    assert ir.normalization_factor == 1.0
    np.testing.assert_allclose(ir.ir, 0.0)


def test_denormalize_group_is_relative_to_the_loudest():
    near, far = make_ir([0.8, 0.1], "near"), make_ir([0.2, 0.05], "far")
    scaled = ImpulseResponse.denormalize_group([near, far])
    np.testing.assert_allclose(scaled[0], [1.0, 0.125])
    np.testing.assert_allclose(scaled[1], [0.25, 0.0625])

    scaled_dict = ImpulseResponse.denormalize_group({"near": near, "far": far})
    assert list(scaled_dict) == ["near", "far"]
    np.testing.assert_allclose(scaled_dict["far"], [0.25, 0.0625])
    assert ImpulseResponse.denormalize_group([]) == []


def test_wav_and_json_round_trip(tmp_path):
    import soundfile as sf

    raw = np.array([[0.0, 0.0], [0.3, -0.1], [-0.15, 0.05]])
    ir = make_ir(raw, "seat_1", generation={"method": "image_source", "image_source_order": 3})
    wav_path, json_path = ir.to_wav(tmp_path / "seat_1.wav")

    # The WAV is peak-normalised; the JSON holds the factor and how it was generated
    written, fs = sf.read(wav_path)
    assert fs == sample_rate_default
    assert np.max(np.abs(written)) == pytest.approx(1.0)
    metadata = json.loads(json_path.read_text())
    assert metadata["normalization"]["factor"] == pytest.approx(0.3)
    assert metadata["normalization"]["method"] == "peak"
    assert metadata["generation"] == {"method": "image_source", "image_source_order": 3}
    assert metadata["source"]["position"] == [1.0, 2.0, 3.0]
    assert metadata["receiver"]["name"] == "r"
    assert metadata["channels"] == 2

    loaded = ImpulseResponse.from_wav(wav_path)
    np.testing.assert_allclose(loaded.denormalized(), raw, atol=1e-7)
    assert loaded.name == "seat_1"
    assert loaded.receiver.position == [4.0, 5.0, 6.0]
    assert loaded.generation == ir.generation


def test_from_wav_without_metadata_uses_the_wav_as_raw(tmp_path):
    import soundfile as sf

    sf.write(str(tmp_path / "plain.wav"), np.array([0.5, -0.25]), sample_rate_default, subtype="FLOAT")
    loaded = ImpulseResponse.from_wav(tmp_path / "plain.wav")
    np.testing.assert_allclose(loaded.denormalized(), [0.5, -0.25])
    assert loaded.name == "plain"


@pytest.mark.parametrize("normalize", [True, False])
def test_simulator_normalises_and_records_generation(normalize):
    from compas_acoustics import AcousticObjectsLibrary
    from compas_acoustics import Scene
    from compas_acoustics import Simulator

    scene = Scene("room")
    scene.add_acoustic_object(AcousticObjectsLibrary.shoebox_room(10, 8, 4))
    scene.add_source(Source("s", 2, 4, 1.5))
    scene.add_receiver(Receiver("near", 3, 4, 1.5))
    scene.add_receiver(Receiver("mid", 5, 4, 1.5))
    rirs = Simulator(scene).rir_from_image_source_method(image_source_order=2, normalize=normalize)

    by_receiver = {ir.receiver.name: ir for ir in rirs.values()}
    for ir in by_receiver.values():
        assert ir.normalized is normalize
        assert ir.generation == {"method": "image_source", "image_source_order": 2, "air_absorption": True, "scene": "room"}
        assert np.max(np.abs(ir.ir)) == pytest.approx(1.0 if normalize else ir.normalization_factor)
    # The raw level keeps the distance: the nearer receiver is louder
    assert by_receiver["near"].normalization_factor > by_receiver["mid"].normalization_factor


@pytest.mark.parametrize("sample_rate", [16000, 48000])
def test_simulated_impulse_responses_use_the_simulator_sample_rate(sample_rate):
    from compas_acoustics import AcousticObjectsLibrary
    from compas_acoustics import Scene
    from compas_acoustics import Simulator

    scene = Scene("room")
    scene.add_acoustic_object(AcousticObjectsLibrary.shoebox_room(10, 8, 4))
    scene.add_source(Source("s", 2, 4, 1.5))
    scene.add_receiver(Receiver("r", 5, 4, 1.5))
    simulator = Simulator(scene)
    simulator.sampling_rate = sample_rate
    # Air absorption above 16 kHz used to fail on a 7- vs 8-band mismatch
    rirs = simulator.rir_from_image_source_method(image_source_order=2, air_absorption=True)
    assert all(ir.sample_rate == sample_rate for ir in rirs.values())


def test_material_coefficients_hold_the_highest_band_above_its_range():
    from compas_acoustics import AcousticMaterial
    from compas_acoustics.pra_converter import material_to_pra
    from compas_acoustics.pra_converter import room_octave_bands

    material = AcousticMaterial("absorber", absorption_coefficients=[0.5, 0.7, 0.9, 0.95, 0.97, 0.98, 0.99])
    converted = material_to_pra(material, room_octave_bands(48000))
    assert len(converted.absorption_coeffs) == 8
    assert converted.absorption_coeffs[-1] == pytest.approx(0.99)
    assert np.all(np.asarray(converted.absorption_coeffs) <= 1.0)
