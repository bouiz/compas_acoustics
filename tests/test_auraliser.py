import pytest
import numpy as np
from pathlib import Path
from compas_acoustics.auraliser import Auraliser

sampling_rate_default = 48000


def test_valid_instance():
    a = Auraliser(
        name="room1",
        impulse_responses={},
        anechoic_sounds={},
    )
    assert a.name == "room1"
    assert a.format == "mono" and a.output_channels == 1


def test_uppercase_name_raises():
    with pytest.raises(ValueError):
        Auraliser(
            name="Room1",
            impulse_responses={},
            anechoic_sounds={},
        )


def test_space_name_raises():
    with pytest.raises(ValueError):
        Auraliser(
            name="Room 1",
            impulse_responses={},
            anechoic_sounds={},
        )


def test_ir_empty():
    with pytest.raises(ValueError):
        Auraliser(
            name="room1",
            impulse_responses={
                "ir1": {
                    "info": "invalid_ir_hoa_order",
                    "sampling_rate": sampling_rate_default,
                    "ambisonic_order": 2,
                    "mic_position": [0.0, 0.0, 1.5],   # x, y, z in meters
                    "ir": [],
                    "normalization": "sn3d",
                    "channel_order": "acn",
                },
            },
            anechoic_sounds={
                "ac1": {
                    "info": "whatevs",
                    "filepath": str(Path(__file__).parent / "test_data" / "test_auraliser_anechoic_sound.mp3"),
                }
            },
        )


def test_ir_sampling_rate_negative():
    with pytest.raises(ValueError):
        Auraliser(
            name="room1",
            impulse_responses={
                "ir1": {
                    "info": "invalid_ir_hoa_order",
                    "sampling_rate": -sampling_rate_default,
                    "ambisonic_order": 2,
                    "mic_position": [0.0, 0.0, 1.5],   # x, y, z in meters
                    "ir": np.zeros([48000,9]),
                    "normalization": "sn3d",
                    "channel_order": "acn",
                },
            },
            anechoic_sounds={
                "ac1": {
                    "info": "whatevs",
                    "filepath": str(Path(__file__).parent / "test_data" / "test_auraliser_anechoic_sound.mp3"),
                }
            },
        )


def test_sampling_rate_negative():
    with pytest.raises(ValueError):
        a = Auraliser(
            name="room1",
            impulse_responses={
                "ir1": {
                    "info": "invalid_ir_hoa_order",
                    "sampling_rate": sampling_rate_default,
                    "ambisonic_order": 2,
                    "mic_position": [0.0, 0.0, 1.5],   # x, y, z in meters
                    "ir": np.zeros([48000,9]),
                    "normalization": "sn3d",
                    "channel_order": "acn",
                },
            },
            anechoic_sounds={
                "ac1": {
                    "info": "whatevs",
                    "filepath": str(Path(__file__).parent / "test_data" / "test_auraliser_anechoic_sound.mp3"),
                }
            },
        )
        a.sampling_rate = -sampling_rate_default


def test_ir_warning_resample():
    with pytest.warns(UserWarning):
        a = Auraliser(
            name="room1",
            impulse_responses={
                "ir1": {
                    "info": "invalid_ir_hoa_order",
                    "sampling_rate": sampling_rate_default,
                    "ambisonic_order": 2,
                    "mic_position": [0.0, 0.0, 1.5],   # x, y, z in meters
                    "ir": np.zeros([96000,9]),
                    "normalization": "sn3d",
                    "channel_order": "acn",
                },
                "ir2": {
                    "info": "invalid_ir_hoa_order",
                    "sampling_rate": sampling_rate_default*2,
                    "ambisonic_order": 2,
                    "mic_position": [0.0, 0.0, 1.5],   # x, y, z in meters
                    "ir": np.zeros([96000,9]),
                    "normalization": "sn3d",
                    "channel_order": "acn",
                },
            },
            anechoic_sounds={
                "ac1": {
                    "info": "whatevs",
                    "filepath": str(Path(__file__).parent / "test_data" / "test_auraliser_anechoic_sound.mp3"),
                },
                "ac2": {
                    "info": "whatevs",
                    "filepath": str(Path(__file__).parent / "test_data" / "test_auraliser_anechoic_sound.mp3"),
                }
            },
        )
        assert a.impulse_responses["ir2"]["sampling_rate"] == sampling_rate_default


def test_ir_warning_ambisonics_order_mismatch():
    with pytest.warns(UserWarning):
        Auraliser(
            name="room1",
            impulse_responses={
                "ir1": {
                    "info": "ambisonics_order_mismatch",
                    "sampling_rate": sampling_rate_default,
                    "ambisonic_order": 3,
                    "mic_position": [0.0, 0.0, 1.5],   # x, y, z in meters
                    "ir": np.zeros([48000,9]),
                    "normalization": "sn3d",
                    "channel_order": "acn",
                },
            },
            anechoic_sounds={
                "ac1": {
                    "info": "whatevs",
                    "filepath": str(Path(__file__).parent / "test_data" / "test_auraliser_anechoic_sound.mp3"),
                }
            },
        )


def test_ir_transposed():
    with pytest.warns(UserWarning):
        Auraliser(
            name="room1",
            impulse_responses={
                "ir1": {
                    "info": "ambisonics_order_mismatch",
                    "sampling_rate": sampling_rate_default,
                    "ambisonic_order": 2,
                    "mic_position": [0.0, 0.0, 1.5],   # x, y, z in meters
                    "ir": np.zeros([9,48000]),
                    "normalization": "sn3d",
                    "channel_order": "acn",
                },
            },
            anechoic_sounds={
                "ac1": {
                    "info": "whatevs",
                    "filepath": str(Path(__file__).parent / "test_data" / "test_auraliser_anechoic_sound.mp3"),
                }
            },
        )


def test_ir_warning_ambisonics_order_empty():
    with pytest.warns(UserWarning):
        Auraliser(
            name="room1",
            impulse_responses={
                "ir1": {
                    "info": "ambisonics_order_empty",
                    "sampling_rate": sampling_rate_default,
                    "ambisonic_order": [],
                    "mic_position": [0.0, 0.0, 1.5],   # x, y, z in meters
                    "ir": np.zeros([48000,9]),
                    "normalization": "sn3d",
                    "channel_order": "acn",
                },
            },
            anechoic_sounds={
                "ac1": {
                    "info": "whatevs",
                    "filepath": str(Path(__file__).parent / "test_data" / "test_auraliser_anechoic_sound.mp3"),
                }
            },
        )


def test_ir_warning_normalization_empty():
    with pytest.warns(UserWarning):
        Auraliser(
            name="room1",
            impulse_responses={
                "ir1": {
                    "info": "ambisonics_order_empty",
                    "sampling_rate": sampling_rate_default,
                    "ambisonic_order": [],
                    "mic_position": [0.0, 0.0, 1.5],   # x, y, z in meters
                    "ir": np.zeros([48000,9]),
                    "channel_order": "acn",
                },
            },
            anechoic_sounds={
                "ac1": {
                    "info": "whatevs",
                    "filepath": str(Path(__file__).parent / "test_data" / "test_auraliser_anechoic_sound.mp3"),
                }
            },
        )


def test_ir_warning_channel_order_empty():
    with pytest.warns(UserWarning):
        Auraliser(
            name="room1",
            impulse_responses={
                "ir1": {
                    "info": "ambisonics_order_empty",
                    "sampling_rate": sampling_rate_default,
                    "ambisonic_order": [],
                    "mic_position": [0.0, 0.0, 1.5],   # x, y, z in meters
                    "ir": np.zeros([48000,9]),
                    "normalization": "sn3d",
                },
            },
            anechoic_sounds={
                "ac1": {
                    "info": "whatevs",
                    "filepath": str(Path(__file__).parent / "test_data" / "test_auraliser_anechoic_sound.mp3"),
                }
            },
        )


def test_ir_invalid_ambisonics_order():
    with pytest.raises(ValueError):
        Auraliser(
            name="room1",
            impulse_responses={
                "ir1": {
                    "info": "invalid_ir_hoa_order",
                    "sampling_rate": sampling_rate_default,
                    "ambisonic_order": 2,
                    "mic_position": [0.0, 0.0, 1.5],   # x, y, z in meters
                    "ir": np.zeros([48000,8]),
                    "normalization": "sn3d",
                    "channel_order": "acn",
                },
            },
            anechoic_sounds={
                "ac1": {
                    "info": "whatevs",
                    "filepath": str(Path(__file__).parent / "test_data" / "test_auraliser_anechoic_sound.mp3"),
                }
            },
        )


def test_ir_fuma_channel_ordering_raises():
    with pytest.raises(ValueError, match="channel order"):
        Auraliser(
            name="room1",
            impulse_responses={
                "ir1": {
                    "info": "fuma_not_supported",
                    "sampling_rate": sampling_rate_default,
                    "ambisonic_order": 1,
                    "mic_position": [0.0, 0.0, 1.5],   # x, y, z in meters
                    "ir": np.zeros([48000,4]),
                    "normalization": "sn3d",
                    "channel_order": "fuma",
                },
            },
            anechoic_sounds={
                "ac1": {
                    "info": "whatevs",
                    "filepath": str(Path(__file__).parent / "test_data" / "test_auraliser_anechoic_sound.mp3"),
                }
            },
        )


def test_warning_ir_n3d_normalization():
    with pytest.warns(UserWarning):
        Auraliser(
            name="room1",
            impulse_responses={
                "ir1": {
                    "info": "invalid_ir_hoa_order",
                    "sampling_rate": sampling_rate_default,
                    "ambisonic_order": 2,
                    "mic_position": [0.0, 0.0, 1.5],   # x, y, z in meters
                    "ir": np.zeros([48000,9]),
                    "normalization": "n3d",
                    "channel_order": "acn",
                },
            },
            anechoic_sounds={
                "ac1": {
                    "info": "whatevs",
                    "filepath": str(Path(__file__).parent / "test_data" / "test_auraliser_anechoic_sound.mp3"),
                }
            },
        )


def test_ir_maximum_length():
    a = Auraliser(
        name="room1",
        impulse_responses={
            "ir1": {
                    "info": "maximum_ambisonic_order",
                    "sampling_rate": sampling_rate_default,
                    "ambisonic_order": 2,
                    "mic_position": [0.0, 0.0, 1.5],   # x, y, z in meters
                    "ir": np.zeros([100000,9]),
                    "normalization": "sn3d",
                    "channel_order": "acn",
                },
            "ir2": {
                    "info": "maximum_ambisonic_order",
                    "sampling_rate": sampling_rate_default,
                    "ambisonic_order": 4,
                    "mic_position": [0.0, 0.0, 1.5],   # x, y, z in meters
                    "ir": np.zeros([48000,25]),
                    "normalization": "sn3d",
                    "channel_order": "acn",
                },
        },
        anechoic_sounds={
                "ac1": {
                    "info": "whatevs",
                    "filepath": str(Path(__file__).parent / "test_data" / "test_auraliser_anechoic_sound.mp3"),
                },
                "ac2": {
                    "info": "whatevs",
                    "filepath": str(Path(__file__).parent / "test_data" / "test_auraliser_anechoic_sound.mp3"),
                }
            },
    )
    assert a.longest_impulse_response_length == 100000


def test_anechoioc_sounds_allowed_extensions():
    with pytest.raises(ValueError):
        Auraliser(
            name="room1",
            impulse_responses={
                "ir1": {
                    "info": "ambisonics_order_mismatch",
                    "sampling_rate": sampling_rate_default,
                    "ambisonic_order": 2,
                    "mic_position": [0.0, 0.0, 1.5],   # x, y, z in meters
                    "ir": np.zeros([48000,9]),
                    "normalization": "sn3d",
                    "channel_order": "acn",
                },
            },
            anechoic_sounds={
                "ac1": {
                    "info": "whatevs",
                    "filepath": str(Path(__file__).parent / "test_data" / "test_auraliser_anechoic_sound.mp4"),
                }
            },
        )


def test_number_of_irs_vs_anechoic_sounds():
    with pytest.raises(ValueError):
        Auraliser(
            name="room1",
            impulse_responses={},
            anechoic_sounds={
                "ac1": {
                    "info": "whatevs",
                    "filepath": str(Path(__file__).parent / "test_data" / "test_auraliser_anechoic_sound.mp3"),
                }
            },
        )


def test_maximum_ambisonic_order():
    a = Auraliser(
        name="room1",
        impulse_responses={
            "ir1": {
                    "info": "maximum_ambisonic_order",
                    "sampling_rate": sampling_rate_default,
                    "ambisonic_order": 2,
                    "mic_position": [0.0, 0.0, 1.5],   # x, y, z in meters
                    "ir": np.zeros([48000,9]),
                    "normalization": "sn3d",
                    "channel_order": "acn",
                },
            "ir2": {
                    "info": "maximum_ambisonic_order",
                    "sampling_rate": sampling_rate_default,
                    "ambisonic_order": 4,
                    "mic_position": [0.0, 0.0, 1.5],   # x, y, z in meters
                    "ir": np.zeros([48000,25]),
                    "normalization": "sn3d",
                    "channel_order": "acn",
                },
        },
        anechoic_sounds={
                "ac1": {
                    "info": "whatevs",
                    "filepath": str(Path(__file__).parent / "test_data" / "test_auraliser_anechoic_sound.mp3"),
                },
                "ac2": {
                    "info": "whatevs",
                    "filepath": str(Path(__file__).parent / "test_data" / "test_auraliser_anechoic_sound.mp3"),
                }
            },
    )
    assert a.maximum_ambisonics_order == 4


def test_impulse_response_and_anechoic_sounds_count():
    a = Auraliser(
        name="room1",
        impulse_responses={
            "ir1": {
                    "info": "ir_count",
                    "sampling_rate": sampling_rate_default,
                    "ambisonic_order": 2,
                    "mic_position": [0.0, 0.0, 1.5],   # x, y, z in meters
                    "ir": np.zeros([48000,9]),
                    "normalization": "sn3d",
                    "channel_order": "acn",
                },
            "ir2": {
                    "info": "ir_count",
                    "sampling_rate": sampling_rate_default,
                    "ambisonic_order": 4,
                    "mic_position": [0.0, 0.0, 1.5],   # x, y, z in meters
                    "ir": np.zeros([48000,25]),
                    "normalization": "sn3d",
                    "channel_order": "acn",
                },
        },
        anechoic_sounds={
                "ac1": {
                    "info": "whatevs",
                    "filepath": str(Path(__file__).parent / "test_data" / "test_auraliser_anechoic_sound.mp3"),
                },
                "ac2": {
                    "info": "whatevs",
                    "filepath": str(Path(__file__).parent / "test_data" / "test_auraliser_anechoic_sound.mp3"),
                }
            },
    )
    assert a.get_impulse_response_count() == 2
    assert a.get_anechoic_sounds_count() == 2


def test_import_anechoic_sound():
    a = Auraliser(
            name="room1",
            impulse_responses={
                "ir1": {
                    "info": "invalid_ir_hoa_order",
                    "sampling_rate": sampling_rate_default,
                    "ambisonic_order": 2,
                    "mic_position": [0.0, 0.0, 1.5],   # x, y, z in meters
                    "ir": np.zeros([48000,9]),
                    "normalization": "sn3d",
                    "channel_order": "acn",
                },
            },
            anechoic_sounds={
                "ac1": {
                    "info": "anechoic_sound_test",
                    "filepath": str(Path(__file__).parent / "test_data" / "test_auraliser_anechoic_sound.wav"),
                }
            },
        )

    filepath = Path(__file__).parent / "test_data" / "test_auraliser_anechoic_sound.wav"
    audio = a.import_anechoic_sound(filepath, sampling_rate_default)

    # Assertions
    assert isinstance(audio, np.ndarray), "Audio should be a numpy array"
    assert audio.size > 0, "Audio array should not be empty"
    assert audio.ndim == 1, "Audio should be mono (1D array)"


def test_convolve_matches_numpy_convolution():
    a = Auraliser(
            name="room1",
            impulse_responses={
                "ir1": {
                    "info": "convolve",
                    "sampling_rate": sampling_rate_default,
                    "ambisonic_order": 0,
                    "mic_position": [0.0, 0.0, 1.5],   # x, y, z in meters
                    "ir": np.array([[1.0], [0.5], [0.0], [0.25]]),
                    "normalization": "sn3d",
                    "channel_order": "acn",
                },
            },
            anechoic_sounds={
                "ac1": {
                    "info": "anechoic_sound_test",
                    "filepath": str(Path(__file__).parent / "test_data" / "test_auraliser_anechoic_sound.wav"),
                }
            },
        )

    output = a.convolve()

    sound = a.import_anechoic_sound(Path(__file__).parent / "test_data" / "test_auraliser_anechoic_sound.wav", sampling_rate_default)
    expected = np.convolve([1.0, 0.5, 0.0, 0.25], sound)
    assert output.shape == (expected.size, 1)
    np.testing.assert_allclose(output[:, 0], expected, atol=1e-10)


def test_convolve_ambisonics():
    a = Auraliser(
        name="room1",
        impulse_responses={
            "ir1": {
                "info": "multi1",
                "sampling_rate": sampling_rate_default,
                "ambisonic_order": 2,
                "mic_position": [0.0, 0.0, 1.5],
                "ir": np.zeros([48000, 9]),
                "normalization": "sn3d",
                "channel_order": "acn",
            },
            "ir2": {
                "info": "multi2",
                "sampling_rate": sampling_rate_default,
                "ambisonic_order": 4,
                "mic_position": [0.0, 0.0, 1.5],
                "ir": np.zeros([48000, 25]),
                "normalization": "sn3d",
                "channel_order": "acn",
            },
        },
        anechoic_sounds={
            "ac1": {
                "info": "whatevs",
                "filepath": str(Path(__file__).parent / "test_data" / "test_auraliser_anechoic_sound.wav"),
            },
            "ac2": {
                "info": "whatevs",
                "filepath": str(Path(__file__).parent / "test_data" / "test_auraliser_anechoic_sound.wav"),
            }
        },
    )

    result = a.convolve()

    # Test 1: result should be a NumPy array
    assert isinstance(result, np.ndarray), f"Expected a NumPy array, got {type(result)}"

    # Test 2: number of channels should match maximum Ambisonics order
    expected_channels = (a.maximum_ambisonics_order + 1) ** 2
    assert result.shape[1] == expected_channels, f"Expected {expected_channels} channels, got {result.shape[1]}"

    # Test 3: number of samples should be >= longest IR length
    longest_ir = max(ir_data['ir'].shape[0] for ir_data in a.impulse_responses.values())
    assert result.shape[0] >= longest_ir, f"Expected at least {longest_ir} samples, got {result.shape[0]}"


def test_export_convolved_sound(tmp_path):
    a = Auraliser(
        name="room1",
        impulse_responses={
            "ir1": {
                "info": "multi1",
                "sampling_rate": sampling_rate_default,
                "ambisonic_order": 2,
                "mic_position": [0.0, 0.0, 1.5],
                "ir": np.zeros([48000, 9]),
                "normalization": "sn3d",
                "channel_order": "acn",
            },
            "ir2": {
                "info": "multi2",
                "sampling_rate": sampling_rate_default,
                "ambisonic_order": 4,
                "mic_position": [0.0, 0.0, 1.5],
                "ir": np.zeros([48000, 25]),
                "normalization": "sn3d",
                "channel_order": "acn",
            },
        },
        anechoic_sounds={
            "ac1": {
                "info": "whatevs",
                "filepath": str(Path(__file__).parent / "test_data" / "test_auraliser_anechoic_sound.wav"),
            },
            "ac2": {
                "info": "whatevs",
                "filepath": str(Path(__file__).parent / "test_data" / "test_auraliser_anechoic_sound.wav"),
            }
        },
    )

    output_file = tmp_path / "test_output.wav"
    result = a.export_convolved_sound(output_file)
    assert a.format == "ambisonic"
    assert result.shape[1] == (a.maximum_ambisonics_order + 1) ** 2 == 25
    assert output_file.exists()
    assert result.dtype == np.float32


# Helpers for the behaviour tests below

def ir_entry(ir, sampling_rate=sampling_rate_default, **kwargs):
    entry = {
        "sampling_rate": sampling_rate,
        "ambisonic_order": int(np.sqrt(ir.shape[1] if ir.ndim == 2 else 1)) - 1,
        "ir": ir,
        "normalization": "sn3d",
        "channel_order": "acn",
    }
    entry.update(kwargs)
    return entry


def write_sound(path, data, sampling_rate=sampling_rate_default):
    import soundfile as sf

    sf.write(str(path), data, sampling_rate, subtype="FLOAT")
    return str(path)


def test_1d_impulse_response_is_accepted():
    a = Auraliser("room1", {"ir1": ir_entry(np.ones(10))}, {"ac1": {"filepath": "a.wav"}})
    assert a.impulse_responses["ir1"]["ir"].shape == (10, 1)


def test_order_mismatch_is_corrected_to_an_int_and_convolves(tmp_path):
    sound = write_sound(tmp_path / "s.wav", np.ones(5))
    with pytest.warns(UserWarning, match="Mismatch"):
        a = Auraliser("room1", {"ir1": ir_entry(np.ones((8, 4)), ambisonic_order=0)}, {"ac1": {"filepath": sound}})
    assert a.maximum_ambisonics_order == 1
    assert isinstance(a.maximum_ambisonics_order, int)
    assert a.convolve().shape == (12, 4)


def test_caller_impulse_responses_are_not_modified():
    ir = np.ones((100, 1))
    data = {"ir1": ir_entry(ir, sampling_rate=44100)}
    with pytest.warns(UserWarning, match="resampled"):
        a = Auraliser("room1", data, {"ac1": {"filepath": "a.wav"}}, sampling_rate=sampling_rate_default)
    assert data["ir1"]["ir"] is ir
    assert data["ir1"]["sampling_rate"] == 44100
    assert a.impulse_responses["ir1"]["sampling_rate"] == sampling_rate_default


def test_unknown_normalization_raises():
    with pytest.raises(ValueError, match="normalization"):
        Auraliser("room1", {"ir1": ir_entry(np.ones((10, 4)), normalization="foo")}, {"ac1": {"filepath": "a.wav"}})


@pytest.mark.parametrize("normalization", ["fuma", "maxn"])
def test_fuma_and_maxn_normalization_raise(normalization):
    with pytest.raises(ValueError, match="normalization"):
        Auraliser("room1", {"ir1": ir_entry(np.ones((10, 4)), normalization=normalization)}, {"ac1": {"filepath": "a.wav"}})


def test_n3d_is_converted_to_sn3d():
    with pytest.warns(UserWarning, match="n3d to sn3d"):
        a = Auraliser("room1", {"ir1": ir_entry(np.ones((10, 9)), normalization="n3d")}, {"ac1": {"filepath": "a.wav"}})
    # SN3D = N3D / sqrt(2n + 1) for order n
    np.testing.assert_allclose(a.impulse_responses["ir1"]["ir"][0], [1.0] + [1 / np.sqrt(3)] * 3 + [1 / np.sqrt(5)] * 5)


def test_resampled_impulse_response_has_no_wraparound_and_keeps_its_gain():
    ir = np.zeros((44100, 1))
    ir[100] = 1.0
    with pytest.warns(UserWarning, match="resampled"):
        a = Auraliser("room1", {"ir1": ir_entry(ir, sampling_rate=44100)}, {"ac1": {"filepath": "a.wav"}}, sampling_rate=sampling_rate_default)
    resampled = a.impulse_responses["ir1"]["ir"][:, 0]
    assert np.max(np.abs(resampled[-4800:])) < 1e-6
    # Matching the frequency response keeps the DC gain of the impulse response
    np.testing.assert_allclose(np.sum(resampled), 1.0, rtol=1e-3)


def test_changing_sampling_rate_resamples_from_the_original_input():
    ir = np.zeros((48000, 1))
    ir[0] = 1.0
    a = Auraliser("room1", {"ir1": ir_entry(ir)}, {"ac1": {"filepath": "a.wav"}})
    with pytest.warns(UserWarning, match="resampled"):
        a.sampling_rate = 24000
    assert a.impulse_responses["ir1"]["ir"].shape == (24000, 1)
    assert a.impulse_responses["ir1"]["sampling_rate"] == 24000


def test_multichannel_anechoic_sound_is_mixed_down(tmp_path):
    stereo = np.column_stack([np.ones(20), -np.ones(20) * 0.5])
    path = write_sound(tmp_path / "stereo.wav", stereo)
    a = Auraliser("room1", {"ir1": ir_entry(np.ones((4, 1)))}, {"ac1": {"filepath": path}})
    with pytest.warns(UserWarning, match="mixing down"):
        audio = a.import_anechoic_sound(path, sampling_rate_default)
    assert audio.ndim == 1
    np.testing.assert_allclose(audio, 0.25)


def test_anechoic_sound_is_resampled_with_time_amplitude(tmp_path):
    path = write_sound(tmp_path / "s.wav", np.full(44100, 0.5), sampling_rate=44100)
    a = Auraliser("room1", {"ir1": ir_entry(np.ones((4, 1)))}, {"ac1": {"filepath": path}})
    audio = a.import_anechoic_sound(path, sampling_rate_default)
    assert audio.shape == (48000,)
    np.testing.assert_allclose(audio[1000:-1000], 0.5, atol=1e-3)


def test_convolve_matches_per_source_convolution(tmp_path):
    rng = np.random.default_rng(0)
    sound_a = write_sound(tmp_path / "a.wav", rng.standard_normal(300))
    sound_b = write_sound(tmp_path / "b.wav", rng.standard_normal(200))
    irs = {
        "s1": ir_entry(rng.standard_normal((50, 4))),
        "s2": ir_entry(rng.standard_normal((80, 4))),
        "s3": ir_entry(rng.standard_normal((30, 9))),
    }
    sounds = {"s1": {"filepath": sound_a}, "s2": {"filepath": sound_b}, "s3": {"filepath": sound_a}}
    a = Auraliser("room1", irs, sounds)
    result = a.convolve()

    singles = []
    for key, data in irs.items():
        sound = a.import_anechoic_sound(sounds[key]["filepath"], sampling_rate_default)
        singles.append(np.stack([np.convolve(ch, sound) for ch in data["ir"].T], axis=1))
    expected = np.zeros((max(s.shape[0] for s in singles), 9))
    for single in singles:
        expected[: single.shape[0], : single.shape[1]] += single
    np.testing.assert_allclose(result, expected, atol=1e-10)


def test_impulse_responses_and_sounds_are_paired_by_id(tmp_path):
    impulse = write_sound(tmp_path / "impulse.wav", np.array([1.0, 0.0]))
    silence = write_sound(tmp_path / "silence.wav", np.zeros(2))
    irs = {"s1": ir_entry(np.ones((3, 1))), "s2": ir_entry(np.zeros((3, 1)))}
    # Same ids in a different order
    sounds = {"s2": {"filepath": silence}, "s1": {"filepath": impulse}}
    result = Auraliser("room1", irs, sounds).convolve()
    np.testing.assert_allclose(result[:3, 0], 1.0)


def test_impulse_response_objects_use_their_source_sound_file(tmp_path):
    from compas_acoustics.ir import ImpulseResponse
    from compas_acoustics.receiver import Receiver
    from compas_acoustics.source import Source

    sound = write_sound(tmp_path / "s.wav", np.array([1.0, 0.0, 0.0]))
    receiver = Receiver("r", 0, 0, 0)
    irs = [
        ImpulseResponse("ir_a", Source("a", 1, 0, 0, sound_file=sound), receiver, np.array([1.0, 0.5]), sampling_rate_default),
        ImpulseResponse("ir_b", Source("b", 2, 0, 0, sound_file=sound), receiver, np.array([0.0, 0.5]), sampling_rate_default),
    ]
    a = Auraliser("room1", irs)
    assert list(a.impulse_responses) == ["ir_a", "ir_b"]
    np.testing.assert_allclose(a.convolve()[:, 0], [1.0, 1.0, 0.0, 0.0])


def test_impulse_response_objects_with_a_list_of_sounds(tmp_path):
    from compas_acoustics.ir import ImpulseResponse
    from compas_acoustics.receiver import Receiver
    from compas_acoustics.source import Source

    sound = write_sound(tmp_path / "s.wav", np.array([2.0]))
    ir = ImpulseResponse("ir_a", Source("a", 1, 0, 0), Receiver("r", 0, 0, 0), np.array([1.0, 0.5]), sampling_rate_default)
    a = Auraliser("room1", [ir], [sound])
    np.testing.assert_allclose(a.convolve()[:, 0], [2.0, 1.0])


def test_impulse_response_objects_without_sound_raise():
    from compas_acoustics.ir import ImpulseResponse
    from compas_acoustics.receiver import Receiver
    from compas_acoustics.source import Source

    ir = ImpulseResponse("ir_a", Source("a", 1, 0, 0), Receiver("r", 0, 0, 0), np.ones(4), sampling_rate_default)
    with pytest.raises(ValueError, match="sound_file"):
        Auraliser("room1", [ir])


# Formats

def auraliser(order=1, binaural=False):
    ir = np.ones((32, 2 if binaural else (order + 1) ** 2))
    return Auraliser("room1", {"ir1": ir_entry(ir, binaural=binaural)}, {"ac1": {"filepath": "a.wav"}})


@pytest.mark.parametrize("order, binaural, fmt, channels", [(0, False, "mono", 1), (1, False, "ambisonic", 4),
                                                            (2, False, "ambisonic", 9), (None, True, "binaural", 2)])
def test_the_format_is_that_of_the_impulse_responses(order, binaural, fmt, channels):
    a = auraliser(order, binaural)
    assert a.format == fmt and a.output_channels == channels


def test_without_impulse_responses_the_format_is_mono():
    assert Auraliser("room1", {}, {}).format == "mono"


@pytest.mark.parametrize("orders, binaural", [([0, 1], [False, False]), ([1, None], [False, True]), ([0, None], [False, True])])
def test_impulse_responses_of_different_formats_raise(orders, binaural):
    entries = {f"ir{i}": ir_entry(np.ones((32, 2 if b else (o + 1) ** 2)), binaural=b) for i, (o, b) in enumerate(zip(orders, binaural))}
    with pytest.raises(ValueError, match="different formats"):
        Auraliser("room1", entries, {key: {"filepath": "a.wav"} for key in entries})


def test_ambisonic_orders_are_padded_to_the_highest(tmp_path):
    sound = write_sound(tmp_path / "s.wav", np.array([1.0]))
    first, second = np.zeros((16, 4)), np.zeros((16, 9))  # more channels than samples would read as transposed
    first[0], second[1] = [1.0, 0.5, 0.25, 0.125], np.arange(1.0, 10.0)
    out = Auraliser("room1", {"a": ir_entry(first), "b": ir_entry(second)}, {"a": {"filepath": sound}, "b": {"filepath": sound}}).convolve()
    assert out.shape == (16, 9)
    np.testing.assert_allclose(out[0], [1.0, 0.5, 0.25, 0.125, 0, 0, 0, 0, 0])
    np.testing.assert_allclose(out[1], np.arange(1.0, 10.0))


def test_export_sets_the_level(tmp_path):
    import soundfile as sf

    a = auraliser(1)
    exported = a.export_convolved_sound(tmp_path / "out.wav", np.array([[0.5, 0.2, 0.0, -0.1], [0.25, 0.0, 0.0, 0.0]]))
    # Default level 'relative' on a single render peaks at the -1 dBFS headroom
    assert np.max(np.abs(exported)) == pytest.approx(10 ** (-1 / 20))
    written, fs = sf.read(tmp_path / "out.wav")
    assert fs == sampling_rate_default and written.shape == (2, 4)
    np.testing.assert_allclose(written, exported, atol=1e-7)


def test_defaults_to_the_first_impulse_response_sampling_rate():
    with pytest.warns(UserWarning, match="resampled"):
        a = Auraliser(
            "room1",
            {"ir1": ir_entry(np.ones((10, 1)), sampling_rate=44100), "ir2": ir_entry(np.ones((10, 1)), sampling_rate=48000)},
            {"ir1": {"filepath": "a.wav"}, "ir2": {"filepath": "b.wav"}},
        )
    assert a.sampling_rate == 44100
    assert a.impulse_responses["ir2"]["sampling_rate"] == 44100


def test_sampling_rate_defaults_to_project_rate_without_impulse_responses():
    from compas_acoustics.defaults import Defaults

    assert Auraliser("room1", {}, {}).sampling_rate == Defaults().SAMPLING_RATE


# Output level

def test_render_gains_relative_keeps_level_differences():
    loud, quiet = np.full((10, 2), 0.5), np.full((10, 2), 0.05)
    gains = Auraliser.render_gains([loud, quiet], "relative", headroom_db=-1)
    assert gains[0] == gains[1]
    assert np.max(np.abs(loud * gains[0])) == pytest.approx(10 ** (-1 / 20))
    assert np.max(np.abs(quiet * gains[1])) == pytest.approx(10 ** (-1 / 20) / 10)


def test_render_gains_individual_and_none():
    loud, quiet = np.full((10, 1), 0.5), np.full((10, 1), 0.05)
    individual = Auraliser.render_gains([loud, quiet], "individual", headroom_db=-3)
    assert np.max(np.abs(quiet * individual[1])) == pytest.approx(10 ** (-3 / 20))
    assert np.max(np.abs(loud * individual[0])) == pytest.approx(10 ** (-3 / 20))
    assert Auraliser.render_gains([loud, quiet], "none") == [1.0, 1.0]


def test_relative_equals_individual_for_one_render():
    signal = [np.full((10, 1), 0.3)]
    assert Auraliser.render_gains(signal, "relative") == Auraliser.render_gains(signal, "individual")


def test_invalid_level_and_headroom_raise():
    with pytest.raises(ValueError, match="level"):
        auraliser(0).level = "loud"
    with pytest.raises(ValueError, match="headroom"):
        auraliser(0).headroom_db = 3


def test_export_without_level_warns_when_clipping(tmp_path):
    a = auraliser(0)
    a.level = "none"
    with pytest.warns(UserWarning, match="dBFS"):
        exported = a.export_convolved_sound(tmp_path / "out.wav", np.full((10, 1), 2.0))
    np.testing.assert_allclose(exported, 2.0)


def test_export_uses_a_given_group_gain(tmp_path):
    a = auraliser(0)
    exported = a.export_convolved_sound(tmp_path / "out.wav", np.full((10, 1), 0.5), gain=0.5)
    np.testing.assert_allclose(exported, 0.25)


def test_impulse_response_objects_are_auralised_at_their_raw_level(tmp_path):
    from compas_acoustics.ir import ImpulseResponse
    from compas_acoustics.receiver import Receiver
    from compas_acoustics.source import Source

    sound = write_sound(tmp_path / "s.wav", np.array([1.0]))
    source = Source("a", 1, 0, 0, sound_file=sound)
    near = ImpulseResponse("near", source, Receiver("r1", 0, 0, 0), np.array([0.5, 0.1]), sampling_rate_default)
    far = ImpulseResponse("far", source, Receiver("r2", 0, 0, 0), np.array([0.05, 0.01]), sampling_rate_default)
    assert np.max(np.abs(far.ir)) == pytest.approx(1.0)  # normalized by default

    renders = [Auraliser(ir.name, [ir]).convolve() for ir in (near, far)]
    np.testing.assert_allclose(renders[0][:, 0], [0.5, 0.1])
    np.testing.assert_allclose(renders[1][:, 0], [0.05, 0.01])


def test_an_n3d_impulse_response_is_auralised_in_sn3d(tmp_path):
    from compas_acoustics.ir import ImpulseResponse
    from compas_acoustics.receiver import Receiver
    from compas_acoustics.source import Source

    sound = write_sound(tmp_path / "s.wav", np.array([1.0]))
    n3d = np.zeros((8, 4))  # (samples, channels); one sample would read as transposed
    n3d[0] = [1.0, np.sqrt(3.0), 0.0, 0.0]  # a left source in N3D
    ir = ImpulseResponse("a", Source("s", 1, 0, 0, sound_file=sound), Receiver("r", 0, 0, 0), n3d, 48000,
                         ambisonic_order=1, normalization="n3d", channel_order="acn")
    with pytest.warns(UserWarning, match="n3d to sn3d"):
        out = Auraliser("room1", [ir]).convolve()
    np.testing.assert_allclose(out[0], [1.0, 1.0, 0.0, 0.0], atol=1e-12)


def test_n3d_to_sn3d_matches_spharpy():
    spharpy = pytest.importorskip("spharpy")
    from compas_acoustics.auraliser import _n3d_to_sn3d

    ir = np.random.default_rng(0).standard_normal((64, 16))
    np.testing.assert_allclose(_n3d_to_sn3d(ir), spharpy.spherical.renormalize(ir, "ACN", "N3D", "SN3D", axis=1))


def test_a_binaural_impulse_response_is_convolved_as_it_is(tmp_path):
    rng = np.random.default_rng(1)
    sound = write_sound(tmp_path / "s.wav", rng.standard_normal(100))
    ir = rng.standard_normal((40, 2))
    a = Auraliser("room1", {"b": ir_entry(ir, binaural=True)}, {"b": {"filepath": sound}})
    signal = a.import_anechoic_sound(sound, sampling_rate_default)
    expected = np.stack([np.convolve(ir[:, e], signal) for e in range(2)], axis=1)
    np.testing.assert_allclose(a.convolve(), expected, atol=1e-10)


def test_a_binaural_impulse_response_needs_two_channels():
    with pytest.raises(ValueError, match="2 channels"):
        Auraliser("room1", {"b": ir_entry(np.ones((8, 4)), binaural=True)}, {"b": {"filepath": "a.wav"}})


def test_binaural_impulse_response_objects_are_recognised(tmp_path):
    from compas_acoustics.ir import ImpulseResponse
    from compas_acoustics.receiver import Receiver
    from compas_acoustics.source import Source

    sound = write_sound(tmp_path / "s.wav", np.array([1.0, 0.0]))
    ir = ImpulseResponse("ir_b", Source("a", 1, 0, 0, sound_file=sound), Receiver("r", 0, 0, 0),
                         np.ones((8, 2)), sampling_rate_default, generation={"output": "binaural"})
    a = Auraliser("room1", [ir])
    assert a.impulse_responses["ir_b"]["binaural"] is True and a.format == "binaural"
