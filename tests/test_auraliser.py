import pytest
import numpy as np
from pathlib import Path
from compas_acoustics.auraliser import Auraliser

sample_rate_default = 48000

def test_valid_instance():
    a = Auraliser(
        name="room1",
        impulse_responses={},
        anechoic_sounds={},
        output_format="mono"
    )
    assert a.name == "room1"
    assert a.output_format == "mono"


def test_uppercase_name_raises():
    with pytest.raises(ValueError):
        Auraliser(
            name="Room1",
            impulse_responses={},
            anechoic_sounds={},
            output_format="mono"
        )


def test_space_name_raises():
    with pytest.raises(ValueError):
        Auraliser(
            name="Room 1",
            impulse_responses={},
            anechoic_sounds={},
            output_format="mono"
        )


def test_uppercase_output_format_raises():
    with pytest.raises(ValueError):
        Auraliser(
            name="room1",
            impulse_responses={},
            anechoic_sounds={},
            output_format="Mono"
        )


def test_space_output_format_raises():
    with pytest.raises(ValueError):
        Auraliser(
            name="room1",
            impulse_responses={},
            anechoic_sounds={},
            output_format="Mo No"
        )


def test_invalid_output_format():
    with pytest.raises(ValueError):
        Auraliser(
            name="room1",
            impulse_responses={},
            anechoic_sounds={},
            output_format="invalid"
        )


def test_ir_empty():
    with pytest.raises(ValueError):
        Auraliser(
            name="room1",
            impulse_responses={
                "ir1": {
                    "info": "invalid_ir_hoa_order",
                    "sample_rate": sample_rate_default,
                    "ambisonic_order": 2,
                    "mic_position": [0.0, 0.0, 1.5],   # x, y, z in meters
                    "ir": [],
                    "normalisation": "sn3d",
                    "channel_order": "acn",
                },
            },
            anechoic_sounds={
                "ac1": {
                    "info": "whatevs",
                    "filepath": str(Path(__file__).parent / "test_data" / "test_auraliser_anechoic_sound.mp3"),
                }
            },
            output_format="mono"
        )


def test_ir_sampling_rate_negative():
    with pytest.raises(ValueError):
        Auraliser(
            name="room1",
            impulse_responses={
                "ir1": {
                    "info": "invalid_ir_hoa_order",
                    "sample_rate": -sample_rate_default,
                    "ambisonic_order": 2,
                    "mic_position": [0.0, 0.0, 1.5],   # x, y, z in meters
                    "ir": np.zeros([48000,9]),
                    "normalisation": "sn3d",
                    "channel_order": "acn",
                },
            },
            anechoic_sounds={
                "ac1": {
                    "info": "whatevs",
                    "filepath": str(Path(__file__).parent / "test_data" / "test_auraliser_anechoic_sound.mp3"),
                }
            },
            output_format="mono"
        )


def test_sampling_rate_negative():
    with pytest.raises(ValueError):
        a = Auraliser(
            name="room1",
            impulse_responses={
                "ir1": {
                    "info": "invalid_ir_hoa_order",
                    "sample_rate": sample_rate_default,
                    "ambisonic_order": 2,
                    "mic_position": [0.0, 0.0, 1.5],   # x, y, z in meters
                    "ir": np.zeros([48000,9]),
                    "normalisation": "sn3d",
                    "channel_order": "acn",
                },
            },
            anechoic_sounds={
                "ac1": {
                    "info": "whatevs",
                    "filepath": str(Path(__file__).parent / "test_data" / "test_auraliser_anechoic_sound.mp3"),
                }
            },
            output_format="mono",
        )
        a.sample_rate = -sample_rate_default


def test_ir_warning_resample():
    with pytest.warns(UserWarning):
        a = Auraliser(
            name="room1",
            impulse_responses={
                "ir1": {
                    "info": "invalid_ir_hoa_order",
                    "sample_rate": sample_rate_default,
                    "ambisonic_order": 2,
                    "mic_position": [0.0, 0.0, 1.5],   # x, y, z in meters
                    "ir": np.zeros([96000,9]),
                    "normalisation": "sn3d",
                    "channel_order": "acn",
                },
                "ir2": {
                    "info": "invalid_ir_hoa_order",
                    "sample_rate": sample_rate_default*2,
                    "ambisonic_order": 2,
                    "mic_position": [0.0, 0.0, 1.5],   # x, y, z in meters
                    "ir": np.zeros([96000,9]),
                    "normalisation": "sn3d",
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
            output_format="mono"
        )
        assert a.impulse_responses["ir2"]["sample_rate"] == sample_rate_default


def test_ir_warning_ambisonics_order_mismatch():
    with pytest.warns(UserWarning):
        Auraliser(
            name="room1",
            impulse_responses={
                "ir1": {
                    "info": "ambisonics_order_mismatch",
                    "sample_rate": sample_rate_default,
                    "ambisonic_order": 3,
                    "mic_position": [0.0, 0.0, 1.5],   # x, y, z in meters
                    "ir": np.zeros([48000,9]),
                    "normalisation": "sn3d",
                    "channel_order": "acn",
                },
            },
            anechoic_sounds={
                "ac1": {
                    "info": "whatevs",
                    "filepath": str(Path(__file__).parent / "test_data" / "test_auraliser_anechoic_sound.mp3"),
                }
            },
            output_format="mono"
        )


def test_ir_transposed():
    with pytest.warns(UserWarning):
        Auraliser(
            name="room1",
            impulse_responses={
                "ir1": {
                    "info": "ambisonics_order_mismatch",
                    "sample_rate": sample_rate_default,
                    "ambisonic_order": 2,
                    "mic_position": [0.0, 0.0, 1.5],   # x, y, z in meters
                    "ir": np.zeros([9,48000]),
                    "normalisation": "sn3d",
                    "channel_order": "acn",
                },
            },
            anechoic_sounds={
                "ac1": {
                    "info": "whatevs",
                    "filepath": str(Path(__file__).parent / "test_data" / "test_auraliser_anechoic_sound.mp3"),
                }
            },
            output_format="mono"
        )


def test_ir_warning_ambisonics_order_empty():
    with pytest.warns(UserWarning):
        Auraliser(
            name="room1",
            impulse_responses={
                "ir1": {
                    "info": "ambisonics_order_empty",
                    "sample_rate": sample_rate_default,
                    "ambisonic_order": [],
                    "mic_position": [0.0, 0.0, 1.5],   # x, y, z in meters
                    "ir": np.zeros([48000,9]),
                    "normalisation": "sn3d",
                    "channel_order": "acn",
                },
            },
            anechoic_sounds={
                "ac1": {
                    "info": "whatevs",
                    "filepath": str(Path(__file__).parent / "test_data" / "test_auraliser_anechoic_sound.mp3"),
                }
            },
            output_format="mono"
        )


def test_ir_warning_normalisation_empty():
    with pytest.warns(UserWarning):
        Auraliser(
            name="room1",
            impulse_responses={
                "ir1": {
                    "info": "ambisonics_order_empty",
                    "sample_rate": sample_rate_default,
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
            output_format="mono"
        )


def test_ir_warning_channel_order_empty():
    with pytest.warns(UserWarning):
        Auraliser(
            name="room1",
            impulse_responses={
                "ir1": {
                    "info": "ambisonics_order_empty",
                    "sample_rate": sample_rate_default,
                    "ambisonic_order": [],
                    "mic_position": [0.0, 0.0, 1.5],   # x, y, z in meters
                    "ir": np.zeros([48000,9]),
                    "normalisation": "sn3d",
                },
            },
            anechoic_sounds={
                "ac1": {
                    "info": "whatevs",
                    "filepath": str(Path(__file__).parent / "test_data" / "test_auraliser_anechoic_sound.mp3"),
                }
            },
            output_format="mono"
        )


def test_ir_invalid_ambisonics_order():
    with pytest.raises(ValueError):
        Auraliser(
            name="room1",
            impulse_responses={
                "ir1": {
                    "info": "invalid_ir_hoa_order",
                    "sample_rate": sample_rate_default,
                    "ambisonic_order": 2,
                    "mic_position": [0.0, 0.0, 1.5],   # x, y, z in meters
                    "ir": np.zeros([48000,8]),
                    "normalisation": "sn3d",
                    "channel_order": "acn",
                },
            },
            anechoic_sounds={
                "ac1": {
                    "info": "whatevs",
                    "filepath": str(Path(__file__).parent / "test_data" / "test_auraliser_anechoic_sound.mp3"),
                }
            },
            output_format="mono"
        )


def test_ir_fuma_channel_ordering_not_1st_order():
    with pytest.raises(ValueError):
        Auraliser(
            name="room1",
            impulse_responses={
                "ir1": {
                    "info": "invalid_ir_hoa_order",
                    "sample_rate": sample_rate_default,
                    "ambisonic_order": 2,
                    "mic_position": [0.0, 0.0, 1.5],   # x, y, z in meters
                    "ir": np.zeros([48000,9]),
                    "normalisation": "n3d",
                    "channel_order": "fuma",
                },
            },
            anechoic_sounds={
                "ac1": {
                    "info": "whatevs",
                    "filepath": str(Path(__file__).parent / "test_data" / "test_auraliser_anechoic_sound.mp3"),
                }
            },
            output_format="mono"
        )


def test_ir_fuma_channel_ordering():
    with pytest.warns(UserWarning):
        Auraliser(
            name="room1",
            impulse_responses={
                "ir1": {
                    "info": "invalid_ir_hoa_order",
                    "sample_rate": sample_rate_default,
                    "ambisonic_order": 1,
                    "mic_position": [0.0, 0.0, 1.5],   # x, y, z in meters
                    "ir": np.zeros([48000,4]),
                    "normalisation": "n3d",
                    "channel_order": "fuma",
                },
            },
            anechoic_sounds={
                "ac1": {
                    "info": "whatevs",
                    "filepath": str(Path(__file__).parent / "test_data" / "test_auraliser_anechoic_sound.mp3"),
                }
            },
            output_format="mono"
        )


def test_warning_ir_n3d_normalisation():
    with pytest.warns(UserWarning):
        Auraliser(
            name="room1",
            impulse_responses={
                "ir1": {
                    "info": "invalid_ir_hoa_order",
                    "sample_rate": sample_rate_default,
                    "ambisonic_order": 2,
                    "mic_position": [0.0, 0.0, 1.5],   # x, y, z in meters
                    "ir": np.zeros([48000,9]),
                    "normalisation": "n3d",
                    "channel_order": "acn",
                },
            },
            anechoic_sounds={
                "ac1": {
                    "info": "whatevs",
                    "filepath": str(Path(__file__).parent / "test_data" / "test_auraliser_anechoic_sound.mp3"),
                }
            },
            output_format="mono"
        )


def test_ir_maximum_length():
    a = Auraliser(
        name="room1",
        impulse_responses={
            "ir1": {
                    "info": "maximum_ambisonic_order",
                    "sample_rate": sample_rate_default,
                    "ambisonic_order": 2,
                    "mic_position": [0.0, 0.0, 1.5],   # x, y, z in meters
                    "ir": np.zeros([100000,9]),
                    "normalisation": "sn3d",
                    "channel_order": "acn",
                },
            "ir2": {
                    "info": "maximum_ambisonic_order",
                    "sample_rate": sample_rate_default,
                    "ambisonic_order": 4,
                    "mic_position": [0.0, 0.0, 1.5],   # x, y, z in meters
                    "ir": np.zeros([48000,25]),
                    "normalisation": "sn3d",
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
        output_format="mono"
    )
    assert a.longest_impulse_response_length == 100000


def test_anechoioc_sounds_allowed_extensions():
    with pytest.raises(ValueError):
        Auraliser(
            name="room1",
            impulse_responses={
                "ir1": {
                    "info": "ambisonics_order_mismatch",
                    "sample_rate": sample_rate_default,
                    "ambisonic_order": 2,
                    "mic_position": [0.0, 0.0, 1.5],   # x, y, z in meters
                    "ir": np.zeros([48000,9]),
                    "normalisation": "sn3d",
                    "channel_order": "acn",
                },
            },
            anechoic_sounds={
                "ac1": {
                    "info": "whatevs",
                    "filepath": str(Path(__file__).parent / "test_data" / "test_auraliser_anechoic_sound.mp4"),
                }
            },
            output_format="mono"
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
            output_format="mono"
        )


def test_maximum_ambisonic_order():
    a = Auraliser(
        name="room1",
        impulse_responses={
            "ir1": {
                    "info": "maximum_ambisonic_order",
                    "sample_rate": sample_rate_default,
                    "ambisonic_order": 2,
                    "mic_position": [0.0, 0.0, 1.5],   # x, y, z in meters
                    "ir": np.zeros([48000,9]),
                    "normalisation": "sn3d",
                    "channel_order": "acn",
                },
            "ir2": {
                    "info": "maximum_ambisonic_order",
                    "sample_rate": sample_rate_default,
                    "ambisonic_order": 4,
                    "mic_position": [0.0, 0.0, 1.5],   # x, y, z in meters
                    "ir": np.zeros([48000,25]),
                    "normalisation": "sn3d",
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
        output_format="mono"
    )
    assert a.maximum_ambisonics_order == 4


def test_impulse_response_and_anechoic_sounds_count():
    a = Auraliser(
        name="room1",
        impulse_responses={
            "ir1": {
                    "info": "ir_count",
                    "sample_rate": sample_rate_default,
                    "ambisonic_order": 2,
                    "mic_position": [0.0, 0.0, 1.5],   # x, y, z in meters
                    "ir": np.zeros([48000,9]),
                    "normalisation": "sn3d",
                    "channel_order": "acn",
                },
            "ir2": {
                    "info": "ir_count",
                    "sample_rate": sample_rate_default,
                    "ambisonic_order": 4,
                    "mic_position": [0.0, 0.0, 1.5],   # x, y, z in meters
                    "ir": np.zeros([48000,25]),
                    "normalisation": "sn3d",
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
        output_format="mono"
    )
    assert a.get_impulse_response_count() == 2
    assert a.get_anechoic_sounds_count() == 2


def test_import_anechoic_sound():
    a = Auraliser(
            name="room1",
            impulse_responses={
                "ir1": {
                    "info": "invalid_ir_hoa_order",
                    "sample_rate": sample_rate_default,
                    "ambisonic_order": 2,
                    "mic_position": [0.0, 0.0, 1.5],   # x, y, z in meters
                    "ir": np.zeros([48000,9]),
                    "normalisation": "sn3d",
                    "channel_order": "acn",
                },
            },
            anechoic_sounds={
                "ac1": {
                    "info": "anechoic_sound_test",
                    "filepath": str(Path(__file__).parent / "test_data" / "test_auraliser_anechoic_sound.wav"),
                }
            },
            output_format="mono"
        )

    filepath = Path(__file__).parent / "test_data" / "test_auraliser_anechoic_sound.wav"
    audio = a.import_anechoic_sound(filepath, sample_rate_default)

    # Assertions
    assert isinstance(audio, np.ndarray), "Audio should be a numpy array"
    assert audio.size > 0, "Audio array should not be empty"
    assert audio.ndim == 1, "Audio should be mono (1D array)"


def test_ambisonics_upmix_downmix():
    a = Auraliser(
            name="room1",
            impulse_responses={
                "ir1": {
                    "info": "invalid_ir_hoa_order",
                    "sample_rate": sample_rate_default,
                    "ambisonic_order": 2,
                    "mic_position": [0.0, 0.0, 1.5],   # x, y, z in meters
                    "ir": np.zeros([48000,9]),
                    "normalisation": "sn3d",
                    "channel_order": "acn",
                },
            },
            anechoic_sounds={
                "ac1": {
                    "info": "anechoic_sound_test",
                    "filepath": str(Path(__file__).parent / "test_data" / "test_auraliser_anechoic_sound.wav"),
                }
            },
            output_format="mono"
        )
    
    # Dummy 2-channel signal
    impulse_response = np.array([
        [1, 0],
        [0.5, 1],
        [0, 0.5]
    ])
    
    # Test upmix to 2nd-order Ambisonics (target 9 channels)
    result = a.ambisonics_upmix_downmix(impulse_response, 2)
    
    # Check shape
    assert result.shape == (3, 9), f"Expected shape (3,9), got {result.shape}"
    
    # Test downmix: provide 16-channel input, target 2nd-order (9 channels)
    dummy_16ch = np.zeros((5, 16))
    result2 = a.ambisonics_upmix_downmix(dummy_16ch, 2)
    assert result2.shape == (5, 9), f"Expected shape (5,9), got {result2.shape}"
    
    # Test no change if already correct channels
    dummy_9ch = np.zeros((4, 9))
    result3 = a.ambisonics_upmix_downmix(dummy_9ch, 2)
    assert result3.shape == (4, 9), f"Expected shape (4,9), got {result3.shape}"


def test_convolve():
    a = Auraliser(
            name="room1",
            impulse_responses={
                "ir1": {
                    "info": "invalid_ir_hoa_order",
                    "sample_rate": sample_rate_default,
                    "ambisonic_order": 2,
                    "mic_position": [0.0, 0.0, 1.5],   # x, y, z in meters
                    "ir": np.zeros([48000,9]),
                    "normalisation": "sn3d",
                    "channel_order": "acn",
                },
            },
            anechoic_sounds={
                "ac1": {
                    "info": "anechoic_sound_test",
                    "filepath": str(Path(__file__).parent / "test_data" / "test_auraliser_anechoic_sound.wav"),
                }
            },
            output_format="mono"
        )


    # Dummy signals
    impulse_response = np.array([
        [1, 0],
        [0.5, 1],
        [0, 0.5]
    ])

    anechoic_sound = np.array([1, 2])

    # Test
    output = a.convolve(impulse_response, anechoic_sound)

    assert isinstance(output, np.ndarray), "Output should be a numpy array"

    expected_length = impulse_response.shape[0] + len(anechoic_sound) - 1
    expected_channels = impulse_response.shape[1]
    assert output.shape == (expected_length, expected_channels), f"Output shape should be {(expected_length, expected_channels)}"

    expected_ch0 = np.convolve(impulse_response[:,0], anechoic_sound)
    np.testing.assert_allclose(output[:,0], expected_ch0)


def test_multiconvolution():
    a = Auraliser(
        name="room1",
        impulse_responses={
            "ir1": {
                "info": "multi1",
                "sample_rate": sample_rate_default,
                "ambisonic_order": 2,
                "mic_position": [0.0, 0.0, 1.5],
                "ir": np.zeros([48000, 9]),
                "normalisation": "sn3d",
                "channel_order": "acn",
            },
            "ir2": {
                "info": "multi2",
                "sample_rate": sample_rate_default,
                "ambisonic_order": 4,
                "mic_position": [0.0, 0.0, 1.5],
                "ir": np.zeros([48000, 25]),
                "normalisation": "sn3d",
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
        output_format="ambisonics"
    )

    result = a.multiconvolution()

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
                "sample_rate": sample_rate_default,
                "ambisonic_order": 2,
                "mic_position": [0.0, 0.0, 1.5],
                "ir": np.zeros([48000, 9]),
                "normalisation": "sn3d",
                "channel_order": "acn",
            },
            "ir2": {
                "info": "multi2",
                "sample_rate": sample_rate_default,
                "ambisonic_order": 4,
                "mic_position": [0.0, 0.0, 1.5],
                "ir": np.zeros([48000, 25]),
                "normalisation": "sn3d",
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
        output_format="ambisonics"
    )

    # Generate convolved signal
    convolved = a.multiconvolution()

    # Temporary output file
    output_file = tmp_path / "test_output.wav"

    allowed = {"mono", "stereo", "ambisonics", "binaural"}
    for out_fmt in allowed:
        a.output_format = out_fmt
        result = a.export_convolved_sound(output_file, convolved)

        # Test 1: result should be numpy array
        assert isinstance(result, np.ndarray), f"Expected numpy array, got {type(result)}"

        # Test 2: channel count
        if out_fmt == "mono":
            expected_channels = 1
        elif out_fmt == "stereo":
            expected_channels = 2
        elif out_fmt == "binaural":
            expected_channels = 2
        elif out_fmt == "ambisonics":
            expected_channels = (a.maximum_ambisonics_order + 1) ** 2
        
        assert result.shape[1] == expected_channels, f"Expected {expected_channels} channels, got {result.shape[1]}"

        # Test 3: output file exists
        assert output_file.exists(), "Output WAV file was not created"

        # Test 4: dtype should be float32
        assert result.dtype == np.float32, f"Expected float32, got {result.dtype}"









