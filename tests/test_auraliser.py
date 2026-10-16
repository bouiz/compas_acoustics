import importlib.util

import pytest
import numpy as np
from pathlib import Path
from compas_acoustics.auraliser import Auraliser

sample_rate_default = 48000

needs_spharpy = pytest.mark.skipif(importlib.util.find_spec("spharpy") is None,
                                   reason="binaural needs spharpy (Python >= 3.11)")

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


def test_ir_fuma_channel_ordering_raises():
    with pytest.raises(ValueError, match="channel order"):
        Auraliser(
            name="room1",
            impulse_responses={
                "ir1": {
                    "info": "fuma_not_supported",
                    "sample_rate": sample_rate_default,
                    "ambisonic_order": 1,
                    "mic_position": [0.0, 0.0, 1.5],   # x, y, z in meters
                    "ir": np.zeros([48000,4]),
                    "normalisation": "sn3d",
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


def test_convolve_matches_numpy_convolution():
    a = Auraliser(
            name="room1",
            impulse_responses={
                "ir1": {
                    "info": "convolve",
                    "sample_rate": sample_rate_default,
                    "ambisonic_order": 0,
                    "mic_position": [0.0, 0.0, 1.5],   # x, y, z in meters
                    "ir": np.array([[1.0], [0.5], [0.0], [0.25]]),
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

    output = a.convolve()

    sound = a.import_anechoic_sound(Path(__file__).parent / "test_data" / "test_auraliser_anechoic_sound.wav", sample_rate_default)
    expected = np.convolve([1.0, 0.5, 0.0, 0.25], sound)
    assert output.shape == (expected.size, 1)
    np.testing.assert_allclose(output[:, 0], expected, atol=1e-10)


def test_convolve_ambisonics():
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

    # Temporary output file
    output_file = tmp_path / "test_output.wav"

    allowed = {"mono", "stereo", "ambisonics", "binaural"}
    if importlib.util.find_spec("spharpy") is None:
        allowed.discard("binaural")
    for out_fmt in allowed:
        a.output_format = out_fmt
        result = a.export_convolved_sound(output_file)

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


# Helpers for the behaviour tests below

def ir_entry(ir, sample_rate=sample_rate_default, **kwargs):
    entry = {
        "sample_rate": sample_rate,
        "ambisonic_order": int(np.sqrt(ir.shape[1] if ir.ndim == 2 else 1)) - 1,
        "ir": ir,
        "normalisation": "sn3d",
        "channel_order": "acn",
    }
    entry.update(kwargs)
    return entry


def write_sound(path, data, sample_rate=sample_rate_default):
    import soundfile as sf

    sf.write(str(path), data, sample_rate, subtype="FLOAT")
    return str(path)


def test_1d_impulse_response_is_accepted():
    a = Auraliser("room1", {"ir1": ir_entry(np.ones(10))}, {"ac1": {"filepath": "a.wav"}}, "mono")
    assert a.impulse_responses["ir1"]["ir"].shape == (10, 1)


def test_order_mismatch_is_corrected_to_an_int_and_convolves(tmp_path):
    sound = write_sound(tmp_path / "s.wav", np.ones(5))
    with pytest.warns(UserWarning, match="Mismatch"):
        a = Auraliser("room1", {"ir1": ir_entry(np.ones((8, 4)), ambisonic_order=0)}, {"ac1": {"filepath": sound}}, "ambisonics")
    assert a.maximum_ambisonics_order == 1
    assert isinstance(a.maximum_ambisonics_order, int)
    assert a.convolve().shape == (12, 4)


def test_caller_impulse_responses_are_not_modified():
    ir = np.ones((100, 1))
    data = {"ir1": ir_entry(ir, sample_rate=44100)}
    with pytest.warns(UserWarning, match="resampled"):
        a = Auraliser("room1", data, {"ac1": {"filepath": "a.wav"}}, "mono", sample_rate=sample_rate_default)
    assert data["ir1"]["ir"] is ir
    assert data["ir1"]["sample_rate"] == 44100
    assert a.impulse_responses["ir1"]["sample_rate"] == sample_rate_default


def test_unknown_normalisation_raises():
    with pytest.raises(ValueError, match="normalisation"):
        Auraliser("room1", {"ir1": ir_entry(np.ones((10, 4)), normalisation="foo")}, {"ac1": {"filepath": "a.wav"}}, "mono")


@pytest.mark.parametrize("normalisation", ["fuma", "maxn"])
def test_fuma_and_maxn_normalisation_raise(normalisation):
    with pytest.raises(ValueError, match="normalisation"):
        Auraliser("room1", {"ir1": ir_entry(np.ones((10, 4)), normalisation=normalisation)}, {"ac1": {"filepath": "a.wav"}}, "mono")


def test_n3d_is_converted_to_sn3d():
    with pytest.warns(UserWarning, match="n3d to sn3d"):
        a = Auraliser("room1", {"ir1": ir_entry(np.ones((10, 9)), normalisation="n3d")}, {"ac1": {"filepath": "a.wav"}}, "mono")
    # SN3D = N3D / sqrt(2n + 1) for order n
    np.testing.assert_allclose(a.impulse_responses["ir1"]["ir"][0], [1.0] + [1 / np.sqrt(3)] * 3 + [1 / np.sqrt(5)] * 5)


def test_resampled_impulse_response_has_no_wraparound_and_keeps_its_gain():
    ir = np.zeros((44100, 1))
    ir[100] = 1.0
    with pytest.warns(UserWarning, match="resampled"):
        a = Auraliser("room1", {"ir1": ir_entry(ir, sample_rate=44100)}, {"ac1": {"filepath": "a.wav"}}, "mono", sample_rate=sample_rate_default)
    resampled = a.impulse_responses["ir1"]["ir"][:, 0]
    assert np.max(np.abs(resampled[-4800:])) < 1e-6
    # Matching the frequency response keeps the DC gain of the impulse response
    np.testing.assert_allclose(np.sum(resampled), 1.0, rtol=1e-3)


def test_changing_sample_rate_resamples_from_the_original_input():
    ir = np.zeros((48000, 1))
    ir[0] = 1.0
    a = Auraliser("room1", {"ir1": ir_entry(ir)}, {"ac1": {"filepath": "a.wav"}}, "mono")
    with pytest.warns(UserWarning, match="resampled"):
        a.sample_rate = 24000
    assert a.impulse_responses["ir1"]["ir"].shape == (24000, 1)
    assert a.impulse_responses["ir1"]["sample_rate"] == 24000


def test_multichannel_anechoic_sound_is_mixed_down(tmp_path):
    stereo = np.column_stack([np.ones(20), -np.ones(20) * 0.5])
    path = write_sound(tmp_path / "stereo.wav", stereo)
    a = Auraliser("room1", {"ir1": ir_entry(np.ones((4, 1)))}, {"ac1": {"filepath": path}}, "mono")
    with pytest.warns(UserWarning, match="mixing down"):
        audio = a.import_anechoic_sound(path, sample_rate_default)
    assert audio.ndim == 1
    np.testing.assert_allclose(audio, 0.25)


def test_anechoic_sound_is_resampled_with_time_amplitude(tmp_path):
    path = write_sound(tmp_path / "s.wav", np.full(44100, 0.5), sample_rate=44100)
    a = Auraliser("room1", {"ir1": ir_entry(np.ones((4, 1)))}, {"ac1": {"filepath": path}}, "mono")
    audio = a.import_anechoic_sound(path, sample_rate_default)
    assert audio.shape == (48000,)
    np.testing.assert_allclose(audio[1000:-1000], 0.5, atol=1e-3)


def test_convolve_matches_per_source_convolution(tmp_path):
    rng = np.random.default_rng(0)
    sound_a = write_sound(tmp_path / "a.wav", rng.standard_normal(300))
    sound_b = write_sound(tmp_path / "b.wav", rng.standard_normal(200))
    irs = {
        "s1": ir_entry(rng.standard_normal((50, 4))),
        "s2": ir_entry(rng.standard_normal((80, 1))),
        "s3": ir_entry(rng.standard_normal((30, 9))),
    }
    sounds = {"s1": {"filepath": sound_a}, "s2": {"filepath": sound_b}, "s3": {"filepath": sound_a}}
    a = Auraliser("room1", irs, sounds, "ambisonics")
    result = a.convolve()

    singles = []
    for key, data in irs.items():
        sound = a.import_anechoic_sound(sounds[key]["filepath"], sample_rate_default)
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
    result = Auraliser("room1", irs, sounds, "mono").convolve()
    np.testing.assert_allclose(result[:3, 0], 1.0)


def test_impulse_response_objects_use_their_source_sound_file(tmp_path):
    from compas_acoustics.ir import ImpulseResponse
    from compas_acoustics.receiver import Receiver
    from compas_acoustics.source import Source

    sound = write_sound(tmp_path / "s.wav", np.array([1.0, 0.0, 0.0]))
    receiver = Receiver("r", 0, 0, 0)
    irs = [
        ImpulseResponse("ir_a", Source("a", 1, 0, 0, sound_file=sound), receiver, np.array([1.0, 0.5]), sample_rate_default, "test"),
        ImpulseResponse("ir_b", Source("b", 2, 0, 0, sound_file=sound), receiver, np.array([0.0, 0.5]), sample_rate_default, "test"),
    ]
    a = Auraliser("room1", irs, output_format="mono")
    assert list(a.impulse_responses) == ["ir_a", "ir_b"]
    np.testing.assert_allclose(a.convolve()[:, 0], [1.0, 1.0, 0.0, 0.0])


def test_impulse_response_objects_with_a_list_of_sounds(tmp_path):
    from compas_acoustics.ir import ImpulseResponse
    from compas_acoustics.receiver import Receiver
    from compas_acoustics.source import Source

    sound = write_sound(tmp_path / "s.wav", np.array([2.0]))
    ir = ImpulseResponse("ir_a", Source("a", 1, 0, 0), Receiver("r", 0, 0, 0), np.array([1.0, 0.5]), sample_rate_default, "test")
    a = Auraliser("room1", [ir], [sound], output_format="mono")
    np.testing.assert_allclose(a.convolve()[:, 0], [2.0, 1.0])


def test_impulse_response_objects_without_sound_raise():
    from compas_acoustics.ir import ImpulseResponse
    from compas_acoustics.receiver import Receiver
    from compas_acoustics.source import Source

    ir = ImpulseResponse("ir_a", Source("a", 1, 0, 0), Receiver("r", 0, 0, 0), np.ones(4), sample_rate_default, "test")
    with pytest.raises(ValueError, match="sound_file"):
        Auraliser("room1", [ir], output_format="mono")


# Decoders

def plane_wave(order, x, y, z):
    """Ambisonic (ACN/SN3D) encoding of a unit impulse arriving from direction (x, y, z)."""
    try:
        import pyfar as pf
        import spharpy
    except ImportError:
        if order > 1:
            pytest.skip("higher-order encoding needs spharpy (Python >= 3.11)")
        r = np.sqrt(x * x + y * y + z * z)
        return np.array([[1.0, y / r, z / r, x / r]])[:, :(order + 1) ** 2]

    direction = pf.Coordinates(x, y, z)
    encoding = np.sqrt(4 * np.pi) * spharpy.spherical.spherical_harmonic_basis_real(
        order, direction, normalization="SN3D", channel_convention="ACN", condon_shortley=False)
    return encoding.reshape(1, -1)


def decoder(output_format, order=1):
    return Auraliser("room1", {"ir1": ir_entry(np.ones((32, (order + 1) ** 2)))}, {"ac1": {"filepath": "a.wav"}}, output_format)


def test_mono_decodes_the_omnidirectional_channel():
    np.testing.assert_allclose(decoder("mono").decode(plane_wave(1, 1, 0, 0)), [[1.0]])


def test_stereo_places_a_left_source_left_and_a_front_source_centre():
    a = decoder("stereo")
    np.testing.assert_allclose(a.decode(plane_wave(1, 0, 1, 0)), [[1.0, 0.0]], atol=1e-12)
    np.testing.assert_allclose(a.decode(plane_wave(1, 0, -1, 0)), [[0.0, 1.0]], atol=1e-12)
    np.testing.assert_allclose(a.decode(plane_wave(1, 1, 0, 0)), [[0.5, 0.5]], atol=1e-12)


def test_stereo_plays_a_mono_signal_on_both_channels():
    a = decoder("stereo", order=0)
    out = a.decode(np.array([[1.0], [0.5]]))
    np.testing.assert_allclose(out[:, 0], out[:, 1])


def test_binaural_with_mono_impulse_responses_raises():
    with pytest.raises(ValueError, match="Binaural output requires"):
        decoder("binaural", order=0)
    a = decoder("mono", order=0)
    with pytest.raises(ValueError, match="Binaural output requires"):
        a.output_format = "binaural"
    with pytest.raises(ValueError, match="Binaural output requires"):
        a.decode_binaural(np.ones((10, 1)))


@needs_spharpy
@pytest.mark.parametrize("y, louder_ear", [(1, 0), (-1, 1)])
def test_binaural_places_a_lateral_source_in_the_near_ear(y, louder_ear):
    out = decoder("binaural", order=3).decode(plane_wave(3, 0, y, 0))
    energy = np.sum(out ** 2, axis=0)
    assert 10 * np.log10(energy[louder_ear] / energy[1 - louder_ear]) > 6


@needs_spharpy
def test_binaural_magls_matches_hrtf_magnitudes():
    import pyfar as pf

    a = decoder("binaural", order=3)
    hrirs, directions, _ = pf.io.read_sofa(str(a.hrtf_path), verbose=False)
    hrirs = pf.dsp.resample(hrirs, sample_rate_default, match_amplitude="freq")
    freqs = np.fft.rfftfreq(4096, 1 / sample_rate_default)
    band = (freqs > 1000) & (freqs < 16000)
    errors = []
    for i in range(0, directions.csize, 25):
        x, y, z = directions.cartesian[i]
        rendered = a.decode_binaural(plane_wave(3, x, y, z))[: hrirs.n_samples].T
        R = np.abs(np.fft.rfft(rendered, 4096))[:, band]
        H = np.abs(np.fft.rfft(hrirs.time[i], 4096))[:, band]
        errors.append(np.mean(np.abs(20 * np.log10((R + 1e-9) / (H + 1e-9)))))
    # A plain least-squares decoder at order 3 is off by about 10 dB here
    assert np.mean(errors) < 4


def test_export_sets_level_after_decoding(tmp_path):
    import soundfile as sf

    a = decoder("stereo")
    exported = a.export_convolved_sound(tmp_path / "out.wav", np.vstack([plane_wave(1, 0, 1, 0), 0.5 * plane_wave(1, 1, 0, 0)]))
    # Default level 'relative' on a single render peaks at the -1 dBFS headroom
    assert np.max(np.abs(exported)) == pytest.approx(10 ** (-1 / 20))
    written, fs = sf.read(tmp_path / "out.wav")
    assert fs == sample_rate_default
    np.testing.assert_allclose(written, exported, atol=1e-7)


def test_defaults_to_stereo_and_the_first_impulse_response_sample_rate():
    with pytest.warns(UserWarning, match="resampled"):
        a = Auraliser(
            "room1",
            {"ir1": ir_entry(np.ones((10, 1)), sample_rate=44100), "ir2": ir_entry(np.ones((10, 1)), sample_rate=48000)},
            {"ir1": {"filepath": "a.wav"}, "ir2": {"filepath": "b.wav"}},
        )
    assert a.output_format == "stereo"
    assert a.sample_rate == 44100
    assert a.impulse_responses["ir2"]["sample_rate"] == 44100


def test_sample_rate_defaults_to_project_rate_without_impulse_responses():
    from compas_acoustics.defaults import Defaults

    assert Auraliser("room1", {}, {}).sample_rate == Defaults().SAMPLING_RATE


@pytest.mark.parametrize("output_format, channels", [("mono", 1), ("stereo", 2), ("binaural", 2), ("ambisonics", 9)])
def test_output_channels(output_format, channels):
    assert decoder(output_format, order=2).output_channels == channels


@pytest.mark.parametrize("output_format", ["mono", "stereo", pytest.param("binaural", marks=needs_spharpy)])
def test_convolve_decodes_impulse_responses_before_convolving(tmp_path, output_format):
    # By linearity, decoding the impulse responses first equals decoding the ambisonic result
    rng = np.random.default_rng(0)
    sound_a = write_sound(tmp_path / "a.wav", rng.standard_normal(300))
    sound_b = write_sound(tmp_path / "b.wav", rng.standard_normal(200))
    irs = {"s1": ir_entry(rng.standard_normal((50, 4))), "s2": ir_entry(rng.standard_normal((80, 9)))}
    sounds = {"s1": {"filepath": sound_a}, "s2": {"filepath": sound_b}}

    a = Auraliser("room1", irs, sounds, "ambisonics")
    ambisonic = a.convolve()
    a.output_format = output_format
    decoded = a.convolve()

    expected = a.decode(ambisonic)
    np.testing.assert_allclose(decoded, expected[: decoded.shape[0]], atol=1e-10)
    np.testing.assert_allclose(expected[decoded.shape[0]:], 0, atol=1e-10)


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
        decoder("mono").level = "loud"
    with pytest.raises(ValueError, match="headroom"):
        decoder("mono").headroom_db = 3


def test_export_without_level_warns_when_clipping(tmp_path):
    a = decoder("mono")
    a.level = "none"
    with pytest.warns(UserWarning, match="dBFS"):
        exported = a.export_convolved_sound(tmp_path / "out.wav", np.full((10, 1), 2.0))
    np.testing.assert_allclose(exported, 2.0)


def test_export_uses_a_given_group_gain(tmp_path):
    a = decoder("mono")
    exported = a.export_convolved_sound(tmp_path / "out.wav", np.full((10, 1), 0.5), gain=0.5)
    np.testing.assert_allclose(exported, 0.25)


def test_impulse_response_objects_are_auralised_at_their_raw_level(tmp_path):
    from compas_acoustics.ir import ImpulseResponse
    from compas_acoustics.receiver import Receiver
    from compas_acoustics.source import Source

    sound = write_sound(tmp_path / "s.wav", np.array([1.0]))
    source = Source("a", 1, 0, 0, sound_file=sound)
    near = ImpulseResponse("near", source, Receiver("r1", 0, 0, 0), np.array([0.5, 0.1]), sample_rate_default, "test")
    far = ImpulseResponse("far", source, Receiver("r2", 0, 0, 0), np.array([0.05, 0.01]), sample_rate_default, "test")
    assert np.max(np.abs(far.ir)) == pytest.approx(1.0)  # normalised by default

    renders = [Auraliser(ir.name, [ir], output_format="mono").convolve() for ir in (near, far)]
    np.testing.assert_allclose(renders[0][:, 0], [0.5, 0.1])
    np.testing.assert_allclose(renders[1][:, 0], [0.05, 0.01])


def test_n3d_to_sn3d_matches_spharpy():
    spharpy = pytest.importorskip("spharpy")
    from compas_acoustics.auraliser import _n3d_to_sn3d

    ir = np.random.default_rng(0).standard_normal((64, 16))
    np.testing.assert_allclose(_n3d_to_sn3d(ir), spharpy.spherical.renormalize(ir, "ACN", "N3D", "SN3D", axis=1))


def test_binaural_without_spharpy_names_the_python_it_needs(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "spharpy", None)
    auraliser = decoder("ambisonics", order=1)
    with pytest.raises(ImportError, match="Python >= 3.11"):
        auraliser.binaural_filters(1)
