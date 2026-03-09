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
                },
                "ir2": {
                    "info": "invalid_ir_hoa_order",
                    "sample_rate": 96000,
                    "ambisonic_order": 2,
                    "mic_position": [0.0, 0.0, 1.5],   # x, y, z in meters
                    "ir": np.zeros([96000,9]),
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
                },
            "ir2": {
                    "info": "maximum_ambisonic_order",
                    "sample_rate": sample_rate_default,
                    "ambisonic_order": 4,
                    "mic_position": [0.0, 0.0, 1.5],   # x, y, z in meters
                    "ir": np.zeros([48000,25]),
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
    assert a.get_maximum_ambisonics_order() == 4


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
                },
            "ir2": {
                    "info": "ir_count",
                    "sample_rate": sample_rate_default,
                    "ambisonic_order": 4,
                    "mic_position": [0.0, 0.0, 1.5],   # x, y, z in meters
                    "ir": np.zeros([48000,25]),
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


def test_main_output():
    a = Auraliser(
        name="room1",
        impulse_responses={
            "ir1": {
                    "info": "maximum_ambisonic_order",
                    "sample_rate": sample_rate_default,
                    "ambisonic_order": 2,
                    "mic_position": [0.0, 0.0, 1.5],   # x, y, z in meters
                    "ir": np.zeros([48000,9]),
                },
            "ir2": {
                    "info": "maximum_ambisonic_order",
                    "sample_rate": sample_rate_default,
                    "ambisonic_order": 4,
                    "mic_position": [0.0, 0.0, 1.5],   # x, y, z in meters
                    "ir": np.zeros([48000,25]),
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
        output_format="ambisonics"
    )



