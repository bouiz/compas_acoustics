
import numpy as np
import pytest
import soundfile as sf
from compas.data import json_dumps
from compas.data import json_loads

from compas_acoustics import Auraliser
from compas_acoustics import ImpulseResponse
from compas_acoustics import Receiver
from compas_acoustics import Source


def stereo_file(folder, name="chord.wav"):
    path = folder / name
    sf.write(str(path), np.column_stack([np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])]), 48000, subtype="FLOAT")
    return str(path)


def test_a_sound_file_is_kept_as_given_and_nothing_is_written_next_to_it(tmp_path):
    path = stereo_file(tmp_path)
    source = Source("s", 0, 0, 0, sound_file=path)
    assert source.sound_file == path
    assert sorted(p.name for p in tmp_path.iterdir()) == ["chord.wav"]
    assert sf.read(path)[0].shape == (3, 2)


def test_a_missing_sound_file_raises_and_none_clears_it(tmp_path):
    with pytest.raises(ValueError, match="does not exist"):
        Source("s", 0, 0, 0, sound_file=str(tmp_path / "missing.wav"))
    source = Source("s", 0, 0, 0, sound_file=stereo_file(tmp_path))
    source.sound_file = None
    assert source.sound_file is None


def test_a_stereo_sound_is_mixed_down_by_the_auraliser(tmp_path):
    path = stereo_file(tmp_path)
    ir = ImpulseResponse("ir", Source("s", 1, 0, 0, sound_file=path), Receiver("r", 0, 0, 0), np.array([1.0, 0.5]), 48000)
    with pytest.warns(UserWarning, match="mixing down to mono"):
        signal = Auraliser("room", [ir], output_format="mono").convolve()
    np.testing.assert_allclose(signal[:, 0], [0.5, 0.75, 0.25, 0.0], atol=1e-6)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["chord.wav"]


def test_the_sound_file_path_survives_json(tmp_path):
    path = stereo_file(tmp_path)
    assert json_loads(json_dumps(Source("s", 1, 2, 3, sound_file=path))).sound_file == path


def test_a_source_faces_x_by_default():
    assert list(Source("s", 0, 0, 0).direction) == [1.0, 0.0, 0.0]
