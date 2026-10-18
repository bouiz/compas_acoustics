import sys
import types

import numpy as np
import pytest

from compas_acoustics import AudioPlayer


@pytest.fixture
def player(monkeypatch):
    monkeypatch.setattr("platform.system", lambda: "Darwin")
    return AudioPlayer()


@pytest.fixture
def sounddevice(monkeypatch):
    module = types.SimpleNamespace(played=[], failure=None, stopped=0)

    def check_output_settings(samplerate=None):
        if module.failure:
            raise module.failure

    module.check_output_settings = check_output_settings
    module.play = lambda data, rate: module.played.append((data, rate))
    module.stop = lambda: setattr(module, "stopped", module.stopped + 1)
    monkeypatch.setitem(sys.modules, "sounddevice", module)
    return module


def test_an_array_is_played_normalised_at_the_given_rate(player, sounddevice):
    player.play(np.array([0.0, 0.25, -0.5]), 44100)
    (data, rate), = sounddevice.played
    assert rate == 44100 and np.max(np.abs(data)) == pytest.approx(0.99)


def test_an_array_can_be_played_at_its_rendered_level_and_clipping_is_a_warning_not_a_print(player, sounddevice, capsys):
    with pytest.warns(UserWarning, match="will clip"):
        player.play(np.array([0.0, 2.0]), 48000, normalize=False)
    (data, _), = sounddevice.played
    assert np.max(data) == 2.0
    assert capsys.readouterr().out == ""


def test_a_failure_to_play_is_raised_not_printed(player, sounddevice, capsys):
    sounddevice.failure = RuntimeError("no output device")
    with pytest.raises(RuntimeError, match="no output device"):
        player.play(np.zeros(4), 48000)
    assert capsys.readouterr().out == ""


def test_a_missing_file_and_an_unsupported_sound_raise(player):
    with pytest.raises(ValueError, match="does not exist"):
        player.play("missing.wav")
    with pytest.raises(TypeError, match="file path or a numpy array"):
        player.play(42)


def test_a_file_is_played_with_afplay_and_stopped(player, monkeypatch, tmp_path):
    calls = []

    class Process:
        def poll(self):
            return None

        def terminate(self):
            calls.append("terminate")

        def wait(self):
            calls.append("wait")

    monkeypatch.setattr("subprocess.Popen", lambda command: calls.append(command) or Process())
    path = tmp_path / "a.wav"
    path.write_bytes(b"")
    player.play(str(path))
    player.stop()
    assert calls == [["afplay", str(path)], "terminate", "wait"]


def test_a_failure_to_stop_is_a_warning(player, sounddevice):
    player.play(np.ones(4), 48000)
    sounddevice.stop = lambda: (_ for _ in ()).throw(RuntimeError("stuck"))
    with pytest.warns(UserWarning, match="could not be stopped: stuck"):
        player.stop()
