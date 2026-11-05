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
    ((data, rate),) = sounddevice.played
    assert rate == 44100 and np.max(np.abs(data)) == pytest.approx(0.99)


def test_the_devices_are_read_again_before_an_array_is_played(player, sounddevice):
    order = []
    sounddevice._terminate = lambda: order.append("terminate")
    sounddevice._initialize = lambda: order.append("initialize")
    sounddevice.play = lambda data, rate: order.append("play")
    player.play(np.array([0.0, 0.5]), 48000)
    player.play(np.array([0.0, 0.5]), 48000)
    assert order == ["terminate", "initialize", "play"] * 2


def test_a_failure_to_read_the_devices_again_is_a_warning_and_the_sound_still_plays(player, sounddevice):
    def terminate():
        raise RuntimeError("PortAudio is busy")

    sounddevice._terminate, sounddevice._initialize = terminate, lambda: None
    with pytest.warns(UserWarning, match="could not be refreshed.*PortAudio is busy"):
        player.play(np.array([0.0, 0.5]), 48000)
    assert len(sounddevice.played) == 1


def test_a_sounddevice_without_the_refresh_functions_just_plays(player, sounddevice, recwarn):
    player.play(np.array([0.0, 0.5]), 48000)
    assert len(sounddevice.played) == 1 and not recwarn.list


def test_an_array_can_be_played_at_its_rendered_level_and_clipping_is_a_warning_not_a_print(player, sounddevice, capsys):
    with pytest.warns(UserWarning, match="will clip"):
        player.play(np.array([0.0, 2.0]), 48000, normalize=False)
    ((data, _),) = sounddevice.played
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


# ---------------------------------------------------------------------------
# Gain, start, pause, resume, position
# ---------------------------------------------------------------------------


@pytest.fixture
def stream(sounddevice):
    """The fake sounddevice's stream: active from a play until a stop, or until the test ends it."""
    state = types.SimpleNamespace(active=False)
    play, stop = sounddevice.play, sounddevice.stop

    def playing(data, rate):
        play(data, rate)
        state.active = True

    def stopping():
        stop()
        state.active = False

    sounddevice.play, sounddevice.stop = playing, stopping
    sounddevice.get_stream = lambda: state
    return state


@pytest.fixture
def clock(player):
    now = types.SimpleNamespace(t=100.0)
    player._clock = lambda: now.t
    return now


def ramp(seconds=2.0, rate=1000):
    return np.linspace(-0.5, 0.5, int(seconds * rate)), rate


def test_play_applies_gain_and_start(player, sounddevice, stream, clock):
    samples, rate = ramp()
    player.play(samples, rate, normalize=True, gain_db=-6.0206, start=0.5)
    ((data, played_rate),) = sounddevice.played
    assert played_rate == rate
    assert np.max(np.abs(data)) == pytest.approx(0.99 / 2, rel=1e-3)
    # the first half second is left out, both channels from one mono signal
    assert data.shape == (1500, 2)
    assert player.position == pytest.approx(0.5)


def test_the_position_runs_with_the_clock_and_stops_at_the_end(player, stream, clock):
    samples, rate = ramp()
    player.play(samples, rate)
    clock.t += 1.25
    assert player.position == pytest.approx(1.25) and player.playing
    clock.t += 5.0
    assert player.position == pytest.approx(2.0)
    stream.active = False
    assert not player.playing


def test_pause_keeps_the_position_and_resume_plays_on(player, sounddevice, stream, clock):
    samples, rate = ramp()
    player.play(samples, rate)
    clock.t += 1.0
    player.pause()
    assert not player.playing and player.paused
    clock.t += 3.0
    assert player.position == pytest.approx(1.0)
    player.resume()
    data, _ = sounddevice.played[-1]
    assert data.shape[0] == 1000 and player.playing
    clock.t += 0.5
    assert player.position == pytest.approx(1.5)


def test_stop_resets_the_position(player, stream, clock):
    samples, rate = ramp()
    player.play(samples, rate)
    clock.t += 1.0
    player.stop()
    assert player.position == 0.0 and not player.playing and not player.paused


def test_a_start_past_the_end_plays_nothing(player, sounddevice, stream, clock):
    samples, rate = ramp()
    player.play(samples, rate, start=5.0)
    assert sounddevice.played == [] and player.position == pytest.approx(2.0)


def test_the_duration_is_known_after_a_play(player, stream, clock):
    samples, rate = ramp(1.5)
    player.play(samples, rate)
    assert player.duration == pytest.approx(1.5)
