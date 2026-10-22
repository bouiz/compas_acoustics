import os
import platform
import time
import warnings

from compas_acoustics.defaults import Defaults

__all__ = ["AudioPlayer"]


class AudioPlayer:
    """
    A class for playing audio file or audio signal.

    Attributes
    ----------
    system : str
        The operating system of the computer. It supports "Windows" and "Darwin" (macOS).
    _active_player : object
        The status of the audio playback. It can be the winsound module on Windows or a Popen object on macOS.
    sound_type : str
        The type of sound being played. It can be 'file' for audio files or 'array' for audio signals represented as numpy arrays.
    """

    DEFAULT_SAMPLING_RATE = Defaults().SAMPLING_RATE

    def __init__(self):
        self.system = platform.system()
        self._active_player = None
        self.sound_type = None  # 'file', 'array' or None
        # the array being played, as sent to the device, its rate, and where and when its stream started
        self._signal = None
        self._rate = None
        self._started = None
        self._paused_at = None
        self._clock = time.monotonic

    @property
    def system(self):
        return self._system
    
    @system.setter
    def system(self, value):
        if value not in ["Windows", "Darwin",]:
            raise ValueError(f"Unsupported operating system {value}.")
        self._system = value

    def play_file(self, file_path):
        """Play an .wav audio file."""
        if not os.path.exists(file_path):
            raise ValueError(f"The file {file_path} does not exist.")

        self.stop()

        if self.system == "Windows":
            import winsound

            winsound.PlaySound(file_path, winsound.SND_FILENAME | winsound.SND_ASYNC)
            self._active_player = winsound
            self.sound_type = 'file'
        elif self.system == "Darwin":
            import subprocess  # type: ignore

            self._active_player = subprocess.Popen(["afplay", file_path])
            self.sound_type = 'file'
        else:
            raise NotImplementedError(f"Sound playback is not implemented for system {self.system}.")

    @staticmethod
    def _refresh_devices(sd):
        """Re-read the audio devices, so that headphones connected after sounddevice was loaded are the default output.

        sounddevice lists the devices once, when it is loaded, and has no public way to list them again.
        """
        if not (hasattr(sd, "_terminate") and hasattr(sd, "_initialize")):
            return
        try:
            sd._terminate()
            sd._initialize()
        except Exception as e:
            warnings.warn(f"The audio devices could not be refreshed, playing on the device found at start: {e}")

    def play_array(self, array, sampling_rate, normalize=True, gain_db=0.0, start=0.0):
        """
        Play an audio signal represented as a numpy array.

        Parameters
        ----------
        array : numpy.ndarray
            The audio signal to play, (samples,) or (samples, channels). A mono signal plays on both channels.
        sampling_rate : int
            The sample rate of the audio signal.
        normalize : bool, optional
            Whether to scale the signal to a peak of 0.99. Disable it to play auralised
            signals at their rendered level. Default is True.
        gain_db : float, optional
            A gain applied after the normalisation, in dB. Default is 0.
        start : float, optional
            Where to start, in seconds from the beginning of the signal. Default is 0.
        """
        import numpy as np
        self.stop()

        if sampling_rate is None:
            sampling_rate = self.DEFAULT_SAMPLING_RATE

        sampling_rate = int(sampling_rate)
        audio_data = np.asarray(array, dtype=np.float32)
        if audio_data.ndim == 1 or audio_data.shape[1] == 1:
            # A mono signal plays on both channels; a one-channel stream reaches only the left on many devices
            audio_data = np.repeat(audio_data.reshape(-1, 1), 2, axis=1)

        max_val = np.max(np.abs(audio_data)) if audio_data.size else 0.0
        if normalize and max_val > 0:
            audio_data = (audio_data / max_val) * 0.99  # Normalize to [-1, 1]
        audio_data = audio_data * np.float32(10 ** (gain_db / 20))
        peak = np.max(np.abs(audio_data)) if audio_data.size else 0.0
        if peak > 1:
            warnings.warn(f"The sound array peaks at {20 * np.log10(peak):+.1f} dBFS and will clip.")

        self._signal, self._rate = audio_data, sampling_rate
        self.sound_type = 'array'
        self._start_stream(start)

    def _start_stream(self, start):
        """Play :attr:`_signal` from ``start`` seconds."""
        import sounddevice as sd  # type: ignore

        start = min(max(float(start), 0.0), self.duration)
        self._started = (self._clock(), start)
        self._paused_at = None
        remaining = self._signal[int(round(start * self._rate)):]
        if len(remaining) == 0:
            return
        self._refresh_devices(sd)
        sd.check_output_settings(samplerate=self._rate)
        sd.play(remaining, self._rate)
        self._active_player = sd

    @property
    def duration(self):
        """The length of the array last played, in seconds; 0 for a file or nothing."""
        return len(self._signal) / self._rate if self._signal is not None else 0.0

    @property
    def paused(self):
        return self._paused_at is not None

    @property
    def position(self):
        """Where the array's playback is, in seconds: it runs with the clock while playing, and holds when paused."""
        if self._paused_at is not None:
            return self._paused_at
        if self._started is None:
            return 0.0
        started_at, start = self._started
        return min(start + (self._clock() - started_at), self.duration)

    @property
    def playing(self):
        """Whether a sound is playing now."""
        if self._active_player is None or self.paused:
            return False
        if self.sound_type == 'array':
            try:
                return bool(self._active_player.get_stream().active)
            except Exception:
                return False
        if self.sound_type == 'file' and self.system == "Darwin":
            return self._active_player.poll() is None
        return self.sound_type == 'file'

    def pause(self):
        """Stop an array's playback and keep its position; :meth:`resume` plays on from it. A file stops."""
        if self.sound_type != 'array' or self._signal is None or self.paused:
            if self.sound_type == 'file':
                self.stop()
            return
        position = self.position
        self._stop_device()
        self._paused_at = position

    def resume(self):
        """Play on from where :meth:`pause` stopped."""
        if self.paused:
            self._start_stream(self._paused_at)

    def play(self, sound, sampling_rate=None, normalize=True, gain_db=0.0, start=0.0):
        """
        Play sound based on its type.
        
        Parameters
        ----------
        sound : str or numpy.ndarray
            The sound to play. It can be a file path (str) or an audio signal (numpy.ndarray).
        sampling_rate : int, optional
            The sample rate of the audio signal. Required if sound is a numpy.ndarray.
        normalize : bool, optional
            Whether to scale a numpy.ndarray sound to a peak of 0.99. Default is True.
        gain_db, start : float, optional
            For an array: the gain in dB and where to start, in seconds. A file plays whole, at its level.

        Raises
        ------
        TypeError
            If ``sound`` is neither a file path nor a numpy array.
        """
        if isinstance(sound, str):
            self.play_file(sound)
        elif hasattr(sound, "shape") and hasattr(sound, "ndim"):
            self.play_array(sound, sampling_rate, normalize, gain_db, start)
        else:
            raise TypeError(f"A sound is a file path or a numpy array, not {type(sound).__name__}.")

    def stop(self):
        """Stop the playback and return to the beginning."""
        self._stop_device()
        self._started = None
        self._paused_at = None

    def _stop_device(self):
        if self._active_player is None:
            return

        if self.sound_type == 'file':
            if self.system == "Windows":
                try:
                    # passing None stops any sound currently played asynchronously
                    self._active_player.PlaySound(None, 0)
                except Exception as e:
                    warnings.warn(f"The sound could not be stopped on Windows: {e}")
                finally:
                    self._active_player = None
            elif self.system == "Darwin":
                try:
                    if self._active_player.poll() is None:  # Check if the process is still running
                        self._active_player.terminate()
                        self._active_player.wait()  # Release resources
                except Exception as e:
                    warnings.warn(f"The sound could not be stopped on macOS: {e}")
                finally:
                    self._active_player = None

        elif self.sound_type == 'array':
            try:
                self._active_player.stop()
            except Exception as e:
                warnings.warn(f"The sound could not be stopped: {e}")
            finally:
                self._active_player = None
