import os
import platform
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

    def play_array(self, array, sampling_rate, normalize=True):
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
        """
        import numpy as np
        import sounddevice as sd  # type: ignore
        self.stop()

        if sampling_rate is None:
            sampling_rate = self.DEFAULT_SAMPLING_RATE

        sampling_rate = int(sampling_rate)
        audio_data = np.asarray(array, dtype=np.float32)
        if audio_data.ndim == 1 or audio_data.shape[1] == 1:
            # A mono signal plays on both channels; a one-channel stream reaches only the left on many devices
            audio_data = np.repeat(audio_data.reshape(-1, 1), 2, axis=1)

        max_val = np.max(np.abs(audio_data))
        if normalize and max_val > 0:
            audio_data = (audio_data / max_val) * 0.99  # Normalize to [-1, 1]
        elif max_val > 1:
            warnings.warn(f"The sound array peaks at {20 * np.log10(max_val):+.1f} dBFS and will clip.")

        self._refresh_devices(sd)
        sd.check_output_settings(samplerate=sampling_rate)

        sd.play(audio_data, sampling_rate)
        self._active_player = sd
        self.sound_type = 'array'


    def play(self, sound, sampling_rate=None, normalize=True):
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

        Raises
        ------
        TypeError
            If ``sound`` is neither a file path nor a numpy array.
        """
        if isinstance(sound, str):
            self.play_file(sound)
        elif hasattr(sound, "shape") and hasattr(sound, "ndim"):
            self.play_array(sound, sampling_rate, normalize)
        else:
            raise TypeError(f"A sound is a file path or a numpy array, not {type(sound).__name__}.")

    def stop(self):
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