import os
import platform

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

    DEFAULT_SAMPLE_RATE = Defaults().SAMPLING_RATE

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
            try:
                winsound.PlaySound(file_path, winsound.SND_FILENAME | winsound.SND_ASYNC)
                self._active_player = winsound
                self.sound_type = 'file'
            except Exception as e:
                print (f"Error playing sound on Windows: {e}")
        elif self.system == "Darwin":
            import subprocess  # type: ignore
            try:
                self._active_player = subprocess.Popen(["afplay", file_path])
                self.sound_type = 'file'
            except Exception as e:
                print (f"Error playing sound on macOS: {e}")
        else:
            raise NotImplementedError(f"Sound playback is not implemented for system {self.system}.")

    def play_array(self, array, sample_rate, normalize=True):
        """
        Play an audio signal represented as a numpy array.

        Parameters
        ----------
        array : numpy.ndarray
            The audio signal to play.
        sample_rate : int
            The sample rate of the audio signal.
        normalize : bool, optional
            Whether to scale the signal to a peak of 0.99. Disable it to play auralised
            signals at their rendered level. Default is True.
        """
        import numpy as np
        import sounddevice as sd  # type: ignore
        self.stop()

        try:
            if sample_rate is None:
                sample_rate = self.DEFAULT_SAMPLE_RATE

            sample_rate = int(sample_rate)
            audio_data = np.asarray(array, dtype=np.float32)

            max_val = np.max(np.abs(audio_data))
            if normalize and max_val > 0:
                audio_data = (audio_data / max_val) * 0.99 # Normalize to [-1, 1]
            elif max_val > 1:
                print(f"Warning: sound array peaks at {20 * np.log10(max_val):+.1f} dBFS and will clip.")

            sd.check_output_settings(samplerate=sample_rate)

            sd.play(audio_data, sample_rate)
            self._active_player = sd
            self.sound_type = 'array'
        except Exception as e:
            print (f"Error playing sound array: {e}")
    
    def play(self, sound, sample_rate=None, normalize=True):
        """
        Play sound based on its type.
        
        Parameters
        ----------
        sound : str or numpy.ndarray
            The sound to play. It can be a file path (str) or an audio signal (numpy.ndarray).
        sample_rate : int, optional
            The sample rate of the audio signal. Required if sound is a numpy.ndarray.
        normalize : bool, optional
            Whether to scale a numpy.ndarray sound to a peak of 0.99. Default is True.
        """
        if isinstance(sound, str):
            try:
                self.play_file(sound)
            except Exception as e:
                print(f"Error playing sound by type: {e}")
        elif hasattr(sound, "shape") and hasattr(sound, "ndim"):
            try:
                self.play_array(sound, sample_rate, normalize)
            except Exception as e:
                print(f"Error playing sound by type: {e}")

    def stop(self):
        if self._active_player is None:
            return
        
        if self.sound_type == 'file':
            if self.system == "Windows":
                try:
                    # passing None stops any sound currently played asynchronously
                    self._active_player.PlaySound(None, 0)
                except Exception as e:
                    print (f"Error stopping sound on Windows: {e}")
                finally:
                    self._active_player = None
            elif self.system == "Darwin":
                try:
                    if self._active_player.poll() is None:  # Check if the process is still running
                        self._active_player.terminate()
                        self._active_player.wait()  # Release resources
                except Exception as e:
                    print (f"Error stopping sound on macOS: {e}")
                finally:
                    self._active_player = None

        elif self.sound_type == 'array':
            try:
                self._active_player.stop()
            except Exception as e:
                print (f"Error stopping sound array: {e}")
            finally:
                self._active_player = None