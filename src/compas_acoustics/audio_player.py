import os
import platform


__all__ = ["AudioPlayer"]


class AudioPlayer:
    """
    A class for playing audio file or audio signal.

    Attributes
    ----------
    system : str
        The operating system of the computer. It supports "Windows" and "Darwin" (macOS).
    _active_player : object
        The status of the audio playback. It can be a SoundPlayer object on Windows or a Popen object on macOS.
    sound_type : str
        The type of sound being played. It can be 'file' for audio files or 'array' for audio signals represented as numpy arrays.
    """

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
            import System.Media as sm  # type: ignore
            try:
                self._active_player = sm.SoundPlayer(file_path)
                self._active_player.Play()
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

    def play_array(self, array, sample_rate):
        """
        Play an audio signal represented as a numpy array.

        Parameters
        ----------
        array : numpy.ndarray
            The audio signal to play.
        sample_rate : int
            The sample rate of the audio signal.
        """
        import sounddevice as sd  # type: ignore
        import numpy as np
        self.stop()

        try:
            max_val = np.max(np.abs(array))
            if max_val > 1.0:
                array = array / max_val  # Normalize to [-1, 1]

            sd.play(array, sample_rate)
            self._active_player = sd
            self.sound_type = 'array'
        except Exception as e:
            print (f"Error playing sound array: {e}")
    
    def play(self, sound, sample_rate=None):
        """
        Play sound based on its type.
        
        Parameters
        ----------
        sound : str or numpy.ndarray
            The sound to play. It can be a file path (str) or an audio signal (numpy.ndarray).
        sample_rate : int, optional
            The sample rate of the audio signal. Required if sound is a numpy.ndarray.
        """
        if isinstance(sound, str):
            try:
                self.play_file(sound)
            except Exception as e:
                print(f"Error playing sound by type: {e}")
        elif hasattr(sound, "shape") and hasattr(sound, "ndim") and sample_rate is not None:
            try:
                self.play_array(sound, sample_rate)
            except Exception as e:
                print(f"Error playing sound by type: {e}")

    def stop(self):
        if self._active_player is None:
            return
        
        if self.sound_type == 'file':
            if self.system == "Windows":
                try:
                    self._active_player.Stop()
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