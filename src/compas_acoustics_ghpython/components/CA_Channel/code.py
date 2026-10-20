import os

import Grasshopper
import numpy as np
import soundfile as sf


def channels_of(sound):
    """Channels of an audio file path, an impulse response, a Source with a sound file, or samples shaped (samples, channels)."""
    if isinstance(sound, (str, os.PathLike)):
        return sf.info(str(sound)).channels
    channels = getattr(sound, "n_channels", None)
    if channels is not None:
        return int(channels)
    if hasattr(sound, "sound_file"):
        if not sound.sound_file:
            raise ValueError("The source has no sound file.")
        return sf.info(sound.sound_file).channels
    samples = np.asarray(sound)
    if samples.ndim in (1, 2) and samples.dtype != object:
        return 1 if samples.ndim == 1 else int(samples.shape[1])
    raise ValueError("Cannot read the channels of a {}.".format(type(sound).__name__))


class CheckChannel(Grasshopper.Kernel.GH_ScriptInstance):

    def RunScript(self, sound):
        if sound is None or (isinstance(sound, str) and not sound):
            ghenv.Component.AddRuntimeMessage(
                Grasshopper.Kernel.GH_RuntimeMessageLevel.Warning, "Input parameter 'sound' failed to collect data."
            )
            ghenv.Component.Message = None
            return None

        try:
            channels = channels_of(sound)
        except Exception as e:
            ghenv.Component.AddRuntimeMessage(Grasshopper.Kernel.GH_RuntimeMessageLevel.Error, str(e))
            ghenv.Component.Message = None
            return None
        ghenv.Component.Message = "channels: {}".format(channels)
        return channels
