import Grasshopper
from compas_acoustics import AudioPlayer
from compas_acoustics_ghpython import ensure_inputs
import scriptcontext as sc
import soundfile as sf


class PlayAudio(Grasshopper.Kernel.GH_ScriptInstance):

    def RunScript(self, sound, play, stop, sampling_rate, normalize):
        ensure_inputs(ghenv.Component, {"play": None, "stop": None})

        if sound is None:
            return

        if isinstance(sound, str):  # file path
            try:
                sampling_rate = sf.info(sound).samplerate
            except Exception:
                pass
        elif hasattr(sound, "sound_file"):  # Source object
            sound = sound.sound_file
            try:
                sampling_rate = sf.info(sound).samplerate
            except Exception:
                pass
        elif hasattr(sound, "shape") and hasattr(sound, "ndim"):  # numpy array
            if sound.ndim == 2 and sound.shape[1] > 2:
                ghenv.Component.AddRuntimeMessage(
                    Grasshopper.Kernel.GH_RuntimeMessageLevel.Error,
                    "Only mono or stereo signals can be played. Save ambisonic signals as a multichannel file, or auralise to binaural for headphone playback."
                )
                return
        else:
            ghenv.Component.AddRuntimeMessage(
                Grasshopper.Kernel.GH_RuntimeMessageLevel.Error, "Input parameter 'sound' is not valid."
            )
            return

        player_key = "player"
        if player_key not in sc.sticky:
            sc.sticky[player_key] = AudioPlayer()

        player = sc.sticky[player_key]

        if play:
            player.play(sound, sampling_rate, normalize is not False)
        if stop:
            player.stop()