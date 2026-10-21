import warnings

import Grasshopper
from compas_acoustics import AudioPlayer
from compas_acoustics_ghpython import ensure_inputs
import scriptcontext as sc
import soundfile as sf


class PlayAudio(Grasshopper.Kernel.GH_ScriptInstance):

    def RunScript(self, audio, sampling_rate, normalize, play, stop):
        ensure_inputs(ghenv.Component, {"play": None, "stop": None})

        if audio is None:
            return

        if isinstance(audio, str):  # file path
            try:
                sampling_rate = sf.info(audio).samplerate
            except Exception:
                pass
        elif hasattr(audio, "sound_file"):  # Source object
            audio = audio.sound_file
            try:
                sampling_rate = sf.info(audio).samplerate
            except Exception:
                pass
        elif hasattr(audio, "shape") and hasattr(audio, "ndim"):  # numpy array
            if audio.ndim == 2 and audio.shape[1] > 2:
                ghenv.Component.AddRuntimeMessage(
                    Grasshopper.Kernel.GH_RuntimeMessageLevel.Warning,
                    "A {}-channel ambisonic signal cannot be played here. Save it with Auralise as a multichannel file for an ambisonic decoder, "
                    "or simulate with Simulate Hybrid output binaural for headphone playback.".format(audio.shape[1])
                )
                return
            if sampling_rate is None:
                ghenv.Component.AddRuntimeMessage(
                    Grasshopper.Kernel.GH_RuntimeMessageLevel.Remark,
                    "No sampling_rate: the signal is played at {} Hz. Connect the sampling_rate of Auralise.".format(AudioPlayer.DEFAULT_SAMPLING_RATE),
                )
        else:
            ghenv.Component.AddRuntimeMessage(
                Grasshopper.Kernel.GH_RuntimeMessageLevel.Error, "Input parameter 'audio' is not valid."
            )
            return

        player_key = "player"
        if player_key not in sc.sticky:
            sc.sticky[player_key] = AudioPlayer()

        player = sc.sticky[player_key]

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                if play:
                    player.play(audio, sampling_rate, normalize is not False)
                if stop:
                    player.stop()
            except Exception as e:
                ghenv.Component.AddRuntimeMessage(Grasshopper.Kernel.GH_RuntimeMessageLevel.Error, "The sound could not be played: {}".format(e))
        for warning in caught:
            ghenv.Component.AddRuntimeMessage(Grasshopper.Kernel.GH_RuntimeMessageLevel.Warning, str(warning.message))