# flake8: noqa
import Grasshopper
import soundfile as sf


class CheckChannel(Grasshopper.Kernel.GH_ScriptInstance):

    def RunScript(self, sound_file):
        if not sound_file:
            ghenv.Component.AddRuntimeMessage(
                Grasshopper.Kernel.GH_RuntimeMessageLevel.Warning, "Input parameter 'sound_file' failed to collect data."
            )
            ghenv.Component.Message = None
            return None

        channels = sf.info(sound_file).channels
        ghenv.Component.Message = "channels: {}".format(channels)
        return channels
