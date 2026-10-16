# flake8: noqa
import Grasshopper
import System
from compas_acoustics import AudioPlayer
import scriptcontext as sc
import soundfile as sf


class PlayAudio(Grasshopper.Kernel.GH_ScriptInstance):

    def offer_buttons(self):
        component = ghenv.Component
        offered_key = "audio_player_buttons_offered_{}".format(component.InstanceGuid)
        if sc.sticky.get(offered_key):
            return
        sc.sticky[offered_key] = True

        doc = component.OnPingDocument()
        if doc is None:
            return

        targets = [(index, nickname) for index, nickname in ((1, "Play"), (2, "Stop")) if component.Params.Input[index].SourceCount == 0]
        if not targets:
            return

        def deferred(gh_doc):
            base = component.Attributes.Pivot
            for row, (index, nickname) in enumerate(targets):
                input_param = component.Params.Input[index]
                if input_param.SourceCount > 0:
                    continue

                button = Grasshopper.Kernel.Special.GH_ButtonObject()
                button.CreateAttributes()
                button.NickName = nickname
                button.Attributes.Pivot = System.Drawing.PointF(base.X - 180, base.Y - 15 + row * 30)
                gh_doc.AddObject(button, False)
                input_param.AddSource(button)
                button.ExpireSolution(True)

        doc.ScheduleSolution(1, deferred)

    def RunScript(self, sound, play, stop, sample_rate, normalize):
        self.offer_buttons()

        if sound is None:
            return

        if isinstance(sound, str):  # file path
            try:
                sample_rate = sf.info(sound).samplerate
            except:
                pass
        elif hasattr(sound, "sound_file"):  # Source object
            sound = sound.sound_file
            try:
                sample_rate = sf.info(sound).samplerate
            except:
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
            player.play(sound, sample_rate, normalize is not False)
        if stop:
            player.stop()