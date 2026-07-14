
import os
from compas_acoustics import play_sound

# Execute
play = True
dire = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(dire, "rir_mono.wav")

if play and path:
    play_sound(path)