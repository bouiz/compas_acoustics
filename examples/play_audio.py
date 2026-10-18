
import os
import time
from compas_acoustics import AudioPlayer

# A. Play a .wav file
play = True
dire = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(dire, "adult_female_speech.wav")

if play and path:
    player = AudioPlayer()
    player.play(path)

    time.sleep(3)  # Let the sound play for 3 seconds
    player.stop()


# B. Play a numpy array as audio
import scipy.io.wavfile as wav
import numpy as np

player = AudioPlayer()
sampling_rate, data_array = wav.read(path)

if data_array.dtype == np.int16:
    data_array = data_array.astype(np.float32) / 32768.0

player.play(data_array, sampling_rate)

time.sleep(3)
player.stop()

