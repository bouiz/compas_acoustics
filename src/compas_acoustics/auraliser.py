from compas.data import Data
import warnings
import numpy as np
from pathlib import Path
from scipy.signal import resample
from scipy.signal import fftconvolve
import soundfile as sf
import librosa

from compas_acoustics.defaults import Defaults


__all__ = ['Auraliser']

class Auraliser(Data):
    """
    A class for handling auralisation.

    Parameters
    ----------
    name : str
        The name of the impulse responses. Lowercase and no spaces.
    description : str
        A text describing the impulse response group.
    impulse_responses : list
        A list of all impulse responses to be used for the auralisation.
    anechoic_sounds : list
        A list of the anechoic sounds to be used to convolve with the impulse responses.
    output_format : str
        The type of  output format. Lowercase and no spaces.
    """

    def __init__(self, name: str, impulse_responses: dict, anechoic_sounds: dict, output_format: str, description: str = None, sample_rate: int = 48000):
        super().__init__()
        self.name = name
        self.description = description
        self._sample_rate = sample_rate
        self.impulse_responses = impulse_responses
        self.anechoic_sounds = anechoic_sounds
        self._sample_rate = sample_rate
        self.output_format = output_format
        self._maximum_ambisonics_order = None
        self.check_number_of_impulse_responses_vs_anechoic_sounds()


    @property
    def name(self):
        return self._name


    @name.setter
    def name(self, name):
        if " " in name:
            raise ValueError("Auraliser name cannot contain spaces.")
        if any(char.isupper() for char in name):
            raise ValueError("Auraliser name must be lowercase.")
        self._name = name


    @property
    def description(self):
        return self._description
    

    @description.setter
    def description(self, description):
        self._description = description


    @property
    def sample_rate(self):
        return self._sample_rate

    @sample_rate.setter
    def sample_rate(self, value: int):
        if value <= 0:
            raise ValueError("Sample rate must be a positive integer.")
        self._sample_rate = value


    @property
    def output_format(self):
        return self._output_format


    @output_format.setter
    def output_format(self, output_format):
        allowed = {"mono", "stereo", "ambisonics", "binaural"}
        if " " in output_format:
            raise ValueError("Auraliser output_format cannot contain spaces.")
        if any(char.isupper() for char in output_format):
            raise ValueError("Auraliser output_format must be lowercase.")
        if output_format not in allowed:
            raise ValueError(f"Auraliser output_format must be one of {sorted(allowed)}.")
        self._output_format = output_format


    @property
    def impulse_responses(self):
        return self._impulse_responses


    @impulse_responses.setter
    def impulse_responses(self, impulse_responses):
        for ir_id, data in impulse_responses.items():
            # Check if nothing has been inserted in impulse response
            if data['ir'] is None or not isinstance(data['ir'], np.ndarray) or data['ir'].size == 0:
                raise ValueError(f"Impulse response {ir_id} is empty or invalid.")

            # Check if impulse response is transposed
            if data['ir'].shape[1] > data['ir'].shape[0]:
                warnings.warn(f"Impulse response {ir_id} is transposed. Impulse response will be transposed to thecorrect format")
                data['ir'] = data['ir'].T

            ambisonics_order_from_ir = np.sqrt(data['ir'].shape[1]) - 1
            # Check if impulse response follows typical ambisonics channel count, (ambisonics_order + 1)**2
            if ambisonics_order_from_ir != int(ambisonics_order_from_ir):
                raise ValueError(f"Impulse response {ir_id}, does not have a valid channel count to be a ambisonics impulse response.")

            # Ensure ambisonic_order is a number
            try:
                ambisonic_order_value = int(data['ambisonic_order'])
            except (TypeError, ValueError):
                ambisonic_order_value = None  # invalid or empty

            # change ambisonics_order variable if channel count and ambisonics order variable do not match
            if ambisonic_order_value != ambisonics_order_from_ir:
                warnings.warn(f"Mismatch detected between({data['ambisonic_order']} != {data['ir'].shape[1]}). Overwriting.")
                data['ambisonic_order'] = ambisonics_order_from_ir

            # check if impulse response is correct sample rate and resample if needed
            if data['sample_rate'] != self.sample_rate:
                warnings.warn(f"Mismatch detected between {ir_id} sample rate of {data['sample_rate']} Hz, and default sample rate of {self.sample_rate} Hz. Impulse response will be resampled to {self.sample_rate} Hz.")
                data['ir'] = self.resample_signal(data['ir'], data['sample_rate'], self.sample_rate)
                data['sample_rate'] = self.sample_rate

        self._impulse_responses = impulse_responses


    @property
    def anechoic_sounds(self):
        return self._anechoic_sounds


    @anechoic_sounds.setter
    def anechoic_sounds(self, anechoic_sounds):
        for ir_id, data in anechoic_sounds.items():
            # Allowed audio extensions
            allowed_extensions = {".wav", ".mp3", ".flac", ".ogg", ".aiff", ".aif"}

            # Convert string to Path
            filepath = Path(data['filepath'])
            if filepath.suffix.lower() not in allowed_extensions:
                raise ValueError(f"File {filepath} must be one of {allowed_extensions}")


        #count_anechoic_sounds = self.get_anechoic_sound_count()
        #count_impulse_reponses = self.get_impulse_response_count()
        #if count_anechoic_sounds != count_impulse_reponses:
        #   raise ValueError("Number of impulse responses does not correspond with the number of anechoic sounds")

        self._anechoic_sounds = anechoic_sounds


    @property
    def maximum_ambisonics_order(self):
        if self._maximum_ambisonics_order is None:
            maximum_ambisonics_order = 0
            for ir_id, data in self.impulse_responses.items():
                print(data['ambisonic_order'])
                if data['ambisonic_order'] > maximum_ambisonics_order:
                    maximum_ambisonics_order = data['ambisonic_order']
            self._maximum_ambisonics_order = maximum_ambisonics_order
        return self._maximum_ambisonics_order
    

    @property
    def longest_impulse_response_length(self):
        if not self.impulse_responses:
            return 0

        return max(data["ir"].shape[0] for data in self.impulse_responses.values())


    def get_impulse_response_count(self):
        return len(self.impulse_responses)
    

    def get_anechoic_sounds_count(self):
        return len(self.anechoic_sounds)
    

    def check_number_of_impulse_responses_vs_anechoic_sounds(self):
        anechoic_sounds_number = self.get_impulse_response_count()
        impulse_responses_number = self.get_anechoic_sounds_count()

        if anechoic_sounds_number != impulse_responses_number:
            raise ValueError("Number of impulse responses not coherent with number of anechoic sounds")

        return 0


    def resample_signal(self, signal_old, sr_old, sr_new):
        num_samples = int(len(signal_old) * sr_new / sr_old)
        signal_new = resample(signal_old, num_samples)

        return signal_new

 
    def import_anechoic_sound(self, filepath, sample_rate):
        filepath = Path(filepath)
        # MP3 handling
        if filepath.suffix.lower() == ".mp3":
            anechoic_sound, sr = librosa.load(filepath, sr=sample_rate, mono=True)
        else:
            anechoic_sound, sr = sf.read(filepath)

        if sr != sample_rate:
            anechoic_sound = self.resample_signal(anechoic_sound, sr, sample_rate)

        return anechoic_sound


    def ambisonics_upmix_downmix(self, sound, ambisonics_order):
        target_channels = (ambisonics_order + 1)**2
        current_channels = sound.shape[1]
        if current_channels < target_channels:
            # Pad extra channels with zeros
            pad_width = ((0, 0), (0, target_channels - current_channels))
            return np.pad(sound, pad_width)
        elif current_channels > target_channels:
            # Truncate channels (downmix naively)
            return sound[:, :target_channels]
        else:
            # Channel count is same as ambisonics order
            return sound
    

    def convolve(self, impulse_response, anechoic_sound):         
        impulse_response_channel_count = impulse_response.shape[1]
        impulse_response_length = impulse_response.shape[0]
        anechoic_sound_length = len(anechoic_sound)
        convolved_sound_length = impulse_response_length + anechoic_sound_length - 1

        convolved_sound = np.zeros([convolved_sound_length, impulse_response_channel_count])
        for ch in range(impulse_response_channel_count):
            convolved_sound[:,ch] = fftconvolve(impulse_response[:,ch], anechoic_sound)
                    
        return convolved_sound


    def multiconvolution(self):
        convolved_sound_summed = np.zeros((0, (self.maximum_ambisonics_order+1)**2))

        for (ir_id, ir_data), (ac_id, ac_data) in zip(
            self.impulse_responses.items(),
            self.anechoic_sounds.items()
        ):

            anechoic_sound = self.import_anechoic_sound(ac_data['filepath'], self.sample_rate)
            impulse_response = ir_data['ir']

            convolved_sound_single = self.convolve(impulse_response, anechoic_sound)

            # Make sure all convolved sounds are the maximum ambisonics order
            convolved_sound_single = self.ambisonics_upmix_downmix(convolved_sound_single, self.maximum_ambisonics_order)

            # Pad summed array dynamically
            new_len = max(convolved_sound_summed.shape[0], convolved_sound_single.shape[0])
            convolved_sound_summed = np.pad(convolved_sound_summed, ((0, new_len - convolved_sound_summed.shape[0]), (0, 0)))
            convolved_sound_single = np.pad(convolved_sound_single, ((0, new_len - convolved_sound_single.shape[0]), (0, 0)))

            
            convolved_sound_summed += convolved_sound_single

        return convolved_sound_summed



