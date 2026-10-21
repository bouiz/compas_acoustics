import warnings
from pathlib import Path

import numpy as np
import pyfar as pf
from scipy.signal import fftconvolve

from compas_acoustics.defaults import Defaults

__all__ = ['Auraliser', 'FORMATS']

#: The formats an impulse response, and with it the auralised signal, can have
FORMATS = ("mono", "binaural", "ambisonic")
_LEVELS = {"relative", "individual", "none"}
_AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".aiff", ".aif"}

# Supported ambisonic conventions: ACN channel order with N3D or SN3D normalization
_CHANNEL_ORDERS = {"acn"}
_NORMALIZATIONS = {"sn3d", "n3d"}


def _n3d_to_sn3d(ir):
    degree = np.floor(np.sqrt(np.arange(ir.shape[1]))).astype(int)
    return ir / np.sqrt(2 * degree + 1)


class Auraliser:
    """
    A class for handling auralisation.

    Each impulse response is convolved with its anechoic sound and the results are
    summed. The signal keeps the format of its impulse responses (:attr:`format`):

    * ``'mono'``: one channel, e.g. from the ``Pyroomacoustics`` method.
    * ``'binaural'``: two channels, ears left and right, e.g. the ``Hybrid`` method's
      default output (or a dict entry with ``'binaural': True`` and two channels).
    * ``'ambisonic'``: ACN/SN3D (AmbiX) channels, (order + 1)**2 of them, at the highest
      order present, to save as a multichannel file for an ambisonic decoder elsewhere.
      N3D responses are converted to SN3D.

    There is no decoding: responses of different formats cannot be auralised together.

    Impulse responses are auralised at their raw level, so distance and level
    differences between them are kept. The output level is set on export by
    :attr:`level`:

    * ``'relative'``: one gain for a group of renders so the loudest peaks at
      :attr:`headroom_db`; level differences between the renders are kept. For a single
      render this is the same as ``'individual'``.
    * ``'individual'``: each render peaks at :attr:`headroom_db`.
    * ``'none'``: no gain; warns when the output exceeds 0 dBFS.

    Parameters
    ----------
    name : str
        The name of the auralisation. Lowercase and no spaces.
    impulse_responses : dict or list
        Either a dict mapping an id to a dict with the keys ``'ir'`` (array shaped
        (samples,) or (samples, channels)), ``'sampling_rate'``, ``'ambisonic_order'``,
        ``'channel_order'`` (``'acn'``) and ``'normalization'`` (``'sn3d'`` or
        ``'n3d'``), or a list of
        :class:`ImpulseResponse` objects, whose ``channel_order`` and ``normalization``
        are used (ACN/SN3D when they state none). All of one format.
    anechoic_sounds : dict or list, optional
        Either a dict mapping an id to a dict with the key ``'filepath'``, or a list of
        file paths. Paired with the impulse responses by id when both are dicts with the
        same ids, otherwise by position. If None, the ``sound_file`` of each impulse
        response's source is used.
    description : str, optional
        A text describing the auralisation.
    sampling_rate : int, optional
        The sample rate of the auralisation. Impulse responses and anechoic sounds are
        resampled to it. Default is the sample rate of the first impulse response.
    level : str, optional
        One of ``'relative'``, ``'individual'`` or ``'none'``. Default is ``'relative'``.
    headroom_db : float, optional
        Target output peak in dBFS for ``'relative'`` and ``'individual'``. Default is -1.

    Examples
    --------
    >>> import compas_acoustics as ca
    >>> sound = ca.Defaults().ANECHOIC_SOUNDFILES_LIBRARY_PATH + "/female_speech.wav"
    >>> scene = ca.Scene("office")
    >>> scene.add_acoustic_objects(ca.AcousticObjectsLibrary.shoebox_room(6, 5, 3, materials=ca.AcousticMaterial("wall", 0.2, 0.1)))
    >>> scene.add_source(ca.Source("talker", 1.5, 1.5, 1.6, sound_file=sound))
    >>> scene.add_receiver(ca.Receiver("listener", 4.0, 3.0, 1.2))
    >>> responses = scene.simulate(ca.Pyroomacoustics(ray_tracing=True, n_rays=2000, duration=0.5))
    >>> auraliser = ca.Auraliser("office", list(responses.values()))
    >>> auraliser.format, auraliser.convolve().shape[1]
    ('mono', 1)
    """

    def __init__(self, name: str, impulse_responses, anechoic_sounds=None, description: str = None, sampling_rate: int = None,
                 level: str = "relative", headroom_db: float = -1.0):
        self.name = name
        self.description = description
        self.level = level
        self.headroom_db = headroom_db
        if not isinstance(impulse_responses, dict):
            impulse_responses = list(impulse_responses)
        self.sampling_rate = sampling_rate if sampling_rate is not None else self._default_sampling_rate(impulse_responses)
        self.impulse_responses = impulse_responses
        if anechoic_sounds is None:
            anechoic_sounds = self._sounds_from_impulse_response_sources(self._impulse_responses_input)
        self.anechoic_sounds = anechoic_sounds
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
    def sampling_rate(self):
        return self._sampling_rate


    @sampling_rate.setter
    def sampling_rate(self, value: int):
        if value <= 0:
            raise ValueError("Sample rate must be a positive integer.")
        self._sampling_rate = int(value)
        # Re-derive the impulse responses at the new rate from the original input
        if hasattr(self, "_impulse_responses_input"):
            self.impulse_responses = self._impulse_responses_input


    @property
    def level(self):
        return self._level


    @level.setter
    def level(self, level):
        if level not in _LEVELS:
            raise ValueError(f"Auraliser level must be one of {sorted(_LEVELS)}.")
        self._level = level


    @property
    def headroom_db(self):
        return self._headroom_db


    @headroom_db.setter
    def headroom_db(self, headroom_db):
        if headroom_db > 0:
            raise ValueError("Auraliser headroom_db must be 0 dBFS or lower.")
        self._headroom_db = float(headroom_db)


    @property
    def impulse_responses(self):
        """dict : Impulse responses as (samples, channels) arrays at :attr:`sampling_rate`, ambisonic ones in ACN/SN3D."""
        return self._impulse_responses


    @impulse_responses.setter
    def impulse_responses(self, impulse_responses):
        if not isinstance(impulse_responses, dict):
            impulse_responses = list(impulse_responses)
        self._impulse_responses_input = impulse_responses
        if not isinstance(impulse_responses, dict):
            impulse_responses = self._impulse_responses_from_objects(impulse_responses)
        # Work on copies so the caller's data is never modified
        prepared = {ir_id: self._prepare_impulse_response(ir_id, dict(data)) for ir_id, data in impulse_responses.items()}
        formats = {ir_id: self._format_of(data) for ir_id, data in prepared.items()}
        if len(set(formats.values())) > 1:
            raise ValueError(
                "The impulse responses are of different formats ("
                + ", ".join(f"{ir_id}: {fmt}" for ir_id, fmt in formats.items())
                + "); auralise each format on its own, there is no decoding between them."
            )
        self._impulse_responses = prepared
        self._format = next(iter(formats.values()), "mono")


    @staticmethod
    def _format_of(data):
        if data.get('binaural'):
            return "binaural"
        return "mono" if data['ambisonic_order'] == 0 else "ambisonic"


    @property
    def anechoic_sounds(self):
        return self._anechoic_sounds


    @anechoic_sounds.setter
    def anechoic_sounds(self, anechoic_sounds):
        if not isinstance(anechoic_sounds, dict):
            anechoic_sounds = {f"sound_{i}": {'filepath': filepath} for i, filepath in enumerate(anechoic_sounds)}

        for sound_id, data in anechoic_sounds.items():
            filepath = Path(data['filepath'])
            if filepath.suffix.lower() not in _AUDIO_EXTENSIONS:
                raise ValueError(f"File {filepath} must be one of {_AUDIO_EXTENSIONS}")

        self._anechoic_sounds = anechoic_sounds


    @property
    def maximum_ambisonics_order(self):
        return max((data['ambisonic_order'] for data in self.impulse_responses.values() if not data.get('binaural')), default=0)


    @property
    def format(self):
        """str : ``'mono'``, ``'binaural'`` or ``'ambisonic'``, the format of the impulse responses and of the signal."""
        return self._format


    @property
    def output_channels(self):
        """int : Number of channels of the auralised signal: 1, 2, or (order + 1)**2 for ambisonics."""
        if self.format == "binaural":
            return 2
        return (self.maximum_ambisonics_order + 1) ** 2


    @property
    def longest_impulse_response_length(self):
        return max((data['ir'].shape[0] for data in self.impulse_responses.values()), default=0)


    def get_impulse_response_count(self):
        return len(self.impulse_responses)


    def get_anechoic_sounds_count(self):
        return len(self.anechoic_sounds)


    def check_number_of_impulse_responses_vs_anechoic_sounds(self):
        if self.get_impulse_response_count() != self.get_anechoic_sounds_count():
            raise ValueError("Number of impulse responses not coherent with number of anechoic sounds")

        return 0


    # Input preparation

    @staticmethod
    def _default_sampling_rate(impulse_responses):
        """The sample rate of the first impulse response, or the project default if there are none."""
        if isinstance(impulse_responses, dict):
            impulse_responses = [data['sampling_rate'] for data in impulse_responses.values()]
        else:
            impulse_responses = [ir.sampling_rate for ir in impulse_responses]
        return impulse_responses[0] if impulse_responses else Defaults().SAMPLING_RATE


    @staticmethod
    def _impulse_responses_from_objects(impulse_responses):
        """Convert a list of ImpulseResponse objects to the dict format, keyed by name, at their raw level."""
        impulse_responses = list(impulse_responses)
        names = [getattr(ir, 'name', None) for ir in impulse_responses]
        use_names = all(names) and len(set(names)) == len(names)

        converted = {}
        for i, ir in enumerate(impulse_responses):
            # Use the raw level so level differences between impulse responses are kept
            ir_array = np.asarray(ir.denormalized() if hasattr(ir, 'denormalized') else ir.ir, dtype=float)
            channels = 1 if ir_array.ndim == 1 else ir_array.shape[1]
            order = getattr(ir, 'ambisonic_order', None)
            converted[names[i] if use_names else f"ir_{i}"] = {
                'ir': ir_array,
                'sampling_rate': ir.sampling_rate,
                'ambisonic_order': int(round(np.sqrt(channels))) - 1 if order is None else int(order),
                # The response's own convention, e.g. a hybrid response in N3D; ACN/SN3D when it states none
                'channel_order': getattr(ir, 'channel_order', None) or 'acn',
                'normalization': getattr(ir, 'normalization', None) or 'sn3d',
                'binaural': (getattr(ir, 'generation', None) or {}).get('output') == 'binaural',
            }
        return converted


    @staticmethod
    def _sounds_from_impulse_response_sources(impulse_responses):
        if isinstance(impulse_responses, dict):
            raise ValueError("anechoic_sounds is required when impulse_responses is a dict.")
        sound_files = []
        for ir in impulse_responses:
            sound_file = getattr(getattr(ir, 'source', None), 'sound_file', None)
            if not sound_file:
                raise ValueError(f"No anechoic sound given and the source of impulse response '{getattr(ir, 'name', ir)}' has no sound_file.")
            sound_files.append(sound_file)
        return sound_files


    def _prepare_impulse_response(self, ir_id, data):
        """Validate one impulse response and convert it to ACN/SN3D at the auraliser sample rate."""
        ir = data.get('ir')
        if ir is None or not isinstance(ir, np.ndarray) or ir.size == 0:
            raise ValueError(f"Impulse response {ir_id} is empty or invalid.")
        ir = np.asarray(ir, dtype=float)
        if ir.ndim == 1:
            ir = ir[:, np.newaxis]
        elif ir.ndim != 2:
            raise ValueError(f"Impulse response {ir_id} must be 1D or 2D, got {ir.ndim}D.")

        if ir.shape[1] > ir.shape[0]:
            warnings.warn(f"Impulse response {ir_id} is transposed. Impulse response will be transposed to the correct format")
            ir = ir.T

        if data['sampling_rate'] <= 0:
            raise ValueError(f"Impulse response {ir_id} has invalid sample rate {data['sampling_rate']}. Must be positive.")

        if data.get('binaural'):
            # Ears, not spherical harmonics: convolved as they are
            if ir.shape[1] != 2:
                raise ValueError(f"Binaural impulse response {ir_id} must have 2 channels, got {ir.shape[1]}.")
            data['binaural'] = True
            data['ambisonic_order'] = None
            data['ir'] = self._at_sampling_rate(ir_id, ir, data)
            return data
        data['binaural'] = False

        # Ambisonic impulse responses have (order + 1)**2 channels
        order = int(round(np.sqrt(ir.shape[1]))) - 1
        if (order + 1) ** 2 != ir.shape[1]:
            raise ValueError(f"Impulse response {ir_id}, does not have a valid channel count to be a ambisonics impulse response.")

        try:
            given_order = int(data.get('ambisonic_order'))
        except (TypeError, ValueError):
            given_order = None
        if given_order != order:
            warnings.warn(f"Mismatch detected between ambisonic order {data.get('ambisonic_order')} and channel count {ir.shape[1]} of {ir_id}. " +
                          f"Using order {order}.")
        data['ambisonic_order'] = order

        channel_order = data.get('channel_order')
        if not channel_order:
            warnings.warn(f"No channel order noted for {ir_id}. acn assumed.")
            channel_order = 'acn'
        channel_order = channel_order.lower()
        if channel_order not in _CHANNEL_ORDERS:
            raise ValueError(f"Impulse response {ir_id} has unknown channel order '{channel_order}'. Must be one of {sorted(_CHANNEL_ORDERS)}.")

        normalization = data.get('normalization')
        if not normalization:
            normalization = 'sn3d'
            warnings.warn(f"No normalization noted for {ir_id}. {normalization} assumed.")
        normalization = normalization.lower()
        if normalization not in _NORMALIZATIONS:
            raise ValueError(f"Impulse response {ir_id} has unknown normalization '{normalization}'. Must be one of {sorted(_NORMALIZATIONS)}.")

        if normalization == 'n3d':
            warnings.warn(f"Converting {ir_id} from n3d to sn3d normalization.")
            ir = _n3d_to_sn3d(ir)
        data['channel_order'] = 'acn'
        data['normalization'] = 'sn3d'

        data['ir'] = self._at_sampling_rate(ir_id, ir, data)
        return data


    def _at_sampling_rate(self, ir_id, ir, data):
        """The impulse response at :attr:`sampling_rate`, resampled with its frequency response kept."""
        if data['sampling_rate'] != self.sampling_rate:
            warnings.warn(f"Mismatch detected between {ir_id} sample rate of {data['sampling_rate']} Hz, and default sample rate of {self.sampling_rate} Hz. " +
                          f"Impulse response will be resampled to {self.sampling_rate} Hz.")
            # Matching the frequency response keeps the impulse response's gain at the new rate
            ir = pf.dsp.resample(pf.Signal(ir.T, data['sampling_rate']), self.sampling_rate, match_amplitude='freq').time.T
            data['sampling_rate'] = self.sampling_rate
        return ir


    def _pairs(self):
        """Pair each impulse response with its anechoic sound filepath, by id when the ids match, otherwise by position."""
        ir_items = list(self.impulse_responses.items())
        if self.impulse_responses.keys() == self.anechoic_sounds.keys():
            return [(ir_id, ir_data, self.anechoic_sounds[ir_id]['filepath']) for ir_id, ir_data in ir_items]
        sounds = list(self.anechoic_sounds.values())
        return [(ir_id, ir_data, sound['filepath']) for (ir_id, ir_data), sound in zip(ir_items, sounds)]


    # Signal processing

    def import_anechoic_sound(self, filepath, sampling_rate):
        """Read an audio file as a mono (samples,) array at ``sampling_rate``.

        Multichannel files are mixed down to mono.
        """
        sound = pf.io.read_audio(filepath)
        if sound.cshape != (1,):
            warnings.warn(f"Anechoic sound {filepath} has {sound.cshape} channels; mixing down to mono.")
            sound = pf.dsp.average(sound)

        if sound.sampling_rate != sampling_rate:
            sound = pf.dsp.resample(sound, sampling_rate, match_amplitude='time')

        return sound.time.reshape(-1)


    @staticmethod
    def _at_order(ir, order):
        """An ACN/SN3D response padded with zero channels up to ``order``."""
        return np.pad(ir, ((0, 0), (0, (int(order) + 1) ** 2 - ir.shape[1])))


    def convolve(self):
        """Convolve each impulse response with its anechoic sound and sum, in the responses' :attr:`format`.

        Impulse responses that share an anechoic sound are summed before convolving,
        which is equivalent by linearity, so each sound is read and convolved once.
        Ambisonic responses of lower orders are padded with zero channels to the
        highest order present.

        Returns
        -------
        np.ndarray
            The auralised signal, shaped (samples, :attr:`output_channels`).
        """
        order = self.maximum_ambisonics_order
        # Sum the impulse responses per anechoic sound
        summed_impulse_responses = {}
        for ir_id, ir_data, filepath in self._pairs():
            ir = ir_data['ir'] if self.format == "binaural" else self._at_order(ir_data['ir'], order)
            key = str(Path(filepath).resolve())
            current = summed_impulse_responses.get(key)
            if current is None:
                summed_impulse_responses[key] = ir
            else:
                length = max(current.shape[0], ir.shape[0])
                current = np.pad(current, ((0, length - current.shape[0]), (0, 0)))
                current[:ir.shape[0]] += ir
                summed_impulse_responses[key] = current

        convolved = []
        for filepath, ir in summed_impulse_responses.items():
            anechoic_sound = self.import_anechoic_sound(filepath, self.sampling_rate)
            convolved.append(fftconvolve(ir, anechoic_sound[:, np.newaxis], axes=0))

        convolved_sound_summed = np.zeros((max((c.shape[0] for c in convolved), default=0), self.output_channels))
        for c in convolved:
            convolved_sound_summed[:c.shape[0]] += c

        return convolved_sound_summed


    @staticmethod
    def peak_dbfs(signal):
        """Peak absolute value of a signal in dBFS (-inf for silence)."""
        peak = float(np.max(np.abs(signal))) if np.size(signal) else 0.0
        return 20.0 * np.log10(peak) if peak > 0 else -np.inf


    @staticmethod
    def render_gains(signals, level="relative", headroom_db=-1.0):
        """Gains that set the output level of a group of auralised signals.

        Parameters
        ----------
        signals : list of np.ndarray
            Signals returned by :meth:`convolve`, e.g. one per receiver or per source.
        level : str, optional
            ``'relative'`` applies one gain to all signals so the loudest peaks at
            ``headroom_db``, keeping their level differences. ``'individual'`` gives each
            signal its own gain to peak at ``headroom_db``. ``'none'`` returns gains of 1.
        headroom_db : float, optional
            Target peak in dBFS. Default is -1.

        Returns
        -------
        list of float
            One gain per signal.
        """
        if level not in _LEVELS:
            raise ValueError(f"Auraliser level must be one of {sorted(_LEVELS)}.")
        target = 10.0 ** (headroom_db / 20.0)
        peaks = [float(np.max(np.abs(signal))) if np.size(signal) else 0.0 for signal in signals]

        if level == "none":
            return [1.0] * len(peaks)
        if level == "individual":
            return [target / peak if peak > 0 else 1.0 for peak in peaks]
        loudest = max(peaks, default=0.0)
        return [target / loudest if loudest > 0 else 1.0] * len(peaks)


    def export_convolved_sound(self, filepath, convolved_sound=None, gain=None):
        """Apply the output level to an auralised signal and write it as a 32-bit float audio file.

        Parameters
        ----------
        filepath : str
            The path of the audio file to write.
        convolved_sound : np.ndarray, optional
            A signal returned by :meth:`convolve`. If None, :meth:`convolve` is called.
        gain : float, optional
            The gain to apply, e.g. from :meth:`render_gains` for a group of renders.
            If None, it is set by :attr:`level` and :attr:`headroom_db` for this signal
            alone.

        Returns
        -------
        np.ndarray
            The written signal, shaped (samples, channels), as float32.
        """
        exported_sound = self.convolve() if convolved_sound is None else convolved_sound
        if gain is None:
            gain = self.render_gains([exported_sound], self.level, self.headroom_db)[0]

        exported_sound = (exported_sound * gain).astype(np.float32)

        peak_dbfs = self.peak_dbfs(exported_sound)
        if peak_dbfs > 0:
            warnings.warn(f"Auralised signal peaks at {peak_dbfs:+.1f} dBFS and will clip on playback. Use level 'relative' or 'individual', or a lower gain.")

        pf.io.write_audio(pf.Signal(exported_sound.T, self.sampling_rate), filepath, subtype='FLOAT')

        return exported_sound
