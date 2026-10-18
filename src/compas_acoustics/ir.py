import json
from datetime import datetime
from datetime import timezone
from math import gcd
from pathlib import Path

import numpy as np
from compas.data import Data
from numpy.typing import NDArray
from scipy import signal

from compas_acoustics.defaults import Defaults
from compas_acoustics.receiver import Receiver
from compas_acoustics.source import Source

__all__ = ['ImpulseResponse']

# Version of the JSON metadata written next to exported impulse responses
_METADATA_VERSION = 1


class ImpulseResponse(Data):
    """An impulse response between a source and a receiver.

    The impulse response is stored at its raw (physical) level together with its
    normalisation factor, the peak absolute value across all channels. By default
    :attr:`ir` returns the peak-normalised response; the raw level, which carries
    distance and level differences between impulse responses, is kept and can be
    recovered with :meth:`denormalized` or :meth:`denormalize_group`.

    Parameters
    ----------
    name : str
        The name of the impulse response.
    source : Source
        The source.
    receiver : Receiver
        The receiver.
    ir : np.ndarray
        The raw impulse response, shaped (samples,) or (samples, channels).
    sampling_rate : int
        The sample rate in Hz.
    normalized : bool, optional
        Whether :attr:`ir` returns the peak-normalised response. Default is True.
    generation : dict, optional
        How the impulse response was generated, e.g. the method and its settings.
        Written to the JSON metadata on export.
    ambisonic_order : int, optional
        The ambisonic order of a multichannel response, or None for a mono or
        non-ambisonic one.
    normalisation : str, optional
        The ambisonic normalisation, ``'sn3d'`` or ``'n3d'``.
    channel_order : str, optional
        The ambisonic channel order, ``'acn'``.
    """

    def __init__(self, name: str, source: Source, receiver: Receiver, ir: np.ndarray, sampling_rate: int,
                 normalized: bool = True, generation: dict = None, ambisonic_order: int = None, normalisation: str = None,
                 channel_order: str = None):
        super().__init__(name=name)
        self.source = source
        self.receiver = receiver
        self.description = f"{receiver.name} - {source.name}"
        self.ir = ir
        self.sampling_rate = sampling_rate
        self.normalized = normalized
        self.generation = dict(generation) if generation else {}
        self.ambisonic_order = ambisonic_order
        self.normalisation = normalisation
        self.channel_order = channel_order


    @property
    def __data__(self):
        # The raw response, as a list in JSON: for long or many responses use :meth:`to_wav`, which also keeps the metadata
        return {
            "name": self.name,
            "source": self.source,
            "receiver": self.receiver,
            "ir": self._raw_ir,
            "sampling_rate": self.sampling_rate,
            "normalized": self.normalized,
            "generation": self.generation,
            "ambisonic_order": self.ambisonic_order,
            "normalisation": self.normalisation,
            "channel_order": self.channel_order,
        }


    @property
    def ir(self):
        """np.ndarray : The impulse response, peak-normalised when :attr:`normalized` is True.

        Setting it sets the raw impulse response.
        """
        if self.normalized:
            return self._raw_ir / self.normalization_factor
        return self._raw_ir


    @ir.setter
    def ir(self, ir):
        ir = np.asarray(ir, dtype=float)
        if ir.ndim not in (1, 2):
            raise ValueError(f"An impulse response is shaped (samples,) or (samples, channels), not {ir.shape}")
        self._raw_ir = ir
        peak = float(np.max(np.abs(self._raw_ir))) if self._raw_ir.size else 0.0
        # A silent impulse response is left unscaled
        self._normalization_factor = peak if peak > 0 else 1.0


    @property
    def normalization_factor(self):
        """float : Peak absolute value of the raw impulse response across all channels."""
        return self._normalization_factor


    @property
    def n_channels(self):
        """int : The number of channels."""
        return 1 if self._raw_ir.ndim == 1 else self._raw_ir.shape[1]


    @property
    def level_db(self):
        """float : The normalisation factor in dB (20 log10 of the raw peak)."""
        return 20.0 * np.log10(self.normalization_factor)


    def denormalized(self):
        """Return the raw impulse response, at its physical level.

        Returns
        -------
        np.ndarray
            A copy of the raw impulse response.
        """
        return self._raw_ir.copy()


    @staticmethod
    def denormalize_group(impulse_responses):
        """Return impulse responses at their relative levels, scaled to the loudest one.

        Each impulse response is scaled by its normalisation factor relative to the
        largest factor in the group, so the loudest has a peak of 1 and the others keep
        their level differences below it.

        Parameters
        ----------
        impulse_responses : list of ImpulseResponse or dict of ImpulseResponse

        Returns
        -------
        list of np.ndarray or dict of np.ndarray
            The scaled impulse responses, in the same container type as the input.
        """
        is_dict = isinstance(impulse_responses, dict)
        items = list(impulse_responses.values()) if is_dict else list(impulse_responses)
        if not items:
            return {} if is_dict else []

        reference = max(ir.normalization_factor for ir in items)
        scaled = [ir._raw_ir / reference for ir in items]
        return dict(zip(impulse_responses.keys(), scaled)) if is_dict else scaled


    def resampled(self, target_sampling_rate: int, method: str = "poly") -> NDArray[np.float64]:
        """Return :attr:`ir` resampled to ``target_sampling_rate``, every channel along axis 0.

        Parameters
        ----------
        target_sampling_rate : int
            Target sample rate in Hz.
        method : str, default="poly"
            Resampling method: "poly" or "fft".

        Returns
        -------
        NDArray[np.float64]
            The resampled impulse response, shaped as :attr:`ir`.
        """
        ir_array = np.asarray(self.ir, dtype=np.float64)

        if self.sampling_rate == target_sampling_rate:
            return ir_array.copy()

        if method == "poly":
            g = gcd(int(self.sampling_rate), int(target_sampling_rate))
            up = int(target_sampling_rate) // g
            down = int(self.sampling_rate) // g
            return signal.resample_poly(ir_array, up, down, axis=0)

        if method == "fft":
            new_len = int(round(len(ir_array) * target_sampling_rate / self.sampling_rate))
            return signal.resample(ir_array, new_len, axis=0)

        raise ValueError("method must be 'poly' or 'fft'")


    def direct_sound_index(self, speed_of_sound: float = None) -> int:
        """Sample of the direct sound.

        On the package scale (``generation['scale'] == 'package'``) the direct
        sound sits on its geometric sample, distance / c, even when early
        ray-traced histogram bins are louder or an obstacle blocks it.
        Otherwise the largest sample of the omnidirectional channel.
        """
        if self.generation.get("scale") == "package":
            c = speed_of_sound or self.generation.get("speed_of_sound") or Defaults().DEFAULT_SPEED_OF_SOUND
            distance = float(np.linalg.norm(np.subtract(self.receiver.position, self.source.position)))
            return int(round(distance / c * self.sampling_rate))
        omni = self._raw_ir if self._raw_ir.ndim == 1 else self._raw_ir[:, 0]
        return int(np.argmax(omni ** 2))


    # Export and import

    @property
    def metadata(self):
        """dict : Metadata describing the impulse response and how it was generated."""
        import compas_acoustics

        return {
            'metadata_version': _METADATA_VERSION,
            'compas_acoustics_version': compas_acoustics.__version__,
            'created': datetime.now(timezone.utc).isoformat(timespec='seconds'),
            'name': self.name,
            'description': self.description,
            'sampling_rate': self.sampling_rate,
            'samples': int(self._raw_ir.shape[0]),
            'channels': self.n_channels,
            'generation': self.generation,
            'normalization': {
                'method': 'peak',
                'wav_normalized': True,
                'factor': self.normalization_factor,
                'level_db': self.level_db,
            },
            'ambisonics': None if self.ambisonic_order is None else {
                'order': int(self.ambisonic_order),
                'normalisation': self.normalisation,
                'channel_order': self.channel_order,
            },
            'source': {
                'name': self.source.name,
                'position': [float(v) for v in self.source.position],
                'directivity': getattr(self.source, 'directivity', None),
                'direction': [float(v) for v in self.source.direction],
                'sound_file': getattr(self.source, 'sound_file', None),
            },
            'receiver': {
                'name': self.receiver.name,
                'position': [float(v) for v in self.receiver.position],
                'direction': [float(v) for v in self.receiver.direction],
                'up': [float(v) for v in self.receiver.up],
            },
        }


    def to_wav(self, filepath):
        """Write the peak-normalised impulse response as a 32-bit float WAV and its metadata as JSON.

        The JSON file has the same name with a ``.json`` extension and holds the
        normalisation factor, so the raw level can be restored with :meth:`from_wav`
        or any other tool.

        Parameters
        ----------
        filepath : str
            The path of the WAV file.

        Returns
        -------
        tuple of Path
            The paths of the WAV and JSON files.
        """
        import soundfile as sf

        wav_path = Path(filepath)
        json_path = wav_path.with_suffix('.json')
        sf.write(str(wav_path), self._raw_ir / self.normalization_factor, int(self.sampling_rate), subtype='FLOAT')
        json_path.write_text(json.dumps(self.metadata, indent=4))
        return wav_path, json_path


    @classmethod
    def from_wav(cls, filepath, source: Source = None, receiver: Receiver = None, normalized: bool = True):
        """Read an impulse response WAV and, if present, its JSON metadata.

        The raw level is restored from the normalisation factor in the metadata.
        Without metadata the WAV is taken as the raw impulse response.

        Parameters
        ----------
        filepath : str
            The path of the WAV file.
        source, receiver : Source, Receiver, optional
            Used instead of the ones described in the metadata.
        normalized : bool, optional
            Whether :attr:`ir` returns the peak-normalised response. Default is True.

        Returns
        -------
        ImpulseResponse
        """
        import soundfile as sf

        wav_path = Path(filepath)
        json_path = wav_path.with_suffix('.json')
        data, sampling_rate = sf.read(str(wav_path), dtype='float64')
        metadata = json.loads(json_path.read_text()) if json_path.exists() else {}

        normalization = metadata.get('normalization', {})
        if normalization.get('wav_normalized'):
            data = data * normalization['factor']

        if source is None:
            source_data = metadata.get('source', {})
            source = Source(source_data.get('name', 'source'), *source_data.get('position', [0.0, 0.0, 0.0]),
                            directivity=source_data.get('directivity'))
        if receiver is None:
            receiver_data = metadata.get('receiver', {})
            receiver = Receiver(receiver_data.get('name', 'receiver'), *receiver_data.get('position', [0.0, 0.0, 0.0]),
                                direction=receiver_data.get('direction'), up=receiver_data.get('up'))
        ambisonics = metadata.get('ambisonics') or {}

        return cls(
            name=metadata.get('name', wav_path.stem),
            source=source,
            receiver=receiver,
            ir=data,
            sampling_rate=sampling_rate,
            normalized=normalized,
            generation=metadata.get('generation'),
            ambisonic_order=ambisonics.get('order'),
            normalisation=ambisonics.get('normalisation'),
            channel_order=ambisonics.get('channel_order'),
        )
