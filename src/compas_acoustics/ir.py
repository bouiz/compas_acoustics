import json
from datetime import datetime
from datetime import timezone
from pathlib import Path

import numpy as np
from compas.data import Data

from compas_acoustics.defaults import Defaults
from compas_acoustics.defaults import _PackageDtype
from compas_acoustics.receiver import Receiver
from compas_acoustics.source import Source

__all__ = ["ImpulseResponse"]


class ImpulseResponse(_PackageDtype, Data):
    """An impulse response between a source and a receiver, stored at its raw level.

    :attr:`ir` is peak-normalized by default (the peak of all channels, :attr:`normalization_factor`);
    :meth:`denormalized` gives the raw level, and :meth:`denormalize_group` scales a group so that its loudest response
    peaks at 1. Both keep the level differences between responses.

    Parameters
    ----------
    name : str
    source : Source
    receiver : Receiver
    ir : np.ndarray
        The raw response, (samples,) or (samples, channels).
    sampling_rate : int
    normalized : bool, optional
        Whether :attr:`ir` is peak-normalized; default True.
    generation : dict, optional
        The method and its settings, written to the JSON metadata on export.
    ambisonic_order : int, optional
        None for a mono or binaural response.
    normalization : str, optional
        ``'sn3d'`` or ``'n3d'``.
    channel_order : str, optional
        ``'acn'``.
    """

    def __init__(
        self,
        name: str,
        source: Source,
        receiver: Receiver,
        ir: np.ndarray,
        sampling_rate: int,
        normalized: bool = True,
        generation: dict = None,
        ambisonic_order: int = None,
        normalization: str = None,
        channel_order: str = None,
    ):
        super().__init__(name=name)
        self.source = source
        self.receiver = receiver
        self.description = f"{receiver.name} - {source.name}"
        self.ir = ir
        self.sampling_rate = sampling_rate
        self.normalized = normalized
        self.generation = dict(generation) if generation else {}
        self.ambisonic_order = ambisonic_order
        self.normalization = normalization
        self.channel_order = channel_order

    @property
    def __data__(self):
        # The raw response as a JSON list: long or many responses are better written with to_wav
        return {
            "name": self.name,
            "source": self.source,
            "receiver": self.receiver,
            "ir": self._raw_ir,
            "sampling_rate": self.sampling_rate,
            "normalized": self.normalized,
            "generation": self.generation,
            "ambisonic_order": self.ambisonic_order,
            "normalization": self.normalization,
            "channel_order": self.channel_order,
        }

    @property
    def ir(self):
        """np.ndarray : Peak-normalized when :attr:`normalized`; setting it sets the raw response."""
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
    def format(self):
        """str : ``'mono'``, ``'binaural'`` (two channels, no ambisonic order) or ``'ambisonic'``."""
        if self.n_channels == 1:
            return "mono"
        if self.n_channels == 2 and self.ambisonic_order is None:
            return "binaural"
        return "ambisonic"

    @property
    def level_db(self):
        """float : The normalization factor in dB (20 log10 of the raw peak)."""
        return 20.0 * np.log10(self.normalization_factor)

    def denormalized(self):
        """A copy of the raw impulse response."""
        return self._raw_ir.copy()

    @staticmethod
    def denormalize_group(impulse_responses):
        """The raw arrays of the responses, scaled together so that the loudest peaks at 1.

        Parameters
        ----------
        impulse_responses : list or dict of ImpulseResponse

        Returns
        -------
        list or dict of np.ndarray
        """
        is_dict = isinstance(impulse_responses, dict)
        items = list(impulse_responses.values()) if is_dict else list(impulse_responses)
        if not items:
            return {} if is_dict else []

        reference = max(ir.normalization_factor for ir in items)
        scaled = [ir._raw_ir / reference for ir in items]
        return dict(zip(impulse_responses.keys(), scaled)) if is_dict else scaled

    def direct_sound_index(self) -> int:
        """Sample of the direct sound.

        On the package scale the geometric one, distance / c, even when an early ray-traced bin is louder or an obstacle
        blocks it; otherwise the largest sample of the omnidirectional channel.
        """
        if self.generation.get("scale") == "package":
            c = self.generation.get("speed_of_sound") or Defaults().DEFAULT_SPEED_OF_SOUND
            distance = float(np.linalg.norm(np.subtract(self.receiver.position, self.source.position)))
            return int(round(distance / c * self.sampling_rate))
        omni = self._raw_ir if self._raw_ir.ndim == 1 else self._raw_ir[:, 0]
        return int(np.argmax(omni**2))

    @property
    def metadata(self):
        """dict : The response and how it was generated, as written next to it on export."""
        import compas_acoustics

        return {
            "metadata_version": 1,
            "compas_acoustics_version": compas_acoustics.__version__,
            "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "name": self.name,
            "description": self.description,
            "sampling_rate": self.sampling_rate,
            "samples": int(self._raw_ir.shape[0]),
            "channels": self.n_channels,
            "generation": self.generation,
            "normalization": {
                "method": "peak",
                "wav_normalized": True,
                "factor": self.normalization_factor,
                "level_db": self.level_db,
            },
            "ambisonics": None
            if self.ambisonic_order is None
            else {
                "order": int(self.ambisonic_order),
                "normalization": self.normalization,
                "channel_order": self.channel_order,
            },
            "source": {
                "name": self.source.name,
                "position": [float(v) for v in self.source.position],
                "directivity": getattr(self.source, "directivity", None),
                "direction": [float(v) for v in self.source.direction],
                "sound_file": getattr(self.source, "sound_file", None),
            },
            "receiver": {
                "name": self.receiver.name,
                "position": [float(v) for v in self.receiver.position],
                "direction": [float(v) for v in self.receiver.direction],
                "up": [float(v) for v in self.receiver.up],
            },
        }

    def to_wav(self, filepath):
        """Write the peak-normalized response as a 32-bit float WAV, and :attr:`metadata` with the normalization factor
        as JSON beside it.

        Returns
        -------
        tuple of Path
            The WAV and JSON files.
        """
        import soundfile as sf

        wav_path = Path(filepath)
        json_path = wav_path.with_suffix(".json")
        sf.write(str(wav_path), self._raw_ir / self.normalization_factor, int(self.sampling_rate), subtype="FLOAT")
        json_path.write_text(json.dumps(self.metadata, indent=4))
        return wav_path, json_path

    @classmethod
    def from_wav(cls, filepath, source: Source = None, receiver: Receiver = None, normalized: bool = True):
        """Read a WAV written by :meth:`to_wav` at the raw level of its JSON metadata; without metadata the WAV is raw.

        Parameters
        ----------
        filepath : str
        source, receiver : Source, Receiver, optional
            Used instead of those of the metadata.
        normalized : bool, optional
            As in :class:`ImpulseResponse`.

        Returns
        -------
        ImpulseResponse
        """
        import soundfile as sf

        wav_path = Path(filepath)
        json_path = wav_path.with_suffix(".json")
        data, sampling_rate = sf.read(str(wav_path), dtype="float64")
        metadata = json.loads(json_path.read_text()) if json_path.exists() else {}

        normalization = metadata.get("normalization", {})
        if normalization.get("wav_normalized"):
            data = data * normalization["factor"]

        if source is None:
            source_data = metadata.get("source", {})
            source = Source(source_data.get("name", "source"), *source_data.get("position", [0.0, 0.0, 0.0]), directivity=source_data.get("directivity"))
        if receiver is None:
            receiver_data = metadata.get("receiver", {})
            receiver = Receiver(
                receiver_data.get("name", "receiver"), *receiver_data.get("position", [0.0, 0.0, 0.0]), direction=receiver_data.get("direction"), up=receiver_data.get("up")
            )
        ambisonics = metadata.get("ambisonics") or {}

        return cls(
            name=metadata.get("name", wav_path.stem),
            source=source,
            receiver=receiver,
            ir=data,
            sampling_rate=sampling_rate,
            normalized=normalized,
            generation=metadata.get("generation"),
            ambisonic_order=ambisonics.get("order"),
            normalization=ambisonics.get("normalization"),
            channel_order=ambisonics.get("channel_order"),
        )
