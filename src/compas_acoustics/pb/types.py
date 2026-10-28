"""Values that travel between the worker and its clients without being compas_acoustics Data types."""

import uuid

import numpy as np

__all__ = ["Audio"]


class Audio:
    """A rendered signal: samples shaped ``(samples,)`` or ``(samples, channels)``, its rate, and its format.

    Parameters
    ----------
    samples : array_like
    sampling_rate : int
    format : str
        ``'mono'``, ``'binaural'`` or ``'ambisonic'``.
    """

    def __init__(self, samples, sampling_rate, format="mono"):
        self.samples = np.asarray(samples, dtype=float)
        self.sampling_rate = int(sampling_rate)
        self.format = format
        self.guid = uuid.uuid4()

    @property
    def channels(self):
        return 1 if self.samples.ndim == 1 else self.samples.shape[1]

    def __repr__(self):
        return f"Audio({self.samples.shape}, {self.sampling_rate} Hz, {self.format})"
