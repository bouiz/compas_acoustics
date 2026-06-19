import numpy as np
from math import gcd

from numpy.typing import ArrayLike, NDArray
from scipy import signal
from compas.data import Data

from compas_acoustics.receiver import Receiver
from compas_acoustics.source import Source


__all__ = ['ImpulseResponse']


class ImpulseResponse(Data):
    def __init__(self, name: str, source: Source, receiver: Receiver, ir: np.ndarray, sample_rate: int, simulation_platform: str):
        super().__init__(name=name)
        self.source = source
        self.receiver = receiver
        self.description = f"{receiver.name} - {source.name}"
        self.ir = ir
        self.sample_rate = sample_rate
        self.simulation_platform = simulation_platform
        
        
    def resampled(self, target_sample_rate: int, method: str = "poly") -> NDArray[np.float64]:
        """
        Resample a single 1D impulse response.

        Parameters
        ----------
        ir : ArrayLike
            1D impulse response.
        orig_sr : int
            Original sample rate.
        target_sr : int
            Target sample rate.
        method : str, default="poly"
            Resampling method: "poly" or "fft".

        Returns
        -------
        NDArray[np.float64]
            Resampled 1D impulse response.
        """
        ir_array = np.asarray(self.ir, dtype=np.float64)

        if ir_array.ndim != 1:
            raise ValueError("ir must be a 1D array")

        if self.sample_rate == target_sample_rate:
            return ir_array.copy()

        if method == "poly":
            g = gcd(self.sample_rate, target_sample_rate)
            up = target_sample_rate // g
            down = self.sample_rate // g
            return signal.resample_poly(ir_array, up, down)

        if method == "fft":
            new_len = int(round(len(ir_array) * target_sample_rate / self.sample_rate))
            return signal.resample(ir_array, new_len)

        raise ValueError("method must be 'poly' or 'fft'")