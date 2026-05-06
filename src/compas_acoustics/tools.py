from math import gcd

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import signal

__all__ = ['Tools']

class Tools:
    
    @staticmethod
    def resample_signal(ir: ArrayLike, orig_sr: int, target_sr: int, method: str = "poly") -> NDArray[np.float64]:
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
        ir_array = np.asarray(ir, dtype=np.float64)

        if ir_array.ndim != 1:
            raise ValueError("ir must be a 1D array")

        if orig_sr == target_sr:
            return ir_array.copy()

        if method == "poly":
            g = gcd(orig_sr, target_sr)
            up = target_sr // g
            down = orig_sr // g
            return signal.resample_poly(ir_array, up, down)

        if method == "fft":
            new_len = int(round(len(ir_array) * target_sr / orig_sr))
            return signal.resample(ir_array, new_len)

        raise ValueError("method must be 'poly' or 'fft'")
