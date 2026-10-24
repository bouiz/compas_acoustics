"""The result of a virtual scattering measurement: coefficients per band and direction, and the polar levels (ISO 17497-1/-2)."""

import numpy as np
from compas.data import Data

from compas_acoustics.defaults import _PackageDtype

from .bands import fill_octaves

__all__ = ["ScatteringResult"]


def _plain(value):
    """``value`` with numpy arrays, scalars and tuples turned into JSON-friendly lists and Python numbers."""
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value


class ScatteringResult(_PackageDtype, Data):
    """Scattering coefficients of a sample, measured by a free-field method (ISO 17497-2; Heimes & Vorländer eq. 3 and eq. 11).

    Use :func:`compas_acoustics.lab.scattering.measure_scattering` to make one.

    Parameters
    ----------
    method : str
        The solver: ``"kirchhoff"`` or ``"bem"``.
    bands : list of int
        The nominal one-third-octave centres in Hz.
    s_bands : list of float
        The random-incidence scattering coefficient per band (Paris weighting of ``s_angle``).
    octave_bands : list of int
        The simulated octaves in Hz: 125...4000 by default, 125...8000 with the extended range.
    s_octaves : list of float
        The mean of the one-third-octave values inside each simulated octave (ISO 354 §8.3, ISO 11654).
    filled_octaves : list of int
        The material octaves that were not simulated and take the nearest simulated octave's value.
    s_angle : numpy.ndarray
        The directional scattering coefficient, shape (n_sources, n_bands): Heimes & Vorländer eq. 3 per frequency, averaged per band (eq. 11).
    source_colatitude, source_azimuth : numpy.ndarray
        The source directions in degrees, shape (n_sources,).
    source_weights : numpy.ndarray
        Their Paris random-incidence weights, summing to 1.
    receiver_directions : numpy.ndarray
        Unit vectors from the sample to the receivers, shape (n_receivers, 3).
    polar_sample_db, polar_reference_db : numpy.ndarray
        The levels at the receivers per source and simulated octave, shape (n_sources, n_octaves, n_receivers), float32: the mean of |p|²
        over the simulation frequencies in the octave, in dB relative to the reference's maximum for the same source and octave.
        Stored rounded to 0.1 dB, the resolution ISO 17497-2 §8.5 presents polar responses at.
    warnings : list of str
        Where the result falls outside the standards or the method's validity.
    generation : dict
        How the result was made: method, preset, band range, speed of sound, grid sizes and the solver's own record.
    """

    def __init__(
        self,
        method,
        bands,
        s_bands,
        octave_bands,
        s_octaves,
        filled_octaves,
        s_angle,
        source_colatitude,
        source_azimuth,
        source_weights,
        receiver_directions,
        polar_sample_db,
        polar_reference_db,
        warnings=None,
        generation=None,
        name=None,
    ):
        super().__init__(name=name)
        self.method = method
        self.bands = [int(b) for b in bands]
        self.s_bands = [float(s) for s in s_bands]
        self.octave_bands = [int(o) for o in octave_bands]
        self.s_octaves = [float(s) for s in s_octaves]
        self.filled_octaves = [int(o) for o in filled_octaves]
        self.s_angle = np.asarray(s_angle, dtype=float)
        self.source_colatitude = np.asarray(source_colatitude, dtype=float)
        self.source_azimuth = np.asarray(source_azimuth, dtype=float)
        self.source_weights = np.asarray(source_weights, dtype=float)
        self.receiver_directions = np.asarray(receiver_directions, dtype=float)
        self.polar_sample_db = np.asarray(polar_sample_db, dtype=np.float32)
        self.polar_reference_db = np.asarray(polar_reference_db, dtype=np.float32)
        self.warnings = list(warnings or [])
        self.generation = dict(generation or {})

    @property
    def scattering_coefficients(self):
        """The coefficients on the 7 material octaves 125...8000 Hz; an octave not simulated takes the nearest simulated one's value."""
        filled, _ = fill_octaves(self.octave_bands, self.s_octaves)
        return [float(s) for s in filled]

    @property
    def __data__(self):
        def polar(levels):
            return np.round(levels.astype(float), 1).tolist()  # 0.1 dB, ISO 17497-2 §8.5

        return {
            "method": self.method,
            "bands": self.bands,
            "s_bands": self.s_bands,
            "octave_bands": self.octave_bands,
            "s_octaves": self.s_octaves,
            "filled_octaves": self.filled_octaves,
            "s_angle": self.s_angle.tolist(),
            "source_colatitude": self.source_colatitude.tolist(),
            "source_azimuth": self.source_azimuth.tolist(),
            "source_weights": self.source_weights.tolist(),
            "receiver_directions": self.receiver_directions.tolist(),
            "polar_sample_db": polar(self.polar_sample_db),
            "polar_reference_db": polar(self.polar_reference_db),
            "warnings": self.warnings,
            "generation": _plain(self.generation),
        }

    @classmethod
    def __from_data__(cls, data):
        return cls(**data)
