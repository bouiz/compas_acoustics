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
    """Scattering and diffusion coefficients of a sample, measured by a free-field method (ISO 17497-2; Heimes & Vorländer eq. 3 and eq. 11).

    Use :func:`compas_acoustics.lab.scattering.measure_scattering` to make one.

    Parameters
    ----------
    method : str
        The method: ``"geometric"`` (heuristic, see ``generation["heuristic"]``), ``"kirchhoff"`` or ``"bem"``.
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
        The directional scattering coefficient, shape (n_sources, n_bands): Heimes & Vorländer eq. 3 per frequency, averaged per band (eq. 11);
        for ``"geometric"`` the heuristic ``s_max·(1 − exp(−(k h)²))`` at the bands' exact centres.
    source_colatitude, source_azimuth : numpy.ndarray
        The source directions in degrees, shape (n_sources,).
    source_weights : numpy.ndarray
        Their Paris random-incidence weights, summing to 1.
    receiver_directions : numpy.ndarray
        Unit vectors from the sample to the receivers, shape (n_receivers, 3).
    polar_sample_db, polar_reference_db : numpy.ndarray
        The levels at the receivers per preview source and simulated octave, shape (n_preview, n_octaves, n_receivers), float32: the mean
        of |p|² over the simulation frequencies in the octave, in dB relative to the reference's maximum for the same source and octave
        (for ``"geometric"``: the cells' energy per steradian, the same at every octave). −300 dB where the level is 0.
        Stored rounded to 0.1 dB, the resolution ISO 17497-2 §8.5 presents polar responses at.
    polar_source_indices : list of int, optional
        The preview sources, as indices into the source arrays (``source_colatitude`` ...): the sources nearest to the ISO 17497-2
        Table 1 directions (:func:`compas_acoustics.lab.scattering.preview_sources`). By default the first n_preview sources.
    unconverged_octaves : list of int, optional
        The simulated octaves holding a frequency at which the solver did not converge (BEM; see ``generation["unconverged"]``):
        their values are unreliable.
    warnings : list of str
        Where the result falls outside the standards or the method's validity.
    generation : dict
        How the result was made: method, preset, band range, speed of sound, grid sizes and the solver's own record.
    d_angle : numpy.ndarray, optional
        The normalised directional diffusion coefficient ``d_n = (d − d_ref)/(1 − d_ref)`` (ISO 17497-2 eq. 6 on the intensities at the
        receivers, weighted by their solid angles, for the sample and for the reference), shape (n_sources, n_bands), averaged per band
        like ``s_angle``. It is negative where the sample's response is less uniform than the reference's. None in a result made before d existed.
    d_bands : list of float, optional
        Its random-incidence value per band (the Paris weights of ``s_bands``).
    d_octaves : list of float, optional
        Its mean per simulated octave, like ``s_octaves``.
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
        polar_source_indices=None,
        unconverged_octaves=None,
        warnings=None,
        generation=None,
        name=None,
        d_angle=None,
        d_bands=None,
        d_octaves=None,
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
        if polar_source_indices is None:
            polar_source_indices = range(len(self.polar_sample_db))
        self.polar_source_indices = [int(i) for i in polar_source_indices]
        self.unconverged_octaves = [int(o) for o in (unconverged_octaves or [])]
        self.warnings = list(warnings or [])
        self.generation = dict(generation or {})
        self.d_angle = None if d_angle is None else np.asarray(d_angle, dtype=float)
        self.d_bands = None if d_bands is None else [float(d) for d in d_bands]
        self.d_octaves = None if d_octaves is None else [float(d) for d in d_octaves]

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
            "polar_source_indices": self.polar_source_indices,
            "unconverged_octaves": self.unconverged_octaves,
            "warnings": self.warnings,
            "generation": _plain(self.generation),
            "d_angle": None if self.d_angle is None else self.d_angle.tolist(),
            "d_bands": self.d_bands,
            "d_octaves": self.d_octaves,
        }

    @classmethod
    def __from_data__(cls, data):
        return cls(**data)
