"""Source directivities: the first-order patterns by name, and measured ones from a SOFA file or a pyroomacoustics object.

:attr:`compas_acoustics.Source.directivity` takes

- a name of :data:`~compas_acoustics.room.pra.CARDIOID_P` (``"omnidirectional"``, ``"cardioid"``, ...);
- the path of a SOFA file of a measured source (``"loudspeaker.sofa"``, or ``"loudspeaker.sofa#2"`` for its
  measurement 2), read by pyroomacoustics' :class:`~pyroomacoustics.directivities.MeasuredDirectivityFile`;
- a :class:`pyroomacoustics.directivities.Directivity` (a measured balloon you built, say), used as given:
  orient it yourself.

A named or SOFA pattern faces :attr:`Source.direction`, its +x axis turned there. A measured pattern is expected
with a gain of 1 on its axis: its impulse responses filter every path that leaves the source.

The image sources (:class:`~compas_acoustics.Mono`, the hybrid methods), the ray tracer and the hybrid's
diffracted paths (UDFA) all apply it; the hybrid weights each image and diffracted path per octave band
(:func:`band_gains`).
"""

from __future__ import annotations

import math
import os

import numpy as np

__all__ = ["is_omnidirectional", "to_pra", "orientation", "band_gains"]


def is_omnidirectional(directivity) -> bool:
    return directivity is None or (isinstance(directivity, str) and directivity == "omnidirectional")


def _split_sofa(directivity: str):
    path, _, measurement = directivity.partition("#")
    if not path.lower().endswith(".sofa"):
        return None, None
    if measurement == "":
        measurement = 0
    elif measurement.lstrip("-").isdigit():
        measurement = int(measurement)
    return path, measurement


def is_sofa(directivity) -> bool:
    return isinstance(directivity, str) and _split_sofa(directivity)[0] is not None


def orientation(direction):
    """The :class:`pyroomacoustics.directivities.Rotation3D` that turns +x to ``direction``."""
    from pyroomacoustics.directivities import Rotation3D

    x, y, z = (float(v) for v in direction)
    norm = math.sqrt(x * x + y * y + z * z)
    if norm == 0.0:
        raise ValueError("a source direction cannot be zero")
    azimuth = math.degrees(math.atan2(y, x))
    elevation = math.degrees(math.asin(max(-1.0, min(1.0, z / norm))))
    return Rotation3D([-elevation, azimuth], rot_order="yz", degrees=True)


class _Variable:
    """A SOFA variable as python-sofa presents it, which pyroomacoustics' readers expect."""

    def __init__(self, values, units=None, kind=None):
        self._values = np.asarray(values)
        self.Units = units
        self.Type = kind

    def get_values(self):
        return self._values

    def dimensions(self):
        if self._values.ndim == 3 and self._values.shape[-1] == 1:
            return ("M", "C", "I")
        return tuple("MCN"[: self._values.ndim])


class _SofarFile:
    """A SOFA file read with sofar (a compas_acoustics dependency), shaped like python-sofa's ``Database``."""

    def __init__(self, path):
        from types import SimpleNamespace

        import sofar

        f = sofar.read_sofa(str(path), verify=False, verbose=False)

        def variable(name):
            return _Variable(getattr(f, name), getattr(f, name + "_Units", None), getattr(f, name + "_Type", None))

        self.convention = SimpleNamespace(name=f.GLOBAL_SOFAConventions)
        self.Data = SimpleNamespace(IR=_Variable(f.Data_IR), SamplingRate=_Variable(np.atleast_1d(f.Data_SamplingRate)))
        self.Source = SimpleNamespace(Position=variable("SourcePosition"))
        self.Receiver = SimpleNamespace(Position=variable("ReceiverPosition"))


def read_sofa(path, fs=None):
    """pyroomacoustics' SOFA reader (``directivities.sofa.open_sofa_file``) on sofar instead of python-sofa.

    Same conventions (GeneralFIR, SimpleFreeFieldHRIR) and the same handling of the DIRPAT files; returns
    ``(irs, fs, source directions, receiver locations, source labels, microphone labels)``.
    """
    from pathlib import Path

    from pyroomacoustics.datasets.sofa import get_sofa_db_info
    from pyroomacoustics.directivities import sofa as pra_sofa

    from compas_acoustics.room.geometry import PRA_STATE_LOCK

    path = Path(path)
    file = _SofarFile(path)
    if pra_sofa.is_dirpat(path):
        # pyroomacoustics' DIRPAT reader only checks that its database knows the file, which it does only for
        # files it downloaded itself; the database is not used otherwise
        with PRA_STATE_LOCK:
            known = pra_sofa.get_sofa_db
            pra_sofa.get_sofa_db = lambda: {path.stem: None}
            try:
                ir, fs, src_loc, mic_loc = pra_sofa._read_dirpat(file, path.stem, fs)
            finally:
                pra_sofa.get_sofa_db = known
    elif file.convention.name == "SimpleFreeFieldHRIR":
        ir, fs, src_loc, mic_loc = pra_sofa._read_simple_free_field_hrir(file, fs)
    elif file.convention.name == "GeneralFIR":
        ir, fs, src_loc, mic_loc = pra_sofa._read_general_fir(file, fs)
    else:
        raise NotImplementedError(f"SOFA convention {file.convention.name} not implemented")
    source_labels = mic_labels = None
    info = get_sofa_db_info().get(path.stem, {})
    contains = info.get("contains")
    if info.get("type") == "sources" and contains is not None and len(contains) == ir.shape[0]:
        source_labels = contains
    elif info.get("type") == "microphones" and contains is not None and len(contains) == ir.shape[1]:
        mic_labels = contains
    return ir, fs, src_loc, mic_loc, source_labels, mic_labels


def to_pra(directivity, direction, fs: int):
    """The pyroomacoustics directivity of a source's ``directivity`` facing ``direction``; None when omnidirectional.

    A measured directivity is (re)built on compas_acoustics' octave grid (its ray energies are per band of the room)
    and at ``fs``.
    """
    import pyroomacoustics as pra

    from compas_acoustics.room.geometry import pra_octave_grid
    from compas_acoustics.room.pra import CARDIOID_P
    from compas_acoustics.room.pra import vector_to_pra

    if is_omnidirectional(directivity):
        return None
    if isinstance(directivity, str):
        if directivity in CARDIOID_P:
            from compas.geometry import Vector

            return pra.CardioidFamily(orientation=vector_to_pra(Vector(*direction)), p=CARDIOID_P[directivity], gain=1.0)
        path, measurement = _split_sofa(directivity)
        if path is None:
            raise ValueError(f"Directivity '{directivity}' is not supported: give one of {sorted(CARDIOID_P)}, a .sofa file or a pyroomacoustics directivity.")
        if not os.path.isfile(path):
            raise ValueError(f"Directivity file '{path}' does not exist.")
        with pra_octave_grid():
            measured = pra.directivities.MeasuredDirectivityFile(path, fs=int(fs), file_reader_callback=read_sofa)
            return measured.get_source_directivity(measurement, orientation(direction))
    if isinstance(directivity, pra.directivities.MeasuredDirectivity):
        irs = np.asarray(directivity._irs, dtype=float)
        if int(directivity.fs) != int(fs):
            from math import gcd

            from scipy.signal import resample_poly

            g = gcd(int(fs), int(directivity.fs))
            irs = resample_poly(irs, int(fs) // g, int(directivity.fs) // g, axis=-1)
        with pra_octave_grid():
            return pra.directivities.MeasuredDirectivity(directivity._orientation, directivity._original_grid, irs, int(fs))
    if isinstance(directivity, pra.directivities.Directivity):
        return directivity
    raise ValueError(f"Directivity {directivity!r} is not supported.")


def band_gains(directivity, directions: np.ndarray, bands=None) -> np.ndarray:
    """Amplitude of ``directivity`` towards each of ``directions`` (3, M), shape (M, 1) for a broadband pattern or (M, n_bands).

    A measured (impulse-response) pattern gives the root mean square of its magnitude over each octave of
    ``bands`` (default :data:`~compas_acoustics.room.geometry.SYNTHESIS_BANDS`); a broadband one its response.
    None gives ones.
    """
    directions = np.asarray(directions, dtype=float).reshape(3, -1)
    if directivity is None:
        return np.ones((directions.shape[1], 1))
    azimuth = np.arctan2(directions[1], directions[0])
    colatitude = np.arccos(np.clip(directions[2] / np.linalg.norm(directions, axis=0), -1.0, 1.0))
    response = np.asarray(directivity.get_response(azimuth=azimuth, colatitude=colatitude, degrees=False), dtype=float)
    if not getattr(directivity, "is_impulse_response", False):
        return response.reshape(-1, 1)
    if bands is None:
        from compas_acoustics.room.geometry import SYNTHESIS_BANDS

        bands = SYNTHESIS_BANDS
    response = response.reshape(directions.shape[1], -1)
    n_fft = max(1024, 1 << int(np.ceil(np.log2(response.shape[1]))))
    power = np.abs(np.fft.rfft(response, n=n_fft, axis=1)) ** 2
    freqs = np.fft.rfftfreq(n_fft, 1.0 / float(directivity.fs))
    out = np.empty((response.shape[0], len(bands)))
    for k, centre in enumerate(bands):
        inside = (freqs >= centre / math.sqrt(2.0)) & (freqs < centre * math.sqrt(2.0))
        if not np.any(inside):  # an octave narrower than the resolution: the nearest bin
            inside = np.abs(freqs - centre) == np.min(np.abs(freqs - centre))
        out[:, k] = np.sqrt(power[:, inside].mean(axis=1))
    return out
