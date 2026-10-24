"""An :class:`~compas_acoustics.materials.AcousticMaterial` from a given absorption and a measured scattering (ISO 17497-1 §6.3.4)."""

import warnings as _warnings

import numpy as np

from compas_acoustics.defaults import Defaults
from compas_acoustics.materials import AcousticMaterial

__all__ = ["lab_material"]

_CLIPPED = "{kind} coefficient {value:.2f} in the {band} Hz octave clipped to {limit:g}"
_MISSING = "no {kind} coefficient in the {band} Hz octave"
_ABSORBING = "ISO 17497-1 6.3.4: the sample absorbs more than 0.5, the scattering coefficient was computed for a rigid sample"
_RIGID_LIMIT = 0.5


def _octaves(values, kind):
    """``values`` as one float per material octave: a single value fills them all; anything else must have one per octave."""
    bands = [int(b) for b in Defaults().DEFAULT_FREQUENCY_BANDS]
    array = np.asarray(values, dtype=float).reshape(-1)
    if array.size == 1:
        array = np.repeat(array, len(bands))
    if array.size != len(bands):
        raise ValueError("{} needs 1 or {} values, one per octave {}-{} Hz, got {}".format(kind, len(bands), bands[0], bands[-1], array.size))
    return bands, array


def _clipped(values, kind):
    """The values in [0, 1], each one clipped warned about; a NaN raises ``ValueError`` naming its octave."""
    bands, array = _octaves(values, kind)
    out = []
    for band, value in zip(bands, array):
        if np.isnan(value):
            raise ValueError(_MISSING.format(kind=kind, band=band))
        limit = min(max(float(value), 0.0), 1.0)
        if limit != value:
            _warnings.warn(_CLIPPED.format(kind=kind, value=float(value), band=band, limit=limit), stacklevel=3)
        out.append(limit)
    return out


def lab_material(name, absorption, scattering=None, category=None, description=None) -> AcousticMaterial:
    """An acoustic material from a given absorption and, optionally, a scattering measured with :func:`~compas_acoustics.lab.measure_scattering`.

    Parameters
    ----------
    name : str
        The material's name.
    absorption : float or sequence of float or object
        The random-incidence absorption coefficients: one value for every octave, or 7 values for 125...8000 Hz. Or anything with an
        ``absorption_coefficients`` attribute (an :class:`~compas_acoustics.materials.AcousticMaterial`), whose 7 values are taken.
    scattering : :class:`~compas_acoustics.lab.results.ScatteringResult`, optional
        The measured scattering: its ``scattering_coefficients`` (the 7 octaves 125...8000 Hz, an octave not simulated filled from the nearest
        one) become the material's. None leaves the material without scattering coefficients.
    category : str or :class:`~compas_acoustics.materials.MaterialCategory`, optional
    description : str, optional

    Returns
    -------
    :class:`~compas_acoustics.materials.AcousticMaterial`
        With ``data_origin`` ``"compas_acoustics.lab: alpha_s given; s by {method} (preset {preset})"`` (``", heuristic"`` after the
        preset for the geometric method) or ``"compas_acoustics.lab: alpha_s given; no scattering"``.

    Warns
    -----
    UserWarning
        For a coefficient outside [0, 1], which is clipped (a material accepts nothing else), and for an absorption above 0.5 in any octave
        together with a scattering result: ISO 17497-1 §6.3.4 requires a rigid sample for the scattering coefficient.

    Raises
    ------
    ValueError
        If a coefficient is NaN (the message names the octave), or there are neither 1 nor 7 values.
    """
    values = getattr(absorption, "absorption_coefficients", absorption)
    alpha = _clipped(values, "absorption")
    s = None
    origin = "compas_acoustics.lab: alpha_s given; no scattering"
    if scattering is not None:
        s = _clipped(scattering.scattering_coefficients, "scattering")
        generation = scattering.generation
        heuristic = ", heuristic" if generation.get("heuristic") else ""
        origin = "compas_acoustics.lab: alpha_s given; s by {} (preset {}{})".format(scattering.method, generation["preset"], heuristic)
        if max(alpha) > _RIGID_LIMIT:
            _warnings.warn(_ABSORBING, stacklevel=2)
    return AcousticMaterial(name, alpha, s, description=description, category=category, data_origin=origin)
