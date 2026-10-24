"""Virtual free-field scattering measurement of a sample (ISO 17497-2; Heimes & Vorländer 2025).

The pipeline of :func:`measure_scattering`: plane waves from a source grid onto the sample and its flat reference, the pressures on a
receiver hemisphere, the correlation coefficient per direction and frequency (Heimes & Vorländer eq. 3), its mean per one-third-octave band
(eq. 11), the Paris random-incidence average, and octave values (ISO 354 §8.3, ISO 11654) for the material.
"""

import warnings as _warnings

import numpy as np

from .bands import ISO_BAND_RANGE
from .bands import band_average
from .bands import fill_octaves
from .bands import fold_to_octaves
from .bands import simulation_frequencies
from .bands import third_octave_centres
from .coefficients import random_incidence
from .coefficients import scattering_freefield
from .grids import ReceiverGrid
from .grids import SourceGrid
from .grids import receiver_spacing
from .results import ScatteringResult

__all__ = ["DEFAULT_PRESETS", "measure_scattering"]

METHODS = ("kirchhoff", "bem")
DEFAULT_PRESETS = {"kirchhoff": "fine", "bem": "standard"}
_ISO_TOP = 5000  # Hz, the highest band of ISO 354 and ISO 17497-1/-2
_BEYOND_ISO = "bands above 5000 Hz are beyond ISO 354 and ISO 17497 (100-5000 Hz)"
_FILLED = "the {f} Hz octave was not simulated; the material takes the {g} Hz value"
_FLOOR_DB = -300.0  # the level given where |p|² is 0 (log10(0))


def _check_method(method):
    if method not in METHODS:
        raise ValueError("unknown method {!r}; supported methods are {}".format(method, ", ".join("'{}'".format(m) for m in METHODS)))


def _solver(method, **solver_options):
    """The solver for a method name: the one place that maps names to solvers. BEM is imported only when asked for (it needs Python ≥ 3.11)."""
    _check_method(method)
    if method == "kirchhoff":
        from .solvers import KirchhoffSolver

        return KirchhoffSolver(**solver_options)
    from .solvers.bem import BemSolver

    return BemSolver(**solver_options)


def _polar_levels(p_sample, p_reference, nominal):
    """Levels per source, simulated octave and receiver in dB, relative to the reference's maximum for the same source and octave.

    The level of an octave is the mean of |p|² over the simulation frequencies inside it: the mean per band of its three frequencies
    (Heimes & Vorländer eq. 11), then the mean of the bands inside the octave (ISO 354 §8.3), which is the same with three per band.

    Returns
    -------
    tuple of numpy.ndarray
        ``(sample_db, reference_db)``, float32, shape (n_sources, n_octaves, n_receivers).
    """
    energies = []
    for p in (p_sample, p_reference):
        _, octave = fold_to_octaves(band_average(np.abs(p) ** 2), nominal)  # (n_src, n_rec, n_oct)
        energies.append(np.moveaxis(octave, -1, 1))
    peak = energies[1].max(axis=2, keepdims=True)
    peak = np.where(peak > 0, peak, 1.0)
    out = []
    for energy in energies:
        ratio = energy / peak
        db = 10.0 * np.log10(np.where(ratio > 0, ratio, 1.0))
        out.append(np.where(ratio > 0, np.maximum(db, _FLOOR_DB), _FLOOR_DB).astype(np.float32))
    return out[0], out[1]


def measure_scattering(sample, method="kirchhoff", preset=None, band_range=ISO_BAND_RANGE, air=None, name=None, **solver_options) -> ScatteringResult:
    """Measure the scattering coefficient of a sample in a virtual free field (ISO 17497-2; Heimes & Vorländer 2025).

    The correlation coefficient (Mommertz 2000; Heimes & Vorländer eq. 3) of the sample's and the reference's pressures on a receiver
    hemisphere of radius ``10·d``, at receiver spacing ``min(5°, 50·λ_min/d)`` (ISO 17497-2 §6.2.2, Heimes eq. 6), is computed per source
    direction and frequency, averaged over the three frequencies of each one-third-octave band (eq. 11), weighted over the sources with
    the Paris weights for random incidence, and averaged per octave (ISO 354 §8.3, ISO 11654).

    Parameters
    ----------
    sample : :class:`compas_acoustics.lab.sample.ScatteringSample`
    method : {"kirchhoff", "bem"}, optional
        Physical optics (fast) or the boundary element method (mesh2scattering/NumCalc, needs Python ≥ 3.11 and the ``lab`` extra).
    preset : {"survey", "standard", "fine"}, optional
        The source grid; by default ``DEFAULT_PRESETS[method]``: ``"fine"`` for kirchhoff, ``"standard"`` for bem.
        Comparisons between methods should pass the same preset.
    band_range : tuple of float, optional
        The nominal one-third-octave centres simulated, inclusive: ``ISO_BAND_RANGE`` (100-5000 Hz, ISO 17497-1/-2 §5) by default,
        ``EXTENDED_BAND_RANGE`` for the extension to 8 kHz beyond the standards.
    air : :class:`compas_acoustics.room.AirConditions`, optional
        Gives the speed of sound; the default air by default.
    name : str, optional
        The result's name; the sample's by default.
    **solver_options
        Passed to the solver: ``KirchhoffSolver(shadowing, chunk)`` or ``BemSolver(cache_dir, numcalc_path, max_elements, ...)``.

    Returns
    -------
    :class:`compas_acoustics.lab.results.ScatteringResult`

    Raises
    ------
    ValueError
        If the method is not supported, or the preset or band range is invalid.
    RuntimeError
        If the solver returned non-finite pressures.
    """
    _check_method(method)
    preset = preset or DEFAULT_PRESETS[method]
    if air is None:
        from compas_acoustics.room import AirConditions

        air = AirConditions()
    c = float(air.speed_of_sound)

    nominal, exact = third_octave_centres(band_range)
    frequencies = simulation_frequencies(band_range)
    sources = SourceGrid.preset(preset, sample.symmetry)
    receivers = ReceiverGrid.hemisphere(receiver_spacing(sample.diameter, frequencies.max(), c), radius=10.0 * sample.diameter)

    solver = _solver(method, **solver_options)
    pressures = solver.solve(sample, sources, receivers, frequencies, c)
    if not (np.isfinite(pressures.sample).all() and np.isfinite(pressures.reference).all()):
        raise RuntimeError("{}: the solver returned non-finite pressures for {!r}".format(method, sample.name))

    s_f = scattering_freefield(pressures.sample, pressures.reference, receivers.weights)  # (n_src, n_f)
    s_angle = band_average(s_f)
    s_bands = random_incidence(s_angle, sources)
    octaves, s_octaves = fold_to_octaves(s_bands, nominal)
    _, filled = fill_octaves(octaves, s_octaves)
    polar_sample, polar_reference = _polar_levels(pressures.sample, pressures.reference, nominal)

    messages = list(sample.warnings) + list(solver.warnings(sample, nominal, exact, c))
    if nominal.max() > _ISO_TOP:
        messages.append(_BEYOND_ISO)
    messages.extend(_FILLED.format(f=f, g=g) for f, g in filled)
    for message in messages:
        _warnings.warn(message)

    generation = {
        "method": method,
        "preset": preset,
        "band_range": [float(b) for b in band_range],
        "c": c,
        "receivers": len(receivers.weights),
        "sources": sources.n,
    }
    generation.update(pressures.generation)
    return ScatteringResult(
        method=method,
        bands=nominal.tolist(),
        s_bands=s_bands.tolist(),
        octave_bands=octaves,
        s_octaves=s_octaves.tolist(),
        filled_octaves=[f for f, _ in filled],
        s_angle=s_angle,
        source_colatitude=sources.colatitude,
        source_azimuth=sources.azimuth,
        source_weights=sources.paris_weights,
        receiver_directions=receivers.directions,
        polar_sample_db=polar_sample,
        polar_reference_db=polar_reference,
        warnings=messages,
        generation=generation,
        name=name or sample.name,
    )
