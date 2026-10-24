"""Virtual free-field scattering measurement of a sample (ISO 17497-2; Heimes & Vorländer 2025).

The pipeline of :func:`measure_scattering`: plane waves from a source grid onto the sample and its flat reference, the pressures on a
receiver hemisphere, the correlation coefficient per direction and frequency (Heimes & Vorländer eq. 3), its mean per one-third-octave band
(eq. 11), the Paris random-incidence average, and octave values (ISO 354 §8.3, ISO 11654) for the material.
"""

import warnings as _warnings

import numpy as np

from compas_acoustics.defaults import Defaults

from .bands import ISO_BAND_RANGE
from .bands import _nearest_octave
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

__all__ = ["DEFAULT_PRESETS", "measure_scattering", "preview_sources"]

METHODS = ("kirchhoff", "bem")
DEFAULT_PRESETS = {"kirchhoff": "fine", "bem": "standard"}
_ISO_TOP = 5000  # Hz, the highest band of ISO 354 and ISO 17497-1/-2
_BEYOND_ISO = "bands above 5000 Hz are beyond ISO 354 and ISO 17497 (100-5000 Hz)"
_FILLED = "the {f} Hz octave was not simulated; the material takes the {g} Hz value"
_FLOOR_DB = -300.0  # the level given where |p|² is 0 (log10(0))
_BATCH_BYTES = 512 * 2**20  # the pressures of a source batch (both bodies); the reduction's temporaries take about as much again
_LONG_RUN = 600.0  # s: a run estimated longer than this is warned about before it starts
_SLOW = (
    "{method}: this run takes about {time} ({sources} sources x {frequencies} frequencies x {receivers} receivers x {faces} faces, {workers} "
    "workers); the 'standard' or 'survey' preset, a narrower band_range or a smaller diameter is faster"
)
_UNCONVERGED = "bem: NumCalc did not converge for the {bodies} at {frequencies} Hz; s in the {octaves} Hz octaves is unreliable"


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


def _duration(seconds):
    if seconds < 3600:
        return "{:.0f} min".format(seconds / 60)
    return "{:.1f} h".format(seconds / 3600)


def preview_sources(sources, symmetry="none"):
    """The sources whose polar levels a result keeps: those nearest the ISO 17497-2 Table 1 directions (the ``"survey"`` preset).

    Parameters
    ----------
    sources : :class:`compas_acoustics.lab.grids.SourceGrid`
        The sources used.
    symmetry : {"none", "rotational"}, optional
        The sample's symmetry, which thins the survey directions as it thins the presets.

    Returns
    -------
    list of int
        Indices into ``sources``, ascending and without repeats: at most 13.
    """
    survey = SourceGrid.preset("survey", symmetry).directions
    nearest = np.argmax(survey @ np.asarray(sources.directions, dtype=float).T, axis=1)
    return sorted(set(int(i) for i in nearest))


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


def _unconverged(entries, exact, nominal):
    """The result warning and the octaves (Hz) of the unconverged BEM steps in ``generation["unconverged"]``; ``(None, [])`` if none."""
    if not entries:
        return None, []
    frequencies = sorted({float(u["frequency"]) for u in entries})
    bodies = [b for b in ("sample", "reference") if any(u["body"] == b for u in entries)]
    standard = list(Defaults._DEFAULT_FREQUENCY_BANDS)
    octaves = set()
    for f in frequencies:
        band = int(np.argmin(np.abs(np.log(exact) - np.log(f))))  # a simulation frequency is within 2^(1/9) of its band's exact centre
        octaves.add(_nearest_octave(nominal[band], standard))
    octaves = sorted(octaves)
    message = _UNCONVERGED.format(bodies=" and the ".join(bodies), frequencies=", ".join("{:.0f}".format(f) for f in frequencies), octaves=", ".join(str(o) for o in octaves))
    return message, octaves


def _pressures(solver, sample, sources, receivers, frequencies, c):
    """``(indices, Pressures)`` source batch by source batch for a batchable solver, else one batch of all sources.

    A batch holds at most ``_BATCH_BYTES`` of pressures, so a ``"fine"`` run never holds the pressures of all its sources.
    """
    if getattr(solver, "batchable", False):
        per_source = 2 * len(receivers.weights) * len(frequencies) * np.dtype(complex).itemsize
        return solver.solve_batches(sample, sources, receivers, frequencies, c, batch_size=max(1, int(_BATCH_BYTES // per_source)))
    return iter([(np.arange(sources.n), solver.solve(sample, sources, receivers, frequencies, c))])


def measure_scattering(sample, method="kirchhoff", preset=None, band_range=ISO_BAND_RANGE, air=None, name=None, **solver_options) -> ScatteringResult:
    """Measure the scattering coefficient of a sample in a virtual free field (ISO 17497-2; Heimes & Vorländer 2025).

    The correlation coefficient (Mommertz 2000; Heimes & Vorländer eq. 3) of the sample's and the reference's pressures on a receiver
    hemisphere of radius ``10·d``, at receiver spacing ``min(5°, 50·λ_min/d)`` (ISO 17497-2 §6.2.2, Heimes eq. 6), is computed per source
    direction and frequency, averaged over the three frequencies of each one-third-octave band (eq. 11), weighted over the sources with
    the Paris weights for random incidence, and averaged per octave (ISO 354 §8.3, ISO 11654).

    A batchable solver (``solver.batchable``: Kirchhoff) runs source batch by source batch, each batch reduced to its correlation
    coefficients before the next, so the pressures of all sources are never held at once (a batch holds at most 512 MB of them). The
    polar levels are kept only for the preview sources (:func:`preview_sources`: the sources nearest to the 13 directions of ISO 17497-2
    Table 1), listed in ``result.polar_source_indices``. A Kirchhoff run estimated longer than 10 minutes is warned about before it
    starts (:meth:`~compas_acoustics.lab.solvers.KirchhoffSolver.estimate_seconds`): the default ``"fine"`` preset on a 0.8 m disc takes
    about 50 minutes on 14 cores.

    Known limitations of the BEM (2026-10-24, Heimes & Vorländer's sine surface, d = 0.8 m): below about 700 Hz NumCalc's iterative solves
    do not converge and those bands are wrong; such steps are listed in ``generation["unconverged"]``, warned about in
    ``result.warnings`` and their octaves in ``result.unconverged_octaves``. Where it converges, the BEM's s sits 0.03–0.10 below Heimes &
    Vorländer's in 11 of the 12 checked one-third octaves (Λ/λ ≥ 0.5, per angle at 31.5° and 61.74°); see
    :class:`~compas_acoustics.lab.solvers.bem.BemSolver`.

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
        Passed to the solver: ``KirchhoffSolver(shadowing, chunk, workers)`` or ``BemSolver(cache_dir, numcalc_path, max_elements, ...)``.

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
    estimate = getattr(solver, "estimate_seconds", None)
    if estimate is not None:
        seconds = estimate(sample, sources, receivers, frequencies)
        if seconds > _LONG_RUN:
            faces = sample.sample.number_of_faces() + sample.reference.number_of_faces()
            _warnings.warn(
                _SLOW.format(
                    method=method,
                    time=_duration(seconds),
                    sources=sources.n,
                    frequencies=len(frequencies),
                    receivers=len(receivers.weights),
                    faces=faces,
                    workers=min(getattr(solver, "workers", 1), sources.n),
                ),
                stacklevel=2,
            )

    octaves, _ = fold_to_octaves(np.zeros(len(nominal)), nominal)
    preview = preview_sources(sources, sample.symmetry)
    s_f = np.empty((sources.n, len(frequencies)))
    polar_sample = np.empty((len(preview), len(octaves), len(receivers.weights)), dtype=np.float32)
    polar_reference = np.empty_like(polar_sample)
    solver_generation = {}
    batches = _pressures(solver, sample, sources, receivers, frequencies, c)
    try:
        for indices, pressures in batches:
            if not (np.isfinite(pressures.sample).all() and np.isfinite(pressures.reference).all()):
                raise RuntimeError("{}: the solver returned non-finite pressures for {!r}".format(method, sample.name))
            # source by source, so that the sums over the receivers do not depend on the batch size
            for row, i in enumerate(indices):
                one = slice(row, row + 1)
                s_f[i] = scattering_freefield(pressures.sample[one], pressures.reference[one], receivers.weights)[0]
                if i in preview:
                    slot = preview.index(i)
                    polar_sample[slot : slot + 1], polar_reference[slot : slot + 1] = _polar_levels(pressures.sample[one], pressures.reference[one], nominal)
            solver_generation = pressures.generation
            del pressures
    finally:
        close = getattr(batches, "close", None)
        if close is not None:
            close()  # stops a batchable solver's workers when a batch raised

    s_angle = band_average(s_f)
    s_bands = random_incidence(s_angle, sources)
    octaves, s_octaves = fold_to_octaves(s_bands, nominal)
    _, filled = fill_octaves(octaves, s_octaves)
    unconverged_message, unconverged_octaves = _unconverged(solver_generation.get("unconverged"), exact, nominal)

    messages = list(sample.warnings) + list(solver.warnings(sample, nominal, exact, c))
    if nominal.max() > _ISO_TOP:
        messages.append(_BEYOND_ISO)
    messages.extend(_FILLED.format(f=f, g=g) for f, g in filled)
    if unconverged_message:
        messages.append(unconverged_message)
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
    generation.update(solver_generation)
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
        polar_source_indices=preview,
        unconverged_octaves=unconverged_octaves,
        warnings=messages,
        generation=generation,
        name=name or sample.name,
    )
