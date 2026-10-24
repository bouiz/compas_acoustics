"""Virtual free-field scattering measurement of a sample (ISO 17497-2; Heimes & Vorländer 2025).

The pipeline of :func:`measure_scattering` for the wave methods (``"kirchhoff"``, ``"bem"``): plane waves from a source grid onto the sample
and its reference, the pressures on a receiver hemisphere, the correlation coefficient per direction and frequency (Heimes & Vorländer
eq. 3), its mean per one-third-octave band (eq. 11), the Paris random-incidence average, and octave values (ISO 354 §8.3, ISO 11654) for the
material. The ``"geometric"`` method replaces the pressures by specular rays binned on the hemisphere (:mod:`.geometric`) and imposes a
heuristic frequency dependence; the averaging that follows is the same.
"""

import inspect
import warnings as _warnings
from dataclasses import dataclass

import numpy as np

from compas_acoustics.defaults import Defaults

from .bands import ISO_BAND_RANGE
from .bands import _nearest_octave
from .bands import band_average
from .bands import fill_octaves
from .bands import fold_to_octaves
from .bands import simulation_frequencies
from .bands import third_octave_centres
from .coefficients import diffusion_coefficient
from .coefficients import normalised_diffusion
from .coefficients import random_incidence
from .coefficients import scattering_freefield
from .geometric import geometric_response
from .geometric import heuristic_s
from .geometric import scattering_share
from .grids import ReceiverGrid
from .grids import SourceGrid
from .grids import receiver_spacing
from .results import ScatteringResult

__all__ = ["DEFAULT_PRESETS", "measure_scattering", "preview_sources"]

METHODS = ("geometric", "kirchhoff", "bem")
DEFAULT_PRESETS = {"geometric": "standard", "kirchhoff": "standard", "bem": "standard"}
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
_HEURISTIC = (
    "geometric: s is frequency-modelled from the geometry (heuristic: s_max·(1 − exp(−(k·h)²)) from specular rays, with no phase and no "
    "diffraction); use kirchhoff or bem for the physics (an sbr method is planned)"
)
_LOST = (
    "geometric: up to {percent:.1f} % of the rays of a source are lost, still bouncing after max_bounces = {bounces}; lost energy of only one of "
    "the sample and the reference counts as scattering; a larger max_bounces loses fewer"
)
_NO_TRIMESH = 'the "geometric" method needs trimesh (Python >= 3.11, pip install compas_acoustics[lab]); pass method="kirchhoff" to run without it'
_GEOMETRIC_OPTIONS = inspect.signature(geometric_response)  # the options of the geometric method, and their defaults
_LOST_LIMIT = 0.01  # the lost fraction of a source above which the geometric method warns


def _check_method(method):
    if method not in METHODS:
        raise ValueError("unknown method {!r}; supported methods are {}".format(method, ", ".join("'{}'".format(m) for m in METHODS)))


def _solver(method, **solver_options):
    """The solver of a wave method: the one place that maps names to solvers. BEM is imported only when asked for (it needs Python ≥ 3.11)."""
    _check_method(method)
    if method == "kirchhoff":
        from .solvers import KirchhoffSolver

        return KirchhoffSolver(**solver_options)
    if method == "bem":
        from .solvers.bem import BemSolver

        return BemSolver(**solver_options)
    raise ValueError("{!r} is not a wave method and has no solver".format(method))


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
    return _levels_db(energies[0], energies[1])


def _levels_db(sample, reference):
    """``(sample_db, reference_db)``, float32: both energies (receivers on the last axis) in dB relative to the reference's maximum, −300 dB where 0."""
    peak = reference.max(axis=-1, keepdims=True)
    peak = np.where(peak > 0, peak, 1.0)
    out = []
    for energy in (sample, reference):
        ratio = energy / peak
        db = 10.0 * np.log10(np.where(ratio > 0, ratio, 1.0))
        out.append(np.where(ratio > 0, np.maximum(db, _FLOOR_DB), _FLOOR_DB).astype(np.float32))
    return out[0], out[1]


def _diffusion(intensity_sample, intensity_reference, weights):
    """The normalised diffusion ``d_n`` of the sample against the reference, from intensities with the receivers on axis 1."""
    return normalised_diffusion(diffusion_coefficient(intensity_sample, weights), diffusion_coefficient(intensity_reference, weights))


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


@dataclass
class _Measured:
    """What a method measured, per source and band, before the averaging that all methods share (:func:`_result`)."""

    s_angle: np.ndarray  # (n_sources, n_bands)
    d_angle: np.ndarray  # (n_sources, n_bands)
    receiver_directions: np.ndarray  # (n_receivers, 3)
    polar_sample: np.ndarray  # (n_preview, n_octaves, n_receivers), float32
    polar_reference: np.ndarray
    messages: list  # the method's own warnings, after the sample's
    generation: dict  # the method's record, added to the shared one
    unconverged_octaves: list


def _correlation(sample, method, sources, preview, band_range, nominal, exact, c, solver_options):
    """A wave method: the solver's pressures on the receiver hemisphere, reduced source by source to s (eq. 3) and d per frequency."""
    frequencies = simulation_frequencies(band_range)
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
                stacklevel=3,
            )

    octaves, _ = fold_to_octaves(np.zeros(len(nominal)), nominal)
    s_f = np.empty((sources.n, len(frequencies)))
    d_f = np.empty_like(s_f)
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
                p_sample, p_reference = pressures.sample[one], pressures.reference[one]  # (1, n_receivers, n_frequencies)
                s_f[i] = scattering_freefield(p_sample, p_reference, receivers.weights)[0]
                d_f[i] = _diffusion(np.abs(p_sample) ** 2, np.abs(p_reference) ** 2, receivers.weights)[0]
                if i in preview:
                    slot = preview.index(i)
                    polar_sample[slot : slot + 1], polar_reference[slot : slot + 1] = _polar_levels(p_sample, p_reference, nominal)
            solver_generation = pressures.generation
            del pressures
    finally:
        close = getattr(batches, "close", None)
        if close is not None:
            close()  # stops a batchable solver's workers when a batch raised

    messages = list(solver.warnings(sample, nominal, exact, c))
    unconverged_message, unconverged_octaves = _unconverged(solver_generation.get("unconverged"), exact, nominal)
    if unconverged_message:
        messages.append(unconverged_message)
    return _Measured(
        s_angle=band_average(s_f),
        d_angle=band_average(d_f),
        receiver_directions=receivers.directions,
        polar_sample=polar_sample,
        polar_reference=polar_reference,
        messages=messages,
        generation=solver_generation,
        unconverged_octaves=unconverged_octaves,
    )


def _geometric(sample, sources, preview, nominal, exact, c, solver_options):
    """The geometric method: specular rays binned on the hemisphere, s_max from the histograms' overlap, s(f) and d from the geometry."""
    try:
        options = _GEOMETRIC_OPTIONS.bind(sample, sources, **solver_options)
    except TypeError as error:
        raise TypeError("geometric: {} (the options are n_rays, bin_deg, max_bounces and seed)".format(error)) from None
    options.apply_defaults()
    try:
        response = geometric_response(*options.args, **options.kwargs)
    except ImportError as error:  # from the ray caster: trimesh or its backend is missing
        raise ImportError("{}: {}".format(_NO_TRIMESH, error)) from error
    n_bands = len(nominal)

    # the lost fraction is one more bin: lost by both bodies it cancels, lost by one it counts as scattering
    s_max = scattering_share(np.column_stack([response.sample, response.lost_sample]), np.column_stack([response.reference, response.lost_reference]))
    s_angle = heuristic_s(s_max, response.depth, exact, c)
    intensity_sample = response.sample / response.weights  # energy per steradian
    intensity_reference = response.reference / response.weights
    d = _diffusion(intensity_sample, intensity_reference, response.weights)
    d_angle = np.repeat(d[:, None], n_bands, axis=1)

    octaves, _ = fold_to_octaves(np.zeros(n_bands), nominal)
    polar_sample, polar_reference = _levels_db(intensity_sample[preview], intensity_reference[preview])
    polar_sample = np.repeat(polar_sample[:, None, :], len(octaves), axis=1)
    polar_reference = np.repeat(polar_reference[:, None, :], len(octaves), axis=1)

    lost = float(max(response.lost_sample.max(), response.lost_reference.max()))
    max_bounces = options.arguments["max_bounces"]
    messages = [_HEURISTIC]
    if lost > _LOST_LIMIT:
        messages.append(_LOST.format(percent=100.0 * lost, bounces=max_bounces))
    generation = {
        "heuristic": True,
        "reference": sample.reference_kind,
        "n_rays": options.arguments["n_rays"],
        "bin_deg": options.arguments["bin_deg"],
        "max_bounces": max_bounces,
        "seed": options.arguments["seed"],
        "lost": lost,
        "depth": float(response.depth),
    }
    return _Measured(
        s_angle=s_angle,
        d_angle=d_angle,
        receiver_directions=response.directions,
        polar_sample=polar_sample,
        polar_reference=polar_reference,
        messages=messages,
        generation=generation,
        unconverged_octaves=[],
    )


def _result(sample, measured, method, preset, band_range, c, sources, preview, nominal, name):
    """The averaging all methods share (random incidence, octaves, filled octaves), the warnings and the result."""
    s_bands = random_incidence(measured.s_angle, sources)
    d_bands = random_incidence(measured.d_angle, sources)
    octaves, s_octaves = fold_to_octaves(s_bands, nominal)
    _, d_octaves = fold_to_octaves(d_bands, nominal)
    _, filled = fill_octaves(octaves, s_octaves)

    messages = list(sample.warnings) + list(measured.messages)
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
        "receivers": len(measured.receiver_directions),
        "sources": sources.n,
    }
    generation.update(measured.generation)
    return ScatteringResult(
        method=method,
        bands=nominal.tolist(),
        s_bands=s_bands.tolist(),
        octave_bands=octaves,
        s_octaves=s_octaves.tolist(),
        filled_octaves=[f for f, _ in filled],
        s_angle=measured.s_angle,
        source_colatitude=sources.colatitude,
        source_azimuth=sources.azimuth,
        source_weights=sources.paris_weights,
        receiver_directions=measured.receiver_directions,
        polar_sample_db=measured.polar_sample,
        polar_reference_db=measured.polar_reference,
        polar_source_indices=preview,
        unconverged_octaves=measured.unconverged_octaves,
        warnings=messages,
        generation=generation,
        name=name or sample.name,
        d_angle=measured.d_angle,
        d_bands=d_bands.tolist(),
        d_octaves=d_octaves.tolist(),
    )


def measure_scattering(sample, method="geometric", preset=None, band_range=ISO_BAND_RANGE, air=None, name=None, **solver_options) -> ScatteringResult:
    """Measure the scattering and diffusion coefficients of a sample in a virtual free field (ISO 17497-2; Heimes & Vorländer 2025).

    Three methods, all on the same source grids, bands and averaging:

    - ``"geometric"`` (the default, seconds): a fast **heuristic** preview, not a wave solution (:mod:`compas_acoustics.lab.geometric`).
      It needs ``trimesh`` (ideally with the ``embreex`` backend), which the ``lab`` extra installs on Python ≥ 3.11 only
      (``pip install compas_acoustics[lab]``); ``method="kirchhoff"`` runs without them (Rhino's Python 3.9 included).
      ``n_rays`` specular rays per source, aimed at the footprint, reflect off the rigid sample and the rigid reference (mounted in a baffle
      at z = 0) and are binned on a receiver hemisphere of ``bin_deg`` cells. ``s_max`` per source is one minus the overlap of the two
      energy histograms, the lost fraction (rays still bouncing after ``max_bounces``) counted as one more bin;
      the per-band ``s = s_max·(1 − exp(−(k h)²))`` is imposed at each band's exact centre, with ``h`` twice the standard deviation of the
      relief over the reference. There is no phase and no diffraction: no rim, no cavity resonance. d comes from the cells' energy per
      steradian and is the same at every band, as are the polar levels. The result is marked ``generation["heuristic"]`` and warned about;
      more than 1 % lost rays for any source is warned about too.
    - ``"kirchhoff"``: physical optics, the correlation coefficient below.
    - ``"bem"``: the boundary element method (mesh2scattering/NumCalc, needs Python ≥ 3.11 and the ``lab`` extra), the same correlation.

    For the wave methods the correlation coefficient (Mommertz 2000; Heimes & Vorländer eq. 3) of the sample's and the reference's
    pressures on a receiver hemisphere of radius ``10·d``, at receiver spacing ``min(5°, 50·λ_min/d)`` (ISO 17497-2 §6.2.2, Heimes eq. 6),
    is computed per source direction and frequency, averaged over the three frequencies of each one-third-octave band (eq. 11), weighted
    over the sources with the Paris weights for random incidence, and averaged per octave (ISO 354 §8.3, ISO 11654).

    Every method also returns the normalised diffusion coefficient ``d_n = (d − d_ref)/(1 − d_ref)``, d from ISO 17497-2 eq. 6 on the
    intensities at the receivers weighted by their solid angles (|p|² for the wave methods, per frequency, then averaged per band like s),
    with the same random incidence and octaves as s (``result.d_angle``, ``d_bands``, ``d_octaves``).

    A batchable solver (``solver.batchable``: Kirchhoff) runs source batch by source batch, each batch reduced to its correlation
    coefficients before the next, so the pressures of all sources are never held at once (a batch holds at most 512 MB of them). The
    polar levels are kept only for the preview sources (:func:`preview_sources`: the sources nearest to the 13 directions of ISO 17497-2
    Table 1), listed in ``result.polar_source_indices``. A Kirchhoff run estimated longer than 10 minutes is warned about before it
    starts (:meth:`~compas_acoustics.lab.solvers.KirchhoffSolver.estimate_seconds`): the ``"fine"`` preset on a 0.8 m disc takes
    about 50 minutes on 14 cores, the default ``"standard"`` about 5.

    Known limitations of the BEM (2026-10-24, Heimes & Vorländer's sine surface, d = 0.8 m): below about 700 Hz NumCalc's iterative solves
    do not converge and those bands are wrong; such steps are listed in ``generation["unconverged"]``, warned about in
    ``result.warnings`` and their octaves in ``result.unconverged_octaves``. Where it converges, the BEM's s sits 0.03–0.10 below Heimes &
    Vorländer's in 11 of the 12 checked one-third octaves (Λ/λ ≥ 0.5, per angle at 31.5° and 61.74°); see
    :class:`~compas_acoustics.lab.solvers.bem.BemSolver`.

    Parameters
    ----------
    sample : :class:`compas_acoustics.lab.sample.ScatteringSample`
    method : {"geometric", "kirchhoff", "bem"}, optional
        The heuristic geometric preview (default, fast), physical optics, or the boundary element method.
    preset : {"survey", "standard", "fine"}, optional
        The source grid; by default ``DEFAULT_PRESETS[method]``: ``"standard"`` for every method.
        Comparisons between methods should pass the same preset.
    band_range : tuple of float, optional
        The nominal one-third-octave centres simulated, inclusive: ``ISO_BAND_RANGE`` (100-5000 Hz, ISO 17497-1/-2 §5) by default,
        ``EXTENDED_BAND_RANGE`` for the extension to 8 kHz beyond the standards.
    air : :class:`compas_acoustics.room.AirConditions`, optional
        Gives the speed of sound; the default air by default.
    name : str, optional
        The result's name; the sample's by default.
    **solver_options
        For ``"geometric"``, passed to :func:`~compas_acoustics.lab.geometric.geometric_response`: ``n_rays`` (rays per source, 50000),
        ``bin_deg`` (receiver cell spacing in degrees, 10), ``max_bounces`` (20) and ``seed`` (of the footprint points, 0); they are
        recorded in ``generation``. For the wave methods, passed to the solver: ``KirchhoffSolver(shadowing, chunk, workers)`` or
        ``BemSolver(cache_dir, numcalc_path, max_elements, ...)``.

    Returns
    -------
    :class:`compas_acoustics.lab.results.ScatteringResult`

    Raises
    ------
    ValueError
        If the method is not supported, or the preset or band range is invalid.
    TypeError
        If a solver option is not one of the method's.
    ImportError
        If ``method="geometric"`` and ``trimesh`` or its ray backend is missing; the message names ``method="kirchhoff"`` as the way out.
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
    sources = SourceGrid.preset(preset, sample.symmetry)
    preview = preview_sources(sources, sample.symmetry)
    if method == "geometric":
        measured = _geometric(sample, sources, preview, nominal, exact, c, solver_options)
    else:
        measured = _correlation(sample, method, sources, preview, band_range, nominal, exact, c, solver_options)
    return _result(sample, measured, method, preset, band_range, c, sources, preview, nominal, name)
