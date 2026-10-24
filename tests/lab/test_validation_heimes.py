"""Validation of the BEM solver against Heimes & Vorländer's simulated sine surface, and the Kirchhoff solver's measured gap to BEM.

Reference data: Heimes, A., Vorländer, M. (2025), results of the "optimal" simulation, Zenodo 15673498, ``results_simulation_optimal.zip``
(CC BY 4.0). Download and extract it outside the repository and set ``HEIMES_RESULTS`` to the extracted folder, the one that holds
``sine.scattering.sofa`` (s per incidence angle, from their ``1.4_calc_scattering_from_s_d.py``) and ``sine.scattering_rand.sofa``
(random incidence, ``1.5_random_scattering.py``). Without it every test here is skipped.

The sample (their ``0.1_simulate_optimal.py``): a sine relief with period Λ = 80 mm across x (grooves along y), n = 10 periods on a disc of
d = 0.8 m, depth h = 0.051/2.5 m, a 10 mm baseplate, c = 346.18 m/s. Their sources are plane waves at azimuth 0, the incidence plane across
the grooves, at colatitudes 1.26° + k·2.52°. Their s comes at 1/9-octave steps of Λ/λ; it is interpolated (linear in log f) onto our three
frequencies per one-third-octave band and averaged as ours are (Heimes eq. 11), so both sides are compared on the same bands of
``EXTENDED_BAND_RANGE``. Λ/λ = 0.5…2 is 2.2…8.7 kHz for this sample. The paper's one-third-octave agreement is |Δs| ≲ 0.1 for
Λ/λ ≥ 0.5; |Δs| is compared at the 0.01 resolution at which ISO 17497-1 §8.3 presents s, i.e. rounded to 0.01 before the check against 0.1.

``HEIMES_BEM_CACHE`` (optional) is the BEM cache folder; the per-angle tests and the gap test share it, so the BEM runs once per angle.

Commands (the BEM cost per frequency is in docs/superpowers/plans/2026-10-23-coefficient-lab-v1-a-spike-findings.md):

- per angle and the Kirchhoff gap (two single-source BEM runs, about 3 h on a 14-core M-series Mac)::

    HEIMES_RESULTS=<folder> HEIMES_BEM_CACHE=<cache> python -m pytest tests/lab/test_validation_heimes.py -v -s -k "per_angle or gap"

- random incidence (97 sources x 60 frequencies x 2 bodies, overnight and longer)::

    HEIMES_RESULTS=<folder> HEIMES_BEM_CACHE=<cache> python -m pytest tests/lab/test_validation_heimes.py -v -s -k random_incidence
"""

import functools
import json
import math
import os
import types

import numpy as np
import pytest
from lab_shapes import sine_tile

from compas_acoustics.lab import EXTENDED_BAND_RANGE
from compas_acoustics.lab import ReceiverGrid
from compas_acoustics.lab import ScatteringSample
from compas_acoustics.lab import SourceGrid
from compas_acoustics.lab import band_average
from compas_acoustics.lab import fold_to_octaves
from compas_acoustics.lab import measure_scattering
from compas_acoustics.lab import receiver_spacing
from compas_acoustics.lab import scattering_freefield
from compas_acoustics.lab import simulation_frequencies
from compas_acoustics.lab import third_octave_centres

pytestmark = [pytest.mark.slow, pytest.mark.skipif("HEIMES_RESULTS" not in os.environ, reason="needs the Zenodo 15673498 results")]

LAMBDA = 0.08  # m, the structural wavelength
C = 346.18  # m/s
DIAMETER = 0.8  # m
DEPTH = 0.051 / 2.5  # m
ANGLES = (31.5, 61.74)  # degrees, two of Heimes's source colatitudes (1.26 + 12·2.52 and 1.26 + 24·2.52)
TOLERANCE = 0.1
RESOLUTION = 2  # decimals: |Δs| is compared at 0.01, the resolution of s in ISO 17497-1 §8.3
RATIO_MIN = 0.5  # Λ/λ from which the tolerance applies
BAND_RANGE = EXTENDED_BAND_RANGE
_AIR = types.SimpleNamespace(speed_of_sound=C)  # measure_scattering reads only air.speed_of_sound


def _ratio(frequency):
    """Λ/λ of a frequency, rounded so that both sides key their bands alike."""
    return round(LAMBDA * float(frequency) / C, 3)


def _bem_options():
    cache = os.environ.get("HEIMES_BEM_CACHE")
    return {"cache_dir": cache} if cache else {}


def _read(name):
    """The FrequencyData and source Coordinates of one of Heimes's SOFA files, checked against the sample's Λ and c."""
    import pyfar as pf
    import sofar

    sofa = sofar.read_sofa(os.path.join(os.environ["HEIMES_RESULTS"], name), verify=False)
    assert math.isclose(float(np.squeeze(sofa.SpeedOfSound)), C) and math.isclose(float(np.squeeze(sofa.SampleStructuralWavelength)), LAMBDA)
    data, sources = pf.io.convert_sofa(sofa)[:2]
    return data, sources


def _onto_bands(frequencies, s):
    """Values at Heimes's frequencies -> {Λ/λ of the band centre: band value} on our bands.

    Interpolated linearly in log f at our three frequencies per band and averaged (Heimes eq. 11), for the bands whose three frequencies
    lie within the data.
    """
    frequencies = np.asarray(frequencies, dtype=float)
    s = np.asarray(s, dtype=float)
    assert np.all(np.diff(frequencies) > 0)
    _, exact = third_octave_centres(BAND_RANGE)
    f = simulation_frequencies(BAND_RANGE).reshape(-1, 3)
    inside = (f.min(axis=1) >= frequencies[0]) & (f.max(axis=1) <= frequencies[-1])
    values = np.interp(np.log(f), np.log(frequencies), s).mean(axis=1)
    return {_ratio(e): float(v) for e, v, ok in zip(exact, values, inside) if ok}


def _by_ratio(values):
    _, exact = third_octave_centres(BAND_RANGE)
    return {_ratio(e): float(v) for e, v in zip(exact, values)}


def heimes_scattering(theta):
    """Heimes's s for the source at colatitude ``theta`` (degrees, azimuth 0), from ``sine.scattering.sofa``: {Λ/λ: s} on our bands."""
    data, sources = _read("sine.scattering.sofa")
    colatitude = np.degrees(sources.colatitude)
    i = int(np.argmin(np.abs(colatitude - theta)))
    assert abs(colatitude[i] - theta) < 0.01 and abs(np.sin(sources.azimuth[i])) < 1e-6, "no source at {} degrees".format(theta)
    return _onto_bands(data.frequencies, np.real(data.freq[i, 0]))


def heimes_scattering_random():
    """Heimes's random-incidence s from ``sine.scattering_rand.sofa``: {Λ/λ: s_rand} on our bands."""
    data, _ = _read("sine.scattering_rand.sofa")
    return _onto_bands(data.frequencies, np.real(data.freq[0, 0]))


@functools.lru_cache(maxsize=None)
def heimes_sample():
    """The sine sample of Heimes's ``0.1_simulate_optimal.py``: Λ = 80 mm across x, h = 0.051/2.5 m, d = 0.8 m, 10 mm baseplate."""
    return ScatteringSample.disc(sine_tile(LAMBDA, DEPTH), diameter=DIAMETER, feature_size=LAMBDA)


def _solver(method):
    if method == "kirchhoff":
        from compas_acoustics.lab.solvers import KirchhoffSolver

        return KirchhoffSolver()
    from compas_acoustics.lab.solvers.bem import BemSolver

    return BemSolver(**_bem_options())


def _s_angle(method, sources):
    """s per source and band, (n_sources, n_bands), on the receivers ``measure_scattering`` uses, at c = 346.18 m/s."""
    sample = heimes_sample()
    frequencies = simulation_frequencies(BAND_RANGE)
    receivers = ReceiverGrid.hemisphere(receiver_spacing(sample.diameter, frequencies.max(), C), radius=10.0 * sample.diameter)
    pressures = _solver(method).solve(sample, sources, receivers, frequencies, C)
    return band_average(scattering_freefield(pressures.sample, pressures.reference, receivers.weights))


@functools.lru_cache(maxsize=None)
def _single(method, theta):
    """s per band of one source at colatitude ``theta``, azimuth 0 (the incidence plane across the grooves)."""
    return _s_angle(method, SourceGrid([theta], [0.0]))[0]


def bem_sine_s_angle(theta):
    """Our BEM s for the source at colatitude ``theta``, azimuth 0: {Λ/λ: s}."""
    return _by_ratio(_single("bem", theta))


def bem_sine_s_random():
    """Our BEM random-incidence s, ``measure_scattering(method="bem", preset="standard")`` at c = 346.18 m/s: {Λ/λ: s_rand}."""
    result = measure_scattering(heimes_sample(), method="bem", preset="standard", band_range=BAND_RANGE, air=_AIR, **_bem_options())
    return _by_ratio(result.s_bands)


def kirchhoff_vs_bem_gap(preset=None):
    """max over the sources of |s_kirchhoff − s_bem| per octave (the octave value: the mean of its one-third-octave bands).

    With ``preset=None`` the sources are the single sources at ``ANGLES`` (azimuth 0) that the per-angle tests run, so the BEM is reused;
    with a preset name both methods run on that source grid.

    Returns
    -------
    dict
        ``{octave centre (str): max |Δs|}``.
    """
    if preset is None:
        s = {m: np.array([_single(m, theta) for theta in ANGLES]) for m in ("kirchhoff", "bem")}
    else:
        sources = SourceGrid.preset(preset, heimes_sample().symmetry)
        s = {m: _s_angle(m, sources) for m in ("kirchhoff", "bem")}
    nominal, _ = third_octave_centres(BAND_RANGE)
    octaves, kirchhoff = fold_to_octaves(s["kirchhoff"], nominal)
    _, bem = fold_to_octaves(s["bem"], nominal)
    print(_band_table(nominal, s))
    return {str(o): round(float(v), 3) for o, v in zip(octaves, np.abs(kirchhoff - bem).max(axis=0))}


def _band_table(nominal, s):
    """Per band and source: s_kirchhoff | s_bem | |Δ|."""
    rows = ["band Hz | " + " | ".join("src {0} kirchhoff | src {0} bem | src {0} |Δ|".format(i) for i in range(s["bem"].shape[0]))]
    for b, f in enumerate(nominal):
        cells = []
        for i in range(s["bem"].shape[0]):
            k, m = s["kirchhoff"][i, b], s["bem"][i, b]
            cells.append("{:.3f} | {:.3f} | {:.3f}".format(k, m, abs(k - m)))
        rows.append("{} | {}".format(f, " | ".join(cells)))
    return "\n".join(rows)


def _compare(ref, ours, label):
    """The comparison table and the bands with Λ/λ ≥ 0.5 that miss the tolerance."""
    nominal, exact = third_octave_centres(BAND_RANGE)
    names = {_ratio(e): int(n) for n, e in zip(nominal, exact)}
    rows = ["{}: band Hz | Λ/λ | ours | Heimes | |Δ| | |Δ| rounded to 0.01 vs 0.1".format(label)]
    failing = []
    for ratio, s_ref in sorted(ref.items()):
        delta = abs(ours[ratio] - s_ref)
        missed = round(delta, RESOLUTION) > TOLERANCE  # "|Δs| ≲ 0.1" at the 0.01 resolution of ISO 17497-1 §8.3
        checked = ratio >= RATIO_MIN
        status = ("FAIL" if missed else "ok") if checked else "-"
        if checked and missed:
            failing.append(ratio)
        rows.append("{} | {:.3f} | {:.3f} | {:.3f} | {:.4f} | {}".format(names[ratio], ratio, ours[ratio], s_ref, delta, status))
    return "\n".join(rows), failing


@pytest.mark.parametrize("theta", ANGLES)
def test_bem_matches_heimes_per_angle(theta):
    ref = heimes_scattering(theta)
    ours = bem_sine_s_angle(theta)
    table, failing = _compare(ref, ours, "theta = {} deg".format(theta))
    print(table)
    assert any(r >= RATIO_MIN for r in ref)
    assert not failing, "|Δs| > {} at Λ/λ = {}\n{}".format(TOLERANCE, failing, table)  # the paper's one-third-octave agreement


def test_bem_matches_heimes_random_incidence():
    ref = heimes_scattering_random()
    ours = bem_sine_s_random()
    table, failing = _compare(ref, ours, "random incidence")
    print(table)
    assert any(r >= RATIO_MIN for r in ref)
    assert not failing, "|Δs| > {} at Λ/λ = {}\n{}".format(TOLERANCE, failing, table)


def test_kirchhoff_gap_to_bem_is_recorded():
    gap = kirchhoff_vs_bem_gap()  # max |s_kirchhoff - s_bem| per octave on the sine sample, at the per-angle tests' sources
    print(json.dumps(gap))  # pasted into KirchhoffSolver's docstring
    assert all(np.isfinite(v) for v in gap.values())
