"""The geometric method against the Kirchhoff method on the sine sample of the Heimes validation: a record of the gap, not a tolerance.

Needs no external data, only ``trimesh`` with ``embreex``. The accuracy record in the docstring of ``compas_acoustics.lab.geometric`` is
the output of this test (``pytest tests/lab/test_geometric_validation.py -v -s``). The sample is the one of ``test_validation_heimes.py``
(its constants are copied here, because that module is skipped as a whole without the Zenodo data).
"""

import math
import time
import types

import numpy as np
import pytest
from lab_shapes import sine_tile

from compas_acoustics.lab import ScatteringSample
from compas_acoustics.lab import measure_scattering

pytest.importorskip("embreex")

pytestmark = pytest.mark.slow

# The sample of Heimes's 0.1_simulate_optimal.py, as in test_validation_heimes.py.
LAMBDA = 0.08  # m, the structural wavelength
DEPTH = 0.051 / 2.5  # m
DIAMETER = 0.8  # m
C = 346.18  # m/s
_AIR = types.SimpleNamespace(speed_of_sound=C)  # measure_scattering reads only air.speed_of_sound


def _timed(sample, **options):
    start = time.perf_counter()
    result = measure_scattering(sample, preset="survey", air=_AIR, **options)
    return result, time.perf_counter() - start


def test_geometric_against_kirchhoff_is_recorded():
    sample = ScatteringSample.disc(sine_tile(LAMBDA, DEPTH), diameter=DIAMETER, feature_size=LAMBDA)

    geometric, t_geometric = _timed(sample, method="geometric")
    kirchhoff, t_kirchhoff = _timed(sample, method="kirchhoff")
    few, _ = _timed(sample, method="geometric", n_rays=5000)
    many, _ = _timed(sample, method="geometric", n_rays=50000)

    assert geometric.octave_bands == kirchhoff.octave_bands
    s_g = np.asarray(geometric.s_octaves, dtype=float)
    s_k = np.asarray(kirchhoff.s_octaves, dtype=float)
    print("\nsine sample, Lambda = {} m, h = {:.4f} m, d = {} m, c = {} m/s, preset 'survey', ISO band range".format(LAMBDA, DEPTH, DIAMETER, C))
    print("{:>8} {:>12} {:>12} {:>8} {:>10} {:>10}".format("octave", "s_geometric", "s_kirchhoff", "|delta|", "g(5000)", "g(50000)"))
    for i, f in enumerate(geometric.octave_bands):
        print("{:>8} {:12.3f} {:12.3f} {:8.3f} {:10.3f} {:10.3f}".format(f, s_g[i], s_k[i], abs(s_g[i] - s_k[i]), few.s_octaves[i], many.s_octaves[i]))
    print("max |delta| = {:.3f}, mean |delta| = {:.3f}".format(float(np.max(np.abs(s_g - s_k))), float(np.mean(np.abs(s_g - s_k)))))
    print("run time: geometric {:.1f} s, kirchhoff {:.1f} s".format(t_geometric, t_kirchhoff))

    for s in (s_g, s_k):
        assert np.all(np.isfinite(s)) and np.all(s >= 0.0) and np.all(s <= 1.0)
    assert all(math.isfinite(v) for v in few.s_octaves)
    assert t_geometric < 0.1 * t_kirchhoff
