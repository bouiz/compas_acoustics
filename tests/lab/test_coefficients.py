import numpy as np
import pytest

from compas_acoustics.lab import SourceGrid
from compas_acoustics.lab import random_incidence
from compas_acoustics.lab import scattering_freefield

rng = np.random.default_rng(0)


def test_identical_patterns_do_not_scatter():
    p = rng.normal(size=(2, 500, 3)) + 1j * rng.normal(size=(2, 500, 3))
    assert np.allclose(scattering_freefield(p, p, np.ones(500)), 0.0, atol=1e-12)


def test_a_common_phase_and_gain_do_not_scatter():
    p = rng.normal(size=(1, 500, 1)) + 1j * rng.normal(size=(1, 500, 1))
    s = scattering_freefield(3.0 * np.exp(1j * 0.7) * p, p, np.ones(500))
    assert np.allclose(s, 0.0, atol=1e-12)


def test_uncorrelated_patterns_scatter_almost_everything():
    p0 = rng.normal(size=(1, 20000, 1)) + 1j * rng.normal(size=(1, 20000, 1))
    p1 = rng.normal(size=(1, 20000, 1)) + 1j * rng.normal(size=(1, 20000, 1))
    assert scattering_freefield(p1, p0, np.ones(20000))[0, 0] > 0.99


def test_all_zero_patterns_give_zero_not_nan():
    z = np.zeros((2, 10, 3), dtype=complex)
    s = scattering_freefield(z, z, np.ones(10))
    assert s.shape == (2, 3)
    assert np.all(np.isfinite(s))
    assert np.all(s == 0.0)


def test_weights_broadcast_along_axis():
    p = rng.normal(size=(7, 5)) + 0j
    q = rng.normal(size=(7, 5)) + 0j
    a = scattering_freefield(p, q, np.ones(7), axis=0)
    assert a.shape == (5,)


def test_random_incidence_of_constant_is_constant():
    g = SourceGrid.preset("standard")
    assert np.allclose(random_incidence(np.full((g.n, 4), 0.37), g), 0.37)


def test_random_incidence_of_sin2_is_one_half():
    # (1/pi) * integral of sin^2(theta) cos(theta) sin(theta) over the hemisphere = 1/2
    g = SourceGrid.preset("fine")
    s = np.sin(np.radians(g.colatitude))[:, None] ** 2
    assert random_incidence(s, g)[0] == pytest.approx(0.5, abs=1e-2)
