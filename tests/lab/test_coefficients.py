import numpy as np
import pytest

from compas_acoustics.lab import SourceGrid
from compas_acoustics.lab import diffusion_coefficient
from compas_acoustics.lab import normalised_diffusion
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


def test_equal_intensity_everywhere_is_fully_diffuse():
    d = diffusion_coefficient(np.ones((1, 50)), rng.uniform(0.1, 2.0, size=50))
    assert d.shape == (1,)
    assert d[0] == pytest.approx(1.0, abs=1e-12)


def test_one_receiver_carrying_everything_is_not_diffuse():
    # With equal weights, a single loaded receiver gives exactly 0. With unequal weights it gives
    # (w_k n / sum(w) - 1) / (n - 1), because the receiver then stands for more or less solid angle.
    i = np.zeros((1, 50))
    i[0, 7] = 3.0
    assert diffusion_coefficient(i, np.ones(50))[0] == pytest.approx(0.0, abs=1e-12)
    w = rng.uniform(0.5, 2.0, size=50)
    expected = (w[7] * 50 / w.sum() - 1.0) / 49.0
    assert diffusion_coefficient(i, w)[0] == pytest.approx(expected, abs=1e-12)


def test_diffusion_matches_a_hand_computed_value():
    # equal weights, I = [1, 3]: 16 / 10 = 1.6, d = 0.6
    assert diffusion_coefficient(np.array([[1.0, 3.0]]), np.ones(2))[0] == pytest.approx(0.6, abs=1e-12)
    # weights [1, 3] -> [0.5, 1.5]: sum wI = 5, sum wI^2 = 14, d = 25/14 - 1 = 11/14
    assert diffusion_coefficient(np.array([[1.0, 3.0]]), np.array([1.0, 3.0]))[0] == pytest.approx(11.0 / 14.0, abs=1e-12)


def test_weights_act_as_solid_angles():
    # The weights are normalised to mean 1, so only their ratios matter, and splitting a receiver into two
    # identical halves of half the weight leaves (sum wI)^2 / (n sum wI^2) unchanged. d itself moves only
    # through the (n - 1) of eq. 6, so that quantity, R = ((n - 1) d + 1) / n, is what is invariant.
    n = 20
    i = rng.uniform(0.1, 2.0, size=(1, n))
    w = rng.uniform(0.5, 2.0, size=n)
    d = diffusion_coefficient(i, w)[0]
    assert diffusion_coefficient(i, 7.3 * w)[0] == pytest.approx(d, abs=1e-12)

    i_split = np.concatenate([i, i[:, :1]], axis=1)
    w_split = np.concatenate([w, w[:1] / 2.0])
    w_split[0] = w[0] / 2.0  # receiver 0 and its duplicate now carry half the weight each
    d_split = diffusion_coefficient(i_split, w_split)[0]
    r = ((n - 1) * d + 1.0) / n
    r_split = (n * d_split + 1.0) / (n + 1)
    assert r_split == pytest.approx(r, abs=1e-12)


def test_an_empty_pattern_gives_zero_not_nan():
    d = diffusion_coefficient(np.zeros((2, 10, 3)), np.ones(10))
    assert d.shape == (2, 3)
    assert np.all(d == 0.0)


def test_normalisation():
    assert normalised_diffusion(0.6, 0.2) == pytest.approx(0.5, abs=1e-12)
    out = normalised_diffusion(np.array([0.3, 1.0]), np.array([1.0, 1.0]))
    assert np.all(np.isfinite(out))
    assert np.all(out == 0.0)


def test_axis_one_on_a_3d_array_matches_per_slice_calls():
    local = np.random.default_rng(12345)
    s, n, f = 4, 30, 3
    intensity = local.uniform(0.05, 3.0, size=(s, n, f))
    weights = local.uniform(0.2, 2.0, size=n)
    d = diffusion_coefficient(intensity, weights, axis=1)
    assert d.shape == (s, f)
    expected = np.array([[diffusion_coefficient(intensity[i, :, j], weights, axis=0) for j in range(f)] for i in range(s)])
    np.testing.assert_allclose(d, expected, rtol=1e-12, atol=1e-12)


def test_negative_and_other_axes_agree_with_axis_one():
    local = np.random.default_rng(54321)
    s, n, f = 4, 30, 3
    intensity = local.uniform(0.05, 3.0, size=(s, n, f))
    weights = local.uniform(0.2, 2.0, size=n)
    d = diffusion_coefficient(intensity, weights, axis=1)
    np.testing.assert_allclose(diffusion_coefficient(intensity, weights, axis=-2), d, rtol=1e-12, atol=1e-12)
    transposed = np.moveaxis(intensity, 1, 0)  # (N, S, F)
    np.testing.assert_allclose(diffusion_coefficient(transposed, weights, axis=0), d, rtol=1e-12, atol=1e-12)
