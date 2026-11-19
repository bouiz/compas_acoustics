"""UDFA for a single rigid wedge (Kirsch and Ewert 2023) and the three-sided
barrier (Kirsch and Ewert 2024, Sec. 3.3)."""

import numpy as np
import pytest

from compas_acoustics.hybrid.udfa import (
    DoubleEdgePath,
    WedgePath,
    double_edge_terms,
    double_edge_zone,
    fractional_lowpass,
    single_term_cutoff,
    term_parameters,
    udfa_single_transfer,
    udfa_transfer,
)

C = 343.0
F = np.logspace(np.log10(20.0), np.log10(16000.0), 240)
F_BARRIER = np.logspace(np.log10(50.0), np.log10(8000.0), 40)
W = 1.5 * np.pi


def db(x):
    return 20.0 * np.log10(np.abs(x))


def test_fractional_lowpass_is_flat_far_below_and_minus_3_db_per_octave_far_above():
    # the (jf / Q fc) ** (1 / b ** r) term rolls in slowly: 0.38 dB already at fc / 100
    h = fractional_lowpass(np.array([0.0, 1e-3, 8000.0, 16000.0]), 100.0)
    assert h[0] == 1.0
    assert abs(db(h[1])) < 0.05
    assert -3.2 < db(h[3]) - db(h[2]) < -2.8


def test_knife_edge_cutoff_matches_eq_9():
    for theta_b in (0.2, 1.0, 2.0):
        path = WedgePath.infinite(2 * np.pi, 0.0, np.pi + theta_b, 2.0, 3.0, np.pi / 2)
        d_star = 2 * 2.0 * 3.0 / 5.0
        expected = C / (np.pi**2 * d_star * (1 - np.cos(theta_b)))
        fc, gain, polarity = term_parameters(path, C)
        np.testing.assert_allclose(fc, expected, rtol=1e-12)
        np.testing.assert_allclose(gain, 1.0, rtol=1e-12)
        np.testing.assert_array_equal(polarity, 1.0)
        assert single_term_cutoff(path, C) == pytest.approx(expected, rel=1e-12)


def test_single_and_two_term_agree_for_a_source_on_a_knife_plane():
    path = WedgePath.infinite(2 * np.pi, 0.0, np.pi + 0.7, 2.0, 2.0, np.pi / 2)
    np.testing.assert_allclose(udfa_transfer(path, F, c=C), udfa_single_transfer(path, F, c=C), rtol=1e-12)


@pytest.mark.parametrize("theta_w", [2 * np.pi, 1.5 * np.pi])
def test_incident_shadow_boundary_jump_is_the_direct_sound(theta_w):
    # 2024 eq. (6): just inside the shadow the diffraction carries what the
    # direct sound carries just outside it
    eps = 1e-7
    s = np.pi / 4
    shadow = WedgePath.infinite(theta_w, s, s + np.pi + eps, 2.0, 2.0, np.pi / 2)
    lit = WedgePath.infinite(theta_w, s, s + np.pi - eps, 2.0, 2.0, np.pi / 2)
    jump = udfa_transfer(shadow, F, c=C) - udfa_transfer(lit, F, c=C)
    np.testing.assert_allclose(jump, 1.0, atol=1e-3)


def test_reflection_boundary_jump_is_the_reflected_sound():
    eps = 1e-7
    s = np.pi / 4
    outside = WedgePath.infinite(1.5 * np.pi, s, np.pi - s + eps, 2.0, 2.0, np.pi / 2)
    inside = WedgePath.infinite(1.5 * np.pi, s, np.pi - s - eps, 2.0, 2.0, np.pi / 2)
    jump = udfa_transfer(outside, F, c=C) - udfa_transfer(inside, F, c=C)
    np.testing.assert_allclose(jump, 1.0, atol=1e-3)


def test_second_reflection_boundary_jump_is_the_reflected_sound():
    # reflection zone off the second face: theta_+ > 2 theta_w - pi (2023 Sec. II)
    eps = 1e-7
    theta_w = 1.5 * np.pi
    s = 0.8 * np.pi
    outside = WedgePath.infinite(theta_w, s, 2 * theta_w - np.pi - s - eps, 2.0, 2.0, np.pi / 2)
    inside = WedgePath.infinite(theta_w, s, 2 * theta_w - np.pi - s + eps, 2.0, 2.0, np.pi / 2)
    jump = udfa_transfer(outside, F, c=C) - udfa_transfer(inside, F, c=C)
    np.testing.assert_allclose(jump, 1.0, atol=1e-3)


@pytest.mark.parametrize(
    "theta_w, theta_s, theta_r",
    [
        (1.5 * np.pi, 0.0, 1.5 * np.pi),  # both on the faces
        (1.5 * np.pi, np.pi / 4, np.pi / 4 + np.pi),  # exactly on the ISB
        (1.5 * np.pi, np.pi / 4, 3 * np.pi / 4),  # exactly on the RSB
        (1.5 * np.pi, 0.8 * np.pi, 1.2 * np.pi),  # exactly on the second RSB
        (2 * np.pi, 0.0, np.pi),  # knife, ISB and RSB together
    ],
)
def test_no_nan_on_exact_boundaries(theta_w, theta_s, theta_r):
    path = WedgePath.infinite(theta_w, theta_s, theta_r, 2.0, 3.0, np.pi / 3)
    assert np.all(np.isfinite(udfa_transfer(path, F, c=C)))
    assert np.all(np.isfinite(udfa_single_transfer(path, F, c=C)))


def test_rejects_invalid_paths():
    with pytest.raises(ValueError):
        WedgePath(np.pi, 0.1, 0.2, 1.0, 1.0, 0.0, 0.0)
    with pytest.raises(ValueError):
        WedgePath(1.5 * np.pi, 0.5, 0.2, 1.0, 1.0, 0.0, 0.0)
    with pytest.raises(ValueError):
        WedgePath(1.5 * np.pi, 0.1, 0.2, 0.0, 1.0, 0.0, 0.0)


def centred(theta_w, theta_s, theta_r, d, length, z=0.0):
    """Source and receiver at distance d, phi = 90 deg, apex at z."""
    return WedgePath(theta_w, theta_s, theta_r, d, d, z, z, length)


def test_a_very_long_finite_edge_is_the_infinite_edge():
    args = (2 * np.pi, np.pi / 4, np.pi / 4 + np.pi + np.pi / 3)
    finite = udfa_transfer(centred(*args, 4.0, 1e4), F, c=C)
    infinite = udfa_transfer(WedgePath.infinite(*args, 4.0, 4.0, np.pi / 2), F, c=C)
    np.testing.assert_allclose(db(finite), db(infinite), atol=0.01)


def test_short_edge_low_frequency_gain_halves_with_the_length():
    # Sec. III-A: -6 dB per halving for L << d*
    args = (2 * np.pi, np.pi / 4, np.pi / 4 + np.pi + np.pi / 3)
    a = udfa_transfer(centred(*args, 4.0, 0.05), np.array([20.0]), c=C)
    b = udfa_transfer(centred(*args, 4.0, 0.025), np.array([20.0]), c=C)
    assert db(a)[0] - db(b)[0] == pytest.approx(6.02, abs=0.5)


def test_apex_crossing_the_edge_end_is_continuous():
    # Fig. 5 geometry: L = 1.41 m, ds = dr = 2 m, theta_s = pi/4, theta_b = pi/3
    args = (2 * np.pi, np.pi / 4, np.pi / 4 + np.pi + np.pi / 3)
    half = 1.41 / 2
    inside = udfa_transfer(centred(*args, 2.0, 1.41, half - 1e-9), F, c=C)
    outside = udfa_transfer(centred(*args, 2.0, 1.41, half + 1e-9), F, c=C)
    np.testing.assert_allclose(db(inside), db(outside), atol=0.01)


def test_apex_far_off_the_edge_falls_at_6_db_per_octave():
    # Sec. III-C: first-order low-pass well above 1 / (pi (t2 - t1))
    args = (2 * np.pi, np.pi / 4, np.pi / 4 + np.pi + np.pi / 3)
    h = udfa_transfer(centred(*args, 4.0, 1.41, 1.41 / 2 + 2.0), np.array([4000.0, 8000.0]), c=C)
    assert -7.0 < db(h[1]) - db(h[0]) < -5.0


def test_finite_edge_no_nan_at_corner_and_for_tiny_edges():
    args = (1.5 * np.pi, np.pi / 4, np.pi / 4 + np.pi + 0.3)
    for path in (centred(*args, 2.0, 1.0, 0.5), centred(*args, 2.0, 1e-4)):
        assert np.all(np.isfinite(udfa_transfer(path, F, c=C)))


def test_grazing_incidence_is_finite():
    path = WedgePath(1.5 * np.pi, np.pi / 4, np.pi / 4 + np.pi + 0.3, 1e-3, 1e-3, -5.0, 5.0, 20.0)
    assert np.all(np.isfinite(udfa_transfer(path, F, c=C)))


def test_finite_single_term_tends_to_the_infinite_one():
    args = (2 * np.pi, np.pi / 4, np.pi / 4 + np.pi + np.pi / 3)
    finite = udfa_single_transfer(centred(*args, 4.0, 1e4), F, c=C)
    infinite = udfa_single_transfer(WedgePath.infinite(*args, 4.0, 4.0, np.pi / 2), F, c=C)
    np.testing.assert_allclose(db(finite), db(infinite), atol=0.01)


def test_finite_single_term_is_the_two_half_wedges():
    # Kirsch and Ewert (2023), Sec. III-B: (g1 H(fc/g1^2) + g2 H(fc/g2^2)) / 2,
    # the finite single term with unit gain (Sec. VII)
    from compas_acoustics.hybrid.udfa import _finite_half_wedge, _half_wedge_gain

    path = WedgePath(2 * np.pi, np.pi / 4, np.pi / 4 + np.pi + 1.0, 2.0, 3.0, -0.3, 0.4, 1.5)
    fc = single_term_cutoff(path, C)
    t1, t2 = sorted(((path.length_via(-0.75) - path.d) / C, (path.length_via(0.75) - path.d) / C))
    expected = (_finite_half_wedge(F, fc, _half_wedge_gain(fc, t1)) + _finite_half_wedge(F, fc, _half_wedge_gain(fc, t2))) / 2
    np.testing.assert_allclose(udfa_single_transfer(path, F, c=C), expected, rtol=1e-12)


# Three-sided barrier


def pressure(path):
    out = np.zeros(F_BARRIER.size, dtype=complex)
    for t in double_edge_terms(path, c=C):
        out += np.exp(-2j * np.pi * F_BARRIER * t.length / C) / t.length * t.transfer(F_BARRIER)
    return out


def polar(origin, angle, r):
    return np.asarray(origin) + r * np.array([np.cos(angle), np.sin(angle)])


def test_zones():
    s = polar((0, 0), -np.pi / 4, 1.0)
    assert double_edge_zone(DoubleEdgePath(W, W, 1.0, s, polar((-1, 0), np.pi + 0.5, 1.0))) == "double_shadow"
    assert double_edge_zone(DoubleEdgePath(W, W, 1.0, s, polar((-1, 0), np.pi - 0.3, 1.0))) == "single_shadow"
    assert double_edge_zone(DoubleEdgePath(W, W, 1.0, s, polar((-1, 0), np.pi - 1.5, 3.0))) == "single_view"
    up = polar((0, 0), np.pi / 4, 1.0)
    assert double_edge_zone(DoubleEdgePath(W, W, 1.0, up, polar((-1, 0), 3 * np.pi / 4, 1.0))) == "double_view"


def test_rejects_points_on_the_wrong_side():
    with pytest.raises(ValueError):
        DoubleEdgePath(W, W, 1.0, np.array([-0.5, 1.0]), np.array([-2.0, -1.0]))


def test_reciprocity():
    s, r = polar((0, 0), -0.6, 1.3), polar((-1, 0), np.pi + 0.4, 0.8)
    a = pressure(DoubleEdgePath(W, W, 1.0, s, r, phi=1.2))
    mirror = lambda p: np.array([-1.0 - p[0], p[1]])
    b = pressure(DoubleEdgePath(W, W, 1.0, mirror(r), mirror(s), phi=1.2))
    np.testing.assert_allclose(a, b, rtol=1e-9)


def test_receiver_crossing_the_top_plane_is_continuous():
    s = polar((0, 0), -np.pi / 4, 1.0)
    above = pressure(DoubleEdgePath(W, W, 1.0, s, np.array([-2.0, 1e-7])))
    below = pressure(DoubleEdgePath(W, W, 1.0, s, np.array([-2.0, -1e-7])))
    np.testing.assert_allclose(db(above), db(below), atol=0.05)


def test_source_crossing_the_top_plane_is_continuous():
    r = polar((-1, 0), 3 * np.pi / 4, 1.5)
    above = pressure(DoubleEdgePath(W, W, 1.0, np.array([1.5, 1e-7]), r))
    below = pressure(DoubleEdgePath(W, W, 1.0, np.array([1.5, -1e-7]), r))
    np.testing.assert_allclose(db(above), db(below), atol=0.05)


def test_a_thin_barrier_is_a_knife_edge():
    s, r = polar((0, 0), -np.pi / 4, 1.0), polar((-1e-4, 0), np.pi + np.pi / 4, 1.0)
    double = pressure(DoubleEdgePath(W, W, 1e-4, s, r))
    knife = WedgePath.infinite(2 * np.pi, np.pi / 4, 2 * np.pi - np.pi / 4, 1.0, 1.0 + 1e-4, np.pi / 2)
    d = knife.d
    single = np.exp(-2j * np.pi * F_BARRIER * d / C) / d * udfa_transfer(knife, F_BARRIER, c=C)
    np.testing.assert_allclose(db(double), db(single), atol=0.5)


def test_double_shadow_falls_at_6_db_per_octave():
    s, r = polar((0, 0), -np.pi / 4, 1.0), polar((-1, 0), np.pi + np.pi / 4, 1.0)
    p = pressure(DoubleEdgePath(W, W, 1.0, s, r))
    hi = np.searchsorted(F_BARRIER, [4000.0, 8000.0])
    slope = (db(p)[hi[1] - 1] - db(p)[hi[0]]) / np.log2(F_BARRIER[hi[1] - 1] / F_BARRIER[hi[0]])
    assert -7.0 < slope < -5.0
