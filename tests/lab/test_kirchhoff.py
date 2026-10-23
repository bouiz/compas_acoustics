import numpy as np
import pytest
from lab_shapes import flat_tile
from lab_shapes import sine_tile

from compas_acoustics.lab.bands import third_octave_centres
from compas_acoustics.lab.coefficients import scattering_freefield
from compas_acoustics.lab.grids import ReceiverGrid
from compas_acoustics.lab.grids import SourceGrid
from compas_acoustics.lab.sample import ScatteringSample
from compas_acoustics.lab.solvers import KirchhoffSolver
from compas_acoustics.lab.solvers import Pressures
from compas_acoustics.lab.solvers import ScatteringSolver
from compas_acoustics.lab.solvers.kirchhoff import triangle_phase_integral

TRI = np.array([[[0.0, 0.0, 0.0], [0.3, 0.0, 0.05], [0.1, 0.2, -0.02]]])


def brute(tri, q, n=400):
    """The reference (not the code under test): the midpoint rule on n^2 and (2n)^2 sub-triangles, Richardson-extrapolated.

    The midpoint rule alone has an error ~ (|q| h)^2 with h = edge / n: 4e-4 relative at n = 400 for the phase scale 50,
    far above the 1e-6 pinned here. Its error is a clean 1/n^2 series, so (4 M(2n) - M(n)) / 3 removes the leading term (3e-8 at scale 50).
    """
    return (4 * _midpoint(tri, q, 2 * n) - _midpoint(tri, q, n)) / 3


def _midpoint(tri, q, n):
    """Midpoint rule on the n^2 congruent sub-triangles of a uniform barycentric subdivision: each centroid carries 1/n^2 of the area."""
    a, b, c = (np.asarray(v, dtype=float) for v in tri)
    area = np.linalg.norm(np.cross(b - a, c - a)) / 2
    i, j = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    up = i + j <= n - 1
    # upward cells: corners (i, j), (i+1, j), (i, j+1) in steps of 1/n along (b - a, c - a); centroid at (i + 1/3, j + 1/3)
    s_up, t_up = (i[up] + 1 / 3) / n, (j[up] + 1 / 3) / n
    down = i + j <= n - 2
    # downward cells: corners (i+1, j), (i, j+1), (i+1, j+1); centroid at (i + 2/3, j + 2/3)
    s_dn, t_dn = (i[down] + 2 / 3) / n, (j[down] + 2 / 3) / n
    s = np.concatenate([s_up, s_dn])
    t = np.concatenate([t_up, t_dn])
    points = a + s[:, None] * (b - a) + t[:, None] * (c - a)
    return area / n**2 * np.exp(1j * points @ np.asarray(q, dtype=float)).sum()


@pytest.mark.parametrize("scale", [0.0, 1e-7, 1e-4, 1e-2, 1.0, 50.0])
def test_triangle_integral_matches_brute_force(scale):
    q = scale * np.array([[3.0, -1.0, 2.0]])
    assert np.allclose(triangle_phase_integral(TRI, q), brute(TRI[0], q[0]), rtol=1e-6, atol=1e-12)


def test_triangle_integral_two_equal_phases():
    tri = np.array([[[0.0, 0.0, 0.0], [0.3, 0.0, 0.0], [0.1, 0.2, 0.0]]])
    q = np.array([[0.0, 40.0, 0.0]])  # vertices 0 and 1 share y = 0 -> two equal phases
    assert np.allclose(triangle_phase_integral(tri, q), brute(tri[0], q[0]), rtol=1e-6)


def test_triangle_integral_nearly_equal_phases():
    tri = np.array([[[0.0, 0.0, 0.0], [0.3, 1e-9, 0.0], [0.1, 0.2, 0.0]]])
    q = np.array([[0.0, 40.0, 0.0]])  # phases 0, 4e-8 and 8: the closed form would cancel catastrophically
    assert np.allclose(triangle_phase_integral(tri, q), brute(tri[0], q[0]), rtol=1e-6)


def test_triangle_integral_broadcasts_over_leading_axes():
    tris = np.concatenate([TRI, TRI + 0.1])
    q = np.random.default_rng(0).normal(size=(4, 5, 3)) * 20
    out = triangle_phase_integral(tris, q)
    assert out.shape == (4, 5, 2)
    assert out[2, 3, 1] == pytest.approx(brute(tris[1], q[2, 3]), rel=1e-6)


def test_degenerate_triangle_is_finite():
    tri = np.array([[[0, 0, 0], [1, 0, 0], [2, 0, 0]]], dtype=float)
    out = triangle_phase_integral(tri, np.array([[5.0, 1.0, 0.0]]))
    assert np.isfinite(out).all() and abs(out[0, 0]) < 1e-12


def test_flat_disc_against_its_own_reference_does_not_scatter():
    s = ScatteringSample.disc(flat_tile(), diameter=0.8)
    src = SourceGrid.preset("survey", symmetry="rotational")
    rec = ReceiverGrid.hemisphere(5.0, 10.0)
    p = KirchhoffSolver().solve(s, src, rec, np.array([500.0, 2000.0]), 343.0)
    assert np.allclose(scattering_freefield(p.sample, p.reference, rec.weights), 0.0, atol=1e-6)


def test_pressures_shape_and_generation():
    s = ScatteringSample.disc(flat_tile(), diameter=0.4)
    src = SourceGrid.preset("survey", symmetry="rotational")
    rec = ReceiverGrid.hemisphere(10.0, 10.0)
    solver = KirchhoffSolver(chunk=7)
    p = solver.solve(s, src, rec, np.array([500.0, 1000.0]), 343.0)
    assert isinstance(solver, ScatteringSolver) and solver.name == "kirchhoff"
    assert isinstance(p, Pressures)
    assert p.sample.shape == p.reference.shape == (src.n, len(rec.directions), 2)
    assert np.iscomplexobj(p.sample) and np.isfinite(p.sample).all()
    assert p.generation["method"] == "kirchhoff" and p.generation["shadowing"] is False
    unchunked = KirchhoffSolver().solve(s, src, rec, np.array([500.0, 1000.0]), 343.0)
    assert np.allclose(p.sample, unchunked.sample, rtol=1e-12, atol=1e-15)


def test_large_plate_reflects_specularly():
    s = ScatteringSample.disc(flat_tile(), diameter=4.0)
    src = SourceGrid(np.array([30.0]), np.array([0.0]))
    rec = ReceiverGrid.hemisphere(2.0, 10.0)
    p = KirchhoffSolver().solve(s, src, rec, np.array([4000.0]), 343.0)
    peak = rec.directions[np.argmax(np.abs(p.reference[0, :, 0]))]
    mirror = src.directions[0] * [-1, -1, 1]
    assert np.degrees(np.arccos(np.clip(peak @ mirror, -1, 1))) < 2.0


def test_sine_surface_scatters_above_the_first_grating_lobe():
    # Heimes: h = 20.4 mm, Lambda = 70.8 mm; at 31.5 deg the first lobe appears at Lambda/lambda = 1/(1 + sin 31.5) = 0.66
    s = ScatteringSample.disc(sine_tile(period=0.0708, depth=0.0204), diameter=0.8)
    src = SourceGrid(np.array([31.5]), np.array([0.0]))  # incidence plane across the grooves
    rec = ReceiverGrid.hemisphere(2.0, 10.0)
    c = 343.0
    f = np.array([0.4, 1.0]) * c / 0.0708
    p = KirchhoffSolver().solve(s, src, rec, f, c)
    s_low, s_high = scattering_freefield(p.sample, p.reference, rec.weights)[0]
    assert s_low < 0.2 and s_high > 0.4


def test_reciprocal_when_every_face_is_lit_from_both_directions():
    # gentle sine (slopes < 15 deg) so that the lit set is the same for both directions; the same azimuth (Ruling R2) so that the rim walls are too
    s = ScatteringSample.disc(sine_tile(period=0.2, depth=0.01), diameter=0.6)
    u1 = SourceGrid(np.array([40.0]), np.array([0.0]))
    u2 = SourceGrid(np.array([20.0]), np.array([0.0]))
    f = np.array([2000.0])
    p12 = KirchhoffSolver().solve(s, u1, ReceiverGrid(u2.directions, np.ones(1), 10.0), f, 343.0).sample[0, 0, 0]
    p21 = KirchhoffSolver().solve(s, u2, ReceiverGrid(u1.directions, np.ones(1), 10.0), f, 343.0).sample[0, 0, 0]
    assert p12 == pytest.approx(p21, rel=1e-9)


def test_shadowing_hides_faces_behind_others():
    s = ScatteringSample.disc(sine_tile(period=0.0708, depth=0.0204), diameter=0.3)
    src = SourceGrid(np.array([80.0]), np.array([0.0]))  # grazing across the grooves: the troughs' far flanks are in shadow
    rec = ReceiverGrid.hemisphere(10.0, 10.0)
    f = np.array([2000.0])
    plain = KirchhoffSolver().solve(s, src, rec, f, 343.0)
    shadowed = KirchhoffSolver(shadowing=True).solve(s, src, rec, f, 343.0)
    assert shadowed.generation["shadowing"] is True
    assert np.isfinite(shadowed.sample).all()
    assert not np.allclose(plain.sample, shadowed.sample)
    assert np.allclose(plain.reference, shadowed.reference)  # nothing on a flat body shadows anything


def test_warns_below_the_feature_scale():
    s = ScatteringSample.disc(sine_tile(period=0.0708, depth=0.0204), diameter=0.8, period=(0.0708, 0.05))
    nominal, exact = third_octave_centres()
    msgs = KirchhoffSolver().warnings(s, nominal, exact, 343.0)
    assert any("kirchhoff: below" in m for m in msgs)


def test_no_feature_warning_without_period_or_feature_size():
    s = ScatteringSample.as_drawn(sine_tile(period=0.0708, depth=0.0204))
    nominal, exact = third_octave_centres()
    assert not any("kirchhoff: below" in m for m in KirchhoffSolver().warnings(s, nominal, exact, 343.0))


def test_warns_on_steep_relief():
    steep = ScatteringSample.disc(sine_tile(period=0.02, depth=0.03), diameter=0.3)
    gentle = ScatteringSample.disc(sine_tile(period=0.2, depth=0.01), diameter=0.6)
    nominal, exact = third_octave_centres()
    assert "kirchhoff: steep relief; multiple reflections are ignored" in KirchhoffSolver().warnings(steep, nominal, exact, 343.0)
    assert "kirchhoff: steep relief; multiple reflections are ignored" not in KirchhoffSolver().warnings(gentle, nominal, exact, 343.0)
