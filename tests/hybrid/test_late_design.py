"""Ported from hybrid-rir (gpapadimitri/hybrid-rir @ 8a70661, tests/test_late_design.py)."""

import numpy as np
import pyFDN
import pytest
import spharpy
from scipy.signal import sosfreqz

from compas_acoustics.hybrid.bands import BANDS, N_BANDS
from compas_acoustics.hybrid.late import ambisonic_output_matrix, design_fdn, render
from compas_acoustics.hybrid.target import RoomTarget

FS = 48000.0


def target_with(rt=0.8, level=None):
    rt = np.full(N_BANDS, rt) if np.isscalar(rt) else np.asarray(rt, dtype=float)
    return RoomTarget(
        rt=rt,
        level=np.ones_like(rt) if level is None else level,
        bands=BANDS.copy(),
        fs=FS,
    )


def delays_of(n, lo=500, hi=1500, seed=0):
    return pyFDN.sample_delay_lengths(n, (lo, hi), coprime=True, rng=seed)


def impulse_response(builds, n_samples):
    """Every line fed 1 / sqrt(N) of an impulse: the impulse-excited FDN."""
    n = np.asarray(builds[0].delays).size
    drive = np.zeros((n_samples, n))
    drive[0] = 1.0 / np.sqrt(n)
    return render(builds, drive)


def test_output_matrix_shape():
    dirs = spharpy.samplings.equal_area(n_max=1, n_points=16)
    assert ambisonic_output_matrix(1, dirs).shape == (4, 16)


def test_output_matrix_is_orthogonal_for_the_default_grid():
    """C @ C.T diagonal means the late field is diffuse, not directionally biased.

    A default spiral_points grid of the same size leaks 21.5 per cent, which is
    why equal_area is used.
    """
    dirs = spharpy.samplings.equal_area(n_max=1, n_points=16)
    c = ambisonic_output_matrix(1, dirs)
    gram = c @ c.T
    off = gram - np.diag(np.diag(gram))
    assert np.max(np.abs(off)) / np.mean(np.abs(np.diag(gram))) < 1e-6


def test_output_matrix_rejects_a_grid_that_cannot_support_the_order():
    dirs = spharpy.samplings.equal_area(n_max=3, n_points=8)
    with pytest.raises(ValueError, match="leaks"):
        ambisonic_output_matrix(3, dirs)


def test_output_matrix_w_row_is_constant():
    dirs = spharpy.samplings.equal_area(n_max=1, n_points=16)
    c = ambisonic_output_matrix(1, dirs)
    np.testing.assert_allclose(c[0], c[0, 0])


def test_design_produces_a_valid_build():
    (build,) = design_fdn(target_with(), delays_of(16), sh_order=1)
    assert isinstance(build, pyFDN.FDNBuild)
    assert build.A.shape == (16, 16)
    assert build.C.shape == (4, 16)
    assert build.post_delay is not None
    assert build.post_output is None


def test_the_given_delays_are_used():
    delays = delays_of(16)
    (build,) = design_fdn(target_with(), delays)
    np.testing.assert_array_equal(build.delays, delays)


def test_design_is_reproducible_under_a_seed():
    (a,) = design_fdn(target_with(), delays_of(16), rng=7)
    (b,) = design_fdn(target_with(), delays_of(16), rng=7)
    np.testing.assert_allclose(a.A, b.A)


def test_feedback_matrix_is_orthogonal():
    (build,) = design_fdn(target_with(), delays_of(16))
    np.testing.assert_allclose(build.A @ build.A.T, np.eye(16), atol=1e-10)


def test_rendered_rt_matches_a_flat_target():
    target = target_with(rt=1.0)
    builds = design_fdn(target, delays_of(16, seed=3), rng=3)
    ir = impulse_response(builds, int(2.5 * FS))
    rt, _ = pyFDN.estimate_rt_bands(ir[:, 0], FS)
    np.testing.assert_allclose(rt[1:-1], target.rt[1:-1, 0], rtol=0.10)


def test_rendered_rt_matches_a_sloped_target():
    rt = np.linspace(1.4, 0.5, N_BANDS)
    builds = design_fdn(target_with(rt=rt), delays_of(16, seed=4), rng=4)
    ir = impulse_response(builds, int(3.5 * FS))
    got, _ = pyFDN.estimate_rt_bands(ir[:, 0], FS)
    np.testing.assert_allclose(got[1:-1], rt[1:-1], rtol=0.15)


def test_multi_slope_weights_keep_level_ratios_and_total_initial_energy():
    """Every slope network takes the same input, so each is weighted by its
    share of the initial level: the ratio between slopes follows target.level,
    and the squared weights sum to one per band."""
    level = np.stack([np.full(N_BANDS, 1.0), np.full(N_BANDS, 0.25)], axis=1)
    target = RoomTarget(
        rt=np.stack([np.full(N_BANDS, 0.3), np.full(N_BANDS, 1.2)], axis=1),
        level=level,
        bands=BANDS.copy(),
        fs=FS,
    )
    builds = design_fdn(target, [delays_of(16, seed=0), delays_of(16, seed=1)])
    w = []
    for build in builds:
        _, h = sosfreqz(build.post_output[:, :, 0], worN=[1000.0], fs=FS)
        w.append(np.abs(h[0]))
    assert w[1] / w[0] == pytest.approx(0.25, rel=0.05)
    assert w[0] ** 2 + w[1] ** 2 == pytest.approx(1.0, rel=0.05)


def test_delays_must_be_one_array_or_one_per_slope():
    with pytest.raises(ValueError, match="one per slope"):
        design_fdn(target_with(), [delays_of(16), delays_of(16, seed=1)])


def test_render_rejects_a_drive_with_the_wrong_number_of_lines():
    builds = design_fdn(target_with(), delays_of(16))
    with pytest.raises(ValueError, match="one column per delay line"):
        render(builds, np.zeros((256, 3)))


def test_render_matches_pyfdn_for_an_impulse_through_b():
    """Driving each line with B times an impulse is the impulse-excited network."""
    build = pyFDN.fdn_build_gallery(
        8, fs=FS, delay_range=(500, 1500), num_outputs=2,
        io_type="normalized", direct_gain=0.0, rt=1.0, rng=0,
    )
    n_samples = 4096
    drive = np.zeros((n_samples, 8))
    drive[0] = build.B[:, 0]
    np.testing.assert_allclose(
        render([build], drive), pyFDN.build_to_impz(build, n_samples)[:, :, 0], atol=1e-9
    )


def test_a_single_line_emerges_after_that_lines_delay():
    delays = delays_of(16)
    builds = design_fdn(target_with(), delays)
    drive = np.zeros((4096, 16))
    drive[0, 3] = 1.0
    out = render(builds, drive)
    assert int(np.argmax(np.abs(out[:, 0]) > 1e-12)) == delays[3]
