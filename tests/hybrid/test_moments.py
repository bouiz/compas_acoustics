"""The ambisonic late field from spherical harmonic energy moments (compas_acoustics.hybrid.moments)."""

import numpy as np
import pytest

pytest.importorskip("pyFDN")
pytest.importorskip("spharpy")

import compas_acoustics as ca  # noqa: E402
from compas_acoustics.hybrid.envelope import octave_band_split  # noqa: E402
from compas_acoustics.hybrid.moments import MOMENT_GRID_POINTS  # noqa: E402
from compas_acoustics.hybrid.moments import LateField  # noqa: E402
from compas_acoustics.hybrid.moments import absorption_shape  # noqa: E402
from compas_acoustics.hybrid.moments import covariance  # noqa: E402
from compas_acoustics.hybrid.moments import gaunt  # noqa: E402
from compas_acoustics.hybrid.moments import histogram_moments  # noqa: E402
from compas_acoustics.hybrid.moments import moment_grid  # noqa: E402
from compas_acoustics.hybrid.moments import moments_from_hits  # noqa: E402
from compas_acoustics.hybrid.moments import network_energy  # noqa: E402
from compas_acoustics.hybrid.moments import ray_late_field  # noqa: E402
from compas_acoustics.hybrid.moments import sh_basis  # noqa: E402
from compas_acoustics.hybrid.moments import synthesize  # noqa: E402
from compas_acoustics.hybrid.target import RoomTarget  # noqa: E402
from compas_acoustics.room.raytracing import DirectionalHistogram  # noqa: E402

FS = 48000


def hits(n, seed=1, lobe=(0.6, 0.3, 0.74)):
    """Half isotropic, half a lobe: an anisotropic late field's rays."""
    rng = np.random.default_rng(seed)
    d = rng.normal(size=(n, 3))
    d[: n // 2] += 2.0 * np.asarray(lobe)
    return d / np.linalg.norm(d, axis=1, keepdims=True), rng.exponential(size=n)


@pytest.mark.parametrize("sh_order", [0, 1, 2, 3])
def test_the_covariance_from_the_moments_is_the_rays_covariance(sh_order):
    """sum_r e Y_i Y_j = sum_k G_ijk M_k exactly: the moments of order 2N carry the order-N covariance."""
    d, e = hits(400)
    Y = sh_basis(sh_order, d)
    direct = (e[:, np.newaxis] * Y).T @ Y
    from_moments = covariance(moments_from_hits(d, e, 2 * sh_order), sh_order)
    np.testing.assert_allclose(from_moments, direct, rtol=0, atol=1e-12 * np.abs(direct).max())
    assert np.linalg.eigvalsh(from_moments).min() > -1e-10 * np.abs(direct).max()


def test_an_isotropic_field_has_uncorrelated_channels_of_equal_energy():
    """Only M_0: C = G_ij0 M_0 = M_0 I in N3D with W = 1."""
    for sh_order in range(4):
        moments = np.zeros((2 * sh_order + 1) ** 2)
        moments[0] = 2.5
        np.testing.assert_allclose(covariance(moments, sh_order), 2.5 * np.eye((sh_order + 1) ** 2), atol=1e-12)
    assert gaunt(1).shape == (4, 4, 9)


def test_a_plane_wave_is_one_direction_at_full_correlation():
    d = np.array([[0.0, 0.0, 1.0]])
    C = covariance(moments_from_hits(d, np.ones(1), 2), 1)
    y = sh_basis(1, d)[0]
    np.testing.assert_allclose(C, np.outer(y, y), atol=1e-12)
    assert np.linalg.matrix_rank(C, tol=1e-9) == 1


def test_the_moment_grid_moves_the_covariance_by_under_half_a_percent():
    """The tracer sorts the rays on the grid: binned to the nearest of its directions, the covariance hardly moves."""
    from scipy.spatial import cKDTree

    grid = moment_grid()
    assert grid.shape == (MOMENT_GRID_POINTS, 3)
    d, e = hits(100_000)
    nearest = cKDTree(grid).query(d)[1]
    binned = np.bincount(nearest, weights=e, minlength=grid.shape[0])
    for sh_order in (1, 2, 3):
        exact = covariance(moments_from_hits(d, e, 2 * sh_order), sh_order)
        sorted_ = covariance(binned @ sh_basis(2 * sh_order, grid), sh_order)
        assert np.linalg.norm(sorted_ - exact) / np.linalg.norm(exact) < 0.005


def test_histogram_moments_turn_the_grid_into_the_receivers_frame():
    """A bin of the grid's direction u in room coordinates is u @ R in the receiver's frame (columns: its axes)."""
    rotation = np.array([[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]])  # facing -x
    directions = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    energy = np.zeros((2, 8, 3))
    energy[0, :, 1] = 1.0  # from +x of the room: behind the receiver
    histogram = DirectionalHistogram(energy=energy, bin_size=0.004, directions=directions, bands=62.5 * 2.0 ** np.arange(8))
    moments = histogram_moments(histogram, 1, rotation)
    assert moments.shape == (8, 3, 4)
    np.testing.assert_allclose(moments[:, 1], np.tile(sh_basis(1, np.array([[-1.0, 0.0, 0.0]])), (8, 1)), atol=1e-12)


def field(energy, shape, sh_order, bin_samples=1, seed=3):
    return LateField(energy=energy, shape=shape, bin_samples=bin_samples, moment_order=2 * sh_order, follows_reference=False, seed=seed, source="rays")


@pytest.mark.parametrize("sh_order", [1, 2, 3])
def test_one_energy_and_shape_in_every_band_is_white_noise_of_that_covariance(sh_order):
    d, e = hits(2000, seed=4)
    moments = moments_from_hits(d, e, 2 * sh_order)
    shape = np.tile(moments / moments[0], (9, 1))
    n = 2 * FS
    y = synthesize(field(np.full((9, n), 1e-6), shape, sh_order), sh_order, FS)
    C = y.T @ y / n
    target = 1e-6 * covariance(shape[0], sh_order)
    assert abs(C[0, 0] / 1e-6 - 1.0) < 0.02
    assert np.linalg.norm(C - target) / np.linalg.norm(target) < 0.05
    spectrum = np.abs(np.fft.rfft(y[:, 0])) ** 2
    halves = np.array_split(spectrum[len(spectrum) // 100 :], 4)
    level = 10 * np.log10([h.mean() for h in halves])
    assert np.ptp(level) < 0.5  # flat


def test_each_band_gets_its_own_energy():
    energy = np.full((9, FS), 1e-6)
    energy[4] = 1e-4  # 1 kHz 20 dB up
    shape = np.zeros((9, 9))
    shape[:, 0] = 1.0
    y = synthesize(field(energy, shape, 1), 1, FS)
    bands = (octave_band_split(y[:, 0], FS) ** 2).sum(axis=1)
    assert 10 * np.log10(bands[4] / bands[2]) > 15.0
    assert y.shape == (FS, 4)


def test_the_noise_follows_the_seed_and_a_constant_shape_per_bin_is_the_constant_shape():
    shape = np.zeros((9, 25))
    shape[:, 0] = 1.0
    shape[:, 2] = 0.5  # a vertical tilt
    energy = np.full((9, FS // 2), 1e-6)
    a = synthesize(field(energy, shape, 2), 2, FS)
    np.testing.assert_array_equal(a, synthesize(field(energy, shape, 2), 2, FS))
    assert not np.allclose(a, synthesize(field(energy, shape, 2, seed=4), 2, FS))
    per_bin = np.repeat(shape[:, np.newaxis], 200, axis=1)
    np.testing.assert_allclose(synthesize(field(energy, per_bin, 2, bin_samples=192), 2, FS), a, atol=1e-12)


def test_an_order_above_half_the_moments_is_refused():
    shape = np.zeros((9, 9))
    shape[:, 0] = 1.0
    with pytest.raises(ValueError, match="order-2 output needs 4"):
        synthesize(field(np.ones((9, 10)), shape, 1), 2, FS)


def test_the_absorption_shape_leans_away_from_the_absorbing_side():
    grid = moment_grid()
    absorption = np.where(grid[:, :1] < 0.0, 0.9, 0.1) * np.ones((1, 8))
    shape, level = absorption_shape(absorption, grid, 2)  # moments to second order: an order-1 output's
    assert shape.shape == (9, 9) and np.allclose(shape[:, 0], 1.0)
    assert np.all(shape[:, 3] > 0.3)  # ACN 3 is x: the energy comes from +x
    np.testing.assert_allclose(level, 0.5, atol=0.02)  # the mean reflection, 1 - (0.9 + 0.1) / 2
    isotropic, ones = absorption_shape(None, grid, 2)
    assert np.allclose(isotropic[:, 1:], 0.0) and np.allclose(ones, 1.0)


def target(rt=0.5, n_slopes=1):
    return RoomTarget(rt=np.full((8, n_slopes), rt), level=np.ones((8, n_slopes)), bands=62.5 * 2.0 ** np.arange(8), fs=FS)


def test_the_network_energy_of_one_image_decays_at_the_target_from_the_first_pass():
    delays = np.arange(400, 801)  # samples
    out = network_energy(delays, (np.array([0.01]), np.ones((1, 8))), target(0.5), FS, FS)
    first = int(0.01 * FS)
    assert np.all(out[:, : first + 400] < 1e-15) and np.all(out[:4, first + 801] > 1e-6)
    # once every line has passed, one unit of energy leaves at 1 / L per sample and decays 60 dB in 0.5 s
    t = np.arange(first + 1000, first + 20000)
    slope = np.polyfit((t - first) / FS, 10 * np.log10(out[3, t]), 1)[0]
    assert slope == pytest.approx(-120.0, rel=0.01)
    assert out[3, t[0]] == pytest.approx(np.exp(-6 * np.log(10) * 1000 / (0.5 * FS)) / 600.0, rel=1e-6)


def test_a_tapped_part_is_heard_at_once():
    delays = np.arange(400, 801)
    out = network_energy(delays, (np.zeros(0), np.zeros((0, 8))), target(), 5000, FS, tap=(np.array([0.01]), np.ones((1, 8))))
    assert out[3, int(0.01 * FS)] > 1e-6 and np.all(out[:, : int(0.01 * FS)] < 1e-15)


def ray_histogram(rt=0.6, n_bins=300, bands=8, onset=10):
    """An exponential decay per band from bin ``onset``, all from +z; a bin holds the energy at its centre."""
    t = (np.arange(n_bins) + 0.5) * 0.004
    energy = np.where(np.arange(n_bins) >= onset, np.exp(-6 * np.log(10) * t / rt), 0.0)
    e = np.zeros((2, bands, n_bins))
    e[0] = energy
    return DirectionalHistogram(energy=e, bin_size=0.004, directions=np.array([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]]), bands=62.5 * 2.0 ** np.arange(bands))


def test_the_rays_energy_is_the_bins_per_sample_and_continues_at_the_target_past_the_hold():
    histogram = ray_histogram()
    n = int(2.0 * FS)  # beyond the 1.2 s of bins
    energy, shape, bin_samples = ray_late_field(histogram, 1, None, target(0.6), n, FS)
    assert bin_samples == 192 and energy.shape == (9, n) and shape.shape[0] == 9
    mid = int(0.3 * FS)
    assert energy[3, mid] == pytest.approx(np.exp(-6 * np.log(10) * 0.3 / 0.6) / 192, rel=0.01)
    late = np.arange(int(1.3 * FS), int(1.9 * FS))
    slope = np.polyfit(late / FS, 10 * np.log10(energy[3, late]), 1)[0]
    assert slope == pytest.approx(-100.0, rel=0.02)
    assert np.all(energy[:, : int(0.02 * FS)] < 1e-3 * energy[:, int(0.06 * FS) : int(0.06 * FS) + 1])
    np.testing.assert_allclose(shape[3, 50], moments_from_hits(np.array([[0.0, 0.0, 1.0]]), np.ones(1), 2), atol=1e-9)
    np.testing.assert_allclose(shape[3, 0], shape[3, 50])  # before the rays: their first shape


def test_one_loud_bin_before_the_hold_does_not_lift_the_continuation():
    """The last bins before the hold hold few rays; the continuation starts from the level of the last HOLD_FIT_MS of
    them, so a bin ten times too loud there moves it by under 1.5 dB (one bin's 10 dB), and the energy has no step at the hold."""
    histogram = ray_histogram()
    edc = np.cumsum(histogram.energy[0, 3, ::-1])[::-1]
    hold = int(np.flatnonzero(edc <= 1e-6 * edc[0])[0])
    histogram.energy[0, :, hold - 1] *= 10.0
    energy, _, _ = ray_late_field(histogram, 1, None, target(0.6), int(1.5 * FS), FS, shape=False)
    expected = np.exp(-6 * np.log(10) * 1.2 / 0.6) / 192
    assert abs(10 * np.log10(energy[3, int(1.2 * FS)] / expected)) < 1.5
    level = 10 * np.log10(energy[3, int(0.3 * FS) : int(1.4 * FS)])
    assert np.max(np.abs(np.diff(level[:: FS // 250]))) < 0.4 + 100.0 / 250  # 4 ms steps: the decay's 0.4 dB, no jump


def test_without_a_16_khz_row_the_8_khz_row_decays_at_the_networks_top_rate():
    energy, _, _ = ray_late_field(ray_histogram(bands=8), 1, None, target(0.6), FS, FS, shape=False)
    nine, _, _ = ray_late_field(ray_histogram(bands=9), 1, None, target(0.6), FS, FS, shape=False)
    t = int(0.5 * FS)
    assert energy[8, t] < 0.5 * energy[7, t]
    early = int(0.3 * FS)  # the histogram's own row, before the continuation fades in at the 16 kHz target's decay
    assert nine[8, early] == pytest.approx(nine[7, early])


# ---------------------------------------------------------------- in the method --
def office():
    scene = ca.Scene("office", sampling_rate=FS)
    scene.add_acoustic_objects(ca.AcousticObjectsLibrary.shoebox_room(6, 5, 3, materials=ca.AcousticMaterial("wall", 0.2, 0.1)))
    scene.add_source(ca.Source("talker", 1.5, 1.5, 1.6))
    scene.add_receiver(ca.Receiver("listener", 4.0, 3.0, 1.2))
    return scene


def test_the_rays_late_field_carries_the_references_late_energy():
    """Per octave 250 Hz to 4 kHz, the late field's W energy is the reference's minus the early field's within 1.5 dB."""
    method = ca.Ambisonics(sh_order=1, duration=0.8)
    office().simulate(method)
    ((key, result),) = method.results.items()
    reference = np.asarray(method.reference_rirs[key].denormalized(), dtype=float).ravel()
    n = min(reference.size, result.early.shape[0])

    def bands(x):
        return (octave_band_split(x, FS) ** 2).sum(axis=1)

    late_part = bands(reference[:n]) - bands(result.early[:n, 0])
    difference = 10 * np.log10(bands(result.late[:n, 0]) / late_part)
    assert result.core.late.source == "rays"
    assert np.all(np.abs(difference[2:7]) < 1.5), difference


def test_with_the_rays_energy_the_images_keep_only_their_specular_part():
    """The reference run's rays carry every reflection's scattered part, so with their energy the images keep
    ``prod sqrt(1 - s)``, as pyroomacoustics' own images do; the FDN's images keep the whole reflection."""
    scene = ca.Scene("scattering", sampling_rate=FS)
    scene.add_acoustic_objects(ca.AcousticObjectsLibrary.shoebox_room(6, 5, 3, materials=ca.AcousticMaterial("wall", 0.2, 0.3)))
    scene.add_source(ca.Source("talker", 1.5, 1.5, 1.6))
    scene.add_receiver(ca.Receiver("listener", 4.0, 3.0, 1.2))
    gains = {}
    for late_field in ("moments", "fdn"):
        method = ca.Ambisonics(sh_order=1, duration=0.5, image_source_order=2, late_field=late_field)
        scene.simulate(method)
        ((result,),) = [list(method.results.values())]
        gains[late_field] = result.core.images
    moments, fdn = gains["moments"], gains["fdn"]
    assert moments.delay.shape == fdn.delay.shape and np.allclose(moments.delay, fdn.delay)
    for order in (0, 1, 2):
        ratio = moments.gain[moments.order == order] ** 2 / fdn.gain[fdn.order == order] ** 2
        np.testing.assert_allclose(ratio, 0.7**order, rtol=1e-6)
