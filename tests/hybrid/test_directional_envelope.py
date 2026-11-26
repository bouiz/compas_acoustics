"""The directional ray-tracing histogram (Schroeder 2011) and the per-line late envelope it shapes."""

import numpy as np
import pyroomacoustics as pra
import pytest

from compas_acoustics import AcousticMaterial
from compas_acoustics import AcousticObjectsLibrary
from compas_acoustics import Scene
from compas_acoustics.hybrid.late import fdn_directions
from compas_acoustics.room.air import AirConditions
from compas_acoustics.room.diffuse import air_attenuation
from compas_acoustics.room.raytracing import DirectionalHistogram
from compas_acoustics.room.raytracing import directional_histograms
from compas_acoustics.room.geometry import RoomGeometry

FS = 48000
SOURCE = np.array([6.0, 3.0, 1.5])
RECEIVER = np.array([2.0, 3.0, 1.5])


def geometry(absorption=0.15, scattering=0.0):
    scene = Scene("histogram", sampling_rate=FS)
    scene.add_acoustic_objects(AcousticObjectsLibrary.shoebox_room(8, 6, 3, materials=AcousticMaterial("wall", absorption, [scattering] * 7)))
    return RoomGeometry.from_scene(scene)


def grid():
    unit = fdn_directions().cartesian
    return unit / np.linalg.norm(unit, axis=1, keepdims=True)


def histograms(g, directions, **kwargs):
    kwargs = dict(dict(max_order=3, duration=0.5, n_rays=5000, air_absorption=False, seed=1), **kwargs)
    return directional_histograms(g, [pra.SoundSource(SOURCE)], RECEIVER[:, np.newaxis], FS, [directions], **kwargs)


# ------------------------------------------------------------- the histogram --
def test_one_histogram_per_microphone_and_source_binned_by_line():
    ((hist,),) = histograms(geometry(), grid())
    assert isinstance(hist, DirectionalHistogram)
    assert hist.energy.shape[:2] == (96, 9)
    assert hist.energy.shape[2] == pytest.approx(0.5 / hist.bin_size, abs=1)
    assert hist.directions.shape == (96, 3)
    assert np.all(hist.energy >= 0.0)


def test_a_run_longer_than_pyroomacoustics_default_cap_keeps_every_line():
    """libroom sizes each line's histogram for 10 s and doubles only the lines hit later; the lines must still stack."""
    # 500 rays: some of the 96 lines are hit after 10 s, others are not
    ((hist,),) = histograms(geometry(absorption=0.02), grid(), duration=12.0, n_rays=500, energy_threshold=1e-8)
    assert hist.energy.shape == (96, 9, 3000)
    assert hist.energy[:, :, 2500:].sum() > 0.0  # rays past 10 s are kept


def test_the_lines_sum_to_pyroomacoustics_own_histogram():
    """Without scattering nothing is random: the omni run and the directional run see the same rays."""
    g = geometry(scattering=0.0)
    ((hist,),) = histograms(g, grid())
    room = g.to_pra(FS, 3, ray_tracing=True, air_absorption=False)
    room.add_soundsource(pra.SoundSource(SOURCE))
    room.add_microphone_array(pra.MicrophoneArray(RECEIVER[:, np.newaxis], FS))
    room.set_ray_tracing(n_rays=5000, receiver_radius=0.5, energy_thres=1e-8, time_thres=0.5, hist_bin_size=0.004)
    room.ray_tracing()
    (omni,) = room.rt_histograms[0][0]
    n = min(omni.shape[1], hist.energy.shape[2])
    # the package scale is 1 / (4 pi) in amplitude
    np.testing.assert_allclose(hist.energy.sum(axis=0)[:, :n] * (4 * np.pi) ** 2, omni[:, :n], rtol=1e-5, atol=1e-12)


def test_energy_is_binned_by_the_direction_it_arrives_from():
    """With image source order 0 the direct sound is logged by the rays: in the line facing the source."""
    directions = grid()
    ((hist,),) = histograms(geometry(), directions, max_order=0)
    first = int(np.flatnonzero(hist.energy.sum(axis=(0, 1)))[0])
    towards_source = np.argmax(directions @ ((SOURCE - RECEIVER) / np.linalg.norm(SOURCE - RECEIVER)))
    assert np.argmax(hist.energy[:, :, first].sum(axis=1)) == towards_source


def test_air_absorption_attenuates_the_late_bins():
    g = geometry()
    ((dry,),) = histograms(g, grid(), air_absorption=False)
    ((damped,),) = histograms(g, grid(), air_absorption=True)
    late = slice(int(0.4 / dry.bin_size), int(0.5 / dry.bin_size))
    ratio = damped.energy[:, 7, late].sum() / dry.energy[:, 7, late].sum()
    # the 8 kHz band in the default air over 0.45 s of travel
    m, c = air_attenuation()[7], AirConditions().speed_of_sound
    assert 10 * np.log10(ratio) == pytest.approx(-10 * np.log10(np.e) * m * 0.45 * c, abs=1.5)
    assert np.all(damped.energy <= dry.energy + 1e-15)
    early = slice(0, int(0.05 / dry.bin_size))
    np.testing.assert_allclose(damped.energy[:, 1, early], dry.energy[:, 1, early], rtol=0.01, atol=1e-15)


# --------------------------------------------------------- the per-line gains --
from compas_acoustics.hybrid.envelope import directional_shares  # noqa: E402
from compas_acoustics.hybrid.envelope import octave_band_split  # noqa: E402
from compas_acoustics.hybrid.envelope import shape_late  # noqa: E402
from compas_acoustics.hybrid.envelope import smooth_over_directions  # noqa: E402

N = int(1.0 * FS)
T = np.arange(N) / FS
BANDS9 = 62.5 * 2.0 ** np.arange(9)


def lines(rt=0.8, onset=0.03, seed=0):
    """96 decaying noise lines from ``onset``, (N, 96)."""
    rng = np.random.default_rng(seed)
    decay = 10 ** (-3 * np.clip(T - onset, 0.0, None) / rt) * (T >= onset)
    return 0.02 * rng.standard_normal((N, 96)) * decay[:, None]


def histogram_of(late, scale=1.0, bin_size=0.004):
    """The histogram the lines themselves describe: a direction's energy per bin is its line's over 96."""
    n_bin = round(bin_size * FS)
    bands = octave_band_split(late, FS) ** 2  # (9, N, 96)
    n_bins = N // n_bin
    per_bin = bands[:, : n_bins * n_bin].reshape(9, n_bins, n_bin, 96).sum(axis=2)  # (9, n_bins, 96)
    energy = np.transpose(per_bin, (2, 0, 1)) / 96.0
    return DirectionalHistogram(energy=energy * scale, bin_size=n_bin / FS, directions=grid(), bands=BANDS9)


def energy_db(x, a, b):
    return 10 * np.log10(np.sum(x[int(a * FS) : int(b * FS)] ** 2, axis=0))


def early_w(late):
    """A few early impulses before the lines start."""
    early = np.zeros(N)
    for t, a in ((0.01, 1.0), (0.015, 0.5), (0.022, 0.4), (0.027, 0.3)):
        early[int(t * FS)] = a * 0.02 * np.sqrt(96) * 3
    return early


def test_the_shares_are_one_where_the_lines_already_hold_the_rays_distribution():
    late = lines()
    bands = octave_band_split(late, FS)
    for b in range(8):
        shares = directional_shares(histogram_of(late, 3.0), bands[b], FS, b)
        assert shares.shape == (N, 96)
        np.testing.assert_allclose(shares[int(0.1 * FS) :], 1.0, atol=0.02)


def test_a_louder_directional_target_leaves_the_lines_alone():
    late = lines()
    reference = early_w(late) + 3.0 * late.sum(axis=1) / np.sqrt(96)
    shaped = shape_late(late, early_w(late), late.sum(axis=1) / np.sqrt(96), reference, FS, histogram=histogram_of(late, 3.0))
    # the 4 ms bins of the histogram against the samples of the lines: within 1 % of the lines' amplitude once the
    # 100 ms window is past the lines' onset, 2.5 % before (a band's share over few cycles)
    np.testing.assert_allclose(shaped[int(0.15 * FS) :], late[int(0.15 * FS) :], atol=0.01 * np.abs(late).max())
    np.testing.assert_allclose(shaped, late, atol=0.025 * np.abs(late).max())


def test_a_quieter_hemisphere_moves_energy_to_the_other_and_keeps_the_total():
    """Half the directions at -6 dB: their share falls, the others' rises, 6 dB apart, the omni level untouched."""
    late = lines()
    directions = grid()
    scale = np.where(directions[:, 0] < 0, 10 ** (-6 / 10), 1.0)[:, None, None]
    reference = early_w(late) + 3.0 * late.sum(axis=1) / np.sqrt(96)
    shaped = shape_late(late, early_w(late), late.sum(axis=1) / np.sqrt(96), reference, FS, histogram=histogram_of(late, scale))
    lost = energy_db(shaped, 0.15, 0.6) - energy_db(late, 0.15, 0.6)
    west, east = lost[directions[:, 0] < -0.5], lost[directions[:, 0] > 0.5]
    assert np.all(np.abs(west - east.mean() + 6.0) < 0.5)
    assert np.all(np.abs(east - east.mean()) < 0.5)
    total = 10 * np.log10(np.sum(shaped[int(0.15 * FS) : int(0.6 * FS)] ** 2) / np.sum(late[int(0.15 * FS) : int(0.6 * FS)] ** 2))
    assert abs(total) < 0.2


def band_db(x, b, a, c):
    """Energy in dB of octave band ``b`` of every line between ``a`` and ``c`` seconds."""
    return energy_db(octave_band_split(x, FS)[b], a, c)


def shaped_by(scale):
    late = lines()
    reference = early_w(late) + 3.0 * late.sum(axis=1) / np.sqrt(96)
    return late, shape_late(late, early_w(late), late.sum(axis=1) / np.sqrt(96), reference, FS, histogram=histogram_of(late, scale))


def test_a_direction_quieter_in_one_band_only_is_quieter_in_that_band_only():
    """The 4 kHz row of the western directions at -6 dB: that band follows, 500 Hz does not, each band keeps its total.

    Measured through the same overlapping octave bank, a -6 dB band reads about -5 dB and spills 0.5 dB into 2 kHz."""
    west = grid()[:, 0] < -0.5
    east = grid()[:, 0] > 0.5
    scale = np.ones((96, 9, 1))
    scale[grid()[:, 0] < 0, 6] = 10 ** (-6 / 10)
    late, shaped = shaped_by(scale)
    for b, low, high in ((6, -6.0, -4.5), (3, -0.1, 0.1)):
        lost = band_db(shaped, b, 0.15, 0.6) - band_db(late, b, 0.15, 0.6)
        assert low < lost[west].mean() - lost[east].mean() < high
        total = 10 * np.log10(np.sum(10 ** (band_db(shaped, b, 0.15, 0.6) / 10)) / np.sum(10 ** (band_db(late, b, 0.15, 0.6) / 10)))
        assert abs(total) < 0.2


def test_the_16_khz_band_takes_its_own_shares():
    """The 16 kHz row of the western directions at -6 dB: that band follows; the 8 kHz row at -20 dB does not."""
    west = grid()[:, 0] < -0.5
    east = grid()[:, 0] > 0.5
    scale = np.ones((96, 9, 1))
    scale[grid()[:, 0] < 0, 8] = 10 ** (-6 / 10)
    late, shaped = shaped_by(scale)
    lost = band_db(shaped, 8, 0.15, 0.6) - band_db(late, 8, 0.15, 0.6)
    assert -6.0 < lost[west].mean() - lost[east].mean() < -4.5
    scale = np.ones((96, 9, 1))
    scale[grid()[:, 0] < 0, 7] = 10 ** (-20 / 10)
    late, shaped = shaped_by(scale)
    lost = band_db(shaped, 8, 0.15, 0.6) - band_db(late, 8, 0.15, 0.6)
    assert abs(lost[west].mean() - lost[east].mean()) < 1.5


def test_a_histogram_on_other_bands_than_the_lines_is_refused():
    late = lines()
    reference = early_w(late) + 3.0 * late.sum(axis=1) / np.sqrt(96)
    h = histogram_of(late, 3.0)
    for bands, energy in ((125.0 * 2.0 ** np.arange(9), h.energy), (BANDS9[:7], h.energy[:, :7])):
        other = DirectionalHistogram(energy=energy, bin_size=h.bin_size, directions=h.directions, bands=bands)
        with pytest.raises(ValueError, match="bands"):
            shape_late(late, early_w(late), late.sum(axis=1) / np.sqrt(96), reference, FS, histogram=other)


def test_a_histogram_without_a_16_khz_row_gives_that_band_its_last_row():
    late = lines()
    reference = early_w(late) + 3.0 * late.sum(axis=1) / np.sqrt(96)
    h = histogram_of(late, 3.0)
    eight = DirectionalHistogram(energy=h.energy[:, :8], bin_size=h.bin_size, directions=h.directions, bands=BANDS9[:8])
    shaped = shape_late(late, early_w(late), late.sum(axis=1) / np.sqrt(96), reference, FS, histogram=eight)
    assert np.all(np.isfinite(shaped))


# --------------------------------------------------------------- the ray count --
from compas_acoustics.hybrid.envelope import histogram_rays  # noqa: E402


def test_the_ray_count_is_the_one_of_a_1_db_error_of_a_directions_slot():
    """Six directions along the axes share no kernel weight: a direction holds a sixth of the hits, over the Hann
    window's 2/3 of 100 ms, and sigma = 4.34 / sqrt(hits) = 1 dB."""
    axes = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], dtype=float)
    hits = (10 / np.log(10)) ** 2
    expected = hits * 1000.0 * 6 / (np.pi * 0.5**2 * 340.0 * 0.1 * 2 / 3)
    assert histogram_rays(1000.0, 0.5, 340.0, axes) == int(np.ceil(expected))
    window = np.hanning(48002)[1:-1]
    assert abs(window.sum() ** 2 / (window**2).sum() / window.size - 2 / 3) < 1e-4


def test_the_ray_count_grows_with_the_volume_and_falls_with_the_pooled_directions():
    one = histogram_rays(1000.0, 0.5, 340.0, grid())
    assert abs(histogram_rays(3000.0, 0.5, 340.0, grid()) / one - 3.0) < 1e-3
    assert abs(histogram_rays(1000.0, 1.0, 340.0, grid()) / one - 0.25) < 1e-3
    hits = (10 / np.log(10)) ** 2
    unpooled = hits * 1000.0 * 96 / (np.pi * 0.5**2 * 340.0 * 0.1 * 2 / 3)
    assert unpooled / 8 < one < unpooled / 3


def test_each_band_holds_its_shares_once_its_own_rays_have_decayed():
    """The 8 kHz row decays to -60 dB by about 0.2 s; a western drop in it after 0.4 s must not reach the 8 kHz band."""
    west = grid()[:, 0] < -0.5
    east = grid()[:, 0] > 0.5
    late = lines()
    h = histogram_of(late)
    t = (np.arange(h.energy.shape[2]) + 0.5) * h.bin_size
    energy = h.energy.copy()
    energy[:, 7] *= 10 ** (-6 * t / 0.2)
    energy[grid()[:, 0] < 0, 7] *= np.where(t > 0.4, 10 ** (-12 / 10), 1.0)
    histogram = DirectionalHistogram(energy=energy * 3.0, bin_size=h.bin_size, directions=h.directions, bands=h.bands)
    reference = early_w(late) + 3.0 * late.sum(axis=1) / np.sqrt(96)
    shaped = shape_late(late, early_w(late), late.sum(axis=1) / np.sqrt(96), reference, FS, histogram=histogram)
    lost = band_db(shaped, 7, 0.5, 0.9) - band_db(late, 7, 0.5, 0.9)
    assert abs(lost[west].mean() - lost[east].mean()) < 0.5


def test_smoothing_over_directions_keeps_the_energy_of_every_bin():
    rng = np.random.default_rng(3)
    energy = rng.random((96, 9, 50)) * (rng.random((96, 9, 50)) > 0.6)
    smoothed = smooth_over_directions(energy, grid())
    assert smoothed.shape == energy.shape
    np.testing.assert_allclose(smoothed.sum(axis=0), energy.sum(axis=0), rtol=0.01)
    # a bin empty in one direction is filled from its neighbours
    assert np.count_nonzero(smoothed) > np.count_nonzero(energy)


# ------------------------------------------------------------- in the hybrid --
import compas_acoustics as ca  # noqa: E402


def asymmetric_scene(direction=None):
    """8 x 6 x 3 m, the wall at x = 0 absorbing 0.9, the rest 0.1; source east of the receiver."""
    materials = {name: ca.AcousticMaterial(name, 0.1, 0.1) for name in ("floor", "ceiling", "left_wall", "right_wall", "front_wall")}
    materials["back_wall"] = ca.AcousticMaterial("absorber", 0.9, 0.1)
    scene = ca.Scene("asymmetric", sampling_rate=FS)
    scene.add_acoustic_objects(ca.AcousticObjectsLibrary.shoebox_room(8, 6, 3, materials=materials))
    scene.add_source(ca.Source("src", 6.0, 3.0, 1.5))
    scene.add_receiver(ca.Receiver("rec", 3.0, 3.0, 1.5, direction=direction))
    return scene


def late_level_by_x(result, a=0.15, b=0.6):
    """Mean late energy in dB of the lines facing -x minus those facing +x, in the receiver's frame."""
    x = result.core.line_directions[:, 0]
    e = np.sum(result.late_lines[int(a * FS) : int(b * FS)] ** 2, axis=0)
    return 10 * np.log10(e[x < -0.5].mean() / e[x > 0.5].mean())


def test_the_directional_envelope_quietens_the_lines_facing_the_absorbing_wall():
    kwargs = dict(tail="ray_tracing", duration=0.8, directional_absorption=False, late_field="fdn")
    plain = ca.Ambisonics(late_envelope=False, **kwargs)
    asymmetric_scene().simulate(plain)
    directional = ca.Ambisonics(late_envelope="directional", **kwargs)
    ir = next(iter(asymmetric_scene().simulate(directional).values()))
    assert ir.generation["late_envelope"] == "directional"
    (plain,), (directional,) = plain.results.values(), directional.results.values()
    assert abs(late_level_by_x(plain)) < 1.0
    assert late_level_by_x(directional) < -1.5


def test_the_omni_envelope_leaves_the_lines_alike():
    """Without the rays' shares the lines keep the network's even spread over directions."""
    omni = ca.Ambisonics(late_envelope="omni", tail="ray_tracing", duration=0.8, directional_absorption=False, late_field="fdn")
    ir = next(iter(asymmetric_scene().simulate(omni).values()))
    assert ir.generation["late_envelope"] == "omni"
    assert ir.generation["histogram_rays"] is None
    (result,) = omni.results.values()
    assert result.core.directional_envelope is None
    assert abs(late_level_by_x(result)) < 1.0


def test_the_histogram_follows_the_receivers_orientation():
    """Facing -x, the absorbing wall is in front: the quiet lines are those at +x of the receiver's frame."""
    kwargs = dict(tail="ray_tracing", duration=0.8, directional_absorption=False, late_envelope="directional", late_field="fdn")
    turned = ca.Ambisonics(**kwargs)
    asymmetric_scene(direction=[-1, 0, 0]).simulate(turned)
    (result,) = turned.results.values()
    assert late_level_by_x(result) > 1.5


def test_the_late_envelope_values():
    assert ca.Ambisonics().late_envelope == "omni"
    assert ca.Ambisonics(reference="mono").late_envelope == "omni"
    assert ca.Ambisonics(tail="eyring").late_envelope is False
    assert ca.Ambisonics(late_envelope="directional").late_envelope == "directional"
    assert ca.Binaural(late_envelope=False).late_envelope is False
    for value in (True, "on", "Omni", 1):
        with pytest.raises(ValueError, match="'directional', 'omni' or False"):
            ca.Ambisonics(late_envelope=value)
    for value in ("omni", "directional"):
        with pytest.raises(ValueError, match="late_envelope"):
            ca.Ambisonics(tail="sabine", late_envelope=value)


def test_the_band_shares_need_no_more_memory_than_the_band_split():
    """Bands are split, shaped and summed one at a time: the peak stays near the omni one's, about 12 line arrays."""
    import tracemalloc

    late = lines()
    reference = early_w(late) + 3.0 * late.sum(axis=1) / np.sqrt(96)
    histogram = histogram_of(late, 3.0)
    tracemalloc.start()
    shape_late(late, early_w(late), late.sum(axis=1) / np.sqrt(96), reference, FS, histogram=histogram)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert peak / late.nbytes < 14.0


# ------------------------------------------------- the reference's own rays --
def test_the_reference_run_sorts_its_own_rays_by_direction():
    """A copy of every microphone in the reference's run: the responses are those of a run without it, and the
    histograms those of their own run, the same rays."""
    from compas_acoustics.room.raytracing import ray_traced_run

    g = geometry(scattering=0.2)
    settings = dict(max_order=3, duration=0.5, n_rays=5000, air_absorption=True, seed=1)
    mics = np.stack([RECEIVER, [4.0, 1.5, 2.0]], axis=1)
    rotated = grid() @ np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]).T
    run = ray_traced_run(g, [pra.SoundSource(SOURCE)], mics, FS, directions=[grid(), rotated], **settings)
    plain = ray_traced_run(g, [pra.SoundSource(SOURCE)], mics, FS, **settings)
    alone = directional_histograms(g, [pra.SoundSource(SOURCE)], mics, FS, [grid(), rotated], **settings)
    assert plain.histograms is None
    for m in range(2):
        assert np.array_equal(run.responses[m][0], plain.responses[m][0])
        assert np.array_equal(run.histograms[m][0].energy, alone[m][0].energy)
        assert np.array_equal(run.histograms[m][0].directions, alone[m][0].directions)


def late_level_by_x_of_the_channels(result, a=0.15, b=0.6):
    """The late field's energy from -x minus that from +x in dB, through cardioids W -+ X / sqrt(3) (N3D: X = sqrt(3) cos)."""
    late = np.asarray(result.late, dtype=float)[int(a * FS) : int(b * FS)]
    w, x = late[:, 0], late[:, 3] * (1.0 if result.normalization == "N3D" else np.sqrt(3.0)) / np.sqrt(3.0)
    return 10 * np.log10(np.sum((w - x) ** 2) / np.sum((w + x) ** 2))


@pytest.mark.parametrize("direction, sign", [(None, -1.0), ([-1, 0, 0], 1.0)])
def test_the_moments_late_field_is_quieter_from_the_absorbing_wall(direction, sign):
    """The rays' moments put less late energy towards the absorbing wall, at -x of the room; turned to face -x, the
    receiver hears it at +x of its frame. "omni" keeps it alike on both sides (no directional absorption). The rays'
    own late field leans about 0.5 dB that way through first-order cardioids here, as the FDN's shares do."""
    kwargs = dict(tail="ray_tracing", duration=0.8, directional_absorption=False, normalization="N3D")
    directional = ca.Ambisonics(late_envelope="directional", **kwargs)
    ir = next(iter(asymmetric_scene(direction).simulate(directional).values()))
    assert ir.generation["late_field"] == "moments" and ir.generation["late_energy"] == "rays"
    (result,) = directional.results.values()
    assert sign * late_level_by_x_of_the_channels(result) > 0.3
    omni = ca.Ambisonics(late_envelope="omni", **kwargs)
    asymmetric_scene(direction).simulate(omni)
    (result,) = omni.results.values()
    assert abs(late_level_by_x_of_the_channels(result)) < 0.2
