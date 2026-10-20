import warnings

import numpy as np
import pytest

from compas_acoustics import AcousticAnalyser
from compas_acoustics.analyser import _PYRATO_PARAMETERS
from compas_acoustics.analyser import _STI_SUPPORTED
from compas_acoustics.analyser import _PARAM_NAMES
from compas_acoustics.defaults import Defaults

needs_sti = pytest.mark.skipif(not _STI_SUPPORTED, reason="STI needs pyrato >= 1.0")


def make_ir(duration=2.0, seed=0):
    # STI requires at least 1.6 s (IEC 60268-16:2020, Section 6.2)
    fs = Defaults().SAMPLING_RATE
    rng = np.random.default_rng(seed)
    t = np.arange(int(duration * fs)) / fs
    ir = rng.standard_normal(t.size) * np.exp(-t / 0.3)
    ir[0] = 5.0
    return ir


def make_foa(omni, seed=1):
    rng = np.random.default_rng(seed)
    directional = rng.standard_normal((omni.size, 3)) * 0.3
    return np.column_stack([omni, directional])


def test_mono_ir_is_stored_unchanged():
    ir = make_ir()
    analyser = AcousticAnalyser(ir)
    assert analyser.ir.shape == ir.shape
    assert np.array_equal(analyser.ir, ir)


def test_mono_ir_does_not_warn():
    ir = make_ir()
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        AcousticAnalyser(ir)
    assert record == []


def test_multichannel_ir_is_reduced_to_first_channel():
    omni = make_ir()
    foa = make_foa(omni)
    with pytest.warns(UserWarning, match="4 channels"):
        analyser = AcousticAnalyser(foa)
    assert analyser.ir.shape == omni.shape
    assert np.array_equal(analyser.ir, omni)


def test_single_column_ir_is_flattened_without_warning():
    ir = make_ir()
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        analyser = AcousticAnalyser(ir.reshape(-1, 1))
    assert record == []
    assert analyser.ir.shape == ir.shape


def test_more_than_two_dimensions_raises():
    with pytest.raises(ValueError, match="1D or 2D"):
        AcousticAnalyser(np.zeros((10, 2, 2)))


def test_multichannel_analysis_matches_first_channel_alone():
    omni = make_ir()
    foa = make_foa(omni)

    with pytest.warns(UserWarning):
        multichannel = AcousticAnalyser(foa).analyse()
    mono = AcousticAnalyser(omni).analyse()

    assert multichannel.keys() == mono.keys()
    for key in mono:
        np.testing.assert_allclose(
            np.asarray(multichannel[key], dtype=float),
            np.asarray(mono[key], dtype=float),
            equal_nan=True,
            err_msg=f"{key} differs between multichannel and mono input",
        )


@needs_sti
def test_sti_is_finite_for_ir_shorter_than_iec_minimum():
    analyser = AcousticAnalyser(make_ir(duration=1.0))
    analyser.analyse()
    assert np.isfinite(analyser.STI_overall)
    assert np.all(np.isfinite(analyser.STI))


@needs_sti
def test_sti_of_short_ir_matches_zero_padded_ir():
    fs = Defaults().SAMPLING_RATE
    short = make_ir(duration=1.0)
    padded = np.pad(short, (0, int(1.6 * fs) - short.size))
    np.testing.assert_allclose(
        AcousticAnalyser(short).speech_transmission_index()["mti"],
        AcousticAnalyser(padded).speech_transmission_index()["mti"],
    )


@needs_sti
def test_sti_failure_warns_instead_of_passing_silently(monkeypatch):
    import pyrato

    def fail(*args, **kwargs):
        raise ValueError("boom")

    monkeypatch.setattr(pyrato.parameters, "modulation_transfer_function", fail)
    with pytest.warns(UserWarning, match="STI computation failed: boom"):
        result = AcousticAnalyser(make_ir()).speech_transmission_index()
    assert np.isnan(result["sti"])


# Stage support and strength

def make_stage_ir(early_db=-12.0, late_db=-15.0, fs=48000, seed=1):
    """Direct sound at 5 ms plus white reflections with known energies relative to it."""
    rng = np.random.default_rng(seed)
    ir = np.zeros(int(1.3 * fs))
    onset = int(0.005 * fs)
    ir[onset] = 1.0
    for (start, stop), level_db in (((0.020, 0.100), early_db), ((0.100, 1.000), late_db)):
        window = slice(onset + int(start * fs), onset + int(stop * fs))
        noise = rng.standard_normal(window.stop - window.start)
        ir[window] = noise * np.sqrt(10 ** (level_db / 10) / np.sum(noise ** 2))
    return ir


def test_stage_support_recovers_known_energy_ratios():
    analyser = AcousticAnalyser(make_stage_ir(-12.0, -15.0))
    result = analyser.stage_support()
    assert result["freqs"] == list(analyser.frequency_bands)
    assert result["ST_early_overall"] == pytest.approx(-12.0, abs=0.5)
    assert result["ST_late_overall"] == pytest.approx(-15.0, abs=0.5)
    # Windowing before filtering keeps the low bands right too
    assert all(abs(v + 12.0) < 1.5 for v in result["ST_early"])


def test_stage_support_warns_when_late_window_is_truncated():
    with pytest.warns(UserWarning, match="ST_late"):
        AcousticAnalyser(make_stage_ir()[:48000 // 2]).stage_support()


@pytest.mark.skipif(not _PYRATO_PARAMETERS, reason="G needs pyrato >= 1.0")
def test_strength_matches_inverse_distance_law():
    from compas_acoustics import ImpulseResponse
    from compas_acoustics import Receiver
    from compas_acoustics import Source

    fs = 48000
    source = Source("s", 0, 0, 0)
    near, reference = np.zeros(fs), np.zeros(fs)
    near[100] = 1.0
    reference[1500] = 0.1
    # Normalized ImpulseResponse objects are evaluated at their raw level
    near_ir = ImpulseResponse("near", source, Receiver("r", 1, 0, 0), near, fs)
    reference_ir = ImpulseResponse("ref", source, Receiver("f", 10, 0, 0), reference, fs)

    result = AcousticAnalyser(near_ir).sound_strength(reference_ir)
    np.testing.assert_allclose(result["G"], 20.0, atol=0.01)
    assert result["G_overall"] == pytest.approx(20.0, abs=0.01)


def test_strength_rejects_mismatched_sampling_rates():
    from compas_acoustics import ImpulseResponse
    from compas_acoustics import Receiver
    from compas_acoustics import Source

    reference = ImpulseResponse("ref", Source("s", 0, 0, 0), Receiver("f", 10, 0, 0), np.ones(10), 44100)
    with pytest.raises(ValueError, match="sample rate"):
        AcousticAnalyser(make_ir()).sound_strength(reference)


def test_impulse_response_object_sets_sampling_rate_and_raw_level():
    from compas_acoustics import ImpulseResponse
    from compas_acoustics import Receiver
    from compas_acoustics import Source

    raw = make_ir() * 0.3
    ir = ImpulseResponse("ir", Source("s", 0, 0, 0), Receiver("r", 1, 0, 0), raw, 44100)
    analyser = AcousticAnalyser(ir)
    assert analyser.sampling_rate == 44100
    np.testing.assert_allclose(analyser.ir, raw)


def test_clarity_definition_and_center_time_are_exact_in_every_band():
    # Direct sound and one reflection of half amplitude 300 ms later: C50 = C80 = 10 log10(1 / 0.25),
    # D50 = 80 %, Ts = 0.3 s * 0.25 / 1.25 = 60 ms. The causal octave filter delayed the low bands'
    # direct sound past the time limits (e.g. Ts was about 35 ms too long at 125 Hz).
    fs = 48000
    ir = np.zeros(2 * fs)
    ir[1000] = 1.0
    ir[1000 + int(0.3 * fs)] = 0.5
    analyser = AcousticAnalyser(ir)
    analyser.sampling_rate = fs

    clarity = analyser.clarity_definition()
    np.testing.assert_allclose(clarity["C50"], 10 * np.log10(4), atol=0.1)
    np.testing.assert_allclose(clarity["C80"], 10 * np.log10(4), atol=0.1)
    np.testing.assert_allclose(clarity["D50"], 80.0, atol=0.5)
    np.testing.assert_allclose(analyser.center_time()["Ts"], 0.060, atol=0.001)


def exponential_decay(reverberation_time, seconds, seed, fs=48000):
    """White noise decaying 60 dB over the reverberation time; no direct sound."""
    t = np.arange(int(seconds * fs)) / fs
    return np.random.default_rng(seed).standard_normal(t.size) * 10 ** (-3 * t / reverberation_time)


def test_reverberation_times_recover_an_exponential_decay_in_every_band():
    # EDT was about 15 % too long at 125 Hz when the decay started on the causal filter's build-up
    results = []
    for seed in range(3):
        analyser = AcousticAnalyser(exponential_decay(1.2, 3.0, seed))
        analyser.sampling_rate = 48000
        results.append(analyser.reverberation_times())
    for key in ("EDT", "T20", "T30"):
        np.testing.assert_allclose(np.mean([r[key] for r in results], axis=0), 1.2, rtol=0.08, err_msg=key)


@needs_sti
def test_modulation_transfer_matches_the_analytic_exponential_decay():
    reverberation_time = 1.2
    k = 6 * np.log(10) / reverberation_time
    modulation_frequencies = np.array([0.63, 0.8, 1, 1.25, 1.6, 2, 2.5, 3.15, 4, 5, 6.3, 8, 10, 12.5])
    m = 1 / np.sqrt(1 + (2 * np.pi * modulation_frequencies / k) ** 2)
    mti = np.mean(np.clip((10 * np.log10(m / (1 - m)) + 15) / 30, 0, 1))
    # IEC 60268-16 male weights; with equal MTIs, STI = MTI * (sum(alpha) - sum(beta))
    sti = mti * (1.381 - 0.381)

    results = []
    for seed in range(3):
        analyser = AcousticAnalyser(exponential_decay(reverberation_time, 3.0, seed))
        analyser.sampling_rate = 48000
        results.append(analyser.speech_transmission_index())
    # Average realisations: a single noise tail scatters by about 0.05 in the 125 Hz band
    np.testing.assert_allclose(np.mean([r["mti"] for r in results], axis=0), mti, atol=0.03)
    assert np.mean([r["sti"] for r in results]) == pytest.approx(sti, abs=0.02)


def test_energy_parameters_are_exact_for_discrete_reflections():
    # Direct sound 1, reflections 0.25 at 60 ms and 0.1 at 300 ms (energies 1, 0.0625, 0.01)
    fs = 48000
    ir = np.zeros(int(1.6 * fs))
    ir[2000] = 1.0
    ir[2000 + int(0.06 * fs)] = 0.25
    ir[2000 + int(0.30 * fs)] = 0.1
    analyser = AcousticAnalyser(ir)
    analyser.sampling_rate = fs
    clarity, drr, ts = analyser.clarity_definition(), analyser.direct_to_reverberant_ratio(), analyser.center_time()
    support = analyser.stage_support()

    np.testing.assert_allclose(clarity["C50"], 10 * np.log10(1 / 0.0725), atol=0.1)
    np.testing.assert_allclose(clarity["C80"], 10 * np.log10(1.0625 / 0.01), atol=0.1)
    np.testing.assert_allclose(clarity["D50"], 100 / 1.0725, atol=0.5)
    np.testing.assert_allclose(drr["DRR"], 10 * np.log10(1 / 0.0725), atol=0.1)
    np.testing.assert_allclose(ts["Ts"], (0.06 * 0.0625 + 0.3 * 0.01) / 1.0725, atol=0.002)
    np.testing.assert_allclose(support["ST_early"], 10 * np.log10(0.0625), atol=0.1)
    np.testing.assert_allclose(support["ST_late"], -20.0, atol=0.1)


def test_a_package_scale_response_is_analysed_from_its_geometric_onset():
    from compas_acoustics import ImpulseResponse, Receiver, Source

    fs = 48000
    ir = make_stage_ir(-12.0, -15.0)  # direct sound at 5 ms
    ir = np.concatenate([np.zeros(4560), ir])  # direct sound now at 4800 samples = 34.3 m
    clean = ir.copy()
    ir[100] = 50.0  # an early bin louder than the direct sound
    package = ImpulseResponse("k", Source("s", 0, 0, 0), Receiver("r", 34.3, 0, 0), ir, fs,
                              generation={"scale": "package"})
    support = AcousticAnalyser(package).stage_support()
    reference = AcousticAnalyser(clean).stage_support()
    assert support["ST_early_overall"] == pytest.approx(reference["ST_early_overall"], abs=0.1)


def test_the_stored_onset_follows_the_response():
    from compas_acoustics import ImpulseResponse, Receiver, Source

    data = np.zeros(2000)
    data[480] = 1.0
    data[10] = 5.0
    package = ImpulseResponse("k", Source("s", 0, 0, 0), Receiver("r", 3.43, 0, 0), data, 48000, generation={"scale": "package"})
    analyser = AcousticAnalyser(package)
    assert analyser.onset == 480
    analyser.ir = data
    assert analyser.onset is None


@pytest.mark.skipif(not _PYRATO_PARAMETERS, reason="G needs pyrato >= 1.0")
def test_g_is_a_main_parameter_and_needs_no_reference_on_the_package_scale():
    from compas_acoustics import ImpulseResponse
    from compas_acoustics import Receiver
    from compas_acoustics import Source

    fs = 48000
    source = Source("s", 0, 0, 0)
    direct = np.zeros(fs)
    direct[int(round(1.0 / 343.0 * fs))] = 1.0 / (4.0 * np.pi * 1.0)
    ir = ImpulseResponse("near", source, Receiver("r", 1, 0, 0), direct, fs,
                         generation={"scale": "package"})
    result = AcousticAnalyser(ir).analyse()
    assert "G" in _PARAM_NAMES
    np.testing.assert_allclose(result["G"], 20.0, atol=0.01)
    assert result["G_overall"] == pytest.approx(20.0, abs=0.01)


def test_g_is_nan_without_the_package_scale():
    result = AcousticAnalyser(make_ir()).analyse()
    assert np.all(np.isnan(result["G"])) and np.isnan(result["G_overall"])
    with pytest.raises(ValueError, match="package-scale"):
        AcousticAnalyser(make_ir()).sound_strength()


@pytest.mark.skipif(not _PYRATO_PARAMETERS, reason="compares with pyrato >= 1.0")
def test_center_time_without_pyrato_parameters_matches_pyrato(monkeypatch):
    import compas_acoustics.analyser as analyser_module

    ir = make_ir()
    with_pyrato = AcousticAnalyser(ir).center_time()["Ts"]
    monkeypatch.setattr(analyser_module, "_PYRATO_PARAMETERS", False)
    fallback = AcousticAnalyser(ir).center_time()["Ts"]
    np.testing.assert_allclose(fallback, with_pyrato, atol=1e-3)


def test_strength_is_nan_with_a_warning_without_pyrato_parameters(monkeypatch):
    import compas_acoustics.analyser as analyser_module

    monkeypatch.setattr(analyser_module, "_PYRATO_PARAMETERS", False)
    ir = make_ir()
    with pytest.warns(UserWarning, match="Sound strength G needs pyrato"):
        result = AcousticAnalyser(ir).sound_strength(ir)
    assert np.isnan(result["G_overall"])


@pytest.mark.skipif(_PYRATO_PARAMETERS, reason="the fallback is for pyrato < 1.0")
def test_analyse_gives_nan_g_without_a_second_warning_on_pyrato_below_1():
    import warnings

    from compas_acoustics import ImpulseResponse
    from compas_acoustics import Receiver
    from compas_acoustics import Source

    data = np.zeros(48000)
    data[140] = 1.0 / (4.0 * np.pi)
    package = ImpulseResponse("ir", Source("s", 0, 0, 0), Receiver("r", 1, 0, 0), data, 48000, generation={"scale": "package"})
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        result = AcousticAnalyser(package).analyse()
    assert not any("Sound strength G" in str(w.message) for w in record)
    assert np.isnan(result["G_overall"])


def test_drr_counts_the_first_5_ms_after_the_direct_sound_as_direct():
    # Haapaniemi and Lokki (2018), eq. 1: a reflection 3 ms after the direct sound is direct, one 6 ms after is not.
    # Their band-pass is applied before the window, so a reflection right before 5 ms would leave part of its
    # filtered response after it: 3 ms keeps it whole.
    fs = 48000
    ir = np.zeros(fs)
    n0 = 4800
    ir[n0] = 1.0
    ir[n0 + int(0.003 * fs)] = 0.5
    ir[n0 + int(0.006 * fs)] = 0.25
    ir[n0 + int(0.04 * fs)] = 0.25
    # (1 + 0.25) / (0.0625 + 0.0625) is 10 dB; a window of +-2.5 ms around the direct sound gave 4.3 dB
    assert AcousticAnalyser(ir).direct_to_reverberant_ratio()["DRR_overall"] == pytest.approx(10.0, abs=0.1)


def test_drr_holds_a_band_limited_direct_sound_whole():
    # A direct sound between two samples spreads over both; the geometric onset rounds to the later one.
    # Truth: the direct sound and the reflections, each filtered on its own at full length.
    from scipy.signal import butter
    from scipy.signal import sosfilt

    from compas_acoustics import ImpulseResponse
    from compas_acoustics import Receiver
    from compas_acoustics import Source

    fs = 48000
    arrival = 4800.6
    n = np.arange(fs)
    direct = np.sinc(n - arrival) * np.exp(-0.5 * ((n - arrival) / 2000.0) ** 2)
    late = np.zeros(fs)
    late[[4800 + 288, 4800 + 1000, 4800 + 5000]] = [0.3, -0.2, 0.1]
    receiver = Receiver("r", arrival / fs * Defaults().DEFAULT_SPEED_OF_SOUND, 0, 0)
    ir = ImpulseResponse("ir", Source("s", 0, 0, 0), receiver, direct + late, fs, generation={"scale": "package"})
    assert ir.direct_sound_index() == 4801
    analyser = AcousticAnalyser(ir)
    drr = analyser.direct_to_reverberant_ratio()

    def band_energy(x, f):
        return np.sum(analyser._filter_impulse_response(np.concatenate([x, np.zeros(fs // 2)]), f) ** 2)

    truth = [10 * np.log10(band_energy(direct, f) / band_energy(late, f)) for f in analyser.frequency_bands]
    np.testing.assert_allclose(drr["DRR"], truth, atol=0.05)
    sos = butter(2, [700, 4000], "bandpass", fs=fs, output="sos")
    overall = 10 * np.log10(np.sum(sosfilt(sos, direct) ** 2) / np.sum(sosfilt(sos, late) ** 2))
    assert drr["DRR_overall"] == pytest.approx(overall, abs=0.05)


def test_the_single_number_drr_is_the_700_to_4000_hz_band():
    analyser = AcousticAnalyser(make_ir())
    result = analyser.analyse()
    assert result["DRR_overall"] == analyser.direct_to_reverberant_ratio()["DRR_overall"]
    assert analyser.DRR_overall == result["DRR_overall"]


def onset_of(ir):
    return AcousticAnalyser(np.asarray(ir, dtype=float))._direct_sound_onset(np.asarray(ir, dtype=float))


def test_the_start_is_the_last_sample_below_20_db_before_the_rise():
    # ita_start_IR (ITA Toolbox, ISO 3382-1): the sample at -26 dB is the start, not the one at -14 dB after it
    ir = np.zeros(4800)
    ir[[1000, 1001, 1002]] = [0.05, 0.2, 1.0]
    assert onset_of(ir) == 1000


def test_an_earlier_arrival_within_14_db_moves_the_start_back():
    ir = np.zeros(4800)
    ir[[500, 1000]] = [0.3, 1.0]  # -10.5 dB, before the peak
    assert onset_of(ir) == 499


def test_an_earlier_arrival_between_20_and_14_db_does_not():
    ir = np.zeros(4800)
    ir[[500, 1000]] = [0.15, 1.0]  # -16.5 dB
    assert onset_of(ir) == 999


def test_a_response_less_than_20_db_above_its_noise_has_no_start():
    ir = np.random.default_rng(0).standard_normal(4800) * 0.2
    ir[1000] = 1.0
    assert onset_of(ir) == 0


def test_a_blocked_direct_sound_is_found_as_ita_finds_it():
    # Package scale, direct path blocked: nothing arrives at the geometric sample (4800). Like ita_start_IR,
    # the analysis starts at the first arrival; the geometric window held nothing and gave a DRR of about -110 dB.
    from compas_acoustics import ImpulseResponse
    from compas_acoustics import Receiver
    from compas_acoustics import Source

    fs = 48000
    rng = np.random.default_rng(3)
    ir = np.zeros(2 * fs)
    ir[[8000, 9000]] = [0.5, 1.0]
    tail = np.arange(9500, ir.size)
    ir[tail] = rng.standard_normal(tail.size) * 0.05 * np.exp(-(tail - 9500) / fs / 0.15)
    package = ImpulseResponse("k", Source("s", 0, 0, 0), Receiver("r", 34.3, 0, 0), ir, fs, generation={"scale": "package"})
    assert package.direct_sound_index() == 4800
    blocked, plain = AcousticAnalyser(package), AcousticAnalyser(ir)
    assert blocked._direct_sound_onset(blocked.ir) == plain._direct_sound_onset(plain.ir) == 7999
    assert blocked.direct_to_reverberant_ratio() == plain.direct_to_reverberant_ratio()


def test_a_ray_traced_tail_that_dies_into_zeros_is_evaluated_in_every_band():
    # pyroomacoustics stops each ray at energy_thres: past about -60 dB the decay steepens, falls to around
    # -150 dB and ends in exact zeros. Schroeder's integral is exact for a response without noise. (On the
    # 408-face office of examples/notebooks/03_whats_new, whose ray-traced tail is also sparse, pyrato's
    # Lundeby search found no intersection and left six of seven bands NaN; this smooth tail does not trip it.)
    fs = 48000
    t = np.arange(int(2.0 * fs)) / fs
    results = []
    for seed in range(3):
        decay = exponential_decay(1.2, 2.0, seed) * 10 ** (-np.clip(t - 0.8, 0, None) ** 2 * 60)
        analyser = AcousticAnalyser(np.concatenate([decay, np.zeros(fs)]))
        analyser.sampling_rate = fs
        results.append(analyser.reverberation_times())
    for key in ("EDT", "T20", "T30"):
        values = np.array([r[key] for r in results])
        assert np.all(np.isfinite(values)), key
        np.testing.assert_allclose(values.mean(axis=0), 1.2, rtol=0.08, err_msg=key)


def test_trailing_zeros_do_not_hide_the_decay():
    # A response padded with exact zeros: the octave filters' ringing decays into them without a noise floor,
    # and pyrato's Lundeby search found none. The ITA Toolbox cuts trailing zeros before its noise detection.
    results = []
    for seed in range(3):
        padded = np.concatenate([exponential_decay(1.2, 2.0, seed), np.zeros(48000)])
        analyser = AcousticAnalyser(padded)
        analyser.sampling_rate = 48000
        results.append(analyser.reverberation_times())
    for key in ("EDT", "T20", "T30"):
        np.testing.assert_allclose(np.mean([r[key] for r in results], axis=0), 1.2, rtol=0.08, err_msg=key)


@pytest.mark.parametrize("name", ["C50", "C80", "D50", "Ts", "EDT", "T20", "T30"])
def test_single_numbers_average_500_and_1000_hz(name):
    # ISO 3382-1:2009 Table A.1; C50 is a function of D50 and takes its bands
    r = AcousticAnalyser(make_ir()).analyse()
    assert r["freqs"][2:4] == [500, 1000]
    assert r[f"{name}_overall"] == pytest.approx(np.nanmean([r[name][2], r[name][3]]), abs=0.01)


def test_the_time_response_is_in_db_re_the_peak_of_all_channels_floored_below_it():
    from compas_acoustics.analyser import time_response

    ir = np.zeros((4, 2))
    ir[0] = [1.0, 0.1]
    ir[2] = [-0.5, 0.0]
    times, level = time_response(ir, 4)
    np.testing.assert_allclose(times, [0, 0.25, 0.5, 0.75])
    np.testing.assert_allclose(level[0], [0.0, -100.0, 20 * np.log10(0.5), -100.0])
    np.testing.assert_allclose(level[1, 0], -20.0)
    _, amplitude = time_response(ir, 4, db=False)
    np.testing.assert_array_equal(amplitude, ir.T)
    assert time_response(np.zeros(3), 4)[1].tolist() == [[-100.0] * 3]


def test_the_frequency_response_of_an_impulse_is_flat_within_the_range():
    from compas_acoustics.analyser import frequency_response

    ir = np.zeros(4800)
    ir[0] = 1.0
    frequencies, level = frequency_response(ir, 48000)
    assert frequencies[0] >= 20 and frequencies[-1] <= 20000 and level.shape == (1, len(frequencies))
    np.testing.assert_allclose(level, 0.0, atol=1e-6)
    _, both = frequency_response(np.column_stack([ir, 0.5 * ir]), 48000)
    np.testing.assert_allclose(both[1], 20 * np.log10(0.5), atol=1e-6)


def test_the_time_response_ends_with_its_decay():
    from compas_acoustics.analyser import time_response

    fs = 1000
    ir = np.exp(-np.arange(10 * fs) / fs * 6.9)  # 60 dB per second in level, 10 s long
    times, _ = time_response(ir, fs, trim_db=60)
    # the Schroeder curve of an exponential decay falls as its level: 60 dB at 1 s, plus 5 %
    assert times[-1] == pytest.approx(1.05, abs=0.01)
    assert len(time_response(ir, fs)[0]) == len(ir)
