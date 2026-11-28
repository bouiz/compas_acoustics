import warnings

import numpy as np
import pytest

from compas_acoustics import Analyser
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
    analyser = Analyser(ir)
    assert analyser.ir.shape == ir.shape
    assert np.array_equal(analyser.ir, ir)


def test_mono_ir_does_not_warn():
    ir = make_ir()
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        Analyser(ir)
    assert record == []


def test_multichannel_ir_is_reduced_to_first_channel():
    omni = make_ir()
    foa = make_foa(omni)
    with pytest.warns(UserWarning, match="4 channels"):
        analyser = Analyser(foa)
    assert analyser.ir.shape == omni.shape
    assert np.array_equal(analyser.ir, omni)


def test_single_column_ir_is_flattened_without_warning():
    ir = make_ir()
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        analyser = Analyser(ir.reshape(-1, 1))
    assert record == []
    assert analyser.ir.shape == ir.shape


def test_more_than_two_dimensions_raises():
    with pytest.raises(ValueError, match="1D or 2D"):
        Analyser(np.zeros((10, 2, 2)))


def test_multichannel_analysis_matches_first_channel_alone():
    omni = make_ir()
    foa = make_foa(omni)

    with pytest.warns(UserWarning):
        multichannel = Analyser(foa).analyse()
    mono = Analyser(omni).analyse()

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
    analyser = Analyser(make_ir(duration=1.0))
    analyser.analyse()
    assert np.isfinite(analyser.analysis_data["STI_overall"])
    assert np.all(np.isfinite(analyser.analysis_data["STI"]))


@needs_sti
def test_sti_of_short_ir_matches_zero_padded_ir():
    fs = Defaults().SAMPLING_RATE
    short = make_ir(duration=1.0)
    padded = np.pad(short, (0, int(1.6 * fs) - short.size))
    np.testing.assert_allclose(
        Analyser(short).speech_transmission_index()["mti"],
        Analyser(padded).speech_transmission_index()["mti"],
    )


@needs_sti
def test_sti_failure_warns_instead_of_passing_silently(monkeypatch):
    def fail(*args, **kwargs):
        raise ValueError("boom")

    monkeypatch.setattr(Analyser, "_modulation_transfer_function", fail)
    with pytest.warns(UserWarning, match="STI computation failed: boom"):
        result = Analyser(make_ir()).speech_transmission_index()
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
        ir[window] = noise * np.sqrt(10 ** (level_db / 10) / np.sum(noise**2))
    return ir


def test_stage_support_recovers_known_energy_ratios():
    analyser = Analyser(make_stage_ir(-12.0, -15.0))
    result = analyser.stage_support()
    assert result["freqs"] == list(analyser.frequency_bands)
    assert result["ST_early_overall"] == pytest.approx(-12.0, abs=0.5)
    assert result["ST_late_overall"] == pytest.approx(-15.0, abs=0.5)
    # Windowing before filtering keeps the low bands right too
    assert all(abs(v + 12.0) < 1.5 for v in result["ST_early"])


def test_stage_support_warns_when_late_window_is_truncated():
    with pytest.warns(UserWarning, match="ST_late"):
        Analyser(make_stage_ir()[: 48000 // 2]).stage_support()


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

    result = Analyser(near_ir).sound_strength(reference_ir)
    np.testing.assert_allclose(result["G"], 20.0, atol=0.01)
    assert result["G_overall"] == pytest.approx(20.0, abs=0.01)


def test_strength_rejects_mismatched_sampling_rates():
    from compas_acoustics import ImpulseResponse
    from compas_acoustics import Receiver
    from compas_acoustics import Source

    reference = ImpulseResponse("ref", Source("s", 0, 0, 0), Receiver("f", 10, 0, 0), np.ones(10), 44100)
    with pytest.raises(ValueError, match="sample rate"):
        Analyser(make_ir()).sound_strength(reference)


def test_impulse_response_object_sets_sampling_rate_and_raw_level():
    from compas_acoustics import ImpulseResponse
    from compas_acoustics import Receiver
    from compas_acoustics import Source

    raw = make_ir() * 0.3
    ir = ImpulseResponse("ir", Source("s", 0, 0, 0), Receiver("r", 1, 0, 0), raw, 44100)
    analyser = Analyser(ir)
    assert analyser.sampling_rate == 44100
    np.testing.assert_allclose(analyser.ir, raw)


def test_clarity_definition_and_center_time_are_exact_in_every_band():
    # Direct sound and one reflection of half amplitude 300 ms later: C50 = C80 = 10 log10(1 / 0.25),
    # D50 = 80 %, Ts = 0.3 s * 0.25 / 1.25 = 60 ms. The low bands catch a causal octave filter's delay of
    # the direct sound past the time limits.
    fs = 48000
    ir = np.zeros(2 * fs)
    ir[1000] = 1.0
    ir[1000 + int(0.3 * fs)] = 0.5
    analyser = Analyser(ir)
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
        analyser = Analyser(exponential_decay(reverberation_time, 3.0, seed))
        analyser.sampling_rate = 48000
        results.append(analyser.speech_transmission_index())
    # Average realisations: a single noise tail scatters beyond the tolerance in the 125 Hz band
    np.testing.assert_allclose(np.mean([r["mti"] for r in results], axis=0), mti, atol=0.03)
    assert np.mean([r["sti"] for r in results]) == pytest.approx(sti, abs=0.02)


def test_energy_parameters_are_exact_for_discrete_reflections():
    # Direct sound 1, reflections 0.25 at 60 ms and 0.1 at 300 ms (energies 1, 0.0625, 0.01)
    fs = 48000
    ir = np.zeros(int(1.6 * fs))
    ir[2000] = 1.0
    ir[2000 + int(0.06 * fs)] = 0.25
    ir[2000 + int(0.30 * fs)] = 0.1
    analyser = Analyser(ir)
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


def test_the_stored_onset_follows_the_response():
    data = np.zeros(2000)
    data[[10, 480]] = [5.0, 1.0]
    analyser = Analyser(data)
    assert analyser.onset == 9
    data = np.roll(data, 100)
    analyser.ir = data
    assert analyser.onset == 109


@pytest.mark.skipif(not _PYRATO_PARAMETERS, reason="G needs pyrato >= 1.0")
def test_g_is_a_main_parameter_and_needs_no_reference_on_the_package_scale():
    from compas_acoustics import ImpulseResponse
    from compas_acoustics import Receiver
    from compas_acoustics import Source

    fs = 48000
    source = Source("s", 0, 0, 0)
    direct = np.zeros(fs)
    direct[int(round(1.0 / 343.0 * fs))] = 1.0 / (4.0 * np.pi * 1.0)
    ir = ImpulseResponse("near", source, Receiver("r", 1, 0, 0), direct, fs, generation={"scale": "package"})
    result = Analyser(ir).analyse()
    assert "G" in _PARAM_NAMES
    np.testing.assert_allclose(result["G"], 20.0, atol=0.01)
    assert result["G_overall"] == pytest.approx(20.0, abs=0.01)


@pytest.mark.skipif(not _PYRATO_PARAMETERS, reason="compares with pyrato >= 1.0")
def test_center_time_without_pyrato_parameters_matches_pyrato(monkeypatch):
    import compas_acoustics.analyser as analyser_module

    ir = make_ir()
    with_pyrato = Analyser(ir).center_time()["Ts"]
    monkeypatch.setattr(analyser_module, "_PYRATO_PARAMETERS", False)
    fallback = Analyser(ir).center_time()["Ts"]
    np.testing.assert_allclose(fallback, with_pyrato, atol=1e-3)


def test_strength_is_nan_with_a_warning_without_pyrato_parameters(monkeypatch):
    import compas_acoustics.analyser as analyser_module

    monkeypatch.setattr(analyser_module, "_PYRATO_PARAMETERS", False)
    ir = make_ir()
    with pytest.warns(UserWarning, match="Sound strength G needs pyrato"):
        result = Analyser(ir).sound_strength(ir)
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
        result = Analyser(package).analyse()
    assert not any("Sound strength G" in str(w.message) for w in record)
    assert np.isnan(result["G_overall"])


def test_drr_counts_the_first_5_ms_after_the_direct_sound_as_direct():
    # A reflection 3 ms after the direct sound is direct, one 6 ms after is not.
    # Their band-pass is applied before the window, so a reflection right before 5 ms would leave part of its
    # filtered response after it: 3 ms keeps it whole.
    fs = 48000
    ir = np.zeros(fs)
    n0 = 4800
    ir[n0] = 1.0
    ir[n0 + int(0.003 * fs)] = 0.5
    ir[n0 + int(0.006 * fs)] = 0.25
    ir[n0 + int(0.04 * fs)] = 0.25
    # (1 + 0.25) / (0.0625 + 0.0625) is 10 dB
    assert Analyser(ir).direct_to_reverberant_ratio()["DRR_overall"] == pytest.approx(10.0, abs=0.1)


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
    analyser = Analyser(ir)
    drr = analyser.direct_to_reverberant_ratio()

    def band_energy(x, f):
        return np.sum(analyser._filter_impulse_response(np.concatenate([x, np.zeros(fs // 2)]), f) ** 2)

    truth = [10 * np.log10(band_energy(direct, f) / band_energy(late, f)) for f in analyser.frequency_bands]
    np.testing.assert_allclose(drr["DRR"], truth, atol=0.05)
    sos = butter(2, [700, 4000], "bandpass", fs=fs, output="sos")
    overall = 10 * np.log10(np.sum(sosfilt(sos, direct) ** 2) / np.sum(sosfilt(sos, late) ** 2))
    assert drr["DRR_overall"] == pytest.approx(overall, abs=0.05)


def test_the_start_is_the_last_sample_below_20_db_before_the_rise():
    # pyfar: the sample at -26 dB is the start, not the one at -14 dB after it
    ir = np.zeros(4800)
    ir[[1000, 1001, 1002]] = [0.05, 0.2, 1.0]
    assert Analyser(ir).onset == 1000


@pytest.mark.parametrize("level", [0.15])  # -16.5 dB
def test_an_earlier_arrival_within_20_db_moves_the_start_back(level):
    ir = np.zeros(4800)
    ir[[500, 1000]] = [level, 1.0]
    assert Analyser(ir).onset == 499


def test_a_response_less_than_20_db_above_its_noise_warns():
    ir = np.random.default_rng(0).standard_normal(4800) * 0.2
    ir[1000] = 1.0
    with pytest.warns(UserWarning, match="SNR"):
        Analyser(ir)


def test_the_onset_of_an_occluded_direct_sound_is_pyfars_start_sample():
    # Package scale, direct path blocked: nothing arrives at the geometric sample (4786); the analysis starts at the
    # first arrival, as for the same response without geometry.
    import pyfar as pf

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
    assert package.direct_sound_index() == 4786  # 34.3 m at 343.98 m/s
    blocked, plain = Analyser(package), Analyser(ir)
    assert blocked.onset == plain.onset == int(np.squeeze(pf.dsp.find_impulse_response_start(pf.Signal(ir, fs)))) == 7999
    assert blocked.direct_to_reverberant_ratio() == plain.direct_to_reverberant_ratio()


def test_a_ray_traced_tail_that_dies_into_zeros_is_evaluated_in_every_band():
    # The ray tracer stops each ray at an energy threshold: past about -60 dB the decay steepens, falls to
    # around -150 dB and ends in exact zeros. Schroeder's integral is exact for a response without noise.
    fs = 48000
    t = np.arange(int(2.0 * fs)) / fs
    results = []
    for seed in range(3):
        decay = exponential_decay(1.2, 2.0, seed) * 10 ** (-(np.clip(t - 0.8, 0, None) ** 2) * 60)
        analyser = Analyser(np.concatenate([decay, np.zeros(fs)]))
        analyser.sampling_rate = fs
        results.append(analyser.reverberation_times())
    for key in ("EDT", "T20", "T30"):
        values = np.array([r[key] for r in results])
        assert np.all(np.isfinite(values)), key
        np.testing.assert_allclose(values.mean(axis=0), 1.2, rtol=0.08, err_msg=key)


def test_trailing_zeros_do_not_hide_the_decay():
    # A response padded with exact zeros: the octave filters' ringing into them would steepen the end of the
    # Schroeder integral, so trailing zeros are cut first.
    results = []
    for seed in range(3):
        padded = np.concatenate([exponential_decay(1.2, 2.0, seed), np.zeros(48000)])
        analyser = Analyser(padded)
        analyser.sampling_rate = 48000
        results.append(analyser.reverberation_times())
    for key in ("EDT", "T20", "T30"):
        np.testing.assert_allclose(np.mean([r[key] for r in results], axis=0), 1.2, rtol=0.08, err_msg=key)


@needs_sti
def test_sti_equals_pyrato_indirect_method():
    """The MTF reduced once gives what pyrato's speech_transmission_index_indirect gives, exactly."""
    import pyfar as pf
    import pyrato

    for seed in range(3):
        ir = make_ir(seed=seed)
        analyser = Analyser(ir)
        sti = analyser.speech_transmission_index()["sti"]
        expected = float(np.squeeze(pyrato.parameters.speech_transmission_index_indirect(pf.Signal(ir, analyser.sampling_rate), snr=np.inf)))
        assert sti == round(expected, 4)


def test_the_octave_bank_is_designed_once_per_rate(monkeypatch):
    import pyfar as pf

    from compas_acoustics import analyser as analysis

    designs = []
    design = pf.dsp.filter.fractional_octave_bands

    def counted(*args, **kwargs):
        designs.append(kwargs)
        return design(*args, **kwargs)

    monkeypatch.setattr(pf.dsp.filter, "fractional_octave_bands", counted)
    analysis._octave_bank.cache_clear()
    analysis._octave_bank_sos.cache_clear()
    for seed in range(2):
        analyser = Analyser(make_ir(seed=seed))
        analyser.reverberation_times()
        analyser.clarity_definition()
        analyser.speech_transmission_index()
    # the causal and the zero-phase bands and the STI all take the one bank
    assert len(designs) == 1


def test_filtering_stops_once_the_filter_has_rung_out():
    # The bands of a segment with 0.5 s of zero padding, computed only until each filter has rung out, are pyfar's
    # bank over the whole padding to well below any rounding of a metric
    import pyfar as pf

    from compas_acoustics import analyser as analysis

    segment = make_ir(duration=0.08, seed=4)
    for fs in (44100, 48000):
        analyser = Analyser(make_ir())
        analyser.sampling_rate = fs
        padded = np.concatenate([segment, np.zeros(fs // 2)])
        exact = analysis._octave_bank(fs).process(pf.Signal(padded, fs)).time.squeeze()
        for k, (f, fast) in enumerate(analyser._band_signals(segment).items()):
            assert fast.shape == exact[k].shape
            assert np.max(np.abs(fast - exact[k])) <= 1e-12 * np.max(np.abs(exact[k])), (fs, f)
            assert np.sum(fast**2) == pytest.approx(np.sum(exact[k] ** 2), rel=1e-12)
            np.testing.assert_array_equal(analyser._filter_impulse_response(padded, f), fast)


def _filter_counter(monkeypatch):
    calls = []
    filter_band = Analyser._filter_impulse_response

    def counted(self, *args):
        calls.append(args)
        return filter_band(self, *args)

    monkeypatch.setattr(Analyser, "_filter_impulse_response", counted)

    def count(run):
        calls.clear()
        run()
        return len(calls)

    return count


def test_the_bands_of_the_decay_are_filtered_once_per_analyser(monkeypatch):
    # The zero-phase bands are shared by the reverberation times, the centre time, the decay curves and the
    # reliability for the analyser's lifetime: reliability() after analyse() filters nothing again
    count = _filter_counter(monkeypatch)
    analyser = Analyser(make_ir())
    alone = count(analyser.center_time)
    assert alone > 0
    for run in (analyser.reverberation_times, analyser.decay_curves, analyser.reliability, analyser.center_time):
        assert count(run) == 0, run.__name__

    analysed = Analyser(make_ir())
    assert count(analysed.analyse) == alone
    assert count(analysed.reliability) == 0
    # another rate or response filters again
    analysed.sampling_rate = 44100
    assert count(analysed.reverberation_times) == alone
    analysed.ir = make_ir(seed=1)
    assert count(analysed.center_time) == alone


def test_a_parameter_subset_matches_the_full_analysis(monkeypatch):
    full = Analyser(make_ir()).analyse()
    wanted = ("T30", "EDT", "C50", "C80", "D50", "STI")
    subset = Analyser(make_ir()).analyse(parameters=wanted)
    assert set(subset) == {"freqs", *wanted, *(f"{p}_overall" for p in wanted)}
    for key in subset:
        np.testing.assert_equal(subset[key], full[key], err_msg=key)
    # the centre time's zero-phase bands are the reverberation times' own; DRR, strength and lateral fraction do not run
    calls = []
    for name in ("direct_to_reverberant_ratio", "sound_strength", "lateral_fraction"):
        monkeypatch.setattr(Analyser, name, lambda self, *a, _name=name, **k: calls.append(_name))
    Analyser(make_ir()).analyse(parameters=wanted)
    assert calls == []
    with pytest.raises(ValueError):
        Analyser(make_ir()).analyse(parameters=["RT60"])


@needs_sti
def test_the_modulation_transfer_function_is_pyratos():
    import pyfar as pf
    import pyrato

    for seed in range(2):
        ir = make_ir(seed=seed)
        analyser = Analyser(ir)
        expected = np.squeeze(np.asarray(pyrato.parameters.modulation_transfer_function(pf.Signal(ir, analyser.sampling_rate))))
        np.testing.assert_allclose(analyser._modulation_transfer_function(ir, np.inf), expected, rtol=1e-9, atol=1e-12)


def test_decay_end_follows_the_energy_decay():
    from compas_acoustics.analyser import decay_end

    fs = Defaults().SAMPLING_RATE
    t = np.arange(3 * fs) / fs
    ir = np.exp(-6.907755278982137 * t)  # 60 dB per second
    assert decay_end(ir, 30.0, margin=0.0) / fs == pytest.approx(0.5, abs=0.01)
    assert decay_end(ir[None, :], 60.0) / fs == pytest.approx(1.05, abs=0.01)
    assert decay_end(np.zeros(100), 60.0) == 100


def test_analyse_of_a_mono_response():
    analyser = Analyser(make_ir())
    r = analyser.analyse()
    # ISO 3382-1:2009 Table A.1; C50 is a function of D50 and takes its bands
    assert r["freqs"][2:4] == [500, 1000]
    for name in ("C50", "C80", "D50", "Ts", "EDT", "T20", "T30"):
        assert r[f"{name}_overall"] == pytest.approx(np.nanmean([r[name][2], r[name][3]]), abs=0.01), name
    assert r["DRR_overall"] == analyser.direct_to_reverberant_ratio()["DRR_overall"]
    assert np.isnan(r["LF_overall"]) and np.isnan(r["LFC_overall"])
    assert _PARAM_NAMES[-2:] == ["LF", "LFC"]
    assert np.all(np.isnan(r["G"])) and np.isnan(r["G_overall"])
    with pytest.raises(ValueError, match="package-scale"):
        Analyser(make_ir()).sound_strength()


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


# ---------------------------------------------------------------- lateral energy fraction (ISO 3382-1 Annex A) --


def _ambisonic(arrivals, source=(0.0, 0.0, 0.0), receiver=(2.0, 0.0, 0.0), facing=(-1.0, 0.0, 0.0), n3d=False, fs=48000):
    """A first-order ACN ambisonic response on the package scale; each arrival ``(time_s, amplitude, doa)``, doa a unit
    vector in the receiver frame, adds ``amplitude * [1, y, z, x]`` (SN3D; N3D: first order times sqrt(3))."""
    from compas_acoustics import ImpulseResponse, Receiver, Source

    data = np.zeros((fs, 4))
    gain = np.sqrt(3.0) if n3d else 1.0
    for t, a, (x, y, z) in arrivals:
        data[int(round(t * fs))] += a * np.array([1.0, gain * y, gain * z, gain * x])
    return ImpulseResponse(
        "lf",
        Source("s", *source),
        Receiver("r", *receiver, direction=list(facing)),
        data,
        fs,
        generation={"scale": "package"},
        ambisonic_order=1,
        normalization="n3d" if n3d else "sn3d",
        channel_order="acn",
    )


def _direct_time(distance=2.0):
    return distance / Defaults().DEFAULT_SPEED_OF_SOUND


@pytest.mark.parametrize("n3d", [False, True])
def test_lateral_fraction_of_one_lateral_reflection(n3d):
    ir = _ambisonic([(_direct_time(), 1.0, (1, 0, 0)), (0.02, 0.5, (0, 1, 0))], n3d=n3d)
    lf = Analyser(ir).lateral_fraction()
    assert lf["LF"][3] == pytest.approx(0.2, abs=0.01)
    assert lf["LF"] == pytest.approx([0.2] * 7, abs=0.03)
    assert lf["LFC"][3] == pytest.approx(0.2, abs=0.01)
    assert lf["LF_overall"] == pytest.approx(0.2, abs=0.02)


def test_lateral_fraction_weights_by_cos_squared():
    ir = _ambisonic([(_direct_time(), 1.0, (1, 0, 0)), (0.02, 0.5, (np.sqrt(0.5), np.sqrt(0.5), 0))])
    lf = Analyser(ir).lateral_fraction()
    assert lf["LF"][3] == pytest.approx(0.125 / 1.25, abs=0.01)
    assert lf["LFC"][3] == pytest.approx(0.5 * 0.5 * np.sqrt(0.5) / 1.25, abs=0.01)


def test_lateral_fraction_ignores_lateral_energy_before_5_ms_and_after_80_ms():
    t0 = _direct_time()
    ir = _ambisonic([(t0, 1.0, (1, 0, 0)), (t0 + 0.003, 0.5, (0, 1, 0)), (t0 + 0.09, 0.5, (0, 1, 0))])
    assert Analyser(ir).lateral_fraction()["LF"][3] == pytest.approx(0.0, abs=0.01)


def test_lateral_fraction_with_the_source_overhead_uses_the_receivers_y_axis():
    ir = _ambisonic([(_direct_time(), 1.0, (0, 0, 1)), (0.02, 0.5, (0, 1, 0))], receiver=(0.0, 0.0, -2.0), facing=(1.0, 0.0, 0.0))
    assert Analyser(ir).lateral_fraction()["LF"][3] == pytest.approx(0.2, abs=0.01)


def test_lateral_fraction_of_an_occluded_pair_takes_the_axis_from_geometry():
    # the first arrival is not from the source: an axis from it would give 0.09 / 0.5
    t0 = _direct_time()
    ir = _ambisonic([(t0 + 0.010, 0.5, (0.6, 0.8, 0)), (t0 + 0.020, 0.5, (0, 1, 0))])
    assert Analyser(ir).lateral_fraction()["LF"][3] == pytest.approx(0.25 / 0.5, abs=0.02)


def test_lateral_fraction_is_nan_for_mono_and_binaural():
    from compas_acoustics import ImpulseResponse, Receiver, Source

    mono = make_ir()
    binaural = ImpulseResponse("b", Source("s", 0, 0, 0), Receiver("r", 2, 0, 0), np.column_stack([mono, mono]), 48000, generation={"scale": "package"})
    for ir in (mono, binaural):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lf = Analyser(ir).lateral_fraction()
        assert np.all(np.isnan(lf["LF"])) and np.all(np.isnan(lf["LFC"]))
        assert np.isnan(lf["LF_overall"]) and np.isnan(lf["LFC_overall"])


def test_analyse_holds_lf_for_an_ambisonic_response():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        data = Analyser(_ambisonic([(_direct_time(), 1.0, (1, 0, 0)), (0.02, 0.5, (0, 1, 0))])).analyse()
    assert data["LF_overall"] == pytest.approx(0.2, abs=0.02)
    assert data["LFC_overall"] == pytest.approx(0.2, abs=0.02)
    assert len(data["LF"]) == len(data["freqs"])


# ------------------------------------------------------------ sound pressure level from the source's level at 1 m --

A_WEIGHTS = [-16.1, -8.6, -3.2, 0.0, 1.2, 1.0, -1.1]


def _free_field(distance=2.0, spl=80.0, fs=48000):
    from compas_acoustics import ImpulseResponse, Receiver, Source

    source = Source("s", 0, 0, 0)
    source.spl = spl
    data = np.zeros(fs)
    data[int(round(distance / Defaults().DEFAULT_SPEED_OF_SOUND * fs))] = 1.0 / (4 * np.pi * distance)
    return ImpulseResponse("ff", source, Receiver("r", distance, 0, 0), data, fs, generation={"scale": "package"})


@pytest.mark.skipif(not _PYRATO_PARAMETERS, reason="SPL comes from G, which needs pyrato >= 1.0")
def test_free_field_spl_falls_6_db_per_doubling_from_the_level_at_1_m():
    spl = Analyser(_free_field()).sound_pressure_level()
    assert spl["SPL"] == pytest.approx([80 - 6.02] * 7, abs=0.1)
    assert spl["SPL_total"] == pytest.approx(80 - 6.02 + 10 * np.log10(7), abs=0.1)


@pytest.mark.skipif(not _PYRATO_PARAMETERS, reason="SPL comes from G, which needs pyrato >= 1.0")
def test_spl_overall_is_the_a_weighted_total():
    spl = Analyser(_free_field()).sound_pressure_level()
    expected = 10 * np.log10(sum(10 ** ((80 - 6.02 + a) / 10) for a in A_WEIGHTS))
    assert spl["SPL_overall"] == pytest.approx(expected, abs=0.1)


@pytest.mark.skipif(not _PYRATO_PARAMETERS, reason="SPL comes from G, which needs pyrato >= 1.0")
def test_spl_takes_a_given_level_over_the_sources():
    spl = Analyser(_free_field(spl=80.0)).sound_pressure_level(source_spl=90.0)
    assert spl["SPL"][3] == pytest.approx(90 - 6.02, abs=0.1)


def test_spl_of_a_response_without_absolute_level_is_nan_and_warns():
    with pytest.warns(UserWarning, match="package scale"):
        spl = Analyser(make_ir()).sound_pressure_level(source_spl=80)
    assert np.all(np.isnan(spl["SPL"])) and np.isnan(spl["SPL_overall"])


@pytest.mark.skipif(not _PYRATO_PARAMETERS, reason="SPL comes from G, which needs pyrato >= 1.0")
def test_analyse_holds_spl():
    data = Analyser(_free_field()).analyse()
    assert data["SPL"][3] == pytest.approx(80 - 6.02, abs=0.1)
    assert "SPL" in _PARAM_NAMES and _PARAM_NAMES.index("SPL") == _PARAM_NAMES.index("G") + 1


def test_clarity_and_definition_are_nan_without_a_late_part():
    ir = np.zeros(4800)
    ir[100] = 1.0  # nothing after the first 50 ms
    clarity = Analyser(ir).clarity_definition()
    assert np.all(np.isnan(clarity["C50"])) and np.all(np.isnan(clarity["C80"])) and np.all(np.isnan(clarity["D50"]))


def test_the_export_keeps_the_stored_values(tmp_path):
    import csv

    analyser = Analyser(make_ir())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        data = analyser.analyse()
        analyser.export_csv(str(tmp_path / "a.csv"), "Ts")
    rows = list(csv.reader(open(tmp_path / "a.csv")))
    assert [float(v) for v in rows[1][1:]] == data["Ts"] + [data["Ts_overall"]]
    assert any(len(v.split(".")[1]) == 4 for v in rows[1][1:-1])
