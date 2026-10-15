import warnings

import numpy as np
import pytest

from compas_acoustics import AcousticAnalyser
from compas_acoustics.analyser import _STI_SUPPORTED
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
        AcousticAnalyser(short).compute_sti()["mti"],
        AcousticAnalyser(padded).compute_sti()["mti"],
    )


@needs_sti
def test_sti_failure_warns_instead_of_passing_silently(monkeypatch):
    import pyrato

    def fail(*args, **kwargs):
        raise ValueError("boom")

    monkeypatch.setattr(pyrato.parameters, "modulation_transfer_function", fail)
    with pytest.warns(UserWarning, match="STI computation failed: boom"):
        result = AcousticAnalyser(make_ir()).compute_sti()
    assert np.isnan(result["sti"])
