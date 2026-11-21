"""Decay curves, reliability and channel choice in the analyser."""

import warnings

import numpy as np
import pytest

import compas_acoustics as ca

FS = 48000


def _decay(rt, seconds, seed=0, second_rt=None, knee_db=-20.0):
    """Exponentially decaying noise of reverberation time ``rt``, with a direct impulse; optionally a second, slower slope."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * FS)) / FS
    envelope = 10 ** (-3 * t / rt)
    if second_rt is not None:
        envelope = np.maximum(envelope, 10 ** (knee_db / 20) * 10 ** (-3 * t / second_rt))
    ir = rng.standard_normal(t.size) * envelope * 0.05
    ir[0] = 1.0
    return ir


def _quiet(function, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return function(*args, **kwargs)


def test_decay_curves_start_at_zero_and_fall():
    curves = ca.Analyser(_decay(0.8, 2.0)).decay_curves()
    assert curves["freqs"] == [125, 250, 500, 1000, 2000, 4000, 8000]
    for edc in curves["edc_db"]:
        assert edc[0] == pytest.approx(0.0)
        finite = edc[np.isfinite(edc)]
        assert np.all(np.diff(finite) <= 1e-9)
    assert curves["times"][1] == pytest.approx(1 / FS)


def test_reliability_flags_a_cut_and_a_bent_decay():
    full = ca.Analyser(_decay(0.8, 2.0)).reliability()
    assert all(full["T30_reliable"][2:6]) and all(full["EDT_reliable"])
    assert max(full["T30_xi"][2:6]) < 10.0
    cut = ca.Analyser(_decay(0.8, 0.3)).reliability()
    assert not any(cut["T30_reliable"])
    assert all(r < 45.0 for r in cut["range_db"])
    coupled = _quiet(ca.Analyser(_decay(1.0, 6.0, second_rt=4.0, knee_db=-20.0)).reliability)  # a coupled volume's knee
    assert min(coupled["T30_xi"][2:6]) > 10.0 and not any(coupled["T30_reliable"][2:6])


def test_the_analyser_reads_the_channel_asked_for():
    four = np.stack([_decay(0.8, 1.0, seed=k) for k in range(4)], axis=1)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        w = ca.Analyser(four, channel=0)
        y = ca.Analyser(four, channel=1)
    np.testing.assert_array_equal(w.ir, four[:, 0])
    np.testing.assert_array_equal(y.ir, four[:, 1])
    with pytest.warns(UserWarning, match="channel 0 only"):
        ca.Analyser(four)
    with pytest.raises(ValueError):
        ca.Analyser(four, channel=4)
    with pytest.raises(ValueError):
        ca.Analyser(four[:, 0], channel=1)
