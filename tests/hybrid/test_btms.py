"""BTMS kernels: properties the toolbox comparison does not pin.

Ported from hybrid-rir (gpapadimitri/hybrid-rir @ 8a70661, tests/test_btms.py).
"""

import numpy as np
import pytest

from compas_acoustics.hybrid.btms import element_grid, wedge1st_ir

FS, C = 48000.0, 344.0


def test_zero_length_edge_gives_silence():
    ir, _, _, zfirst = wedge1st_ir(FS, C, np.pi / 2, 1.0, 0.3, 0.0, 1.5, 4.0, 0.2, np.array([0.4, 0.4]))
    assert np.all(ir == 0.0) and zfirst is None


def test_segments_sum_to_the_whole_edge():
    args = (FS, C, np.pi / 2, 1.0, 0.3, 0.0, 1.5, 4.0, 0.7)
    whole, *_ = wedge1st_ir(*args, np.array([-1.0, 2.0]))
    cuts = np.array([-1.0, -0.2, 0.31, 1.1, 2.0])
    parts = [wedge1st_ir(*args, cuts[i:i + 2])[0] for i in range(4)]
    total = np.zeros(max(len(whole), *(len(p) for p in parts)))
    for p in parts:
        total[:len(p)] += p
    # the analytic apex approximation (zrelmax = 0.1) is good to ~1.6e-5 of peak and its range moves with the cut
    np.testing.assert_allclose(total[:len(whole)], whole, rtol=0, atol=5e-5 * np.max(np.abs(whole)))


def test_a_very_short_segment_is_finite_and_small():
    long_, *_ = wedge1st_ir(FS, C, np.pi / 2, 1.0, 0.3, 0.0, 1.5, 4.0, 0.7, np.array([-1.0, 1.0]))
    short, *_ = wedge1st_ir(FS, C, np.pi / 2, 1.0, 0.3, 0.0, 1.5, 4.0, 0.7, np.array([0.5, 0.5 + 1e-4]))
    assert np.all(np.isfinite(short))
    assert np.sum(short**2) < 1e-3 * np.sum(long_**2)


def test_a_segment_shorter_than_one_sample_gets_one_element():
    steps, dz = element_grid(np.array([1e-4, 0.5]), C, FS)
    assert dz[0] == 1e-4
    assert steps.shape[0] == int(np.ceil(0.5 / (C / FS)))


def test_an_irstart_after_the_apex_arrival_is_refused():
    args = (FS, C, np.pi / 2, 1.0, 0.3, 0.0, 1.5, 4.0, 0.7, np.array([-1.0, 2.0]))
    R0 = np.hypot(2.5, 0.7)
    with pytest.raises(ValueError, match="r_irstart"):
        wedge1st_ir(*args, r_irstart=R0 + 3 * C / FS)
    ir, *_ = wedge1st_ir(*args, r_irstart=R0 + 0.4 * C / FS)
    assert np.all(np.isfinite(ir))


def test_a_segment_ending_on_the_apex_is_finite():
    args = (FS, C, np.pi / 2, 1.0, 0.3, 1.0, 1.5, 4.0, 1.0)
    for zw in ([0.5, 1.0], [1.0, 1.6]):
        ir, *_ = wedge1st_ir(*args, np.array(zw))
        assert np.all(np.isfinite(ir)) and np.any(ir != 0.0)
