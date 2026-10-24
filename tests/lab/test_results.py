import numpy as np
from compas import json_dumps
from compas import json_loads

from compas_acoustics.lab.results import ScatteringResult


def _result(**kwargs):
    rng = np.random.default_rng(0)
    return ScatteringResult(
        method="geometric",
        bands=[1000, 1250, 1600, 2000],
        s_bands=[0.1, 0.2, 0.3, 0.4],
        octave_bands=[1000, 2000],
        s_octaves=[0.2, 0.4],
        filled_octaves=[125, 250, 500, 4000, 8000],
        s_angle=rng.random((3, 4)),
        source_colatitude=[0.0, 30.0, 60.0],
        source_azimuth=[0.0, 0.0, 0.0],
        source_weights=[0.2, 0.4, 0.4],
        receiver_directions=[[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]],
        polar_sample_db=np.zeros((1, 2, 2)),
        polar_reference_db=np.zeros((1, 2, 2)),
        **kwargs,
    )


def test_results_round_trip_with_d():
    d_angle = np.random.default_rng(1).random((3, 4))
    r = _result(d_angle=d_angle, d_bands=[0.5, 0.6, 0.7, 0.8], d_octaves=[0.6, 0.75])
    assert r.d_angle.shape == (3, 4) and r.d_bands == [0.5, 0.6, 0.7, 0.8] and r.d_octaves == [0.6, 0.75]
    t = json_loads(json_dumps(r))
    assert isinstance(t, ScatteringResult)
    assert np.array_equal(t.d_angle, r.d_angle) and t.d_bands == r.d_bands and t.d_octaves == r.d_octaves


def test_a_result_without_d_round_trips_and_old_data_loads():
    r = _result()
    assert r.d_angle is None and r.d_bands is None and r.d_octaves is None
    t = json_loads(json_dumps(r))
    assert t.d_angle is None and t.d_bands is None and t.d_octaves is None
    old = {k: v for k, v in r.__data__.items() if not k.startswith("d_")}  # written before d existed
    loaded = ScatteringResult.__from_data__(old)
    assert loaded.d_angle is None and loaded.d_bands is None and loaded.d_octaves is None and loaded.s_octaves == r.s_octaves
