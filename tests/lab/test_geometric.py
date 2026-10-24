import numpy as np
import pytest
from lab_shapes import flat_tile
from lab_shapes import sawtooth_tile
from lab_shapes import v_groove_tile

from compas_acoustics.lab import ReceiverGrid
from compas_acoustics.lab import ScatteringSample
from compas_acoustics.lab import SourceGrid
from compas_acoustics.lab.geometric import geometric_response
from compas_acoustics.lab.geometric import heuristic_s
from compas_acoustics.lab.geometric import scattering_share


def _cell(response, direction):
    """The index of the cell whose direction is nearest to ``direction``."""
    return int(np.argmax(response.directions @ np.asarray(direction, dtype=float)))


def _mean_colatitude(response, energy):
    """The energy-weighted mean colatitude in degrees of one row of energy fractions."""
    colatitude = np.degrees(np.arccos(np.clip(response.directions[:, 2], -1.0, 1.0)))
    return float((energy * colatitude).sum() / energy.sum())


def test_a_flat_sample_reflects_everything_into_the_specular_cell():
    pytest.importorskip("embreex")
    sample = ScatteringSample.disc(flat_tile(), 0.4)
    sources = SourceGrid.preset("survey")
    response = geometric_response(sample, sources, n_rays=2000)
    for i, u in enumerate(sources.directions):
        cell = _cell(response, [-u[0], -u[1], u[2]])
        assert response.sample[i, cell] == pytest.approx(1.0, abs=1e-9)
        assert response.reference[i, cell] == pytest.approx(1.0, abs=1e-9)
    assert np.allclose(scattering_share(response.sample, response.reference), 0.0, atol=1e-9)
    assert np.allclose(response.lost_sample, 0.0) and np.allclose(response.lost_reference, 0.0)
    assert response.depth == pytest.approx(0.0, abs=1e-12)
    assert response.directions.shape[1] == 3 and response.weights.sum() == pytest.approx(2 * np.pi, rel=1e-6)


def test_energy_is_conserved():
    pytest.importorskip("embreex")
    sample = ScatteringSample.disc(sawtooth_tile(), 0.4)
    response = geometric_response(sample, SourceGrid.preset("standard"), n_rays=1000)
    assert np.allclose(response.sample.sum(axis=1) + response.lost_sample, 1.0, atol=1e-9)
    assert np.allclose(response.reference.sum(axis=1) + response.lost_reference, 1.0, atol=1e-9)


def test_a_tilted_facet_sends_normal_incidence_to_twice_its_slope():
    pytest.importorskip("embreex")
    sample = ScatteringSample.disc(sawtooth_tile(slope_deg=20.0), 0.4)
    response = geometric_response(sample, SourceGrid.preset("survey"), n_rays=20000)
    # the first survey source is at colatitude 0
    assert scattering_share(response.sample[0], response.reference[0]) > 0.95
    assert _mean_colatitude(response, response.sample[0]) == pytest.approx(40.0, abs=5.0)


def test_a_right_angle_groove_retroreflects_with_enough_bounces():
    pytest.importorskip("embreex")
    sample = ScatteringSample.disc(v_groove_tile(), 0.4)
    sources = SourceGrid.preset("survey")
    response = geometric_response(sample, sources, n_rays=5000, max_bounces=4)
    up = _cell(response, [0.0, 0.0, 1.0])
    assert response.sample[0, up] > 0.9
    assert scattering_share(response.sample[0], response.reference[0]) < 0.1
    response = geometric_response(sample, sources, n_rays=5000, max_bounces=1)
    assert response.lost_sample[0] > 0.8


def test_a_sample_equal_to_its_proxy_scatters_nothing():
    pytest.importorskip("embreex")
    # a closed disc body, drawn as is, with itself as the proxy (a single bare tile would lose every ray that meets its free-standing side walls)
    body = ScatteringSample.disc(sawtooth_tile(), 0.4).sample
    sample = ScatteringSample.as_drawn(body, reference=body)
    assert sample.reference_kind == "proxy"
    response = geometric_response(sample, SourceGrid.preset("survey"), n_rays=2000)
    # rays lost at the rim are lost identically in both; the lost fraction is one more bin of the histograms (Task 5 does the same)
    with_lost = (np.column_stack([response.sample, response.lost_sample]), np.column_stack([response.reference, response.lost_reference]))
    assert response.lost_sample.max() > 0
    assert np.allclose(scattering_share(*with_lost), 0.0, atol=1e-9)
    assert response.depth == pytest.approx(0.0, abs=1e-9)


def test_the_same_seed_gives_the_same_response_and_another_seed_a_close_one():
    pytest.importorskip("embreex")
    sample = ScatteringSample.disc(sawtooth_tile(), 0.4)
    sources = SourceGrid.preset("survey")
    a = geometric_response(sample, sources, n_rays=5000, seed=0)
    b = geometric_response(sample, sources, n_rays=5000, seed=0)
    c = geometric_response(sample, sources, n_rays=5000, seed=1)
    assert np.array_equal(a.sample, b.sample) and np.array_equal(a.reference, b.reference)
    assert np.array_equal(a.lost_sample, b.lost_sample)
    assert np.allclose(scattering_share(a.sample, a.reference), scattering_share(c.sample, c.reference), atol=0.05)


def test_depth_is_twice_the_standard_deviation_of_the_height():
    pytest.importorskip("embreex")
    period, slope = 0.05, 20.0
    sample = ScatteringSample.disc(sawtooth_tile(period=period, slope_deg=slope), 0.4)
    response = geometric_response(sample, SourceGrid.preset("survey"), n_rays=20000)
    height = period * np.tan(np.radians(slope))
    assert response.depth == pytest.approx(2 * height / np.sqrt(12), rel=0.1)


def test_the_depth_curve():
    s = heuristic_s(1.0, 0.01, np.array([10.0, 1e5]), 343.0)
    assert s.shape == (2,)
    assert s[0] < 1e-3 and s[1] > 0.999
    f = np.geomspace(10, 1e5, 30)
    assert np.all(np.diff(heuristic_s(1.0, 0.01, f, 343.0)) >= 0)
    assert np.all(heuristic_s(0.4, 0.0, f, 343.0) == 0.0)
    assert heuristic_s(np.array([0.2, 0.5, 1.0]), 0.01, f, 343.0).shape == (3, 30)


def test_scattering_share_is_one_minus_the_histogram_overlap():
    a = np.array([[0.5, 0.5, 0.0], [1.0, 0.0, 0.0]])
    b = np.array([[0.5, 0.0, 0.5], [0.0, 1.0, 0.0]])
    assert np.allclose(scattering_share(a, b), [0.5, 1.0])


def test_the_receiver_cells_are_the_hemisphere_grid_at_bin_deg():
    pytest.importorskip("embreex")
    sample = ScatteringSample.disc(flat_tile(), 0.4)
    response = geometric_response(sample, SourceGrid.preset("survey"), bin_deg=15.0, n_rays=500)
    grid = ReceiverGrid.hemisphere(15.0, 1.0)
    assert np.allclose(response.directions, grid.directions) and np.allclose(response.weights, grid.weights)
