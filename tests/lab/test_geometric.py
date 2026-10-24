import numpy as np
import pytest
from compas.datastructures import Mesh
from lab_shapes import flat_tile
from lab_shapes import sawtooth_tile
from lab_shapes import v_groove_tile
from scipy.spatial import cKDTree

from compas_acoustics.lab import ReceiverGrid
from compas_acoustics.lab import ScatteringSample
from compas_acoustics.lab import SourceGrid
from compas_acoustics.lab.raycast import RayCaster
from compas_acoustics.lab.geometric import _bottom_triangles
from compas_acoustics.lab.geometric import _inside
from compas_acoustics.lab.geometric import _trace
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
    sources = SourceGrid.preset("standard")
    assert sources.colatitude.max() == 80.0  # the grazing ring is in the grid
    response = geometric_response(sample, sources, n_rays=1000)
    # the binned energy and the lost fraction are counted separately, so their sum is a check
    np.testing.assert_allclose(response.sample.sum(axis=1) + response.lost_sample, 1.0, atol=1e-12)
    np.testing.assert_allclose(response.reference.sum(axis=1) + response.lost_reference, 1.0, atol=1e-12)
    assert (response.lost_sample >= 0).all() and (response.lost_reference >= 0).all()


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


def _closed_disc():
    return ScatteringSample.disc(sawtooth_tile(), 0.4)


def test_cells_alone_may_count_common_losses_as_scattering_and_the_lost_bin_removes_them():
    pytest.importorskip("embreex")
    body = _closed_disc().sample
    sample = ScatteringSample.as_drawn(body, reference=body)
    response = geometric_response(sample, SourceGrid.preset("standard"), n_rays=2000)
    # a body against itself: whatever is lost is lost in both
    cells = scattering_share(response.sample, response.reference)
    with_lost = scattering_share(np.column_stack([response.sample, response.lost_sample]), np.column_stack([response.reference, response.lost_reference]))
    np.testing.assert_allclose(with_lost, 0.0, atol=1e-12)
    np.testing.assert_allclose(cells, response.lost_sample, atol=1e-12)


def test_a_flat_disc_at_80_degrees_is_the_same_as_its_reference_and_loses_nothing():
    pytest.importorskip("embreex")
    sample = ScatteringSample.disc(flat_tile(), 0.4)
    sources = SourceGrid.preset("standard")
    response = geometric_response(sample, sources, n_rays=5000)
    ring = sources.colatitude == 80.0
    assert ring.sum() == 12
    assert np.array_equal(response.sample[ring], response.reference[ring])
    assert (response.lost_sample == 0).all() and (response.lost_reference == 0).all()


def test_the_object_is_mounted_in_a_baffle_so_a_windward_wall_does_not_throw_rays_below_the_horizon():
    pytest.importorskip("embreex")
    sample = _closed_disc()
    sources = SourceGrid.preset("standard")
    response = geometric_response(sample, sources, n_rays=20000)
    assert (response.lost_reference == 0).all()
    ring = sources.colatitude == 80.0
    # measured with the baffle: 0 on this ring (20000 rays, one ray is 5e-5), free-standing it was 0.12 to 0.27; what remains elsewhere in the preset
    # (at most 0.011) are rays that leave exactly along the horizon (colatitude 50 = 90 - 2 x 20 reflects off the slope to d_z = 0) or run out of bounces
    assert response.lost_sample[ring].max() < 1e-3


def _box(top=0.2, bottom=-0.1):
    """A closed unit-square box from z = ``bottom`` to ``top``, outward oriented."""
    vertices = [[0, 0, bottom], [1, 0, bottom], [1, 1, bottom], [0, 1, bottom], [0, 0, top], [1, 0, top], [1, 1, top], [0, 1, top]]
    faces = [[3, 2, 1, 0], [4, 5, 6, 7], [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]]
    return Mesh.from_vertices_and_faces(vertices, faces)


def _exit(start, direction, max_bounces=4, body=None, footprint=None):
    """Trace one ray off a body (the box by default, its footprint the unit square); return the cell directions, the energy per cell and the lost fraction."""
    grid = ReceiverGrid.hemisphere(10.0, 1.0)
    if footprint is None:
        footprint = np.array([[[0, 0], [1, 0], [1, 1]], [[0, 0], [1, 1], [0, 1]]], dtype=float)
    d = np.asarray(direction, dtype=float)
    caster = RayCaster(_box() if body is None else body)
    energy, lost = _trace(caster, footprint, np.array([start], dtype=float), d / np.linalg.norm(d), cKDTree(grid.directions), len(grid.weights), max_bounces)
    return grid.directions, energy, lost


def test_a_ray_reflected_downward_by_a_wall_bounces_on_the_baffle_and_leaves_upward():
    pytest.importorskip("embreex")
    # a ray along (-sin60, 0, -cos60) meets the east wall x = 1 at z = 0.1 (it starts 1 m east of it and 0.1 + 1/tan(60) = 0.677 high), reflects to
    # (+sin60, 0, -cos60), reaches z = 0 at x = 1 + 0.1 tan(60) = 1.173 and is reflected there to (+sin60, 0, +cos60): colatitude 60, azimuth 0
    a, b = np.sin(np.radians(60.0)), np.cos(np.radians(60.0))
    directions, energy, lost = _exit([2.0, 0.5, 0.1 + b / a], [-a, 0.0, -b])
    assert lost == 0.0 and energy.sum() == pytest.approx(1.0, abs=1e-12)
    assert np.argmax(energy) == np.argmax(directions @ [a, 0.0, b])
    # the ground bounce is a bounce: with one reflection allowed the ray is lost
    _, energy, lost = _exit([2.0, 0.5, 0.1 + b / a], [-a, 0.0, -b], max_bounces=1)
    assert lost == 1.0 and energy.sum() == 0.0


def test_a_ray_whose_first_face_is_below_the_baffle_outside_the_footprint_meets_the_baffle_first():
    pytest.importorskip("embreex")
    # a 45 degree ray from (2, 0.5, 0.95) would meet the east wall x = 1 at z = -0.05, but it crosses z = 0 at x = 1.05, 0.05 outside the footprint
    # (the unit square): there it is reflected to (-1, 0, +1)/sqrt2, meets the wall at z = 0.05 and is reflected to (+1, 0, +1)/sqrt2:
    # colatitude 45, azimuth 0 (two bounces; with one it is lost)
    s = np.sqrt(0.5)
    directions, energy, lost = _exit([2.0, 0.5, 0.95], [-s, 0.0, -s])
    assert lost == 0.0 and energy.sum() == pytest.approx(1.0, abs=1e-12)
    assert np.argmax(energy) == np.argmax(directions @ [s, 0.0, s])
    _, energy, lost = _exit([2.0, 0.5, 0.95], [-s, 0.0, -s], max_bounces=1)
    assert lost == 1.0 and energy.sum() == 0.0


def test_the_footprint_test_includes_its_edge_and_nothing_beyond():
    pytest.importorskip("embreex")
    disc = ScatteringSample.disc(flat_tile(), 0.4)
    corners, _ = _bottom_triangles(disc.reference, disc.baseplate)
    footprint = corners[:, :, :2]
    outline = footprint.reshape(-1, 2)
    centre = outline.mean(axis=0)
    corner = outline[np.linalg.norm(outline - centre, axis=1).argmax()]  # an outline vertex, the polygon's farthest point along its direction
    radius = np.linalg.norm(corner - centre)
    direction = (corner - centre) / radius
    points = np.array([centre + (radius - 1e-6) * direction, centre + radius * direction, centre + (radius + 1e-6) * direction, centre])
    assert _inside(footprint, points).tolist() == [True, True, False, True]


def test_the_baffle_is_only_outside_the_footprint():
    pytest.importorskip("embreex")
    # the wedge as drawn (baseplate 0.1): top z = 0.1 - 0.2 x, below the plane z = 0 for x > 0.5, bottom at z = -0.1, footprint the unit square
    sample = ScatteringSample.as_drawn(_wedge(), baseplate=0.1)
    corners, _ = _bottom_triangles(sample.reference, sample.baseplate)
    footprint = corners[:, :, :2]
    s = np.sqrt(0.5)
    # crossing z = 0 at x = 0.9, inside: the ray meets the slope at x = 1/1.2, z = -0.0667, below the plane, and reflects off it as drawn:
    # n = (0.2, 0, 1)/1.0198, d' = d - 2 (d.n) n = (-0.3807, 0, 0.9249)
    directions, energy, lost = _exit([1.9, 0.5, 1.0], [-s, 0.0, -s], body=sample.sample, footprint=footprint)
    assert lost == 0.0 and energy.sum() == pytest.approx(1.0, abs=1e-12)
    assert np.argmax(energy) == np.argmax(directions @ [-0.3807, 0.0, 0.9249])
    # crossing at x = 1.2, outside: the baffle reflects it to (-1, 0, 1)/sqrt2, which clears the wedge
    directions, energy, lost = _exit([2.2, 0.5, 1.0], [-s, 0.0, -s], body=sample.sample, footprint=footprint)
    assert lost == 0.0 and energy.sum() == pytest.approx(1.0, abs=1e-12)
    assert np.argmax(energy) == np.argmax(directions @ [-s, 0.0, s])


def test_a_ray_that_never_leaves_the_horizontal_is_lost():
    pytest.importorskip("embreex")
    _, energy, lost = _exit([2.0, 0.5, 0.15], [-1.0, 0.0, 0.0], max_bounces=4)  # reflects off the east wall to +x, d_z = 0: neither up nor down
    assert lost == 1.0 and energy.sum() == 0.0


def _wedge():
    """A closed prism with a sloped top: z = 0.2 (1 - x) over the unit square, a knife edge at x = 1."""
    vertices = [[0, 0, 0], [1, 0, 0], [0, 0, 0.2], [0, 1, 0], [1, 1, 0], [0, 1, 0.2]]
    faces = [[0, 3, 4, 1], [0, 2, 5, 3], [1, 4, 5, 2], [0, 1, 2], [3, 5, 4]]
    return Mesh.from_vertices_and_faces(vertices, faces)


@pytest.mark.parametrize("baseplate", [0.01, 0.1])
def test_a_sloped_face_in_the_baseplate_band_inside_the_footprint_reflects_as_drawn(baseplate):
    pytest.importorskip("embreex")
    # a closed mesh as drawn has its lowest point at z = -baseplate, so part of its slope is below z = 0; the baffle is only outside the footprint
    sample = ScatteringSample.as_drawn(_wedge(), baseplate=baseplate)
    response = geometric_response(sample, SourceGrid.preset("survey"), n_rays=5000)
    colatitude = np.degrees(np.arccos(np.clip(response.directions[:, 2], -1.0, 1.0)))
    energy = response.sample[0]
    # normal incidence on a slope of 0.2 leaves at 2 atan(0.2) = 22.6 degrees; the 10 degree cells are within +-5 degrees of it
    assert energy[np.abs(colatitude - 22.6) < 6.0].sum() > 0.99
    assert energy[colatitude < 5.0].sum() == 0.0
    assert response.lost_sample[0] == 0.0
