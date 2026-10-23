import numpy as np
import pytest
from compas.data import json_dumps
from compas.data import json_loads
from compas.datastructures import Mesh
from compas.geometry import volume_polyhedron

from compas_acoustics.lab.sample import ScatteringSample


def flat_tile(size=0.1):
    return Mesh.from_vertices_and_faces([[0, 0, 0], [size, 0, 0], [size, size, 0], [0, size, 0]], [[0, 1, 2, 3]])


def test_flat_disc_is_closed_and_has_the_disc_volume():
    s = ScatteringSample.disc(flat_tile(), diameter=0.8, baseplate=0.01, segments=256)
    assert s.sample.is_closed() and s.reference.is_closed()
    assert volume_polyhedron(s.sample.to_vertices_and_faces()) == pytest.approx(np.pi * 0.4**2 * 0.01, rel=1e-3)
    assert s.structural_depth == pytest.approx(0.0, abs=1e-9)


def test_relief_minimum_and_reference_top_are_at_z0(sine_tile):
    s = ScatteringSample.disc(sine_tile(period=0.0708, depth=0.0204), diameter=0.8)
    zs = np.array([s.sample.vertex_coordinates(v)[2] for v in s.sample.vertices()])
    assert zs[zs > -1e-9].min() == pytest.approx(0.0, abs=1e-9)
    assert s.structural_depth == pytest.approx(0.0204, rel=1e-2)
    assert max(s.reference.vertex_coordinates(v)[2] for v in s.reference.vertices()) == pytest.approx(0.0)


def test_reference_has_the_sample_footprint(sine_tile, footprint_area):
    s = ScatteringSample.disc(sine_tile(period=0.0708, depth=0.0204), diameter=0.8)
    assert footprint_area(s.sample) == pytest.approx(footprint_area(s.reference), rel=1e-9)


def test_disc_from_unwelded_quad_tile_is_closed():
    # two quads sharing an edge, but with duplicated vertices along it (as Rhino exports split seams)
    tile = Mesh.from_vertices_and_faces(
        [[0, 0, 0], [0.05, 0, 0.01], [0.05, 0.1, 0.01], [0, 0.1, 0], [0.05, 0, 0.01], [0.1, 0, 0], [0.1, 0.1, 0], [0.05, 0.1, 0.01]], [[0, 1, 2, 3], [4, 5, 6, 7]]
    )
    assert ScatteringSample.disc(tile, diameter=0.5).sample.is_closed()


def test_disc_from_tile_with_hole_raises(tile_with_hole):
    with pytest.raises(ValueError, match="one closed outline"):
        ScatteringSample.disc(tile_with_hole(), diameter=0.5)


def test_as_drawn_open_patch_is_closed_with_matching_reference(open_l_shaped_relief, footprint_area):
    s = ScatteringSample.as_drawn(open_l_shaped_relief())
    assert s.mode == "as_drawn" and s.sample.is_closed() and s.reference.is_closed()
    assert footprint_area(s.sample) == pytest.approx(footprint_area(s.reference), rel=1e-9)


def test_as_drawn_two_boundaries_raises(two_separate_patches):
    with pytest.raises(ValueError, match="one closed outline"):
        ScatteringSample.as_drawn(two_separate_patches())


def test_deep_relief_warns_iso_17497_1(sine_tile):
    s = ScatteringSample.disc(sine_tile(period=0.1, depth=0.1), diameter=0.8)
    assert any("d/16" in w for w in s.warnings)


def test_sample_larger_than_20_m_warns():
    s = ScatteringSample.disc(flat_tile(100.0), diameter=800.0)
    assert any("millimetres" in w for w in s.warnings)


def test_sample_round_trips_through_json():
    s = ScatteringSample.disc(flat_tile(), diameter=0.4)
    t = json_loads(json_dumps(s))
    assert isinstance(t, ScatteringSample) and t.diameter == 0.4 and t.sample.number_of_faces() == s.sample.number_of_faces()
