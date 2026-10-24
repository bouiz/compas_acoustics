import numpy as np
import pytest
from compas.data import json_dumps
from compas.data import json_loads
from compas.datastructures import Mesh
from compas.geometry import volume_polyhedron

from lab_shapes import _grid
from lab_shapes import bumped_box
from lab_shapes import flat_tile
from lab_shapes import footprint_area
from lab_shapes import open_l_shaped_relief
from lab_shapes import sine_tile
from lab_shapes import sawtooth_tile
from lab_shapes import step_tile
from lab_shapes import tile_with_hole
from lab_shapes import two_separate_patches
from lab_shapes import v_groove_tile

from compas_acoustics.lab.sample import ScatteringSample


def test_flat_disc_is_closed_and_has_the_disc_volume():
    s = ScatteringSample.disc(flat_tile(), diameter=0.8, baseplate=0.01, segments=256)
    assert s.sample.is_closed() and s.reference.is_closed()
    assert volume_polyhedron(s.sample.to_vertices_and_faces()) == pytest.approx(np.pi * 0.4**2 * 0.01, rel=1e-3)
    assert s.structural_depth == pytest.approx(0.0, abs=1e-9)


def test_relief_minimum_and_reference_top_are_at_z0():
    s = ScatteringSample.disc(sine_tile(period=0.0708, depth=0.0204), diameter=0.8)
    zs = np.array([s.sample.vertex_coordinates(v)[2] for v in s.sample.vertices()])
    assert zs[zs > -1e-9].min() == pytest.approx(0.0, abs=1e-9)
    assert s.structural_depth == pytest.approx(0.0204, rel=1e-2)
    assert max(s.reference.vertex_coordinates(v)[2] for v in s.reference.vertices()) == pytest.approx(0.0)


def test_reference_has_the_sample_footprint():
    s = ScatteringSample.disc(sine_tile(period=0.0708, depth=0.0204), diameter=0.8)
    assert footprint_area(s.sample) == pytest.approx(footprint_area(s.reference), rel=1e-9)


def test_disc_from_unwelded_quad_tile_is_closed():
    # two quads sharing an edge, but with duplicated vertices along it (as Rhino exports split seams)
    tile = Mesh.from_vertices_and_faces(
        [[0, 0, 0], [0.05, 0, 0.01], [0.05, 0.1, 0.01], [0, 0.1, 0], [0.05, 0, 0.01], [0.1, 0, 0], [0.1, 0.1, 0], [0.05, 0.1, 0.01]], [[0, 1, 2, 3], [4, 5, 6, 7]]
    )
    assert ScatteringSample.disc(tile, diameter=0.5).sample.is_closed()


def test_disc_from_tile_with_hole_raises():
    with pytest.raises(ValueError, match="one closed outline"):
        ScatteringSample.disc(tile_with_hole(), diameter=0.5)


def test_as_drawn_open_patch_is_closed_with_matching_reference():
    s = ScatteringSample.as_drawn(open_l_shaped_relief())
    assert s.mode == "as_drawn" and s.sample.is_closed() and s.reference.is_closed()
    assert footprint_area(s.sample) == pytest.approx(footprint_area(s.reference), rel=1e-9)


def test_as_drawn_two_boundaries_raises():
    with pytest.raises(ValueError, match="one closed outline"):
        ScatteringSample.as_drawn(two_separate_patches())


def test_deep_relief_warns_iso_17497_1():
    s = ScatteringSample.disc(sine_tile(period=0.1, depth=0.1), diameter=0.8)
    assert any("d/16" in w for w in s.warnings)


def test_sample_larger_than_20_m_warns():
    s = ScatteringSample.disc(flat_tile(100.0), diameter=800.0)
    assert any("millimetres" in w for w in s.warnings)


def test_sample_round_trips_through_json():
    s = ScatteringSample.disc(flat_tile(), diameter=0.4)
    t = json_loads(json_dumps(s))
    assert isinstance(t, ScatteringSample) and t.diameter == 0.4 and t.sample.number_of_faces() == s.sample.number_of_faces()


def test_disc_rim_across_vertical_risers_is_closed():
    # the rim cuts the step tile's vertical risers, so the outline has points above one another
    s = ScatteringSample.disc(step_tile(), diameter=0.5, segments=64)
    assert s.sample.is_closed() and s.reference.is_closed()
    assert footprint_area(s.sample) == pytest.approx(footprint_area(s.reference), rel=1e-9)
    assert s.structural_depth == pytest.approx(0.02)


def test_closed_as_drawn_mesh_sits_on_the_baseplate():
    s = ScatteringSample.as_drawn(bumped_box(size=1.0, thickness=0.01, bump=0.02), baseplate=0.01)
    assert s.sample.is_closed() and s.reference.is_closed()
    assert s.structural_depth == pytest.approx(0.02)
    assert not any("d/16" in w for w in s.warnings)
    assert footprint_area(s.sample) == pytest.approx(footprint_area(s.reference), rel=1e-9)
    zs = [s.sample.vertex_coordinates(v)[2] for v in s.sample.vertices()]
    assert min(zs) == pytest.approx(-0.01)


def _z(body):
    return np.array([body.vertex_coordinates(v) for v in body.vertices()])


def _flat_l_patch():
    return _grid(4, 4, 0.1, keep=lambda i, j: i < 2 or j < 2)


def test_a_proxy_replaces_the_flat_reference():
    s = ScatteringSample.as_drawn(bumped_box(size=1.0, thickness=0.01, bump=0.02), reference=bumped_box(size=1.0, thickness=0.01, bump=0.0))
    assert s.reference_kind == "proxy"
    assert s.reference.is_closed()
    P = _z(s.reference)
    away = np.abs(P[:, :2] - 0.5).max(axis=1) > 1e-6
    assert P[away, 2].max() == pytest.approx(0.0)
    assert P[:, 2].min() == pytest.approx(-0.01)
    assert any("proxy" in w for w in s.warnings)
    assert "not the ISO 17497 coefficient" in " ".join(s.warnings)


def test_without_a_proxy_the_reference_is_flat():
    s = ScatteringSample.as_drawn(bumped_box())
    assert s.reference_kind == "flat"
    assert not any("proxy" in w for w in s.warnings)


def test_a_proxy_with_another_outline_is_refused():
    with pytest.raises(ValueError, match="outline"):
        ScatteringSample.as_drawn(bumped_box(size=1.0), reference=bumped_box(size=2.0, bump=0.0))


def test_an_open_proxy_is_closed_like_the_sample():
    s = ScatteringSample.as_drawn(open_l_shaped_relief(), reference=_flat_l_patch())
    assert s.reference_kind == "proxy" and s.sample.is_closed() and s.reference.is_closed()
    P = _z(s.reference)
    assert P[:, 2].min() == pytest.approx(-0.01) and P[:, 2].max() == pytest.approx(0.0)
    assert footprint_area(s.sample) == pytest.approx(footprint_area(s.reference), rel=1e-9)


def test_an_open_proxy_is_registered_to_the_sample_lowest_point():
    # a proxy lying 0.003 m above the sample's lowest point stays there
    proxy = _grid(4, 4, 0.1, z=lambda i, j: 0.003, keep=lambda i, j: i < 2 or j < 2)
    s = ScatteringSample.as_drawn(open_l_shaped_relief(), reference=proxy)
    assert _z(s.reference)[:, 2].max() == pytest.approx(0.003)


def test_a_mixed_open_and_closed_pair_is_refused():
    with pytest.raises(ValueError, match="open.*closed|closed.*open"):
        ScatteringSample.as_drawn(open_l_shaped_relief(), reference=bumped_box())
    with pytest.raises(ValueError, match="open.*closed|closed.*open"):
        ScatteringSample.as_drawn(bumped_box(), reference=_flat_l_patch())


def test_reference_kind_round_trips():
    s = ScatteringSample.as_drawn(bumped_box(), reference=bumped_box(bump=0.0))
    assert ScatteringSample.__from_data__(s.__data__).reference_kind == "proxy"
    data = ScatteringSample.as_drawn(bumped_box()).__data__
    del data["reference_kind"]
    assert ScatteringSample.__from_data__(data).reference_kind == "flat"


def test_v_groove_tile_tiles_into_a_closed_sample():
    s = ScatteringSample.disc(v_groove_tile(), 0.4)
    assert s.sample.is_closed() and s.reference.is_closed()
    assert s.structural_depth == pytest.approx(0.02)


def test_sawtooth_tile_tiles_into_a_closed_sample():
    s = ScatteringSample.disc(sawtooth_tile(), 0.4)
    assert s.sample.is_closed() and s.reference.is_closed()
    assert s.structural_depth == pytest.approx(0.05 * np.tan(np.radians(20.0)))


def _polar_disc_patch(radius=0.5, n=64, rings=8, z=0.0, wobble=0.0):
    """An open disc patch on a polar grid (a 64-gon), at height ``z`` plus ``wobble·sin(20x)·sin(20y)``."""
    V = [[0.0, 0.0, z]]
    for k in range(1, rings + 1):
        for j in range(n):
            x, y = radius * k / rings * np.cos(2 * np.pi * j / n), radius * k / rings * np.sin(2 * np.pi * j / n)
            V.append([x, y, z + wobble * np.sin(20 * x) * np.sin(20 * y)])
    F = [[0, 1 + j, 1 + (j + 1) % n] for j in range(n)]
    for k in range(1, rings):
        b0, b1 = 1 + (k - 1) * n, 1 + k * n
        F += [[b0 + j, b1 + j, b1 + (j + 1) % n, b0 + (j + 1) % n] for j in range(n)]
    return Mesh.from_vertices_and_faces(V, F)


def _square_patch(size=1.0, n=4):
    xs = np.linspace(-size / 2, size / 2, n + 1)
    V = [[x, y, 0.0] for y in xs for x in xs]
    F = [[j * (n + 1) + i, j * (n + 1) + i + 1, (j + 1) * (n + 1) + i + 1, (j + 1) * (n + 1) + i] for j in range(n) for i in range(n)]
    return Mesh.from_vertices_and_faces(V, F)


def _box_with_a_pointed_bottom():
    """A closed unit-square body from a point at (0.5, 0.5, 0) up to a flat top at z = 0.5: its lowest point is a vertex, it has no flat bottom."""
    V = [[0, 0, 0.1], [1, 0, 0.1], [1, 1, 0.1], [0, 1, 0.1], [0, 0, 0.5], [1, 0, 0.5], [1, 1, 0.5], [0, 1, 0.5], [0.5, 0.5, 0.0]]
    F = [[4, 5, 6, 7], [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7], [1, 0, 8], [2, 1, 8], [3, 2, 8], [0, 3, 8]]
    return Mesh.from_vertices_and_faces(V, F)


def _lifted(mesh, dz):
    for v in mesh.vertices():
        mesh.vertex_attribute(v, "z", mesh.vertex_attribute(v, "z") + dz)
    return mesh


@pytest.mark.parametrize(
    "relief, proxy, match",
    [
        # the same bounding box, another footprint: a flat square over a flat disc (20 % of the rays fell through the square's corners)
        (_polar_disc_patch, _square_patch, "footprint"),
        # a closed proxy floating 0.1 m above the sample's lowest point
        (lambda: bumped_box(size=1.0, thickness=0.5, bump=0.0), lambda: _lifted(bumped_box(size=1.0, thickness=0.3, bump=0.0), 0.1), "lowest point"),
        # a closed proxy whose lowest point is a vertex
        (lambda: bumped_box(size=1.0, thickness=0.5, bump=0.0), _box_with_a_pointed_bottom, "flat bottom"),
        # an open proxy 0.03 m below the sample's floor, more than the 0.01 m baseplate
        (lambda: _polar_disc_patch(wobble=0.02), lambda: _polar_disc_patch(z=-0.05), "below"),
        # an open proxy of two parts
        (open_l_shaped_relief, two_separate_patches, "outline"),
    ],
)
def test_a_proxy_that_does_not_fit_the_sample_is_refused_naming_the_proxy(relief, proxy, match):
    with pytest.raises(ValueError, match=match) as raised:
        ScatteringSample.as_drawn(relief(), reference=proxy())
    assert "proxy" in str(raised.value)


def test_a_proxy_of_the_same_footprint_is_accepted():
    s = ScatteringSample.as_drawn(_polar_disc_patch(wobble=0.02), reference=_polar_disc_patch())
    assert s.reference_kind == "proxy" and footprint_area(s.reference) == pytest.approx(footprint_area(s.sample), rel=1e-9)
