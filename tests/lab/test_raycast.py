import sys
import time

import numpy as np
import pytest
from compas.datastructures import Mesh
from lab_shapes import bumped_box

from compas_acoustics.lab.raycast import RayCaster


def _box():
    return bumped_box(size=1.0, thickness=1.0, bump=0.0)


def test_a_ray_down_onto_a_box_hits_its_top_face():
    pytest.importorskip("embreex")
    caster = RayCaster(_box())
    triangle, point = caster.first_hits(np.array([[0.3, 0.4, 3.0]]), np.array([[0.0, 0.0, -1.0]]))
    assert triangle.shape == (1,) and point.shape == (1, 3)
    assert triangle[0] >= 0
    assert point[0, 2] == pytest.approx(1.0, abs=1e-9)
    assert np.allclose(point[0, :2], [0.3, 0.4], atol=1e-9)
    assert np.allclose(caster.normals[triangle[0]], [0.0, 0.0, 1.0])


def test_a_ray_beside_the_box_misses():
    pytest.importorskip("embreex")
    caster = RayCaster(_box())
    triangle, point = caster.first_hits(np.array([[5.0, 5.0, 3.0]]), np.array([[0.0, 0.0, -1.0]]))
    assert triangle[0] == -1
    assert point.shape == (1, 3)


def test_the_first_hit_is_the_nearest_one_and_hits_and_misses_are_scattered_back_in_order():
    pytest.importorskip("embreex")
    caster = RayCaster(_box())
    origins = np.array([[5.0, 5.0, 3.0], [0.3, 0.4, -3.0], [0.7, 0.2, 3.0]])
    directions = np.array([[0.0, 0.0, -1.0], [0.0, 0.0, 1.0], [0.0, 0.0, -1.0]])
    triangle, point = caster.first_hits(origins, directions)
    assert triangle[0] == -1
    assert point[1, 2] == pytest.approx(0.0, abs=1e-9)  # from below: the bottom, not the top
    assert np.allclose(caster.normals[triangle[1]], [0.0, 0.0, -1.0])
    assert point[2, 2] == pytest.approx(1.0, abs=1e-9)  # from above: the top, not the bottom


def test_a_ray_from_a_surface_point_does_not_hit_its_own_face():
    pytest.importorskip("embreex")
    caster = RayCaster(_box())
    triangle, _ = caster.first_hits(np.array([[0.3, 0.4, 1.0]]), np.array([[0.0, 0.0, 1.0]]))
    assert triangle[0] == -1
    triangle, point = caster.first_hits(np.array([[0.3, 0.4, 1.0]]), np.array([[0.0, 0.0, -1.0]]))
    assert point[0, 2] == pytest.approx(0.0, abs=1e-9)


def test_a_body_far_from_the_origin_is_hit_where_it_is():
    pytest.importorskip("embreex")
    box = _box()
    box.translate([1000.0, -2000.0, 500.0])
    caster = RayCaster(box)
    triangle, point = caster.first_hits(np.array([[1000.3, -1999.6, 503.0]]), np.array([[0.0, 0.0, -1.0]]))
    assert triangle[0] >= 0
    assert point[0, 2] == pytest.approx(501.0, abs=1e-9)
    triangle, _ = caster.first_hits(np.array([[1000.3, -1999.6, 501.0]]), np.array([[0.0, 0.0, 1.0]]))
    assert triangle[0] == -1


def test_no_rays_give_empty_arrays():
    pytest.importorskip("embreex")
    triangle, point = RayCaster(_box()).first_hits(np.zeros((0, 3)), np.zeros((0, 3)))
    assert triangle.shape == (0,) and point.shape == (0, 3)


def test_the_normals_are_outward_unit_vectors():
    pytest.importorskip("embreex")
    caster = RayCaster(_box())
    assert caster.normals.shape == (30, 3)
    assert np.allclose(np.linalg.norm(caster.normals, axis=1), 1.0)


def test_the_module_imports_without_trimesh(monkeypatch):
    monkeypatch.setitem(sys.modules, "trimesh", None)
    monkeypatch.delitem(sys.modules, "compas_acoustics.lab.raycast", raising=False)
    import compas_acoustics.lab.raycast as raycast

    assert hasattr(raycast, "RayCaster")


def test_a_missing_trimesh_raises_an_import_error_with_an_install_hint(monkeypatch):
    monkeypatch.setitem(sys.modules, "trimesh", None)
    with pytest.raises(ImportError, match=r"compas_acoustics\[lab\]"):
        RayCaster(_box())


@pytest.mark.slow
def test_throughput_is_recorded():
    pytest.importorskip("embreex")
    import trimesh

    sphere = trimesh.creation.icosphere(subdivisions=5)  # 20 480 faces
    body = Mesh.from_vertices_and_faces(sphere.vertices.tolist(), sphere.faces.tolist())
    caster = RayCaster(body)
    rng = np.random.default_rng(0)
    n = 200_000
    directions = rng.normal(size=(n, 3))
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    origins = -3.0 * directions  # all outside, aimed at the centre
    start = time.perf_counter()
    triangle, _ = caster.first_hits(origins, directions)
    rate = n / (time.perf_counter() - start)
    print("\nthroughput: {:.3g} rays/s over {} faces".format(rate, len(sphere.faces)))
    assert (triangle >= 0).all()
