import math

import pytest
from compas.geometry import Vector

from compas_acoustics import AirConditions
from compas_acoustics import Ray
from compas_acoustics import RayField
from compas_acoustics import ScatteredRay
from compas_acoustics.defaults import Defaults

C = Defaults().DEFAULT_SPEED_OF_SOUND


def ms(distance):
    """The time sound takes over ``distance`` metres, in ms."""
    return distance / C * 1000.0


def make_ray(**kwargs):
    # source -> hit1 -> hit2 -> hit3, segment lengths 3, 4, 3
    points = [[0, 0, 0], [3, 0, 0], [3, 4, 0], [3, 4, 3]]
    return Ray(points, **kwargs)


def test_segment_and_cumulative_distances():
    ray = make_ray()
    assert ray.segment_lengths == [3.0, 4.0, 3.0]
    assert ray.cumulative_distances == [0.0, 3.0, 7.0, 10.0]


def test_total_distance_and_travel_time():
    ray = make_ray()
    assert ray.total_distance == 10.0
    assert math.isclose(ray.travel_time, 10.0 / C * 1000.0)


def test_reflection_order():
    ray = make_ray()
    assert ray.reflection_order == 3


def test_direction_defaults_to_first_segment_and_is_unit():
    ray = make_ray()
    assert math.isclose(ray.direction.length, 1.0)
    assert list(ray.direction) == [1.0, 0.0, 0.0]


def test_distance_offset_included_in_totals():
    points = [[0, 0, 0], [0, 0, 2]]
    ray = Ray(points, distance_offset=3.0)
    assert ray.cumulative_distances == [3.0, 5.0]
    assert ray.total_distance == 5.0


def test_up_to_reflection_order_truncates_points():
    ray = make_ray()
    truncated = ray.up_to_reflection_order(2)
    assert [list(p) for p in truncated.points] == [[0, 0, 0], [3, 0, 0], [3, 4, 0]]


def test_up_to_distance_interpolates_final_segment():
    ray = make_ray()
    # distance 5 falls halfway along the second segment (3..7)
    truncated = ray.up_to_distance(5.0)
    assert [list(p) for p in truncated.points] == [[0, 0, 0], [3, 0, 0], [3, 2, 0]]


def test_up_to_distance_before_emergence_returns_none():
    points = [[0, 0, 0], [0, 0, 2]]
    ray = Ray(points, distance_offset=3.0)
    assert ray.up_to_distance(2.0) is None


def test_up_to_time_converts_time_to_distance():
    ray = make_ray()
    time_ms = ms(5.0)
    truncated = ray.up_to_time(time_ms)
    assert [list(p) for p in truncated.points] == [[0, 0, 0], [3, 0, 0], [3, 2, 0]]


def test_is_in_cone():
    ray = make_ray()
    assert ray.is_in_cone(Vector(1, 0, 0), 10) is True
    assert ray.is_in_cone(Vector(0, 1, 0), 10) is False
    # 90 degree half angle includes a perpendicular direction
    assert ray.is_in_cone(Vector(0, 1, 0), 90) is True


def test_scattered_ray_reflection_order_is_origin_order():
    scattered = ScatteredRay(
        [[3, 0, 0], [3, 0, 2]],
        parent_ray_index=7,
        energy=0.5,
        origin_reflection_order=2,
        distance_offset=3.0,
    )
    assert scattered.reflection_order == 2
    assert scattered.total_distance == 5.0


def test_rayfield_filter_by_source_index():
    ray_a = make_ray(source_index=0, ray_index=0)
    ray_b = make_ray(source_index=1, ray_index=0)
    field = RayField([ray_a, ray_b])
    assert len(field.filter(source=0)) == 1
    assert field.filter(source=0).rays[0] is ray_a


def test_rayfield_filter_by_ray_index():
    ray_a = make_ray(source_index=0, ray_index=3)
    ray_b = make_ray(source_index=0, ray_index=9)
    field = RayField([ray_a, ray_b])
    assert len(field.filter(ray_index=3)) == 1
    assert len(field.filter(ray_index=[3, 9])) == 2


def test_rayfield_truncated_reflection_order_carries_scattered():
    ray = make_ray(source_index=0, ray_index=0)
    scattered = ScatteredRay(
        [[3, 0, 0], [3, 0, 2]],
        parent_ray_index=0,
        origin_reflection_order=1,
        distance_offset=3.0,
    )
    ray.scattered_reflections = [scattered]
    field = RayField([ray])

    # budget = 2 - 1 = 1 -> scattered ray kept
    kept = field.truncated(reflection_order=2)
    assert len(kept.scattered_rays) == 1

    # budget = 1 - 1 = 0 -> scattered ray dropped
    dropped = field.truncated(reflection_order=1)
    assert len(dropped.scattered_rays) == 0


def test_particles_sit_at_c_t_along_each_ray():
    ray = Ray([[0, 0, 0], [10, 0, 0], [10, 10, 0]])
    points, indices, orders = RayField([ray]).particles(ms(15.0))
    assert list(points[0]) == pytest.approx([10, 5, 0])


def test_a_ray_that_ended_has_no_particle():
    short = Ray([[0, 0, 0], [5, 0, 0]], ray_index=0)
    long = Ray([[0, 0, 0], [20, 0, 0]], ray_index=1)
    assert RayField([short, long]).particles(ms(10.0))[1] == [1]


def test_duration_is_the_longest_ray():
    rays = [Ray([[0, 0, 0], [d, 0, 0]]) for d in (5, 20)]
    assert RayField(rays).duration_ms == pytest.approx(ms(20.0))


def test_particles_clamp_time():
    field = RayField([Ray([[0, 0, 0], [20, 0, 0]])])
    assert list(field.particles(-5.0)[0][0]) == pytest.approx([0, 0, 0])
    assert list(field.particles(ms(99.0))[0][0]) == pytest.approx([20, 0, 0])
    assert RayField([]).particles(1.0) == ([], [], [])


def test_particles_start_once_the_ray_has_emerged_from_its_offset():
    # 5 m before the first point, then one 10 m segment
    field = RayField([Ray([[0, 0, 0], [10, 0, 0]], distance_offset=5.0)])
    assert field.particles(ms(3.0)) == ([], [], [])
    assert list(field.particles(ms(5.0))[0][0]) == pytest.approx([0, 0, 0])
    assert list(field.particles(ms(10.0))[0][0]) == pytest.approx([5, 0, 0])


def test_times_are_at_the_speed_of_sound_of_the_default_air():
    assert C == AirConditions().speed_of_sound == pytest.approx(343.98, abs=0.01)
    assert make_ray().travel_time == pytest.approx(10.0 / AirConditions().speed_of_sound * 1000.0)


def test_a_ray_field_keeps_no_speed_of_sound_and_loads_one_stored_with_it():
    import compas

    field = RayField([make_ray(ray_index=2)])
    data = compas.json_loads(compas.json_dumps(field)).__data__
    assert "sound_speed" not in data and "sound_speed" not in data["rays"][0].__data__
    stored = compas.json_dumps(field).replace('"distance_offset"', '"sound_speed": 343.0, "distance_offset"')
    stored = stored.replace('"rays":', '"sound_speed": 343.0, "rays":')
    assert stored.count("sound_speed") == 2
    back = compas.json_loads(stored)
    assert back.rays[0].ray_index == 2 and back.rays[0].travel_time == pytest.approx(ms(10.0))


def _two_source_scene(first_outside=False):
    from compas_acoustics import AcousticMaterial
    from compas_acoustics import AcousticObjectsLibrary
    from compas_acoustics import Scene
    from compas_acoustics import Source

    scene = Scene("rays")
    scene.add_acoustic_objects(AcousticObjectsLibrary.shoebox_room(7, 5, 3, materials=AcousticMaterial("wall", 0.2, 0.1)))
    scene.add_source(Source("a", 50, 50, 50) if first_outside else Source("a", 1.5, 1.5, 1.6))
    scene.add_source(Source("b", 5, 3, 1.5))
    return scene


def test_rays_keep_the_scene_index_of_their_source():
    scene = _two_source_scene()
    field = RayField.from_scene(scene, n_rays=5, sources=[scene.sources[1]])
    assert {ray.source_index for ray in field} == {1}
    assert all(ray.source is scene.sources[1] for ray in field)


def test_a_source_outside_the_room_does_not_shift_the_others_index():
    with pytest.warns(UserWarning, match="outside"):
        field = RayField.from_scene(_two_source_scene(first_outside=True), n_rays=5)
    assert {ray.source_index for ray in field} == {1}


def _directive_scene(directivity, direction=(1, 0, 0), second=None):
    from compas_acoustics import AcousticMaterial
    from compas_acoustics import AcousticObjectsLibrary
    from compas_acoustics import Scene
    from compas_acoustics import Source

    scene = Scene("rays")
    scene.add_acoustic_objects(AcousticObjectsLibrary.shoebox_room(7, 5, 3, materials=AcousticMaterial("wall", 0.2, 0.1)))
    scene.add_source(Source("a", 3.5, 2.5, 1.5, directivity=directivity, direction=Vector(*direction)))
    if second is not None:
        scene.add_source(Source("b", 2, 2, 1.5, directivity=directivity, direction=Vector(*second)))
    return scene


def _cosines(field, axis, source_index=0):
    axis = [a / math.sqrt(sum(c * c for c in axis)) for a in axis]
    return [sum(d * a for d, a in zip(ray.direction, axis)) for ray in field if ray.source_index == source_index]


def _pattern_cdf(u, p):
    # the share of the energy |p + (1 - p) cos|^2 radiated within cos < u
    q = 2 * p - 1
    return ((p + (1 - p) * u) ** 3 - q**3) / (1 - q**3)


def test_an_omnidirectional_source_emits_the_uniform_lattice():
    import numpy as np
    import pyroomacoustics as pra

    field = RayField.from_scene(_directive_scene("omnidirectional"), n_rays=50, max_reflection_order=1)
    expected = np.asarray(pra.fibonacci_spherical_sampling(50)).T
    assert np.allclose([list(ray.direction) for ray in field], expected / np.linalg.norm(expected, axis=1)[:, None])


@pytest.mark.parametrize("directivity, p", [("cardioid", 0.5), ("supercardioid", 0.37), ("hypercardioid", 0.25), ("bidirectional", 0.0)])
def test_rays_leave_a_directive_source_with_the_density_of_its_radiated_energy(directivity, p):
    n = 2000
    cosines = _cosines(RayField.from_scene(_directive_scene(directivity), n_rays=n, max_reflection_order=1), (1, 0, 0))
    assert len(cosines) == n
    for u in (-0.5, 0.0, 0.5):
        assert sum(c < u for c in cosines) / n == pytest.approx(_pattern_cdf(u, p), abs=2.0 / n)


def test_a_cardioid_sends_almost_nothing_straight_back_and_a_figure_of_eight_nothing_sideways():
    n = 1000
    cardioid = _cosines(RayField.from_scene(_directive_scene("cardioid"), n_rays=n, max_reflection_order=1), (1, 0, 0))
    assert sum(c < -0.8 for c in cardioid) / n < 0.002
    eight = _cosines(RayField.from_scene(_directive_scene("bidirectional"), n_rays=n, max_reflection_order=1), (1, 0, 0))
    assert sum(abs(c) < 0.2 for c in eight) / n < 0.01


def test_each_source_radiates_along_its_own_facing():
    field = RayField.from_scene(_directive_scene("cardioid", (1, 0, 0), second=(0, -1, 0)), n_rays=400, max_reflection_order=1)
    assert sum(_cosines(field, (1, 0, 0), 0)) / 400 == pytest.approx(0.5, abs=0.01)
    assert sum(_cosines(field, (0, -1, 0), 1)) / 400 == pytest.approx(0.5, abs=0.01)


def test_a_cone_keeps_its_rays_and_weights_them_by_the_pattern():
    import numpy as np

    n = 600
    omni = RayField.from_scene(_directive_scene("omnidirectional"), n_rays=n, max_reflection_order=1, direction=[0, 1, 0], angle=60)
    cardioid = RayField.from_scene(_directive_scene("cardioid"), n_rays=n, max_reflection_order=1, direction=[0, 1, 0], angle=60)
    assert len(cardioid.rays) == n
    # every ray inside the cone, distinct, and leaning towards the source's facing (+x) where omni is even
    assert min(_cosines(cardioid, (0, 1, 0))) >= math.cos(math.radians(60)) - 1e-9
    assert len({tuple(np.round(list(ray.direction), 9)) for ray in cardioid}) == n
    assert abs(np.mean(_cosines(omni, (1, 0, 0)))) < 0.01
    assert np.mean(_cosines(cardioid, (1, 0, 0))) > 0.1


def test_a_ray_field_traces_each_ray_for_its_duration():
    import inspect

    field = RayField.from_scene(_directive_scene("omnidirectional"), n_rays=20, max_reflection_order=50, duration=0.02)
    limit = C * 0.02
    # a ray stops at the first reflection past its duration's distance
    assert all(sum(ray.segment_lengths[:-1]) <= limit for ray in field)
    assert any(ray.total_distance > 0.5 * limit for ray in field)
    assert "duration" in inspect.signature(RayField.from_scene).parameters
    assert "time_thres" not in inspect.signature(RayField.from_scene).parameters


def test_particles_carry_the_reflection_order_they_have_reached():
    ray = Ray([[0, 0, 0], [10, 0, 0], [10, 10, 0], [0, 10, 0]], ray_index=0)
    field = RayField([ray])
    # A particle on a wall has not been reflected yet; the end of the last segment is not a reflection
    times = [ms(d) for d in (0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0)]
    assert [field.particles(t)[2] for t in times] == [[0], [0], [0], [1], [1], [2], [2]]
