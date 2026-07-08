import math

from compas.geometry import Vector

from compas_acoustics import Ray
from compas_acoustics import RayField
from compas_acoustics import ScatteredRay


def make_ray(**kwargs):
    # source -> hit1 -> hit2 -> hit3, segment lengths 3, 4, 3
    points = [[0, 0, 0], [3, 0, 0], [3, 4, 0], [3, 4, 3]]
    return Ray(points, sound_speed=343.0, **kwargs)


def test_segment_and_cumulative_distances():
    ray = make_ray()
    assert ray.segment_lengths == [3.0, 4.0, 3.0]
    assert ray.cumulative_distances == [0.0, 3.0, 7.0, 10.0]


def test_total_distance_and_travel_time():
    ray = make_ray()
    assert ray.total_distance == 10.0
    assert math.isclose(ray.travel_time, 10.0 / 343.0 * 1000.0)


def test_reflection_order():
    ray = make_ray()
    assert ray.reflection_order == 3


def test_direction_defaults_to_first_segment_and_is_unit():
    ray = make_ray()
    assert math.isclose(ray.direction.length, 1.0)
    assert list(ray.direction) == [1.0, 0.0, 0.0]


def test_distance_offset_included_in_totals():
    points = [[0, 0, 0], [0, 0, 2]]
    ray = Ray(points, sound_speed=343.0, distance_offset=3.0)
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
    time_ms = 5.0 / 343.0 * 1000.0
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
