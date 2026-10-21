import compas
import pytest

from compas_acoustics.specular import SpecularPath
from compas_acoustics.specular import SpecularReflections


def path(order=1, s=0, r=0):
    return SpecularPath([[0, 0, 0], [3, 4, 0], [6, 0, 0]], order=order, walls=[2] * order,
                        image=[0, 8, 0], source_index=s, receiver_index=r, sound_speed=343.98)


def test_path_length_and_travel_time():
    p = path()
    assert p.length == pytest.approx(10.0)
    assert p.travel_time == pytest.approx(10.0 / 343.98 * 1000)


def test_filter_by_pair_and_exact_order():
    field = SpecularReflections([path(1, 0, 0), path(2, 0, 0), path(1, 1, 0), path(1, 0, 1)])
    assert len(field.filter(source_index=0, receiver_index=0)) == 2
    assert [p.order for p in field.filter(order=2)] == [2]
    assert len(field.filter(order=[1, 2], source_index=0, receiver_index=0)) == 2
    assert field.orders == [1, 2]


def test_json_round_trip_keeps_every_attribute():
    back = compas.json_loads(compas.json_dumps(SpecularReflections([path(2, 1, 0)])))
    p = back.paths[0]
    assert (p.order, p.walls, p.source_index, p.receiver_index) == (2, [2, 2], 1, 0)
    assert list(p.image) == [0, 8, 0]
