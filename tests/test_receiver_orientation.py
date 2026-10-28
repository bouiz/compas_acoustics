import numpy as np
import pytest

from compas_acoustics import Receiver


def test_default_orientation_is_room_aligned():
    np.testing.assert_allclose(Receiver("r", 1, 2, 3).rotation, np.eye(3), atol=1e-15)


def test_rotation_is_proper():
    rot = Receiver("r", 0, 0, 0, direction=[1, 2, 0.5], up=[0.1, 0, 1]).rotation
    np.testing.assert_allclose(rot.T @ rot, np.eye(3), atol=1e-12)
    assert np.linalg.det(rot) == pytest.approx(1.0)


def test_facing_plus_y_maps_world_y_ahead_and_world_minus_x_left():
    rot = Receiver("r", 0, 0, 0, direction=[0, 1, 0]).rotation
    np.testing.assert_allclose(rot.T @ [0, 1, 0], [1, 0, 0], atol=1e-12)
    np.testing.assert_allclose(rot.T @ [-1, 0, 0], [0, 1, 0], atol=1e-12)


def test_up_is_made_orthogonal_to_the_facing_direction():
    rot = Receiver("r", 0, 0, 0, direction=[1, 0, 0], up=[1, 0, 1]).rotation
    np.testing.assert_allclose(rot[:, 2], [0, 0, 1], atol=1e-12)


def test_direction_parallel_to_up_raises():
    receiver = Receiver("r", 0, 0, 0, direction=[0, 0, 1])
    with pytest.raises(ValueError, match="parallel"):
        receiver.rotation


def test_up_must_be_a_vector():
    with pytest.raises(ValueError):
        Receiver("r", 0, 0, 0, up="z")


def test_the_orientation_survives_json_and_copy():
    import compas

    receiver = Receiver("r", 1, 2, 3, direction=[0, 1, 0], up=[0, 0.1, 1], radius=0.3)
    for other in (compas.json_loads(compas.json_dumps(receiver)), receiver.copy()):
        assert isinstance(other, Receiver)
        assert (other.name, other.position, other.radius) == ("r", [1.0, 2.0, 3.0], 0.3)
        np.testing.assert_allclose(other.rotation, receiver.rotation)


def test_the_frame_is_facing_left_up():
    receiver = Receiver("r", 1, 2, 3, direction=[0, 1, 0])
    frame = receiver.frame
    np.testing.assert_allclose(frame.point, [1, 2, 3])
    np.testing.assert_allclose([frame.xaxis, frame.yaxis, frame.zaxis], [[0, 1, 0], [-1, 0, 0], [0, 0, 1]], atol=1e-12)


def test_a_receiver_from_a_frame_has_its_orientation():
    from compas.geometry import Frame

    frame = Frame([1, 2, 3], [0, 1, 0], [-1, 0, 0])
    receiver = Receiver.from_frame("r", frame)
    assert receiver.position == [1.0, 2.0, 3.0]
    np.testing.assert_allclose(receiver.rotation, np.column_stack([frame.xaxis, frame.yaxis, frame.zaxis]), atol=1e-12)
    receiver.frame = Frame([0, 0, 1], [1, 0, 0], [0, 1, 0])
    np.testing.assert_allclose(receiver.rotation, np.eye(3), atol=1e-12)


def test_transforming_a_receiver_turns_it_too():
    from compas.geometry import Rotation
    from compas.geometry import Translation

    receiver = Receiver("r", 1, 0, 0)
    receiver.transform(Translation.from_vector([0, 0, 2]) * Rotation.from_axis_and_angle([0, 0, 1], np.pi / 2))
    np.testing.assert_allclose(receiver.position, [0, 1, 2], atol=1e-12)
    np.testing.assert_allclose(receiver.rotation[:, 0], [0, 1, 0], atol=1e-12)
    np.testing.assert_allclose(list(receiver.up), [0, 0, 1], atol=1e-12)


def test_a_zero_direction_raises_at_once():
    with pytest.raises(ValueError, match="non-zero"):
        Receiver("r", 0, 0, 0, direction=[0, 0, 0])
    with pytest.raises(ValueError, match="non-zero"):
        Receiver("r", 0, 0, 0).up = [0, 0, 0]


def test_a_source_survives_json_with_its_direction_and_directivity():
    import compas

    from compas_acoustics import Source

    source = Source("s", 1, 2, 3, directivity="cardioid", direction=[1, 0, 0])
    source.spl = [70.0] * 7
    loaded = compas.json_loads(compas.json_dumps(source))
    assert (loaded.name, loaded.position, loaded.directivity, loaded.spl) == ("s", [1.0, 2.0, 3.0], "cardioid", [70.0] * 7)
    assert list(loaded.direction) == [1.0, 0.0, 0.0]


def test_receiver_name_error_names_the_receiver():
    with pytest.raises(ValueError, match="Receiver name cannot contain spaces"):
        Receiver("A b", 0, 0, 0)
    with pytest.raises(ValueError, match="Receiver name must be lowercase"):
        Receiver("A", 0, 0, 0)
