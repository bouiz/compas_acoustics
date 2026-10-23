import numpy as np
import pytest

from compas_acoustics.lab.grids import ReceiverGrid
from compas_acoustics.lab.grids import SourceGrid
from compas_acoustics.lab.grids import receiver_spacing


@pytest.mark.parametrize("name, n_none, n_rot", [("survey", 13, 3), ("standard", 97, 9), ("fine", 864, 36)])
def test_preset_sizes(name, n_none, n_rot):
    assert SourceGrid.preset(name).n == n_none
    assert SourceGrid.preset(name, symmetry="rotational").n == n_rot


def test_paris_weights_sum_to_one_and_follow_sin2():
    g = SourceGrid.preset("survey")
    assert g.paris_weights.sum() == pytest.approx(1.0)
    # ring bounds at 0, 15, 45, 90 degrees -> sin^2 differences, split over each ring's azimuths
    assert g.paris_weights[g.colatitude == 0].sum() == pytest.approx(np.sin(np.radians(15)) ** 2)
    assert g.paris_weights[g.colatitude == 60].sum() == pytest.approx(1 - np.sin(np.radians(45)) ** 2)


def test_paris_weights_sum_to_one_for_all_presets():
    for name in ("survey", "standard", "fine"):
        for sym in ("none", "rotational"):
            assert SourceGrid.preset(name, symmetry=sym).paris_weights.sum() == pytest.approx(1.0)


def test_directions_point_up_towards_the_source():
    g = SourceGrid(np.array([0.0, 60.0]), np.array([0.0, 90.0]))
    assert np.allclose(g.directions, [[0, 0, 1], [0, np.sin(np.radians(60)), 0.5]])


def test_unknown_preset_or_symmetry_raises_naming_valid_ones():
    with pytest.raises(ValueError, match="survey"):
        SourceGrid.preset("coarse")
    with pytest.raises(ValueError, match="rotational"):
        SourceGrid.preset("survey", symmetry="mirror")


def test_receiver_weights_cover_the_hemisphere():
    r = ReceiverGrid.hemisphere(spacing_deg=5.0, radius=10.0)
    assert (r.directions[:, 2] > 0).all()
    assert r.weights.sum() == pytest.approx(2 * np.pi, rel=1e-6)
    assert np.allclose(np.linalg.norm(r.points, axis=1), 10.0)


def test_receiver_spacing_matches_heimes_example():
    # Heimes sec. 3: d = 0.8 m, c = 346 m/s -> 5 degrees is valid up to 4.3 kHz
    assert receiver_spacing(0.8, 4300.0, 346.0) == pytest.approx(5.0, abs=0.05)
    assert receiver_spacing(0.8, 1000.0, 346.0) == 5.0  # never coarser than ISO 17497-2's 5 degrees
