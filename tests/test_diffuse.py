import numpy as np
import pytest

from compas_acoustics.room import diffuse


@pytest.mark.parametrize("model", ["sabine", "eyring"])
@pytest.mark.parametrize("alpha", [0.05, 0.3, 0.9])
def test_closed_form_matches_pyrato(model, alpha):
    pyrato = pytest.importorskip("pyrato")
    if not hasattr(pyrato, "parametric"):
        pytest.skip("compares with pyrato >= 1.0")
    reference = {"sabine": pyrato.parametric.reverberation_time_sabine, "eyring": pyrato.parametric.reverberation_time_eyring}[model]
    kwargs = dict(volume=200.0, surface_area=220.0, mean_absorption=np.full(8, alpha), speed_of_sound=343.0)
    np.testing.assert_allclose(getattr(diffuse, f"_{model}")(**kwargs), reference(**kwargs), rtol=1e-10)


def test_a_room_without_absorption_never_decays():
    for model in ("sabine", "eyring"):
        rt = getattr(diffuse, f"_{model}")(volume=200.0, surface_area=220.0, mean_absorption=np.zeros(8), speed_of_sound=343.0)
        assert np.all(np.isinf(rt))


def test_eyring_of_a_fully_absorbing_room_is_finite_and_milliseconds():
    """Eyring goes to zero only logarithmically: 24 ln(10) V / (c S ln(1e12)) = 5.3 ms here."""
    rt = diffuse._eyring(volume=200.0, surface_area=220.0, mean_absorption=np.full(8, 1.0 - 1e-12), speed_of_sound=343.0)
    assert np.all(np.isfinite(rt))
    np.testing.assert_allclose(rt, 24 * np.log(10) * 200.0 / (343.0 * 220.0 * np.log(1e12)), rtol=1e-6)
