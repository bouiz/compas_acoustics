import pytest

from compas_acoustics import AcousticMaterial


def test_zero_is_a_value_not_an_absence():
    m = AcousticMaterial("m", 0.0, 0.0)
    assert m.absorption_coefficients == [0.0] * 7
    assert m.scattering_coefficients == [0.0] * 7


def test_absent_scattering_is_none():
    assert AcousticMaterial("m", 0.1).scattering_coefficients is None
    assert AcousticMaterial("m", 0.1, []).scattering_coefficients is None


def test_absorption_is_required():
    with pytest.raises(ValueError):
        AcousticMaterial("m", None)


def test_material_bands_must_be_the_package_bands():
    material = AcousticMaterial("m", 0.1)
    material.frequency_bands = [125, 250, 500, 1000, 2000, 4000, 8000]
    with pytest.raises(ValueError, match="bands"):
        material.frequency_bands = [100, 200, 400, 800, 1600, 3200, 6400]
